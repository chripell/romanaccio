/**
 * @file moe_router.cpp
 * @brief Host-side MoE gate, top-k selection and per-die dispatch grouping.
 *
 * See moe_router.hpp for the semantics and where each one comes from.
 *
 * Two implementation notes:
 *
 *  1. All scratch is sized in init(), so route() performs no allocation. It is called
 *     once per layer per token (26 times per token for V2-Lite), and any per-call malloc
 *     would show up directly in decode latency.
 *  2. Selection is a total order on (score descending, index ascending), so ties resolve
 *     deterministically toward the lower expert index. std::sort alone is not stable, so
 *     the index has to be part of the comparator rather than left to the sort.
 */

#include "moe_router.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace eswin {

bool MoERouter::init(const MoERouterConfig &cfg, const float *gate_weight,
                     const float *gate_bias, std::string &err) {
    m_ready = false;
    err.clear();

    auto fail = [&err](const std::string &m) { err = "MoERouter: " + m; return false; };

    if (!gate_weight) return fail("gate_weight is null");
    if (cfg.hidden_size == 0) return fail("hidden_size must be non-zero");
    if (cfg.num_experts == 0) return fail("num_experts must be non-zero");
    if (cfg.top_k == 0) return fail("top_k must be non-zero");
    if (cfg.top_k > cfg.num_experts) {
        std::ostringstream os;
        os << "top_k (" << cfg.top_k << ") exceeds num_experts (" << cfg.num_experts << ")";
        return fail(os.str());
    }
    if (cfg.has_gate_bias && !gate_bias) return fail("has_gate_bias set but gate_bias is null");
    if (cfg.num_dies == 0) return fail("num_dies must be non-zero");
    if (cfg.num_experts % cfg.num_dies != 0) {
        return fail("num_experts must divide evenly across num_dies for expert parallelism");
    }
    if (cfg.num_shared_experts % cfg.num_dies != 0) {
        return fail("num_shared_experts must divide evenly across num_dies");
    }
    if (cfg.n_group == 0) return fail("n_group must be non-zero");
    if (cfg.num_experts % cfg.n_group != 0) {
        return fail("num_experts must divide evenly into n_group groups");
    }
    if (cfg.topk_group == 0 || cfg.topk_group > cfg.n_group) {
        return fail("topk_group must be in [1, n_group]");
    }
    if (cfg.topk_method == MoETopKMethod::GroupLimitedGreedy) {
        // Otherwise masking could leave fewer than top_k selectable experts, and the
        // "selected" set would contain -inf scores.
        const uint32_t group_size = cfg.num_experts / cfg.n_group;
        if (cfg.topk_group * group_size < cfg.top_k) {
            std::ostringstream os;
            os << "topk_group (" << cfg.topk_group << ") x group_size (" << group_size
               << ") cannot supply top_k (" << cfg.top_k << ") experts";
            return fail(os.str());
        }
    }

    m_cfg = cfg;
    m_gate = gate_weight;
    m_bias = cfg.has_gate_bias ? gate_bias : nullptr;

    m_logits.assign(cfg.num_experts, 0.0f);
    m_scores.assign(cfg.num_experts, 0.0f);
    m_order.resize(cfg.num_experts);
    m_group_score.assign(cfg.n_group, 0.0f);

    m_ready = true;
    return true;
}

void MoERouter::logits(const float *hidden, float *out_logits) const {
    if (!m_ready || !hidden || !out_logits) return;

    const uint32_t n_experts = m_cfg.num_experts;
    const uint32_t dim = m_cfg.hidden_size;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (n_experts >= 16)
#endif
    for (uint32_t e = 0; e < n_experts; ++e) {
        const float *w = m_gate + static_cast<size_t>(e) * dim;
        float acc = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            acc += w[i] * hidden[i];
        }
        if (m_bias) acc += m_bias[e];
        out_logits[e] = acc;
    }
}

void MoERouter::scores_from_logits(const float *logits, float *scores) const {
    const uint32_t n = m_cfg.num_experts;

    if (m_cfg.scoring == MoEScoring::Sigmoid) {
        for (uint32_t e = 0; e < n; ++e) {
            scores[e] = 1.0f / (1.0f + std::exp(-logits[e]));
        }
        return;
    }

    // Numerically stable softmax: subtract the max before exponentiating.
    float mx = logits[0];
    for (uint32_t e = 1; e < n; ++e) {
        if (logits[e] > mx) mx = logits[e];
    }
    double sum = 0.0;
    for (uint32_t e = 0; e < n; ++e) {
        scores[e] = std::exp(logits[e] - mx);
        sum += scores[e];
    }
    if (sum > 0.0) {
        const float inv = static_cast<float>(1.0 / sum);
        for (uint32_t e = 0; e < n; ++e) scores[e] *= inv;
    }
}

void MoERouter::apply_group_mask(float *scores) const {
    const uint32_t n = m_cfg.num_experts;
    const uint32_t ng = m_cfg.n_group;
    const uint32_t group_size = n / ng;

    // Best score in each group.
    for (uint32_t g = 0; g < ng; ++g) {
        float best = -std::numeric_limits<float>::infinity();
        const float *base = scores + static_cast<size_t>(g) * group_size;
        for (uint32_t i = 0; i < group_size; ++i) {
            if (base[i] > best) best = base[i];
        }
        m_group_score[g] = best;
    }

    // Rank the groups by their best score, ties toward the lower group index.
    std::vector<uint32_t> gorder(ng);
    std::iota(gorder.begin(), gorder.end(), 0u);
    std::partial_sort(gorder.begin(),
                      gorder.begin() + std::min<uint32_t>(m_cfg.topk_group, ng),
                      gorder.end(),
                      [this](uint32_t a, uint32_t b) {
                          if (m_group_score[a] != m_group_score[b])
                              return m_group_score[a] > m_group_score[b];
                          return a < b;
                      });

    std::vector<bool> keep(ng, false);
    for (uint32_t i = 0; i < m_cfg.topk_group && i < ng; ++i) keep[gorder[i]] = true;

    // Mask out every expert in a non-selected group so it cannot be picked.
    for (uint32_t g = 0; g < ng; ++g) {
        if (keep[g]) continue;
        float *base = scores + static_cast<size_t>(g) * group_size;
        for (uint32_t i = 0; i < group_size; ++i) {
            base[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

uint16_t MoERouter::die_for_expert(uint32_t expert) const noexcept {
    const uint32_t per_die = experts_per_die();
    if (per_die == 0) return 0;
    uint16_t die = static_cast<uint16_t>(expert / per_die);
    return (die < m_cfg.num_dies) ? die : static_cast<uint16_t>(m_cfg.num_dies - 1);
}

uint32_t MoERouter::experts_per_die() const noexcept {
    return (m_cfg.num_dies > 0) ? (m_cfg.num_experts / m_cfg.num_dies) : m_cfg.num_experts;
}

uint16_t MoERouter::die_for_shared_expert(uint32_t shared_index) const noexcept {
    return (m_cfg.num_dies > 0)
               ? static_cast<uint16_t>(shared_index % m_cfg.num_dies)
               : 0;
}

size_t MoERouter::expert_blob_offset(uint32_t layer, uint32_t expert,
                                     size_t bytes_per_expert) const noexcept {
    const uint32_t per_die = experts_per_die();
    if (per_die == 0) return 0;
    const uint16_t die = die_for_expert(expert);
    const uint32_t slot = expert - static_cast<uint32_t>(die) * per_die;
    // [layer][die_slot] within this die's routed-expert blob.
    return (static_cast<size_t>(layer) * per_die + slot) * bytes_per_expert;
}

size_t MoERouter::shared_blob_offset(uint32_t layer, uint32_t shared_index,
                                     size_t bytes_per_expert) const noexcept {
    const uint32_t dies = m_cfg.num_dies ? m_cfg.num_dies : 1;
    const uint32_t per_die = m_cfg.num_shared_experts / dies;
    if (per_die == 0) return 0;
    const uint16_t die = die_for_shared_expert(shared_index);
    const uint32_t slot = shared_index - static_cast<uint32_t>(die) * per_die;
    return (static_cast<size_t>(layer) * per_die + slot) * bytes_per_expert;
}

bool MoERouter::route(const float *hidden, MoERouterResult &out) const {
    if (!m_ready || !hidden) return false;

    const auto t0 = std::chrono::steady_clock::now();

    const uint32_t n = m_cfg.num_experts;
    const uint32_t k = m_cfg.top_k;

    logits(hidden, m_logits.data());
    scores_from_logits(m_logits.data(), m_scores.data());

    // Argmax over the unmasked scores: the most-likely expert regardless of grouping.
    uint32_t argmax = 0;
    float best = m_scores[0];
    for (uint32_t e = 1; e < n; ++e) {
        if (m_scores[e] > best) { best = m_scores[e]; argmax = e; }
    }

    if (m_cfg.topk_method == MoETopKMethod::GroupLimitedGreedy && m_cfg.n_group > 1) {
        apply_group_mask(m_scores.data());
    }

    // Total order: score descending, then index ascending for deterministic ties.
    std::iota(m_order.begin(), m_order.end(), 0u);
    std::partial_sort(m_order.begin(), m_order.begin() + k, m_order.begin() + n,
                      [this](uint32_t a, uint32_t b) {
                          if (m_scores[a] != m_scores[b]) return m_scores[a] > m_scores[b];
                          return a < b;
                      });

    out.routed.clear();
    out.routed.reserve(k);
    out.die_experts[0].clear();
    out.die_experts[1].clear();

    float sum = 0.0f;
    for (uint32_t i = 0; i < k; ++i) {
        RoutedExpert re;
        re.index = m_order[i];
        re.score = m_scores[re.index];
        re.die_id = die_for_expert(re.index);
        re.die_slot = re.index - static_cast<uint32_t>(re.die_id) * experts_per_die();
        out.routed.push_back(re);
        sum += re.score;
    }

    // norm_topk_prob=false for V2-Lite: weights stay as the raw softmax probabilities and
    // sum to less than 1. Only renormalise when the model asks for it.
    if (m_cfg.norm_topk_prob && sum > 0.0f) {
        const float inv = 1.0f / sum;
        for (auto &re : out.routed) re.score *= inv;
        sum = 1.0f;
    }
    if (m_cfg.routed_scaling_factor != 1.0f) {
        for (auto &re : out.routed) re.score *= m_cfg.routed_scaling_factor;
        sum *= m_cfg.routed_scaling_factor;
    }

    for (const auto &re : out.routed) {
        if (re.die_id < 2) out.die_experts[re.die_id].push_back(re.index);
    }

    out.score_sum = sum;
    out.max_score = best;
    out.argmax_expert = argmax;

    const auto t1 = std::chrono::steady_clock::now();
    out.elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return true;
}

} // namespace eswin
