#include "sampler.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <limits>

namespace eswin {

Sampler::Sampler(const SamplerParams &params)
    : m_params(params)
{
    if (m_params.seed == 0) {
        std::random_device rd;
        m_rng.seed(rd());
    } else {
        m_rng.seed(m_params.seed);
    }
}

void Sampler::accept(int32_t token_id) {
    m_history.push_back(token_id);
}

void Sampler::reset() {
    m_history.clear();
}

void Sampler::apply_repetition_penalty(float *logits, size_t vocab_size) {
    if (m_params.repeat_penalty == 1.0f || m_history.empty()) return;

    int start_idx = std::max(0, static_cast<int>(m_history.size()) - m_params.repeat_window);
    std::unordered_map<int32_t, int> seen;

    for (size_t i = start_idx; i < m_history.size(); ++i) {
        int32_t id = m_history[i];
        if (id >= 0 && static_cast<size_t>(id) < vocab_size) {
            seen[id]++;
        }
    }

    for (const auto &pair : seen) {
        int32_t id = pair.first;
        if (logits[id] > 0.0f) {
            logits[id] /= m_params.repeat_penalty;
        } else {
            logits[id] *= m_params.repeat_penalty;
        }
    }
}

void Sampler::apply_loop_breaking(float *logits, size_t vocab_size) {
    if (m_params.max_consecutive_duplicates <= 0 || m_history.empty()) return;

    // Check for consecutive duplicates (e.g. "the the")
    int count = 1;
    int32_t last_token = m_history.back();
    for (int i = static_cast<int>(m_history.size()) - 2; i >= 0; --i) {
        if (m_history[i] == last_token) count++;
        else break;
    }

    if (count >= m_params.max_consecutive_duplicates) {
        if (last_token >= 0 && static_cast<size_t>(last_token) < vocab_size) {
            logits[last_token] = -std::numeric_limits<float>::infinity();
        }
    }

    // Cyclic 2-gram / 3-gram loop detector (Issue #47 fix)
    size_t sz = m_history.size();
    if (sz >= 6) {
        // Check for ABABAB
        if (m_history[sz - 1] == m_history[sz - 3] &&
            m_history[sz - 2] == m_history[sz - 4] &&
            m_history[sz - 3] == m_history[sz - 5]) {
            int32_t loop_token = m_history[sz - 2];
            if (loop_token >= 0 && static_cast<size_t>(loop_token) < vocab_size) {
                logits[loop_token] = -std::numeric_limits<float>::infinity();
            }
        }
    }
}

int32_t Sampler::sample(float *logits, size_t vocab_size) {
    if (!logits || vocab_size == 0) return 0;

    // 1. Repetition penalty
    apply_repetition_penalty(logits, vocab_size);

    // 2. Loop breaking
    apply_loop_breaking(logits, vocab_size);

    // 3. Greedy sampling if temperature <= 0
    if (m_params.temperature <= 0.0f) {
        int32_t best_id = 0;
        float max_l = logits[0];
        for (size_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_l) {
                max_l = logits[i];
                best_id = static_cast<int32_t>(i);
            }
        }
        return best_id;
    }

    // 4. Temperature scaling
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] /= m_params.temperature;
    }

    // 5. Gather candidates and sort
    std::vector<std::pair<float, int32_t>> candidates(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        candidates[i] = {logits[i], static_cast<int32_t>(i)};
    }

    size_t k = (m_params.top_k > 0 && static_cast<size_t>(m_params.top_k) < vocab_size)
               ? static_cast<size_t>(m_params.top_k) : vocab_size;

    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
    candidates.resize(k);

    // 6. Softmax over Top-K
    float max_logit = candidates[0].first;
    float sum_exp = 0.0f;
    for (auto &c : candidates) {
        c.first = std::exp(c.first - max_logit);
        sum_exp += c.first;
    }
    for (auto &c : candidates) {
        c.first /= sum_exp;
    }

    // 7. Min-P filtering
    if (m_params.min_p > 0.0f) {
        float max_prob = candidates[0].first;
        float min_thresh = max_prob * m_params.min_p;
        size_t cutoff = 1;
        while (cutoff < candidates.size() && candidates[cutoff].first >= min_thresh) {
            cutoff++;
        }
        candidates.resize(cutoff);
    }

    // 8. Top-P (nucleus) filtering
    if (m_params.top_p < 1.0f) {
        float cum = 0.0f;
        size_t cutoff = candidates.size();
        for (size_t i = 0; i < candidates.size(); ++i) {
            cum += candidates[i].first;
            if (cum >= m_params.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates.resize(cutoff);
    }

    // 9. Renormalize remaining probabilities
    float total_p = 0.0f;
    for (const auto &c : candidates) total_p += c.first;

    if (total_p <= 0.0f) {
        return candidates[0].second;
    }

    // 10. Sample
    std::uniform_real_distribution<float> dist(0.0f, total_p);
    float r = dist(m_rng);
    float acc = 0.0f;
    for (const auto &c : candidates) {
        acc += c.first;
        if (r <= acc) {
            return c.second;
        }
    }

    return candidates.back().second;
}

} // namespace eswin
