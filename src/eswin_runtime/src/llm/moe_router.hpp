/**
 * @file moe_router.hpp
 * @brief Host-side Mixture-of-Experts gate, top-k selection and per-die dispatch grouping.
 *
 * DeepSeek-V2-Lite Phase 2. The router is the only part of an MoE decode step that cannot
 * be compiled into a loadable, because its output *selects which weights get read*. It runs
 * on the host CPU once per layer per token, and it has to be fast enough not to matter:
 * 26 MoE layers at the 60 us target is 1.6 ms/token, roughly 1% of the decode budget.
 *
 * Semantics are taken from the verified `deepseek-ai/DeepSeek-V2-Lite/config.json`, not
 * from assumption (see src/DeepSeek-V2-Lite/PLAN.md section 6.9):
 *
 *   scoring_func        softmax      -> scores are a softmax over all 64 experts
 *   norm_topk_prob      false        -> the selected top-6 weights are used UNNORMALISED,
 *                                       so they sum to less than 1. Renormalising them
 *                                       would silently rescale every expert's contribution
 *                                       and is the single easiest way to get plausible-looking
 *                                       but wrong output from this model.
 *   topk_method         greedy       -> plain top-k by score
 *   n_group / topk_group 1 / 1       -> group-limited selection is degenerate here, but is
 *                                       implemented because the V3 family uses it
 *   routed_scaling_factor 1.0        -> applied after selection
 *   attention_bias      false        -> the gate has no bias term in V2-Lite, but the field
 *                                       is supported for models that do
 *
 * The gate matrix is [num_experts][hidden_size] row-major, owned by the caller.
 */

#ifndef ES_MOE_ROUTER_HPP
#define ES_MOE_ROUTER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace eswin {

/// How raw gate logits become routing scores.
enum class MoEScoring {
    Softmax,   ///< DeepSeek-V2 / V2-Lite
    Sigmoid,   ///< DeepSeek-V3
};

/// How the top-k set is chosen.
enum class MoETopKMethod {
    Greedy,             ///< Global top-k by score (V2-Lite).
    GroupLimitedGreedy, ///< Pick topk_group groups by their best score, mask the rest, then
                        ///< take a global top-k. Degenerates to Greedy when n_group == 1.
};

struct MoERouterConfig {
    uint32_t      hidden_size          = 2048;
    uint32_t      num_experts          = 64;
    uint32_t      top_k                = 6;
    uint32_t      num_shared_experts   = 2;    ///< Always active; reported, not selected.
    MoEScoring    scoring              = MoEScoring::Softmax;
    MoETopKMethod topk_method          = MoETopKMethod::Greedy;
    bool          norm_topk_prob       = false;///< V2-Lite: false. See the header note.
    uint32_t      n_group              = 1;
    uint32_t      topk_group           = 1;
    float         routed_scaling_factor = 1.0f;
    bool          has_gate_bias        = false;
    uint16_t      num_dies             = 2;    ///< Expert-parallel split of the expert space.
};

/// One selected routed expert, with everything the dispatcher needs.
struct RoutedExpert {
    uint32_t index     = 0;    ///< Global expert id, 0..num_experts-1.
    float    score     = 0.0f; ///< Post-selection weight actually used for accumulation.
    uint16_t die_id    = 0;    ///< Die whose MMZ holds this expert's weights.
    uint32_t die_slot  = 0;    ///< Index within that die's expert range.
};

struct MoERouterResult {
    std::vector<RoutedExpert> routed;      ///< top_k entries, descending by score.
    std::vector<uint32_t>     die_experts[2]; ///< Global ids grouped by die, for dispatch.
    float    score_sum      = 0.0f;        ///< Sum of routed scores (< 1 unless normalised).
    float    max_score      = 0.0f;
    uint32_t argmax_expert  = 0;
    double   elapsed_us     = 0.0;         ///< Wall time of the last route() call.
};

class MoERouter {
public:
    MoERouter() = default;

    /**
     * @brief Validates the configuration and takes a non-owning pointer to the gate matrix.
     *
     * @param gate_weight Row-major [num_experts][hidden_size]. Must outlive the router.
     * @param gate_bias   Optional [num_experts]; ignored unless cfg.has_gate_bias.
     */
    bool init(const MoERouterConfig &cfg, const float *gate_weight,
              const float *gate_bias, std::string &err);

    bool is_ready() const noexcept { return m_ready; }
    const MoERouterConfig &config() const noexcept { return m_cfg; }

    /**
     * @brief Routes one token's hidden state.
     *
     * Fills @p out with the selected experts, their scores, and the per-die grouping.
     * Deterministic: ties are broken toward the lower expert index.
     */
    bool route(const float *hidden, MoERouterResult &out) const;

    /// Raw gate logits for @p hidden, without selection. Useful for golden tests.
    void logits(const float *hidden, float *out_logits) const;

    // --- Addressing helpers for the "weights as a dynamic input" execution model ------

    /// Die that owns a given global expert id under a contiguous expert-parallel split.
    uint16_t die_for_expert(uint32_t expert) const noexcept;

    /// Experts per die under the contiguous split.
    uint32_t experts_per_die() const noexcept;

    /**
     * @brief Byte offset of one expert's weights inside its die's routed-expert blob.
     *
     * This is the value that goes into `DevBuf.offset`, which is what makes a single
     * per-layer loadable able to run any of the 64 experts: the weights stay resident in
     * one big MMZ blob and expert selection becomes an offset, not a different model.
     * Layout is [layer][die_slot] so that all of one layer's experts for a die are
     * contiguous.
     */
    size_t expert_blob_offset(uint32_t layer, uint32_t expert,
                              size_t bytes_per_expert) const noexcept;

    /// Shared-expert byte offset within its die's shared blob ([layer][shared_index]).
    size_t shared_blob_offset(uint32_t layer, uint32_t shared_index,
                              size_t bytes_per_expert) const noexcept;

    /// Die that owns a given shared expert (shared expert i -> die i % num_dies).
    uint16_t die_for_shared_expert(uint32_t shared_index) const noexcept;

private:
    void scores_from_logits(const float *logits, float *scores) const;
    void apply_group_mask(float *scores) const;

    MoERouterConfig m_cfg;
    const float    *m_gate = nullptr;
    const float    *m_bias = nullptr;
    bool            m_ready = false;

    // Scratch, sized at init so route() does not allocate per token.
    mutable std::vector<float>    m_logits;
    mutable std::vector<float>    m_scores;
    mutable std::vector<uint32_t> m_order;
    mutable std::vector<float>    m_group_score;
};

} // namespace eswin

#endif // ES_MOE_ROUTER_HPP
