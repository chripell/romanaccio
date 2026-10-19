#ifndef ESWIN_SAMPLER_HPP
#define ESWIN_SAMPLER_HPP

#include <vector>
#include <cstdint>
#include <random>

namespace eswin {

struct SamplerParams {
    float   temperature = 0.7f;     // <= 0.0f enables greedy decoding
    int32_t top_k = 40;             // 0 disables top-k
    float   top_p = 0.9f;           // 1.0f disables nucleus sampling
    float   min_p = 0.05f;          // 0.0f disables min-p
    float   repeat_penalty = 1.15f; // 1.0f disables repetition penalty
    int32_t repeat_window = 64;     // tokens to look back for repetition
    int32_t max_consecutive_duplicates = 2; // Prevents "the the the" loops
    uint32_t seed = 0;              // 0 initializes random seed from system
};

class Sampler {
public:
    explicit Sampler(const SamplerParams &params = SamplerParams());

    // Sample next token ID from logits array
    int32_t sample(float *logits, size_t vocab_size);

    // Record sampled token to update internal history (repetition window)
    void accept(int32_t token_id);

    // Reset history
    void reset();

    // Accessors
    const SamplerParams& params() const { return m_params; }
    void set_params(const SamplerParams &params) { m_params = params; }
    const std::vector<int32_t>& history() const { return m_history; }

private:
    void apply_repetition_penalty(float *logits, size_t vocab_size);
    void apply_loop_breaking(float *logits, size_t vocab_size);

    SamplerParams        m_params;
    std::vector<int32_t> m_history;
    std::mt19937         m_rng;
};

} // namespace eswin

#endif // ESWIN_SAMPLER_HPP
