#ifndef ESWIN_TRANSFORMER_SESSION_HPP
#define ESWIN_TRANSFORMER_SESSION_HPP

#include "../model/model_manifest.hpp"
#include "../backend/worker_pool.hpp"
#include "../backend/dma_buffer.hpp"
#include "../nlp/tokenizer.hpp"
#include "../nlp/sampler.hpp"
#include "kv_cache.hpp"

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace eswin {

// -----------------------------------------------------------------------------
// Portable Half-Precision Types
// -----------------------------------------------------------------------------
#if defined(__riscv) || defined(__riscv__) || defined(__FLT16_MANT_DIG__)
typedef _Float16 es_fp16_t;
static inline es_fp16_t f32_to_f16(float f) { return static_cast<es_fp16_t>(f); }
static inline float f16_to_f32(es_fp16_t h) { return static_cast<float>(h); }
#else
typedef uint16_t es_fp16_t;
static inline es_fp16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3ff;
    if (exp <= 0) return static_cast<es_fp16_t>(sign);
    if (exp >= 31) return static_cast<es_fp16_t>(sign | 0x7c00);
    return static_cast<es_fp16_t>(sign | (exp << 10) | mant);
}
static inline float f16_to_f32(es_fp16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = (h & 0x3ff) << 13;
    if (exp == 0) return 0.0f;
    exp = exp - 15 + 127;
    uint32_t x = sign | (exp << 23) | mant;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}
#endif

struct NpuLayerSubgraphs {
    uint16_t die_id = 0;
    uint32_t model_qkv = 0;
    uint32_t model_attn_out = 0;
    uint32_t model_mlp = 0;
    bool has_qkv = false;
    bool has_attn_out = false;
    bool has_mlp = false;
};

struct SessionStats {
    double prompt_eval_ms = 0.0;
    double eval_ms = 0.0;
    int    prompt_tokens = 0;
    int    eval_tokens = 0;
    double tokens_per_sec = 0.0;
};

class TransformerSession {
public:
    using TokenCallback = std::function<bool(const std::string &piece, int32_t token_id)>;

    TransformerSession();
    ~TransformerSession();

    // Initialize session with model manifest and tokenizer
    bool init(const ModelManifest &manifest,
              const std::string &model_dir,
              const std::string &tokenizer_path,
              std::string &err);

    // Set sampler parameters
    void set_sampler_params(const SamplerParams &params);

    // Reset sequence context and KV-cache
    void reset();

    // Autoregressively generate tokens from prompt
    bool generate(
        const std::string &prompt,
        int max_new_tokens,
        TokenCallback token_cb,
        SessionStats *stats = nullptr
    );

    // Forward single token and produce next token ID
    int32_t step(int32_t token_id, int pos);

    // Accessors
    const ModelManifest& manifest() const { return m_manifest; }
    Tokenizer& tokenizer() { return m_tokenizer; }
    Sampler& sampler() { return m_sampler; }
    KvCache* kv_cache() { return m_kv_cache.get(); }
    const SessionStats& last_stats() const { return m_stats; }

private:
    void forward_lm_head_dual_die(const float *hidden_state, float *out_logits);
    bool load_weights(const std::string &bin_path);
    bool load_rms_norms(const std::string &bin_path);
    bool load_embeddings_fp16(const std::string &bin_path);
    void discover_and_load_layer_subgraphs(const std::string &model_dir);
    void rmsnorm(float *out, const float *x, const float *weight, int size, float eps = 1e-6f);
    void apply_rope(float *q, float *k, int pos);

    ModelManifest                  m_manifest;
    Tokenizer                      m_tokenizer;
    Sampler                        m_sampler;
    std::unique_ptr<KvCache>       m_kv_cache;
    std::unique_ptr<WorkerPool>    m_worker_pool;

    // Weights buffers for local execution
    bool                           m_has_weights = false;
    bool                           m_has_norms = false;
    bool                           m_has_embed_fp16 = false;
    std::vector<es_fp16_t>         m_embed_fp16;    // [vocab_size, hidden_size] in FP16
    std::vector<float>             m_embed;         // [vocab_size, hidden_size]
    std::vector<float>             m_rms_att;       // [n_layers, hidden_size]
    std::vector<float>             m_wq;            // [n_layers, hidden_size, hidden_size]
    std::vector<float>             m_wk;            // [n_layers, hidden_size, kv_dim]
    std::vector<float>             m_wv;            // [n_layers, hidden_size, kv_dim]
    std::vector<float>             m_wo;            // [n_layers, hidden_size, hidden_size]
    std::vector<float>             m_rms_ffn;       // [n_layers, hidden_size]
    std::vector<float>             m_w_gate;        // [n_layers, hidden_size, intermediate_size]
    std::vector<float>             m_w_down;        // [n_layers, intermediate_size, hidden_size]
    std::vector<float>             m_w_up;          // [n_layers, hidden_size, intermediate_size]
    std::vector<float>             m_rms_final;     // [hidden_size]
    std::vector<float>             m_lm_head_weights;// [vocab_size, hidden_size] (host fallback)

    // Working scratch buffers
    std::vector<float>             m_hidden_state;
    std::vector<float>             m_logits;
    std::vector<float>             m_x_buf;
    std::vector<float>             m_x_norm_buf;
    std::vector<float>             m_q_buf;
    std::vector<float>             m_k_buf;
    std::vector<float>             m_v_buf;
    std::vector<float>             m_att_out_buf;
    std::vector<float>             m_mlp_act_buf;
    std::vector<float>             m_scores_buf;

    // Fast contiguous KV cache for transformer layers
    std::vector<float>             m_k_cache_flat;
    std::vector<float>             m_v_cache_flat;
    int                            m_max_seq = 2048;

    // Dual-die NPU buffers (LM Head offload)
    bool                           m_npu_dual_die = false;
    uint32_t                       m_model_d0 = 0;
    uint32_t                       m_model_d1 = 0;
    std::unique_ptr<DmaBuffer>     m_d0_in;
    std::unique_ptr<DmaBuffer>     m_d0_out;
    std::unique_ptr<DmaBuffer>     m_d1_in;
    std::unique_ptr<DmaBuffer>     m_d1_out;

    // Decomposed layer subgraphs & per-die GEMM scratch buffers
    std::vector<NpuLayerSubgraphs> m_layer_subgraphs;
    bool                           m_has_any_npu_layers = false;
    std::unique_ptr<DmaBuffer>     m_d0_gemm_in;
    std::unique_ptr<DmaBuffer>     m_d0_gemm_out;
    std::unique_ptr<DmaBuffer>     m_d1_gemm_in;
    std::unique_ptr<DmaBuffer>     m_d1_gemm_out;

    SessionStats                   m_stats;
    int                            m_current_pos = 0;
};

} // namespace eswin

#endif // ESWIN_TRANSFORMER_SESSION_HPP
