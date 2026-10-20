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
    void rmsnorm(float *out, const float *x, const float *weight, int size, float eps = 1e-5f);
    void apply_rope(float *q, float *k, int pos);

    ModelManifest                  m_manifest;
    Tokenizer                      m_tokenizer;
    Sampler                        m_sampler;
    std::unique_ptr<KvCache>       m_kv_cache;
    std::unique_ptr<WorkerPool>    m_worker_pool;

    // Weights buffers for local execution
    bool                           m_has_weights = false;
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

    // Fast contiguous KV cache for transformer layers
    std::vector<float>             m_k_cache_flat;
    std::vector<float>             m_v_cache_flat;
    int                            m_max_seq = 2048;

    // Dual-die NPU buffers (if offloaded)
    bool                           m_npu_dual_die = false;
    uint32_t                       m_model_d0 = 0;
    uint32_t                       m_model_d1 = 0;
    std::unique_ptr<DmaBuffer>     m_d0_in;
    std::unique_ptr<DmaBuffer>     m_d0_out;
    std::unique_ptr<DmaBuffer>     m_d1_in;
    std::unique_ptr<DmaBuffer>     m_d1_out;

    SessionStats                   m_stats;
    int                            m_current_pos = 0;
};

} // namespace eswin

#endif // ESWIN_TRANSFORMER_SESSION_HPP
