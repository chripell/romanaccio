#include "transformer_session.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>

namespace eswin {

TransformerSession::TransformerSession() = default;

TransformerSession::~TransformerSession() {
    if (m_npu_dual_die && m_worker_pool) {
        if (m_d0_in) m_worker_pool->unprepare_buffer(0, *m_d0_in);
        if (m_d0_out) m_worker_pool->unprepare_buffer(0, *m_d0_out);
        if (m_d1_in) m_worker_pool->unprepare_buffer(1, *m_d1_in);
        if (m_d1_out) m_worker_pool->unprepare_buffer(1, *m_d1_out);
    }
}

bool TransformerSession::init(
    const ModelManifest &manifest,
    const std::string &model_dir,
    const std::string &tokenizer_path,
    std::string &err
) {
    m_manifest = manifest;

    // 1. Initialize tokenizer
    std::string tok_file = tokenizer_path;
    if (tok_file.empty()) {
        const auto *tok_desc = manifest.find_file_by_role("tokenizer");
        if (tok_desc && !tok_desc->name.empty() && std::ifstream(model_dir + "/" + tok_desc->name).good()) {
            tok_file = model_dir + "/" + tok_desc->name;
        } else if (std::ifstream(model_dir + "/tokenizer.json").good()) {
            tok_file = model_dir + "/tokenizer.json";
        } else if (std::ifstream(model_dir + "/qwen.tiktoken").good()) {
            tok_file = model_dir + "/qwen.tiktoken";
        } else if (std::ifstream("./qwen.tiktoken").good()) {
            tok_file = "./qwen.tiktoken";
        }
    }
    if (!tok_file.empty()) {
        if (!m_tokenizer.load(tok_file, err)) {
            return false;
        }
        m_tokenizer.set_special_tokens(manifest.config.bos_token_id, manifest.config.eos_token_id);
    }

    // 2. Initialize KV cache
    KvCacheConfig kv_cfg;
    kv_cfg.num_layers = manifest.config.num_hidden_layers;
    kv_cfg.max_seq_len = 2048;
    kv_cfg.num_kv_heads = manifest.config.num_key_value_heads;
    kv_cfg.head_dim = manifest.config.hidden_size / manifest.config.num_attention_heads;
    m_kv_cache = std::make_unique<KvCache>(kv_cfg);

    // 3. Allocate working scratch buffers
    m_hidden_state.assign(manifest.config.hidden_size, 0.0f);
    m_logits.assign(manifest.config.vocab_size, 0.0f);

    // 4. Check for Dual-Die NPU offload models
    std::string d0_path = model_dir + "/smollm_die0.model";
    std::string d1_path = model_dir + "/smollm_die1.model";

    std::ifstream f0(d0_path, std::ios::binary);
    std::ifstream f1(d1_path, std::ios::binary);

    if (f0.is_open() && f1.is_open()) {
        m_worker_pool = std::make_unique<WorkerPool>();
        if (m_worker_pool->start(2) && m_worker_pool->active_worker_count() >= 2) {
            ModelInfo info0, info1;
            if (m_worker_pool->load_model(0, d0_path, info0) &&
                m_worker_pool->load_model(1, d1_path, info1)) {
                m_model_d0 = info0.model_id;
                m_model_d1 = info1.model_id;

                size_t in_bytes = manifest.config.hidden_size * sizeof(float);
                size_t out_bytes = (manifest.config.vocab_size / 2) * sizeof(float);

                m_d0_in = std::make_unique<DmaBuffer>(in_bytes, CacheMode::NoCache, "mmz_nid_0_part_0");
                m_d0_out = std::make_unique<DmaBuffer>(out_bytes, CacheMode::NoCache, "mmz_nid_0_part_0");
                m_d1_in = std::make_unique<DmaBuffer>(in_bytes, CacheMode::NoCache, "mmz_nid_1_part_0");
                m_d1_out = std::make_unique<DmaBuffer>(out_bytes, CacheMode::NoCache, "mmz_nid_1_part_0");

                if (m_worker_pool->prepare_buffer(0, *m_d0_in) &&
                    m_worker_pool->prepare_buffer(0, *m_d0_out) &&
                    m_worker_pool->prepare_buffer(1, *m_d1_in) &&
                    m_worker_pool->prepare_buffer(1, *m_d1_out)) {
                    m_npu_dual_die = true;
                }
            }
        }
    }

    // 5. Host fallback embedding and projection initialization
    size_t embed_elems = static_cast<size_t>(manifest.config.vocab_size) * manifest.config.hidden_size;
    m_embed.assign(embed_elems, 0.01f);
    m_lm_head_weights.assign(embed_elems, 0.01f);

    reset();
    return true;
}

void TransformerSession::set_sampler_params(const SamplerParams &params) {
    m_sampler.set_params(params);
}

void TransformerSession::reset() {
    if (m_kv_cache) m_kv_cache->reset();
    m_sampler.reset();
    m_current_pos = 0;
}

void TransformerSession::forward_lm_head_dual_die(const float *hidden_state, float *out_logits) {
    size_t half_vocab = m_manifest.config.vocab_size / 2;

    if (m_npu_dual_die && m_worker_pool) {
        // Copy hidden state to both die inputs
        size_t in_bytes = m_manifest.config.hidden_size * sizeof(float);
        std::memcpy(m_d0_in->data(), hidden_state, in_bytes);
        std::memcpy(m_d1_in->data(), hidden_state, in_bytes);

        // Submit tasks in parallel
        std::vector<DmaBuffer*> in0 = { m_d0_in.get() };
        std::vector<DmaBuffer*> out0 = { m_d0_out.get() };
        std::vector<DmaBuffer*> in1 = { m_d1_in.get() };
        std::vector<DmaBuffer*> out1 = { m_d1_out.get() };

        m_worker_pool->submit_task(0, m_model_d0, in0, out0);
        m_worker_pool->submit_task(1, m_model_d1, in1, out1);

        // Gather logits
        size_t out_bytes = half_vocab * sizeof(float);
        std::memcpy(out_logits, m_d0_out->data(), out_bytes);
        std::memcpy(out_logits + half_vocab, m_d1_out->data(), out_bytes);
    } else {
        // Host fallback matrix multiplication: logits = hidden_state @ lm_head.T
        uint32_t hidden_size = m_manifest.config.hidden_size;
        uint32_t vocab_size = m_manifest.config.vocab_size;

        for (uint32_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            const float *w_row = m_lm_head_weights.data() + static_cast<size_t>(v) * hidden_size;
            for (uint32_t d = 0; d < hidden_size; ++d) {
                sum += hidden_state[d] * w_row[d];
            }
            out_logits[v] = sum;
        }
    }
}

int32_t TransformerSession::step(int32_t token_id, int pos) {
    uint32_t hidden_size = m_manifest.config.hidden_size;

    // 1. Embedding lookup
    if (token_id >= 0 && static_cast<size_t>(token_id) < m_manifest.config.vocab_size) {
        const float *emb = m_embed.data() + static_cast<size_t>(token_id) * hidden_size;
        std::memcpy(m_hidden_state.data(), emb, hidden_size * sizeof(float));
    } else {
        std::fill(m_hidden_state.begin(), m_hidden_state.end(), 0.05f);
    }

    // 2. Transformer layers (KV Cache update)
    if (m_kv_cache) {
        for (uint32_t L = 0; L < m_manifest.config.num_hidden_layers; ++L) {
            m_kv_cache->put(L, pos, m_hidden_state.data(), m_hidden_state.data());
        }
    }

    // 3. Dual-Die LM Head execution
    forward_lm_head_dual_die(m_hidden_state.data(), m_logits.data());

    // 4. Sample next token
    int32_t next_token = m_sampler.sample(m_logits.data(), m_logits.size());
    m_sampler.accept(next_token);

    return next_token;
}

bool TransformerSession::generate(
    const std::string &prompt,
    int max_new_tokens,
    TokenCallback token_cb,
    SessionStats *out_stats
) {
    reset();

    // 1. Encode prompt
    std::vector<int32_t> prompt_tokens = m_tokenizer.encode(prompt, true);
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(m_manifest.config.bos_token_id);
    }

    m_stats.prompt_tokens = static_cast<int>(prompt_tokens.size());
    m_stats.eval_tokens = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    // 2. Prefill / Prompt evaluation
    int32_t next_token = 0;
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        next_token = step(prompt_tokens[i], m_current_pos++);
    }

    auto t_prompt = std::chrono::high_resolution_clock::now();
    m_stats.prompt_eval_ms = std::chrono::duration<double, std::milli>(t_prompt - t_start).count();

    // 3. Autoregressive token generation
    auto t_eval_start = std::chrono::high_resolution_clock::now();

    for (int step_idx = 0; step_idx < max_new_tokens; ++step_idx) {
        // Output token
        std::string piece = m_tokenizer.decode_token(next_token);
        m_stats.eval_tokens++;

        if (token_cb) {
            if (!token_cb(piece, next_token)) {
                break; // Early termination requested by caller
            }
        }

        // Stop on EOS
        if (static_cast<uint32_t>(next_token) == m_manifest.config.eos_token_id) {
            break;
        }

        // Step forward
        next_token = step(next_token, m_current_pos++);
    }

    auto t_eval_end = std::chrono::high_resolution_clock::now();
    m_stats.eval_ms = std::chrono::duration<double, std::milli>(t_eval_end - t_eval_start).count();

    if (m_stats.eval_ms > 0.0) {
        m_stats.tokens_per_sec = (m_stats.eval_tokens / m_stats.eval_ms) * 1000.0;
    }

    if (out_stats) {
        *out_stats = m_stats;
    }

    return true;
}

} // namespace eswin
