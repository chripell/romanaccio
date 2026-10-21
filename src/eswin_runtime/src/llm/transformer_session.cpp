#include "transformer_session.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace eswin {

TransformerSession::TransformerSession() = default;

TransformerSession::~TransformerSession() {
    if (m_worker_pool) {
        if (m_d0_in) m_worker_pool->unprepare_buffer(0, *m_d0_in);
        if (m_d0_out) m_worker_pool->unprepare_buffer(0, *m_d0_out);
        if (m_d1_in) m_worker_pool->unprepare_buffer(1, *m_d1_in);
        if (m_d1_out) m_worker_pool->unprepare_buffer(1, *m_d1_out);
        if (m_d0_gemm_in) m_worker_pool->unprepare_buffer(0, *m_d0_gemm_in);
        if (m_d0_gemm_out) m_worker_pool->unprepare_buffer(0, *m_d0_gemm_out);
        if (m_d1_gemm_in) m_worker_pool->unprepare_buffer(1, *m_d1_gemm_in);
        if (m_d1_gemm_out) m_worker_pool->unprepare_buffer(1, *m_d1_gemm_out);
    }
}

bool TransformerSession::load_rms_norms(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    uint32_t header[4];
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!f || header[0] != 0x524D534E) return false;

    uint32_t n_layers = header[1];
    uint32_t dim = header[2];

    m_rms_att.resize(static_cast<size_t>(n_layers) * dim);
    m_rms_ffn.resize(static_cast<size_t>(n_layers) * dim);
    m_rms_final.resize(dim);

    f.read(reinterpret_cast<char*>(m_rms_att.data()), m_rms_att.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_rms_ffn.data()), m_rms_ffn.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_rms_final.data()), m_rms_final.size() * sizeof(float));

    if (!f) return false;
    m_has_norms = true;
    return true;
}

bool TransformerSession::load_embeddings_fp16(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "[load_embeddings_fp16] Failed to open: " << path << std::endl;
        return false;
    }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);

    size_t total_elements = static_cast<size_t>(m_manifest.config.vocab_size) * m_manifest.config.hidden_size;
    size_t expected_bytes = total_elements * sizeof(es_fp16_t);
    if (static_cast<size_t>(size) < expected_bytes) {
        std::cerr << "[load_embeddings_fp16] File too small! size=" << size << " < " << expected_bytes << std::endl;
        return false;
    }

    try {
        m_embed_fp16.resize(total_elements);
    } catch (const std::exception &e) {
        std::cerr << "[load_embeddings_fp16] resize threw: " << e.what() << std::endl;
        return false;
    }

    // Read in chunks of 16MB to avoid any std::streamsize limits
    size_t bytes_read = 0;
    char *buf = reinterpret_cast<char*>(m_embed_fp16.data());
    while (bytes_read < expected_bytes && f) {
        size_t chunk = std::min(size_t(16 * 1024 * 1024), expected_bytes - bytes_read);
        f.read(buf + bytes_read, chunk);
        bytes_read += f.gcount();
    }

    if (bytes_read < expected_bytes) {
        std::cerr << "[load_embeddings_fp16] Read failed! Only read " << bytes_read << " / " << expected_bytes << " bytes" << std::endl;
        m_embed_fp16.clear();
        return false;
    }

    m_has_embed_fp16 = true;
    return true;
}

void TransformerSession::discover_and_load_layer_subgraphs(const std::string &model_dir) {
    if (!m_worker_pool) return;
    int n_layers = m_manifest.config.num_hidden_layers;
    m_layer_subgraphs.resize(n_layers);

    size_t max_gemm_bytes = 16384;
    if (m_manifest.config.hidden_size * 2 * 2 > max_gemm_bytes) {
        max_gemm_bytes = m_manifest.config.hidden_size * 2 * 2;
    }

    bool need_d0_gemm = false;
    bool need_d1_gemm = false;

    for (int L = 0; L < n_layers; ++L) {
        auto &sub = m_layer_subgraphs[L];
        sub.die_id = (m_manifest.tensor_parallel >= 2 && L >= n_layers / 2 && m_worker_pool->active_worker_count() >= 2) ? 1 : 0;

        std::string sL = std::to_string(L);
        std::vector<std::string> qkv_cands = {
            model_dir + "/layers/layer_" + sL + "_qkv.model",
            model_dir + "/qwen2_5_layer" + sL + "_qkv_npu_b1.model",
            model_dir + "/compiled_npu/qwen2_5_layer" + sL + "_qkv_npu_b1.model",
            model_dir + "/layer" + sL + "_qkv.model"
        };
        for (const auto &p : qkv_cands) {
            if (std::ifstream(p, std::ios::binary).good()) {
                ModelInfo info;
                if (m_worker_pool->load_model(sub.die_id, p, info)) {
                    sub.model_qkv = info.model_id;
                    sub.has_qkv = true;
                    m_has_any_npu_layers = true;
                    if (sub.die_id == 0) need_d0_gemm = true; else need_d1_gemm = true;
                    break;
                }
            }
        }

        std::vector<std::string> attn_cands = {
            model_dir + "/layers/layer_" + sL + "_attn_out.model",
            model_dir + "/qwen2_5_layer" + sL + "_attn_out_npu_b1.model",
            model_dir + "/compiled_npu/qwen2_5_layer" + sL + "_attn_out_npu_b1.model",
            model_dir + "/layer" + sL + "_attn_out.model"
        };
        for (const auto &p : attn_cands) {
            if (std::ifstream(p, std::ios::binary).good()) {
                ModelInfo info;
                if (m_worker_pool->load_model(sub.die_id, p, info)) {
                    sub.model_attn_out = info.model_id;
                    sub.has_attn_out = true;
                    m_has_any_npu_layers = true;
                    if (sub.die_id == 0) need_d0_gemm = true; else need_d1_gemm = true;
                    break;
                }
            }
        }

        std::vector<std::string> mlp_cands = {
            model_dir + "/layers/layer_" + sL + "_mlp.model",
            model_dir + "/qwen2_5_layer" + sL + "_mlp_npu_b1.model",
            model_dir + "/compiled_npu/qwen2_5_layer" + sL + "_mlp_npu_b1.model",
            (L == 0 ? model_dir + "/qwen2_5_mlp_opset13_npu_b1.model" : ""),
            (L == 0 ? model_dir + "/compiled_npu/qwen2_5_mlp_opset13_npu_b1.model" : ""),
            model_dir + "/layer" + sL + "_mlp.model"
        };
        for (const auto &p : mlp_cands) {
            if (!p.empty() && std::ifstream(p, std::ios::binary).good()) {
                ModelInfo info;
                if (m_worker_pool->load_model(sub.die_id, p, info)) {
                    sub.model_mlp = info.model_id;
                    sub.has_mlp = true;
                    m_has_any_npu_layers = true;
                    if (sub.die_id == 0) need_d0_gemm = true; else need_d1_gemm = true;
                    break;
                }
            }
        }
    }

    if (need_d0_gemm) {
        m_d0_gemm_in = std::make_unique<DmaBuffer>(max_gemm_bytes, CacheMode::NoCache, "mmz_nid_0_part_0");
        m_d0_gemm_out = std::make_unique<DmaBuffer>(max_gemm_bytes, CacheMode::NoCache, "mmz_nid_0_part_0");
        m_worker_pool->prepare_buffer(0, *m_d0_gemm_in);
        m_worker_pool->prepare_buffer(0, *m_d0_gemm_out);
    }
    if (need_d1_gemm && m_worker_pool->active_worker_count() >= 2) {
        m_d1_gemm_in = std::make_unique<DmaBuffer>(max_gemm_bytes, CacheMode::NoCache, "mmz_nid_1_part_0");
        m_d1_gemm_out = std::make_unique<DmaBuffer>(max_gemm_bytes, CacheMode::NoCache, "mmz_nid_1_part_0");
        m_worker_pool->prepare_buffer(1, *m_d1_gemm_in);
        m_worker_pool->prepare_buffer(1, *m_d1_gemm_out);
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
        } else if (std::ifstream(model_dir + "/tokenizer.bin").good()) {
            tok_file = model_dir + "/tokenizer.bin";
        } else if (std::ifstream(model_dir + "/tokenizer.json").good()) {
            tok_file = model_dir + "/tokenizer.json";
        } else if (std::ifstream(model_dir + "/qwen.tiktoken").good()) {
            tok_file = model_dir + "/qwen.tiktoken";
        } else if (std::ifstream("./tokenizer.bin").good()) {
            tok_file = "./tokenizer.bin";
        } else if (std::ifstream("./tokenizer.json").good()) {
            tok_file = "./tokenizer.json";
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

    // 2. Initialize KV cache. The window comes from the package manifest (or the
    // weights blob header later), never from a fixed constant.
    m_max_seq = manifest.config.context_length > 0
                    ? static_cast<int>(manifest.config.context_length)
                    : 2048;

    KvCacheConfig kv_cfg;
    kv_cfg.num_layers = manifest.config.num_hidden_layers;
    kv_cfg.max_seq_len = m_max_seq;
    kv_cfg.num_kv_heads = manifest.config.num_key_value_heads;
    kv_cfg.head_dim = manifest.config.hidden_size / manifest.config.num_attention_heads;
    m_kv_cache = std::make_unique<KvCache>(kv_cfg);

    // 3. Allocate working scratch buffers
    int dim = manifest.config.hidden_size;
    int hidden_dim = manifest.config.intermediate_size;
    int n_heads = manifest.config.num_attention_heads;
    int n_kv_heads = manifest.config.num_key_value_heads;
    int head_dim = n_heads > 0 ? (dim / n_heads) : 64;
    int kv_dim = n_kv_heads * head_dim;

    m_hidden_state.assign(dim, 0.0f);
    m_logits.assign(manifest.config.vocab_size, 0.0f);
    m_x_buf.assign(dim, 0.0f);
    m_x_norm_buf.assign(dim, 0.0f);
    m_q_buf.assign(dim, 0.0f);
    m_k_buf.assign(kv_dim, 0.0f);
    m_v_buf.assign(kv_dim, 0.0f);
    m_att_out_buf.assign(dim, 0.0f);
    m_mlp_act_buf.assign(hidden_dim, 0.0f);
    m_scores_buf.assign(static_cast<size_t>(n_heads) * m_max_seq, 0.0f);
    m_k_cache_flat.assign(static_cast<size_t>(manifest.config.num_hidden_layers) * m_max_seq * kv_dim, 0.0f);
    m_v_cache_flat.assign(static_cast<size_t>(manifest.config.num_hidden_layers) * m_max_seq * kv_dim, 0.0f);

    // 4. Try loading RMSNorms (rms_norms.bin)
    std::string norms_path;
    const auto *norm_desc = manifest.find_file_by_role("norms");
    if (norm_desc && !norm_desc->name.empty() && std::ifstream(model_dir + "/" + norm_desc->name).good()) {
        norms_path = model_dir + "/" + norm_desc->name;
    } else if (std::ifstream(model_dir + "/rms_norms.bin").good()) {
        norms_path = model_dir + "/rms_norms.bin";
    } else if (std::ifstream(model_dir + "/compiled_npu/rms_norms.bin").good()) {
        norms_path = model_dir + "/compiled_npu/rms_norms.bin";
    }
    if (!norms_path.empty()) {
        load_rms_norms(norms_path);
    }

    // 5. Try loading FP16 embeddings (embedding.bin)
    std::string emb_path;
    const auto *emb_desc = manifest.find_file_by_role("embeddings");
    if (emb_desc && !emb_desc->name.empty() && std::ifstream(model_dir + "/" + emb_desc->name).good()) {
        emb_path = model_dir + "/" + emb_desc->name;
    } else if (std::ifstream(model_dir + "/embedding.bin").good()) {
        emb_path = model_dir + "/embedding.bin";
    } else if (std::ifstream(model_dir + "/exported/embedding.bin").good()) {
        emb_path = model_dir + "/exported/embedding.bin";
    } else if (std::ifstream(model_dir + "/compiled_npu/embedding.bin").good()) {
        emb_path = model_dir + "/compiled_npu/embedding.bin";
    }
    if (!emb_path.empty()) {
        load_embeddings_fp16(emb_path);
    }

    // 6. Check for Dual-Die LM Head models & Layer Subgraphs
    std::string d0_path, d1_path;
    std::vector<std::string> d0_cands = {
        model_dir + "/qwen2_5_lm_head_d0_npu_b1.model",
        model_dir + "/compiled_npu/qwen2_5_lm_head_d0_npu_b1.model",
        model_dir + "/lm_head_d0.model",
        model_dir + "/smollm_die0.model"
    };
    std::vector<std::string> d1_cands = {
        model_dir + "/qwen2_5_lm_head_d1_npu_b1.model",
        model_dir + "/compiled_npu/qwen2_5_lm_head_d1_npu_b1.model",
        model_dir + "/lm_head_d1.model",
        model_dir + "/smollm_die1.model"
    };
    for (const auto &p : d0_cands) {
        if (std::ifstream(p, std::ios::binary).good()) { d0_path = p; break; }
    }
    for (const auto &p : d1_cands) {
        if (std::ifstream(p, std::ios::binary).good()) { d1_path = p; break; }
    }

    bool has_subgraphs = std::ifstream(model_dir + "/qwen2_5_layer0_qkv_npu_b1.model").good() ||
                         std::ifstream(model_dir + "/compiled_npu/qwen2_5_layer0_qkv_npu_b1.model").good();

    if (!d0_path.empty() || !d1_path.empty() || has_subgraphs) {
        m_worker_pool = std::make_unique<WorkerPool>();
        uint16_t num_workers = (manifest.tensor_parallel >= 2) ? 2 : 1;
        if (m_worker_pool->start(num_workers)) {
            // Load LM Head if both die models present
            if (!d0_path.empty() && !d1_path.empty() && m_worker_pool->active_worker_count() >= 2) {
                ModelInfo info0, info1;
                if (m_worker_pool->load_model(0, d0_path, info0) &&
                    m_worker_pool->load_model(1, d1_path, info1)) {
                    m_model_d0 = info0.model_id;
                    m_model_d1 = info1.model_id;

                    size_t in_bytes = manifest.config.hidden_size * sizeof(es_fp16_t);
                    size_t out_bytes = (manifest.config.vocab_size == 151936)
                        ? (76032 * sizeof(es_fp16_t))
                        : ((manifest.config.vocab_size / 2) * sizeof(es_fp16_t));

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

            // Discover and load layer subgraphs
            discover_and_load_layer_subgraphs(model_dir);
        }
    }

    // 7. Load model weights if role "weights" exists or for legacy SmolLM
    std::string weights_path;
    const auto *w_desc = manifest.find_file_by_role("weights");
    if (w_desc && !w_desc->name.empty() && std::ifstream(model_dir + "/" + w_desc->name).good()) {
        weights_path = model_dir + "/" + w_desc->name;
    } else if (manifest.config.architecture != "qwen2" && manifest.config.hidden_size == 576) {
        if (std::ifstream(model_dir + "/smollm_135m.bin").good()) {
            weights_path = model_dir + "/smollm_135m.bin";
        }
    }

    if (!weights_path.empty()) {
        if (!load_weights(weights_path)) {
            std::cerr << "[TransformerSession] Warning: Failed to load binary weights from " << weights_path << std::endl;
        }
    }

    if (!m_has_weights && !m_has_embed_fp16) {
        // Fallback placeholder embedding for test stubs
        size_t embed_elems = static_cast<size_t>(manifest.config.vocab_size) * manifest.config.hidden_size;
        m_embed.assign(embed_elems, 0.01f);
        m_lm_head_weights.assign(embed_elems, 0.01f);
        m_scores_buf.assign(static_cast<size_t>(manifest.config.num_attention_heads) * m_max_seq, 0.0f);
    }

    reset();
    return true;
}

static std::vector<float> transpose_weight_matrix(const std::vector<float> &src, int n_matrices, int rows, int cols) {
    std::vector<float> dst(src.size());
    for (int m = 0; m < n_matrices; ++m) {
        const float *src_m = src.data() + m * rows * cols;
        float *dst_m = dst.data() + m * cols * rows;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                dst_m[c * rows + r] = src_m[r * cols + c];
            }
        }
    }
    return dst;
}

bool TransformerSession::load_weights(const std::string &bin_path) {
    std::ifstream f(bin_path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }

    int header[7];
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!f) return false;

    int dim = header[0];
    int hidden_dim = header[1];
    int n_layers = header[2];
    int n_heads = header[3];
    int n_kv_heads = header[4];
    int vocab_size = header[5];
    int seq_len = header[6];
    int head_dim = n_heads > 0 ? (dim / n_heads) : 64;

    m_manifest.config.hidden_size = dim;
    m_manifest.config.intermediate_size = hidden_dim;
    m_manifest.config.num_hidden_layers = n_layers;
    m_manifest.config.num_attention_heads = n_heads;
    m_manifest.config.num_key_value_heads = n_kv_heads;
    m_manifest.config.vocab_size = vocab_size;

    m_embed.resize(static_cast<size_t>(vocab_size) * dim);
    m_rms_att.resize(static_cast<size_t>(n_layers) * dim);
    m_wq.resize(static_cast<size_t>(n_layers) * dim * dim);
    m_wk.resize(static_cast<size_t>(n_layers) * dim * (n_kv_heads * head_dim));
    m_wv.resize(static_cast<size_t>(n_layers) * dim * (n_kv_heads * head_dim));
    m_wo.resize(static_cast<size_t>(n_layers) * dim * dim);
    m_rms_ffn.resize(static_cast<size_t>(n_layers) * dim);
    m_w_gate.resize(static_cast<size_t>(n_layers) * dim * hidden_dim);
    m_w_down.resize(static_cast<size_t>(n_layers) * hidden_dim * dim);
    m_w_up.resize(static_cast<size_t>(n_layers) * dim * hidden_dim);
    m_rms_final.resize(dim);

    f.read(reinterpret_cast<char*>(m_embed.data()), m_embed.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_rms_att.data()), m_rms_att.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_wq.data()), m_wq.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_wk.data()), m_wk.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_wv.data()), m_wv.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_wo.data()), m_wo.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_rms_ffn.data()), m_rms_ffn.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_w_gate.data()), m_w_gate.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_w_down.data()), m_w_down.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_w_up.data()), m_w_up.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(m_rms_final.data()), m_rms_final.size() * sizeof(float));

    if (!f) return false;

    // Transpose all weight matrices for contiguous row-major SIMD inner products
    int kv_dim = n_kv_heads * head_dim;
    m_wq = transpose_weight_matrix(m_wq, n_layers, dim, dim);
    m_wk = transpose_weight_matrix(m_wk, n_layers, dim, kv_dim);
    m_wv = transpose_weight_matrix(m_wv, n_layers, dim, kv_dim);
    m_wo = transpose_weight_matrix(m_wo, n_layers, dim, dim);
    m_w_gate = transpose_weight_matrix(m_w_gate, n_layers, dim, hidden_dim);
    m_w_up = transpose_weight_matrix(m_w_up, n_layers, dim, hidden_dim);
    m_w_down = transpose_weight_matrix(m_w_down, n_layers, hidden_dim, dim);

    m_max_seq = seq_len > 0 ? seq_len : m_max_seq;
    size_t kv_size = static_cast<size_t>(n_layers) * m_max_seq * (n_kv_heads * head_dim);
    m_k_cache_flat.assign(kv_size, 0.0f);
    m_v_cache_flat.assign(kv_size, 0.0f);

    m_x_buf.resize(dim);
    m_x_norm_buf.resize(dim);
    m_q_buf.resize(dim);
    m_k_buf.resize(n_kv_heads * head_dim);
    m_v_buf.resize(n_kv_heads * head_dim);
    m_att_out_buf.resize(dim);
    m_mlp_act_buf.resize(hidden_dim);
    m_hidden_state.resize(dim);
    m_logits.resize(vocab_size);
    m_scores_buf.assign(static_cast<size_t>(n_heads) * m_max_seq, 0.0f);

    m_has_weights = true;
    return true;
}

void TransformerSession::rmsnorm(float *out, const float *x, const float *weight, int size, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) sum += x[i] * x[i];
    float inv_norm = 1.0f / std::sqrt((sum / size) + eps);
    for (int i = 0; i < size; ++i) {
        out[i] = x[i] * inv_norm * weight[i];
    }
}

void TransformerSession::apply_rope(float *q, float *k, int pos) {
    int n_heads = m_manifest.config.num_attention_heads;
    int n_kv_heads = m_manifest.config.num_key_value_heads;
    int head_dim = m_manifest.config.hidden_size / n_heads;
    int half_dim = head_dim / 2;
    float theta = m_manifest.config.rope_theta > 0.0f ? m_manifest.config.rope_theta : 10000.0f;

    for (int i = 0; i < half_dim; ++i) {
        float freq = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(half_dim));
        float angle = pos * freq;
        float cos_v = std::cos(angle);
        float sin_v = std::sin(angle);

        // Rotate Q heads
        for (int h = 0; h < n_heads; ++h) {
            float *qh = q + h * head_dim;
            float v0 = qh[i];
            float v1 = qh[i + half_dim];
            qh[i] = v0 * cos_v - v1 * sin_v;
            qh[i + half_dim] = v0 * sin_v + v1 * cos_v;
        }

        // Rotate K heads
        for (int h = 0; h < n_kv_heads; ++h) {
            float *kh = k + h * head_dim;
            float v0 = kh[i];
            float v1 = kh[i + half_dim];
            kh[i] = v0 * cos_v - v1 * sin_v;
            kh[i + half_dim] = v0 * sin_v + v1 * cos_v;
        }
    }
}

void TransformerSession::set_sampler_params(const SamplerParams &params) {
    m_sampler.set_params(params);
}

void TransformerSession::reset() {
    if (m_kv_cache) m_kv_cache->reset();
    if (!m_k_cache_flat.empty()) std::fill(m_k_cache_flat.begin(), m_k_cache_flat.end(), 0.0f);
    if (!m_v_cache_flat.empty()) std::fill(m_v_cache_flat.begin(), m_v_cache_flat.end(), 0.0f);
    m_sampler.reset();
    m_current_pos = 0;
}

void TransformerSession::forward_lm_head_dual_die(const float *hidden_state, float *out_logits) {
    size_t hidden_size = m_manifest.config.hidden_size;

    if (m_npu_dual_die && m_worker_pool) {
        // Convert input hidden state (FP32) to FP16 for hardware NPU inputs
        es_fp16_t *d0_in = reinterpret_cast<es_fp16_t*>(m_d0_in->data());
        es_fp16_t *d1_in = reinterpret_cast<es_fp16_t*>(m_d1_in->data());
        for (size_t i = 0; i < hidden_size; ++i) {
            es_fp16_t val = f32_to_f16(hidden_state[i]);
            d0_in[i] = val;
            d1_in[i] = val;
        }

        // Submit tasks in parallel across both NPU dies
        std::vector<DmaBuffer*> in0 = { m_d0_in.get() };
        std::vector<DmaBuffer*> out0 = { m_d0_out.get() };
        std::vector<DmaBuffer*> in1 = { m_d1_in.get() };
        std::vector<DmaBuffer*> out1 = { m_d1_out.get() };

#if defined(_OPENMP)
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                m_worker_pool->submit_task(0, m_model_d0, in0, out0);
            }
            #pragma omp section
            {
                m_worker_pool->submit_task(1, m_model_d1, in1, out1);
            }
        }
#else
        m_worker_pool->submit_task(0, m_model_d0, in0, out0);
        m_worker_pool->submit_task(1, m_model_d1, in1, out1);
#endif

        // Convert output logits from FP16 to FP32 float
        const es_fp16_t *d0_out = reinterpret_cast<const es_fp16_t*>(m_d0_out->data());
        const es_fp16_t *d1_out = reinterpret_cast<const es_fp16_t*>(m_d1_out->data());

        if (m_manifest.config.vocab_size == 151936) {
            size_t half_valid = 75968;
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (size_t i = 0; i < half_valid; ++i) {
                out_logits[i] = f16_to_f32(d0_out[i]);
                out_logits[half_valid + i] = f16_to_f32(d1_out[i]);
            }
        } else {
            size_t half_vocab = m_manifest.config.vocab_size / 2;
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (size_t i = 0; i < half_vocab; ++i) {
                out_logits[i] = f16_to_f32(d0_out[i]);
                out_logits[half_vocab + i] = f16_to_f32(d1_out[i]);
            }
        }
    } else {
        // Host fallback matrix multiplication: logits = hidden_state @ lm_head.T (tied embeddings)
        uint32_t vocab_size = m_manifest.config.vocab_size;
        const float *weights = m_lm_head_weights.empty() ? m_embed.data() : m_lm_head_weights.data();

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (uint32_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            const float *w_row = weights + static_cast<size_t>(v) * hidden_size;
            for (uint32_t d = 0; d < hidden_size; ++d) {
                sum += hidden_state[d] * w_row[d];
            }
            out_logits[v] = sum;
        }
    }
}

int32_t TransformerSession::step(int32_t token_id, int pos) {
    int dim = m_manifest.config.hidden_size;
    int hidden_dim = m_manifest.config.intermediate_size;
    int n_layers = m_manifest.config.num_hidden_layers;
    int n_heads = m_manifest.config.num_attention_heads;
    int n_kv_heads = m_manifest.config.num_key_value_heads;
    int head_dim = n_heads > 0 ? (dim / n_heads) : 64;
    int kv_dim = n_kv_heads * head_dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    float norm_eps = 1e-6f;

    // 1. Embedding lookup
    if (m_has_embed_fp16 && token_id >= 0 && static_cast<size_t>(token_id) < m_manifest.config.vocab_size) {
        const es_fp16_t *emb_ptr = m_embed_fp16.data() + static_cast<size_t>(token_id) * dim;
        for (int i = 0; i < dim; ++i) {
            m_x_buf[i] = f16_to_f32(emb_ptr[i]);
        }
    } else if (m_has_weights && token_id >= 0 && static_cast<size_t>(token_id) < m_manifest.config.vocab_size) {
        const float *emb_ptr = m_embed.data() + static_cast<size_t>(token_id) * dim;
        std::memcpy(m_x_buf.data(), emb_ptr, dim * sizeof(float));
    } else {
        std::fill(m_x_buf.begin(), m_x_buf.end(), 0.0f);
    }

    // 2. Transformer layers (0 .. n_layers - 1)
    for (int L = 0; L < n_layers; ++L) {
        const NpuLayerSubgraphs *npu = (L < static_cast<int>(m_layer_subgraphs.size())) ? &m_layer_subgraphs[L] : nullptr;

        // Attention RMSNorm
        const float *rms_att_w = m_rms_att.empty() ? nullptr : (m_rms_att.data() + static_cast<size_t>(L) * dim);
        if (rms_att_w) {
            rmsnorm(m_x_norm_buf.data(), m_x_buf.data(), rms_att_w, dim, norm_eps);
        } else {
            std::memcpy(m_x_norm_buf.data(), m_x_buf.data(), dim * sizeof(float));
        }

        // Q, K, V Projections
        if (npu && npu->has_qkv && m_worker_pool) {
            uint16_t die = npu->die_id;
            DmaBuffer *in_buf = (die == 0) ? m_d0_gemm_in.get() : m_d1_gemm_in.get();
            DmaBuffer *out_buf = (die == 0) ? m_d0_gemm_out.get() : m_d1_gemm_out.get();

            es_fp16_t *in_ptr = reinterpret_cast<es_fp16_t*>(in_buf->data());
            for (int i = 0; i < dim; ++i) {
                in_ptr[i] = f32_to_f16(m_x_norm_buf[i]);
            }

            m_worker_pool->submit_task(die, npu->model_qkv, { in_buf }, { out_buf });

            const es_fp16_t *out_ptr = reinterpret_cast<const es_fp16_t*>(out_buf->data());
            // Q: 0 .. dim - 1
            for (int i = 0; i < dim; ++i) {
                m_q_buf[i] = f16_to_f32(out_ptr[i]);
            }
            // K: dim .. dim + kv_dim - 1
            for (int i = 0; i < kv_dim; ++i) {
                m_k_buf[i] = f16_to_f32(out_ptr[dim + i]);
            }
            // V: dim + kv_dim .. dim + 2*kv_dim - 1
            for (int i = 0; i < kv_dim; ++i) {
                m_v_buf[i] = f16_to_f32(out_ptr[dim + kv_dim + i]);
            }
        } else if (m_has_weights) {
            const float *wq_l = m_wq.data() + static_cast<size_t>(L) * dim * dim;
            const float *wk_l = m_wk.data() + static_cast<size_t>(L) * kv_dim * dim;
            const float *wv_l = m_wv.data() + static_cast<size_t>(L) * kv_dim * dim;

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < dim; ++i) {
                const float *w_row = wq_l + i * dim;
                float sum = 0.0f;
                for (int j = 0; j < dim; ++j) sum += m_x_norm_buf[j] * w_row[j];
                m_q_buf[i] = sum;
            }

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < kv_dim; ++i) {
                const float *wk_row = wk_l + i * dim;
                const float *wv_row = wv_l + i * dim;
                float sum_k = 0.0f, sum_v = 0.0f;
                for (int j = 0; j < dim; ++j) {
                    sum_k += m_x_norm_buf[j] * wk_row[j];
                    sum_v += m_x_norm_buf[j] * wv_row[j];
                }
                m_k_buf[i] = sum_k;
                m_v_buf[i] = sum_v;
            }
        }

        // Apply RoPE
        apply_rope(m_q_buf.data(), m_k_buf.data(), pos);

        // Save to KV cache
        float *k_cache_pos = m_k_cache_flat.data() + (static_cast<size_t>(L) * m_max_seq + pos) * kv_dim;
        float *v_cache_pos = m_v_cache_flat.data() + (static_cast<size_t>(L) * m_max_seq + pos) * kv_dim;
        std::memcpy(k_cache_pos, m_k_buf.data(), kv_dim * sizeof(float));
        std::memcpy(v_cache_pos, m_v_buf.data(), kv_dim * sizeof(float));

        // Grouped Query Attention (GQA)
        std::fill(m_att_out_buf.begin(), m_att_out_buf.end(), 0.0f);
        int group_ratio = (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1;

        #pragma omp parallel for schedule(static)
        for (int h = 0; h < n_heads; ++h) {
            int kv_h = h / group_ratio;
            const float *q_head = m_q_buf.data() + h * head_dim;
            float *out_head = m_att_out_buf.data() + h * head_dim;

            float *scores = m_scores_buf.data() + h * m_max_seq;
            float max_s = -1e9f;
            for (int t = 0; t <= pos; ++t) {
                const float *k_head = m_k_cache_flat.data() + (static_cast<size_t>(L) * m_max_seq + t) * kv_dim + kv_h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += q_head[d] * k_head[d];
                scores[t] = dot * scale;
                if (scores[t] > max_s) max_s = scores[t];
            }

            float exp_sum = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                scores[t] = std::exp(scores[t] - max_s);
                exp_sum += scores[t];
            }
            float inv_exp_sum = 1.0f / (exp_sum > 0.0f ? exp_sum : 1.0f);

            for (int t = 0; t <= pos; ++t) {
                float weight = scores[t] * inv_exp_sum;
                const float *v_head = m_v_cache_flat.data() + (static_cast<size_t>(L) * m_max_seq + t) * kv_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    out_head[d] += weight * v_head[d];
                }
            }
        }

        // Attention Out Projection + Residual
        if (npu && npu->has_attn_out && m_worker_pool) {
            uint16_t die = npu->die_id;
            DmaBuffer *in_buf = (die == 0) ? m_d0_gemm_in.get() : m_d1_gemm_in.get();
            DmaBuffer *out_buf = (die == 0) ? m_d0_gemm_out.get() : m_d1_gemm_out.get();

            es_fp16_t *in_ptr = reinterpret_cast<es_fp16_t*>(in_buf->data());
            for (int i = 0; i < dim; ++i) {
                in_ptr[i] = f32_to_f16(m_att_out_buf[i]);
            }

            m_worker_pool->submit_task(die, npu->model_attn_out, { in_buf }, { out_buf });

            const es_fp16_t *out_ptr = reinterpret_cast<const es_fp16_t*>(out_buf->data());
            for (int i = 0; i < dim; ++i) {
                m_x_buf[i] += f16_to_f32(out_ptr[i]);
            }
        } else if (m_has_weights) {
            const float *wo_l = m_wo.data() + static_cast<size_t>(L) * dim * dim;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < dim; ++i) {
                const float *wo_row = wo_l + i * dim;
                float sum = 0.0f;
                for (int j = 0; j < dim; ++j) sum += m_att_out_buf[j] * wo_row[j];
                m_x_buf[i] += sum;
            }
        }

        // Post-Attention RMSNorm
        const float *rms_ffn_w = m_rms_ffn.empty() ? nullptr : (m_rms_ffn.data() + static_cast<size_t>(L) * dim);
        if (rms_ffn_w) {
            rmsnorm(m_x_norm_buf.data(), m_x_buf.data(), rms_ffn_w, dim, norm_eps);
        } else {
            std::memcpy(m_x_norm_buf.data(), m_x_buf.data(), dim * sizeof(float));
        }

        // SwiGLU MLP + Residual
        if (npu && npu->has_mlp && m_worker_pool) {
            uint16_t die = npu->die_id;
            DmaBuffer *in_buf = (die == 0) ? m_d0_gemm_in.get() : m_d1_gemm_in.get();
            DmaBuffer *out_buf = (die == 0) ? m_d0_gemm_out.get() : m_d1_gemm_out.get();

            es_fp16_t *in_ptr = reinterpret_cast<es_fp16_t*>(in_buf->data());
            for (int i = 0; i < dim; ++i) {
                in_ptr[i] = f32_to_f16(m_x_norm_buf[i]);
            }

            m_worker_pool->submit_task(die, npu->model_mlp, { in_buf }, { out_buf });

            const es_fp16_t *out_ptr = reinterpret_cast<const es_fp16_t*>(out_buf->data());
            for (int i = 0; i < dim; ++i) {
                m_x_buf[i] += f16_to_f32(out_ptr[i]);
            }
        } else if (m_has_weights) {
            const float *w_gate_l = m_w_gate.data() + static_cast<size_t>(L) * hidden_dim * dim;
            const float *w_up_l = m_w_up.data() + static_cast<size_t>(L) * hidden_dim * dim;
            const float *w_down_l = m_w_down.data() + static_cast<size_t>(L) * dim * hidden_dim;

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < hidden_dim; ++i) {
                const float *wg_row = w_gate_l + i * dim;
                const float *wu_row = w_up_l + i * dim;
                float g = 0.0f, u = 0.0f;
                for (int j = 0; j < dim; ++j) {
                    g += m_x_norm_buf[j] * wg_row[j];
                    u += m_x_norm_buf[j] * wu_row[j];
                }
                float silu = g / (1.0f + std::exp(-g));
                m_mlp_act_buf[i] = silu * u;
            }

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < dim; ++i) {
                const float *wd_row = w_down_l + i * hidden_dim;
                float sum = 0.0f;
                for (int j = 0; j < hidden_dim; ++j) sum += m_mlp_act_buf[j] * wd_row[j];
                m_x_buf[i] += sum;
            }
        }
    }

    // 3. Final RMSNorm
    if (!m_rms_final.empty()) {
        rmsnorm(m_x_norm_buf.data(), m_x_buf.data(), m_rms_final.data(), dim, norm_eps);
    } else {
        std::memcpy(m_x_norm_buf.data(), m_x_buf.data(), dim * sizeof(float));
    }

    // 4. Dual-Die LM Head execution (or host fallback)
    forward_lm_head_dual_die(m_x_norm_buf.data(), m_logits.data());

    // 5. Sample next token
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
    bool add_bos = (m_manifest.config.architecture != "qwen2" && prompt.find("<|im_start|>") == std::string::npos);
    std::vector<int32_t> prompt_tokens = m_tokenizer.encode(prompt, add_bos);
    if (prompt_tokens.empty() && m_manifest.config.bos_token_id >= 0 && add_bos) {
        prompt_tokens.push_back(m_manifest.config.bos_token_id);
    }

    m_stats.prompt_tokens = static_cast<int>(prompt_tokens.size());
    m_stats.eval_tokens = 0;

    // The flat KV buffers are indexed by absolute position with no bounds check, so a
    // turn that would walk past the compiled window must be shrunk rather than silently
    // corrupting attention state — the overflow class behind Framework Issue #47.
    if (m_max_seq > 0) {
        if (static_cast<int>(prompt_tokens.size()) >= m_max_seq) {
            return false;
        }
        int room = m_max_seq - static_cast<int>(prompt_tokens.size());
        if (max_new_tokens > room) {
            max_new_tokens = room;
        }
    }

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

        // Stop on EOS (<|im_end|>, <|endoftext|>, or manifest eos_token_id)
        bool is_eos = (static_cast<uint32_t>(next_token) == m_manifest.config.eos_token_id);
        if (m_manifest.config.architecture == "qwen2" || m_manifest.config.vocab_size > 100000) {
            if (next_token == 151643 || next_token == 151645) { // <|endoftext|>, <|im_end|>
                is_eos = true;
            }
        } else {
            if (next_token == 0 || next_token == 2) {
                is_eos = true;
            }
        }
        if (is_eos) {
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
