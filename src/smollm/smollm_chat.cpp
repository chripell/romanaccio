/*
 * smollm_chat.cpp - SmolLM-135M Interactive Chat Engine
 * 
 * Hardware Acceleration: ESWIN EIC7700X Dual-NPU (or High-Performance CPU OpenMP)
 * 
 * Features:
 *   - Autoregressive Transformer inference (30 layers, 576 dim, SwiGLU, RoPE, GQA)
 *   - Key-Value (KV) caching for low-latency token generation
 *   - Optional Dual-Die NPU offload for LM Head via ESWIN EDLA runtime
 *   - Automatic high-speed CPU OpenMP fallback on host or if NPU is uninitialized
 *   - Byte-level BPE tokenizer (encodes arbitrary text, spaces, subwords, symbols)
 *   - Advanced sampler (Temperature, Top-K, Top-P, Sliding-Window Repetition Penalty)
 *   - Fix for Issue #47: Loop mitigation and sliding-window penalty
 *   - Interactive REPL prompt with real-time streaming output and tokens/s metrics
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <memory>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <dlfcn.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// -----------------------------------------------------------------------------
// 1. ESWIN NPU Types and Function Pointers (for runtime dlopen)
// -----------------------------------------------------------------------------
typedef struct {
    uint64_t memFd;
    uint64_t offset;
    uint64_t size;
    uint64_t reserve;
} ES_DEV_BUF_S;

typedef struct {
    uint32_t taskId;
    uint32_t modelId;
    uint8_t inputFdNum;
    uint8_t outputFdNum;
    void *callback;
    void *callbackArg;
    uint8_t reserve[32];
    ES_DEV_BUF_S inputFd[16];
    ES_DEV_BUF_S outputFd[16];
} NPU_TASK_S;

typedef int (*fn_ES_NPU_GetNumDevices)(uint16_t *deviceNum);
typedef int (*fn_ES_NPU_SetDevice)(uint16_t deviceId);
typedef int (*fn_ES_NPU_ReleaseDevice)(uint16_t deviceId);
typedef int (*fn_ES_NPU_LoadModelFromFile)(uint32_t *modelId, const char *modelPath);
typedef int (*fn_ES_NPU_UnloadModel)(uint32_t modelId);
typedef int (*fn_ES_NPU_Prepare)(ES_DEV_BUF_S *devBuf);
typedef int (*fn_ES_NPU_Unprepare)(ES_DEV_BUF_S *devBuf);
typedef int (*fn_ES_NPU_Submit)(NPU_TASK_S *tasks, uint32_t numTasks);

typedef int (*fn_ES_SYS_MemAlloc)(uint64_t *pMemFd, int enRemapMode, const char *pMmzName, const char *pPartName, uint64_t u64Size);
typedef int (*fn_ES_SYS_MemFree)(uint64_t u64MemFd);
typedef void* (*fn_ES_SYS_Mmap)(uint64_t u64MemFd, uint64_t u64Size, int enRemapMode);
typedef int (*fn_ES_SYS_Munmap)(void *pVirtAddr, uint64_t u64Size);

class DualNpuLmHeadEngine {
public:
    bool available = false;
    void *h_npu = nullptr;
    void *h_sys = nullptr;

    fn_ES_NPU_GetNumDevices p_GetNumDevices = nullptr;
    fn_ES_NPU_SetDevice p_SetDevice = nullptr;
    fn_ES_NPU_ReleaseDevice p_ReleaseDevice = nullptr;
    fn_ES_NPU_LoadModelFromFile p_LoadModel = nullptr;
    fn_ES_NPU_UnloadModel p_UnloadModel = nullptr;
    fn_ES_NPU_Prepare p_Prepare = nullptr;
    fn_ES_NPU_Unprepare p_Unprepare = nullptr;
    fn_ES_NPU_Submit p_Submit = nullptr;

    fn_ES_SYS_MemAlloc p_MemAlloc = nullptr;
    fn_ES_SYS_MemFree p_MemFree = nullptr;
    fn_ES_SYS_Mmap p_Mmap = nullptr;
    fn_ES_SYS_Munmap p_Munmap = nullptr;

    uint32_t model_d0 = 0;
    uint32_t model_d1 = 0;
    NPU_TASK_S task_d0;
    NPU_TASK_S task_d1;
    float *in_d0_map = nullptr;
    float *in_d1_map = nullptr;
    float *out_d0_map = nullptr;
    float *out_d1_map = nullptr;

    bool init(const std::string &d0_model_path, const std::string &d1_model_path) {
        h_npu = dlopen("/lib/libedla_runtime.so.1.0", RTLD_NOW);
        if (!h_npu) h_npu = dlopen("libedla_runtime.so", RTLD_NOW);
        h_sys = dlopen("/lib/libes_sys.so.1", RTLD_NOW);
        if (!h_sys) h_sys = dlopen("libes_sys.so", RTLD_NOW);

        if (!h_npu || !h_sys) {
            return false;
        }

        p_GetNumDevices = (fn_ES_NPU_GetNumDevices)dlsym(h_npu, "ES_NPU_GetNumDevices");
        p_SetDevice = (fn_ES_NPU_SetDevice)dlsym(h_npu, "ES_NPU_SetDevice");
        p_ReleaseDevice = (fn_ES_NPU_ReleaseDevice)dlsym(h_npu, "ES_NPU_ReleaseDevice");
        p_LoadModel = (fn_ES_NPU_LoadModelFromFile)dlsym(h_npu, "ES_NPU_LoadModelFromFile");
        p_UnloadModel = (fn_ES_NPU_UnloadModel)dlsym(h_npu, "ES_NPU_UnloadModel");
        p_Prepare = (fn_ES_NPU_Prepare)dlsym(h_npu, "ES_NPU_Prepare");
        p_Unprepare = (fn_ES_NPU_Unprepare)dlsym(h_npu, "ES_NPU_Unprepare");
        p_Submit = (fn_ES_NPU_Submit)dlsym(h_npu, "ES_NPU_Submit");

        p_MemAlloc = (fn_ES_SYS_MemAlloc)dlsym(h_sys, "ES_SYS_MemAlloc");
        p_MemFree = (fn_ES_SYS_MemFree)dlsym(h_sys, "ES_SYS_MemFree");
        p_Mmap = (fn_ES_SYS_Mmap)dlsym(h_sys, "ES_SYS_Mmap");
        p_Munmap = (fn_ES_SYS_Munmap)dlsym(h_sys, "ES_SYS_Munmap");

        if (!p_GetNumDevices || !p_SetDevice || !p_LoadModel || !p_Submit || !p_MemAlloc || !p_Mmap) {
            return false;
        }

        uint16_t numDev = 0;
        if (p_GetNumDevices(&numDev) != 0 || numDev < 2) {
            return false;
        }

        // Initialize Die 0
        p_SetDevice(0);
        if (p_LoadModel(&model_d0, d0_model_path.c_str()) != 0) return false;

        // Initialize Die 1
        p_SetDevice(1);
        if (p_LoadModel(&model_d1, d1_model_path.c_str()) != 0) return false;

        // Allocate buffers: input is [1, 576, 1, 1] FP32 (2304 bytes)
        // output is [1, 24576, 1, 1] FP32 (98304 bytes)
        const size_t in_size = 576 * sizeof(float);
        const size_t out_size = 24576 * sizeof(float);

        // Die 0 buffers
        p_SetDevice(0);
        memset(&task_d0, 0, sizeof(task_d0));
        task_d0.modelId = model_d0;
        task_d0.inputFdNum = 1;
        task_d0.outputFdNum = 1;

        p_MemAlloc(&task_d0.inputFd[0].memFd, 0, "smollm", "mmz_nid_0_part_0", in_size);
        task_d0.inputFd[0].size = in_size;
        p_Prepare(&task_d0.inputFd[0]);
        in_d0_map = (float*)p_Mmap(task_d0.inputFd[0].memFd, in_size, 0);

        p_MemAlloc(&task_d0.outputFd[0].memFd, 0, "smollm", "mmz_nid_0_part_0", out_size);
        task_d0.outputFd[0].size = out_size;
        p_Prepare(&task_d0.outputFd[0]);
        out_d0_map = (float*)p_Mmap(task_d0.outputFd[0].memFd, out_size, 0);

        // Die 1 buffers
        p_SetDevice(1);
        memset(&task_d1, 0, sizeof(task_d1));
        task_d1.modelId = model_d1;
        task_d1.inputFdNum = 1;
        task_d1.outputFdNum = 1;

        p_MemAlloc(&task_d1.inputFd[0].memFd, 0, "smollm", "mmz_nid_1_part_0", in_size);
        task_d1.inputFd[0].size = in_size;
        p_Prepare(&task_d1.inputFd[0]);
        in_d1_map = (float*)p_Mmap(task_d1.inputFd[0].memFd, in_size, 0);

        p_MemAlloc(&task_d1.outputFd[0].memFd, 0, "smollm", "mmz_nid_1_part_0", out_size);
        task_d1.outputFd[0].size = out_size;
        p_Prepare(&task_d1.outputFd[0]);
        out_d1_map = (float*)p_Mmap(task_d1.outputFd[0].memFd, out_size, 0);

        available = true;
        return true;
    }

    void forward(const float *x_norm, float *out_logits) {
        if (!available) return;

        // Copy input to both dies
        memcpy(in_d0_map, x_norm, 576 * sizeof(float));
        memcpy(in_d1_map, x_norm, 576 * sizeof(float));

        // Submit to Die 0
        p_SetDevice(0);
        p_Submit(&task_d0, 1);

        // Submit to Die 1
        p_SetDevice(1);
        p_Submit(&task_d1, 1);

        // Gather logits: Die 0 (0..24575), Die 1 (24576..49151)
        memcpy(out_logits, out_d0_map, 24576 * sizeof(float));
        memcpy(out_logits + 24576, out_d1_map, 24576 * sizeof(float));
    }

    ~DualNpuLmHeadEngine() {
        if (available) {
            p_SetDevice(0);
            p_Unprepare(&task_d0.inputFd[0]);
            p_Unprepare(&task_d0.outputFd[0]);
            p_Munmap(in_d0_map, 576 * sizeof(float));
            p_Munmap(out_d0_map, 24576 * sizeof(float));
            p_MemFree(task_d0.inputFd[0].memFd);
            p_MemFree(task_d0.outputFd[0].memFd);
            p_UnloadModel(model_d0);
            p_ReleaseDevice(0);

            p_SetDevice(1);
            p_Unprepare(&task_d1.inputFd[0]);
            p_Unprepare(&task_d1.outputFd[0]);
            p_Munmap(in_d1_map, 576 * sizeof(float));
            p_Munmap(out_d1_map, 24576 * sizeof(float));
            p_MemFree(task_d1.inputFd[0].memFd);
            p_MemFree(task_d1.outputFd[0].memFd);
            p_UnloadModel(model_d1);
            p_ReleaseDevice(1);
        }
        if (h_npu) dlclose(h_npu);
        if (h_sys) dlclose(h_sys);
    }
};

// -----------------------------------------------------------------------------
// 2. SmolLM-135M Configuration
// -----------------------------------------------------------------------------
struct SmolLMConfig {
    int dim = 576;
    int hidden_dim = 1536;
    int n_layers = 30;
    int n_heads = 9;
    int n_kv_heads = 3;
    int vocab_size = 49152;
    int seq_len = 2048;
    int head_dim = 64; // dim / n_heads
};

// -----------------------------------------------------------------------------
// 3. Byte-Level BPE Tokenizer
// -----------------------------------------------------------------------------
class SmolLMTokenizer {
public:
    int vocab_size = 0;
    std::vector<std::string> id_to_piece;
    std::unordered_map<std::string, int> piece_to_id;

    bool load(const std::string &bin_path) {
        std::ifstream f(bin_path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "[-] Error opening tokenizer file: " << bin_path << std::endl;
            return false;
        }

        f.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
        id_to_piece.resize(vocab_size);

        for (int i = 0; i < vocab_size; ++i) {
            int len = 0;
            f.read(reinterpret_cast<char*>(&len), sizeof(int));
            std::string s(len, '\0');
            f.read(&s[0], len);
            id_to_piece[i] = s;
            piece_to_id[s] = i;
        }
        return true;
    }

    std::string decode_token(int id) const {
        if (id < 0 || id >= vocab_size) return "";
        return id_to_piece[id];
    }

    // Longest-prefix subword greedy match (standard BPE encoding)
    std::vector<int> encode(const std::string &text) const {
        std::vector<int> tokens;
        size_t i = 0;
        while (i < text.size()) {
            bool matched = false;
            size_t max_search = std::min((size_t)32, text.size() - i);
            for (size_t l = max_search; l >= 1; --l) {
                std::string sub = text.substr(i, l);
                auto it = piece_to_id.find(sub);
                if (it != piece_to_id.end()) {
                    tokens.push_back(it->second);
                    i += l;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                i++; // skip unknown byte
            }
        }
        return tokens;
    }
};

// -----------------------------------------------------------------------------
// 4. SmolLM-135M Model Weights & Transformer Implementation
// -----------------------------------------------------------------------------
class SmolLMModel {
public:
    SmolLMConfig cfg;

    // Weights buffers
    std::vector<float> embed;        // [vocab_size, dim]
    std::vector<float> rms_att;      // [n_layers, dim]
    std::vector<float> wq;           // [n_layers, dim, dim]
    std::vector<float> wk;           // [n_layers, dim, n_kv_heads * head_dim]
    std::vector<float> wv;           // [n_layers, dim, n_kv_heads * head_dim]
    std::vector<float> wo;           // [n_layers, dim, dim]
    std::vector<float> rms_ffn;      // [n_layers, dim]
    std::vector<float> w_gate;       // [n_layers, dim, hidden_dim]
    std::vector<float> w_down;       // [n_layers, hidden_dim, dim]
    std::vector<float> w_up;         // [n_layers, dim, hidden_dim]
    std::vector<float> rms_final;    // [dim]

    // KV Cache: [layers, max_seq, n_kv_heads, head_dim]
    int max_seq = 2048;
    std::vector<float> k_cache;
    std::vector<float> v_cache;

    // Fast working scratch buffers
    std::vector<float> x_buf;
    std::vector<float> x_norm_buf;
    std::vector<float> q_buf;
    std::vector<float> k_buf;
    std::vector<float> v_buf;
    std::vector<float> att_out_buf;
    std::vector<float> gate_buf;
    std::vector<float> up_buf;
    std::vector<float> mlp_act_buf;
    std::vector<float> logits_buf;

    // Dual-Die NPU LM Head engine
    DualNpuLmHeadEngine npu_engine;

    bool load_weights(const std::string &bin_path) {
        std::ifstream f(bin_path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "[-] Error opening model file: " << bin_path << std::endl;
            return false;
        }

        // Read header
        int header[7];
        f.read(reinterpret_cast<char*>(header), sizeof(header));
        cfg.dim = header[0];
        cfg.hidden_dim = header[1];
        cfg.n_layers = header[2];
        cfg.n_heads = header[3];
        cfg.n_kv_heads = header[4];
        cfg.vocab_size = header[5];
        cfg.seq_len = header[6];
        cfg.head_dim = cfg.dim / cfg.n_heads;

        // Allocate weights
        embed.resize(cfg.vocab_size * cfg.dim);
        rms_att.resize(cfg.n_layers * cfg.dim);
        wq.resize(cfg.n_layers * cfg.dim * cfg.dim);
        wk.resize(cfg.n_layers * cfg.dim * (cfg.n_kv_heads * cfg.head_dim));
        wv.resize(cfg.n_layers * cfg.dim * (cfg.n_kv_heads * cfg.head_dim));
        wo.resize(cfg.n_layers * cfg.dim * cfg.dim);
        rms_ffn.resize(cfg.n_layers * cfg.dim);
        w_gate.resize(cfg.n_layers * cfg.dim * cfg.hidden_dim);
        w_down.resize(cfg.n_layers * cfg.hidden_dim * cfg.dim);
        w_up.resize(cfg.n_layers * cfg.dim * cfg.hidden_dim);
        rms_final.resize(cfg.dim);

        // Read arrays
        f.read(reinterpret_cast<char*>(embed.data()), embed.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(rms_att.data()), rms_att.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(wq.data()), wq.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(wk.data()), wk.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(wv.data()), wv.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(wo.data()), wo.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(rms_ffn.data()), rms_ffn.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(w_gate.data()), w_gate.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(w_down.data()), w_down.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(w_up.data()), w_up.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(rms_final.data()), rms_final.size() * sizeof(float));

        init_buffers();
        return true;
    }

    void init_buffers() {
        k_cache.assign(cfg.n_layers * max_seq * cfg.n_kv_heads * cfg.head_dim, 0.0f);
        v_cache.assign(cfg.n_layers * max_seq * cfg.n_kv_heads * cfg.head_dim, 0.0f);

        x_buf.resize(cfg.dim);
        x_norm_buf.resize(cfg.dim);
        q_buf.resize(cfg.n_heads * cfg.head_dim);
        k_buf.resize(cfg.n_kv_heads * cfg.head_dim);
        v_buf.resize(cfg.n_kv_heads * cfg.head_dim);
        att_out_buf.resize(cfg.dim);
        gate_buf.resize(cfg.hidden_dim);
        up_buf.resize(cfg.hidden_dim);
        mlp_act_buf.resize(cfg.hidden_dim);
        logits_buf.resize(cfg.vocab_size);
    }

    void reset_kv_cache() {
        std::fill(k_cache.begin(), k_cache.end(), 0.0f);
        std::fill(v_cache.begin(), v_cache.end(), 0.0f);
    }

    // RMSNorm: (x / sqrt(mean(x^2) + eps)) * weight
    void rmsnorm(float *out, const float *x, const float *weight, int size, float eps = 1e-5f) {
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) sum += x[i] * x[i];
        float inv_norm = 1.0f / std::sqrt((sum / size) + eps);
        for (int i = 0; i < size; ++i) {
            out[i] = x[i] * inv_norm * weight[i];
        }
    }

    // RoPE Rotary Positional Embedding
    void apply_rope(float *q, float *k, int pos) {
        int half_dim = cfg.head_dim / 2; // 32
        float theta = 10000.0f;

        for (int i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(theta, (float)(i) / (float)half_dim);
            float angle = pos * freq;
            float cos_v = std::cos(angle);
            float sin_v = std::sin(angle);

            // Rotate Q heads
            for (int h = 0; h < cfg.n_heads; ++h) {
                float *qh = q + h * cfg.head_dim;
                float v0 = qh[i];
                float v1 = qh[i + half_dim];
                qh[i] = v0 * cos_v - v1 * sin_v;
                qh[i + half_dim] = v0 * sin_v + v1 * cos_v;
            }

            // Rotate K heads
            for (int h = 0; h < cfg.n_kv_heads; ++h) {
                float *kh = k + h * cfg.head_dim;
                float v0 = kh[i];
                float v1 = kh[i + half_dim];
                kh[i] = v0 * cos_v - v1 * sin_v;
                kh[i + half_dim] = v0 * sin_v + v1 * cos_v;
            }
        }
    }

    // Forward single token through 30 layers
    float* forward_token(int token_id, int pos) {
        // 1. Embedding lookup
        const float *emb_ptr = embed.data() + token_id * cfg.dim;
        std::memcpy(x_buf.data(), emb_ptr, cfg.dim * sizeof(float));

        float scale = 1.0f / std::sqrt((float)cfg.head_dim);

        // 2. Transformer layers
        for (int L = 0; L < cfg.n_layers; ++L) {
            // Attention RMSNorm
            rmsnorm(x_norm_buf.data(), x_buf.data(), rms_att.data() + L * cfg.dim, cfg.dim);

            // Q, K, V matrix projections: y = x @ W
            const float *wq_l = wq.data() + L * cfg.dim * cfg.dim;
            const float *wk_l = wk.data() + L * cfg.dim * (cfg.n_kv_heads * cfg.head_dim);
            const float *wv_l = wv.data() + L * cfg.dim * (cfg.n_kv_heads * cfg.head_dim);

            #pragma omp parallel for
            for (int i = 0; i < cfg.dim; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < cfg.dim; ++j) {
                    sum += x_norm_buf[j] * wq_l[j * cfg.dim + i];
                }
                q_buf[i] = sum;
            }

            int kv_dim = cfg.n_kv_heads * cfg.head_dim;
            #pragma omp parallel for
            for (int i = 0; i < kv_dim; ++i) {
                float sum_k = 0.0f;
                float sum_v = 0.0f;
                for (int j = 0; j < cfg.dim; ++j) {
                    sum_k += x_norm_buf[j] * wk_l[j * kv_dim + i];
                    sum_v += x_norm_buf[j] * wv_l[j * kv_dim + i];
                }
                k_buf[i] = sum_k;
                v_buf[i] = sum_v;
            }

            // Apply RoPE
            apply_rope(q_buf.data(), k_buf.data(), pos);

            // Save to KV cache
            float *k_cache_pos = k_cache.data() + (L * max_seq + pos) * kv_dim;
            float *v_cache_pos = v_cache.data() + (L * max_seq + pos) * kv_dim;
            std::memcpy(k_cache_pos, k_buf.data(), kv_dim * sizeof(float));
            std::memcpy(v_cache_pos, v_buf.data(), kv_dim * sizeof(float));

            // Grouped Query Attention (9 Q heads, 3 KV heads: 3:1 ratio)
            std::fill(att_out_buf.begin(), att_out_buf.end(), 0.0f);

            #pragma omp parallel for
            for (int h = 0; h < cfg.n_heads; ++h) {
                int kv_h = h / 3;
                const float *q_head = q_buf.data() + h * cfg.head_dim;
                float *out_head = att_out_buf.data() + h * cfg.head_dim;

                // Compute attention scores over history [0 .. pos]
                std::vector<float> scores(pos + 1);
                float max_s = -1e9f;
                for (int t = 0; t <= pos; ++t) {
                    const float *k_head = k_cache.data() + (L * max_seq + t) * kv_dim + kv_h * cfg.head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < cfg.head_dim; ++d) {
                        dot += q_head[d] * k_head[d];
                    }
                    scores[t] = dot * scale;
                    if (scores[t] > max_s) max_s = scores[t];
                }

                // Softmax
                float exp_sum = 0.0f;
                for (int t = 0; t <= pos; ++t) {
                    scores[t] = std::exp(scores[t] - max_s);
                    exp_sum += scores[t];
                }
                float inv_exp_sum = 1.0f / exp_sum;

                // Weighted sum of values
                for (int t = 0; t <= pos; ++t) {
                    float weight = scores[t] * inv_exp_sum;
                    const float *v_head = v_cache.data() + (L * max_seq + t) * kv_dim + kv_h * cfg.head_dim;
                    for (int d = 0; d < cfg.head_dim; ++d) {
                        out_head[d] += weight * v_head[d];
                    }
                }
            }

            // Output projection + residual: x += att_out @ wo
            const float *wo_l = wo.data() + L * cfg.dim * cfg.dim;
            #pragma omp parallel for
            for (int i = 0; i < cfg.dim; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < cfg.dim; ++j) {
                    sum += att_out_buf[j] * wo_l[j * cfg.dim + i];
                }
                x_buf[i] += sum;
            }

            // FFN RMSNorm
            rmsnorm(x_norm_buf.data(), x_buf.data(), rms_ffn.data() + L * cfg.dim, cfg.dim);

            // SwiGLU MLP: DownProj(SiLU(Gate) * Up)
            const float *w_gate_l = w_gate.data() + L * cfg.dim * cfg.hidden_dim;
            const float *w_up_l = w_up.data() + L * cfg.dim * cfg.hidden_dim;
            const float *w_down_l = w_down.data() + L * cfg.hidden_dim * cfg.dim;

            #pragma omp parallel for
            for (int i = 0; i < cfg.hidden_dim; ++i) {
                float g = 0.0f;
                float u = 0.0f;
                for (int j = 0; j < cfg.dim; ++j) {
                    g += x_norm_buf[j] * w_gate_l[j * cfg.hidden_dim + i];
                    u += x_norm_buf[j] * w_up_l[j * cfg.hidden_dim + i];
                }
                float silu = g / (1.0f + std::exp(-g));
                mlp_act_buf[i] = silu * u;
            }

            // Down projection + residual: x += mlp_act @ w_down
            #pragma omp parallel for
            for (int i = 0; i < cfg.dim; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < cfg.hidden_dim; ++j) {
                    sum += mlp_act_buf[j] * w_down_l[j * cfg.dim + i];
                }
                x_buf[i] += sum;
            }
        }

        // 3. Final RMSNorm
        rmsnorm(x_norm_buf.data(), x_buf.data(), rms_final.data(), cfg.dim);

        // 4. LM Head projection (NPU offload or CPU fallback)
        if (npu_engine.available) {
            npu_engine.forward(x_norm_buf.data(), logits_buf.data());
        } else {
            #pragma omp parallel for
            for (int i = 0; i < cfg.vocab_size; ++i) {
                const float *w_row = embed.data() + i * cfg.dim;
                float sum = 0.0f;
                for (int j = 0; j < cfg.dim; ++j) {
                    sum += x_norm_buf[j] * w_row[j];
                }
                logits_buf[i] = sum;
            }
        }

        return logits_buf.data();
    }
};

// -----------------------------------------------------------------------------
// 5. Sampler with Repetition Penalty & Temperature (Fix for Issue #47)
// -----------------------------------------------------------------------------
int sample_token(float *logits, int vocab_size, const std::vector<int> &output_ids,
                 float temperature = 0.7f, int top_k = 40, float top_p = 0.9f,
                 float repeat_penalty = 1.15f, int repeat_window = 64)
{
    // Apply sliding-window repetition penalty
    if (repeat_penalty != 1.0f && !output_ids.empty()) {
        int start_idx = std::max(0, (int)output_ids.size() - repeat_window);
        std::vector<bool> seen(vocab_size, false);
        for (size_t i = start_idx; i < output_ids.size(); ++i) {
            int id = output_ids[i];
            if (!seen[id]) {
                seen[id] = true;
                if (logits[id] > 0.0f) logits[id] /= repeat_penalty;
                else logits[id] *= repeat_penalty;
            }
        }
    }

    // Greedy sampling if temperature <= 0
    if (temperature <= 0.0f) {
        int best_id = 0;
        float max_l = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_l) {
                max_l = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    // Temperature scale
    for (int i = 0; i < vocab_size; ++i) logits[i] /= temperature;

    // Top-K selection
    std::vector<std::pair<float, int>> token_probs(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        token_probs[i] = {logits[i], i};
    }
    if (top_k > 0 && top_k < vocab_size) {
        std::partial_sort(token_probs.begin(), token_probs.begin() + top_k, token_probs.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });
        token_probs.resize(top_k);
    } else {
        std::sort(token_probs.begin(), token_probs.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });
    }

    // Softmax
    float max_logit = token_probs[0].first;
    float sum_exp = 0.0f;
    for (auto &tp : token_probs) {
        tp.first = std::exp(tp.first - max_logit);
        sum_exp += tp.first;
    }
    for (auto &tp : token_probs) {
        tp.first /= sum_exp;
    }

    // Top-P (nucleus) cutoff
    float cumulative = 0.0f;
    size_t cutoff = token_probs.size();
    for (size_t i = 0; i < token_probs.size(); ++i) {
        cumulative += token_probs[i].first;
        if (cumulative >= top_p) {
            cutoff = i + 1;
            break;
        }
    }
    token_probs.resize(cutoff);

    // Renormalize
    float re_sum = 0.0f;
    for (const auto &tp : token_probs) re_sum += tp.first;

    // Sample from distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, re_sum);
    float r = dist(gen);

    float acc = 0.0f;
    for (const auto &tp : token_probs) {
        acc += tp.first;
        if (r <= acc) return tp.second;
    }
    return token_probs.back().second;
}

// -----------------------------------------------------------------------------
// 6. Interactive Chat CLI
// -----------------------------------------------------------------------------
void print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --model <path>           Path to binary model weights (default: models/smollm_135m.bin)\n"
              << "  --tokenizer <path>       Path to binary tokenizer (default: models/tokenizer.bin)\n"
              << "  --cpu                    Force CPU backend (disable NPU offloading)\n"
              << "  --temp <float>           Sampling temperature (default: 0.7, 0 = greedy)\n"
              << "  --top-k <int>            Top-K vocabulary filtering (default: 40)\n"
              << "  --top-p <float>          Top-P nucleus filtering (default: 0.9)\n"
              << "  --repeat-penalty <float> Repetition penalty (default: 1.15)\n"
              << "  --max-tokens <int>       Maximum tokens to generate per response (default: 256)\n"
              << "  --help                   Display this help message\n";
}

int main(int argc, char **argv) {
    std::string model_path = "models/smollm_135m.bin";
    std::string tokenizer_path = "models/tokenizer.bin";
    std::string d0_path = "models/smollm_lm_head_d0_npu_b1.model";
    std::string d1_path = "models/smollm_lm_head_d1_npu_b1.model";

    bool force_cpu = false;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    float repeat_penalty = 1.15f;
    int max_tokens = 256;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--cpu") force_cpu = true;
        else if (arg == "--temp" && i + 1 < argc) temperature = std::stof(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) top_k = std::stoi(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc) top_p = std::stof(argv[++i]);
        else if (arg == "--repeat-penalty" && i + 1 < argc) repeat_penalty = std::stof(argv[++i]);
        else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "\033[1;36m========================================================\033[0m\n";
    std::cout << "\033[1;36m  SmolLM-135M Interactive Chat Engine (Dual-NPU / CPU)  \033[0m\n";
    std::cout << "\033[1;36m  Hardware: ESWIN EIC7700X / RISC-V 64-bit & x86_64     \033[0m\n";
    std::cout << "\033[1;36m========================================================\033[0m\n\n";

    SmolLMTokenizer tokenizer;
    std::cout << "[*] Loading tokenizer from " << tokenizer_path << "...\n";
    if (!tokenizer.load(tokenizer_path)) {
        std::cerr << "[-] Failed to load tokenizer from " << tokenizer_path << std::endl;
        return 1;
    }
    std::cout << "[+] Loaded vocabulary with " << tokenizer.vocab_size << " tokens.\n";

    SmolLMModel model;
    std::cout << "[*] Loading model weights from " << model_path << "...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!model.load_weights(model_path)) {
        std::cerr << "[-] Failed to load weights from " << model_path << std::endl;
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[+] Loaded 30 Transformer layers in " << load_s << "s.\n";

    // Initialize NPU if requested and available
    if (!force_cpu) {
        std::cout << "[*] Probing ESWIN Dual-Die NPU acceleration...\n";
        if (model.npu_engine.init(d0_path, d1_path)) {
            std::cout << "\033[1;32m[+] Dual-Die NPU Acceleration Active (/dev/npu0 + /dev/npu1)\033[0m\n";
        } else {
            std::cout << "[!] NPU driver not detected or single-die only. Using high-performance CPU backend.\n";
        }
    } else {
        std::cout << "[*] CPU mode forced by user.\n";
    }

    std::cout << "\n\033[1;32m[+] SmolLM Chat is ready! Type your message, or 'exit'/'quit' to leave.\033[0m\n\n";

    std::string prompt;
    while (true) {
        std::cout << "\033[1;32m[YOU]:\033[0m ";
        if (!std::getline(std::cin, prompt)) break;
        if (prompt.empty()) continue;
        if (prompt == "exit" || prompt == "quit") break;

        std::vector<int> input_ids = tokenizer.encode(prompt);
        if (input_ids.empty()) {
            std::cout << "[SmolLM]: (empty prompt)\n\n";
            continue;
        }

        std::cout << "\033[1;36m[SmolLM]:\033[0m " << std::flush;

        model.reset_kv_cache();
        std::vector<int> output_ids;
        int pos = 0;

        // Prefill prompt tokens
        for (size_t i = 0; i + 1 < input_ids.size(); ++i) {
            model.forward_token(input_ids[i], pos++);
        }

        int next_token = input_ids.back();
        int tokens_gen = 0;

        auto gen_start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < max_tokens; ++step) {
            float *logits = model.forward_token(next_token, pos++);

            next_token = sample_token(logits, model.cfg.vocab_size, output_ids,
                                      temperature, top_k, top_p, repeat_penalty, 64);

            // EOS tokens: 0 (<|endoftext|>), 2 (<|im_end|>)
            if (next_token == 0 || next_token == 2) break;

            output_ids.push_back(next_token);
            std::cout << tokenizer.decode_token(next_token) << std::flush;
            tokens_gen++;

            // Loop break detector: 4 consecutive duplicate tokens (Issue #47 mitigation)
            if (output_ids.size() >= 5) {
                bool loop = true;
                for (size_t k = output_ids.size() - 4; k < output_ids.size(); ++k) {
                    if (output_ids[k] != output_ids.back()) {
                        loop = false;
                        break;
                    }
                }
                if (loop) break;
            }
        }

        auto gen_end = std::chrono::high_resolution_clock::now();
        double gen_s = std::chrono::duration<double>(gen_end - gen_start).count();
        double speed = (gen_s > 0) ? (tokens_gen / gen_s) : 0.0;

        std::cout << "\n\033[0;90m(" << tokens_gen << " tokens generated in " << gen_s << "s, "
                  << speed << " tokens/s)\033[0m\n\n";
    }

    return 0;
}
