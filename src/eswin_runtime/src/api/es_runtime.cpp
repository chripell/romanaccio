/**
 * @file es_runtime.cpp
 * @brief Implementation of Public C11 API for eswin_runtime.
 */

#include "es_runtime.h"
#include "../backend/driver_wrapper.hpp"
#include "../backend/dma_buffer.hpp"

#include <memory>
#include <new>
#include <cstring>

// =============================================================================
// Internal Context & Handle Implementations
// =============================================================================

struct es_context_t {
    eswin::EswinDriver &driver;
    uint16_t num_devices;

    es_context_t(eswin::EswinDriver &d, uint16_t count)
        : driver(d), num_devices(count) {}
};

struct es_dma_buf_t {
    std::unique_ptr<eswin::DmaBuffer> buffer;

    explicit es_dma_buf_t(std::unique_ptr<eswin::DmaBuffer> b)
        : buffer(std::move(b)) {}
};

// =============================================================================
// Status Helper
// =============================================================================

const char* es_status_to_string(es_status_t status) {
    switch (status) {
        case ES_SUCCESS:               return "ES_SUCCESS: Operation completed successfully";
        case ES_ERR_INVALID_ARG:       return "ES_ERR_INVALID_ARG: Invalid argument or null pointer";
        case ES_ERR_NO_MEMORY:         return "ES_ERR_NO_MEMORY: Out of host or DMA memory";
        case ES_ERR_DEVICE_BUSY:       return "ES_ERR_DEVICE_BUSY: Hardware device or engine busy";
        case ES_ERR_HARDWARE_FAULT:    return "ES_ERR_HARDWARE_FAULT: Hardware fault or driver failure";
        case ES_ERR_NOT_INITIALIZED:   return "ES_ERR_NOT_INITIALIZED: Runtime subsystem not initialized";
        case ES_ERR_NOT_SUPPORTED:     return "ES_ERR_NOT_SUPPORTED: Requested operation is not supported";
        case ES_ERR_IO:                return "ES_ERR_IO: File system or I/O failure";
        case ES_ERR_CHECKSUM_MISMATCH: return "ES_ERR_CHECKSUM_MISMATCH: SHA-256 verification mismatch";
        case ES_ERR_TIMEOUT:           return "ES_ERR_TIMEOUT: Operation exceeded execution deadline";
        default:                       return "ES_ERR_UNKNOWN: Unknown error code";
    }
}

// =============================================================================
// Context Management
// =============================================================================

es_status_t es_init(es_context_t **ctx) {
    if (!ctx) return ES_ERR_INVALID_ARG;

    auto &driver = eswin::EswinDriver::instance();
    driver.initialize();

    uint16_t devices = driver.get_num_devices();
    auto new_ctx = std::make_unique<es_context_t>(driver, devices);
    *ctx = new_ctx.release();

    return ES_SUCCESS;
}

void es_free(es_context_t *ctx) {
    if (ctx) {
        delete ctx;
    }
}

es_status_t es_get_device_count(es_context_t *ctx, uint16_t *num_devices) {
    if (!ctx || !num_devices) return ES_ERR_INVALID_ARG;
    *num_devices = ctx->num_devices;
    return ES_SUCCESS;
}

// =============================================================================
// DMA Memory Management
// =============================================================================

es_status_t es_dma_alloc(
    es_context_t *ctx,
    size_t size,
    es_cache_mode_t cache_mode,
    es_dma_buf_t **buf
) {
    if (!ctx || !buf || size == 0) return ES_ERR_INVALID_ARG;

    try {
        eswin::CacheMode mode = (cache_mode == ES_MEM_CACHED)
            ? eswin::CacheMode::Cached
            : eswin::CacheMode::NoCache;

        auto dma = std::make_unique<eswin::DmaBuffer>(size, mode);
        if (!dma->is_valid()) {
            return ES_ERR_NO_MEMORY;
        }

        auto wrapper = std::make_unique<es_dma_buf_t>(std::move(dma));
        *buf = wrapper.release();
        return ES_SUCCESS;
    } catch (const std::bad_alloc&) {
        return ES_ERR_NO_MEMORY;
    } catch (...) {
        return ES_ERR_HARDWARE_FAULT;
    }
}

void es_dma_free(es_dma_buf_t *buf) {
    if (buf) {
        delete buf;
    }
}

void* es_dma_get_host_ptr(es_dma_buf_t *buf) {
    if (!buf || !buf->buffer) return nullptr;
    return buf->buffer->data();
}

uint64_t es_dma_get_fd(es_dma_buf_t *buf) {
    if (!buf || !buf->buffer) return 0;
    return buf->buffer->mem_fd();
}

size_t es_dma_get_size(es_dma_buf_t *buf) {
    if (!buf || !buf->buffer) return 0;
    return buf->buffer->size();
}

es_status_t es_dma_flush_cache(es_dma_buf_t *buf) {
    if (!buf || !buf->buffer) return ES_ERR_INVALID_ARG;
    return buf->buffer->flush_cache() ? ES_SUCCESS : ES_ERR_HARDWARE_FAULT;
}

// =============================================================================
// Worker Pool C API Implementation
// =============================================================================

#include "../backend/worker_pool.hpp"

struct es_worker_pool_t {
    std::unique_ptr<eswin::WorkerPool> pool;

    explicit es_worker_pool_t(std::unique_ptr<eswin::WorkerPool> p)
        : pool(std::move(p)) {}
};

es_status_t es_worker_pool_create(es_context_t *ctx, uint16_t num_dies, es_worker_pool_t **pool) {
    if (!ctx || !pool) return ES_ERR_INVALID_ARG;

    auto p = std::make_unique<eswin::WorkerPool>();
    if (!p->start(num_dies)) {
        return ES_ERR_HARDWARE_FAULT;
    }

    auto wrapper = std::make_unique<es_worker_pool_t>(std::move(p));
    *pool = wrapper.release();
    return ES_SUCCESS;
}

void es_worker_pool_destroy(es_worker_pool_t *pool) {
    if (pool) {
        delete pool;
    }
}

es_status_t es_worker_pool_load_model(es_worker_pool_t *pool, uint16_t die_id, const char *model_path, uint32_t *model_id) {
    if (!pool || !pool->pool || !model_path || !model_id) return ES_ERR_INVALID_ARG;
    return pool->pool->load_model(die_id, model_path, *model_id) ? ES_SUCCESS : ES_ERR_HARDWARE_FAULT;
}

es_status_t es_worker_pool_unload_model(es_worker_pool_t *pool, uint16_t die_id, uint32_t model_id) {
    if (!pool || !pool->pool) return ES_ERR_INVALID_ARG;
    return pool->pool->unload_model(die_id, model_id) ? ES_SUCCESS : ES_ERR_HARDWARE_FAULT;
}

es_status_t es_worker_pool_prepare_buffer(es_worker_pool_t *pool, uint16_t die_id, es_dma_buf_t *buf) {
    if (!pool || !pool->pool || !buf || !buf->buffer) return ES_ERR_INVALID_ARG;
    return pool->pool->prepare_buffer(die_id, *buf->buffer) ? ES_SUCCESS : ES_ERR_HARDWARE_FAULT;
}

es_status_t es_worker_pool_submit_task(
    es_worker_pool_t *pool,
    uint16_t die_id,
    uint32_t model_id,
    es_dma_buf_t *input_buf,
    es_dma_buf_t *output_buf,
    uint64_t *elapsed_ns
) {
    if (!pool || !pool->pool || !input_buf || !input_buf->buffer || !output_buf || !output_buf->buffer) {
        return ES_ERR_INVALID_ARG;
    }

    std::vector<eswin::DmaBuffer*> inputs = { input_buf->buffer.get() };
    std::vector<eswin::DmaBuffer*> outputs = { output_buf->buffer.get() };

    return pool->pool->submit_task(die_id, model_id, inputs, outputs, elapsed_ns)
        ? ES_SUCCESS : ES_ERR_HARDWARE_FAULT;
}

// =============================================================================
// NLP Tokenizer & Sampler Implementations
// =============================================================================

#include "../nlp/tokenizer.hpp"
#include "../nlp/sampler.hpp"

struct es_tokenizer_t {
    std::unique_ptr<eswin::Tokenizer> tokenizer;
};

struct es_sampler_t {
    std::unique_ptr<eswin::Sampler> sampler;
};

es_status_t es_tokenizer_load(const char *filepath, es_tokenizer_t **out_tokenizer) {
    if (!filepath || !out_tokenizer) return ES_ERR_INVALID_ARG;
    auto tok = std::make_unique<eswin::Tokenizer>();
    std::string err;
    if (!tok->load(filepath, err)) {
        return ES_ERR_HARDWARE_FAULT;
    }
    auto handle = new es_tokenizer_t();
    handle->tokenizer = std::move(tok);
    *out_tokenizer = handle;
    return ES_SUCCESS;
}

void es_tokenizer_free(es_tokenizer_t *tokenizer) {
    if (tokenizer) {
        delete tokenizer;
    }
}

size_t es_tokenizer_vocab_size(const es_tokenizer_t *tokenizer) {
    if (!tokenizer || !tokenizer->tokenizer) return 0;
    return tokenizer->tokenizer->vocab_size();
}

int32_t es_tokenizer_encode(const es_tokenizer_t *tokenizer, const char *text, int32_t *out_tokens, size_t max_tokens) {
    if (!tokenizer || !tokenizer->tokenizer || !text || !out_tokens || max_tokens == 0) return -1;
    auto tokens = tokenizer->tokenizer->encode(text);
    size_t count = std::min(max_tokens, tokens.size());
    for (size_t i = 0; i < count; ++i) {
        out_tokens[i] = tokens[i];
    }
    return static_cast<int32_t>(count);
}

int32_t es_tokenizer_decode(const es_tokenizer_t *tokenizer, const int32_t *tokens, size_t num_tokens, char *out_text, size_t max_len) {
    if (!tokenizer || !tokenizer->tokenizer || !tokens || !out_text || max_len == 0) return -1;
    std::vector<int32_t> tok_vec(tokens, tokens + num_tokens);
    std::string text = tokenizer->tokenizer->decode(tok_vec);
    size_t copy_len = std::min(max_len - 1, text.size());
    std::memcpy(out_text, text.data(), copy_len);
    out_text[copy_len] = '\0';
    return static_cast<int32_t>(copy_len);
}

es_status_t es_sampler_create(const es_sampler_params_t *params, es_sampler_t **out_sampler) {
    if (!out_sampler) return ES_ERR_INVALID_ARG;
    eswin::SamplerParams p;
    if (params) {
        p.temperature = params->temperature;
        p.top_k = params->top_k;
        p.top_p = params->top_p;
        p.min_p = params->min_p;
        p.repeat_penalty = params->repeat_penalty;
        p.repeat_window = params->repeat_window;
        p.max_consecutive_duplicates = params->max_consecutive_duplicates;
        p.seed = params->seed;
    }
    auto handle = new es_sampler_t();
    handle->sampler = std::make_unique<eswin::Sampler>(p);
    *out_sampler = handle;
    return ES_SUCCESS;
}

void es_sampler_free(es_sampler_t *sampler) {
    if (sampler) {
        delete sampler;
    }
}

int32_t es_sampler_sample(es_sampler_t *sampler, float *logits, size_t vocab_size) {
    if (!sampler || !sampler->sampler || !logits || vocab_size == 0) return -1;
    return sampler->sampler->sample(logits, vocab_size);
}

void es_sampler_accept(es_sampler_t *sampler, int32_t token_id) {
    if (sampler && sampler->sampler) {
        sampler->sampler->accept(token_id);
    }
}

void es_sampler_reset(es_sampler_t *sampler) {
    if (sampler && sampler->sampler) {
        sampler->sampler->reset();
    }
}

// =============================================================================
// High-Level Model & Session APIs (Milestone 6)
// =============================================================================

#include "../model/model_manifest.hpp"
#include "../model/downloader.hpp"
#include "../llm/transformer_session.hpp"
#include "../llm/kv_cache.hpp"
#include <sys/stat.h>

const char* es_version(void) {
    return "1.0.0-open";
}

struct es_model_t {
    eswin::ModelManifest manifest;
    std::string model_dir;
    std::string tokenizer_path;
    es_device_mode_t device_mode = ES_DEVICE_DUAL_DIE_AUTO;
    std::unique_ptr<es_tokenizer_t> tokenizer_handle;
};

struct es_session_t {
    es_model_t *model = nullptr;
    es_session_config_t config{};
    std::unique_ptr<eswin::TransformerSession> session;
};

static bool file_exists(const std::string &path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
}

static bool dir_exists(const std::string &path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

es_status_t es_model_load(es_context_t *ctx, const es_model_config_t *config, es_model_t **out_model) {
    if (!ctx || !config || !config->model_path || !out_model) return ES_ERR_INVALID_ARG;

    std::string path = config->model_path;
    std::string model_dir;
    std::string manifest_path;

    if (dir_exists(path)) {
        model_dir = path;
        manifest_path = path + "/manifest.json";
    } else if (file_exists(path)) {
        if (path.length() >= 5 && path.substr(path.length() - 5) == ".json") {
            manifest_path = path;
            size_t slash = path.find_last_of('/');
            model_dir = (slash != std::string::npos) ? path.substr(0, slash) : ".";
        } else {
            size_t slash = path.find_last_of('/');
            model_dir = (slash != std::string::npos) ? path.substr(0, slash) : ".";
        }
    } else {
        std::string cache = config->cache_dir ? config->cache_dir : (std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.cache/eswin/models");
        std::string cached_manifest = cache + "/" + path + "/manifest.json";
        if (file_exists(cached_manifest)) {
            model_dir = cache + "/" + path;
            manifest_path = cached_manifest;
        } else {
            return ES_ERR_IO;
        }
    }

    auto model = std::make_unique<es_model_t>();
    model->model_dir = model_dir;
    model->device_mode = config->device_mode;

    if (!manifest_path.empty() && file_exists(manifest_path)) {
        std::string err;
        if (!model->manifest.load_from_file(manifest_path, err)) {
            return ES_ERR_IO;
        }
    } else {
        model->manifest.name = "Custom NPU Model";
        model->manifest.config.architecture = "Transformer";
        model->manifest.config.vocab_size = 49152;
        model->manifest.config.hidden_size = 576;
        model->manifest.config.num_hidden_layers = 30;
        model->manifest.config.num_attention_heads = 9;
        model->manifest.config.num_key_value_heads = 3;
    }

    if (config->tokenizer_path && *config->tokenizer_path) {
        model->tokenizer_path = config->tokenizer_path;
    } else {
        const auto *tok_desc = model->manifest.find_file_by_role("tokenizer");
        if (tok_desc && !tok_desc->name.empty() && file_exists(model_dir + "/" + tok_desc->name)) {
            model->tokenizer_path = model_dir + "/" + tok_desc->name;
        } else if (file_exists(model_dir + "/tokenizer.json")) {
            model->tokenizer_path = model_dir + "/tokenizer.json";
        } else if (file_exists(model_dir + "/qwen.tiktoken")) {
            model->tokenizer_path = model_dir + "/qwen.tiktoken";
        } else if (file_exists(model_dir + "/vocab.json")) {
            model->tokenizer_path = model_dir + "/vocab.json";
        } else if (file_exists(model_dir + "/tokenizer.model")) {
            model->tokenizer_path = model_dir + "/tokenizer.model";
        } else if (file_exists("./qwen.tiktoken")) {
            model->tokenizer_path = "./qwen.tiktoken";
        } else if (file_exists("../qwen2_fix/deepseek_7b_1k_int8_peer/qwen.tiktoken")) {
            model->tokenizer_path = "../qwen2_fix/deepseek_7b_1k_int8_peer/qwen.tiktoken";
        } else if (file_exists("/opt/eswin/sample-code/npu_sample/qwen_sample/src/qwen2_0_5b_1k_int8/qwen.tiktoken")) {
            model->tokenizer_path = "/opt/eswin/sample-code/npu_sample/qwen_sample/src/qwen2_0_5b_1k_int8/qwen.tiktoken";
        }
    }

    if (!model->tokenizer_path.empty()) {
        es_tokenizer_t *tok = nullptr;
        if (es_tokenizer_load(model->tokenizer_path.c_str(), &tok) == ES_SUCCESS) {
            model->tokenizer_handle.reset(tok);
        }
    }

    *out_model = model.release();
    return ES_SUCCESS;
}

void es_model_unload(es_model_t *model) {
    if (model) {
        delete model;
    }
}

es_status_t es_model_get_info(const es_model_t *model, es_model_info_t *info) {
    if (!model || !info) return ES_ERR_INVALID_ARG;
    std::memset(info, 0, sizeof(*info));
    std::strncpy(info->name, model->manifest.name.c_str(), sizeof(info->name) - 1);
    std::strncpy(info->architecture, model->manifest.config.architecture.c_str(), sizeof(info->architecture) - 1);
    info->vocab_size = model->manifest.config.vocab_size;
    info->hidden_size = model->manifest.config.hidden_size;
    info->num_layers = model->manifest.config.num_hidden_layers;
    info->num_heads = model->manifest.config.num_attention_heads;
    info->num_kv_heads = model->manifest.config.num_key_value_heads;
    info->max_seq_len = model->manifest.config.context_length;
    return ES_SUCCESS;
}

const es_tokenizer_t* es_model_get_tokenizer(const es_model_t *model) {
    if (!model) return nullptr;
    return model->tokenizer_handle.get();
}

es_status_t es_session_create(es_model_t *model, const es_session_config_t *config, es_session_t **out_session) {
    if (!model || !out_session) return ES_ERR_INVALID_ARG;

    auto sess_obj = std::make_unique<eswin::TransformerSession>();
    std::string err;
    if (!sess_obj->init(model->manifest, model->model_dir, model->tokenizer_path, err)) {
        return ES_ERR_HARDWARE_FAULT;
    }

    if (config) {
        eswin::SamplerParams sp;
        sp.temperature = config->sampler_params.temperature;
        sp.top_k = config->sampler_params.top_k;
        sp.top_p = config->sampler_params.top_p;
        sp.min_p = config->sampler_params.min_p;
        sp.repeat_penalty = config->sampler_params.repeat_penalty;
        sp.repeat_window = config->sampler_params.repeat_window;
        sp.max_consecutive_duplicates = config->sampler_params.max_consecutive_duplicates;
        sp.seed = config->sampler_params.seed;
        sess_obj->set_sampler_params(sp);
    }

    auto sess_wrapper = std::make_unique<es_session_t>();
    sess_wrapper->model = model;
    if (config) {
        sess_wrapper->config = *config;
    } else {
        sess_wrapper->config.max_seq_len = model->manifest.config.context_length;
        sess_wrapper->config.max_new_tokens = 512;
        sess_wrapper->config.enable_dual_die = true;
        sess_wrapper->config.sampler_params.temperature = 0.7f;
        sess_wrapper->config.sampler_params.top_k = 40;
        sess_wrapper->config.sampler_params.top_p = 0.9f;
        sess_wrapper->config.sampler_params.repeat_penalty = 1.15f;
    }
    sess_wrapper->session = std::move(sess_obj);
    *out_session = sess_wrapper.release();
    return ES_SUCCESS;
}

void es_session_free(es_session_t *session) {
    if (session) {
        delete session;
    }
}

es_status_t es_session_reset(es_session_t *session) {
    if (!session || !session->session) return ES_ERR_INVALID_ARG;
    session->session->reset();
    return ES_SUCCESS;
}

es_status_t es_session_generate(
    es_session_t *session,
    const char *prompt,
    es_token_callback_t callback,
    void *user_data,
    es_perf_stats_t *stats
) {
    if (!session || !session->session || !prompt) return ES_ERR_INVALID_ARG;

    eswin::SessionStats s_stats;
    int max_tokens = session->config.max_new_tokens > 0 ? session->config.max_new_tokens : 512;

    auto cb_wrapper = [callback, user_data](const std::string &piece, int32_t token_id) -> bool {
        if (!callback) return true;
        return callback(token_id, piece.c_str(), user_data);
    };

    bool ok = session->session->generate(prompt, max_tokens, cb_wrapper, &s_stats);
    if (!ok) return ES_ERR_HARDWARE_FAULT;

    if (stats) {
        stats->prompt_tokens = s_stats.prompt_tokens;
        stats->eval_tokens = s_stats.eval_tokens;
        stats->prompt_eval_ms = s_stats.prompt_eval_ms;
        stats->eval_ms = s_stats.eval_ms;
        stats->tokens_per_sec = s_stats.tokens_per_sec;
    }

    return ES_SUCCESS;
}
