/*
 * es_runtime.h - Open ESWIN NPU Runtime Public C11 API
 *
 * Provides a modern, ergonomic, linkable C API for accelerating deep learning
 * and LLM inference across the EIC7700X dual-die NPU and Cadence Vision Q7 DSP cores.
 *
 * Strictly C11 compliant with zero C++ syntax or dependencies.
 */

#ifndef ES_RUNTIME_H
#define ES_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * 1. Status & Error Codes
 * ============================================================================= */
typedef enum {
    ES_SUCCESS               =  0,  /* Operation succeeded */
    ES_ERR_INVALID_ARG       = -1,  /* Invalid argument or null pointer */
    ES_ERR_NO_MEMORY         = -2,  /* Out of host or contiguous DMA memory */
    ES_ERR_DEVICE_BUSY       = -3,  /* NPU hardware engine or stream busy */
    ES_ERR_HARDWARE_FAULT    = -4,  /* NPU kernel driver or hardware timeout */
    ES_ERR_NOT_INITIALIZED   = -5,  /* Subsystem or context not initialized */
    ES_ERR_NOT_SUPPORTED     = -6,  /* Requested feature or format unsupported */
    ES_ERR_IO                = -7,  /* File I/O or filesystem failure */
    ES_ERR_CHECKSUM_MISMATCH = -8,  /* Model artifact SHA-256 verification failed */
    ES_ERR_TIMEOUT           = -9   /* Task execution exceeded deadline */
} es_status_t;

/* Returns a constant human-readable description of an es_status_t code */
const char* es_status_to_string(es_status_t status);

/* =============================================================================
 * 2. Hardware Device & Cache Modes
 * ============================================================================= */
typedef enum {
    ES_DEVICE_DIE0           = 0,   /* Target Die 0 (/dev/npu0, NUMA node 0) */
    ES_DEVICE_DIE1           = 1,   /* Target Die 1 (/dev/npu1, NUMA node 1) */
    ES_DEVICE_DUAL_DIE_AUTO  = 2    /* Automatic concurrent dual-die execution */
} es_device_mode_t;

typedef enum {
    ES_MEM_NOCACHE           = 0,   /* Direct DMA memory (no CPU caching) */
    ES_MEM_CACHED            = 1    /* Cached memory (requires manual flush before DMA) */
} es_cache_mode_t;

/* =============================================================================
 * 3. Opaque Handles
 * ============================================================================= */
typedef struct es_context_t   es_context_t;   /* Primary runtime context */
typedef struct es_dma_buf_t   es_dma_buf_t;   /* Contiguous DMA buffer */
typedef struct es_model_t     es_model_t;     /* Loaded NPU model */
typedef struct es_session_t   es_session_t;   /* Active inference session */
typedef struct es_tokenizer_t es_tokenizer_t; /* Tokenizer instance */
typedef struct es_sampler_t   es_sampler_t;   /* Token sampling engine */

/* =============================================================================
 * 4. Runtime Context Management
 * ============================================================================= */

/**
 * @brief Initializes the ESWIN runtime subsystem.
 *
 * Probes hardware drivers (libedla_runtime.so, libes_memory.so), verifies device
 * count, initializes memory zones, and creates the primary runtime context.
 *
 * @param[out] ctx Pointer to receive the created context handle.
 * @return ES_SUCCESS on success, or appropriate error code.
 */
es_status_t es_init(es_context_t **ctx);

/**
 * @brief Releases all resources associated with the runtime context.
 *
 * @param ctx Context handle to release (safe if NULL).
 */
void es_free(es_context_t *ctx);

/**
 * @brief Queries the number of operational NPU dies present in the system.
 *
 * @param ctx Valid runtime context.
 * @param[out] num_devices Pointer to receive device count (typically 2 on EIC7700X).
 * @return ES_SUCCESS on success.
 */
es_status_t es_get_device_count(es_context_t *ctx, uint16_t *num_devices);

/* =============================================================================
 * 5. Contiguous DMA Memory Management
 * ============================================================================= */

/**
 * @brief Allocates a physically contiguous DMA buffer suitable for zero-copy NPU execution.
 *
 * @param ctx Valid runtime context.
 * @param size Buffer size in bytes.
 * @param cache_mode Caching behavior (ES_MEM_NOCACHE or ES_MEM_CACHED).
 * @param[out] buf Pointer to receive the allocated DMA buffer handle.
 * @return ES_SUCCESS on success, or ES_ERR_NO_MEMORY if allocation fails.
 */
es_status_t es_dma_alloc(es_context_t *ctx, size_t size, es_cache_mode_t cache_mode, es_dma_buf_t **buf);

/**
 * @brief Frees a contiguous DMA buffer and unmaps its memory.
 *
 * @param buf DMA buffer handle to free (safe if NULL).
 */
void es_dma_free(es_dma_buf_t *buf);

/**
 * @brief Obtains the userspace virtual address pointer to a mapped DMA buffer.
 *
 * @param buf Valid DMA buffer handle.
 * @return Pointer to accessible host memory, or NULL on error.
 */
void* es_dma_get_host_ptr(es_dma_buf_t *buf);

/**
 * @brief Obtains the physical memory file descriptor of a DMA buffer.
 *
 * @param buf Valid DMA buffer handle.
 * @return File descriptor (memFd) used for driver ioctls.
 */
uint64_t es_dma_get_fd(es_dma_buf_t *buf);

/**
 * @brief Obtains the capacity of the allocated DMA buffer in bytes.
 *
 * @param buf Valid DMA buffer handle.
 * @return Buffer capacity in bytes.
 */
size_t es_dma_get_size(es_dma_buf_t *buf);

/**
 * @brief Flushes CPU data cache to memory for cached DMA buffers.
 *
 * Must be invoked prior to dispatching NPU hardware tasks if ES_MEM_CACHED was used.
 *
 * @param buf Valid DMA buffer handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_dma_flush_cache(es_dma_buf_t *buf);

// =============================================================================
// Multi-Die Process Isolation & Worker Pool APIs (Milestone 2)
// =============================================================================

typedef struct es_worker_pool_t es_worker_pool_t;

/**
 * @brief Creates a multi-process isolated worker pool pinned to NUMA nodes.
 *
 * Spawns isolated worker daemons for Die 0 (cores 0-3) and Die 1 (cores 4-7)
 * to circumvent the vendor single-process driver limitation (0xA00F6046).
 *
 * @param ctx Valid runtime context.
 * @param num_dies Number of NPU dies to manage (1 or 2).
 * @param[out] pool Pointer to receive the allocated worker pool handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_worker_pool_create(es_context_t *ctx, uint16_t num_dies, es_worker_pool_t **pool);

/**
 * @brief Destroys the worker pool and terminates all worker daemon processes.
 *
 * @param pool Valid worker pool handle (safe if NULL).
 */
void es_worker_pool_destroy(es_worker_pool_t *pool);

/**
 * @brief Loads a compiled .model into the target die's worker process.
 *
 * @param pool Valid worker pool handle.
 * @param die_id Target NPU die ID (0 or 1).
 * @param model_path Path to the .model file on the filesystem.
 * @param[out] model_id Assigned model identifier on that die.
 * @return ES_SUCCESS on success.
 */
es_status_t es_worker_pool_load_model(es_worker_pool_t *pool, uint16_t die_id, const char *model_path, uint32_t *model_id);

/**
 * @brief Unloads a model from the target die's worker process.
 *
 * @param pool Valid worker pool handle.
 * @param die_id Target NPU die ID (0 or 1).
 * @param model_id Model identifier to unload.
 * @return ES_SUCCESS on success.
 */
es_status_t es_worker_pool_unload_model(es_worker_pool_t *pool, uint16_t die_id, uint32_t model_id);

/**
 * @brief Prepares and maps a DMA buffer into the target die's SMMU page table.
 *
 * @param pool Valid worker pool handle.
 * @param die_id Target NPU die ID (0 or 1).
 * @param buf Valid DMA buffer handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_worker_pool_prepare_buffer(es_worker_pool_t *pool, uint16_t die_id, es_dma_buf_t *buf);

/**
 * @brief Submits a synchronous inference task to the target die worker process.
 *
 * @param pool Valid worker pool handle.
 * @param die_id Target NPU die ID (0 or 1).
 * @param model_id Loaded model ID.
 * @param input_buf Input DMA buffer.
 * @param output_buf Output DMA buffer.
 * @param[out] elapsed_ns Optional pointer to receive hardware latency in nanoseconds.
 * @return ES_SUCCESS on success.
 */
es_status_t es_worker_pool_submit_task(
    es_worker_pool_t *pool,
    uint16_t die_id,
    uint32_t model_id,
    es_dma_buf_t *input_buf,
    es_dma_buf_t *output_buf,
    uint64_t *elapsed_ns
);

// =============================================================================
// NLP Tokenizer & Sampler APIs (Milestone 4)
// =============================================================================

typedef struct {
    float    temperature;
    int32_t  top_k;
    float    top_p;
    float    min_p;
    float    repeat_penalty;
    int32_t  repeat_window;
    int32_t  max_consecutive_duplicates;
    uint32_t seed;
} es_sampler_params_t;

es_status_t es_tokenizer_load(const char *filepath, es_tokenizer_t **out_tokenizer);
void es_tokenizer_free(es_tokenizer_t *tokenizer);
int32_t es_tokenizer_encode(const es_tokenizer_t *tokenizer, const char *text, int32_t *out_tokens, size_t max_tokens);
int32_t es_tokenizer_decode(const es_tokenizer_t *tokenizer, const int32_t *tokens, size_t num_tokens, char *out_text, size_t max_len);

es_status_t es_sampler_create(const es_sampler_params_t *params, es_sampler_t **out_sampler);
void es_sampler_free(es_sampler_t *sampler);
int32_t es_sampler_sample(es_sampler_t *sampler, float *logits, size_t vocab_size);
void es_sampler_accept(es_sampler_t *sampler, int32_t token_id);
void es_sampler_reset(es_sampler_t *sampler);

// =============================================================================
// High-Level Model & Session APIs (Milestone 6)
// =============================================================================

/**
 * @brief Returns the runtime version string.
 */
const char* es_version(void);

typedef struct {
    const char       *model_path;     /* Directory containing manifest.json or path to .model */
    const char       *tokenizer_path; /* Path to tokenizer.json / vocab file (optional if in model dir) */
    const char       *cache_dir;      /* Model cache directory (default ~/.cache/eswin/models if NULL) */
    es_device_mode_t  device_mode;    /* Target device mode (default ES_DEVICE_DUAL_DIE_AUTO) */
} es_model_config_t;

typedef struct {
    char     name[128];
    char     architecture[64];
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t max_seq_len;
} es_model_info_t;

/**
 * @brief Loads an NPU model package and prepares execution resources.
 *
 * @param ctx Valid runtime context.
 * @param config Model configuration structure.
 * @param[out] out_model Pointer to receive the allocated model handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_model_load(es_context_t *ctx, const es_model_config_t *config, es_model_t **out_model);

/**
 * @brief Unloads a model and releases all associated resources.
 *
 * @param model Model handle to unload (safe if NULL).
 */
void es_model_unload(es_model_t *model);

/**
 * @brief Queries metadata and structural parameters of a loaded model.
 *
 * @param model Valid model handle.
 * @param[out] info Pointer to receive model information.
 * @return ES_SUCCESS on success.
 */
es_status_t es_model_get_info(const es_model_t *model, es_model_info_t *info);

/**
 * @brief Obtains a reference to the model's internal tokenizer.
 *
 * @param model Valid model handle.
 * @return Pointer to tokenizer handle, or NULL if unavailable.
 */
const es_tokenizer_t* es_model_get_tokenizer(const es_model_t *model);

typedef struct {
    size_t              max_seq_len;     /* Maximum sequence context window (default: 2048) */
    int32_t             max_new_tokens;  /* Maximum tokens to generate per turn (default: 512) */
    bool                enable_dual_die; /* Enable dual-die parallel execution */
    es_sampler_params_t sampler_params;  /* Sampling parameters */
} es_session_config_t;

/**
 * @brief Token generation callback.
 *
 * Invoked during autoregressive generation as each token piece is produced.
 *
 * @param token The sampled token ID.
 * @param piece The decoded UTF-8 text piece for the token.
 * @param user_data User-provided context pointer.
 * @return true to continue generation, or false to halt/cancel early.
 */
typedef bool (*es_token_callback_t)(int32_t token, const char *piece, void *user_data);

typedef struct {
    int32_t prompt_tokens;
    int32_t eval_tokens;
    double  prompt_eval_ms;
    double  eval_ms;
    double  tokens_per_sec;
} es_perf_stats_t;

/**
 * @brief Creates an autoregressive inference session for a loaded model.
 *
 * Allocates KV-cache, working scratch buffers, and initializes samplers.
 *
 * @param model Valid loaded model handle.
 * @param config Session configuration (or NULL for defaults).
 * @param[out] out_session Pointer to receive allocated session handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_session_create(es_model_t *model, const es_session_config_t *config, es_session_t **out_session);

/**
 * @brief Frees an inference session and its KV-cache.
 *
 * @param session Session handle to free (safe if NULL).
 */
void es_session_free(es_session_t *session);

/**
 * @brief Resets sequence position and clears KV-cache for a new conversation turn.
 *
 * @param session Valid session handle.
 * @return ES_SUCCESS on success.
 */
es_status_t es_session_reset(es_session_t *session);

/**
 * @brief Executes prompt prefill and autoregressive token generation.
 *
 * Streams generated tokens sequentially to the provided callback.
 *
 * @param session Valid session handle.
 * @param prompt Input text prompt to encode and evaluate.
 * @param callback Callback invoked for each generated token piece (optional).
 * @param user_data Context pointer passed to the callback.
 * @param[out] stats Optional pointer to receive performance telemetry.
 * @return ES_SUCCESS on success.
 */
es_status_t es_session_generate(
    es_session_t *session,
    const char *prompt,
    es_token_callback_t callback,
    void *user_data,
    es_perf_stats_t *stats
);

#ifdef __cplusplus
}
#endif

#endif /* ES_RUNTIME_H */
