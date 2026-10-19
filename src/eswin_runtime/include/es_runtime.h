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

#ifdef __cplusplus
}
#endif

#endif /* ES_RUNTIME_H */
