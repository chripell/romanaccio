/**
 * @file es_runtime.cpp
 * @brief Implementation of Public C11 API for eswin_runtime.
 */

#include "es_runtime.h"
#include "../backend/driver_wrapper.hpp"
#include "../backend/dma_buffer.hpp"

#include <memory>
#include <new>

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
