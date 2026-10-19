/**
 * @file dma_buffer.hpp
 * @brief RAII Contiguous DMA Buffer Allocator and Memory Mapping Manager.
 *
 * Provides physical contiguous DMA memory management for zero-copy NPU tensor I/O
 * with automatic virtual address mapping, cache maintenance, and SMMU preparation.
 */

#ifndef ES_DMA_BUFFER_HPP
#define ES_DMA_BUFFER_HPP

#include "driver_wrapper.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace eswin {

class DmaBuffer {
public:
    /**
     * @brief Allocates a contiguous DMA buffer.
     *
     * @param size Required capacity in bytes (automatically aligned to page boundaries).
     * @param cache_mode CacheMode::NoCache for direct DMA, CacheMode::Cached for CPU caching.
     * @param zone Optional MMZ zone name ("mmz_nid_0_part_0" or nullptr for default).
     */
    explicit DmaBuffer(size_t size, CacheMode cache_mode = CacheMode::NoCache, const char *zone = nullptr);

    ~DmaBuffer();

    // Move semantics (strictly non-copyable)
    DmaBuffer(DmaBuffer &&other) noexcept;
    DmaBuffer& operator=(DmaBuffer &&other) noexcept;

    DmaBuffer(const DmaBuffer&) = delete;
    DmaBuffer& operator=(const DmaBuffer&) = delete;

    // Accessors
    bool is_valid() const noexcept { return m_vaddr != nullptr; }
    void* data() noexcept { return m_vaddr; }
    const void* data() const noexcept { return m_vaddr; }

    template<typename T>
    T* as() noexcept { return static_cast<T*>(m_vaddr); }

    template<typename T>
    const T* as() const noexcept { return static_cast<const T*>(m_vaddr); }

    size_t size() const noexcept { return m_size; }
    uint64_t mem_fd() const noexcept { return m_mem_fd; }
    CacheMode cache_mode() const noexcept { return m_cache_mode; }
    bool is_prepared() const noexcept { return m_is_prepared; }

    uint16_t prepared_device() const noexcept { return m_prepared_device; }

    const DevBuf& dev_buf() const noexcept { return m_dev_buf; }
    DevBuf* dev_buf_ptr() noexcept { return &m_dev_buf; }

    // Memory operations
    bool flush_cache();
    bool prepare(uint16_t device_id = 0);
    bool unprepare(uint16_t device_id = 0);

private:
    void release();

    uint64_t   m_mem_fd = 0;
    void      *m_vaddr = nullptr;
    size_t     m_size = 0;
    CacheMode  m_cache_mode = CacheMode::NoCache;
    bool       m_is_prepared = false;
    uint16_t   m_prepared_device = 0;
    bool       m_is_fallback = false;  // Fallback host memory when running off-board
    DevBuf     m_dev_buf{0, 0, 0, 0};
};

} // namespace eswin

#endif // ES_DMA_BUFFER_HPP
