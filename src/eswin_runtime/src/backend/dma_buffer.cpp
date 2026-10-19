/**
 * @file dma_buffer.cpp
 * @brief Implementation of RAII Contiguous DMA Buffer Allocator.
 */

#include "dma_buffer.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace eswin {

static constexpr size_t PAGE_ALIGN = 4096;

static inline size_t align_to_page(size_t s) {
    return (s + PAGE_ALIGN - 1) & ~(PAGE_ALIGN - 1);
}

DmaBuffer::DmaBuffer(size_t size, CacheMode cache_mode, const char *zone)
    : m_size(align_to_page(size > 0 ? size : PAGE_ALIGN)),
      m_cache_mode(cache_mode)
{
    auto &driver = EswinDriver::instance();

    if (driver.is_hardware_available()) {
        int ret = driver.mem_alloc(&m_mem_fd, m_size, m_cache_mode, zone);
        if (ret == 0 && m_mem_fd != 0) {
            m_vaddr = driver.mem_mmap(m_mem_fd, m_size, m_cache_mode);
            if (m_vaddr) {
                m_dev_buf.mem_fd = m_mem_fd;
                m_dev_buf.offset = 0;
                m_dev_buf.size = m_size;
                m_dev_buf.reserved = 0;
                m_is_fallback = false;
                std::memset(m_vaddr, 0, m_size);
                return;
            }
            // Mmap failed, release memory
            driver.mem_free(m_mem_fd);
            m_mem_fd = 0;
        }
    }

    // Host development / fallback path
    void *ptr = nullptr;
    if (posix_memalign(&ptr, PAGE_ALIGN, m_size) == 0 && ptr != nullptr) {
        m_vaddr = ptr;
        m_mem_fd = 0xDEADBEEF; // Mock file descriptor
        m_dev_buf.mem_fd = m_mem_fd;
        m_dev_buf.offset = 0;
        m_dev_buf.size = m_size;
        m_dev_buf.reserved = 0;
        m_is_fallback = true;
        std::memset(m_vaddr, 0, m_size);
    } else {
        m_vaddr = nullptr;
        m_size = 0;
        throw std::bad_alloc();
    }
}

DmaBuffer::~DmaBuffer() {
    release();
}

void DmaBuffer::release() {
    if (m_is_prepared) {
        unprepare(m_prepared_device);
    }

    if (m_vaddr) {
        if (!m_is_fallback) {
            auto &driver = EswinDriver::instance();
            driver.mem_munmap(m_vaddr, m_size);
            if (m_mem_fd != 0) {
                driver.mem_free(m_mem_fd);
            }
        } else {
            std::free(m_vaddr);
        }
        m_vaddr = nullptr;
    }

    m_mem_fd = 0;
    m_size = 0;
    m_is_prepared = false;
    m_prepared_device = 0;
    m_is_fallback = false;
}

DmaBuffer::DmaBuffer(DmaBuffer &&other) noexcept
    : m_mem_fd(other.m_mem_fd),
      m_vaddr(other.m_vaddr),
      m_size(other.m_size),
      m_cache_mode(other.m_cache_mode),
      m_is_prepared(other.m_is_prepared),
      m_prepared_device(other.m_prepared_device),
      m_is_fallback(other.m_is_fallback),
      m_dev_buf(other.m_dev_buf)
{
    other.m_mem_fd = 0;
    other.m_vaddr = nullptr;
    other.m_size = 0;
    other.m_is_prepared = false;
    other.m_prepared_device = 0;
    other.m_is_fallback = false;
}

DmaBuffer& DmaBuffer::operator=(DmaBuffer &&other) noexcept {
    if (this != &other) {
        release();

        m_mem_fd = other.m_mem_fd;
        m_vaddr = other.m_vaddr;
        m_size = other.m_size;
        m_cache_mode = other.m_cache_mode;
        m_is_prepared = other.m_is_prepared;
        m_prepared_device = other.m_prepared_device;
        m_is_fallback = other.m_is_fallback;
        m_dev_buf = other.m_dev_buf;

        other.m_mem_fd = 0;
        other.m_vaddr = nullptr;
        other.m_size = 0;
        other.m_is_prepared = false;
        other.m_prepared_device = 0;
        other.m_is_fallback = false;
    }
    return *this;
}

bool DmaBuffer::flush_cache() {
    if (!m_vaddr) return false;
    if (m_is_fallback) return true; // Host memory is already coherent

    if (m_cache_mode == CacheMode::Cached) {
        return (EswinDriver::instance().mem_flush_cache(m_mem_fd) == 0);
    }
    return true;
}

bool DmaBuffer::prepare(uint16_t device_id) {
    if (!m_vaddr || m_is_prepared) return m_is_prepared;
    if (m_is_fallback) {
        m_is_prepared = true;
        m_prepared_device = device_id;
        return true;
    }

    auto &driver = EswinDriver::instance();
    // ES_NPU_Prepare requires an active device context set on the thread
    driver.set_device(device_id);

    int ret = driver.prepare_dev_buf(&m_dev_buf);
    if (ret == 0) {
        m_is_prepared = true;
        m_prepared_device = device_id;
        return true;
    }
    std::cerr << "DmaBuffer::prepare failed on device " << device_id 
              << ": " << driver.decode_error(ret) << std::endl;
    return false;
}

bool DmaBuffer::unprepare(uint16_t device_id) {
    if (!m_is_prepared) return true;
    if (m_is_fallback) {
        m_is_prepared = false;
        m_prepared_device = 0;
        return true;
    }

    auto &driver = EswinDriver::instance();
    driver.set_device(device_id);

    int ret = driver.unprepare_dev_buf(&m_dev_buf);
    if (ret == 0) {
        m_is_prepared = false;
        m_prepared_device = 0;
        return true;
    }
    std::cerr << "DmaBuffer::unprepare failed on device " << device_id 
              << ": " << driver.decode_error(ret) << std::endl;
    return false;
}

} // namespace eswin
