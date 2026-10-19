/**
 * @file test_dma_alloc.cpp
 * @brief Standalone verification test for EIC7700X Contiguous DMA Memory Allocator.
 *
 * Verifies physical allocation, userspace mmap, memory pattern validation,
 * cache maintenance, SMMU preparation, and RAII cleanup across 1 MB, 16 MB, and 128 MB buffers.
 */

#include "es_runtime.h"
#include "../src/backend/driver_wrapper.hpp"
#include "../src/backend/dma_buffer.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "\033[1;31m[FAILED]\033[0m %s:%d: %s\n", __FILE__, __LINE__, msg); \
            std::exit(1); \
        } \
    } while (0)

static void test_cpp_dma_buffer(size_t size_bytes, const char *label) {
    std::cout << "[*] Testing C++ DmaBuffer: " << label << " (" << (size_bytes / (1024 * 1024)) << " MB)..." << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Allocate buffer
    eswin::DmaBuffer buf(size_bytes, eswin::CacheMode::NoCache);
    TEST_ASSERT(buf.is_valid(), "DmaBuffer is_valid() returned false");
    TEST_ASSERT(buf.data() != nullptr, "DmaBuffer data() returned null pointer");
    TEST_ASSERT(buf.size() >= size_bytes, "Allocated size less than requested size");

    // 2. Write pattern and verify
    uint32_t *ptr = buf.as<uint32_t>();
    size_t num_elements = size_bytes / sizeof(uint32_t);

    for (size_t i = 0; i < num_elements; i += 1024) {
        ptr[i] = static_cast<uint32_t>(0xCAFE0000 + i);
    }

    for (size_t i = 0; i < num_elements; i += 1024) {
        TEST_ASSERT(ptr[i] == static_cast<uint32_t>(0xCAFE0000 + i), "Pattern verification mismatch");
    }

    // 3. Test Cache Flush
    TEST_ASSERT(buf.flush_cache(), "flush_cache() failed");

    // 4. Test SMMU Prepare & Unprepare
    TEST_ASSERT(buf.prepare(0), "prepare(0) failed to map SMMU table on Die 0");
    TEST_ASSERT(buf.is_prepared(), "is_prepared() should be true");
    TEST_ASSERT(buf.unprepare(0), "unprepare(0) failed to unmap SMMU table on Die 0");
    TEST_ASSERT(!buf.is_prepared(), "is_prepared() should be false after unprepare");

    // Dual-die test if Die 1 is detected
    if (eswin::EswinDriver::instance().get_num_devices() > 1) {
        TEST_ASSERT(buf.prepare(1), "prepare(1) failed to map SMMU table on Die 1");
        TEST_ASSERT(buf.is_prepared(), "is_prepared() should be true on Die 1");
        TEST_ASSERT(buf.unprepare(1), "unprepare(1) failed to unmap SMMU table on Die 1");
        TEST_ASSERT(!buf.is_prepared(), "is_prepared() should be false after unprepare");
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "\033[1;32m[PASS]\033[0m " << label << " allocated, verified, and released in " << ms << " ms (memFd=" << buf.mem_fd() << ")" << std::endl;
}

static void test_move_semantics() {
    std::cout << "[*] Testing DmaBuffer move semantics (ownership transfer)..." << std::endl;

    eswin::DmaBuffer b1(1024 * 1024);
    TEST_ASSERT(b1.is_valid(), "b1 valid");
    uint64_t fd1 = b1.mem_fd();
    void *ptr1 = b1.data();

    // Move construct
    eswin::DmaBuffer b2(std::move(b1));
    TEST_ASSERT(!b1.is_valid(), "b1 should be invalidated after move");
    TEST_ASSERT(b1.data() == nullptr, "b1 data should be null");
    TEST_ASSERT(b2.is_valid(), "b2 should be valid");
    TEST_ASSERT(b2.data() == ptr1, "b2 pointer should match ptr1");
    TEST_ASSERT(b2.mem_fd() == fd1, "b2 mem_fd should match fd1");

    // Move assign
    eswin::DmaBuffer b3(2 * 1024 * 1024);
    b3 = std::move(b2);
    TEST_ASSERT(!b2.is_valid(), "b2 should be invalidated after move assign");
    TEST_ASSERT(b3.is_valid(), "b3 should be valid");
    TEST_ASSERT(b3.data() == ptr1, "b3 pointer should match ptr1");

    std::cout << "\033[1;32m[PASS]\033[0m Move semantics verified successfully." << std::endl;
}

static void test_c_api() {
    std::cout << "[*] Testing Public C11 API (es_runtime.h)..." << std::endl;

    es_context_t *ctx = nullptr;
    es_status_t st = es_init(&ctx);
    TEST_ASSERT(st == ES_SUCCESS, "es_init failed");
    TEST_ASSERT(ctx != nullptr, "ctx is null");

    uint16_t dev_count = 0;
    st = es_get_device_count(ctx, &dev_count);
    TEST_ASSERT(st == ES_SUCCESS, "es_get_device_count failed");
    std::cout << "    Detected NPU device count: " << dev_count << std::endl;

    // Test C DMA allocation (16 MB)
    es_dma_buf_t *dma = nullptr;
    st = es_dma_alloc(ctx, 16 * 1024 * 1024, ES_MEM_NOCACHE, &dma);
    TEST_ASSERT(st == ES_SUCCESS, "es_dma_alloc failed");
    TEST_ASSERT(dma != nullptr, "dma is null");

    void *host_ptr = es_dma_get_host_ptr(dma);
    TEST_ASSERT(host_ptr != nullptr, "host_ptr is null");
    TEST_ASSERT(es_dma_get_size(dma) >= 16 * 1024 * 1024, "size mismatch");

    // Write data
    std::memset(host_ptr, 0xA5, 1024 * 1024);
    uint8_t *bptr = static_cast<uint8_t*>(host_ptr);
    TEST_ASSERT(bptr[0] == 0xA5 && bptr[1024] == 0xA5, "byte pattern mismatch");

    // Flush cache
    st = es_dma_flush_cache(dma);
    TEST_ASSERT(st == ES_SUCCESS, "es_dma_flush_cache failed");

    // Free
    es_dma_free(dma);
    es_free(ctx);

    std::cout << "\033[1;32m[PASS]\033[0m Public C11 API verified successfully." << std::endl;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  EIC7700X Contiguous DMA Memory Allocator Test Suite   \n";
    std::cout << "========================================================\n\n";

    // Initialize Driver
    auto &driver = eswin::EswinDriver::instance();
    bool hw = driver.initialize();
    std::cout << "[*] Hardware NPU & MMZ driver status: "
              << (hw ? "\033[1;32mActive (Hardware Detected)\033[0m" : "\033[1;33mHost Fallback (No Hardware)\033[0m")
              << " (" << driver.get_num_devices() << " dies)" << std::endl << std::endl;

    // Run tests
    test_cpp_dma_buffer(1 * 1024 * 1024,   "1 MB Buffer");
    test_cpp_dma_buffer(16 * 1024 * 1024,  "16 MB Buffer");
    test_cpp_dma_buffer(128 * 1024 * 1024, "128 MB Buffer");

    test_move_semantics();
    test_c_api();

    std::cout << "\n========================================================\n";
    std::cout << "\033[1;32m  ALL MILESTONE 1 DMA TESTS COMPLETED SUCCESSFULLY!     \033[0m\n";
    std::cout << "========================================================\n";

    return 0;
}
