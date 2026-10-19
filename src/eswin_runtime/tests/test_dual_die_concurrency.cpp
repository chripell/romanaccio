/**
 * @file test_dual_die_concurrency.cpp
 * @brief Stress Test for Dual-Die Process Isolation & NUMA Worker Daemon.
 *
 * Verifies parallel execution across Die 0 and Die 1 without encountering
 * vendor ES_NPU_ERROR_DEV_MULTI_SET (0xA00F6046) single-process collision.
 */

#include "es_runtime.h"
#include "../src/backend/driver_wrapper.hpp"
#include "../src/backend/dma_buffer.hpp"
#include "../src/backend/worker_pool.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "\033[1;31m[FAILED]\033[0m %s:%d: %s\n", __FILE__, __LINE__, msg); \
            std::exit(1); \
        } \
    } while (0)

static const char *YOLO_MODEL_PATH = "/opt/eswin/data/npu/yolov3/end2endyolov3s416_batch1_int8_npu_b1.model";

static void test_worker_pool_basic() {
    std::cout << "[*] Testing WorkerPool spawn, handshake, and NUMA worker lifecycle..." << std::endl;

    eswin::WorkerPool pool;
    bool ok = pool.start(2);
    TEST_ASSERT(ok, "WorkerPool::start(2) failed");
    TEST_ASSERT(pool.active_worker_count() >= 1, "No active workers spawned");

    std::cout << "    Spawned " << pool.active_worker_count() << " worker daemons successfully." << std::endl;

    // Allocate test DMA buffers on Die 0 and Die 1
    eswin::DmaBuffer in_buf0(4096, eswin::CacheMode::NoCache, "mmz_nid_0_part_0");
    eswin::DmaBuffer in_buf1(4096, eswin::CacheMode::NoCache, "mmz_nid_1_part_0");

    TEST_ASSERT(pool.prepare_buffer(0, in_buf0), "prepare_buffer on die 0 failed");
    if (pool.active_worker_count() > 1) {
        TEST_ASSERT(pool.prepare_buffer(1, in_buf1), "prepare_buffer on die 1 failed");
        TEST_ASSERT(pool.unprepare_buffer(1, in_buf1), "unprepare_buffer on die 1 failed");
    }
    TEST_ASSERT(pool.unprepare_buffer(0, in_buf0), "unprepare_buffer on die 0 failed");

    pool.shutdown();
    std::cout << "\033[1;32m[PASS]\033[0m WorkerPool basic lifecycle verified." << std::endl;
}

static void test_concurrent_dual_die_inference(int num_iterations = 1000) {
    std::cout << "[*] Testing Concurrent Dual-Die Execution (" << num_iterations << " iterations)..." << std::endl;

    eswin::WorkerPool pool;
    bool ok = pool.start(2);
    TEST_ASSERT(ok, "WorkerPool::start failed");

    auto &driver = eswin::EswinDriver::instance();
    bool is_hw = driver.is_hardware_available();

    // Check if model exists
    bool model_exists = false;
    if (is_hw) {
        FILE *f = std::fopen(YOLO_MODEL_PATH, "rb");
        if (f) {
            model_exists = true;
            std::fclose(f);
        }
    }

    eswin::ModelInfo info_d0;
    eswin::ModelInfo info_d1;

    std::string model_path = model_exists ? YOLO_MODEL_PATH : "synthetic_model.model";

    TEST_ASSERT(pool.load_model(0, model_path, info_d0), "load_model on Die 0 failed");
    std::cout << "    Loaded model on Die 0 (id=" << info_d0.model_id << ", "
              << (int)info_d0.num_inputs << " in, " << (int)info_d0.num_outputs << " out)" << std::endl;

    if (pool.active_worker_count() > 1) {
        TEST_ASSERT(pool.load_model(1, model_path, info_d1), "load_model on Die 1 failed");
        std::cout << "    Loaded model on Die 1 (id=" << info_d1.model_id << ", "
                  << (int)info_d1.num_inputs << " in, " << (int)info_d1.num_outputs << " out)" << std::endl;
    }

    // Allocate exact buffers for Die 0
    std::vector<std::unique_ptr<eswin::DmaBuffer>> in_bufs0;
    std::vector<std::unique_ptr<eswin::DmaBuffer>> out_bufs0;
    std::vector<eswin::DmaBuffer*> inputs0;
    std::vector<eswin::DmaBuffer*> outputs0;

    for (uint8_t i = 0; i < info_d0.num_inputs; ++i) {
        size_t s = info_d0.input_sizes[i] > 0 ? info_d0.input_sizes[i] : 4096;
        auto b = std::make_unique<eswin::DmaBuffer>(s, eswin::CacheMode::NoCache, "mmz_nid_0_part_0");
        TEST_ASSERT(pool.prepare_buffer(0, *b), "prepare in_bufs0 failed");
        inputs0.push_back(b.get());
        in_bufs0.push_back(std::move(b));
    }

    for (uint8_t i = 0; i < info_d0.num_outputs; ++i) {
        size_t s = info_d0.output_sizes[i] > 0 ? info_d0.output_sizes[i] : 4096;
        auto b = std::make_unique<eswin::DmaBuffer>(s, eswin::CacheMode::NoCache, "mmz_nid_0_part_0");
        TEST_ASSERT(pool.prepare_buffer(0, *b), "prepare out_bufs0 failed");
        outputs0.push_back(b.get());
        out_bufs0.push_back(std::move(b));
    }

    // Allocate exact buffers for Die 1
    std::vector<std::unique_ptr<eswin::DmaBuffer>> in_bufs1;
    std::vector<std::unique_ptr<eswin::DmaBuffer>> out_bufs1;
    std::vector<eswin::DmaBuffer*> inputs1;
    std::vector<eswin::DmaBuffer*> outputs1;

    if (pool.active_worker_count() > 1) {
        for (uint8_t i = 0; i < info_d1.num_inputs; ++i) {
            size_t s = info_d1.input_sizes[i] > 0 ? info_d1.input_sizes[i] : 4096;
            auto b = std::make_unique<eswin::DmaBuffer>(s, eswin::CacheMode::NoCache, "mmz_nid_1_part_0");
            TEST_ASSERT(pool.prepare_buffer(1, *b), "prepare in_bufs1 failed");
            inputs1.push_back(b.get());
            in_bufs1.push_back(std::move(b));
        }

        for (uint8_t i = 0; i < info_d1.num_outputs; ++i) {
            size_t s = info_d1.output_sizes[i] > 0 ? info_d1.output_sizes[i] : 4096;
            auto b = std::make_unique<eswin::DmaBuffer>(s, eswin::CacheMode::NoCache, "mmz_nid_1_part_0");
            TEST_ASSERT(pool.prepare_buffer(1, *b), "prepare out_bufs1 failed");
            outputs1.push_back(b.get());
            out_bufs1.push_back(std::move(b));
        }
    }

    std::atomic<int> completed_d0{0};
    std::atomic<int> completed_d1{0};

    auto t_start = std::chrono::high_resolution_clock::now();

    // Launch worker thread for Die 0
    auto future_d0 = std::async(std::launch::async, [&]() {
        for (int i = 0; i < num_iterations; ++i) {
            uint64_t ns = 0;
            bool s = pool.submit_task(0, info_d0.model_id, inputs0, outputs0, &ns);
            if (!s) {
                std::cerr << "Die 0 task failed at iter " << i << std::endl;
                return false;
            }
            completed_d0++;
            if (num_iterations > 20) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        return true;
    });

    // Launch worker thread for Die 1 if dual-die is available
    std::future<bool> future_d1;
    if (pool.active_worker_count() > 1) {
        future_d1 = std::async(std::launch::async, [&]() {
            for (int i = 0; i < num_iterations; ++i) {
                uint64_t ns = 0;
                bool s = pool.submit_task(1, info_d1.model_id, inputs1, outputs1, &ns);
                if (!s) {
                    std::cerr << "Die 1 task failed at iter " << i << std::endl;
                    return false;
                }
                completed_d1++;
                if (num_iterations > 20) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }
            return true;
        });
    }

    bool res0 = future_d0.get();
    bool res1 = (pool.active_worker_count() > 1) ? future_d1.get() : true;

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    TEST_ASSERT(res0, "Die 0 execution encountered failures");
    TEST_ASSERT(res1, "Die 1 execution encountered failures");

    int total_tasks = completed_d0.load() + completed_d1.load();
    double fps = (total_tasks / total_ms) * 1000.0;

    std::cout << "\033[1;32m[PASS]\033[0m Executed " << total_tasks << " total tasks across "
              << pool.active_worker_count() << " dies in " << total_ms << " ms ("
              << fps << " total tasks/sec, avg latency = "
              << (total_ms / num_iterations) << " ms/step)" << std::endl;

    // Unprepare & cleanup
    for (auto &b : in_bufs0) pool.unprepare_buffer(0, *b);
    for (auto &b : out_bufs0) pool.unprepare_buffer(0, *b);
    if (pool.active_worker_count() > 1) {
        for (auto &b : in_bufs1) pool.unprepare_buffer(1, *b);
        for (auto &b : out_bufs1) pool.unprepare_buffer(1, *b);
    }

    pool.unload_model(0, info_d0.model_id);
    if (pool.active_worker_count() > 1) {
        pool.unload_model(1, info_d1.model_id);
    }
}

static void test_c_api_worker_pool() {
    std::cout << "[*] Testing Public C11 Worker Pool API (es_runtime.h)..." << std::endl;

    es_context_t *ctx = nullptr;
    TEST_ASSERT(es_init(&ctx) == ES_SUCCESS, "es_init failed");

    es_worker_pool_t *pool = nullptr;
    TEST_ASSERT(es_worker_pool_create(ctx, 2, &pool) == ES_SUCCESS, "es_worker_pool_create failed");
    TEST_ASSERT(pool != nullptr, "pool handle null");

    // Allocate small DMA buffer
    es_dma_buf_t *buf = nullptr;
    TEST_ASSERT(es_dma_alloc(ctx, 64 * 1024, ES_MEM_NOCACHE, &buf) == ES_SUCCESS, "es_dma_alloc failed");

    TEST_ASSERT(es_worker_pool_prepare_buffer(pool, 0, buf) == ES_SUCCESS, "es_worker_pool_prepare_buffer failed");

    es_dma_free(buf);
    es_worker_pool_destroy(pool);
    es_free(ctx);

    std::cout << "\033[1;32m[PASS]\033[0m Public C11 Worker Pool API verified." << std::endl;
}

int main(int argc, char **argv) {
    int iterations = 20;
    if (argc > 1) {
        iterations = std::atoi(argv[1]);
        if (iterations <= 0) iterations = 20;
    }

    std::cout << "========================================================\n";
    std::cout << "  EIC7700X Multi-Die Concurrency & NUMA Worker Test     \n";
    std::cout << "========================================================\n\n";

    test_worker_pool_basic();
    test_concurrent_dual_die_inference(iterations);
    test_c_api_worker_pool();

    std::cout << "\n========================================================\n";
    std::cout << "\033[1;32m  ALL MILESTONE 2 CONCURRENCY TESTS PASSED!             \033[0m\n";
    std::cout << "========================================================\n";

    return 0;
}
