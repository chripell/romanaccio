/**
 * @file probe_submit_overhead.cpp
 * @brief Measure per-submit and per-load cost on real NPU hardware (DeepSeek-V2-Lite Phase 0).
 *
 * Why this gates the MoE design: expert-parallel decode issues roughly 318 submissions
 * per token (26 MoE layers x (6 routed + 2 shared) experts, plus attention projections and
 * the LM head). If a single submit costs more than ~0.3 ms, submission overhead alone caps
 * throughput below 10 tok/s no matter how much bandwidth the weights have — so the number
 * decides whether per-expert-per-layer submission is viable at all, or whether experts must
 * be batched into fewer, larger graphs.
 *
 * Reports:
 *   - wall time per ES_NPU_Submit (mean / p50 / p95 / max) over a repeat loop
 *   - ES_NPU_LoadModelFromFile time for one model, and the number of models that can be
 *     held loaded at once (approach (a) extrapolates from this)
 *
 * Caveat printed at runtime: this measures wall time around submit_task(), which includes
 * driver completion wait. It is the number a decode loop actually pays, but it is NOT a
 * pure ioctl cost, and it is measured on one loadable shape.
 *
 * Build (board):
 *   riscv64-linux-gnu-g++ -std=c++17 -O2 -pthread -Iinclude tools/probe_submit_overhead.cpp \
 *     -o probe_so -Lbuild -les_runtime -ldl -pthread
 * Run:
 *   sudo LD_LIBRARY_PATH=. ./probe_so ./model.model 1152 1152 200
 */

#include "../src/backend/driver_wrapper.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

using namespace eswin;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Buf {
    uint64_t fd = 0;
    void *va = nullptr;
};

bool alloc(EswinDriver &drv, Buf &b, size_t size) {
    if (drv.mem_alloc(&b.fd, size, CacheMode::Cached, "mmz_nid_0_part_0") != 0) return false;
    b.va = drv.mem_mmap(b.fd, size, CacheMode::Cached);
    if (!b.va) return false;
    std::memset(b.va, 0x11, size);
    drv.mem_flush_cache(b.fd);
    return true;
}

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t i = static_cast<size_t>(p * static_cast<double>(v.size() - 1));
    return v[i];
}

} // namespace

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "test_swiglu_mlp_npu_b1.model";
    const size_t in_bytes = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 1152;
    const size_t out_bytes = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 1152;
    const int iters = (argc > 4) ? std::atoi(argv[4]) : 200;

    EswinDriver &drv = EswinDriver::instance();
    if (!drv.initialize()) { std::printf("SKIP: no NPU hardware\n"); return 2; }
    if (drv.set_device(0) != 0) { std::printf("FAIL: set_device(0)\n"); return 1; }
    std::printf("model=%s in=%zu out=%zu iters=%d devices=%u\n\n",
                path, in_bytes, out_bytes, iters, drv.get_num_devices());

    Buf act, out;
    if (!alloc(drv, act, in_bytes) || !alloc(drv, out, out_bytes)) {
        std::printf("FAIL: DMA allocation\n");
        return 1;
    }

    // ---- 1. load time -------------------------------------------------------
    uint32_t model_id = 0;
    {
        auto t0 = Clock::now();
        int rc = drv.load_model(&model_id, path);
        double el = ms_since(t0);
        if (rc != 0) {
            std::printf("FAIL: load_model rc=%d (%s)\n", rc, EswinDriver::decode_error(rc).c_str());
            return 1;
        }
        std::printf("load: %.3f ms (model_id=%u)\n", el, model_id);
    }

    DevBuf in_d{act.fd, 0, in_bytes, 0};
    DevBuf out_d{out.fd, 0, out_bytes, 0};
    if (drv.prepare_dev_buf(&in_d) != 0 || drv.prepare_dev_buf(&out_d) != 0) {
        std::printf("FAIL: prepare_dev_buf\n");
        return 1;
    }

    // ---- 2. per-submit wall time -------------------------------------------
    NpuTask task;
    std::memset(&task, 0, sizeof(task));
    task.taskId = 1;
    task.modelId = model_id;
    task.inputFdNum = 1;
    task.outputFdNum = 1;
    task.inputFd[0] = in_d;
    task.outputFd[0] = out_d;

    // Warm-up, then measure.
    int warm = std::min(10, iters);
    for (int i = 0; i < warm; ++i) {
        std::memset(&task.state, 0, sizeof(task.state));
        drv.submit_task(&task, 1);
    }

    std::vector<double> samples;
    samples.reserve(iters);
    int failures = 0;
    for (int i = 0; i < iters; ++i) {
        std::memset(&task.sdkPrivate, 0, sizeof(task.sdkPrivate));
        task.state = 0;
        auto t0 = Clock::now();
        int rc = drv.submit_task(&task, 1);
        double el = ms_since(t0);
        if (rc != 0) { ++failures; continue; }
        samples.push_back(el);
    }

    if (samples.empty()) {
        std::printf("FAIL: every submit failed (%d)\n", failures);
        return 1;
    }

    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size());
    const double worst = *std::max_element(samples.begin(), samples.end());
    std::printf("submit: n=%zu mean=%.3f ms  p50=%.3f  p95=%.3f  max=%.3f  (%d failed)\n",
                samples.size(), mean, pct(samples, 0.50), pct(samples, 0.95),
                worst, failures);

    // ---- 3. how many models can stay loaded at once ------------------------
    std::vector<uint32_t> ids;
    int load_rc = 0;
    for (int i = 0; i < 40; ++i) {
        uint32_t id = 0;
        auto t0 = Clock::now();
        load_rc = drv.load_model(&id, path);
        double el = ms_since(t0);
        if (load_rc != 0) {
            std::printf("concurrent loads: stopped at %d (rc=%d after %.2f ms)\n",
                        static_cast<int>(ids.size()), load_rc, el);
            break;
        }
        ids.push_back(id);
        if (i == 0 || i == 9 || i == 19 || i == 39) {
            std::printf("  %2d loaded, last=%.3f ms (id=%u)\n", i + 1, el, id);
        }
    }
    if (ids.size() >= 40) std::printf("concurrent loads: >=40 succeeded\n");
    for (uint32_t id : ids) drv.unload_model(id);

    // ---- 4. what it means for the MoE decode loop -------------------------
    const double ms_per = mean;
    std::printf("\n--- implication for expert-parallel decode ---\n");
    std::printf("  %.0f submits/token x %.3f ms = %.1f ms/token -> %.2f tok/s ceiling\n",
               318.0, ms_per, 318.0 * ms_per, 1000.0 / (318.0 * ms_per));
    std::printf("  budget for <=5 tok/s: <= %.3f ms per submit\n", 1000.0 / 5.0 / 318.0);
    std::printf("  budget for >=10 tok/s: <= %.3f ms per submit\n", 1000.0 / 10.0 / 318.0);
    std::printf("  NOTE: wall time includes driver completion wait, not a bare ioctl.\n");

    drv.unload_model(model_id);
    return 0;
}
