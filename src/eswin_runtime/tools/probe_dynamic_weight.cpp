/**
 * @file probe_dynamic_weight.cpp
 * @brief Acceptance test for eswin_compiler Milestone 7: can a loadable read its WEIGHTS from
 *        a buffer supplied at submit time?
 *
 * This is the MoE pattern exactly. One weight blob is allocated holding two different weight
 * sets back to back; the same loadable is submitted twice, selecting the set via
 * DevBuf.offset. If the outputs differ, a single per-layer loadable can serve any expert by
 * changing an offset - 27 loadables instead of 1716.
 *
 * The loadable is produced by deep/artifacts/scratch/make_dynamic_weight_model.py, which
 * converts the weight memory entry to Class B (contents=[], flags=5, bind_id=1) and drops the
 * weight blobs. bind_id=1 means the weight is input slot 1, so inputFdNum=2:
 *   inputFd[0] = activation (bind_id 0)
 *   inputFd[1] = weight     (bind_id 1)
 *
 * Build (board): riscv64-linux-gnu-g++ -std=c++17 -O2 -pthread -I<inc> \
 *                    tools/probe_dynamic_weight.cpp -o probe -L<build> -les_runtime -ldl -pthread
 * Run:  sudo LD_LIBRARY_PATH=. ./probe ./dyn_noreloc.model
 */

#include "../src/backend/driver_wrapper.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

using namespace eswin;

namespace {

constexpr size_t kInSize = 1152;       // 576 x fp16 activation
constexpr size_t kOutSize = 3072;      // 1536 x fp16 output
constexpr size_t kWeightSize = 1769472;  // 3 x 589824, exactly 432 pages so offset stays aligned

struct Buf {
    uint64_t fd = 0;
    void *va = nullptr;
};

bool alloc(EswinDriver &drv, Buf &b, size_t size, const char *what) {
    if (drv.mem_alloc(&b.fd, size, CacheMode::Cached, "mmz_nid_0_part_0") != 0) {
        std::printf("FAIL: mem_alloc %s (%zu bytes)\n", what, size);
        return false;
    }
    b.va = drv.mem_mmap(b.fd, size, CacheMode::Cached);
    if (!b.va) {
        std::printf("FAIL: mem_mmap %s\n", what);
        return false;
    }
    std::memset(b.va, 0, size);
    drv.mem_flush_cache(b.fd);
    return true;
}

void fill(void *dst, uint16_t bits, size_t bytes) {
    uint16_t *p = static_cast<uint16_t *>(dst);
    for (size_t i = 0; i < bytes / 2; ++i) p[i] = bits;
}

// Submits with the activation at input slot 0 and the weight at slot 1, selected by offset.
bool run(EswinDriver &drv, uint32_t model_id, const Buf &act, const Buf &wblob,
         size_t weight_offset, Buf &out, const char *label) {
    DevBuf in_act{act.fd, 0, kInSize, 0};
    DevBuf in_w{wblob.fd, weight_offset, kWeightSize, 0};
    DevBuf o{out.fd, 0, kOutSize, 0};

    if (drv.prepare_dev_buf(&in_act) != 0) { std::printf("  [%s] prepare act FAILED\n", label); return false; }
    if (drv.prepare_dev_buf(&in_w) != 0)   { std::printf("  [%s] prepare weight FAILED\n", label); return false; }
    if (drv.prepare_dev_buf(&o) != 0)      { std::printf("  [%s] prepare out FAILED\n", label); return false; }

    NpuTask task;
    std::memset(&task, 0, sizeof(task));
    task.taskId = 1;
    task.modelId = model_id;
    task.inputFdNum = 2;          // activation + runtime-bound weight
    task.outputFdNum = 1;
    task.inputFd[0] = in_act;
    task.inputFd[1] = in_w;
    task.outputFd[0] = o;

    int rc = drv.submit_task(&task, 1);
    if (rc != 0) {
        std::printf("  [%s] submit FAILED rc=%d (%s)\n", label, rc,
                    EswinDriver::decode_error(rc).c_str());
        return false;
    }
    std::printf("  [%s] submit ok (weight offset=%zu)\n", label, weight_offset);
    return true;
}

double mean_u16(const void *p, size_t bytes) {
    const uint16_t *v = static_cast<const uint16_t *>(p);
    double s = 0.0;
    for (size_t i = 0; i < bytes / 2; ++i) s += static_cast<double>(v[i]);
    return s / static_cast<double>(bytes / 2);
}

bool all_zero(const void *p, size_t bytes) {
    const uint8_t *v = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < bytes; ++i) if (v[i]) return false;
    return true;
}

} // namespace

int main(int argc, char **argv) {
    const char *model = (argc > 1) ? argv[1] : "dyn_noreloc.model";

    EswinDriver &drv = EswinDriver::instance();
    // initialize() dlopens the vendor libs and calls ES_NPU_GetNumDevices; instance() alone
    // reports no hardware even on the board.
    if (!drv.initialize()) {
        std::printf("SKIP: no NPU hardware\n");
        return 2;
    }
    if (drv.set_device(0) != 0) { std::printf("FAIL: set_device(0)\n"); return 1; }

    uint32_t model_id = 0;
    int rc = drv.load_model(&model_id, model);
    if (rc != 0) {
        std::printf("FAIL: load_model('%s') rc=%d (%s)\n", model, rc,
                    EswinDriver::decode_error(rc).c_str());
        return 1;
    }
    std::printf("loaded '%s' as model_id=%u (devices=%u)\n", model, model_id,
                drv.get_num_devices());

    Buf act, wblob, o0, o1;
    // One blob holding two different weight sets back to back - the MoE expert blob.
    if (!alloc(drv, act, kInSize, "activation")) return 1;
    if (!alloc(drv, wblob, kWeightSize * 2, "weight blob (2 sets)")) return 1;
    if (!alloc(drv, o0, kOutSize, "out0")) return 1;
    if (!alloc(drv, o1, kOutSize, "out1")) return 1;

    constexpr uint16_t kAct = 0x3800;    // +0.5 activation
    constexpr uint16_t kW0 = 0x3800;     // weight set 0: +0.5
    constexpr uint16_t kW1 = 0x3C00;     // weight set 1: +1.0  (different result expected)
    fill(act.va, kAct, kInSize);
    fill(wblob.va, kW0, kWeightSize);
    fill(static_cast<char *>(wblob.va) + kWeightSize, kW1, kWeightSize);
    drv.mem_flush_cache(act.fd);
    drv.mem_flush_cache(wblob.fd);

    std::printf("\n--- two submits, same loadable, different weight offset ---\n");
    bool ok = run(drv, model_id, act, wblob, 0, o0, "W0 offset=0") &&
              run(drv, model_id, act, wblob, kWeightSize, o1, "W1 offset=1769472");
    if (!ok) { drv.unload_model(model_id); return 1; }

    drv.mem_flush_cache(o0.fd);
    drv.mem_flush_cache(o1.fd);

    const bool differ = std::memcmp(o0.va, o1.va, kOutSize) != 0;
    std::printf("\n--- results ---\n");
    std::printf("  mean|out W0| = %.3f%s\n", mean_u16(o0.va, kOutSize),
                all_zero(o0.va, kOutSize) ? "   (ALL ZERO - NPU did not write)" : "");
    std::printf("  mean|out W1| = %.3f%s\n", mean_u16(o1.va, kOutSize),
                all_zero(o1.va, kOutSize) ? "   (ALL ZERO - NPU did not write)" : "");
    std::printf("  outputs differ: %s\n", differ ? "yes" : "no");

    std::printf("\n--- verdict ---\n");
    if (all_zero(o0.va, kOutSize) || all_zero(o1.va, kOutSize)) {
        std::printf("  INCONCLUSIVE: an output buffer was never written, so the submit did\n"
                    "                  not execute the graph. Check bind_id/inputFdNum.\n");
    } else if (differ) {
        std::printf("  RUNTIME-BOUND WEIGHT WORKS: the same loadable produced different\n"
                    "  output purely by changing DevBuf.offset into one resident blob.\n"
                    "  => MoE approach (b) confirmed end to end on hardware.\n");
    } else {
        std::printf("  WEIGHT NOT READ FROM THE SUPPLIED BUFFER: outputs are identical, so\n"
                    "                  the NPU used something else (stale blob, or the Class B\n"
                    "                  weight was ignored). Try the dyn_reloc.model variant.\n");
    }

    drv.unload_model(model_id);
    return 0;
}
