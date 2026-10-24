/**
 * @file probe_devbuf_offset.cpp
 * @brief Does the NPU driver honour a non-zero DevBuf.offset when binding a Class B tensor?
 *
 * This is open question 1 of eswin_compiler Milestone 7, and it gates the DeepSeek-V2-Lite
 * MoE design: approach (b) wants ONE resident expert blob per die with expert selection done
 * by changing DevBuf.offset per submit. If the driver ignores offset, that design is dead and
 * we need one buffer per expert instead.
 *
 * Method — three runs of the same loadable (test_linear_1x1conv_npu_b1.model, input 1152 B =
 * 576 fp16, output 3072 B = 1536 fp16):
 *
 *   A: one DMA buffer, pattern P0 at offset 0,      submit with DevBuf.offset = 0
 *   B: the SAME buffer,        pattern P1 at offset OFF, submit with DevBuf.offset = OFF
 *   C: a SEPARATE buffer,      pattern P1 at offset 0,   submit with DevBuf.offset = 0
 *
 * Interpretation:
 *   offset honoured  => B == C  and  A != B   (B really read P1, from the shared buffer)
 *   offset ignored   => A == B                (both read whatever sits at offset 0, i.e. P0)
 *
 * Build (host):    g++ -std=c++17 -O2 -I../include probe_devbuf_offset.cpp -o probe \
 *                      -L../build -les_runtime -ldl -pthread
 * Build (board):   riscv64-linux-gnu-g++ -std=c++17 -O2 -I<inc> probe_devbuf_offset.cpp -o probe \
 *                      -L<build> -les_runtime -ldl -pthread
 * Run:             sudo LD_LIBRARY_PATH=. ./probe ./test_linear_1x1conv_npu_b1.model
 */

#include "../src/backend/driver_wrapper.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace eswin;

namespace {

constexpr size_t kInSize = 1152;    // 576 x fp16
constexpr size_t kOutSize = 3072;   // 1536 x fp16
constexpr size_t kOff = 4096;       // page-aligned second slot inside the shared buffer
constexpr size_t kSharedSize = kOff + kInSize + 4096;

// fp16 bit patterns, so we do not depend on a half-precision type here.
constexpr uint16_t kP0 = 0x3800;    //  0.5
constexpr uint16_t kP1 = 0xBE00;    // -0.5

struct Buf {
    uint64_t fd = 0;
    void *va = nullptr;
    size_t size = 0;
};

bool alloc(EswinDriver &drv, Buf &b, size_t size, const char *what) {
    if (drv.mem_alloc(&b.fd, size, CacheMode::Cached, "mmz_nid_0_part_0") != 0) {
        std::printf("FAIL: mem_alloc for %s\n", what);
        return false;
    }
    b.va = drv.mem_mmap(b.fd, size, CacheMode::Cached);
    if (!b.va) {
        std::printf("FAIL: mem_mmap for %s\n", what);
        return false;
    }
    b.size = size;
    std::memset(b.va, 0, size);
    drv.mem_flush_cache(b.fd);
    return true;
}

void fill_pattern(void *dst, uint16_t bits, size_t bytes) {
    uint16_t *p = static_cast<uint16_t *>(dst);
    for (size_t i = 0; i < bytes / 2; ++i) p[i] = bits;
}

bool run(EswinDriver &drv, uint32_t model_id, DevBuf in, DevBuf out, const char *label) {
    if (drv.prepare_dev_buf(&in) != 0) {
        std::printf("  [%s] prepare input FAILED\n", label);
        return false;
    }
    if (drv.prepare_dev_buf(&out) != 0) {
        std::printf("  [%s] prepare output FAILED\n", label);
        return false;
    }

    NpuTask task;
    std::memset(&task, 0, sizeof(task));
    task.taskId = 1;
    task.modelId = model_id;
    task.inputFdNum = 1;
    task.outputFdNum = 1;
    task.inputFd[0] = in;
    task.outputFd[0] = out;

    int rc = drv.submit_task(&task, 1);
    if (rc != 0) {
        std::printf("  [%s] submit FAILED rc=%d (%s)\n", label, rc,
                    EswinDriver::decode_error(rc).c_str());
        return false;
    }
    std::printf("  [%s] submit ok (input offset=%llu size=%llu)\n", label,
                static_cast<unsigned long long>(in.offset),
                static_cast<unsigned long long>(in.size));
    return true;
}

double mean_abs(const void *p, size_t bytes) {
    // Raw fp16 bit patterns averaged as unsigned; only used to tell runs apart, and a
    // constant input gives a constant output so any difference is meaningful.
    const uint16_t *v = static_cast<const uint16_t *>(p);
    double s = 0.0;
    for (size_t i = 0; i < bytes / 2; ++i) s += static_cast<double>(v[i]);
    return s / static_cast<double>(bytes / 2);
}

} // namespace

int main(int argc, char **argv) {
    const char *model = (argc > 1) ? argv[1] : "test_linear_1x1conv_npu_b1.model";

    EswinDriver &drv = EswinDriver::instance();
    // initialize() is what dlopens the vendor libraries and calls ES_NPU_GetNumDevices.
    // instance() alone leaves m_hardware_available false, which reads as "no NPU here"
    // even on the board.
    if (!drv.initialize()) {
        std::printf("SKIP: no NPU hardware on this host\n");
        return 2;
    }
    std::printf("devices=%u\n", drv.get_num_devices());
    if (drv.set_device(0) != 0) {
        std::printf("FAIL: set_device(0)\n");
        return 1;
    }

    uint32_t model_id = 0;
    int rc = drv.load_model(&model_id, model);
    if (rc != 0) {
        std::printf("FAIL: load_model('%s') rc=%d (%s)\n", model, rc,
                    EswinDriver::decode_error(rc).c_str());
        return 1;
    }
    std::printf("model_id=%u  (%s)\n", model_id, model);

    Buf shared, ctrl, oa, ob, oc;
    if (!alloc(drv, shared, kSharedSize, "shared input")) return 1;
    if (!alloc(drv, ctrl, kInSize, "control input")) return 1;
    if (!alloc(drv, oa, kOutSize, "out A")) return 1;
    if (!alloc(drv, ob, kOutSize, "out B")) return 1;
    if (!alloc(drv, oc, kOutSize, "out C")) return 1;

    // Pattern P0 at offset 0 and P1 at offset kOff inside the SAME buffer.
    fill_pattern(shared.va, kP0, kInSize);
    fill_pattern(static_cast<char *>(shared.va) + kOff, kP1, kInSize);
    drv.mem_flush_cache(shared.fd);
    // Separate buffer holding only P1 at offset 0.
    fill_pattern(ctrl.va, kP1, kInSize);
    drv.mem_flush_cache(ctrl.fd);

    DevBuf in_a{shared.fd, 0, kInSize, 0};
    DevBuf in_b{shared.fd, kOff, kInSize, 0};
    DevBuf in_c{ctrl.fd, 0, kInSize, 0};
    DevBuf out_a{oa.fd, 0, kOutSize, 0};
    DevBuf out_b{ob.fd, 0, kOutSize, 0};
    DevBuf out_c{oc.fd, 0, kOutSize, 0};

    std::printf("\n--- three runs ---\n");
    bool ok = run(drv, model_id, in_a, out_a, "A offset=0 P0") &&
              run(drv, model_id, in_b, out_b, "B offset=4096 P1") &&
              run(drv, model_id, in_c, out_c, "C separate buf P1");
    if (!ok) return 1;

    // Outputs were written by the NPU; invalidate our cached mapping before reading.
    drv.mem_flush_cache(oa.fd);
    drv.mem_flush_cache(ob.fd);
    drv.mem_flush_cache(oc.fd);

    const bool ab_same = std::memcmp(oa.va, ob.va, kOutSize) == 0;
    const bool bc_same = std::memcmp(ob.va, oc.va, kOutSize) == 0;
    const bool ac_same = std::memcmp(oa.va, oc.va, kOutSize) == 0;

    std::printf("\n--- results ---\n");
    std::printf("  mean|A|=%.3f  mean|B|=%.3f  mean|C|=%.3f\n",
                mean_abs(oa.va, kOutSize), mean_abs(ob.va, kOutSize), mean_abs(oc.va, kOutSize));
    std::printf("  A == B : %s\n", ab_same ? "yes" : "no");
    std::printf("  B == C : %s\n", bc_same ? "yes" : "no");
    std::printf("  A == C : %s\n", ac_same ? "yes" : "no");

    std::printf("\n--- verdict ---\n");
    if (!ab_same && bc_same) {
        std::printf("  OFFSET HONOURED: run B read P1 from offset %zu of the shared buffer\n"
                    "                   and matched the separate-buffer control C.\n"
                    "  => MoE approach (b) is viable: one resident expert blob per die,\n"
                    "     expert selection via DevBuf.offset.\n", kOff);
    } else if (ab_same && !ac_same) {
        std::printf("  OFFSET IGNORED: runs A and B produced identical output, i.e. B read\n"
                    "                   offset 0 (P0), not offset %zu (P1).\n"
                    "  => MoE approach (b) via DevBuf.offset is NOT viable; would need one\n"
                    "     buffer per expert.\n", kOff);
    } else {
        std::printf("  INCONCLUSIVE: unexpected combination (A==B:%d B==C:%d A==C:%d).\n"
                    "  The model may be insensitive to a constant input, or output buffers\n"
                    "  were not written. Check mean values above and re-probe with a\n"
                    "  non-constant input pattern.\n",
                    (int)ab_same, (int)bc_same, (int)ac_same);
    }

    drv.unload_model(model_id);
    return 0;
}
