/**
 * @file probe_weight_bitmatch.cpp
 * @brief Does a Class B (runtime-bound) weight produce the SAME result as the baked-in blob?
 *
 * FINDINGS §27 proved a loadable reads weights from a submit-time buffer and that changing
 * DevBuf.offset changes the output. What it did not prove is that the arithmetic is correct —
 * "outputs differ" is also what garbage looks like. This closes that gap.
 *
 * Method: run the unmodified static loadable (weights baked into blobs) and the patched
 * Class B loadable (weight blobs deleted, supplied at submit) against the SAME activation,
 * feeding the dynamic one the original three WsplitM_* blobs concatenated in the exact
 * layout the Class A memory entry declared. If the binding is faithful the two outputs must
 * be byte-identical.
 *
 *   ./probe_weight_bitmatch <static.model> <dynamic.model> <weights.bin>
 *
 * weights.bin is produced by deep/artifacts/scratch extraction: 1,769,472 bytes, being
 * WsplitM_0_0 @0, WsplitM_1_1 @589824, WsplitM_2_2 @1179648.
 *
 * The models are loaded one at a time (load, submit, unload) so a failed comparison cannot
 * be blamed on two models being resident at once.
 */

#include "../src/backend/driver_wrapper.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace eswin;

namespace {

constexpr size_t kInSize = 1152;         // 576 x fp16
constexpr size_t kOutSize = 3072;        // 1536 x fp16
constexpr size_t kWeightSize = 1769472;  // 3 x 589824

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
    if (!b.va) { std::printf("FAIL: mem_mmap %s\n", what); return false; }
    std::memset(b.va, 0, size);
    drv.mem_flush_cache(b.fd);
    return true;
}

// A varied, deterministic activation so the GEMM is genuinely exercised rather than
// multiplying a constant by a constant.
void fill_activation(void *dst, size_t bytes) {
    static const uint16_t kPat[] = {0x3800, 0x3C00, 0x3555, 0xBC00, 0x4200, 0xB800};
    uint16_t *p = static_cast<uint16_t *>(dst);
    for (size_t i = 0; i < bytes / 2; ++i) p[i] = kPat[i % 6];
}

bool load_file_into(void *dst, size_t cap, const char *path, size_t *out_len) {
    FILE *f = std::fopen(path, "rb");
    if (!f) { std::printf("FAIL: cannot open %s\n", path); return false; }
    size_t n = std::fread(dst, 1, cap, f);
    std::fclose(f);
    if (n != cap) {
        std::printf("FAIL: %s is %zu bytes, expected %zu\n", path, n, cap);
        return false;
    }
    if (out_len) *out_len = n;
    return true;
}

// Runs one model. weight_fd == 0 means the weight is baked in (static loadable).
bool run_model(EswinDriver &drv, const char *path, const Buf &act, const Buf &w,
               uint64_t weight_fd, Buf &out, const char *label) {
    uint32_t model_id = 0;
    int rc = drv.load_model(&model_id, path);
    if (rc != 0) {
        std::printf("FAIL: load_model('%s') rc=%d (%s)\n", path, rc,
                    EswinDriver::decode_error(rc).c_str());
        return false;
    }

    DevBuf in_act{act.fd, 0, kInSize, 0};
    DevBuf o{out.fd, 0, kOutSize, 0};
    if (drv.prepare_dev_buf(&in_act) != 0 || drv.prepare_dev_buf(&o) != 0) {
        std::printf("FAIL: [%s] prepare activation/output\n", label);
        drv.unload_model(model_id);
        return false;
    }

    NpuTask task;
    std::memset(&task, 0, sizeof(task));
    task.taskId = 1;
    task.modelId = model_id;
    task.outputFdNum = 1;
    task.outputFd[0] = o;
    task.inputFd[0] = in_act;

    if (weight_fd != 0) {
        DevBuf in_w{weight_fd, 0, kWeightSize, 0};
        if (drv.prepare_dev_buf(&in_w) != 0) {
            std::printf("FAIL: [%s] prepare weight\n", label);
            drv.unload_model(model_id);
            return false;
        }
        task.inputFdNum = 2;          // slot 0 activation, slot 1 runtime-bound weight
        task.inputFd[1] = in_w;
    } else {
        task.inputFdNum = 1;          // weight is baked into the loadable
    }

    rc = drv.submit_task(&task, 1);
    if (rc != 0) {
        std::printf("FAIL: [%s] submit rc=%d (%s)\n", label, rc,
                    EswinDriver::decode_error(rc).c_str());
        drv.unload_model(model_id);
        return false;
    }
    std::printf("  [%s] model_id=%u inputFdNum=%u submit ok\n", label, model_id,
                static_cast<unsigned>(task.inputFdNum));
    drv.unload_model(model_id);
    return true;
}

double mean_u16(const void *p, size_t bytes) {
    const uint16_t *v = static_cast<const uint16_t *>(p);
    double s = 0.0;
    for (size_t i = 0; i < bytes / 2; ++i) s += static_cast<double>(v[i]);
    return s / static_cast<double>(bytes / 2);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::printf("usage: %s <static.model> <dynamic.model> <weights.bin>\n", argv[0]);
        return 1;
    }
    const char *static_model = argv[1];
    const char *dynamic_model = argv[2];
    const char *weights_bin = argv[3];

    EswinDriver &drv = EswinDriver::instance();
    if (!drv.initialize()) { std::printf("SKIP: no NPU hardware\n"); return 2; }
    if (drv.set_device(0) != 0) { std::printf("FAIL: set_device(0)\n"); return 1; }

    Buf act, w, o_s, o_d;
    if (!alloc(drv, act, kInSize, "activation")) return 1;
    if (!alloc(drv, w, kWeightSize, "weight")) return 1;
    if (!alloc(drv, o_s, kOutSize, "out static")) return 1;
    if (!alloc(drv, o_d, kOutSize, "out dynamic")) return 1;

    fill_activation(act.va, kInSize);
    if (!load_file_into(w.va, kWeightSize, weights_bin, nullptr)) return 1;
    drv.mem_flush_cache(act.fd);
    drv.mem_flush_cache(w.fd);
    std::printf("loaded %zu bytes of original weights into the DMA blob\n", kWeightSize);

    std::printf("\n--- run 1: static loadable (weights baked into blobs) ---\n");
    if (!run_model(drv, static_model, act, w, 0, o_s, "static")) return 1;
    std::printf("--- run 2: Class B loadable (weights supplied at submit) ---\n");
    if (!run_model(drv, dynamic_model, act, w, w.fd, o_d, "dynamic")) return 1;

    drv.mem_flush_cache(o_s.fd);
    drv.mem_flush_cache(o_d.fd);

    const int cmp = std::memcmp(o_s.va, o_d.va, kOutSize);
    size_t first_diff = 0;
    if (cmp != 0) {
        const uint8_t *a = static_cast<const uint8_t *>(o_s.va);
        const uint8_t *b = static_cast<const uint8_t *>(o_d.va);
        while (first_diff < kOutSize && a[first_diff] == b[first_diff]) ++first_diff;
    }

    std::printf("\n--- results ---\n");
    std::printf("  mean|static | = %.4f\n", mean_u16(o_s.va, kOutSize));
    std::printf("  mean|dynamic| = %.4f\n", mean_u16(o_d.va, kOutSize));
    std::printf("  byte-identical over %zu bytes: %s\n", kOutSize, cmp == 0 ? "YES" : "NO");
    if (cmp != 0) {
        const uint16_t *a = static_cast<const uint16_t *>(o_s.va);
        const uint16_t *b = static_cast<const uint16_t *>(o_d.va);
        size_t el = first_diff / 2;
        std::printf("  first difference at byte %zu (element %zu): static=0x%04x dynamic=0x%04x\n",
                    first_diff, el, a[el], b[el]);
    }

    std::printf("\n--- verdict ---\n");
    if (cmp == 0) {
        std::printf("  BIT-EXACT: a Class B runtime-bound weight reproduces the baked-in\n"
                    "  blob result exactly. The emitter change is safe to build on.\n");
    } else {
        std::printf("  MISMATCH: the dynamic-weight path computed something different, so\n"
                    "  the binding is not faithful even though it is read. Do not build the\n"
                    "  emitter on it yet.\n");
    }
    return cmp == 0 ? 0 : 3;
}
