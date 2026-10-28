/**
 * @file probe_prepare_scope.cpp
 * @brief Does a sliced DevBuf have to be PREPARED as that slice, or does preparing the whole
 *        buffer cover any offset/size submitted inside it?
 *
 * Milestone 13 assumed the cheap answer ("the SMMU maps mem_fd, not a range"), but every
 * silicon proof so far (probe_dynamic_weight, probe_weight_bitmatch) prepared exactly the
 * {fd, offset, size} triple it then submitted, so the assumption was never tested. A board run
 * through WorkerPool returned four byte-identical outputs for two different weight sets, which
 * is what a silently-unbound weight tensor looks like. This settles it in one process with the
 * known-good harness settings (direct driver, CacheMode::Cached, explicit flushes), so nothing
 * but the prepare scope can explain the result.
 *
 *   sudo ./probe_prepare_scope <static.model> <classb.model> <weights.bin> [in out weight]
 *
 * Runs, one variable each (defaults are the INT8 DeepSeek-V2-Lite expert FFN):
 *   golden : static loadable, weights baked in                        -> reference bytes
 *   refA   : classb, A bound whole in its own buffer                  -> must equal golden
 *   refB   : classb, A' (page-rotated, distinct bytes) bound whole    -> the answer for A'
 *   W0/WW  : blob [A | A'] prepared ONCE as the whole 2x range, then submitted as slice @0
 *            and slice @W                                             -> expect golden / refB
 *   P0/PW  : the same two slices, each range prepared explicitly      -> expect golden / refB
 *   over   : refA with an activation descriptor bigger than the tensor (page-rounded buffer)
 *
 * If WW yields golden while PW yields refB, the prepared range is what binds a slice, and any
 * slicing API must prepare per slice.
 */

#include "../src/backend/driver_wrapper.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace eswin;

namespace {

size_t kInSize = 2048;         // INT8 expert FFN activation
size_t kOutSize = 4096;        // its output
size_t kWeightSize = 8908800;  // one expert's Class B region (page-aligned)

struct Buf {
    uint64_t fd = 0;
    void *va = nullptr;
    size_t size = 0;
};

Buf alloc(EswinDriver &drv, size_t size, const char *what) {
    Buf b{};
    b.size = size;
    if (drv.mem_alloc(&b.fd, size, CacheMode::Cached, "mmz_nid_0_part_0") != 0) {
        std::printf("FAIL: mem_alloc %s (%zu bytes)\n", what, size);
        return b;
    }
    b.va = drv.mem_mmap(b.fd, size, CacheMode::Cached);
    if (!b.va) { std::printf("FAIL: mem_mmap %s\n", what); return b; }
    std::memset(b.va, 0, size);
    return b;
}

void fill_activation(void *dst, size_t bytes) {
    static const uint8_t kPat[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x01};
    uint8_t *p = static_cast<uint8_t *>(dst);
    for (size_t i = 0; i < bytes; ++i) p[i] = kPat[i % 8];
}

bool read_file(std::vector<uint8_t> &dst, const char *path) {
    FILE *f = std::fopen(path, "rb");
    if (!f) { std::printf("FAIL: cannot open %s\n", path); return false; }
    std::fseek(f, 0, SEEK_END);
    const size_t n = static_cast<size_t>(std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    dst.assign(n, 0);
    const size_t got = std::fread(dst.data(), 1, n, f);
    std::fclose(f);
    if (got != n) { std::printf("FAIL: short read %s\n", path); return false; }
    return true;
}

/**
 * @brief Prepares the given ranges (nullptr weight => do not prepare one, so the run relies on
 *        an earlier whole-buffer prepare) and submits the given descriptors.
 *
 * Prepared and submitted triples are separate arguments on purpose: their difference is the
 * variable under test.
 */
bool run_case(EswinDriver &drv, uint32_t model_id, const DevBuf &act, const DevBuf *w_prep,
              const DevBuf &w_sub, const DevBuf &out, const Buf &out_store, const char *label) {
    DevBuf p_act = act;
    if (drv.prepare_dev_buf(&p_act) != 0) {
        std::printf("  [%-6s] prepare act FAILED\n", label);
        return false;
    }
    if (w_prep) {
        DevBuf p_w = *w_prep;
        if (drv.prepare_dev_buf(&p_w) != 0) {
            std::printf("  [%-6s] prepare weight FAILED\n", label);
            return false;
        }
    }
    DevBuf p_out = out;
    if (drv.prepare_dev_buf(&p_out) != 0) {
        std::printf("  [%-6s] prepare out FAILED\n", label);
        return false;
    }

    NpuTask task;
    std::memset(&task, 0, sizeof(task));
    task.taskId = 1;
    task.modelId = model_id;
    task.inputFdNum = w_sub.mem_fd ? 2 : 1;   // static loadable takes only the activation
    task.inputFd[0] = act;
    if (w_sub.mem_fd) task.inputFd[1] = w_sub;
    task.outputFdNum = 1;
    task.outputFd[0] = out;

    const int rc = drv.submit_task(&task, 1);
    if (rc != 0) {
        std::printf("  [%-6s] submit FAILED rc=%d (%s)\n", label, rc,
                    EswinDriver::decode_error(rc).c_str());
        return false;
    }
    drv.mem_flush_cache(out_store.fd);
    std::printf("  [%-6s] submitted inputFdNum=%u weight{fd=0x%llx off=%zu size=%zu}%s\n", label,
                static_cast<unsigned>(task.inputFdNum), (unsigned long long)w_sub.mem_fd,
                w_sub.offset, w_sub.size, w_prep ? "" : " (weight range NOT prepared here)");
    return true;
}

std::string fingerprint(const Buf &b) {
    const uint8_t *p = static_cast<const uint8_t *>(b.va);
    size_t nz = 0;
    unsigned long long sum = 0;
    for (size_t i = 0; i < b.size; ++i) { if (p[i]) ++nz; sum += p[i]; }
    char buf[128];
    std::snprintf(buf, sizeof(buf), "nonzero=%zu/%zu sum=%llu b0=%02x", nz, b.size, sum % 100000000,
                  p[0]);
    return buf;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::printf("usage: %s <static.model> <classb.model> <weights.bin> [in out weight]\n"
                    "  defaults: in=2048 out=4096 weight=8908800 (INT8 V2-Lite expert FFN)\n",
                    argv[0]);
        return 1;
    }
    if (argc == 8) {
        kInSize = std::strtoull(argv[4], nullptr, 10);
        kOutSize = std::strtoull(argv[5], nullptr, 10);
        kWeightSize = std::strtoull(argv[6], nullptr, 10);
        if (!kInSize || !kOutSize || !kWeightSize || kWeightSize % 4096) {
            std::printf("FAIL: bad geometry (weight size must be 4096-aligned)\n");
            return 1;
        }
        std::printf("geometry: in=%zu out=%zu weight=%zu\n", kInSize, kOutSize, kWeightSize);
    }

    std::vector<uint8_t> a;
    if (!read_file(a, argv[3]) || a.size() != kWeightSize) {
        std::printf("FAIL: weights file is %zu bytes, geometry says %zu\n", a.size(), kWeightSize);
        return 1;
    }
    std::vector<uint8_t> b(kWeightSize);  // A rotated one page: distinct bytes, same size
    std::memcpy(b.data(), a.data() + 4096, kWeightSize - 4096);
    std::memcpy(b.data() + kWeightSize - 4096, a.data(), 4096);

    EswinDriver &drv = EswinDriver::instance();
    if (!drv.initialize()) { std::printf("SKIP: no NPU hardware\n"); return 2; }
    if (drv.set_device(0) != 0) { std::printf("FAIL: set_device(0)\n"); return 1; }

    const Buf act = alloc(drv, kInSize, "act");
    const Buf wA = alloc(drv, kWeightSize, "whole A");
    const Buf wB = alloc(drv, kWeightSize, "whole B");
    const Buf blob = alloc(drv, 2 * kWeightSize, "blob A|B");
    Buf out_g = alloc(drv, kOutSize, "out golden");
    Buf out_a = alloc(drv, kOutSize, "out refA");
    Buf out_b = alloc(drv, kOutSize, "out refB");
    Buf out_w0 = alloc(drv, kOutSize, "out W0");
    Buf out_ww = alloc(drv, kOutSize, "out WW");
    Buf out_p0 = alloc(drv, kOutSize, "out P0");
    Buf out_pw = alloc(drv, kOutSize, "out PW");
    Buf out_ov = alloc(drv, kOutSize, "out over");
    const Buf *all[] = {&act, &wA, &wB, &blob, &out_g, &out_a, &out_b, &out_w0, &out_ww,
                        &out_p0, &out_pw, &out_ov};
    for (const Buf *x : all) {
        if (!x->fd || !x->va) return 1;
    }

    fill_activation(act.va, kInSize);
    std::memcpy(wA.va, a.data(), kWeightSize);
    std::memcpy(wB.va, b.data(), kWeightSize);
    std::memcpy(blob.va, a.data(), kWeightSize);
    std::memcpy(static_cast<uint8_t *>(blob.va) + kWeightSize, b.data(), kWeightSize);
    drv.mem_flush_cache(act.fd);
    drv.mem_flush_cache(wA.fd);
    drv.mem_flush_cache(wB.fd);
    drv.mem_flush_cache(blob.fd);

    auto outd = [](const Buf &b) { return DevBuf{b.fd, 0, kOutSize, 0}; };
    const DevBuf A_act{act.fd, 0, kInSize, 0};
    const DevBuf wA_sub{wA.fd, 0, kWeightSize, 0};
    const DevBuf wB_sub{wB.fd, 0, kWeightSize, 0};
    const DevBuf slice0{blob.fd, 0, kWeightSize, 0};
    const DevBuf sliceW{blob.fd, kWeightSize, kWeightSize, 0};
    const DevBuf whole_blob{blob.fd, 0, 2 * kWeightSize, 0};

    uint32_t id_static = 0, id_classb = 0;
    if (drv.load_model(&id_static, argv[1]) != 0) { std::printf("FAIL: load static\n"); return 1; }
    std::printf("static model_id=%u\n", id_static);
    if (!run_case(drv, id_static, A_act, nullptr, DevBuf{0, 0, 0, 0}, outd(out_g), out_g, "golden"))
        return 1;
    drv.unload_model(id_static);

    if (drv.load_model(&id_classb, argv[2]) != 0) { std::printf("FAIL: load classb\n"); return 1; }
    std::printf("classb model_id=%u\n\n", id_classb);

    bool ok = true;
    ok &= run_case(drv, id_classb, A_act, &wA_sub, wA_sub, outd(out_a), out_a, "refA");
    ok &= run_case(drv, id_classb, A_act, &wB_sub, wB_sub, outd(out_b), out_b, "refB");

    // Whole-blob prepare, then two slice submits that do NOT prepare their own range.
    DevBuf prep = whole_blob;
    if (drv.prepare_dev_buf(&prep) != 0) { std::printf("FAIL: prepare whole blob\n"); return 1; }
    std::printf("  prepared whole blob once: fd=0x%llx offset=0 size=%zu\n",
                (unsigned long long)blob.fd, 2 * kWeightSize);
    ok &= run_case(drv, id_classb, A_act, nullptr, slice0, outd(out_w0), out_w0, "W0");
    ok &= run_case(drv, id_classb, A_act, nullptr, sliceW, outd(out_ww), out_ww, "WW");

    // The same two slices, each range prepared explicitly.
    ok &= run_case(drv, id_classb, A_act, &slice0, slice0, outd(out_p0), out_p0, "P0");
    ok &= run_case(drv, id_classb, A_act, &sliceW, sliceW, outd(out_pw), out_pw, "PW");

    // Over-describe the activation: page-rounded buffer, tensor is smaller.
    const size_t over = ((kInSize + 4095) / 4096) * 4096;
    const Buf act_big = alloc(drv, over, "act big");
    if (!act_big.fd) return 1;
    std::memcpy(act_big.va, act.va, kInSize);
    drv.mem_flush_cache(act_big.fd);
    const DevBuf big_act{act_big.fd, 0, over, 0};
    ok &= run_case(drv, id_classb, big_act, &wA_sub, wA_sub, outd(out_ov), out_ov, "over");

    drv.unload_model(id_classb);
    if (!ok) { std::printf("\nFAIL: a case did not execute\n"); return 1; }

    std::printf("\n--- fingerprints ---\n");
    const Buf *fps[] = {&out_g, &out_a, &out_b, &out_w0, &out_ww, &out_p0, &out_pw, &out_ov};
    const char *names[] = {"golden", "refA", "refB", "W0", "WW", "P0", "PW", "over"};
    for (size_t i = 0; i < sizeof(fps) / sizeof(fps[0]); ++i)
        std::printf("  %-6s %s\n", names[i], fingerprint(*fps[i]).c_str());

    struct Pair { const char *la; const Buf *a; const char *lb; const Buf *b; bool same; };
    const Pair pairs[] = {
        {"golden", &out_g, "refA", &out_a, true},
        {"golden", &out_g, "W0", &out_w0, true},
        {"golden", &out_g, "P0", &out_p0, true},
        {"refB", &out_b, "WW", &out_ww, true},
        {"refB", &out_b, "PW", &out_pw, true},
        {"golden", &out_g, "refB", &out_b, false},
        {"W0", &out_w0, "WW", &out_ww, false},
        {"P0", &out_p0, "PW", &out_pw, false},
        {"golden", &out_g, "over", &out_ov, true},
    };
    int fails = 0;
    std::printf("\n--- expectations ---\n");
    for (const Pair &p : pairs) {
        const int c = std::memcmp(p.a->va, p.b->va, kOutSize);
        const bool pass = p.same ? (c == 0) : (c != 0);
        if (!pass) ++fails;
        std::string tag = std::string(p.la) + " vs " + p.lb + (p.same ? " [equal]" : " [differ]");
        std::printf("  %-32s %s\n", tag.c_str(), pass ? "ok" : "VIOLATED");
    }

    const bool whole_ok = std::memcmp(out_g.va, out_w0.va, kOutSize) == 0 &&
                          std::memcmp(out_b.va, out_ww.va, kOutSize) == 0;
    const bool slice_ok = std::memcmp(out_g.va, out_p0.va, kOutSize) == 0 &&
                          std::memcmp(out_b.va, out_pw.va, kOutSize) == 0;
    std::printf("\n--- verdict ---\n");
    std::printf("  prepare WHOLE  + submit slice : %s\n",
                whole_ok ? "binds the right bytes" : "does NOT bind (slice ignored)");
    std::printf("  prepare SLICE  + submit slice : %s\n",
                slice_ok ? "binds the right bytes" : "does NOT bind");
    std::printf("  over-sized activation descriptor : %s\n",
                std::memcmp(out_g.va, out_ov.va, kOutSize) == 0 ? "harmless" : "CHANGES the result");
    std::printf("\n%s\n", fails == 0 ? "ALL EXPECTATIONS MET"
                                     : (std::to_string(fails) + " EXPECTATION(S) VIOLATED").c_str());
    return fails == 0 ? 0 : 3;
}
