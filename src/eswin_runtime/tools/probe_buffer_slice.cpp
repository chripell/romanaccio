/**
 * @file probe_buffer_slice.cpp
 * @brief Milestone 13 on-board acceptance: does a DmaBufferSlice bind the right bytes?
 *
 * The unit tests (tests/test_buffer_slice.cpp) pin the marshalling, but only silicon can
 * answer the question that matters for MoE dispatch: when one resident blob is submitted as
 * two different slices, does the NPU actually read the two different windows?
 *
 * Method — no second expert compilation needed. Bind the same weight region two ways and
 * require the results to agree:
 *   ref A : weights in their own buffer, bound whole (the legacy pointer overload, so the
 *           adapter that every validated caller now goes through is itself re-verified)
 *   ref B : a byte-rotated copy of the region, bound whole (a stand-in for "expert B")
 *   sl A  : the 2x blob, slice at offset 0
 *   sl B  : the same blob, slice at offset W
 * Expect sl A == ref A, sl B == ref B, ref A != ref B. If the offset were ignored, sl B would
 * silently reproduce ref A and the second equality would fail -- "outputs differ" alone is
 * also what garbage looks like, which is why the whole-buffer runs are the reference.
 *
 * All buffers are CacheMode::NoCache on purpose: the point of comparison is which BYTES the
 * NPU read, and Cached MMZ would confound that with the unresolved flush-direction question
 * (audit finding 1.1). Output is 1152 bytes, so the slow-CPU-read penalty is irrelevant here.
 *
 *   sudo ./probe_buffer_slice <classb.model> <weights.bin> [mmz_zone]
 *
 * weights.bin is the extracted Class B region (INT8 DeepSeek-V2-Lite expert: 8,908,800 bytes;
 * fp16 reference GEMM: 1,769,472). Must be a multiple of 4096, since sub-page offsets are
 * what WorkerPool::submit_task deliberately refuses.
 */

#include "../src/backend/dma_buffer.hpp"
#include "../src/backend/worker_pool.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace eswin;

namespace {

struct Run {
    const char *label;
    bool pointer_api;   // true = legacy vector<DmaBuffer*> overload
    std::vector<DmaBuffer*> in_ptrs, out_ptrs;
    std::vector<DmaBufferSlice> in_slices, out_slices;
};

bool run_one(WorkerPool &pool, uint32_t model_id, Run &r, uint64_t *elapsed) {
    bool ok = r.pointer_api
                  ? pool.submit_task(0, model_id, r.in_ptrs, r.out_ptrs, elapsed)
                  : pool.submit_task(0, model_id, r.in_slices, r.out_slices, elapsed);
    std::printf("  [%s] submit %s (%.3f ms)\n", r.label, ok ? "OK" : "FAILED",
                static_cast<double>(*elapsed) / 1e6);
    return ok;
}

int cmp_bufs(const DmaBuffer &a, const DmaBuffer &b, size_t bytes, const char *la,
             const char *lb, const char *what) {
    const int c = std::memcmp(a.data(), b.data(), bytes);
    std::printf("  %s vs %s: %s\n", la, lb, c == 0 ? "byte-identical" : "different");
    if (c == 0 && std::strcmp(what, "differ") == 0) return 1;
    if (c != 0 && std::strcmp(what, "equal") == 0) {
        const uint8_t *pa = static_cast<const uint8_t *>(a.data());
        const uint8_t *pb = static_cast<const uint8_t *>(b.data());
        size_t i = 0;
        while (i < bytes && pa[i] == pb[i]) ++i;
        std::printf("    first difference at byte %zu: 0x%02x vs 0x%02x\n", i, pa[i], pb[i]);
        return 1;
    }
    return 0;
}

/// What a range actually holds, so an all-zero or unexpectedly-shared result is visible instead
/// of having to be inferred from a comparison.
void fingerprint(const char *label, const void *v, size_t bytes) {
    const uint8_t *p = static_cast<const uint8_t *>(v);
    size_t nz = 0;
    unsigned long long sum = 0;
    for (size_t i = 0; i < bytes; ++i) { if (p[i]) ++nz; sum += p[i]; }
    std::printf("  %-10s nonzero=%zu/%zu sum=%llu first4=%02x %02x %02x %02x\n", label, nz, bytes,
                sum % 100000000, p[0], bytes > 1 ? p[1] : 0, bytes > 2 ? p[2] : 0,
                bytes > 3 ? p[3] : 0);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::printf("usage: %s <classb.model> <weights.bin> [mmz_zone]\n"
                    "  e.g. %s /tmp/i8/expert_int8_classb.model /tmp/i8/expert_int8_weights_mem1.bin\n",
                    argv[0], argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *weights_path = argv[2];
    const char *zone = (argc > 3) ? argv[3] : "mmz_nid_0_part_0";
    // "cached" reproduces the harness that already passed the direct-driver probes; the
    // default NoCache removes CPU-cache coherence from the question entirely.
    CacheMode mode = CacheMode::NoCache;
    if (argc > 4 && std::strcmp(argv[4], "cached") == 0) mode = CacheMode::Cached;
    std::printf("cache mode: %s\n", mode == CacheMode::Cached ? "Cached" : "NoCache");

    FILE *f = std::fopen(weights_path, "rb");
    if (!f) { std::printf("FAIL: cannot open %s\n", weights_path); return 1; }
    std::fseek(f, 0, SEEK_END);
    const long wlong = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    const size_t W = static_cast<size_t>(wlong);
    std::vector<uint8_t> host_a(W);
    if (std::fread(host_a.data(), 1, W, f) != W) {
        std::printf("FAIL: short read of %s\n", weights_path);
        std::fclose(f);
        return 1;
    }
    std::fclose(f);
    if (W == 0 || W % 4096 != 0) {
        std::printf("FAIL: weight region %zu bytes is empty or not page-aligned\n", W);
        return 1;
    }
    std::printf("weight region: %zu bytes (one expert)\n", W);

    // "Expert B" = the region cyclically rotated by one page. Same size, guaranteed distinct
    // bytes, no second EsAAC compile. What is under test is the binding, not the arithmetic.
    std::vector<uint8_t> host_b(W);
    std::memcpy(host_b.data(), host_a.data() + 4096, W - 4096);
    std::memcpy(host_b.data() + W - 4096, host_a.data(), 4096);

    WorkerPool pool;
    if (!pool.start(1)) { std::printf("FAIL: WorkerPool::start(1) -- run with sudo\n"); return 1; }

    ModelInfo info;
    if (!pool.load_model(0, model_path, info)) {
        std::printf("FAIL: load_model('%s')\n", model_path);
        return 1;
    }
    const size_t in_bytes = info.input_sizes[0];
    const size_t out_bytes = info.output_sizes[0];
    if (in_bytes == 0 || out_bytes == 0) {
        std::printf("FAIL: model reports in=%zu out=%zu\n", in_bytes, out_bytes);
        return 1;
    }
    std::printf("model_id=%u inputs=%u outputs=%u  in=%zu out=%zu",
                info.model_id, info.num_inputs, info.num_outputs, in_bytes, out_bytes);
    if (info.num_inputs > 1 && info.input_sizes[1] != 0 && info.input_sizes[1] != W) {
        std::printf("  (WARNING: slot 1 declares %zu, file has %zu)", info.input_sizes[1], W);
    }
    std::printf("\n");

    DmaBuffer act(in_bytes, mode, zone);
    DmaBuffer whole_a(W, mode, zone), whole_b(W, mode, zone);
    DmaBuffer blob(2 * W, mode, zone);
    DmaBuffer out_ref_a(out_bytes, mode, zone);
    DmaBuffer out_ref_b(out_bytes, mode, zone);
    DmaBuffer out_sl_a(out_bytes, mode, zone);
    DmaBuffer out_sl_b(out_bytes, mode, zone);
    for (const DmaBuffer *b : {&act, &whole_a, &whole_b, &blob, &out_ref_a, &out_ref_b,
                               &out_sl_a, &out_sl_b}) {
        if (!b->is_valid()) { std::printf("FAIL: allocation in zone %s\n", zone); return 1; }
    }

    // The same varied pattern the direct-driver control uses (probe_prepare_scope). A uniform
    // fill made every run of the INT8 expert FFN return all zeros, which is indistinguishable
    // from a binding failure -- do not "simplify" this back to memset.
    static const uint8_t kPat[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x01};
    {
        uint8_t *p = static_cast<uint8_t *>(act.data());
        for (size_t i = 0; i < in_bytes; ++i) p[i] = kPat[i % 8];
    }
    std::memcpy(whole_a.data(), host_a.data(), W);
    std::memcpy(whole_b.data(), host_b.data(), W);
    std::memcpy(blob.data(), host_a.data(), W);
    std::memcpy(static_cast<uint8_t *>(blob.data()) + W, host_b.data(), W);

    for (DmaBuffer *b : {&act, &whole_a, &whole_b, &blob, &out_ref_a, &out_ref_b, &out_sl_a,
                        &out_sl_b}) {
        b->flush_cache();
        if (!pool.prepare_buffer(0, *b)) {
            std::printf("FAIL: prepare_buffer(fd 0x%llx)\n", (unsigned long long)b->mem_fd());
            return 1;
        }
    }
    std::printf("prepared %zu buffers on die 0 (blob fd=0x%llx, %zu bytes)\n", size_t{8},
                (unsigned long long)blob.mem_fd(), blob.size());

    // --- the four binds ---------------------------------------------------------
    Run ref_a{"ref whole A", true, {&act, &whole_a}, {&out_ref_a}, {}, {}};
    Run ref_b{"ref whole B", true, {&act, &whole_b}, {&out_ref_b}, {}, {}};
    Run sla{"slice A @0", false, {}, {}, {DmaBufferSlice{&act}, DmaBufferSlice{&blob, 0, W}},
            {DmaBufferSlice{&out_sl_a}}};
    Run slb{"slice B @W", false, {}, {}, {DmaBufferSlice{&act}, DmaBufferSlice{&blob, W, W}},
            {DmaBufferSlice{&out_sl_b}}};

    int fails = 0;
    for (Run *r : {&ref_a, &ref_b, &sla, &slb}) {
        uint64_t elapsed = 0;
        if (!run_one(pool, info.model_id, *r, &elapsed)) ++fails;
    }
    if (fails) { std::printf("FAIL: %d submit(s) rejected\n", fails); return 1; }

    for (DmaBuffer *b : {&out_ref_a, &out_ref_b, &out_sl_a, &out_sl_b}) b->flush_cache();
    std::printf("\n--- what each run produced ---\n");
    fingerprint("ref A", out_ref_a.data(), out_bytes);
    fingerprint("ref B", out_ref_b.data(), out_bytes);
    fingerprint("slice A", out_sl_a.data(), out_bytes);
    fingerprint("slice B", out_sl_b.data(), out_bytes);
    std::printf("  (inputs, to see whether the NPU touched them)\n");
    fingerprint("act", act.data(), in_bytes);
    fingerprint("blob@0", blob.data(), W);
    fingerprint("blob@W", static_cast<const uint8_t *>(blob.data()) + W, W);

    std::printf("\n--- comparisons (%zu output bytes each) ---\n", out_bytes);
    fails += cmp_bufs(out_ref_a, out_sl_a, out_bytes, "ref A ", "slice A", "equal");
    fails += cmp_bufs(out_ref_b, out_sl_b, out_bytes, "ref B ", "slice B", "equal");
    fails += cmp_bufs(out_ref_a, out_ref_b, out_bytes, "ref A ", "ref B ", "differ");
    fails += cmp_bufs(out_sl_a, out_sl_b, out_bytes, "slice A", "slice B", "differ");

    // --- rejections must be side-effect-free ------------------------------------
    std::printf("\n--- malformed slices (all must be refused before the worker sees them) ---\n");
    DmaBuffer out_bad(out_bytes, mode, zone);
    if (!out_bad.is_valid() || !pool.prepare_buffer(0, out_bad)) {
        std::printf("FAIL: prepare out_bad\n");
        return 1;
    }
    struct Bad { const char *what; std::vector<DmaBufferSlice> in; };
    std::vector<DmaBufferSlice> outs{DmaBufferSlice{&out_bad}};
    const std::vector<Bad> bad = {
        {"offset not page-aligned",
         {DmaBufferSlice{&act}, DmaBufferSlice{&blob, 100, W}}},
        {"slice runs past the buffer",
         {DmaBufferSlice{&act}, DmaBufferSlice{&blob, W, 2 * W}}},
        {"offset at the very end",
         {DmaBufferSlice{&act}, DmaBufferSlice{&blob, 2 * W, W}}},
        {"null buffer", {DmaBufferSlice{&act}, DmaBufferSlice{nullptr, 0, W}}},
    };
    for (const Bad &b : bad) {
        uint64_t e = 0;
        const bool accepted = pool.submit_task(0, info.model_id, b.in, outs, &e);
        std::printf("  %-30s -> %s\n", b.what, accepted ? "ACCEPTED (WRONG)" : "rejected");
        if (accepted) ++fails;
    }
    std::vector<DmaBufferSlice> too_many(kIpcBufferSlots + 1, DmaBufferSlice{&act});
    {
        uint64_t e = 0;
        const bool accepted = pool.submit_task(0, info.model_id, too_many, outs, &e);
        std::printf("  %-30s -> %s\n", "more inputs than driver slots",
                    accepted ? "ACCEPTED (WRONG)" : "rejected");
        if (accepted) ++fails;
    }

    uint64_t elapsed = 0;
    Run again{"slice B again", false, {}, {},
              {DmaBufferSlice{&act}, DmaBufferSlice{&blob, W, W}}, {DmaBufferSlice{&out_sl_b}}};
    if (!run_one(pool, info.model_id, again, &elapsed)) ++fails;
    fails += cmp_bufs(out_ref_b, out_sl_b, out_bytes, "ref B ", "slice B'", "equal");

    for (DmaBuffer *b : {&act, &whole_a, &whole_b, &blob, &out_ref_a, &out_ref_b, &out_sl_a,
                        &out_sl_b, &out_bad}) {
        pool.unprepare_buffer(0, *b);
    }
    pool.unload_model(0, info.model_id);
    pool.shutdown();

    std::printf("\n--- verdict ---\n");
    if (fails == 0) {
        std::printf("  PASS: a DmaBufferSlice binds exactly the bytes the same content bound\n"
                    "  whole, the two experts in one resident blob are distinguishable, and\n"
                    "  every malformed slice is refused without disturbing the working path.\n"
                    "  MoE expert dispatch can be built on this.\n");
        return 0;
    }
    std::printf("  FAIL: %d problem(s) above -- do not build expert dispatch on slicing yet.\n",
                fails);
    return 3;
}
