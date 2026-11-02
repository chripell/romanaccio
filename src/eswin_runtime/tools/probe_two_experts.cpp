/**
 * @file probe_two_experts.cpp
 * @brief Does one resident per-die blob + `DevBuf.offset` really select between two REAL
 *        DeepSeek-V2-Lite experts, INT8, through the WorkerPool slice API?
 *
 * §32 proved the mechanism with a page-rotated stand-in for "expert B". This uses the real
 * thing: layer 1 experts 0 and 1 of DeepSeek-V2-Lite (weights read from the 31.4 GB
 * checkpoint), each compiled by EsAAC with a synthesized `--quant-stats` table, converted to
 * Class B, and concatenated into one 17,817,600-byte per-die blob.
 *
 * The interesting question is not just "does offset select the expert" but **which loadable may
 * serve it**. Activation thresholds are baked into a loadable's command streams, and the weight
 * region carries requant constants derived from them (only 7,394 of 8,908,800 bytes, 0.083%,
 * differ between the same expert compiled with its own table vs a shared one). So expert 1 is
 * compiled twice: with its own calibration, and with expert 0's activations
 * (`make_quant_table.py --activations-from`). If the shared-table pair matches bit-exactly while
 * the mixed pair does not, then one loadable per layer serving all 64 experts requires one
 * common activation calibration — a hard constraint on the MoE pipeline, and a cheap one.
 *
 * Geometry (INT8 expert FFN, measured): input 2048 int8 codes, output 4096 bytes = 2048 fp16.
 *
 *   sudo ./probe_two_experts --static0 m --static1 m --static1s m \
 *        --classb0 m --classb1 m --classb1s m --blob two_expert_blob.bin [--w1own region.bin]
 */

#include "../src/backend/dma_buffer.hpp"
#include "../src/backend/worker_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace eswin;

namespace {

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {                                  // subnormal: renormalise
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; --exp; }
            bits = sign | (exp << 23) | ((man & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);  // inf / nan
    } else {
        bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

std::vector<float> decode_fp16(const std::vector<uint8_t> &b) {
    std::vector<float> v(b.size() / 2);
    for (size_t i = 0; i < v.size(); ++i) {
        uint16_t h;
        std::memcpy(&h, b.data() + 2 * i, 2);
        v[i] = half_to_float(h);
    }
    return v;
}

/// The INT8 expert FFN declares output (1,2048,1,1) INT8 = 2048 bytes inside a 4096-byte
/// page-padded surface (eswin-sim --info on the compiled loadable). Decoding the whole surface
/// as int8 codes is safe for these metrics: the padding is zero, and zeros contribute nothing
/// to a dot product or a norm, so cosine and max-diff are unaffected.
std::vector<float> decode_int8(const std::vector<uint8_t> &b) {
    std::vector<float> v(b.size());
    for (size_t i = 0; i < b.size(); ++i) v[i] = static_cast<float>(static_cast<int8_t>(b[i]));
    return v;
}

std::vector<float> decode_out(const std::vector<uint8_t> &b, const std::string &dtype) {
    return dtype == "fp16" ? decode_fp16(b) : decode_int8(b);
}

double cosine(const std::vector<float> &a, const std::vector<float> &b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    if (na == 0 || nb == 0) return 0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

double max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return m;
}

double mean_abs(const std::vector<float> &a) {
    double s = 0;
    for (float x : a) s += std::fabs(x);
    return a.empty() ? 0 : s / a.size();
}

struct Cfg {
    std::string static0, static1, static1s, classb0, classb1, classb1s, blob, w1own;
    size_t in = 2048, out = 4096, weight = 8908800;
    const char *zone = "mmz_nid_0_part_0";
    std::string out_dtype = "int8";   // INT8 graph; "fp16" for the fp16 reference loadables
    int load_repeats = 0;             // measure Class B load+unload cost
    int preload = 0;                  // how many Class B loadables stay resident at once
    bool same_class = false;          // the two loadables are byte-identical: assert, don't expect
};

bool read_file(std::vector<uint8_t> &dst, const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) { std::printf("FAIL: cannot open %s\n", path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END);
    const size_t n = static_cast<size_t>(std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    dst.assign(n, 0);
    const size_t got = std::fread(dst.data(), 1, n, f);
    std::fclose(f);
    if (got != n) { std::printf("FAIL: short read %s\n", path.c_str()); return false; }
    return true;
}

/// A varied, deterministic int8 activation. Uniform fills make this INT8 graph emit zeros
/// regardless of the weights (§32.3), which is indistinguishable from a binding failure.
void fill_activation(uint8_t *p, size_t bytes) {
    uint32_t s = 0x2545F491u;
    int8_t peak = 0;
    for (size_t i = 0; i < bytes; ++i) {
        s = s * 1664525u + 1013904223u;
        const int8_t v = static_cast<int8_t>((s >> 24) % 129 - 64);   // -64..64
        p[i] = static_cast<uint8_t>(v);
        if (std::abs(static_cast<int>(v)) > std::abs(static_cast<int>(peak))) peak = v;
    }
    std::printf("activation: %zu int8 codes, varied LCG pattern, peak |code| = %d "
                "(of 127)\n", bytes, static_cast<int>(std::abs(peak)));
}

} // namespace

int main(int argc, char **argv) {
    Cfg cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char *what) -> const char * {
            if (i + 1 >= argc) { std::printf("FAIL: %s needs a value\n", what); std::exit(1); }
            return argv[++i];
        };
        if (a == "--static0")        cfg.static0 = need("--static0");
        else if (a == "--static1")   cfg.static1 = need("--static1");
        else if (a == "--static1s")  cfg.static1s = need("--static1s");
        else if (a == "--classb0")   cfg.classb0 = need("--classb0");
        else if (a == "--classb1")   cfg.classb1 = need("--classb1");
        else if (a == "--classb1s")  cfg.classb1s = need("--classb1s");
        else if (a == "--blob")      cfg.blob = need("--blob");
        else if (a == "--w1own")     cfg.w1own = need("--w1own");
        else if (a == "--in")        cfg.in = std::strtoull(need("--in"), nullptr, 10);
        else if (a == "--out")       cfg.out = std::strtoull(need("--out"), nullptr, 10);
        else if (a == "--weight")    cfg.weight = std::strtoull(need("--weight"), nullptr, 10);
        else if (a == "--zone")      cfg.zone = need("--zone");
        else if (a == "--out-dtype") cfg.out_dtype = need("--out-dtype");
        else if (a == "--load-repeats") cfg.load_repeats = std::atoi(need("--load-repeats"));
        else if (a == "--preload")  cfg.preload = std::atoi(need("--preload"));
        else if (a == "--same-class") cfg.same_class = true;
        else { std::printf("FAIL: unknown option '%s'\n", a.c_str()); return 1; }
    }
    if (cfg.static0.empty() || cfg.static1s.empty() || cfg.classb0.empty() || cfg.blob.empty()) {
        std::printf("usage: %s --static0 M --static1 M --static1s M --classb0 M --classb1 M "
                    "--classb1s M --blob BLOB [--w1own REGION] [--in N] [--out N] [--weight N]\n",
                    argv[0]);
        return 1;
    }
    if (cfg.weight % 4096 != 0) {
        std::printf("FAIL: weight region %zu is not page-aligned\n", cfg.weight);
        return 1;
    }

    std::vector<uint8_t> blob_file;
    if (!read_file(blob_file, cfg.blob) || blob_file.size() != 2 * cfg.weight) {
        std::printf("FAIL: %s is %zu bytes, expected 2 x %zu\n", cfg.blob.c_str(),
                    blob_file.size(), cfg.weight);
        return 1;
    }
    std::printf("per-die blob: %zu bytes = 2 experts x %zu (expert 1 at offset %zu)\n",
                blob_file.size(), cfg.weight, cfg.weight);

    WorkerPool pool;
    if (!pool.start(1)) { std::printf("FAIL: WorkerPool::start(1) -- run with sudo\n"); return 1; }

    DmaBuffer act(cfg.in, CacheMode::Cached, cfg.zone);
    DmaBuffer blobbuf(2 * cfg.weight, CacheMode::Cached, cfg.zone);
    DmaBuffer outbuf(cfg.out, CacheMode::Cached, cfg.zone);
    for (const DmaBuffer *b : {&act, &blobbuf, &outbuf}) {
        if (!b->is_valid()) { std::printf("FAIL: allocation in %s\n", cfg.zone); return 1; }
    }
    fill_activation(static_cast<uint8_t *>(act.data()), cfg.in);
    std::memcpy(blobbuf.data(), blob_file.data(), blob_file.size());

    DmaBuffer w1own(cfg.weight, CacheMode::Cached, cfg.zone);
    const bool have_w1own = !cfg.w1own.empty();
    if (have_w1own) {
        std::vector<uint8_t> w1;
        if (!read_file(w1, cfg.w1own) || w1.size() != cfg.weight) {
            std::printf("FAIL: --w1own region must be %zu bytes\n", cfg.weight);
            return 1;
        }
        std::memcpy(w1own.data(), w1.data(), cfg.weight);
    }

    for (DmaBuffer *b : {&act, &blobbuf, &outbuf, &w1own}) {
        b->flush_cache();
        if (!pool.prepare_buffer(0, *b)) {
            std::printf("FAIL: prepare_buffer(fd 0x%llx)\n", (unsigned long long)b->mem_fd());
            return 1;
        }
    }

    const DmaBufferSlice act_slice{&act, 0, cfg.in};
    const DmaBufferSlice out_slice{&outbuf, 0, cfg.out};
    const DmaBufferSlice exp0{&blobbuf, 0, cfg.weight};
    const DmaBufferSlice exp1{&blobbuf, cfg.weight, cfg.weight};
    const DmaBufferSlice w1own_slice{&w1own, 0, cfg.weight};

    struct Job { const char *name; const char *model; const DmaBufferSlice *weight; };
    const std::vector<Job> jobs = {
        {"G0  static expert0 (baked)",        cfg.static0.c_str(),  nullptr},
        {"G1  static expert1 own table",      cfg.static1.c_str(),  nullptr},
        {"G1s static expert1 shared table",   cfg.static1s.c_str(), nullptr},
        {"R1  classb0 + slice@0",             cfg.classb0.c_str(),  &exp0},
        {"R2  classb0 + slice@W",             cfg.classb0.c_str(),  &exp1},
        {"R4  classb1s + slice@W",            cfg.classb1s.c_str(), &exp1},
        {"R3  classb1 + own region (whole)",  have_w1own ? cfg.classb1.c_str() : nullptr,
                                              have_w1own ? &w1own_slice : nullptr},
    };

    std::vector<std::vector<uint8_t>> results;
    int hard_fails = 0;
    for (const Job &j : jobs) {
        if (j.model == nullptr || *j.model == '\0') {
            std::printf("  [%s] skipped (model not supplied)\n", j.name);
            results.emplace_back();
            continue;
        }
        ModelInfo info;
        if (!pool.load_model(0, j.model, info)) {
            std::printf("  [%s] FAIL load_model\n", j.name);
            ++hard_fails;
            results.emplace_back();
            continue;
        }
        std::vector<DmaBufferSlice> ins{act_slice};
        if (j.weight) ins.push_back(*j.weight);
        const std::vector<DmaBufferSlice> outs{out_slice};
        uint64_t elapsed = 0;
        const bool ok = pool.submit_task(0, info.model_id, ins, outs, &elapsed);
        pool.unload_model(0, info.model_id);
        if (!ok) {
            std::printf("  [%s] FAIL submit (model_id=%u inputs=%zu)\n", j.name, info.model_id,
                        ins.size());
            ++hard_fails;
            results.emplace_back();
            continue;
        }
        outbuf.flush_cache();
        results.emplace_back(static_cast<const uint8_t *>(outbuf.data()),
                            static_cast<const uint8_t *>(outbuf.data()) + cfg.out);
        std::printf("  [%s] model_id=%u inputs=%zu submit %.3f ms\n", j.name, info.model_id,
                    ins.size(), static_cast<double>(elapsed) / 1e6);
    }

    auto dec = [&](size_t i) { return decode_out(results[i], cfg.out_dtype); };
    if (!results.empty() && !results[0].empty()) {
        size_t tail_zeros = 0;
        for (size_t i = cfg.in; i < results[0].size(); ++i)
            if (results[0][i] == 0) ++tail_zeros;
        std::printf("\noutput surface %zu bytes decoded as %s; bytes [%zu,%zu) that are zero: "
                    "%zu (the tensor is 2048 int8, the surface is page-padded)\n",
                    cfg.out, cfg.out_dtype.c_str(), cfg.in, results[0].size(), tail_zeros);
    }
    // expect: 1 = must be byte-identical, -1 = must differ, 0 = report only.
    auto report = [&](size_t i, size_t j, const char *what, int expect) {
        if (results[i].empty() || results[j].empty()) {
            std::printf("  %-46s skipped\n", what);
            return;
        }
        const bool same = results[i] == results[j];
        const auto a = dec(i), b = dec(j);
        std::printf("  %-46s %s  cos=%.6f maxdiff=%.4g mean|a|=%.4g\n", what,
                    same ? "BYTE-IDENTICAL" : "differs     ", cosine(a, b), max_abs_diff(a, b),
                    mean_abs(a));
        if (expect > 0 && !same) ++hard_fails;
        if (expect < 0 && same) ++hard_fails;
    };

    std::printf("\n--- expectations (byte-exact unless stated) ---\n");
    report(3, 0, "R1 classb0+slice@0  == G0 expert0 static", 1);
    report(5, 2, "R4 classb1s+slice@W == G1s expert1 shared", 1);
    report(6, 1, "R3 classb1+own      == G1 expert1 own", 1);
    report(3, 4, "expert0 output      != expert1 output", -1);
    report(0, 2, "G0 static expert0   != G1s static expert1", -1);
    // Measured, and the reason a layer cannot share one loadable across *arbitrary* experts: the
    // compiled kmd-task-op_list carries weight-derived requant bytes (6 differ between two
    // experts of one layer, 0x18 vs 0x19, and nothing else in the 283,384-byte file), so binding
    // expert 1's region into expert 0's loadable is close but wrong (cos 0.958).
    //
    // With --same-class the two loadables are byte-identical by construction -- they came from the
    // same stream class (FINDINGS §35: 104 real experts across all 26 MoE layers collapse to 15
    // classes) -- so the cross-binding MUST reproduce the golden exactly. That is the end-to-end
    // proof that one loadable per class can serve every expert in it, across layers.
    report(4, 2, cfg.same_class ? "R2 CROSS-CLASS bind  == G1s (loadables interchangeable)"
                                : "R2 cross-loadable (e1 region in e0 loadable) != G1s",
           cfg.same_class ? 1 : -1);

    std::printf("\n--- reported, not required ---\n");
    report(4, 1, "R2 classb0+slice@W  vs G1 expert1 own table", 0);
    report(2, 1, "G1s shared static   vs G1 own-table static", 0);

    for (DmaBuffer *b : {&act, &blobbuf, &outbuf, &w1own}) pool.unprepare_buffer(0, *b);

    // --- what does the "one loadable per expert" fallback cost? -------------------
    // The two experts' Class B loadables differ in exactly 6 bytes, so a layer could keep one
    // template and patch it per expert -- if 64 (or 1664) loadables can be resident at once.
    // Load cost and the concurrent-model ceiling are the two numbers that decide it.
    if (cfg.load_repeats > 0 && !cfg.classb0.empty()) {
        std::vector<double> ms;
        for (int i = 0; i < cfg.load_repeats; ++i) {
            ModelInfo info;
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = pool.load_model(0, cfg.classb0, info);
            const auto t1 = std::chrono::steady_clock::now();
            if (!ok) { std::printf("FAIL: load #%d of %s\n", i, cfg.classb0.c_str()); break; }
            pool.unload_model(0, info.model_id);
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        if (!ms.empty()) {
            std::vector<double> s = ms;
            std::sort(s.begin(), s.end());
            double sum = 0;
            for (double v : s) sum += v;
            std::printf("\nClass B load+unload of %s (%zu reps): mean %.2f ms  p50 %.2f  "
                        "min %.2f  max %.2f\n",
                        cfg.classb0.c_str(), s.size(), sum / s.size(),
                        s[s.size() / 2], s.front(), s.back());
        }
    }
    if (cfg.preload > 0 && !cfg.classb0.empty()) {
        std::vector<uint32_t> ids;
        const auto t0 = std::chrono::steady_clock::now();
        while (static_cast<int>(ids.size()) < cfg.preload) {
            ModelInfo info;
            if (!pool.load_model(0, cfg.classb0, info)) {
                std::printf("preload stopped at %zu resident models (load failed)\n", ids.size());
                break;
            }
            ids.push_back(info.model_id);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double total = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("preload: %zu Class B loadables resident concurrently in %.1f ms "
                    "(%.2f ms each); model_ids %u..%u\n",
                    ids.size(), total, ids.empty() ? 0.0 : total / ids.size(), ids.front(),
                    ids.back());
        // Prove a resident one still computes after the others loaded.
        if (ids.size() > 1) {
            const std::vector<DmaBufferSlice> ins{act_slice, exp1};
            const std::vector<DmaBufferSlice> outs{out_slice};
            uint64_t el = 0;
            const bool ok = pool.submit_task(0, ids.front(), ins, outs, &el);
            outbuf.flush_cache();
            std::vector<uint8_t> after(static_cast<const uint8_t *>(outbuf.data()),
                                       static_cast<const uint8_t *>(outbuf.data()) + cfg.out);
            std::printf("submit against the FIRST resident model after %zu loads: %s, %s\n",
                        ids.size(), ok ? "ok" : "FAILED",
                        (!results[4].empty() && after == results[4]) ? "still byte-identical"
                                                                    : "OUTPUT CHANGED");
        }
        for (uint32_t id : ids) pool.unload_model(0, id);
    }

    pool.shutdown();

    std::printf("\n--- verdict ---\n");
    if (hard_fails == 0) {
        std::printf("  PASS: two real DeepSeek-V2-Lite INT8 experts live in one per-die blob and\n"
                    "  are selected purely by DevBuf.offset, each reproducing its own static\n"
                    "  loadable byte-for-byte through the WorkerPool slice API.\n");
        return 0;
    }
    std::printf("  FAIL: %d expectation(s) violated -- see the table above.\n", hard_fails);
    return 3;
}
