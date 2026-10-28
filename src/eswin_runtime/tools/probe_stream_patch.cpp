/**
 * @file probe_stream_patch.cpp
 * @brief Milestone 14 gate: find a loaded `kmd-task-op_list` in this process's memory, patch the
 *        bytes that make a loadable expert-specific, and see whether the NPU obeys.
 *
 * Why this matters (FINDINGS §33): two real DeepSeek-V2-Lite INT8 experts compiled from a
 * byte-identical quant table produce Class B loadables differing in exactly **6 bytes of
 * 283,384** — one weight-derived requant shift per graph node, all inside `kmd-task-op_list`.
 * Binding expert 1's weight region into expert 0's loadable gives cos 0.958, i.e. wrong. The
 * alternatives are measured dead: per-expert loadables cost 34.25 ms to load, only 62 stay
 * resident per die, and `per_tensor` quantization (which would remove the coupling) makes EsAAC
 * SIGSEGV. So either a loaded stream can be patched, or MoE needs a different shape.
 *
 * Two search phases, because the first one already taught us something:
 *   1. readable memory via /proc/self/mem. This finds the loader's *heap* copy of the most
 *      recently loaded stream — patching it changed nothing, so it is a leftover parse buffer,
 *      not what the NPU reads. Kept because "we looked and it did not work" is evidence.
 *   2. the `/dmabuf:npu` mappings that /proc/self/mem returns EIO for. They are `rw-s`, i.e.
 *      mapped readable AND writable into this process, so they are scanned by direct dereference
 *      under a SIGSEGV guard. This is where a DMA-fetchable command stream has to live.
 *
 * Phase 2 cannot match the file bytes exactly: the loader applies `reloc_list` patches (IOVAs)
 * to the stream. So it searches a 64-byte window centred on a patch site with the centre byte
 * masked out, then scores the whole stream against both experts to see how much still matches.
 *
 *   sudo ./probe_stream_patch --classb0 A.model --classb1 B.model --weights0 w0.bin \
 *        --weights1 w1.bin --stream0 A_op_list.bin --stream1 B_op_list.bin [--flush-all]
 */

#include "../src/backend/driver_wrapper.hpp"

#include <algorithm>
#include <csignal>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

using namespace eswin;

namespace {

size_t kInSize = 2048;         // INT8 activation codes
size_t kOutSize = 4096;        // page-padded surface for a 2048-byte int8 tensor
constexpr size_t kMaxScanRegion = 256u << 20;   // skip MMZ-pool-sized mappings
constexpr size_t kSigLen = 64;                  // head bytes used to find candidate copies
constexpr size_t kWindow = 64;                  // masked window around a patch site (phase 2)

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

/// A varied int8 activation. A uniform fill makes this graph emit zeros regardless of the
/// weights (FINDINGS §32.3), which is indistinguishable from a failed patch.
void fill_activation(uint8_t *p, size_t bytes) {
    uint32_t s = 0x2545F491u;
    for (size_t i = 0; i < bytes; ++i) {
        s = s * 1664525u + 1013904223u;
        p[i] = static_cast<uint8_t>(static_cast<int8_t>((s >> 24) % 129 - 64));
    }
}

struct Region {
    uintptr_t start = 0, end = 0;
    char perms[8] = {0};
    std::string path;
};

struct ScanStats {
    size_t regions = 0, skipped = 0, bytes_read = 0, read_errors = 0;
    int first_errno = 0;
    uintptr_t first_error_addr = 0;
    std::vector<Region> unreadable;   // what phase 1 could not see; phase 2 retries these
};

std::vector<Region> parse_maps(size_t max_region, ScanStats &st) {
    std::vector<Region> regions;
    FILE *maps = std::fopen("/proc/self/maps", "r");
    if (!maps) { std::printf("FAIL: cannot open /proc/self/maps\n"); return regions; }
    char line[2048];
    while (std::fgets(line, sizeof(line), maps)) {
        Region r;
        char path[512] = {0};
        unsigned long long off = 0, ino = 0;
        // dev is "%x:%x", which one integer conversion cannot parse -- swallow it explicitly.
        const int n = std::sscanf(line, "%lx-%lx %7s %llx %*x:%*x %llu %511[^\n]", &r.start,
                                  &r.end, r.perms, &off, &ino, path);
        if (n < 5 || r.perms[0] != 'r') continue;
        if (r.end - r.start > max_region) { ++st.skipped; continue; }
        if (std::strstr(line, "[vvar]") || std::strstr(line, "[vsyscall]")) continue;
        r.path = (n >= 6 && path[0]) ? path : (std::strstr(line, "[heap]") ? "[heap]" : "[anon]");
        regions.push_back(r);
    }
    std::fclose(maps);
    return regions;
}

/// Reads through /proc/self/mem so guard pages fail cleanly instead of SIGSEGVing.
bool read_region(int memfd, uintptr_t at, void *dst, size_t n, ScanStats &st) {
    const ssize_t got = pread(memfd, dst, n, static_cast<off_t>(at));
    if (got == static_cast<ssize_t>(n)) { st.bytes_read += n; return true; }
    ++st.read_errors;
    if (!st.first_errno) { st.first_errno = errno ? errno : -1; st.first_error_addr = at; }
    return false;
}

// --- SIGSEGV guard for direct dereference of device mappings -----------------------
sigjmp_buf g_jmp;
volatile sig_atomic_t g_faulted = 0;
void (*g_old_segv)(int) = nullptr;

void segv_handler(int) {
    g_faulted = 1;
    siglongjmp(g_jmp, 1);
}

struct SegvGuard {
    SegvGuard() { g_old_segv = std::signal(SIGSEGV, segv_handler); }
    ~SegvGuard() { std::signal(SIGSEGV, g_old_segv); }
};

struct RawHit {
    uintptr_t addr = 0;
    Region region;
};

/**
 * @brief Phase 1: find @p needle in every region readable through /proc/self/mem.
 *
 * Generic on purpose so main() can self-test it with planted bait: a scanner that cannot find a
 * pattern we know is present makes "not found" meaningless.
 */
std::vector<RawHit> find_pattern(const std::vector<uint8_t> &needle, ScanStats &st,
                                 std::vector<Region> &regions) {
    std::vector<RawHit> hits;
    if (needle.empty()) return hits;
    const int memfd = open("/proc/self/mem", O_RDONLY);
    if (memfd < 0) { std::printf("FAIL: cannot open /proc/self/mem (errno %d)\n", errno); return hits; }

    std::vector<uint8_t> chunk(4u << 20);
    for (const Region &r : regions) {
        const size_t sz = r.end - r.start;
        ++st.regions;
        const size_t step = chunk.size() - (needle.size() - 1);
        bool failed = false;
        for (size_t base = 0; base < sz; base += step) {
            const size_t want = std::min(chunk.size(), sz - base);
            if (want < needle.size()) break;
            if (!read_region(memfd, r.start + base, chunk.data(), want, st)) { failed = true; break; }
            for (size_t i = 0; i + needle.size() <= want; ++i) {
                if (chunk[i] != needle[0]) continue;
                if (std::memcmp(chunk.data() + i, needle.data(), needle.size()) != 0) continue;
                hits.push_back(RawHit{r.start + base + i, r});
                i += needle.size() - 1;
            }
        }
        if (failed) st.unreadable.push_back(r);
    }
    close(memfd);
    return hits;
}

/**
 * @brief Phase 2: search one region by direct dereference for a window whose centre byte is
 *        masked, because the loader relocates addresses inside the stream and may also hold the
 *        other expert's value at the patch site.
 */
struct WindowHit {
    uintptr_t addr = 0;      // address of the window's first byte
    uintptr_t base = 0;      // implied start of the stream
    size_t match0 = 0, match1 = 0;
    uint8_t centre = 0;
    Region region;
};

std::vector<WindowHit> find_window(const Region &r, const std::vector<uint8_t> &window,
                                   size_t centre_idx, size_t stream_len, uintptr_t window_file_off,
                                   const std::vector<uint8_t> &s0, const std::vector<uint8_t> &s1) {
    std::vector<WindowHit> out;
    const size_t sz = r.end - r.start;
    if (sz < window.size()) return out;
    const uint8_t *p = reinterpret_cast<const uint8_t *>(r.start);
    for (size_t i = 0; i + window.size() <= sz; ++i) {
        if (p[i] != window[0]) continue;
        bool ok = true;
        for (size_t k = 1; k < window.size() && ok; ++k) {
            if (k == centre_idx) continue;                 // masked: expert-specific value
            if (p[i + k] != window[k]) ok = false;
        }
        if (!ok) continue;
        WindowHit h;
        h.addr = r.start + i;
        h.region = r;
        h.centre = p[i + centre_idx];
        if (window_file_off <= i) {
            h.base = h.addr - window_file_off;
            if (h.base >= r.start && h.base + stream_len <= r.end) {
                const uint8_t *b = reinterpret_cast<const uint8_t *>(h.base);
                for (size_t k = 0; k < stream_len; ++k) {
                    if (b[k] == s0[k]) ++h.match0;
                    if (b[k] == s1[k]) ++h.match1;
                }
            }
        }
        out.push_back(h);
        i += window.size() - 1;
    }
    return out;
}

struct Own { const void *base; size_t size; const char *what; };

} // namespace

int main(int argc, char **argv) {
    std::string classb0, classb1, w0path, w1path, s0path, s1path;
    bool flush_all = false;
    size_t corrupt_count = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char *w) -> const char * {
            if (i + 1 >= argc) { std::printf("FAIL: %s needs a value\n", w); std::exit(1); }
            return argv[++i];
        };
        if (a == "--classb0") classb0 = need(a.c_str());
        else if (a == "--classb1") classb1 = need(a.c_str());
        else if (a == "--weights0") w0path = need(a.c_str());
        else if (a == "--weights1") w1path = need(a.c_str());
        else if (a == "--stream0") s0path = need(a.c_str());
        else if (a == "--stream1") s1path = need(a.c_str());
        else if (a == "--in") kInSize = std::strtoull(need(a.c_str()), nullptr, 10);
        else if (a == "--out") kOutSize = std::strtoull(need(a.c_str()), nullptr, 10);
        else if (a == "--flush-all") flush_all = true;
        else if (a == "--corrupt") corrupt_count = std::strtoull(need(a.c_str()), nullptr, 10);
        else { std::printf("FAIL: unknown option '%s'\n", a.c_str()); return 1; }
    }
    if (classb0.empty() || classb1.empty() || w0path.empty() || w1path.empty() ||
        s0path.empty() || s1path.empty()) {
        std::printf("usage: %s --classb0 A.model --classb1 B.model --weights0 w0.bin "
                    "--weights1 w1.bin --stream0 A_op_list.bin --stream1 B_op_list.bin "
                    "[--in N] [--out N] [--flush-all]\n", argv[0]);
        return 1;
    }

    std::vector<uint8_t> s0, s1, w0, w1;
    if (!read_file(s0, s0path) || !read_file(s1, s1path)) return 1;
    if (!read_file(w0, w0path) || !read_file(w1, w1path)) return 1;
    if (s0.size() != s1.size() || w0.size() != w1.size()) {
        std::printf("FAIL: stream or weight sizes differ between experts\n");
        return 1;
    }
    const size_t W = w0.size();
    std::vector<size_t> patch_off;
    for (size_t i = 0; i < s0.size(); ++i)
        if (s0[i] != s1[i]) patch_off.push_back(i);
    std::printf("op_list %zu bytes; %zu bytes to patch:", s0.size(), patch_off.size());
    for (size_t o : patch_off) std::printf(" %zu(0x%02x->0x%02x)", o, s0[o], s1[o]);
    std::printf("\nweight region %zu bytes each\n", W);
    if (patch_off.empty()) { std::printf("FAIL: streams identical, nothing to test\n"); return 1; }

    EswinDriver &drv = EswinDriver::instance();
    if (!drv.initialize()) { std::printf("SKIP: no NPU hardware\n"); return 2; }
    if (drv.set_device(0) != 0) { std::printf("FAIL: set_device(0)\n"); return 1; }

    const Buf act = alloc(drv, kInSize, "act");
    const Buf blob = alloc(drv, 2 * W, "blob w0|w1");
    Buf out = alloc(drv, kOutSize, "out");
    if (!act.fd || !blob.fd || !out.fd) return 1;

    fill_activation(static_cast<uint8_t *>(act.va), kInSize);
    std::memcpy(blob.va, w0.data(), W);
    std::memcpy(static_cast<uint8_t *>(blob.va) + W, w1.data(), W);
    drv.mem_flush_cache(act.fd);
    drv.mem_flush_cache(blob.fd);

    DevBuf d_act{act.fd, 0, kInSize, 0}, d_out{out.fd, 0, kOutSize, 0};
    const DevBuf d_r0{blob.fd, 0, W, 0}, d_r1{blob.fd, W, W, 0};
    if (drv.prepare_dev_buf(&d_act) != 0 || drv.prepare_dev_buf(&d_out) != 0) {
        std::printf("FAIL: prepare act/out\n"); return 1;
    }
    DevBuf prep_blob{blob.fd, 0, 2 * W, 0};
    if (drv.prepare_dev_buf(&prep_blob) != 0) { std::printf("FAIL: prepare blob\n"); return 1; }

    uint32_t idA = 0, idB = 0;
    if (drv.load_model(&idA, classb0.c_str()) != 0) { std::printf("FAIL: load A\n"); return 1; }
    if (drv.load_model(&idB, classb1.c_str()) != 0) { std::printf("FAIL: load B\n"); return 1; }
    std::printf("loaded A=%u (%s) and B=%u (%s)\n", idA, classb0.c_str(), idB, classb1.c_str());

    auto run = [&](uint32_t id, const DevBuf &w, std::vector<uint8_t> &dst, const char *label) {
        NpuTask task;
        std::memset(&task, 0, sizeof(task));
        task.taskId = 1;
        task.modelId = id;
        task.inputFdNum = 2;
        task.inputFd[0] = d_act;
        task.inputFd[1] = w;
        task.outputFdNum = 1;
        task.outputFd[0] = d_out;
        const int rc = drv.submit_task(&task, 1);
        if (rc != 0) {
            std::printf("  [%s] submit FAILED rc=%d (%s)\n", label, rc,
                        EswinDriver::decode_error(rc).c_str());
            return false;
        }
        drv.mem_flush_cache(out.fd);
        const uint8_t *p = static_cast<const uint8_t *>(out.va);
        dst.assign(p, p + kOutSize);
        size_t nz = 0;
        for (size_t i = 0; i < kOutSize; ++i) if (p[i]) ++nz;
        std::printf("  [%s] ok, nonzero=%zu/%zu\n", label, nz, kOutSize);
        return true;
    };

    std::vector<uint8_t> a_r0, b_r1, a_r1, b_r0;
    std::printf("\n--- before any patch ---\n");
    if (!run(idA, d_r0, a_r0, "A + region0 (correct pair)")) return 1;
    if (!run(idB, d_r1, b_r1, "B + region1 (correct pair)")) return 1;
    if (!run(idA, d_r1, a_r1, "A + region1 (cross)")) return 1;
    if (!run(idB, d_r0, b_r0, "B + region0 (cross)")) return 1;
    std::printf("  A+r1 vs B+r1: %s    B+r0 vs A+r0: %s  (both must differ pre-patch)\n",
                a_r1 == b_r1 ? "IDENTICAL?!" : "differs",
                b_r0 == a_r0 ? "IDENTICAL?!" : "differs");

    // --- phase 1: readable memory ----------------------------------------------------
    std::vector<uint8_t> bait(8192, 0xA5);
    const std::string marker = "M14-BAIT-7f3c91d0-5e2a";
    std::memcpy(bait.data() + 4096, marker.data(), marker.size());
    ScanStats bst;
    std::vector<Region> regions = parse_maps(kMaxScanRegion, bst);
    const std::vector<uint8_t> needle(marker.begin(), marker.end());
    const std::vector<RawHit> bait_hits = find_pattern(needle, bst, regions);
    bool scanner_ok = false;
    for (const RawHit &h : bait_hits)
        if (h.addr >= reinterpret_cast<uintptr_t>(bait.data()) &&
            h.addr < reinterpret_cast<uintptr_t>(bait.data()) + bait.size()) scanner_ok = true;
    std::printf("\n--- phase 1 scanner self-test: %s (regions=%zu, %.1f MiB read) ---\n",
                scanner_ok ? "PASS" : "FAIL -- its negatives mean nothing", bst.regions,
                bst.bytes_read / 1048576.0);

    ScanStats st;
    const std::vector<uint8_t> sig(s0.begin(), s0.begin() + std::min(kSigLen, s0.size()));
    const std::vector<RawHit> raw = find_pattern(sig, st, regions);
    std::vector<Own> ours = {{s0.data(), s0.size(), "probe's stream0"},
                            {s1.data(), s1.size(), "probe's stream1"}};
    size_t head_only = 0, full0 = 0, full1 = 0, external_full = 0;
    const RawHit *heap_copy = nullptr;
    int heap_copy_kind = 0;
    std::printf("\n--- phase 1: /proc/self/mem scan (regions=%zu read=%.1f MiB errors=%zu) ---\n",
                st.regions, st.bytes_read / 1048576.0, st.read_errors);
    for (const RawHit &h : raw) {
        const uint8_t *p = reinterpret_cast<const uint8_t *>(h.addr);
        const bool m0 = std::memcmp(p, s0.data(), s0.size()) == 0;
        const bool m1 = !m0 && std::memcmp(p, s1.data(), s1.size()) == 0;
        if (!m0 && !m1) { ++head_only; continue; }
        const Own *mine = nullptr;
        for (const Own &o : ours) {
            const uintptr_t ob = reinterpret_cast<uintptr_t>(o.base);
            if (h.addr >= ob && h.addr + s0.size() <= ob + o.size) { mine = &o; break; }
        }
        if (m0) ++full0; else ++full1;
        std::printf("  0x%lx %-8s %s  %s\n", (unsigned long)h.addr, h.region.perms,
                    h.region.path.c_str(),
                    mine ? mine->what : (m0 ? "FULL MATCH A's op_list" : "FULL MATCH B's op_list"));
        if (!mine) { ++external_full; heap_copy = &h; heap_copy_kind = m0 ? 1 : 2; }
    }
    std::printf("  summary: %zu full A, %zu full B, %zu head-only, %zu external\n", full0, full1,
                head_only, external_full);
    std::printf("  phase 1 could not read %zu region(s):", st.unreadable.size());
    for (const Region &r : st.unreadable)
        std::printf(" %s(%.0f KiB)", r.path.c_str(), (r.end - r.start) / 1024.0);
    std::printf("\n");

    // --- phase 2: the device mappings phase 1 could not read -------------------------
    std::printf("\n--- phase 2: direct scan of unreadable mappings (window=%zu B, centre masked) "
                "---\n", kWindow);
    const size_t centre = kWindow / 2;
    const size_t wfile = patch_off[0] >= centre ? patch_off[0] - centre : 0;
    const std::vector<uint8_t> window(s0.begin() + wfile, s0.begin() + wfile + kWindow);
    SegvGuard guard;
    std::vector<WindowHit> best;
    size_t scanned = 0, faulted = 0;
    for (const Region &r : st.unreadable) {
        g_faulted = 0;
        if (sigsetjmp(g_jmp, 1) != 0) { ++faulted; continue; }
        std::vector<WindowHit> hs =
            find_window(r, window, centre, s0.size(), wfile, s0, s1);
        ++scanned;
        for (WindowHit &h : hs) {
            std::printf("  %s 0x%lx-0x%lx %-8s: window at 0x%lx centre=0x%02x stream base=0x%lx "
                        "match A=%.2f%% B=%.2f%%\n",
                        r.path.c_str(), (unsigned long)r.start, (unsigned long)r.end, r.perms,
                        (unsigned long)h.addr, h.centre, (unsigned long)h.base,
                        100.0 * h.match0 / s0.size(), 100.0 * h.match1 / s0.size());
            best.push_back(h);
        }
    }
    std::printf("  scanned %zu device region(s), %zu faulted, %zu window hit(s)\n", scanned,
                faulted, best.size());

    // --- patch the most promising device copy and re-run -----------------------------
    int verdict = 4;
    const WindowHit *target = nullptr;
    size_t best_score = 0;
    for (const WindowHit &h : best) {
        if (h.base == 0) continue;
        const size_t score = std::max(h.match0, h.match1);
        if (score > best_score) { best_score = score; target = &h; }
    }
    if (!target) {
        std::printf("\nNo device-resident copy of the stream was found. Either it is not mapped "
                    "into this\nprocess at all, or relocation rewrote enough of it that the "
                    "window no longer matches.\n");
        if (heap_copy) std::printf("(phase 1 did find a heap copy of %s's stream, but patching "
                                   "leftovers cannot affect the NPU)\n",
                                   heap_copy_kind == 1 ? "A" : "B");
        verdict = 3;
    } else if (corrupt_count > 0) {
        // Liveness control. The patch test below cannot distinguish "this copy is not what the
        // NPU fetches" from "the field is consumed at load time", but corruption can: if mangling
        // the copy leaves the output bit-identical, the NPU is not reading it at submit at all.
        uint8_t *base = reinterpret_cast<uint8_t *>(target->base);
        const size_t at = patch_off[0] + 2;
        const uint32_t model = (base[patch_off[0]] == s0[patch_off[0]]) ? idA : idB;
        const DevBuf region = (model == idA) ? d_r0 : d_r1;
        const std::vector<uint8_t> &golden = (model == idA) ? a_r0 : b_r1;
        std::printf("\n--- liveness control: corrupting %zu byte(s) at stream offset %zu of the "
                    "copy at 0x%lx ---\n", corrupt_count, at, (unsigned long)target->base);
        std::vector<uint8_t> saved(base + at, base + at + corrupt_count);
        for (size_t i = 0; i < corrupt_count; ++i) base[at + i] ^= 0xFF;
        if (flush_all) {
            void *h = dlopen("libes_memory.so", RTLD_NOW | RTLD_GLOBAL);
            using Fn = int (*)();
            Fn f = h ? reinterpret_cast<Fn>(dlsym(h, "ES_SYS_MemFlushCacheAll")) : nullptr;
            std::printf("  ES_SYS_MemFlushCacheAll: %s\n", f ? "called" : "NOT AVAILABLE");
        }
        std::vector<uint8_t> after;
        if (!run(model, region, after, "corrupted stream + its own region")) return 1;
        const bool changed = after != golden;
        std::printf("\n--- liveness verdict ---\n");
        std::printf("  output changed by corruption: %s\n", changed ? "YES" : "no");
        if (changed) {
            std::printf("  The copy IS live: the NPU fetches it at submit. So the 6 requant bytes\n"
                        "  are consumed at LOAD time, not run time -- patching cannot switch\n"
                        "  experts, and Milestone 14's patch route is dead.\n");
            verdict = 3;
        } else {
            std::printf("  The copy is NOT live: corrupting %zu bytes changed nothing, so the\n"
                        "  stream the NPU fetches is somewhere else (kernel-side or another\n"
                        "  mapping). The patch route is neither proven nor disproven.\n",
                        corrupt_count);
            verdict = 4;
        }
        std::memcpy(base + at, saved.data(), saved.size());
        std::printf("  (restored the %zu corrupted bytes)\n", saved.size());
    } else {
        uint8_t *base = reinterpret_cast<uint8_t *>(target->base);
        // Ownership cannot come from the match score: the two streams differ in only 6 of
        // 101,184 bytes, so both score 99.95% and tie. The patch sites themselves are the only
        // discriminating bytes, so count how many already hold each expert's value.
        size_t eq0 = 0, eq1 = 0;
        for (size_t o : patch_off) {
            if (base[o] == s0[o]) ++eq0;
            if (base[o] == s1[o]) ++eq1;
        }
        const bool isA = eq0 > eq1;
        std::printf("\n  copy holds A's values at %zu/%zu patch sites and B's at %zu/%zu -> "
                    "this is %s's stream\n", eq0, patch_off.size(), eq1, patch_off.size(),
                    isA ? "A" : "B");
        if (eq0 != patch_off.size() && eq1 != patch_off.size())
            std::printf("  WARNING: patch sites are mixed (%zu/%zu), so ownership is a guess\n",
                        std::max(eq0, eq1), patch_off.size());
        const uint32_t model = isA ? idA : idB;
        const DevBuf region = isA ? d_r1 : d_r0;
        const std::vector<uint8_t> &golden = isA ? b_r1 : a_r0;
        const std::vector<uint8_t> &before = isA ? a_r1 : b_r0;
        std::printf("\n--- patching the device copy at 0x%lx (%.2f%% match, %zu of %zu bytes "
                    "already equal to the file) ---\n",
                    (unsigned long)target->base, 100.0 * best_score / s0.size(), best_score,
                    s0.size());
        for (size_t o : patch_off) {
            const uint8_t want = isA ? s1[o] : s0[o];
            std::printf("  [%zu] 0x%02x -> 0x%02x\n", o, base[o], want);
            base[o] = want;
        }
        size_t ok = 0;
        for (size_t o : patch_off) if (base[o] == (isA ? s1[o] : s0[o])) ++ok;
        std::printf("  readback: %zu/%zu applied\n", ok, patch_off.size());
        if (flush_all) {
            void *h = dlopen("libes_memory.so", RTLD_NOW | RTLD_GLOBAL);
            using Fn = int (*)();
            Fn f = h ? reinterpret_cast<Fn>(dlsym(h, "ES_SYS_MemFlushCacheAll")) : nullptr;
            std::printf("  ES_SYS_MemFlushCacheAll: %s\n",
                        f ? (f() == 0 ? "called, returned 0" : "called, non-zero")
                          : "NOT AVAILABLE");
        }
        std::vector<uint8_t> after;
        if (!run(model, region, after, "patched stream + other expert's region")) return 1;
        const bool now_matches = after == golden;
        const bool changed = after != before;
        std::printf("\n--- verdict ---\n");
        std::printf("  output changed by the patch : %s\n", changed ? "YES" : "no");
        std::printf("  equals the correct pair     : %s\n",
                    now_matches ? "YES (byte-identical)" : "no");
        if (now_matches) {
            std::printf("  PASS: the NPU reads this copy at submit, so ONE loadable per layer can\n"
                        "  serve every expert by patching %zu stream bytes on switch. Milestone "
                        "14 is viable.\n", patch_off.size());
            verdict = 0;
        } else if (!changed) {
            std::printf("  INCONCLUSIVE: no change. Either this copy is not the one the NPU "
                        "fetches,\n  or it is cached device memory (%s).\n",
                        flush_all ? "--flush-all was used" : "retry with --flush-all");
            verdict = 4;
        } else {
            std::printf("  FAIL: the output changed but is not the correct pair -- these bytes "
                        "are not\n  the whole story, so patching cannot replace a loadable.\n");
            verdict = 3;
        }
    }

    drv.unload_model(idA);
    drv.unload_model(idB);
    return verdict;
}
