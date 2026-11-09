/**
 * @file prefix_cache.hpp
 * @brief Persistent 1 TB Hybrid Prefix Cache (LFMKVB01 + RadixCache) for LFM3 & LFM2.5-MoE.
 *
 * Implements Milestone 17A (src/LFM3/PLAN.md §8) and Phase F (src/LFM2.5-8B-A1B/PLAN.md):
 *  - Chained 128-bit content-addressed block keys (B = 64 tokens, matching kPrefillBatch = 64),
 *    automatically incorporating model weight/scale fingerprints and multimodal vision slices.
 *  - 4 KiB-aligned .kvb ("LFMKVB01") chunk files split into causal hybrid state planes:
 *      * Section 0 (0x0000): 256 B KvbHeader + token_ids[64] + last_h[hidden_dim]
 *      * Section 1 (Plane 1): Global Attention KV (u8 + sc + k_sum) — read for chunks 0..k
 *      * Section 2 (Plane 2): SWA 256-token KV delta (FP32)         — read for max(0,k-3)..k
 *      * Section 3 (Plane 3): ShortConv rolling state (FP32)        — read for chunk k only
 *      * Section 4 (Plane 1b): Short-ctx Global FP32 KV (k < 4)     — read only if ctx < 256
 *  - Crash-safe WAL journal (index.wal + index.snap), 256-way fanout directories, background
 *    async NVMe writer with zero-latency in-flight memory hit support, per-section 64-bit
 *    integrity checksums, and leaf-first Greedy-Dual-Size-Frequency (GDSF) eviction with pinning.
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace eswin::prefix_cache {

static constexpr uint32_t kKvbFormatVersion = 1;
static constexpr uint32_t kBlockTokens      = 64;
static constexpr size_t   kPageAlign        = 4096;
static constexpr char     kKvbMagic[8]      = {'L', 'F', 'M', 'K', 'V', 'B', '0', '1'};

enum KvbFlags : uint32_t {
    KVB_FLAG_NONE          = 0,
    KVB_FLAG_HAS_SWA       = 1u << 0, // Section 2 populated (LFM3 = 1, LFM2.5 = 0)
    KVB_FLAG_HAS_GLOB_FP32 = 1u << 1, // Section 4 populated (block_idx < 4)
    KVB_FLAG_HAS_VISION    = 1u << 2, // Multimodal visual_embeds mixed into block_key
    KVB_FLAG_PINNED        = 1u << 3, // Immortal until explicit unpin/clear
};

enum class ModelArchId : uint32_t {
    LFM3_1_7B     = 1,
    LFM2_5_8B_A1B = 2,
};

inline constexpr size_t align_up_4k(size_t n) noexcept {
    return (n + (kPageAlign - 1)) & ~(kPageAlign - 1);
}

// =============================================================================
// Fast Deterministic 128-Bit & 64-Bit Non-Cryptographic Hashing (WyMix-128)
// =============================================================================

struct Hash128 {
    uint64_t lo = 0;
    uint64_t hi = 0;

    constexpr bool operator==(const Hash128 &o) const noexcept {
        return lo == o.lo && hi == o.hi;
    }
    constexpr bool operator!=(const Hash128 &o) const noexcept {
        return !(*this == o);
    }
    constexpr bool is_zero() const noexcept {
        return lo == 0 && hi == 0;
    }

    std::string to_hex() const {
        static constexpr char kHex[] = "0123456789abcdef";
        std::string out(32, '0');
        uint64_t words[2] = {hi, lo};
        for (int w = 0; w < 2; ++w) {
            uint64_t v = words[w];
            for (int i = 15; i >= 0; --i) {
                out[w * 16 + i] = kHex[v & 0xFu];
                v >>= 4;
            }
        }
        return out;
    }

    static bool from_hex(const std::string &hex, Hash128 &out) noexcept {
        if (hex.size() != 32) return false;
        auto hex_val = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
            if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
            return -1;
        };
        uint64_t words[2] = {0, 0};
        for (int w = 0; w < 2; ++w) {
            uint64_t acc = 0;
            for (int i = 0; i < 16; ++i) {
                int nib = hex_val(hex[w * 16 + i]);
                if (nib < 0) return false;
                acc = (acc << 4) | static_cast<uint64_t>(nib);
            }
            words[w] = acc;
        }
        out.hi = words[0];
        out.lo = words[1];
        return true;
    }
};

struct Hash128Hasher {
    size_t operator()(const Hash128 &h) const noexcept {
        uint64_t x = h.lo ^ (h.hi * 0x9e3779b97f4a7c15ULL + 0x517cc1b727220a95ULL);
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<size_t>(x);
    }
};

namespace detail {

inline void wymix128(uint64_t a, uint64_t b, uint64_t &lo, uint64_t &hi) noexcept {
#if defined(__SIZEOF_INT128__)
    __uint128_t prod = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    lo = static_cast<uint64_t>(prod);
    hi = static_cast<uint64_t>(prod >> 64);
#else
    uint64_t a_lo = a & 0xFFFFFFFFULL, a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFFULL, b_hi = b >> 32;
    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;
    uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
    lo = (p0 & 0xFFFFFFFFULL) | (mid << 32);
    hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
#endif
}

inline uint64_t read_le64(const uint8_t *p) noexcept {
    uint64_t v = 0;
    std::memcpy(&v, p, sizeof(uint64_t));
    return v;
}

inline uint64_t read_le32(const uint8_t *p) noexcept {
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(uint32_t));
    return v;
}

class Hasher128 {
public:
    explicit Hasher128(uint64_t seed_lo = 0xa0761d6478bd642fULL,
                       uint64_t seed_hi = 0xe7037ed1a0b428dbULL) noexcept
        : m_s0(seed_lo), m_s1(seed_hi) {}

    void update(const void *data, size_t len) noexcept {
        const uint8_t *p = static_cast<const uint8_t *>(data);
        m_total_len += len;

        if (m_buf_len > 0) {
            size_t take = std::min(len, 16 - m_buf_len);
            std::memcpy(m_buf + m_buf_len, p, take);
            m_buf_len += take;
            p += take;
            len -= take;
            if (m_buf_len == 16) {
                consume_block(read_le64(m_buf), read_le64(m_buf + 8));
                m_buf_len = 0;
            }
        }

        while (len >= 16) {
            consume_block(read_le64(p), read_le64(p + 8));
            p += 16;
            len -= 16;
        }

        if (len > 0) {
            std::memcpy(m_buf, p, len);
            m_buf_len = len;
        }
    }

    template <typename T>
    void update_pod(const T &v) noexcept {
        static_assert(std::is_trivially_copyable_v<T>, "POD required");
        update(&v, sizeof(T));
    }

    void update_str(const std::string &s) noexcept {
        uint64_t n = static_cast<uint64_t>(s.size());
        update_pod(n);
        if (!s.empty()) update(s.data(), s.size());
    }

    Hash128 finalize() const noexcept {
        uint64_t s0 = m_s0 ^ m_total_len;
        uint64_t s1 = m_s1 ^ (m_total_len * 0x8ebc6af09c88c6e3ULL);
        if (m_buf_len > 0) {
            uint8_t tail[16] = {0};
            std::memcpy(tail, m_buf, m_buf_len);
            tail[15] = static_cast<uint8_t>(m_buf_len);
            uint64_t a = read_le64(tail);
            uint64_t b = read_le64(tail + 8);
            uint64_t lo0, hi0, lo1, hi1;
            wymix128(a ^ 0xa0761d6478bd642fULL, s0 ^ 0xe7037ed1a0b428dbULL, lo0, hi0);
            wymix128(b ^ 0x8ebc6af09c88c6e3ULL, s1 ^ 0x589965cc75374cc3ULL, lo1, hi1);
            s0 = lo0 ^ hi1;
            s1 = lo1 ^ hi0;
        }
        uint64_t f0_lo, f0_hi, f1_lo, f1_hi;
        wymix128(s0 ^ 0x1d8e4e27c47d124fULL, s1 ^ 0xa0761d6478bd642fULL, f0_lo, f0_hi);
        wymix128(s1 ^ 0xe7037ed1a0b428dbULL, f0_lo ^ 0x589965cc75374cc3ULL, f1_lo, f1_hi);
        return Hash128{f0_lo ^ f1_hi, f1_lo ^ f0_hi};
    }

private:
    void consume_block(uint64_t a, uint64_t b) noexcept {
        uint64_t lo0, hi0, lo1, hi1;
        wymix128(a ^ 0xa0761d6478bd642fULL, m_s0 ^ b, lo0, hi0);
        wymix128(b ^ 0x8ebc6af09c88c6e3ULL, m_s1 ^ a, lo1, hi1);
        m_s0 = lo0 ^ hi1;
        m_s1 = lo1 ^ hi0;
    }

    uint64_t m_s0;
    uint64_t m_s1;
    uint64_t m_total_len = 0;
    size_t   m_buf_len   = 0;
    uint8_t  m_buf[16]   = {0};
};

inline uint64_t hash64_bytes(const void *data, size_t len, uint64_t seed = 0x517cc1b727220a95ULL) noexcept {
    if (!data || len == 0) return 0;
    Hasher128 h(seed, seed ^ 0x9e3779b97f4a7c15ULL);
    h.update(data, len);
    Hash128 r = h.finalize();
    return r.lo ^ r.hi;
}

inline Hash128 hash128_bytes(const void *data, size_t len, uint64_t seed = 0xa0761d6478bd642fULL) noexcept {
    Hasher128 h(seed, seed ^ 0xe7037ed1a0b428dbULL);
    if (data && len > 0) h.update(data, len);
    return h.finalize();
}

inline uint64_t unix_time_ms() noexcept {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}

inline bool pread_exact(int fd, void *dst, size_t len, uint64_t offset) noexcept {
#if defined(__unix__) || defined(__APPLE__)
    uint8_t *p = static_cast<uint8_t *>(dst);
    size_t done = 0;
    while (done < len) {
        ssize_t n = ::pread(fd, p + done, len - done, static_cast<off_t>(offset + done));
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
#else
    (void)fd; (void)dst; (void)len; (void)offset;
    return false;
#endif
}

inline bool write_exact(int fd, const void *src, size_t len) noexcept {
#if defined(__unix__) || defined(__APPLE__)
    const uint8_t *p = static_cast<const uint8_t *>(src);
    size_t done = 0;
    while (done < len) {
        ssize_t n = ::write(fd, p + done, len - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
#else
    (void)fd; (void)src; (void)len;
    return false;
#endif
}

} // namespace detail

// =============================================================================
// Exact 256-Byte .kvb ("LFMKVB01") On-Disk Header & Geometry Structures
// =============================================================================

struct alignas(64) KvbHeader {
    char     magic[8];              // "LFMKVB01"
    uint32_t format_version;        // 1
    uint32_t header_bytes;          // 256
    uint32_t flags;                 // KvbFlags bitmask
    uint32_t arch_id;               // ModelArchId
    uint32_t block_tokens;          // 64
    uint32_t block_idx;             // k (covers token positions [64*k .. 64*k+63])

    uint16_t hidden_dim;            // 2048
    uint16_t num_glob_layers;       // 6
    uint16_t num_swa_layers;        // 5 (LFM3) or 0 (LFM2.5)
    uint16_t num_conv_layers;       // 13 (LFM3) or 18 (LFM2.5)
    uint16_t kv_heads;              // 4 (LFM3) or 8 (LFM2.5)
    uint16_t head_dim;              // 128 (LFM3) or 64 (LFM2.5)
    uint16_t conv_state_taps;       // 6 (LFM3) or 2 (LFM2.5)
    uint16_t reserved_u16;          // 0

    Hash128  model_fingerprint;     // Hash(weights_die0/1 FNV + quant_chan_scales + tp/ep mode)
    Hash128  parent_key;            // {0,0} for block_idx == 0, else block_key[k-1]
    Hash128  block_key;             // Chained 128-bit key for this 64-token block
    Hash128  vision_hash;           // Hash of visual_embeds spanning this block (0 if text-only)

    uint64_t off_glob_u8;           // 12288 (0x3000)
    uint64_t len_glob_u8_raw;       // 408576 (LFM3) or 423936 (LFM2.5)
    uint64_t len_glob_u8_aligned;   // 409600 (LFM3) or 425984 (LFM2.5)
    uint64_t off_swa_f32;           // 421888 (LFM3) or 438272 (LFM2.5)
    uint64_t len_swa_f32;           // 1310720 (LFM3) or 0 (LFM2.5)
    uint64_t off_conv_f32;          // 1732608 (LFM3) or 438272 (LFM2.5)
    uint64_t len_conv_f32;          // 638976 (LFM3) or 294912 (LFM2.5)
    uint64_t off_glob_f32;          // 2371584 (LFM3) or 733184 (LFM2.5) when block_idx < 4, else 0
    uint64_t len_glob_f32;          // 1572864 when block_idx < 4, else 0

    uint64_t payload_checksum_lo;   // 64-bit integrity hash of Sec 0 payload [0x0100..off_glob_u8) + Sec 1
    uint64_t payload_checksum_hi;   // 64-bit integrity hash of Sec 2 (SWA)
    uint64_t conv_checksum;         // 64-bit integrity hash of Sec 3 (ShortConv)
    uint64_t glob_f32_checksum;     // 64-bit integrity hash of Sec 4 (Short-Ctx Global FP32)
    uint64_t created_unix_ms;
    uint8_t  reserved_pad[32];
};
static_assert(sizeof(KvbHeader) == 256, "KvbHeader must be exactly 256 bytes");

struct ModelCacheGeometry {
    ModelArchId arch_id         = ModelArchId::LFM3_1_7B;
    std::string model_namespace = "lfm3-1.7b";
    Hash128     model_fingerprint{};
    uint16_t    hidden_dim      = 2048;
    uint16_t    num_glob_layers = 6;
    uint16_t    num_swa_layers  = 5;
    uint16_t    num_conv_layers = 13;
    uint16_t    kv_heads        = 4;
    uint16_t    head_dim        = 128;
    uint16_t    conv_state_taps = 6;
    uint32_t    swa_cap         = 320; // kSwaCap = 256 + 64
    uint32_t    max_ctx         = 4096;

    static ModelCacheGeometry make_lfm3(const std::string &ns, Hash128 fp, uint32_t max_ctx = 4096) {
        ModelCacheGeometry g;
        g.arch_id           = ModelArchId::LFM3_1_7B;
        g.model_namespace   = ns;
        g.model_fingerprint = fp;
        g.hidden_dim        = 2048;
        g.num_glob_layers   = 6;
        g.num_swa_layers    = 5;
        g.num_conv_layers   = 13;
        g.kv_heads          = 4;
        g.head_dim          = 128;
        g.conv_state_taps   = 6;
        g.swa_cap           = 320;
        g.max_ctx           = max_ctx;
        return g;
    }

    static ModelCacheGeometry make_lfm2_moe(const std::string &ns, Hash128 fp, uint32_t max_ctx = 4096) {
        ModelCacheGeometry g;
        g.arch_id           = ModelArchId::LFM2_5_8B_A1B;
        g.model_namespace   = ns;
        g.model_fingerprint = fp;
        g.hidden_dim        = 2048;
        g.num_glob_layers   = 6;
        g.num_swa_layers    = 0;
        g.num_conv_layers   = 18;
        g.kv_heads          = 8;
        g.head_dim          = 64;
        g.conv_state_taps   = 2;
        g.swa_cap           = 0;
        g.max_ctx           = max_ctx;
        return g;
    }
};

struct KvbSectionLayout {
    size_t sec0_bytes          = 0; // Header (256) + tokens (256) + last_h (hidden*4) -> 4K aligned
    size_t off_glob_u8         = 0;
    size_t k_u8_bytes          = 0;
    size_t v_u8_bytes          = 0;
    size_t k_sc_bytes          = 0;
    size_t v_sc_bytes          = 0;
    size_t k_sum_bytes         = 0;
    size_t len_glob_u8_raw     = 0;
    size_t len_glob_u8_aligned = 0;
    size_t off_swa_f32         = 0;
    size_t swa_half_bytes      = 0;
    size_t len_swa_f32         = 0;
    size_t off_conv_f32        = 0;
    size_t raw_conv_f32        = 0;
    size_t len_conv_f32        = 0;
    size_t off_glob_f32        = 0;
    size_t glob_f32_half_bytes = 0;
    size_t len_glob_f32        = 0;
    size_t total_file_bytes    = 0;

    static KvbSectionLayout compute(const ModelCacheGeometry &g, bool include_glob_f32) noexcept {
        KvbSectionLayout L{};
        const size_t B = kBlockTokens;
        L.sec0_bytes = align_up_4k(sizeof(KvbHeader) + B * sizeof(int32_t) +
                                   static_cast<size_t>(g.hidden_dim) * sizeof(float));
        L.off_glob_u8 = L.sec0_bytes;

        const size_t gh = static_cast<size_t>(g.num_glob_layers) * g.kv_heads;
        L.k_u8_bytes  = gh * B * g.head_dim * sizeof(uint8_t);
        L.v_u8_bytes  = gh * B * g.head_dim * sizeof(uint8_t);
        L.k_sc_bytes  = gh * B * sizeof(float);
        L.v_sc_bytes  = gh * B * sizeof(float);
        L.k_sum_bytes = gh * B * sizeof(uint16_t);
        L.len_glob_u8_raw     = L.k_u8_bytes + L.v_u8_bytes + L.k_sc_bytes + L.v_sc_bytes + L.k_sum_bytes;
        L.len_glob_u8_aligned = align_up_4k(L.len_glob_u8_raw);

        L.off_swa_f32 = L.off_glob_u8 + L.len_glob_u8_aligned;
        if (g.num_swa_layers > 0) {
            L.swa_half_bytes = static_cast<size_t>(g.num_swa_layers) * g.kv_heads * B * g.head_dim * sizeof(float);
            L.len_swa_f32    = align_up_4k(2 * L.swa_half_bytes);
        }

        L.off_conv_f32 = L.off_swa_f32 + L.len_swa_f32;
        L.raw_conv_f32 = static_cast<size_t>(g.num_conv_layers) * g.hidden_dim * g.conv_state_taps * sizeof(float);
        L.len_conv_f32 = align_up_4k(L.raw_conv_f32);

        if (include_glob_f32) {
            L.off_glob_f32        = L.off_conv_f32 + L.len_conv_f32;
            L.glob_f32_half_bytes = gh * B * g.head_dim * sizeof(float);
            L.len_glob_f32        = align_up_4k(2 * L.glob_f32_half_bytes);
            L.total_file_bytes    = L.off_glob_f32 + L.len_glob_f32;
        } else {
            L.off_glob_f32        = 0;
            L.glob_f32_half_bytes = 0;
            L.len_glob_f32        = 0;
            L.total_file_bytes    = L.off_conv_f32 + L.len_conv_f32;
        }
        return L;
    }
};

// Non-owning pointers into Lfm3Engine's live NUMA-pinned state buffers
struct EngineStateView {
    uint8_t  *glob_k_u8  = nullptr; // [num_glob_layers][kv_heads][max_ctx][head_dim]
    uint8_t  *glob_v_u8  = nullptr;
    float    *glob_k_sc  = nullptr; // [num_glob_layers][kv_heads][max_ctx]
    float    *glob_v_sc  = nullptr;
    uint16_t *glob_k_sum = nullptr; // [num_glob_layers][kv_heads][max_ctx]
    float    *glob_k_f32 = nullptr; // [num_glob_layers][kv_heads][max_ctx][head_dim] (pos < 256)
    float    *glob_v_f32 = nullptr;
    float    *swa_k_f32  = nullptr; // [num_swa_layers][kv_heads][swa_cap][head_dim]
    float    *swa_v_f32  = nullptr;
    float    *conv_state = nullptr; // [num_conv_layers][hidden_dim][conv_state_taps]
};

struct PrefixLookupResult {
    int                      matched_blocks = 0;
    int                      matched_tokens = 0;
    std::vector<Hash128>     chain_keys;
    std::vector<Hash128>     vision_hashes;
    std::vector<std::string> hit_paths;
};

struct CacheTelemetry {
    int         prompt_tokens       = 0;
    int         cached_tokens       = 0; // OpenAI usage.prompt_tokens_details.cached_tokens
    int         cached_tier0_tokens = 0; // Hot in-RAM sticky turn state
    int         cached_disk_tokens  = 0; // Restored from NVMe .kvb
    int         computed_tokens     = 0; // Evaluated on NPU this turn
    double      lookup_ms           = 0.0;
    double      restore_ms          = 0.0;
    double      async_enqueue_ms    = 0.0;
    std::string cache_status;            // "HIT" | "PARTIAL" | "MISS" | "BYPASS"
    std::string cache_tier;              // "ram_sticky" | "nvme_disk" | "mixed" | "none"
};

struct CacheStatsSnapshot {
    uint64_t resident_blocks    = 0;
    uint64_t pinned_blocks      = 0;
    uint64_t disk_bytes_used    = 0;
    uint64_t disk_bytes_quota   = 0;
    uint64_t total_lookups      = 0;
    uint64_t total_block_hits   = 0;
    uint64_t total_tokens_saved = 0;
    uint64_t total_evictions    = 0;
    uint64_t pending_writes     = 0;
};

// =============================================================================
// RadixCache: Chained Radix Index + WAL + Selective Restore + GDSF Eviction
// =============================================================================

class RadixCache {
public:
    RadixCache() = default;

    ~RadixCache() {
        shutdown();
    }

    RadixCache(const RadixCache &) = delete;
    RadixCache &operator=(const RadixCache &) = delete;

    void shutdown() {
        flush_async_writes();
        {
            std::lock_guard<std::mutex> lk(m_writer_mtx);
            m_writer_stop = true;
        }
        m_writer_cv.notify_all();
        if (m_writer_thread.joinable()) {
            m_writer_thread.join();
        }
        std::lock_guard<std::mutex> lk(m_mtx);
        if (m_initialized) {
            write_snapshot_locked();
            m_initialized = false;
        }
    }

    bool init(const std::filesystem::path &root_dir,
              const ModelCacheGeometry &geom,
              uint64_t max_disk_bytes = 256ULL * 1024 * 1024 * 1024) {
        shutdown();

        if (geom.hidden_dim == 0 || geom.num_glob_layers == 0 || geom.kv_heads == 0 ||
            geom.head_dim == 0 || geom.max_ctx < kBlockTokens) {
            return false;
        }
        if (geom.num_swa_layers > 0 && geom.swa_cap < 256) {
            return false;
        }

        std::lock_guard<std::mutex> lk(m_mtx);
        m_geom           = geom;
        m_max_disk_bytes = max_disk_bytes;
        m_ns_dir         = root_dir / sanitize_ns(geom.model_namespace);

        std::error_code ec;
        std::filesystem::create_directories(m_ns_dir, ec);
        if (ec) return false;

        // Pre-create 256 fanout directories ("00" .. "ff") so background writes never contend on mkdir
        static constexpr char kHex[] = "0123456789abcdef";
        for (int i = 0; i < 256; ++i) {
            char sub[3] = {kHex[(i >> 4) & 0xF], kHex[i & 0xF], '\0'};
            std::filesystem::create_directories(m_ns_dir / sub, ec);
            if (ec) return false;
        }

        m_snap_path = m_ns_dir / "index.snap";
        m_wal_path  = m_ns_dir / "index.wal";

        m_nodes.clear();
        m_inflight_blobs.clear();
        m_disk_bytes_used    = 0;
        m_clock              = 0.0;
        m_tick_seq           = 0;
        m_total_lookups      = 0;
        m_total_block_hits   = 0;
        m_total_tokens_saved = 0;
        m_total_evictions    = 0;

        cleanup_orphan_temps_locked();
        load_index_locked();

        m_writer_stop = false;
        m_writer_thread = std::thread([this]() { writer_loop(); });
        m_initialized = true;
        return true;
    }

    Hash128 compute_block_key(uint32_t block_idx,
                              const Hash128 &parent_key,
                              const int32_t token_ids[kBlockTokens],
                              const Hash128 &vision_hash = Hash128{},
                              const std::string &cache_namespace = "") const noexcept {
        detail::Hasher128 h;
        h.update(kKvbMagic, sizeof(kKvbMagic));
        h.update_pod(kKvbFormatVersion);
        uint32_t arch = static_cast<uint32_t>(m_geom.arch_id);
        h.update_pod(arch);
        h.update_pod(m_geom.hidden_dim);
        h.update_pod(m_geom.num_glob_layers);
        h.update_pod(m_geom.num_swa_layers);
        h.update_pod(m_geom.num_conv_layers);
        h.update_pod(m_geom.kv_heads);
        h.update_pod(m_geom.head_dim);
        h.update_pod(m_geom.conv_state_taps);
        h.update_pod(m_geom.model_fingerprint.lo);
        h.update_pod(m_geom.model_fingerprint.hi);
        h.update_str(cache_namespace);
        h.update_pod(block_idx);
        h.update_pod(parent_key.lo);
        h.update_pod(parent_key.hi);
        h.update(token_ids, kBlockTokens * sizeof(int32_t));
        h.update_pod(vision_hash.lo);
        h.update_pod(vision_hash.hi);
        return h.finalize();
    }

    PrefixLookupResult lookup(const int32_t *tokens,
                              size_t n_tokens,
                              const float *visual_embeds = nullptr,
                              size_t num_visual_tokens = 0,
                              int32_t image_token_id = 125030,
                              const std::string &cache_namespace = "") {
        PrefixLookupResult res;
        if (!tokens || n_tokens < kBlockTokens) {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_total_lookups++;
            return res;
        }

        const size_t max_blocks_ctx = m_geom.max_ctx / kBlockTokens;
        const size_t n_full_blocks  = std::min(n_tokens / kBlockTokens, max_blocks_ctx);
        res.chain_keys.resize(n_full_blocks);
        res.vision_hashes.resize(n_full_blocks);

        size_t vis_cursor = 0;
        Hash128 parent{};
        for (size_t b = 0; b < n_full_blocks; ++b) {
            const int32_t *blk_toks = tokens + b * kBlockTokens;
            size_t img_cnt = 0;
            for (size_t i = 0; i < kBlockTokens; ++i) {
                if (blk_toks[i] == image_token_id) img_cnt++;
            }
            Hash128 vhash{};
            if (img_cnt > 0) {
                detail::Hasher128 vh(0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL);
                vh.update_pod(img_cnt);
                if (visual_embeds && vis_cursor < num_visual_tokens) {
                    size_t avail = std::min(img_cnt, num_visual_tokens - vis_cursor);
                    const float *vp = visual_embeds + vis_cursor * m_geom.hidden_dim;
                    vh.update(vp, avail * static_cast<size_t>(m_geom.hidden_dim) * sizeof(float));
                    vis_cursor += avail;
                }
                vhash = vh.finalize();
            }
            res.vision_hashes[b] = vhash;
            Hash128 key = compute_block_key(static_cast<uint32_t>(b), parent, blk_toks, vhash, cache_namespace);
            res.chain_keys[b] = key;
            parent = key;
        }

        std::lock_guard<std::mutex> lk(m_mtx);
        m_total_lookups++;
        Hash128 exp_parent{};
        for (size_t b = 0; b < n_full_blocks; ++b) {
            const Hash128 &k = res.chain_keys[b];
            auto it = m_nodes.find(k);
            if (it == m_nodes.end()) break;
            CacheNode &node = it->second;
            if (node.block_idx != b || node.parent_key != exp_parent) break;

            const bool in_mem = (m_inflight_blobs.find(k) != m_inflight_blobs.end());
            std::filesystem::path p = kvb_path_for(k);
            if (!in_mem && !node.on_disk) break;
            if (!in_mem) {
                std::error_code ec;
                if (!std::filesystem::exists(p, ec)) {
                    remove_subtree_locked(k);
                    break;
                }
            }

            node.hit_count++;
            node.last_access_tick = ++m_tick_seq;
            update_gdsf_key(node);

            res.matched_blocks = static_cast<int>(b + 1);
            res.matched_tokens = res.matched_blocks * static_cast<int>(kBlockTokens);
            res.hit_paths.push_back(p.string());
            exp_parent = k;
        }

        if (res.matched_blocks > 0) {
            m_total_block_hits   += static_cast<uint64_t>(res.matched_blocks);
            m_total_tokens_saved += static_cast<uint64_t>(res.matched_tokens);
        }
        return res;
    }

    bool restore_blocks(const PrefixLookupResult &hit,
                        int blocks_to_restore,
                        EngineStateView &view,
                        float *out_last_h,
                        double &out_restore_ms,
                        const int32_t *expected_tokens = nullptr) {
        auto t0 = std::chrono::steady_clock::now();
        out_restore_ms = 0.0;

        if (blocks_to_restore <= 0 || blocks_to_restore > hit.matched_blocks) return false;
        if (static_cast<uint32_t>(blocks_to_restore) * kBlockTokens > m_geom.max_ctx) return false;
        if (!view.glob_k_u8 || !view.glob_v_u8 || !view.glob_k_sc || !view.glob_v_sc ||
            !view.glob_k_sum || !view.conv_state) {
            return false;
        }
        if (m_geom.num_swa_layers > 0 && (!view.swa_k_f32 || !view.swa_v_f32)) {
            return false;
        }

        const int matched_tokens   = blocks_to_restore * static_cast<int>(kBlockTokens);
        const bool need_glob_f32   = (matched_tokens < 256) &&
                                     (view.glob_k_f32 != nullptr) && (view.glob_v_f32 != nullptr);
        const int swa_start_blk    = (m_geom.num_swa_layers > 0)
                                     ? std::max(0, blocks_to_restore - 4)
                                     : blocks_to_restore;
        const int term_blk         = blocks_to_restore - 1;

        // Snapshot any in-flight memory buffers under lock so file I/O runs completely lock-free
        std::vector<std::shared_ptr<const std::vector<uint8_t>>> mem_blobs(blocks_to_restore);
        std::vector<std::string> paths(blocks_to_restore);
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            for (int b = 0; b < blocks_to_restore; ++b) {
                const Hash128 &k = hit.chain_keys[b];
                auto it_mem = m_inflight_blobs.find(k);
                if (it_mem != m_inflight_blobs.end()) {
                    mem_blobs[b] = it_mem->second;
                }
                paths[b] = (b < static_cast<int>(hit.hit_paths.size()))
                           ? hit.hit_paths[b]
                           : kvb_path_for(k).string();
            }
        }

        std::atomic<int> first_corrupt_blk{std::numeric_limits<int>::max()};

        auto process_one_block = [&](int b) {
            if (first_corrupt_blk.load(std::memory_order_relaxed) <= b) return;

            const bool read_swa      = (b >= swa_start_blk) && (m_geom.num_swa_layers > 0);
            const bool read_conv     = (b == term_blk);
            const bool read_glob_f32 = need_glob_f32 && (b < 4);
            const KvbSectionLayout L = KvbSectionLayout::compute(m_geom, b < 4);

            size_t prefix_read_end = L.off_glob_u8 + L.len_glob_u8_aligned;
            if (read_swa)      prefix_read_end = L.off_swa_f32 + L.len_swa_f32;
            if (read_conv)     prefix_read_end = L.off_conv_f32 + L.len_conv_f32;
            if (read_glob_f32) prefix_read_end = L.off_glob_f32 + L.len_glob_f32;

            std::vector<uint8_t> local_buf;
            const uint8_t *raw = nullptr;
            size_t avail_len = 0;

            if (mem_blobs[b]) {
                raw = mem_blobs[b]->data();
                avail_len = mem_blobs[b]->size();
                if (avail_len < prefix_read_end) {
                    int prev = first_corrupt_blk.load();
                    while (b < prev && !first_corrupt_blk.compare_exchange_weak(prev, b)) {}
                    return;
                }
            } else {
                int fd = ::open(paths[b].c_str(), O_RDONLY);
                if (fd < 0) {
                    int prev = first_corrupt_blk.load();
                    while (b < prev && !first_corrupt_blk.compare_exchange_weak(prev, b)) {}
                    return;
                }
#if defined(POSIX_FADV_SEQUENTIAL)
                // Keep .kvb hot in Linux VFS page cache (NEVER POSIX_FADV_DONTNEED)
                ::posix_fadvise(fd, 0, static_cast<off_t>(prefix_read_end), POSIX_FADV_SEQUENTIAL);
#endif
                local_buf.resize(prefix_read_end);
                bool ok = detail::pread_exact(fd, local_buf.data(), prefix_read_end, 0);
                // If need_glob_f32 is true for b < 4 and b != term_blk, prefix_read_end already
                // spans [0 .. off_glob_f32 + len_glob_f32]. Close fd immediately.
                ::close(fd);
                if (!ok) {
                    int prev = first_corrupt_blk.load();
                    while (b < prev && !first_corrupt_blk.compare_exchange_weak(prev, b)) {}
                    return;
                }
                raw = local_buf.data();
                avail_len = local_buf.size();
            }

            // 1. Validate 256-byte KvbHeader & exact token/hash invariants
            KvbHeader hdr{};
            std::memcpy(&hdr, raw, sizeof(KvbHeader));
            const Hash128 exp_parent = (b == 0) ? Hash128{} : hit.chain_keys[b - 1];
            bool valid = (std::memcmp(hdr.magic, kKvbMagic, 8) == 0) &&
                         (hdr.format_version == kKvbFormatVersion) &&
                         (hdr.header_bytes == sizeof(KvbHeader)) &&
                         (hdr.arch_id == static_cast<uint32_t>(m_geom.arch_id)) &&
                         (hdr.block_tokens == kBlockTokens) &&
                         (hdr.block_idx == static_cast<uint32_t>(b)) &&
                         (hdr.hidden_dim == m_geom.hidden_dim) &&
                         (hdr.num_glob_layers == m_geom.num_glob_layers) &&
                         (hdr.num_swa_layers == m_geom.num_swa_layers) &&
                         (hdr.num_conv_layers == m_geom.num_conv_layers) &&
                         (hdr.kv_heads == m_geom.kv_heads) &&
                         (hdr.head_dim == m_geom.head_dim) &&
                         (hdr.conv_state_taps == m_geom.conv_state_taps) &&
                         (hdr.model_fingerprint == m_geom.model_fingerprint) &&
                         (hdr.parent_key == exp_parent) &&
                         (hdr.block_key == hit.chain_keys[b]) &&
                         (hdr.off_glob_u8 == L.off_glob_u8) &&
                         (hdr.len_glob_u8_aligned == L.len_glob_u8_aligned);

            if (valid && expected_tokens) {
                const int32_t *stored_toks = reinterpret_cast<const int32_t *>(raw + sizeof(KvbHeader));
                if (std::memcmp(stored_toks, expected_tokens + static_cast<size_t>(b) * kBlockTokens,
                                kBlockTokens * sizeof(int32_t)) != 0) {
                    valid = false;
                }
            }

            // 2. Verify per-section 64-bit integrity checksums BEFORE scattering any bytes
            if (valid) {
                const size_t sec01_len = (L.off_glob_u8 - sizeof(KvbHeader)) + L.len_glob_u8_aligned;
                uint64_t cs01 = detail::hash64_bytes(raw + sizeof(KvbHeader), sec01_len, 0x1111222233334444ULL);
                if (cs01 != hdr.payload_checksum_lo) valid = false;
            }
            if (valid && read_swa) {
                uint64_t cs_swa = detail::hash64_bytes(raw + L.off_swa_f32, L.len_swa_f32, 0x5555666677778888ULL);
                if (cs_swa != hdr.payload_checksum_hi) valid = false;
            }
            if (valid && read_conv) {
                uint64_t cs_conv = detail::hash64_bytes(raw + L.off_conv_f32, L.len_conv_f32, 0x9999aaaabbbbccccULL);
                if (cs_conv != hdr.conv_checksum) valid = false;
            }
            if (valid && read_glob_f32) {
                if (!(hdr.flags & KVB_FLAG_HAS_GLOB_FP32) || avail_len < L.off_glob_f32 + L.len_glob_f32) {
                    valid = false;
                } else {
                    uint64_t cs_gf = detail::hash64_bytes(raw + L.off_glob_f32, L.len_glob_f32, 0xddddeeeeffff0001ULL);
                    if (cs_gf != hdr.glob_f32_checksum) valid = false;
                }
            }

            if (!valid) {
                int prev = first_corrupt_blk.load();
                while (b < prev && !first_corrupt_blk.compare_exchange_weak(prev, b)) {}
                return;
            }

            // 3. Scatter Plane 1 (Global KV u8 + sc + k_sum) into Head-Major [6][kv_heads][max_ctx][head_dim]
            scatter_glob_u8_block(static_cast<uint32_t>(b), raw + L.off_glob_u8, L, view);

            // 4. Scatter Plane 2 (SWA 256-token ring delta) for chunks max(0, k-3) .. k
            if (read_swa) {
                scatter_swa_f32_block(static_cast<uint32_t>(b), raw + L.off_swa_f32, L, view);
            }

            // 5. Scatter Plane 3 (ShortConv rolling state) + last_h[hidden_dim] for terminal chunk k
            if (read_conv) {
                std::memcpy(view.conv_state, raw + L.off_conv_f32, L.raw_conv_f32);
                if (out_last_h) {
                    const uint8_t *lh_src = raw + sizeof(KvbHeader) + kBlockTokens * sizeof(int32_t);
                    std::memcpy(out_last_h, lh_src, static_cast<size_t>(m_geom.hidden_dim) * sizeof(float));
                }
            }

            // 6. Scatter Plane 1b (Short-Ctx Global FP32 KV) when matched_tokens < 256
            if (read_glob_f32) {
                scatter_glob_f32_block(static_cast<uint32_t>(b), raw + L.off_glob_f32, L, view);
            }
        };

#if defined(_OPENMP)
        #pragma omp parallel for if(blocks_to_restore >= 4) schedule(dynamic, 1)
#endif
        for (int b = 0; b < blocks_to_restore; ++b) {
            process_one_block(b);
        }

        const int bad = first_corrupt_blk.load();
        if (bad < blocks_to_restore) {
            std::lock_guard<std::mutex> lk(m_mtx);
            remove_subtree_locked(hit.chain_keys[bad]);
            return false;
        }

        auto t1 = std::chrono::steady_clock::now();
        out_restore_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return true;
    }

    void enqueue_save_block(uint32_t block_idx,
                            const Hash128 &parent_key,
                            const Hash128 &block_key,
                            const Hash128 &vision_hash,
                            const int32_t token_ids[kBlockTokens],
                            const float *last_h,
                            const EngineStateView &view,
                            bool pinned = false) {
        if (!token_ids || !view.glob_k_u8 || !view.glob_v_u8 || !view.glob_k_sc ||
            !view.glob_v_sc || !view.glob_k_sum || !view.conv_state) {
            return;
        }
        if ((block_idx + 1) * kBlockTokens > m_geom.max_ctx) return;

        {
            std::lock_guard<std::mutex> lk(m_mtx);
            if (!m_initialized) return;
            auto it = m_nodes.find(block_key);
            if (it != m_nodes.end()) {
                if (pinned && !(it->second.flags & KVB_FLAG_PINNED)) {
                    it->second.flags |= KVB_FLAG_PINNED;
                    append_wal_pin_locked(block_key, true);
                }
                return;
            }
        }

        const bool has_glob_f32 = (block_idx < 4) &&
                                  (view.glob_k_f32 != nullptr) && (view.glob_v_f32 != nullptr);
        const KvbSectionLayout L = KvbSectionLayout::compute(m_geom, has_glob_f32);
        auto blob = std::make_shared<std::vector<uint8_t>>(L.total_file_bytes, 0);
        uint8_t *dst = blob->data();

        // Gather Section 0 payload: token_ids[64] + last_h[hidden_dim]
        std::memcpy(dst + sizeof(KvbHeader), token_ids, kBlockTokens * sizeof(int32_t));
        if (last_h) {
            std::memcpy(dst + sizeof(KvbHeader) + kBlockTokens * sizeof(int32_t),
                        last_h, static_cast<size_t>(m_geom.hidden_dim) * sizeof(float));
        }

        // Gather Section 1: Global KV (u8 + sc + k_sum)
        gather_glob_u8_block(block_idx, view, dst + L.off_glob_u8, L);

        // Gather Section 2: SWA 64-token delta (FP32)
        if (m_geom.num_swa_layers > 0 && view.swa_k_f32 && view.swa_v_f32) {
            gather_swa_f32_block(block_idx, view, dst + L.off_swa_f32, L);
        }

        // Gather Section 3: ShortConv rolling state snapshot (FP32)
        std::memcpy(dst + L.off_conv_f32, view.conv_state, L.raw_conv_f32);

        // Gather Section 4: Short-context Global FP32 KV (blocks 0..3 only)
        if (has_glob_f32) {
            gather_glob_f32_block(block_idx, view, dst + L.off_glob_f32, L);
        }

        // Populate 256-byte KvbHeader & per-section 64-bit integrity checksums
        KvbHeader hdr{};
        std::memcpy(hdr.magic, kKvbMagic, 8);
        hdr.format_version = kKvbFormatVersion;
        hdr.header_bytes   = sizeof(KvbHeader);
        hdr.flags          = KVB_FLAG_NONE;
        if (m_geom.num_swa_layers > 0) hdr.flags |= KVB_FLAG_HAS_SWA;
        if (has_glob_f32)              hdr.flags |= KVB_FLAG_HAS_GLOB_FP32;
        if (!vision_hash.is_zero())    hdr.flags |= KVB_FLAG_HAS_VISION;
        if (pinned)                    hdr.flags |= KVB_FLAG_PINNED;

        hdr.arch_id             = static_cast<uint32_t>(m_geom.arch_id);
        hdr.block_tokens        = kBlockTokens;
        hdr.block_idx           = block_idx;
        hdr.hidden_dim          = m_geom.hidden_dim;
        hdr.num_glob_layers     = m_geom.num_glob_layers;
        hdr.num_swa_layers      = m_geom.num_swa_layers;
        hdr.num_conv_layers     = m_geom.num_conv_layers;
        hdr.kv_heads            = m_geom.kv_heads;
        hdr.head_dim            = m_geom.head_dim;
        hdr.conv_state_taps     = m_geom.conv_state_taps;
        hdr.reserved_u16        = 0;
        hdr.model_fingerprint   = m_geom.model_fingerprint;
        hdr.parent_key          = parent_key;
        hdr.block_key           = block_key;
        hdr.vision_hash         = vision_hash;
        hdr.off_glob_u8         = L.off_glob_u8;
        hdr.len_glob_u8_raw     = L.len_glob_u8_raw;
        hdr.len_glob_u8_aligned = L.len_glob_u8_aligned;
        hdr.off_swa_f32         = L.off_swa_f32;
        hdr.len_swa_f32         = L.len_swa_f32;
        hdr.off_conv_f32        = L.off_conv_f32;
        hdr.len_conv_f32        = L.len_conv_f32;
        hdr.off_glob_f32        = L.off_glob_f32;
        hdr.len_glob_f32        = L.len_glob_f32;

        const size_t sec01_len  = (L.off_glob_u8 - sizeof(KvbHeader)) + L.len_glob_u8_aligned;
        hdr.payload_checksum_lo = detail::hash64_bytes(dst + sizeof(KvbHeader), sec01_len, 0x1111222233334444ULL);
        hdr.payload_checksum_hi = (L.len_swa_f32 > 0)
                                  ? detail::hash64_bytes(dst + L.off_swa_f32, L.len_swa_f32, 0x5555666677778888ULL)
                                  : 0ULL;
        hdr.conv_checksum       = detail::hash64_bytes(dst + L.off_conv_f32, L.len_conv_f32, 0x9999aaaabbbbccccULL);
        hdr.glob_f32_checksum   = (L.len_glob_f32 > 0)
                                  ? detail::hash64_bytes(dst + L.off_glob_f32, L.len_glob_f32, 0xddddeeeeffff0001ULL)
                                  : 0ULL;
        hdr.created_unix_ms     = detail::unix_time_ms();
        std::memcpy(dst, &hdr, sizeof(KvbHeader));

        // Register immediately in RadixCache + in-flight map, pre-evicting lowest-priority
        // unpinned leaves BEFORE inserting the new block (Cherkasova GDSF admission rule:
        // updates m_clock to the evicted victim's priority first so newly admitted blocks get
        // gdsf_key = m_clock + C/S instead of immediately cannibalizing themselves).
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            if (!m_initialized) return;
            if (m_nodes.find(block_key) != m_nodes.end()) return;
            if (!pinned && L.total_file_bytes > m_max_disk_bytes) return;

            if (!pinned && m_disk_bytes_used + L.total_file_bytes > m_max_disk_bytes) {
                evict_to_target_locked(m_max_disk_bytes - L.total_file_bytes,
                                       (block_idx > 0) ? parent_key : Hash128{});
            }
            if (block_idx > 0 && m_nodes.find(parent_key) == m_nodes.end()) {
                // Parent was evicted (or never existed); keep radix tree invariant intact
                return;
            }
            if (!pinned && m_disk_bytes_used + L.total_file_bytes > m_max_disk_bytes) {
                // Remaining blocks are all pinned or protected ancestors; skip unpinned admission
                return;
            }

            CacheNode node;
            node.block_key        = block_key;
            node.parent_key       = parent_key;
            node.block_idx        = block_idx;
            node.file_bytes       = L.total_file_bytes;
            node.flags            = hdr.flags;
            node.hit_count        = 1;
            node.last_access_tick = ++m_tick_seq;
            node.created_unix_ms  = hdr.created_unix_ms;
            node.on_disk          = false;
            update_gdsf_key(node);

            if (block_idx > 0) {
                m_nodes[parent_key].num_children++;
            }
            m_nodes.emplace(block_key, node);
            m_inflight_blobs[block_key] = blob;
            m_disk_bytes_used += node.file_bytes;

            if (pinned && m_disk_bytes_used > m_max_disk_bytes) {
                evict_to_target_locked(m_max_disk_bytes, block_key);
            }
        }

        {
            std::lock_guard<std::mutex> wlk(m_writer_mtx);
            m_write_queue.push_back(WriteJob{block_key, parent_key, block_idx, hdr.flags,
                                             hdr.created_unix_ms, std::move(blob)});
        }
        m_writer_cv.notify_one();
    }

    size_t evict_to_quota() {
        std::lock_guard<std::mutex> lk(m_mtx);
        return evict_to_quota_locked();
    }

    bool set_pinned(const std::vector<Hash128> &keys, bool pinned) {
        std::lock_guard<std::mutex> lk(m_mtx);
        bool any = false;
        for (const Hash128 &k : keys) {
            auto it = m_nodes.find(k);
            if (it == m_nodes.end()) continue;
            uint32_t old_f = it->second.flags;
            if (pinned) it->second.flags |= KVB_FLAG_PINNED;
            else        it->second.flags &= ~KVB_FLAG_PINNED;
            if (it->second.flags != old_f) {
                append_wal_pin_locked(k, pinned);
                any = true;
            }
        }
        return any;
    }

    size_t clear(bool include_pinned = false) {
        flush_async_writes();
        std::lock_guard<std::mutex> lk(m_mtx);
        size_t removed = 0;
        if (include_pinned) {
            for (const auto &kv : m_nodes) {
                std::error_code ec;
                std::filesystem::remove(kvb_path_for(kv.first), ec);
                removed++;
            }
            m_nodes.clear();
            m_inflight_blobs.clear();
            m_disk_bytes_used = 0;
            write_snapshot_locked();
            return removed;
        }

        // Leaf-by-leaf removal of all unpinned blocks (protects pinned blocks & their ancestors)
        while (true) {
            std::vector<Hash128> leaves;
            for (const auto &kv : m_nodes) {
                if (kv.second.num_children == 0 && !(kv.second.flags & KVB_FLAG_PINNED)) {
                    leaves.push_back(kv.first);
                }
            }
            if (leaves.empty()) break;
            for (const Hash128 &k : leaves) {
                evict_single_leaf_locked(k, false);
                removed++;
            }
        }
        write_snapshot_locked();
        return removed;
    }

    void flush_async_writes() {
        std::unique_lock<std::mutex> wlk(m_writer_mtx);
        m_flushed_cv.wait(wlk, [&]() {
            return m_write_queue.empty() && !m_writer_active;
        });
    }

    std::filesystem::path block_file_path(const Hash128 &block_key) const {
        std::lock_guard<std::mutex> lk(m_mtx);
        return kvb_path_for(block_key);
    }

    CacheStatsSnapshot stats() const {
        CacheStatsSnapshot s{};
        {
            std::lock_guard<std::mutex> wlk(m_writer_mtx);
            s.pending_writes = m_write_queue.size() + (m_writer_active ? 1 : 0);
        }
        std::lock_guard<std::mutex> lk(m_mtx);
        s.resident_blocks    = m_nodes.size();
        s.disk_bytes_used    = m_disk_bytes_used;
        s.disk_bytes_quota   = m_max_disk_bytes;
        s.total_lookups      = m_total_lookups;
        s.total_block_hits   = m_total_block_hits;
        s.total_tokens_saved = m_total_tokens_saved;
        s.total_evictions    = m_total_evictions;
        for (const auto &kv : m_nodes) {
            if (kv.second.flags & KVB_FLAG_PINNED) s.pinned_blocks++;
        }
        return s;
    }

private:
    struct CacheNode {
        Hash128  block_key{};
        Hash128  parent_key{};
        uint32_t block_idx        = 0;
        uint64_t file_bytes       = 0;
        uint32_t flags            = 0;
        uint64_t hit_count        = 1;
        uint64_t last_access_tick = 0;
        uint64_t created_unix_ms  = 0;
        double   gdsf_key         = 0.0;
        uint32_t num_children     = 0;
        bool     on_disk          = false;
    };

    struct WriteJob {
        Hash128                                block_key;
        Hash128                                parent_key;
        uint32_t                               block_idx;
        uint32_t                               flags;
        uint64_t                               created_unix_ms;
        std::shared_ptr<std::vector<uint8_t>>  blob;
    };

    static std::string sanitize_ns(const std::string &ns) {
        std::string out;
        out.reserve(ns.size());
        for (char c : ns) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.') {
                out.push_back(c);
            } else {
                out.push_back('_');
            }
        }
        return out.empty() ? "default_model" : out;
    }

    std::filesystem::path kvb_path_for(const Hash128 &k) const {
        std::string hex = k.to_hex();
        return m_ns_dir / hex.substr(0, 2) / (hex + ".kvb");
    }

    void update_gdsf_key(CacheNode &n) const noexcept {
        // Deeper blocks save slightly higher O(N) attention prefill cost
        const double cost_weight = 1.0 + 0.05 * static_cast<double>(n.block_idx);
        const double size_mib    = std::max(0.0625, static_cast<double>(n.file_bytes) / (1024.0 * 1024.0));
        n.gdsf_key = m_clock + (static_cast<double>(n.hit_count) * cost_weight) / size_mib;
    }

    // -------------------------------------------------------------------------
    // Head-Major Gather / Scatter Helpers (NUMA-friendly across KV heads)
    // -------------------------------------------------------------------------

    void gather_glob_u8_block(uint32_t block_idx, const EngineStateView &view,
                              uint8_t *sec1, const KvbSectionLayout &L) const noexcept {
        const size_t B     = kBlockTokens;
        const size_t G     = m_geom.num_glob_layers;
        const size_t H     = m_geom.kv_heads;
        const size_t D     = m_geom.head_dim;
        const size_t s_cap = m_geom.max_ctx;
        const size_t s0    = static_cast<size_t>(block_idx) * B;

        uint8_t  *dst_ku8  = sec1;
        uint8_t  *dst_vu8  = dst_ku8 + L.k_u8_bytes;
        float    *dst_ksc  = reinterpret_cast<float *>(dst_vu8 + L.v_u8_bytes);
        float    *dst_vsc  = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(dst_ksc) + L.k_sc_bytes);
        uint16_t *dst_ksum = reinterpret_cast<uint16_t *>(reinterpret_cast<uint8_t *>(dst_vsc) + L.v_sc_bytes);

        for (size_t g = 0; g < G; ++g) {
            for (size_t kh = 0; kh < H; ++kh) {
                const size_t gh_idx   = g * H + kh;
                const size_t src_slot = gh_idx * s_cap + s0;
                const size_t dst_slot = gh_idx * B;

                std::memcpy(dst_ku8 + dst_slot * D, view.glob_k_u8 + src_slot * D, B * D);
                std::memcpy(dst_vu8 + dst_slot * D, view.glob_v_u8 + src_slot * D, B * D);
                std::memcpy(dst_ksc + dst_slot,     view.glob_k_sc + src_slot,     B * sizeof(float));
                std::memcpy(dst_vsc + dst_slot,     view.glob_v_sc + src_slot,     B * sizeof(float));
                std::memcpy(dst_ksum + dst_slot,    view.glob_k_sum + src_slot,    B * sizeof(uint16_t));
            }
        }
    }

    void scatter_glob_u8_block(uint32_t block_idx, const uint8_t *sec1,
                               const KvbSectionLayout &L, EngineStateView &view) const noexcept {
        const size_t B     = kBlockTokens;
        const size_t G     = m_geom.num_glob_layers;
        const size_t H     = m_geom.kv_heads;
        const size_t D     = m_geom.head_dim;
        const size_t s_cap = m_geom.max_ctx;
        const size_t s0    = static_cast<size_t>(block_idx) * B;

        const uint8_t  *src_ku8  = sec1;
        const uint8_t  *src_vu8  = src_ku8 + L.k_u8_bytes;
        const float    *src_ksc  = reinterpret_cast<const float *>(src_vu8 + L.v_u8_bytes);
        const float    *src_vsc  = reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(src_ksc) + L.k_sc_bytes);
        const uint16_t *src_ksum = reinterpret_cast<const uint16_t *>(reinterpret_cast<const uint8_t *>(src_vsc) + L.v_sc_bytes);

        for (size_t g = 0; g < G; ++g) {
            for (size_t kh = 0; kh < H; ++kh) {
                const size_t gh_idx   = g * H + kh;
                const size_t dst_slot = gh_idx * s_cap + s0;
                const size_t src_slot = gh_idx * B;

                std::memcpy(view.glob_k_u8 + dst_slot * D, src_ku8 + src_slot * D, B * D);
                std::memcpy(view.glob_v_u8 + dst_slot * D, src_vu8 + src_slot * D, B * D);
                std::memcpy(view.glob_k_sc + dst_slot,     src_ksc + src_slot,     B * sizeof(float));
                std::memcpy(view.glob_v_sc + dst_slot,     src_vsc + src_slot,     B * sizeof(float));
                std::memcpy(view.glob_k_sum + dst_slot,    src_ksum + src_slot,    B * sizeof(uint16_t));
            }
        }
    }

    void gather_swa_f32_block(uint32_t block_idx, const EngineStateView &view,
                              uint8_t *sec2, const KvbSectionLayout &L) const noexcept {
        const size_t B       = kBlockTokens;
        const size_t S       = m_geom.num_swa_layers;
        const size_t H       = m_geom.kv_heads;
        const size_t D       = m_geom.head_dim;
        const size_t swa_cap = m_geom.swa_cap;
        const size_t abs_base = static_cast<size_t>(block_idx) * B;

        float *dst_k = reinterpret_cast<float *>(sec2);
        float *dst_v = reinterpret_cast<float *>(sec2 + L.swa_half_bytes);

        for (size_t l = 0; l < S; ++l) {
            for (size_t kh = 0; kh < H; ++kh) {
                const size_t lh_idx  = l * H + kh;
                const float *src_kh  = view.swa_k_f32 + lh_idx * swa_cap * D;
                const float *src_vh  = view.swa_v_f32 + lh_idx * swa_cap * D;
                float       *dst_kh  = dst_k + lh_idx * B * D;
                float       *dst_vh  = dst_v + lh_idx * B * D;

                const size_t sl0 = abs_base % swa_cap;
                if (sl0 + B <= swa_cap) {
                    std::memcpy(dst_kh, src_kh + sl0 * D, B * D * sizeof(float));
                    std::memcpy(dst_vh, src_vh + sl0 * D, B * D * sizeof(float));
                } else {
                    for (size_t t = 0; t < B; ++t) {
                        size_t sl = (abs_base + t) % swa_cap;
                        std::memcpy(dst_kh + t * D, src_kh + sl * D, D * sizeof(float));
                        std::memcpy(dst_vh + t * D, src_vh + sl * D, D * sizeof(float));
                    }
                }
            }
        }
    }

    void scatter_swa_f32_block(uint32_t block_idx, const uint8_t *sec2,
                               const KvbSectionLayout &L, EngineStateView &view) const noexcept {
        const size_t B       = kBlockTokens;
        const size_t S       = m_geom.num_swa_layers;
        const size_t H       = m_geom.kv_heads;
        const size_t D       = m_geom.head_dim;
        const size_t swa_cap = m_geom.swa_cap;
        const size_t abs_base = static_cast<size_t>(block_idx) * B;

        const float *src_k = reinterpret_cast<const float *>(sec2);
        const float *src_v = reinterpret_cast<const float *>(sec2 + L.swa_half_bytes);

        for (size_t l = 0; l < S; ++l) {
            for (size_t kh = 0; kh < H; ++kh) {
                const size_t lh_idx  = l * H + kh;
                float       *dst_kh  = view.swa_k_f32 + lh_idx * swa_cap * D;
                float       *dst_vh  = view.swa_v_f32 + lh_idx * swa_cap * D;
                const float *src_kh  = src_k + lh_idx * B * D;
                const float *src_vh  = src_v + lh_idx * B * D;

                const size_t sl0 = abs_base % swa_cap;
                if (sl0 + B <= swa_cap) {
                    std::memcpy(dst_kh + sl0 * D, src_kh, B * D * sizeof(float));
                    std::memcpy(dst_vh + sl0 * D, src_vh, B * D * sizeof(float));
                } else {
                    for (size_t t = 0; t < B; ++t) {
                        size_t sl = (abs_base + t) % swa_cap;
                        std::memcpy(dst_kh + sl * D, src_kh + t * D, D * sizeof(float));
                        std::memcpy(dst_vh + sl * D, src_vh + t * D, D * sizeof(float));
                    }
                }
            }
        }
    }

    void gather_glob_f32_block(uint32_t block_idx, const EngineStateView &view,
                               uint8_t *sec4, const KvbSectionLayout &L) const noexcept {
        const size_t B     = kBlockTokens;
        const size_t G     = m_geom.num_glob_layers;
        const size_t H     = m_geom.kv_heads;
        const size_t D     = m_geom.head_dim;
        const size_t s_cap = m_geom.max_ctx;
        const size_t s0    = static_cast<size_t>(block_idx) * B;

        float *dst_k = reinterpret_cast<float *>(sec4);
        float *dst_v = reinterpret_cast<float *>(sec4 + L.glob_f32_half_bytes);

        for (size_t g = 0; g < G; ++g) {
            for (size_t kh = 0; kh < H; ++kh) {
                const size_t gh_idx = g * H + kh;
                std::memcpy(dst_k + gh_idx * B * D,
                            view.glob_k_f32 + (gh_idx * s_cap + s0) * D,
                            B * D * sizeof(float));
                std::memcpy(dst_v + gh_idx * B * D,
                            view.glob_v_f32 + (gh_idx * s_cap + s0) * D,
                            B * D * sizeof(float));
            }
        }
    }

    void scatter_glob_f32_block(uint32_t block_idx, const uint8_t *sec4,
                                const KvbSectionLayout &L, EngineStateView &view) const noexcept {
        const size_t B     = kBlockTokens;
        const size_t G     = m_geom.num_glob_layers;
        const size_t H     = m_geom.kv_heads;
        const size_t D     = m_geom.head_dim;
        const size_t s_cap = m_geom.max_ctx;
        const size_t s0    = static_cast<size_t>(block_idx) * B;

        const float *src_k = reinterpret_cast<const float *>(sec4);
        const float *src_v = reinterpret_cast<const float *>(sec4 + L.glob_f32_half_bytes);

        for (size_t g = 0; g < G; ++g) {
            for (size_t kh = 0; kh < H; ++kh) {
                const size_t gh_idx = g * H + kh;
                std::memcpy(view.glob_k_f32 + (gh_idx * s_cap + s0) * D,
                            src_k + gh_idx * B * D,
                            B * D * sizeof(float));
                std::memcpy(view.glob_v_f32 + (gh_idx * s_cap + s0) * D,
                            src_v + gh_idx * B * D,
                            B * D * sizeof(float));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Eviction, Subtree Quarantine & WAL Persistence
    // -------------------------------------------------------------------------

    void evict_single_leaf_locked(const Hash128 &k, bool count_as_eviction) {
        auto it = m_nodes.find(k);
        if (it == m_nodes.end()) return;
        const CacheNode victim = it->second;

        if (victim.block_idx > 0) {
            auto pit = m_nodes.find(victim.parent_key);
            if (pit != m_nodes.end() && pit->second.num_children > 0) {
                pit->second.num_children--;
            }
        }
        if (m_disk_bytes_used >= victim.file_bytes) m_disk_bytes_used -= victim.file_bytes;
        else                                        m_disk_bytes_used = 0;

        m_inflight_blobs.erase(k);
        m_nodes.erase(it);

        std::error_code ec;
        std::filesystem::remove(kvb_path_for(k), ec);
        append_wal_del_locked(k);
        if (count_as_eviction) m_total_evictions++;
    }

    size_t evict_to_target_locked(uint64_t target_bytes, const Hash128 &protected_leaf = Hash128{}) {
        size_t evicted = 0;
        while (m_disk_bytes_used > target_bytes) {
            auto best_it = m_nodes.end();
            for (auto it = m_nodes.begin(); it != m_nodes.end(); ++it) {
                const CacheNode &n = it->second;
                if (n.num_children > 0) continue;
                if (n.flags & KVB_FLAG_PINNED) continue;
                if (!protected_leaf.is_zero() && it->first == protected_leaf) continue;
                if (best_it == m_nodes.end() ||
                    n.gdsf_key < best_it->second.gdsf_key ||
                    (n.gdsf_key == best_it->second.gdsf_key &&
                     n.last_access_tick < best_it->second.last_access_tick)) {
                    best_it = it;
                }
            }
            if (best_it == m_nodes.end()) break;
            m_clock = best_it->second.gdsf_key;
            evict_single_leaf_locked(best_it->first, true);
            evicted++;
        }
        return evicted;
    }

    size_t evict_to_quota_locked() {
        return evict_to_target_locked(m_max_disk_bytes);
    }

    void remove_subtree_locked(const Hash128 &root_bad) {
        if (m_nodes.find(root_bad) == m_nodes.end()) return;
        std::unordered_map<Hash128, std::vector<Hash128>, Hash128Hasher> children_map;
        for (const auto &kv : m_nodes) {
            if (kv.second.block_idx > 0) {
                children_map[kv.second.parent_key].push_back(kv.first);
            }
        }
        // Post-order traversal so leaves are removed before their parents
        std::vector<Hash128> stack{root_bad};
        std::vector<Hash128> post_order;
        while (!stack.empty()) {
            Hash128 cur = stack.back();
            stack.pop_back();
            post_order.push_back(cur);
            auto cit = children_map.find(cur);
            if (cit != children_map.end()) {
                for (const Hash128 &ch : cit->second) stack.push_back(ch);
            }
        }
        for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
            evict_single_leaf_locked(*it, false);
        }
    }

    void cleanup_orphan_temps_locked() {
        std::error_code ec;
        for (auto const &dir_entry : std::filesystem::recursive_directory_iterator(m_ns_dir, ec)) {
            if (ec) break;
            if (!dir_entry.is_regular_file(ec)) continue;
            std::string fn = dir_entry.path().filename().string();
            if (fn.find(".kvb.tmp.") != std::string::npos) {
                std::filesystem::remove(dir_entry.path(), ec);
            }
        }
    }

    void append_wal_put_locked(const CacheNode &n) {
        std::ofstream ofs(m_wal_path, std::ios::app);
        if (!ofs) return;
        ofs << "PUT " << n.block_key.to_hex() << ' ' << n.parent_key.to_hex() << ' '
            << n.block_idx << ' ' << n.file_bytes << ' ' << n.flags << ' '
            << n.hit_count << ' ' << n.created_unix_ms << '\n';
    }

    void append_wal_del_locked(const Hash128 &k) {
        std::ofstream ofs(m_wal_path, std::ios::app);
        if (!ofs) return;
        ofs << "DEL " << k.to_hex() << '\n';
    }

    void append_wal_pin_locked(const Hash128 &k, bool pinned) {
        std::ofstream ofs(m_wal_path, std::ios::app);
        if (!ofs) return;
        ofs << "PIN " << k.to_hex() << ' ' << (pinned ? 1 : 0) << '\n';
    }

    void write_snapshot_locked() {
        std::filesystem::path tmp = m_ns_dir / "index.snap.tmp";
        {
            std::ofstream ofs(tmp, std::ios::trunc);
            if (!ofs) return;
            ofs << "LFMKVB_SNAP_V1 " << m_geom.model_fingerprint.to_hex() << ' ' << m_clock << '\n';
            for (const auto &kv : m_nodes) {
                const CacheNode &n = kv.second;
                if (!n.on_disk) continue;
                ofs << "PUT " << n.block_key.to_hex() << ' ' << n.parent_key.to_hex() << ' '
                    << n.block_idx << ' ' << n.file_bytes << ' ' << n.flags << ' '
                    << n.hit_count << ' ' << n.created_unix_ms << '\n';
            }
            ofs.flush();
        }
        std::error_code ec;
        std::filesystem::rename(tmp, m_snap_path, ec);
        if (!ec) {
            std::ofstream wal_trunc(m_wal_path, std::ios::trunc);
        }
    }

    void replay_stream_locked(std::istream &is, bool check_fp_header) {
        std::string line;
        if (check_fp_header && std::getline(is, line)) {
            std::istringstream hss(line);
            std::string tag, fp_hex;
            double saved_clock = 0.0;
            hss >> tag >> fp_hex >> saved_clock;
            Hash128 saved_fp{};
            if (tag != "LFMKVB_SNAP_V1" || !Hash128::from_hex(fp_hex, saved_fp) ||
                saved_fp != m_geom.model_fingerprint) {
                return;
            }
            m_clock = std::max(m_clock, saved_clock);
        }

        while (std::getline(is, line)) {
            if (line.empty()) continue;
            std::istringstream ls(line);
            std::string op;
            ls >> op;
            if (op == "PUT") {
                std::string k_hex, p_hex;
                CacheNode n{};
                if (!(ls >> k_hex >> p_hex >> n.block_idx >> n.file_bytes >> n.flags >>
                      n.hit_count >> n.created_unix_ms)) {
                    continue;
                }
                if (!Hash128::from_hex(k_hex, n.block_key) ||
                    !Hash128::from_hex(p_hex, n.parent_key)) {
                    continue;
                }
                n.on_disk          = true;
                n.last_access_tick = ++m_tick_seq;
                update_gdsf_key(n);
                m_nodes[n.block_key] = n;
            } else if (op == "DEL") {
                std::string k_hex;
                Hash128 k{};
                if ((ls >> k_hex) && Hash128::from_hex(k_hex, k)) {
                    m_nodes.erase(k);
                }
            } else if (op == "PIN") {
                std::string k_hex;
                int p = 0;
                Hash128 k{};
                if ((ls >> k_hex >> p) && Hash128::from_hex(k_hex, k)) {
                    auto it = m_nodes.find(k);
                    if (it != m_nodes.end()) {
                        if (p) it->second.flags |= KVB_FLAG_PINNED;
                        else   it->second.flags &= ~KVB_FLAG_PINNED;
                    }
                }
            }
        }
    }

    void load_index_locked() {
        {
            std::ifstream snap_ifs(m_snap_path);
            if (snap_ifs) replay_stream_locked(snap_ifs, true);
        }
        {
            std::ifstream wal_ifs(m_wal_path);
            if (wal_ifs) replay_stream_locked(wal_ifs, false);
        }

        // Prune any entries whose .kvb file is missing or whose parent chain is broken, then
        // recompute num_children and m_disk_bytes_used deterministically.
        std::vector<Hash128> missing;
        for (auto &kv : m_nodes) {
            kv.second.num_children = 0;
            std::error_code ec;
            auto p = kvb_path_for(kv.first);
            if (!std::filesystem::exists(p, ec)) {
                missing.push_back(kv.first);
            } else {
                // Verify header fingerprint matches current model_fingerprint
                KvbHeader hdr{};
                int fd = ::open(p.c_str(), O_RDONLY);
                bool ok = (fd >= 0) && detail::pread_exact(fd, &hdr, sizeof(KvbHeader), 0);
                if (fd >= 0) ::close(fd);
                if (!ok || std::memcmp(hdr.magic, kKvbMagic, 8) != 0 ||
                    hdr.model_fingerprint != m_geom.model_fingerprint) {
                    missing.push_back(kv.first);
                }
            }
        }
        for (const Hash128 &k : missing) m_nodes.erase(k);

        // Iteratively prune orphans whose parent_key is absent
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = m_nodes.begin(); it != m_nodes.end(); ) {
                if (it->second.block_idx > 0 && m_nodes.find(it->second.parent_key) == m_nodes.end()) {
                    std::error_code ec;
                    std::filesystem::remove(kvb_path_for(it->first), ec);
                    it = m_nodes.erase(it);
                    changed = true;
                } else {
                    ++it;
                }
            }
        }

        m_disk_bytes_used = 0;
        for (auto &kv : m_nodes) {
            m_disk_bytes_used += kv.second.file_bytes;
            if (kv.second.block_idx > 0) {
                m_nodes[kv.second.parent_key].num_children++;
            }
        }
        evict_to_quota_locked();
    }

    // -------------------------------------------------------------------------
    // Asynchronous Background NVMe Writer Loop (Atomic .tmp + fdatasync + rename)
    // -------------------------------------------------------------------------

    void writer_loop() {
        uint64_t thread_seq = 0;
        while (true) {
            WriteJob job{};
            {
                std::unique_lock<std::mutex> wlk(m_writer_mtx);
                m_writer_cv.wait(wlk, [&]() {
                    return m_writer_stop || !m_write_queue.empty();
                });
                if (m_write_queue.empty() && m_writer_stop) {
                    break;
                }
                job = std::move(m_write_queue.front());
                m_write_queue.pop_front();
                m_writer_active = true;
            }

            // Skip disk write if block was already evicted while queued
            bool still_alive = false;
            std::filesystem::path final_path;
            {
                std::lock_guard<std::mutex> lk(m_mtx);
                still_alive = (m_nodes.find(job.block_key) != m_nodes.end());
                final_path  = kvb_path_for(job.block_key);
            }

            bool wrote_ok = false;
            if (still_alive && job.blob) {
                std::filesystem::path tmp_path =
                    final_path.string() + ".tmp." + std::to_string(++thread_seq);
                int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd >= 0) {
                    wrote_ok = detail::write_exact(fd, job.blob->data(), job.blob->size());
                    if (wrote_ok) {
#if defined(__linux__)
                        wrote_ok = (::fdatasync(fd) == 0);
#elif defined(__unix__) || defined(__APPLE__)
                        wrote_ok = (::fsync(fd) == 0);
#endif
                    }
                    ::close(fd);
                    if (wrote_ok) {
                        std::error_code ec;
                        std::filesystem::rename(tmp_path, final_path, ec);
                        if (ec) {
                            wrote_ok = false;
                            std::filesystem::remove(tmp_path, ec);
                        }
                    } else {
                        std::error_code ec;
                        std::filesystem::remove(tmp_path, ec);
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lk(m_mtx);
                auto it = m_nodes.find(job.block_key);
                if (it != m_nodes.end()) {
                    if (wrote_ok) {
                        it->second.on_disk = true;
                        m_inflight_blobs.erase(job.block_key);
                        append_wal_put_locked(it->second);
                    } else {
                        remove_subtree_locked(job.block_key);
                    }
                } else {
                    m_inflight_blobs.erase(job.block_key);
                    if (wrote_ok) {
                        std::error_code ec;
                        std::filesystem::remove(final_path, ec);
                    }
                }
            }

            {
                std::lock_guard<std::mutex> wlk(m_writer_mtx);
                m_writer_active = false;
                if (m_write_queue.empty()) {
                    m_flushed_cv.notify_all();
                }
            }
        }
    }

    ModelCacheGeometry m_geom{};
    uint64_t           m_max_disk_bytes = 0;
    std::filesystem::path m_ns_dir;
    std::filesystem::path m_snap_path;
    std::filesystem::path m_wal_path;
    bool               m_initialized = false;

    mutable std::mutex m_mtx;
    std::unordered_map<Hash128, CacheNode, Hash128Hasher> m_nodes;
    std::unordered_map<Hash128, std::shared_ptr<const std::vector<uint8_t>>, Hash128Hasher> m_inflight_blobs;

    uint64_t m_disk_bytes_used    = 0;
    double   m_clock              = 0.0;
    uint64_t m_tick_seq           = 0;
    uint64_t m_total_lookups      = 0;
    uint64_t m_total_block_hits   = 0;
    uint64_t m_total_tokens_saved = 0;
    uint64_t m_total_evictions    = 0;

    mutable std::mutex      m_writer_mtx;
    std::condition_variable m_writer_cv;
    std::condition_variable m_flushed_cv;
    std::deque<WriteJob>    m_write_queue;
    bool                    m_writer_active = false;
    bool                    m_writer_stop   = false;
    std::thread             m_writer_thread;
};

} // namespace eswin::prefix_cache
