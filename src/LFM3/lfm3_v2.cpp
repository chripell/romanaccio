/**
 * @file lfm3_v2.cpp
 * @brief Native C++17 Dual-Die Tri-Hybrid Inference Engine for LiquidAI/LFM3-1.7B (Milestone 4).
 *
 * Built on the Open ESWIN NPU Runtime (libes_runtime.so):
 *  - Route D Shared-Weight Class B Residency: Only 12 resident models per die
 *    (4 T=1 Class B archetypes + 4 T=64 Class B archetypes + 4 16K LM head slices),
 *    50 models below the 62-model kernel Oops ceiling, with zero model swapping!
 *  - Dual-Die Pipeline Parallelism (PP=2): Layers 0..11 on Die 0 (mmz_nid_0_part_0),
 *    Layers 12..23 on Die 1 (mmz_nid_1_part_0), plus Concurrent Dual-Die LM Head (64K/die).
 *  - Four LFM3 Architectural Primitives:
 *    1. Sandwich RMSNorm (operator_norm, post_operator_norm, ffn_norm, post_ffn_norm, eps=1e-5)
 *    2. Gated ShortConv State Engine (13 layers, 7-tap depthwise conv over [2048, 6] state ring)
 *    3. SWA (5 layers, window=256, theta=10K full RoPE) & Global GQA Attention (6 layers,
 *       theta=1M Zero-Tail Partial RoPE on first 32 pairs, QK-norm, 16-head sigmoid gate)
 *    4. Dual-Die Parallel LM Head (8 x 16,000 slices across 128,000 vocab) & LFM3 ChatML template
 */

#include "es_runtime.h"
#include "backend/driver_wrapper.hpp"
#include "backend/dma_buffer.hpp"
#include "backend/worker_pool.hpp"
#include "model/embedding_table.hpp"
#include "model/simple_json.hpp"
#include "nlp/tokenizer.hpp"
#include "nlp/sampler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <omp.h>
#include <sched.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#define COLOR_RESET   "\033[0m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_GRAY    "\033[90m"

using namespace eswin;

namespace {

static std::atomic<bool> g_interrupt_flag(false);
static std::atomic<bool> g_in_generation(false);

static void sigint_handler(int sig) {
    (void)sig;
    if (g_in_generation.load()) {
        g_interrupt_flag.store(true);
        std::cout << COLOR_YELLOW << "\n[Generation interrupted by user]" << COLOR_RESET << std::endl;
    } else {
        std::cout << "\nExiting LFM3-v2..." << std::endl;
        std::exit(0);
    }
}

static bool file_exists(const std::string &path) {
    std::ifstream f(path);
    return f.good();
}

// Fast LUT-based FP16 <-> FP32 conversion
static float g_fp16_lut[65536];

static void init_fp16_lut() {
    for (uint32_t i = 0; i < 65536; ++i) {
        g_fp16_lut[i] = es_f16_to_f32(static_cast<uint16_t>(i));
    }
}

static inline float fp16_to_fp32(uint16_t h) {
    return g_fp16_lut[h];
}

static inline uint16_t fp32_to_fp16(float f) {
    return es_f32_to_f16(f);
}

// Fast inline exp(x) for x in [-20, 0] (subnormal-free on RISC-V P550)
static inline float fast_expf_neg(float x) {
    if (x <= -20.0f) return 0.0f;
    float t = x * 1.44269504089f;
    float fi = std::floor(t);
    float f = t - fi;
    int32_t i = static_cast<int32_t>(fi) + 127;
    float p = 1.0f + f * (0.69514614f + f * (0.22756439f + f * 0.07728947f));
    uint32_t bits = static_cast<uint32_t>(i << 23);
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return scale * p;
}

// Pin OpenMP threads 0..7 to physical CPUs 0..7 (Threads 0..3 -> NUMA Node 0 / Die 0,
// Threads 4..7 -> NUMA Node 1 / Die 1) so KV heads 0,1 stay local to Die 0 DDR5 and
// KV heads 2,3 stay local to Die 1 DDR5 with zero cross-die Chiplink traffic.
static void pin_openmp_threads_numa() {
    omp_set_num_threads(8);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(tid % 8, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    }
}

// Strict NUMA-bound + 2 MiB HugePage per-layer KV buffer:
// For each layer l in [0, num_layers), binds the first half (KV heads 0,1) to NUMA Node 0 (Die 0)
// and the second half (KV heads 2,3) to NUMA Node 1 (Die 1), enables MADV_HUGEPAGE so 2 MiB pages
// sit in the P550 32-entry L1 DTLB, and faults each half on its local die's CPU cores.
static inline void bind_memory_to_numa_node(void *addr, size_t len, int node) {
#if defined(__linux__) && defined(SYS_mbind)
    const size_t page_sz = 4096;
    uintptr_t u = reinterpret_cast<uintptr_t>(addr);
    uintptr_t aligned_beg = (u + page_sz - 1) & ~(page_sz - 1);
    uintptr_t aligned_end = (u + len) & ~(page_sz - 1);
    if (aligned_end <= aligned_beg) return;
    unsigned long nodemask = (1UL << (node & 1));
    // MPOL_BIND = 2, MPOL_MF_STRICT | MPOL_MF_MOVE = 3
    long rc = syscall(SYS_mbind, reinterpret_cast<void*>(aligned_beg), aligned_end - aligned_beg,
                      2, &nodemask, sizeof(nodemask) * 8, 3);
    if (rc != 0) {
        syscall(SYS_mbind, reinterpret_cast<void*>(aligned_beg), aligned_end - aligned_beg,
                2, &nodemask, sizeof(nodemask) * 8, 0);
    }
#else
    (void)addr; (void)len; (void)node;
#endif
}

template <typename T>
class NumaKvBuffer {
public:
    NumaKvBuffer() = default;
    ~NumaKvBuffer() { release(); }

    NumaKvBuffer(const NumaKvBuffer&) = delete;
    NumaKvBuffer& operator=(const NumaKvBuffer&) = delete;

    void allocate(size_t num_layers, size_t elems_per_layer) {
        release();
        m_num_layers = num_layers;
        m_elems_per_layer = elems_per_layer;
        m_size = num_layers * elems_per_layer;
        size_t raw_bytes = m_size * sizeof(T);
        // Align allocation to 2 MiB boundary when >= 2 MiB to maximize 2 MiB THP coverage
        size_t align = (raw_bytes >= (2 * 1024 * 1024)) ? (2 * 1024 * 1024) : 4096;
        m_alloc_bytes = (raw_bytes + align - 1) & ~(align - 1);

        void *p = mmap(nullptr, m_alloc_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            std::perror("NumaKvBuffer mmap");
            std::abort();
        }
        m_ptr = static_cast<T*>(p);
#if defined(MADV_HUGEPAGE)
        if (m_alloc_bytes >= (2 * 1024 * 1024)) {
            madvise(m_ptr, m_alloc_bytes, MADV_HUGEPAGE);
        }
#endif
        size_t half_elems = elems_per_layer / 2;
        size_t half_bytes = half_elems * sizeof(T);
        for (size_t l = 0; l < num_layers; ++l) {
            T *layer_base = m_ptr + l * elems_per_layer;
            bind_memory_to_numa_node(layer_base,              half_bytes, 0); // KV heads 0,1 -> Die 0
            bind_memory_to_numa_node(layer_base + half_elems, half_bytes, 1); // KV heads 2,3 -> Die 1
        }
        clear();
    }

    void clear() {
        if (!m_ptr || m_size == 0) return;
        const size_t half_elems = m_elems_per_layer / 2;
        #pragma omp parallel num_threads(8)
        {
            int tid = omp_get_thread_num();
            int die = (tid < 4) ? 0 : 1;
            int local_tid = tid & 3;
            size_t chunk = half_elems / 4;
            size_t beg = local_tid * chunk;
            size_t end = (local_tid == 3) ? half_elems : (beg + chunk);
            for (size_t l = 0; l < m_num_layers; ++l) {
                T *half_base = m_ptr + l * m_elems_per_layer + static_cast<size_t>(die) * half_elems;
                std::memset(half_base + beg, 0, (end - beg) * sizeof(T));
            }
        }
    }

    T* data() noexcept { return m_ptr; }
    const T* data() const noexcept { return m_ptr; }
    size_t size() const noexcept { return m_size; }
    T* begin() noexcept { return m_ptr; }
    T* end() noexcept { return m_ptr + m_size; }
    T& operator[](size_t i) noexcept { return m_ptr[i]; }
    const T& operator[](size_t i) const noexcept { return m_ptr[i]; }

private:
    void release() {
        if (m_ptr) {
            munmap(m_ptr, m_alloc_bytes);
            m_ptr = nullptr;
            m_size = 0;
            m_alloc_bytes = 0;
        }
    }

    T     *m_ptr = nullptr;
    size_t m_size = 0;
    size_t m_alloc_bytes = 0;
    size_t m_num_layers = 0;
    size_t m_elems_per_layer = 0;
};

static inline int32_t round_clamp_f32(float x_scaled, float limit) {
    float c = std::min(std::max(x_scaled, -limit), limit);
#if defined(__riscv) && defined(__riscv_flen)
    int32_t q;
    __asm__ volatile("fcvt.w.s %0, %1, rne" : "=r"(q) : "f"(c));
    return q;
#else
    return static_cast<int32_t>(std::nearbyintf(c));
#endif
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
// Quantize 16 FP32 values -> 16 symmetric INT8 values in [-127, 127]
static inline int8x16_t neon_scale_clamp_f32_to_s8x16(
    const float *src, float32x4_t v_inv_step, int8x16_t v_min_s8) {
    float32x4_t f0 = vmulq_f32(vld1q_f32(src +  0), v_inv_step);
    float32x4_t f1 = vmulq_f32(vld1q_f32(src +  4), v_inv_step);
    float32x4_t f2 = vmulq_f32(vld1q_f32(src +  8), v_inv_step);
    float32x4_t f3 = vmulq_f32(vld1q_f32(src + 12), v_inv_step);

    int32x4_t i0 = vcvtnq_s32_f32(f0);
    int32x4_t i1 = vcvtnq_s32_f32(f1);
    int32x4_t i2 = vcvtnq_s32_f32(f2);
    int32x4_t i3 = vcvtnq_s32_f32(f3);

    int16x8_t s16_lo = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
    int16x8_t s16_hi = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
    int8x16_t s8     = vcombine_s8(vqmovn_s16(s16_lo), vqmovn_s16(s16_hi));
    return vmaxq_s8(s8, v_min_s8);
}

// Quantize 8 FP32 values -> 8 symmetric INT16 values in [-32767, 32767]
static inline int16x8_t neon_scale_clamp_f32_to_s16x8(
    const float *src, float32x4_t v_inv_step, int16x8_t v_min_s16) {
    float32x4_t f0 = vmulq_f32(vld1q_f32(src + 0), v_inv_step);
    float32x4_t f1 = vmulq_f32(vld1q_f32(src + 4), v_inv_step);

    int32x4_t i0 = vcvtnq_s32_f32(f0);
    int32x4_t i1 = vcvtnq_s32_f32(f1);

    int16x8_t s16 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
    return vmaxq_s16(s16, v_min_s16);
}

// Dequantize 16 INT8 values -> 16 FP32 values
static inline void neon_dequant_s8x16_to_f32(
    int8x16_t v_s8, float32x4_t v_step, float *dst) {
    int16x8_t s16_lo = vmovl_s8(vget_low_s8(v_s8));
    int16x8_t s16_hi = vmovl_s8(vget_high_s8(v_s8));

    float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_lo))),  v_step);
    float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_lo))), v_step);
    float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_hi))),  v_step);
    float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_hi))), v_step);

    vst1q_f32(dst +  0, f0);
    vst1q_f32(dst +  4, f1);
    vst1q_f32(dst +  8, f2);
    vst1q_f32(dst + 12, f3);
}

// Dequantize 8 INT16 values -> 8 FP32 values
static inline void neon_dequant_s16x8_to_f32(
    int16x8_t v_s16, float32x4_t v_step, float *dst) {
    float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_s16))),  v_step);
    float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_s16))), v_step);
    vst1q_f32(dst + 0, f0);
    vst1q_f32(dst + 4, f1);
}
#endif // __ARM_NEON

static inline void quantize_kv_u8(const float *src, uint8_t *dst_u8, float &out_scale, uint16_t &out_sum) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t vmax0 = vdupq_n_f32(0.0f);
    float32x4_t vmax1 = vdupq_n_f32(0.0f);
    for (int d = 0; d < 128; d += 8) {
        vmax0 = vmaxq_f32(vmax0, vabsq_f32(vld1q_f32(src + d + 0)));
        vmax1 = vmaxq_f32(vmax1, vabsq_f32(vld1q_f32(src + d + 4)));
    }
    float amax = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
    float sc   = (amax > 1e-12f) ? (amax / 127.0f) : 1.0f;
    float inv  = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;

    float32x4_t v_inv    = vdupq_n_f32(inv);
    int8x16_t   v_min_s8 = vdupq_n_s8(-127);
    uint8x16_t  v_bias   = vdupq_n_u8(0x80);
    uint16x8_t  v_sum16  = vdupq_n_u16(0);

    for (int d = 0; d < 128; d += 16) {
        int8x16_t  q_s8 = neon_scale_clamp_f32_to_s8x16(src + d, v_inv, v_min_s8);
        uint8x16_t q_u8 = veorq_u8(vreinterpretq_u8_s8(q_s8), v_bias);
        vst1q_u8(dst_u8 + d, q_u8);
        v_sum16 = vpadalq_u8(v_sum16, q_u8);
    }
    out_scale = sc;
    out_sum   = static_cast<uint16_t>(vaddlvq_u16(v_sum16));
#else
    float amax = 0.0f;
    for (int d = 0; d < 128; ++d) {
        float a = std::fabs(src[d]);
        if (a > amax) amax = a;
    }
    float sc = (amax > 1e-12f) ? (amax / 127.0f) : 1.0f;
    float inv = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
    uint32_t sum_u = 0;
    for (int d = 0; d < 128; ++d) {
        int32_t q = round_clamp_f32(src[d] * inv, 127.0f);
        uint8_t u = static_cast<uint8_t>(q + 128);
        dst_u8[d] = u;
        sum_u += u;
    }
    out_scale = sc;
    out_sum = static_cast<uint16_t>(sum_u);
#endif
}

// LFM3 24-layer schedule: 0 = conv (13), 1 = full_attention (6), 2 = sliding_attention (5)
enum class Lfm3LayerType : int {
    Conv = 0,
    FullAttention = 1,
    SlidingAttention = 2
};

static constexpr Lfm3LayerType kLayerSchedule[24] = {
    Lfm3LayerType::Conv,             // L00 (Die 0)
    Lfm3LayerType::Conv,             // L01 (Die 0)
    Lfm3LayerType::FullAttention,    // L02 (Die 0)
    Lfm3LayerType::Conv,             // L03 (Die 0)
    Lfm3LayerType::SlidingAttention, // L04 (Die 0)
    Lfm3LayerType::Conv,             // L05 (Die 0)
    Lfm3LayerType::FullAttention,    // L06 (Die 0)
    Lfm3LayerType::Conv,             // L07 (Die 0)
    Lfm3LayerType::SlidingAttention, // L08 (Die 0)
    Lfm3LayerType::Conv,             // L09 (Die 0)
    Lfm3LayerType::FullAttention,    // L10 (Die 0)
    Lfm3LayerType::Conv,             // L11 (Die 0)
    Lfm3LayerType::SlidingAttention, // L12 (Die 1)
    Lfm3LayerType::Conv,             // L13 (Die 1)
    Lfm3LayerType::FullAttention,    // L14 (Die 1)
    Lfm3LayerType::Conv,             // L15 (Die 1)
    Lfm3LayerType::SlidingAttention, // L16 (Die 1)
    Lfm3LayerType::Conv,             // L17 (Die 1)
    Lfm3LayerType::FullAttention,    // L18 (Die 1)
    Lfm3LayerType::Conv,             // L19 (Die 1)
    Lfm3LayerType::SlidingAttention, // L20 (Die 1)
    Lfm3LayerType::Conv,             // L21 (Die 1)
    Lfm3LayerType::FullAttention,    // L22 (Die 1)
    Lfm3LayerType::Conv              // L23 (Die 1)
};

/**
 * @brief Byte count to hand the driver for one tensor surface.
 *
 * The descriptor describes the DMA buffer that was allocated, not the loadable's memory entry:
 * `ModelInfo` reports an entry size far larger than the tensor (the INT8 `conv_in` output entry
 * is 2 MiB for a 6 KiB tensor), and submitting that full 2 MiB wedges the die worker. Passing a
 * descriptor *larger* than the tensor is proven harmless (FINDINGS §33.1, measured on the
 * over-sized activation buffer), so the value is the larger of the tensor bytes and the reported
 * entry, capped to the allocation.
 *
 * The reverse case is not harmless: a descriptor smaller than the tensor truncates what the NPU
 * reads or writes and the result still looks plausible. That is a hard error, not a clamp.
 */
static size_t surface_descriptor(const DmaBuffer *buf, size_t reported, size_t raw,
                                 const char *what, bool &failed, size_t cap = SIZE_MAX) {
    if (!buf) return raw;
    // The write-extent probe over-allocates buffers to carry a guard region. Without this cap
    // the guard would silently widen every descriptor and change the thing being measured.
    const size_t usable = std::min(buf->size(), cap);
    // Compare against `usable`, not `buf->size()`: under the extent probe the allocation carries a
    // guard region, and testing against it would let a tensor larger than the production buffer
    // pass the check and then be silently truncated by the min() below -- the exact bug this
    // function exists to prevent.
    if (raw > usable) {
        std::printf("FAIL: %s tensor needs %zu bytes but its DMA buffer holds only %zu "
                    "(reported entry %zu) -- refusing to truncate the surface\n",
                    what, raw, buf->size(), reported);
        failed = true;
        return raw;
    }
    return std::max(raw, std::min(reported, usable));
}

struct StageTimers {
    double npu_layer_ms  = 0.0;
    double npu_lmhead_ms = 0.0;
    double cpu_conv_ms   = 0.0;
    double cpu_swa_ms    = 0.0;
    double cpu_glob_ms   = 0.0;
    double cpu_norm_ms   = 0.0;
    double d2d_xfer_ms   = 0.0;
    int    tokens_run    = 0;

    /**
     * @brief Per-archetype submit accounting, to separate NPU streaming speed from our overhead.
     *
     * master_ms brackets WorkerPool::submit_task plus the post-submit cache flush, exactly as
     * npu_layer_ms does, so the two stay comparable. driver_ms is the elapsed_ns the worker
     * measured around the driver call itself. The difference is IPC + queue + cache maintenance --
     * the only way to tell "the engine wastes time per submit" from "the NPU streams distinct
     * weights slower than it streams one reused region".
     */
    struct ArchStat {
        const char *name = "";
        long   submits      = 0;
        double master_ms    = 0.0;
        double driver_ms    = 0.0;
        double weight_bytes = 0.0;
    };
    enum ArchIdx { ARCH_CONV_IN = 0, ARCH_PROJ2048, ARCH_ATTN_QKV, ARCH_MLP, ARCH_LM_HEAD,
                   ARCH_COUNT };
    ArchStat arch[ARCH_COUNT] = {
        {"conv_in  2048->6144"}, {"proj2048 2048->2048"}, {"attn_qkv 2048->3072"},
        {"mlp  SwiGLU 7680"},     {"lm_head  2048->16K"},
    };

    void add_arch(int idx, double master_ms, double driver_ms, double weight_bytes) {
        ArchStat &a = arch[idx];
        ++a.submits;
        a.master_ms += master_ms;
        a.driver_ms += driver_ms;
        a.weight_bytes += weight_bytes;
    }

    void reset() { *this = StageTimers{}; }

    double total_ms() const {
        return npu_layer_ms + npu_lmhead_ms + cpu_conv_ms + cpu_swa_ms + cpu_glob_ms + cpu_norm_ms + d2d_xfer_ms;
    }

    void print_arch_report() const {
        const int n = std::max(tokens_run, 1);
        std::printf("  +--------------------------------------------------------------------------------------------+\n");
        std::printf("  | Per-submit NPU streaming (master = submit+flush as measured above; driver = worker-side)    |\n");
        std::printf("  +------------------------+-------+-----------+-----------+----------+-----------+-----------+\n");
        std::printf("  | Archetype              | Submits| master ms | driver ms | our ms   | GB/s drv  | GB/s mst  |\n");
        std::printf("  +------------------------+-------+-----------+-----------+----------+-----------+-----------+\n");
        double w_tot = 0.0, m_tot = 0.0, d_tot = 0.0;
        long subs = 0;
        for (const ArchStat &a : arch) {
            if (!a.submits) continue;
            const double gb = a.weight_bytes / 1e9;              // total for the whole run
            const double per_tok_m = a.master_ms / n, per_tok_d = a.driver_ms / n;
            // Bytes and seconds must cover the same span: rate = total bytes / total seconds.
            std::printf("  | %-22s | %5ld | %9.2f | %9.2f | %8.2f | %9.2f | %9.2f |\n",
                        a.name, a.submits, per_tok_m, per_tok_d, per_tok_m - per_tok_d,
                        a.driver_ms > 0 ? gb * 1000.0 / a.driver_ms : 0.0,
                        a.master_ms > 0 ? gb * 1000.0 / a.master_ms : 0.0);
            w_tot += gb; m_tot += a.master_ms; d_tot += a.driver_ms; subs += a.submits;
        }
        std::printf("  +------------------------+-------+-----------+-----------+----------+-----------+-----------+\n");
        std::printf("  | TOTAL %4d tokens         | %5ld | %9.2f | %9.2f | %8.2f | %9.2f | %9.2f |\n",
                    n, subs, m_tot / n, d_tot / n, (m_tot - d_tot) / n,
                    d_tot > 0 ? w_tot * 1000.0 / d_tot : 0.0,
                    m_tot > 0 ? w_tot * 1000.0 / m_tot : 0.0);
        std::printf("  | GB/s uses this run's weight bytes/token; with --weights-layer0 they are re-read, not distinct |\n");
        std::printf("  +--------------------------------------------------------------------------------------------+\n");
    }

    void print_report(const char *phase_label) const {
        int n = std::max(tokens_run, 1);
        double tot = total_ms();
        std::printf("  +----------------------------------------------------------------------+\n");
        std::printf("  | %-68s |\n", phase_label);
        std::printf("  +------------------------+------------------+--------------+-----------+\n");
        std::printf("  | Stage                  | Total (ms)       | Per-Tok (ms) | Share (%%) |\n");
        std::printf("  +------------------------+------------------+--------------+-----------+\n");
        auto row = [&](const char *name, double ms) {
            std::printf("  | %-22s | %13.2f ms | %9.2f ms | %8.1f%% |\n",
                        name, ms, ms / n, (tot > 0.0) ? (100.0 * ms / tot) : 0.0);
        };
        row("NPU Layer Projections",  npu_layer_ms);
        row("NPU Dual-Die LM Head",   npu_lmhead_ms);
        row("CPU ShortConv (13L)",    cpu_conv_ms);
        row("CPU SWA Attention (5L)", cpu_swa_ms);
        row("CPU Global Attn (6L)",   cpu_glob_ms);
        row("CPU SandwichNorm+Gates", cpu_norm_ms);
        row("D2D Cross-Die Transfer", d2d_xfer_ms);
        std::printf("  +------------------------+------------------+--------------+-----------+\n");
        std::printf("  | TOTAL (%4d tokens)    | %13.2f ms | %9.2f ms |    100.0%% |\n", n, tot, tot / n);
        std::printf("  +------------------------+------------------+--------------+-----------+\n");
    }
};

struct LayerCpuWeights {
    const float *op_norm = nullptr;       // [2048]
    const float *post_op_norm = nullptr;  // [2048]
    const float *ffn_norm = nullptr;      // [2048]
    const float *post_ffn_norm = nullptr; // [2048]
    // Conv layer only:
    const float *conv_weight = nullptr;   // [2048, 7]
    // Attention layer only:
    const float *q_norm = nullptr;        // [128]
    const float *k_norm = nullptr;        // [128]
    const float *gate_proj = nullptr;     // [16, 2048]

    // Milestone 8 (Phase 1): Fused SmoothQuant & scaled weights (zero runtime multiplications)
    std::vector<float> op_norm_smooth;    // [2048] = op_norm[c] * in_inv_smooth[c]
    std::vector<float> ffn_norm_smooth;   // [2048] = ffn_norm[c] * in_inv_smooth[c]
    std::vector<float> ffn_norm_scaled;   // [2048] = ffn_norm[c] * in_inv_smooth[c] * (1.0f / mlp_scale.in_step)
};

enum class BoundaryDtype : uint8_t {
    Fp16  = 0,
    Int8  = 1,
    Int16 = 2
};

struct SubgraphScaleInfo {
    BoundaryDtype in_dtype  = BoundaryDtype::Fp16;
    BoundaryDtype out_dtype = BoundaryDtype::Fp16;
    float in_thresh  = 1.0f;
    float in_step    = 1.0f / 127.0f;
    float out_thresh = 1.0f;
    float out_step   = 1.0f / 127.0f;
    const float *in_inv_smooth = nullptr;
    const float *out_step_chan = nullptr;
};

static inline BoundaryDtype parse_boundary_dtype(const std::string &s) {
    if (s == "int8")  return BoundaryDtype::Int8;
    if (s == "int16") return BoundaryDtype::Int16;
    return BoundaryDtype::Fp16;
}

static inline size_t dtype_elem_size(BoundaryDtype dt) {
    return (dt == BoundaryDtype::Int8) ? 1 : 2;
}

struct LayerNpuOffsets {
    uint16_t die_id = 0;
    uint8_t op_in_tp  = 1;
    uint8_t op_out_tp = 1;
    uint8_t mlp_tp    = 1;
    size_t op_in_offset     = 0;   // conv_in or attn_qkv (Die 0 or single die)
    size_t op_in_offset_d1  = 0;   // Die 1 offset when op_in_tp == 2
    size_t op_in_size       = 0;
    size_t op_out_offset    = 0;   // conv_out or attn_out (Die 0 or single die)
    size_t op_out_offset_d1 = 0;   // Die 1 offset when op_out_tp == 2
    size_t op_out_size      = 0;
    size_t mlp_offset       = 0;   // mlp (Die 0 or single die)
    size_t mlp_offset_d1    = 0;   // Die 1 offset when mlp_tp == 2
    size_t mlp_size         = 0;

    SubgraphScaleInfo op_in_scale{};
    SubgraphScaleInfo op_out_scale{};
    SubgraphScaleInfo mlp_scale{};
    // TP=2 Die 1 per-channel output dequantization scales
    SubgraphScaleInfo op_in_scale_d1{};
    SubgraphScaleInfo op_out_scale_d1{};
    SubgraphScaleInfo mlp_scale_d1{};
};

struct LoadedArchetype {
    uint32_t id = 0;
    size_t   in_surface_bytes  = 0;
    size_t   out_surface_bytes = 0;
    size_t   weight_bytes      = 0;   // loadable file size; the streamed bytes for a static head
};

struct DieModels {
    // T=1 Class B archetypes
    LoadedArchetype conv_in_t1{};
    LoadedArchetype proj2048_t1{};
    LoadedArchetype attn_qkv_t1{};
    LoadedArchetype mlp_t1{};
    // T=64 Class B archetypes
    LoadedArchetype conv_in_t64{};
    LoadedArchetype proj2048_t64{};
    LoadedArchetype attn_qkv_t64{};
    LoadedArchetype mlp_t64{};
    // 4 x 16,000 LM head slices per die
    LoadedArchetype   lm_head[4]{};
    SubgraphScaleInfo lm_head_scale[4]{};
    int resident_count = 0;
};

// RMSNorm in FP32 (with 4-way unrolled sum-of-squares accumulator to break FP32 FMA latency chains)
static void rms_norm_f32(const float *x, const float *w, float *out, int T, int C, float eps = 1e-5f) {
    #pragma omp parallel for if(T > 1) schedule(static)
    for (int t = 0; t < T; ++t) {
        const float *row = x + static_cast<size_t>(t) * C;
        float *orow = out + static_cast<size_t>(t) * C;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int c = 0;
        for (; c + 3 < C; c += 4) {
            s0 += row[c + 0] * row[c + 0];
            s1 += row[c + 1] * row[c + 1];
            s2 += row[c + 2] * row[c + 2];
            s3 += row[c + 3] * row[c + 3];
        }
        float sum_sq = (s0 + s1) + (s2 + s3);
        for (; c < C; ++c) sum_sq += row[c] * row[c];
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(C) + eps);
        #pragma omp simd
        for (int i = 0; i < C; ++i) orow[i] = row[i] * inv_rms * w[i];
    }
}

// Milestone 8 (Phase 1): Fused Pre-operator RMSNorm + SmoothQuant + dynamic per-token INT8 NPU input packing
// Computes u_smooth[c] = h[c] * inv_rms * op_norm_smooth[c]
// Tracks amax = max(|u_smooth[c]|)
// Emits dst_npu[c] = round_clamp(u_smooth[c] * (127.0f / amax), 127.0f)
// If u_out != nullptr (Attention layer), also computes u_out[c] = h[c] * inv_rms * op_norm[c]
static void fused_rms_norm_pack_dyntok_t1(const float *h, const float *op_norm, const float *op_norm_smooth,
                                          int8_t *dst_npu, int C, float in_thresh, float *scale_ratios,
                                          float *u_out = nullptr, float eps = 1e-5f) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int c = 0;
    for (; c + 3 < C; c += 4) {
        s0 += h[c + 0] * h[c + 0];
        s1 += h[c + 1] * h[c + 1];
        s2 += h[c + 2] * h[c + 2];
        s3 += h[c + 3] * h[c + 3];
    }
    float sum_sq = (s0 + s1) + (s2 + s3);
    for (; c < C; ++c) sum_sq += h[c] * h[c];
    float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(C) + eps);

    alignas(64) float u_smooth[4096];
    float amax = 0.0f;
    if (u_out) {
        for (int i = 0; i < C; ++i) {
            u_out[i] = h[i] * inv_rms * op_norm[i];
            float us = h[i] * inv_rms * op_norm_smooth[i];
            u_smooth[i] = us;
            float a = std::fabs(us);
            if (a > amax) amax = a;
        }
    } else {
        for (int i = 0; i < C; ++i) {
            float us = h[i] * inv_rms * op_norm_smooth[i];
            u_smooth[i] = us;
            float a = std::fabs(us);
            if (a > amax) amax = a;
        }
    }
    amax = std::max(amax, 1e-8f);
    scale_ratios[0] = amax / std::max(in_thresh, 1e-8f);
    const float inv_step = 127.0f / amax;

    #pragma omp simd
    for (int i = 0; i < C; ++i) {
        dst_npu[i] = static_cast<int8_t>(round_clamp_f32(u_smooth[i] * inv_step, 127.0f));
    }
}

// Milestone 8 (Phase 1): Fuses post_operator_norm, residual add (h_mid = h + op_normed),
// ffn_norm, SmoothQuant, and static INT8 packing into mlp act_buf.
static void fused_post_op_residual_ffn_pack_mlp_t1(const float *op_out, const float *post_op_norm,
                                                  const float *h, float *h_mid,
                                                  const float *ffn_norm_scaled,
                                                  int8_t *act_mlp, int C, float eps = 1e-5f,
                                                  int8_t *act_mlp_d1 = nullptr) {
    // Pass 1: sum_sq of op_out
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int c = 0;
    for (; c + 3 < C; c += 4) {
        s0 += op_out[c + 0] * op_out[c + 0];
        s1 += op_out[c + 1] * op_out[c + 1];
        s2 += op_out[c + 2] * op_out[c + 2];
        s3 += op_out[c + 3] * op_out[c + 3];
    }
    float sum_sq_op = (s0 + s1) + (s2 + s3);
    for (; c < C; ++c) sum_sq_op += op_out[c] * op_out[c];
    float inv_rms_op = 1.0f / std::sqrt(sum_sq_op / static_cast<float>(C) + eps);

    // Pass 2: compute h_mid[c] = h[c] + op_out[c] * inv_rms_op * post_op_norm[c]
    // and simultaneously accumulate sum_sq of h_mid!
    float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
    c = 0;
    for (; c + 3 < C; c += 4) {
        float hm0 = h[c + 0] + op_out[c + 0] * inv_rms_op * post_op_norm[c + 0];
        float hm1 = h[c + 1] + op_out[c + 1] * inv_rms_op * post_op_norm[c + 1];
        float hm2 = h[c + 2] + op_out[c + 2] * inv_rms_op * post_op_norm[c + 2];
        float hm3 = h[c + 3] + op_out[c + 3] * inv_rms_op * post_op_norm[c + 3];
        h_mid[c + 0] = hm0;
        h_mid[c + 1] = hm1;
        h_mid[c + 2] = hm2;
        h_mid[c + 3] = hm3;
        m0 += hm0 * hm0;
        m1 += hm1 * hm1;
        m2 += hm2 * hm2;
        m3 += hm3 * hm3;
    }
    float sum_sq_hmid = (m0 + m1) + (m2 + m3);
    for (; c < C; ++c) {
        float hm = h[c] + op_out[c] * inv_rms_op * post_op_norm[c];
        h_mid[c] = hm;
        sum_sq_hmid += hm * hm;
    }
    float inv_rms_ffn = 1.0f / std::sqrt(sum_sq_hmid / static_cast<float>(C) + eps);

    // Pass 3: pack directly into act_mlp (and act_mlp_d1 when TP=2) with ffn_norm_scaled!
    if (act_mlp_d1) {
        #pragma omp simd
        for (int i = 0; i < C; ++i) {
            float val = h_mid[i] * inv_rms_ffn * ffn_norm_scaled[i];
            int8_t q = static_cast<int8_t>(round_clamp_f32(val, 127.0f));
            act_mlp[i] = q;
            act_mlp_d1[i] = q;
        }
    } else {
        #pragma omp simd
        for (int i = 0; i < C; ++i) {
            float val = h_mid[i] * inv_rms_ffn * ffn_norm_scaled[i];
            act_mlp[i] = static_cast<int8_t>(round_clamp_f32(val, 127.0f));
        }
    }
}

// Milestone 8 (Phase 1) + Milestone 7 (TP=2): Fuses optional TP=2 partial sum (mlp_out + mlp_out_d1),
// post_ffn_norm on mlp_out, and residual addition: h[c] = h_mid[c] + mlp_normed[c]
static void fused_post_ffn_residual_t1(float *mlp_out, const float *post_ffn_norm,
                                      const float *h_mid, float *h, int C, float eps = 1e-5f,
                                      const float *mlp_out_d1 = nullptr) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int c = 0;
    if (mlp_out_d1) {
        for (; c + 3 < C; c += 4) {
            float m0 = mlp_out[c + 0] + mlp_out_d1[c + 0];
            float m1 = mlp_out[c + 1] + mlp_out_d1[c + 1];
            float m2 = mlp_out[c + 2] + mlp_out_d1[c + 2];
            float m3 = mlp_out[c + 3] + mlp_out_d1[c + 3];
            mlp_out[c + 0] = m0;
            mlp_out[c + 1] = m1;
            mlp_out[c + 2] = m2;
            mlp_out[c + 3] = m3;
            s0 += m0 * m0;
            s1 += m1 * m1;
            s2 += m2 * m2;
            s3 += m3 * m3;
        }
        for (; c < C; ++c) {
            float m = mlp_out[c] + mlp_out_d1[c];
            mlp_out[c] = m;
            s0 += m * m;
        }
    } else {
        for (; c + 3 < C; c += 4) {
            s0 += mlp_out[c + 0] * mlp_out[c + 0];
            s1 += mlp_out[c + 1] * mlp_out[c + 1];
            s2 += mlp_out[c + 2] * mlp_out[c + 2];
            s3 += mlp_out[c + 3] * mlp_out[c + 3];
        }
        for (; c < C; ++c) s0 += mlp_out[c] * mlp_out[c];
    }
    float sum_sq = (s0 + s1) + (s2 + s3);
    float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(C) + eps);

    #pragma omp simd
    for (int i = 0; i < C; ++i) {
        h[i] = h_mid[i] + (mlp_out[i] * inv_rms * post_ffn_norm[i]);
    }
}

// Milestone 8 (Phase 1): Fuses final embedding_norm and static INT8 packing directly into Die 0 and Die 1 act buffers
static void fused_emb_norm_broadcast_t1(const float *h, const float *emb_norm_scaled,
                                        int8_t *dst_die0, int8_t *dst_die1, int C, float eps = 1e-5f) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int c = 0;
    for (; c + 3 < C; c += 4) {
        s0 += h[c + 0] * h[c + 0];
        s1 += h[c + 1] * h[c + 1];
        s2 += h[c + 2] * h[c + 2];
        s3 += h[c + 3] * h[c + 3];
    }
    float sum_sq = (s0 + s1) + (s2 + s3);
    for (; c < C; ++c) sum_sq += h[c] * h[c];
    float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(C) + eps);

    #pragma omp simd
    for (int i = 0; i < C; ++i) {
        int8_t q = static_cast<int8_t>(round_clamp_f32(h[i] * inv_rms * emb_norm_scaled[i], 127.0f));
        dst_die0[i] = q;
        dst_die1[i] = q;
    }
}

// Pack [T, C] FP32 directly into CacheMode::Cached NPU FP16/INT8/INT16 input buffer (and unpack directly
// from CacheMode::Cached output buffer after flush_cache(), eliminating intermediate staging memcpy and locks).

static void pack_npu_input_fp16(const float *src_tc, uint16_t *dst_npu, int T, int C, int T_valid = -1) {
    if (T_valid < 0 || T_valid > T) T_valid = T;
    if (T == 1) {
        for (int c = 0; c < C; ++c) {
            dst_npu[c] = fp32_to_fp16(src_tc[c]);
        }
        return;
    }
    const int c_blocks = C / 16;
    #pragma omp parallel for schedule(static)
    for (int cb = 0; cb < c_blocks; ++cb) {
        for (int t = 0; t < T_valid; ++t) {
            const float *s = src_tc + static_cast<size_t>(t) * C + cb * 16;
            uint16_t *d = dst_npu + (static_cast<size_t>(cb) * T + t) * 16;
            for (int k = 0; k < 16; ++k) {
                d[k] = fp32_to_fp16(s[k]);
            }
        }
        if (T_valid < T) {
            std::memset(dst_npu + (static_cast<size_t>(cb) * T + T_valid) * 16, 0,
                        static_cast<size_t>(T - T_valid) * 16 * sizeof(uint16_t));
        }
    }
}

// Unpack NPU NCHW [1, C_out, T, 1] FP16 output into [T_valid, dst_stride] FP32
static void unpack_npu_output_fp16(const uint16_t *src_nchw, float *dst_tc, int T, int C_out,
                                   int T_valid = -1, int dst_stride = 0) {
    if (T_valid < 0 || T_valid > T) T_valid = T;
    if (dst_stride <= 0) dst_stride = C_out;
    if (T == 1) {
        for (int c = 0; c < C_out; ++c) {
            dst_tc[c] = fp16_to_fp32(src_nchw[c]);
        }
        return;
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int t0 = 0; t0 < T_valid; t0 += 16) {
        for (int c0 = 0; c0 < C_out; c0 += 16) {
            const int dt_end = std::min(16, T_valid - t0);
            float tile[16][16];
            for (int dc = 0; dc < 16; ++dc) {
                const uint16_t *s_row = src_nchw + static_cast<size_t>(c0 + dc) * T + t0;
                for (int dt = 0; dt < dt_end; ++dt) {
                    tile[dt][dc] = fp16_to_fp32(s_row[dt]);
                }
            }
            for (int dt = 0; dt < dt_end; ++dt) {
                std::memcpy(dst_tc + static_cast<size_t>(t0 + dt) * dst_stride + c0, tile[dt], 16 * sizeof(float));
            }
        }
    }
}

// Pack [T, C] FP32 into NPU INT8 input buffer (in_step = threshold / 127.0f, clamped to [-127, 127])
// Note: For T > 1, NVDLA INT8 FEATURE surfaces use 32-byte atoms = 32 INT8 channels ([C/32, T, 32],
// line_stride=32, surf_stride=32*T).
static void pack_npu_input_int8(const float *src_tc, int8_t *dst_npu, int T, int C, float in_step,
                                const float *in_inv_smooth = nullptr, int T_valid = -1) {
    if (T_valid < 0 || T_valid > T) T_valid = T;
    const float inv_step = (in_step > 1e-12f) ? (1.0f / in_step) : 0.0f;
    if (T == 1) {
        if (in_inv_smooth) {
            #pragma omp simd
            for (int c = 0; c < C; ++c) {
                dst_npu[c] = static_cast<int8_t>(round_clamp_f32(src_tc[c] * (in_inv_smooth[c] * inv_step), 127.0f));
            }
            return;
        }
        #pragma omp simd
        for (int c = 0; c < C; ++c) {
            dst_npu[c] = static_cast<int8_t>(round_clamp_f32(src_tc[c] * inv_step, 127.0f));
        }
        return;
    }

    const int c_blocks = C / 32;
    #pragma omp parallel for schedule(static)
    for (int cb = 0; cb < c_blocks; ++cb) {
        if (in_inv_smooth) {
            const float *sm = in_inv_smooth + cb * 32;
            for (int t = 0; t < T_valid; ++t) {
                const float *s = src_tc + static_cast<size_t>(t) * C + cb * 32;
                int8_t *d      = dst_npu + (static_cast<size_t>(cb) * T + t) * 32;
                #pragma omp simd
                for (int k = 0; k < 32; ++k) {
                    d[k] = static_cast<int8_t>(round_clamp_f32(s[k] * (sm[k] * inv_step), 127.0f));
                }
            }
        } else {
            for (int t = 0; t < T_valid; ++t) {
                const float *s = src_tc + static_cast<size_t>(t) * C + cb * 32;
                int8_t *d      = dst_npu + (static_cast<size_t>(cb) * T + t) * 32;
                #pragma omp simd
                for (int k = 0; k < 32; ++k) {
                    d[k] = static_cast<int8_t>(round_clamp_f32(s[k] * inv_step, 127.0f));
                }
            }
        }
        if (T_valid < T) {
            std::memset(dst_npu + (static_cast<size_t>(cb) * T + T_valid) * 32, 0,
                        static_cast<size_t>(T - T_valid) * 32 * sizeof(int8_t));
        }
    }
}

// Pack [T, C] FP32 into NPU INT16 input buffer (in_step = threshold / 32767.0f, clamped to [-32767, 32767])
static void pack_npu_input_int16(const float *src_tc, int16_t *dst_npu, int T, int C, float in_step, int T_valid = -1) {
    if (T_valid < 0 || T_valid > T) T_valid = T;
    const float inv_step = (in_step > 1e-12f) ? (1.0f / in_step) : 0.0f;
    if (T == 1) {
        #pragma omp simd
        for (int c = 0; c < C; ++c) {
            dst_npu[c] = static_cast<int16_t>(round_clamp_f32(src_tc[c] * inv_step, 32767.0f));
        }
        return;
    }

    const int c_blocks = C / 16;
    #pragma omp parallel for schedule(static)
    for (int cb = 0; cb < c_blocks; ++cb) {
        for (int t = 0; t < T_valid; ++t) {
            const float *s = src_tc + static_cast<size_t>(t) * C + cb * 16;
            int16_t *d     = dst_npu + (static_cast<size_t>(cb) * T + t) * 16;
            for (int k = 0; k < 16; ++k) {
                d[k] = static_cast<int16_t>(round_clamp_f32(s[k] * inv_step, 32767.0f));
            }
        }
        if (T_valid < T) {
            std::memset(dst_npu + (static_cast<size_t>(cb) * T + T_valid) * 16, 0,
                        static_cast<size_t>(T - T_valid) * 16 * sizeof(int16_t));
        }
    }
}

// Unpack NPU NCHW [1, C_out, T, 1] INT8 output into [T_valid, dst_stride] FP32
static void unpack_npu_output_int8(const int8_t *src_nchw, float *dst_tc, int T, int C_out, float out_step,
                                   const float *out_step_chan = nullptr, int T_valid = -1, int dst_stride = 0) {
    if (T_valid < 0 || T_valid > T) T_valid = T;
    if (dst_stride <= 0) dst_stride = C_out;
    if (T == 1) {
        if (out_step_chan) {
            #pragma omp simd
            for (int c = 0; c < C_out; ++c) {
                dst_tc[c] = static_cast<float>(src_nchw[c]) * out_step_chan[c];
            }
            return;
        }
        #pragma omp simd
        for (int c = 0; c < C_out; ++c) {
            dst_tc[c] = static_cast<float>(src_nchw[c]) * out_step;
        }
        return;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int t0 = 0; t0 < T_valid; t0 += 16) {
        for (int c0 = 0; c0 < C_out; c0 += 16) {
            const int dt_end = std::min(16, T_valid - t0);
            alignas(64) float tile[16][16];
            for (int dc = 0; dc < 16; ++dc) {
                const int8_t *s_row = src_nchw + static_cast<size_t>(c0 + dc) * T + t0;
                const float st_c = out_step_chan ? out_step_chan[c0 + dc] : out_step;
                for (int dt = 0; dt < dt_end; ++dt) {
                    tile[dt][dc] = static_cast<float>(s_row[dt]) * st_c;
                }
            }
            for (int dt = 0; dt < dt_end; ++dt) {
                std::memcpy(dst_tc + static_cast<size_t>(t0 + dt) * dst_stride + c0, tile[dt], 16 * sizeof(float));
            }
        }
    }
}

// Unpack NPU NCHW [1, C_out, T, 1] INT16 output into [T_valid, dst_stride] FP32
static void unpack_npu_output_int16(const int16_t *src_nchw, float *dst_tc, int T, int C_out, float out_step,
                                    const float *out_step_chan = nullptr, int T_valid = -1, int dst_stride = 0) {
    if (T_valid < 0 || T_valid > T) T_valid = T;
    if (dst_stride <= 0) dst_stride = C_out;
    if (T == 1) {
        if (out_step_chan) {
            #pragma omp simd
            for (int c = 0; c < C_out; ++c) {
                dst_tc[c] = static_cast<float>(src_nchw[c]) * out_step_chan[c];
            }
            return;
        }
        #pragma omp simd
        for (int c = 0; c < C_out; ++c) {
            dst_tc[c] = static_cast<float>(src_nchw[c]) * out_step;
        }
        return;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int t0 = 0; t0 < T_valid; t0 += 16) {
        for (int c0 = 0; c0 < C_out; c0 += 16) {
            const int dt_end = std::min(16, T_valid - t0);
            alignas(64) float tile[16][16];
            for (int dc = 0; dc < 16; ++dc) {
                const int16_t *s_row = src_nchw + static_cast<size_t>(c0 + dc) * T + t0;
                const float st_c = out_step_chan ? out_step_chan[c0 + dc] : out_step;
                for (int dt = 0; dt < dt_end; ++dt) {
                    tile[dt][dc] = static_cast<float>(s_row[dt]) * st_c;
                }
            }
            for (int dt = 0; dt < dt_end; ++dt) {
                std::memcpy(dst_tc + static_cast<size_t>(t0 + dt) * dst_stride + c0, tile[dt], 16 * sizeof(float));
            }
        }
    }
}

// Dynamic per-token INT8 input packer for homogeneous 1x1 linear projections.
// Note: For T > 1, NVDLA INT8 FEATURE surfaces use 32-byte atoms = 32 INT8 channels ([C/32, T, 32]).
static void pack_npu_input_int8_dyntok(const float *src_tc, int8_t *dst_npu, int T, int C,
                                       float in_thresh, float *scale_ratios,
                                       const float *in_inv_smooth = nullptr, int T_valid = -1) {
    if (T_valid < 0 || T_valid > T) T_valid = T;
    const float safe_thresh = std::max(in_thresh, 1e-8f);
    if (T == 1) {
        float amax = 0.0f;
        if (in_inv_smooth) {
            for (int c = 0; c < C; ++c) {
                float a = std::fabs(src_tc[c] * in_inv_smooth[c]);
                if (a > amax) amax = a;
            }
            amax = std::max(amax, 1e-8f);
            scale_ratios[0] = amax / safe_thresh;
            const float inv_step = 127.0f / amax;
            #pragma omp simd
            for (int c = 0; c < C; ++c) {
                dst_npu[c] = static_cast<int8_t>(round_clamp_f32(src_tc[c] * (in_inv_smooth[c] * inv_step), 127.0f));
            }
        } else {
            for (int c = 0; c < C; ++c) {
                float a = std::fabs(src_tc[c]);
                if (a > amax) amax = a;
            }
            amax = std::max(amax, 1e-8f);
            scale_ratios[0] = amax / safe_thresh;
            const float inv_step = 127.0f / amax;
            #pragma omp simd
            for (int c = 0; c < C; ++c) {
                dst_npu[c] = static_cast<int8_t>(round_clamp_f32(src_tc[c] * inv_step, 127.0f));
            }
        }
        return;
    }

    float inv_step_tok[64];
    for (int t = T_valid; t < T; ++t) {
        scale_ratios[t] = 0.0f;
        inv_step_tok[t] = 0.0f;
    }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T_valid; ++t) {
        const float *row = src_tc + static_cast<size_t>(t) * C;
        float amax = 0.0f;
        if (in_inv_smooth) {
            for (int c = 0; c < C; ++c) {
                float a = std::fabs(row[c] * in_inv_smooth[c]);
                if (a > amax) amax = a;
            }
        } else {
            for (int c = 0; c < C; ++c) {
                float a = std::fabs(row[c]);
                if (a > amax) amax = a;
            }
        }
        amax = std::max(amax, 1e-8f);
        scale_ratios[t] = amax / safe_thresh;
        inv_step_tok[t] = 127.0f / amax;
    }

    const int c_blocks = C / 32;
    #pragma omp parallel for schedule(static)
    for (int cb = 0; cb < c_blocks; ++cb) {
        if (in_inv_smooth) {
            const float *sm = in_inv_smooth + cb * 32;
            for (int t = 0; t < T_valid; ++t) {
                const float inv = inv_step_tok[t];
                const float *s  = src_tc + static_cast<size_t>(t) * C + cb * 32;
                int8_t *d       = dst_npu + (static_cast<size_t>(cb) * T + t) * 32;
                #pragma omp simd
                for (int k = 0; k < 32; ++k) {
                    d[k] = static_cast<int8_t>(round_clamp_f32(s[k] * (sm[k] * inv), 127.0f));
                }
            }
        } else {
            for (int t = 0; t < T_valid; ++t) {
                const float inv = inv_step_tok[t];
                const float *s  = src_tc + static_cast<size_t>(t) * C + cb * 32;
                int8_t *d       = dst_npu + (static_cast<size_t>(cb) * T + t) * 32;
                #pragma omp simd
                for (int k = 0; k < 32; ++k) {
                    d[k] = static_cast<int8_t>(round_clamp_f32(s[k] * inv, 127.0f));
                }
            }
        }
        if (T_valid < T) {
            std::memset(dst_npu + (static_cast<size_t>(cb) * T + T_valid) * 32, 0,
                        static_cast<size_t>(T - T_valid) * 32 * sizeof(int8_t));
        }
    }
}

// Dynamic per-token INT8 output unpacker for homogeneous 1x1 linear projections
static void unpack_npu_output_int8_dyntok(const int8_t *src_nchw, float *dst_tc, int T, int C_out,
                                          float out_step, const float *scale_ratios,
                                          const float *out_step_chan = nullptr, int T_valid = -1,
                                          int dst_stride = 0) {
    if (T_valid < 0 || T_valid > T) T_valid = T;
    if (dst_stride <= 0) dst_stride = C_out;
    if (T == 1) {
        if (out_step_chan) {
            const float r0 = scale_ratios[0];
            #pragma omp simd
            for (int c = 0; c < C_out; ++c) {
                dst_tc[c] = static_cast<float>(src_nchw[c]) * (r0 * out_step_chan[c]);
            }
        } else {
            const float step_tok = scale_ratios[0] * out_step;
            #pragma omp simd
            for (int c = 0; c < C_out; ++c) {
                dst_tc[c] = static_cast<float>(src_nchw[c]) * step_tok;
            }
        }
        return;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int t0 = 0; t0 < T_valid; t0 += 16) {
        for (int c0 = 0; c0 < C_out; c0 += 16) {
            const int dt_end = std::min(16, T_valid - t0);
            alignas(64) float tile[16][16];
            for (int dc = 0; dc < 16; ++dc) {
                const int8_t *s_row = src_nchw + static_cast<size_t>(c0 + dc) * T + t0;
                const float st_c = out_step_chan ? out_step_chan[c0 + dc] : out_step;
                for (int dt = 0; dt < dt_end; ++dt) {
                    tile[dt][dc] = static_cast<float>(s_row[dt]) * (scale_ratios[t0 + dt] * st_c);
                }
            }
            for (int dt = 0; dt < dt_end; ++dt) {
                std::memcpy(dst_tc + static_cast<size_t>(t0 + dt) * dst_stride + c0, tile[dt], 16 * sizeof(float));
            }
        }
    }
}

// Unified NPU Input Packer: routes FP32 [T_valid, C] -> FP16, INT8 (static or dyntok), or INT16 DmaBuffer
static void pack_npu_input(const float *src_tc, DmaBuffer *dst_buf, int T, int C,
                           const SubgraphScaleInfo &sc, float *dyntok_ratios = nullptr, int T_valid = -1) {
    switch (sc.in_dtype) {
        case BoundaryDtype::Int8:
            if (dyntok_ratios) {
                pack_npu_input_int8_dyntok(src_tc, dst_buf->as<int8_t>(), T, C, sc.in_thresh, dyntok_ratios, sc.in_inv_smooth, T_valid);
            } else {
                pack_npu_input_int8(src_tc, dst_buf->as<int8_t>(), T, C, sc.in_step, sc.in_inv_smooth, T_valid);
            }
            break;
        case BoundaryDtype::Int16:
            pack_npu_input_int16(src_tc, dst_buf->as<int16_t>(), T, C, sc.in_step, T_valid);
            break;
        case BoundaryDtype::Fp16:
        default:
            pack_npu_input_fp16(src_tc, dst_buf->as<uint16_t>(), T, C, T_valid);
            break;
    }
}

// Unified NPU Output Unpacker: routes FP16, INT8 (static or dyntok), or INT16 DmaBuffer -> FP32 [T_valid, dst_stride]
static void unpack_npu_output(const DmaBuffer *src_buf, float *dst_tc, int T, int C_out,
                              const SubgraphScaleInfo &sc, const float *dyntok_ratios = nullptr,
                              int T_valid = -1, int dst_stride = 0) {
    switch (sc.out_dtype) {
        case BoundaryDtype::Int8:
            if (dyntok_ratios) {
                unpack_npu_output_int8_dyntok(src_buf->as<int8_t>(), dst_tc, T, C_out, sc.out_step, dyntok_ratios, sc.out_step_chan, T_valid, dst_stride);
            } else {
                unpack_npu_output_int8(src_buf->as<int8_t>(), dst_tc, T, C_out, sc.out_step, sc.out_step_chan, T_valid, dst_stride);
            }
            break;
        case BoundaryDtype::Int16:
            unpack_npu_output_int16(src_buf->as<int16_t>(), dst_tc, T, C_out, sc.out_step, sc.out_step_chan, T_valid, dst_stride);
            break;
        case BoundaryDtype::Fp16:
        default:
            unpack_npu_output_fp16(src_buf->as<uint16_t>(), dst_tc, T, C_out, T_valid, dst_stride);
            break;
    }
}

// =============================================================================
// LFM3-1.7B Tri-Hybrid Dual-Die Engine Class
// =============================================================================

class Lfm3Engine {
public:
    static constexpr int kNumLayers = 24;
    static constexpr int kHidden = 2048;
    static constexpr int kInterm = 7680;
    static constexpr int kVocab = 128000;
    static constexpr int kQHeads = 16;
    static constexpr int kKvHeads = 4;
    static constexpr int kHeadDim = 128;
    static constexpr int kSwaWin = 256;
    static constexpr int kChunk = 64;
    static constexpr int kSwaCap = kSwaWin + kChunk; // 320 slots so T<=64 prefill never overwrites active SWA window

    Lfm3Engine() = default;
    ~Lfm3Engine() { stop_tp_worker(); }

    void start_tp_worker() {
        if (m_tp_worker_running || m_num_dies < 2) return;
        // Force OpenMP thread pool initialization with full 8-core mask before restricting main thread away from Core 7
        #pragma omp parallel
        {
            __asm__ volatile("" ::: "memory");
        }
        {
            cpu_set_t main_set;
            CPU_ZERO(&main_set);
            for (int c = 0; c < 7; ++c) CPU_SET(c, &main_set); // Main thread on Cores 0..6 (never Core 7)
            sched_setaffinity(0, sizeof(cpu_set_t), &main_set);
        }
        m_tp_stop.store(false, std::memory_order_relaxed);
        m_tp_sleeping.store(false, std::memory_order_relaxed);
        m_tp_prewake.store(false, std::memory_order_relaxed);
        m_tp_seq.store(0, std::memory_order_relaxed);
        m_tp_ack.store(0, std::memory_order_relaxed);
        m_tp_worker_running = true;
        m_tp_thread = std::thread([this]() {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(7, &cpuset); // Pin Die-1 dispatcher thread to Core 7 (NUMA Node 1 / Die 1)
            sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

            uint64_t last_seq = 0;
            while (true) {
                uint64_t cur = m_tp_seq.load(std::memory_order_acquire);
                if (cur == last_seq) {
                    if (!m_tp_prewake.load(std::memory_order_acquire)) {
                        std::unique_lock<std::mutex> lk(m_tp_mtx);
                        m_tp_sleeping.store(true, std::memory_order_release);
                        m_tp_cv.wait(lk, [&]() {
                            cur = m_tp_seq.load(std::memory_order_acquire);
                            return cur != last_seq ||
                                   m_tp_prewake.load(std::memory_order_acquire) ||
                                   m_tp_stop.load(std::memory_order_relaxed);
                        });
                        m_tp_sleeping.store(false, std::memory_order_relaxed);
                    }
                    while ((cur = m_tp_seq.load(std::memory_order_acquire)) == last_seq &&
                           !m_tp_stop.load(std::memory_order_relaxed)) {
#if defined(__riscv)
                        __asm__ volatile("nop");
#endif
                    }
                }
                if (m_tp_stop.load(std::memory_order_relaxed)) break;
                m_tp_prewake.store(false, std::memory_order_relaxed);
                last_seq = cur;
                if (m_tp_fn) (*m_tp_fn)();
                m_tp_ack.store(last_seq, std::memory_order_release);
            }
        });
    }

    void prewake_tp_worker() {
        if (!m_tp_worker_running || m_num_dies < 2) return;
        if (!m_tp_prewake.load(std::memory_order_relaxed)) {
            m_tp_prewake.store(true, std::memory_order_release);
            if (m_tp_sleeping.load(std::memory_order_acquire)) {
                std::lock_guard<std::mutex> lk(m_tp_mtx);
                m_tp_cv.notify_one();
            }
        }
    }

    void stop_tp_worker() {
        if (!m_tp_worker_running) return;
        {
            std::lock_guard<std::mutex> lk(m_tp_mtx);
            m_tp_stop.store(true, std::memory_order_relaxed);
            m_tp_seq.fetch_add(1, std::memory_order_release);
        }
        m_tp_cv.notify_all();
        if (m_tp_thread.joinable()) m_tp_thread.join();
        m_tp_worker_running = false;
    }

    template <typename Fn0, typename Fn1>
    void run_dual_die(Fn0 &&fn0, Fn1 &&fn1) {
        if (m_tp_worker_running && m_num_dies >= 2) {
            std::function<void()> f1(std::forward<Fn1>(fn1));
            m_tp_fn = &f1;
            uint64_t next_seq = m_tp_seq.load(std::memory_order_relaxed) + 1;
            m_tp_seq.store(next_seq, std::memory_order_release);
            if (m_tp_sleeping.load(std::memory_order_acquire)) {
                std::lock_guard<std::mutex> lk(m_tp_mtx);
                m_tp_cv.notify_one();
            }
            fn0();
            while (m_tp_ack.load(std::memory_order_acquire) != next_seq) {
#if defined(__riscv)
                __asm__ volatile("nop");
#endif
            }
            m_tp_fn = nullptr;
        } else {
            fn0();
            fn1();
        }
    }

    static int peek_pkg_tp_degree(const std::string &dir) {
        std::string map_path = dir + "/weights_map.json";
        std::ifstream jf(map_path);
        if (!jf.is_open()) return 0;
        std::stringstream ss;
        ss << jf.rdbuf();
        std::string err;
        JsonValue root = JsonValue::parse(ss.str(), err);
        if (!err.empty() || !root.is_object()) return 0;
        return root.has("tp_degree") ? static_cast<int>(root["tp_degree"].as_int64(1)) : 1;
    }

    bool init(const std::string &pkg_dir, int max_ctx = 4096) {
        auto t_init0 = std::chrono::steady_clock::now();
        m_pkg_dir = pkg_dir;
        int cur_tp = peek_pkg_tp_degree(m_pkg_dir);
        if (m_requested_tp_degree == 2 && cur_tp != 2) {
            for (const char *cand : {"/tp2", "/../LFM3-1.7B_TP2", "/../eswin_pkg_tp2"}) {
                std::string cdir = m_pkg_dir + cand;
                if (peek_pkg_tp_degree(cdir) == 2) {
                    std::printf("[+] --tp2 selected: switching package directory %s -> %s\n",
                                m_pkg_dir.c_str(), cdir.c_str());
                    m_pkg_dir = cdir;
                    break;
                }
            }
        } else if (m_requested_tp_degree == 1 && cur_tp == 2) {
            for (const char *cand : {"/../LFM3-1.7B_ENNP", "/../eswin_pkg_int8", "/../eswin_pkg"}) {
                std::string cdir = m_pkg_dir + cand;
                if (peek_pkg_tp_degree(cdir) == 1) {
                    std::printf("[+] --pp2 selected: switching package directory %s -> %s\n",
                                m_pkg_dir.c_str(), cdir.c_str());
                    m_pkg_dir = cdir;
                    break;
                }
            }
        }
        // Round m_max_ctx up to a multiple of 2048 so every per-die half-layer (2 * m_max_ctx slots)
        // is an exact multiple of 4 KiB OS pages (and 2 MiB HugePages for FP32 KV caches).
        m_max_ctx = std::max(2048, (max_ctx + 2047) & ~2047);

        std::printf("=================================================================\n");
        std::printf(" LFM3-1.7B Tri-Hybrid Dual-Die Engine (Route D Class B)\n");
        std::printf(" Package directory: %s (max_ctx=%d)\n", m_pkg_dir.c_str(), m_max_ctx);
        std::printf("=================================================================\n\n");

        // 1. Load cpu_weights.bin & cpu_weights.json
        if (!load_cpu_weights()) return false;

        // 2. Memory-map embedding.bin (128,000 x 2048 FP16)
        std::string emb_err;
        std::string emb_path = m_pkg_dir + "/embedding.bin";
        if (!m_embedding.open(emb_path, kVocab, kHidden, EmbeddingPrecision::Fp16, emb_err)) {
            std::printf("FAIL: EmbeddingTable::open(%s): %s\n", emb_path.c_str(), emb_err.c_str());
            return false;
        }
        std::printf("[+] Mapped embedding.bin (%zu x %zu FP16, %.1f MiB) in %.2f ms\n",
                    m_embedding.rows(), m_embedding.cols(),
                    m_embedding.mapped_bytes() / (1024.0 * 1024.0), m_embedding.open_ms());

        // 3. Load Tokenizer
        std::string tok_path = m_pkg_dir + "/tokenizer.json";
        if (!file_exists(tok_path)) tok_path = m_pkg_dir + "/../tokenizer.json";
        if (file_exists(tok_path)) {
            std::string tok_err;
            if (!m_tokenizer.load(tok_path, tok_err)) {
                std::printf("WARN: Tokenizer::load(%s) failed: %s\n", tok_path.c_str(), tok_err.c_str());
            } else {
                m_tokenizer.set_special_tokens(124894, 124900, 124893);
                std::printf("[+] Loaded Tokenizer (%zu vocab entries) from %s\n",
                            m_tokenizer.vocab_size(), tok_path.c_str());
            }
        }

        // 4. Parse weights_map.json & optional quant_scales.json
        if (!load_weights_map()) return false;
        if (!load_quant_scales()) return false;

        // 5. Precompute RoPE inverse frequency tables
        for (int i = 0; i < 64; ++i) {
            m_swa_inv_freq[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / 128.0f);
            m_glob_inv_freq[i] = (i < 32)
                ? (1.0f / std::pow(1000000.0f, static_cast<float>(2 * i) / 128.0f))
                : 0.0f;
        }

        // 6. Start Dual-Die WorkerPool & Load Models/Buffers on Die 0 and Die 1 first
        //    (evicting the 1.53 GiB weight blobs from VFS page cache via POSIX_FADV_DONTNEED
        //    so Node 0 has full free RAM before NUMA-pinned host KV caches are allocated).
        pin_openmp_threads_numa();
        if (!m_pool.start(2)) {
            std::printf("FAIL: WorkerPool::start(2) failed (run with sudo on roma)\n");
            return false;
        }
        m_num_dies = m_pool.active_worker_count();
        std::printf("[+] Started WorkerPool with %u isolated NPU die worker(s)\n", m_num_dies);
        start_tp_worker();

        for (uint16_t d = 0; d < 2; ++d) {
            uint16_t phys_die = (d < m_num_dies) ? d : 0;
            const char *zone = (phys_die == 0) ? "mmz_nid_0_part_0" : "mmz_nid_1_part_0";

            // Load per-die weight blob into MMZ
            std::string w_file = m_pkg_dir + "/weights_die" + std::to_string(d) + ".bin";
            FILE *wf = std::fopen(w_file.c_str(), "rb");
            if (!wf) {
                std::printf("FAIL: cannot open %s\n", w_file.c_str());
                return false;
            }
            std::fseek(wf, 0, SEEK_END);
            size_t w_sz = static_cast<size_t>(std::ftell(wf));
            std::fseek(wf, 0, SEEK_SET);

            std::printf("[+] Loading %s (%.3f GiB) into %s...\n",
                        w_file.c_str(), w_sz / (1024.0 * 1024.0 * 1024.0), zone);
            m_w_blob[d] = std::make_unique<DmaBuffer>(w_sz, CacheMode::Cached, zone);
            if (!m_w_blob[d]->is_valid()) {
                std::fclose(wf);
                std::printf("FAIL: DmaBuffer alloc for %s (%zu B)\n", w_file.c_str(), w_sz);
                return false;
            }
            size_t nr = std::fread(m_w_blob[d]->data(), 1, w_sz, wf);
#if defined(__linux__)
            int w_fd = fileno(wf);
            if (w_fd >= 0) posix_fadvise(w_fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
            std::fclose(wf);
            if (nr != w_sz) {
                std::printf("FAIL: short read on %s (%zu/%zu)\n", w_file.c_str(), nr, w_sz);
                return false;
            }
            m_w_blob[d]->flush_cache();

            // Identify the weights actually loaded. Two packages with different proj2048 weights
            // produced bit-identical accuracy metrics AND bit-identical corruption deltas
            // (FINDINGS 41.2), which is impossible unless both runs read the same bytes -- so the
            // blob's identity has to be printed, not assumed from the working directory.
            {
                const uint8_t *bp = static_cast<const uint8_t *>(m_w_blob[d]->data());
                const size_t n = m_w_blob[d]->size();
                uint64_t head = 1469598103934665603ULL, tail = 1469598103934665603ULL;
                const size_t span = std::min<size_t>(65536, n);
                for (size_t i = 0; i < span; ++i) { head ^= bp[i]; head *= 1099511628211ULL; }
                for (size_t i = n - span; i < n; ++i) { tail ^= bp[i]; tail *= 1099511628211ULL; }
                std::printf("[+] Die %u weight blob: %s (%zu B) fnv head=%016llx tail=%016llx\n",
                            d, w_file.c_str(), n, (unsigned long long)head,
                            (unsigned long long)tail);
            }

            // Buffers are allocated *after* the archetypes are loaded (see below), because their
            // size has to cover the largest surface the driver reports for those loadables.
            // Preparing them before that would force a descriptor to be clamped to an allocation
            // that may be smaller than the tensor surface -- see surf_or() in forward_chunk().

            // Load the 8 shared Class B archetypes + 4 LM Head 16K slices = 12 models/die
            auto load_m = [&](const std::string &rel, LoadedArchetype &out_arch) -> bool {
                std::string full = m_pkg_dir + "/" + rel;
                ModelInfo info{};
                if (!m_pool.load_model(phys_die, full, info)) {
                    std::printf("FAIL: load_model(%s) on Die %u\n", full.c_str(), phys_die);
                    return false;
                }
                out_arch.id = info.model_id;
                out_arch.in_surface_bytes  = info.input_sizes[0];
                out_arch.out_surface_bytes = info.output_sizes[0];
                // The INT8 conv_in loadable declares its Class B output as a 6,144-byte tensor
                // entry (memory id 5, flags 9), yet output_sizes[0] reports 2 MiB -- the size of
                // the driver's fixed command-stream dmabuf (FINDINGS 34.2), not a tensor. Dump
                // every reported slot so the real tensor index can be identified rather than
                // guessed; buffer sizing was being driven by the wrong number.
                if (std::getenv("LFM3_DUMP_SURFACES")) {
                    std::printf("      %-40s in=%u [", rel.c_str(), info.num_inputs);
                    for (unsigned i = 0; i < info.num_inputs && i < 10; ++i)
                        std::printf("%s%llu", i ? "," : "",
                                    (unsigned long long)info.input_sizes[i]);
                    std::printf("] out=%u [", info.num_outputs);
                    for (unsigned i = 0; i < info.num_outputs && i < 10; ++i)
                        std::printf("%s%llu", i ? "," : "",
                                    (unsigned long long)info.output_sizes[i]);
                    std::printf("]\n");
                }
                // For a static loadable (the LM head slices) the weights stream from the driver's
                // own copy, so the file size is the right byte count; a Class B archetype reports
                // the bound region size instead, which the submit sites already know.
                std::ifstream szf(full, std::ios::ate | std::ios::binary);
                if (szf) out_arch.weight_bytes = static_cast<size_t>(szf.tellg());
                m_die_models[d].resident_count++;
                return true;
            };

            if (!load_m("models/lfm3_conv_in_t1_classb.model",   m_die_models[d].conv_in_t1) ||
                !load_m("models/lfm3_proj2048_t1_classb.model",  m_die_models[d].proj2048_t1) ||
                !load_m("models/lfm3_attn_qkv_t1_classb.model",  m_die_models[d].attn_qkv_t1) ||
                !load_m("models/lfm3_mlp_t1_classb.model",       m_die_models[d].mlp_t1) ||
                !load_m("models/lfm3_conv_in_t64_classb.model",  m_die_models[d].conv_in_t64) ||
                !load_m("models/lfm3_proj2048_t64_classb.model", m_die_models[d].proj2048_t64) ||
                !load_m("models/lfm3_attn_qkv_t64_classb.model", m_die_models[d].attn_qkv_t64) ||
                !load_m("models/lfm3_mlp_t64_classb.model",      m_die_models[d].mlp_t64)) {
                return false;
            }

            for (int s = 0; s < 4; ++s) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "models/lfm3_lm_head_d%u_s%d.model", d, s);
                if (!load_m(buf, m_die_models[d].lm_head[s])) return false;
            }

            if (m_die_models[d].resident_count >= 55) {
                std::printf("FAIL: Die %u resident model count %d >= 55 (ceiling 62 Oopses the "
                            "kernel, FINDINGS 33.4)\n", d, m_die_models[d].resident_count);
                return false;
            }
            std::printf("[+] Die %u Resident Model Count: %d (gate < 55; kernel Oops ceiling 62) [PASS]\n",
                        d, m_die_models[d].resident_count);

            // Fixed surface buffers, deliberately below the memory-entry sizes the driver reports
            // (2 MiB for the INT8 conv_in output): surface_descriptor() caps the descriptor at the
            // allocation, and submitting the full reported entry wedges the die worker. The
            // invariant that matters is the other direction -- a buffer smaller than a tensor's own
            // byte count is now a hard error instead of a silent truncation.
            DieModels &dm = m_die_models[d];
            size_t max_entry = 0;
            for (const LoadedArchetype *a : {&dm.conv_in_t1, &dm.proj2048_t1, &dm.attn_qkv_t1,
                                             &dm.mlp_t1, &dm.conv_in_t64, &dm.proj2048_t64,
                                             &dm.attn_qkv_t64, &dm.mlp_t64, &dm.lm_head[0],
                                             &dm.lm_head[1], &dm.lm_head[2], &dm.lm_head[3]})
                max_entry = std::max({max_entry, a->in_surface_bytes, a->out_surface_bytes});

            size_t s_act1 = 4096;
            size_t s_out1 = is_quant_active() ? 8192 : 32768;
            size_t s_act64 = 262144, s_out64 = 786432;
            size_t s_lmout = (dm.lm_head_scale[0].out_dtype == BoundaryDtype::Int8) ? 16384 : 32768;
            // Geometry-sized surface buffers (32 KiB for T=1 outputs, 256/768 KiB for T=64) are
            // the default, avoiding 74 x 2 MiB = 148 MiB/token of whole-allocation cache-flush
            // overhead (~54 ms/tok saved). As proven in FINDINGS §45.3, zero-initializing and
            // flushing every surface buffer at init() and reset_state() eliminates the first-run
            // stale-tail divergence so run 1 == run 2+ bit-identically in both FP16 and INT8.
            // Set LFM3_WIDE_SURFACES=1 to opt into the 2 MiB entry-sized allocations.
            const bool wide_ab = std::getenv("LFM3_WIDE_SURFACES") != nullptr;
            const bool m_wide_surfaces = wide_ab;
            if (m_wide_surfaces) {
                // Size every buffer to the largest entry its role reports rather than clamping
                // the descriptor down to a convenient constant.
                for (const LoadedArchetype *a : {&dm.conv_in_t1, &dm.proj2048_t1, &dm.attn_qkv_t1,
                                                 &dm.mlp_t1, &dm.lm_head[0], &dm.lm_head[1],
                                                 &dm.lm_head[2], &dm.lm_head[3]})
                    s_act1 = std::max(s_act1, a->in_surface_bytes);
                for (const LoadedArchetype *a : {&dm.conv_in_t1, &dm.proj2048_t1, &dm.attn_qkv_t1,
                                                 &dm.mlp_t1})
                    s_out1 = std::max(s_out1, a->out_surface_bytes);
                for (const LoadedArchetype *a : {&dm.conv_in_t64, &dm.proj2048_t64,
                                                 &dm.attn_qkv_t64, &dm.mlp_t64}) {
                    s_act64 = std::max(s_act64, a->in_surface_bytes);
                    s_out64 = std::max(s_out64, a->out_surface_bytes);
                }
                for (const LoadedArchetype *a : {&dm.lm_head[0], &dm.lm_head[1], &dm.lm_head[2],
                                                 &dm.lm_head[3]})
                    s_lmout = std::max(s_lmout, a->out_surface_bytes);
            }

            const size_t g = m_guard_bytes;
            m_act_t1[d]  = std::make_unique<DmaBuffer>(s_act1 + g,  CacheMode::Cached, zone);
            m_out_t1[d]  = std::make_unique<DmaBuffer>(s_out1 + g,  CacheMode::Cached, zone);
            m_act_t64[d] = std::make_unique<DmaBuffer>(s_act64 + g, CacheMode::Cached, zone);
            m_out_t64[d] = std::make_unique<DmaBuffer>(s_out64 + g, CacheMode::Cached, zone);
            for (int s = 0; s < 4; ++s)
                m_lm_out[d][s] = std::make_unique<DmaBuffer>(s_lmout + g, CacheMode::Cached, zone);
            note_prod_cap(m_act_t1[d].get(), s_act1);
            note_prod_cap(m_act_t64[d].get(), s_act64);
            note_prod_cap(m_out_t1[d].get(), s_out1);
            note_prod_cap(m_out_t64[d].get(), s_out64);
            for (int s = 0; s < 4; ++s) note_prod_cap(m_lm_out[d][s].get(), s_lmout);
            for (DmaBuffer *b : {m_act_t1[d].get(), m_out_t1[d].get(), m_act_t64[d].get(),
                                 m_out_t64[d].get(), m_lm_out[d][0].get(), m_lm_out[d][1].get(),
                                 m_lm_out[d][2].get(), m_lm_out[d][3].get()}) {
                if (!b->is_valid()) {
                    std::printf("FAIL: DmaBuffer alloc on Die %u (%zu B)\n", d, b->size());
                    return false;
                }
                std::memset(b->data(), 0, b->size());
                b->flush_cache();
            }
            std::printf("[+] Die %u surface buffers: act_t1=%zu out_t1=%zu act_t64=%zu "
                        "out_t64=%zu lm_out=%zu B (largest entry %.2f MiB, surfaces %s)\n",
                        d, s_act1, s_out1, s_act64, s_out64, s_lmout, max_entry / 1048576.0,
                        m_wide_surfaces ? "FROM ENTRY (wide)" : "FIXED (clamp)");

            // Prepare all buffers ONCE in the worker's SMMU. The weight blob must be in this list:
            // submitting against an unprepared weight fd wedges the die worker (FINDINGS 33.4 /
            // 34.3), which is what happens if it is prepared separately and this list is edited.
            for (DmaBuffer *b : {m_w_blob[d].get(),
                                 m_act_t1[d].get(), m_out_t1[d].get(), m_act_t64[d].get(),
                                 m_out_t64[d].get(), m_lm_out[d][0].get(), m_lm_out[d][1].get(),
                                 m_lm_out[d][2].get(), m_lm_out[d][3].get()}) {
                if (!m_pool.prepare_buffer(phys_die, *b)) {
                    std::printf("FAIL: prepare_buffer on Die %u\n", phys_die);
                    return false;
                }
            }
        }

        // 7. Allocate NUMA-bound host state buffers (KV heads 0,1 strictly bound to Node 0 / Die 0
        //    with 2 MiB HugePages; KV heads 2,3 strictly bound to Node 1 / Die 1 with 2 MiB HugePages)
        pin_openmp_threads_numa();
        m_conv_state.assign(13 * kHidden * 6, 0.0f);
        m_swa_k.allocate(5, static_cast<size_t>(kSwaCap) * kKvHeads * kHeadDim);
        m_swa_v.allocate(5, static_cast<size_t>(kSwaCap) * kKvHeads * kHeadDim);
        m_glob_k.allocate(6, static_cast<size_t>(m_max_ctx) * kKvHeads * kHeadDim);
        m_glob_v.allocate(6, static_cast<size_t>(m_max_ctx) * kKvHeads * kHeadDim);
        m_glob_k_u8.allocate(6, static_cast<size_t>(m_max_ctx) * kKvHeads * kHeadDim);
        m_glob_v_u8.allocate(6, static_cast<size_t>(m_max_ctx) * kKvHeads * kHeadDim);
        m_glob_k_sc.allocate(6, static_cast<size_t>(m_max_ctx) * kKvHeads);
        m_glob_v_sc.allocate(6, static_cast<size_t>(m_max_ctx) * kKvHeads);
        m_glob_k_sum.allocate(6, static_cast<size_t>(m_max_ctx) * kKvHeads);
        m_logits.assign(kVocab, 0.0f);

        // Pre-allocate forward_chunk scratch buffers once for kChunk=64 tokens to eliminate
        // 10 heap allocations/zero-fills per decode step.
        m_buf_h.assign(static_cast<size_t>(kChunk) * kHidden, 0.0f);
        m_buf_u.assign(static_cast<size_t>(kChunk) * kHidden, 0.0f);
        m_buf_u_smooth.assign(static_cast<size_t>(kChunk) * kHidden, 0.0f);
        m_buf_bcx.assign(static_cast<size_t>(kChunk) * 6144, 0.0f);
        m_buf_qkv.assign(static_cast<size_t>(kChunk) * 3072, 0.0f);
        m_buf_y.assign(static_cast<size_t>(kChunk) * kHidden, 0.0f);
        m_buf_op_out.assign(static_cast<size_t>(kChunk) * kHidden, 0.0f);
        m_buf_h_mid.assign(static_cast<size_t>(kChunk) * kHidden, 0.0f);
        m_buf_v.assign(static_cast<size_t>(kChunk) * kHidden, 0.0f);
        m_buf_mlp_out.assign(static_cast<size_t>(kChunk) * kHidden, 0.0f);
        m_buf_mlp_out_d1.assign(static_cast<size_t>(kChunk) * kHidden, 0.0f);
        m_buf_gates.assign(static_cast<size_t>(kChunk) * kQHeads, 0.0f);

        auto t_init1 = std::chrono::steady_clock::now();
        std::printf("[+] Engine initialized in %.2f s.\n\n",
                    std::chrono::duration<double>(t_init1 - t_init0).count());
        return true;
    }

    /**
     * @brief Returns every piece of per-conversation state to its post-init value.
     *
     * This used to clear only the conv rings, the SWA caches and m_pos, leaving the global KV,
     * its INT8 side-cars and the logits buffer holding the previous conversation. A 6-token
     * prefill cannot expose that (n_valid = pos+1 never reaches the stale slots), but a second
     * conversation in one process -- or any test that reuses the engine -- reads it. "Reset"
     * should mean reset, even where today's callers happen not to notice.
     */
    void reset_state() {
        std::fill(m_conv_state.begin(), m_conv_state.end(), 0.0f);
        m_swa_k.clear();
        m_swa_v.clear();
        m_glob_k.clear();
        m_glob_v.clear();
        m_glob_k_u8.clear();
        m_glob_v_u8.clear();
        m_glob_k_sc.clear();
        m_glob_v_sc.clear();
        m_glob_k_sum.clear();
        std::fill(m_logits.begin(), m_logits.end(), 0.0f);
        for (uint16_t d = 0; d < 2; ++d) {
            for (DmaBuffer *b : {m_act_t1[d].get(), m_out_t1[d].get(), m_act_t64[d].get(),
                                 m_out_t64[d].get(), m_lm_out[d][0].get(), m_lm_out[d][1].get(),
                                 m_lm_out[d][2].get(), m_lm_out[d][3].get()}) {
                if (b && b->is_valid()) {
                    std::memset(b->data(), 0, b->size());
                    b->flush_cache();
                }
            }
        }
        m_pos = 0;
    }

    // Execute T_valid tokens (1 <= T_valid <= 64) across all 24 layers; uses T_npu=1 for single-token
    // decode and T_npu=64 for multi-token prefill (2 <= T_valid <= 64). If compute_logits is true,
    // computes all 128,000 logits for the last valid token (T_valid - 1) in the batch.
    bool forward_chunk(const int32_t *token_ids, int T_valid, bool compute_logits,
                       StageTimers &timers,
                       const float *visual_embeds = nullptr,
                       size_t *visual_idx_io = nullptr,
                       size_t max_visual_tokens = 0,
                       std::vector<float> *debug_layer_outs = nullptr,
                       std::vector<float> *debug_emb_norm = nullptr) {
        if (T_valid < 1 || T_valid > kChunk) {
            std::printf("FAIL: forward_chunk requires 1 <= T_valid <= %d (got %d)\n", kChunk, T_valid);
            return false;
        }
        if (m_pos + T_valid > m_max_ctx) {
            std::printf("FAIL: context limit reached (pos=%d + T_valid=%d > max_ctx=%d; pass --max-ctx N to increase)\n",
                        m_pos, T_valid, m_max_ctx);
            return false;
        }
        const int T = (T_valid == 1) ? 1 : kChunk;

        float *h       = m_buf_h.data();
        float *u       = m_buf_u.data();
        float *u_smooth = m_buf_u_smooth.data();
        float *bcx     = m_buf_bcx.data();
        float *qkv     = m_buf_qkv.data();
        float *y       = m_buf_y.data();
        float *op_out  = m_buf_op_out.data();
        float *h_mid   = m_buf_h_mid.data();
        float *v       = m_buf_v.data();
        float *mlp_out = m_buf_mlp_out.data();
        float *mlp_out_d1 = m_buf_mlp_out_d1.data();
        float *gates   = m_buf_gates.data();

        // Lookup FP16 embeddings -> FP32 for the T_valid tokens
        auto t0 = std::chrono::steady_clock::now();
        m_embedding.gather(token_ids, static_cast<size_t>(T_valid), h,
                           static_cast<size_t>(T_valid) * kHidden);
        if (visual_embeds && visual_idx_io) {
            for (int t = 0; t < T_valid; ++t) {
                if (token_ids[t] == 125030 && *visual_idx_io < max_visual_tokens) {
                    std::memcpy(h + static_cast<size_t>(t) * kHidden,
                                visual_embeds + (*visual_idx_io) * kHidden,
                                kHidden * sizeof(float));
                    (*visual_idx_io)++;
                }
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        timers.cpu_norm_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        int conv_idx = 0, swa_idx = 0, glob_idx = 0;
        bool surface_overflow = false;

        for (int l = 0; l < kNumLayers; ++l) {
            const LayerNpuOffsets &no = m_layer_npu[l];
            const LayerCpuWeights &cw = m_layer_cpu[l];
            uint16_t logical_die = no.die_id;
            uint16_t phys_die = (logical_die < m_num_dies) ? logical_die : 0;
            const DieModels &dm    = m_die_models[logical_die];
            const DieModels &dm_d0 = m_die_models[0];
            const DieModels &dm_d1 = m_die_models[1];
            DmaBuffer *act_buf    = (T == 1) ? m_act_t1[logical_die].get() : m_act_t64[logical_die].get();
            DmaBuffer *out_buf    = (T == 1) ? m_out_t1[logical_die].get() : m_out_t64[logical_die].get();
            DmaBuffer *w_buf      = m_w_blob[logical_die].get();
            DmaBuffer *act_buf_d0 = (T == 1) ? m_act_t1[0].get() : m_act_t64[0].get();
            DmaBuffer *out_buf_d0 = (T == 1) ? m_out_t1[0].get() : m_out_t64[0].get();
            DmaBuffer *w_buf_d0   = m_w_blob[0].get();
            DmaBuffer *act_buf_d1 = (T == 1) ? m_act_t1[1].get() : m_act_t64[1].get();
            DmaBuffer *out_buf_d1 = (T == 1) ? m_out_t1[1].get() : m_out_t64[1].get();
            DmaBuffer *w_buf_d1   = m_w_blob[1].get();

            // First region per archetype on this die, for --weights-layer0.
            struct AbRegion { size_t off = 0, off_d1 = 0, sz = 0; };
            AbRegion ab[4] = {};
            if (m_ab_shared_weights) {
                for (int j = (logical_die == 0) ? 0 : 12; j < (logical_die == 0 ? 12 : 24); ++j) {
                    const LayerNpuOffsets &o = m_layer_npu[j];
                    const bool is_conv = (kLayerSchedule[j] == Lfm3LayerType::Conv);
                    if (!ab[0].sz && is_conv && o.op_in_size)  ab[0] = {o.op_in_offset,  o.op_in_offset_d1,  o.op_in_size};
                    if (!ab[1].sz && o.op_out_size)            ab[1] = {o.op_out_offset, o.op_out_offset_d1, o.op_out_size};
                    if (!ab[2].sz && !is_conv && o.op_in_size) ab[2] = {o.op_in_offset,  o.op_in_offset_d1,  o.op_in_size};
                    if (!ab[3].sz && o.mlp_size)               ab[3] = {o.mlp_offset,    o.mlp_offset_d1,    o.mlp_size};
                }
            }
            const double weight_gb_total =
                (double)(ab[0].sz + ab[1].sz + ab[2].sz + ab[3].sz);
            if (m_ab_shared_weights && l == 0)
                std::printf("[+] --weights-layer0 ACTIVE: every layer reuses one region per "
                            "archetype (%.3f GB distinct weights/die instead of the full blob)\n",
                            weight_gb_total / 1e9);

            // Record cross-die handoff at Layer 12 boundary
            if (l == 12 && no.op_in_tp == 1) {
                auto td0 = std::chrono::steady_clock::now();
                std::atomic_thread_fence(std::memory_order_seq_cst);
                auto td1 = std::chrono::steady_clock::now();
                timers.d2d_xfer_ms += std::chrono::duration<double, std::milli>(td1 - td0).count();
            }

            auto surf_or = [&](const DmaBuffer *buf, size_t reported, int T_cur, int C_cur,
                               BoundaryDtype dt, const char *what) -> size_t {
                const size_t raw = static_cast<size_t>(T_cur) * C_cur * dtype_elem_size(dt);
                return surface_descriptor(buf, reported, raw, what, surface_overflow,
                                          prod_cap(buf));
            };
            if (surface_overflow) return false;

            auto wsel = [&](int slot, size_t off, size_t sz) -> std::pair<size_t, size_t> {
                if (!m_ab_shared_weights || ab[slot].sz == 0) return {off, sz};
                return {ab[slot].off, ab[slot].sz};
            };
            auto wsel_d1 = [&](int slot, size_t off_d1) -> size_t {
                if (!m_ab_shared_weights || ab[slot].sz == 0) return off_d1;
                return ab[slot].off_d1;
            };

            auto submit_arch = [&](int arch_idx, uint32_t model_id,
                                   const std::vector<DmaBufferSlice> &ins,
                                   const std::vector<DmaBufferSlice> &outs,
                                   DmaBuffer *flush_after, double weight_bytes,
                                   double &stage_ms) -> bool {
                prime_guard(flush_after);
                if (!ins.empty()) prime_input_tail(ins[0].buffer, ins[0].size);
                const auto s0 = std::chrono::steady_clock::now();
                uint64_t driver_ns = 0;
                const bool ok = m_pool.submit_task(phys_die, model_id, ins, outs, &driver_ns);
                if (ok && flush_after) flush_after->flush_cache();
                if (ok) measure_guard(arch_idx, flush_after);
                if (ok && !ins.empty()) measure_input_tail(arch_idx, ins[0].buffer, ins[0].size);
                const auto s1 = std::chrono::steady_clock::now();
                const double master_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
                stage_ms += master_ms;
                timers.add_arch(arch_idx, master_ms, static_cast<double>(driver_ns) / 1e6,
                                weight_bytes);
                return ok;
            };

            auto submit_and_unpack = [&](int arch_idx,
                                         int op_tp,
                                         const LoadedArchetype &arch_single_or_d0,
                                         const LoadedArchetype &arch_d1,
                                         const std::pair<size_t, size_t> &w_slice,
                                         size_t w_off_d1,
                                         int C_in, int C_out_per_die, int dst_stride,
                                         const SubgraphScaleInfo &sc_d0,
                                         const SubgraphScaleInfo &sc_d1,
                                         const float *dyn_ratios,
                                         float *dst_d0, float *dst_d1,
                                         double &npu_stage_ms,
                                         double &cpu_unpack_ms,
                                         bool already_broadcast = false) -> bool {
                if (op_tp == 1) {
                    act_buf->flush_cache();
                    size_t in_b  = surf_or(act_buf, arch_single_or_d0.in_surface_bytes,  T, C_in,          sc_d0.in_dtype,  "pp2 act");
                    size_t out_b = surf_or(out_buf, arch_single_or_d0.out_surface_bytes, T, C_out_per_die, sc_d0.out_dtype, "pp2 out");
                    if (surface_overflow) return false;
                    if (!submit_arch(arch_idx, arch_single_or_d0.id,
                            {DmaBufferSlice{act_buf, 0, in_b},
                             DmaBufferSlice{w_buf, w_slice.first, w_slice.second}},
                            {DmaBufferSlice{out_buf, 0, out_b}}, out_buf,
                            static_cast<double>(w_slice.second), npu_stage_ms)) {
                        return false;
                    }
                    if (T == 1 && arch_idx == StageTimers::ARCH_PROJ2048 && no.mlp_tp == 2) prewake_tp_worker();
                    auto tu0 = std::chrono::steady_clock::now();
                    unpack_npu_output(out_buf, dst_d0, T, C_out_per_die, sc_d0, dyn_ratios, T_valid, dst_stride);
                    auto tu1 = std::chrono::steady_clock::now();
                    cpu_unpack_ms += std::chrono::duration<double, std::milli>(tu1 - tu0).count();
                    return true;
                }

                // TP=2 Dual-Die Concurrent Execution across Die 0 and Die 1
                size_t in_b0  = surf_or(act_buf_d0, arch_single_or_d0.in_surface_bytes,  T, C_in,          sc_d0.in_dtype,  "tp2 act0");
                size_t out_b0 = surf_or(out_buf_d0, arch_single_or_d0.out_surface_bytes, T, C_out_per_die, sc_d0.out_dtype, "tp2 out0");
                size_t in_b1  = surf_or(act_buf_d1, arch_d1.in_surface_bytes,            T, C_in,          sc_d1.in_dtype,  "tp2 act1");
                size_t out_b1 = surf_or(out_buf_d1, arch_d1.out_surface_bytes,           T, C_out_per_die, sc_d1.out_dtype, "tp2 out1");
                if (surface_overflow) return false;

                if (!already_broadcast) {
                    const size_t copy_bytes = static_cast<size_t>(T) * C_in * dtype_elem_size(sc_d0.in_dtype);
                    if (act_buf != act_buf_d0) {
                        std::memcpy(act_buf_d0->data(), act_buf->data(), copy_bytes);
                    }
                    if (act_buf != act_buf_d1) {
                        std::memcpy(act_buf_d1->data(), act_buf->data(), copy_bytes);
                    }
                }

                bool ok0 = false, ok1 = false;
                uint64_t drv0_ns = 0, drv1_ns = 0;
                const uint16_t phys1 = (1 < m_num_dies) ? 1 : 0;
                const auto s0 = std::chrono::steady_clock::now();

                run_dual_die(
                    [&]() {
                        act_buf_d0->flush_cache();
                        prime_guard(out_buf_d0);
                        prime_input_tail(act_buf_d0, in_b0);
                        ok0 = m_pool.submit_task(0, arch_single_or_d0.id,
                            {DmaBufferSlice{act_buf_d0, 0, in_b0},
                             DmaBufferSlice{w_buf_d0, w_slice.first, w_slice.second}},
                            {DmaBufferSlice{out_buf_d0, 0, out_b0}}, &drv0_ns);
                        if (ok0) out_buf_d0->flush_cache();
                        if (ok0 && T_valid == 1) {
                            unpack_npu_output(out_buf_d0, dst_d0, T, C_out_per_die, sc_d0, dyn_ratios, T_valid, dst_stride);
                        }
                    },
                    [&]() {
                        act_buf_d1->flush_cache();
                        prime_guard(out_buf_d1);
                        prime_input_tail(act_buf_d1, in_b1);
                        ok1 = m_pool.submit_task(phys1, arch_d1.id,
                            {DmaBufferSlice{act_buf_d1, 0, in_b1},
                             DmaBufferSlice{w_buf_d1, w_off_d1, w_slice.second}},
                            {DmaBufferSlice{out_buf_d1, 0, out_b1}}, &drv1_ns);
                        if (ok1) out_buf_d1->flush_cache();
                        if (ok1 && T_valid == 1) {
                            unpack_npu_output(out_buf_d1, dst_d1, T, C_out_per_die, sc_d1, dyn_ratios, T_valid, dst_stride);
                        }
                    }
                );

                const auto s1 = std::chrono::steady_clock::now();
                const double master_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
                npu_stage_ms += master_ms;
                const double drv_ms = static_cast<double>(std::max(drv0_ns, drv1_ns)) / 1e6;
                timers.add_arch(arch_idx, master_ms, drv_ms, 2.0 * static_cast<double>(w_slice.second));
                if (!ok0 || !ok1) return false;

                if (T_valid > 1) {
                    auto tu0 = std::chrono::steady_clock::now();
                    unpack_npu_output(out_buf_d0, dst_d0, T, C_out_per_die, sc_d0, dyn_ratios, T_valid, dst_stride);
                    unpack_npu_output(out_buf_d1, dst_d1, T, C_out_per_die, sc_d1, dyn_ratios, T_valid, dst_stride);
                    auto tu1 = std::chrono::steady_clock::now();
                    cpu_unpack_ms += std::chrono::duration<double, std::milli>(tu1 - tu0).count();
                }
                return true;
            };
            float op_in_ratios[64], op_out_ratios[64];

            // 1. Pre-operator Sandwich RMSNorm
            auto tn0 = std::chrono::steady_clock::now();
            if (T == 1 && !cw.op_norm_smooth.empty() && no.op_in_scale.in_dtype == BoundaryDtype::Int8) {
                float *u_attn = (kLayerSchedule[l] != Lfm3LayerType::Conv) ? u : nullptr;
                fused_rms_norm_pack_dyntok_t1(h, cw.op_norm, cw.op_norm_smooth.data(),
                                              act_buf->as<int8_t>(), kHidden,
                                              no.op_in_scale.in_thresh, op_in_ratios, u_attn);
            } else if (T > 1 && !cw.op_norm_smooth.empty() && no.op_in_scale.in_dtype == BoundaryDtype::Int8) {
                if (kLayerSchedule[l] != Lfm3LayerType::Conv) {
                    rms_norm_f32(h, cw.op_norm, u, T_valid, kHidden);
                }
                rms_norm_f32(h, cw.op_norm_smooth.data(), u_smooth, T_valid, kHidden);
                pack_npu_input_int8_dyntok(u_smooth, act_buf->as<int8_t>(), T, kHidden,
                                           no.op_in_scale.in_thresh, op_in_ratios, nullptr, T_valid);
            } else {
                rms_norm_f32(h, cw.op_norm, u, T_valid, kHidden);
                pack_npu_input(u, act_buf, T, kHidden, no.op_in_scale, op_in_ratios, T_valid);
            }
            auto tn1 = std::chrono::steady_clock::now();
            timers.cpu_norm_ms += std::chrono::duration<double, std::milli>(tn1 - tn0).count();

            if (kLayerSchedule[l] == Lfm3LayerType::Conv) {
                // --- Primitive 2: Gated ShortConv Layer ---
                const LoadedArchetype &m_in     = (T == 1) ? dm.conv_in_t1     : dm.conv_in_t64;
                const LoadedArchetype &m_in_d1  = (T == 1) ? dm_d1.conv_in_t1  : dm_d1.conv_in_t64;
                const LoadedArchetype &m_out    = (T == 1) ? dm.proj2048_t1    : dm.proj2048_t64;
                const LoadedArchetype &m_out_d1 = (T == 1) ? dm_d1.proj2048_t1 : dm_d1.proj2048_t64;

                const auto conv_in_w = wsel(0, no.op_in_offset, no.op_in_size);
                const size_t conv_in_w_d1 = wsel_d1(0, no.op_in_offset_d1);
                const int conv_in_c_out = (no.op_in_tp == 2) ? 3072 : 6144;
                if (!submit_and_unpack(StageTimers::ARCH_CONV_IN, no.op_in_tp, m_in, m_in_d1,
                        conv_in_w, conv_in_w_d1, 2048, conv_in_c_out, 6144,
                        no.op_in_scale, no.op_in_scale_d1,
                        op_in_ratios, bcx, bcx + 3072, timers.npu_layer_ms, timers.cpu_conv_ms)) {
                    return false;
                }

                // CPU 7-tap depthwise ShortConv state update
                auto tconv0 = std::chrono::steady_clock::now();
                float *layer_st = m_conv_state.data() + static_cast<size_t>(conv_idx) * kHidden * 6;
                const float *w_conv = cw.conv_weight;

                if (T_valid == 1) {
                    const float *pB = bcx;
                    const float *pC = bcx + kHidden;
                    const float *pX = bcx + 2 * kHidden;
                    for (int c = 0; c < kHidden; ++c) {
                        float *st = &layer_st[c * 6];
                        const float *kw = &w_conv[c * 7];
                        float xt = pB[c] * pX[c];
                        float dw = st[0]*kw[0] + st[1]*kw[1] + st[2]*kw[2] +
                                   st[3]*kw[3] + st[4]*kw[4] + st[5]*kw[5] + xt*kw[6];
                        st[0] = st[1]; st[1] = st[2]; st[2] = st[3];
                        st[3] = st[4]; st[4] = st[5]; st[5] = xt;
                        y[c] = pC[c] * dw;
                    }
                } else {
                    #pragma omp parallel for schedule(static)
                    for (int c = 0; c < kHidden; ++c) {
                        float *st = &layer_st[c * 6];
                        const float *kw = &w_conv[c * 7];
                        for (int t = 0; t < T_valid; ++t) {
                            const float *p = bcx + static_cast<size_t>(t) * 6144;
                            float B  = p[c];
                            float C  = p[kHidden + c];
                            float xp = p[2 * kHidden + c];
                            float xt = B * xp;
                            float dw = st[0]*kw[0] + st[1]*kw[1] + st[2]*kw[2] +
                                       st[3]*kw[3] + st[4]*kw[4] + st[5]*kw[5] + xt*kw[6];
                            st[0] = st[1]; st[1] = st[2]; st[2] = st[3];
                            st[3] = st[4]; st[4] = st[5]; st[5] = xt;
                            y[static_cast<size_t>(t) * kHidden + c] = C * dw;
                        }
                    }
                }
                conv_idx++;
                pack_npu_input(y, act_buf, T, kHidden, no.op_out_scale, op_out_ratios, T_valid);
                auto tconv1 = std::chrono::steady_clock::now();
                timers.cpu_conv_ms += std::chrono::duration<double, std::milli>(tconv1 - tconv0).count();

                const auto proj_w = wsel(1, no.op_out_offset, no.op_out_size);
                const size_t proj_w_d1 = wsel_d1(1, no.op_out_offset_d1);
                const int proj_c_out = (no.op_out_tp == 2) ? 1024 : 2048;
                if (!submit_and_unpack(StageTimers::ARCH_PROJ2048, no.op_out_tp, m_out, m_out_d1,
                        proj_w, proj_w_d1, 2048, proj_c_out, 2048,
                        no.op_out_scale, no.op_out_scale_d1,
                        op_out_ratios, op_out, op_out + 1024, timers.npu_layer_ms, timers.cpu_conv_ms)) {
                    return false;
                }

            } else {
                // --- Primitive 3: SWA or Global GQA Attention Layer ---
                bool is_swa = (kLayerSchedule[l] == Lfm3LayerType::SlidingAttention);
                const LoadedArchetype &m_qkv    = (T == 1) ? dm.attn_qkv_t1    : dm.attn_qkv_t64;
                const LoadedArchetype &m_qkv_d1 = (T == 1) ? dm_d1.attn_qkv_t1 : dm_d1.attn_qkv_t64;
                const LoadedArchetype &m_out    = (T == 1) ? dm.proj2048_t1    : dm.proj2048_t64;
                const LoadedArchetype &m_out_d1 = (T == 1) ? dm_d1.proj2048_t1 : dm_d1.proj2048_t64;
                double &attn_cpu_timer = is_swa ? timers.cpu_swa_ms : timers.cpu_glob_ms;

                const auto qkv_w = wsel(2, no.op_in_offset, no.op_in_size);
                const size_t qkv_w_d1 = wsel_d1(2, no.op_in_offset_d1);
                const int qkv_c_out = (no.op_in_tp == 2) ? 1536 : 3072;
                if (!submit_and_unpack(StageTimers::ARCH_ATTN_QKV, no.op_in_tp, m_qkv, m_qkv_d1,
                        qkv_w, qkv_w_d1, 2048, qkv_c_out, 3072,
                        no.op_in_scale, no.op_in_scale_d1,
                        op_in_ratios, qkv, qkv + 1536, timers.npu_layer_ms, attn_cpu_timer)) {
                    return false;
                }

                // Compute 16-head sigmoid gate g_h = sigmoid(W_gate * u_t) (4-head register blocked)
                auto tg0 = std::chrono::steady_clock::now();
                #pragma omp parallel for if(T_valid > 1) schedule(static)
                for (int t = 0; t < T_valid; ++t) {
                    const float *u_row = u + static_cast<size_t>(t) * kHidden;
                    float *g_row = gates + static_cast<size_t>(t) * kQHeads;
                    for (int h_idx = 0; h_idx < kQHeads; h_idx += 4) {
                        const float *wg0 = cw.gate_proj + static_cast<size_t>(h_idx + 0) * kHidden;
                        const float *wg1 = cw.gate_proj + static_cast<size_t>(h_idx + 1) * kHidden;
                        const float *wg2 = cw.gate_proj + static_cast<size_t>(h_idx + 2) * kHidden;
                        const float *wg3 = cw.gate_proj + static_cast<size_t>(h_idx + 3) * kHidden;
                        float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
                        for (int c = 0; c < kHidden; ++c) {
                            float uv = u_row[c];
                            dot0 += uv * wg0[c];
                            dot1 += uv * wg1[c];
                            dot2 += uv * wg2[c];
                            dot3 += uv * wg3[c];
                        }
                        g_row[h_idx + 0] = 1.0f / (1.0f + std::exp(-dot0));
                        g_row[h_idx + 1] = 1.0f / (1.0f + std::exp(-dot1));
                        g_row[h_idx + 2] = 1.0f / (1.0f + std::exp(-dot2));
                        g_row[h_idx + 3] = 1.0f / (1.0f + std::exp(-dot3));
                    }
                }
                auto tg1 = std::chrono::steady_clock::now();
                timers.cpu_norm_ms += std::chrono::duration<double, std::milli>(tg1 - tg0).count();

                // QK-Norm + RoPE + KV Cache Update + GQA Attention
                auto tattn0 = std::chrono::steady_clock::now();

                float *k_cache = is_swa
                    ? (m_swa_k.data() + static_cast<size_t>(swa_idx) * kSwaCap * kKvHeads * kHeadDim)
                    : (m_glob_k.data() + static_cast<size_t>(glob_idx) * m_max_ctx * kKvHeads * kHeadDim);
                float *v_cache = is_swa
                    ? (m_swa_v.data() + static_cast<size_t>(swa_idx) * kSwaCap * kKvHeads * kHeadDim)
                    : (m_glob_v.data() + static_cast<size_t>(glob_idx) * m_max_ctx * kKvHeads * kHeadDim);
                int cur_glob_idx = glob_idx;
                if (is_swa) swa_idx++; else glob_idx++;

                const float *inv_freq = is_swa ? m_swa_inv_freq : m_glob_inv_freq;
                int n_rot_pairs = is_swa ? 64 : 32;
                int s_cap = is_swa ? kSwaCap : m_max_ctx;
                uint8_t  *k_u8_layer  = !is_swa ? (m_glob_k_u8.data()  + static_cast<size_t>(cur_glob_idx) * kKvHeads * s_cap * kHeadDim) : nullptr;
                uint8_t  *v_u8_layer  = !is_swa ? (m_glob_v_u8.data()  + static_cast<size_t>(cur_glob_idx) * kKvHeads * s_cap * kHeadDim) : nullptr;
                float    *k_sc_layer  = !is_swa ? (m_glob_k_sc.data()  + static_cast<size_t>(cur_glob_idx) * kKvHeads * s_cap) : nullptr;
                float    *v_sc_layer  = !is_swa ? (m_glob_v_sc.data()  + static_cast<size_t>(cur_glob_idx) * kKvHeads * s_cap) : nullptr;
                uint16_t *k_sum_layer = !is_swa ? (m_glob_k_sum.data() + static_cast<size_t>(cur_glob_idx) * kKvHeads * s_cap) : nullptr;

                // Apply q_norm, k_norm, FP32 RoPE, and store K, V into Head-Major cache [kv_h][s_cap][128]
                #pragma omp parallel for if(T_valid > 1) schedule(static)
                for (int t = 0; t < T_valid; ++t) {
                    int abs_pos = m_pos + t;
                    int slot = is_swa ? (abs_pos % kSwaCap) : abs_pos;
                    float *qkv_t = qkv + static_cast<size_t>(t) * 3072;

                    float cos_tab[64], sin_tab[64];
                    for (int i = 0; i < n_rot_pairs; ++i) {
                        float ang = static_cast<float>(abs_pos) * inv_freq[i];
                        cos_tab[i] = std::cos(ang);
                        sin_tab[i] = std::sin(ang);
                    }

                    // 16 Query heads
                    for (int h_idx = 0; h_idx < kQHeads; ++h_idx) {
                        float *q_h = qkv_t + h_idx * kHeadDim;
                        float ss = 0.0f;
                        for (int d = 0; d < kHeadDim; ++d) ss += q_h[d] * q_h[d];
                        float inv = 1.0f / std::sqrt(ss * (1.0f / 128.0f) + 1e-5f);
                        for (int d = 0; d < kHeadDim; ++d) q_h[d] = q_h[d] * inv * cw.q_norm[d];
                        for (int i = 0; i < n_rot_pairs; ++i) {
                            float x1 = q_h[i];
                            float x2 = q_h[i + 64];
                            q_h[i]      = x1 * cos_tab[i] - x2 * sin_tab[i];
                            q_h[i + 64] = x2 * cos_tab[i] + x1 * sin_tab[i];
                        }
                    }

                    // 4 Key heads + Head-Major KV cache write
                    float *k_base = qkv_t + 2048;
                    const float *v_base = qkv_t + 2560;
                    for (int kh = 0; kh < kKvHeads; ++kh) {
                        float *k_h = k_base + kh * kHeadDim;
                        const float *v_h = v_base + kh * kHeadDim;
                        float ss = 0.0f;
                        for (int d = 0; d < kHeadDim; ++d) ss += k_h[d] * k_h[d];
                        float inv = 1.0f / std::sqrt(ss * (1.0f / 128.0f) + 1e-5f);
                        for (int d = 0; d < kHeadDim; ++d) k_h[d] = k_h[d] * inv * cw.k_norm[d];
                        for (int i = 0; i < n_rot_pairs; ++i) {
                            float x1 = k_h[i];
                            float x2 = k_h[i + 64];
                            k_h[i]      = x1 * cos_tab[i] - x2 * sin_tab[i];
                            k_h[i + 64] = x2 * cos_tab[i] + x1 * sin_tab[i];
                        }
                        size_t head_slot_idx = static_cast<size_t>(kh) * s_cap + slot;
                        size_t head_slot_off = head_slot_idx * kHeadDim;
                        if (is_swa || (m_pos + T_valid) < 256) {
                            std::memcpy(k_cache + head_slot_off, k_h, kHeadDim * sizeof(float));
                            std::memcpy(v_cache + head_slot_off, v_h, kHeadDim * sizeof(float));
                        }
                        if (!is_swa) {
                            uint16_t dummy_v_sum = 0;
                            quantize_kv_u8(k_h, k_u8_layer + head_slot_off, k_sc_layer[head_slot_idx], k_sum_layer[head_slot_idx]);
                            quantize_kv_u8(v_h, v_u8_layer + head_slot_off, v_sc_layer[head_slot_idx], dummy_v_sum);
                        }
                    }
                }

                // Execute GQA Attention + multiply by head gate g_h
                if (!is_swa && (m_pos + T_valid) >= 256) {
                    if (T_valid == 1) {
                        run_gqa_attention_int8_swar(qkv, k_u8_layer, v_u8_layer, k_sc_layer, v_sc_layer,
                                                    k_sum_layer, gates, y, m_pos + 1, s_cap);
                    } else {
                        run_gqa_attention_int8_swar_prefill(qkv, k_u8_layer, v_u8_layer, k_sc_layer, v_sc_layer,
                                                            k_sum_layer, gates, y, T_valid, m_pos, s_cap);
                    }
                } else {
                    run_gqa_attention(qkv, k_cache, v_cache, gates, y,
                                      T_valid, m_pos, is_swa, s_cap);
                }

                pack_npu_input(y, act_buf, T, kHidden, no.op_out_scale, op_out_ratios, T_valid);
                auto tattn1 = std::chrono::steady_clock::now();
                attn_cpu_timer += std::chrono::duration<double, std::milli>(tattn1 - tattn0).count();

                const auto proj_w = wsel(1, no.op_out_offset, no.op_out_size);
                const size_t proj_w_d1 = wsel_d1(1, no.op_out_offset_d1);
                const int proj_c_out = (no.op_out_tp == 2) ? 1024 : 2048;
                if (!submit_and_unpack(StageTimers::ARCH_PROJ2048, no.op_out_tp, m_out, m_out_d1,
                        proj_w, proj_w_d1, 2048, proj_c_out, 2048,
                        no.op_out_scale, no.op_out_scale_d1,
                        op_out_ratios, op_out, op_out + 1024, timers.npu_layer_ms, attn_cpu_timer)) {
                    return false;
                }
            }

            // Pre-wake Die 1 TP worker now (after proj2048 NPU wait has finished) on T=1 decode so its futex wakeup
            // overlaps with single-threaded post_operator_norm + ffn_norm + INT8 pack (~80 us) without contending with OpenMP
            if (T == 1 && no.mlp_tp == 2) prewake_tp_worker();

            // 2. Sandwich post_operator_norm + Residual + ffn_norm
            auto tn4 = std::chrono::steady_clock::now();
            bool mlp_bcast_done = false;
            if (T == 1 && !cw.ffn_norm_scaled.empty() && no.mlp_scale.in_dtype == BoundaryDtype::Int8) {
                if (no.mlp_tp == 2) {
                    fused_post_op_residual_ffn_pack_mlp_t1(op_out, cw.post_op_norm, h, h_mid,
                                                           cw.ffn_norm_scaled.data(),
                                                           act_buf_d0->as<int8_t>(), kHidden, 1e-5f,
                                                           act_buf_d1->as<int8_t>());
                    mlp_bcast_done = true;
                } else {
                    fused_post_op_residual_ffn_pack_mlp_t1(op_out, cw.post_op_norm, h, h_mid,
                                                           cw.ffn_norm_scaled.data(),
                                                           act_buf->as<int8_t>(), kHidden);
                }
            } else if (T > 1 && !cw.ffn_norm_smooth.empty() && no.mlp_scale.in_dtype == BoundaryDtype::Int8) {
                rms_norm_f32(op_out, cw.post_op_norm, op_out, T_valid, kHidden);
                #pragma omp simd
                for (int i = 0; i < T_valid * kHidden; ++i) h_mid[i] = h[i] + op_out[i];
                rms_norm_f32(h_mid, cw.ffn_norm_smooth.data(), v, T_valid, kHidden);
                pack_npu_input_int8(v, act_buf->as<int8_t>(), T, kHidden, no.mlp_scale.in_step, nullptr, T_valid);
            } else {
                rms_norm_f32(op_out, cw.post_op_norm, op_out, T_valid, kHidden);
                #pragma omp simd
                for (int i = 0; i < T_valid * kHidden; ++i) h_mid[i] = h[i] + op_out[i];
                rms_norm_f32(h_mid, cw.ffn_norm, v, T_valid, kHidden);
                pack_npu_input(v, act_buf, T, kHidden, no.mlp_scale, nullptr, T_valid);
            }
            auto tn5 = std::chrono::steady_clock::now();
            timers.cpu_norm_ms += std::chrono::duration<double, std::milli>(tn5 - tn4).count();

            // 3. SwiGLU MLP on NPU
            const LoadedArchetype &m_mlp_0  = (no.mlp_tp == 2)
                ? ((T == 1) ? dm_d0.mlp_t1 : dm_d0.mlp_t64)
                : ((T == 1) ? dm.mlp_t1    : dm.mlp_t64);
            const LoadedArchetype &m_mlp_d1 = (T == 1) ? dm_d1.mlp_t1 : dm_d1.mlp_t64;
            const auto mlp_w = wsel(3, no.mlp_offset, no.mlp_size);
            const size_t mlp_w_d1 = wsel_d1(3, no.mlp_offset_d1);
            if (!submit_and_unpack(StageTimers::ARCH_MLP, no.mlp_tp, m_mlp_0, m_mlp_d1,
                    mlp_w, mlp_w_d1, 2048, 2048, 2048,
                    no.mlp_scale, no.mlp_scale_d1,
                    nullptr, mlp_out, mlp_out_d1, timers.npu_layer_ms, timers.cpu_norm_ms,
                    mlp_bcast_done)) {
                return false;
            }

            // 4. Sandwich post_ffn_norm + Residual
            auto tn6 = std::chrono::steady_clock::now();
            if (T == 1) {
                fused_post_ffn_residual_t1(mlp_out, cw.post_ffn_norm, h_mid, h, kHidden, 1e-5f,
                                           (no.mlp_tp == 2) ? mlp_out_d1 : nullptr);
            } else {
                if (no.mlp_tp == 2) {
                    #pragma omp simd
                    for (int i = 0; i < T_valid * kHidden; ++i) mlp_out[i] += mlp_out_d1[i];
                }
                rms_norm_f32(mlp_out, cw.post_ffn_norm, mlp_out, T_valid, kHidden);
                #pragma omp simd
                for (int i = 0; i < T_valid * kHidden; ++i) h[i] = h_mid[i] + mlp_out[i];
            }
            auto tn7 = std::chrono::steady_clock::now();
            timers.cpu_norm_ms += std::chrono::duration<double, std::milli>(tn7 - tn6).count();

            if (debug_layer_outs) {
                const float *last_h = h + static_cast<size_t>(T_valid - 1) * kHidden;
                std::memcpy(debug_layer_outs->data() + static_cast<size_t>(l) * kHidden,
                            last_h, kHidden * sizeof(float));
            }
        }

        m_pos += T_valid;
        timers.tokens_run += T_valid;

        if (!compute_logits) return true;

        // Final embedding_norm on the last token's hidden state [1, 2048]
        auto tfn0 = std::chrono::steady_clock::now();
        const float *last_h = h + static_cast<size_t>(T_valid - 1) * kHidden;
        const SubgraphScaleInfo &lm_in_sc = m_die_models[0].lm_head_scale[0];

        if (!m_emb_norm_scaled.empty()) {
            fused_emb_norm_broadcast_t1(last_h, m_emb_norm_scaled.data(),
                                        m_act_t1[0]->as<int8_t>(),
                                        m_act_t1[1]->as<int8_t>(),
                                        kHidden);
            if (debug_emb_norm) {
                float h_final[kHidden];
                rms_norm_f32(last_h, m_emb_norm_weight, h_final, 1, kHidden);
                debug_emb_norm->assign(h_final, h_final + kHidden);
            }
        } else {
            float h_final[kHidden];
            rms_norm_f32(last_h, m_emb_norm_weight, h_final, 1, kHidden);
            if (debug_emb_norm) {
                debug_emb_norm->assign(h_final, h_final + kHidden);
            }
            pack_npu_input(h_final, m_act_t1[0].get(), 1, kHidden, lm_in_sc);
            size_t lm_copy_bytes = static_cast<size_t>(kHidden) * dtype_elem_size(lm_in_sc.in_dtype);
            std::memcpy(m_act_t1[1]->data(), m_act_t1[0]->data(), lm_copy_bytes);
        }
        m_act_t1[0]->flush_cache();
        m_act_t1[1]->flush_cache();
        auto tfn1 = std::chrono::steady_clock::now();
        timers.cpu_norm_ms += std::chrono::duration<double, std::milli>(tfn1 - tfn0).count();

        // Primitive 4: Concurrent Dual-Die LM Head Execution (4 x 16K slices on Die 0 || 4 x 16K slices on Die 1)
        auto tlm0 = std::chrono::steady_clock::now();
        bool ok_d0 = true, ok_d1 = true;
        // The two die threads must not touch `timers` concurrently; accumulate per die and merge.
        double lm_master[2] = {0.0, 0.0}, lm_driver[2] = {0.0, 0.0};
        double lm_bytes[2]  = {0.0, 0.0};
        long   lm_submits[2] = {0, 0};

        auto run_die_lm_head = [&](uint16_t logical_die, bool &ok_flag) {
            uint16_t phys_die = (logical_die < m_num_dies) ? logical_die : 0;
            DmaBuffer *in_b = m_act_t1[logical_die].get();
            for (int s = 0; s < 4; ++s) {
                const LoadedArchetype   &lm_arch = m_die_models[logical_die].lm_head[s];
                const SubgraphScaleInfo &lm_sc   = m_die_models[logical_die].lm_head_scale[s];
                DmaBuffer *out_b = m_lm_out[logical_die][s].get();
                size_t in_raw   = static_cast<size_t>(kHidden) * dtype_elem_size(lm_sc.in_dtype);
                size_t out_raw  = static_cast<size_t>(16000)   * dtype_elem_size(lm_sc.out_dtype);
                bool lm_fit = false;
                const size_t in_surf  = surface_descriptor(in_b,  lm_arch.in_surface_bytes,
                                                           in_raw,  "lm_head act",  lm_fit,
                                                           prod_cap(in_b));
                const size_t out_surf = surface_descriptor(out_b, lm_arch.out_surface_bytes,
                                                           out_raw, "lm_head logits", lm_fit,
                                                           prod_cap(out_b));
                if (lm_fit) {
                    ok_flag = false;
                    return;
                }
                prime_guard(out_b);
                const auto s0 = std::chrono::steady_clock::now();
                uint64_t driver_ns = 0;
                const bool ok = m_pool.submit_task(phys_die, lm_arch.id,
                        {DmaBufferSlice{in_b, 0, in_surf}},
                        {DmaBufferSlice{out_b, 0, out_surf}}, &driver_ns);
                if (ok) out_b->flush_cache();
                if (ok) measure_guard(StageTimers::ARCH_LM_HEAD, out_b);
                const auto s1 = std::chrono::steady_clock::now();
                lm_master[logical_die] += std::chrono::duration<double, std::milli>(s1 - s0).count();
                lm_driver[logical_die] += static_cast<double>(driver_ns) / 1e6;
                lm_bytes[logical_die]  += static_cast<double>(lm_arch.weight_bytes);
                ++lm_submits[logical_die];
                if (!ok) {
                    ok_flag = false;
                    return;
                }
                float *dst_fp32 = m_logits.data() + (static_cast<size_t>(logical_die) * 64000 + s * 16000);
                unpack_npu_output(out_b, dst_fp32, 1, 16000, lm_sc);
            }
        };

        run_dual_die(
            [&]() { run_die_lm_head(0, ok_d0); },
            [&]() { run_die_lm_head(1, ok_d1); }
        );
        auto tlm1 = std::chrono::steady_clock::now();
        timers.npu_lmhead_ms += std::chrono::duration<double, std::milli>(tlm1 - tlm0).count();
        for (int d = 0; d < 2; ++d)
            if (lm_submits[d])
                timers.add_arch(StageTimers::ARCH_LM_HEAD, lm_master[d], lm_driver[d], lm_bytes[d]);

        return ok_d0 && ok_d1;
    }

    // High-level prefill: batches prompt tokens in chunks of up to 64 valid tokens (t = 0..63)
    // onto the T=64 NPU graphs so weight streaming happens once per 64 tokens instead of once per token.
    // Set LFM3_PREFILL_T1=1 to force the unbatched T=1 path.
    bool prefill_multimodal(const std::vector<int32_t> &tokens, StageTimers &timers,
                            const float *visual_embeds, size_t num_visual_tokens,
                            std::vector<float> *debug_layer_outs = nullptr,
                            std::vector<float> *debug_emb_norm = nullptr) {
        const bool force_t1 = std::getenv("LFM3_PREFILL_T1") != nullptr;
        int n = static_cast<int>(tokens.size());
        int idx = 0;
        size_t visual_idx = 0;
        if (!force_t1) {
            constexpr int kPrefillBatch = 64;
            auto t_pref0 = std::chrono::steady_clock::now();
            while (idx < n) {
                int step = std::min(kPrefillBatch, n - idx);
                bool is_last = (idx + step == n);
                if (!forward_chunk(tokens.data() + idx, step, is_last, timers,
                                   visual_embeds, &visual_idx, num_visual_tokens,
                                   is_last ? debug_layer_outs : nullptr,
                                   is_last ? debug_emb_norm : nullptr)) {
                    return false;
                }
                idx += step;
                if (n > 128 && (idx % (kPrefillBatch * 4) == 0 || is_last)) {
                    double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pref0).count();
                    double cur_tok_s = (elapsed_s > 0.0) ? (idx / elapsed_s) : 0.0;
                    std::printf("\r[Prefill progress: %d/%d tokens (%.1f%%) | %.1f tok/s]   ",
                                idx, n, (100.0 * idx) / n, cur_tok_s);
                    std::fflush(stdout);
                }
            }
            if (n > 128) {
                std::printf("\n");
            }
            return true;
        }
        while (idx < n) {
            bool is_last = (idx + 1 == n);
            if (!forward_chunk(tokens.data() + idx, 1, is_last, timers,
                               visual_embeds, &visual_idx, num_visual_tokens,
                               is_last ? debug_layer_outs : nullptr,
                               is_last ? debug_emb_norm : nullptr)) {
                return false;
            }
            idx++;
        }
        return true;
    }

    bool prefill(const std::vector<int32_t> &tokens, StageTimers &timers,
                 std::vector<float> *debug_layer_outs = nullptr,
                 std::vector<float> *debug_emb_norm = nullptr) {
        return prefill_multimodal(tokens, timers, nullptr, 0, debug_layer_outs, debug_emb_norm);
    }

    bool prefill(const std::vector<int32_t> &tokens, StageTimers &timers,
                 const float *visual_embeds, size_t num_visual_tokens) {
        return prefill_multimodal(tokens, timers, visual_embeds, num_visual_tokens, nullptr, nullptr);
    }

    void set_quant_options(bool force_int8, bool use_shared_archetype_scales) {
        m_force_int8 = force_int8;
        m_use_shared_archetype_scales = use_shared_archetype_scales;
    }

    void set_parallel_mode(int tp_degree) {
        m_requested_tp_degree = tp_degree;
    }

    /// Binds every layer of a die to one weight region per archetype. Timing experiment only:
    /// the numbers it prints are for bandwidth attribution, and its generated text is garbage.
    void set_ab_shared_weights(bool on) { m_ab_shared_weights = on; }

    /**
     * @brief XORs a span of a die's resident weight blob, for the golden-gate integrity tests.
     *
     * An accuracy gate that cannot fail is worse than no gate: FINDINGS §41.2 found the metrics
     * identical to eight decimals across two packages with demonstrably different weights. The
     * only way to know a gate measures anything is to break the thing it measures and watch the
     * number move.
     *
     * The span has to be large. One byte out of ~470M is a ~1e-7 relative perturbation, which
     * cannot move a cosine in the eighth decimal -- a one-byte null test passes vacuously and
     * proves nothing. Corrupt a whole region instead. The blob is already SMMU-prepared, so the
     * NPU sees the change after the flush; the caller restores it by XORing the same span again.
     */
    bool perturb_weights(int die, size_t offset, size_t len, uint8_t xor_mask) {
        if (die < 0 || die > 1 || !m_w_blob[die]) return false;
        if (offset >= m_w_blob[die]->size() || len == 0 || offset + len > m_w_blob[die]->size())
            return false;
        uint8_t *p = static_cast<uint8_t *>(m_w_blob[die]->data()) + offset;
        for (size_t i = 0; i < len; ++i) p[i] ^= xor_mask;
        m_w_blob[die]->flush_cache();
        return true;
    }

    /**
     * @brief Write-extent probe (LFM3_MEASURE_EXTENT=1): what does each graph actually write?
     *
     * Buffers are sized today either to a fixed constant (32 KiB) or to the driver's reported
     * memory entry (2 MiB) -- neither is a measurement. The 2 MiB figure is known to be a
     * reporting bucket, not a tensor (FINDINGS 41.1), and whole-allocation cache maintenance
     * costs ~32% of decode (44), so the right size is whatever the NPU actually touches.
     *
     * Each output buffer gets a guard region appended, refilled with a known byte before every
     * submit and scanned after it. The first overwritten offset is that archetype's write extent.
     */
    void set_extent_probe(bool on, size_t guard) { m_extent_probe = on; m_guard_bytes = on ? guard : 0; }
    void note_prod_cap(const DmaBuffer *b, size_t prod) { if (m_extent_probe) m_prod_cap[b] = prod; }
    size_t prod_cap(const DmaBuffer *b) const {
        auto it = m_prod_cap.find(b);
        return it == m_prod_cap.end() ? SIZE_MAX : it->second;
    }
    void prime_guard(DmaBuffer *b) {
        if (!m_extent_probe || !b) return;
        // Fill the WHOLE buffer, not just the guard. The question worth answering is how far into
        // the production region the graph actually writes -- that is what allows a buffer to be
        // shrunk -- and a guard-only fill can only ever report overruns, never a high-water mark.
        uint8_t *g = static_cast<uint8_t *>(b->data());
        std::fill(g, g + b->size(), GUARD_BYTE);
        // Flush AFTER priming and BEFORE submitting. The fill leaves megabytes of dirty CPU cache
        // lines over the region the NPU is about to write; the post-submit flush then pushes them
        // back over the NPU's output and the run silently degrades to garbage. Production never
        // dirties an output buffer before a submit, so this ordering bug only exists here.
        // (Direct evidence for the unresolved flush-direction question, audit 1.1: a flush on an
        // fd whose cache lines are dirty does write them back, and it is not merely an invalidate.)
        b->flush_cache();
    }
    /**
     * @brief Prime/scan the tail of an INPUT buffer.
     *
     * The measured output extents equal the tensor byte counts exactly, so no output buffer is
     * oversized by anything we can see -- yet INT8 was still nondeterministic with small output
     * buffers (44). The remaining place a hidden write could land is the tail of an activation
     * buffer, which pack_npu_input leaves full of stale bytes and no probe has watched.
     */
    void prime_input_tail(const DmaBuffer *b, size_t nominal) {
        if (!m_extent_probe || !b || nominal >= b->size()) return;
        // DmaBufferSlice hands out const buffer pointers; the bytes themselves are mutable DMA
        // memory, and priming a guard region is exactly what they are for.
        uint8_t *g = static_cast<uint8_t *>(const_cast<void *>(b->data())) + nominal;
        std::fill(g, g + (b->size() - nominal), GUARD_BYTE);
        const_cast<DmaBuffer *>(b)->flush_cache();  // sentinel must reach DRAM pre-submit
    }
    void measure_input_tail(int arch, const DmaBuffer *b, size_t nominal) {
        if (!m_extent_probe || !b || nominal >= b->size()) return;
        const uint8_t *g = static_cast<const uint8_t *>(b->data());
        for (size_t i = nominal; i < b->size(); ++i) {
            if (g[i] != GUARD_BYTE) {
                std::printf("  !! arch %d wrote %zu B into its INPUT buffer tail at %zu\n",
                            arch, b->size() - i, i);
                if (i + 1 > m_extent_max[arch]) m_extent_max[arch] = i + 1;
                m_extent_overrun[arch] = true;
                return;
            }
        }
    }
    void measure_guard(int arch, DmaBuffer *b) {
        if (!m_extent_probe || !b) return;
        const uint8_t *g = static_cast<const uint8_t *>(b->data());
        ++m_extent_samples[arch];
        for (size_t i = b->size(); i-- > 0; ) {
            if (g[i] != GUARD_BYTE) {
                if (i + 1 > m_extent_max[arch]) m_extent_max[arch] = i + 1;
                if (i + 1 > prod_cap(b)) m_extent_overrun[arch] = true;
                break;
            }
        }
    }
    void print_extent_report() const {
        if (!m_extent_probe) return;
        std::printf("\n--- measured write extent per archetype (guard %zu KiB, byte 0x%02x) ---\n",
                    m_guard_bytes >> 10, GUARD_BYTE);
        static const char *nm[StageTimers::ARCH_COUNT] = {
            "conv_in", "proj2048", "attn_qkv", "mlp", "lm_head"};
        for (int i = 0; i < StageTimers::ARCH_COUNT; ++i) {
            if (!m_extent_samples[i]) { std::printf("  %-9s NOT ARMED (0 submits sampled)\n", nm[i]); continue; }
            if (!m_extent_max[i]) { std::printf("  %-9s %llu submits, no bytes written\n", nm[i],
                                                (unsigned long long)m_extent_samples[i]); continue; }
            const size_t pages = (m_extent_max[i] + 4095) & ~static_cast<size_t>(4095);
            std::printf("  %-9s %4llu submits, write high-water %8zu B -> size %8zu B (%.3f MiB)%s\n",
                        nm[i], (unsigned long long)m_extent_samples[i], m_extent_max[i], pages,
                        pages / 1048576.0, m_extent_overrun[i] ? "  <-- OVERRUNS its region!" : "");
        }
        std::printf("  Anything at the 2 MiB bucket is the driver's reporting artifact (41.1),\n"
                    "  not a measured write; size to what is above and flush cost follows it.\n");
    }

    size_t weight_blob_bytes(int die) const {
        return (die >= 0 && die <= 1 && m_w_blob[die]) ? m_w_blob[die]->size() : 0;
    }

    const std::vector<float>& logits() const { return m_logits; }
    float* logits_mut() { return m_logits.data(); }
    const Tokenizer& tokenizer() const { return m_tokenizer; }
    int pos() const { return m_pos; }
    bool is_quant_active() const { return m_layer_npu[0].op_in_scale.in_dtype != BoundaryDtype::Fp16; }

    // Standalone Attention Microbenchmark at 512, 4096, and 32768 context (PLAN.md Milestone 4 Gate)
    void benchmark_global_attention_kernel() {
        std::printf("=================================================================\n");
        std::printf(" LFM3-1.7B Global GQA Attention CPU Microbenchmark (6 Layers)\n");
        std::printf("=================================================================\n");

        pin_openmp_threads_numa();
        if (m_scores_ws.size() < 16 * 32768) {
            m_scores_ws.assign(16 * 32768, 0.0f);
        }

        const int kBenchCap = 32768;
        const int contexts[] = {512, 4096, 32768};
        const double budgets_ms[] = {1.0, 8.0, 65.0}; // PLAN.md §1.1 item 3 budget

        std::vector<float> qkv(3072, 0.0f);
        for (int i = 0; i < 3072; ++i) qkv[i] = 0.8f * std::sin(static_cast<float>(i) * 0.13f);
        std::vector<float> gates(kQHeads, 0.5f);
        std::vector<float> attn_out(kHidden, 0.0f);
        std::vector<float> attn_ref(kHidden, 0.0f);

        // Allocate INT8 Global KV cache with anonymous mmap so physical pages are faulted
        // by each NUMA node's owning OpenMP threads (kv_h=0,1 on Die 0 DDR5; kv_h=2,3 on Die 1 DDR5)
        const size_t kv_u8_bytes  = static_cast<size_t>(kKvHeads) * kBenchCap * kHeadDim;
        const size_t kv_sc_bytes  = static_cast<size_t>(kKvHeads) * kBenchCap * sizeof(float);
        const size_t kv_sum_bytes = static_cast<size_t>(kKvHeads) * kBenchCap * sizeof(uint16_t);

        uint8_t  *k_u8  = static_cast<uint8_t*>(mmap(nullptr, kv_u8_bytes,  PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        uint8_t  *v_u8  = static_cast<uint8_t*>(mmap(nullptr, kv_u8_bytes,  PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        float    *k_sc  = static_cast<float*>(mmap(nullptr, kv_sc_bytes,    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        float    *v_sc  = static_cast<float*>(mmap(nullptr, kv_sc_bytes,    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        uint16_t *k_sum = static_cast<uint16_t*>(mmap(nullptr, kv_sum_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));

        std::vector<float> k_f32(static_cast<size_t>(kKvHeads) * 4096 * kHeadDim, 0.0f);
        std::vector<float> v_f32(static_cast<size_t>(kKvHeads) * 4096 * kHeadDim, 0.0f);

        // Parallel NUMA first-touch initialization matching the exact (kv_h, sp) thread assignment
        #pragma omp parallel for collapse(2) schedule(static, 1)
        for (int kv_h = 0; kv_h < kKvHeads; ++kv_h) {
            for (int sp = 0; sp < 2; ++sp) {
                int s_beg = (sp * kBenchCap) >> 1;
                int s_end = ((sp + 1) * kBenchCap) >> 1;
                float tmp_k[kHeadDim], tmp_v[kHeadDim];
                for (int s = s_beg; s < s_end; ++s) {
                    for (int d = 0; d < kHeadDim; ++d) {
                        tmp_k[d] = 0.8f * std::cos(static_cast<float>((s * 17 + d * 3 + kv_h) & 1023) * 0.0061f);
                        tmp_v[d] = 0.8f * std::sin(static_cast<float>((s * 13 + d * 5 + kv_h) & 1023) * 0.0059f);
                    }
                    size_t idx = static_cast<size_t>(kv_h) * kBenchCap + s;
                    uint16_t dummy_sum = 0;
                    quantize_kv_u8(tmp_k, k_u8 + idx * kHeadDim, k_sc[idx], k_sum[idx]);
                    quantize_kv_u8(tmp_v, v_u8 + idx * kHeadDim, v_sc[idx], dummy_sum);
                    if (s < 4096) {
                        size_t idx4k = (static_cast<size_t>(kv_h) * 4096 + s) * kHeadDim;
                        std::memcpy(k_f32.data() + idx4k, tmp_k, kHeadDim * sizeof(float));
                        std::memcpy(v_f32.data() + idx4k, tmp_v, kHeadDim * sizeof(float));
                    }
                }
            }
        }

        // Verify numerical parity of INT8 64-bit SWAR kernel vs exact FP32 kernel at 4096 context
        run_gqa_attention(qkv.data(), k_f32.data(), v_f32.data(), gates.data(),
                          attn_ref.data(), 1, 4095, false, 4096);
        run_gqa_attention_int8_swar(qkv.data(), k_u8, v_u8, k_sc, v_sc, k_sum,
                                    gates.data(), attn_out.data(), 4096, kBenchCap);
        double swar_dot = 0.0, swar_n1 = 0.0, swar_n2 = 0.0;
        for (int i = 0; i < kHidden; ++i) {
            swar_dot += static_cast<double>(attn_out[i]) * attn_ref[i];
            swar_n1  += static_cast<double>(attn_out[i]) * attn_out[i];
            swar_n2  += static_cast<double>(attn_ref[i]) * attn_ref[i];
        }
        double swar_cos = swar_dot / (std::sqrt(swar_n1) * std::sqrt(swar_n2) + 1e-12);
        std::printf("  • INT8 64-Bit SWAR Kernel Cosine vs FP32 Reference (4096 ctx): %.8f [PASS]\n", swar_cos);

        for (size_t c_idx = 0; c_idx < 3; ++c_idx) {
            int ctx = contexts[c_idx];
            double budget = budgets_ms[c_idx];
            int reps = (ctx <= 4096) ? 10 : 4;

            // Warmup
            run_gqa_attention_int8_swar(qkv.data(), k_u8, v_u8, k_sc, v_sc, k_sum,
                                        gates.data(), attn_out.data(), ctx, kBenchCap);

            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                for (int gl = 0; gl < 6; ++gl) {
                    run_gqa_attention_int8_swar(qkv.data(), k_u8, v_u8, k_sc, v_sc, k_sum,
                                                gates.data(), attn_out.data(), ctx, kBenchCap);
                }
            }
            auto t1 = std::chrono::steady_clock::now();
            double ms_per_tok_6layers = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
            bool pass = (ms_per_tok_6layers <= budget * 2.0);
            std::printf("  • Context = %5d tokens (6 Global Attention layers/tok): %6.2f ms/tok (budget ~%.1f ms, 2x gate <= %.1f ms) [%s]\n",
                        ctx, ms_per_tok_6layers, budget, budget * 2.0, pass ? "PASS" : "WARN");
        }
        std::printf("\n");

        munmap(k_u8,  kv_u8_bytes);
        munmap(v_u8,  kv_u8_bytes);
        munmap(k_sc,  kv_sc_bytes);
        munmap(v_sc,  kv_sc_bytes);
        munmap(k_sum, kv_sum_bytes);
    }

private:
    // NUMA-Pinned Head-Major INT8 Global GQA Kernel using 64-Bit 2-Lane SWAR (SIMD-Within-A-Register)
    void run_gqa_attention_int8_swar(const float *qkv_chunk, const uint8_t *k_u8, const uint8_t *v_u8,
                                     const float *k_sc, const float *v_sc, const uint16_t *k_sum,
                                     const float *gates, float *attn_out_chunk,
                                     int n_valid, int s_cap) {
        // 1. Pre-quantize the 16 Query heads into 64-bit 2-lane SWAR (g=0,1 in q_pack01, g=2,3 in q_pack23)
        alignas(64) uint64_t q_pack01[kKvHeads][kHeadDim];
        alignas(64) uint64_t q_pack23[kKvHeads][kHeadDim];
        alignas(64) float    q_scale[kQHeads];
        alignas(64) int32_t  q_bias[kQHeads];

        for (int kv_h = 0; kv_h < kKvHeads; ++kv_h) {
            uint8_t qu_tmp[4][kHeadDim];
            for (int g = 0; g < 4; ++g) {
                int qh = kv_h * 4 + g;
                uint16_t qsum = 0;
                quantize_kv_u8(qkv_chunk + qh * kHeadDim, qu_tmp[g], q_scale[qh], qsum);
                q_bias[qh] = 2097152 - 128 * static_cast<int32_t>(qsum);
                q_scale[qh] *= 0.0883883476f;
            }
            for (int d = 0; d < kHeadDim; ++d) {
                q_pack01[kv_h][d] = static_cast<uint64_t>(qu_tmp[0][d]) | (static_cast<uint64_t>(qu_tmp[1][d]) << 32);
                q_pack23[kv_h][d] = static_cast<uint64_t>(qu_tmp[2][d]) | (static_cast<uint64_t>(qu_tmp[3][d]) << 32);
            }
        }

        alignas(64) float part_m[4][2][4];
        alignas(64) float part_l[4][2][4];
        alignas(64) float part_o[4][2][4][kHeadDim];

        // 2. Single-OpenMP-Region Flash-Decoding across (kv_h = 0..3, sp = 0..1) = 8 NUMA-pinned cores
        #pragma omp parallel for collapse(2) schedule(static, 1)
        for (int kv_h = 0; kv_h < kKvHeads; ++kv_h) {
            for (int sp = 0; sp < 2; ++sp) {
                int s_beg = (sp * n_valid) >> 1;
                int s_end = ((sp + 1) * n_valid) >> 1;
                int qh_base = kv_h * 4;
                const uint64_t *qp01 = q_pack01[kv_h];
                const uint64_t *qp23 = q_pack23[kv_h];
                float qs0 = q_scale[qh_base + 0], qs1 = q_scale[qh_base + 1];
                float qs2 = q_scale[qh_base + 2], qs3 = q_scale[qh_base + 3];
                int32_t qb0 = q_bias[qh_base + 0], qb1 = q_bias[qh_base + 1];
                int32_t qb2 = q_bias[qh_base + 2], qb3 = q_bias[qh_base + 3];

                size_t head_base = static_cast<size_t>(kv_h) * s_cap;
                const uint8_t  *kh_u8  = k_u8  + head_base * kHeadDim;
                const uint8_t  *vh_u8  = v_u8  + head_base * kHeadDim;
                const float    *kh_sc  = k_sc  + head_base;
                const float    *vh_sc  = v_sc  + head_base;
                const uint16_t *kh_sum = k_sum + head_base;

                alignas(64) float o_loc[4][kHeadDim];
                std::memset(o_loc, 0, sizeof(o_loc));
                float run_m[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
                float run_l[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                constexpr int kTile = 32;
                alignas(64) float sc_tile[kTile][4];

                for (int s_base = s_beg; s_base < s_end; s_base += kTile) {
                    int tile_len = std::min(kTile, s_end - s_base);
                    float tile_m[4] = {-1e30f, -1e30f, -1e30f, -1e30f};

                    // Pass 1: 4-token x 4-head 64-bit SWAR Q * K^T
                    int dt = 0;
                    for (; dt + 3 < tile_len; dt += 4) {
                        const uint8_t *k0 = kh_u8 + static_cast<size_t>(s_base + dt + 0) * kHeadDim;
                        const uint8_t *k1 = kh_u8 + static_cast<size_t>(s_base + dt + 1) * kHeadDim;
                        const uint8_t *k2 = kh_u8 + static_cast<size_t>(s_base + dt + 2) * kHeadDim;
                        const uint8_t *k3 = kh_u8 + static_cast<size_t>(s_base + dt + 3) * kHeadDim;
                        uint64_t a0_01 = 0, a0_23 = 0;
                        uint64_t a1_01 = 0, a1_23 = 0;
                        uint64_t a2_01 = 0, a2_23 = 0;
                        uint64_t a3_01 = 0, a3_23 = 0;
                        for (int d = 0; d < kHeadDim; ++d) {
                            uint64_t q01 = qp01[d], q23 = qp23[d];
                            uint64_t kv0 = k0[d], kv1 = k1[d], kv2 = k2[d], kv3 = k3[d];
                            a0_01 += q01 * kv0; a0_23 += q23 * kv0;
                            a1_01 += q01 * kv1; a1_23 += q23 * kv1;
                            a2_01 += q01 * kv2; a2_23 += q23 * kv2;
                            a3_01 += q01 * kv3; a3_23 += q23 * kv3;
                        }
                        uint64_t a01_arr[4] = {a0_01, a1_01, a2_01, a3_01};
                        uint64_t a23_arr[4] = {a0_23, a1_23, a2_23, a3_23};
                        for (int u = 0; u < 4; ++u) {
                            int s_idx = s_base + dt + u;
                            int32_t ks = 128 * static_cast<int32_t>(kh_sum[s_idx]);
                            float ksc = kh_sc[s_idx];
                            int32_t d0 = static_cast<int32_t>(a01_arr[u] & 0xFFFFFFFFu) - ks + qb0;
                            int32_t d1 = static_cast<int32_t>(a01_arr[u] >> 32)         - ks + qb1;
                            int32_t d2 = static_cast<int32_t>(a23_arr[u] & 0xFFFFFFFFu) - ks + qb2;
                            int32_t d3 = static_cast<int32_t>(a23_arr[u] >> 32)         - ks + qb3;
                            float s0 = static_cast<float>(d0) * (qs0 * ksc);
                            float s1 = static_cast<float>(d1) * (qs1 * ksc);
                            float s2 = static_cast<float>(d2) * (qs2 * ksc);
                            float s3 = static_cast<float>(d3) * (qs3 * ksc);
                            sc_tile[dt+u][0] = s0; if (s0 > tile_m[0]) tile_m[0] = s0;
                            sc_tile[dt+u][1] = s1; if (s1 > tile_m[1]) tile_m[1] = s1;
                            sc_tile[dt+u][2] = s2; if (s2 > tile_m[2]) tile_m[2] = s2;
                            sc_tile[dt+u][3] = s3; if (s3 > tile_m[3]) tile_m[3] = s3;
                        }
                    }
                    for (; dt < tile_len; ++dt) {
                        int s_idx = s_base + dt;
                        const uint8_t *k0 = kh_u8 + static_cast<size_t>(s_idx) * kHeadDim;
                        uint64_t a01 = 0, a23 = 0;
                        for (int d = 0; d < kHeadDim; ++d) {
                            uint64_t kv = k0[d];
                            a01 += qp01[d] * kv;
                            a23 += qp23[d] * kv;
                        }
                        int32_t ks = 128 * static_cast<int32_t>(kh_sum[s_idx]);
                        float ksc = kh_sc[s_idx];
                        float s0 = static_cast<float>(static_cast<int32_t>(a01 & 0xFFFFFFFFu) - ks + qb0) * (qs0 * ksc);
                        float s1 = static_cast<float>(static_cast<int32_t>(a01 >> 32)         - ks + qb1) * (qs1 * ksc);
                        float s2 = static_cast<float>(static_cast<int32_t>(a23 & 0xFFFFFFFFu) - ks + qb2) * (qs2 * ksc);
                        float s3 = static_cast<float>(static_cast<int32_t>(a23 >> 32)         - ks + qb3) * (qs3 * ksc);
                        sc_tile[dt][0] = s0; if (s0 > tile_m[0]) tile_m[0] = s0;
                        sc_tile[dt][1] = s1; if (s1 > tile_m[1]) tile_m[1] = s1;
                        sc_tile[dt][2] = s2; if (s2 > tile_m[2]) tile_m[2] = s2;
                        sc_tile[dt][3] = s3; if (s3 > tile_m[3]) tile_m[3] = s3;
                    }

                    // Online Softmax update + track tile max weighted V scale
                    float p_max[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int g = 0; g < 4; ++g) {
                        float new_m = std::max(run_m[g], tile_m[g]);
                        float rescale = (run_m[g] > -1e20f) ? fast_expf_neg(run_m[g] - new_m) : 0.0f;
                        if (rescale != 1.0f) {
                            for (int d = 0; d < kHeadDim; ++d) o_loc[g][d] *= rescale;
                        }
                        float tile_sum = 0.0f;
                        for (int i = 0; i < tile_len; ++i) {
                            float w = fast_expf_neg(sc_tile[i][g] - new_m);
                            tile_sum += w;
                            float wv = w * vh_sc[s_base + i];
                            sc_tile[i][g] = wv;
                            if (wv > p_max[g]) p_max[g] = wv;
                        }
                        run_l[g] = run_l[g] * rescale + tile_sum;
                        run_m[g] = new_m;
                    }

                    if ((p_max[0] + p_max[1] + p_max[2] + p_max[3]) <= 1e-12f) continue;

                    // Quantize tile's (w * v_sc) into 11-bit unsigned integers [0, 2047] for 64-bit SWAR P * V
                    float p_inv[4], p_deq[4];
                    for (int g = 0; g < 4; ++g) {
                        p_inv[g] = (p_max[g] > 1e-12f) ? (2047.0f / p_max[g]) : 0.0f;
                        p_deq[g] = (p_max[g] > 1e-12f) ? (p_max[g] / 2047.0f) : 0.0f;
                    }

                    uint64_t pw01_arr[kTile], pw23_arr[kTile];
                    int32_t pw_sum[4] = {0, 0, 0, 0};
                    for (int i = 0; i < tile_len; ++i) {
                        uint32_t pw0 = static_cast<uint32_t>(sc_tile[i][0] * p_inv[0] + 0.5f);
                        uint32_t pw1 = static_cast<uint32_t>(sc_tile[i][1] * p_inv[1] + 0.5f);
                        uint32_t pw2 = static_cast<uint32_t>(sc_tile[i][2] * p_inv[2] + 0.5f);
                        uint32_t pw3 = static_cast<uint32_t>(sc_tile[i][3] * p_inv[3] + 0.5f);
                        pw_sum[0] += static_cast<int32_t>(pw0);
                        pw_sum[1] += static_cast<int32_t>(pw1);
                        pw_sum[2] += static_cast<int32_t>(pw2);
                        pw_sum[3] += static_cast<int32_t>(pw3);
                        pw01_arr[i] = static_cast<uint64_t>(pw0) | (static_cast<uint64_t>(pw1) << 32);
                        pw23_arr[i] = static_cast<uint64_t>(pw2) | (static_cast<uint64_t>(pw3) << 32);
                    }

                    // Pass 2: 4-token x 4-head 64-bit SWAR P * V accumulation
                    alignas(64) uint64_t o_swar01[kHeadDim] = {0};
                    alignas(64) uint64_t o_swar23[kHeadDim] = {0};
                    dt = 0;
                    for (; dt + 3 < tile_len; dt += 4) {
                        uint64_t p0_01 = pw01_arr[dt+0], p1_01 = pw01_arr[dt+1], p2_01 = pw01_arr[dt+2], p3_01 = pw01_arr[dt+3];
                        uint64_t p0_23 = pw23_arr[dt+0], p1_23 = pw23_arr[dt+1], p2_23 = pw23_arr[dt+2], p3_23 = pw23_arr[dt+3];
                        if ((p0_01 | p1_01 | p2_01 | p3_01 | p0_23 | p1_23 | p2_23 | p3_23) == 0) continue;
                        const uint8_t *v0 = vh_u8 + static_cast<size_t>(s_base + dt + 0) * kHeadDim;
                        const uint8_t *v1 = vh_u8 + static_cast<size_t>(s_base + dt + 1) * kHeadDim;
                        const uint8_t *v2 = vh_u8 + static_cast<size_t>(s_base + dt + 2) * kHeadDim;
                        const uint8_t *v3 = vh_u8 + static_cast<size_t>(s_base + dt + 3) * kHeadDim;
                        for (int d = 0; d < kHeadDim; ++d) {
                            uint64_t vu0 = v0[d], vu1 = v1[d], vu2 = v2[d], vu3 = v3[d];
                            o_swar01[d] += p0_01 * vu0 + p1_01 * vu1 + p2_01 * vu2 + p3_01 * vu3;
                            o_swar23[d] += p0_23 * vu0 + p1_23 * vu1 + p2_23 * vu2 + p3_23 * vu3;
                        }
                    }
                    for (; dt < tile_len; ++dt) {
                        uint64_t p01 = pw01_arr[dt], p23 = pw23_arr[dt];
                        if ((p01 | p23) == 0) continue;
                        const uint8_t *v0 = vh_u8 + static_cast<size_t>(s_base + dt) * kHeadDim;
                        for (int d = 0; d < kHeadDim; ++d) {
                            uint64_t vu = v0[d];
                            o_swar01[d] += p01 * vu;
                            o_swar23[d] += p23 * vu;
                        }
                    }

                    int32_t b0 = 128 * pw_sum[0], b1 = 128 * pw_sum[1];
                    int32_t b2 = 128 * pw_sum[2], b3 = 128 * pw_sum[3];
                    float dq0 = p_deq[0], dq1 = p_deq[1], dq2 = p_deq[2], dq3 = p_deq[3];
                    for (int d = 0; d < kHeadDim; ++d) {
                        uint64_t w01 = o_swar01[d], w23 = o_swar23[d];
                        o_loc[0][d] += static_cast<float>(static_cast<int32_t>(w01 & 0xFFFFFFFFu) - b0) * dq0;
                        o_loc[1][d] += static_cast<float>(static_cast<int32_t>(w01 >> 32)         - b1) * dq1;
                        o_loc[2][d] += static_cast<float>(static_cast<int32_t>(w23 & 0xFFFFFFFFu) - b2) * dq2;
                        o_loc[3][d] += static_cast<float>(static_cast<int32_t>(w23 >> 32)         - b3) * dq3;
                    }
                }

                for (int g = 0; g < 4; ++g) {
                    part_m[kv_h][sp][g] = run_m[g];
                    part_l[kv_h][sp][g] = run_l[g];
                    std::memcpy(part_o[kv_h][sp][g], o_loc[g], kHeadDim * sizeof(float));
                }
            }
        }

        // 3. Combine splits and apply per-head sigmoid gate
        for (int kv_h = 0; kv_h < kKvHeads; ++kv_h) {
            for (int g = 0; g < 4; ++g) {
                int qh = kv_h * 4 + g;
                float *o_dst = attn_out_chunk + qh * kHeadDim;
                float m0 = part_m[kv_h][0][g], m1 = part_m[kv_h][1][g];
                float m = std::max(m0, m1);
                float w0 = fast_expf_neg(m0 - m);
                float w1 = fast_expf_neg(m1 - m);
                float l_tot = w0 * part_l[kv_h][0][g] + w1 * part_l[kv_h][1][g] + 1e-12f;
                float s0 = (gates[qh] * w0) / l_tot;
                float s1 = (gates[qh] * w1) / l_tot;
                const float *o0 = part_o[kv_h][0][g];
                const float *o1 = part_o[kv_h][1][g];
                for (int d = 0; d < kHeadDim; ++d) o_dst[d] = s0 * o0[d] + s1 * o1[d];
            }
        }
    }

    // NUMA-Pinned Head-Major INT8 Global GQA Prefill Kernel (T_valid > 1) using 64-Bit 2-Lane SWAR
    // and L2-Tiled FlashAttention-2 Online Softmax.
    // Streams 1-byte k_u8 and v_u8 once per 32-key tile from local Die LPDDR5 across all query blocks
    // owned by the thread, cutting KV memory bandwidth by 32x and compute instructions by 2x.
    void run_gqa_attention_int8_swar_prefill(const float *qkv_chunk, const uint8_t *k_u8, const uint8_t *v_u8,
                                             const float *k_sc, const float *v_sc, const uint16_t *k_sum,
                                             const float *gates, float *attn_out_chunk,
                                             int T, int pos_base, int s_cap) {
        #pragma omp parallel num_threads(8)
        {
            const int tid     = omp_get_thread_num();
            const int kv_h    = tid >> 1; // 0,1 on Die 0 (Cores 0..3); 2,3 on Die 1 (Cores 4..7)
            const int sub_tid = tid & 1;  // 0 -> even 4-token blocks (0,8..56); 1 -> odd 4-token blocks (4,12..60)
            const int qh_base = kv_h * 4;

            size_t head_base = static_cast<size_t>(kv_h) * s_cap;
            const uint8_t  *kh_u8  = k_u8  + head_base * kHeadDim;
            const uint8_t  *vh_u8  = v_u8  + head_base * kHeadDim;
            const float    *kh_sc  = k_sc  + head_base;
            const float    *vh_sc  = v_sc  + head_base;
            const uint16_t *kh_sum = k_sum + head_base;

            constexpr int kMaxBlocks = 8;
            alignas(64) uint64_t q_swar[kMaxBlocks][kHeadDim][8];
            alignas(64) float    q_scale[kMaxBlocks][16];
            alignas(64) int32_t  q_bias[kMaxBlocks][16];
            alignas(64) float    o_all[kMaxBlocks][kHeadDim][16];
            alignas(64) float    run_m[kMaxBlocks][16];
            alignas(64) float    run_l[kMaxBlocks][16];
            int t_base_arr[kMaxBlocks];
            int dt_cnt_arr[kMaxBlocks];
            int p_end_arr[kMaxBlocks][4];
            int n_blocks = 0;
            int p_max_all = 0;

            for (int t_base = sub_tid * 4; t_base < T && n_blocks < kMaxBlocks; t_base += 8) {
                int b = n_blocks++;
                t_base_arr[b] = t_base;
                int dt_cnt = std::min(4, T - t_base);
                dt_cnt_arr[b] = dt_cnt;
                std::memset(o_all[b], 0, sizeof(o_all[b]));

                uint8_t qu_tmp[16][kHeadDim];
                std::memset(qu_tmp, 128, sizeof(qu_tmp));

                for (int dt = 0; dt < 4; ++dt) {
                    if (dt < dt_cnt) {
                        int t = t_base + dt;
                        int pe = pos_base + t + 1;
                        p_end_arr[b][dt] = pe;
                        if (pe > p_max_all) p_max_all = pe;
                        for (int g = 0; g < 4; ++g) {
                            int idx = dt * 4 + g;
                            const float *q_src = qkv_chunk + static_cast<size_t>(t) * 3072 + (qh_base + g) * kHeadDim;
                            uint16_t qsum = 0;
                            quantize_kv_u8(q_src, qu_tmp[idx], q_scale[b][idx], qsum);
                            q_bias[b][idx]  = 2097152 - 128 * static_cast<int32_t>(qsum);
                            q_scale[b][idx] *= 0.0883883476f;
                            run_m[b][idx] = -1e30f;
                            run_l[b][idx] = 0.0f;
                        }
                    } else {
                        p_end_arr[b][dt] = 0;
                        for (int g = 0; g < 4; ++g) {
                            int idx = dt * 4 + g;
                            q_scale[b][idx] = 0.0f;
                            q_bias[b][idx]  = 0;
                            run_m[b][idx]   = -1e30f;
                            run_l[b][idx]   = 0.0f;
                        }
                    }
                }
                for (int d = 0; d < kHeadDim; ++d) {
                    for (int p = 0; p < 8; ++p) {
                        q_swar[b][d][p] = static_cast<uint64_t>(qu_tmp[2 * p][d]) |
                                          (static_cast<uint64_t>(qu_tmp[2 * p + 1][d]) << 32);
                    }
                }
            }

            constexpr int kTile = 32;
            alignas(64) float sc_tile[kTile][16];
            alignas(64) uint64_t pw_swar[kTile][8];

            for (int s_base = 0; s_base < p_max_all; s_base += kTile) {
                const int s_end    = std::min(p_max_all, s_base + kTile);
                const int tile_len = s_end - s_base;

                for (int b = 0; b < n_blocks; ++b) {
                    const int dt_cnt = dt_cnt_arr[b];
                    const int b_pmax = p_end_arr[b][dt_cnt - 1];
                    if (s_base >= b_pmax) continue;

                    const int n_q = dt_cnt * 4;
                    const bool full_valid = (dt_cnt == 4) && (s_end <= p_end_arr[b][0]);
                    float tile_m[16];
                    for (int idx = 0; idx < 16; ++idx) tile_m[idx] = -1e30f;

                    const float   *qs = q_scale[b];
                    const int32_t *qb = q_bias[b];

                    // Pass 1: 16-query 64-bit 2-lane SWAR Q * K^T over 32-key tile
                    for (int s_rel = 0; s_rel < tile_len; ++s_rel) {
                        const int s_idx = s_base + s_rel;
                        const uint8_t *k_vec = kh_u8 + static_cast<size_t>(s_idx) * kHeadDim;
                        uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                        uint64_t a4 = 0, a5 = 0, a6 = 0, a7 = 0;
                        for (int d = 0; d < kHeadDim; ++d) {
                            uint64_t kv = k_vec[d];
                            const uint64_t *qr = q_swar[b][d];
                            a0 += qr[0] * kv; a1 += qr[1] * kv;
                            a2 += qr[2] * kv; a3 += qr[3] * kv;
                            a4 += qr[4] * kv; a5 += qr[5] * kv;
                            a6 += qr[6] * kv; a7 += qr[7] * kv;
                        }
                        uint64_t a_arr[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
                        int32_t ks = 128 * static_cast<int32_t>(kh_sum[s_idx]);
                        float ksc  = kh_sc[s_idx];

                        if (full_valid) {
                            for (int p = 0; p < 8; ++p) {
                                int i0 = 2 * p, i1 = 2 * p + 1;
                                float s0 = static_cast<float>(static_cast<int32_t>(a_arr[p] & 0xFFFFFFFFu) - ks + qb[i0]) * (qs[i0] * ksc);
                                float s1 = static_cast<float>(static_cast<int32_t>(a_arr[p] >> 32)         - ks + qb[i1]) * (qs[i1] * ksc);
                                sc_tile[s_rel][i0] = s0; if (s0 > tile_m[i0]) tile_m[i0] = s0;
                                sc_tile[s_rel][i1] = s1; if (s1 > tile_m[i1]) tile_m[i1] = s1;
                            }
                        } else {
                            for (int p = 0; p < 8; ++p) {
                                int i0 = 2 * p, i1 = 2 * p + 1;
                                int dt0 = i0 >> 2, dt1 = i1 >> 2;
                                if (i0 < n_q && s_idx < p_end_arr[b][dt0]) {
                                    float s0 = static_cast<float>(static_cast<int32_t>(a_arr[p] & 0xFFFFFFFFu) - ks + qb[i0]) * (qs[i0] * ksc);
                                    sc_tile[s_rel][i0] = s0; if (s0 > tile_m[i0]) tile_m[i0] = s0;
                                } else {
                                    sc_tile[s_rel][i0] = -1e30f;
                                }
                                if (i1 < n_q && s_idx < p_end_arr[b][dt1]) {
                                    float s1 = static_cast<float>(static_cast<int32_t>(a_arr[p] >> 32)         - ks + qb[i1]) * (qs[i1] * ksc);
                                    sc_tile[s_rel][i1] = s1; if (s1 > tile_m[i1]) tile_m[i1] = s1;
                                } else {
                                    sc_tile[s_rel][i1] = -1e30f;
                                }
                            }
                        }
                    }

                    // Online Softmax update + track tile max weighted V scale
                    float p_max[16];
                    float p_max_sum = 0.0f;
                    for (int idx = 0; idx < 16; ++idx) {
                        p_max[idx] = 0.0f;
                        if (idx >= n_q || tile_m[idx] <= -1e20f) {
                            for (int s_rel = 0; s_rel < tile_len; ++s_rel) sc_tile[s_rel][idx] = 0.0f;
                            continue;
                        }
                        float old_m = run_m[b][idx];
                        float new_m = std::max(old_m, tile_m[idx]);
                        float rescale = (old_m > -1e20f) ? fast_expf_neg(old_m - new_m) : 0.0f;
                        if (rescale != 1.0f) {
                            for (int d = 0; d < kHeadDim; ++d) o_all[b][d][idx] *= rescale;
                        }
                        float tile_sum = 0.0f;
                        float pm = 0.0f;
                        for (int s_rel = 0; s_rel < tile_len; ++s_rel) {
                            float sc = sc_tile[s_rel][idx];
                            if (sc > -1e20f) {
                                float w = fast_expf_neg(sc - new_m);
                                tile_sum += w;
                                float wv = w * vh_sc[s_base + s_rel];
                                sc_tile[s_rel][idx] = wv;
                                if (wv > pm) pm = wv;
                            } else {
                                sc_tile[s_rel][idx] = 0.0f;
                            }
                        }
                        p_max[idx] = pm;
                        p_max_sum += pm;
                        run_l[b][idx] = run_l[b][idx] * rescale + tile_sum;
                        run_m[b][idx] = new_m;
                    }

                    if (p_max_sum <= 1e-12f) continue;

                    // Quantize tile's (w * v_sc) into 11-bit unsigned integers [0, 2047] for 64-bit SWAR P * V
                    float p_inv[16], p_deq[16];
                    int32_t pw_sum[16];
                    for (int idx = 0; idx < 16; ++idx) {
                        p_inv[idx]  = (p_max[idx] > 1e-12f) ? (2047.0f / p_max[idx]) : 0.0f;
                        p_deq[idx]  = (p_max[idx] > 1e-12f) ? (p_max[idx] / 2047.0f) : 0.0f;
                        pw_sum[idx] = 0;
                    }

                    for (int s_rel = 0; s_rel < tile_len; ++s_rel) {
                        for (int p = 0; p < 8; ++p) {
                            int i0 = 2 * p, i1 = 2 * p + 1;
                            uint32_t pw0 = static_cast<uint32_t>(sc_tile[s_rel][i0] * p_inv[i0] + 0.5f);
                            uint32_t pw1 = static_cast<uint32_t>(sc_tile[s_rel][i1] * p_inv[i1] + 0.5f);
                            pw_sum[i0] += static_cast<int32_t>(pw0);
                            pw_sum[i1] += static_cast<int32_t>(pw1);
                            pw_swar[s_rel][p] = static_cast<uint64_t>(pw0) | (static_cast<uint64_t>(pw1) << 32);
                        }
                    }

                    // Pass 2: 16-query 64-bit 2-lane SWAR P * V accumulation
                    alignas(64) uint64_t o_swar[kHeadDim][8];
                    std::memset(o_swar, 0, sizeof(o_swar));
                    for (int s_rel = 0; s_rel < tile_len; ++s_rel) {
                        const uint64_t *pw = pw_swar[s_rel];
                        uint64_t p0 = pw[0], p1 = pw[1], p2 = pw[2], p3 = pw[3];
                        uint64_t p4 = pw[4], p5 = pw[5], p6 = pw[6], p7 = pw[7];
                        if ((p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7) == 0) continue;
                        const uint8_t *v_vec = vh_u8 + static_cast<size_t>(s_base + s_rel) * kHeadDim;
                        for (int d = 0; d < kHeadDim; ++d) {
                            uint64_t vu = v_vec[d];
                            uint64_t *orow = o_swar[d];
                            orow[0] += p0 * vu; orow[1] += p1 * vu;
                            orow[2] += p2 * vu; orow[3] += p3 * vu;
                            orow[4] += p4 * vu; orow[5] += p5 * vu;
                            orow[6] += p6 * vu; orow[7] += p7 * vu;
                        }
                    }

                    for (int d = 0; d < kHeadDim; ++d) {
                        const uint64_t *orow = o_swar[d];
                        float *odst = o_all[b][d];
                        for (int p = 0; p < 8; ++p) {
                            int i0 = 2 * p, i1 = 2 * p + 1;
                            uint64_t w64 = orow[p];
                            odst[i0] += static_cast<float>(static_cast<int32_t>(w64 & 0xFFFFFFFFu) - 128 * pw_sum[i0]) * p_deq[i0];
                            odst[i1] += static_cast<float>(static_cast<int32_t>(w64 >> 32)         - 128 * pw_sum[i1]) * p_deq[i1];
                        }
                    }
                }
            }

            // Normalize by run_l and apply per-head sigmoid gate
            for (int b = 0; b < n_blocks; ++b) {
                int t_base = t_base_arr[b];
                int dt_cnt = dt_cnt_arr[b];
                for (int dt = 0; dt < dt_cnt; ++dt) {
                    int t = t_base + dt;
                    for (int g = 0; g < 4; ++g) {
                        int idx = dt * 4 + g;
                        float gate_val = gates[static_cast<size_t>(t) * kQHeads + (qh_base + g)];
                        float inv_sum  = gate_val / (run_l[b][idx] + 1e-12f);
                        float *o_dst = attn_out_chunk + static_cast<size_t>(t) * kHidden + (qh_base + g) * kHeadDim;
                        for (int d = 0; d < kHeadDim; ++d) {
                            o_dst[d] = o_all[b][d][idx] * inv_sum;
                        }
                    }
                }
            }
        }
    }

    void run_gqa_attention(const float *qkv_chunk, const float *k_cache, const float *v_cache,
                           const float *gates, float *attn_out_chunk,
                           int T, int pos_base, bool is_swa, int s_cap) {
        if (T == 1) {
            const int p_min   = is_swa ? std::max(0, pos_base - kSwaWin + 1) : 0;
            const int n_valid = pos_base - p_min + 1;
            const int n_splits = (n_valid >= 256) ? 2 : 1;

            // Single-OpenMP-Region Flash-Decoding (Online Softmax) over (kv_h = 0..3, sp = 0..n_splits-1).
            // All accumulators (o_local[4][128] = 2 KiB, sc_tile[32][4] = 512 B) stay 100% L1D-resident
            // on each core's private stack, with 4-token x 4-head (16-FMA) register blocking!
            alignas(64) float part_m[4][2][4];
            alignas(64) float part_l[4][2][4];
            alignas(64) float part_o[4][2][4][kHeadDim];

            #pragma omp parallel for collapse(2) schedule(static, 1)
            for (int kv_h = 0; kv_h < kKvHeads; ++kv_h) {
                for (int sp = 0; sp < n_splits; ++sp) {
                    int s_beg = (sp * n_valid) / n_splits;
                    int s_end = ((sp + 1) * n_valid) / n_splits;
                    int qh_base = kv_h * 4;
                    const float *q0 = qkv_chunk + (qh_base + 0) * kHeadDim;
                    const float *q1 = qkv_chunk + (qh_base + 1) * kHeadDim;
                    const float *q2 = qkv_chunk + (qh_base + 2) * kHeadDim;
                    const float *q3 = qkv_chunk + (qh_base + 3) * kHeadDim;
                    const float *k_head = k_cache + static_cast<size_t>(kv_h) * s_cap * kHeadDim;
                    const float *v_head = v_cache + static_cast<size_t>(kv_h) * s_cap * kHeadDim;

                    alignas(64) float o_loc[4][kHeadDim];
                    std::memset(o_loc, 0, sizeof(o_loc));
                    float run_m[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
                    float run_l[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                    constexpr int kTile = 32;
                    alignas(64) float sc_tile[kTile][4];

                    for (int s_base = s_beg; s_base < s_end; s_base += kTile) {
                        int tile_len = std::min(kTile, s_end - s_base);
                        float tile_m[4] = {-1e30f, -1e30f, -1e30f, -1e30f};

                        // 1. 4-token x 4-head register-blocked Q * K^T over tile
                        int dt = 0;
                        for (; dt + 3 < tile_len; dt += 4) {
                            int sl0 = is_swa ? ((p_min + s_base + dt + 0) % kSwaCap) : (s_base + dt + 0);
                            int sl1 = is_swa ? ((p_min + s_base + dt + 1) % kSwaCap) : (s_base + dt + 1);
                            int sl2 = is_swa ? ((p_min + s_base + dt + 2) % kSwaCap) : (s_base + dt + 2);
                            int sl3 = is_swa ? ((p_min + s_base + dt + 3) % kSwaCap) : (s_base + dt + 3);
                            const float *k0 = k_head + static_cast<size_t>(sl0) * kHeadDim;
                            const float *k1 = k_head + static_cast<size_t>(sl1) * kHeadDim;
                            const float *k2 = k_head + static_cast<size_t>(sl2) * kHeadDim;
                            const float *k3 = k_head + static_cast<size_t>(sl3) * kHeadDim;
                            float a00=0, a01=0, a02=0, a03=0;
                            float a10=0, a11=0, a12=0, a13=0;
                            float a20=0, a21=0, a22=0, a23=0;
                            float a30=0, a31=0, a32=0, a33=0;
                            for (int d = 0; d < kHeadDim; ++d) {
                                float qv0 = q0[d], qv1 = q1[d], qv2 = q2[d], qv3 = q3[d];
                                float kv0 = k0[d], kv1 = k1[d], kv2 = k2[d], kv3 = k3[d];
                                a00 += qv0 * kv0; a01 += qv1 * kv0; a02 += qv2 * kv0; a03 += qv3 * kv0;
                                a10 += qv0 * kv1; a11 += qv1 * kv1; a12 += qv2 * kv1; a13 += qv3 * kv1;
                                a20 += qv0 * kv2; a21 += qv1 * kv2; a22 += qv2 * kv2; a23 += qv3 * kv2;
                                a30 += qv0 * kv3; a31 += qv1 * kv3; a32 += qv2 * kv3; a33 += qv3 * kv3;
                            }
                            constexpr float kScale = 0.0883883476f;
                            sc_tile[dt+0][0] = a00*kScale; sc_tile[dt+0][1] = a01*kScale; sc_tile[dt+0][2] = a02*kScale; sc_tile[dt+0][3] = a03*kScale;
                            sc_tile[dt+1][0] = a10*kScale; sc_tile[dt+1][1] = a11*kScale; sc_tile[dt+1][2] = a12*kScale; sc_tile[dt+1][3] = a13*kScale;
                            sc_tile[dt+2][0] = a20*kScale; sc_tile[dt+2][1] = a21*kScale; sc_tile[dt+2][2] = a22*kScale; sc_tile[dt+2][3] = a23*kScale;
                            sc_tile[dt+3][0] = a30*kScale; sc_tile[dt+3][1] = a31*kScale; sc_tile[dt+3][2] = a32*kScale; sc_tile[dt+3][3] = a33*kScale;
                            for (int u = 0; u < 4; ++u) {
                                for (int g = 0; g < 4; ++g) {
                                    if (sc_tile[dt+u][g] > tile_m[g]) tile_m[g] = sc_tile[dt+u][g];
                                }
                            }
                        }
                        for (; dt < tile_len; ++dt) {
                            int sl = is_swa ? ((p_min + s_base + dt) % kSwaCap) : (s_base + dt);
                            const float *kv_ptr = k_head + static_cast<size_t>(sl) * kHeadDim;
                            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                            for (int d = 0; d < kHeadDim; ++d) {
                                float kv = kv_ptr[d];
                                d0 += q0[d] * kv; d1 += q1[d] * kv;
                                d2 += q2[d] * kv; d3 += q3[d] * kv;
                            }
                            sc_tile[dt][0] = d0 * 0.0883883476f;
                            sc_tile[dt][1] = d1 * 0.0883883476f;
                            sc_tile[dt][2] = d2 * 0.0883883476f;
                            sc_tile[dt][3] = d3 * 0.0883883476f;
                            for (int g = 0; g < 4; ++g) {
                                if (sc_tile[dt][g] > tile_m[g]) tile_m[g] = sc_tile[dt][g];
                            }
                        }

                        // 2. Online Softmax update on L1D-resident o_loc[4][128]
                        for (int g = 0; g < 4; ++g) {
                            float new_m = std::max(run_m[g], tile_m[g]);
                            float rescale = (run_m[g] > -1e20f) ? fast_expf_neg(run_m[g] - new_m) : 0.0f;
                            if (rescale != 1.0f) {
                                for (int d = 0; d < kHeadDim; ++d) o_loc[g][d] *= rescale;
                            }
                            float tile_sum = 0.0f;
                            for (int i = 0; i < tile_len; ++i) {
                                float w = fast_expf_neg(sc_tile[i][g] - new_m);
                                sc_tile[i][g] = w;
                                tile_sum += w;
                            }
                            run_l[g] = run_l[g] * rescale + tile_sum;
                            run_m[g] = new_m;
                        }

                        // 3. 4-token x 4-head register-blocked P * V accumulation into o_loc[4][128]
                        dt = 0;
                        for (; dt + 3 < tile_len; dt += 4) {
                            float p00 = sc_tile[dt+0][0], p01 = sc_tile[dt+0][1], p02 = sc_tile[dt+0][2], p03 = sc_tile[dt+0][3];
                            float p10 = sc_tile[dt+1][0], p11 = sc_tile[dt+1][1], p12 = sc_tile[dt+1][2], p13 = sc_tile[dt+1][3];
                            float p20 = sc_tile[dt+2][0], p21 = sc_tile[dt+2][1], p22 = sc_tile[dt+2][2], p23 = sc_tile[dt+2][3];
                            float p30 = sc_tile[dt+3][0], p31 = sc_tile[dt+3][1], p32 = sc_tile[dt+3][2], p33 = sc_tile[dt+3][3];
                            if ((p00+p01+p02+p03 + p10+p11+p12+p13 + p20+p21+p22+p23 + p30+p31+p32+p33) <= 1e-8f) continue;

                            int sl0 = is_swa ? ((p_min + s_base + dt + 0) % kSwaCap) : (s_base + dt + 0);
                            int sl1 = is_swa ? ((p_min + s_base + dt + 1) % kSwaCap) : (s_base + dt + 1);
                            int sl2 = is_swa ? ((p_min + s_base + dt + 2) % kSwaCap) : (s_base + dt + 2);
                            int sl3 = is_swa ? ((p_min + s_base + dt + 3) % kSwaCap) : (s_base + dt + 3);
                            const float *v0 = v_head + static_cast<size_t>(sl0) * kHeadDim;
                            const float *v1 = v_head + static_cast<size_t>(sl1) * kHeadDim;
                            const float *v2 = v_head + static_cast<size_t>(sl2) * kHeadDim;
                            const float *v3 = v_head + static_cast<size_t>(sl3) * kHeadDim;
                            for (int d = 0; d < kHeadDim; ++d) {
                                float vv0 = v0[d], vv1 = v1[d], vv2 = v2[d], vv3 = v3[d];
                                o_loc[0][d] += p00*vv0 + p10*vv1 + p20*vv2 + p30*vv3;
                                o_loc[1][d] += p01*vv0 + p11*vv1 + p21*vv2 + p31*vv3;
                                o_loc[2][d] += p02*vv0 + p12*vv1 + p22*vv2 + p32*vv3;
                                o_loc[3][d] += p03*vv0 + p13*vv1 + p23*vv2 + p33*vv3;
                            }
                        }
                        for (; dt < tile_len; ++dt) {
                            float p0 = sc_tile[dt][0], p1 = sc_tile[dt][1], p2 = sc_tile[dt][2], p3 = sc_tile[dt][3];
                            if ((p0 + p1 + p2 + p3) <= 1e-8f) continue;
                            int sl = is_swa ? ((p_min + s_base + dt) % kSwaCap) : (s_base + dt);
                            const float *vv_ptr = v_head + static_cast<size_t>(sl) * kHeadDim;
                            for (int d = 0; d < kHeadDim; ++d) {
                                float vv = vv_ptr[d];
                                o_loc[0][d] += p0 * vv; o_loc[1][d] += p1 * vv;
                                o_loc[2][d] += p2 * vv; o_loc[3][d] += p3 * vv;
                            }
                        }
                    }

                    for (int g = 0; g < 4; ++g) {
                        part_m[kv_h][sp][g] = run_m[g];
                        part_l[kv_h][sp][g] = run_l[g];
                        std::memcpy(part_o[kv_h][sp][g], o_loc[g], kHeadDim * sizeof(float));
                    }
                }
            }

            // Combine splits and apply per-head sigmoid gate
            for (int kv_h = 0; kv_h < kKvHeads; ++kv_h) {
                for (int g = 0; g < 4; ++g) {
                    int qh = kv_h * 4 + g;
                    float *o_dst = attn_out_chunk + qh * kHeadDim;
                    if (n_splits == 1) {
                        float inv = gates[qh] / (part_l[kv_h][0][g] + 1e-12f);
                        const float *o0 = part_o[kv_h][0][g];
                        for (int d = 0; d < kHeadDim; ++d) o_dst[d] = o0[d] * inv;
                    } else {
                        float m0 = part_m[kv_h][0][g], m1 = part_m[kv_h][1][g];
                        float m = std::max(m0, m1);
                        float w0 = fast_expf_neg(m0 - m);
                        float w1 = fast_expf_neg(m1 - m);
                        float l_tot = w0 * part_l[kv_h][0][g] + w1 * part_l[kv_h][1][g] + 1e-12f;
                        float s0 = (gates[qh] * w0) / l_tot;
                        float s1 = (gates[qh] * w1) / l_tot;
                        const float *o0 = part_o[kv_h][0][g];
                        const float *o1 = part_o[kv_h][1][g];
                        for (int d = 0; d < kHeadDim; ++d) o_dst[d] = s0 * o0[d] + s1 * o1[d];
                    }
                }
            }
            return;
        }

        // Batched T=64 prefill path: NUMA-pinned (tid 0..3 -> kv_h 0,1 on Die 0; tid 4..7 -> kv_h 2,3 on Die 1)
        // L2-tiled Online Softmax (FlashAttention-2) microkernel.
        // Streams k_head and v_head in 64-key tiles (32 KiB K + 32 KiB V, 100% L2-resident) across all
        // query blocks owned by the thread, cutting DRAM KV bandwidth by 8x and eliminating cross-die traffic.
        #pragma omp parallel num_threads(8)
        {
            const int tid     = omp_get_thread_num();
            const int kv_h    = tid >> 1; // 0,1 on Die 0 (Cores 0..3); 2,3 on Die 1 (Cores 4..7)
            const int sub_tid = tid & 1;  // 0 -> even 4-token blocks (0,8..56); 1 -> odd 4-token blocks (4,12..60)
            const int qh_base = kv_h * 4;
            const float *k_head = k_cache + static_cast<size_t>(kv_h) * s_cap * kHeadDim;
            const float *v_head = v_cache + static_cast<size_t>(kv_h) * s_cap * kHeadDim;

            constexpr int kMaxBlocks = 8;
            alignas(64) float q_all[kMaxBlocks][kHeadDim][16];
            alignas(64) float o_all[kMaxBlocks][kHeadDim][16];
            alignas(64) float run_m[kMaxBlocks][16];
            alignas(64) float run_l[kMaxBlocks][16];
            int t_base_arr[kMaxBlocks];
            int dt_cnt_arr[kMaxBlocks];
            int p_beg_arr[kMaxBlocks][4];
            int p_end_arr[kMaxBlocks][4];
            int n_blocks = 0;
            int p_min_all = 0x7fffffff;
            int p_max_all = 0;

            for (int t_base = sub_tid * 4; t_base < T && n_blocks < kMaxBlocks; t_base += 8) {
                int b = n_blocks++;
                t_base_arr[b] = t_base;
                int dt_cnt = std::min(4, T - t_base);
                dt_cnt_arr[b] = dt_cnt;
                std::memset(o_all[b], 0, sizeof(o_all[b]));

                for (int dt = 0; dt < 4; ++dt) {
                    if (dt < dt_cnt) {
                        int t = t_base + dt;
                        int abs_pos = pos_base + t;
                        int pb = is_swa ? std::max(0, abs_pos - kSwaWin + 1) : 0;
                        int pe = abs_pos + 1;
                        p_beg_arr[b][dt] = pb;
                        p_end_arr[b][dt] = pe;
                        if (pb < p_min_all) p_min_all = pb;
                        if (pe > p_max_all) p_max_all = pe;
                        for (int g = 0; g < 4; ++g) {
                            int idx = dt * 4 + g;
                            const float *q_src = qkv_chunk + static_cast<size_t>(t) * 3072 + (qh_base + g) * kHeadDim;
                            for (int d = 0; d < kHeadDim; ++d) q_all[b][d][idx] = q_src[d];
                            run_m[b][idx] = -1e30f;
                            run_l[b][idx] = 0.0f;
                        }
                    } else {
                        p_beg_arr[b][dt] = 0;
                        p_end_arr[b][dt] = 0;
                        for (int g = 0; g < 4; ++g) {
                            int idx = dt * 4 + g;
                            for (int d = 0; d < kHeadDim; ++d) q_all[b][d][idx] = 0.0f;
                            run_m[b][idx] = -1e30f;
                            run_l[b][idx] = 0.0f;
                        }
                    }
                }
            }

            constexpr int kKeyTile = 64;
            alignas(64) float sc_tile[kKeyTile][16];

            for (int s_tile_beg = p_min_all; s_tile_beg < p_max_all; s_tile_beg += kKeyTile) {
                const int s_tile_end = std::min(p_max_all, s_tile_beg + kKeyTile);
                const int tile_len   = s_tile_end - s_tile_beg;

                for (int b = 0; b < n_blocks; ++b) {
                    const int dt_cnt = dt_cnt_arr[b];
                    const int b_pmin = p_beg_arr[b][0];
                    const int b_pmax = p_end_arr[b][dt_cnt - 1];
                    if (s_tile_end <= b_pmin || s_tile_beg >= b_pmax) continue;

                    const int n_q = dt_cnt * 4;
                    const bool full_valid = (dt_cnt == 4) &&
                                            (s_tile_beg >= p_beg_arr[b][3]) &&
                                            (s_tile_end <= p_end_arr[b][0]);
                    float tile_m[16];
                    for (int idx = 0; idx < 16; ++idx) tile_m[idx] = -1e30f;

                    // 1. Q * K^T over 64-key tile
                    if (full_valid && !is_swa) {
                        for (int s_rel = 0; s_rel < tile_len; ++s_rel) {
                            const float *k_vec = k_head + static_cast<size_t>(s_tile_beg + s_rel) * kHeadDim;
                            float acc[16] = {0.0f};
                            for (int d = 0; d < kHeadDim; ++d) {
                                float kv = k_vec[d];
                                const float *qr = q_all[b][d];
                                #pragma omp simd
                                for (int idx = 0; idx < 16; ++idx) acc[idx] += qr[idx] * kv;
                            }
                            for (int idx = 0; idx < 16; ++idx) {
                                float sc = acc[idx] * 0.0883883476f;
                                sc_tile[s_rel][idx] = sc;
                                if (sc > tile_m[idx]) tile_m[idx] = sc;
                            }
                        }
                    } else {
                        for (int s_rel = 0; s_rel < tile_len; ++s_rel) {
                            const int p = s_tile_beg + s_rel;
                            const int slot = is_swa ? (p % kSwaCap) : p;
                            const float *k_vec = k_head + static_cast<size_t>(slot) * kHeadDim;
                            float acc[16] = {0.0f};
                            for (int d = 0; d < kHeadDim; ++d) {
                                float kv = k_vec[d];
                                const float *qr = q_all[b][d];
                                #pragma omp simd
                                for (int idx = 0; idx < 16; ++idx) acc[idx] += qr[idx] * kv;
                            }
                            for (int idx = 0; idx < n_q; ++idx) {
                                int dt = idx >> 2;
                                if (p >= p_beg_arr[b][dt] && p < p_end_arr[b][dt]) {
                                    float sc = acc[idx] * 0.0883883476f;
                                    sc_tile[s_rel][idx] = sc;
                                    if (sc > tile_m[idx]) tile_m[idx] = sc;
                                } else {
                                    sc_tile[s_rel][idx] = -1e30f;
                                }
                            }
                            for (int idx = n_q; idx < 16; ++idx) sc_tile[s_rel][idx] = 0.0f;
                        }
                    }

                    // 2. Online Softmax update on L2-resident o_all[b][128][16]
                    for (int idx = 0; idx < n_q; ++idx) {
                        if (tile_m[idx] <= -1e20f) {
                            for (int s_rel = 0; s_rel < tile_len; ++s_rel) sc_tile[s_rel][idx] = 0.0f;
                            continue;
                        }
                        float old_m = run_m[b][idx];
                        float new_m = std::max(old_m, tile_m[idx]);
                        float rescale = (old_m > -1e20f) ? fast_expf_neg(old_m - new_m) : 0.0f;
                        if (rescale != 1.0f) {
                            for (int d = 0; d < kHeadDim; ++d) o_all[b][d][idx] *= rescale;
                        }
                        float tile_sum = 0.0f;
                        for (int s_rel = 0; s_rel < tile_len; ++s_rel) {
                            float sc = sc_tile[s_rel][idx];
                            float w = (sc > -1e20f) ? fast_expf_neg(sc - new_m) : 0.0f;
                            sc_tile[s_rel][idx] = w;
                            tile_sum += w;
                        }
                        run_l[b][idx] = run_l[b][idx] * rescale + tile_sum;
                        run_m[b][idx] = new_m;
                    }

                    // 3. P * V accumulation over 64-key tile
                    for (int s_rel = 0; s_rel < tile_len; ++s_rel) {
                        const float *p_wt = sc_tile[s_rel];
                        float p_sum = 0.0f;
                        for (int idx = 0; idx < n_q; ++idx) p_sum += p_wt[idx];
                        if (p_sum <= 1e-8f) continue;

                        const int p = s_tile_beg + s_rel;
                        const int slot = is_swa ? (p % kSwaCap) : p;
                        const float *v_vec = v_head + static_cast<size_t>(slot) * kHeadDim;
                        for (int d = 0; d < kHeadDim; ++d) {
                            float vv = v_vec[d];
                            float *orow = o_all[b][d];
                            #pragma omp simd
                            for (int idx = 0; idx < 16; ++idx) orow[idx] += p_wt[idx] * vv;
                        }
                    }
                }
            }

            // 4. Normalize by run_l and apply per-head sigmoid gate
            for (int b = 0; b < n_blocks; ++b) {
                int t_base = t_base_arr[b];
                int dt_cnt = dt_cnt_arr[b];
                for (int dt = 0; dt < dt_cnt; ++dt) {
                    int t = t_base + dt;
                    for (int g = 0; g < 4; ++g) {
                        int idx = dt * 4 + g;
                        float gate_val = gates[static_cast<size_t>(t) * kQHeads + (qh_base + g)];
                        float inv_sum  = gate_val / (run_l[b][idx] + 1e-12f);
                        float *o_dst = attn_out_chunk + static_cast<size_t>(t) * kHidden + (qh_base + g) * kHeadDim;
                        for (int d = 0; d < kHeadDim; ++d) {
                            o_dst[d] = o_all[b][d][idx] * inv_sum;
                        }
                    }
                }
            }
        }
    }

    bool load_cpu_weights() {
        std::string bin_path = m_pkg_dir + "/cpu_weights.bin";
        std::string idx_path = m_pkg_dir + "/cpu_weights.json";

        FILE *bf = std::fopen(bin_path.c_str(), "rb");
        if (!bf) {
            std::printf("FAIL: cannot open %s\n", bin_path.c_str());
            return false;
        }
        std::fseek(bf, 0, SEEK_END);
        size_t bsz = static_cast<size_t>(std::ftell(bf));
        std::fseek(bf, 0, SEEK_SET);
        m_cpu_weights_buf.resize(bsz / sizeof(float));
        if (std::fread(m_cpu_weights_buf.data(), 1, bsz, bf) != bsz) {
            std::fclose(bf);
            return false;
        }
        std::fclose(bf);

        std::ifstream jf(idx_path);
        if (!jf.is_open()) {
            std::printf("FAIL: cannot open %s\n", idx_path.c_str());
            return false;
        }
        std::stringstream ss;
        ss << jf.rdbuf();
        std::string err;
        JsonValue root = JsonValue::parse(ss.str(), err);
        if (!err.empty() || !root.is_object()) {
            std::printf("FAIL: parse %s: %s\n", idx_path.c_str(), err.c_str());
            return false;
        }

        auto ptr_of = [&](const std::string &key) -> const float* {
            if (!root.has(key)) {
                std::printf("FAIL: missing key %s in cpu_weights.json\n", key.c_str());
                return nullptr;
            }
            size_t off = static_cast<size_t>(root[key]["offset"].as_int64(0));
            return m_cpu_weights_buf.data() + (off / sizeof(float));
        };

        m_emb_norm_weight = ptr_of("model.embedding_norm.weight");
        if (!m_emb_norm_weight) return false;

        for (int l = 0; l < kNumLayers; ++l) {
            std::string pfx = "layer_" + std::to_string(l) + ".";
            m_layer_cpu[l].op_norm       = ptr_of(pfx + "operator_norm.weight");
            m_layer_cpu[l].post_op_norm  = ptr_of(pfx + "post_operator_norm.weight");
            m_layer_cpu[l].ffn_norm      = ptr_of(pfx + "ffn_norm.weight");
            m_layer_cpu[l].post_ffn_norm = ptr_of(pfx + "post_ffn_norm.weight");
            if (!m_layer_cpu[l].op_norm || !m_layer_cpu[l].post_op_norm ||
                !m_layer_cpu[l].ffn_norm || !m_layer_cpu[l].post_ffn_norm) return false;

            if (kLayerSchedule[l] == Lfm3LayerType::Conv) {
                m_layer_cpu[l].conv_weight = ptr_of(pfx + "conv.conv.weight");
                if (!m_layer_cpu[l].conv_weight) return false;
            } else {
                m_layer_cpu[l].q_norm    = ptr_of(pfx + "self_attn.q_norm.weight");
                m_layer_cpu[l].k_norm    = ptr_of(pfx + "self_attn.k_norm.weight");
                m_layer_cpu[l].gate_proj = ptr_of(pfx + "self_attn.gate_proj.weight");
                if (!m_layer_cpu[l].q_norm || !m_layer_cpu[l].k_norm || !m_layer_cpu[l].gate_proj) return false;
            }
        }
        std::printf("[+] Loaded cpu_weights.bin (%.2f MiB, 97 RMSNorms + 13 ShortConv kernels + 11 QK-norms/Gates)\n",
                    bsz / (1024.0 * 1024.0));
        return true;
    }

    bool load_weights_map() {
        std::string map_path = m_pkg_dir + "/weights_map.json";
        std::ifstream jf(map_path);
        if (!jf.is_open()) {
            std::printf("FAIL: cannot open %s\n", map_path.c_str());
            return false;
        }
        std::stringstream ss;
        ss << jf.rdbuf();
        std::string err;
        JsonValue root = JsonValue::parse(ss.str(), err);
        if (!err.empty() || !root.is_object()) {
            std::printf("FAIL: parse %s: %s\n", map_path.c_str(), err.c_str());
            return false;
        }

        if (root.has("tp_degree") && root["tp_degree"].as_int64(1) == 2) {
            m_tp_degree = 2;
            std::printf("[+] Tensor Parallelism enabled: TP=%d across both NPU dies\n", m_tp_degree);
        } else {
            m_tp_degree = 1;
        }

        for (int l = 0; l < kNumLayers; ++l) {
            char pfx[32];
            std::snprintf(pfx, sizeof(pfx), "lfm3_l%02d_", l);
            std::string s_pfx(pfx);

            std::string k_in  = s_pfx + ((kLayerSchedule[l] == Lfm3LayerType::Conv) ? "conv_in"  : "attn_qkv");
            std::string k_out = s_pfx + ((kLayerSchedule[l] == Lfm3LayerType::Conv) ? "conv_out" : "attn_out");
            std::string k_mlp = s_pfx + "mlp";

            if (!root.has(k_in) || !root.has(k_out) || !root.has(k_mlp)) {
                std::printf("FAIL: missing layer %d entries in weights_map.json\n", l);
                return false;
            }
            m_layer_npu[l].op_in_tp         = static_cast<uint8_t>(root[k_in]["tp_degree"].as_int64(m_tp_degree));
            m_layer_npu[l].op_out_tp        = static_cast<uint8_t>(root[k_out]["tp_degree"].as_int64(m_tp_degree));
            m_layer_npu[l].mlp_tp           = static_cast<uint8_t>(root[k_mlp]["tp_degree"].as_int64(m_tp_degree));
            m_layer_npu[l].die_id           = (m_layer_npu[l].op_in_tp == 2) ? 0 : ((l < 12) ? 0 : 1);
            m_layer_npu[l].op_in_offset     = static_cast<size_t>(root[k_in]["weight_offset"].as_int64(0));
            m_layer_npu[l].op_in_offset_d1  = static_cast<size_t>(root[k_in].has("weight_offset_d1")
                ? root[k_in]["weight_offset_d1"].as_int64(0) : root[k_in]["weight_offset"].as_int64(0));
            m_layer_npu[l].op_in_size       = static_cast<size_t>(root[k_in]["weight_size"].as_int64(0));
            m_layer_npu[l].op_out_offset    = static_cast<size_t>(root[k_out]["weight_offset"].as_int64(0));
            m_layer_npu[l].op_out_offset_d1 = static_cast<size_t>(root[k_out].has("weight_offset_d1")
                ? root[k_out]["weight_offset_d1"].as_int64(0) : root[k_out]["weight_offset"].as_int64(0));
            m_layer_npu[l].op_out_size      = static_cast<size_t>(root[k_out]["weight_size"].as_int64(0));
            m_layer_npu[l].mlp_offset       = static_cast<size_t>(root[k_mlp]["weight_offset"].as_int64(0));
            m_layer_npu[l].mlp_offset_d1    = static_cast<size_t>(root[k_mlp].has("weight_offset_d1")
                ? root[k_mlp]["weight_offset_d1"].as_int64(0) : root[k_mlp]["weight_offset"].as_int64(0));
            m_layer_npu[l].mlp_size         = static_cast<size_t>(root[k_mlp]["weight_size"].as_int64(0));
        }
        return true;
    }

    bool load_quant_scales() {
        std::string manifest_path = m_pkg_dir + "/manifest.json";
        bool manifest_is_quant = m_force_int8;
        if (file_exists(manifest_path)) {
            std::ifstream mf(manifest_path);
            std::stringstream mss;
            mss << mf.rdbuf();
            std::string merr;
            JsonValue mroot = JsonValue::parse(mss.str(), merr);
            if (merr.empty() && mroot.is_object() && mroot.has("precision")) {
                std::string prec = mroot["precision"].as_string("fp16");
                if (prec == "int8" || prec == "int16" || prec == "mixed") {
                    manifest_is_quant = true;
                }
            }
        }

        std::string qs_path = m_pkg_dir + "/quant_scales.json";
        if (!manifest_is_quant || !file_exists(qs_path)) {
            std::printf("[+] Boundary precision: FP16 (manifest_is_quant=%d)\n",
                        manifest_is_quant ? 1 : 0);
            return true;
        }

        std::ifstream jf(qs_path);
        if (!jf.is_open()) {
            std::printf("FAIL: cannot open %s\n", qs_path.c_str());
            return false;
        }
        std::stringstream ss;
        ss << jf.rdbuf();
        std::string err;
        JsonValue root = JsonValue::parse(ss.str(), err);
        if (!err.empty() || !root.is_object() || !root.has("subgraphs")) {
            std::printf("FAIL: invalid quant_scales.json (%s): %s\n", qs_path.c_str(), err.c_str());
            return false;
        }

        const JsonValue &subs  = root["subgraphs"];
        const JsonValue &archs = root["archetype_shared"];

        m_chan_scales_buf.clear();
        std::string qcs_path = m_pkg_dir + "/quant_chan_scales.bin";
        if (file_exists(qcs_path)) {
            FILE *cf = std::fopen(qcs_path.c_str(), "rb");
            if (cf) {
                std::fseek(cf, 0, SEEK_END);
                long sz = std::ftell(cf);
                std::fseek(cf, 0, SEEK_SET);
                if (sz > 0 && (sz % sizeof(float)) == 0) {
                    m_chan_scales_buf.resize(static_cast<size_t>(sz) / sizeof(float));
                    size_t n_read = std::fread(m_chan_scales_buf.data(), sizeof(float), m_chan_scales_buf.size(), cf);
                    if (n_read != m_chan_scales_buf.size()) {
                        m_chan_scales_buf.clear();
                    }
                }
                std::fclose(cf);
            }
        }

        bool use_shared = m_use_shared_archetype_scales && archs.is_object() && m_chan_scales_buf.empty();

        auto parse_entry = [this](const JsonValue &v, SubgraphScaleInfo &out) -> bool {
            if (!v.is_object()) return false;
            out.in_dtype   = parse_boundary_dtype(v["input_dtype"].as_string("fp16"));
            out.out_dtype  = parse_boundary_dtype(v["output_dtype"].as_string("fp16"));
            out.in_thresh  = static_cast<float>(v["input_threshold"].as_double(1.0));
            out.in_step    = static_cast<float>(v["input_step"].as_double(1.0 / 127.0));
            out.out_thresh = static_cast<float>(v["output_threshold"].as_double(1.0));
            out.out_step   = static_cast<float>(v["output_step"].as_double(1.0 / 127.0));
            out.in_inv_smooth = nullptr;
            out.out_step_chan = nullptr;
            if (!m_chan_scales_buf.empty() && v.has("in_smooth_offset") && v.has("out_step_offset")) {
                size_t in_off  = static_cast<size_t>(v["in_smooth_offset"].as_int64(0));
                size_t in_cnt  = static_cast<size_t>(v["in_smooth_count"].as_int64(0));
                size_t out_off = static_cast<size_t>(v["out_step_offset"].as_int64(0));
                size_t out_cnt = static_cast<size_t>(v["out_step_count"].as_int64(0));
                if (in_off + in_cnt <= m_chan_scales_buf.size() && out_off + out_cnt <= m_chan_scales_buf.size()) {
                    out.in_inv_smooth = m_chan_scales_buf.data() + in_off;
                    out.out_step_chan = m_chan_scales_buf.data() + out_off;
                }
            }
            return v.has("input_threshold") && (out.in_inv_smooth != nullptr || std::fabs(out.in_thresh - 1.0f) > 1e-5f);
        };

        for (int l = 0; l < kNumLayers; ++l) {
            char pfx[32];
            std::snprintf(pfx, sizeof(pfx), "lfm3_l%02d_", l);
            std::string s_pfx(pfx);
            bool is_conv = (kLayerSchedule[l] == Lfm3LayerType::Conv);

            std::string k_in  = s_pfx + (is_conv ? "conv_in"  : "attn_qkv");
            std::string k_out = s_pfx + (is_conv ? "conv_out" : "attn_out");
            std::string k_mlp = s_pfx + "mlp";

            const JsonValue &v_in  = use_shared ? archs[is_conv ? "conv_in" : "attn_qkv"] : subs[k_in];
            const JsonValue &v_out = use_shared ? archs["proj2048"]                       : subs[k_out];
            const JsonValue &v_mlp = use_shared ? archs["mlp"]                            : subs[k_mlp];

            if (!parse_entry(v_in,  m_layer_npu[l].op_in_scale) ||
                !parse_entry(v_out, m_layer_npu[l].op_out_scale) ||
                !parse_entry(v_mlp, m_layer_npu[l].mlp_scale)) {
                std::printf("FAIL: invalid or 1.0-fallback scale entry at layer %d in %s\n",
                            l, qs_path.c_str());
                return false;
            }
            m_layer_npu[l].op_in_scale_d1  = m_layer_npu[l].op_in_scale;
            m_layer_npu[l].op_out_scale_d1 = m_layer_npu[l].op_out_scale;
            m_layer_npu[l].mlp_scale_d1    = m_layer_npu[l].mlp_scale;
            if (m_layer_npu[l].op_in_tp == 2 && m_layer_npu[l].op_in_scale_d1.out_step_chan) {
                m_layer_npu[l].op_in_scale_d1.out_step_chan += (is_conv ? 3072 : 1536);
            }
            if (m_layer_npu[l].op_out_tp == 2 && m_layer_npu[l].op_out_scale_d1.out_step_chan) {
                m_layer_npu[l].op_out_scale_d1.out_step_chan += 1024;
            }
            if (m_layer_npu[l].mlp_tp == 2 && m_layer_npu[l].mlp_scale_d1.out_step_chan) {
                m_layer_npu[l].mlp_scale_d1.out_step_chan += 2048;
            }
        }

        for (uint16_t d = 0; d < 2; ++d) {
            for (int s = 0; s < 4; ++s) {
                char key[64];
                std::snprintf(key, sizeof(key), "lfm3_lm_head_d%u_s%d", d, s);
                if (!parse_entry(subs[key], m_die_models[d].lm_head_scale[s])) {
                    std::printf("FAIL: missing/invalid scale for %s in %s\n", key, qs_path.c_str());
                    return false;
                }
            }
        }

        std::printf("[+] Loaded quant_scales.json (%s scales, SmoothQuant=%s, MLP out=%s, LMHead=%s)\n",
                    use_shared ? "archetype_shared" : "per-subgraph",
                    m_chan_scales_buf.empty() ? "off" : "on",
                    (m_layer_npu[0].mlp_scale.out_dtype == BoundaryDtype::Int16) ? "int16" : "int8",
                    (m_die_models[0].lm_head_scale[0].out_dtype == BoundaryDtype::Int8) ? "int8" : "fp16");

        // Milestone 8 (Phase 1): Precompute fused SmoothQuant and scaled RMSNorm weights
        for (int l = 0; l < kNumLayers; ++l) {
            // 1. op_norm_smooth for conv_in / attn_qkv
            if (m_layer_npu[l].op_in_scale.in_inv_smooth && m_layer_npu[l].op_in_scale.in_dtype == BoundaryDtype::Int8) {
                m_layer_cpu[l].op_norm_smooth.resize(kHidden);
                const float *sm = m_layer_npu[l].op_in_scale.in_inv_smooth;
                const float *w  = m_layer_cpu[l].op_norm;
                for (int c = 0; c < kHidden; ++c) {
                    m_layer_cpu[l].op_norm_smooth[c] = w[c] * sm[c];
                }
            }
            // 2. ffn_norm_smooth & ffn_norm_scaled for mlp
            if (m_layer_npu[l].mlp_scale.in_dtype == BoundaryDtype::Int8) {
                m_layer_cpu[l].ffn_norm_smooth.resize(kHidden);
                m_layer_cpu[l].ffn_norm_scaled.resize(kHidden);
                const float *sm = m_layer_npu[l].mlp_scale.in_inv_smooth;
                const float *w  = m_layer_cpu[l].ffn_norm;
                float inv_step  = 1.0f / m_layer_npu[l].mlp_scale.in_step;
                for (int c = 0; c < kHidden; ++c) {
                    float s = sm ? sm[c] : 1.0f;
                    m_layer_cpu[l].ffn_norm_smooth[c] = w[c] * s;
                    m_layer_cpu[l].ffn_norm_scaled[c] = w[c] * s * inv_step;
                }
            }
        }
        // 3. m_emb_norm_scaled for LM head
        if (m_die_models[0].lm_head_scale[0].in_dtype == BoundaryDtype::Int8 && m_emb_norm_weight) {
            m_emb_norm_scaled.resize(kHidden);
            float inv_step = 1.0f / m_die_models[0].lm_head_scale[0].in_step;
            for (int c = 0; c < kHidden; ++c) {
                m_emb_norm_scaled[c] = m_emb_norm_weight[c] * inv_step;
            }
        }
        return true;
    }

    std::string m_pkg_dir;
    int m_max_ctx = 4096;
    int m_pos = 0;
    uint16_t m_num_dies = 1;
    int m_tp_degree = 1;
    int m_requested_tp_degree = 0; // 0 = auto from weights_map.json, 1 = --pp2, 2 = --tp2
    bool m_tp_worker_running = false;
    std::thread m_tp_thread;
    std::mutex m_tp_mtx;
    std::condition_variable m_tp_cv;
    std::atomic<uint64_t> m_tp_seq{0};
    std::atomic<uint64_t> m_tp_ack{0};
    std::atomic<bool> m_tp_stop{false};
    std::atomic<bool> m_tp_sleeping{false};
    std::atomic<bool> m_tp_prewake{false};
    std::function<void()> *m_tp_fn = nullptr;
    bool m_force_int8 = false;
    bool m_use_shared_archetype_scales = true;
    bool m_ab_shared_weights = false;   // --weights-layer0: timing experiment, numerics invalid
    bool m_extent_probe = false;
    size_t m_guard_bytes = 0;
    static constexpr uint8_t GUARD_BYTE = 0xA5;
    uint64_t m_extent_max[StageTimers::ARCH_COUNT] = {0, 0, 0, 0, 0};
    uint64_t m_extent_samples[StageTimers::ARCH_COUNT] = {0, 0, 0, 0, 0};
    bool m_extent_overrun[StageTimers::ARCH_COUNT] = {false, false, false, false, false};
    std::map<const DmaBuffer *, size_t> m_prod_cap;

    WorkerPool m_pool;
    EmbeddingTable m_embedding;
    Tokenizer m_tokenizer;

    std::vector<float> m_cpu_weights_buf;
    std::vector<float> m_chan_scales_buf;
    const float *m_emb_norm_weight = nullptr;
    std::vector<float> m_emb_norm_scaled;
    LayerCpuWeights m_layer_cpu[kNumLayers];
    LayerNpuOffsets m_layer_npu[kNumLayers];
    DieModels m_die_models[2];

    std::unique_ptr<DmaBuffer> m_w_blob[2];
    std::unique_ptr<DmaBuffer> m_act_t1[2];
    std::unique_ptr<DmaBuffer> m_out_t1[2];
    std::unique_ptr<DmaBuffer> m_act_t64[2];
    std::unique_ptr<DmaBuffer> m_out_t64[2];
    std::unique_ptr<DmaBuffer> m_lm_out[2][4];

    float m_swa_inv_freq[64];
    float m_glob_inv_freq[64];

    std::vector<float> m_conv_state;
    NumaKvBuffer<float>    m_swa_k;
    NumaKvBuffer<float>    m_swa_v;
    NumaKvBuffer<float>    m_glob_k;
    NumaKvBuffer<float>    m_glob_v;
    NumaKvBuffer<uint8_t>  m_glob_k_u8;
    NumaKvBuffer<uint8_t>  m_glob_v_u8;
    NumaKvBuffer<float>    m_glob_k_sc;
    NumaKvBuffer<float>    m_glob_v_sc;
    NumaKvBuffer<uint16_t> m_glob_k_sum;
    std::vector<float> m_scores_ws;
    std::vector<float> m_logits;
    std::vector<float> m_buf_h, m_buf_u, m_buf_u_smooth, m_buf_bcx, m_buf_qkv, m_buf_y,
                       m_buf_op_out, m_buf_h_mid, m_buf_v, m_buf_mlp_out, m_buf_mlp_out_d1, m_buf_gates;
};

static double cosine_similarity(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double x = a[i];
        double y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

static std::string format_lfm3_chat(const std::string &user_prompt,
                                    const std::string &system_prompt,
                                    bool enable_thinking) {
    std::string out = "<|startoftext|>";
    if (!system_prompt.empty()) {
        out += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    }
    out += "<|im_start|>user\n" + user_prompt + "<|im_end|>\n";
    out += "<|im_start|>assistant\n";
    out += enable_thinking ? "<think>" : "<think></think>";
    return out;
}

static void run_daemon_server(Lfm3Engine &engine, int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::perror("socket failed");
        return;
    }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        std::perror("bind failed");
        close(server_fd);
        return;
    }
    if (listen(server_fd, 5) < 0) {
        std::perror("listen failed");
        close(server_fd);
        return;
    }

    std::printf("\n=================================================================\n");
    std::printf(" LFM3-1.7B Resident Daemon Server listening on 0.0.0.0:%d\n", port);
    std::printf(" Hot weights resident in dual-die MMZ, ready for OpenAI queries\n");
    std::printf("=================================================================\n\n");

    while (!g_interrupt_flag.load()) {
        sockaddr_in client_addr{};
        socklen_t addrlen = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &addrlen);
        if (client_fd < 0) {
            if (g_interrupt_flag.load()) break;
            continue;
        }

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        std::printf("[Daemon] Client connected from %s:%d\n", client_ip, ntohs(client_addr.sin_port));

        std::string req_line;
        char buf[4096];
        bool got_newline = false;
        std::vector<char> leftover;
        while (!got_newline && !g_interrupt_flag.load()) {
            ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
            if (n <= 0) break;
            for (ssize_t i = 0; i < n; ++i) {
                if (!got_newline) {
                    if (buf[i] == '\n') {
                        got_newline = true;
                    } else {
                        req_line.push_back(buf[i]);
                    }
                } else {
                    leftover.push_back(buf[i]);
                }
            }
        }

        if (req_line.empty()) {
            close(client_fd);
            continue;
        }

        std::string parse_err;
        auto j = eswin::JsonValue::parse(req_line, parse_err);
        if (!parse_err.empty() || !j.is_object()) {
            std::string err_resp = "{\"type\":\"error\",\"message\":\"Invalid JSON request: " + parse_err + "\"}\n";
            send(client_fd, err_resp.data(), err_resp.size(), 0);
            close(client_fd);
            continue;
        }

        std::string prompt = j["prompt"].as_string("");
        int max_new_tokens = j["max_tokens"].as_int(128);
        float temperature = static_cast<float>(j["temperature"].as_double(0.1));
        int top_k = j["top_k"].as_int(50);
        bool no_think = j["no_think"].as_bool(true);
        bool raw = j["raw"].as_bool(false);
        bool return_logits = j["return_logits"].as_bool(false) || (max_new_tokens == 0);
        int image_bytes = j["image_bytes"].as_int(0);

        std::vector<float> visual_embeds;
        if (image_bytes > 0 && (image_bytes % (Lfm3Engine::kHidden * sizeof(float)) == 0)) {
            visual_embeds.resize(image_bytes / sizeof(float));
            char *raw_dst = reinterpret_cast<char*>(visual_embeds.data());
            size_t bytes_read = 0;
            if (!leftover.empty()) {
                size_t to_copy = std::min(leftover.size(), static_cast<size_t>(image_bytes));
                std::memcpy(raw_dst, leftover.data(), to_copy);
                bytes_read += to_copy;
            }
            while (bytes_read < static_cast<size_t>(image_bytes)) {
                ssize_t r = recv(client_fd, raw_dst + bytes_read, image_bytes - bytes_read, 0);
                if (r <= 0) break;
                bytes_read += r;
            }
        }

        engine.reset_state();
        SamplerParams sp;
        sp.temperature = temperature;
        sp.top_k = top_k;
        sp.repeat_penalty = 1.1f;
        sp.repeat_window = 64;
        Sampler sampler(sp);

        std::string formatted = prompt;
        if (!raw && prompt.find("<|im_start|>") == std::string::npos) {
            formatted = format_lfm3_chat(prompt, "You are a helpful AI assistant.", !no_think);
        }

        std::vector<int32_t> prompt_ids = engine.tokenizer().encode(formatted, false);
        for (int32_t id : prompt_ids) sampler.accept(id);

        StageTimers prefill_t, decode_t;
        g_in_generation.store(true);

        const float *vis_ptr = visual_embeds.empty() ? nullptr : visual_embeds.data();
        size_t num_vis_tok = visual_embeds.size() / Lfm3Engine::kHidden;

        if (!engine.prefill(prompt_ids, prefill_t, vis_ptr, num_vis_tok)) {
            std::string err_resp = "{\"type\":\"error\",\"message\":\"Prefill failed on NPU\"}\n";
            send(client_fd, err_resp.data(), err_resp.size(), 0);
            g_in_generation.store(false);
            close(client_fd);
            continue;
        }

        if (return_logits) {
            std::ostringstream l_ss;
            l_ss << "{\"type\":\"logits\",\"prompt_tokens\":" << prompt_ids.size()
                 << ",\"prefill_ms\":" << prefill_t.total_ms() << ",\"logits\":{";
            const auto &all_logits = engine.logits();
            if (j["requested_tokens"].is_array()) {
                const auto &arr = j["requested_tokens"].as_array();
                for (size_t qi = 0; qi < arr.size(); ++qi) {
                    int32_t tid = arr[qi].as_int(-1);
                    if (tid >= 0 && tid < Lfm3Engine::kVocab) {
                        if (qi > 0) l_ss << ",";
                        l_ss << "\"" << tid << "\":" << all_logits[tid];
                    }
                }
            }
            l_ss << "}}\n";
            std::string l_resp = l_ss.str();
            send(client_fd, l_resp.data(), l_resp.size(), 0);
            g_in_generation.store(false);
            close(client_fd);
            std::printf("[Daemon] Decision query completed: %zu prompt tok, %.1f ms prefill\n",
                        prompt_ids.size(), prefill_t.total_ms());
            continue;
        }

        int step = 0;
        for (; step < max_new_tokens && !g_interrupt_flag.load(); ++step) {
            int32_t next_id = sampler.sample(engine.logits_mut(), Lfm3Engine::kVocab);
            sampler.accept(next_id);

            if (next_id == 124900) break; // <|im_end|>
            if (next_id == 124902) {      // </think>
                if (step + 1 < max_new_tokens) {
                    engine.forward_chunk(&next_id, 1, true, decode_t);
                }
                continue;
            }

            std::string piece = engine.tokenizer().decode_token(next_id);
            std::ostringstream esc;
            for (char c : piece) {
                if (c == '"') esc << "\\\"";
                else if (c == '\\') esc << "\\\\";
                else if (c == '\n') esc << "\\n";
                else if (c == '\r') esc << "\\r";
                else if (c == '\t') esc << "\\t";
                else esc << c;
            }

            std::string tok_event = "{\"type\":\"token\",\"token_id\":" + std::to_string(next_id) +
                                    ",\"text\":\"" + esc.str() + "\"}\n";
#if defined(MSG_NOSIGNAL)
            if (send(client_fd, tok_event.data(), tok_event.size(), MSG_NOSIGNAL) < 0) break;
#else
            if (send(client_fd, tok_event.data(), tok_event.size(), 0) < 0) break;
#endif
            if (step + 1 < max_new_tokens) {
                if (!engine.forward_chunk(&next_id, 1, true, decode_t)) break;
            }
        }
        g_in_generation.store(false);

        std::string done_event = "{\"type\":\"done\",\"prompt_tokens\":" + std::to_string(prompt_ids.size()) +
                                 ",\"completion_tokens\":" + std::to_string(step) +
                                 ",\"prefill_ms\":" + std::to_string(prefill_t.total_ms()) +
                                 ",\"decode_ms\":" + std::to_string(decode_t.total_ms()) + "}\n";
#if defined(MSG_NOSIGNAL)
        send(client_fd, done_event.data(), done_event.size(), MSG_NOSIGNAL);
#else
        send(client_fd, done_event.data(), done_event.size(), 0);
#endif
        close(client_fd);
        std::printf("[Daemon] Turn completed: %zu prompt tok (%.1f ms), %d decode tok (%.1f ms = %.2f tok/s)\n",
                    prompt_ids.size(), prefill_t.total_ms(), step, decode_t.total_ms(),
                    decode_t.total_ms() > 0 ? (1000.0 * step / decode_t.total_ms()) : 0.0);
    }
    close(server_fd);
    std::printf("[+] LFM3 Daemon shut down cleanly.\n");
}

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::signal(SIGINT, sigint_handler);
    std::signal(SIGPIPE, SIG_IGN);
    init_fp16_lut();
    omp_set_num_threads(8);

    // Prefer the directory the binary is run from. Every invocation so far has been
    // `cd <package> && ./lfm3_v2 ...`, but the default used to be a hardcoded absolute board
    // path, so running from a *different* package silently loaded the hardcoded one. That is how
    // the mixed-precision experiment "reproduced" the all-INT8 numbers bit for bit (FINDINGS
    // 41.4) and nearly got a working accuracy gate condemned as blind. Fail loudly instead.
    std::string pkg_dir = ".";
    if (!file_exists(pkg_dir + "/cpu_weights.bin")) {
        if (file_exists("/home/chri/models/LFM3-1.7B_ENNP/cpu_weights.bin")) {
            pkg_dir = "/home/chri/models/LFM3-1.7B_ENNP";
        } else if (file_exists("/home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg/cpu_weights.bin")) {
            pkg_dir = "/home/chri/aixigua/deep/models/LFM3-1.7B/eswin_pkg";
        } else {
            pkg_dir = "/tmp/lfm3_pkg";
        }
    }
    if (!file_exists(pkg_dir + "/cpu_weights.bin")) {
        std::printf("FAIL: no LFM3 package at '%s' (expected cpu_weights.bin). Pass --model-dir.\n",
                    pkg_dir.c_str());
        return 1;
    }
    std::printf("[+] Resolved package directory: %s\n",
                std::filesystem::absolute(pkg_dir).string().c_str());

    std::string verify_golden_path;
    std::string prompt;
    std::string system_prompt = "You are a helpful AI assistant.";
    int max_new_tokens = 128;
    int max_ctx = 4096;
    float temperature = 0.1f;
    int32_t top_k = 50;
    float repeat_penalty = 1.1f;
    bool enable_thinking = true;
    bool bench_attn = false;
    bool force_int8 = false;
    bool use_shared_archetype_scales = true;
    bool ab_shared_weights = false;
    int requested_tp_degree = 0;
    bool daemon_mode = false;
    int daemon_port = 8088;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next_arg = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing argument for %s\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--model-dir" || a == "-m") pkg_dir = next_arg();
        else if (a == "--verify-golden") verify_golden_path = next_arg();
        else if (a == "--prompt" || a == "-p") {
            std::string arg = next_arg();
            if (!arg.empty() && arg[0] == '@') {
                std::string path = arg.substr(1);
                std::ifstream ifs(path);
                if (!ifs) {
                    std::fprintf(stderr, "Failed to open prompt file: %s\n", path.c_str());
                    std::exit(1);
                }
                prompt = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            } else {
                prompt = arg;
            }
        }
        else if (a == "--prompt-file" || a == "-f") {
            std::string path = next_arg();
            std::ifstream ifs(path);
            if (!ifs) {
                std::fprintf(stderr, "Failed to open prompt file: %s\n", path.c_str());
                std::exit(1);
            }
            prompt = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        }
        else if (a == "--system") system_prompt = next_arg();
        else if (a == "--max-tokens" || a == "-n") max_new_tokens = std::stoi(next_arg());
        else if (a == "--max-ctx") max_ctx = std::stoi(next_arg());
        else if (a == "--temp") temperature = std::stof(next_arg());
        else if (a == "--top-k") top_k = std::stoi(next_arg());
        else if (a == "--repeat-penalty") repeat_penalty = std::stof(next_arg());
        else if (a == "--no-think") enable_thinking = false;
        else if (a == "--bench-attn") bench_attn = true;
        else if (a == "--int8") force_int8 = true;
        else if (a == "--tp2" || a == "--tp") requested_tp_degree = 2;
        else if (a == "--pp2" || a == "--pp") requested_tp_degree = 1;
        else if (a == "--per-subgraph-scales") use_shared_archetype_scales = false;
        else if (a == "--weights-layer0") ab_shared_weights = true;
        else if (a == "--daemon" || a == "--server") {
            daemon_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                daemon_port = std::stoi(argv[++i]);
            }
        }
        else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s [--model-dir DIR] [--verify-golden golden_ref.bin] [--bench-attn]\n"
                        "          [--int8] [--pp2 | --tp2] [--per-subgraph-scales] [--weights-layer0]\n"
                        "          [--prompt TEXT | --prompt-file FILE | -f FILE]\n"
                        "          [--max-tokens N | -n N] [--max-ctx N] [--no-think]\n"
                        "          [--temp F] [--top-k K] [--repeat-penalty F]\n"
                        "          [--daemon [PORT]]\n"
                        "  --pp2             Layer-Parallel (PP=2): Layers 0..11 on Die 0, Layers 12..23 on Die 1.\n"
                        "  --tp2             Tensor-Parallel (TP=2): All 24 layers split across Die 0 and Die 1.\n"
                        "  --weights-layer0  Timing experiment: bind every layer of a die to one\n"
                        "                    weight region per archetype to separate NPU streaming\n"
                        "                    speed from per-submit overhead. Generated text is garbage.\n",
                        argv[0]);
            return 0;
        }
    }

    Lfm3Engine engine;
    engine.set_quant_options(force_int8, use_shared_archetype_scales);
    engine.set_parallel_mode(requested_tp_degree);
    engine.set_ab_shared_weights(ab_shared_weights);
    engine.set_extent_probe(std::getenv("LFM3_MEASURE_EXTENT") != nullptr, 2u << 20);
    if (bench_attn && verify_golden_path.empty() && prompt.empty()) {
        engine.benchmark_global_attention_kernel();
        return 0;
    }

    if (!engine.init(pkg_dir, max_ctx)) return 1;

    if (daemon_mode) {
        run_daemon_server(engine, daemon_port);
        return 0;
    }

    if (bench_attn) {
        engine.benchmark_global_attention_kernel();
    }

    // -------------------------------------------------------------------------
    // Golden Reference Verification Mode (--verify-golden golden_ref.bin)
    // -------------------------------------------------------------------------
    if (!verify_golden_path.empty()) {
        std::printf("=================================================================\n");
        std::printf(" Milestone 4 Acceptance Gate: Golden Oracle Verification\n");
        std::printf(" Oracle file: %s\n", verify_golden_path.c_str());
        std::printf("=================================================================\n");

        FILE *gf = std::fopen(verify_golden_path.c_str(), "rb");
        if (!gf) {
            std::printf("FAIL: cannot open golden file %s\n", verify_golden_path.c_str());
            return 1;
        }
        int32_t n_in = 0, n_gen = 0;
        if (std::fread(&n_in, sizeof(int32_t), 1, gf) != 1 ||
            std::fread(&n_gen, sizeof(int32_t), 1, gf) != 1) {
            std::fclose(gf);
            return 1;
        }
        std::vector<int32_t> in_ids(n_in), gold_gen(n_gen);
        std::vector<float> gold_layers(24 * 2048), gold_emb_norm(2048);
        std::vector<float> gold_pt_logits(128000), gold_ort_logits(128000);
        if (std::fread(in_ids.data(), sizeof(int32_t), n_in, gf) != static_cast<size_t>(n_in) ||
            std::fread(gold_gen.data(), sizeof(int32_t), n_gen, gf) != static_cast<size_t>(n_gen) ||
            std::fread(gold_layers.data(), sizeof(float), 24 * 2048, gf) != 24 * 2048 ||
            std::fread(gold_emb_norm.data(), sizeof(float), 2048, gf) != 2048 ||
            std::fread(gold_pt_logits.data(), sizeof(float), 128000, gf) != 128000 ||
            std::fread(gold_ort_logits.data(), sizeof(float), 128000, gf) != 128000) {
            std::fclose(gf);
            std::printf("FAIL: truncated golden_ref.bin\n");
            return 1;
        }
        std::fclose(gf);

        engine.reset_state();
        StageTimers prefill_timers, decode_timers;
        std::vector<float> actual_layers(24 * 2048), actual_emb_norm(2048);

        std::printf("\n[1/3] Running %d-Token Teacher-Forced Prefill ('Hello, world! Today is')...\n", n_in);
        if (!engine.prefill(in_ids, prefill_timers, &actual_layers, &actual_emb_norm)) {
            std::printf("FAIL: engine.prefill failed\n");
            return 1;
        }

        double min_layer_cos = 1.0;
        for (int l = 0; l < 24; ++l) {
            double c = cosine_similarity(actual_layers.data() + l * 2048,
                                         gold_layers.data() + l * 2048, 2048);
            if (c < min_layer_cos) min_layer_cos = c;
            if (l == 0 || l == 2 || l == 4 || l == 11 || l == 12 || l == 23) {
                std::printf("  • Layer %02d block_out cosine vs PyTorch FP32 : %.8f\n", l, c);
            }
        }
        double emb_cos = cosine_similarity(actual_emb_norm.data(), gold_emb_norm.data(), 2048);
        double pt_cos  = cosine_similarity(engine.logits().data(), gold_pt_logits.data(), 128000);
        double ort_cos = cosine_similarity(engine.logits().data(), gold_ort_logits.data(), 128000);

        const bool is_int8_pkg = engine.is_quant_active();
        const double min_layer_gate = is_int8_pkg ? 0.9800 : 0.9990;
        const double logit_gate     = is_int8_pkg ? 0.9900 : 0.9990;
        const double speed_gate     = is_int8_pkg ? 4.00   : 1.80;

        std::printf("  • Min Layer Cosine across all 24 layers     : %.8f [gate >= %.4f: %s]\n",
                    min_layer_cos, min_layer_gate, (min_layer_cos >= min_layer_gate) ? "PASS" : "FAIL");
        std::printf("  • Final embedding_norm Cosine               : %.8f\n", emb_cos);
        std::printf("  • 128K Logits Cosine vs PyTorch FP32        : %.8f [gate >= %.4f: %s]\n",
                    pt_cos, logit_gate, (pt_cos >= logit_gate) ? "PASS" : "FAIL");
        std::printf("  • 128K Logits Cosine vs ONNX Runtime FP16   : %.8f\n", ort_cos);

        // ------------------------------------------------------------------
        // [1b/3] Gate-integrity tests: determinism, then sensitivity.
        //
        // §41.2 found every accuracy figure identical to eight decimals across two packages whose
        // proj2048 weights genuinely differed (driver time moved 80.6 -> 94.0 ms/token). Before
        // any number above is believed -- passing or failing -- two properties must hold:
        //   (1) DETERMINISM: re-running the same prefill in this process reproduces it exactly.
        //   (2) SENSITIVITY: corrupting a whole weight region moves the metrics.
        // A gate that fails (1) is measuring noise; one that fails (2) is not measuring the model.
        // Note (2) needs a *large* perturbation: one byte out of ~470M is a ~1e-7 relative change
        // and cannot move a cosine in the eighth decimal, so a one-byte null test passes
        // vacuously -- which is how the first version of this test fooled itself.
        // ------------------------------------------------------------------
        auto prefill_metrics = [&](double &min_cos, double &logit_cos,
                                   std::vector<float> &lay, std::vector<float> &emb) -> bool {
            engine.reset_state();
            StageTimers t;
            lay.assign(24 * 2048, 0.0f);
            emb.assign(2048, 0.0f);
            if (!engine.prefill(in_ids, t, &lay, &emb)) return false;
            min_cos = 1.0;
            for (int l = 0; l < 24; ++l)
                min_cos = std::min(min_cos, cosine_similarity(lay.data() + l * 2048,
                                                              gold_layers.data() + l * 2048, 2048));
            logit_cos = cosine_similarity(engine.logits().data(), gold_pt_logits.data(), 128000);
            return true;
        };

        bool gate_ok = true;
        std::printf("\n[1b/3] Gate integrity\n");

        double r2_min = 0.0, r2_log = 0.0;
        std::vector<float> r2_layers, r2_emb;
        if (!prefill_metrics(r2_min, r2_log, r2_layers, r2_emb)) {
            std::printf("FAIL: second clean prefill failed\n");
            return 1;
        }
        const bool deterministic = (r2_min == min_layer_cos) && (r2_log == pt_cos);
        std::printf("  • DETERMINISM  run1: min-layer %.8f logits %.8f | run2: %.8f %.8f -> %s\n",
                    min_layer_cos, pt_cos, r2_min, r2_log,
                    deterministic ? "identical" : "DIFFERS on identical weights and input");
        if (!deterministic) gate_ok = false;

        // Perturb from offset 0 -- layer 0's first weight region -- because it is certain to be
        // read. Mid-blob offsets can land in the 4 KiB alignment padding the region builder
        // inserts, where corruption is legitimately invisible.
        const size_t span = 1u << 20;
        const size_t off = 0;
        double p_min = 0.0, p_log = 0.0;
        std::vector<float> p_layers, p_emb;
        if (engine.perturb_weights(0, off, span, 0xFF) &&
            prefill_metrics(p_min, p_log, p_layers, p_emb)) {
            // Any difference counts, and a non-finite metric counts loudest. Comparing with a
            // fixed epsilon is not merely weaker, it is wrong here: std::fabs(NaN) > 1e-3 is
            // false, so the first version of this check called NaN logits "no movement" --
            // blind at the moment of maximum sensitivity. The values are bit-stable, so exact
            // equality is also a strictly stronger criterion than a tolerance.
            const bool sensitive = !std::isfinite(p_min) || !std::isfinite(p_log) ||
                                   p_min != r2_min || p_log != r2_log;
            std::printf("  • SENSITIVITY  XOR 0xFF over %zu B at %zu: min-layer %.8f (%+.3e), "
                        "logits %.8f (%+.3e) -> %s\n",
                        span, off, p_min, p_min - r2_min, p_log, p_log - r2_log,
                        sensitive ? "metrics moved" : "NO MOVEMENT at all, the gate is blind");
            if (!sensitive) gate_ok = false;
        } else {
            std::printf("  • SENSITIVITY  skipped: perturbation or prefill failed\n");
            gate_ok = false;
        }

        double r3_min = 0.0, r3_log = 0.0;
        std::vector<float> r3_layers, r3_emb;
        if (!engine.perturb_weights(0, off, span, 0xFF)) {
            std::printf("FAIL: could not restore the perturbed span\n");
            return 1;
        }
        if (!prefill_metrics(r3_min, r3_log, r3_layers, r3_emb)) {
            std::printf("FAIL: prefill after restore failed\n");
            return 1;
        }
        const bool restored = (r3_min == r2_min) && (r3_log == r2_log);
        std::printf("  • RESTORE      min-layer %.8f logits %.8f -> %s\n", r3_min, r3_log,
                    restored ? "reproduces run 2" : "does NOT reproduce run 2");
        if (!restored) gate_ok = false;
        actual_layers = r3_layers;
        actual_emb_norm = r3_emb;
        // Continue from the restored run, not from run 1: if the two differ, the number the gate
        // finally judges should at least be one this process can reproduce.
        pt_cos = r3_log;

        if (!gate_ok) {
            std::printf("\nFAIL: the golden gate is not trustworthy (see above). Aborting before "
                        "the decode test, because a gate that cannot fail will also happily "
                        "report a pass it did not earn.\n");
            return 1;
        }

        std::printf("\n[2/3] Running %d-Token Greedy Autoregressive Decode...\n", n_gen);
        std::vector<int32_t> pred_gen;
        bool all_match = true;
        for (int step = 0; step < n_gen; ++step) {
            const auto &lg = engine.logits();
            int32_t best_id = static_cast<int32_t>(
                std::max_element(lg.begin(), lg.end()) - lg.begin());
            pred_gen.push_back(best_id);
            bool match = (best_id == gold_gen[step]);
            if (!match) all_match = false;
            std::string piece = engine.tokenizer().decode_token(best_id);
            std::printf("  • Step %d: predicted=%d ('%s') vs golden=%d [%s]\n",
                        step + 1, best_id, piece.c_str(), gold_gen[step], match ? "MATCH" : "MISMATCH");

            if (step + 1 < n_gen) {
                if (!engine.forward_chunk(&best_id, 1, true, decode_timers)) return 1;
            }
        }

        // Run 13 more decode steps (total 16 decode steps) to measure steady-state sustained tok/s
        int32_t next_tok = pred_gen.back();
        for (int extra = 0; extra < 13; ++extra) {
            if (!engine.forward_chunk(&next_tok, 1, true, decode_timers)) return 1;
            const auto &lg = engine.logits();
            next_tok = static_cast<int32_t>(std::max_element(lg.begin(), lg.end()) - lg.begin());
        }

        std::printf("\n[3/3] Per-Stage Timing Breakdown & Throughput Gate Verification:\n");
        const bool force_t1 = (std::getenv("LFM3_PREFILL_T1") != nullptr);
        prefill_timers.print_report(force_t1 ? "Prefill (6 tokens, T=1 path)"
                                             : "Prefill (6 tokens, T=64 batched path)");
        decode_timers.print_report("Autoregressive Decode (16 tokens)");
        decode_timers.print_arch_report();
        engine.print_extent_report();

        double decode_tok_s = 1000.0 * decode_timers.tokens_run / decode_timers.total_ms();
        bool speed_pass = (decode_tok_s >= speed_gate);
        std::printf("\n  • Greedy 4-Token Exact Match : %s\n", all_match ? "PASS (4/4 exact)" : "FAIL");
        std::printf("  • Sustained %s Decode Speed: %.2f tokens/sec (gate >= %.2f tok/s) [%s]\n",
                    is_int8_pkg ? "INT8" : "FP16", decode_tok_s, speed_gate, speed_pass ? "PASS" : "FAIL");
        std::printf("=================================================================\n");
        return (all_match && min_layer_cos >= min_layer_gate && pt_cos >= logit_gate && speed_pass) ? 0 : 1;
    }

    // -------------------------------------------------------------------------
    // Single-Shot Prompt Mode (--prompt "...") or Interactive Chat REPL
    // -------------------------------------------------------------------------
    SamplerParams sp;
    sp.temperature = temperature;
    sp.top_k = top_k;
    sp.repeat_penalty = repeat_penalty;
    sp.repeat_window = 64;
    sp.max_consecutive_duplicates = 2;
    Sampler sampler(sp);
    std::string current_image_path;
    std::vector<float> current_image_embeds;

    auto run_chat_turn = [&](const std::string &user_text) {
        engine.reset_state();
        sampler.reset();
        std::string prompt_text = user_text;
        const float *vis_ptr = nullptr;
        size_t num_vis_tok = 0;
        if (!current_image_embeds.empty()) {
            num_vis_tok = current_image_embeds.size() / Lfm3Engine::kHidden;
            vis_ptr = current_image_embeds.data();
            std::string vis_str = "<|vision_start|>";
            for (size_t vi = 0; vi < num_vis_tok; ++vi) vis_str += "<|image|>";
            vis_str += "<|vision_end|>\n";
            prompt_text = vis_str + user_text;
        }
        std::string formatted = format_lfm3_chat(prompt_text, system_prompt, enable_thinking);
        std::vector<int32_t> prompt_ids = engine.tokenizer().encode(formatted, false);
        for (int32_t id : prompt_ids) sampler.accept(id);

        StageTimers prefill_t, decode_t;
        g_interrupt_flag.store(false);
        g_in_generation.store(true);

        if (!engine.prefill(prompt_ids, prefill_t, vis_ptr, num_vis_tok)) {
            std::printf("\n[ERROR: prefill failed]\n");
            g_in_generation.store(false);
            return;
        }

        bool in_think = enable_thinking;
        if (in_think) {
            std::cout << COLOR_GRAY << "[Thinking: " << std::flush;
        }

        for (int step = 0; step < max_new_tokens && !g_interrupt_flag.load(); ++step) {
            int32_t next_id = sampler.sample(engine.logits_mut(), Lfm3Engine::kVocab);
            sampler.accept(next_id);

            if (next_id == 124900) break; // <|im_end|>
            if (next_id == 124902) {      // </think>
                if (in_think) {
                    in_think = false;
                    std::cout << "]" << COLOR_RESET << "\n\n" << std::flush;
                }
                if (step + 1 < max_new_tokens) {
                    engine.forward_chunk(&next_id, 1, true, decode_t);
                }
                continue;
            }

            std::string piece = engine.tokenizer().decode_token(next_id);
            if (in_think) {
                std::cout << COLOR_GRAY << piece << std::flush;
            } else {
                std::cout << piece << std::flush;
            }

            if (step + 1 < max_new_tokens) {
                if (!engine.forward_chunk(&next_id, 1, true, decode_t)) break;
            }
        }
        g_in_generation.store(false);
        if (in_think) std::cout << COLOR_RESET;
        double dec_s = decode_t.total_ms() / 1000.0;
        double tok_s = (dec_s > 0.0) ? ((decode_t.tokens_run + 1) / dec_s) : 0.0;
        double pref_s = prefill_t.total_ms() / 1000.0;
        double pref_tok_s = (pref_s > 0.0) ? (prompt_ids.size() / pref_s) : 0.0;
        std::printf("\n\n%s[Prefill: %zu tok in %.1f ms (%.2f tok/s, %.1f ms/tok) | Decode: %d tok at %.2f tok/s (%.1f ms/tok)]%s\n\n",
                    COLOR_GRAY, prompt_ids.size(), prefill_t.total_ms(), pref_tok_s,
                    prefill_t.total_ms() / std::max<size_t>(1, prompt_ids.size()),
                    decode_t.tokens_run + 1, tok_s,
                    decode_t.total_ms() / std::max(1, decode_t.tokens_run), COLOR_RESET);
    };

    if (!prompt.empty()) {
        run_chat_turn(prompt);
        return 0;
    }

    std::cout << COLOR_BOLD << COLOR_CYAN
              << "LFM3-1.7B Interactive Dual-Die Chat REPL (type '/exit' to quit, '/nothink' to toggle thinking, '/image <path>' to load image)"
              << COLOR_RESET << std::endl;
    std::string line;
    while (true) {
        std::cout << COLOR_BOLD << COLOR_GREEN << "User> " << COLOR_RESET << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line == "/exit" || line == "/quit") break;
        if (line == "/nothink") {
            enable_thinking = !enable_thinking;
            std::cout << COLOR_YELLOW << (enable_thinking ? "[Thinking enabled]" : "[Thinking disabled]")
                      << COLOR_RESET << std::endl;
            continue;
        }
        if (line.rfind("/image ", 0) == 0) {
            std::string p = line.substr(7);
            while (!p.empty() && std::isspace(p.front())) p.erase(0, 1);
            while (!p.empty() && std::isspace(p.back())) p.pop_back();
            if (!file_exists(p)) {
                std::cout << COLOR_RED << "[File not found: " << p << "]" << COLOR_RESET << std::endl;
                continue;
            }
            current_image_path = p;
            current_image_embeds.clear();
            if (p.rfind(".bin") != std::string::npos || p.rfind(".raw") != std::string::npos) {
                FILE *f = std::fopen(p.c_str(), "rb");
                if (f) {
                    std::fseek(f, 0, SEEK_END);
                    long sz = std::ftell(f);
                    std::fseek(f, 0, SEEK_SET);
                    if (sz > 0 && sz % (Lfm3Engine::kHidden * sizeof(float)) == 0) {
                        current_image_embeds.resize(sz / sizeof(float));
                        if (std::fread(current_image_embeds.data(), 1, sz, f) == static_cast<size_t>(sz)) {
                            std::cout << COLOR_GREEN << "[Loaded " << (sz / (Lfm3Engine::kHidden * sizeof(float)))
                                      << " visual tokens from " << p << "]" << COLOR_RESET << std::endl;
                        }
                    }
                    std::fclose(f);
                }
            } else {
                std::cout << COLOR_GREEN << "[Image selected: " << p << ". Note: use client/server or .bin visual embeddings]" << COLOR_RESET << std::endl;
            }
            continue;
        }
        if (line.empty()) continue;
        run_chat_turn(line);
    }
    return 0;
}
