/**
 * @file hardware_kv_cache.hpp
 * @brief Physically contiguous MMZ KV-cache with per-package layout (Milestone 9).
 *
 * The Milestone 5 KvCache keeps K and V in host std::vectors. The compiled NPU block
 * loadables instead expect both caches as DMA-resident tensors bound into the task
 * descriptor, with each step's write position expressed as a byte offset into that
 * resident buffer.
 *
 * ## The layout is a property of the loadable, not of the runtime
 *
 * Two different, mutually incompatible KV layouts are documented for this toolchain:
 *
 *  A. Measured directly from a real Qwen2-0.5B 1k INT8 block loadable
 *     (`modified_block_0_npu_b1.model`, inspected with `eswin-sim --info -v`):
 *       K input  (2, 64, 1, 1024) = [num_kv_heads, head_dim, 1, kv_len]  -> t contiguous
 *       V input  (2, 1024, 1, 64) = [num_kv_heads, kv_len, 1, head_dim]  -> d contiguous
 *       K output (1, 128, 1024, 1) = [1, heads*head_dim, kv_len, 1]
 *       V output (1, 2048, 64, 1)  = [1, heads*kv_len, head_dim, 1]
 *     with kv_len == 1024 == the package's `token_num`.
 *
 *  B. The vendor formulas recorded in FINDINGS.md section 5 (from the DeepSeek-7B
 *     `_peer` package), where K is token-major and V is tiled:
 *       K: elem * kv_dim * token
 *       V: elem * (token_n2 * head_dim * (token / token_n2) + (token % token_n2))
 *
 * EsAAC picks the layout per compile, so a single hard-coded formula is wrong for some
 * packages - and getting it wrong does not crash, it silently attends to the wrong
 * positions and reproduces the Issue #47 repetition/gibberish failure mode. Both layouts
 * are therefore implemented and selected explicitly, defaulting to (A) because that is
 * the one verified against a real loadable. The authoritative source is always the
 * loadable's own tensor descriptors; see the "Layout discovery" note in
 * docs/HARDWARE_KV_CACHE_GUIDE.md for the gap that currently prevents reading them
 * automatically.
 *
 * ## NUMA
 *
 * Buffers are allocated in the MMZ zone of the die that runs the layer, so a dual-die
 * split (layers 0..L/2 on Die 0, L/2..L on Die 1) keeps each layer's cache on the NUMA
 * node that touches it. Cross-die allocation costs up to 7.7x on writes and 3.25x on
 * latency, so the split is decided once at allocation, never per step.
 *
 * reset() rewinds the write position without reallocating or re-preparing anything,
 * which is what makes multi-turn conversation cheap.
 */

#ifndef ES_HARDWARE_KV_CACHE_HPP
#define ES_HARDWARE_KV_CACHE_HPP

#include "../backend/dma_buffer.hpp"
#include "../backend/worker_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eswin {

/// K/V element ordering inside one layer's cache buffer.
enum class KvLayout {
    /// (A) [head][dim][token] for K and [head][token][dim] for V. Verified against a
    /// real Qwen2-0.5B 1k INT8 loadable.
    HeadMajor,
    /// (B) [token][kv_dim] for K, tiled [tile][dim][token_in_tile] for V. The vendor
    /// formula pair from FINDINGS.md section 5.
    TokenMajorTiled,
};

struct HardwareKvConfig {
    uint32_t num_layers   = 0;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim     = 0;
    uint32_t max_seq_len  = 1024;  ///< Compiled KV window (vendor `token_num`).
    size_t   elem_bytes   = 2;     ///< 2 for fp16, which is what the loadables use.
    KvLayout layout       = KvLayout::HeadMajor;
    uint32_t token_n2     = 0;     ///< V tile height for TokenMajorTiled; 0 picks a divisor.
    bool     dual_die     = false;///< Split layers across mmz_nid_0 / mmz_nid_1.
    bool     wrap         = true;  ///< Circular addressing; false rejects past the window.
};

class HardwareKvCache {
public:
    HardwareKvCache() = default;
    ~HardwareKvCache();

    HardwareKvCache(const HardwareKvCache &) = delete;
    HardwareKvCache &operator=(const HardwareKvCache &) = delete;

    /**
     * @brief Allocates one K and one V buffer per layer.
     *
     * @param pool When non-null, used to SMMU-prepare each buffer inside the worker
     *             process that owns its die. The vendor driver requires a per-buffer
     *             ES_NPU_Prepare in the submitting process (FINDINGS section 19), so
     *             preparing from the master would fail with 0xA00F604A. Pass nullptr
     *             off-board, or when the caller prepares the buffers itself.
     */
    bool allocate(const HardwareKvConfig &cfg, WorkerPool *pool, std::string &err);

    /// Unprepares and frees every buffer. Safe to call repeatedly.
    void release();

    bool is_allocated() const noexcept { return !m_k.empty(); }
    const HardwareKvConfig &config() const noexcept { return m_cfg; }

    uint32_t capacity() const noexcept { return m_cfg.max_seq_len; }
    /// Tokens currently held, i.e. the next write position before wrapping.
    uint32_t length() const noexcept { return m_length; }
    size_t   total_bytes() const noexcept { return m_total_bytes; }
    size_t   bytes_per_layer() const noexcept;

    /// Die whose MMZ zone owns this layer's buffers.
    uint16_t die_for_layer(uint32_t layer) const noexcept;

    /// Storage slot for an absolute token position (wraps when cfg.wrap is set).
    uint32_t slot_for(uint32_t token) const noexcept;

    /// Byte offset of one element within a layer's K / V buffer.
    size_t k_offset(uint32_t head, uint32_t dim, uint32_t token) const noexcept;
    size_t v_offset(uint32_t head, uint32_t token, uint32_t dim) const noexcept;

    /// Vendor-form byte offset of a whole token step, for building task descriptors.
    size_t k_token_offset(uint32_t token) const noexcept;
    size_t v_token_offset(uint32_t token) const noexcept;

    /**
     * @brief Advances the write position by @p n tokens.
     * @return false when the window would be exceeded and cfg.wrap is disabled.
     */
    bool advance(uint32_t n = 1);

    /// Rewinds to an empty cache without reallocating (multi-turn reuse).
    void reset(bool zero_fill = false);

    // --- Buffer binding -----------------------------------------------------
    DmaBuffer *k_buffer(uint32_t layer);
    DmaBuffer *v_buffer(uint32_t layer);
    const DmaBuffer *k_buffer(uint32_t layer) const;
    const DmaBuffer *v_buffer(uint32_t layer) const;

    /// memFd to place in a task descriptor's DevBuf for this layer's cache.
    uint64_t k_mem_fd(uint32_t layer) const;
    uint64_t v_mem_fd(uint32_t layer) const;

    // --- CPU-side access (host fallback, and verification against golden data) --
    /// Writes kv_dim values for (layer, token); @p values is [head][dim] order.
    bool write_k(uint32_t layer, uint32_t token, const float *values);
    /// Writes kv_dim values for (layer, token); @p values is [head][dim] order.
    bool write_v(uint32_t layer, uint32_t token, const float *values);
    bool read_k(uint32_t layer, uint32_t token, float *out) const;
    bool read_v(uint32_t layer, uint32_t token, float *out) const;

    /// Picks a V tile height that divides max_seq_len, for TokenMajorTiled.
    static uint32_t default_token_n2(uint32_t max_seq_len) noexcept;

private:
    size_t kv_dim() const noexcept {
        return static_cast<size_t>(m_cfg.num_kv_heads) * m_cfg.head_dim;
    }
    const void *k_elem(uint32_t layer, uint32_t head, uint32_t dim, uint32_t token) const;
    const void *v_elem(uint32_t layer, uint32_t head, uint32_t token, uint32_t dim) const;
    void *k_elem_mut(uint32_t layer, uint32_t head, uint32_t dim, uint32_t token);
    void *v_elem_mut(uint32_t layer, uint32_t head, uint32_t token, uint32_t dim);

    HardwareKvConfig m_cfg;
    uint32_t m_token_n2 = 0;
    WorkerPool *m_pool = nullptr;      ///< Not owned; used for prepare/unprepare only.
    std::vector<std::unique_ptr<DmaBuffer>> m_k;
    std::vector<std::unique_ptr<DmaBuffer>> m_v;
    std::vector<uint16_t> m_layer_die;
    uint32_t m_length = 0;
    size_t   m_total_bytes = 0;
};

} // namespace eswin

#endif // ES_HARDWARE_KV_CACHE_HPP
