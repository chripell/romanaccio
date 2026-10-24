/**
 * @file pipeline_executor.hpp
 * @brief Zero-copy multi-layer block chaining engine for dual-die sequential graphs.
 *
 * Milestone 7. Large models (DeepSeek-7B, Qwen2.5) are compiled into N sequential
 * transformer block loadables spread over both dies. Chaining them naively means
 * copying each layer's activation back to the CPU and re-uploading it for the next
 * layer, which costs a full DMA round-trip per layer.
 *
 * This executor keeps the activation resident in MMZ for the whole chain:
 *
 *   - Two activation buffers per die form a ping-pong pair. Stage i reads buffer A
 *     and writes buffer B; stage i+1 reads B and writes A. The output buffer of one
 *     stage therefore *is* the input buffer of the next, with no CPU round-trip.
 *   - Buffers are allocated in the MMZ zone of the die that executes the stage
 *     (mmz_nid_0_part_0 / mmz_nid_1_part_0) and SMMU-prepared once at init, so the
 *     per-stage cost is submission only.
 *   - When consecutive stages live on different dies (pipeline parallelism: layers
 *     0..N/2 on Die 0, N/2..N on Die 1) the activation is handed over once per die
 *     boundary through the mapped virtual addresses, with cache maintenance, and the
 *     cost is accounted separately as a cross-die handoff.
 *
 * Off-board (host fallback) WorkerPool::submit_task succeeds without touching the
 * buffers, so an optional host stage function can stand in for the NPU. That keeps
 * the chaining order, ping-pong alternation and handoff accounting verifiable in CI
 * without hardware; it is never used when real dies are present.
 */

#ifndef ES_PIPELINE_EXECUTOR_HPP
#define ES_PIPELINE_EXECUTOR_HPP

#include "worker_pool.hpp"
#include "dma_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace eswin {

/// One compiled block loadable in the chain.
struct PipelineStage {
    uint16_t    die_id    = 0;   ///< Die whose worker executes this stage.
    uint32_t    model_id  = 0;   ///< Loaded .model handle on that die.
    size_t      in_bytes  = 0;   ///< Activation bytes consumed.
    size_t      out_bytes = 0;   ///< Activation bytes produced.
    std::string label;           ///< Diagnostics only (e.g. "layer_13_mlp").
};

/// Timing and topology accounting for one run() of the chain.
struct PipelineStats {
    size_t              stages             = 0;
    uint64_t            total_ns           = 0;
    uint64_t            submit_ns          = 0;  ///< Inside WorkerPool::submit_task.
    uint64_t            handoff_ns         = 0;  ///< Cross-die activation transfers.
    size_t              cross_die_handoffs = 0;
    double              avg_stage_us       = 0.0;
    std::vector<uint64_t> stage_ns;              ///< Per-stage wall time.
};

class PipelineExecutor {
public:
    explicit PipelineExecutor(WorkerPool &pool);
    ~PipelineExecutor();

    PipelineExecutor(const PipelineExecutor &) = delete;
    PipelineExecutor &operator=(const PipelineExecutor &) = delete;

    /**
     * @brief Sizes, allocates and SMMU-prepares the ping-pong activation buffers.
     *
     * Must be called once before run(). Safe to call again to reconfigure; the
     * previous buffers are released first.
     */
    bool init(const std::vector<PipelineStage> &stages, std::string &err);

    /**
     * @brief Executes the full chain over one activation vector.
     *
     * @param seed       Host input activation copied into the first stage's buffer.
     * @param seed_bytes Bytes of @p seed (must fit the first stage's in_bytes).
     * @param out        Host buffer receiving the final activation (may be null).
     * @param out_bytes  Capacity of @p out.
     */
    bool run(const void *seed, size_t seed_bytes,
             void *out, size_t out_bytes,
             PipelineStats *stats = nullptr);

    /// Host-fallback stage transform: out[i] = fn(in, n). Ignored when hardware is present.
    using StageFn = std::function<bool(const float *in, float *out, size_t n)>;
    void set_host_stage_fn(StageFn fn) { m_host_stage_fn = std::move(fn); }

    /// Releases DMA buffers and unmaps them from the dies that prepared them.
    void release();

    size_t stage_count() const noexcept { return m_stages.size(); }
    bool   is_ready() const noexcept { return m_ready; }
    bool   using_hardware() const noexcept { return m_hardware; }
    const PipelineStage &stage(size_t i) const { return m_stages.at(i); }

private:
    /// Index of the ping-pong slot a stage reads from (even stages read slot 0).
    static size_t read_slot(size_t stage_index) noexcept { return stage_index & 1u; }

    bool copy_across_dies(const DmaBuffer &src, uint16_t src_die,
                          DmaBuffer &dst, uint16_t dst_die,
                          size_t bytes, uint64_t *elapsed_ns);

    WorkerPool                            &m_pool;
    std::vector<PipelineStage>             m_stages;
    std::vector<std::unique_ptr<DmaBuffer>> m_buf[2];  ///< [slot][die] activation buffers
    std::vector<bool>                      m_prepared[2];
    bool                                   m_ready = false;
    bool                                   m_hardware = false;
    StageFn                                m_host_stage_fn;
};

} // namespace eswin

#endif // ES_PIPELINE_EXECUTOR_HPP
