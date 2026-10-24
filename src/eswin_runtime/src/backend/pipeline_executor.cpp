/**
 * @file pipeline_executor.cpp
 * @brief Zero-copy multi-layer block chaining engine (Milestone 7).
 *
 * See pipeline_executor.hpp for the architecture. Two implementation notes that are
 * easy to get wrong on this SoC:
 *
 *  1. Activation buffers are allocated CacheMode::Cached, not NoCache. The cross-die
 *     handoff is performed by the CPU through the mapped virtual addresses, and
 *     measured NoCache CPU reads collapse to 10-60 MB/s (single uncacheable bus
 *     beats) versus 1.74-2.89 GB/s cached. Every CPU write is followed by
 *     flush_cache() so the NPU sees it.
 *  2. Buffers are prepared on their own die only. Preparing a Die 0 buffer on Die 1
 *     wastes an SMMU mapping and, on the vendor driver, can leave stale IOVA entries
 *     behind after a fault.
 */

#include "pipeline_executor.hpp"
#include "driver_wrapper.hpp"

#include <chrono>
#include <cstring>
#include <sstream>

namespace eswin {

namespace {

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

const char *zone_for_die(uint16_t die_id) {
    return (die_id == 1) ? "mmz_nid_1_part_0" : "mmz_nid_0_part_0";
}

} // namespace

PipelineExecutor::PipelineExecutor(WorkerPool &pool)
    : m_pool(pool)
    , m_hardware(EswinDriver::instance().is_hardware_available()) {
    for (int s = 0; s < 2; ++s) {
        m_buf[s].resize(2);
        m_prepared[s].assign(2, false);
    }
}

PipelineExecutor::~PipelineExecutor() {
    release();
}

void PipelineExecutor::release() {
    for (int slot = 0; slot < 2; ++slot) {
        for (uint16_t die = 0; die < m_buf[slot].size(); ++die) {
            if (m_buf[slot][die]) {
                if (m_prepared[slot][die]) {
                    m_pool.unprepare_buffer(die, *m_buf[slot][die]);
                    m_prepared[slot][die] = false;
                }
                m_buf[slot][die].reset();
            }
        }
    }
    m_stages.clear();
    m_ready = false;
}

bool PipelineExecutor::init(const std::vector<PipelineStage> &stages, std::string &err) {
    if (stages.empty()) {
        err = "PipelineExecutor: stage list is empty";
        return false;
    }

    release();
    m_stages = stages;

    // Largest activation any stage on a given die needs; both ping-pong slots are the
    // same size so a stage's output buffer can always serve as the next stage's input.
    size_t die_bytes[2] = {0, 0};
    for (const auto &st : m_stages) {
        if (st.die_id > 1) {
            std::ostringstream os;
            os << "PipelineExecutor: stage '" << st.label << "' has invalid die_id "
               << st.die_id;
            err = os.str();
            release();
            return false;
        }
        size_t need = st.in_bytes > st.out_bytes ? st.in_bytes : st.out_bytes;
        if (need > die_bytes[st.die_id]) die_bytes[st.die_id] = need;
    }

    for (uint16_t die = 0; die < 2; ++die) {
        if (die_bytes[die] == 0) continue;
        for (int slot = 0; slot < 2; ++slot) {
            auto buf = std::make_unique<DmaBuffer>(die_bytes[die], CacheMode::Cached,
                                                   zone_for_die(die));
            if (!buf->is_valid()) {
                err = "PipelineExecutor: failed to allocate activation buffer (die " +
                      std::to_string(die) + ", slot " + std::to_string(slot) + ", " +
                      std::to_string(die_bytes[die]) + " bytes)";
                release();
                return false;
            }
            if (!m_pool.prepare_buffer(die, *buf)) {
                err = "PipelineExecutor: SMMU prepare failed for die " +
                      std::to_string(die) + " slot " + std::to_string(slot);
                release();
                return false;
            }
            m_prepared[slot][die] = true;
            m_buf[slot][die] = std::move(buf);
        }
    }

    m_ready = true;
    return true;
}

bool PipelineExecutor::copy_across_dies(const DmaBuffer &src, uint16_t src_die,
                                        DmaBuffer &dst, uint16_t dst_die,
                                        size_t bytes, uint64_t *elapsed_ns) {
    (void)src_die;
    (void)dst_die;
    if (!src.is_valid() || !dst.is_valid() || bytes == 0) return false;
    size_t n = bytes < src.size() ? bytes : src.size();
    if (n > dst.size()) n = dst.size();

    uint64_t t0 = now_ns();
    std::memcpy(dst.data(), src.data(), n);
    // CPU wrote the destination: make it visible to the receiving die's NPU.
    bool flushed = dst.flush_cache();
    if (elapsed_ns) *elapsed_ns = now_ns() - t0;
    return flushed;
}

bool PipelineExecutor::run(const void *seed, size_t seed_bytes,
                           void *out, size_t out_bytes,
                           PipelineStats *stats) {
    if (!m_ready || m_stages.empty()) return false;

    PipelineStats local;
    local.stages = m_stages.size();
    local.stage_ns.assign(m_stages.size(), 0);

    const uint64_t t_total0 = now_ns();

    // Seed the first stage's input slot.
    {
        const PipelineStage &st = m_stages.front();
        DmaBuffer *in = m_buf[read_slot(0)][st.die_id].get();
        if (!in) return false;
        size_t n = seed_bytes ? seed_bytes : st.in_bytes;
        if (n > in->size()) n = in->size();
        if (seed && n) {
            std::memcpy(in->data(), seed, n);
            in->flush_cache();
        }
    }

    uint64_t submit_ns = 0;
    uint64_t handoff_ns = 0;

    for (size_t i = 0; i < m_stages.size(); ++i) {
        const PipelineStage &st = m_stages[i];
        DmaBuffer *in  = m_buf[read_slot(i)][st.die_id].get();
        DmaBuffer *ot  = m_buf[read_slot(i) ^ 1u][st.die_id].get();
        if (!in || !ot) return false;

        const uint64_t t0 = now_ns();
        bool ok = true;

        if (m_hardware && st.model_id != 0) {
            std::vector<DmaBuffer *> ins{in};
            std::vector<DmaBuffer *> outs{ot};
            uint64_t elapsed = 0;
            ok = m_pool.submit_task(st.die_id, st.model_id, ins, outs, &elapsed);
            submit_ns += elapsed;
        } else {
            // Off-board: WorkerPool::submit_task succeeds without computing anything,
            // so run the caller's stand-in transform to keep chaining observable.
            size_t n_floats = st.out_bytes / sizeof(float);
            if (n_floats * sizeof(float) > ot->size()) {
                n_floats = ot->size() / sizeof(float);
            }
            if (m_host_stage_fn) {
                ok = m_host_stage_fn(static_cast<const float *>(in->data()),
                                     static_cast<float *>(ot->data()), n_floats);
            } else if (n_floats) {
                std::memcpy(ot->data(), in->data(), n_floats * sizeof(float));
            }
            ot->flush_cache();
        }

        local.stage_ns[i] = now_ns() - t0;
        if (!ok) return false;

        // Hand the activation to the next die when the chain crosses a die boundary.
        if (i + 1 < m_stages.size()) {
            const PipelineStage &nxt = m_stages[i + 1];
            if (nxt.die_id != st.die_id) {
                DmaBuffer *dst = m_buf[read_slot(i + 1)][nxt.die_id].get();
                if (!dst) return false;
                size_t bytes = st.out_bytes ? st.out_bytes : nxt.in_bytes;
                uint64_t elapsed = 0;
                if (!copy_across_dies(*ot, st.die_id, *dst, nxt.die_id, bytes, &elapsed)) {
                    return false;
                }
                handoff_ns += elapsed;
                ++local.cross_die_handoffs;
            }
        }
    }

    // Final activation back to the caller.
    if (out && out_bytes) {
        const PipelineStage &last = m_stages.back();
        DmaBuffer *src = m_buf[read_slot(m_stages.size() - 1) ^ 1u][last.die_id].get();
        if (src) {
            size_t n = last.out_bytes ? last.out_bytes : out_bytes;
            if (n > out_bytes) n = out_bytes;
            if (n > src->size()) n = src->size();
            std::memcpy(out, src->data(), n);
        }
    }

    local.total_ns = now_ns() - t_total0;
    local.submit_ns = submit_ns;
    local.handoff_ns = handoff_ns;
    local.avg_stage_us = m_stages.empty()
        ? 0.0
        : static_cast<double>(local.total_ns) / 1000.0 / static_cast<double>(m_stages.size());

    if (stats) *stats = local;
    return true;
}

} // namespace eswin
