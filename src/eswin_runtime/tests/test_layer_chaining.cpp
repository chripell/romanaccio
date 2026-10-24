/**
 * @file test_layer_chaining.cpp
 * @brief Milestone 7 test suite: multi-layer block chaining & dual-die pipeline engine.
 *
 * Verifies PipelineExecutor chaining semantics without needing compiled NPU graphs:
 * a host stage transform stands in for each block loadable, so the activation path,
 * ping-pong alternation, cross-die handoff accounting and teardown are all observable.
 * On board `roma` the same suite exercises real MMZ DMA buffers and SMMU mappings.
 */

#include "../src/backend/pipeline_executor.hpp"
#include "../src/backend/worker_pool.hpp"
#include "../src/backend/driver_wrapper.hpp"
#include "../src/model/model_manifest.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define TEST_ASSERT(cond, msg)                                                  \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << "\033[1;31m[FAIL]\033[0m " << msg << " (" << #cond     \
                      << ") at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

using namespace eswin;

namespace {

// Each stage adds exactly 1.0 to every element, so after N stages the activation is
// seed + N. Any skipped, duplicated or out-of-order stage shows up immediately.
bool increment_stage(const float *in, float *out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = in[i] + 1.0f;
    return true;
}

std::vector<PipelineStage> make_chain(size_t stages, size_t floats, uint16_t dies) {
    std::vector<PipelineStage> chain;
    chain.reserve(stages);
    for (size_t i = 0; i < stages; ++i) {
        PipelineStage st;
        // Pipeline parallelism layout: first half on Die 0, second half on Die 1.
        st.die_id = (dies >= 2 && i >= stages / 2) ? 1 : 0;
        st.model_id = 0;  // 0 => host stage transform (no compiled loadable needed)
        st.in_bytes = floats * sizeof(float);
        st.out_bytes = floats * sizeof(float);
        st.label = "layer_" + std::to_string(i);
        chain.push_back(st);
    }
    return chain;
}

void run_chain(WorkerPool &pool, size_t stages, size_t floats, uint16_t dies,
               bool expect_cross_die) {
    PipelineExecutor exec(pool);
    exec.set_host_stage_fn(increment_stage);

    std::string err;
    auto chain = make_chain(stages, floats, dies);
    TEST_ASSERT(exec.init(chain, err), "init failed for " + std::to_string(stages) +
                                           " stages: " + err);
    TEST_ASSERT(exec.is_ready(), "executor not ready after init");
    TEST_ASSERT(exec.stage_count() == stages, "stage count mismatch");

    std::vector<float> seed(floats, 0.0f);
    std::vector<float> result(floats, -1.0f);

    PipelineStats stats;
    TEST_ASSERT(exec.run(seed.data(), seed.size() * sizeof(float),
                         result.data(), result.size() * sizeof(float), &stats),
                "run failed for " + std::to_string(stages) + " stages");

    // Chaining correctness: every element advanced by exactly one per stage.
    for (size_t i = 0; i < floats; ++i) {
        TEST_ASSERT(std::fabs(result[i] - static_cast<float>(stages)) < 1e-4f,
                    "activation after " + std::to_string(stages) +
                        " stages is wrong at index " + std::to_string(i) +
                        " (got " + std::to_string(result[i]) + ")");
    }

    TEST_ASSERT(stats.stages == stages, "stats.stages mismatch");
    TEST_ASSERT(stats.stage_ns.size() == stages, "per-stage timing vector size mismatch");
    TEST_ASSERT(stats.total_ns > 0, "total elapsed time not measured");

    if (expect_cross_die) {
        TEST_ASSERT(stats.cross_die_handoffs == 1,
                    "expected exactly one die boundary handoff, got " +
                        std::to_string(stats.cross_die_handoffs));
        TEST_ASSERT(stats.handoff_ns > 0, "cross-die handoff time not measured");
    } else {
        TEST_ASSERT(stats.cross_die_handoffs == 0,
                    "unexpected cross-die handoffs on a single-die chain");
    }

    std::cout << "    " << stages << " stages / " << dies << " die(s): total "
              << (stats.total_ns / 1000.0) << " us, avg "
              << stats.avg_stage_us << " us/stage, handoffs "
              << stats.cross_die_handoffs << " (" << (stats.handoff_ns / 1000.0)
              << " us)" << std::endl;

    exec.release();
    TEST_ASSERT(!exec.is_ready(), "executor still ready after release");
}

long read_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string key;
    while (f >> key) {
        if (key == "VmRSS:") {
            long kb = 0;
            f >> kb;
            return kb;
        }
        std::string rest;
        std::getline(f, rest);
    }
    return -1;
}

// Manifest block-chain parsing: the open "blocks" spelling, the vendor ENNP
// "model_list"/"block_models" spelling, serialization round-trip, and driving a real
// PipelineExecutor chain straight out of a parsed manifest.
void test_manifest_block_chain(WorkerPool &pool) {
    std::string err;

    const char *open_json = R"({
        "name": "ChainPkg",
        "tensor_parallel": 2,
        "config": { "architecture": "qwen2", "num_hidden_layers": 4 },
        "blocks": [
            { "name": "blk0.model", "die_id": 0, "stage": 0 },
            { "name": "blk1.model", "die_id": 0, "stage": 1 },
            { "name": "blk2.model", "die_id": 1, "stage": 2 },
            { "name": "blk3.model", "die_id": 1, "stage": 3 }
        ]
    })";

    ModelManifest m;
    TEST_ASSERT(m.load_from_string(open_json, err), "open block manifest parse failed: " + err);
    TEST_ASSERT(m.blocks.size() == 4, "expected 4 block entries");
    TEST_ASSERT(m.blocks_for_die(0).size() == 2, "die 0 should own 2 blocks");
    TEST_ASSERT(m.blocks_for_die(1).size() == 2, "die 1 should own 2 blocks");
    TEST_ASSERT(m.blocks_for_die(0)[0].name == "blk0.model", "die 0 stage order wrong");
    TEST_ASSERT(m.blocks_for_die(1)[1].name == "blk3.model", "die 1 stage order wrong");
    std::cout << "    Parsed open 'blocks' manifest (4 stages, 2 per die)." << std::endl;

    // Vendor ENNP config.json: model_list[].device_id + block_models[]
    const char *vendor_json = R"({
        "name": "VendorPkg",
        "token_num": 1024,
        "model_list": [
            { "device_id": 0,
              "block_models": ["modified_block_0_npu_b1.model", "modified_block_1_npu_b1.model"] },
            { "device_id": 1,
              "block_models": ["modified_block_2_npu_b1.model"] }
        ]
    })";

    ModelManifest v;
    TEST_ASSERT(v.load_from_string(vendor_json, err), "vendor manifest parse failed: " + err);
    TEST_ASSERT(v.blocks.size() == 3, "expected 3 vendor block entries");
    TEST_ASSERT(v.blocks_for_die(0).size() == 2, "vendor die 0 block count wrong");
    TEST_ASSERT(v.blocks_for_die(1).size() == 1, "vendor die 1 block count wrong");
    TEST_ASSERT(v.blocks_for_die(1)[0].name == "modified_block_2_npu_b1.model",
                "vendor die 1 block name wrong");
    TEST_ASSERT(v.blocks_for_die(0)[1].stage == 1, "vendor stage numbering wrong");
    TEST_ASSERT(v.config.context_length == 1024, "vendor token_num should map to context_length");
    std::cout << "    Parsed vendor 'model_list' manifest (device_id -> die_id)." << std::endl;

    // Serialization round-trip preserves the chain.
    ModelManifest rt;
    TEST_ASSERT(rt.load_from_string(m.to_json_string(2), err), "round-trip parse failed: " + err);
    TEST_ASSERT(rt.blocks.size() == 4, "round-trip lost block entries");
    TEST_ASSERT(rt.blocks_for_die(1)[0].name == "blk2.model", "round-trip reordered blocks");
    std::cout << "    Serialization round-trip preserved the chain." << std::endl;

    // Drive an executor directly from the parsed manifest.
    std::vector<PipelineStage> chain;
    const size_t kFloats = 256;
    for (const auto &b : m.blocks) {
        PipelineStage st;
        st.die_id = b.die_id;
        st.model_id = 0;  // host stage transform; a real package resolves loaded ids
        st.in_bytes = kFloats * sizeof(float);
        st.out_bytes = kFloats * sizeof(float);
        st.label = b.name;
        chain.push_back(st);
    }

    PipelineExecutor exec(pool);
    exec.set_host_stage_fn(increment_stage);
    TEST_ASSERT(exec.init(chain, err), "manifest-driven init failed: " + err);

    std::vector<float> seed(kFloats, 0.0f), result(kFloats, -1.0f);
    PipelineStats stats;
    TEST_ASSERT(exec.run(seed.data(), seed.size() * sizeof(float),
                         result.data(), result.size() * sizeof(float), &stats),
                "manifest-driven run failed");
    TEST_ASSERT(std::fabs(result[0] - 4.0f) < 1e-4f,
                "manifest chain did not apply all 4 stages");
    TEST_ASSERT(stats.cross_die_handoffs == 1,
                "manifest chain should cross the die boundary exactly once");
    std::cout << "    Ran a manifest-defined 4-stage chain ("
              << stats.cross_die_handoffs << " cross-die handoff)." << std::endl;

    exec.release();
}

} // namespace

int main() {
    std::cout << "========================================================\n"
              << "  ESWIN Multi-Layer Block Chaining Test Suite (MS 7)    \n"
              << "========================================================\n\n";

    const bool hw = EswinDriver::instance().is_hardware_available();
    std::cout << "[*] Backend: " << (hw ? "Hardware NPU (MMZ DMA)" : "Host Fallback")
              << std::endl;

    WorkerPool pool;
    TEST_ASSERT(pool.start(2), "WorkerPool failed to start");
    uint16_t dies = pool.active_worker_count();
    std::cout << "[*] Active workers: " << dies << "\n" << std::endl;

    // 1. Rejected configurations.
    std::cout << "[*] Testing configuration validation..." << std::endl;
    {
        PipelineExecutor exec(pool);
        std::string err;
        TEST_ASSERT(!exec.init({}, err), "empty stage list should be rejected");
        TEST_ASSERT(!err.empty(), "empty stage list produced no error message");

        std::vector<PipelineStage> bad = make_chain(2, 16, 1);
        bad[1].die_id = 7;
        err.clear();
        TEST_ASSERT(!exec.init(bad, err), "invalid die_id should be rejected");
        TEST_ASSERT(err.find("die_id") != std::string::npos,
                    "invalid die_id error message not specific");
        std::cout << "    Rejected empty chain and invalid die_id." << std::endl;
    }

    // 2. Chain lengths from the plan: 2, 8, and a full 28-layer model.
    std::cout << "\n[*] Testing sequential chaining (activation = seed + stages)..."
              << std::endl;
    const size_t kFloats = 1536;  // Qwen2.5-1.5B hidden size
    run_chain(pool, 2, kFloats, 1, false);
    run_chain(pool, 8, kFloats, 1, false);
    run_chain(pool, 28, kFloats, 1, false);

    // 3. Dual-die pipeline parallelism (layers split across both dies).
    if (dies >= 2) {
        std::cout << "\n[*] Testing dual-die pipeline parallelism..." << std::endl;
        run_chain(pool, 2, kFloats, 2, true);
        run_chain(pool, 8, kFloats, 2, true);
        run_chain(pool, 28, kFloats, 2, true);
    } else {
        std::cout << "\n[*] Dual-die chaining SKIPPED (only " << dies
                  << " worker active)." << std::endl;
    }

    // 4. Manifest-declared block chains (open + vendor ENNP spellings).
    std::cout << "\n[*] Testing manifest block-chain parsing and execution..." << std::endl;
    test_manifest_block_chain(pool);

    // 5. Per-stage handoff latency target (< 40 us per layer, per PLAN.md).
    std::cout << "\n[*] Measuring per-layer activation handoff latency..." << std::endl;
    {
        PipelineExecutor exec(pool);
        exec.set_host_stage_fn(increment_stage);
        std::string err;
        auto chain = make_chain(28, kFloats, dies >= 2 ? 2 : 1);
        TEST_ASSERT(exec.init(chain, err), "latency chain init failed: " + err);

        std::vector<float> seed(kFloats, 0.0f), result(kFloats, 0.0f);
        PipelineStats stats;
        TEST_ASSERT(exec.run(seed.data(), seed.size() * sizeof(float),
                             result.data(), result.size() * sizeof(float), &stats),
                    "latency chain run failed");

        std::cout << "    28-layer chain: " << stats.avg_stage_us
                  << " us/stage average" << std::endl;
        if (hw) {
            TEST_ASSERT(stats.avg_stage_us < 40.0,
                        "per-layer handoff exceeds the 40 us/layer target on hardware");
        } else {
            std::cout << "    (host fallback: 40 us/layer target not enforced off-board)"
                      << std::endl;
        }
        exec.release();
    }

    // 6. Resource unwinding: repeated init/run/release must not grow RSS.
    std::cout << "\n[*] Testing resource unwinding over 40 init/run/release cycles..."
              << std::endl;
    {
        PipelineExecutor warm(pool);
        warm.set_host_stage_fn(increment_stage);
        std::string err;
        auto chain = make_chain(28, kFloats, 1);
        TEST_ASSERT(warm.init(chain, err), "warmup init failed");
        std::vector<float> seed(kFloats, 0.0f), result(kFloats, 0.0f);
        warm.run(seed.data(), seed.size() * sizeof(float),
                 result.data(), result.size() * sizeof(float), nullptr);
        warm.release();

        long rss_before = read_rss_kb();
        for (int cycle = 0; cycle < 40; ++cycle) {
            PipelineExecutor exec(pool);
            exec.set_host_stage_fn(increment_stage);
            err.clear();
            TEST_ASSERT(exec.init(chain, err), "cycle init failed: " + err);
            TEST_ASSERT(exec.run(seed.data(), seed.size() * sizeof(float),
                                 result.data(), result.size() * sizeof(float), nullptr),
                        "cycle run failed");
            exec.release();
        }
        long rss_after = read_rss_kb();
        if (rss_before > 0 && rss_after > 0) {
            long growth = rss_after - rss_before;
            std::cout << "    RSS delta over 40 cycles: " << growth << " KiB" << std::endl;
            TEST_ASSERT(growth < 8192, "RSS grew by " + std::to_string(growth) +
                                           " KiB across cycles (possible leak)");
        } else {
            std::cout << "    /proc/self/status unavailable; RSS check skipped."
                      << std::endl;
        }
    }

    pool.shutdown();

    std::cout << "\n========================================================\n"
              << "  ALL MILESTONE 7 LAYER CHAINING TESTS PASSED!          \n"
              << "========================================================\n";
    return 0;
}
