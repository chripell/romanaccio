/**
 * @file test_hardware_kv_cache.cpp
 * @brief Milestone 9 test suite: MMZ-resident KV cache with per-package layout.
 *
 * The geometry used throughout is the real Qwen2-0.5B 1k INT8 block loadable, read off
 * `modified_block_0_npu_b1.model` with `eswin-sim --info -v`:
 *
 *   24 layers, 2 KV heads, head_dim 64, kv_len 1024 (= vendor `token_num`), fp16
 *   K input (2, 64, 1, 1024) = 256.00 KiB
 *   V input (2, 1024, 1, 64) = 256.00 KiB
 *
 * so bytes_per_layer() must come out at exactly 262144. That assertion ties the offset
 * arithmetic to observed hardware truth rather than to a guess.
 */

#include "../src/llm/hardware_kv_cache.hpp"
#include "../src/backend/worker_pool.hpp"
#include "../src/backend/driver_wrapper.hpp"
#include "../src/model/embedding_table.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
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

// Qwen2-0.5B 1k INT8, as compiled by EsAAC.
HardwareKvConfig qwen2_05b_config(KvLayout layout) {
    HardwareKvConfig c;
    c.num_layers = 24;
    c.num_kv_heads = 2;
    c.head_dim = 64;
    c.max_seq_len = 1024;
    c.elem_bytes = 2;
    c.layout = layout;
    return c;
}

void test_geometry_matches_loadable() {
    std::cout << "[*] Testing geometry against the real Qwen2-0.5B loadable..." << std::endl;

    HardwareKvCache kv;
    std::string err;
    auto cfg = qwen2_05b_config(KvLayout::HeadMajor);
    TEST_ASSERT(kv.allocate(cfg, nullptr, err), "allocate failed: " + err);

    // eswin-sim reports 256.00 KiB for both the K and the V input tensor.
    TEST_ASSERT(kv.bytes_per_layer() == 262144,
                "per-layer cache is " + std::to_string(kv.bytes_per_layer()) +
                    " bytes, expected 262144 (256 KiB)");
    TEST_ASSERT(kv.capacity() == 1024, "capacity should equal the compiled token_num");
    TEST_ASSERT(kv.total_bytes() == 2 * 262144 * 24, "total byte accounting wrong");
    TEST_ASSERT(kv.length() == 0, "fresh cache should be empty");

    const DmaBuffer *k = kv.k_buffer(0);
    const DmaBuffer *v = kv.v_buffer(0);
    TEST_ASSERT(k && k->is_valid() && k->size() == 262144, "K buffer wrong size");
    TEST_ASSERT(v && v->is_valid() && v->size() == 262144, "V buffer wrong size");
    TEST_ASSERT(kv.k_buffer(24) == nullptr, "out-of-range layer should be null");

    // Highest-addressable element must sit inside the buffer for both layouts.
    const size_t last_k = kv.k_offset(cfg.num_kv_heads - 1, cfg.head_dim - 1, cfg.max_seq_len - 1);
    const size_t last_v = kv.v_offset(cfg.num_kv_heads - 1, cfg.max_seq_len - 1, cfg.head_dim - 1);
    TEST_ASSERT(last_k + 2 <= 262144, "K offset runs past the buffer");
    TEST_ASSERT(last_v + 2 <= 262144, "V offset runs past the buffer");

    std::cout << "    24 layers x (256 KiB K + 256 KiB V) = " << kv.total_bytes()
              << " bytes; offsets stay in bounds." << std::endl;
    kv.release();
    TEST_ASSERT(!kv.is_allocated(), "still allocated after release");
}

void test_roundtrip(KvLayout layout, const char *label) {
    std::cout << "\n[*] Testing write/read round-trip (" << label << ")..." << std::endl;

    HardwareKvCache kv;
    std::string err;
    auto cfg = qwen2_05b_config(layout);
    cfg.num_layers = 3;  // enough to exercise per-layer indexing
    TEST_ASSERT(kv.allocate(cfg, nullptr, err), "allocate failed: " + err);

    const size_t kv_dim = static_cast<size_t>(cfg.num_kv_heads) * cfg.head_dim;
    std::vector<float> in(kv_dim), out_k(kv_dim, 0.0f), out_v(kv_dim, 0.0f);

    // Distinct value per (layer, token, channel) so a swapped index shows up.
    const uint32_t kTokens[] = {0, 1, 7, 63, 1023};
    for (uint32_t layer = 0; layer < cfg.num_layers; ++layer) {
        for (uint32_t t : kTokens) {
            for (size_t i = 0; i < kv_dim; ++i) {
                in[i] = static_cast<float>(layer + 1) * 0.25f +
                        static_cast<float>(t) * 0.001f + static_cast<float>(i) * 0.01f;
            }
            TEST_ASSERT(kv.write_k(layer, t, in.data()), "write_k failed");
            TEST_ASSERT(kv.write_v(layer, t, in.data()), "write_v failed");

            TEST_ASSERT(kv.read_k(layer, t, out_k.data()), "read_k failed");
            TEST_ASSERT(kv.read_v(layer, t, out_v.data()), "read_v failed");
            for (size_t i = 0; i < kv_dim; ++i) {
                // fp16 storage: allow half-precision rounding.
                TEST_ASSERT(std::fabs(out_k[i] - in[i]) < 0.02f,
                            "K mismatch at layer/token/channel");
                TEST_ASSERT(std::fabs(out_v[i] - in[i]) < 0.02f,
                            "V mismatch at layer/token/channel");
            }
        }
    }

    // Layers must not alias each other.
    for (size_t i = 0; i < kv_dim; ++i) in[i] = 3.5f;
    TEST_ASSERT(kv.write_k(1, 5, in.data()), "write_k layer 1 failed");
    TEST_ASSERT(kv.read_k(0, 5, out_k.data()), "read_k layer 0 failed");
    TEST_ASSERT(std::fabs(out_k[0] - 3.5f) > 0.1f, "layer 1 write leaked into layer 0");

    std::cout << "    Round-trip verified over tokens {0,1,7,63,1023} x 3 layers." << std::endl;
    kv.release();
}

void test_vendor_offset_formulas() {
    std::cout << "\n[*] Testing the FINDINGS section 5 vendor offset formulas..." << std::endl;

    HardwareKvCache kv;
    std::string err;
    auto cfg = qwen2_05b_config(KvLayout::TokenMajorTiled);
    cfg.num_layers = 1;
    TEST_ASSERT(kv.allocate(cfg, nullptr, err), "allocate failed: " + err);

    const size_t kv_dim = static_cast<size_t>(cfg.num_kv_heads) * cfg.head_dim;  // 128
    const size_t elem = cfg.elem_bytes;                                          // 2

    // K: precision * embeddingDim2 * token_idx
    for (uint32_t t : {0u, 1u, 5u, 1023u}) {
        TEST_ASSERT(kv.k_token_offset(t) == elem * kv_dim * t,
                    "vendor K offset formula mismatch at token " + std::to_string(t));
    }

    // V: precision * (tokenN2 * embeddingDim * (token_idx / tokenN2) + (token_idx % tokenN2))
    const uint32_t n2 = HardwareKvCache::default_token_n2(cfg.max_seq_len);
    TEST_ASSERT(n2 > 0 && cfg.max_seq_len % n2 == 0, "token_n2 must divide max_seq_len");
    for (uint32_t t : {0u, 1u, n2 - 1, n2, n2 + 1, 1023u}) {
        const size_t expect = elem * (static_cast<size_t>(n2) * cfg.head_dim * (t / n2) + (t % n2));
        TEST_ASSERT(kv.v_token_offset(t) == expect,
                    "vendor V offset formula mismatch at token " + std::to_string(t));
    }
    std::cout << "    K token-major and V tiled (token_n2=" << n2
              << ") formulas match FINDINGS section 5." << std::endl;
    kv.release();
}

void test_circular_and_strict() {
    std::cout << "\n[*] Testing circular addressing, strict mode and reset..." << std::endl;

    HardwareKvCache kv;
    std::string err;
    auto cfg = qwen2_05b_config(KvLayout::HeadMajor);
    cfg.num_layers = 1;
    TEST_ASSERT(kv.allocate(cfg, nullptr, err), "allocate failed: " + err);

    // Circular: position 1024 lands back on slot 0.
    TEST_ASSERT(kv.slot_for(0) == 0, "slot 0 wrong");
    TEST_ASSERT(kv.slot_for(1023) == 1023, "slot 1023 wrong");
    TEST_ASSERT(kv.slot_for(1024) == 0, "circular wrap failed");
    TEST_ASSERT(kv.slot_for(1025) == 1, "circular wrap off by one");
    TEST_ASSERT(kv.k_offset(0, 0, 1024) == kv.k_offset(0, 0, 0),
                "wrapped token should alias slot 0");

    // advance() saturates at capacity in wrap mode instead of overflowing.
    TEST_ASSERT(kv.advance(1000), "advance(1000) failed");
    TEST_ASSERT(kv.length() == 1000, "length not tracked");
    TEST_ASSERT(kv.advance(100), "advance past capacity failed in wrap mode");
    TEST_ASSERT(kv.length() == 1024, "length should saturate at capacity");

    // reset() must not reallocate: the buffer identity has to survive.
    const DmaBuffer *before = kv.k_buffer(0);
    const uint64_t fd_before = kv.k_mem_fd(0);
    kv.reset(false);
    TEST_ASSERT(kv.length() == 0, "reset did not rewind length");
    TEST_ASSERT(kv.k_buffer(0) == before, "reset reallocated the buffer");
    TEST_ASSERT(kv.k_mem_fd(0) == fd_before, "reset changed the memFd");

    // zero_fill actually clears resident data.
    const size_t kv_dim = static_cast<size_t>(cfg.num_kv_heads) * cfg.head_dim;
    std::vector<float> in(kv_dim, 1.5f), out(kv_dim, -1.0f);
    TEST_ASSERT(kv.write_k(0, 3, in.data()), "write_k failed");
    kv.reset(true);
    TEST_ASSERT(kv.read_k(0, 3, out.data()), "read_k failed");
    TEST_ASSERT(out[0] == 0.0f, "zero_fill did not clear the cache");

    // Strict mode refuses to walk past the compiled window (the Issue #47 guard).
    kv.release();
    cfg.wrap = false;
    TEST_ASSERT(kv.allocate(cfg, nullptr, err), "strict allocate failed: " + err);
    TEST_ASSERT(kv.slot_for(1024) == 1024, "strict mode must not wrap");
    TEST_ASSERT(kv.advance(1024), "advance to capacity should succeed");
    TEST_ASSERT(!kv.advance(1), "strict mode must refuse one token past the window");
    TEST_ASSERT(kv.length() == 1024, "length changed on a refused advance");
    std::cout << "    Circular wrap, saturation, cheap reset and strict refusal verified."
              << std::endl;
    kv.release();
}

void test_dual_die_placement() {
    std::cout << "\n[*] Testing NUMA placement across both dies..." << std::endl;

    HardwareKvCache kv;
    std::string err;
    auto cfg = qwen2_05b_config(KvLayout::HeadMajor);
    cfg.dual_die = true;
    TEST_ASSERT(kv.allocate(cfg, nullptr, err), "dual-die allocate failed: " + err);

    for (uint32_t layer = 0; layer < cfg.num_layers; ++layer) {
        const uint16_t expect = (layer >= cfg.num_layers / 2) ? 1 : 0;
        TEST_ASSERT(kv.die_for_layer(layer) == expect,
                    "layer " + std::to_string(layer) + " on the wrong die");
    }
    std::cout << "    Layers 0-11 on Die 0, 12-23 on Die 1 (local MMZ zones)." << std::endl;
    kv.release();
}

void test_rejected_configs() {
    std::cout << "\n[*] Testing configuration validation..." << std::endl;

    std::string err;
    auto base = qwen2_05b_config(KvLayout::HeadMajor);

    struct { const char *what; HardwareKvConfig cfg; } bad[] = {
        {"zero layers",        [](HardwareKvConfig c){ c.num_layers = 0;   return c; }(base)},
        {"zero kv heads",      [](HardwareKvConfig c){ c.num_kv_heads = 0; return c; }(base)},
        {"zero head_dim",      [](HardwareKvConfig c){ c.head_dim = 0;     return c; }(base)},
        {"zero window",        [](HardwareKvConfig c){ c.max_seq_len = 0;  return c; }(base)},
        {"3-byte elements",    [](HardwareKvConfig c){ c.elem_bytes = 3;   return c; }(base)},
        {"token_n2 not a divisor",
                               [](HardwareKvConfig c){ c.token_n2 = 7; c.layout = KvLayout::TokenMajorTiled; return c; }(base)},
    };

    for (const auto &b : bad) {
        HardwareKvCache kv;
        err.clear();
        TEST_ASSERT(!kv.allocate(b.cfg, nullptr, err),
                    std::string("accepted an invalid config: ") + b.what);
        TEST_ASSERT(!err.empty(), std::string("no error message for ") + b.what);
        TEST_ASSERT(!kv.is_allocated(), std::string("left allocated after failure: ") + b.what);
    }
    std::cout << "    All 6 invalid configurations rejected with a message." << std::endl;
}

void test_fp16_codec() {
    std::cout << "\n[*] Testing the fp16 codec used for resident cache elements..." << std::endl;

    const float values[] = {0.0f, 1.0f, -1.0f, 0.5f, 3.25f, -7.75f, 65504.0f, 1e-4f};
    for (float v : values) {
        const float back = es_f16_to_f32(es_f32_to_f16(v));
        const float tol = std::fabs(v) * 0.002f + 1e-6f;
        TEST_ASSERT(std::fabs(back - v) <= tol,
                    "fp16 round-trip failed for " + std::to_string(v));
    }
    TEST_ASSERT(std::isinf(es_f16_to_f32(es_f32_to_f16(1e9f))), "overflow should reach infinity");
    std::cout << "    fp16 encode/decode round-trips within half precision." << std::endl;
}

} // namespace

int main() {
    std::cout << "========================================================\n"
              << "  ESWIN Hardware MMZ KV-Cache Test Suite (Milestone 9)  \n"
              << "========================================================\n\n";

    const bool hw = EswinDriver::instance().is_hardware_available();
    std::cout << "[*] Backend: " << (hw ? "Hardware NPU (MMZ DMA)" : "Host Fallback")
              << std::endl;

    // A real pool exercises the SMMU prepare path on board; off-board it is a no-op.
    WorkerPool pool;
    const bool pool_ok = pool.start(2);
    std::cout << "[*] WorkerPool: " << (pool_ok ? "started" : "unavailable")
              << " (" << pool.active_worker_count() << " workers)\n" << std::endl;

    test_fp16_codec();
    test_geometry_matches_loadable();
    test_roundtrip(KvLayout::HeadMajor, "HeadMajor, as compiled for Qwen2-0.5B");
    test_roundtrip(KvLayout::TokenMajorTiled, "TokenMajorTiled, FINDINGS section 5");
    test_vendor_offset_formulas();
    test_circular_and_strict();
    test_dual_die_placement();
    test_rejected_configs();

    // Multi-turn reuse: reset between turns must keep the same physical memory.
    std::cout << "\n[*] Testing multi-turn reuse over 200 reset cycles..." << std::endl;
    {
        HardwareKvCache kv;
        std::string err;
        auto cfg = qwen2_05b_config(KvLayout::HeadMajor);
        if (pool_ok && hw) {
            TEST_ASSERT(kv.allocate(cfg, &pool, err),
                        "on-hardware allocate failed: " + err);
        } else {
            TEST_ASSERT(kv.allocate(cfg, nullptr, err), "allocate failed: " + err);
        }
        const DmaBuffer *first = kv.k_buffer(0);
        const size_t kv_dim = static_cast<size_t>(cfg.num_kv_heads) * cfg.head_dim;
        std::vector<float> in(kv_dim, 0.5f);
        for (int turn = 0; turn < 200; ++turn) {
            for (uint32_t t = 0; t < 8; ++t) {
                TEST_ASSERT(kv.write_k(0, t, in.data()), "write_k failed mid-cycle");
                TEST_ASSERT(kv.advance(1), "advance failed mid-cycle");
            }
            kv.reset(false);
            TEST_ASSERT(kv.length() == 0, "reset left a non-zero length");
        }
        TEST_ASSERT(kv.k_buffer(0) == first, "buffers were reallocated across turns");
        std::cout << "    200 turns reused the same physical allocation." << std::endl;
        kv.release();
    }

    if (pool_ok) pool.shutdown();

    std::cout << "\n========================================================\n"
              << "  ALL MILESTONE 9 HARDWARE KV-CACHE TESTS PASSED!       \n"
              << "========================================================\n";
    return 0;
}
