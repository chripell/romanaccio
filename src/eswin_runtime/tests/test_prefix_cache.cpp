/**
 * @file test_prefix_cache.cpp
 * @brief Milestone 17A Host Unit Test Suite: LFMKVB01 Binary Layout & RadixCache.
 *
 * Exercises all 4 acceptance tests specified in src/LFM3/PLAN.md §8.4 and
 * src/LFM2.5-8B-A1B/PLAN.md Phase F without requiring physical NPU hardware:
 *   1. Round-trip bit-exactness across LFM3-1.7B (4x128, 5 SWA, 13x6 ShortConv) and
 *      LFM2.5-8B-A1B (8x64, 0 SWA, 18x2 ShortConv) for both short context (k=2, 192 tok,
 *      Section 4 glob_f32) and long context (k=9, 640 tok, SWA 320-slot ring wrap-around),
 *      including WAL + snapshot persistence across RadixCache restarts.
 *   2. Divergent sequence branching, SWA/ShortConv isolation across branches, and multimodal
 *      <|image|> (token 125030) visual_embeds hash disambiguation.
 *   3. Model fingerprint invalidation (1-bit flip) and per-section checksum corruption
 *      detection + automatic subtree quarantine without crashing.
 *   4. Leaf-first GDSF quota eviction (10 MiB budget) with frequency weighting and
 *      immortal root prefix pinning.
 */

#include "../include/prefix_cache.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
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

using namespace eswin::prefix_cache;

namespace {

int g_pass = 0;

void ok(const char *label) {
    std::cout << "\033[1;32m[PASS]\033[0m " << label << std::endl;
    ++g_pass;
}

struct SimulatedEngineBuffers {
    ModelCacheGeometry    geom;
    std::vector<uint8_t>  glob_k_u8;
    std::vector<uint8_t>  glob_v_u8;
    std::vector<float>    glob_k_sc;
    std::vector<float>    glob_v_sc;
    std::vector<uint16_t> glob_k_sum;
    std::vector<float>    glob_k_f32;
    std::vector<float>    glob_v_f32;
    std::vector<float>    swa_k_f32;
    std::vector<float>    swa_v_f32;
    std::vector<float>    conv_state;
    std::vector<float>    last_h;

    explicit SimulatedEngineBuffers(const ModelCacheGeometry &g, bool alloc_glob_f32 = true)
        : geom(g) {
        const size_t gh_slots = static_cast<size_t>(g.num_glob_layers) * g.kv_heads * g.max_ctx;
        glob_k_u8.assign(gh_slots * g.head_dim, 0);
        glob_v_u8.assign(gh_slots * g.head_dim, 0);
        glob_k_sc.assign(gh_slots, 0.0f);
        glob_v_sc.assign(gh_slots, 0.0f);
        glob_k_sum.assign(gh_slots, 0);
        if (alloc_glob_f32) {
            glob_k_f32.assign(gh_slots * g.head_dim, 0.0f);
            glob_v_f32.assign(gh_slots * g.head_dim, 0.0f);
        }
        if (g.num_swa_layers > 0) {
            const size_t swa_slots = static_cast<size_t>(g.num_swa_layers) * g.kv_heads * g.swa_cap;
            swa_k_f32.assign(swa_slots * g.head_dim, 0.0f);
            swa_v_f32.assign(swa_slots * g.head_dim, 0.0f);
        }
        conv_state.assign(static_cast<size_t>(g.num_conv_layers) * g.hidden_dim * g.conv_state_taps, 0.0f);
        last_h.assign(g.hidden_dim, 0.0f);
    }

    EngineStateView view() {
        EngineStateView v;
        v.glob_k_u8  = glob_k_u8.data();
        v.glob_v_u8  = glob_v_u8.data();
        v.glob_k_sc  = glob_k_sc.data();
        v.glob_v_sc  = glob_v_sc.data();
        v.glob_k_sum = glob_k_sum.data();
        v.glob_k_f32 = glob_k_f32.empty() ? nullptr : glob_k_f32.data();
        v.glob_v_f32 = glob_v_f32.empty() ? nullptr : glob_v_f32.data();
        v.swa_k_f32  = swa_k_f32.empty()  ? nullptr : swa_k_f32.data();
        v.swa_v_f32  = swa_v_f32.empty()  ? nullptr : swa_v_f32.data();
        v.conv_state = conv_state.data();
        return v;
    }

    void poison(uint8_t byte_val = 0xA5) {
        std::memset(glob_k_u8.data(),  byte_val, glob_k_u8.size() * sizeof(uint8_t));
        std::memset(glob_v_u8.data(),  byte_val, glob_v_u8.size() * sizeof(uint8_t));
        std::memset(glob_k_sc.data(),  byte_val, glob_k_sc.size() * sizeof(float));
        std::memset(glob_v_sc.data(),  byte_val, glob_v_sc.size() * sizeof(float));
        std::memset(glob_k_sum.data(), byte_val, glob_k_sum.size() * sizeof(uint16_t));
        if (!glob_k_f32.empty()) {
            std::memset(glob_k_f32.data(), byte_val, glob_k_f32.size() * sizeof(float));
            std::memset(glob_v_f32.data(), byte_val, glob_v_f32.size() * sizeof(float));
        }
        if (!swa_k_f32.empty()) {
            std::memset(swa_k_f32.data(), byte_val, swa_k_f32.size() * sizeof(float));
            std::memset(swa_v_f32.data(), byte_val, swa_v_f32.size() * sizeof(float));
        }
        std::memset(conv_state.data(), byte_val, conv_state.size() * sizeof(float));
        std::memset(last_h.data(),     byte_val, last_h.size() * sizeof(float));
    }

    // Simulates forward_chunk() writing 64 tokens at chunk `block_idx` (`pos = 64*block_idx .. +63`),
    // mirroring the exact Head-Major & SWA ring indexing of lfm3_v2.cpp and lfm2_moe.cpp.
    void simulate_prefill_block(uint32_t block_idx, uint64_t branch_seed) {
        std::mt19937_64 rng(branch_seed * 0x9e3779b97f4a7c15ULL + (block_idx + 1) * 0xbf58476d1ce4e5b9ULL);
        const size_t B     = kBlockTokens;
        const size_t G     = geom.num_glob_layers;
        const size_t S     = geom.num_swa_layers;
        const size_t H     = geom.kv_heads;
        const size_t D     = geom.head_dim;
        const size_t s_cap = geom.max_ctx;
        const size_t s0    = static_cast<size_t>(block_idx) * B;

        for (size_t g = 0; g < G; ++g) {
            for (size_t kh = 0; kh < H; ++kh) {
                const size_t gh_base = (g * H + kh) * s_cap;
                for (size_t t = 0; t < B; ++t) {
                    const size_t slot = gh_base + s0 + t;
                    uint32_t ksum = 0;
                    for (size_t d = 0; d < D; ++d) {
                        uint8_t ku = static_cast<uint8_t>(rng() & 0xFFu);
                        uint8_t vu = static_cast<uint8_t>(rng() & 0xFFu);
                        glob_k_u8[slot * D + d] = ku;
                        glob_v_u8[slot * D + d] = vu;
                        ksum += ku;
                        if (block_idx < 4 && !glob_k_f32.empty()) {
                            glob_k_f32[slot * D + d] = (static_cast<int>(ku) - 128) * 0.03125f;
                            glob_v_f32[slot * D + d] = (static_cast<int>(vu) - 128) * 0.03125f;
                        }
                    }
                    glob_k_sc[slot]  = 0.01f + 0.001f * static_cast<float>(rng() & 0xFFu);
                    glob_v_sc[slot]  = 0.02f + 0.001f * static_cast<float>(rng() & 0xFFu);
                    glob_k_sum[slot] = static_cast<uint16_t>(ksum);
                }
            }
        }

        if (S > 0) {
            const size_t swa_cap = geom.swa_cap;
            for (size_t l = 0; l < S; ++l) {
                for (size_t kh = 0; kh < H; ++kh) {
                    const size_t lh_base = (l * H + kh) * swa_cap;
                    for (size_t t = 0; t < B; ++t) {
                        const size_t sl = (s0 + t) % swa_cap;
                        for (size_t d = 0; d < D; ++d) {
                            swa_k_f32[(lh_base + sl) * D + d] = static_cast<float>(static_cast<int32_t>(rng() & 0xFFFFu) - 32768) * 1e-4f;
                            swa_v_f32[(lh_base + sl) * D + d] = static_cast<float>(static_cast<int32_t>(rng() & 0xFFFFu) - 32768) * 1e-4f;
                        }
                    }
                }
            }
        }

        for (float &v : conv_state) {
            v = static_cast<float>(static_cast<int32_t>(rng() & 0xFFFFu) - 32768) * 1e-3f;
        }
        for (float &v : last_h) {
            v = static_cast<float>(static_cast<int32_t>(rng() & 0xFFFFu) - 32768) * 1e-3f;
        }
    }
};

void assert_active_state_bit_identical(const SimulatedEngineBuffers &gold,
                                       const SimulatedEngineBuffers &restored,
                                       int blocks_restored,
                                       const char *ctx_label) {
    const ModelCacheGeometry &g = gold.geom;
    const size_t matched_tokens = static_cast<size_t>(blocks_restored) * kBlockTokens;
    const size_t G     = g.num_glob_layers;
    const size_t S     = g.num_swa_layers;
    const size_t H     = g.kv_heads;
    const size_t D     = g.head_dim;
    const size_t s_cap = g.max_ctx;

    // 1. Plane 1: Global KV u8 + sc + k_sum over [0 .. matched_tokens - 1]
    for (size_t l = 0; l < G; ++l) {
        for (size_t kh = 0; kh < H; ++kh) {
            const size_t gh_base = (l * H + kh) * s_cap;
            TEST_ASSERT(std::memcmp(&gold.glob_k_u8[gh_base * D], &restored.glob_k_u8[gh_base * D],
                                    matched_tokens * D) == 0,
                        std::string(ctx_label) + ": glob_k_u8 mismatch");
            TEST_ASSERT(std::memcmp(&gold.glob_v_u8[gh_base * D], &restored.glob_v_u8[gh_base * D],
                                    matched_tokens * D) == 0,
                        std::string(ctx_label) + ": glob_v_u8 mismatch");
            TEST_ASSERT(std::memcmp(&gold.glob_k_sc[gh_base], &restored.glob_k_sc[gh_base],
                                    matched_tokens * sizeof(float)) == 0,
                        std::string(ctx_label) + ": glob_k_sc mismatch");
            TEST_ASSERT(std::memcmp(&gold.glob_v_sc[gh_base], &restored.glob_v_sc[gh_base],
                                    matched_tokens * sizeof(float)) == 0,
                        std::string(ctx_label) + ": glob_v_sc mismatch");
            TEST_ASSERT(std::memcmp(&gold.glob_k_sum[gh_base], &restored.glob_k_sum[gh_base],
                                    matched_tokens * sizeof(uint16_t)) == 0,
                        std::string(ctx_label) + ": glob_k_sum mismatch");

            if (matched_tokens < 256 && !gold.glob_k_f32.empty()) {
                TEST_ASSERT(std::memcmp(&gold.glob_k_f32[gh_base * D], &restored.glob_k_f32[gh_base * D],
                                        matched_tokens * D * sizeof(float)) == 0,
                            std::string(ctx_label) + ": short-ctx glob_k_f32 mismatch");
                TEST_ASSERT(std::memcmp(&gold.glob_v_f32[gh_base * D], &restored.glob_v_f32[gh_base * D],
                                        matched_tokens * D * sizeof(float)) == 0,
                            std::string(ctx_label) + ": short-ctx glob_v_f32 mismatch");
            }
        }
    }

    // 2. Plane 2: SWA active 256-token window [max(0, matched_tokens - 256) .. matched_tokens - 1]
    if (S > 0) {
        const size_t p_beg = (matched_tokens > 256) ? (matched_tokens - 256) : 0;
        for (size_t l = 0; l < S; ++l) {
            for (size_t kh = 0; kh < H; ++kh) {
                const size_t lh_base = (l * H + kh) * g.swa_cap;
                for (size_t p = p_beg; p < matched_tokens; ++p) {
                    const size_t sl = p % g.swa_cap;
                    TEST_ASSERT(std::memcmp(&gold.swa_k_f32[(lh_base + sl) * D],
                                            &restored.swa_k_f32[(lh_base + sl) * D],
                                            D * sizeof(float)) == 0,
                                std::string(ctx_label) + ": SWA K active window mismatch");
                    TEST_ASSERT(std::memcmp(&gold.swa_v_f32[(lh_base + sl) * D],
                                            &restored.swa_v_f32[(lh_base + sl) * D],
                                            D * sizeof(float)) == 0,
                                std::string(ctx_label) + ": SWA V active window mismatch");
                }
            }
        }
    }

    // 3. Plane 3: ShortConv rolling state & Section 0 last_h
    TEST_ASSERT(std::memcmp(gold.conv_state.data(), restored.conv_state.data(),
                            gold.conv_state.size() * sizeof(float)) == 0,
                std::string(ctx_label) + ": ShortConv state mismatch");
    TEST_ASSERT(std::memcmp(gold.last_h.data(), restored.last_h.data(),
                            gold.last_h.size() * sizeof(float)) == 0,
                std::string(ctx_label) + ": last_h mismatch");
}

// -----------------------------------------------------------------------------
// Test 1: LFMKVB01 Layout Invariants & Round-Trip Bit-Exactness (LFM3 + LFM2.5)
// -----------------------------------------------------------------------------

void test_1_round_trip_bit_exactness(const std::filesystem::path &tmp_root) {
    const Hash128 fp_lfm3{0x123456789abcdef0ULL, 0xfedcba9876543210ULL};
    const Hash128 fp_moe{0xaabbccddeeff0011ULL, 0x2233445566778899ULL};

    ModelCacheGeometry g_lfm3 = ModelCacheGeometry::make_lfm3("lfm3-1.7b-tp2", fp_lfm3, 2048);
    ModelCacheGeometry g_moe  = ModelCacheGeometry::make_lfm2_moe("lfm2.5-8b-a1b", fp_moe, 2048);

    // Verify exact byte-layout table against PLAN.md §8.2 & Phase F.1
    const auto L3_root = KvbSectionLayout::compute(g_lfm3, true);
    const auto L3_steady = KvbSectionLayout::compute(g_lfm3, false);
    TEST_ASSERT(L3_steady.sec0_bytes          == 12288,   "LFM3 sec0 != 12288 B");
    TEST_ASSERT(L3_steady.len_glob_u8_raw     == 408576,  "LFM3 glob_u8 raw != 408576 B");
    TEST_ASSERT(L3_steady.len_glob_u8_aligned == 409600,  "LFM3 glob_u8 aligned != 409600 B");
    TEST_ASSERT(L3_steady.len_swa_f32         == 1310720, "LFM3 swa_f32 != 1310720 B");
    TEST_ASSERT(L3_steady.len_conv_f32        == 638976,  "LFM3 conv_f32 != 638976 B");
    TEST_ASSERT(L3_steady.total_file_bytes    == 2371584, "LFM3 steady .kvb != 2,371,584 B");
    TEST_ASSERT(L3_root.total_file_bytes      == 3944448, "LFM3 root .kvb != 3,944,448 B");

    const auto L2_root = KvbSectionLayout::compute(g_moe, true);
    const auto L2_steady = KvbSectionLayout::compute(g_moe, false);
    TEST_ASSERT(L2_steady.sec0_bytes          == 12288,   "LFM2.5 sec0 != 12288 B");
    TEST_ASSERT(L2_steady.len_glob_u8_raw     == 423936,  "LFM2.5 glob_u8 raw != 423936 B");
    TEST_ASSERT(L2_steady.len_glob_u8_aligned == 425984,  "LFM2.5 glob_u8 aligned != 425984 B");
    TEST_ASSERT(L2_steady.len_swa_f32         == 0,       "LFM2.5 swa_f32 != 0 B");
    TEST_ASSERT(L2_steady.len_conv_f32        == 294912,  "LFM2.5 conv_f32 != 294912 B");
    TEST_ASSERT(L2_steady.total_file_bytes    == 733184,  "LFM2.5 steady .kvb != 733,184 B");
    TEST_ASSERT(L2_root.total_file_bytes      == 2306048, "LFM2.5 root .kvb != 2,306,048 B");

    for (const ModelCacheGeometry &geom : {g_lfm3, g_moe}) {
        const bool is_lfm3 = (geom.arch_id == ModelArchId::LFM3_1_7B);
        std::vector<int32_t> tokens(10 * kBlockTokens);
        for (size_t i = 0; i < tokens.size(); ++i) {
            tokens[i] = static_cast<int32_t>(1000 + (i * 17) % 50000);
        }

        SimulatedEngineBuffers sim(geom);
        std::vector<SimulatedEngineBuffers> snaps;
        snaps.reserve(10);

        {
            RadixCache cache;
            TEST_ASSERT(cache.init(tmp_root, geom), "RadixCache::init failed");

            PrefixLookupResult miss = cache.lookup(tokens.data(), tokens.size());
            TEST_ASSERT(miss.matched_blocks == 0, "Cold lookup must return 0 matched blocks");
            TEST_ASSERT(miss.chain_keys.size() == 10, "Expected 10 chain keys for 640 tokens");

            for (uint32_t k = 0; k < 10; ++k) {
                sim.simulate_prefill_block(k, is_lfm3 ? 101ULL : 202ULL);
                snaps.push_back(sim);
                Hash128 parent = (k == 0) ? Hash128{} : miss.chain_keys[k - 1];
                EngineStateView v = sim.view();
                cache.enqueue_save_block(k, parent, miss.chain_keys[k], miss.vision_hashes[k],
                                         tokens.data() + k * kBlockTokens, sim.last_h.data(), v);
            }

            // Exercise in-flight memory hit BEFORE flushing async disk writes, then flush
            PrefixLookupResult hot_hit = cache.lookup(tokens.data(), tokens.size());
            TEST_ASSERT(hot_hit.matched_blocks == 10 && hot_hit.matched_tokens == 640,
                        "In-flight lookup must match all 10 blocks");
            cache.flush_async_writes();
        } // Destructor writes index.snap and shuts down writer thread

        // Re-open fresh RadixCache from persisted disk (.kvb + index.snap)
        RadixCache reopened;
        TEST_ASSERT(reopened.init(tmp_root, geom), "Re-init from persisted index.snap failed");
        TEST_ASSERT(reopened.stats().resident_blocks == 10, "Expected 10 persisted blocks after restart");

        PrefixLookupResult disk_hit = reopened.lookup(tokens.data(), tokens.size());
        TEST_ASSERT(disk_hit.matched_blocks == 10, "Persisted lookup must match all 10 blocks");

        // Restore at k = 2 (3 blocks = 192 tokens < 256 -> exercises Section 4 glob_f32)
        SimulatedEngineBuffers rest_k2(geom);
        rest_k2.poison(0xA5);
        EngineStateView v_k2 = rest_k2.view();
        double ms_k2 = 0.0;
        TEST_ASSERT(reopened.restore_blocks(disk_hit, 3, v_k2, rest_k2.last_h.data(), ms_k2, tokens.data()),
                    "restore_blocks(k=2) failed");
        assert_active_state_bit_identical(snaps[2], rest_k2, 3, is_lfm3 ? "LFM3 k=2" : "LFM2.5 k=2");

        // Restore at k = 9 (10 blocks = 640 tokens -> exercises SWA 320-slot ring wrap-around)
        SimulatedEngineBuffers rest_k9(geom);
        rest_k9.poison(0x5A);
        EngineStateView v_k9 = rest_k9.view();
        double ms_k9 = 0.0;
        TEST_ASSERT(reopened.restore_blocks(disk_hit, 10, v_k9, rest_k9.last_h.data(), ms_k9, tokens.data()),
                    "restore_blocks(k=9) failed");
        assert_active_state_bit_identical(snaps[9], rest_k9, 10, is_lfm3 ? "LFM3 k=9" : "LFM2.5 k=9");
    }

    ok("Test 1: LFMKVB01 byte layout, WAL reboot recovery & LFM3/LFM2.5 bit-exactness (k=2 & k=9)");
}

// -----------------------------------------------------------------------------
// Test 2: Divergent Branching, SWA Ring Isolation & Multimodal Vision Collision Guard
// -----------------------------------------------------------------------------

void test_2_divergent_branching_and_vision(const std::filesystem::path &tmp_root) {
    const Hash128 fp{0x9988776655443322ULL, 0x1100ffeeddccbbaaULL};
    ModelCacheGeometry geom = ModelCacheGeometry::make_lfm3("lfm3-branch-test", fp, 2048);

    RadixCache cache;
    TEST_ASSERT(cache.init(tmp_root, geom), "init failed");

    // Sequence A (8 blocks = 512 tok) and Sequence B (9 blocks = 576 tok) share chunks 0..3 (256 tok)
    std::vector<int32_t> seq_a(8 * kBlockTokens);
    std::vector<int32_t> seq_b(9 * kBlockTokens);
    for (size_t i = 0; i < 4 * kBlockTokens; ++i) {
        int32_t tok = static_cast<int32_t>(200 + i);
        seq_a[i] = tok;
        seq_b[i] = tok;
    }
    for (size_t i = 4 * kBlockTokens; i < seq_a.size(); ++i) seq_a[i] = static_cast<int32_t>(5000 + i);
    for (size_t i = 4 * kBlockTokens; i < seq_b.size(); ++i) seq_b[i] = static_cast<int32_t>(9000 + i);

    SimulatedEngineBuffers sim_a(geom);
    PrefixLookupResult lk_a0 = cache.lookup(seq_a.data(), seq_a.size());
    for (uint32_t k = 0; k < 8; ++k) {
        sim_a.simulate_prefill_block(k, (k < 4) ? 777ULL : 888ULL);
        Hash128 parent = (k == 0) ? Hash128{} : lk_a0.chain_keys[k - 1];
        EngineStateView va = sim_a.view();
        cache.enqueue_save_block(k, parent, lk_a0.chain_keys[k], Hash128{},
                                 seq_a.data() + k * kBlockTokens, sim_a.last_h.data(), va);
    }
    const SimulatedEngineBuffers snap_a_k7 = sim_a;

    // Lookup Sequence B: must hit exact shared 4-block prefix (256 tokens) and diverge at block 4
    PrefixLookupResult lk_b = cache.lookup(seq_b.data(), seq_b.size());
    TEST_ASSERT(lk_b.matched_blocks == 4 && lk_b.matched_tokens == 256,
                "Branch B must match exactly 4 shared prefix blocks (256 tokens)");

    SimulatedEngineBuffers sim_b(geom);
    sim_b.poison(0xCC);
    EngineStateView vb = sim_b.view();
    double ms = 0.0;
    TEST_ASSERT(cache.restore_blocks(lk_b, 4, vb, sim_b.last_h.data(), ms, seq_b.data()),
                "Restore shared prefix for Branch B failed");

    // Advance Branch B through blocks 4..8 (overwriting SWA ring slots & conv_state)
    for (uint32_t k = 4; k < 9; ++k) {
        sim_b.simulate_prefill_block(k, 999ULL);
        EngineStateView vb_step = sim_b.view();
        cache.enqueue_save_block(k, lk_b.chain_keys[k - 1], lk_b.chain_keys[k], Hash128{},
                                 seq_b.data() + k * kBlockTokens, sim_b.last_h.data(), vb_step);
    }
    const SimulatedEngineBuffers snap_b_k8 = sim_b;
    cache.flush_async_writes();

    // Switch back to Sequence A at k = 7 and verify zero cross-branch SWA/Conv aliasing
    PrefixLookupResult lk_a_again = cache.lookup(seq_a.data(), seq_a.size());
    TEST_ASSERT(lk_a_again.matched_blocks == 8, "Sequence A must still match all 8 blocks");
    SimulatedEngineBuffers check_a(geom);
    check_a.poison(0x11);
    EngineStateView v_ca = check_a.view();
    TEST_ASSERT(cache.restore_blocks(lk_a_again, 8, v_ca, check_a.last_h.data(), ms, seq_a.data()),
                "Restore Sequence A at k=7 failed");
    assert_active_state_bit_identical(snap_a_k7, check_a, 8, "Branch A k=7 isolation");

    // Switch back to Sequence B at k = 8 and verify bit-identical restore
    PrefixLookupResult lk_b_again = cache.lookup(seq_b.data(), seq_b.size());
    TEST_ASSERT(lk_b_again.matched_blocks == 9, "Sequence B must match all 9 blocks");
    SimulatedEngineBuffers check_b(geom);
    check_b.poison(0x22);
    EngineStateView v_cb = check_b.view();
    TEST_ASSERT(cache.restore_blocks(lk_b_again, 9, v_cb, check_b.last_h.data(), ms, seq_b.data()),
                "Restore Sequence B at k=8 failed");
    assert_active_state_bit_identical(snap_b_k8, check_b, 9, "Branch B k=8 isolation");

    // Verify multimodal <|image|> (125030) hash disambiguation on identical token IDs
    std::vector<int32_t> img_toks(kBlockTokens, 42);
    img_toks[10] = 125030;
    img_toks[11] = 125030;
    std::vector<float> vis_img1(2 * geom.hidden_dim, 0.25f);
    std::vector<float> vis_img2(2 * geom.hidden_dim, -0.75f);
    auto r_v1 = cache.lookup(img_toks.data(), img_toks.size(), vis_img1.data(), 2, 125030);
    auto r_v2 = cache.lookup(img_toks.data(), img_toks.size(), vis_img2.data(), 2, 125030);
    TEST_ASSERT(r_v1.chain_keys[0] != r_v2.chain_keys[0],
                "Distinct visual_embeds on identical <|image|> tokens must never collide");

    ok("Test 2: Divergent branching (13 unique blocks), SWA ring isolation & <|image|> hash guard");
}

// -----------------------------------------------------------------------------
// Test 3: Model Fingerprint Invalidation & Per-Section Checksum Corruption Guard
// -----------------------------------------------------------------------------

void test_3_fingerprint_and_corruption_guard(const std::filesystem::path &tmp_root) {
    const Hash128 fp_v1{0xdeadbeefcafebabeULL, 0x0123456789abcdefULL};
    ModelCacheGeometry geom_v1 = ModelCacheGeometry::make_lfm2_moe("lfm2-corrupt-test", fp_v1, 2048);

    std::vector<int32_t> tokens(6 * kBlockTokens);
    for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = static_cast<int32_t>(300 + i);

    PrefixLookupResult keys_v1;
    {
        RadixCache cache;
        TEST_ASSERT(cache.init(tmp_root, geom_v1), "init v1 failed");
        keys_v1 = cache.lookup(tokens.data(), tokens.size());
        SimulatedEngineBuffers sim(geom_v1);
        for (uint32_t k = 0; k < 6; ++k) {
            sim.simulate_prefill_block(k, 555ULL);
            Hash128 parent = (k == 0) ? Hash128{} : keys_v1.chain_keys[k - 1];
            EngineStateView v = sim.view();
            cache.enqueue_save_block(k, parent, keys_v1.chain_keys[k], Hash128{},
                                     tokens.data() + k * kBlockTokens, sim.last_h.data(), v);
        }
        cache.flush_async_writes();
    }

    // 3a. Flip 1 bit in model_fingerprint (simulating Milestone 16 / Phase E dyntok re-calibration)
    Hash128 fp_v2 = fp_v1;
    fp_v2.lo ^= 1ULL;
    ModelCacheGeometry geom_v2 = ModelCacheGeometry::make_lfm2_moe("lfm2-corrupt-test-v2", fp_v2, 2048);
    {
        RadixCache cache_v2;
        TEST_ASSERT(cache_v2.init(tmp_root, geom_v2), "init v2 failed");
        PrefixLookupResult hit_v2 = cache_v2.lookup(tokens.data(), tokens.size());
        TEST_ASSERT(hit_v2.matched_blocks == 0, "1-bit model_fingerprint change must produce 0 hits");
        TEST_ASSERT(hit_v2.chain_keys[0] != keys_v1.chain_keys[0],
                    "model_fingerprint change must alter root block_key");
    }

    // 3b. Flip 1 byte inside Section 1 (glob_u8) of block 2 on disk and verify clean rejection
    {
        RadixCache cache;
        TEST_ASSERT(cache.init(tmp_root, geom_v1), "re-init v1 failed");
        PrefixLookupResult hit = cache.lookup(tokens.data(), tokens.size());
        TEST_ASSERT(hit.matched_blocks == 6, "Expected 6 blocks before corruption");

        std::filesystem::path bad_file = cache.block_file_path(hit.chain_keys[2]);
        {
            std::fstream fs(bad_file, std::ios::in | std::ios::out | std::ios::binary);
            TEST_ASSERT(static_cast<bool>(fs), "Failed to open block 2 .kvb for bit-flip test");
            const std::streamoff sec1_byte_off = 12288 + 1024; // Inside Section 1 (glob_u8)
            fs.seekg(sec1_byte_off);
            char b = 0;
            fs.read(&b, 1);
            b ^= static_cast<char>(0xFF);
            fs.seekp(sec1_byte_off);
            fs.write(&b, 1);
            fs.flush();
        }

        SimulatedEngineBuffers dest(geom_v1);
        EngineStateView v = dest.view();
        double ms = 0.0;
        bool ok_restore = cache.restore_blocks(hit, 6, v, dest.last_h.data(), ms, tokens.data());
        TEST_ASSERT(!ok_restore, "Corrupted Section 1 .kvb must be rejected by checksum");

        // Subtree [2..5] must be automatically quarantined, leaving clean prefix [0..1] intact!
        PrefixLookupResult after = cache.lookup(tokens.data(), tokens.size());
        TEST_ASSERT(after.matched_blocks == 2 && after.matched_tokens == 128,
                    "Corrupted block 2 subtree [2..5] must be quarantined, leaving blocks [0..1]");
        TEST_ASSERT(cache.restore_blocks(after, 2, v, dest.last_h.data(), ms, tokens.data()),
                    "Clean prefix [0..1] must still restore cleanly after quarantine");
    }

    ok("Test 3: 1-bit model_fingerprint invalidation & 1-byte Section 1 corruption quarantine");
}

// -----------------------------------------------------------------------------
// Test 4: Leaf-First GDSF Eviction Under 10 MiB Quota & Immortal Root Pinning
// -----------------------------------------------------------------------------

void test_4_leaf_first_gdsf_eviction_and_pinning(const std::filesystem::path &tmp_root) {
    const Hash128 fp{0xfeedface12345678ULL, 0x87654321deadc0deULL};
    ModelCacheGeometry geom = ModelCacheGeometry::make_lfm2_moe("lfm2-gdsf-evict", fp, 2048);
    // Disable optional short-ctx FP32 plane so every block has uniform 733,184 B (716 KiB) size:
    // 10 MiB = 10,485,760 B -> holds at most 14 blocks (14 * 733,184 = 10,264,576 B <= 10 MiB).
    SimulatedEngineBuffers sim(geom, /*alloc_glob_f32=*/false);
    EngineStateView view = sim.view();

    const uint64_t quota_10_mib = 10ULL * 1024 * 1024;
    RadixCache cache;
    TEST_ASSERT(cache.init(tmp_root, geom, quota_10_mib), "init with 10 MiB quota failed");

    // 1. Insert & PIN 2-block root system prompt (blocks 0..1 = 128 tokens)
    std::vector<int32_t> sys_prefix(2 * kBlockTokens);
    for (size_t i = 0; i < sys_prefix.size(); ++i) sys_prefix[i] = static_cast<int32_t>(10 + i);

    PrefixLookupResult sys_lk = cache.lookup(sys_prefix.data(), sys_prefix.size());
    for (uint32_t k = 0; k < 2; ++k) {
        sim.simulate_prefill_block(k, 1ULL);
        Hash128 parent = (k == 0) ? Hash128{} : sys_lk.chain_keys[k - 1];
        cache.enqueue_save_block(k, parent, sys_lk.chain_keys[k], Hash128{},
                                 sys_prefix.data() + k * kBlockTokens, sim.last_h.data(), view,
                                 /*pinned=*/true);
    }

    // 2. Insert 20 branches under blocks 0..1 (each branch adds block 2 + block 3 = 2 blocks;
    //    total inserted = 2 pinned + 40 branch blocks = 42 blocks = 29.4 MiB >> 10 MiB quota).
    //    Make branch #0 "hot" by giving it 15 lookup hits on every iteration!
    std::vector<std::vector<int32_t>> branches(20);
    for (int br = 0; br < 20; ++br) {
        branches[br] = sys_prefix;
        branches[br].resize(4 * kBlockTokens);
        for (size_t i = 2 * kBlockTokens; i < 4 * kBlockTokens; ++i) {
            branches[br][i] = static_cast<int32_t>(10000 + br * 500 + i);
        }

        PrefixLookupResult blk_lk = cache.lookup(branches[br].data(), branches[br].size());
        TEST_ASSERT(blk_lk.matched_blocks == 2, "Pinned root blocks 0..1 must always hit!");

        for (uint32_t k = 2; k < 4; ++k) {
            sim.simulate_prefill_block(k, static_cast<uint64_t>(100 + br));
            cache.enqueue_save_block(k, blk_lk.chain_keys[k - 1], blk_lk.chain_keys[k], Hash128{},
                                     branches[br].data() + k * kBlockTokens, sim.last_h.data(), view,
                                     /*pinned=*/false);
        }

        // Boost GDSF frequency of branch 0 so it stays hot against cold single-use branches
        for (int h = 0; h < 6; ++h) {
            auto hot_lk = cache.lookup(branches[0].data(), branches[0].size());
            TEST_ASSERT(hot_lk.matched_blocks == 4, "Hot branch 0 must outlive cold branches under GDSF");
        }
    }

    cache.flush_async_writes();
    CacheStatsSnapshot st = cache.stats();

    TEST_ASSERT(st.disk_bytes_used <= quota_10_mib,
                "Disk usage exceeded 10 MiB quota");
    TEST_ASSERT(st.pinned_blocks == 2,
                "Pinned block count must remain 2");
    TEST_ASSERT(st.total_evictions >= 28,
                "Expected >= 28 leaf blocks evicted to stay within 14-block (10 MiB) budget");

    // Pinned root 0..1 and hot branch 0 must both survive 100%
    TEST_ASSERT(cache.lookup(sys_prefix.data(), sys_prefix.size()).matched_blocks == 2,
                "Pinned root 0..1 was evicted");
    TEST_ASSERT(cache.lookup(branches[0].data(), branches[0].size()).matched_blocks == 4,
                "Hot branch 0 should survive GDSF eviction");

    // Clear unpinned entries: pinned root 0..1 must survive; clear(true) removes everything
    cache.clear(/*include_pinned=*/false);
    TEST_ASSERT(cache.stats().resident_blocks == 2 && cache.stats().pinned_blocks == 2,
                "clear(false) must retain only the 2 pinned root blocks");
    cache.clear(/*include_pinned=*/true);
    TEST_ASSERT(cache.stats().resident_blocks == 0 && cache.stats().disk_bytes_used == 0,
                "clear(true) must empty cache completely");

    ok("Test 4: Leaf-first GDSF eviction (10 MiB quota), hot-branch retention & root pinning");
}

} // namespace

int main() {
    auto t0 = std::chrono::steady_clock::now();
    std::filesystem::path tmp_root =
        std::filesystem::temp_directory_path() / ("romanaccio_test_kvb_" + std::to_string(::getpid()));
    std::error_code ec;
    std::filesystem::remove_all(tmp_root, ec);

    std::cout << "=== Milestone 17A: Hybrid Prefix Cache (LFMKVB01 + RadixCache) Host Suite ===" << std::endl;

    test_1_round_trip_bit_exactness(tmp_root / "t1");
    test_2_divergent_branching_and_vision(tmp_root / "t2");
    test_3_fingerprint_and_corruption_guard(tmp_root / "t3");
    test_4_leaf_first_gdsf_eviction_and_pinning(tmp_root / "t4");

    std::filesystem::remove_all(tmp_root, ec);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "\nAll " << g_pass << "/4 Milestone 17A Prefix Cache tests PASSED in "
              << elapsed_ms << " ms (" << (elapsed_ms / 1000.0) << " s <= 2.0 s gate)." << std::endl;
    TEST_ASSERT(elapsed_ms <= 2000.0, "Unit test runtime exceeded 2.0 s acceptance gate");
    return 0;
}
