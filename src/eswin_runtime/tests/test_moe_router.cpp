/**
 * @file test_moe_router.cpp
 * @brief Phase 2 test suite: MoE gate, top-k selection and per-die dispatch grouping.
 *
 * Correctness is checked against an independent double-precision reference implemented
 * inline here, which shares no code with MoERouter — so a bug in the router's softmax,
 * selection or normalisation cannot be mirrored by the thing checking it.
 *
 * The default configuration is the verified DeepSeek-V2-Lite one: 64 experts, top-6,
 * softmax scoring, norm_topk_prob=false, 2-die expert-parallel split.
 */

#include "../src/llm/moe_router.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
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

using namespace eswin;

namespace {

// --- Independent reference (double precision, deliberately naive) --------------

struct RefOut {
    std::vector<uint32_t> idx;
    std::vector<double>   w;
    double sum = 0.0;
    uint32_t argmax = 0;
};

RefOut reference_route(const std::vector<float> &gate, const std::vector<float> &x,
                       uint32_t n_exp, uint32_t dim, uint32_t k, bool sigmoid,
                       bool norm, double scaling) {
    std::vector<double> logits(n_exp, 0.0);
    for (uint32_t e = 0; e < n_exp; ++e) {
        double acc = 0.0;
        for (uint32_t i = 0; i < dim; ++i) {
            acc += static_cast<double>(gate[static_cast<size_t>(e) * dim + i]) *
                   static_cast<double>(x[i]);
        }
        logits[e] = acc;
    }

    std::vector<double> sc(n_exp, 0.0);
    if (sigmoid) {
        for (uint32_t e = 0; e < n_exp; ++e) sc[e] = 1.0 / (1.0 + std::exp(-logits[e]));
    } else {
        double mx = *std::max_element(logits.begin(), logits.end());
        double s = 0.0;
        for (uint32_t e = 0; e < n_exp; ++e) { sc[e] = std::exp(logits[e] - mx); s += sc[e]; }
        for (uint32_t e = 0; e < n_exp; ++e) sc[e] /= s;
    }

    uint32_t argmax = 0;
    for (uint32_t e = 1; e < n_exp; ++e) if (sc[e] > sc[argmax]) argmax = e;

    std::vector<uint32_t> ord(n_exp);
    std::iota(ord.begin(), ord.end(), 0u);
    std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
        if (sc[a] != sc[b]) return sc[a] > sc[b];
        return a < b;
    });

    RefOut out;
    double sum = 0.0;
    for (uint32_t i = 0; i < k; ++i) {
        out.idx.push_back(ord[i]);
        out.w.push_back(sc[ord[i]]);
        sum += sc[ord[i]];
    }
    if (norm && sum > 0.0) {
        for (auto &w : out.w) w /= sum;
        sum = 1.0;
    }
    for (auto &w : out.w) w *= scaling;
    out.sum = sum * scaling;
    out.argmax = argmax;
    return out;
}

// Verified DeepSeek-V2-Lite gate configuration.
MoERouterConfig v2lite_config() {
    MoERouterConfig c;
    c.hidden_size = 2048;
    c.num_experts = 64;
    c.top_k = 6;
    c.num_shared_experts = 2;
    c.scoring = MoEScoring::Softmax;
    c.topk_method = MoETopKMethod::Greedy;
    c.norm_topk_prob = false;   // confirmed from config.json
    c.n_group = 1;
    c.topk_group = 1;
    c.routed_scaling_factor = 1.0f;
    c.has_gate_bias = false;
    c.num_dies = 2;
    return c;
}

std::vector<float> random_gate(uint32_t n_exp, uint32_t dim, unsigned seed, float scale = 0.02f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> g(static_cast<size_t>(n_exp) * dim);
    for (auto &v : g) v = nd(rng);
    return g;
}

std::vector<float> random_hidden(uint32_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> h(dim);
    for (auto &v : h) v = nd(rng);
    return h;
}

// --- Tests --------------------------------------------------------------------

void test_v2lite_against_reference() {
    std::cout << "[*] Differential test vs an independent reference (V2-Lite config)..."
              << std::endl;

    auto cfg = v2lite_config();
    auto gate = random_gate(cfg.num_experts, cfg.hidden_size, 12345);

    MoERouter router;
    std::string err;
    TEST_ASSERT(router.init(cfg, gate.data(), nullptr, err), "init failed: " + err);
    TEST_ASSERT(router.is_ready(), "router not ready after init");

    int mismatch = 0;
    for (int trial = 0; trial < 200; ++trial) {
        auto x = random_hidden(cfg.hidden_size, static_cast<unsigned>(trial) + 1);

        MoERouterResult got;
        TEST_ASSERT(router.route(x.data(), got), "route failed");

        RefOut want = reference_route(gate, x, cfg.num_experts, cfg.hidden_size, cfg.top_k,
                                      false, cfg.norm_topk_prob, cfg.routed_scaling_factor);

        if (got.routed.size() != want.idx.size()) { ++mismatch; continue; }
        for (size_t i = 0; i < want.idx.size(); ++i) {
            if (got.routed[i].index != want.idx[i]) { ++mismatch; break; }
            if (std::fabs(got.routed[i].score - want.w[i]) > 1e-5) { ++mismatch; break; }
        }
        if (got.argmax_expert != want.argmax) ++mismatch;
    }
    TEST_ASSERT(mismatch == 0, std::to_string(mismatch) + "/200 trials disagreed with the reference");
    std::cout << "    200 random tokens: expert set, ordering, weights and argmax all match."
              << std::endl;
}

void test_unnormalised_weights() {
    std::cout << "\n[*] Testing norm_topk_prob semantics (the V2-Lite trap)..." << std::endl;

    auto gate = random_gate(64, 256, 777);
    auto x = random_hidden(256, 778);

    // V2-Lite: norm_topk_prob = false -> raw softmax probabilities, summing to < 1.
    MoERouterConfig off = v2lite_config();
    off.hidden_size = 256;
    off.norm_topk_prob = false;
    MoERouter r_off;
    std::string err;
    TEST_ASSERT(r_off.init(off, gate.data(), nullptr, err), "init failed: " + err);
    MoERouterResult got_off;
    TEST_ASSERT(r_off.route(x.data(), got_off), "route failed");
    TEST_ASSERT(got_off.score_sum < 1.0f,
                "unnormalised top-6 softmax weights must sum to less than 1");
    TEST_ASSERT(got_off.score_sum > 0.0f, "scores must be positive");

    // With normalisation the same selection must sum to exactly 1.
    MoERouterConfig on = off;
    on.norm_topk_prob = true;
    MoERouter r_on;
    TEST_ASSERT(r_on.init(on, gate.data(), nullptr, err), "init failed: " + err);
    MoERouterResult got_on;
    TEST_ASSERT(r_on.route(x.data(), got_on), "route failed");
    TEST_ASSERT(std::fabs(got_on.score_sum - 1.0f) < 1e-5f,
                "normalised weights must sum to 1");

    // Same experts selected either way; only the scaling differs.
    for (size_t i = 0; i < got_off.routed.size(); ++i) {
        TEST_ASSERT(got_off.routed[i].index == got_on.routed[i].index,
                    "normalisation changed which experts were selected");
    }
    const float ratio = got_on.routed[0].score / got_off.routed[0].score;
    TEST_ASSERT(std::fabs(ratio - 1.0f / got_off.score_sum) < 1e-4f,
                "normalisation is not a uniform rescale");

    std::cout << "    norm=false sums to " << got_off.score_sum
              << " (<1, correct); norm=true sums to " << got_on.score_sum
              << "; same expert set, uniform rescale." << std::endl;
}

void test_sigmoid_and_scaling() {
    std::cout << "\n[*] Testing sigmoid scoring and routed_scaling_factor..." << std::endl;

    auto gate = random_gate(64, 128, 31337);
    auto x = random_hidden(128, 31338);
    std::string err;

    MoERouterConfig cfg = v2lite_config();
    cfg.hidden_size = 128;
    cfg.scoring = MoEScoring::Sigmoid;
    MoERouter r;
    TEST_ASSERT(r.init(cfg, gate.data(), nullptr, err), "init failed: " + err);
    MoERouterResult got;
    TEST_ASSERT(r.route(x.data(), got), "route failed");

    RefOut want = reference_route(gate, x, 64, 128, 6, true, false, 1.0);
    for (size_t i = 0; i < want.idx.size(); ++i) {
        TEST_ASSERT(got.routed[i].index == want.idx[i], "sigmoid selection mismatch");
        TEST_ASSERT(std::fabs(got.routed[i].score - want.w[i]) < 1e-5, "sigmoid score mismatch");
    }

    // routed_scaling_factor multiplies every selected weight.
    MoERouterConfig scaled = cfg;
    scaled.routed_scaling_factor = 2.5f;
    MoERouter r2;
    TEST_ASSERT(r2.init(scaled, gate.data(), nullptr, err), "init failed: " + err);
    MoERouterResult got2;
    TEST_ASSERT(r2.route(x.data(), got2), "route failed");
    for (size_t i = 0; i < got.routed.size(); ++i) {
        TEST_ASSERT(std::fabs(got2.routed[i].score - got.routed[i].score * 2.5f) < 1e-5f,
                    "routed_scaling_factor not applied");
    }
    std::cout << "    Sigmoid matches the reference; scaling factor applies uniformly."
              << std::endl;
}

void test_gate_bias() {
    std::cout << "\n[*] Testing an optional gate bias..." << std::endl;

    const uint32_t n_exp = 8, dim = 32;
    std::vector<float> gate(n_exp * dim, 0.0f);   // all-zero weights: logits are bias only
    std::vector<float> bias(n_exp, 0.0f);
    bias[5] = 10.0f;                              // expert 5 must always win
    std::vector<float> x(dim, 1.0f);

    MoERouterConfig cfg = v2lite_config();
    cfg.hidden_size = dim;
    cfg.num_experts = n_exp;
    cfg.top_k = 3;
    cfg.has_gate_bias = true;
    cfg.num_dies = 2;

    MoERouter r;
    std::string err;
    TEST_ASSERT(r.init(cfg, gate.data(), bias.data(), err), "init failed: " + err);
    MoERouterResult got;
    TEST_ASSERT(r.route(x.data(), got), "route failed");
    TEST_ASSERT(got.routed[0].index == 5, "biased expert should rank first");
    TEST_ASSERT(got.argmax_expert == 5, "argmax should be the biased expert");
    TEST_ASSERT(got.max_score > 0.9f, "softmax over a 10-logit gap should be near 1");
    std::cout << "    Bias-only logits rank expert 5 first with score " << got.max_score
              << std::endl;
}

void test_group_limited_greedy() {
    std::cout << "\n[*] Testing group-limited greedy selection (V3-family path)..." << std::endl;

    // 8 experts in 4 groups of 2. The group holding the global maximum is ALWAYS
    // selected (groups are ranked by their own best score), so masking can only ever
    // exclude an expert that is not its group's best-ranked representative. Here the
    // global top-3 spans groups 0, 1 and 2, but only groups 0 and 1 are kept, so
    // expert 4 must be dropped in favour of the next-best expert inside a kept group.
    const uint32_t n_exp = 8, dim = 16;
    std::vector<float> gate(n_exp * dim, 0.0f);   // zero weights: logits come from the bias
    std::vector<float> bias(n_exp, -5.0f);
    bias[0] = 10.0f;   // group 0, best overall
    bias[2] = 9.0f;    // group 1
    bias[4] = 8.0f;    // group 2 - third globally, but its group loses
    bias[6] = 7.0f;    // group 3
    std::vector<float> x(dim, 1.0f);

    MoERouterConfig cfg = v2lite_config();
    cfg.hidden_size = dim;
    cfg.num_experts = n_exp;
    cfg.top_k = 3;
    cfg.has_gate_bias = true;
    cfg.n_group = 4;
    cfg.topk_group = 2;
    cfg.topk_method = MoETopKMethod::GroupLimitedGreedy;

    MoERouter r;
    std::string err;
    TEST_ASSERT(r.init(cfg, gate.data(), bias.data(), err), "init failed: " + err);
    MoERouterResult got;
    TEST_ASSERT(r.route(x.data(), got), "route failed");

    bool picked_four = false;
    for (const auto &re : got.routed) if (re.index == 4) picked_four = true;
    TEST_ASSERT(!picked_four,
                "group-limited greedy must drop an expert whose group was not selected");
    TEST_ASSERT(got.routed[0].index == 0, "global best should rank first");
    TEST_ASSERT(got.routed[1].index == 2, "second-best (in a kept group) should rank second");
    TEST_ASSERT(got.routed[2].index == 1,
                "third slot should fall to the best remaining expert in a kept group");
    TEST_ASSERT(got.argmax_expert == 0, "argmax should report the true global maximum");

    // With topk_group == n_group nothing is masked, so it degenerates to plain greedy
    // and expert 4 comes back.
    MoERouterConfig degen = cfg;
    degen.topk_group = 4;
    MoERouter r2;
    TEST_ASSERT(r2.init(degen, gate.data(), bias.data(), err), "init failed: " + err);
    MoERouterResult got2;
    TEST_ASSERT(r2.route(x.data(), got2), "route failed");
    TEST_ASSERT(got2.routed[0].index == 0 && got2.routed[1].index == 2 &&
                got2.routed[2].index == 4,
            "with every group selected, selection must be plain global top-3");

    std::cout << "    Groups 0,1 kept -> expert 4 (3rd globally) dropped for expert 1;"
                 "\n    topk_group == n_group degenerates to plain greedy top-3." << std::endl;
}

void test_die_grouping_and_offsets() {
    std::cout << "\n[*] Testing expert-parallel die grouping and blob offsets..." << std::endl;

    auto cfg = v2lite_config();
    auto gate = random_gate(cfg.num_experts, cfg.hidden_size, 4242);
    MoERouter r;
    std::string err;
    TEST_ASSERT(r.init(cfg, gate.data(), nullptr, err), "init failed: " + err);

    TEST_ASSERT(r.experts_per_die() == 32, "64 experts over 2 dies should be 32 each");
    TEST_ASSERT(r.die_for_expert(0) == 0 && r.die_for_expert(31) == 0, "experts 0-31 on die 0");
    TEST_ASSERT(r.die_for_expert(32) == 1 && r.die_for_expert(63) == 1, "experts 32-63 on die 1");

    // Offsets are relative to *that die's own* blob, because each die has a separate
    // DmaBuffer/mem_fd. So die 0 and die 1 both legitimately start at 0; uniqueness and
    // density must be checked per die, not globally.
    const size_t kBytes = 8u * 1024u * 1024u;   // 8.05 MiB expert, rounded
    for (uint16_t die = 0; die < 2; ++die) {
        std::vector<size_t> seen;
        for (uint32_t layer = 0; layer < 3; ++layer) {
            for (uint32_t slot = 0; slot < 32; ++slot) {
                const uint32_t e = static_cast<uint32_t>(die) * 32 + slot;
                seen.push_back(r.expert_blob_offset(layer, e, kBytes));
            }
        }
        std::sort(seen.begin(), seen.end());
        TEST_ASSERT(std::adjacent_find(seen.begin(), seen.end()) == seen.end(),
                    "offsets collide within die " + std::to_string(die) + "'s blob");
        TEST_ASSERT(seen.size() == 96, "expected 3 layers x 32 slots");
        TEST_ASSERT(seen.front() == 0, "die blob should start at offset 0");
        TEST_ASSERT(seen.back() == 95 * kBytes, "offsets should be dense within the die");
    }

    // Layer 1 expert 0 sits exactly one die-blob stride (32 experts) past layer 0.
    TEST_ASSERT(r.expert_blob_offset(1, 0, kBytes) == 32 * kBytes, "per-layer stride wrong");
    // Expert 32 is slot 0 on die 1, so it starts that die's blob at 0.
    TEST_ASSERT(r.expert_blob_offset(0, 32, kBytes) == 0,
                "die 1 slot 0 should start at offset 0 in its own blob");
    // And within a die, consecutive experts are consecutive slots.
    TEST_ASSERT(r.expert_blob_offset(2, 33, kBytes) == (2 * 32 + 1) * kBytes,
                "die 1 slot indexing wrong");

    // Shared experts: one per die, indexed by layer.
    TEST_ASSERT(r.die_for_shared_expert(0) == 0 && r.die_for_shared_expert(1) == 1,
                "shared experts should split one per die");
    TEST_ASSERT(r.shared_blob_offset(0, 0, kBytes) == 0, "shared offset base wrong");
    TEST_ASSERT(r.shared_blob_offset(1, 0, kBytes) == kBytes, "shared per-layer stride wrong");
    TEST_ASSERT(r.shared_blob_offset(0, 1, kBytes) == 0,
                "die 1's shared expert starts its own blob at 0");

    // A real routing decision must split across both dies and account for all 6.
    bool any_cross = false;
    for (int trial = 0; trial < 50; ++trial) {
        auto x = random_hidden(cfg.hidden_size, static_cast<unsigned>(trial) + 99);
        MoERouterResult got;
        TEST_ASSERT(r.route(x.data(), got), "route failed");
        TEST_ASSERT(got.die_experts[0].size() + got.die_experts[1].size() == 6,
                    "die grouping lost experts");
        for (uint32_t e : got.die_experts[0]) TEST_ASSERT(e < 32, "die 0 got a high expert");
        for (uint32_t e : got.die_experts[1]) TEST_ASSERT(e >= 32, "die 1 got a low expert");
        if (!got.die_experts[0].empty() && !got.die_experts[1].empty()) any_cross = true;

        // die_slot must be the index within that die's own range.
        for (const auto &re : got.routed) {
            TEST_ASSERT(re.die_slot < 32, "die_slot out of range");
            TEST_ASSERT(re.index == static_cast<uint32_t>(re.die_id) * 32 + re.die_slot,
                        "die_slot does not reconstruct the global index");
        }
    }
    TEST_ASSERT(any_cross, "no trial exercised both dies (test would be vacuous)");
    std::cout << "    32/die split, dense collision-free offsets, and 50 routings all"
                 "\n    accounted for across both dies." << std::endl;
}

void test_deterministic_ties() {
    std::cout << "\n[*] Testing deterministic tie-breaking..." << std::endl;

    // Identical gate rows produce identical scores, so ties are guaranteed.
    const uint32_t n_exp = 8, dim = 16;
    std::vector<float> gate(n_exp * dim, 0.0f);
    for (uint32_t e = 0; e < n_exp; ++e)
        for (uint32_t i = 0; i < dim; ++i)
            gate[static_cast<size_t>(e) * dim + i] = static_cast<float>(i + 1);
    std::vector<float> x(dim, 1.0f);

    MoERouterConfig cfg = v2lite_config();
    cfg.hidden_size = dim;
    cfg.num_experts = n_exp;
    cfg.top_k = 3;
    MoERouter r;
    std::string err;
    TEST_ASSERT(r.init(cfg, gate.data(), nullptr, err), "init failed: " + err);

    MoERouterResult first, second;
    TEST_ASSERT(r.route(x.data(), first), "route failed");
    TEST_ASSERT(r.route(x.data(), second), "route failed");
    for (size_t i = 0; i < first.routed.size(); ++i) {
        TEST_ASSERT(first.routed[i].index == second.routed[i].index,
                    "routing is not deterministic across calls");
    }
    TEST_ASSERT(first.routed[0].index == 0 && first.routed[1].index == 1 &&
                first.routed[2].index == 2,
            "ties must break toward the lower expert index");
    std::cout << "    All-equal scores resolve to experts 0,1,2 repeatably." << std::endl;
}

void test_config_validation() {
    std::cout << "\n[*] Testing configuration validation..." << std::endl;

    auto gate = random_gate(64, 32, 5);
    std::string err;
    auto base = v2lite_config();
    base.hidden_size = 32;

    struct Bad { const char *what; MoERouterConfig cfg; const float *bias; };
    MoERouterConfig c1 = base; c1.top_k = 65;
    MoERouterConfig c2 = base; c2.num_experts = 0;
    MoERouterConfig c3 = base; c3.top_k = 0;
    MoERouterConfig c4 = base; c4.hidden_size = 0;
    MoERouterConfig c5 = base; c5.num_dies = 0;
    MoERouterConfig c6 = base; c6.num_experts = 7;              // not divisible by 2 dies
    MoERouterConfig c7 = base; c7.n_group = 5;                  // 64 % 5 != 0
    MoERouterConfig c8 = base; c8.topk_group = 2; c8.n_group = 1;
    MoERouterConfig c9 = base; c9.has_gate_bias = true;         // bias pointer null
    MoERouterConfig c10 = base; c10.topk_method = MoETopKMethod::GroupLimitedGreedy;
    c10.n_group = 32; c10.topk_group = 1;                        // 1 group of 2 cannot supply 6

    const Bad bad[] = {
        {"top_k > num_experts", c1, nullptr}, {"zero experts", c2, nullptr},
        {"zero top_k", c3, nullptr},          {"zero hidden_size", c4, nullptr},
        {"zero dies", c5, nullptr},           {"experts not divisible by dies", c6, nullptr},
        {"experts not divisible by groups", c7, nullptr},
        {"topk_group > n_group", c8, nullptr},
        {"bias flag without bias pointer", c9, nullptr},
        {"groups cannot supply top_k", c10, nullptr},
    };

    for (const auto &b : bad) {
        MoERouter r;
        err.clear();
        TEST_ASSERT(!r.init(b.cfg, gate.data(), b.bias, err),
                    std::string("accepted an invalid config: ") + b.what);
        TEST_ASSERT(!err.empty(), std::string("no message for ") + b.what);
        TEST_ASSERT(!r.is_ready(), std::string("ready after rejected init: ") + b.what);
    }

    MoERouter rn;
    TEST_ASSERT(!rn.init(base, nullptr, nullptr, err), "null gate accepted");

    // A rejected router must not crash when asked to route.
    MoERouterResult out;
    TEST_ASSERT(!rn.route(gate.data(), out), "uninitialised router should refuse to route");

    std::cout << "    11 invalid configurations rejected with specific messages." << std::endl;
}

void test_routing_latency() {
    std::cout << "\n[*] Measuring routing latency (target < 60 us/layer)..." << std::endl;

    auto cfg = v2lite_config();
    auto gate = random_gate(cfg.num_experts, cfg.hidden_size, 999);
    MoERouter r;
    std::string err;
    TEST_ASSERT(r.init(cfg, gate.data(), nullptr, err), "init failed: " + err);

    // Warm up, then average over many tokens.
    auto x = random_hidden(cfg.hidden_size, 1000);
    MoERouterResult got;
    for (int i = 0; i < 100; ++i) r.route(x.data(), got);

    const int kTrials = 20000;
    double total = 0.0, worst = 0.0;
    for (int i = 0; i < kTrials; ++i) {
        r.route(x.data(), got);
        total += got.elapsed_us;
        if (got.elapsed_us > worst) worst = got.elapsed_us;
    }
    const double avg = total / kTrials;
    std::cout << "    64x2048 gate, top-6: avg " << avg << " us, worst " << worst << " us"
              << std::endl;
    std::cout << "    26 MoE layers/token => " << (avg * 26 / 1000.0) << " ms/token ("
              << (avg * 26 / 1000.0 / 160.0 * 100.0) << "% of a 6.25 tok/s budget)" << std::endl;
    TEST_ASSERT(avg < 60.0, "average routing latency exceeds the 60 us/layer target");
}

} // namespace

int main() {
    std::cout << "========================================================\n"
              << "  ESWIN MoE Router Test Suite (DeepSeek-V2-Lite P2)     \n"
              << "========================================================\n\n";

    test_v2lite_against_reference();
    test_unnormalised_weights();
    test_sigmoid_and_scaling();
    test_gate_bias();
    test_group_limited_greedy();
    test_die_grouping_and_offsets();
    test_deterministic_ties();
    test_config_validation();
    test_routing_latency();

    std::cout << "\n========================================================\n"
              << "  ALL MoE ROUTER TESTS PASSED!                          \n"
              << "========================================================\n";
    return 0;
}
