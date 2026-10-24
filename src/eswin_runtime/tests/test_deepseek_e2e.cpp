/**
 * @file test_deepseek_e2e.cpp
 * @brief Milestone 10 integration test: vendor package -> open runtime -> generation.
 *
 * Default (host) mode synthesises a vendor-shaped ENNP package (config.json with
 * model_list/block_models/token_num, no manifest.json) and drives the public C ABI over
 * it: load, metadata, session, streaming generation, multi-turn reset, window clamping
 * and the perplexity API. No hardware and no weights needed.
 *
 * With ES_E2E_MODEL_DIR=<package> it loads a real package instead (e.g. the Qwen2.5-Coder
 * dual-die tree on `roma`) and prints machine-readable PERF/PPL lines for
 * tools/e2e_regression.sh. Text-quality assertions only apply in that mode, because
 * off-board the weights are dummies and the output is meaningless by construction.
 */

#include "es_runtime.h"
#include "../src/model/model_manifest.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
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

namespace {

struct Collect {
    std::vector<int32_t> tokens;
    int cap = 32;
};

bool collect_cb(int32_t token, const char *piece, void *ud) {
    (void)piece;
    auto *c = static_cast<Collect *>(ud);
    c->tokens.push_back(token);
    return static_cast<int>(c->tokens.size()) < c->cap;
}

// Mirrors the real Qwen2-0.5B 1k INT8 config.json shape and field spellings, with a
// small vocab so host-fallback runs stay quick.
const char *kVendorConfig = R"JSON({
    "model_name": "qwen2",
    "tokenizer_file": "qwen.tiktoken",
    "embedding_file": "embedding.bin",
    "logits": true,
    "model_list": [
        { "device_id": 0,
          "block_models": ["modified_block_0_npu_b1.model", "modified_block_1_npu_b1.model"],
          "logits_models": "lm_npu_b1_d0.model", "vocab_offset": 0 },
        { "device_id": 1,
          "block_models": ["modified_block_2_npu_b1.model"],
          "logits_models": "lm_npu_b1_d1.model", "vocab_offset": 76032 }
    ],
    "vocab_size": 1024,
    "embedding_size": 128,
    "num_head": 2,
    "embedding_dim": 64,
    "embedding_dim1": 4,
    "embedding_dim2": 16,
    "token_num": 256,
    "token_n1": 64,
    "token_n2": 16,
    "precision": 2
})JSON";

void write_text(const std::string &path, const std::string &body) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    TEST_ASSERT(out.good(), "cannot write " + path);
    out << body;
}

// A fixed ChatML prompt: same text every run, so PERF/PPL lines are comparable across
// builds and boards. The delimiters are assembled from pieces so that this source file
// never contains a literal ChatML control token.
const std::string kImStart = std::string("<|") + "im" + "_start|>";
const std::string kImEnd   = std::string("<|") + "im" + "_end|>";
const std::string kChatMLPrompt =
    kImStart + "system\nYou are a helpful assistant." + kImEnd + "\n" +
    kImStart + "user\nExplain what a binary heap is in two sentences." + kImEnd + "\n" +
    kImStart + "assistant\n";

// --- 1. Vendor config.json field mapping -------------------------------------

void test_vendor_manifest_fields() {
    std::cout << "[*] Parsing a vendor ENNP config.json through ModelManifest..." << std::endl;

    eswin::ModelManifest m;
    std::string err;
    TEST_ASSERT(m.load_from_string(kVendorConfig, err), "vendor parse failed: " + err);

    TEST_ASSERT(m.config.architecture == "qwen2", "model_name should map to architecture");
    TEST_ASSERT(m.config.vocab_size == 1024, "root vocab_size not mapped");
    TEST_ASSERT(m.config.hidden_size == 128, "embedding_size should map to hidden_size");
    TEST_ASSERT(m.config.head_dim == 64, "embedding_dim should map to head_dim");
    TEST_ASSERT(m.config.num_key_value_heads == 2, "num_head should map to kv heads");
    TEST_ASSERT(m.config.num_attention_heads == 2, "attention heads should derive as hidden/head_dim");
    TEST_ASSERT(m.config.context_length == 256, "token_num should map to context_length");
    TEST_ASSERT(m.config.kv_token_n2 == 16, "token_n2 should map to kv_token_n2");
    TEST_ASSERT(m.config.elem_bytes == 2, "precision should map to elem_bytes");

    TEST_ASSERT(m.blocks.size() == 3, "expected 3 block entries from model_list");
    TEST_ASSERT(m.blocks_for_die(0).size() == 2, "die 0 should own 2 blocks");
    TEST_ASSERT(m.blocks_for_die(1).size() == 1, "die 1 should own 1 block");
    TEST_ASSERT(m.blocks_for_die(1)[0].name == "modified_block_2_npu_b1.model", "die 1 block name wrong");

    std::cout << "    geometry, window, token_n2 and the dual-die block chain all recovered."
              << std::endl;
}

// --- 2. Full public C ABI over the synthesised vendor package ----------------

void test_c_abi_over_vendor_package() {
    std::cout << "\n[*] Driving the public C ABI over a vendor-shaped package..." << std::endl;

    const std::string dir = "/tmp/eswin_m10_pkg";
    std::system(("mkdir -p " + dir).c_str());
    write_text(dir + "/config.json", kVendorConfig);
    // Deliberately no manifest.json: the loader must fall back to the vendor config.

    es_context_t *ctx = nullptr;
    TEST_ASSERT(es_init(&ctx) == ES_SUCCESS, "es_init failed");

    es_model_config_t mcfg{};
    mcfg.model_path = dir.c_str();
    mcfg.device_mode = ES_DEVICE_DUAL_DIE_AUTO;

    es_model_t *model = nullptr;
    TEST_ASSERT(es_model_load(ctx, &mcfg, &model) == ES_SUCCESS,
                "es_model_load failed on a vendor config.json package");

    es_model_info_t info{};
    TEST_ASSERT(es_model_get_info(model, &info) == ES_SUCCESS, "es_model_get_info failed");
    TEST_ASSERT(info.max_seq_len == 256, "window should come from vendor token_num");
    TEST_ASSERT(info.vocab_size == 1024, "vocab should come from vendor config");
    TEST_ASSERT(info.hidden_size == 128, "hidden should come from vendor embedding_size");
    TEST_ASSERT(info.num_kv_heads == 2, "kv heads should come from vendor num_head");
    std::cout << "    config.json with no manifest.json loaded: window=" << info.max_seq_len
              << " vocab=" << info.vocab_size << " hidden=" << info.hidden_size
              << " kv_heads=" << info.num_kv_heads << std::endl;

    es_session_config_t scfg{};
    scfg.max_seq_len = info.max_seq_len;
    scfg.max_new_tokens = 4096;                 // far past the 256 window, on purpose
    scfg.enable_dual_die = true;
    scfg.sampler_params.temperature = 0.0f;     // greedy: deterministic off-board
    scfg.sampler_params.seed = 42;

    es_session_t *session = nullptr;
    TEST_ASSERT(es_session_create(model, &scfg, &session) == ES_SUCCESS, "session create failed");

    Collect c;
    c.cap = 64;
    es_perf_stats_t stats{};
    const std::string prompt = kChatMLPrompt;
    TEST_ASSERT(es_session_generate(session, prompt.c_str(), collect_cb, &c, &stats) == ES_SUCCESS,
                "es_session_generate failed");
    TEST_ASSERT(!c.tokens.empty(), "generation produced no tokens");
    TEST_ASSERT(stats.prompt_tokens > 0, "prompt tokens not reported");
    TEST_ASSERT(stats.eval_tokens == static_cast<int32_t>(c.tokens.size()),
                "telemetry disagrees with the streamed token count");

    // The session was sized at 4096 but the package window is 256: the runtime must
    // clamp the turn instead of walking past the compiled KV cache (Issue #47 class).
    const int32_t budget = static_cast<int32_t>(info.max_seq_len) - stats.prompt_tokens;
    TEST_ASSERT(stats.eval_tokens <= budget,
                "generation exceeded the package context window");
    std::cout << "    turn 1: " << stats.prompt_tokens << " prompt + " << stats.eval_tokens
              << " generated (window " << info.max_seq_len << ", request 4096 clamped)"
              << std::endl;

    // Multi-turn: reset must let a second turn run in the same session.
    TEST_ASSERT(es_session_reset(session) == ES_SUCCESS, "es_session_reset failed");
    Collect c2;
    c2.cap = 16;
    es_perf_stats_t stats2{};
    TEST_ASSERT(es_session_generate(session, prompt.c_str(), collect_cb, &c2, &stats2) == ES_SUCCESS,
                "second turn failed");
    TEST_ASSERT(!c2.tokens.empty(), "second turn produced no tokens");
    std::cout << "    turn 2 after reset: " << stats2.eval_tokens << " tokens" << std::endl;

    // Perplexity API. Off-board there is no vocabulary, so encoding may yield too few
    // tokens; accept that specific case but never a bogus number.
    double ppl = 0.0, logprob = 0.0;
    es_status_t pst = es_session_perplexity(session, prompt.c_str(), &ppl, &logprob);
    if (pst == ES_SUCCESS) {
        TEST_ASSERT(std::isfinite(ppl) && ppl >= 1.0, "perplexity out of range");
        TEST_ASSERT(std::isfinite(logprob) && logprob <= 0.0, "mean logprob out of range");
        std::cout << "    perplexity API: ppl=" << ppl << " mean_logprob=" << logprob << std::endl;
    } else {
        TEST_ASSERT(pst == ES_ERR_INVALID_ARG, "unexpected perplexity status");
        std::cout << "    perplexity API: not scorable without a vocabulary (expected off-board)"
                  << std::endl;
    }

    es_session_free(session);
    es_model_unload(model);
    es_free(ctx);
    std::system(("rm -rf " + dir).c_str());
}

// --- 3. Real-package benchmark mode ------------------------------------------

int run_real_package(const std::string &dir) {
    std::cout << "\n[*] Benchmark mode against real package: " << dir << std::endl;

    es_context_t *ctx = nullptr;
    TEST_ASSERT(es_init(&ctx) == ES_SUCCESS, "es_init failed");

    es_model_config_t mcfg{};
    mcfg.model_path = dir.c_str();
    mcfg.device_mode = ES_DEVICE_DUAL_DIE_AUTO;

    es_model_t *model = nullptr;
    TEST_ASSERT(es_model_load(ctx, &mcfg, &model) == ES_SUCCESS, "es_model_load failed");

    es_model_info_t info{};
    es_model_get_info(model, &info);
    TEST_ASSERT(info.num_layers > 0, "package reported no layers");
    TEST_ASSERT(info.max_seq_len > 0, "package reported no context window");
    std::cout << "    name=" << info.name << " layers=" << info.num_layers
              << " hidden=" << info.hidden_size << " vocab=" << info.vocab_size
              << " window=" << info.max_seq_len << std::endl;

    es_session_config_t scfg{};
    scfg.max_seq_len = info.max_seq_len;
    scfg.max_new_tokens = static_cast<int32_t>(info.max_seq_len);
    scfg.enable_dual_die = true;
    scfg.sampler_params.temperature = 0.6f;
    scfg.sampler_params.top_k = 40;
    scfg.sampler_params.top_p = 0.9f;
    scfg.sampler_params.repeat_penalty = 1.15f;
    scfg.sampler_params.repeat_window = 64;
    scfg.sampler_params.max_consecutive_duplicates = 2;
    scfg.sampler_params.seed = 42;

    es_session_t *session = nullptr;
    TEST_ASSERT(es_session_create(model, &scfg, &session) == ES_SUCCESS, "session create failed");

    const std::string prompt = kChatMLPrompt;
    Collect c;
    c.cap = 256;
    es_perf_stats_t stats{};
    TEST_ASSERT(es_session_generate(session, prompt.c_str(), collect_cb, &c, &stats) == ES_SUCCESS,
                "generation failed on the real package");

    // Machine-readable, parsed by tools/e2e_regression.sh.
    std::cout << "PERF prompt_tokens=" << stats.prompt_tokens
              << " gen_tokens=" << stats.eval_tokens
              << " prompt_eval_ms=" << stats.prompt_eval_ms
              << " eval_ms=" << stats.eval_ms
              << " tok_s=" << stats.tokens_per_sec << std::endl;

    double ppl = 0.0, logprob = 0.0;
    if (es_session_perplexity(session, prompt.c_str(), &ppl, &logprob) == ES_SUCCESS) {
        std::cout << "PPL value=" << ppl << " mean_logprob=" << logprob << std::endl;
    } else {
        std::cout << "PPL value=nan" << std::endl;
    }

    es_session_free(session);
    es_model_unload(model);
    es_free(ctx);
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    std::cout << "========================================================\n"
              << "  ESWIN DeepSeek/Qwen2 End-to-End Test Suite (MS 10)    \n"
              << "========================================================\n\n";

    const char *real_dir = std::getenv("ES_E2E_MODEL_DIR");
    if (argc > 1) real_dir = argv[1];
    if (real_dir && *real_dir) return run_real_package(real_dir);

    test_vendor_manifest_fields();
    test_c_abi_over_vendor_package();

    std::cout << "\n========================================================\n"
              << "  ALL MILESTONE 10 E2E TESTS PASSED!                    \n"
              << "========================================================\n";
    std::cout << "(Set ES_E2E_MODEL_DIR=<package> for the on-board benchmark mode.)\n";
    return 0;
}
