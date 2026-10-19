/**
 * @file test_runtime_e2e.cpp
 * @brief End-to-end integration test suite for Open ESWIN Runtime Milestone 6.
 *
 * Verifies public C11 API:
 * - es_version & status string conversions
 * - Runtime initialization and device discovery
 * - Model package loading & metadata inspection
 * - Inference session lifecycle and KV-cache reset
 * - Autoregressive token generation with streaming callbacks & telemetry
 */

#include "es_runtime.h"

#include <iostream>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>

static const char *TEST_MANIFEST_PATH = "/tmp/test_runtime_e2e_manifest.json";

static void create_test_manifest() {
    std::ofstream f(TEST_MANIFEST_PATH);
    f << "{\n"
      << "  \"name\": \"E2E-Test-Model\",\n"
      << "  \"version\": \"1.0.0\",\n"
      << "  \"description\": \"Milestone 6 integration test synthetic model\",\n"
      << "  \"tensor_parallel\": 2,\n"
      << "  \"config\": {\n"
      << "    \"architecture\": \"llama\",\n"
      << "    \"vocab_size\": 1024,\n"
      << "    \"hidden_size\": 128,\n"
      << "    \"intermediate_size\": 256,\n"
      << "    \"num_hidden_layers\": 4,\n"
      << "    \"num_attention_heads\": 4,\n"
      << "    \"num_key_value_heads\": 2,\n"
      << "    \"rope_theta\": 10000.0,\n"
      << "    \"bos_token_id\": 1,\n"
      << "    \"eos_token_id\": 2\n"
      << "  },\n"
      << "  \"files\": []\n"
      << "}\n";
    f.close();
}

struct CallbackState {
    std::vector<int32_t> tokens;
    std::string accumulated_text;
    int max_to_collect = 10;
};

static bool test_token_cb(int32_t token, const char *piece, void *user_data) {
    auto *state = static_cast<CallbackState*>(user_data);
    state->tokens.push_back(token);
    if (piece) {
        state->accumulated_text += piece;
    }
    return (state->tokens.size() < static_cast<size_t>(state->max_to_collect));
}

int main() {
    std::cout << "[TEST] Starting End-to-End Runtime Integration Tests (Milestone 6)..." << std::endl;

    // 1. Version and status strings
    const char *ver = es_version();
    assert(ver != nullptr);
    assert(std::strlen(ver) > 0);
    std::cout << "  - Runtime version: " << ver << " [OK]" << std::endl;

    const char *s_ok = es_status_to_string(ES_SUCCESS);
    assert(s_ok != nullptr && std::strstr(s_ok, "ES_SUCCESS") != nullptr);
    const char *s_inv = es_status_to_string(ES_ERR_INVALID_ARG);
    assert(s_inv != nullptr && std::strstr(s_inv, "ES_ERR_INVALID_ARG") != nullptr);
    std::cout << "  - Status string mappings verified [OK]" << std::endl;

    // 2. Context initialization
    es_context_t *ctx = nullptr;
    es_status_t status = es_init(&ctx);
    assert(status == ES_SUCCESS);
    assert(ctx != nullptr);
    std::cout << "  - Runtime context initialized [OK]" << std::endl;

    uint16_t dev_count = 0;
    status = es_get_device_count(ctx, &dev_count);
    assert(status == ES_SUCCESS);
    assert(dev_count >= 0);
    std::cout << "  - Discovered operational dies: " << dev_count
              << (dev_count > 0 ? " (Hardware NPU)" : " (Host Fallback)") << " [OK]" << std::endl;

    // 3. Create synthetic test manifest and load model
    create_test_manifest();

    es_model_config_t mcfg{};
    mcfg.model_path = TEST_MANIFEST_PATH;
    mcfg.device_mode = ES_DEVICE_DUAL_DIE_AUTO;

    es_model_t *model = nullptr;
    status = es_model_load(ctx, &mcfg, &model);
    assert(status == ES_SUCCESS);
    assert(model != nullptr);
    std::cout << "  - Model package loaded from manifest [OK]" << std::endl;

    // 4. Inspect model metadata
    es_model_info_t info{};
    status = es_model_get_info(model, &info);
    assert(status == ES_SUCCESS);
    assert(std::strcmp(info.name, "E2E-Test-Model") == 0);
    assert(std::strcmp(info.architecture, "llama") == 0);
    assert(info.vocab_size == 1024);
    assert(info.hidden_size == 128);
    assert(info.num_layers == 4);
    assert(info.num_heads == 4);
    assert(info.num_kv_heads == 2);
    std::cout << "  - Model metadata inspected and verified [OK]" << std::endl;

    // 5. Create inference session
    es_session_config_t scfg{};
    scfg.max_seq_len = 512;
    scfg.max_new_tokens = 20;
    scfg.enable_dual_die = true;
    scfg.sampler_params.temperature = 0.5f;
    scfg.sampler_params.top_k = 20;
    scfg.sampler_params.top_p = 0.85f;
    scfg.sampler_params.repeat_penalty = 1.15f;
    scfg.sampler_params.seed = 42;

    es_session_t *session = nullptr;
    status = es_session_create(model, &scfg, &session);
    assert(status == ES_SUCCESS);
    assert(session != nullptr);
    std::cout << "  - Inference session created with KV cache [OK]" << std::endl;

    // 6. Test autoregressive generation with streaming callback
    CallbackState cb_state;
    cb_state.max_to_collect = 15;

    es_perf_stats_t stats{};
    status = es_session_generate(
        session,
        "DeepSeek test input sequence",
        test_token_cb,
        &cb_state,
        &stats
    );
    assert(status == ES_SUCCESS);
    assert(!cb_state.tokens.empty());
    assert(stats.prompt_tokens > 0);
    assert(stats.eval_tokens == static_cast<int32_t>(cb_state.tokens.size()));
    assert(stats.tokens_per_sec >= 0.0);
    std::cout << "  - Autoregressive generation completed ("
              << stats.eval_tokens << " tokens, "
              << stats.tokens_per_sec << " t/s) [OK]" << std::endl;

    // 7. Test session reset and turn 2
    status = es_session_reset(session);
    assert(status == ES_SUCCESS);

    CallbackState cb_state2;
    cb_state2.max_to_collect = 8;
    es_perf_stats_t stats2{};
    status = es_session_generate(
        session,
        "Second turn prompt",
        test_token_cb,
        &cb_state2,
        &stats2
    );
    assert(status == ES_SUCCESS);
    assert(cb_state2.tokens.size() == 8);
    std::cout << "  - Session context reset and multi-turn execution verified [OK]" << std::endl;

    // 8. Resource cleanup
    es_session_free(session);
    es_model_unload(model);
    es_free(ctx);
    std::remove(TEST_MANIFEST_PATH);

    std::cout << "[TEST] All Milestone 6 End-to-End Integration Tests PASSED!\n" << std::endl;
    return 0;
}
