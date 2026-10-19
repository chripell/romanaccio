#include "../src/llm/kv_cache.hpp"
#include "../src/llm/transformer_session.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cstdlib>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\033[1;31m[FAIL]\033[0m " << msg << " (" << #cond << ") at " \
                      << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

static void test_kv_cache() {
    std::cout << "[*] Testing KvCache allocation, indexing, and reset..." << std::endl;

    eswin::KvCacheConfig cfg;
    cfg.num_layers = 4;
    cfg.max_seq_len = 128;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 32;

    eswin::KvCache cache(cfg);
    TEST_ASSERT(cache.kv_dim() == 64, "kv_dim mismatch (expected 64)");
    TEST_ASSERT(cache.total_elements_per_cache() == 4 * 128 * 64, "total_elements mismatch");

    // Write slice
    std::vector<float> k_data(64, 1.234f);
    std::vector<float> v_data(64, 5.678f);
    cache.put(2, 10, k_data.data(), v_data.data());

    float *k_ptr = cache.get_k_layer_pos(2, 10);
    float *v_ptr = cache.get_v_layer_pos(2, 10);
    TEST_ASSERT(k_ptr != nullptr, "k_ptr is null");
    TEST_ASSERT(v_ptr != nullptr, "v_ptr is null");
    TEST_ASSERT(std::abs(k_ptr[0] - 1.234f) < 1e-4f, "K cache readback failed");
    TEST_ASSERT(std::abs(v_ptr[0] - 5.678f) < 1e-4f, "V cache readback failed");

    // Reset
    cache.reset();
    TEST_ASSERT(std::abs(k_ptr[0]) < 1e-6f, "K cache reset failed");
    TEST_ASSERT(std::abs(v_ptr[0]) < 1e-6f, "V cache reset failed");

    std::cout << "\033[1;32m[PASS]\033[0m KvCache verified successfully." << std::endl;
}

static void test_transformer_session_lifecycle() {
    std::cout << "[*] Testing TransformerSession generation and telemetry..." << std::endl;

    eswin::ModelManifest manifest;
    manifest.name = "SmolLM-Test";
    manifest.config.architecture = "smollm";
    manifest.config.vocab_size = 256;
    manifest.config.hidden_size = 64;
    manifest.config.intermediate_size = 128;
    manifest.config.num_hidden_layers = 2;
    manifest.config.num_attention_heads = 2;
    manifest.config.num_key_value_heads = 2;
    manifest.config.bos_token_id = 1;
    manifest.config.eos_token_id = 2;

    eswin::TransformerSession session;
    std::string err;
    TEST_ASSERT(session.init(manifest, "", "", err), "Session init failed: " + err);

    // Setup dummy tokenizer vocabulary
    for (int i = 0; i < 256; ++i) {
        // Tokenizer operates on internal vocabulary
    }

    // Configure greedy sampling for determinism
    eswin::SamplerParams sp;
    sp.temperature = 0.0f;
    session.set_sampler_params(sp);

    // Run multi-step generation
    int count = 0;
    std::vector<int32_t> generated_tokens;
    eswin::SessionStats stats;

    session.generate("Hello world", 10, [&](const std::string &piece, int32_t id) {
        (void)piece;
        generated_tokens.push_back(id);
        count++;
        return true;
    }, &stats);

    TEST_ASSERT(count > 0, "No tokens generated");
    TEST_ASSERT(stats.eval_tokens == count, "Stats eval_tokens mismatch");
    std::cout << "    Generated " << count << " tokens in " << stats.eval_ms
              << " ms (" << stats.tokens_per_sec << " tokens/s)" << std::endl;

    std::cout << "\033[1;32m[PASS]\033[0m TransformerSession verified successfully." << std::endl;
}

int main() {
    std::cout << "========================================================\n"
              << "  ESWIN KV-Cache & Transformer Session Test Suite       \n"
              << "========================================================\n\n";

    test_kv_cache();
    test_transformer_session_lifecycle();

    std::cout << "\n========================================================\n"
              << "\033[1;32m  ALL MILESTONE 5 LLM SESSION TESTS PASSED!             \033[0m\n"
              << "========================================================\n";
    return 0;
}
