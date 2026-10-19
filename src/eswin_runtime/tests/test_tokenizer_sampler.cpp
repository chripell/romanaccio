#include "../src/nlp/sampler.hpp"
#include "../src/nlp/tokenizer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>
#include <cstdlib>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\033[1;31m[FAIL]\033[0m " << msg << " (" << #cond << ") at " \
                      << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

static void test_sampler_greedy_and_temp() {
    std::cout << "[*] Testing Sampler Greedy and Temperature Scaling..." << std::endl;

    eswin::SamplerParams p;
    p.temperature = 0.0f; // Greedy
    eswin::Sampler sampler(p);

    float logits[5] = {1.0f, 2.5f, 0.5f, 8.2f, 3.1f};
    int32_t token = sampler.sample(logits, 5);
    TEST_ASSERT(token == 3, "Greedy sampling must select token with highest logit (3)");

    std::cout << "    Greedy argmax selected token 3 (logit 8.2) correctly." << std::endl;
}

static void test_sampler_repetition_and_loop_breaker() {
    std::cout << "[*] Testing Sampler Repetition Penalty and Anti-Looping (Issue #47 Fix)..." << std::endl;

    eswin::SamplerParams p;
    p.temperature = 0.0f; // deterministic
    p.repeat_penalty = 2.0f;
    p.repeat_window = 10;
    p.max_consecutive_duplicates = 2;

    eswin::Sampler sampler(p);

    // Initial logits: token 1 has highest logit (10.0), token 2 has 6.0
    float logits[4] = {1.0f, 10.0f, 6.0f, 2.0f};

    // First sample: token 1 should win
    int32_t t1 = sampler.sample(logits, 4);
    TEST_ASSERT(t1 == 1, "First sample should be token 1");
    sampler.accept(t1);

    // After accepting token 1, repetition penalty applies: 10.0 / 2.0 = 5.0
    // Now token 2 (logit 6.0) should win over token 1 (5.0)!
    float logits2[4] = {1.0f, 10.0f, 6.0f, 2.0f};
    int32_t t2 = sampler.sample(logits2, 4);
    TEST_ASSERT(t2 == 2, "Repetition penalty should demote token 1 below token 2");
    std::cout << "    Repetition penalty demoted token 1 correctly." << std::endl;

    // Test consecutive duplicate suppression:
    sampler.reset();
    sampler.accept(1);
    sampler.accept(1); // 2 consecutive occurrences of token 1

    float logits3[4] = {1.0f, 100.0f, 6.0f, 2.0f};
    int32_t t3 = sampler.sample(logits3, 4);
    TEST_ASSERT(t3 != 1, "Consecutive duplicate suppressor must block 3rd consecutive token 1!");
    std::cout << "    Consecutive duplicate suppressor blocked 3rd repeat." << std::endl;

    // Test cyclic loop detector (ABABAB)
    sampler.reset();
    sampler.accept(1);
    sampler.accept(2);
    sampler.accept(1);
    sampler.accept(2);
    sampler.accept(1);
    sampler.accept(2);

    float logits4[4] = {1.0f, 100.0f, 2.0f, 2.0f};
    int32_t t4 = sampler.sample(logits4, 4);
    TEST_ASSERT(t4 != 1, "Cyclic loop detector must break ABABAB cycle!");
    std::cout << "    Cyclic n-gram loop detector broke repeating cycle." << std::endl;

    std::cout << "\033[1;32m[PASS]\033[0m Sampler algorithms verified." << std::endl;
}

static void test_tokenizer_synthetic() {
    std::cout << "[*] Testing Tokenizer JSON Loading, Encoding & Decoding..." << std::endl;

    std::string json_vocab = R"({
        "model": {
            "vocab": {
                "hello": 0,
                "Ġworld": 1,
                "!": 2,
                "Ġthis": 3,
                "Ġis": 4,
                "Ġa": 5,
                "Ġtest": 6
            }
        },
        "added_tokens": [
            {"id": 100, "content": "<|im_start|>"},
            {"id": 101, "content": "<|im_end|>"}
        ]
    })";

    eswin::Tokenizer tok;
    std::string err;
    TEST_ASSERT(tok.load_json(json_vocab, err), "Failed to load JSON vocab: " + err);
    TEST_ASSERT(tok.vocab_size() >= 102, "Vocab size mismatch");

    std::vector<int32_t> tokens = tok.encode("hello world !");
    TEST_ASSERT(!tokens.empty(), "Encoding produced no tokens");
    TEST_ASSERT(tokens.size() == 3, "Expected 3 tokens");
    TEST_ASSERT(tokens[0] == 0, "token[0] != hello (0)");
    TEST_ASSERT(tokens[1] == 1, "token[1] != world (1)");
    TEST_ASSERT(tokens[2] == 2, "token[2] != ! (2)");

    std::string decoded = tok.decode(tokens);
    TEST_ASSERT(decoded == "hello world!", "Decoded string mismatch: " + decoded);
    std::cout << "    Encoded 'hello world !' -> [0, 1, 2] -> decoded '" << decoded << "'" << std::endl;

    // ChatML test
    std::string chatml = eswin::Tokenizer::format_chatml("What is RISC-V?", "You are an assistant.");
    TEST_ASSERT(chatml.find("<|im_start|>system") != std::string::npos, "Missing system prompt in ChatML");
    TEST_ASSERT(chatml.find("What is RISC-V?") != std::string::npos, "Missing user query in ChatML");
    TEST_ASSERT(chatml.find("<|im_start|>assistant") != std::string::npos, "Missing assistant prompt in ChatML");
    std::cout << "    ChatML formatting verified." << std::endl;

    std::cout << "\033[1;32m[PASS]\033[0m Tokenizer implementation verified." << std::endl;
}

int main() {
    std::cout << "========================================================\n"
              << "  ESWIN Embedded Tokenizer & Sampler Test Suite (NLP)   \n"
              << "========================================================\n\n";

    test_sampler_greedy_and_temp();
    test_sampler_repetition_and_loop_breaker();
    test_tokenizer_synthetic();

    std::cout << "\n========================================================\n"
              << "\033[1;32m  ALL MILESTONE 4 TOKENIZER & SAMPLER TESTS PASSED!     \033[0m\n"
              << "========================================================\n";
    return 0;
}
