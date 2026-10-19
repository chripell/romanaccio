/**
 * @file qwen2_v2.cpp
 * @brief Qwen2 & DeepSeek Interactive Chat Engine (Version 2).
 *
 * Built on the Open ESWIN NPU Runtime (libes_runtime.so).
 * Features native anti-looping protections resolving GitHub Issue #47:
 * - Sliding-window repetition penalty (window=64, penalty=1.15)
 * - Consecutive duplicate suppression (max_repeats=2)
 * - Cyclic n-gram loop detector (breaks ABABAB cycles)
 * - Process-isolated dual-die NPU worker execution
 */

#include "es_runtime.h"

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <csignal>
#include <atomic>
#include <cstring>
#include <cstdlib>

#define COLOR_RESET   "\033[0m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_GRAY    "\033[90m"

static std::atomic<bool> g_interrupt_flag(false);
static std::atomic<bool> g_in_generation(false);

static void sigint_handler(int sig) {
    (void)sig;
    if (g_in_generation.load()) {
        g_interrupt_flag.store(true);
        std::cout << COLOR_YELLOW << "\n[Generation paused]" << COLOR_RESET << std::endl;
    } else {
        std::cout << "\nExiting Qwen2-v2..." << std::endl;
        std::exit(0);
    }
}

static bool token_callback(int32_t token, const char *piece, void *user_data) {
    (void)token;
    (void)user_data;

    if (g_interrupt_flag.load()) {
        return false;
    }

    if (piece) {
        std::cout << piece << std::flush;
    }
    return true;
}

static std::string format_chatml(const std::string &user_text) {
    return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
           "<|im_start|>user\n" + user_text + "<|im_end|>\n"
           "<|im_start|>assistant\n";
}

int main(int argc, char *argv[]) {
    std::string model_path = ".";
    std::string tokenizer_path;
    float temp = 0.6f;
    int32_t top_k = 40;
    float top_p = 0.9f;
    float repeat_penalty = 1.15f;
    int32_t repeat_window = 64;
    int32_t max_tokens = 512;
    es_device_mode_t dev_mode = ES_DEVICE_DUAL_DIE_AUTO;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (++i < argc) model_path = argv[i];
        } else if (arg == "-t" || arg == "--tokenizer") {
            if (++i < argc) tokenizer_path = argv[i];
        } else if (arg == "--temp") {
            if (++i < argc) temp = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i < argc) top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i < argc) top_p = std::stof(argv[i]);
        } else if (arg == "--repeat-penalty") {
            if (++i < argc) repeat_penalty = std::stof(argv[i]);
        } else if (arg == "--repeat-window") {
            if (++i < argc) repeat_window = std::stoi(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i < argc) max_tokens = std::stoi(argv[i]);
        } else if (arg == "--single-die") {
            dev_mode = ES_DEVICE_DIE0;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  -m, --model <path>        Model directory or manifest.json (default: .)\n"
                      << "  -t, --tokenizer <path>    Explicit tokenizer.json / vocab path\n"
                      << "  --temp <float>            Sampling temperature (default: 0.6)\n"
                      << "  --top-k <int>             Top-K threshold (default: 40)\n"
                      << "  --top-p <float>           Top-P nucleus threshold (default: 0.9)\n"
                      << "  --repeat-penalty <float>  Repetition penalty for Issue #47 (default: 1.15)\n"
                      << "  --repeat-window <int>     Repetition penalty sliding window (default: 64)\n"
                      << "  --max-tokens <int>        Max tokens per generation turn (default: 512)\n"
                      << "  --single-die              Force single die execution\n"
                      << "  -h, --help                Show this help\n";
            return 0;
        }
    }

    std::signal(SIGINT, sigint_handler);

    // 1. Initialize Open Runtime
    es_context_t *ctx = nullptr;
    es_status_t status = es_init(&ctx);
    if (status != ES_SUCCESS) {
        std::cerr << COLOR_RED << "Error: Failed to initialize runtime: "
                  << es_status_to_string(status) << COLOR_RESET << std::endl;
        return 1;
    }

    // 2. Load Model via Package Manifest
    es_model_config_t mcfg{};
    mcfg.model_path = model_path.c_str();
    mcfg.tokenizer_path = tokenizer_path.empty() ? nullptr : tokenizer_path.c_str();
    mcfg.device_mode = dev_mode;

    es_model_t *model = nullptr;
    status = es_model_load(ctx, &mcfg, &model);
    if (status != ES_SUCCESS) {
        std::cerr << COLOR_RED << "Error: Failed to load model from '" << model_path << "': "
                  << es_status_to_string(status) << COLOR_RESET << std::endl;
        es_free(ctx);
        return 1;
    }

    es_model_info_t info{};
    es_model_get_info(model, &info);

    // 3. Create Inference Session with Issue #47 Anti-Looping Protections
    es_session_config_t scfg{};
    scfg.max_seq_len = 2048;
    scfg.max_new_tokens = max_tokens;
    scfg.enable_dual_die = (dev_mode == ES_DEVICE_DUAL_DIE_AUTO);
    scfg.sampler_params.temperature = temp;
    scfg.sampler_params.top_k = top_k;
    scfg.sampler_params.top_p = top_p;
    scfg.sampler_params.min_p = 0.05f;
    scfg.sampler_params.repeat_penalty = repeat_penalty;
    scfg.sampler_params.repeat_window = repeat_window;
    scfg.sampler_params.max_consecutive_duplicates = 2; // Strict Issue #47 anti-stutter

    es_session_t *session = nullptr;
    status = es_session_create(model, &scfg, &session);
    if (status != ES_SUCCESS) {
        std::cerr << COLOR_RED << "Error: Failed to create session: "
                  << es_status_to_string(status) << COLOR_RESET << std::endl;
        es_model_unload(model);
        es_free(ctx);
        return 1;
    }

    std::cout << COLOR_CYAN << COLOR_BOLD
              << "===================================================================\n"
              << "     Qwen2 / DeepSeek Chat Engine v2 (Powered by Open Runtime)     \n"
              << "===================================================================\n"
              << COLOR_RESET;
    std::cout << COLOR_GRAY << " Model:         " << COLOR_RESET << info.name << "\n"
              << COLOR_GRAY << " Vocab / Hidden:" << COLOR_RESET << " " << info.vocab_size << " / " << info.hidden_size << "\n"
              << COLOR_GRAY << " Architecture:  " << COLOR_RESET << info.num_layers << " layers, " << info.num_heads << " heads\n"
              << COLOR_GRAY << " Protections:   " << COLOR_RESET << COLOR_GREEN "Issue #47 Fixed (rep_pen=" << repeat_penalty
              << ", win=" << repeat_window << ", anti_stutter=2)" << COLOR_RESET << "\n"
              << COLOR_GRAY << " Acceleration:  " << COLOR_RESET
              << (dev_mode == ES_DEVICE_DUAL_DIE_AUTO ? COLOR_GREEN "Dual-Die Auto (Die 0 + Die 1)" : COLOR_YELLOW "Single Die")
              << COLOR_RESET << "\n"
              << COLOR_CYAN << "-------------------------------------------------------------------\n" << COLOR_RESET;

    // 4. REPL
    es_perf_stats_t last_stats{};
    std::string line;

    while (true) {
        std::cout << COLOR_GREEN << COLOR_BOLD << "\nUser > " << COLOR_RESET;
        if (!std::getline(std::cin, line)) break;

        while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.erase(0, 1);
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) line.pop_back();

        if (line.empty()) continue;
        if (line == "/exit" || line == "/quit") break;
        if (line == "/reset") {
            es_session_reset(session);
            std::cout << COLOR_YELLOW << "[KV cache reset]" << COLOR_RESET << std::endl;
            continue;
        }

        std::string prompt = format_chatml(line);

        std::cout << COLOR_CYAN << COLOR_BOLD << "Qwen2 > " << COLOR_RESET;

        g_interrupt_flag.store(false);
        g_in_generation.store(true);

        status = es_session_generate(session, prompt.c_str(), token_callback, nullptr, &last_stats);

        g_in_generation.store(false);
        std::cout << std::endl;

        if (status == ES_SUCCESS) {
            std::cout << COLOR_GRAY
                      << "[" << last_stats.prompt_tokens << " prompt tokens (" << std::fixed << std::setprecision(1) << last_stats.prompt_eval_ms << "ms) | "
                      << last_stats.eval_tokens << " gen tokens (" << last_stats.eval_ms << "ms) | "
                      << std::setprecision(2) << last_stats.tokens_per_sec << " t/s]"
                      << COLOR_RESET << std::endl;
        }
    }

    es_session_free(session);
    es_model_unload(model);
    es_free(ctx);

    std::cout << "Done." << std::endl;
    return 0;
}
