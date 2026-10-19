/**
 * @file eswin_chat.cpp
 * @brief Interactive Streaming Chat CLI for Open ESWIN Runtime.
 *
 * Utilizes the high-level C11 API (es_model_*, es_session_*) to provide an
 * interactive chat REPL with streaming terminal output, Markdown colors,
 * live tokens/sec telemetry, and Ctrl+C generation cancellation.
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

// ANSI Colors
#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_RED     "\033[31m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_BLUE    "\033[34m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_GRAY    "\033[90m"

static std::atomic<bool> g_interrupt_flag(false);
static std::atomic<bool> g_in_generation(false);

static void sigint_handler(int sig) {
    (void)sig;
    if (g_in_generation.load()) {
        g_interrupt_flag.store(true);
        std::cout << ANSI_YELLOW << "\n[Generation interrupted by user]" << ANSI_RESET << std::endl;
    } else {
        std::cout << "\nExiting..." << std::endl;
        std::exit(0);
    }
}

static bool token_stream_callback(int32_t token, const char *piece, void *user_data) {
    (void)token;
    (void)user_data;

    if (g_interrupt_flag.load()) {
        return false; // Request generation halt
    }

    if (piece) {
        std::cout << piece << std::flush;
    }
    return true;
}

static void print_banner(const es_model_info_t &info, const es_model_config_t &cfg) {
    std::cout << ANSI_CYAN << ANSI_BOLD;
    std::cout << "===================================================================\n";
    std::cout << "           ESWIN NPU Interactive Chat Engine (v" << es_version() << ")\n";
    std::cout << "===================================================================\n";
    std::cout << ANSI_RESET;
    std::cout << ANSI_GRAY << " Model:         " << ANSI_RESET << ANSI_BOLD << info.name << ANSI_RESET << "\n";
    std::cout << ANSI_GRAY << " Architecture:  " << ANSI_RESET << info.architecture << "\n";
    std::cout << ANSI_GRAY << " Vocabulary:    " << ANSI_RESET << info.vocab_size << " tokens\n";
    std::cout << ANSI_GRAY << " Dimensions:    " << ANSI_RESET << "hidden=" << info.hidden_size
              << ", layers=" << info.num_layers
              << ", heads=" << info.num_heads << " (kv=" << info.num_kv_heads << ")\n";
    std::cout << ANSI_GRAY << " Acceleration:  " << ANSI_RESET
              << (cfg.device_mode == ES_DEVICE_DUAL_DIE_AUTO ? ANSI_GREEN "Dual-Die Auto (Die 0 + Die 1)" : ANSI_YELLOW "Single Die")
              << ANSI_RESET << "\n";
    std::cout << ANSI_GRAY << " Commands:      " << ANSI_RESET << "/reset, /info, /stats, /help, /exit\n";
    std::cout << ANSI_CYAN << "-------------------------------------------------------------------\n" << ANSI_RESET;
}

static void print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -m, --model <path>         Path to model directory or manifest.json (required)\n"
              << "  -t, --tokenizer <path>     Path to tokenizer.json / vocab (optional)\n"
              << "  --temp <float>             Sampling temperature (default: 0.7, 0.0 = greedy)\n"
              << "  --top-k <int>              Top-K sampling limit (default: 40)\n"
              << "  --top-p <float>            Top-P nucleus threshold (default: 0.9)\n"
              << "  --repeat-penalty <float>   Repetition penalty (default: 1.15)\n"
              << "  --repeat-window <int>      Repetition window in tokens (default: 64)\n"
              << "  --max-tokens <int>         Max tokens per turn (default: 512)\n"
              << "  --single-die               Force single die execution\n"
              << "  -h, --help                 Display this help menu\n";
}

int main(int argc, char *argv[]) {
    std::string model_path;
    std::string tokenizer_path;
    float temp = 0.7f;
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
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        std::cerr << ANSI_RED << "Error: --model parameter is required." << ANSI_RESET << "\n\n";
        print_usage(argv[0]);
        return 1;
    }

    std::signal(SIGINT, sigint_handler);

    // 1. Initialize Runtime
    es_context_t *ctx = nullptr;
    es_status_t status = es_init(&ctx);
    if (status != ES_SUCCESS) {
        std::cerr << ANSI_RED << "Failed to initialize ESWIN runtime: " << es_status_to_string(status) << ANSI_RESET << std::endl;
        return 1;
    }

    // 2. Load Model
    es_model_config_t mcfg{};
    mcfg.model_path = model_path.c_str();
    mcfg.tokenizer_path = tokenizer_path.empty() ? nullptr : tokenizer_path.c_str();
    mcfg.device_mode = dev_mode;

    es_model_t *model = nullptr;
    status = es_model_load(ctx, &mcfg, &model);
    if (status != ES_SUCCESS) {
        std::cerr << ANSI_RED << "Failed to load model from '" << model_path << "': "
                  << es_status_to_string(status) << ANSI_RESET << std::endl;
        es_free(ctx);
        return 1;
    }

    es_model_info_t info{};
    es_model_get_info(model, &info);

    // 3. Create Session
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
    scfg.sampler_params.max_consecutive_duplicates = 2;

    es_session_t *session = nullptr;
    status = es_session_create(model, &scfg, &session);
    if (status != ES_SUCCESS) {
        std::cerr << ANSI_RED << "Failed to create session: " << es_status_to_string(status) << ANSI_RESET << std::endl;
        es_model_unload(model);
        es_free(ctx);
        return 1;
    }

    print_banner(info, mcfg);

    // 4. Interactive REPL Loop
    es_perf_stats_t last_stats{};
    std::string line;

    while (true) {
        std::cout << ANSI_GREEN << ANSI_BOLD << "\nUser > " << ANSI_RESET;
        if (!std::getline(std::cin, line)) {
            break; // EOF
        }

        // Trim leading / trailing whitespace
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.erase(0, 1);
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) line.pop_back();

        if (line.empty()) continue;

        // Command handling
        if (line == "/exit" || line == "/quit") {
            break;
        } else if (line == "/reset" || line == "/clear") {
            es_session_reset(session);
            std::cout << ANSI_YELLOW << "[Session context reset - KV cache cleared]" << ANSI_RESET << std::endl;
            continue;
        } else if (line == "/info") {
            print_banner(info, mcfg);
            continue;
        } else if (line == "/stats") {
            std::cout << ANSI_GRAY << "Last Turn Telemetry:\n"
                      << "  Prompt tokens:     " << last_stats.prompt_tokens << " (" << std::fixed << std::setprecision(1) << last_stats.prompt_eval_ms << " ms)\n"
                      << "  Generated tokens:  " << last_stats.eval_tokens << " (" << last_stats.eval_ms << " ms)\n"
                      << "  Decode speed:      " << last_stats.tokens_per_sec << " tokens/sec\n"
                      << ANSI_RESET;
            continue;
        } else if (line == "/help") {
            std::cout << ANSI_GRAY << "Commands:\n"
                      << "  /reset   - Clear conversation history and reset KV cache\n"
                      << "  /stats   - Show inference speed and timing stats\n"
                      << "  /info    - Display model structure and hardware configuration\n"
                      << "  /help    - Display this help message\n"
                      << "  /exit    - Exit chat REPL\n"
                      << ANSI_RESET;
            continue;
        }

        // Assistant generation
        std::cout << ANSI_CYAN << ANSI_BOLD << "Assistant > " << ANSI_RESET;

        g_interrupt_flag.store(false);
        g_in_generation.store(true);

        status = es_session_generate(session, line.c_str(), token_stream_callback, nullptr, &last_stats);

        g_in_generation.store(false);
        std::cout << std::endl;

        if (status == ES_SUCCESS) {
            std::cout << ANSI_GRAY
                      << "[" << last_stats.prompt_tokens << " prompt tokens in "
                      << std::fixed << std::setprecision(1) << last_stats.prompt_eval_ms << "ms | "
                      << last_stats.eval_tokens << " tokens in " << last_stats.eval_ms << "ms | "
                      << std::setprecision(2) << last_stats.tokens_per_sec << " t/s]"
                      << ANSI_RESET << std::endl;
        } else {
            std::cout << ANSI_RED << "[Generation returned status: " << es_status_to_string(status) << "]" << ANSI_RESET << std::endl;
        }
    }

    std::cout << "\nClosing session and releasing resources..." << std::endl;
    es_session_free(session);
    es_model_unload(model);
    es_free(ctx);

    return 0;
}
