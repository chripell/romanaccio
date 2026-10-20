/**
 * @file smollm_v2.cpp
 * @brief SmolLM-135M Interactive Chat Engine (Version 2).
 *
 * Built on the Open ESWIN NPU Runtime (libes_runtime.so).
 * Supports dual-die NPU offload with process-isolated NUMA workers,
 * embedded ChatML formatting, and anti-looping sampling.
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
#include <fstream>
#include <sys/stat.h>

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
        std::cout << "\nExiting SmolLM-v2..." << std::endl;
        std::exit(0);
    }
}

static bool token_callback(int32_t token, const char *piece, void *user_data) {
    (void)user_data;

    if (g_interrupt_flag.load()) {
        return false;
    }

    if (piece && piece[0] != '\0') {
        std::cout << piece << std::flush;
    } else {
        std::cout << "[#" << token << "]" << std::flush;
    }
    return true;
}

static bool file_exists_local(const std::string &path) {
    if (path.empty()) return false;
    std::ifstream f(path);
    return f.good();
}

static void displayInitialCard(
    const es_model_info_t &info,
    const std::string &model_dir,
    const std::string &tokenizer_path,
    size_t vocab_count,
    es_device_mode_t dev_mode,
    float temp,
    int32_t top_k,
    float top_p,
    float repeat_penalty,
    int32_t max_tokens)
{
    std::cout << "\n\033[1;36m================================================================================\033[0m\n"
              << "\033[1;36m     SmolLM-135M Chat Engine v2 (Powered by Open ESWIN Runtime)                 \033[0m\n"
              << "\033[1;36m================================================================================\033[0m\n";

    std::vector<std::string> warnings;

    // 1. Configuration & Model Info
    std::cout << "\033[1m[Configuration & Model]\033[0m\n";
    std::cout << "  • Model Name        : " << info.name << "\n";
    std::cout << "  • Model Directory   : " << model_dir;
    struct stat dirSt;
    if (stat(model_dir.c_str(), &dirSt) == 0 && S_ISDIR(dirSt.st_mode)) {
        std::cout << " \033[32m[OK]\033[0m\n";
    } else {
        std::cout << " \033[31m[DIRECTORY NOT FOUND]\033[0m\n";
        warnings.push_back("Model directory does not exist: " + model_dir);
    }

    std::string manifest_file = model_dir + "/manifest.json";
    if (file_exists_local(manifest_file)) {
        std::cout << "  • Package Manifest  : " << manifest_file << " \033[32m[OK]\033[0m\n";
    } else {
        std::cout << "  • Package Manifest  : " << manifest_file << " \033[33m[MISSING / DEFAULT PARAMETERS]\033[0m\n";
        warnings.push_back("Package manifest (manifest.json) not found in " + model_dir);
    }
    std::cout << "  • Architecture      : " << info.num_layers << " layers, "
              << info.num_heads << " heads, hidden dim=" << info.hidden_size
              << ", vocab=" << info.vocab_size << "\n\n";

    // 2. Tokenizer & Vocabulary
    std::cout << "\033[1m[Tokenizer & Vocabulary]\033[0m\n";
    std::string tok_path = !tokenizer_path.empty() ? tokenizer_path : (model_dir + "/tokenizer.json");
    if (vocab_count > 0 && file_exists_local(tok_path)) {
        std::cout << "  • Tokenizer File    : " << tok_path << " (" << vocab_count << " vocabulary tokens loaded) \033[32m[OK]\033[0m\n";
    } else if (file_exists_local(tok_path)) {
        std::cout << "  • Tokenizer File    : " << tok_path << " \033[32m[OK]\033[0m\n";
    } else {
        std::cout << "  • Tokenizer File    : " << tok_path << " \033[31m[NOT LOADED]\033[0m\n";
        warnings.push_back("Tokenizer vocabulary not loaded! Tokens will be displayed in numeric [#id] form.");
    }
    std::cout << "  • Special Tokens    : <|im_start|>, <|im_end|> \033[32m[OK]\033[0m\n\n";

    // 3. Hardware NPU Acceleration
    std::cout << "\033[1m[Hardware NPU Acceleration]\033[0m\n";
    std::string d0_path = model_dir + "/smollm_die0.model";
    std::string d1_path = model_dir + "/smollm_die1.model";
    bool d0_ok = file_exists_local(d0_path);
    bool d1_ok = file_exists_local(d1_path);

    std::cout << "  • Die 0 LM-Head     : " << d0_path << (d0_ok ? " \033[32m[OK]\033[0m\n" : " \033[31m[NOT FOUND]\033[0m\n");
    std::cout << "  • Die 1 LM-Head     : " << d1_path << (d1_ok ? " \033[32m[OK]\033[0m\n" : " \033[31m[NOT FOUND]\033[0m\n");

    std::cout << "  • Device Target     : "
              << (dev_mode == ES_DEVICE_DUAL_DIE_AUTO ? "Dual-Die Auto (Die 0 + Die 1)" : "Single Die (Die 0)") << "\n";

    if (d0_ok && d1_ok) {
        std::cout << "  • Acceleration Mode : Concurrent Dual-Die Pipeline \033[32m[HARDWARE NPU]\033[0m\n";
    } else {
        std::cout << "  • Acceleration Mode : Host CPU Fallback \033[33m[FALLBACK DUMMY WEIGHTS - NO HARDWARE OFF-LOAD]\033[0m\n";
        warnings.push_back("Hardware NPU models (.model files) missing! Execution is running on host CPU fallback with dummy weights.");
    }
    std::cout << "\n";

    // 4. Inference Hyperparameters & Protections
    std::cout << "\033[1m[Inference & Protections]\033[0m\n";
    std::cout << "  • Sampling Mode     : " << (temp > 0.0f ? "Top-P / Top-K Sampling" : "Greedy (ArgMax)")
              << " (temp=" << std::fixed << std::setprecision(2) << temp
              << ", top_k=" << top_k << ", top_p=" << top_p << ")\n";
    std::cout << "  • Anti-Looping      : Repetition penalty=" << repeat_penalty
              << " (win=64), anti-stutter=2 \033[32m[ACTIVE]\033[0m\n";
    std::cout << "  • Max Response Len  : " << max_tokens << " tokens\n";
    std::cout << "\033[1;36m--------------------------------------------------------------------------------\033[0m\n";

    // Warnings Box
    if (!warnings.empty()) {
        std::cout << "\n\033[1;41m  WARNINGS DETECTED (" << warnings.size() << ")  \033[0m\n";
        for (size_t w = 0; w < warnings.size(); ++w) {
            std::cout << " \033[1;33m[!] Warning " << (w + 1) << ": " << warnings[w] << "\033[0m\n";
        }
        std::cout << "\n\033[33mDue to missing pieces, generation output will be degraded or fallback dummy.\033[0m\n"
                  << "\033[36mTip: Ensure real .model files and tokenizer.json are present or symlinked.\033[0m\n\n";
    } else {
        std::cout << "\033[1;32m[+] All required model files and tokenizer pieces loaded successfully!\033[0m\n\n";
    }
}

static std::string format_chatml(const std::string &user_text) {
    return "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
           "<|im_start|>user\n" + user_text + "<|im_end|>\n"
           "<|im_start|>assistant\n";
}

int main(int argc, char *argv[]) {
    std::string model_dir = ".";
    std::string tokenizer_path;
    float temp = 0.7f;
    int32_t top_k = 40;
    float top_p = 0.9f;
    float repeat_penalty = 1.15f;
    int32_t max_tokens = 512;
    es_device_mode_t dev_mode = ES_DEVICE_DUAL_DIE_AUTO;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (++i < argc) model_dir = argv[i];
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
        } else if (arg == "--max-tokens") {
            if (++i < argc) max_tokens = std::stoi(argv[i]);
        } else if (arg == "--single-die") {
            dev_mode = ES_DEVICE_DIE0;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  -m, --model <dir>         Model directory with manifest.json (default: .)\n"
                      << "  -t, --tokenizer <path>    Explicit tokenizer.json path\n"
                      << "  --temp <float>            Sampling temperature (default: 0.7)\n"
                      << "  --top-k <int>             Top-K threshold (default: 40)\n"
                      << "  --top-p <float>           Top-P nucleus threshold (default: 0.9)\n"
                      << "  --repeat-penalty <float>  Repetition penalty (default: 1.15)\n"
                      << "  --max-tokens <int>        Max tokens per turn (default: 512)\n"
                      << "  --single-die              Run on single die (Die 0)\n"
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
    mcfg.model_path = model_dir.c_str();
    mcfg.tokenizer_path = tokenizer_path.empty() ? nullptr : tokenizer_path.c_str();
    mcfg.device_mode = dev_mode;

    es_model_t *model = nullptr;
    status = es_model_load(ctx, &mcfg, &model);
    if (status != ES_SUCCESS) {
        std::cerr << COLOR_RED << "Error: Failed to load model from '" << model_dir << "': "
                  << es_status_to_string(status) << COLOR_RESET << std::endl;
        es_free(ctx);
        return 1;
    }

    es_model_info_t info{};
    es_model_get_info(model, &info);

    const es_tokenizer_t *tok_handle = es_model_get_tokenizer(model);
    size_t vocab_count = es_tokenizer_vocab_size(tok_handle);

    // 3. Create Inference Session
    es_session_config_t scfg{};
    scfg.max_seq_len = 2048;
    scfg.max_new_tokens = max_tokens;
    scfg.enable_dual_die = (dev_mode == ES_DEVICE_DUAL_DIE_AUTO);
    scfg.sampler_params.temperature = temp;
    scfg.sampler_params.top_k = top_k;
    scfg.sampler_params.top_p = top_p;
    scfg.sampler_params.min_p = 0.05f;
    scfg.sampler_params.repeat_penalty = repeat_penalty;
    scfg.sampler_params.repeat_window = 64;
    scfg.sampler_params.max_consecutive_duplicates = 2;

    es_session_t *session = nullptr;
    status = es_session_create(model, &scfg, &session);
    if (status != ES_SUCCESS) {
        std::cerr << COLOR_RED << "Error: Failed to create session: "
                  << es_status_to_string(status) << COLOR_RESET << std::endl;
        es_model_unload(model);
        es_free(ctx);
        return 1;
    }

    // 4. Display comprehensive model card and loaded components status
    displayInitialCard(
        info, model_dir, tokenizer_path, vocab_count, dev_mode,
        temp, top_k, top_p, repeat_penalty, max_tokens
    );

    std::cout << "Interactive chat session ready. Type your question and press Enter.\n"
              << "Type '/reset' to clear KV cache, or '/exit' to quit.\n";

    // 5. REPL
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

        std::cout << COLOR_CYAN << COLOR_BOLD << "SmolLM > " << COLOR_RESET;

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
