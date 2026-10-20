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
 * - Thinking token streaming (--show-thinking default enabled, --max-thinking-tokens)
 * - Real-time token debug fallback mode (--show-tokens)
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
#include <sstream>

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
        std::cout << COLOR_YELLOW << "\n[Generation paused by user]" << COLOR_RESET << std::endl;
    } else {
        std::cout << "\nExiting Qwen2-v2..." << std::endl;
        std::exit(0);
    }
}

static bool file_exists_local(const std::string &path) {
    if (path.empty()) return false;
    std::ifstream f(path);
    return f.good();
}

static std::string resolve_tokenizer_path(const std::string &model_path, const std::string &explicit_path) {
    if (!explicit_path.empty() && file_exists_local(explicit_path)) return explicit_path;
    if (file_exists_local(model_path + "/qwen.tiktoken")) return model_path + "/qwen.tiktoken";
    if (file_exists_local("./qwen.tiktoken")) return "./qwen.tiktoken";
    if (file_exists_local("../qwen2_fix/deepseek_7b_1k_int8_peer/qwen.tiktoken")) return "../qwen2_fix/deepseek_7b_1k_int8_peer/qwen.tiktoken";
    if (file_exists_local("/opt/eswin/sample-code/npu_sample/qwen_sample/src/qwen2_0_5b_1k_int8/qwen.tiktoken"))
        return "/opt/eswin/sample-code/npu_sample/qwen_sample/src/qwen2_0_5b_1k_int8/qwen.tiktoken";
    if (file_exists_local(model_path + "/tokenizer.json")) return model_path + "/tokenizer.json";
    if (file_exists_local(model_path + "/vocab.json")) return model_path + "/vocab.json";
    return "";
}

struct GenerationContext {
    bool show_thinking = true;
    bool show_tokens = false;
    int  max_thinking_tokens = 128;
    bool in_thinking = false;
    int  thinking_count = 0;
    int  total_count = 0;
    bool thinking_limit_reached = false;
};

static bool token_callback(int32_t token, const char *piece, void *user_data) {
    if (g_interrupt_flag.load()) {
        return false;
    }

    auto *ctx = static_cast<GenerationContext*>(user_data);
    ctx->total_count++;
    std::string str = piece ? piece : "";

    // Detect thinking tags (Qwen2 / DeepSeek-R1: 151648 = <think>, 151649 = </think>)
    bool is_think_start = (token == 151648 || str == "<think>" || str.find("<think>") != std::string::npos);
    bool is_think_end   = (token == 151649 || str == "</think>" || str.find("</think>") != std::string::npos);

    if (is_think_start) {
        ctx->in_thinking = true;
        ctx->thinking_count = 0;
        ctx->thinking_limit_reached = false;
        if (ctx->show_tokens) {
            std::cout << COLOR_GRAY << "[#" << token << ":<think>]" << COLOR_RESET;
        }
        if (ctx->show_thinking) {
            std::cout << COLOR_GRAY << "\n[Thinking: " << COLOR_RESET << std::flush;
        } else {
            std::cout << COLOR_GRAY << "[Thinking...]" << COLOR_RESET << std::flush;
        }
        return true;
    }

    if (is_think_end) {
        if (ctx->in_thinking) {
            ctx->in_thinking = false;
            if (ctx->show_thinking) {
                std::cout << COLOR_GRAY << "]\n\n" << COLOR_RESET << std::flush;
            } else {
                std::cout << "\r              \r" << std::flush;
            }
        }
        if (ctx->show_tokens) {
            std::cout << COLOR_GRAY << "[#" << token << ":</think>]" << COLOR_RESET << std::flush;
        }
        return true;
    }

    if (ctx->in_thinking) {
        ctx->thinking_count++;
        if (ctx->max_thinking_tokens > 0 && ctx->thinking_count >= ctx->max_thinking_tokens && !ctx->thinking_limit_reached) {
            ctx->thinking_limit_reached = true;
            ctx->in_thinking = false;
            if (ctx->show_thinking) {
                std::cout << COLOR_YELLOW << "\n[Thinking limit of " << ctx->max_thinking_tokens
                          << " tokens reached, concluding reasoning]\n\n" << COLOR_RESET << std::flush;
            } else {
                std::cout << COLOR_YELLOW << "\n[Thinking limit reached]\n" << COLOR_RESET << std::flush;
            }
            return true;
        }

        if (ctx->show_thinking) {
            if (ctx->show_tokens) {
                if (!str.empty()) {
                    std::cout << COLOR_GRAY << "[#" << token << ":\"" << str << "\"]" << COLOR_RESET << std::flush;
                } else {
                    std::cout << COLOR_GRAY << "[#" << token << "]" << COLOR_RESET << std::flush;
                }
            } else {
                if (!str.empty()) {
                    std::cout << COLOR_GRAY << str << COLOR_RESET << std::flush;
                } else {
                    std::cout << COLOR_GRAY << "[#" << token << "]" << COLOR_RESET << std::flush;
                }
            }
        }
        return true;
    }

    // Normal generation outside thinking mode
    if (ctx->show_tokens) {
        if (!str.empty()) {
            std::cout << "[#" << token << ":\"" << str << "\"]" << std::flush;
        } else {
            std::cout << "[#" << token << "]" << std::flush;
        }
    } else {
        if (!str.empty()) {
            std::cout << str << std::flush;
        } else {
            // Fallback for unmapped or empty tokens: NEVER leave the user staring at a blank console!
            std::cout << "[#" << token << "]" << std::flush;
        }
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
    int32_t max_tokens = 256;
    bool show_thinking = true;         // Default: show thinking
    int32_t max_thinking_tokens = 128; // Default limit for thinking reasoning
    bool show_tokens = false;          // Debug token mode
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
        } else if (arg == "--show-thinking") {
            show_thinking = true;
        } else if (arg == "--hide-thinking" || arg == "--no-thinking") {
            show_thinking = false;
        } else if (arg == "--max-thinking-tokens") {
            if (++i < argc) max_thinking_tokens = std::stoi(argv[i]);
        } else if (arg == "--show-tokens" || arg == "--debug-tokens") {
            show_tokens = true;
        } else if (arg == "--single-die") {
            dev_mode = ES_DEVICE_DIE0;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  -m, --model <path>             Model directory or manifest.json (default: .)\n"
                      << "  -t, --tokenizer <path>         Explicit tokenizer.json / qwen.tiktoken path\n"
                      << "  --show-thinking                Stream thinking tokens in gray (DEFAULT: on)\n"
                      << "  --hide-thinking                Hide thinking tokens, showing only final response\n"
                      << "  --max-thinking-tokens <int>    Cap thinking tokens before answering (default: 128)\n"
                      << "  --show-tokens                  Debug mode: display [#id:\"piece\"] for all tokens\n"
                      << "  --temp <float>                 Sampling temperature (default: 0.6)\n"
                      << "  --top-k <int>                  Top-K threshold (default: 40)\n"
                      << "  --top-p <float>                Top-P nucleus threshold (default: 0.9)\n"
                      << "  --repeat-penalty <float>       Repetition penalty for Issue #47 (default: 1.15)\n"
                      << "  --repeat-window <int>          Repetition penalty sliding window (default: 64)\n"
                      << "  --max-tokens <int>             Max tokens per generation turn (default: 256)\n"
                      << "  --single-die                   Force single die execution\n"
                      << "  -h, --help                     Show this help\n";
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

    // 2. Resolve Tokenizer
    std::string resolved_tok = resolve_tokenizer_path(model_path, tokenizer_path);

    // 3. Load Model via Package Manifest
    es_model_config_t mcfg{};
    mcfg.model_path = model_path.c_str();
    mcfg.tokenizer_path = resolved_tok.empty() ? nullptr : resolved_tok.c_str();
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

    const es_tokenizer_t *tok_handle = es_model_get_tokenizer(model);
    size_t vocab_count = es_tokenizer_vocab_size(tok_handle);

    // 4. Create Inference Session with Issue #47 Anti-Looping Protections
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
              << COLOR_GRAY << " Architecture:  " << COLOR_RESET << info.num_layers << " layers, " << info.num_heads << " heads\n";

    if (vocab_count > 0) {
        std::cout << COLOR_GRAY << " Tokenizer:     " << COLOR_RESET << COLOR_GREEN << resolved_tok
                  << " (" << vocab_count << " vocabulary tokens loaded)" << COLOR_RESET << "\n";
    } else {
        std::cout << COLOR_GRAY << " Tokenizer:     " << COLOR_RESET << COLOR_RED
                  << "NOT LOADED! (Will display [#id] for each token)" << COLOR_RESET << "\n";
    }

    std::cout << COLOR_GRAY << " Thinking:      " << COLOR_RESET
              << (show_thinking ? COLOR_GREEN "ENABLED (default" : COLOR_YELLOW "HIDDEN")
              << ", max=" << max_thinking_tokens << " tokens)" << COLOR_RESET << "\n"
              << COLOR_GRAY << " Token Debug:   " << COLOR_RESET
              << (show_tokens ? COLOR_GREEN "ENABLED (showing [#id:\"piece\"])" : COLOR_GRAY "OFF (type /tokens on to enable)")
              << COLOR_RESET << "\n"
              << COLOR_GRAY << " Protections:   " << COLOR_RESET << COLOR_GREEN "Issue #47 Fixed (rep_pen=" << repeat_penalty
              << ", win=" << repeat_window << ", anti_stutter=2)" << COLOR_RESET << "\n"
              << COLOR_GRAY << " Acceleration:  " << COLOR_RESET
              << (dev_mode == ES_DEVICE_DUAL_DIE_AUTO ? COLOR_GREEN "Dual-Die Auto (Die 0 + Die 1)" : COLOR_YELLOW "Single Die")
              << COLOR_RESET << "\n"
              << COLOR_CYAN << "-------------------------------------------------------------------\n" << COLOR_RESET;

    // 5. REPL
    es_perf_stats_t last_stats{};
    std::string line;
    GenerationContext gen_ctx;
    gen_ctx.show_thinking = show_thinking;
    gen_ctx.show_tokens = show_tokens;
    gen_ctx.max_thinking_tokens = max_thinking_tokens;

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
        if (line == "/thinking on") {
            gen_ctx.show_thinking = true;
            std::cout << COLOR_GREEN << "[Thinking display enabled]" << COLOR_RESET << std::endl;
            continue;
        }
        if (line == "/thinking off") {
            gen_ctx.show_thinking = false;
            std::cout << COLOR_YELLOW << "[Thinking display disabled]" << COLOR_RESET << std::endl;
            continue;
        }
        if (line == "/tokens on") {
            gen_ctx.show_tokens = true;
            std::cout << COLOR_GREEN << "[Token debug display enabled]" << COLOR_RESET << std::endl;
            continue;
        }
        if (line == "/tokens off") {
            gen_ctx.show_tokens = false;
            std::cout << COLOR_YELLOW << "[Token debug display disabled]" << COLOR_RESET << std::endl;
            continue;
        }
        if (line.rfind("/max-thinking", 0) == 0) {
            std::istringstream iss(line);
            std::string cmd;
            int n = 0;
            if (iss >> cmd >> n && n >= 0) {
                gen_ctx.max_thinking_tokens = n;
                std::cout << COLOR_GREEN << "[Max thinking tokens set to " << n << "]" << COLOR_RESET << std::endl;
            } else {
                std::cout << "Usage: /max-thinking <int>" << std::endl;
            }
            continue;
        }
        if (line.rfind("/max-tokens", 0) == 0) {
            std::istringstream iss(line);
            std::string cmd;
            int n = 0;
            if (iss >> cmd >> n && n > 0) {
                max_tokens = n;
                std::cout << COLOR_GREEN << "[Max tokens set to " << n << "]" << COLOR_RESET << std::endl;
            } else {
                std::cout << "Usage: /max-tokens <int>" << std::endl;
            }
            continue;
        }
        if (line == "/stats") {
            std::cout << COLOR_GRAY
                      << "Last stats: " << last_stats.prompt_tokens << " prompt tokens ("
                      << std::fixed << std::setprecision(1) << last_stats.prompt_eval_ms << "ms) | "
                      << last_stats.eval_tokens << " gen tokens (" << last_stats.eval_ms << "ms) | "
                      << std::setprecision(2) << last_stats.tokens_per_sec << " t/s"
                      << COLOR_RESET << std::endl;
            continue;
        }
        if (line == "/help") {
            std::cout << "Available commands:\n"
                      << "  /thinking [on|off]     Toggle streaming of thinking tokens\n"
                      << "  /tokens [on|off]       Toggle raw token ID debug mode\n"
                      << "  /max-thinking <N>      Set max thinking tokens (default: 128)\n"
                      << "  /max-tokens <N>        Set max generated tokens (default: 256)\n"
                      << "  /reset                 Clear KV cache and conversation context\n"
                      << "  /stats                 Show last turn timing and throughput\n"
                      << "  /quit, /exit           Exit program\n";
            continue;
        }

        std::string prompt = format_chatml(line);

        std::cout << COLOR_CYAN << COLOR_BOLD << "Qwen2 > " << COLOR_RESET;

        g_interrupt_flag.store(false);
        g_in_generation.store(true);
        gen_ctx.in_thinking = false;
        gen_ctx.thinking_count = 0;
        gen_ctx.total_count = 0;
        gen_ctx.thinking_limit_reached = false;

        status = es_session_generate(session, prompt.c_str(), token_callback, &gen_ctx, &last_stats);

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
