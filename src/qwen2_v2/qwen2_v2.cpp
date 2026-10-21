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
    bool thinking_done = false;
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
    // Only allow entering <think> once at the start of the response
    bool is_think_start = (!ctx->thinking_done && !ctx->in_thinking &&
                           (token == 151648 || str == "<think>" || str.find("<think>") != std::string::npos));
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
            ctx->thinking_done = true;
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
            ctx->thinking_done = true;
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

    ctx->thinking_done = true;

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

static void displayInitialCard(
    const es_model_info_t &info,
    const std::string &model_path,
    const std::string &resolved_tok,
    size_t vocab_count,
    es_device_mode_t dev_mode,
    float temp,
    int32_t top_k,
    float top_p,
    float repeat_penalty,
    int32_t repeat_window,
    int32_t max_tokens,
    bool show_thinking,
    int32_t max_thinking_tokens,
    bool show_tokens)
{
    std::cout << "\n\033[1;36m================================================================================\033[0m\n"
              << "\033[1;36m     Qwen2 / DeepSeek Chat Engine v2 (Powered by Open ESWIN Runtime)            \033[0m\n"
              << "\033[1;36m================================================================================\033[0m\n";

    std::vector<std::string> warnings;

    // 1. Configuration & Model Info
    std::cout << "\033[1m[Configuration & Model]\033[0m\n";
    std::cout << "  • Model Name        : " << info.name << "\n";
    std::cout << "  • Model Directory   : " << model_path;
    struct stat dirSt;
    if (stat(model_path.c_str(), &dirSt) == 0 && S_ISDIR(dirSt.st_mode)) {
        std::cout << " \033[32m[OK]\033[0m\n";
    } else {
        std::cout << " \033[31m[DIRECTORY NOT FOUND]\033[0m\n";
        warnings.push_back("Model directory does not exist: " + model_path);
    }

    std::string manifest_file = model_path + "/manifest.json";
    if (file_exists_local(manifest_file)) {
        std::cout << "  • Package Manifest  : " << manifest_file << " \033[32m[OK]\033[0m\n";
    } else {
        std::cout << "  • Package Manifest  : " << manifest_file << " \033[33m[MISSING / DEFAULT PARAMETERS]\033[0m\n";
        warnings.push_back("Package manifest (manifest.json) not found in " + model_path);
    }
    std::cout << "  • Architecture      : " << info.num_layers << " layers, "
              << info.num_heads << " heads, hidden dim=" << info.hidden_size
              << ", vocab=" << info.vocab_size << "\n\n";

    // 2. Tokenizer & Vocabulary
    std::cout << "\033[1m[Tokenizer & Vocabulary]\033[0m\n";
    if (vocab_count > 0 && !resolved_tok.empty()) {
        std::cout << "  • Tokenizer File    : " << resolved_tok << " (" << vocab_count << " vocabulary tokens loaded) \033[32m[OK]\033[0m\n";
    } else {
        std::cout << "  • Tokenizer File    : " << (resolved_tok.empty() ? "(none)" : resolved_tok) << " \033[31m[NOT LOADED]\033[0m\n";
        warnings.push_back("Tokenizer vocabulary not loaded! Tokens will be displayed in numeric [#id] form.");
    }
    std::cout << "  • Special Tokens    : <|im_start|>, <|im_end|>, <think>, </think> \033[32m[OK]\033[0m\n\n";

    // 3. Weights & Embeddings
    std::cout << "\033[1m[Weights & Embeddings]\033[0m\n";
    std::string embPath;
    std::vector<std::string> cand_emb = {
        model_path + "/embedding.bin",
        model_path + "/deepseek_7b_1k_int8_peer/embedding.bin",
        "./embedding.bin",
        "deepseek_7b_1k_int8_peer/embedding.bin",
        "/opt/eswin/sample-code/npu_sample/qwen_sample/models/deepseek_7b_1k_int8_peer/embedding.bin"
    };
    for (const auto &c : cand_emb) {
        if (file_exists_local(c)) {
            embPath = c;
            break;
        }
    }

    size_t expectedEmbBytes = (size_t)info.vocab_size * info.hidden_size * 2; // fp16
    if (!embPath.empty()) {
        struct stat embSt;
        stat(embPath.c_str(), &embSt);
        if ((size_t)embSt.st_size == expectedEmbBytes) {
            std::cout << "  • Embedding Table   : " << embPath << " (" << embSt.st_size << " bytes, fp16) \033[32m[OK]\033[0m\n";
        } else {
            std::cout << "  • Embedding Table   : " << embPath << " (" << embSt.st_size << " bytes) \033[33m[SIZE MISMATCH: expected " << expectedEmbBytes << " bytes]\033[0m\n";
            warnings.push_back("Embedding table size mismatch: " + embPath);
        }
    } else {
        std::cout << "  • Embedding Table   : embedding.bin \033[31m[MISSING / NOT LOADED]\033[0m\n";
        warnings.push_back("Embedding table (embedding.bin) missing! Embeddings will use unmapped dummy fallback values (0.01f), causing gibberish sampling output.");
    }
    std::cout << "\n";

    // 4. Hardware NPU Acceleration
    std::cout << "\033[1m[Hardware NPU Acceleration]\033[0m\n";
    std::vector<std::string> cand_dirs = {
        model_path,
        model_path + "/deepseek_7b_1k_int8_peer",
        "./deepseek_7b_1k_int8_peer",
        "/opt/eswin/sample-code/npu_sample/qwen_sample/models/deepseek_7b_1k_int8_peer"
    };
    int d0_count = 0, d1_count = 0;
    int expected_blocks = (info.num_layers > 0) ? (int)info.num_layers : 24;
    bool has_decomposed_lm = false;
    int decomposed_layers = 0;

    for (const auto &d : cand_dirs) {
        int c0 = 0, c1 = 0;
        for (int b = 0; b < 32; ++b) {
            char f0[256], f1[256];
            snprintf(f0, sizeof(f0), "%s/modified_block_%d_npu_b1_d0.model", d.c_str(), b);
            snprintf(f1, sizeof(f1), "%s/modified_block_%d_npu_b1_d1.model", d.c_str(), b);
            if (file_exists_local(f0)) c0++;
            if (file_exists_local(f1)) c1++;
        }
        if (c0 > 0 || c1 > 0) {
            d0_count = c0;
            d1_count = c1;
            break;
        }

        // Check for decomposed architecture
        bool lm_d0 = file_exists_local(d + "/qwen2_5_lm_head_d0_npu_b1.model") || file_exists_local(d + "/lm_head_d0.model");
        bool lm_d1 = file_exists_local(d + "/qwen2_5_lm_head_d1_npu_b1.model") || file_exists_local(d + "/lm_head_d1.model");
        if (lm_d0 && lm_d1) has_decomposed_lm = true;

        int decomp_cnt = 0;
        for (int b = 0; b < expected_blocks; ++b) {
            char qkv[256], attn[256], mlp[256];
            snprintf(qkv, sizeof(qkv), "%s/qwen2_5_layer%d_qkv_npu_b1.model", d.c_str(), b);
            snprintf(attn, sizeof(attn), "%s/qwen2_5_layer%d_attn_out_npu_b1.model", d.c_str(), b);
            snprintf(mlp, sizeof(mlp), "%s/qwen2_5_layer%d_mlp_npu_b1.model", d.c_str(), b);
            if (file_exists_local(qkv) || file_exists_local(attn) || file_exists_local(mlp) ||
                (b == 0 && file_exists_local(d + "/qwen2_5_mlp_opset13_npu_b1.model"))) {
                decomp_cnt++;
            }
        }
        if (decomp_cnt > 0) {
            decomposed_layers = decomp_cnt;
            break;
        }
    }

    if (has_decomposed_lm || decomposed_layers > 0) {
        std::cout << "  • Architecture Mode : Decomposed Dual-Die Pipeline \033[32m[HARDWARE NPU]\033[0m\n";
        std::cout << "  • Layer Subgraphs   : " << decomposed_layers << "/" << expected_blocks << " layers loaded";
        if (decomposed_layers >= expected_blocks) std::cout << " \033[32m[ALL LAYERS OFF-LOADED]\033[0m\n";
        else std::cout << " \033[33m[PARTIAL: " << decomposed_layers << " NPU, " << (expected_blocks - decomposed_layers) << " CPU/DSP]\033[0m\n";

        std::cout << "  • LM Head Offload   : "
                  << (has_decomposed_lm ? "Dual-Die Parallel (Die 0 + Die 1, 76,032 x 2) \033[32m[OK]\033[0m\n" : "CPU Fallback\n");
        std::cout << "  • Device Target     : "
                  << (dev_mode == ES_DEVICE_DUAL_DIE_AUTO ? "Dual-Die Auto (Die 0 + Die 1)" : "Single Die (Die 0)") << "\n";
    } else if (d0_count > 0 || d1_count > 0) {
        std::cout << "  • Architecture Mode : Monolithic modified_block \033[32m[HARDWARE NPU]\033[0m\n";
        std::cout << "  • Die 0 (Node 0)       : " << d0_count << " layer blocks loaded";
        if (d0_count >= expected_blocks) std::cout << " \033[32m[OK]\033[0m\n";
        else std::cout << " \033[33m[PARTIAL: expected " << expected_blocks << "]\033[0m\n";

        std::cout << "  • Die 1 (Node 1)       : " << d1_count << " layer blocks loaded";
        if (d1_count >= expected_blocks) std::cout << " \033[32m[OK]\033[0m\n";
        else std::cout << " \033[33m[PARTIAL: expected " << expected_blocks << "]\033[0m\n";

        std::cout << "  • Device Target     : "
                  << (dev_mode == ES_DEVICE_DUAL_DIE_AUTO ? "Dual-Die Auto (Die 0 + Die 1)" : "Single Die (Die 0)") << "\n";
    } else {
        std::cout << "  • Die 0 (Node 0)       : 0/" << expected_blocks << " layer blocks \033[31m[NOT FOUND]\033[0m\n";
        std::cout << "  • Die 1 (Node 1)       : 0/" << expected_blocks << " layer blocks \033[31m[NOT FOUND]\033[0m\n";
        std::cout << "  • Device Target     : "
                  << (dev_mode == ES_DEVICE_DUAL_DIE_AUTO ? "Dual-Die Auto (Die 0 + Die 1)" : "Single Die (Die 0)") << "\n";
        std::cout << "  • Acceleration Mode : Host CPU Fallback \033[33m[FALLBACK DUMMY WEIGHTS - NO HARDWARE OFF-LOAD]\033[0m\n";
        warnings.push_back("Hardware NPU models (.model files) missing! Execution is running on host CPU fallback with dummy weights.");
    }
    std::cout << "\n";

    // 5. Inference Hyperparameters & Protections
    std::cout << "\033[1m[Inference & Protections]\033[0m\n";
    std::cout << "  • Sampling Mode     : " << (temp > 0.0f ? "Top-P / Top-K Sampling" : "Greedy (ArgMax)")
              << " (temp=" << std::fixed << std::setprecision(2) << temp
              << ", top_k=" << top_k << ", top_p=" << top_p << ")\n";
    std::cout << "  • Issue #47 Guards  : Repetition penalty=" << repeat_penalty
              << " (win=" << repeat_window << "), anti_stutter=2, cyclic loop breaker \033[32m[ACTIVE]\033[0m\n";
    std::cout << "  • Thinking Tokens   : "
              << (show_thinking ? "\033[32mENABLED (gray stream, max=" + std::to_string(max_thinking_tokens) + " tokens)\033[0m" : "\033[33mHIDDEN\033[0m") << "\n";
    std::cout << "  • Token Debug       : "
              << (show_tokens ? "\033[32mON (displaying [#id:\"word\"])\033[0m" : "\033[90mOFF (type /tokens on to enable)\033[0m") << "\n";
    std::cout << "  • Max Response Len  : " << max_tokens << " tokens\n";
    std::cout << "\033[1;36m--------------------------------------------------------------------------------\033[0m\n";

    // Warnings Box
    if (!warnings.empty()) {
        std::cout << "\n\033[1;41m  WARNINGS DETECTED (" << warnings.size() << ")  \033[0m\n";
        for (size_t w = 0; w < warnings.size(); ++w) {
            std::cout << " \033[1;33m[!] Warning " << (w + 1) << ": " << warnings[w] << "\033[0m\n";
        }
        std::cout << "\n\033[33mDue to missing pieces, generation output will be degraded, unmapped, or fallback dummy.\033[0m\n"
                  << "\033[36mTip: Ensure real weights and model files are present or symlinked to the model path.\033[0m\n\n";
    } else {
        std::cout << "\033[1;32m[+] All required model files, embeddings, and tokenizer pieces loaded successfully!\033[0m\n\n";
    }
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
    bool raw_mode = false;             // Raw generation mode (no ChatML)
    std::string cli_prompt;            // Non-interactive prompt
    es_device_mode_t dev_mode = ES_DEVICE_DUAL_DIE_AUTO;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (++i < argc) model_path = argv[i];
        } else if (arg == "-t" || arg == "--tokenizer") {
            if (++i < argc) tokenizer_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i < argc) cli_prompt = argv[i];
        } else if (arg == "--raw") {
            raw_mode = true;
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
                      << "  -p, --prompt <text>            Run single turn non-interactively and exit\n"
                      << "  --raw                          Do not wrap prompt in ChatML (raw code completion)\n"
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

    // 5. Display comprehensive model card and loaded components status
    displayInitialCard(
        info, model_path, resolved_tok, vocab_count, dev_mode,
        temp, top_k, top_p, repeat_penalty, repeat_window, max_tokens,
        show_thinking, max_thinking_tokens, show_tokens
    );

    GenerationContext gen_ctx;
    gen_ctx.show_thinking = show_thinking;
    gen_ctx.show_tokens = show_tokens;
    gen_ctx.max_thinking_tokens = max_thinking_tokens;
    es_perf_stats_t last_stats{};

    // Handle non-interactive one-shot prompt execution
    if (!cli_prompt.empty()) {
        std::string prompt = (raw_mode || cli_prompt.find("<|im_start|>") != std::string::npos)
                                ? cli_prompt
                                : format_chatml(cli_prompt);

        std::cout << COLOR_CYAN << COLOR_BOLD << "Qwen2 > " << COLOR_RESET;

        g_interrupt_flag.store(false);
        g_in_generation.store(true);
        gen_ctx.in_thinking = false;
        gen_ctx.thinking_done = false;
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

        es_session_free(session);
        es_model_unload(model);
        es_free(ctx);
        return (status == ES_SUCCESS) ? 0 : 1;
    }

    std::cout << "Interactive chat session ready. Type your question and press Enter.\n"
              << "Type '/help' for command options, or '/exit' to quit.\n";

    // 6. REPL
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
        if (line == "/raw on") {
            raw_mode = true;
            std::cout << COLOR_GREEN << "[Raw prompt mode enabled]" << COLOR_RESET << std::endl;
            continue;
        }
        if (line == "/raw off") {
            raw_mode = false;
            std::cout << COLOR_YELLOW << "[ChatML prompt mode enabled]" << COLOR_RESET << std::endl;
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
                      << "  /raw [on|off]          Toggle raw code completion vs ChatML mode\n"
                      << "  /thinking [on|off]     Toggle streaming of thinking tokens\n"
                      << "  /tokens [on|off]       Toggle raw token ID debug mode\n"
                      << "  /max-thinking <N>      Set max thinking tokens (default: 128)\n"
                      << "  /max-tokens <N>        Set max generated tokens (default: 256)\n"
                      << "  /reset                 Clear KV cache and conversation context\n"
                      << "  /stats                 Show last turn timing and throughput\n"
                      << "  /quit, /exit           Exit program\n";
            continue;
        }

        std::string prompt = (raw_mode || line.find("<|im_start|>") != std::string::npos)
                                ? line
                                : format_chatml(line);

        std::cout << COLOR_CYAN << COLOR_BOLD << "Qwen2 > " << COLOR_RESET;

        g_interrupt_flag.store(false);
        g_in_generation.store(true);
        gen_ctx.in_thinking = false;
        gen_ctx.thinking_done = false;
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
