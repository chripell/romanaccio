// ============================================================================
// es_qwen2_demo.cpp - Interactive Conversational LLM Engine for ESWIN EIC7700X
//
// Description:
//   A streamlined, modern command-line AI chatbot engine running on the ESWIN
//   EIC7700X Dual-NPU hardware platform. Replaces the legacy multi-choice menu
//   with a standard question/response chat REPL, full command-line customization
//   (system prompt, temperature, top-k, top-p, repetition penalty), and robust
//   signal handling.
//
// Features:
//   - Pure Question/Answer REPL loop ([YOU] -> [Assistant])
//   - Customizable System Prompt / Role via CLI argument (-s / --system)
//   - Sampling controls: Temperature, Top-K, Top-P, Repetition Penalty
//   - Hardware acceleration: Dual-Die NPU parallel execution (Die 0 & Die 1)
//   - Clean signal handling for SIGINT (Ctrl+C) and SIGTSTP (Ctrl+Z)
// ============================================================================

#include <csignal>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <limits>
#include <thread>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <sys/resource.h>

#include "es_qwen2_client.hpp"

// Global pointer to the client instance for signal handling cleanup
std::shared_ptr<Qwen2Client> gQwen2Client = nullptr;

// -----------------------------------------------------------------------------
// Signal Handlers
// -----------------------------------------------------------------------------
// Handles interrupt signals to cleanly release NPU hardware resources and SMMU
// DMA buffers instead of aborting abruptly.
void signalHandler(int signal)
{
    if (signal == SIGINT) {
        std::cout << "\n[!] Caught SIGINT (Ctrl+C). Terminating chat session cleanly...\n";
        if (gQwen2Client) {
            gQwen2Client->exitClient();
        }
        // Redirect stdin to EOF so any blocking getline unblocks immediately
        std::freopen("/dev/null", "r", stdin);
    } else if (signal == SIGTSTP) {
        std::cout << "\n[!] Caught SIGTSTP. Stopping active token generation...\n";
        Qwen2Client::mStopped.store(true);
    }
}

// -----------------------------------------------------------------------------
// Resource Limits Setup
// -----------------------------------------------------------------------------
// Increases the maximum open file descriptors limit (RLIMIT_NOFILE).
// The NPU multi-block runtime opens multiple memory-mapped DMA buffers, model
// handles, and device descriptors across both dies; setting this to 4096 prevents
// EMFILE / "Too many open files" errors.
int setFdLimit(int limit = 4096)
{
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == -1) {
        perror("getrlimit failed");
        return -1;
    }

    if (rl.rlim_cur < (rlim_t)limit) {
        rl.rlim_cur = limit;
        if (setrlimit(RLIMIT_NOFILE, &rl) == -1) {
            perror("setrlimit failed");
            return -1;
        }
    }
    return 0;
}

// -----------------------------------------------------------------------------
// Command-Line Help Banner
// -----------------------------------------------------------------------------
void printUsage(const char* progName)
{
    std::cout << "\n================================================================================\n"
              << "  ESWIN EIC7700X Dual-NPU Large Language Model Chatbot Engine\n"
              << "================================================================================\n\n"
              << "Usage:\n"
              << "  " << progName << " [options]\n\n"
              << "Options:\n"
              << "  -c, --config <path>           Path to model JSON configuration file.\n"
              << "                                (Default: auto-detect deepseek_7b_1k_int8_peer/config.json)\n"
              << "  -s, --system <prompt>         Custom system prompt / assistant persona.\n"
              << "                                (Default: \"You are a helpful and intelligent AI assistant.\")\n"
              << "  --show-thinking               Stream thinking tokens in gray (DEFAULT: on)\n"
              << "  --hide-thinking               Hide thinking tokens, showing only final response\n"
              << "  --max-thinking-tokens <int>   Cap thinking tokens before answering (default: 128)\n"
              << "  --show-tokens                 Debug mode: display [#id:\"piece\"] for all tokens\n"
              << "  -t, --temp <float>            Sampling temperature (e.g. 0.6, 0.7). Set to 0.0 for greedy argmax.\n"
              << "  -k, --top-k <int>             Top-K vocabulary filtering (e.g. 20, 40).\n"
              << "  -p, --top-p <float>           Top-P nucleus sampling threshold (e.g. 0.85, 0.90).\n"
              << "  -r, --repeat-penalty <float>  Repetition penalty factor (e.g. 1.10, 1.15).\n"
              << "  -w, --repeat-window <int>     Sliding window size for repetition penalty (Default: 64).\n"
              << "  -m, --max-tokens <int>        Maximum tokens generated per response.\n"
              << "  -h, --help                    Display this help message and exit.\n\n"
              << "Examples:\n"
              << "  # Run with default settings (auto-discovers Dual-Die DeepSeek-7B):\n"
              << "  " << progName << "\n\n"
              << "  # Custom persona with thinking tokens capped at 128:\n"
              << "  " << progName << " -s \"You are a witty coding assistant.\" --max-thinking-tokens 128\n\n"
              << "  # Clean answer only (no thinking reasoning displayed):\n"
              << "  " << progName << " --hide-thinking\n\n";
}

// -----------------------------------------------------------------------------
// Main Application Entry Point
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Auto-discover configuration file
    std::string configFile = QWEN2_CONFIG_FILE;
    std::ifstream test_f1("deepseek_7b_1k_int8_peer/config.json");
    std::ifstream test_f2("../qwen2_fix/deepseek_7b_1k_int8_peer/config.json");
    std::ifstream test_f3("/home/chri/qwen2_fix/deepseek_7b_1k_int8_peer/config.json");
    if (test_f1.good()) configFile = "deepseek_7b_1k_int8_peer/config.json";
    else if (test_f2.good()) configFile = "../qwen2_fix/deepseek_7b_1k_int8_peer/config.json";
    else if (test_f3.good()) configFile = "/home/chri/qwen2_fix/deepseek_7b_1k_int8_peer/config.json";

    std::string systemPrompt = "You are a helpful and intelligent AI assistant.";

    // Command-line override flags (negative or unset indicates use config file value)
    float cliTemperature = -1.0f;
    int cliTopK = -1;
    float cliTopP = -1.0f;
    float cliRepeatPenalty = -1.0f;
    int cliRepeatWindow = -1;
    int cliMaxTokens = -1;
    bool cliShowThinking = true;
    bool cliShowTokens = false;
    int cliMaxThinkingTokens = 128;

    // -------------------------------------------------------------------------
    // 1. Parse Command-Line Arguments
    // -------------------------------------------------------------------------
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if ((arg == "-h" || arg == "--help")) {
            printUsage(argv[0]);
            return 0;
        } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            configFile = argv[++i];
        } else if ((arg == "-s" || arg == "--system") && i + 1 < argc) {
            systemPrompt = argv[++i];
        } else if ((arg == "-t" || arg == "--temp" || arg == "--temperature") && i + 1 < argc) {
            cliTemperature = std::stof(argv[++i]);
        } else if ((arg == "-k" || arg == "--top-k") && i + 1 < argc) {
            cliTopK = std::stoi(argv[++i]);
        } else if ((arg == "-p" || arg == "--top-p") && i + 1 < argc) {
            cliTopP = std::stof(argv[++i]);
        } else if ((arg == "-r" || arg == "--repeat-penalty") && i + 1 < argc) {
            cliRepeatPenalty = std::stof(argv[++i]);
        } else if ((arg == "-w" || arg == "--repeat-window") && i + 1 < argc) {
            cliRepeatWindow = std::stoi(argv[++i]);
        } else if ((arg == "-m" || arg == "--max-tokens") && i + 1 < argc) {
            cliMaxTokens = std::stoi(argv[++i]);
        } else if (arg == "--show-thinking") {
            cliShowThinking = true;
        } else if (arg == "--hide-thinking" || arg == "--no-thinking") {
            cliShowThinking = false;
        } else if (arg == "--max-thinking-tokens" && i + 1 < argc) {
            cliMaxThinkingTokens = std::stoi(argv[++i]);
        } else if (arg == "--show-tokens" || arg == "--debug-tokens") {
            cliShowTokens = true;
        } else if (arg[0] != '-' && i == 1 && argc == 2) {
            // Backwards compatibility: allow passing config file directly as first argument
            configFile = arg;
        } else {
            std::cerr << "[!] Unknown or incomplete argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // -------------------------------------------------------------------------
    // 2. System and Signal Setup
    // -------------------------------------------------------------------------
    if (setFdLimit(4096) != 0) {
        std::cerr << "[!] Warning: Unable to increase file descriptor limit to 4096.\n";
    }

    // Register POSIX signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTSTP, signalHandler);

    std::cout << "================================================================================\n";
    std::cout << "  ESWIN EIC7700X Large Language Model Engine (DeepSeek / Qwen2)\n";
    std::cout << "  Dual-NPU Accelerated Hardware Inference\n";
    std::cout << "================================================================================\n\n";

    std::cout << "[*] Loading configuration: " << configFile << "\n";
    std::cout << "[*] System Prompt        : \"" << systemPrompt << "\"\n";

    // -------------------------------------------------------------------------
    // 3. Initialize Qwen2 Client & NPU Accelerator
    // -------------------------------------------------------------------------
    try {
        gQwen2Client = std::make_shared<Qwen2Client>();

        // Load models, tokenizer, and embedding tables onto the dual NPUs
        gQwen2Client->start(configFile);

        // Retrieve reference to loaded configuration to apply any CLI overrides
        CLIENT_CONFIG_S &cfg = gQwen2Client->getConfig();

        if (cliTemperature >= 0.0f) {
            cfg.temperature = cliTemperature;
            if (cliTemperature == 0.0f) {
                // Greedy argmax decoding
                cfg.doSample = false;
                cfg.topK = 1;
            } else {
                cfg.doSample = true;
            }
        }
        if (cliTopK > 0) {
            cfg.topK = cliTopK;
        }
        if (cliTopP >= 0.0f) {
            cfg.topP = cliTopP;
        }
        if (cliRepeatPenalty > 0.0f) {
            cfg.repetitionPenalty = cliRepeatPenalty;
        }
        if (cliRepeatWindow > 0) {
            cfg.repeatWindow = cliRepeatWindow;
        }
        if (cliMaxTokens > 0) {
            cfg.responseMaxLen = cliMaxTokens;
        }

        cfg.showThinking = cliShowThinking;
        cfg.showTokens = cliShowTokens;
        if (cliMaxThinkingTokens >= 0) {
            cfg.maxThinkingTokens = cliMaxThinkingTokens;
        }

        // Display active inference hyperparameters
        std::cout << "\n[*] Active Generation Hyperparameters:\n"
                  << "    - Sampling Mode      : " << (cfg.doSample ? "Top-P / Top-K Sampling" : "Greedy (ArgMax)") << "\n"
                  << "    - Temperature        : " << std::fixed << std::setprecision(2) << cfg.temperature << "\n"
                  << "    - Top-P Nucleus      : " << cfg.topP << "\n"
                  << "    - Top-K              : " << cfg.topK << "\n"
                  << "    - Repetition Penalty : " << cfg.repetitionPenalty << " (Sliding Window: " << cfg.repeatWindow << ")\n"
                  << "    - Max Response Length: " << cfg.responseMaxLen << " tokens\n"
                  << "    - Thinking Tokens    : " << (cfg.showThinking ? "\033[32mENABLED (gray stream, max=" + std::to_string(cfg.maxThinkingTokens) + ")\033[0m" : "\033[33mHIDDEN\033[0m") << "\n"
                  << "    - Token Debug        : " << (cfg.showTokens ? "\033[32mON (displaying [#id:\"word\"])\033[0m" : "\033[90mOFF (type /tokens on to enable)\033[0m") << "\n\n";

        std::cout << "[+] Model ready! Interactive chat session initiated.\n";
        std::cout << "    Type your question and press Enter. Type 'exit' or 'quit' to leave.\n\n";

        // ---------------------------------------------------------------------
        // 4. Standard Interactive Chat REPL Loop
        // ---------------------------------------------------------------------
        std::string inputPrompt;

        while (!Qwen2Client::mExited.load()) {
            std::cout << "[YOU]: " << std::flush;

            // Read complete user input line from stdin
            if (!std::getline(std::cin, inputPrompt)) {
                // EOF encountered (e.g. Ctrl+D)
                break;
            }

            // Skip empty inputs (e.g. accidental extra enter)
            if (inputPrompt.empty()) {
                continue;
            }

            // Clean exit check
            if (inputPrompt == "exit" || inputPrompt == "quit" || inputPrompt == ":q" || inputPrompt == "/exit" || inputPrompt == "/quit") {
                std::cout << "\n[*] Exiting chat session...\n";
                gQwen2Client->exitClient();
                break;
            }

            if (inputPrompt == "/thinking on") {
                cfg.showThinking = true;
                std::cout << "\033[32m[Thinking display enabled]\033[0m\n\n";
                continue;
            }
            if (inputPrompt == "/thinking off") {
                cfg.showThinking = false;
                std::cout << "\033[33m[Thinking display disabled]\033[0m\n\n";
                continue;
            }
            if (inputPrompt == "/tokens on") {
                cfg.showTokens = true;
                std::cout << "\033[32m[Token debug display enabled]\033[0m\n\n";
                continue;
            }
            if (inputPrompt == "/tokens off") {
                cfg.showTokens = false;
                std::cout << "\033[33m[Token debug display disabled]\033[0m\n\n";
                continue;
            }
            if (inputPrompt.rfind("/max-thinking", 0) == 0) {
                std::istringstream iss(inputPrompt);
                std::string cmd;
                int n = 0;
                if (iss >> cmd >> n && n >= 0) {
                    cfg.maxThinkingTokens = n;
                    std::cout << "\033[32m[Max thinking tokens set to " << n << "]\033[0m\n\n";
                } else {
                    std::cout << "Usage: /max-thinking <int>\n\n";
                }
                continue;
            }
            if (inputPrompt == "/help") {
                std::cout << "Available commands:\n"
                          << "  /thinking [on|off]   Toggle real-time streaming of thinking tokens\n"
                          << "  /tokens [on|off]     Toggle raw token ID debug mode\n"
                          << "  /max-thinking <N>    Set max thinking tokens (default: 128)\n"
                          << "  exit, quit, /exit    Exit chat engine\n\n";
                continue;
            }

            // Format response label matching the active model
            std::string assistantLabel = cfg.modelName.empty() ? "Assistant" : cfg.modelName;
            std::cout << "[" << assistantLabel << "]: " << std::flush;

            // Execute end-to-end NPU inference (prefill prompt + autoregressive decode)
            // Tokens are streamed directly to stdout as they are produced by the NPU master thread
            gQwen2Client->genResponse(systemPrompt, inputPrompt);

            // Spacing for readability between dialogue turns
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[-] Fatal error during execution: " << e.what() << std::endl;
        if (gQwen2Client) {
            gQwen2Client->exitClient();
        }
        return 1;
    }

    std::cout << "[+] Goodbye!\n";
    return 0;
}
