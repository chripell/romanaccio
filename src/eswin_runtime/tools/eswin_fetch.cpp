#include "../src/model/model_manifest.hpp"
#include "../src/model/downloader.hpp"
#include "../src/model/sha256.hpp"
#include <iostream>
#include <string>
#include <cstring>

static void print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " <manifest.json | model-name> [options]\n\n"
              << "Options:\n"
              << "  --cache-dir <dir>   Specify local cache directory (default: ~/.cache/eswin/models)\n"
              << "  --verify-only       Verify existing local files without downloading\n"
              << "  --help              Display this help message\n\n"
              << "Standard Built-in Models:\n"
              << "  smollm-135m         SmolLM 135M int8 dual-die loadable\n"
              << "  qwen2.5-0.5b        Qwen 2.5 0.5B int8 dual-die loadable\n"
              << "  yolov3              YOLOv3 object detection dual-die loadable\n";
}

static eswin::ModelManifest create_builtin_manifest(const std::string &name) {
    eswin::ModelManifest m;
    if (name == "smollm-135m" || name == "smollm") {
        m.name = "SmolLM-135M-int8";
        m.version = "1.0.0";
        m.description = "SmolLM 135M Int8 Dual-Die NPU model";
        m.tensor_parallel = 2;
        m.config.architecture = "smollm";
        m.config.vocab_size = 49152;
        m.config.hidden_size = 576;
        m.config.intermediate_size = 1536;
        m.config.num_hidden_layers = 30;
        m.config.num_attention_heads = 9;
        m.config.num_key_value_heads = 3;
        m.config.rope_theta = 10000.0f;
        m.config.bos_token_id = 0;
        m.config.eos_token_id = 0;

        m.files.push_back({
            "smollm_die0.model",
            "weights_die0",
            "",
            0,
            "https://huggingface.co/HuggingFaceTB/SmolLM-135M/raw/main/config.json"
        });
        m.files.push_back({
            "smollm_die1.model",
            "weights_die1",
            "",
            0,
            "https://huggingface.co/HuggingFaceTB/SmolLM-135M/raw/main/config.json"
        });
        m.files.push_back({
            "tokenizer.json",
            "tokenizer",
            "",
            0,
            "https://huggingface.co/HuggingFaceTB/SmolLM-135M/raw/main/tokenizer.json"
        });
    } else if (name == "qwen2.5-0.5b" || name == "qwen2") {
        m.name = "Qwen2.5-0.5B-int8";
        m.version = "1.0.0";
        m.description = "Qwen 2.5 0.5B Int8 Dual-Die NPU model";
        m.tensor_parallel = 2;
        m.config.architecture = "qwen2";
        m.config.vocab_size = 151936;
        m.config.hidden_size = 896;
        m.config.intermediate_size = 4864;
        m.config.num_hidden_layers = 24;
        m.config.num_attention_heads = 14;
        m.config.num_key_value_heads = 2;
        m.config.rope_theta = 1000000.0f;
        m.config.bos_token_id = 151643;
        m.config.eos_token_id = 151643;
    } else if (name == "yolov3") {
        m.name = "yolov3";
        m.version = "1.0.0";
        m.description = "YOLOv3 416x416 Darknet loadable";
        m.tensor_parallel = 1;
        m.config.architecture = "darknet_yolo";
    }
    return m;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string target = argv[1];
    if (target == "--help" || target == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    std::string cache_dir = eswin::Downloader::get_default_cache_dir();
    bool verify_only = false;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cache-dir") == 0 && i + 1 < argc) {
            cache_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--verify-only") == 0) {
            verify_only = true;
        }
    }

    eswin::ModelManifest manifest;
    std::string err;

    if (target.find(".json") != std::string::npos) {
        if (!manifest.load_from_file(target, err)) {
            std::cerr << "[ERROR] Failed to parse manifest from " << target << ": " << err << std::endl;
            return 1;
        }
    } else {
        manifest = create_builtin_manifest(target);
        if (manifest.name.empty()) {
            std::cerr << "[ERROR] Unknown built-in model: " << target << std::endl;
            return 1;
        }
    }

    std::cout << "========================================================\n"
              << "  ESWIN Model Fetch & Verification Tool (eswin-fetch)    \n"
              << "========================================================\n\n"
              << "[*] Model:            " << manifest.name << " (v" << manifest.version << ")\n"
              << "[*] Architecture:     " << manifest.config.architecture << "\n"
              << "[*] Tensor Parallel:  " << manifest.tensor_parallel << " die(s)\n"
              << "[*] Target Cache:     " << cache_dir << "/" << manifest.name << "\n\n";

    if (verify_only) {
        std::string verify_dir = cache_dir + "/" + manifest.name;
        if (manifest.verify_files(verify_dir, err)) {
            std::cout << "\033[1;32m[PASS]\033[0m All model files verified successfully in "
                      << verify_dir << std::endl;
            return 0;
        } else {
            std::cerr << "\033[1;31m[FAIL]\033[0m Verification failed: " << err << std::endl;
            return 2;
        }
    }

    std::string out_dir;
    auto progress = [](const std::string &name, float p) {
        std::cout << "    [" << static_cast<int>(p * 100) << "%] " << name << std::endl;
    };

    if (!eswin::Downloader::ensure_model(manifest, cache_dir, out_dir, err, progress)) {
        std::cerr << "\033[1;31m[ERROR]\033[0m Failed to fetch model: " << err << std::endl;
        return 2;
    }

    std::cout << "\n\033[1;32m[SUCCESS]\033[0m Model is ready at: " << out_dir << std::endl;
    return 0;
}
