#ifndef ESWIN_MODEL_MANIFEST_HPP
#define ESWIN_MODEL_MANIFEST_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace eswin {

struct ModelFileDesc {
    std::string name;       // Local filename (e.g. "model_d0.model")
    std::string role;       // "weights_die0", "weights_die1", "tokenizer", "config"
    std::string sha256;     // SHA-256 hash for integrity verification
    uint64_t    size_bytes = 0;
    std::string url;        // Remote download URL (Hugging Face / ModelScope / CDN)
};

/// One compiled block loadable in a sequential multi-layer chain (Milestone 7).
struct BlockModelEntry {
    std::string name;         // .model filename, relative to the package directory
    uint16_t    die_id = 0;   // Die whose worker executes this block
    int32_t     stage = -1;   // Position in the chain; -1 keeps declaration order
};

struct ModelConfig {
    std::string architecture = "transformer";
    uint32_t    vocab_size = 49152;
    uint32_t    hidden_size = 576;
    uint32_t    intermediate_size = 1536;
    uint32_t    num_hidden_layers = 30;
    uint32_t    num_attention_heads = 9;
    uint32_t    num_key_value_heads = 3;
    float       rope_theta = 10000.0f;
    uint32_t    bos_token_id = 0;
    uint32_t    eos_token_id = 0;
    // Context window baked into the compiled loadables (the KV cache length they
    // were built with). Vendor ENNP config.json names this "token_num".
    uint32_t    context_length = 2048;
    // Per-head dimension. 0 means derive as hidden_size / num_attention_heads.
    // Vendor ENNP config.json names this "embedding_dim" (64 for Qwen2-0.5B).
    uint32_t    head_dim = 0;
    // V-cache tiling height, vendor "token_n2". Must come from the package: it is 16
    // for Qwen2-0.5B 1k, and guessing it silently mis-addresses the V cache.
    uint32_t    kv_token_n2 = 0;
    // Bytes per activation element, vendor "precision" (2 = fp16).
    uint32_t    elem_bytes = 2;
};

class ModelManifest {
public:
    ModelManifest() = default;

    bool load_from_file(const std::string &path, std::string &err);
    bool load_from_string(const std::string &json_str, std::string &err);
    bool save_to_file(const std::string &path, std::string &err) const;
    std::string to_json_string(int indent = 2) const;

    // Integrity check for files inside a base directory
    bool verify_files(const std::string &base_dir, std::string &err) const;

    // Helper accessors
    const ModelFileDesc* find_file_by_role(const std::string &role) const;

    /// Block chain entries assigned to one die, ordered by stage (then declaration).
    std::vector<BlockModelEntry> blocks_for_die(uint16_t die_id) const;

    // Properties
    std::string                name;
    std::string                version = "1.0.0";
    std::string                description;
    uint16_t                   tensor_parallel = 1; // 1 = Single Die, 2 = Dual Die
    ModelConfig                config;
    std::vector<ModelFileDesc> files;
    std::vector<BlockModelEntry> blocks;  // Sequential per-die block chain (Milestone 7)
};

} // namespace eswin

#endif // ESWIN_MODEL_MANIFEST_HPP
