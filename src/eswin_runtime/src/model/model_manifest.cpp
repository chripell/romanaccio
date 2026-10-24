#include "model_manifest.hpp"
#include "simple_json.hpp"
#include "sha256.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>

namespace eswin {

bool ModelManifest::load_from_string(const std::string &json_str, std::string &err) {
    JsonValue root = JsonValue::parse(json_str, err);
    if (!err.empty()) return false;
    if (!root.is_object()) {
        err = "Manifest root must be a JSON object";
        return false;
    }

    name = root["name"].as_string();
    version = root["version"].as_string("1.0.0");
    description = root["description"].as_string();
    tensor_parallel = static_cast<uint16_t>(root["tensor_parallel"].as_int(1));

    if (root.has("config") && root["config"].is_object()) {
        const auto &cfg = root["config"];
        config.architecture = cfg["architecture"].as_string("transformer");
        config.vocab_size = static_cast<uint32_t>(cfg["vocab_size"].as_int(49152));
        config.hidden_size = static_cast<uint32_t>(cfg["hidden_size"].as_int(576));
        config.intermediate_size = static_cast<uint32_t>(cfg["intermediate_size"].as_int(1536));
        config.num_hidden_layers = static_cast<uint32_t>(cfg["num_hidden_layers"].as_int(30));
        config.num_attention_heads = static_cast<uint32_t>(cfg["num_attention_heads"].as_int(9));
        config.num_key_value_heads = static_cast<uint32_t>(cfg["num_key_value_heads"].as_int(3));
        config.rope_theta = static_cast<float>(cfg["rope_theta"].as_double(10000.0));
        config.bos_token_id = static_cast<uint32_t>(cfg["bos_token_id"].as_int(0));
        config.eos_token_id = static_cast<uint32_t>(cfg["eos_token_id"].as_int(0));
    }

    // Context window. Open packages nest it under "config"; the vendor ENNP
    // config.json carries "token_num" at the document root and has no nested config
    // object at all, so both locations are honoured, config taking precedence.
    static const char *ctx_keys[] = {"context_length", "max_seq_len", "kv_cache_len", "token_num"};
    bool ctx_found = false;
    if (root.has("config") && root["config"].is_object()) {
        const auto &cfg = root["config"];
        for (const char *k : ctx_keys) {
            if (cfg.has(k)) {
                config.context_length = static_cast<uint32_t>(cfg[k].as_int(2048));
                ctx_found = true;
                break;
            }
        }
    }
    if (!ctx_found) {
        for (const char *k : ctx_keys) {
            if (root.has(k)) {
                config.context_length = static_cast<uint32_t>(root[k].as_int(2048));
                break;
            }
        }
    }

    files.clear();
    if (root.has("files") && root["files"].is_array()) {
        for (const auto &item : root["files"].as_array()) {
            if (!item.is_object()) continue;
            ModelFileDesc desc;
            desc.name = item["name"].as_string();
            desc.role = item["role"].as_string();
            desc.sha256 = item["sha256"].as_string();
            desc.size_bytes = static_cast<uint64_t>(item["size_bytes"].as_int64(0));
            desc.url = item["url"].as_string();
            files.push_back(desc);
        }
    }

    blocks.clear();
    if (root.has("blocks") && root["blocks"].is_array()) {
        // Open package format: explicit per-block die assignment and chain position.
        int32_t order = 0;
        for (const auto &item : root["blocks"].as_array()) {
            if (!item.is_object()) continue;
            BlockModelEntry b;
            b.name = item["name"].as_string();
            if (b.name.empty()) continue;
            b.die_id = static_cast<uint16_t>(item["die_id"].as_int(0));
            b.stage = static_cast<int32_t>(item["stage"].as_int(order));
            blocks.push_back(b);
            ++order;
        }
    } else if (root.has("model_list") && root["model_list"].is_array()) {
        // Vendor ENNP config.json format: one entry per device holding block_models[].
        int32_t order = 0;
        for (const auto &dev : root["model_list"].as_array()) {
            if (!dev.is_object() || !dev.has("block_models")) continue;
            const uint16_t die = static_cast<uint16_t>(dev["device_id"].as_int(0));
            for (const auto &bm : dev["block_models"].as_array()) {
                BlockModelEntry b;
                b.name = bm.as_string();
                if (b.name.empty()) continue;
                b.die_id = die;
                b.stage = order++;
                blocks.push_back(b);
            }
        }
    }

    return true;
}

bool ModelManifest::load_from_file(const std::string &path, std::string &err) {
    std::ifstream file(path);
    if (!file.is_open()) {
        err = "Could not open manifest file: " + path;
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return load_from_string(buffer.str(), err);
}

std::string ModelManifest::to_json_string(int indent) const {
    JsonValue root;
    root["name"] = name;
    root["version"] = version;
    root["description"] = description;
    root["tensor_parallel"] = static_cast<int>(tensor_parallel);

    JsonValue cfg;
    cfg["architecture"] = config.architecture;
    cfg["vocab_size"] = static_cast<int>(config.vocab_size);
    cfg["hidden_size"] = static_cast<int>(config.hidden_size);
    cfg["intermediate_size"] = static_cast<int>(config.intermediate_size);
    cfg["num_hidden_layers"] = static_cast<int>(config.num_hidden_layers);
    cfg["num_attention_heads"] = static_cast<int>(config.num_attention_heads);
    cfg["num_key_value_heads"] = static_cast<int>(config.num_key_value_heads);
    cfg["rope_theta"] = static_cast<double>(config.rope_theta);
    cfg["bos_token_id"] = static_cast<int>(config.bos_token_id);
    cfg["eos_token_id"] = static_cast<int>(config.eos_token_id);
    cfg["context_length"] = static_cast<int>(config.context_length);
    root["config"] = cfg;

    JsonValue arr;
    for (const auto &f : files) {
        JsonValue item;
        item["name"] = f.name;
        item["role"] = f.role;
        item["sha256"] = f.sha256;
        item["size_bytes"] = static_cast<int64_t>(f.size_bytes);
        item["url"] = f.url;
        arr.push_back(item);
    }
    root["files"] = arr;

    JsonValue barr;
    for (const auto &b : blocks) {
        JsonValue item;
        item["name"] = b.name;
        item["die_id"] = static_cast<int>(b.die_id);
        item["stage"] = static_cast<int>(b.stage);
        barr.push_back(item);
    }
    root["blocks"] = barr;

    return root.dump(indent);
}

bool ModelManifest::save_to_file(const std::string &path, std::string &err) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        err = "Could not create manifest file: " + path;
        return false;
    }
    file << to_json_string(2);
    return true;
}

bool ModelManifest::verify_files(const std::string &base_dir, std::string &err) const {
    for (const auto &f : files) {
        std::string full_path = base_dir;
        if (!full_path.empty() && full_path.back() != '/') full_path += '/';
        full_path += f.name;

        std::ifstream test(full_path, std::ios::binary | std::ios::ate);
        if (!test.is_open()) {
            err = "Missing model component: " + full_path;
            return false;
        }

        if (f.size_bytes > 0) {
            uint64_t actual_size = static_cast<uint64_t>(test.tellg());
            if (actual_size != f.size_bytes) {
                err = "Size mismatch for " + f.name + ": expected " +
                      std::to_string(f.size_bytes) + " bytes, got " + std::to_string(actual_size);
                return false;
            }
        }

        if (!f.sha256.empty()) {
            std::string actual_hash = Sha256::hash_file(full_path);
            if (actual_hash != f.sha256) {
                err = "SHA-256 checksum mismatch for " + f.name + ": expected " +
                      f.sha256 + ", got " + actual_hash;
                return false;
            }
        }
    }
    return true;
}

const ModelFileDesc* ModelManifest::find_file_by_role(const std::string &role) const {
    for (const auto &f : files) {
        if (f.role == role) return &f;
    }
    return nullptr;
}

std::vector<BlockModelEntry> ModelManifest::blocks_for_die(uint16_t die_id) const {
    std::vector<BlockModelEntry> out;
    for (const auto &b : blocks) {
        if (b.die_id == die_id) out.push_back(b);
    }
    // Declaration order is the fallback for entries that omit an explicit stage, so
    // the sort must be stable to keep those in file order.
    std::stable_sort(out.begin(), out.end(),
                     [](const BlockModelEntry &a, const BlockModelEntry &b) {
                         return a.stage < b.stage;
                     });
    return out;
}

} // namespace eswin
