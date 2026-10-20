#include "tokenizer.hpp"
#include "../model/simple_json.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace eswin {

static const std::string BASE64_CHARS =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static std::string base64_decode(const std::string &in) {
    std::string out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[static_cast<uint8_t>(BASE64_CHARS[i])] = i;

    int val = 0, valb = -8;
    for (uint8_t c : in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

void Tokenizer::add_token(int32_t id, const std::string &piece) {
    if (id < 0) return;
    if (static_cast<size_t>(id) >= m_id_to_piece.size()) {
        m_id_to_piece.resize(id + 1);
    }
    m_id_to_piece[id] = piece;
    m_piece_to_id[piece] = id;

    if (!m_uses_gpt2_space && piece.size() >= 2 &&
        static_cast<unsigned char>(piece[0]) == 0xC4 && static_cast<unsigned char>(piece[1]) == 0xA0) {
        m_uses_gpt2_space = true;
    }
    if (!m_uses_spm_space && piece.size() >= 3 &&
        static_cast<unsigned char>(piece[0]) == 0xE2 && static_cast<unsigned char>(piece[1]) == 0x96 && static_cast<unsigned char>(piece[2]) == 0x81) {
        m_uses_spm_space = true;
    }
}

bool Tokenizer::load_json(const std::string &json_str, std::string &err) {
    JsonValue root = JsonValue::parse(json_str, err);
    if (!err.empty()) return false;

    // Check "model" -> "vocab"
    if (root.has("model") && root["model"].is_object()) {
        const auto &model = root["model"];
        if (model.has("vocab") && model["vocab"].is_object()) {
            for (const auto &pair : model["vocab"].as_object()) {
                int32_t id = pair.second.as_int(-1);
                if (id >= 0) add_token(id, pair.first);
            }
        }
    }

    // Check "added_tokens"
    if (root.has("added_tokens") && root["added_tokens"].is_array()) {
        for (const auto &item : root["added_tokens"].as_array()) {
            if (item.is_object() && item.has("id") && item.has("content")) {
                int32_t id = item["id"].as_int(-1);
                std::string content = item["content"].as_string();
                if (id >= 0 && !content.empty()) {
                    add_token(id, content);
                }
            }
        }
    }

    if (m_id_to_piece.empty()) {
        err = "No vocabulary items parsed from JSON";
        return false;
    }
    return true;
}

bool Tokenizer::load_binary(const std::string &bin_path, std::string &err) {
    std::ifstream f(bin_path, std::ios::binary);
    if (!f.is_open()) {
        err = "Failed to open binary tokenizer file: " + bin_path;
        return false;
    }

    int32_t count = 0;
    if (!f.read(reinterpret_cast<char*>(&count), sizeof(int32_t))) {
        err = "Failed to read vocab size from " + bin_path;
        return false;
    }

    m_id_to_piece.clear();
    m_piece_to_id.clear();
    m_id_to_piece.resize(count);

    for (int32_t i = 0; i < count; ++i) {
        int32_t len = 0;
        if (!f.read(reinterpret_cast<char*>(&len), sizeof(int32_t))) break;
        std::string s(len, '\0');
        if (len > 0) {
            f.read(&s[0], len);
        }
        m_id_to_piece[i] = s;
        m_piece_to_id[s] = i;
    }

    return true;
}

bool Tokenizer::load_tiktoken(const std::string &tiktoken_path, std::string &err) {
    std::ifstream file(tiktoken_path);
    if (!file.is_open()) {
        err = "Failed to open tiktoken file: " + tiktoken_path;
        return false;
    }

    m_id_to_piece.clear();
    m_piece_to_id.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;

        std::string b64 = line.substr(0, sp);
        int32_t rank = std::stoi(line.substr(sp + 1));
        std::string piece = base64_decode(b64);
        add_token(rank, piece);
    }

    // Add standard Qwen / DeepSeek special tokens if not already registered
    const std::vector<std::pair<int32_t, std::string>> qwen_specials = {
        {151643, "<|endoftext|>"},
        {151644, "<|im_start|>"},
        {151645, "<|im_end|>"},
        {151646, "<|object_ref_start|>"},
        {151647, "<|object_ref_end|>"},
        {151648, "<think>"},
        {151649, "</think>"},
        {151650, "<|box_start|>"},
        {151651, "<|box_end|>"},
        {151652, "<|quad_start|>"},
        {151653, "<|quad_end|>"},
        {151654, "<|vision_start|>"},
        {151655, "<|vision_end|>"},
        {151656, "<|vision_pad|>"},
        {151657, "<|image_pad|>"},
        {151658, "<|video_pad|>"}
    };
    for (const auto &sp : qwen_specials) {
        if (static_cast<size_t>(sp.first) >= m_id_to_piece.size() || m_id_to_piece[sp.first].empty()) {
            add_token(sp.first, sp.second);
        }
    }

    return !m_id_to_piece.empty();
}

bool Tokenizer::load(const std::string &filepath, std::string &err) {
    if (filepath.rfind(".json") != std::string::npos) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            err = "Cannot open file: " + filepath;
            return false;
        }
        std::stringstream ss;
        ss << file.rdbuf();
        return load_json(ss.str(), err);
    } else if (filepath.rfind(".tiktoken") != std::string::npos) {
        return load_tiktoken(filepath, err);
    } else {
        return load_binary(filepath, err);
    }
}

void Tokenizer::set_special_tokens(int32_t bos, int32_t eos, int32_t pad) {
    m_bos_id = bos;
    m_eos_id = eos;
    m_pad_id = pad;
}

bool Tokenizer::contains(const std::string &piece) const {
    return m_piece_to_id.find(piece) != m_piece_to_id.end();
}

int32_t Tokenizer::piece_to_id(const std::string &piece) const {
    auto it = m_piece_to_id.find(piece);
    return (it != m_piece_to_id.end()) ? it->second : -1;
}

std::vector<int32_t> Tokenizer::encode(const std::string &text, bool add_bos) const {
    std::vector<int32_t> tokens;
    if (add_bos && m_bos_id >= 0) {
        tokens.push_back(m_bos_id);
    }

    if (text.empty() || m_id_to_piece.empty()) {
        return tokens;
    }

    std::string processed;
    if (m_uses_gpt2_space) {
        for (char c : text) {
            if (c == ' ') processed += "\xC4\xA0"; // "Ġ"
            else processed += c;
        }
    } else if (m_uses_spm_space) {
        for (char c : text) {
            if (c == ' ') processed += "\xE2\x96\x81"; // " "
            else processed += c;
        }
    } else {
        processed = text;
    }

    // Subword greedy longest prefix match
    size_t i = 0;
    while (i < processed.size()) {
        bool matched = false;
        size_t max_search = std::min(size_t(64), processed.size() - i);
        for (size_t len = max_search; len >= 1; --len) {
            std::string sub = processed.substr(i, len);
            auto it = m_piece_to_id.find(sub);
            if (it != m_piece_to_id.end()) {
                tokens.push_back(it->second);
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            i++; // Skip unknown byte
        }
    }

    return tokens;
}

std::string Tokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || static_cast<size_t>(token_id) >= m_id_to_piece.size()) {
        return "";
    }
    const std::string &raw = m_id_to_piece[token_id];

    // Replace Ġ and   with standard space
    std::string clean;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (i + 1 < raw.size() && static_cast<unsigned char>(raw[i]) == 0xC4 && static_cast<unsigned char>(raw[i+1]) == 0xA0) {
            clean += ' ';
            i++;
        } else if (i + 2 < raw.size() && static_cast<unsigned char>(raw[i]) == 0xE2 &&
                   static_cast<unsigned char>(raw[i+1]) == 0x96 && static_cast<unsigned char>(raw[i+2]) == 0x81) {
            clean += ' ';
            i += 2;
        } else {
            clean += raw[i];
        }
    }
    return clean;
}

std::string Tokenizer::decode(const std::vector<int32_t> &tokens) const {
    std::string out;
    for (int32_t t : tokens) {
        out += decode_token(t);
    }
    return out;
}

std::string Tokenizer::format_chatml(const std::string &user_prompt, const std::string &system_prompt) {
    std::string out;
    if (!system_prompt.empty()) {
        out += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    }
    out += "<|im_start|>user\n" + user_prompt + "<|im_end|>\n";
    out += "<|im_start|>assistant\n";
    return out;
}

} // namespace eswin
