#ifndef ESWIN_TOKENIZER_HPP
#define ESWIN_TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace eswin {

class Tokenizer {
public:
    Tokenizer() = default;

    // Load from Hugging Face tokenizer.json, TikToken file, or binary vocab
    bool load(const std::string &filepath, std::string &err);
    bool load_json(const std::string &json_str, std::string &err);
    bool load_binary(const std::string &bin_path, std::string &err);
    bool load_tiktoken(const std::string &tiktoken_path, std::string &err);

    // Encoding
    std::vector<int32_t> encode(const std::string &text, bool add_bos = false) const;

    // Decoding
    std::string decode(const std::vector<int32_t> &tokens) const;
    std::string decode_token(int32_t token_id) const;

    // Special token IDs
    int32_t bos_token_id() const { return m_bos_id; }
    int32_t eos_token_id() const { return m_eos_id; }
    int32_t pad_token_id() const { return m_pad_id; }
    void set_special_tokens(int32_t bos, int32_t eos, int32_t pad = -1);

    // Vocabulary properties
    size_t vocab_size() const { return m_id_to_piece.size(); }
    bool contains(const std::string &piece) const;
    int32_t piece_to_id(const std::string &piece) const;

    // Chat templating helper (ChatML formatting)
    static std::string format_chatml(
        const std::string &user_prompt,
        const std::string &system_prompt = "You are a helpful AI assistant."
    );

private:
    void add_token(int32_t id, const std::string &piece);

    std::vector<std::string>             m_id_to_piece;
    std::unordered_map<std::string, int> m_piece_to_id;
    int32_t                              m_bos_id = 0;
    int32_t                              m_eos_id = 0;
    int32_t                              m_pad_id = -1;
    bool                                 m_uses_gpt2_space = false;
    bool                                 m_uses_spm_space = false;
};

} // namespace eswin

#endif // ESWIN_TOKENIZER_HPP
