#include <iostream>
#include <fstream>
#include "base64.h"
#include "llm_tokenizer.hpp"

const std::string ChatMessage::ROLE_USER = "user";
const std::string ChatMessage::ROLE_ASSISTANT = "assistant";
const std::string ChatMessage::ROLE_SYSTEM = "system";

// Init tokenizer
QwenTokenizer::QwenTokenizer(const std::string &tiktokenPath, const std::vector<std::string> &specialTokens) {
    std::ifstream file(tiktokenPath);
    if (!file) {
        throw std::runtime_error("failed to open encoder file: " + tiktokenPath);
    }

    std::unordered_map<std::string, int> encoder;
    std::string line;
    while (std::getline(file, line)) {
        auto [token, rank] = parseTiktoken(line);

        if (!encoder.emplace(std::move(token), rank).second) {
            throw std::runtime_error("duplicate item: " + line);
        }
    }

    std::vector<std::string> special_tokens_s(specialTokens);
    char buffer[14];
    for (size_t i = 0; i < 205; i++) { // 205 for extra control token
        snprintf(buffer, 14, "<|extra_%zu|>", i);
        special_tokens_s.push_back(buffer);
    }

    size_t encoder_size = encoder.size();
    std::unordered_map<std::string, int> special_tokens;
    special_tokens.reserve(special_tokens_s.size());
    for (size_t i = 0; i < special_tokens_s.size(); i++) {
        special_tokens[special_tokens_s[i]] = encoder_size + i;
    }

    mTokenizer = tiktoken::tiktoken(std::move(encoder), special_tokens, PAT_STR);
}

auto QwenTokenizer::encode(const std::string &text, int maxLength) const -> std::vector<int> {
    // auto ids = mTokenizer.encode_ordinary(text);
    auto ids = mTokenizer.encode(text);
    if ((int)ids.size() > maxLength) {
        ids.erase(ids.begin(), ids.end() - maxLength);
    }
    return ids;
}

auto QwenTokenizer::decode(const std::vector<int> &ids) const -> std::string {
    std::vector<int> normal_ids(ids);
    // normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id) { return is_special_id(id); }),
    //                  normal_ids.end());
    auto text = mTokenizer.decode(normal_ids);
    return text;
}

auto QwenTokenizer::isSpecialId(int id) const -> bool {
    return id == mEosTokenId || id == mImStartId || id == mImEndId;
}

std::pair<std::string, int> QwenTokenizer::parseTiktoken(const std::string &line) {
    auto pos = line.find(" ");
    if (pos == std::string::npos) {
        throw std::runtime_error("invalid encoder line: " + line);
    }

    auto token = base64::decode({line.data(), pos});
    int rank = 0;
    try {
        rank = std::stoul(line.substr(pos + 1));
    } catch (const std::exception &) {
        throw std::runtime_error("invalid encoder rank: " + line);
    }

    return {std::move(token), rank};
}

std::vector<int> QwenTokenizer::encode_messages(const std::vector<ChatMessage> &messages, int max_length) const {
    std::string prompt = build_prompt(messages);
    std::vector<int> input_ids = encode(prompt, max_length);
    return input_ids;
}

std::string QwenTokenizer::build_prompt(const std::vector<ChatMessage> &messages) {

    // check_chat_messages(messages);

    std::ostringstream oss_prompt;

    // chatml template example
    // <|im_start|>system
    // You are a helpful assistant.<|im_end|>
    // <|im_start|>user
    // 帮我介绍一下上海迪士尼的游玩攻略<|im_end|>
    // <|im_start|>assistant

    oss_prompt << "<|im_start|>system\n"
               << messages.front().content << "<|im_end|>"; // apply system. we have default system prompt.

    for (size_t i = 1; i < messages.size(); i += 1) { // get all history messages
        if (messages[i].role == ChatMessage::ROLE_USER) {
            oss_prompt << "\n<|im_start|>user\n" << messages[i].content << "<|im_end|>\n<|im_start|>assistant\n";
        } else {
            oss_prompt << messages[i].content << "<|im_end|>";
        }
    }

    // std::cout << oss_prompt.str()<<std::endl; //debug

    return oss_prompt.str();
}

DeepSeekTokenizer::DeepSeekTokenizer(const std::string &tiktoken_path, const std::vector<std::string> &specialTokens)
    : QwenTokenizer(tiktoken_path, specialTokens) {
    std::ifstream file(tiktoken_path);
    if (!file) {
        throw std::runtime_error("failed to open encoder file: " + tiktoken_path);
    }

    std::unordered_map<std::string, int> encoder;
    std::string line;
    while (std::getline(file, line)) {
        auto [token, rank] = parseTiktoken(line);

        if (!encoder.emplace(std::move(token), rank).second) {
            throw std::runtime_error("duplicate item: " + line);
        }
    }

    std::vector<std::string> special_tokens_s(specialTokens);

    char buffer[31];
    for (size_t i = 5; i < 251; i++) {
        snprintf(buffer, 31, "<|reserved_special_token_%zu|>", i);
        special_tokens_s.push_back(buffer);
    }

    size_t encoder_size = encoder.size();
    std::unordered_map<std::string, int> special_tokens;
    special_tokens.reserve(special_tokens_s.size());
    for (size_t i = 0; i < special_tokens_s.size(); i++) {
        special_tokens[special_tokens_s[i]] = encoder_size + i;
    }

    mTokenizer = tiktoken::tiktoken(std::move(encoder), special_tokens, PAT_STR);
}

std::vector<int> DeepSeekTokenizer::encode_messages(const std::vector<ChatMessage> &messages, int max_length) const {
    std::string prompt = build_prompt(messages);
    std::vector<int> input_ids = encode(prompt, max_length);
    return input_ids;
}

std::string DeepSeekTokenizer::build_prompt(const std::vector<ChatMessage> &messages) {

    // check_chat_messages(messages);

    std::ostringstream oss_prompt;

    // deepseek_r1_distill_qwen template example
    // <｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>

    // You are a helpful assistant.

    // <｜User｜>帮我介绍一下上海迪士尼的游玩攻略<｜Assistant｜>

    oss_prompt << "<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>" << messages.front().content;

    for (size_t i = 1; i < messages.size(); i += 1) {
        if (messages[i].role == ChatMessage::ROLE_USER) { // user
            oss_prompt << "<｜User｜>" << messages[i].content << "<｜Assistant｜>";
        } else {
            oss_prompt << messages[i].content << "";
        }
    }

    // std::cout << oss_prompt.str()<<std::endl; //debug

    return oss_prompt.str();
}