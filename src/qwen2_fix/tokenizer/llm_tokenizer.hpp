#include <string>
#include <vector>
#include <iomanip>
#include "tiktoken.h"

static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

struct ChatMessage {
    std::string role;
    std::string content;

    static const std::string ROLE_USER;
    static const std::string ROLE_ASSISTANT;
    static const std::string ROLE_SYSTEM;

    ChatMessage() = default;
    ChatMessage(std::string role, std::string content) : role(std::move(role)), content(std::move(content)) {}

    friend std::ostream &operator<<(std::ostream &os, const ChatMessage &self) {
        os << "ChatMessage(role=" << std::quoted(self.role) << ", content=" << std::quoted(self.content);
        return os << ")";
    }
};

class QwenTokenizer
{
public:
    QwenTokenizer(const std::string &tiktokenPath, const std::vector<std::string> &specialTokens);

    auto encode(const std::string &text, int maxLength) const -> std::vector<int>;

    auto decode(const std::vector<int> &ids) const -> std::string;

    virtual std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const;

    auto isSpecialId(int id) const -> bool;

    static std::string build_prompt(const std::vector<ChatMessage> &messages);

    tiktoken::tiktoken mTokenizer;
    int mEosTokenId;
    int mImStartId;
    int mImEndId;

private:

protected:
    std::pair<std::string, int> parseTiktoken(const std::string &line);
};

class DeepSeekTokenizer : public QwenTokenizer {
  public:
    DeepSeekTokenizer(const std::string &tiktoken_path, const std::vector<std::string> &specialTokens);

    std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const;

    static std::string build_prompt(const std::vector<ChatMessage> &messages);
};