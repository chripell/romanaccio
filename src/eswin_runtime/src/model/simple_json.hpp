#ifndef ESWIN_SIMPLE_JSON_HPP
#define ESWIN_SIMPLE_JSON_HPP

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cctype>
#include <cstdint>

namespace eswin {

class JsonValue {
public:
    enum class Type {
        Null,
        Boolean,
        Number,
        String,
        Array,
        Object
    };

    JsonValue() : m_type(Type::Null), m_bool(false), m_num(0.0) {}
    JsonValue(bool b) : m_type(Type::Boolean), m_bool(b), m_num(0.0) {}
    JsonValue(int n) : m_type(Type::Number), m_bool(false), m_num(n) {}
    JsonValue(int64_t n) : m_type(Type::Number), m_bool(false), m_num(static_cast<double>(n)) {}
    JsonValue(uint64_t n) : m_type(Type::Number), m_bool(false), m_num(static_cast<double>(n)) {}
    JsonValue(double d) : m_type(Type::Number), m_bool(false), m_num(d) {}
    JsonValue(const char *s) : m_type(Type::String), m_bool(false), m_num(0.0), m_str(s) {}
    JsonValue(const std::string &s) : m_type(Type::String), m_bool(false), m_num(0.0), m_str(s) {}
    JsonValue(const std::vector<JsonValue> &arr) : m_type(Type::Array), m_bool(false), m_num(0.0), m_arr(arr) {}
    JsonValue(const std::map<std::string, JsonValue> &obj) : m_type(Type::Object), m_bool(false), m_num(0.0), m_obj(obj) {}

    Type type() const { return m_type; }
    bool is_null() const { return m_type == Type::Null; }
    bool is_bool() const { return m_type == Type::Boolean; }
    bool is_number() const { return m_type == Type::Number; }
    bool is_string() const { return m_type == Type::String; }
    bool is_array() const { return m_type == Type::Array; }
    bool is_object() const { return m_type == Type::Object; }

    bool as_bool(bool def = false) const { return is_bool() ? m_bool : def; }
    int64_t as_int64(int64_t def = 0) const { return is_number() ? static_cast<int64_t>(m_num) : def; }
    int as_int(int def = 0) const { return is_number() ? static_cast<int>(m_num) : def; }
    double as_double(double def = 0.0) const { return is_number() ? m_num : def; }
    std::string as_string(const std::string &def = "") const { return is_string() ? m_str : def; }

    const std::vector<JsonValue>& as_array() const { return m_arr; }
    std::vector<JsonValue>& as_array() { return m_arr; }

    const std::map<std::string, JsonValue>& as_object() const { return m_obj; }
    std::map<std::string, JsonValue>& as_object() { return m_obj; }

    bool has(const std::string &key) const {
        return is_object() && m_obj.find(key) != m_obj.end();
    }

    const JsonValue& operator[](const std::string &key) const {
        static const JsonValue s_null;
        if (!is_object()) return s_null;
        auto it = m_obj.find(key);
        return (it != m_obj.end()) ? it->second : s_null;
    }

    JsonValue& operator[](const std::string &key) {
        if (!is_object()) {
            m_type = Type::Object;
            m_obj.clear();
        }
        return m_obj[key];
    }

    const JsonValue& operator[](size_t index) const {
        static const JsonValue s_null;
        if (!is_array() || index >= m_arr.size()) return s_null;
        return m_arr[index];
    }

    JsonValue& operator[](size_t index) {
        if (!is_array()) {
            m_type = Type::Array;
            m_arr.clear();
        }
        if (index >= m_arr.size()) m_arr.resize(index + 1);
        return m_arr[index];
    }

    void push_back(const JsonValue &val) {
        if (!is_array()) {
            m_type = Type::Array;
            m_arr.clear();
        }
        m_arr.push_back(val);
    }

    std::string dump(int indent = 0, int current_indent = 0) const {
        std::string ind(current_indent, ' ');
        std::string next_ind(current_indent + (indent > 0 ? indent : 0), ' ');

        switch (m_type) {
            case Type::Null: return "null";
            case Type::Boolean: return m_bool ? "true" : "false";
            case Type::Number: {
                if (m_num == static_cast<int64_t>(m_num)) {
                    return std::to_string(static_cast<int64_t>(m_num));
                }
                std::ostringstream ss;
                ss << m_num;
                return ss.str();
            }
            case Type::String: {
                std::string s = "\"";
                for (char c : m_str) {
                    if (c == '"') s += "\\\"";
                    else if (c == '\\') s += "\\\\";
                    else if (c == '\b') s += "\\b";
                    else if (c == '\f') s += "\\f";
                    else if (c == '\n') s += "\\n";
                    else if (c == '\r') s += "\\r";
                    else if (c == '\t') s += "\\t";
                    else s += c;
                }
                s += "\"";
                return s;
            }
            case Type::Array: {
                if (m_arr.empty()) return "[]";
                std::string s = "[";
                if (indent > 0) s += "\n";
                for (size_t i = 0; i < m_arr.size(); ++i) {
                    if (indent > 0) s += next_ind;
                    s += m_arr[i].dump(indent, current_indent + indent);
                    if (i + 1 < m_arr.size()) s += ",";
                    if (indent > 0) s += "\n";
                }
                if (indent > 0) s += ind;
                s += "]";
                return s;
            }
            case Type::Object: {
                if (m_obj.empty()) return "{}";
                std::string s = "{";
                if (indent > 0) s += "\n";
                size_t count = 0;
                for (const auto &pair : m_obj) {
                    if (indent > 0) s += next_ind;
                    s += "\"" + pair.first + "\": ";
                    s += pair.second.dump(indent, current_indent + indent);
                    if (++count < m_obj.size()) s += ",";
                    if (indent > 0) s += "\n";
                }
                if (indent > 0) s += ind;
                s += "}";
                return s;
            }
        }
        return "null";
    }

    static JsonValue parse(const std::string &src, std::string &err) {
        size_t pos = 0;
        skip_whitespace(src, pos);
        try {
            JsonValue v = parse_value(src, pos);
            skip_whitespace(src, pos);
            if (pos < src.size()) {
                err = "Extra characters after JSON payload at position " + std::to_string(pos);
            }
            return v;
        } catch (const std::exception &e) {
            err = e.what();
            return JsonValue();
        }
    }

private:
    static void skip_whitespace(const std::string &s, size_t &pos) {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) {
            pos++;
        }
    }

    static JsonValue parse_value(const std::string &s, size_t &pos) {
        skip_whitespace(s, pos);
        if (pos >= s.size()) throw std::runtime_error("Unexpected end of input");

        char c = s[pos];
        if (c == '{') return parse_object(s, pos);
        if (c == '[') return parse_array(s, pos);
        if (c == '"') return parse_string(s, pos);
        if (c == 't' || c == 'f') return parse_boolean(s, pos);
        if (c == 'n') return parse_null(s, pos);
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number(s, pos);

        throw std::runtime_error(std::string("Unexpected token '") + c + "' at position " + std::to_string(pos));
    }

    static JsonValue parse_object(const std::string &s, size_t &pos) {
        pos++; // skip '{'
        std::map<std::string, JsonValue> obj;
        skip_whitespace(s, pos);
        if (pos < s.size() && s[pos] == '}') {
            pos++;
            return JsonValue(obj);
        }

        while (pos < s.size()) {
            skip_whitespace(s, pos);
            if (pos >= s.size() || s[pos] != '"') throw std::runtime_error("Expected string key in object at position " + std::to_string(pos));
            std::string key = parse_string(s, pos).as_string();
            skip_whitespace(s, pos);
            if (pos >= s.size() || s[pos] != ':') throw std::runtime_error("Expected ':' after key at position " + std::to_string(pos));
            pos++; // skip ':'
            JsonValue val = parse_value(s, pos);
            obj[key] = val;

            skip_whitespace(s, pos);
            if (pos < s.size() && s[pos] == ',') {
                pos++;
            } else if (pos < s.size() && s[pos] == '}') {
                pos++;
                break;
            } else {
                throw std::runtime_error("Expected ',' or '}' in object at position " + std::to_string(pos));
            }
        }
        return JsonValue(obj);
    }

    static JsonValue parse_array(const std::string &s, size_t &pos) {
        pos++; // skip '['
        std::vector<JsonValue> arr;
        skip_whitespace(s, pos);
        if (pos < s.size() && s[pos] == ']') {
            pos++;
            return JsonValue(arr);
        }

        while (pos < s.size()) {
            arr.push_back(parse_value(s, pos));
            skip_whitespace(s, pos);
            if (pos < s.size() && s[pos] == ',') {
                pos++;
            } else if (pos < s.size() && s[pos] == ']') {
                pos++;
                break;
            } else {
                throw std::runtime_error("Expected ',' or ']' in array at position " + std::to_string(pos));
            }
        }
        return JsonValue(arr);
    }

    static JsonValue parse_string(const std::string &s, size_t &pos) {
        pos++; // skip opening '"'
        std::string res;
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '"') return JsonValue(res);
            if (c == '\\') {
                if (pos >= s.size()) throw std::runtime_error("Unterminated escape sequence in string");
                char esc = s[pos++];
                switch (esc) {
                    case '"': res += '"'; break;
                    case '\\': res += '\\'; break;
                    case '/': res += '/'; break;
                    case 'b': res += '\b'; break;
                    case 'f': res += '\f'; break;
                    case 'n': res += '\n'; break;
                    case 'r': res += '\r'; break;
                    case 't': res += '\t'; break;
                    case 'u': {
                        if (pos + 4 > s.size()) throw std::runtime_error("Incomplete unicode escape in string");
                        pos += 4; // skip 4 hex digits for now
                        res += '?';
                        break;
                    }
                    default: res += esc; break;
                }
            } else {
                res += c;
            }
        }
        throw std::runtime_error("Unterminated string");
    }

    static JsonValue parse_boolean(const std::string &s, size_t &pos) {
        if (s.compare(pos, 4, "true") == 0) {
            pos += 4;
            return JsonValue(true);
        }
        if (s.compare(pos, 5, "false") == 0) {
            pos += 5;
            return JsonValue(false);
        }
        throw std::runtime_error("Invalid boolean at position " + std::to_string(pos));
    }

    static JsonValue parse_null(const std::string &s, size_t &pos) {
        if (s.compare(pos, 4, "null") == 0) {
            pos += 4;
            return JsonValue();
        }
        throw std::runtime_error("Invalid null at position " + std::to_string(pos));
    }

    static JsonValue parse_number(const std::string &s, size_t &pos) {
        size_t start = pos;
        if (s[pos] == '-') pos++;
        while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) pos++;
        if (pos < s.size() && s[pos] == '.') {
            pos++;
            while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) pos++;
        }
        if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
            pos++;
            if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) pos++;
            while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) pos++;
        }
        std::string num_str = s.substr(start, pos - start);
        double val = std::stod(num_str);
        return JsonValue(val);
    }

    Type m_type;
    bool m_bool;
    double m_num;
    std::string m_str;
    std::vector<JsonValue> m_arr;
    std::map<std::string, JsonValue> m_obj;
};

} // namespace eswin

#endif // ESWIN_SIMPLE_JSON_HPP
