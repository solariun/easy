#include "easyai/tool.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

namespace easyai {

// --------------------------------------------------------------------------
// Tool::Builder::build  —  emit a JSON-schema object from the typed params.
// --------------------------------------------------------------------------
static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

Tool Tool::Builder::build() const {
    std::ostringstream js;
    js << "{\"type\":\"object\",\"properties\":{";
    bool first = true;
    for (const auto & p : params_) {
        if (!first) js << ',';
        first = false;
        js << '"' << json_escape(p.name) << "\":{\"type\":\""
           << json_escape(p.type) << "\"";
        if (!p.description.empty()) {
            js << ",\"description\":\"" << json_escape(p.description) << "\"";
        }
        js << '}';
    }
    js << "},\"required\":[";
    first = true;
    for (const auto & p : params_) {
        if (!p.required) continue;
        if (!first) js << ',';
        first = false;
        js << '"' << json_escape(p.name) << '"';
    }
    js << "]}";

    return Tool::make(name_, desc_, js.str(), handler_);
}

// --------------------------------------------------------------------------
// Tiny JSON arg getters — good enough for flat object payloads from LLMs.
// They do NOT implement a full JSON parser; they look for `"key"` then walk
// past whitespace + ':' and read a string / number / bool literal.
// --------------------------------------------------------------------------
namespace {

size_t find_key(const std::string & j, const std::string & key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = j.find(needle, pos)) != std::string::npos) {
        // ensure we matched a real key (preceded by '{' or ',' modulo ws)
        size_t b = pos;
        while (b > 0 && std::isspace(static_cast<unsigned char>(j[b - 1]))) --b;
        if (b == 0 || j[b - 1] == '{' || j[b - 1] == ',') {
            size_t after = pos + needle.size();
            while (after < j.size() &&
                   std::isspace(static_cast<unsigned char>(j[after]))) ++after;
            if (after < j.size() && j[after] == ':') return after + 1;
        }
        pos += needle.size();
    }
    return std::string::npos;
}

size_t skip_ws(const std::string & j, size_t i) {
    while (i < j.size() && std::isspace(static_cast<unsigned char>(j[i]))) ++i;
    return i;
}

bool read_json_string(const std::string & j, size_t i, std::string & out) {
    i = skip_ws(j, i);
    if (i >= j.size() || j[i] != '"') return false;
    ++i;
    out.clear();
    while (i < j.size()) {
        char c = j[i++];
        if (c == '"') return true;
        if (c == '\\' && i < j.size()) {
            char esc = j[i++];
            switch (esc) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'u': {
                    if (i + 4 > j.size()) return false;
                    std::string hex = j.substr(i, 4);
                    i += 4;
                    unsigned code = std::strtoul(hex.c_str(), nullptr, 16);
                    if (code < 0x80) {
                        out += static_cast<char>(code);
                    } else if (code < 0x800) {
                        out += static_cast<char>(0xC0 | (code >> 6));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (code >> 12));
                        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    }
                    break;
                }
                default: out += esc;
            }
        } else {
            out += c;
        }
    }
    return false;
}

}  // namespace

namespace args {

bool get_string(const std::string & json, const std::string & key, std::string & out) {
    size_t i = find_key(json, key);
    if (i == std::string::npos) return false;
    return read_json_string(json, i, out);
}

bool get_int(const std::string & json, const std::string & key, long long & out) {
    size_t i = find_key(json, key);
    if (i == std::string::npos) return false;
    i = skip_ws(json, i);
    char * end = nullptr;
    out = std::strtoll(json.c_str() + i, &end, 10);
    return end != json.c_str() + i;
}

bool get_double(const std::string & json, const std::string & key, double & out) {
    size_t i = find_key(json, key);
    if (i == std::string::npos) return false;
    i = skip_ws(json, i);
    char * end = nullptr;
    out = std::strtod(json.c_str() + i, &end);
    return end != json.c_str() + i;
}

bool get_bool(const std::string & json, const std::string & key, bool & out) {
    size_t i = find_key(json, key);
    if (i == std::string::npos) return false;
    i = skip_ws(json, i);
    if (json.compare(i, 4, "true")  == 0) { out = true;  return true; }
    if (json.compare(i, 5, "false") == 0) { out = false; return true; }
    return false;
}

}  // namespace args
}  // namespace easyai
