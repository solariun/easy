#include "easyai/engine.hpp"
#include "easyai/tool.hpp"        // easyai::args::get_string for tool_call recovery

#include "common.h"
#include "sampling.h"
#include "chat.h"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace easyai {

// ===========================================================================
// Tool-call recovery helpers
// ---------------------------------------------------------------------------
// Some models (notably Qwen2.5-Instruct in its current GGUF chat-template
// builds) emit tool calls with DOUBLED braces:
//
//     <tool_call>
//     {{"name":"web_search","arguments":{"query":"hello"}}}
//     </tool_call>
//
// The Jinja chat-template's example uses `{{ ... }}` for variable
// interpolation, and the template itself isn't escaping them when rendering
// the tool-call demo, so the model imitates the literal form.  The PEG
// parser in llama.cpp (correctly) refuses that JSON, falls through, and we
// end up with no tool_calls at all even though the user clearly asked for
// one.
//
// These helpers recover those calls by hand: we scan the raw output for
// <tool_call>...</tool_call> blocks and pull `name` + `arguments` out of
// each, tolerant of extra wrapping braces.
// ===========================================================================
namespace {

// Walk a JSON object starting at `s[i]` (which must be '{'), respecting
// strings.  Returns the index ONE PAST the matching '}', or npos on failure.
size_t walk_balanced_braces(const std::string & s, size_t i) {
    if (i >= s.size() || s[i] != '{') return std::string::npos;
    int  depth = 0;
    bool in_str = false, esc = false;
    for (; i < s.size(); ++i) {
        char c = s[i];
        if (in_str) {
            if (esc)             esc = false;
            else if (c == '\\')  esc = true;
            else if (c == '"')   in_str = false;
            continue;
        }
        if      (c == '"') in_str = true;
        else if (c == '{') ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0) return i + 1;
        }
    }
    return std::string::npos;
}

// Scan `raw` for <tool_call>...</tool_call> blocks (terminated or
// unterminated) and return whatever {name, arguments} pairs we can recover.
std::vector<common_chat_tool_call> recover_qwen_tool_calls(const std::string & raw) {
    std::vector<common_chat_tool_call> out;
    static const std::string open_tag  = "<tool_call>";
    static const std::string close_tag = "</tool_call>";

    size_t pos = 0;
    while (true) {
        size_t a = raw.find(open_tag, pos);
        if (a == std::string::npos) break;
        size_t body_begin = a + open_tag.size();
        size_t b = raw.find(close_tag, body_begin);
        std::string body = (b == std::string::npos)
                               ? raw.substr(body_begin)
                               : raw.substr(body_begin, b - body_begin);

        // Pull out "name" — `args::get_string` does a forgiving top-level
        // key scan that already tolerates the doubled-brace wrapper.
        std::string name;
        if (!args::get_string(body, "name", name) || name.empty()) {
            pos = (b == std::string::npos) ? raw.size() : b + close_tag.size();
            continue;
        }

        // Pull out "arguments" — find the colon, then walk the JSON object.
        std::string arguments_json = "{}";
        size_t k = body.find("\"arguments\"");
        if (k != std::string::npos) {
            k = body.find(':', k);
            if (k != std::string::npos) {
                ++k;
                while (k < body.size() &&
                       std::isspace((unsigned char) body[k])) ++k;
                if (k < body.size() && body[k] == '{') {
                    size_t end = walk_balanced_braces(body, k);
                    if (end != std::string::npos) {
                        arguments_json = body.substr(k, end - k);
                    }
                }
            }
        }

        common_chat_tool_call tc;
        tc.name      = std::move(name);
        tc.arguments = std::move(arguments_json);
        out.push_back(std::move(tc));

        pos = (b == std::string::npos) ? raw.size() : b + close_tag.size();
    }
    return out;
}

// Drop any text inside <tool_call>...</tool_call> blocks (so the user
// doesn't see the raw JSON in the visible content after recovery).
std::string strip_tool_call_blocks(std::string s) {
    static const std::string open_tag  = "<tool_call>";
    static const std::string close_tag = "</tool_call>";
    size_t pos = 0;
    while ((pos = s.find(open_tag, pos)) != std::string::npos) {
        size_t end = s.find(close_tag, pos + open_tag.size());
        if (end == std::string::npos) { s.erase(pos); break; }
        s.erase(pos, end + close_tag.size() - pos);
    }
    return s;
}

// Hermes-style tool call recovery.  Some Qwen3 fine-tunes (notably the
// user's eng_v5 35B-A3) emit tool calls in this XML-ish format instead of
// the Qwen3 JSON one:
//
//     <tool_call>
//     <function=web_search>
//     <parameter=query>
//     top news today
//     </parameter>
//     <parameter=max_results>
//     10
//     </parameter>
//     </function>
//     </tool_call>
//
// The PEG parser refuses it and the whole block leaks into msg.content.
// We rebuild a {name, JSON-arguments} pair by scanning the inner XML.
// Numeric-looking parameter values are emitted as JSON numbers; everything
// else is emitted as JSON strings.
std::vector<common_chat_tool_call> recover_hermes_tool_calls(const std::string & raw) {
    std::vector<common_chat_tool_call> out;
    static const std::string open_tag  = "<tool_call>";
    static const std::string close_tag = "</tool_call>";
    static const std::string fn_open   = "<function=";
    static const std::string fn_close  = "</function>";
    static const std::string par_open  = "<parameter=";
    static const std::string par_close = "</parameter>";

    auto trim = [](std::string s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char) s[a])) ++a;
        while (b > a && std::isspace((unsigned char) s[b - 1])) --b;
        return s.substr(a, b - a);
    };
    auto looks_numeric = [](const std::string & s) {
        if (s.empty()) return false;
        size_t i = 0;
        if (s[0] == '-' || s[0] == '+') ++i;
        bool seen_digit = false, seen_dot = false;
        for (; i < s.size(); ++i) {
            char c = s[i];
            if (c >= '0' && c <= '9') { seen_digit = true; continue; }
            if (c == '.' && !seen_dot) { seen_dot = true; continue; }
            return false;
        }
        return seen_digit;
    };
    auto json_escape = [](const std::string & s) {
        std::string o; o.reserve(s.size() + 4);
        for (char c : s) {
            switch (c) {
                case '\\': o += "\\\\"; break;
                case '"':  o += "\\\""; break;
                case '\n': o += "\\n";  break;
                case '\r': o += "\\r";  break;
                case '\t': o += "\\t";  break;
                case '\b': o += "\\b";  break;
                case '\f': o += "\\f";  break;
                default:
                    if ((unsigned char) c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) c);
                        o += buf;
                    } else {
                        o += c;
                    }
            }
        }
        return o;
    };

    size_t pos = 0;
    while (true) {
        size_t a = raw.find(open_tag, pos);
        if (a == std::string::npos) break;
        size_t body_begin = a + open_tag.size();
        size_t b = raw.find(close_tag, body_begin);
        std::string body = (b == std::string::npos)
                               ? raw.substr(body_begin)
                               : raw.substr(body_begin, b - body_begin);

        // Find <function=NAME>...</function> inside the body.
        size_t fn_a = body.find(fn_open);
        if (fn_a == std::string::npos) {
            pos = (b == std::string::npos) ? raw.size() : b + close_tag.size();
            continue;
        }
        size_t name_begin = fn_a + fn_open.size();
        size_t name_end = body.find('>', name_begin);
        if (name_end == std::string::npos) {
            pos = (b == std::string::npos) ? raw.size() : b + close_tag.size();
            continue;
        }
        std::string name = trim(body.substr(name_begin, name_end - name_begin));
        if (name.empty()) {
            pos = (b == std::string::npos) ? raw.size() : b + close_tag.size();
            continue;
        }
        size_t fn_b = body.find(fn_close, name_end);
        std::string fn_body = (fn_b == std::string::npos)
                                  ? body.substr(name_end + 1)
                                  : body.substr(name_end + 1, fn_b - (name_end + 1));

        // Walk every <parameter=KEY>VAL</parameter> inside the function body.
        std::ostringstream args;
        args << "{";
        size_t p = 0;
        bool first = true;
        while (true) {
            size_t pa = fn_body.find(par_open, p);
            if (pa == std::string::npos) break;
            size_t key_begin = pa + par_open.size();
            size_t key_end = fn_body.find('>', key_begin);
            if (key_end == std::string::npos) break;
            std::string key = trim(fn_body.substr(key_begin, key_end - key_begin));
            size_t val_begin = key_end + 1;
            size_t pb = fn_body.find(par_close, val_begin);
            std::string val = (pb == std::string::npos)
                                  ? fn_body.substr(val_begin)
                                  : fn_body.substr(val_begin, pb - val_begin);
            val = trim(val);
            if (!first) args << ",";
            args << "\"" << json_escape(key) << "\":";
            if (looks_numeric(val))           args << val;
            else if (val == "true" || val == "false" || val == "null") args << val;
            else                              args << "\"" << json_escape(val) << "\"";
            first = false;
            p = (pb == std::string::npos) ? fn_body.size() : pb + par_close.size();
        }
        args << "}";

        common_chat_tool_call tc;
        tc.name      = std::move(name);
        tc.arguments = args.str();
        out.push_back(std::move(tc));

        pos = (b == std::string::npos) ? raw.size() : b + close_tag.size();
    }
    return out;
}

// Extract a <think>...</think> span out of `s`.  Returns reasoning text
// (without the wrapping tags); `s` is left with the reasoning span removed
// in-place so the remainder is safe to use as visible content.  Tolerates
// missing opener (Qwen3 prefills <think>) and unterminated </think>.
std::string extract_think_block(std::string & s) {
    static const std::string open_tag  = "<think>";
    static const std::string close_tag = "</think>";

    size_t close_pos = s.find(close_tag);
    if (close_pos == std::string::npos) {
        // No closer at all — nothing safe to extract; leave content alone.
        return {};
    }
    size_t open_pos  = s.find(open_tag);
    size_t reasoning_begin, span_begin;
    if (open_pos != std::string::npos && open_pos < close_pos) {
        // Both tags present — span = [<think> ... </think>].
        reasoning_begin = open_pos + open_tag.size();
        span_begin      = open_pos;
    } else {
        // Only </think> — Qwen3-style prefill where <think> is implicit.
        // Treat everything before </think> as reasoning.
        reasoning_begin = 0;
        span_begin      = 0;
    }
    std::string reasoning = s.substr(reasoning_begin, close_pos - reasoning_begin);
    s.erase(span_begin, close_pos + close_tag.size() - span_begin);
    // Trim leading whitespace/newlines that the closing tag left behind.
    size_t lead = 0;
    while (lead < s.size() && (s[lead] == '\n' || s[lead] == '\r' ||
                               s[lead] == ' ' || s[lead] == '\t')) ++lead;
    if (lead) s.erase(0, lead);
    // Trim around the reasoning too.
    size_t a = 0, b = reasoning.size();
    while (a < b && std::isspace((unsigned char) reasoning[a])) ++a;
    while (b > a && std::isspace((unsigned char) reasoning[b - 1])) --b;
    return reasoning.substr(a, b - a);
}

// Markdown-style fake tool-call recovery.
// ---------------------------------------------------------------------------
// Some Qwen3 fine-tunes, when they "lose confidence" in the <tool_call>
// XML format mid-conversation (typically after a few real tool calls,
// some of which errored), give up on the syntax and instead emit a
// *visual* indicator in markdown that mimics how chat UIs render tool
// invocations.  Observed shape:
//
//     *🔧 datetime*
//     *🔧 web_search(query="Hugging Face Daily Papers latest")*
//
// or with bold instead of italics: `**🔧 web_fetch(url="...")**`.
//
// The engine sees these as plain content with tool_calls=0, treats it as
// the final answer, and the user sees an empty-looking bubble.  We
// recover by scanning for the pattern, parsing the args, and re-emitting
// real common_chat_tool_call entries so the agentic loop continues.
//
// Heuristic — the marker must contain the wrench emoji (🔧, UTF-8
// F0 9F 94 A7) so we don't misfire on legitimate prose.
namespace {

bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

std::string trim_str(std::string s) {
    size_t a = 0, b = s.size();
    while (a < b && is_ws(s[a])) ++a;
    while (b > a && is_ws(s[b - 1])) --b;
    return s.substr(a, b - a);
}

// Convert a free-form `key=value, key="quoted", key=12` argument list
// into a JSON object string.  Tolerant of either single or double
// quotes, and of bare numeric / boolean values.
std::string kv_args_to_json(const std::string & args) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    size_t i = 0;
    while (i < args.size()) {
        while (i < args.size() && (is_ws(args[i]) || args[i] == ',')) ++i;
        if (i >= args.size()) break;
        // Key: identifier chars + dashes/dots.
        size_t k0 = i;
        while (i < args.size() &&
               (std::isalnum((unsigned char) args[i]) || args[i] == '_' ||
                args[i] == '-' || args[i] == '.')) ++i;
        if (i == k0) break;
        std::string key = args.substr(k0, i - k0);
        while (i < args.size() && is_ws(args[i])) ++i;
        if (i >= args.size() || args[i] != '=') break;
        ++i;  // past '='
        while (i < args.size() && is_ws(args[i])) ++i;
        if (i >= args.size()) break;
        // Value: quoted or bare.
        std::string raw_val;
        bool quoted = false;
        if (args[i] == '"' || args[i] == '\'') {
            char q = args[i++];
            std::string buf;
            while (i < args.size() && args[i] != q) {
                if (args[i] == '\\' && i + 1 < args.size()) {
                    buf += args[i + 1];
                    i += 2;
                    continue;
                }
                buf += args[i++];
            }
            if (i < args.size()) ++i;  // past closing quote
            raw_val = std::move(buf);
            quoted = true;
        } else {
            size_t v0 = i;
            while (i < args.size() && args[i] != ',' && !is_ws(args[i])) ++i;
            raw_val = args.substr(v0, i - v0);
        }
        if (!first) out << ",";
        first = false;
        // JSON escape the key.
        out << "\"";
        for (char c : key) {
            if (c == '"' || c == '\\') out << '\\';
            out << c;
        }
        out << "\":";
        // Decide value type: quoted -> string; bare numeric / bool / null -> as-is.
        if (!quoted) {
            // Numeric?
            bool numeric = !raw_val.empty();
            for (size_t j = 0; j < raw_val.size() && numeric; ++j) {
                char c = raw_val[j];
                if (j == 0 && (c == '-' || c == '+')) continue;
                if (c >= '0' && c <= '9') continue;
                if (c == '.' || c == 'e' || c == 'E') continue;
                numeric = false;
            }
            if (numeric)                                            { out << raw_val; continue; }
            if (raw_val == "true" || raw_val == "false" || raw_val == "null") {
                out << raw_val;
                continue;
            }
        }
        out << "\"";
        for (char c : raw_val) {
            if (c == '"' || c == '\\') out << '\\';
            else if (c == '\n')        { out << "\\n"; continue; }
            else if (c == '\r')        { out << "\\r"; continue; }
            else if (c == '\t')        { out << "\\t"; continue; }
            out << c;
        }
        out << "\"";
    }
    out << "}";
    return out.str();
}

// Find every `*🔧 NAME(...)*` (or `**🔧 NAME(...)**`) pattern in `text`.
// Returns the list of recovered tool_calls AND fills `out_marker_spans`
// with the [begin, end) byte ranges of the markers so the caller can
// strip them from the visible content.
struct MdMarkerSpan { size_t begin; size_t end; };
std::vector<common_chat_tool_call>
recover_markdown_tool_calls(const std::string & text,
                            std::vector<MdMarkerSpan> * out_marker_spans) {
    std::vector<common_chat_tool_call> out;
    // 🔧 (wrench, U+1F527) in UTF-8.
    static const char wrench[]   = "\xF0\x9F\x94\xA7";
    static const char hammer_w[] = "\xF0\x9F\x9B\xA0";  // 🛠 base; we accept variants
    size_t pos = 0;
    while (pos < text.size()) {
        // Find next '*' that may start a marker.
        size_t a = text.find('*', pos);
        if (a == std::string::npos) break;
        size_t star_begin = a;
        size_t i = a + 1;
        if (i < text.size() && text[i] == '*') ++i;  // **
        // Skip whitespace after the leading stars.
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
        // Check for the wrench emoji.
        bool has_emoji = false;
        if (i + 4 <= text.size() && std::memcmp(&text[i], wrench, 4) == 0) {
            i += 4;
            has_emoji = true;
        } else if (i + 4 <= text.size() && std::memcmp(&text[i], hammer_w, 4) == 0) {
            i += 4;
            has_emoji = true;
            // Skip optional VS16 (U+FE0F, EF B8 8F) emoji presentation selector.
            if (i + 3 <= text.size() &&
                (unsigned char) text[i]     == 0xEF &&
                (unsigned char) text[i + 1] == 0xB8 &&
                (unsigned char) text[i + 2] == 0x8F) i += 3;
        }
        if (!has_emoji) { pos = star_begin + 1; continue; }
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
        // Tool name: identifier chars.
        size_t n0 = i;
        while (i < text.size() &&
               (std::isalnum((unsigned char) text[i]) || text[i] == '_')) ++i;
        if (i == n0) { pos = star_begin + 1; continue; }
        std::string name = text.substr(n0, i - n0);
        // Optional `(args)` block — match parens with depth + quote awareness.
        std::string args_inner;
        bool got_args = false;
        if (i < text.size() && text[i] == '(') {
            size_t arg_begin = i + 1;
            int depth = 1;
            bool in_str = false;
            char q = 0;
            bool esc = false;
            size_t j = arg_begin;
            for (; j < text.size() && depth > 0; ++j) {
                char c = text[j];
                if (esc) { esc = false; continue; }
                if (in_str) {
                    if (c == '\\') esc = true;
                    else if (c == q) in_str = false;
                    continue;
                }
                if (c == '"' || c == '\'') { in_str = true; q = c; continue; }
                if (c == '(') ++depth;
                else if (c == ')') --depth;
            }
            if (depth == 0) {
                args_inner = text.substr(arg_begin, (j - 1) - arg_begin);
                i = j;
                got_args = true;
            } else {
                // Unbalanced parens — bail on this candidate.
                pos = star_begin + 1;
                continue;
            }
        }
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
        // Closing star(s).
        if (i < text.size() && text[i] == '*') {
            ++i;
            if (i < text.size() && text[i] == '*') ++i;
        } else {
            // No closing star — this may still be a one-line pattern at
            // end-of-stream; accept as long as we already captured a name.
        }
        size_t end = i;
        // Eat trailing newline so the content strips cleanly.
        if (end < text.size() && text[end] == '\n') ++end;

        common_chat_tool_call tc;
        tc.name      = std::move(name);
        tc.arguments = got_args ? kv_args_to_json(trim_str(args_inner)) : std::string("{}");
        out.push_back(std::move(tc));
        if (out_marker_spans) out_marker_spans->push_back({ star_begin, end });
        pos = end;
    }
    return out;
}

// Strip the spans returned by recover_markdown_tool_calls from `s`.
std::string strip_marker_spans(const std::string & s,
                               const std::vector<MdMarkerSpan> & spans) {
    if (spans.empty()) return s;
    std::string out;
    out.reserve(s.size());
    size_t cursor = 0;
    for (const auto & sp : spans) {
        if (sp.begin > cursor) out.append(s, cursor, sp.begin - cursor);
        cursor = sp.end;
    }
    if (cursor < s.size()) out.append(s, cursor, s.size() - cursor);
    return trim_str(out);
}

}  // namespace (markdown-recovery helpers)

}  // namespace (top-level helpers)

// ===========================================================================
// Engine::Impl
// ===========================================================================
struct Engine::Impl {
    common_params               params;
    common_init_result_ptr      init;
    common_chat_templates_ptr   templates;
    common_sampler            * sampler = nullptr;

    std::vector<common_chat_msg> history;
    std::vector<Tool>            tools;

    std::string  system_prompt;
    int          max_new_tokens = -1;
    bool         loaded         = false;
    bool         verbose        = false;
    bool         enable_thinking = true;   // sent to chat templates that use it
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool         parallel_tool_calls    = false;

    TokenCallback    on_token;
    ToolCallback     on_tool;
    HopResetCallback on_hop_reset;

    std::string last_error;
    std::string backend_summary;

    Impl() {
        // Sensible chat defaults overlaid on the llama.cpp baseline.
        params.n_ctx                 = 4096;
        params.n_batch               = 4096;
        params.n_predict             = -1;
        params.cpuparams.n_threads   = std::max(1, (int) std::thread::hardware_concurrency() / 2);
        params.warmup                = false;
        params.n_gpu_layers          = -1;  // auto
        params.sampling.temp         = 0.7f;
        params.sampling.top_p        = 0.95f;
        params.sampling.top_k        = 40;
        params.sampling.min_p        = 0.05f;
        params.sampling.penalty_repeat = 1.1f;
    }

    ~Impl() {
        if (sampler) {
            common_sampler_free(sampler);
            sampler = nullptr;
        }
        // common_init_result_ptr + common_chat_templates_ptr free themselves.
    }

    // -------- helpers ---------------------------------------------------
    llama_model   * model() const { return init ? init->model()   : nullptr; }
    llama_context * ctx()   const { return init ? init->context() : nullptr; }

    std::vector<common_chat_tool> chat_tools() const {
        std::vector<common_chat_tool> out;
        out.reserve(tools.size());
        for (const auto & t : tools) {
            out.push_back({ t.name, t.description, t.parameters_json });
        }
        return out;
    }

    // Build the rendered prompt for the *full* current history (incl. any
    // assistant/tool messages). add_generation_prompt asks for the assistant
    // turn header.
    common_chat_params render(bool add_generation_prompt) const {
        common_chat_templates_inputs in;
        in.messages              = history;
        in.add_generation_prompt = add_generation_prompt;
        in.use_jinja             = true;
        in.tools                 = chat_tools();
        in.tool_choice           = tool_choice;
        in.parallel_tool_calls   = parallel_tool_calls;
        in.enable_thinking       = enable_thinking;
        // Tell the template builder to wire reasoning extraction into the
        // PEG parser it produces.  Without this the parser leaves <think>
        // content inside msg.content, so the streaming code can't split
        // reasoning_content from content.  AUTO maps to DEEPSEEK (the
        // canonical "extract <think>...</think> blocks") for every
        // template that supports thinking.
        in.reasoning_format      = COMMON_REASONING_FORMAT_AUTO;
        return common_chat_templates_apply(templates.get(), in);
    }

    bool feed_prompt(const std::string & prompt, int & n_past_inout) {
        // Tokenize and decode the new prompt span past whatever is already
        // in the KV cache (n_past tokens for sequence 0).
        const llama_vocab * vocab = llama_model_get_vocab(model());
        std::vector<llama_token> toks =
            common_tokenize(vocab, prompt, /*add_special=*/n_past_inout == 0,
                            /*parse_special=*/true);

        if (toks.empty()) return true;

        const int n_ctx = llama_n_ctx(ctx());
        if (n_past_inout + (int) toks.size() > n_ctx) {
            last_error = "prompt does not fit context (need "
                         + std::to_string(n_past_inout + toks.size())
                         + ", have " + std::to_string(n_ctx) + ")";
            return false;
        }

        const int n_batch = params.n_batch > 0 ? params.n_batch : 512;
        for (size_t i = 0; i < toks.size(); i += n_batch) {
            int n = std::min<int>(n_batch, toks.size() - i);
            llama_batch b = llama_batch_get_one(toks.data() + i, n);
            if (llama_decode(ctx(), b) != 0) {
                last_error = "llama_decode failed while feeding prompt";
                return false;
            }
            n_past_inout += n;
        }
        return true;
    }

    // Generate tokens until EOG, tool-call grammar trigger, or max_new_tokens.
    // Returns the raw assistant text (may contain tool-call syntax).
    std::string generate_until_done(int & n_past_inout) {
        const llama_vocab * vocab = llama_model_get_vocab(model());

        std::string raw;
        int generated = 0;
        const int budget = max_new_tokens > 0
                               ? max_new_tokens
                               : params.n_predict > 0 ? params.n_predict : -1;

        const int n_ctx = llama_n_ctx(ctx());

        llama_token id = 0;
        while (true) {
            id = common_sampler_sample(sampler, ctx(), -1);
            common_sampler_accept(sampler, id, /*accept_grammar=*/true);

            if (llama_vocab_is_eog(vocab, id)) break;

            std::string piece = common_token_to_piece(ctx(), id, /*special=*/false);
            raw += piece;
            if (on_token) on_token(piece);

            ++generated;
            if (budget > 0 && generated >= budget) break;
            if (n_past_inout + 1 >= n_ctx) {
                if (verbose) std::fprintf(stderr, "[easyai] context full, stopping\n");
                break;
            }

            llama_batch b = llama_batch_get_one(&id, 1);
            if (llama_decode(ctx(), b) != 0) {
                last_error = "llama_decode failed during generation";
                break;
            }
            ++n_past_inout;
        }
        return raw;
    }

    // Parse model output into a structured chat message according to the
    // current chat template (handles native + PEG tool-call formats).
    // The PEG arena is shipped as a serialized string in chat_params.parser;
    // we load it into the parser_params before dispatching.
    common_chat_msg parse_assistant(const std::string & raw, const common_chat_params & p) {
        common_chat_parser_params pp(p);
        pp.parse_tool_calls = true;
        if (!p.parser.empty()) {
            try { pp.parser.load(p.parser); }
            catch (const std::exception & e) {
                if (verbose) std::fprintf(stderr,
                    "[easyai] failed to load chat parser arena: %s\n", e.what());
            }
        }
        // 1) Try the official parser first.
        common_chat_msg msg;
        bool parser_threw = false;
        try {
            msg = common_chat_parse(raw, /*is_partial=*/false, pp);
        } catch (const std::exception & e) {
            if (verbose) std::fprintf(stderr,
                "[easyai] chat parser failed (%s) — attempting recovery\n", e.what());
            parser_threw = true;
            msg.role    = "assistant";
            msg.content = raw;
        }

        // 2a) When the parser threw and dumped raw into content, we still
        //     want to keep reasoning separated from visible content so the
        //     streaming layer doesn't double-render the <think> block (once
        //     as reasoning_content_delta from the partial parses that DID
        //     succeed, and again as content_delta from the last-resort
        //     fallback that re-emits the engine's final text).
        if (parser_threw && msg.reasoning_content.empty()) {
            std::string r = extract_think_block(msg.content);
            if (!r.empty()) {
                msg.reasoning_content = std::move(r);
                if (verbose) std::fprintf(stderr,
                    "[easyai] split <think> block from raw fallback (reasoning=%zu, "
                    "content=%zu)\n", msg.reasoning_content.size(), msg.content.size());
            }
        }

        // 2b) Recovery pass — if the official parser produced no tool_calls
        //     but the raw output contains <tool_call> markers, the model
        //     most likely emitted Qwen-style doubled-brace JSON OR the
        //     Hermes-XML <function=name>/<parameter=key> shape that the PEG
        //     grammar refused.  Pull what we can out of it by hand.
        if (msg.tool_calls.empty() && raw.find("<tool_call>") != std::string::npos) {
            auto recovered = recover_qwen_tool_calls(raw);
            const char * recovery_kind = "qwen";
            if (recovered.empty() || recovered.front().arguments == "{}") {
                auto hermes = recover_hermes_tool_calls(raw);
                if (!hermes.empty()) { recovered = std::move(hermes); recovery_kind = "hermes"; }
            }
            if (!recovered.empty()) {
                msg.tool_calls = std::move(recovered);
                msg.content    = strip_tool_call_blocks(msg.content);
                if (verbose) std::fprintf(stderr,
                    "[easyai] recovered %zu tool call(s) from malformed output (%s)\n",
                    msg.tool_calls.size(), recovery_kind);
            }
        }

        // 2c) Markdown-marker recovery — when the model abandons the
        //     <tool_call> XML and instead writes `*🔧 toolname(args)*`
        //     as plain content (typically after a few real calls some
        //     of which errored), the engine would otherwise return the
        //     markers as the FINAL answer (tool_calls=0) and the chat
        //     loop terminates with the user seeing two italicised
        //     wrench lines as the "reply".  Recover the intent and
        //     surface real tool_calls so the agentic loop continues.
        if (msg.tool_calls.empty() && !msg.content.empty() &&
                msg.content.find("\xF0\x9F\x94\xA7") != std::string::npos) {
            std::vector<MdMarkerSpan> spans;
            auto recovered = recover_markdown_tool_calls(msg.content, &spans);
            if (!recovered.empty()) {
                msg.tool_calls = std::move(recovered);
                msg.content    = strip_marker_spans(msg.content, spans);
                if (verbose) std::fprintf(stderr,
                    "[easyai] recovered %zu tool call(s) from markdown markers "
                    "(model abandoned <tool_call> syntax — agentic loop continues)\n",
                    msg.tool_calls.size());
            }
        }
        return msg;
    }

    // Find the registered tool by name.
    const Tool * find_tool(const std::string & name) const {
        for (const auto & t : tools) if (t.name == name) return &t;
        return nullptr;
    }
};

// ===========================================================================
// Engine — public API
// ===========================================================================
Engine::Engine() : p_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;
Engine::Engine(Engine &&) noexcept = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

Engine & Engine::model(std::string path)        { p_->params.model.path = std::move(path); return *this; }
Engine & Engine::context(int n)                 { p_->params.n_ctx = n; if (p_->params.n_batch > n) p_->params.n_batch = n; return *this; }
Engine & Engine::batch(int n)                   { p_->params.n_batch = n; return *this; }
Engine & Engine::gpu_layers(int n)              { p_->params.n_gpu_layers = n; return *this; }
Engine & Engine::threads(int n)                 { p_->params.cpuparams.n_threads = n; p_->params.cpuparams_batch.n_threads = n; return *this; }
Engine & Engine::seed(uint32_t s)               { p_->params.sampling.seed = s; return *this; }
Engine & Engine::system(std::string s)          { p_->system_prompt = std::move(s); return *this; }
Engine & Engine::temperature(float t)           { p_->params.sampling.temp = t; return *this; }
Engine & Engine::top_p(float v)                 { p_->params.sampling.top_p = v; return *this; }
Engine & Engine::top_k(int v)                   { p_->params.sampling.top_k = v; return *this; }
Engine & Engine::min_p(float v)                 { p_->params.sampling.min_p = v; return *this; }
Engine & Engine::repeat_penalty(float v)        { p_->params.sampling.penalty_repeat = v; return *this; }
Engine & Engine::max_tokens(int n)              { p_->max_new_tokens = n; return *this; }
Engine & Engine::tool_choice_auto()             { p_->tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;     return *this; }
Engine & Engine::tool_choice_required()         { p_->tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED; return *this; }
Engine & Engine::tool_choice_none()             { p_->tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;     return *this; }
Engine & Engine::parallel_tool_calls(bool e)    { p_->parallel_tool_calls = e; return *this; }
Engine & Engine::verbose(bool v)                { p_->verbose = v; return *this; }

// ---------------------------------------------------------------------------
// KV cache & model overrides
// ---------------------------------------------------------------------------
namespace {

// Map a ggml_type_name() string ("f16", "q8_0", …) back to the enum.
// Returns GGML_TYPE_COUNT on miss (used as the "invalid" sentinel).
ggml_type ggml_type_from_name(const std::string & s) {
    // Restrict to types that llama.cpp's KV cache actually supports — F32,
    // F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL.  Anything else would
    // be silently ignored by ggml so we'd rather flag it explicitly.
    static const ggml_type allowed[] = {
        GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16,
        GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL,
    };
    for (auto t : allowed) {
        if (s == ggml_type_name(t)) return t;
    }
    return GGML_TYPE_COUNT;
}

}  // namespace

Engine & Engine::cache_type_k(const std::string & name) {
    ggml_type t = ggml_type_from_name(name);
    if (t == GGML_TYPE_COUNT) {
        p_->last_error = "cache_type_k: unsupported ggml type '" + name + "'";
    } else {
        p_->params.cache_type_k = t;
    }
    return *this;
}

Engine & Engine::cache_type_v(const std::string & name) {
    ggml_type t = ggml_type_from_name(name);
    if (t == GGML_TYPE_COUNT) {
        p_->last_error = "cache_type_v: unsupported ggml type '" + name + "'";
    } else {
        p_->params.cache_type_v = t;
    }
    return *this;
}

Engine & Engine::no_kv_offload(bool on) { p_->params.no_kv_offload = on; return *this; }
Engine & Engine::kv_unified  (bool on) { p_->params.kv_unified    = on; return *this; }

Engine & Engine::add_kv_override(const std::string & spec) {
    // Parse "key=type:value" — minimal but strict.
    auto eq = spec.find('=');
    if (eq == std::string::npos || eq == 0) {
        p_->last_error = "add_kv_override: missing '=' in '" + spec + "'";
        return *this;
    }
    std::string key  = spec.substr(0, eq);
    std::string rest = spec.substr(eq + 1);
    auto colon = rest.find(':');
    if (colon == std::string::npos) {
        p_->last_error = "add_kv_override: missing ':' in '" + spec + "' (expected key=type:value)";
        return *this;
    }
    std::string type  = rest.substr(0, colon);
    std::string value = rest.substr(colon + 1);
    if (key.size() >= sizeof(((llama_model_kv_override*)0)->key)) {
        p_->last_error = "add_kv_override: key too long (max 127 chars)";
        return *this;
    }

    llama_model_kv_override ov{};
    std::strncpy(ov.key, key.c_str(), sizeof(ov.key) - 1);
    ov.key[sizeof(ov.key) - 1] = '\0';

    if (type == "int" || type == "i") {
        ov.tag     = LLAMA_KV_OVERRIDE_TYPE_INT;
        ov.val_i64 = std::strtoll(value.c_str(), nullptr, 10);
    } else if (type == "float" || type == "f") {
        ov.tag     = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        ov.val_f64 = std::strtod(value.c_str(), nullptr);
    } else if (type == "bool" || type == "b") {
        ov.tag      = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        ov.val_bool = (value == "true" || value == "1" || value == "yes");
    } else if (type == "str" || type == "s") {
        ov.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (value.size() >= sizeof(ov.val_str)) value.resize(sizeof(ov.val_str) - 1);
        std::strncpy(ov.val_str, value.c_str(), sizeof(ov.val_str) - 1);
        ov.val_str[sizeof(ov.val_str) - 1] = '\0';
    } else {
        p_->last_error = "add_kv_override: unknown type '" + type
                          + "' (expected int|float|bool|str)";
        return *this;
    }

    // The vector must be passed to llama with a final empty-key sentinel; we
    // append the real entry now and add the sentinel at load() time.
    p_->params.kv_overrides.push_back(ov);
    return *this;
}

// ---------------------------------------------------------------------------
// Compute / memory knobs
// ---------------------------------------------------------------------------
Engine & Engine::flash_attn(bool on) {
    p_->params.flash_attn_type = on ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                    : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return *this;
}
Engine & Engine::use_mlock(bool on)        { p_->params.use_mlock = on; return *this; }
Engine & Engine::use_mmap (bool on)        { p_->params.use_mmap  = on; return *this; }
Engine & Engine::threads_batch(int n)      { p_->params.cpuparams_batch.n_threads = n; return *this; }

Engine & Engine::numa(const std::string & strategy) {
    if      (strategy == "" || strategy == "off" || strategy == "disabled")
        p_->params.numa = GGML_NUMA_STRATEGY_DISABLED;
    else if (strategy == "distribute") p_->params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
    else if (strategy == "isolate")    p_->params.numa = GGML_NUMA_STRATEGY_ISOLATE;
    else if (strategy == "numactl")    p_->params.numa = GGML_NUMA_STRATEGY_NUMACTL;
    else if (strategy == "mirror")     p_->params.numa = GGML_NUMA_STRATEGY_MIRROR;
    else p_->last_error = "numa: unknown strategy '" + strategy + "'";
    return *this;
}

// ---------------------------------------------------------------------------
// Reasoning toggle — propagated through chat-template rendering.
// ---------------------------------------------------------------------------
Engine & Engine::enable_thinking(bool on) {
    p_->enable_thinking = on;
    return *this;
}

Engine & Engine::add_tool(Tool t)               { p_->tools.push_back(std::move(t)); return *this; }
Engine & Engine::clear_tools()                  { p_->tools.clear(); return *this; }
Engine & Engine::on_token(TokenCallback cb)         { p_->on_token     = std::move(cb); return *this; }
Engine & Engine::on_tool(ToolCallback cb)           { p_->on_tool      = std::move(cb); return *this; }
Engine & Engine::on_hop_reset(HopResetCallback cb)  { p_->on_hop_reset = std::move(cb); return *this; }

bool Engine::load() {
    if (p_->loaded) return true;
    if (p_->params.model.path.empty()) {
        p_->last_error = "model path not set; call .model(\"path/to/file.gguf\") first";
        return false;
    }

    // quiet logs unless verbose
    if (!p_->verbose) {
        llama_log_set([](enum ggml_log_level lvl, const char * txt, void *) {
            if (lvl >= GGML_LOG_LEVEL_ERROR) std::fprintf(stderr, "%s", txt);
        }, nullptr);
    }
    ggml_backend_load_all();

    // common_init_from_params asserts that kv_overrides ends with an
    // empty-key sentinel; honour that contract.
    if (!p_->params.kv_overrides.empty() &&
        p_->params.kv_overrides.back().key[0] != '\0') {
        llama_model_kv_override term{};
        p_->params.kv_overrides.push_back(term);
    }

    p_->init = common_init_from_params(p_->params);
    if (!p_->init || !p_->init->model() || !p_->init->context()) {
        p_->last_error = "failed to load model: " + p_->params.model.path;
        return false;
    }

    p_->sampler = common_sampler_init(p_->init->model(), p_->params.sampling);
    if (!p_->sampler) {
        p_->last_error = "failed to initialize sampler";
        return false;
    }

    p_->templates = common_chat_templates_init(p_->init->model(), /*override=*/"");
    if (!p_->templates) {
        p_->last_error = "model has no usable chat template";
        return false;
    }

    if (!p_->system_prompt.empty()) {
        p_->history.push_back({ "system", p_->system_prompt, {}, {}, "", "", "" });
    }

    // ---- backend summary --------------------------------------------------
    {
        std::ostringstream s;
        const int n_dev = ggml_backend_dev_count();
        bool any = false;
        for (int i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            // Anything that's not the bare CPU device counts as an offload
            // backend (GPU, ACCEL, IPU, etc.). RADV+Vulkan in particular
            // sometimes reports GPU as ACCEL, which the strict GPU filter
            // missed and made the banner say "CPU" even when offloading.
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                if (any) s << ", ";
                s << ggml_backend_dev_name(dev);
                any = true;
            }
        }
        p_->backend_summary = any ? s.str() : "CPU";
    }

    p_->loaded = true;
    return true;
}

bool Engine::is_loaded() const { return p_->loaded; }

void Engine::reset() {
    p_->history.clear();
    if (!p_->system_prompt.empty()) {
        p_->history.push_back({ "system", p_->system_prompt, {}, {}, "", "", "" });
    }
    if (p_->ctx())     llama_memory_clear(llama_get_memory(p_->ctx()), true);
    if (p_->sampler)   common_sampler_reset(p_->sampler);
}

void Engine::clear_kv() {
    if (p_->ctx())   llama_memory_clear(llama_get_memory(p_->ctx()), true);
    if (p_->sampler) common_sampler_reset(p_->sampler);
}

void Engine::pop_last(size_t n) {
    while (n-- > 0 && !p_->history.empty()) {
        p_->history.pop_back();
    }
}

std::string Engine::generate() {
    if (!p_->loaded) { p_->last_error = "engine not loaded"; return {}; }

    auto chat_p = p_->render(/*add_generation_prompt=*/true);

    // Compute how many KV tokens are already cached so we only feed the new
    // tail. Sequence 0 is what we use; pos_max+1 is the next empty position.
    int n_past = llama_memory_seq_pos_max(llama_get_memory(p_->ctx()), 0) + 1;
    if (n_past < 0) n_past = 0;

    // For simplicity we re-render from scratch and feed whatever isn't yet
    // in cache. Tokenize the *full* rendered prompt and skip the prefix that
    // matches the existing KV. (Simple safe approach: tokenize all, decode
    // only the suffix beyond n_past.)
    const llama_vocab * vocab = llama_model_get_vocab(p_->init->model());
    std::vector<llama_token> all =
        common_tokenize(vocab, chat_p.prompt, /*add_special=*/n_past == 0,
                        /*parse_special=*/true);

    if ((int) all.size() < n_past) {
        // History was rolled back (reset/edit); restart from scratch.
        llama_memory_clear(llama_get_memory(p_->ctx()), true);
        n_past = 0;
    }
    std::vector<llama_token> tail(all.begin() + n_past, all.end());
    if (!tail.empty()) {
        const int n_ctx = llama_n_ctx(p_->ctx());
        if (n_past + (int) tail.size() > n_ctx) {
            p_->last_error = "prompt overflows context window";
            return {};
        }
        const int n_batch = p_->params.n_batch > 0 ? p_->params.n_batch : 512;
        for (size_t i = 0; i < tail.size(); i += n_batch) {
            int n = std::min<int>(n_batch, tail.size() - i);
            llama_batch b = llama_batch_get_one(tail.data() + i, n);
            if (llama_decode(p_->ctx(), b) != 0) {
                p_->last_error = "llama_decode failed feeding prompt";
                return {};
            }
            n_past += n;
        }
    }

    return p_->generate_until_done(n_past);
}

std::string Engine::chat(const std::string & user_message) {
    if (!p_->loaded) { p_->last_error = "engine not loaded"; return {}; }

    p_->history.push_back({ "user", user_message, {}, {}, "", "", "" });
    return chat_continue();
}

std::string Engine::chat_continue() {
    if (!p_->loaded) { p_->last_error = "engine not loaded"; return {}; }

    constexpr int kMaxToolHops      = 8;
    constexpr int kMaxThoughtRetries = 2;   // budget for "thought-only" retries
    int thought_retries = 0;
    std::string final_text;

    for (int hop = 0; hop < kMaxToolHops; ++hop) {
        auto chat_p = p_->render(/*add_generation_prompt=*/true);
        std::string raw = generate();
        if (!p_->last_error.empty() && raw.empty()) return {};

        common_chat_msg msg = p_->parse_assistant(raw, chat_p);
        msg.role = "assistant";

        if (p_->verbose) {
            std::fprintf(stderr,
                "[easyai] hop %d: raw=%zu content=%zu reasoning=%zu tool_calls=%zu\n",
                hop, raw.size(), msg.content.size(),
                msg.reasoning_content.size(), msg.tool_calls.size());
            if (!raw.empty()) {
                const size_t tail = std::min<size_t>(140, raw.size());
                std::fprintf(stderr, "[easyai] hop %d raw tail: %.*s\n",
                    hop, (int) tail, raw.c_str() + raw.size() - tail);
            }
        }

        // Detect "thought-only" turn — model emitted reasoning but produced
        // neither content nor tool_calls.  Some Qwen3 fine-tunes (the
        // user's eng_v5 in particular) terminate after </think> when they
        // intended to call a tool but failed to emit the tool_call header.
        // Discard the empty turn, clear KV, and retry — sampling is
        // stochastic so the second pass usually produces a real answer.
        const bool thought_only =
            msg.tool_calls.empty()
            && msg.content.empty()
            && !msg.reasoning_content.empty();

        if (thought_only && thought_retries < kMaxThoughtRetries
                         && hop + 1 < kMaxToolHops) {
            ++thought_retries;
            if (p_->verbose) std::fprintf(stderr,
                "[easyai] hop %d: thought-only turn — clearing KV and retrying (%d/%d)\n",
                hop, thought_retries, kMaxThoughtRetries);
            llama_memory_clear(llama_get_memory(p_->ctx()), true);
            if (p_->on_hop_reset) p_->on_hop_reset();
            continue;
        }

        // If we exhausted retries on a thought-only turn, fall back to
        // promoting reasoning_content into content so the user at least
        // sees the model's thoughts instead of an empty bubble.
        if (msg.tool_calls.empty() && msg.content.empty()
                && !msg.reasoning_content.empty()) {
            msg.content = msg.reasoning_content;
            if (p_->verbose) std::fprintf(stderr,
                "[easyai] hop %d: retry budget exhausted — promoting reasoning to content\n", hop);
            // Also push the synthesized text through on_token so the
            // streaming HTTP layer (which builds its SSE diffs from the
            // token stream, not from history) emits a content delta.
            // Without this, the engine ends up with content in history
            // but the client only ever saw reasoning_content deltas and
            // shows an empty bubble.  We feed it as a single chunk —
            // the partial parser appends it after any prior <think>…
            // </think> block so it parses cleanly as content.
            if (p_->on_token && !msg.content.empty()) {
                p_->on_token(msg.content);
            }
        }

        p_->history.push_back(msg);

        if (msg.tool_calls.empty()) {
            final_text = msg.content;
            // Highlight empty-content turns at hop end — usually means
            // the model thought, decided not to call a tool, and then
            // emitted EOS without any visible reply.  The streaming
            // layer's last-resort fallback will paint the bubble with
            // the engine's promoted reasoning if it ran (logged
            // upstream), but if even that came up empty, the user's
            // bubble will be blank — surface it loudly so the operator
            // can correlate against journalctl.
            if (final_text.empty() && p_->verbose) {
                std::fprintf(stderr,
                    "[easyai] hop %d: WARN final content is EMPTY after %d hop(s); "
                    "reasoning=%zu tool_calls=%zu — model gave up without an "
                    "answer.  Common causes: tool error chain (rate limits, "
                    "network); over-prescriptive system prompt; model "
                    "exhausted on a niche question.\n",
                    hop, hop + 1,
                    msg.reasoning_content.size(),
                    msg.tool_calls.size());
            }
            break;
        }

        // Run each tool call; append a tool message for each result.
        for (const auto & tc : msg.tool_calls) {
            const Tool * tool = p_->find_tool(tc.name);
            ToolResult result;
            ToolCall   call{ tc.name, tc.arguments, tc.id };

            if (!tool) {
                result = ToolResult::error("unknown tool: " + tc.name);
            } else {
                try {
                    result = tool->handler(call);
                } catch (const std::exception & e) {
                    result = ToolResult::error(std::string("tool threw: ") + e.what());
                } catch (...) {
                    result = ToolResult::error("tool threw unknown exception");
                }
            }

            if (p_->on_tool) p_->on_tool(call, result);

            common_chat_msg tool_msg{};
            tool_msg.role         = "tool";
            tool_msg.content      = result.content;
            tool_msg.tool_name    = tc.name;
            tool_msg.tool_call_id = tc.id;
            p_->history.push_back(std::move(tool_msg));

            if (!p_->parallel_tool_calls) break;  // serial dispatch
        }
        // Loop again to let the model digest the tool output.
    }

    return final_text;
}

std::string Engine::last_error()    const { return p_->last_error; }
int         Engine::turns()         const { return (int) p_->history.size(); }
const std::vector<Tool> & Engine::tools() const { return p_->tools; }
std::string Engine::backend_summary() const { return p_->backend_summary; }
int         Engine::n_ctx()         const { return p_->ctx() ? llama_n_ctx(p_->ctx()) : p_->params.n_ctx; }
std::string Engine::model_path()    const { return p_->params.model.path; }

::common_chat_params Engine::chat_params_for_current_state(bool add_generation_prompt) const {
    if (!p_->loaded || !p_->templates) return {};
    return p_->render(add_generation_prompt);
}

Engine::PerfData Engine::perf_data() const {
    PerfData out;
    if (!p_->loaded || !p_->ctx()) return out;
    auto d = llama_perf_context(p_->ctx());
    out.n_prompt_tokens    = d.n_p_eval;
    out.n_predicted_tokens = d.n_eval;
    out.prompt_ms          = d.t_p_eval_ms;
    out.predicted_ms       = d.t_eval_ms;
    out.n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(p_->ctx()), 0) + 1;
    if (out.n_ctx_used < 0) out.n_ctx_used = 0;
    return out;
}

void Engine::perf_reset() {
    if (p_->loaded && p_->ctx()) llama_perf_context_reset(p_->ctx());
}

// ---------------------------------------------------------------------------
// set_sampling — rebuild the underlying common_sampler with new values.
// We free the old one (so its KV-state and grammar arrays are reclaimed) and
// rebuild from scratch. -1.0f / -1 means "leave unchanged".
// ---------------------------------------------------------------------------
Engine & Engine::set_sampling(float temperature, float top_p, int top_k, float min_p) {
    if (temperature >= 0.0f) p_->params.sampling.temp           = temperature;
    if (top_p       >= 0.0f) p_->params.sampling.top_p          = top_p;
    if (top_k       >= 0)    p_->params.sampling.top_k          = top_k;
    if (min_p       >= 0.0f) p_->params.sampling.min_p          = min_p;

    if (p_->loaded) {
        if (p_->sampler) {
            common_sampler_free(p_->sampler);
            p_->sampler = nullptr;
        }
        p_->sampler = common_sampler_init(p_->init->model(), p_->params.sampling);
        if (!p_->sampler) {
            p_->last_error = "set_sampling: failed to rebuild sampler";
        }
    }
    return *this;
}

// ---------------------------------------------------------------------------
// push_message — append a message of any role to history without generating.
// Used by HTTP server to replay an OpenAI request and by tool-result feeding.
// ---------------------------------------------------------------------------
Engine & Engine::push_message(std::string role,
                              std::string content,
                              std::string tool_name,
                              std::string tool_call_id) {
    common_chat_msg m{};
    m.role         = std::move(role);
    m.content      = std::move(content);
    m.tool_name    = std::move(tool_name);
    m.tool_call_id = std::move(tool_call_id);
    p_->history.push_back(std::move(m));
    return *this;
}

void Engine::clear_history() {
    p_->history.clear();
    if (p_->ctx())   llama_memory_clear(llama_get_memory(p_->ctx()), true);
    if (p_->sampler) common_sampler_reset(p_->sampler);
}

void Engine::replace_history(const std::vector<std::pair<std::string, std::string>> & messages) {
    clear_history();
    bool has_system = false;
    for (const auto & m : messages) if (m.first == "system") { has_system = true; break; }
    if (!has_system && !p_->system_prompt.empty()) {
        p_->history.push_back({ "system", p_->system_prompt, {}, {}, "", "", "" });
    }
    for (const auto & m : messages) {
        common_chat_msg msg{};
        msg.role    = m.first;
        msg.content = m.second;
        p_->history.push_back(std::move(msg));
    }
}

// ---------------------------------------------------------------------------
// generate_one — single-pass generation. Renders prompt, decodes, parses, and
// returns the structured message. Used by HTTP layer when client provides its
// own tools (we do NOT dispatch them; we forward them back).
// ---------------------------------------------------------------------------
Engine::GeneratedTurn Engine::generate_one() {
    GeneratedTurn out{};
    if (!p_->loaded) { out.finish_reason = "error"; p_->last_error = "engine not loaded"; return out; }

    auto chat_p = p_->render(/*add_generation_prompt=*/true);
    std::string raw = generate();
    if (!p_->last_error.empty() && raw.empty()) {
        out.finish_reason = "error";
        return out;
    }

    common_chat_msg msg = p_->parse_assistant(raw, chat_p);
    msg.role = "assistant";

    if (p_->verbose) {
        // Single-turn equivalent of chat_continue's per-hop dump.  Lets
        // an operator running --verbose see EXACTLY what the model
        // emitted in client_tools mode (where a thought-only turn used
        // to silently produce an empty bubble).  Tail is bounded so a
        // long generation doesn't flood the journal.
        std::fprintf(stderr,
            "[easyai] generate_one: raw=%zu content=%zu reasoning=%zu tool_calls=%zu\n",
            raw.size(), msg.content.size(),
            msg.reasoning_content.size(), msg.tool_calls.size());
        if (!raw.empty()) {
            const size_t tail = std::min<size_t>(220, raw.size());
            std::fprintf(stderr, "[easyai] generate_one raw tail: %.*s\n",
                         (int) tail, raw.c_str() + raw.size() - tail);
        }
    }

    p_->history.push_back(msg);

    out.content   = msg.content;
    out.reasoning = msg.reasoning_content;
    out.tool_calls.reserve(msg.tool_calls.size());
    out.tool_call_ids.reserve(msg.tool_calls.size());
    for (const auto & tc : msg.tool_calls) {
        out.tool_calls.emplace_back(tc.name, tc.arguments);
        out.tool_call_ids.push_back(tc.id);
    }
    out.finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    return out;
}

}  // namespace easyai
