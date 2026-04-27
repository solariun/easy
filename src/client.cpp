// client.cpp — easyai::Client (libeasyai-cli).
//
// HTTP/SSE transport over cpp-httplib + nlohmann::json + a small
// SSE event splitter; agentic multi-hop loop that mirrors what
// Engine::chat_continue() does locally.  Tools registered via
// add_tool() are dispatched in-process whenever the remote model
// emits a tool_calls finish_reason; the result is appended to the
// conversation as a `role: tool` message and the next turn is
// requested automatically.
//
// Wire-format support:
//   delta.content              — visible reply text
//   delta.reasoning_content    — easyai-server / llama-server / DeepSeek
//   delta.tool_calls           — OpenAI tool-call deltas (incremental
//                                 arguments concatenated by index)
//
// Custom easyai-server SSE events (`event: easyai.tool_call`,
// `event: easyai.tool_result`) are ignored — when we sent `tools=[...]`
// in the request body the server forwards real delta.tool_calls, so
// the custom events are pure UI noise from our side.
#include "easyai/client.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace easyai {

namespace {

using nlohmann::json;
using ordered_json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// URL parsing
// ---------------------------------------------------------------------------
struct ParsedUrl {
    std::string scheme;   // "http" or "https"
    std::string host;
    int         port = 80;
    std::string base_path;
    bool        ok = false;
};

ParsedUrl parse_url(const std::string & url) {
    ParsedUrl out;
    auto sp = url.find("://");
    if (sp == std::string::npos) return out;
    out.scheme = url.substr(0, sp);
    if (out.scheme != "http" && out.scheme != "https") return out;
    size_t i = sp + 3;
    out.port = (out.scheme == "https") ? 443 : 80;
    auto path_start = url.find('/', i);
    std::string authority =
        (path_start == std::string::npos) ? url.substr(i)
                                          : url.substr(i, path_start - i);
    if (path_start != std::string::npos) out.base_path = url.substr(path_start);
    auto colon = authority.find(':');
    if (colon == std::string::npos) {
        out.host = authority;
    } else {
        out.host = authority.substr(0, colon);
        try { out.port = std::stoi(authority.substr(colon + 1)); }
        catch (...) { return out; }
    }
    if (out.host.empty()) return out;
    // Strip any trailing slash from base_path so path joining is clean.
    while (!out.base_path.empty() && out.base_path.back() == '/')
        out.base_path.pop_back();
    out.ok = true;
    return out;
}

// ---------------------------------------------------------------------------
// Tool → JSON (OpenAI tool spec shape)
// ---------------------------------------------------------------------------
ordered_json tool_to_json(const Tool & t) {
    ordered_json fn;
    fn["name"]        = t.name;
    fn["description"] = t.description;
    auto empty_schema = []() {
        ordered_json e;
        e["type"]       = "object";
        e["properties"] = ordered_json::object();
        return e;
    };
    if (!t.parameters_json.empty()) {
        try { fn["parameters"] = ordered_json::parse(t.parameters_json); }
        catch (...) { fn["parameters"] = empty_schema(); }
    } else {
        fn["parameters"] = empty_schema();
    }
    ordered_json tool;
    tool["type"]     = "function";
    tool["function"] = std::move(fn);
    return tool;
}

// ---------------------------------------------------------------------------
// SSE event splitter — feed bytes, get back fully-buffered events as
// {evt_type, data} pairs.
// ---------------------------------------------------------------------------
struct SseEvent {
    std::string event;   // "message" if no `event:` line
    std::string data;
};

class SseBuffer {
public:
    void feed(const char * bytes, size_t n) {
        buf_.append(bytes, n);
    }
    bool next(SseEvent & out) {
        // SSE event terminator is a blank line — accept either \n\n
        // or \r\n\r\n.
        size_t end = std::string::npos;
        size_t terminator_len = 0;
        size_t a = buf_.find("\n\n");
        size_t b = buf_.find("\r\n\r\n");
        if (a != std::string::npos && (b == std::string::npos || a < b)) {
            end = a; terminator_len = 2;
        } else if (b != std::string::npos) {
            end = b; terminator_len = 4;
        }
        if (end == std::string::npos) return false;

        out = SseEvent{};
        out.event = "message";
        std::string raw = buf_.substr(0, end);
        buf_.erase(0, end + terminator_len);

        std::string data;
        size_t i = 0;
        while (i < raw.size()) {
            size_t nl = raw.find('\n', i);
            std::string line = raw.substr(i, nl == std::string::npos
                                             ? raw.size() - i : nl - i);
            i = (nl == std::string::npos) ? raw.size() : nl + 1;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == ':') continue;  // comment
            auto colon = line.find(':');
            std::string field = (colon == std::string::npos)
                                    ? line : line.substr(0, colon);
            std::string val = (colon == std::string::npos)
                                  ? std::string() : line.substr(colon + 1);
            if (!val.empty() && val.front() == ' ') val.erase(0, 1);
            if      (field == "event") out.event = val;
            else if (field == "data")  { if (!data.empty()) data += "\n"; data += val; }
            else                       { /* ignore id/retry/etc */ }
        }
        out.data = std::move(data);
        return true;
    }
private:
    std::string buf_;
};

}  // namespace

// ===========================================================================
// Pending tool call accumulator (tool_calls deltas come in pieces).
// ===========================================================================
namespace {
struct PendingToolCall {
    int         index = -1;
    std::string id;
    std::string name;
    std::string arguments;     // built up across deltas
};
}  // namespace

// ===========================================================================
// Impl
// ===========================================================================
struct Client::Impl {
    // Transport / auth.
    std::string endpoint;
    std::string api_key;
    int         timeout_seconds = 600;
    bool        verbose         = false;
    bool        tls_insecure    = false;   // skip peer cert verification
    std::string tls_ca_path;                // PEM bundle for custom CAs

    // Request shape.  -1 / -1.0f / empty == "leave server default in place".
    std::string              model_id;
    std::string              system_prompt;
    float                    temperature       = -1.0f;
    float                    top_p             = -1.0f;
    int                      top_k             = -1;
    float                    min_p             = -1.0f;
    float                    repeat_penalty    = -1.0f;
    float                    frequency_penalty = -2.0f;  // -2 = unset (real OpenAI range starts at -2)
    float                    presence_penalty  = -2.0f;
    long long                seed              = -1;
    int                      max_tokens        = -1;
    std::vector<std::string> stop_sequences;
    std::string              extra_body_raw;            // JSON object literal

    // Tools (registered locally; their handlers run in this process).
    std::vector<Tool> tools;

    // Callbacks.
    Client::TokenCallback on_token;
    Client::TokenCallback on_reason;
    Client::ToolCallback  on_tool;

    // Conversation state.  Each entry is one OpenAI message (raw JSON
    // object) so we don't leak nlohmann::json into the public ABI.
    std::vector<std::string> history_json;

    std::string last_error;

    // -----------------------------------------------------------------
    // Transport helpers
    // -----------------------------------------------------------------
    std::unique_ptr<httplib::Client> make_http() {
        ParsedUrl u = parse_url(endpoint);
        if (!u.ok) {
            last_error = "invalid endpoint URL: " + endpoint;
            return nullptr;
        }
        // Build the authority-only URL httplib's URL constructor wants.
        // (It dispatches internally to ClientImpl for http:// or to
        // SSLClient for https:// when CPPHTTPLIB_OPENSSL_SUPPORT is on.)
        std::string auth = u.scheme + "://" + u.host + ":"
                         + std::to_string(u.port);
        auto cli = std::make_unique<httplib::Client>(auth);
        if (!cli->is_valid()) {
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
            if (u.scheme == "https") {
                last_error = "HTTPS endpoint requires OpenSSL — install "
                             "libssl-dev and rebuild libeasyai-cli";
                return nullptr;
            }
#endif
            last_error = "transport setup failed for endpoint " + endpoint;
            return nullptr;
        }
        if (u.scheme == "https") {
            if (tls_insecure) {
                cli->enable_server_certificate_verification(false);
            }
            if (!tls_ca_path.empty()) {
                cli->set_ca_cert_path(tls_ca_path.c_str());
            }
        }
        cli->set_read_timeout (timeout_seconds, 0);
        cli->set_write_timeout(timeout_seconds, 0);
        cli->set_keep_alive(true);
        return cli;
    }

    httplib::Headers headers_with_auth(const std::string & accept) const {
        httplib::Headers h;
        if (!accept.empty()) h.emplace("Accept", accept);
        if (!api_key.empty()) h.emplace("Authorization", "Bearer " + api_key);
        return h;
    }

    std::string base_path() const {
        ParsedUrl u = parse_url(endpoint);
        return u.ok ? u.base_path : std::string();
    }

    // -----------------------------------------------------------------
    // Build messages array from history + system prompt.
    // -----------------------------------------------------------------
    ordered_json messages_array() const {
        ordered_json arr = ordered_json::array();
        if (!system_prompt.empty()) {
            ordered_json sys;
            sys["role"]    = "system";
            sys["content"] = system_prompt;
            arr.push_back(std::move(sys));
        }
        for (const auto & raw : history_json) {
            try {
                arr.push_back(ordered_json::parse(raw));
            } catch (...) {
                // Drop malformed entries silently — we shouldn't have
                // pushed bad JSON in the first place.
            }
        }
        return arr;
    }

    // Build the POST body for /v1/chat/completions.
    std::string build_chat_body() const {
        ordered_json body;
        if (!model_id.empty()) body["model"] = model_id;
        body["messages"] = messages_array();
        body["stream"]   = true;
        if (temperature       >= 0.0f) body["temperature"]       = temperature;
        if (top_p             >= 0.0f) body["top_p"]             = top_p;
        if (top_k             >= 0)    body["top_k"]             = top_k;
        if (min_p             >= 0.0f) body["min_p"]             = min_p;
        if (repeat_penalty    >  0.0f) body["repeat_penalty"]    = repeat_penalty;
        if (frequency_penalty > -2.0f) body["frequency_penalty"] = frequency_penalty;
        if (presence_penalty  > -2.0f) body["presence_penalty"]  = presence_penalty;
        if (seed              >= 0)    body["seed"]              = seed;
        if (max_tokens        >= 0)    body["max_tokens"]        = max_tokens;
        if (!stop_sequences.empty()) {
            ordered_json arr = ordered_json::array();
            for (const auto & s : stop_sequences) arr.push_back(s);
            body["stop"] = std::move(arr);
        }
        if (!tools.empty()) {
            ordered_json tarr = ordered_json::array();
            for (const auto & t : tools) tarr.push_back(tool_to_json(t));
            body["tools"]       = std::move(tarr);
            body["tool_choice"] = "auto";
        }
        // Merge user-supplied extras last so they can override anything above.
        if (!extra_body_raw.empty()) {
            try {
                ordered_json extras = ordered_json::parse(extra_body_raw);
                if (extras.is_object()) {
                    for (auto it = extras.begin(); it != extras.end(); ++it) {
                        body[it.key()] = it.value();
                    }
                }
            } catch (...) {
                // Silently skip — the public setter validates at set time
                // (or the body simply lacks the extras the caller wanted).
            }
        }
        return body.dump();
    }

    // -----------------------------------------------------------------
    // SSE → callbacks; populates `out_msg` with what the assistant
    // produced this turn (content + tool_calls + finish_reason).
    // -----------------------------------------------------------------
    struct AssistantTurn {
        std::string                  content;
        std::string                  reasoning;
        std::vector<PendingToolCall> tool_calls;
        std::string                  finish_reason = "stop";
    };

    bool stream_chat(AssistantTurn & out) {
        out = AssistantTurn{};
        auto cli = make_http();
        if (!cli) return false;

        SseBuffer sse;
        std::map<int, PendingToolCall> tc_by_index;
        bool received_anything = false;

        auto on_chunk = [&](const char * data, size_t len) -> bool {
            sse.feed(data, len);
            SseEvent ev;
            while (sse.next(ev)) {
                if (ev.data.empty() || ev.data == "[DONE]") continue;
                // Skip our server's UI-only custom events.
                if (ev.event == "easyai.tool_call" ||
                    ev.event == "easyai.tool_result") continue;
                ordered_json j;
                try { j = ordered_json::parse(ev.data); }
                catch (...) { continue; }

                if (!j.contains("choices") || !j["choices"].is_array()
                        || j["choices"].empty()) continue;
                received_anything = true;
                static const ordered_json kEmptyObj = ordered_json::object();
                auto & ch    = j["choices"][0];
                const ordered_json & delta = ch.contains("delta")
                                                 ? ch["delta"]
                                                 : kEmptyObj;

                if (delta.contains("reasoning_content")
                        && delta["reasoning_content"].is_string()) {
                    const auto & s = delta["reasoning_content"].get_ref<const std::string &>();
                    out.reasoning += s;
                    if (on_reason) on_reason(s);
                }
                if (delta.contains("content") && delta["content"].is_string()) {
                    const auto & s = delta["content"].get_ref<const std::string &>();
                    out.content += s;
                    if (on_token) on_token(s);
                }
                if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
                    for (const auto & tcj : delta["tool_calls"]) {
                        int idx = tcj.value("index", 0);
                        auto & p = tc_by_index[idx];
                        if (p.index < 0) p.index = idx;
                        if (tcj.contains("id") && tcj["id"].is_string())
                            p.id = tcj["id"].get<std::string>();
                        if (tcj.contains("function") && tcj["function"].is_object()) {
                            const auto & fn = tcj["function"];
                            if (fn.contains("name") && fn["name"].is_string()) {
                                if (p.name.empty())
                                    p.name = fn["name"].get<std::string>();
                            }
                            if (fn.contains("arguments") && fn["arguments"].is_string()) {
                                p.arguments += fn["arguments"].get<std::string>();
                            }
                        }
                    }
                }
                if (ch.contains("finish_reason") && ch["finish_reason"].is_string()) {
                    out.finish_reason = ch["finish_reason"].get<std::string>();
                }
            }
            return true;
        };

        std::string body = build_chat_body();
        std::string path = base_path() + "/v1/chat/completions";

        if (verbose) {
            std::fprintf(stderr,
                "[easyai-cli] POST %s%s  (body=%zu bytes, tools=%zu, hist=%zu)\n",
                endpoint.c_str(), path.c_str(), body.size(),
                tools.size(), history_json.size());
        }

        auto headers = headers_with_auth("text/event-stream");
        auto res = cli->Post(path, headers, body, "application/json",
                             [&](const char * d, size_t l) {
                                 return on_chunk(d, l);
                             });

        if (!res) {
            last_error = "HTTP request failed: "
                       + httplib::to_string(res.error());
            return false;
        }
        if (res->status < 200 || res->status >= 300) {
            last_error = "HTTP " + std::to_string(res->status) + ": " + res->body;
            return false;
        }
        if (!received_anything) {
            last_error = "empty SSE stream from server";
            return false;
        }
        // Promote tool-call map → vector in stable index order.
        for (auto & kv : tc_by_index) out.tool_calls.push_back(std::move(kv.second));
        return true;
    }

    // -----------------------------------------------------------------
    // Build the assistant message JSON we want to push into history.
    // OpenAI shape: {"role":"assistant","content":...,"tool_calls":[...]}
    // -----------------------------------------------------------------
    std::string assistant_msg_json(const AssistantTurn & turn) const {
        ordered_json msg;
        msg["role"]    = "assistant";
        msg["content"] = turn.content;
        if (!turn.tool_calls.empty()) {
            ordered_json tcs = ordered_json::array();
            for (const auto & p : turn.tool_calls) {
                ordered_json tc;
                tc["id"]       = p.id.empty() ? ("call_" + std::to_string(p.index)) : p.id;
                tc["type"]     = "function";
                tc["function"] = { {"name", p.name},
                                   {"arguments", p.arguments.empty()
                                                     ? std::string("{}")
                                                     : p.arguments } };
                tcs.push_back(std::move(tc));
            }
            msg["tool_calls"] = std::move(tcs);
        }
        return msg.dump();
    }

    std::string tool_msg_json(const std::string & tool_call_id,
                              const std::string & tool_name,
                              const std::string & content,
                              bool                is_error) const {
        ordered_json msg;
        msg["role"]         = "tool";
        msg["tool_call_id"] = tool_call_id;
        msg["name"]         = tool_name;
        msg["content"]      = is_error ? ("ERROR: " + content) : content;
        return msg.dump();
    }

    // -----------------------------------------------------------------
    // Local tool dispatch.  Returns the rendered tool message JSON.
    // -----------------------------------------------------------------
    std::string dispatch_one_tool(const PendingToolCall & p) {
        const Tool * found = nullptr;
        for (const auto & t : tools) if (t.name == p.name) { found = &t; break; }

        ToolCall   call{ p.name, p.arguments.empty() ? "{}" : p.arguments,
                         p.id.empty() ? ("call_" + std::to_string(p.index)) : p.id };
        ToolResult result;

        if (!found) {
            result = ToolResult::error("unknown tool: " + p.name);
        } else {
            try {
                result = found->handler(call);
            } catch (const std::exception & e) {
                result = ToolResult::error(std::string("tool threw: ") + e.what());
            } catch (...) {
                result = ToolResult::error("tool threw unknown exception");
            }
        }
        if (on_tool) on_tool(call, result);
        return tool_msg_json(call.id, call.name, result.content, result.is_error);
    }

    // -----------------------------------------------------------------
    // Agentic multi-hop loop.  Mirrors Engine::chat_continue limits.
    // -----------------------------------------------------------------
    static constexpr int kMaxHops = 8;

    std::string run_chat_loop() {
        for (int hop = 0; hop < kMaxHops; ++hop) {
            AssistantTurn turn;
            if (!stream_chat(turn)) return {};
            history_json.push_back(assistant_msg_json(turn));

            if (turn.finish_reason != "tool_calls" || turn.tool_calls.empty()) {
                return turn.content;
            }
            for (const auto & p : turn.tool_calls) {
                history_json.push_back(dispatch_one_tool(p));
            }
            // Loop continues — request next turn with tool results in history.
        }
        last_error = "max tool hops (" + std::to_string(kMaxHops) + ") exceeded";
        return {};
    }

    // -----------------------------------------------------------------
    // Direct-endpoint helpers — share a synchronous GET/POST path.
    // -----------------------------------------------------------------
    bool simple_get(const std::string & path,
                    const std::string & accept,
                    std::string & out_body,
                    int *         out_status = nullptr) {
        auto cli = make_http();
        if (!cli) return false;
        auto headers = headers_with_auth(accept);
        auto res = cli->Get(base_path() + path, headers);
        if (!res) {
            last_error = "HTTP request failed: "
                       + httplib::to_string(res.error());
            return false;
        }
        if (out_status) *out_status = res->status;
        out_body = res->body;
        if (res->status < 200 || res->status >= 300) {
            last_error = "HTTP " + std::to_string(res->status) + ": " + res->body;
            return false;
        }
        return true;
    }

    bool simple_post(const std::string & path,
                     const std::string & body,
                     std::string & out_body) {
        auto cli = make_http();
        if (!cli) return false;
        auto headers = headers_with_auth("application/json");
        auto res = cli->Post(base_path() + path, headers, body, "application/json");
        if (!res) {
            last_error = "HTTP request failed: "
                       + httplib::to_string(res.error());
            return false;
        }
        out_body = res->body;
        if (res->status < 200 || res->status >= 300) {
            last_error = "HTTP " + std::to_string(res->status) + ": " + res->body;
            return false;
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------
Client::Client() : p_(std::make_unique<Impl>()) {}
Client::~Client() = default;
Client::Client(Client &&) noexcept = default;
Client & Client::operator=(Client &&) noexcept = default;

Client & Client::endpoint        (std::string url) { p_->endpoint        = std::move(url); return *this; }
Client & Client::api_key         (std::string key) { p_->api_key         = std::move(key); return *this; }
Client & Client::timeout_seconds (int  s)          { p_->timeout_seconds = s;   return *this; }
Client & Client::verbose         (bool v)          { p_->verbose         = v;   return *this; }
Client & Client::tls_insecure    (bool v)          { p_->tls_insecure    = v;   return *this; }
Client & Client::ca_cert_path    (std::string p)   { p_->tls_ca_path     = std::move(p); return *this; }

Client & Client::model              (std::string id)     { p_->model_id          = std::move(id);     return *this; }
Client & Client::system             (std::string prompt) { p_->system_prompt     = std::move(prompt); return *this; }
Client & Client::temperature        (float t)            { p_->temperature       = t;                 return *this; }
Client & Client::top_p              (float v)            { p_->top_p             = v;                 return *this; }
Client & Client::top_k              (int   v)            { p_->top_k             = v;                 return *this; }
Client & Client::min_p              (float v)            { p_->min_p             = v;                 return *this; }
Client & Client::repeat_penalty     (float v)            { p_->repeat_penalty    = v;                 return *this; }
Client & Client::frequency_penalty  (float v)            { p_->frequency_penalty = v;                 return *this; }
Client & Client::presence_penalty   (float v)            { p_->presence_penalty  = v;                 return *this; }
Client & Client::seed               (long long s)        { p_->seed              = s;                 return *this; }
Client & Client::max_tokens         (int   n)            { p_->max_tokens        = n;                 return *this; }
Client & Client::stop               (std::vector<std::string> s) { p_->stop_sequences = std::move(s); return *this; }
Client & Client::extra_body_json    (std::string raw)    { p_->extra_body_raw    = std::move(raw);    return *this; }

Client & Client::add_tool   (Tool t) { p_->tools.push_back(std::move(t)); return *this; }
Client & Client::clear_tools()       { p_->tools.clear();                  return *this; }
const std::vector<Tool> & Client::tools() const { return p_->tools; }

Client & Client::on_token  (TokenCallback cb) { p_->on_token  = std::move(cb); return *this; }
Client & Client::on_reason (TokenCallback cb) { p_->on_reason = std::move(cb); return *this; }
Client & Client::on_tool   (ToolCallback  cb) { p_->on_tool   = std::move(cb); return *this; }

std::string Client::chat(const std::string & user_message) {
    p_->last_error.clear();
    ordered_json u;
    u["role"]    = "user";
    u["content"] = user_message;
    p_->history_json.push_back(u.dump());
    return p_->run_chat_loop();
}

std::string Client::chat_continue() {
    p_->last_error.clear();
    return p_->run_chat_loop();
}

void Client::clear_history() {
    p_->history_json.clear();
    p_->last_error.clear();
}

// ---- direct endpoints -----------------------------------------------------
bool Client::list_models(std::vector<RemoteModel> & out) {
    out.clear();
    std::string body;
    if (!p_->simple_get("/v1/models", "application/json", body)) return false;
    try {
        auto j = json::parse(body);
        if (!j.contains("data") || !j["data"].is_array()) return true;
        for (const auto & e : j["data"]) {
            RemoteModel m;
            m.id       = e.value("id", "");
            m.owned_by = e.value("owned_by", "");
            m.created  = e.value("created", 0L);
            out.push_back(std::move(m));
        }
        return true;
    } catch (const std::exception & e) {
        p_->last_error = std::string("list_models: bad JSON: ") + e.what();
        return false;
    }
}

bool Client::list_remote_tools(std::vector<RemoteTool> & out) {
    out.clear();
    std::string body;
    if (!p_->simple_get("/v1/tools", "application/json", body)) return false;
    try {
        auto j = json::parse(body);
        if (!j.contains("data") || !j["data"].is_array()) return true;
        for (const auto & e : j["data"]) {
            RemoteTool t;
            t.name        = e.value("name", "");
            t.description = e.value("description", "");
            out.push_back(std::move(t));
        }
        return true;
    } catch (const std::exception & e) {
        p_->last_error = std::string("list_remote_tools: bad JSON: ") + e.what();
        return false;
    }
}

bool Client::health() {
    std::string body;
    return p_->simple_get("/health", "application/json", body);
}

bool Client::metrics(std::string & out_text) {
    return p_->simple_get("/metrics", "text/plain", out_text);
}

bool Client::props(std::string & out_json) {
    return p_->simple_get("/props", "application/json", out_json);
}

bool Client::set_preset(const std::string & preset_name) {
    ordered_json req;
    req["preset"] = preset_name;
    std::string body;
    return p_->simple_post("/v1/preset", req.dump(), body);
}

std::string Client::last_error() const { return p_->last_error; }

}  // namespace easyai
