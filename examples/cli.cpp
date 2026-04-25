// =============================================================================
//  easyai-cli — drop-in REPL like llama-cli, with several superpowers.
//
//   * Talks to either a LOCAL GGUF model (loaded in-process via easyai::Engine)
//     or a REMOTE OpenAI-compatible endpoint over HTTP — works with
//     easyai-server, openai.com, ollama, vLLM, anything that speaks
//     `/v1/chat/completions`.
//   * Has a one-shot mode (`-p`/`--prompt`) so it slots into shell scripts
//     and pipelines.  Banners go to stderr; only the model's text goes to
//     stdout, so `result=$(easyai-cli -p '...')` works.
//   * Streams tokens token-by-token in both modes (SSE for remote).
//   * Inline preset commands (`creative 0.9 …`) and slash commands
//     (`/temp 0.5`, `/system …`, `/reset`, `/tools`, …).
//   * Optional `<think>…</think>` stripper for noisy reasoning models —
//     thinking is shown by default, `--no-think` turns suppression on.
//
//  Examples:
//
//    easyai-cli -m models/qwen2.5-1.5b.gguf
//    easyai-cli --url http://127.0.0.1:8080/v1
//    easyai-cli --url https://api.openai.com/v1 \
//               --api-key $OPENAI_API_KEY --remote-model gpt-4o-mini
//    easyai-cli -m model.gguf -p "What is 2+2?"
//    easyai-cli --url http://127.0.0.1:8080/v1 --no-think -p "explain BGP"
//
//  Memory hygiene: only RAII / unique_ptr; no raw new/delete; the libcurl
//  handle is owned by a unique_ptr-with-custom-deleter.  HTTP body sizes are
//  capped, JSON parse errors are caught at the boundary, signals only flip
//  flags / call cooperative shutdown.
// =============================================================================

#include "easyai/easyai.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(EASYAI_HAVE_CURL)
#include <curl/curl.h>
#include <nlohmann/json.hpp>
using nlohmann::json;
#endif

// ============================================================================
//  helpers
// ============================================================================
namespace {

// Read a small text file fully. Capped to 1 MiB so a stray --system-file at
// /dev/random / a multi-GB log can't eat the heap.
std::string read_text_file(const std::string & path,
                           size_t max_bytes = 1u << 20) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    if (sz <= 0) return {};
    if ((size_t) sz > max_bytes) sz = (std::streamoff) max_bytes;
    f.seekg(0, std::ios::beg);
    std::string out((size_t) sz, '\0');
    f.read(out.data(), sz);
    out.resize((size_t) f.gcount());
    return out;
}

// Pretty-print all built-in presets.
void print_presets() {
    std::fprintf(stderr, "\nAvailable presets:\n");
    for (const auto & p : easyai::all_presets()) {
        std::fprintf(stderr, "  %-14s  temp=%.2f top_p=%.2f top_k=%d  — %s\n",
                     p.name.c_str(), p.temperature, p.top_p, p.top_k,
                     p.description.c_str());
    }
    std::fprintf(stderr, "\nAlso: 'temp <number>', '/reset', '/tools', '/system <text>', '/quit'\n\n");
}

// Ctrl-C trap: a single SIGINT during generation interrupts; a second one
// inside ~1s exits the process. We only flip an std::atomic flag from inside
// the handler — never touch the engine from a signal context.
std::atomic<bool>    g_interrupt{false};
std::atomic<int64_t> g_last_sigint_ms{0};

void handle_sigint(int) {
    using namespace std::chrono;
    auto now = duration_cast<milliseconds>(
                   steady_clock::now().time_since_epoch()).count();
    if (now - g_last_sigint_ms.load() < 1000) {
        std::fprintf(stderr, "\n[easyai-cli] interrupted twice — exiting\n");
        std::_Exit(130);
    }
    g_last_sigint_ms.store(now);
    g_interrupt.store(true);
}
void install_sigint() { std::signal(SIGINT, handle_sigint); }

}  // namespace

// ============================================================================
//  ThinkStripper — streaming filter for <think>…</think> / <thinking>…</thinking>
//
//  The model emits tokens piece-by-piece, so a tag may be split across pieces
//  ("<thi" + "nk>" + "..."). We maintain a small text buffer between calls and
//  only emit bytes once we know they belong to the visible region.
//
//  Algorithm (per call to filter(piece)):
//    1. Append piece to buffer.
//    2. While we can advance, alternate between two states:
//         - OUTSIDE think: scan for the next <think>/<thinking>.
//             • Found  → emit everything before it; jump past tag; → INSIDE
//             • Not    → emit everything except a small trailing "safe" margin
//                        (long enough to be a partial open tag), keep margin
//                        in buffer; return.
//         - INSIDE think: scan for the next </think>/</thinking>.
//             • Found  → drop everything up to & including it; → OUTSIDE
//             • Not    → drop everything except a trailing margin (length of
//                        the longest possible close tag), keep margin; return.
//    3. flush() at end of stream emits whatever remained outside think mode.
//
//  Buffer growth is bounded — when no tag is found we always either emit or
//  truncate, so memory stays O(margin).
// ============================================================================
class ThinkStripper {
   public:
    bool enabled = false;

    // Emit visible text for a streamed piece. Returns the bytes the caller
    // should print to the user.
    std::string filter(const std::string & piece) {
        if (!enabled) return piece;
        buffer_ += piece;
        std::string out;

        for (;;) {
            if (in_think_) {
                size_t end = find_close(buffer_);
                if (end == std::string::npos) {
                    // Keep at most the longest possible close tag length so a
                    // partial close on the next call can still match.
                    if (buffer_.size() > kCloseMargin) {
                        buffer_.erase(0, buffer_.size() - kCloseMargin);
                    }
                    return out;
                }
                size_t close = buffer_.find('>', end);
                if (close == std::string::npos) return out;  // malformed
                buffer_.erase(0, close + 1);
                in_think_ = false;
            } else {
                size_t start = find_open(buffer_);
                if (start == std::string::npos) {
                    // Emit everything except a small safe margin where a
                    // partial open tag could begin.
                    size_t safe = buffer_.size() > kOpenMargin
                                      ? buffer_.size() - kOpenMargin : 0;
                    if (safe > 0) {
                        out += buffer_.substr(0, safe);
                        buffer_.erase(0, safe);
                    }
                    return out;
                }
                out += buffer_.substr(0, start);
                size_t close = buffer_.find('>', start);
                if (close == std::string::npos) {
                    // The opening '<' is there but '>' hasn't streamed yet —
                    // park what we have, wait for the rest.
                    buffer_.erase(0, start);
                    return out;
                }
                buffer_.erase(0, close + 1);
                in_think_ = true;
            }
        }
    }

    // Emit any trailing bytes accumulated in the buffer at end of stream.
    std::string flush() {
        std::string out;
        if (!enabled || !in_think_) {
            out = std::move(buffer_);
        }
        buffer_.clear();
        in_think_ = false;
        return out;
    }

    void reset() {
        buffer_.clear();
        in_think_ = false;
    }

   private:
    // Margins must be >= length of the longest tag we recognise.
    static constexpr size_t kOpenMargin  = 10;  // "<thinking" is 9
    static constexpr size_t kCloseMargin = 12;  // "</thinking>" is 11

    std::string buffer_;
    bool        in_think_ = false;

    static size_t find_open(const std::string & s) {
        size_t a = s.find("<think>");
        size_t b = s.find("<thinking>");
        if (a == std::string::npos) return b;
        if (b == std::string::npos) return a;
        return std::min(a, b);
    }
    static size_t find_close(const std::string & s) {
        size_t a = s.find("</think>");
        size_t b = s.find("</thinking>");
        if (a == std::string::npos) return b;
        if (b == std::string::npos) return a;
        return std::min(a, b);
    }
};

// ============================================================================
//  Backend interface — the abstraction over local engine vs. remote HTTP.
// ============================================================================
class Backend {
   public:
    using Tokenizer = std::function<void(const std::string &)>;

    virtual ~Backend() = default;
    virtual bool        init(std::string & err)                                    = 0;
    virtual std::string chat(const std::string & user_text, const Tokenizer & cb)  = 0;
    virtual void        reset()                                                    = 0;
    virtual void        set_system(const std::string & text)                       = 0;
    virtual void        set_sampling(float temp, float top_p, int top_k, float min_p) = 0;
    virtual std::string info() const                                               = 0;
    virtual std::string last_error() const                                         = 0;
    virtual size_t      tool_count() const                                         = 0;
    virtual std::vector<std::pair<std::string,std::string>> tool_list() const      = 0;
};

// ============================================================================
//  LocalBackend — wraps easyai::Engine.
// ============================================================================
class LocalBackend final : public Backend {
   public:
    struct Config {
        std::string model_path;
        std::string system_prompt;
        std::string sandbox = ".";
        int  n_ctx     = 4096;
        int  n_batch   = 0;        // 0 = follow ctx
        int  ngl       = -1;
        int  n_threads = 0;
        bool load_tools = true;
        easyai::Preset preset{};
        // Sampling overrides (applied after preset).  -1 / 0 means "unset".
        float repeat_penalty = -1.0f;
        int   max_tokens     = -1;
        uint32_t seed        = 0u;
        // KV cache & GGUF-metadata overrides
        std::string cache_type_k;       // "" = leave default
        std::string cache_type_v;
        bool no_kv_offload = false;
        bool kv_unified    = false;
        std::vector<std::string> kv_overrides;
    };

    explicit LocalBackend(Config c) : cfg_(std::move(c)) {}

    bool init(std::string & err) override {
        engine_.model      (cfg_.model_path)
               .context    (cfg_.n_ctx)
               .gpu_layers (cfg_.ngl)
               .system     (cfg_.system_prompt)
               .verbose    (false)
               .on_token   ([this](const std::string & p){ if (cb_) cb_(p); });
        if (cfg_.n_threads > 0) engine_.threads(cfg_.n_threads);
        if (cfg_.n_batch   > 0) engine_.batch  (cfg_.n_batch);
        if (cfg_.seed      > 0) engine_.seed   (cfg_.seed);
        if (cfg_.max_tokens >= 0) engine_.max_tokens(cfg_.max_tokens);
        if (!cfg_.cache_type_k.empty()) engine_.cache_type_k(cfg_.cache_type_k);
        if (!cfg_.cache_type_v.empty()) engine_.cache_type_v(cfg_.cache_type_v);
        if (cfg_.no_kv_offload) engine_.no_kv_offload(true);
        if (cfg_.kv_unified)    engine_.kv_unified(true);
        for (const auto & ov : cfg_.kv_overrides) engine_.add_kv_override(ov);
        if (cfg_.preset.name.empty()) {
            const auto * p = easyai::find_preset("balanced");
            if (p) cfg_.preset = *p;
        }
        engine_.temperature(cfg_.preset.temperature)
               .top_p      (cfg_.preset.top_p)
               .top_k      (cfg_.preset.top_k)
               .min_p      (cfg_.preset.min_p);
        if (cfg_.repeat_penalty > 0) engine_.repeat_penalty(cfg_.repeat_penalty);

        if (cfg_.load_tools) {
            engine_.add_tool(easyai::tools::datetime())
                   .add_tool(easyai::tools::web_fetch())
                   .add_tool(easyai::tools::web_search())
                   .add_tool(easyai::tools::fs_list_dir (cfg_.sandbox))
                   .add_tool(easyai::tools::fs_read_file(cfg_.sandbox))
                   .add_tool(easyai::tools::fs_glob     (cfg_.sandbox))
                   .add_tool(easyai::tools::fs_grep     (cfg_.sandbox));
        }
        engine_.on_tool([](const easyai::ToolCall & c, const easyai::ToolResult & r){
            std::fprintf(stderr,
                "\n\033[36m[tool] %s -> %s%.200s%s\033[0m\n",
                c.name.c_str(),
                r.is_error ? "ERR " : "",
                r.content.c_str(),
                r.content.size() > 200 ? "…" : "");
        });

        if (!engine_.load()) {
            err = engine_.last_error();
            return false;
        }
        return true;
    }

    std::string chat(const std::string & user_text, const Tokenizer & cb) override {
        cb_ = cb;
        try {
            return engine_.chat(user_text);
        } catch (const std::exception & e) {
            last_err_ = std::string("local engine error: ") + e.what();
            return {};
        }
    }

    void reset() override                              { engine_.clear_history(); }
    void set_system(const std::string & t) override    { engine_.system(t); engine_.clear_history(); }
    void set_sampling(float t, float p, int k, float m) override {
        engine_.set_sampling(t, p, k, m);
    }
    std::string info() const override {
        std::ostringstream o;
        o << "loaded " << cfg_.model_path
          << "  backend=" << engine_.backend_summary()
          << "  ctx=" << engine_.n_ctx()
          << "  tools=" << engine_.tools().size();
        return o.str();
    }
    std::string last_error() const override { return last_err_.empty() ? engine_.last_error() : last_err_; }
    size_t tool_count() const override { return engine_.tools().size(); }
    std::vector<std::pair<std::string,std::string>> tool_list() const override {
        std::vector<std::pair<std::string,std::string>> out;
        for (const auto & t : engine_.tools()) out.emplace_back(t.name, t.description);
        return out;
    }

   private:
    Config            cfg_;
    easyai::Engine    engine_;
    Tokenizer         cb_;
    std::string       last_err_;
};

// ============================================================================
//  RemoteBackend — talks OpenAI-compatible HTTP via libcurl.
//
//  History is kept client-side (OpenAI's API is stateless). Each chat() does:
//    1. Append user message.
//    2. POST /chat/completions with stream=true.
//    3. Parse SSE chunks; fire on_token for each delta; accumulate full text.
//    4. Append assistant message.
//    5. Return final text.
//
//  Compiled out when libcurl isn't available — we still let the binary build
//  so people can use easyai-cli purely as a local CLI.
// ============================================================================
#if defined(EASYAI_HAVE_CURL)

namespace {

// libcurl handle owned by unique_ptr with the right deleter.
struct CurlDeleter { void operator()(CURL * c) const { if (c) curl_easy_cleanup(c); } };
using CurlPtr = std::unique_ptr<CURL, CurlDeleter>;

// curl_slist owned likewise.
struct SListDeleter { void operator()(curl_slist * s) const { if (s) curl_slist_free_all(s); } };
using SListPtr = std::unique_ptr<curl_slist, SListDeleter>;

// Streaming write context for SSE parsing.
struct StreamCtx {
    std::string                                buffer;     // bytes still being assembled into events
    std::string                                final_text; // accumulated assistant content
    std::function<void(const std::string &)>   on_piece;
    bool                                       saw_done = false;
};

// libcurl write callback: called for each chunk of HTTP response body.
size_t curl_stream_cb(char * ptr, size_t size, size_t nmemb, void * userdata) {
    auto * ctx = static_cast<StreamCtx *>(userdata);
    const size_t n = size * nmemb;
    ctx->buffer.append(ptr, n);

    // SSE events are separated by a blank line. Keep popping complete events
    // off the front of the buffer until we hit a partial one.
    for (;;) {
        size_t pos = ctx->buffer.find("\n\n");
        if (pos == std::string::npos) break;
        std::string event = ctx->buffer.substr(0, pos);
        ctx->buffer.erase(0, pos + 2);

        // An event consists of one or more "field: value" lines. The OpenAI
        // protocol only ever uses the "data:" field. Concatenate all data:
        // values inside this event (some servers split them).
        std::string payload;
        size_t line_start = 0;
        while (line_start <= event.size()) {
            size_t line_end = event.find('\n', line_start);
            if (line_end == std::string::npos) line_end = event.size();
            std::string line = event.substr(line_start, line_end - line_start);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.compare(0, 5, "data:") == 0) {
                size_t v = line.find_first_not_of(" \t", 5);
                if (v != std::string::npos) {
                    if (!payload.empty()) payload += '\n';
                    payload += line.substr(v);
                } else {
                    if (!payload.empty()) payload += '\n';
                }
            }
            if (line_end >= event.size()) break;
            line_start = line_end + 1;
        }

        if (payload.empty()) continue;
        if (payload == "[DONE]") { ctx->saw_done = true; continue; }

        try {
            auto j = json::parse(payload);
            if (j.contains("choices") && j["choices"].is_array() &&
                !j["choices"].empty()) {
                const auto & ch = j["choices"][0];
                if (ch.contains("delta") && ch["delta"].is_object()) {
                    const auto & d = ch["delta"];
                    if (d.contains("content") && d["content"].is_string()) {
                        std::string piece = d["content"].get<std::string>();
                        if (!piece.empty()) {
                            ctx->final_text += piece;
                            if (ctx->on_piece) ctx->on_piece(piece);
                        }
                    }
                }
            }
        } catch (const std::exception &) {
            // Malformed payload — ignore and keep streaming.
        }
    }
    return n;
}

// Append "/chat/completions" to the base URL if not already present.
std::string compose_chat_url(std::string base) {
    while (!base.empty() && base.back() == '/') base.pop_back();
    static const std::string suffix = "/chat/completions";
    if (base.size() >= suffix.size() &&
        base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return base;
    }
    return base + suffix;
}

}  // namespace

class RemoteBackend final : public Backend {
   public:
    struct Config {
        std::string base_url;       // e.g. "http://127.0.0.1:8080/v1"
        std::string api_key;        // optional Bearer token
        std::string model = "easyai";
        std::string system_prompt;
        easyai::Preset preset{};
        long timeout_seconds = 300;
        // Optional sampling/decoder overrides — passed through to the server
        // in the JSON body.  -1 / 0 means "leave it to the server".
        int      max_tokens = -1;
        uint32_t seed       = 0u;
    };

    explicit RemoteBackend(Config c) : cfg_(std::move(c)) {
        if (cfg_.preset.name.empty()) {
            const auto * p = easyai::find_preset("balanced");
            if (p) cfg_.preset = *p;
        }
        url_ = compose_chat_url(cfg_.base_url);
        if (!cfg_.system_prompt.empty()) {
            history_.push_back({{"role","system"},{"content",cfg_.system_prompt}});
        }
    }

    bool init(std::string & err) override {
        // Single global init per-process is recommended — cheap to call repeatedly.
        curl_global_init(CURL_GLOBAL_DEFAULT);
        if (!cfg_.base_url.empty()) return true;
        err = "remote backend: --url is required";
        return false;
    }

    std::string chat(const std::string & user_text, const Tokenizer & cb) override {
        history_.push_back({{"role","user"},{"content",user_text}});

        json body;
        body["model"]       = cfg_.model;
        body["messages"]    = history_;
        body["temperature"] = cfg_.preset.temperature;
        body["top_p"]       = cfg_.preset.top_p;
        body["stream"]      = true;
        // top_k / min_p are non-standard OpenAI fields but supported by
        // easyai-server, ollama, vLLM, etc. Sent as best-effort hints.
        if (cfg_.preset.top_k > 0) body["top_k"] = cfg_.preset.top_k;
        if (cfg_.preset.min_p > 0) body["min_p"] = cfg_.preset.min_p;
        if (cfg_.max_tokens   > 0) body["max_tokens"] = cfg_.max_tokens;
        if (cfg_.seed         > 0) body["seed"]       = cfg_.seed;

        const std::string body_str = body.dump();

        StreamCtx ctx;
        ctx.on_piece = cb;

        CurlPtr curl(curl_easy_init());
        if (!curl) { last_err_ = "curl_easy_init failed"; return {}; }

        // headers
        SListPtr headers(curl_slist_append(nullptr, "Content-Type: application/json"));
        headers.reset(curl_slist_append(headers.release(), "Accept: text/event-stream"));
        std::string auth;
        if (!cfg_.api_key.empty()) {
            auth = "Authorization: Bearer " + cfg_.api_key;
            headers.reset(curl_slist_append(headers.release(), auth.c_str()));
        }

        curl_easy_setopt(curl.get(), CURLOPT_URL,             url_.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER,      headers.get());
        curl_easy_setopt(curl.get(), CURLOPT_POST,            1L);
        curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS,      body_str.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE,   (long) body_str.size());
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION,   curl_stream_cb);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA,       &ctx);
        curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT,         cfg_.timeout_seconds);
        curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION,  1L);
        curl_easy_setopt(curl.get(), CURLOPT_MAXREDIRS,       5L);
        curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL,        1L);
        curl_easy_setopt(curl.get(), CURLOPT_USERAGENT,       "easyai-cli/0.1");

        CURLcode rc = curl_easy_perform(curl.get());
        long http_code = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);

        if (rc != CURLE_OK) {
            last_err_ = std::string("curl: ") + curl_easy_strerror(rc);
            // roll back the user message we tentatively added so the next
            // call doesn't include it twice.
            if (!history_.empty()) history_.pop_back();
            return {};
        }
        if (http_code >= 400) {
            // Try to surface the server's JSON error if any was buffered.
            std::string msg = "HTTP " + std::to_string(http_code);
            if (!ctx.buffer.empty()) {
                msg += " — " + ctx.buffer.substr(0, 512);
            }
            last_err_ = std::move(msg);
            if (!history_.empty()) history_.pop_back();
            return {};
        }

        // ----- non-streaming fallback -------------------------------------
        // If we asked for stream=true but the server replied with a single
        // JSON body (some endpoints just ignore the flag), our SSE parser
        // won't have produced any deltas. In that case parse what's left in
        // the buffer as a regular chat-completion response.
        if (ctx.final_text.empty() && !ctx.buffer.empty()) {
            try {
                auto j = json::parse(ctx.buffer);
                if (j.contains("choices") && j["choices"].is_array() &&
                    !j["choices"].empty()) {
                    const auto & m = j["choices"][0]["message"];
                    if (m.contains("content") && m["content"].is_string()) {
                        ctx.final_text = m["content"].get<std::string>();
                        if (cb && !ctx.final_text.empty()) cb(ctx.final_text);
                    }
                } else if (j.contains("error")) {
                    last_err_ = "remote error: " + j["error"].dump();
                    if (!history_.empty()) history_.pop_back();
                    return {};
                }
            } catch (const std::exception &) {
                // Body wasn't JSON either — leave final_text empty.
            }
        }

        // Append assistant turn so future requests see the full conversation.
        history_.push_back({{"role","assistant"},{"content",ctx.final_text}});
        return ctx.final_text;
    }

    void reset() override {
        history_.clear();
        if (!cfg_.system_prompt.empty()) {
            history_.push_back({{"role","system"},{"content",cfg_.system_prompt}});
        }
    }
    void set_system(const std::string & t) override {
        cfg_.system_prompt = t;
        reset();
    }
    void set_sampling(float t, float p, int k, float m) override {
        if (t >= 0) cfg_.preset.temperature = t;
        if (p >= 0) cfg_.preset.top_p       = p;
        if (k >= 0) cfg_.preset.top_k       = k;
        if (m >= 0) cfg_.preset.min_p       = m;
    }
    std::string info() const override {
        std::ostringstream o;
        o << "remote " << cfg_.base_url
          << "  (model=" << cfg_.model << ")"
          << (cfg_.api_key.empty() ? "" : "  [auth]");
        return o.str();
    }
    std::string last_error() const override { return last_err_; }
    size_t tool_count() const override { return 0; }
    std::vector<std::pair<std::string,std::string>> tool_list() const override { return {}; }

   private:
    Config              cfg_;
    std::string         url_;
    std::vector<json>   history_;
    std::string         last_err_;
};

#endif  // EASYAI_HAVE_CURL

// ============================================================================
//  Argument parsing
// ============================================================================
struct CliArgs {
    // mode selectors (exactly one required)
    std::string model_path;
    std::string url;

    // remote-mode auth
    std::string api_key;
    std::string remote_model = "easyai";

    // common config
    std::string system_path;
    std::string system_inline;
    std::string preset = "balanced";
    std::string prompt;          // -p one-shot mode; empty => REPL
    bool        no_think = false;

    // sampling overrides — when set, win over the preset baseline.
    // Sentinel values: <0 / 0u means "unset" (use preset value).
    float temperature    = -1.0f;
    float top_p          = -1.0f;
    int   top_k          = -1;
    float min_p          = -1.0f;
    float repeat_penalty = -1.0f;
    int   max_tokens     = -1;     // -1 = until EOG / context full
    uint32_t seed        = 0u;     // 0 = leave as preset/library default

    // local-mode tuning
    int  n_ctx = 4096, ngl = -1, n_threads = 0;
    int  n_batch = 0;              // 0 = follow ctx
    bool load_tools = true;
    std::string sandbox = ".";

    // KV cache controls
    std::string cache_type_k;      // empty = library default (f16)
    std::string cache_type_v;
    bool no_kv_offload = false;
    bool kv_unified    = false;
    std::vector<std::string> kv_overrides;  // each: "key=type:value"
};

[[noreturn]] static void die_usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s (-m model.gguf | --url <api-base>) [options]\n\n"
        "Mode (one required):\n"
        "  -m, --model <path>            Local GGUF model file\n"
        "      --url <api-base>          OpenAI-compatible HTTP endpoint\n"
        "                                 e.g. http://127.0.0.1:8080/v1\n"
        "                                      https://api.openai.com/v1\n"
        "\nRemote-mode auth & model:\n"
        "      --api-key <key>           Bearer token (env: EASYAI_API_KEY,\n"
        "                                 OPENAI_API_KEY also honoured)\n"
        "      --remote-model <name>     Model id to send (default 'easyai')\n"
        "\nCommon options:\n"
        "  -p, --prompt <text>           One-shot: run prompt, print, exit\n"
        "  -s, --system-file <path>      Read system prompt from file\n"
        "      --system <text>           Inline system prompt\n"
        "      --preset <name>           Initial preset (default 'balanced')\n"
        "      --no-think                Strip <think>...</think> from output\n"
        "                                 (thinking is shown by default)\n"
        "\nSampling overrides (apply on top of --preset):\n"
        "      --temperature <f>         Override temperature (0.0-2.0)\n"
        "      --top-p <f>               Override nucleus sampling p\n"
        "      --top-k <n>               Override top-k\n"
        "      --min-p <f>               Override min-p\n"
        "      --repeat-penalty <f>      (local mode only) repeat penalty\n"
        "      --max-tokens <n>          Cap tokens generated per turn\n"
        "      --seed <u32>              RNG seed (0 = random)\n"
        "\nLocal-mode tuning:\n"
        "  -c, --ctx <n>                 Context size (default 4096)\n"
        "      --batch <n>               Logical batch size (default = ctx)\n"
        "      --ngl <n>                 GPU layers (-1=auto, 0=CPU)\n"
        "  -t, --threads <n>             CPU threads\n"
        "      --no-tools                Don't register the built-in toolbelt\n"
        "      --sandbox <dir>           Root for fs_* tools (default '.')\n"
        "\nKV cache (local mode, all optional):\n"
        " -ctk, --cache-type-k <type>    K-cache dtype (f32|f16|bf16|q8_0|q4_0|q4_1|q5_0|q5_1|iq4_nl)\n"
        " -ctv, --cache-type-v <type>    V-cache dtype (same options) — quantising V saves a lot of VRAM\n"
        "-nkvo, --no-kv-offload          Keep KV cache on CPU even with GPU layers\n"
        "      --kv-unified              Use a single unified KV buffer across sequences\n"
        "      --override-kv <k=t:v>     Override a GGUF metadata entry (repeatable).\n"
        "                                 Types: int|float|bool|str.\n"
        "                                 Example: --override-kv tokenizer.ggml.add_bos_token=bool:false\n"
        "\n  -h, --help                    Show this help and exit\n",
        argv0);
    std::exit(1);
}

static CliArgs parse(int argc, char ** argv) {
    CliArgs a;
    auto need = [&](int & i, const char * flag) -> const char * {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", flag);
            die_usage(argv[0]);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "-m" || s == "--model")         a.model_path    = need(i, "-m");
        else if (s == "--url")                        a.url           = need(i, "--url");
        else if (s == "--api-key")                    a.api_key       = need(i, "--api-key");
        else if (s == "--remote-model")               a.remote_model  = need(i, "--remote-model");
        else if (s == "-s" || s == "--system-file")   a.system_path   = need(i, "-s");
        else if (s == "--system")                     a.system_inline = need(i, "--system");
        else if (s == "--preset")                     a.preset        = need(i, "--preset");
        else if (s == "-p" || s == "--prompt")        a.prompt        = need(i, "-p");
        else if (s == "--no-think")                   a.no_think      = true;
        else if (s == "--temperature" || s == "--temp") a.temperature  = std::atof(need(i, "--temperature"));
        else if (s == "--top-p")                      a.top_p         = std::atof(need(i, "--top-p"));
        else if (s == "--top-k")                      a.top_k         = std::atoi(need(i, "--top-k"));
        else if (s == "--min-p")                      a.min_p         = std::atof(need(i, "--min-p"));
        else if (s == "--repeat-penalty")             a.repeat_penalty= std::atof(need(i, "--repeat-penalty"));
        else if (s == "--max-tokens")                 a.max_tokens    = std::atoi(need(i, "--max-tokens"));
        else if (s == "--seed")                       a.seed          = (uint32_t) std::strtoul(need(i, "--seed"), nullptr, 10);
        else if (s == "-c" || s == "--ctx")           a.n_ctx         = std::atoi(need(i, "-c"));
        else if (s == "--batch")                      a.n_batch       = std::atoi(need(i, "--batch"));
        else if (s == "--ngl")                        a.ngl           = std::atoi(need(i, "--ngl"));
        else if (s == "-t" || s == "--threads")       a.n_threads     = std::atoi(need(i, "-t"));
        else if (s == "--no-tools")                   a.load_tools    = false;
        else if (s == "--sandbox")                    a.sandbox       = need(i, "--sandbox");
        // KV controls (local mode)
        else if (s == "-ctk" || s == "--cache-type-k") a.cache_type_k = need(i, "-ctk");
        else if (s == "-ctv" || s == "--cache-type-v") a.cache_type_v = need(i, "-ctv");
        else if (s == "-nkvo" || s == "--no-kv-offload") a.no_kv_offload = true;
        else if (s == "--kv-unified")                 a.kv_unified    = true;
        else if (s == "--override-kv")                a.kv_overrides.push_back(need(i, "--override-kv"));
        else if (s == "-h" || s == "--help")          die_usage(argv[0]);
        else { std::fprintf(stderr, "unknown arg: %s\n", s.c_str()); die_usage(argv[0]); }
    }
    if (a.model_path.empty() && a.url.empty()) {
        std::fprintf(stderr, "error: pass either -m <model> or --url <api-base>\n\n");
        die_usage(argv[0]);
    }
    if (!a.model_path.empty() && !a.url.empty()) {
        std::fprintf(stderr, "error: -m and --url are mutually exclusive\n\n");
        die_usage(argv[0]);
    }
    // env-var fallbacks for api key
    if (a.api_key.empty()) {
        if (const char * k = std::getenv("EASYAI_API_KEY"))  a.api_key = k;
        else if (const char * k = std::getenv("OPENAI_API_KEY")) a.api_key = k;
    }
    return a;
}

// ============================================================================
//  main
// ============================================================================
int main(int argc, char ** argv) {
    CliArgs args = parse(argc, argv);
    install_sigint();

    // Resolve system prompt: --system inline > -s file > built-in default.
    // The default discourages a small model from calling tools on simple
    // greetings (a noticeable problem with 0.5B-3B GGUFs).
    static constexpr char kBuiltinSystem[] =
        "You are a helpful, concise assistant.\n"
        "Answer directly for greetings, chitchat, math, and anything you "
        "already know — do NOT call a tool for those.\n"
        "Use a tool only when the request truly needs one:\n"
        "  - up-to-date / 'today' / 'latest' info → web_search, THEN web_fetch\n"
        "  - the current date/time                → datetime\n"
        "  - reading / listing files              → fs_read_file / fs_list_dir / fs_glob / fs_grep\n"
        "\n"
        "CRITICAL — every rule is mandatory:\n"
        " 1. web_search returns titles + 1-2 sentence snippets. The snippets "
        "    are NOT enough to summarise from. After every web_search you "
        "    MUST immediately call web_fetch on the top 1-3 most relevant "
        "    URLs and base your answer on the fetched body text.\n"
        " 2. Two web_search calls in a row is wrong. Search ONCE, then fetch.\n"
        " 3. NEVER announce a tool call without making it. Phrases like "
        "    \"I will fetch...\", \"let me search...\", \"I'll get...\" are "
        "    forbidden when followed by silence — either invoke the tool in "
        "    the same turn, or write the final answer right away. Saying you "
        "    are going to do something is NOT the same as doing it.\n"
        " 4. If a fetch fails (HTTP 4xx/5xx), retry with the next URL from "
        "    the search results. Do not fall back to summarising snippets.\n"
        " 5. When you cite an article, cite the URL you actually fetched.";

    std::string system_prompt = args.system_inline;
    if (system_prompt.empty() && !args.system_path.empty()) {
        system_prompt = read_text_file(args.system_path);
        if (system_prompt.empty()) {
            std::fprintf(stderr, "[easyai-cli] WARNING: failed to read system file '%s'\n",
                         args.system_path.c_str());
        }
    }
    if (system_prompt.empty() && args.load_tools) system_prompt = kBuiltinSystem;

    const easyai::Preset * p0 = easyai::find_preset(args.preset);
    easyai::Preset preset = p0 ? *p0 : *easyai::find_preset("balanced");
    // Overlay any explicit --temperature/--top-p/--top-k/--min-p on top of the
    // chosen preset so the user's flags always win.
    if (args.temperature >= 0) preset.temperature = args.temperature;
    if (args.top_p       >= 0) preset.top_p       = args.top_p;
    if (args.top_k       >= 0) preset.top_k       = args.top_k;
    if (args.min_p       >= 0) preset.min_p       = args.min_p;

    // ----- build backend ---------------------------------------------------
    std::unique_ptr<Backend> backend;
    if (!args.url.empty()) {
#if defined(EASYAI_HAVE_CURL)
        RemoteBackend::Config rc;
        rc.base_url      = args.url;
        rc.api_key       = args.api_key;
        rc.model         = args.remote_model;
        rc.system_prompt = system_prompt;
        rc.preset        = preset;
        rc.max_tokens    = args.max_tokens;
        rc.seed          = args.seed;
        backend = std::make_unique<RemoteBackend>(std::move(rc));
#else
        std::fprintf(stderr,
            "[easyai-cli] this binary was built without libcurl; --url is unavailable\n");
        return 2;
#endif
    } else {
        LocalBackend::Config lc;
        lc.model_path     = args.model_path;
        lc.system_prompt  = system_prompt;
        lc.sandbox        = args.sandbox;
        lc.n_ctx          = args.n_ctx;
        lc.n_batch        = args.n_batch;
        lc.ngl            = args.ngl;
        lc.n_threads      = args.n_threads;
        lc.load_tools     = args.load_tools;
        lc.preset         = preset;
        lc.repeat_penalty = args.repeat_penalty;
        lc.max_tokens     = args.max_tokens;
        lc.seed           = args.seed;
        lc.cache_type_k   = args.cache_type_k;
        lc.cache_type_v   = args.cache_type_v;
        lc.no_kv_offload  = args.no_kv_offload;
        lc.kv_unified     = args.kv_unified;
        lc.kv_overrides   = args.kv_overrides;
        backend = std::make_unique<LocalBackend>(std::move(lc));
    }

    std::string err;
    if (!backend->init(err)) {
        std::fprintf(stderr, "[easyai-cli] init failed: %s\n", err.c_str());
        return 1;
    }

    // ----- one-shot mode --------------------------------------------------
    if (!args.prompt.empty()) {
        // Banners → stderr so stdout is clean for piping.
        std::fprintf(stderr, "[easyai-cli] %s\n", backend->info().c_str());

        ThinkStripper strip;
        strip.enabled = args.no_think;

        // Honour an inline preset prefix in the prompt too.
        std::string text = args.prompt;
        easyai::PresetResult pr = easyai::parse_preset(text);
        if (!pr.applied.empty()) {
            backend->set_sampling(pr.temperature, pr.top_p, pr.top_k, pr.min_p);
            std::fprintf(stderr, "[preset → %s]\n", pr.applied.c_str());
            text = text.substr(pr.consumed);
        }

        try {
            backend->chat(text, [&](const std::string & p){
                std::string visible = strip.filter(p);
                if (!visible.empty()) std::cout << visible << std::flush;
            });
        } catch (const std::exception & e) {
            std::fprintf(stderr, "\n[easyai-cli] error: %s\n", e.what());
            return 1;
        }
        std::string tail = strip.flush();
        if (!tail.empty()) std::cout << tail;
        std::cout << std::endl;
        return 0;
    }

    // ----- REPL mode ------------------------------------------------------
    std::fprintf(stderr,
        "[easyai-cli] %s  preset=%s%s\n"
        "             type '/help' for commands, '/quit' to exit\n",
        backend->info().c_str(), preset.name.c_str(),
        args.no_think ? "  [no-think]" : "");

    ThinkStripper strip;
    strip.enabled = args.no_think;

    std::string line;
    while (true) {
        std::cout << "\n\033[32m> \033[0m" << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        // -------- meta-commands -------------------------------------------
        if (line == "/quit" || line == "/exit") break;
        if (line == "/help" || line == "/?")    { print_presets(); continue; }
        if (line == "/reset") {
            backend->reset();
            strip.reset();
            std::cout << "[history cleared]\n";
            continue;
        }
        if (line == "/think")    { strip.enabled = false; std::cout << "[thinking shown]\n"; continue; }
        if (line == "/no-think") { strip.enabled = true;  std::cout << "[thinking hidden]\n"; continue; }
        if (line == "/tools") {
            for (const auto & [n, d] : backend->tool_list()) {
                std::cout << "  " << n << " — " << d << "\n";
            }
            if (backend->tool_count() == 0) std::cout << "[no tools registered]\n";
            continue;
        }
        if (line.rfind("/system ", 0) == 0) {
            backend->set_system(line.substr(8));
            std::cout << "[system prompt updated; history cleared]\n";
            continue;
        }

        // -------- preset / temperature command ----------------------------
        easyai::PresetResult pr = easyai::parse_preset(line);
        if (!pr.applied.empty()) {
            backend->set_sampling(pr.temperature, pr.top_p, pr.top_k, pr.min_p);
            std::fprintf(stderr, "[preset → %s]\n", pr.applied.c_str());
            if (pr.consumed >= line.size()) continue;
            line = line.substr(pr.consumed);
        }

        // -------- normal generation ---------------------------------------
        g_interrupt.store(false);
        std::cout << "\033[33m";
        try {
            backend->chat(line, [&](const std::string & p){
                std::string visible = strip.filter(p);
                if (!visible.empty()) std::cout << visible << std::flush;
            });
            std::string tail = strip.flush();
            if (!tail.empty()) std::cout << tail;
        } catch (const std::exception & e) {
            std::fprintf(stderr, "\n[easyai-cli] error: %s\n", e.what());
        }
        std::cout << "\033[0m" << std::endl;

        if (!backend->last_error().empty()) {
            std::fprintf(stderr, "[easyai-cli] %s\n", backend->last_error().c_str());
        }
    }

    std::fprintf(stderr, "[easyai-cli] bye\n");
    return 0;
}
