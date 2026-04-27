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
#include "easyai/client.hpp"   // RemoteBackend uses libeasyai-cli for HTTP/SSE

#include <atomic>
#include <chrono>
#include <unistd.h>     // isatty
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

// libcurl is only used by the LOCAL backend's web tools (web_fetch /
// web_search) — the REMOTE backend now goes through libeasyai-cli's
// httplib transport, which has its own HTTPS support via OpenSSL.
#if defined(EASYAI_HAVE_CURL)
#include <curl/curl.h>
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
        std::string sandbox;            // empty = fs_* tools NOT registered
        bool        allow_bash = false; // explicit opt-in for the `bash` tool
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
            easyai::cli::Toolbelt()
                .sandbox   (cfg_.sandbox)
                .allow_bash(cfg_.allow_bash)
                .apply     (engine_);
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
//  RemoteBackend — talks OpenAI-compatible HTTP via libeasyai-cli.
//
//  Was a ~270-line manual curl + SSE parser.  Now it's a thin shim over
//  easyai::Client which already does:
//    * HTTP and HTTPS (the latter when libssl-dev was found at configure
//      time and CPPHTTPLIB_OPENSSL_SUPPORT got compiled in)
//    * SSE parsing with the same delta.{content,reasoning_content,
//      tool_calls} routing the easyai-cli-remote binary uses
//    * History persistence as raw OpenAI-shape JSON strings
//    * Bearer auth + connection timeouts + fluent sampling overrides
//
//  Per-Backend tools are NOT registered on the Client — easyai-cli's
//  --url mode is meant to behave like a vanilla OpenAI client.  If the
//  user wants local tool dispatch, they should reach for the
//  easyai-cli-remote binary instead, which is purpose-built for that.
// ============================================================================

class RemoteBackend final : public Backend {
   public:
    struct Config {
        std::string base_url;       // e.g. "http://127.0.0.1:8080" (or "/v1")
        std::string api_key;        // optional Bearer token
        std::string model = "easyai";
        std::string system_prompt;
        std::string sandbox;        // empty = fs_* tools NOT registered (even with --with-tools)
        bool        allow_bash = false;  // explicit opt-in for the `bash` tool
        easyai::Preset preset{};
        long timeout_seconds = 300;
        int      max_tokens  = -1;
        long long seed       = -1;
        bool     with_tools  = false;   // when true, register builtin tools on Client
        bool     tls_insecure = false;  // skip peer cert verification (https only)
        std::string ca_cert_path;       // PEM bundle for custom CAs
    };

    explicit RemoteBackend(Config c) : cfg_(std::move(c)) {
        if (cfg_.preset.name.empty()) {
            const auto * p = easyai::find_preset("balanced");
            if (p) cfg_.preset = *p;
        }
        rebuild_client_();
    }

    bool init(std::string & err) override {
        if (cfg_.base_url.empty()) {
            err = "remote backend: --url is required";
            return false;
        }
        return true;
    }

    std::string chat(const std::string & user_text, const Tokenizer & cb) override {
        client_->on_token(cb);
        // libeasyai-cli runs the agentic loop and returns the final visible
        // content; cb is fired for each delta.content piece along the way.
        std::string answer = client_->chat(user_text);
        if (answer.empty() && !client_->last_error().empty()) {
            last_err_ = client_->last_error();
            return {};
        }
        return answer;
    }

    void reset() override {
        client_->clear_history();
        last_err_.clear();
    }
    void set_system(const std::string & t) override {
        cfg_.system_prompt = t;
        rebuild_client_();
    }
    void set_sampling(float t, float p, int k, float m) override {
        if (t >= 0) cfg_.preset.temperature = t;
        if (p >= 0) cfg_.preset.top_p       = p;
        if (k >= 0) cfg_.preset.top_k       = k;
        if (m >= 0) cfg_.preset.min_p       = m;
        push_sampling_();
    }
    std::string info() const override {
        std::ostringstream o;
        o << "remote " << cfg_.base_url
          << "  (model=" << cfg_.model << ")"
          << (cfg_.api_key.empty() ? "" : "  [auth]");
        return o.str();
    }
    std::string last_error() const override { return last_err_; }
    size_t tool_count() const override {
        return client_ ? client_->tools().size() : 0;
    }
    std::vector<std::pair<std::string,std::string>> tool_list() const override {
        std::vector<std::pair<std::string,std::string>> out;
        if (!client_) return out;
        for (const auto & t : client_->tools()) out.emplace_back(t.name, t.description);
        return out;
    }

   private:
    void rebuild_client_() {
        client_ = std::make_unique<easyai::Client>();
        client_->endpoint(cfg_.base_url)
                .model   (cfg_.model)
                .timeout_seconds((int) cfg_.timeout_seconds);
        if (!cfg_.api_key.empty())       client_->api_key(cfg_.api_key);
        if (!cfg_.system_prompt.empty()) client_->system (cfg_.system_prompt);
        push_sampling_();
        if (cfg_.max_tokens > 0)  client_->max_tokens(cfg_.max_tokens);
        if (cfg_.seed       >= 0) client_->seed      (cfg_.seed);

        // TLS knobs — both no-ops on http:// and on builds without OpenSSL.
        if (cfg_.tls_insecure)            client_->tls_insecure(true);
        if (!cfg_.ca_cert_path.empty())   client_->ca_cert_path(cfg_.ca_cert_path);

        // Optional: register the same built-in tool catalogue the LOCAL
        // backend uses, dispatched IN THIS PROCESS while the remote model
        // does the reasoning.  Off by default (--url mode historically
        // behaved like a vanilla OpenAI streamer); flip on with
        // --with-tools to turn easyai-cli into a remote agentic CLI.
        if (cfg_.with_tools) {
            easyai::cli::Toolbelt()
                .sandbox   (cfg_.sandbox)
                .allow_bash(cfg_.allow_bash)
                .apply     (*client_);
        }
    }
    void push_sampling_() {
        if (!client_) return;
        if (cfg_.preset.temperature >= 0)  client_->temperature(cfg_.preset.temperature);
        if (cfg_.preset.top_p       >= 0)  client_->top_p      (cfg_.preset.top_p);
        if (cfg_.preset.top_k       >  0)  client_->top_k      (cfg_.preset.top_k);
        if (cfg_.preset.min_p       >  0)  client_->min_p      (cfg_.preset.min_p);
    }

    Config                              cfg_;
    std::unique_ptr<easyai::Client>     client_;
    std::string                         last_err_;
};

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

    // remote-mode TLS knobs (https:// only; no-ops on http://)
    bool        tls_insecure = false;
    std::string ca_cert_path;

    // remote-mode optional agentic loop — register libeasyai's builtin
    // tools on the Client so the remote model can call them and we
    // dispatch in-process.  Off by default (--url historically was a
    // vanilla OpenAI streamer).
    bool        with_tools  = false;

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
    std::string sandbox;        // empty = fs_* tools NOT registered
    bool allow_bash = false;    // explicit opt-in for `bash`

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
        "      --insecure-tls            skip peer cert verification (https,\n"
        "                                 DEV ONLY — never in prod)\n"
        "      --ca-cert <path>          trust this CA bundle (PEM) for https://\n"
        "      --with-tools              register libeasyai's builtin tools on\n"
        "                                 the Client so the remote model can\n"
        "                                 call them and we dispatch locally\n"
        "                                 (datetime, web_search, web_fetch, fs_*).\n"
        "                                 Default off (vanilla OpenAI streamer).\n"
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
        "      --sandbox <dir>           Enable fs_* tools (read_file,\n"
        "                                 list_dir, glob, grep, write_file),\n"
        "                                 ALL scoped to <dir>. Without\n"
        "                                 --sandbox these tools are NOT\n"
        "                                 registered.\n"
        "      --allow-bash              Register the `bash` tool (run shell\n"
        "                                 commands). cwd = --sandbox dir if\n"
        "                                 given, otherwise CWD. NOT a\n"
        "                                 hardened sandbox — the command\n"
        "                                 runs with your user privileges.\n"
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
        else if (s == "--insecure-tls")               a.tls_insecure  = true;
        else if (s == "--ca-cert")                    a.ca_cert_path  = need(i, "--ca-cert");
        else if (s == "--with-tools")                 a.with_tools    = true;
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
        else if (s == "--allow-bash")                 a.allow_bash    = true;
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
        // Convenience: auto-prepend http:// when the user gives us a
        // bare host[:port] like "ai.local:8080".  HTTPS still needs
        // the explicit `https://` prefix.
        if (args.url.compare(0, 7, "http://")  != 0
            && args.url.compare(0, 8, "https://") != 0) {
            args.url = "http://" + args.url;
        }
        // Remote backend transport now lives in libeasyai-cli (no
        // libcurl dependency); --url works regardless of EASYAI_HAVE_CURL.
        // libcurl is still optional for the LOCAL backend's web tools.
        RemoteBackend::Config rc;
        rc.base_url      = args.url;
        rc.api_key       = args.api_key;
        rc.model         = args.remote_model;
        rc.system_prompt = system_prompt;
        rc.sandbox       = args.sandbox;
        rc.allow_bash    = args.allow_bash;
        rc.preset        = preset;
        rc.max_tokens    = args.max_tokens;
        rc.seed          = args.seed;
        rc.with_tools    = args.with_tools;
        rc.tls_insecure  = args.tls_insecure;
        rc.ca_cert_path  = args.ca_cert_path;
        backend = std::make_unique<RemoteBackend>(std::move(rc));
    } else {
        LocalBackend::Config lc;
        lc.model_path     = args.model_path;
        lc.system_prompt  = system_prompt;
        lc.sandbox        = args.sandbox;
        lc.allow_bash     = args.allow_bash;
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
        easyai::ui::Spinner spinner(/*enabled=*/true);
        spinner.start_heartbeat();

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
                if (!visible.empty()) spinner.write(visible);
                // empty visible (hidden think) — heartbeat keeps the
                // glyph alive on its own.
            });
        } catch (const std::exception & e) {
            spinner.stop_heartbeat();
            spinner.finish();
            std::fprintf(stderr, "\n[easyai-cli] error: %s\n", e.what());
            return 1;
        }
        std::string tail = strip.flush();
        if (!tail.empty()) spinner.write(tail);
        spinner.stop_heartbeat();
        spinner.finish();
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
    easyai::ui::Spinner spinner(/*enabled=*/true);

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
        spinner.start_heartbeat();
        try {
            backend->chat(line, [&](const std::string & p){
                std::string visible = strip.filter(p);
                if (!visible.empty()) spinner.write(visible);
            });
            std::string tail = strip.flush();
            if (!tail.empty()) spinner.write(tail);
            spinner.stop_heartbeat();
            spinner.finish();
        } catch (const std::exception & e) {
            spinner.stop_heartbeat();
            spinner.finish();
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
