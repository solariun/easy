#include "easyai/engine.hpp"

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

    TokenCallback on_token;
    ToolCallback  on_tool;

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
        try {
            return common_chat_parse(raw, /*is_partial=*/false, pp);
        } catch (const std::exception & e) {
            if (verbose) std::fprintf(stderr,
                "[easyai] chat parser failed (%s) - returning raw content\n", e.what());
            common_chat_msg fallback;
            fallback.role    = "assistant";
            fallback.content = raw;
            return fallback;
        }
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
Engine & Engine::on_token(TokenCallback cb)     { p_->on_token = std::move(cb); return *this; }
Engine & Engine::on_tool(ToolCallback cb)       { p_->on_tool  = std::move(cb); return *this; }

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
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
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

    constexpr int kMaxToolHops = 8;
    std::string final_text;

    for (int hop = 0; hop < kMaxToolHops; ++hop) {
        auto chat_p = p_->render(/*add_generation_prompt=*/true);
        std::string raw = generate();
        if (!p_->last_error.empty() && raw.empty()) return {};

        common_chat_msg msg = p_->parse_assistant(raw, chat_p);
        msg.role = "assistant";
        p_->history.push_back(msg);

        if (msg.tool_calls.empty()) {
            final_text = msg.content;
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
