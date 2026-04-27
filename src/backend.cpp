// libeasyai-side: LocalBackend impl (wraps easyai::Engine).
// RemoteBackend (wraps Client) lives in src/cli_client.cpp / libeasyai-cli.
#include "easyai/backend.hpp"

#include "easyai/cli.hpp"
#include "easyai/engine.hpp"
#include "easyai/presets.hpp"
#include "easyai/tool.hpp"

#include <cstdio>
#include <sstream>

namespace easyai {

// --------------------------------------------------------------- LocalBackend
struct LocalBackend::Impl {
    Config         cfg;
    Engine         engine;
    Tokenizer      cb;
    std::string    last_err;
};

LocalBackend::LocalBackend(Config c) : p_(std::make_unique<Impl>()) { p_->cfg = std::move(c); }
LocalBackend::~LocalBackend() = default;

bool LocalBackend::init(std::string & err) {
    auto & cfg    = p_->cfg;
    auto & engine = p_->engine;

    engine.model      (cfg.model_path)
          .context    (cfg.n_ctx)
          .gpu_layers (cfg.ngl)
          .system     (cfg.system_prompt)
          .verbose    (false)
          .on_token   ([this](const std::string & s){ if (p_->cb) p_->cb(s); });
    if (cfg.n_threads  > 0) engine.threads   (cfg.n_threads);
    if (cfg.n_batch    > 0) engine.batch     (cfg.n_batch);
    if (cfg.seed       > 0) engine.seed      (cfg.seed);
    if (cfg.max_tokens >= 0) engine.max_tokens(cfg.max_tokens);
    if (!cfg.cache_type_k.empty()) engine.cache_type_k(cfg.cache_type_k);
    if (!cfg.cache_type_v.empty()) engine.cache_type_v(cfg.cache_type_v);
    if (cfg.no_kv_offload) engine.no_kv_offload(true);
    if (cfg.kv_unified)    engine.kv_unified(true);
    for (const auto & ov : cfg.kv_overrides) engine.add_kv_override(ov);

    if (cfg.preset.name.empty()) {
        if (const auto * p = find_preset("balanced")) cfg.preset = *p;
    }
    engine.temperature(cfg.preset.temperature)
          .top_p      (cfg.preset.top_p)
          .top_k      (cfg.preset.top_k)
          .min_p      (cfg.preset.min_p);
    if (cfg.repeat_penalty > 0) engine.repeat_penalty(cfg.repeat_penalty);

    if (cfg.load_tools) {
        cli::Toolbelt()
            .sandbox   (cfg.sandbox)
            .allow_bash(cfg.allow_bash)
            .apply     (engine);
    }
    engine.on_tool([](const ToolCall & c, const ToolResult & r){
        std::fprintf(stderr,
            "\n\033[36m[tool] %s -> %s%.200s%s\033[0m\n",
            c.name.c_str(),
            r.is_error ? "ERR " : "",
            r.content.c_str(),
            r.content.size() > 200 ? "…" : "");
    });

    if (!engine.load()) {
        err = engine.last_error();
        return false;
    }
    return true;
}

std::string LocalBackend::chat(const std::string & user_text, const Tokenizer & cb) {
    p_->cb = cb;
    try {
        return p_->engine.chat(user_text);
    } catch (const std::exception & e) {
        p_->last_err = std::string("local engine error: ") + e.what();
        return {};
    }
}

void LocalBackend::reset()                                                      { p_->engine.clear_history(); }
void LocalBackend::set_system(const std::string & t)                            { p_->engine.system(t); p_->engine.clear_history(); }
void LocalBackend::set_sampling(float t, float p, int k, float m)               { p_->engine.set_sampling(t, p, k, m); }

std::string LocalBackend::info() const {
    std::ostringstream o;
    o << "loaded "      << p_->cfg.model_path
      << "  backend="   << p_->engine.backend_summary()
      << "  ctx="       << p_->engine.n_ctx()
      << "  tools="     << p_->engine.tools().size();
    return o.str();
}

std::string LocalBackend::last_error() const {
    return p_->last_err.empty() ? p_->engine.last_error() : p_->last_err;
}

std::size_t LocalBackend::tool_count() const { return p_->engine.tools().size(); }

std::vector<std::pair<std::string,std::string>> LocalBackend::tool_list() const {
    std::vector<std::pair<std::string,std::string>> out;
    for (const auto & t : p_->engine.tools()) out.emplace_back(t.name, t.description);
    return out;
}

}  // namespace easyai
