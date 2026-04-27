// easyai::Agent — friendly facade.  Lives in libeasyai-cli because it
// can transparently use either the local (Engine) or remote (Client)
// flavour.  Linking only libeasyai gets you LocalBackend; Agent itself
// requires both libs.
#include "easyai/agent.hpp"

#include "easyai/backend.hpp"
#include "easyai/presets.hpp"

#include <stdexcept>
#include <utility>

namespace easyai {

struct Agent::Impl {
    // Either-or — only one is set during construction; init() picks
    // which Backend to instantiate based on which Config is filled in.
    LocalBackend::Config  local_cfg;
    RemoteBackend::Config remote_cfg;
    bool                  is_remote = false;

    std::unique_ptr<Backend> backend;   // built lazily on first ask()
    Tokenizer                token_cb;
    std::string              last_err;

    void ensure_started() {
        if (backend) return;
        // Resolve preset by name if the user picked one with .preset(...)
        const std::string preset_name = is_remote ? remote_cfg.preset.name
                                                  : local_cfg.preset.name;
        if (!preset_name.empty()) {
            if (const auto * p = find_preset(preset_name)) {
                if (is_remote) remote_cfg.preset = *p;
                else           local_cfg.preset  = *p;
            }
        }
        if (is_remote) backend = std::make_unique<RemoteBackend>(std::move(remote_cfg));
        else           backend = std::make_unique<LocalBackend> (std::move(local_cfg));

        std::string err;
        if (!backend->init(err)) {
            last_err = err;
            backend.reset();
            throw std::runtime_error(err);
        }
    }
};

// ---------- construction ----------------------------------------------------
Agent::Agent(std::string model_path) : p_(std::make_unique<Impl>()) {
    p_->is_remote              = false;
    p_->local_cfg.model_path = std::move(model_path);
    // ngl=-1 is already the LocalBackend default (auto).
    // Default toolset: datetime / web_search / web_fetch.  fs_* and
    // bash stay off until the user opts in via .sandbox()/.allow_bash().
}

Agent Agent::remote(std::string base_url, std::string api_key) {
    Agent a("");
    a.p_->is_remote        = true;
    a.p_->local_cfg        = {};                   // unused
    a.p_->remote_cfg.base_url = std::move(base_url);
    a.p_->remote_cfg.api_key  = std::move(api_key);
    a.p_->remote_cfg.with_tools = true;            // wire the standard tools by default
    return a;
}

Agent::~Agent() = default;
Agent::Agent(Agent &&) noexcept            = default;
Agent & Agent::operator=(Agent &&) noexcept = default;

// ---------- fluent setters --------------------------------------------------
//
// All updates go straight into the Config struct that init() will see.
// Once the Backend is materialised (after the first ask), structural
// fields (sandbox, allow_bash, model, url) become read-only; soft
// fields (system, sampling, on_token) keep working through Backend.
Agent & Agent::system(std::string prompt) {
    if (p_->is_remote) p_->remote_cfg.system_prompt = prompt;
    else               p_->local_cfg.system_prompt  = prompt;
    if (p_->backend) p_->backend->set_system(prompt);
    return *this;
}

Agent & Agent::sandbox(std::string dir) {
    if (p_->is_remote) p_->remote_cfg.sandbox = std::move(dir);
    else               p_->local_cfg.sandbox  = std::move(dir);
    return *this;
}

Agent & Agent::allow_bash(bool on) {
    if (p_->is_remote) p_->remote_cfg.allow_bash = on;
    else               p_->local_cfg.allow_bash  = on;
    return *this;
}

Agent & Agent::preset(std::string name) {
    Preset & target = p_->is_remote ? p_->remote_cfg.preset
                                    : p_->local_cfg.preset;
    target.name = std::move(name);   // resolved in ensure_started()
    return *this;
}

Agent & Agent::remote_model(std::string id) {
    if (p_->is_remote) p_->remote_cfg.model = std::move(id);
    return *this;
}

Agent & Agent::temperature(float t) {
    Preset & ps = p_->is_remote ? p_->remote_cfg.preset : p_->local_cfg.preset;
    ps.temperature = t;
    if (p_->backend) p_->backend->set_sampling(t, -1, -1, -1);
    return *this;
}
Agent & Agent::top_p(float v) {
    Preset & ps = p_->is_remote ? p_->remote_cfg.preset : p_->local_cfg.preset;
    ps.top_p = v;
    if (p_->backend) p_->backend->set_sampling(-1, v, -1, -1);
    return *this;
}
Agent & Agent::top_k(int k) {
    Preset & ps = p_->is_remote ? p_->remote_cfg.preset : p_->local_cfg.preset;
    ps.top_k = k;
    if (p_->backend) p_->backend->set_sampling(-1, -1, k, -1);
    return *this;
}
Agent & Agent::min_p(float v) {
    Preset & ps = p_->is_remote ? p_->remote_cfg.preset : p_->local_cfg.preset;
    ps.min_p = v;
    if (p_->backend) p_->backend->set_sampling(-1, -1, -1, v);
    return *this;
}

Agent & Agent::on_token(Tokenizer cb) {
    p_->token_cb = std::move(cb);
    return *this;
}

// ---------- conversation ----------------------------------------------------
std::string Agent::ask(const std::string & text) {
    p_->ensure_started();   // throws on init failure
    Tokenizer cb = p_->token_cb ? p_->token_cb : Tokenizer{};
    std::string reply = p_->backend->chat(text, cb);
    if (reply.empty()) {
        std::string err = p_->backend->last_error();
        if (!err.empty()) p_->last_err = std::move(err);
    }
    return reply;
}

void Agent::reset() {
    if (p_->backend) p_->backend->reset();
    p_->last_err.clear();
}

std::string Agent::last_error() const { return p_->last_err; }

Backend & Agent::backend() {
    p_->ensure_started();
    return *p_->backend;
}

}  // namespace easyai
