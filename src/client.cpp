// client.cpp — easyai::Client (libeasyai-cli).
//
// Phase 1: skeleton only — public API present, every method either
// stores configuration or returns a "not yet implemented" error.
// Phase 2 will fill in the HTTP/SSE transport, the agentic loop, and
// the direct-endpoint helpers.
#include "easyai/client.hpp"

#include <utility>

namespace easyai {

struct Client::Impl {
    // Transport / auth.
    std::string endpoint;
    std::string api_key;
    int         timeout_seconds = 600;
    bool        verbose         = false;

    // Request shape.
    std::string model_id;
    std::string system_prompt;
    float       temperature = -1.0f;   // -1 means "don't pin"
    float       top_p       = -1.0f;
    int         top_k       = -1;
    int         max_tokens  = -1;

    // Tools (registered locally; their handlers run in this process).
    std::vector<Tool> tools;

    // Callbacks.
    TokenCallback on_token;
    TokenCallback on_reason;
    ToolCallback  on_tool;

    // Conversation state.
    // Stored as raw JSON strings to avoid leaking nlohmann::json into
    // the public ABI.  Each entry is one OpenAI message object:
    //   {"role":"system","content":"..."} etc.
    std::vector<std::string> history_json;

    // Diagnostic.
    std::string last_error;
};

Client::Client() : p_(std::make_unique<Impl>()) {}
Client::~Client() = default;
Client::Client(Client &&) noexcept = default;
Client & Client::operator=(Client &&) noexcept = default;

// ----- transport / auth ----------------------------------------------------
Client & Client::endpoint        (std::string url) { p_->endpoint        = std::move(url); return *this; }
Client & Client::api_key         (std::string key) { p_->api_key         = std::move(key); return *this; }
Client & Client::timeout_seconds (int  s)          { p_->timeout_seconds = s;   return *this; }
Client & Client::verbose         (bool v)          { p_->verbose         = v;   return *this; }

// ----- request shape -------------------------------------------------------
Client & Client::model       (std::string id)     { p_->model_id      = std::move(id);     return *this; }
Client & Client::system      (std::string prompt) { p_->system_prompt = std::move(prompt); return *this; }
Client & Client::temperature (float t)            { p_->temperature   = t;                 return *this; }
Client & Client::top_p       (float v)            { p_->top_p         = v;                 return *this; }
Client & Client::top_k       (int   v)            { p_->top_k         = v;                 return *this; }
Client & Client::max_tokens  (int   n)            { p_->max_tokens    = n;                 return *this; }

// ----- tools ---------------------------------------------------------------
Client & Client::add_tool   (Tool t) { p_->tools.push_back(std::move(t)); return *this; }
Client & Client::clear_tools()       { p_->tools.clear();                  return *this; }
const std::vector<Tool> & Client::tools() const { return p_->tools; }

// ----- callbacks -----------------------------------------------------------
Client & Client::on_token  (TokenCallback cb) { p_->on_token  = std::move(cb); return *this; }
Client & Client::on_reason (TokenCallback cb) { p_->on_reason = std::move(cb); return *this; }
Client & Client::on_tool   (ToolCallback  cb) { p_->on_tool   = std::move(cb); return *this; }

// ----- chat (phase 2 stubs) ------------------------------------------------
std::string Client::chat(const std::string & /*user*/) {
    p_->last_error = "easyai::Client::chat — not implemented yet (phase 2)";
    return {};
}
std::string Client::chat_continue() {
    p_->last_error = "easyai::Client::chat_continue — not implemented yet (phase 2)";
    return {};
}
void Client::clear_history() {
    p_->history_json.clear();
    p_->last_error.clear();
}

// ----- direct endpoints (phase 2 stubs) ------------------------------------
bool Client::list_models      (std::vector<RemoteModel> & /*out*/)
    { p_->last_error = "list_models — phase 2";       return false; }
bool Client::list_remote_tools(std::vector<RemoteTool>  & /*out*/)
    { p_->last_error = "list_remote_tools — phase 2"; return false; }
bool Client::health()
    { p_->last_error = "health — phase 2";            return false; }
bool Client::metrics(std::string & /*out*/)
    { p_->last_error = "metrics — phase 2";           return false; }
bool Client::props(std::string & /*out*/)
    { p_->last_error = "props — phase 2";             return false; }
bool Client::set_preset(const std::string & /*name*/)
    { p_->last_error = "set_preset — phase 2";        return false; }

std::string Client::last_error() const { return p_->last_error; }

}  // namespace easyai
