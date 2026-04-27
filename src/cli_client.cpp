// libeasyai-cli-side overloads — symbols that touch easyai::Client live
// here so the engine-only library doesn't drag in the HTTP client.
#include "easyai/cli.hpp"
#include "easyai/client.hpp"
#include "easyai/tool.hpp"
#include "easyai/ui.hpp"

namespace easyai::cli {

void Toolbelt::apply(Client & client) const {
    for (auto & t : tools()) client.add_tool(t);
    if (allow_bash_) client.max_tool_hops(99999);
}

bool client_has_tool(const Client & client, const std::string & name) {
    for (const auto & t : client.tools()) if (t.name == name) return true;
    return false;
}

}  // namespace easyai::cli

namespace easyai::ui {

// Streaming::attach(Client &) — Client-specific because Engine has no
// on_reason channel (reasoning is only streamed in the SSE/remote
// path).  Same dispatch as the Engine variant otherwise.
Streaming & Streaming::attach(Client & client) {
    client.on_token ([this](const std::string & p){ this->on_token_(p); });
    client.on_reason([this](const std::string & p){ this->on_reason_(p); });
    client.on_tool  ([this](const ToolCall & c, const ToolResult & r){
        this->on_tool_(c, r);
    });
    return *this;
}

}  // namespace easyai::ui
