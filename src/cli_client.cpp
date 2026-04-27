// libeasyai-cli-side overloads — symbols that touch easyai::Client live
// here so the engine-only library doesn't drag in the HTTP client.
#include "easyai/cli.hpp"
#include "easyai/client.hpp"
#include "easyai/tool.hpp"
#include "easyai/ui.hpp"

#include <cstdio>

namespace easyai::cli {

void Toolbelt::apply(Client & client) const {
    for (auto & t : tools()) client.add_tool(t);
    if (allow_bash_) client.max_tool_hops(99999);
}

bool client_has_tool(const Client & client, const std::string & name) {
    for (const auto & t : client.tools()) if (t.name == name) return true;
    return false;
}

namespace {
// One-liner for "operation failed" — prints the Client's last_error
// in red on stderr (so stdout stays clean for piping) and returns 1.
int report_error(Client & client, const ui::Style & st) {
    std::fprintf(stderr, "%serror:%s %s\n",
                 st.red(), st.reset(), client.last_error().c_str());
    return 1;
}
}  // namespace

int print_models(Client & client, const ui::Style & st, std::FILE * out) {
    std::vector<RemoteModel> ms;
    if (!client.list_models(ms)) return report_error(client, st);
    for (const auto & m : ms) {
        std::fprintf(out, "%s%s%s  (owned_by=%s)\n",
                     st.bold(), m.id.c_str(), st.reset(), m.owned_by.c_str());
    }
    return 0;
}

int print_local_tools(Client & client, const ui::Style & st, std::FILE * out) {
    if (client.tools().empty()) {
        std::fprintf(stderr,
            "%sno tools registered.%s  Use --tools / --sandbox / --allow-bash "
            "to enable some.\n", st.dim(), st.reset());
        return 0;
    }
    std::fprintf(out, "%slocal tools (%zu):%s\n",
                 st.bold(), client.tools().size(), st.reset());
    for (const auto & t : client.tools()) {
        ui::print_tool_row(t.name, t.description, st, out);
    }
    return 0;
}

int print_remote_tools(Client & client, const ui::Style & st, std::FILE * out) {
    std::vector<RemoteTool> ts;
    if (!client.list_remote_tools(ts)) return report_error(client, st);
    std::fprintf(out, "%sremote tools (%zu):%s\n",
                 st.bold(), ts.size(), st.reset());
    for (const auto & t : ts) {
        ui::print_tool_row(t.name, t.description, st, out);
    }
    return 0;
}

int print_health(Client & client, const ui::Style & st, std::FILE * out) {
    if (!client.health()) {
        std::fprintf(stderr, "%sunhealthy:%s %s\n",
                     st.red(), st.reset(), client.last_error().c_str());
        return 1;
    }
    std::fprintf(out, "%sok%s\n", st.green(), st.reset());
    return 0;
}

int print_props(Client & client, std::FILE * out) {
    std::string body;
    if (!client.props(body)) {
        std::fprintf(stderr, "error: %s\n", client.last_error().c_str());
        return 1;
    }
    std::fputs(body.c_str(), out);
    std::fputc('\n', out);
    return 0;
}

int print_metrics(Client & client, std::FILE * out) {
    std::string body;
    if (!client.metrics(body)) {
        std::fprintf(stderr, "error: %s\n", client.last_error().c_str());
        return 1;
    }
    std::fputs(body.c_str(), out);
    return 0;
}

int set_preset(Client & client, const std::string & name,
               const ui::Style & st, std::FILE * out) {
    if (!client.set_preset(name)) return report_error(client, st);
    std::fprintf(out, "%spreset → %s%s\n", st.green(), name.c_str(), st.reset());
    return 0;
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
