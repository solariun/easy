// easyai/cli.hpp — high-level helpers for building CLIs on top of libeasyai.
//
// What's here is everything the example binaries (easyai-cli,
// easyai-cli-remote, easyai-server) ended up writing the same way three
// times: standard tool registration, opening a tee log file with a
// pid+timestamp path and a header, sandbox validation.  Lifted into the
// lib so a third-party agent can do the same in a handful of lines.
//
// None of this is mandatory.  You can still register tools by hand with
// engine.add_tool() / client.add_tool() — Toolbelt just spares you
// writing the same `if (sandbox.empty()) ... else ...` for the fifth
// time.
#pragma once

#include "tool.hpp"
#include "plan.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace easyai {
class Engine;
class Client;
}

namespace easyai::cli {

// ---------- Toolbelt: standard agent toolset, fluently configured ----------
//
// Composes the canonical "agent flavour" of the built-in tools:
//   - datetime           (always)
//   - web_search/fetch   (unless no_web())
//   - plan tool          (when with_plan(Plan&) is called)
//   - fs_*               (when sandbox(<dir>) is called — scoped to <dir>)
//   - bash               (when allow_bash() is called)
//
// `apply(Engine&)` and `apply(Client&)` register the tools AND, when
// bash is enabled, bump the agentic-loop max_tool_hops to 99999 (bash
// flows naturally span far more turns than the default 8 cap allows).
//
// Example:
//     easyai::cli::Toolbelt()
//         .sandbox("/srv/data")
//         .allow_bash()
//         .with_plan(plan)
//         .apply(client);
class Toolbelt {
public:
    Toolbelt & sandbox      (std::string dir);   // "" stays the default (no fs_*)
    Toolbelt & allow_bash   (bool on = true);
    Toolbelt & with_plan    (Plan & plan);
    Toolbelt & no_web       (bool on = true);    // drop web_search/web_fetch
    Toolbelt & no_datetime  (bool on = true);    // drop datetime

    // Materialise the configured tool list.  Order is the canonical
    // one shown above; callers can append their own tools after.
    std::vector<Tool> tools() const;

    // Convenience: register the tools onto an Engine / Client and
    // (if bash is enabled) bump max_tool_hops to 99999.
    void apply(Engine & engine) const;
    void apply(Client & client) const;

    // Inspectors — handy when callers want to drive their own help
    // text or banner ("registered N tools, sandbox=<dir>, bash=on").
    const std::string & sandbox_dir() const { return sandbox_; }
    bool                bash_on   () const { return allow_bash_; }

private:
    std::string sandbox_;
    bool        allow_bash_  = false;
    bool        no_web_      = false;
    bool        no_datetime_ = false;
    Plan *      plan_        = nullptr;
};

// ---------- Log file helper ------------------------------------------------
//
// Opens a tee log file at `path` (or auto-picks /tmp/<prefix>-<pid>-<epoch>.log
// if path is empty), writes a header line listing argv, and registers the
// FILE* with easyai::log so subsequent `easyai::log::write()` calls tee
// into it.  Returns the FILE* (still owned by the caller — fclose it,
// then call easyai::log::set_file(nullptr)).  Returns nullptr on failure.
//
// `prefix` controls the auto-path basename.  argc/argv are recorded in
// the header so a stray log file is self-describing.
std::FILE * open_log_tee(const std::string & path,
                         const std::string & prefix,
                         int argc, char ** argv,
                         std::string * resolved_path = nullptr);

// Pair to open_log_tee: writes a "END OF LOG" footer, fcloses, and
// clears the easyai::log sink.  Safe to call on nullptr.
void close_log_tee(std::FILE * fp);

// ---------- Client introspection helpers -----------------------------------

// Returns true if the Client has a tool with the given name registered
// (matches Tool::name).  Inline so consumers don't take the
// libeasyai-cli link dep just for this lookup.
bool client_has_tool(const Client & client, const std::string & name);


// ---------- Management subcommands -----------------------------------------
//
// Standard "introspect / drive a remote easyai-server" commands lifted
// from cli_remote's run_management() so any consumer of libeasyai-cli
// can ship the same toolbelt.  Each returns a process exit code
// (0 on success, non-zero on transport / parse failure) and writes
// the human-readable result to `out`.
}  // namespace easyai::cli
namespace easyai::ui { struct Style; }
namespace easyai::cli {

// /v1/models — list everything the server is willing to serve.
int print_models       (Client & client, const ui::Style & st, std::FILE * out = stdout);
// Locally-registered tools (what we send to the model in the
// request body's tools[]).
int print_local_tools  (Client & client, const ui::Style & st, std::FILE * out = stdout);
// /v1/tools — easyai-server extension: catalogue of tools the
// server registered.
int print_remote_tools (Client & client, const ui::Style & st, std::FILE * out = stdout);
// /health — boolean check (prints "ok" / "unhealthy: <reason>").
int print_health       (Client & client, const ui::Style & st, std::FILE * out = stdout);
// /props — dump the server's props JSON.
int print_props        (Client & client, std::FILE * out = stdout);
// /metrics — dump Prometheus exposition.
int print_metrics      (Client & client, std::FILE * out = stdout);
// /v1/preset — set the server's ambient sampling preset.
int set_preset         (Client & client, const std::string & name,
                        const ui::Style & st, std::FILE * out = stdout);


// ---------- Sandbox dir validation -----------------------------------------
//
// Returns true if `path` is empty (caller didn't pass --sandbox) or
// names an existing directory.  Returns false (with `err` set to a
// human-readable reason) otherwise.  Lifts the duplicate "does it
// exist? is it a dir?" block out of every CLI's main().
bool validate_sandbox(const std::string & path, std::string & err);

}  // namespace easyai::cli
