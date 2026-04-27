// examples/cli_remote.cpp — full agentic OpenAI-protocol CLI built on
// libeasyai-cli.  Talks to any /v1/chat/completions endpoint (our
// easyai-server, llama-server, OpenAI itself).  Tools execute LOCALLY
// in this process — the model picks which tool to call, the Client
// dispatches it, and the result is fed back into the conversation.
//
// Modes:
//   easyai-cli-remote --url URL [-p PROMPT]        one-shot (exits after)
//   easyai-cli-remote --url URL                    interactive REPL
//   easyai-cli-remote --url URL --list-models      management subcommand
//   easyai-cli-remote --url URL --list-tools       management subcommand
//   easyai-cli-remote --url URL --health           management subcommand
//   easyai-cli-remote --url URL --props            management subcommand
//   easyai-cli-remote --url URL --metrics          management subcommand
//   easyai-cli-remote --url URL --set-preset NAME  management subcommand
//
// Built-in tools (off by default in the model's choice list — supplied so
// it CAN call them):
//   datetime, plan          (always)
//   web_search, web_fetch   (when libeasyai was built with curl — runtime
//                            check via the tool returning an error if not)
//   fs_list_dir, fs_read_file, fs_glob, fs_grep
//                           (only when --sandbox DIR is given; root scoped)
//
// REPL specials:
//   /exit, /quit       leave
//   /clear             clear conversation history (keep tools + system)
//   /reset             clear history AND plan
//   /plan              re-render the plan checklist
//   /tools             list registered tools and their descriptions
//
// Configuration is layered: CLI flags > env vars > defaults.  Env vars:
//   EASYAI_URL, EASYAI_API_KEY, EASYAI_MODEL.
//
// Output styling: ANSI dim for reasoning_content, cyan for tool-call
// indicators, yellow for plan checklist updates, bold for final answer.
// Auto-disabled when stdout is not a TTY.

#include "easyai/builtin_tools.hpp"
#include "easyai/client.hpp"
#include "easyai/plan.hpp"
#include "easyai/tool.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>      // isatty
#include <vector>

namespace {

// ---- ANSI helpers ---------------------------------------------------------
struct Style {
    bool color = false;
    const char * reset () const { return color ? "\033[0m"  : ""; }
    const char * dim   () const { return color ? "\033[2m"  : ""; }
    const char * bold  () const { return color ? "\033[1m"  : ""; }
    const char * cyan  () const { return color ? "\033[36m" : ""; }
    const char * yellow() const { return color ? "\033[33m" : ""; }
    const char * red   () const { return color ? "\033[31m" : ""; }
    const char * green () const { return color ? "\033[32m" : ""; }
};

Style detect_style() {
    Style s;
    s.color = ::isatty(STDOUT_FILENO) != 0
              && std::getenv("NO_COLOR") == nullptr;
    return s;
}

// ---- options + parsing ----------------------------------------------------
struct Options {
    std::string url;
    std::string api_key;
    std::string model = "EasyAi";
    std::string system_prompt;
    std::string system_file;
    std::string sandbox;
    std::set<std::string> tools_enabled;       // empty = all defaults
    std::string prompt;                        // -p one-shot
    // Sampling / penalty knobs — -1 / -2 / empty == server default.
    float                    temperature       = -1.0f;
    float                    top_p             = -1.0f;
    int                      top_k             = -1;
    float                    min_p             = -1.0f;
    float                    repeat_penalty    = -1.0f;
    float                    frequency_penalty = -2.0f;
    float                    presence_penalty  = -2.0f;
    long long                seed              = -1;
    int                      max_tokens        = -1;
    std::vector<std::string> stop_sequences;
    std::string              extra_body;       // JSON object literal
    int                      timeout           = 600;
    bool        show_reasoning = false;
    bool        verbose        = false;
    bool        no_plan        = false;        // skip auto-registering Plan

    // Management subcommands (mutually exclusive with prompt mode).
    bool        list_models = false;
    bool        list_tools  = false;
    bool        health      = false;
    bool        props       = false;
    bool        metrics     = false;
    std::string set_preset;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
"Usage: %s [options]\n"
"\n"
"  Connection (env fallback in parens):\n"
"    --url URL                  OpenAI-compat endpoint (EASYAI_URL)\n"
"    --api-key KEY              Bearer auth (EASYAI_API_KEY)\n"
"    --model NAME               request body 'model' field (EASYAI_MODEL)\n"
"    --timeout SECONDS          read+write timeout (default 600)\n"
"\n"
"  Conversation shape:\n"
"    --system TEXT              system prompt as inline string\n"
"    --system-file PATH         system prompt from a file\n"
"\n"
"  Sampling / penalty (omit any to keep the server default):\n"
"    --temperature F            sampling temperature\n"
"    --top-p F                  nucleus top-p\n"
"    --top-k N                  top-k cutoff\n"
"    --min-p F                  llama-server / easyai min-p\n"
"    --repeat-penalty F         llama-server / easyai repetition penalty\n"
"    --frequency-penalty F      OpenAI standard ([-2.0, 2.0])\n"
"    --presence-penalty F       OpenAI standard ([-2.0, 2.0])\n"
"    --seed N                   deterministic sampling seed\n"
"    --max-tokens N             cap reply length\n"
"    --stop SEQ                 add a stop string (repeatable)\n"
"    --extra-json '{...}'       free-form JSON merged into the request body\n"
"\n"
"  Tools:\n"
"    --tools LIST               comma list: datetime,plan,web_search,\n"
"                                  web_fetch,fs_read_file,fs_list_dir,\n"
"                                  fs_glob,fs_grep,fs_write_file\n"
"                               default: datetime,plan,web_search,web_fetch\n"
"    --sandbox DIR              enable fs_* tools, scoped to DIR\n"
"    --no-plan                  don't auto-register the planning tool\n"
"\n"
"  Behaviour:\n"
"    -p, --prompt TEXT          one-shot prompt; without this you get a REPL\n"
"    --show-reasoning           render delta.reasoning_content (dim) inline\n"
"    --verbose                  log HTTP+SSE traffic to stderr\n"
"\n"
"  Management subcommands (use one, no chat):\n"
"    --list-models              GET /v1/models\n"
"    --list-tools               GET /v1/tools (easyai-server extension)\n"
"    --health                   GET /health\n"
"    --props                    GET /props\n"
"    --metrics                  GET /metrics (Prometheus text)\n"
"    --set-preset NAME          POST /v1/preset {preset:NAME}\n"
"\n"
"  Misc:\n"
"    -h, --help                 this help\n",
                 argv0);
}

bool parse_args(int argc, char ** argv, Options & o) {
    auto need = [&](int & i, const char * flag) -> std::string {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", flag);
            return std::string();
        }
        return argv[++i];
    };
    if (const char * v = std::getenv("EASYAI_URL"))     o.url     = v;
    if (const char * v = std::getenv("EASYAI_API_KEY")) o.api_key = v;
    if (const char * v = std::getenv("EASYAI_MODEL"))   o.model   = v;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--url")            o.url     = need(i, "--url");
        else if (a == "--api-key")        o.api_key = need(i, "--api-key");
        else if (a == "--model")          o.model   = need(i, "--model");
        else if (a == "--timeout")        o.timeout = std::stoi(need(i, "--timeout"));
        else if (a == "--system")         o.system_prompt = need(i, "--system");
        else if (a == "--system-file")    o.system_file   = need(i, "--system-file");
        else if (a == "--sandbox")        o.sandbox       = need(i, "--sandbox");
        else if (a == "--temperature")    o.temperature       = std::stof(need(i, "--temperature"));
        else if (a == "--top-p")          o.top_p             = std::stof(need(i, "--top-p"));
        else if (a == "--top-k")          o.top_k             = std::stoi(need(i, "--top-k"));
        else if (a == "--min-p")          o.min_p             = std::stof(need(i, "--min-p"));
        else if (a == "--repeat-penalty") o.repeat_penalty    = std::stof(need(i, "--repeat-penalty"));
        else if (a == "--frequency-penalty") o.frequency_penalty = std::stof(need(i, "--frequency-penalty"));
        else if (a == "--presence-penalty")  o.presence_penalty  = std::stof(need(i, "--presence-penalty"));
        else if (a == "--seed")           o.seed              = std::stoll(need(i, "--seed"));
        else if (a == "--max-tokens")     o.max_tokens        = std::stoi(need(i, "--max-tokens"));
        else if (a == "--stop")           o.stop_sequences.push_back(need(i, "--stop"));
        else if (a == "--extra-json")     o.extra_body        = need(i, "--extra-json");
        else if (a == "--tools") {
            std::string list = need(i, "--tools");
            std::stringstream ss(list);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) o.tools_enabled.insert(tok);
            }
        }
        else if (a == "--no-plan")        o.no_plan = true;
        else if (a == "--show-reasoning") o.show_reasoning = true;
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "-p" || a == "--prompt") o.prompt = need(i, "--prompt");
        else if (a == "--list-models")    o.list_models = true;
        else if (a == "--list-tools")     o.list_tools  = true;
        else if (a == "--health")         o.health      = true;
        else if (a == "--props")          o.props       = true;
        else if (a == "--metrics")        o.metrics     = true;
        else if (a == "--set-preset")     o.set_preset  = need(i, "--set-preset");
        else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return false;
        }
    }

    if (!o.system_file.empty()) {
        std::ifstream f(o.system_file);
        if (!f) {
            std::fprintf(stderr, "cannot read --system-file %s\n", o.system_file.c_str());
            return false;
        }
        std::stringstream ss; ss << f.rdbuf();
        o.system_prompt = ss.str();
    }
    return true;
}

bool any_management(const Options & o) {
    return o.list_models || o.list_tools || o.health
        || o.props || o.metrics || !o.set_preset.empty();
}

// ---- tool registration ----------------------------------------------------
// Default catalog when --tools isn't given.
const std::vector<std::string> kDefaultTools = {
    "datetime", "plan", "web_search", "web_fetch",
};

void register_tools(easyai::Client & cli,
                    easyai::Plan & plan,
                    const Options & o,
                    const Style & st) {
    auto wants = [&](const std::string & name) {
        if (o.tools_enabled.empty()) {
            for (const auto & d : kDefaultTools) if (d == name) return true;
            return false;
        }
        return o.tools_enabled.count(name) != 0;
    };

    if (wants("datetime"))   cli.add_tool(easyai::tools::datetime());
    if (!o.no_plan && wants("plan")) cli.add_tool(plan.tool());
    if (wants("web_search")) cli.add_tool(easyai::tools::web_search());
    if (wants("web_fetch"))  cli.add_tool(easyai::tools::web_fetch());

    // fs_* — scoped to --sandbox if given, otherwise CWD.
    const std::string root = o.sandbox.empty() ? "." : o.sandbox;
    if (wants("fs_list_dir"))   cli.add_tool(easyai::tools::fs_list_dir(root));
    if (wants("fs_read_file"))  cli.add_tool(easyai::tools::fs_read_file(root));
    if (wants("fs_glob"))       cli.add_tool(easyai::tools::fs_glob(root));
    if (wants("fs_grep"))       cli.add_tool(easyai::tools::fs_grep(root));
    if (wants("fs_write_file")) cli.add_tool(easyai::tools::fs_write_file(root));

    if (o.verbose) {
        std::fprintf(stderr,
            "%s[easyai-cli-remote]%s registered %zu tool(s):",
            st.dim(), st.reset(), cli.tools().size());
        for (const auto & t : cli.tools()) std::fprintf(stderr, " %s", t.name.c_str());
        std::fputc('\n', stderr);
    }
}

// Trim a string to N chars with ellipsis suffix for log lines.
std::string trim_for_log(std::string s, size_t max_chars) {
    if (s.size() <= max_chars) return s;
    s.resize(max_chars);
    s += "…";
    // Strip newlines so the log line stays single-line.
    for (char & c : s) if (c == '\n' || c == '\r') c = ' ';
    return s;
}

// ---- management subcommand handlers ---------------------------------------
int run_management(easyai::Client & cli, const Options & o, const Style & st) {
    if (o.list_models) {
        std::vector<easyai::RemoteModel> ms;
        if (!cli.list_models(ms)) {
            std::fprintf(stderr, "%serror:%s %s\n", st.red(), st.reset(),
                         cli.last_error().c_str());
            return 1;
        }
        for (const auto & m : ms) {
            std::printf("%s%s%s  (owned_by=%s)\n",
                        st.bold(), m.id.c_str(), st.reset(), m.owned_by.c_str());
        }
        return 0;
    }
    if (o.list_tools) {
        std::vector<easyai::RemoteTool> ts;
        if (!cli.list_remote_tools(ts)) {
            std::fprintf(stderr, "%serror:%s %s\n", st.red(), st.reset(),
                         cli.last_error().c_str());
            return 1;
        }
        for (const auto & t : ts) {
            std::printf("%s%s%s\n  %s%s%s\n",
                        st.bold(), t.name.c_str(), st.reset(),
                        st.dim(),  t.description.c_str(), st.reset());
        }
        return 0;
    }
    if (o.health) {
        if (!cli.health()) {
            std::fprintf(stderr, "%sunhealthy:%s %s\n", st.red(), st.reset(),
                         cli.last_error().c_str());
            return 1;
        }
        std::printf("%sok%s\n", st.green(), st.reset());
        return 0;
    }
    if (o.props) {
        std::string body;
        if (!cli.props(body)) {
            std::fprintf(stderr, "%serror:%s %s\n", st.red(), st.reset(),
                         cli.last_error().c_str());
            return 1;
        }
        std::printf("%s\n", body.c_str());
        return 0;
    }
    if (o.metrics) {
        std::string body;
        if (!cli.metrics(body)) {
            std::fprintf(stderr, "%serror:%s %s\n", st.red(), st.reset(),
                         cli.last_error().c_str());
            return 1;
        }
        std::fputs(body.c_str(), stdout);
        return 0;
    }
    if (!o.set_preset.empty()) {
        if (!cli.set_preset(o.set_preset)) {
            std::fprintf(stderr, "%serror:%s %s\n", st.red(), st.reset(),
                         cli.last_error().c_str());
            return 1;
        }
        std::printf("preset applied: %s\n", o.set_preset.c_str());
        return 0;
    }
    return 0;
}

// ---- callback wiring ------------------------------------------------------
void render_plan(const easyai::Plan & p, const Style & st) {
    std::fprintf(stdout, "\n%s── plan ──%s\n", st.yellow(), st.reset());
    std::ostringstream ss;
    p.render(ss);
    std::fputs(ss.str().c_str(), stdout);
    std::fputs("\n", stdout);
    std::fflush(stdout);
}

void wire_callbacks(easyai::Client & cli, easyai::Plan & plan,
                    const Options & o, const Style & st) {
    cli.on_token([](const std::string & piece) {
        std::fputs(piece.c_str(), stdout);
        std::fflush(stdout);
    });
    if (o.show_reasoning) {
        cli.on_reason([&st](const std::string & piece) {
            std::fprintf(stdout, "%s%s%s",
                         st.dim(), piece.c_str(), st.reset());
            std::fflush(stdout);
        });
    }
    cli.on_tool([&st](const easyai::ToolCall & call,
                       const easyai::ToolResult & result) {
        const char * marker = result.is_error ? "✗" : "🔧";
        const char * color  = result.is_error ? st.red() : st.cyan();
        std::fprintf(stdout, "\n%s%s %s(%s)%s",
                     color, marker, call.name.c_str(),
                     trim_for_log(call.arguments_json, 80).c_str(),
                     st.reset());
        if (result.is_error) {
            std::fprintf(stdout, " %s%s%s",
                         st.red(), trim_for_log(result.content, 100).c_str(), st.reset());
        }
        std::fputs("\n", stdout);
        std::fflush(stdout);
    });
    plan.on_change([&st](const easyai::Plan & p) { render_plan(p, st); });
}

// ---- repl helpers ---------------------------------------------------------
bool is_special(const std::string & line, const std::string & cmd) {
    return line == cmd
        || (line.size() > cmd.size() && line.rfind(cmd + " ", 0) == 0);
}

int run_one(easyai::Client & cli, const std::string & prompt,
            const Style & st) {
    std::string answer = cli.chat(prompt);
    std::fputc('\n', stdout);
    if (answer.empty() && !cli.last_error().empty()) {
        std::fprintf(stderr, "%serror:%s %s\n", st.red(), st.reset(),
                     cli.last_error().c_str());
        return 1;
    }
    return 0;
}

int run_repl(easyai::Client & cli, easyai::Plan & plan, const Style & st) {
    std::fprintf(stderr,
        "%seasyai-cli-remote%s — interactive.  /exit to quit, /help for commands.\n",
        st.bold(), st.reset());
    std::string line;
    while (true) {
        std::fprintf(stdout, "%s>%s ", st.cyan(), st.reset());
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) { std::fputc('\n', stdout); break; }
        if (line.empty()) continue;

        if (is_special(line, "/exit") || is_special(line, "/quit")) break;
        if (is_special(line, "/clear")) {
            cli.clear_history();
            std::fprintf(stderr, "%shistory cleared%s\n", st.dim(), st.reset());
            continue;
        }
        if (is_special(line, "/reset")) {
            cli.clear_history(); plan.clear();
            std::fprintf(stderr, "%shistory + plan cleared%s\n",
                         st.dim(), st.reset());
            continue;
        }
        if (is_special(line, "/plan")) { render_plan(plan, st); continue; }
        if (is_special(line, "/tools")) {
            for (const auto & t : cli.tools()) {
                std::fprintf(stdout, "%s%s%s\n  %s%s%s\n",
                             st.bold(), t.name.c_str(), st.reset(),
                             st.dim(),  t.description.c_str(), st.reset());
            }
            continue;
        }
        if (is_special(line, "/help")) {
            std::fputs("/exit /quit /clear /reset /plan /tools /help\n", stdout);
            continue;
        }
        if (line[0] == '/') {
            std::fprintf(stderr, "unknown command: %s — try /help\n", line.c_str());
            continue;
        }

        run_one(cli, line, st);
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) { usage(argv[0]); return 2; }
    Style st = detect_style();

    if (o.url.empty()) {
        std::fprintf(stderr, "%serror:%s --url (or EASYAI_URL) is required\n",
                     st.red(), st.reset());
        usage(argv[0]);
        return 2;
    }

    easyai::Client cli;
    cli.endpoint(o.url).model(o.model).timeout_seconds(o.timeout);
    if (!o.api_key.empty())            cli.api_key(o.api_key);
    if (!o.system_prompt.empty())      cli.system(o.system_prompt);
    if (o.temperature       >= 0.0f)   cli.temperature(o.temperature);
    if (o.top_p             >= 0.0f)   cli.top_p(o.top_p);
    if (o.top_k             >= 0)      cli.top_k(o.top_k);
    if (o.min_p             >= 0.0f)   cli.min_p(o.min_p);
    if (o.repeat_penalty    >  0.0f)   cli.repeat_penalty(o.repeat_penalty);
    if (o.frequency_penalty > -2.0f)   cli.frequency_penalty(o.frequency_penalty);
    if (o.presence_penalty  > -2.0f)   cli.presence_penalty(o.presence_penalty);
    if (o.seed              >= 0)      cli.seed(o.seed);
    if (o.max_tokens        >= 0)      cli.max_tokens(o.max_tokens);
    if (!o.stop_sequences.empty())     cli.stop(o.stop_sequences);
    if (!o.extra_body.empty())         cli.extra_body_json(o.extra_body);
    if (o.verbose)                     cli.verbose(true);

    // Management subcommands take precedence over chat — they don't
    // need tools registered, so handle them first.
    if (any_management(o)) return run_management(cli, o, st);

    easyai::Plan plan;
    register_tools(cli, plan, o, st);
    wire_callbacks(cli, plan, o, st);

    if (!o.prompt.empty()) return run_one(cli, o.prompt, st);
    return run_repl(cli, plan, st);
}
