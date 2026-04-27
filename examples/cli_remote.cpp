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
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>      // isatty
#include <vector>

namespace {

// ---- TokenSpinner — '|/-\\' frames overwriting via \b ---------------------
//
// Ticks once per stream callback (token / reasoning / tool round-trip)
// when nothing visible has been printed for the current turn.  As soon
// as the first visible byte goes out, the spinner is erased and never
// re-shown for that turn.  Auto-disabled outside a TTY.
class TokenSpinner {
   public:
    explicit TokenSpinner(bool en)
        : enabled_(en && ::isatty(fileno(stdout)) != 0) {}

    void tick_silent() {
        if (!enabled_ || ever_printed_) return;
        static const char frames[] = { '|', '/', '-', '\\' };
        if (active_) std::fputc('\b', stdout);
        std::fputc(frames[frame_ % 4], stdout);
        std::fflush(stdout);
        active_ = true;
        ++frame_;
    }
    void about_to_print() {
        if (!enabled_) return;
        if (active_) { std::fputs("\b \b", stdout); active_ = false; }
        ever_printed_ = true;
    }
    void finish() {
        if (active_) { std::fputs("\b \b", stdout); active_ = false; }
        ever_printed_ = false;
        frame_        = 0;
    }

   private:
    bool enabled_      = false;
    bool active_       = false;
    bool ever_printed_ = false;
    int  frame_        = 0;
};

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

// ===========================================================================
// Inline system-info tools — demonstrate how to wire your own custom
// Tool right inside the CLI.  All four are Linux-specific (read /proc),
// they return a clear "Linux only" error on macOS / *BSD.  Hooking
// these into the model gives it observability over the host running
// the agent: "is this box paging?", "is one core saturated?", etc.
//
// Cookbook for adding your own:
//   1. Build an easyai::Tool with Tool::builder("name").describe(...)
//      .param(...).handle([](const ToolCall &){ ... }).build()
//   2. Pass it to cli.add_tool().  That's it.
//
// The model sees `name` + `description` + `parameters` (auto-generated
// from .param() calls) and decides when to call it.  Your handler runs
// in this process when the model invokes it; whatever you return as
// ToolResult::ok(text) becomes the tool message the model sees next.
// ===========================================================================

namespace systools {

// ---- /proc parsing helpers ------------------------------------------------
bool slurp_file(const std::string & path, std::string & out) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss; ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Parse "key: NUM unit\n" lines into a kB-valued map (kB is the unit
// /proc/meminfo always uses, despite the "kB" suffix).
std::map<std::string, long long> parse_proc_meminfo(const std::string & text) {
    std::map<std::string, long long> kv;
    std::stringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key  = line.substr(0, colon);
        std::string rest = line.substr(colon + 1);
        try { kv[key] = std::stoll(rest); }
        catch (...) { /* skip malformed line */ }
    }
    return kv;
}

// Per-cpu cumulative ticks from a single /proc/stat line.
struct CpuTicks {
    std::string  label;     // "cpu", "cpu0", "cpu1", …
    long long    user      = 0;
    long long    nice      = 0;
    long long    system    = 0;
    long long    idle      = 0;
    long long    iowait    = 0;
    long long    irq       = 0;
    long long    softirq   = 0;
    long long    steal     = 0;
    long long    total() const { return user + nice + system + idle + iowait + irq + softirq + steal; }
    long long    busy()  const { return total() - idle - iowait; }
};

std::vector<CpuTicks> parse_proc_stat(const std::string & text) {
    std::vector<CpuTicks> out;
    std::stringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.size() < 3 || line.compare(0, 3, "cpu") != 0) continue;
        CpuTicks t;
        std::stringstream ls(line);
        ls >> t.label
           >> t.user >> t.nice >> t.system >> t.idle
           >> t.iowait >> t.irq >> t.softirq >> t.steal;
        out.push_back(t);
    }
    return out;
}

// ---- Tool factories -------------------------------------------------------
easyai::Tool make_system_meminfo() {
    return easyai::Tool::builder("system_meminfo")
        .describe("Return total / available / free / buffers / cached memory and "
                  "swap totals from /proc/meminfo, in MiB.  Linux only.  No args.")
        .handle([](const easyai::ToolCall &) -> easyai::ToolResult {
            std::string raw;
            if (!slurp_file("/proc/meminfo", raw))
                return easyai::ToolResult::error("/proc/meminfo unreadable (Linux only)");
            auto kv = parse_proc_meminfo(raw);
            auto get = [&](const char * k) -> long long {
                auto it = kv.find(k); return it == kv.end() ? 0 : it->second;
            };
            std::ostringstream out;
            out << "Memory (MiB):\n"
                << "  Total:     " << get("MemTotal")     / 1024 << "\n"
                << "  Available: " << get("MemAvailable") / 1024 << "\n"
                << "  Free:      " << get("MemFree")      / 1024 << "\n"
                << "  Buffers:   " << get("Buffers")      / 1024 << "\n"
                << "  Cached:    " << get("Cached")       / 1024 << "\n"
                << "  Used (total - available): "
                <<     (get("MemTotal") - get("MemAvailable")) / 1024 << "\n"
                << "Swap (MiB):\n"
                << "  Total: " << get("SwapTotal") / 1024 << "\n"
                << "  Free:  " << get("SwapFree")  / 1024 << "\n"
                << "  Used:  " << (get("SwapTotal") - get("SwapFree")) / 1024 << "\n";
            return easyai::ToolResult::ok(out.str());
        }).build();
}

easyai::Tool make_system_loadavg() {
    return easyai::Tool::builder("system_loadavg")
        .describe("Return the 1, 5 and 15 minute load averages plus the "
                  "running/total process counter from /proc/loadavg.  Linux only.")
        .handle([](const easyai::ToolCall &) -> easyai::ToolResult {
            std::string raw;
            if (!slurp_file("/proc/loadavg", raw))
                return easyai::ToolResult::error("/proc/loadavg unreadable (Linux only)");
            float l1 = 0, l5 = 0, l15 = 0;
            char  procs[64] = {0};
            int   last_pid  = 0;
            std::sscanf(raw.c_str(), "%f %f %f %63s %d",
                        &l1, &l5, &l15, procs, &last_pid);
            std::ostringstream out;
            out << "Load average:\n"
                << "  1m:  " << l1  << "\n"
                << "  5m:  " << l5  << "\n"
                << "  15m: " << l15 << "\n"
                << "  running/total: " << procs << "\n"
                << "  last pid:      " << last_pid << "\n";
            return easyai::ToolResult::ok(out.str());
        }).build();
}

easyai::Tool make_system_cpu_usage() {
    return easyai::Tool::builder("system_cpu_usage")
        .describe("Sample /proc/stat twice with a configurable gap and report "
                  "per-CPU busy% (1.0 = 100% saturated).  Useful when the "
                  "user asks 'how loaded is the box right now'.  Linux only.")
        .param("sample_ms", "integer",
               "Window between samples in milliseconds.  Default 200, max 2000.",
               false)
        .handle([](const easyai::ToolCall & call) -> easyai::ToolResult {
            long long sample = easyai::args::get_int_or(
                call.arguments_json, "sample_ms", 200);
            if (sample < 50)   sample = 50;
            if (sample > 2000) sample = 2000;

            std::string a;
            if (!slurp_file("/proc/stat", a))
                return easyai::ToolResult::error("/proc/stat unreadable (Linux only)");
            std::this_thread::sleep_for(std::chrono::milliseconds(sample));
            std::string b;
            if (!slurp_file("/proc/stat", b))
                return easyai::ToolResult::error("/proc/stat unreadable on second sample");

            auto va = parse_proc_stat(a);
            auto vb = parse_proc_stat(b);
            // Index by label so we don't rely on order.
            std::map<std::string, CpuTicks> by_label;
            for (const auto & c : va) by_label[c.label] = c;

            std::ostringstream out;
            out << "CPU usage over " << sample << " ms:\n";
            for (const auto & y : vb) {
                auto it = by_label.find(y.label);
                if (it == by_label.end()) continue;
                long long dt   = y.total() - it->second.total();
                long long dbsy = y.busy()  - it->second.busy();
                if (dt <= 0) continue;
                double pct = double(dbsy) / double(dt);
                out << "  " << y.label << ": "
                    << int(pct * 100.0 + 0.5) << "%\n";
            }
            return easyai::ToolResult::ok(out.str());
        }).build();
}

easyai::Tool make_system_swaps() {
    return easyai::Tool::builder("system_swaps")
        .describe("List configured swap devices/files with size and used "
                  "amount, from /proc/swaps.  Linux only.")
        .handle([](const easyai::ToolCall &) -> easyai::ToolResult {
            std::string raw;
            if (!slurp_file("/proc/swaps", raw))
                return easyai::ToolResult::error("/proc/swaps unreadable (Linux only)");
            return easyai::ToolResult::ok(raw);
        }).build();
}

}  // namespace systools

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
    bool        show_reasoning   = true;   // default ON; --no-reasoning to opt out
    bool        verbose          = false;
    bool        no_plan          = false;     // skip auto-registering Plan
    bool        tls_insecure     = false;     // skip peer cert verification
    std::string tls_ca_path;                  // PEM bundle for custom CAs

    // Management subcommands (mutually exclusive with prompt mode).
    bool        list_models       = false;
    bool        list_tools        = false;    // local tools (registered in this process)
    bool        list_remote_tools = false;    // server tools (GET /v1/tools)
    bool        health            = false;
    bool        props             = false;
    bool        metrics           = false;
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
"    --insecure-tls             skip peer cert check (https only — DEV ONLY)\n"
"    --ca-cert PATH             trust this CA bundle (PEM) for https://\n"
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
"    --tools LIST               comma list, valid names:\n"
"                                 datetime, plan, web_search, web_fetch,\n"
"                                 fs_read_file, fs_list_dir, fs_glob,\n"
"                                 fs_grep, fs_write_file,\n"
"                                 system_meminfo, system_loadavg,\n"
"                                 system_cpu_usage, system_swaps\n"
"                               default: datetime,plan,web_search,web_fetch,\n"
"                                 system_meminfo,system_loadavg,\n"
"                                 system_cpu_usage,system_swaps\n"
"    --sandbox DIR              enable fs_* tools, scoped to DIR\n"
"    --no-plan                  don't auto-register the planning tool\n"
"\n"
"  Behaviour:\n"
"    -p, --prompt TEXT          one-shot prompt; without this you get a REPL\n"
"                               (you can also pass the prompt as a positional\n"
"                                arg, or pipe it via stdin)\n"
"    --no-reasoning             hide delta.reasoning_content (default: shown\n"
"                                inline in dim grey).  --hide-reasoning is\n"
"                                an alias.  --show-reasoning is now a no-op\n"
"                                (kept for backwards compat).\n"
"    --verbose                  log HTTP+SSE traffic to stderr (timestamps +\n"
"                                per-piece diagnostics)\n"
"\n"
"  Management subcommands (use one, no chat):\n"
"    --list-tools               list LOCAL tools (registered in this CLI)\n"
"                               with their full descriptions — useful to\n"
"                               see exactly what the model will be told\n"
"    --list-remote-tools        GET /v1/tools — list server-side tools\n"
"                               (easyai-server extension, may not exist on\n"
"                               other OpenAI-compat servers)\n"
"    --list-models              GET /v1/models\n"
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
        else if (a == "--show-reasoning") o.show_reasoning = true;   // no-op now (kept for compat)
        else if (a == "--no-reasoning"
              || a == "--hide-reasoning") o.show_reasoning = false;
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--insecure-tls")   o.tls_insecure = true;
        else if (a == "--ca-cert")        o.tls_ca_path  = need(i, "--ca-cert");
        else if (a == "-p" || a == "--prompt") o.prompt = need(i, "--prompt");
        else if (a == "--list-models")    o.list_models       = true;
        else if (a == "--list-tools")     o.list_tools        = true;
        else if (a == "--list-remote-tools") o.list_remote_tools = true;
        else if (a == "--health")         o.health      = true;
        else if (a == "--props")          o.props       = true;
        else if (a == "--metrics")        o.metrics     = true;
        else if (a == "--set-preset")     o.set_preset  = need(i, "--set-preset");
        else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
        else if (!a.empty() && a[0] != '-' && o.prompt.empty()) {
            // Positional argument is treated as the one-shot prompt, so
            // `easyai-cli-remote --url ai.local "what date is today?"`
            // works without the explicit -p / --prompt flag.  Multiple
            // positionals get joined with a space.
            o.prompt = a;
            for (++i; i < argc; ++i) {
                std::string extra = argv[i];
                if (!extra.empty() && extra[0] != '-') {
                    o.prompt += " ";
                    o.prompt += extra;
                } else {
                    --i;
                    break;
                }
            }
        }
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
    return o.list_models || o.list_tools || o.list_remote_tools
        || o.health      || o.props      || o.metrics
        || !o.set_preset.empty();
}

// ---- tool registration ----------------------------------------------------
// Default catalog when --tools isn't given.
const std::vector<std::string> kDefaultTools = {
    "datetime", "plan", "web_search", "web_fetch",
    "system_meminfo", "system_loadavg", "system_cpu_usage", "system_swaps",
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

    // Inline system-info tools — defined above in `namespace systools`.
    // They demonstrate how to add your own custom Tool with a couple of
    // lines using Tool::builder().
    if (wants("system_meminfo"))   cli.add_tool(systools::make_system_meminfo());
    if (wants("system_loadavg"))   cli.add_tool(systools::make_system_loadavg());
    if (wants("system_cpu_usage")) cli.add_tool(systools::make_system_cpu_usage());
    if (wants("system_swaps"))     cli.add_tool(systools::make_system_swaps());

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
// Pretty-print a tool catalog with name + multi-line description.
void print_tool_row(const std::string & name,
                    const std::string & description,
                    const Style & st) {
    std::printf("%s%s%s\n", st.bold(), name.c_str(), st.reset());
    // Indent each line of the description by two spaces and dim it.
    const std::string & d = description;
    size_t i = 0;
    while (i < d.size()) {
        size_t nl = d.find('\n', i);
        std::string line = (nl == std::string::npos)
                               ? d.substr(i)
                               : d.substr(i, nl - i);
        std::printf("  %s%s%s\n", st.dim(), line.c_str(), st.reset());
        if (nl == std::string::npos) break;
        i = nl + 1;
    }
}

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
        // LOCAL tools — these are what cli-remote sends to the model in
        // the request body's `tools[]`.  Most useful for users since
        // it answers "what can the model actually do right now".
        if (cli.tools().empty()) {
            std::fprintf(stderr,
                "%sno tools registered.%s  Use --tools and/or --sandbox to "
                "enable some.\n", st.dim(), st.reset());
            return 0;
        }
        std::printf("%slocal tools (%zu):%s\n",
                    st.bold(), cli.tools().size(), st.reset());
        for (const auto & t : cli.tools()) {
            print_tool_row(t.name, t.description, st);
        }
        return 0;
    }
    if (o.list_remote_tools) {
        // Server-side catalog via /v1/tools (easyai-server extension).
        std::vector<easyai::RemoteTool> ts;
        if (!cli.list_remote_tools(ts)) {
            std::fprintf(stderr, "%serror:%s %s\n", st.red(), st.reset(),
                         cli.last_error().c_str());
            return 1;
        }
        std::printf("%sremote tools (%zu):%s\n",
                    st.bold(), ts.size(), st.reset());
        for (const auto & t : ts) {
            print_tool_row(t.name, t.description, st);
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

// One spinner instance per process — captured by the closures below
// so each stream tick advances the same animation.  We keep counters
// for the verbose summary (--verbose).
struct StreamStats {
    int      content_pieces  = 0;
    int      reason_pieces   = 0;
    int      tool_calls      = 0;
    int      tool_errors     = 0;
    long     ms_to_first_tok = -1;
    std::chrono::steady_clock::time_point started{};
    void reset() {
        content_pieces  = 0;
        reason_pieces   = 0;
        tool_calls      = 0;
        tool_errors     = 0;
        ms_to_first_tok = -1;
        started         = std::chrono::steady_clock::now();
    }
    long elapsed_ms() const {
        using namespace std::chrono;
        return (long) duration_cast<milliseconds>(
                   steady_clock::now() - started).count();
    }
};

static TokenSpinner * g_spinner = nullptr;
static StreamStats  * g_stats   = nullptr;

void wire_callbacks(easyai::Client & cli, easyai::Plan & plan,
                    const Options & o, const Style & st) {
    cli.on_token([&st, &o](const std::string & piece) {
        if (g_stats) {
            ++g_stats->content_pieces;
            if (g_stats->ms_to_first_tok < 0)
                g_stats->ms_to_first_tok = g_stats->elapsed_ms();
        }
        if (g_spinner) g_spinner->about_to_print();
        std::fputs(piece.c_str(), stdout);
        std::fflush(stdout);
        if (o.verbose) {
            std::fprintf(stderr, "%s[content %d, +%ldms]%s\n",
                         st.dim(),
                         g_stats ? g_stats->content_pieces : 0,
                         g_stats ? g_stats->elapsed_ms() : 0,
                         st.reset());
        }
    });
    cli.on_reason([&st, &o](const std::string & piece) {
        if (g_stats) ++g_stats->reason_pieces;
        if (o.show_reasoning) {
            if (g_spinner) g_spinner->about_to_print();
            std::fprintf(stdout, "%s%s%s",
                         st.dim(), piece.c_str(), st.reset());
            std::fflush(stdout);
        } else if (g_spinner) {
            // Hidden reasoning still ticks the spinner so the user
            // sees activity during long thinking phases.
            g_spinner->tick_silent();
        }
        if (o.verbose) {
            std::fprintf(stderr, "%s[reason %d, +%ldms]%s\n",
                         st.dim(),
                         g_stats ? g_stats->reason_pieces : 0,
                         g_stats ? g_stats->elapsed_ms() : 0,
                         st.reset());
        }
    });
    cli.on_tool([&st, &o](const easyai::ToolCall & call,
                           const easyai::ToolResult & result) {
        if (g_stats) {
            ++g_stats->tool_calls;
            if (result.is_error) ++g_stats->tool_errors;
        }
        if (g_spinner) g_spinner->about_to_print();
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
        if (o.verbose) {
            std::fprintf(stderr,
                "%s[tool %s%s name=%s args_bytes=%zu result_bytes=%zu +%ldms]%s\n",
                st.dim(),
                result.is_error ? "FAIL " : "ok ",
                call.name.c_str(),
                call.name.c_str(),
                call.arguments_json.size(),
                result.content.size(),
                g_stats ? g_stats->elapsed_ms() : 0,
                st.reset());
        }
    });
    plan.on_change([&st](const easyai::Plan & p) {
        if (g_spinner) g_spinner->about_to_print();
        render_plan(p, st);
    });
}

// ---- repl helpers ---------------------------------------------------------
bool is_special(const std::string & line, const std::string & cmd) {
    return line == cmd
        || (line.size() > cmd.size() && line.rfind(cmd + " ", 0) == 0);
}

int run_one(easyai::Client & cli, const std::string & prompt,
            const Style & st) {
    TokenSpinner spinner(/*en=*/true);
    StreamStats  stats;  stats.reset();
    g_spinner = &spinner;
    g_stats   = &stats;

    std::string answer = cli.chat(prompt);
    spinner.finish();
    g_spinner = nullptr;
    g_stats   = nullptr;
    std::fputc('\n', stdout);
    if (answer.empty() && !cli.last_error().empty()) {
        std::fprintf(stderr, "%serror:%s %s\n", st.red(), st.reset(),
                     cli.last_error().c_str());
        return 1;
    }
    // Empty answer with no error => the model finished without
    // emitting any content_delta.  This used to be a silent failure
    // (blank prompt waiting for a response that never came).  Render
    // a yellow placeholder so the user knows something happened, and
    // suggest --show-reasoning + journalctl as the diagnostic path.
    if (answer.empty()) {
        std::fprintf(stdout,
            "%s(empty response — model produced no visible content.  "
            "Re-run with --show-reasoning + --verbose, and check "
            "`journalctl -u easyai-server -f` on the box for "
            "[easyai-server] WARN empty response …)%s\n",
            st.yellow(), st.reset());
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

    // Auto-prepend http:// when --url omits a scheme — convenience for
    // local dev so `--url ai.local:8080` Just Works.  https endpoints
    // still need the explicit `https://` prefix.
    if (!o.url.empty()
        && o.url.compare(0, 7, "http://")  != 0
        && o.url.compare(0, 8, "https://") != 0) {
        o.url = "http://" + o.url;
    }

    // Pipe / heredoc input — when stdin isn't a TTY and no prompt was
    // given on the command line, read all of stdin as a one-shot
    // prompt.  Lets you do:
    //   echo "que dia eh hoje" | easyai-cli-remote --url ai.local
    //   easyai-cli-remote --url ai.local <<EOF
    //   ... long question ...
    //   EOF
    if (o.prompt.empty() && ::isatty(fileno(stdin)) == 0) {
        std::string buf, line;
        while (std::getline(std::cin, line)) {
            if (!buf.empty()) buf += "\n";
            buf += line;
        }
        if (!buf.empty()) o.prompt = std::move(buf);
    }

    // --list-tools is purely LOCAL — it prints the tools registered in
    // this CLI process, no network needed.  When that's the ONLY thing
    // requested, skip the --url requirement so `--list-tools` works on
    // its own without an endpoint argument.
    const bool only_local_listing =
        o.list_tools
        && !o.list_models && !o.list_remote_tools && !o.health
        && !o.props && !o.metrics && o.set_preset.empty();
    if (!only_local_listing && o.url.empty()) {
        std::fprintf(stderr, "%serror:%s --url (or EASYAI_URL) is required\n",
                     st.red(), st.reset());
        usage(argv[0]);
        return 2;
    }

    easyai::Client cli;
    if (!o.url.empty()) cli.endpoint(o.url);
    cli.model(o.model).timeout_seconds(o.timeout);
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
    if (o.tls_insecure)                cli.tls_insecure(true);
    if (!o.tls_ca_path.empty())        cli.ca_cert_path(o.tls_ca_path);

    // Plan + tools registered up-front so --list-tools (which prints the
    // LOCAL catalog this CLI sends to the model) can show them and the
    // chat path / REPL also has them ready.
    easyai::Plan plan;
    register_tools(cli, plan, o, st);

    if (any_management(o)) return run_management(cli, o, st);

    wire_callbacks(cli, plan, o, st);

    if (!o.prompt.empty()) return run_one(cli, o.prompt, st);
    return run_repl(cli, plan, st);
}
