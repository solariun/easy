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
#include "easyai/cli.hpp"
#include "easyai/client.hpp"
#include "easyai/log.hpp"
#include "easyai/plan.hpp"
#include "easyai/text.hpp"
#include "easyai/tool.hpp"
#include "easyai/ui.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <unistd.h>      // getpid
#include <vector>

namespace {

using easyai::ui::Style;
using easyai::ui::Spinner;
using easyai::ui::StreamStats;

// Shorthand: easyai::log::write tees stderr + the optional --log-file FILE.
// vlog(...) is just the historical name we kept for in-file readability.
inline void vlog(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
inline void vlog(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    // libstdc++ has no public va_list overload of write(); just expand
    // through a small buffer.  Logging volume is low (per-hop summaries),
    // so this is fine.
    char buf[4096];
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    easyai::log::write("%s", buf);
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
using easyai::text::slurp_file;

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
    bool        allow_bash      = false;       // opt-in: register `bash` tool
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
    bool        quiet            = false;  // --quiet/-q: disable spinner + ctx-% gauge
                                            // (batch / scripted / service usage)
    std::string log_file_path;             // explicit --log-file override
    int         max_reasoning    = 0;      // 0 = unlimited (disable runaway abort)
    bool        retry_on_incomplete = true;    // matches libeasyai-cli default; --no-retry-on-incomplete to opt out
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
"                                 fs_grep, fs_write_file, bash,\n"
"                                 system_meminfo, system_loadavg,\n"
"                                 system_cpu_usage, system_swaps\n"
"                               default: datetime,plan,web_search,web_fetch,\n"
"                                 system_meminfo,system_loadavg,\n"
"                                 system_cpu_usage,system_swaps\n"
"    --sandbox DIR              enable fs_list_dir, fs_read_file,\n"
"                                 fs_glob, fs_grep AND fs_write_file,\n"
"                                 ALL scoped to DIR.  Without --sandbox\n"
"                                 the model has no file access.\n"
"    --allow-bash               register the `bash` tool (run shell\n"
"                                 commands).  WARNING: NOT a hardened\n"
"                                 sandbox — the command runs with your\n"
"                                 user privileges (network, full FS, etc).\n"
"                                 cwd is set to --sandbox DIR if given,\n"
"                                 otherwise the current working dir.\n"
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
"    --max-reasoning N          abort the SSE stream when this turn's\n"
"                                accumulated reasoning_content exceeds N\n"
"                                chars.  Useful for chatty thinking models\n"
"                                that fall into long deliberation loops on\n"
"                                niche questions.  0 = unlimited (default).\n"
"    --no-retry-on-incomplete   disable the auto-retry-with-nudge for\n"
"                                incomplete turns (default: ON).  When the\n"
"                                server flags a turn as incomplete\n"
"                                (timings.incomplete=true — model produced\n"
"                                no tool_call AND only a tiny reply, e.g.\n"
"                                'I'll search…'), the client by default\n"
"                                drops that turn, appends a corrective user\n"
"                                nudge, and re-issues ONCE.  Use this flag\n"
"                                if you want the raw incomplete signal\n"
"                                without recovery.\n"
"    --retry-on-incomplete      legacy alias for the now-default behaviour;\n"
"                                kept for backwards compatibility, no-op.\n"
"    --verbose                  log HTTP+SSE traffic to stderr (timestamps +\n"
"                                per-piece diagnostics).  When set,\n"
"                                ALSO writes the same diagnostics PLUS\n"
"                                the raw HTTP transaction (request body,\n"
"                                every SSE chunk byte-for-byte, every\n"
"                                tool dispatch input/output) into a log\n"
"                                file at /tmp/easyai-cli-{pid}-{epoch}.log.\n"
"                                The path is printed at startup.  Override\n"
"                                with --log-file PATH.\n"
"    -q, --quiet                disable the spinner glyph + context-fill\n"
"                                gauge (e.g. |45%%).  Use for batch / scripted\n"
"                                runs where stdout is captured.  Streamed\n"
"                                content + tool markers still print; only\n"
"                                the in-place spinner is suppressed.\n"
"    --log-file PATH            write the raw transaction log here instead\n"
"                                of the auto-generated /tmp path.  Implies\n"
"                                --verbose if --verbose wasn't passed.\n"
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
        else if (a == "--allow-bash")     o.allow_bash    = true;
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
        else if (a == "--max-reasoning")     o.max_reasoning      = std::stoi(need(i, "--max-reasoning"));
        else if (a == "--retry-on-incomplete")    o.retry_on_incomplete = true;   // legacy no-op (now default)
        else if (a == "--no-retry-on-incomplete") o.retry_on_incomplete = false;
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--quiet"   || a == "-q") o.quiet   = true;
        else if (a == "--log-file")       o.log_file_path     = need(i, "--log-file");
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

// fs_* tools that are auto-enabled by --sandbox DIR (when --tools is
// empty).  Passing --sandbox is the explicit "give the model file
// access scoped here" gesture; without --tools the docs promise
// "enable fs_* tools" — this set is what that means.  fs_write_file
// is included: --sandbox is the user's explicit write authorisation.
const std::vector<std::string> kSandboxFsTools = {
    "fs_list_dir", "fs_read_file", "fs_glob", "fs_grep", "fs_write_file",
};

void register_tools(easyai::Client & cli,
                    easyai::Plan & plan,
                    const Options & o,
                    const Style & st) {
    auto wants = [&](const std::string & name) {
        if (o.tools_enabled.empty()) {
            for (const auto & d : kDefaultTools) if (d == name) return true;
            if (!o.sandbox.empty()) {
                for (const auto & d : kSandboxFsTools) if (d == name) return true;
            }
            // bash is opt-in by --allow-bash (NOT auto-enabled by --sandbox).
            if (o.allow_bash && name == "bash") return true;
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

    // bash — same root as fs_*; opt-in via --allow-bash or --tools bash.
    if (wants("bash")) {
        cli.add_tool(easyai::tools::bash(root));
        // Bash flows naturally span many hops (compile → run → fix →
        // re-run → grep logs → …); the default 8-hop cap chokes them.
        // Bump to effectively-unlimited.  Other safety nets (per-tool
        // timeouts, output caps, retry_on_incomplete) still apply.
        cli.max_tool_hops(99999);
    }

    // Inline system-info tools — defined above in `namespace systools`.
    // They demonstrate how to add your own custom Tool with a couple of
    // lines using Tool::builder().
    if (wants("system_meminfo"))   cli.add_tool(systools::make_system_meminfo());
    if (wants("system_loadavg"))   cli.add_tool(systools::make_system_loadavg());
    if (wants("system_cpu_usage")) cli.add_tool(systools::make_system_cpu_usage());
    if (wants("system_swaps"))     cli.add_tool(systools::make_system_swaps());

    if (o.verbose) {
        vlog("%s[easyai-cli-remote]%s registered %zu tool(s):\n",
             st.dim(), st.reset(), cli.tools().size());
        for (const auto & t : cli.tools()) {
            // Squash the JSON schema down to a single line for the log.
            std::string schema = t.parameters_json;
            for (char & c : schema) if (c == '\n' || c == '\r') c = ' ';
            // Collapse runs of spaces to one, just for readability.
            std::string compact;
            compact.reserve(schema.size());
            bool prev_space = false;
            for (char c : schema) {
                if (c == ' ') {
                    if (!prev_space) compact.push_back(' ');
                    prev_space = true;
                } else {
                    compact.push_back(c);
                    prev_space = false;
                }
            }
            // Trim long descriptions in the log so the line stays
            // scannable; the model still sees the full text.
            std::string desc = t.description;
            for (char & c : desc) if (c == '\n' || c == '\r') c = ' ';
            if (desc.size() > 120) desc = desc.substr(0, 120) + "…";
            vlog("%s  - %s%s%s  desc=\"%s\"\n",
                 st.dim(),
                 st.bold(), t.name.c_str(), st.reset(),
                 desc.c_str());
            vlog("%s    schema=%s%s\n",
                 st.dim(), compact.c_str(), st.reset());
        }
    }
}

// Trim a string to N chars with ellipsis suffix for log lines.
using easyai::text::trim_for_log;

using easyai::ui::print_tool_row;

int run_management(easyai::Client & cli, const Options & o, const Style & st) {
    // Thin dispatcher — the actual work lives in easyai::cli helpers
    // (see include/easyai/cli.hpp).  Each helper returns the process
    // exit code (0 on success, 1 on transport / parse failure).
    if (o.list_models)        return easyai::cli::print_models       (cli, st);
    if (o.list_tools)         return easyai::cli::print_local_tools  (cli, st);
    if (o.list_remote_tools)  return easyai::cli::print_remote_tools (cli, st);
    if (o.health)             return easyai::cli::print_health       (cli, st);
    if (o.props)              return easyai::cli::print_props        (cli);
    if (o.metrics)            return easyai::cli::print_metrics      (cli);
    if (!o.set_preset.empty())return easyai::cli::set_preset         (cli, o.set_preset, st);
    return 0;
}

// ---- callback wiring ------------------------------------------------------
using easyai::ui::render_plan;

// Per-turn streaming wiring is set up directly inside run_one() now,
// using easyai::ui::Streaming.  No globals — the Spinner, StreamStats
// and Streaming objects all live on the run_one stack frame and are
// torn down cleanly at end of turn.

// ---- repl helpers ---------------------------------------------------------
bool is_special(const std::string & line, const std::string & cmd) {
    return line == cmd
        || (line.size() > cmd.size() && line.rfind(cmd + " ", 0) == 0);
}

using easyai::text::prompt_wants_file_write;

using easyai::cli::client_has_tool;

int run_one(easyai::Client & cli, easyai::Plan & plan,
            const std::string & prompt,
            const Options & o, const Style & st) {
    // Tip targets newcomers who've forgotten --sandbox.  If the user
    // already passed --sandbox we stay quiet — they know about it; the
    // missing tool would be from an explicit --tools filter, where the
    // tip's "pass --sandbox" advice is wrong anyway.
    // The factory is `easyai::tools::fs_write_file`, but the tool it
    // builds is registered with the model under the bare name
    // `write_file` (see builtin_tools.cpp).  Use the registered name.
    if (o.sandbox.empty()
        && prompt_wants_file_write(prompt)
        && !client_has_tool(cli, "write_file")) {
        std::fprintf(stderr,
            "%s[easyai-cli-remote] tip:%s your prompt looks like it wants "
            "the model to write a file, but fs_write_file is NOT registered. "
            "Pass %s--sandbox DIR%s to give the model file read+write access "
            "scoped to DIR (or %s--tools fs_write_file%s explicitly). "
            "Without it the model will research, then stall when it tries "
            "to save and finds no write tool.\n",
            st.yellow(), st.reset(),
            st.bold(),  st.reset(),
            st.bold(),  st.reset());
    }

    // --quiet/-q disables the in-place spinner glyph + ctx-% gauge.
    // Streamed content + tool markers still print; only the cursor-
    // tracking decoration is suppressed, which is what batch / scripted
    // / service consumers want (no \b dance polluting captured stdout).
    Spinner     spinner(/*enabled=*/!o.quiet);
    StreamStats stats;  stats.reset();

    // Attach the canonical streaming UX (spinner-locked content + dim
    // reasoning + 🔧/✗ tool markers + live plan render) to the client
    // and plan.  Lifetime: this Streaming object outlives cli.chat()
    // below, which is the only context in which the lambdas fire.
    easyai::ui::Streaming(spinner, stats, st)
        .show_reasoning(o.show_reasoning)
        .verbose       (o.verbose)
        .attach        (cli)
        .attach        (plan);

    spinner.initial_draw();
    spinner.start_heartbeat();

    std::string answer = cli.chat(prompt);

    spinner.stop_heartbeat();
    spinner.finish();
    std::fputc('\n', stdout);
    if (answer.empty() && !cli.last_error().empty()) {
        std::fprintf(stderr, "%serror:%s %s\n", st.red(), st.reset(),
                     cli.last_error().c_str());
        return 1;
    }
    // Single placeholder path, fed by the same `timings.incomplete`
    // signal the webui consumes — the two surfaces report the same
    // diagnosis for the same turn.  Triggers when the server flagged
    // the turn (no tool_call, content < 80 bytes) OR the answer
    // string came back empty for any other reason.  When
    // --retry-on-incomplete was on this only fires AFTER the retry
    // also failed.
    if (answer.empty() || cli.last_turn_was_incomplete()) {
        std::fprintf(stdout,
            "%s(incomplete response — the model produced no tool_call and "
            "only a tiny visible reply, AND the auto-retry with corrective "
            "nudge ALSO failed%s. The model is repeatedly announcing a "
            "tool without emitting it. Try rephrasing more specifically "
            "(e.g. \"use write_file to save X to Y\"), shorten the prompt, "
            "or check the tool list — a missing tool the model thinks it "
            "needs can produce this loop.)%s\n",
            st.yellow(),
            o.retry_on_incomplete ? "" : " (auto-retry is OFF)",
            st.reset());
    }
    return 0;
}

int run_repl(easyai::Client & cli, easyai::Plan & plan,
             const Options & o, const Style & st) {
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

        run_one(cli, plan, line, o, st);
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) { usage(argv[0]); return 2; }
    Style st = easyai::ui::detect_style();

    // Validate --sandbox up front so the user gets a clear error
    // BEFORE the agent loop starts hitting "No such file or directory".
    if (std::string err; !easyai::cli::validate_sandbox(o.sandbox, err)) {
        std::fprintf(stderr, "%serror:%s %s\n",
                     st.red(), st.reset(), err.c_str());
        return 2;
    }

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

    // --log-file implies --verbose so the user gets the full diagnostic
    // stream (otherwise the file would only carry the wire-level RAW data
    // and miss CLI-side context).
    if (!o.log_file_path.empty()) o.verbose = true;

    // Open the diagnostic log file when --verbose (or --log-file) is on.
    // open_log_tee handles auto-path (/tmp/easyai-cli-remote-<pid>-<epoch>.log),
    // header (timestamp + pid + argv), and registers the FILE* as the
    // sink for easyai::log so vlog() tees here.  libeasyai-cli also
    // writes raw SSE here via cli.log_file(log_fp) below.
    std::string resolved_log_path;
    std::FILE * log_fp = nullptr;
    if (o.verbose) {
        log_fp = easyai::cli::open_log_tee(
            o.log_file_path, "easyai-cli-remote",
            argc, argv, &resolved_log_path);
        if (!log_fp) {
            std::fprintf(stderr,
                "%swarning:%s could not open log file %s — continuing without raw log.\n",
                st.yellow(), st.reset(), resolved_log_path.c_str());
        } else {
            std::fprintf(stderr,
                "%s[easyai-cli-remote]%s raw transaction log: %s%s%s\n",
                st.dim(), st.reset(),
                st.bold(), resolved_log_path.c_str(), st.reset());
        }
    }

    easyai::Client cli;
    if (log_fp) cli.log_file(log_fp);
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
    if (o.max_reasoning   > 0)         cli.max_reasoning_chars(o.max_reasoning);
    if (o.retry_on_incomplete)         cli.retry_on_incomplete(true);

    // Plan + tools registered up-front so --list-tools (which prints the
    // LOCAL catalog this CLI sends to the model) can show them and the
    // chat path / REPL also has them ready.
    easyai::Plan plan;
    register_tools(cli, plan, o, st);

    auto close_log_fp = [&]() {
        easyai::cli::close_log_tee(log_fp);
        log_fp = nullptr;
    };

    int rc;
    if (any_management(o)) {
        rc = run_management(cli, o, st);
    } else {
        // Streaming wiring happens INSIDE run_one (per-turn) now —
        // no longer a one-shot wire_callbacks at the top level.
        if (!o.prompt.empty()) rc = run_one(cli, plan, o.prompt, o, st);
        else                   rc = run_repl(cli, plan, o, st);
    }
    close_log_fp();
    return rc;
}
