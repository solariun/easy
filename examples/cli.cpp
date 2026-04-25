// =============================================================================
//  easyai-cli — drop-in REPL like llama-cli, but with the easyai toolbelt
//               loaded by default and friendly preset commands.
//
//  Usage:
//
//    easyai-cli -m model.gguf
//    easyai-cli -m model.gguf -s system.txt
//    easyai-cli -m model.gguf -ngl 99 --no-tools
//
//  Inside the REPL, every line you type goes to the model, EXCEPT lines that
//  start with a recognised preset/temperature command:
//
//      precise            (sticky: stays in 'precise' until next change)
//      creative 0.9       (set 'creative' preset, override temp to 0.9)
//      /temp 0.5          (just bump temperature)
//      /reset             (clear conversation history)
//      /tools             (list registered tools)
//      /system <text>     (replace system prompt; clears history)
//      /quit | /exit      (leave)
//
//  Single empty line submits the prompt as-is.  Two blank lines exit.
//
//  The CLI deliberately uses RAII for every owned resource and never touches
//  raw new/delete — see comments inline for the memory hygiene rationale.
// =============================================================================
#include "easyai/easyai.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

// ---------------------------------------------------------------------------
// Read an entire small text file into a string. Returns empty string on any
// failure (caller decides whether that's fatal).
//
// We intentionally cap the size at 1 MiB so that pointing -s at a huge file by
// accident can't OOM the engine.
// ---------------------------------------------------------------------------
std::string read_text_file(const std::string & path, size_t max_bytes = 1u << 20) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    if (sz <= 0) return {};
    if ((size_t) sz > max_bytes) sz = (std::streamsize) max_bytes;
    f.seekg(0, std::ios::beg);
    std::string out((size_t) sz, '\0');
    f.read(out.data(), sz);
    out.resize((size_t) f.gcount());
    return out;
}

// ---------------------------------------------------------------------------
// Pretty-print all built-in presets so users discover them.
// ---------------------------------------------------------------------------
void print_presets() {
    std::fprintf(stderr, "\nAvailable presets:\n");
    for (const auto & p : easyai::all_presets()) {
        std::fprintf(stderr, "  %-14s  temp=%.2f top_p=%.2f top_k=%d  — %s\n",
                     p.name.c_str(), p.temperature, p.top_p, p.top_k,
                     p.description.c_str());
    }
    std::fprintf(stderr, "\nAlso: 'temp <number>', '/reset', '/tools', '/system <text>', '/quit'\n\n");
}

// ---------------------------------------------------------------------------
// Ctrl-C handling.  We trap SIGINT once so the user can interrupt a streaming
// generation without killing the process.  A second SIGINT inside one second
// exits.  We do NOT touch the engine from the handler — we only flip an
// std::atomic flag the main loop polls.
// ---------------------------------------------------------------------------
static std::atomic<bool>    g_interrupt{false};
static std::atomic<int64_t> g_last_sigint_ms{0};

void handle_sigint(int) {
    using namespace std::chrono;
    auto now = duration_cast<milliseconds>(
                   steady_clock::now().time_since_epoch()).count();
    if (now - g_last_sigint_ms.load() < 1000) {
        // double-tap → terminate
        std::fprintf(stderr, "\n[easyai] interrupted twice — exiting\n");
        std::_Exit(130);
    }
    g_last_sigint_ms.store(now);
    g_interrupt.store(true);
}

void install_sigint() {
    std::signal(SIGINT, handle_sigint);
}

// ---------------------------------------------------------------------------
// Argument parsing kept deliberately simple — we don't pull in any CLI lib.
// ---------------------------------------------------------------------------
struct CliArgs {
    std::string model_path;
    std::string system_path;
    std::string system_inline;
    int         n_ctx       = 4096;
    int         ngl         = -1;
    int         n_threads   = 0;
    bool        load_tools  = true;
    std::string sandbox     = ".";
    std::string preset      = "balanced";
};

[[noreturn]] void die_usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s -m model.gguf [options]\n"
        "  -m, --model <path>          GGUF model file (required)\n"
        "  -s, --system-file <path>    Read system prompt from a text file\n"
        "      --system <text>         Inline system prompt\n"
        "  -c, --ctx <n>               Context size (default 4096)\n"
        "      --ngl <n>               GPU layers (-1=auto, 0=CPU)\n"
        "  -t, --threads <n>           CPU threads\n"
        "      --no-tools              Don't register the built-in toolbelt\n"
        "      --sandbox <dir>         Root for fs_* tools (default '.')\n"
        "      --preset <name>         Initial preset (default 'balanced')\n"
        "  -h, --help                  Show this help and exit\n",
        argv0);
    std::exit(1);
}

CliArgs parse(int argc, char ** argv) {
    CliArgs a;
    auto need = [&](int & i, const char * flag) -> const char * {
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); die_usage(argv[0]); }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "-m" || s == "--model")        a.model_path    = need(i, "-m");
        else if (s == "-s" || s == "--system-file")  a.system_path   = need(i, "-s");
        else if (s == "--system")                    a.system_inline = need(i, "--system");
        else if (s == "-c" || s == "--ctx")          a.n_ctx         = std::atoi(need(i, "-c"));
        else if (s == "--ngl")                       a.ngl           = std::atoi(need(i, "--ngl"));
        else if (s == "-t" || s == "--threads")      a.n_threads     = std::atoi(need(i, "-t"));
        else if (s == "--no-tools")                  a.load_tools    = false;
        else if (s == "--sandbox")                   a.sandbox       = need(i, "--sandbox");
        else if (s == "--preset")                    a.preset        = need(i, "--preset");
        else if (s == "-h" || s == "--help")         die_usage(argv[0]);
        else { std::fprintf(stderr, "unknown arg: %s\n", s.c_str()); die_usage(argv[0]); }
    }
    if (a.model_path.empty()) die_usage(argv[0]);
    return a;
}

}  // namespace

int main(int argc, char ** argv) {
    CliArgs args = parse(argc, argv);
    install_sigint();

    // Resolve the system prompt: --system inline beats -s file beats nothing.
    std::string system_prompt = args.system_inline;
    if (system_prompt.empty() && !args.system_path.empty()) {
        system_prompt = read_text_file(args.system_path);
        if (system_prompt.empty()) {
            std::fprintf(stderr, "[easyai-cli] WARNING: failed to read system file '%s'\n",
                         args.system_path.c_str());
        }
    }

    // The engine owns all native resources via RAII — when it goes out of
    // scope at the end of main() the model, sampler, KV cache, and templates
    // are released in the correct order.
    easyai::Engine engine;
    engine.model       (args.model_path)
          .context     (args.n_ctx)
          .gpu_layers  (args.ngl)
          .system      (system_prompt)
          .verbose     (false)
          .on_token([](const std::string & p){ std::cout << p << std::flush; })
          .on_tool ([](const easyai::ToolCall & c, const easyai::ToolResult & r){
              std::fprintf(stderr,
                  "\n\033[36m[tool] %s -> %s%.200s%s\033[0m\n",
                  c.name.c_str(),
                  r.is_error ? "ERR " : "",
                  r.content.c_str(),
                  r.content.size() > 200 ? "…" : "");
          });
    if (args.n_threads > 0) engine.threads(args.n_threads);

    // Apply the initial preset if it exists.
    if (const easyai::Preset * p0 = easyai::find_preset(args.preset)) {
        engine.temperature(p0->temperature)
              .top_p      (p0->top_p)
              .top_k      (p0->top_k)
              .min_p      (p0->min_p);
    }

    // Default toolbelt — opt-out via --no-tools.
    if (args.load_tools) {
        engine.add_tool(easyai::tools::datetime())
              .add_tool(easyai::tools::web_fetch())
              .add_tool(easyai::tools::web_search())
              .add_tool(easyai::tools::fs_list_dir (args.sandbox))
              .add_tool(easyai::tools::fs_read_file(args.sandbox))
              .add_tool(easyai::tools::fs_glob     (args.sandbox))
              .add_tool(easyai::tools::fs_grep     (args.sandbox));
    }

    if (!engine.load()) {
        std::fprintf(stderr, "[easyai-cli] load failed: %s\n", engine.last_error().c_str());
        return 1;
    }

    std::fprintf(stderr,
        "[easyai-cli] loaded %s\n"
        "             backend=%s  ctx=%d  tools=%zu  preset=%s\n"
        "             type '/help' for commands, '/quit' to exit\n",
        args.model_path.c_str(), engine.backend_summary().c_str(),
        engine.n_ctx(), engine.tools().size(), args.preset.c_str());

    // ---------------------------------------------------------------------
    // REPL — line-buffered.  We read until EOF or /quit.
    //
    // Memory note: we re-use the same `line` std::string each iteration; it
    // grows to its largest seen capacity and stays there. No heap thrash.
    // ---------------------------------------------------------------------
    std::string line;
    while (true) {
        std::cout << "\n\033[32m> \033[0m" << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        // -------- meta-commands ----------------------------------------
        if (line == "/quit" || line == "/exit") break;
        if (line == "/help" || line == "/?")    { print_presets(); continue; }
        if (line == "/reset")                   { engine.clear_history(); std::cout << "[history cleared]\n"; continue; }
        if (line == "/tools") {
            for (const auto & t : engine.tools()) {
                std::cout << "  " << t.name << " — " << t.description << "\n";
            }
            continue;
        }
        if (line.rfind("/system ", 0) == 0) {
            std::string s = line.substr(8);
            engine.system(s);
            engine.clear_history();
            std::cout << "[system prompt updated; history cleared]\n";
            continue;
        }

        // -------- preset / temperature command --------------------------
        easyai::PresetResult pr = easyai::parse_preset(line);
        if (!pr.applied.empty()) {
            engine.set_sampling(pr.temperature, pr.top_p, pr.top_k, pr.min_p);
            std::fprintf(stderr, "[preset → %s]\n", pr.applied.c_str());
            // If the user *only* typed the preset, don't generate; otherwise
            // continue with whatever follows the preset words.
            if (pr.consumed >= line.size()) continue;
            line = line.substr(pr.consumed);
        }

        // -------- normal generation ------------------------------------
        g_interrupt.store(false);
        std::cout << "\033[33m";
        try {
            engine.chat(line);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "\n[easyai-cli] generation error: %s\n", e.what());
        }
        std::cout << "\033[0m" << std::endl;
    }

    std::fprintf(stderr, "[easyai-cli] bye\n");
    return 0;
}
