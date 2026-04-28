// easyai/ui.hpp — terminal UI helpers shared by the example CLIs.
//
// All optional and side-effect-free until you actually call them: a
// Style with color=false renders to no escape codes; a Spinner created
// with enabled=false is a no-op object.  Safe to use on non-TTY
// stdout (the auto-detected Style + Spinner both notice and stay
// silent).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

namespace easyai::ui {

// ---------- ANSI styling ----------------------------------------------------
//
// Each helper returns either the escape sequence or "" when color is
// off — so you can sprinkle them in printfs without branching:
//
//   std::printf("%serror:%s %s\n", st.red(), st.reset(), msg);
//
// detect_style() turns color on iff stdout is a TTY and NO_COLOR is unset.
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

// Honours the NO_COLOR convention (https://no-color.org).
Style detect_style();


// ---------- streaming spinner ----------------------------------------------
//
// A 1-glyph progress indicator that lives at the END of stdout, gets
// erased before each callback writes, and reappears after — so the
// agent's streamed text reads cleanly.  A separate heartbeat thread
// keeps the glyph animating during dead air (waiting on first SSE
// byte, hidden reasoning, slow tool calls).
//
// Usage:
//   easyai::ui::Spinner sp(/*enabled=*/true);
//   sp.initial_draw();
//   sp.start_heartbeat();
//   // ... run agent loop, callbacks call sp.write(text) ...
//   sp.stop_heartbeat();
//   sp.finish();
class Spinner {
public:
    explicit Spinner(bool enabled);
    ~Spinner();

    Spinner(const Spinner &)             = delete;
    Spinner & operator=(const Spinner &) = delete;

    // RAII bracket: locks the spinner's stdout mutex, erases the active
    // glyph, runs the caller's writes, advances the frame (throttled
    // ~10 Hz) and redraws.  Use scoped() if you need to stream
    // multiple writes inside one bracket; for the common single-write
    // case prefer write(text) below.
    class WriteScope {
    public:
        explicit WriteScope(Spinner & s);
        ~WriteScope();
        WriteScope(const WriteScope &)             = delete;
        WriteScope & operator=(const WriteScope &) = delete;
    private:
        Spinner * s_;
    };
    WriteScope scoped();

    // Convenience: write `text` to stdout under the spinner's lock and
    // redraw afterwards.  No-op when the spinner is disabled (writes
    // straight to stdout).
    void write(const std::string & text);

    // Draw the very first glyph (typically right after starting a request,
    // before any token has arrived).
    void initial_draw();

    // Wipe the glyph at end-of-turn.
    void finish();

    // Heartbeat — refreshes the glyph every interval_ms of idle so the
    // animation never freezes during long dead-air windows.
    void start_heartbeat(int interval_ms = 250);
    void stop_heartbeat();

    // Show a context-fill percentage (0..100) next to the spinner
    // glyph — drawn as `<glyph><pct>%` so the operator can tell at a
    // glance how close the chat is to filling n_ctx.  Pass -1 to hide
    // the suffix.  The next redraw picks the new value up; nothing
    // forces an immediate refresh, the heartbeat handles that.
    void set_context_pct(int pct);

private:
    void maybe_advance_locked_();
    void erase_active_locked_();
    void draw_locked_();
    void heartbeat_loop_();

    static constexpr int kFrameAdvanceMs = 100;     // 10 Hz throttle floor

    bool enabled_      = false;
    bool active_       = false;
    int  frame_        = 0;
    int  active_width_ = 0;   // chars currently on stdout — backspace count for erase
    int  context_pct_  = -1;  // -1 = no suffix; 0..100 = "<pct>%"
    std::chrono::steady_clock::time_point last_advance_{};

    std::mutex              mu_;               // stdout + state
    std::atomic<bool>       hb_running_{false};
    int                     interval_ms_ = 250;
    std::thread             hb_thread_;
    std::mutex              hb_wait_mu_;
    std::condition_variable hb_cv_;
};


// ---------- pretty-printers (small, dependency-free helpers) ---------------
//
// Forward-declarations so consumers don't have to pull plan.hpp /
// presets.hpp transitively just for the ui header.
}  // namespace easyai::ui
namespace easyai { class Plan; }
namespace easyai::ui {

// Print all built-in presets (one per line: name + summary) to `out`.
// Uses dim style for the summary suffix.
void print_presets(const Style & st, std::FILE * out = stdout);

// Print a plan's GitHub-style markdown checklist below a "── plan ──"
// banner.  Used by REPLs to show progress when a Plan callback fires.
void render_plan(const Plan & plan, const Style & st, std::FILE * out = stdout);

// Pretty-print one tool's name + multi-line description in the
// "name: <bold>\n  desc-line\n  desc-line" shape used by --list-tools.
void print_tool_row(const std::string & name,
                    const std::string & description,
                    const Style & st,
                    std::FILE * out = stdout);


// ---------- streaming stats -------------------------------------------------
//
// Counters + timers that track an SSE turn from start to finish.  Free
// to use directly or expose to your callbacks — easyai's example CLIs
// hand it to on_token/on_reason/on_tool to log things like
// "[hop N: content=… reason=… tools=… +Tms]".
struct StreamStats {
    int  content_pieces  = 0;
    int  reason_pieces   = 0;
    int  tool_calls      = 0;
    int  tool_errors     = 0;
    long ms_to_first_tok = -1;
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

}  // namespace easyai::ui


// ---------- streaming wiring helper ----------------------------------------
//
// Forward-declarations so consumers don't pull engine.hpp / client.hpp /
// plan.hpp transitively unless they actually attach to those types.
namespace easyai {
class Engine;
class Client;
class Plan;
struct ToolCall;
struct ToolResult;
}

namespace easyai::ui {

// Streaming — fluent helper that wires the canonical SSE/agent-loop
// streaming UX onto an Engine, Client, or Plan: spinner-locked content
// writes, dimmed reasoning, tool-call markers (🔧/✗), per-piece
// transition newlines (so reasoning's last token doesn't glue to
// content's first), and live plan rendering.
//
// Usage:
//     easyai::ui::Spinner spinner(true);
//     easyai::ui::StreamStats stats;
//     easyai::ui::Streaming(spinner, stats, style)
//         .show_reasoning(true)
//         .verbose       (true)
//         .attach        (client)
//         .attach        (plan);
//
// The class holds borrowed references — caller owns the Spinner /
// StreamStats / Style.  Safe to construct on the stack inside a
// run_one() / run_repl() body around each turn.
class Streaming {
public:
    Streaming(Spinner & spinner, StreamStats & stats, const Style & style);

    Streaming & show_reasoning(bool v);   // dim reasoning prints (default true)
    Streaming & verbose       (bool v);   // verbose tool dump via easyai::log

    // Attach the canonical callbacks.  Each returns *this for chaining.
    Streaming & attach(Engine & engine);  // on_token + on_tool
    Streaming & attach(Client & client);  // on_token + on_reason + on_tool
    Streaming & attach(Plan   & plan);    // on_change → render_plan

private:
    // Helpers shared between Engine/Client attach paths.
    void on_token_(const std::string & piece);
    void on_reason_(const std::string & piece);
    void on_tool_(const ToolCall & call, const ToolResult & result);

    Spinner &     spinner_;
    StreamStats & stats_;
    const Style & style_;
    bool          show_reasoning_ = true;
    bool          verbose_        = false;

    // Tracks which stream emitted last so we can insert a newline on
    // the reasoning↔content transition (otherwise the two glue
    // together as "...phase.I have created..." in the terminal).
    enum class StreamKind { NONE, REASON, CONTENT };
    StreamKind last_kind_ = StreamKind::NONE;
};

}  // namespace easyai::ui
