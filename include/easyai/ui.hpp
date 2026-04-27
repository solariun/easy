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

private:
    void maybe_advance_locked_();
    void draw_locked_();
    void heartbeat_loop_();

    static constexpr int kFrameAdvanceMs = 100;     // 10 Hz throttle floor

    bool enabled_ = false;
    bool active_  = false;
    int  frame_   = 0;
    std::chrono::steady_clock::time_point last_advance_{};

    std::mutex              mu_;               // stdout + state
    std::atomic<bool>       hb_running_{false};
    int                     interval_ms_ = 250;
    std::thread             hb_thread_;
    std::mutex              hb_wait_mu_;
    std::condition_variable hb_cv_;
};


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
