#include "easyai/ui.hpp"

#include "easyai/engine.hpp"
#include "easyai/log.hpp"
#include "easyai/plan.hpp"
#include "easyai/presets.hpp"
#include "easyai/text.hpp"
#include "easyai/tool.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string_view>
#include <unistd.h>

namespace easyai::ui {

// ---------------------------------------------------------------- Style
Style detect_style() {
    Style s;
    s.color = ::isatty(STDOUT_FILENO) != 0
              && std::getenv("NO_COLOR") == nullptr;
    return s;
}

// ---------------------------------------------------------------- Spinner
Spinner::Spinner(bool enabled)
    : enabled_(enabled && ::isatty(fileno(stdout)) != 0) {}

Spinner::~Spinner() { stop_heartbeat(); }

Spinner::WriteScope::WriteScope(Spinner & s)
    : s_(s.enabled_ ? &s : nullptr) {
    if (!s_) return;
    s_->mu_.lock();
    s_->erase_active_locked_();
}

Spinner::WriteScope::~WriteScope() {
    if (!s_) return;
    std::fflush(stdout);
    s_->maybe_advance_locked_();
    s_->draw_locked_();
    s_->mu_.unlock();
}

Spinner::WriteScope Spinner::scoped() { return WriteScope(*this); }

void Spinner::write(const std::string & text) {
    if (enabled_) {
        auto _g = scoped();
        std::fputs(text.c_str(), stdout);
    } else {
        std::fputs(text.c_str(), stdout);
        std::fflush(stdout);
    }
}

void Spinner::initial_draw() {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lg(mu_);
    if (!active_) draw_locked_();
}

void Spinner::finish() {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lg(mu_);
    erase_active_locked_();
    std::fflush(stdout);
    frame_ = 0;
}

void Spinner::set_context_pct(int pct) {
    if (!enabled_) return;
    if (pct < 0)         pct = -1;
    else if (pct > 100)  pct = 100;
    std::lock_guard<std::mutex> lg(mu_);
    if (context_pct_ == pct) return;
    context_pct_ = pct;
    // Repaint immediately if a glyph is currently visible so the
    // suffix updates without waiting for the next heartbeat tick.
    if (active_) {
        erase_active_locked_();
        draw_locked_();
        std::fflush(stdout);
    }
}

void Spinner::start_heartbeat(int interval_ms) {
    if (!enabled_) return;
    if (hb_running_.exchange(true)) return;
    interval_ms_ = interval_ms;
    hb_thread_   = std::thread(&Spinner::heartbeat_loop_, this);
}

void Spinner::stop_heartbeat() {
    if (!hb_running_.exchange(false)) return;
    hb_cv_.notify_all();
    if (hb_thread_.joinable()) hb_thread_.join();
}

void Spinner::maybe_advance_locked_() {
    const auto now = std::chrono::steady_clock::now();
    const auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - last_advance_).count();
    if (ms >= kFrameAdvanceMs) {
        ++frame_;
        last_advance_ = now;
    }
}

void Spinner::erase_active_locked_() {
    if (!active_ || active_width_ <= 0) return;
    // Wipe `active_width_` chars to the left of the cursor.  We can't
    // assume \b is repeatable across all terminals when crossing a
    // line boundary, but the spinner is always drawn after the most
    // recent newline, so a simple backspace-space-backspace per char
    // is safe.
    for (int i = 0; i < active_width_; ++i) std::fputc('\b', stdout);
    for (int i = 0; i < active_width_; ++i) std::fputc(' ',  stdout);
    for (int i = 0; i < active_width_; ++i) std::fputc('\b', stdout);
    active_       = false;
    active_width_ = 0;
}

void Spinner::draw_locked_() {
    static const char frames[] = { '|', '/', '-', '\\' };
    char buf[16];
    int  n;
    if (context_pct_ < 0) {
        buf[0] = frames[frame_ % 4];
        n = 1;
    } else {
        // `<glyph><pct>%` — no separator, so the suffix sits flush
        // with the glyph and tracks the cursor as it moves.
        n = std::snprintf(buf, sizeof(buf), "%c%d%%",
                          frames[frame_ % 4], context_pct_);
        if (n < 0)                  n = 1;          // snprintf failure
        if (n >= (int) sizeof(buf)) n = (int) sizeof(buf) - 1;
    }
    std::fwrite(buf, 1, (size_t) n, stdout);
    std::fflush(stdout);
    active_       = true;
    active_width_ = n;
}

void Spinner::heartbeat_loop_() {
    while (hb_running_) {
        {
            std::unique_lock<std::mutex> wait_lk(hb_wait_mu_);
            hb_cv_.wait_for(wait_lk,
                            std::chrono::milliseconds(interval_ms_),
                            [this]{ return !hb_running_; });
        }
        if (!hb_running_) break;
        std::lock_guard<std::mutex> lg(mu_);
        if (!active_) continue;            // nothing drawn yet
        erase_active_locked_();
        ++frame_;
        last_advance_ = std::chrono::steady_clock::now();
        draw_locked_();
    }
}

// ---------------------------------------------------------------- pretty-print
void print_presets(const Style & st, std::FILE * out) {
    for (const auto & p : easyai::all_presets()) {
        std::fprintf(out, "  %s%s%s  %s%s%s\n",
                     st.bold(),  p.name.c_str(),        st.reset(),
                     st.dim(),   p.description.c_str(), st.reset());
    }
}

void render_plan(const Plan & plan, const Style & st, std::FILE * out) {
    std::fprintf(out, "\n%s── plan ──%s\n", st.yellow(), st.reset());
    std::ostringstream ss;
    plan.render(ss, st.color);
    std::fputs(ss.str().c_str(), out);
    std::fputc('\n', out);
    std::fflush(out);
}

void print_tool_row(const std::string & name,
                    const std::string & description,
                    const Style & st,
                    std::FILE * out) {
    std::fprintf(out, "%s%s%s\n", st.bold(), name.c_str(), st.reset());
    std::size_t i = 0;
    while (i < description.size()) {
        std::size_t nl = description.find('\n', i);
        std::string line = (nl == std::string::npos)
                               ? description.substr(i)
                               : description.substr(i, nl - i);
        std::fprintf(out, "  %s%s%s\n",
                     st.dim(), line.c_str(), st.reset());
        if (nl == std::string::npos) break;
        i = nl + 1;
    }
}

// ---------------------------------------------------------------- Streaming
Streaming::Streaming(Spinner & spinner, StreamStats & stats, const Style & style)
    : spinner_(spinner), stats_(stats), style_(style) {}

Streaming & Streaming::show_reasoning(bool v) { show_reasoning_ = v; return *this; }
Streaming & Streaming::verbose       (bool v) { verbose_        = v; return *this; }

void Streaming::emit_content_(const std::string & seg) {
    if (seg.empty()) return;
    std::string out = seg;
    if (last_kind_ == StreamKind::REASON) out.insert(0, "\n");
    last_kind_ = StreamKind::CONTENT;
    spinner_.write(out);
}

void Streaming::emit_reason_(const std::string & seg) {
    if (seg.empty()) return;
    if (show_reasoning_) {
        std::string buf;
        buf.reserve(seg.size() + 16);
        if (last_kind_ == StreamKind::CONTENT) buf += "\n";
        buf += style_.dim();
        buf += seg;
        buf += style_.reset();
        spinner_.write(buf);
    }
    last_kind_ = StreamKind::REASON;
    // When show_reasoning is off, the spinner heartbeat keeps the glyph
    // ticking on its own — no explicit refresh needed here.
}

namespace {

// Strip bare <think>/<thinking> open and close markers from `s`. Used
// on the reasoning_content stream as a defense in depth — the parser
// usually removes the wrappers, but a few templates leak them through.
std::string strip_think_markers(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ) {
        if (s.compare(i, 7,  "<think>")     == 0) { i += 7;  continue; }
        if (s.compare(i, 10, "<thinking>")  == 0) { i += 10; continue; }
        if (s.compare(i, 8,  "</think>")    == 0) { i += 8;  continue; }
        if (s.compare(i, 11, "</thinking>") == 0) { i += 11; continue; }
        out.push_back(s[i++]);
    }
    return out;
}

// True if `tail` could still extend into one of the four think markers.
// Used to decide which trailing bytes of the content buffer to hold for
// the next chunk vs. flush immediately.
bool tail_is_partial_think_marker(const std::string & tail) {
    static const char * const kTags[] = {
        "<think>", "<thinking>", "</think>", "</thinking>",
    };
    for (const char * tag : kTags) {
        std::string_view tv(tag);
        if (tail.size() < tv.size() && tv.compare(0, tail.size(), tail) == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace

void Streaming::on_token_(const std::string & piece_in) {
    ++stats_.content_pieces;
    if (stats_.ms_to_first_tok < 0) stats_.ms_to_first_tok = stats_.elapsed_ms();

    // Stateful filter: strip <think>/</think> markers from the content
    // stream and reroute any text between them through emit_reason_ so
    // it gets dim styling (or is dropped when --no-reasoning).  Defends
    // against servers/parsers that occasionally let the bare tags slip
    // into delta.content — typically the first piece after a tool call,
    // when the partial-parse state has just been reset.
    think_buf_ += piece_in;
    while (true) {
        if (in_think_) {
            std::size_t a = think_buf_.find("</think>");
            std::size_t b = think_buf_.find("</thinking>");
            std::size_t close = std::min(a, b);  // npos == -1 stays largest
            if (close == std::string::npos) {
                // No close marker yet.  Flush as reasoning everything
                // up to the last byte sequence that could still be the
                // start of a "</thinking>" / "</think>"; hold the rest.
                std::size_t hold = 0;
                for (std::size_t k = std::min<std::size_t>(think_buf_.size(), 11);
                     k > 0; --k) {
                    if (tail_is_partial_think_marker(
                            think_buf_.substr(think_buf_.size() - k))) {
                        hold = k;
                        break;
                    }
                }
                emit_reason_(think_buf_.substr(0, think_buf_.size() - hold));
                think_buf_.erase(0, think_buf_.size() - hold);
                return;
            }
            emit_reason_(think_buf_.substr(0, close));
            std::size_t end = think_buf_.find('>', close);
            if (end == std::string::npos) return;  // wait for tag close
            think_buf_.erase(0, end + 1);
            in_think_ = false;
        } else {
            std::size_t a = think_buf_.find("<think>");
            std::size_t b = think_buf_.find("<thinking>");
            std::size_t open = std::min(a, b);
            if (open == std::string::npos) {
                // No open marker.  Same trailing-bytes-might-be-tag dance.
                std::size_t hold = 0;
                for (std::size_t k = std::min<std::size_t>(think_buf_.size(), 10);
                     k > 0; --k) {
                    if (tail_is_partial_think_marker(
                            think_buf_.substr(think_buf_.size() - k))) {
                        hold = k;
                        break;
                    }
                }
                emit_content_(think_buf_.substr(0, think_buf_.size() - hold));
                think_buf_.erase(0, think_buf_.size() - hold);
                return;
            }
            emit_content_(think_buf_.substr(0, open));
            std::size_t end = think_buf_.find('>', open);
            if (end == std::string::npos) return;  // wait for tag close
            think_buf_.erase(0, end + 1);
            in_think_ = true;
        }
    }
}

void Streaming::on_reason_(const std::string & piece_in) {
    ++stats_.reason_pieces;
    emit_reason_(strip_think_markers(piece_in));
}

void Streaming::on_tool_(const ToolCall & call, const ToolResult & result) {
    ++stats_.tool_calls;
    if (result.is_error) ++stats_.tool_errors;

    const char * marker = result.is_error ? "✗" : "🔧";
    const char * color  = result.is_error ? style_.red() : style_.cyan();
    std::ostringstream ss;
    ss << "\n" << color << marker << " " << call.name
       << "(" << easyai::text::trim_for_log(call.arguments_json, 80) << ")"
       << style_.reset();
    if (result.is_error) {
        ss << " " << style_.red()
           << easyai::text::trim_for_log(result.content, 100)
           << style_.reset();
    }
    ss << "\n";
    spinner_.write(ss.str());

    if (verbose_) {
        easyai::log::write(
            "%s[tool %s name=%s args_bytes=%zu result_bytes=%zu +%ldms]%s\n",
            style_.dim(),
            result.is_error ? "FAIL" : "ok",
            call.name.c_str(),
            call.arguments_json.size(),
            result.content.size(),
            stats_.elapsed_ms(),
            style_.reset());
        std::string args_prev = easyai::text::trim_for_log(call.arguments_json, 240);
        easyai::log::write("%s         args=%s%s\n",
                           style_.dim(), args_prev.c_str(), style_.reset());
        std::string res_prev = easyai::text::trim_for_log(result.content, 240);
        easyai::log::write("%s         result=%s%s\n",
                           style_.dim(), res_prev.c_str(), style_.reset());
    }
}

Streaming & Streaming::attach(Engine & engine) {
    engine.on_token([this](const std::string & p){ this->on_token_(p); });
    engine.on_tool ([this](const ToolCall & c, const ToolResult & r){
        this->on_tool_(c, r);
    });
    return *this;
}

// Streaming::attach(Client &) lives in src/cli_client.cpp — libeasyai-cli.

Streaming & Streaming::attach(Plan & plan) {
    plan.on_change([this](const Plan & p){
        std::ostringstream ss;
        ss << "\n" << style_.yellow() << "── plan ──" << style_.reset() << "\n";
        p.render(ss, style_.color);
        ss << "\n";
        spinner_.write(ss.str());
    });
    return *this;
}

}  // namespace easyai::ui
