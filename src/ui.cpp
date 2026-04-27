#include "easyai/ui.hpp"

#include <cstdio>
#include <cstdlib>
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
    if (s_->active_) {
        std::fputs("\b \b", stdout);
        s_->active_ = false;
    }
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
    if (active_) {
        std::fputs("\b \b", stdout);
        std::fflush(stdout);
        active_ = false;
    }
    frame_ = 0;
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

void Spinner::draw_locked_() {
    static const char frames[] = { '|', '/', '-', '\\' };
    std::fputc(frames[frame_ % 4], stdout);
    std::fflush(stdout);
    active_ = true;
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
        std::fputs("\b \b", stdout);
        ++frame_;
        last_advance_ = std::chrono::steady_clock::now();
        draw_locked_();
    }
}

}  // namespace easyai::ui
