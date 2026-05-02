#include "easyai/log.hpp"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>    // open, O_*
#include <mutex>
#include <sys/stat.h> // mode_t
#include <unistd.h>   // getpid

namespace easyai::log {

namespace {
std::FILE *  g_sink         = nullptr;
std::FILE *  g_owned        = nullptr;   // non-null only when WE opened it
std::mutex   g_mu;
bool         g_atexit_armed = false;

void write_v_locked(const char * fmt, std::va_list ap) {
    std::va_list ap2;
    va_copy(ap2, ap);
    std::vfprintf(stderr, fmt, ap);
    if (g_sink) {
        std::vfprintf(g_sink, fmt, ap2);
        std::fflush(g_sink);
    }
    va_end(ap2);
}

}  // namespace

void set_file(std::FILE * fp) {
    std::lock_guard<std::mutex> lk(g_mu);
    // If the lib previously auto-opened a /tmp sink and the caller is
    // now taking over with their own file, close ours first so we don't
    // leak the FILE*.  (Skip the close when they're handing us back the
    // SAME pointer — that would be a double fclose.)
    if (g_owned && g_owned != fp) {
        std::fputs("\n========== END OF AUTO LOG (replaced by caller) ==========\n", g_owned);
        std::fclose(g_owned);
    }
    g_owned = nullptr;
    g_sink  = fp;
}

std::FILE * file() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_sink;
}

void write(const char * fmt, ...) {
    std::lock_guard<std::mutex> lk(g_mu);
    std::va_list ap;
    va_start(ap, fmt);
    write_v_locked(fmt, ap);
    va_end(ap);
}

void error(const char * fmt, ...) {
    std::lock_guard<std::mutex> lk(g_mu);
    // 1) Plain stderr-tee line so the operator sees it live.
    std::va_list ap;
    va_start(ap, fmt);
    write_v_locked(fmt, ap);
    va_end(ap);

    // 2) Greppable banner in the raw log file (only there — stderr
    //    already showed the message above).
    if (g_sink) {
        char ts[32] = {0};
        const auto t = std::time(nullptr);
        std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
        std::fprintf(g_sink, "\n!!!!! ERROR @ %s !!!!!\n", ts);
        std::va_list ap2;
        va_start(ap2, fmt);
        std::vfprintf(g_sink, fmt, ap2);
        va_end(ap2);
        // Make sure the banner is on its own block whether the format
        // string ended with \n or not.
        std::fputs("\n!!!!! /ERROR !!!!!\n", g_sink);
        std::fflush(g_sink);
    }
}

void mark_problem(const char * fmt, ...) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_sink) return;
    char ts[32] = {0};
    const auto t = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    std::fprintf(g_sink,
        "\n!!!!! PROBLEMATIC TRANSACTION @ %s !!!!!\n", ts);
    std::va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_sink, fmt, ap);
    va_end(ap);
    std::fputs("\n!!!!! /PROBLEMATIC TRANSACTION !!!!!\n", g_sink);
    std::fflush(g_sink);
}

std::FILE * auto_open(const char * prefix, std::string * resolved_path) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_sink) {
        // Already wired up — either by the application (open_log_tee
        // from a CLI binary) or by an earlier auto_open call.  Don't
        // clobber it.
        return g_sink;
    }
    if (const char * v = std::getenv("EASYAI_NO_AUTO_LOG")) {
        if (v[0] != '\0' && std::strcmp(v, "0") != 0) return nullptr;
    }
    if (!prefix || !*prefix) prefix = "easyai";

    char path[256];
    std::snprintf(path, sizeof(path), "/tmp/%s-%d-%ld.log",
                  prefix, (int) ::getpid(), (long) std::time(nullptr));
    // O_EXCL refuses atomically if the path already exists (regular file
    // OR symlink) — closes the predictable-name attack where a local
    // attacker pre-creates `/tmp/easyai-<pid>-<epoch>.log` as a symlink to
    // e.g. ~/.bashrc and tricks us into truncating and overwriting it.
    // O_NOFOLLOW is belt-and-suspenders. Mode 0600 keeps logs (which may
    // echo prompts containing secrets) out of other users' view.
    const int fd = ::open(path,
                          O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                          0600);
    if (fd < 0) return nullptr;
    std::FILE * fp = ::fdopen(fd, "w");
    if (!fp) { ::close(fd); return nullptr; }

    char ts[32] = {0};
    const auto t = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    std::fprintf(fp,
        "%s raw transaction log (auto)\n"
        "started: %s   pid: %d\n"
        "Set EASYAI_NO_AUTO_LOG=1 to disable.\n"
        "----------------------------------------\n",
        prefix, ts, (int) ::getpid());
    std::fflush(fp);

    g_sink  = fp;
    g_owned = fp;
    if (resolved_path) *resolved_path = path;

    if (!g_atexit_armed) {
        g_atexit_armed = true;
        // auto_close is mutex-safe; std::atexit ordering is fine here.
        std::atexit([]() { auto_close(); });
    }
    // Mirror to stderr so the operator sees where the log went.
    std::fprintf(stderr,
        "[easyai] raw transaction log: %s "
        "(set EASYAI_NO_AUTO_LOG=1 to disable)\n", path);
    return fp;
}

void auto_close() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_owned) return;
    std::fputs("\n========== END OF LOG ==========\n", g_owned);
    std::fclose(g_owned);
    if (g_sink == g_owned) g_sink = nullptr;
    g_owned = nullptr;
}

}  // namespace easyai::log
