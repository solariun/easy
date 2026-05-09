// libeasyai-side bits of easyai::cli — the Client overload of
// Toolbelt::apply lives in src/cli_client.cpp (libeasyai-cli) so the
// engine-only library doesn't end up depending on the HTTP client.
#include "easyai/cli.hpp"

#include "easyai/builtin_tools.hpp"
#include "easyai/engine.hpp"
#include "easyai/log.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>    // open, O_*
#include <filesystem>
#include <sys/stat.h> // mode_t
#include <unistd.h>   // getpid

namespace fs = std::filesystem;

namespace easyai::cli {

// ============================================================================
// Toolbelt
// ============================================================================
Toolbelt & Toolbelt::sandbox    (std::string dir) { sandbox_     = std::move(dir); return *this; }
Toolbelt & Toolbelt::allow_fs   (bool on)         { allow_fs_    = on;             return *this; }
Toolbelt & Toolbelt::allow_bash (bool on)         { allow_bash_  = on;             return *this; }
Toolbelt & Toolbelt::show_bash  (bool on)         { show_bash_   = on;             return *this; }
Toolbelt & Toolbelt::with_plan  (Plan & plan)     { plan_        = &plan;          return *this; }
Toolbelt & Toolbelt::no_web     (bool on)         { no_web_      = on;             return *this; }
Toolbelt & Toolbelt::no_datetime(bool on)         { no_datetime_ = on;             return *this; }
Toolbelt & Toolbelt::use_google (bool on)         { use_google_  = on;             return *this; }

std::vector<Tool> Toolbelt::tools() const {
    std::vector<Tool> out;
    out.reserve(8);
    if (!no_datetime_) out.push_back(easyai::tools::datetime());
    if (plan_)         out.push_back(plan_->tool());
    // Unified `web` tool: action="search" (engine ddg/google) +
    // action="fetch". google_enabled is the operator opt-in gate
    // (--use-google) that keeps the billed Google CSE off by default;
    // env vars are still re-read on every call so key rotation works
    // without a restart and a missing key surfaces a clear error.
    if (!no_web_) {
        bool google = false;
        if (use_google_) {
            const char * gk = std::getenv("GOOGLE_API_KEY");
            const char * gx = std::getenv("GOOGLE_CSE_ID");
            google = (gk && *gk && gx && *gx);
        }
        out.push_back(easyai::tools::web(google));
    }
    // fs and bash share a working root: the configured sandbox if set,
    // otherwise ".". Whenever either category is enabled, both should be
    // available — bash is strictly more permissive than fs, so allowing
    // bash without fs is incoherent (and traps models into using bash
    // for file work because there's nothing else). Conversely, anyone
    // pointing at a sandbox wants file access in it.
    const bool fs_on   = allow_fs_ && (!sandbox_.empty() || allow_bash_);
    const bool bash_on = allow_bash_;
    const std::string fs_root = sandbox_.empty() ? "." : sandbox_;
    if (fs_on) {
        // Single unified `fs` tool. Eight actions: read, write, list,
        // glob, grep, check_path, cwd, sandbox. The cwd / sandbox
        // actions replace the old standalone get_current_dir /
        // get_sandbox_path tools.
        out.push_back(easyai::tools::fs(fs_root));
    }
    if (bash_on) {
        out.push_back(easyai::tools::bash(fs_root, show_bash_));
    }
    return out;
}

void Toolbelt::apply(Engine & engine) const {
    for (auto & t : tools()) engine.add_tool(t);
    if (allow_bash_) engine.max_tool_hops(99999);
}
// Toolbelt::apply(Client &) lives in src/cli_client.cpp — libeasyai-cli.

// ============================================================================
// open_log_tee / close_log_tee
// ============================================================================
std::FILE * open_log_tee(const std::string & path,
                         const std::string & prefix,
                         int argc, char ** argv,
                         std::string * resolved_path) {
    std::string resolved;
    const bool auto_path = path.empty();
    if (!auto_path) {
        resolved = path;
    } else {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "/tmp/%s-%d-%ld.log",
                      prefix.empty() ? "easyai" : prefix.c_str(),
                      (int) ::getpid(),
                      (long) std::time(nullptr));
        resolved = buf;
    }
    if (resolved_path) *resolved_path = resolved;

    // Auto-generated /tmp paths are predictable (pid + epoch), so we
    // refuse atomically if the path already exists (O_EXCL) — a local
    // attacker who pre-creates the path as a symlink would otherwise
    // redirect our truncating write to an arbitrary user-writable file.
    // Caller-supplied paths skip O_EXCL because operators legitimately
    // want overwrite semantics for log rotation, but we still pin
    // O_NOFOLLOW (refuse if the leaf is a symlink — operators planting
    // a symlink there is suspicious) and mode 0600 (logs may echo
    // prompts that contain secrets).
    int flags = O_WRONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC;
    flags |= auto_path ? O_EXCL : O_TRUNC;
    const int fd = ::open(resolved.c_str(), flags, 0600);
    if (fd < 0) return nullptr;
    std::FILE * fp = ::fdopen(fd, "w");
    if (!fp) { ::close(fd); return nullptr; }

    // Header — keeps a stray log self-describing.
    char ts[32] = {0};
    const auto t = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    std::fprintf(fp,
        "%s raw transaction log\n"
        "started: %s   pid: %d\nargv:",
        prefix.empty() ? "easyai" : prefix.c_str(),
        ts, (int) ::getpid());
    for (int k = 0; k < argc; ++k) std::fprintf(fp, " %s", argv[k]);
    std::fputc('\n', fp);
    std::fflush(fp);

    easyai::log::set_file(fp);
    return fp;
}

void close_log_tee(std::FILE * fp) {
    if (!fp) return;
    std::fputs("\n========== END OF LOG ==========\n", fp);
    std::fclose(fp);
    easyai::log::set_file(nullptr);
}

// ============================================================================
// validate_sandbox
// ============================================================================
bool validate_sandbox(const std::string & path, std::string & err) {
    if (path.empty()) return true;
    std::error_code ec;
    auto stat = fs::status(path, ec);
    if (ec || !fs::exists(stat)) {
        err = "--sandbox " + path + " does not exist";
        return false;
    }
    if (!fs::is_directory(stat)) {
        err = "--sandbox " + path + " is not a directory";
        return false;
    }
    return true;
}

}  // namespace easyai::cli
