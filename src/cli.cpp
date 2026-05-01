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
#include <filesystem>
#include <unistd.h>   // getpid

namespace fs = std::filesystem;

namespace easyai::cli {

// ============================================================================
// Toolbelt
// ============================================================================
Toolbelt & Toolbelt::sandbox    (std::string dir) { sandbox_     = std::move(dir); return *this; }
Toolbelt & Toolbelt::allow_fs   (bool on)         { allow_fs_    = on;             return *this; }
Toolbelt & Toolbelt::allow_bash (bool on)         { allow_bash_  = on;             return *this; }
Toolbelt & Toolbelt::with_plan  (Plan & plan)     { plan_        = &plan;          return *this; }
Toolbelt & Toolbelt::no_web     (bool on)         { no_web_      = on;             return *this; }
Toolbelt & Toolbelt::no_datetime(bool on)         { no_datetime_ = on;             return *this; }
Toolbelt & Toolbelt::use_google (bool on)         { use_google_  = on;             return *this; }

std::vector<Tool> Toolbelt::tools() const {
    std::vector<Tool> out;
    out.reserve(12);
    if (!no_datetime_) out.push_back(easyai::tools::datetime());
    if (plan_)         out.push_back(plan_->tool());
    if (!no_web_) {
        out.push_back(easyai::tools::web_search());
        out.push_back(easyai::tools::web_fetch());
    }
    // web_google requires explicit opt-in (--use-google in the CLI →
    // Toolbelt::use_google()) AND the GOOGLE_API_KEY + GOOGLE_CSE_ID env
    // vars present. Both gates are intentional: the API counts against a
    // quota and may incur cost, so we never auto-expose it. The tool
    // itself rechecks the env at call time, so a key rotation mid-
    // session surfaces a clear error rather than silent disappearance.
    if (use_google_) {
        const char * gk = std::getenv("GOOGLE_API_KEY");
        const char * gx = std::getenv("GOOGLE_CSE_ID");
        if (gk && *gk && gx && *gx) {
            out.push_back(easyai::tools::web_google());
        }
    }
    if (allow_fs_ && !sandbox_.empty()) {
        out.push_back(easyai::tools::fs_list_dir  (sandbox_));
        out.push_back(easyai::tools::fs_read_file (sandbox_));
        out.push_back(easyai::tools::fs_glob      (sandbox_));
        out.push_back(easyai::tools::fs_grep      (sandbox_));
        out.push_back(easyai::tools::fs_write_file(sandbox_));
    }
    if (allow_bash_) {
        const std::string root = sandbox_.empty() ? "." : sandbox_;
        out.push_back(easyai::tools::bash(root));
    }
    // get_current_dir is registered whenever ANY filesystem-flavoured
    // tool is on (fs_*, bash). It costs nothing, takes no parameters,
    // and is the canonical way for the model to learn the sandbox path
    // it should anchor relative paths against. Without an fs/bash tool
    // there's nothing for it to anchor, so we keep it off — minimum
    // surface area.
    const bool any_fs_like = (allow_fs_ && !sandbox_.empty()) || allow_bash_;
    if (any_fs_like) out.push_back(easyai::tools::get_current_dir());
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
    if (!path.empty()) {
        resolved = path;
    } else {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "/tmp/%s-%d-%ld.log",
                      prefix.empty() ? "easyai" : prefix.c_str(),
                      (int) ::getpid(),
                      (long) std::time(nullptr));
        resolved = buf;
    }

    std::FILE * fp = std::fopen(resolved.c_str(), "w");
    if (resolved_path) *resolved_path = resolved;
    if (!fp) return nullptr;

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
