// easyai/builtin_tools.hpp — batteries-included tools for agents.
#pragma once

#include "tool.hpp"

#include <string>

namespace easyai::tools {

// ---------- time --------------------------------------------------------
Tool datetime();   // returns current UTC + local time in ISO-8601

// ---------- web (require libcurl at build time) -------------------------
// web_fetch:  GET a URL, return text (HTML stripped of tags + trimmed).
Tool web_fetch();

// web_search: search the web via DuckDuckGo's HTML endpoint (no API key, no
// external service required) and return the top results as a numbered list
// of title / url / snippet.
Tool web_search();

// web_google: search the web via Google's Custom Search JSON API. Returns
// the same numbered title/url/snippet format as web_search. Requires two
// environment variables — read at call time, not at registration:
//
//   GOOGLE_API_KEY  — your Google Cloud API key with Custom Search enabled.
//   GOOGLE_CSE_ID   — the cx parameter of a Programmable Search Engine.
//                     Configure it to "Search the entire web" for general use.
//
// Get both at https://programmablesearchengine.google.com (the CSE) and
// https://console.cloud.google.com/apis/credentials (the key). Free tier:
// 100 queries/day. When either env var is missing the tool returns a
// clear error at call time so the model can fall back to web_search.
Tool web_google();

// ---------- filesystem --------------------------------------------------
// All filesystem tools sandbox to a root directory you provide.
// Pass "" or "." to allow the current working directory tree.
Tool fs_read_file (std::string root = ".");
Tool fs_write_file(std::string root = ".");
Tool fs_list_dir  (std::string root = ".");
Tool fs_glob      (std::string root = ".");
Tool fs_grep      (std::string root = ".");

// get_current_dir: returns the absolute path of the process's current
// working directory at call time (not at registration time — getcwd is
// invoked on every call so a process that chdir'd later still reports
// truthfully). Pair with --sandbox: the CLI / server chdir into the
// sandbox at startup so the model's "current directory" matches the
// boundary that fs_* and bash already enforce.
//
// No parameters; no arguments to validate; cannot fail in a meaningful
// way except an OS-level getcwd error which we surface as a tool error.
Tool get_current_dir();

// ---------- shell -------------------------------------------------------
// Run a shell command via /bin/sh -c. Working directory is set to `root`.
//
// IMPORTANT: this is NOT a hardened sandbox. The child runs with the
// caller's full uid/gid, can read/write any file the caller can, can
// hit the network, can spawn long-lived processes, etc. The only
// safety nets are:
//   - cwd is fixed to `root` (so the model's relative paths land there);
//   - merged stdout/stderr is captured and capped at 32 KB;
//   - a hard timeout (default 30s, capped at 300s) sends SIGTERM then
//     SIGKILL.
// Caller is responsible for deciding whether bash is appropriate for
// their threat model — we surface it only when the user opts in
// (e.g. --allow-bash in cli-remote).
Tool bash         (std::string root = ".");

}  // namespace easyai::tools
