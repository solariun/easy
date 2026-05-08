// easyai/builtin_tools.hpp — batteries-included tools for agents.
#pragma once

#include "tool.hpp"

#include <functional>
#include <string>
#include <vector>

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
//
// Path convention surfaced to the model: RELATIVE paths under the
// sandbox root (e.g. `report.md`, `src/main.cpp`). Absolute / `..`-laden
// inputs are silently re-anchored under the root for safety, but the
// tool descriptions tell the model to use relatives only — it makes
// the boundary obvious and matches what bash with the pinned cwd sees.
Tool fs_read_file (std::string root = ".");
Tool fs_write_file(std::string root = ".");
Tool fs_list_dir  (std::string root = ".");
Tool fs_glob      (std::string root = ".");
Tool fs_grep      (std::string root = ".");

// fs_check_path: pre-flight stat + access-rights probe at a sandbox-
// relative path. The model is told (via every fs_*/bash description)
// to call this BEFORE attempting any read/write so the sandbox
// boundary, file existence, and effective r/w/x rights for the
// running process are all confirmed up-front. With `touch=true` the
// tool also creates an empty file at the path (parent dirs created
// as needed) when nothing exists there yet — the equivalent of
// `mkdir -p && touch` for cheap "can I write here?" probing.
//
// Output is a short multi-line block (path, absolute, exists, type,
// size, mode, readable, writable, executable, mtime) — easy for the
// model to parse and quote back. Errors return a single-line
// `error:` message just like the other fs_* tools.
Tool fs_check_path(std::string root = ".");

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

// get_sandbox_path: returns the absolute path of the configured sandbox
// root. Distinct from get_current_dir: the sandbox path is pinned at
// registration time (via the `root` argument the toolbelt passes), so
// it is always the boundary fs_* and bash operate inside, regardless
// of what the process's cwd happens to be. When no sandbox is
// configured (`root` empty or "."), this tool falls back to the
// current working directory — same as get_current_dir would.
//
// No parameters. Use this when you need the real on-disk path of
// where your work is landing (fs_* otherwise speaks a virtual `/`).
Tool get_sandbox_path(std::string root = ".");

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
//
// `show_output`: when true, the merged child stdout/stderr is also
// mirrored to the parent's stderr in real time as it drains. The model
// still receives the full captured buffer as the tool result; the
// stderr mirror is a diagnostic aid for the operator watching the
// terminal so a long-running command (build, test suite) doesn't look
// like a stalled session. Off by default to keep tool output strictly
// in-band; the CLI flips it on unless --no-show-bash is passed.
Tool bash         (std::string root = ".", bool show_output = false);

// ---------- introspection -----------------------------------------------
// tool_lookup: return the currently-registered tool catalogue, optionally
// filtered by a substring match on the tool name (case-insensitive,
// partial). Format is a numbered list 1..N with each tool's name and its
// declared description.
//
// The model uses this to verify what's actually wired up before it tries
// to call something. Without this tool, smaller models hallucinate
// "obvious" names (`write`, `read`, `ls`, `curl`, `python`) and produce
// "unknown tool: …" errors mid-task. With it, the model can sanity-check
// "is `write_file` actually available here?" in one hop, and only ever
// dispatches names that exist.
//
// `get_tools` is a callable that returns a snapshot of the live tool
// catalogue as (name, description) pairs.  Wire it at registration time
// to your Engine's or Client's `tools()` accessor:
//
//     engine.add_tool(easyai::tools::tool_lookup([&engine]() {
//         std::vector<std::pair<std::string,std::string>> v;
//         for (const auto & t : engine.tools()) {
//             v.emplace_back(t.name, t.description);
//         }
//         return v;
//     }));
//
// (Handlers — `std::function<...>` — aren't part of the snapshot, so
// no expensive closure copies. Add tool_lookup AFTER all other tools so
// it sees the complete list. tool_lookup's own entry will appear in
// the result, which is intentional: the model knows it has this
// affordance.)
//
// Parameters (the model fills these in):
//   `name` — string, optional. Substring; case-insensitive; matches the
//            tool name. Empty / missing = "list everything".
//
// Output shape (what the model sees):
//   - "1. <name>: <description>"
//   - "2. ..."
//   - "(no tools match: …)" when a filter found nothing
//   - "(no tools registered)" when the registry is empty
//
// Read-only over the registry; never spawns a process or touches the
// filesystem.
using ToolCatalog    = std::vector<std::pair<std::string, std::string>>;
using ToolListGetter = std::function<ToolCatalog()>;
Tool tool_lookup(ToolListGetter get_tools);

}  // namespace easyai::tools
