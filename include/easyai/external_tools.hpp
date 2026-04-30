// easyai/external_tools.hpp — load custom tools from a JSON manifest.
//
// The OPERATOR (not the model) declares a JSON file that lists which
// external commands the model is allowed to call, what arguments each
// takes, and the resource caps (timeout, output size, env). The library
// loads the manifest, validates it strictly, and registers each entry
// as a regular `Tool` that can be added to an Engine or Client just
// like a built-in.
//
// Trust model
// -----------
// * The JSON file is part of the operator's deploy artefact. Treat it
//   like a sudoers file — anyone who can write it can run arbitrary
//   commands as the agent's user.
// * The model never writes to the manifest. It only fills in parameter
//   values, which are validated against the per-tool JSON Schema and
//   passed as argv elements (NEVER through a shell).
//
// Security guarantees enforced at load time
// -----------------------------------------
// * `command` MUST be an absolute path to a regular, executable file.
//   Relative names are rejected — no PATH search, no PATH-hijack risk.
// * Each `argv` template element is either a literal string or a single
//   placeholder of the form "{name}" where `name` is a declared
//   parameter. Placeholders embedded inside larger strings (e.g.
//   "--flag={x}") are rejected: the model's value flows through as one
//   whole argv element or not at all.
// * Tool names match `^[a-zA-Z][a-zA-Z0-9_]{0,63}$` and must not collide
//   with built-ins (bash, read_file, write_file, etc.).
// * `timeout_ms` clamped to [100, 300000] (5 min hard ceiling).
// * `max_output_bytes` clamped to [1024, 4 MiB].
// * `env_passthrough` is an opt-in allowlist (default empty). Only
//   listed env vars from the parent process are inherited; everything
//   else is wiped.
// * stdin is closed before exec.
// * `cwd` is fixed by the manifest (literal absolute path or the magic
//   token `$SANDBOX` resolved at load against the current process cwd).
//
// Security guarantees enforced at call time
// -----------------------------------------
// * Model arguments are JSON-Schema-validated against the declared
//   parameters before fork() is even called.
// * Subprocess runs in its own process group. On timeout we send
//   SIGTERM to the group, then SIGKILL after a 1 s grace.
// * Output capped at `max_output_bytes`; further bytes are discarded
//   (process keeps running so its stdout doesn't block on a full pipe).
//
// Reject any deviation. There is no "lenient mode".
#pragma once

#include "tool.hpp"

#include <string>
#include <vector>

namespace easyai {

// Result of loading a manifest. On success, `tools` is non-empty and
// `error` is empty. On failure, `tools` is empty and `error` carries a
// human-readable diagnostic with the offending tool name / field path.
//
// `warnings` collects non-fatal sanity-check observations produced
// during a successful load (shell wrappers, dynamic-linker env
// passthrough, world-writable binaries / manifests, …). The tool
// still loads — the warning is informational, intended for the
// operator's startup log. Quiet-mode CLIs may suppress the warning
// stream while still surfacing the (rare) hard error.
//
// Failure is all-or-nothing: a single bad entry rejects the whole file
// so the operator notices on startup instead of at call time.
struct ExternalToolsLoad {
    std::vector<Tool>        tools;
    std::vector<std::string> warnings;
    std::string              error;
};

// Load and validate a manifest file. Returns ExternalToolsLoad with
// either tools or error populated (never both). The manifest file
// itself is read once; the returned tools own copies of every string
// they need so the manifest can safely be deleted afterwards.
//
// `reserved_names` is the list of names that already exist in the
// caller's tool registry (built-ins + previously registered tools).
// We use it to surface collisions as a load error rather than letting
// the engine silently dispatch to whichever was registered last.
ExternalToolsLoad load_external_tools_from_json(
    const std::string &              json_path,
    const std::vector<std::string> & reserved_names);

// ---------------------------------------------------------------------------
// Directory loader — multi-file deploy / collaboration mode.
// ---------------------------------------------------------------------------
//
// Real deployments quickly outgrow a single manifest: the operator
// has system-level tools, the on-call has incident-response tools,
// individual users have their own helpers — keeping all of those in
// one JSON file is a coordination headache and a code-review
// nightmare.
//
// Solution: a directory of manifests. Drop a file in, it's a tool
// pack. Remove it, the tools go away. Each file is a self-contained
// unit; one bad file does NOT prevent the others from loading.
//
// File-naming convention
// ----------------------
// Only files matching `EASYAI-<anything>.tools` are loaded:
//
//     EASYAI-system.tools         loaded
//     EASYAI-deploy.tools         loaded
//     EASYAI-user-gustavo.tools   loaded
//     EASYAI-disabled.tools.bak   skipped (wrong suffix)
//     mytools.json                skipped (wrong prefix)
//     README.md                   skipped (wrong shape)
//
// To "disable" a file without deleting it, rename its extension
// (e.g. `.tools.disabled`). The pattern is exact, case-sensitive,
// and ignores subdirectories — only the top level of `dir` is
// scanned.
//
// Load order is alphabetic (sorted filenames) so duplicate-name
// resolution is deterministic across machines and reboots: the
// first file that declares a given tool name wins; subsequent files
// trying to redeclare it surface a clear collision error AND get
// skipped, while their other tools (with non-conflicting names)
// still load.
//
// Fault isolation
// ---------------
// A parse error, schema error, or sanity-check failure in ONE file
// must not break the rest. Each file is loaded in isolation:
//
//   - `tools` accumulates every successfully-validated tool from
//     every successfully-loaded file.
//   - `errors` lists files that failed to load (with file path +
//     human-readable diagnostic). The agent starts up regardless;
//     the operator notices the error in the startup log.
//   - `warnings` lists non-fatal sanity-check observations (shell
//     wrappers, dynamic-linker env passthrough, world-writable
//     binaries / manifests, …). These are informational; the tool
//     still loads.
//   - `loaded_files` and `skipped_files` are reported for
//     diagnostic logs — handy when the operator is wondering
//     "why didn't my tool show up?"
//
// The `reserved_names` list grows as files load, so a name
// declared in the first sorted file blocks the same name in
// subsequent files (the second declaration becomes a load error
// localised to its file).
struct ExternalToolsDirLoad {
    std::vector<Tool>        tools;          // every successfully-loaded tool
    std::vector<std::string> warnings;       // sanity-check observations (loaded anyway)
    std::vector<std::string> errors;         // per-file load errors (file skipped)
    std::vector<std::string> loaded_files;   // files contributing to `tools`
    std::vector<std::string> skipped_files;  // files skipped by name pattern (top-level only)
};

// Scan `dir` for files matching `EASYAI-*.tools` (top-level, exact
// pattern, case-sensitive). Each matched file is parsed and
// validated independently; a failure in one file is recorded and
// the rest still load.
//
// `reserved_names` carries the caller's already-registered tool
// names (built-ins + any tools added before this call). The loader
// extends this set as it goes — names from earlier files reserve
// against later ones.
//
// Returning even when some files failed is intentional: an agent
// that is allowed to start with 8/10 tools is more useful than one
// that refuses to start because someone left a stray brace in
// `EASYAI-experimental.tools`. The errors list lets the operator
// see exactly what went wrong.
//
// On `dir` not existing or not being a directory, `errors` carries
// a single explanatory entry and the other vectors are empty.
ExternalToolsDirLoad load_external_tools_from_dir(
    const std::string &              dir,
    const std::vector<std::string> & reserved_names);

}  // namespace easyai
