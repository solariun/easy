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
// Failure is all-or-nothing: a single bad entry rejects the whole file
// so the operator notices on startup instead of at call time.
struct ExternalToolsLoad {
    std::vector<Tool> tools;
    std::string       error;
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

}  // namespace easyai
