# easyai — security audit (v0.1.0 + post-FUTURE pass)

A static review pass over `src/` and `examples/` looking for memory
overflow, injection, SSRF, regex DoS, TLS, and concurrency hazards.
Fixes applied in this same commit are noted in-line.

---

## 1. PATH TRAVERSAL — `Sandbox::resolve` (HIGH, FIXED)

**File:** `src/builtin_tools.cpp` — function `Sandbox::resolve`.

**Issue:** the check was a raw string-prefix:

```cpp
if (cs.compare(0, rs.size(), rs) != 0) { err = "..."; return false; }
```

With `root="/srv/user"`, the canonical path `/srv/userMALICIOUS/secret`
would PASS this check (string prefix matches) even though it lives in
a totally different directory tree.  Any model invocation of
`fs_read_file` / `fs_list_dir` / `fs_glob` / `fs_grep` could escape the
sandbox.

**Fix:** require canonical to either equal root OR start with
`root + path-separator`:

```cpp
const bool inside =
    cs == rs ||
    (cs.size() > rs.size()
     && cs.compare(0, rs.size(), rs) == 0
     && (cs[rs.size()] == fs::path::preferred_separator
         || cs[rs.size()] == '/'));
```

---

## 2. SSRF — `web_fetch` / DDG `web_search` (HIGH, FIXED)

**File:** `src/builtin_tools.cpp` — `http_get` and `http_post_form`.

**Issue:** `CURLOPT_URL` accepts ANY scheme.  The model could ask for:
- `file:///etc/passwd` (local file read)
- `http://localhost:6379/...` (Redis on the loopback)
- `http://169.254.169.254/...` (cloud metadata)
- `gopher://internal-rabbitmq:5672/...` (protocol abuse)

The previous `curl_write_cb` also appended unbounded — server replies
larger than `max_bytes` were fully buffered before the `body.resize`
post-check, allowing a remote attacker (or DDG) to OOM the process.

**Fix (3 layers):**

1. Refuse anything but `http(s)://` at the URL level via
   `url_is_safe_scheme()`.
2. Pin curl protocols at the transport layer
   (`CURLOPT_PROTOCOLS_STR="http,https"` plus the same for redirects).
3. New `HttpSink` wrapper enforces `max_bytes` IN the write callback
   so RAM stays bounded against an adversarial endpoint.  We still
   return the requested length to curl so the connection drains
   cleanly, but only the first `max_bytes` are kept; `truncated=true`
   is recorded.

This does NOT block `http://localhost` / RFC1918 ranges by design —
those are sometimes legitimate (local dev, internal docs).  When
running easyai-server on a hostile network, ship behind a firewall
or wrap web_fetch in a proxy that enforces destination policy.

---

## 3. REGEX DoS — `fs_glob` (MEDIUM, FIXED)

**File:** `src/builtin_tools.cpp` — `fs_glob` handler.

**Issue:** the glob-to-regex conversion produces ECMAScript regex
that's then compiled with `std::regex(re)` — without a try/catch.
A model-supplied pattern with unbalanced brackets crashed the
process (uncaught `std::regex_error`).  In addition, libstdc++'s
regex has known stack-overflow behaviour on adversarial backtracking
patterns; we mitigate but can't eliminate that here.

**Fix:** wrap both compile and match in try/catch.  Compile errors
return a clean tool-error string ("bad glob pattern: …"); match
errors on a per-entry basis are silently skipped so the rest of the
listing still works.

`fs_grep` already had try/catch on compile (no fix needed).

The DDG-result regexes (`re_title`, `re_snippet`) use lazy `[\s\S]*?`
on bounded windows (≤ 2 KiB at a time), so catastrophic backtracking
is not reachable from external HTML.

---

## 4. STD::REGEX in `strip_html` — N/A (already migrated)

Previously this was the textbook stack-overflow vector — libstdc++'s
recursive backtracker on `<(script|style)[^>]*>[\s\S]*?</\1>` blew
the stack on big news pages.  Migrated to a hand-rolled forward-only
scanner in commit `3dec718`.  No regex anywhere in `strip_html`
today.

---

## 5. JSON parsing — INJECTION-AWARE

`nlohmann::json::parse` is invoked at every API boundary:

- `parse_chat_request`            — request body
- `Client::list_models`           — server response
- `Client::list_remote_tools`     — server response
- `recover_qwen_tool_calls`       — model output (raw text scan)
- `recover_hermes_tool_calls`     — model output (raw text scan)
- `recover_markdown_tool_calls`   — model output (raw text scan)
- `args::get_*`                   — tool args (single-level scan only)

All parse calls are inside try/catch.  The recovery helpers use
hand-written scanners over the raw text (no JSON dep), so a malformed
emit can never crash the engine — it's caught inside
`parse_assistant`'s try/catch and turned into a recovery pass or a
final "raw content" fallback.

`args::get_*` is single-level — nested objects must be parsed by the
tool author.  Our built-in tools that need nested args
(`recover_hermes_tool_calls` builds JSON strings from XML pieces)
escape values via a hand-written `json_escape` helper that handles
backslash, quote, and the C0 control range.

---

## 6. URL DECODING — DDG redirect

`decode_ddg_redirect` URL-decodes the `uddg=` parameter using a
hex scan that explicitly bounds the read range.  No buffer overruns;
malformed `%XX` sequences fall back to passing the bytes through.

---

## 7. PATH-INJECTION via TOOL ARGUMENTS

`fs_*` tool arguments are parsed via the
`Sandbox::resolve` containment check (see #1).  After the fix the
check is path-component aware, so adversarial inputs like
`../../etc/passwd`, `/srv/userMALICIOUS/...`, and symlink dances
through `weakly_canonical` are all rejected.

`fs_write_file` deliberately requires explicit registration — the
LocalBackend in `easyai-cli` hides it behind `--with-tools` (off by
default for `--url` mode), and `easyai-cli-remote` requires
`--sandbox DIR` to even be wirable.  Don't enable this on a
production agent without the sandbox firmly set.

---

## 8. HTTP server — `examples/server.cpp`

* `set_payload_max_length(args.max_body)` caps request bodies at
  8 MiB by default (`--max-body N` to change).  Prevents a malicious
  client from sending a multi-GB JSON.
* `set_read_timeout` / `set_write_timeout` enforce socket-level
  timeouts; slow-loris connections bail at 60s.
* All handlers run inside `try/catch` via
  `svr.set_exception_handler` so a thrown C++ exception cannot tear
  down the process.
* `engine_mu` (mutex) serialises engine access across cpp-httplib
  worker threads — no race on the single shared `easyai::Engine`.
* Bearer auth (`require_auth`) is constant-time-equality-free
  (`std::string ==`); leaks single-bit timing info on key length.
  Acceptable for chat servers; if you publish on the public Internet
  with a long-lived shared key, consider switching to
  `CRYPTO_memcmp` or rate-limiting failed attempts at a reverse proxy.

---

## 9. WEBUI INJECTION — `examples/server.cpp`

The webui is the bundle of llama-server's compiled SvelteKit, served
verbatim plus our own DOM-injection layer.  All dynamic strings that
end up in HTML / JS contexts go through:

* `html_escape` for inline text (used in the embedded title pin).
* `json(title).dump()` for JS string literals (escapes quotes +
  backslashes via nlohmann).
* `str_replace_all` against literal needles (no regex; can't be
  attacker-tricked into matching unexpected substrings).

The injected scripts run in the user's own browser against the user's
own model; they're not a multi-tenant boundary.  Cross-Origin policy
headers (`Cross-Origin-Embedder-Policy`, `Cross-Origin-Opener-Policy`)
are set on `GET /`.

---

## 10. CONCURRENCY

* `easyai-server` runs ONE engine guarded by `engine_mu`.  Streaming
  paths hold the lock for the duration of the SSE stream.  No
  concurrent `llama_decode` is possible, which matches llama.cpp's
  single-context invariant.
* `n_requests` / `n_errors` / `n_tool_calls` are `std::atomic<...>` —
  safe lock-free counters.
* `easyai::Client` is move-only and not thread-safe by design.  One
  `Client` per concurrent conversation.
* `easyai::Plan` is single-threaded — its `on_change` callback fires
  inside the tool handler, on whichever thread `chat()` ran on.

---

## 11. MEMORY / RESOURCE OWNERSHIP

* No raw `new`/`delete` outside vendored llama.cpp / cpp-httplib code.
* Every native handle is owned by a smart pointer or a value type
  with a custom destructor:
    - `Engine` → `unique_ptr<Impl>` (pImpl).
    - `Client` → `unique_ptr<Impl>` (pImpl).
    - libcurl handles in `http_get` / `http_post_form` are RAII via
      explicit cleanup at the function tail; the linear control
      flow makes leaks trivially auditable.
    - cpp-httplib `httplib::Client` instances are `make_unique` in
      `Client::make_http()` and live for the duration of one request.
* `body.reserve(html.size())` in `strip_html` bounds the output
  allocation upfront — output never exceeds input length, no
  exponential growth.

---

## 12. AGENTIC LOOP — bounded

Both `Engine::chat_continue()` and `Client::run_chat_loop()` cap at
8 hops.  A model in a tool-loop runaway hits the cap and bails with
the last partial answer instead of consuming infinite tokens / RAM.

The model can't escalate by emitting more tool calls per hop because
each `delta.tool_calls` is bounded by the chat template's grammar.
Tool-result content fed back into the model is clipped at the
individual tool's `clip()` budget (typically 8-16 KiB).

---

## 13. KNOWN LIMITATIONS / ACCEPTED RISK

* **TLS verification is required by default for `https://` in
  `easyai::Client`**, and `--insecure-tls` is documented as DEV ONLY.
  We cannot programmatically force admins not to use it; document it
  loudly.
* **Prompt injection is out of scope of this audit.**  The
  authoritative-datetime preamble (`build_authoritative_preamble`)
  does try to harden against post-cutoff hallucination, but a
  determined prompt injector can still steer the model.  This is a
  model-layer concern.
* **Worst-case regex behaviour on user-controlled patterns** —
  fs_grep + fs_glob accept arbitrary regex from the model.  Even
  with try/catch, a sufficiently adversarial pattern can stack-blow
  libstdc++'s regex.  For air-gapped agents that's tolerable; for
  hostile-tenant deployments, consider linking RE2 via a separate
  build option (out of scope for v0.1.0).

---

## 14. RECOMMENDATIONS FOR HIGH-TRUST DEPLOYMENTS

1. Always run easyai-server behind a reverse proxy that terminates
   TLS, enforces destination-IP egress policy (block private ranges
   if you don't need them), and rate-limits failed Bearer attempts.
2. Set `--max-body` lower than 8 MiB if your real-world prompts
   never approach that.
3. Set a strict `--sandbox` directory if you enable any `fs_*`
   tools.  Don't pass `.` (CWD) on a multi-tenant box.
4. Run the unit as a system user (`scripts/install_easyai_server.sh`
   already does this) so a hypothetical RCE only sees what that
   user can read.
5. Keep `--inject-datetime on` to suppress hallucinations about
   "today" / post-cutoff facts.

---

*Last reviewed against commit immediately before the security-fixes
commit.  Re-run when adding a new tool or a new HTTP boundary.*

---

## 15. SECOND PASS — late-2026 additions

### 15.1 SSE pending-buffer growth (MEDIUM, FIXED)

**File:** `src/client.cpp` — `SseBuffer::feed`.

**Issue:** `buf_.append(bytes, n)` had no upper bound.  A malformed
or malicious SSE response that never emits an event terminator
(`\n\n` / `\r\n\r\n`) would push the buffer to OOM.

**Fix:** capped the pending buffer at 16 MiB (`kMaxPending`).  On
overflow, `feed` returns false; the content_receiver propagates the
abort, the request fails cleanly with `last_error` set to
`"SSE pending buffer exceeded 16 MiB — abandoning stream..."`.

### 15.2 Regex-DoS in fs_grep (MEDIUM, FIXED)

**File:** `src/builtin_tools.cpp` — `fs_grep` line iteration.

**Issue:** libstdc++'s `std::regex` is recursive and is famously
vulnerable to catastrophic backtracking on patterns like
`(a+)+$`.  A user-supplied pattern run against a multi-megabyte
single line (binary blob, minified JS, base64 dump) hangs the
tool dispatch thread for minutes.

**Fix:** skip lines longer than 64 KiB before feeding them to
`std::regex_search`.  Bounds worst-case regex work without
touching the typical source-tree case.

### 15.3 Bash command runner (NEW SURFACE — by design)

**File:** `src/builtin_tools.cpp` — `bash` factory.

The `bash` tool is **explicitly NOT a hardened sandbox**.  It runs
`/bin/sh -c <user_supplied_string>` with the caller's full privileges,
inside `cwd` set to the configured root directory.  Mitigations are
cooperative, not isolating:

- 32 KiB output cap (silently truncates with marker)
- Per-command timeout (default 30 s, max 300 s; SIGTERM then SIGKILL
  +2 s grace)
- Server side: requires `--allow-bash` opt-in to register at all
  (default off, never appears in webui's tool list)
- CLI side: `--allow-bash` opt-in in cli/cli-remote
- Hop-cap bump: when bash is enabled, agentic loop runs up to 99999
  hops (other safety nets — per-tool timeouts, output caps,
  retry_on_incomplete — still apply)

For threat models requiring isolation, run `easyai-server --allow-bash`
inside a container / firejail / unprivileged user with disabled
network egress; the tool's contract assumes the OS provides
isolation, not the framework.

### 15.4 Audit-cleared items (NO ACTION)

- format strings — every `*printf(stderr, ...)` reviewed; all use
  literal format strings, no user-string-as-format.
- snprintf buffers — stack buffers (8, 32, 60, 64) reviewed against
  worst-case content; all have ≥4× headroom.
- HTTP body caps — web_fetch 2 MB GET, web_search 4 MB POST
  (`HttpSink::max_bytes`).
- JSON parse exceptions — every `nlohmann::json::parse(user_data)`
  is wrapped in try/catch.

---

## 16. External tools manifest — `src/external_tools.cpp` (NEW SURFACE)

Introduced in commit `d0f7965`, hardened in `e966cf1`. Operator
declares custom commands in a JSON manifest; the library compiles
each entry into a regular `Tool` that the model can dispatch like a
built-in. New surface, audited from scratch in the same pass.

The trust boundary is **operator → model**: the manifest is part of
the operator's deploy artefact (treat it like a sudoers file —
anyone who can write it can run arbitrary commands as the agent's
user). The model only fills in parameter values, which are
type-checked against the per-tool JSON Schema before any process is
spawned.

### 16.1 Guarantees enforced at LOAD time

1. **Absolute command path.** Validated via `stat()` + `S_ISREG` +
   `access(X_OK)`. Relative names rejected → no PATH search → no
   PATH-hijack.
2. **Manifest is a regular file.** `slurp()` does `stat()` first
   and rejects directories, FIFOs, devices, sockets. Stops a
   misconfigured manifest path pointed at `/dev/zero` from spinning instead of
   erroring cleanly.
3. **Whole-element argv placeholders only.** `"{name}"` accepted,
   `"--flag={x}"` rejected. The model's value flows through as one
   whole argv element — quoting, `;`, `$(…)`, backticks, embedded
   newlines cannot escape into adjacent elements.
4. **No name collisions.** Tool names cannot shadow built-ins
   (`bash`, `read_file`, `get_current_dir`, …) or
   already-registered tools. Operator notices on startup, not at
   first call.
5. **Hard caps** (each closes a class of DoS):
   - manifest size: 1 MiB
   - tools per manifest: 128
   - params per tool: 32
   - argv elements: 256
   - per-arg bytes: 4 KiB
   - env passthrough entries: 16
   - timeout: clamped to [100 ms, 300 000 ms]
   - output cap: clamped to [1 KiB, 4 MiB]
   - tool-name length: 64 chars (matches the validator regex)
   - tool-description length: 4096 chars
   - parameter-description length: 2048 chars

### 16.2 Guarantees enforced at CALL time

1. **Schema-validated arguments.** Type errors and missing-required
   args surface as `ToolResult::error` *before* `fork()` is called.
2. **Non-finite numbers rejected.** `nlohmann::json` accepts NaN /
   Inf as valid numbers; we explicitly call `std::isfinite()` so
   the wrapped command never receives the literal string `"nan"` or
   `"inf"` as an argv value.
3. **No shell.** Spawn is `fork()` + `execve(absolute_path, argv,
   envp)`. Never `/bin/sh -c …`.
4. **Closed stdin.** Child gets `/dev/null` on fd 0; the model
   cannot feed bytes into the subprocess.
5. **Bounded fd inheritance.** All fds ≥ 3 closed in the child
   between `fork()` and `execve()`. The close-loop is bounded by
   `kMaxFdScan = 65536` regardless of `RLIMIT_NOFILE`, so
   `ulimit -n unlimited` (which produces `rlim_cur = RLIM_INFINITY`
   ≈ `ULONG_MAX`, casts to `-1`, silently disables a naive loop)
   does NOT leak parent fds into the child.
6. **Linux PDEATHSIG.** Child sets `prctl(PR_SET_PDEATHSIG,
   SIGKILL)`. If the agent process dies (segfault, OOM-kill,
   `kill -9`) before the subprocess finishes, the kernel sends the
   subprocess SIGKILL instead of leaving an orphan reparented to
   PID 1.
7. **Process-group lifetime.** Both child and parent call `setpgid`
   so the parent's `kill(-pid, …)` reaches grandchildren. Timeout
   path: SIGTERM, then SIGKILL after a 1 s grace, then a blocking
   `waitpid` after 5 s if the child still hasn't been reaped
   (uninterruptible-sleep / kernel-bug safety net).
8. **Output capped.** `max_output_bytes` enforced in the read loop
   on the parent side; the child stays unblocked because we keep
   draining the pipe (and discarding overflow) instead of letting
   it stall on a full buffer.
9. **Clean env by default.** Only operator-listed `env_passthrough`
   vars inherit; `LD_PRELOAD`, `PATH`, etc. don't leak in unless
   asked. Each passthrough value is also capped at
   `kMaxArgElementBytes` so a hostile env (`HOME=<2 MB string>`)
   can't push the execve table toward `ARG_MAX`.

### 16.3 Issues found during the second-pass review (FIXED)

| # | Issue | Severity | Fix landed |
| --- | --- | --- | --- |
| 1 | `RLIMIT_NOFILE = RLIM_INFINITY` skipped the close-loop, leaking every parent fd into the child | HIGH | `kMaxFdScan` cap, `RLIM_INFINITY` guard (e966cf1) |
| 2 | `slurp()` on `/dev/zero` / FIFO / dir gave misleading sizes; cap merely limited the damage | MEDIUM | stat-first, reject non-regular (e966cf1) |
| 3 | `nlohmann::json` accepts NaN/Inf; `std::to_string` rendered them as literal `"nan"` / `"inf"` to the wrapped command | MEDIUM | `std::isfinite()` reject (e966cf1) |
| 4 | Orphan subprocess survived agent crash | LOW | `PR_SET_PDEATHSIG(SIGKILL)` on Linux (e966cf1) |
| 5 | `env_passthrough` value length uncapped — 2 MB `HOME` would push toward `ARG_MAX` | LOW | per-value cap = `kMaxArgElementBytes` (e966cf1) |
| 6 | Magic numbers (poll cadence, kill grace, drain buffer, exit codes) inline | quality | promoted to named `constexpr` with rationale (e966cf1) |

### 16.4 Accepted residual risk

- **TOCTOU between `stat()`+`access(X_OK)` at load and `execve()` at
  call.** If the operator's command binary is replaced between manifest
  load and the first call, the agent runs whatever the new file
  contains. Operator-controlled environment, no PATH involved, low
  practical risk. Could be tightened with `O_PATH` + `fexecve()` if
  ever needed; not done.
- **Argv injection via leading dash.** The library guarantees one
  argv slot per model argument; it cannot know whether the wrapped
  binary parses `pattern = "-V"` as a flag. Mitigation is
  manifest-side: insert `"--"` literal before string placeholders.
  Documented in `manual.md` §3.3.4 and demonstrated in
  `examples/EASYAI-example.tools` (`pgrep` entry uses `["-a", "--",
  "{pattern}"]`).
- **`kBuiltInNames` hard-coded list duplicates the actual builtin
  registry.** A future builtin added to `src/builtin_tools.cpp` must
  also be added to the reservation list, or a manifest could
  shadow it. Defence-in-depth: callers also pass their own
  `reserved_names`; the duplicated list only matters if the caller
  forgets. Acceptable given the small surface.

### 16.5 The new `get_current_dir` builtin

Zero-parameter tool that returns `getcwd()` at call time. The CLIs
`chdir()` into `--sandbox` at startup, so what `get_current_dir`
reports is exactly the directory the model's `bash` / `fs_*` tools
operate against. No security implications beyond the
already-audited sandbox semantics — `getcwd` is async-signal-safe
and bounded by `PATH_MAX` on Linux.

### 16.6 Directory loader + per-file fault isolation

The `--external-tools DIR` form (replaces the original `--tools-json
PATH`) scans `DIR` for files matching `EASYAI-<name>.tools` and
loads each one independently. Security-relevant behaviour:

- **Top-level only.** Subdirectories are skipped. Operators use a
  `disabled/` or `archive/` subfolder to keep `.tools` files out of
  the load path without renaming.
- **Pattern is exact and case-sensitive.** A file named
  `easyai-foo.tools` (lowercase) does NOT match. Same for
  `EASYAI-foo.json`. Disabling without deletion is achieved by
  adding any suffix after `.tools` (e.g. `.tools.disabled`,
  `.tools.bak`).
- **Deterministic load order.** Filenames are sorted before loading
  so duplicate-name resolution is stable across machines and
  reboots: alphabetically-earlier file wins; later file's load
  fails with a collision error AND the rest of that file's tools
  are skipped (file-level all-or-nothing within the dir-level
  per-file isolation).
- **Per-file fault isolation.** A parse / schema / sanity error in
  `EASYAI-experimental.tools` does NOT prevent
  `EASYAI-system.tools` from loading. The agent starts; the
  operator sees the error in the journal.
- **Empty dir is a normal state.** No error, no warning. Loading
  zero tools from a dir that exists is the design — operators can
  enable `--external-tools` in advance and add tools later.

### 16.6b RAG — persistent registry surface

Lives in `src/rag_tools.cpp`. Five tools (`rag_save`, `rag_search`,
`rag_load`, `rag_list`, `rag_delete`) the agent uses to write to
and read from a directory of small Markdown files. Path-traversal
is the only security-relevant primitive; the rest is correctness.

- **Path-traversal closed at the regex.** Title and keyword
  identifiers must match `^[A-Za-z0-9._+-]{1,64}$` /
  `^[A-Za-z0-9._+-]{1,32}$`. Slashes, dots, NULs, spaces are
  rejected at validation. The validated title is concatenated as
  `<title>.md` to the configured root and passed to `std::ofstream`
  / `std::ifstream`. There is no path-traversal surface — `..`,
  `/etc/passwd`, `/proc/self/mem` etc. are all blocked at parse.
- **Atomic writes.** Tempfile + `rename(2)`. Concurrent readers
  always see either the old or new full content, never partial.
- **Bounded reads.** Slurp is capped at 256 KiB + 4 KiB header
  slack. A symlink-based attempt to read `/dev/zero` or a giant
  file via the title is impossible (title regex), but the cap is
  defence-in-depth for hand-edited dirs.
- **No JSON parser involvement.** The on-disk format is plain text
  with one `keywords:` header. Corrupt files are silently skipped
  at index time; the index doesn't crash the agent.
- **Mutex-guarded index.** All five tools share an in-memory
  `std::map<title, EntryMeta>` guarded by a `std::mutex`. The
  body is read off-disk per `rag_load`, never cached, so memory
  doesn't grow with entry count beyond the metadata.
- **Filesystem permissions are the access boundary.** The installer
  creates `/var/lib/easyai/rag/` mode 750 owned by the `easyai`
  service user. The agent is the only thing that can read or
  write. Sharing across processes / users is an OS-level concern;
  RAG inherits whatever ACLs the operator put on the dir.

Not in scope:

- **No encryption at rest.** Operator decides whether to encrypt
  the underlying filesystem. Roadmap: optional symmetric
  encryption with a key from env var (`RAG.md` §10).
- **No audit log of writes.** The mtime is the only signal. A
  future "history" tool could keep a log; not implemented.
- **No multi-tenant namespacing.** One process, one RAG dir.
  Multi-user requires the operator to run multiple servers or
  wait for the roadmap item.

### 16.7 Security sanity-check warnings (new audit pass)

Beyond the load-time hard rejections (§16.1), the loader runs a
non-fatal audit on each parsed spec. These produce human-readable
warnings to stderr (or journal); the tool still loads. Quiet mode
(`-q` on `easyai-cli` / `easyai-local`) suppresses warnings,
preserving errors. Server (daemon) always logs both.

Warning classes:

| Class | Trigger | Why it's flagged |
| --- | --- | --- |
| Shell wrapper | `command` is sh/bash/dash/zsh/ksh/ash/fish AND argv has `-c {placeholder}` | Reintroduces shell-injection surface that the manifest design exists to remove. |
| Shell binary as command (no -c) | `command` is a shell binary, argv shape isn't standard `-c {placeholder}` | argv may still reach shell parsing depending on shell mode. |
| Dynamic-linker env passthrough | `env_passthrough` includes `LD_PRELOAD`, `LD_LIBRARY_PATH`, `LD_AUDIT`, `DYLD_INSERT_LIBRARIES`, `DYLD_LIBRARY_PATH` | Lets the model influence dynamic linking in the subprocess. Almost always wrong. |
| World-writable command | wrapped binary has `S_IWOTH` mode bit | Anyone with shell on the host can replace it; agent runs the replacement on next call. |
| World-writable manifest | the `.tools` file has `S_IWOTH` mode bit | Anyone with write access can register additional tools that survive the next restart. |

The warnings are intentionally narrow — false positives would
train the operator to ignore them. False negatives (a manifest
the audit *fails* to flag but should) are a real risk; the audit
is best-effort, not a substitute for code review.

