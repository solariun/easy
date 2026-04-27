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
