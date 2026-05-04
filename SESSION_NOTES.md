# SESSION NOTES — easyai

> Context dump for resuming work in a fresh chat session.  Paste this into
> the new conversation (or point at the file URL on github.com/solariun/easy).

## 1. Project at a glance

`easyai` is a C++17 framework around `llama.cpp` that ships **two
libraries** (`find_package(easyai)` exports `easyai::engine` and
`easyai::cli`) plus six binaries:

| Artifact              | Type    | Role                                                                                                                                    |
|-----------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `libeasyai`           | library | local llama.cpp engine — `Engine`, `Tool`, `Plan`, built-in tools (datetime, web_search/fetch, fs_*), presets.  Linked via `easyai::engine`.    |
| `libeasyai-cli`       | library | OpenAI-protocol client — `Client` mirrors `Engine`'s fluent API but the model runs remote and tools execute locally.  Linked via `easyai::cli`. |
| `easyai-local`        | binary  | Local-only REPL: loads a GGUF in-process via `easyai::Engine`. Drop-in `llama-cli` replacement.                                       |
| `easyai-cli`          | binary  | Agentic OpenAI-protocol client built on `libeasyai-cli` — no local model.  REPL or `-p`, full sampling control, plan tool, server-management subcommands. |
| `easyai-server`       | binary  | Drop-in `llama-server` replacement: embeds llama-server's SvelteKit webui, no MCP                                                      |
| `easyai-agent`        | binary  | Demo agent showing every built-in tool                                                                                                  |
| `easyai-chat`         | binary  | Bare REPL                                                                                                                               |
| `easyai-recipes`      | binary  | Tutorial agent paired with manual.md ch. 3.8                                                                                            |

GitHub: **https://github.com/solariun/easy** (branch `main`, tag `v0.1.0`
is the first formal release).
Sibling repo path: `develop/easyai/easyai/` next to `develop/easyai/llama.cpp/`.

## 2. User context (important — read this)

- **Owner**: Gustavo Campos (`solariun` on GitHub, `lgustavocampos@gmail.com`).
- **Communicates in Portuguese** (Brazilian).  Mix Portuguese in replies; Brazilian tech jargon is fine.
- **Hardware (the "AI box")**: MINISFORUM UM690L Slim — Ryzen 9 6900HX (8C/16T) +
  Radeon 680M iGPU (RDNA2, gfx1035) + 32 GB DDR5.  Linux + Vulkan/RADV.  GTT
  set to **28 GiB** via `ttm.pages_limit=7340032` kernel cmdline.
- **Models in use**:
  - Qwen2.5-3B-Instruct-Q4_K_M (dev / smoke testing on Mac M3)
  - Qwen3.6-35B-A3_eng_q4_k_m.gguf (production on AI box; MoE, ~22 GB on GPU)
  - Gemma4-31B Q4_K_M (alternate)
- **Preferences observed**:
  - Wants visible status / observability (status pills, tokens/s, context usage).
  - Hates flaky UI behaviour (flickering stats, panels out of place).
  - Hates relying on bundle-internal renderers — prefers our own DOM injection.
  - Wants thinking blocks rendered as **small monospace gray collapsible
    panels** above the message body (NOT inline as plain text).
  - Wants tone preset selector (`deterministic / precise / balanced / creative`)
    near the input.
  - Wants metrics (`ctx X/Y · last N tok · s · t/s`) BELOW the input, lifted
    so it doesn't overlap the textarea.
  - Wants tool calls visible as a **single-line indicator** beside the
    copy/edit/fork/delete actions (NOT a big body of text).
  - Frustrated when something is "easy and yet you didn't deliver".
- **Replaces `install_llama_server.sh`** (a 3000-line bash installer in the
  user's gist).  Keeps the same flag set (drop-in compat) but cuts MCP /
  SearXNG / proxy.  Ours: `scripts/install_easyai_server.sh`.

## 3. Repo layout

```
easyai/
├── include/easyai/{engine,tool,builtin_tools,presets,plan,client,
│                   ui,text,log,cli,backend,agent,easyai}.hpp                  # public API
├── src/{engine,tool,builtin_tools,presets,plan,client,
│        ui,log,cli,cli_client,backend,agent}.cpp                              # impl
├── examples/{local,cli,server,agent,chat,recipes}.cpp                         # binaries
├── webui/{index.html,bundle.js,bundle.css,loading.html,AI-brain.svg}          # llama-server fork
├── cmake/{xxd.cmake,easyaiConfig.cmake.in}                                    # build helpers + find_package
├── scripts/install_easyai_server.sh                                           # Linux installer
├── README.md  design.md  manual.md  SESSION_NOTES.md (this file)
└── CMakeLists.txt
```

Build: `cmake -S . -B build && cmake --build build -j` (Metal auto on macOS;
`-DGGML_VULKAN=ON` / `-DGGML_CUDA=ON` / `-DGGML_HIP=ON` for other GPUs).
Selective targets: `cmake --build build --target easyai|easyai_cli|easyai-server|easyai-cli|easyai-local|...`.

Install: `cmake --install build --prefix /usr/local` lays out
`<prefix>/lib/libeasyai{,-cli}.so.0.1.0`,
`<prefix>/include/easyai/*.hpp`, and
`<prefix>/lib/cmake/easyai/easyaiConfig.cmake` so downstream projects
do `find_package(easyai 0.1 REQUIRED)` + `target_link_libraries(myapp
PRIVATE easyai::engine easyai::cli)`.  The hand-rolled config dodges
install(EXPORT)'s issue with the `EXCLUDE_FROM_ALL` llama.cpp subdir
by creating IMPORTED targets at find_package time.

## 4. Architecture

**Engine** (`src/engine.cpp`)
- pImpl with `common_init_from_params` from llama.cpp's common library.
- `chat()` runs the agentic tool-call loop (default cap 8 hops, configurable
  via `Engine::max_tool_hops(int)` — bumped to 99999 when bash registers);
  `generate_one()` is single-pass for forwarding tool_calls to clients.
- Has setters for every tunable: `temperature/top_p/top_k/min_p/repeat_penalty/
  max_tokens/seed/batch/threads/threads_batch/cache_type_k/cache_type_v/
  no_kv_offload/kv_unified/add_kv_override/flash_attn/use_mlock/use_mmap/
  numa/enable_thinking`.
- Public `chat_params_for_current_state(bool)` returns the rendered
  `common_chat_params` so the HTTP layer can do incremental parsing.
- `perf_data()` wraps `llama_perf_context()` for tokens/s timings.
- Three-layer recovery in `parse_assistant` for malformed tool_calls:
  (1) Qwen-style doubled-brace JSON, (2) Hermes-XML `<function=NAME>/
  <parameter=K>V`, (3) markdown indicator `*🔧 NAME(args)*` (when the
  model abandons the structured format mid-conversation).  Wrench-emoji
  byte sequence (F0 9F 94 A7) is the gate so prose doesn't trip recovery.
- `extract_think_block` splits `<think>...</think>` out of raw content
  when the official parser threw, so the streaming layer's last-resort
  fallback never re-emits reasoning that was already streamed as
  `delta.reasoning_content`.

**Client** (`src/client.cpp`, libeasyai-cli)
- pImpl with raw OpenAI-shape JSON history strings — no nlohmann::json
  in the public ABI.
- HTTP/SSE via cpp-httplib; URL parser supports `http(s)://host[:port][/base]`.
  HTTPS reports a clear "not in this build" error (no OpenSSL link yet).
- Streaming SSE parser handles `\n\n` / `\r\n\r\n` terminators, ignores
  comments and our server's UI-only `easyai.tool_call`/`tool_result`
  custom events.
- Tool-call deltas accumulated by `index` into `PendingToolCall`
  (string-built `arguments` across deltas — exactly how OpenAI streams
  them).  Multi-hop loop default-bounded at 8 (`Client::max_tool_hops(int)`
  to lift), mirrors `Engine::chat_continue`.
- SSE pending buffer capped at 16 MiB; on overflow the client aborts
  with `last_error` set.  Stops a malformed/runaway stream from OOMing
  the process.
- Auto-retry-with-nudge on `timings.incomplete=true`: discards the
  bad assistant turn, appends a corrective user message
  ("don't announce, execute"), re-issues once.  Default ON in
  `easyai-cli`; opt-out with `--no-retry-on-incomplete`.
- Full sampling/penalty surface: `temperature`, `top_p`, `top_k`, `min_p`,
  `repeat_penalty`, `frequency_penalty`, `presence_penalty`, `seed`,
  `max_tokens`, `stop(vector)`, `extra_body_json` (free-form JSON merged
  last so it overrides anything the typed setters wrote).
- Direct-endpoint helpers (`list_models`, `list_remote_tools`, `health`,
  `metrics`, `props`, `set_preset`) round out the SDK so downstream apps
  can manage an easyai-server without touching curl.

**Plan** (`src/plan.cpp`, `include/easyai/plan.hpp`)
- In-memory checklist of `{id, text, status}` items + `ChangeCallback`.
- `Plan::tool()` returns a single `Tool` with `action=add|start|done|list`
  schema — wires into `Engine::add_tool` or `Client::add_tool` the same way.
- `render_string()` produces a GitHub-style markdown checklist
  (`- [ ] / [~] / [x]`).

**Authoritative datetime + cutoff hint** (`examples/server.cpp`,
`build_authoritative_preamble` + `prepare_engine_for_request`)
- Per-request fresh timestamp injected into the system prompt: current
  date+time+TZ + knowledge-cutoff rule (verify post-cutoff facts via
  tools or say "not certain").  Default ON via `--inject-datetime on`.
- Preamble is APPENDED to whichever system message reaches the model:
  if the client sent its own `system` (opencode / Claude-Code style),
  we splice the preamble into the LAST one of those; otherwise we
  append to ctx.default_system.  Either way, exactly one system block
  reaches the chat template.
- Per-request override via `X-Easyai-Inject: on|off` HTTP header.
  Defaults to the server-side flag if header absent or unrecognised.

**Server** (`examples/server.cpp`)
- cpp-httplib + nlohmann::json (vendored by llama.cpp).
- Endpoints: `GET /` (webui), `/bundle.{js,css}`, `/loading.html`, `/favicon{.ico}`,
  `/health`, `/metrics`, `/v1/models`, `POST /v1/chat/completions` (OpenAI-compat
  with `stream:true`), `POST /v1/preset`.
- Streaming pipeline now mirrors **llama-server's exact approach**:
  accumulate raw text → `common_chat_parse(text, is_partial=true)` per token
  → `common_chat_msg_diff::compute_diffs(prev, new)` → emit standard OpenAI
  deltas with `delta.reasoning_content` / `delta.content`.  Tool calls
  surfaced via custom SSE events (`event: easyai.tool_call`,
  `event: easyai.tool_result`) AND an inline single-line indicator (`*🔧 name*`).
- Per-request override: when client supplies `system` and/or `tools`, those
  override server defaults for that one call (key feature for opencode /
  claude-code / OpenAI clients).
- `--api-key` Bearer auth on `/v1/*`.

**Webui** = llama-server's compiled SvelteKit bundle at `webui/`, embedded
into the binary at build time via `cmake/xxd.cmake`.  Customisations live in
two layers:

1. **At-startup string substitution on `bundle.js`**:
   - `>llama.cpp</h1>` → `>{title}</h1>` (sidebar + welcome brand)
   - `llama.cpp - AI Chat Interface` → `{title}` (page title)
   - `Initializing connection to llama.cpp server...` → `... {title} server …`
   - `} - llama.cpp` → `} - {title}` (per-conversation page title)
   - `Type a message...` / `Type a message` → `--webui-placeholder`

2. **At-runtime injection into `<head>` of served `index.html`**:
   - Title pin via `Object.defineProperty(document, 'title', {set: ...})`.
   - LocalStorage seed: `LlamaCppWebui.mcpDefaultEnabled=false` +
     `keepStatsVisible=true`, `showMessageStats=true`.
   - DOM scrubber: MutationObserver finds elements whose visible text matches
     `MCP Servers` / `MCP Prompt` / `Sign in` / `Login` / `Authorize` /
     `Load model` / `Use Pyodide` etc and hides their container ancestors.
   - `fetch` interceptor: 501s `/authorize`, `/token`, `/register`,
     `/.well-known/*`, `/models/load`, `/cors-proxy`, `/dev/poll`,
     `/home/web_user/*`; stubs `/properties` with `{}`.
   - **Tone chip + metrics bar** (single Shadow DOM host
     `__easyaiBarHost`) at `bottom: 0.55rem`, centred — has tone selector
     (`deterministic / precise / balanced / creative` → set via fetch
     interceptor injecting temperature into request body) + ctx + last
     response stats.
   - **Per-message status chip**: appended to the action toolbar (next to
     copy/edit/fork/delete) of each `[aria-label="Assistant message with
     actions"]`; shows `thinking` / `answering` / `fetching · <tool>` during
     streaming, then `complete · 98 tok · 4.4s · 22.3 t/s`.
   - **Per-message thinking panel** (`<details class="__easyai-thinking">`):
     created on first `delta.reasoning_content`, plain text body, max-height
     18em, monospace gray, click to expand/collapse, auto-collapses on first
     `delta.content` or `finish_reason`.
   - SSE monitor inside fetch interceptor: drives the status pill, thinking
     panel, and metrics bar from the streamed events.
   - Input form gets `margin-bottom: 44px !important` injected so the
     metrics bar fits below it without overlap.

**CLI utilities** (`src/{ui,text,log,cli,cli_client}.cpp` + matching headers)
Lifted out of the example binaries during the 2026-04-27 refactor so a
third-party agent can build the same CLI experience in a handful of lines.
- `easyai::ui::Style` + `detect_style()` — ANSI colour helpers; auto-disabled
  on non-TTY stdout and when `NO_COLOR` is set.
- `easyai::ui::Spinner` — `'|/-\\'` glyph that follows the cursor; throttled
  frame advance (~10 Hz) plus a heartbeat thread that keeps the animation
  alive during dead air (slow tool calls, hidden reasoning).  Exposes a
  RAII `WriteScope` AND a one-call `write(text)` for the common path.
- `easyai::ui::StreamStats` — counters + timing that on_token/on_reason/
  on_tool callbacks update; lets the verbose summary show
  "[hop N: content=… reason=… tools=… +Tms]".
- `easyai::text::*` — `slurp_file`, `punctuate_think_tags` (newlines around
  `<think>`/`</think>` tags so they don't visually glue to the stream),
  `prompt_wants_file_write` heuristic for the missing-fs_write_file tip.
- `easyai::log::set_file(FILE*) / write(fmt, …)` — single-sink tee to
  stderr + an optional `--log-file` FILE.  `easyai-cli`'s vlog is now a
  4-line wrapper around this; libeasyai-cli ALSO writes raw SSE bytes
  to the SAME FILE via `Client::log_file(fp)` so timestamps interleave.
- `easyai::cli::Toolbelt` — fluent builder.  `.sandbox(dir).allow_bash()
  .with_plan(plan).apply(engine_or_client)`.  apply() bumps
  `max_tool_hops` to 99999 when bash is enabled.  Engine variant lives
  in libeasyai (`src/cli.cpp`); Client variant in libeasyai-cli
  (`src/cli_client.cpp`) so the engine-only library doesn't drag in
  the HTTP client.
- `easyai::cli::open_log_tee / close_log_tee` — opens
  `/tmp/<prefix>-<pid>-<epoch>.log` with a header listing argv,
  registers it as the global log sink.
- `easyai::cli::validate_sandbox(path, &err)` — uniform "exists? is a
  dir?" check.

**Backend abstraction** (`include/easyai/backend.hpp`,
`src/backend.cpp` + `src/cli_client.cpp`)
The local↔remote unification.  `easyai::Backend` is the abstract
interface (`init/chat/reset/set_system/set_sampling/info/tool_list/
tool_count/last_error`).  `easyai::LocalBackend` wraps `Engine` and
ships in libeasyai; `easyai::RemoteBackend` wraps `Client` and ships
in libeasyai-cli — so the engine-only library doesn't drag in the
HTTP client unless the consumer actually links against it.  Each has
a public `Config` struct (sandbox, allow_bash, preset, sampling
overrides, KV cache controls for local, TLS knobs for remote).

**Agent — Tier-1 façade** (`include/easyai/agent.hpp`,
`src/agent.cpp` in libeasyai-cli)
The "extremely easy for all skill levels" entry point.  Three-line
hello world: `easyai::Agent a("model.gguf"); a.ask("…");`.  Remote
variant via the static factory `easyai::Agent::remote(url, key?)`.
Fluent setters (`system / sandbox / allow_bash / preset /
remote_model / temperature / top_p / top_k / min_p / on_token`) all
queue into the underlying Backend's Config; the Backend is built
lazily on first `ask()`.  Escape hatch: `agent.backend()` returns the
materialised `Backend &` for everything Agent doesn't surface
directly.  Default toolset: datetime + web_search + web_fetch on;
fs_* and bash off until the user opts in.

**The four-tier API rule** (Gustavo's standard pattern, codified
2026-04-27):

  1. **Tier 1 — 3-line "hello world"** via `Agent` (or any future
     façade) with sensible defaults.
  2. **Tier 2 — fluent customisation** (`Toolbelt`, `Streaming`,
     `Agent` setters) chainable on the same façade.
  3. **Tier 3 — explicit composables** (`Engine`, `Client`,
     `Backend`, `Tool::builder`).  Users who outgrow Tier 1/2 step
     here without rewriting.
  4. **Tier 4 — escape hatch** (`Agent::backend()`, raw llama.cpp
     handles, raw HTTP).  Power users never hit a wall.

  Higher tiers are ALWAYS implemented on top of lower tiers — never
  parallel codepaths.  That's how Tier 1 stays trustworthy.

**Deep — default assistant persona** (server.cpp's `kBuiltinSystem`)
A fresh `easyai-server` boots up as **Deep**, an expert system
engineer who answers from CHECKED FACTS.  Operating loop:
`TIME → THINK → PLAN → EXECUTE → VERIFY`.  "Time first" is its own
rule — `datetime` is the first tool call any time the answer touches
"now", "today", a deadline, a release version, or a fact that could
have changed since cutoff.  Operators who want a different persona
pass `--system` or `-s` — Deep is the default, not hardcoded.
Webui title default also flips to `"Deep"`.

**Installer** (`scripts/install_easyai_server.sh`)
- Linux/Debian only.  Backend auto-detect: `nvidia-smi` → CUDA;
  `rocminfo` → ROCm/HIP; `vulkaninfo` / AMD `lspci` → Vulkan; else CPU.
- Installs libs to `/usr/lib/easyai/` (isolated from system) +
  `LD_LIBRARY_PATH=/usr/lib/easyai` in systemd unit (no ldconfig dependency).
- Drop-in compat: accepts every flag from `install_llama_server.sh` —
  unsupported features (MCP, draft model, webui-title-rebuild, thinking-
  budget, list-tags) become friendly no-op warnings.
- Systemd unit hardening: `OOMScoreAdjust=-700`, `CPUSchedulingPolicy=fifo`
  priority 50, `LimitMEMLOCK=infinity`, `RADV_PERFTEST=gpl`, `mlock`,
  `--no-mmap`, `flash-attn`, `q8_0` KV cache, render+video groups.
- `--upgrade` does `git fetch + git pull --ff-only` (was just fetch — bug
  fixed in commit `e03705e`).

## 5. Recent commits (most recent first)

```
2026-05-04 — RAG default flipped to single-tool dispatcher; --split-rag
                 opts back into legacy seven; concise default prompt.

(pending commit) Two user-driven changes:

  (1) RAG default flip. The unified `rag(action=...)` dispatcher
      used to be opt-in behind --experimental-rag; it is now the
      DEFAULT for every binary (easyai-server, easyai-cli,
      easyai-local, easyai-mcp-server). The legacy seven separate
      `rag_*` tools are still available behind a renamed flag,
      --split-rag (and INI key SERVER.split_rag), for operators
      driving weak / 1-bit-quant tool callers (Bonsai-class) that
      handle many flat schemas more reliably than one
      discriminated schema. On-disk format, locking, and
      fix-memory rules unchanged — only the catalog shape
      differs. Files: include/easyai/rag_tools.hpp (comments
      flipped), src/rag_tools.cpp (description prose, drop
      "EXPERIMENTAL —" prefix, dispatcher section header),
      include/easyai/backend.hpp + src/backend.cpp (Config gains
      split_rag bool; LocalBackend picks the factory by it),
      examples/cli.cpp (--split-rag flag, register_tools branch
      flipped, --tools/--RAG help refreshed), examples/server.cpp
      (FlagDef table: split_rag/--split-rag/SERVER.split_rag,
      registration block flipped, --RAG/--split-rag help
      refreshed), examples/mcp_server.cpp (same FlagDef + INI
      key + registration flip), examples/local.cpp (--split-rag
      arg, plumbed into LocalBackend::Config). All four affected
      binaries build clean. easyai-cli --list-tools --RAG <dir>
      reports "rag" by default; with --split-rag added, reports
      the seven legacy names. Server INI sanity-check confirms
      [SERVER] split_rag = on logs "RAG enabled (split: seven
      rag_* tools)" while the absence logs "RAG enabled (single
      rag tool)".

  (2) Default system prompt rewritten to emphasise plan → act →
      iterate in tight steps, kept brief on purpose so the user
      has room to refine. examples/server.cpp's kBuiltinSystem
      and examples/local.cpp's kBuiltinSystem went from ~30-90
      lines of operating-loop / rule prose down to ~20-30 lines
      that say: answer briefly; for real work, plan ONE small
      next step, act on it in the same turn, read the result,
      then finish or take ONE more step; stop as soon as you
      have something useful. Retains the no-announce-without-
      call rule and the search → fetch discipline. The server
      prompt is what plays through /v1/chat/completions when
      no client supplies its own system message; easyai-cli
      remains a thin client and uses whatever prompt the
      backing server defaults to.

  Docs updated: README.md (new What's new entry + every
  --experimental-rag mention rewritten to --split-rag with
  flipped polarity in the options/flags tables), RAG.md (intro
  example switched to rag(action=...), §4 dispatcher rewritten
  as default + opt-in seven), easyai-server.md (INI table row
  + "All options" row flipped), LINUX_SERVER.md (default-paths
  block + tool-count math reflect new default), MCP.md (catalog
  references mention both layouts), manual.md (RAG quickstart
  shows both factories), design.md ("Why seven tools" rewritten
  as "One tool by default, seven under --split-rag" + four-tier
  table updated), easyai-mcp-server.md (tool-source table now
  has BOTH rows), SECURITY_AUDIT.md (mutex paragraph clarifies
  both layouts share the same RagStore + locks).

2026-05-02 (later) — RAG: rag_append + user-focus prompts.

(pending commit) Adds a seventh tool to RagTools: rag_append.
                 Read-modify-write on an existing memory: reads
                 body off disk, appends new content after a
                 Markdown horizontal rule (`---`), atomically
                 rewrites via tempfile + rename (same
                 save_locked path rag_save uses). The whole RMW
                 runs under one std::unique_lock<shared_mutex>,
                 so concurrent appenders to the SAME title queue
                 cleanly (both appendices land), and concurrent
                 saves/deletes/reads serialise / parallelise via
                 the existing reader/writer discipline. Refuses
                 on titles that don't exist, on fix-easyai-* (the
                 immutability invariant covers append the same
                 way it covers save/delete), and when merged
                 size > kMaxContentBytes (256 KiB).
                 Optional keywords[] arg merges into the existing
                 keyword list (deduped, total still capped at 8;
                 oldest wins on overflow). New handler:
                 make_append_handler in src/rag_tools.cpp;
                 wired into make_rag_tools as RagTools::append
                 and into make_unified_rag_tool as
                 action="append" (kSubs entry added so legacy
                 prose references like rag_append in the inner
                 description get rewritten to
                 rag(action="append") inside unified mode).
                 User-focus prompt update: rag_save and
                 rag_append descriptions now explicitly tell the
                 model to prioritise notes about the user
                 themselves (name, role, hardware, projects,
                 working style, corrections, likes, dislikes)
                 and prefer rag_append on the existing profile
                 memory over rag_save (which would overwrite).
                 Suggested user-* titles: user-profile,
                 user-prefs, user-projects, user-hardware,
                 user-corrections.
                 All four consumers updated to register the new
                 tool: examples/server.cpp, examples/mcp_server.cpp,
                 examples/cli.cpp (with tools_enabled gating
                 mirroring the rest), src/backend.cpp (used by
                 easyai-local + easyai-chat). Help-text strings
                 also updated — the lib went from 5/6 to seven
                 tools and every CLI reference (server.cpp,
                 cli.cpp, local.cpp, mcp_server.cpp) now lists
                 the canonical seven names.
                 Docs: README.md (What's new + options table),
                 RAG.md (TOC, intro, quickstart, full
                 rag_append section in §4, dispatcher §,
                 tool-flow ASCII diagram restructured into
                 write-row + read-row), manual.md, design.md
                 (renamed "Why five tools" → "Why seven tools"),
                 LINUX_SERVER.md, MCP.md (3 places),
                 easyai-server.md (INI table + flag reference +
                 cross-refs), easyai-mcp-server.md, SECURITY_AUDIT.md
                 (§16.6b reflects shared_mutex + seven tools).
                 All 7 binaries build clean. easyai-cli
                 --list-tools confirms the new tool registers
                 correctly (16 tools when --RAG is on: 9 default
                 + 7 RAG).

2026-05-02 — Fourth-pass security audit + readability batch.

(commits 5143799 + b44b615) Two small commits, no public API
                 change. Fourth security pass found one MEDIUM:
                 the auto-generated transaction log at
                 /tmp/easyai-<pid>-<epoch>.log was opened with
                 std::fopen("w") (follows symlinks) and process
                 umask (typically 0644 → world-readable). Path is
                 predictable (16-bit PID + 1-second epoch), so a
                 local attacker on a multi-tenant host could
                 plant a symlink at the predicted path pointing
                 at any user-writable file (~/.bashrc, ~/.ssh/…)
                 and have the next easyai-* process truncate-
                 and-overwrite it. Mode 0644 also leaked prompts
                 (which can contain API keys or PII) to other
                 accounts on the same box. Fixed in src/log.cpp
                 (auto_open) and src/cli.cpp (open_log_tee) by
                 swapping fopen for ::open with
                 O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,
                 mode 0600, then ::fdopen. Caller-supplied paths
                 keep O_TRUNC for log rotation but still gain
                 O_NOFOLLOW + 0600. Validated with a standalone
                 symlink-attack smoke test (errno=EEXIST,
                 victim file untouched). Documented in
                 SECURITY_AUDIT.md §19. The same audit pass
                 cleared nine other findings as already-mitigated
                 / by-design / accepted-risk (§19.2).
                 Readability batch: three inline patterns lifted
                 into named helpers — file_mtime_unix() in
                 rag_tools.cpp (3 dup), glob_to_regex() +
                 kGlobRegexMetachars in builtin_tools.cpp, and
                 looks_like_announce_phrase() in engine.cpp
                 (used twice inside chat_continue). Helpers carry
                 the WHY comments; call sites are now one line.
                 Net diff +103/-85 across three files; all 7
                 binaries build clean.

2026-04-30 (late evening) — Central INI config (/etc/easyai/easyai.ini).

(pending commit) Single config file replaces ~17-flag systemd
                 ExecStart. Sections: [SERVER], [ENGINE], [MCP_USER],
                 [TOOLS] (reserved). Precedence: CLI > INI > hardcoded.
                 New `--config <path>` flag (default
                 /etc/easyai/easyai.ini); missing file = use defaults
                 + open MCP. INI parser in src/config.cpp (no deps,
                 ~80 lines, tolerates missing/malformed lines as
                 warnings). cli_set tracks explicit CLI flags so
                 INI applies only as defaults. New `--no-mcp-auth`
                 force-opens /mcp even with [MCP_USER] populated.
                 [SERVER] mcp_auth=on/off/auto INI-side equivalent.
                 [MCP_USER] populates Bearer-token auth: each line
                 `name = token`, audit log shows
                 `[mcp] request from user 'name'`. Empty section
                 = open. Install script writes a fully-populated
                 easyai.ini on fresh install; leaves it alone on
                 --upgrade (operator edits win). Systemd ExecStart
                 shrunk from ~17 flags to:
                   easyai-server --config /etc/easyai/easyai.ini -m <model>
                 [+ --api-key '${EASYAI_API_KEY}' if /etc/easyai/api_key
                  exists; + --webui-icon if installer was given one].
                 NEW doc INI_KFlags.md — full key reference, every key with
                 type / CLI equivalent / default / notes; worked
                 examples (minimal local-dev / production-with-auth /
                 CLI overrides); cross-refs from README, MCP.md,
                 LINUX_SERVER.md, SECURITY_AUDIT.md.

2026-04-30 (evening) — easyai-server speaks MCP + Ollama list-models.

(pending commit) MCP server: POST /mcp endpoint exposes the full
                 tool catalogue (built-ins + RAG + external-tools) as
                 a JSON-RPC 2.0 MCP provider. Stateless request/
                 response, methods: initialize / tools/list /
                 tools/call / ping plus notification no-ops. Pure
                 function dispatcher in src/mcp.cpp — no global
                 state, never throws. Tool exceptions become MCP
                 isError=true (not JSON-RPC errors).
                 Auth: OPEN by V1 design (Bearer-gate planned for
                 V2; SECURITY_AUDIT.md §17.1 documents accepted
                 risk and compensating controls).
                 Compatibility shims: GET /api/tags + /api/show
                 (Ollama-shape list-models for LobeChat / OpenWebUI
                 / Continue's Ollama provider). /v1/models was
                 already there (OpenAI). /health gains a `compat`
                 block listing all three protocol surfaces.
                 Stdio bridge: scripts/mcp-stdio-bridge.py — Python,
                 stdlib only — for Claude Desktop which only speaks
                 stdio MCP. Cursor / Continue talk HTTP direct.
                 NEW doc MCP.md with quickstart, per-client
                 connection cookbook (Claude Desktop / Cursor /
                 Continue / custom), auth roadmap, troubleshooting.
                 README, design.md §6c, LINUX_SERVER.md, and
                 SECURITY_AUDIT.md §17 all updated.

2026-04-30 (afternoon) — RAG: agent's persistent registry / long-term memory.

(pending commit) RAG: a tag-keyed file-backed long-term memory the
                 model writes to via 5 tools (rag_save, rag_search,
                 rag_load, rag_list, rag_delete). Directory of
                 .md files <title>.md, format `keywords: a, b, c\n
                 \n<body>` — hand-editable, grep-able, scp-able. Title
                 regex closes path traversal. Bounded sizes (title 64,
                 keyword 32, max 8/entry, body 256 KiB, max 4 loads/call).
                 Atomic writes (tempfile + rename). In-memory index
                 lazy-built. New `--RAG <dir>` flag on easyai-server,
                 easyai-cli, easyai-local; install script creates
                 /var/lib/easyai/rag (owned by service user, 750) and
                 systemd unit always passes the flag. Tool descriptions
                 actively encourage save-aggressively /
                 search-before-assuming / delete-stale. New top-level
                 docs: RAG.md (full guide w/ workflows, document
                 ingestion cycle, roadmap) and LINUX_SERVER.md (operator
                 guide for the systemd-installed server, file layout,
                 perf tips, gotchas, API examples, backup/upgrade).

2026-04-30 (morning) — External tools v2: directory loader + sanity-check warnings.

(pending commit) External tools: rename --tools-json PATH to
                 --external-tools DIR, scanning EASYAI-*.tools files in
                 a directory. Per-file fault isolation: a syntax/schema
                 error in one file is logged and skipped, others still
                 load. Empty dir is a normal state (silent, no
                 error/warning). Sanity-check pass at load: warns on
                 shell wrappers, dynamic-linker env passthrough
                 (LD_PRELOAD etc), world-writable command binaries,
                 world-writable manifest files. easyai-cli -q
                 suppresses warnings; load errors always emit. Install
                 script creates /etc/easyai/external-tools/ empty +
                 README + EASYAI-example.tools.disabled. New top-level
                 doc EXTERNAL_TOOLS.md (collaboration guide with 10
                 recipes, anti-patterns, corner cases, troubleshooting).
                 README sales pitch + cross-link.

2026-04-29 — External tools v1 (operator-defined commands via JSON manifest).

e966cf1  External tools: harden child setup, reject NaN/Inf, retire magic numbers
                    (review follow-up: kMaxFdScan caps the close() loop so
                     RLIMIT_NOFILE=infinity no longer leaks parent fds; PR_SET_PDEATHSIG
                     ties subprocess lifetime to the agent; std::isfinite() rejects
                     NaN/Inf model args; slurp() stat-first; env_passthrough value
                     length cap; all magic numbers promoted to named constexprs.)
d0f7965  Tools: get_current_dir builtin + JSON-manifest external tools
                    (Two-part feature. (1) get_current_dir: zero-param builtin
                     returning getcwd() at call time; CLIs chdir() into --sandbox
                     at startup so what the tool reports == bash/fs_*'s effective
                     dir. Toolbelt registers it whenever any fs-tool is enabled.
                     (2) load_external_tools_from_json: operator declares
                     name/description/abs-command/argv-template/parameters/timeout
                     /max_output/cwd/env_passthrough/stderr in a JSON manifest;
                     each entry compiles to a regular Tool. Fork+execve, never a
                     shell; absolute path validated at load (no PATH-hijack);
                     whole-element placeholders only ("{x}", never "--flag={x}");
                     hard caps on manifest size / tools / params / argv / env;
                     env passthrough opt-in allowlist; closed stdin; setpgid +
                     SIGTERM-then-SIGKILL kills grandchildren. New --tools-json
                     PATH flag in easyai-local / easyai-cli / easyai-server.
                     manual.md §3.3.3 + §3.3.4. examples/tools.example.json.)

2026-04-29 (earlier) — robustness pass on server + CLI streaming.

84009a0  Cancel: stop server job when the client connection drops
a3038fa  CLI: tolerate non-UTF-8 bytes in tool output and model content
2af3fe8  Docs: add AI_TOOLS.md — a vendor-neutral book on AI tool calling
6939e4c  CLI: stop <think> tags from bleeding into streamed content

2026-04-28 — webui polish + fs/bash gating split.

c10c812  Webui: hide the floating top-right Settings gear button
4b3f9fa  Server: --allow-fs gate for fs_*; webui thinking lock + tool URL log
36c5f10  Webui: remove HTML-like markup from AI-brain.svg <style> comment
cedad05  Webui: restore original AI-brain favicon, add theme-aware brand fill
f9ea948  Webui: narrow SVG theme rule to hardcoded near-black colors only

2026-04-27 (afternoon) — Deep + lib-first refactor (phase 5/6).

2dd5e79  Deep — name + persona for the default easyai-server assistant
866abde  Phase 6 — easyai::Agent, the friendly Tier-1 front door (3-line hello)
1280aac  Phase 5c — Backend abstraction (LocalBackend + RemoteBackend) in lib
22ece56  Phase 5d — easyai::cli management subcommands (print_models, …)
3b66832  Phase 5b — easyai::ui::Streaming wires the canonical agent UX
344ff4b  Phase 5a — extract small reusable utilities into the lib
                    (ThinkStripper, trim_for_log, render_plan, print_presets,
                     print_tool_row, slurp_file w/ max_bytes, client_has_tool)
8314eb6  Client::retry_on_incomplete now defaults ON in libeasyai-cli
dec031f  client: fix json::type_error 302 in retry-on-incomplete nudge

2026-04-27 (morning)  — sandbox virtualisation + bash tool + library refactor.

8b60868  SESSION_NOTES update for the morning's work
25bf165  Docs refresh: Toolbelt, bash tool, --sandbox/--allow-bash gating
2572bc0  Static analysis pass: SSE buffer cap + grep regex DoS guard
bc28474  easyai::cli — Toolbelt builder + log file helper + sandbox validator
2fa381a  Extract CLI utilities into the public lib (easyai::ui / text / log)
21590b9  max_tool_hops: configurable; --allow-bash bumps to 99999
b8eda29  easyai-cli: same fs_*/bash gating as cli-remote and server
8e957af  Auto-retry-with-nudge for incomplete turns; default ON in cli-remote
0a93741  bash tool + tighten fs_*/bash gating in server and cli-remote
430f9b0  fs tools: virtual /-rooted view; sandbox base hidden from the model
56d44c3  Sandbox::resolve: total containment via path-component filtering
443468a  cli-remote: drop per-piece [content N, +Tms] / [reason N, +Tms] traces
fb8becb  cli-remote: newline on reasoning↔content transition
afe7a57  Server: incomplete-detection two-tier threshold (80 B / 350 B post-tool)
e3cf422  libeasyai-cli + cli-remote: RAW transaction log file for offline analysis

v0.1.0 (tag) — first formal release, 2026-04-26.

dc74102  easyai-cli phase 3 + full sampling control on Client
8e6c4e4  libeasyai-cli phase 2: full Client implementation (HTTP/SSE + agentic)
5bf32c0  Lib-ise easyai + scaffold libeasyai-cli (phase 1 of 3)
8a1ca33  Webui: live ctx during stream, override favicon, tools badge, lock UX
46903e3  Engine: recover tool_calls from markdown markers when model abandons XML
d7f638e  Webui ctx counter: cumulative session usage + near-limit input lock
b2f77ff  Webui: move AI-brain.svg src/->webui/, reopen think panel, diag console
870be9b  Webui: fix SyntaxError that killed tone-badge IIFE + bundle SSE noise
c2eb22c  Webui: chip always visible, tone badge unstuck, smaller response body
9b31a38  Stop reasoning + Hermes tool_call XML bleeding into chat body
e9297fe  Brand to EasyAi + embed brain SVG favicon + harden bar/tone visibility
f8d83d1  Stack-overflow audit chapter in design.md
3dec718  Fix SIGSEGV stack-overflow from std::regex in strip_html + silent SSE
7976dbe  Webui follow-ups + Bug B fallback now reaches SSE stream
731d441  Webui: blue brain panel, hide bundle Reasoning, live metrics + pulsing pill
2ff181e  Framework polish: args defaults, recipes example, docs, installer
5f7fa7d  Bug B fix + streaming UI normalisation
186c20a  Update SESSION_NOTES.md with reasoning-format fix and Bug A/B status.
709684b  THE missing piece: set reasoning_format on common_chat_templates_inputs
3a94bd3  Fix Jinja crash: push user msg before chat_params render in streaming
882abed  Add SESSION_NOTES.md
e03705e  installer: --upgrade now actually does git pull, not just git fetch
15e6056  Streaming pipeline: incremental common_chat_parse + diff (llama-server parity)
204e376  Real per-message thinking panel via DOM injection
25d43de  Route thinking to delta.reasoning_content
25e8095  --verbose flag + auto-prepend <think> for Qwen3-thinking
f1282e4  UTF-8 tolerant SSE dumps + count any non-CPU device as backend
89bab48  installer: default --ngl to -1 (auto-fit) instead of 99
8649ddd  Linux installer: ship libs into /usr/lib/easyai + LD_LIBRARY_PATH
0fcaad4  Linux installer: ship every llama.cpp .so + force --ngl based on what built
b660bbf  installer: parity pass with install_llama_server.sh on systemd unit + sysctl
1e9542b  installer: set RADV_PERFTEST=gpl + HOME/XDG_CACHE_HOME
0fa4c0a  Lift the input form to make room for the metrics bar below it
4b727d1  Move bar below textarea + chip alignment fix
c6a09d6  Single combined bar above the textarea (tone + ctx + last)
```

`git log --oneline | head -30` for the rest.

### Recent rabbit-hole notes

- **Sandbox containment refactored to total-anchor (2026-04-27)**: the
  original `Sandbox::resolve` rejected any path whose canonical form
  escaped the root.  Models would research, then emit `write_file
  path: "/2026-04-27_news.md"` and the rejection killed the whole
  deliverable.  Two iterations:
  1. `56d44c3` — iterate the input's path components, drop empty
     fragments, separators (`/`, `\`), and pure-dot components
     (`.`, `..`, `...`, …), join survivors onto sandbox root.
     Trade-off: a malicious symlink already inside the sandbox
     could now follow itself out, since we no longer call
     `weakly_canonical`.  Acceptable — the sandbox is user-owned
     and we don't expose symlink creation to the model.
  2. `430f9b0` — in addition, render the sandbox path as `/`-rooted
     in tool descriptions and result/error messages.  The model
     thinks the world starts at `/`, never sees the real prefix.
     Stops the model from inventing `/home/user/`, `root/`, etc as
     phantom subdirs (those choices were prompted by ambiguous
     descriptions saying "default = root").

- **Auto-retry with corrective nudge for incomplete turns
  (2026-04-27, `8e957af`)**: the previous `--retry-on-incomplete`
  did a blind retry on the same prompt, which usually reproduced
  the same "Let me fetch a few more sources." stop-without-tool_call
  bailout.  Now the retry appends a synthetic user message:
  *"Your previous reply only announced an action without emitting
  any tool_call. Do NOT say 'let me…' / 'I'll…' unless the tool_call
  follows in the SAME turn. Either call the next tool you actually
  need, or give the user the final answer."*  Default ON in
  cli-remote; one retry max (no spirals).  Reproduces across
  multiple models including Gemini, so it's an infra-level fix.

- **bash tool design choices (2026-04-27, `0a93741`)**: `/bin/sh -c`
  for full shell features (pipes, redirects, &&); cwd pinned to
  --sandbox; output capped at 32 KB; SIGTERM at deadline, SIGKILL
  +2 s grace; per-command timeout default 30 s, max 300 s.  Honest
  in its own description: NOT a hardened sandbox — it's a normal
  shell process with the caller's user privileges.  Opt-in only via
  `--allow-bash`.  Bumps `max_tool_hops` to 99999 because bash flows
  span many turns (compile → run → fix → re-run).

- **Static analysis findings (2026-04-27, `2572bc0`)**: SSE pending
  buffer was unbounded — a malformed stream that never emits a
  terminator would OOM the client.  Capped at 16 MiB with a clean
  abort.  Separately, `fs_grep` ran user regexes against
  multi-megabyte single lines (binary blobs, minified JS) — libstdc++'s
  recursive regex blows up on `(a+)+` style patterns.  Skip lines
  >64 KiB.  Audit also cleared format-string vulns (no
  `printf(user_string)`), URL allowlist (http/https only), HTTP body
  caps (2/4 MB), JSON parse exception handling.

- **Why thinking was rendering inline as message body** (root cause found
  in `709684b`): `Engine::Impl::render()` left `common_chat_templates_inputs::
  reasoning_format` defaulted to `NONE`.  Every PEG parser builder under
  `common/chat.cpp` does `auto extract_reasoning = inputs.reasoning_format
  != COMMON_REASONING_FORMAT_NONE;` — with NONE, the parser leaves
  `<think>...</think>` inside `msg.content`, our diff dumps it as
  `delta.content`, and the webui paints it as the regular reply.  Fixed
  by setting `in.reasoning_format = COMMON_REASONING_FORMAT_AUTO`.
  AS OF 2026-04-26 this fix is unverified on the user's AI box.

- **Why streaming was crashing with "No user query found in messages"**
  (fixed in `3a94bd3`): Qwen3 templates raise a Jinja exception if the
  history doesn't contain a user message at template-apply time.  We were
  calling `Engine::chat_params_for_current_state(true)` BEFORE pushing the
  user message into history.  Fix: push first, then render.  Required
  splitting `Engine::chat()` into `chat()` (does push + chat_continue) and
  the new `chat_continue()` (assumes user is already last in history) so
  the streaming path can render in between.

- **Engine::chat_continue()** (added in `3a94bd3`) is what the streaming
  HTTP path now calls in agentic mode.  Same multi-hop loop as `chat()`
  minus the initial `push_message("user", …)`.

- **Bug B root cause and fix** (this session, 2026-04-26):
  Qwen3.6-35B-A3 fine-tune `eng_v5` sometimes terminated the turn after
  `</think>` without emitting either content or a tool_call — model's
  last reasoning chunk was literally *"(Let's execute the tools now)"*.
  Curl trace showed 0 content + 0 tool_calls + finish_reason=stop.
  Adding `--verbose` confirmed parser was NOT engulfing — model truly
  emitted EOS.  Fix: in `chat_continue()`, detect
  `tool_calls.empty() && content.empty() && reasoning_content non-empty`,
  discard the empty turn, clear KV, fire `on_hop_reset` (new callback
  that streaming layer hooks to drop accumulated/prev_msg), and loop.
  Budget = 2 retries.  Final fallback promotes reasoning → content so
  the user never sees an empty bubble.  Validated via curl: now lands
  reliably on a real tool_call (web_search) + content reply (~970 chars).

- **`compute_diffs` strict-monotonic exception** discovered while
  testing the retry path: sometimes the partial parser temporarily
  assembles a tool_call and then "unassembles" it as more tokens
  arrive; `common_chat_msg_diff::compute_diffs` throws
  *"Invalid diff: now finding less tool calls!"*.  Fix: try/catch
  around the diff in the streaming `on_token` lambda — hold prev_msg
  on last good state and wait for next token to settle.

## 6. Known issues / pending validation

- **HTTPS for `easyai::Client`** — not wired yet.  Needs cpp-httplib's
  `CPPHTTPLIB_OPENSSL_SUPPORT` define + linking OpenSSL into the
  `easyai_cli` target + a `make_http()` branch returning an SSLClient
  base pointer (httplib's SSLClient inherits from `httplib::Client`).
  Until this lands the `Client` returns a clear error on `https://`
  endpoints.  Workaround: terminate TLS at a reverse proxy.

- **(A) Webui thinking-panel rendering** — ✅ **FIX VALIDATED**.
  Commit `709684b` made the per-message reasoning panel light up
  correctly.  Curl confirmed (812 / 1165 / 2027 reasoning_content
  deltas across runs vs 0 content deltas pre-fix; bundle now paints
  its own native panel).  This session also disabled our custom
  `__easyai-thinking` panel (dormant behind `window.__easyaiCustomThink`)
  so we don't get two glued panels — only the bundle's "Reasoning"
  panel renders, font shrunk to 0.72rem mono, default open during
  streaming, auto-collapse on finish_reason.

- **(B) Qwen3-thinking model stops after `</think>`** —
  ✅ **FIXED via thought-only retry path** (see "Bug B root cause" above).
  Validated via curl in this session.  Pending user webui validation.

- **RAG (`--RAG DIR`)** — agent's persistent registry / long-term memory.
  Five tools (rag_save / rag_search / rag_load / rag_list / rag_delete)
  share a directory of `.md` files with `keywords: a,b,c` headers and
  free-form bodies. Server enables by default
  (`/var/lib/easyai/rag`, systemd unit always passes `--RAG`); CLI
  variants opt in. End-to-end smoke test passed (round-trip save →
  list → search → load → delete). Authoritative doc: `RAG.md`.
  Operator guide: `LINUX_SERVER.md`.

- **External tools (`--external-tools DIR`)** — landed in `d0f7965`,
  hardened in `e966cf1`, evolved to dir loader + sanity warnings in
  the 2026-04-30 commit. Operators drop `EASYAI-<name>.tools` files
  in a directory; library scans the directory at startup (per-file
  fault isolation: a bad file is logged + skipped, others still
  load; empty dir is silent). Spawn path is `fork`+`execve` (never
  a shell). Security guarantees enforced at load time (absolute
  command path, no embedded placeholders, hard caps on
  manifest/argv/params/env/output/timeout, regular-file check), at
  call time (JSON-Schema arg validation, NaN/Inf rejection,
  `kMaxFdScan` close-loop, `PR_SET_PDEATHSIG`, process-group SIGTERM
  → SIGKILL, env-value cap), and at deploy time (manifest is the
  operator's trusted artefact, treat like sudoers). Sanity-check
  warnings at load: shell wrappers, `LD_*` passthrough,
  world-writable binaries / manifests. Authoritative doc:
  `EXTERNAL_TOOLS.md`. Also: paired `get_current_dir` builtin so
  the model can find out the absolute path it's running in
  (Toolbelt auto-registers it when any fs-tool is enabled).

- **Webui core dump under heavy load** — user reported coredump while
  using webui.  Did NOT reproduce in this session.  Tooling now in
  place to capture next occurrence:
    - `systemd-coredump` package installed (also baked into installer's
      apt deps so future installs get it)
    - `LimitCORE=infinity` in service drop-in (also baked into installer's
      unit template so future installs get it)
    - `--verbose` available via `--enable-verbose` installer flag (default OFF)
  Next time it crashes, run `coredumpctl list easyai-server.service`
  then `coredumpctl gdb <PID>` to get a stack.

- **llama-server parity reference**: when the user reports that
  `llama-server` does X correctly and we don't, look at `llama.cpp/tools/
  server/server-{task,chat,context}.cpp` for their stream pipeline.
  We've already mirrored:
      - common_chat_parse(text, is_partial=true, params) per token
      - common_chat_msg_diff::compute_diffs(prev, new)
      - server_chat_msg_diff_to_json_oaicompat-style delta envelope
        (.reasoning_content + .content + tool_calls)
  Differences that may still matter: their reasoning_format passed at
  task params construction time; their handling of `multi_step_tool`
  flag in chat templates; their default sampling params.

## 7. Common pitfalls / debugging tips

- After any `examples/server.cpp` change that affects the served HTML/JS
  injection, the user MUST do **Cmd+Shift+R** in the browser to bypass cache.
- The bundle.js is 6.2 MB; binary size jumps from ~1.5 MB to ~8.3 MB.
- `--ngl 99` poisons `common_fit_params` (refuses to lower a user-pinned
  value).  Always default `--ngl -1` (auto-fit) so the engine picks how
  many layers fit.
- Vulkan/RADV often reports devices as `ACCEL` not `GPU` — the engine's
  `backend_summary` accepts any non-CPU type now.
- `<think>` content can split across two model tokens — the streaming pipe
  uses `common_chat_parse(is_partial=true)` which handles this; don't
  reintroduce regex-based tag matching.
- The bundle hashes class names — never rely on `[class*="something"]`
  selectors.  Use `aria-label`, `data-testid`, or text-content matching.
- Shadow DOM mounts (`__easyaiBarHost`) survive Svelte body re-renders
  because they're attached to `<html>`, not `<body>`.

## 8. How the user typically iterates

```
# On Mac M3 (development):
cd ~/develop/easyai/easyai
cmake --build build -j
./build/easyai-server -m models/qwen2.5-3b-instruct-q4_k_m.gguf --webui-title "AI Box"
# user opens http://127.0.0.1:8080, Cmd+Shift+R, tests

# On Linux AI box (production):
cd ~/easy
git pull
sudo systemctl stop easyai-server
./scripts/install_easyai_server.sh --upgrade --enable-now
sudo journalctl -u easyai-server -f
```

## 9. Communication style

- Reply in Brazilian Portuguese.
- Keep replies tight; the user is technically fluent and prefers concrete
  diffs / commands over long explanations.
- Show what changed, why, and the exact command to reproduce on the AI box.
- When you commit, push immediately — the user often pulls right after.
- Use task tracking via TaskCreate/TaskUpdate when work spans multiple
  steps.

---

*End of session notes.  Update this file as work progresses so future
sessions resume cleanly.*
