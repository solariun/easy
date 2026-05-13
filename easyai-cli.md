# easyai-cli — the OpenAI-compatible chat client

> **A drop-in client for any OpenAI-compatible chat endpoint** —
> easyai-server, llama-server, vLLM, OpenAI itself, anything that speaks
> `/v1/chat/completions`. Renders responses with reasoning streams,
> registers tools client-side, dispatches their handlers in-process, and
> pushes the results back. Single binary, REPL or one-shot, no model
> loaded — pure protocol.

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Connection — endpoint, model, auth](#2-connection--endpoint-model-auth)
3. [Modes — REPL, one-shot, piped, management](#3-modes--repl-one-shot-piped-management)
4. [Command-line flags](#4-command-line-flags)
5. [Tool registration](#5-tool-registration)
6. [System prompt + injected blocks](#6-system-prompt--injected-blocks)
7. [Sampling and penalty knobs](#7-sampling-and-penalty-knobs)
8. [Reasoning streams](#8-reasoning-streams)
9. [The raw transaction log](#9-the-raw-transaction-log)
10. [Session persistence](#10-session-persistence)
11. [RAG — persistent memory](#11-rag--persistent-memory)
12. [External tools](#12-external-tools)
13. [Management subcommands](#13-management-subcommands)
14. [Worked examples](#14-worked-examples)
15. [Cross-references](#15-cross-references)

---

## 1. Quick start

```bash
# 1) Point it at any OpenAI-compatible endpoint.
easyai-cli --url http://ai.local:8080 -p "what time is it?"

# 2) REPL — drop the prompt, type interactively.
easyai-cli --url http://ai.local:8080

# 3) Coding agent — sandbox + bash + plan, all auto-wired.
easyai-cli --url http://ai.local:8080 \
           --allow-bash --sandbox ~/projects/foo \
           "implement a tetris in C++ with SOLID design"

# 4) Pipe a prompt in.
echo "summarise this" | easyai-cli --url http://ai.local:8080
```

Connection details are remembered via env vars so the per-command line
stays short:

```bash
export EASYAI_URL=http://ai.local:8080
export EASYAI_API_KEY=...   # if the server is auth-on
easyai-cli "what's new on hacker news today?"
```

---

## 2. Connection — endpoint, model, auth

The transport layer is plain HTTP(S) `POST /v1/chat/completions`. The
client streams the SSE response, parses `delta.{content,reasoning,tool_calls}`,
dispatches any tool calls in-process, and posts the next turn.

| Flag | Env var | Default | Notes |
| --- | --- | --- | --- |
| `--url <URL>` | `EASYAI_URL` | (none — required) | Base URL of the server. `/v1/chat/completions` is appended automatically. https:// works if the binary was built with OpenSSL. |
| `--api-key <KEY>` | `EASYAI_API_KEY` | (empty) | Bearer token sent as `Authorization: Bearer <KEY>` on every request. |
| `--model <NAME>` | `EASYAI_MODEL` | `EasyAi` | The `model` field of the request body. easyai-server returns whatever it has loaded under any name; other servers may match strictly. |
| `--timeout <SEC>` | `EASYAI_TIMEOUT` | `1800` (30 min) | Read/write timeout on the streaming connection. Bumped from the usual 60 s to accommodate long thinking turns. |
| `--http-retries <N>` | `EASYAI_HTTP_RETRIES` | `5` | Extra attempts on transient HTTP failures (connect refused, read timeout, 5xx). 4xx never retries. Each retry logs to stderr. 0 disables. |
| `--insecure-tls` | — | off | Skip peer cert verification (https only). Dev / self-signed only. |
| `--ca-cert <PATH>` | — | (system) | Trust the PEM bundle at `<PATH>` for https. |

If `--url` is omitted and `EASYAI_URL` is unset, the binary errors out
at startup with a usage hint.

**Connection lifecycle (since 2026-05-08):** the cli holds a single
persistent `httplib::Client` for the entire session — every agentic
hop (chat completion + tool dispatch + chat completion + …) reuses
the same TCP connection thanks to HTTP keep-alive. This was a real
bug before that date: the cli rebuilt the Client per request, so
each hop opened a fresh connection that piled up in `TIME_WAIT` for
~60 s on the client. A 50-tool-call session opened 50 sockets and
on long sessions exhausted the ephemeral-port range, surfacing as
`Connection timed out` retry storms. The fix is purely on the cli
side and transparent to anything connecting to easyai-cli's
upstream. To confirm keep-alive is working in production, point the
cli at an easyai-server with `[SERVER] verbose = on` and watch the
`http: in_flight=...` field of the periodic METRICS line plus the
per-request `→` / `←` log: a healthy session shows steady
`reqs=N` increments with `in_flight=0..1` between hops, and the
system-wide `tcp: time_wait` count stays low. Before the fix, every
hop bumped `tcp: time_wait` and eventually drove the
`TIME_WAIT N/M ephemeral ports (X.X% …)` indicator into the
elevated / HIGH / CRITICAL bands.

---

## 3. Modes — REPL, one-shot, piped, management

The same binary covers four operating modes; they're selected by what's
on the command line and stdin.

| Mode | Trigger | Behaviour |
| --- | --- | --- |
| **REPL** | No `-p`, no positional prompt, stdin is a TTY | Interactive prompt loop. `Ctrl-D` to exit. History persists during the session. `Ctrl-C` during a turn → graceful exit (see below). |
| **One-shot** | `-p <text>` OR a positional argument | Send the single prompt, stream the reply, exit. |
| **Piped** | stdin is a pipe (anything redirected in) | Reads stdin into the prompt and runs once. Same as one-shot. |
| **Management** | `--list-models`, `--list-tools`, `--list-remote-tools`, `--health`, `--props`, `--metrics`, `--set-preset`, `--show-system-prompt` | Hits the named endpoint (or, for `--show-system-prompt`, just resolves locally), prints the result, exits. No chat. See [§13](#13-management-subcommands). |

The four are mutually exclusive: passing `-p` AND a management flag is
an error.

### Ctrl-C and SIGTERM

Two signal-handling modes — `--quiet` switches between them.

| Mode | First Ctrl-C / SIGTERM | Second Ctrl-C |
| --- | --- | --- |
| **interactive** (default) | Mid-turn: prints `<exiting: waiting for the ai session to be finished. Ctrl-C again to force.>` and lets the in-flight chat finish naturally. The conversation isn't truncated mid-stream; the program exits cleanly (rc=0) once the turn ends. At a REPL prompt (no chat in flight): exits immediately, same as `Ctrl-D`. | Hard cancel (rc=130) — the server's decode loop is told to stop, the SSE stream aborts. Use this when the model is genuinely stuck. |
| **`--quiet`** | Hard cancel immediately (rc=130). This is the expected behavior for `kill <pid>` in a script — no graceful waiting, no extra stderr noise. | Same — already cancelled. |

The first-Ctrl-C-is-graceful behavior in interactive mode is there
because for a long thinking turn the user usually wants to *finish
this answer and stop*, not to truncate it. The second-Ctrl-C escape
hatch covers the "model got stuck" case.

---

## 4. Command-line flags

Full reference, grouped the way `--help` shows them. Env-var fallbacks
appear next to the matching flag.

### Connection

| Flag | Env | Notes |
| --- | --- | --- |
| `--url URL` | `EASYAI_URL` | Required (or set via env). |
| `--api-key KEY` | `EASYAI_API_KEY` | Bearer auth. |
| `--model NAME` | `EASYAI_MODEL` | Default `EasyAi`. |
| `--timeout SEC` | `EASYAI_TIMEOUT` | Default 1800. |
| `--http-retries N` | `EASYAI_HTTP_RETRIES` | Default 5. |
| `--insecure-tls` | — | https only — DEV ONLY. |
| `--ca-cert PATH` | — | PEM bundle for custom CAs. |

### Conversation shape

| Flag | Notes |
| --- | --- |
| `--system TEXT` | Inline system prompt. |
| `--system-file PATH` | System prompt loaded from a file. Beats `--system` if both are given (but you'd usually use one). |

When neither is passed, the server's default persona handles the system
message. Either flag still gets the [environment] + [guidance] injection
prepended (see [§6](#6-system-prompt--injected-blocks)).

### Sampling and penalty (omit any to keep server default)

| Flag | Range | Notes |
| --- | --- | --- |
| `--temperature F` | typically 0–2 | OpenAI standard. |
| `--top-p F` | 0–1 | Nucleus top-p. |
| `--top-k N` | int ≥0 | Top-k cutoff. |
| `--min-p F` | 0–1 | llama.cpp / easyai min-p. |
| `--repeat-penalty F` | ≥ 0 | **Default 1.15** — anti-loop safety net for thinking models. Pass `1.0` to disable. |
| `--frequency-penalty F` | -2..2 | OpenAI standard. |
| `--presence-penalty F` | -2..2 | OpenAI standard. |
| `--seed N` | int | Deterministic sampling. |
| `--max-tokens N` | int | Cap reply length. |
| `--stop SEQ` | repeatable | Add a stop string. |
| `--extra-json '{...}'` | JSON | Free-form object merged into the request body — escape hatch for server-specific fields. |

### Tools

| Flag | Notes |
| --- | --- |
| `--tools LIST` | Comma list, overrides the default catalog. See [§5](#5-tool-registration) for valid names. |
| `--sandbox DIR` | Working root for `fs` / `bash` / `python3`. **Auto-registers the unified `fs` tool** (action=read / write / list / glob / grep / check_path / cwd / sandbox). `bash` and `python3` still require their respective `--allow-*` flags. |
| `--allow-bash` | Register `bash`. **Implies `fs`** (bash subsumes it). cwd = `--sandbox` if given, else the binary's CWD. WARNING: not a hardened sandbox. |
| `--no-python` | Drop the auto-registered `python3` tool. By default `python3` is **ON** whenever `--sandbox` or `--allow-bash` is set. Stdlib-only interpreter (no PYTHON* env, no site-packages, no cwd on `sys.path`); disk access auto-restricted to the sandbox root via a Python preamble. WARNING: defense-in-depth, not a hardened sandbox — `import os` / `import socket` / `import subprocess` still work. |
| `--use-google` | Enable `engine="google"` inside the unified `web` tool (Google Custom Search JSON API). Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` env vars. |
| `--RAG DIR` | Enable RAG persistent memory rooted at DIR. Registers ONE `rag(action=...)` tool. |
| `--external-tools DIR` | Load every `EASYAI-*.tools` manifest in DIR. See `EXTERNAL_TOOLS.md`. |
| `--no-plan` | Don't auto-register the `plan` tool. |

### Behaviour

| Flag | Notes |
| --- | --- |
| `-p TEXT`, `--prompt TEXT` | One-shot prompt. (You can also pass it as a positional arg or pipe via stdin.) |
| `--no-reasoning`, `--hide-reasoning` | Hide `delta.reasoning_content` (default: shown inline in dim grey). |
| `--max-reasoning N` | Abort the SSE stream when this turn's reasoning exceeds N chars. 0 = unlimited (default). Useful for thinking models that fall into long deliberation loops. |
| `--no-retry-on-incomplete` | Disable the auto-retry-with-nudge for incomplete turns (default: ON). |
| `--retry-on-incomplete` | Legacy alias for the now-default behaviour. No-op. |
| `--verbose`, `-v` | Log HTTP+SSE diagnostics to stderr (timestamps + per-piece traces). Stderr-only — does NOT create a /tmp log file (use `--log-file` for that). |
| `-q`, `--quiet` | Disable the spinner glyph + context-fill gauge. Use for batch / scripted runs. **Also changes `Ctrl-C` / `SIGTERM` semantics** — see [§3 → Ctrl-C and SIGTERM](#ctrl-c-and-sigterm). |
| `--log-file PATH` | Opt in to a raw transaction log at PATH (request body + every SSE chunk + every tool dispatch input/output, mode 0600). Default OFF — no log file is written without this flag. Implies `--verbose`. |
| `--continue` | Load `.easyai_session` from cwd before the first prompt. **Default OFF** (since 2026-05-13) — any existing session file is ignored and overwritten on the first turn unless this flag is set. INI: `[cli] auto_continue = true\|false`. See [§10](#10-session-persistence). |
| `--no-continue` | Explicit form of the default — ignore any existing `.easyai_session` and overwrite on the first turn. Useful to override `[cli] auto_continue = on` set in INI. |
| `--compress` | After loading, ask the model for one lossless recap of the conversation and replace the history with that recap. Also reachable mid-REPL via `/compress`. No-op without `--continue` (nothing in memory to recap). INI: `[cli] auto_compress = true\|false`. |

### Management subcommands (one only, no chat)

See [§13](#13-management-subcommands) for the full picture.

| Flag | Result |
| --- | --- |
| `--list-tools` | Local tools (registered in this CLI), with full descriptions. |
| `--list-remote-tools` | `GET /v1/tools` — server-side tools (easyai-server extension). |
| `--list-models` | `GET /v1/models`. |
| `--health` | `GET /health`. |
| `--props` | `GET /props`. |
| `--metrics` | `GET /metrics` (Prometheus text). |
| `--set-preset NAME` | `POST /v1/preset {preset:NAME}`. |
| `--show-system-prompt` | Print the **resolved** system prompt (built-in `[environment]` + `[guidance]` injection PLUS `--system` / `--system-file` content) and exit. Does NOT contact the server — useful for confirming what the model would see, including without a working `--url`. |

### Misc

| Flag | Notes |
| --- | --- |
| `-h`, `--help` | Print the full help and exit. |

---

## 5. Tool registration

The CLI registers tools **client-side**: their handlers run in the
binary's own process, not on the server. The server is told what tools
exist (their names + JSON schemas) and asks for them when needed; the
client dispatches and posts the result back as a tool message.

### Default catalog

When `--tools` is **not** given, the CLI auto-registers:

```
datetime, plan, web,
system_meminfo, system_loadavg, system_cpu_usage, system_swaps
```

…plus, conditionally:

| Trigger | Adds |
| --- | --- |
| `--sandbox DIR` **OR** `--allow-bash` | The unified `fs` tool AND `python3` (the latter unless `--no-python`) |
| `--allow-bash` | `bash` (and bumps the agentic loop's `max_tool_hops` to 99999) |
| `--no-python` | drops the auto-on `python3` tool (otherwise on whenever fs is on) |
| `--use-google` (+ env vars set) | Enables `engine="google"` inside the unified `web` tool |
| `--RAG DIR` | `rag` (single-tool dispatcher; sub-actions save / append / search / load / list / delete / keywords) |
| `--external-tools DIR` | every tool from each loaded `EASYAI-*.tools` manifest |

### Why `--sandbox` and `--allow-bash` both register `fs`

Bash is strictly more permissive than the unified `fs` tool — if the
operator trusts the model with bash, they trust it with
`fs(action="read")` etc. by construction. Requiring an extra
`--allow-fs` flag for the narrower surface produced sessions where
the model had bash but no `fs` and fell back to `cat > file` /
`cat <<EOF` / `sed -i` for ordinary file work. The new defaults
eliminate that trap: any flag that says "the model can touch files"
registers `fs` automatically.

`fs(action="sandbox")` is one of the unified tool's sub-actions, so
the model can always resolve the real on-disk path of where its work
is landing — distinct from `fs(action="cwd")`, which reports the live
process cwd and can drift.

### Restricting the catalog with `--tools`

Pass `--tools LIST` to override the auto-catalog. Valid names:

```
datetime, plan, web, fs, bash,
system_meminfo, system_loadavg, system_cpu_usage, system_swaps,
rag
```

`bash` / `rag` still require their respective opt-in flags even when
explicitly listed; `engine="google"` inside `web` likewise depends on
`--use-google` plus the env vars.

### Inspecting what got registered

```bash
easyai-cli --url ... --sandbox ~/foo --allow-bash --list-tools
```

Prints every registered tool's name + full description, in the same
order they're sent to the server. Useful for debugging "why didn't the
model use my tool?".

See [`AI_TOOLS.md`](AI_TOOLS.md) for the deep dive on what a tool is, and
[`manual.md`](manual.md) §3.2 / §3.2.1 for how to author your own.

---

## 6. System prompt + injected blocks

When the agent has any create/mutate affordance (fs_* / bash / plan),
the CLI prepends two small in-binary blocks to the user's system prompt:

```
[environment]
sandbox root: /Users/.../projects/foo
fs_* tools' virtual `/` maps here; bash runs with this as its cwd.

[guidance]
When asked to create something, pick one viable implementation and
carry it through to a working end state. Do not enumerate options,
branch on hypotheticals, or stop at a draft. Choose, build, verify
it runs, then report. The user can ask for refinements after they
see it working.
```

Why:

* **`[environment]`** — without it, the first move of any coding agent
  is "where am I?" (`fs(action="cwd")` / `pwd`). Injecting the resolved
  absolute path saves that hop on every task.
* **`[guidance]`** — smaller models otherwise enumerate options, ask
  permission for every choice, or stop at a draft. The assertive
  framing shifts them toward a working result.

The user's `--system` / `--system-file` content (if any) appears
**after** these blocks, so user intent has the last word.

When the agent has no fs/bash/plan tool — pure chat with web search and
nothing else — neither block is injected.

---

## 7. Sampling and penalty knobs

All knobs are server-side parameters; the CLI just forwards what you
pass. Omitting any leaves the server's default in place.

The penalties (`--repeat-penalty`, `--frequency-penalty`,
`--presence-penalty`) all bias generation *against* tokens that have
already been produced — but they bite differently:

| Flag | Form | Bites on |
| --- | --- | --- |
| `--repeat-penalty F` | multiplicative on recent logits | tight literal repetition ("I'll write X / Let me write X / OK, creating X") |
| `--frequency-penalty F` | additive, scales with token count | over-use of common tokens ("the the the") |
| `--presence-penalty F` | additive, fixed cost per token-already-seen | topic stickiness without per-occurrence ramp-up |

`--repeat-penalty 1.15` (the CLI's only non-obvious default) is the
anti-loop safety net.  Pass `1.0` to disable when you *want* the model
to repeat itself — for example when calling the same tool many times
in an agentic flow and you don't want the model paraphrasing tool
names after the third call.

`--presence-penalty F` (OpenAI standard, range `[-2.0, 2.0]`,
default `0.0`) is the gentler companion.  Reach for it when:

* You're running long agentic flows where `repeat_penalty=1.15`
  starts making the model invent tool-name synonyms.
* The model has correct content but keeps rehearsing the same
  topic instead of moving on.
* You want "introduce new vocabulary" pressure without the
  per-occurrence cost ramp of `repeat_penalty`.

Typical pairings:

| Workload | `repeat_penalty` | `presence_penalty` |
|---|---|---|
| Short chat / single-tool turns | `1.15` (default) | `0.0` |
| Long agentic flows (10+ hops) | `1.0` (off) | `1.0` to `1.5` |
| Brainstorm / creative writing | `1.15` | `0.6` to `1.0` |
| Code generation, structured output | `1.15` | `0.0` |

See [`design.md` §4b](design.md#4b-sampling-and-the-penalty-stack)
for the full rationale on why the two penalties exist and when to
pick which.

`--extra-json` is the escape hatch for fields the CLI doesn't know
about. Whatever JSON object you pass is merged shallowly into the
request body before send, so server-specific extensions (vendor sampling
modes, custom routing hints) work without recompiling.

---

## 8. Reasoning streams

For models that emit `reasoning_content` (Qwen-thinking, GPT-o1-class,
Claude 4.x extended thinking), the CLI prints the reasoning stream
inline in dim grey, separate from the visible content. This is on by
default. `--no-reasoning` (or `--hide-reasoning`) suppresses it.

`--max-reasoning N` is a defensive cap: if the accumulated
`reasoning_content` for a single turn exceeds N characters, the SSE
stream is aborted and the turn is treated as incomplete (which then
triggers the auto-retry-with-nudge unless that's disabled). Default 0
(unlimited). Useful when a thinking model falls into a deliberation
loop on a niche question.

Incomplete-turn handling: when the server flags a turn as
`timings.incomplete=true` (model produced no tool_call AND only a tiny
reply, e.g. "I'll search…"), the CLI by default drops that turn,
appends a corrective user nudge, and re-issues ONCE.
`--no-retry-on-incomplete` opts out — useful when you want to see the
raw incomplete signal for debugging.

---

## 9. The raw transaction log

The raw transaction log is **opt-in** via `--log-file PATH`. Without
that flag, no log file is created — neither the binary nor the
library writes to `/tmp` by default.

```bash
easyai-cli --url http://ai.local --log-file /tmp/run.log "your prompt"
```

The log at `PATH` is a verbatim record of:

* The HTTP request body (every turn — including the resolved system
  prompt with injected blocks, the full tools array, the message
  history).
* Every SSE chunk byte-for-byte.
* Every tool call dispatched: input arguments, output content,
  duration.
* Connection-level events (retries, timeouts, status codes).

Mode 0600. `--log-file` implies `--verbose` (so the file carries
CLI-side diagnostics alongside the raw wire bytes). Suitable for
replaying / diffing / grepping.

For one-off debugging without a persistent file, `--verbose` alone
streams the same diagnostics to stderr.

> **What changed (2026-05-12):** prior versions auto-opened
> `/tmp/easyai-cli-{pid}-{epoch}.log` whenever `--verbose` was set,
> AND the library-side `easyai::Client` opened a separate
> `/tmp/easyai-client-{pid}-{epoch}.log` on every construction
> unless `EASYAI_NO_AUTO_LOG=1` was in the env. Both auto-opens are
> now disabled by the cli binary so a default invocation leaves no
> artifacts behind. To restore the library auto-open behaviour, set
> `EASYAI_NO_AUTO_LOG=0` explicitly in the environment.

---

## 10. Session persistence

Every `easyai-cli` invocation writes a `.easyai_session` file in the
current working directory after each chat turn (atomic tempfile +
`rename(2)`, mode 0600, `O_NOFOLLOW`).  The file is the OpenAI-shape
message array — same format the CLI sends on the wire — so it's
plain-text greppable, diffable, and re-loadable in a future
invocation.

**Loading is default-OFF since 2026-05-13.**  Even when a
`.easyai_session` already exists in the current directory,
`easyai-cli` starts fresh silently — and overwrites the file on the
first turn.  Pass `--continue` (or set `[cli] auto_continue = on` in
INI) to resume from the existing file before the first prompt.
Saving on every turn is unchanged.

```bash
$ cd ~/project
$ easyai-cli --url http://ai.local
> fix the build error in src/main.cpp
[turn completes; .easyai_session updated]
> /exit

# Tomorrow, same project — resume requires --continue:
$ cd ~/project
$ easyai-cli --url http://ai.local --continue
[easyai-cli-remote] continued from .easyai_session in /Users/x/project
> what was the build error again?
[model has the prior context]

# Without --continue the existing file is overwritten on the first turn:
$ cd ~/project
$ easyai-cli --url http://ai.local
> hello
[turn completes; .easyai_session overwritten with this fresh history]
```

Four control points:

| Surface | What it does |
| --- | --- |
| (no flag) | **Default**: ignore any `.easyai_session` and overwrite it on the first turn.  Save on every turn. |
| `--continue` | Load the existing `.easyai_session` (if any) before the first prompt; otherwise start fresh.  Overrides `[cli] auto_continue = off`. |
| `--no-continue` | Explicit form of the default — useful to override an operator's `[cli] auto_continue = on` for this invocation. |
| `--compress` | After loading, ask the model for one lossless recap of the conversation and replace history with the recap.  No-op without `--continue` (nothing in memory to recap). |
| `/compress` (in the REPL) | Same compress flow, fired mid-session when context gets long. |

### Save cadence (force-exit survival)

The `.easyai_session` is checkpointed at three layers, in this order:

1. **After every tool dispatch** during a turn — written from the
   `on_tool` callback, so even mid-turn progress hits disk before
   the model continues reasoning.
2. **After every `chat()` return** in `run_one()` — covers graceful
   completion and stage-1 cancel (Ctrl-C once, model lets the SSE
   close).
3. **After every history-mutating slash command** (`/clear`,
   `/reset`, `/compress`).

The first layer is what makes a **force-exit** (Ctrl-C 3×, stage 3 →
`_exit(130)` from the signal handler) still leave a useful session
behind: the file on disk reflects the conversation up to the last
completed tool round-trip, only the in-flight partial reply is lost.
Stages 1 and 2 (graceful + cancel) also work because their `chat()`
returns normally and layer 2 fires.

The compress prompt instructs the model to preserve verbatim: every
file path, every decision made, every code change, every error with
its cause, every tool result still relevant, every user-stated
constraint or preference.  And to strip: pleasantries, abandoned
exploratory branches, retries of the same query.  The output replaces
history as a synthetic two-message pair
(`{user: "Previous conversation summarised below; continue from here."}
{assistant: "<recap>"}`) so the chat template sees a normal turn
shape.

History-mutating slash commands (`/clear`, `/reset`, `/compress`)
also save `.easyai_session` so a later resume picks up the
post-command state.

### INI mapping

Every session-related knob is also reachable via `[cli]` keys in
`/etc/easyai/easyai-cli.ini` (override with `--config PATH`).
Precedence: CLI flag > INI > hardcoded default.

| INI key (`[cli]`) | Default | CLI flag(s) | Effect |
| --- | --- | --- | --- |
| `auto_continue` | `false` | `--continue` / `--no-continue` | Load `.easyai_session` from cwd before the first prompt. |
| `auto_compress` | `false` | `--compress` | Run the compress flow on every load (rare; usually you want `/compress` on demand). |
| `log_file`      | `""`    | `--log-file PATH` | Raw transaction log path.  Empty = no log file. |
| `auto_log`      | `false` | (no flag) | When `true`, removes the cli's default `EASYAI_NO_AUTO_LOG=1` so the library reopens its legacy `/tmp/easyai-client-{pid}-{epoch}.log` per Client.  Keep off unless you want that postmortem trail. |
| `show_bash`     | `true`  | `--show-bash` / `--no-show-bash` | Print bash subprocess input/output to the operator's terminal in real time. |
| `show_python`   | `true`  | `--show-python` / `--no-show-python` | Same for `python3`. |

Example `easyai-cli.ini` for an "always resume, never auto-log" workstation
(flip `auto_continue` to `true` so every invocation resumes without
needing `--continue` on the command line):

```ini
[cli]
auto_continue = true
auto_compress = false
log_file      =
auto_log      = false
show_bash     = true
show_python   = true
```

Operators who don't want session files in cwd at all: leave
`auto_continue = false` (the default) so existing files are
overwritten rather than read, and `rm .easyai_session` if it leaks
past — there's no `--no-session` flag today.  The file is local to
**cwd**, not `~`, so the unit of persistence is naturally the
project directory you're working in: two projects in two different
dirs have two independent sessions.

---

## 11. RAG — persistent memory

`--RAG <dir>` mounts a directory as the agent's long-term memory. It
exposes ONE `rag` tool with seven sub-actions (`save`, `append`,
`search`, `load`, `list`, `delete`, `keywords`); each memory is a
single Markdown file in `<dir>` that the operator can hand-edit.

Memories whose title starts with `fix-easyai-` are immutable: save /
append / delete refuse them. Pass `fix=true` (sub-action `save`) to
mint one.

See [`RAG.md`](RAG.md) for the full guide.

---

## 12. External tools

`--external-tools <dir>` loads every `EASYAI-<name>.tools` JSON manifest
in `<dir>` as an operator-defined tool pack. Per-file fault isolation —
a broken manifest doesn't take down the others. Tools spawn via
`fork`+`execve`, never a shell, so a manifest is the supported way to
give the model focused powers without flipping `--allow-bash`.

See [`EXTERNAL_TOOLS.md`](EXTERNAL_TOOLS.md) for the manifest schema and
worked examples.

---

## 13. Management subcommands

Each one hits a known endpoint, prints the result, and exits. They're
mutually exclusive with chat; if you pass any of them with `-p` or a
positional prompt, the chat is dropped and only the management call
runs.

| Flag | What it does |
| --- | --- |
| `--list-tools` | Print every LOCAL tool (the catalog the CLI sends to the server in `tools[]`) with name + full description. The fastest way to confirm what the model will see. |
| `--list-remote-tools` | `GET /v1/tools`. easyai-server extension — lists tools the *server* registered (its built-ins + RAG + external + MCP-fetched). May 404 against other OpenAI-compat servers. |
| `--list-models` | `GET /v1/models`. Standard. |
| `--health` | `GET /health`. Prints `ok` / `unhealthy: <reason>`. |
| `--props` | `GET /props`. Server-side configuration dump. |
| `--metrics` | `GET /metrics`. Prometheus exposition. |
| `--set-preset NAME` | `POST /v1/preset {preset:NAME}`. Switches the server's ambient sampling preset (easyai-server extension). |
| `--show-system-prompt` | Resolve and print the system prompt the CLI would send on the next turn — built-in `[environment]` + `[guidance]` injection plus any `--system` / `--system-file` content. Does NOT contact the server, so it works without a reachable `--url`. The fastest way to verify "is the model actually seeing my persona / sandbox / guidance?". |

The connection flags (`--url`, `--api-key`, `--insecure-tls`,
`--ca-cert`) apply to management subcommands the same way they apply to
chat. `--show-system-prompt` is the one exception — it never makes a
network call and works without `--url`.

---

## 14. Worked examples

### One-shot chat

```bash
easyai-cli --url http://ai.local:8080 -p "what's the capital of Mongolia?"
```

### Coding agent (the canonical one)

```bash
easyai-cli --url http://ai.local:8080 \
           --allow-bash --sandbox ~/projects/tetris \
           "implement a tetris in C++ with SOLID design, write tests, and document"
```

What this gives the model:

* `bash` rooted at `~/projects/tetris`
* the unified `fs` tool (action=read / write / list / glob / grep /
  check_path / cwd / sandbox), all rooted there too
* `fs(action="sandbox")` returning `~/projects/tetris`
* `plan` for a visible step checklist
* `[environment]` block with the resolved absolute path
* `[guidance]` block with the assertiveness rule

### Pure chat with no shell access

```bash
easyai-cli --url http://ai.local:8080 -p "summarise transformers in 5 lines"
```

No sandbox, no `--allow-bash` → the model has only `datetime`, `plan`,
`web`, and `system_*`. No `[environment]` / `[guidance]` injection
because there's no file / shell affordance.

### Restrict to specific tools

```bash
easyai-cli --url http://ai.local:8080 --tools datetime,web \
           "find the latest CVE for libcurl"
```

`--tools` overrides the auto-catalog completely. `--allow-bash` /
`--sandbox` / `--RAG` are still respected for their specific tools but
the rest of the catalog is whatever's in the explicit list.

### Confirming the system prompt

```bash
easyai-cli --sandbox /tmp/foo --allow-bash --show-system-prompt
```

Prints exactly what the model would receive on the next turn,
including the resolved absolute path in `[environment]` and the
`[guidance]` block. Doesn't contact the server — works without
`--url`. Use this whenever you tweak `--system` / `--system-file` /
`--sandbox` / `--allow-bash` and want to confirm the result before
the chat starts.

Equivalent `--system` overlay:

```bash
easyai-cli --sandbox /tmp/foo --allow-bash \
           --system "You are a senior C++ engineer." \
           --show-system-prompt
```

Output: `[environment]` block, `[guidance]` block, blank line, then
your `You are a senior C++ engineer.` — same order the model sees them
in the next request.

### Pipe a prompt

```bash
cat README.md | easyai-cli --url http://ai.local:8080 \
                           -p "summarise this in 3 bullets"
```

Stdin overrides any positional prompt and is appended to the prompt
text.

### Use it from a script (one-shot, quiet)

```bash
ANSWER=$(easyai-cli --url $URL --quiet -p "is 17 prime? answer y/n only")
[ "$ANSWER" = "y" ] && echo "prime"
```

`--quiet` drops the spinner so stdout is clean.

### Switch server preset on the fly

```bash
easyai-cli --url $URL --set-preset deterministic
easyai-cli --url $URL --set-preset balanced
```

Affects every subsequent request to the server until changed again.
Server-side feature.

### Talk to OpenAI directly

```bash
easyai-cli --url https://api.openai.com --api-key $OPENAI_API_KEY \
           --model gpt-4o-mini -p "hi"
```

Works against any OpenAI-compat endpoint; reasoning streams pass
through cleanly for models that emit them.

---

## 15. Cross-references

- [`README.md`](README.md) — sales overview + quickstart for the whole
  project.
- [`easyai-server.md`](easyai-server.md) — the matching server: tool
  gating, MCP surface, INI config, the Deep persona.
- [`manual.md`](manual.md) — embedding `easyai::Client` in your own
  binaries, authoring tools, the agentic-loop walkthrough.
- [`design.md`](design.md) — architecture and "why" decisions.
- [`AI_TOOLS.md`](AI_TOOLS.md) — what a tool is, JSON-schema, the loop.
- [`EXTERNAL_TOOLS.md`](EXTERNAL_TOOLS.md) — operator-defined external
  tools (`EASYAI-*.tools` manifests).
- [`RAG.md`](RAG.md) — persistent registry / long-term memory.
- [`MCP.md`](MCP.md) — Model Context Protocol surface.
