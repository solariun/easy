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
10. [RAG — persistent memory](#10-rag--persistent-memory)
11. [External tools](#11-external-tools)
12. [Management subcommands](#12-management-subcommands)
13. [Worked examples](#13-worked-examples)
14. [Cross-references](#14-cross-references)

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

---

## 3. Modes — REPL, one-shot, piped, management

The same binary covers four operating modes; they're selected by what's
on the command line and stdin.

| Mode | Trigger | Behaviour |
| --- | --- | --- |
| **REPL** | No `-p`, no positional prompt, stdin is a TTY | Interactive prompt loop. `Ctrl-D` to exit. History persists during the session. `Ctrl-C` during a turn → graceful exit (see below). |
| **One-shot** | `-p <text>` OR a positional argument | Send the single prompt, stream the reply, exit. |
| **Piped** | stdin is a pipe (anything redirected in) | Reads stdin into the prompt and runs once. Same as one-shot. |
| **Management** | `--list-models`, `--list-tools`, `--list-remote-tools`, `--health`, `--props`, `--metrics`, `--set-preset`, `--show-system-prompt` | Hits the named endpoint (or, for `--show-system-prompt`, just resolves locally), prints the result, exits. No chat. See [§12](#12-management-subcommands). |

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
| `--sandbox DIR` | Working root for fs and bash. **Auto-registers `fs_*` + `get_sandbox_path`.** Bash still requires `--allow-bash`. |
| `--allow-bash` | Register `bash`. **Implies fs_*** (bash subsumes them). cwd = `--sandbox` if given, else the binary's CWD. WARNING: not a hardened sandbox. |
| `--use-google` | Register `web_google` (Google Custom Search JSON API). Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` env vars. |
| `--RAG DIR` | Enable RAG persistent memory rooted at DIR. Default registers ONE `rag(action=...)` tool. |
| `--split-rag` | Opt back into the legacy seven `rag_*` tools. |
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
| `--verbose`, `-v` | Log HTTP+SSE traffic to stderr. ALSO writes the raw transaction (request body, every SSE chunk, every tool dispatch input/output) to `/tmp/easyai-cli-{pid}-{epoch}.log`. |
| `-q`, `--quiet` | Disable the spinner glyph + context-fill gauge. Use for batch / scripted runs. **Also changes `Ctrl-C` / `SIGTERM` semantics** — see [§3 → Ctrl-C and SIGTERM](#ctrl-c-and-sigterm). |
| `--log-file PATH` | Override the auto-generated transaction log path. Implies `--verbose`. |

### Management subcommands (one only, no chat)

See [§12](#12-management-subcommands) for the full picture.

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
datetime, plan, web_search, web_fetch, get_current_dir,
system_meminfo, system_loadavg, system_cpu_usage, system_swaps
```

…plus, conditionally:

| Trigger | Adds |
| --- | --- |
| `--sandbox DIR` **OR** `--allow-bash` | `fs_list_dir`, `fs_read_file`, `fs_glob`, `fs_grep`, `fs_write_file`, `get_sandbox_path` |
| `--allow-bash` | `bash` (and bumps the agentic loop's `max_tool_hops` to 99999) |
| `--use-google` (+ env vars set) | `web_google` |
| `--RAG DIR` | `rag` (single dispatcher) — or seven `rag_*` tools when `--split-rag` is also set |
| `--external-tools DIR` | every tool from each loaded `EASYAI-*.tools` manifest |

### Why `--sandbox` and `--allow-bash` both register fs_*

Bash is strictly more permissive than the `fs_*` tools — if the operator
trusts the model with bash, they trust it with `fs_read_file` etc. by
construction. Requiring an extra `--allow-fs` flag for the narrower
surface produced sessions where the model had bash but no `fs_*` and
fell back to `cat > file` / `cat <<EOF` / `sed -i` for ordinary file
work. The new defaults eliminate that trap: any flag that says "the
model can touch files" registers all the file tools at once.

`get_sandbox_path` ships alongside so the model can resolve the real
on-disk path of where its work is landing — distinct from
`get_current_dir`, which reports the live process cwd and can drift.

### Restricting the catalog with `--tools`

Pass `--tools LIST` to override the auto-catalog. Valid names:

```
datetime, plan, web_search, web_fetch, web_google,
get_current_dir, get_sandbox_path,
fs_read_file, fs_list_dir, fs_glob, fs_grep, fs_write_file, bash,
system_meminfo, system_loadavg, system_cpu_usage, system_swaps,
rag (single-dispatcher),
rag_save, rag_append, rag_search, rag_load,
rag_list, rag_delete, rag_keywords (split layout)
```

`web_google` / `bash` / `rag*` still require their respective opt-in
flags even when explicitly listed.

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
  is "where am I?" (`get_current_dir` / `pwd`). Injecting the resolved
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
pass. Omitting any leaves the server's default in place. The one
non-obvious default in the CLI itself is `--repeat-penalty 1.15` —
that's an anti-loop safety net for thinking models that otherwise
rephrase the same plan three times before acting ("I'll write types.h /
Let me write types.h / OK, creating types.h"). Pass `--repeat-penalty
1.0` to disable.

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

`--verbose` enables stderr diagnostics AND writes a complete
transaction log to disk. Default path:

```
/tmp/easyai-cli-{pid}-{epoch}.log
```

Override with `--log-file PATH` (which implies `--verbose`).

The log is a verbatim record of:

* The HTTP request body (every turn — including the resolved system
  prompt with injected blocks, the full tools array, the message
  history).
* Every SSE chunk byte-for-byte.
* Every tool call dispatched: input arguments, output content,
  duration.
* Connection-level events (retries, timeouts, status codes).

Mode 0600. Suitable for replaying / diffing / grepping. The CLI prints
the resolved path at startup unless `EASYAI_NO_AUTO_LOG=1` is set.

To see what the model actually saw:

```bash
easyai-cli --url ... --log-file /tmp/run.log "your prompt"
python3 -c "import json,re; t=open('/tmp/run.log').read();
            # extract the first request body's system message
            ..."
```

(See [§13](#13-worked-examples) for a copy-pasteable extractor.)

---

## 10. RAG — persistent memory

`--RAG <dir>` mounts a directory as the agent's long-term memory. The
default layout exposes ONE `rag` tool with seven sub-actions (`save`,
`append`, `search`, `load`, `list`, `delete`, `keywords`); each memory
is a single Markdown file in `<dir>` that the operator can hand-edit.

Pass `--split-rag` to register seven separate `rag_*` tools instead —
useful for weak / 1-bit-quant tool callers that handle many flat schemas
better than one discriminated schema.

Memories whose title starts with `fix-easyai-` are immutable: save /
append / delete refuse them. Pass `fix=true` (sub-action `save`) to
mint one.

See [`RAG.md`](RAG.md) for the full guide.

---

## 11. External tools

`--external-tools <dir>` loads every `EASYAI-<name>.tools` JSON manifest
in `<dir>` as an operator-defined tool pack. Per-file fault isolation —
a broken manifest doesn't take down the others. Tools spawn via
`fork`+`execve`, never a shell, so a manifest is the supported way to
give the model focused powers without flipping `--allow-bash`.

See [`EXTERNAL_TOOLS.md`](EXTERNAL_TOOLS.md) for the manifest schema and
worked examples.

---

## 12. Management subcommands

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

## 13. Worked examples

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
* `fs_*` (read / write / list / glob / grep) all rooted there too
* `get_sandbox_path` returning `~/projects/tetris`
* `plan` for a visible step checklist
* `[environment]` block with the resolved absolute path
* `[guidance]` block with the assertiveness rule

### Pure chat with no shell access

```bash
easyai-cli --url http://ai.local:8080 -p "summarise transformers in 5 lines"
```

No sandbox, no `--allow-bash` → the model has only `datetime`, `plan`,
`web_*`, and `system_*`. No `[environment]` / `[guidance]` injection
because there's no file / shell affordance.

### Restrict to specific tools

```bash
easyai-cli --url http://ai.local:8080 --tools datetime,web_search,web_fetch \
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

## 14. Cross-references

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
