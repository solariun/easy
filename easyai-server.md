# easyai-server — the OpenAI-compatible chat server

> **A drop-in `llama-server` replacement that loads a GGUF, exposes
> `/v1/chat/completions` (streaming SSE + tool calls), serves an
> embedded SvelteKit webui, and answers MCP / Ollama clients out of the
> same process.** Single binary, hardened systemd unit, central INI
> config, RAG-backed long-term memory, operator-defined external tools.

---

## Table of contents

1. [Configuration — `/etc/easyai/easyai.ini`](#1-configuration--etceasyaieasyaiini)
2. [Command-line flags](#2-command-line-flags)
3. [API endpoints](#3-api-endpoints)
4. [Tool gating + sandbox](#4-tool-gating--sandbox)
5. [Authentication](#5-authentication)
6. [The default persona — Deep](#6-the-default-persona--deep)
7. [Webui customisation](#7-webui-customisation)
8. [Performance tuning](#8-performance-tuning)
9. [Hardening / security](#9-hardening--security)
10. [Cross-references](#10-cross-references)

---

## 1. Configuration — `/etc/easyai/easyai.ini`

Every CLI flag `easyai-server` accepts has a matching INI entry. The
systemd unit's `ExecStart` is intentionally short — operators tweak
this file and restart, not the unit.

### File location

| Path | Notes |
| --- | --- |
| `/etc/easyai/easyai.ini` | Default — what the installer drops and what the systemd unit reads via `--config`. |
| `<other path>` | Pass `--config /path/to.ini` on the binary's command line. |

The file is owned `root:easyai`, mode `640` — readable by the
service user, world-unreadable.

### Precedence

```
   CLI flag    >    INI value    >    hardcoded default
   (highest)         (this file)         (in the binary)
```

If the operator passed `--port 8080` in the systemd unit AND the INI
says `port = 9090`, the server listens on **8080**. Drop the CLI flag
and the INI value takes over.

### Sections

| Section | Purpose | Status |
| --- | --- | --- |
| `[SERVER]` | HTTP layer, paths, tool gating, MCP auth posture | active |
| `[ENGINE]` | Model loading + inference tunables | active |
| `[MCP_USER]` | Bearer-token auth for `/mcp` (one user per line) | active |
| `[TOOLS]` | Per-tool ACL (`mcp_allowed = …`) | reserved for a future release |

The `[TOOLS]` section is recognised by the parser but **not yet
consumed** by code. Populating it does no harm; the keys are ignored
until a future release wires them.

### Format quick reference

```ini
# comments start with # or ;
; both work

[Section]
key = value
key2 = "value with spaces or trailing whitespace"   ; quotes optional
empty_key =
```

- Whitespace around `=`, section names, keys, and values is trimmed.
- Boolean values accept: `on` / `off`, `true` / `false`, `yes` / `no`,
  `1` / `0`, `enable` / `disable`, `enabled` / `disabled`.
- Surrounding double-quotes on values are stripped, so values can
  preserve internal whitespace by being quoted.
- Duplicate keys within a section: last value wins.
- Missing file: server starts with hardcoded defaults.

### `[SERVER]`

The HTTP layer, paths, tool gating, MCP auth.

| Key | Type | CLI equivalent | Default | Notes |
| --- | --- | --- | --- | --- |
| `model` | path | `-m`, `--model` | (none — REQUIRED) | GGUF file the engine loads. |
| `host` | string | `--host` | `127.0.0.1` | Bind address. `0.0.0.0` to listen on every interface. |
| `port` | int | `--port` | `8080` | TCP port. |
| `alias` | string | `-a`, `--alias` | basename of `model` | Public model id reported by `/v1/models` and `/api/tags`. |
| `sandbox` | path | `--sandbox` | (none) | Root directory for `bash` and `fs_*` tools. |
| `system_file` | path | `-s`, `--system-file` | (none — uses built-in default) | File containing the server-default system prompt. |
| `system_inline` | string | `--system` | (none) | Inline system prompt. Beats `system_file` if both are set. |
| `external_tools` | path | `--external-tools` | (none — feature off) | Directory of `EASYAI-*.tools` manifests. See `EXTERNAL_TOOLS.md`. |
| `rag` | path | `--RAG` | (none — feature off) | Directory of RAG entries. See `RAG.md`. |
| `webui_title` | string | `--webui-title` | `Deep` | Document title pinned in the embedded webui. |
| `webui_icon` | path | `--webui-icon` | (none) | `.ico` / `.png` / `.svg` / `.gif` / `.jpg` / `.webp`. |
| `webui_mode` | enum | `--webui` | `modern` | `modern` (embedded llama-server bundle) or `minimal` (inline). |
| `webui_placeholder` | string | `--webui-placeholder` | `Type a message…` | Input box hint. |
| `metrics` | bool | `--metrics` | `off` | Expose Prometheus `/metrics`. |
| `verbose` | bool | `-v`, `--verbose` | `off` | Noisy logs. |
| `allow_fs` | bool | `--allow-fs` | `off` | Register `fs_read_file / fs_write_file / fs_list_dir / fs_glob / fs_grep` (sandbox required). |
| `allow_bash` | bool | `--allow-bash` | `off` | Register the `bash` tool. **Not** a hardened sandbox. |
| `use_google` | bool | `--use-google` | `off` | Register the `web_google` tool (Google Custom Search JSON API). Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` env vars. Counts against your Google quota (free tier: 100 queries/day per key). When either env var is missing the tool is silently skipped. |
| `experimental_rag` | bool | `--experimental-rag` | `off` | Collapse the six `rag_*` tools into a single `rag(action=...)` tool. On-disk format unchanged; the six tools are NOT registered when this is on. Smaller catalog at the cost of accuracy on weak / 1-bit-quant tool callers — leave off when running Bonsai-class models. |
| `mcp` | string | `--mcp` | (none — MCP client off) | URL of an upstream MCP server to connect to as a CLIENT. Format: `http(s)://host:port` (the `/mcp` endpoint is appended). Tools fetched from the upstream are merged into the local catalogue; local-tool names take precedence on collision. Failure at startup logs a warning and continues with whatever local / RAG tools were registered. |
| `mcp_token` | string | `--mcp-token` | (empty) | Bearer token sent on every request to the upstream `mcp` URL. Empty = no `Authorization` header — appropriate when the upstream is in open mode. Don't put a real token in the INI directly if you can help it; load it from a separate file (analogous to how `api_key` is wired through `${EASYAI_API_KEY}` in the systemd installer). |
| `local_tools` | bool | `--no-local-tools` (negative) | `on` | Master switch for the LOCAL built-in toolbelt (datetime, web_*, fs_*, bash, ...). Set `off` (or pass `--no-local-tools`) to register zero local default tools. Has no effect on RAG, external tools, or remote tools fetched via `mcp` — those have their own switches. `allow_fs` / `allow_bash` / `use_google` further opt in. **Renamed from `load_tools` / `--no-tools`** to make clear the MCP client (`mcp`) is unaffected. |
| `max_body` | int | `--max-body` | `8388608` (8 MiB) | Max HTTP request body size. |
| `api_key` | string | `--api-key` | (none — `/v1/*` open) | Bearer token for `/v1/*`. Don't put real keys in INI directly — use `/etc/easyai/api_key` (file-based, the installer wires `${EASYAI_API_KEY}`). |
| `mcp_auth` | enum | (no CLI; `--no-mcp-auth` overrides) | `auto` | `auto` (auth iff `[MCP_USER]` non-empty), `on` (force require), `off` (force open). |
| `no_think` | bool | `--no-think` | `off` | Strip `<think>` tags from responses. |
| `inject_datetime` | bool | `--inject-datetime` | `on` | Authoritative date/time + knowledge-cutoff injection in the system prompt. |
| `knowledge_cutoff` | string | `--knowledge-cutoff` | `2024-10` | YYYY-MM hint for the model. |
| `reasoning` | bool | `--reasoning on/off` | `on` | Honour the model's reasoning channel. |

### `[ENGINE]`

Model loading and inference tunables.

| Key | Type | CLI equivalent | Default | Notes |
| --- | --- | --- | --- | --- |
| `context` | int | `-c`, `--ctx` | `4096` | Context window size in tokens. |
| `ngl` | int | `--ngl` | `-1` (auto-fit) | GPU layers. `-1` = auto, `0` = CPU only, `99` = force all on GPU (will OOM if it doesn't fit). |
| `threads` | int | `-t`, `--threads` | `0` (lib default) | CPU threads for prompt processing. |
| `threads_batch` | int | `-tb`, `--threads-batch` | `0` (lib default) | CPU threads for batched inference. |
| `batch` | int | `--batch` | `0` (follows ctx) | Logical batch size. |
| `parallel` | int | `-np`, `--parallel` | `1` | Llama-server compat. |
| `preset` | string | `--preset` | `balanced` | `deterministic` / `precise` / `balanced` / `creative` / `wild`. |
| `flash_attn` | bool | `-fa`, `--flash-attn` | `off` | Free perf on every backend that supports it. |
| `mlock` | bool | `--mlock` | `off` | Pin model weights in RAM. Needs `LimitMEMLOCK=infinity` on the unit. |
| `no_mmap` | bool | `--no-mmap` | `off` | Required with `mlock` for portability. |
| `no_kv_offload` | bool | `-nkvo`, `--no-kv-offload` | `off` | Keep KV cache on CPU even with GPU layers. |
| `kv_unified` | bool | `--kv-unified` | `off` | Single unified KV buffer across sequences. |
| `cache_type_k` | enum | `-ctk`, `--cache-type-k` | `f16` | `f32 / f16 / bf16 / q8_0 / q4_0 / q4_1 / q5_0 / q5_1 / iq4_nl`. |
| `cache_type_v` | enum | `-ctv`, `--cache-type-v` | `f16` | Same options as K. Quantising V saves a lot of VRAM. |
| `numa` | string | `--numa` | (none) | Llama-server compat. |
| `override_kv` | list | `--override-kv` (repeat) | (empty) | GGUF metadata overrides. Comma-separated list of `key=type:value` triples (e.g. `tokenizer.ggml.eos_token_id=int:151645`). On the CLI, repeat `--override-kv` per entry; in INI, comma-separate. |
| `temperature` | float | `--temperature`, `--temp` | (preset) | Sampling override. |
| `top_p` | float | `--top-p` | (preset) | Sampling override. |
| `top_k` | int | `--top-k` | (preset) | Sampling override. |
| `min_p` | float | `--min-p` | (preset) | Sampling override. |
| `repeat_penalty` | float | `--repeat-penalty` | (preset) | Sampling override. |
| `max_tokens` | int | `--max-tokens` | `-1` (until EOS / ctx full) | Per-turn cap. |
| `seed` | uint32 | `--seed` | `0` (random) | RNG seed. |
| `max_incomplete_retries` | int | `--max-incomplete-retries` | `10` | How many times the engine discards + nudges + retries when the model finishes a turn with no tool_call and only an "announce" snippet ("Let me…", "I'll…"). `0` disables retries (equivalent to `retry_on_incomplete = off`). Bump to 15-20 for weak / 1-bit-quant models that keep announcing-without-acting. Each retry surfaces in the webui Thinking panel as `↻ Retry N/max`. |

### `[MCP_USER]`

Bearer-token authentication for `POST /mcp`. Each line registers one
user:

```ini
[MCP_USER]
gustavo  = abcdef0123456789...   ; openssl rand -hex 32
ci       = different-strong-token
claude   = token-for-claude-desktop
```

- The username on the matching line is logged per request:
  `[mcp] request from user 'gustavo'`. The token is never logged.
- If the section is missing or empty, `/mcp` is open (subject to
  `[SERVER] mcp_auth` and `--no-mcp-auth`).
- The first matching token wins. Duplicate tokens (operator config
  bug) silently take the alphabetically-first user.
- Generate strong tokens with `openssl rand -hex 32` or
  `python3 -c 'import secrets; print(secrets.token_hex(32))'`.
- Restart the server to pick up changes:
  `sudo systemctl restart easyai-server`.

Full per-client connection guides (Cursor / Continue / Claude Desktop
/ curl) live in `MCP.md` §4–7.

### `[TOOLS]` (RESERVED for a future release)

Per-tool ACL controlled by glob patterns. **Not yet consumed by
code** — the parser accepts the section, the binary ignores it.
Documented up-front so operators can plan / pre-populate.

```ini
[TOOLS]
mcp_allowed = rag_*, datetime, web_search, web_fetch
mcp_denied  = bash, fs_write_file
```

Future semantics (subject to refinement):

- `mcp_allowed` — comma-separated globs of tool names that may be
  exposed via `/mcp` and dispatched. Empty / missing = all tools
  allowed.
- `mcp_denied` — same shape; takes precedence over `mcp_allowed`.

### Worked examples

#### Minimal local-dev INI

```ini
[SERVER]
model = /home/me/models/qwen2.5-3b.gguf
host  = 127.0.0.1
port  = 8080

[ENGINE]
context = 32768
ngl     = -1
preset  = balanced
```

Runs on localhost only. MCP open (no `[MCP_USER]`). Auto-fit GPU.

#### Production server with auth

```ini
[SERVER]
model           = /var/lib/easyai/models/ai.gguf
host            = 0.0.0.0
port            = 80
alias           = EasyAi
sandbox         = /var/lib/easyai/workspace
system_file     = /etc/easyai/system.txt
external_tools  = /etc/easyai/external-tools
rag             = /var/lib/easyai/rag
webui_title     = EasyAi
metrics         = on
verbose         = off
mcp_auth        = on

[ENGINE]
context         = 128000
ngl             = -1
threads         = 16
threads_batch   = 16
preset          = balanced
flash_attn      = on
cache_type_k    = q8_0
cache_type_v    = q8_0
mlock           = on
no_mmap         = on

[MCP_USER]
gustavo  = REPLACE-WITH-OPENSSL-RAND-HEX-32
ci       = ANOTHER-STRONG-TOKEN
```

Systemd unit's `ExecStart` is just:

```
ExecStart=/usr/bin/easyai-server --config /etc/easyai/easyai.ini
```

(The installer additionally appends `-m <model>` and `--api-key
'${EASYAI_API_KEY}'` for runtime substitution from
`/etc/easyai/api_key`.)

#### Override one value via CLI

Operator wants to test a different model without editing the INI:

```
ExecStart=/usr/bin/easyai-server --config /etc/easyai/easyai.ini -m /tmp/test-model.gguf
```

CLI `-m` overrides INI `model`. Everything else still comes from
the INI.

#### Disable MCP auth temporarily

```
ExecStart=/usr/bin/easyai-server --config /etc/easyai/easyai.ini --no-mcp-auth
```

CLI flag wins over `[SERVER] mcp_auth = on` and over `[MCP_USER]`.

---

## 2. Command-line flags

The CLI accepts every key from the INI plus a handful of
binary-only flags (`--config`, `--no-mcp-auth`). Pass `--help` for
the full list at the version you're running.

```
easyai-server -m model.gguf [options]
```

Required:

```
  -m, --model <path>           GGUF model file
```

The full list mirrors the `[SERVER]` and `[ENGINE]` tables in §1 —
each row's "CLI equivalent" column is the CLI alias, and the "Default"
column is the value when neither CLI nor INI sets it.

Binary-only flags (no INI equivalent):

| Flag | Purpose |
| --- | --- |
| `--config <path>` | Override the default INI path (`/etc/easyai/easyai.ini`). |
| `--no-mcp-auth` | Force `/mcp` open even if `[MCP_USER]` populated. Emergency override. |
| `-h`, `--help` | Print usage and exit. |

---

## 3. API endpoints

The server speaks **three** API dialects so most AI clients work
unchanged.

| Verb | Path | Dialect | Auth | Notes |
| --- | --- | --- | --- | --- |
| GET | `/` | webui | (open) | Embedded SvelteKit chat UI. |
| GET | `/bundle.{js,css}` | webui | (open) | Bundle assets. |
| GET | `/loading.html` | webui | (open) | Loading splash. |
| GET | `/favicon` (+ `.ico`/`.svg`) | webui | (open) | Operator-supplied or embedded brain SVG. |
| GET | `/health` | easyai | (open) | `{model, backend, tools, preset, compat:{...}}` — liveness probe. |
| GET | `/metrics` | easyai | api_key | Prometheus exposition (only when `--metrics` is on). |
| GET | `/v1/models` | OpenAI | api_key | OpenAI-shape list-models. |
| GET | `/v1/tools` | easyai | api_key | Tool catalogue for the webui popover. |
| POST | `/v1/chat/completions` | OpenAI | api_key | The workhorse — streaming SSE, tools, sampling controls. |
| POST | `/v1/preset` | easyai | api_key | Swap the ambient preset. |
| GET | `/api/tags` | Ollama | api_key | Ollama-shape list-models (LobeChat, OpenWebUI in Ollama mode, etc.). |
| GET/POST | `/api/show` | Ollama | api_key | Ollama-shape model detail. |
| POST | `/mcp` | MCP | `[MCP_USER]` | JSON-RPC 2.0 — full tool catalogue exposed to other AI apps. See `MCP.md`. |
| GET | `/mcp` | MCP | (open) | Reserved for future SSE notification stream; currently `405 Method Not Allowed`. |

### `POST /v1/chat/completions` — the killer feature

When a *client* (Claude Code, an OpenAI SDK, LiteLLM, LangChain…)
posts its **own** `system` message and/or **own** `tools`, those win
for that single request:

* Client provides `tools` → easyai forwards generated tool calls back
  to the client and *does not* dispatch them locally. The client
  controls the loop.
* Client provides no `tools` → easyai uses its own toolbelt and runs
  the multi-hop loop server-side, returning the final assistant
  message.

Either way the server-supplied `system.txt` is used **only** when the
request doesn't already include a `system` message. The
authoritative-datetime preamble (§"Hardening" below) appends to
whichever system message reaches the model.

### `X-Easyai-Inject: off` to skip the date/time preamble

Useful for A/B regression suites:

```bash
curl -H "X-Easyai-Inject: off" ...
```

Header values: `off` (skip preamble for this request only), `on`
(force injection on this request even when the server was launched
with `--inject-datetime off`), or absent (defer to the server flag).

### `POST /mcp` — Model Context Protocol

JSON-RPC 2.0 over a single endpoint. Methods: `initialize`,
`tools/list`, `tools/call`, `ping`, plus the standard MCP
notifications. Full surface in `MCP.md`.

```sh
curl -fsS http://localhost/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | jq '.result.tools[] | .name'
```

Auth model: see §5 below. For a higher-throughput tool API without
the model in the loop, use [`easyai-mcp-server`](easyai-mcp-server.md).

---

## 4. Tool gating + sandbox

Default is **safe**: no filesystem access, no shell. Every privileged
opt-in is logged at startup with sanity warnings.

| Flag / INI key | What it enables |
| --- | --- |
| (no flag) | `datetime`, `web_search`, `web_fetch`. |
| `--sandbox <dir>` (`[SERVER] sandbox`) | Sets the root for filesystem-flavoured tools. The binary `chdir`s into `<dir>` so `get_current_dir` reports the sandbox path back to the model. Required for `--allow-fs`. |
| `--allow-fs` (`[SERVER] allow_fs`) | `fs_read_file / fs_write_file / fs_list_dir / fs_glob / fs_grep` plus `get_current_dir`, ALL scoped to the sandbox. |
| `--allow-bash` (`[SERVER] allow_bash`) | `bash` (run `/bin/sh -c`). cwd = sandbox if given, else the binary's CWD. NOT a hardened sandbox — runs with this process's user privileges. Also bumps the agentic-loop `max_tool_hops` to 99999 (bash flows naturally span many turns). |
| `--use-google` (`[SERVER] use_google`) | `web_google` (Google Custom Search JSON API). Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` env vars; tool is silently skipped if either is missing so a key rotation that briefly drops the env doesn't take down the server. Counts against your Google quota (free tier: 100 queries/day per key). |
| `--external-tools <dir>` (`[SERVER] external_tools`) | Load every `EASYAI-<name>.tools` file in `<dir>` as an operator-defined tool pack. Per-file fault isolation. Spawns via `fork`+`execve` — never a shell. **The supported way to give the model focused powers without flipping `--allow-bash`.** See [`EXTERNAL_TOOLS.md`](EXTERNAL_TOOLS.md). |
| `--RAG <dir>` (`[SERVER] rag`) | Enable RAG, the agent's persistent **memory** (search / store / recall / update / forget). Six tools by default — `rag_save`, `rag_search`, `rag_load`, `rag_list`, `rag_delete`, `rag_keywords` — each entry one Markdown file in `<dir>`, operator-readable and hand-editable. Memories whose title starts with `fix-easyai-` are immutable: save/delete refuse them. Pass `fix=true` to `rag_save` to mint one. The systemd-installed server passes this by default (`/var/lib/easyai/rag`). See [`RAG.md`](RAG.md). |
| `--experimental-rag` (`[SERVER] experimental_rag`) | Replace the six `rag_*` tools with a single `rag(action=...)` dispatcher. Same `RagStore`, same on-disk format, same fix-memory semantics — only the catalog shape changes. Smaller catalog (1 tool entry vs 6) at the cost of weaker performance on small / 1-bit-quant tool callers; leave off for Bonsai-class models. Has no effect when `--RAG` is also off. |
| `--mcp <url>` (`[SERVER] mcp`) | Connect to a remote MCP server as a CLIENT. The upstream's tool catalogue is merged into ours via `tools/list` at startup; each remote tool's handler proxies `tools/call` over HTTP. Local-tool names take precedence on collision (warning logged, remote dup skipped). Pair with `--mcp-token` for bearer-auth servers. Connect failure logs a warning and continues. |
| `--mcp-token <token>` (`[SERVER] mcp_token`) | Bearer token attached to every `--mcp` request. Empty = no auth header. |
| `--no-local-tools` (`[SERVER] local_tools = off`) | Skip the LOCAL built-in toolbelt entirely (renamed from `--no-tools` / `load_tools`). Useful when you want ONLY external-tools, ONLY RAG, or ONLY tools fetched via `--mcp`. The MCP client remains active even with this flag set. |

Sandbox semantics: paths sent by the model are anchored to the root
by iterating path components and dropping any `..`, `.`, or absolute
markers before joining. Total containment by construction — there is
no path the model can construct that escapes. The model sees a
virtual `/`-rooted filesystem (`/report.md`, `/docs/spec.md`); the
real sandbox path is hidden from descriptions and result messages.

Concurrency: built-in tools that share state (RAG's index,
web_fetch's LRU cache) use lock-free or fine-grained synchronisation
internally. RAG specifically uses `std::shared_mutex` so parallel
reads from multiple workers don't serialise — the same tools used by
[`easyai-mcp-server`](easyai-mcp-server.md) under thousands-of-clients
load. See [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) §16, §16.6b.

---

## 5. Authentication

Two auth surfaces, two mechanisms.

### `/v1/*` — single shared API key (`--api-key`)

`require_auth` checks every `/v1/*` request for `Authorization: Bearer
<key>`. When `--api-key` (or `[SERVER] api_key`) is set, every request
without a valid match returns 401. `/health` stays open so liveness
probes don't need a credential.

The Authorization header is capped at 4 KiB before string comparison
to defend against multi-megabyte headers being sent on every probe.

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost/v1/models
```

The systemd installer wires `${EASYAI_API_KEY}` from
`/etc/easyai/api_key` (mode 600, owned by `easyai`) so the literal
key never appears in unit files.

### `/mcp` — per-user Bearer tokens (`[MCP_USER]`)

Three-way precedence on the gate:

1. `--no-mcp-auth` on the CLI (or `[SERVER] mcp_auth = off`) →
   force open; the `[MCP_USER]` table is cleared even if populated.
2. `[MCP_USER]` populated → Bearer required, looked up in the
   token→username map at request time. Audit log per request:
   `[mcp] request from user 'gustavo'`. The token is never logged.
3. `[MCP_USER]` empty/missing → open (zero-friction local-dev
   default).

The same 4 KiB header cap applies here. Generate strong tokens with
`openssl rand -hex 32` and rotate by editing `[MCP_USER]` and
restarting the server (no token cache, no in-memory survival of old
tokens after restart).

### Compensating controls for high-trust deployments

1. **Bind to LAN only** — `[SERVER] host = 127.0.0.1`; SSH-tunnel
   from clients.
2. **Reverse proxy with mTLS / IP allowlist** — nginx / Caddy in
   front, require client cert or restrict by source net.
3. **Token rotation** — `[MCP_USER]` edits land on next restart;
   old tokens are immediately invalid.
4. **Don't enable `--allow-bash` with auth-open mode** — the worst
   `/mcp` can dispatch is RAG + read-only `web_*` + your
   `--external-tools` allowlist.

Full security model: [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) §17.

---

## 6. The default persona — Deep

A fresh `easyai-server` boots up as **Deep** — an expert system
engineer who answers from CHECKED FACTS, not impressions. Built into
the default system prompt so a small open-weights model behaves like
an engineer instead of a chatbot from minute one.

Deep's operating loop is: **TIME → THINK → PLAN → EXECUTE → VERIFY**.

- **Time first.** Any question that touches "now", "today", a
  deadline, a release version, or a fact that could have changed
  since training cutoff → `datetime` is the first tool call. Anchors
  the rest of the turn to the real wall clock.
- **Think.** State the goal, identify what's known vs. needs lookup,
  what could go wrong.
- **Plan.** Multi-step tasks call `plan(action='add', text=…)` first
  so the user can see and intervene live.
- **Execute.** Every registered tool is fair game.
- **Verify.** Before claiming success — does the file exist? does
  the test pass? does the URL really say that? When in doubt, run
  another tool instead of guessing.

Old behaviour rules carry over: `RULE 1` (execute or answer, never
just announce), `web_search → web_fetch` mandatory, citations stick
to the URL actually fetched.

Operators who want a different persona pass `--system "<text>"`
(`[SERVER] system_inline`) or `-s persona.txt` (`[SERVER]
system_file`) — Deep is the default, not a hardcoded identity.

---

## 7. Webui customisation

The webui shipped is the compiled SvelteKit bundle from `llama-server`,
embedded into the binary at build time via `cmake/xxd.cmake` (one
`.hpp` per asset). Customisations are runtime DOM injection +
at-startup string substitution on `bundle.js`:

- **Title pin** via `Object.defineProperty(document, 'title', {set:})` so
  the bundle's hard-coded `"llama.cpp - AI Chat Interface"` doesn't
  win.
- **DOM scrubber** — a `MutationObserver` matches MCP / Sign-in /
  Authorize / Load-model / Use-Pyodide chrome by visible text and
  hides the containing card / list-item / dialog.
- **Fetch interceptor** — 501s `/authorize`, `/token`, `/register`,
  `/.well-known/*`, `/models/load`, `/cors-proxy`, `/dev/poll`,
  `/home/web_user/*`; stubs `/properties` with `{}`; and tees the SSE
  response of `/v1/chat/completions` into a status-pill state machine.
- **Tone chip + metrics bar** — `deterministic / precise / balanced /
  creative` selector + `ctx X/Y · last N tok · s · t/s` overview.
- **Per-message status pill** — appended to each assistant action
  toolbar; shows `thinking` / `answering` / `fetching · <tool>` /
  `complete · 98 tok · 4.4s · 22.3 t/s`.
- **Reasoning panel shrink** — the bundle's native Reasoning panel is
  capped at 18em tall and auto-collapses on `finish_reason`.

Operator-tunable knobs (all match an INI key — `webui_title`,
`webui_icon`, `webui_mode`, `webui_placeholder`):

| CLI | INI | Default | Notes |
| --- | --- | --- | --- |
| `--webui-title <str>` | `webui_title` | `Deep` | Sidebar / window title text. |
| `--webui-icon <path>` | `webui_icon` | (embedded brain SVG) | `.ico` / `.png` / `.svg` / `.gif` / `.jpg` / `.webp`. |
| `--webui <mode>` | `webui_mode` | `modern` | `modern` (embedded bundle) or `minimal` (single-file inline UI). |
| `--webui-placeholder <str>` | `webui_placeholder` | `Type a message…` | Input box hint. |

Browser cache caveat: after any change to served HTML/JS, hit
**Cmd+Shift+R** (Linux: Ctrl+Shift+R) to force-reload. The bundle is
hashed so a stale CSS file is the usual culprit.

---

## 8. Performance tuning

### Context size

```
-c 128000
```

128k tokens covers most chat sessions without reaching the cap. Keep
in mind the KV cache scales linearly with context.

### KV cache quantisation

```
-ctk q8_0 -ctv q8_0
```

Halves the KV cache footprint at no measurable quality loss on modern
models. `-ctv q4_0` halves it again at a small perplexity cost — try
if you're VRAM-bound. The systemd installer ships `q8_0` by default.

### Flash attention

```
-fa
```

Free perf on every backend that supports it (CUDA, Metal, Vulkan ≥
recent). Required for some KV-quant combinations.

### GPU layers (`--ngl`)

```
--ngl -1
```

Auto-fit. Manually pinning `--ngl 99` can poison `common_fit_params`
if it doesn't fit (it refuses to lower a user-pinned value), so let
auto-fit decide.

### `--mlock` and `--no-mmap`

Pinning model weights in RAM prevents the kernel paging them under
memory pressure (catastrophic for token latency). `--no-mmap` is
needed for `--mlock` to be portable across all GPU backends. The
drop-in's `LimitMEMLOCK=infinity` is what makes `mlock` actually work.

### CPU threads

```
-t <jobs> -tb <jobs>
```

For a single-user agent box, set both to `nproc`. For shared hardware,
reduce so the agent doesn't starve the rest of the system.

### Tool budget

`Engine::chat()` caps at 8 tool hops by default; bumps to 99999 when
`--allow-bash` is on (bash flows span many turns). A model that runs
away calling tools without converging will hit the cap and bail with
the last partial answer.

### Incomplete-turn retry budget

`--max-incomplete-retries N` (default 10) — how many times the engine
discards + nudges + retries when the model finishes a turn announcing
an action ("Let me…", "I'll…") without actually emitting the
tool_call. Bump to 15-20 for weak / 1-bit-quant models; set to 0 to
disable retries entirely. Each retry surfaces in the webui Thinking
panel as `↻ Retry N/max`.

---

## 9. Hardening / security

Highlights of the work documented in [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md):

- **`std::regex` is banned on hostile input.** A 2026-04-26 production
  SIGSEGV (94 766 stack frames in libstdc++'s recursive regex engine
  on an HTML page from `web_fetch`) drove the rule. All scanners over
  model output / HTTP bodies / file contents are forward-only.
- **JSON depth cap (64 levels)** on every accepted-from-network body
  (`/v1/chat/completions`, `/mcp`). Iterative walk so the validator
  itself doesn't recurse.
- **Authorization header cap (4 KiB)** on `/v1/*` and `/mcp` — defends
  against multi-megabyte Bearer headers used to amortise CPU on
  hostile probes.
- **Body cap (default 8 MiB, `--max-body`)** at the cpp-httplib layer.
- **Fork+execve hardening** for `bash` and external-tools subprocesses:
  `setpgid` for process-group lifetime, `PR_SET_PDEATHSIG(SIGKILL)`
  on Linux, fd close-loop bounded by `kMaxFdScan = 65536` (defeats
  `RLIMIT_NOFILE = unlimited`), stdin → `/dev/null`, opt-in
  `env_passthrough`.
- **Sandbox symlink-escape closed.** `Sandbox::resolve` runs
  `fs::weakly_canonical()` + path-component containment, plus
  `O_NOFOLLOW | O_CLOEXEC` on `fs_read_file` / `fs_write_file` so a
  TOCTOU race can't follow a last-second symlink.
- **RAG entries written mode 0600** so the OS-level ACL is
  owner-only even if the operator's umask leaves a wider default.
- **Authoritative date/time preamble** appended to whichever system
  message reaches the model (server's default OR client-supplied).
  Suppresses post-cutoff hallucination by anchoring "today" to the
  real wall clock and explicitly telling the model to call a tool or
  state uncertainty for facts beyond `--knowledge-cutoff`.
- **Three audit passes** — first / second / third (latest 2026-04-30,
  closing 3 HIGH and 7 MEDIUM findings including the `apply_ini_to_args`
  dead-code path, `--no-mcp-auth` disconnect, sandbox symlink escape,
  and `bash` fork-hardening parity with external-tools).

For threat models requiring OS-level isolation: run easyai-server
inside a container / firejail / unprivileged user with disabled
network egress. The sandbox + tool gates bound what the *model* can
ask for; the OS bounds what the *agent process* can do.

---

## 10. Cross-references

- [`README.md`](README.md) — sales overview + quickstart.
- [`LINUX_SERVER.md`](LINUX_SERVER.md) — operator's guide for the
  systemd-installed server (file layout, the unit file, perf tuning,
  gotchas, backup / upgrade / uninstall).
- [`easyai-mcp-server.md`](easyai-mcp-server.md) — standalone
  MCP-only server: same tools, no model, dedicated for high-concurrency
  multi-client deployments.
- [`MCP.md`](MCP.md) — Model Context Protocol surface;
  per-client connection cookbook (Claude Desktop / Cursor / Continue /
  curl).
- [`RAG.md`](RAG.md) — persistent registry, the six tools, workflows.
- [`EXTERNAL_TOOLS.md`](EXTERNAL_TOOLS.md) — operator-defined
  external tools (`EASYAI-*.tools` JSON manifests).
- [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) — three audit passes,
  HIGH / MEDIUM / LOW findings, accepted residual risk.
- [`design.md`](design.md) — architecture + why decisions.
- [`manual.md`](manual.md) — hands-on developer manual.
