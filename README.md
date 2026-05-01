# easyai

> **A C++17 framework anyone can use to build AI agents that talk to
> their own services — no llama.cpp, JSON-Schema, or template-engine
> knowledge required.**

easyai turns [llama.cpp](https://github.com/ggml-org/llama.cpp) into an
*agent engine* you can drop into any program in a dozen lines.  You give
it C++ functions; it gives the model the ability to call them.  That's
the whole pitch.

It ships **two libraries** you can `find_package(easyai)` and link
against, plus six ready-to-run binaries:

| Library             | Purpose                                                                                                                                       |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `libeasyai`         | Local llama.cpp engine — `easyai::Engine`, `easyai::Tool`, built-in tools, presets, `easyai::Plan`.  Linked via `easyai::engine`.            |
| `libeasyai-cli`     | OpenAI-protocol client — `easyai::Client` mirrors `Engine` but the model runs on a remote `/v1/chat/completions` endpoint while tools execute locally.  Linked via `easyai::cli`. |

| Binary               | What it gives you                                                                                                                                  |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `easyai-local`       | Local-only REPL: loads a GGUF in-process via `easyai::Engine`. Drop-in `llama-cli` replacement — one-shot scripting (`-p`), tools, presets, optional `<think>` strip, sandboxed `fs_*` tools, opt-in `bash` tool. |
| `easyai-cli`         | Agentic OpenAI-protocol client built on `libeasyai-cli` — no local model.  REPL or `-p`, full sampling control (`--temperature`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`, `--frequency-penalty`, `--presence-penalty`, `--seed`, `--max-tokens`, `--stop`), plan tool, server-management subcommands (`--list-models`, `--list-tools`, `--health`, `--props`, `--metrics`, `--set-preset`).  HTTPS via OpenSSL; `--insecure-tls` / `--ca-cert` for dev/internal CAs. |
| `easyai-server`      | Drop-in `llama-server` replacement: OpenAI-compat HTTP **with full SSE streaming**, embedded SvelteKit webui, Bearer auth, Prometheus `/metrics`, KV-cache controls, flash-attn, mlock.  Speaks MCP, OpenAI, Ollama from one process.  Full doc: [`easyai-server.md`](easyai-server.md). |
| `easyai-mcp-server`  | **Standalone Model Context Protocol provider — no model loaded.** Same tool catalogue as `easyai-server` (built-ins + RAG + external-tools), exposed over `POST /mcp` with a configurable cpp-httplib worker pool (`--threads`) and an in-flight `tools/call` cap (`--max-concurrent-calls`) for thousands-of-clients deployments.  Full doc: [`easyai-mcp-server.md`](easyai-mcp-server.md). |
| `easyai-agent`       | A demo agent showing every built-in tool plus an inline custom tool.                                                                                |
| `easyai-recipes`     | Tutorial agent paired with `manual.md` — implements `today_is` and `weather` (HTTP-calling) from scratch.                                          |
| `easyai-chat`        | A bare-bones REPL with no tools — useful as a sanity check.                                                                                          |

> **Status** — used in production on a Linux Vulkan box (Radeon 680M)
> as a self-hosted ChatGPT-style assistant.  Apple Silicon (Metal),
> Linux/Windows Vulkan, NVIDIA CUDA, and AMD ROCm are all wired up out
> of the box.  `scripts/install_easyai_server.sh` handles the whole
> Debian/Ubuntu deployment in one command (systemd-coredump,
> hardened unit, optional `--enable-verbose`, drop-in compat with
> `install_llama_server.sh`).

---

## What's new

A running log of user-facing changes. Latest first — keep this list
current as features land so anyone returning to the repo (or
landing on it for the first time) sees what shipped recently.

### 2026-04-30 — `easyai-mcp-server` (standalone MCP provider)

* **New binary `easyai-mcp-server`.** Same tool catalogue as
  `easyai-server` (built-ins + RAG + operator-defined external-tools)
  exposed over `POST /mcp` with **no GGUF model loaded** — designed
  for high-concurrency multi-client deployments. Configurable
  cpp-httplib worker pool (`--threads`, default 256) and a separate
  in-flight `tools/call` cap (`--max-concurrent-calls`, default 256)
  that returns 503 + `Retry-After` on saturation instead of unbounded
  queueing. Full doc: [`easyai-mcp-server.md`](easyai-mcp-server.md).
* **RAG concurrency upgrade.** `RagStore::mu` is now
  `std::shared_mutex`; `rag_search` / `rag_load` / `rag_list` /
  `rag_keywords` take `std::shared_lock` so parallel readers don't
  serialise on the write path. Benefits every consumer of libeasyai
  — `easyai-server`, `easyai-cli` with `--RAG`, any third-party
  program calling `make_rag_tools()`. Atomic-rename writes already
  made on-disk reads tear-free; the lock relaxation is safe.
* **Doc restructure.** `INI_KFlags.md` content has moved to the top
  of the new [`easyai-server.md`](easyai-server.md) so the chat
  server's INI / CLI / API / persona / hardening reference lives in
  one file. `LINUX_SERVER.md` is unchanged — it remains the
  systemd-installer-specific operator's guide.

### 2026-04-30 — Tunable incomplete-retry budget + live retry visibility

* **`--max-incomplete-retries N` (also `[ENGINE] max_incomplete_retries`).**
  Default 10 — how many times the engine discards + nudges + retries
  when the model finishes a turn announcing an action ("Let me…",
  "I'll…") without actually emitting the tool_call. Bump to 15-20
  for weak / 1-bit-quant models (Bonsai-8B-Q1_0 frequently needs
  the extra budget); set to 0 to disable retries entirely.
* **Retries now visible in the Thinking panel.** Engine fires a new
  `on_incomplete_retry(attempt, max, reason)` callback per retry,
  the server pipes it into the SSE `reasoning_content` channel, and
  the webui renders `↻ Retry 3/10: model said: "Let me search…" (no
  tool_call) — nudging.` while it happens. No more frozen UI for 10
  silent retries followed by a blank bubble.
* **Engine warnings always log** (regardless of `--verbose`):
  cancellation, thought-only retry, reasoning→content fallback,
  incomplete-retry, empty final content. `--verbose` is for raw
  per-token / per-hop diagnostic noise; actionable warnings stay on
  so operators see them in `journalctl` without flipping a flag.

### 2026-04-30 — Bonsai 8B Q1_0 onboarding + security pass

* **One-shot installers for macOS and Raspberry Pi 4/5.**
  `scripts/install_easyai_macos.sh` builds with Metal/AMX, drops the
  model, prints the run command. `scripts/install_easyai_pi.sh` does
  the full Pi appliance: systemd unit, mDNS so the box answers as
  **`pi-ai.local`** on your LAN, port 80 with
  `CAP_NET_BIND_SERVICE`. Both clone the **PrismML fork** of
  llama.cpp (the only one with the Q1_0 kernel — upstream loads the
  GGUF then fails at decode).
* **Security third-pass audit** — 3 HIGH and 7 MEDIUM findings fixed.
  The INI overlay used to be silently ignored (every `[ENGINE]` /
  `[SERVER]` key was a no-op); `--no-mcp-auth` was disconnected from
  the gate; the sandbox could be escaped by a symlink planted via
  `bash`. All closed. The `bash` tool now gets the same
  fork-hardening as external tools — `PR_SET_PDEATHSIG`, fd
  close-loop bounded against `RLIMIT_NOFILE = unlimited`, process-
  group kill on timeout. Plus JSON-depth caps on every parser, a
  bounded INI parser, mode 0600 on RAG entries, and a
  body-size-bounded auth header. See [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) §18.
* **MCP server.** `easyai-server` is now a Model Context Protocol
  provider on `POST /mcp` (protocol 2024-11-05). Claude Desktop,
  Cursor, Continue list and dispatch every registered tool — your
  built-ins, your RAG, your `--external-tools` manifests — over a
  single endpoint. Bearer auth via `[MCP_USER]` in the INI; a
  Python stdio bridge ships at `scripts/mcp-stdio-bridge.py` for
  Claude Desktop. See [`MCP.md`](MCP.md).
* **Single INI config — `/etc/easyai/easyai.ini`.** Every CLI flag
  has an INI key (FlagDef table refactor); precedence is CLI > INI
  > hardcoded default. Edit the file, `systemctl restart`, done.
  Full reference in [`easyai-server.md`](easyai-server.md) §1.
* **RAG: persistent memory.** Six tools (`rag_save`, `rag_search`,
  `rag_load`, `rag_list`, `rag_delete`, `rag_keywords`).
  Multi-keyword search with adaptive threshold + pagination. One
  Markdown file per entry — operator-readable, hand-editable. See
  [`RAG.md`](RAG.md).

### 2026-04-29 — External tools v2

* **Operator-defined tool packs** via `EASYAI-<name>.tools` JSON
  manifests dropped in `/etc/easyai/external-tools/`. Per-file
  fault isolation, sanity warnings (shell-wrapper detection,
  world-writable binaries, `LD_*` env passthrough), full
  `fork`+`execve` hardening — never a shell. Give the model
  focused powers without flipping `--allow-bash`. See
  [`EXTERNAL_TOOLS.md`](EXTERNAL_TOOLS.md).
* **`get_current_dir` builtin** — the model can ask where it is,
  so relative paths in `bash` / `fs_*` calls land where you expect.
* **Cancel-on-disconnect on the server** — closing the browser
  tab actually stops the decode loop. No more zombie generation
  eating tokens after the user walked away.
* **Tolerant tool output** — non-UTF-8 bytes in tool results no
  longer abort the SSE stream; the bytes get a U+FFFD substitute
  and the stream stays alive.

---

## Why try it

**Your assistant. Your tools. Your hardware. No cloud subscription,
no API bill, no data leaving the box.**

* **Runs on a Raspberry Pi.** Bonsai 8B Q1_0 weighs in at ~1.2 GB
  resident. A Pi 4 (8 GB) or any Pi 5 holds it with a 4 K context
  comfortably — and one install script puts a chat server at
  `http://pi-ai.local` for everyone on your home network.

* **Runs on your Mac.** Same one-script flow, Metal on Apple
  Silicon, full webui at `http://localhost:8080`. No Docker, no
  Conda, no Python venv. Uninstall is `rm -rf` of the checkout.

* **Plugs into the AI apps you already use.** OpenAI-compatible
  (`/v1/chat/completions`) — Claude Code, the OpenAI SDK,
  LiteLLM, LangChain, LobeChat, OpenWebUI all point at it without
  any easyai-specific configuration. Ollama-compat shims
  (`/api/tags`, `/api/show`) cover clients that prefer that shape.

* **Speaks MCP.** Claude Desktop, Cursor, Continue and any other
  Model Context Protocol client auto-discovers the tool catalogue.
  **Write one tool — every AI app on your machine can call it.**

* **Long-term memory built in.** RAG: six tools the agent uses to
  save, search, load, list, delete, and inventory its own
  knowledge. One human-readable Markdown file per entry — `cat`,
  `vim`, `grep` it. No vector DB to babysit.

* **Operator-defined tool packs.** Drop a JSON manifest in
  `/etc/easyai/external-tools/`, the agent picks it up at startup.
  Give the model exactly the powers it needs (a database probe, a
  deploy command, a metrics query) without ever flipping
  `--allow-bash`.

* **Safe defaults.** No filesystem, no shell, no writes — until
  you opt in. Every privileged opt-in is logged at startup with
  sanity warnings (shell wrappers, world-writable binaries,
  dynamic-linker env passthrough). Three rounds of security
  audits in [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md).

* **A C++17 framework, not a wrapper.** Three lines wrap
  llama.cpp into a real agent. Fluent builder for tools, full
  sampling control, streaming callbacks, plan tool, named
  sampling presets. Link `libeasyai`, ship one binary.

* **Ops-ready.** Prometheus `/metrics`, Bearer auth, systemd unit
  with `mlock` + `LimitMEMLOCK=infinity`, flash-attn, KV-cache
  quantisation (`q8_0` / `q4_0` / `iq4_nl`), per-request body
  cap, slow-loris timeouts. The Linux installer handles the whole
  Debian/Ubuntu deploy in one command.

### Get going in 60 seconds

```sh
# Raspberry Pi 4 / Pi 5 (Pi OS 64-bit) — your LAN's AI appliance:
git clone https://github.com/solariun/easy && cd easy
sudo ./scripts/install_easyai_pi.sh
# → http://pi-ai.local on every device on your network

# Mac (Apple Silicon or Intel):
git clone https://github.com/solariun/easy && cd easy
./scripts/install_easyai_macos.sh
# → http://localhost:8080

# Linux server (Debian / Ubuntu):
git clone https://github.com/solariun/easy && cd easy
sudo ./scripts/install_easyai_server.sh --model /path/to/your.gguf
# → http://0.0.0.0:80 with full systemd + auth + Prometheus metrics
```

Then open the URL in any browser, or point your favourite OpenAI
client at the same address. That's it.

---

## At a glance

The pitch in three lines:

```cpp
#include "easyai/easyai.hpp"

int main() {
    easyai::Agent a("models/qwen2.5-1.5b-instruct.gguf");
    std::cout << a.ask("What time is it in Tokyo right now?") << "\n";
}
```

That's the whole thing.  Construct an `Agent`, ask, print.  Default
toolset (datetime + web_search + web_fetch) is already wired in;
fs_* and bash stay off until you opt in.  Remote endpoints work the
same way:

```cpp
auto a = easyai::Agent::remote("http://127.0.0.1:8080/v1");
a.system("Be terse.")
 .on_token([](auto p){ std::cout << p << std::flush; });
a.ask("Summarise this commit.");
```

When you outgrow the 3-line shape, the same library exposes every
layer below — Tier 2 fluent builders (`Toolbelt`, `Streaming`),
Tier 3 explicit composables (`Engine`, `Client`, `Backend`,
`Tool::builder`), Tier 4 raw escape hatches (`Agent::backend()`,
llama.cpp handles).  Higher tiers are implemented on top of lower
ones — no parallel codepaths — so Tier 1 stays trustworthy as the
project evolves.

```cpp
// Tier 2 example: wire the canonical toolset onto an Engine in
// three fluent lines instead of seven add_tool calls.
easyai::Engine engine;
engine.model("models/qwen2.5-1.5b-instruct.gguf").gpu_layers(99).context(4096);

easyai::cli::Toolbelt()
    .sandbox   ("/srv/data")    // enables fs_read_file/list_dir/glob/grep/write
    .allow_bash()                // enables bash + bumps max_tool_hops to 99999
    .apply     (engine);

engine.load();
engine.chat("Find all .md files larger than 1 KB and summarise them.");
```

`Engine::chat()` runs the **full** tool-call/tool-result loop for you — up to
8 hops by default (lift the cap with `engine.max_tool_hops(N)` for shell-driven
flows, or just register `bash` and the helpers do it for you).

Tool definitions are 6 lines:

```cpp
engine.add_tool(
    easyai::Tool::builder("flip_coin")
        .describe("Returns 'heads' or 'tails' uniformly at random.")
        .handle([](const easyai::ToolCall &) {
            return easyai::ToolResult::ok((std::rand() & 1) ? "heads" : "tails");
        })
        .build());
```

---

## What's in the box

### Library (`libeasyai`, link target `easyai::engine`)

* `easyai::Engine` — high-level wrapper around llama.cpp's model + context +
  sampler + chat templates. Fluent setters, RAII-owned native resources.
* `easyai::Tool` — name + description + JSON-schema params + handler. Builder
  API generates the schema for you.
* `easyai::Plan` — agent-friendly checklist with one multi-action tool
  (add / start / done / list).  Pluggable into `Engine` or `Client`; fires a
  callback on every mutation so you can render live.
* `easyai::tools::*` — built-in tools:
  * `datetime` (no deps)
  * `web_fetch` (libcurl, HTML→text)
  * `web_search` (DuckDuckGo HTML, no API key, no external service)
  * `fs_read_file`, `fs_write_file`, `fs_list_dir`, `fs_glob`, `fs_grep`
    — sandboxed to a root directory you provide; the model sees a virtual
    `/`-rooted filesystem (real sandbox path is hidden).
  * `bash` — shell command runner. `/bin/sh -c`, cwd pinned to the
    sandbox root, stdout/stderr merged + capped, configurable timeout.
    Honest about what it is: NOT a hardened sandbox — runs with your
    user privileges. Opt-in.
* `easyai::presets` — named sampling profiles
  (`deterministic / precise / balanced / creative / wild`) plus a tiny parser
  that turns chat lines like `"creative 0.9"` or `"/temp 0.5"` into sampling
  overrides.
* `easyai::ui` — terminal UI helpers (`Style`, `Spinner`, `StreamStats`).
  Auto-detect TTY, honour `NO_COLOR`, heartbeat-driven spinner so the
  glyph keeps animating during long tool calls.
* `easyai::text` — small string helpers (`punctuate_think_tags`,
  `slurp_file`, `prompt_wants_file_write` heuristic).
* `easyai::log` — `set_file(FILE*)` + `write(fmt, ...)`: tee diagnostic
  output to stderr **and** an optional log file.
* `easyai::cli` — CLI infrastructure:
  * `Toolbelt` — fluent builder that registers the canonical agent
    toolset on an `Engine` or `Client` and bumps `max_tool_hops` to
    99999 when bash is enabled.
  * `open_log_tee / close_log_tee` — open `/tmp/<prefix>-<pid>-<epoch>.log`
    with header, register as the global log sink.
  * `validate_sandbox(path, &err)` — uniform "exists? is a dir?" check.
  * `client_has_tool(client, name)`, `print_models / print_local_tools /
    print_remote_tools / print_health / print_props / print_metrics /
    set_preset` — management subcommand helpers that drive an
    `easyai-server` from a one-line dispatcher.
* `easyai::Backend` (+ `LocalBackend`, `RemoteBackend`) — common
  interface for "give me a model, local or remote, with the same
  chat/reset/set_system shape".  Linking only `easyai::engine` gets
  you LocalBackend; adding `easyai::cli` adds RemoteBackend without
  duplicating the abstraction.
* `easyai::Agent` — the friendly Tier-1 façade over Backend.  3-line
  hello-world, fluent setters for system/sandbox/allow_bash/preset,
  and `backend()` as the escape hatch back to Tier 3 power.

### Library (`libeasyai-cli`, link target `easyai::cli`)

* `easyai::Client` — same fluent API shape as `Engine`, but the model runs on
  a remote `/v1/chat/completions` endpoint and **tools execute locally**.
  Configures HTTP transport (`endpoint`, `api_key`, `timeout_seconds`,
  `verbose`) plus the full sampling/penalty surface (`temperature`, `top_p`,
  `top_k`, `min_p`, `repeat_penalty`, `frequency_penalty`, `presence_penalty`,
  `seed`, `max_tokens`, `stop(vector)`, `extra_body_json`).  Streaming
  callbacks (`on_token`, `on_reason`, `on_tool`) and an agentic multi-hop
  loop mirror `Engine::chat_continue` semantics.
* Direct-endpoint helpers — `list_models`, `list_remote_tools`, `health`,
  `metrics`, `props`, `set_preset` — let downstream apps script and
  introspect an `easyai-server` without ever touching curl.

### Binaries

#### Tool gating across all three CLIs

All three example CLIs (`easyai-local`, `easyai-cli`, `easyai-server`)
follow the same gating model. Default is **safe**: no filesystem access,
no shell.

| Flag                  | What it enables                                                         |
|-----------------------|-------------------------------------------------------------------------|
| (no flag)             | `datetime`, `web_search`, `web_fetch` only.                             |
| `--sandbox <dir>`     | `fs_read_file / fs_write_file / fs_list_dir / fs_glob / fs_grep` plus `get_current_dir`, ALL scoped to `<dir>`. The CLIs `chdir` into `<dir>` so `get_current_dir` reports the sandbox path back to the model. |
| `--allow-bash`        | `bash` (run `/bin/sh -c`). cwd = `--sandbox <dir>` if given, otherwise the binary's CWD. NOT a hardened sandbox — runs with your user privileges. Also bumps the agentic-loop `max_tool_hops` to 99999 (bash flows naturally span many turns). |
| `--use-google`        | `web_google` (Google Custom Search JSON API). Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` env vars. Counts against your Google quota — free tier is 100 queries/day per key. Silently skipped if either env var is missing. |
| `--external-tools <dir>` | Load every `EASYAI-<name>.tools` file in `<dir>` as an operator-defined tool pack. Per-file fault isolation (a bad file is logged + skipped, the agent still starts). Spawns via `fork`+`execve` — never a shell. **This is the supported way to give the model focused powers without flipping `--allow-bash`.** See [`EXTERNAL_TOOLS.md`](EXTERNAL_TOOLS.md). |
| `--RAG <dir>`         | Enable RAG, the agent's persistent **memory** (search / store / recall / update / forget). Six tools by default — `rag_save`, `rag_search`, `rag_load`, `rag_list`, `rag_delete`, `rag_keywords` — each memory one Markdown file in `<dir>`. Memories whose title starts with `fix-easyai-` are immutable: pass `fix=true` to `rag_save` to mint one. The systemd-installed server passes this by default (`/var/lib/easyai/rag`). See [`RAG.md`](RAG.md). |
| `--experimental-rag`  | Replace the six `rag_*` tools with a single `rag(action=...)` dispatcher. Same on-disk format, same fix-memory semantics — only the catalog shape changes. Smaller catalog (1 entry vs 6) at the cost of accuracy on small / 1-bit-quant tool callers. Has no effect when `--RAG` is also off. |
| `--mcp <url>`         | Connect to a remote MCP server as a CLIENT (e.g. another `easyai-server` or `easyai-mcp-server`). The upstream's tool catalogue is fetched via `tools/list` and merged into the local one; each remote tool's handler proxies `tools/call` back to it. Local tool names win on collision (remote dup skipped with a warning). Pair with `--mcp-token <token>` when the upstream requires bearer auth. |
| `--no-local-tools`    | Skip the LOCAL built-in toolbelt entirely (datetime, web_*, fs_*, bash, ...). Useful when you want ONLY external tools, ONLY RAG, or ONLY tools fetched via `--mcp`. Does NOT disable the MCP client — that's controlled by `--mcp`. **Renamed from `--no-tools`.** |

#### Single config file: `/etc/easyai/easyai.ini`

The systemd-installed server reads every operator-tunable knob —
host, port, alias, sandbox, RAG dir, KV cache types, mlock, flash-attn,
threads, MCP auth, the works — from one INI file. **CLI flags on the
unit override INI values; INI overrides hardcoded defaults.** So
tweak the file + restart, no `systemctl edit` cadence:

```ini
[SERVER]
host       = 0.0.0.0
port       = 80
alias      = EasyAi
mcp_auth   = on              ; require Bearer on /mcp

[ENGINE]
ngl        = -1              ; auto-fit GPU
flash_attn = on
mlock      = on
cache_type_k = q8_0
cache_type_v = q8_0

[MCP_USER]
gustavo    = REPLACE-WITH-OPENSSL-RAND-HEX-32
```

Full key reference + worked examples: [`easyai-server.md`](easyai-server.md) §1.

#### easyai-server speaks **MCP** — every tool also reachable from Claude Desktop / Cursor / Continue

`easyai-server` exposes its full tool catalogue (built-ins + RAG + every operator-defined `--external-tools` pack) via the **Model Context Protocol** at `POST /mcp`. Other AI applications connect, list, and dispatch:

```
Claude Desktop ──► [stdio bridge] ──► POST /mcp ──┐
Cursor          ─────────────────────► POST /mcp ──┤── easyai-server
Continue        ─────────────────────► POST /mcp ──┘   (one tool catalogue,
                                                        many consumers)
```

You build the tools once. Your RAG, your deploy CLI, your monitoring queries — written ONCE for your easyai-server — become available in every AI app you already use. No plugin per app.

```sh
# List tools the server is exposing right now
curl -fsS http://localhost/mcp -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | jq '.result.tools[] | .name'
```

It also speaks the **OpenAI** (`/v1/chat/completions`, `/v1/models`) and **Ollama** (`/api/tags`, `/api/show`) list-models APIs so OpenAI-SDK, LangChain, LiteLLM, LobeChat, OpenWebUI, etc. auto-discover the loaded model and chat without any easyai-specific configuration.

Full guide: [`MCP.md`](MCP.md). Bridge script for Claude Desktop: `scripts/mcp-stdio-bridge.py`.

#### Why `--RAG` makes the agent useful

Without long-term memory, every session starts from zero: the model
re-derives your preferences, re-learns your project, re-asks the same
questions. With `--RAG`, the model decides what's worth remembering
and writes it to a directory of small Markdown files. Next session,
it `rag_search`es by keyword, finds what its past self saved, and
picks up where you left off.

```
> "I prefer terse responses in PT-BR."
[model: rag_save("user-prefs", ["user","prefs","locale"], "...")]

[next session]
> "build easyai on the AI box"
[model: rag_search(["easyai"]) → finds your saved build recipe]
[model loads it and answers in your style]
```

The dir is at `/var/lib/easyai/rag/` on the installed server. You
can `cat`, `vim`, `grep`, hand-author entries, back it up with `tar`
— it's a directory of plain text files. The model is the curator;
you, the operator, can read and edit anything it decided to keep.

Future evolution (see `RAG.md`): progressive recall on session start,
automatic document ingestion, per-user namespaces. The on-disk format
won't change.

#### Why `--external-tools` is the answer to "give the model more power"

Most agent frameworks force a binary choice: either you ship the model
with the tools the framework's authors thought of, or you give it a
generic shell. The framework's authors don't know about your internal
deploy CLI, your jq wrappers, your monitoring queries — and a generic
shell is a structurally unsafe surface no matter how careful you are.

easyai's `--external-tools` is the missing third option. Drop a JSON
file in the configured directory:

```json
{
  "version": 1,
  "tools": [
    {
      "name": "deploy_status",
      "description": "Status of one of our services in the control plane.",
      "command": "/opt/internal/bin/deploy-cli",
      "argv": ["status", "--", "{service}"],
      "parameters": {
        "type": "object",
        "properties": { "service": {"type":"string"} },
        "required": ["service"]
      },
      "timeout_ms": 10000,
      "max_output_bytes": 32768,
      "cwd": "$SANDBOX",
      "env_passthrough": ["DEPLOY_TOKEN"]
    }
  ]
}
```

Restart the server. The model can now ask for `deploy_status(service:"billing-api")`. The framework guarantees:

- **No shell.** `fork` + `execve` directly. The model's argument fills exactly one argv slot — `; rm -rf /` cannot escape it.
- **No PATH-hijack.** Absolute command paths are mandatory and validated at load.
- **No quoting bugs.** Whole-element placeholders only; `--flag={x}` is rejected at load (split into `["--flag","{x}"]`).
- **Schema-validated arguments.** Type errors rejected before `fork()`.
- **Bounded resources.** Timeout, output size, env-var inheritance, fd inheritance — every channel capped.
- **Per-file fault isolation.** A typo in `EASYAI-experimental.tools` doesn't prevent `EASYAI-system.tools` from loading.
- **Operator/user collaboration.** Drop additional `EASYAI-*.tools` files in the dir and they appear after a restart. Different teams can own different files. `chmod o-w` enforced at the directory level.
- **Sanity-check warnings at load.** Wrap a shell? Let the model influence `LD_PRELOAD`? Manifest world-writable? You'll see it in the startup log.

The default install creates `/etc/easyai/external-tools/` empty — drop your first `.tools` file in and you're live. Full guide and ten worked recipes in [`EXTERNAL_TOOLS.md`](EXTERNAL_TOOLS.md).

#### `easyai-local`

```
easyai-local -m model.gguf [-s system.txt] [--ngl 99] [--no-tools]
              [--sandbox DIR] [--allow-bash]
```

Local-only REPL.  Type any line to talk; type any of these to control the engine:

| Command                | Effect                                                                |
|------------------------|-----------------------------------------------------------------------|
| `precise`              | Switch to the `precise` preset                                        |
| `creative 0.9`         | Switch to `creative`, override temperature to 0.9                     |
| `/temp 0.5`            | Set temperature only                                                  |
| `/system <text>`       | Replace system prompt and clear history                               |
| `/reset`               | Clear conversation history                                            |
| `/tools`               | List currently-registered tools                                       |
| `/help`                | Show all presets                                                      |
| `/quit`                | Leave                                                                 |

Loads a `system.txt` if you pass `-s`; this is the *server-default* system
prompt (in the CLI's case, just the system prompt for that REPL session).

#### `easyai-server`

```
easyai-server -m model.gguf [-s system.txt] [--port 8080] [--ngl 99]
              [--sandbox DIR] [--allow-bash]
```

OpenAI-compatible HTTP server. Endpoints:

| Verb | Path                       | Notes                                                                                         |
|------|----------------------------|-----------------------------------------------------------------------------------------------|
| GET  | `/`                        | Embedded single-file webui (chat + preset bar)                                                |
| GET  | `/health`                  | JSON status (model, backend, tool count, ambient preset)                                      |
| GET  | `/v1/models`               | Lists the loaded model in OpenAI format                                                       |
| POST | `/v1/chat/completions`     | OpenAI-shape request, including optional `tools`, `temperature`, `top_p`, `top_k` overrides   |
| POST | `/v1/preset`               | `{"preset":"creative"}` — change the ambient preset for the webui                             |

**The killer feature** — when a *client* (Claude Code, an OpenAI SDK, LiteLLM,
LangChain…) posts its **own** `system` message and/or **own** `tools` to
`/v1/chat/completions`, those win for that single request:

* Client provides `tools` → easyai forwards generated tool calls back to the
  client and *does not* dispatch them locally. The client controls the loop.
* Client provides no `tools` → easyai uses its own toolbelt and runs the
  multi-hop loop server-side, returning the final assistant message.

Either way the server-supplied `system.txt` is used **only** when the request
doesn't already include a `system` message.

This makes `easyai-server` look like a real OpenAI-compatible backend to any
client that expects one.

---

## Meet Deep — the default assistant persona

A fresh `easyai-server` boots up as **Deep** — an expert system
engineer who answers from CHECKED FACTS, not impressions.  Built into
the default system prompt so a small open-weights model behaves like
an engineer instead of a chatbot from minute one.

Deep's operating loop is: **TIME → THINK → PLAN → EXECUTE → VERIFY**.

- **Time first.** Any question that touches "now", "today", a
  deadline, a release version, or a fact that could have changed
  since training cutoff → `datetime` is the first tool call.  Anchors
  the rest of the turn to the real wall clock.
- **Think.** State the goal, identify what's known vs. needs lookup,
  what could go wrong.
- **Plan.** Multi-step tasks call `plan(action='add', text=…)` first
  so the user can see and intervene live.
- **Execute.** Every registered tool is fair game.
- **Verify.** Before claiming success — does the file exist? does
  the test pass? does the URL really say that?  When in doubt, run
  another tool instead of guessing.

Old behaviour rules carry over: `RULE 1` (execute or answer, never
just announce), `web_search → web_fetch` mandatory, citations stick
to the URL actually fetched.

Operators who want a different persona pass `--system "<text>"` or
`-s persona.txt` — Deep is the default, not a hardcoded identity.

---

## Quick start

```
develop/
├── easyai/        # this project
└── llama.cpp/     # cloned next to it (https://github.com/ggml-org/llama.cpp)
```

```bash
cd easyai
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # see "Build for your hardware" below
cmake --build build -j

# Local REPL with everything wired up
./build/easyai-local -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Agentic REPL talking to a remote OpenAI-compatible endpoint
./build/easyai-cli --url http://127.0.0.1:8080
./build/easyai-cli --url https://api.openai.com/v1 \
                   --api-key $OPENAI_API_KEY --model gpt-4o-mini

# One-shot mode (great in scripts — banners on stderr, model text on stdout)
./build/easyai-local -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p "What is 2+2?"
result=$(./build/easyai-cli --url http://127.0.0.1:8080 --no-reasoning -p "summarise this commit")

# Open http://127.0.0.1:8080 in a browser
./build/easyai-server -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Point any OpenAI client at it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"easyai","messages":[{"role":"user","content":"Hi!"}]}'
```

For Claude Code (or any tool that takes an OpenAI-compatible base URL), set
`http://127.0.0.1:8080/v1` as the base. Any tools the client declares will
be forwarded; any tools it doesn't declare will use the server's toolbelt.

### Selective builds — only what you need

Every target is independent.  Configure once, then build whichever
subset matters for your situation:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release           # configure once

# Just the engine library (libeasyai.so + headers):
cmake --build build -j --target easyai

# Just the OpenAI-protocol client library (libeasyai-cli.so):
cmake --build build -j --target easyai_cli

# Just the agentic remote CLI (links libeasyai-cli):
cmake --build build -j --target easyai-cli

# Just the local-only REPL (links libeasyai):
cmake --build build -j --target easyai-local

# Just the server:
cmake --build build -j --target easyai-server

# Drop the examples entirely (lib-only consumers):
cmake -S . -B build -DEASYAI_BUILD_EXAMPLES=OFF
cmake --build build -j

# Drop the embedded webui from easyai-server (smaller binary):
cmake -S . -B build -DEASYAI_BUILD_WEBUI=OFF
cmake --build build -j

# Drop libcurl-using tools (web_fetch / web_search):
cmake -S . -B build -DEASYAI_WITH_CURL=OFF
cmake --build build -j

# Clean rebuild from scratch:
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Just delete object files but keep configuration:
cmake --build build --target clean
```

After `cmake --install build --prefix /usr/local`, downstream projects
can `find_package(easyai 0.1 REQUIRED)`:

```cmake
# Your CMakeLists.txt:
find_package(easyai 0.1 REQUIRED)
add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE
    easyai::engine    # libeasyai.so — local llama.cpp wrapper
    easyai::cli       # libeasyai-cli.so — OpenAI-protocol client
)
```

Both targets export their public include directory and `cxx_std_17`
feature, so consumers don't need any extra include flags.

### Build for your hardware

Pick the matching configure command for your machine; rebuild with
`cmake --build build -j`.

| Hardware                                  | Configure command                                                                  | Notes                                                                                |
|-------------------------------------------|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| **Apple Silicon / Intel Mac (Metal)**     | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`                                   | Metal is auto-detected on macOS — nothing extra to set.                              |
| **NVIDIA GPU (CUDA)**                     | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON`                    | Needs the CUDA Toolkit (`nvcc`). Optionally pin GPU arch with `-DCMAKE_CUDA_ARCHITECTURES=89`. |
| **AMD / Intel / cross-vendor (Vulkan)**   | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON`                  | Needs the Vulkan SDK on Linux/Windows. Works on AMD RX/Pro, Intel Arc, NVIDIA too.   |
| **AMD on Linux (ROCm/HIP)**               | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100` | Replace the gfx ID with your card's. Requires ROCm 6+.                                |
| **CPU-only (any OS)**                     | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`                                   | Then run with `-ngl 0` (CLI/server) or `.gpu_layers(0)` (lib).                       |

Add `-DGGML_OPENBLAS=ON` (Linux) or `-DGGML_BLAS=ON` (macOS uses Accelerate
automatically) for a faster CPU prompt-eval path.

If both Metal and CUDA libraries are present (rare), keep one and disable
the other explicitly with `-DGGML_METAL=OFF` / `-DGGML_CUDA=OFF`.

### Web search

`web_search` hits DuckDuckGo's HTML endpoint (`html.duckduckgo.com/html/`)
directly and parses the result page. No API key, no external service to
run, no environment variable to set. Works as long as the host has outbound
HTTPS to duckduckgo.com.

---

## Documentation

* [`manual.md`](manual.md) — hands-on developer manual.  Includes a
  step-by-step **"Recipe book — write your first tools"** chapter
  (section 3.8) that walks through `examples/recipes.cpp` line by
  line in a friendly, accessible style.  Best place to start if you
  want to extend easyai with your own services.
* [`design.md`](design.md) — architecture, data flow, why we build on top of
  `common/` instead of just `include/llama.h`.
* [`scripts/install_easyai_server.sh`](scripts/install_easyai_server.sh) —
  one-shot Debian/Ubuntu installer; **drop-in replacement** for the
  `install_llama_server.sh` workflow. Clones llama.cpp + easyai, builds
  with the right backend (auto-detects Vulkan / CUDA / ROCm / CPU),
  creates a system user + `/var/lib/easyai`, drops a hardened systemd
  unit with mlock + flash-attn + q8_0 KV cache + Bearer auth +
  Prometheus `/metrics`. Accepts every flag the original took
  (`--with-mcp`, `--draft-model`, `--webui-title`, etc.) — built-in
  features become no-ops with a friendly warning so existing automation
  keeps working.

---

## Memory hygiene

Every native resource is owned by a smart pointer or a value type with a
custom destructor:

* `Engine` — `std::unique_ptr<Impl>` pImpl pattern. The `Impl` destructor
  frees the sampler explicitly; the model, context, and chat-templates are
  unique-pointer-owned.
* `easyai-server` — single `std::unique_ptr<ServerCtx>` lives for the
  process lifetime. A `std::mutex` serialises the engine across httplib's
  worker threads.
* HTTP handlers cap request bodies at 8 MiB (configurable via `--max-body`)
  and catch every `std::exception` at the boundary so a malformed request
  cannot tear down the server.
* No raw `new`/`delete` anywhere in `src/` or `examples/`.

---

## License

Inherits the MIT license of llama.cpp. See `LICENSE`.
