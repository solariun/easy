# `easyai.ini` — central configuration reference

> **The single place to configure easyai-server.** Every CLI flag
> the binary accepts has a corresponding entry in this file. The
> systemd unit's `ExecStart` is short on purpose — operators tweak
> this file and restart, not the unit.

---

## File location

| Path | Notes |
| --- | --- |
| `/etc/easyai/easyai.ini` | Default — what the systemd unit reads via `--config`. |
| `<other path>` | Pass `--config /path/to.ini` on the binary's command line. |

The file is owned `root:easyai`, mode `640` — readable by the
service user, world-unreadable.

## Precedence

```
   CLI flag    >    INI value    >    hardcoded default
   (highest)         (this file)         (in the binary)
```

If the operator passed `--port 8080` in the systemd unit AND the
INI says `port = 9090`, the server listens on **8080**. Drop the
CLI flag and the INI value takes over.

## Sections

| Section | Purpose | Status |
| --- | --- | --- |
| `[SERVER]` | HTTP layer, paths, tool gating, MCP auth posture | active |
| `[ENGINE]` | Model loading + inference tunables | active |
| `[MCP_USER]` | Bearer-token auth for `/mcp` (one user per line) | active |
| `[TOOLS]` | Per-tool ACL (`mcp_allowed = …`) | reserved for a future release |

The TOOLS section is recognised by the parser but **not yet
consumed** by code. Populating it does no harm; the keys are
ignored until a future release wires them.

## Format quick reference

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

---

## `[SERVER]`

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
| `load_tools` | bool | `--no-tools` (negative) | `on` | Master switch for the built-in toolbelt. Set `off` (or pass `--no-tools`) to register zero default tools. `allow_fs` / `allow_bash` further opt in. |
| `max_body` | int | `--max-body` | `8388608` (8 MiB) | Max HTTP request body size. |
| `api_key` | string | `--api-key` | (none — `/v1/*` open) | Bearer token for `/v1/*`. Don't put real keys in INI directly — use `/etc/easyai/api_key` (file-based, the installer wires `${EASYAI_API_KEY}`). |
| `mcp_auth` | enum | (no CLI; `--no-mcp-auth` overrides) | `auto` | `auto` (auth iff `[MCP_USER]` non-empty), `on` (force require), `off` (force open). |
| `no_think` | bool | `--no-think` | `off` | Strip `<think>` tags from responses. |
| `inject_datetime` | bool | `--inject-datetime` | `on` | Authoritative date/time + knowledge-cutoff injection in the system prompt. |
| `knowledge_cutoff` | string | `--knowledge-cutoff` | `2024-10` | YYYY-MM hint for the model. |
| `reasoning` | bool | `--reasoning on/off` | `on` | Honour the model's reasoning channel. |

---

## `[ENGINE]`

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

---

## `[MCP_USER]`

Bearer-token authentication for `POST /mcp`. Each line registers
one user:

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
- The first matching token wins. Duplicate tokens (operator
  config bug) silently take the alphabetically-first user.
- Generate strong tokens with `openssl rand -hex 32` or
  `python3 -c 'import secrets; print(secrets.token_hex(32))'`.
- Restart the server to pick up changes:
  `sudo systemctl restart easyai-server`.

Full per-client connection guides (Cursor / Continue / Claude
Desktop / curl) live in `MCP.md` §4–7.

---

## `[TOOLS]` (RESERVED for a future release)

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

---

## Worked examples

### Minimal local-dev INI

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

### Production server with auth

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

(The installer additionally appends `-m <model>` and
`--api-key '${EASYAI_API_KEY}'` for runtime substitution from
`/etc/easyai/api_key`.)

### Override one value via CLI

Operator wants to test a different model without editing the INI:

```
ExecStart=/usr/bin/easyai-server --config /etc/easyai/easyai.ini -m /tmp/test-model.gguf
```

CLI `-m` overrides INI `model`. Everything else still comes from
the INI.

### Disable MCP auth temporarily

```
ExecStart=/usr/bin/easyai-server --config /etc/easyai/easyai.ini --no-mcp-auth
```

CLI flag wins over `[SERVER] mcp_auth = on` and over `[MCP_USER]`.

---

## Cross-references

- [`README.md`](README.md) — sales overview + quick start.
- [`LINUX_SERVER.md`](LINUX_SERVER.md) — full operator's guide for
  the systemd-installed server. Section 3 lists the config files.
- [`MCP.md`](MCP.md) — Model Context Protocol surface; §9 covers
  auth + per-client config.
- [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) §17 — security model
  for the MCP endpoint.
- [`EXTERNAL_TOOLS.md`](EXTERNAL_TOOLS.md) — operator-defined
  external tools (the `[SERVER] external_tools` path).
- [`RAG.md`](RAG.md) — persistent registry (the `[SERVER] rag`
  path).
