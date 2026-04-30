# easyai-server on Linux — operator's guide

This document is for the person who runs `install_easyai_server.sh`
on a Linux box and wants to know what landed where, how to configure
it, what to watch out for, and how to keep it healthy.

If you're a developer, see `design.md` and `manual.md`. If you're
writing tool manifests, see `EXTERNAL_TOOLS.md`. If you want to
understand the agent's long-term memory, see `RAG.md`. If you want
to expose easyai's tools to other AI applications (Claude Desktop,
Cursor, Continue), see `MCP.md`.

---

## Table of contents

1. [What gets installed where](#1-what-gets-installed-where)
2. [The systemd unit](#2-the-systemd-unit)
3. [Configuration files](#3-configuration-files)
4. [The four mutable directories](#4-the-four-mutable-directories)
5. [The `--external-tools` directory](#5-the---external-tools-directory)
6. [The RAG directory](#6-the-reg-directory)
7. [Performance tuning](#7-performance-tuning)
8. [Common gotchas](#8-common-gotchas)
9. [Hitting the API](#9-hitting-the-api)
10. [Health checks and verification](#10-health-checks-and-verification)
11. [Backup, restore, migration](#11-backup-restore-migration)
12. [Upgrading](#12-upgrading)
13. [Uninstalling](#13-uninstalling)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. What gets installed where

The installer (`scripts/install_easyai_server.sh`) follows the FHS
roughly:

| Path | Owned by | Mode | Purpose |
| --- | --- | --- | --- |
| `/usr/bin/easyai-server` | root:root | 755 | the binary |
| `/usr/bin/easyai-cli` | root:root | 755 | OpenAI-shape client (talks to a remote server) |
| `/usr/bin/easyai-local` | root:root | 755 | single-process REPL with a local model |
| `/usr/lib/easyai/` | root:root | 755 | bundled `.so` files (libllama, libggml, libeasyai, …) |
| `/etc/easyai/` | root:easyai | 750 | operator configuration |
| `/etc/easyai/system.txt` | root:easyai | 640 | system prompt |
| `/etc/easyai/api_key` | easyai:easyai | 600 | optional bearer-token gate |
| `/etc/easyai/external-tools/` | root:easyai | 750 | operator-defined tools (`EASYAI-*.tools`) |
| `/etc/easyai/favicon[.ext]` | root:easyai | 644 | optional webui favicon |
| `/var/lib/easyai/` | easyai:easyai | 750 | mutable agent state |
| `/var/lib/easyai/rag/` | easyai:easyai | 750 | RAG long-term memory |
| `/var/lib/easyai/workspace/` | easyai:easyai | 750 | sandbox for fs_* and bash tools |
| `/var/lib/easyai/models/` | easyai:easyai | 750 | the GGUF symlink target |
| `/etc/systemd/system/easyai-server.service` | root:root | 644 | the unit file |
| `/etc/systemd/system/easyai-server.service.d/override.conf` | root:root | 644 | LimitMEMLOCK / LimitCORE / Environment overrides |

The installer creates the `easyai` system user / group if missing.

---

## 2. The systemd unit

```bash
systemctl cat easyai-server
```

Roughly:

```ini
[Unit]
Description=easyai-server (llama.cpp + OpenAI shim)
After=network.target

[Service]
User=easyai
Group=easyai
ExecStart=/bin/sh -c '...EASYAI_API_KEY=$(cat /etc/easyai/api_key) ...
                      exec /usr/bin/easyai-server \
                          -m /var/lib/easyai/models/current.gguf \
                          --host 0.0.0.0 --port 80 \
                          --alias EasyAi \
                          -c 128000 \
                          --ngl -1 \
                          -t <jobs> -tb <jobs> \
                          --preset balanced \
                          --sandbox /var/lib/easyai/workspace \
                          --system-file /etc/easyai/system.txt \
                          --external-tools /etc/easyai/external-tools \
                          --RAG /var/lib/easyai/rag \
                          ... '
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
```

Important pieces:

- `User=easyai`. The agent runs unprivileged. `bash`, fs_*, every
  external tool inherits this uid. THE single biggest "isolation"
  you have. Don't run as root.
- `--sandbox /var/lib/easyai/workspace`. Where the agent's `bash` /
  `fs_*` tools land. The agent `chdir`s here at startup so
  `get_current_dir` reports this path.
- `--external-tools /etc/easyai/external-tools`. Operator-defined
  tools live here. Empty dir is a normal state.
- `--RAG /var/lib/easyai/rag`. The agent's persistent registry /
  long-term memory.
- `LimitMEMLOCK=infinity` (in the drop-in) so `mlock` works.
- `LimitCORE=infinity` (in the drop-in) so coredumps land for
  forensics.

### Drop-in for environment overrides

```bash
sudo systemctl edit easyai-server
```

Add `Environment=` lines. Common ones:

```
[Service]
Environment=RADV_PERFTEST=gpl
Environment=DEPLOY_TOKEN=xxx     # if your --external-tools needs it
```

Reload + restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart easyai-server
```

---

## 3. Configuration files

### `/etc/easyai/easyai.ini` — the central config

> **Full reference:** [`INI.md`](INI.md) lists every key the binary
> understands, what section it belongs to, what the CLI equivalent
> is, and gives worked examples.

All operator-tunable knobs live in one INI file. The systemd unit's
`ExecStart` is intentionally short — `--config /etc/easyai/easyai.ini`
plus the model path and the api-key plumbing — and **everything else**
(host, port, alias, sandbox, RAG dir, KV cache types, mlock, flash-attn,
threads, MCP auth, …) lives in this file.

Precedence: **CLI flag in the systemd unit > INI value > hardcoded
default in the binary.** So tweak this file for the normal case;
flip a CLI flag only for one-off overrides.

Sections:

| Section | Purpose | Status |
| --- | --- | --- |
| `[SERVER]` | HTTP layer + paths + tool gating + MCP auth posture | active |
| `[ENGINE]` | Model loading + inference tunables (context, ngl, KV, mlock, flash-attn, sampling) | active |
| `[MCP_USER]` | Bearer-token auth for `/mcp` (one user per line, `name = token`) | active |
| `[TOOLS]` | Per-tool ACL (`mcp_allowed = …, mcp_denied = …`) | reserved for future |

The installer drops a fully-populated `easyai.ini` (every key
documented inline). On `--upgrade` we **leave it alone** — your edits
win — so any keys we add in newer versions need to be merged
manually. We may at some point grow a polite `upsert`; for now the
release notes call out each new key.

To enable MCP Bearer auth: edit `[MCP_USER]`, uncomment one of the
example lines, replace the placeholder token with output from
`openssl rand -hex 32`. Restart the server. See `MCP.md` §9 for the
full guide and per-client config.

To temporarily reopen `/mcp` without editing the INI: pass
`--no-mcp-auth` on the systemd unit's `ExecStart` (or run the
binary by hand). The flag always wins.

### `/etc/easyai/system.txt`

The system prompt. Plain text. Edit with `sudo nano`. Restart the
server:

```bash
sudo systemctl restart easyai-server
```

The default the installer drops is short and tool-friendly. Customise
to add domain context, persona, language preferences. If you want
the model to use RAG aggressively, mention it here:

```
You have a persistent registry called RAG. Save important things
the user tells you (preferences, project facts, recipes that worked)
with rag_save. Search RAG with rag_search before assuming you don't
know something the user might have told you in a past session.
```

The installer also injects an authoritative date/time prefix at
runtime — see `design.md` §5c.

### `/etc/easyai/api_key`

If present, the server requires `Authorization: Bearer <key>` on
every request. Mode 600, owned by `easyai`.

```bash
echo -n "my-secret-token" | sudo tee /etc/easyai/api_key
sudo chown easyai:easyai /etc/easyai/api_key
sudo chmod 600 /etc/easyai/api_key
sudo systemctl restart easyai-server
```

To disable auth:

```bash
sudo rm /etc/easyai/api_key
sudo systemctl restart easyai-server
```

### `/etc/easyai/favicon[.ext]`

Optional webui favicon. The installer accepts `.ico`, `.png`, `.svg`,
`.gif`, `.jpg`, `.webp`. Re-run the installer with `--webui-icon` to
swap.

---

## 4. The four mutable directories

| Dir | What lives here | Watch out for |
| --- | --- | --- |
| `/var/lib/easyai/models/` | GGUF symlink target. The unit's `-m` arg points here. | Big files. Easy to fill the disk. |
| `/var/lib/easyai/workspace/` | The sandbox for `bash` / `fs_*` tools. | The agent reads / writes here. Keep it on a partition with room. |
| `/var/lib/easyai/rag/` | RAG long-term memory (one `.md` per entry). | Tiny. Backup-friendly. See `RAG.md`. |
| `/etc/easyai/external-tools/` | Operator-defined tools (`EASYAI-*.tools`). | Operator-curated. See `EXTERNAL_TOOLS.md`. |

---

## 5. The `--external-tools` directory

The installer creates `/etc/easyai/external-tools/` (mode 750,
root:easyai) and drops a README plus `EASYAI-example.tools.disabled`
in it. The systemd unit always passes `--external-tools` so a restart
picks up new files.

**To add a tool pack:**

```bash
sudo cp my-tools.tools /etc/easyai/external-tools/EASYAI-my-tools.tools
sudo chmod 640 /etc/easyai/external-tools/EASYAI-my-tools.tools
sudo chown root:easyai /etc/easyai/external-tools/EASYAI-my-tools.tools
sudo systemctl restart easyai-server
sudo journalctl -u easyai-server -n 30 --no-pager | grep external-tools
```

**To disable without deleting:**

```bash
sudo mv /etc/easyai/external-tools/EASYAI-my-tools.tools{,.disabled}
sudo systemctl restart easyai-server
```

**Full reference:** `EXTERNAL_TOOLS.md`. Schema, recipes,
anti-patterns, troubleshooting, collaboration workflow.

---

## 6. The RAG directory

Active by default. The systemd unit always passes
`--RAG /var/lib/easyai/rag`. The agent writes here at runtime — that
is why it's under `/var/lib` (mutable state) rather than `/etc`
(operator config).

**Quick checks:**

```bash
# How many entries does the agent have?
ls /var/lib/easyai/rag/*.md 2>/dev/null | wc -l

# What did it save most recently?
ls -lt /var/lib/easyai/rag/*.md | head -10

# What's in a specific entry?
sudo -u easyai cat /var/lib/easyai/rag/<title>.md

# Search across all entries
sudo grep -l "user-prefs" /var/lib/easyai/rag/*.md
```

**Hand-author an entry:**

```bash
sudo -u easyai bash -c 'cat > /var/lib/easyai/rag/welcome.md' <<'EOF'
keywords: user-prefs, locale

The user prefers PT-BR responses with technical jargon in English.
Keep responses terse — code over explanation.
EOF
sudo systemctl restart easyai-server
```

**Full reference:** `RAG.md`. File format, the five tools, workflows,
roadmap, troubleshooting.

---

## 7. Performance tuning

### Context size

```
-c 128000
```

128k tokens covers most chat sessions without reaching the cap. Keep
in mind the KV cache scales linearly with context; see below.

### KV cache quantisation

The installer defaults to:

```
-ctk q8_0 -ctv q8_0
```

This halves the KV cache footprint at no measurable quality loss
on modern models. `-ctv q4_0` halves it again at a small perplexity
cost — try if you're VRAM-bound.

### Flash attention

```
-fa
```

On by default. Free perf on every backend that supports it (CUDA,
Metal, Vulkan ≥ recent). Required for some KV-quant combinations.

### GPU layers (`--ngl`)

```
--ngl -1
```

Auto-fit. The installer uses this. Manually pinning `--ngl 99` can
poison `common_fit_params` if it doesn't fit (it refuses to lower a
user-pinned value), so let auto-fit decide.

### `--mlock` and `--no-mmap`

The installer enables both. Pinning model weights in RAM prevents
the kernel paging them under memory pressure (catastrophic for token
latency). `--no-mmap` is needed for `--mlock` to be portable across
all GPU backends.

The drop-in's `LimitMEMLOCK=infinity` is what makes `mlock` actually
work (the systemd default is too small).

### `RADV_PERFTEST=gpl` (AMD Radeon)

The installer adds this `Environment=` line on AMD GPUs. Substantial
shader-compile speedup on RADV.

### CPU threads

```
-t <jobs> -tb <jobs>
```

The installer sets these to `nproc`. For a single-user agent box
that's right. For shared hardware, reduce so the agent doesn't
starve the rest of the system.

---

## 8. Common gotchas

### `mlock failed: Cannot allocate memory`

`LimitMEMLOCK` isn't `infinity`. Re-run the installer with `--upgrade`
to refresh the drop-in, or edit `systemctl edit easyai-server`
manually.

### `n_gpu_layers already set by user, abort`

Something pinned `--ngl` to a value that doesn't fit. Reset to
auto-fit (`--ngl -1`) and let llama.cpp decide.

### "model not found" on startup

The unit's `-m` arg points at a symlink that doesn't resolve. Check:

```bash
ls -la /var/lib/easyai/models/current.gguf
```

The installer accepts a `--model` flag to swap; or `ln -sfn` the
target manually.

### GTT exhaustion on AMD Radeon (Vulkan/RADV)

Symptom: large model load → GPU hangs → `journalctl -k` shows
`amdgpu: ttm pool full`.

Fix: increase the GTT page limit in your boot config (kernel param
`amdgpu.gttsize` or `ttm.pages_limit`). Gustavo's MINISFORUM UM690L
(Radeon 680M, 32 GB system RAM) uses `ttm.pages_limit=7340032` (28 GiB
GTT).

### Webui blank / shows yesterday's UI

Browser cache. **Cmd+Shift+R** (Linux: Ctrl+Shift+R) to hard-reload.
The bundle is hashed so a stale CSS file is the usual culprit.

### Coredump under load

Already wired by the installer:

- `systemd-coredump` package installed
- `LimitCORE=infinity` in the drop-in
- Coredumps land in `/var/lib/systemd/coredump/`

To examine:

```bash
coredumpctl list easyai-server.service
coredumpctl gdb <PID>
```

### "external-tools dir does not exist"

The path doesn't exist. Re-run installer with `--upgrade`, or:

```bash
sudo install -d -o root -g easyai -m 750 /etc/easyai/external-tools
sudo systemctl restart easyai-server
```

### Agent doesn't seem to remember anything

Either RAG isn't enabled or the dir is wrong:

```bash
journalctl -u easyai-server | grep "RAG enabled"
```

If absent, re-run installer or check `systemctl cat` for `--RAG`.

---

## 9. Hitting the API

The server speaks **three** API dialects so most AI clients work
unchanged. Endpoints:

| Verb | Path | API | Notes |
| --- | --- | --- | --- |
| GET | `/` | webui | embedded webui |
| GET | `/health` | easyai | `{model, backend, tools, preset, compat:{...}}` |
| GET | `/v1/models` | OpenAI | OpenAI-shape list-models |
| POST | `/v1/chat/completions` | OpenAI | the workhorse — streaming SSE, tools, sampling controls |
| POST | `/v1/preset` | easyai | swap the ambient preset |
| GET | `/v1/tools` | easyai | tool catalogue for the webui popover |
| GET | `/api/tags` | Ollama | Ollama-shape list-models (LobeChat, OpenWebUI in Ollama mode, etc.) |
| GET/POST | `/api/show` | Ollama | Ollama-shape model detail |
| POST | `/mcp` | MCP | JSON-RPC 2.0 — the full tool catalogue exposed to other AI apps. See `MCP.md`. |
| GET | `/mcp` | MCP | reserved for future SSE notifications; currently `405 Method Not Allowed`. |

### Bearer auth (if `/etc/easyai/api_key` present)

```bash
curl -H "Authorization: Bearer $(sudo cat /etc/easyai/api_key)" \
     http://localhost/health
```

### One-shot completion via curl

```bash
curl -sN \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -X POST http://localhost/v1/chat/completions \
  -d '{
    "model": "easyai",
    "messages": [
      {"role": "user", "content": "Show me last week’s deploys."}
    ],
    "stream": true
  }'
```

The server streams SSE deltas. Tools fire automatically — no special
client setup required.

### Calling from the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost/v1",
    api_key="my-token",
)

resp = client.chat.completions.create(
    model="easyai",
    messages=[{"role": "user", "content": "What's our deploy status?"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

The model dispatches whatever tools the operator declared on the
server side (built-ins + `--external-tools` + RAG).

### `X-Easyai-Inject: off` to skip date/time injection

Useful for A/B regression suites:

```bash
curl -H "X-Easyai-Inject: off" ...
```

---

## 10. Health checks and verification

### Server alive?

```bash
curl -fsS http://localhost/health | jq .
```

```json
{
  "status": "ok",
  "model": "easyai",
  "backend": "cuda|metal|vulkan|cpu",
  "tool_count": 12,
  "preset": "balanced"
}
```

### Tools registered?

```bash
curl -fsS http://localhost/health | jq .tool_count
```

Expected (rough):

- 4 (datetime, web_search, web_fetch, plan)
- + 5 (RAG: rag_save / search / load / list / delete)
- + 6 (`--allow-fs`: read_file / write_file / list_dir / glob / grep / get_current_dir)
- + 1 (`--allow-bash`: bash)
- + N (your `--external-tools` packs)

### RAG working?

```bash
ls /var/lib/easyai/rag/
journalctl -u easyai-server | grep "RAG enabled"
```

### External tools loaded?

```bash
journalctl -u easyai-server | grep external-tools
```

Expected:

```
easyai-server: loaded N external tool(s) from M file(s) in /etc/easyai/external-tools
```

If a file failed to parse:

```
easyai-server: [external-tools] error: /etc/easyai/external-tools/EASYAI-foo.tools: ...
```

---

## 11. Backup, restore, migration

### What to back up

| Path | Frequency | Why |
| --- | --- | --- |
| `/etc/easyai/` | on change | system prompt, api key, external tools |
| `/var/lib/easyai/rag/` | regular | the agent's accumulated knowledge |
| `/var/lib/easyai/workspace/` | maybe | depends what you let the agent write here |
| `/var/lib/easyai/models/` | usually no | GGUFs are big and re-downloadable |

### One-shot snapshot

```bash
sudo tar -czf easyai-backup-$(date +%F).tar.gz \
    /etc/easyai \
    /var/lib/easyai/rag \
    /var/lib/easyai/workspace \
    /etc/systemd/system/easyai-server.service.d
```

### Restoring on a fresh install

1. Run `install_easyai_server.sh` on the new host (creates user, dirs,
   unit).
2. `sudo systemctl stop easyai-server`.
3. Untar over `/`:
   ```bash
   sudo tar -xzpf easyai-backup-*.tar.gz -C /
   ```
4. Fix ownership:
   ```bash
   sudo chown -R easyai:easyai /var/lib/easyai/rag /var/lib/easyai/workspace
   ```
5. `sudo systemctl start easyai-server`.

---

## 12. Upgrading

```bash
cd ~/easy
git pull
./scripts/install_easyai_server.sh --upgrade --enable-now
```

`--upgrade`:

- Refreshes `/usr/bin/easyai-*` and `/usr/lib/easyai/`
- Re-renders the systemd unit (so flag changes propagate)
- Does NOT touch `/etc/easyai/system.txt`, `/etc/easyai/api_key`,
  `/var/lib/easyai/rag/*` — your data is safe
- Does NOT touch `/etc/easyai/external-tools/*` — your manifests
  are safe
- WILL refresh the README in `external-tools/` and the
  `EASYAI-example.tools.disabled` sample

`--enable-now`:

- `systemctl enable easyai-server` (auto-start on boot)
- `systemctl start easyai-server`

After upgrading, sanity-check:

```bash
journalctl -u easyai-server -n 50 --no-pager | grep -E "RAG enabled|external-tools|loaded"
```

---

## 13. Uninstalling

The installer doesn't ship an uninstaller (yet). Manual:

```bash
sudo systemctl disable --now easyai-server
sudo rm /etc/systemd/system/easyai-server.service
sudo rm -rf /etc/systemd/system/easyai-server.service.d
sudo systemctl daemon-reload

sudo rm -f /usr/bin/easyai-server /usr/bin/easyai-cli /usr/bin/easyai-local
sudo rm -rf /usr/lib/easyai/

# data — back up first if you might want it back
sudo rm -rf /var/lib/easyai/

# config — same
sudo rm -rf /etc/easyai/

# user
sudo userdel easyai
```

---

## 14. Troubleshooting

### Server won't start, journal shows "model not found"

```bash
ls -la /var/lib/easyai/models/
```

The unit's `-m` arg points at a symlink. If broken, re-symlink:

```bash
sudo -u easyai ln -sfn /path/to/your.gguf /var/lib/easyai/models/current.gguf
sudo systemctl restart easyai-server
```

### "Permission denied" writing to `--sandbox`

The agent runs as `easyai`. The dir must be `easyai`-writable:

```bash
sudo chown -R easyai:easyai /var/lib/easyai/workspace
sudo chmod 750 /var/lib/easyai/workspace
```

### High RSS / OOM-killer killing easyai-server

Likely the model is mmap'd but pinned, or the KV cache is huge. Check:

```bash
ps -o rss,cmd -p $(systemctl show -p MainPID --value easyai-server)
```

If RSS approaches your physical RAM, drop `-ctv q8_0` to `-ctv q4_0`,
or shrink `-c 128000` to something smaller.

### Webui seems to lose connection mid-stream

`X-Accel-Buffering: no` is required if you have nginx in front of
easyai-server. The server sets it on streams, but a misconfigured
proxy can override.

### External tool calls hang forever

The tool's `timeout_ms` is too high (cap is 5 min). Edit the
`.tools` file, lower `timeout_ms`, restart the server.

### RAG entries appear duplicated

Two things to check:

1. Is the same dir mounted twice? `mount | grep var/lib/easyai`.
2. Are two easyai-server processes running? `pgrep -af easyai-server`
   should return exactly one.

### Coredump

```bash
coredumpctl list easyai-server.service
coredumpctl gdb <PID>
```

In gdb: `bt`, `bt full`, `info threads`, `thread apply all bt`. File
the bug at the easyai issue tracker with the trace.

### Verbose logs

Re-run installer with `--enable-verbose`, OR transient:

```bash
sudo systemctl edit --runtime easyai-server
# Add:
[Service]
Environment=EASYAI_VERBOSE=1
```

Then `daemon-reload && restart`. Drop the override when done.

---

*See also:* `RAG.md`, `EXTERNAL_TOOLS.md`, `manual.md`, `design.md`,
`SECURITY_AUDIT.md`.
