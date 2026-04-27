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
| `easyai-cli`         | Drop-in `llama-cli` replacement: local **or** remote OpenAI-compatible endpoint (HTTPS via OpenSSL), one-shot scripting (`-p`), tools, presets, optional `<think>` strip. `--with-tools` opts in to local tool dispatch over a remote model (agentic). `--insecure-tls` / `--ca-cert` for dev/internal CAs. |
| `easyai-cli-remote`  | Pure agentic CLI built on `libeasyai-cli` — no local model required.  REPL or `-p`, full sampling control (`--temperature`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`, `--frequency-penalty`, `--presence-penalty`, `--seed`, `--max-tokens`, `--stop`), plan tool, server-management subcommands (`--list-models`, `--list-tools`, `--health`, `--props`, `--metrics`, `--set-preset`). |
| `easyai-server`      | Drop-in `llama-server` replacement: OpenAI-compat HTTP **with full SSE streaming**, embedded SvelteKit webui, Bearer auth, Prometheus `/metrics`, KV-cache controls, flash-attn, mlock. |
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

## At a glance

```cpp
#include "easyai/easyai.hpp"

int main() {
    easyai::Engine engine;
    engine.model("models/qwen2.5-1.5b-instruct.gguf")
          .gpu_layers(99)              // Metal on macOS, Vulkan elsewhere
          .context(4096)
          .system("You are a helpful agent.")
          .add_tool(easyai::tools::datetime())
          .add_tool(easyai::tools::web_search())
          .add_tool(easyai::tools::web_fetch())
          .add_tool(easyai::tools::fs_read_file("."))
          .on_token([](const std::string & t){ std::cout << t << std::flush; })
          .load();

    engine.chat("What time is it in Tokyo right now?");
}
```

`Engine::chat()` runs the **full** tool-call/tool-result loop for you — up to
8 hops by default (lift the cap with `engine.max_tool_hops(N)` for shell-driven
flows, or just register `bash` and the helpers do it for you).

If you want the canonical agent toolset wired in three lines instead of
seven, use the Toolbelt builder:

```cpp
easyai::Engine engine;
engine.model("models/qwen2.5-1.5b-instruct.gguf").gpu_layers(99).context(4096);

easyai::cli::Toolbelt()
    .sandbox   ("/srv/data")    // enables fs_read_file/list_dir/glob/grep/write
    .allow_bash()                // enables bash + bumps max_tool_hops to 99999
    .apply     (engine);

engine.load();
engine.chat("Find all .md files larger than 1 KB and summarise them.");
```

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

All three example CLIs (`easyai-cli`, `easyai-cli-remote`, `easyai-server`)
follow the same gating model. Default is **safe**: no filesystem access,
no shell.

| Flag             | What it enables                                                         |
|------------------|-------------------------------------------------------------------------|
| (no flag)        | `datetime`, `web_search`, `web_fetch` only.                             |
| `--sandbox <dir>`| `fs_read_file / fs_write_file / fs_list_dir / fs_glob / fs_grep`, ALL scoped to `<dir>`. The model sees a virtual `/`-rooted filesystem; real path is hidden. |
| `--allow-bash`   | `bash` (run `/bin/sh -c`). cwd = `--sandbox <dir>` if given, otherwise the binary's CWD. NOT a hardened sandbox — runs with your user privileges. Also bumps the agentic-loop `max_tool_hops` to 99999 (bash flows naturally span many turns). |

#### `easyai-cli`

```
easyai-cli -m model.gguf [-s system.txt] [--ngl 99] [--no-tools]
            [--sandbox DIR] [--allow-bash]
```

Drop-in REPL.  Type any line to talk; type any of these to control the engine:

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

# REPL with everything wired up — local model
./build/easyai-cli -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf

# REPL talking to a remote OpenAI-compatible endpoint instead of loading a model
./build/easyai-cli --url http://127.0.0.1:8080/v1
./build/easyai-cli --url https://api.openai.com/v1 \
                   --api-key $OPENAI_API_KEY --remote-model gpt-4o-mini

# One-shot mode (great in scripts — banners on stderr, model text on stdout)
./build/easyai-cli -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p "What is 2+2?"
result=$(./build/easyai-cli --url http://127.0.0.1:8080/v1 --no-think -p "summarise this commit")

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
cmake --build build -j --target easyai-cli-remote

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
