# easyai

> **A C++17 framework anyone can use to build AI agents that talk to
> their own services — no llama.cpp, JSON-Schema, or template-engine
> knowledge required.**

easyai turns [llama.cpp](https://github.com/ggml-org/llama.cpp) into an
*agent engine* you can drop into any program in a dozen lines.  You give
it C++ functions; it gives the model the ability to call them.  That's
the whole pitch.

It also ships four ready-to-run binaries built on top of the same
library:

| Binary           | What it gives you                                                                                                                                  |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `easyai-cli`     | Drop-in `llama-cli` replacement: local **or** remote OpenAI-compatible endpoint, one-shot scripting (`-p`), tools, presets, optional `<think>` strip. |
| `easyai-server`  | Drop-in `llama-server` replacement: OpenAI-compat HTTP **with full SSE streaming**, embedded SvelteKit webui, Bearer auth, Prometheus `/metrics`, KV-cache controls, flash-attn, mlock. |
| `easyai-agent`   | A demo agent showing every built-in tool plus an inline custom tool.                                                                                |
| `easyai-recipes` | Tutorial agent paired with `manual.md` — implements `today_is` and `weather` (HTTP-calling) from scratch.                                          |
| `easyai-chat`    | A bare-bones REPL with no tools — useful as a sanity check.                                                                                          |

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
8 hops by default. Tool definitions are 6 lines:

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

### Library (`libeasyai`)

* `easyai::Engine` — high-level wrapper around llama.cpp's model + context +
  sampler + chat templates. Fluent setters, RAII-owned native resources.
* `easyai::Tool` — name + description + JSON-schema params + handler. Builder
  API generates the schema for you.
* `easyai::tools::*` — built-in tools:
  * `datetime` (no deps)
  * `web_fetch` (libcurl, HTML→text)
  * `web_search` (DuckDuckGo HTML, no API key, no external service)
  * `fs_read_file`, `fs_write_file`, `fs_list_dir`, `fs_glob`, `fs_grep`
    (sandboxed to a root directory you provide)
* `easyai::presets` — named sampling profiles
  (`deterministic / precise / balanced / creative / wild`) plus a tiny parser
  that turns chat lines like `"creative 0.9"` or `"/temp 0.5"` into sampling
  overrides.

### Binaries

#### `easyai-cli`

```
easyai-cli -m model.gguf [-s system.txt] [--ngl 99] [--no-tools]
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
