# easyai — developer manual

This is the **hands-on** book. It assumes nothing beyond "I can compile a
C++17 program". By the end you will know how to:

* compile easyai and download a model
* run `easyai-local` and talk to it
* host `easyai-server` and call it from Claude Code, OpenAI SDKs, or curl
* embed `easyai::Engine` in your own program (local llama.cpp)
* embed `easyai::Client` in your own program (remote OpenAI-compatible
  server, with local tools)
* drive a remote server end-to-end with `easyai-cli`, including
  a planning tool and live system-observability tools
* write a custom tool, with typed parameters
* tune the sampler with presets and runtime overrides
* deploy easyai-server as a hardened Linux service and operate it
* debug common issues (context overflow, malformed tool calls, GPU
  fallback, TLS, rate limits)

---

## Table of contents

| Part | Chapter | What you get |
|------|---------|--------------|
| **1** | Getting set up         | Prereqs, repo layout, building, GPUs, models |
| **2** | Using the binaries     | `easyai-local`, `easyai-server`, `easyai-cli`, `easyai-agent`, `easyai-chat`, `easyai-recipes` |
| **3** | Embedding `libeasyai`  | `Agent` (3-line hello), `Backend` (local↔remote), `Engine` API top-to-bottom, callbacks, presets, tools, escape hatches |
| **4** | Embedding `libeasyai-cli` | `Client` API top-to-bottom — your code drives a remote model with local tools |
| **5** | Authoring custom tools | Builder API, schemas, sandboxes, error handling, the `Plan` tool, `system_*` tools cookbook |
| **6** | Deploying easyai-server | Single-binary install, systemd unit, nginx TLS termination, multiple-server fan-out |
| **7** | Operating the server   | `/health` and `/metrics`, presets at runtime, log rotation, crash capture |
| **8** | Performance & tuning   | KV cache types, flash-attn, mlock, ngl auto-fit, prompt-eval throughput, sampler choices |
| **9** | Recipes (cookbook)     | Real prompts + flag combinations, including the planning agent, papers digest, host triage |
| **10** | Troubleshooting       | Build, GPU, runtime, model, tool, network, TLS issues |
| **11** | Design references     | Pointers into `design.md` for the deeper "why" |

> If you're new, read 1 → 2 → 3.  If you want to ship something to a
> remote model right now, jump to **Part 4**.  If you want to write
> your own tool, **Part 5** is the cookbook.  If something's broken,
> **Part 10** has a triage matrix.

---

## Part 1 — getting set up

### 1.1 Layout

easyai expects llama.cpp as a sibling directory:

```
develop/
├── easyai/        # this project
└── llama.cpp/     # https://github.com/ggml-org/llama.cpp
```

Clone llama.cpp if you haven't:

```bash
cd ~/develop
git clone https://github.com/ggml-org/llama.cpp
```

### 1.2 Dependencies

| Required               | Why                                |
|------------------------|------------------------------------|
| CMake ≥ 3.18           | build system                       |
| A C++17 compiler       | the library is C++17               |
| (Apple) Xcode CLT      | Metal headers for GPU acceleration |
| (Linux/Win) Vulkan SDK | optional; pass `-DGGML_VULKAN=ON`  |

| Optional        | Used by             |
|-----------------|---------------------|
| libcurl         | `web_fetch`, `web_search` |

On macOS:

```bash
brew install cmake curl
```

### 1.3 First build

#### 1.3.1 Pick the right configure command for your hardware

| Target                            | Configure command                                                                  |
|-----------------------------------|------------------------------------------------------------------------------------|
| **Apple Silicon / Intel Mac (Metal)** | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`                                |
| **NVIDIA GPU (CUDA)**             | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON`                    |
| **AMD / Intel / cross-vendor (Vulkan)** | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON`            |
| **AMD on Linux (ROCm/HIP)**       | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100` |
| **CPU-only (any OS)**             | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` (then run with `-ngl 0`)          |

**NVIDIA / CUDA** — install the CUDA Toolkit so `nvcc` is on `PATH`. If
CMake complains about an unknown architecture, pin one explicitly:
`-DCMAKE_CUDA_ARCHITECTURES=89` (e.g. for RTX 4090) or use `native`.

**AMD / Vulkan** — install the Vulkan SDK
([LunarG](https://vulkan.lunarg.com/sdk/home) on Win/macOS, distro
`vulkan-tools libvulkan-dev` on Linux). On Linux, also install the GPU
driver's Vulkan ICD (`mesa-vulkan-drivers` for AMD/Intel, NVIDIA driver
ships its own).

**AMD / ROCm** — set `AMDGPU_TARGETS` to your card's gfx version.
Check with `rocminfo`.

**CPU-only** — same configure command as Metal but always pass `-ngl 0` at
runtime (or `engine.gpu_layers(0)` in code) so layers stay on CPU.

#### 1.3.2 Build

```bash
cd easyai
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # use the right line above
cmake --build build -j
```

Outputs land in `build/`:

```
build/easyai-local    # local-only REPL (loads a GGUF in-process)
build/easyai-cli      # agentic REPL talking to a remote OpenAI-compat endpoint
build/easyai-server   # HTTP server + webui
build/easyai-agent    # demo agent (every tool + a custom one)
build/easyai-chat     # bare REPL (no tools)
build/libeasyai.dylib # the library
```

If the configure step says `easyai: libcurl found — web_fetch / web_search enabled`,
both web tools will work out of the box (no extra service to run).

### 1.4 Get a model

Tiny, fast, decent at tools — start here:

```bash
mkdir -p models
curl -L -o models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true'
```

For real work upgrade to Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct.

---

## Part 2 — using the binaries

### 2.1 Hello, REPL

```bash
./build/easyai-local -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

You'll see something like:

```
[easyai-local] loaded models/qwen2.5-1.5b-instruct-q4_k_m.gguf
               backend=MTL0  ctx=4096  tools=7  preset=balanced
               type '/help' for commands, '/quit' to exit
> what's 2+2
2 + 2 equals 4.
>
```

Try a tool:

```
> What time is it right now in UTC?
[tool] datetime -> {"utc":"2026-04-25T10:20:49Z","local":"…"}
The current UTC time is 2026-04-25 10:20:49.
```

Try a preset:

```
> creative
[preset → creative]
> write a haiku about silicon
Quiet wafer hums,
moonlit traces drink the dawn —
glass dreams in the dust.
```

`creative 0.9 …` does both at once: switch preset, override temperature
just for this generation, then run the rest of the line as a prompt.

Use `/help` to list every preset; `/system <text>` to swap the system
prompt mid-session; `/reset` to wipe history.

### 2.2 Hello, server

```bash
./build/easyai-server -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf
./build/easyai-server -m models/...gguf --sandbox ./work --allow-bash
./build/easyai-server -m models/...gguf -s system.txt
```

Without `-s`, the server boots up as **Deep** — an expert system
engineer persona built into the default system prompt.  Deep
operates a `TIME → THINK → PLAN → EXECUTE → VERIFY` loop and treats
`datetime` as the first tool call any time the answer touches "now"
or "today".  Operators who want a different voice supply their own
`--system "<text>"` or `-s persona.txt` — Deep is the default, not
hardcoded.

`--sandbox <dir>` enables fs_* tools scoped to `<dir>`; `--allow-bash`
adds the shell tool.  Both default OFF — fresh installs don't expose
write access or shell to the model until the operator opts in.

If you pass `-s system.txt`, that text becomes the default system
prompt for any request that doesn't already include one.

Open `http://127.0.0.1:8080` in a browser to use the bundled webui, or
talk to it via curl:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"easyai","messages":[{"role":"user","content":"Hi!"}]}'
```

#### 2.2.1 Pointing Claude Code at it

Use whatever Claude Code's "OpenAI-compatible base URL" setting is called in
your version (`--api-base`, env var, or settings file) and set it to
`http://127.0.0.1:8080/v1`. Anything Claude Code declares as a tool will be
forwarded; anything it doesn't declare will use easyai's built-in toolbelt.

#### 2.2.2 Pointing the OpenAI SDK at it

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-checked")
print(client.chat.completions.create(
    model="easyai",
    messages=[{"role":"user","content":"Hi!"}]
).choices[0].message.content)
```

#### 2.2.3 Override temperature inline

Every request body can carry `temperature`, `top_p`, `top_k`. Or your user
can put a preset right in the message:

```json
{ "messages": [{"role":"user","content":"creative 0.9 write me a poem"}] }
```

The server peels `creative 0.9 ` off, applies the override, and the model
sees just `write me a poem`.

### 2.3 Demo agent (every tool, plus a custom one)

```bash
./build/easyai-agent -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Look at `examples/agent.cpp` to see how the tools are registered. The
inline `flip_coin` example is six lines.

---

## Part 3 — embedding the library

### 3.0 Three-line hello world (`easyai::Agent`)

If you remember nothing else, remember this:

```cpp
// hello.cpp
#include "easyai/easyai.hpp"

int main() {
    easyai::Agent a("models/qwen2.5-1.5b-instruct-q4_k_m.gguf");
    std::cout << a.ask("What's 2+2?") << "\n";
}
```

`Agent` is the friendly Tier-1 façade.  Construct, ask, print.
Default toolset (datetime + web_search + web_fetch) is wired in;
fs_* and bash stay off until you opt in via `.sandbox()` or
`.allow_bash()`.  Streaming output is one chained call away:

```cpp
easyai::Agent a("model.gguf");
a.system  ("Be terse.")
 .sandbox ("./workspace")
 .preset  ("creative")
 .on_token([](auto p){ std::cout << p << std::flush; });

a.ask("Read README.md and summarise it.");
```

A remote model works the same way:

```cpp
auto a = easyai::Agent::remote("http://127.0.0.1:8080/v1");
auto a = easyai::Agent::remote("https://api.openai.com/v1",
                               std::getenv("OPENAI_API_KEY"));
```

`Agent` is built on top of `Backend` (3.1.5) which is built on top
of `Engine` (3.1) and `Client` (3.10).  When you need access to the
underlying knobs, `agent.backend()` is the escape hatch — it returns
the materialised `Backend &` so you can reach into `Engine::*` /
`Client::*` setters that `Agent` doesn't surface directly.

CMake:

```cmake
find_package(easyai 0.1 REQUIRED)
add_executable(hello hello.cpp)
target_link_libraries(hello PRIVATE easyai::engine easyai::cli)
```

`Agent` lives in `libeasyai-cli` (because it can transparently
dispatch to either flavour of `Backend`), so link both targets.
If you only need the local engine, drop `easyai::cli` and use
`easyai::Engine` directly (3.1).

### 3.1 Minimal hello

```cpp
// hello.cpp
#include "easyai/easyai.hpp"

int main() {
    easyai::Engine engine;
    engine.model("models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
          .gpu_layers(99)
          .system("Be concise.")
          .on_token([](const std::string & t){ std::cout << t << std::flush; });

    if (!engine.load()) { std::fprintf(stderr, "load failed: %s\n",
                                       engine.last_error().c_str()); return 1; }

    engine.chat("What's 2+2?");
    return 0;
}
```

Add it to `CMakeLists.txt`:

```cmake
add_executable(hello hello.cpp)
target_link_libraries(hello PRIVATE easyai)
```

If your project lives outside this tree and you've installed easyai
(`cmake --install build --prefix /usr/local`), use `find_package`
instead:

```cmake
find_package(easyai 0.1 REQUIRED)
add_executable(hello hello.cpp)
target_link_libraries(hello PRIVATE easyai::engine)
```

`easyai::engine` is the link target for `libeasyai.so` (local llama.cpp
wrapper).  For the OpenAI-protocol client described in 3.9, swap to
`easyai::cli` (or link both side by side).

### 3.1.5 Backend — local OR remote, same shape

If your program needs to handle EITHER a local `-m model.gguf`
flavour OR a remote `--url base` flavour without if-tree
duplication, the abstraction you want is `easyai::Backend`:

```cpp
std::unique_ptr<easyai::Backend> b;
if (!url.empty()) {
    easyai::RemoteBackend::Config rc;
    rc.base_url = url;
    rc.api_key  = api_key;
    rc.with_tools = true;             // dispatch tools locally
    b = std::make_unique<easyai::RemoteBackend>(std::move(rc));
} else {
    easyai::LocalBackend::Config lc;
    lc.model_path = model_path;
    lc.sandbox    = "./workspace";
    lc.allow_bash = true;
    b = std::make_unique<easyai::LocalBackend>(std::move(lc));
}

std::string err;
if (!b->init(err)) { std::cerr << err << "\n"; return 1; }

b->set_system("Be terse.");
auto reply = b->chat("hello?", [](auto p){ std::cout << p << std::flush; });
```

`Backend` is the Tier-3 abstraction `Agent` is built on top of.  Use
it when you want the local↔remote switch but still want to manage the
chat loop yourself, register custom tools, or hook tool callbacks.
The `Config` struct exposes every Engine/Client setting that's
relevant to "configuring an agent" (sampling preset, sandbox,
allow_bash, KV cache controls for local, TLS/timeout for remote).

`LocalBackend` ships in `libeasyai`; `RemoteBackend` in
`libeasyai-cli`.  Linking only the engine library gives you the
local flavour; adding `easyai::cli` adds the remote flavour without
duplicating the abstract base.

### 3.2 Adding a tool

The 6-line shape:

```cpp
engine.add_tool(
    easyai::Tool::builder("today_is")
        .describe("Returns the day of the week.")
        .handle([](const easyai::ToolCall &){
            return easyai::ToolResult::ok("Saturday"); })
        .build());
```

With typed parameters:

```cpp
engine.add_tool(
    easyai::Tool::builder("send_email")
        .describe("Send an email via the company SMTP relay.")
        .param("to",      "string", "Recipient address",  /*required=*/true)
        .param("subject", "string", "Subject line",       true)
        .param("body",    "string", "Plain-text body",    true)
        .param("cc",      "string", "Optional CC address",false)
        .handle([](const easyai::ToolCall & call){
            std::string to, subject, body, cc;
            easyai::args::get_string(call.arguments_json, "to",      to);
            easyai::args::get_string(call.arguments_json, "subject", subject);
            easyai::args::get_string(call.arguments_json, "body",    body);
            easyai::args::get_string(call.arguments_json, "cc",      cc);

            if (to.empty())      return easyai::ToolResult::error("missing 'to'");
            if (subject.empty()) return easyai::ToolResult::error("missing 'subject'");

            // … your real send code …
            return easyai::ToolResult::ok("sent.");
        })
        .build());
```

`Tool::builder` automatically synthesises the JSON schema. If you need
something fancier (nested objects, enums) build the schema string yourself
and use `Tool::make(name, description, schema_json, handler)`.

### 3.3 Sandboxed filesystem tools

Always pass a root directory:

```cpp
engine.add_tool(easyai::tools::fs_read_file ("./workspace"));
engine.add_tool(easyai::tools::fs_write_file("./workspace"));
engine.add_tool(easyai::tools::fs_list_dir  ("./workspace"));
engine.add_tool(easyai::tools::fs_glob      ("./workspace"));
engine.add_tool(easyai::tools::fs_grep      ("./workspace"));
```

Paths sent by the model are anchored to the root by **iterating path
components and dropping any `..`, `.`, or absolute markers** before
joining onto the root.  Total containment by construction — there is
no path the model can construct that escapes.

The model sees a virtual `/`-rooted filesystem (`/report.md`,
`/docs/spec.md`); the real sandbox path is hidden from descriptions
and result messages.

### 3.3.1 Toolbelt — register the canonical agent toolset in 3 lines

Instead of hand-rolling the standard tool registration:

```cpp
easyai::cli::Toolbelt()
    .sandbox   ("./workspace")    // enables all fs_* tools, scoped here
    .allow_bash()                  // adds bash, bumps max_tool_hops to 99999
    .with_plan (plan)              // adds the plan tool
    .apply     (engine);           // or .apply(client) for the remote variant
```

The Toolbelt always includes `datetime` + `web_search` + `web_fetch`;
fs_* and bash are gated behind `.sandbox()` and `.allow_bash()` so a
fresh agent installation can't accidentally expose write or shell.

### 3.3.2 Bash tool — when you need a real shell

```cpp
engine.add_tool(easyai::tools::bash("./workspace"));
engine.max_tool_hops(99999);   // bash flows span many turns
```

`bash` is a `/bin/sh -c` runner. Output (stdout + stderr) is captured
and capped at 32 KiB; per-command timeout defaults to 30 s, max 300 s
(SIGTERM, then SIGKILL +2 s grace). The cwd is pinned to the root.

This is **NOT** a hardened sandbox — the command runs with your user
privileges. It's appropriate for local single-user agents; for
anything multi-tenant or production, run easyai-server inside a
container / firejail / unprivileged user.

### 3.3.3 `get_current_dir` — anchor relative paths

```cpp
engine.add_tool(easyai::tools::get_current_dir());
```

A zero-parameter tool that returns the absolute path of the process's
current working directory at call time. Pair it with `--sandbox`: the
CLIs and server `chdir` into the sandbox at startup, so what
`get_current_dir` reports is exactly the directory `bash`, `read_file`,
`write_file`, `list_dir`, `glob` and `grep` operate against. Models
that don't already know the path should call it once at the start of
a task; for any subsequent file op, relative paths just work.

The `Toolbelt` adds it automatically when any filesystem-flavoured
tool is enabled (`allow_fs` or `allow_bash`); register it manually if
you build the toolbelt by hand.

### 3.3.4 External tools manifest — declare commands in JSON

For tools that wrap an existing CLI binary (`uname`, `pgrep`, `git`,
internal scripts, etc.) you can declare them in a JSON manifest
without writing C++. The `--tools-json PATH` flag is supported by
`easyai-local`, `easyai-cli`, and `easyai-server`. From C++:

```cpp
auto loaded = easyai::load_external_tools_from_json(path, /*reserved=*/{});
if (!loaded.error.empty()) {
    std::fprintf(stderr, "tools-json: %s\n", loaded.error.c_str());
    return 1;
}
for (auto & t : loaded.tools) engine.add_tool(std::move(t));
```

Manifest schema (one entry — see `examples/tools.example.json` for
more):

```json
{
  "version": 1,
  "tools": [
    {
      "name": "list_processes",
      "description": "List running processes whose name matches a regex pattern.",
      "command": "/usr/bin/pgrep",
      "argv": ["-a", "{pattern}"],
      "parameters": {
        "type": "object",
        "properties": {
          "pattern": { "type": "string", "description": "Regex." }
        },
        "required": ["pattern"]
      },
      "timeout_ms": 5000,
      "max_output_bytes": 65536,
      "cwd": "$SANDBOX",
      "env_passthrough": ["PATH"],
      "stderr": "discard",
      "treat_nonzero_exit_as_error": false
    }
  ]
}
```

**Field-by-field reference**

| Field | Required | Notes |
| --- | --- | --- |
| `name` | yes | `^[a-zA-Z][a-zA-Z0-9_]{0,63}$`. Must not collide with built-ins (`bash`, `read_file`, …) or already-registered tools. |
| `description` | yes | Plain-English text for the model. 1..4096 chars. The model uses this to decide *when* to call your tool, so write it well. |
| `command` | yes | **Absolute** path to a regular, executable file. Relative names are rejected at load (no PATH search → no PATH-hijack risk). |
| `argv` | yes | Array of strings. Each element is either a literal (no `{` or `}`) or exactly `"{paramname}"`. Embedded placeholders (`"--flag={x}"`) are rejected — split into two elements (`["--flag", "{x}"]`) instead. |
| `parameters` | optional | JSON-Schema-shaped: `{type:"object", properties:{...}, required:[...]}`. Types accepted: `string`, `integer`, `number`, `boolean`. |
| `timeout_ms` | optional | Default 10000. Clamped to [100, 300000]. |
| `max_output_bytes` | optional | Default 65536. Clamped to [1024, 4 MiB]. Excess output is silently discarded; the response notes the truncation. |
| `cwd` | optional | Either an absolute path or the magic token `"$SANDBOX"` which resolves to the process's CWD at load time. Default: `"$SANDBOX"`. |
| `env_passthrough` | optional | Allowlist of parent-process env vars to inherit. **Default empty** — the subprocess gets a clean env. Add `"PATH"`, `"HOME"`, etc. only when the wrapped command needs them. |
| `stderr` | optional | `"merge"` (default) or `"discard"`. |
| `treat_nonzero_exit_as_error` | optional | Default `true`. Set `false` for tools whose non-zero exit is informational (`pgrep` returns 1 when nothing matches). |

**Security guarantees** — these are enforced, not aspirational:

1. **No shell.** The runner uses `fork` + `execve` with an argv array.
   The model's argument never passes through a shell parser, so
   quoting / `;` / backticks / `$(…)` cannot escape its argv slot.
2. **Absolute command path.** Validated at load (regular file +
   executable bit). No PATH lookup, no PATH-hijack.
3. **Whole-element placeholders only.** A model argument fills exactly
   one argv element; it can't be concatenated into a literal.
4. **Schema-validated arguments.** Type errors are surfaced as a
   `ToolResult::error` *before* anything is spawned. Required-but-
   missing arguments are rejected.
5. **Hard caps.** Manifest size (1 MiB), tools per manifest (128),
   params per tool (32), env passthrough size (16), argv elements
   (256), per-arg bytes (4 KiB). Each cap closes a class of DoS.
6. **Clean env by default.** Only listed `env_passthrough` vars
   inherit. `LD_PRELOAD`, `PATH`, etc. don't leak in unless asked.
7. **Closed stdin.** No way to feed the subprocess from the model.
8. **Process-group timeout.** SIGTERM to the group on `timeout_ms`,
   SIGKILL after a 1 s grace — kills any grandchildren the command
   spawned, not just the top-level process.
9. **Inherited fds closed.** All fds ≥ 3 are closed in the child
   before exec, so the agent's HTTP transport / log files / database
   handles do not leak into the spawned command.

The manifest is the operator's deploy artefact — treat it like a
sudoers file. Anyone who can write it can run arbitrary commands as
the agent's user.

```sh
# enable from the CLIs
easyai-local --sandbox ./work --tools-json mytools.json
easyai-cli   --sandbox ./work --tools-json mytools.json --url http://...
easyai-server -m model.gguf --sandbox /srv/agent --tools-json /srv/agent/tools.json
```

### 3.4 Streaming token output

Just register `on_token`:

```cpp
engine.on_token([](const std::string & piece){
    std::cout << piece << std::flush;
});
```

Pieces are *substrings of UTF-8 tokens*. Most pieces are full tokens, but
multi-byte characters can split across pieces — buffer if you need
character-precise rendering.

### 3.5 Listening for tool calls (telemetry / UI hooks)

```cpp
engine.on_tool([](const easyai::ToolCall & c, const easyai::ToolResult & r){
    log_metric("tool_call", { {"name", c.name}, {"is_error", r.is_error} });
});
```

The callback fires after every dispatched tool, success or failure.

### 3.6 Resetting between conversations

```cpp
engine.clear_history();   // wipes history + KV cache + sampler state
engine.system("You are now a different assistant.");
```

If you want to programmatically replay a conversation, e.g. when restoring
from a database:

```cpp
engine.replace_history({
    {"system",    "You are a helpful assistant."},
    {"user",      "What's the capital of Brazil?"},
    {"assistant", "Brasília."},
    {"user",      "And of France?"},
});
auto reply = engine.chat("");  // generate the next assistant turn
```

### 3.7 Switching presets at runtime

```cpp
const easyai::Preset * p = easyai::find_preset("creative");
if (p) engine.set_sampling(p->temperature, p->top_p, p->top_k, p->min_p);
```

Or to honour a chat-line command from your own UI:

```cpp
auto pr = easyai::parse_preset(user_line);
if (!pr.applied.empty()) {
    engine.set_sampling(pr.temperature, pr.top_p, pr.top_k, pr.min_p);
    user_line = user_line.substr(pr.consumed);   // strip prefix
}
engine.chat(user_line);
```

### 3.8 Recipe book — write your first tools, step by step

> **In this chapter, you'll learn to:**
> * understand what an "AI tool" really is (it's just a C++ function!)
> * write a tool that returns today's date
> * write a tool that fetches live weather from the internet
> * give your agent both tools and watch it answer real questions
> * recognise when to reach for the more advanced building blocks
>
> **You don't need to know:** llama.cpp, JSON Schema, Jinja templates,
> or anything about how language models work under the hood.

This is the chapter every other chapter has been pointing at.  When
people ask *"what's so cool about easyai?"* — this is the answer.
You're going to give a small AI model two new abilities in about
fifty lines of code, and at the end you'll have a working agent that
genuinely reaches out to the internet on your behalf.

There's a finished version of everything below in
**`examples/recipes.cpp`**.  Build it now so you can compare:

```bash
cmake --build build -j --target easyai-recipes
```

We'll come back to that binary at the end and run it.

---

#### Chapter opener — what *is* a tool, anyway?

Imagine you hire a brilliant intern.  They're fast, polite, and they
know almost everything — but they joined the company yesterday so
they don't know your customer database, they don't have your VPN, and
they can't see today's calendar.  How do you make them useful?

You give them a phone book of internal services and you tell them:
*"if anyone asks about a customer, call this number; if they ask about
billing, call that one."*

That's exactly what a tool is to an AI model.  Each tool you register
is a phone-book entry.  The model gets to read three things about it:

| Field          | What goes here                                              | Read by    |
|----------------|-------------------------------------------------------------|------------|
| **name**       | A short identifier — e.g. `today_is`, `weather`            | the model  |
| **description**| One sentence: *what does this do, when should I use it?*   | the model  |
| **handler**    | A normal C++ function that gets called for you             | easyai     |

When the model decides "I should use the weather tool", easyai catches
that intent, runs your handler with whatever arguments the model picked,
and feeds the result back so the model can finish its answer.

The whole dance, drawn out:

```
   user                   model                   easyai            your handler
    │                        │                       │                     │
    │  "What's the weather   │                       │                     │
    │   in São Paulo?"  ───▶ │                       │                     │
    │                        │  "I'll call            │                     │
    │                        │   weather(city=…)" ──▶│                     │
    │                        │                       │  weather(...)  ───▶ │
    │                        │                       │                     │ … HTTP call …
    │                        │                       │ ◀──── "São Paulo:   │
    │                        │ ◀──── tool result ────│       ⛅ +24°C"     │
    │                        │                       │                     │
    │  "São Paulo is a       │                       │                     │
    │   pleasant 24°C…" ◀────│                       │                     │
```

You write the handler.  Everything else is automatic.

---

#### Recipe 1 — your first tool: "what is today's date?"

Most tiny AI models have **no idea what today's date is**.  Their
training data ended months (sometimes years) ago.  Ask Qwen2.5-1.5B
"what's today's date?" and you'll usually get a confident-sounding
hallucination.

Let's fix that with eight lines.

##### Type this

```cpp
easyai::Tool today_is() {
    return easyai::Tool::builder("today_is")
        .describe("Returns today's date in ISO-8601 format (YYYY-MM-DD, UTC).")
        .handle([](const easyai::ToolCall &) {
            auto now = std::chrono::system_clock::now();
            auto t   = std::chrono::system_clock::to_time_t(now);
            char buf[16];
            std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::gmtime(&t));
            return easyai::ToolResult::ok(buf);
        })
        .build();
}
```

##### What just happened?

Read it line by line — there's nothing magical:

1. **`Tool::builder("today_is")`** — pick a name for the tool.  Use
   `snake_case`.  This is the name the model will speak when it wants
   to use the tool.
2. **`.describe(...)`** — write a one-line description that you'd give
   a smart intern.  *"Returns today's date in ISO-8601 format"* is
   crystal-clear.  *"Useful for date stuff"* would not be.
3. **`.handle(...)`** — the C++ that does the real work.  Here it's a
   little lambda that calls the standard library.  No llama.cpp, no
   JSON, no AI-specific API.
4. **`ToolResult::ok(buf)`** — pack the string into a success result.
   Whatever you pass here is what the model sees back as the tool's
   reply.
5. **`.build()`** — turn the recipe into the actual `Tool` object.

> **Tip.**  The description is the *only* hint the model has about
> when to call your tool.  Write it for an LLM, not for your IDE.
> Be specific, give an example output, mention units.

##### Hand it to the engine

```cpp
easyai::Engine engine;
engine.model("models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
      .add_tool(today_is())     // ← your new tool
      .load();
engine.chat("What's the date today?");
```

That's it.  Eight lines for the tool plus three for the wiring, and
your agent now has reliable date access.

> **Try it.**  Wrap the snippet above in a `main()`, link against
> `easyai`, build, and run.  Or just look at `examples/recipes.cpp` —
> it's the same code, ready to go.

---

#### Recipe 2 — talking to the internet: a "weather" tool

Today's date is fun, but the real point of giving an AI tools is so
it can reach out to systems you control: your database, your APIs,
your filesystem, the internet.

Let's write a `weather` tool.  We'll use **wttr.in** — a free,
no-signup service that takes a city name and replies in plain text:

```
$ curl 'https://wttr.in/Sao Paulo?format=3'
São Paulo: ⛅ +24°C
```

That's the whole API.  Our job is to wrap that in a tool.

We're going to do this in four small steps so nothing feels like a
leap.

##### Step 1 — Declare the input parameter

This time the tool needs a parameter (`city`).  The builder makes
that one extra line:

```cpp
easyai::Tool::builder("weather")
    .describe("Returns the current weather for a city.  Backed by wttr.in "
              "— free, no API key, plain-text reply.")
    .param("city", "string",
           "City name, e.g. 'Berlin' or 'Sao Paulo'.  Required.",
           /*required=*/true)
```

`param(name, type, description, required)` is all you ever need.
The valid `type` values are:

| `type`      | C++ in your handler             |
|-------------|---------------------------------|
| `"string"`  | `std::string` via `args::get_string_or(...)` |
| `"integer"` | `long long` via `args::get_int_or(...)` |
| `"number"`  | `double` via `args::get_double_or(...)` |
| `"boolean"` | `bool` via `args::get_bool_or(...)` |
| `"array"`   | parse the JSON yourself         |
| `"object"`  | parse the JSON yourself         |

> **Heads-up.**  Tiny models occasionally forget required parameters.
> Always validate inside your handler — see step 3.

##### Step 2 — Read the parameter inside the handler

The model packs the arguments into a JSON blob (e.g.
`{"city":"Sao Paulo"}`).  easyai gives you a tiny scanner so you
don't need a JSON library:

```cpp
.handle([](const easyai::ToolCall & call) {
    std::string city = easyai::args::get_string_or(
        call.arguments_json, "city", "");
    if (city.empty()) {
        return easyai::ToolResult::error("missing 'city' argument");
    }
    ...
```

That **one line** with `get_string_or` replaces the four lines of
"declare, get, check, default" pattern you'd write in plain C++.

The full helper menu:

| Helper                                              | Returns…                          |
|-----------------------------------------------------|-----------------------------------|
| `args::get_string_or(json, key, default)`           | the value, or your default        |
| `args::get_int_or   (json, key, default)`           | same idea, `long long`            |
| `args::get_double_or(json, key, default)`           | same idea, `double`               |
| `args::get_bool_or  (json, key, default)`           | same idea, `bool`                 |
| `args::has(json, key)`                              | `bool` — did the model fill it in?|

(There's an older `bool args::get_string(json, key, &out)` form
that's still around when you need to tell *"absent"* apart from
*"present but empty"*.)

##### Step 3 — Make the actual call

Anything you can do in C++ goes here: hit a REST API, query SQLite,
shell out to a Python script, send a Slack message, ring a bell on
the desk next to you.  In our case it's an HTTP GET, and libcurl
takes about ten lines:

```cpp
CURL * h = curl_easy_init();
char * escaped = curl_easy_escape(h, city.c_str(), 0);   // URL-safe
std::string url = "https://wttr.in/";
url += escaped ? escaped : city.c_str();
url += "?format=3";                                       // one-line summary
if (escaped) curl_free(escaped);

std::string body;
curl_easy_setopt(h, CURLOPT_URL,            url.c_str());
curl_easy_setopt(h, CURLOPT_USERAGENT,      "easyai-recipes/0.1");
curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
curl_easy_setopt(h, CURLOPT_TIMEOUT,        15L);
curl_easy_setopt(h, CURLOPT_WRITEFUNCTION,  capture_body);   // see recipes.cpp
curl_easy_setopt(h, CURLOPT_WRITEDATA,      &body);
CURLcode rc   = curl_easy_perform(h);
long     code = 0;
curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &code);
curl_easy_cleanup(h);
```

> **Don't panic** at the libcurl block — copy and paste it into any
> tool that needs the network and tweak the URL.  The boilerplate is
> the same every time.  Half of your future tools will be exactly
> this shape.

##### Step 4 — Return success or a typed error

```cpp
if (rc != CURLE_OK) {
    return easyai::ToolResult::error(
        std::string("HTTP transport error: ") + curl_easy_strerror(rc));
}
if (code >= 400) {
    return easyai::ToolResult::error(
        "wttr.in returned HTTP " + std::to_string(code));
}
return easyai::ToolResult::ok(body);
```

Two return flavours, only:

* **`ToolResult::ok(text)`** — the model sees `text` as the reply.
* **`ToolResult::error(msg)`** — easyai marks the message as a failure
  so the model knows to recover (try a different tool, ask the user,
  apologise).

> **Why this matters.**  When a tool errors, well-trained models *do
> the right thing*.  They don't pretend the call worked.  They tell
> the user *"the weather service is unavailable, want me to try
> again later?"*  Use `error` for anything that isn't a success.

---

#### Putting it together — your first running agent

The whole `main()` is in `examples/recipes.cpp`:

```cpp
easyai::Engine engine;
engine.model(model_path)
      .context(4096)
      .gpu_layers(99)
      .system("You are a concise assistant.  Use tools whenever they help.")
      .add_tool(today_is())
      .add_tool(weather())
      .on_token([](const std::string & p){ std::cout << p << std::flush; });

if (!engine.load()) {
    std::fprintf(stderr, "load failed: %s\n", engine.last_error().c_str());
    return 1;
}
engine.chat("What's today's date, and what's the weather in Sao Paulo right now?");
```

Run it:

```
$ ./build/easyai-recipes models/qwen2.5-1.5b-instruct-q4_k_m.gguf
[recipes] backend=Metal  ctx=4096  tools=2

Today is 2026-04-26.  São Paulo currently shows ⛅ +24°C, so light
clothes with a thin layer for the evening should be perfect.
```

##### What just happened?

* The model received your one English sentence.
* It noticed it didn't know the date — so it called `today_is`.
* It noticed it didn't know the weather — so it called `weather`
  with `{"city":"Sao Paulo"}`.
* easyai ran both your handlers, captured both replies, and fed
  them back into the model.
* The model wove them into one fluent answer.

You wrote two C++ functions.  easyai did the rest.

---

#### Going further (when you're ready)

This sub-section is a quick tour of doors you can walk through next.
Each is fully optional.

##### More than one parameter

Mix and match types, mark some as optional, use `_or` helpers to
thread defaults right through:

```cpp
easyai::Tool::builder("send_alert")
    .describe("Push a one-line alert to the on-call channel.")
    .param("text",       "string",  "Message body.  Required.",                   true)
    .param("severity",   "string",  "info | warning | critical.  Default 'info'.", false)
    .param("notify_now", "boolean", "Page on-call immediately?",                   false)
    .handle([](const easyai::ToolCall & call) {
        auto text     = easyai::args::get_string_or(call.arguments_json, "text", "");
        auto severity = easyai::args::get_string_or(call.arguments_json, "severity", "info");
        auto pageNow  = easyai::args::get_bool_or  (call.arguments_json, "notify_now", false);

        if (text.empty()) return easyai::ToolResult::error("missing 'text'");
        // … your real send code …
        return easyai::ToolResult::ok("alert dispatched.");
    })
    .build();
```

##### When the builder isn't enough

The builder makes a flat JSON-Schema (just `properties` + `required`).
For 95% of tools that's plenty.  Need enums, nested objects, arrays?
Drop down to **`Tool::make()`** with a hand-written schema:

```cpp
engine.add_tool(easyai::Tool::make(
    "create_ticket",
    "Open a Jira ticket.",
    R"({
      "type": "object",
      "properties": {
        "project":  { "type": "string" },
        "summary":  { "type": "string" },
        "priority": { "type": "string", "enum": ["P0","P1","P2","P3"] },
        "labels":   { "type": "array", "items": { "type": "string" } }
      },
      "required": ["project","summary"]
    })",
    [](const easyai::ToolCall & call) {
        // parse with nlohmann::json (vendored at ../llama.cpp/vendor) …
        return easyai::ToolResult::ok("JRA-1234");
    }));
```

Same engine, same callback shape, full schema control.

##### Where to read more

* **`src/builtin_tools.cpp`** — `web_search`, `web_fetch`, and the
  filesystem tools.  All written with the exact API you've been using.
  No internal magic; copy any of them as a starting point.
* **`examples/agent.cpp`** — every built-in plus a one-liner
  `flip_coin` for the shortest possible custom tool.
* [3.3 Sandboxed filesystem tools](#33-sandboxed-filesystem-tools) —
  expose a directory to the model without giving away the whole disk.
* [3.5 Listening for tool calls](#35-listening-for-tool-calls-telemetry--ui-hooks)
  — log every dispatch, light up a UI spinner, push to Prometheus.

##### Ten things you can build in an afternoon

If you want practice, pick one and tell us what you came up with:

1. `now()` — current time in any timezone (parameter `tz`).
2. `coin_flip()` — heads/tails (no parameters).
3. `roll_dice()` — `count` + `sides` parameters.
4. `unit_convert()` — temp/length/weight; HTTP-free.
5. `wikipedia_summary()` — calls `en.wikipedia.org/api/rest_v1/page/summary/<title>`.
6. `slack_post()` — your incoming-webhook URL goes in code.
7. `sqlite_query()` — read-only, parameter `sql`.  Sandbox to one DB.
8. `git_log()` — last N commits of a sandboxed repo.
9. `prometheus_query()` — point at your local `/api/v1/query` endpoint.
10. `home_assistant()` — toggle a light by entity ID.  Now you've
    built the front-end of a smart home.

> **You're done with the chapter.**  Anything you can call from C++,
> you can hand to your AI agent.  That's the entire promise of easyai
> as a framework — and you have everything you need.

### 3.9 The `generate_one()` escape hatch

Use this when you want **one** assistant turn out (no internal tool loop) so
*you* can decide what to do with any tool calls — exactly what the HTTP
server does when the client provides its own tools:

```cpp
engine.push_message("user", "Call get_weather for Tokyo.");
auto turn = engine.generate_one();

if (turn.finish_reason == "tool_calls") {
    for (size_t i = 0; i < turn.tool_calls.size(); ++i) {
        const auto & [name, args] = turn.tool_calls[i];
        std::string result = my_remote_executor(name, args);
        engine.push_message("tool", result, name, turn.tool_call_ids[i]);
    }
    auto final = engine.generate_one();   // model digests tool result
    std::cout << final.content;
} else {
    std::cout << turn.content;
}
```

### 3.10 Talking to a remote endpoint with `easyai::Client`

`libeasyai-cli` is the network-side counterpart of `libeasyai`.  Same
fluent API, same `Tool` registration model, same agentic loop — the
model runs on a remote `/v1/chat/completions` endpoint while your
tools execute locally.

```cpp
// remote.cpp
#include "easyai/client.hpp"
#include "easyai/builtin_tools.hpp"
#include "easyai/plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>

int main() {
    easyai::Client cli;
    cli.endpoint("http://ai.local:8080")
       .api_key(std::getenv("EASYAI_API_KEY") ? std::getenv("EASYAI_API_KEY") : "")
       .model("EasyAi")
       .system("You are a planning agent. Be concise.")
       .temperature(0.2f)
       .top_p(0.95f)
       .seed(42);

    cli.add_tool(easyai::tools::datetime());
    cli.add_tool(easyai::tools::web_search());
    cli.add_tool(easyai::tools::web_fetch());

    easyai::Plan plan;
    plan.on_change([](const easyai::Plan & p){
        std::cout << "\n[plan]\n";
        p.render(std::cout);
    });
    cli.add_tool(plan.tool());

    cli.on_token ([](const std::string & p){ std::cout << p << std::flush; });
    cli.on_reason([](const std::string & p){ std::cerr << p << std::flush; });
    cli.on_tool  ([](const easyai::ToolCall & call, const easyai::ToolResult & r){
        std::fprintf(stderr, "%s %s(%s)\n",
                     r.is_error ? "✗" : "🔧",
                     call.name.c_str(),
                     call.arguments_json.c_str());
    });

    std::string answer = cli.chat("Resumo dos 3 papers mais citados sobre Mamba este ano.");
    if (answer.empty() && !cli.last_error().empty()) {
        std::fprintf(stderr, "error: %s\n", cli.last_error().c_str());
        return 1;
    }
    std::cout << "\n";
    return 0;
}
```

CMake — find_package style (after `cmake --install`):

```cmake
find_package(easyai 0.1 REQUIRED)
add_executable(remote remote.cpp)
target_link_libraries(remote PRIVATE easyai::cli)
```

`easyai::cli` transitively pulls `easyai::engine` so `Tool` / `Plan` /
the `easyai::tools::*` factories are available without extra link
flags.

**Sampling and penalty knobs** are all there as fluent setters:
`temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`,
`frequency_penalty`, `presence_penalty`, `seed`, `max_tokens`,
`stop(vector)`, `extra_body_json` (free-form JSON merged last so it can
override anything the typed setters wrote, useful for non-standard
server extensions like `{"reasoning_effort":"high"}`).

**Server management** without touching curl:

```cpp
std::vector<easyai::RemoteModel> models;
cli.list_models(models);

std::vector<easyai::RemoteTool> remote_tools;
cli.list_remote_tools(remote_tools);     // GET /v1/tools

if (!cli.health()) std::fprintf(stderr, "down: %s\n", cli.last_error().c_str());

std::string props_json;
cli.props(props_json);                    // GET /props (raw JSON)

std::string prom_text;
cli.metrics(prom_text);                   // GET /metrics (Prometheus)

cli.set_preset("creative");               // POST /v1/preset
```

The `easyai-cli` binary (`examples/cli.cpp`) is a
ready-to-run reference for all of the above — REPL or one-shot, every
sampling knob exposed as a flag, seven management subcommands
(`--list-models`, `--list-tools`, `--list-remote-tools`, `--health`,
`--props`, `--metrics`, `--set-preset NAME`).

---

## Part 4 — embedding `libeasyai-cli` (remote agent)

This part is the deep dive on `easyai::Client`.  Use this when the
*model* lives on another machine (or another process) and you want
*your* code to drive the conversation with locally-executed tools.
That's the canonical "agent" architecture — model is rented, brain
trusts itself, hands stay on your laptop.

### 4.1 What `Client` does for you

* Builds a valid OpenAI `/v1/chat/completions` request body.
* Streams the SSE response back, splitting `delta.content`,
  `delta.reasoning_content`, and incremental `delta.tool_calls` into
  your callbacks as they arrive.
* When the model emits `finish_reason="tool_calls"`, dispatches the
  matching `easyai::Tool` *in your process*, captures the result, and
  re-issues the request with the tool message appended — repeating
  until the model emits a non-tool `finish_reason`.
* Caps the agentic loop at 8 hops (matches `Engine::chat_continue`).
* Stores the conversation as raw OpenAI-shape JSON strings internally
  so no JSON type ever leaks through the public ABI.

### 4.2 Setting up the client (fluent)

```cpp
#include "easyai/client.hpp"
#include "easyai/builtin_tools.hpp"
#include "easyai/plan.hpp"

easyai::Client cli;
cli.endpoint("http://ai.local:8080")     // any /v1/chat/completions URL
   .api_key(std::getenv("OPENAI_API_KEY") ? std::getenv("OPENAI_API_KEY") : "")
   .model("EasyAi")                       // request body 'model' field
   .system("You are a planning agent. Be concise.")
   .timeout_seconds(600)                  // connect + read
   .verbose(false);                       // true = log SSE traffic to stderr
```

`endpoint` accepts any HTTP or HTTPS URL.  When the build was linked
with OpenSSL (default if `libssl-dev` is present at configure time)
HTTPS just works.  For dev with a self-signed cert:

```cpp
cli.tls_insecure(true);                  // skip peer cert verification
// or:
cli.ca_cert_path("/etc/ssl/certs/internal-ca.pem");  // trust a custom CA
```

### 4.3 Sampling and penalty knobs

Every standard OpenAI / llama-server / easyai-server field is a
fluent setter.  Pin only the ones you care about — leaving any of
them alone keeps the server's default in effect.

```cpp
cli.temperature(0.2f)
   .top_p(0.95f)
   .top_k(40)
   .min_p(0.05f)               // llama-server / easyai
   .repeat_penalty(1.1f)       // llama-server / easyai
   .frequency_penalty(0.0f)    // OpenAI standard, [-2.0, 2.0]
   .presence_penalty(0.0f)     // OpenAI standard, [-2.0, 2.0]
   .seed(42)                   // deterministic; -1 = randomise
   .max_tokens(512)
   .stop({ "\n\nUSER:", "\n\nQ:" });
```

For non-standard server fields (`reasoning_effort`, `tool_choice`,
provider-specific extensions) there's an escape hatch:

```cpp
cli.extra_body_json(R"({"reasoning_effort":"high","logit_bias":{"50256":-100}})");
```

The string MUST parse as a JSON object; its keys merge into the
request body **last**, so they override anything the typed setters
wrote (handy for emergency one-offs).

### 4.4 Tools (registered locally)

Same `easyai::Tool` type used by `Engine`.  The handler runs in your
process when the model picks the tool.

```cpp
// Built-in tools (compiled into libeasyai):
cli.add_tool(easyai::tools::datetime());
cli.add_tool(easyai::tools::web_search());
cli.add_tool(easyai::tools::web_fetch());
cli.add_tool(easyai::tools::fs_read_file("/data"));   // sandbox to /data
cli.add_tool(easyai::tools::fs_list_dir ("/data"));

// Built-in plan tool — separate object so you can render its state.
easyai::Plan plan;
plan.on_change([](const easyai::Plan & p){
    std::cout << "\n[plan]\n";
    p.render(std::cout);
});
cli.add_tool(plan.tool());

// Your own tool, inline:
cli.add_tool(easyai::Tool::builder("flip_coin")
    .describe("Returns 'heads' or 'tails' with uniform probability.")
    .handle([](const easyai::ToolCall &){
        return easyai::ToolResult::ok((std::rand() & 1) ? "heads" : "tails");
    }).build());
```

There is **no API difference** between a `Tool` registered on `Engine`
and one registered on `Client` — your authoring code is portable
across "local model" and "remote model" deployments.

### 4.5 Streaming callbacks

```cpp
cli.on_token([](const std::string & piece) {
    std::fputs(piece.c_str(), stdout);
    std::fflush(stdout);
});
cli.on_reason([](const std::string & piece) {
    // Optional: render the model's hidden reasoning in dim grey.
    std::fprintf(stderr, "\033[2m%s\033[0m", piece.c_str());
});
cli.on_tool([](const easyai::ToolCall & call,
                const easyai::ToolResult & r) {
    std::fprintf(stderr, "[tool] %s%s -> %s\n",
                 r.is_error ? "FAIL " : "",
                 call.name.c_str(),
                 r.content.substr(0, 120).c_str());
});
```

`on_reason` is opt-in by design — many UIs hide reasoning by default
(it's noisy, and some servers don't emit it at all).  `on_token` is
the visible reply; `on_tool` fires once per dispatched tool round-trip
(call + result already paired).

### 4.6 Driving the conversation

```cpp
std::string answer = cli.chat("Resumo dos 3 papers mais citados sobre Mamba este ano.");

if (answer.empty() && !cli.last_error().empty()) {
    std::fprintf(stderr, "error: %s\n", cli.last_error().c_str());
    std::exit(1);
}
```

`chat()` pushes the user message into history, runs the agentic loop,
and returns the final visible content.  Successive `chat()` calls
keep the conversation going (history is preserved).  To start over:

```cpp
cli.clear_history();
```

For more control (e.g. injecting tool results from outside), use
`chat_continue()` after pushing your own messages onto history via
the lower-level shape — but `chat()` is what 99% of agents want.

### 4.7 Server-management endpoints

Each method maps 1:1 to the matching easyai-server route, returns
`true` on success, and writes diagnostic detail to `last_error()` on
failure.  Together they make the lib enough to script and recreate a
server's state from scratch.

```cpp
std::vector<easyai::RemoteModel> models;
cli.list_models(models);                  // GET /v1/models

std::vector<easyai::RemoteTool> tools;
cli.list_remote_tools(tools);             // GET /v1/tools (easyai extension)

if (!cli.health()) {                       // GET /health
    std::fprintf(stderr, "down: %s\n", cli.last_error().c_str());
}

std::string props;
cli.props(props);                          // GET /props (raw JSON)

std::string prom;
cli.metrics(prom);                         // GET /metrics (Prometheus text)

cli.set_preset("creative");                // POST /v1/preset
```

### 4.8 The `easyai-cli` binary as a reference

Everything above is exposed as flags on `examples/cli.cpp`.
Read its source to see one possible "wire it all up" pattern; lift
chunks into your own app verbatim.

```bash
# REPL with the default tool set (datetime, plan, web_search,
# web_fetch, system_*); EASYAI_URL / EASYAI_API_KEY env vars work too.
easyai-cli --url http://ai.local:8080

# One-shot scripted call with a custom tool whitelist:
easyai-cli --url https://api.openai.com \
  --api-key $OPENAI_API_KEY --model gpt-4o-mini \
  --tools datetime,plan,web_search,web_fetch \
  -p "Investigate today's most-cited mamba arxiv papers; produce a 5-bullet summary."

# Pin sampling + add stop sequences:
easyai-cli --url http://ai.local:8080 \
  --temperature 0.0 --top-p 0.9 --seed 42 --stop "USER:" --stop "Q:" \
  -p "Translate the next sentence to PT-BR: ..."

# Non-standard reasoning_effort field via --extra-json:
easyai-cli --url https://api.openai.com --api-key $K --model o1-preview \
  --extra-json '{"reasoning_effort":"high"}' \
  -p "Plan the Mars-mission trajectory."

# List local tools and exit (what the model will be told about):
easyai-cli --url http://x --list-tools

# List server-side tools (easyai-server-only extension):
easyai-cli --url http://ai.local:8080 --list-remote-tools
```

REPL specials inside the interactive mode:

| Command         | Effect                                                  |
|-----------------|---------------------------------------------------------|
| `/exit /quit`   | leave                                                   |
| `/clear`        | clear conversation history (keep tools + system)        |
| `/reset`        | clear history AND clear plan                            |
| `/plan`         | re-print the plan checklist                             |
| `/tools`        | list locally-registered tools                           |
| `/help`         | show specials                                           |

---

## Part 5 — authoring custom tools

This is the cookbook for adding tools the model can call.  Every tool
in `libeasyai`'s built-in set was written exactly the way you'll
write yours.

### 5.1 Anatomy of a Tool

```cpp
struct Tool {
    std::string name;
    std::string description;
    std::string parameters_json;      // JSON schema
    ToolHandler handler;              // std::function<ToolResult(const ToolCall &)>;
};
```

Four fields.  The first three feed the chat template's tool-call
section so the model knows what's available; the fourth is your
function pointer.

### 5.2 Two ways to build a Tool

**Builder** (the typed shorthand, generates the JSON schema for you):

```cpp
easyai::Tool::builder("weather")
    .describe("Return the current weather for a city, in metric units.")
    .param("city", "string", "Name of the city, e.g. 'Lisbon'", /*required=*/true)
    .param("units", "string", "'metric' (default) or 'imperial'.",  false)
    .handle([](const easyai::ToolCall & c) -> easyai::ToolResult {
        std::string city  = easyai::args::get_string_or(c.arguments_json, "city",  "");
        std::string units = easyai::args::get_string_or(c.arguments_json, "units", "metric");
        if (city.empty()) return easyai::ToolResult::error("'city' is required");
        // …call wttr.in…
        return easyai::ToolResult::ok("23 °C, sunny");
    })
    .build();
```

**`Tool::make`** (raw schema string, when you need nested objects /
enums / oneOf that the typed param API can't express):

```cpp
easyai::Tool::make(
    "rgba_set",
    "Set the LED RGBA at index.",
    R"({"type":"object",
        "properties":{
          "i":{"type":"integer","minimum":0,"maximum":31},
          "color":{"type":"object","properties":{
            "r":{"type":"integer"},"g":{"type":"integer"},
            "b":{"type":"integer"},"a":{"type":"integer"}
          },"required":["r","g","b"]}
        },
        "required":["i","color"]})",
    [](const easyai::ToolCall & c) -> easyai::ToolResult {
        // For nested args, parse the JSON yourself; nlohmann is vendored
        // by llama.cpp at vendor/nlohmann/json.hpp if you want it.
        return easyai::ToolResult::ok("set");
    });
```

### 5.3 Reading arguments without a JSON dependency

`easyai::args::*` are tiny single-level scanners.  They're enough for
~95% of tool authors:

```cpp
std::string  q   = args::get_string_or(c.arguments_json, "q", "");
long long    max = args::get_int_or   (c.arguments_json, "max", 10);
bool         dry = args::get_bool_or  (c.arguments_json, "dry_run", false);
double       t   = args::get_double_or(c.arguments_json, "threshold", 0.5);
bool         has = args::has          (c.arguments_json, "verbose");
```

For nested args (objects, arrays of objects), include
`<nlohmann/json.hpp>` in your handler and parse normally — no easyai
limitation there.

### 5.4 Returning results

```cpp
return easyai::ToolResult::ok("the answer is 42");
return easyai::ToolResult::error("network unreachable");
```

`error` results are tagged `is_error=true` so the streaming layer can
render them differently (`✗` instead of `🔧` in the cli-remote
output).  The model still sees the content — it's just hinted that
the call failed.

Best practices:

* Keep ok-content short and structured (the model reads it as plain
  text; line breaks are fine).
* Truncate raw output to a reasonable budget — 8–16 KB is plenty.
* Format errors as imperative ("missing 'path' argument") — the
  model will often retry with the fix.

### 5.5 Sandboxing

The built-in `fs_*` family takes a root directory and refuses to
escape it (`..` and absolute paths are rejected).  Pattern for your
own filesystem-touching tools:

```cpp
easyai::Tool::builder("read_log")
    .describe("Read the last N lines of a service log under /var/log.")
    .param("name", "string", "Service name (e.g. 'easyai-server.service').", true)
    .param("n",    "integer", "How many lines (max 5000). Default 200.",     false)
    .handle([](const easyai::ToolCall & c) -> easyai::ToolResult {
        std::string name = args::get_string_or(c.arguments_json, "name", "");
        if (name.find('/') != std::string::npos)
            return easyai::ToolResult::error("name must not contain slashes");
        std::filesystem::path p = std::filesystem::path("/var/log") / (name + ".log");
        if (!std::filesystem::exists(p))
            return easyai::ToolResult::error("no log: " + p.string());
        // …tail the file…
        return easyai::ToolResult::ok("…");
    })
    .build();
```

### 5.6 The `Plan` tool

`easyai::Plan` is a checklist with four sub-actions exposed as a
single tool:

```cpp
easyai::Plan plan;
cli.add_tool(plan.tool());      // or engine.add_tool(...)

plan.on_change([](const easyai::Plan & p){
    std::cout << "\n=== plan ===\n";
    p.render(std::cout);          // GitHub-style "- [ ] / [~] / [x]" checklist
});
```

The model is told it can call `plan(action="add"|"start"|"done"|"list", text=…, id=…)`.
On non-trivial multi-step tasks, prompt it to "use the plan tool to
break the task into steps and tick them off as you go" — works
reliably with any tool-call-capable model (Qwen 2.5+, Llama 3+,
DeepSeek, OpenAI o-series, Anthropic Claude via OpenAI-compat
proxies).

You can also seed the plan from your code before letting the model
take over:

```cpp
plan.add("fetch arxiv listing");
plan.add("triage by citation count");
plan.add("draft 5-bullet digest");
```

### 5.7 Cookbook — system observability tools

`examples/cli.cpp` ships four inline `system_*` tools that
read `/proc/*` and report back.  The whole pattern is:

1. Read a `/proc` file with `ifstream`.
2. Parse it (helper functions live in `namespace systools`).
3. Format a human-readable string.
4. Return `ToolResult::ok(text)`.

These tools turn the cli-remote process into an observability agent
that can answer "is the server paging?", "which CPU is hot?", "what
swap device is configured?" — entirely model-driven.  Look at the
file from the top (~line 60) for a guided tour with comments.

To add your own:

* `system_disk_usage` — `df -h` worth of info (read `/proc/mounts`,
  call `statvfs`).
* `system_processes` — `ps`-equivalent (walk `/proc/<pid>/stat`).
* `system_network` — interfaces + traffic counters
  (`/proc/net/dev`).

Copy the existing helpers and ship.

### 5.8 Hot-loading vs. registration

Once you call `cli.add_tool(...)` (or `engine.add_tool(...)`) the
tool is *registered for the lifetime of that object*.  There's no
"unregister" — destroy the Client/Engine to drop them.  This is by
design: the tool list is a property of the conversation contract
(the model was told what's available); changing it mid-flight would
confuse the chat-template renderer.

If you need conditional tools per-conversation, build a fresh
`Client` for that conversation.  `Client` is move-only; constructing
one is cheap (no I/O until `chat()`).

---

## Part 6 — deploying easyai-server

The official path on Linux is `scripts/install_easyai_server.sh`.
Run it from a fresh checkout:

```bash
git clone https://github.com/solariun/easy.git
git clone https://github.com/ggml-org/llama.cpp.git
cd easy
sudo scripts/install_easyai_server.sh \
    --model /path/to/your-model.gguf \
    --webui-title "Box AI" \
    --enable-now
```

It detects the GPU backend (`nvidia-smi` → CUDA, `rocminfo` → ROCm,
`vulkaninfo` / AMD `lspci` → Vulkan, else CPU), builds the right
flavour, installs the libs into `/usr/lib/easyai/` (isolated from
system), creates an `easyai` system user, and drops a hardened
systemd unit with `mlock`, flash-attn, q8_0 KV cache, Bearer auth,
and Prometheus `/metrics`.

### 6.1 Frontend TLS via nginx

`libeasyai-cli` already speaks HTTPS, but easyai-server itself is
plain HTTP by design.  Terminate TLS at nginx:

```nginx
server {
    listen 443 ssl http2;
    server_name ai.example.com;
    ssl_certificate     /etc/letsencrypt/live/ai.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ai.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        # SSE keepalive — agentic loops can run minutes:
        proxy_buffering    off;
        proxy_read_timeout 600;
        proxy_send_timeout 600;
    }
}
```

Then point the client at `https://ai.example.com` and the build's
OpenSSL link will Just Work.

### 6.2 Multiple models

Run one easyai-server per model on different ports
(`--port 8080` / `--port 8081`).  Have your client switch between
them via `cli.endpoint(...)` — there is no notion of "model swap"
inside a single server process by design.

### 6.3 Updating without downtime

```bash
sudo scripts/install_easyai_server.sh --upgrade
```

The script does `git fetch` + `git pull --ff-only` + rebuild +
`systemctl restart easyai-server` in that order.  In-flight SSE
streams are aborted when the old process dies; the client gets a
`HTTP request failed: connection closed` error and can retry.

### 6.4 Backups

Stateless except for whatever you put in `/var/lib/easyai/`
(model files, sandboxed fs_* roots).  Snapshot that directory.

---

## Part 7 — operating the server

### 7.1 Health & metrics

* `GET /health` — JSON status.  Cheap, use it as a liveness probe.
* `GET /metrics` — Prometheus text exposition (only when `--metrics`
  was passed).  Counters: `easyai_requests_total`,
  `easyai_tool_calls_total`, `easyai_errors_total`.
* `GET /props` — full server config snapshot (n_ctx, model alias,
  build info).

### 7.2 Live preset switching

```bash
curl -H 'Content-Type: application/json' \
     -d '{"preset":"creative"}' \
     http://ai.local:8080/v1/preset
```

Or from the lib:

```cpp
cli.set_preset("creative");
```

Affects subsequent requests until changed again.  Per-request
sampling (set in the request body) still wins for that one call.

### 7.3 Verbose mode

`--enable-verbose` (installer flag) or `--verbose` (binary flag) makes
the engine log raw model output, parser actions, and SSE events to
stderr.  Tail it with `journalctl -u easyai-server -f`.

### 7.4 Crash capture

`scripts/install_easyai_server.sh` installs `systemd-coredump` and
sets `LimitCORE=infinity` on the unit.  When the process dies:

```bash
coredumpctl list easyai-server.service
coredumpctl gdb <PID>      # opens gdb on the most recent core
```

---

## Part 8 — performance & tuning

### 8.1 Context size vs. throughput

`-c, --ctx N` sets the model's sequence window.  Bigger ctx = more
KV cache memory per token.  Rule of thumb on Vulkan/RADV with
gfx1035: keep ctx + n_predict ≤ what fits in `--ngl auto`.

### 8.2 KV cache quantisation

`--cache-type-k q8_0 --cache-type-v q8_0` cuts KV memory ~3× vs.
default `f16` with negligible quality loss for chat workloads.  The
installer ships `q8_0` by default.

### 8.3 flash-attn

`--flash-attn` enables fused attention — faster + less memory on
backends that support it.  CUDA and Metal: yes.  Vulkan: works on
RDNA2+ with recent llama.cpp (validated on gfx1035).

### 8.4 mlock

`--mlock` pins the model in RAM so the OS can't page it out under
pressure.  Required on the AI box because GTT-mapped pages would
otherwise be swap candidates.  Needs `LimitMEMLOCK=infinity` in the
systemd unit (the installer sets this).

### 8.5 Sampling

Presets order:
* `deterministic`  — temp 0.0, greedy.  Best for code / format-strict.
* `precise`        — temp 0.2.  Default for tool-call workloads.
* `balanced`       — temp 0.7.  Good general default.
* `creative`       — temp 1.0, top_p 0.95.  Open-ended writing.
* `wild`           — temp 1.4 + relaxed.  Brainstorming, comedy.

Per-request: pin temp + top_p + top_k + min_p in the request body
(via the `--temperature` / `--top-p` / etc. flags on cli-remote, or
the matching `Client::*` setters in code).

### 8.6 Tool budget

`Engine::chat()` caps at 8 tool hops; `Client::chat()` does the
same.  A model that runs away calling tools without converging will
hit the cap and bail out with the last partial answer.  Visible in
verbose mode as `[easyai] hop 7: …`.

---

## Part 9 — recipes (cookbook)

### 9.0 Local vs. remote — pick the right binary

Since the rename, **two independent binaries** cover the two use cases.
No more dual-mode flag juggling on a single binary.

| Binary         | What it loads                  | Library link    |
|----------------|--------------------------------|-----------------|
| `easyai-local` | a local GGUF (no HTTP at all)  | `libeasyai`     |
| `easyai-cli`   | a remote `/v1/chat/completions`| `libeasyai-cli` |

`easyai-cli` (remote) supports the standard TLS + agentic flags:

| Flag             | Effect                                                                |
|------------------|-----------------------------------------------------------------------|
| `--insecure-tls` | Skip peer certificate verification (DEV ONLY, https only).            |
| `--ca-cert <path>` | Trust a custom CA bundle (PEM) for `https://` endpoints.            |

Both binaries share the same preset commands, the same `/help`, and the
same streaming-aware `<think>` stripper (`--no-reasoning` on `easyai-cli`,
`--no-think` on `easyai-local`).

```bash
# point at easyai-server running on the LAN
./build/easyai-cli --url http://10.0.0.5:8080

# point at openai.com (env vars EASYAI_API_KEY also work)
./build/easyai-cli --url https://api.openai.com/v1 \
                   --api-key sk-... \
                   --model gpt-4o-mini

# point at a llama-server / vLLM / ollama endpoint — anything that speaks /v1
./build/easyai-cli --url http://127.0.0.1:11434/v1 --model llama3.1:8b
```

One-shot mode for scripting:

```bash
# Local one-liner; banners go to stderr so capturing stdout is clean.
answer=$(./build/easyai-local -m model.gguf -p "summarise: $(cat file.txt)")

# Remote with reasoning suppressed:
./build/easyai-cli --url http://localhost:8080 --no-reasoning \
    -p "explain BGP route reflectors in two sentences" \
    > brief.md
```

`--no-think` (local) and `--no-reasoning` (remote) strip `<think>…</think>`
(and `<thinking>…</thinking>`) blocks from output. The filter is
streaming-aware and works even when the open or close tag is split across
two model-emitted token chunks.

### 9.1 Web search

`web_search` works out of the box — it talks to DuckDuckGo's HTML endpoint
directly via libcurl. There is nothing to configure and no API key.

If DDG starts rate-limiting your IP (rare), the tool returns an explicit
error message instead of silently failing. If you need a different backend
(Bing, Brave, your own SearXNG), the implementation lives in
`src/builtin_tools.cpp::web_search()` — copy that handler, swap the URL and
the regex pair, and register your variant via `engine.add_tool(my_search())`.

### 9.2 Forcing CPU-only

Pass `--ngl 0` (CLI/server) or `engine.gpu_layers(0)` (lib).

### 9.3 Force-disable a built-in tool

Just don't add it. There is no global "remove" — easyai has no global
state. To run `easyai-local` without any tools at all:

```bash
./build/easyai-local -m … --no-tools
```

For the server: same flag, `--no-tools`.

### 9.4 Production deployment — replacing `llama-server`

`easyai-server` is a drop-in replacement for `llama-server` for almost
every flag a deployment script cares about. A long-running production
launch looks like:

```bash
./build/easyai-server \
    --model      /var/lib/easyai/models/ai.gguf \
    --alias      SolariunAI_Box \
    --host       0.0.0.0 --port 8080 \
    --ctx        128000 \
    --ngl        99 \
    --threads    8  --threads-batch 8 \
    --flash-attn \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --mlock --no-mmap \
    --preset balanced --temperature 0.6 --top-p 0.9 --top-k 20 \
    --api-key    "$EASYAI_API_KEY" \
    --metrics \
    --system-file /etc/easyai/system.txt \
    --sandbox    /var/lib/easyai/workspace
```

Flag map vs. `llama-server`:

| llama-server flag        | easyai-server flag       |
|--------------------------|--------------------------|
| `-m / --model`           | `-m / --model`           |
| `--host` / `--port`      | `--host` / `--port`      |
| `-a / --alias`           | `-a / --alias`           |
| `-c / --ctx-size`        | `-c / --ctx`             |
| `--n-gpu-layers`         | `--ngl`                  |
| `-t / --threads`         | `-t / --threads`         |
| `-tb / --threads-batch`  | `-tb / --threads-batch`  |
| `-fa / --flash-attn`     | `-fa / --flash-attn`     |
| `-ctk / -ctv`            | `-ctk / -ctv`            |
| `--mlock` / `--no-mmap`  | `--mlock` / `--no-mmap`  |
| `--api-key`              | `--api-key`              |
| `--metrics`              | `--metrics`              |
| `--reasoning <on/off>`   | `--reasoning <on/off>`   |
| `--override-kv`          | `--override-kv`          |
| `-np / --parallel`       | accepted; warns since the engine is single-context |

When `--api-key` is set, every `/v1/*` request must carry
`Authorization: Bearer <key>`. `/`, `/health`, and `/metrics` stay open
(useful for liveness probes and Prometheus scrapes).

`/metrics` exposes Prometheus-style counters
(`easyai_requests_total`, `easyai_errors_total`, `easyai_tool_calls_total`)
that you can wire into Grafana or alertmanager.

### 9.5 Behind a reverse proxy

The server speaks plain HTTP and supports CORS. Stick nginx/Caddy in front
to add TLS, auth, and rate limiting. Example Caddyfile:

```
ai.example.com {
    reverse_proxy 127.0.0.1:8080
    basicauth {
        gus  $2a$14$…   # bcrypt hash of password
    }
}
```

### 9.6 Multiple models, one host

Run one `easyai-server` per model on different ports, then add a tiny
proxy that maps `model` field → upstream port. The single-mutex design
inside one server is the right unit; between servers you scale by process.

---

## Part 10 — troubleshooting

### "load failed: failed to load model"

* Did the GGUF download fully? Check the file size; small files often mean
  HTML 404 pages.
* Wrong architecture? llama.cpp prints the supported-arch list during load
  with `--verbose`. Add `engine.verbose(true)`.
* On macOS, run `xcode-select --install` once if Metal headers are missing.

### "context size exceeded"

Conversations grow until the KV cache fills. Either:

* `engine.clear_history()` between turns
* `engine.context(8192)` for a longer window (subject to model training)
* In the server, this can't happen because every request resets the engine.

### Model emits garbled tool calls (e.g. `{{` and `}}`)

Smaller models (≤ 1B parameters) often miss the chat template's tool-call
syntax. easyai catches the parser exception and returns the raw text as the
assistant message. To see what the model emitted, set `engine.verbose(true)`.

Move up to a 3-7B model with native tool-calling support (Qwen2.5-Instruct,
Llama-3.1-Instruct, Mistral-Nemo) and the issue disappears.

### "unknown tool: …"

The model invented a tool name that isn't registered. easyai injects a
`ToolResult::error("unknown tool: …")` into the conversation; usually the
model recovers next turn. If it doesn't, lower the temperature or be
more specific in your system prompt.

### "permission denied" / "path escapes sandbox"

A filesystem tool was called with a path outside the root you passed to
`fs_read_file("…")` etc. By design — pick a wider root or move the file in.

### Server returns 500 with `engine error: …`

Something inside `chat()` threw. The engine remains usable for the next
request (we lock + reset on every call). Check `engine.verbose(true)` and
re-run for stack-level detail in `stderr`.

### `pkill -INT easyai-server` gives exit code 1

It's actually fine — the printed line "stopped cleanly" tells the truth.
Some shells/wrappers report a non-zero code because of the signal, but
`main()` returned 0.

### Forcing the model to ignore the server-injected datetime (QA only)

The server appends an authoritative date/time + knowledge-cutoff
preamble to whichever system message reaches the model
(`--inject-datetime on` is the default).  For regression testing the
preamble can be disabled per-request without restarting the server:

```bash
curl http://ai.local:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -H 'X-Easyai-Inject: off' \
     -d '{"model":"easyai","messages":[{"role":"user","content":"What year is it?"}]}'
```

Header values:
* `off` — skip the preamble for this request only.
* `on`  — force injection on this request even when the server was
          launched with `--inject-datetime off`.
* (anything else, or absent header) — defer to the server flag.

WHY DEFAULT ON: most production deployments want the model to trust
the server clock and to flag post-cutoff facts as uncertain.  Turning
the preamble off removes a real safety net — only do it for A/B QA
runs where you're explicitly comparing pre-injection behaviour.

### `easyai::Client` — "HTTPS endpoint requires OpenSSL"

Configure-time message in the cmake summary:

```
-- easyai-cli: OpenSSL NOT found — HTTPS endpoints will be rejected at runtime
```

Install `libssl-dev` (Debian / Ubuntu) or `openssl-devel` (Fedora /
RHEL), wipe the build dir, and reconfigure.  At runtime the
`Client::endpoint("https://…")` call will then succeed.

If your server uses a self-signed cert, either:

* `cli.tls_insecure(true);` — DEV ONLY, skips peer verification.
* `cli.ca_cert_path("/path/to/ca.pem");` — trust a custom CA bundle.

The `cli-remote` binary exposes the same as `--insecure-tls` and
`--ca-cert PATH`.

### `easyai::Client` — "HTTP request failed: …"

The full text after the colon is what cpp-httplib reported:

* `Connection refused` — the server isn't listening on that
  host/port. Check `--url` value and `nc -vz host port`.
* `SSL handshake failed` — TLS mismatch.  Check the cert hostname
  matches what you're connecting to, the chain is complete, and that
  your client's CA store has the issuer (or pass `--ca-cert`).
* `read timeout` — the model is taking longer than
  `--timeout`.  Bump it (`--timeout 1200`) or raise it in code
  (`cli.timeout_seconds(1200)`).

### `cli-remote` — `Output: 0 / 128000 (0%)` even after a long reply

The cumulative ctx counter on `easyai-server`'s webui needs the new
`ctx_used` field that the server only added in commit `d7f638e`.
On older builds you'll see the per-request count instead — upgrade
the server or just ignore the bar's percentage.

### Model "abandons" `<tool_call>` mid-conversation, emits markdown

After 2-3 successful tool calls some Qwen3 fine-tunes give up on the
XML format and output `*🔧 toolname(args)*` in markdown instead.
Engine recovers automatically (commit `46903e3`); look for this line
in `journalctl -u easyai-server`:

```
[easyai] recovered N tool call(s) from markdown markers (model abandoned <tool_call> syntax — agentic loop continues)
```

If you see the message and the loop continues, you're fine.  If not,
add `--enable-verbose` and check `journalctl` for `[easyai] hop N
raw tail:` lines — those show what the model actually emitted, which
helps tune the system prompt.

### `web_search` — "no results parsed (DuckDuckGo may have rate-limited…)"

DuckDuckGo's HTML endpoint serves a CAPTCHA / "anomaly" page when it
suspects a bot.  Wait a minute, lower request rate, or use a
different network.  No API key option exists — that's the point of
the DDG-HTML approach.

### Conversation feels stale / model insists on outdated info

Knowledge cutoff is real and the model can't tell what date it is
unless told.  Easiest fix: enable `datetime` in the tool list and
prompt it to call that first when in doubt.  An even harder
constraint can be enforced at the server level — see the upcoming
"authoritative datetime injection" feature on easyai-server (commit
soon to follow).

---

## Part 11 — design references

If you want to go deeper:

* `design.md` — internal architecture and "why" decisions, including
  Section 0 (full dependency inventory: llama, cpp-httplib,
  nlohmann::json, libcurl, OpenSSL, …) and Section 5b (the
  OpenAI-protocol client agentic loop).
* `include/easyai/engine.hpp` — every public method of the local
  engine, with doc comments.
* `include/easyai/client.hpp` — every public method of the OpenAI
  client lib, mirroring `engine.hpp` shape.
* `include/easyai/tool.hpp` — `Tool`, `ToolCall`, `ToolResult`,
  `Tool::Builder` (used identically by `Engine` and `Client`).
* `include/easyai/plan.hpp` — `Plan` checklist + `Plan::tool()`
  factory.
* `include/easyai/builtin_tools.hpp` — factories for `datetime`,
  `web_search`, `web_fetch`, `fs_*`.
* `include/easyai/presets.hpp` — sampling presets and the runtime
  override parser (`/temp`, `creative 0.9`, …).
* `src/engine.cpp` — the `chat()` loop is annotated step by step;
  three-layer tool-call recovery (Qwen / Hermes / markdown) lives in
  `parse_assistant`.
* `src/client.cpp` — HTTP/SSE transport, agentic loop mirroring
  `Engine::chat_continue`, request-body assembly with the full
  sampling/penalty surface.
* `src/plan.cpp` — multi-action plan tool with `add/start/done/list`.
* `examples/server.cpp` — the per-request flow is annotated; great
  starting point for a custom HTTP layer.
* `examples/cli.cpp` — REPL + management subcommands +
  inline `system_*` tools, doubles as the cookbook for adding your
  own tool to a `Client`-based agent.
* `scripts/install_easyai_server.sh` — production deployment as a
  hardened systemd unit on Linux (CUDA / ROCm / Vulkan / CPU
  auto-detect, mlock, flash-attn, q8_0 KV).
* `cmake/easyaiConfig.cmake.in` — the find_package shim;
  `find_package(easyai 0.1 REQUIRED)` returns
  `easyai::engine` (libeasyai) and `easyai::cli` (libeasyai-cli) as
  IMPORTED targets your project links against.
* `SESSION_NOTES.md` — running project journal: recent commits,
  pending validations, common pitfalls.  Useful for resuming context
  in a fresh chat.
* `README.md` — top-level pitch + selective-build cheatsheet.

Happy hacking.
