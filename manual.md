# easyai — developer manual

This is the **hands-on** guide. It assumes nothing beyond "I can compile a
C++17 program". By the end you will know how to:

* compile easyai and download a model
* run `easyai-cli` and talk to it
* host `easyai-server` and call it from Claude Code, OpenAI SDKs, or curl
* embed `easyai::Engine` in your own program
* write a custom tool, with typed parameters
* tune the sampler with presets and runtime overrides
* debug common issues (context overflow, malformed tool calls, GPU
  fallback, SearXNG configuration)

---

## Part 1 — getting set up

### 1.1 Layout

easyai expects llama.cpp as a sibling directory:

```
develop/
├── easyai/        # this project
├── llama.cpp/     # https://github.com/ggml-org/llama.cpp
└── searxng/       # optional, only needed for the web_search tool
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
| A SearXNG host  | `web_search`              |

On macOS:

```bash
brew install cmake curl
```

### 1.3 First build

```bash
cd easyai
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Outputs land in `build/`:

```
build/easyai-cli      # REPL with built-in tools
build/easyai-server   # HTTP server + webui
build/easyai-agent    # demo agent (every tool + a custom one)
build/easyai-chat     # bare REPL (no tools)
build/libeasyai.dylib # the library
```

If the configure step says `easyai: libcurl found — web_fetch / web_search enabled`,
you can skip the SearXNG section if you only care about `web_fetch`.

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
./build/easyai-cli -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

You'll see something like:

```
[easyai-cli] loaded models/qwen2.5-1.5b-instruct-q4_k_m.gguf
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
./build/easyai-server -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf -s system.txt
```

Where `system.txt` (any text file) becomes the **default** system prompt
for any request that doesn't already include one.

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
engine.add_tool(easyai::tools::fs_read_file("./workspace"));
engine.add_tool(easyai::tools::fs_glob     ("./workspace"));
engine.add_tool(easyai::tools::fs_grep     ("./workspace"));
```

Paths sent by the model are resolved (relative or absolute) and rejected if
they escape the root via `..` or symlinks. The check uses
`std::filesystem::weakly_canonical` and a string-prefix comparison — safe
against path-traversal in current macOS/Linux/Win semantics.

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

### 3.8 The `generate_one()` escape hatch

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

---

## Part 4 — recipes

### 4.1 SearXNG locally

```bash
cd ../searxng
make run                       # follow the project's quickstart
# Then make sure JSON is enabled in settings.yml:
#   search:
#     formats:
#       - html
#       - json
```

Tell easyai where it is:

```bash
export EASYAI_SEARXNG_URL=http://127.0.0.1:8080
./build/easyai-cli -m … 
```

(Or set it inside your shell profile.)

### 4.2 Forcing CPU-only

Pass `--ngl 0` (CLI/server) or `engine.gpu_layers(0)` (lib).

### 4.3 Force-disable a built-in tool

Just don't add it. There is no global "remove" — easyai has no global
state. To run `easyai-cli` without any tools at all:

```bash
./build/easyai-cli -m … --no-tools
```

For the server: same flag, `--no-tools`.

### 4.4 Behind a reverse proxy

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

### 4.5 Multiple models, one host

Run one `easyai-server` per model on different ports, then add a tiny
proxy that maps `model` field → upstream port. The single-mutex design
inside one server is the right unit; between servers you scale by process.

---

## Part 5 — troubleshooting

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

---

## Part 6 — design references

If you want to go deeper:

* `design.md` — internal architecture and "why" decisions.
* `include/easyai/engine.hpp` — every public method, with doc comments.
* `src/engine.cpp` — the `chat()` loop is annotated step by step.
* `examples/server.cpp` — the per-request flow is annotated; great
  starting point for a custom HTTP layer.

Happy hacking.
