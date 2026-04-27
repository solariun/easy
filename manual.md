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
  fallback)

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
build/easyai-cli      # REPL with built-in tools
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

The `easyai-cli-remote` binary (`examples/cli_remote.cpp`) is a
ready-to-run reference for all of the above — REPL or one-shot, every
sampling knob exposed as a flag, six management subcommands
(`--list-models`, `--list-tools`, `--health`, `--props`, `--metrics`,
`--set-preset NAME`).

---

## Part 4 — recipes

### 4.0 `easyai-cli` against any OpenAI-compatible endpoint

`easyai-cli` runs a REPL in two modes that share the same UI: local model
(`-m model.gguf`) or remote server (`--url <api-base>`). Same preset
commands, same `/help`, same `--no-think` switch.

```bash
# point at easyai-server running on the LAN
./build/easyai-cli --url http://10.0.0.5:8080/v1

# point at openai.com (env vars EASYAI_API_KEY or OPENAI_API_KEY also work)
./build/easyai-cli --url https://api.openai.com/v1 \
                   --api-key sk-... \
                   --remote-model gpt-4o-mini

# point at a llama-server / vLLM / ollama endpoint — anything that speaks /v1
./build/easyai-cli --url http://127.0.0.1:11434/v1 --remote-model llama3.1:8b
```

One-shot mode for scripting:

```bash
# Grab a one-line answer; banners go to stderr so capturing stdout is clean.
answer=$(./build/easyai-cli -m model.gguf -p "summarise: $(cat file.txt)")

# In remote mode with thinking suppressed:
./build/easyai-cli --url http://localhost:8080/v1 --no-think \
    -p "explain BGP route reflectors in two sentences" \
    > brief.md
```

`--no-think` strips `<think>…</think>` (and `<thinking>…</thinking>`) blocks
from output. The filter is streaming-aware and works even when the open or
close tag is split across two model-emitted token chunks.

### 4.1 Web search

`web_search` works out of the box — it talks to DuckDuckGo's HTML endpoint
directly via libcurl. There is nothing to configure and no API key.

If DDG starts rate-limiting your IP (rare), the tool returns an explicit
error message instead of silently failing. If you need a different backend
(Bing, Brave, your own SearXNG), the implementation lives in
`src/builtin_tools.cpp::web_search()` — copy that handler, swap the URL and
the regex pair, and register your variant via `engine.add_tool(my_search())`.

### 4.2 Forcing CPU-only

Pass `--ngl 0` (CLI/server) or `engine.gpu_layers(0)` (lib).

### 4.3 Force-disable a built-in tool

Just don't add it. There is no global "remove" — easyai has no global
state. To run `easyai-cli` without any tools at all:

```bash
./build/easyai-cli -m … --no-tools
```

For the server: same flag, `--no-tools`.

### 4.4 Production deployment — replacing `llama-server`

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

### 4.5 Behind a reverse proxy

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

### 4.6 Multiple models, one host

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
