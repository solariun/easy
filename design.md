# easyai — design

This document explains *why* easyai is shaped the way it is and how its
internal pieces fit together. It assumes you've at least skimmed the
[`README.md`](README.md).

---

## 1. Goals & non-goals

### Goals

1. **Make llama.cpp feel like an SDK.** A C++ developer should be able to
   load a GGUF file and start an agent loop in ten lines, without learning
   the `llama_*` C API or the structure of `common_chat_msg`.
2. **Tools are first-class and trivial to write.** Adding a tool should be
   ≤10 lines and require no JSON-schema knowledge.
3. **Be a credible OpenAI-compatible server.** Anything that posts to
   `POST /v1/chat/completions` should "just work", including clients that
   bring their own `system` prompt and `tools`.
4. **No surprises with memory.** Native resources are owned by RAII types,
   the HTTP server is bounded in payload size, and a single `std::mutex`
   serialises the engine.

### Non-goals (for now)

* **Distributed inference** or batched multi-tenant serving — the engine is
  single-context, single-mutex.
* **Streaming** at the HTTP layer — the OpenAI endpoint returns a full reply
  per request. Streaming is a clean follow-up but would force us to maintain
  a thread-safe SSE plumbing layer that's out of scope for v0.
* **Speculative decoding, RAG, embeddings** — all already in `llama.cpp`,
  but easyai stays out of their way to keep the surface small.

---

## 2. Why we build on top of `common/`, not just `include/llama.h`

llama.cpp ships two layers:

| layer       | header(s)                          | what's there                            |
|-------------|------------------------------------|-----------------------------------------|
| **core**    | `include/llama.h`, `ggml.h`        | model, context, sampling primitives.    |
| **common**  | `common/common.h`, `common/chat.h` | high-level helpers: `common_init_from_params`, Jinja chat templates, OpenAI-shape parsing, JSON-schema-to-grammar, PEG-based tool-call parser. |

Building tool-calling on the core layer alone would mean re-implementing
Jinja templating, the per-model tool-call grammar, and a JSON-schema parser.
That work already exists in `common/`, so we link against it.

The trade-off: `llama-common` is a moving target (it's a library only
internally). We pin our implementation to a sibling clone of `llama.cpp` and
update both together.

---

## 3. End-to-end data flow

```
┌────────────────┐     user msg      ┌──────────────────────────┐
│ caller (CLI/   │ ────────────────▶ │   Engine::chat(text)     │
│ HTTP / lib)    │                   └────────────┬─────────────┘
└────────────────┘                                │
                                                  ▼
              ┌─────────────────────────────────────────────────────┐
              │ render = common_chat_templates_apply(history+tools) │
              └────────────────────────┬────────────────────────────┘
                                       ▼
              ┌─────────────────────────────────────────────────────┐
              │ tokenize, decode (Metal/Vulkan), sample loop         │
              │ (Engine::Impl::generate_until_done)                  │
              └────────────────────────┬────────────────────────────┘
                                       ▼  raw assistant text
              ┌─────────────────────────────────────────────────────┐
              │ parse = common_chat_parse(raw, parser_arena)         │
              │   → common_chat_msg { content, tool_calls, ... }     │
              └────────────────────────┬────────────────────────────┘
                                       ▼
                              tool_calls.empty() ?
                                yes ──▶ return content
                                no  ──▶ for each call: dispatch + push
                                                         ┌─ tool result ─┐
                                                         ▼               │
                                                   loop back ────────────┘
                                                   (max 8 hops by default)
```

Two single-pass exits exist for the HTTP server:

* `Engine::generate_one()` — runs one render+decode+parse cycle, appends the
  result to history, and returns the parsed `GeneratedTurn` so the caller
  can inspect tool calls and *forward them to a remote client* without
  dispatching them locally.
* `Engine::push_message(role, content, tool_name, tool_call_id)` — append a
  message to the history without generating. Used by the HTTP server to
  rebuild the conversation per request and by client-side tool-result
  feeding.

---

## 4. The `Engine` class

### Public surface (fluent)

```cpp
Engine().model("…").context(4096).gpu_layers(99)
        .system("…").temperature(0.7).top_p(0.95)
        .add_tool(…).on_token(…).load();
```

* All setters return `Engine &` so they chain.
* Setters are *staged* — they only modify the internal `common_params`
  struct; the model, context, and sampler are built when `.load()` is called.
* After `load()`, `set_sampling()` rebuilds the sampler in place. Other
  setters (model path, context size) require a fresh Engine.

### `Engine::Impl` (pimpl)

Holds the four llama.cpp resources and our extras:

```
common_params               params;          // mutated by setters
common_init_result_ptr      init;            // model + context (RAII)
common_chat_templates_ptr   templates;       // Jinja templates (RAII)
common_sampler            * sampler;         // freed in dtor
std::vector<common_chat_msg> history;        // conversation
std::vector<Tool>            tools;          // registered tools
TokenCallback               on_token;
ToolCallback                on_tool;
```

### KV-cache handling

We use `llama_memory_seq_pos_max(seq=0) + 1` as `n_past`. When we render the
prompt for a new turn, we tokenize the *full* current prompt and feed only
the suffix beyond `n_past` to `llama_decode`. This is the simplest correct
behaviour across all model architectures (recurrent / hybrid models can't
remove tokens from cache).

If `replace_history` is called we wipe the KV cache via
`llama_memory_clear(true)` so we never feed misaligned tokens.

---

## 5. Tools & schema generation

A `Tool` is just:

```cpp
struct Tool {
    std::string name;
    std::string description;
    std::string parameters_json;   // JSON-schema (object)
    ToolHandler handler;           // std::function<ToolResult(const ToolCall&)>
};
```

The `Tool::Builder` pattern emits the JSON-schema for you so callers don't
need to know the schema spec:

```cpp
Tool::builder("read_file")
    .describe("Read a UTF-8 file")
    .param("path",   "string",  "Path to the file", true)
    .param("offset", "integer", "Skip this many bytes", false)
    .handle([](const ToolCall & c) { … })
    .build();
```

The generated schema is the minimal `{"type":"object","properties":{…},"required":[…]}`
that satisfies most chat-template tool-call grammars. Power users that want
nested objects, enums, or `$ref`s can call `Tool::make(name, desc, schema_json, handler)`
directly with their own schema string.

### Argument parsing helpers

Handlers receive the raw `arguments_json` from the model. The library
ships `easyai::args::get_string / get_int / get_double / get_bool` —
deliberately single-level scanners that don't pull a JSON dependency into
your handler code. For nested args, include `nlohmann/json.hpp` yourself
(it's vendored by llama.cpp).

---

## 6. The HTTP server

The server is **one-engine**, **one-mutex**, **one-process**. No connection
pool, no engine pool, no warmup workers. That's enough to compete with
`llama-server` on a single-user machine and is straightforward to scale by
running N processes behind a load balancer.

### Per-request flow

```
┌──────── POST /v1/chat/completions ─────────┐
│ 1. Parse JSON body                         │
│ 2. acquire engine_mu                        │
│ 3. reset_engine_defaults() — system, tools, │
│    sampling all back to ambient defaults    │
│ 4. If body.tools present → swap tools for   │
│    stub-handler shells (no local dispatch)  │
│ 5. Apply per-request sampling overrides     │
│ 6. Peel off any preset prefix in last user  │
│    message ("creative 0.9 …")               │
│ 7. replace_history(messages[:-1])           │
│ 8. If tools came from request:              │
│      generate_one() → return tool_calls     │
│    Else (server tools):                     │
│      chat(last_user) → loops until done     │
│ 9. Build OpenAI envelope, respond           │
│10. release engine_mu                        │
└────────────────────────────────────────────┘
```

### "Server-as-competitor" semantics

Two override points:

| What the request brings   | What the server does                                                 |
|---------------------------|-----------------------------------------------------------------------|
| `system` message present  | use it; ignore `system.txt`                                           |
| `system` message absent   | inject `system.txt` as message[0]                                     |
| `tools` array present     | register stubs; *forward* tool_calls back to client (single-pass)     |
| `tools` array absent      | use built-in toolbelt; *dispatch* server-side (multi-hop loop)        |
| `temperature` etc present | apply for this request                                                |
| `temperature` etc absent  | use ambient preset                                                    |

A client like Claude Code can use the server in two completely different
modes — bring-your-own-everything, or trust the server defaults — without
any configuration switch.

### Why per-request `replace_history` instead of incremental append?

Stateless requests are easier to reason about. The cost is that we re-decode
the prompt every time, but llama.cpp's KV cache lookup is fast (we feed only
the suffix beyond what's already cached, when caching across requests is
possible). Trading a little perf for *no chance of cross-request leakage* is
worth it for v0.

### CORS

Permissive (`*`) by default so a static HTML page on `file://` or another
origin can talk to the server. Tighten via a reverse proxy if exposing on a
network you don't fully control.

### What "stop" looks like

We trap `SIGINT` and `SIGTERM`, the handler calls `httplib::Server::stop()`
which causes `listen()` to return; main() returns 0. No threads, no engine
calls happen in the signal handler — only `Server::stop()` is signal-safe-ish
under cpp-httplib.

---

## 7. The webui

A single `<500-line` HTML file embedded in `server.cpp` as a `constexpr char[]`.
It POSTs full conversation history each turn (stateless, like the OpenAI API),
includes a preset bar that hits `POST /v1/preset`, and renders raw text plus
any returned tool_calls.

Embedded inline because:

* one binary, no install path issues
* no `--www-dir` flag to forget
* no risk of serving an arbitrary file by mistake

If you want a richer UI (markdown, copy buttons, code highlighting, file
upload), point a real frontend at `/v1/chat/completions` — the UI in-binary
is intentionally minimal.

---

## 8. Memory & failure model

### Resource ownership

| resource                      | owned by                                | freed when                             |
|-------------------------------|-----------------------------------------|----------------------------------------|
| `llama_model`, `llama_context`| `common_init_result_ptr` (unique_ptr)   | `Engine::Impl` dtor                    |
| `common_chat_templates`       | `common_chat_templates_ptr` (unique_ptr)| `Engine::Impl` dtor                    |
| `common_sampler`              | raw pointer + manual free               | `Engine::Impl` dtor                    |
| `Engine::Impl`                | `unique_ptr<Impl>`                      | `Engine` dtor                          |
| HTTP server                   | `httplib::Server` (stack)               | `main()` return                        |
| `ServerCtx`                   | `unique_ptr<ServerCtx>`                 | `main()` return                        |
| Per-request strings/JSON      | stack / `nlohmann::json`                | end of handler                         |

### Failure modes & responses

| failure                                  | response                                                                                         |
|------------------------------------------|---------------------------------------------------------------------------------------------------|
| Malformed JSON request                   | 400 + OpenAI error envelope                                                                       |
| `messages` missing / empty               | 400 + descriptive error                                                                           |
| Engine throws during generation          | 500 + error envelope; engine remains usable                                                       |
| Chat-template parser throws (model bug)  | Caught in `parse_assistant`; raw text returned as content; finish_reason="stop"                   |
| Tool handler throws                      | Caught in chat loop; result becomes `ToolResult::error("tool threw: …")`; agent continues         |
| Unknown tool called by model             | `ToolResult::error("unknown tool: …")` injected; agent continues                                  |
| Context overflow during decode           | Engine sets `last_error`, returns partial output; subsequent calls require `clear_history`        |
| Request body > `--max-body`              | httplib aborts the request before we see it                                                       |
| `SIGINT` mid-generation                  | CLI flips a flag; second SIGINT exits hard. Server: stop() then orderly exit                      |

---

## 9. What changes when llama.cpp updates?

Most changes are absorbed automatically because we use `add_subdirectory()`.
Things to watch:

* **Sampler API churn** — we use `common_sampler_init / sample / accept`. If
  fields move under `common_params_sampling`, `set_sampling()` may need a
  patch.
* **Chat-template format** — new `common_chat_format` enum values can land
  any time. Unknown formats fall back through our `parse_assistant` try/catch
  and the assistant text is returned as plain content.
* **`common_init_from_params`** — its signature is stable across recent
  releases; if it grows, we mirror via the same setter→params plumbing.

The recommended workflow is to pin both `easyai/` and `llama.cpp/` as
git submodules in your application repo so an upgrade is a single commit.
