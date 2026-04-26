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
* **Speculative decoding, RAG, embeddings** — all already in `llama.cpp`,
  but easyai stays out of their way to keep the surface small.

### What changed since the original v0 plan

* **Streaming is in.** The HTTP layer now mirrors `llama-server`'s
  incremental pipeline: every generated token is fed to
  `common_chat_parse(text, is_partial=true, parser_params)`,
  diffed against the previous parsed message via
  `common_chat_msg_diff::compute_diffs()`, and emitted as standard
  OpenAI-shape SSE deltas (`delta.reasoning_content` /
  `delta.content`).  Tool calls surface via the custom
  `easyai.tool_call` / `easyai.tool_result` SSE events plus an inline
  one-line markdown indicator so generic OpenAI clients still see
  *something* when a tool fires.
* **Webui is the llama-server SvelteKit bundle.** Embedded at build
  time via `cmake/xxd.cmake` (`webui/{index.html,bundle.js,
  bundle.css,loading.html}`).  We ship customisations as runtime
  patches: at-startup string substitutions on `bundle.js`, plus
  injected `<script>` blocks that scrub MCP/Sign-in chrome,
  shrink the bundle's native Reasoning panel, and drive a
  per-message status pill from the SSE stream.

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
              │   reasoning_format = AUTO (extract <think> blocks)  │
              └────────────────────────┬────────────────────────────┘
                                       ▼
              ┌─────────────────────────────────────────────────────┐
              │ tokenize, decode (Metal/Vulkan), sample loop         │
              │ (Engine::Impl::generate_until_done)                  │
              │   on_token() fires per piece — used by SSE layer     │
              └────────────────────────┬────────────────────────────┘
                                       ▼  raw assistant text
              ┌─────────────────────────────────────────────────────┐
              │ parse = common_chat_parse(raw, parser_arena)         │
              │   → common_chat_msg { content, reasoning_content,    │
              │                         tool_calls, ... }            │
              └────────────────────────┬────────────────────────────┘
                                       ▼
                       thought-only?  (content empty AND
                       tool_calls empty AND reasoning non-empty)
                          │
                          ├─ yes → discard turn, clear KV,
                          │        fire on_hop_reset, retry
                          │        (up to 2 retries; then fall
                          │        back to promoting reasoning
                          │        → content)
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

A third entry point is used by streaming requests:

* `Engine::chat_continue()` — same multi-hop loop as `chat()` but assumes
  the user message is *already* the last entry in history. Required because
  the server pushes the user message first, then renders
  `chat_params_for_current_state()` to wire the parser, *then* calls into
  the engine. Splitting the entry points avoids the user message being
  pushed twice.

### The thought-only retry path

Some fine-tunes (notably custom Qwen3 trims) sometimes terminate the
turn after `</think>` without emitting either content or a tool_call.
To avoid surfacing an empty bubble to the user, `chat_continue()`
detects that condition and:

1. Throws away the empty turn (does NOT push it to history).
2. Clears the KV cache so the next iteration re-feeds the prompt clean.
3. Fires `on_hop_reset` so the streaming layer can drop its
   `accumulated` raw-text buffer and `prev_msg` diff baseline.
4. Loops back. Sampling is stochastic (`temp > 0`), so the second pass
   typically completes correctly.

A budget of 2 retries is hard-coded. If both pass-throughs are still
thought-only, the engine falls back to promoting `reasoning_content`
into `content` so the user sees the model's thinking instead of an
empty reply. The behaviour is logged when `Engine::verbose(true)`.

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
TokenCallback               on_token;        // per-piece streaming hook
ToolCallback                on_tool;         // post-dispatch tool hook
HopResetCallback            on_hop_reset;    // fired when a hop is discarded
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

The webui shipped is the compiled SvelteKit bundle from `llama-server`,
embedded into the easyai-server binary at build time via
`cmake/xxd.cmake` (one `.hpp` per asset, generated from
`webui/{index.html,bundle.js,bundle.css,loading.html}`).  Total binary
size goes from ~1.5 MB to ~8.3 MB; in exchange we get a polished chat
UI with markdown rendering, code highlighting, preset switching, file
attachments, and per-message stats — all without us maintaining any
of it.

### Customisations: two layers

1. **Build-time string substitution on `bundle.js`** — at server
   startup we patch a few hard-coded llama.cpp brand strings:
   * `>llama.cpp</h1>` → `>{title}</h1>` (sidebar + welcome brand)
   * `llama.cpp - AI Chat Interface` → `{title}` (page title)
   * `Initializing connection to llama.cpp server...` → `... {title} server …`
   * `} - llama.cpp` → `} - {title}` (per-conversation page title)
   * `Type a message...` placeholder, replaced via `--webui-placeholder`

2. **Runtime DOM injection** into the served `index.html`'s `<head>`
   via several `<script>` IIFE blocks:
   * **Title pin** via `Object.defineProperty(document, 'title', {set:})`.
   * **LocalStorage seeding** to disable MCP defaults and force
     `keepStatsVisible=true` / `showMessageStats=true`.
   * **DOM scrubber** — a `MutationObserver` on `<body>` matches
     visible-text NEEDLES (`/^MCP\b/`, `/^Sign in/`, `/Load model/`,
     etc.) and hides their containing card / list-item / dialog so
     unsupported chrome doesn't reach the user.
   * **`fetch` interceptor** that 501s `/authorize`, `/token`,
     `/register`, `/.well-known/*`, `/models/load`, `/cors-proxy`,
     `/dev/poll`, `/home/web_user/*`; stubs `/properties` with `{}`;
     and tees the SSE response of `/v1/chat/completions` into a
     status-pill state machine.
   * **Tone chip + metrics bar** in a Shadow-DOM host
     (`__easyaiBarHost`) attached to `<html>` (so it survives Svelte
     body re-renders) — selector for
     `deterministic / precise / balanced / creative` plus
     `ctx X/Y · last N tok · s · t/s` overview.
   * **Per-message status pill** appended to each assistant action
     toolbar — shows `thinking` / `answering` / `fetching · <tool>` /
     `complete · 98 tok · 4.4s · 22.3 t/s`.
   * **Reasoning-panel shrink** — another `MutationObserver` finds
     `<details>` whose summary text matches `/^Reasoning/i`, applies
     a smaller monospace gray style so the trace doesn't dominate
     the bubble, defaults `open=true` during streaming, and
     auto-collapses on `finish_reason`.
   * **Legacy custom thinking panel** (`__easyai-thinking`) ships
     dormant behind `window.__easyaiCustomThink = false`.  Kept for
     re-enabling on demand if the bundle's native panel ever
     regresses.

### Why the bundle approach

* Zero install footprint — operators get a single `easyai-server`
  binary, no `--www-dir` to remember.
* Existing llama-server users feel at home immediately.
* Markdown, syntax highlighting, multi-attachment chat, etc. are
  hard problems we don't need to solve.

The cost is that the bundle hashes class names on every rebuild, so
*all* customisations must use `aria-label`, `data-testid`, or
visible-text matching.  Never rely on `[class*=…]`.

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
