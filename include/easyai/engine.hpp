// easyai/engine.hpp — high-level llama.cpp wrapper for building agent engines.
//
// Usage:
//
//   easyai::Engine engine;
//   engine.model("models/qwen2.5-0.5b.gguf")
//         .gpu_layers(99)               // -1 == all (default), 0 == CPU only
//         .context(4096)
//         .system("You are a concise assistant.")
//         .temperature(0.8f)
//         .top_p(0.95f)
//         .add_tool(easyai::tools::datetime())
//         .add_tool(easyai::tools::web_fetch())
//         .load();
//
//   engine.on_token([](const std::string & piece){ std::cout << piece; });
//   std::string reply = engine.chat("What time is it in Tokyo?");
//
#pragma once

#include "tool.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Forward-declared at global namespace so chat_params_for_current_state can
// return one without forcing every easyai::Engine consumer to include the
// hefty common/chat.h.  Callers that actually use the result include it
// themselves.
struct common_chat_params;

namespace easyai {

using TokenCallback    = std::function<void(const std::string & piece)>;
using ToolCallback     = std::function<void(const ToolCall &, const ToolResult &)>;
// Fires at the start of every multi-hop iteration AFTER the first.
// Streaming layers register this to drop their per-turn accumulators
// (e.g. the running raw-text buffer + previous parsed message used to
// compute SSE diffs) so that retries (thought-only loops, tool-result
// follow-ups) don't bleed state across turns.
using HopResetCallback = std::function<void()>;

class Engine {
   public:
    Engine();
    ~Engine();

    Engine(const Engine &)             = delete;
    Engine & operator=(const Engine &) = delete;
    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    // ---------------- configuration (chainable, take effect on load()) -----
    Engine & model         (std::string gguf_path);
    Engine & context       (int n_ctx);            // default 4096
    Engine & batch         (int n_batch);          // default = n_ctx
    Engine & gpu_layers    (int n);                // -1 = all, 0 = CPU only
    Engine & threads       (int n);                // default = hw threads
    Engine & seed          (uint32_t s);           // default = random
    Engine & system        (std::string prompt);
    Engine & temperature   (float t);              // default 0.7
    Engine & top_p         (float p);              // default 0.95
    Engine & top_k         (int   k);              // default 40
    Engine & min_p         (float p);              // default 0.05
    Engine & repeat_penalty(float r);              // default 1.1
    Engine & max_tokens    (int   n);              // per chat() call, -1 = until ctx
    Engine & tool_choice_auto    ();
    Engine & tool_choice_required();
    Engine & tool_choice_none    ();
    Engine & parallel_tool_calls (bool enable);    // default false
    Engine & verbose       (bool on);              // default false

    // Agentic-loop safety cap: how many tool round-trips chat() will run
    // before bailing out.  Default 8 — fine for small research flows.
    // Bash flows often need many more (compile → run → fix → re-run);
    // when --allow-bash is in play, callers bump this to ~99999.
    Engine & max_tool_hops (int n);

    // Auto-retry-with-nudge for "incomplete" turns — model finished without
    // a tool_call AND emitted only a tiny visible reply (typical
    // "Let me search…" / "I'll do that now" announce-without-action
    // pattern).  When enabled (default ON) chat_continue discards the bad
    // assistant turn, appends a corrective synthetic user message
    // ("don't announce, execute"), fires on_hop_reset so streaming
    // consumers can drop their accumulated parse state, and retries once.
    // Mirrors libeasyai-cli's Client::retry_on_incomplete behaviour at the
    // engine layer so every consumer (server, agent, recipes) gets the
    // same recovery without each app rolling its own.
    Engine & retry_on_incomplete (bool on = true);

    // ---------------- KV cache & model overrides ----------------------------
    // KV cache data type — accepts ggml_type names: "f32", "f16", "bf16",
    // "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl". Lower precision
    // dramatically cuts VRAM / RAM at a small quality cost.  Defaults to f16.
    // Invalid names are recorded in last_error() and otherwise ignored.
    Engine & cache_type_k (const std::string & ggml_type_name);
    Engine & cache_type_v (const std::string & ggml_type_name);
    // Keep KV cache on CPU even when layers are on GPU — useful when VRAM is
    // tiny.  Trades GPU bandwidth for capacity.
    Engine & no_kv_offload(bool on = true);
    // Use a single unified KV buffer across the input sequences when computing
    // attention (recent llama.cpp feature; mostly for speculative + parallel).
    Engine & kv_unified   (bool on = true);
    // Override a key-value entry in the loaded GGUF.  Format:
    //   "key=int:42"        "key=float:0.75"
    //   "key=bool:true"     "key=str:hello"
    // Repeatable.  Useful for fixing tokenizer or rope parameters at load time.
    Engine & add_kv_override(const std::string & spec);

    // ---------------- compute / memory knobs --------------------------------
    // Flash attention — auto, on, off.  Default 'auto' lets llama.cpp decide
    // based on backend capability.  Pass true to force on.
    Engine & flash_attn   (bool on = true);
    // Pin model weights in physical memory so they aren't paged out.  Costs
    // RAM but improves latency consistency on heavily-loaded hosts.
    Engine & use_mlock    (bool on = true);
    // Set to false to disable mmap and read the GGUF straight into RAM.
    // Slightly slower start-up; sometimes needed on network filesystems.
    Engine & use_mmap     (bool on = true);
    // Separate thread pool size for batch (prompt-eval) compute.
    Engine & threads_batch(int n);
    // NUMA strategy: "distribute", "isolate", "numactl", "" (default off).
    Engine & numa         (const std::string & strategy);

    // ---------------- reasoning / thinking ----------------------------------
    // Toggle the chat-template `enable_thinking` flag (used by Qwen3, R1,
    // etc.). Default ON: the model sees thinking as enabled and may emit
    // <think> blocks. Pass false to ask the model not to think aloud.
    Engine & enable_thinking(bool on = true);

    // ---------------- tools -------------------------------------------------
    Engine & add_tool   (Tool t);
    Engine & clear_tools();

    // ---------------- callbacks --------------------------------------------
    Engine & on_token     (TokenCallback    cb);
    Engine & on_tool      (ToolCallback     cb);
    Engine & on_hop_reset (HopResetCallback cb);

    // ---------------- lifecycle --------------------------------------------
    bool load();              // loads gguf + builds context. returns true on success.
    bool is_loaded() const;
    void reset();             // wipes conversation history + KV cache.
    void clear_kv();          // wipes ONLY the KV cache (history kept intact).
                              // useful between retry attempts so the next
                              // generate_one() re-renders the same history
                              // with a fresh starting state.

    // ---------------- runtime sampler reconfig ------------------------------
    // The sampler is built at load() time. Use this to re-create it with new
    // values mid-conversation (server preset switching, /temp commands).
    // Pass any negative value to leave that field untouched.
    Engine & set_sampling(float temperature  = -1.0f,
                          float top_p        = -1.0f,
                          int   top_k        = -1,
                          float min_p        = -1.0f);

    // ---------------- conversation primitives -------------------------------
    // Push a message of any role onto the history WITHOUT generating.
    // Useful for replaying full conversations (HTTP server) or seeding the
    // chat with assistant priming. tool_name / tool_call_id are only
    // meaningful when role == "tool".
    Engine & push_message(std::string role,
                          std::string content,
                          std::string tool_name    = "",
                          std::string tool_call_id = "");

    // Replace the entire history at once. The first system message coming
    // from .system() is kept as a baseline only when `messages` does not
    // already contain one.
    void replace_history(const std::vector<std::pair<std::string, std::string>> & messages);

    // Direct access for advanced HTTP scenarios.
    void clear_history();

    // Pop the last N entries off history (no-op if N >= history.size).
    // Used by callers that need to rollback a synthetic nudge they
    // pushed before a retry, so the conversation handed back to the
    // client doesn't leak internal-only messages.
    void pop_last(size_t n = 1);

    // ---------------- inference --------------------------------------------
    // chat() runs a full turn including any tool-call/tool-result loops.
    // The returned string is the final assistant message content.
    std::string chat(const std::string & user_message);

    // Same as chat() but assumes the user message is ALREADY the last
    // entry in history (e.g. you called push_message("user", ...) before
    // and want to render+inspect the chat template's params without
    // duplicating the user message).  Used by the HTTP streaming path
    // that needs the rendered common_chat_params at on_token wiring time.
    std::string chat_continue();

    // Generate exactly ONE assistant turn from the current history. Returns
    // the parsed message (so the caller can inspect tool_calls and decide
    // whether to dispatch them or forward them to a remote client).
    //
    // The message is appended to the history so subsequent generate_one()
    // calls see it. If the model emitted tool calls and you want to feed in
    // tool results, use push_message("tool", result, name, id) and call
    // generate_one() again.
    struct GeneratedTurn {
        std::string                       content;
        std::string                       reasoning;
        std::vector<std::pair<std::string /*name*/, std::string /*args_json*/>> tool_calls;
        std::vector<std::string>          tool_call_ids;
        std::string                       finish_reason;  // "stop" | "tool_calls" | "length" | "error"
    };
    GeneratedTurn generate_one();

    // Lower-level: just generate raw text from current state.
    std::string generate();

    // ---------------- introspection -----------------------------------------
    std::string                 last_error()        const;
    int                         turns()             const;
    const std::vector<Tool>   & tools()             const;
    std::string                 backend_summary()   const;  // e.g. "Metal (GPU)"
    int                         n_ctx()             const;  // configured context window
    std::string                 model_path()        const;

    // Performance counters (cumulative since the last perf_reset).  Used by
    // the HTTP server to emit per-request timings in the SSE stream that the
    // webui renders (tokens/s, prompt/gen times, KV cache pressure).
    struct PerfData {
        int    n_prompt_tokens    = 0;   // tokens in the prompt that needed eval
        int    n_predicted_tokens = 0;   // tokens generated
        double prompt_ms          = 0.0; // total prompt-eval wall time
        double predicted_ms       = 0.0; // total generation wall time
        int    n_ctx_used         = 0;   // tokens currently in the KV cache
    };
    PerfData perf_data()  const;
    void     perf_reset();

    // Render the chat-template state for the current history+tools and
    // return the resulting common_chat_params.  Exposed so the HTTP layer
    // can build a parser (with the right PEG arena + reasoning_format)
    // and call common_chat_parse incrementally during streaming —
    // matching how llama-server splits reasoning_content from content.
    // Pass true to include the assistant generation prompt suffix.
    //
    // Returns a global-namespace common_chat_params (fully qualified to
    // dodge ADL into our own easyai:: namespace).
    ::common_chat_params chat_params_for_current_state(bool add_generation_prompt = true) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace easyai
