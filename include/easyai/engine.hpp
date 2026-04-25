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

namespace easyai {

using TokenCallback = std::function<void(const std::string & piece)>;
using ToolCallback  = std::function<void(const ToolCall &, const ToolResult &)>;

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

    // ---------------- tools -------------------------------------------------
    Engine & add_tool   (Tool t);
    Engine & clear_tools();

    // ---------------- callbacks --------------------------------------------
    Engine & on_token (TokenCallback cb);
    Engine & on_tool  (ToolCallback  cb);

    // ---------------- lifecycle --------------------------------------------
    bool load();              // loads gguf + builds context. returns true on success.
    bool is_loaded() const;
    void reset();             // wipes conversation history + KV cache.

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

    // ---------------- inference --------------------------------------------
    // chat() runs a full turn including any tool-call/tool-result loops.
    // The returned string is the final assistant message content.
    std::string chat(const std::string & user_message);

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

   private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace easyai
