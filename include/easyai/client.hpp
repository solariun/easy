// easyai/client.hpp — OpenAI-protocol counterpart of Engine.
//
// Same fluent API and tool model as the local Engine, but the model
// itself runs on a remote server (any /v1/chat/completions endpoint:
// our easyai-server, an upstream llama-server, OpenAI itself, etc.).
// Tools execute LOCALLY in the consumer process — the model picks
// which tool to call, the Client dispatches it, and the result is
// fed back into the conversation.
//
// Intended use:
//
//   easyai::Client cli;
//   cli.endpoint("http://ai.local")
//      .api_key(getenv("EASYAI_KEY"))
//      .model("EasyAi")
//      .system("You are a planning agent.");
//   cli.add_tool(easyai::tools::web_search());
//   cli.add_tool(easyai::tools::web_fetch());
//   cli.on_token([](const std::string & p){ std::fputs(p.c_str(), stdout); });
//   std::string answer = cli.chat("summarise today's arxiv ml posts");
//
// The class is move-only (one outstanding HTTP transport per instance).
#pragma once

#include "easyai/tool.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace easyai {

// Lightweight description of a remote model returned by /v1/models.
struct RemoteModel {
    std::string id;        // e.g. "EasyAi", "gpt-4o-mini"
    std::string owned_by;  // e.g. "easyai", "openai"
    long        created = 0;
};

// Lightweight description of a remote tool returned by /v1/tools
// (an easyai-server extension; not present on stock OpenAI).
struct RemoteTool {
    std::string name;
    std::string description;
};

class Client {
public:
    Client();
    ~Client();
    Client(const Client &)             = delete;
    Client & operator=(const Client &) = delete;
    Client(Client &&) noexcept;
    Client & operator=(Client &&) noexcept;

    // ----- transport / auth (fluent) ---------------------------------------
    Client & endpoint        (std::string url);             // http(s)://host[:port]
    Client & api_key         (std::string key);             // Bearer
    Client & timeout_seconds (int  s);                      // connect+read; default 600
    Client & verbose         (bool v);                      // log SSE lines to stderr

    // ----- request shape (fluent) ------------------------------------------
    // Every sampling/penalty knob below maps directly to the matching
    // OpenAI / llama-server / easyai-server field; -1.0f / -1 / "" are
    // "unset, server picks the default".  Multiple knobs can be pinned
    // at once and the request body only includes the ones you set.
    Client & model              (std::string id);            // request body field
    Client & system             (std::string prompt);        // 0..N system msgs
    Client & temperature        (float t);
    Client & top_p              (float v);
    Client & top_k              (int   v);
    Client & min_p              (float v);                   // llama-server / easyai
    Client & repeat_penalty     (float v);                   // llama-server / easyai
    Client & frequency_penalty  (float v);                   // OpenAI standard
    Client & presence_penalty   (float v);                   // OpenAI standard
    Client & seed               (long long s);               // -1 = randomise
    Client & max_tokens         (int   n);
    Client & stop               (std::vector<std::string> sequences);
    // Free-form passthrough for fields the public setters above don't cover.
    // The string MUST be a valid JSON object literal; its keys are merged
    // into the request body verbatim.  Useful for non-standard server
    // extensions (e.g. {"reasoning_effort":"high"}).
    Client & extra_body_json    (std::string raw_json);

    // ----- tool registration (mirrors Engine) ------------------------------
    Client & add_tool        (Tool t);
    Client & clear_tools     ();
    const std::vector<Tool> & tools() const;

    // ----- streaming callbacks ---------------------------------------------
    using TokenCallback = std::function<void(const std::string &)>;
    using ToolCallback  = std::function<void(const ToolCall &, const ToolResult &)>;

    Client & on_token  (TokenCallback);   // delta.content (visible reply)
    Client & on_reason (TokenCallback);   // delta.reasoning_content (thinking)
    Client & on_tool   (ToolCallback);    // every dispatched tool round-trip

    // ----- chat ------------------------------------------------------------
    // chat() pushes the user message, runs the agentic multi-hop loop
    // until the model emits a non-tool finish_reason, and returns the
    // final visible content.  chat_continue() is the same minus the
    // user push (for callers that want to inject tool results manually).
    std::string chat          (const std::string & user_message);
    std::string chat_continue ();
    void        clear_history ();

    // ----- direct endpoints (optional helpers) -----------------------------
    // Each method returns false on transport / HTTP failure; on success
    // the parsed value is written to the out parameter.  See last_error()
    // for diagnostic detail.
    bool list_models      (std::vector<RemoteModel> & out);
    bool list_remote_tools(std::vector<RemoteTool>  & out);   // /v1/tools
    bool health           ();                                  // GET /health
    bool metrics          (std::string & out_text);            // Prometheus
    bool props            (std::string & out_json);            // raw JSON
    bool set_preset       (const std::string & preset_name);   // /v1/preset

    // ----- introspection ---------------------------------------------------
    std::string last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace easyai
