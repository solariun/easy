// src/mcp.cpp — MCP server implementation.
//
// JSON-RPC 2.0 dispatcher for a stateless POST /mcp endpoint.
// Threat model and protocol surface are documented in the public
// header. This file is the enforcement of those contracts.
//
// Design choices worth calling out:
//
//   * The dispatcher is a pure function. No mutex, no global
//     state, no I/O. Everything it needs comes through the
//     `tools` and `info` arguments. Concurrency is the caller's
//     problem (httplib's worker threads share `default_tools`
//     read-only after startup; tool dispatch is already mutex-
//     guarded inside the engine layer).
//
//   * Errors never throw. Every code path that could blow up —
//     JSON parse, missing field, type mismatch, tool not found,
//     handler exception — is caught and converted to a proper
//     JSON-RPC error envelope. Returning an exception across the
//     httplib boundary would have been caught there anyway, but
//     surfacing it as JSON-RPC is more informative for clients.
//
//   * Each tool's `parameters_json` is parsed-then-embedded as the
//     MCP `inputSchema`. We assume the schema is already valid
//     JSON (every Tool::Builder builds it from a typed C++
//     description). If parse fails for any reason, we fall back to
//     a permissive empty schema rather than dropping the tool.

#include "easyai/mcp.hpp"
#include "easyai/tool.hpp"

#include <nlohmann/json.hpp>

#include <exception>
#include <string>
#include <vector>

namespace easyai::mcp {

namespace {

using json = nlohmann::json;

// JSON-RPC 2.0 error codes. -32000 through -32099 are reserved for
// application-defined server errors; we use one of those for
// "tool execution failed".
constexpr int kErrParse           = -32700;
constexpr int kErrInvalidRequest  = -32600;
constexpr int kErrMethodNotFound  = -32601;
constexpr int kErrInvalidParams   = -32602;
constexpr int kErrInternal        = -32603;
constexpr int kErrToolFailed      = -32000;
constexpr int kErrToolNotFound    = -32001;

// Build a JSON-RPC 2.0 error envelope. `id` is whatever the
// request's id was (passed through), or null if the request was
// unparseable.
json make_error(const json & id, int code, const std::string & message,
                json data = json()) {
    json e;
    e["code"]    = code;
    e["message"] = message;
    if (!data.is_null()) e["data"] = std::move(data);

    json env;
    env["jsonrpc"] = "2.0";
    env["id"]      = id.is_null() ? json(nullptr) : id;
    env["error"]   = std::move(e);
    return env;
}

json make_result(const json & id, json result) {
    json env;
    env["jsonrpc"] = "2.0";
    env["id"]      = id;
    env["result"]  = std::move(result);
    return env;
}

// Parse a tool's `parameters_json` into a JSON object suitable for
// MCP's `inputSchema`. Falls back to `{ "type": "object" }` if the
// stored schema can't be parsed (defence against a hand-crafted
// Tool with a malformed schema string).
json tool_input_schema(const Tool & t) {
    if (t.parameters_json.empty()) {
        return json{{"type", "object"}};
    }
    try {
        json s = json::parse(t.parameters_json);
        if (!s.is_object()) {
            return json{{"type", "object"}};
        }
        return s;
    } catch (...) {
        return json{{"type", "object"}};
    }
}

// Render one tool as an MCP tool descriptor (the per-entry shape
// of the array `tools/list` returns).
json tool_descriptor(const Tool & t) {
    json e;
    e["name"]        = t.name;
    e["description"] = t.description;
    e["inputSchema"] = tool_input_schema(t);
    return e;
}

// `initialize`: advertise capabilities + serverInfo.
json handle_initialize(const json & /*params*/, const ServerInfo & info) {
    json caps;
    json tools_cap;
    tools_cap["listChanged"] = false;   // no hot-reload notifications
    caps["tools"] = std::move(tools_cap);
    // We don't currently expose resources or prompts; clients see
    // an empty object for forward compat (presence ⇒ not advertised).

    json server_info;
    server_info["name"]    = info.name;
    server_info["version"] = info.version;

    json result;
    result["protocolVersion"] = info.protocol_version;
    result["capabilities"]    = std::move(caps);
    result["serverInfo"]      = std::move(server_info);
    return result;
}

// `tools/list`: enumerate every Tool in the catalogue.
json handle_tools_list(const json & /*params*/,
                       const std::vector<Tool> & tools) {
    json arr = json::array();
    arr.get_ptr<json::array_t *>()->reserve(tools.size());
    for (const auto & t : tools) {
        arr.push_back(tool_descriptor(t));
    }
    json result;
    result["tools"] = std::move(arr);
    return result;
}

// Find a tool by exact name. Returns nullptr if not registered.
const Tool * lookup_tool(const std::vector<Tool> & tools,
                          const std::string &       name) {
    for (const auto & t : tools) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

// `tools/call`: dispatch a tool by name. Errors map to JSON-RPC
// errors with appropriate codes:
//   - tool not registered             → kErrToolNotFound
//   - missing/malformed params object → kErrInvalidParams
//   - tool handler threw / returned isError=true → kErrToolFailed
//
// MCP successful tool result shape:
//   { "content": [{"type": "text", "text": "..."}], "isError": false }
//
// We always emit the `isError` field so clients don't have to
// guess; setting `isError: true` lets the model see the failure
// without us tearing down the request as a JSON-RPC error.
//
// IMPORTANT: easyai's ToolResult::is_error means "the tool ran but
// reported a logical failure" (e.g. missing argument). That's
// distinct from a transport-level error. We surface it through the
// MCP `isError` field rather than the JSON-RPC `error` envelope.
json handle_tools_call(const json &             params,
                        const std::vector<Tool> & tools,
                        std::string &            err_msg_out,
                        int &                    err_code_out) {
    err_msg_out.clear();
    err_code_out = 0;

    if (!params.is_object()) {
        err_code_out = kErrInvalidParams;
        err_msg_out  = "params must be an object";
        return json();
    }
    if (!params.contains("name") || !params["name"].is_string()) {
        err_code_out = kErrInvalidParams;
        err_msg_out  = "missing or non-string `name`";
        return json();
    }
    const std::string name = params["name"].get<std::string>();
    const Tool * t = lookup_tool(tools, name);
    if (!t) {
        err_code_out = kErrToolNotFound;
        err_msg_out  = "tool not registered: " + name;
        return json();
    }

    // Stringify `arguments` for the easyai::ToolCall envelope. The
    // builtin handlers parse this themselves via args::get_*; the
    // external_tools handler parses with nlohmann directly. Either
    // way, they want a JSON object string.
    std::string args_json = "{}";
    if (params.contains("arguments")) {
        const auto & a = params["arguments"];
        if (!a.is_null()) {
            if (!a.is_object()) {
                err_code_out = kErrInvalidParams;
                err_msg_out  = "`arguments` must be a JSON object";
                return json();
            }
            args_json = a.dump();
        }
    }

    ToolCall   call;
    call.name           = name;
    call.arguments_json = std::move(args_json);

    ToolResult tr;
    try {
        tr = t->handler(call);
    } catch (const std::exception & e) {
        // Tool handler threw — surface as MCP isError=true so the
        // model sees the failure as a tool result, not as a
        // protocol breakdown. Keep the JSON-RPC envelope as a
        // result.
        json content = json::array({
            json{{"type", "text"},
                 {"text", std::string("tool handler threw: ") + e.what()}}
        });
        json result;
        result["content"] = std::move(content);
        result["isError"] = true;
        return result;
    } catch (...) {
        json content = json::array({
            json{{"type", "text"},
                 {"text", "tool handler threw unknown exception"}}
        });
        json result;
        result["content"] = std::move(content);
        result["isError"] = true;
        return result;
    }

    json content = json::array({
        json{{"type", "text"}, {"text", tr.content}}
    });
    json result;
    result["content"] = std::move(content);
    result["isError"] = tr.is_error;
    return result;
}

// `ping`: cheap round-trip used by clients to check the server is
// alive and the auth/transport are working.
json handle_ping() {
    return json::object();
}

}  // namespace

std::string handle_request(const std::string &       request_body,
                           const std::vector<Tool> & tools,
                           const ServerInfo &        info) {
    json req;
    try {
        req = json::parse(request_body);
    } catch (const std::exception & e) {
        return make_error(json(nullptr), kErrParse,
                          std::string("parse error: ") + e.what()).dump();
    }
    if (!req.is_object()) {
        return make_error(json(nullptr), kErrInvalidRequest,
                          "request must be a JSON object").dump();
    }

    // Pull id BEFORE validating the rest so error responses echo
    // it correctly. id may be a number, string, or null.
    json id = req.contains("id") ? req["id"] : json(nullptr);

    // Validate the JSON-RPC envelope.
    if (!req.contains("jsonrpc") || !req["jsonrpc"].is_string()
            || req["jsonrpc"].get<std::string>() != "2.0") {
        return make_error(id, kErrInvalidRequest,
                          "missing or invalid `jsonrpc: \"2.0\"`").dump();
    }
    if (!req.contains("method") || !req["method"].is_string()) {
        return make_error(id, kErrInvalidRequest,
                          "missing or non-string `method`").dump();
    }
    const std::string method = req["method"].get<std::string>();
    const json &      params = req.contains("params") ? req["params"] : json(nullptr);

    // ----- notifications (no id) -----
    // JSON-RPC distinguishes "request" (has id) from "notification"
    // (no id). Notifications must not generate a response. We
    // recognise the standard MCP notifications and return empty.
    if (!req.contains("id")) {
        if (method == "notifications/initialized" ||
            method == "notifications/cancelled"   ||
            method == "notifications/progress") {
            return std::string();   // empty body — caller returns 204
        }
        // Unknown notification: still return empty (per spec we
        // must not respond, even with an error).
        return std::string();
    }

    // ----- methods that produce results -----
    try {
        if (method == "initialize") {
            return make_result(id, handle_initialize(params, info)).dump();
        }
        if (method == "tools/list") {
            return make_result(id, handle_tools_list(params, tools)).dump();
        }
        if (method == "tools/call") {
            std::string err_msg;
            int         err_code = 0;
            json result = handle_tools_call(params, tools, err_msg, err_code);
            if (err_code != 0) {
                return make_error(id, err_code, err_msg).dump();
            }
            return make_result(id, std::move(result)).dump();
        }
        if (method == "ping") {
            return make_result(id, handle_ping()).dump();
        }
        return make_error(id, kErrMethodNotFound,
                          "method not implemented: " + method).dump();
    } catch (const std::exception & e) {
        return make_error(id, kErrInternal,
                          std::string("internal error: ") + e.what()).dump();
    } catch (...) {
        return make_error(id, kErrInternal,
                          "internal error: unknown exception").dump();
    }
}

std::string render_tool_catalog(const std::vector<Tool> & tools) {
    json arr = json::array();
    arr.get_ptr<json::array_t *>()->reserve(tools.size());
    for (const auto & t : tools) {
        arr.push_back(tool_descriptor(t));
    }
    return arr.dump(2);
}

}  // namespace easyai::mcp
