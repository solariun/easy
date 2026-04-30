// easyai/mcp.hpp — Model Context Protocol server.
//
// easyai-server is also an MCP server. Other AI applications
// (Claude Desktop via stdio bridge, Cursor / Continue via HTTP,
// custom JSON-RPC clients) can list our tools and dispatch them,
// reusing every built-in tool plus RAG plus the operator's
// external-tools manifests as a single tool catalogue.
//
// Protocol surface (Model Context Protocol 2024-11-05):
//
//     initialize             handshake; declare capabilities
//     tools/list             enumerate all registered Tools
//     tools/call             dispatch a tool by name
//     ping                   round-trip health check
//
// Transport in this V1 is JSON-RPC 2.0 over a single POST /mcp
// endpoint. Stateless request/response — no SSE, no streaming
// notifications, no session tracking. The MCP spec accommodates
// this; richer transports (HTTP+SSE for server-pushed
// `tools/list_changed` notifications, stdio for Claude Desktop)
// can layer on later.
//
// Auth, body-size limit, and exception isolation are handled by
// the same httplib middleware that protects `/v1/chat/completions`.
// Nothing in this header touches the network — the dispatcher is a
// pure function from a JSON-RPC request body to a response body.
#pragma once

#include "tool.hpp"

#include <string>
#include <vector>

namespace easyai::mcp {

// Server identity advertised in the `initialize` response.
struct ServerInfo {
    std::string name             = "easyai";
    std::string version          = "0.1.0";
    std::string protocol_version = "2024-11-05";
};

// Process a single JSON-RPC 2.0 request body and return a JSON-RPC
// response body. Never throws — every failure path produces a
// proper JSON-RPC error envelope.
//
// `tools` is the set of tools currently registered with the server.
// They are advertised verbatim under `tools/list` (their
// `parameters_json` is parsed once and treated as the MCP
// `inputSchema`); `tools/call` looks them up by name and dispatches
// directly to their handler.
//
// `info` is the server identity. It's read by `initialize`
// (populates `serverInfo` in the response) and otherwise unused.
std::string handle_request(const std::string &              request_body,
                           const std::vector<Tool> &        tools,
                           const ServerInfo &               info);

// Render the MCP tool catalogue (the inner array of tool descriptors
// returned by `tools/list`) as a stand-alone JSON value. Useful for
// `/health`-style diagnostic endpoints that want to enumerate the
// MCP surface without going through a JSON-RPC round-trip.
std::string render_tool_catalog(const std::vector<Tool> & tools);

}  // namespace easyai::mcp
