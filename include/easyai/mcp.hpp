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

#include <map>
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

// ---------------------------------------------------------------------------
// Bearer-token auth helpers
// ---------------------------------------------------------------------------
//
// Both `easyai-server` and `easyai-mcp-server` enforce the same auth
// model on /mcp: an [MCP_USER] INI section maps usernames to tokens,
// the runtime gates each request by `Authorization: Bearer <token>`.
// The two binaries used to carry near-identical copies of this code;
// these helpers consolidate the shared pieces here so future fixes
// (token-rotation logic, audit-log shape, header caps) land in one
// place.
//
// The helpers are deliberately transport-agnostic — they take a raw
// `Authorization` header value and return a structured verdict so
// the calling binary (cpp-httplib in our case, but anything else
// works too) can render the response in its own framework's idiom.
// libeasyai stays free of httplib / nlohmann::json / OpenSSL deps
// here; any consumer of the lib can stand up an MCP server in their
// own HTTP transport.

// ---------------------------------------------------------------------------
// Shared transport-layer caps. Public so any consumer building an
// MCP HTTP server can apply them at the framework level (cpp-httplib,
// Boost.Beast, ASIO, …).
// ---------------------------------------------------------------------------
//
// kMaxAuthHeaderBytes  Cap on the Authorization header before string
//                      comparison. A real Bearer token is ≤ a few
//                      hundred bytes; 4 KiB is generous, beneath any
//                      sensible threshold, and short enough that a
//                      hostile client can't amortise CPU on every
//                      probe with a multi-megabyte header.
constexpr std::size_t kMaxAuthHeaderBytes = 4 * 1024;

// Bearer-auth verdict for one request.
//
// `ok = true`   → authorised; `user` is the [MCP_USER] username on
//                 the matching line (empty in open mode); the other
//                 fields are unused.
// `ok = false`  → 401; the caller writes `status`, `body`, and
//                 `www_authenticate` into its response and returns.
//
// `body` is a complete JSON-RPC 2.0 error envelope ready to ship.
// `www_authenticate` is the value for the `WWW-Authenticate`
// response header.
struct AuthResult {
    bool        ok = false;
    int         status = 0;             // 0 in open / authorised mode, 401 otherwise
    std::string user;                   // empty in open mode
    std::string body;                   // JSON-RPC error envelope when !ok
    std::string www_authenticate;       // WWW-Authenticate header when !ok
};

// Check a Bearer token against a token→username map.
//
// `mcp_keys` empty → open mode: returns ok=true, user="" regardless
//                    of the header. Lets the caller skip the auth
//                    response branch entirely.
// `mcp_keys` non-empty → Bearer required:
//                    - Authorization absent / oversize / malformed
//                      / unknown → ok=false with the proper 401.
//                    - Authorization matches a token → ok=true,
//                      user=<matching name>.
//
// Header is checked against `kMaxAuthHeaderBytes` before any string
// comparison.
AuthResult check_bearer(
    const std::map<std::string, std::string> & mcp_keys,
    const std::string &                        authorization_header);

// Build a token→username map from an [MCP_USER] INI section
// (username→token in the source).
//
// Skips empty tokens (operator typo). On duplicate tokens the first
// alphabetically-earlier username wins (deterministic, matches what
// the dispatchers in both servers used to do by accident).
//
// `ini_section` is what `easyai::config::Ini::section_or_empty
// ("MCP_USER")` returns.
std::map<std::string, std::string>
load_mcp_users(const std::map<std::string, std::string> & ini_section);

}  // namespace easyai::mcp
