#!/usr/bin/env python3
"""mcp-stdio-bridge.py — bridge stdio MCP ↔ easyai-server's HTTP /mcp.

Why this exists
---------------
Claude Desktop only speaks the MCP "stdio" transport: it spawns the
MCP server as a subprocess and exchanges JSON-RPC over stdin/stdout.
easyai-server is HTTP-only. This script is the adapter — it lives
on the operator's machine, gets spawned by Claude Desktop, and
forwards every JSON-RPC frame it reads from stdin to a POST /mcp
on the easyai-server, then writes the response back to stdout.

Cursor / Continue / OpenWebUI talk HTTP MCP directly — they don't
need this script. Use it ONLY for stdio-only clients like Claude
Desktop.

Configuration in Claude Desktop
-------------------------------
Edit ~/.config/Claude/claude_desktop_config.json (or the platform
equivalent). Add:

    {
      "mcpServers": {
        "easyai": {
          "command": "/usr/bin/python3",
          "args": [
            "/path/to/mcp-stdio-bridge.py",
            "--url", "http://192.168.1.10:80"
          ]
        }
      }
    }

If your easyai-server has a Bearer token configured, pass it via
`--token TOKEN` or the EASYAI_API_KEY env var. If neither is set
the request is sent without an Authorization header.

Operational notes
-----------------
* This script is dependency-free — only the standard library. No
  pip install. Run it with whatever Python 3 is on the host.
* Errors (network down, easyai not reachable, malformed body)
  surface as JSON-RPC errors so Claude Desktop's UI shows the
  actual reason instead of a silent disconnect.
* Notifications (JSON-RPC requests with no `id`) are forwarded but
  no response is awaited — Claude Desktop doesn't expect one.
* The bridge does NOT touch the body it forwards; the easyai-server
  side is the source of truth for the protocol semantics.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def emit(obj):
    """Write a JSON-RPC frame to stdout, line-delimited, then flush."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def emit_error(req_id, code, message):
    emit({
        "jsonrpc": "2.0",
        "id":      req_id,
        "error":   {"code": code, "message": message},
    })


def forward_one(line: str, url: str, token: str, timeout: float):
    line = line.strip()
    if not line:
        return

    # Try to parse so we can echo `id` back on transport errors.
    req_id = None
    try:
        parsed = json.loads(line)
        req_id = parsed.get("id")
    except json.JSONDecodeError:
        # Bridge is dumb — if the upstream client (Claude Desktop)
        # sent malformed JSON, easyai-server will reject it cleanly
        # and we just forward the response. Don't second-guess.
        parsed = None

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body = line.encode("utf-8")
    target = url.rstrip("/") + "/mcp"
    req = urllib.request.Request(target, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp_body = r.read()
            status    = r.status
    except urllib.error.HTTPError as e:
        try:
            resp_body = e.read()
        except Exception:
            resp_body = b""
        status = e.code
        # If the server returned a JSON-RPC envelope on a 4xx/5xx,
        # forward it verbatim so the client sees the real reason.
        if resp_body:
            sys.stdout.write(resp_body.decode("utf-8", errors="replace"))
            if not resp_body.endswith(b"\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
            return
        emit_error(req_id, -32000, f"HTTP {status} from easyai-server")
        return
    except urllib.error.URLError as e:
        emit_error(req_id, -32000,
                   f"cannot reach easyai-server at {target}: {e.reason}")
        return
    except Exception as e:
        emit_error(req_id, -32603, f"bridge error: {e!s}")
        return

    # 204 No Content — server treated this as a notification. The
    # JSON-RPC contract says no response, so we don't write one.
    if status == 204 or not resp_body:
        return

    # Forward the body verbatim — line-delimited, with a trailing
    # newline if the server didn't include one.
    sys.stdout.write(resp_body.decode("utf-8", errors="replace"))
    if not resp_body.endswith(b"\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(
        description="stdio↔HTTP bridge for the easyai MCP server")
    ap.add_argument("--url", required=True,
                    help="Base URL of the easyai-server, e.g. "
                         "http://localhost:80 or http://10.0.0.5:8080")
    ap.add_argument("--token", default=os.environ.get("EASYAI_API_KEY", ""),
                    help="Bearer token (default: $EASYAI_API_KEY env var)")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="HTTP timeout in seconds (default 300; matches "
                         "easyai-server's read-timeout ceiling)")
    args = ap.parse_args()

    # Read line-delimited JSON-RPC frames from stdin until EOF.
    # The MCP stdio transport is line-delimited per the spec.
    for line in sys.stdin:
        forward_one(line, args.url, args.token, args.timeout)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
