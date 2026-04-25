// easyai/builtin_tools.hpp — batteries-included tools for agents.
#pragma once

#include "tool.hpp"

#include <string>

namespace easyai::tools {

// ---------- time --------------------------------------------------------
Tool datetime();   // returns current UTC + local time in ISO-8601

// ---------- web (require libcurl at build time) -------------------------
// web_fetch:  GET a URL, return text (HTML stripped of tags + trimmed).
Tool web_fetch();

// web_search: search the web via DuckDuckGo's HTML endpoint (no API key, no
// external service required) and return the top results as a numbered list
// of title / url / snippet.
Tool web_search();

// ---------- filesystem --------------------------------------------------
// All filesystem tools sandbox to a root directory you provide.
// Pass "" or "." to allow the current working directory tree.
Tool fs_read_file (std::string root = ".");
Tool fs_write_file(std::string root = ".");
Tool fs_list_dir  (std::string root = ".");
Tool fs_glob      (std::string root = ".");
Tool fs_grep      (std::string root = ".");

}  // namespace easyai::tools
