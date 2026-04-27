// easyai/text.hpp — small string helpers used by the example CLIs.
//
// Header-only; everything here is short, hot, and trivially inlinable.
// Pulled out of cli_remote/cli so the helpers are reusable from third-
// party agents without copy-pasting.
#pragma once

#include <fstream>
#include <sstream>
#include <string>

namespace easyai::text {

// Read a whole file into a string.  Returns false if open fails.
inline bool slurp_file(const std::string & path, std::string & out) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Insert a newline before <think> and after </think> so the markers
// don't get visually joined to surrounding stream content (when the
// model leaks them through despite the chat template stripping).
// Best-effort whole-piece scan — if a tag splits across SSE chunks
// the join glitch survives, but llama.cpp tokenizes both as single
// tokens so the split case is rare.
inline std::string punctuate_think_tags(const std::string & in) {
    std::string out;
    out.reserve(in.size() + 4);
    for (size_t i = 0; i < in.size(); ) {
        if (in.compare(i, 7, "<think>") == 0) {
            if (!out.empty() && out.back() != '\n') out.push_back('\n');
            out.append("<think>");
            if (i + 7 < in.size() && in[i + 7] != '\n') out.push_back('\n');
            i += 7;
        } else if (in.compare(i, 8, "</think>") == 0) {
            if (!out.empty() && out.back() != '\n') out.push_back('\n');
            out.append("</think>");
            if (i + 8 < in.size() && in[i + 8] != '\n') out.push_back('\n');
            i += 8;
        } else {
            out.push_back(in[i++]);
        }
    }
    return out;
}

// Heuristic: does the user's prompt look like it wants the model to
// write a file?  Used by example CLIs to surface a "you forgot
// --sandbox" hint when fs_write_file isn't registered.  Not exact;
// false positives just produce a friendly extra message.
inline bool prompt_wants_file_write(const std::string & prompt) {
    std::string lo;
    lo.reserve(prompt.size());
    for (char c : prompt) lo.push_back((char) std::tolower((unsigned char) c));
    static const char * needles[] = {
        "write to ",   "write it ", "write a file", "write the file",
        "save to ",    "save it ",  "save as ",     "save a file",
        "create a file","create the file",
        "into a file", "to a file",
        ".md",         ".txt",      ".json",        ".csv",
        ".html",       ".log",
    };
    for (const char * n : needles) {
        if (lo.find(n) != std::string::npos) return true;
    }
    return false;
}

}  // namespace easyai::text
