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
// `max_bytes` caps the read so a stray /dev/random or multi-GB log
// can't eat the heap; default 1 MiB is generous for system prompts /
// recipe text but cheap to override per call.
inline bool slurp_file(const std::string & path, std::string & out,
                       std::size_t max_bytes = 1u << 20) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    if (sz <= 0) { out.clear(); return true; }
    if ((std::size_t) sz > max_bytes) sz = (std::streamoff) max_bytes;
    f.seekg(0, std::ios::beg);
    out.assign((std::size_t) sz, '\0');
    f.read(out.data(), sz);
    out.resize((std::size_t) f.gcount());
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

// Truncate a string to at most `max_chars`, append "…" if cut, and
// flatten any newline/CR to a space so the result fits a single log
// line.  Handy for summarising tool args / results in --verbose dumps.
inline std::string trim_for_log(std::string s, std::size_t max_chars) {
    if (s.size() > max_chars) {
        s.resize(max_chars);
        s += "…";
    }
    for (char & c : s) if (c == '\n' || c == '\r') c = ' ';
    return s;
}

// Streaming filter that strips <think>…</think> / <thinking>…</thinking>
// blocks from token-by-token output even when the tag spans multiple
// pieces.  Pattern:
//
//   easyai::text::ThinkStripper strip;
//   strip.enabled = true;
//   on_token([&](const std::string & p){
//       std::cout << strip.filter(p) << std::flush;
//   });
//   /* end of stream */
//   std::cout << strip.flush();
//
// Memory is bounded — the buffer never grows past a small margin
// (the longest tag we recognise) when no tag is in flight.
class ThinkStripper {
public:
    bool enabled = false;

    // Emit visible text for a streamed piece. Returns the bytes the
    // caller should print to the user.
    std::string filter(const std::string & piece) {
        if (!enabled) return piece;
        buffer_ += piece;
        std::string out;
        for (;;) {
            if (in_think_) {
                std::size_t end = find_close(buffer_);
                if (end == std::string::npos) {
                    if (buffer_.size() > kCloseMargin)
                        buffer_.erase(0, buffer_.size() - kCloseMargin);
                    return out;
                }
                std::size_t close = buffer_.find('>', end);
                if (close == std::string::npos) return out;  // malformed
                buffer_.erase(0, close + 1);
                in_think_ = false;
            } else {
                std::size_t start = find_open(buffer_);
                if (start == std::string::npos) {
                    std::size_t safe = buffer_.size() > kOpenMargin
                                           ? buffer_.size() - kOpenMargin : 0;
                    if (safe > 0) {
                        out += buffer_.substr(0, safe);
                        buffer_.erase(0, safe);
                    }
                    return out;
                }
                out += buffer_.substr(0, start);
                std::size_t close = buffer_.find('>', start);
                if (close == std::string::npos) {
                    buffer_.erase(0, start);
                    return out;
                }
                buffer_.erase(0, close + 1);
                in_think_ = true;
            }
        }
    }

    // Emit any trailing bytes accumulated at end of stream.
    std::string flush() {
        std::string out;
        if (!enabled || !in_think_) out = std::move(buffer_);
        buffer_.clear();
        in_think_ = false;
        return out;
    }

    void reset() { buffer_.clear(); in_think_ = false; }

private:
    static constexpr std::size_t kOpenMargin  = 10;  // "<thinking" is 9
    static constexpr std::size_t kCloseMargin = 12;  // "</thinking>" is 11

    std::string buffer_;
    bool        in_think_ = false;

    static std::size_t find_open(const std::string & s) {
        std::size_t a = s.find("<think>");
        std::size_t b = s.find("<thinking>");
        if (a == std::string::npos) return b;
        if (b == std::string::npos) return a;
        return std::min(a, b);
    }
    static std::size_t find_close(const std::string & s) {
        std::size_t a = s.find("</think>");
        std::size_t b = s.find("</thinking>");
        if (a == std::string::npos) return b;
        if (b == std::string::npos) return a;
        return std::min(a, b);
    }
};

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
