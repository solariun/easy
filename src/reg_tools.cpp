// src/reg_tools.cpp — implementation of the REG persistent registry.
//
// On-disk format and constraints are documented in the public
// header. This file enforces those contracts plus the four tool
// handlers.
//
// Design choices worth calling out:
//
//   * The format is INTENTIONALLY trivial. One header line
//     (`keywords: a, b, c`), one blank line, then a free-form body.
//     The operator can `cat` an entry, edit it with `vim`, drop
//     a hand-authored note in the dir — no JSON parser, no
//     escape rules, no "did I get the structure right?".
//
//   * The keyword index lives in process memory, lazily built on
//     first use. Each entry's metadata (keywords + mtime) is small,
//     so even 10 000 entries is well under a megabyte resident.
//
//   * Saves are atomic via tempfile + rename(2). Readers either
//     see the old content or the new content, never a partial
//     write.
//
//   * Listing/searching never reads the body from disk — only
//     the title (= filename) and the cached keywords from the index.
//     `reg_load` does the single read for the body when the
//     model picks an entry.
//
//   * The title is the filesystem name. Validating it with a
//     strict regex (`^[A-Za-z0-9_-]+$`, ≤ 64 bytes) is what
//     closes the path-traversal door — there is no way for the
//     model to write `..`, slashes, NUL bytes, etc.

#include "easyai/reg_tools.hpp"
#include "easyai/tool.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace easyai::tools {

namespace {

namespace fs = std::filesystem;
using json   = nlohmann::json;

// ---------------------------------------------------------------------------
// Constants — every limit named, every cap explained.
// ---------------------------------------------------------------------------

// Title and keyword identifier shape. The title is the filesystem name
// of the entry's file, so banning slashes / dots / spaces also
// closes path traversal as a side effect.
constexpr std::size_t kMaxTitleBytes      = 64;
constexpr std::size_t kMaxKeywordBytes        = 32;
constexpr std::size_t kMaxKeywordsPerEntry    = 8;

// Content cap. 256 KiB is generous for "a piece of knowledge worth
// remembering" — long enough to fit a code recipe + commentary, short
// enough that 1 000 entries fit comfortably on disk and in memory.
constexpr std::size_t kMaxContentBytes    = 256u * 1024u;

// reg_load can fan out to up to 4 entries per call. The cap is
// deliberate: more than 4 means the model is probably about to
// drown the prompt in stale content rather than focusing on what
// matters, and the agent loop stays cheaper too.
constexpr std::size_t kMaxLoadAtOnce      = 4;

// Search result cap. The model gets a list of (title, keywords,
// preview) and picks 1..4 to reg_load — so we don't need a huge
// list, just enough to cover obvious keyword clusters.
constexpr std::size_t kSearchResultsMax   = 20;
constexpr std::size_t kSearchResultsDflt  = 10;

// Bytes from the body we render in a search-result preview. Long
// enough for the model to recognise "yes, that's the entry I want",
// short enough to keep the prompt slim.
constexpr std::size_t kSearchPreviewBytes = 240;

// reg_list cap. Browsing-only; no body read.
constexpr std::size_t kListResultsMax     = 200;
constexpr std::size_t kListResultsDflt    = 50;

// On-disk file extension. `.md` keeps entries human-readable in
// any text editor, lets the operator `cat` / `vim` / `grep`, and
// renders nicely as markdown when the body uses it.
constexpr char        kEntrySuffix[]      = ".md";
constexpr std::size_t kEntrySuffixLen     = sizeof(kEntrySuffix) - 1;

// ---------------------------------------------------------------------------
// Validators — pure, no I/O, no allocation past the input.
// ---------------------------------------------------------------------------

bool is_valid_id(const std::string & s, std::size_t max_len) {
    if (s.empty() || s.size() > max_len) return false;
    for (char c : s) {
        const bool ok =
            (c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '-' || c == '_';
        if (!ok) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Format helpers — read/write the tiny "keywords: a, b, c\n\n<body>" shape.
// ---------------------------------------------------------------------------

// Trim ASCII whitespace (space/tab) from both ends. Newlines / CR
// are NOT stripped here — callers handle those on a line basis.
std::string trim_inline(std::string s) {
    auto issp = [](unsigned char c) { return c == ' ' || c == '\t'; };
    while (!s.empty() && issp(s.front())) s.erase(s.begin());
    while (!s.empty() && issp(s.back()))  s.pop_back();
    return s;
}

// Split `value` on commas, trim each piece, drop empties. Caller
// validates each keyword against `is_valid_id` separately.
std::vector<std::string> split_keywords(const std::string & value) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : value) {
        if (c == ',') {
            cur = trim_inline(cur);
            if (!cur.empty()) out.push_back(std::move(cur));
            cur.clear();
        } else {
            cur += c;
        }
    }
    cur = trim_inline(cur);
    if (!cur.empty()) out.push_back(std::move(cur));
    return out;
}

// Read a whole file with a hard size cap so a malicious / corrupt
// entry doesn't pull the process into oversized allocations.
bool slurp_capped(const fs::path & p, std::size_t max_bytes,
                  std::string & out, std::string & err) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { err = "cannot open: " + p.string(); return false; }
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    if (sz < 0) { err = "tellg failed: " + p.string(); return false; }
    if ((std::size_t) sz > max_bytes) {
        err = "entry exceeds " + std::to_string(max_bytes) + " bytes";
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.assign((std::size_t) sz, '\0');
    f.read(out.data(), sz);
    out.resize((std::size_t) f.gcount());
    return true;
}

// Parse the on-disk text into keywords + body. The grammar is:
//
//     <header-line>*
//     <blank-line>
//     <body>
//
// where each header is `key: value` (we currently recognise `keywords`).
// A file with NO blank line is treated as body-only (no header).
//
// Headers ARE optional. A file like:
//
//     just a free-form note dropped here by the operator
//
// is loaded as `keywords = []`, `body = "<entire content>"`.
// `reg_search` won't find it (no keywords) but `reg_list` will.
struct ParsedEntry {
    std::vector<std::string> keywords;
    std::string              body;
};

ParsedEntry parse_entry(const std::string & raw) {
    ParsedEntry out;
    if (raw.empty()) return out;

    // Look at the first line. If it doesn't look like a header
    // (`<key>: ...`), treat the entire file as body.
    auto looks_like_header = [](const std::string & line) {
        const auto colon = line.find(':');
        if (colon == std::string::npos || colon == 0) return false;
        for (std::size_t i = 0; i < colon; ++i) {
            const unsigned char c = static_cast<unsigned char>(line[i]);
            if (!(std::isalnum(c) || c == '_' || c == '-')) return false;
        }
        return true;
    };

    std::size_t pos = 0;
    auto next_line = [&](std::string & line) -> bool {
        if (pos >= raw.size()) return false;
        std::size_t end = raw.find('\n', pos);
        if (end == std::string::npos) {
            line.assign(raw, pos, raw.size() - pos);
            pos = raw.size();
        } else {
            line.assign(raw, pos, end - pos);
            pos = end + 1;
        }
        // Strip CR for CRLF inputs.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
    };

    std::string first;
    std::size_t first_line_pos = pos;
    if (!next_line(first)) return out;
    if (!looks_like_header(first)) {
        // Body-only file. Restore position and consume everything.
        out.body.assign(raw, first_line_pos, raw.size() - first_line_pos);
        return out;
    }

    // Process the header lines until a blank line or non-header.
    auto try_consume_header = [&](const std::string & line) {
        if (line.empty()) return;
        if (!looks_like_header(line)) return;
        const auto colon = line.find(':');
        std::string key = line.substr(0, colon);
        std::string val = trim_inline(line.substr(colon + 1));
        // Lowercase the key for case-insensitive matching.
        for (auto & c : key) c = (char) std::tolower((unsigned char) c);
        if (key == "keywords") {
            out.keywords = split_keywords(val);
        }
        // Future header keys land here. Unknown keys are silently
        // ignored — a hand-edit can leave stray fields without
        // breaking the load.
    };

    try_consume_header(first);

    std::string line;
    bool body_started = false;
    while (next_line(line)) {
        if (!body_started) {
            if (line.empty()) { body_started = true; continue; }
            if (looks_like_header(line)) { try_consume_header(line); continue; }
            // Non-header, non-blank → start body here, keeping this line.
            out.body = line;
            if (pos < raw.size()) {
                out.body += '\n';
                out.body.append(raw, pos, raw.size() - pos);
            }
            return out;
        }
        // body_started — append.
        if (!out.body.empty()) out.body += '\n';
        out.body += line;
    }
    return out;
}

// Render the on-disk text from keywords + body. Keeps the trailing
// newline so `cat` doesn't print without one.
std::string render_entry(const std::vector<std::string> & keywords,
                         const std::string & body) {
    std::string out;
    out.reserve(body.size() + 64);
    out += "keywords:";
    for (std::size_t i = 0; i < keywords.size(); ++i) {
        out += (i == 0 ? " " : ", ");
        out += keywords[i];
    }
    out += "\n\n";
    out += body;
    if (out.empty() || out.back() != '\n') out += '\n';
    return out;
}

// ---------------------------------------------------------------------------
// EntryMeta — what the in-memory index stores per title.
// ---------------------------------------------------------------------------
struct EntryMeta {
    std::vector<std::string> keywords;
    std::int64_t             modified_unix = 0;
    std::size_t              content_bytes = 0;
};

// ---------------------------------------------------------------------------
// RegStore — the shared state for all four tools.
// ---------------------------------------------------------------------------
struct RegStore {
    fs::path                            root;
    std::mutex                          mu;
    std::map<std::string, EntryMeta>    index;
    bool                                index_loaded = false;

    explicit RegStore(std::string r) {
        if (!r.empty()) {
            std::error_code ec;
            root = fs::absolute(r, ec);
            if (ec) root = fs::path(std::move(r));
        }
    }

    bool root_set() const { return !root.empty(); }

    // Ensure root exists and is a directory. Called from anywhere
    // that wants to write; reads tolerate a missing dir (returns
    // empty list).
    bool ensure_dir(std::string & err) {
        if (!root_set()) { err = "REG root path is empty"; return false; }
        std::error_code ec;
        if (fs::exists(root, ec)) {
            if (!fs::is_directory(root, ec)) {
                err = "REG root is not a directory: " + root.string();
                return false;
            }
            return true;
        }
        fs::create_directories(root, ec);
        if (ec) {
            err = "create REG dir failed: " + ec.message();
            return false;
        }
        return true;
    }

    // Walk the directory, parse each entry's header for keywords, build
    // the in-memory index. Files that don't match the suffix or whose
    // title is invalid are silently skipped.
    void load_index_locked() {
        if (index_loaded) return;
        index.clear();
        index_loaded = true;
        if (!root_set()) return;
        std::error_code ec;
        if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) {
            return;
        }
        for (const auto & e : fs::directory_iterator(root, ec)) {
            if (ec) break;
            if (!e.is_regular_file()) continue;
            const auto fname = e.path().filename().string();
            if (fname.size() <= kEntrySuffixLen) continue;
            if (fname.compare(fname.size() - kEntrySuffixLen,
                              kEntrySuffixLen, kEntrySuffix) != 0) {
                continue;
            }
            const std::string title =
                fname.substr(0, fname.size() - kEntrySuffixLen);
            if (!is_valid_id(title, kMaxTitleBytes)) continue;

            std::string raw, err;
            if (!slurp_capped(e.path(),
                              kMaxContentBytes + 4096 /* header room */,
                              raw, err)) {
                continue;
            }
            ParsedEntry pe = parse_entry(raw);

            EntryMeta m;
            for (auto & t : pe.keywords) {
                if (is_valid_id(t, kMaxKeywordBytes)) {
                    m.keywords.push_back(std::move(t));
                    if (m.keywords.size() == kMaxKeywordsPerEntry) break;
                }
            }
            m.content_bytes = pe.body.size();
            // Modified time from filesystem stat. fs::last_write_time
            // returns a file_time_type whose epoch isn't guaranteed;
            // convert to system_clock via a C++17 trick to get unix.
            auto ftime = fs::last_write_time(e.path(), ec);
            if (!ec) {
                auto sctp = std::chrono::time_point_cast<
                    std::chrono::system_clock::duration>(
                    ftime - decltype(ftime)::clock::now()
                          + std::chrono::system_clock::now());
                m.modified_unix = std::chrono::duration_cast<
                    std::chrono::seconds>(
                    sctp.time_since_epoch()).count();
            }
            index[title] = std::move(m);
        }
    }

    // Atomic write: tempfile + rename(2). Updates the in-memory
    // index on success.
    bool save_locked(const std::string &              title,
                     const std::vector<std::string> & keywords,
                     const std::string &              content,
                     std::string &                    err) {
        if (!ensure_dir(err)) return false;
        const auto target = root / (title + kEntrySuffix);
        const auto tmp    = root / (title + std::string(kEntrySuffix)
                                          + ".tmp." + std::to_string(::getpid()));

        const std::string text = render_entry(keywords, content);

        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) {
                err = "cannot open REG tempfile in " + root.string();
                return false;
            }
            out.write(text.data(), (std::streamsize) text.size());
            out.flush();
            if (!out) {
                err = "write to REG tempfile failed";
                std::error_code ec;
                fs::remove(tmp, ec);
                return false;
            }
        }
        std::error_code ec;
        fs::rename(tmp, target, ec);
        if (ec) {
            err = "atomic rename failed: " + ec.message();
            fs::remove(tmp, ec);
            return false;
        }

        EntryMeta m;
        m.keywords = keywords;
        // Read mtime back from the file we just wrote so the index
        // matches what the FS will report.
        auto ftime = fs::last_write_time(target, ec);
        if (!ec) {
            auto sctp = std::chrono::time_point_cast<
                std::chrono::system_clock::duration>(
                ftime - decltype(ftime)::clock::now()
                      + std::chrono::system_clock::now());
            m.modified_unix = std::chrono::duration_cast<
                std::chrono::seconds>(sctp.time_since_epoch()).count();
        }
        m.content_bytes = content.size();
        index[title] = std::move(m);
        return true;
    }

    // Read the full entry off disk. Returns the parsed body in
    // `body_out`. Caller doesn't need to hold the mutex — file
    // content is written via atomic rename, so reads always see
    // a complete file.
    bool load_one(const std::string & title,
                  std::vector<std::string> & keywords_out,
                  std::string & body_out,
                  std::int64_t & modified_unix_out,
                  std::string & err) {
        if (!root_set()) { err = "REG root path is empty"; return false; }
        const auto target = root / (title + kEntrySuffix);
        std::error_code ec;
        if (!fs::exists(target, ec)) {
            err = "no REG entry titled \"" + title + "\"";
            return false;
        }
        std::string raw;
        if (!slurp_capped(target, kMaxContentBytes + 4096, raw, err)) {
            return false;
        }
        ParsedEntry pe = parse_entry(raw);
        keywords_out.clear();
        for (auto & t : pe.keywords) {
            if (is_valid_id(t, kMaxKeywordBytes)) keywords_out.push_back(std::move(t));
        }
        body_out = std::move(pe.body);
        modified_unix_out = 0;
        auto ftime = fs::last_write_time(target, ec);
        if (!ec) {
            auto sctp = std::chrono::time_point_cast<
                std::chrono::system_clock::duration>(
                ftime - decltype(ftime)::clock::now()
                      + std::chrono::system_clock::now());
            modified_unix_out = std::chrono::duration_cast<
                std::chrono::seconds>(sctp.time_since_epoch()).count();
        }
        return true;
    }

    // Delete an entry. Idempotent: deleting a non-existent title
    // returns true with `existed = false` so the caller can decide
    // whether to surface "no such entry" or stay quiet. Removes the
    // index entry too.
    bool delete_locked(const std::string & title, bool & existed,
                       std::string & err) {
        existed = false;
        if (!root_set()) { err = "REG root path is empty"; return false; }
        const auto target = root / (title + kEntrySuffix);
        std::error_code ec;
        if (!fs::exists(target, ec)) {
            // Drop any stale index entry just in case.
            index.erase(title);
            return true;   // idempotent success
        }
        existed = true;
        if (!fs::remove(target, ec)) {
            err = "delete failed: " + ec.message();
            return false;
        }
        index.erase(title);
        return true;
    }
};

// ---------------------------------------------------------------------------
// Helper: parse a JSON array of strings out of the model's tool args.
// args::get_string handles flat strings; arrays we parse with
// nlohmann directly so the model can pass proper structured input.
// ---------------------------------------------------------------------------
bool parse_string_array(const std::string & args_json,
                        const std::string & key,
                        std::vector<std::string> & out,
                        std::string & err) {
    try {
        auto j = json::parse(args_json.empty() ? std::string("{}") : args_json);
        if (!j.is_object()) {
            err = "arguments must be a JSON object";
            return false;
        }
        if (!j.contains(key)) return true;          // absent → empty
        const auto & v = j[key];
        if (v.is_null()) return true;
        if (!v.is_array()) {
            err = "argument " + key + ": expected array of strings";
            return false;
        }
        out.clear();
        out.reserve(v.size());
        for (const auto & e : v) {
            if (!e.is_string()) {
                err = "argument " + key + ": every element must be a string";
                return false;
            }
            out.push_back(e.get<std::string>());
        }
        return true;
    } catch (const std::exception & e) {
        err = std::string("invalid JSON: ") + e.what();
        return false;
    }
}

// Render a UTF-8 preview of `content`, capped at `max_bytes` AND
// snapped back to a code-point boundary so we never split a
// multi-byte character. A trailing "…" marker tells the model the
// snippet is truncated.
std::string make_preview(const std::string & content, std::size_t max_bytes) {
    if (content.size() <= max_bytes) return content;
    std::size_t cut = max_bytes;
    while (cut > 0 && (static_cast<unsigned char>(content[cut]) & 0xC0) == 0x80) {
        --cut;
    }
    std::string out(content, 0, cut);
    out += "…";
    return out;
}

// ---------------------------------------------------------------------------
// Tool handler factories
// ---------------------------------------------------------------------------

ToolHandler make_save_handler(std::shared_ptr<RegStore> store) {
    return [store](const ToolCall & c) -> ToolResult {
        std::string title;
        if (!args::get_string(c.arguments_json, "title", title) || title.empty()) {
            return ToolResult::error("missing required argument: title");
        }
        if (!is_valid_id(title, kMaxTitleBytes)) {
            return ToolResult::error(
                "title \"" + title + "\" is invalid; must match "
                "[A-Za-z0-9_-]{1," + std::to_string(kMaxTitleBytes) + "}");
        }

        std::vector<std::string> keywords;
        std::string err;
        if (!parse_string_array(c.arguments_json, "keywords", keywords, err)) {
            return ToolResult::error(err);
        }
        if (keywords.empty()) {
            return ToolResult::error(
                "keywords must be a non-empty array (1.."
                + std::to_string(kMaxKeywordsPerEntry)
                + " short keywords). Why: keywords are how reg_search finds this "
                  "entry later — an entry with no keywords is unreachable by keyword "
                  "search (only reg_list can find it).");
        }
        if (keywords.size() > kMaxKeywordsPerEntry) {
            return ToolResult::error(
                "too many keywords (max " + std::to_string(kMaxKeywordsPerEntry) + ")");
        }
        for (const auto & t : keywords) {
            if (!is_valid_id(t, kMaxKeywordBytes)) {
                return ToolResult::error(
                    "keyword \"" + t + "\" is invalid; must match "
                    "[A-Za-z0-9_-]{1," + std::to_string(kMaxKeywordBytes) + "}");
            }
        }

        std::string content;
        if (!args::get_string(c.arguments_json, "content", content)) {
            return ToolResult::error("missing required argument: content");
        }
        if (content.size() > kMaxContentBytes) {
            return ToolResult::error(
                "content exceeds " + std::to_string(kMaxContentBytes)
                + " bytes; split into multiple entries");
        }

        std::lock_guard<std::mutex> lock(store->mu);
        if (!store->save_locked(title, keywords, content, err)) {
            return ToolResult::error(err);
        }

        std::ostringstream o;
        o << "saved \"" << title << kEntrySuffix << "\" ("
          << content.size() << " bytes, "
          << keywords.size() << " keyword" << (keywords.size() == 1 ? "" : "s") << ")";
        return ToolResult::ok(o.str());
    };
}

ToolHandler make_search_handler(std::shared_ptr<RegStore> store) {
    return [store](const ToolCall & c) -> ToolResult {
        std::string keyword;
        if (!args::get_string(c.arguments_json, "keyword", keyword) || keyword.empty()) {
            return ToolResult::error("missing required argument: keyword");
        }
        if (!is_valid_id(keyword, kMaxKeywordBytes)) {
            return ToolResult::error(
                "keyword \"" + keyword + "\" is invalid; must match "
                "[A-Za-z0-9_-]{1," + std::to_string(kMaxKeywordBytes) + "}");
        }
        long long max_results = (long long) kSearchResultsDflt;
        args::get_int(c.arguments_json, "max_results", max_results);
        if (max_results < 1) max_results = 1;
        if ((std::size_t) max_results > kSearchResultsMax) {
            max_results = (long long) kSearchResultsMax;
        }

        struct Hit { std::string title; EntryMeta meta; };
        std::vector<Hit> hits;
        {
            std::lock_guard<std::mutex> lock(store->mu);
            store->load_index_locked();
            for (const auto & [t, m] : store->index) {
                for (const auto & et : m.keywords) {
                    if (et == keyword) {
                        hits.push_back({ t, m });
                        break;
                    }
                }
            }
        }
        std::sort(hits.begin(), hits.end(), [](const Hit & a, const Hit & b) {
            return a.meta.modified_unix > b.meta.modified_unix;
        });
        if ((long long) hits.size() > max_results) {
            hits.resize((std::size_t) max_results);
        }

        if (hits.empty()) {
            return ToolResult::ok(
                "no entries match keyword \"" + keyword + "\". "
                "Use reg_list to browse all titles, or reg_save to add new ones.");
        }

        // Render plain-text + structured (markdown-friendly) output.
        // The model parses this with no JSON dependency on its side
        // and the operator can `cat` it from a journal log.
        std::ostringstream o;
        o << hits.size() << " entr" << (hits.size() == 1 ? "y" : "ies")
          << " match keyword \"" << keyword << "\" (newest first):\n\n";
        for (std::size_t i = 0; i < hits.size(); ++i) {
            const auto & h = hits[i];
            std::vector<std::string> body_keywords;
            std::string              body_text;
            std::int64_t             mtime = 0;
            std::string              load_err;
            std::string              preview;
            if (store->load_one(h.title, body_keywords, body_text, mtime, load_err)) {
                preview = make_preview(body_text, kSearchPreviewBytes);
            } else {
                preview = "(could not read body: " + load_err + ")";
            }

            o << (i + 1) << ". " << h.title << "\n";
            o << "   keywords: ";
            for (std::size_t k = 0; k < h.meta.keywords.size(); ++k) {
                if (k) o << ", ";
                o << h.meta.keywords[k];
            }
            o << "  (" << h.meta.content_bytes << " bytes)\n";
            // Indent preview lines by 3 so it's easy to skim.
            std::string preview_in;
            preview_in.reserve(preview.size() + preview.size() / 60 * 3);
            for (char ch : preview) {
                preview_in += ch;
                if (ch == '\n') preview_in += "   ";
            }
            o << "   " << preview_in << "\n\n";
        }
        o << "Use reg_load with up to " << kMaxLoadAtOnce
          << " of these titles for full content.\n";
        return ToolResult::ok(o.str());
    };
}

ToolHandler make_load_handler(std::shared_ptr<RegStore> store) {
    return [store](const ToolCall & c) -> ToolResult {
        std::vector<std::string> titles;
        std::string err;
        if (!parse_string_array(c.arguments_json, "titles", titles, err)) {
            return ToolResult::error(err);
        }
        if (titles.empty()) {
            return ToolResult::error("missing required argument: titles "
                                     "(non-empty array)");
        }
        if (titles.size() > kMaxLoadAtOnce) {
            return ToolResult::error(
                "too many titles requested (max "
                + std::to_string(kMaxLoadAtOnce)
                + " per call); narrow your reg_search first");
        }
        for (const auto & t : titles) {
            if (!is_valid_id(t, kMaxTitleBytes)) {
                return ToolResult::error(
                    "title \"" + t + "\" is invalid; must match "
                    "[A-Za-z0-9_-]{1," + std::to_string(kMaxTitleBytes) + "}");
            }
        }

        std::ostringstream o;
        o << "loaded " << titles.size() << " entr"
          << (titles.size() == 1 ? "y" : "ies") << ":\n";
        for (const auto & title : titles) {
            std::vector<std::string> keywords;
            std::string body;
            std::int64_t mtime = 0;
            std::string e_err;
            std::lock_guard<std::mutex> lock(store->mu);
            if (!store->load_one(title, keywords, body, mtime, e_err)) {
                o << "\n--- " << title << " ---\n"
                  << "ERROR: " << e_err << "\n";
                continue;
            }
            o << "\n--- " << title << " ---\n";
            o << "keywords: ";
            for (std::size_t i = 0; i < keywords.size(); ++i) {
                if (i) o << ", ";
                o << keywords[i];
            }
            o << "\nmodified_unix: " << mtime << "\n\n";
            o << body;
            if (!body.empty() && body.back() != '\n') o << '\n';
        }
        return ToolResult::ok(o.str());
    };
}

ToolHandler make_list_handler(std::shared_ptr<RegStore> store) {
    return [store](const ToolCall & c) -> ToolResult {
        std::string prefix;
        args::get_string(c.arguments_json, "prefix", prefix);
        if (!prefix.empty() && !is_valid_id(prefix, kMaxTitleBytes)) {
            return ToolResult::error(
                "prefix \"" + prefix + "\" is invalid; must match "
                "[A-Za-z0-9_-]{1," + std::to_string(kMaxTitleBytes) + "}");
        }
        long long max = (long long) kListResultsDflt;
        args::get_int(c.arguments_json, "max", max);
        if (max < 1) max = 1;
        if ((std::size_t) max > kListResultsMax) max = (long long) kListResultsMax;

        struct Row { std::string title; EntryMeta meta; };
        std::vector<Row> rows;
        {
            std::lock_guard<std::mutex> lock(store->mu);
            store->load_index_locked();
            for (const auto & [t, m] : store->index) {
                if (!prefix.empty() &&
                    (t.size() < prefix.size() ||
                     t.compare(0, prefix.size(), prefix) != 0)) {
                    continue;
                }
                rows.push_back({ t, m });
                if ((long long) rows.size() >= max) break;
            }
        }

        if (rows.empty()) {
            return ToolResult::ok(prefix.empty()
                ? "REGistry is empty. Use reg_save to add entries."
                : "no titles match prefix \"" + prefix + "\".");
        }

        std::ostringstream o;
        o << rows.size() << " entr" << (rows.size() == 1 ? "y" : "ies");
        if (!prefix.empty()) o << " matching prefix \"" << prefix << "\"";
        o << ":\n\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto & r = rows[i];
            o << (i + 1) << ". " << r.title;
            if (!r.meta.keywords.empty()) {
                o << "  [";
                for (std::size_t k = 0; k < r.meta.keywords.size(); ++k) {
                    if (k) o << ", ";
                    o << r.meta.keywords[k];
                }
                o << "]";
            } else {
                o << "  (no keywords)";
            }
            o << "  " << r.meta.content_bytes << " bytes";
            o << "  modified_unix=" << r.meta.modified_unix << "\n";
        }
        return ToolResult::ok(o.str());
    };
}

ToolHandler make_delete_handler(std::shared_ptr<RegStore> store) {
    return [store](const ToolCall & c) -> ToolResult {
        std::string title;
        if (!args::get_string(c.arguments_json, "title", title) || title.empty()) {
            return ToolResult::error("missing required argument: title");
        }
        if (!is_valid_id(title, kMaxTitleBytes)) {
            return ToolResult::error(
                "title \"" + title + "\" is invalid; must match "
                "[A-Za-z0-9_-]{1," + std::to_string(kMaxTitleBytes) + "}");
        }
        std::lock_guard<std::mutex> lock(store->mu);
        store->load_index_locked();
        bool existed = false;
        std::string err;
        if (!store->delete_locked(title, existed, err)) {
            return ToolResult::error(err);
        }
        if (!existed) {
            return ToolResult::ok(
                "no entry titled \"" + title + "\" — nothing to delete");
        }
        return ToolResult::ok(
            "deleted \"" + title + kEntrySuffix + "\"");
    };
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
RegTools make_reg_tools(std::string root_dir) {
    auto store = std::make_shared<RegStore>(std::move(root_dir));

    RegTools out;

    out.save = Tool::builder("reg_save")
        .describe(
            "Save a piece of knowledge to the persistent REGistry — your long-term "
            "memory across sessions. USE THIS AGGRESSIVELY for anything worth "
            "remembering: the user's stated preferences and constraints, project "
            "structure and decisions you've learned, technical facts you had to look "
            "up, recipes / commands that worked, error patterns and their fixes, "
            "domain knowledge from documents the user fed you. The more carefully "
            "you populate the registry, the smarter you become over time — future "
            "conversations will reg_search and find what THIS conversation taught "
            "you.\n"
            "\n"
            "Pick a short descriptive title (letters / digits / dash / underscore "
            "only — no spaces) and 2-5 short keywords so reg_search can find it later. "
            "Keywords should be stable and reusable: 'user-prefs', 'project-easyai', "
            "'cmd-recipe', 'fix-vulkan-radv'. Overwrites if a title already exists "
            "(useful for refining notes as you learn more). If an entry becomes "
            "stale or wrong, use reg_delete to remove it — keeping the registry "
            "tidy makes future searches sharper. Each entry is a small Markdown "
            "file on disk (title.md) so the operator can hand-edit too."
        )
        .param("title",   "string",
               "Identifier. 1..64 chars. Letters, digits, dash, underscore only. "
               "Becomes the filename on disk. Examples: 'gustavo-ai-box', "
               "'easyai-build-recipe', 'qwen3-thinking-fix'.", true)
        .param("keywords",    "array",
               "1..8 short keywords classifying this entry. Same character set as "
               "title. Use stable reusable keywords so future reg_search calls find "
               "this entry. Example: [\"user-prefs\", \"hardware\", \"radv\"].",
               true)
        .param("content", "string",
               "The actual content to remember. Free-form UTF-8 text up to 256 KB. "
               "Markdown is fine; structured snippets fine; prose fine. Be "
               "specific — vague notes won't help future-you.", true)
        .handle(make_save_handler(store))
        .build();

    out.search = Tool::builder("reg_search")
        .describe(
            "Search the REGistry by keyword. Returns up to 20 matching entries with "
            "their title, keywords, modification time, and a short content preview. "
            "USE THIS BEFORE assuming you don't know something the user might "
            "have told you in a past session — your past self may have already "
            "saved the answer. After this returns, pick the 1-4 most relevant "
            "titles and use reg_load to read their full content.\n\n"
            "Returns newest-first. If no entries match, the result tells you so "
            "(not an error) — try a different keyword, or use reg_list to browse."
        )
        .param("keyword",         "string",
               "Keyword to search for. Single keyword, exact match (case-sensitive). "
               "1..32 chars, [A-Za-z0-9_-]. Example: 'user-prefs'.", true)
        .param("max_results", "integer",
               "Maximum entries to return (default 10, max 20).", false)
        .handle(make_search_handler(store))
        .build();

    out.load = Tool::builder("reg_load")
        .describe(
            "Load up to 4 entries from the REGistry by exact title and return "
            "their FULL content. Use this after reg_search to read the bodies "
            "of entries that looked promising in the preview. Pass exact titles "
            "from a previous reg_search result.\n\n"
            "Cap is 4 per call. If you need more than 4, you're probably "
            "trying to drown the prompt in stale content; narrow your search "
            "first."
        )
        .param("titles", "array",
               "Array of 1..4 exact entry titles to load. "
               "Example: [\"gustavo-ai-box\", \"easyai-build-recipe\"].", true)
        .handle(make_load_handler(store))
        .build();

    out.list = Tool::builder("reg_list")
        .describe(
            "List REGistry titles, optionally filtered by title prefix. Use to "
            "browse what you've saved when you're not sure what keyword to search by, "
            "or to confirm whether you've saved a particular note. Returns title, "
            "keywords, content_bytes, and modified time — body NOT included (use "
            "reg_load for that). Untagged entries (operator-dropped notes with "
            "no keywords: header) show here but don't appear in reg_search."
        )
        .param("prefix", "string",
               "Optional prefix to filter titles by. Empty = list all. "
               "Same character set as title. Example: 'easyai-'.", false)
        .param("max",    "integer",
               "Maximum entries to return (default 50, max 200).", false)
        .handle(make_list_handler(store))
        .build();

    out.del = Tool::builder("reg_delete")
        .describe(
            "Delete a REGistry entry by exact title. Use this when an entry has "
            "become stale, wrong, or just irrelevant — keeping the registry "
            "tidy makes future reg_search results sharper. Common reasons to "
            "delete:\n"
            "  - the user corrected something and the old note is now wrong\n"
            "  - a project / preference / fact has changed\n"
            "  - you want to refine a note: delete + reg_save replaces it\n"
            "    cleanly (reg_save also overwrites, but delete-first is\n"
            "    appropriate when the title or keywords need to change too)\n"
            "  - the entry was a one-off scratch note that's no longer needed\n"
            "\n"
            "Idempotent: deleting a non-existent title is not an error. The "
            "deletion is permanent — there's no trash. Be certain before "
            "calling."
        )
        .param("title", "string",
               "Exact title to delete. Letters, digits, dash, underscore. "
               "1..64 chars.", true)
        .handle(make_delete_handler(store))
        .build();

    return out;
}

}  // namespace easyai::tools
