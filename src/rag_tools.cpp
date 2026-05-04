// src/rag_tools.cpp — implementation of the RAG persistent registry.
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
//     `rag_load` does the single read for the body when the
//     model picks an entry.
//
//   * The title is the filesystem name. Validating it with a
//     strict regex (`^[A-Za-z0-9._+-]+$`, ≤ 64 bytes) is what
//     closes the path-traversal door — there is no way for the
//     model to write `..`, slashes, NUL bytes, etc.

#include "easyai/rag_tools.hpp"
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
#include <shared_mutex>
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

// rag_load can fan out to up to 4 entries per call. The cap is
// deliberate: more than 4 means the model is probably about to
// drown the prompt in stale content rather than focusing on what
// matters, and the agent loop stays cheaper too.
constexpr std::size_t kMaxLoadAtOnce      = 4;

// Search result cap. The model gets a list of (title, keywords,
// preview) and picks 1..4 to rag_load — so we don't need a huge
// list, just enough to cover obvious keyword clusters.
constexpr std::size_t kSearchResultsMax   = 20;
constexpr std::size_t kSearchResultsDflt  = 10;

// Bytes from the body we render in a search-result preview. Long
// enough for the model to recognise "yes, that's the entry I want",
// short enough to keep the prompt slim.
constexpr std::size_t kSearchPreviewBytes = 240;

// rag_list cap. Browsing-only; no body read.
constexpr std::size_t kListResultsMax     = 200;
constexpr std::size_t kListResultsDflt    = 50;

// rag_keywords cap. Vocabulary overview — the model uses this to
// see which keywords it's already been using before saving a new
// entry. A typical RAG converges to a few dozen stable keywords;
// the higher cap is for power users with deep vocabularies.
constexpr std::size_t kKeywordsResultsMax  = 500;
constexpr std::size_t kKeywordsResultsDflt = 200;

// On-disk file extension. `.md` keeps entries human-readable in
// any text editor, lets the operator `cat` / `vim` / `grep`, and
// renders nicely as markdown when the body uses it.
constexpr char        kEntrySuffix[]      = ".md";
constexpr std::size_t kEntrySuffixLen     = sizeof(kEntrySuffix) - 1;

// Title prefix that marks a memory as IMMUTABLE. Memories with a
// title starting `fix-easyai-` cannot be overwritten by rag_save and
// cannot be removed by rag_delete — they survive every session until
// the operator deletes the file from disk by hand. Used to seed the
// agent with system designs, hard rules, domain knowledge that must
// not drift. The prefix is part of the title (so it shows in every
// search/list/load result) and lives in the keyword namespace
// `[A-Za-z0-9._+-]` so existing validation still applies.
constexpr char        kFixedTitlePrefix[]    = "fix-easyai-";
constexpr std::size_t kFixedTitlePrefixLen   = sizeof(kFixedTitlePrefix) - 1;

bool title_is_fixed(const std::string & title) {
    return title.size() > kFixedTitlePrefixLen
        && title.compare(0, kFixedTitlePrefixLen, kFixedTitlePrefix) == 0;
}

// Render a unix timestamp as "YYYY-MM-DD HH:MM:SS" in local time. Used
// in search/list/load output so the model can reason about recency
// without doing the math on a raw integer. Local time matches what the
// operator sees in `ls -l`. Returns "?" on a zero/negative epoch (e.g.
// last_write_time failed) so output stays parseable either way.
std::string format_local_time(std::int64_t unix_seconds) {
    if (unix_seconds <= 0) return "?";
    std::time_t t = static_cast<std::time_t>(unix_seconds);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Validators — pure, no I/O, no allocation past the input.
// ---------------------------------------------------------------------------
//
// Allowed character set for keywords AND titles:
//   [a-zA-Z0-9._+-]
//
// Why each non-alnum is allowed (or not):
//
//   `-` `_`  classic word separators
//   `.`      versions ("v1.0"), namespaces ("project.easyai"),
//            file references ("nginx.conf"). REQUIRES extra title
//            validation (see is_valid_title) so `.` / `..` /
//            leading-dot can't sneak through to the filesystem.
//   `+`      niche but real: "c++", "git+ssh", "a+b" recipes.
//   space    NO — filesystem ambiguity, shell-quoting trap.
//   `/` `\`  NO — path-component separators on every OS.
//   `:`      NO — reserved on Windows + ADS-style abuse.
//   anything else (quotes, $, `, etc.): NO — shell / display traps.
//
// Keywords use is_valid_id directly. Titles use is_valid_title,
// which adds filesystem-specific rejections on top.
bool is_valid_id(const std::string & s, std::size_t max_len) {
    if (s.empty() || s.size() > max_len) return false;
    for (char c : s) {
        const bool ok =
            (c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '-' || c == '_' || c == '.' || c == '+';
        if (!ok) return false;
    }
    return true;
}

// is_valid_title — id rules PLUS filesystem-safety constraints:
//
//   * exact "." and ".." are rejected (POSIX path-traversal aliases —
//     even though our regex already blocks slashes, allowing ".."
//     as a title means a hand-edited dir with `..md` is ambiguous
//     and confuses operators).
//   * leading "." is rejected — dotfiles are easy to miss in `ls`,
//     and the agent's persistent memory shouldn't be hidden by
//     accident.
//   * the title must contain at least one alnum — purely-symbol
//     titles like "...", "+--", "_._" are valid by character set
//     but useless and confusing on disk.
//
// Returns false in all those cases; passes anything is_valid_id
// would pass that doesn't trip the extras.
bool is_valid_title(const std::string & s, std::size_t max_len) {
    if (!is_valid_id(s, max_len)) return false;
    if (s == "." || s == "..")    return false;
    if (s.front() == '.')         return false;
    for (char c : s) {
        if ((c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9')) {
            return true;   // contains alnum, all good
        }
    }
    return false;          // no alnum → reject
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
// `rag_search` won't find it (no keywords) but `rag_list` will.
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

// Filesystem mtime as Unix seconds. fs::file_time_type's epoch isn't
// guaranteed by the standard, so we shift it onto system_clock via the
// C++17 idiom (subtract the file_clock "now", add the system_clock
// "now"). Returns 0 when the path can't be stat'd — every caller pairs
// this with other index fields, so a "stat failed" zero keeps the
// in-memory index self-consistent without forcing an error channel
// through three otherwise-different read sites.
inline std::int64_t file_mtime_unix(const fs::path & p) {
    std::error_code ec;
    const auto ftime = fs::last_write_time(p, ec);
    if (ec) return 0;
    const auto sctp = std::chrono::time_point_cast<
        std::chrono::system_clock::duration>(
        ftime - decltype(ftime)::clock::now()
              + std::chrono::system_clock::now());
    return std::chrono::duration_cast<std::chrono::seconds>(
        sctp.time_since_epoch()).count();
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
// RagStore — the shared state for all seven tools.
// ---------------------------------------------------------------------------
//
// `mu` is a shared_mutex so high-concurrency MCP traffic (rag_search /
// rag_list / rag_load / rag_keywords reads from many in-flight requests)
// doesn't serialise on the write path (rag_save, rag_delete). Readers
// take std::shared_lock; writers take std::unique_lock.
//
// The index is populated EAGERLY by `make_rag_tools()` under a unique
// lock so every subsequent reader can rely on `index_loaded == true`
// without the upgrade dance. Single-process is the supported model
// (header §"Concurrency"), so we never re-scan the directory after
// startup; save/delete keep the index in sync as the model edits.
struct RagStore {
    fs::path                            root;
    std::shared_mutex                   mu;
    std::map<std::string, EntryMeta>    index;
    bool                                index_loaded = false;

    explicit RagStore(std::string r) {
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
        if (!root_set()) { err = "RAG root path is empty"; return false; }
        std::error_code ec;
        if (fs::exists(root, ec)) {
            if (!fs::is_directory(root, ec)) {
                err = "RAG root is not a directory: " + root.string();
                return false;
            }
            return true;
        }
        fs::create_directories(root, ec);
        if (ec) {
            err = "create RAG dir failed: " + ec.message();
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
            if (!is_valid_title(title, kMaxTitleBytes)) continue;

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
            m.modified_unix = file_mtime_unix(e.path());
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
                err = "cannot open RAG tempfile in " + root.string();
                return false;
            }
            out.write(text.data(), (std::streamsize) text.size());
            out.flush();
            if (!out) {
                err = "write to RAG tempfile failed";
                std::error_code ec;
                fs::remove(tmp, ec);
                return false;
            }
        }
        // Tighten permissions BEFORE rename so the new entry is never
        // visible to other users on disk, even briefly. The process
        // umask defaults to 022 on most systems → files end up 0644
        // (world-readable). RAG entries can carry sensitive content
        // the model was told to memorise; lock to owner-only.
        {
            std::error_code ec;
            fs::permissions(tmp, fs::perms::owner_read | fs::perms::owner_write,
                            fs::perm_options::replace, ec);
            (void) ec;   // best-effort; rename below is the durable step
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
        m.modified_unix = file_mtime_unix(target);
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
        if (!root_set()) { err = "RAG root path is empty"; return false; }
        const auto target = root / (title + kEntrySuffix);
        std::error_code ec;
        if (!fs::exists(target, ec)) {
            err = "no RAG entry titled \"" + title + "\"";
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
        modified_unix_out = file_mtime_unix(target);
        return true;
    }

    // Delete an entry. Idempotent: deleting a non-existent title
    // returns true with `existed = false` so the caller can decide
    // whether to surface "no such entry" or stay quiet. Removes the
    // index entry too.
    bool delete_locked(const std::string & title, bool & existed,
                       std::string & err) {
        existed = false;
        if (!root_set()) { err = "RAG root path is empty"; return false; }
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

ToolHandler make_save_handler(std::shared_ptr<RagStore> store) {
    return [store](const ToolCall & c) -> ToolResult {
        std::string title;
        if (!args::get_string(c.arguments_json, "title", title) || title.empty()) {
            return ToolResult::error("missing required argument: title");
        }

        // `fix=true` promotes the memory to immutable. Two ways to ask
        // for it: (a) pass fix=true and any title — we auto-prepend the
        // kFixedTitlePrefix so the immutability invariant lives in the
        // filename itself. (b) pass a title that already starts with
        // kFixedTitlePrefix — we honour that, fix=true is implied. The
        // invariant we maintain: an entry is fixed IFF its title starts
        // with kFixedTitlePrefix, so search / load / delete only need
        // the title to know.
        bool fix = false;
        args::get_bool(c.arguments_json, "fix", fix);
        if (fix && !title_is_fixed(title)) {
            title = std::string(kFixedTitlePrefix) + title;
        }

        if (!is_valid_title(title, kMaxTitleBytes)) {
            return ToolResult::error(
                "title \"" + title + "\" is invalid; must match "
                "[A-Za-z0-9._+-]{1," + std::to_string(kMaxTitleBytes) + "} "
                "(fix=true auto-prepends \"" + std::string(kFixedTitlePrefix)
                + "\" — keep the rest under the title length cap)");
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
                + " short keywords). Why: keywords are how rag_search finds this "
                  "memory later — a memory with no keywords is unreachable by "
                  "keyword search (only rag_list can find it).");
        }
        if (keywords.size() > kMaxKeywordsPerEntry) {
            return ToolResult::error(
                "too many keywords (max " + std::to_string(kMaxKeywordsPerEntry) + ")");
        }
        for (const auto & t : keywords) {
            if (!is_valid_id(t, kMaxKeywordBytes)) {
                return ToolResult::error(
                    "keyword \"" + t + "\" is invalid; must match "
                    "[A-Za-z0-9._+-]{1," + std::to_string(kMaxKeywordBytes) + "}");
            }
        }

        std::string content;
        if (!args::get_string(c.arguments_json, "content", content)) {
            return ToolResult::error("missing required argument: content");
        }
        if (content.size() > kMaxContentBytes) {
            return ToolResult::error(
                "content exceeds " + std::to_string(kMaxContentBytes)
                + " bytes; split into multiple memories");
        }

        // WRITE: takes unique_lock so concurrent readers can't observe a
        // half-updated index. The on-disk write inside save_locked is
        // already atomic (tempfile + rename) so reads through the
        // FILESYSTEM are tear-free regardless of this lock.
        std::unique_lock<std::shared_mutex> lock(store->mu);

        // Immutability gate: an existing fixed entry can never be
        // overwritten — not even by another fix=true save. The check
        // runs UNDER the unique_lock so the index is authoritative
        // (load_index_locked ran at startup; saves keep it in sync).
        if (title_is_fixed(title) && store->index.count(title) > 0) {
            return ToolResult::error(
                "memory \"" + title + "\" is fixed (immutable) — cannot "
                "overwrite. To replace it, the operator must remove the "
                "file from disk manually. Pick a different title for a "
                "new memory.");
        }

        if (!store->save_locked(title, keywords, content, err)) {
            return ToolResult::error(err);
        }

        std::ostringstream o;
        o << "saved \"" << title << kEntrySuffix << "\" ("
          << content.size() << " bytes, "
          << keywords.size() << " keyword" << (keywords.size() == 1 ? "" : "s")
          << (title_is_fixed(title) ? ", FIXED — immutable from now on" : "")
          << ")";
        return ToolResult::ok(o.str());
    };
}

// rag_append — read-modify-write an existing memory.
//
// Concurrency contract (this is the part that has to be right):
//   * The whole RMW (existence check + load_one + merge +
//     save_locked) runs under ONE std::unique_lock<shared_mutex>,
//     same writer-discipline as rag_save / rag_delete. So all
//     four scenarios serialise correctly:
//       - two threads appending to the SAME title  → ordered;
//         both appendices land, last writer's appendix is last.
//       - two threads appending to DIFFERENT titles → still
//         serialised (one shared_mutex per RagStore); cheap
//         compared to disk I/O.
//       - append vs concurrent rag_save / rag_delete → also
//         under unique_lock; whichever lands first wins, the
//         other sees the post-state (not-found or merged-content).
//       - append vs concurrent reads (rag_search / rag_load /
//         rag_list / rag_keywords) → readers hold shared_lock,
//         block until our write commits, then proceed.
//   * save_locked writes via tempfile + rename(2), so a reader
//     that obtains the file path some other way (e.g. another
//     process slurp_capped'ing the .md directly) sees either the
//     old full body or the new merged body — never a half-applied
//     append. The single-process invariant in the header
//     ("Concurrency: ... single-process is the supported model")
//     means cross-process semantics are best-effort, not contract.
//   * load_one and save_locked are designed to be called WITH the
//     unique_lock already held — they don't re-acquire, so there
//     is no upgrade dance and no risk of self-deadlock.
ToolHandler make_append_handler(std::shared_ptr<RagStore> store) {
    return [store](const ToolCall & c) -> ToolResult {
        std::string title;
        if (!args::get_string(c.arguments_json, "title", title) || title.empty()) {
            return ToolResult::error("missing required argument: title");
        }
        if (!is_valid_title(title, kMaxTitleBytes)) {
            return ToolResult::error(
                "title \"" + title + "\" is invalid; must match "
                "[A-Za-z0-9._+-]{1," + std::to_string(kMaxTitleBytes) + "}");
        }

        std::string suffix;
        if (!args::get_string(c.arguments_json, "content", suffix)) {
            return ToolResult::error("missing required argument: content");
        }
        if (suffix.empty()) {
            return ToolResult::error(
                "content is empty — nothing to append. If you want to "
                "replace the whole memory, use rag_save with the same title.");
        }

        // Optional: extra keywords to merge into the existing list.
        // Validated up-front (before we hold the lock) so a malformed
        // call returns a clean error without churning state.
        std::vector<std::string> extra_keywords;
        std::string err;
        if (args::has(c.arguments_json, "keywords")
                && !parse_string_array(c.arguments_json, "keywords",
                                       extra_keywords, err)) {
            return ToolResult::error(err);
        }
        for (const auto & k : extra_keywords) {
            if (!is_valid_id(k, kMaxKeywordBytes)) {
                return ToolResult::error(
                    "keyword \"" + k + "\" is invalid; must match "
                    "[A-Za-z0-9._+-]{1," + std::to_string(kMaxKeywordBytes) + "}");
            }
        }

        // WRITE: unique_lock for the whole RMW so a concurrent
        // rag_save (also unique_lock) can't slip between our read of
        // the existing body and the rewrite of the merged content.
        std::unique_lock<std::shared_mutex> lock(store->mu);

        // Existence check before we touch disk: the model should
        // know to call rag_save instead of rag_append for a brand-
        // new memory. We use the in-memory index as the source of
        // truth (load_index_locked ran at startup).
        if (store->index.count(title) == 0) {
            return ToolResult::error(
                "no memory titled \"" + title + "\" — use rag_save to "
                "create a new memory, or rag_list / rag_search to find "
                "the title you meant.");
        }

        // Immutability gate: fixed memories live forever as written.
        // Even an append would mutate them, so refuse the same way
        // rag_save and rag_delete do.
        if (title_is_fixed(title)) {
            return ToolResult::error(
                "memory \"" + title + "\" is fixed (immutable) — cannot "
                "append. Pick a different title for a related memory, "
                "or have the operator remove the file from disk first.");
        }

        // Read the existing body + keywords back from disk via the
        // store helper (slurp + parse_entry, capped at
        // kMaxContentBytes + 4 KiB header room — same cap rag_load
        // uses, so an oversized hand-edited file is rejected with
        // the same message the rest of the surface produces).
        std::vector<std::string> old_keywords;
        std::string              old_body;
        std::int64_t             old_mtime = 0;   // unused but required by the API
        if (!store->load_one(title, old_keywords, old_body, old_mtime, err)) {
            return ToolResult::error(err);
        }

        // Compose the merged body. We insert a Markdown horizontal
        // rule as the separator so the operator opening the .md file
        // sees exactly where the appendix begins. Trim any trailing
        // newlines from the existing body first so the rule sits on
        // a clean blank line regardless of how the previous save
        // happened to terminate.
        std::string merged = old_body;
        while (!merged.empty() && (merged.back() == '\n' || merged.back() == '\r')) {
            merged.pop_back();
        }
        if (!merged.empty()) merged += "\n\n---\n\n";
        merged += suffix;

        if (merged.size() > kMaxContentBytes) {
            return ToolResult::error(
                "appended memory would exceed " + std::to_string(kMaxContentBytes)
                + " bytes (existing " + std::to_string(old_body.size())
                + " B + appendix " + std::to_string(suffix.size())
                + " B + separator). Split into a new memory with rag_save "
                  "instead, or condense the appendix.");
        }

        // Merge keywords: keep the existing order (the model relies on
        // search ranking that's stable across appends), then append any
        // extras the model passed that weren't already there. Cap at
        // kMaxKeywordsPerEntry; oldest stays.
        std::vector<std::string> merged_keywords = old_keywords;
        for (const auto & k : extra_keywords) {
            const bool already = std::find(merged_keywords.begin(),
                                           merged_keywords.end(), k)
                                 != merged_keywords.end();
            if (!already && merged_keywords.size() < kMaxKeywordsPerEntry) {
                merged_keywords.push_back(k);
            }
        }
        if (merged_keywords.empty()) {
            // Defensive — every saved memory has at least one keyword
            // (rag_save enforces it). If somehow we read back an entry
            // with none (operator hand-edited the keywords: header out),
            // require the model to supply them on append rather than
            // writing a search-invisible memory.
            return ToolResult::error(
                "existing memory has no keywords (someone may have hand-"
                "edited it); pass keywords[] to rag_append so the merged "
                "memory remains searchable.");
        }

        if (!store->save_locked(title, merged_keywords, merged, err)) {
            return ToolResult::error(err);
        }

        std::ostringstream o;
        o << "appended to \"" << title << kEntrySuffix << "\" ("
          << "+" << suffix.size() << " B → " << merged.size() << " B total, "
          << merged_keywords.size() << " keyword"
          << (merged_keywords.size() == 1 ? "" : "s") << ")";
        return ToolResult::ok(o.str());
    };
}

ToolHandler make_search_handler(std::shared_ptr<RagStore> store) {
    return [store](const ToolCall & c) -> ToolResult {
        // Multi-keyword search.
        //   - 1 keyword passed:  match entries that have THAT keyword
        //                        (require ≥1 match — same as the
        //                        original single-keyword behaviour)
        //   - 2+ keywords passed: match entries that have AT LEAST 2
        //                        of the queried keywords (require ≥2)
        //
        // The threshold is adaptive on purpose: a single-keyword query
        // is still a useful broad sweep, but the moment the model
        // sends two or more it's clearly trying to narrow — and we
        // reward that by demanding overlap. Each result reports how
        // many of the queried keywords it matched, so the model can
        // rank / pick.
        std::vector<std::string> keywords;
        std::string err;
        if (!parse_string_array(c.arguments_json, "keywords", keywords, err)) {
            return ToolResult::error(err);
        }
        if (keywords.empty()) {
            return ToolResult::error(
                "missing required argument: keywords (non-empty array)");
        }
        if (keywords.size() > kMaxKeywordsPerEntry) {
            return ToolResult::error(
                "too many keywords (max "
                + std::to_string(kMaxKeywordsPerEntry) + " per query)");
        }
        for (const auto & k : keywords) {
            if (!is_valid_id(k, kMaxKeywordBytes)) {
                return ToolResult::error(
                    "keyword \"" + k + "\" is invalid; must match "
                    "[A-Za-z0-9._+-]{1," + std::to_string(kMaxKeywordBytes) + "}");
            }
        }
        // Deduplicate the query — repeated keywords would otherwise
        // distort the match count.
        {
            std::sort(keywords.begin(), keywords.end());
            keywords.erase(std::unique(keywords.begin(), keywords.end()),
                           keywords.end());
        }
        const std::size_t min_matches = (keywords.size() >= 2) ? 2 : 1;

        long long max_results = (long long) kSearchResultsDflt;
        args::get_int(c.arguments_json, "max_results", max_results);
        if (max_results < 1) max_results = 1;
        if ((std::size_t) max_results > kSearchResultsMax) {
            max_results = (long long) kSearchResultsMax;
        }

        struct Hit {
            std::string title;
            EntryMeta   meta;
            std::size_t matched = 0;
            std::vector<std::string> matched_keywords;
        };
        std::vector<Hit> hits;
        {
            // READ: shared_lock — many concurrent searches can iterate
            // the index in parallel. The index was populated eagerly by
            // make_rag_tools() under a unique lock, so we never need to
            // upgrade here.
            std::shared_lock<std::shared_mutex> lock(store->mu);
            for (const auto & [t, m] : store->index) {
                Hit h;
                h.title = t;
                h.meta  = m;
                for (const auto & q : keywords) {
                    for (const auto & et : m.keywords) {
                        if (et == q) {
                            h.matched_keywords.push_back(q);
                            break;
                        }
                    }
                }
                h.matched = h.matched_keywords.size();
                if (h.matched >= min_matches) {
                    hits.push_back(std::move(h));
                }
            }
        }
        // Rank: more overlap first, ties broken by recency.
        std::sort(hits.begin(), hits.end(), [](const Hit & a, const Hit & b) {
            if (a.matched != b.matched) return a.matched > b.matched;
            return a.meta.modified_unix > b.meta.modified_unix;
        });

        // Pagination — the model can ask for `page=N` to walk the rest
        // of a large result set without re-issuing a different query.
        // We compute totals on the FULL ranked list, then slice.
        long long page = 1;
        args::get_int(c.arguments_json, "page", page);
        if (page < 1) page = 1;

        const std::size_t total       = hits.size();
        const std::size_t per_page    = (std::size_t) max_results;
        const std::size_t total_pages =
            total == 0 ? 0 : (total + per_page - 1) / per_page;
        const std::size_t offset      =
            (std::size_t)((page - 1) * (long long) per_page);

        // Build a "queried [a, b, c]" string once for the response prose.
        std::string queried_str;
        for (std::size_t i = 0; i < keywords.size(); ++i) {
            queried_str += (i ? ", " : "");
            queried_str += keywords[i];
        }

        if (hits.empty()) {
            std::ostringstream o;
            o << "total_entries: 0\n";
            o << "page: " << page << " of 0\n\n";
            o << "no entries match";
            if (keywords.size() == 1) {
                o << " keyword \"" << queried_str << "\"";
            } else {
                o << " at least " << min_matches
                  << " of [" << queried_str << "]";
            }
            o << ". Use rag_list to browse, or rag_save to add new entries.";
            return ToolResult::ok(o.str());
        }

        if (offset >= total) {
            std::ostringstream o;
            o << "total_entries: " << total << "\n";
            o << "page: " << page << " of " << total_pages
              << "  (past the end)\n\n";
            o << "page " << page << " is past the last page ("
              << total_pages << "). The full result set is "
              << total << " entr" << (total == 1 ? "y" : "ies")
              << " — use page=1.." << total_pages << ".";
            return ToolResult::ok(o.str());
        }

        const std::size_t slice_end = std::min(offset + per_page, total);
        const std::size_t shown     = slice_end - offset;
        const bool        has_more  = slice_end < total;

        // Render plain-text + structured (markdown-friendly) output.
        // The model parses this with no JSON dependency on its side
        // and the operator can `cat` it from a journal log.
        //
        // Header layout (machine-readable lines first, then prose) so
        // the model can grep `total_entries:` / `page:` / `has_more:`
        // without parsing the body.
        std::ostringstream o;
        o << "total_entries: " << total << "\n";
        o << "page: "          << page << " of " << total_pages << "\n";
        o << "showing: "       << shown
          << "  (entries " << (offset + 1) << ".." << slice_end << ")\n";
        o << "has_more: "      << (has_more ? "true" : "false") << "\n\n";

        if (keywords.size() == 1) {
            o << "match keyword \"" << queried_str
              << "\" (newest first):\n\n";
        } else {
            o << "match ≥" << min_matches << " of ["
              << queried_str
              << "] (best-overlap first, then newest):\n\n";
        }
        for (std::size_t i = offset; i < slice_end; ++i) {
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

            // Number entries by their absolute position in the ranked
            // list so the model can correlate across pages.
            o << (i + 1) << ". " << h.title;
            if (title_is_fixed(h.title)) o << "  [FIXED]";
            if (keywords.size() > 1) {
                o << "  [matched " << h.matched << "/" << keywords.size()
                  << ": ";
                for (std::size_t k = 0; k < h.matched_keywords.size(); ++k) {
                    if (k) o << ", ";
                    o << h.matched_keywords[k];
                }
                o << "]";
            }
            o << "\n";
            o << "   keywords: ";
            for (std::size_t k = 0; k < h.meta.keywords.size(); ++k) {
                if (k) o << ", ";
                o << h.meta.keywords[k];
            }
            o << "  (" << h.meta.content_bytes << " bytes)\n";
            o << "   modified: " << format_local_time(h.meta.modified_unix)
              << "  (unix=" << h.meta.modified_unix << ")\n";
            // Indent preview lines by 3 so it's easy to skim.
            std::string preview_in;
            preview_in.reserve(preview.size() + preview.size() / 60 * 3);
            for (char ch : preview) {
                preview_in += ch;
                if (ch == '\n') preview_in += "   ";
            }
            o << "   " << preview_in << "\n\n";
        }
        if (has_more) {
            o << "Use rag_search with the same keywords + page="
              << (page + 1) << " to see the next "
              << std::min(per_page, total - slice_end)
              << " result" << (total - slice_end == 1 ? "" : "s")
              << " (page " << (page + 1) << " of " << total_pages << ").\n";
        }
        o << "Use rag_load with up to " << kMaxLoadAtOnce
          << " of these titles for full content.\n";
        return ToolResult::ok(o.str());
    };
}

ToolHandler make_load_handler(std::shared_ptr<RagStore> store) {
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
                + " per call); narrow your rag_search first");
        }
        for (const auto & t : titles) {
            if (!is_valid_title(t, kMaxTitleBytes)) {
                return ToolResult::error(
                    "title \"" + t + "\" is invalid; must match "
                    "[A-Za-z0-9._+-]{1," + std::to_string(kMaxTitleBytes) + "} "
                    "(no leading dot, not '.' or '..', must contain alnum)");
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
            // READ: shared_lock — load_one() reads the file off disk
            // (atomic-rename guarantees a consistent view) and never
            // touches the index. Multiple parallel rag_load calls run
            // concurrently with no contention on the mutex itself.
            std::shared_lock<std::shared_mutex> lock(store->mu);
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
            o << "\nmodified: " << format_local_time(mtime)
              << "  (unix=" << mtime << ")\n";
            o << "fixed: " << (title_is_fixed(title) ? "yes" : "no") << "\n";
            if (title_is_fixed(title)) {
                o << "note: this memory is immutable — rag_save and rag_delete "
                     "will refuse to change or remove it.\n";
            }
            o << "\n";
            o << body;
            if (!body.empty() && body.back() != '\n') o << '\n';
        }
        return ToolResult::ok(o.str());
    };
}

ToolHandler make_list_handler(std::shared_ptr<RagStore> store) {
    return [store](const ToolCall & c) -> ToolResult {
        std::string prefix;
        args::get_string(c.arguments_json, "prefix", prefix);
        if (!prefix.empty() && !is_valid_title(prefix, kMaxTitleBytes)) {
            return ToolResult::error(
                "prefix \"" + prefix + "\" is invalid; must match "
                "[A-Za-z0-9._+-]{1," + std::to_string(kMaxTitleBytes) + "}");
        }
        long long max = (long long) kListResultsDflt;
        args::get_int(c.arguments_json, "max", max);
        if (max < 1) max = 1;
        if ((std::size_t) max > kListResultsMax) max = (long long) kListResultsMax;

        struct Row { std::string title; EntryMeta meta; };
        std::vector<Row> rows;
        {
            // READ: shared_lock — same justification as rag_search.
            std::shared_lock<std::shared_mutex> lock(store->mu);
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
                ? "RAG is empty. Use rag_save to add entries."
                : "no titles match prefix \"" + prefix + "\".");
        }

        std::ostringstream o;
        o << rows.size() << " entr" << (rows.size() == 1 ? "y" : "ies");
        if (!prefix.empty()) o << " matching prefix \"" << prefix << "\"";
        o << ":\n\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto & r = rows[i];
            o << (i + 1) << ". " << r.title;
            if (title_is_fixed(r.title)) o << "  [FIXED]";
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
            o << "  modified=" << format_local_time(r.meta.modified_unix) << "\n";
        }
        return ToolResult::ok(o.str());
    };
}

ToolHandler make_delete_handler(std::shared_ptr<RagStore> store) {
    return [store](const ToolCall & c) -> ToolResult {
        std::string title;
        if (!args::get_string(c.arguments_json, "title", title) || title.empty()) {
            return ToolResult::error("missing required argument: title");
        }
        if (!is_valid_title(title, kMaxTitleBytes)) {
            return ToolResult::error(
                "title \"" + title + "\" is invalid; must match "
                "[A-Za-z0-9._+-]{1," + std::to_string(kMaxTitleBytes) + "}");
        }
        // Fixed memories are immutable by design — refuse the delete
        // before we even take the write lock. The operator can still
        // remove the file from disk by hand if they truly need to;
        // exposing that path through a tool defeats the whole point of
        // the prefix.
        if (title_is_fixed(title)) {
            return ToolResult::error(
                "memory \"" + title + "\" is fixed (immutable) — cannot "
                "be forgotten through this tool. The operator can remove "
                "the file from disk manually if it really needs to go.");
        }
        // WRITE: unique_lock — delete_locked mutates the index AND
        // removes the on-disk file. fs::remove is itself atomic, so
        // parallel readers either see the entry or don't, never a
        // half-deleted state.
        std::unique_lock<std::shared_mutex> lock(store->mu);
        bool existed = false;
        std::string err;
        if (!store->delete_locked(title, existed, err)) {
            return ToolResult::error(err);
        }
        if (!existed) {
            return ToolResult::ok(
                "no memory titled \"" + title + "\" — nothing to forget");
        }
        return ToolResult::ok(
            "forgot \"" + title + kEntrySuffix + "\"");
    };
}

// rag_keywords — vocabulary overview. Returns each distinct
// keyword used across the RAG together with how many entries
// reference it. The model uses this to:
//
//   - learn its own vocabulary before saving (avoid creating a
//     new keyword like `user_pref` when `user-prefs` already
//     exists)
//   - discover dimensions of stored knowledge it forgot about
//   - frame rag_search queries against keywords that actually
//     return results
//
// Output: header lines (`total_keywords:`, `total_entries:`,
// `showing:`) followed by sorted rows. Sort order: count
// descending, then keyword name ascending so the response is
// stable across calls.
ToolHandler make_keywords_handler(std::shared_ptr<RagStore> store) {
    return [store](const ToolCall & c) -> ToolResult {
        long long min_count = 1;
        args::get_int(c.arguments_json, "min_count", min_count);
        if (min_count < 1) min_count = 1;

        long long max = (long long) kKeywordsResultsDflt;
        args::get_int(c.arguments_json, "max", max);
        if (max < 1) max = 1;
        if ((std::size_t) max > kKeywordsResultsMax) {
            max = (long long) kKeywordsResultsMax;
        }

        // Phase 1: under shared_lock, collect counts from the index.
        // READ — same justification as rag_search; many concurrent
        // rag_keywords calls run in parallel without serialising.
        std::map<std::string, std::size_t> counts;
        std::size_t total_entries = 0;
        {
            std::shared_lock<std::shared_mutex> lock(store->mu);
            total_entries = store->index.size();
            for (const auto & [_, m] : store->index) {
                for (const auto & k : m.keywords) {
                    counts[k] += 1;
                }
            }
        }

        // Phase 2: filter by min_count and sort by (count desc, name asc).
        struct Row { std::string keyword; std::size_t count; };
        std::vector<Row> rows;
        rows.reserve(counts.size());
        for (const auto & [k, n] : counts) {
            if (n >= (std::size_t) min_count) {
                rows.push_back({ k, n });
            }
        }
        std::sort(rows.begin(), rows.end(), [](const Row & a, const Row & b) {
            if (a.count != b.count) return a.count > b.count;
            return a.keyword < b.keyword;
        });
        const std::size_t total_kw = rows.size();
        if ((long long) rows.size() > max) {
            rows.resize((std::size_t) max);
        }

        // Render.
        std::ostringstream o;
        o << "total_keywords: " << total_kw << "\n";
        o << "total_entries: "  << total_entries << "\n";
        o << "showing: "        << rows.size();
        if (min_count > 1) {
            o << "  (min_count=" << min_count << ")";
        }
        o << "\n\n";

        if (rows.empty()) {
            if (total_entries == 0) {
                o << "RAG is empty. Use rag_save to add the first entry.";
            } else if (min_count > 1) {
                o << "no keywords reach min_count=" << min_count
                  << ". The RAG has " << total_entries
                  << " entr" << (total_entries == 1 ? "y" : "ies")
                  << " but every keyword is below the threshold. "
                  << "Try min_count=1 (default) to see the full list.";
            } else {
                o << "no keywords found. Some entries may be untagged "
                  << "(no `keywords:` header) — those don't appear here. "
                  << "Use rag_list to see them.";
            }
            return ToolResult::ok(o.str());
        }

        // Pad the keyword column for legibility. Cap at the longest
        // keyword in the result so we don't waste tokens.
        std::size_t pad = 0;
        for (const auto & r : rows) pad = std::max(pad, r.keyword.size());
        if (pad > kMaxKeywordBytes) pad = kMaxKeywordBytes;

        for (const auto & r : rows) {
            o << r.keyword;
            for (std::size_t i = r.keyword.size(); i < pad + 2; ++i) {
                o << ' ';
            }
            o << r.count
              << " entr" << (r.count == 1 ? "y" : "ies") << "\n";
        }
        return ToolResult::ok(o.str());
    };
}

// Build a RagStore at `root_dir` and eager-load its index. Shared
// between the seven-tool factory (make_rag_tools, opt-in via
// --split-rag) and the default single-tool factory
// (make_unified_rag_tool) so both shapes see the same on-disk
// content and there's no chance of a divergent eager-load policy.
// After this returns, every read path can take a shared_lock and
// observe `index_loaded == true` without racing.
std::shared_ptr<RagStore> build_rag_store(std::string root_dir) {
    auto store = std::make_shared<RagStore>(std::move(root_dir));
    {
        std::unique_lock<std::shared_mutex> init_lock(store->mu);
        store->load_index_locked();
    }
    return store;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
RagTools make_rag_tools(std::string root_dir) {
    auto store = build_rag_store(std::move(root_dir));

    RagTools out;

    out.save = Tool::builder("rag_save")
        .describe(
            "Store a memory — your long-term memory across sessions. USE THIS "
            "AGGRESSIVELY for anything worth remembering: the user's stated "
            "preferences and constraints, project structure and decisions you've "
            "learned, technical facts you had to look up, recipes / commands "
            "that worked, error patterns and their fixes, domain knowledge from "
            "documents the user fed you. The more carefully you populate your "
            "memory, the smarter you become over time — future conversations "
            "will rag_search and find what THIS conversation taught you.\n"
            "\n"
            "USER FOCUS — prioritise notes about the user themselves. "
            "Whenever the user reveals who they are or how they work — name, "
            "role, hardware, projects, working style, tools they prefer, "
            "things they've corrected you about, things they like, things "
            "they hate — that's a memory worth keeping. Suggested titles: "
            "`user-profile`, `user-prefs`, `user-projects`, `user-hardware`, "
            "`user-corrections`. The next conversation (could be tomorrow, "
            "could be in three months) starts by recalling these, so the "
            "user doesn't have to explain themselves twice. When you learn "
            "ONE MORE thing about the user later, prefer rag_append on the "
            "matching profile memory over rag_save (which would overwrite); "
            "the running profile keeps growing without losing context.\n"
            "\n"
            "Pick a short descriptive title (letters / digits / dash / underscore "
            "only — no spaces) and 2-5 short keywords so rag_search can recall it "
            "later. Keywords should be stable and reusable: 'user-prefs', "
            "'project-easyai', 'cmd-recipe', 'vulkan-radv'.\n"
            "\n"
            "GRANULARITY: prefer many small focused memories over one large one. "
            "Store 'easyai-build-mac' and 'easyai-build-linux' as TWO memories, "
            "not one combined 'easyai-build'. When rag_load returns a memory, "
            "the FULL body becomes part of your prompt — a 200-line memory burns "
            "1000+ tokens you didn't ask for. The search-then-recall flow is "
            "designed so you can pick up to 4 small memories instead of drowning "
            "in one giant one. Rule of thumb: if a body is over ~500 words, "
            "it's probably two or more memories pretending to be one.\n"
            "\n"
            "BEFORE storing, consider calling rag_keywords to see what vocabulary "
            "you've already been using — reusing existing keywords like "
            "'user-prefs' instead of inventing 'preferences' or 'user_pref' "
            "keeps your memory coherent and future searches sharper. Drift in "
            "keyword vocabulary is the #1 way memory slowly becomes useless.\n"
            "\n"
            "UPDATING: rag_save with the same title overwrites the previous "
            "memory — that's how you refine a note as you learn more. To forget "
            "a memory entirely use rag_delete. Each memory is a small Markdown "
            "file on disk (title.md) so the operator can hand-edit too.\n"
            "\n"
            "FIXED MEMORIES (immutable): set fix=true to store a memory the "
            "system MUST NOT alter or forget — system designs, hard rules, "
            "domain definitions, anything the user told you to learn as ground "
            "truth. Fixed memories are recorded with a 'fix-easyai-' title "
            "prefix; rag_save will refuse to overwrite them and rag_delete will "
            "refuse to forget them. Use this when the user says \"learn this "
            "as fixed\", \"remember this as a rule\", \"this is the design — "
            "memorise it\". Once stored as fixed, the only way to change a "
            "memory is for the human operator to remove the file from disk."
        )
        .param("title",   "string",
               "Identifier. 1..64 chars. Letters, digits, dash, underscore only. "
               "Becomes the filename on disk. Examples: 'gustavo-ai-box', "
               "'easyai-build-recipe', 'qwen3-thinking'. With fix=true the "
               "'fix-easyai-' prefix is auto-added if not present.", true)
        .param("keywords",    "array",
               "1..8 short keywords classifying this memory. Same character set "
               "as title. Use stable reusable keywords so future rag_search "
               "calls recall this memory. Example: [\"user-prefs\", \"hardware\", "
               "\"radv\"].",
               true)
        .param("content", "string",
               "The actual content to remember. Free-form UTF-8 text up to 256 KB. "
               "Markdown is fine; structured snippets fine; prose fine. Be "
               "specific — vague notes won't help future-you.", true)
        .param("fix", "boolean",
               "If true, store as IMMUTABLE memory: the title is prefixed with "
               "'fix-easyai-' (auto-added) and this memory cannot be overwritten "
               "by future rag_save or removed by rag_delete. Use only when the "
               "user explicitly asks to learn something as a fixed rule, system "
               "design, or ground-truth definition. Default false.", false)
        .handle(make_save_handler(store))
        .build();

    out.append = Tool::builder("rag_append")
        .describe(
            "Add new content to the end of an EXISTING memory. Use this "
            "when you learned more about something you already wrote down "
            "— refining a user's preferences, accumulating a project's "
            "running log, growing a debugging trail across sessions — "
            "WITHOUT losing the previous content.\n"
            "\n"
            "When to prefer rag_append over rag_save:\n"
            "  - the existing memory is still correct, you just have more\n"
            "    to add (a new fact, a clarification, a follow-up note)\n"
            "  - you're keeping a chronological log (decisions, attempts,\n"
            "    observations about the user)\n"
            "  - the user told you 'add this to what you already know\n"
            "    about X' — append is the natural verb\n"
            "\n"
            "When to use rag_save instead: when the existing memory is\n"
            "wrong / superseded / has the wrong title or wrong keywords.\n"
            "rag_save with the same title overwrites cleanly; rag_append\n"
            "preserves history.\n"
            "\n"
            "How the appendix looks on disk: the new content is added\n"
            "after a Markdown horizontal rule (`---`) so the operator\n"
            "reading the .md file sees exactly where each appendix\n"
            "begins. Multiple appends stack — old → rule → newer →\n"
            "rule → newest.\n"
            "\n"
            "USER FOCUS — track the user as they reveal themselves: their\n"
            "name, role, hardware, projects, working style, tools they\n"
            "prefer, things they've corrected you about, what they liked\n"
            "or disliked. Each conversation is a chance to learn one more\n"
            "thing. Whenever the user says \"I am…\", \"I prefer…\",\n"
            "\"my setup is…\", \"don't…\", \"always…\", that's a memory\n"
            "worth keeping — and on the next conversation, rag_append is\n"
            "how you grow it without losing what you already learned.\n"
            "\n"
            "ERRORS:\n"
            "  - title not found → use rag_save to create it; use\n"
            "    rag_list / rag_search if you're not sure of the exact\n"
            "    title\n"
            "  - title is fixed (`fix-easyai-` prefix) → fixed memories\n"
            "    are immutable; pick a different title for a related\n"
            "    memory (e.g. `<topic>-notes`)\n"
            "  - merged content would exceed 256 KiB → split into a new\n"
            "    memory with rag_save instead, or condense the appendix\n"
            "\n"
            "Optional `keywords` extends the memory's keyword list (for\n"
            "rag_search recall) — existing keywords are preserved, new\n"
            "ones are deduped against them, total still capped at 8."
        )
        .param("title", "string",
               "Exact title of an existing memory. Letters, digits, dash, "
               "underscore. 1..64 chars. Use rag_list or rag_search first "
               "if you don't remember the exact spelling. "
               "Titles starting with 'fix-easyai-' are immutable and "
               "will be rejected.", true)
        .param("content", "string",
               "Text to append after the existing body. Free-form UTF-8 "
               "up to whatever room is left (existing + separator + new ≤ "
               "256 KB). Markdown / prose / code blocks all fine. The "
               "library inserts a `---` Markdown rule before this text so "
               "you don't need to add your own separator.", true)
        .param("keywords", "array",
               "Optional. 0..8 additional keywords to merge into the "
               "memory's keyword list. Existing keywords are preserved; "
               "new ones are deduped and the total stays capped at 8 "
               "(oldest wins on overflow). Use this when the appendix "
               "broadens the memory's topic — e.g. a memory tagged "
               "[\"user-prefs\"] gaining a section about hardware should "
               "add \"hardware\" so future rag_search reaches it.", false)
        .handle(make_append_handler(store))
        .build();

    out.search = Tool::builder("rag_search")
        .describe(
            "Search your memory by one or more keywords. USE THIS BEFORE assuming "
            "you don't know something the user might have told you in a past "
            "session — your past self may have already remembered the answer.\n\n"
            "Not sure which keywords to try? Call rag_keywords first to see "
            "the vocabulary you've actually been using; pick from THERE rather "
            "than guessing. A search against keywords you've never used just "
            "returns 0 memories.\n\n"
            "Pass an ARRAY of keywords. Threshold is adaptive:\n"
            "  - 1 keyword passed  → memories that have that keyword (broad sweep)\n"
            "  - 2+ keywords       → memories that match AT LEAST 2 of them\n"
            "                        (narrow query, ranked by overlap)\n"
            "\n"
            "Each result reports `matched N/M` so you can rank: a memory that "
            "matched 3 of your 4 queried keywords is more relevant than one "
            "that matched only 2. Use this to widen or narrow your search "
            "without an extra round-trip — start with 3-4 related keywords, "
            "see which memories score highest, then rag_load the 1-4 best.\n"
            "\n"
            "Each result line shows the title, keywords, content size, and a "
            "`modified:` date so you can tell fresh memories from stale ones. "
            "Memories whose title starts with `fix-easyai-` are tagged "
            "[FIXED] — they're immutable: you can recall them but cannot "
            "overwrite or forget them.\n"
            "\n"
            "Pagination: the response always includes machine-readable header "
            "lines `total_entries: T`, `page: P of N`, `has_more: true|false`. "
            "When `has_more: true` you can issue the SAME query with `page=P+1` "
            "to walk the rest of the result set. Don't paginate unless you "
            "actually need more — the first page is already ranked best-first, "
            "so the top memories are usually enough.\n"
            "\n"
            "Returns up to `max_results` memories per page (best-overlap first, "
            "ties broken by recency). If no memory matches the threshold, "
            "`total_entries: 0` is returned (not an error) — try fewer keywords, "
            "or use rag_list to browse."
        )
        .param("keywords",    "array",
               "1..8 keywords to search for. Each: 1..32 chars, [A-Za-z0-9._+-]. "
               "Example: [\"user-prefs\", \"hardware\"]. With one keyword the "
               "search is broad (any entry with it); with 2+, the search "
               "narrows (entries matching ≥2 of them, ranked by overlap).",
               true)
        .param("max_results", "integer",
               "Maximum entries to return PER PAGE (default 10, max 20). "
               "This is the page size; total result count comes back in "
               "`total_entries`.", false)
        .param("page",        "integer",
               "1-based page index to fetch (default 1). Use to walk a large "
               "result set when the previous response had `has_more: true`.",
               false)
        .handle(make_search_handler(store))
        .build();

    out.load = Tool::builder("rag_load")
        .describe(
            "Recall up to 4 memories by exact title and return their FULL "
            "content. Use this after rag_search to read the bodies of memories "
            "that looked promising in the preview. Pass exact titles from a "
            "previous rag_search result.\n\n"
            "Each recalled memory comes back with its keywords, `modified` "
            "date (human-readable + unix), and a `fixed:` line indicating "
            "whether the memory is immutable (yes when the title starts with "
            "`fix-easyai-`).\n\n"
            "Cap is 4 per call. If you need more than 4, you're probably "
            "trying to drown the prompt in stale content; narrow your search "
            "first."
        )
        .param("titles", "array",
               "Array of 1..4 exact memory titles to recall. "
               "Example: [\"gustavo-ai-box\", \"easyai-build-recipe\"].", true)
        .handle(make_load_handler(store))
        .build();

    out.list = Tool::builder("rag_list")
        .describe(
            "Browse your memories — list titles, optionally filtered by title "
            "prefix. Use this when you're not sure what keyword to search by, "
            "or to confirm whether you remember a particular note. Returns "
            "title, keywords, content_bytes, and modified date (human-"
            "readable) — body NOT included (use rag_load for that). Memories "
            "whose title starts with `fix-easyai-` are tagged [FIXED] —- "
            "they're immutable. Untagged memories (operator-dropped notes "
            "with no keywords: header) show here but don't appear in "
            "rag_search.\n\n"
            "Tip: `prefix='fix-easyai-'` lists every fixed memory in one "
            "call, useful when you want to see all the ground-truth rules "
            "the user has had you learn."
        )
        .param("prefix", "string",
               "Optional prefix to filter titles by. Empty = list all. "
               "Same character set as title. Example: 'easyai-' or "
               "'fix-easyai-'.", false)
        .param("max",    "integer",
               "Maximum memories to return (default 50, max 200).", false)
        .handle(make_list_handler(store))
        .build();

    out.del = Tool::builder("rag_delete")
        .describe(
            "Forget a memory by exact title. Use this when a memory has "
            "become stale, wrong, or just irrelevant — keeping memory tidy "
            "makes future rag_search results sharper. Common reasons to forget:\n"
            "  - the user corrected something and the old memory is now wrong\n"
            "  - a project / preference / fact has changed\n"
            "  - you want to refine a memory: forget + rag_save replaces it\n"
            "    cleanly (rag_save also overwrites, but forget-first is\n"
            "    appropriate when the title or keywords need to change too)\n"
            "  - the memory was a one-off scratch note no longer needed\n"
            "\n"
            "FIXED MEMORIES CANNOT BE FORGOTTEN: any title starting with "
            "`fix-easyai-` is immutable — rag_delete will refuse the call "
            "and tell you why. The operator can remove the file from disk by "
            "hand if it really must go.\n"
            "\n"
            "Idempotent on non-fixed memories: forgetting a non-existent "
            "title is not an error. The forget is permanent — there's no "
            "trash. Be certain before calling."
        )
        .param("title", "string",
               "Exact title to forget. Letters, digits, dash, underscore. "
               "1..64 chars. Titles starting with `fix-easyai-` are "
               "immutable and will be rejected.", true)
        .handle(make_delete_handler(store))
        .build();

    out.keywords = Tool::builder("rag_keywords")
        .describe(
            "Vocabulary overview of your memory: list every distinct keyword "
            "you've used, with the number of memories that reference it. "
            "CALL THIS BEFORE rag_save when you're not sure which keywords "
            "to use, and BEFORE rag_search when you're not sure what's in "
            "your memory.\n"
            "\n"
            "Why it matters: over time, an agent that doesn't check its own "
            "vocabulary creates near-duplicates ('user-prefs' vs 'user_pref' "
            "vs 'preferences') and memory fragments — old memories become "
            "unreachable to new searches. Calling rag_keywords first lets "
            "you reuse the vocabulary you've already established, keeping "
            "your memory coherent.\n"
            "\n"
            "Output is sorted by frequency (most-used keyword first) so the "
            "first lines are the dimensions you've been investing in. The "
            "long tail at the bottom is one-off keywords — candidates for "
            "consolidation. Default cap is 200 keywords; set max=500 for the "
            "full picture, or min_count=2 to hide the long tail and focus on "
            "established vocabulary."
        )
        .param("min_count", "integer",
               "Hide keywords used by fewer than this many memories (default 1 = "
               "show all). Set min_count=2 to filter out one-offs and see only "
               "your established vocabulary.", false)
        .param("max",       "integer",
               "Maximum keywords to return (default 200, max 500). The list "
               "is sorted by frequency, so the cap drops the long tail "
               "first.", false)
        .handle(make_keywords_handler(store))
        .build();

    return out;
}

// ---------------------------------------------------------------------------
// Default layout: single-tool dispatcher
// ---------------------------------------------------------------------------
// Wraps the same six handlers behind one Tool that takes an `action`
// parameter. The handler closures (make_save_handler / make_search_handler
// / etc.) read every other parameter directly out of `arguments_json`,
// so the dispatcher doesn't need to translate anything — it just picks
// the right closure by `action` and forwards the original ToolCall.
//
// Schema is a kitchen sink (every parameter optional except `action`)
// because JSON Schema's discriminated-union shapes (oneOf with a
// discriminator) trip up smaller / quantised tool-callers far more
// often than a flat "everything optional" schema does. Validation
// stays runtime: each handler rejects calls missing its required
// fields with the same crisp messages the legacy seven-tool flow uses.
//
// The on-disk format and locking discipline are byte-identical to the
// seven-tool build — same RagStore, same load_index_locked() pre-warm,
// same shared/unique mutex split.
Tool make_unified_rag_tool(std::string root_dir) {
    auto store = build_rag_store(std::move(root_dir));

    // Capture each per-action handler once; the dispatcher closes
    // over the resulting std::function set. Same store is shared, so
    // index updates from `save` / `delete` are visible to subsequent
    // `search` / `load` / `list` / `keywords` calls inside the same
    // process.
    auto h_save     = make_save_handler    (store);
    auto h_append   = make_append_handler  (store);
    auto h_search   = make_search_handler  (store);
    auto h_load     = make_load_handler    (store);
    auto h_list     = make_list_handler    (store);
    auto h_delete   = make_delete_handler  (store);
    auto h_keywords = make_keywords_handler(store);

    return Tool::builder("rag")
        .describe(
            "Your memory, accessed through one tool. Pick an action; the "
            "parameters needed depend on which action you choose. "
            "Seven actions are supported:\n"
            "\n"
            "  action=\"save\"\n"
            "    Store a memory (creates new or overwrites existing). Required: "
            "    title, keywords (array, 1..8), content. Optional: fix (boolean "
            "    — when true, the title is auto-prepended with `fix-easyai-` "
            "    and the memory becomes immutable: future save / delete / "
            "    append on it are refused).\n"
            "\n"
            "  action=\"append\"\n"
            "    Add new content to the end of an EXISTING memory without losing "
            "    the previous body. Required: title (must already exist), "
            "    content. Optional: keywords (array — extra keywords to merge, "
            "    deduped against existing, total still capped at 8). The "
            "    library inserts a Markdown `---` rule before the appendix so "
            "    each addition is visually delimited on disk. Use this for "
            "    chronological logs and for growing what you already know "
            "    about the user. Refuses on fix-easyai-* (immutable) titles "
            "    and on titles that don't exist (use save to create).\n"
            "\n"
            "  action=\"search\"\n"
            "    Search memories by keyword(s). Required: keywords (array). "
            "    With one keyword you get any memory carrying it; with two "
            "    or more, only memories matching ≥2 of them, ranked by "
            "    overlap. Optional: max_results (default 10, max 20), "
            "    page (default 1; the response includes `total_entries`, "
            "    `page: P of N`, `has_more` for pagination).\n"
            "\n"
            "  action=\"load\"\n"
            "    Recall the FULL content of up to 4 memories by exact "
            "    title. Required: titles (array of 1..4 exact titles from "
            "    a previous search). Cap is 4 to keep the prompt slim — "
            "    narrow your search if you need more.\n"
            "\n"
            "  action=\"list\"\n"
            "    Browse memories without loading bodies. Optional: prefix "
            "    (filter titles; pass `fix-easyai-` to see every immutable "
            "    memory), max (default 50, max 200).\n"
            "\n"
            "  action=\"delete\"\n"
            "    Forget a memory. Required: title (exact). Memories whose "
            "    title starts with `fix-easyai-` are immutable and cannot "
            "    be forgotten through this tool.\n"
            "\n"
            "  action=\"keywords\"\n"
            "    Vocabulary overview: every distinct keyword you've used "
            "    with how many memories carry it. CALL BEFORE save / search "
            "    when you're not sure what vocabulary you've established. "
            "    Optional: min_count (default 1), max (default 200, max 500).\n"
            "\n"
            "Search and list responses include a human-readable `modified` "
            "date on every memory and tag immutable ones with [FIXED]. "
            "Load responses include the same plus a `fixed: yes/no` line.\n"
            "\n"
            "USE THIS AGGRESSIVELY: anything worth remembering across "
            "sessions — user preferences, project decisions, command "
            "recipes, error fixes, domain knowledge — should land here. "
            "Future conversations will rag(action=\"search\") and find "
            "what THIS conversation taught you."
        )
        .param("action",      "string",
               "Required. One of: \"save\", \"append\", \"search\", "
               "\"load\", \"list\", \"delete\", \"keywords\". Each action "
               "consumes a subset of the other parameters; see the tool "
               "description for the per-action requirements.", true)
        .param("title",       "string",
               "Used by save / append / delete. 1..64 chars [A-Za-z0-9._+-]. "
               "With fix=true on a save the prefix `fix-easyai-` is auto-"
               "added if not present.", false)
        .param("titles",      "array",
               "Used by load. Array of 1..4 exact memory titles to recall.",
               false)
        .param("keywords",    "array",
               "Used by save / append / search. 1..8 short keywords "
               "[A-Za-z0-9._+-]. On save: tags this memory; on append: "
               "extra keywords merged into the existing list (deduped, "
               "total still capped at 8); on search: the query.",
               false)
        .param("content",     "string",
               "Used by save / append. On save: the full memory body. On "
               "append: text added after the existing body (separated by "
               "a Markdown `---` rule). Free-form UTF-8; total size on "
               "disk capped at 256 KB.",
               false)
        .param("fix",         "boolean",
               "Used by save. When true, store as IMMUTABLE — title is "
               "prefixed with `fix-easyai-` and future save / append / "
               "delete on this memory are refused. Use only when the user "
               "explicitly asks to learn something as a fixed rule / "
               "system design / ground-truth definition. Default false.",
               false)
        .param("prefix",      "string",
               "Used by list. Filter titles starting with this prefix. "
               "Pass `fix-easyai-` to see every immutable memory.", false)
        .param("max",         "integer",
               "Used by list / keywords. Result cap. List default 50 "
               "(max 200); keywords default 200 (max 500).", false)
        .param("max_results", "integer",
               "Used by search. Page size (default 10, max 20).", false)
        .param("page",        "integer",
               "Used by search. 1-based page index (default 1). Use when "
               "a previous search returned `has_more: true`.", false)
        .param("min_count",   "integer",
               "Used by keywords. Hide keywords used by fewer than this "
               "many memories (default 1 = show all). Set min_count=2 to "
               "filter out one-offs.", false)
        .handle([h_save, h_append, h_search, h_load, h_list, h_delete, h_keywords]
                (const ToolCall & c) -> ToolResult {
            std::string action;
            if (!args::get_string(c.arguments_json, "action", action)
                    || action.empty()) {
                return ToolResult::error(
                    "missing required argument: action. Use one of "
                    "\"save\", \"append\", \"search\", \"load\", "
                    "\"list\", \"delete\", \"keywords\".");
            }
            // Each branch forwards the original ToolCall — the per-
            // action handlers parse their own params out of
            // arguments_json with the same helpers (args::get_string,
            // parse_string_array, etc.) the legacy seven-tool flow uses,
            // so error messages and validation stay byte-identical.
            ToolResult r;
            if      (action == "save")     r = h_save(c);
            else if (action == "append")   r = h_append(c);
            else if (action == "search")   r = h_search(c);
            else if (action == "load")     r = h_load(c);
            else if (action == "list")     r = h_list(c);
            else if (action == "delete")   r = h_delete(c);
            else if (action == "keywords") r = h_keywords(c);
            else {
                return ToolResult::error(
                    "unknown action \"" + action + "\". Valid: \"save\", "
                    "\"append\", \"search\", \"load\", \"list\", "
                    "\"delete\", \"keywords\".");
            }

            // The inner handlers still cite the seven-tool names
            // (rag_save, rag_append, rag_search, ...) in their
            // guidance prose. In unified mode the model has only
            // `rag` in its catalog, so any literal `rag_<verb>`
            // reference would be a dangling identifier. Rewrite
            // each occurrence in place to the dispatch form the
            // model can actually call. Cheap (O(n) over a small
            // message); the set of substitutions is closed and
            // stable.
            struct Sub { const char * from; const char * to; };
            static const Sub kSubs[] = {
                // Order matters: rag_append must come before rag_a... siblings
                // would, but only rag_save shares the leading 'rag_' so any
                // order works. Keep alphabetical for grep-ability.
                { "rag_append",   "rag(action=\"append\")"   },
                { "rag_delete",   "rag(action=\"delete\")"   },
                { "rag_keywords", "rag(action=\"keywords\")" },
                { "rag_list",     "rag(action=\"list\")"     },
                { "rag_load",     "rag(action=\"load\")"     },
                { "rag_save",     "rag(action=\"save\")"     },
                { "rag_search",   "rag(action=\"search\")"   },
            };
            for (const auto & s : kSubs) {
                std::string from = s.from;
                std::string to   = s.to;
                size_t pos = 0;
                while ((pos = r.content.find(from, pos)) != std::string::npos) {
                    r.content.replace(pos, from.size(), to);
                    pos += to.size();
                }
            }
            return r;
        })
        .build();
}

}  // namespace easyai::tools
