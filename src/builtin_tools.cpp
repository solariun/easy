#include "easyai/builtin_tools.hpp"
#include "easyai/tool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#if defined(EASYAI_HAVE_CURL)
#include <curl/curl.h>
#endif

namespace easyai::tools {

namespace fs = std::filesystem;

// ---------------------------------------------------------------- helpers
static std::string trim(std::string s) {
    auto issp = [](unsigned char c){ return std::isspace(c); };
    while (!s.empty() && issp(s.front())) s.erase(s.begin());
    while (!s.empty() && issp(s.back()))  s.pop_back();
    return s;
}

static std::string strip_html(const std::string & html) {
    // Drop <script>/<style> blocks, then all remaining tags. Collapse ws.
    static const std::regex re_block(R"(<(script|style)[^>]*>[\s\S]*?</\1>)",
                                     std::regex::icase);
    static const std::regex re_tag  ("<[^>]+>");
    static const std::regex re_ws   (R"([ \t\r\n]+)");
    std::string s = std::regex_replace(html, re_block, " ");
    s = std::regex_replace(s, re_tag, " ");
    // basic entity unescape
    struct E { const char * from; const char * to; };
    static const E ents[] = {
        {"&nbsp;", " "}, {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"},
        {"&quot;", "\""}, {"&#39;", "'"}, {"&apos;", "'"},
    };
    for (const auto & e : ents) {
        size_t pos = 0;
        while ((pos = s.find(e.from, pos)) != std::string::npos) {
            s.replace(pos, std::strlen(e.from), e.to);
            pos += std::strlen(e.to);
        }
    }
    s = std::regex_replace(s, re_ws, " ");
    return trim(s);
}

static std::string clip(std::string s, size_t n) {
    if (s.size() > n) {
        s.resize(n);
        s += "\n…[truncated]";
    }
    return s;
}

static std::string url_encode(const std::string & v) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(v.size() * 3);
    for (unsigned char c : v) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 0xF];
        }
    }
    return out;
}

#if defined(EASYAI_HAVE_CURL)
static size_t curl_write_cb(void * buf, size_t sz, size_t n, void * ud) {
    auto * out = static_cast<std::string *>(ud);
    out->append(static_cast<char *>(buf), sz * n);
    return sz * n;
}

static bool http_get(const std::string & url,
                     const std::vector<std::string> & extra_headers,
                     std::string & body, std::string & err,
                     long timeout_s = 20, long max_bytes = 2 * 1024 * 1024) {
    CURL * c = curl_easy_init();
    if (!c) { err = "curl_easy_init failed"; return false; }

    body.clear();
    curl_easy_setopt(c, CURLOPT_URL, url.c_str());
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, timeout_s);
    curl_easy_setopt(c, CURLOPT_USERAGENT, "easyai/0.1 (+https://github.com/)");
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_ACCEPT_ENCODING, "");

    curl_slist * headers = nullptr;
    for (const auto & h : extra_headers) headers = curl_slist_append(headers, h.c_str());
    if (headers) curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers);

    CURLcode rc = curl_easy_perform(c);
    long http_code = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &http_code);
    if (headers) curl_slist_free_all(headers);
    curl_easy_cleanup(c);

    if (rc != CURLE_OK) {
        err = curl_easy_strerror(rc);
        return false;
    }
    if (http_code >= 400) {
        err = "HTTP " + std::to_string(http_code);
        return false;
    }
    if ((long) body.size() > max_bytes) body.resize(max_bytes);
    return true;
}
#endif

// ============================================================================
// datetime
// ============================================================================
Tool datetime() {
    return Tool::builder("datetime")
        .describe("Returns the current date and time in UTC and local time. "
                  "Useful when the user asks 'what time is it' or for date "
                  "arithmetic.")
        .handle([](const ToolCall &) {
            using namespace std::chrono;
            auto now = system_clock::now();
            std::time_t t = system_clock::to_time_t(now);
            std::tm utc{}, loc{};
#if defined(_WIN32)
            gmtime_s(&utc, &t);
            localtime_s(&loc, &t);
#else
            gmtime_r(&t, &utc);
            localtime_r(&t, &loc);
#endif
            char b1[64], b2[64];
            std::strftime(b1, sizeof(b1), "%Y-%m-%dT%H:%M:%SZ",      &utc);
            std::strftime(b2, sizeof(b2), "%Y-%m-%dT%H:%M:%S%z",     &loc);
            std::ostringstream o;
            o << "{\"utc\":\"" << b1 << "\",\"local\":\"" << b2 << "\"}";
            return ToolResult::ok(o.str());
        })
        .build();
}

// ============================================================================
// web_fetch
// ============================================================================
Tool web_fetch() {
    return Tool::builder("web_fetch")
        .describe("Fetch a URL and return its text content (HTML stripped). "
                  "Use for reading web pages.")
        .param("url",     "string", "Fully-qualified http(s) URL to fetch", true)
        .param("as_html", "boolean","If true, return raw HTML instead of stripped text", false)
        .handle([](const ToolCall & c) {
#if !defined(EASYAI_HAVE_CURL)
            (void) c;
            return ToolResult::error("web_fetch unavailable: easyai built without libcurl");
#else
            std::string url; bool as_html = false;
            if (!args::get_string(c.arguments_json, "url", url) || url.empty()) {
                return ToolResult::error("missing required arg: url");
            }
            args::get_bool(c.arguments_json, "as_html", as_html);

            std::string body, err;
            if (!http_get(url, {}, body, err)) {
                return ToolResult::error("fetch failed: " + err);
            }
            return ToolResult::ok(clip(as_html ? body : strip_html(body), 16 * 1024));
#endif
        })
        .build();
}

// ============================================================================
// web_search  (SearXNG JSON API)
// ============================================================================
//
// We hit `${SEARXNG_URL}/search?q=...&format=json`. SearXNG must be configured
// to allow the `json` format (search.formats in settings.yml).
//
Tool web_search() {
    return Tool::builder("web_search")
        .describe("Search the web via a SearXNG instance. Returns a list of "
                  "title / url / snippet results.")
        .param("query",      "string",  "Search query", true)
        .param("max_results","integer", "Maximum results to return (default 5)", false)
        .handle([](const ToolCall & c) {
#if !defined(EASYAI_HAVE_CURL)
            (void) c;
            return ToolResult::error("web_search unavailable: easyai built without libcurl");
#else
            std::string query;
            if (!args::get_string(c.arguments_json, "query", query) || query.empty()) {
                return ToolResult::error("missing required arg: query");
            }
            long long max_results = 5;
            args::get_int(c.arguments_json, "max_results", max_results);
            if (max_results < 1)  max_results = 1;
            if (max_results > 20) max_results = 20;

            const char * env = std::getenv("EASYAI_SEARXNG_URL");
            std::string base = env ? env : "http://127.0.0.1:8080";
            while (!base.empty() && base.back() == '/') base.pop_back();

            std::string url = base + "/search?q=" + url_encode(query)
                            + "&format=json&safesearch=1&language=en";
            std::string body, err;
            if (!http_get(url, {"Accept: application/json"}, body, err)) {
                return ToolResult::error("search failed (" + base + "): " + err);
            }

            // Lightweight extraction of the top-N {title,url,content} entries
            // from the SearXNG JSON. Avoids pulling a JSON dep.
            std::ostringstream out;
            out << "Top results for: " << query << "\n";
            size_t pos = body.find("\"results\"");
            if (pos == std::string::npos) {
                return ToolResult::error("no results array in SearXNG response");
            }
            int    count = 0;
            size_t i = body.find('[', pos);
            if (i == std::string::npos) {
                return ToolResult::error("malformed SearXNG response");
            }

            auto pull = [](const std::string & s, size_t from, const std::string & key,
                           std::string & out_val) -> bool {
                std::string needle = "\"" + key + "\":";
                size_t k = s.find(needle, from);
                if (k == std::string::npos) return false;
                k += needle.size();
                while (k < s.size() && std::isspace((unsigned char) s[k])) ++k;
                if (k >= s.size() || s[k] != '"') return false;
                ++k;
                out_val.clear();
                while (k < s.size() && s[k] != '"') {
                    if (s[k] == '\\' && k + 1 < s.size()) { out_val += s[k+1]; k += 2; }
                    else                                  { out_val += s[k];   ++k;  }
                }
                return true;
            };

            while (count < max_results) {
                size_t obj = body.find('{', i + 1);
                size_t end = body.find('}', i + 1);
                if (obj == std::string::npos || end == std::string::npos) break;
                if (obj > end) break;
                std::string title, urlv, content;
                pull(body, obj, "title",   title);
                pull(body, obj, "url",     urlv);
                pull(body, obj, "content", content);
                if (urlv.empty()) break;
                out << "\n" << (count + 1) << ". " << title << "\n   " << urlv
                    << "\n   " << clip(content, 300) << "\n";
                ++count;
                i = end;
            }
            if (count == 0) return ToolResult::ok("No results.");
            return ToolResult::ok(clip(out.str(), 8 * 1024));
#endif
        })
        .build();
}

// ============================================================================
// filesystem tools — sandboxed under `root`
// ============================================================================
namespace {

struct Sandbox {
    fs::path root;
    explicit Sandbox(std::string r) {
        if (r.empty()) r = ".";
        root = fs::weakly_canonical(fs::absolute(r));
    }
    // Resolve a user-supplied path inside the sandbox; returns false if it
    // escapes the root.
    bool resolve(const std::string & in, fs::path & out, std::string & err) const {
        fs::path p = in;
        if (!p.is_absolute()) p = root / p;
        std::error_code ec;
        fs::path canon = fs::weakly_canonical(p, ec);
        if (canon.empty()) canon = p.lexically_normal();

        auto rs = root.string();
        auto cs = canon.string();
        if (cs.compare(0, rs.size(), rs) != 0) {
            err = "path escapes sandbox root: " + cs;
            return false;
        }
        out = canon;
        return true;
    }
};

}  // namespace

Tool fs_read_file(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("read_file")
        .describe("Read a UTF-8 text file from disk and return its contents.")
        .param("path",   "string",  "Path to the file (relative or absolute, inside the sandbox)", true)
        .param("offset", "integer", "Skip this many bytes from the start", false)
        .param("limit",  "integer", "Maximum bytes to return (default 64KB)", false)
        .handle([sb](const ToolCall & c) {
            std::string path; long long offset = 0, limit = 64 * 1024;
            if (!args::get_string(c.arguments_json, "path", path))
                return ToolResult::error("missing arg: path");
            args::get_int(c.arguments_json, "offset", offset);
            args::get_int(c.arguments_json, "limit",  limit);
            if (offset < 0) offset = 0;
            if (limit  < 1) limit  = 1;
            if (limit > 1024 * 1024) limit = 1024 * 1024;

            fs::path p; std::string err;
            if (!sb->resolve(path, p, err)) return ToolResult::error(err);
            std::ifstream f(p, std::ios::binary);
            if (!f) return ToolResult::error("cannot open: " + p.string());
            f.seekg(offset);
            std::string buf((size_t) limit, '\0');
            f.read(buf.data(), limit);
            buf.resize(f.gcount());
            return ToolResult::ok(std::move(buf));
        })
        .build();
}

Tool fs_write_file(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("write_file")
        .describe("Write text to a file (overwrites). Creates parent directories.")
        .param("path",    "string",  "Destination path (inside the sandbox)", true)
        .param("content", "string",  "UTF-8 text content to write", true)
        .param("append",  "boolean", "If true, append instead of overwriting", false)
        .handle([sb](const ToolCall & c) {
            std::string path, content; bool append = false;
            if (!args::get_string(c.arguments_json, "path",    path))
                return ToolResult::error("missing arg: path");
            if (!args::get_string(c.arguments_json, "content", content))
                return ToolResult::error("missing arg: content");
            args::get_bool(c.arguments_json, "append", append);

            fs::path p; std::string err;
            if (!sb->resolve(path, p, err)) return ToolResult::error(err);

            std::error_code ec;
            fs::create_directories(p.parent_path(), ec);
            std::ofstream f(p, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
            if (!f) return ToolResult::error("cannot open for write: " + p.string());
            f.write(content.data(), content.size());
            return ToolResult::ok("wrote " + std::to_string(content.size())
                                  + " bytes to " + p.string());
        })
        .build();
}

Tool fs_list_dir(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("list_dir")
        .describe("List the entries (files and directories) inside a directory.")
        .param("path", "string", "Directory path (inside the sandbox)", true)
        .handle([sb](const ToolCall & c) {
            std::string path;
            if (!args::get_string(c.arguments_json, "path", path))
                return ToolResult::error("missing arg: path");

            fs::path p; std::string err;
            if (!sb->resolve(path, p, err)) return ToolResult::error(err);
            if (!fs::is_directory(p))
                return ToolResult::error("not a directory: " + p.string());

            std::ostringstream o;
            for (auto & e : fs::directory_iterator(p)) {
                o << (e.is_directory() ? "d " : "f ") << e.path().filename().string();
                if (e.is_regular_file()) {
                    std::error_code ec;
                    auto sz = fs::file_size(e.path(), ec);
                    if (!ec) o << "  (" << sz << " B)";
                }
                o << "\n";
            }
            return ToolResult::ok(clip(o.str(), 16 * 1024));
        })
        .build();
}

Tool fs_glob(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("glob")
        .describe("Find files whose names match a wildcard pattern (e.g. '*.cpp', "
                  "'src/**/*.h'). Recursive by default.")
        .param("pattern", "string", "Wildcard pattern (* matches any chars except /; ** matches across dirs)", true)
        .param("path",    "string", "Optional starting directory inside the sandbox (default = root)", false)
        .handle([sb](const ToolCall & c) {
            std::string pattern, sub;
            if (!args::get_string(c.arguments_json, "pattern", pattern))
                return ToolResult::error("missing arg: pattern");
            args::get_string(c.arguments_json, "path", sub);

            fs::path start = sb->root; std::string err;
            if (!sub.empty() && !sb->resolve(sub, start, err))
                return ToolResult::error(err);

            // wildcard -> regex
            std::string re = "^";
            for (size_t i = 0; i < pattern.size(); ++i) {
                char ch = pattern[i];
                if (ch == '*') {
                    if (i + 1 < pattern.size() && pattern[i + 1] == '*') {
                        re += ".*"; ++i;
                    } else {
                        re += "[^/]*";
                    }
                } else if (ch == '?') {
                    re += "[^/]";
                } else if (std::strchr(".+()|^$\\{}[]", ch)) {
                    re += '\\'; re += ch;
                } else {
                    re += ch;
                }
            }
            re += "$";
            std::regex rx(re);

            std::ostringstream o;
            int n = 0;
            for (auto & e : fs::recursive_directory_iterator(start)) {
                if (!e.is_regular_file()) continue;
                std::string rel = fs::relative(e.path(), sb->root).generic_string();
                if (std::regex_match(rel, rx)) {
                    o << rel << "\n";
                    if (++n >= 500) { o << "...[stopped at 500 matches]\n"; break; }
                }
            }
            if (n == 0) return ToolResult::ok("No matches.");
            return ToolResult::ok(o.str());
        })
        .build();
}

Tool fs_grep(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("grep")
        .describe("Search file contents for a regular expression. Returns matching lines with file:line prefixes.")
        .param("pattern",       "string",  "Regular expression to search for", true)
        .param("path",          "string",  "Optional starting directory (default = sandbox root)", false)
        .param("file_glob",     "string",  "Optional filename glob to limit search (e.g. '*.cpp')", false)
        .param("max_matches",   "integer", "Stop after this many matches (default 100)", false)
        .param("case_insensitive", "boolean", "Case-insensitive match (default false)", false)
        .handle([sb](const ToolCall & c) {
            std::string pattern, sub, file_glob;
            long long max_matches = 100;
            bool ci = false;
            if (!args::get_string(c.arguments_json, "pattern", pattern))
                return ToolResult::error("missing arg: pattern");
            args::get_string(c.arguments_json, "path",      sub);
            args::get_string(c.arguments_json, "file_glob", file_glob);
            args::get_int   (c.arguments_json, "max_matches", max_matches);
            args::get_bool  (c.arguments_json, "case_insensitive", ci);

            fs::path start = sb->root; std::string err;
            if (!sub.empty() && !sb->resolve(sub, start, err))
                return ToolResult::error(err);

            std::regex::flag_type rf = std::regex::ECMAScript;
            if (ci) rf |= std::regex::icase;
            std::regex rx;
            try { rx = std::regex(pattern, rf); }
            catch (const std::regex_error & e) {
                return ToolResult::error(std::string("bad regex: ") + e.what());
            }

            std::regex glob_rx(".*");
            if (!file_glob.empty()) {
                std::string r = "^";
                for (char ch : file_glob) {
                    if      (ch == '*') r += "[^/]*";
                    else if (ch == '?') r += "[^/]";
                    else if (std::strchr(".+()|^$\\{}[]", ch)) { r += '\\'; r += ch; }
                    else r += ch;
                }
                r += "$";
                glob_rx = std::regex(r);
            }

            std::ostringstream o;
            int n = 0;
            for (auto & e : fs::recursive_directory_iterator(start)) {
                if (!e.is_regular_file()) continue;
                if (e.file_size() > 4 * 1024 * 1024) continue;  // skip huge files
                std::string fname = e.path().filename().string();
                if (!file_glob.empty() && !std::regex_match(fname, glob_rx)) continue;
                std::ifstream f(e.path());
                if (!f) continue;
                std::string line; int lineno = 0;
                std::string rel = fs::relative(e.path(), sb->root).generic_string();
                while (std::getline(f, line)) {
                    ++lineno;
                    if (std::regex_search(line, rx)) {
                        o << rel << ":" << lineno << ": " << clip(line, 240) << "\n";
                        if (++n >= max_matches) goto done;
                    }
                }
            }
        done:
            if (n == 0) return ToolResult::ok("No matches.");
            return ToolResult::ok(clip(o.str(), 32 * 1024));
        })
        .build();
}

}  // namespace easyai::tools
