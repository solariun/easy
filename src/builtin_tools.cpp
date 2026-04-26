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

// Strip HTML to plain text without ever touching std::regex.  The
// libstdc++ regex implementation is recursive on backtracking patterns
// like   <(script|style)[^>]*>[\s\S]*?</\1>   and crashes the process
// (SIGSEGV from stack overflow) on adversarial / oversized HTML — which
// is exactly what web_fetch sees when an LLM picks a beefy news page.
// Forward-only scanning + char-by-char whitespace collapse: O(n), zero
// recursion, no stack risk regardless of input.
static std::string strip_html(const std::string & html) {
    auto ieq = [](char a, char b) {
        return std::tolower((unsigned char) a) == std::tolower((unsigned char) b);
    };
    auto starts_with_ci = [&](size_t i, const char * needle) {
        size_t n = std::strlen(needle);
        if (i + n > html.size()) return false;
        for (size_t k = 0; k < n; ++k) {
            if (!ieq(html[i + k], needle[k])) return false;
        }
        return true;
    };

    std::string out;
    out.reserve(html.size());

    size_t i = 0;
    while (i < html.size()) {
        char c = html[i];

        // <script ...>...</script>  and  <style ...>...</style>
        // Skip the entire block, including any tag-internal noise, by
        // searching forward for the closing tag.  If we never find it
        // (truncated HTML), drop the rest.
        if (c == '<' && i + 1 < html.size()) {
            const char * tag = nullptr;
            const char * end = nullptr;
            if (starts_with_ci(i + 1, "script")
                    && (i + 7 >= html.size() ||
                        !std::isalnum((unsigned char) html[i + 7]))) {
                tag = "<script"; end = "</script>";
            } else if (starts_with_ci(i + 1, "style")
                    && (i + 6 >= html.size() ||
                        !std::isalnum((unsigned char) html[i + 6]))) {
                tag = "<style"; end = "</style>";
            }
            if (tag) {
                size_t after_open = i + 1;
                // skip until the '>' of the opening tag
                while (after_open < html.size() && html[after_open] != '>') {
                    ++after_open;
                }
                if (after_open >= html.size()) break;
                // search closing tag (case-insensitive)
                size_t end_len = std::strlen(end);
                size_t j = after_open + 1;
                bool found = false;
                while (j + end_len <= html.size()) {
                    if (html[j] == '<' && starts_with_ci(j, end)) {
                        i = j + end_len;
                        found = true;
                        break;
                    }
                    ++j;
                }
                if (!found) break;        // unterminated → drop the rest
                out += ' ';
                continue;
            }
            // Generic tag: skip to next '>'.  Tags don't nest.
            size_t j = i + 1;
            while (j < html.size() && html[j] != '>') ++j;
            i = (j < html.size()) ? j + 1 : html.size();
            out += ' ';
            continue;
        }

        // Entity decode — small fixed table.
        if (c == '&') {
            struct E { const char * from; const char * to; };
            static const E ents[] = {
                {"&nbsp;", " "}, {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"},
                {"&quot;", "\""}, {"&#39;", "'"}, {"&apos;", "'"},
            };
            bool matched = false;
            for (const auto & e : ents) {
                size_t flen = std::strlen(e.from);
                if (i + flen <= html.size()
                        && html.compare(i, flen, e.from) == 0) {
                    out += e.to;
                    i += flen;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        // Whitespace collapse: turn any run of [ \t\r\n] into a single space.
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (!out.empty() && out.back() != ' ') out += ' ';
            ++i;
            continue;
        }

        out += c;
        ++i;
    }

    return trim(out);
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

// Decodes a percent-encoded form-urlencoded string ('+' → space, '%XX' → byte).
// Tolerant of malformed input — leaves bad sequences as-is rather than failing.
static std::string url_decode(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '+') {
            out += ' ';
        } else if (c == '%' && i + 2 < s.size() &&
                   std::isxdigit((unsigned char) s[i + 1]) &&
                   std::isxdigit((unsigned char) s[i + 2])) {
            char hex[3] = { s[i + 1], s[i + 2], 0 };
            out += static_cast<char>(std::strtol(hex, nullptr, 16));
            i += 2;
        } else {
            out += c;
        }
    }
    return out;
}

// DuckDuckGo wraps result URLs in `/l/?uddg=ENCODED_URL[&rut=...]` redirects.
// If `href` carries that wrapper, extract and decode the uddg param.
// Falls back to returning the href unchanged (or with the protocol prefixed if
// it's protocol-relative) when there's no wrapper.
static std::string decode_ddg_redirect(const std::string & href) {
    static const std::string marker = "uddg=";
    size_t k = href.find(marker);
    if (k == std::string::npos) {
        if (href.compare(0, 4, "http") == 0) return href;
        if (href.compare(0, 2, "//")  == 0)  return "https:" + href;
        return href;
    }
    k += marker.size();
    size_t end = href.find('&', k);
    std::string enc = href.substr(k, end == std::string::npos
                                       ? std::string::npos
                                       : end - k);
    return url_decode(enc);
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

// POST application/x-www-form-urlencoded.  Used by the DDG search backend
// because POST is far less likely than GET to hit the bot/captcha gate.
static bool http_post_form(const std::string & url,
                           const std::string & form_body,
                           std::string & body, std::string & err,
                           long timeout_s = 20,
                           long max_bytes = 4 * 1024 * 1024) {
    CURL * c = curl_easy_init();
    if (!c) { err = "curl_easy_init failed"; return false; }

    body.clear();
    curl_easy_setopt(c, CURLOPT_URL,             url.c_str());
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION,  1L);
    curl_easy_setopt(c, CURLOPT_MAXREDIRS,       5L);
    curl_easy_setopt(c, CURLOPT_TIMEOUT,         timeout_s);
    // A real-browser User-Agent is required by html.duckduckgo.com.
    curl_easy_setopt(c, CURLOPT_USERAGENT,
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/16.6 Safari/605.1.15");
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION,   curl_write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA,       &body);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL,        1L);
    curl_easy_setopt(c, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(c, CURLOPT_POST,            1L);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS,      form_body.c_str());
    curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE,   (long) form_body.size());

    curl_slist * headers = nullptr;
    headers = curl_slist_append(headers,
        "Content-Type: application/x-www-form-urlencoded");
    headers = curl_slist_append(headers,
        "Accept: text/html,application/xhtml+xml");
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers);

    CURLcode rc = curl_easy_perform(c);
    long http_code = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(c);

    if (rc != CURLE_OK)   { err = curl_easy_strerror(rc); return false; }
    if (http_code >= 400) { err = "HTTP " + std::to_string(http_code); return false; }
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
        .describe("Fetch a URL and return its text content (HTML stripped, "
                  "trimmed to ~16 KB). This is the ONLY way to read a web "
                  "page's actual content — web_search returns titles and "
                  "short snippets, not the page body. Always call this after "
                  "web_search when the user wants the contents of an article, "
                  "documentation page, or news story.")
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
// web_search — DuckDuckGo HTML scraper
// ----------------------------------------------------------------------------
// No external service required: we POST the query to html.duckduckgo.com and
// parse the resulting HTML for the result list.  DDG's HTML page has a stable
// class-based markup (`result__a`, `result__snippet`) that's been the same for
// years; if/when that changes, only this function needs updating.
//
// We POST instead of GET because DDG is much more aggressive about throwing
// CAPTCHAs at GET requests from scripted clients.
//
// The URLs returned by DDG are wrapped in a tracking redirect of the form
// `//duckduckgo.com/l/?uddg=ENCODED_URL&rut=...`; decode_ddg_redirect() unwraps
// them so the caller (the LLM) gets real, fetchable URLs.
//
// No global state, no env vars — works out of the box.
// ============================================================================
Tool web_search() {
    return Tool::builder("web_search")
        .describe("Search the web (DuckDuckGo). Returns a numbered list of "
                  "title / url / snippet results. The snippets are 1-2 short "
                  "sentences only — NEVER summarize a topic from them alone. "
                  "After this call, you MUST call web_fetch on the top 1-3 "
                  "most relevant URLs from the result list to read the actual "
                  "page content, then base your answer on the fetched text.")
        .param("query",       "string",  "Search query", true)
        .param("max_results", "integer", "Maximum results to return "
                                         "(default 5, max 20)", false)
        .handle([](const ToolCall & c) {
#if !defined(EASYAI_HAVE_CURL)
            (void) c;
            return ToolResult::error(
                "web_search unavailable: easyai built without libcurl");
#else
            std::string query;
            if (!args::get_string(c.arguments_json, "query", query) || query.empty()) {
                return ToolResult::error("missing required arg: query");
            }
            long long max_results = 5;
            args::get_int(c.arguments_json, "max_results", max_results);
            if (max_results < 1)  max_results = 1;
            if (max_results > 20) max_results = 20;

            // POST html.duckduckgo.com/html/  with a form-encoded query.
            // kl=us-en pins the locale so results are reproducible.
            const std::string url       = "https://html.duckduckgo.com/html/";
            const std::string post_body = "q=" + url_encode(query) + "&kl=us-en";
            std::string body, err;
            if (!http_post_form(url, post_body, body, err)) {
                return ToolResult::error("search failed: " + err);
            }
            if (body.empty()) {
                return ToolResult::error("search failed: empty response");
            }

            // ----- parse HTML --------------------------------------------------
            // Each result block contains:
            //   <a class="result__a" href="REDIRECT">TITLE_HTML</a>
            //   ...
            //   <a class="result__snippet" ...>SNIPPET_HTML</a>     (most cases)
            //     OR
            //   <div class="result__snippet">SNIPPET_HTML</div>     (some)
            //
            // We iterate over title matches and look for the *next* snippet
            // element after each title's match end.
            //
            // The regexes use [\s\S] for "any char incl. newline" because
            // std::regex's "." doesn't match newlines by default.
            static const std::regex re_title(
                R"DDG(<a[^>]*class\s*=\s*"[^"]*result__a[^"]*"[^>]*href\s*=\s*"([^"]+)"[^>]*>([\s\S]*?)</a>)DDG",
                std::regex::icase);
            static const std::regex re_snippet(
                R"DDG(<(?:a|div)[^>]*class\s*=\s*"[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)</(?:a|div)>)DDG",
                std::regex::icase);

            std::ostringstream out;
            out << "Top web results for: " << query << "\n";
            int count = 0;

            auto t_end = std::sregex_iterator();
            for (auto it = std::sregex_iterator(body.begin(), body.end(), re_title);
                 it != t_end && count < max_results; ++it) {

                std::string href  = (*it)[1].str();
                std::string title = strip_html((*it)[2].str());
                std::string real  = decode_ddg_redirect(href);
                if (real.empty() || title.empty()) continue;

                // Look for a snippet inside the next ~2 KiB of HTML after this
                // title (snippets sit immediately below their titles in the DOM).
                size_t after = it->position(0) + it->length(0);
                std::string tail = body.substr(after,
                    std::min<size_t>(2048, body.size() - after));
                std::string snippet;
                std::smatch sm;
                if (std::regex_search(tail, sm, re_snippet)) {
                    snippet = strip_html(sm[1].str());
                }

                out << "\n" << (count + 1) << ". " << title
                    << "\n   " << real
                    << "\n   " << clip(snippet, 300) << "\n";
                ++count;
            }

            if (count == 0) {
                // DDG occasionally serves a CAPTCHA / "anomaly" interstitial
                // when it thinks we're a bot — surface that explicitly.
                return ToolResult::error(
                    "no results parsed (DuckDuckGo may have rate-limited the "
                    "request; try again in a minute)");
            }
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
