#include "easyai/builtin_tools.hpp"
#include "easyai/tool.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>     // PATH_MAX
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <mutex>
#include <unordered_map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

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

// Capped write callback — bails out the moment the body exceeds the
// configured max.  We can't trust the server's Content-Length (or
// chunked-transfer absence of one), so policing in the callback is the
// only way to keep RAM bounded against an adversarial endpoint.
struct HttpSink {
    std::string * body;
    size_t        max_bytes;
    bool          truncated = false;
};

static size_t curl_write_cb(void * buf, size_t sz, size_t n, void * ud) {
    auto * sink = static_cast<HttpSink *>(ud);
    const size_t incoming = sz * n;
    const size_t already  = sink->body->size();
    if (already >= sink->max_bytes) {
        // Already at the cap — pretend we accepted to keep curl happy
        // (returning 0 here would force curl to abort with an error,
        // which we don't want — we just want to truncate).
        sink->truncated = true;
        return incoming;
    }
    const size_t room = sink->max_bytes - already;
    const size_t take = (incoming <= room) ? incoming : room;
    sink->body->append(static_cast<char *>(buf), take);
    if (take < incoming) sink->truncated = true;
    return incoming;
}

// Microsoft Edge on Windows 11 — current-stable User-Agent string.  Many
// sites (Cloudflare-fronted, news outlets, search engines) gate or
// degrade content for "easyai/0.1" and similar non-browser UAs; this
// pretends to be a vanilla Edge install so the bot heuristics treat us
// like an interactive user.  Bumped manually as Edge releases — kept
// in one place so web_search and web_fetch share the same persona.
static const char * const kEdgeUserAgent =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0";

// Companion headers Edge always sends.  Appended to whatever header list
// the caller is already building so we don't clobber Content-Type for
// the search POST.
static curl_slist * append_edge_browser_headers(curl_slist * headers) {
    headers = curl_slist_append(headers,
        "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7");
    headers = curl_slist_append(headers, "Accept-Language: en-US,en;q=0.9");
    headers = curl_slist_append(headers,
        "sec-ch-ua: \"Chromium\";v=\"130\", \"Microsoft Edge\";v=\"130\", "
        "\"Not?A_Brand\";v=\"99\"");
    headers = curl_slist_append(headers, "sec-ch-ua-mobile: ?0");
    headers = curl_slist_append(headers, "sec-ch-ua-platform: \"Windows\"");
    headers = curl_slist_append(headers, "Upgrade-Insecure-Requests: 1");
    headers = curl_slist_append(headers, "Sec-Fetch-Dest: document");
    headers = curl_slist_append(headers, "Sec-Fetch-Mode: navigate");
    headers = curl_slist_append(headers, "Sec-Fetch-Site: none");
    headers = curl_slist_append(headers, "Sec-Fetch-User: ?1");
    return headers;
}

// SECURITY: refuse non-http(s) URLs at the top of every fetch.  Without
// this gate, the model could ask for `file:///etc/passwd`, `gopher://...`,
// or `dict://internal`, all of which curl supports by default.  We also
// pin CURLOPT_PROTOCOLS to belt-and-braces the same restriction at the
// transport layer.
static bool url_is_safe_scheme(const std::string & url) {
    auto lower_starts_with = [&](const char * p) {
        size_t n = std::strlen(p);
        if (url.size() < n) return false;
        for (size_t i = 0; i < n; ++i) {
            char a = url[i], b = p[i];
            if (a >= 'A' && a <= 'Z') a = (char) (a + 32);
            if (a != b) return false;
        }
        return true;
    };
    return lower_starts_with("http://") || lower_starts_with("https://");
}

static bool http_get(const std::string & url,
                     const std::vector<std::string> & extra_headers,
                     std::string & body, std::string & err,
                     long timeout_s = 20, long max_bytes = 2 * 1024 * 1024) {
    if (!url_is_safe_scheme(url)) {
        err = "only http:// and https:// URLs are allowed";
        return false;
    }

    CURL * c = curl_easy_init();
    if (!c) { err = "curl_easy_init failed"; return false; }

    body.clear();
    HttpSink sink{ &body, (size_t) max_bytes, false };
    curl_easy_setopt(c, CURLOPT_URL, url.c_str());
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_MAXREDIRS, 5L);
    // CURLOPT_PROTOCOLS / _REDIR_PROTOCOLS deprecated in libcurl 7.85.0 in
    // favour of the *_STR variants. The new ones are enums (not macros), so
    // an `#ifdef CURLOPT_PROTOCOLS_STR` test silently falls through; gate on
    // the version macro instead.
#if defined(LIBCURL_VERSION_NUM) && LIBCURL_VERSION_NUM >= 0x075500
    curl_easy_setopt(c, CURLOPT_PROTOCOLS_STR,         "http,https");
    curl_easy_setopt(c, CURLOPT_REDIR_PROTOCOLS_STR,   "http,https");
#else
    curl_easy_setopt(c, CURLOPT_PROTOCOLS,
                     (long) (CURLPROTO_HTTP | CURLPROTO_HTTPS));
    curl_easy_setopt(c, CURLOPT_REDIR_PROTOCOLS,
                     (long) (CURLPROTO_HTTP | CURLPROTO_HTTPS));
#endif
    curl_easy_setopt(c, CURLOPT_TIMEOUT, timeout_s);
    // Impersonate Microsoft Edge — Cloudflare/news/etc gate generic UAs.
    curl_easy_setopt(c, CURLOPT_USERAGENT, kEdgeUserAgent);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &sink);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_ACCEPT_ENCODING, "");

    curl_slist * headers = nullptr;
    for (const auto & h : extra_headers) headers = curl_slist_append(headers, h.c_str());
    headers = append_edge_browser_headers(headers);
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
    return true;
}

// POST application/x-www-form-urlencoded.  Used by the DDG search backend
// because POST is far less likely than GET to hit the bot/captcha gate.
static bool http_post_form(const std::string & url,
                           const std::string & form_body,
                           std::string & body, std::string & err,
                           long timeout_s = 20,
                           long max_bytes = 4 * 1024 * 1024) {
    if (!url_is_safe_scheme(url)) {
        err = "only http:// and https:// URLs are allowed";
        return false;
    }
    CURL * c = curl_easy_init();
    if (!c) { err = "curl_easy_init failed"; return false; }

    body.clear();
    HttpSink sink{ &body, (size_t) max_bytes, false };
    curl_easy_setopt(c, CURLOPT_URL,             url.c_str());
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION,  1L);
    curl_easy_setopt(c, CURLOPT_MAXREDIRS,       5L);
    // CURLOPT_PROTOCOLS / _REDIR_PROTOCOLS deprecated in libcurl 7.85.0 in
    // favour of the *_STR variants. The new ones are enums (not macros), so
    // an `#ifdef CURLOPT_PROTOCOLS_STR` test silently falls through; gate on
    // the version macro instead.
#if defined(LIBCURL_VERSION_NUM) && LIBCURL_VERSION_NUM >= 0x075500
    curl_easy_setopt(c, CURLOPT_PROTOCOLS_STR,         "http,https");
    curl_easy_setopt(c, CURLOPT_REDIR_PROTOCOLS_STR,   "http,https");
#else
    curl_easy_setopt(c, CURLOPT_PROTOCOLS,
                     (long) (CURLPROTO_HTTP | CURLPROTO_HTTPS));
    curl_easy_setopt(c, CURLOPT_REDIR_PROTOCOLS,
                     (long) (CURLPROTO_HTTP | CURLPROTO_HTTPS));
#endif
    curl_easy_setopt(c, CURLOPT_TIMEOUT,         timeout_s);
    // Impersonate Microsoft Edge — html.duckduckgo.com (and most search
    // backends) reject or degrade non-browser UAs.
    curl_easy_setopt(c, CURLOPT_USERAGENT,        kEdgeUserAgent);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION,   curl_write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA,       &sink);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL,        1L);
    curl_easy_setopt(c, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(c, CURLOPT_POST,            1L);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS,      form_body.c_str());
    curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE,   (long) form_body.size());

    // Form POST + Edge persona.  append_edge_browser_headers() sets
    // Accept/Accept-Language/sec-ch-ua/etc; we add Content-Type for
    // the form payload.  The richer Accept from the Edge helper
    // supersedes the trimmed one we used to send.
    curl_slist * headers = nullptr;
    headers = curl_slist_append(headers,
        "Content-Type: application/x-www-form-urlencoded");
    headers = append_edge_browser_headers(headers);
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers);

    CURLcode rc = curl_easy_perform(c);
    long http_code = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(c);

    if (rc != CURLE_OK)   { err = curl_easy_strerror(rc); return false; }
    if (http_code >= 400) { err = "HTTP " + std::to_string(http_code); return false; }
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
// ----------------------------------------------------------------------------
// Process-wide LRU cache for fetched bodies.  The model often re-asks for
// the same URL within a single conversation (it digests results, then
// circles back).  Caching avoids:
//  - re-hitting the source (politeness + DDG rate-limit avoidance)
//  - waste of latency / tokens on the model side
//
// Cache key:  "<url>?as_html=<bool>"
// Capacity:   16 entries (well within RAM for ~16 KiB clipped bodies)
// TTL:        5 minutes (long enough to survive a multi-hop research,
//                        short enough to keep "live" pages fresh)
// ============================================================================
#if defined(EASYAI_HAVE_CURL)
namespace {

struct WebFetchCache {
    struct Entry {
        std::string                                 body;     // already stripped + clipped
        std::chrono::steady_clock::time_point       inserted;
        bool                                        as_html;
    };
    static constexpr size_t kCapacity = 16;
    static constexpr int    kTtlMs    = 5 * 60 * 1000;  // 5 min

    std::mutex                                  mu;
    std::list<std::pair<std::string, Entry>>    items;             // MRU at front
    std::unordered_map<std::string,
        std::list<std::pair<std::string, Entry>>::iterator> index;

    bool get(const std::string & key, std::string & out_body) {
        std::lock_guard<std::mutex> lock(mu);
        auto it = index.find(key);
        if (it == index.end()) return false;
        const auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - it->second->second.inserted).count();
        if (age_ms > kTtlMs) {
            items.erase(it->second);
            index.erase(it);
            return false;
        }
        // promote to MRU
        items.splice(items.begin(), items, it->second);
        out_body = it->second->second.body;
        return true;
    }
    void put(const std::string & key, std::string body, bool as_html) {
        std::lock_guard<std::mutex> lock(mu);
        auto existing = index.find(key);
        if (existing != index.end()) {
            items.erase(existing->second);
            index.erase(existing);
        }
        items.emplace_front(key, Entry{
            std::move(body), std::chrono::steady_clock::now(), as_html});
        index[key] = items.begin();
        while (items.size() > kCapacity) {
            index.erase(items.back().first);
            items.pop_back();
        }
    }
};

WebFetchCache & web_fetch_cache() {
    static WebFetchCache c;
    return c;
}

}  // namespace
#endif

Tool web_fetch() {
    return Tool::builder("web_fetch")
        .describe("Fetch a URL and return its text content (HTML stripped + "
                  "trimmed).  This is the ONLY way to read a web page's "
                  "actual content — web_search returns titles and short "
                  "snippets, not the page body.  Always call this after "
                  "web_search when the user wants the contents of an "
                  "article, documentation page, or news story.\n"
                  "\n"
                  "Pagination: fetched bodies are clipped to the first "
                  "8 KB by default.  When a body is truncated the response "
                  "ends with `[truncated: N more bytes; pass start=N to "
                  "continue]` — call web_fetch again with the same URL plus "
                  "`start=N` to read the next slice.\n"
                  "\n"
                  "Repeated fetches of the same URL within 5 minutes are "
                  "served from an in-process cache; you do NOT need to "
                  "re-fetch the same URL within one conversation.")
        .param("url",     "string", "Fully-qualified http(s) URL to fetch", true)
        .param("as_html", "boolean","If true, return raw HTML instead of stripped text", false)
        .param("start",   "integer","Byte offset into the (already stripped) body, "
                                    "for pagination.  Default 0 = beginning.", false)
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
            long long start = std::max<long long>(0,
                args::get_int_or(c.arguments_json, "start", 0));

            // Cache key: url + as_html flag.  start is applied AFTER cache
            // hit so we don't multiply storage by every offset.
            const std::string key = url + (as_html ? "|html" : "|text");
            std::string processed;
            if (!web_fetch_cache().get(key, processed)) {
                std::string body, err;
                if (!http_get(url, {}, body, err)) {
                    return ToolResult::error("fetch failed: " + err);
                }
                processed = as_html ? body : strip_html(body);
                web_fetch_cache().put(key, processed, as_html);
            }

            // Apply pagination + 8 KiB window.
            constexpr size_t kWindow = 8 * 1024;
            if ((size_t) start >= processed.size()) {
                std::ostringstream oss;
                oss << "[start=" << start << " is past end of body (size="
                    << processed.size() << "); nothing to return]";
                return ToolResult::ok(oss.str());
            }
            std::string slice = processed.substr(start, kWindow);
            const size_t remaining =
                processed.size() - start - slice.size();
            if (remaining > 0) {
                std::ostringstream oss;
                oss << slice
                    << "\n\n[truncated: " << remaining
                    << " more bytes; pass start="
                    << (start + slice.size())
                    << " to continue]";
                return ToolResult::ok(oss.str());
            }
            return ToolResult::ok(slice);
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
    //
    // SECURITY: the containment check must be PATH-COMPONENT aware, not a
    // raw string-prefix.  Otherwise a sandbox at "/srv/user" would be
    // happily satisfied by "/srv/userMALICIOUS/secrets" (same prefix,
    // different directory tree).  We require canonical to either equal
    // root or have root + path-separator as its prefix.
    // Always succeeds.  We never reject — instead we ALWAYS anchor
    // the resulting path inside the sandbox by:
    //   1. iterating the input's path components,
    //   2. dropping any "..", ".", and absolute markers ("/", "\"),
    //   3. joining the survivors onto the sandbox root.
    //
    // This means:
    //   "/news.md"          → <sandbox>/news.md
    //   "../etc/passwd"     → <sandbox>/etc/passwd        (contained)
    //   "a/../b/news.md"    → <sandbox>/a/b/news.md       (contained)
    //   "C:\\foo\\bar"      → <sandbox>/C:/foo/bar        (Windows-y but
    //                                                       still inside)
    //
    // Trade-off vs the previous canonical-resolution + containment
    // check: a malicious symlink already inside the sandbox could
    // be followed out of it, since we no longer call weakly_canonical.
    // Acceptable because the sandbox is user-owned (they pick the dir
    // with --sandbox); we don't expose symlink creation to the model.
    // In return: the model can NEVER produce a path that "escapes" —
    // any input becomes a real file path under the sandbox.
    //
    // The `err` out-param is kept for API stability (callers still
    // check the return value) but is never written; this method
    // always returns true.
    bool resolve(const std::string & in, fs::path & out, std::string & /*err*/) const {
        auto is_only_dots = [](const std::string & s) {
            if (s.empty()) return false;
            for (char c : s) if (c != '.') return false;
            return true;
        };
        fs::path raw = in;
        fs::path rel;
        for (const auto & part : raw) {
            const std::string s = part.string();
            if (s.empty())                  continue;
            if (s == "/" || s == "\\")      continue;
            if (is_only_dots(s))            continue;   // ".", "..", "...", "....", …
            rel /= part;
        }
        out = (root / rel).lexically_normal();
        return true;
    }
    // Render a real on-disk path back into the model's "/"-rooted view.
    // The sandbox base is hidden; the model only ever sees /file.md,
    // /subdir/file.md, etc.  This is the inverse of resolve() for display.
    std::string virtual_path(const fs::path & real) const {
        std::error_code ec;
        fs::path rel = fs::relative(real, root, ec);
        if (ec || rel.empty() || rel == ".") return "/";
        std::string s = rel.generic_string();
        if (s.front() != '/') s.insert(0, "/");
        return s;
    }
};

}  // namespace

Tool fs_read_file(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("read_file")
        .describe("Read a UTF-8 text file from disk and return its contents. "
                  "The filesystem you see is rooted at `/`; use paths like `/report.md` or `/docs/spec.md`.")
        .param("path",   "string",  "Path to the file, e.g. `/report.md` or `/docs/spec.md`.", true)
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
            if (!f) return ToolResult::error("cannot open: " + sb->virtual_path(p));
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
        .describe("Write text to a file (overwrites). Creates parent directories. "
                  "The filesystem you see is rooted at `/`; use paths like `/report.md` or `/docs/notes.md`.")
        .param("path",    "string",  "Destination path, e.g. `/report.md` or `/docs/notes.md`.", true)
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
            if (!f) return ToolResult::error("cannot open for write: " + sb->virtual_path(p));
            f.write(content.data(), content.size());
            return ToolResult::ok("wrote " + std::to_string(content.size())
                                  + " bytes to " + sb->virtual_path(p));
        })
        .build();
}

Tool fs_list_dir(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("list_dir")
        .describe("List the entries (files and directories) inside a directory. "
                  "The filesystem you see is rooted at `/`; use `/` for the top, `/subdir` for nested paths.")
        .param("path", "string", "Directory path, e.g. `/` or `/subdir`.", true)
        .handle([sb](const ToolCall & c) {
            std::string path;
            if (!args::get_string(c.arguments_json, "path", path))
                return ToolResult::error("missing arg: path");

            fs::path p; std::string err;
            if (!sb->resolve(path, p, err)) return ToolResult::error(err);
            if (!fs::is_directory(p))
                return ToolResult::error("not a directory: " + sb->virtual_path(p));

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
                  "'src/**/*.h'). Recursive by default. "
                  "The filesystem you see is rooted at `/`; results are returned as `/`-rooted paths.")
        .param("pattern", "string", "Wildcard pattern (* matches any chars except /; ** matches across dirs)", true)
        .param("path",    "string", "Optional starting directory, e.g. `/src` (default = `/`).", false)
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
            std::regex rx;
            try { rx = std::regex(re); }
            catch (const std::regex_error & e) {
                return ToolResult::error(std::string("bad glob pattern: ") + e.what());
            }

            std::ostringstream o;
            int n = 0;
            for (auto & e : fs::recursive_directory_iterator(start)) {
                if (!e.is_regular_file()) continue;
                std::string rel = fs::relative(e.path(), sb->root).generic_string();
                bool m = false;
                try { m = std::regex_match(rel, rx); }
                catch (const std::regex_error &) { /* unsupported; skip entry */ }
                if (m) {
                    o << sb->virtual_path(e.path()) << "\n";
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
        .describe("Search file contents for a regular expression. Returns matching lines with file:line prefixes. "
                  "The filesystem you see is rooted at `/`; result paths come back as `/`-rooted.")
        .param("pattern",       "string",  "Regular expression to search for", true)
        .param("path",          "string",  "Optional starting directory, e.g. `/src` (default = `/`).", false)
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
                std::string vpath = sb->virtual_path(e.path());
                while (std::getline(f, line)) {
                    ++lineno;
                    // libstdc++'s regex engine is recursive; running a
                    // user-supplied pattern against a multi-megabyte
                    // single line (binary blob, minified JS, base64
                    // dump) is a DoS vector via catastrophic
                    // backtracking. 64 KiB is plenty for source code
                    // and short of the failure regime.
                    if (line.size() > 64 * 1024) continue;
                    if (std::regex_search(line, rx)) {
                        o << vpath << ":" << lineno << ": " << clip(line, 240) << "\n";
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

// ============================================================================
// shell — run /bin/sh -c with merged stdout/stderr, cwd pinned to sandbox.
// ============================================================================
//
// NOT a hardened sandbox.  The child has the caller's full privileges.
// Capping is purely cooperative:
//   - chdir(root) before exec
//   - 32 KB output cap (the rest is silently dropped, with a marker)
//   - SIGTERM at deadline, SIGKILL 2s later
//
// We deliberately use /bin/sh -c so the model can pipe, redirect, &&,
// quote, etc — i.e. behave like a normal shell user.
// get_current_dir — returns the process's CWD. Cheap, parameterless,
// safe. The CLI / server chdir() into the --sandbox root at startup, so
// the path this returns is exactly the directory the model's other
// tools (bash, fs_*) will operate in. We resolve via getcwd() at call
// time (not at registration) so a process that chdir'd later still
// reports truthfully.
//
// Two error paths surfaced as ToolResult::error:
//   - getcwd returns nullptr (path too long, racing rmdir, EACCES on
//     a parent component): we surface errno via strerror.
//   - The path doesn't fit in PATH_MAX. POSIX guarantees PATH_MAX
//     bytes are enough for any valid path that the kernel would let
//     us land in via chdir, so this is a "should never happen" branch
//     kept defensively.
Tool get_current_dir() {
    return Tool::builder("get_current_dir")
        .describe(
            "Returns the absolute path of the directory this agent is "
            "running in. Other tools that take relative paths (bash, "
            "read_file, write_file, list_dir, glob, grep) resolve them "
            "against this directory. Call this once at the start of a "
            "task if you need to know the absolute path; otherwise just "
            "use relative paths and trust they land here. No parameters."
        )
        .handle([](const ToolCall & /*c*/) {
            // PATH_MAX-sized stack buffer is the standard POSIX idiom;
            // getcwd writes at most PATH_MAX bytes (incl. NUL) so this
            // is bounded by the platform.
            char buf[PATH_MAX];
            if (::getcwd(buf, sizeof(buf)) == nullptr) {
                return ToolResult::error(
                    std::string("getcwd failed: ") + std::strerror(errno));
            }
            // strnlen is overkill (getcwd guarantees NUL-terminated on
            // success) but treat it as a defence-in-depth bound anyway.
            const size_t n = ::strnlen(buf, sizeof(buf));
            return ToolResult::ok(std::string(buf, n));
        })
        .build();
}

Tool bash(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("bash")
        .describe(
            "Run a shell command via `/bin/sh -c`. Output is stdout and stderr "
            "merged; the working directory is the sandbox root — call "
            "`get_current_dir` first if you need to know the absolute path "
            "of where your command will run. Relative paths in your command "
            "(e.g. `./build`, `src/main.cpp`) resolve against that directory. "
            "Use this for grep | xargs, find, git, package managers, anything "
            "you'd type in a terminal. "
            "WARNING: this is NOT a hardened sandbox — the command runs with the "
            "caller's user privileges and can read/write files, hit the network, "
            "spawn processes, etc. Prefer the dedicated tools (read_file, "
            "write_file, glob, grep) for simple file ops; reach for bash only "
            "when you actually need shell features."
        )
        .param("command", "string",
               "Shell command line. Quoted, piped, redirected etc. as you would "
               "type it in a terminal. Example: `ls -la | head -20`.", true)
        .param("timeout_sec", "integer",
               "Max seconds to run before SIGTERM/SIGKILL. Default 30, max 300.",
               false)
        .handle([sb](const ToolCall & c) {
            std::string cmd;
            long long timeout_sec = 30;
            if (!args::get_string(c.arguments_json, "command", cmd) || cmd.empty())
                return ToolResult::error("missing arg: command");
            args::get_int(c.arguments_json, "timeout_sec", timeout_sec);
            if (timeout_sec < 1)   timeout_sec = 1;
            if (timeout_sec > 300) timeout_sec = 300;

            int pipefd[2];
            if (::pipe(pipefd) < 0)
                return ToolResult::error(std::string("pipe() failed: ") + std::strerror(errno));

            const std::string cwd = sb->root.string();
            pid_t pid = ::fork();
            if (pid < 0) {
                ::close(pipefd[0]); ::close(pipefd[1]);
                return ToolResult::error(std::string("fork() failed: ") + std::strerror(errno));
            }
            if (pid == 0) {
                // child
                ::close(pipefd[0]);
                ::dup2(pipefd[1], 1);
                ::dup2(pipefd[1], 2);
                ::close(pipefd[1]);
                if (!cwd.empty() && ::chdir(cwd.c_str()) != 0) {
                    std::fprintf(stderr, "chdir(%s) failed: %s\n",
                                 cwd.c_str(), std::strerror(errno));
                    ::_exit(126);
                }
                ::execl("/bin/sh", "sh", "-c", cmd.c_str(), (char *) nullptr);
                ::_exit(127);
            }
            // parent
            ::close(pipefd[1]);
            ::fcntl(pipefd[0], F_SETFL, O_NONBLOCK);

            constexpr size_t kCap = 32 * 1024;
            std::string out;
            out.reserve(4096);

            auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::seconds(timeout_sec);
            bool sent_term = false;
            bool killed_for_timeout = false;
            int  status = 0;
            bool reaped = false;

            auto drain_pipe = [&]() {
                char buf[4096];
                ssize_t n;
                while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0) {
                    if (out.size() < kCap) {
                        size_t take = std::min((size_t) n, kCap - out.size());
                        out.append(buf, take);
                    }
                }
            };

            for (;;) {
                auto now = std::chrono::steady_clock::now();
                if (now >= deadline) {
                    if (!sent_term) {
                        ::kill(pid, SIGTERM);
                        sent_term          = true;
                        killed_for_timeout = true;   // we initiated it
                        deadline           = now + std::chrono::seconds(2); // grace
                    } else {
                        ::kill(pid, SIGKILL);
                        break;
                    }
                }
                int wait_ms = (int) std::chrono::duration_cast<std::chrono::milliseconds>(
                                  deadline - now).count();
                if (wait_ms < 0)   wait_ms = 0;
                if (wait_ms > 200) wait_ms = 200;

                struct pollfd pfd { pipefd[0], POLLIN, 0 };
                int rc = ::poll(&pfd, 1, wait_ms);
                if (rc < 0 && errno == EINTR) continue;
                if (rc > 0 && (pfd.revents & (POLLIN | POLLHUP))) {
                    drain_pipe();
                }

                pid_t r = ::waitpid(pid, &status, WNOHANG);
                if (r == pid) {
                    drain_pipe();
                    reaped = true;
                    break;
                }
            }

            if (!reaped) {
                drain_pipe();
                ::waitpid(pid, &status, 0);
            }
            ::close(pipefd[0]);

            std::ostringstream oss;
            if (killed_for_timeout) {
                oss << "exit=-1  [killed: timeout after " << timeout_sec << "s]\n";
            } else if (WIFEXITED(status)) {
                oss << "exit=" << WEXITSTATUS(status) << "\n";
            } else if (WIFSIGNALED(status)) {
                oss << "exit=signal:" << WTERMSIG(status) << "\n";
            } else {
                oss << "exit=?\n";
            }
            std::string body = std::move(out);
            if (body.size() >= kCap) body += "\n[truncated at 32 KB]\n";
            return ToolResult::ok(oss.str() + body);
        })
        .build();
}

}  // namespace easyai::tools
