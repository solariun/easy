#include "easyai/builtin_tools.hpp"
#include "easyai/log.hpp"
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
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#if defined(EASYAI_HAVE_CURL)
#include <curl/curl.h>
#endif

// nlohmann/json reaches us transitively through llama-common's interface
// includes (vendor copy at llama.cpp/vendor/nlohmann/json.hpp). Used by
// web_google to parse the Custom Search JSON API response.
#include <nlohmann/json.hpp>

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

// Default extra-attempts on transient curl/HTTP failures for web tools.
// Web fetches go to the open internet so transient failures are normal —
// 5 retries gives operator-grade resilience without spending forever on
// a hard-down site.  Each retry logs through easyai::log::error so it
// surfaces on stderr without --verbose.
static constexpr int kWebHttpRetries = 5;

static bool web_curl_retryable(CURLcode rc) {
    switch (rc) {
        case CURLE_COULDNT_CONNECT:
        case CURLE_COULDNT_RESOLVE_HOST:
        case CURLE_COULDNT_RESOLVE_PROXY:
        case CURLE_OPERATION_TIMEDOUT:
        case CURLE_RECV_ERROR:
        case CURLE_SEND_ERROR:
        case CURLE_GOT_NOTHING:
        case CURLE_PARTIAL_FILE:
            return true;
        default:
            return false;
    }
}

static int web_retry_backoff_ms(int attempt) {
    int ms = 250 << (attempt > 4 ? 4 : attempt);
    return ms > 4000 ? 4000 : ms;
}

static bool http_get(const std::string & url,
                     const std::vector<std::string> & extra_headers,
                     std::string & body, std::string & err,
                     long timeout_s = 20, long max_bytes = 2 * 1024 * 1024,
                     int retries = kWebHttpRetries) {
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

    const int max_attempts = (retries < 0 ? 0 : retries) + 1;
    CURLcode rc = CURLE_OK;
    long http_code = 0;
    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        body.clear();
        sink.body      = &body;
        sink.truncated = false;
        rc = curl_easy_perform(c);
        http_code = 0;
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &http_code);

        const bool curl_ok = (rc == CURLE_OK);
        const bool http_ok = (http_code >= 200 && http_code < 300) || http_code == 0;
        if (curl_ok && http_ok) break;

        const bool retryable = (!curl_ok && web_curl_retryable(rc))
                            || (curl_ok && http_code >= 500 && http_code < 600);
        if (!retryable) break;
        if (attempt >= max_attempts) {
            easyai::log::error(
                "[easyai-web] GET %s attempt %d/%d failed (%s) — "
                "retry budget exhausted\n",
                url.c_str(), attempt, max_attempts,
                curl_ok ? ("HTTP " + std::to_string(http_code)).c_str()
                        : curl_easy_strerror(rc));
            break;
        }
        const int backoff = web_retry_backoff_ms(attempt - 1);
        easyai::log::error(
            "[easyai-web] GET %s attempt %d/%d failed (%s); retrying in %dms\n",
            url.c_str(), attempt, max_attempts,
            curl_ok ? ("HTTP " + std::to_string(http_code)).c_str()
                    : curl_easy_strerror(rc),
            backoff);
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff));
    }
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
                           long max_bytes = 4 * 1024 * 1024,
                           int  retries   = kWebHttpRetries) {
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

    const int max_attempts = (retries < 0 ? 0 : retries) + 1;
    CURLcode rc = CURLE_OK;
    long http_code = 0;
    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        body.clear();
        sink.body      = &body;
        sink.truncated = false;
        rc = curl_easy_perform(c);
        http_code = 0;
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &http_code);

        const bool curl_ok = (rc == CURLE_OK);
        const bool http_ok = (http_code >= 200 && http_code < 300) || http_code == 0;
        if (curl_ok && http_ok) break;

        const bool retryable = (!curl_ok && web_curl_retryable(rc))
                            || (curl_ok && http_code >= 500 && http_code < 600);
        if (!retryable) break;
        if (attempt >= max_attempts) {
            easyai::log::error(
                "[easyai-web] POST %s attempt %d/%d failed (%s) — "
                "retry budget exhausted\n",
                url.c_str(), attempt, max_attempts,
                curl_ok ? ("HTTP " + std::to_string(http_code)).c_str()
                        : curl_easy_strerror(rc));
            break;
        }
        const int backoff = web_retry_backoff_ms(attempt - 1);
        easyai::log::error(
            "[easyai-web] POST %s attempt %d/%d failed (%s); retrying in %dms\n",
            url.c_str(), attempt, max_attempts,
            curl_ok ? ("HTTP " + std::to_string(http_code)).c_str()
                    : curl_easy_strerror(rc),
            backoff);
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff));
    }
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
        .describe(
            "Return the current wall-clock time. Output is two lines: "
            "`UTC:   YYYY-MM-DD HH:MM:SS` and `Local: YYYY-MM-DD HH:MM:SS "
            "<TZ>` (server's local timezone). No parameters.\n"
            "\n"
            "Use when the user asks for the time/date, when you need an "
            "anchor for relative phrasing (\"yesterday\", \"in 3 days\"), "
            "or before any date arithmetic. Example call: {}.")
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
// web_google — Google Custom Search JSON API client
// ----------------------------------------------------------------------------
// Why an API rather than a scraper: google.com/search aggressively gates
// scripted clients (CAPTCHA, "unusual traffic", JS-only result panels) so
// a scraper version is unreliable in seconds. The Custom Search JSON API
// is the supported, stable interface — small JSON, predictable schema,
// no UA games — at the cost of needing two env vars (GOOGLE_API_KEY +
// GOOGLE_CSE_ID). Free tier is 100 queries/day; over that you pay per 1k.
//
// We read the env vars at CALL time (not registration) so:
//   1. A long-running server picks up newly-rotated keys without restart.
//   2. The tool can produce a clear actionable error when a key is
//      missing instead of silently disappearing from the catalog.
// The cli.cpp / Toolbelt registration still gates exposure on the env
// being set, so the model only sees the tool when it would actually work
// — but if it does see it and the keys vanish later, the error message
// tells the user exactly which variable to set.
// ============================================================================
Tool web_google() {
    return Tool::builder("web_google")
        .describe("Search the web via Google's Custom Search JSON API. "
                  "Returns a numbered list of title / url / snippet results "
                  "— same shape as web_search but using Google's index. "
                  "Snippets are 1-2 short sentences; NEVER summarize from "
                  "them alone. After this call, you MUST call web_fetch on "
                  "the top 1-3 most relevant URLs to read the actual page "
                  "content, then base your answer on the fetched text.\n"
                  "\n"
                  "Requires GOOGLE_API_KEY and GOOGLE_CSE_ID environment "
                  "variables. Free tier is capped at 100 queries/day "
                  "across the whole API key.")
        .param("query",       "string",  "Search query", true)
        .param("max_results", "integer", "Maximum results to return "
                                         "(default 5, max 10 — the per-call "
                                         "ceiling of Google CSE)", false)
        .handle([](const ToolCall & c) {
#if !defined(EASYAI_HAVE_CURL)
            (void) c;
            return ToolResult::error(
                "web_google unavailable: easyai built without libcurl");
#else
            std::string query;
            if (!args::get_string(c.arguments_json, "query", query) || query.empty()) {
                return ToolResult::error("missing required arg: query");
            }
            long long max_results = 5;
            args::get_int(c.arguments_json, "max_results", max_results);
            if (max_results < 1)  max_results = 1;
            if (max_results > 10) max_results = 10;   // CSE per-call ceiling

            const char * api_key = std::getenv("GOOGLE_API_KEY");
            const char * cse_id  = std::getenv("GOOGLE_CSE_ID");
            if (!api_key || !*api_key) {
                return ToolResult::error(
                    "GOOGLE_API_KEY env var not set — get one at "
                    "https://console.cloud.google.com/apis/credentials "
                    "(enable 'Custom Search API' first)");
            }
            if (!cse_id || !*cse_id) {
                return ToolResult::error(
                    "GOOGLE_CSE_ID env var not set — create a Programmable "
                    "Search Engine at https://programmablesearchengine.google.com "
                    "and copy the 'cx' value");
            }

            // Build the GET URL. The API ignores trailing whitespace but we
            // url_encode every component so a query containing &, =, # or
            // unicode is preserved verbatim.
            std::string url = "https://www.googleapis.com/customsearch/v1?";
            url += "key=" + url_encode(api_key);
            url += "&cx=" + url_encode(cse_id);
            url += "&q="  + url_encode(query);
            url += "&num=" + std::to_string(max_results);
            // Safe-search default — Google's "off" leaves NSFW in the
            // mix, "active" filters explicit content. We pick "active"
            // for the same reason web_search uses kl=us-en: predictable
            // results in a tool the model is steering autonomously.
            url += "&safe=active";

            std::string body, err;
            // 1 MiB cap is plenty — CSE responses are typically <50 KiB
            // even at num=10. Default 20s timeout is fine for an API call.
            if (!http_get(url, {}, body, err, 20, 1024 * 1024)) {
                // Google returns 4xx with a JSON error body explaining
                // what's wrong (bad key, quota, disabled API). http_get
                // dropped the body on HTTP>=400, so we can only surface
                // the status code — but the message is still actionable
                // because the most common 4xx is "key invalid / quota".
                return ToolResult::error("google search failed: " + err
                    + " (verify GOOGLE_API_KEY, GOOGLE_CSE_ID, and that "
                      "Custom Search API is enabled in your Cloud project)");
            }

            // ----- parse JSON -----------------------------------------------
            // Schema:
            //   { "items": [ { "title", "link", "snippet" }, ... ],
            //     "searchInformation": { "totalResults": "..." } }
            // Missing items = no results; we don't treat that as an error.
            nlohmann::json j;
            try {
                j = nlohmann::json::parse(body);
            } catch (const std::exception & e) {
                return ToolResult::error(
                    std::string("google search: malformed JSON response: ") + e.what());
            }

            // Some error conditions (e.g. malformed cx) return 200 with
            // {"error": {"message": "..."}}. Surface that explicitly.
            if (j.contains("error") && j["error"].is_object()) {
                std::string msg = j["error"].value("message", "unknown error");
                return ToolResult::error("google search rejected request: " + msg);
            }

            std::ostringstream out;
            out << "Top web results for: " << query << "\n";
            int count = 0;
            if (j.contains("items") && j["items"].is_array()) {
                for (const auto & item : j["items"]) {
                    if (count >= max_results) break;
                    std::string title = item.value("title",   std::string{});
                    std::string link  = item.value("link",    std::string{});
                    std::string snip  = item.value("snippet", std::string{});
                    // Google's snippet field has literal "\n" sequences for
                    // line wraps in the rendered card; flatten them so the
                    // result list reads as a single line per snippet.
                    std::replace(snip.begin(), snip.end(), '\n', ' ');
                    if (link.empty() || title.empty()) continue;
                    out << "\n" << (count + 1) << ". " << title
                        << "\n   " << link
                        << "\n   " << clip(snip, 300) << "\n";
                    ++count;
                }
            }

            if (count == 0) {
                return ToolResult::error(
                    "no results — try a broader query, or verify the CSE "
                    "is configured to search the entire web (not just a "
                    "single site) at https://programmablesearchengine.google.com");
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
    // Render a real on-disk path back into the model's relative view.
    // The sandbox base is hidden; the model sees `report.md`,
    // `src/main.cpp`, `.` for the root itself.  Relative form matches
    // what every fs_* / bash description tells the model to USE as
    // input — so grep/glob output can be fed straight back into
    // read_file without a leading-slash dance.
    std::string virtual_path(const fs::path & real) const {
        std::error_code ec;
        fs::path rel = fs::relative(real, root, ec);
        if (ec || rel.empty() || rel == ".") return ".";
        return rel.generic_string();
    }
    // Belt-and-braces containment check, called at file-access time.
    // resolve() mechanically anchors the input under root; this method
    // runs the canonical-resolution check once we hold the final path
    // so a malicious symlink already inside the sandbox (e.g. planted
    // by the bash tool) cannot redirect a follow-up open() outside.
    //
    // Two-layer strategy:
    //   1. Lexical containment — cheap, no FS access. resolve() always
    //      anchors its output under root by construction, so any path
    //      we got from resolve() passes this. Required: false here is
    //      a definitive "not under root, not even on paper".
    //   2. Symlink-safe canonical check via weakly_canonical. Catches
    //      symlinks anywhere in the path that would redirect us out of
    //      root. When canonicalisation FAILS (typically EACCES on a
    //      0000 parent dir, or ENOENT racing rmdir), we fall back to
    //      the lexical answer instead of rejecting — failing closed
    //      here used to break fs_check_path on exactly the paths the
    //      operator most wanted to probe (e.g. files inside an
    //      0000-perm parent that the model wants to know about).
    bool inside_sandbox(const fs::path & p) const {
        // Path-component prefix match — avoids the string-prefix bug
        // where "/srv/user" prefix-matches "/srv/userMALICIOUS/secret".
        auto component_prefix = [](const fs::path & prefix,
                                   const fs::path & full) {
            auto it_pre = prefix.begin();
            auto it_ful = full.begin();
            for (; it_pre != prefix.end(); ++it_pre, ++it_ful) {
                if (it_ful == full.end() || *it_ful != *it_pre) return false;
            }
            return true;
        };
        // Layer 1: lexical containment. Fail-closed if the path on
        // paper is outside the sandbox (would only happen if a caller
        // constructed `p` themselves rather than going through resolve()).
        if (!component_prefix(root, p.lexically_normal())) return false;

        // Layer 2: bonus symlink defence. If canonicalisation fails
        // (perm error, race), trust layer 1 — we know the path is
        // lexically inside, and the subsequent open() / lstat() will
        // surface the real errno far more usefully than a blanket
        // "escapes sandbox" rejection.
        std::error_code ec;
        fs::path canon_p = fs::weakly_canonical(p, ec);
        if (ec) return true;
        fs::path canon_r = fs::weakly_canonical(root, ec);
        if (ec) return true;
        return component_prefix(canon_r, canon_p);
    }
};

// Regex metacharacters that need escaping when we lift a glob pattern
// into a regex. Curly braces and brackets are included even though they
// only become metachars in some POSIX dialects — std::regex (ECMAScript
// by default) tolerates the over-escape and we'd rather be conservative
// than leak a syntax error from a model-supplied pattern.
constexpr const char * kGlobRegexMetachars = ".+()|^$\\{}[]";

// Convert a shell-style glob into an anchored ECMAScript regex.
//   *   -> [^/]*    (matches anything except a path separator)
//   **  -> .*       (matches anything, including separators)
//   ?   -> [^/]     (one char except a separator)
//   regex metachars are escaped; everything else passes through.
// The output is wrapped in ^...$ so std::regex_match treats it as a
// whole-string predicate — same semantics as ::fnmatch(3) without the
// platform's globbing edge cases.
inline std::string glob_to_regex(const std::string & pattern) {
    std::string re = "^";
    re.reserve(pattern.size() * 2 + 2);
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        const char ch = pattern[i];
        if (ch == '*') {
            const bool is_double_star =
                (i + 1 < pattern.size() && pattern[i + 1] == '*');
            if (is_double_star) { re += ".*"; ++i; }
            else                { re += "[^/]*"; }
        } else if (ch == '?') {
            re += "[^/]";
        } else if (std::strchr(kGlobRegexMetachars, ch)) {
            re += '\\';
            re += ch;
        } else {
            re += ch;
        }
    }
    re += '$';
    return re;
}

}  // namespace

Tool fs_read_file(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("read_file")
        .describe(
            "Read a UTF-8 text file from disk and return its contents.\n"
            "\n"
            "AUTHORITATIVE SANDBOX RULE — before your first fs_*/bash "
            "call in a task, run `get_sandbox_path` (the absolute on-disk "
            "root, pinned at registration; this is the truth) and then "
            "`fs_check_path` against the file you intend to read so you "
            "have confirmed it exists and is readable. Skipping this is "
            "the most common cause of avoidable error loops.\n"
            "\n"
            "PATHS ARE RELATIVE to the sandbox root. Use `report.md`, "
            "`docs/spec.md`, `src/main.cpp` — NEVER prefix with `/`. "
            "Required: path. Optional: offset, limit (default returns the "
            "first 64 KB; pass offset to page through larger files).\n"
            "\n"
            "Examples:\n"
            "  {path:\"report.md\"}\n"
            "  {path:\"docs/spec.md\", offset:65536, limit:65536}\n"
            "\n"
            "Errors return a single-line message starting with `error:`. "
            "Reading a binary file returns the raw bytes — prefer the "
            "dedicated tools or `bash` (e.g. `file <path>`) for those.")
        .param("path",   "string",
               "Required. RELATIVE file path under the sandbox root, e.g. "
               "`report.md` or `docs/spec.md`. No leading `/`. Confirm with "
               "`fs_check_path` first.", true)
        .param("offset", "integer",
               "Optional. Skip this many bytes from the start of the file "
               "before reading. Default 0. Use the previous read's "
               "(offset + bytes_returned) to page forward.", false)
        .param("limit",  "integer",
               "Optional. Maximum bytes to return. Default 65536 (64 KB). "
               "Larger files are truncated; raise this only when you "
               "really need a bigger chunk.", false)
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
            if (!sb->inside_sandbox(p)) {
                return ToolResult::error("path escapes sandbox via symlink: "
                                         + sb->virtual_path(p));
            }
            // Open with O_NOFOLLOW + O_CLOEXEC so a symlink at the leaf
            // (e.g. one planted by the bash tool) cannot redirect us
            // out of the sandbox between the containment check and the
            // open(). The check above already canonicalises but a TOCTOU
            // race on fast-changing filesystems would still escape it
            // without O_NOFOLLOW.
            int fd = ::open(p.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
            if (fd < 0) {
                return ToolResult::error(std::string("cannot open: ")
                                         + sb->virtual_path(p)
                                         + " (" + std::strerror(errno) + ")");
            }
            if (offset > 0 && ::lseek(fd, offset, SEEK_SET) < 0) {
                ::close(fd);
                return ToolResult::error(std::string("seek failed: ")
                                         + std::strerror(errno));
            }
            std::string buf((size_t) limit, '\0');
            ssize_t n = ::read(fd, buf.data(), (size_t) limit);
            ::close(fd);
            if (n < 0) {
                return ToolResult::error(std::string("read failed: ")
                                         + std::strerror(errno));
            }
            buf.resize((size_t) n);
            return ToolResult::ok(std::move(buf));
        })
        .build();
}

Tool fs_write_file(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("write_file")
        .describe(
            "Write UTF-8 text to a file. Default mode OVERWRITES any "
            "existing content; pass append=true to extend instead. Missing "
            "parent directories are created automatically.\n"
            "\n"
            "AUTHORITATIVE SANDBOX RULE — before your first fs_*/bash "
            "call in a task, run `get_sandbox_path` (the absolute on-disk "
            "root, pinned at registration; this is the truth) and then "
            "`fs_check_path` with `touch:true` for the destination so "
            "you have confirmed write access in the sandbox. Skipping "
            "this turns permission errors into surprises.\n"
            "\n"
            "PATHS ARE RELATIVE to the sandbox root. Use `report.md`, "
            "`docs/notes.md`, `src/main.cpp` — NEVER prefix with `/`. "
            "Required: path, content. Optional: append.\n"
            "\n"
            "Examples:\n"
            "  {path:\"report.md\", content:\"# Title\\n\\nBody...\"}\n"
            "  {path:\"log.txt\", content:\"line\\n\", append:true}\n"
            "\n"
            "On success returns `wrote N bytes to <path>`. Errors return a "
            "single-line message starting with `error:`.")
        .param("path",    "string",
               "Required. RELATIVE destination file path under the sandbox "
               "root. No leading `/`. Parent directories are created if "
               "missing. Examples: `report.md`, `docs/notes.md`. Validate "
               "writability with `fs_check_path` first.", true)
        .param("content", "string",
               "Required. UTF-8 text to write. Use `\\n` for newlines. "
               "Binary content is not supported — use bash for that.", true)
        .param("append",  "boolean",
               "Optional. If true, append to the file instead of "
               "overwriting it (creates the file if missing). Default "
               "false (overwrite).", false)
        .handle([sb](const ToolCall & c) {
            std::string path, content; bool append = false;
            if (!args::get_string(c.arguments_json, "path",    path))
                return ToolResult::error("missing arg: path");
            if (!args::get_string(c.arguments_json, "content", content))
                return ToolResult::error("missing arg: content");
            args::get_bool(c.arguments_json, "append", append);

            fs::path p; std::string err;
            if (!sb->resolve(path, p, err)) return ToolResult::error(err);
            if (!sb->inside_sandbox(p)) {
                return ToolResult::error("path escapes sandbox via symlink: "
                                         + sb->virtual_path(p));
            }

            std::error_code ec;
            fs::create_directories(p.parent_path(), ec);

            // Open with O_NOFOLLOW so a symlink at the leaf cannot
            // redirect the write outside the sandbox. O_CLOEXEC keeps
            // the fd off subsequently-spawned children. Mode 0600 —
            // model-written files are private to the service user
            // (containing potentially-sensitive content the model was
            // told to save).
            int flags = O_WRONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC
                      | (append ? O_APPEND : O_TRUNC);
            int fd = ::open(p.c_str(), flags, 0600);
            if (fd < 0) {
                return ToolResult::error(std::string("cannot open for write: ")
                                         + sb->virtual_path(p)
                                         + " (" + std::strerror(errno) + ")");
            }
            const char * data = content.data();
            size_t       left = content.size();
            while (left > 0) {
                ssize_t n = ::write(fd, data, left);
                if (n < 0) {
                    if (errno == EINTR) continue;
                    ::close(fd);
                    return ToolResult::error(std::string("write failed: ")
                                             + std::strerror(errno));
                }
                data += n;
                left -= (size_t) n;
            }
            ::close(fd);
            return ToolResult::ok("wrote " + std::to_string(content.size())
                                  + " bytes to " + sb->virtual_path(p));
        })
        .build();
}

Tool fs_list_dir(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("list_dir")
        .describe(
            "List the entries (files and directories) inside one directory, "
            "non-recursively. Use `glob` for recursive / pattern matching.\n"
            "\n"
            "AUTHORITATIVE SANDBOX RULE — before your first fs_*/bash "
            "call in a task, run `get_sandbox_path` (the absolute on-disk "
            "root, pinned at registration; this is the truth) and then "
            "`fs_check_path` against the directory you intend to list so "
            "you have confirmed it exists and is readable.\n"
            "\n"
            "Output is one entry per line, sorted, with a trailing `/` on "
            "directories. Hidden entries (dotfiles) are included. PATHS "
            "ARE RELATIVE to the sandbox root — use `.` for the root "
            "itself, `subdir` / `src/headers` for nested. NEVER prefix "
            "with `/`. Required: path.\n"
            "\n"
            "Examples:\n"
            "  {path:\".\"}              # sandbox root entries\n"
            "  {path:\"src\"}            # one nested level\n"
            "\n"
            "Errors (path missing, not a directory) return a single-line "
            "message starting with `error:`.")
        .param("path", "string",
               "Required. RELATIVE directory path under the sandbox root. "
               "Use `.` for the root, `subdir` for nested paths. No leading "
               "`/`. Trailing slashes accepted. Validate with `fs_check_path` "
               "first.", true)
        .handle([sb](const ToolCall & c) {
            std::string path;
            if (!args::get_string(c.arguments_json, "path", path))
                return ToolResult::error("missing arg: path");

            fs::path p; std::string err;
            if (!sb->resolve(path, p, err)) return ToolResult::error(err);
            if (!sb->inside_sandbox(p)) {
                return ToolResult::error("path escapes sandbox via symlink: "
                                         + sb->virtual_path(p));
            }
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
        .describe(
            "Find files by wildcard pattern. Recursive by default. The "
            "`path` argument names a STARTING DIRECTORY (or is omitted, "
            "defaulting to the sandbox root). Pointing it at a single "
            "file is an error — globs scan trees, not files.\n"
            "\n"
            "AUTHORITATIVE SANDBOX RULE — before your first fs_*/bash "
            "call in a task, run `get_sandbox_path` (the absolute on-disk "
            "root, pinned at registration; this is the truth) and then "
            "`fs_check_path` against the starting directory so you have "
            "confirmed it exists, is readable, and is `type: directory` "
            "before you scan it.\n"
            "\n"
            "Pattern grammar: `*` matches any run of characters except `/`; "
            "`**` matches across directory boundaries; `?` matches one "
            "character; `[abc]` matches one of a set. Required: pattern. "
            "Optional: path (the starting directory; defaults to `.`, the "
            "sandbox root).\n"
            "\n"
            "PATHS ARE RELATIVE to the sandbox root. Output is one matching "
            "RELATIVE path per line, sorted. Examples:\n"
            "  {pattern:\"*.cpp\"}                       # any .cpp anywhere\n"
            "  {pattern:\"**/*.hpp\", path:\"src\"}       # .hpp under src/\n"
            "  {pattern:\"src/**/*.{c,cc,cpp}\"}          # multiple suffixes\n"
            "\n"
            "Use `grep` to search file contents; use `list_dir` to browse "
            "a single non-recursive level.")
        .param("pattern", "string",
               "Required. Wildcard pattern. `*` matches anything except "
               "`/`, `**` crosses directory boundaries, `?` matches one "
               "char, `[abc]` matches a character class.", true)
        .param("path",    "string",
               "Optional. RELATIVE starting directory under the sandbox "
               "root. Default `.` (the root). No leading `/`. Example: "
               "`src`.", false)
        .handle([sb](const ToolCall & c) {
            std::string pattern, sub;
            if (!args::get_string(c.arguments_json, "pattern", pattern))
                return ToolResult::error("missing arg: pattern");
            args::get_string(c.arguments_json, "path", sub);

            fs::path start = sb->root; std::string err;
            if (!sub.empty() && !sb->resolve(sub, start, err))
                return ToolResult::error(err);
            if (!sb->inside_sandbox(start)) {
                return ToolResult::error("start dir escapes sandbox via symlink: "
                                         + sb->virtual_path(start));
            }
            // Precheck: a clear `error:` line beats a stray
            // `filesystem_error` exception when the model passes a path
            // that doesn't exist or names a regular file.
            std::error_code ec_pre;
            if (!fs::is_directory(start, ec_pre)) {
                return ToolResult::error(
                    std::string("not a directory: ") + sb->virtual_path(start)
                    + (ec_pre ? std::string(" (") + ec_pre.message() + ")"
                              : std::string()));
            }

            std::regex rx;
            try { rx = std::regex(glob_to_regex(pattern)); }
            catch (const std::regex_error & e) {
                return ToolResult::error(std::string("bad glob pattern: ") + e.what());
            }

            // skip_permission_denied lets the iterator silently step
            // past 0700/0000 subdirs the running user can't enumerate
            // instead of throwing mid-loop. Combined with the ec
            // constructor and the ec-overloads on per-entry queries
            // below, glob now degrades gracefully across mixed-perms
            // sandboxes (e.g. a `--sandbox $HOME` with a ~/.gnupg in
            // it).
            constexpr auto kIterOpts =
                fs::directory_options::skip_permission_denied;
            std::error_code ec_it;
            fs::recursive_directory_iterator it(start, kIterOpts, ec_it);
            if (ec_it) {
                return ToolResult::error(
                    std::string("cannot iterate: ") + sb->virtual_path(start)
                    + " (" + ec_it.message() + ")");
            }
            const fs::recursive_directory_iterator end;

            std::ostringstream o;
            int n = 0;
            // Hand-rolled loop with an ec-aware increment so a flake
            // mid-traversal (vanished entry, race) skips the bad
            // subtree instead of tearing down the whole call. Ranged-
            // for would call the throwing operator++ overload.
            for (; it != end; ) {
                std::error_code ec_q;
                if (it->is_regular_file(ec_q) && !ec_q) {
                    std::string rel =
                        fs::relative(it->path(), sb->root, ec_q).generic_string();
                    if (!ec_q) {
                        bool m = false;
                        try { m = std::regex_match(rel, rx); }
                        catch (const std::regex_error &) { /* skip entry */ }
                        if (m) {
                            o << sb->virtual_path(it->path()) << "\n";
                            if (++n >= 500) {
                                o << "...[stopped at 500 matches]\n";
                                break;
                            }
                        }
                    }
                }
                std::error_code ec_step;
                it.increment(ec_step);
                if (ec_step) {
                    // Drop the offending subtree and keep going. We
                    // already opted into skip_permission_denied so
                    // this branch is reserved for genuinely flaky
                    // errors (vanished entries on a busy FS, etc.).
                    if (it == end) break;
                    it.pop();
                    if (it == end) break;
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
        .describe(
            "Search file contents for a regular expression, recursively "
            "across a DIRECTORY TREE. The `path` argument MUST name a "
            "directory (or be omitted, defaulting to the sandbox root). "
            "Passing a single file's path is an error — to search within "
            "one file, call `read_file` and look directly, or use `bash` "
            "(`grep PATTERN file.txt`).\n"
            "\n"
            "AUTHORITATIVE SANDBOX RULE — before your first fs_*/bash "
            "call in a task, run `get_sandbox_path` (the absolute on-disk "
            "root, pinned at registration; this is the truth) and then "
            "`fs_check_path` against the starting directory so you have "
            "confirmed it exists, is readable, and is `type: directory` "
            "before you scan it.\n"
            "\n"
            "Regex flavor is ECMAScript (same as JavaScript / std::regex): "
            "`.`, `*`, `+`, `?`, `()`, `|`, `[]`, `\\d`, `\\w`, `\\s`, "
            "anchors `^`/`$`. Each line is matched independently.\n"
            "\n"
            "Output: one match per line as `<path>:<lineno>:<line>`, paths "
            "RELATIVE to the sandbox root, sorted by path. Stops after "
            "`max_matches` (default 100).\n"
            "\n"
            "Required: pattern. Optional: path (start dir, default `.` — "
            "the sandbox root), file_glob (limit by filename), "
            "max_matches, case_insensitive. NEVER prefix paths with `/`.\n"
            "\n"
            "Examples:\n"
            "  {pattern:\"TODO\"}\n"
            "  {pattern:\"class\\\\s+\\\\w+\", path:\"src\", file_glob:\"*.hpp\"}\n"
            "  {pattern:\"error\", case_insensitive:true, max_matches:50}\n"
            "\n"
            "Use `glob` to find files by name; use `read_file` to read one.")
        .param("pattern",       "string",
               "Required. ECMAScript regular expression. Each line is "
               "tested with regex_search (substring match), so anchor with "
               "`^` / `$` if you want full-line matches.", true)
        .param("path",          "string",
               "Optional. RELATIVE starting directory under the sandbox "
               "root. Default `.` (the root). No leading `/`. Example: "
               "`src`.", false)
        .param("file_glob",     "string",
               "Optional. Wildcard pattern restricting which filenames are "
               "searched (matched against the basename). Examples: "
               "`*.cpp`, `*.{c,h}`, `test_*.py`.", false)
        .param("max_matches",   "integer",
               "Optional. Stop after this many matching lines. Default 100. "
               "Lines past the cap are silently dropped.", false)
        .param("case_insensitive", "boolean",
               "Optional. If true, the regex matches case-insensitively "
               "(adds std::regex::icase). Default false.", false)
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
            if (!sb->inside_sandbox(start)) {
                return ToolResult::error("start dir escapes sandbox via symlink: "
                                         + sb->virtual_path(start));
            }
            // Precheck: a clear `error:` line beats a stray
            // `filesystem_error` exception when the model passes a path
            // that doesn't exist or names a regular file. Without this,
            // `recursive_directory_iterator` throws on construction
            // and the exception escapes the handler.
            std::error_code ec_pre;
            if (!fs::is_directory(start, ec_pre)) {
                return ToolResult::error(
                    std::string("not a directory: ") + sb->virtual_path(start)
                    + (ec_pre ? std::string(" (") + ec_pre.message() + ")"
                              : std::string()));
            }

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
                // Compile defensively — `file_glob` is model-supplied and
                // a stray `[` (or any other regex metachar surviving the
                // escape table) would otherwise throw an uncaught
                // regex_error and tear down the request handler.
                try { glob_rx = std::regex(r); }
                catch (const std::regex_error & e) {
                    return ToolResult::error(std::string("bad file_glob: ") + e.what());
                }
            }

            // Mixed-perms sandboxes (e.g. `--sandbox $HOME` with a
            // ~/.gnupg or ~/.cache somewhere underneath) used to
            // crash grep mid-walk; skip_permission_denied silently
            // steps past unreadable subtrees, and the ec-aware
            // increment below catches any other flakes without
            // tearing down the call.
            constexpr auto kIterOpts =
                fs::directory_options::skip_permission_denied;
            std::error_code ec_it;
            fs::recursive_directory_iterator it(start, kIterOpts, ec_it);
            if (ec_it) {
                return ToolResult::error(
                    std::string("cannot iterate: ") + sb->virtual_path(start)
                    + " (" + ec_it.message() + ")");
            }
            const fs::recursive_directory_iterator end;

            std::ostringstream o;
            int n = 0;
            // Hand-rolled loop using ec-overloads everywhere so a
            // single bad entry (vanished, racy, exotic permission
            // flavor) skips that subtree instead of throwing.
            for (; it != end; ) {
                std::error_code ec_q;
                const bool is_reg = it->is_regular_file(ec_q);
                if (ec_q || !is_reg) goto step;
                {
                    const auto sz = it->file_size(ec_q);
                    if (ec_q || sz > 4 * 1024 * 1024) goto step;
                    std::string fname = it->path().filename().string();
                    if (!file_glob.empty() && !std::regex_match(fname, glob_rx))
                        goto step;
                    std::ifstream f(it->path());
                    if (!f) goto step;
                    std::string line; int lineno = 0;
                    std::string vpath = sb->virtual_path(it->path());
                    while (std::getline(f, line)) {
                        ++lineno;
                        // libstdc++'s regex engine is recursive;
                        // running a user-supplied pattern against a
                        // multi-megabyte single line (binary blob,
                        // minified JS, base64 dump) is a DoS vector
                        // via catastrophic backtracking. 64 KiB is
                        // plenty for source code and short of the
                        // failure regime.
                        if (line.size() > 64 * 1024) continue;
                        if (std::regex_search(line, rx)) {
                            o << vpath << ":" << lineno << ": "
                              << clip(line, 240) << "\n";
                            if (++n >= max_matches) goto done;
                        }
                    }
                }
            step:
                std::error_code ec_step;
                it.increment(ec_step);
                if (ec_step) {
                    // Drop the offending subtree (typically a flaky
                    // entry the iterator can't even step over) and
                    // keep going.
                    if (it == end) break;
                    it.pop();
                    if (it == end) break;
                }
            }
        done:
            if (n == 0) return ToolResult::ok("No matches.");
            return ToolResult::ok(clip(o.str(), 32 * 1024));
        })
        .build();
}

// ============================================================================
// fs_check_path — sandbox-relative stat + access-rights probe.
// ============================================================================
//
// The "look before you leap" tool. Other fs_* tools and bash all tell
// the model (in their descriptions) to call fs_check_path before any
// read/write so the sandbox boundary, file existence, and effective
// r/w/x rights are confirmed up-front instead of discovered through
// open() failures.
//
// Behaviour:
//   - resolve(path) → real fs::path inside the sandbox (same anchoring
//     as every other fs_* tool, so paths can never escape the root);
//   - inside_sandbox() check protects against symlinks-to-outside;
//   - if the path doesn't exist and `touch=true`, create an empty
//     file there (parent dirs auto-created, mode 0600, O_NOFOLLOW so
//     a planted symlink at the leaf can't redirect us out);
//   - stat() the resolved path;
//   - access() with R_OK / W_OK / X_OK for effective rights of the
//     running process — these answer "can I really read/write/exec
//     this *as me*", which is what the model actually wants.
//
// Output is multi-line key:value, with one mandatory `error:` line
// surfaced on failure (consistent with the other fs_* tools).
Tool fs_check_path(std::string root) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("fs_check_path")
        .describe(
            "AUTHORITATIVE PRE-FLIGHT — call this BEFORE any read_file, "
            "write_file, list_dir, glob, grep, or bash command that "
            "touches a path. It confirms the path is inside the "
            "sandbox AND that the running process has the read / write "
            "/ execute rights it needs. Skipping it causes the model to "
            "trip on permission errors mid-task; running it once at the "
            "start of every fs/bash subtask is the contract.\n"
            "\n"
            "Paths are RELATIVE to the sandbox root (e.g. `report.md`, "
            "`src/main.cpp`, `docs`). Call `get_sandbox_path` first if "
            "you need the absolute on-disk root — that path is the "
            "authoritative anchor the fs_*/bash tools resolve against.\n"
            "\n"
            "Required: path. Optional: touch (default false). When "
            "`touch=true` and the path doesn't exist, an empty file is "
            "created there (parent directories auto-created) so you "
            "can probe write access without committing real content.\n"
            "\n"
            "Output is multi-line key:value:\n"
            "  path: <relative path you passed>\n"
            "  absolute: <full on-disk path under the sandbox>\n"
            "  exists: yes|no|unknown    (unknown = can't tell, e.g. parent\n"
            "                             dir blocks lstat with EACCES)\n"
            "  type: file|directory|symlink|other|missing|unknown\n"
            "  size: <bytes>           (files only)\n"
            "  mode: 0NNN              (octal permission bits)\n"
            "  readable: yes|no        (effective R_OK for this process)\n"
            "  writable: yes|no        (effective W_OK for this process)\n"
            "  executable: yes|no      (effective X_OK for this process)\n"
            "  mtime: <ISO-8601 UTC>   (last modification)\n"
            "  created: yes            (only when touch=true created the file)\n"
            "  error: <message>        (only when exists is unknown)\n"
            "\n"
            "Examples:\n"
            "  {path:\"report.md\"}                        # exists? readable? writable?\n"
            "  {path:\"src/main.cpp\"}                     # before read_file / grep\n"
            "  {path:\"build/output.log\", touch:true}     # confirm write access\n"
            "  {path:\".\"}                                # the sandbox root itself")
        .param("path",  "string",
               "Required. Sandbox-relative path. Use `.` for the sandbox root, "
               "`subdir` for a directory, `subdir/file.ext` for a file. Absolute "
               "or `..`-bearing inputs are re-anchored under the sandbox.", true)
        .param("touch", "boolean",
               "Optional. If true and the path does NOT yet exist, create an "
               "empty file at that path (parent dirs auto-created, mode 0600). "
               "Lets you probe write rights without writing real content. "
               "Default false.", false)
        .handle([sb](const ToolCall & c) {
            std::string path; bool touch = false;
            if (!args::get_string(c.arguments_json, "path", path))
                return ToolResult::error("missing arg: path");
            args::get_bool(c.arguments_json, "touch", touch);

            fs::path p; std::string err;
            if (!sb->resolve(path, p, err)) return ToolResult::error(err);
            if (!sb->inside_sandbox(p)) {
                return ToolResult::error("path escapes sandbox via symlink: "
                                         + sb->virtual_path(p));
            }

            // Touch path if requested AND it doesn't exist.
            // Symmetric with fs_write_file: O_NOFOLLOW guards against a
            // last-component symlink, parent dirs are created, and the
            // mode is the same 0600 we use for model-written files.
            bool created = false;
            std::error_code ec_exist;
            const bool exists_pre = fs::exists(p, ec_exist);
            if (touch && !exists_pre) {
                std::error_code ec_mk;
                if (!p.parent_path().empty()) {
                    fs::create_directories(p.parent_path(), ec_mk);
                }
                int fd = ::open(p.c_str(),
                                O_WRONLY | O_CREAT | O_EXCL
                                  | O_NOFOLLOW | O_CLOEXEC,
                                0600);
                if (fd < 0) {
                    return ToolResult::error(
                        std::string("touch failed: ") + sb->virtual_path(p)
                        + " (" + std::strerror(errno) + ")");
                }
                ::close(fd);
                created = true;
            }

            // lstat — we want the truth about the leaf, not whatever a
            // symlink at the leaf points to. inside_sandbox above
            // canonicalised, so we won't mis-report a symlink-to-
            // outside as if it were the target.
            struct stat st {};
            const int    lstat_rc  = ::lstat(p.c_str(), &st);
            const int    lstat_err = (lstat_rc == 0) ? 0 : errno;
            const bool   exists_now = (lstat_rc == 0);

            std::ostringstream o;
            // Echo the relative path the caller passed AND the absolute
            // resolved path. Two anchors so the model can quote either
            // back without recomputing.
            o << "path: "     << path << "\n";
            o << "absolute: " << p.string() << "\n";

            if (!exists_now) {
                // ENOENT really is "no such file"; anything else
                // (typically EACCES on a 0000-perm parent dir) is
                // "we can't tell — the FS won't let us look". Lying
                // and saying `exists: no` for the latter would steer
                // the model into creating a duplicate at a different
                // path or giving up entirely; surface the truth.
                if (lstat_err == ENOENT) {
                    o << "exists: no\n";
                    o << "type: missing\n";
                } else {
                    o << "exists: unknown\n";
                    o << "type: unknown\n";
                    o << "error: lstat: " << std::strerror(lstat_err) << "\n";
                }
                if (touch && !created && lstat_err == ENOENT) {
                    // Touch was requested but the file already existed
                    // pre-call; the missing branch can only be reached
                    // if a concurrent rmdir raced us. Surface that
                    // honestly so the model retries instead of looping.
                    o << "note: vanished between exists check and lstat\n";
                }
                return ToolResult::ok(o.str());
            }

            o << "exists: yes\n";
            const mode_t m = st.st_mode;
            const char * type_str = "other";
            if      (S_ISREG (m)) type_str = "file";
            else if (S_ISDIR (m)) type_str = "directory";
            else if (S_ISLNK (m)) type_str = "symlink";
            else if (S_ISCHR (m)) type_str = "char-device";
            else if (S_ISBLK (m)) type_str = "block-device";
            else if (S_ISFIFO(m)) type_str = "fifo";
            else if (S_ISSOCK(m)) type_str = "socket";
            o << "type: " << type_str << "\n";

            if (S_ISREG(m)) {
                o << "size: " << (long long) st.st_size << "\n";
            }
            // Permission bits — the low 12 bits cover suid/sgid/sticky
            // plus owner/group/other rwx. Mask off file-type bits so
            // the model sees a pure mode.
            char modebuf[8];
            std::snprintf(modebuf, sizeof(modebuf), "0%o",
                          (unsigned) (m & 07777));
            o << "mode: " << modebuf << "\n";

            // Effective access — answers "as me, right now". This is
            // what actually matters for the next read/write call.
            o << "readable: "   << (::access(p.c_str(), R_OK) == 0 ? "yes" : "no") << "\n";
            o << "writable: "   << (::access(p.c_str(), W_OK) == 0 ? "yes" : "no") << "\n";
            o << "executable: " << (::access(p.c_str(), X_OK) == 0 ? "yes" : "no") << "\n";

            // mtime as ISO-8601 UTC. gmtime_r is reentrant; strftime's
            // size cap is comfortably under 32 bytes for this format.
            char tbuf[40] = {0};
            struct tm tmv;
            const time_t mt = (time_t) st.st_mtime;
            if (::gmtime_r(&mt, &tmv) != nullptr) {
                std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
            }
            if (tbuf[0]) o << "mtime: " << tbuf << "\n";

            if (created) o << "created: yes\n";
            return ToolResult::ok(o.str());
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
// reports truthfully. Distinct from get_sandbox_path: this is the
// process cwd (changes if the process chdir'd), get_sandbox_path is the
// pinned sandbox boundary (always the truth).
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
            "Returns the absolute path of the process's current working "
            "directory. In the standard CLI / server setup the process "
            "chdir's into the sandbox at startup, so this typically "
            "matches the sandbox root — but if you specifically want "
            "the sandbox boundary, use `get_sandbox_path` (its answer "
            "is pinned at registration and won't drift). No parameters."
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

// get_sandbox_path — returns the absolute path of the sandbox root.
// Captures the configured root at registration time (so the answer is
// stable even if the process chdir'd later). When `root` is empty or
// ".", we resolve the process's cwd at call time as a fallback — that
// matches what `bash` with no sandbox actually uses.
//
// One error path surfaced as ToolResult::error: getcwd fails when no
// sandbox was configured (path too long, racing rmdir, EACCES on a
// parent component). We surface errno via strerror.
Tool get_sandbox_path(std::string root) {
    // Resolve the configured root to an absolute path NOW, so registration
    // pins the answer. realpath() returns nullptr on transient failures
    // (e.g. relative path with stale cwd) — we keep the original string
    // as a fallback in that case.
    std::string resolved;
    if (!root.empty() && root != ".") {
        char buf[PATH_MAX];
        if (::realpath(root.c_str(), buf) != nullptr) {
            resolved.assign(buf);
        } else {
            resolved = root;
        }
    }
    // Capture by value into the closure; std::string move keeps it cheap.
    return Tool::builder("get_sandbox_path")
        .describe(
            "AUTHORITATIVE SANDBOX ROOT — read this BEFORE anything "
            "else when a task involves the filesystem or shell. The "
            "value is pinned at tool registration and is the single "
            "source of truth for where every fs_* tool resolves its "
            "RELATIVE paths and where `bash` has its cwd. Do NOT "
            "guess, do NOT extrapolate from `get_current_dir`, do "
            "NOT trust prior turns — call this tool fresh whenever "
            "you start touching files, then pair with `fs_check_path` "
            "to confirm read/write rights at your target before "
            "issuing any read/write operation.\n"
            "\n"
            "Returns the absolute filesystem path of your sandbox "
            "root. All filesystem tools (read_file, write_file, "
            "list_dir, glob, grep, fs_check_path) operate inside "
            "this directory; bash runs with this as its cwd. Their "
            "RELATIVE paths anchor here.\n"
            "\n"
            "Use this when you need to mention the real on-disk path "
            "in user-facing output (e.g. \"I created the file at "
            "<sandbox>/main.cpp\") or in commands handed to external "
            "tools that don't share our cwd. For day-to-day file "
            "work, you DON'T paste this path into fs_*/bash arguments "
            "— those expect relative paths.\n"
            "\n"
            "No parameters. Example call: {}."
        )
        .handle([resolved](const ToolCall & /*c*/) {
            if (!resolved.empty()) return ToolResult::ok(resolved);
            // Fallback: no sandbox configured — answer with the
            // process's current cwd, which is what `bash` actually
            // uses when registered with root=".".
            char buf[PATH_MAX];
            if (::getcwd(buf, sizeof(buf)) == nullptr) {
                return ToolResult::error(
                    std::string("getcwd failed: ") + std::strerror(errno));
            }
            const size_t n = ::strnlen(buf, sizeof(buf));
            return ToolResult::ok(std::string(buf, n));
        })
        .build();
}

Tool bash(std::string root, bool show_output) {
    auto sb = std::make_shared<Sandbox>(std::move(root));
    return Tool::builder("bash")
        .describe(
            "Run a shell command via `/bin/sh -c`. Output is stdout and "
            "stderr merged.\n"
            "\n"
            "AUTHORITATIVE SANDBOX RULE — before your first fs_*/bash "
            "call in a task, run `get_sandbox_path` (the absolute on-disk "
            "root, pinned at registration; this is the truth) and then "
            "`fs_check_path` against any file the command will read from "
            "or write to. The command runs with cwd PINNED to the "
            "sandbox root, so the path you probe with `fs_check_path` "
            "is the same path the shell will see. Skipping the precheck "
            "is the most common cause of avoidable error loops.\n"
            "\n"
            "WORKING DIRECTORY & PATHS — cwd is the sandbox root. USE "
            "RELATIVE PATHS in your commands: `ls`, `cat report.md`, "
            "`./build/run`, `src/main.cpp`. Do NOT type absolute paths "
            "for sandbox files — they break portability across "
            "sandboxes. If you need the absolute root for a log line "
            "or an external command, fetch it once via "
            "`get_sandbox_path` and quote that.\n"
            "\n"
            "FILE WORK — PREFER THE DEDICATED TOOLS. For reading, "
            "writing, listing, finding, or searching files use "
            "`read_file`, `write_file`, `list_dir`, `glob`, and `grep`. "
            "They're faster, can't be foot-gunned with quoting / "
            "redirection mistakes, and produce structured output the "
            "model parses reliably. Reach for `bash` only when you need "
            "shell features the dedicated tools don't have:\n"
            "\n"
            "  - pipelines (`grep | xargs`, `find ... -exec`, `... | "
            "    awk ...`)\n"
            "  - process orchestration (running a build, a test "
            "    suite, a script)\n"
            "  - tooling that has no equivalent fs tool (git, package "
            "    managers, `make`, `cmake`, `python` / `node` REPLs, "
            "    diff, file)\n"
            "  - non-trivial in-place edits (sed/awk for line-range "
            "    replacements; the `write_file` tool overwrites or "
            "    appends only)\n"
            "\n"
            "If your command is `cat > file` / `cat <<EOF` / "
            "`echo \"...\" > file` / `mkdir`, call `write_file` "
            "instead — it does the same thing without the shell-quoting "
            "minefield.\n"
            "\n"
            "WARNING: this is NOT a hardened sandbox — the command "
            "runs with the caller's full uid/gid and can read/write "
            "files, hit the network, spawn long-lived processes, etc. "
            "Output is capped at 32 KB and a SIGTERM/SIGKILL deadline "
            "(default 30s, max 300s) bounds the runtime."
        )
        .param("command", "string",
               "Shell command line. Quoted, piped, redirected etc. as you "
               "would type it in a terminal. Use RELATIVE paths under the "
               "sandbox cwd. Example: `ls -la src | head -20`.", true)
        .param("timeout_sec", "integer",
               "Max seconds to run before SIGTERM/SIGKILL. Default 30, max 300.",
               false)
        .handle([sb, show_output](const ToolCall & c) {
            std::string cmd;
            long long timeout_sec = 30;
            if (!args::get_string(c.arguments_json, "command", cmd) || cmd.empty())
                return ToolResult::error("missing arg: command");
            args::get_int(c.arguments_json, "timeout_sec", timeout_sec);
            if (timeout_sec < 1)   timeout_sec = 1;
            if (timeout_sec > 300) timeout_sec = 300;

            // Diagnostic banner: surface the command being run before
            // the first line of output drains. Operator-facing only —
            // the model sees the unchanged ToolResult string.
            if (show_output) {
                std::fprintf(stderr, "\n[bash] $ %s\n", cmd.c_str());
                std::fflush(stderr);
            }

            int pipefd[2];
            if (::pipe(pipefd) < 0)
                return ToolResult::error(std::string("pipe() failed: ") + std::strerror(errno));

            // Cap on the inherited-fd close loop in the child. Mirrors
            // external_tools.cpp's kMaxFdScan: RLIMIT_NOFILE = unlimited
            // wraps to -1 and silently disables a naive loop, leaking
            // every parent fd into the child. 65 536 is more than enough
            // for any well-behaved process.
            constexpr long kMaxFdScan = 65536;

            const std::string cwd = sb->root.string();
            pid_t pid = ::fork();
            if (pid < 0) {
                ::close(pipefd[0]); ::close(pipefd[1]);
                return ToolResult::error(std::string("fork() failed: ") + std::strerror(errno));
            }
            if (pid == 0) {
                // ---- CHILD ----
                // Async-signal-safe operations only until execve.
                ::setpgid(0, 0);   // own process group → kill(-pgid) reaches grandchildren
#if defined(__linux__)
                // Tie our lifetime to the parent: if the agent crashes
                // (segfault, OOM-kill, kill -9) before we exec or while
                // we run, the kernel sends us SIGKILL. Without this an
                // orphaned shell would survive (reparented to PID 1)
                // and keep running until its own timeout.
                ::prctl(PR_SET_PDEATHSIG, SIGKILL);
#endif
                ::close(pipefd[0]);
                ::dup2(pipefd[1], 1);
                ::dup2(pipefd[1], 2);
                ::close(pipefd[1]);

                // stdin → /dev/null. The model has no way to feed
                // bytes into the subprocess, but a shell that reads
                // stdin (e.g. `cat`) would otherwise inherit our
                // controlling-terminal stdin and block.
                int devnull = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
                if (devnull >= 0) {
                    ::dup2(devnull, 0);
                    ::close(devnull);
                } else {
                    ::close(0);
                }

                // Close every other inherited fd. The parent has the
                // HTTP listener, log file, llama.cpp's mmap'd model,
                // etc. None of those should be visible to a /bin/sh
                // command.
                struct rlimit rl{};
                long maxfd = kMaxFdScan;
                if (::getrlimit(RLIMIT_NOFILE, &rl) == 0
                        && rl.rlim_cur != RLIM_INFINITY
                        && rl.rlim_cur > 0
                        && rl.rlim_cur < (rlim_t) kMaxFdScan) {
                    maxfd = (long) rl.rlim_cur;
                }
                for (int fd = 3; fd < (int) maxfd; ++fd) {
                    ::close(fd);
                }

                if (!cwd.empty() && ::chdir(cwd.c_str()) != 0) {
                    const char m[] = "bash: chdir failed\n";
                    ssize_t w = ::write(1, m, sizeof(m) - 1); (void) w;
                    ::_exit(126);
                }
                ::execl("/bin/sh", "sh", "-c", cmd.c_str(), (char *) nullptr);
                const char m[] = "bash: execl failed\n";
                ssize_t w = ::write(1, m, sizeof(m) - 1); (void) w;
                ::_exit(127);
            }
            // ---- PARENT ----
            // Mirror the child's setpgid so kill(-pid) reaches the right
            // group regardless of scheduling order.
            ::setpgid(pid, pid);
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
                    // Live mirror: every byte the child writes (stdout
                    // and stderr were dup2'd onto the same pipe in the
                    // child) goes to the parent's stderr so the operator
                    // can watch a long-running build / test suite scroll
                    // by. The model still receives the full captured
                    // buffer; this is a parallel "human-facing" channel.
                    if (show_output) {
                        ::fwrite(buf, 1, (size_t) n, stderr);
                        std::fflush(stderr);
                    }
                }
            };

            for (;;) {
                auto now = std::chrono::steady_clock::now();
                if (now >= deadline) {
                    if (!sent_term) {
                        // Negative pid in ::kill targets the whole
                        // process group (we set setpgid above) so any
                        // grandchildren the shell spawned receive the
                        // signal too. A bare `kill(pid, …)` on the
                        // shell leader leaves grandchildren behind.
                        ::kill(-pid, SIGTERM);
                        sent_term          = true;
                        killed_for_timeout = true;   // we initiated it
                        deadline           = now + std::chrono::seconds(2); // grace
                    } else {
                        ::kill(-pid, SIGKILL);
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
            // Closing banner mirrors the opening one: the operator sees
            // the exit status and a trailing newline so the next agent
            // line doesn't run into the bash output. Newline first in
            // case the child's last byte wasn't a '\n'.
            if (show_output) {
                std::fprintf(stderr, "\n[bash] %s", oss.str().c_str());
                std::fflush(stderr);
            }
            return ToolResult::ok(oss.str() + body);
        })
        .build();
}

}  // namespace easyai::tools
