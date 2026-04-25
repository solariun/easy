// =============================================================================
//  easyai-server — OpenAI-compatible HTTP server backed by an easyai::Engine.
//
//  Goals
//  -----
//   * Drop-in compatibility with anything that speaks /v1/chat/completions
//     (Claude Code's `--api-base`, OpenAI clients, LiteLLM, LangChain…).
//   * "Competitor" semantics: when the *client* posts its own `system` message
//     and `tools`, those win for that single request — the server-side defaults
//     are only used when the request omits them.
//   * Built-in toolbelt (datetime / web_fetch / web_search / fs_*) auto-
//     dispatched server-side when the client does NOT provide tools.
//   * Friendly preset commands ("precise", "creative 0.9", "/temp 0.5") inside
//     the user message — the server peels them off, applies them, then
//     forwards the rest of the message to the model.
//   * Tiny single-file webui at GET /  with a slider + preset buttons.
//
//  Memory hygiene
//  --------------
//   * One Engine, guarded by std::mutex so requests serialise cleanly across
//     httplib's worker threads (no concurrent llama_decode allowed).
//   * Per-request: history is *replaced*, not appended — no unbounded growth
//     across HTTP calls.
//   * Default tool list is owned by the Server (a vector<Tool> built once),
//     and we never share Tool objects across mutated requests; we copy the
//     vector contents into the engine before each call.
//   * Maximum request body size is clamped (default 8 MiB) so a hostile or
//     buggy client cannot RAM-bomb the process.
//   * Catches std::exception at every HTTP boundary so a malformed request
//     can never tear down the server.
//   * No raw new/delete anywhere in this file.  shared_ptr only where lambdas
//     actually need to extend lifetime past handler return.
// =============================================================================

#include "easyai/easyai.hpp"

#include "httplib.h"          // vendored by llama.cpp
#include "nlohmann/json.hpp"  // vendored by llama.cpp

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using nlohmann::json;
using nlohmann::ordered_json;

// ============================================================================
// Inline webui
// ----------------------------------------------------------------------------
// A self-contained <500-line single-file chat UI. Talks to /v1/chat/completions
// via the OpenAI shape (no streaming for simplicity — it polls one full reply
// per submit).  The preset bar dispatches to /v1/preset which sets server-wide
// defaults for *this UI session only*.
// ============================================================================
constexpr char kWebUI[] = R"HTML(<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__EASYAI_TITLE__</title>
<link rel="icon" href="/favicon">
<style>
:root {
  --bg: #0b0d10;
  --bg-elev: #15191f;
  --bg-input: #0f1318;
  --border: #2a313b;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --accent: #1f6feb;
  --accent-dim: #1f6feb33;
  --tool: #4fb0ff;
  --tool-bg: #11212e;
  --think: #b97df3;
  --think-bg: #1c142a;
  --good: #3fb950;
  --bad: #f85149;
}
* { box-sizing: border-box; }
html, body { margin:0; height:100vh; }
body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif; background:var(--bg); color:var(--text); display:flex; flex-direction:column; }
header { padding:.6rem 1rem; background:var(--bg-elev); border-bottom:1px solid var(--border); display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; }
header h1 { font-size:1rem; margin:0; font-weight:600; letter-spacing:.01em; }
.pill { font-size:.72rem; color:var(--text-dim); padding:.15rem .55rem; border:1px solid var(--border); border-radius:999px; }
.btn { background:transparent; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:.3rem .65rem; cursor:pointer; font-size:.8rem; transition: background .12s; }
.btn:hover { background:var(--bg-input); }
.btn.active { background:var(--accent); border-color:var(--accent); color:#fff; }
.spacer { flex:1; }
.label { font-size:.75rem; color:var(--text-dim); margin: 0 .3rem 0 .6rem; }

#chat { flex:1; overflow-y:auto; padding:1rem 0; }
.msg { max-width: 78ch; margin: 0 auto 1.4rem; padding: 0 1rem; }
.msg .role { font-size:.7rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:.06em; margin-bottom:.25rem; }
.msg.user .body { background:#0d2543; border:1px solid #1f4a77; padding:.55rem .8rem; border-radius:8px; white-space: pre-wrap; word-wrap: break-word; }
.msg.assistant .content { padding:.2rem 0; word-wrap:break-word; overflow-wrap:anywhere; }
.msg.assistant .content p { margin: .3rem 0; }
.msg.assistant .content pre { background:var(--bg-input); border:1px solid var(--border); padding:.6rem .8rem; border-radius:6px; overflow-x:auto; font-size:.85em; line-height:1.45; }
.msg.assistant .content code { background:var(--bg-input); padding:.08rem .35rem; border-radius:3px; font-size:.88em; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.msg.assistant .content pre code { background:transparent; padding:0; font-size:1em; }
.msg.assistant .content a { color:var(--accent); text-decoration:none; }
.msg.assistant .content a:hover { text-decoration:underline; }

.thinking { border:1px solid var(--think-bg); background:var(--think-bg); border-left:3px solid var(--think); border-radius:6px; margin:.45rem 0; padding:.3rem .6rem; font-size:.85em; color:var(--text-dim); }
.thinking summary { cursor:pointer; color:var(--think); font-weight:600; outline:none; user-select:none; }
.thinking summary::marker { color:var(--think); }
.thinking[open] summary { margin-bottom:.4rem; }
.thinking .think-body { white-space:pre-wrap; font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size:.82rem; line-height:1.5; max-height:30em; overflow-y:auto; }

.tool-card { border:1px solid var(--tool-bg); background:var(--tool-bg); border-left:3px solid var(--tool); border-radius:6px; margin:.45rem 0; padding:.4rem .6rem; font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size:.8rem; }
.tool-card.error { border-left-color: var(--bad); background:#22151a; }
.tool-card .head { display:flex; align-items:baseline; gap:.4rem; flex-wrap:wrap; }
.tool-card .name { color:var(--tool); font-weight:600; }
.tool-card.error .name { color:var(--bad); }
.tool-card .args { color:var(--text-dim); word-break:break-all; }
.tool-card .result { color:var(--text); margin-top:.3rem; white-space:pre-wrap; max-height:14em; overflow-y:auto; padding-top:.3rem; border-top:1px dashed var(--border); }
.tool-card .result.pending { color:var(--text-dim); font-style:italic; }

.cursor { display:inline-block; width:.5em; height:1em; background:var(--text); animation: blink 1s steps(1, end) infinite; vertical-align:text-bottom; margin-left:1px; }
@keyframes blink { 50% { opacity: 0; } }

.stats { font-size:.7rem; color:var(--text-dim); margin-top:.3rem; font-family: ui-monospace, "SF Mono", Menlo, monospace; }

form { display:flex; gap:.5rem; padding:.7rem 1rem; background:var(--bg-elev); border-top:1px solid var(--border); }
textarea { flex:1; resize:vertical; min-height:2.4rem; max-height:30vh; padding:.5rem .65rem; background:var(--bg-input); color:var(--text); border:1px solid var(--border); border-radius:6px; font:inherit; }
textarea:focus { outline: 2px solid var(--accent-dim); border-color:var(--accent); }
.send { background:var(--accent); color:#fff; border:0; border-radius:6px; padding: 0 1.1rem; cursor:pointer; font-weight:600; }
.send:disabled { opacity:.6; cursor:wait; }
small.hint { color:#5b626a; padding: 0 1rem .4rem; font-size:.72rem; }
.error-msg { color: var(--bad); background:#22151a; border:1px solid #6a2a31; border-radius:6px; padding:.4rem .6rem; }
</style></head>
<body>
<header>
  <h1>__EASYAI_TITLE__</h1>
  <span class="pill" id="model">loading…</span>
  <span class="pill" id="backend"></span>
  <span class="pill" id="ntools"></span>
  <span class="spacer"></span>
  <span class="label">preset</span>
  <button class="btn" data-p="deterministic">deterministic</button>
  <button class="btn" data-p="precise">precise</button>
  <button class="btn active" data-p="balanced">balanced</button>
  <button class="btn" data-p="creative">creative</button>
  <button class="btn" data-p="wild">wild</button>
  <button class="btn" id="thinkToggle" title="Toggle visibility of <think> blocks">thinking ◐</button>
  <button class="btn" id="reset">reset</button>
</header>
<div id="chat"></div>
<small class="hint">Ctrl/⌘+Enter sends • inline preset works too: <code>creative 0.9 write me a poem about the moon</code></small>
<form id="f">
  <textarea id="t" placeholder="Type a message…" rows="1"></textarea>
  <button class="send" id="send" type="submit">Send</button>
</form>
<script>
"use strict";

// ============================================================================
// helpers
// ============================================================================
const $ = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
const chatEl = $('#chat');
const ta = $('#t');
const sendBtn = $('#send');

function escHTML(s){
  return String(s).replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}

// Tiny markdown subset: code blocks, inline code, bold, italics, links,
// auto-link, line breaks. Run on already-escaped HTML, then mark code-block
// content as "do not touch" before applying inline rules.
function renderMarkdown(s){
  if (!s) return '';
  let html = escHTML(s);
  // We park "atomic" HTML fragments (code blocks, anchors) behind a sentinel
  // so subsequent passes don't re-match the URL/text we already rewrote.
  // Restore happens at the very end.
  const parked = [];
  const park = (frag) => { parked.push(frag); return `\u0001${parked.length-1}\u0001`; };

  // 1. fenced code blocks ```lang\n...\n```
  html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) =>
    park(`<pre><code class="lang-${escHTML(lang)}">${code}</code></pre>`));

  // 2. inline code (single backticks, no newlines)
  html = html.replace(/`([^`\n]+)`/g, (_, c) => park(`<code>${c}</code>`));

  // 3. explicit markdown links: [text](url) — park the resulting <a> so
  //    the auto-link pass below cannot re-match the URL inside its href.
  html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,
    (_, t, u) => park(`<a href="${u}" target="_blank" rel="noopener">${t}</a>`));

  // 4. bold + italics
  html = html.replace(/\*\*([^\*\n]+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/(^|[^\*])\*([^\*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');

  // 5. auto-link bare http(s)://… URLs.  Trailing punctuation .,;:!?) is
  //    excluded so it isn't swallowed into the link.
  html = html.replace(/(\bhttps?:\/\/[^\s<]+?)([.,;:!?)\]]*)(?=\s|$)/g,
    (_, u, tail) => park(`<a href="${u}" target="_blank" rel="noopener">${u}</a>`) + tail);

  // 6. newlines → <br>.  Skip ones inside parked fragments since those are
  //    sentinels (single-line by construction).
  html = html.replace(/\n/g, '<br>');

  // 7. restore everything we parked.
  html = html.replace(/\u0001(\d+)\u0001/g, (_, i) => parked[+i]);
  return html;
}

// ============================================================================
// streaming SSE consumer
// ============================================================================
async function streamChat(messages, handlers){
  const res = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ model:'easyai', messages, stream:true })
  });
  if (!res.ok){
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt.slice(0,400)}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true){
    const {value, done} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream:true});
    while (true){
      const idx = buffer.indexOf('\n\n');
      if (idx === -1) break;
      const event = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      // Parse one SSE event: optional `event:` line + one or more `data:`
      // lines (concatenated by newline).
      let evtType = 'message';
      const dataLines = [];
      for (let line of event.split('\n')){
        if (line.startsWith('\r')) line = line.slice(1);
        if (line.endsWith('\r')) line = line.slice(0, -1);
        if (line.startsWith('event:')) evtType = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /,''));
      }
      const data = dataLines.join('\n');
      if (!data) continue;
      if (data === '[DONE]') return;
      try {
        const j = JSON.parse(data);
        if (evtType === 'easyai.tool_call')   { handlers.onToolCall && handlers.onToolCall(j); continue; }
        if (evtType === 'easyai.tool_result') { handlers.onToolResult && handlers.onToolResult(j); continue; }
        // standard OpenAI delta envelope
        const ch = j?.choices?.[0];
        if (!ch) continue;
        if (ch.delta?.content) handlers.onContent && handlers.onContent(ch.delta.content);
        if (ch.delta?.tool_calls) handlers.onClientToolCalls && handlers.onClientToolCalls(ch.delta.tool_calls);
        if (ch.finish_reason) handlers.onFinish && handlers.onFinish(ch.finish_reason);
        if (j.error) throw new Error(j.error.message || 'server error');
      } catch (e) {
        console.warn('SSE parse', data, e);
      }
    }
  }
}

// ============================================================================
// turn renderer — one assistant message
// ============================================================================
class AssistantTurn {
  constructor(showThinking){
    this.raw = '';
    this.tokens = 0;
    this.startMs = performance.now();
    this.showThinking = showThinking;
    this.toolCards = [];
    this.pendingToolByName = new Map();

    this.el = document.createElement('div');
    this.el.className = 'msg assistant';
    this.el.innerHTML = `
      <div class="role">assistant</div>
      <div class="content"></div>
      <div class="stats"></div>`;
    this.contentEl = $('.content', this.el);
    this.statsEl   = $('.stats', this.el);
    chatEl.appendChild(this.el);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  addContent(piece){
    this.tokens += 1;
    this.raw += piece;
    this.render();
  }

  render(){
    // Split raw into [thinking blocks..., content with tags removed].
    let content = this.raw;
    let thinking = [];
    let openTag = null;
    const re = /<think(?:ing)?>([\s\S]*?)(?:<\/think(?:ing)?>|$)/g;
    let m;
    let cleaned = '';
    let last = 0;
    while ((m = re.exec(this.raw)) !== null){
      cleaned += this.raw.slice(last, m.index);
      thinking.push(m[1]);
      last = m.index + m[0].length;
    }
    cleaned += this.raw.slice(last);
    content = cleaned;

    // (Re)build the thinking block if present.
    let thinkEl = $(':scope > .thinking', this.el);
    if (thinking.length){
      const txt = thinking.join('\n\n— —\n\n');
      if (!thinkEl){
        thinkEl = document.createElement('details');
        thinkEl.className = 'thinking';
        if (this.showThinking) thinkEl.open = true;
        thinkEl.innerHTML = '<summary>Thinking</summary><div class="think-body"></div>';
        // Insert above the content (and above tool cards if any).
        this.el.insertBefore(thinkEl, this.contentEl);
      }
      $('.think-body', thinkEl).textContent = txt;
    }

    // Render markdown content + a blinking cursor while streaming.
    this.contentEl.innerHTML = renderMarkdown(content) +
        (this._done ? '' : '<span class="cursor"></span>');
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  // A tool was just dispatched server-side.
  addToolCall(evt){
    const card = document.createElement('div');
    card.className = 'tool-card';
    let argStr = '';
    try { argStr = JSON.stringify(JSON.parse(evt.arguments)); }
    catch { argStr = evt.arguments || ''; }
    card.innerHTML = `
      <div class="head">
        <span class="name"></span>
        <span class="args"></span>
      </div>
      <div class="result pending">…running…</div>`;
    $('.name', card).textContent = evt.name;
    $('.args', card).textContent = '(' + argStr + ')';
    // Insert cards above the live content area, in chronological order.
    this.el.insertBefore(card, this.contentEl);
    this.toolCards.push(card);
    if (!this.pendingToolByName.has(evt.name)) this.pendingToolByName.set(evt.name, []);
    this.pendingToolByName.get(evt.name).push(card);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  // Pair a tool result with the most-recent unfilled card for that name.
  addToolResult(evt){
    const stack = this.pendingToolByName.get(evt.name);
    if (!stack || !stack.length) return;
    const card = stack.shift();
    const resEl = $('.result', card);
    resEl.classList.remove('pending');
    resEl.textContent = evt.content || '';
    if (evt.is_error) card.classList.add('error');
  }

  // For client-tools mode: a tool_calls array arrived in the OpenAI delta —
  // render them as informational cards with no result (the upstream client,
  // e.g. opencode, would dispatch these).
  addClientToolCalls(tcs){
    for (const tc of tcs){
      if (!tc.function) continue;
      this.addToolCall({
        name: tc.function.name || '?',
        arguments: tc.function.arguments || '{}',
        id: tc.id || ''
      });
      const stack = this.pendingToolByName.get(tc.function.name);
      if (stack && stack.length){
        const card = stack.shift();
        const resEl = $('.result', card);
        resEl.classList.remove('pending');
        resEl.textContent = '(forwarded to client for execution)';
        resEl.style.fontStyle = 'italic';
      }
    }
  }

  finish(reason){
    this._done = true;
    const elapsed = (performance.now() - this.startMs) / 1000;
    const tps = this.tokens / Math.max(elapsed, 0.001);
    this.statsEl.textContent = `${this.tokens} chunks · ${elapsed.toFixed(2)}s · ${tps.toFixed(1)} chunks/s · ${reason}`;
    this.render();
  }

  fail(msg){
    this._done = true;
    this.contentEl.innerHTML = `<div class="error-msg">⚠︎ ${escHTML(msg)}</div>`;
  }

  finalText(){
    // Strip <think>...</think> for sending back to the server in the next turn.
    return this.raw.replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/g, '').trim();
  }
}

// ============================================================================
// app state
// ============================================================================
let history = [];
let showThinking = true;  // expand <think> blocks by default

function addUserMsg(text){
  const m = document.createElement('div');
  m.className = 'msg user';
  m.innerHTML = '<div class="role">you</div><div class="body"></div>';
  $('.body', m).textContent = text;
  chatEl.appendChild(m);
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function send(text){
  addUserMsg(text);
  history.push({role:'user', content:text});
  sendBtn.disabled = true;
  const turn = new AssistantTurn(showThinking);
  try {
    await streamChat(history, {
      onContent:         p => turn.addContent(p),
      onToolCall:        e => turn.addToolCall(e),
      onToolResult:      e => turn.addToolResult(e),
      onClientToolCalls: t => turn.addClientToolCalls(t),
      onFinish:          r => turn.finish(r),
    });
    history.push({role:'assistant', content: turn.finalText()});
  } catch (e) {
    turn.fail(e.message || String(e));
  } finally {
    sendBtn.disabled = false; ta.focus();
  }
}

// ============================================================================
// wiring
// ============================================================================
$('#f').addEventListener('submit', e => {
  e.preventDefault();
  const text = ta.value.trim();
  if (!text || sendBtn.disabled) return;
  ta.value = '';
  send(text);
});
ta.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)){
    e.preventDefault();
    $('#f').requestSubmit();
  }
});
$('#reset').onclick = () => { history = []; chatEl.innerHTML = ''; };
$('#thinkToggle').onclick = e => {
  showThinking = !showThinking;
  e.target.textContent = showThinking ? 'thinking ◐' : 'thinking ○';
  // Toggle existing blocks too.
  $$('details.thinking').forEach(d => d.open = showThinking);
};
$$('header [data-p]').forEach(b => {
  b.onclick = async () => {
    $$('header [data-p]').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    try {
      await fetch('/v1/preset', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({preset: b.dataset.p})
      });
    } catch (e) { console.warn(e); }
  };
});

// status pills
fetch('/health').then(r => r.json()).then(j => {
  $('#model').textContent   = j.model || '?';
  $('#backend').textContent = j.backend ? 'backend: ' + j.backend : '';
  $('#ntools').textContent  = (j.tools ?? 0) + ' tools';
}).catch(()=>{});
</script>
</body></html>
)HTML";

// ============================================================================
// Helpers
// ============================================================================

// ---------------------------------------------------------------------------
// Read a small text file fully into a string. Capped to 1 MiB to avoid the
// "oops --system pointed at /dev/random" footgun.
// ---------------------------------------------------------------------------
static std::string read_text_file(const std::string & path,
                                  size_t max_bytes = 1u << 20) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    if (sz <= 0) return {};
    if ((size_t) sz > max_bytes) sz = (std::streamoff) max_bytes;
    f.seekg(0, std::ios::beg);
    std::string out((size_t) sz, '\0');
    f.read(out.data(), sz);
    out.resize((size_t) f.gcount());
    return out;
}

// ---------------------------------------------------------------------------
// HTML escape — used to keep an operator-supplied --webui-title string from
// breaking out of the <title> / <h1> elements (defense in depth: only the
// admin can set this flag, but better safe than embarrassed).
// ---------------------------------------------------------------------------
static std::string html_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '<':  out += "&lt;";   break;
            case '>':  out += "&gt;";   break;
            case '&':  out += "&amp;";  break;
            case '"':  out += "&quot;"; break;
            case '\'': out += "&#39;";  break;
            default:   out += c;
        }
    }
    return out;
}

// Substitute every occurrence of `from` with `to`. Returns a copy.
static std::string str_replace_all(std::string s,
                                   const std::string & from,
                                   const std::string & to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Read a small binary file fully (icon).  Capped at 256 KiB — anything
// larger is almost certainly a mistake (a webfont or a hi-res splash).
static std::string read_binary_file(const std::string & path,
                                    size_t max_bytes = 256u * 1024u) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    if (sz <= 0) return {};
    if ((size_t) sz > max_bytes) sz = (std::streamoff) max_bytes;
    f.seekg(0, std::ios::beg);
    std::string out((size_t) sz, '\0');
    f.read(out.data(), sz);
    out.resize((size_t) f.gcount());
    return out;
}

// Lower-case the file extension to pick a Content-Type for the favicon.
static std::string mime_for_icon(const std::string & path) {
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return "application/octet-stream";
    std::string ext = path.substr(dot + 1);
    for (auto & c : ext) c = (char) std::tolower((unsigned char) c);
    if (ext == "ico")           return "image/x-icon";
    if (ext == "png")           return "image/png";
    if (ext == "svg")           return "image/svg+xml";
    if (ext == "gif")           return "image/gif";
    if (ext == "jpg" ||
        ext == "jpeg")          return "image/jpeg";
    if (ext == "webp")          return "image/webp";
    return "application/octet-stream";
}

// ---------------------------------------------------------------------------
// Build an OpenAI-compatible error envelope.
// ---------------------------------------------------------------------------
static std::string error_json(const std::string & msg, const char * type = "invalid_request_error") {
    json j;
    j["error"]["message"] = msg;
    j["error"]["type"]    = type;
    return j.dump();
}

// ---------------------------------------------------------------------------
// Serialize a generated turn into an OpenAI chat-completion response.
// ---------------------------------------------------------------------------
static std::string build_chat_response(const std::string & model_id,
                                       const std::string & content,
                                       const std::vector<std::pair<std::string, std::string>> & tool_calls,
                                       const std::vector<std::string> & tool_call_ids,
                                       const std::string & finish_reason) {
    using clock = std::chrono::system_clock;
    ordered_json msg;
    msg["role"] = "assistant";
    msg["content"] = content;
    if (!tool_calls.empty()) {
        ordered_json tcs = json::array();
        for (size_t i = 0; i < tool_calls.size(); ++i) {
            ordered_json tc;
            tc["id"] = i < tool_call_ids.size() && !tool_call_ids[i].empty()
                          ? tool_call_ids[i]
                          : ("call_" + std::to_string(i));
            tc["type"] = "function";
            tc["function"]["name"]      = tool_calls[i].first;
            tc["function"]["arguments"] = tool_calls[i].second;
            tcs.push_back(tc);
        }
        msg["tool_calls"] = tcs;
    }
    ordered_json choice;
    choice["index"]         = 0;
    choice["message"]       = msg;
    choice["finish_reason"] = finish_reason;

    ordered_json env;
    env["id"]      = "chatcmpl-easyai";
    env["object"]  = "chat.completion";
    env["created"] = std::chrono::duration_cast<std::chrono::seconds>(
                        clock::now().time_since_epoch()).count();
    env["model"]   = model_id;
    env["choices"] = json::array({choice});
    env["usage"]   = { {"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0} };
    return env.dump();
}

// ---------------------------------------------------------------------------
// Build a "tool" with a stub handler so the engine can register the schema
// without ever invoking it.  Used when the *client* provides its own tools —
// we only forward tool_calls back to the client, never dispatch them here.
// ---------------------------------------------------------------------------
static easyai::Tool make_stub_tool(const std::string & name,
                                   const std::string & description,
                                   const std::string & params_json) {
    return easyai::Tool::make(name, description, params_json,
        [name](const easyai::ToolCall &) {
            return easyai::ToolResult::error(
                "tool '" + name + "' must be executed by the client");
        });
}

// ============================================================================
// ServerCtx — owns the engine and the HTTP server. RAII & mutexed.
// ============================================================================
struct ServerCtx {
    easyai::Engine             engine;
    std::mutex                 engine_mu;       // protects every engine_* call
    std::vector<easyai::Tool>  default_tools;   // copied into engine per request
    std::string                default_system;
    easyai::Preset             default_preset;  // current "ambient" preset
    std::string                model_id;        // basename of model file
    std::string                api_key;         // empty = auth disabled
    bool                       no_think = false;// strip <think> from responses

    // Webui customisation (built once at start-up so the / handler can just
    // hand back the prebuilt buffer).
    std::string                webui_html;      // HTML with title substituted
    std::string                webui_icon;      // raw icon bytes (empty = none)
    std::string                webui_icon_mime; // "image/x-icon" etc.

    // /metrics counters (atomics so /metrics can read without holding engine_mu)
    std::atomic<uint64_t>      n_requests{0};
    std::atomic<uint64_t>      n_errors{0};
    std::atomic<uint64_t>      n_tool_calls{0};

    // sampling defaults that survive across requests (set via /v1/preset)
    float def_temperature = 0.7f;
    float def_top_p       = 0.95f;
    int   def_top_k       = 40;
    float def_min_p       = 0.05f;

    // ---------------------------------------------------------------------
    // Apply server defaults to the engine. Always called BEFORE handling
    // a request so we start from a known state.
    // ---------------------------------------------------------------------
    void reset_engine_defaults() {
        engine.clear_history();
        engine.system(default_system);
        engine.set_sampling(def_temperature, def_top_p, def_top_k, def_min_p);

        engine.clear_tools();
        for (const auto & t : default_tools) engine.add_tool(t);
    }

    // ---------------------------------------------------------------------
    // Update the *ambient* defaults (used by the webui preset bar).
    // ---------------------------------------------------------------------
    void apply_preset(const easyai::Preset & p) {
        def_temperature = p.temperature;
        def_top_p       = p.top_p;
        def_top_k       = p.top_k;
        def_min_p       = p.min_p;
        default_preset  = p;
    }
};

// ============================================================================
// Request handlers
// ============================================================================

// ---------------------------------------------------------------------------
// Chat-completions request — parsed once, used by both sync and stream paths.
// ---------------------------------------------------------------------------
struct ChatRequest {
    std::vector<std::pair<std::string, std::string>> hist;          // full history
    std::string                                       last_user;    // peeled-off
    bool                                              client_tools = false;
    json                                              tools_blob;   // raw OpenAI tools[]
    double                                            temp_override  = -1.0;
    double                                            top_p_override = -1.0;
    double                                            top_k_override = -1.0;
    bool                                              stream = false;
    easyai::PresetResult                              preset_inline; // applied=true if peeled
};

// Returns false on bad request (and writes the error to `res`).
static bool parse_chat_request(const httplib::Request & req,
                               httplib::Response & res,
                               ChatRequest & out) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception & e) {
        res.status = 400;
        res.set_content(error_json(std::string("invalid JSON: ") + e.what()),
                        "application/json");
        return false;
    }

    if (!body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
        res.status = 400;
        res.set_content(error_json("'messages' is required and must be a non-empty array"),
                        "application/json");
        return false;
    }

    out.hist.reserve(body["messages"].size());
    bool last_is_user = false;
    for (const auto & m : body["messages"]) {
        std::string role = m.value("role", "user");
        std::string content;
        if (m.contains("content") && m["content"].is_string()) {
            content = m["content"].get<std::string>();
        } else if (m.contains("content") && m["content"].is_array()) {
            for (const auto & part : m["content"]) {
                if (part.value("type", "") == "text") content += part.value("text", "");
            }
        }
        out.hist.emplace_back(role, content);
        last_is_user = (role == "user");
        if (last_is_user) out.last_user = content;
    }
    if (!last_is_user) {
        res.status = 400;
        res.set_content(error_json("the final message must have role='user'"),
                        "application/json");
        return false;
    }

    out.client_tools = body.contains("tools") && body["tools"].is_array() &&
                        !body["tools"].empty();
    if (out.client_tools) out.tools_blob = body["tools"];

    auto get_num = [&](const char * k, double dflt) -> double {
        if (body.contains(k) && body[k].is_number()) return body[k].get<double>();
        return dflt;
    };
    out.temp_override  = get_num("temperature", -1.0);
    out.top_p_override = get_num("top_p",       -1.0);
    out.top_k_override = get_num("top_k",       -1.0);
    out.stream         = body.value("stream", false);

    // Inline preset prefix in the last user message ("creative 0.9 …").
    out.preset_inline = easyai::parse_preset(out.last_user);
    if (!out.preset_inline.applied.empty()) {
        out.last_user = out.last_user.substr(out.preset_inline.consumed);
        out.hist.back().second = out.last_user;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Apply request-level overrides + replace history. Caller already holds
// engine_mu and has called reset_engine_defaults().
// ---------------------------------------------------------------------------
static void prepare_engine_for_request(ServerCtx & ctx, const ChatRequest & req) {
    if (req.client_tools) {
        ctx.engine.clear_tools();
        for (const auto & t : req.tools_blob) {
            const json & f = t.contains("function") ? t["function"] : t;
            std::string name        = f.value("name", "");
            std::string description = f.value("description", "");
            std::string params_json = f.contains("parameters")
                                          ? f["parameters"].dump()
                                          : std::string("{\"type\":\"object\",\"properties\":{}}");
            if (name.empty()) continue;
            ctx.engine.add_tool(make_stub_tool(name, description, params_json));
        }
    }
    ctx.engine.set_sampling(
        req.temp_override  >= 0 ? (float) req.temp_override  : -1.0f,
        req.top_p_override >= 0 ? (float) req.top_p_override : -1.0f,
        req.top_k_override >= 0 ? (int)   req.top_k_override : -1,
        -1.0f);
    if (!req.preset_inline.applied.empty()) {
        ctx.engine.set_sampling(req.preset_inline.temperature,
                                 req.preset_inline.top_p,
                                 req.preset_inline.top_k,
                                 req.preset_inline.min_p);
    }
    std::vector<std::pair<std::string, std::string>> hist_minus_last(
        req.hist.begin(), req.hist.end() - 1);
    ctx.engine.replace_history(hist_minus_last);
}

// ---------------------------------------------------------------------------
// Strip <think>…</think> blocks (server-side).  Used for the non-streaming
// reply when --no-think was passed.  The streaming path uses a state-machine
// version inline so it can act per-token.
// ---------------------------------------------------------------------------
static std::string strip_think_blocks(const std::string & content) {
    if (content.empty()) return content;
    std::string out;
    out.reserve(content.size());
    size_t i = 0;
    while (i < content.size()) {
        size_t a = content.find("<think>",    i);
        size_t b = content.find("<thinking>", i);
        size_t open = std::min(a == std::string::npos ? std::string::npos : a,
                               b == std::string::npos ? std::string::npos : b);
        if (open == std::string::npos) { out.append(content, i, std::string::npos); break; }
        out.append(content, i, open - i);
        size_t ca = content.find("</think>",    open);
        size_t cb = content.find("</thinking>", open);
        size_t close = std::min(ca == std::string::npos ? std::string::npos : ca,
                                cb == std::string::npos ? std::string::npos : cb);
        if (close == std::string::npos) break;
        size_t close_end = content.find('>', close) + 1;
        i = close_end;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Non-streaming path — kept exactly as before. Caller guarantees ctx.engine_mu
// is held.
// ---------------------------------------------------------------------------
static void handle_chat_sync(ServerCtx & ctx, ChatRequest & req,
                             httplib::Response & res) {
    std::string content;
    std::vector<std::pair<std::string, std::string>> tool_calls;
    std::vector<std::string> tool_call_ids;
    std::string finish_reason = "stop";

    try {
        if (req.client_tools) {
            ctx.engine.push_message("user", req.last_user);
            auto turn = ctx.engine.generate_one();
            content       = std::move(turn.content);
            tool_calls    = std::move(turn.tool_calls);
            tool_call_ids = std::move(turn.tool_call_ids);
            finish_reason = std::move(turn.finish_reason);
        } else {
            content = ctx.engine.chat(req.last_user);
        }
    } catch (const std::exception & e) {
        ctx.n_errors.fetch_add(1, std::memory_order_relaxed);
        res.status = 500;
        res.set_content(error_json(std::string("engine error: ") + e.what(),
                                   "internal_error"),
                        "application/json");
        return;
    }
    if (!tool_calls.empty()) ctx.n_tool_calls.fetch_add(tool_calls.size(),
                                                        std::memory_order_relaxed);
    if (ctx.no_think) content = strip_think_blocks(content);

    res.status = 200;
    res.set_content(build_chat_response(ctx.model_id, content,
                                         tool_calls, tool_call_ids,
                                         finish_reason),
                     "application/json");
}

// ---------------------------------------------------------------------------
// Streaming path (Server-Sent Events).
//
// Standard OpenAI delta envelope is emitted for each generated piece:
//
//   data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}
//
// Plus two custom event types so the embedded webui can render tool activity
// inline:
//
//   event: easyai.tool_call
//   data: {"name":"web_search","arguments":"{...}","id":"call_1"}
//
//   event: easyai.tool_result
//   data: {"name":"web_search","content":"...","is_error":false}
//
// Generic OpenAI clients ignore unknown event types, so this is additive.
//
// All engine work happens INSIDE the chunked-content lambda so we can hold
// the engine mutex from there.  shared_ptr<ChatRequest> keeps the parsed
// state alive past `route_chat_completions`'s return.
// ---------------------------------------------------------------------------
static void handle_chat_stream(ServerCtx & ctx,
                               std::shared_ptr<ChatRequest> req_state,
                               httplib::Response & res) {
    res.set_header("Cache-Control",     "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    res.set_chunked_content_provider("text/event-stream",
        [&ctx, req_state](size_t /*offset*/, httplib::DataSink & sink) -> bool {
            std::lock_guard<std::mutex> lock(ctx.engine_mu);

            ctx.reset_engine_defaults();
            prepare_engine_for_request(ctx, *req_state);

            // ---- emit helpers --------------------------------------------
            auto emit_data = [&sink](const std::string & ev) {
                std::string s = "data: " + ev + "\n\n";
                sink.write(s.data(), s.size());
            };
            auto emit_event = [&sink](const std::string & evt_type, const std::string & ev) {
                std::string s = "event: " + evt_type + "\ndata: " + ev + "\n\n";
                sink.write(s.data(), s.size());
            };

            // ---- streaming tag stripper --------------------------------
            // Always suppresses <tool_call>…</tool_call> blocks — those are
            // protocol artefacts, and the webui already renders the call as
            // a card via easyai.tool_call/result events.
            //
            // Optionally also suppresses <think>…</think> when the operator
            // passed --no-think.
            //
            // The state machine keeps a small buffer (longest tag length)
            // so a tag split across token chunks still matches.
            struct StripState {
                bool        strip_think = false;   // gated by --no-think
                bool        in_drop = false;       // currently inside a stripped span
                const char *close_needle  = nullptr; // which close tag to look for
                std::string buf;

                // Detect any open tag we care about at index i in buf.
                // Returns {pos_of_open, pos_after_open, close_needle} or pos==npos.
                struct Open { size_t pos = std::string::npos; size_t after = 0; const char *close = nullptr; };
                Open find_open() const {
                    Open best{};
                    auto consider = [&](const char * open, const char * close) {
                        size_t p = buf.find(open);
                        if (p == std::string::npos) return;
                        if (best.pos == std::string::npos || p < best.pos) {
                            best.pos   = p;
                            best.after = p + std::strlen(open);
                            best.close = close;
                        }
                    };
                    consider("<tool_call>", "</tool_call>");
                    if (strip_think) {
                        consider("<think>",    "</think>");
                        consider("<thinking>", "</thinking>");
                    }
                    return best;
                }

                std::string filter(std::string piece) {
                    buf += piece;
                    std::string out;
                    // Longest open we care about is "<thinking>" (10).
                    // Longest close is "</thinking>" (11).  Keep that many
                    // trailing chars unflushed so a partial tag can complete.
                    constexpr size_t open_margin  = 10;
                    constexpr size_t close_margin = 12;
                    for (;;) {
                        if (in_drop) {
                            size_t p = buf.find(close_needle);
                            if (p == std::string::npos) {
                                if (buf.size() > close_margin)
                                    buf.erase(0, buf.size() - close_margin);
                                return out;
                            }
                            buf.erase(0, p + std::strlen(close_needle));
                            in_drop = false;
                            close_needle = nullptr;
                        } else {
                            Open o = find_open();
                            if (o.pos == std::string::npos) {
                                size_t safe = buf.size() > open_margin
                                                 ? buf.size() - open_margin : 0;
                                if (safe > 0) {
                                    out.append(buf, 0, safe);
                                    buf.erase(0, safe);
                                }
                                return out;
                            }
                            out.append(buf, 0, o.pos);
                            buf.erase(0, o.after);
                            in_drop = true;
                            close_needle = o.close;
                        }
                    }
                }
                std::string flush() {
                    std::string out;
                    if (!in_drop) out = std::move(buf);
                    buf.clear();
                    in_drop = false;
                    close_needle = nullptr;
                    return out;
                }
            } strip;
            strip.strip_think = ctx.no_think;

            // ---- on_token callback: stream OpenAI-shape deltas ----------
            ctx.engine.on_token([&](const std::string & piece) {
                std::string visible = strip.filter(piece);
                if (visible.empty()) return;
                ordered_json delta;
                delta["choices"] = json::array({{
                    {"index", 0},
                    {"delta", {{"content", visible}}},
                    {"finish_reason", nullptr},
                }});
                emit_data(delta.dump());
            });

            // ---- on_tool callback: custom events for the webui ----------
            ctx.engine.on_tool([&](const easyai::ToolCall & c, const easyai::ToolResult & r) {
                ctx.n_tool_calls.fetch_add(1, std::memory_order_relaxed);
                ordered_json call_evt;
                call_evt["name"]      = c.name;
                call_evt["arguments"] = c.arguments_json;
                call_evt["id"]        = c.id;
                emit_event("easyai.tool_call", call_evt.dump());

                ordered_json res_evt;
                res_evt["name"]     = c.name;
                res_evt["content"]  = r.content;
                res_evt["is_error"] = r.is_error;
                emit_event("easyai.tool_result", res_evt.dump());
            });

            // ---- run the engine ----------------------------------------
            std::string finish_reason = "stop";
            std::vector<std::pair<std::string, std::string>> tool_calls;
            std::vector<std::string> tool_call_ids;
            try {
                if (req_state->client_tools) {
                    ctx.engine.push_message("user", req_state->last_user);
                    auto turn = ctx.engine.generate_one();
                    tool_calls    = std::move(turn.tool_calls);
                    tool_call_ids = std::move(turn.tool_call_ids);
                    finish_reason = turn.tool_calls.empty() ? "stop" : "tool_calls";
                } else {
                    ctx.engine.chat(req_state->last_user);
                }
            } catch (const std::exception & e) {
                ctx.n_errors.fetch_add(1, std::memory_order_relaxed);
                ordered_json err;
                err["error"] = { {"message", e.what()}, {"type", "internal_error"} };
                emit_data(err.dump());
            }

            // Drain whatever the strip state machine was holding.
            std::string tail = strip.flush();
            if (!tail.empty()) {
                ordered_json delta;
                delta["choices"] = json::array({{
                    {"index", 0},
                    {"delta", {{"content", tail}}},
                    {"finish_reason", nullptr},
                }});
                emit_data(delta.dump());
            }

            // For client-tools mode, emit the assembled tool_calls array as
            // a single delta so OpenAI clients (and our webui) see them.
            if (!tool_calls.empty()) {
                ordered_json tc_arr = json::array();
                for (size_t i = 0; i < tool_calls.size(); ++i) {
                    ordered_json tc;
                    tc["index"]    = (int) i;
                    tc["id"]       = i < tool_call_ids.size() && !tool_call_ids[i].empty()
                                         ? tool_call_ids[i]
                                         : ("call_" + std::to_string(i));
                    tc["type"]     = "function";
                    tc["function"] = { {"name", tool_calls[i].first},
                                       {"arguments", tool_calls[i].second} };
                    tc_arr.push_back(tc);
                }
                ordered_json delta;
                delta["choices"] = json::array({{
                    {"index", 0},
                    {"delta", {{"tool_calls", tc_arr}}},
                    {"finish_reason", nullptr},
                }});
                emit_data(delta.dump());
            }

            // Final close-out chunk — empty delta + finish_reason.
            ordered_json done_delta;
            done_delta["choices"] = json::array({{
                {"index", 0},
                {"delta", json::object()},
                {"finish_reason", finish_reason},
            }});
            emit_data(done_delta.dump());
            emit_data("[DONE]");

            sink.done();
            return true;
        });
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions — dispatch to sync or stream path.
// ---------------------------------------------------------------------------
static void route_chat_completions(ServerCtx & ctx, const httplib::Request & req,
                                   httplib::Response & res) {
    auto state = std::make_shared<ChatRequest>();
    if (!parse_chat_request(req, res, *state)) return;

    ctx.n_requests.fetch_add(1, std::memory_order_relaxed);

    if (state->stream) {
        // Streaming path: lock + engine work happen inside the chunked
        // content provider lambda (which runs after this function returns).
        handle_chat_stream(ctx, state, res);
    } else {
        std::lock_guard<std::mutex> lock(ctx.engine_mu);
        ctx.reset_engine_defaults();
        prepare_engine_for_request(ctx, *state);
        handle_chat_sync(ctx, *state, res);
    }
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------
static void route_models(ServerCtx & ctx, const httplib::Request &,
                         httplib::Response & res) {
    ordered_json env;
    env["object"] = "list";
    env["data"]   = json::array({{
        {"id",      ctx.model_id},
        {"object",  "model"},
        {"created", 0},
        {"owned_by","easyai"},
    }});
    res.set_content(env.dump(), "application/json");
}

// ---------------------------------------------------------------------------
// POST /v1/preset  body: {"preset": "creative"}  → ambient defaults change.
// ---------------------------------------------------------------------------
static void route_preset(ServerCtx & ctx, const httplib::Request & req,
                         httplib::Response & res) {
    try {
        json b = json::parse(req.body);
        std::string name = b.value("preset", "balanced");
        const easyai::Preset * p = easyai::find_preset(name);
        if (!p) {
            res.status = 400;
            res.set_content(error_json("unknown preset: " + name), "application/json");
            return;
        }
        std::lock_guard<std::mutex> lock(ctx.engine_mu);
        ctx.apply_preset(*p);
        res.set_content(json{{"applied", p->name}}.dump(), "application/json");
    } catch (const std::exception & e) {
        res.status = 400;
        res.set_content(error_json(std::string("bad request: ") + e.what()),
                        "application/json");
    }
}

// ---------------------------------------------------------------------------
// GET /health   →  small status JSON.
// ---------------------------------------------------------------------------
static void route_health(ServerCtx & ctx, const httplib::Request &,
                         httplib::Response & res) {
    ordered_json j;
    j["status"]  = "ok";
    j["model"]   = ctx.model_id;
    j["backend"] = ctx.engine.backend_summary();
    j["tools"]   = ctx.default_tools.size();
    j["preset"]  = ctx.default_preset.name;
    res.set_content(j.dump(), "application/json");
}

// ---------------------------------------------------------------------------
// GET /metrics  →  Prometheus-style text exposition.
// Always available when --metrics was passed at start-up.
// ---------------------------------------------------------------------------
static void route_metrics(ServerCtx & ctx, const httplib::Request &,
                          httplib::Response & res) {
    std::ostringstream o;
    o << "# HELP easyai_requests_total Total /v1/chat/completions requests received.\n"
      << "# TYPE easyai_requests_total counter\n"
      << "easyai_requests_total " << ctx.n_requests.load() << "\n"
      << "# HELP easyai_errors_total Total chat-completion handler errors.\n"
      << "# TYPE easyai_errors_total counter\n"
      << "easyai_errors_total " << ctx.n_errors.load() << "\n"
      << "# HELP easyai_tool_calls_total Total tool_calls returned to clients.\n"
      << "# TYPE easyai_tool_calls_total counter\n"
      << "easyai_tool_calls_total " << ctx.n_tool_calls.load() << "\n";
    res.set_content(o.str(), "text/plain; version=0.0.4");
}

// ---------------------------------------------------------------------------
// Bearer-token auth check.  Returns true to continue; on failure the response
// is filled with 401 and the caller should bail.
// ---------------------------------------------------------------------------
static bool require_auth(const ServerCtx & ctx, const httplib::Request & req,
                         httplib::Response & res) {
    if (ctx.api_key.empty()) return true;  // auth disabled
    auto it = req.headers.find("Authorization");
    std::string expected = "Bearer " + ctx.api_key;
    if (it == req.headers.end() || it->second != expected) {
        res.status = 401;
        res.set_content(error_json("missing or invalid Bearer token",
                                   "authentication_error"),
                        "application/json");
        return false;
    }
    return true;
}

// ============================================================================
// main
// ============================================================================

[[noreturn]] static void die_usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s -m model.gguf [options]\n\n"
        "Required:\n"
        "  -m, --model <path>           GGUF model file\n"
        "\nNetwork:\n"
        "      --host <addr>            Bind address (default 127.0.0.1).\n"
        "                                Use 0.0.0.0 to listen on every IPv4\n"
        "                                interface (LAN, docker, etc.).\n"
        "      --port <n>               TCP port (default 8080)\n"
        "      --max-body <bytes>       Max request body size (default 8 MiB)\n"
        "\nDefault system prompt + tools:\n"
        "  -s, --system-file <path>     Server-default system prompt from file\n"
        "      --system <text>          Inline system prompt\n"
        "      --no-tools               Don't expose the built-in toolbelt\n"
        "      --sandbox <dir>          Root for fs_* tools (default '.')\n"
        "\nModel tuning (apply on top of --preset):\n"
        "      --preset <name>          Ambient preset (default 'balanced')\n"
        "      --temperature <f>        Override temperature (0.0-2.0)\n"
        "      --top-p <f>              Override nucleus sampling p\n"
        "      --top-k <n>              Override top-k\n"
        "      --min-p <f>              Override min-p\n"
        "      --repeat-penalty <f>     Override repeat penalty\n"
        "      --max-tokens <n>         Cap tokens generated per request\n"
        "      --seed <u32>             RNG seed (0 = random)\n"
        "\nCompute / memory:\n"
        "  -c, --ctx <n>                Context size (default 8192)\n"
        "      --batch <n>              Logical batch size (default = ctx)\n"
        "      --ngl <n>                GPU layers (-1=auto, 0=CPU)\n"
        "  -t, --threads <n>            CPU threads\n"
        "\nKV cache (all optional):\n"
        " -ctk, --cache-type-k <type>   K-cache dtype (f32|f16|bf16|q8_0|q4_0|q4_1|q5_0|q5_1|iq4_nl)\n"
        " -ctv, --cache-type-v <type>   V-cache dtype (same options)\n"
        "-nkvo, --no-kv-offload         Keep KV cache on CPU even with GPU layers\n"
        "      --kv-unified             Use a single unified KV buffer across sequences\n"
        "      --override-kv <k=t:v>    Override a GGUF metadata entry (repeatable)\n"
        "                                Types: int|float|bool|str\n"
        "\nllama-server compatibility:\n"
        "  -a,  --alias <name>          Public model id reported by /v1/models\n"
        "       --api-key <key>         Require Bearer auth on every /v1 route\n"
        "  -fa, --flash-attn            Force flash attention on\n"
        "  -tb, --threads-batch <n>     Threads used for prompt-eval batches\n"
        "  -np, --parallel <n>          Accepted for compat; warns when >1\n"
        "       --mlock                 mlock model weights into RAM\n"
        "       --no-mmap               Disable mmap (read GGUF straight in)\n"
        "       --numa <strategy>       distribute|isolate|numactl|mirror\n"
        "       --metrics               Expose Prometheus /metrics endpoint\n"
        "       --reasoning <on|off>    Enable model thinking (default on)\n"
        "       --no-think              Strip <think>...</think> from replies\n"
        "\nWebui rebrand:\n"
        "       --webui-title <text>    Title shown in the browser tab and the\n"
        "                                <h1> at the top of the page (default 'easyai')\n"
        "       --webui-icon <path>     Favicon file (.ico|.png|.svg|.gif|.jpg|.webp)\n"
        "                                served at /favicon and /favicon.ico\n"
        "\n  -h, --help                   Show this help and exit\n",
        argv0);
    std::exit(1);
}

struct ServerArgs {
    std::string model_path;
    std::string system_path;
    std::string system_inline;
    std::string host       = "127.0.0.1";   // pass 0.0.0.0 for any-iface
    int         port       = 8080;
    int         n_ctx      = 8192;
    int         n_batch    = 0;             // 0 = follow ctx
    int         ngl        = -1;
    int         n_threads  = 0;
    bool        load_tools = true;
    std::string sandbox    = ".";
    std::string preset     = "balanced";
    size_t      max_body   = 8u * 1024u * 1024u;

    // sampling overrides (apply on top of preset)
    float       temperature    = -1.0f;
    float       top_p          = -1.0f;
    int         top_k          = -1;
    float       min_p          = -1.0f;
    float       repeat_penalty = -1.0f;
    int         max_tokens     = -1;
    uint32_t    seed           = 0u;

    // KV cache controls
    std::string cache_type_k;
    std::string cache_type_v;
    bool        no_kv_offload = false;
    bool        kv_unified    = false;
    std::vector<std::string> kv_overrides;

    // llama-server compatibility / production knobs
    std::string alias;            // exposed as model id in /v1/models
    std::string api_key;          // Bearer token required if set
    std::string numa;             // distribute|isolate|numactl|mirror
    int         threads_batch  = 0;
    int         parallel       = 1;
    bool        flash_attn     = false;
    bool        mlock          = false;
    bool        no_mmap        = false;
    bool        metrics        = false;
    bool        reasoning      = true;   // enable_thinking flag default ON
    bool        no_think       = false;  // strip <think>…</think> from /v1 responses

    // webui rebrand
    std::string webui_title    = "easyai";
    std::string webui_icon;              // optional path to .ico/.png/.svg
};

static ServerArgs parse_args(int argc, char ** argv) {
    ServerArgs a;
    auto need = [&](int & i, const char * flag) -> const char * {
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); die_usage(argv[0]); }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "-m" || s == "--model")        a.model_path     = need(i, "-m");
        else if (s == "-s" || s == "--system-file")  a.system_path    = need(i, "-s");
        else if (s == "--system")                    a.system_inline  = need(i, "--system");
        else if (s == "--host")                      a.host           = need(i, "--host");
        else if (s == "--port")                      a.port           = std::atoi(need(i, "--port"));
        else if (s == "-c" || s == "--ctx")          a.n_ctx          = std::atoi(need(i, "-c"));
        else if (s == "--ngl")                       a.ngl            = std::atoi(need(i, "--ngl"));
        else if (s == "-t" || s == "--threads")      a.n_threads      = std::atoi(need(i, "-t"));
        else if (s == "--no-tools")                  a.load_tools     = false;
        else if (s == "--sandbox")                   a.sandbox        = need(i, "--sandbox");
        else if (s == "--preset")                    a.preset         = need(i, "--preset");
        else if (s == "--max-body")                  a.max_body       = (size_t) std::atoll(need(i, "--max-body"));
        else if (s == "--batch")                     a.n_batch        = std::atoi(need(i, "--batch"));
        // sampling overrides
        else if (s == "--temperature" || s == "--temp") a.temperature  = std::atof(need(i, "--temperature"));
        else if (s == "--top-p")                     a.top_p          = std::atof(need(i, "--top-p"));
        else if (s == "--top-k")                     a.top_k          = std::atoi(need(i, "--top-k"));
        else if (s == "--min-p")                     a.min_p          = std::atof(need(i, "--min-p"));
        else if (s == "--repeat-penalty")            a.repeat_penalty = std::atof(need(i, "--repeat-penalty"));
        else if (s == "--max-tokens")                a.max_tokens     = std::atoi(need(i, "--max-tokens"));
        else if (s == "--seed")                      a.seed           = (uint32_t) std::strtoul(need(i, "--seed"), nullptr, 10);
        // KV cache
        else if (s == "-ctk" || s == "--cache-type-k") a.cache_type_k  = need(i, "-ctk");
        else if (s == "-ctv" || s == "--cache-type-v") a.cache_type_v  = need(i, "-ctv");
        else if (s == "-nkvo" || s == "--no-kv-offload") a.no_kv_offload = true;
        else if (s == "--kv-unified")                a.kv_unified     = true;
        else if (s == "--override-kv")               a.kv_overrides.push_back(need(i, "--override-kv"));
        // llama-server compat
        else if (s == "-a"  || s == "--alias")       a.alias          = need(i, "-a");
        else if (s == "--api-key")                   a.api_key        = need(i, "--api-key");
        else if (s == "--numa")                      a.numa           = need(i, "--numa");
        else if (s == "-tb" || s == "--threads-batch") a.threads_batch = std::atoi(need(i, "-tb"));
        else if (s == "-np" || s == "--parallel")    a.parallel       = std::atoi(need(i, "-np"));
        else if (s == "-fa" || s == "--flash-attn")  a.flash_attn     = true;
        else if (s == "--mlock")                     a.mlock          = true;
        else if (s == "--no-mmap")                   a.no_mmap        = true;
        else if (s == "--metrics")                   a.metrics        = true;
        else if (s == "--reasoning") {
            std::string v = need(i, "--reasoning");
            a.reasoning = (v == "on" || v == "1" || v == "true" || v == "yes");
        }
        else if (s == "--no-think")                  a.no_think       = true;
        else if (s == "--webui-title")               a.webui_title    = need(i, "--webui-title");
        else if (s == "--webui-icon")                a.webui_icon     = need(i, "--webui-icon");
        else if (s == "-h" || s == "--help")         die_usage(argv[0]);
        else { std::fprintf(stderr, "unknown arg: %s\n", s.c_str()); die_usage(argv[0]); }
    }
    if (a.model_path.empty()) die_usage(argv[0]);
    return a;
}

// Graceful shutdown — flag set by SIGINT/SIGTERM, polled by main loop.
static std::atomic<httplib::Server *> g_server{nullptr};
static void on_signal(int) {
    httplib::Server * s = g_server.load();
    if (s) s->stop();
}

int main(int argc, char ** argv) {
    ServerArgs args = parse_args(argc, argv);

    // -------- resolve system prompt --------------------------------------
    // Precedence: --system inline > -s file > built-in default. The default
    // exists because a *small* model with NO system prompt and a tool list
    // is very likely to over-eagerly call tools on simple greetings ("hi"
    // → web_search).  Operators can fully replace it via -s.
    static constexpr char kBuiltinSystem[] =
        "You are a helpful, concise assistant.\n"
        "Answer directly when you can — for greetings, chitchat, basic facts, "
        "math, and anything in your training data, just respond.\n"
        "Use a tool ONLY when the request truly needs one:\n"
        "  - up-to-date / time-sensitive info → web_search, then web_fetch\n"
        "  - the current date/time            → datetime\n"
        "  - reading or listing files         → fs_read_file / fs_list_dir / fs_glob / fs_grep\n"
        "Never call a tool just to look busy.  When you do call one, cite the result.";

    std::string default_system = args.system_inline;
    if (default_system.empty() && !args.system_path.empty()) {
        default_system = read_text_file(args.system_path);
        if (default_system.empty()) {
            std::fprintf(stderr, "[easyai-server] WARN: failed to read system file '%s'\n",
                         args.system_path.c_str());
        }
    }
    if (default_system.empty()) default_system = kBuiltinSystem;

    // -------- build the context (heap-allocated so HTTP lambdas can capture
    // a stable reference; lifetime tied to main()'s scope via unique_ptr).
    auto ctx = std::make_unique<ServerCtx>();
    ctx->default_system = default_system;
    {
        // Compute model_id = basename(path) without extension.
        std::string p = args.model_path;
        auto slash = p.find_last_of("/\\");
        if (slash != std::string::npos) p = p.substr(slash + 1);
        auto dot = p.find_last_of('.');
        if (dot != std::string::npos) p = p.substr(0, dot);
        ctx->model_id = p;
    }

    // Start with ambient preset, then overlay any explicit numeric overrides.
    {
        easyai::Preset base = *easyai::find_preset("balanced");
        if (const easyai::Preset * pp = easyai::find_preset(args.preset)) base = *pp;
        if (args.temperature >= 0) base.temperature = args.temperature;
        if (args.top_p       >= 0) base.top_p       = args.top_p;
        if (args.top_k       >= 0) base.top_k       = args.top_k;
        if (args.min_p       >= 0) base.min_p       = args.min_p;
        ctx->apply_preset(base);
    }

    // Default toolbelt — opt-out via --no-tools.
    if (args.load_tools) {
        ctx->default_tools.push_back(easyai::tools::datetime());
        ctx->default_tools.push_back(easyai::tools::web_fetch());
        ctx->default_tools.push_back(easyai::tools::web_search());
        ctx->default_tools.push_back(easyai::tools::fs_list_dir (args.sandbox));
        ctx->default_tools.push_back(easyai::tools::fs_read_file(args.sandbox));
        ctx->default_tools.push_back(easyai::tools::fs_glob     (args.sandbox));
        ctx->default_tools.push_back(easyai::tools::fs_grep     (args.sandbox));
    }

    // -------- production knobs / auth --------------------------------------
    ctx->api_key  = args.api_key;
    ctx->no_think = args.no_think;
    if (!args.alias.empty()) ctx->model_id = args.alias;

    // ----- webui rebrand: substitute the title placeholder once and load
    //       the optional favicon into memory.
    {
        std::string title = args.webui_title.empty() ? std::string("easyai")
                                                     : args.webui_title;
        ctx->webui_html = str_replace_all(kWebUI, "__EASYAI_TITLE__",
                                          html_escape(title));
        if (!args.webui_icon.empty()) {
            ctx->webui_icon = read_binary_file(args.webui_icon);
            if (ctx->webui_icon.empty()) {
                std::fprintf(stderr,
                    "[easyai-server] WARN: failed to read --webui-icon '%s'\n",
                    args.webui_icon.c_str());
            } else {
                ctx->webui_icon_mime = mime_for_icon(args.webui_icon);
                std::fprintf(stderr,
                    "[easyai-server] webui icon loaded: %s (%zu bytes, %s)\n",
                    args.webui_icon.c_str(), ctx->webui_icon.size(),
                    ctx->webui_icon_mime.c_str());
            }
        }
    }

    if (args.parallel > 1) {
        std::fprintf(stderr,
            "[easyai-server] note: --parallel %d requested but the engine is "
            "single-context; requests are still serialised.\n", args.parallel);
    }

    // -------- configure & load engine ------------------------------------
    ctx->engine.model      (args.model_path)
               .context    (args.n_ctx)
               .gpu_layers (args.ngl)
               .system     (default_system)
               .verbose    (false);
    if (args.n_threads > 0)     ctx->engine.threads(args.n_threads);
    if (args.threads_batch > 0) ctx->engine.threads_batch(args.threads_batch);
    if (args.n_batch   > 0)  ctx->engine.batch  (args.n_batch);
    if (args.seed      > 0)  ctx->engine.seed   (args.seed);
    if (args.max_tokens >= 0) ctx->engine.max_tokens(args.max_tokens);
    if (args.repeat_penalty > 0) ctx->engine.repeat_penalty(args.repeat_penalty);
    if (!args.cache_type_k.empty()) ctx->engine.cache_type_k(args.cache_type_k);
    if (!args.cache_type_v.empty()) ctx->engine.cache_type_v(args.cache_type_v);
    if (args.no_kv_offload)  ctx->engine.no_kv_offload(true);
    if (args.kv_unified)     ctx->engine.kv_unified(true);
    if (args.flash_attn)     ctx->engine.flash_attn(true);
    if (args.mlock)          ctx->engine.use_mlock(true);
    if (args.no_mmap)        ctx->engine.use_mmap(false);
    if (!args.numa.empty())  ctx->engine.numa(args.numa);
    if (!args.reasoning)     ctx->engine.enable_thinking(false);
    for (const auto & ov : args.kv_overrides) ctx->engine.add_kv_override(ov);
    for (const auto & t : ctx->default_tools) ctx->engine.add_tool(t);
    ctx->engine.set_sampling(ctx->def_temperature, ctx->def_top_p,
                             ctx->def_top_k, ctx->def_min_p);

    if (!ctx->engine.load()) {
        std::fprintf(stderr, "[easyai-server] load failed: %s\n",
                     ctx->engine.last_error().c_str());
        return 1;
    }

    std::fprintf(stderr,
        "[easyai-server] %s loaded\n"
        "                backend=%s  ctx=%d  tools=%zu  preset=%s\n"
        "                listening on http://%s:%d  (webui at /)\n",
        ctx->model_id.c_str(), ctx->engine.backend_summary().c_str(),
        ctx->engine.n_ctx(), ctx->default_tools.size(),
        ctx->default_preset.name.c_str(),
        args.host.c_str(), args.port);

    // -------- http server -------------------------------------------------
    httplib::Server svr;
    svr.set_payload_max_length(args.max_body);
    svr.set_read_timeout (60);
    svr.set_write_timeout(60);

    // CORS — permissive to be friendly with browser-based clients. Tighten if
    // exposing on a public network.
    svr.set_default_headers({
        {"Access-Control-Allow-Origin",  "*"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
    });
    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    // Routes — every handler captures `ctx_ref` by reference.  Lifetime is
    // safe because main() does not return until svr.listen() exits.
    auto & ctx_ref = *ctx;
    svr.Get ("/",                     [&](const httplib::Request &, httplib::Response & res){
        res.set_content(ctx_ref.webui_html, "text/html; charset=utf-8");
    });
    // Favicon: serve the operator-supplied icon if --webui-icon was given;
    // otherwise return 204 No Content so the browser stops asking.
    svr.Get ("/favicon",              [&](const httplib::Request &, httplib::Response & res){
        if (ctx_ref.webui_icon.empty()) { res.status = 204; return; }
        res.set_content(ctx_ref.webui_icon, ctx_ref.webui_icon_mime.c_str());
    });
    svr.Get ("/favicon.ico",          [&](const httplib::Request &, httplib::Response & res){
        if (ctx_ref.webui_icon.empty()) { res.status = 204; return; }
        res.set_content(ctx_ref.webui_icon, ctx_ref.webui_icon_mime.c_str());
    });
    svr.Get ("/health",               [&](const auto & q, auto & r){ route_health  (ctx_ref, q, r); });
    if (args.metrics) {
        svr.Get ("/metrics",          [&](const auto & q, auto & r){ route_metrics (ctx_ref, q, r); });
    }
    svr.Get ("/v1/models",            [&](const auto & q, auto & r){
        if (!require_auth(ctx_ref, q, r)) return;
        route_models(ctx_ref, q, r);
    });
    svr.Post("/v1/chat/completions",  [&](const auto & q, auto & r){
        if (!require_auth(ctx_ref, q, r)) return;
        route_chat_completions(ctx_ref, q, r);
    });
    svr.Post("/v1/preset",            [&](const auto & q, auto & r){
        if (!require_auth(ctx_ref, q, r)) return;
        route_preset(ctx_ref, q, r);
    });

    // Last-chance error handler — never let a thrown exception propagate
    // out of the HTTP layer (httplib would close the socket abruptly).
    svr.set_exception_handler([](const auto &, auto & res, std::exception_ptr ep) {
        try { if (ep) std::rethrow_exception(ep); }
        catch (const std::exception & e) {
            res.status = 500;
            res.set_content(error_json(std::string("uncaught: ") + e.what(),
                                        "internal_error"),
                            "application/json");
        }
    });

    g_server.store(&svr);
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    bool ok = svr.listen(args.host.c_str(), args.port);
    g_server.store(nullptr);
    std::fprintf(stderr, "[easyai-server] %s\n", ok ? "stopped cleanly" : "listen failed");
    return ok ? 0 : 1;
}
