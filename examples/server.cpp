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

// llama.cpp common — for incremental chat parsing during SSE streaming.
// Not part of easyai's public API; the server is allowed to reach in.
#include "chat.h"
#include "common.h"

#include "httplib.h"          // vendored by llama.cpp
#include "nlohmann/json.hpp"  // vendored by llama.cpp

#if defined(EASYAI_BUILD_WEBUI)
// Each *.hpp declares  unsigned char NAME[]  +  unsigned int NAME_len
// where NAME is the filename with '.' / '-' replaced by '_' (xxd convention).
#include "webui_index.html.hpp"
#include "webui_bundle.js.hpp"
#include "webui_bundle.css.hpp"
#include "webui_loading.html.hpp"
// AI-brain.svg → identifier AI_brain_svg (xxd's '.'/'-' → '_' rule).
#include "easyai_brand_svg.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// (chat.h pulls in nlohmann::json + a `using json = nlohmann::ordered_json`
// alias, so we don't redeclare those here.)
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
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__EASYAI_TITLE__</title>
<link rel="icon" href="/favicon">
<!-- Optional rich-rendering libs (used when reachable; fallback otherwise) -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<style>
:root {
  --bg: #0b0d10;
  --bg-elev: #14181e;
  --bg-elev2: #1a1f27;
  --bg-input: #0f1318;
  --bg-hover: #1c2129;
  --border: #2a313b;
  --border-hi: #3a414b;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --text-faint: #5b626a;
  --accent: #1f6feb;
  --accent-hi: #2f7ffb;
  --accent-dim: #1f6feb33;
  --tool: #4fb0ff;
  --tool-bg: #102230;
  --think: #b97df3;
  --think-bg: #1a1330;
  --good: #3fb950;
  --bad: #f85149;
  --warn: #d29922;
  --shadow: 0 4px 20px rgba(0,0,0,.4);
}
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; height: 100vh; overflow: hidden; }
body {
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui,
        "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
}

/* ---------- layout shell ------------------------------------------------ */
#app { display: grid; grid-template-columns: 260px 1fr; height: 100vh; }
#app.no-sidebar { grid-template-columns: 0 1fr; }
@media (max-width: 760px) {
  #app { grid-template-columns: 0 1fr; }
  #app.show-sidebar { grid-template-columns: 240px 1fr; }
}

/* ---------- sidebar ----------------------------------------------------- */
aside#sidebar {
  background: var(--bg-elev);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  overflow: hidden;
  transition: width .15s ease;
}
.no-sidebar #sidebar { display: none; }
@media (max-width: 760px) { #sidebar { display: none; } .show-sidebar #sidebar { display: flex; } }

#sidebar .head {
  padding: .55rem .55rem; border-bottom: 1px solid var(--border);
  display: flex; gap: .4rem; align-items: center;
}
#newChatBtn {
  flex: 1; background: transparent; color: var(--text);
  border: 1px dashed var(--border-hi); border-radius: 6px;
  padding: .45rem .6rem; font: inherit; cursor: pointer;
  display: flex; align-items: center; gap: .4rem;
  transition: background .12s, border-color .12s;
}
#newChatBtn:hover { background: var(--bg-hover); border-color: var(--accent); }
#newChatBtn .plus { font-weight: 700; color: var(--accent); }

#convList { list-style: none; margin: 0; padding: .35rem; overflow-y: auto; flex: 1; }
.conv-item {
  display: flex; align-items: center; gap: .4rem;
  padding: .45rem .55rem; border-radius: 6px;
  cursor: pointer; color: var(--text-dim); font-size: .85rem;
  position: relative;
}
.conv-item:hover { background: var(--bg-hover); color: var(--text); }
.conv-item.active { background: var(--accent-dim); color: var(--text); }
.conv-item .title { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.conv-item .stamp { font-size: .68rem; color: var(--text-faint); flex-shrink: 0; }
.conv-item .del {
  display: none; background: none; border: 0; color: var(--text-dim);
  font-size: .9rem; cursor: pointer; padding: 0 .2rem; line-height: 1;
}
.conv-item:hover .del { display: inline-block; }
.conv-item .del:hover { color: var(--bad); }

#sidebar .foot {
  border-top: 1px solid var(--border); padding: .45rem .55rem;
  font-size: .7rem; color: var(--text-faint);
  display: flex; align-items: center; gap: .4rem;
}

/* ---------- main / header ----------------------------------------------- */
main { display: flex; flex-direction: column; min-width: 0; }
header.topbar {
  padding: .55rem .85rem;
  background: var(--bg-elev);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: .55rem; flex-wrap: wrap;
}
header.topbar h1 { font-size: 1rem; margin: 0; font-weight: 600; letter-spacing: .01em; }
.pill {
  font-size: .7rem; color: var(--text-dim);
  padding: .15rem .55rem; border: 1px solid var(--border); border-radius: 999px;
  white-space: nowrap;
}
.icon-btn {
  background: transparent; color: var(--text-dim); border: 1px solid transparent;
  border-radius: 6px; padding: .25rem .45rem; cursor: pointer; font-size: 1rem;
  transition: background .12s, color .12s, border-color .12s;
}
.icon-btn:hover { background: var(--bg-hover); color: var(--text); border-color: var(--border); }
.spacer { flex: 1; }

/* ---------- chat area --------------------------------------------------- */
#chat { flex: 1; overflow-y: auto; padding: 1.2rem 0 .8rem; }
.empty {
  height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center;
  color: var(--text-faint); padding: 2rem; text-align: center; gap: .6rem;
}
.empty h2 { color: var(--text-dim); font-weight: 500; margin: 0; }
.empty .ex { display: flex; flex-wrap: wrap; gap: .4rem; justify-content: center; max-width: 540px; }
.empty .ex button {
  background: var(--bg-elev); color: var(--text-dim);
  border: 1px solid var(--border); border-radius: 999px;
  padding: .35rem .8rem; font-size: .8rem; cursor: pointer;
}
.empty .ex button:hover { color: var(--text); border-color: var(--accent); }

.msg { max-width: 78ch; margin: 0 auto 1.4rem; padding: 0 1.2rem; }
.msg .role {
  font-size: .68rem; color: var(--text-faint); text-transform: uppercase;
  letter-spacing: .08em; margin-bottom: .25rem;
  display: flex; align-items: center; gap: .5rem;
}
.msg .role .actions { margin-left: auto; opacity: 0; transition: opacity .12s; }
.msg:hover .role .actions { opacity: 1; }
.msg .role .actions button {
  background: none; border: 0; color: var(--text-dim); padding: 0 .3rem;
  cursor: pointer; font-size: .75rem; border-radius: 3px;
}
.msg .role .actions button:hover { color: var(--text); background: var(--bg-hover); }

.msg.user .body {
  background: #0d2543; border: 1px solid #1f4a77;
  padding: .6rem .85rem; border-radius: 8px;
  white-space: pre-wrap; word-wrap: break-word;
}

.msg.assistant .content { padding: .15rem 0; word-wrap: break-word; overflow-wrap: anywhere; }
.msg.assistant .content p { margin: .35rem 0; }
.msg.assistant .content p:first-child { margin-top: 0; }
.msg.assistant .content p:last-child  { margin-bottom: 0; }
.msg.assistant .content h1, .msg.assistant .content h2, .msg.assistant .content h3 {
  font-size: 1rem; margin: 1rem 0 .35rem; font-weight: 600;
}
.msg.assistant .content h1 { font-size: 1.15rem; }
.msg.assistant .content h2 { font-size: 1.05rem; }
.msg.assistant .content ul, .msg.assistant .content ol { padding-left: 1.4rem; margin: .35rem 0; }
.msg.assistant .content li { margin: .15rem 0; }
.msg.assistant .content blockquote {
  border-left: 3px solid var(--border); margin: .4rem 0; padding: .1rem .8rem; color: var(--text-dim);
}
.msg.assistant .content pre {
  background: #0a0d11; border: 1px solid var(--border);
  padding: .65rem .8rem; border-radius: 6px;
  overflow-x: auto; font-size: .85em; line-height: 1.45; margin: .5rem 0;
}
.msg.assistant .content code {
  background: var(--bg-input); padding: .1rem .35rem;
  border-radius: 3px; font-size: .88em;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
}
.msg.assistant .content pre code { background: transparent; padding: 0; font-size: 1em; }
.msg.assistant .content a { color: var(--accent); text-decoration: none; }
.msg.assistant .content a:hover { text-decoration: underline; }
.msg.assistant .content table { border-collapse: collapse; margin: .5rem 0; }
.msg.assistant .content th, .msg.assistant .content td {
  border: 1px solid var(--border); padding: .25rem .5rem; font-size: .9em;
}
.msg.assistant .content th { background: var(--bg-elev); font-weight: 600; }

/* thinking + tool cards keep their own visual identity */
.thinking {
  border: 1px solid var(--think-bg); background: var(--think-bg);
  border-left: 3px solid var(--think); border-radius: 6px;
  margin: .5rem 0; padding: .35rem .6rem; font-size: .85em; color: var(--text-dim);
}
.thinking summary {
  cursor: pointer; color: var(--think); font-weight: 600;
  outline: none; user-select: none; list-style: none;
}
.thinking summary::-webkit-details-marker { display: none; }
.thinking summary::before { content: "▸ "; color: var(--think); font-weight: 700; }
.thinking[open] summary::before { content: "▾ "; }
.thinking[open] summary { margin-bottom: .4rem; }
.thinking .think-body {
  white-space: pre-wrap; font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: .82rem; line-height: 1.5; max-height: 24em; overflow-y: auto;
}

.tool-card {
  border: 1px solid var(--tool-bg); background: var(--tool-bg);
  border-left: 3px solid var(--tool); border-radius: 6px;
  margin: .5rem 0; padding: .45rem .65rem;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .8rem;
}
.tool-card.error { border-left-color: var(--bad); background: #22151a; }
.tool-card .head { display: flex; align-items: baseline; gap: .4rem; flex-wrap: wrap; }
.tool-card .name { color: var(--tool); font-weight: 600; }
.tool-card.error .name { color: var(--bad); }
.tool-card .args { color: var(--text-dim); word-break: break-all; }
.tool-card .result {
  color: var(--text); margin-top: .35rem; white-space: pre-wrap;
  max-height: 16em; overflow-y: auto; padding-top: .35rem;
  border-top: 1px dashed var(--border);
}
.tool-card .result.pending { color: var(--text-dim); font-style: italic; }

.cursor {
  display: inline-block; width: .5em; height: 1em;
  background: var(--text); animation: blink 1s steps(1, end) infinite;
  vertical-align: text-bottom; margin-left: 1px;
}
@keyframes blink { 50% { opacity: 0; } }

.stats {
  font-size: .68rem; color: var(--text-faint); margin-top: .35rem;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
}
.error-msg {
  color: var(--bad); background: #22151a;
  border: 1px solid #6a2a31; border-radius: 6px;
  padding: .5rem .7rem; font-size: .9em;
}

/* ---------- input bar --------------------------------------------------- */
#inputWrap {
  border-top: 1px solid var(--border); background: var(--bg-elev);
  padding: .55rem .85rem .7rem; position: relative;
}
#inputBar { display: flex; gap: .45rem; align-items: flex-end; }
#textArea {
  flex: 1; resize: none; min-height: 2.4rem; max-height: 30vh;
  padding: .55rem .7rem;
  background: var(--bg-input); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px;
  font: inherit; line-height: 1.5;
}
#textArea:focus { outline: 2px solid var(--accent-dim); border-color: var(--accent); }
#sendBtn, #stopBtn {
  background: var(--accent); color: #fff; border: 0;
  border-radius: 8px; padding: 0 1.05rem; cursor: pointer; font-weight: 600;
  height: 2.4rem; align-self: flex-end;
}
#sendBtn:hover { background: var(--accent-hi); }
#sendBtn:disabled { opacity: .55; cursor: wait; }
#stopBtn { background: var(--bad); }
#stopBtn:hover { filter: brightness(1.1); }
#settingsToggle, #sidebarToggle {
  background: var(--bg-input); color: var(--text-dim);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 0 .65rem; cursor: pointer; font-size: 1rem;
  height: 2.4rem; align-self: flex-end;
}
#settingsToggle:hover, #sidebarToggle:hover { background: var(--bg-hover); color: var(--text); }
#settingsToggle.active { color: var(--accent); border-color: var(--accent); }

#settingsHint {
  color: var(--text-faint); font-size: .68rem; padding-top: .3rem;
  display: flex; gap: 1rem; flex-wrap: wrap;
}
#settingsHint .pill { font-size: .65rem; padding: .05rem .45rem; cursor: pointer; }
#settingsHint .pill:hover { color: var(--text); border-color: var(--border-hi); }

/* settings popover */
#settingsPanel {
  position: absolute; bottom: calc(100% + 2px); right: .85rem;
  width: min(380px, calc(100vw - 1.7rem));
  background: var(--bg-elev2); color: var(--text);
  border: 1px solid var(--border-hi); border-radius: 10px;
  box-shadow: var(--shadow); padding: .7rem .85rem;
  display: none; z-index: 50;
}
#settingsPanel.open { display: block; }
#settingsPanel h3 {
  font-size: .82rem; margin: 0 0 .55rem; font-weight: 600;
  color: var(--text-dim); text-transform: uppercase; letter-spacing: .06em;
}
.preset-row {
  display: flex; flex-wrap: wrap; gap: .25rem;
  margin-bottom: .65rem;
}
.preset-row button {
  background: transparent; color: var(--text-dim);
  border: 1px solid var(--border); border-radius: 999px;
  padding: .25rem .65rem; cursor: pointer; font-size: .75rem;
}
.preset-row button:hover { background: var(--bg-hover); color: var(--text); }
.preset-row button.active { background: var(--accent); border-color: var(--accent); color: white; }
.slider-row {
  display: grid; grid-template-columns: 80px 1fr 56px;
  gap: .55rem; align-items: center; margin: .35rem 0;
  font-size: .82rem;
}
.slider-row label { color: var(--text-dim); }
.slider-row input[type=range] { width: 100%; accent-color: var(--accent); }
.slider-row input[type=number] {
  width: 100%; background: var(--bg-input); color: var(--text);
  border: 1px solid var(--border); border-radius: 4px; padding: .15rem .3rem;
  font: inherit; font-size: .82rem; text-align: right;
}
#settingsPanel .reset-link {
  font-size: .72rem; color: var(--text-dim); text-decoration: underline;
  background: none; border: 0; cursor: pointer; padding: 0; margin-top: .3rem;
}

/* small helper: invisible on narrow screens */
.hide-narrow { display: inline-flex; }
@media (max-width: 760px) { .hide-narrow { display: none; } }
</style></head>
<body>
<div id="app" class="show-sidebar">
  <aside id="sidebar">
    <div class="head">
      <button id="newChatBtn"><span class="plus">+</span> New chat</button>
    </div>
    <ul id="convList"></ul>
    <div class="foot">
      <span id="model" class="pill">…</span>
      <span id="backend" class="pill"></span>
      <span id="ntools" class="pill"></span>
    </div>
  </aside>

  <main>
    <header class="topbar">
      <button class="icon-btn" id="sidebarToggle" title="Toggle sidebar">☰</button>
      <h1>__EASYAI_TITLE__</h1>
      <span class="spacer"></span>
      <button class="icon-btn hide-narrow" id="thinkToggleHdr" title="Show / hide reasoning blocks">◐ thinking</button>
    </header>

    <div id="chat"></div>

    <div id="inputWrap">
      <div id="settingsPanel">
        <h3>preset</h3>
        <div class="preset-row" id="presetRow">
          <button data-p="deterministic">deterministic</button>
          <button data-p="precise">precise</button>
          <button data-p="balanced" class="active">balanced</button>
          <button data-p="creative">creative</button>
          <button data-p="wild">wild</button>
        </div>
        <h3>sampling</h3>
        <div class="slider-row">
          <label for="sTemp">temperature</label>
          <input type="range" id="sTemp" min="0" max="2" step="0.05" value="0.7">
          <input type="number" id="sTempN" min="0" max="2" step="0.05" value="0.7">
        </div>
        <div class="slider-row">
          <label for="sTopP">top_p</label>
          <input type="range" id="sTopP" min="0" max="1" step="0.01" value="0.95">
          <input type="number" id="sTopPN" min="0" max="1" step="0.01" value="0.95">
        </div>
        <div class="slider-row">
          <label for="sTopK">top_k</label>
          <input type="range" id="sTopK" min="1" max="100" step="1" value="40">
          <input type="number" id="sTopKN" min="1" max="100" step="1" value="40">
        </div>
        <div class="slider-row">
          <label for="sMinP">min_p</label>
          <input type="range" id="sMinP" min="0" max="0.5" step="0.01" value="0.05">
          <input type="number" id="sMinPN" min="0" max="0.5" step="0.01" value="0.05">
        </div>
        <div class="slider-row">
          <label for="sMaxTok">max_tokens</label>
          <input type="range" id="sMaxTok" min="0" max="4096" step="64" value="0">
          <input type="number" id="sMaxTokN" min="0" max="32768" step="1" value="0">
        </div>
        <button class="reset-link" id="settingsReset">reset to balanced</button>
      </div>

      <div id="inputBar">
        <button id="settingsToggle" title="Sampling settings">⚙</button>
        <textarea id="textArea" rows="1" placeholder="Type a message…"></textarea>
        <button id="stopBtn" hidden>stop</button>
        <button id="sendBtn">send</button>
      </div>

      <div id="settingsHint">
        <span>Ctrl/⌘+Enter to send</span>
        <span class="pill" data-quick="0">temp 0.0 (deterministic)</span>
        <span class="pill" data-quick="0.7">temp 0.7 (balanced)</span>
        <span class="pill" data-quick="1.0">temp 1.0 (creative)</span>
        <span style="color:var(--text-faint)">or inline: <code>creative 0.9 …</code></span>
      </div>
    </div>
  </main>
</div>

<!-- markdown lib (loaded async, with graceful fallback) -->
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js" defer></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js" defer></script>

<script>
"use strict";

// ============================================================================
//  helpers
// ============================================================================
const $  = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));
const escHTML = s => String(s).replace(/[&<>"']/g, c => ({
  '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
}[c]));

function relTime(ts){
  const s = (Date.now() - ts) / 1000;
  if (s < 60)        return Math.floor(s) + 's';
  if (s < 3600)      return Math.floor(s/60) + 'm';
  if (s < 86400)     return Math.floor(s/3600) + 'h';
  if (s < 86400*7)   return Math.floor(s/86400) + 'd';
  return new Date(ts).toLocaleDateString();
}

// Tiny markdown fallback used when marked.js failed to load (offline boxes).
function tinyMD(s){
  if (!s) return '';
  let html = escHTML(s);
  const parked = [];
  const park = f => (parked.push(f), `\u0001${parked.length-1}\u0001`);
  html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, l, c) =>
    park(`<pre><code class="lang-${escHTML(l)}">${c}</code></pre>`));
  html = html.replace(/`([^`\n]+)`/g, (_, c) => park(`<code>${c}</code>`));
  html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, u) =>
    park(`<a href="${u}" target="_blank" rel="noopener">${t}</a>`));
  html = html.replace(/\*\*([^\*\n]+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/(^|[^\*])\*([^\*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');
  html = html.replace(/(\bhttps?:\/\/[^\s<]+?)([.,;:!?)\]]*)(?=\s|$)/g,
    (_, u, t) => park(`<a href="${u}" target="_blank" rel="noopener">${u}</a>`) + t);
  html = html.replace(/\n/g, '<br>');
  html = html.replace(/\u0001(\d+)\u0001/g, (_, i) => parked[+i]);
  return html;
}

function renderMD(s){
  if (window.marked) {
    try {
      // Configure once.
      if (!window._mdReady) {
        window.marked.setOptions({ breaks: true, gfm: true });
        window._mdReady = true;
      }
      let html = window.marked.parse(s || '');
      // Run highlight.js on freshly built code blocks.
      if (window.hljs) {
        const tmp = document.createElement('div');
        tmp.innerHTML = html;
        $$('pre code', tmp).forEach(b => {
          try { window.hljs.highlightElement(b); } catch {}
        });
        html = tmp.innerHTML;
      }
      return html;
    } catch (e) {
      console.warn('markdown fallback', e);
    }
  }
  return tinyMD(s);
}

// ============================================================================
//  IndexedDB-backed conversation storage
// ============================================================================
const DB_NAME = 'easyai';
const STORE   = 'conversations';

function openDB(){
  return new Promise((res, rej) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id' });
    req.onsuccess = () => res(req.result);
    req.onerror   = () => rej(req.error);
  });
}
async function dbAll(){
  const db = await openDB();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => res(req.result || []);
    req.onerror   = () => rej(req.error);
  });
}
async function dbPut(conv){
  const db = await openDB();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(conv);
    tx.oncomplete = () => res();
    tx.onerror    = () => rej(tx.error);
  });
}
async function dbDel(id){
  const db = await openDB();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(id);
    tx.oncomplete = () => res();
    tx.onerror    = () => rej(tx.error);
  });
}

// ============================================================================
//  app state
// ============================================================================
const state = {
  convs:        [],           // [{id, title, created, updated, messages, settings}]
  currentId:    null,
  showThinking: true,
  abort:        null,         // current AbortController
  settings: loadSettings(),   // sampling defaults persisted in localStorage
};

const PRESETS = {
  deterministic: { temperature: 0.0, top_p: 1.0, top_k: 1,  min_p: 0.0  },
  precise:       { temperature: 0.2, top_p: 0.95, top_k: 40, min_p: 0.10 },
  balanced:      { temperature: 0.7, top_p: 0.95, top_k: 40, min_p: 0.05 },
  creative:      { temperature: 1.0, top_p: 0.95, top_k: 40, min_p: 0.05 },
  wild:          { temperature: 1.4, top_p: 0.98, top_k: 60, min_p: 0.0  },
};

function loadSettings(){
  try {
    const j = JSON.parse(localStorage.getItem('easyai-settings') || '{}');
    return Object.assign({
      preset:       'balanced',
      temperature:  PRESETS.balanced.temperature,
      top_p:        PRESETS.balanced.top_p,
      top_k:        PRESETS.balanced.top_k,
      min_p:        PRESETS.balanced.min_p,
      max_tokens:   0,            // 0 = no client cap
    }, j);
  } catch { return loadDefaultSettings(); }
}
function loadDefaultSettings(){
  return Object.assign({ preset:'balanced', max_tokens: 0 }, PRESETS.balanced);
}
function saveSettings(){ localStorage.setItem('easyai-settings', JSON.stringify(state.settings)); }

function newConv(){
  return {
    id: (crypto && crypto.randomUUID) ? crypto.randomUUID() : (Date.now() + '-' + Math.random()),
    title: 'New chat',
    created: Date.now(),
    updated: Date.now(),
    messages: [],
  };
}

async function loadAllConvs(){
  state.convs = (await dbAll().catch(() => [])).sort((a,b) => b.updated - a.updated);
  if (state.convs.length === 0) {
    const c = newConv();
    state.convs.push(c);
    state.currentId = c.id;
    await dbPut(c).catch(()=>{});
  } else {
    state.currentId = state.convs[0].id;
  }
}

const currentConv = () => state.convs.find(c => c.id === state.currentId);

// ============================================================================
//  rendering — sidebar
// ============================================================================
function renderSidebar(){
  const list = $('#convList');
  list.innerHTML = '';
  for (const c of state.convs) {
    const li = document.createElement('li');
    li.className = 'conv-item' + (c.id === state.currentId ? ' active' : '');
    li.innerHTML = `
      <span class="title"></span>
      <span class="stamp"></span>
      <button class="del" title="Delete">×</button>`;
    $('.title', li).textContent = c.title || 'untitled';
    $('.stamp', li).textContent = relTime(c.updated);
    li.onclick = e => {
      if (e.target.classList.contains('del')) return;
      switchConv(c.id);
    };
    $('.del', li).onclick = async () => {
      if (!confirm('Delete this conversation?')) return;
      await dbDel(c.id).catch(()=>{});
      state.convs = state.convs.filter(x => x.id !== c.id);
      if (state.currentId === c.id) {
        if (state.convs.length === 0) {
          const nc = newConv(); state.convs.push(nc); state.currentId = nc.id;
          await dbPut(nc).catch(()=>{});
        } else state.currentId = state.convs[0].id;
      }
      renderSidebar(); renderChat();
    };
    list.appendChild(li);
  }
}

function switchConv(id){
  if (state.abort) state.abort.abort();
  state.currentId = id;
  renderSidebar();
  renderChat();
}

async function startNewConv(){
  if (state.abort) state.abort.abort();
  const c = newConv();
  state.convs.unshift(c);
  state.currentId = c.id;
  await dbPut(c).catch(()=>{});
  renderSidebar();
  renderChat();
  $('#textArea').focus();
}

// ============================================================================
//  rendering — chat
// ============================================================================
const chatEl = $('#chat');

function renderChat(){
  const c = currentConv();
  chatEl.innerHTML = '';
  if (!c || c.messages.length === 0) {
    chatEl.innerHTML = `
      <div class="empty">
        <h2>How can I help?</h2>
        <div class="ex">
          <button data-pp="What time is it now?">What time is it now?</button>
          <button data-pp="Search the web for the latest llama.cpp release and summarise.">Latest llama.cpp release</button>
          <button data-pp="creative 0.9 write me a haiku about silicon">creative 0.9 haiku</button>
          <button data-pp="List the files in the current directory.">List files</button>
        </div>
      </div>`;
    $$('.empty .ex button').forEach(b => b.onclick = () => {
      $('#textArea').value = b.dataset.pp;
      $('#textArea').focus();
    });
    return;
  }
  for (const m of c.messages) renderMessage(m);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function appendUserMsg(text){
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `<div class="role">you</div><div class="body"></div>`;
  $('.body', el).textContent = text;
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  return el;
}

function renderMessage(m){
  if (m.role === 'user') {
    appendUserMsg(m.content);
    return;
  }
  if (m.role === 'assistant') {
    const turn = new AssistantTurn(state.showThinking, /*live=*/false);
    turn.raw = m.content || '';
    if (m.toolCards) {
      for (const t of m.toolCards) turn.replayToolCard(t);
    }
    turn.finish('stop');
  }
}

// ============================================================================
//  AssistantTurn — one streaming reply
// ============================================================================
class AssistantTurn {
  constructor(showThinking, live=true){
    this.raw = '';
    this.tokens = 0;
    this.startMs = performance.now();
    this.showThinking = showThinking;
    this.toolCards = [];          // {name, args, result, error}
    this.pendingByName = new Map();
    this._done = !live;

    this.el = document.createElement('div');
    this.el.className = 'msg assistant';
    this.el.innerHTML = `
      <div class="role">
        <span>assistant</span>
        <span class="actions">
          <button class="copy" title="Copy">copy</button>
        </span>
      </div>
      <div class="content"></div>
      <div class="stats"></div>`;
    this.contentEl = $('.content', this.el);
    this.statsEl   = $('.stats',   this.el);
    chatEl.appendChild(this.el);
    chatEl.scrollTop = chatEl.scrollHeight;

    $('.copy', this.el).onclick = () => {
      const t = this.finalText();
      navigator.clipboard?.writeText(t).catch(()=>{});
    };
  }

  addContent(piece){
    this.tokens += 1;
    this.raw += piece;
    this.render();
  }

  render(){
    // Split out <think>...</think> blocks; render content with markdown.
    let cleaned = '';
    const thinks = [];
    const re = /<think(?:ing)?>([\s\S]*?)(?:<\/think(?:ing)?>|$)/g;
    let m, last = 0;
    while ((m = re.exec(this.raw)) !== null) {
      cleaned += this.raw.slice(last, m.index);
      thinks.push(m[1]);
      last = m.index + m[0].length;
    }
    cleaned += this.raw.slice(last);

    let thinkEl = $(':scope > .thinking', this.el);
    if (thinks.length) {
      const txt = thinks.join('\n\n— —\n\n');
      if (!thinkEl) {
        thinkEl = document.createElement('details');
        thinkEl.className = 'thinking';
        if (this.showThinking) thinkEl.open = true;
        thinkEl.innerHTML = '<summary>Thinking</summary><div class="think-body"></div>';
        this.el.insertBefore(thinkEl, this.contentEl);
      }
      $('.think-body', thinkEl).textContent = txt;
    }

    this.contentEl.innerHTML = renderMD(cleaned) +
        (this._done ? '' : '<span class="cursor"></span>');
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  // server-side tool dispatch — card with running placeholder
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
    this.el.insertBefore(card, this.contentEl);
    this.toolCards.push({ el: card, name: evt.name, args: argStr });
    if (!this.pendingByName.has(evt.name)) this.pendingByName.set(evt.name, []);
    this.pendingByName.get(evt.name).push(card);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  addToolResult(evt){
    const stack = this.pendingByName.get(evt.name);
    if (!stack || !stack.length) return;
    const card = stack.shift();
    const resEl = $('.result', card);
    resEl.classList.remove('pending');
    resEl.textContent = evt.content || '';
    if (evt.is_error) card.classList.add('error');
    // Keep a record for replay/persistence.
    const rec = this.toolCards.find(t => t.el === card);
    if (rec) { rec.result = evt.content || ''; rec.error = !!evt.is_error; }
  }

  // For client-tools mode (rare from the webui).
  addClientToolCalls(tcs){
    for (const tc of tcs) {
      if (!tc.function) continue;
      this.addToolCall({
        name: tc.function.name || '?',
        arguments: tc.function.arguments || '{}',
        id: tc.id || ''
      });
      const stack = this.pendingByName.get(tc.function.name);
      if (stack && stack.length) {
        const card = stack.shift();
        $('.result', card).classList.remove('pending');
        $('.result', card).textContent = '(forwarded to client for execution)';
        $('.result', card).style.fontStyle = 'italic';
      }
    }
  }

  // Used during conversation replay (loading from IDB).
  replayToolCard(t){
    const card = document.createElement('div');
    card.className = 'tool-card' + (t.error ? ' error' : '');
    card.innerHTML = `
      <div class="head">
        <span class="name"></span>
        <span class="args"></span>
      </div>
      <div class="result"></div>`;
    $('.name', card).textContent = t.name;
    $('.args', card).textContent = '(' + (t.args || '') + ')';
    $('.result', card).textContent = t.result || '';
    this.el.insertBefore(card, this.contentEl);
    this.toolCards.push(t);
  }

  finish(reason){
    this._done = true;
    const elapsed = (performance.now() - this.startMs) / 1000;
    if (this.tokens > 0) {
      const tps = this.tokens / Math.max(elapsed, 0.001);
      this.statsEl.textContent = `${this.tokens} chunks · ${elapsed.toFixed(2)}s · ${tps.toFixed(1)} chunks/s · ${reason}`;
    }
    this.render();
  }
  fail(msg){
    this._done = true;
    this.contentEl.innerHTML = `<div class="error-msg">⚠︎ ${escHTML(msg)}</div>`;
  }

  // Strip <think> blocks — what we save into history for the next request.
  finalText(){
    return this.raw.replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/g, '').trim();
  }
}

// ============================================================================
//  SSE consumer
// ============================================================================
async function streamChat(messages, settings, handlers, signal){
  const body = {
    model: 'easyai',
    messages,
    stream: true,
    temperature: settings.temperature,
    top_p:       settings.top_p,
    top_k:       settings.top_k,
    min_p:       settings.min_p,
  };
  if (settings.max_tokens > 0) body.max_tokens = settings.max_tokens;

  const res = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt.slice(0, 400)}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    while (true) {
      const idx = buf.indexOf('\n\n');
      if (idx === -1) break;
      const event = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let evtType = 'message';
      const dataLines = [];
      for (let line of event.split('\n')) {
        if (line.endsWith('\r')) line = line.slice(0, -1);
        if (line.startsWith('event:')) evtType = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
      }
      const data = dataLines.join('\n');
      if (!data) continue;
      if (data === '[DONE]') return;
      try {
        const j = JSON.parse(data);
        if (evtType === 'easyai.tool_call')   { handlers.onToolCall && handlers.onToolCall(j); continue; }
        if (evtType === 'easyai.tool_result') { handlers.onToolResult && handlers.onToolResult(j); continue; }
        const ch = j?.choices?.[0];
        if (!ch) continue;
        if (ch.delta?.content)    handlers.onContent && handlers.onContent(ch.delta.content);
        if (ch.delta?.tool_calls) handlers.onClientToolCalls && handlers.onClientToolCalls(ch.delta.tool_calls);
        if (ch.finish_reason)     handlers.onFinish && handlers.onFinish(ch.finish_reason);
        if (j.error) throw new Error(j.error.message || 'server error');
      } catch (e) {
        if (e.name === 'AbortError') throw e;
        console.warn('SSE parse', data, e);
      }
    }
  }
}

// ============================================================================
//  send pipeline
// ============================================================================
async function send(text){
  const c = currentConv();
  if (!c) return;
  c.messages.push({ role: 'user', content: text });
  if (c.messages.length === 1) c.title = (text.slice(0, 40) || 'New chat');
  c.updated = Date.now();
  appendUserMsg(text);
  // remove the empty placeholder if present
  $$('.empty', chatEl).forEach(e => e.remove());
  await dbPut(c).catch(()=>{});
  renderSidebar();

  const turn = new AssistantTurn(state.showThinking);
  $('#sendBtn').hidden = true;
  $('#stopBtn').hidden = false;
  state.abort = new AbortController();

  // Build the API messages: include past assistant content sans tool cards.
  const apiMsgs = c.messages.map(m =>
    m.role === 'assistant' ? { role: 'assistant', content: m.content || '' }
                           : { role: m.role, content: m.content });

  try {
    await streamChat(apiMsgs, state.settings, {
      onContent:         p => turn.addContent(p),
      onToolCall:        e => turn.addToolCall(e),
      onToolResult:      e => turn.addToolResult(e),
      onClientToolCalls: t => turn.addClientToolCalls(t),
      onFinish:          r => turn.finish(r),
    }, state.abort.signal);
    c.messages.push({
      role: 'assistant',
      content: turn.finalText(),
      toolCards: turn.toolCards.map(t => ({ name: t.name, args: t.args, result: t.result, error: t.error })),
    });
    c.updated = Date.now();
    await dbPut(c).catch(()=>{});
    renderSidebar();
  } catch (e) {
    if (e.name === 'AbortError') {
      turn.fail('stopped');
    } else {
      turn.fail(e.message || String(e));
    }
  } finally {
    $('#sendBtn').hidden = false;
    $('#stopBtn').hidden = true;
    state.abort = null;
    $('#textArea').focus();
  }
}

function stopGeneration(){
  if (state.abort) state.abort.abort();
}

// ============================================================================
//  settings UI wiring
// ============================================================================
function syncSettingsUI(){
  $('#sTemp').value  = $('#sTempN').value  = state.settings.temperature;
  $('#sTopP').value  = $('#sTopPN').value  = state.settings.top_p;
  $('#sTopK').value  = $('#sTopKN').value  = state.settings.top_k;
  $('#sMinP').value  = $('#sMinPN').value  = state.settings.min_p;
  $('#sMaxTok').value = $('#sMaxTokN').value = state.settings.max_tokens;
  $$('#presetRow button').forEach(b => b.classList.toggle('active', b.dataset.p === state.settings.preset));
}

function applyPreset(name){
  const p = PRESETS[name];
  if (!p) return;
  state.settings.preset = name;
  Object.assign(state.settings, p);
  saveSettings();
  syncSettingsUI();
}

function bindSlider(rangeId, numId, key, min, max){
  const r = $('#' + rangeId), n = $('#' + numId);
  const update = v => {
    let f = parseFloat(v);
    if (isNaN(f)) return;
    if (f < min) f = min; if (f > max) f = max;
    state.settings[key] = f;
    state.settings.preset = 'custom';
    $$('#presetRow button').forEach(b => b.classList.remove('active'));
    saveSettings();
    r.value = f; n.value = f;
  };
  r.addEventListener('input', () => update(r.value));
  n.addEventListener('change', () => update(n.value));
}
bindSlider('sTemp',  'sTempN',  'temperature', 0,    2);
bindSlider('sTopP',  'sTopPN',  'top_p',       0,    1);
bindSlider('sTopK',  'sTopKN',  'top_k',       1,    1024);
bindSlider('sMinP',  'sMinPN',  'min_p',       0,    0.5);
bindSlider('sMaxTok','sMaxTokN','max_tokens',  0,    65536);

$$('#presetRow button').forEach(b => {
  b.onclick = () => applyPreset(b.dataset.p);
});

$('#settingsReset').onclick = () => applyPreset('balanced');

$('#settingsToggle').onclick = e => {
  $('#settingsPanel').classList.toggle('open');
  e.currentTarget.classList.toggle('active');
};
document.addEventListener('click', e => {
  if (!e.target.closest('#settingsPanel') && !e.target.closest('#settingsToggle')) {
    $('#settingsPanel').classList.remove('open');
    $('#settingsToggle').classList.remove('active');
  }
});

// quick-temp pills under the textarea
$$('#settingsHint .pill[data-quick]').forEach(p => {
  p.onclick = () => {
    const v = parseFloat(p.dataset.quick);
    state.settings.temperature = v;
    state.settings.preset = 'custom';
    saveSettings(); syncSettingsUI();
  };
});

// ============================================================================
//  app wiring
// ============================================================================
const ta = $('#textArea');

ta.addEventListener('input', () => {
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, window.innerHeight * 0.3) + 'px';
});
ta.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    submitMsg();
  }
});

function submitMsg(){
  const text = ta.value.trim();
  if (!text || state.abort) return;
  ta.value = ''; ta.style.height = 'auto';
  send(text);
}

$('#sendBtn').onclick = submitMsg;
$('#stopBtn').onclick = stopGeneration;
$('#newChatBtn').onclick = startNewConv;
$('#sidebarToggle').onclick = () => {
  $('#app').classList.toggle('show-sidebar');
  $('#app').classList.toggle('no-sidebar');
};
$('#thinkToggleHdr').onclick = () => {
  state.showThinking = !state.showThinking;
  $$('details.thinking').forEach(d => d.open = state.showThinking);
  $('#thinkToggleHdr').textContent = (state.showThinking ? '◐' : '○') + ' thinking';
};

// /health pills
fetch('/health').then(r => r.json()).then(j => {
  $('#model').textContent   = j.model || '?';
  $('#backend').textContent = j.backend ? 'backend: ' + j.backend : '';
  $('#ntools').textContent  = (j.tools ?? 0) + ' tools';
}).catch(()=>{});

// ============================================================================
//  bootstrap
// ============================================================================
(async () => {
  syncSettingsUI();
  await loadAllConvs();
  renderSidebar();
  renderChat();
  ta.focus();
})();
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
    bool                       inject_datetime    = true;        // server-side authoritative date/time injection
    std::string                knowledge_cutoff   = "2024-10";   // model training-data cutoff hint
    easyai::Preset             default_preset;  // current "ambient" preset
    std::string                model_id;        // basename of model file
    std::string                api_key;         // empty = auth disabled
    bool                       no_think = false;// strip <think> from responses

    // Webui customisation (built once at start-up so the / handler can just
    // hand back the prebuilt buffer).
    std::string                webui_html;      // HTML with title substituted
    std::string                webui_bundle_js; // bundle.js with brand strings substituted
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
    // Per-request override of the server-side --inject-datetime flag,
    // populated from the X-Easyai-Inject HTTP header.  Empty leaves
    // the server default in effect; "on" forces injection on; "off"
    // forces it off.  Use sparingly — disabling the preamble removes
    // a real safety net (the model will start hallucinating "today"
    // and post-cutoff facts without it).
    std::string                                       inject_override;
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

    // Optional per-request override of the authoritative-datetime
    // injection.  The header lookup uses httplib's case-insensitive
    // Headers comparator, so any casing reaches us.
    auto hi = req.headers.find("X-Easyai-Inject");
    if (hi != req.headers.end()) {
        std::string v = hi->second;
        for (auto & c : v) c = (char) std::tolower((unsigned char) c);
        if (v == "on" || v == "off") out.inject_override = v;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Apply request-level overrides + replace history. Caller already holds
// engine_mu and has called reset_engine_defaults().
// ---------------------------------------------------------------------------
// Build an authoritative preamble that's appended to the SERVER's
// default system prompt on every request.  Per user request: pass the
// model the current date/time + timezone, and tell it about its
// knowledge cutoff so it knows when to defer to tools instead of
// hallucinating post-cutoff facts.
//
// We rebuild this per request so the timestamp is always fresh.
static std::string build_authoritative_preamble(const ServerCtx & ctx) {
    auto now    = std::chrono::system_clock::now();
    auto tt     = std::chrono::system_clock::to_time_t(now);
    std::tm lt{};
#if defined(_WIN32)
    localtime_s(&lt, &tt);
#else
    localtime_r(&tt, &lt);
#endif
    char ts[64]; char tz[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S %z", &lt);
    std::strftime(tz, sizeof(tz), "%Z",                    &lt);

    std::ostringstream out;
    out << "\n\n# AUTHORITATIVE DATE/TIME (do not ignore, do not second-guess)\n"
        << "Current date and time: " << ts << " (" << tz << ").\n"
        << "Trust this over any training-data intuition about \"today\".\n"
        << "If the user mentions \"today\", \"now\", \"this year\" etc., use the\n"
        << "value above.  When unsure, call the `datetime` tool first.\n"
        << "\n# KNOWLEDGE CUTOFF\n"
        << "Your training data ends around " << ctx.knowledge_cutoff << ".\n"
        << "For ANY claim about events, people, products, prices, releases,\n"
        << "leaders, scores, weather, or facts after that cutoff you MUST\n"
        << "either:\n"
        << "  1. Call a tool (web_search, web_fetch, datetime, …) to verify, OR\n"
        << "  2. Explicitly state that you are not certain.\n"
        << "Never present a post-cutoff fact as known.  Hallucination is\n"
        << "considered a critical failure for this assistant.\n";
    return out.str();
}

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

    // Build the request's history (everything except the final user
    // message — that one is pushed by the chat loop).  We work on a
    // mutable copy so we can splice the authoritative-datetime
    // preamble into whichever system message exists, regardless of
    // whether the CLIENT supplied one or we fall back to the
    // server's default.
    //
    // Critical scenario: opencode / Claude-Code style clients send
    // their OWN system message in the messages array.  That message
    // *replaces* the server's default in practice (the chat template
    // would render two system blocks otherwise, which most templates
    // collapse into one with unpredictable order).  We want the
    // datetime + cutoff hint to ride along regardless of who owns
    // the system prompt — so we APPEND the preamble to whichever
    // system message goes into the prompt:
    //
    //   * Client sent a system message → append to its content.
    //   * Client sent no system message → append to ctx.default_system.
    //
    // Either way, the preamble is in there exactly once and lands as
    // part of the model's actual context.
    std::vector<std::pair<std::string, std::string>> hist_minus_last(
        req.hist.begin(), req.hist.end() - 1);

    // Resolve the effective inject toggle: server default + optional
    // X-Easyai-Inject header override.  Default is on; QA / regression
    // suites that want to A/B the preamble pass `X-Easyai-Inject: off`.
    bool inject_now = ctx.inject_datetime;
    if      (req.inject_override == "on")  inject_now = true;
    else if (req.inject_override == "off") inject_now = false;

    if (inject_now) {
        const std::string preamble = build_authoritative_preamble(ctx);
        // Find the LAST system message in the client-supplied history.
        // (Most clients put it at index 0, but we walk backwards to be
        // safe — multi-system histories pick the most recent.)
        auto it = std::find_if(hist_minus_last.rbegin(), hist_minus_last.rend(),
            [](const std::pair<std::string,std::string> & m) {
                return m.first == "system";
            });
        if (it != hist_minus_last.rend()) {
            it->second += preamble;
            ctx.engine.system(ctx.default_system);
        } else {
            ctx.engine.system(ctx.default_system + preamble);
        }
    } else {
        ctx.engine.system(ctx.default_system);
    }

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

            // Snapshot perf counters BEFORE this request so we can diff
            // them out at the end (llama_perf_context() returns cumulative
            // values across the lifetime of the llama_context).
            const auto perf_before = ctx.engine.perf_data();

            // ---- emit helpers --------------------------------------------
            auto emit_data = [&sink](const std::string & ev) {
                std::string s = "data: " + ev + "\n\n";
                sink.write(s.data(), s.size());
            };
            auto emit_event = [&sink](const std::string & evt_type, const std::string & ev) {
                std::string s = "event: " + evt_type + "\ndata: " + ev + "\n\n";
                sink.write(s.data(), s.size());
            };
            // Dump JSON tolerantly: a multi-byte UTF-8 character can split
            // across two model tokens (one piece ends with the lead byte,
            // the next starts with the continuation byte 0x80-0xBF), and
            // strict-mode dump throws on an isolated continuation byte
            // ("invalid UTF-8 byte at index 0: 0x8B"), aborting the
            // service.  ::replace substitutes a U+FFFD glyph for invalid
            // bytes and keeps the stream alive.
            auto safe_dump = [](const ordered_json & j) {
                return j.dump(-1, ' ', false,
                              ordered_json::error_handler_t::replace);
            };

            // ---- streaming pipeline (llama-server-compatible) -----------
            // Mirror llama-server's emit pipeline:
            //   1. accumulate the model's raw text as it streams
            //   2. after each token, common_chat_parse(text, is_partial=true)
            //      with the current chat template's reasoning_format/parser,
            //      which extracts msg.reasoning_content + msg.content
            //   3. compute common_chat_msg_diff vs the previous parse
            //   4. emit standard OpenAI-shape SSE deltas with the new
            //      reasoning_content_delta / content_delta fields
            //
            // This is exactly what tools/server/server-task.cpp does, so the
            // embedded webui's native parsers (.choices[0].delta.content +
            // .choices[0].delta.reasoning_content) light up correctly.
            //
            // IMPORTANT: many chat templates (Qwen3-* in particular) raise
            // a Jinja exception if there's no user message in history, so
            // we must push the last user message BEFORE calling
            // chat_params_for_current_state — otherwise the render throws.
            // Once it's pushed we call chat_continue() (or generate_one
            // for client-tool mode) which doesn't re-push.
            ctx.engine.push_message("user", req_state->last_user);
            const bool user_already_pushed = true;

            common_chat_params chat_p;
            try {
                chat_p = ctx.engine.chat_params_for_current_state(true);
            } catch (const std::exception & e) {
                std::fprintf(stderr,
                    "[easyai-server] chat_params render threw: %s — "
                    "falling back to generic parser\n", e.what());
            }
            common_chat_parser_params parser_params(chat_p);
            parser_params.parse_tool_calls = true;
            parser_params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            if (!chat_p.parser.empty()) {
                try { parser_params.parser.load(chat_p.parser); }
                catch (...) { /* will fall through to non-PEG parse */ }
            }

            std::string accumulated;
            common_chat_msg prev_msg;
            prev_msg.role = "assistant";
            const bool drop_thinking = ctx.no_think;
            // Tracks whether ANY content delta has been emitted during
            // this request.  When the parser blows up on malformed
            // tool_call syntax (Qwen-style doubled-brace JSON, etc.) the
            // partial parses all throw and the engine silently falls
            // back to msg.content=raw.  Without this flag the client
            // would see an empty bubble — see the post-chat_continue
            // last-resort emit below.
            bool any_content_emitted = false;

            auto emit_diff = [&](const common_chat_msg_diff & d) {
                ordered_json delta = ordered_json::object();
                if (!d.reasoning_content_delta.empty() && !drop_thinking) {
                    delta["reasoning_content"] = d.reasoning_content_delta;
                }
                if (!d.content_delta.empty()) {
                    delta["content"] = d.content_delta;
                    any_content_emitted = true;
                }
                // tool_call deltas: skipped — we already surface them via
                // the easyai.tool_call / easyai.tool_result custom events.
                if (delta.empty()) return;
                ordered_json env;
                env["choices"] = json::array({{
                    {"index", 0},
                    {"delta", delta},
                    {"finish_reason", nullptr},
                }});
                emit_data(safe_dump(env));
            };

            ctx.engine.on_token([&](const std::string & piece) {
                accumulated += piece;
                common_chat_msg new_msg;
                try {
                    new_msg = common_chat_parse(accumulated,
                                                /*is_partial=*/true,
                                                parser_params);
                } catch (const std::exception &) {
                    // Incomplete tag mid-stream — wait for more bytes.
                    return;
                }
                new_msg.role = "assistant";
                std::vector<common_chat_msg_diff> diffs;
                try {
                    diffs = common_chat_msg_diff::compute_diffs(prev_msg, new_msg);
                } catch (const std::exception &) {
                    // compute_diffs is strict about monotonic tool_calls
                    // growth; incremental parses sometimes assemble a
                    // tool_call and then "unassemble" it as later tokens
                    // arrive (e.g. the parser briefly treats partial
                    // arguments as complete).  Rather than killing the
                    // stream, hold prev_msg on the last good state and
                    // wait for the next token to settle.
                    return;
                }
                prev_msg = new_msg;
                for (const auto & d : diffs) emit_diff(d);
            });

            // ---- on_tool: surface server-side dispatches to the client ----
            // Two channels:
            //   easyai.tool_call / easyai.tool_result custom SSE events for
            //     our own webui (drives the per-message status pill and any
            //     sidecar tool log in the future)
            //   a one-line inline markdown indicator so OpenAI-shape clients
            //     (incl. the bundle's own renderer) at least show that
            //     something happened
            // After firing we reset the incremental-parse state because
            // the next agentic hop is a fresh model turn — its text starts
            // from scratch even though it shares the same on_token callback.
            ctx.engine.on_tool([&](const easyai::ToolCall & c, const easyai::ToolResult & r) {
                ctx.n_tool_calls.fetch_add(1, std::memory_order_relaxed);

                // The embedded webui's stream parser ignores the SSE
                // `event:` field and inspects every `data:` line as if it
                // were an OpenAI chunk — i.e. it does parsed.choices[0]
                // unguarded.  Our custom payloads have no .choices, so the
                // bundle throws "undefined is not an object" on every one.
                // Add an empty choices[] stub so the bundle's destructure
                // is harmless; our own monitorSSE keys off `evtType` and
                // never reads .choices on these payloads.
                ordered_json call_evt;
                call_evt["choices"]   = json::array({{
                    {"index", 0},
                    {"delta", json::object()},
                    {"finish_reason", nullptr},
                }});
                call_evt["name"]      = c.name;
                call_evt["arguments"] = c.arguments_json;
                call_evt["id"]        = c.id;
                emit_event("easyai.tool_call", safe_dump(call_evt));

                ordered_json res_evt;
                res_evt["choices"]  = json::array({{
                    {"index", 0},
                    {"delta", json::object()},
                    {"finish_reason", nullptr},
                }});
                res_evt["name"]     = c.name;
                res_evt["content"]  = r.content;
                res_evt["is_error"] = r.is_error;
                emit_event("easyai.tool_result", safe_dump(res_evt));

                std::ostringstream md;
                md << "\n*" << (r.is_error ? "❌ " : "🔧 ") << c.name;
                if (r.is_error) {
                    std::string reason = r.content;
                    if (reason.size() > 80) { reason.resize(80); reason += "…"; }
                    md << " — " << reason;
                }
                md << "*\n";

                ordered_json delta;
                delta["choices"] = json::array({{
                    {"index", 0},
                    {"delta", {{"content", md.str()}}},
                    {"finish_reason", nullptr},
                }});
                emit_data(safe_dump(delta));

                // Reset incremental-parse state for the next iteration.
                accumulated.clear();
                prev_msg = common_chat_msg{};
                prev_msg.role = "assistant";
            });

            // Engine fires this when chat_continue() throws away a turn and
            // restarts (e.g. thought-only retry path for Qwen3 fine-tunes
            // that terminate after </think> without an answer).  We need to
            // drop our incremental-parse state so the next hop's tokens are
            // diff'd from a clean baseline instead of being concatenated to
            // the discarded turn's reasoning.
            ctx.engine.on_hop_reset([&]() {
                accumulated.clear();
                prev_msg = common_chat_msg{};
                prev_msg.role = "assistant";
            });

            // ---- run the engine ----------------------------------------
            std::string finish_reason = "stop";
            std::vector<std::pair<std::string, std::string>> tool_calls;
            std::vector<std::string> tool_call_ids;
            std::string engine_final_content;     // captured from chat_continue
            try {
                // The user message has ALREADY been pushed above (so the
                // chat_params render works for templates that require it).
                // Use generate_one / chat_continue here so we don't push
                // it a second time.
                (void) user_already_pushed;
                if (req_state->client_tools) {
                    auto turn = ctx.engine.generate_one();
                    tool_calls    = std::move(turn.tool_calls);
                    tool_call_ids = std::move(turn.tool_call_ids);
                    finish_reason = turn.tool_calls.empty() ? "stop" : "tool_calls";
                } else {
                    engine_final_content = ctx.engine.chat_continue();
                }
            } catch (const std::exception & e) {
                ctx.n_errors.fetch_add(1, std::memory_order_relaxed);
                ordered_json err;
                err["error"] = { {"message", e.what()}, {"type", "internal_error"} };
                emit_data(safe_dump(err));
            }

            // Final non-partial parse to flush any reasoning/content the
            // partial parses may have held back due to incomplete tags.
            try {
                auto final_msg = common_chat_parse(accumulated,
                                                   /*is_partial=*/false,
                                                   parser_params);
                final_msg.role = "assistant";
                auto diffs = common_chat_msg_diff::compute_diffs(prev_msg, final_msg);
                for (const auto & d : diffs) emit_diff(d);
            } catch (...) { /* best-effort drain */ }

            // Last-resort fallback: when EVERY partial parse threw (which
            // happens when the model emits malformed Qwen-style tool_call
            // syntax), on_token returned silently for every token and the
            // client never got a content delta.  The engine itself fell
            // back to msg.content=raw inside parse_assistant; surface that
            // as one synthesised delta so the bubble isn't empty.
            //
            // DEFENSIVE: strip any reasoning that we ALREADY streamed via
            // reasoning_content_delta — otherwise the user sees the same
            // thinking text twice (in our blue brain panel AND in the
            // message body).  We do this by removing any <think>…</think>
            // span from the synthesised content; if the engine already
            // separated reasoning out (post-2026-04-26 fix in
            // src/engine.cpp) this is a no-op.
            if (!any_content_emitted && !engine_final_content.empty()
                    && !req_state->client_tools) {
                std::string body = engine_final_content;
                // Drop everything between <think> and </think> (inclusive)
                // since reasoning has already been streamed.  Plain string
                // scan — std::regex blew the stack here once already
                // (commit 3dec718) so we keep it iterative.
                {
                    static const std::string open_tag  = "<think>";
                    static const std::string close_tag = "</think>";
                    size_t close_pos = body.find(close_tag);
                    if (close_pos != std::string::npos) {
                        size_t open_pos = body.find(open_tag);
                        size_t span_begin = (open_pos != std::string::npos
                                              && open_pos < close_pos)
                                               ? open_pos : 0;
                        body.erase(span_begin,
                                   close_pos + close_tag.size() - span_begin);
                        size_t lead = 0;
                        while (lead < body.size() &&
                               (body[lead] == '\n' || body[lead] == '\r' ||
                                body[lead] == ' '  || body[lead] == '\t')) ++lead;
                        if (lead) body.erase(0, lead);
                    }
                }
                if (!body.empty()) {
                    ordered_json env;
                    env["choices"] = json::array({{
                        {"index", 0},
                        {"delta", {{"content", body}}},
                        {"finish_reason", nullptr},
                    }});
                    emit_data(safe_dump(env));
                    any_content_emitted = true;
                }
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
                emit_data(safe_dump(delta));
            }

            // Final close-out chunk — empty delta + finish_reason + the
            // llama-server-style `timings` block the embedded webui
            // consumes to show tokens/s, Reading/Generation phases, and
            // context usage.  We also include OpenAI-standard `usage` so
            // generic clients see the prompt/completion counts.
            const auto perf_after = ctx.engine.perf_data();
            const int  prompt_n     = std::max(0, perf_after.n_prompt_tokens    - perf_before.n_prompt_tokens);
            const int  predicted_n  = std::max(0, perf_after.n_predicted_tokens - perf_before.n_predicted_tokens);
            const double prompt_ms    = std::max(0.0, perf_after.prompt_ms    - perf_before.prompt_ms);
            const double predicted_ms = std::max(0.0, perf_after.predicted_ms - perf_before.predicted_ms);

            ordered_json done_delta;
            done_delta["choices"] = json::array({{
                {"index", 0},
                {"delta", json::object()},
                {"finish_reason", finish_reason},
            }});
            // NOTE: do NOT set "agentic" as a boolean — the embedded webui
            // expects timings.agentic to be an object of shape
            // {llm:{prompt_n,prompt_ms,predicted_n,predicted_ms},perTurn:[…]}
            // and crashes the stats component when it gets a primitive.
            // Leaving the field out makes the renderer fall back to the
            // top-level timings, which is what we want.
            // Cumulative session usage for the bar's ctx counter.
            // perf_after.n_ctx_used is the live KV-cache fill — exactly
            // how many tokens of n_ctx the chat history is occupying
            // RIGHT NOW, including system prompt + every prior turn.
            // We also send n_ctx so the webui doesn't need /props for
            // the denominator.  cache_n keeps its prior meaning (tokens
            // already in cache before THIS turn started) for callers
            // that cared about the prefix-match cache hit ratio.
            const int n_ctx_total = ctx.engine.n_ctx();
            const int n_ctx_used  = perf_after.n_ctx_used;
            done_delta["timings"] = {
                {"prompt_n",     prompt_n},
                {"prompt_ms",    prompt_ms},
                {"predicted_n",  predicted_n},
                {"predicted_ms", predicted_ms},
                {"cache_n",      perf_before.n_ctx_used},
                {"ctx_used",     n_ctx_used},
                {"n_ctx",        n_ctx_total},
            };
            done_delta["usage"] = {
                {"prompt_tokens",     prompt_n},
                {"completion_tokens", predicted_n},
                {"total_tokens",      prompt_n + predicted_n},
            };
            emit_data(safe_dump(done_delta));
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
// GET /v1/tools  →  tool catalog so the webui can render a popover with
// {name, description} pairs.  Read-only; no auth needed when the rest of
// /v1/* is open, gated by require_auth otherwise (wired at route mount).
// ---------------------------------------------------------------------------
static void route_tools(ServerCtx & ctx, const httplib::Request &,
                        httplib::Response & res) {
    ordered_json arr = json::array();
    for (const auto & t : ctx.default_tools) {
        ordered_json e;
        e["name"]        = t.name;
        e["description"] = t.description;
        arr.push_back(std::move(e));
    }
    ordered_json env;
    env["object"] = "list";
    env["data"]   = std::move(arr);
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
        "       --inject-datetime <on|off>\n"
        "                               Append authoritative date/time + TZ + a\n"
        "                               knowledge-cutoff hint to the system prompt\n"
        "                               on EVERY request (default on).  Tells the\n"
        "                               model to trust the server's clock and to\n"
        "                               verify post-cutoff facts via tools.\n"
        "       --knowledge-cutoff <YYYY-MM>\n"
        "                               Model training-data cutoff hint used by\n"
        "                               --inject-datetime.  Default '2024-10'.\n"
        "  -v,  --verbose               Engine logs raw model output + parser\n"
        "                                 actions to stderr — useful for debugging\n"
        "                                 'why did it stop?' moments\n"
        "\nWebui:\n"
        "       --webui <mode>          'modern' (default — embedded llama-server-\n"
        "                                derived bundle) or 'minimal' (small inline UI)\n"
        "       --webui-title <text>    Title in the browser tab AND the sidebar\n"
        "                                brand (default 'Box EasyAI')\n"
        "       --webui-icon <path>     Favicon file (.ico|.png|.svg|.gif|.jpg|.webp);\n"
        "                                also rendered before the brand title in the\n"
        "                                sidebar / topbar\n"
        "       --webui-placeholder <s> Input box placeholder (default 'Type a message…')\n"
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

    // Authoritative date/time injection — see build_authoritative_preamble
    // in this file.  ON by default because most users want the model to
    // trust the wall clock instead of guessing about "today".
    bool        inject_datetime  = true;
    std::string knowledge_cutoff = "2024-10";

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
    bool        verbose        = false;  // engine.verbose(true) — log model raw output

    // webui rebrand
    std::string webui_title    = "Box EasyAI";
    std::string webui_icon;              // optional path to .ico/.png/.svg
    std::string webui_mode     = "modern"; // "modern" (embedded llama-server fork) | "minimal" (inline)
    std::string webui_placeholder = "Type a message…";
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
        else if (s == "--inject-datetime") {
            std::string v = need(i, "--inject-datetime");
            a.inject_datetime = (v == "on" || v == "1" || v == "true" || v == "yes");
        }
        else if (s == "--knowledge-cutoff")          a.knowledge_cutoff = need(i, "--knowledge-cutoff");
        else if (s == "-v" || s == "--verbose")      a.verbose        = true;
        else if (s == "--webui-title")               a.webui_title    = need(i, "--webui-title");
        else if (s == "--webui-icon")                a.webui_icon     = need(i, "--webui-icon");
        else if (s == "--webui-placeholder")         a.webui_placeholder = need(i, "--webui-placeholder");
        else if (s == "--webui")                     a.webui_mode     = need(i, "--webui");
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
        "Answer directly for greetings, chitchat, math, and anything you "
        "already know — do NOT call a tool for those.\n"
        "Use a tool only when the request truly needs one:\n"
        "  - up-to-date / 'today' / 'latest' info → web_search, THEN web_fetch\n"
        "  - the current date/time                → datetime\n"
        "  - reading / listing files              → fs_read_file / fs_list_dir / fs_glob / fs_grep\n"
        "\n"
        "CRITICAL — every rule is mandatory:\n"
        " 1. web_search returns titles + 1-2 sentence snippets. The snippets "
        "    are NOT enough to summarise from. After every web_search you "
        "    MUST immediately call web_fetch on the top 1-3 most relevant "
        "    URLs and base your answer on the fetched body text.\n"
        " 2. Two web_search calls in a row is wrong. Search ONCE, then fetch.\n"
        " 3. NEVER announce a tool call without making it. Phrases like "
        "    \"I will fetch...\", \"let me search...\", \"I'll get...\" are "
        "    forbidden when followed by silence — either invoke the tool in "
        "    the same turn, or write the final answer right away. Saying you "
        "    are going to do something is NOT the same as doing it.\n"
        " 4. If a fetch fails (HTTP 4xx/5xx), retry with the next URL from "
        "    the search results. Do not fall back to summarising snippets.\n"
        " 5. When you cite an article, cite the URL you actually fetched.";

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
    ctx->default_system   = default_system;
    ctx->inject_datetime  = args.inject_datetime;
    ctx->knowledge_cutoff = args.knowledge_cutoff;
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

    // ----- webui rebrand: build the served HTML once and load any custom
    //       favicon into memory.
    {
        std::string title = args.webui_title.empty() ? std::string("easyai")
                                                     : args.webui_title;

#if defined(EASYAI_BUILD_WEBUI)
        if (args.webui_mode != "minimal") {
            // Start from the llama-server-derived index.html and inject:
            //   - a <script> that pins document.title to our --webui-title
            //     even if the Svelte app tries to set it later (MutationObserver)
            //   - a <link rel="icon"> that points to our /favicon route when a
            //     --webui-icon was given
            //   - a <style> that hides MCP-related UI sections
            std::string html(reinterpret_cast<const char *>(index_html),
                              index_html_len);

            // Strip the bundle's hard-coded inline favicon: the embedded
            // index.html ships a giant <link rel="icon" href="data:...">
            // base64 data URL with the orange llama-cpp logo.  We want
            // /favicon (which serves either --webui-icon or our embedded
            // brain SVG) to win, so erase the bundle's link tag entirely.
            // We then unconditionally inject our own link below — this
            // also covers the case where the operator did NOT pass
            // --webui-icon (default brain SVG).
            {
                const std::string needle = "<link rel=\"icon\"";
                size_t p = html.find(needle);
                if (p != std::string::npos) {
                    size_t end = html.find('>', p);
                    if (end != std::string::npos) {
                        html.erase(p, end + 1 - p);
                    }
                }
            }

            std::ostringstream inj;

            // ----- title pin (HARD) -----------------------------------------
            // The bundle has `document.title="llama.cpp - AI Chat Interface"`
            // hard-coded in several places.  A MutationObserver isn't enough
            // because the property is also re-assigned imperatively.  We
            // intercept *every* write to document.title via Object.defineProperty
            // and force our own value back.  Also pre-populates the localStorage
            // flag the bundle uses for "is MCP enabled".
            inj << "<script>(()=>{"
                // Diagnostic marker — if you see all five [easyai-inject]
                // logs in DevTools Console, every <script> block parsed.
                // Missing logs indicate a SyntaxError that killed that block.
                << "console.log('[easyai-inject] block1 title-pin');"
                << "const T=" << json(title).dump() << ";"
                << "try{Object.defineProperty(document,'title',{"
                <<   "configurable:true,"
                <<   "get(){return T;},"
                <<   "set(_){const e=document.querySelector('title');"
                <<        "if(e)e.textContent=T;}});}catch(e){}"
                << "try{localStorage.setItem('LlamaCppWebui.mcpDefaultEnabled','false');}"
                <<   "catch(e){}"
                // Force the bundle's keepStatsVisible flag to true so its
                // native per-message stats line stops flickering off after
                // generation completes.  We still inject our own chip
                // because we want the stats to live alongside copy/edit/...
                << "try{"
                <<   "const c=JSON.parse(localStorage.getItem('LlamaCppWebui.config')||'{}');"
                <<   "c.keepStatsVisible=true;c.showMessageStats=true;"
                <<   "localStorage.setItem('LlamaCppWebui.config',JSON.stringify(c));"
                << "}catch(e){}"
                << "const apply=()=>{const e=document.querySelector('title');"
                <<   "if(e&&e.textContent!==T)e.textContent=T;};"
                << "apply();"
                << "document.addEventListener('DOMContentLoaded',()=>{apply();"
                <<   "new MutationObserver(apply).observe(document.head,"
                <<     "{subtree:true,characterData:true,childList:true});"
                << "});"
                << "})();</script>";

            // ----- runtime DOM scrubber for MCP and other unsupported UI ----
            // The Svelte build produces hashed class names, so [class*=mcp]
            // selectors don't match anything.  Instead, identify offending
            // elements by their visible text (the UI strings are stable) and
            // hide their containing card / list-item / dialog / menu-item.
            inj <<
              "<script>(()=>{"
                "console.log('[easyai-inject] block2 mcp-scrubber');"
                "const NEEDLES=["
                  "/^MCP\\b/i,"
                  "/^MCP Server/i,"
                  "/MCP Servers?$/i,"
                  "/^MCP Prompt/i,"
                  "/^MCP Resource/i,"
                  "/Add (files, )?(system prompt or )?(configure )?MCP/i,"
                  "/No MCP /i,"
                  "/All MCP server connections/i,"
                  "/^Sign in/i,/^Log in$/i,/^Login$/i,"
                  "/^Authorize/i,"
                  "/^Load model/i,/^Unload model/i,/^Manage models/i,"
                  "/^Use Pyodide/i,/^Python interpreter/i"
                "];"
                "const isMatch=(el)=>{"
                  "const t=(el.innerText||el.textContent||'').trim();"
                  "if(!t||t.length>200)return false;"
                  "for(const re of NEEDLES)if(re.test(t))return true;"
                  "return false;"
                "};"
                "const ANCESTORS=["
                  "'[role=\"dialog\"]',"
                  "'[role=\"menuitem\"]',"
                  "'[role=\"tab\"]',"
                  "'[role=\"tabpanel\"]',"
                  "'fieldset',"
                  "'.menu li','.menu-item',"
                  "'li','.collapse','.card','.tab','.tab-content'"
                "];"
                "const hide=(el)=>{"
                  "for(const sel of ANCESTORS){"
                    "const a=el.closest(sel);"
                    "if(a){a.style.setProperty('display','none','important');return;}"
                  "}"
                  "el.style.setProperty('display','none','important');"
                "};"
                "const scrub=()=>{"
                  "document.querySelectorAll("
                    "'h1,h2,h3,h4,label,a,button,summary,[role=\"menuitem\"],[role=\"tab\"]'"
                  ").forEach(e=>{if(isMatch(e))hide(e);});"
                "};"
                "let tries=0;"
                "const tick=()=>{"
                  "scrub();"
                  "if(++tries<60)setTimeout(tick,300);"
                "};"
                "document.addEventListener('DOMContentLoaded',()=>{"
                  "tick();"
                  // keep an observer running for late-mounted dialogs
                  "new MutationObserver(scrub).observe(document.body,"
                    "{childList:true,subtree:true});"
                "});"
              "})();</script>";

            // ----- hide the bundle's native Reasoning panel -----------------
            // The bundle renders reasoning blocks via a custom Svelte
            // component (CollapsibleContentBlock) with the wrapper class
            // "my-2" — NOT as <details><summary>.  We already render our
            // own blue brain-iconed custom panel above the message body,
            // so the bundle's copy is just a duplicate — hide it.
            //
            // Strategy: walk every element that has "my-2" in its class
            // list, look at its trimmed innerText (label + body text),
            // and hide the element when the text starts with "Reasoning"
            // (covers both "Reasoning" closed and "Reasoning…" streaming
            // labels, as well as the expanded body whose first line is
            // still "Reasoning").  Tool-call panels share the same Svelte
            // component but their label is the tool name, so they don't
            // match and stay visible.
            inj <<
              "<script>(()=>{"
                "console.log('[easyai-inject] block3 hide-bundle-reasoning');"
                "const isReasoning=(t)=>{"
                  "if(!t)return false;t=t.trim();"
                  "if(t.length>500)return false;"
                  "return /^Reasoning(?:\\.\\.\\.|\\u2026|\\b)/i.test(t);"
                "};"
                "const hide=()=>{"
                  // First sweep: anything with my-2 (the wrapper class
                  // CollapsibleContentBlock receives).
                  "document.querySelectorAll('[class*=\"my-2\"]').forEach(el=>{"
                    "if(el.dataset.easyaiHidden)return;"
                    "const txt=(el.innerText||el.textContent||'').trim();"
                    "if(!isReasoning(txt))return;"
                    "el.dataset.easyaiHidden='1';"
                    "el.style.setProperty('display','none','important');"
                  "});"
                  // Fallback sweep: any leaf element whose trimmed text
                  // is exactly the Reasoning label.  Walks up to find the
                  // nearest plausible card wrapper and hides that.
                  "document.querySelectorAll('span,div,h1,h2,h3,h4,p').forEach(el=>{"
                    "if(el.children.length)return;"
                    "const t=(el.innerText||el.textContent||'').trim();"
                    "if(!t||!/^Reasoning(\\.\\.\\.|\\u2026)?$/i.test(t))return;"
                    "let p=el;"
                    "for(let i=0;i<8&&p;i++){"
                      "if(p.dataset&&p.dataset.easyaiHidden)return;"
                      "if(p.className&&/\\bmy-2\\b|\\bcollapse\\b|\\bcard\\b/i.test(p.className+'')){"
                        "p.dataset.easyaiHidden='1';"
                        "p.style.setProperty('display','none','important');"
                        "return;"
                      "}"
                      "p=p.parentElement;"
                    "}"
                    // Last-resort: hide just the label so the empty
                    // shell doesn't take vertical space.
                    "if(!el.dataset.easyaiHidden){"
                      "el.dataset.easyaiHidden='1';"
                      "el.style.setProperty('display','none','important');"
                    "}"
                  "});"
                "};"
                // Kept as a no-op so existing call sites in monitorSSE
                // (window.__easyaiCollapseReasoning) keep linking; our
                // custom panel handles its own collapse via closeThinking.
                "window.__easyaiCollapseReasoning=()=>{};"
                "let n=0;"
                "const tick=()=>{hide();if(++n<120)setTimeout(tick,300);};"
                "document.addEventListener('DOMContentLoaded',()=>{"
                  "tick();"
                  "new MutationObserver(hide).observe(document.body,"
                    "{childList:true,subtree:true,characterData:true});"
                "});"
              "})();</script>";

            // ----- fetch interceptor: stub unsupported endpoints AND inject
            //       sampling overrides from the tone dropdown ----------------
            inj <<
              "<script>(()=>{"
                "console.log('[easyai-inject] block4 fetch-interceptor + monitorSSE + thinking-panel');"
                "const orig=window.fetch.bind(window);"
                "const stub=(b,s=200)=>new Response(JSON.stringify(b),{"
                  "status:s,headers:{'Content-Type':'application/json'}"
                "});"
                "const reject=stub({error:{message:'not supported on easyai-server'}},501);"
                "const TONES={"
                  "deterministic:{t:0.0,top_p:1.0,top_k:1},"
                  "precise:{t:0.2,top_p:0.95,top_k:40},"
                  "balanced:{t:0.7,top_p:0.95,top_k:40},"
                  "creative:{t:1.0,top_p:0.95,top_k:40}"
                "};"
                "try{window.__easyaiTone=localStorage.getItem('easyai-tone')||'balanced';}"
                  "catch(e){window.__easyaiTone='balanced';}"
                "window.fetch=async(input,init)=>{"
                  "let url=typeof input==='string'?input:(input&&input.url)||'';"
                  "try{const u=new URL(url,location.origin);url=u.pathname;}catch(e){}"
                  "if(url==='/authorize'||url==='/token'||url==='/register'"
                    "||url.startsWith('/.well-known/')) return reject;"
                  "if(url==='/models/load'||url==='/models/unload') return reject;"
                  "if(url==='/cors-proxy'||url.startsWith('/home/web_user/')"
                    "||url==='/dev/poll') return reject;"
                  "if(url==='/properties') return stub({});"
                  // Inject tone-based sampling when the request body doesn't
                  // already pin those fields, and tee the response so we
                  // can drive the live activity pill from the SSE stream.
                  "if(url.endsWith('/v1/chat/completions')&&init&&init.body){"
                    "try{"
                      "const body=JSON.parse(init.body);"
                      "const t=TONES[window.__easyaiTone];"
                      "if(t){"
                        "if(body.temperature===undefined)body.temperature=t.t;"
                        "if(body.top_p===undefined)body.top_p=t.top_p;"
                        "if(body.top_k===undefined)body.top_k=t.top_k;"
                      "}"
                      "init={...init,body:JSON.stringify(body)};"
                    "}catch(e){}"
                    "const r=await orig(input,init);"
                    "if(r.body&&typeof r.body.tee==='function'){"
                      "const [a,b]=r.body.tee();"
                      "monitorSSE(b);"   // a goes to caller, b to pill
                      "return new Response(a,{headers:r.headers,status:r.status,statusText:r.statusText});"
                    "}"
                    "return r;"
                  "}"
                  "return orig(input,init);"
                "};"

                // SSE → activity pill state machine.
                // ---- per-message thinking panel (ENABLED) ------------------
                // We render our own collapsible reasoning panel into each
                // assistant bubble: blue-accented, brain-iconed, monospace
                // gray body, max-height cap, click to expand.  The bundle's
                // own native "Reasoning" panel is hidden via the observer
                // a few blocks below to avoid duplication.  Toggle the
                // flag below to fall back to the bundle's panel.
                "window.__easyaiCustomThink=true;"
                "const THINK=new WeakMap();"      // msg → currently-open panel
                "const findLastAssistantMsg=()=>{"
                  "const all=document.querySelectorAll("
                    "'[aria-label=\"Assistant message with actions\"]');"
                  "return all.length?all[all.length-1]:null;"
                "};"
                "const ensureThinkPanel=(msg)=>{"
                  "let p=THINK.get(msg);"
                  "if(p&&document.contains(p)&&(p.parentNode===msg||msg.contains(p))){"
                    // Re-opening case: a previous thinking phase ran,
                    // closeThinking() collapsed the panel, then a new
                    // <think>...</think> arrived (e.g. agentic multi-hop
                    // turn).  Re-open the same panel and reset its
                    // summary label so the user sees the new thinking.
                    "if(!p.open){"
                      "p.open=true;"
                      "const lab=p.querySelector('.sumlabel');"
                      "if(lab)lab.textContent='thinking…';"
                    "}"
                    "return p;"
                  "}"
                  // Svelte sometimes re-mounts the assistant bubble during a
                  // generation (state-store flush), giving us a brand-new
                  // `msg` node.  When that happens our WeakMap entry is
                  // stale, but the previous panel may already have been
                  // re-parented into the new msg.  Dedupe by DOM query: if
                  // ANY __easyai-thinking already lives inside the current
                  // msg, reuse it; if MORE than one exist, keep the first
                  // and remove duplicates so the user never sees stacked
                  // empty panels.
                  "const existing=msg.querySelectorAll(':scope .__easyai-thinking');"
                  "if(existing.length>0){"
                    "for(let i=1;i<existing.length;i++)existing[i].remove();"
                    "const e0=existing[0];"
                    // Same re-opening logic as above.
                    "if(!e0.open){"
                      "e0.open=true;"
                      "const lab=e0.querySelector('.sumlabel');"
                      "if(lab)lab.textContent='thinking…';"
                    "}"
                    "THINK.set(msg,e0);"
                    "return e0;"
                  "}"
                  "p=document.createElement('details');"
                  "p.className='__easyai-thinking';"
                  "p.open=true;"
                  // Blue accent (was purple) + brain icon in summary, per
                  // user request 2026-04-26.  Open during streaming, the
                  // SSE handler closes it on first content delta or finish.
                  "p.style.cssText="
                    "'margin:.5rem 0;padding:.35rem .55rem;"
                    " border-left:2px solid #5b8dee;"
                    " background:rgba(91,141,238,.06);"
                    " border-radius:4px;"
                    " font-size:.7rem;color:#8b949e;"
                    " font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
                    " line-height:1.5;';"
                  "p.innerHTML="
                    "'<summary style=\"cursor:pointer;color:#5b8dee;"
                      "font-weight:600;font-size:.72rem;list-style:none;"
                      "outline:none;user-select:none;display:flex;"
                      "align-items:center;gap:.35rem\">"
                      "<span style=\"font-size:.9rem;line-height:1\">🧠</span>"
                      "<span class=\"sumlabel\">thinking…</span>"
                    "</summary>'"
                    "+'<div class=\"t\" style=\"margin-top:.4rem;"
                      "white-space:pre-wrap;max-height:18em;"
                      "overflow-y:auto;padding-right:.3rem\"></div>';"
                  "msg.insertBefore(p,msg.firstChild);"
                  "THINK.set(msg,p);"
                  "return p;"
                "};"
                "function appendThinking(text){"
                  "if(!window.__easyaiCustomThink)return;"
                  "const msg=findLastAssistantMsg();"
                  "if(!msg)return;"
                  "const p=ensureThinkPanel(msg);"
                  "const t=p.querySelector('.t');"
                  "if(t){t.textContent+=text;t.scrollTop=t.scrollHeight;}"
                "};"
                "function closeThinking(){"
                  "if(!window.__easyaiCustomThink)return;"
                  "const msg=findLastAssistantMsg();"
                  "if(!msg)return;"
                  "const p=THINK.get(msg);"
                  "if(!p)return;"
                  "p.open=false;"
                  "const lab=p.querySelector('.sumlabel');"
                  "if(lab)lab.textContent='thinking';"
                  "THINK.delete(msg);"
                "};"
                "async function monitorSSE(stream){"
                  "const set=window.__easyaiSetStatus||(()=>{});"
                  "set('answering');"
                  "const reader=stream.getReader();"
                  "const dec=new TextDecoder();"
                  "let buf='',inThink=false,inToolCallTag=false,startMs=performance.now();"
                  "let lastTimings=null;"
                  // Live metrics: count emitted chunks (close enough to
                  // tokens for UX) and clock the elapsed time so the chip
                  // and the bar can show "234 tok · 4.2s · 55 t/s" while
                  // generation is still in flight.  Updated on every
                  // delta below; rendered through liveTick().
                  "let liveTok=0,liveStart=0;"
                  "const liveExtra=()=>{"
                    "const elapsed=Math.max(1,performance.now()-liveStart);"
                    "const tps=liveTok/(elapsed/1000);"
                    "return{"
                      "tokens:liveTok,"
                      "elapsedMs:elapsed,"
                      "tps:tps,"
                      "live:true,"
                    "};"
                  "};"
                  "let lastState='answering';"
                  "const liveTick=setInterval(()=>{"
                    "if(liveTok>0&&lastState!=='complete'&&lastState!=='error')"
                      "set(lastState,liveExtra());"
                  "},200);"
                  "const setLive=(s,e)=>{lastState=s;set(s,e);};"
                  "try{while(true){"
                    "const{value,done}=await reader.read();"
                    "if(done)break;"
                    "buf+=dec.decode(value,{stream:true});"
                    "while(true){"
                      "const idx=buf.indexOf('\\n\\n');"
                      "if(idx<0)break;"
                      "const ev=buf.slice(0,idx);buf=buf.slice(idx+2);"
                      "let evtType='message';const dataLines=[];"
                      "for(let line of ev.split('\\n')){"
                        "if(line.endsWith('\\r'))line=line.slice(0,-1);"
                        "if(line.startsWith('event:'))evtType=line.slice(6).trim();"
                        "else if(line.startsWith('data:'))dataLines.push(line.slice(5).replace(/^ /,''));"
                      "}"
                      "const data=dataLines.join('\\n');"
                      "if(!data||data==='[DONE]')continue;"
                      "if(evtType==='easyai.tool_call'){"
                        "try{const j=JSON.parse(data);setLive('fetching',j.name||'tool');}"
                        "catch(e){setLive('fetching');}"
                        "continue;"
                      "}"
                      "if(evtType==='easyai.tool_result'){"
                        // Back to whatever generation phase we're in.
                        "setLive(inThink?'thinking':'answering',liveExtra());"
                        "continue;"
                      "}"
                      "try{"
                        "const j=JSON.parse(data);"
                        "if(j.timings){lastTimings=j.timings;"
                        "if(window.__easyaiPushTimings)window.__easyaiPushTimings(j.timings);}"
                        "const ch=j.choices&&j.choices[0];"
                        "if(ch&&ch.delta){"
                          // Any reasoning or content delta = a new chunk.
                          // Counting CHUNKS (not characters) is closer to
                          // token count for UX purposes — the model emits
                          // ~one chunk per token in the streaming path.
                          "if(liveStart===0)liveStart=performance.now();"
                          // -- THINKING PANEL --------------------------------
                          "if(typeof ch.delta.reasoning_content==='string'){"
                            "liveTok++;"
                            "appendThinking(ch.delta.reasoning_content);"
                            "setLive('thinking',liveExtra());"
                          "}"
                          // -- VISIBLE CONTENT -------------------------------
                          "if(typeof ch.delta.content==='string'){"
                            "const c=ch.delta.content;"
                            "liveTok++;"
                            "if(c.length>0)closeThinking();"
                            "if(!inThink&&/<think(?:ing)?>/i.test(c)){inThink=true;setLive('thinking',liveExtra());}"
                            "else if(inThink&&/<\\/think(?:ing)?>/i.test(c)){inThink=false;setLive('answering',liveExtra());}"
                            "else if(!inThink)setLive('answering',liveExtra());"
                          "}"
                          // Push our own synthetic timings so the bar's
                          // 'last' field tracks live tok/s mid-stream.
                          "if(window.__easyaiPushTimings){"
                            "const now=performance.now();"
                            "if(liveStart>0)window.__easyaiPushTimings({"
                              "predicted_n:liveTok,"
                              "predicted_ms:now-liveStart,"
                              "live:true,"
                            "});"
                          "}"
                        "}"
                        "if(ch&&ch.finish_reason){"
                          "clearInterval(liveTick);"
                          "closeThinking();"
                          "if(window.__easyaiCollapseReasoning)window.__easyaiCollapseReasoning();"
                          "const t=lastTimings;"
                          "let msg='';"
                          "if(t&&t.predicted_n&&t.predicted_ms){"
                            "const tps=(t.predicted_n/(t.predicted_ms/1000)).toFixed(1);"
                            "msg=t.predicted_n+' tok · '+(t.predicted_ms/1000).toFixed(1)+'s · '+tps+' t/s';"
                          "}"
                          "setLive('complete',msg);"
                        "}"
                      "}catch(e){}"
                    "}"
                  "}}catch(e){clearInterval(liveTick);setLive('error');}"
                "}"
              "})();</script>";

            // ----- live metrics bar + tone chip + per-message status -------
            //   Metrics bar — its own Shadow DOM host, anchored ABOVE the
            //                 prompt textarea.  Live-updates ctx / token
            //                 count / elapsed / t/s while the model is
            //                 streaming, then freezes on finish_reason.
            //   Tone chip   — separate Shadow DOM host, anchored at the
            //                 right edge of the prompt form so it sits
            //                 visually next to the model-name badge and
            //                 the send arrow.  Both hosts use fixed
            //                 positioning anchored via getBoundingClientRect
            //                 of the form so layout follows the bundle.
            //   Per-msg     — for each assistant message, we append a small
            //   status        live status indicator inline next to the
            //                 copy/edit/fork/delete actions; chip text is
            //                 updated by window.__easyaiSetStatus.
            inj <<
              "<script>(()=>{"
                "console.log('[easyai-inject] block5 metrics-bar + tone-badge + chip');"

                // --- shared style snippets ---
                "const SHARED_STYLE='"
                  ":host,*{box-sizing:border-box}"
                  ".pill{pointer-events:auto;display:inline-flex;"
                    "align-items:center;gap:.55rem;"
                    "background:rgba(15,19,24,.92);"
                    "border:1px solid #2a313b;border-radius:999px;"
                    "padding:.22rem .7rem;color:#8b949e;"
                    "font:.72rem -apple-system,system-ui,sans-serif;"
                    "backdrop-filter:blur(6px);"
                    "box-shadow:0 2px 12px rgba(0,0,0,.35);}"
                  ".grp{display:inline-flex;align-items:center;gap:.3rem}"
                  ".sep{color:#3a414b}"
                  ".v{color:#e6edf3}"
                  ".mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.7rem}"
                  "select{background:transparent;color:#e6edf3;border:0;"
                    "font:inherit;cursor:pointer;outline:none;padding:0 .15rem}"
                  "select option{background:#15191f;color:#e6edf3}"
                  "';"

                // --- METRICS BAR (above textarea) ---
                "const BAR_ID='__easyaiBarHost';"
                "let barRoot=null,barHost=null;"
                "const ensureBar=()=>{"
                  "let h=document.getElementById(BAR_ID);"
                  "if(h&&barRoot){barHost=h;return barRoot;}"
                  "if(!document.documentElement)return null;"
                  "h=document.createElement('div');"
                  "h.id=BAR_ID;"
                  "h.setAttribute('aria-hidden','false');"
                  "h.style.cssText="
                    "'all:initial;position:fixed;top:0px;left:0px;"
                    "z-index:2147483646;pointer-events:none;"
                    "font:14px/1 -apple-system,system-ui,sans-serif;';"
                  "document.documentElement.appendChild(h);"
                  "barHost=h;"
                  "const root=h.attachShadow({mode:'open'});"
                  "root.innerHTML='<style>'+SHARED_STYLE+'</style>'+"
                    "'<div class=\"pill\">"
                      "<span class=\"grp mono\">"
                        "<span>ctx</span><span class=\"v ctx\">—</span>"
                      "</span>"
                      "<span class=\"sep\">·</span>"
                      "<span class=\"grp mono\">"
                        "<span>last</span><span class=\"v last\">—</span>"
                      "</span>"
                    "</div>';"
                  "barRoot=root;"
                  "return root;"
                "};"

                // --- TONE BADGE (same look as the bundle's model badge,
                //                 anchored at bottom-LEFT of the form, same
                //                 vertical height as the model badge) ----
                "const TONE_ID='__easyaiToneHost';"
                "let toneRoot=null,toneHost=null;"
                "const ensureTone=()=>{"
                  "let h=document.getElementById(TONE_ID);"
                  "if(h&&toneRoot){toneHost=h;return toneRoot;}"
                  "if(!document.documentElement)return null;"
                  "h=document.createElement('div');"
                  "h.id=TONE_ID;"
                  "h.setAttribute('aria-hidden','false');"
                  "h.style.cssText="
                    "'all:initial;position:fixed;top:0px;left:0px;"
                    "z-index:2147483646;pointer-events:none;"
                    "font:14px/1 -apple-system,system-ui,sans-serif;';"
                  "document.documentElement.appendChild(h);"
                  "toneHost=h;"
                  "const root=h.attachShadow({mode:'open'});"
                  // Match the bundle model-badge look: rounded rectangle
                  // (not full pill), subtle dark fill, thin border, small
                  // icon + label.  Slider-knob icon for "tone".
                  // NOTE: backticks (JS template literal) on the outer
                  // strings — the inline SVG inside the CSS background-image
                  // contains xmlns='http://...' (single quotes), and the
                  // <label class="badge"> uses double quotes.  Mixing both
                  // inside a JS string requires backticks; otherwise the
                  // first inner quote terminates the literal early and the
                  // parser explodes on the bare `http` identifier (which
                  // is exactly what was killing the tone-badge IIFE).
                  "root.innerHTML="
                    "`<style>"
                      ":host,*{box-sizing:border-box}"
                      ".badge{pointer-events:auto;display:inline-flex;"
                        "align-items:center;gap:.4rem;"
                        "background:rgba(30,33,38,.85);"
                        "border:1px solid #2a313b;border-radius:.5rem;"
                        "padding:.32rem .6rem;color:#c9d1d9;"
                        "font:.78rem -apple-system,system-ui,sans-serif;"
                        "cursor:pointer;"
                        "transition:background .15s ease;"
                      "}"
                      ".badge:hover{background:rgba(45,49,55,.9)}"
                      ".badge svg{width:14px;height:14px;flex-shrink:0;"
                        "stroke:#8b949e;stroke-width:2;fill:none;"
                        "stroke-linecap:round;stroke-linejoin:round}"
                      ".badge select{background:transparent;color:inherit;"
                        "border:0;font:inherit;cursor:pointer;outline:none;"
                        "padding:0;appearance:none;-webkit-appearance:none;"
                        "padding-right:.9rem;"
                        "background-image:url(\"data:image/svg+xml;utf8,"
                          "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 6' fill='none'"
                          " stroke='%238b949e' stroke-width='1.5' stroke-linecap='round'"
                          " stroke-linejoin='round'><polyline points='1 1 5 5 9 1'/></svg>\");"
                        "background-repeat:no-repeat;background-position:right center;"
                        "background-size:.55rem auto}"
                      ".badge select option{background:#15191f;color:#e6edf3}"
                    "</style>`+"
                    "`<label class=\"badge\">"
                      // Sliders icon (tabler-ish): two horizontal rails
                      // with knob handles, matches the geometric flat look
                      // of the bundle's model badge cube icon.
                      "<svg viewBox=\"0 0 24 24\">"
                        "<line x1=\"4\" y1=\"7\" x2=\"20\" y2=\"7\"/>"
                        "<line x1=\"4\" y1=\"17\" x2=\"20\" y2=\"17\"/>"
                        "<circle cx=\"10\" cy=\"7\" r=\"2.4\" fill=\"#0d1117\"/>"
                        "<circle cx=\"15\" cy=\"17\" r=\"2.4\" fill=\"#0d1117\"/>"
                      "</svg>"
                      "<select aria-label=\"tone\">"
                        "<option value=\"deterministic\">deterministic</option>"
                        "<option value=\"precise\">precise</option>"
                        "<option value=\"balanced\">balanced</option>"
                        "<option value=\"creative\">creative</option>"
                      "</select>"
                    "</label>`;"
                  "const sel=root.querySelector('select');"
                  "sel.value=window.__easyaiTone||'balanced';"
                  "sel.onchange=()=>{"
                    "window.__easyaiTone=sel.value;"
                    "try{localStorage.setItem('easyai-tone',sel.value);}catch(e){}"
                  "};"
                  "toneRoot=root;"
                  "return root;"
                "};"

                // --- TOOLS BADGE (sibling of tone, click → popover with
                //                  catalog + per-tool descriptions) ----
                "const TOOLS_ID='__easyaiToolsHost';"
                "let toolsRoot=null,toolsHost=null;"
                "const ensureTools=()=>{"
                  "let h=document.getElementById(TOOLS_ID);"
                  "if(h&&toolsRoot){toolsHost=h;return toolsRoot;}"
                  "if(!document.documentElement)return null;"
                  "h=document.createElement('div');"
                  "h.id=TOOLS_ID;"
                  "h.setAttribute('aria-hidden','false');"
                  "h.style.cssText="
                    "'all:initial;position:fixed;top:0px;left:0px;"
                    "z-index:2147483646;pointer-events:none;"
                    "font:14px/1 -apple-system,system-ui,sans-serif;';"
                  "document.documentElement.appendChild(h);"
                  "toolsHost=h;"
                  "const root=h.attachShadow({mode:'open'});"
                  "root.innerHTML="
                    "`<style>"
                      ":host,*{box-sizing:border-box}"
                      ".badge{pointer-events:auto;display:inline-flex;"
                        "align-items:center;gap:.4rem;"
                        "background:rgba(30,33,38,.85);"
                        "border:1px solid #2a313b;border-radius:.5rem;"
                        "padding:.32rem .6rem;color:#c9d1d9;"
                        "font:.78rem -apple-system,system-ui,sans-serif;"
                        "cursor:pointer;"
                        "transition:background .15s ease;"
                        "user-select:none;"
                      "}"
                      ".badge:hover{background:rgba(45,49,55,.9)}"
                      ".badge svg{width:14px;height:14px;flex-shrink:0;"
                        "stroke:#8b949e;stroke-width:2;fill:none;"
                        "stroke-linecap:round;stroke-linejoin:round}"
                      ".count{color:#8b949e;font-size:.7rem}"
                      ".pop{position:absolute;bottom:calc(100% + .35rem);"
                        "left:0;display:none;pointer-events:auto;"
                        "min-width:14rem;max-width:22rem;"
                        "background:#0f1318;border:1px solid #2a313b;"
                        "border-radius:.5rem;padding:.45rem .5rem;"
                        "box-shadow:0 6px 24px rgba(0,0,0,.5);"
                        "color:#c9d1d9;font-size:.72rem;"
                        "max-height:60vh;overflow-y:auto}"
                      ".pop.open{display:block}"
                      ".pop .row{padding:.32rem .35rem;border-radius:.3rem;"
                        "cursor:default;display:flex;flex-direction:column;"
                        "gap:.15rem;line-height:1.35}"
                      ".pop .row+.row{border-top:1px solid #1f242b}"
                      ".pop .row:hover{background:rgba(91,141,238,.08)}"
                      ".pop .name{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
                        "font-size:.72rem;color:#5b8dee;font-weight:600}"
                      ".pop .desc{color:#8b949e;font-size:.68rem;"
                        "white-space:pre-wrap;line-height:1.4}"
                      ".pop .empty{color:#8b949e;padding:.4rem .35rem}"
                    "</style>"
                    "<div style=\"position:relative;display:inline-block\">"
                      "<label class=\"badge\" data-trigger=\"1\">"
                        "<svg viewBox=\"0 0 24 24\">"
                          // Wrench icon
                          "<path d=\"M14.7 6.3a3.5 3.5 0 0 1 4.6 4.6l-2.8-2.8-1.8 1.8 2.8 2.8a3.5 3.5 0 0 1-4.6-4.6\"/>"
                          "<path d=\"M5 19l8.5-8.5\"/>"
                        "</svg>"
                        "<span>tools</span>"
                        "<span class=\"count\"></span>"
                      "</label>"
                      "<div class=\"pop\" role=\"dialog\" aria-label=\"available tools\"></div>"
                    "</div>`;"
                  "const badge=root.querySelector('.badge');"
                  "const pop=root.querySelector('.pop');"
                  "const cnt=root.querySelector('.count');"
                  "const renderList=(tools)=>{"
                    "if(!tools||!tools.length){"
                      "pop.innerHTML='<div class=\"empty\">no tools registered</div>';"
                      "cnt.textContent='0';return;"
                    "}"
                    "cnt.textContent=String(tools.length);"
                    "let html='';"
                    "for(const t of tools){"
                      "const name=(t.name||'').replace(/[<>&]/g,c=>"
                        "c==='<'?'&lt;':c==='>'?'&gt;':'&amp;');"
                      "const desc=(t.description||'').replace(/[<>&]/g,c=>"
                        "c==='<'?'&lt;':c==='>'?'&gt;':'&amp;');"
                      "html+='<div class=\"row\" title=\"'+desc+'\">'+"
                        "'<span class=\"name\">'+name+'</span>'+"
                        "'<span class=\"desc\">'+desc+'</span>'+"
                      "'</div>';"
                    "}"
                    "pop.innerHTML=html;"
                  "};"
                  "renderList([]);"
                  "fetch('/v1/tools').then(r=>r.json()).then(j=>{"
                    "renderList((j&&j.data)||[]);"
                  "}).catch(()=>{renderList([]);});"
                  "badge.addEventListener('click',(e)=>{"
                    "e.preventDefault();e.stopPropagation();"
                    "pop.classList.toggle('open');"
                  "});"
                  // Close on outside click.
                  "document.addEventListener('click',(e)=>{"
                    "if(!toolsHost)return;"
                    "if(!toolsHost.contains(e.target))pop.classList.remove('open');"
                  "});"
                  "toolsRoot=root;"
                  "return root;"
                "};"

                // Anchor both hosts via getBoundingClientRect of the
                // prompt form.  The metrics bar sits ~36px above the form;
                // the tone chip sits at the form's bottom-right (where the
                // model name badge + send arrow live in the bundle).
                "const reposition=()=>{"
                  "if(!barHost||!toneHost)return;"
                  // Try to find the prompt textarea.  Fall back to fixed
                  // viewport-bottom positions if anything looks wrong, so
                  // both hosts ALWAYS show up — better a slightly wrong
                  // position than an invisible one (the previous bug).
                  "const ta=document.querySelector('textarea');"
                  "let r=null;"
                  "if(ta){"
                    "const form=ta.closest('form')||ta.parentElement;"
                    "if(form){"
                      "const rr=form.getBoundingClientRect();"
                      // Sanity-check: form must be within the viewport,
                      // wider than 60 px.  Otherwise this is some hidden
                      // scratch element.
                      "if(rr.width>60&&rr.height>20"
                         "&&rr.bottom>0&&rr.top<window.innerHeight){"
                        "r=rr;"
                      "}"
                    "}"
                  "}"
                  "if(r){"
                    // Metrics bar — above the form, centered horizontally.
                    // CRITICAL: clear `bottom` from any prior fallback tick;
                    // setting top+bottom simultaneously stretches the host
                    // to 0 effective height and the inner pill disappears.
                    "barHost.style.bottom='';"
                    "barHost.style.top=(Math.max(4,r.top-34))+'px';"
                    "barHost.style.left=(r.left+r.width/2)+'px';"
                    "barHost.style.transform='translateX(-50%)';"
                    // Tone badge — anchored at form's bottom-left, same
                    // y as the bundle's model badge if we can find it.
                    "let badgeY=r.bottom-26;"
                    "const cand=document.querySelectorAll("
                      "'[class*=badge i],[data-testid*=model i],"
                      "[aria-label*=model i]'"
                    ");"
                    "for(const el of cand){"
                      "const t=(el.innerText||el.textContent||'').trim();"
                      "if(!t||t.length>40)continue;"
                      "const er=el.getBoundingClientRect();"
                      "if(er.bottom>r.top+r.height/2&&er.bottom<r.bottom+10){"
                        "badgeY=er.top+er.height/2;break;"
                      "}"
                    "}"
                    "toneHost.style.bottom='';"
                    "toneHost.style.top=badgeY+'px';"
                    "toneHost.style.left=(r.left+12)+'px';"
                    "toneHost.style.transform='translateY(-50%)';"
                    // Tools badge sits to the right of the tone badge.
                    // We measure tone's rendered width (its shadow root
                    // exposes the .badge element via the host bbox)
                    // and offset by that + a small gap.
                    "if(toolsHost){"
                      "let toneWidth=120;"
                      "const tBox=toneHost.getBoundingClientRect();"
                      "if(tBox.width>20)toneWidth=tBox.width;"
                      "toolsHost.style.bottom='';"
                      "toolsHost.style.top=badgeY+'px';"
                      "toolsHost.style.left=(r.left+12+toneWidth+8)+'px';"
                      "toolsHost.style.transform='translateY(-50%)';"
                    "}"
                  "}else{"
                    // Fallback: viewport-anchored bottom strip.  Same
                    // bottom/top swap concern: clear top before setting
                    // bottom or the host gets stuck offscreen.
                    "barHost.style.top='';"
                    "barHost.style.bottom='6.5rem';"
                    "barHost.style.left='50%';"
                    "barHost.style.transform='translateX(-50%)';"
                    "toneHost.style.top='';"
                    "toneHost.style.bottom='1rem';"
                    "toneHost.style.left='1rem';"
                    "toneHost.style.transform='none';"
                    "if(toolsHost){"
                      "toolsHost.style.top='';"
                      "toolsHost.style.bottom='1rem';"
                      "toolsHost.style.left='9rem';"
                      "toolsHost.style.transform='none';"
                    "}"
                  "}"
                "};"

                "if(document.documentElement){ensureBar();ensureTone();ensureTools();reposition();}"
                "document.addEventListener('DOMContentLoaded',()=>{"
                  "ensureBar();ensureTone();ensureTools();reposition();"
                "});"
                "window.addEventListener('resize',reposition);"
                "window.addEventListener('scroll',reposition,true);"
                "setInterval(()=>{"
                  "if(!document.getElementById(BAR_ID)){barRoot=null;ensureBar();}"
                  "if(!document.getElementById(TONE_ID)){toneRoot=null;ensureTone();}"
                  "if(!document.getElementById(TOOLS_ID)){toolsRoot=null;ensureTools();}"
                  "reposition();"
                "},250);"

                // --- per-message inline status chip -----------------------
                // For every <... aria-label='Assistant message with actions'>
                // we attach a small status indicator inside the action row
                // (next to copy/edit/fork/delete).  The chip is updated by
                // window.__easyaiSetStatus(state, extra) called from the
                // SSE monitor below.  Bundle re-renders are handled by the
                // MutationObserver re-attaching when the chip is missing.
                "const CHIP_MARK='__eaiChip';"
                "const STATE_DOT={"
                  "thinking:'#b97df3',answering:'#1f6feb',"
                  "fetching:'#4fb0ff',error:'#f85149',"
                  "complete:'#3fb950',idle:'#5b626a'"
                "};"
                "const findActionRow=(msg)=>{"
                  // The action toolbar is the parent whose DIRECT children
                  // are buttons (>=2). Walk depth-first and pick the
                  // deepest match so we don't accidentally land on the
                  // whole message wrapper (which has lots of descendant
                  // buttons via code-block 'copy code' etc.).
                  "let best=null,bestDepth=-1;"
                  "const walk=(el,d)=>{"
                    "let direct=0;"
                    "for(const c of el.children){"
                      "if(c.tagName==='BUTTON'||c.getAttribute&&c.getAttribute('role')==='button')direct++;"
                    "}"
                    "if(direct>=2&&d>bestDepth){best=el;bestDepth=d;}"
                    "for(const c of el.children)walk(c,d+1);"
                  "};"
                  "walk(msg,0);"
                  "return best;"
                "};"
                // Inject a stylesheet ONCE for the pulsing dot keyframes.
                // Using a single global rule beats inlining @keyframes per
                // chip and lets us toggle the animation just by adding /
                // removing the .pulse class on the dot.
                "if(!document.getElementById('__easyaiChipStyles')){"
                  "const st=document.createElement('style');"
                  "st.id='__easyaiChipStyles';"
                  "st.textContent="
                    "'@keyframes __easyaiPulse{"
                      "0%,100%{opacity:1}"
                      "50%{opacity:.25}"
                    "}"
                    ".__easyaiDot.pulse{"
                      "animation:__easyaiPulse 1.05s ease-in-out infinite}';"
                  "document.head&&document.head.appendChild(st);"
                "}"
                // Build a chip element with the inline action-row look.
                // Default visible so user sees something the moment SSE
                // starts; setStatus updates the label/dot.  setStatus(idle)
                // hides it explicitly.
                "const buildChip=()=>{"
                  "const chip=document.createElement('span');"
                  "chip.style.cssText="
                    "'display:inline-flex;align-items:center;gap:.35rem;'+"
                    "'align-self:center;vertical-align:middle;line-height:1;'+"
                    "'margin:0 0 0 .5rem;padding:.2rem .65rem;'+"
                    "'border:1px solid #2a313b;border-radius:999px;'+"
                    "'color:#c9d1d9;background:rgba(20,26,32,.85);'+"
                    "'font:.72rem -apple-system,system-ui,sans-serif;'+"
                    "'flex-shrink:0;white-space:nowrap;'+"
                    "'box-shadow:0 1px 3px rgba(0,0,0,.3);';"
                  "chip.innerHTML="
                    "'<span class=\"d __easyaiDot\" style=\"width:.5rem;height:.5rem;"
                      "border-radius:50%;background:#5b8dee;flex-shrink:0\"></span>"
                    "<span class=\"l\">starting…</span>';"
                  "return chip;"
                "};"
                // attachChip(msg): place the chip in the best available
                // anchor.  During streaming the bundle has NOT yet rendered
                // the copy/edit/fork/delete row, so findActionRow returns
                // null — in that case overlay the chip on the assistant
                // bubble using position:absolute (top-right, anchored to
                // msg whose position becomes relative).  That guarantees
                // visibility regardless of the bubble's internal layout.
                // Once a real action row appears, we migrate the chip
                // into it inline and drop the absolute positioning.
                "const attachChip=(msg)=>{"
                  "let chip=msg[CHIP_MARK];"
                  "const row=findActionRow(msg);"
                  "if(chip&&document.contains(chip)){"
                    "if(row&&chip.dataset.fallback==='1'&&!row.contains(chip)){"
                      "chip.style.position='static';"
                      "chip.style.top='';chip.style.right='';"
                      "chip.style.margin='0 0 0 .5rem';"
                      "row.appendChild(chip);"
                      "delete chip.dataset.fallback;"
                    "}"
                    "return;"
                  "}"
                  "chip=buildChip();"
                  "if(row){"
                    "chip.style.margin='0 0 0 .5rem';"
                    "row.appendChild(chip);"
                  "}else{"
                    "chip.dataset.fallback='1';"
                    // Anchor the bubble for absolute positioning if it
                    // doesn't already have a positioning context.
                    "const cs=getComputedStyle(msg);"
                    "if(cs.position==='static')msg.style.position='relative';"
                    "chip.style.position='absolute';"
                    "chip.style.top='.4rem';"
                    "chip.style.right='.6rem';"
                    "chip.style.margin='0';"
                    "chip.style.zIndex='10';"
                    "msg.appendChild(chip);"
                  "}"
                  "msg[CHIP_MARK]=chip;"
                "};"
                "const scanMessages=()=>{"
                  "document.querySelectorAll("
                    "'[aria-label=\"Assistant message with actions\"]'"
                  ").forEach(attachChip);"
                "};"
                "const mo=new MutationObserver(scanMessages);"
                "if(document.body)mo.observe(document.body,{childList:true,subtree:true});"
                "document.addEventListener('DOMContentLoaded',()=>{"
                  "scanMessages();"
                  "mo.observe(document.body,{childList:true,subtree:true});"
                "});"

                // Public helper: drives the inline chip on the LATEST
                // assistant message (the one currently streaming or just
                // finished).  After completion the chip stays visible with
                // the stats text, so the user always sees per-response
                // performance numbers next to copy/edit/fork/delete.
                "window.__easyaiSetStatus=(state,extra)=>{"
                  "scanMessages();"
                  "const all=document.querySelectorAll("
                    "'[aria-label=\"Assistant message with actions\"]'"
                  ");"
                  "if(!all.length)return;"
                  "const last=all[all.length-1];"
                  "const chip=last[CHIP_MARK];"
                  "if(!chip)return;"
                  "if(state==='idle'){"
                    "chip.style.display='none';"
                    "return;"
                  "}"
                  "chip.style.display='inline-flex';"
                  "const dot=chip.querySelector('.d');"
                  "const lab=chip.querySelector('.l');"
                  "if(dot){"
                    "dot.style.background=STATE_DOT[state]||STATE_DOT.idle;"
                    // Pulse while we're actively working; stop on
                    // terminal states so the user sees the difference at
                    // a glance.
                    "const active=(state==='thinking'||state==='answering'||state==='fetching');"
                    "if(active)dot.classList.add('pulse');"
                    "else dot.classList.remove('pulse');"
                  "}"
                  "if(lab){"
                    "let txt=state;"
                    "if(state==='complete'&&typeof extra==='string'&&extra){"
                      "txt=extra;"
                    "}else if(state==='fetching'&&typeof extra==='string'&&extra){"
                      "txt='fetching · '+extra;"
                    "}else if(extra&&typeof extra==='object'&&extra.live){"
                      // Live metrics: state + tok count + elapsed + t/s.
                      "const sec=(extra.elapsedMs/1000).toFixed(1);"
                      "const tps=extra.tps?extra.tps.toFixed(1):'0.0';"
                      "txt=state+' · '+extra.tokens+' tok · '+sec+'s · '+tps+' t/s';"
                    "}else if(state==='fetching'&&extra&&extra.tokens){"
                      "txt='fetching';"
                    "}"
                    "lab.textContent=txt;"
                  "}"
                "};"

                // --- overview values inside the combined bar ----------------
                // Fetch n_ctx once at boot so we can render Context: X/Y%
                // even before the first reply.  Then monitorSSE feeds us
                // timings via __easyaiPushTimings, and we paint into the
                // .ctx and .last spans of the bar.
                "let lastTimings=null;"
                "let lastSessionUsed=0;"   // sticky cumulative ctx used
                "window.__easyaiNCtx=0;"
                "fetch('/props').then(r=>r.json()).then(j=>{"
                  "window.__easyaiNCtx="
                    "(j.default_generation_settings&&j.default_generation_settings.n_ctx)||0;"
                  "renderOverview();"
                "}).catch(()=>{});"
                // Soft + hard thresholds for the context-window guard.
                // Soft (>=85%): bar turns amber as a warning.
                // Hard (>=98%): bar turns red AND we disable the textarea
                //               + Send button so the user can't fire a
                //               request that would overflow.  The user
                //               clears the conversation (or shrinks ctx)
                //               via the bundle's own UI to reset.
                "const CTX_SOFT=0.85,CTX_HARD=0.98;"
                "const setInputLocked=(locked,reason)=>{"
                  "const ta=document.querySelector('textarea');"
                  "if(!ta)return;"
                  "if(locked){"
                    "window.__easyaiCtxFull=true;"
                    "if(ta.dataset.easyaiLockedReason===reason)return;"
                    "ta.dataset.easyaiLockedReason=reason;"
                    "ta.dataset.easyaiPrevPlaceholder=ta.placeholder||'';"
                    "ta.dataset.easyaiPrevBorder=ta.style.border||'';"
                    "ta.dataset.easyaiPrevOpacity=ta.style.opacity||'';"
                    "ta.placeholder=reason;"
                    "ta.disabled=true;"
                    "ta.style.border='1px solid #f85149';"
                    "ta.style.opacity='.55';"
                    "const form=ta.closest('form');"
                    "if(form){"
                      "form.querySelectorAll('button').forEach(b=>{"
                        "if(b.type==='submit'||/send|enviar/i.test(b.getAttribute('aria-label')||''))"
                          "b.disabled=true;"
                      "});"
                    "}"
                  "}else if(ta.dataset.easyaiLockedReason){"
                    "window.__easyaiCtxFull=false;"
                    "ta.placeholder=ta.dataset.easyaiPrevPlaceholder||'';"
                    "ta.style.border=ta.dataset.easyaiPrevBorder||'';"
                    "ta.style.opacity=ta.dataset.easyaiPrevOpacity||'';"
                    "delete ta.dataset.easyaiLockedReason;"
                    "delete ta.dataset.easyaiPrevPlaceholder;"
                    "delete ta.dataset.easyaiPrevBorder;"
                    "delete ta.dataset.easyaiPrevOpacity;"
                    "ta.disabled=false;"
                    "const form=ta.closest('form');"
                    "if(form)form.querySelectorAll('button[disabled]').forEach(b=>{b.disabled=false;});"
                  "}"
                "};"
                "function renderOverview(){"
                  "const r=ensureBar();if(!r)return;"
                  "const ctxEl=r.querySelector('.ctx');"
                  "const lastEl=r.querySelector('.last');"
                  "if(!ctxEl||!lastEl)return;"
                  "const t=lastTimings;"
                  // Cumulative session usage:
                  //   Prefer ctx_used (post-turn KV fill from the engine).
                  //   Fall back to cache_n + prompt_n + predicted_n only
                  //   if the server didn't send ctx_used (older build).
                  //   During live streaming, show lastSessionUsed PLUS
                  //   the live token count so the user sees the bar move
                  //   while generation runs — snaps to the real value
                  //   when the final timings arrive at finish_reason.
                  "let used=lastSessionUsed;"
                  "if(t&&typeof t.ctx_used==='number'&&t.ctx_used>0){"
                    "used=t.ctx_used;lastSessionUsed=used;"
                  "}else if(t&&!t.live){"
                    "const u=(t.cache_n||0)+(t.prompt_n||0)+(t.predicted_n||0);"
                    "if(u>used){used=u;lastSessionUsed=used;}"
                  "}else if(t&&t.live&&typeof t.predicted_n==='number'){"
                    // Live overlay: previous-session base + this turn's
                    // running token count.  Doesn't account for the new
                    // turn's prompt_n (we won't know that until finish),
                    // so we visually under-report by prompt_n during
                    // streaming.  Acceptable trade-off for live motion.
                    "used=lastSessionUsed+t.predicted_n;"
                  "}"
                  "const total=(t&&t.n_ctx)||window.__easyaiNCtx||0;"
                  "if(total&&!window.__easyaiNCtx)window.__easyaiNCtx=total;"
                  "const pct=total?used/total:0;"
                  "if(total){"
                    "ctxEl.textContent="
                      "used.toLocaleString()+' / '+total.toLocaleString()+"
                      "' ('+Math.round(pct*100)+'%)';"
                  "}else{"
                    "ctxEl.textContent=used.toLocaleString();"
                  "}"
                  // Color the ctx number based on pressure.
                  "ctxEl.style.color="
                    "pct>=CTX_HARD?'#f85149':"
                    "pct>=CTX_SOFT?'#d29922':"
                    "'#e6edf3';"
                  // Lock input if we've crossed the hard threshold.
                  "if(total){"
                    "if(pct>=CTX_HARD){"
                      "setInputLocked(true,'context full ('+Math.round(pct*100)+"
                        "'%) — clear the chat to continue');"
                    "}else{"
                      "setInputLocked(false);"
                    "}"
                  "}"
                  "if(t&&t.predicted_n&&t.predicted_ms){"
                    "const tps=(t.predicted_n/(t.predicted_ms/1000));"
                    "lastEl.textContent="
                      "t.predicted_n+' tok · '+(t.predicted_ms/1000).toFixed(1)+'s · '+tps.toFixed(1)+' t/s';"
                  "}else if(!t){"
                    "lastEl.textContent='—';"
                  "}"
                "};"
                "window.__easyaiPushTimings=(t)=>{"
                  "if(t)lastTimings=t;"
                  "renderOverview();"
                "};"

                // Reached the end of block5 — if you see this in the
                // console you know the bar/tone/chip IIFE parsed and
                // executed completely.  No log == SyntaxError further up.
                "console.log('[easyai-inject] block5 OK — bar/tone/chip live');"
              "})();</script>";

            // ----- CSS hiding for features we don't support ------------------
            // The Svelte build minifies class names but keeps human-readable
            // ARIA labels, data-testids, and a few stable kebab-case classnames
            // (e.g. "mcp-*", "model-card-*").  We hide aggressively by name
            // patterns; if upstream renames things we'll need to revisit.
            inj <<
              "<style>"
                // MCP — explicit user request.
                "[class*=\"mcp\" i],[class*=\"Mcp\"],"
                "[data-testid*=\"mcp\" i],"
                "a[href*=\"mcp\" i],button[aria-label*=\"mcp\" i],"
                // Model load/unload (we serve a single fixed model).
                "[data-testid*=\"model-load\" i],[data-testid*=\"model-unload\" i],"
                "[aria-label*=\"load model\" i],[aria-label*=\"unload model\" i],"
                "[aria-label*=\"manage models\" i],"
                // OAuth / login (we use --api-key Bearer auth).
                "[data-testid*=\"oauth\" i],[data-testid*=\"login\" i],"
                "[data-testid*=\"authorize\" i],[aria-label*=\"sign in\" i],"
                // Pyodide / Python interpreter.
                "[class*=\"pyodide\" i],[data-testid*=\"pyodide\" i],"
                "[class*=\"python\" i][role],[aria-label*=\"python\" i],"
                // Slot usage display (we have one slot).
                "[data-testid*=\"slot\" i]:not([data-testid*=\"slot-machine\"]),"
                "[aria-label*=\"slot usage\" i],"
                // Audio recording / microphone / camera.
                "[aria-label*=\"microphone\" i],[aria-label*=\"record audio\" i],"
                "[aria-label*=\"camera\" i],[data-testid*=\"audio-record\" i]"
                "{display:none !important;visibility:hidden !important;"
                " width:0 !important;height:0 !important;overflow:hidden !important;}"

                // Style thinking blocks: smaller, dimmer, monospace.  Catch
                // both the bundle's own thinking renderer (whichever class
                // it ends up minified to) and a literal &lt;think&gt; tag if
                // it ever survives sanitisation.
                " details[data-thinking],details.thinking,"
                " [data-testid*=\"thinking\" i],[class*=\"reasoning\" i],"
                " [class*=\"think-block\" i]{"
                "  font-size:.78rem !important;color:#8b949e !important;"
                "  font-family:ui-monospace,SFMono-Regular,Menlo,monospace !important;"
                "  border-left:2px solid #b97df3 !important;"
                "  padding-left:.6rem !important;margin:.4rem 0 !important;"
                "  opacity:.85;"
                " }"

                // Shrink the assistant message body to a comfortable read
                // size, closer to our chip / tone-badge font.  We target
                // the prose-style markdown wrapper inside the assistant
                // bubble while leaving the user message and code blocks
                // alone (code blocks have their own monospace size that
                // we don't want to mess with).
                " [aria-label=\"Assistant message with actions\"]{"
                "  font-size:.85rem !important;"
                "  line-height:1.55 !important;"
                " }"
                " [aria-label=\"Assistant message with actions\"] p,"
                " [aria-label=\"Assistant message with actions\"] li,"
                " [aria-label=\"Assistant message with actions\"] blockquote{"
                "  font-size:.85rem !important;line-height:1.55 !important;"
                " }"
                " [aria-label=\"Assistant message with actions\"] h1{"
                "  font-size:1.05rem !important;"
                " }"
                " [aria-label=\"Assistant message with actions\"] h2{"
                "  font-size:.98rem !important;"
                " }"
                " [aria-label=\"Assistant message with actions\"] h3,"
                " [aria-label=\"Assistant message with actions\"] h4{"
                "  font-size:.92rem !important;"
                " }"
                // Don't shrink code/pre — they have their own visual rules.
                " [aria-label=\"Assistant message with actions\"] code,"
                " [aria-label=\"Assistant message with actions\"] pre{"
                "  font-size:.78rem !important;"
                " }"
              "</style>";

            // Always point the browser at our /favicon route — that
            // route serves either --webui-icon (when the operator gave
            // one) or the embedded AI-brain.svg as the default.  We also
            // keep an observer running that re-asserts our link if the
            // Svelte app later mounts its own (Svelte often replaces
            // <head> children on hydration).
            inj << "<link id=\"__easyaiFavicon\" rel=\"icon\" href=\"/favicon\">";
            inj << "<script>(()=>{"
                << "console.log('[easyai-inject] block6 favicon-keeper');"
                << "const want='/favicon';"
                << "const ours=()=>document.getElementById('__easyaiFavicon');"
                << "const nuke=()=>{"
                  << "document.querySelectorAll('link[rel~=\"icon\"]').forEach(l=>{"
                    << "if(l.id!=='__easyaiFavicon')l.remove();"
                  << "});"
                  << "let our=ours();"
                  << "if(!our){"
                    << "our=document.createElement('link');"
                    << "our.id='__easyaiFavicon';"
                    << "our.rel='icon';"
                    << "our.href=want;"
                    << "document.head&&document.head.appendChild(our);"
                  << "}else if(our.getAttribute('href')!==want){"
                    << "our.setAttribute('href',want);"
                  << "}"
                << "};"
                << "nuke();"
                << "if(document.head){"
                  << "new MutationObserver(nuke).observe("
                    << "document.head,{childList:true,subtree:false,attributes:true,"
                    << "attributeFilter:['href','rel']});"
                << "}"
                << "document.addEventListener('DOMContentLoaded',()=>{"
                  << "nuke();"
                  << "if(document.head){"
                    << "new MutationObserver(nuke).observe("
                      << "document.head,{childList:true,subtree:false,attributes:true,"
                      << "attributeFilter:['href','rel']});"
                  << "}"
                << "});"
                << "})();</script>";

            // Splice immediately after <head>.
            const std::string head_open = "<head>";
            size_t pos = html.find(head_open);
            if (pos != std::string::npos) {
                html.insert(pos + head_open.size(), inj.str());
            }
            ctx->webui_html = std::move(html);

            // ----- bundle.js text-substitute (real, not just CSS) ------
            // The Svelte bundle has hard-coded brand strings inside string
            // literals — DOM scrubbers can't catch the welcome <h1> or the
            // input placeholder.  Patch them now, once, into a buffer
            // we'll serve from /bundle.js.
            std::string js(reinterpret_cast<const char *>(bundle_js),
                            bundle_js_len);

            // Page-title surface (already pinned by document.title interceptor,
            // but we also clean these up so view-source / window.title is correct).
            js = str_replace_all(js, "llama.cpp - AI Chat Interface", title);
            js = str_replace_all(js,
                "Initializing connection to llama.cpp server...",
                "Initializing connection to " + title + " server…");
            js = str_replace_all(js, "} - llama.cpp", "} - " + title);

            // Sidebar / topbar brand markup.  We prepend an <img> pointing
            // at /favicon (which serves either the operator's --webui-icon
            // or the bundled AI-brain.svg) so the brand reads as
            // "<icon> EasyAi" instead of just text.  The styling keeps the
            // icon vertically aligned with the title text and sized to the
            // h1's line-height; brand h1 in the bundle is small (about 1rem),
            // so 1.05em hits "slightly bigger than the text" for visual
            // weight.  No external image lookup, no CDN.
            const std::string brand_html =
                "><img src=\"/favicon\" alt=\"\" style=\"height:1.05em;"
                "width:1.05em;vertical-align:-.18em;margin-right:.35em;"
                "border-radius:4px;background:transparent;display:inline-block\">"
                + title + "</h1>";
            js = str_replace_all(js, ">llama.cpp</h1>", brand_html);

            // Input placeholder.
            if (!args.webui_placeholder.empty()) {
                js = str_replace_all(js, "Type a message...", args.webui_placeholder);
                js = str_replace_all(js, "Type a message",    args.webui_placeholder);
            }

            ctx->webui_bundle_js = std::move(js);
        } else
#endif
        {
            // Minimal inline webui — substitute the title placeholder we
            // baked into the kWebUI string.
            ctx->webui_html = str_replace_all(kWebUI, "__EASYAI_TITLE__",
                                              html_escape(title));
        }

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
               .verbose    (args.verbose);
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

    // ---- webui routes ---------------------------------------------------
    // We serve the embedded llama-server-derived bundle by default. The
    // operator can fall back to the small inline kWebUI by passing
    // --webui minimal at startup.  When EASYAI_BUILD_WEBUI was OFF at
    // configure time, the modern path simply isn't available and we
    // always serve the minimal one.
    const bool serve_modern = (args.webui_mode != "minimal");
#if defined(EASYAI_BUILD_WEBUI)
    if (serve_modern) {
        svr.Get("/", [&](const httplib::Request &, httplib::Response & res) {
            // Required by some embedded resources (matches llama-server defaults).
            res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
            res.set_header("Cross-Origin-Opener-Policy",   "same-origin");
            res.set_content(ctx_ref.webui_html, "text/html; charset=utf-8");
        });
        svr.Get("/bundle.js", [&](const httplib::Request &, httplib::Response & res) {
            // Serve the customised bundle (brand / placeholder / icon
            // substitutions baked in at startup) when available; fall back
            // to the raw embedded bytes otherwise.
            if (!ctx_ref.webui_bundle_js.empty()) {
                res.set_content(ctx_ref.webui_bundle_js,
                                "application/javascript; charset=utf-8");
            } else {
                res.set_content(reinterpret_cast<const char*>(bundle_js),
                                bundle_js_len,
                                "application/javascript; charset=utf-8");
            }
        });
        svr.Get("/bundle.css", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(reinterpret_cast<const char*>(bundle_css), bundle_css_len,
                            "text/css; charset=utf-8");
        });
        svr.Get("/loading.html", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len,
                            "text/html; charset=utf-8");
        });
        // Minimal /props stub — tells the webui what the server can do.
        // We advertise thinking + tool calls so the webui's <think>...</think>
        // and tool-call renderers light up.  No image / audio modalities, no
        // model swapping, no slots.
        svr.Get("/props", [&](const httplib::Request &, httplib::Response & res) {
            ordered_json p;
            p["model_alias"]   = ctx_ref.model_id;
            p["model_path"]    = ctx_ref.engine.model_path();
            p["total_slots"]   = 1;
            p["modalities"]    = { {"vision", false}, {"audio", false} };
            p["chat_template"] = "";
            p["chat_template_caps"] = {
                {"supports_tools",                true},
                {"supports_tool_calls",           true},
                {"supports_system_role",          true},
                {"supports_parallel_tool_calls",  false},
                {"supports_preserve_reasoning",   true},   // pass <think>...</think> through
            };
            p["bos_token"]     = "";
            p["eos_token"]     = "";
            p["build_info"]    = "easyai-server";
            p["webui"]         = "easyai";
            p["webui_settings"] = json::object();
            p["default_generation_settings"] = {
                {"params", {
                    {"temperature", ctx_ref.def_temperature},
                    {"top_p",       ctx_ref.def_top_p},
                    {"top_k",       ctx_ref.def_top_k},
                    {"min_p",       ctx_ref.def_min_p},
                }},
                {"n_ctx", ctx_ref.engine.n_ctx()},
            };
            p["endpoint_props"]   = false;
            p["endpoint_slots"]   = false;
            p["endpoint_metrics"] = ctx_ref.api_key.empty();   // hint
            p["is_sleeping"]      = false;
            res.set_content(p.dump(), "application/json");
        });
    } else
#endif
    {
        // Minimal inline webui (legacy / fallback / when EASYAI_BUILD_WEBUI=OFF)
        svr.Get("/", [&](const httplib::Request &, httplib::Response & res) {
            res.set_content(ctx_ref.webui_html, "text/html; charset=utf-8");
        });
    }

    // Favicon: serve the operator-supplied icon when --webui-icon was
    // given; otherwise serve the embedded brain SVG that ships with the
    // binary so the page is never branded as a generic globe.
    auto serve_favicon = [&](const httplib::Request &, httplib::Response & res){
        if (!ctx_ref.webui_icon.empty()) {
            res.set_content(ctx_ref.webui_icon, ctx_ref.webui_icon_mime.c_str());
            return;
        }
#if defined(EASYAI_BUILD_WEBUI)
        // Embedded default — the AI-brain.svg shipped under src/.
        res.set_content(reinterpret_cast<const char *>(AI_brain_svg),
                        AI_brain_svg_len, "image/svg+xml");
#else
        res.status = 204;
#endif
    };
    svr.Get("/favicon",     serve_favicon);
    svr.Get("/favicon.ico", serve_favicon);
    svr.Get("/favicon.svg", serve_favicon);
    svr.Get ("/health",               [&](const auto & q, auto & r){ route_health  (ctx_ref, q, r); });
    if (args.metrics) {
        svr.Get ("/metrics",          [&](const auto & q, auto & r){ route_metrics (ctx_ref, q, r); });
    }
    svr.Get ("/v1/models",            [&](const auto & q, auto & r){
        if (!require_auth(ctx_ref, q, r)) return;
        route_models(ctx_ref, q, r);
    });
    svr.Get ("/v1/tools",             [&](const auto & q, auto & r){
        if (!require_auth(ctx_ref, q, r)) return;
        route_tools(ctx_ref, q, r);
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
