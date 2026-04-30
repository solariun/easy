# RAG — the agent's persistent registry

> *"A model that forgets between sessions is an expensive autocomplete.
> A model that remembers is a colleague. RAG is the cheapest path
> from the first to the second."*

This document is the authoritative guide to easyai's RAG: a tag-
indexed, file-backed long-term memory the model can use to remember
things across sessions.

It's a deliberately minimal take on Retrieval-Augmented Generation:
no embedding model, no vector store, no similarity index, no
database — just a directory of small Markdown files the agent
reads, writes, and curates by itself. The agent classifies its own
memory with keywords; we keep the directory and the index. That's
the whole system.

---

## Table of contents

1. [What RAG is, and why](#1-what-rag-is-and-why)
2. [Quickstart](#2-quickstart)
3. [The file format on disk](#3-the-file-format-on-disk)
4. [The five tools](#4-the-five-tools)
5. [How the model is encouraged to use it](#5-how-the-model-is-encouraged-to-use-it)
6. [Workflows](#6-workflows)
7. [Best practices](#7-best-practices)
8. [Corner cases](#8-corner-cases)
9. [Operator workflows](#9-operator-workflows)
10. [Roadmap — progressive recall, document ingestion, multi-user](#10-roadmap)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What RAG is, and why

RAG is a **keyword-indexed key/value store**, owned by the agent,
persisted on disk, accessible to the agent via six tools:

```
rag_save(title, keywords[], content)        write / overwrite
rag_search(keywords[], page?, max_results?) find by 1+ keywords (≥2 matches when 2+ given), paginated
rag_load(titles[1..4])                      read up to 4 full bodies
rag_list(prefix?, max?)                     browse titles
rag_delete(title)                           remove a stale entry
rag_keywords(min_count?, max?)              vocabulary overview
```

That is the whole API.

The model decides what to remember and how to classify it. Keywords
are how it finds entries again later. The directory is the index.
There is no embedding model, no similarity scoring, no neighbours.
The model already has everything it needs to reason about its own
memory; we just give it a place to put the bits it cared about.

### How information flows

```
  ┌──────────────────────────────────────────────────────────────┐
  │                         THE MODEL                             │
  │              (sees the 6 tools in its toolbelt)               │
  └──────────────────────────────────────────────────────────────┘
           │           │           │           │           │           │
           ▼           ▼           ▼           ▼           ▼           ▼
        ┌──────┐   ┌────────┐  ┌────────┐  ┌──────┐  ┌────────┐  ┌──────────┐
        │ save │   │ search │  │  load  │  │ list │  │ delete │  │ keywords │
        │      │   │        │  │        │  │      │  │        │  │          │
        │write │   │ find   │  │ read   │  │browse│  │ prune  │  │vocab     │
        │ /    │   │ by     │  │ up to  │  │titles│  │ stale  │  │overview  │
        │amend │   │keyword │  │  4     │  │      │  │        │  │ (counts) │
        └──┬───┘   └───┬────┘  └───┬────┘  └──┬───┘  └───┬────┘  └────┬─────┘
           │           │           │           │           │           │
           ▼           ▼           ▼           ▼           ▼           ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                        RagStore (in-process)                  │
  │  ┌────────────────────────────────────────────────────────┐  │
  │  │  in-memory index: title → { keywords, mtime, bytes }   │  │
  │  │  lazy-loaded from disk on first call, kept fresh by    │  │
  │  │  every save / delete; mutex-guarded                     │  │
  │  └────────────────────────────────────────────────────────┘  │
  │                                                               │
  │   search / list / keywords  ─→  index lookup, no disk read    │
  │   load                      ─→  one file read off disk        │
  │   save                      ─→  atomic tempfile + rename(2)   │
  │   delete                    ─→  unlink + index erase          │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │            /var/lib/easyai/rag/    (filesystem)               │
  │                                                               │
  │   gustavo-prefs.md      keywords: user-prefs, locale          │
  │   easyai-build.md       keywords: easyai, build, recipe       │
  │   mqtt-qos.md           keywords: mqtt, qos, protocol         │
  │   README.md             (no keywords header — untagged)       │
  └──────────────────────────────────────────────────────────────┘
```

The lifecycle of a piece of knowledge:

```
  SESSION 1 (Mon)
  ───────────────
  user:  "I prefer terse PT-BR responses."
  model: rag_keywords()                      ← sees "user-prefs" already
                                                exists in vocabulary
         rag_save("user-prefs",              ← reuses existing keyword
                  ["user-prefs","locale"],
                  "Prefers PT-BR, terse...")
                                              ← atomic write to disk
  ────────────────────────────────────  end of session ─────────

  SESSION 2 (Wed, fresh process)
  ──────────────────────────────
  user:  "build the project"
  model: rag_search(["easyai","build"])      ← matches 2/2 → easyai-build.md
         rag_load(["easyai-build"])          ← reads body
         rag_search(["user-prefs"])          ← finds gustavo-prefs.md
         rag_load(["user-prefs"])            ← reads body
                                              ← model now answers
                                                in PT-BR, terse, with
                                                the right build command

  ────────────────────────────────────  later that week ─────────

  user:  "we dropped the X feature"
  model: rag_search(["x-feature"])           ← finds 3 stale entries
         rag_delete("x-feature-rationale")
         rag_delete("x-feature-roadmap")     ← curation keeps the
         rag_delete("x-feature-userflow")      vocabulary clean for
                                                future searches
```

Three things make this loop work:

1. **The model classifies its own memory.** It saw the conversation;
   it picks a stable, descriptive title and 2–5 reusable keywords.
2. **Keywords are the index.** No embeddings, no GPU, no opaque
   ranking — exact-match lookup over a small in-memory map.
3. **The model curates.** `rag_keywords` lets it see what vocabulary
   it has built; `rag_delete` lets it prune. Without curation, the
   index drifts and old entries become unreachable.

### Why no vector store / embeddings?

The popular flavour of RAG plugs an embedding model + a vector store
in front of the LLM and ranks chunks by cosine similarity. That's
the right answer when you have a huge corpus that nobody
classified.

easyai's RAG flips the assumption: **the agent IS the classifier**.
When the model decides to remember something, it tells you (in
clear language, in the same call) what the entry is about. Put
that classification in the filename + a small header and you can
find the entry in O(1) per lookup, with no GPU, no embedding
inference, no opaque ranking, no schema migrations.

When easyai later grows progressive recall (auto-inject the N
most-relevant entries on every session start), THAT layer can add
similarity scoring on top of RAG without changing what's on disk.
RAG itself stays simple: files and keywords.

### Why files, not a database?

So you, the operator, can `cat`, `vim`, `grep`, drop a hand-written
note into the dir, archive an entry by moving the file, share a
snippet by copying it. There is no ceremony. The agent's memory is
human-inspectable at all times.

---

## 2. Quickstart

### On the installed server (RAG is on by default)

The systemd-installed easyai-server already passes `--RAG
/var/lib/easyai/rag` for you. Verify:

```bash
sudo journalctl -u easyai-server | grep "RAG enabled"
# easyai-server: RAG enabled, root = /var/lib/easyai/rag

ls -la /var/lib/easyai/rag/
# -rw-r----- root easyai 0 README.md   (empty initially)
```

Open the webui or hit the API. The model now has rag_save / rag_search
/ rag_load / rag_list / rag_delete in its tool list. Tell it
something memorable and it will save it without further prompting.

### From easyai-cli (remote model, local memory)

```bash
mkdir -p ~/easyai-reg
easyai-cli --url http://127.0.0.1:8080 --RAG ~/easyai-reg
```

Same five tools, but the memory lives in your home directory.

### From easyai-local (single-process REPL with RAG)

```bash
easyai-local -m model.gguf --RAG ~/easyai-reg
```

### Verifying it works

After a chat that should have triggered a save:

```bash
ls /var/lib/easyai/rag/
cat /var/lib/easyai/rag/<title>.md
```

Or from the model:

```
> rag_list everything I know
```

---

## 3. The file format on disk

Each entry is one file: `<title>.md`. Format is intentionally trivial:

```
keywords: user-prefs, hardware, radv

Body content here.
Free-form UTF-8 text. Can be Markdown, code blocks, structured snippets,
plain prose — whatever the model wanted to remember. Up to 256 KB.
```

The grammar:

1. The first line, if it looks like `<key>: <value>`, is the header.
2. We currently recognise one key: `keywords:`. Comma-separated values.
3. A blank line ends the header.
4. Everything after is the body.

A file with NO header (no `keywords:` line) is treated as
**untagged**. It shows up in `rag_list` but never in `rag_search`.
This is by design — operators can drop hand-written notes into the
dir and the model will list them as available context, but won't
consider them "tagged knowledge" until the operator (or the model)
adds keywords.

### Hand-authoring an entry

```bash
sudo -u easyai bash -c 'cat > /var/lib/easyai/rag/welcome.md' <<'EOF'
keywords: user-prefs, language, locale

The user prefers responses in Brazilian Portuguese (PT-BR). Technical
jargon in English is fine. Keep responses terse — favour code or
commands over long explanations. Default code style: C++17, snake_case
identifiers, no exceptions in hot paths.
EOF
```

Restart the server (or wait for the next session) and the model has
it on its first rag_search by `user-prefs`.

### Constraints

- Title (= filename stem): `^[A-Za-z0-9_-]{1,64}$`. No spaces, no
  slashes, no dots. The strict regex closes path-traversal — there is
  no way for the model to write outside the RAG dir.
- Keyword: same character set, ≤ 32 bytes, ≤ 8 keywords per entry.
- Content body: ≤ 256 KiB.

---

## 4. The six tools

Each tool description below is what the MODEL sees. The descriptions
were written to actively encourage use — see §5.

### rag_save

```
rag_save(title: string, keywords: string[], content: string) -> ok
```

Writes `<root>/<title>.md`. Overwrites if it already exists. Atomic
on POSIX (tempfile + rename). Refuses invalid title or keywords with
a clear error.

### rag_search

```
rag_search(keywords: string[], max_results: integer = 10) -> list of {title, keywords, preview, matched/total}
```

Pass an array of 1..8 keywords. The threshold is **adaptive**:

- **1 keyword** → returns entries that have that keyword (broad sweep).
- **2+ keywords** → returns entries that match **at least 2** of them
  (narrow query, ranked by overlap).

Each result reports `matched N/M` so the model can rank: an entry
that matched 3 of the 4 queried keywords is more relevant than one
that matched only 2. Best-overlap first, ties broken by recency.

Returns up to 20 entries (preview ≈ 240 bytes per entry). The model
picks the most relevant 1–4 titles and calls rag_load to read their
bodies.

**Optimisation pattern:** start a query with 3-4 related keywords. If
some entries score `M/M` (full match), you've found exact hits; if
the best is `2/4`, your space of related notes is sparser than you
thought — widen your query (drop 1-2 keywords) or use rag_list.

**Pagination.** Every response begins with three machine-readable
header lines:

```
total_entries: 47
page: 1 of 5
showing: 10  (entries 1..10)
has_more: true
```

When `has_more: true`, issue the SAME query with `page=P+1` to walk
the rest. The first page is already best-first, so you usually
don't need more — the model decides. Asking for `page=99` past the
end gets a clear "past the last page" message, not an error.

### rag_load

```
rag_load(titles: string[1..4]) -> entries with full body
```

Reads up to 4 full entries off disk. Cap is deliberate: more than 4
means the model is drowning the prompt. Missing titles surface as
per-entry errors in the response, the others still load.

### rag_list

```
rag_list(prefix: string?, max: integer = 50) -> list of titles
```

Browse mode. Returns title, keywords, content_bytes, modified_unix
for every entry (or every entry whose title starts with `prefix`).
Body NOT included — use rag_load for that.

### rag_delete

```
rag_delete(title: string) -> ok
```

Permanent. Removes the file from disk and the in-memory index.
Idempotent: deleting a non-existent title is not an error. No trash,
no recovery — be certain.

### rag_keywords

```
rag_keywords(min_count: integer = 1, max: integer = 200)
  -> { total_keywords, total_entries, showing, [keyword, count]* }
```

Vocabulary overview. Lists every distinct keyword used across the
RAG with the number of entries that reference it. Sorted by
frequency (most-used first), tie-broken alphabetically.

**Why it matters.** Without rag_keywords, an agent that doesn't
check its own vocabulary creates near-duplicates over time —
`user-prefs` vs `user_pref` vs `preferences`, `cmd-recipe` vs
`command-recipe`, etc. — and the index slowly fragments. Old
entries become unreachable to new searches because the queries
target slightly-different keywords. Calling rag_keywords before
rag_save (or before rag_search when you don't know what's in the
RAG) keeps the vocabulary stable and the index coherent.

**Filters.** `min_count=2` hides one-off keywords (those used by a
single entry), surfacing only the established vocabulary. `max`
caps the result count; the long tail is dropped first.

**Example output:**

```
total_keywords: 23
total_entries: 47
showing: 23

user-prefs       12 entries
project-easyai    9 entries
cmd-recipe        7 entries
fix-vulkan        5 entries
mqtt              4 entries
qos               3 entries
…
asyncio           1 entry
django            1 entry
```

The first lines tell the model which dimensions of knowledge it
has invested in; the long tail is candidates for either
consolidation (rename to a more general keyword + rag_save) or
deletion.

---

## 5. How the model is encouraged to use it

Tool descriptions are not boilerplate. They are the most direct
incentive structure we have: the model reads them on every turn and
they shape its behaviour.

RAG's descriptions push three behaviours:

1. **Save aggressively.** `rag_save`'s description literally says "USE
   THIS AGGRESSIVELY for: the user's stated preferences and
   constraints, project structure and decisions you've learned,
   technical facts you had to look up, recipes / commands that
   worked, error patterns and their fixes, domain knowledge from
   documents the user fed you. The more carefully you populate the
   registry, the smarter you become over time."

2. **Search before assuming.** `rag_search`'s description says "USE
   THIS BEFORE assuming you don't know something the user might have
   told you in a past session — your past self may have already
   saved the answer."

3. **Tidy up.** `rag_delete`'s description encourages removing stale
   entries: "keeping the registry tidy makes future searches sharper."

The more often the model exercises these, the more useful RAG
becomes. Future versions can add automatic injection of
high-relevance entries into the system prompt, but the manual loop
already works.

---

## 6. Workflows

### A. The natural session loop

```
[user opens chat]
  ↓
model: rag_search(["user-prefs"]) → finds "gustavo-prefs"
model: rag_load(["gustavo-prefs"])  → reads the body
model: now knows the user prefers PT-BR, terse style, ...

[user asks a question]
  ↓
model answers in PT-BR, terse.

[user shares a new fact]
  ↓
model: rag_save("project-foo-bar", ["project", "foo"], "...")

[user corrects something]
  ↓
model: rag_search(["foo"]) → finds the old note
model: rag_save(SAME title, ...)   ← overwrites with corrected version
                                     OR
model: rag_delete("foo-old")
model: rag_save("foo-new", ...)
```

### B. Document ingestion (the positive cycle)

You feed the model a long document — a manual, a runbook, a paper.
The model summarises it, then **chunks the summary into RAG entries**:

```
You:    "Read this MQTT spec PDF. The link is /sandbox/mqtt-v5.pdf.
         Save the important parts to RAG so future me doesn't have
         to re-read it."

Model:  [reads the PDF via fs_read_file or web_fetch]
        [identifies 6 key topics: connect, publish, subscribe,
         retained messages, will, qos]
        rag_save("mqtt-connect",   ["mqtt", "protocol"], "...")
        rag_save("mqtt-publish",   ["mqtt", "protocol"], "...")
        rag_save("mqtt-qos",       ["mqtt", "protocol"], "...")
        ...
        "Saved 6 entries under keywords 'mqtt' + 'protocol'. Future you
         can rag_search(['mqtt']) to find any of them, or narrow with
         rag_search(['mqtt', 'qos']) to focus on QoS-related ones."
```

Now the next session, when you ask about MQTT, the model
rag_searches with `["mqtt"]`, finds the 6 titles, loads up to 4, and
answers from the saved knowledge — no re-reading. If the question
is more specific ("MQTT QoS levels"), the model uses
`rag_search(["mqtt", "qos"])` and gets only the entries that score
on both keywords, ranked by overlap.

This is the **positive cycle**: feed knowledge, saved knowledge,
searched knowledge, recalled knowledge. Each ingestion makes the
agent smarter for every future session.

### C. The "remember that for next time" pattern

```
You:    "Whenever I ask about easyai builds on the AI box,
         the answer is: cmake -B build -DCMAKE_BUILD_TYPE=Release
         && cmake --build build -j. Remember this."

Model:  rag_save("easyai-build-aibox",
                 ["easyai", "build", "ai-box"],
                 "On the AI box (Linux/Vulkan), build easyai with:
                  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j")
        "Saved. I'll find this with keyword 'easyai' or 'build' next time."
```

Next session:

```
You:    "build easyai on ai box"
Model:  rag_search(["easyai", "build"])  →  finds easyai-build-aibox (matched 2/2)
        rag_load(["easyai-build-aibox"])
        "Use: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
```

### D. Curate (delete)

```
You:    "We dropped the X feature. Remove anything about it from your
         memory."

Model:  rag_search(["x-feature"])
        [3 entries match]
        rag_delete("x-feature-rationale")
        rag_delete("x-feature-roadmap")
        rag_delete("x-feature-userflow")
        "Deleted 3 entries about the dropped X feature."
```

---

## 7. Best practices

### Choosing titles

- **Domain-prefix the title.** `easyai-build-recipe`, not `build-recipe`.
  Prefixes group entries when you `rag_list prefix:"easyai-"`.
- **Keep titles unique by purpose.** If you have ten build recipes for
  ten projects, the title carries the disambiguator
  (`easyai-build-recipe`, `nginx-build-recipe`).
- **Don't dump-everything.** A title called `random-stuff` defeats the
  purpose. If you can't pick a good title, the entry probably
  shouldn't exist as a single unit — break it up.

### Choosing keywords

- **Prefer stable, reusable keywords.** `user-prefs`, `project-easyai`,
  `cmd-recipe`, `fix-vulkan-radv`, `mqtt`, `protocol`. Avoid
  one-off keywords (`note-from-monday`).
- **3–5 keywords is the sweet spot.** Fewer means harder to find;
  more dilutes the index.
- **Cross-tag aggressively.** A note about the AI box's RADV bug
  fixes belongs under `ai-box`, `vulkan`, `radv`, `bug-fix`. The
  model will search by any of those.

### Body content

- **Granularity matters more than anything else.** Save many small
  focused entries; not a few sprawling ones. `easyai-build-mac`
  and `easyai-build-linux` should be **two** entries, not one
  combined `easyai-build`. Why: when rag_load returns an entry,
  the FULL body lands in the model's prompt — a 200-line note
  costs 1000+ tokens whether the model needed all of it or not.
  The search-then-load flow is built around this: rag_search
  ranks N candidates by overlap, the model picks up to 4 of the
  best, and each loaded body is small enough that all four fit
  comfortably. **It is always better to ask for two more loads
  than to swallow one giant one.** Rule of thumb: bodies over
  ~500 words are usually two or more entries pretending to be one.
- **Be specific.** "Use q8_0 KV cache" is vague; "Use `-ctk q8_0 -ctv
  q8_0` to halve the KV cache footprint at no measurable quality
  loss on the 35B model" is useful.
- **Include the answer, not the reasoning.** The model already
  reasoned its way to the conclusion; what future-you wants is the
  conclusion.
- **Markdown if structure helps.** Headers, lists, code fences. The
  body is plain text, so anything readable to a human is fine.

### When NOT to save

- Anything the user said only in passing ("I'm tired today" — not
  worth keeping).
- Anything that's already in the codebase (the model can `git grep`).
- Per-conversation scratch state — that's what conversation history
  is for.
- PII the user didn't authorise persisting.

### When to delete

- The user corrects a fact and the old entry is now wrong.
- A project ends, dies, or pivots.
- You realise you saved something at the wrong granularity (delete +
  rag_save with new title / keywords).

---

## 8. Corner cases

| Situation | What happens |
| --- | --- |
| `--RAG` not given to the CLI | The five reg_* tools aren't registered. Model has no long-term memory. |
| `--RAG` points to a non-existent dir | Created on first rag_save. No error at startup. |
| `--RAG` points to a file (not a dir) | First rag_save returns "RAG root is not a directory". Other tools also error. |
| Two processes share the same RAG dir | Reads work; the in-memory index of one process won't see writes from the other until that process restarts. Single-process is the supported model. |
| Title matches an existing entry | rag_save overwrites (atomic). Useful for refining notes. |
| Keyword used by no entry | rag_search returns "no entries match" (not an error). |
| rag_load asks for 5 titles | Rejected: "max 4 per call". |
| rag_load asks for a missing title | The OTHER titles still load; the missing one shows up as `--- title ---\nERROR: no RAG entry titled "title"\n` in the response. |
| Hand-authored file with no `keywords:` header | Loaded as untagged. Shows in rag_list, never in rag_search. The body is fully accessible via rag_load. |
| Hand-authored file with garbage in the body | Loaded fine. The body is opaque to RAG. |
| File > 256 KB | Skipped at index time; rag_load returns "entry exceeds 262144 bytes". Operator should split. |
| Filename with spaces / dots / slashes | Skipped at index time (doesn't match the title regex). |
| Filename without `.md` extension | Skipped. The agent only sees `.md` files in the dir. |
| Subdirectory inside the RAG dir | Ignored — only top-level scanned. |
| Empty RAG dir | Normal state. rag_list returns "RAG is empty". |
| `rag_delete` on a non-existent title | Returns ok with "nothing to delete". Idempotent. |

---

## 9. Operator workflows

### Backing up RAG

```bash
tar -czf reg-backup-$(date +%Y%m%d).tar.gz /var/lib/easyai/rag/
```

The dir is tiny (KB-scale typically). Toss it in your normal backup.

### Sharing an entry between machines

```bash
scp /var/lib/easyai/rag/important.md other-host:/var/lib/easyai/rag/
sudo systemctl restart easyai-server  # on the other host (picks up new file)
```

### Auditing what the agent has saved

```bash
ls -lt /var/lib/easyai/rag/ | head -20      # newest first
grep -l "user-prefs" /var/lib/easyai/rag/*.md
cat /var/lib/easyai/rag/<entry>.md
```

### Pruning old entries

```bash
# entries not modified in 6 months
find /var/lib/easyai/rag/ -name "*.md" -mtime +180 -ls
# delete after review
find /var/lib/easyai/rag/ -name "*.md" -mtime +180 -delete
```

(Or use the model: tell it to rag_list and rag_delete things it
considers stale.)

### Bulk-importing notes

Drop hand-authored `.md` files into the dir. Each must follow the
format:

```
keywords: tag1, tag2

body
```

Restart the server. The model picks them up on next rag_search /
rag_list.

---

## 10. Roadmap

RAG today is the simplest thing that could work. Future evolutions
that fit cleanly on top:

### Progressive recall on session start

The system prompt currently doesn't include any RAG content. Future:
on session start, automatically load the K most-relevant entries
(by some heuristic — recency, keyword overlap with the current
prompt, semantic similarity if we add embeddings). This makes the
agent immediately aware of its own memory without needing a
conscious rag_search.

### Document ingestion helper

Today the model has to chunk a document into RAG entries by hand.
Future: a `reg_ingest_document(path, base_keywords)` tool that takes
a long text, segments it (semantic boundaries, fixed-size chunks,
or model-driven topics), and saves each chunk as an entry —
returning a manifest of what was saved.

### Cross-entry references

Today entries are independent. Future: a soft reference syntax
(e.g. `see-also: title1, title2` in the header) so the model can
build small knowledge graphs and rag_load resolves them
transitively.

### Entry expiry

Today entries live forever. Future: optional `expires:` header so
seasonal / temporary knowledge auto-cleans without operator action.

### Multi-user / multi-namespace

Today one process owns one RAG dir. Future: a per-user namespace
(scope by client id, by API key, by something) so a multi-tenant
server can give every user their own memory without leaking.

### Encryption at rest

Today the dir is mode 750 — OS-level access control. Future:
optional symmetric encryption with a key from `EASYAI_REG_KEY` env
var, for sensitive deployments.

---

## 11. Troubleshooting

### "RAG enabled" never appears in the journal

The systemd unit isn't passing `--RAG`. Check:

```bash
systemctl cat easyai-server | grep -- --RAG
```

If missing, re-run `install_easyai_server.sh --upgrade --enable-now`
to refresh the unit.

### Model doesn't seem to use RAG

Two possible causes:

1. **The tools aren't registered.** Confirm:
   ```bash
   curl http://127.0.0.1:8080/health | jq .tool_count
   # should include the 5 reg_* tools
   ```
   If not, re-check the `--RAG` flag is reaching the binary.

2. **The model isn't being prompted to use it.** The descriptions
   already encourage it, but a strong system prompt overrides
   defaults. If your `system.txt` says "do not call tools", the
   model honours that. Edit `/etc/easyai/system.txt` to clarify
   that RAG is encouraged.

### "RAG root is not a directory"

A file exists at the path. Move it out of the way and recreate the
dir:

```bash
sudo mv /var/lib/easyai/rag /var/lib/easyai/rag.broken
sudo install -d -o easyai -g easyai -m 750 /var/lib/easyai/rag
sudo systemctl restart easyai-server
```

### "create RAG dir failed: Permission denied"

The agent runs as `easyai`. Verify:

```bash
ls -ld /var/lib/easyai/
# drwxr-x--- root easyai
ls -ld /var/lib/easyai/rag
# drwxr-x--- easyai easyai   (note: easyai, not root)
```

If owner is wrong, fix:

```bash
sudo chown -R easyai:easyai /var/lib/easyai/rag
sudo chmod 750 /var/lib/easyai/rag
```

### Entries vanish between sessions

Likely the agent ran with the wrong `--RAG` path (e.g. CLI vs server
disagree). Confirm both invocations point at the same dir.

### Entry was saved but `rag_search` doesn't find it

Check the on-disk file:

```bash
cat /var/lib/easyai/rag/<title>.md
```

The first line must be `keywords: <comma-separated>`. If the model
forgot keywords, the entry is untagged — visible in rag_list, not
rag_search.

### `cat`-ing an entry shows weird characters

The body is UTF-8 with no escaping. If you see `\n`, that's the
model's mistake — it wrote a literal backslash-n instead of a real
newline. Edit the file by hand to fix it.

---

*See also:* `LINUX_SERVER.md` (operator's guide for the systemd
install), `manual.md` (general easyai reference), `EXTERNAL_TOOLS.md`
(the operator-defined tools subsystem — different surface, similar
philosophy).
