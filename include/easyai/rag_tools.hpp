// easyai/rag_tools.hpp — RAG: the agent's persistent registry.
//
// RAG is a keyword-indexed, file-backed long-term memory for an
// agent. The model writes notes (title + keywords + free-form
// content), searches by keyword, and loads up to 4 entries at a time
// to read back their content.
//
// On-disk format is intentionally minimal so an operator can
// `cat` / `vim` / `grep` entries with no tooling:
//
//   <root>/<title>.md
//
//     keywords: user-prefs, hardware, radv
//
//     Body content here, free-form UTF-8 text.
//     Multiple paragraphs, markdown, code blocks — anything.
//
// First line is the header (`keywords: <comma-separated list>`),
// then a blank line, then the body. A file with no header is
// treated as "untagged" — it shows in `rag_list` but never in
// `rag_search`. `created` / `modified` come from the filesystem's
// own mtime; we don't store them redundantly.
//
// Why a keyword registry, not a vector store?
// ---------------------------------------
// A vector store needs an embedding model, a similarity index,
// and a notion of "neighbours". RAG needs zero of those: it's
// a keyword-keyed key/value store with tiny indexing cost. The model
// is the one deciding what to remember and how to classify it,
// which is the part vector stores get wrong anyway. When we
// later want progressive recall ("load my last 5 most-relevant
// entries on session start"), that layer can sit on top — RAG
// itself stays simple and cheap.
//
// Constraints
// -----------
//   * Title and keywords match `^[A-Za-z0-9._+-]+$`. No spaces, no
//     slashes — the title is the filename component, so this
//     also means there's no path-traversal surface.
//   * Title  ≤ 64 bytes; keyword ≤ 32 bytes; up to 8 keywords per entry.
//   * Content ≤ 256 KiB.
//
// Concurrency
// -----------
//   * Saves are atomic: we write a tempfile then rename(2).
//   * The in-memory index is guarded by a mutex.
//   * Multiple processes sharing the same root works for
//     reads, but the in-memory index won't see concurrent
//     writes from other processes — single-process is the
//     supported model.
#pragma once

#include "tool.hpp"

#include <string>

namespace easyai::tools {

// Build the RAG tool rooted at `root_dir`. Single-tool dispatcher:
// exposes one `rag` tool with an `action` parameter selecting one of
// "save" / "append" / "search" / "load" / "list" / "delete" /
// "keywords"; remaining params (title, keywords, content, fix,
// titles, prefix, max, max_results, page, min_count) are routed to
// the matching action's handler internally.
//
// The directory is created on demand at first save; missing-directory
// at registration time is NOT an error (operator may not have
// provisioned it yet).
//
// `root_dir` must not be empty; an empty path is a programmer error
// and the tool will reject every call with a clear message.
//
// Fixed memories: action="save" accepts a `fix=true` argument that
// promotes the saved entry to immutable. Immutable memories have a
// `fix-easyai-` title prefix; save refuses to overwrite them and
// delete refuses to remove them. Use this to seed system designs /
// domain knowledge / hard rules the model must not rewrite mid-
// conversation. action="search" / action="load" always see fixed
// entries.
Tool make_rag_tool(std::string root_dir);

}  // namespace easyai::tools
