// easyai/log.hpp — tee-to-stderr-and-FILE printf helper.
//
// Pattern: examples want to log diagnostic noise to stderr AND
// optionally also to a long-lived `--log-file path.log` so the user
// can replay sessions or share them with us when something goes
// wrong.  Two writes per call gets old fast; this is the helper
// that does it once, with a single global FILE* sink the binary
// owns and can swap.
//
// The sink pointer is borrowed.  Caller is responsible for fclose().
// Pass nullptr to clear.
#pragma once

#include <cstdarg>
#include <cstdio>
#include <string>

namespace easyai::log {

// Set the secondary sink (in addition to stderr).  Borrowed pointer —
// caller fclose()'s.  Pass nullptr to disable.
void set_file(std::FILE * fp);

// Read back the current secondary sink (nullptr if none).  Used by
// libs that want to write extra structured detail (raw bytes, multi-
// line dumps) directly into the same file, rather than going through
// the stderr-tee printf wrapper.
std::FILE * file();

// printf-family: writes to stderr and (if set_file was called) to the
// secondary sink, flushing the latter so partial logs survive crashes.
void write(const char * fmt, ...) __attribute__((format(printf, 1, 2)));

// Same as `write` but ALSO mirrors the message in a clearly-marked
// "ERROR" block in the raw log file (when one is attached) so a quick
// grep pulls every problem line out of a long session.  Stderr still
// receives the plain message.  Use this from the libs whenever
// something fails, retries, or otherwise deserves operator attention.
void error(const char * fmt, ...) __attribute__((format(printf, 1, 2)));

// Mark a "problematic transaction" in the raw log file.  Writes a
// banner block — `!!!!! PROBLEMATIC TRANSACTION !!!!!` — followed by
// the printf-formatted body, so an operator skimming a /tmp log can
// jump straight to the problem turn without re-reading every kilobyte
// of SSE bytes.  No-op when no log file is attached.  Stderr is NOT
// touched (use easyai::log::write for that).
void mark_problem(const char * fmt, ...) __attribute__((format(printf, 1, 2)));

// Auto-open a /tmp/<prefix>-<pid>-<epoch>.log raw transaction log,
// register it as the secondary sink, and return the FILE*.  No-op
// (returns the already-attached FILE*) if a sink is already set.
// No-op + returns nullptr when EASYAI_NO_AUTO_LOG is set in the env.
//
// Called automatically from Engine::load() and Client::Client() so
// every server / CLI built on the libs inherits a raw transaction
// log without having to wire one up by hand.  Ownership: the lib
// owns the FILE* once auto_open returns it; auto_close() (also
// registered as an atexit handler on first call) flushes and
// closes it at shutdown.
//
// `resolved_path` (optional) receives the actual path opened so
// callers can print "raw log: <path>" to the operator at startup.
std::FILE * auto_open(const char * prefix,
                      std::string * resolved_path = nullptr);

// Close the auto-opened sink (if one was opened by auto_open).
// Safe to call from atexit handlers — no-op when no sink is set.
void auto_close();

}  // namespace easyai::log
