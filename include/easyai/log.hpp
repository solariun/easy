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

namespace easyai::log {

// Set the secondary sink (in addition to stderr).  Borrowed pointer —
// caller fclose()'s.  Pass nullptr to disable.
void set_file(std::FILE * fp);

// printf-family: writes to stderr and (if set_file was called) to the
// secondary sink, flushing the latter so partial logs survive crashes.
void write(const char * fmt, ...) __attribute__((format(printf, 1, 2)));

}  // namespace easyai::log
