#include "easyai/log.hpp"

namespace easyai::log {

namespace {
std::FILE * g_sink = nullptr;
}

void set_file(std::FILE * fp) { g_sink = fp; }

void write(const char * fmt, ...) {
    va_list ap1, ap2;
    va_start(ap1, fmt);
    va_copy(ap2, ap1);

    std::vfprintf(stderr, fmt, ap1);
    va_end(ap1);

    if (g_sink) {
        std::vfprintf(g_sink, fmt, ap2);
        std::fflush(g_sink);
    }
    va_end(ap2);
}

}  // namespace easyai::log
