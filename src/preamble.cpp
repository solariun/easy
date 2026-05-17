#include "easyai/preamble.hpp"
#include "easyai/rag_tools.hpp"   // render_memory_vocabulary

#include <chrono>
#include <ctime>
#include <sstream>

namespace easyai::preamble {

std::string build(const Options & opt) {
    std::ostringstream out;

    if (opt.inject_datetime) {
        auto now = std::chrono::system_clock::now();
        auto tt  = std::chrono::system_clock::to_time_t(now);
        std::tm lt{};
#if defined(_WIN32)
        localtime_s(&lt, &tt);
#else
        localtime_r(&tt, &lt);
#endif
        char ts[64], tz[32];
        std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S %z", &lt);
        std::strftime(tz, sizeof(tz), "%Z",                    &lt);

        out << "\n\n# AUTHORITATIVE DATE/TIME (do not ignore, do not "
               "second-guess)\n"
            << "Current date and time: " << ts << " (" << tz << ").\n"
            << "Trust this over any training-data intuition about "
               "\"today\".\n"
            << "If the user mentions \"today\", \"now\", \"this year\" "
               "etc., use the\n"
            << "value above.  When unsure, call the `datetime` tool "
               "first.\n";

        if (!opt.knowledge_cutoff.empty()) {
            out << "\n# KNOWLEDGE CUTOFF\n"
                << "Your training data ends around "
                << opt.knowledge_cutoff << ".\n"
                << "For TIME-SENSITIVE claims after that cutoff —\n"
                << "current events, prices, scores, weather, latest\n"
                << "releases, who-holds-what-office — verify with a\n"
                << "tool (web search/fetch, datetime) OR state\n"
                << "uncertainty. Never present a post-cutoff\n"
                << "time-sensitive fact as known.\n"
                << "\n"
                << "STABLE facts (definitions, syntax, architecture,\n"
                << "math, algorithms) don't need verification just\n"
                << "because the topic is recent — trust your\n"
                << "knowledge unless the user asks for the latest\n"
                << "state. One verification per topic is enough; don't\n"
                << "re-verify after every fetch.\n";
        }
    }

    if (!opt.memory_root.empty()) {
        std::string vocab = easyai::tools::render_memory_vocabulary(
            opt.memory_root);
        if (!vocab.empty()) {
            // The leading "\n\n" lets this block stand alone when
            // it's the first thing in the preamble (inject_datetime
            // was false). When the date/time block precedes it,
            // the extra blank line is absorbed by the renderer's
            // own trailing newline.
            out << "\n\n# MEMORY VOCABULARY (the keywords your "
                   "private memory currently has tagged — the FIRST "
                   "place to look for anything you might already "
                   "know)\n"
                << vocab << "\n";
        }
    }

    return out.str();
}

}  // namespace easyai::preamble
