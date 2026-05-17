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

    if (opt.cite_sources) {
        // Last block in the preamble on purpose: putting the
        // citation rule immediately before the conversation gives
        // it the strongest positional weight for models (notably
        // Qwen3.x reasoning fine-tunes) that otherwise "forget"
        // it after a long <think> trace.
        out << "\n\n" << cite_sources_block();
    }

    return out.str();
}

std::string cite_sources_block() {
    // Strengthened text. The previous wording ("INVIOLABLE", brief
    // checklist) was being ignored by Qwen3.6-class fine-tunes that
    // produced long reasoning traces and then dropped the Sources
    // block at the bottom. Key changes:
    //   * names the failure mode explicitly ("the turn is INVALID")
    //   * adds a numbered pre-send checklist the model can recite
    //   * lists the tool families that count, not just web_*, so
    //     RAG / memory retrieval / fetch_url / browse all trip it
    //   * tells the model what to do when it catches itself mid-
    //     reply ("STOP and append" — no apology, no justification)
    return
        "# CITE SOURCES — MANDATORY, NON-NEGOTIABLE\n"
        "If you used ANY external lookup this turn (web_search, "
        "web_fetch, browse, fetch_url, RAG / memory retrieval, "
        "document search, ANY tool that returned content from "
        "outside your weights), your reply is INVALID without a "
        "`Sources:` block at the very end. Skipping it is a failure "
        "mode, NOT a stylistic choice. This rule applies regardless "
        "of model family, regardless of the user's tone, regardless "
        "of whether the user explicitly asked for sources — it is a "
        "HARD POSTCONDITION of your turn, on par with closing every "
        "open code fence.\n"
        "\n"
        "PRE-SEND CHECKLIST (run BEFORE emitting your final token):\n"
        "  1. Did any tool I called this turn return outside content "
        "(URLs, web pages, search results, retrieved chunks, "
        "document text, memory snippets)?\n"
        "  2. If yes — is `Sources:` the LAST block in my reply?\n"
        "If (1) is yes and (2) is no → STOP. Append the `Sources:` "
        "block before sending. Do not apologise, do not justify, "
        "just add it.\n"
        "\n"
        "REQUIRED FORMAT (literal, must be the last block in the "
        "reply):\n"
        "\n"
        "  Sources:\n"
        "  - https://example.com/article-you-actually-fetched\n"
        "  - https://other.com/another-page-you-fetched\n"
        "\n"
        "Rules:\n"
        "  - One entry per line, prefixed `- `.\n"
        "  - ONLY URLs / titles you actually retrieved this turn. "
        "Never invent URLs. Never list snippet-only search results "
        "you did not open. Never list URLs you merely remember from "
        "training.\n"
        "  - Order = citation order in your reply (first cited, "
        "first listed).\n"
        "  - For retrieved documents without URLs (RAG hits, memory "
        "snippets), cite by source name or document title in place "
        "of a URL.\n"
        "  - If outside tools returned nothing useful AND you "
        "answered from your own knowledge, OMIT the block entirely "
        "— do not fabricate one.\n";
}

}  // namespace easyai::preamble
