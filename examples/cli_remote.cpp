// cli_remote.cpp — easyai-cli-remote binary (skeleton, phase 1).
//
// Phase 3 will turn this into a full agentic CLI: command-line + REPL,
// built-in tools (datetime, web_search, web_fetch, fs_*, plan), live
// status output, --list-models / --list-tools / --health / --props /
// --metrics / --set-preset etc. for managing a remote easyai-server.
//
// For now it just demonstrates the public surface of libeasyai-cli so
// downstream packagers can verify the library links cleanly.

#include "easyai/client.hpp"
#include "easyai/plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s --url <endpoint> [--api-key <key>] [--model <id>] [-p <prompt>]\n"
        "\n"
        "Phase 1 stub — full agentic mode lands in phase 3.\n",
        argv0);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string url   = std::getenv("EASYAI_URL")     ? std::getenv("EASYAI_URL")     : "";
    std::string key   = std::getenv("EASYAI_API_KEY") ? std::getenv("EASYAI_API_KEY") : "";
    std::string model = "EasyAi";
    std::string prompt;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (a == "--url")     url    = need("--url");
        else if (a == "--api-key") key    = need("--api-key");
        else if (a == "--model")   model  = need("--model");
        else if (a == "-p" || a == "--prompt") prompt = need("--prompt");
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }

    if (url.empty()) {
        std::fprintf(stderr, "error: --url (or EASYAI_URL env) is required\n");
        usage(argv[0]);
        return 2;
    }

    easyai::Client cli;
    cli.endpoint(url).model(model).verbose(true);
    if (!key.empty()) cli.api_key(key);

    // Show that the Plan tool wires in cleanly.
    easyai::Plan plan;
    plan.on_change([](const easyai::Plan & p) {
        std::fputs("\n[plan]\n", stdout);
        p.render(std::cout);
    });
    cli.add_tool(plan.tool());

    if (prompt.empty()) {
        std::fprintf(stderr,
            "[easyai-cli-remote] phase-1 skeleton up.\n"
            "  endpoint = %s\n"
            "  model    = %s\n"
            "  tools    = %zu (plan registered)\n"
            "Pass -p '<prompt>' to send a message once phase 2 ships.\n",
            url.c_str(), model.c_str(), cli.tools().size());
        return 0;
    }

    std::string answer = cli.chat(prompt);
    if (answer.empty() && !cli.last_error().empty()) {
        std::fprintf(stderr, "error: %s\n", cli.last_error().c_str());
        return 1;
    }
    std::fputs(answer.c_str(), stdout);
    std::fputs("\n", stdout);
    return 0;
}
