// examples/chat.cpp — minimal REPL with no tools.
//
//   ./easyai-chat -m models/qwen2.5-0.5b-instruct.gguf
//
#include "easyai/easyai.hpp"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m model.gguf [-c ctx] [-ngl n] [-t threads] [--system \"prompt\"]\n",
        argv0);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string system_prompt = "You are a helpful, concise assistant.";
    int  n_ctx = 4096, ngl = -1, n_threads = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(1); }
            return argv[++i];
        };
        if      (a == "-m")        model_path     = next("-m");
        else if (a == "-c")        n_ctx          = std::stoi(next("-c"));
        else if (a == "-ngl")      ngl            = std::stoi(next("-ngl"));
        else if (a == "-t")        n_threads      = std::stoi(next("-t"));
        else if (a == "--system")  system_prompt  = next("--system");
        else { print_usage(argv[0]); return 1; }
    }
    if (model_path.empty()) { print_usage(argv[0]); return 1; }

    easyai::Engine engine;
    engine.model(model_path)
          .context(n_ctx)
          .gpu_layers(ngl)
          .system(system_prompt)
          .on_token([](const std::string & p){ std::cout << p << std::flush; });
    if (n_threads > 0) engine.threads(n_threads);

    if (!engine.load()) {
        std::fprintf(stderr, "load failed: %s\n", engine.last_error().c_str());
        return 1;
    }
    std::fprintf(stderr, "[easyai] loaded; backend = %s\n", engine.backend_summary().c_str());

    std::string line;
    while (true) {
        std::cout << "\n\033[32m> \033[0m";
        if (!std::getline(std::cin, line) || line.empty()) break;
        std::cout << "\033[33m";
        engine.chat(line);
        std::cout << "\033[0m\n";
    }
    return 0;
}
