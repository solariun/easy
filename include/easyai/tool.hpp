// easyai/tool.hpp — dead-simple tool definition for agents.
#pragma once

#include <functional>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace easyai {

// ToolResult is what a handler returns. content goes back to the model;
// is_error=true tags the message so the model knows the call failed.
struct ToolResult {
    std::string content;
    bool        is_error = false;

    static ToolResult ok(std::string s)    { return { std::move(s), false }; }
    static ToolResult error(std::string s) { return { std::move(s), true  }; }
};

// ToolCall is what easyai hands the handler. `arguments_json` is the raw JSON
// string emitted by the model; for almost every tool, the helpers below let
// you skip touching it and just declare typed parameters.
struct ToolCall {
    std::string name;
    std::string arguments_json;
    std::string id;
};

using ToolHandler = std::function<ToolResult(const ToolCall &)>;

// A Tool is name + description + JSON-schema parameters + handler.
//
// Two ways to build one:
//   1. Tool::make("name", "desc", R"({"type":"object",...})", handler);
//   2. Tool::builder("name").describe("desc")
//                           .param("query", "string", "what to search", true)
//                           .handle([](const ToolCall & c){ ... }).build();
struct Tool {
    std::string name;
    std::string description;
    std::string parameters_json;   // JSON schema (object)
    ToolHandler handler;

    static Tool make(std::string n, std::string d, std::string p, ToolHandler h) {
        return Tool{ std::move(n), std::move(d), std::move(p), std::move(h) };
    }

    class Builder {
       public:
        explicit Builder(std::string name) : name_(std::move(name)) {}

        Builder & describe(std::string d) { desc_ = std::move(d); return *this; }

        // Add a typed parameter to the JSON schema.
        // type: "string" | "integer" | "number" | "boolean" | "array" | "object"
        Builder & param(std::string name,
                        std::string type,
                        std::string description = "",
                        bool        required    = false) {
            params_.push_back({ std::move(name), std::move(type),
                                std::move(description), required });
            return *this;
        }

        Builder & handle(ToolHandler h) { handler_ = std::move(h); return *this; }

        Tool build() const;  // see tool.cpp

       private:
        struct P { std::string name, type, description; bool required; };
        std::string    name_;
        std::string    desc_;
        std::vector<P> params_;
        ToolHandler    handler_;
    };

    static Builder builder(std::string name) { return Builder(std::move(name)); }
};

// Tiny JSON-arg helpers so handlers don't need a JSON dep.
// They scan the raw arguments_json for top-level "key":value pairs.
// Return std::nullopt-style behavior via the bool return.
namespace args {
    bool get_string(const std::string & json, const std::string & key, std::string & out);
    bool get_int   (const std::string & json, const std::string & key, long long  & out);
    bool get_double(const std::string & json, const std::string & key, double     & out);
    bool get_bool  (const std::string & json, const std::string & key, bool       & out);
}

}  // namespace easyai
