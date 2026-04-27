// plan.cpp — multi-action Plan tool implementation.
//
// Kept dependency-free (no nlohmann::json) so libeasyai's existing
// link-line stays minimal — we use the easyai::args helpers for the
// tool's argument parsing and emit the tool's JSON-schema description
// as a hand-written constant string.
#include "easyai/plan.hpp"
#include "easyai/tool.hpp"

#include <sstream>
#include <utility>

namespace easyai {

Plan::Plan() = default;

void Plan::on_change(ChangeCallback cb) { on_change_ = std::move(cb); }

const std::vector<PlanItem> & Plan::items() const { return items_; }
bool                          Plan::empty() const { return items_.empty(); }

void Plan::render(std::ostream & out) const {
    if (items_.empty()) { out << "(plan is empty)\n"; return; }
    for (const auto & it : items_) {
        const char * box =
            it.status == "done"  ? "[x]" :
            it.status == "doing" ? "[~]" :
                                   "[ ]";
        out << "- " << box << " " << it.id << ". " << it.text << "\n";
    }
}

std::string Plan::render_string() const {
    std::ostringstream s;
    render(s);
    return s.str();
}

std::string Plan::add(std::string text) {
    PlanItem it{ std::to_string(next_id_++), std::move(text), "pending" };
    std::string id = it.id;
    items_.push_back(std::move(it));
    fire_changed_();
    return id;
}

bool Plan::start(const std::string & id) {
    for (auto & it : items_) {
        if (it.id == id) { it.status = "doing"; fire_changed_(); return true; }
    }
    return false;
}

bool Plan::done(const std::string & id) {
    for (auto & it : items_) {
        if (it.id == id) { it.status = "done"; fire_changed_(); return true; }
    }
    return false;
}

void Plan::clear() {
    items_.clear();
    next_id_ = 1;
    fire_changed_();
}

void Plan::fire_changed_() {
    if (on_change_) on_change_(*this);
}

// ---------------------------------------------------------------------------
// Tool exposure — a single tool with an `action` dispatcher.  Keeping it
// as one tool (instead of four separate ones) reduces the model's tool-
// choice fan-out and matches how chat templates render tool descriptions.
// ---------------------------------------------------------------------------
Tool Plan::tool() {
    static const std::string kSchema =
        R"({"type":"object","properties":{)"
        R"("action":{"type":"string","enum":["add","start","done","list"],)"
            R"("description":"What to do: 'add' enqueues a new pending step, )"
            R"('start' marks an existing step as in-progress, 'done' marks it complete, )"
            R"('list' returns the current checklist."},)"
        R"("text":{"type":"string","description":)"
            R"("Required for action='add' — short imperative description of the step )"
            R"((e.g. 'fetch arxiv listing')."},)"
        R"("id":{"type":"string","description":)"
            R"("Required for action='start' or 'done' — id returned by a previous 'add' )"
            R"((e.g. '3').")"
        R"(}},"required":["action"]})";

    Plan * self = this;
    return Tool::make(
        "plan",
        "Maintain an ordered task plan that the user can see live. "
        "Call action='add' to enqueue steps, 'start'/'done' to mark "
        "progress, 'list' to re-print.  Use this for any non-trivial "
        "multi-step task so the user sees what you're doing.",
        kSchema,
        [self](const ToolCall & call) -> ToolResult {
            const std::string action = args::get_string_or(
                call.arguments_json, "action", "");
            if (action == "add") {
                std::string text = args::get_string_or(
                    call.arguments_json, "text", "");
                if (text.empty()) return ToolResult::error("plan: 'add' needs non-empty text");
                std::string id = self->add(std::move(text));
                return ToolResult::ok("added id=" + id + "\n" + self->render_string());
            }
            if (action == "start") {
                std::string id = args::get_string_or(
                    call.arguments_json, "id", "");
                if (id.empty()) return ToolResult::error("plan: 'start' needs id");
                if (!self->start(id)) return ToolResult::error("plan: unknown id " + id);
                return ToolResult::ok("started " + id + "\n" + self->render_string());
            }
            if (action == "done") {
                std::string id = args::get_string_or(
                    call.arguments_json, "id", "");
                if (id.empty()) return ToolResult::error("plan: 'done' needs id");
                if (!self->done(id)) return ToolResult::error("plan: unknown id " + id);
                return ToolResult::ok("done " + id + "\n" + self->render_string());
            }
            if (action == "list") {
                return ToolResult::ok(self->render_string());
            }
            return ToolResult::error("plan: unknown action: " + action);
        });
}

}  // namespace easyai
