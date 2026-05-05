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
#include <vector>

namespace easyai {

static constexpr int kMaxBatch = 20;

Plan::Plan() = default;

void Plan::on_change(ChangeCallback cb) { on_change_ = std::move(cb); }

const std::vector<PlanItem> & Plan::items() const { return items_; }
bool                          Plan::empty() const { return items_.empty(); }

void Plan::render(std::ostream & out, bool color) const {
    if (items_.empty()) { out << "(plan is empty)\n"; return; }
    for (const auto & it : items_) {
        if (color) {
            if (it.status == "deleted") {
                out << "\033[2;9m- [-] " << it.id << ". " << it.text << "\033[0m\n";
            } else if (it.status == "error") {
                out << "\033[31m- [!] " << it.id << ". " << it.text << "\033[0m\n";
            } else if (it.status == "done") {
                out << "\033[2m- [x] " << it.id << ". " << it.text << "\033[0m\n";
            } else if (it.status == "working") {
                out << "\033[1;36m- [~] " << it.id << ". " << it.text << "\033[0m\n";
            } else {
                out << "\033[1m- [ ] " << it.id << ". " << it.text << "\033[0m\n";
            }
        } else {
            const char * box =
                it.status == "done"    ? "[x]" :
                it.status == "working" ? "[~]" :
                it.status == "error"   ? "[!]" :
                it.status == "deleted" ? "[-]" :
                                         "[ ]";
            out << "- " << box << " " << it.id << ". " << it.text << "\n";
        }
    }
}

std::string Plan::render_string(bool color) const {
    std::ostringstream s;
    render(s, color);
    return s.str();
}

std::string Plan::add(std::string text) {
    PlanItem it{ std::to_string(next_id_++), std::move(text), "pending" };
    std::string id = it.id;
    items_.push_back(std::move(it));
    fire_changed_();
    return id;
}

bool Plan::update(const std::string & id,
                  const std::string & text,
                  const std::string & status) {
    for (auto & it : items_) {
        if (it.id == id) {
            if (!text.empty())   it.text   = text;
            if (!status.empty()) it.status = status;
            fire_changed_();
            return true;
        }
    }
    return false;
}

bool Plan::remove(const std::string & id) {
    for (auto & it : items_) {
        if (it.id == id) { it.status = "deleted"; fire_changed_(); return true; }
    }
    return false;
}

bool Plan::start(const std::string & id) {
    for (auto & it : items_) {
        if (it.id == id) { it.status = "working"; fire_changed_(); return true; }
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
// Tool exposure — a single tool with an `action` dispatcher.
// ---------------------------------------------------------------------------
Tool Plan::tool() {
    static const std::string kSchema =
        R"({"type":"object","properties":{)"
        R"("action":{"type":"string","enum":["add","update","delete","list"],)"
            R"("description":"add: append new steps (pending). )"
            R"(update: change text or status of existing steps by id. )"
            R"(delete: remove steps by id (id='all' clears everything). )"
            R"(list: return the current plan."},)"
        R"("items":{"type":"array","maxItems":20,)"
            R"("description":"Batch mode: real JSON array (NOT a string). )"
            R"(add  example: [{\"text\":\"step one\"},{\"text\":\"step two\"}]. )"
            R"(update example: [{\"id\":\"1\",\"status\":\"done\"}]. )"
            R"(delete example: [{\"id\":\"2\"}].",)"
            R"("items":{"type":"object","properties":{)"
                R"("id":{"type":"string"},)"
                R"("text":{"type":"string"},)"
                R"("status":{"type":"string",)"
                    R"("enum":["pending","working","done","error","deleted"]})"
            R"(}}},)"
        R"("id":{"type":"string","description":)"
            R"("Item id for single update/delete. Use 'all' with delete to clear."},)"
        R"("text":{"type":"string","description":)"
            R"("Step text for single add, or new text for single update."},)"
        R"("status":{"type":"string",)"
            R"("enum":["pending","working","done","error","deleted"],)"
            R"("description":"New status for single update."}")"
        R"(},"required":["action"]})";

    Plan * self = this;
    return Tool::make(
        "plan",
        "Track a step-by-step plan visible to the user in real time. "
        "Simplest forms (prefer these): "
        "{action:\"add\",text:\"step\"}, "
        "{action:\"update\",id:\"1\",status:\"working\"}, "
        "{action:\"update\",id:\"1\",status:\"done\"}, "
        "{action:\"delete\",id:\"all\"}, "
        "{action:\"list\"}. "
        "Batch form uses a real JSON array in 'items' (not a quoted string): "
        "{action:\"add\",items:[{text:\"a\"},{text:\"b\"}]}. "
        "Statuses: pending (default), working, done, error, deleted. "
        "Never re-add an existing step — use update to change its text or status.",
        kSchema,
        [self](const ToolCall & call) -> ToolResult {
            const std::string action = args::get_string_or(
                call.arguments_json, "action", "");

            // ---- add ----
            if (action == "add") {
                std::vector<std::string> elems;
                if (args::get_array(call.arguments_json, "items", elems)) {
                    if (elems.empty())
                        return ToolResult::error("plan: 'add' items array is empty");
                    if ((int) elems.size() > kMaxBatch)
                        return ToolResult::error("plan: max 20 items per call");
                    std::string ids;
                    for (const auto & e : elems) {
                        std::string t = args::get_string_or(e, "text", "");
                        if (t.empty()) continue;
                        std::string id = self->add(std::move(t));
                        if (!ids.empty()) ids += ",";
                        ids += id;
                    }
                    if (ids.empty())
                        return ToolResult::error("plan: no valid items to add");
                    return ToolResult::ok("added ids=" + ids + "\n" +
                                          self->render_string());
                }
                std::string text = args::get_string_or(
                    call.arguments_json, "text", "");
                if (text.empty())
                    return ToolResult::error(
                        "plan: 'add' needs either text or items. Examples: "
                        "{action:\"add\",text:\"my step\"} or "
                        "{action:\"add\",items:[{text:\"a\"},{text:\"b\"}]}. "
                        "items must be a real JSON array, not a quoted string.");
                std::string id = self->add(std::move(text));
                return ToolResult::ok("added id=" + id + "\n" +
                                      self->render_string());
            }

            // ---- update ----
            if (action == "update") {
                std::vector<std::string> elems;
                if (args::get_array(call.arguments_json, "items", elems)) {
                    if (elems.empty())
                        return ToolResult::error("plan: 'update' items array is empty");
                    if ((int) elems.size() > kMaxBatch)
                        return ToolResult::error("plan: max 20 items per call");
                    int ok_count = 0;
                    for (const auto & e : elems) {
                        std::string id = args::get_string_or(e, "id", "");
                        if (id.empty()) continue;
                        std::string t = args::get_string_or(e, "text", "");
                        std::string s = args::get_string_or(e, "status", "");
                        if (self->update(id, t, s)) ++ok_count;
                    }
                    return ToolResult::ok("updated " + std::to_string(ok_count) +
                                          " items\n" + self->render_string());
                }
                std::string id = args::get_string_or(
                    call.arguments_json, "id", "");
                if (id.empty())
                    return ToolResult::error(
                        "plan: 'update' needs either id or items. Examples: "
                        "{action:\"update\",id:\"1\",status:\"done\"} or "
                        "{action:\"update\",items:[{id:\"1\",status:\"done\"}]}.");
                std::string text = args::get_string_or(
                    call.arguments_json, "text", "");
                std::string status = args::get_string_or(
                    call.arguments_json, "status", "");
                if (!self->update(id, text, status))
                    return ToolResult::error("plan: unknown id " + id);
                return ToolResult::ok("updated " + id + "\n" +
                                      self->render_string());
            }

            // ---- delete ----
            if (action == "delete") {
                std::string id = args::get_string_or(
                    call.arguments_json, "id", "");
                if (id == "all") {
                    self->clear();
                    return ToolResult::ok("plan cleared\n" +
                                          self->render_string());
                }
                std::vector<std::string> elems;
                if (args::get_array(call.arguments_json, "items", elems)) {
                    if (elems.empty())
                        return ToolResult::error("plan: 'delete' items array is empty");
                    if ((int) elems.size() > kMaxBatch)
                        return ToolResult::error("plan: max 20 items per call");
                    int ok_count = 0;
                    for (const auto & e : elems) {
                        std::string eid = args::get_string_or(e, "id", "");
                        if (!eid.empty() && self->remove(eid)) ++ok_count;
                    }
                    return ToolResult::ok("deleted " + std::to_string(ok_count) +
                                          " items\n" + self->render_string());
                }
                if (id.empty())
                    return ToolResult::error(
                        "plan: 'delete' needs id, id='all', or items. Examples: "
                        "{action:\"delete\",id:\"2\"}, "
                        "{action:\"delete\",id:\"all\"}, or "
                        "{action:\"delete\",items:[{id:\"2\"}]}.");
                if (!self->remove(id))
                    return ToolResult::error("plan: unknown id " + id);
                return ToolResult::ok("deleted " + id + "\n" +
                                      self->render_string());
            }

            // ---- list ----
            if (action == "list") {
                return ToolResult::ok(self->render_string());
            }

            return ToolResult::error("plan: unknown action: " + action);
        });
}

}  // namespace easyai
