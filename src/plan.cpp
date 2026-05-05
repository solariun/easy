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
    if (batch_depth_ > 0) {
        batch_dirty_ = true;
        return;
    }
    if (on_change_) on_change_(*this);
}

// RAII batch guard — coalesces on_change callbacks across a sequence of
// mutations into a single fire at scope exit. Nestable; only the
// outermost scope actually fires.
Plan::Batch::Batch(Plan & p) : p_(&p) { ++p_->batch_depth_; }
Plan::Batch::~Batch() {
    if (--p_->batch_depth_ == 0 && p_->batch_dirty_) {
        p_->batch_dirty_ = false;
        if (p_->on_change_) p_->on_change_(*p_);
    }
}

// ---------------------------------------------------------------------------
// Tool exposure — a single tool with an `action` dispatcher.
// ---------------------------------------------------------------------------
Tool Plan::tool() {
    static const std::string kSchema =
        R"({"type":"object","properties":{)"
        R"("action":{"type":"string","enum":["add","update","delete","list"],)"
            R"("description":"Required. One of: \"add\", \"update\", \"delete\", )"
            R"(\"list\". Each action consumes a subset of the other parameters; )"
            R"(see the tool description for the per-action requirements."},)"
        R"("items":{"type":"array","maxItems":20,)"
            R"("description":"Used by add / update / delete. Real JSON array )"
            R"((NOT a quoted string), max 20 entries. add: each entry is )"
            R"({text} with optional status. update: each entry is {id} plus )"
            R"(text and/or status to change. delete: each entry is {id}. )"
            R"(See tool description for examples.",)"
            R"("items":{"type":"object","properties":{)"
                R"("id":{"type":"string"},)"
                R"("text":{"type":"string"},)"
                R"("status":{"type":"string",)"
                    R"("enum":["pending","working","done","error","deleted"]})"
            R"(}}},)"
        R"("id":{"type":"string","description":)"
            R"("Used by single update / delete. The integer id shown in the )"
            R"(rendered plan, passed as a string. With delete, pass \"all\" )"
            R"(to clear the entire plan."},)"
        R"("text":{"type":"string","description":)"
            R"("Used by single add (the new step's content) or single update )"
            R"((the replacement text for the existing step)."},)"
        R"("status":{"type":"string",)"
            R"("enum":["pending","working","done","error","deleted"],)"
            R"("description":"Used by single update, or as an optional field )"
            R"(on items in add. One of: pending (default), working (in )"
            R"(progress), done, error, deleted (struck through)."})"
        R"(},"required":["action"]})";

    Plan * self = this;
    return Tool::make(
        "plan",
        "Track a step-by-step plan visible to the user in real time. Pick an "
        "action; the parameters needed depend on which action you choose. "
        "Four actions are supported:\n"
        "\n"
        "  action=\"add\"\n"
        "    Append new pending steps to the plan. Required: text (single "
        "step) OR items (batch, real JSON array). Optional per item: status "
        "— defaults to \"pending\"; pass \"working\" to start the step in "
        "the same call. Examples:\n"
        "      {action:\"add\", text:\"step one\"}\n"
        "      {action:\"add\", items:[{text:\"a\"},{text:\"b\"}]}\n"
        "      {action:\"add\", items:[{text:\"first\",status:\"working\"},"
        "{text:\"next\"}]}\n"
        "\n"
        "  action=\"update\"\n"
        "    Change text or status of existing steps by id. Required: id "
        "(single) plus at least one of text / status, OR items (batch). Use "
        "the integer ids shown in the rendered plan. Examples:\n"
        "      {action:\"update\", id:\"1\", status:\"working\"}\n"
        "      {action:\"update\", id:\"1\", status:\"done\"}\n"
        "      {action:\"update\", id:\"3\", text:\"refined wording\", "
        "status:\"done\"}\n"
        "      {action:\"update\", items:[{id:\"1\",status:\"done\"},"
        "{id:\"2\",status:\"working\"}]}\n"
        "\n"
        "  action=\"delete\"\n"
        "    Remove steps. Required: id (single, or the special \"all\" to "
        "clear the entire plan), OR items (batch of {id}). Examples:\n"
        "      {action:\"delete\", id:\"2\"}\n"
        "      {action:\"delete\", id:\"all\"}\n"
        "      {action:\"delete\", items:[{id:\"2\"},{id:\"4\"}]}\n"
        "\n"
        "  action=\"list\"\n"
        "    Return the current plan. No other parameters. Example:\n"
        "      {action:\"list\"}\n"
        "\n"
        "Statuses: pending (default), working (in progress), done, error, "
        "deleted (struck through).\n"
        "\n"
        "NEVER re-add an existing step — use update to change its text or "
        "status. Re-adding creates a duplicate with a new id.\n"
        "\n"
        "The 'items' array MUST be a real JSON array, not a quoted string. "
        "Pass at most 20 items per call.",
        kSchema,
        [self](const ToolCall & call) -> ToolResult {
            // Coalesce all mutations in this call into a single
            // on_change notification — without this the UI re-renders
            // the plan once per item in a batch.
            Plan::Batch batch(*self);

            std::string action = args::get_string_or(
                call.arguments_json, "action", "");

            // Synonym tolerance — small models often pick a near-miss verb.
            if      (action == "create" || action == "append" ||
                     action == "insert" || action == "new")    action = "add";
            else if (action == "modify" || action == "change" ||
                     action == "edit"   || action == "set")    action = "update";
            else if (action == "remove" || action == "rm")     action = "delete";
            else if (action == "show"   || action == "get" ||
                     action == "view")                         action = "list";

            // Inference — many smaller models omit 'action' and rely on
            // 'items' to convey intent. Disambiguate via plan state:
            //   id missing or unknown    → add
            //   id known + text/status   → update
            //   id known, no text/status → delete
            // Top-level fields fall back to the same heuristic.
            if (action.empty()) {
                std::vector<std::string> probe;
                if (args::get_array(call.arguments_json, "items", probe) &&
                    !probe.empty()) {
                    std::string first_id = args::get_string_or(probe[0], "id", "");
                    bool it_text   = args::has(probe[0], "text");
                    bool it_status = args::has(probe[0], "status");
                    bool id_known = false;
                    if (!first_id.empty()) {
                        for (const auto & it : self->items()) {
                            if (it.id == first_id) { id_known = true; break; }
                        }
                    }
                    if (!id_known)                action = "add";
                    else if (it_text || it_status) action = "update";
                    else                           action = "delete";
                } else if (args::has(call.arguments_json, "text")) {
                    action = "add";
                } else if (args::has(call.arguments_json, "status")) {
                    action = "update";
                } else if (args::has(call.arguments_json, "id")) {
                    action = "delete";
                } else {
                    action = "list";
                }
            }

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
                        // Honor an optional per-item status so models can
                        // create + mark "working" in one call.
                        std::string s = args::get_string_or(e, "status", "");
                        if (!s.empty() && s != "pending")
                            self->update(id, "", s);
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
