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

static constexpr int kMaxBatch     = 20;

// Per-step 'text' byte cap. Rejects the single-text-stuffed-numbered-
// list anti-pattern (e.g. "1. analyze X 2. research Y 3. modernize Z…")
// that small models fall into when they don't realise items=[…] is the
// way to track multi-step work. 80 bytes comfortably fits a short
// imperative phrase in any Latin script while making "more than one
// step in one field" impossible to encode silently.
static constexpr int kMaxTextChars = 80;

// Build the rejection message for an over-long 'text'. Echoes the head
// of the offending string so the model can see exactly what tripped the
// guard and self-correct on retry without another tool round-trip.
static std::string text_too_long_error(const std::string & text,
                                       const std::string & where) {
    std::string preview = text.substr(0, 60);
    if (text.size() > 60) preview += "…";
    // Replace any newlines / tabs in the preview so the error reads as
    // one line in the model's tool_result view (the raw text the model
    // sent often has the embedded numbered list on a single physical
    // line, but defensive in case it doesn't).
    for (char & c : preview) {
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    }
    return "plan: " + where + " 'text' is "
         + std::to_string(text.size()) + " chars (max "
         + std::to_string(kMaxTextChars) + "). Each step is a SHORT "
         "imperative phrase, NEVER a numbered list. Split this into "
         "multiple items via items=[{text:\"step 1\"},{text:\"step 2\"},…] "
         "instead. Got: \"" + preview + "\"";
}

// Strip C0 control bytes (incl. ESC, 0x1b) and DEL from model-supplied
// plan-item text before we render it. Preserves UTF-8 multi-byte
// sequences (all 0x80+ bytes pass through). Without this guard a model
// could emit "\x1b]0;HACKED\a" or "\x1b[2J" inside `text` and hijack
// the operator's terminal — the plan render path uses our own ANSI
// codes for status colouring, so we hand the terminal escapes-by-trust
// only for our literal box brackets.  Tabs, CR, and LF are also
// stripped because plan items are conceptually one-line — preserving
// them would break the `- [x] N. text` row layout.
static std::string sanitize_plan_text(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    for (unsigned char c : in) {
        if (c < 0x20 || c == 0x7f) {
            // Drop. (Whitespace already collapsed; see comment above.)
        } else {
            out += static_cast<char>(c);
        }
    }
    return out;
}

Plan::Plan() = default;

void Plan::on_change(ChangeCallback cb) { on_change_ = std::move(cb); }

const std::vector<PlanItem> & Plan::items() const { return items_; }
bool                          Plan::empty() const { return items_.empty(); }

void Plan::render(std::ostream & out, bool color) const {
    if (items_.empty()) { out << "(plan is empty)\n"; return; }
    for (const auto & it : items_) {
        // Strip control bytes from model-supplied text — see
        // sanitize_plan_text() for rationale (terminal-escape injection).
        const std::string text = sanitize_plan_text(it.text);
        if (color) {
            if (it.status == "deleted") {
                out << "\033[2;9m- [-] " << it.id << ". " << text << "\033[0m\n";
            } else if (it.status == "error") {
                out << "\033[31m- [!] " << it.id << ". " << text << "\033[0m\n";
            } else if (it.status == "done") {
                out << "\033[2m- [x] " << it.id << ". " << text << "\033[0m\n";
            } else if (it.status == "working") {
                out << "\033[1;36m- [~] " << it.id << ". " << text << "\033[0m\n";
            } else {
                out << "\033[1m- [ ] " << it.id << ". " << text << "\033[0m\n";
            }
        } else {
            const char * box =
                it.status == "done"    ? "[x]" :
                it.status == "working" ? "[~]" :
                it.status == "error"   ? "[!]" :
                it.status == "deleted" ? "[-]" :
                                         "[ ]";
            out << "- " << box << " " << it.id << ". " << text << "\n";
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
            R"("description":"REQUIRED. EXACTLY one of: \"add\" | \"update\" | )"
            R"(\"delete\" | \"list\". Each action requires a different subset )"
            R"(of the other fields — see the tool description for the )"
            R"(per-action contract. NEVER pass any other string."},)"
        R"("items":{"type":"array","maxItems":20,)"
            R"("description":"Used by add / update / delete to BATCH several )"
            R"(mutations in ONE tool call. MUST be a real JSON array, NOT a )"
            R"(quoted string containing JSON. Max 20 entries. Per action: )"
            R"(add → each entry is {text} with optional status; update → each )"
            R"(entry is {id} plus text and/or status; delete → each entry is )"
            R"({id}. When 'items' is set, 'id'/'text'/'status' at the top )"
            R"(level are ignored.",)"
            R"("items":{"type":"object","properties":{)"
                R"("id":{"type":"string"},)"
                R"("text":{"type":"string","maxLength":80,)"
                    R"("description":"HARD LIMIT 80 chars. ONE short )"
                    R"(imperative phrase per entry. NEVER a numbered list."},)"
                R"("status":{"type":"string",)"
                    R"("enum":["pending","working","done","error","deleted"]})"
            R"(}}},)"
        R"("id":{"type":"string","description":)"
            R"("Used by single update / delete. MUST be the integer id shown )"
            R"(by the last 'list' / 'add' result, passed as a string )"
            R"((e.g. \"3\"). NEVER invent an id — only ids you have just )"
            R"(seen in a tool result are valid. With action='delete' the )"
            R"(literal string \"all\" wipes every step at once."},)"
        R"("text":{"type":"string","maxLength":80,"description":)"
            R"("Used by single add (the new step's content) or single update )"
            R"((replacement text for the existing step). HARD LIMIT: 80 )"
            R"(characters per step. MUST be ONE SHORT IMPERATIVE PHRASE )"
            R"((e.g. \"fetch arxiv index\", \"write Makefile\"). NEVER stuff )"
            R"(a numbered list, comma-separated list, or multi-step plan into )"
            R"(this field — split into separate items via )"
            R"(items=[{text:\"step 1\"},{text:\"step 2\"},…]. The server will )"
            R"(REJECT any text > 80 chars with an error. Control characters )"
            R"(are stripped."},)"
        R"("status":{"type":"string",)"
            R"("enum":["pending","working","done","error","deleted"],)"
            R"("description":"Used by single update, or as an optional field )"
            R"(on entries in add. EXACTLY one of: \"pending\" (default — not )"
            R"(started), \"working\" (in progress; AT MOST ONE step should be )"
            R"('working' at any moment), \"done\" (completed successfully), )"
            R"(\"error\" (attempted and failed), \"deleted\" (struck through; )"
            R"(prefer action='delete' over this status). To mark a step done, )"
            R"(call update with status='done' — NEVER call add a second time."})"
        R"(},"required":["action"]})";

    Plan * self = this;
    return Tool::make(
        "plan",
        // Tool description — kept strict and unambiguous on purpose so even
        // small / quantised models invoke it correctly without guessing.
        // Every action contract is spelled out with a worked example; rules
        // that have caused real misuse in production (re-adding to mark
        // done, inventing ids, stringified items array) are repeated as
        // hard NEVER lines because models routinely ignore advisory tone.
        "Tracks a multi-step plan the user sees live (rendered above the "
        "assistant's reply). USE this tool whenever a task has 3+ distinct "
        "steps you can name in advance, or whenever the user asks you to "
        "plan / track / break down work. SKIP for one-shot answers, single "
        "tool calls, or pure information lookups.\n"
        "\n"
        "CRITICAL — never violate:\n"
        "  • Each step's 'text' is HARD-CAPPED at 80 characters AND must "
        "be ONE short imperative phrase (e.g. \"write Makefile\", "
        "\"fetch arxiv index\"). NEVER cram a numbered list, comma list, "
        "or multi-step plan into a single 'text' field — the server will "
        "REJECT it. To track several steps at once, pass them as separate "
        "entries: items=[{text:\"step 1\"},{text:\"step 2\"},…].\n"
        "  • Each step has an INTEGER id assigned by 'add' (rendered as "
        "\"1\", \"2\", \"3\", …). IDs are stable and never change.\n"
        "  • To mark a step STARTED:  action=\"update\", id=\"N\", "
        "status=\"working\".  AT MOST ONE step is 'working' at a time — "
        "mark the current one 'done' (or 'error') BEFORE starting the "
        "next.\n"
        "  • To mark a step FINISHED: action=\"update\", id=\"N\", "
        "status=\"done\".  NEVER call action=\"add\" a second time for the "
        "same step — that creates a duplicate with a new id.\n"
        "  • NEVER invent ids. If you don't know which ids exist, fire "
        "action=\"list\" first (read-only, no side effects) and then "
        "update by the ids you saw.\n"
        "  • Prefer ONE batched call (items=[…]) over N separate "
        "tool_calls when mutating several steps at once.\n"
        "\n"
        "Pick EXACTLY ONE 'action':\n"
        "\n"
        "  action=\"add\"  — append new pending step(s).\n"
        "    Single step:  {action:\"add\", text:\"do X\"}\n"
        "    Batch:        {action:\"add\", items:[{text:\"a\"},"
        "{text:\"b\"},{text:\"c\"}]}\n"
        "    Optional per item: status (default \"pending\"; pass "
        "\"working\" to start it immediately in the same call).\n"
        "    NEVER use 'add' to mark a step done — use 'update'.\n"
        "\n"
        "  action=\"update\"  — change text and/or status of EXISTING "
        "steps by id.\n"
        "    Mark started: {action:\"update\", id:\"1\", "
        "status:\"working\"}\n"
        "    Mark done:    {action:\"update\", id:\"1\", "
        "status:\"done\"}\n"
        "    Rename+done:  {action:\"update\", id:\"3\", "
        "text:\"refined wording\", status:\"done\"}\n"
        "    Batch:        {action:\"update\", items:[{id:\"1\","
        "status:\"done\"},{id:\"2\",status:\"working\"}]}\n"
        "\n"
        "  action=\"delete\"  — remove step(s).\n"
        "    Single:  {action:\"delete\", id:\"2\"}\n"
        "    Wipe all:{action:\"delete\", id:\"all\"}\n"
        "    Batch:   {action:\"delete\", items:[{id:\"2\"},{id:\"4\"}]}\n"
        "\n"
        "  action=\"list\"  — return the current plan. READ-ONLY, NO side "
        "effects. Use when you've lost track of ids or step state.\n"
        "    {action:\"list\"}\n"
        "\n"
        "Statuses (status enum): \"pending\" (default), \"working\" (in "
        "progress — exactly one at a time), \"done\" (completed), "
        "\"error\" (attempted and failed), \"deleted\" (struck through; "
        "prefer action=\"delete\" instead).\n"
        "\n"
        "Format rules:\n"
        "  • 'action' is REQUIRED. Must be one of: add | update | delete | "
        "list.\n"
        "  • 'items' MUST be a real JSON array — NEVER a quoted string "
        "containing JSON. Max 20 entries per call.\n"
        "  • 'id' MUST be an integer id from the most recent list/add "
        "result, passed as a string (e.g. \"3\"); for delete only, the "
        "literal \"all\" wipes every step.\n"
        "  • When 'items' is provided, top-level 'id' / 'text' / 'status' "
        "are ignored.",
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
                    int idx = 0;
                    for (const auto & e : elems) {
                        ++idx;
                        std::string t = args::get_string_or(e, "text", "");
                        if (t.empty()) continue;
                        if ((int) t.size() > kMaxTextChars)
                            return ToolResult::error(text_too_long_error(
                                t, "add items[" + std::to_string(idx - 1) + "]"));
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
                if ((int) text.size() > kMaxTextChars)
                    return ToolResult::error(text_too_long_error(text, "add"));
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
                    int idx = 0;
                    for (const auto & e : elems) {
                        ++idx;
                        std::string id = args::get_string_or(e, "id", "");
                        if (id.empty()) continue;
                        std::string t = args::get_string_or(e, "text", "");
                        std::string s = args::get_string_or(e, "status", "");
                        if (!t.empty() && (int) t.size() > kMaxTextChars)
                            return ToolResult::error(text_too_long_error(
                                t, "update items[" + std::to_string(idx - 1) + "]"));
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
                if (!text.empty() && (int) text.size() > kMaxTextChars)
                    return ToolResult::error(text_too_long_error(text, "update"));
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
