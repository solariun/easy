// easyai/plan.hpp — agent planning helper as a single multi-action Tool.
//
// A Plan is a typed list of {id, text, status} items the model can
// manipulate via one Tool with sub-actions.  When wired into either an
// Engine or a Client, the model can:
//
//   plan(action="add",   text="...")    → adds a pending item, returns id
//   plan(action="start", id="...")      → marks doing
//   plan(action="done",  id="...")      → marks done
//   plan(action="list")                 → returns the markdown checklist
//
// The Plan owns an in-memory vector and fires on_change after every
// mutation so the UI / stdout can re-render the checklist live.
#pragma once

#include "easyai/tool.hpp"

#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace easyai {

struct PlanItem {
    std::string id;       // monotonic "1", "2", ...
    std::string text;
    std::string status;   // "pending" | "doing" | "done"
};

class Plan {
public:
    using ChangeCallback = std::function<void(const Plan &)>;

    Plan();
    ~Plan() = default;
    Plan(const Plan &)             = delete;
    Plan & operator=(const Plan &) = delete;
    Plan(Plan &&) noexcept            = default;
    Plan & operator=(Plan &&) noexcept = default;

    // Returns a Tool that mutates this Plan.  The returned Tool's
    // handler captures `this` by pointer — keep the Plan alive for as
    // long as the Engine / Client holds the Tool.
    Tool tool();

    // Subscribers fire on every mutation (add/start/done/clear).
    void on_change(ChangeCallback cb);

    // Read-only access.
    const std::vector<PlanItem> & items() const;
    bool empty() const;

    // Render as a GitHub-style markdown checklist (`- [ ] / [~] / [x]`).
    void render(std::ostream & out) const;
    std::string render_string() const;

    // Manual mutation (for callers that want to seed items before the
    // model takes over, or reset between runs).
    std::string add  (std::string text);     // returns the new id
    bool        start(const std::string & id);
    bool        done (const std::string & id);
    void        clear();

private:
    std::vector<PlanItem>          items_;
    int                            next_id_ = 1;
    ChangeCallback                 on_change_;

    void fire_changed_();
};

}  // namespace easyai
