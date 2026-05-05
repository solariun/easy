// easyai/plan.hpp — agent planning helper as a single multi-action Tool.
//
// A Plan is a typed list of {id, text, status} items the model can
// manipulate via one Tool with sub-actions.  When wired into either an
// Engine or a Client, the model can:
//
//   plan(action="add",    text="...")                 → new pending step
//   plan(action="add",    items=[{text}, ...])        → batch add (max 20)
//   plan(action="update", id="...", status="done")    → change status/text
//   plan(action="update", items=[{id,status?,...}])   → batch update
//   plan(action="delete", id="..." | id="all")        → mark deleted / clear
//   plan(action="delete", items=[{id}, ...])          → batch delete
//   plan(action="list")                               → return checklist
//
// Statuses: pending, working, done, error, deleted.
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
    std::string status;   // "pending" | "working" | "done" | "error" | "deleted"
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

    // Subscribers fire on every mutation (add/update/delete/clear), or
    // once at the end of a Batch if one is active.
    void on_change(ChangeCallback cb);

    // RAII guard: while alive, suppresses on_change notifications and
    // coalesces them into a single fire when destroyed (only if any
    // mutation actually happened during the batch). Nestable; the
    // outermost scope is what drives the single fire.
    //
    //   { Plan::Batch b(plan);
    //     for (...) plan.add(...);   // no per-item callback
    //   }                            // one callback here
    class Batch {
    public:
        explicit Batch(Plan & p);
        ~Batch();
        Batch(const Batch &)             = delete;
        Batch & operator=(const Batch &) = delete;
    private:
        Plan * p_;
    };

    // Read-only access.
    const std::vector<PlanItem> & items() const;
    bool empty() const;

    // Render as a GitHub-style markdown checklist.
    // When color=true, emits ANSI codes: bold for active items, dim for
    // done, red for error, strikethrough+dim for deleted.
    void render(std::ostream & out, bool color = false) const;
    std::string render_string(bool color = false) const;

    // Manual mutation (for callers that want to seed items before the
    // model takes over, or reset between runs).
    std::string add   (std::string text);     // returns the new id
    bool        update(const std::string & id,
                       const std::string & text,
                       const std::string & status);
    bool        remove(const std::string & id);  // marks as "deleted"
    bool        start (const std::string & id);  // shorthand → "working"
    bool        done  (const std::string & id);  // shorthand → "done"
    void        clear ();

private:
    std::vector<PlanItem>          items_;
    int                            next_id_ = 1;
    ChangeCallback                 on_change_;
    int                            batch_depth_ = 0;
    bool                           batch_dirty_ = false;

    void fire_changed_();
};

}  // namespace easyai
