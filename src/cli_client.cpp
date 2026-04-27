// libeasyai-cli-side overload: Toolbelt::apply(Client &).  Kept in this
// translation unit (linked into libeasyai-cli) so the engine-only lib
// doesn't drag in the HTTP client.
#include "easyai/cli.hpp"
#include "easyai/client.hpp"

namespace easyai::cli {

void Toolbelt::apply(Client & client) const {
    for (auto & t : tools()) client.add_tool(t);
    if (allow_bash_) client.max_tool_hops(99999);
}

}  // namespace easyai::cli
