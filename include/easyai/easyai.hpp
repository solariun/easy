// easyai.hpp — single-include convenience header.
//
// libeasyai (this header bundle) covers the LOCAL llama.cpp engine.
// libeasyai-cli ships separately and provides easyai::Client for
// talking to a remote OpenAI-compatible server with the same fluent
// API.  Include "easyai/client.hpp" + link easyai::cli to use it.
#pragma once
#include "engine.hpp"
#include "tool.hpp"
#include "builtin_tools.hpp"
#include "external_tools.hpp"
#include "reg_tools.hpp"
#include "presets.hpp"
#include "plan.hpp"
#include "ui.hpp"
#include "text.hpp"
#include "log.hpp"
#include "cli.hpp"
#include "backend.hpp"
#include "agent.hpp"
