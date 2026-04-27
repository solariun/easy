cmake_minimum_required(VERSION 3.18)

# Single-source-of-truth version — bumped via `git tag` + this line.
project(easyai
    VERSION   0.1.0
    LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EASYAI_BUILD_EXAMPLES "Build easyai example agents"             ON)
option(EASYAI_WITH_CURL      "Build web_fetch / web_search"            ON)
option(EASYAI_BUILD_WEBUI    "Embed the llama-server-style webui"      ON)
option(EASYAI_INSTALL        "Generate install + find_package targets" ON)

# ------------------------------------------------------------------ llama.cpp
# Pulled in as a sibling directory; we disable its own apps so the build
# is fast and we only get the libraries we need.
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
# We DO need the `common` static library — it provides chat templates,
# tool-call parsing and the high-level sampler used by easyai.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)

# GPU backends — auto-enable per platform, user can override via -D...=OFF.
if (APPLE)
    set(GGML_METAL              ON  CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()
option(GGML_VULKAN "ggml: enable Vulkan backend" OFF)

if (NOT TARGET llama)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp
                     ${CMAKE_BINARY_DIR}/_deps/llama.cpp
                     EXCLUDE_FROM_ALL)
endif()

# ============================================================================
# libeasyai — local llama.cpp engine library
# ============================================================================
set(EASYAI_SOURCES
    src/engine.cpp
    src/tool.cpp
    src/builtin_tools.cpp
    src/presets.cpp
    src/plan.cpp
)

add_library(easyai ${EASYAI_SOURCES})
add_library(easyai::engine ALIAS easyai)
set_target_properties(easyai PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "easyai")
target_include_directories(easyai
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# BUILD_INTERFACE wraps the link so the dep doesn't leak into the
# exported `easyaiTargets.cmake` — downstream consumers don't see
# llama / llama-common as required CMake targets, they just link
# libllama.so / libllama-common.so transparently at runtime via the
# installed libeasyai.so's RPATH/system loader.
target_link_libraries(easyai PRIVATE
    "$<BUILD_INTERFACE:llama>"
    "$<BUILD_INTERFACE:llama-common>")
target_compile_features(easyai PUBLIC cxx_std_17)

if (EASYAI_WITH_CURL)
    find_package(CURL QUIET)
    if (CURL_FOUND)
        target_link_libraries(easyai PRIVATE CURL::libcurl)
        target_compile_definitions(easyai PRIVATE EASYAI_HAVE_CURL=1)
        message(STATUS "easyai: libcurl found — web_fetch / web_search enabled")
    else()
        message(WARNING "easyai: libcurl not found — web tools will return errors at runtime")
    endif()
endif()

# ============================================================================
# libeasyai-cli — OpenAI-protocol client library
# ----------------------------------------------------------------------------
# Target NAME is `easyai_cli` (CMake-internal — needs to be different from
# the existing `easyai-cli` executable target below).  OUTPUT_NAME forces
# the on-disk artifact to `libeasyai-cli.so` / `libeasyai-cli.a`.  Public
# alias `easyai::cli` keeps downstream link-lines clean.
# ============================================================================
# NOTE: we compile cpp-httplib's httplib.cpp DIRECTLY into easyai_cli
# instead of linking the vendored cpp-httplib static target.  Reason:
# llama.cpp's cpp-httplib build doesn't define CPPHTTPLIB_OPENSSL_SUPPORT,
# so the static lib has no SSLClient symbols — if we just defined the
# macro on our side and tried to instantiate SSLClient, the linker
# would fail.  Re-compiling the .cpp here lets us add the macro and
# link OpenSSL ourselves, without touching the upstream vendor build
# that easyai-server already relies on.
set(EASYAI_CLI_SOURCES
    src/client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor/cpp-httplib/httplib.cpp
)

add_library(easyai_cli ${EASYAI_CLI_SOURCES})
add_library(easyai::cli ALIAS easyai_cli)
set_target_properties(easyai_cli PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "easyai-cli")
target_include_directories(easyai_cli
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor             # nlohmann/json.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor/cpp-httplib) # httplib.h
target_link_libraries(easyai_cli PUBLIC easyai)   # reuse Tool/ToolCall/ToolResult
target_compile_features(easyai_cli PUBLIC cxx_std_17)

# OpenSSL -> HTTPS support inside libeasyai-cli.  Optional: when libssl
# isn't available we skip the define and Client returns a clear
# "HTTPS not in this build" error at request time.
find_package(OpenSSL QUIET)
if (OpenSSL_FOUND)
    target_compile_definitions(easyai_cli PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT=1)
    target_link_libraries(easyai_cli PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "easyai-cli: OpenSSL found — HTTPS endpoints enabled")
else()
    message(STATUS "easyai-cli: OpenSSL NOT found — HTTPS endpoints will be rejected at runtime")
endif()

# ============================================================================
# examples (binaries)
# ============================================================================
if (EASYAI_BUILD_EXAMPLES)
    add_executable(easyai-chat examples/chat.cpp)
    target_link_libraries(easyai-chat PRIVATE easyai)

    add_executable(easyai-agent examples/agent.cpp)
    target_link_libraries(easyai-agent PRIVATE easyai)

    # Tutorial agent paired with manual.md → "Writing your own tools".
    if (CURL_FOUND)
        add_executable(easyai-recipes examples/recipes.cpp)
        target_link_libraries(easyai-recipes PRIVATE easyai CURL::libcurl)
    endif()

    add_executable(easyai-cli examples/cli.cpp)
    # Links libeasyai-cli (which transitively pulls in libeasyai) so the
    # RemoteBackend can drive a remote OpenAI-compatible endpoint via
    # easyai::Client (HTTP + HTTPS via cpp-httplib + OpenSSL).  libcurl
    # is now only needed for the LOCAL backend's web tools.
    target_link_libraries(easyai-cli PRIVATE easyai_cli)
    target_include_directories(easyai-cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor)
    if (EASYAI_WITH_CURL AND CURL_FOUND)
        target_link_libraries(easyai-cli PRIVATE CURL::libcurl)
        target_compile_definitions(easyai-cli PRIVATE EASYAI_HAVE_CURL=1)
    endif()

    # ----- embed the llama-server-derived webui (xxd-style binary embed) ---
    set(EASYAI_WEBUI_HEADERS "")
    if (EASYAI_BUILD_WEBUI)
        set(WEBUI_ASSETS index.html bundle.js bundle.css loading.html)
        foreach(asset ${WEBUI_ASSETS})
            set(_in  "${CMAKE_CURRENT_SOURCE_DIR}/webui/${asset}")
            set(_out "${CMAKE_CURRENT_BINARY_DIR}/webui_${asset}.hpp")
            list(APPEND EASYAI_WEBUI_HEADERS ${_out})
            add_custom_command(
                DEPENDS "${_in}"
                OUTPUT  "${_out}"
                COMMAND ${CMAKE_COMMAND} -DINPUT=${_in} -DOUTPUT=${_out}
                        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xxd.cmake
                COMMENT "[easyai] embed webui asset: ${asset}"
            )
            set_source_files_properties(${_out} PROPERTIES GENERATED TRUE)
        endforeach()
    endif()

    set(_brand_in  "${CMAKE_CURRENT_SOURCE_DIR}/webui/AI-brain.svg")
    set(_brand_out "${CMAKE_CURRENT_BINARY_DIR}/easyai_brand_svg.hpp")
    list(APPEND EASYAI_WEBUI_HEADERS ${_brand_out})
    add_custom_command(
        DEPENDS "${_brand_in}"
        OUTPUT  "${_brand_out}"
        COMMAND ${CMAKE_COMMAND} -DINPUT=${_brand_in} -DOUTPUT=${_brand_out}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xxd.cmake
        COMMENT "[easyai] embed brand asset: AI-brain.svg"
    )
    set_source_files_properties(${_brand_out} PROPERTIES GENERATED TRUE)

    add_executable(easyai-server examples/server.cpp ${EASYAI_WEBUI_HEADERS})
    target_link_libraries(easyai-server PRIVATE easyai cpp-httplib llama-common)
    target_include_directories(easyai-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor/cpp-httplib
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor
        ${CMAKE_CURRENT_BINARY_DIR})
    if (EASYAI_BUILD_WEBUI)
        target_compile_definitions(easyai-server PRIVATE EASYAI_BUILD_WEBUI=1)
    endif()

    # easyai-cli-remote — agentic OpenAI-protocol client demo, talks to any
    # /v1/chat/completions endpoint (our easyai-server, llama-server, etc.).
    add_executable(easyai-cli-remote examples/cli_remote.cpp)
    target_link_libraries(easyai-cli-remote PRIVATE easyai_cli)
    target_include_directories(easyai-cli-remote PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor)
endif()

# ============================================================================
# install rules + find_package(easyai) support
# ----------------------------------------------------------------------------
# After `cmake --install build`, downstream projects can do:
#
#   find_package(easyai 0.1 REQUIRED)
#   target_link_libraries(myapp PRIVATE easyai::engine easyai::cli)
#
# The lib targets are exported with VERSION/SOVERSION matching the project
# version, public include dir relocated to <prefix>/include, and the cmake
# config files written to <prefix>/lib/cmake/easyai/.
# ============================================================================
if (EASYAI_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # We deliberately DON'T use install(EXPORT) — easyai's transitive
    # link to llama / llama-common from llama.cpp (added via
    # EXCLUDE_FROM_ALL) confuses CMake's exported-target tracker.  Hand-
    # rolling the config + IMPORTED targets sidesteps that and gives
    # downstream projects a clean find_package(easyai) experience.
    install(TARGETS easyai easyai_cli
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    install(DIRECTORY include/easyai
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/easyaiConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/easyaiConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/easyai
        PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR)

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/easyaiConfigVersion.cmake
        VERSION       ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/easyaiConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/easyaiConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/easyai)
endif()
