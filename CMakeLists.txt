cmake_minimum_required(VERSION 3.18)
project(easyai CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EASYAI_BUILD_EXAMPLES "Build easyai example agents"             ON)
option(EASYAI_WITH_CURL      "Build web_fetch / web_search"            ON)
option(EASYAI_BUILD_WEBUI    "Embed the llama-server-style webui"      ON)

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

# ------------------------------------------------------------------ easyai lib
set(EASYAI_SOURCES
    src/engine.cpp
    src/tool.cpp
    src/builtin_tools.cpp
    src/presets.cpp
)

add_library(easyai ${EASYAI_SOURCES})
target_include_directories(easyai
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(easyai PUBLIC  llama
                              PRIVATE llama-common)
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

# ------------------------------------------------------------------ examples
if (EASYAI_BUILD_EXAMPLES)
    add_executable(easyai-chat examples/chat.cpp)
    target_link_libraries(easyai-chat PRIVATE easyai)

    add_executable(easyai-agent examples/agent.cpp)
    target_link_libraries(easyai-agent PRIVATE easyai)

    # Tutorial agent paired with manual.md → "Writing your own tools".
    # Demonstrates a zero-parameter tool (today_is) and an HTTP-calling
    # tool (weather → wttr.in).  Only built when libcurl is available
    # because the weather recipe links to it directly.
    if (CURL_FOUND)
        add_executable(easyai-recipes examples/recipes.cpp)
        target_link_libraries(easyai-recipes PRIVATE easyai CURL::libcurl)
    endif()

    add_executable(easyai-cli examples/cli.cpp)
    target_link_libraries(easyai-cli PRIVATE easyai)
    # nlohmann::json is vendored by llama.cpp at vendor/nlohmann/json.hpp
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

    # Embed our brain logo SVG as a default favicon + brand-icon source so
    # the binary ships visually-branded out of the box (no --webui-icon
    # required).
    set(_brand_in  "${CMAKE_CURRENT_SOURCE_DIR}/src/AI-brain.svg")
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
    # cpp-httplib's include dir isn't PUBLIC-exported; reach in directly.
    target_include_directories(easyai-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor/cpp-httplib
        ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor
        ${CMAKE_CURRENT_BINARY_DIR})           # for webui_*.hpp
    if (EASYAI_BUILD_WEBUI)
        target_compile_definitions(easyai-server PRIVATE EASYAI_BUILD_WEBUI=1)
    endif()
endif()
