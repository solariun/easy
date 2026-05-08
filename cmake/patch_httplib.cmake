# patch_httplib.cmake — make a build-dir copy of cpp-httplib/httplib.h
# with `process_and_close_socket` moved from `private:` to `protected:`,
# so easyai-server's VerboseServer subclass can override + delegate to
# log per-TCP-connection accept/close events.
#
# Why this script exists: the vendored cpp-httplib (under
# ../llama.cpp/vendor/cpp-httplib/) is shared with the upstream llama.cpp
# build. We don't modify the upstream file in-tree — instead we copy it
# into ${CMAKE_BINARY_DIR}/easyai-cpp-httplib/httplib.h, patch the copy,
# and prepend that directory on the easyai-server target's include path
# so its TU sees the patched header. Other targets (easyai_cli,
# cpp-httplib's own .cpp build) keep using upstream — they don't override
# Server's virtuals so they don't care.
#
# Idempotent: if the source already carries the [easyai-patch] sentinel
# we just copy it as-is. Re-runs cleanly on every configure.
#
# Usage from CMakeLists.txt:
#     execute_process(COMMAND ${CMAKE_COMMAND}
#         -DSOURCE=<path/to/upstream/httplib.h>
#         -DDEST=<path/to/build/easyai-cpp-httplib/httplib.h>
#         -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_httplib.cmake)

if(NOT DEFINED SOURCE OR NOT DEFINED DEST)
    message(FATAL_ERROR "patch_httplib: -DSOURCE=... and -DDEST=... required")
endif()
if(NOT EXISTS "${SOURCE}")
    message(FATAL_ERROR "patch_httplib: source not found: ${SOURCE}")
endif()

file(READ "${SOURCE}" _content)

if(_content MATCHES "easyai-patch")
    # Source already carries the patch (e.g. someone applied it
    # in-place upstream). Copy verbatim — no second-application.
    get_filename_component(_dest_dir "${DEST}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dest_dir}")
    configure_file("${SOURCE}" "${DEST}" COPYONLY)
    return()
endif()

# Upstream form. The exact line we replace, verbatim:
set(_target_line "  virtual bool process_and_close_socket(socket_t sock);")

string(FIND "${_content}" "${_target_line}" _idx)
if(_idx EQUAL -1)
    message(FATAL_ERROR
        "patch_httplib: could not find the target declaration in ${SOURCE}.\n"
        "Has cpp-httplib been bumped to a version that renames or re-shapes\n"
        "process_and_close_socket()? Update cmake/patch_httplib.cmake to\n"
        "match the new declaration.")
endif()

# Replacement: bracket the method with `protected:` / `private:` so the
# subclass can both override AND delegate (calling the base impl is what
# preserves cpp-httplib's accept/keep-alive/parse pipeline).
set(_replacement
"  // [easyai-patch] private -> protected so easyai-server's VerboseServer
  // subclass can override process_and_close_socket and delegate back to
  // the base impl for per-TCP-connection logging. Copy lives in build/
  // dir; upstream file is untouched. See cmake/patch_httplib.cmake.
protected:
  virtual bool process_and_close_socket(socket_t sock);
private:")

string(REPLACE "${_target_line}" "${_replacement}" _patched "${_content}")

get_filename_component(_dest_dir "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${_dest_dir}")
file(WRITE "${DEST}" "${_patched}")
message(STATUS "[easyai] patched httplib.h -> ${DEST}")
