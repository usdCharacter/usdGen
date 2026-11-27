# CheckNoStageInclude.cmake
# Gate B-1 (include half): grep source dirs for forbidden pxr includes.
#
# Inputs:
#   SOURCE_DIR - repo root
#   FORBIDDEN  - regex matching forbidden include paths
#
# The protected source set is fixed (libs/base/usdGenMath, libs/core,
# testutils/usdGenTestUtils). Scanning ZERO files is a false pass and is
# therefore a hard error: every protected dir must exist and contribute at
# least one file.

if(NOT DEFINED SOURCE_DIR OR SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "CheckNoStageInclude: SOURCE_DIR is required")
endif()
if(NOT DEFINED FORBIDDEN OR FORBIDDEN STREQUAL "")
    message(FATAL_ERROR "CheckNoStageInclude: FORBIDDEN regex is required")
endif()

set(_search_dirs
    "${SOURCE_DIR}/libs/base/usdGenMath"
    "${SOURCE_DIR}/libs/core"
    "${SOURCE_DIR}/testutils/usdGenTestUtils"
)

set(_all_files "")
foreach(dir IN LISTS _search_dirs)
    if(NOT EXISTS "${dir}")
        message(FATAL_ERROR
            "CheckNoStageInclude: protected source dir is missing: ${dir}")
    endif()
    # Two separate glob arguments — a single quoted "dir/*.h;dir/*.cpp" string
    # is one invalid pattern and scans zero files (silent false pass).
    file(GLOB_RECURSE _files "${dir}/*.h" "${dir}/*.cpp")
    list(LENGTH _files _n)
    if(_n EQUAL 0)
        message(FATAL_ERROR
            "CheckNoStageInclude: protected source dir contains no .h/.cpp files: ${dir}")
    endif()
    list(APPEND _all_files ${_files})
    foreach(f IN LISTS _files)
        file(STRINGS "${f}" _lines)
        foreach(line IN LISTS _lines)
            if(line MATCHES "#include.*${FORBIDDEN}")
                message(FATAL_ERROR
                    "Gate B-1 include VIOLATION: ${f}: ${line}")
            endif()
        endforeach()
    endforeach()
endforeach()

list(LENGTH _all_files _total)
message(STATUS "B-1 include check passed (${_total} files scanned)")
