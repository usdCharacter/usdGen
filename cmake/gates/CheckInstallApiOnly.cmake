# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# testUsdGenInstallOnlyApi (12.5, D16): install the build tree into PREFIX and
# fail on any installed header outside include/usdGen/api/ and the usdGenMath
# headers api/ includes, on an export set other than usdGen::usdGen, on any
# link interface on usdGen::usdGen (no pxr), and on any usdGen archive or
# import library except usdGen's own.
#
#   cmake -DBUILD_DIR=<build> [-DCONFIG=<cfg>] -DPREFIX=<scratch prefix>
#         -P cmake/gates/CheckInstallApiOnly.cmake
cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0057 NEW)  # IN_LIST

foreach(_var BUILD_DIR PREFIX)
    if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
        message(FATAL_ERROR "CheckInstallApiOnly: ${_var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${PREFIX}")
set(_args --install "${BUILD_DIR}" --prefix "${PREFIX}")
if(DEFINED CONFIG AND NOT CONFIG STREQUAL "")
    list(APPEND _args --config "${CONFIG}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" ${_args}
    RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "CheckInstallApiOnly: cmake --install failed (${_rc}):\n${_out}")
endif()

set(_failures "")
set(_math "(half|vec|matrix|quat|range|math|rbf)")

# Headers: only api/ and the math headers.
file(GLOB_RECURSE _headers RELATIVE "${PREFIX}" "${PREFIX}/include/*")
set(_api_count 0)
foreach(_f IN LISTS _headers)
    if(_f MATCHES "^include/usdGen/api/.+[.]h$")
        math(EXPR _api_count "${_api_count} + 1")
    elseif(NOT _f MATCHES "^include/usdGenMath/${_math}[.]h$")
        list(APPEND _failures "installed outside api/: ${_f}")
    endif()
endforeach()
foreach(_need include/usdGen/api/usdGen.h include/usdGen/api/export.h
        include/usdGen/api/engine.h include/usdGen/api/session.h
        include/usdGen/api/ext/registrar.h include/usdGenMath/vec.h)
    if(NOT EXISTS "${PREFIX}/${_need}")
        list(APPEND _failures "missing: ${_need}")
    endif()
endforeach()

# Export set: usdGen::usdGen only, with no link interface.
file(GLOB_RECURSE _targets_files "${PREFIX}/*/usdGenTargets.cmake")
if(NOT _targets_files)
    list(APPEND _failures "missing: usdGenTargets.cmake")
endif()
foreach(_tf IN LISTS _targets_files)
    file(STRINGS "${_tf}" _adds REGEX "add_library\\(")
    foreach(_add IN LISTS _adds)
        if(NOT _add MATCHES "add_library\\(usdGen::usdGen SHARED IMPORTED\\)")
            list(APPEND _failures "exported target other than usdGen::usdGen: ${_add}")
        endif()
    endforeach()
    file(STRINGS "${_tf}" _links REGEX "^[ ]+INTERFACE_LINK_LIBRARIES[ ]+\"|pxr")
    foreach(_l IN LISTS _links)
        list(APPEND _failures "usdGen::usdGen link interface: ${_l}")
    endforeach()
endforeach()

# Archives and import libraries: usdGen's import library only.
file(GLOB_RECURSE _archives RELATIVE "${PREFIX}" "${PREFIX}/*usdGen*.lib" "${PREFIX}/*usdGen*.a")
foreach(_a IN LISTS _archives)
    get_filename_component(_name "${_a}" NAME)
    if(NOT _name STREQUAL "usdGen.lib")
        list(APPEND _failures "archive or import library installed: ${_a}")
    endif()
endforeach()

if(_failures)
    list(JOIN _failures "\n  " _msg)
    message(FATAL_ERROR "testUsdGenInstallOnlyApi: FAIL\n  ${_msg}")
endif()
message(STATUS "testUsdGenInstallOnlyApi: PASS (${_api_count} api headers; usdGen::usdGen only)")