# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
# One forwarding header per moved header: <build>/shim/<old> -> <new>.
# Retired by step 12; testUsdGenNoShimIncludes ratchets old spellings down.
set(USDGEN_SHIM_DIR "${CMAKE_BINARY_DIR}/shim")
set(USDGEN_SHIM_OLD_SPELLINGS)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/includeShim.txt" _usdgen_shim REGEX "^[^#]")
foreach(_line IN LISTS _usdgen_shim)
    string(STRIP "${_line}" _line)
    if(NOT _line MATCHES "^([^ ]+) ([^ ]+)$")
        message(FATAL_ERROR "includeShim.txt: bad line '${_line}'")
    endif()
    list(APPEND USDGEN_SHIM_OLD_SPELLINGS "${CMAKE_MATCH_1}")
    file(CONFIGURE OUTPUT "${USDGEN_SHIM_DIR}/${CMAKE_MATCH_1}"
         CONTENT "#pragma once\n#include \"${CMAKE_MATCH_2}\"\n" @ONLY)
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/includeShim.txt")
