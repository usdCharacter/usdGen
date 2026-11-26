# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# CheckLinkAllow.cmake: a target's direct links against an allow or deny
# regex (link gates 8d). Fails on any link not matching ALLOW or matching DENY.
#
# Inputs:
#   NAME    gate name
#   TARGET  target name (for messages)
#   LINKS   the target's LINK_LIBRARIES (generator-expanded, ;-list)
#   ALLOW   optional regex every link must match
#   DENY    optional regex no link may match

foreach(_v NAME TARGET)
    if(NOT DEFINED ${_v} OR "${${_v}}" STREQUAL "")
        message(FATAL_ERROR "CheckLinkAllow: ${_v} is required")
    endif()
endforeach()
if((NOT DEFINED ALLOW OR ALLOW STREQUAL "") AND (NOT DEFINED DENY OR DENY STREQUAL ""))
    message(FATAL_ERROR "CheckLinkAllow: ALLOW or DENY is required")
endif()
# ctest passes the joined list with escaped separators.
string(REPLACE "\\;" ";" LINKS "${LINKS}")
set(_fail 0)
set(_n 0)
foreach(_lib IN LISTS LINKS)
    if(_lib STREQUAL "")
        continue()
    endif()
    math(EXPR _n "${_n} + 1")
    if(DEFINED ALLOW AND NOT ALLOW STREQUAL "" AND NOT _lib MATCHES "${ALLOW}")
        set(_fail 1)
        message(SEND_ERROR "${NAME}: ${TARGET} links ${_lib}, which ALLOW does not match")
    endif()
    if(DEFINED DENY AND NOT DENY STREQUAL "" AND _lib MATCHES "${DENY}")
        set(_fail 1)
        message(SEND_ERROR "${NAME}: ${TARGET} links ${_lib}, which DENY matches")
    endif()
endforeach()
if(_fail)
    message(FATAL_ERROR "${NAME}: FAILED")
endif()
message(STATUS "${NAME}: ${TARGET} links ${_n} libraries (${LINKS}); all allowed")
