# CheckLinkRatchet.cmake: the link half of the no-pxr-below-hosts gate.
# A below-host target may link only what its baseline lists; the baseline
# may only shrink. The target is no pxr library at all.
#
# Inputs:
#   NAME      gate name
#   TARGET    target name (for messages)
#   LINKS     the target's LINK_LIBRARIES (generator-expanded, ;-list)
#   PATTERN   regex for a pxr library name
#   BASELINE  file listing allowed pxr libraries, one per line, # comments

foreach(_v NAME TARGET PATTERN BASELINE)
    if(NOT DEFINED ${_v} OR "${${_v}}" STREQUAL "")
        message(FATAL_ERROR "CheckLinkRatchet: ${_v} is required")
    endif()
endforeach()
# ctest passes the joined list with escaped separators.
string(REPLACE "\\;" ";" LINKS "${LINKS}")
set(_allowed "")
if(EXISTS "${BASELINE}")
    file(STRINGS "${BASELINE}" _allowed REGEX "^[^#]")
endif()
set(_fail 0)
set(_found "")
foreach(_lib IN LISTS LINKS)
    # usdGen* are this project's libraries; PATTERN's usd[A-Za-z]* would match them.
    if(_lib MATCHES "^usdGen")
        continue()
    endif()
    if(_lib MATCHES "${PATTERN}")
        list(APPEND _found "${_lib}")
        if(NOT _lib IN_LIST _allowed)
            set(_fail 1)
            message(SEND_ERROR "${NAME}: ${TARGET} links ${_lib}, which its baseline does not allow")
        endif()
    endif()
endforeach()
list(REMOVE_DUPLICATES _found)
list(LENGTH _found _n)
if(_fail)
    message(FATAL_ERROR "${NAME}: FAILED")
endif()
message(STATUS "${NAME}: ${TARGET} links ${_n} pxr libraries (${_found}); target 0")
