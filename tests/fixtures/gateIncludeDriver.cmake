# gateIncludeDriver.cmake
# Test driver for the B-1 include gate (cmake/CheckNoStageInclude.cmake).
# Runs the gate against a fixture tree and verifies the gate's verdict
# matches the expectation, so the gate is proven to fail on forbidden
# input and pass on clean input (no silent false passes either way).
#
# Inputs (via -D):
#   FIXTURE_DIR - root of the fixture tree (must mirror libs/base/usdGenMath,
#                 libs/core, testutils/usdGenTestUtils)
#   FORBIDDEN   - the same forbidden-include regex the real gate uses
#   EXPECT      - "fail" (gate must FATAL_ERROR) or "pass" (gate must pass)

if(NOT DEFINED FIXTURE_DIR OR NOT DEFINED FORBIDDEN OR NOT DEFINED EXPECT)
    message(FATAL_ERROR "gateIncludeDriver: FIXTURE_DIR, FORBIDDEN and EXPECT are required")
endif()
if(NOT EXPECT MATCHES "^(pass|fail)$")
    message(FATAL_ERROR "gateIncludeDriver: EXPECT must be 'pass' or 'fail', got '${EXPECT}'")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${FIXTURE_DIR}
        -DFORBIDDEN=${FORBIDDEN}
        -P ${CMAKE_CURRENT_LIST_DIR}/../../cmake/CheckNoStageInclude.cmake
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err)

if(EXPECT STREQUAL "fail")
    if(_rc EQUAL 0)
        message(FATAL_ERROR
            "gateIncludeDriver: B-1 include gate PASSED on forbidden fixture ${FIXTURE_DIR} "
            "(expected a hard failure):\n${_out}")
    endif()
    # A configuration error (missing dir, etc.) is not proof the gate works:
    # the failure must be a real include VIOLATION on the fixture content.
    if(NOT _err MATCHES "Gate B-1 include VIOLATION")
        message(FATAL_ERROR
            "gateIncludeDriver: gate did not fail with a real include VIOLATION; "
            "wrong reason or misconfiguration:\n${_out}\n${_err}")
    endif()
    message(STATUS "ok: B-1 include gate correctly FAILED on forbidden fixture (${FIXTURE_DIR})")
else()
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR
            "gateIncludeDriver: B-1 include gate FAILED on clean fixture ${FIXTURE_DIR} "
            "(expected a pass):\n${_err}")
    endif()
    message(STATUS "ok: B-1 include gate correctly passed on clean fixture (${FIXTURE_DIR})")
endif()
