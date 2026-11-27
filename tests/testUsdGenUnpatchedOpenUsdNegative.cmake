# Exercise every retired-token alternative against isolated disposable
# fixtures. The guard must fail for the expected reason rather than merely
# return an unrelated nonzero status.
if(NOT DEFINED USDGEN_SOURCE_DIR)
    message(FATAL_ERROR "USDGEN_SOURCE_DIR is required")
endif()

set(_guard "${USDGEN_SOURCE_DIR}/tests/testUsdGenUnpatchedOpenUsd.cmake")
string(TIMESTAMP _timestamp "%s")
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef _nonce)
set(_fixture_root "/tmp/usdgen-unpatched-openusd-${_timestamp}-${_nonce}")

function(_write_empty_fixture root)
    file(MAKE_DIRECTORY "${root}/patches/openusd"
        "${root}/plugin/usdGenImaging/resources"
        "${root}/libs/hosts/usdGenHydra/usdGenImaging")
    file(WRITE "${root}/CMakeLists.txt" "")
    file(WRITE "${root}/plugin/usdGenImaging/resources/plugInfo.json.in" "")
    file(WRITE "${root}/libs/hosts/usdGenHydra/usdGenImaging/probe.cpp" "")
endfunction()

function(_run_guard root result output)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -DUSDGEN_SOURCE_DIR=${root} -P "${_guard}"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr)
    set(${result} "${_result}" PARENT_SCOPE)
    set(${output} "${_stdout}${_stderr}" PARENT_SCOPE)
endfunction()

set(_positive_probe "${_fixture_root}/positive")
_write_empty_fixture("${_positive_probe}")
_run_guard("${_positive_probe}" _positive_result _positive_output)
if(NOT _positive_result EQUAL 0)
    message(FATAL_ERROR
        "Guard rejected an empty fixture: ${_positive_output}")
endif()

set(_tokens
    USDGEN_TEST_STORM_POST_COMMIT
    USDGEN_PATCHED_OPENUSD_SOURCE
    usdGenPatchedStorm
    usdGenPatchedHd
    usdGenPrivateImaging
    hdStBasisCurvesGpu
    renderDelegateSceneIndexObserver
    BasisCurvesGpuGroup
    gpuCurveGroupStaging)

foreach(_token IN LISTS _tokens)
    set(_probe "${_fixture_root}/${_token}")
    _write_empty_fixture("${_probe}")
    file(WRITE "${_probe}/CMakeLists.txt" "${_token}\n")
    _run_guard("${_probe}" _result _output)
    if(_result EQUAL 0)
        message(FATAL_ERROR
            "Guard accepted retired token ${_token}: ${_output}")
    endif()
    string(FIND "${_output}"
        "Retired private OpenUSD/Storm hook in active source:" _reason)
    string(FIND "${_output}" "${_probe}/CMakeLists.txt" _path)
    if(_reason EQUAL -1 OR _path EQUAL -1)
        message(FATAL_ERROR
            "Guard rejected ${_token} for the wrong reason: ${_output}")
    endif()
endforeach()

# Recursive patch discovery is a separate admission condition.
set(_patch_probe "${_fixture_root}/nested-patch")
_write_empty_fixture("${_patch_probe}")
file(MAKE_DIRECTORY "${_patch_probe}/patches/openusd/nested")
file(WRITE "${_patch_probe}/patches/openusd/nested/retired.patch" "test\n")
_run_guard("${_patch_probe}" _patch_result _patch_output)
if(_patch_result EQUAL 0)
    message(FATAL_ERROR
        "Guard accepted nested patch: ${_patch_output}")
endif()
string(FIND "${_patch_output}" "Retired OpenUSD patches are present:" _patch_reason)
string(FIND "${_patch_output}"
    "${_patch_probe}/patches/openusd/nested/retired.patch" _patch_path)
if(_patch_reason EQUAL -1 OR _patch_path EQUAL -1)
    message(FATAL_ERROR
        "Guard rejected nested patch for the wrong reason: ${_patch_output}")
endif()
