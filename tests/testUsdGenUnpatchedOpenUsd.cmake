# Guard the shipped source tree against rediscovering the retired private
# OpenUSD/Storm experiment. Historical plans and the patch README are outside
# this active-source scan by design.
if(NOT DEFINED USDGEN_SOURCE_DIR)
    message(FATAL_ERROR "USDGEN_SOURCE_DIR is required")
endif()

file(GLOB_RECURSE _patches "${USDGEN_SOURCE_DIR}/patches/openusd/*.patch")
if(_patches)
    message(FATAL_ERROR "Retired OpenUSD patches are present: ${_patches}")
endif()

set(_active_files
    "${USDGEN_SOURCE_DIR}/CMakeLists.txt"
    "${USDGEN_SOURCE_DIR}/plugin/usdGenImaging/resources/plugInfo.json.in")
file(GLOB_RECURSE _imaging_sources
    "${USDGEN_SOURCE_DIR}/libs/hosts/usdGenHydra/usdGenImaging/*.cpp"
    "${USDGEN_SOURCE_DIR}/libs/hosts/usdGenHydra/usdGenImaging/*.h")
list(APPEND _active_files ${_imaging_sources})

string(CONCAT _retired_pattern
    "USDGEN_TEST_STORM_POST_COMMIT|USDGEN_PATCHED_OPENUSD_SOURCE|"
    "usdGenPatchedStorm|usdGenPatchedHd|usdGenPrivateImaging|"
    "hdStBasisCurvesGpu|renderDelegateSceneIndexObserver|"
    "BasisCurvesGpuGroup|gpuCurveGroupStaging")
foreach(_file IN LISTS _active_files)
    if(EXISTS "${_file}")
        file(READ "${_file}" _contents)
        if(_contents MATCHES "${_retired_pattern}")
            message(FATAL_ERROR
                "Retired private OpenUSD/Storm hook in active source: ${_file}")
        endif()
    endif()
endforeach()
