# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Tests: helpers, environment plumbing, execution and concurrency units. Included from the root CMakeLists.txt inside if(USDGEN_BUILD_TESTS).

enable_testing()

# Some harnesses are inherently Unix-only: the POSIX process/loader/ELF tests
# (execv/setenv/LD_LIBRARY_PATH, ELF symbol inspection) and the EGL/Storm GPU
# tests (dlopen + a headless EGL device) include unistd.h/dlfcn.h, so MSVC
# cannot compile them, and the bash-script checks need a POSIX shell. Define
# the executable only where those APIs exist; on other hosts register an honest
# ctest skip so the test-name and gate-label set stays identical across
# platforms instead of breaking the build. Portable targets keep calling
# add_test() directly.
function(usdgen_add_unix_only_test name)
    if(WIN32)
        add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E echo
            "SKIP ${name} (Unix-only harness: POSIX process/loader, shell, or EGL/Storm)")
        set_tests_properties(${name} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP")
    else()
        add_test(NAME ${name} COMMAND ${ARGN})
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Test environment plumbing (cross-platform)
#
# CTest's ENVIRONMENT property is the wrong tool for path lists, and on
# Windows it is actively wrong twice over:
#
#   * ENVIRONMENT *replaces* a variable. The host loader variable there is
#     PATH itself, so "PATH=<usd>/lib" strips the inherited system PATH from
#     the test process instead of extending it.
#   * ENVIRONMENT is a CMake list, i.e. ';'-separated, and ';' is also the
#     native path-list separator on Windows. A multi-directory value silently
#     splits into extra list entries, so the variable is truncated after its
#     first directory and the remaining directories become junk entries.
#
# ENVIRONMENT_MODIFICATION (CMake >= 3.22) avoids both: it edits the
# inherited value rather than replacing it, and path_list_* joins with the
# host's native separator, so each operation carries exactly one directory
# and no value ever embeds a ';'.
# ---------------------------------------------------------------------------

# Emit one ENVIRONMENT_MODIFICATION operation per directory.
function(usdgen_test_path_list_ops out_var name op)
    set(_ops "")
    foreach(_dir IN LISTS ARGN)
        list(APPEND _ops "${name}=${op}:${_dir}")
    endforeach()
    set(${out_var} "${_ops}" PARENT_SCOPE)
endfunction()

# PXR_PLUGINPATH_NAME is rebuilt from scratch ("set:" on the first directory)
# so a value already exported in the developer's shell cannot take precedence
# over the build-tree plugin resources. The rest append in declaration order,
# which is the order the plugInfo ordering contract (gate SI-5) asserts.
function(usdgen_test_pluginpath_ops out_var)
    set(_ops "")
    foreach(_dir IN LISTS ARGN)
        if(_ops STREQUAL "")
            list(APPEND _ops "PXR_PLUGINPATH_NAME=set:${_dir}")
        else()
            list(APPEND _ops "PXR_PLUGINPATH_NAME=path_list_append:${_dir}")
        endif()
    endforeach()
    set(${out_var} "${_ops}" PARENT_SCOPE)
endfunction()

# Directories a test binary must be able to load shared libraries from.
# Windows has no RPATH, so PATH is the only mechanism there; ELF/Mach-O builds
# already resolve the build tree through CMake's build RPATH, but naming the
# OpenUSD prefix is required on every host, so one list serves all three.
# OpenUSD's Windows install splits its runtime in two: the usd_*.dll core
# lands in lib/ while the bundled third-party DLLs (tbb, embree, ...) land in
# bin/, so both directories are needed there.
set(USDGEN_TEST_RUNTIME_DIRS "${USDGEN_TEST_USD_DIR}/lib")
if(WIN32)
    list(APPEND USDGEN_TEST_RUNTIME_DIRS "${USDGEN_TEST_USD_DIR}/bin")
endif()
list(APPEND USDGEN_TEST_RUNTIME_DIRS "$<TARGET_FILE_DIR:usdGen>")

add_executable(testUsdGenExecutionPipeline tests/testUsdGenExecutionPipeline.cpp)
target_link_libraries(testUsdGenExecutionPipeline PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenExecutionPipeline COMMAND testUsdGenExecutionPipeline)
set_tests_properties(testUsdGenExecutionPipeline PROPERTIES LABELS "T1;scheduling;sessions" TIMEOUT 30)
add_executable(testUsdGenCutFailure tests/testUsdGenCutFailure.cpp)
target_link_libraries(testUsdGenCutFailure PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenCutFailure COMMAND testUsdGenCutFailure)
set_tests_properties(testUsdGenCutFailure PROPERTIES LABELS "T0;scheduling;collision" TIMEOUT 30)
add_executable(testUsdGenExecutionSequenceWindow tests/testUsdGenExecutionSequenceWindow.cpp)
target_link_libraries(testUsdGenExecutionSequenceWindow PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenExecutionSequenceWindow COMMAND testUsdGenExecutionSequenceWindow)
set_tests_properties(testUsdGenExecutionSequenceWindow PROPERTIES LABELS "T0;scheduling;resources" TIMEOUT 30)
add_executable(testUsdGenSchedulerAffinity tests/testUsdGenSchedulerAffinity.cpp)
target_link_libraries(testUsdGenSchedulerAffinity PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenSchedulerAffinity COMMAND testUsdGenSchedulerAffinity)
set_tests_properties(testUsdGenSchedulerAffinity PROPERTIES
    LABELS "T0;scheduling;affinity" TIMEOUT 60
    SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
add_executable(testUsdGenPool tests/testUsdGenPool.cpp)
target_link_libraries(testUsdGenPool PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenPool COMMAND testUsdGenPool)
set_tests_properties(testUsdGenPool PROPERTIES LABELS "T0;pool" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenWorkerPool tests/testUsdGenWorkerPool.cpp)
target_link_libraries(testUsdGenWorkerPool PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenWorkerPool COMMAND testUsdGenWorkerPool)
set_tests_properties(testUsdGenWorkerPool PROPERTIES LABELS "T0;scheduling;pool" RUN_SERIAL ON TIMEOUT 60)
add_executable(testUsdGenElementSet tests/testUsdGenElementSet.cpp)
target_link_libraries(testUsdGenElementSet PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenElementSet COMMAND testUsdGenElementSet)
set_tests_properties(testUsdGenElementSet PROPERTIES LABELS "T0;elementSet" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenValueTypeTable tests/testUsdGenValueTypeTable.cpp)
target_link_libraries(testUsdGenValueTypeTable PRIVATE usdGenUsd sdf tf)
add_test(NAME testUsdGenValueTypeTable COMMAND testUsdGenValueTypeTable)
set_tests_properties(testUsdGenValueTypeTable PROPERTIES LABELS "T0;valueType" TIMEOUT 60)
# Step 11.4: usdGenMath against Gf 26.08, bit for bit (half conversions exhaustive).
add_executable(testUsdGenMathGfParity tests/usdGenUsd/testUsdGenMathGfParity.cpp)
target_link_libraries(testUsdGenMathGfParity PRIVATE usdGenMath gf tf)
add_test(NAME testUsdGenMathGfParity COMMAND testUsdGenMathGfParity)
set_tests_properties(testUsdGenMathGfParity PROPERTIES LABELS "T0;math" TIMEOUT 300)
add_executable(testUsdGenNodeKeyMap tests/testUsdGenNodeKeyMap.cpp)
target_link_libraries(testUsdGenNodeKeyMap PRIVATE usdGenUsd)
add_test(NAME testUsdGenNodeKeyMap COMMAND testUsdGenNodeKeyMap)
set_tests_properties(testUsdGenNodeKeyMap PROPERTIES LABELS "T0;nodeKey" TIMEOUT 60)
add_executable(testUsdGenCurveSet tests/testUsdGenCurveSet.cpp tests/testUsdGenCurveSetChunk.cpp)
target_link_libraries(testUsdGenCurveSet PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenCurveSet COMMAND testUsdGenCurveSet)
set_tests_properties(testUsdGenCurveSet PROPERTIES LABELS "T0;elementSet" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenRootSet tests/testUsdGenRootSet.cpp)
target_link_libraries(testUsdGenRootSet PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenRootSet COMMAND testUsdGenRootSet)
set_tests_properties(testUsdGenRootSet PROPERTIES LABELS "T0;elementSet" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenEpoch tests/testUsdGenEpoch.cpp)
target_link_libraries(testUsdGenEpoch PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenEpoch COMMAND testUsdGenEpoch)
set_tests_properties(testUsdGenEpoch PROPERTIES LABELS "T0;scheduling;epoch" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenLockFreeStress tests/testUsdGenLockFreeStress.cpp)
target_link_libraries(testUsdGenLockFreeStress PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenLockFreeStress COMMAND testUsdGenLockFreeStress)
set_tests_properties(testUsdGenLockFreeStress PROPERTIES LABELS "T1;scheduling;stress" RUN_SERIAL ON TIMEOUT 300)
add_executable(testUsdGenTaskScheduler tests/testUsdGenTaskScheduler.cpp)
target_link_libraries(testUsdGenTaskScheduler PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenTaskScheduler COMMAND testUsdGenTaskScheduler)
set_tests_properties(testUsdGenTaskScheduler PROPERTIES LABELS "T0;scheduling" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenVulkanEpochCell tests/testUsdGenVulkanEpochCell.cpp)
target_link_libraries(testUsdGenVulkanEpochCell PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenVulkanEpochCell COMMAND testUsdGenVulkanEpochCell)
set_tests_properties(testUsdGenVulkanEpochCell PROPERTIES LABELS "T0;scheduling;epoch;vulkan" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenImagingAtomicShared tests/testUsdGenImagingAtomicShared.cpp)
target_include_directories(testUsdGenImagingAtomicShared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenHydra)
target_link_libraries(testUsdGenImagingAtomicShared PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenImagingAtomicShared COMMAND testUsdGenImagingAtomicShared)
set_tests_properties(testUsdGenImagingAtomicShared PROPERTIES LABELS "T0;scheduling;epoch" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenCleanupQueue tests/testUsdGenCleanupQueue.cpp)
target_include_directories(testUsdGenCleanupQueue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenHydra)
target_link_libraries(testUsdGenCleanupQueue PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenCleanupQueue COMMAND testUsdGenCleanupQueue)
set_tests_properties(testUsdGenCleanupQueue PROPERTIES LABELS "T0;scheduling;stress" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenExecutionTaskGraph tests/testUsdGenExecutionTaskGraph.cpp)
target_link_libraries(testUsdGenExecutionTaskGraph PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenExecutionTaskGraph COMMAND testUsdGenExecutionTaskGraph)
set_tests_properties(testUsdGenExecutionTaskGraph PROPERTIES LABELS "T1;scheduling" TIMEOUT 30)
add_executable(testUsdGenExecutionCache tests/testUsdGenExecutionCache.cpp)
target_link_libraries(testUsdGenExecutionCache PRIVATE usdGenWithPrimitives usdGenUsd)
target_link_libraries(testUsdGenExecutionCache PRIVATE usdGenTestUtils usdGenHydraCurves)
add_test(NAME testUsdGenExecutionCache COMMAND testUsdGenExecutionCache)
set_tests_properties(testUsdGenExecutionCache PROPERTIES LABELS "T1;scheduling" TIMEOUT 30)
add_executable(testUsdGenExecutionBackend tests/testUsdGenExecutionBackend.cpp)
target_link_libraries(testUsdGenExecutionBackend PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
if(USDGEN_ENABLE_CUDA)
    target_compile_definitions(testUsdGenExecutionBackend PRIVATE USDGEN_ENABLE_CUDA)
endif()
if(USDGEN_ENABLE_VULKAN_RUNTIME)
    target_compile_definitions(testUsdGenExecutionBackend PRIVATE USDGEN_ENABLE_VULKAN_RUNTIME)
endif()
add_test(NAME testUsdGenExecutionBackend COMMAND testUsdGenExecutionBackend)
set_tests_properties(testUsdGenExecutionBackend PROPERTIES LABELS "T0;backend" TIMEOUT 30)
add_executable(testUsdGenExecutionBackendNeutral
    tests/testUsdGenExecutionBackendNeutral.cpp)
target_link_libraries(testUsdGenExecutionBackendNeutral PRIVATE usdGenWithPrimitives usdGenUsd)
if(USDGEN_ENABLE_VULKAN_RUNTIME)
    target_compile_definitions(testUsdGenExecutionBackendNeutral PRIVATE USDGEN_ENABLE_VULKAN_RUNTIME)
endif()
add_test(NAME testUsdGenExecutionBackendNeutral
    COMMAND testUsdGenExecutionBackendNeutral)
set_tests_properties(testUsdGenExecutionBackendNeutral
    PROPERTIES LABELS "T0;backend" TIMEOUT 30)
