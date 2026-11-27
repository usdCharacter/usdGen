# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Tests: helpers, environment plumbing, execution and concurrency units. Included from the root CMakeLists.txt inside if(USDGEN_BUILD_TESTS).

enable_testing()

# Test sources live in tier folders: tests/internal/<lib> (drives private
# classes; only built with USDGEN_EXPORT_INTERNALS), tests/hosts/<lib> (host
# API) and tests/api. tests/ is on every test's include path, so shared
# fixtures are spelled relative to it.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tests)
file(GLOB _usdgen_test_tier_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/internal/* ${CMAKE_CURRENT_SOURCE_DIR}/tests/hosts/*)
list(APPEND _usdgen_test_tier_dirs tests/api)
# add_test records each test's executable (CTest's COMMAND is not readable at
# configure time); the tail of the root CMakeLists.txt labels tests/internal
# ones from it. Arguments are re-quoted as bracket arguments, so ';', empty
# and generator-expression arguments reach add_test unchanged.
function(add_test)
    set(_code "_add_test(")
    math(EXPR _last "${ARGC} - 1")
    foreach(_i RANGE ${_last})
        string(APPEND _code " [==[${ARGV${_i}}]==]")
        if(ARGV${_i} STREQUAL "COMMAND" AND ARGV0 STREQUAL "NAME")
            math(EXPR _n "${_i} + 1")
            set_property(GLOBAL PROPERTY USDGEN_TEST_EXE_${ARGV1} "${ARGV${_n}}")
        endif()
    endforeach()
    string(APPEND _code ")")
    cmake_language(EVAL CODE "${_code}")
endfunction()
# Name-driven loops resolve a file name to its tier folder. QUIET: empty out if none.
function(usdgen_test_source out file_name)
    set(${out} "" PARENT_SCOPE)
    foreach(_dir IN LISTS _usdgen_test_tier_dirs)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/${file_name}")
            set(${out} "${_dir}/${file_name}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    if(NOT "${ARGN}" STREQUAL "QUIET")
        message(FATAL_ERROR "usdGen tests: no source named ${file_name} under tests/{internal,hosts,api}")
    endif()
endfunction()

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

add_executable(testUsdGenExecutionPipeline tests/api/testUsdGenExecutionPipeline.cpp)
target_link_libraries(testUsdGenExecutionPipeline PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenExecutionPipeline COMMAND testUsdGenExecutionPipeline)
set_tests_properties(testUsdGenExecutionPipeline PROPERTIES LABELS "T1;scheduling;sessions" TIMEOUT 30)
add_executable(testUsdGenCutFailure tests/internal/usdGen/testUsdGenCutFailure.cpp)
target_link_libraries(testUsdGenCutFailure PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenCutFailure COMMAND testUsdGenCutFailure)
set_tests_properties(testUsdGenCutFailure PROPERTIES LABELS "T0;scheduling;collision" TIMEOUT 30)
add_executable(testUsdGenExecutionSequenceWindow tests/api/testUsdGenExecutionSequenceWindow.cpp)
target_link_libraries(testUsdGenExecutionSequenceWindow PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenExecutionSequenceWindow COMMAND testUsdGenExecutionSequenceWindow)
set_tests_properties(testUsdGenExecutionSequenceWindow PROPERTIES LABELS "T0;scheduling;resources" TIMEOUT 30)
add_executable(testUsdGenSchedulerAffinity tests/internal/usdGen/testUsdGenSchedulerAffinity.cpp)
target_link_libraries(testUsdGenSchedulerAffinity PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenSchedulerAffinity COMMAND testUsdGenSchedulerAffinity)
set_tests_properties(testUsdGenSchedulerAffinity PROPERTIES
    LABELS "T0;scheduling;affinity" TIMEOUT 60
    SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
add_executable(testUsdGenPool tests/internal/usdGen/testUsdGenPool.cpp)
target_link_libraries(testUsdGenPool PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenPool COMMAND testUsdGenPool)
set_tests_properties(testUsdGenPool PROPERTIES LABELS "T0;pool" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenWorkerPool tests/internal/usdGen/testUsdGenWorkerPool.cpp)
target_link_libraries(testUsdGenWorkerPool PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenWorkerPool COMMAND testUsdGenWorkerPool)
set_tests_properties(testUsdGenWorkerPool PROPERTIES LABELS "T0;scheduling;pool" RUN_SERIAL ON TIMEOUT 60)
add_executable(testUsdGenElementSet tests/internal/usdGen/testUsdGenElementSet.cpp)
target_link_libraries(testUsdGenElementSet PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenElementSet COMMAND testUsdGenElementSet)
set_tests_properties(testUsdGenElementSet PROPERTIES LABELS "T0;elementSet" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenValueTypeTable tests/internal/usdGenExpr/testUsdGenValueTypeTable.cpp)
target_link_libraries(testUsdGenValueTypeTable PRIVATE usdGenUsd sdf tf)
add_test(NAME testUsdGenValueTypeTable COMMAND testUsdGenValueTypeTable)
set_tests_properties(testUsdGenValueTypeTable PROPERTIES LABELS "T0;valueType" TIMEOUT 60)
# Step 11.4: usdGenMath against Gf 26.08, bit for bit (half conversions exhaustive).
add_executable(testUsdGenMathGfParity tests/hosts/usdGenUsd/testUsdGenMathGfParity.cpp)
target_link_libraries(testUsdGenMathGfParity PRIVATE usdGenMath gf tf)
add_test(NAME testUsdGenMathGfParity COMMAND testUsdGenMathGfParity)
set_tests_properties(testUsdGenMathGfParity PROPERTIES LABELS "T0;math" TIMEOUT 300)

add_executable(testUsdGenToken tests/hosts/usdGenUsd/testUsdGenToken.cpp)
target_link_libraries(testUsdGenToken PRIVATE usdGen usdGenUsd tf)
add_test(NAME testUsdGenToken COMMAND testUsdGenToken)
set_tests_properties(testUsdGenToken PROPERTIES LABELS "T0;registries" TIMEOUT 300)
# Step 11.6: CowArray detach parity with VtArray, external arenas, zero-copy host conversions.
add_executable(testUsdGenCowArray tests/hosts/usdGenUsd/testUsdGenCowArray.cpp)
target_link_libraries(testUsdGenCowArray PRIVATE usdGen usdGenUsd tf vt gf)
add_test(NAME testUsdGenCowArray COMMAND testUsdGenCowArray)
set_tests_properties(testUsdGenCowArray PROPERTIES LABELS "T0;elementSet" TIMEOUT 60)
# Step 11.8: env parsing as TfGetenv, TF_DEBUG as TfDebug, trace and report sinks.
add_executable(testUsdGenDiagnostics tests/hosts/usdGenUsd/testUsdGenDiagnostics.cpp)
target_link_libraries(testUsdGenDiagnostics PRIVATE usdGen usdGenUsd tf)
add_test(NAME testUsdGenDiagnostics COMMAND testUsdGenDiagnostics)
set_tests_properties(testUsdGenDiagnostics PROPERTIES LABELS "T0;registries" TIMEOUT 60
    ENVIRONMENT "TF_DEBUG=USDGEN_C* USDGEN_SCH* -USDGEN_SCHEDULE USDGEN_FUR")
# Step 11.7: UsdGenValue against VtValue: every value-type row round trips, exact types, hash laws.
add_executable(testUsdGenValue tests/internal/usdGenExpr/testUsdGenValue.cpp)
target_link_libraries(testUsdGenValue PRIVATE usdGen usdGenUsd sdf tf vt gf)
add_test(NAME testUsdGenValue COMMAND testUsdGenValue)
set_tests_properties(testUsdGenValue PROPERTIES LABELS "T0;valueType" TIMEOUT 60)
add_executable(testUsdGenNodeKeyMap tests/hosts/usdGenUsd/testUsdGenNodeKeyMap.cpp)
target_link_libraries(testUsdGenNodeKeyMap PRIVATE usdGenUsd)
add_test(NAME testUsdGenNodeKeyMap COMMAND testUsdGenNodeKeyMap)
set_tests_properties(testUsdGenNodeKeyMap PROPERTIES LABELS "T0;nodeKey" TIMEOUT 60)
add_executable(testUsdGenCurveSet tests/internal/usdGen/testUsdGenCurveSet.cpp tests/internal/usdGenCurves/testUsdGenCurveSetChunk.cpp)
target_link_libraries(testUsdGenCurveSet PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenCurveSet COMMAND testUsdGenCurveSet)
set_tests_properties(testUsdGenCurveSet PROPERTIES LABELS "T0;elementSet" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenRootSet tests/internal/usdGen/testUsdGenRootSet.cpp)
target_link_libraries(testUsdGenRootSet PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenRootSet COMMAND testUsdGenRootSet)
set_tests_properties(testUsdGenRootSet PROPERTIES LABELS "T0;elementSet" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenEpoch tests/api/testUsdGenEpoch.cpp)
target_link_libraries(testUsdGenEpoch PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenEpoch COMMAND testUsdGenEpoch)
set_tests_properties(testUsdGenEpoch PROPERTIES LABELS "T0;scheduling;epoch" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenLockFreeStress tests/internal/usdGen/testUsdGenLockFreeStress.cpp)
target_link_libraries(testUsdGenLockFreeStress PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenLockFreeStress COMMAND testUsdGenLockFreeStress)
set_tests_properties(testUsdGenLockFreeStress PROPERTIES LABELS "T1;scheduling;stress" RUN_SERIAL ON TIMEOUT 300)
add_executable(testUsdGenTaskScheduler tests/api/testUsdGenTaskScheduler.cpp)
target_link_libraries(testUsdGenTaskScheduler PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenTaskScheduler COMMAND testUsdGenTaskScheduler)
set_tests_properties(testUsdGenTaskScheduler PROPERTIES LABELS "T0;scheduling" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenVulkanEpochCell tests/api/testUsdGenVulkanEpochCell.cpp)
target_link_libraries(testUsdGenVulkanEpochCell PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenVulkanEpochCell COMMAND testUsdGenVulkanEpochCell)
set_tests_properties(testUsdGenVulkanEpochCell PROPERTIES LABELS "T0;scheduling;epoch;vulkan" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenImagingAtomicShared tests/api/testUsdGenImagingAtomicShared.cpp)
target_include_directories(testUsdGenImagingAtomicShared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenHydra)
target_link_libraries(testUsdGenImagingAtomicShared PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenImagingAtomicShared COMMAND testUsdGenImagingAtomicShared)
set_tests_properties(testUsdGenImagingAtomicShared PROPERTIES LABELS "T0;scheduling;epoch" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenCleanupQueue tests/hosts/usdGenHydra/testUsdGenCleanupQueue.cpp)
target_include_directories(testUsdGenCleanupQueue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenHydra)
target_link_libraries(testUsdGenCleanupQueue PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenCleanupQueue COMMAND testUsdGenCleanupQueue)
set_tests_properties(testUsdGenCleanupQueue PROPERTIES LABELS "T0;scheduling;stress" RUN_SERIAL ON TIMEOUT 120)
add_executable(testUsdGenExecutionTaskGraph tests/hosts/usdGenUsd/testUsdGenExecutionTaskGraph.cpp)
target_link_libraries(testUsdGenExecutionTaskGraph PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenExecutionTaskGraph COMMAND testUsdGenExecutionTaskGraph)
set_tests_properties(testUsdGenExecutionTaskGraph PROPERTIES LABELS "T1;scheduling" TIMEOUT 30)
add_executable(testUsdGenExecutionCache tests/internal/usdGen/testUsdGenExecutionCache.cpp)
target_link_libraries(testUsdGenExecutionCache PRIVATE usdGenWithPrimitives usdGenUsd)
target_link_libraries(testUsdGenExecutionCache PRIVATE usdGenTestUtils usdGenHydraCurves)
add_test(NAME testUsdGenExecutionCache COMMAND testUsdGenExecutionCache)
set_tests_properties(testUsdGenExecutionCache PROPERTIES LABELS "T1;scheduling" TIMEOUT 30)
add_executable(testUsdGenExecutionBackend tests/internal/usdGen/testUsdGenExecutionBackend.cpp)
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
    tests/internal/usdGen/testUsdGenExecutionBackendNeutral.cpp)
target_link_libraries(testUsdGenExecutionBackendNeutral PRIVATE usdGenWithPrimitives usdGenUsd)
if(USDGEN_ENABLE_VULKAN_RUNTIME)
    target_compile_definitions(testUsdGenExecutionBackendNeutral PRIVATE USDGEN_ENABLE_VULKAN_RUNTIME)
endif()
add_test(NAME testUsdGenExecutionBackendNeutral
    COMMAND testUsdGenExecutionBackendNeutral)
set_tests_properties(testUsdGenExecutionBackendNeutral
    PROPERTIES LABELS "T0;backend" TIMEOUT 30)
