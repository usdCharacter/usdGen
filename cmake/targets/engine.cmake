# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# usdGen, the engine core. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# usdGen (SHARED — the engine core, no usd/usdImaging/hd linkage)
# ---------------------------------------------------------------------------
# Step 6.4: the primitives are their own DLLs (below); usdGen compiles the
# engine and, until step 8, the backends.
file(GLOB_RECURSE USDGEN_CORE_SOURCES libs/core/usdGen/*.cpp libs/core/backends/*.cpp)
# Native Vulkan owners are opt-in and must not leak SDK headers/runtime
# dependencies into CUDA-only or CPU-only core builds.
list(FILTER USDGEN_CORE_SOURCES EXCLUDE REGEX "/usdGenVulkan/vulkan/.*\\.cpp$")
# Include roots plus the old-spelling shim (cmake/IncludeShim.cmake).
include(cmake/IncludeShim.cmake)
set(USDGEN_ENGINE_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/usdGen>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenVulkan>
    $<BUILD_INTERFACE:${USDGEN_SHIM_DIR}>)
# Primitive roots: each primitive target publishes its own. usdGen keeps
# them public until step 8: the backends compiled into usdGen read primitive
# headers (the engine itself includes none since 6.5; gate
# testUsdGenNoPrimitiveInEngine).
set(USDGEN_PRIMITIVE_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenCurves>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenSurface>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenExpr>)
set(USDGEN_CORE_INCLUDE_DIRS ${USDGEN_ENGINE_INCLUDE_DIRS} ${USDGEN_PRIMITIVE_INCLUDE_DIRS})
# exprEval.{h,cpp} (a stale SeExpr2-engine evaluator design) and its unwired
# tests/testUsdGenExpr.cpp are deleted: the IR interpreter in
# expressions/irExec.h is the design, shared by the CUDA kernel and by
# expressions/cpuEvaluator.cpp.
list(REMOVE_ITEM USDGEN_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/usdGen/usdGen/exec/device/executionResources.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/usdGen/usdGen/exec/device/executionRetirement.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/executionBackendCudaProvider.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenVulkan/usdGenVulkan/executionBackendVulkanProvider.cpp"
    # CUDA-only helper exposes CUDA vector types; do not compile it in the
    # CPU-only core build selected by ordinary consumers.
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/cudaScatterInput.cpp"
    # CUDA completion lane and stream fence include cuda_runtime.h; the CUDA
    # block below adds them to usdGen and usdGenGpu.
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/gpu/cudaCompletionLane.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/gpu/cudaFence.cpp")
add_library(usdGenExecutionResources SHARED
    libs/core/usdGen/usdGen/exec/device/executionResources.cpp
    libs/core/usdGen/usdGen/exec/device/executionRetirement.cpp
    # Private copy of the parking primitive for the retirement drain.
    libs/core/usdGen/usdGen/exec/parkingWord.cpp)
target_include_directories(usdGenExecutionResources PUBLIC
    ${USDGEN_ENGINE_INCLUDE_DIRS}
    $<INSTALL_INTERFACE:include>)
target_link_libraries(usdGenExecutionResources PRIVATE Threads::Threads)
set_target_properties(usdGenExecutionResources PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
    target_link_options(usdGenExecutionResources PRIVATE "LINKER:--exclude-libs,ALL")
endif()
add_library(usdGen SHARED ${USDGEN_CORE_SOURCES})
target_sources(usdGen PRIVATE
    libs/core/backends/usdGenCuda/usdGenCuda/executionBackendCudaProvider.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/executionBackendVulkanProvider.cpp)
# The host suffix port (deformEvaluateCpu.h, inlined into
# deformPipeline.cpp) needs sqrt vectorized: with math-errno the vectorizer
# keeps the explicit-fma loop's sqrt scalar and the suffix runs ~2x slower.
# Values are unchanged (errno only); nothing in the library reads errno.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/deformPipeline.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-math-errno")
endif()
target_include_directories(usdGen PUBLIC
    ${USDGEN_CORE_INCLUDE_DIRS}
    $<INSTALL_INTERFACE:include>)
# Public headers expose TfToken/GfVec3f/VtArray, so those base libs must be
# PUBLIC for consumers (tests) to link them transitively. No sdf (step 7.4):
# SdfPath and Sdf value types live in the hosts (usdGenUsd links sdf).
# usdGenMath is PRIVATE (N-7, plan §1.2): it is not installed, so it must
# not sit in usdGen's export interface (install(EXPORT) would then reference
# a non-installed target); usdGen's own sources still see its headers
# through the link, and no public header includes usdGenMath/...
target_link_libraries(usdGen
    PUBLIC tf gf vt usdGenExecutionResources
    PRIVATE usdGenMath arch js plug ts ar work trace hio pxOsd
            usdGen_seexpr usdGenExprFrontend ${CMAKE_DL_LIBS})
if(WIN32)
    target_link_libraries(usdGen PRIVATE Synchronization)  # WaitOnAddress
endif()
# Vendored Ptex (usdGenSurface/maps/ptexMap.cpp, engine-resident until 6.5);
# private like the other archives.
target_link_libraries(usdGen PRIVATE usdGen_ptex)
# usdGen is a SHARED plugin-host library: its public API (UsdGenSession,
# graph/tile types, ...) is consumed across the shared boundary by
# libusdGenImaging.so, so it must keep default visibility. Third-party code
# stays hidden: usdGenMath and usdGen_seexpr are compiled with
# CXX_VISIBILITY_PRESET hidden, and --exclude-libs,ALL keeps their symbols
# out of libusdGen.so's export table (gate B-1 NoThirdPartyExports).
set_target_properties(usdGen PROPERTIES
    VISIBILITY_INLINES_HIDDEN ON
    # The Linux build keeps its C++ API visible by default. Windows requires
    # an explicit export table for the same API so usdGenImaging can link
    # against the core DLL; CMake ignores this property elsewhere.
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
    target_link_options(usdGen PRIVATE "LINKER:--exclude-libs,ALL")
endif()
