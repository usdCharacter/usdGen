# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# usdGen, the engine core. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# usdGen (SHARED — the engine core, no usd/usdImaging/hd linkage)
# ---------------------------------------------------------------------------
# Step 6.4: the primitives are their own DLLs (below); usdGen compiles the
# engine only. usdGenCuda (cuda.cmake, step 8.3) and usdGenVulkan
# (vulkan.cmake, step 8.2) are their own DLLs.
file(GLOB_RECURSE USDGEN_CORE_SOURCES libs/core/usdGen/*.cpp)
# Include roots plus the old-spelling shim (cmake/IncludeShim.cmake).
include(cmake/IncludeShim.cmake)
set(USDGEN_ENGINE_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/usdGen>
    # Step 11.2: api and engine headers name the math types, so the
    # "usdGenMath/<type>.h" root is part of usdGen's interface (D16). Headers
    # only: the alias-phase math types are inline over Gf; no link to usdGenMath.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/base/usdGenMath>
    $<BUILD_INTERFACE:${USDGEN_SHIM_DIR}>)
# Primitive roots: each primitive target publishes its own; usdGen publishes
# none (step 8.5). The backends still read primitive headers in their public
# headers until step 8.6 and republish these.
set(USDGEN_PRIMITIVE_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenCurves>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenSurface>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenExpr>)
# exprEval.{h,cpp} (a stale SeExpr2-engine evaluator design) and its unwired
# tests/testUsdGenExpr.cpp are deleted: the IR interpreter in
# expressions/irExec.h is the design, shared by the CUDA kernel and by
# expressions/cpuEvaluator.cpp.
add_library(usdGen SHARED ${USDGEN_CORE_SOURCES})
# The host suffix port (deformEvaluateCpu.h, inlined into
# deformPipeline.cpp) needs sqrt vectorized: with math-errno the vectorizer
# keeps the explicit-fma loop's sqrt scalar and the suffix runs ~2x slower.
# Values are unchanged (errno only); nothing in the library reads errno.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(libs/core/primitives/usdGenCurves/usdGenCurves/vulkan/deformPipeline.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-math-errno")
endif()
target_include_directories(usdGen PUBLIC
    ${USDGEN_ENGINE_INCLUDE_DIRS}
    $<INSTALL_INTERFACE:include>)
# No pxr library (step 11.8, Option B): the API spells the engine's own base
# types, and trace, debug and report go through host-installed sinks. Hosts
# link what they use (usdGenUsd re-exports tf gf vt sdf).
# usdGenMath is PRIVATE (N-7, plan §1.2): it is not installed, so it must
# not sit in usdGen's export interface (install(EXPORT) would then reference
# a non-installed target); usdGen's own sources still see its headers
# through the link, and no public header includes usdGenMath/...
target_link_libraries(usdGen
    PRIVATE usdGenMath ${CMAKE_DL_LIBS})
if(WIN32)
    target_link_libraries(usdGen PRIVATE Synchronization)  # WaitOnAddress
endif()
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
