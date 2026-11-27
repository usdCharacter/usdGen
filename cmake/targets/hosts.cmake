# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Hosts: usdGenUsd, usdGenSchema, usdGenHydraCurves, usdGenImaging, usdGenPomade, test utils, plugin resources. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# usdGenUsd (SHARED, host): the only code that maps SdfPath to engine node
# keys (step 7). Links usdGen + sdf + usd; nothing in libs/core links it.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE USDGEN_USD_SOURCES libs/hosts/usdGenUsd/usdGenUsd/*.cpp)
# The stage builder is fenced to T0/offline/tests: usdGenTestUtilsHd compiles it.
list(FILTER USDGEN_USD_SOURCES EXCLUDE REGEX "usdGenGraphDescBuilderStage\\.cpp$")
add_library(usdGenUsd SHARED ${USDGEN_USD_SOURCES})
target_include_directories(usdGenUsd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenUsd>
    $<INSTALL_INTERFACE:include>)
# tf gf vt sdf are PUBLIC: its headers spell them, and tests that link
# usdGenUsd get them from here (usdGen links no pxr library, step 11.8).
target_link_libraries(usdGenUsd
    PUBLIC usdGenWithPrimitives sdf tf gf vt
    PRIVATE usd hio)
set_target_properties(usdGenUsd PROPERTIES
    CXX_VISIBILITY_PRESET default
    WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ---------------------------------------------------------------------------
# usdGenSchema (SHARED, minimal — tokens + extent registration)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE USDGEN_SCHEMA_SOURCES libs/hosts/usdGenUsd/schema/*.cpp)
add_library(usdGenSchema SHARED ${USDGEN_SCHEMA_SOURCES})
target_include_directories(usdGenSchema PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenUsd/schema>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(usdGenSchema
    PUBLIC usd
    PRIVATE tf usdGeom usdGenMath)

# ---------------------------------------------------------------------------
# usdGenHydraCurves (SHARED, host): the curves' Hydra adapter (step 9.2). The
# curve publisher and publication types, fur occlusion, preview colours, the
# tile publisher and the instancer. Reads curve sets through usdGen/api by
# channel name; never links a primitive or a backend (link gate 9c).
# ---------------------------------------------------------------------------
file(GLOB USDGEN_HYDRA_CURVES_SOURCES
    libs/hosts/usdGenHydra/primitives/curves/usdGenHydraCurves/*.cpp)
add_library(usdGenHydraCurves SHARED ${USDGEN_HYDRA_CURVES_SOURCES})
target_include_directories(usdGenHydraCurves PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenHydra/primitives/curves>
    $<INSTALL_INTERFACE:include>)
# usdGenMath (build tree only): publication.h reaches usdGenMath/rbf.h through
# api/deformBinding.h.
target_link_libraries(usdGenHydraCurves
    PUBLIC usdGen usdGenUsd $<BUILD_INTERFACE:usdGenMath> hd sdf tf gf vt
    PRIVATE trace work)
set_target_properties(usdGenHydraCurves PROPERTIES
    CXX_VISIBILITY_PRESET default
    WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ---------------------------------------------------------------------------
# usdGenImaging (SHARED — scene-index plugins + adapters)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE USDGEN_IMAGING_SOURCES libs/hosts/usdGenHydra/usdGenImaging/*.cpp)
add_library(usdGenImaging SHARED ${USDGEN_IMAGING_SOURCES})
target_include_directories(usdGenImaging PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenHydra>
    $<INSTALL_INTERFACE:include>)
# usdGenSurface and usdGenExpr stay for the brush and tools APIs (step 13).
target_link_libraries(usdGenImaging
    PUBLIC hd hdsi hf usd usdGeom usdShade sdr usdImaging
           sdf tf gf vt trace pxOsd usdGen usdGenSurface usdGenExpr usdGenUsd
           usdGenHydraCurves $<BUILD_INTERFACE:usdGenMath>
    PRIVATE hio js
    PRIVATE usdGenSchema)
set_target_properties(usdGenImaging PROPERTIES
    CXX_VISIBILITY_PRESET default
    # Produce the import library that downstream Windows consumers need.
    # This is ignored by non-Windows generators.
    WINDOWS_EXPORT_ALL_SYMBOLS ON)  # plugins need exported symbols
# -fno-math-errno on the scatter compile (GNU/Clang only): libm errno
# / fenv is never read anywhere in libs/ (grep clean), so IEEE results
# are bit-identical while every builtin-expanded sqrt drops its
# errno-check branch. Needs re-verifying if errno use appears.
# Deliberately scatter-only: the same flag on usdGenInstancer.cpp also
# wins cards/twist/notwist (-0.3..-0.7ms) but perturbs the
# register-fragile tangent path (+0.7..+1.8ms); declined.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        libs/core/primitives/usdGenSurface/usdGenSurface/ops/scatter.cpp
        PROPERTIES COMPILE_OPTIONS -fno-math-errno)
endif()

# M1: the usdGenShaders SdrDiscoveryPlugin compiles into usdGenImaging; its
# resource dir is resolved at runtime by name (PlugRegistry lookup of
# "usdGenShaders"), so no per-target macro is needed (10-build-dependencies-
# testing.md §4.4 alternative).

# ---------------------------------------------------------------------------
# usdGenPomade (SHARED — plan/17 Pomade authoring model + scene index, P0)
#
# PomadeModel, the pomadeApi C ABI, the host-staged transport, the selection
# and overlay state (pomadeSelection.cpp, pomadeGizmo.cpp: plan/18 V1), the
# UsdGenPomadeSceneIndexPlugin/SceneIndex (libs/usdGenPomade/usdGenPomade/,
# imaging under imaging/) and the P1 PomadeCommitter (pomadeCommit.cpp, the
# only TU that touches usd, for hydrate and Save groom; gate B-1 stays true
# for usdGen itself). Links usdGen only for the gpu/ primitives
# (DeviceBuffer, execution resources). The CUDA kernel (pomadeKernels.cu)
# joins in cuda.cmake's USDGEN_ENABLE_CUDA branch, next to the usdGenCuda wiring.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE USDGEN_POMADE_SOURCES libs/usdGenPomade/*.cpp)
list(FILTER USDGEN_POMADE_SOURCES EXCLUDE REGEX "\\.cu$")
add_library(usdGenPomade SHARED ${USDGEN_POMADE_SOURCES})
target_include_directories(usdGenPomade PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/usdGenPomade>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(usdGenPomade
    PUBLIC hd sdf tf gf vt
    PRIVATE usd usdGeom hio usdGenWithPrimitives usdGen_ptex usdGenImaging)
set_target_properties(usdGenPomade PROPERTIES
    CXX_VISIBILITY_PRESET default
    WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ---------------------------------------------------------------------------
# usdGenTestUtils (STATIC, T0 — no Hydra)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE USDGEN_TESTUTILS_SOURCES testutils/usdGenTestUtils/*.cpp)
add_library(usdGenTestUtils STATIC ${USDGEN_TESTUTILS_SOURCES})
target_include_directories(usdGenTestUtils PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/testutils")
target_link_libraries(usdGenTestUtils PUBLIC usdGenWithPrimitives usdGenUsd)

# ---------------------------------------------------------------------------
# usdGenTestUtilsHd (STATIC, T1/T2/T3 — Hydra allowed)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE USDGEN_TESTUTILS_HD_SOURCES
    testutils/usdGenTestUtilsHd/*.cpp)
add_library(usdGenTestUtilsHd STATIC ${USDGEN_TESTUTILS_HD_SOURCES})
target_sources(usdGenTestUtilsHd PRIVATE
    libs/hosts/usdGenUsd/usdGenUsd/walk/usdGenGraphDescBuilderStage.cpp)
target_include_directories(usdGenTestUtilsHd PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/testutils")
target_link_libraries(usdGenTestUtilsHd PUBLIC
    usdGenImaging usdGenTestUtils
    hd hdsi usd usdImaging usdImagingGL hdSt hgiGL usdAppUtils hio)

# ---------------------------------------------------------------------------
# Build-tree plugin resources (generated at configure time so that the
# $<TARGET_FILE_NAME:*> generator expressions are expanded; mirrors the
# usdRig pattern of build/usd/<name>/resources).
# ---------------------------------------------------------------------------
set(_usdgen_schema_resources "${CMAKE_CURRENT_BINARY_DIR}/usd/usdGenSchema/resources")
set(_usdgen_schema_pluginfo "${_usdgen_schema_resources}/plugInfo.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenUsd/schema/resources/plugInfo.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenUsd/schema/resources/generatedSchema.usda")
file(READ
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenUsd/schema/resources/plugInfo.json"
    _usdgen_schema_pluginfo_content)
string(REPLACE "\"Type\": \"resource\"" "\"Type\": \"library\""
    _usdgen_schema_pluginfo_content "${_usdgen_schema_pluginfo_content}")
string(REPLACE
    "\"schemaIdentifier\": \"UsdGenDescription\""
    "\"implementsComputeExtent\": true, \"schemaIdentifier\": \"UsdGenDescription\""
    _usdgen_schema_pluginfo_content "${_usdgen_schema_pluginfo_content}")
string(REPLACE "\"Name\": \"usdGenSchema\""
    "\"LibraryPath\": \"../../../$<TARGET_FILE_NAME:usdGenSchema>\", \"Name\": \"usdGenSchema\""
    _usdgen_schema_pluginfo_content "${_usdgen_schema_pluginfo_content}")
file(GENERATE OUTPUT "${_usdgen_schema_pluginfo}"
    CONTENT "${_usdgen_schema_pluginfo_content}")
file(COPY
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenUsd/schema/resources/generatedSchema.usda"
    DESTINATION "${_usdgen_schema_resources}")

set(_usdgen_imaging_resources "${CMAKE_CURRENT_BINARY_DIR}/usd/usdGenImaging/resources")
set(_usdgen_imaging_pluginfo "${_usdgen_imaging_resources}/plugInfo.json")
file(READ
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenImaging/resources/plugInfo.json.in"
    _usdgen_imaging_pluginfo_content)
file(GENERATE OUTPUT "${_usdgen_imaging_pluginfo}"
    CONTENT "${_usdgen_imaging_pluginfo_content}")
