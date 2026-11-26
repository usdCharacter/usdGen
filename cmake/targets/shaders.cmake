# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# usdGenShaders and the offline utilities. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# usdGenShaders (9th target, M1 — plan §1.2/§1.6): the usdGen hair preview
# glslfx shader defs + resources. No C++ of its own: the discovery plugin
# (UsdGenShadersDiscoveryPlugin, base SdrDiscoveryPlugin) is compiled into
# usdGenImaging, so this is an INTERFACE target carrying the resource
# installation. plugInfo points LibraryPath at libusdGenImaging and
# ShaderResources at shaders/ (mirrors stock usdShaders; 10 §4.4).
# ---------------------------------------------------------------------------
add_library(usdGenShaders INTERFACE)
add_library(usdGen::usdGenShaders ALIAS usdGenShaders)

# Offline authoring utility; leaves PointInstancer geometry instanced.
add_executable(usdGenBakeFur tools/bakeFur.cpp)
target_link_libraries(usdGenBakeFur PRIVATE usdGenWithPrimitives usdGeom usd)
install(TARGETS usdGenBakeFur RUNTIME DESTINATION bin)
# Profiling utility: plays a scene through the groom scene index frame by
# frame and reports per-frame cost, per-operator work and a Chrome trace
# (bin/trace_playback.ps1).
add_executable(usdGenTracePlayback tools/tracePlayback.cpp)
target_link_libraries(usdGenTracePlayback PRIVATE
    usdGenWithPrimitives usdGenImaging usdImaging hd trace tf usd sdf)
install(TARGETS usdGenTracePlayback RUNTIME DESTINATION bin)
# Offline authoring utility; writes procedural region/clump Ptex maps for a mesh.
add_executable(usdGenBakePtex tools/bakePtex.cpp)
target_link_libraries(usdGenBakePtex PRIVATE usdGenWithPrimitives usdGen_ptex usdGeom usd)
install(TARGETS usdGenBakePtex RUNTIME DESTINATION bin)

set(_usdgen_shader_resources "${CMAKE_CURRENT_BINARY_DIR}/usd/usdGenShaders/resources")
set(_usdgen_shader_pluginfo "${_usdgen_shader_resources}/plugInfo.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/resources/plugInfo.json.in")
file(READ
    "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/resources/plugInfo.json.in"
    _usdgen_shader_pluginfo_content)
file(GENERATE OUTPUT "${_usdgen_shader_pluginfo}"
    CONTENT "${_usdgen_shader_pluginfo_content}")
file(COPY
    "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/resources/shaders/"
    DESTINATION "${_usdgen_shader_resources}/shaders"
    PATTERN "usdGenHairModel.glslfx" EXCLUDE)
file(COPY
    "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/resources/hairLook.usda"
    DESTINATION "${_usdgen_shader_resources}")

# The strand model UsdGenHairStrands compiles (docs/storm-fur.md, "Strand
# models"). Storm compiles glslfx at draw time, so the choice travels as the
# one-line usdGenHairModel.glslfx the build writes beside the other shaders;
# the source copy selects the default so a source checkout still parses.
#   karis      Karis R/TT/TRT with the matching controls (default)
#   marschner  the absorption-based Marschner model imported grooms were
#              shaded with; also makes the fur bake publish the furDensity
#              primvar its hair occlusion reads (USDGEN_FUR_DENSITY)
#   rtxcr      RTXCR far-field scattering with Storm visibility and ambient
set(USDGEN_HAIR_SHADING_MODEL "karis" CACHE STRING
    "Strand shading model of UsdGenHairStrands: karis, marschner or rtxcr")
set_property(CACHE USDGEN_HAIR_SHADING_MODEL PROPERTY STRINGS karis marschner rtxcr)
string(TOLOWER "${USDGEN_HAIR_SHADING_MODEL}" _usdgen_hair_model)
if(NOT _usdgen_hair_model MATCHES "^(karis|marschner|rtxcr)$")
    message(FATAL_ERROR "USDGEN_HAIR_SHADING_MODEL must be karis, marschner or rtxcr, "
        "not '${USDGEN_HAIR_SHADING_MODEL}'")
endif()
string(TOUPPER "${_usdgen_hair_model}" _usdgen_hair_model_upper)
set(_usdgen_hair_model_source
    "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/resources/shaders/usdGenHairModel.glslfx")
set(_usdgen_hair_model_file "${_usdgen_shader_resources}/shaders/usdGenHairModel.glslfx")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_usdgen_hair_model_source}")
file(READ "${_usdgen_hair_model_source}" _usdgen_hair_model_content)
string(REGEX REPLACE "#define USDGEN_STRANDS_MODEL_[A-Z]+ 1"
    "#define USDGEN_STRANDS_MODEL_${_usdgen_hair_model_upper} 1"
    _usdgen_hair_model_content "${_usdgen_hair_model_content}")
file(CONFIGURE OUTPUT "${_usdgen_hair_model_file}"
    CONTENT "${_usdgen_hair_model_content}" @ONLY)
# Only the curve publisher reads it (the USDGEN_FUR_DENSITY default), so it is a source
# property: switching models recompiles that one file, not the library.
if(_usdgen_hair_model MATCHES "^(marschner|rtxcr)$")
    set_property(SOURCE libs/core/primitives/usdGenCurves/usdGenCurves/publish/curvePublisher.cpp APPEND PROPERTY
        COMPILE_DEFINITIONS USDGEN_FUR_LOCAL_DENSITY=1)
endif()
message(STATUS "usdGen: strand shading model ${_usdgen_hair_model}")
