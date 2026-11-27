# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Primitive DLLs (usdGenSurface, usdGenExpr, usdGenCurves). Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# Primitives (SHARED, step 6.4): usdGenSurface <- usdGenExpr <- usdGenCurves,
# usdGenSurface <- usdGenCurves. Each exports usdGenRegister (register.cpp);
# UsdGenEngine::Create loads them from usdGen's directory (usdGenExtensions.txt).
# ---------------------------------------------------------------------------
function(_usdgen_add_primitive _name _root)
    file(GLOB_RECURSE _sources "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/${_root}/*.cpp")
    # cuda/ and vulkan/ build into the primitive's GPU targets (cuda.cmake,
    # vulkan.cmake).
    list(FILTER _sources EXCLUDE REGEX "/primitives/${_root}/${_root}/(cuda|vulkan)/")
    if(ARGN)
        list(REMOVE_ITEM _sources ${ARGN})
    endif()
    add_library(${_name} SHARED ${_sources})
    target_include_directories(${_name} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/${_root}>
        $<INSTALL_INTERFACE:include>)
    # Hidden; the library's export.h macros name what crosses (12.3f).
    usdgen_hidden_library(${_name})
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
        target_link_options(${_name} PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
endfunction()

# frontend.cpp builds the vendored SeExpr frontend's renamed namespace
# (usdGenExpr links usdGenExprFrontend).
set_source_files_properties(libs/core/primitives/usdGenExpr/usdGenExpr/expressions/frontend.cpp PROPERTIES
    COMPILE_DEFINITIONS "SeExpr2=UsdGenSeExprFrontend;SeExprInternal2=UsdGenSeExprFrontendInternal;USDGEN_SEEXPR_SANDBOX"
    INCLUDE_DIRECTORIES "${_usdgen_frontend_dir}")

set(_usdgen_curves_hosted
    libs/core/primitives/usdGenSurface/usdGenSurface/ops/scatter.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/ops/scatterModes.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/surfaceCageSource.cpp
    libs/core/primitives/usdGenExpr/usdGenExpr/ops/exprOp.cpp)
list(TRANSFORM _usdgen_curves_hosted PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
_usdgen_add_primitive(usdGenSurface usdGenSurface ${_usdgen_curves_hosted})
usdgen_opensubdiv_target(_usdgen_osd_cpu)
target_link_libraries(usdGenSurface
    PUBLIC usdGen
    # attributeBake.cpp writes Ptex files: its own PtexWriter copy, no state
    # shared with usdGen's reader cache.
    PRIVATE usdGenMath ${_usdgen_osd_cpu} usdGen_ptex ${CMAKE_DL_LIBS})
target_include_directories(usdGenSurface PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenCurves)
_usdgen_add_primitive(usdGenExpr usdGenExpr ${_usdgen_curves_hosted})
target_link_libraries(usdGenExpr
    PUBLIC usdGen usdGenSurface
    PRIVATE usdGenMath usdGen_seexpr usdGenExprFrontend ${CMAKE_DL_LIBS})
target_include_directories(usdGenExpr PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenCurves>)
_usdgen_add_primitive(usdGenCurves usdGenCurves)
target_sources(usdGenCurves PRIVATE ${_usdgen_curves_hosted})
# deform.h and api/deformBinding.h expose usdGenMath/rbf.h. usdGenMath is not
# installed (N-7), so it stays out of the export interface.
target_link_libraries(usdGenCurves
    PUBLIC usdGen usdGenSurface usdGenExpr $<BUILD_INTERFACE:usdGenMath>
    PRIVATE usdGen_seexpr ${CMAKE_DL_LIBS})
# usdGen loads them at UsdGenEngine::Create, from its own directory, in this
# order; the backend files follow (cuda.cmake and vulkan.cmake append,
# vulkan.cmake writes the manifest usdGenExtensions.txt).
set(USDGEN_EXTENSION_TARGETS usdGenSurface usdGenExpr usdGenCurves)
# What hosts and tests link instead of usdGen (step 6.4).
add_library(usdGenWithPrimitives INTERFACE)
target_link_libraries(usdGenWithPrimitives INTERFACE usdGen usdGenSurface usdGenExpr usdGenCurves)
add_dependencies(usdGenWithPrimitives usdGenSurface usdGenExpr usdGenCurves)
add_library(usdGen::usdGenWithPrimitives ALIAS usdGenWithPrimitives)
