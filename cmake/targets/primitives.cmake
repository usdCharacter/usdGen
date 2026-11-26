# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Primitive DLLs (usdGenSurface, usdGenExpr, usdGenCurves). Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# Primitives (SHARED, step 6.4): usdGenSurface <- usdGenExpr <- usdGenCurves,
# usdGenSurface <- usdGenCurves. Each exports usdGenRegister (register.cpp);
# UsdGenEngineEnsureInitialized loads them from usdGen's directory.
# ---------------------------------------------------------------------------
function(_usdgen_add_primitive _name _root)
    file(GLOB_RECURSE _sources "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/${_root}/*.cpp")
    list(REMOVE_ITEM _sources ${ARGN})
    add_library(${_name} SHARED ${_sources})
    target_include_directories(${_name} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/${_root}>
        $<INSTALL_INTERFACE:include>)
    # Same visibility contract as usdGen until step 12 (hidden + explicit API).
    set_target_properties(${_name} PROPERTIES
        VISIBILITY_INLINES_HIDDEN ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
        target_link_options(${_name} PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
endfunction()

# Primitive sources that stay compiled into usdGen: the CUDA/Vulkan backends
# still call them directly (the engine reaches them only through registered
# hooks since 6.5). Step 8 moves the backends out; then this list empties.
# Everything else in a primitive folder builds into that primitive's DLL.
set(USDGEN_ENGINE_RESIDENT_PRIMITIVE_SOURCES
    # input kinds, their views and engine hooks (backends read the views)
    libs/core/primitives/usdGenSurface/usdGenSurface/surfaceInput.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/maps/mapInput.cpp
    libs/core/primitives/usdGenCurves/usdGenCurves/types/curveSetInput.cpp
    libs/core/primitives/usdGenExpr/usdGenExpr/exprInput.cpp
    # CurveSet layout/view and limit surface (backends)
    libs/core/primitives/usdGenCurves/usdGenCurves/types/curveSet.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/limitSurface.cpp
    # CUDA/Vulkan backends (step 8)
    libs/core/primitives/usdGenExpr/usdGenExpr/expressions/context.cpp
    libs/core/primitives/usdGenExpr/usdGenExpr/expressions/frontend.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/surfaceCageSource.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/surfaceCageInterpolate.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/surfaceRootFrames.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/surfaceRootBindings.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/imagePayload.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/maps/ptexMap.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/ops/scatter.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/ops/scatterModes.cpp
    libs/core/primitives/usdGenSurface/usdGenSurface/types/rootSet.cpp
    libs/core/primitives/usdGenCurves/usdGenCurves/curveRootCapture.cpp
    libs/core/primitives/usdGenCurves/usdGenCurves/curveLoader.cpp
    libs/core/primitives/usdGenCurves/usdGenCurves/ops/rbfField.cpp
    libs/core/primitives/usdGenCurves/usdGenCurves/ops/width.cpp
    libs/core/primitives/usdGenCurves/usdGenCurves/ops/strandUtil.cpp
    libs/core/primitives/usdGenCurves/usdGenCurves/types/curveChunk.cpp
    libs/core/primitives/usdGenCurves/usdGenCurves/publish/concaveMaterialRemap.cpp)
list(TRANSFORM USDGEN_ENGINE_RESIDENT_PRIMITIVE_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/"
    OUTPUT_VARIABLE _usdgen_resident_abs)
target_sources(usdGen PRIVATE ${USDGEN_ENGINE_RESIDENT_PRIMITIVE_SOURCES})
set_source_files_properties(libs/core/primitives/usdGenExpr/usdGenExpr/expressions/frontend.cpp PROPERTIES
    COMPILE_DEFINITIONS "SeExpr2=UsdGenSeExprFrontend;SeExprInternal2=UsdGenSeExprFrontendInternal;USDGEN_SEEXPR_SANDBOX"
    INCLUDE_DIRECTORIES "${_usdgen_frontend_dir}")

_usdgen_add_primitive(usdGenSurface usdGenSurface ${_usdgen_resident_abs})
target_link_libraries(usdGenSurface
    PUBLIC usdGen
    # attributeBake.cpp writes Ptex files: its own PtexWriter copy, no state
    # shared with usdGen's reader cache.
    PRIVATE usdGenMath arch work trace pxOsd usdGen_ptex ${CMAKE_DL_LIBS})
_usdgen_add_primitive(usdGenExpr usdGenExpr ${_usdgen_resident_abs})
target_link_libraries(usdGenExpr
    PUBLIC usdGen usdGenSurface
    PRIVATE usdGenMath arch work trace usdGen_seexpr usdGenExprFrontend ${CMAKE_DL_LIBS})
_usdgen_add_primitive(usdGenCurves usdGenCurves ${_usdgen_resident_abs})
target_link_libraries(usdGenCurves
    PUBLIC usdGen usdGenSurface usdGenExpr
    PRIVATE usdGenMath arch work trace usdGen_seexpr ${CMAKE_DL_LIBS})
# usdGen loads them at registry build (UsdGenEngineEnsureInitialized), from
# its own directory, in this order.
set_property(SOURCE libs/core/usdGen/usdGen/registries/registrar.cpp APPEND PROPERTY
    COMPILE_DEFINITIONS
    "USDGEN_SURFACE_FILE=\"$<TARGET_FILE_NAME:usdGenSurface>\""
    "USDGEN_EXPR_FILE=\"$<TARGET_FILE_NAME:usdGenExpr>\""
    "USDGEN_CURVES_FILE=\"$<TARGET_FILE_NAME:usdGenCurves>\"")
# What hosts and tests link instead of usdGen (step 6.4).
add_library(usdGenWithPrimitives INTERFACE)
target_link_libraries(usdGenWithPrimitives INTERFACE usdGen usdGenSurface usdGenExpr usdGenCurves)
add_dependencies(usdGenWithPrimitives usdGenSurface usdGenExpr usdGenCurves)
add_library(usdGen::usdGenWithPrimitives ALIAS usdGenWithPrimitives)
