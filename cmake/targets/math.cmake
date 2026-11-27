# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# usdGenMath and the fp-contract policy. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# usdGenMath (STATIC, -ffp-contract=off)
# ---------------------------------------------------------------------------
set(_usdgen_frontend_dir "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/seexprFrontend/SeExpr2")
file(GLOB _usdgen_frontend_sources "${_usdgen_frontend_dir}/*.cpp")
list(FILTER _usdgen_frontend_sources EXCLUDE REGEX "ExprLLVMCodeGeneration\\.cpp$")
add_library(usdGenExprFrontend STATIC ${_usdgen_frontend_sources}
    "${_usdgen_frontend_dir}/generated/ExprParser.cpp"
    "${_usdgen_frontend_dir}/generated/ExprParserLex.cpp")
target_include_directories(usdGenExprFrontend PRIVATE
    "${_usdgen_frontend_dir}" "${_usdgen_frontend_dir}/generated")
target_compile_definitions(usdGenExprFrontend PRIVATE
    SeExpr2=UsdGenSeExprFrontend SeExprInternal2=UsdGenSeExprFrontendInternal
    USDGEN_SEEXPR_SANDBOX)
# The vendored SeExpr frontend uses its own platform switch rather than the
# compiler's _WIN32 macro in a few generated and plugin-loader sources.  In
# particular, without this definition MSVC tries to include dlfcn.h and
# unistd.h. Keep this target-local so its public API remains platform-neutral.
if(WIN32)
    target_compile_definitions(usdGenExprFrontend PRIVATE SEEXPR_WIN32)
endif()
target_link_libraries(usdGenExprFrontend PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(usdGenExprFrontend PROPERTIES
    USDGEN_VENDORED ON
    POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

file(GLOB_RECURSE USDGEN_MATH_SOURCES libs/base/usdGenMath/*.cpp)
add_library(usdGenMath STATIC ${USDGEN_MATH_SOURCES})
# Two roots: "usdGenMath/usdGenMath/x.h" (older spelling) and "usdGenMath/x.h".
target_include_directories(usdGenMath PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/base>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/base/usdGenMath>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/usdGen>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(usdGenMath PUBLIC
    nanoflann
    usdGen_seexpr)
if(TARGET rigExec::rigExecMath)
    target_link_libraries(usdGenMath PUBLIC rigExec::rigExecMath)
endif()
set_target_properties(usdGenMath PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# GPU parity: noise.cpp computes the Capture field the kernel replicates bit-exactly; no contraction (mirrors usdGenMath).
include(CheckCXXCompilerFlag)
set(_usdgen_host_expr_fp_options)
if(MSVC)
    if(USDGEN_FP_CONTRACT STREQUAL "off")
        # /fp:strict prevents contraction without enabling reassociation.
        set(_usdgen_host_expr_fp_options /fp:strict)
    elseif(USDGEN_FP_CONTRACT STREQUAL "fast")
        # /fp:fast also changes other arithmetic. Enable contraction alone
        # while retaining precise operations on toolchains that support it.
        check_cxx_compiler_flag("/fp:contract" USDGEN_FP_CONTRACT_MSVC_OK)
        if(NOT USDGEN_FP_CONTRACT_MSVC_OK)
            message(FATAL_ERROR "USDGEN_FP_CONTRACT=fast requires MSVC /fp:contract support")
        endif()
        set(_usdgen_host_expr_fp_options /fp:precise /fp:contract)
    else()
        message(FATAL_ERROR "USDGEN_FP_CONTRACT must be off or fast")
    endif()
else()
    check_cxx_compiler_flag("-ffp-contract=${USDGEN_FP_CONTRACT}" USDGEN_FP_CONTRACT_OK)
    if(USDGEN_FP_CONTRACT_OK)
        set(_usdgen_host_expr_fp_options "-ffp-contract=${USDGEN_FP_CONTRACT}")
    endif()
endif()
if(_usdgen_host_expr_fp_options)
    foreach(_usdgen_fp_target usdGen_seexpr usdGenExprFrontend)
        target_compile_options(${_usdgen_fp_target} PRIVATE ${_usdgen_host_expr_fp_options})
    endforeach()
    # usdGenMath per source: rbf.cpp keeps the toolchain default it had in
    # usdGen, the policy of the callers that inline CubicField::Displacement.
    set(_usdgen_fp_usdGenMath_sources ${USDGEN_MATH_SOURCES})
    list(REMOVE_ITEM _usdgen_fp_usdGenMath_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/base/usdGenMath/usdGenMath/rbf.cpp")
    # Every host instantiation of the shared IR/builtin headers must use the
    # same policy as the vendored SeExpr oracle. Otherwise ARM GCC's default
    # contraction emits FMAs while CUDA --fmad=false keeps separate rounding.
    # Keep these source-local; unrelated engine arithmetic is unchanged.
    # Grouped by the target that compiles them (step 6.4); source properties,
    # so a file keeps its policy when it changes target.
    # usdGenSurface.
    set(_usdgen_fp_usdGenSurface_sources
        libs/core/primitives/usdGenSurface/usdGenSurface/ops/scatterModes.cpp)
    set(_usdgen_fp_usdGenExpr_sources
        libs/core/primitives/usdGenExpr/usdGenExpr/vulkan/exprVkProgram.cpp
        libs/core/primitives/usdGenExpr/usdGenExpr/vulkan/exprVkParameters.cpp
        libs/core/primitives/usdGenExpr/usdGenExpr/vulkan/exprVkRun.cpp
        libs/core/primitives/usdGenExpr/usdGenExpr/vulkan/exprVkPlan.cpp
        libs/core/primitives/usdGenExpr/usdGenExpr/expressions/frontend.cpp
        libs/core/primitives/usdGenExpr/usdGenExpr/cpuParameters.cpp
        libs/core/primitives/usdGenExpr/usdGenExpr/expressions/cpuEvaluator.cpp
        libs/core/primitives/usdGenExpr/usdGenExpr/expressions/samplers.cpp)
    set(_usdgen_fp_tests_sources
        tests/testUsdGenSeExprOracle.cpp
        tests/usdGenSeExprOracle.cpp
        tests/testUsdGenExpressionCpuEval.cpp
        tests/testUsdGenExpressionSamplers.cpp
        tests/testUsdGenCudaExpressionParity.cpp
        tests/testUsdGenVulkanExprVkParityOracle.cpp)
    # usdGenCurves. strandUtil and the imported-groom kernels reproduce a
    # reference generator bit for bit; keep their float evaluation order
    # independent of the toolchain.
    set(_usdgen_fp_usdGenCurves_sources
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/strandUtil.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/noise.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/wind.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/wholeStrand.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/guideBlend.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/guideInterpolate.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/clumpModes.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/braid.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/twist.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/frizz.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/lengthRange.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/trim.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/tieClump.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/cardSource.cpp
        libs/core/primitives/usdGenCurves/usdGenCurves/ops/cardGeometry.cpp)
    set_property(SOURCE
        ${_usdgen_fp_usdGenMath_sources} ${_usdgen_fp_usdGenSurface_sources}
        ${_usdgen_fp_usdGenExpr_sources}
        ${_usdgen_fp_usdGenCurves_sources} ${_usdgen_fp_tests_sources}
        APPEND PROPERTY COMPILE_OPTIONS ${_usdgen_host_expr_fp_options})
endif()
