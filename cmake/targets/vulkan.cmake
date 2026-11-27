# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Vulkan backend. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# Vulkan host backend (init, dispatch, memory, errors) plus operator pipelines
# ---------------------------------------------------------------------------
# Vulkan runtime auto-detection: default ON when a toolchain (headers +
# loader, glslang/glslc + spirv-val) is present, OFF otherwise. An explicit
# -DUSDGEN_ENABLE_VULKAN_RUNTIME=ON/OFF is always respected; the probe runs
# only when the variable is undefined. When VULKAN_SDK is unset, the vendored
# .vulkan-toolchain/root/usr tree is used as a fallback hint.
#
# Loader hint: hosts often carry only the runtime libvulkan.so.1 (no dev
# symlink), which FindVulkan's name search misses. A versioned runtime file
# still links fine by explicit path, so probe both well-known locations.
function(_usdgen_hint_vulkan_loader)
    if(DEFINED Vulkan_LIBRARY)
        return()
    endif()
    foreach(_lib
        "/lib/aarch64-linux-gnu/libvulkan.so.1"
        "/usr/lib/aarch64-linux-gnu/libvulkan.so.1"
        "/lib/x86_64-linux-gnu/libvulkan.so.1"
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vklocal/sdk/lib/libvulkan.so")
        if(EXISTS "${_lib}")
            set(Vulkan_LIBRARY "${_lib}" CACHE FILEPATH
                "Vulkan loader (host runtime fallback hint)")
            break()
        endif()
    endforeach()
endfunction()
# USDGEN_ENABLE_VULKAN mirrors USDGEN_ENABLE_CUDA naming and is an alias of
# the canonical USDGEN_ENABLE_VULKAN_RUNTIME below. An explicit -D of the
# alias seeds the canonical variable so the auto-probe is skipped, exactly
# as if the canonical name had been given (see the reconcile after option()).
if(DEFINED USDGEN_ENABLE_VULKAN AND NOT DEFINED USDGEN_ENABLE_VULKAN_RUNTIME)
    set(USDGEN_ENABLE_VULKAN_RUNTIME "${USDGEN_ENABLE_VULKAN}" CACHE BOOL
        "Build the Vulkan execution runtime (production backend route plus injected test provider)" FORCE)
endif()
set(_usdgen_vulkan_runtime_default OFF)
if(NOT DEFINED USDGEN_ENABLE_VULKAN_RUNTIME)
    set(_usdgen_vulkan_sdk_hint "$ENV{VULKAN_SDK}")
    if(_usdgen_vulkan_sdk_hint STREQUAL "" OR NOT EXISTS "${_usdgen_vulkan_sdk_hint}")
        set(_usdgen_vulkan_sdk_hint "${CMAKE_CURRENT_SOURCE_DIR}/.vulkan-toolchain/root/usr")
    endif()
    # Vendored headers as a FindVulkan cache hint (never ENV: the vendored
    # tree carries no runtime loader, and pointing VULKAN_SDK at it would
    # hide the host loader from FindVulkan further down the search order).
    # -DVulkan_INCLUDE_DIR always wins.
    if(EXISTS "${_usdgen_vulkan_sdk_hint}/include/vulkan/vulkan_core.h" AND
       NOT DEFINED Vulkan_INCLUDE_DIR)
        set(Vulkan_INCLUDE_DIR "${_usdgen_vulkan_sdk_hint}/include"
            CACHE PATH "Vulkan headers (vendored fallback hint)")
    endif()
    _usdgen_hint_vulkan_loader()
    find_package(Vulkan 1.2 QUIET)
    find_program(_usdgen_vulkan_probe_glslang NAMES glslangValidator glslang glslc
        HINTS "$ENV{VULKAN_SDK}/bin" "${_usdgen_vulkan_sdk_hint}/bin")
    find_program(_usdgen_vulkan_probe_spirv NAMES spirv-val
        HINTS "$ENV{VULKAN_SDK}/bin" "${_usdgen_vulkan_sdk_hint}/bin")
    if(Vulkan_FOUND AND _usdgen_vulkan_probe_glslang AND _usdgen_vulkan_probe_spirv)
        set(_usdgen_vulkan_runtime_default ON)
        message(STATUS "usdGen Vulkan: auto-enabled (toolchain at ${_usdgen_vulkan_sdk_hint})")
    else()
        message(STATUS "usdGen Vulkan: no toolchain; runtime stays OFF (set VULKAN_SDK or -DUSDGEN_ENABLE_VULKAN_RUNTIME=ON)")
    endif()
    unset(_usdgen_vulkan_probe_glslang CACHE)
    unset(_usdgen_vulkan_probe_spirv CACHE)
    unset(_usdgen_vulkan_sdk_hint)
endif()
option(USDGEN_ENABLE_VULKAN_RUNTIME "Build the Vulkan execution runtime (production backend route plus injected test provider)" ${_usdgen_vulkan_runtime_default})
option(USDGEN_ENABLE_VULKAN "Build the Vulkan execution runtime (alias of USDGEN_ENABLE_VULKAN_RUNTIME; mirrors USDGEN_ENABLE_CUDA)" ${USDGEN_ENABLE_VULKAN_RUNTIME})
if(NOT "${USDGEN_ENABLE_VULKAN}" STREQUAL "${USDGEN_ENABLE_VULKAN_RUNTIME}")
    message(WARNING "usdGen Vulkan: USDGEN_ENABLE_VULKAN=${USDGEN_ENABLE_VULKAN} disagrees with "
        "USDGEN_ENABLE_VULKAN_RUNTIME=${USDGEN_ENABLE_VULKAN_RUNTIME}; using the RUNTIME value")
    set(USDGEN_ENABLE_VULKAN "${USDGEN_ENABLE_VULKAN_RUNTIME}" CACHE BOOL
        "Build the Vulkan execution runtime (alias of USDGEN_ENABLE_VULKAN_RUNTIME; mirrors USDGEN_ENABLE_CUDA)" FORCE)
endif()
option(USDGEN_BUILD_VULKAN_TESTS "Build native Vulkan compute and injected-provider tests" OFF)
# Vulkan source list: SHARED usdGenVulkan in production (RUNTIME ON) and
# test-only builds (RUNTIME OFF + VULKAN_TESTS ON). The canonical UUID
# registry always lives in the same DSO as DeviceContext, never one static
# copy per consumer.
set(_usdgen_vulkan_sources
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/deviceFactory.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/deviceContext.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/chargedBuffer.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/memoryPool.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/queueTimeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/completionService.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/externalRetirement.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/defaultProvider.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/compile/program.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/exec/shaderLibrary.cpp)
if(USDGEN_ENABLE_VULKAN_RUNTIME OR (USDGEN_BUILD_TESTS AND USDGEN_BUILD_VULKAN_TESTS))
    # Vendored header fallback when VULKAN_SDK is unset (mirrors the
    # auto-probe; cache hint, never ENV, so the host loader stays visible).
    if("$ENV{VULKAN_SDK}" STREQUAL "")
        set(_usdgen_vulkan_fallback_sdk "${CMAKE_CURRENT_SOURCE_DIR}/.vulkan-toolchain/root/usr")
        if(EXISTS "${_usdgen_vulkan_fallback_sdk}/include/vulkan/vulkan_core.h"
           AND NOT DEFINED Vulkan_INCLUDE_DIR)
            set(Vulkan_INCLUDE_DIR "${_usdgen_vulkan_fallback_sdk}/include"
                CACHE PATH "Vulkan headers (vendored fallback hint)")
        endif()
        unset(_usdgen_vulkan_fallback_sdk)
    endif()
    _usdgen_hint_vulkan_loader()
    find_package(Vulkan 1.2 REQUIRED)
    # usdGenVulkan (SHARED, step 8.2): the Vulkan runtime. The registry loads it after the primitives
    # (usdGenExtensions.txt); its usdGenRegister adds the Vulkan kernel
    # rows and registers the "vulkan" backend, unavailable without the
    # runtime (test-only builds), where tests inject a provider. One DSO holds the canonical UUID registry.
    add_library(usdGenVulkan SHARED ${_usdgen_vulkan_sources}
        libs/core/backends/usdGenVulkan/usdGenVulkan/executionBackendVulkanProvider.cpp
        libs/core/backends/usdGenVulkan/usdGenVulkan/register.cpp)
    # usdGen and the Vulkan SDK only (gate testUsdGenNoDisallowedLink_usdGenVulkan).
    target_link_libraries(usdGenVulkan
        PUBLIC usdGen Vulkan::Vulkan Threads::Threads
        PRIVATE ${CMAKE_DL_LIBS})
    target_include_directories(usdGenVulkan PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenVulkan>
        "$<INSTALL_INTERFACE:include>")
    set_target_properties(usdGenVulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)
    usdgen_hidden_library(usdGenVulkan)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
        target_link_options(usdGenVulkan PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
    if(USDGEN_ENABLE_VULKAN_RUNTIME)
        target_compile_definitions(usdGenVulkan PRIVATE USDGEN_ENABLE_VULKAN_RUNTIME)
    endif()
    add_dependencies(usdGenWithPrimitives usdGenVulkan)
    list(APPEND USDGEN_EXTENSION_TARGETS usdGenVulkan)
    # Primitive Vulkan targets (step 8.6a), each sourced by GLOB_RECURSE of
    # its folder so a git mv moves membership (reconfigure after one):
    # SHARED usdGenCurvesVulkan (usdGenCurves/vulkan/) and OBJECT
    # usdGenSurfaceVulkan, usdGenExprVulkan (usdGen{Surface,Expr}/vulkan/,
    # linked into their consumer, usdGenCurvesVulkan). An OBJECT folder
    # without sources yet makes no target.
    file(GLOB_RECURSE _usdgen_curves_vulkan_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenCurves/usdGenCurves/vulkan/*.cpp")
    add_library(usdGenCurvesVulkan SHARED ${_usdgen_curves_vulkan_sources})
    target_link_libraries(usdGenCurvesVulkan
        PUBLIC usdGen
        PRIVATE usdGenVulkan usdGenSurface usdGenExpr usdGenCurves usdGenMath ${CMAKE_DL_LIBS})
    foreach(_usdgen_gpu_object usdGenSurface usdGenExpr)
        file(GLOB_RECURSE _usdgen_gpu_object_sources
            "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/${_usdgen_gpu_object}/${_usdgen_gpu_object}/vulkan/*.cpp")
        if(_usdgen_gpu_object_sources)
            add_library(${_usdgen_gpu_object}Vulkan OBJECT ${_usdgen_gpu_object_sources})
            # Same includes and deps as usdGenVulkan; its symbols resolve
            # through usdGenCurvesVulkan, which links it.
            target_link_libraries(${_usdgen_gpu_object}Vulkan PRIVATE
                usdGen Vulkan::Vulkan Threads::Threads usdGenSurface usdGenExpr usdGenCurves usdGenMath)
            target_include_directories(${_usdgen_gpu_object}Vulkan PRIVATE
                ${USDGEN_PRIMITIVE_INCLUDE_DIRS}
                ${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenVulkan)
            if(USDGEN_ENABLE_VULKAN_RUNTIME)
                target_compile_definitions(${_usdgen_gpu_object}Vulkan PRIVATE USDGEN_ENABLE_VULKAN_RUNTIME)
            endif()
            set_target_properties(${_usdgen_gpu_object}Vulkan PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                CXX_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON)
            # Linked into usdGenCurvesVulkan: its symbols export with that DLL's macro.
            target_compile_definitions(${_usdgen_gpu_object}Vulkan PRIVATE usdGenCurvesVulkan_EXPORTS)
            # The pipelines that call them live in usdGenCurvesVulkan (8.6d).
            target_sources(usdGenCurvesVulkan PRIVATE $<TARGET_OBJECTS:${_usdgen_gpu_object}Vulkan>)
        endif()
    endforeach()
    # RTLD_LOCAL dlopen finds libusdGenVulkan.so beside it in an install.
    set_target_properties(usdGenCurvesVulkan PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        INSTALL_RPATH "$ORIGIN")
    usdgen_hidden_library(usdGenCurvesVulkan)
    target_include_directories(usdGenCurvesVulkan PUBLIC
        ${USDGEN_PRIMITIVE_INCLUDE_DIRS}
        "$<INSTALL_INTERFACE:include>")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
        target_link_options(usdGenCurvesVulkan PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
    add_dependencies(usdGenWithPrimitives usdGenCurvesVulkan)
    # Loaded after usdGenVulkan: its usdGenRegister names the "vulkan" backend.
    list(APPEND USDGEN_EXTENSION_TARGETS usdGenCurvesVulkan)
    install(TARGETS usdGenCurvesVulkan
        RUNTIME DESTINATION "${USDGEN_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${USDGEN_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${USDGEN_INSTALL_LIBDIR}"
            COMPONENT usdGenImportLibraries EXCLUDE_FROM_ALL)
    # Test link lines name these; both are usdGenVulkan.
    # The curve pipelines live in usdGenCurvesVulkan, which tests use directly.
    add_library(usdGenVulkanNative INTERFACE)
    # usdGenMath: curve headers spell the UsdGen math types (step 11).
    target_link_libraries(usdGenVulkanNative INTERFACE usdGenVulkan
        $<BUILD_INTERFACE:usdGenCurvesVulkan> $<BUILD_INTERFACE:usdGenMath>)
    add_library(usdGenVulkanFactory INTERFACE)
    target_link_libraries(usdGenVulkanFactory INTERFACE usdGenVulkan
        $<BUILD_INTERFACE:usdGenCurvesVulkan> $<BUILD_INTERFACE:usdGenMath>)
    find_program(USDGEN_GLSLANG_EXECUTABLE NAMES glslangValidator glslang glslc
        HINTS "$ENV{VULKAN_SDK}/bin" "${CMAKE_CURRENT_SOURCE_DIR}/.vulkan-toolchain/root/usr/bin" REQUIRED)
    find_program(USDGEN_SPIRV_VAL_EXECUTABLE NAMES spirv-val
        HINTS "$ENV{VULKAN_SDK}/bin" "${CMAKE_CURRENT_SOURCE_DIR}/.vulkan-toolchain/root/usr/bin" REQUIRED)
    # Shader compiler spelling: glslangValidator takes `-V --target-env
    # <env> <in> -o <out>`; glslc (Shaderc, same SPIR-V) takes
    # `--target-env=<env> -fshader-stage=compute <in> -o <out>`. The flag
    # variable keeps every shader rule below compiler-agnostic.
    if(USDGEN_GLSLANG_EXECUTABLE MATCHES "[Gg]lslc(\\.exe)?$")
        set(_usdgen_vulkan_glsl_flags --target-env=vulkan1.2 -fshader-stage=compute)
    else()
        set(_usdgen_vulkan_glsl_flags -V --target-env vulkan1.2)
    endif()
    set(_usdgen_vulkan_spirv_files)
    set(_usdgen_vulkan_curve_shader_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenCurves/usdGenCurves/vulkan/shaders")
    set(_usdgen_vulkan_runtime_shader_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/shaders")
    set(_usdgen_vulkan_surface_shader_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenSurface/usdGenSurface/vulkan/shaders")
    set(_usdgen_vulkan_expr_shader_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenExpr/usdGenExpr/vulkan/shaders")
    function(_usdgen_compile_vulkan_shader _dir _tag _file)
        set(_shader "${_dir}/${_file}.comp")
        set(_spv "${CMAKE_CURRENT_BINARY_DIR}/vulkan/${_file}.spv")
        add_custom_command(OUTPUT "${_spv}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/vulkan"
            COMMAND "${USDGEN_GLSLANG_EXECUTABLE}" ${_usdgen_vulkan_glsl_flags}
                "${_shader}" -o "${_spv}"
            COMMAND "${USDGEN_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.2 "${_spv}"
            DEPENDS "${_shader}" ${ARGN} VERBATIM)
        set(_usdgen_vulkan_${_tag}_shader "${_shader}" PARENT_SCOPE)
        set(_usdgen_vulkan_${_tag}_spv "${_spv}" PARENT_SCOPE)
        list(APPEND _usdgen_vulkan_spirv_files "${_spv}")
        set(_usdgen_vulkan_spirv_files "${_usdgen_vulkan_spirv_files}" PARENT_SCOPE)
    endfunction()
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} width width)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} length lengthScale)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} blend widthBlend)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} compare nonWidthCompare)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} compaction lengthCompaction)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} set lengthSet)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} topo topologyCompare)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} po pointOverride)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} grow curveGrow)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} scatter_grow scatterGrow)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_surface_shader_dir} rfval rootFramesValidate)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_surface_shader_dir} rfbuild rootFrames)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_surface_shader_dir} rfgather rootFrameGather)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} noise_validate noiseValidate)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} noise_generate noiseGenerate)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} deform_evaluate deformEvaluate)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} deform_apply deformApply)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} deform_evaluate_verify deformEvaluateVerify)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} deform_evaluate_fill deformEvaluateFill)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} deform_evaluate_cached deformEvaluateCached)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} overlap widthOverlapWitness)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} resample_vk resampleVk)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_surface_shader_dir} mapvk_sample mapVkSample)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} rbfvk_extent rbfVkExtent)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} rbfvk_gram rbfVkGram)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} rbfvk_build_matrix rbfVkBuildMatrix)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} rbfvk_lu rbfVkLu)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} rbfvk_rhs rbfVkRhs)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} rbfvk_trisolve rbfVkTriSolve)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} rbfvk_evaluate rbfVkEvaluate)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} rbfvk_apply rbfVkApply)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} rbfvk_verify rbfVkVerify)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} rbfvk_fill rbfVkFill)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} rbfvk_evaluate_cached rbfVkEvaluateCached)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} picktile_pick picktileVkPick)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} picktile_footprint picktileVkFootprint)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} picktile_tiles picktileVkTiles)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} picktile_bounds picktileVkBounds)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_expr_shader_dir} exprvk_evaluate exprVkEvaluate)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_expr_shader_dir} exprvk_context exprVkContext)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_expr_shader_dir} exprvk_width exprVkWidth)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_expr_shader_dir} exprvk_exprop exprVkExprOp)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_expr_shader_dir} exprvk_pack exprVkPack)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} cut_extend lengthCutExtend)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} reparam lengthReparam)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} minimum lengthMinimum)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} literal_v1 lengthLiteralV1)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} envelope_v1 lengthEnvelopeV1
        "${_usdgen_vulkan_curve_shader_dir}/float32Envelope.glsl")
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} device_v1 lengthDeviceV1)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} stylevk_ramp styleVkWidthRamp)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} stylevk_length styleVkLength)
    _usdgen_compile_vulkan_shader(${_usdgen_vulkan_curve_shader_dir} stylevk_grow styleVkGrow)
    add_custom_target(usdGenVulkanShaders ALL DEPENDS ${_usdgen_vulkan_spirv_files})
    add_dependencies(usdGenVulkan usdGenVulkanShaders)
    add_dependencies(usdGenCurvesVulkan usdGenVulkanShaders)
    if(USDGEN_ENABLE_VULKAN_RUNTIME)
        install(FILES ${_usdgen_vulkan_spirv_files} DESTINATION share/usdGen/vulkan)
    endif()
    get_filename_component(_usdgen_vulkan_library_dir "${USDGEN_INSTALL_LIBDIR}"
        ABSOLUTE BASE_DIR "${CMAKE_INSTALL_PREFIX}")
    file(RELATIVE_PATH _usdgen_vulkan_shader_relative_dir
        "${_usdgen_vulkan_library_dir}" "${CMAKE_INSTALL_PREFIX}/share/usdGen/vulkan")
    target_compile_definitions(usdGenVulkan PRIVATE
        USDGEN_VULKAN_SHADER_RELATIVE_DIR="${_usdgen_vulkan_shader_relative_dir}")
endif()

# The extension manifest (12.0): usdGenExtensions.txt beside usdGen, one
# file name per line in load order. UsdGenEngine::Create reads it when
# USDGEN_EXTENSIONS is unset; the install tree carries it beside the library.
set(_usdgen_ext_lines "")
foreach(_t IN LISTS USDGEN_EXTENSION_TARGETS)
    string(APPEND _usdgen_ext_lines "$<TARGET_FILE_NAME:${_t}>\n")
endforeach()
file(GENERATE OUTPUT "$<TARGET_FILE_DIR:usdGen>/usdGenExtensions.txt"
    CONTENT "${_usdgen_ext_lines}")
install(FILES "$<TARGET_FILE_DIR:usdGen>/usdGenExtensions.txt"
    DESTINATION "${USDGEN_INSTALL_LIBDIR}")
unset(_usdgen_ext_lines)
