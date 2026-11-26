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
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/sourceGeneration.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/deviceGenerationAdapter.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/publicationJob.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/executionPlan.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/planExecutor.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/sessionProvider.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/defaultProvider.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/sourceWidthJob.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/lengthScalePipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/lengthCompactionPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/widthBlendPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/nonWidthComparePipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/topologyPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/pointOverridePipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/curveGrowPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/scatterGrowPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/rootFramesPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/rootFrameGatherPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/noisePipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/deformPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/rbfVkBinding.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/rbfVkDeformPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/surfaceVk.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/widthPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/resampleVkPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/mapVkImage.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/styleVkPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/picktileVk.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/exprVkProgram.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/exprVkEvaluate.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/exprVkContext.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/exprVkApply.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/exprVkParameters.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/exprVkPlan.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/exprVkRun.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/exprVkStructural.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/exprVkPack.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/growVkPipeline.cpp
    libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/growVkPlan.cpp
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
    # usdGenVulkan (SHARED, step 8.2): the Vulkan runtime and, until step 8.6,
    # the curve pipelines. The registry loads it after the primitives
    # (USDGEN_DEFAULT_EXTENSIONS); its usdGenRegister adds the Vulkan kernel
    # rows and registers the "vulkan" backend, unavailable without the
    # runtime (test-only builds), where tests inject a provider. One DSO holds the canonical UUID registry.
    add_library(usdGenVulkan SHARED ${_usdgen_vulkan_sources}
        libs/core/backends/usdGenVulkan/usdGenVulkan/executionBackendVulkanProvider.cpp
        libs/core/backends/usdGenVulkan/usdGenVulkan/registerCurvesVulkan.cpp
        libs/core/backends/usdGenVulkan/usdGenVulkan/register.cpp)
    # The primitives by name, not usdGenWithPrimitives: that target depends on
    # this one so every consumer of the engine finds it built.
    target_link_libraries(usdGenVulkan
        PUBLIC usdGen Vulkan::Vulkan Threads::Threads
        PRIVATE usdGenSurface usdGenExpr usdGenCurves usdGenMath arch ${CMAKE_DL_LIBS})
    target_include_directories(usdGenVulkan PUBLIC
        ${USDGEN_PRIMITIVE_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenVulkan>
        "$<INSTALL_INTERFACE:include>")
    set_target_properties(usdGenVulkan PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
        target_link_options(usdGenVulkan PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
    if(USDGEN_ENABLE_VULKAN_RUNTIME)
        target_compile_definitions(usdGenVulkan PRIVATE USDGEN_ENABLE_VULKAN_RUNTIME)
    endif()
    add_dependencies(usdGenWithPrimitives usdGenVulkan)
    list(APPEND USDGEN_DEFAULT_EXTENSION_TARGETS usdGenVulkan)
    # Primitive Vulkan targets (step 8.6a), each sourced by GLOB_RECURSE of
    # its folder so a git mv moves membership (reconfigure after one):
    # SHARED usdGenCurvesVulkan (usdGenCurves/vulkan/) and OBJECT
    # usdGenSurfaceVulkan, usdGenExprVulkan (usdGen{Surface,Expr}/vulkan/,
    # linked into usdGenCurvesVulkan, their only consumer). An OBJECT folder
    # without sources yet makes no target.
    file(GLOB_RECURSE _usdgen_curves_vulkan_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/usdGenCurves/usdGenCurves/vulkan/*.cpp")
    add_library(usdGenCurvesVulkan SHARED ${_usdgen_curves_vulkan_sources})
    target_link_libraries(usdGenCurvesVulkan
        PUBLIC usdGen
        PRIVATE usdGenVulkan usdGenSurface usdGenExpr usdGenCurves usdGenMath arch ${CMAKE_DL_LIBS})
    foreach(_usdgen_gpu_object usdGenSurface usdGenExpr)
        file(GLOB_RECURSE _usdgen_gpu_object_sources
            "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/primitives/${_usdgen_gpu_object}/${_usdgen_gpu_object}/vulkan/*.cpp")
        if(_usdgen_gpu_object_sources)
            add_library(${_usdgen_gpu_object}Vulkan OBJECT ${_usdgen_gpu_object_sources})
            target_link_libraries(${_usdgen_gpu_object}Vulkan PRIVATE usdGenVulkan ${_usdgen_gpu_object})
            set_target_properties(${_usdgen_gpu_object}Vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)
            target_sources(usdGenCurvesVulkan PRIVATE $<TARGET_OBJECTS:${_usdgen_gpu_object}Vulkan>)
        endif()
    endforeach()
    # RTLD_LOCAL dlopen finds libusdGenVulkan.so beside it in an install.
    set_target_properties(usdGenCurvesVulkan PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        INSTALL_RPATH "$ORIGIN")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
        target_link_options(usdGenCurvesVulkan PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
    add_dependencies(usdGenWithPrimitives usdGenCurvesVulkan)
    # Loaded after usdGenVulkan: its usdGenRegister names the "vulkan" backend.
    list(APPEND USDGEN_DEFAULT_EXTENSION_TARGETS usdGenCurvesVulkan)
    install(TARGETS usdGenCurvesVulkan
        RUNTIME DESTINATION "${USDGEN_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${USDGEN_INSTALL_LIBDIR}")
    # Test link lines name these; both are usdGenVulkan.
    add_library(usdGenVulkanNative INTERFACE)
    target_link_libraries(usdGenVulkanNative INTERFACE usdGenVulkan)
    add_library(usdGenVulkanFactory INTERFACE)
    target_link_libraries(usdGenVulkanFactory INTERFACE usdGenVulkan)
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
    function(_usdgen_compile_vulkan_shader _tag _file)
        set(_shader "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/shaders/${_file}.comp")
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
    _usdgen_compile_vulkan_shader(width width)
    _usdgen_compile_vulkan_shader(length lengthScale)
    _usdgen_compile_vulkan_shader(blend widthBlend)
    _usdgen_compile_vulkan_shader(compare nonWidthCompare)
    _usdgen_compile_vulkan_shader(compaction lengthCompaction)
    _usdgen_compile_vulkan_shader(set lengthSet)
    _usdgen_compile_vulkan_shader(topo topologyCompare)
    _usdgen_compile_vulkan_shader(po pointOverride)
    _usdgen_compile_vulkan_shader(grow curveGrow)
    _usdgen_compile_vulkan_shader(scatter_grow scatterGrow)
    _usdgen_compile_vulkan_shader(rfval rootFramesValidate)
    _usdgen_compile_vulkan_shader(rfbuild rootFrames)
    _usdgen_compile_vulkan_shader(rfgather rootFrameGather)
    _usdgen_compile_vulkan_shader(noise_validate noiseValidate)
    _usdgen_compile_vulkan_shader(noise_generate noiseGenerate)
    _usdgen_compile_vulkan_shader(deform_evaluate deformEvaluate)
    _usdgen_compile_vulkan_shader(deform_apply deformApply)
    _usdgen_compile_vulkan_shader(deform_evaluate_verify deformEvaluateVerify)
    _usdgen_compile_vulkan_shader(deform_evaluate_fill deformEvaluateFill)
    _usdgen_compile_vulkan_shader(deform_evaluate_cached deformEvaluateCached)
    _usdgen_compile_vulkan_shader(overlap widthOverlapWitness)
    _usdgen_compile_vulkan_shader(resample_vk resampleVk)
    _usdgen_compile_vulkan_shader(mapvk_sample mapVkSample)
    _usdgen_compile_vulkan_shader(rbfvk_extent rbfVkExtent)
    _usdgen_compile_vulkan_shader(rbfvk_gram rbfVkGram)
    _usdgen_compile_vulkan_shader(rbfvk_build_matrix rbfVkBuildMatrix)
    _usdgen_compile_vulkan_shader(rbfvk_lu rbfVkLu)
    _usdgen_compile_vulkan_shader(rbfvk_rhs rbfVkRhs)
    _usdgen_compile_vulkan_shader(rbfvk_trisolve rbfVkTriSolve)
    _usdgen_compile_vulkan_shader(rbfvk_evaluate rbfVkEvaluate)
    _usdgen_compile_vulkan_shader(rbfvk_apply rbfVkApply)
    _usdgen_compile_vulkan_shader(rbfvk_verify rbfVkVerify)
    _usdgen_compile_vulkan_shader(rbfvk_fill rbfVkFill)
    _usdgen_compile_vulkan_shader(rbfvk_evaluate_cached rbfVkEvaluateCached)
    _usdgen_compile_vulkan_shader(picktile_pick picktileVkPick)
    _usdgen_compile_vulkan_shader(picktile_footprint picktileVkFootprint)
    _usdgen_compile_vulkan_shader(picktile_tiles picktileVkTiles)
    _usdgen_compile_vulkan_shader(picktile_bounds picktileVkBounds)
    _usdgen_compile_vulkan_shader(picktile_indices picktileVkIndices)
    _usdgen_compile_vulkan_shader(exprvk_evaluate exprVkEvaluate)
    _usdgen_compile_vulkan_shader(exprvk_context exprVkContext)
    _usdgen_compile_vulkan_shader(exprvk_width exprVkWidth)
    _usdgen_compile_vulkan_shader(exprvk_exprop exprVkExprOp)
    _usdgen_compile_vulkan_shader(exprvk_pack exprVkPack)
    _usdgen_compile_vulkan_shader(cut_extend lengthCutExtend)
    _usdgen_compile_vulkan_shader(reparam lengthReparam)
    _usdgen_compile_vulkan_shader(minimum lengthMinimum)
    _usdgen_compile_vulkan_shader(literal_v1 lengthLiteralV1)
    _usdgen_compile_vulkan_shader(envelope_v1 lengthEnvelopeV1
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenVulkan/usdGenVulkan/vulkan/shaders/float32Envelope.glsl")
    _usdgen_compile_vulkan_shader(device_v1 lengthDeviceV1)
    _usdgen_compile_vulkan_shader(stylevk_ramp styleVkWidthRamp)
    _usdgen_compile_vulkan_shader(stylevk_length styleVkLength)
    _usdgen_compile_vulkan_shader(stylevk_grow styleVkGrow)
    add_custom_target(usdGenVulkanShaders ALL DEPENDS ${_usdgen_vulkan_spirv_files})
    add_dependencies(usdGenVulkan usdGenVulkanShaders)
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

# The extensions usdGen loads from its own directory when USDGEN_EXTENSIONS
# is unset, in load order: file names joined by '|' (registrar.cpp).
set(_usdgen_default_extensions "")
foreach(_t IN LISTS USDGEN_DEFAULT_EXTENSION_TARGETS)
    if(_usdgen_default_extensions STREQUAL "")
        set(_usdgen_default_extensions "$<TARGET_FILE_NAME:${_t}>")
    else()
        string(APPEND _usdgen_default_extensions "|$<TARGET_FILE_NAME:${_t}>")
    endif()
endforeach()
set_property(SOURCE libs/core/usdGen/usdGen/registries/registrar.cpp APPEND PROPERTY
    COMPILE_DEFINITIONS "USDGEN_DEFAULT_EXTENSIONS=\"${_usdgen_default_extensions}\"")
unset(_usdgen_default_extensions)
