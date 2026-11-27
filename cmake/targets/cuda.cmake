# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# CUDA toolchain and usdGenCuda. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# CUDA foundations and legacy optional GPU noise
# ---------------------------------------------------------------------------
# UsdGenNoiseOp::Capture's CUDA path (usdGenCurves/cuda/noise_gpu.cu) is
# compiled into usdGenCurvesCuda; USE_GPU_NOISE makes its registration set
# the hook (ops/noiseGpuHook.h). Default OFF: Capture stays on the CPU.
option(USE_GPU_NOISE
    "Route usdGen noise capture through the CUDA kernel (requires USDGEN_ENABLE_CUDA)" OFF)
option(USDGEN_ENABLE_CUDA
    "Build persistent CUDA buffers and RBF execution (requires CUDA toolkit)" OFF)
if(USE_GPU_NOISE AND NOT USDGEN_ENABLE_CUDA)
    message(FATAL_ERROR "USE_GPU_NOISE requires USDGEN_ENABLE_CUDA (the kernel lives in usdGenCurvesCuda)")
endif()
if(USDGEN_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "usdGen CUDA requested but no CUDA compiler was found; "
            "install a CUDA toolkit or set CMAKE_CUDA_COMPILER")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Exact float op order vs the CPU FBM path (04 §2.9): no fused FMA.
    # 1394-D is nvcc's spelling of C4251 (see the MSVC warning block above).
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false -diag-suppress 1394")
    # nvcc establishes its own Visual Studio environment by re-running
    # vcvars on every invocation; nested inside an already-configured shell
    # that re-entry overflows cmd's line limit ("input line too long") and
    # every kernel compile dies before it starts. Kernel builds only ever
    # run where cl.exe already works, so hand nvcc that same environment
    # instead of a broken nested one (the arch probe below makes the same
    # choice per configure environment).
    if(MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use-local-env")
    endif()
    foreach(_config RELEASE RELWITHDEBINFO MINSIZEREL)
        string(REGEX REPLACE "[/-]DNDEBUG" "" CMAKE_CUDA_FLAGS_${_config}
            "${CMAKE_CUDA_FLAGS_${_config}}")
    endforeach()
    # Configure-time arch probe. The GPU in this host decides: nvidia-smi
    # reports its compute capability (12.1 on the GB10 dev host, 8.9 on an
    # Ada workstation), and that arch is tried first, so a checkout builds
    # kernels the local card can actually run. The GB10 pair stays on as the
    # fallback for a host with the toolkit but no queryable GPU. A trivial
    # kernel is compiled directly with nvcc per arch (-gencode); first
    # success wins, which also rejects an arch this toolkit is too old for.
    set(_usdgen_cuda_archs "")
    find_program(USDGEN_NVIDIA_SMI nvidia-smi)
    if(USDGEN_NVIDIA_SMI)
        execute_process(
            COMMAND "${USDGEN_NVIDIA_SMI}" --query-gpu=compute_cap
                    --format=csv,noheader
            OUTPUT_VARIABLE _usdgen_compute_caps
            RESULT_VARIABLE _usdgen_smi_rc
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_usdgen_smi_rc EQUAL 0)
            string(REPLACE "\n" ";" _usdgen_compute_caps "${_usdgen_compute_caps}")
            foreach(_cap IN LISTS _usdgen_compute_caps)
                # "8.9" -> 89. A driver too old to answer prints "[N/A]";
                # anything that is not major.minor is dropped, not fed to nvcc.
                if(_cap MATCHES "([0-9]+)\\.([0-9]+)")
                    list(APPEND _usdgen_cuda_archs
                        "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
                endif()
            endforeach()
        endif()
    endif()
    list(APPEND _usdgen_cuda_archs 121 120)
    list(REMOVE_DUPLICATES _usdgen_cuda_archs)
    file(WRITE "${CMAKE_BINARY_DIR}/usdgen_cuda_arch_probe.cu"
        "__global__ void usdgen_cuda_arch_probe() {}\n")
    set(USDGEN_CUDA_ARCH "")
    set(_usdgen_probe_local_env "")
    if(DEFINED ENV{VCToolsInstallDir})
        # Configured from a Visual Studio shell: nvcc must reuse this
        # environment instead of re-running vcvars into it (see above).
        # From a clean shell the flag stays off and nvcc sets Visual
        # Studio up itself.
        set(_usdgen_probe_local_env "--use-local-env")
    endif()
    foreach(_arch IN LISTS _usdgen_cuda_archs)
        execute_process(
            COMMAND "${CMAKE_CUDA_COMPILER}" ${_usdgen_probe_local_env} -c
                    "-gencode=arch=compute_${_arch},code=sm_${_arch}"
                    "-o" "${CMAKE_BINARY_DIR}/usdgen_cuda_arch_probe.o"
                    "${CMAKE_BINARY_DIR}/usdgen_cuda_arch_probe.cu"
            RESULT_VARIABLE _usdgen_cuda_arch_rc
            OUTPUT_VARIABLE _usdgen_cuda_arch_out
            ERROR_VARIABLE _usdgen_cuda_arch_err)
        if(_usdgen_cuda_arch_rc EQUAL 0)
            set(USDGEN_CUDA_ARCH "${_arch}")
            break()
        endif()
        # Say why: "input line is too long" here means nvcc's nested vcvars
        # overflowed cmd (an over-long PATH), not a toolkit problem.
        message(STATUS "usdGen: sm_${_arch} probe failed: "
            "${_usdgen_cuda_arch_out}${_usdgen_cuda_arch_err}")
    endforeach()
    if(NOT USDGEN_CUDA_ARCH)
        string(REPLACE ";" ", sm_" _usdgen_cuda_archs_tried "${_usdgen_cuda_archs}")
        message(FATAL_ERROR "usdGen CUDA requested but no architecture in "
            "(sm_${_usdgen_cuda_archs_tried}) compiles with "
            "${CMAKE_CUDA_COMPILER} (CUDA ${CUDAToolkit_VERSION}); install a "
            "toolkit new enough for this host's GPU, or set "
            "CMAKE_CUDA_COMPILER at the one that is. Last nvcc output: "
            "${_usdgen_cuda_arch_out}${_usdgen_cuda_arch_err}")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${USDGEN_CUDA_ARCH}" CACHE STRING
        "CUDA arch for the noise kernel (probed at configure time)" FORCE)
    # usdGen (line 117) predates the variable, so its empty default was
    # frozen at creation: pin the property explicitly.
    set_target_properties(usdGen PROPERTIES CUDA_ARCHITECTURES "${USDGEN_CUDA_ARCH}")
    message(STATUS "usdGen CUDA: ${CUDAToolkit_VERSION} (${CMAKE_CUDA_COMPILER}), arch sm_${USDGEN_CUDA_ARCH}")
endif()

# ---------------------------------------------------------------------------
# usdGenCuda (SHARED, step 8.3): the CUDA runtime (device selection, streams,
# the one block cache in gpu/deviceBuffer.cu, device resources, completion
# lane, retirement payloads) and the CUDA Session provider; the curves' kernels
# and executor live in usdGenCurvesCuda. Built in every configuration like
# usdGenVulkan's kernel rows: without the toolkit it carries the CPU-side
# admission (capability matrix, ValidateCudaGraph) and registers "cuda" as
# unavailable. The registry loads it after the primitives
# (usdGenExtensions.txt). It is the only target that may compile
# deviceBuffer.cu (Gates.cmake, single block cache).
# ---------------------------------------------------------------------------
file(GLOB_RECURSE _usdgen_cuda_cpp_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/*.cpp")
set(_usdgen_cuda_runtime_only
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/exec/provider.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/exec/workspace.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/gpu/cudaCompletionLane.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/gpu/cudaFence.cpp")
if(NOT USDGEN_ENABLE_CUDA)
    list(REMOVE_ITEM _usdgen_cuda_cpp_sources ${_usdgen_cuda_runtime_only})
endif()
add_library(usdGenCuda SHARED ${_usdgen_cuda_cpp_sources})
# usdGen and the CUDA SDK only (gate testUsdGenNoDisallowedLink_usdGenCuda).
target_link_libraries(usdGenCuda
    PUBLIC usdGen
    PRIVATE ${CMAKE_DL_LIBS})
# Bare "gpu/X.h" spellings resolve against the backend folder.
target_include_directories(usdGenCuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda>
    $<INSTALL_INTERFACE:include>)
set_target_properties(usdGenCuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
usdgen_hidden_library(usdGenCuda)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
    target_link_options(usdGenCuda PRIVATE "LINKER:--exclude-libs,ALL")
endif()
add_dependencies(usdGenWithPrimitives usdGenCuda)
list(APPEND USDGEN_EXTENSION_TARGETS usdGenCuda)
if(USDGEN_ENABLE_CUDA)
    target_sources(usdGenCuda PRIVATE
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/deviceBuffer.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/deviceResources.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/cudaRetirement.cu)
    # cuda_driver backs the cudaStreamGetDevice shim in gpu/cudaCompat.h
    # on toolkits older than 12.8.
    target_link_libraries(usdGenCuda PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::cusolver)
    target_compile_definitions(usdGenCuda PRIVATE USDGEN_ENABLE_CUDA)
    set_target_properties(usdGenCuda PROPERTIES CUDA_ARCHITECTURES "${USDGEN_CUDA_ARCH}")
    # The imaging interop reads channels through the neutral device owner
    # (UsdGenDeviceGeneration::ChannelAddress): the CUDA runtime API only,
    # no backend library. Pomade shares usdGenCuda's one block cache.
    target_compile_definitions(usdGenImaging PUBLIC USDGEN_HAS_CUDA_GL_INTEROP)
    # Storm curve index builder (step 9.4): in the adapter, CUDA-GL interop
    # builds only. Takes only the cudaCompat.h shim from usdGenCuda, no link.
    target_sources(usdGenHydraCurves PRIVATE
        libs/hosts/usdGenHydra/primitives/curves/usdGenHydraCurves/cuda/stormIndices.cu)
    target_compile_definitions(usdGenHydraCurves PUBLIC USDGEN_HAS_CUDA_GL_INTEROP)
    target_include_directories(usdGenHydraCurves PRIVATE
        $<TARGET_PROPERTY:usdGenCuda,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(usdGenHydraCurves PRIVATE CUDA::cudart CUDA::cuda_driver)
    set_target_properties(usdGenHydraCurves PROPERTIES CUDA_ARCHITECTURES "${USDGEN_CUDA_ARCH}")
    target_link_libraries(usdGenImaging PUBLIC hdSt PRIVATE CUDA::cudart hgiGL)
    # Pomade CUDA lane (plan/17 P0): the tessellation kernel joins
    # usdGenPomade, and the model/transport compile their device mirrors.
    target_sources(usdGenPomade PRIVATE
        libs/usdGenPomade/usdGenPomade/pomadeKernels.cu)
    target_compile_definitions(usdGenPomade PUBLIC USDGEN_POMADE_HAS_CUDA)
    target_link_libraries(usdGenPomade PRIVATE usdGenCuda CUDA::cudart)
    set_target_properties(usdGenPomade PROPERTIES
        CUDA_ARCHITECTURES "${USDGEN_CUDA_ARCH}")
endif()

# ---------------------------------------------------------------------------
# Primitive CUDA targets (step 8.6a), each sourced by GLOB_RECURSE of its
# folder so a git mv moves membership (reconfigure after one):
#   usdGenCurvesCuda (SHARED, every configuration)  libs/core/primitives/usdGenCurves/usdGenCurves/cuda/
#   usdGenSurfaceCuda, usdGenExprCuda (OBJECT, linked into usdGenCurvesCuda,
#   their only consumer)       usdGen{Surface,Expr}/usdGen{Surface,Expr}/cuda/
# An OBJECT folder without sources yet makes no target. CUDA builds only.
# ---------------------------------------------------------------------------
function(_usdgen_glob_gpu_folder _out _folder)
    file(GLOB_RECURSE _found
        "${CMAKE_CURRENT_SOURCE_DIR}/${_folder}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/${_folder}/*.cu")
    set(${_out} "${_found}" PARENT_SCOPE)
endfunction()
# The curves' CUDA device program and kernels. Built in every configuration
# like usdGenCuda: without the toolkit it carries the CPU-side admission and
# the capability-matrix rows; the .cu kernels and the scatter input (CUDA
# runtime API) join only with USDGEN_ENABLE_CUDA.
_usdgen_glob_gpu_folder(_usdgen_curves_cuda_sources
    libs/core/primitives/usdGenCurves/usdGenCurves/cuda)
if(NOT USDGEN_ENABLE_CUDA)
    list(FILTER _usdgen_curves_cuda_sources EXCLUDE REGEX "[.]cu$|/cudaScatterInput[.]cpp$")
endif()
add_library(usdGenCurvesCuda SHARED ${_usdgen_curves_cuda_sources})
target_link_libraries(usdGenCurvesCuda
    PUBLIC usdGenCuda usdGen
    PRIVATE usdGenSurface usdGenExpr usdGenCurves usdGenMath ${CMAKE_DL_LIBS})
target_include_directories(usdGenCurvesCuda PUBLIC
    ${USDGEN_PRIMITIVE_INCLUDE_DIRS}
    "$<INSTALL_INTERFACE:include>")
set_target_properties(usdGenCurvesCuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    INSTALL_RPATH "$ORIGIN")
usdgen_hidden_library(usdGenCurvesCuda)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
    target_link_options(usdGenCurvesCuda PRIVATE "LINKER:--exclude-libs,ALL")
endif()
set(USDGEN_SURFACE_EXPR_CUDA_HOST usdGenCurvesCuda)
if(USDGEN_ENABLE_CUDA)
    target_compile_definitions(usdGenCurvesCuda PRIVATE USDGEN_ENABLE_CUDA)
    if(USE_GPU_NOISE)
        target_compile_definitions(usdGenCurvesCuda PRIVATE USDGEN_USE_GPU_NOISE)
    endif()
    set_target_properties(usdGenCurvesCuda PROPERTIES CUDA_ARCHITECTURES "${USDGEN_CUDA_ARCH}")
    foreach(_usdgen_gpu_object usdGenSurface usdGenExpr)
        _usdgen_glob_gpu_folder(_usdgen_gpu_object_sources
            libs/core/primitives/${_usdgen_gpu_object}/${_usdgen_gpu_object}/cuda)
        if(_usdgen_gpu_object_sources)
            add_library(${_usdgen_gpu_object}Cuda OBJECT ${_usdgen_gpu_object_sources})
            target_link_libraries(${_usdgen_gpu_object}Cuda PRIVATE
                usdGen ${_usdgen_gpu_object} CUDA::cudart)
            target_include_directories(${_usdgen_gpu_object}Cuda PRIVATE
                ${USDGEN_PRIMITIVE_INCLUDE_DIRS}
                $<TARGET_PROPERTY:usdGenCuda,INTERFACE_INCLUDE_DIRECTORIES>)
            # Linked into usdGenCurvesCuda: its symbols export with that DLL's macro.
            target_compile_definitions(${_usdgen_gpu_object}Cuda PRIVATE USDGEN_ENABLE_CUDA usdGenCurvesCuda_EXPORTS)
            set_target_properties(${_usdgen_gpu_object}Cuda PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                CXX_VISIBILITY_PRESET hidden
                CUDA_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON
                CUDA_ARCHITECTURES "${USDGEN_CUDA_ARCH}")
            # The kernels' callers live in usdGenCurvesCuda (8.6b), so the
            # objects link there.
            target_sources(${USDGEN_SURFACE_EXPR_CUDA_HOST} PRIVATE $<TARGET_OBJECTS:${_usdgen_gpu_object}Cuda>)
        endif()
    endforeach()
endif()
add_dependencies(usdGenWithPrimitives usdGenCurvesCuda)
# Loaded after usdGenCuda: its usdGenRegister names the "cuda" backend.
list(APPEND USDGEN_EXTENSION_TARGETS usdGenCurvesCuda)
install(TARGETS usdGenCurvesCuda
    RUNTIME DESTINATION "${USDGEN_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${USDGEN_INSTALL_LIBDIR}")
