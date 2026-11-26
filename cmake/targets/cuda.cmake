# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# CUDA toolchain and usdGenCuda. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# CUDA foundations and legacy optional GPU noise
# ---------------------------------------------------------------------------
# SeExpr2::FBM<3,1,false,float> (noise.cpp:181-277) gains a CUDA kernel
# (noise_gpu.cu, landed by a later step). Default OFF: without it no CUDA
# compiler, toolkit, or language is involved, so configure stays green on
# GPU-less CI hosts.
option(USE_GPU_NOISE
    "Build the CUDA noise kernel for usdGen noise ops (requires CUDA toolchain)" OFF)
option(USDGEN_ENABLE_CUDA
    "Build persistent CUDA buffers and RBF execution (requires CUDA toolkit)" OFF)
if(USE_GPU_NOISE OR USDGEN_ENABLE_CUDA)
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
    # The kernel lands at libs/core/primitives/usdGenCurves/usdGenCurves/ops/noise_gpu.cu (CUDA1b); wire
    # it into usdGen the moment the file exists (hook stays inert until then).
    if(USE_GPU_NOISE)
        target_sources(usdGenCurves PRIVATE libs/core/primitives/usdGenCurves/usdGenCurves/ops/noise_gpu.cu)
        set_target_properties(usdGenCurves PROPERTIES CUDA_ARCHITECTURES "${USDGEN_CUDA_ARCH}")
        # Enables the #ifdef USDGEN_USE_GPU_NOISE branch in noise.cpp (04 §2.9).
        target_compile_definitions(usdGenCurves PRIVATE USDGEN_USE_GPU_NOISE)
    endif()
endif()

# ---------------------------------------------------------------------------
# usdGenCuda (SHARED, step 8.3): the CUDA runtime (device selection, streams,
# the one block cache in gpu/deviceBuffer.cu, device resources, completion
# lane, retirement payloads), the CUDA Session provider and, until step 8.6,
# the curve kernels and executor. Built in every configuration like
# usdGenVulkan's kernel rows: without the toolkit it carries the CPU-side
# admission (capability matrix, ValidateCudaGraph) and registers "cuda" as
# unavailable. The registry loads it after the primitives
# (USDGEN_DEFAULT_EXTENSIONS). It is the only target that may compile
# deviceBuffer.cu (Gates.cmake, single block cache).
# ---------------------------------------------------------------------------
file(GLOB_RECURSE _usdgen_cuda_cpp_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/*.cpp")
set(_usdgen_cuda_runtime_only
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/cudaScatterInput.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/exec/provider.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/exec/workspace.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/gpu/cudaCompletionLane.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda/gpu/cudaFence.cpp")
if(NOT USDGEN_ENABLE_CUDA)
    list(REMOVE_ITEM _usdgen_cuda_cpp_sources ${_usdgen_cuda_runtime_only})
endif()
add_library(usdGenCuda SHARED ${_usdgen_cuda_cpp_sources})
# The primitives by name, not usdGenWithPrimitives: that target depends on
# this one so every consumer of the engine finds it built.
target_link_libraries(usdGenCuda
    PUBLIC usdGen
    PRIVATE usdGenSurface usdGenExpr usdGenCurves usdGenMath arch work trace ${CMAKE_DL_LIBS})
# Bare "gpu/X.h" spellings resolve against the backend folder.
target_include_directories(usdGenCuda PUBLIC
    ${USDGEN_PRIMITIVE_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libs/core/backends/usdGenCuda/usdGenCuda>
    $<INSTALL_INTERFACE:include>)
set_target_properties(usdGenCuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
    target_link_options(usdGenCuda PRIVATE "LINKER:--exclude-libs,ALL")
endif()
add_dependencies(usdGenWithPrimitives usdGenCuda)
list(APPEND USDGEN_DEFAULT_EXTENSION_TARGETS usdGenCuda)
if(USDGEN_ENABLE_CUDA)
    target_sources(usdGenCuda PRIVATE
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/deviceBuffer.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/deviceResources.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/imageSampler.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/rbf.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/curveGeometry.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/curveSource.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/scatterGrow.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/curveGrow.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/expressionContext.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/deformCurves.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/styleOps.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/width.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/widthBlend.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/widthOverlapWitness.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/surfaceBinding.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/length.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/curveCompaction.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/curveResample.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/namedChannelTopology.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/picking.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/pointOverride.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/topology.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/nonWidthCompare.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/curveIndices.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/curveTiles.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/curveTileBounds.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/rootFrames.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/rootFrameGather.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/cudaRetirement.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/noise.cu
        libs/core/backends/usdGenCuda/usdGenCuda/gpu/expression.cu)
    # cuda_driver backs the cudaStreamGetDevice shim in gpu/cudaCompat.h
    # on toolkits older than 12.8.
    target_link_libraries(usdGenCuda PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::cusolver)
    # The kernels still include pxr/ headers (step 8 removes them).
    target_include_directories(usdGenCuda SYSTEM PUBLIC
        $<TARGET_PROPERTY:tf,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(usdGenCuda PRIVATE USDGEN_ENABLE_CUDA)
    set_target_properties(usdGenCuda PROPERTIES CUDA_ARCHITECTURES "${USDGEN_CUDA_ARCH}")
    # The imaging interop reads channels through the neutral device owner
    # (UsdGenDeviceGeneration::ChannelAddress): the CUDA runtime API only,
    # no backend library. Pomade shares usdGenCuda's one block cache.
    target_compile_definitions(usdGenImaging PUBLIC USDGEN_HAS_CUDA_GL_INTEROP)
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
#   usdGenCurvesCuda (SHARED)  libs/core/primitives/usdGenCurves/usdGenCurves/cuda/
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
if(USDGEN_ENABLE_CUDA)
    _usdgen_glob_gpu_folder(_usdgen_curves_cuda_sources
        libs/core/primitives/usdGenCurves/usdGenCurves/cuda)
    add_library(usdGenCurvesCuda SHARED ${_usdgen_curves_cuda_sources})
    target_link_libraries(usdGenCurvesCuda
        PUBLIC usdGen
        PRIVATE usdGenCuda usdGenSurface usdGenExpr usdGenCurves usdGenMath arch ${CMAKE_DL_LIBS})
    foreach(_usdgen_gpu_object usdGenSurface usdGenExpr)
        _usdgen_glob_gpu_folder(_usdgen_gpu_object_sources
            libs/core/primitives/${_usdgen_gpu_object}/${_usdgen_gpu_object}/cuda)
        if(_usdgen_gpu_object_sources)
            add_library(${_usdgen_gpu_object}Cuda OBJECT ${_usdgen_gpu_object_sources})
            target_link_libraries(${_usdgen_gpu_object}Cuda PRIVATE usdGenCuda ${_usdgen_gpu_object})
            set_target_properties(${_usdgen_gpu_object}Cuda PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                CUDA_ARCHITECTURES "${USDGEN_CUDA_ARCH}")
            target_sources(usdGenCurvesCuda PRIVATE $<TARGET_OBJECTS:${_usdgen_gpu_object}Cuda>)
        endif()
    endforeach()
    set_target_properties(usdGenCurvesCuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "${USDGEN_CUDA_ARCH}"
        INSTALL_RPATH "$ORIGIN")
    add_dependencies(usdGenWithPrimitives usdGenCurvesCuda)
    # Loaded after usdGenCuda: its usdGenRegister names the "cuda" backend.
    list(APPEND USDGEN_DEFAULT_EXTENSION_TARGETS usdGenCurvesCuda)
    install(TARGETS usdGenCurvesCuda
        RUNTIME DESTINATION "${USDGEN_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${USDGEN_INSTALL_LIBDIR}")
endif()
