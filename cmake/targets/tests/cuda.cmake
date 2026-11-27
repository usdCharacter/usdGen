# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Tests: CUDA backend. Included from the root CMakeLists.txt inside if(USDGEN_BUILD_TESTS).

if(USDGEN_ENABLE_CUDA)
    add_executable(testUsdGenCudaCurveGrowFailure tests/internal/usdGenCurves/testUsdGenCudaCurveGrowFailure.cpp)
    target_compile_definitions(testUsdGenCudaCurveGrowFailure PRIVATE USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenCudaCurveGrowFailure PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaCurveGrowFailure COMMAND testUsdGenCudaCurveGrowFailure)
    add_test(NAME testUsdGenCudaCurveGrowNamedFailure COMMAND testUsdGenCudaCurveGrowFailure --named)
    add_test(NAME testUsdGenCudaCurveGrowNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --native)
    add_test(NAME testUsdGenCudaCurveGrowNamedNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --named-native)
    add_test(NAME testUsdGenCudaCurveGrowSourceTerminalFailure COMMAND testUsdGenCudaCurveGrowFailure --install --source-terminal)
    add_test(NAME testUsdGenCudaCurveGrowSourceTerminalNamedFailure COMMAND testUsdGenCudaCurveGrowFailure --named --source-terminal)
    add_test(NAME testUsdGenCudaCurveGrowSourceTerminalNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --native --source-terminal)
    add_test(NAME testUsdGenCudaCurveGrowSourceTerminalNamedNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --named-native --source-terminal)
    add_test(NAME testUsdGenCudaGrowLengthFailure COMMAND testUsdGenCudaCurveGrowFailure --install --length)
    add_test(NAME testUsdGenCudaGrowLengthCountsFailure COMMAND testUsdGenCudaCurveGrowFailure --named --length)
    add_test(NAME testUsdGenCudaGrowLengthNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --native --length)
    add_test(NAME testUsdGenCudaGrowLengthCountsNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --named-native --length)
    add_test(NAME testUsdGenCudaMixedGrowLengthFailure COMMAND testUsdGenCudaCurveGrowFailure --install --mixed)
    add_test(NAME testUsdGenCudaMixedGrowLengthCountsFailure COMMAND testUsdGenCudaCurveGrowFailure --named --mixed)
    add_test(NAME testUsdGenCudaMixedGrowLengthNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --native --mixed)
    add_test(NAME testUsdGenCudaMixedGrowLengthCountsNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --named-native --mixed)
    add_test(NAME testUsdGenCudaMultiGrowFailure COMMAND testUsdGenCudaCurveGrowFailure --install --multi-grow)
    add_test(NAME testUsdGenCudaMultiGrowNamedFailure COMMAND testUsdGenCudaCurveGrowFailure --named --multi-grow)
    add_test(NAME testUsdGenCudaMultiGrowNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --native --multi-grow)
    add_test(NAME testUsdGenCudaMultiGrowNamedNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --named-native --multi-grow)
    add_test(NAME testUsdGenCudaNoiseGrowFailure COMMAND testUsdGenCudaCurveGrowFailure --install --noise-grow)
    add_test(NAME testUsdGenCudaNoiseGrowNamedFailure COMMAND testUsdGenCudaCurveGrowFailure --named --noise-grow)
    add_test(NAME testUsdGenCudaNoiseGrowNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --native --noise-grow)
    add_test(NAME testUsdGenCudaNoiseGrowNamedNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --named-native --noise-grow)
    add_test(NAME testUsdGenCudaCulledNoiseGrowFailure COMMAND testUsdGenCudaCurveGrowFailure --install --culled-noise-grow)
    add_test(NAME testUsdGenCudaCulledNoiseGrowNamedFailure COMMAND testUsdGenCudaCurveGrowFailure --named --culled-noise-grow)
    add_test(NAME testUsdGenCudaCulledNoiseGrowNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --native --culled-noise-grow)
    add_test(NAME testUsdGenCudaCulledNoiseGrowNamedNativeFailure COMMAND testUsdGenCudaCurveGrowFailure --named-native --culled-noise-grow)
    set_tests_properties(testUsdGenCudaCurveGrowFailure testUsdGenCudaCurveGrowNamedFailure
        testUsdGenCudaCurveGrowNativeFailure testUsdGenCudaCurveGrowNamedNativeFailure
        testUsdGenCudaCurveGrowSourceTerminalFailure testUsdGenCudaCurveGrowSourceTerminalNamedFailure
        testUsdGenCudaCurveGrowSourceTerminalNativeFailure testUsdGenCudaCurveGrowSourceTerminalNamedNativeFailure
        testUsdGenCudaGrowLengthFailure testUsdGenCudaGrowLengthCountsFailure
        testUsdGenCudaGrowLengthNativeFailure testUsdGenCudaGrowLengthCountsNativeFailure
        testUsdGenCudaMixedGrowLengthFailure testUsdGenCudaMixedGrowLengthCountsFailure
        testUsdGenCudaMixedGrowLengthNativeFailure testUsdGenCudaMixedGrowLengthCountsNativeFailure
        testUsdGenCudaMultiGrowFailure testUsdGenCudaMultiGrowNamedFailure
        testUsdGenCudaMultiGrowNativeFailure testUsdGenCudaMultiGrowNamedNativeFailure
        testUsdGenCudaNoiseGrowFailure testUsdGenCudaNoiseGrowNamedFailure
        testUsdGenCudaNoiseGrowNativeFailure testUsdGenCudaNoiseGrowNamedNativeFailure
        testUsdGenCudaCulledNoiseGrowFailure testUsdGenCudaCulledNoiseGrowNamedFailure
        testUsdGenCudaCulledNoiseGrowNativeFailure testUsdGenCudaCulledNoiseGrowNamedNativeFailure PROPERTIES
        LABELS "T0;cuda;topology;curve-grow;cow;quarantine" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaCurveGrowSession tests/internal/usdGen/testUsdGenCudaCurveGrowSession.cpp)
    target_compile_definitions(testUsdGenCudaCurveGrowSession PRIVATE USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenCudaCurveGrowSession PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda usdGenMath usdGen_seexpr)
    add_test(NAME testUsdGenCudaCurveGrowSession COMMAND testUsdGenCudaCurveGrowSession)
    set_tests_properties(testUsdGenCudaCurveGrowSession PROPERTIES
        LABELS "T0;cuda;topology;curve-grow;cow;session" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaCurveGrowGeneration tests/internal/usdGenCurves/testUsdGenCudaCurveGrowGeneration.cpp)
    target_link_libraries(testUsdGenCudaCurveGrowGeneration PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaCurveGrowGeneration COMMAND testUsdGenCudaCurveGrowGeneration)
    set_tests_properties(testUsdGenCudaCurveGrowGeneration PROPERTIES
        LABELS "T0;cuda;topology;curve-grow;cow;generation" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaCurveGrow tests/internal/usdGenCurves/testUsdGenCudaCurveGrow.cu)
    target_link_libraries(testUsdGenCudaCurveGrow PRIVATE usdGenCurvesCuda usdGenWithPrimitives usdGenUsd)
    target_include_directories(testUsdGenCudaCurveGrow PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/base/usdGenMath")
    add_test(NAME testUsdGenCudaCurveGrow COMMAND testUsdGenCudaCurveGrow)
    add_test(NAME testUsdGenCudaCurveGrowQuarantine COMMAND testUsdGenCudaCurveGrow --quarantine)
    set_tests_properties(testUsdGenCudaCurveGrowQuarantine PROPERTIES
        LABELS "T0;cuda;topology;curve-grow;quarantine;cow" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    set_tests_properties(testUsdGenCudaCurveGrow PROPERTIES
        LABELS "T0;cuda;topology;curve-grow;cow" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaScatterGrow tests/internal/usdGenCurves/testUsdGenCudaScatterGrow.cu)
    target_link_libraries(testUsdGenCudaScatterGrow PRIVATE usdGenCurvesCuda usdGenWithPrimitives usdGenUsd)
    target_include_directories(testUsdGenCudaScatterGrow PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/base/usdGenMath")
    add_test(NAME testUsdGenCudaScatterGrow COMMAND testUsdGenCudaScatterGrow)
    set_tests_properties(testUsdGenCudaScatterGrow PROPERTIES
        LABELS "T0;cuda;topology;scatter-grow" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaScatterGrowPreflight tests/internal/usdGenCurves/testUsdGenCudaScatterGrowPreflight.cu)
    target_link_libraries(testUsdGenCudaScatterGrowPreflight PRIVATE usdGenCurvesCuda usdGenWithPrimitives usdGenUsd)
    target_include_directories(testUsdGenCudaScatterGrowPreflight PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/base/usdGenMath")
    add_test(NAME testUsdGenCudaScatterGrowPreflight COMMAND testUsdGenCudaScatterGrowPreflight)
    add_test(NAME testUsdGenCudaScatterGrowHostQuarantine
        COMMAND testUsdGenCudaScatterGrowPreflight --quarantine-host)
    set_tests_properties(testUsdGenCudaScatterGrowHostQuarantine PROPERTIES
        LABELS "T0;cuda;topology;scatter-grow;quarantine" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    set_tests_properties(testUsdGenCudaScatterGrowPreflight PROPERTIES
        LABELS "T0;cuda;topology;scatter-grow" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    # ValidateRoots is CPU-only: no device, no SKIP, no GPU lock.
    add_executable(testUsdGenCudaScatterGrowValidate tests/internal/usdGenCurves/testUsdGenCudaScatterGrowValidate.cpp)
    target_link_libraries(testUsdGenCudaScatterGrowValidate PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaScatterGrowValidate COMMAND testUsdGenCudaScatterGrowValidate)
    set_tests_properties(testUsdGenCudaScatterGrowValidate PROPERTIES
        LABELS "T0;cuda;scatter-grow;validate" TIMEOUT 120)
    add_executable(testUsdGenCudaScatterGrowOverlap tests/internal/usdGenCurves/testUsdGenCudaScatterGrowOverlap.cpp)
    target_link_libraries(testUsdGenCudaScatterGrowOverlap PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaScatterGrowOverlap COMMAND testUsdGenCudaScatterGrowOverlap)
    set_tests_properties(testUsdGenCudaScatterGrowOverlap PROPERTIES
        LABELS "T0;cuda;topology;scatter-grow" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaScatterGrowOverlapFault tests/internal/usdGenCurves/testUsdGenCudaScatterGrowOverlapFault.cpp)
    target_link_libraries(testUsdGenCudaScatterGrowOverlapFault PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaScatterGrowOverlapFault COMMAND testUsdGenCudaScatterGrowOverlapFault)
    set_tests_properties(testUsdGenCudaScatterGrowOverlapFault PROPERTIES
        LABELS "T0;cuda;topology;scatter-grow" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaScatterInput tests/internal/usdGenCurves/testUsdGenCudaScatterInput.cpp)
    add_executable(testUsdGenCudaScatterGrowSession tests/internal/usdGen/testUsdGenCudaScatterGrowSession.cpp)
    add_executable(testUsdGenCudaScatterGrowWidthDag tests/internal/usdGen/testUsdGenCudaScatterGrowWidthDag.cpp)
    add_executable(testUsdGenCudaScatterGrowValueDag tests/internal/usdGenCurves/testUsdGenCudaScatterGrowValueDag.cpp)
    add_executable(testUsdGenCudaCpuSourceParity tests/internal/usdGen/testUsdGenCpuCurveSource.cpp)
    add_executable(testUsdGenCudaSurfaceBindingCache tests/internal/usdGenCurves/testUsdGenSurfaceBindingCache.cpp)
    target_compile_definitions(testUsdGenCudaSurfaceBindingCache PRIVATE USDGEN_TEST_CUDA_BINDING_CACHE)
    target_link_libraries(testUsdGenCudaSurfaceBindingCache PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaSurfaceBindingCache COMMAND testUsdGenCudaSurfaceBindingCache)
    set_tests_properties(testUsdGenCudaSurfaceBindingCache PROPERTIES
        LABELS "T1;cuda;sources;cache;cow" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaSurfaceRootFrameParity tests/internal/usdGenSurface/testUsdGenSurfaceRootFrames.cpp)
    target_compile_definitions(testUsdGenCudaSurfaceRootFrameParity PRIVATE USDGEN_TEST_CUDA_ROOT_FRAME_PARITY)
    target_link_libraries(testUsdGenCudaSurfaceRootFrameParity PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaSurfaceRootFrameParity COMMAND testUsdGenCudaSurfaceRootFrameParity)
    set_tests_properties(testUsdGenCudaSurfaceRootFrameParity PROPERTIES
        LABELS "T0;cuda;sources;root-frames" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    target_compile_definitions(testUsdGenCudaCpuSourceParity PRIVATE USDGEN_TEST_CUDA_SOURCE_PARITY)
    target_link_libraries(testUsdGenCudaCpuSourceParity PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaCpuSourceParity COMMAND testUsdGenCudaCpuSourceParity)
    set_tests_properties(testUsdGenCudaCpuSourceParity PROPERTIES
        LABELS "T1;cuda;sources;cow" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    target_link_libraries(testUsdGenCudaScatterGrowWidthDag PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaScatterGrowWidthDag COMMAND testUsdGenCudaScatterGrowWidthDag)
    set_tests_properties(testUsdGenCudaScatterGrowWidthDag PROPERTIES
        LABELS "T1;cuda;scatter-grow;dag;cow" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    target_link_libraries(testUsdGenCudaScatterGrowValueDag PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaScatterGrowValueDag COMMAND testUsdGenCudaScatterGrowValueDag)
    set_tests_properties(testUsdGenCudaScatterGrowValueDag PROPERTIES
        LABELS "T1;cuda;scatter-grow;dag;cow" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_test(NAME testUsdGenCudaScatterGrowRbfDag COMMAND testUsdGenCudaScatterGrowValueDag --rbf)
    set_tests_properties(testUsdGenCudaScatterGrowRbfDag PROPERTIES
        LABELS "T1;cuda;scatter-grow;dag;cow;rbf;transaction" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    target_link_libraries(testUsdGenCudaScatterGrowSession PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaScatterGrowSession COMMAND testUsdGenCudaScatterGrowSession)
    add_test(NAME testUsdGenCudaScatterGrowSourceInstallFailure
        COMMAND testUsdGenCudaScatterGrowSession --source-install-failure)
    add_test(NAME testUsdGenCudaScatterGrowSourceNativeFailure
        COMMAND testUsdGenCudaScatterGrowSession --source-native-failure)
    set_tests_properties(testUsdGenCudaScatterGrowSourceInstallFailure
        testUsdGenCudaScatterGrowSourceNativeFailure PROPERTIES
        LABELS "T1;cuda;scatter-grow;quarantine" TIMEOUT 30 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    set_tests_properties(testUsdGenCudaScatterGrowSession PROPERTIES
        LABELS "T1;cuda;scatter-grow;session;cow" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    target_link_libraries(testUsdGenCudaScatterInput PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaScatterInput COMMAND testUsdGenCudaScatterInput)
    set_tests_properties(testUsdGenCudaScatterInput PROPERTIES
        LABELS "T0;cuda;topology;scatter-input" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaImageSampler tests/internal/usdGenSurface/testUsdGenCudaImageSampler.cu)
    target_link_libraries(testUsdGenCudaImageSampler PRIVATE usdGenCurvesCuda usdGenWithPrimitives usdGenUsd)
    add_test(NAME testUsdGenCudaImageSampler COMMAND testUsdGenCudaImageSampler)
    set_tests_properties(testUsdGenCudaImageSampler PROPERTIES
        LABELS "T0;cuda;image;sampler" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(benchUsdGenCudaCook tests/internal/usdGenCurves/benchUsdGenCudaCook.cpp)
    target_link_libraries(benchUsdGenCudaCook PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME benchUsdGenCudaCook COMMAND benchUsdGenCudaCook --runs 30)
    set_tests_properties(benchUsdGenCudaCook PROPERTIES
        LABELS "perf;cuda" TIMEOUT 300 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaCompletionLane tests/internal/usdGenCuda/testUsdGenCudaCompletionLane.cpp)
    target_link_libraries(testUsdGenCudaCompletionLane PRIVATE usdGenWithPrimitives usdGenUsd usdGenCuda)
    add_test(NAME testUsdGenCudaCompletionLane COMMAND testUsdGenCudaCompletionLane)
    set_tests_properties(testUsdGenCudaCompletionLane PROPERTIES
        LABELS "T1;cuda;threading" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaExecutionResources tests/internal/usdGenCuda/testUsdGenCudaExecutionResources.cpp)
    target_link_libraries(testUsdGenCudaExecutionResources PRIVATE usdGenCuda usdGen)
    add_test(NAME testUsdGenCudaExecutionResources COMMAND testUsdGenCudaExecutionResources)
    set_tests_properties(testUsdGenCudaExecutionResources PROPERTIES
        LABELS "T0;cuda;resources" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaDeviceBufferClasses tests/internal/usdGenCuda/testUsdGenCudaDeviceBufferClasses.cpp)
    target_link_libraries(testUsdGenCudaDeviceBufferClasses PRIVATE usdGenCuda usdGen)
    add_test(NAME testUsdGenCudaDeviceBufferClasses COMMAND testUsdGenCudaDeviceBufferClasses)
    set_tests_properties(testUsdGenCudaDeviceBufferClasses PROPERTIES
        LABELS "T1;cuda;threading" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaDeviceBufferPoolStress tests/internal/usdGenCuda/testUsdGenCudaDeviceBufferPoolStress.cpp)
    target_link_libraries(testUsdGenCudaDeviceBufferPoolStress PRIVATE usdGenCuda usdGen)
    add_test(NAME testUsdGenCudaDeviceBufferPoolStress COMMAND testUsdGenCudaDeviceBufferPoolStress)
    set_tests_properties(testUsdGenCudaDeviceBufferPoolStress PROPERTIES
        LABELS "T0;cuda;resources" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaRetirement tests/internal/usdGenCuda/testUsdGenCudaRetirement.cpp)
    target_link_libraries(testUsdGenCudaRetirement PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaRetirement COMMAND testUsdGenCudaRetirement)
    set_tests_properties(testUsdGenCudaRetirement PROPERTIES
        LABELS "T0;cuda;resources;retirement" TIMEOUT 30 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_test(NAME testUsdGenCudaRetirementExit COMMAND testUsdGenCudaRetirement --exit-retire)
    set_tests_properties(testUsdGenCudaRetirementExit PROPERTIES
        LABELS "T1;cuda;resources;retirement;process-exit" TIMEOUT 30 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    # The optional oracle uses the actual native Storm index builder. Its
    # private header is not installed with the SDK; never add this source-tree
    # dependency to production targets or installed interfaces.
    set(USDGEN_OPENUSD_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../OpenUSD" CACHE PATH
        "Matching OpenUSD source tree for native Storm index oracle tests")
    add_executable(testUsdGenCudaCurveIndices tests/internal/usdGenCurves/testUsdGenCudaCurveIndices.cpp)
    target_link_libraries(testUsdGenCudaCurveIndices PRIVATE usdGenHydraCurves usdGenCurvesCuda usdGen hdSt)
    if(EXISTS "${USDGEN_OPENUSD_SOURCE_DIR}/pxr/imaging/hdSt/basisCurvesTopology.h")
        target_include_directories(testUsdGenCudaCurveIndices SYSTEM PRIVATE "${USDGEN_OPENUSD_SOURCE_DIR}")
        target_compile_definitions(testUsdGenCudaCurveIndices PRIVATE USDGEN_HAS_STORM_INDEX_ORACLE)
    endif()
    add_test(NAME testUsdGenCudaCurveIndices COMMAND testUsdGenCudaCurveIndices)
    add_test(NAME testUsdGenCudaCurveIndicesStorm COMMAND testUsdGenCudaCurveIndices --native)
    set_tests_properties(testUsdGenCudaCurveIndices testUsdGenCudaCurveIndicesStorm PROPERTIES
        LABELS "T1;cuda;topology;interop" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    set_tests_properties(testUsdGenCudaCurveIndicesStorm PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(testUsdGenCudaCurveTiles tests/internal/usdGenCurves/testUsdGenCudaCurveTiles.cpp)
    target_link_libraries(testUsdGenCudaCurveTiles PRIVATE usdGenCurvesCuda usdGen)
    add_test(NAME testUsdGenCudaCurveTiles COMMAND testUsdGenCudaCurveTiles)
    set_tests_properties(testUsdGenCudaCurveTiles PROPERTIES
        LABELS "T1;cuda;topology;tiles" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaCurveResample tests/internal/usdGenCurves/testUsdGenCudaCurveResample.cpp)
    target_link_libraries(testUsdGenCudaCurveResample PRIVATE usdGenCurvesCuda usdGen)
    add_test(NAME testUsdGenCudaCurveResample COMMAND testUsdGenCudaCurveResample)
    set_tests_properties(testUsdGenCudaCurveResample PROPERTIES
        LABELS "T1;cuda;topology;resample" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaCurveTileBounds tests/internal/usdGenCurves/testUsdGenCudaCurveTileBounds.cpp)
    target_link_libraries(testUsdGenCudaCurveTileBounds PRIVATE usdGenCurvesCuda usdGen)
    add_test(NAME testUsdGenCudaCurveTileBounds COMMAND testUsdGenCudaCurveTileBounds)
    set_tests_properties(testUsdGenCudaCurveTileBounds PROPERTIES
        LABELS "T1;cuda;topology;tiles" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaTileLease tests/internal/usdGenCurves/testUsdGenCudaTileLease.cpp)
    target_compile_definitions(testUsdGenCudaTileLease PRIVATE USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenCudaTileLease PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaTileLease COMMAND testUsdGenCudaTileLease)
    set_tests_properties(testUsdGenCudaTileLease PROPERTIES
        LABELS "T1;cuda;tilelease" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaTileExecution tests/internal/usdGenCurves/testUsdGenCudaTileExecution.cpp)
    target_compile_definitions(testUsdGenCudaTileExecution PRIVATE USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenCudaTileExecution PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    target_link_libraries(testUsdGenCudaTileExecution PRIVATE usdGenTestUtils usdGenHydraCurves)
    add_test(NAME testUsdGenCudaTileExecution COMMAND testUsdGenCudaTileExecution)
    set_tests_properties(testUsdGenCudaTileExecution PROPERTIES
        LABELS "T1;cuda;tiles;sessions" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaDevicePresentation tests/internal/usdGenCurves/testUsdGenCudaDevicePresentation.cpp)
    target_compile_definitions(testUsdGenCudaDevicePresentation PRIVATE USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenCudaDevicePresentation PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    target_link_libraries(testUsdGenCudaDevicePresentation PRIVATE usdGenTestUtils usdGenHydraCurves)
    add_test(NAME testUsdGenCudaDevicePresentation COMMAND testUsdGenCudaDevicePresentation)
    set_tests_properties(testUsdGenCudaDevicePresentation PROPERTIES
        LABELS "T1;cuda;sessions" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    if(NOT WIN32)
        add_executable(testUsdGenCudaGlInterop tests/internal/usdGenCurves/testUsdGenCudaGlInterop.cpp)
        target_compile_definitions(testUsdGenCudaGlInterop PRIVATE USDGEN_ENABLE_CUDA)
        target_include_directories(testUsdGenCudaGlInterop PRIVATE ${PROJECT_SOURCE_DIR}/libs/hosts/usdGenHydra/primitives/curves/shaders/test)
        target_link_libraries(testUsdGenCudaGlInterop PRIVATE usdGenImaging usdGenCuda hdSt hgiGL garch ${CMAKE_DL_LIBS})
    endif()
    usdgen_add_unix_only_test(testUsdGenCudaGlInterop testUsdGenCudaGlInterop)
    set_tests_properties(testUsdGenCudaGlInterop PROPERTIES LABELS "T2;cuda;interop" RUN_SERIAL TRUE SKIP_RETURN_CODE 77 TIMEOUT 60)
    add_executable(testUsdGenCudaExecutionQueue tests/internal/usdGenCurves/testUsdGenCudaExecutionQueue.cpp)
    target_link_libraries(testUsdGenCudaExecutionQueue PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaExecutionQueue COMMAND testUsdGenCudaExecutionQueue)
    set_tests_properties(testUsdGenCudaExecutionQueue PROPERTIES LABELS "T1;cuda;scheduling;rbf" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaExecutionPlan tests/internal/usdGenCurves/testUsdGenCudaExecutionPlan.cpp)
    target_link_libraries(testUsdGenCudaExecutionPlan PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaExecutionPlan COMMAND testUsdGenCudaExecutionPlan)
    set_tests_properties(testUsdGenCudaExecutionPlan PROPERTIES
        LABELS "T0;cuda;scheduling;metadata" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaWidthDag tests/internal/usdGen/testUsdGenCudaWidthDag.cpp)
    target_compile_definitions(testUsdGenCudaWidthDag PRIVATE USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenCudaWidthDag PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaWidthDag COMMAND testUsdGenCudaWidthDag)
    set_tests_properties(testUsdGenCudaWidthDag PROPERTIES
        LABELS "T1;cuda;scheduling;dag" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaNamedChannels tests/internal/usdGenCuda/testUsdGenCudaNamedChannels.cpp)
    target_compile_definitions(testUsdGenCudaNamedChannels PRIVATE USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenCudaNamedChannels PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaNamedChannels COMMAND testUsdGenCudaNamedChannels)
    set_tests_properties(testUsdGenCudaNamedChannels PROPERTIES
        LABELS "T1;cuda;resources;cow" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaNamedChannelTopology tests/internal/usdGenCurves/testUsdGenCudaNamedChannelTopology.cu)
    target_link_libraries(testUsdGenCudaNamedChannelTopology PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaNamedChannelTopology COMMAND testUsdGenCudaNamedChannelTopology)
    set_tests_properties(testUsdGenCudaNamedChannelTopology PROPERTIES
        LABELS "T1;cuda;topology;cow" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenCudaRbfTransaction tests/internal/usdGenCurves/testUsdGenCudaRbfTransaction.cpp)
    target_compile_definitions(testUsdGenCudaRbfTransaction PRIVATE USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenCudaRbfTransaction PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaRbfTransaction COMMAND testUsdGenCudaRbfTransaction)
    add_test(NAME testUsdGenCudaRbfValueDag COMMAND testUsdGenCudaRbfTransaction --value-dag)
    set_tests_properties(testUsdGenCudaRbfValueDag PROPERTIES
        LABELS "T1;cuda;rbf;transaction;dag;cow" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    set_tests_properties(testUsdGenCudaRbfTransaction PROPERTIES
        LABELS "T1;cuda;rbf;transaction" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaExecutionStages tests/internal/usdGenCurves/testUsdGenCudaExecutionStages.cpp)
    target_compile_definitions(testUsdGenCudaExecutionStages PRIVATE USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenCudaExecutionStages PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaExecutionStages COMMAND testUsdGenCudaExecutionStages)
    set_tests_properties(testUsdGenCudaExecutionStages PROPERTIES LABELS "T1;cuda;scheduling;rbf" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaTools tests/internal/usdGenCurves/testUsdGenCudaTools.cpp)
    target_link_libraries(testUsdGenCudaTools PRIVATE usdGenImaging usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    target_link_libraries(testUsdGenCudaTools PRIVATE usdGenTestUtils usdGenHydraCurves)
    add_test(NAME testUsdGenCudaTools COMMAND testUsdGenCudaTools)
    set_tests_properties(testUsdGenCudaTools PROPERTIES LABELS "T1;cuda;tools;sessions" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaParameters tests/internal/usdGenCurves/testUsdGenCudaParameters.cpp)
    target_compile_definitions(testUsdGenCudaParameters PRIVATE USDGEN_ENABLE_CUDA)
    set_target_properties(testUsdGenCudaParameters PROPERTIES USDGEN_KEEP_ASSERTS ON)
    target_link_libraries(testUsdGenCudaParameters PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaParameters COMMAND testUsdGenCudaParameters)
    set_tests_properties(testUsdGenCudaParameters PROPERTIES LABELS "T1;cuda;expressions" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaSurfaceInput tests/internal/usdGenCurves/testUsdGenCudaSurfaceInput.cpp)
    target_compile_definitions(testUsdGenCudaSurfaceInput PRIVATE USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenCudaSurfaceInput PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaSurfaceInput COMMAND testUsdGenCudaSurfaceInput)
    set_tests_properties(testUsdGenCudaSurfaceInput PROPERTIES LABELS "T0;cuda;sources")
    add_executable(testUsdGenCudaRbfSession tests/internal/usdGenCurves/testUsdGenCudaRbfSession.cpp)
    target_link_libraries(testUsdGenCudaRbfSession PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaRbfSession COMMAND testUsdGenCudaRbfSession)
    set_tests_properties(testUsdGenCudaRbfSession PROPERTIES LABELS "T1;cuda;rbf;sessions" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaLengthSession tests/internal/usdGenCurves/testUsdGenCudaLengthSession.cpp)
    target_link_libraries(testUsdGenCudaLengthSession PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaLengthSession COMMAND testUsdGenCudaLengthSession)
    set_tests_properties(testUsdGenCudaLengthSession PROPERTIES LABELS "T1;cuda;operators;sessions" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaAsyncCook tests/internal/usdGen/testUsdGenCudaAsyncCook.cpp)
    target_link_libraries(testUsdGenCudaAsyncCook PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaAsyncCook COMMAND testUsdGenCudaAsyncCook)
    set_tests_properties(testUsdGenCudaAsyncCook PROPERTIES LABELS "T1;cuda;sessions" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu SKIP_RETURN_CODE 77)
    add_executable(testUsdGenCudaSession tests/internal/usdGen/testUsdGenCudaSession.cpp)
    target_link_libraries(testUsdGenCudaSession PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    target_link_libraries(testUsdGenCudaSession PRIVATE usdGenTestUtils usdGenHydraCurves)
    add_test(NAME testUsdGenCudaSession COMMAND testUsdGenCudaSession)
    set_tests_properties(testUsdGenCudaSession PROPERTIES LABELS "T1;cuda;sessions" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaSessionTaskGraph tests/internal/usdGenCurves/testUsdGenCudaSessionTaskGraph.cpp)
    target_link_libraries(testUsdGenCudaSessionTaskGraph PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    target_link_libraries(testUsdGenCudaSessionTaskGraph PRIVATE usdGenTestUtils usdGenHydraCurves)
    add_test(NAME testUsdGenCudaSessionTaskGraph COMMAND testUsdGenCudaSessionTaskGraph)
    set_tests_properties(testUsdGenCudaSessionTaskGraph PROPERTIES LABELS "T1;cuda;sessions;scheduling" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaSourceInput tests/internal/usdGenCurves/testUsdGenCudaSourceInput.cpp)
    target_link_libraries(testUsdGenCudaSourceInput PRIVATE usdGenWithPrimitives usdGenUsd usdGenCurvesCuda)
    add_test(NAME testUsdGenCudaSourceInput COMMAND testUsdGenCudaSourceInput)
    set_tests_properties(testUsdGenCudaSourceInput PROPERTIES LABELS "T0;cuda;sources")
    foreach(_cuda_test CurveSource ExpressionContext Width SurfaceBinding Length CurveCompaction Picking PointOverride Topology)
        usdgen_test_source(_usdgen_src testUsdGenCuda${_cuda_test}.cu)
        add_executable(testUsdGenCuda${_cuda_test} ${_usdgen_src})
        target_link_libraries(testUsdGenCuda${_cuda_test} PRIVATE usdGenCurvesCuda usdGenWithPrimitives usdGenUsd)
        set_target_properties(testUsdGenCuda${_cuda_test} PROPERTIES USDGEN_KEEP_ASSERTS ON)
        add_test(NAME testUsdGenCuda${_cuda_test} COMMAND testUsdGenCuda${_cuda_test})
        set_tests_properties(testUsdGenCuda${_cuda_test} PROPERTIES
            LABELS "T1;cuda" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    endforeach()
    add_executable(testUsdGenCudaNonWidthCompare tests/internal/usdGenCurves/testUsdGenCudaNonWidthCompare.cu)
    target_link_libraries(testUsdGenCudaNonWidthCompare PRIVATE usdGenCurvesCuda usdGenWithPrimitives usdGenUsd)
    set_target_properties(testUsdGenCudaNonWidthCompare PROPERTIES USDGEN_KEEP_ASSERTS ON)
    add_test(NAME testUsdGenCudaNonWidthCompare COMMAND testUsdGenCudaNonWidthCompare)
    set_tests_properties(testUsdGenCudaNonWidthCompare PROPERTIES
        LABELS "T1;cuda;dag;cow;topology" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    foreach(_cuda_test Rbf Deform)
        usdgen_test_source(_usdgen_src testUsdGenCuda${_cuda_test}.cu)
        add_executable(testUsdGenCuda${_cuda_test} ${_usdgen_src})
        target_link_libraries(testUsdGenCuda${_cuda_test} PRIVATE usdGenCurvesCuda usdGen)
        # These CUDA oracles use assert for both execution and checks.
        # Release's -DNDEBUG must never turn them into false-green no-ops.
        set_target_properties(testUsdGenCuda${_cuda_test} PROPERTIES USDGEN_KEEP_ASSERTS ON)
        add_test(NAME testUsdGenCuda${_cuda_test} COMMAND testUsdGenCuda${_cuda_test})
        set_tests_properties(testUsdGenCuda${_cuda_test} PROPERTIES
            LABELS "T1;cuda;rbf" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    endforeach()
    add_executable(testUsdGenCudaStyle tests/internal/usdGenCurves/testUsdGenCudaStyle.cu)
    target_link_libraries(testUsdGenCudaStyle PRIVATE usdGenCurvesCuda usdGen)
    add_test(NAME testUsdGenCudaStyle COMMAND testUsdGenCudaStyle)
    set_tests_properties(testUsdGenCudaStyle PROPERTIES
        LABELS "T1;cuda;operators" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaNoise tests/internal/usdGenCurves/testUsdGenCudaNoise.cpp)
    # The oracle calls the explicitly-instantiated SeExpr 3->3 double FBM;
    # CUDA itself is deliberately float and built with global --fmad=false.
    target_link_libraries(testUsdGenCudaNoise PRIVATE usdGenCurvesCuda usdGen usdGenMath usdGen_seexpr)
    add_test(NAME testUsdGenCudaNoise COMMAND testUsdGenCudaNoise)
    set_tests_properties(testUsdGenCudaNoise PROPERTIES
        LABELS "T1;cuda;operators;noise" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaNoiseSession tests/internal/usdGenCurves/testUsdGenCudaNoiseSession.cpp)
    target_link_libraries(testUsdGenCudaNoiseSession PRIVATE usdGenCurvesCuda usdGenWithPrimitives usdGenUsd)
    add_test(NAME testUsdGenCudaNoiseSession COMMAND testUsdGenCudaNoiseSession)
    set_tests_properties(testUsdGenCudaNoiseSession PROPERTIES
        LABELS "T1;cuda;operators;noise;session;cow" TIMEOUT 120
        RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaRootFrames tests/internal/usdGenSurface/testUsdGenCudaRootFrames.cpp)
    target_link_libraries(testUsdGenCudaRootFrames PRIVATE usdGenCurvesCuda usdGen gf)
    add_test(NAME testUsdGenCudaRootFrames COMMAND testUsdGenCudaRootFrames)
    set_tests_properties(testUsdGenCudaRootFrames PROPERTIES
        LABELS "T1;cuda;root-frames" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaRootFrameGather tests/internal/usdGenSurface/testUsdGenCudaRootFrameGather.cpp)
    target_link_libraries(testUsdGenCudaRootFrameGather PRIVATE usdGenCurvesCuda usdGen)
    add_test(NAME testUsdGenCudaRootFrameGather COMMAND testUsdGenCudaRootFrameGather)
    set_tests_properties(testUsdGenCudaRootFrameGather PROPERTIES
        LABELS "T1;cuda;root-frames" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    add_executable(testUsdGenCudaExpression tests/internal/usdGenExpr/testUsdGenCudaExpression.cpp
        tests/api/usdGenSeExprOracle.cpp)
    set_source_files_properties(tests/api/usdGenSeExprOracle.cpp PROPERTIES
        COMPILE_DEFINITIONS "SeExpr2=UsdGenSeExprFrontend;SeExprInternal2=UsdGenSeExprFrontendInternal"
        INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/seexprFrontend/SeExpr2")
    target_link_libraries(testUsdGenCudaExpression PRIVATE usdGenCurvesCuda usdGenWithPrimitives usdGenUsd usdGenExprFrontend)
    add_test(NAME testUsdGenCudaExpression COMMAND testUsdGenCudaExpression)
    set_tests_properties(testUsdGenCudaExpression PROPERTIES
        LABELS "T1;cuda;expressions" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
    # Bit-for-bit CPU/CUDA parity over the shared IR interpreter.
    add_executable(testUsdGenCudaExpressionParity tests/internal/usdGenExpr/testUsdGenCudaExpressionParity.cpp)
    target_link_libraries(testUsdGenCudaExpressionParity PRIVATE usdGenCurvesCuda usdGenWithPrimitives usdGenUsd usdGenExprFrontend)
    add_test(NAME testUsdGenCudaExpressionParity COMMAND testUsdGenCudaExpressionParity)
    set_tests_properties(testUsdGenCudaExpressionParity PROPERTIES
        LABELS "T1;cuda;expressions;parity" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()
