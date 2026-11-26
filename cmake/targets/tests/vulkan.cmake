# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Tests: Vulkan backend. Included from the root CMakeLists.txt inside if(USDGEN_BUILD_TESTS).

if(USDGEN_BUILD_VULKAN_TESTS)
    # Vulkan host-backend core tests: init (factory/lifecycle), memory
    # (resources), dispatch (completion service, timeline notification), and
    # error rendering (errors). Native tests require a compatible Vulkan
    # device and exit 77 when unavailable; ctest treats that as SKIP.
    add_executable(testUsdGenVulkanRetirementStress tests/testUsdGenVulkanRetirementStress.cpp)
    target_link_libraries(testUsdGenVulkanRetirementStress PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanRetirementStress COMMAND testUsdGenVulkanRetirementStress)
    add_test(NAME testUsdGenVulkanRetirementStartFail COMMAND testUsdGenVulkanRetirementStress --start-fail)
    set_tests_properties(testUsdGenVulkanRetirementStress testUsdGenVulkanRetirementStartFail PROPERTIES
        LABELS "T0;retirement;vulkan" TIMEOUT 120)
    add_executable(testUsdGenVulkanExternalRetirement tests/testUsdGenVulkanExternalRetirement.cpp)
    target_link_libraries(testUsdGenVulkanExternalRetirement PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanExternalRetirement COMMAND testUsdGenVulkanExternalRetirement)
    add_test(NAME testUsdGenVulkanExternalRetirementExit COMMAND testUsdGenVulkanExternalRetirement --exit)
    set_tests_properties(testUsdGenVulkanExternalRetirement testUsdGenVulkanExternalRetirementExit PROPERTIES
        LABELS "T0;vulkan;retirement" TIMEOUT 30)
    add_executable(testUsdGenVulkanErrors tests/testUsdGenVulkanErrors.cpp)
    target_link_libraries(testUsdGenVulkanErrors PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanErrors COMMAND testUsdGenVulkanErrors)
    set_tests_properties(testUsdGenVulkanErrors PROPERTIES
        LABELS "T0;vulkan;errors" TIMEOUT 30)
    add_executable(testUsdGenVulkanCompletionService tests/testUsdGenVulkanCompletionService.cpp)
    target_link_libraries(testUsdGenVulkanCompletionService PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanCompletionService COMMAND testUsdGenVulkanCompletionService)
    add_test(NAME testUsdGenVulkanCompletionServiceClosePending COMMAND testUsdGenVulkanCompletionService --close-pending)
    add_test(NAME testUsdGenVulkanCompletionServiceOwnerUnavailable COMMAND testUsdGenVulkanCompletionService --owner-unavailable)
    add_test(NAME testUsdGenVulkanCompletionServiceIdleClose COMMAND testUsdGenVulkanCompletionService --idle-close)
    set_tests_properties(testUsdGenVulkanCompletionService testUsdGenVulkanCompletionServiceClosePending testUsdGenVulkanCompletionServiceOwnerUnavailable testUsdGenVulkanCompletionServiceIdleClose PROPERTIES
        LABELS "T1;vulkan;notification;retirement" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanTimelineNotification tests/testUsdGenVulkanTimelineNotification.cpp)
    target_link_libraries(testUsdGenVulkanTimelineNotification PRIVATE Vulkan::Vulkan Threads::Threads)
    add_test(NAME testUsdGenVulkanTimelineNotification COMMAND testUsdGenVulkanTimelineNotification)
    set_tests_properties(testUsdGenVulkanTimelineNotification PROPERTIES
        LABELS "T1;vulkan;notification" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanFactory tests/testUsdGenVulkanFactory.cpp)
    target_link_libraries(testUsdGenVulkanFactory PRIVATE usdGenVulkanFactory)
    add_test(NAME testUsdGenVulkanFactory COMMAND testUsdGenVulkanFactory)
    set_tests_properties(testUsdGenVulkanFactory PROPERTIES
        LABELS "T1;vulkan;resources;identity" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenVulkanDefaultResources tests/testUsdGenVulkanDefaultResources.cpp)
    target_link_libraries(testUsdGenVulkanDefaultResources PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanDefaultResources COMMAND testUsdGenVulkanDefaultResources)
    set_tests_properties(testUsdGenVulkanDefaultResources PROPERTIES
        LABELS "T0;vulkan;resources" TIMEOUT 30)

    if(USDGEN_ENABLE_VULKAN_RUNTIME)
        add_executable(testUsdGenVulkanDefaultSession tests/testUsdGenVulkanDefaultSession.cpp)
        target_link_libraries(testUsdGenVulkanDefaultSession PRIVATE usdGenUsd usdGenVulkanNative)
        add_dependencies(testUsdGenVulkanDefaultSession usdGenVulkanShaders)
        add_test(NAME testUsdGenVulkanDefaultSession COMMAND testUsdGenVulkanDefaultSession
            "${_usdgen_vulkan_width_spv}")
        add_test(NAME testUsdGenVulkanDefaultSessionMissing COMMAND testUsdGenVulkanDefaultSession --missing)
        add_test(NAME testUsdGenVulkanDefaultSessionLifecycle COMMAND testUsdGenVulkanDefaultSession
            "${_usdgen_vulkan_width_spv}" --lifecycle)
        set_tests_properties(testUsdGenVulkanDefaultSessionLifecycle PROPERTIES
            LABELS "T1;vulkan;retirement" RESOURCE_LOCK usdgen_gpu
            TIMEOUT 180 SKIP_RETURN_CODE 77
            FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Testing/shader-discovery")
        add_test(NAME testUsdGenVulkanDefaultSessionDiscovery COMMAND testUsdGenVulkanDefaultSession --auto)
        set_tests_properties(testUsdGenVulkanDefaultSessionDiscovery PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Testing/shader-discovery")
        set_tests_properties(testUsdGenVulkanDefaultSession testUsdGenVulkanDefaultSessionMissing
            testUsdGenVulkanDefaultSessionDiscovery PROPERTIES
            LABELS "T1;vulkan;session;default-provider" RESOURCE_LOCK usdgen_gpu
            TIMEOUT 120 SKIP_RETURN_CODE 77
            FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
        if(USDGEN_ENABLE_CUDA)
            add_executable(testUsdGenCudaVulkanSessionParity tests/testUsdGenCudaVulkanSessionParity.cpp)
            target_link_libraries(testUsdGenCudaVulkanSessionParity PRIVATE usdGenUsd usdGenVulkanNative usdGenCuda usdGen_ptex CUDA::cudart)
            add_dependencies(testUsdGenCudaVulkanSessionParity usdGenVulkanShaders)
            add_test(NAME testUsdGenCudaVulkanSessionParity COMMAND testUsdGenCudaVulkanSessionParity
                "${CMAKE_CURRENT_BINARY_DIR}/vulkan")
            set_tests_properties(testUsdGenCudaVulkanSessionParity PROPERTIES
                LABELS "T1;vulkan;cuda;session;cuda-vulkan-parity" RESOURCE_LOCK usdgen_gpu
                TIMEOUT 300 SKIP_RETURN_CODE 77
                FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
            # Every Vulkan buffer, fresh or reused from the memory pool, starts
            # as 0xFF bytes: no output may depend on zero-filled allocations.
            add_test(NAME testUsdGenCudaVulkanSessionParityPoisoned COMMAND testUsdGenCudaVulkanSessionParity
                "${CMAKE_CURRENT_BINARY_DIR}/vulkan")
            set_tests_properties(testUsdGenCudaVulkanSessionParityPoisoned PROPERTIES
                LABELS "T1;vulkan;cuda;session;cuda-vulkan-parity" RESOURCE_LOCK usdgen_gpu
                TIMEOUT 300 SKIP_RETURN_CODE 77 ENVIRONMENT "USDGEN_VULKAN_POISON_MEMORY=1"
                FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
        endif()
    endif()
    add_executable(testUsdGenVulkanLifecycle tests/testUsdGenVulkanLifecycle.cpp)
    add_executable(testUsdGenVulkanSurfaceCageSource tests/testUsdGenVulkanSurfaceCageSource.cpp)
    target_link_libraries(testUsdGenVulkanSurfaceCageSource PRIVATE usdGenUsd usdGenVulkanNative usdGen_ptex)
    add_test(NAME testUsdGenVulkanSurfaceCageSource COMMAND testUsdGenVulkanSurfaceCageSource)
    set_tests_properties(testUsdGenVulkanSurfaceCageSource PROPERTIES LABELS "T0;vulkan;source" TIMEOUT 60)
    target_link_libraries(testUsdGenVulkanLifecycle PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanLifecycle COMMAND testUsdGenVulkanLifecycle)
    add_test(NAME testUsdGenVulkanLifecycleQuarantine COMMAND testUsdGenVulkanLifecycle --quarantine)
    set_tests_properties(testUsdGenVulkanLifecycle testUsdGenVulkanLifecycleQuarantine PROPERTIES
        LABELS "T1;vulkan;resources;retirement" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenVulkanResources tests/testUsdGenVulkanResources.cpp)
    target_link_libraries(testUsdGenVulkanResources PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanResources COMMAND testUsdGenVulkanResources)
    add_test(NAME testUsdGenVulkanResourcesQuarantine COMMAND testUsdGenVulkanResources --quarantine)
    set_tests_properties(testUsdGenVulkanResources testUsdGenVulkanResourcesQuarantine PROPERTIES
        LABELS "T1;vulkan;cow;resources;retirement" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(testUsdGenVulkanFenceFd tests/testUsdGenVulkanFenceFd.cpp)
        target_link_libraries(testUsdGenVulkanFenceFd PRIVATE Vulkan::Vulkan)
        add_test(NAME testUsdGenVulkanFenceFd COMMAND testUsdGenVulkanFenceFd)
        set_tests_properties(testUsdGenVulkanFenceFd PROPERTIES
            LABELS "T1;vulkan;notification" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
            SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
            FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
        add_executable(testUsdGenVulkanSemaphoreFd tests/testUsdGenVulkanSemaphoreFd.cpp)
        target_link_libraries(testUsdGenVulkanSemaphoreFd PRIVATE Vulkan::Vulkan)
        add_test(NAME testUsdGenVulkanSemaphoreFd COMMAND testUsdGenVulkanSemaphoreFd)
        set_tests_properties(testUsdGenVulkanSemaphoreFd PROPERTIES
            LABELS "T1;vulkan;notification" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
            SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
            FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    endif()
    add_executable(testUsdGenVulkanSource tests/testUsdGenVulkanSource.cpp)
    add_executable(testUsdGenVulkanSourcePreparation tests/testUsdGenVulkanSourcePreparation.cpp)
    add_executable(testUsdGenVulkanExecutionPlan tests/testUsdGenVulkanExecutionPlan.cpp)
    target_link_libraries(testUsdGenVulkanExecutionPlan PRIVATE usdGenUsd usdGenVulkanNative)
    if(USDGEN_ENABLE_VULKAN_RUNTIME)
        target_compile_definitions(testUsdGenVulkanExecutionPlan PRIVATE USDGEN_ENABLE_VULKAN_RUNTIME)
    endif()
    add_test(NAME testUsdGenVulkanExecutionPlan COMMAND testUsdGenVulkanExecutionPlan)
    set_tests_properties(testUsdGenVulkanExecutionPlan PROPERTIES LABELS "T1;vulkan;planning;metadata" TIMEOUT 60)
    # CUDA-gap disposition gate: host-only proof that every large CUDA gap
    # without a Vulkan counterpart rejects Vulkan plans with a clear
    # diagnostic (or is already served on Vulkan, proven here). The lease
    # fail-closed leg needs the CUDA lane only when it is compiled in.
    add_executable(testUsdGenVulkanGapRejections tests/testUsdGenVulkanGapRejections.cpp)
    target_link_libraries(testUsdGenVulkanGapRejections PRIVATE usdGenUsd usdGenVulkanNative)
    if(USDGEN_ENABLE_CUDA)
        target_link_libraries(testUsdGenVulkanGapRejections PRIVATE usdGenCuda)
        target_compile_definitions(testUsdGenVulkanGapRejections PRIVATE USDGEN_ENABLE_CUDA)
    endif()
    add_test(NAME testUsdGenVulkanGapRejections COMMAND testUsdGenVulkanGapRejections)
    set_tests_properties(testUsdGenVulkanGapRejections PROPERTIES LABELS "T1;vulkan;planning;gaps" TIMEOUT 60)
    add_executable(testUsdGenVulkanWidthOverlapWitness tests/testUsdGenVulkanWidthOverlapWitness.cpp
        "${_usdgen_vulkan_overlap_spv}")
    target_link_libraries(testUsdGenVulkanWidthOverlapWitness PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanWidthOverlapWitness COMMAND testUsdGenVulkanWidthOverlapWitness
        "${_usdgen_vulkan_overlap_spv}")
    set_tests_properties(testUsdGenVulkanWidthOverlapWitness PROPERTIES
        LABELS "T1;vulkan;operators;witness" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 120 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanSourceCapture tests/testUsdGenVulkanSourceCapture.cpp)
    target_link_libraries(testUsdGenVulkanSourceCapture PRIVATE usdGenUsd usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanSourceCapture COMMAND testUsdGenVulkanSourceCapture)
    set_tests_properties(testUsdGenVulkanSourceCapture PROPERTIES
        LABELS "T1;vulkan;sources;root-bindings;cow" TIMEOUT 60)
    target_link_libraries(testUsdGenVulkanSourcePreparation PRIVATE usdGenVulkanNative usdGenWithPrimitives usdGenUsd)
    add_test(NAME testUsdGenVulkanSourcePreparation COMMAND testUsdGenVulkanSourcePreparation)
    set_tests_properties(testUsdGenVulkanSourcePreparation PROPERTIES
        LABELS "T1;vulkan;sources;cow" TIMEOUT 60)
    add_executable(testUsdGenVulkanSourceStride tests/testUsdGenVulkanSourceStride.cpp)
    target_link_libraries(testUsdGenVulkanSourceStride PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanSourceStride COMMAND testUsdGenVulkanSourceStride)
    set_tests_properties(testUsdGenVulkanSourceStride PROPERTIES
        LABELS "T1;vulkan;sources" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanSource PRIVATE usdGenVulkanNative)
    # Exact-value oracles must match unfused GPU evaluation order.
    target_compile_options(testUsdGenVulkanSource PRIVATE -ffp-contract=off)
    add_test(NAME testUsdGenVulkanSource COMMAND testUsdGenVulkanSource)
    add_test(NAME testUsdGenVulkanSourceQuarantine COMMAND testUsdGenVulkanSource --quarantine)
    set_tests_properties(testUsdGenVulkanSource testUsdGenVulkanSourceQuarantine PROPERTIES
        LABELS "T1;vulkan;cow;sources;retirement" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenVulkanWidth tests/testUsdGenVulkanWidth.cpp "${_usdgen_vulkan_width_spv}")
    add_executable(testUsdGenVulkanNonWidthCompare tests/testUsdGenVulkanNonWidthCompare.cpp "${_usdgen_vulkan_width_spv}" "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_compare_spv}" "${_usdgen_vulkan_blend_spv}")
    target_link_libraries(testUsdGenVulkanNonWidthCompare PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanNonWidthCompare COMMAND testUsdGenVulkanNonWidthCompare "${_usdgen_vulkan_width_spv}" "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_compare_spv}" "${_usdgen_vulkan_blend_spv}")
    set_tests_properties(testUsdGenVulkanNonWidthCompare PROPERTIES
        LABELS "T1;vulkan;cow;operators;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanRootFrameCow tests/testUsdGenVulkanRootFrameCow.cpp
        "${_usdgen_vulkan_width_spv}" "${_usdgen_vulkan_length_spv}"
        "${_usdgen_vulkan_blend_spv}" "${_usdgen_vulkan_compare_spv}")
    add_executable(testUsdGenVulkanLengthCompaction tests/testUsdGenVulkanLengthCompaction.cpp
        "${_usdgen_vulkan_compaction_spv}")
    add_executable(testUsdGenVulkanResampleVk tests/testUsdGenVulkanResampleVk.cpp
        "${_usdgen_vulkan_resample_vk_spv}")
    add_executable(testUsdGenVulkanCullAdmission tests/testUsdGenVulkanCullAdmission.cpp
        "${_usdgen_vulkan_compaction_spv}")
    add_executable(testUsdGenVulkanTopologyPipeline tests/testUsdGenVulkanTopologyPipeline.cpp "${_usdgen_vulkan_topo_spv}")
    target_link_libraries(testUsdGenVulkanTopologyPipeline PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanTopologyPipeline COMMAND testUsdGenVulkanTopologyPipeline "${_usdgen_vulkan_topo_spv}")
    set_tests_properties(testUsdGenVulkanTopologyPipeline PROPERTIES
        LABELS "T1;vulkan;operators;topology" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanPointOverridePipeline tests/testUsdGenVulkanPointOverridePipeline.cpp "${_usdgen_vulkan_po_spv}")
    target_link_libraries(testUsdGenVulkanPointOverridePipeline PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanPointOverridePipeline COMMAND testUsdGenVulkanPointOverridePipeline "${_usdgen_vulkan_po_spv}")
    set_tests_properties(testUsdGenVulkanPointOverridePipeline PROPERTIES
        LABELS "T1;vulkan;operators;pointOverride" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    # Phase 2 picktiles: Vulkan picking + tile spans/bounds/indices vs the
    # CUDA twins. CUDA-only registration (compares against gpu/ lane).
    if(USDGEN_ENABLE_CUDA)
        add_executable(testUsdGenVulkanPicktileVkParity tests/testUsdGenVulkanPicktileVkParity.cpp
            "${_usdgen_vulkan_picktile_pick_spv}" "${_usdgen_vulkan_picktile_footprint_spv}"
            "${_usdgen_vulkan_picktile_tiles_spv}" "${_usdgen_vulkan_picktile_bounds_spv}"
            "${_usdgen_vulkan_picktile_indices_spv}")
        target_link_libraries(testUsdGenVulkanPicktileVkParity PRIVATE usdGenVulkanNative usdGenCuda usdGenWithPrimitives usdGenUsd)
        target_compile_definitions(testUsdGenVulkanPicktileVkParity PRIVATE USDGEN_ENABLE_CUDA)
        if(USDGEN_ENABLE_VULKAN_RUNTIME)
            target_compile_definitions(testUsdGenVulkanPicktileVkParity PRIVATE USDGEN_ENABLE_VULKAN_RUNTIME)
        endif()
        # Exact-value oracles must match unfused GPU evaluation order.
        target_compile_options(testUsdGenVulkanPicktileVkParity PRIVATE -ffp-contract=off)
        add_test(NAME testUsdGenVulkanPicktileVkParity COMMAND testUsdGenVulkanPicktileVkParity
            "${_usdgen_vulkan_picktile_pick_spv}" "${_usdgen_vulkan_picktile_footprint_spv}"
            "${_usdgen_vulkan_picktile_tiles_spv}" "${_usdgen_vulkan_picktile_bounds_spv}"
            "${_usdgen_vulkan_picktile_indices_spv}")
        set_tests_properties(testUsdGenVulkanPicktileVkParity PROPERTIES
            LABELS "T1;parity;cuda;vulkan;picking;tiles" RESOURCE_LOCK usdgen_gpu
            TIMEOUT 600 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    endif()
    add_executable(testUsdGenVulkanCurveGrowPipeline tests/testUsdGenVulkanCurveGrowPipeline.cpp "${_usdgen_vulkan_grow_spv}")
    target_link_libraries(testUsdGenVulkanCurveGrowPipeline PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanCurveGrowPipeline COMMAND testUsdGenVulkanCurveGrowPipeline "${_usdgen_vulkan_grow_spv}")
    set_tests_properties(testUsdGenVulkanCurveGrowPipeline PROPERTIES
        LABELS "T1;vulkan;operators;grow" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_test(NAME testUsdGenVulkanCurveGrowPipelineQuarantine COMMAND testUsdGenVulkanCurveGrowPipeline "${_usdgen_vulkan_grow_spv}" --quarantine)
    set_tests_properties(testUsdGenVulkanCurveGrowPipelineQuarantine PROPERTIES
        LABELS "T1;vulkan;operators;grow;quarantine" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanScatterGrowPipeline tests/testUsdGenVulkanScatterGrowPipeline.cpp "${_usdgen_vulkan_scatter_grow_spv}")
    target_link_libraries(testUsdGenVulkanScatterGrowPipeline PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanScatterGrowPipeline COMMAND testUsdGenVulkanScatterGrowPipeline "${_usdgen_vulkan_scatter_grow_spv}")
    set_tests_properties(testUsdGenVulkanScatterGrowPipeline PROPERTIES
        LABELS "T1;vulkan;operators;scatterGrow" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanScatterGrowCpuReject tests/testUsdGenVulkanScatterGrowCpuReject.cpp)
    target_link_libraries(testUsdGenVulkanScatterGrowCpuReject PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanScatterGrowCpuReject COMMAND testUsdGenVulkanScatterGrowCpuReject)
    set_tests_properties(testUsdGenVulkanScatterGrowCpuReject PROPERTIES
        LABELS "T0;vulkan;operators;scatterGrowCpuReject" TIMEOUT 60)
    add_executable(testUsdGenVulkanRootFrames tests/testUsdGenVulkanRootFrames.cpp
        "${_usdgen_vulkan_rfval_spv}" "${_usdgen_vulkan_rfbuild_spv}")
    target_link_libraries(testUsdGenVulkanRootFrames PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanRootFrames COMMAND testUsdGenVulkanRootFrames
        "${_usdgen_vulkan_rfval_spv}" "${_usdgen_vulkan_rfbuild_spv}")
    set_tests_properties(testUsdGenVulkanRootFrames PROPERTIES
        LABELS "T1;vulkan;operators;rootFrames" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_test(NAME testUsdGenVulkanRootFramesQuarantine COMMAND testUsdGenVulkanRootFrames
        "${_usdgen_vulkan_rfval_spv}" "${_usdgen_vulkan_rfbuild_spv}" --quarantine)
    set_tests_properties(testUsdGenVulkanRootFramesQuarantine PROPERTIES
        LABELS "T1;vulkan;operators;rootFrames;quarantine" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanRootFrameGather tests/testUsdGenVulkanRootFrameGatherPipeline.cpp
        "${_usdgen_vulkan_rfgather_spv}")
    target_link_libraries(testUsdGenVulkanRootFrameGather PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanRootFrameGather COMMAND testUsdGenVulkanRootFrameGather
        "${_usdgen_vulkan_rfgather_spv}")
    set_tests_properties(testUsdGenVulkanRootFrameGather PROPERTIES
        LABELS "T1;vulkan;operators;rootFrameGather" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_test(NAME testUsdGenVulkanRootFrameGatherQuarantine COMMAND testUsdGenVulkanRootFrameGather
        "${_usdgen_vulkan_rfgather_spv}" --quarantine)
    set_tests_properties(testUsdGenVulkanRootFrameGatherQuarantine PROPERTIES
        LABELS "T1;vulkan;operators;rootFrameGather;quarantine" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanNoise tests/testUsdGenVulkanNoise.cpp
        "${_usdgen_vulkan_noise_validate_spv}" "${_usdgen_vulkan_noise_generate_spv}")
    target_link_libraries(testUsdGenVulkanNoise PRIVATE usdGenVulkanNative usdGen_seexpr usdGenMath)
    add_test(NAME testUsdGenVulkanNoise COMMAND testUsdGenVulkanNoise
        "${_usdgen_vulkan_noise_validate_spv}" "${_usdgen_vulkan_noise_generate_spv}")
    set_tests_properties(testUsdGenVulkanNoise PROPERTIES
        LABELS "T1;vulkan;operators;noise" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    # CUDA-vs-Vulkan equivalence gate: one shared Noise fixture runs on both
    # GPU lanes (the CUDA leg only when USDGEN_ENABLE_CUDA) and is compared
    # against the common SeExpr oracle plus cross-backend. Skips 77 when no
    # GPU backend is available, so single-backend builds stay green.
    add_executable(testUsdGenCudaVulkanParity tests/testUsdGenCudaVulkanParity.cpp
        "${_usdgen_vulkan_noise_validate_spv}" "${_usdgen_vulkan_noise_generate_spv}")
    target_link_libraries(testUsdGenCudaVulkanParity PRIVATE usdGenVulkanNative usdGen_seexpr usdGenMath)
    if(USDGEN_ENABLE_CUDA)
        target_link_libraries(testUsdGenCudaVulkanParity PRIVATE usdGenCuda)
        target_compile_definitions(testUsdGenCudaVulkanParity PRIVATE USDGEN_ENABLE_CUDA)
    endif()
    add_test(NAME testUsdGenCudaVulkanParity COMMAND testUsdGenCudaVulkanParity
        "${_usdgen_vulkan_noise_validate_spv}" "${_usdgen_vulkan_noise_generate_spv}")
    set_tests_properties(testUsdGenCudaVulkanParity PROPERTIES
        LABELS "T1;parity;cuda;vulkan;operators;noise" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 120 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    # Phase-2 styleops gap: CUDA-vs-Vulkan equivalence gate for the
    # device-only style primitives (StyleVkPipeline vs CudaStyleOps).
    # Skips 77 when no GPU backend is available.
    add_executable(testUsdGenVulkanStyleVkParity tests/testUsdGenVulkanStyleVkParity.cpp
        "${_usdgen_vulkan_stylevk_ramp_spv}" "${_usdgen_vulkan_stylevk_length_spv}" "${_usdgen_vulkan_stylevk_grow_spv}")
    target_link_libraries(testUsdGenVulkanStyleVkParity PRIVATE usdGenVulkanNative)
    if(USDGEN_ENABLE_CUDA)
        target_link_libraries(testUsdGenVulkanStyleVkParity PRIVATE usdGenCuda)
        target_compile_definitions(testUsdGenVulkanStyleVkParity PRIVATE USDGEN_ENABLE_CUDA)
    endif()
    add_test(NAME testUsdGenVulkanStyleVkParity COMMAND testUsdGenVulkanStyleVkParity
        "${_usdgen_vulkan_stylevk_ramp_spv}" "${_usdgen_vulkan_stylevk_length_spv}" "${_usdgen_vulkan_stylevk_grow_spv}")
    set_tests_properties(testUsdGenVulkanStyleVkParity PROPERTIES
        LABELS "T1;parity;cuda;vulkan;operators;style" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 120 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanDeform tests/testUsdGenVulkanDeform.cpp
        "${_usdgen_vulkan_deform_evaluate_spv}" "${_usdgen_vulkan_deform_apply_spv}"
        "${_usdgen_vulkan_deform_evaluate_verify_spv}" "${_usdgen_vulkan_deform_evaluate_fill_spv}"
        "${_usdgen_vulkan_deform_evaluate_cached_spv}")
    target_link_libraries(testUsdGenVulkanDeform PRIVATE usdGenVulkanNative usdGenMath)
    add_test(NAME testUsdGenVulkanDeform COMMAND testUsdGenVulkanDeform
        "${_usdgen_vulkan_deform_evaluate_spv}" "${_usdgen_vulkan_deform_apply_spv}"
        "${_usdgen_vulkan_deform_evaluate_verify_spv}" "${_usdgen_vulkan_deform_evaluate_fill_spv}"
        "${_usdgen_vulkan_deform_evaluate_cached_spv}")
    set_tests_properties(testUsdGenVulkanDeform PROPERTIES
        LABELS "T1;vulkan;operators;deform" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    # Many threads on one cached deform pipeline and one queue timeline.
    add_executable(testUsdGenVulkanDeformStress tests/testUsdGenVulkanDeformStress.cpp
        "${_usdgen_vulkan_deform_evaluate_spv}" "${_usdgen_vulkan_deform_apply_spv}"
        "${_usdgen_vulkan_deform_evaluate_verify_spv}" "${_usdgen_vulkan_deform_evaluate_fill_spv}"
        "${_usdgen_vulkan_deform_evaluate_cached_spv}")
    target_link_libraries(testUsdGenVulkanDeformStress PRIVATE usdGenVulkanNative usdGenMath)
    add_test(NAME testUsdGenVulkanDeformStress COMMAND testUsdGenVulkanDeformStress
        "${_usdgen_vulkan_deform_evaluate_spv}" "${_usdgen_vulkan_deform_apply_spv}"
        "${_usdgen_vulkan_deform_evaluate_verify_spv}" "${_usdgen_vulkan_deform_evaluate_fill_spv}"
        "${_usdgen_vulkan_deform_evaluate_cached_spv}")
    set_tests_properties(testUsdGenVulkanDeformStress PROPERTIES
        LABELS "T1;vulkan;deform;stress;scheduling" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 120 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanQueueTimeline tests/testUsdGenVulkanQueueTimeline.cpp)
    target_link_libraries(testUsdGenVulkanQueueTimeline PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanQueueTimeline COMMAND testUsdGenVulkanQueueTimeline)
    set_tests_properties(testUsdGenVulkanQueueTimeline PROPERTIES
        LABELS "T1;vulkan;stress;scheduling" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 120 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanSurfaceVk tests/testUsdGenVulkanSurfaceVk.cpp
        "${_usdgen_vulkan_deform_evaluate_spv}" "${_usdgen_vulkan_deform_apply_spv}")
    target_link_libraries(testUsdGenVulkanSurfaceVk PRIVATE usdGenUsd usdGenVulkanNative usdGenMath)
    if(USDGEN_ENABLE_CUDA)
        target_link_libraries(testUsdGenVulkanSurfaceVk PRIVATE usdGenCuda)
        target_compile_definitions(testUsdGenVulkanSurfaceVk PRIVATE USDGEN_ENABLE_CUDA)
    endif()
    add_test(NAME testUsdGenVulkanSurfaceVk COMMAND testUsdGenVulkanSurfaceVk
        "${_usdgen_vulkan_deform_evaluate_spv}" "${_usdgen_vulkan_deform_apply_spv}")
    set_tests_properties(testUsdGenVulkanSurfaceVk PROPERTIES
        LABELS "T1;parity;cuda;vulkan;operators;deform;surface" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 120 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanRbfVkParity tests/testUsdGenVulkanRbfVkParity.cpp
        "${_usdgen_vulkan_rbfvk_extent_spv}" "${_usdgen_vulkan_rbfvk_gram_spv}" "${_usdgen_vulkan_rbfvk_build_matrix_spv}" "${_usdgen_vulkan_rbfvk_lu_spv}" "${_usdgen_vulkan_rbfvk_rhs_spv}" "${_usdgen_vulkan_rbfvk_trisolve_spv}" "${_usdgen_vulkan_rbfvk_evaluate_spv}" "${_usdgen_vulkan_rbfvk_apply_spv}" "${_usdgen_vulkan_rbfvk_verify_spv}" "${_usdgen_vulkan_rbfvk_fill_spv}" "${_usdgen_vulkan_rbfvk_evaluate_cached_spv}")
    target_link_libraries(testUsdGenVulkanRbfVkParity PRIVATE usdGenUsd usdGenVulkanNative usdGenMath)
    if(USDGEN_ENABLE_CUDA)
        target_link_libraries(testUsdGenVulkanRbfVkParity PRIVATE usdGenCuda)
        target_compile_definitions(testUsdGenVulkanRbfVkParity PRIVATE USDGEN_ENABLE_CUDA)
    endif()
    add_test(NAME testUsdGenVulkanRbfVkParity COMMAND testUsdGenVulkanRbfVkParity
        "${_usdgen_vulkan_rbfvk_extent_spv}" "${_usdgen_vulkan_rbfvk_gram_spv}" "${_usdgen_vulkan_rbfvk_build_matrix_spv}" "${_usdgen_vulkan_rbfvk_lu_spv}" "${_usdgen_vulkan_rbfvk_rhs_spv}" "${_usdgen_vulkan_rbfvk_trisolve_spv}" "${_usdgen_vulkan_rbfvk_evaluate_spv}" "${_usdgen_vulkan_rbfvk_apply_spv}" "${_usdgen_vulkan_rbfvk_verify_spv}" "${_usdgen_vulkan_rbfvk_fill_spv}" "${_usdgen_vulkan_rbfvk_evaluate_cached_spv}")
    set_tests_properties(testUsdGenVulkanRbfVkParity PROPERTIES
        LABELS "T1;parity;cuda;vulkan;operators;rbf;deform" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 780 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanCsourceVkParity tests/testUsdGenVulkanCsourceVkParity.cpp
        "${_usdgen_vulkan_deform_evaluate_spv}" "${_usdgen_vulkan_deform_apply_spv}")
    target_link_libraries(testUsdGenVulkanCsourceVkParity PRIVATE usdGenUsd usdGenVulkanNative)
    if(USDGEN_ENABLE_CUDA)
        target_link_libraries(testUsdGenVulkanCsourceVkParity PRIVATE usdGenCuda)
        target_compile_definitions(testUsdGenVulkanCsourceVkParity PRIVATE USDGEN_ENABLE_CUDA)
    endif()
    add_test(NAME testUsdGenVulkanCsourceVkParity COMMAND testUsdGenVulkanCsourceVkParity
        "${_usdgen_vulkan_deform_evaluate_spv}" "${_usdgen_vulkan_deform_apply_spv}")
    set_tests_properties(testUsdGenVulkanCsourceVkParity PROPERTIES
        LABELS "T1;parity;cuda;vulkan;sources;csource" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 300 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanExprVkParity tests/testUsdGenVulkanExprVkParity.cpp
        tests/testUsdGenVulkanExprVkParityOracle.cpp
        "${_usdgen_vulkan_exprvk_evaluate_spv}" "${_usdgen_vulkan_exprvk_context_spv}"
        "${_usdgen_vulkan_exprvk_width_spv}" "${_usdgen_vulkan_exprvk_exprop_spv}")
    target_link_libraries(testUsdGenVulkanExprVkParity PRIVATE usdGenUsd usdGenVulkanNative usdGenExprFrontend)
    if(USDGEN_ENABLE_CUDA)
        target_link_libraries(testUsdGenVulkanExprVkParity PRIVATE usdGenCuda)
        target_compile_definitions(testUsdGenVulkanExprVkParity PRIVATE USDGEN_ENABLE_CUDA)
    endif()
    target_compile_options(testUsdGenVulkanExprVkParity PRIVATE ${_usdgen_host_expr_fp_options})
    if(MSVC)
        set_property(SOURCE tests/testUsdGenVulkanExprVkParityOracle.cpp
            APPEND PROPERTY COMPILE_OPTIONS /O1)
    else()
        set_property(SOURCE tests/testUsdGenVulkanExprVkParityOracle.cpp
            APPEND PROPERTY COMPILE_OPTIONS -O1 -fno-fast-math)
    endif()
    add_test(NAME testUsdGenVulkanExprVkParity COMMAND testUsdGenVulkanExprVkParity
        "${_usdgen_vulkan_exprvk_evaluate_spv}" "${_usdgen_vulkan_exprvk_context_spv}"
        "${_usdgen_vulkan_exprvk_width_spv}" "${_usdgen_vulkan_exprvk_exprop_spv}")
    set_tests_properties(testUsdGenVulkanExprVkParity PROPERTIES
        LABELS "T1;parity;cuda;vulkan;operators;expression" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 600 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanLengthSet tests/testUsdGenVulkanLengthSet.cpp
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}")
    target_link_libraries(testUsdGenVulkanLengthSet PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanLengthSet COMMAND testUsdGenVulkanLengthSet
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}")
    set_tests_properties(testUsdGenVulkanLengthSet PROPERTIES
        LABELS "T1;vulkan;cow;length-set;operators;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanLengthCutExtend tests/testUsdGenVulkanLengthCutExtend.cpp
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}" "${_usdgen_vulkan_cut_extend_spv}")
    target_link_libraries(testUsdGenVulkanLengthCutExtend PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanLengthCutExtend COMMAND testUsdGenVulkanLengthCutExtend
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}" "${_usdgen_vulkan_cut_extend_spv}")
    set_tests_properties(testUsdGenVulkanLengthCutExtend PROPERTIES
        LABELS "T1;vulkan;cow;length-cut-extend;operators;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanLengthReparam tests/testUsdGenVulkanLengthReparam.cpp
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}"
        "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}")
    target_link_libraries(testUsdGenVulkanLengthReparam PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanLengthReparam COMMAND testUsdGenVulkanLengthReparam
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}"
        "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}")
    set_tests_properties(testUsdGenVulkanLengthReparam PROPERTIES
        LABELS "T1;vulkan;cow;length-reparam;operators;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanLengthMinimum tests/testUsdGenVulkanLengthMinimum.cpp
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}"
        "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}"
        "${_usdgen_vulkan_minimum_spv}")
    target_link_libraries(testUsdGenVulkanLengthMinimum PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanLengthMinimum COMMAND testUsdGenVulkanLengthMinimum
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}"
        "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}"
        "${_usdgen_vulkan_minimum_spv}")
    set_tests_properties(testUsdGenVulkanLengthMinimum PROPERTIES
        LABELS "T1;vulkan;cow;length-minimum;operators;retirement;resources" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanLengthLiteralV1 tests/testUsdGenVulkanLengthLiteralV1.cpp
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}"
        "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}"
        "${_usdgen_vulkan_minimum_spv}" "${_usdgen_vulkan_literal_v1_spv}")
    target_link_libraries(testUsdGenVulkanLengthLiteralV1 PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanLengthLiteralV1 COMMAND testUsdGenVulkanLengthLiteralV1
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}"
        "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}"
        "${_usdgen_vulkan_minimum_spv}" "${_usdgen_vulkan_literal_v1_spv}")
    set_tests_properties(testUsdGenVulkanLengthLiteralV1 PROPERTIES
        LABELS "T1;vulkan;cow;length-random;operators;retirement;resources" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanLengthEnvelopeV1 tests/testUsdGenVulkanLengthEnvelopeV1.cpp
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}"
        "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}"
        "${_usdgen_vulkan_minimum_spv}" "${_usdgen_vulkan_literal_v1_spv}" "${_usdgen_vulkan_envelope_v1_spv}")
    target_link_libraries(testUsdGenVulkanLengthEnvelopeV1 PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanLengthEnvelopeV1 COMMAND testUsdGenVulkanLengthEnvelopeV1
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}"
        "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}"
        "${_usdgen_vulkan_minimum_spv}" "${_usdgen_vulkan_literal_v1_spv}" "${_usdgen_vulkan_envelope_v1_spv}")
    set_tests_properties(testUsdGenVulkanLengthEnvelopeV1 PROPERTIES
        LABELS "T1;vulkan;cow;length-envelope;operators;retirement;resources" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanLengthDeviceV1 tests/testUsdGenVulkanLengthDeviceV1.cpp)
    target_link_libraries(testUsdGenVulkanLengthDeviceV1 PRIVATE usdGenVulkanNative)
    add_dependencies(testUsdGenVulkanLengthDeviceV1 usdGenVulkanShaders)
    add_test(NAME testUsdGenVulkanLengthDeviceV1 COMMAND testUsdGenVulkanLengthDeviceV1
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}"
        "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}"
        "${_usdgen_vulkan_minimum_spv}" "${_usdgen_vulkan_literal_v1_spv}"
        "${_usdgen_vulkan_envelope_v1_spv}" "${_usdgen_vulkan_device_v1_spv}"
        "${_usdgen_vulkan_compaction_spv}")
    set_tests_properties(testUsdGenVulkanLengthDeviceV1 PROPERTIES
        LABELS "T1;vulkan;length;expressions" RESOURCE_LOCK usdgen_gpu TIMEOUT 120
        SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    set(_usdgen_vulkan_arithmetic_spv "${CMAKE_CURRENT_BINARY_DIR}/vulkan/envelopeArithmetic.spv")
    add_custom_command(OUTPUT "${_usdgen_vulkan_arithmetic_spv}"
        COMMAND "${USDGEN_GLSLANG_EXECUTABLE}" ${_usdgen_vulkan_glsl_flags}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/envelopeArithmetic.comp"
            -o "${_usdgen_vulkan_arithmetic_spv}"
        COMMAND "${USDGEN_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.2 "${_usdgen_vulkan_arithmetic_spv}"
        DEPENDS tests/shaders/envelopeArithmetic.comp
            libs/core/primitives/usdGenCurves/usdGenCurves/vulkan/shaders/float32Envelope.glsl VERBATIM)
    add_executable(testUsdGenVulkanEnvelopeArithmetic tests/testUsdGenVulkanEnvelopeArithmetic.cpp
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}"
        "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}"
        "${_usdgen_vulkan_minimum_spv}" "${_usdgen_vulkan_literal_v1_spv}"
        "${_usdgen_vulkan_envelope_v1_spv}" "${_usdgen_vulkan_arithmetic_spv}")
    target_link_libraries(testUsdGenVulkanEnvelopeArithmetic PRIVATE usdGenVulkanNative)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(testUsdGenVulkanEnvelopeArithmetic PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME testUsdGenVulkanEnvelopeArithmetic COMMAND testUsdGenVulkanEnvelopeArithmetic
        "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_set_spv}"
        "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}"
        "${_usdgen_vulkan_minimum_spv}" "${_usdgen_vulkan_literal_v1_spv}"
        "${_usdgen_vulkan_envelope_v1_spv}" "${_usdgen_vulkan_arithmetic_spv}")
    set_tests_properties(testUsdGenVulkanEnvelopeArithmetic PROPERTIES
        LABELS "T1;vulkan;length-envelope;numeric" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanCullAdmission PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanCullAdmission COMMAND testUsdGenVulkanCullAdmission
        "${_usdgen_vulkan_compaction_spv}")
    set_tests_properties(testUsdGenVulkanCullAdmission PROPERTIES
        LABELS "T1;vulkan;cow;topology;resources;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanLengthCompaction PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanLengthCompaction COMMAND testUsdGenVulkanLengthCompaction
        "${_usdgen_vulkan_compaction_spv}")
    set_tests_properties(testUsdGenVulkanLengthCompaction PROPERTIES
        LABELS "T1;vulkan;cow;topology;operators;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanResampleVk PRIVATE usdGenUsd usdGenVulkanNative)
    if(USDGEN_ENABLE_CUDA)
        target_link_libraries(testUsdGenVulkanResampleVk PRIVATE usdGenCuda)
        target_compile_definitions(testUsdGenVulkanResampleVk PRIVATE USDGEN_ENABLE_CUDA)
    endif()
    add_test(NAME testUsdGenVulkanResampleVk COMMAND testUsdGenVulkanResampleVk
        "${_usdgen_vulkan_resample_vk_spv}")
    set_tests_properties(testUsdGenVulkanResampleVk PROPERTIES
        LABELS "T1;vulkan;operators;resample;parity" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 120 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanRootFrameCow PRIVATE usdGenUsd usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanRootFrameCow COMMAND testUsdGenVulkanRootFrameCow
        "${_usdgen_vulkan_width_spv}" "${_usdgen_vulkan_length_spv}"
        "${_usdgen_vulkan_blend_spv}" "${_usdgen_vulkan_compare_spv}")
    set_tests_properties(testUsdGenVulkanRootFrameCow PROPERTIES
        LABELS "T1;vulkan;cow;root-frames;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_sources(testUsdGenVulkanSource PRIVATE "${_usdgen_vulkan_width_spv}")
    target_sources(testUsdGenVulkanSource PRIVATE "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_blend_spv}")
    add_test(NAME testUsdGenVulkanSourceWidthCow COMMAND testUsdGenVulkanSource --cow "${_usdgen_vulkan_width_spv}")
    add_test(NAME testUsdGenVulkanSourceLengthCow COMMAND testUsdGenVulkanSource "${_usdgen_vulkan_width_spv}" --length "${_usdgen_vulkan_length_spv}")
    add_test(NAME testUsdGenVulkanSourceWidthBlendCow COMMAND testUsdGenVulkanSource --blend "${_usdgen_vulkan_width_spv}" "${_usdgen_vulkan_blend_spv}")
    set_tests_properties(testUsdGenVulkanSourceLengthCow testUsdGenVulkanSourceWidthBlendCow PROPERTIES
        LABELS "T1;vulkan;cow;operators;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanSourceAccounting tests/testUsdGenVulkanSourceAccounting.cpp "${_usdgen_vulkan_width_spv}")
    add_executable(testUsdGenVulkanDeviceGenerationAdapter tests/testUsdGenVulkanDeviceGenerationAdapter.cpp "${_usdgen_vulkan_width_spv}")
    target_link_libraries(testUsdGenVulkanDeviceGenerationAdapter PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanDeviceGenerationAdapter COMMAND testUsdGenVulkanDeviceGenerationAdapter "${_usdgen_vulkan_width_spv}")
    add_test(NAME testUsdGenVulkanDeviceGenerationAdapterCloseHeld COMMAND testUsdGenVulkanDeviceGenerationAdapter "${_usdgen_vulkan_width_spv}" --close-held)
    add_test(NAME testUsdGenVulkanDeviceGenerationAdapterExitHeld COMMAND testUsdGenVulkanDeviceGenerationAdapter "${_usdgen_vulkan_width_spv}" --exit-held)
    set_tests_properties(testUsdGenVulkanDeviceGenerationAdapter testUsdGenVulkanDeviceGenerationAdapterCloseHeld testUsdGenVulkanDeviceGenerationAdapterExitHeld PROPERTIES
        LABELS "T1;vulkan;cow;generation;retirement" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanSourceWidthJob tests/testUsdGenVulkanSourceWidthJob.cpp "${_usdgen_vulkan_width_spv}")
    add_executable(testUsdGenVulkanSessionProvider tests/testUsdGenVulkanSessionProvider.cpp "${_usdgen_vulkan_width_spv}")
    add_executable(testUsdGenVulkanSession tests/testUsdGenVulkanSession.cpp "${_usdgen_vulkan_width_spv}")
    target_sources(testUsdGenVulkanSession PRIVATE "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_blend_spv}")
    add_executable(testUsdGenVulkanSessionCoalescing tests/testUsdGenVulkanSessionCoalescing.cpp "${_usdgen_vulkan_width_spv}")
    add_executable(testUsdGenVulkanSessionContextLoss tests/testUsdGenVulkanSessionContextLoss.cpp "${_usdgen_vulkan_width_spv}")
    target_link_libraries(testUsdGenVulkanSessionContextLoss PRIVATE usdGenUsd usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanSessionContextLoss COMMAND testUsdGenVulkanSessionContextLoss "${_usdgen_vulkan_width_spv}")
    set_tests_properties(testUsdGenVulkanSessionContextLoss PROPERTIES
        LABELS "T1;vulkan;session;context-loss;cow;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanSessionCoalescing PRIVATE usdGenUsd usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanSessionCoalescing COMMAND testUsdGenVulkanSessionCoalescing "${_usdgen_vulkan_width_spv}")
    add_test(NAME testUsdGenVulkanSessionFollowerCancel COMMAND testUsdGenVulkanSessionCoalescing "${_usdgen_vulkan_width_spv}" --cancel-follower)
    set_tests_properties(testUsdGenVulkanSessionCoalescing testUsdGenVulkanSessionFollowerCancel PROPERTIES
        LABELS "T1;vulkan;session;coalescing;cow;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanSession PRIVATE usdGenUsd usdGenVulkanNative)
    # Exact-value oracles must match unfused GPU evaluation order.
    target_compile_options(testUsdGenVulkanSession PRIVATE -ffp-contract=off)
    add_test(NAME testUsdGenVulkanSession COMMAND testUsdGenVulkanSession "${_usdgen_vulkan_width_spv}")
    add_test(NAME testUsdGenVulkanSessionLostReturn COMMAND testUsdGenVulkanSession "${_usdgen_vulkan_width_spv}" --lost-return)
    add_test(NAME testUsdGenVulkanSessionAdmissionRollback COMMAND testUsdGenVulkanSession "${_usdgen_vulkan_width_spv}" --admission-rollback)
    add_test(NAME testUsdGenVulkanSessionLength COMMAND testUsdGenVulkanSession "${_usdgen_vulkan_width_spv}" --length "${_usdgen_vulkan_length_spv}")
    add_test(NAME testUsdGenVulkanSessionWidthDag COMMAND testUsdGenVulkanSession "${_usdgen_vulkan_width_spv}" --dag "${_usdgen_vulkan_blend_spv}")
    add_test(NAME testUsdGenVulkanSessionLengthWidthDag COMMAND testUsdGenVulkanSession "${_usdgen_vulkan_width_spv}" --dag "${_usdgen_vulkan_blend_spv}" --length "${_usdgen_vulkan_length_spv}")
    target_sources(testUsdGenVulkanSession PRIVATE "${_usdgen_vulkan_compare_spv}")
    add_test(NAME testUsdGenVulkanSessionCrossOriginDag COMMAND testUsdGenVulkanSession "${_usdgen_vulkan_width_spv}" --dag "${_usdgen_vulkan_blend_spv}" --length "${_usdgen_vulkan_length_spv}" --compare "${_usdgen_vulkan_compare_spv}")
    add_test(NAME testUsdGenVulkanSessionNamedChannels COMMAND testUsdGenVulkanSession "${_usdgen_vulkan_width_spv}" --dag "${_usdgen_vulkan_blend_spv}" --length "${_usdgen_vulkan_length_spv}" --compare "${_usdgen_vulkan_compare_spv}" --named)
    set_tests_properties(testUsdGenVulkanSessionNamedChannels PROPERTIES
        LABELS "T1;vulkan;cow;operators;retirement;named" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_test(NAME testUsdGenVulkanSessionRootBindings COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --roots --named
        --length "${_usdgen_vulkan_length_spv}" --dag "${_usdgen_vulkan_blend_spv}"
        --compare "${_usdgen_vulkan_compare_spv}")
    set_tests_properties(testUsdGenVulkanSessionRootBindings PROPERTIES
        LABELS "T1;vulkan;session;root-bindings;cow;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_test(NAME testUsdGenVulkanSessionAffineSource COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --roots --named --affine
        --length "${_usdgen_vulkan_length_spv}" --dag "${_usdgen_vulkan_blend_spv}"
        --compare "${_usdgen_vulkan_compare_spv}")
    target_sources(testUsdGenVulkanSession PRIVATE "${_usdgen_vulkan_compaction_spv}")
    add_test(NAME testUsdGenVulkanSessionLengthCull COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --cull "${_usdgen_vulkan_compaction_spv}")
    add_test(NAME testUsdGenVulkanSessionLengthCullRoots COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --roots --cull "${_usdgen_vulkan_compaction_spv}")
    target_sources(testUsdGenVulkanSession PRIVATE "${_usdgen_vulkan_set_spv}")
    add_test(NAME testUsdGenVulkanSessionLengthSet COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --length "${_usdgen_vulkan_length_spv}" --set "${_usdgen_vulkan_set_spv}")
    add_test(NAME testUsdGenVulkanSessionLengthSetRootDag COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --roots --named
        --length "${_usdgen_vulkan_length_spv}" --set "${_usdgen_vulkan_set_spv}"
        --dag "${_usdgen_vulkan_blend_spv}" --compare "${_usdgen_vulkan_compare_spv}")
    target_sources(testUsdGenVulkanSession PRIVATE "${_usdgen_vulkan_cut_extend_spv}")
    add_test(NAME testUsdGenVulkanSessionCutExtend COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cut "${_usdgen_vulkan_cut_extend_spv}"
        --cull "${_usdgen_vulkan_compaction_spv}" --dag "${_usdgen_vulkan_blend_spv}"
        --compare "${_usdgen_vulkan_compare_spv}")
    add_test(NAME testUsdGenVulkanSessionRootCutExtend COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --roots --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cut "${_usdgen_vulkan_cut_extend_spv}"
        --cull "${_usdgen_vulkan_compaction_spv}" --dag "${_usdgen_vulkan_blend_spv}"
        --compare "${_usdgen_vulkan_compare_spv}")
    target_sources(testUsdGenVulkanSession PRIVATE "${_usdgen_vulkan_reparam_spv}")
    add_test(NAME testUsdGenVulkanSessionReparam COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cut "${_usdgen_vulkan_cut_extend_spv}"
        --reparam "${_usdgen_vulkan_reparam_spv}" --cull "${_usdgen_vulkan_compaction_spv}"
        --dag "${_usdgen_vulkan_blend_spv}" --compare "${_usdgen_vulkan_compare_spv}")
    add_test(NAME testUsdGenVulkanSessionRootReparam COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --roots --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cut "${_usdgen_vulkan_cut_extend_spv}"
        --reparam "${_usdgen_vulkan_reparam_spv}" --cull "${_usdgen_vulkan_compaction_spv}"
        --dag "${_usdgen_vulkan_blend_spv}" --compare "${_usdgen_vulkan_compare_spv}")
    add_test(NAME testUsdGenVulkanSessionMixedLength COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cull "${_usdgen_vulkan_compaction_spv}"
        --dag "${_usdgen_vulkan_blend_spv}" --compare "${_usdgen_vulkan_compare_spv}")
    add_test(NAME testUsdGenVulkanSessionMixedLengthRoots COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --roots --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cull "${_usdgen_vulkan_compaction_spv}"
        --dag "${_usdgen_vulkan_blend_spv}" --compare "${_usdgen_vulkan_compare_spv}")
    set_tests_properties(testUsdGenVulkanSessionMixedLength testUsdGenVulkanSessionMixedLengthRoots PROPERTIES
        LABELS "T1;vulkan;session;length-set;topology;cow;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    set_tests_properties(testUsdGenVulkanSessionLengthSet testUsdGenVulkanSessionLengthSetRootDag PROPERTIES
        LABELS "T1;vulkan;session;length-set;cow;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    set_tests_properties(testUsdGenVulkanSessionCutExtend testUsdGenVulkanSessionRootCutExtend PROPERTIES
        LABELS "T1;vulkan;session;length-cut-extend;cow;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    set_tests_properties(testUsdGenVulkanSessionReparam testUsdGenVulkanSessionRootReparam PROPERTIES
        LABELS "T1;vulkan;session;length-reparam;cow;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_sources(testUsdGenVulkanSession PRIVATE "${_usdgen_vulkan_minimum_spv}")
    add_test(NAME testUsdGenVulkanSessionMinimum COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cut "${_usdgen_vulkan_cut_extend_spv}"
        --reparam "${_usdgen_vulkan_reparam_spv}" --minimum "${_usdgen_vulkan_minimum_spv}"
        --cull "${_usdgen_vulkan_compaction_spv}" --dag "${_usdgen_vulkan_blend_spv}"
        --compare "${_usdgen_vulkan_compare_spv}")
    add_test(NAME testUsdGenVulkanSessionRootMinimum COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --roots --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cut "${_usdgen_vulkan_cut_extend_spv}"
        --reparam "${_usdgen_vulkan_reparam_spv}" --minimum "${_usdgen_vulkan_minimum_spv}"
        --cull "${_usdgen_vulkan_compaction_spv}" --dag "${_usdgen_vulkan_blend_spv}"
        --compare "${_usdgen_vulkan_compare_spv}")
    set_tests_properties(testUsdGenVulkanSessionMinimum testUsdGenVulkanSessionRootMinimum PROPERTIES
        LABELS "T1;vulkan;session;length-minimum;cow;topology;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_sources(testUsdGenVulkanSession PRIVATE "${_usdgen_vulkan_literal_v1_spv}")
    add_test(NAME testUsdGenVulkanSessionLiteralV1 COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cut "${_usdgen_vulkan_cut_extend_spv}"
        --reparam "${_usdgen_vulkan_reparam_spv}" --minimum "${_usdgen_vulkan_minimum_spv}"
        --literal-v1 "${_usdgen_vulkan_literal_v1_spv}"
        --cull "${_usdgen_vulkan_compaction_spv}" --dag "${_usdgen_vulkan_blend_spv}"
        --compare "${_usdgen_vulkan_compare_spv}")
    add_test(NAME testUsdGenVulkanSessionRootLiteralV1 COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --roots --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cut "${_usdgen_vulkan_cut_extend_spv}"
        --reparam "${_usdgen_vulkan_reparam_spv}" --minimum "${_usdgen_vulkan_minimum_spv}"
        --literal-v1 "${_usdgen_vulkan_literal_v1_spv}"
        --cull "${_usdgen_vulkan_compaction_spv}" --dag "${_usdgen_vulkan_blend_spv}"
        --compare "${_usdgen_vulkan_compare_spv}")
    set_tests_properties(testUsdGenVulkanSessionLiteralV1 testUsdGenVulkanSessionRootLiteralV1 PROPERTIES
        LABELS "T1;vulkan;session;length-random;cow;topology;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_sources(testUsdGenVulkanSession PRIVATE "${_usdgen_vulkan_envelope_v1_spv}")
    add_test(NAME testUsdGenVulkanSessionEnvelopeV1 COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cut "${_usdgen_vulkan_cut_extend_spv}"
        --reparam "${_usdgen_vulkan_reparam_spv}" --minimum "${_usdgen_vulkan_minimum_spv}"
        --literal-v1 "${_usdgen_vulkan_literal_v1_spv}"
        --envelope-v1 "${_usdgen_vulkan_envelope_v1_spv}"
        --cull "${_usdgen_vulkan_compaction_spv}" --dag "${_usdgen_vulkan_blend_spv}"
        --compare "${_usdgen_vulkan_compare_spv}")
    add_test(NAME testUsdGenVulkanSessionRootEnvelopeV1 COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --roots --length "${_usdgen_vulkan_length_spv}"
        --set "${_usdgen_vulkan_set_spv}" --cut "${_usdgen_vulkan_cut_extend_spv}"
        --reparam "${_usdgen_vulkan_reparam_spv}" --minimum "${_usdgen_vulkan_minimum_spv}"
        --literal-v1 "${_usdgen_vulkan_literal_v1_spv}"
        --envelope-v1 "${_usdgen_vulkan_envelope_v1_spv}"
        --cull "${_usdgen_vulkan_compaction_spv}" --dag "${_usdgen_vulkan_blend_spv}"
        --compare "${_usdgen_vulkan_compare_spv}")
    set_tests_properties(testUsdGenVulkanSessionEnvelopeV1 testUsdGenVulkanSessionRootEnvelopeV1 PROPERTIES
        LABELS "T1;vulkan;session;length-envelope;cow;topology;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_test(NAME testUsdGenVulkanSessionCullDag COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --cull "${_usdgen_vulkan_compaction_spv}"
        --dag "${_usdgen_vulkan_blend_spv}" --compare "${_usdgen_vulkan_compare_spv}")
    add_test(NAME testUsdGenVulkanSessionCullRootDag COMMAND testUsdGenVulkanSession
        "${_usdgen_vulkan_width_spv}" --roots --cull "${_usdgen_vulkan_compaction_spv}"
        --dag "${_usdgen_vulkan_blend_spv}" --compare "${_usdgen_vulkan_compare_spv}")
    set_tests_properties(testUsdGenVulkanSessionCullDag testUsdGenVulkanSessionCullRootDag PROPERTIES
        LABELS "T1;vulkan;session;dag;topology;cow;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    set_tests_properties(testUsdGenVulkanSessionLengthCull testUsdGenVulkanSessionLengthCullRoots PROPERTIES
        LABELS "T1;vulkan;session;topology;cow;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    set_tests_properties(testUsdGenVulkanSessionAffineSource PROPERTIES
        LABELS "T1;vulkan;session;root-bindings;affine;cow;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    set_tests_properties(testUsdGenVulkanSessionCrossOriginDag PROPERTIES
        LABELS "T1;vulkan;cow;operators;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    set_tests_properties(testUsdGenVulkanSession testUsdGenVulkanSessionLostReturn testUsdGenVulkanSessionAdmissionRollback testUsdGenVulkanSessionLength testUsdGenVulkanSessionWidthDag testUsdGenVulkanSessionLengthWidthDag PROPERTIES
        LABELS "T1;vulkan;session;generation;cow;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanSessionProvider PRIVATE usdGenUsd usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanSessionProvider COMMAND testUsdGenVulkanSessionProvider "${_usdgen_vulkan_width_spv}")
    set_tests_properties(testUsdGenVulkanSessionProvider PROPERTIES
        LABELS "T1;vulkan;generation;retirement;provider" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanPublicationJob tests/testUsdGenVulkanPublicationJob.cpp "${_usdgen_vulkan_width_spv}")
    add_executable(testUsdGenVulkanPlanExecutor tests/testUsdGenVulkanPlanExecutor.cpp "${_usdgen_vulkan_width_spv}")
    target_sources(testUsdGenVulkanPlanExecutor PRIVATE "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_blend_spv}" "${_usdgen_vulkan_set_spv}" "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}" "${_usdgen_vulkan_minimum_spv}" "${_usdgen_vulkan_literal_v1_spv}" "${_usdgen_vulkan_noise_validate_spv}" "${_usdgen_vulkan_noise_generate_spv}")
    target_link_libraries(testUsdGenVulkanPlanExecutor PRIVATE usdGenUsd usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanPlanExecutor COMMAND testUsdGenVulkanPlanExecutor "${_usdgen_vulkan_width_spv}" "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_blend_spv}" "${_usdgen_vulkan_set_spv}" "${_usdgen_vulkan_cut_extend_spv}" "${_usdgen_vulkan_reparam_spv}" "${_usdgen_vulkan_minimum_spv}" "${_usdgen_vulkan_literal_v1_spv}" "${_usdgen_vulkan_noise_validate_spv}" "${_usdgen_vulkan_noise_generate_spv}")
    set_tests_properties(testUsdGenVulkanPlanExecutor PROPERTIES
        LABELS "T1;vulkan;sources;operators;generation;retirement" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanPublicationJob PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanPublicationJob COMMAND testUsdGenVulkanPublicationJob "${_usdgen_vulkan_width_spv}")
    add_test(NAME testUsdGenVulkanPublicationQueuedCancel COMMAND testUsdGenVulkanPublicationJob "${_usdgen_vulkan_width_spv}" --queued-cancel)
    add_test(NAME testUsdGenVulkanPublicationPrepareReject COMMAND testUsdGenVulkanPublicationJob "${_usdgen_vulkan_width_spv}" --prepare-reject)
    add_test(NAME testUsdGenVulkanPublicationOwnerUnavailable COMMAND testUsdGenVulkanPublicationJob "${_usdgen_vulkan_width_spv}" --owner-unavailable)
    set_tests_properties(testUsdGenVulkanPublicationJob testUsdGenVulkanPublicationQueuedCancel
        testUsdGenVulkanPublicationPrepareReject testUsdGenVulkanPublicationOwnerUnavailable PROPERTIES
        LABELS "T1;vulkan;sources;generation;retirement" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanSourcePreparationLifecycle tests/testUsdGenVulkanSourcePreparationLifecycle.cpp "${_usdgen_vulkan_width_spv}")
    target_link_libraries(testUsdGenVulkanSourcePreparationLifecycle PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanSourcePreparationCancel COMMAND testUsdGenVulkanSourcePreparationLifecycle "${_usdgen_vulkan_width_spv}")
    add_test(NAME testUsdGenVulkanSourcePreparationReject COMMAND testUsdGenVulkanSourcePreparationLifecycle "${_usdgen_vulkan_width_spv}" --reject)
    add_test(NAME testUsdGenVulkanSourcePreparationOwnerUnavailable COMMAND testUsdGenVulkanSourcePreparationLifecycle "${_usdgen_vulkan_width_spv}" --owner-unavailable)
    set_tests_properties(testUsdGenVulkanSourcePreparationCancel testUsdGenVulkanSourcePreparationReject testUsdGenVulkanSourcePreparationOwnerUnavailable PROPERTIES
        LABELS "T1;vulkan;sources;retirement;resources" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanWidthAdmission tests/testUsdGenVulkanWidthAdmission.cpp "${_usdgen_vulkan_width_spv}")
    target_link_libraries(testUsdGenVulkanWidthAdmission PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanWidthAdmission COMMAND testUsdGenVulkanWidthAdmission "${_usdgen_vulkan_width_spv}")
    set_tests_properties(testUsdGenVulkanWidthAdmission PROPERTIES
        LABELS "T1;vulkan;notification;resources;cow" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_executable(testUsdGenVulkanSourceWidthAsync tests/testUsdGenVulkanSourceWidthAsync.cpp "${_usdgen_vulkan_width_spv}")
    target_link_libraries(testUsdGenVulkanSourceWidthAsync PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanSourceWidthAsync COMMAND testUsdGenVulkanSourceWidthAsync "${_usdgen_vulkan_width_spv}")
    add_test(NAME testUsdGenVulkanSourceWidthAsyncEarly COMMAND testUsdGenVulkanSourceWidthAsync "${_usdgen_vulkan_width_spv}" --early-notifications)
    add_test(NAME testUsdGenVulkanSourceWidthPrepared COMMAND testUsdGenVulkanSourceWidthAsync "${_usdgen_vulkan_width_spv}" --prepared)
    add_test(NAME testUsdGenVulkanSourceWidthPreparedEarly COMMAND testUsdGenVulkanSourceWidthAsync "${_usdgen_vulkan_width_spv}" --prepared --early-notifications)
    set_tests_properties(testUsdGenVulkanSourceWidthAsync testUsdGenVulkanSourceWidthAsyncEarly testUsdGenVulkanSourceWidthPrepared testUsdGenVulkanSourceWidthPreparedEarly PROPERTIES
        LABELS "T1;vulkan;notification;cow;retirement" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_sources(testUsdGenVulkanSourceWidthAsync PRIVATE "${_usdgen_vulkan_picktile_tiles_spv}" "${_usdgen_vulkan_picktile_bounds_spv}")
    add_test(NAME testUsdGenVulkanSourceWidthAsyncTiles COMMAND testUsdGenVulkanSourceWidthAsync "${_usdgen_vulkan_width_spv}" --tiles "${_usdgen_vulkan_picktile_tiles_spv}" "${_usdgen_vulkan_picktile_bounds_spv}")
    set_tests_properties(testUsdGenVulkanSourceWidthAsyncTiles PROPERTIES
        LABELS "T1;vulkan;notification;cow;tiles" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanSourceWidthJob PRIVATE usdGenVulkanNative)
    target_sources(testUsdGenVulkanSourceWidthJob PRIVATE "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_blend_spv}" "${_usdgen_vulkan_compaction_spv}")
    add_test(NAME testUsdGenVulkanSourceWidthJob COMMAND testUsdGenVulkanSourceWidthJob "${_usdgen_vulkan_width_spv}")
    add_test(NAME testUsdGenVulkanSourceLengthWidthJob COMMAND testUsdGenVulkanSourceWidthJob "${_usdgen_vulkan_width_spv}" --length "${_usdgen_vulkan_length_spv}")
    add_test(NAME testUsdGenVulkanSourceCullCancel COMMAND testUsdGenVulkanSourceWidthJob
        "${_usdgen_vulkan_width_spv}" --cull "${_usdgen_vulkan_compaction_spv}")
    set_tests_properties(testUsdGenVulkanSourceCullCancel PROPERTIES
        LABELS "T1;vulkan;cow;topology;cancellation;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    add_test(NAME testUsdGenVulkanSourceWidthDagCancel COMMAND testUsdGenVulkanSourceWidthJob "${_usdgen_vulkan_width_spv}" --dag "${_usdgen_vulkan_blend_spv}")
    add_test(NAME testUsdGenVulkanDisabledWidth COMMAND testUsdGenVulkanSourceWidthJob "${_usdgen_vulkan_width_spv}" --disabled)
    set_tests_properties(testUsdGenVulkanDisabledWidth PROPERTIES
        LABELS "T1;vulkan;controls;passthrough" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_sources(testUsdGenVulkanSourceWidthJob PRIVATE "${_usdgen_vulkan_compare_spv}")
    add_test(NAME testUsdGenVulkanSourceWidthCrossCancel COMMAND testUsdGenVulkanSourceWidthJob "${_usdgen_vulkan_width_spv}" --cross "${_usdgen_vulkan_blend_spv}" "${_usdgen_vulkan_length_spv}" "${_usdgen_vulkan_compare_spv}")
    set_tests_properties(testUsdGenVulkanSourceWidthCrossCancel PROPERTIES
        LABELS "T1;vulkan;cow;operators;retirement" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 60 SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_sources(testUsdGenVulkanSourceWidthJob PRIVATE "${_usdgen_vulkan_picktile_tiles_spv}" "${_usdgen_vulkan_picktile_bounds_spv}")
    add_test(NAME testUsdGenVulkanSourceWidthTiles COMMAND testUsdGenVulkanSourceWidthJob "${_usdgen_vulkan_width_spv}" --tiles "${_usdgen_vulkan_picktile_tiles_spv}" "${_usdgen_vulkan_picktile_bounds_spv}")
    set_tests_properties(testUsdGenVulkanSourceWidthTiles PROPERTIES
        LABELS "T1;vulkan;sources;cow;tiles" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    set_tests_properties(testUsdGenVulkanSourceWidthJob testUsdGenVulkanSourceLengthWidthJob testUsdGenVulkanSourceWidthDagCancel PROPERTIES
        LABELS "T1;vulkan;sources;cow;retirement" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanSourceAccounting PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanSourceAccounting COMMAND testUsdGenVulkanSourceAccounting "${_usdgen_vulkan_width_spv}")
    set_tests_properties(testUsdGenVulkanSourceAccounting PROPERTIES
        LABELS "T1;vulkan;sources;cow;resources" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    set_tests_properties(testUsdGenVulkanSourceWidthCow PROPERTIES
        LABELS "T1;vulkan;cow;sources;operators" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenVulkanWidthPipeline tests/testUsdGenVulkanWidthPipeline.cpp "${_usdgen_vulkan_width_spv}")
    target_link_libraries(testUsdGenVulkanWidthPipeline PRIVATE usdGenVulkanNative)
    add_test(NAME testUsdGenVulkanWidthPipeline COMMAND testUsdGenVulkanWidthPipeline "${_usdgen_vulkan_width_spv}")
    add_test(NAME testUsdGenVulkanWidthPipelineQuarantine COMMAND testUsdGenVulkanWidthPipeline "${_usdgen_vulkan_width_spv}" --quarantine)
    set_tests_properties(testUsdGenVulkanWidthPipeline testUsdGenVulkanWidthPipelineQuarantine PROPERTIES
        LABELS "T1;vulkan;cow;operators;retirement" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    add_executable(testUsdGenVulkanMapVkSample tests/testUsdGenVulkanMapVkSample.cpp
        "${_usdgen_vulkan_mapvk_sample_spv}")
    # usdGenSurface: the CPU image sampler oracle (usdGen links no primitive).
    target_link_libraries(testUsdGenVulkanMapVkSample PRIVATE usdGenVulkanNative usdGenSurface)
    if(USDGEN_ENABLE_CUDA)
        target_link_libraries(testUsdGenVulkanMapVkSample PRIVATE usdGenCuda)
        target_compile_definitions(testUsdGenVulkanMapVkSample PRIVATE USDGEN_ENABLE_CUDA)
    endif()
    # The decomposed-rounding model must not be re-fused by the host compiler.
    if(NOT MSVC)
        target_compile_options(testUsdGenVulkanMapVkSample PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME testUsdGenVulkanMapVkSample
        COMMAND testUsdGenVulkanMapVkSample "${_usdgen_vulkan_mapvk_sample_spv}")
    add_test(NAME testUsdGenVulkanMapVkSampleQuarantine
        COMMAND testUsdGenVulkanMapVkSample "${_usdgen_vulkan_mapvk_sample_spv}" --quarantine)
    set_tests_properties(testUsdGenVulkanMapVkSample testUsdGenVulkanMapVkSampleQuarantine PROPERTIES
        LABELS "T1;vulkan;operators;maps;parity" TIMEOUT 600 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    target_link_libraries(testUsdGenVulkanWidth PRIVATE Vulkan::Vulkan)
    if(NOT MSVC)
        target_compile_options(testUsdGenVulkanWidth PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME testUsdGenVulkanWidth COMMAND testUsdGenVulkanWidth "${_usdgen_vulkan_width_spv}")
    set_tests_properties(testUsdGenVulkanWidth PROPERTIES
        LABELS "T1;vulkan;cow;operators" TIMEOUT 60 RESOURCE_LOCK usdgen_gpu
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    # A validation diagnostic must fail CTest even if the driver returns success.
    set_tests_properties(testUsdGenVulkanSource testUsdGenVulkanSourceQuarantine
        testUsdGenVulkanSourceWidthCow
        testUsdGenVulkanWidthPipeline testUsdGenVulkanWidthPipelineQuarantine
        testUsdGenVulkanWidth
        PROPERTIES FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    # A validation diagnostic must fail CTest even if the driver returns success.
    set_tests_properties(testUsdGenVulkanFactory testUsdGenVulkanLifecycle testUsdGenVulkanLifecycleQuarantine
        testUsdGenVulkanResources testUsdGenVulkanResourcesQuarantine
        PROPERTIES FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
    get_property(_usdgen_vulkan_tests DIRECTORY PROPERTY TESTS)
    foreach(_usdgen_test IN LISTS _usdgen_vulkan_tests)
        get_test_property(${_usdgen_test} LABELS _usdgen_labels)
        if("vulkan" IN_LIST _usdgen_labels)
            set_tests_properties(${_usdgen_test} PROPERTIES SKIP_RETURN_CODE 77
                FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
        endif()
    endforeach()
    if(USDGEN_ENABLE_CUDA)
        set_property(TEST testUsdGenCudaVulkanParity testUsdGenVulkanStyleVkParity
            testUsdGenVulkanSurfaceVk testUsdGenVulkanRbfVkParity
            testUsdGenVulkanCsourceVkParity testUsdGenVulkanExprVkParity
            testUsdGenVulkanMapVkSample testUsdGenVulkanPicktileVkParity
            testUsdGenVulkanResampleVk APPEND PROPERTY LABELS "cuda-vulkan-parity")
    endif()
endif()
