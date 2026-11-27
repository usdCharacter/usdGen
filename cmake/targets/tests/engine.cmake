# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Tests: plugin discovery, engine T0 tests, benches. Included from the root CMakeLists.txt inside if(USDGEN_BUILD_TESTS).

# M0: T1 tests
# The plugin resource dirs (build tree) must be on the plugin search path so the
# codeless schema types and the plugInfo entries are discoverable.
set(USDGEN_TEST_PLUGINPATH
    "${_usdgen_schema_resources};${_usdgen_imaging_pluginfo_dir}")
set(USDGEN_TEST_PLUGINPATH "${_usdgen_schema_resources};${CMAKE_CURRENT_BINARY_DIR}/usd/usdGenImaging/resources")

# sol S-3/X-2 fixture: a synthetic renderer-level HdSceneIndexPlugin carrying
# the plugInfo tag "hdPrman:motionBlur" so the chain test can prove usdGen's
# plugInfo ordering contract ("before: [hdPrman:motionBlur]") without
# RenderMan installed. Loaded purely through PXR_PLUGINPATH_NAME.
add_library(fakeHdPrmanPlugin SHARED tests/fixtures/fakeHdPrmanPlugin.cpp)
set_target_properties(fakeHdPrmanPlugin PROPERTIES
    CXX_VISIBILITY_PRESET default)
target_link_libraries(fakeHdPrmanPlugin PRIVATE hd hdsi tf arch pxOsd)
set(_fake_prman_resources "${CMAKE_CURRENT_BINARY_DIR}/usd/fakeHdPrman/resources")
set(_fake_prman_pluginfo "${_fake_prman_resources}/plugInfo.json")
file(READ
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/fakeHdPrman/resources/plugInfo.json.in"
    _fake_prman_pluginfo_content)
file(GENERATE OUTPUT "${_fake_prman_pluginfo}"
    CONTENT "${_fake_prman_pluginfo_content}")

add_executable(testUsdGenChainOrder tests/hosts/usdGenUsd/testUsdGenChainOrder.cpp)
# Deliberately does NOT link usdGenImaging: the usdGen (and fixture) plugins
# are discovered via PXR_PLUGINPATH_NAME and dlopen'd by the registry during
# the test, so a TfErrorMark inside main catches any coding error emitted
# while the plugin libraries load (sol S-4/X-8 regression guard).
target_link_libraries(testUsdGenChainOrder PRIVATE usdImaging hdsi hd usd tf arch)
add_test(NAME testUsdGenChainOrder COMMAND testUsdGenChainOrder)
# N-4: this is the plugInfo load-order check, i.e. gate SI-5 (09 §5.2);
# the gate label belongs on the registration.
usdgen_test_pluginpath_ops(_usdgen_chain_order_env
    "${_usdgen_schema_resources}" "${_usdgen_imaging_resources}"
    "${_usdgen_pomade_resources}" "${_fake_prman_resources}"
    "${USDGEN_TEST_USD_DIR}/plugin/usd" "${USDGEN_TEST_USD_DIR}/lib/usd")
# The loader path is applied to every test by the sweep at the end of this
# block, so only the plugin search path is per-test here.
set_tests_properties(testUsdGenChainOrder PROPERTIES
    LABELS "T1;gate:SI-5"
    ENVIRONMENT_MODIFICATION "${_usdgen_chain_order_env}")

add_executable(testUsdGenPluginDiscovery tests/hosts/usdGenUsd/testUsdGenPluginDiscovery.cpp)
target_link_libraries(testUsdGenPluginDiscovery PRIVATE usdGenImaging usdGenTestUtilsHd)
add_test(NAME testUsdGenPluginDiscovery
    COMMAND testUsdGenPluginDiscovery
        ${_usdgen_schema_resources}
        ${_usdgen_imaging_resources}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/scenes/scene_empty_groom.usda)
usdgen_test_pluginpath_ops(_usdgen_plugin_discovery_env
    "${_usdgen_schema_resources}" "${_usdgen_imaging_resources}"
    "${USD_INSTALL_DIR}/plugin/usd")
set_tests_properties(testUsdGenPluginDiscovery PROPERTIES
    LABELS "T1"
    ENVIRONMENT_MODIFICATION "${_usdgen_plugin_discovery_env}")

# ---------------------------------------------------------------------------
# M1 tests (plan 11 §2.2; names/labels from 10-build-dependencies-testing.md
# §5.6 + §6.5; re-baselined by plan/13-codebase-alignment.md §1: all sources
# are implemented — no "M1-STUB" string remains under tests/ or libs/. The
# SKIP_RETURN_CODE 77 / SKIP_REGULAR_EXPRESSION "M1-STUB" properties below are
# retained only as the honest-skip convention (EGL-gated Storm tests exit 77
# without a GPU); they must never read as a failure. CTest names, tiers and
# gate labels are frozen here (docs/m1/interfaces.md §8).
# ---------------------------------------------------------------------------
# M1 plugin search path: M0 schema + imaging resources plus the usdGenShaders
# resources (Sdr discovery plugin lands in usdGenImaging; 10 §4.4).
# M-13: the dirs must not be joined into one ENVIRONMENT value. Doing so put
# CMake's ';' list separator inside the value ("PXR_PLUGINPATH_NAME=a;b;c=d"),
# which cut the plugin path after the first dir and mis-split the USD dirs.
# Joining with an explicit ':' only moved the bug to Windows, where the native
# separator *is* ';'. One ENVIRONMENT_MODIFICATION op per dir is separator-free
# and therefore correct on every host.
set(_usdgen_m1_pluginpath_dirs
    "${_usdgen_schema_resources}" "${_usdgen_imaging_resources}"
    "${_usdgen_shader_resources}"
    "${USDGEN_TEST_USD_DIR}/plugin/usd" "${USDGEN_TEST_USD_DIR}/lib/usd")
usdgen_test_pluginpath_ops(_usdgen_m1_env ${_usdgen_m1_pluginpath_dirs})

# M1 T0 engine tests (no Hydra, no stage — gate B-1 keeps them clean).
foreach(_t testUsdGenGraph testUsdGenKernelDeterminism testUsdGenWidth
           testUsdGenCpuFanout testUsdGenCpuReferenceMap testUsdGenCurveRagged
           testUsdGenClumpMotion
           testUsdGenCpuCurveSource testUsdGenLengthRaggedLayout testUsdGenSurfaceRootFrames
           testUsdGenSurfaceRootBindings testUsdGenSurfaceRootBindingBoundaries
           testUsdGenSurfaceBindingCache testUsdGenScatterDensity testUsdGenLimitScatter
           testUsdGenScatterTwoPass testUsdGenScatterSortSkips
           testUsdGenScatterDegenerateFan testUsdGenScatterTwoPassMult)
    usdgen_test_source(_usdgen_src ${_t}.cpp)
    add_executable(${_t} ${_usdgen_src})
    target_link_libraries(${_t} PRIVATE usdGenTestUtils)
    add_test(NAME ${_t} COMMAND ${_t})
    set_tests_properties(${_t} PROPERTIES SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endforeach()
# Reads publications: the curves' Hydra adapter registers its publisher.
target_link_libraries(testUsdGenScatterDensity PRIVATE usdGenHydraCurves)
set_tests_properties(testUsdGenGraph PROPERTIES LABELS "T0;gate:E-6")
set_tests_properties(testUsdGenKernelDeterminism PROPERTIES LABELS "T0;gate:E-8")
set_tests_properties(testUsdGenWidth PROPERTIES LABELS "T0")
set_tests_properties(testUsdGenCpuFanout PROPERTIES LABELS "T0;scheduling;dag")
set_tests_properties(testUsdGenClumpMotion PROPERTIES LABELS "T0;clump;motion;topology")
add_executable(benchUsdGenClumpMotion tests/internal/usdGenCurves/benchUsdGenClumpMotion.cpp)
target_link_libraries(benchUsdGenClumpMotion PRIVATE usdGenWithPrimitives usdGenUsd)
# Informational benchmark: run manually on an otherwise quiet host.
set_tests_properties(testUsdGenCpuReferenceMap PROPERTIES LABELS "T0;scheduling;reference;map")
set_tests_properties(testUsdGenCurveRagged PROPERTIES LABELS "T0;topology;cow;sources")
set_tests_properties(testUsdGenCpuCurveSource PROPERTIES LABELS "T0;topology;cow;sources")
set_tests_properties(testUsdGenLengthRaggedLayout PROPERTIES LABELS "T0;topology;cow;operators")
set_tests_properties(testUsdGenSurfaceRootFrames PROPERTIES LABELS "T0;sources;root-frames")
set_tests_properties(testUsdGenSurfaceRootBindings PROPERTIES LABELS "T0;sources;root-bindings;cow")
set_tests_properties(testUsdGenSurfaceRootBindingBoundaries PROPERTIES LABELS "T0;sources;root-bindings")
set_tests_properties(testUsdGenSurfaceBindingCache PROPERTIES LABELS "T0;sources;cache;cow")
set_tests_properties(testUsdGenScatterDensity PROPERTIES LABELS "T0;sources;scatter;density")
set_tests_properties(testUsdGenScatterTwoPass PROPERTIES LABELS "T0;sources;scatter;twopass")
target_link_libraries(testUsdGenScatterTwoPass PRIVATE usdGenMath)
set_tests_properties(testUsdGenScatterSortSkips PROPERTIES LABELS "T0;sources;scatter;sortskips")
target_link_libraries(testUsdGenScatterSortSkips PRIVATE usdGenMath)
set_tests_properties(testUsdGenScatterDegenerateFan PROPERTIES LABELS "T0;sources;scatter;degenfan")
target_link_libraries(testUsdGenScatterDegenerateFan PRIVATE usdGenMath)
set_tests_properties(testUsdGenScatterTwoPassMult PROPERTIES LABELS "T0;sources;scatter;twopassmult")
target_link_libraries(testUsdGenScatterTwoPassMult PRIVATE usdGenMath)

# Groom-import operators: whole-strand stylers (Braid, Twist, Trim, Frizz,
# TieClump), the extended Scatter/GuideInterpolate/Clump/Length/Collide/
# Smooth/Width/Resample modes, and the CardSource generator.
foreach(_t testUsdGenStrandUtil testUsdGenScatterModes testUsdGenGuideBlend
           testUsdGenClumpModes testUsdGenBraid testUsdGenTwistFrizz
           testUsdGenLengthTrim testUsdGenTieCollide testUsdGenCardSource)
    usdgen_test_source(_usdgen_src ${_t}.cpp)
    add_executable(${_t} ${_usdgen_src})
    target_link_libraries(${_t} PRIVATE usdGenTestUtils usdGenMath)
    add_test(NAME ${_t} COMMAND ${_t})
    set_tests_properties(${_t} PROPERTIES SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
                         LABELS "T0;operators;groom-import")
endforeach()
# Offline validation utility (test-only: the stage-sourced graph builder is
# fenced to test utilities): cooks every description of a stage on the CPU
# lane and prints per-description strand/CV counts, optionally baking them.
# It drives the private compiler and scheduler, so it needs the internals.
if(USDGEN_EXPORT_INTERNALS)
    add_executable(usdGenCookStats tools/cookStats.cpp)
    target_link_libraries(usdGenCookStats PRIVATE usdGenTestUtilsHd)
endif()

# M1 T0 perf benches (E-1/E-7 via --threads; E-2 sparse; carried in from
# prototypes/data-plane-benchmark, 11 §2.2).
foreach(_t benchUsdGenChain benchUsdGenSparse)
    usdgen_test_source(_usdgen_src ${_t}.cpp)
    add_executable(${_t} ${_usdgen_src})
    target_link_libraries(${_t} PRIVATE usdGenTestUtils)
    add_test(NAME ${_t} COMMAND ${_t})
    set_tests_properties(${_t} PROPERTIES SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endforeach()
set_tests_properties(benchUsdGenChain PROPERTIES LABELS "T0;perf;gate:E-1;gate:E-7")
set_tests_properties(benchUsdGenSparse PROPERTIES LABELS "T0;perf;gate:E-2")
add_executable(benchUsdGenRbfDeform tests/internal/usdGen/benchUsdGenRbfDeform.cpp)
target_link_libraries(benchUsdGenRbfDeform PRIVATE usdGenTestUtils)
if(USDGEN_ENABLE_CUDA)
    target_compile_definitions(benchUsdGenRbfDeform PRIVATE USDGEN_BENCH_CUDA)
    target_link_libraries(benchUsdGenRbfDeform PRIVATE usdGenCurvesCuda)
endif()
add_test(NAME benchUsdGenRbfDeform COMMAND benchUsdGenRbfDeform)
set_tests_properties(benchUsdGenRbfDeform PROPERTIES
    LABELS "T0;perf;rbf;deform"
    TIMEOUT 600 SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
if(USDGEN_ENABLE_VULKAN_RUNTIME)
    target_compile_definitions(benchUsdGenRbfDeform PRIVATE USDGEN_BENCH_VULKAN)
    target_link_libraries(benchUsdGenRbfDeform PRIVATE usdGenVulkanNative)
    target_sources(benchUsdGenRbfDeform PRIVATE
        "${_usdgen_vulkan_deform_evaluate_spv}" "${_usdgen_vulkan_deform_apply_spv}"
        "${_usdgen_vulkan_deform_evaluate_verify_spv}" "${_usdgen_vulkan_deform_evaluate_fill_spv}"
        "${_usdgen_vulkan_deform_evaluate_cached_spv}"
        "${_usdgen_vulkan_rbfvk_extent_spv}" "${_usdgen_vulkan_rbfvk_gram_spv}"
        "${_usdgen_vulkan_rbfvk_build_matrix_spv}" "${_usdgen_vulkan_rbfvk_lu_spv}"
        "${_usdgen_vulkan_rbfvk_rhs_spv}" "${_usdgen_vulkan_rbfvk_trisolve_spv}"
        "${_usdgen_vulkan_rbfvk_evaluate_spv}")
    add_test(NAME benchUsdGenRbfDeformVulkan COMMAND benchUsdGenRbfDeform
        --vulkan-spv "${CMAKE_CURRENT_BINARY_DIR}/vulkan")
    set_tests_properties(benchUsdGenRbfDeformVulkan PROPERTIES
        LABELS "T1;perf;vulkan;rbf;deform" RESOURCE_LOCK usdgen_gpu
        TIMEOUT 600 SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-;SYNC-HAZARD")
endif()
# Restructure before/after numbers (engine cook, compile, Hydra publish/pull,
# allocations). Informational: run manually, never gates on timing.
add_executable(benchUsdGenRestructure tests/internal/usdGen/benchUsdGenRestructure.cpp)
target_link_libraries(benchUsdGenRestructure PRIVATE usdGenTestUtilsHd)
add_executable(benchUsdGenSurfaceBindings tests/internal/usdGenSurface/benchUsdGenSurfaceBindings.cpp)
target_link_libraries(benchUsdGenSurfaceBindings PRIVATE usdGenTestUtils)
add_test(NAME benchUsdGenSurfaceBindings COMMAND benchUsdGenSurfaceBindings)
set_tests_properties(benchUsdGenSurfaceBindings PROPERTIES
    LABELS "T0;perf;sources;root-bindings;cache"
    TIMEOUT 600 SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

# Instancing/scattering per-stage timing + checksums (autoresearch
# instscatter). GPU stages compile in only when their backend does, and skip
# at runtime without a device (CUDA) or --vk-spv (Vulkan dispatch), so the
# registered run stays green on CPU-only CI.
add_executable(benchUsdGenInstanceScatter tests/internal/usdGen/benchUsdGenInstanceScatter.cpp)
target_link_libraries(benchUsdGenInstanceScatter PRIVATE usdGenTestUtilsHd)
target_include_directories(benchUsdGenInstanceScatter PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests")
if(USDGEN_ENABLE_CUDA)
    target_link_libraries(benchUsdGenInstanceScatter PRIVATE usdGenCurvesCuda)
    target_compile_definitions(benchUsdGenInstanceScatter PRIVATE USDGEN_ENABLE_CUDA)
endif()
if(USDGEN_ENABLE_VULKAN_RUNTIME)
    target_link_libraries(benchUsdGenInstanceScatter PRIVATE usdGenVulkanNative)
    target_compile_definitions(benchUsdGenInstanceScatter PRIVATE USDGEN_ENABLE_VULKAN_RUNTIME)
endif()
add_test(NAME benchUsdGenInstanceScatter COMMAND benchUsdGenInstanceScatter)
set_tests_properties(benchUsdGenInstanceScatter PROPERTIES
    LABELS "T1;perf;instance;scatter" TIMEOUT 900 RESOURCE_LOCK usdgen_gpu
    SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
