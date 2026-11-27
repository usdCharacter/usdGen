# Restructure gates (ctest -L gate). Each holds today's count as a ceiling
# that may only go down; the restructure's target for each is zero.
#
#   testUsdGenNoLocks          no lock types or TBB anywhere under libs/
#   testUsdGenNoGuardedStatics no function-local static with dynamic
#                              initialisation under libs/ (MSVC guards
#                              those with a lock when contended)
#   testUsdGenNoPxrIncludes    no pxr/ include below the hosts
#   testUsdGenNoPxrLink_*      no pxr library linked below the hosts
#   testUsdGenWarmAllocations  no heap allocation in a warm cook
#
# After removing hits, refresh a baseline with, for example:
#   cmake -DWRITE=ON -DSOURCE_DIR=$PWD -DNAME=no-locks -DDIRS=libs \
#         "-DPATTERN=<pattern>" -DBASELINE=$PWD/cmake/gates/noLocks.txt \
#         -P cmake/gates/RatchetScan.cmake
# (add "-DEXCLUDE=<exclude>" for testUsdGenNoGuardedStatics)

set(USDGEN_LOCK_PATTERN
    "std::(recursive_|timed_|shared_|recursive_timed_|shared_timed_)?mutex|std::condition_variable|std::(lock_guard|unique_lock|scoped_lock|shared_lock)|std::(once_flag|call_once)|std::(counting_semaphore|binary_semaphore|latch|barrier)|std::(future|promise|shared_future|packaged_task)|std::atomic_(load|store|exchange|compare_exchange_[a-z]+)(_explicit)?[ ]*\\(|atomic<std::shared_ptr|pthread_(mutex|cond|rwlock)|CRITICAL_SECTION|SRWLOCK|Tf(Spin|BigRW)Mutex|tbb::|#[ ]*include[ ]*[<\"](oneapi/)?tbb[/.]|oneapi/tbb")
# Below the hosts: base/ and core/ (engine, backends, primitives).
set(USDGEN_BELOW_HOSTS_DIRS "libs/base;libs/core")
# An indented "static <type> <name> = / {" that is not constexpr,
# thread_local or inline, and whose type is not an atomic, a fundamental
# type or a std::array of one (constant-initialised in practice). Misses
# "static T x(args);" (indistinguishable from a member declaration) and
# fundamental-typed statics with a dynamic initialiser.
set(USDGEN_GUARDED_STATIC_PATTERN
    "^[ ]+static[ ]+(const[ ]+)?[A-Za-z_][A-Za-z0-9_:<>,* &]*[ *&]+[A-Za-z_][A-Za-z0-9_]*([[][^=]*[]])?[ ]*(=|{)")
set(_usdgen_fundamental "(unsigned[ ]+|signed[ ]+)?(bool|char|short|int|long|float|double|size_t|u?int[0-9]+_t)")
set(USDGEN_GUARDED_STATIC_EXCLUDE
    "static[ ]+(constexpr|thread_local|inline)[ ]|static[ ]+(const[ ]+)?(std::atomic<|${_usdgen_fundamental}[ ]|std::array<${_usdgen_fundamental}[ ,])")
set(USDGEN_PXR_INCLUDE_PATTERN "#[ ]*include[ ]*[<\"]pxr/")

add_test(NAME testUsdGenNoLocks
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-locks
        -DDIRS=libs
        "-DPATTERN=${USDGEN_LOCK_PATTERN}"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noLocks.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenNoLocks PROPERTIES LABELS "T0;build;gate;gate:no-locks")

add_test(NAME testUsdGenNoGuardedStatics
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-guarded-statics
        -DDIRS=libs
        "-DPATTERN=${USDGEN_GUARDED_STATIC_PATTERN}"
        "-DEXCLUDE=${USDGEN_GUARDED_STATIC_EXCLUDE}"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noGuardedStatics.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenNoGuardedStatics PROPERTIES LABELS "T0;build;gate;gate:no-guarded-statics")

# Comma-separated on the command line; RatchetScan splits it.
string(REPLACE ";" "," _usdgen_below_dirs "${USDGEN_BELOW_HOSTS_DIRS}")
add_test(NAME testUsdGenNoPxrIncludes
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-pxr-below-hosts
        "-DDIRS=${_usdgen_below_dirs}"
        "-DPATTERN=${USDGEN_PXR_INCLUDE_PATTERN}"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noPxrIncludes.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenNoPxrIncludes PROPERTIES LABELS "T0;build;gate;gate:no-pxr")

# Step 7: no Sdf type or sdf include below the hosts (SdfPath is a host
# NodeKey map, Sdf value types are usdGenExpr's closed table). Empty baseline.
add_test(NAME testUsdGenNoSdfBelowHosts
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-sdf-below-hosts
        "-DDIRS=${_usdgen_below_dirs}"
        "-DPATTERN=Sdf[A-Z_]|pxr/usd/sdf/"
        "-DEXCLUDE=^[ ]*//"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noSdfBelowHosts.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenNoSdfBelowHosts PROPERTIES LABELS "T0;build;gate;gate:no-pxr")

# Step 10 (gate 10a): no pxr/usd or pxr/imaging include below the hosts
# (limitSurface.cpp calls OpenSubdiv Far directly). Empty baseline.
add_test(NAME testUsdGenNoPxrUsdImagingBelowHosts
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-pxr-usd-imaging-below-hosts
        "-DDIRS=${_usdgen_below_dirs}"
        "-DPATTERN=#[ ]*include[ ]*[<\"]pxr/(usd|imaging)/"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noPxrUsdImagingBelowHosts.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenNoPxrUsdImagingBelowHosts PROPERTIES LABELS "T0;build;gate;gate:no-pxr")

# 8.6f: the runtimes and primitives join (baselines noPxrLink_<t>.txt).
foreach(_t usdGen usdGenMath usdGenCuda usdGenVulkan usdGenSurface usdGenExpr usdGenCurves
        usdGenCurvesCuda usdGenCurvesVulkan)
    if(NOT TARGET ${_t})
        continue()
    endif()
    add_test(NAME testUsdGenNoPxrLink_${_t}
        COMMAND ${CMAKE_COMMAND}
            -DNAME=no-pxr-link-below-hosts
            -DTARGET=${_t}
            "-DLINKS=$<JOIN:$<TARGET_PROPERTY:${_t},LINK_LIBRARIES>,\;>"
            "-DPATTERN=^(arch|tf|gf|vt|js|trace|work|plug|ts|pegtl|sdf|ar|kind|pcp|usd[A-Za-z]*|hd[A-Za-z]*|hgi[A-Za-z]*|hio|glf|garch|cameraUtil|pxOsd|geomUtil|hf)$"
            -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noPxrLink_${_t}.txt
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/CheckLinkRatchet.cmake)
    set_tests_properties(testUsdGenNoPxrLink_${_t} PROPERTIES LABELS "T0;build;gate;gate:no-pxr")
endforeach()

# Link gates 8d (step 8.6f): a backend runtime links usdGen and its SDK only;
# usdGen links no backend, primitive or GPU SDK; hosts link no backend.
# 9c (step 9.2): usdGenImaging and the curves' Hydra adapter link no
# usdGenCurves either (publication lives in usdGenHydraCurves).
# usdGenPomade (app layer) may link usdGenCuda (one block cache, D12).
set(_usdgen_link_gate_rows
    "usdGenCuda|ALLOW|^(usdGen|CUDA::(cudart|cuda_driver|cusolver)|Threads::Threads|Synchronization|dl|arch|tf|gf|vt)$"
    "usdGenVulkan|ALLOW|^(usdGen|Vulkan::Vulkan|Threads::Threads|Synchronization|dl|arch|tf|gf|vt)$"
    "usdGen|DENY|^(usdGen(Cuda|Vulkan|Curves|Surface|Expr|Gpu|WithPrimitives|_ptex|_seexpr|ExprFrontend)|CUDA::.*|Vulkan::.*|pxOsd|hio)$"
    "usdGenImaging|DENY|^(usdGen(Cuda|Vulkan|Gpu|CurvesCuda|CurvesVulkan|Curves|WithPrimitives))$"
    "usdGenHydraCurves|DENY|^(usdGen(Cuda|Vulkan|Gpu|CurvesCuda|CurvesVulkan|Curves|WithPrimitives))$")
foreach(_row IN LISTS _usdgen_link_gate_rows)
    string(REGEX MATCH "^([^|]+)[|]([A-Z]+)[|](.*)$" _m "${_row}")
    set(_t "${CMAKE_MATCH_1}")
    set(_kind "${CMAKE_MATCH_2}")
    set(_re "${CMAKE_MATCH_3}")
    if(NOT TARGET ${_t})
        continue()
    endif()
    add_test(NAME testUsdGenNoDisallowedLink_${_t}
        COMMAND ${CMAKE_COMMAND}
            -DNAME=link-allow
            -DTARGET=${_t}
            "-DLINKS=$<JOIN:$<TARGET_PROPERTY:${_t},LINK_LIBRARIES>,\;>"
            "-D${_kind}=${_re}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/CheckLinkAllow.cmake)
    set_tests_properties(testUsdGenNoDisallowedLink_${_t} PROPERTIES LABELS "T0;build;gate;gate:link")
endforeach()

# Step 4 registry gates (design-step4 section 6). Today's counts as ceilings;
# each step-4 commit lowers its own files' rows.
#
#   testUsdGenNoOpNameLiterals    no op type name literal in the engine
#                                 outside an op's registration line
#   testUsdGenNoTokenIntern       no TfToken built from text or a string:
#                                 the interner is the one door
#   testUsdGenEngineNoOpCoupling  no op header and no cast to an op class
#                                 outside ops/
#
# Gate 1's exclude omits the registration line's trailing ';' (a ';' would
# split the ctest argument).
set(USDGEN_OP_NAME_LITERAL_PATTERN
    "TfToken[ ]*[({][ ]*\"UsdGen[A-Z]|\"UsdGen[A-Z][A-Za-z]*\"")
set(USDGEN_OP_NAME_LITERAL_EXCLUDE
    "^[ ]*//|^[ ]*[A-Za-z_][A-Za-z0-9_]*[.]name[ ]*=[ ]*\"UsdGen[A-Z][A-Za-z]*\"")
set(USDGEN_TOKEN_INTERN_PATTERN
    "TfToken[ ]*[({][ ]*[^ )}]|TfToken[ ]+[A-Za-z_][A-Za-z0-9_]*[ ]*[({][ ]*\"")
set(USDGEN_OP_COUPLING_PATTERN
    "#[ ]*include[ ]*\"usdGen/ops/|(dynamic|static)_cast<[ ]*(const[ ]+)?UsdGen[A-Za-z]*Op[ ]*(const[ ]*)?[*&]")

add_test(NAME testUsdGenNoOpNameLiterals
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-op-name-literals
        -DDIRS=libs/core
        "-DPATTERN=${USDGEN_OP_NAME_LITERAL_PATTERN}"
        "-DEXCLUDE=${USDGEN_OP_NAME_LITERAL_EXCLUDE}"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noOpNameLiterals.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
add_test(NAME testUsdGenNoTokenIntern
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-token-intern
        -DDIRS=libs/core
        "-DPATTERN=${USDGEN_TOKEN_INTERN_PATTERN}"
        "-DEXCLUDE=^[ ]*//"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noTokenIntern.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
add_test(NAME testUsdGenEngineNoOpCoupling
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=engine-no-op-coupling
        -DDIRS=libs/core/usdGen,libs/core/backends
        # The curves' kernel-row files leave the backends at 8.6; then no exclude.
        "-DEXCLUDE_PATHS=^libs/core/backends/usdGenVulkan/usdGenVulkan/registerCurvesVulkan[.]cpp$"
        "-DPATTERN=${USDGEN_OP_COUPLING_PATTERN}"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/engineNoOpCoupling.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenNoOpNameLiterals testUsdGenNoTokenIntern
    testUsdGenEngineNoOpCoupling PROPERTIES LABELS "T0;build;gate;gate:registries")

# Step 5.6: the engine works on element sets; the curve builder buffer, its
# chunk and tile types and curveBuffer.h stay out of libs/core/usdGen. The
# baseline is empty since 6.3 (publication behind UsdGenPublisherDef; hosts
# read usdGenCurves/publish until step 9). Comment lines are no hit.
add_test(NAME testUsdGenEngineNoCurveBuffer
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=engine-no-curve-buffer
        -DDIRS=libs/core/usdGen
        "-DPATTERN=curveBuffer[.]h|UsdGenCurveBuffer|UsdGenChunkDesc|UsdGenTileView"
        "-DEXCLUDE=^[ ]*//"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/engineNoCurveBuffer.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenEngineNoCurveBuffer PROPERTIES LABELS "T0;build;gate;gate:engine-generic")

# Step 6.5: the engine (libs/core/usdGen) names no op, primitive word,
# basisCurves, render namespace, or CUDA/Vulkan/primitive header. Comments
# count. CMake regex has no word boundary, hence the explicit classes. The
# baseline holds only the backend plumbing step 8 moves out; it subsumes
# testUsdGenEngineNoCurveBuffer.
set(USDGEN_PRIMITIVE_IN_ENGINE_PATTERN
    "UsdGen(Scatter|Grow|Noise|Length|Width|WidthBlend|CurveSource|Deform|ReferenceSource|Clump|GuideInterpolate|Curl|Bend|Wave|Part|Direction|Smooth|Resample|Scale|Straighten|Displace|SculptLayer|Wind|ExprOp|Freeze|Collide|Braid|Twist|Trim|Frizz|TieClump|CardSource)([^A-Za-z]|$)|[Cc]urves?|(^|[^A-Za-z])CVs?([^A-Za-z]|$)|[Cc]v(s|Count|Offsets|Span)|[Ss]trands?|[Ff]eathers?|(^|[^A-Za-z])[Ss]cales([^A-Za-z]|$)|basisCurves|__usdGenRender|#[ ]*include[ ]*[<\"](cuda|usdGenCuda/|usdGenVulkan/|usdGenCurves/|usdGenSurface/|usdGenExpr/|usdGen/(cuda|gpu/|vulkan/|ops/|expressions/))")
add_test(NAME testUsdGenNoPrimitiveInEngine
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-primitive-in-engine
        -DDIRS=libs/core/usdGen
        "-DPATTERN=${USDGEN_PRIMITIVE_IN_ENGINE_PATTERN}"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noPrimitiveInEngine.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenNoPrimitiveInEngine PROPERTIES LABELS "T0;build;gate;gate:engine-generic")

# Step 8.0 backend gates (design-step8-10 section 7). Today's counts.
#   testUsdGenNoBackendInEngine  the engine names no backend (comments count);
#                                empty at 8.4
#   testUsdGenNoBlockingGpuWait  no blocking GPU wait in backends/primitives;
#                                after 8.8 only "// shutdown wait" sites remain
#   testUsdGenTimelineSubmitOnly every Vulkan submit through the queue
#                                timeline; 8.8e leaves queueTimeline.cpp only
add_test(NAME testUsdGenNoBackendInEngine
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-backend-in-engine
        -DDIRS=libs/core/usdGen
        "-DPATTERN=[Cc]uda|CUDA|[Vv]ulkan|VULKAN|[Mm]etal([^A-Za-z]|$)|USDGEN_ENABLE_"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noBackendInEngine.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenNoBackendInEngine PROPERTIES LABELS "T0;build;gate;gate:engine-generic")

add_test(NAME testUsdGenNoBlockingGpuWait
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-blocking-gpu-wait
        -DDIRS=libs/core/backends,libs/core/primitives
        "-DPATTERN=CudaFenceStream[ ]*[(]|cuda(Stream|Event|Device)Synchronize|vkWaitForFences|vkQueueWaitIdle|vkDeviceWaitIdle"
        "-DEXCLUDE=^[ ]*//"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noBlockingGpuWait.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
add_test(NAME testUsdGenTimelineSubmitOnly
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=timeline-submit-only
        -DDIRS=libs/core
        "-DPATTERN=vkQueueSubmit[ ]*[(]|vkCreateFence[ ]*[(]"
        "-DEXCLUDE=^[ ]*//"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/timelineSubmitOnly.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenNoBlockingGpuWait testUsdGenTimelineSubmitOnly
    PROPERTIES LABELS "T0;build;gate;gate:gpu-async")


# Gate 9b (step 9.2): the Hydra host includes no curve primitive header (it
# reads curve sets through usdGen/api; publication is usdGenHydraCurves').
# Empty baseline.
add_test(NAME testUsdGenHydraNoCurvesInclude
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=hydra-no-curves-include
        -DDIRS=libs/hosts/usdGenHydra
        "-DPATTERN=#[ ]*include[ ]*[<\"](usdGenCurves/|usdGen/(rbfAnimation|furOcclusion|valuePreview|curveBuffer|curveSet|curveSetInput|curvePublication|curvePublisher)[.]h)"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/hydraNoCurvesInclude.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenHydraNoCurvesInclude PROPERTIES LABELS "T0;build;gate;gate:hydra")

# Gate 9a (step 9.3): the engine (libs/core/usdGen) carries no Hydra
# vocabulary (presentation, primvar and curve-basis words). "pinned" is
# matched as a literal and Material only as a word: the epoch's pinned slots
# and the inputs' materialize hook are not Hydra words. Empty baseline.
add_test(NAME testUsdGenNoHydraInEngine
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-hydra-in-engine
        -DDIRS=libs/core/usdGen
        "-DPATTERN=primvars|displayColor|[Pp]urpose|[Vv]isibility([^(]|$)|(^|[^A-Za-z])[Mm]aterial([^A-Za-z]|$)|allPurpose|scalpShadow|hairT|hairId|bakeColor|velocities|Hd[A-Z]|[Ss]torm|bspline|catmullRom|\"pinned\"|nonperiodic|cubic"
        "-DEXCLUDE=^[ ]*//"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noHydraInEngine.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenNoHydraInEngine PROPERTIES LABELS "T0;build;gate;gate:hydra")

add_executable(testUsdGenWarmAllocations tests/testUsdGenWarmAllocations.cpp)
target_link_libraries(testUsdGenWarmAllocations PRIVATE usdGen usdGenUsd)
add_test(NAME testUsdGenWarmAllocations COMMAND testUsdGenWarmAllocations)
set_tests_properties(testUsdGenWarmAllocations PROPERTIES
    LABELS "T0;perf;gate;gate:zero-alloc" RUN_SERIAL TRUE)

# Directory move (M0): no include through the old spellings that
# cmake/IncludeShim.cmake forwards. Step 12 empties the baseline and deletes
# the shim; a file without a baseline line must have zero hits.
string(REPLACE "." "[.]" _usdgen_shim_alts "${USDGEN_SHIM_OLD_SPELLINGS}")
string(REPLACE ";" "|" _usdgen_shim_alts "${_usdgen_shim_alts}")
add_test(NAME testUsdGenNoShimIncludes
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DNAME=no-shim-includes
        -DDIRS=libs,tests,testutils,plugin
        "-DPATTERN=^[ ]*#[ ]*include[ ]*[<\"](${_usdgen_shim_alts})[>\"]"
        -DBASELINE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/noShimIncludes.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gates/RatchetScan.cmake)
set_tests_properties(testUsdGenNoShimIncludes PROPERTIES LABELS "T0;build;gate;gate:no-shim")
# Gate 8e (step 8.3), checked at configure: one CUDA block cache. Only
# usdGenCuda compiles the runtime's gpu/deviceBuffer.cu, and no STATIC
# target holds a .cu file of the CUDA runtime (each would be a copy).
get_property(_usdgen_gate_targets DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY BUILDSYSTEM_TARGETS)
foreach(_usdgen_gate_target IN LISTS _usdgen_gate_targets)
    get_target_property(_usdgen_gate_type ${_usdgen_gate_target} TYPE)
    if(_usdgen_gate_type STREQUAL "INTERFACE_LIBRARY" OR _usdgen_gate_type STREQUAL "UTILITY")
        continue()
    endif()
    get_target_property(_usdgen_gate_sources ${_usdgen_gate_target} SOURCES)
    if(NOT _usdgen_gate_sources)
        continue()
    endif()
    foreach(_usdgen_gate_source IN LISTS _usdgen_gate_sources)
        if(NOT _usdgen_gate_source MATCHES "backends/usdGenCuda/.*[.]cu$")
            continue()
        endif()
        if(_usdgen_gate_source MATCHES "deviceBuffer[.]cu$" AND
           NOT _usdgen_gate_target STREQUAL "usdGenCuda")
            message(FATAL_ERROR "usdGen gate 8e: ${_usdgen_gate_target} compiles "
                "${_usdgen_gate_source}; only usdGenCuda may (one block cache)")
        endif()
        if(_usdgen_gate_type STREQUAL "STATIC_LIBRARY")
            message(FATAL_ERROR "usdGen gate 8e: STATIC ${_usdgen_gate_target} holds "
                "the CUDA runtime source ${_usdgen_gate_source}")
        endif()
    endforeach()
endforeach()
