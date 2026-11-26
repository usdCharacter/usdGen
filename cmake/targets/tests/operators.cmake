# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Tests: output guards, registries, operators, expressions, maps, schema. Included from the root CMakeLists.txt inside if(USDGEN_BUILD_TESTS).

# Restructure step 4 guards: the CPU lane's output of record and the op
# contract snapshot. Both compare against tests/golden/*.txt byte for byte.
foreach(_t testUsdGenOutputGoldens testUsdGenOpContracts)
    add_executable(${_t} tests/${_t}.cpp tests/golden/goldenGraphs.cpp)
    target_link_libraries(${_t} PRIVATE usdGenTestUtilsHd)
    target_include_directories(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_compile_definitions(${_t} PRIVATE
        USDGEN_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        USDGEN_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden")
    add_test(NAME ${_t} COMMAND ${_t})
    set_tests_properties(${_t} PROPERTIES
        ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T0;gate;gate:golden"
        TIMEOUT 900)
endforeach()
# The output golden belongs to one toolchain; another skips with 77.
set_tests_properties(testUsdGenOutputGoldens PROPERTIES SKIP_RETURN_CODE 77)

# Restructure 5.0: published geometry of record over frames 0/1/12 and a
# value and topology edit (tests/golden/cookDigests.<os>-<cc>.txt; a
# toolchain without its own file skips with 77).
add_executable(testUsdGenCookGoldens tests/testUsdGenCookGoldens.cpp tests/golden/goldenGraphs.cpp)
target_link_libraries(testUsdGenCookGoldens PRIVATE usdGenTestUtilsHd)
target_include_directories(testUsdGenCookGoldens PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_definitions(testUsdGenCookGoldens PRIVATE
    USDGEN_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    USDGEN_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden")
add_test(NAME testUsdGenCookGoldens COMMAND testUsdGenCookGoldens)
set_tests_properties(testUsdGenCookGoldens PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T0;gate;gate:golden"
    SKIP_RETURN_CODE 77 TIMEOUT 1800)

# Restructure 5.3a/5.3f: ops over buffer-backed set views (the old op
# family and USDGEN_BRIDGE_VERIFY are gone).
add_executable(testUsdGenOpBridge tests/testUsdGenOpBridge.cpp)
target_link_libraries(testUsdGenOpBridge PRIVATE usdGenTestUtils)
add_test(NAME testUsdGenOpBridge COMMAND testUsdGenOpBridge)
set_tests_properties(testUsdGenOpBridge PROPERTIES LABELS "T0;bridge")

# Step 4 registries: handles, lookups, freeze, interner counters.
add_executable(testUsdGenRegistries tests/testUsdGenRegistries.cpp tests/golden/goldenGraphs.cpp)
target_link_libraries(testUsdGenRegistries PRIVATE usdGenTestUtils)
target_include_directories(testUsdGenRegistries PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_test(NAME testUsdGenRegistries COMMAND testUsdGenRegistries)
set_tests_properties(testUsdGenRegistries PROPERTIES LABELS "T0;registries")

# Step 6.4: the primitives are DLLs that usdGen loads; links usdGen only.
add_executable(testUsdGenRegistrarLoad tests/testUsdGenRegistrarLoad.cpp)
target_link_libraries(testUsdGenRegistrarLoad PRIVATE usdGen)
add_dependencies(testUsdGenRegistrarLoad usdGenSurface usdGenExpr usdGenCurves usdGenSchema)
target_compile_definitions(testUsdGenRegistrarLoad PRIVATE
    "USDGEN_TEST_SURFACE=\"$<TARGET_FILE:usdGenSurface>\""
    "USDGEN_TEST_EXPR=\"$<TARGET_FILE:usdGenExpr>\""
    "USDGEN_TEST_CURVES=\"$<TARGET_FILE:usdGenCurves>\""
    "USDGEN_TEST_NOT_EXTENSION=\"$<TARGET_FILE:usdGenSchema>\"")
add_test(NAME testUsdGenRegistrarLoad COMMAND testUsdGenRegistrarLoad load)
add_test(NAME testUsdGenRegistrarLoadDefaults COMMAND testUsdGenRegistrarLoad defaults)
set_tests_properties(testUsdGenRegistrarLoad testUsdGenRegistrarLoadDefaults PROPERTIES
    LABELS "T0;registries"
    ENVIRONMENT_MODIFICATION "USDGEN_EXTENSIONS=unset:")

# Step 6.5: the engine builds and runs without any primitive (links usdGen only).
add_executable(testUsdGenEngineBare tests/core/testUsdGenEngineBare.cpp)
target_link_libraries(testUsdGenEngineBare PRIVATE usdGen)
add_test(NAME testUsdGenEngineBare COMMAND testUsdGenEngineBare)
set_tests_properties(testUsdGenEngineBare PROPERTIES
    LABELS "T0;registries"
    ENVIRONMENT_MODIFICATION "USDGEN_EXTENSIONS=unset:")

# Step 4 gate 4: an exec-phase intern fails the cook (warns with checks off).
add_executable(testUsdGenTokenChecks tests/testUsdGenTokenChecks.cpp)
target_link_libraries(testUsdGenTokenChecks PRIVATE usdGenTestUtils)
add_test(NAME testUsdGenTokenChecks COMMAND testUsdGenTokenChecks)
add_test(NAME testUsdGenTokenChecksOff COMMAND testUsdGenTokenChecks off)
set_tests_properties(testUsdGenTokenChecksOff PROPERTIES ENVIRONMENT "USDGEN_TOKEN_CHECKS=0")
set_tests_properties(testUsdGenTokenChecks testUsdGenTokenChecksOff PROPERTIES
    LABELS "T0;registries;gate;gate:registries")

# UsdGenDeform on the CPU lane: hair deformed by animated curves (RBF).
add_executable(testUsdGenRbfDeform tests/testUsdGenRbfDeform.cpp)
target_link_libraries(testUsdGenRbfDeform PRIVATE usdGenTestUtilsHd)
target_compile_definitions(testUsdGenRbfDeform PRIVATE
    USDGEN_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME testUsdGenRbfDeform COMMAND testUsdGenRbfDeform)
set_tests_properties(testUsdGenRbfDeform PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T1;examples;deform" TIMEOUT 300)
add_test(NAME testUsdGenRbfImmediatePose COMMAND testUsdGenRbfDeform --gpu-pose)
set_tests_properties(testUsdGenRbfImmediatePose PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}"
    ENVIRONMENT "USDGEN_STORM_GPU_DEFORM=1;USDGEN_SCALP_SHADOW=0"
    LABELS "T1;deform;publication" TIMEOUT 60)

# usdGen:preview:* — the viewport value preview through the groom scene index.
add_executable(testUsdGenValuePreview tests/testUsdGenValuePreview.cpp)
target_link_libraries(testUsdGenValuePreview PRIVATE usdGenTestUtilsHd)
target_compile_definitions(testUsdGenValuePreview PRIVATE
    USDGEN_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME testUsdGenValuePreview COMMAND testUsdGenValuePreview)
set_tests_properties(testUsdGenValuePreview PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T1;expressions;examples;preview" TIMEOUT 300)

add_executable(testUsdGenExpressionTransport tests/testUsdGenExpressionTransport.cpp)
target_link_libraries(testUsdGenExpressionTransport PRIVATE usdGenTestUtilsHd)
target_compile_definitions(testUsdGenExpressionTransport PRIVATE
    USDGEN_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME testUsdGenExpressionTransport COMMAND testUsdGenExpressionTransport)
set_tests_properties(testUsdGenExpressionTransport PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T1;expressions" TIMEOUT 120)

add_executable(testUsdGenSessionIsolation tests/testUsdGenSessionIsolation.cpp)
target_link_libraries(testUsdGenSessionIsolation PRIVATE usdGenImaging)
add_test(NAME testUsdGenSessionIsolation COMMAND testUsdGenSessionIsolation)
set_tests_properties(testUsdGenSessionIsolation PROPERTIES
    LABELS "T1;sessions" TIMEOUT 120)

foreach(_expr_test ExpressionContext ExpressionBindings ExpressionFrontend ExpressionIr ExpressionCpuEval)
    add_executable(testUsdGen${_expr_test} tests/testUsdGen${_expr_test}.cpp)
    target_link_libraries(testUsdGen${_expr_test} PRIVATE usdGenWithPrimitives usdGenUsd)
    add_test(NAME testUsdGen${_expr_test} COMMAND testUsdGen${_expr_test})
    set_tests_properties(testUsdGen${_expr_test} PROPERTIES
        LABELS "T0;expressions" TIMEOUT 120)
endforeach()
# geoSampler()/ptex(): links usdGen_ptex itself to author its fixture .ptx.
add_executable(testUsdGenExpressionSamplers tests/testUsdGenExpressionSamplers.cpp)
target_link_libraries(testUsdGenExpressionSamplers PRIVATE usdGenWithPrimitives usdGenUsd usdGen_ptex)
add_test(NAME testUsdGenExpressionSamplers COMMAND testUsdGenExpressionSamplers)
set_tests_properties(testUsdGenExpressionSamplers PROPERTIES
    LABELS "T0;expressions;maps" TIMEOUT 120)
# Curl/Bend/Wave/Part stylers plus the primitive-domain Grow lift: registry
# contracts, expression-target validation and deterministic CPU cooks.
add_executable(testUsdGenCurlBendWavePart tests/testUsdGenCurlBendWavePart.cpp)
target_link_libraries(testUsdGenCurlBendWavePart PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenCurlBendWavePart COMMAND testUsdGenCurlBendWavePart)
set_tests_properties(testUsdGenCurlBendWavePart PROPERTIES
    LABELS "T0;operators;expressions" TIMEOUT 120)
# Direction/Smooth/Resample/Scale/Straighten/Displace stylers: same shape,
# one target per operator.
# SculptLayer/Wind/ExprOp/Freeze/Collide stylers: same shape, appended to
# the operator foreach above.
foreach(_styler_test Direction Smooth Resample Scale Straighten Displace SculptLayer Wind ExprOp Freeze Collide)
    add_executable(testUsdGen${_styler_test} tests/testUsdGen${_styler_test}.cpp)
    target_link_libraries(testUsdGen${_styler_test} PRIVATE usdGenWithPrimitives usdGenUsd)
    add_test(NAME testUsdGen${_styler_test} COMMAND testUsdGen${_styler_test})
    set_tests_properties(testUsdGen${_styler_test} PROPERTIES
        LABELS "T0;operators;expressions" TIMEOUT 120)
endforeach()
target_compile_definitions(testUsdGenCollide PRIVATE
    USDGEN_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
# testUsdGenScale recomputes UsdGenDraw01 expectations; usdGenMath is PRIVATE
# to usdGen (N-7), so the test links it explicitly (CudaCurveGrowSession
# precedent).
target_link_libraries(testUsdGenScale PRIVATE usdGenMath)
# Ptex face ids/coordinates and the sampler. Links usdGen_ptex itself so the
# test can author its fixture .ptx with PtexWriter.
add_executable(testUsdGenPtexMap tests/testUsdGenPtexMap.cpp)
target_link_libraries(testUsdGenPtexMap PRIVATE usdGenWithPrimitives usdGenUsd usdGen_ptex)
add_test(NAME testUsdGenPtexMap COMMAND testUsdGenPtexMap)
set_tests_properties(testUsdGenPtexMap PROPERTIES LABELS "T0;maps" TIMEOUT 120)
# Paintable per-surface attribute maps + stroke accumulation (the host-groomer
# brush model over per-face storage). Engine-only, no USD dependency.
add_executable(testUsdGenAttributeMap tests/testUsdGenAttributeMap.cpp)
target_link_libraries(testUsdGenAttributeMap PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenAttributeMap COMMAND testUsdGenAttributeMap)
set_tests_properties(testUsdGenAttributeMap PROPERTIES LABELS "T0;maps" TIMEOUT 120)
# Internal digest contract (determinism + sensitivity, never golden values).
add_executable(testUsdGenDigest tests/testUsdGenDigest.cpp)
target_link_libraries(testUsdGenDigest PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenDigest COMMAND testUsdGenDigest)
set_tests_properties(testUsdGenDigest PROPERTIES LABELS "T0;digest" TIMEOUT 120)
# Morton key contract: magic-mask interleave proven against the loop form.
add_executable(testUsdGenMorton tests/testUsdGenMorton.cpp)
target_link_libraries(testUsdGenMorton PRIVATE usdGenMath)
add_test(NAME testUsdGenMorton COMMAND testUsdGenMorton)
set_tests_properties(testUsdGenMorton PROPERTIES LABELS "T0;morton" TIMEOUT 120)
# Painted attribute maps: the .ptx bake and the async bake-to-stage worker.
# Engine-only (threads + filesystem, no USD/Hydra), like the map it bakes.
add_executable(testUsdGenAttributeBake tests/testUsdGenAttributeBake.cpp)
target_link_libraries(testUsdGenAttributeBake PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenAttributeBake COMMAND testUsdGenAttributeBake)
set_tests_properties(testUsdGenAttributeBake PROPERTIES LABELS "T1;maps" TIMEOUT 120)
# Authored maps into groom cooking/instancing: keep/drop + prototype index
# per strand root, CPU-only and separate from Pomade's region-map flow.
add_executable(testUsdGenAttributeInstance tests/testUsdGenAttributeInstance.cpp)
target_link_libraries(testUsdGenAttributeInstance PRIVATE usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenAttributeInstance COMMAND testUsdGenAttributeInstance)
set_tests_properties(testUsdGenAttributeInstance PROPERTIES LABELS "T0;maps" TIMEOUT 120)
# The live paint-preview overlay: app -> Hydra displayColor for the brushed
# surface and the groom, over the real UsdImaging chain.
add_executable(testUsdGenAttributePreview tests/testUsdGenAttributePreview.cpp)
target_link_libraries(testUsdGenAttributePreview PRIVATE usdGenTestUtilsHd)
add_test(NAME testUsdGenAttributePreview COMMAND testUsdGenAttributePreview)
set_tests_properties(testUsdGenAttributePreview PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T1;maps;preview" TIMEOUT 120)
# The attribute-brush C ABI: pick/footprint, dab kernel with hardness and
# erase, the cross-face corner smooth, commit == working, the 100k-face
# per-move cost, and the global preview overlay.
add_executable(testUsdGenBrushApi tests/testUsdGenBrushApi.cpp)
target_link_libraries(testUsdGenBrushApi PRIVATE usdGenTestUtilsHd)
add_test(NAME testUsdGenBrushApi COMMAND testUsdGenBrushApi)
set_tests_properties(testUsdGenBrushApi PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T1;maps;preview;brush" TIMEOUT 120)
add_executable(testUsdGenSurfaceCageInterpolate tests/testUsdGenSurfaceCageInterpolate.cpp)
target_link_libraries(testUsdGenSurfaceCageInterpolate PRIVATE usdGenWithPrimitives usdGenUsd usdGen_ptex)
add_test(NAME testUsdGenSurfaceCageInterpolate COMMAND testUsdGenSurfaceCageInterpolate)
set_tests_properties(testUsdGenSurfaceCageInterpolate PROPERTIES
    LABELS "T0;maps;sources" TIMEOUT 120)
# The function library against upstream SeExpr2. This is the only test that
# links the vendored frontend archive directly: expressions/exprMath.h claims to
# reproduce SeExpr2's builtins for the device, and the claim is only worth
# something if it is checked against the original. The include directory is
# SYSTEM so the vendored headers keep their own warning level, and the SeExpr2
# namespace rename must match the one usdGenExprFrontend was compiled with.
add_executable(testUsdGenSeExprOracle tests/testUsdGenSeExprOracle.cpp)
target_link_libraries(testUsdGenSeExprOracle PRIVATE usdGenWithPrimitives usdGenUsd usdGenExprFrontend)
target_include_directories(testUsdGenSeExprOracle SYSTEM PRIVATE "${_usdgen_frontend_dir}")
target_compile_definitions(testUsdGenSeExprOracle PRIVATE
    SeExpr2=UsdGenSeExprFrontend SeExprInternal2=UsdGenSeExprFrontendInternal
    USDGEN_SEEXPR_SANDBOX)
if(WIN32)
    target_compile_definitions(testUsdGenSeExprOracle PRIVATE SEEXPR_WIN32)
endif()
add_test(NAME testUsdGenSeExprOracle COMMAND testUsdGenSeExprOracle)
set_tests_properties(testUsdGenSeExprOracle PROPERTIES
    LABELS "T0;expressions;parity" TIMEOUT 120)
# Unix-only: the harness is a bash script that sources bin/_env.sh, which is
# POSIX shell and resolves a Linux interpreter path. Registered unconditionally
# this invoked WSL's bash on Windows and failed on a 120 s timeout rather than
# reporting anything about the schema.
usdgen_add_unix_only_test(testUsdGenExpressionSchema
    bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/checks/run_expression_schema.sh")
set_tests_properties(testUsdGenExpressionSchema PROPERTIES
    LABELS "T1;expressions;schema" TIMEOUT 120
    ENVIRONMENT_MODIFICATION "GENBUILD=set:${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
    SKIP_REGULAR_EXPRESSION "SKIP")

# Note: testUsdGenInvalidation already carries T1;gate:SI-2 (set at line ~446).
# The SI-2 overlay extension (bare primvars locator, section 7) is appended
# within that same test and shares its executable; no new CMake target needed.
set_tests_properties(testUsdGenScheduling PROPERTIES LABELS "T1;gate:SI-3")
set_tests_properties(testUsdGenSnapshotRace PROPERTIES LABELS "T1;gate:SI-4")
set_tests_properties(testUsdGenPopulation PROPERTIES LABELS "T1;gate:SI-6" TIMEOUT 120)
set_tests_properties(testUsdGenAdapter PROPERTIES LABELS "T1;gate:SI-7")
set_tests_properties(testUsdGenRestAdapter PROPERTIES LABELS "T1;gate:SI-7")
set_tests_properties(testUsdGenCurveRestAdapter PROPERTIES LABELS "T1;gate:SI-7")
if(USDGEN_ENABLE_CUDA)
    add_executable(testUsdGenCudaHierarchy tests/testUsdGenCudaHierarchy.cpp)
    target_link_libraries(testUsdGenCudaHierarchy PRIVATE usdGenTestUtilsHd usdGenGpu)
    target_compile_definitions(testUsdGenCudaHierarchy PRIVATE USDGEN_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME testUsdGenCudaHierarchy COMMAND testUsdGenCudaHierarchy)
    set_tests_properties(testUsdGenCudaHierarchy PROPERTIES
        ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T1;cuda;hierarchy" TIMEOUT 120 RESOURCE_LOCK usdgen_gpu)
endif()
# testUsdGenAutoApply (gate:SI-8) tested UsdGenMaskAPI auto-apply; the mask
# feature (mask.cpp/h, maskParams.h) and its test were removed together, so
# there is no longer a target here to attach a LABELS override to.
set_tests_properties(testUsdGenContracts PROPERTIES LABELS "T1")

# N-5 (09 §5.2, 10 §5.6): schema regeneration-drift guard — the checked-in
# libs/hosts/usdGenUsd/schema/resources must stay byte-identical to a fresh
# usdGenSchema run over libs/hosts/usdGenUsd/schema/schema.usda (bin/gen_schema.sh is
# the reference recipe; the N-2 plugInfo post-processing drift is what this
# catches in CI). The source has landed (plan/13 §1); the else() echo-stub
# branch below is retained only for missing sources (never a fake pass).
if(WIN32)
    # The drift guard runs bin/gen_schema.sh (a POSIX set -eu/mktemp/cmp recipe)
    # through popen, so it is Unix-only.
    usdgen_add_unix_only_test(testUsdGenSchemaUpToDate testUsdGenSchemaUpToDate)
    set(_usdgen_schema_uptodate_props SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/testUsdGenSchemaUpToDate.cpp")
    add_executable(testUsdGenSchemaUpToDate tests/testUsdGenSchemaUpToDate.cpp)
    target_link_libraries(testUsdGenSchemaUpToDate PRIVATE usdGenSchema usd tf arch)
    target_compile_definitions(testUsdGenSchemaUpToDate PRIVATE
        USDGEN_TEST_SCHEMA_USDA="${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenUsd/schema/schema.usda"
        USDGEN_TEST_SCHEMA_RESOURCES="${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenUsd/schema/resources"
        USDGEN_TEST_USD_GEN_SCHEMA_TOOL="${USDGEN_TEST_USD_DIR}/bin/usdGenSchema")
    add_test(NAME testUsdGenSchemaUpToDate COMMAND testUsdGenSchemaUpToDate)
    set(_usdgen_schema_uptodate_props SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
else()
    add_test(NAME testUsdGenSchemaUpToDate
        COMMAND ${CMAKE_COMMAND} -E echo
            "M1-STUB testUsdGenSchemaUpToDate (tests/testUsdGenSchemaUpToDate.cpp pending wave 3)")
    set(_usdgen_schema_uptodate_props SKIP_REGULAR_EXPRESSION "M1-STUB")
endif()
set_tests_properties(testUsdGenSchemaUpToDate PROPERTIES
    LABELS "T1" ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" ${_usdgen_schema_uptodate_props})
