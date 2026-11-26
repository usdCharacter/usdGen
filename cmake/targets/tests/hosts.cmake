# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Tests: imaging T1 tests over the UsdImaging chain. Included from the root CMakeLists.txt inside if(USDGEN_BUILD_TESTS).

# M1 T1 imaging tests (scene index over the real UsdImaging chain).
foreach(_t testUsdGenTileContract testUsdGenStockStormContract testUsdGenInvalidation testUsdGenScheduling
           testUsdGenSnapshotRace testUsdGenPopulation testUsdGenAdapter testUsdGenScenePublication
           testUsdGenNativeTimePublication
           testUsdGenRestAdapter testUsdGenCurveRestAdapter testUsdGenContracts
           testUsdGenExpressionPrimConnection testUsdGenViewportOverlay testUsdGenFurOcclusion
           testUsdGenSourceColor testUsdGenLookPrimvars
           testUsdGenInstance testUsdGenReferenceSource testUsdGenPaintLengthRecook
           testUsdGenDormantGroom)
    add_executable(${_t} tests/${_t}.cpp)
    target_link_libraries(${_t} PRIVATE usdGenTestUtilsHd)
    add_test(NAME ${_t} COMMAND ${_t})
    set_tests_properties(${_t} PROPERTIES
        ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endforeach()
set_tests_properties(testUsdGenTileContract PROPERTIES LABELS "T1;gate:SI-1")
set_tests_properties(testUsdGenStockStormContract PROPERTIES
    LABELS "T1;stock-hydra;unpatched-openusd")
set_tests_properties(testUsdGenFurOcclusion PROPERTIES LABELS "T1;fur;shadows")
set_tests_properties(testUsdGenSourceColor PROPERTIES LABELS "T1;look;sources")
set_tests_properties(testUsdGenReferenceSource PROPERTIES LABELS "T1;sources;reference")
set_tests_properties(testUsdGenLookPrimvars PROPERTIES LABELS "T1;look;primvars")
set_tests_properties(testUsdGenPaintLengthRecook PROPERTIES
    LABELS "T1;imaging;paint;brush" TIMEOUT 120)
# The 4098-generation coalescing burst fits in 60s on a quiet runner and
# misses it when the canonical collide playback test is cooking on the
# other cores. Serialize so the budget measures the burst, not the neighbor.
set_tests_properties(testUsdGenScenePublication PROPERTIES
    LABELS "T1;scheduling;imaging;scene-owner" TIMEOUT 60 RUN_SERIAL TRUE)
# Canonical Collide playback cooks the 8000-density example (128 collide
# iterations, cut-then-collide) several times, plus the test's own 5s
# playback window. The workflow's 180s --timeout is only the backstop for
# tests that forgot a TIMEOUT; this one needs its own, and it needs the
# machine. Parallel with the rest of T0/T1 it is killed at 180s every run.
set_tests_properties(testUsdGenNativeTimePublication PROPERTIES
    LABELS "T1;scheduling;imaging;operators" TIMEOUT 600 RUN_SERIAL TRUE)
target_compile_definitions(testUsdGenNativeTimePublication PRIVATE
    USDGEN_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
set_tests_properties(testUsdGenDormantGroom PROPERTIES
    LABELS "T1;scheduling;imaging;scene-owner" TIMEOUT 60)
set_tests_properties(testUsdGenInvalidation PROPERTIES LABELS "T1;gate:SI-2")

add_test(NAME testUsdGenUnpatchedOpenUsd
    COMMAND ${CMAKE_COMMAND}
        -DUSDGEN_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${PROJECT_SOURCE_DIR}/tests/testUsdGenUnpatchedOpenUsd.cmake)
set_tests_properties(testUsdGenUnpatchedOpenUsd PROPERTIES
    LABELS "T0;unpatched-openusd")
add_test(NAME testUsdGenUnpatchedOpenUsdNegative
    COMMAND ${CMAKE_COMMAND}
        -DUSDGEN_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${PROJECT_SOURCE_DIR}/tests/testUsdGenUnpatchedOpenUsdNegative.cmake)
set_tests_properties(testUsdGenUnpatchedOpenUsdNegative PROPERTIES
    LABELS "T0;unpatched-openusd")

# ---- P4 / SI-10: testUsdGenSessions registration ----
# Two indices on one session ⇒ identical generation, prim set and frame for one commit.
# Adds testUsdGenSessions as a T1 gate:SI-10 test (imaging lane).
# Placement: after the existing testUsdGenInvalidation block (~line 446).

add_executable(testUsdGenSessions
    tests/testUsdGenSessions.cpp
    )
target_link_libraries(testUsdGenSessions
    PUBLIC usdGenWithPrimitives usdGenUsd
    PUBLIC usdGenImaging
)
set_target_properties(testUsdGenSessions PROPERTIES
    VISIBILITY_INLINES_HIDDEN ON
)

add_test(NAME testUsdGenSessions
    COMMAND testUsdGenSessions
)
set_tests_properties(testUsdGenSessions
    PROPERTIES
    LABELS "T1;gate:SI-10"
    TIMEOUT 120
)

# ---- SI-12: testUsdGenHydraParity registration ----
# Hydra/stage builder parity on G1–G4 (imaging lane, V2-11 item 4): both
# builders must produce an identical UsdGenGraphDesc before the stage path
# is unplugged. Fixture dir arrives as USDGEN_TEST_FIXTURE_DIR (absolute
# source path, mirrors the bench --scene pattern); plugin discovery comes
# from _usdgen_m1_env like every adapter-dependent T1 test.
add_executable(testUsdGenHydraParity
    tests/testUsdGenHydraParity.cpp
)
target_link_libraries(testUsdGenHydraParity
    PUBLIC usdGenTestUtilsHd
    PUBLIC usdGenImaging
)
target_compile_definitions(testUsdGenHydraParity PRIVATE
    USDGEN_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/usd/fixtures")
set_target_properties(testUsdGenHydraParity PROPERTIES
    VISIBILITY_INLINES_HIDDEN ON
)
add_test(NAME testUsdGenHydraParity
    COMMAND testUsdGenHydraParity
)
set_tests_properties(testUsdGenHydraParity
    PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}"
    LABELS "T1;gate:SI-12"
    TIMEOUT 300
    SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
)

# GeomSubset targets wherever usdGen needs a mesh (02 §2.20, R15): both
# builders, Scatter restriction/union/diagnostics, subset paint and
# geoSampler inputs, cooked on the CPU lane from an authored stage.
add_executable(testUsdGenGeomSubset
    tests/testUsdGenGeomSubset.cpp
)
target_link_libraries(testUsdGenGeomSubset
    PUBLIC usdGenTestUtilsHd
    PUBLIC usdGenImaging
)
set_target_properties(testUsdGenGeomSubset PROPERTIES
    VISIBILITY_INLINES_HIDDEN ON
)
add_test(NAME testUsdGenGeomSubset
    COMMAND testUsdGenGeomSubset
)
set_tests_properties(testUsdGenGeomSubset
    PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}"
    LABELS "T1"
    TIMEOUT 300
)

add_executable(testUsdGenHierarchy tests/testUsdGenHierarchy.cpp)
target_link_libraries(testUsdGenHierarchy PRIVATE usdGenTestUtilsHd)
add_test(NAME testUsdGenHierarchy COMMAND testUsdGenHierarchy)
set_tests_properties(testUsdGenHierarchy PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T1;hierarchy" TIMEOUT 120)

add_executable(testUsdGenEnabledToggle tests/testUsdGenEnabledToggle.cpp)
target_link_libraries(testUsdGenEnabledToggle PRIVATE usdGenTestUtilsHd)
add_test(NAME testUsdGenEnabledToggle COMMAND testUsdGenEnabledToggle)
set_tests_properties(testUsdGenEnabledToggle PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T1;imaging;scheduling" TIMEOUT 120)

# The shipped root-to-tip width example, cooked on the CPU reference lane.
add_executable(testUsdGenExpressionWidthExample tests/testUsdGenExpressionWidthExample.cpp)
target_link_libraries(testUsdGenExpressionWidthExample PRIVATE usdGenTestUtilsHd)
target_compile_definitions(testUsdGenExpressionWidthExample PRIVATE
    USDGEN_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME testUsdGenExpressionWidthExample COMMAND testUsdGenExpressionWidthExample)
set_tests_properties(testUsdGenExpressionWidthExample PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T1;expressions;examples" TIMEOUT 120)

# The guide-interpolation and Ptex clump examples, including dirty
# propagation from sampled geometry and map prims through the scene index.
# Reads the baked examples/maps/*.ptx.
add_executable(testUsdGenGroomExamples tests/testUsdGenGroomExamples.cpp)
target_link_libraries(testUsdGenGroomExamples PRIVATE usdGenTestUtilsHd)
target_compile_definitions(testUsdGenGroomExamples PRIVATE
    USDGEN_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME testUsdGenGroomExamples COMMAND testUsdGenGroomExamples)
set_tests_properties(testUsdGenGroomExamples PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" LABELS "T1;expressions;examples;maps" TIMEOUT 300)
