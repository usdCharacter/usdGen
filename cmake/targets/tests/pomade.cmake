# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Tests: Pomade, usdGenTools and usdview-driven tests. Included from the root CMakeLists.txt inside if(USDGEN_BUILD_TESTS).

# usdGenTools C ABI (usdGenImaging/usdGenToolsApi.h): the compile/list entry
# points the usdview expression editor calls through ctypes. Pure C ABI over
# the engine frontend, so it is an ordinary T1 with no stage and no Hydra.
add_executable(testUsdGenToolsApi tests/hosts/usdGenHydra/testUsdGenToolsApi.cpp)
target_link_libraries(testUsdGenToolsApi PRIVATE usdGenImaging)
add_test(NAME testUsdGenToolsApi COMMAND testUsdGenToolsApi)
set_tests_properties(testUsdGenToolsApi PROPERTIES
    LABELS "T1;tools;expressions" TIMEOUT 60)

# usdGenPomade scene-index publication + C ABI (plan/17 section 6, plan/18
# section 2.6). Portable T1s (no stage, no renderer, no Unix-only APIs): the
# index test drives UsdGenPomadeSceneIndex over an empty retained scene index
# and asserts the registry join, the per-level publication and leaf-exact
# dirties; the API test drives pomadeApi.h, including the V0 registry and
# level-display entries with no index attached.
add_executable(testUsdGenPomadeIndex tests/hosts/usdGenPomade/testUsdGenPomadeIndex.cpp)
target_link_libraries(testUsdGenPomadeIndex PRIVATE usdGenPomade hd)
add_test(NAME testUsdGenPomadeIndex COMMAND testUsdGenPomadeIndex)
# G6 times a 17.6 MB retained-array copy against an 8 ms move budget.
# On the 4-vCPU ctest job that copy lands while other tests are in flight
# and reports ~17 ms; alone it is a memcpy. Serialize this case only
# (the --no-test-tube twin returns before G6).
set_tests_properties(testUsdGenPomadeIndex PROPERTIES
    ENVIRONMENT "USDGENPOMADE_TEST_TUBE=1"
    LABELS "T1;pomade" TIMEOUT 120 RUN_SERIAL TRUE)
# The opt-out the gate record_usd.ps1 relies on: with USDGENPOMADE_TEST_TUBE=0
# (and by default — the binary clears the ambient variable first, so this
# pins the default itself) the index constructs empty (no scaffolding in
# headless frames) and still publishes the model's levels on Pomade_Activate.
add_test(NAME testUsdGenPomadeIndexNoTestTube
    COMMAND testUsdGenPomadeIndex --no-test-tube)
set_tests_properties(testUsdGenPomadeIndexNoTestTube PROPERTIES
    ENVIRONMENT "USDGENPOMADE_TEST_TUBE=0"
    LABELS "T1;pomade" TIMEOUT 120)
# The lock-free Pomade handoffs (lane mailbox, registry tables, published
# model state) under many threads.
add_executable(testUsdGenPomadeLanes tests/hosts/usdGenPomade/testUsdGenPomadeLanes.cpp)
target_link_libraries(testUsdGenPomadeLanes PRIVATE usdGenPomade usdGenWithPrimitives usdGenUsd)
add_test(NAME testUsdGenPomadeLanes COMMAND testUsdGenPomadeLanes)
set_tests_properties(testUsdGenPomadeLanes PROPERTIES
    LABELS "T0;pomade" TIMEOUT 120)
add_executable(testUsdGenPomadeApi tests/hosts/usdGenPomade/testUsdGenPomadeApi.cpp)
target_link_libraries(testUsdGenPomadeApi PRIVATE usdGenPomade)
add_test(NAME testUsdGenPomadeApi COMMAND testUsdGenPomadeApi)
set_tests_properties(testUsdGenPomadeApi PROPERTIES
    LABELS "T1;pomade" TIMEOUT 60)

# usdGenPomade P1 (plan/17 section 6): the asynchronous commit pipeline -
# worker layer builds, idle swaps, coalescing, cancellation,
# no-swap-during-gesture, the partial-transfer fallback, hydrate round-trip,
# the GuideInterpolate fill-in, Save groom, and gate TN-4 (reference-scale
# swap <= 5 ms). Needs the schema plugin for the typed groom prims, so it
# carries the same environment as every adapter-dependent T1 test.
add_executable(testUsdGenPomadeCommit tests/hosts/usdGenPomade/testUsdGenPomadeCommit.cpp)
target_link_libraries(testUsdGenPomadeCommit PRIVATE usdGenPomade usdGenImaging usd usdGeom sdf)
add_test(NAME testUsdGenPomadeCommit COMMAND testUsdGenPomadeCommit)
# TN-4 bounds each idle TransferContent slot at 8 ms. The first attempt on
# the parallel 4-vCPU job misses that (about 11 ms) and the retry passes.
# Same serialization as the hierarchy fan-out budget below.
set_tests_properties(testUsdGenPomadeCommit PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}"
    LABELS "T1;pomade;gate:TN-4" TIMEOUT 300 RUN_SERIAL TRUE
    SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

# Sparse Output: cage guides and a PTex ownership map feed runtime
# CurveSource interpolation -> Width, evaluated by the real engine.
add_executable(testUsdGenPomadeOutput tests/internal/usdGen/testUsdGenPomadeOutput.cpp)
target_link_libraries(testUsdGenPomadeOutput PRIVATE usdGenPomade usdGenTestUtilsHd)
add_test(NAME testUsdGenPomadeOutput COMMAND testUsdGenPomadeOutput)
set_tests_properties(testUsdGenPomadeOutput PROPERTIES
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}"
    LABELS "T1;pomade;output" TIMEOUT 300
    SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

# P2 scalp graph + region bake (plan/17). T0: K1/K2/K3 CPU twins, G1 graph
# ops, the model graph layer and the P2 C ABI (TN-6 parity runs inside when
# the build has CUDA and the host has a device).
add_executable(testUsdGenPomadeGraph tests/hosts/usdGenPomade/testUsdGenPomadeGraph.cpp)
target_link_libraries(testUsdGenPomadeGraph PRIVATE usdGenPomade)
if(USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenPomadeGraph PRIVATE CUDA::cudart)
endif()
add_test(NAME testUsdGenPomadeGraph COMMAND testUsdGenPomadeGraph)
set_tests_properties(testUsdGenPomadeGraph PROPERTIES
    LABELS "T0;pomade" TIMEOUT 300)

# P3 tubes + fill (plan/17). T0: K4/K5/K8-K11 CPU twins, the model tube/fill
# layers, auto-tube, the P3 C ABI (TN-6 parity runs inside when the build
# has CUDA and the host has a device).
add_executable(testUsdGenPomadeTubes tests/hosts/usdGenPomade/testUsdGenPomadeTubes.cpp)
target_link_libraries(testUsdGenPomadeTubes PRIVATE usdGenPomade)
if(USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenPomadeTubes PRIVATE CUDA::cudart)
endif()
add_test(NAME testUsdGenPomadeTubes COMMAND testUsdGenPomadeTubes)
set_tests_properties(testUsdGenPomadeTubes PROPERTIES
    LABELS "T0;pomade" TIMEOUT 300)

# Exact sub-face graph regions: guide roots and final K9/K10 guide starts
# must follow closed polygons rather than the coarse face map's centroid.
add_executable(testUsdGenPomadeSubface tests/hosts/usdGenPomade/testUsdGenPomadeSubface.cpp)
target_link_libraries(testUsdGenPomadeSubface PRIVATE usdGenPomade)
add_test(NAME testUsdGenPomadeSubface COMMAND testUsdGenPomadeSubface)
set_tests_properties(testUsdGenPomadeSubface PROPERTIES
    LABELS "T0;pomade" TIMEOUT 300)

# Region-rooted tubes retain their support-plane frame at section zero while
# later center CV edits bend the shaft.  This verifies the final host mesh,
# snapshot/hydrate, and the unpinned legacy K4 path.
add_executable(testUsdGenPomadeRootFrame tests/hosts/usdGenPomade/testUsdGenPomadeRootFrame.cpp)
target_link_libraries(testUsdGenPomadeRootFrame PRIVATE usdGenPomade)
add_test(NAME testUsdGenPomadeRootFrame COMMAND testUsdGenPomadeRootFrame)
set_tests_properties(testUsdGenPomadeRootFrame PROPERTIES
    LABELS "T0;pomade" TIMEOUT 300)

# Kernel registry + per-move chain smoke (plan/17 section 8). T0: the section
# 4.2 move chain on the CPU twins, determinism, P4 param validation, the
# K6/K7/K14 twins, and the TN-6 parity of every hierarchy device lane over
# the section 7 reference fanout. A CUDA build with no device runs the CPU
# half and then exits 77: TN-6 is what this test is for, so an unprovable
# run reports SKIPPED rather than passing quietly. A CPU-only build makes no
# parity claim and exits 0.
add_executable(testUsdGenPomadeKernels tests/hosts/usdGenPomade/testUsdGenPomadeKernels.cpp)
target_link_libraries(testUsdGenPomadeKernels PRIVATE usdGenPomade)
if(USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenPomadeKernels PRIVATE CUDA::cudart)
endif()
add_test(NAME testUsdGenPomadeKernels COMMAND testUsdGenPomadeKernels)
set_tests_properties(testUsdGenPomadeKernels PROPERTIES
    LABELS "T0;pomade" TIMEOUT 300
    SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

# P4 hierarchy core (plan/17 section 8). T0: K14 subdivide/merge (Hausdorff
# union bar, bit-exact round-trip via the persistent-parent hint,
# determinism, edge mode) + K7 averaging (idempotence, propagation).
add_executable(testUsdGenPomadeHierarchy tests/hosts/usdGenPomade/testUsdGenPomadeHierarchy.cpp)
target_link_libraries(testUsdGenPomadeHierarchy PRIVATE usdGenPomade)
if(USDGEN_ENABLE_CUDA)
    target_link_libraries(testUsdGenPomadeHierarchy PRIVATE CUDA::cudart)
endif()
add_test(NAME testUsdGenPomadeHierarchy COMMAND testUsdGenPomadeHierarchy)
# TN-1 bounds one hierarchy move at 4 ms over the reference fan-out.
# Under parallel ctest the same move reports ~6 ms and fails; the retry,
# once the machine has drained, passes in a few hundred milliseconds.
set_tests_properties(testUsdGenPomadeHierarchy PROPERTIES
    LABELS "T0;pomade" TIMEOUT 300 RUN_SERIAL TRUE)

# T1: the versioned .ptx through the real ptex() expression (per-strand
# equality with the live primvar), the bake worker, the swap, and the
# graph/primvar/map commit round-trip.
add_executable(testUsdGenPomadeRegionBake tests/internal/usdGenCurves/testUsdGenPomadeRegionBake.cpp)
target_link_libraries(testUsdGenPomadeRegionBake PRIVATE
    usdGenPomade usdGenWithPrimitives usdGenUsd usdGen_ptex usd sdf usdGeom)
add_test(NAME testUsdGenPomadeRegionBake COMMAND testUsdGenPomadeRegionBake)
set_tests_properties(testUsdGenPomadeRegionBake PROPERTIES
    LABELS "T1;pomade" TIMEOUT 600)


# usdGenTools SeExpr editor: drives usdview through testusdview against the
# expression example. Needs the prefix's usdview (a Python script) and a
# python that imports its pxr, so it registers only where both exist, and it
# opens a window, so it is T2 (not part of the T0|T1 run).
# The interpreter must be the one that imports the prefix's pxr: PY in the
# environment (the usdRig convention), else the shebang testusdview itself
# carries, else whatever is on PATH. On Windows a plain find_program would
# pick the Store's python3.exe alias, which cannot load these modules.
set(_usdgen_test_python "$ENV{PY}")
if(NOT _usdgen_test_python AND EXISTS "${USD_INSTALL_DIR}/bin/testusdview")
    file(STRINGS "${USD_INSTALL_DIR}/bin/testusdview" _usdgen_testusdview_shebang LIMIT_COUNT 1)
    string(REGEX REPLACE "^#![ \t]*" "" _usdgen_shebang_python "${_usdgen_testusdview_shebang}")
    if(EXISTS "${_usdgen_shebang_python}")
        set(_usdgen_test_python "${_usdgen_shebang_python}")
    endif()
endif()
if(NOT _usdgen_test_python)
    find_program(_usdgen_test_python NAMES python3 python NO_CACHE)
endif()
if(_usdgen_test_python AND EXISTS "${USD_INSTALL_DIR}/bin/testusdview")
    add_test(NAME testUsdGenToolsExprEditor
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/testenv/testUsdGenToolsExprEditor.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/expression-width-plane.usda")
    usdgen_test_pluginpath_ops(_usdgen_tools_plugin_env
        "${_usdgen_schema_resources}" "${_usdgen_imaging_resources}"
        "${_usdgen_shader_resources}" "${_usdgen_tools_resources}"
        "${USDGEN_TEST_USD_DIR}/plugin/usd" "${USDGEN_TEST_USD_DIR}/lib/usd")
    # Both python package layouts OpenUSD installs on Windows/POSIX, plus
    # build/python where the usdGenTools package is staged. Last, the
    # nanousd scrub (bin/launch_env/sitecustomize.py): the last prepend
    # lands first on PYTHONPATH, so the scrub's sitecustomize wins and a
    # nanousd editable install cannot shadow the prefix's pxr. A no-op
    # where nanousd is absent.
    usdgen_test_path_list_ops(_usdgen_tools_python_env PYTHONPATH path_list_prepend
        "${USDGEN_TEST_USD_DIR}/lib/python" "${USDGEN_TEST_USD_DIR}/Lib/site-packages"
        "${CMAKE_CURRENT_BINARY_DIR}/python"
        "${CMAKE_CURRENT_SOURCE_DIR}/bin/launch_env")
    set_tests_properties(testUsdGenToolsExprEditor PROPERTIES
        LABELS "T2;tools;usdview" TIMEOUT 180
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_tools_plugin_env};${_usdgen_tools_python_env}")

    # Brush tool through REAL mouse events (plan/08 section 2.1 exit):
    # bind a UsdGenPaintMap to /Plane, press-drag-release over the
    # viewport, the session preview repaints while the drag is live, and
    # release bakes the paint primvar into the edit target. T3: opens a
    # window through testusdview like the Pomade mode scripts.
    add_test(NAME testUsdviewBrushStroke
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/testenv/testUsdviewBrushStroke.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/test-plane.usda")
    set_tests_properties(testUsdviewBrushStroke PROPERTIES
        LABELS "T3;tools;brush;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_tools_plugin_env};${_usdgen_tools_python_env}")

    # Description setup through REAL mouse events: the palette's Setup
    # defines a ready-to-cook description (Scatter -> Grow Ops), paint-to
    # binds its Maps PaintMap, the cursor ring tracks the hover, and a
    # stroke bakes into the Maps primvar. The extra FAIL tokens are the
    # engine errors an operator-less description used to emit per frame.
    add_test(NAME testUsdviewBrushSetup
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/testenv/testUsdviewBrushSetup.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/test-plane.usda")
    set_tests_properties(testUsdviewBrushSetup PROPERTIES
        LABELS "T3;tools;brush;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback|operatorOrder|no operator nodes|CaptureGraphDescFromHydra|commit rejected"
        ENVIRONMENT_MODIFICATION "${_usdgen_tools_plugin_env};${_usdgen_tools_python_env}")
endif()

# Graph mode under testusdview. T3: opens a window, needs the staged
# usdGenPomadeTools package (same build/python) and the usdGenPomade DLL
# (USDGENPOMADE_DLL, defaulting to the build tree).
#
# testUsdviewPomadeGraph drives REAL mouse and key events over the StageView
# through the V2 viewport controller (plan/18 V2 exit). The ABI-level
# scripts beside it -- testPomadeAbi* -- call the C entry points directly and
# never open the tool; they keep their coverage under an honest name
# (plan/18 section 4 "T3"), and the modes whose loops land in V4/V5 have
# only the ABI script until then.
if(_usdgen_test_python AND EXISTS "${USD_INSTALL_DIR}/bin/testusdview")
    add_test(NAME testUsdviewPomadeGraph
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeGraph.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    usdgen_test_pluginpath_ops(_usdgen_pomade_plugin_env
        "${_usdgen_schema_resources}" "${_usdgen_imaging_resources}"
        "${_usdgen_shader_resources}"
        "${_usdgen_pomade_resources}" "${_usdgen_pomade_tools_resources}"
        "${USDGEN_TEST_USD_DIR}/plugin/usd" "${USDGEN_TEST_USD_DIR}/lib/usd")
    # bin/launch_env last so its sitecustomize lands first on PYTHONPATH and
    # scrubs a nanousd editable install's pxr shadow (no-op otherwise).
    usdgen_test_path_list_ops(_usdgen_pomade_python_env PYTHONPATH path_list_prepend
        "${USDGEN_TEST_USD_DIR}/lib/python" "${USDGEN_TEST_USD_DIR}/Lib/site-packages"
        "${CMAKE_CURRENT_BINARY_DIR}/python"
        "${CMAKE_CURRENT_SOURCE_DIR}/bin/launch_env")
    set_tests_properties(testUsdviewPomadeGraph PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # The graph model through the C ABI: two adjoining regions, welded, with
    # the shared boundary checked (plan/17 P2 exit). No viewport.
    add_test(NAME testPomadeAbiGraph
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testPomadeAbiGraph.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testPomadeAbiGraph PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # P3 Tube+Fill modes through the C ABI (same stage + environment).
    add_test(NAME testPomadeAbiTube
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testPomadeAbiTube.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testPomadeAbiTube PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # P4 Hierarchy walk L1 -> L3 and back (Part B self-skips until the P4 C
    # entries land, then runs strict).
    add_test(NAME testPomadeAbiLevels
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testPomadeAbiLevels.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testPomadeAbiLevels PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # TN-1/TN-2 probes + TN-3/TN-4/TN-7 skip-harness (plan/17 section 7 stub).
    add_test(NAME testUsdviewPomadePerf
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadePerf.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadePerf PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # V0 publication (plan/18 section 2.1/2.2 exit): one model created
    # through the C ABI, edited, activated and published, then asserted
    # against what Hydra ended up with -- the tubes/L1 and tubes/L2 prims
    # exist with the face counts the model's census predicts, and a pick
    # over the viewport hits them.
    add_test(NAME testUsdviewPomadePublish
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadePublish.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadePublish PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT "USDGENPOMADE_TEST_TUBE=1"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # V5 Hierarchy mode through REAL mouse and key events (plan/18 V5
    # exit): select, Shift+D, Ctrl+Down/Up, Shift+M, Ctrl+Z, asserted
    # against what the scene index published per level.
    add_test(NAME testUsdviewPomadeLevels
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeLevels.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeLevels PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # V5 Sculpt mode: the brush ring under a real cursor, a real grab
    # stroke that keeps its arc length, and Escape restoring bit-exactly.
    add_test(NAME testUsdviewPomadeSculpt
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeSculpt.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeSculpt PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # V4 Tube + Fill through the real controller (plan/18 V4 exit): a real
    # stroke leaves a tube stub behind it (G14), a real click selects a
    # center CV and a real drag moves the published centers, Ctrl+Z puts
    # them back, the Fill panel's density reaches the guides, and TN-1 is
    # re-measured over 50 real mouse moves.
    add_test(NAME testUsdviewPomadeTube
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeTube.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeTube PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # Real Qt input acceptance for hover, click and Shift-marquee selection.
    # This stays separate from the Tube workflow so selection routing and
    # StageView rollover ownership have a focused failure report.
    add_test(NAME testUsdviewPomadeSelection
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeSelection.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeSelection PROPERTIES
        LABELS "T3;pomade;tools;usdview;selection" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # An artist-level hierarchy edit: subdivide in the dock, select one
    # displayed child component, drag its manipulator, and prove the other
    # child descriptors and world points remain untouched through commit/undo.
    add_test(NAME testUsdviewPomadeIsolatedEditing
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeIsolatedEditing.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeIsolatedEditing PROPERTIES
        LABELS "T3;pomade;tools;usdview;selection;hierarchy" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # TN-3 pick accuracy: K11 vs StageView.pick at 100 pixels (P6 exit).
    add_test(NAME testUsdviewPomadePick
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadePick.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadePick PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # The full interactive workflow on the plane fixture, ending in 4
    # connected tubes: a graph stroke grows the stub, Shift+D grows three
    # children, double-click drills to them, a child tip CV takes the
    # gizmo for a drag (undo/redo), a sculpt stroke moves the second
    # child, the Fill panel feeds the guides, and the commit lands nested
    # on the stage.
    add_test(NAME testUsdviewPomadeWorkflow
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeWorkflow.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeWorkflow PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # Commit-enabled Output through the real viewport controller.  The
    # script owns the panel workflow; this mirrors Workflow's fixture and
    # staged plugin/Python/DLL environment.
    add_test(NAME testUsdviewPomadeOutput
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeOutput.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeOutput PROPERTIES
        LABELS "T3;pomade;tools;usdview;output" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # Click-by-click Graph regions on a single quad: Bind Geometry through
    # the dock, shared Ptex density UI, CV drafts, both closure paths, and
    # the one-step region+auto-tube undo/redo transaction.
    add_test(NAME testUsdviewPomadeCvRegions
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeCvRegions.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-single-quad.usda")
    set_tests_properties(testUsdviewPomadeCvRegions PROPERTIES
        LABELS "T3;pomade;tools;usdview;cv-regions" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    add_test(NAME testUsdviewPomadeTubeProfile
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeTubeProfile.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-sphere-scalp.usda")
    set_tests_properties(testUsdviewPomadeTubeProfile PROPERTIES
        LABELS "T3;pomade;tools;usdview;tube-profile" TIMEOUT 180
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # SS-01: File > Reopen keeps the live model (graph, undo stack and the
    # committed groom); a stage without the scalp keeps the session
    # detached with a missing-scalp warning instead of re-binding.
    add_test(NAME testUsdviewPomadeReattach
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeReattach.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-single-quad.usda")
    set_tests_properties(testUsdviewPomadeReattach PROPERTIES
        LABELS "T3;pomade;tools;usdview;session" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # SS-03: a dock save with "add to the scene" sublayers the groom in the
    # root layer, survives File > Reopen and comes back through Resume
    # groom; binding over a saved groom asks Resume / Start new / Cancel.
    # The script works on a temp copy of the scene (it saves the root layer).
    add_test(NAME testUsdviewPomadeFileCommands
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeFileCommands.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-single-quad.usda")
    set_tests_properties(testUsdviewPomadeFileCommands PROPERTIES
        LABELS "T3;pomade;tools;usdview;session" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # Graph Reposition through actual StageView drags: shared CV and edge
    # motion stays live in both incident regions, preserves stable topology
    # and root descriptors, and owns one undo / Escape transaction.
    add_test(NAME testUsdviewPomadeReposition
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeReposition.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-single-quad.usda")
    set_tests_properties(testUsdviewPomadeReposition PROPERTIES
        LABELS "T3;pomade;tools;usdview;graph;reposition" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # TN-7 bake isolation (plan/17 section 7): the TN-1 drag re-measured
    # with a full bake of the REFERENCE SCALP in flight, on the reference
    # scene rather than the 4x4 test scalp -- the point of the gate is
    # that a real bake's worker thread and lowest-priority CUDA stream
    # take nothing from the gesture. The script proves the bake was still
    # running when the drag ended, so a bake that finished early fails
    # instead of passing vacuously.
    add_test(NAME testUsdviewPomadeBakeIsolation
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeBakeIsolation.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-reference.usda")
    set_tests_properties(testUsdviewPomadeBakeIsolation PROPERTIES
        LABELS "T3;pomade;tools;usdview;gate:TN-7" TIMEOUT 600
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # DK-05 artist status strip: the sync pill follows a real drag's commit
    # (Committing... then Synced), a selection publish stays calm, Relax and
    # an undo past the bottom reach the dock's message area, and every
    # Display control writes its state field.
    add_test(NAME testUsdviewPomadeDock
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeDock.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeDock PROPERTIES
        LABELS "T3;pomade;tools;usdview;dock" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # TS-04 artist walkthrough: one session start to finish with real input
    # (menu open, gated dock, picker bind, two regions, Tube gizmo/tweak/
    # Escape/undo, Fill density + ramp, Hierarchy subdivide/merge/re-split,
    # Sculpt, Output build, dock Save, warnings row, dock close/reopen, the
    # Synced pill); every CONFUSING line of the 2026-09-23 walkthrough is a
    # hard check.
    add_test(NAME testUsdviewPomadeArtistWalkthrough
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeArtistWalkthrough.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-single-quad.usda")
    set_tests_properties(testUsdviewPomadeArtistWalkthrough PROPERTIES
        LABELS "T3;pomade;tools;usdview;walkthrough" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # TS-02 hotkey sweep: every documented key in every mode through real
    # QtTest key events (modes, sub-modes, F8-F11, Q/W/E/R, [ ], L, + = -,
    # P, J/X holds mid-drag, Escape cancel-only, Delete and its gate,
    # Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z / Shift+Z, Ctrl+A / Ctrl+Shift+A /
    # Ctrl+I, Hierarchy level keys, Ctrl+Shift+S).
    add_test(NAME testUsdviewPomadeHotkeys
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeHotkeys.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeHotkeys PROPERTIES
        LABELS "T3;pomade;tools;usdview;hotkeys" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # TS-02 navigation and focus: Alt camera drags over an active scalp,
    # host navigation presets (the Display group's Navigation combo), wheel
    # mid-drag, one render per camera move, no usdview rollover/pick/menu,
    # focus back from a dock field, dock close/reopen, F framing.
    add_test(NAME testUsdviewPomadeNavigation
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeNavigation.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeNavigation PROPERTIES
        LABELS "T3;pomade;tools;usdview;navigation" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # TS-03 mode coverage through real clicks and keys: the Graph click
    # tools (Connect/Link/Unweld/Delete, Shift+W/U), Fill (tube pick,
    # density row, length-ramp drag), Hierarchy ops (double-click,
    # Group/Make persistent, Merge selected, Solo, locks) and the Tube
    # actions (Relax, Match surface, Snap root). Shared fixture:
    # testenv/pomadeT3Groom.py.
    add_test(NAME testUsdviewPomadeGraphTools
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeGraphTools.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeGraphTools PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    add_test(NAME testUsdviewPomadeFill
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeFill.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeFill PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    add_test(NAME testUsdviewPomadeHierarchyOps
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeHierarchyOps.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeHierarchyOps PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    add_test(NAME testUsdviewPomadeTubeActions
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdviewPomadeTubeActions.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testUsdviewPomadeTubeActions PROPERTIES
        LABELS "T3;pomade;tools;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
    # TN-5 through the REAL controller (plan/18 V7): 30 minutes of QtTest
    # mouse and key gestures across every mode. T4 like its ABI sibling --
    # a wall-clock test, run with `ctest -L T4`, never by the default
    # suite (`ctest -R Pomade -LE T4`).
    add_test(NAME testPomadeSoakController
        COMMAND "${_usdgen_test_python}" "${USD_INSTALL_DIR}/bin/testusdview"
            --testScript "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testPomadeSoakController.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/pomade-graph-scalp.usda")
    set_tests_properties(testPomadeSoakController PROPERTIES
        LABELS "T4;pomade;tools;usdview;gate:TN-5" TIMEOUT 2400
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT "USDGENPOMADE_SOAK_MINUTES=30;USDGENPOMADE_SOAK_SEED=7"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
endif()

# Pomade render goldens (plan/18 section 4 "Image", G15). Headless like
# testUsdGenStormLook -- bin/record_pomade.py makes its own offscreen GL
# context -- but unlike usdrecord it hydrates the committed groom into a
# live PomadeModel first, so the tubes, level colours, center curves and
# CV dots the scene index publishes are in the frame. Same plugin path
# and PYTHONPATH as the T3 scripts, plus the DLL directory.
if(_usdgen_test_python AND EXISTS "${USD_INSTALL_DIR}/bin/testusdview")
    add_test(NAME testPomadeGolden
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testPomadeGolden.py")
    set_tests_properties(testPomadeGolden PROPERTIES
        LABELS "T2;pomade;tools;golden" TIMEOUT 900
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomade_plugin_env};${_usdgen_pomade_python_env};PATH=path_list_prepend:$<TARGET_FILE_DIR:usdGenPomade>")
endif()

# The Qt-free half of the same plugin -- the ctypes binding, the text helpers
# behind the literal controls, and the connection authoring -- runs in a plain
# python process, so it is a T1 and needs no usdview. It shares the editor
# test's interpreter and environment, minus the window.
if(_usdgen_test_python)
    add_test(NAME testUsdGenToolsExprApi
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/testenv/testUsdGenToolsExprApi.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/expression-width-plane.usda")
    usdgen_test_pluginpath_ops(_usdgen_toolsapi_plugin_env
        "${_usdgen_schema_resources}" "${_usdgen_imaging_resources}"
        "${_usdgen_shader_resources}" "${_usdgen_tools_resources}"
        "${USDGEN_TEST_USD_DIR}/plugin/usd" "${USDGEN_TEST_USD_DIR}/lib/usd")
    # bin/launch_env last: the last prepend lands first on PYTHONPATH, so
    # its sitecustomize scrubs a nanousd editable install's pxr shadow
    # before the test imports (a no-op where nanousd is absent).
    usdgen_test_path_list_ops(_usdgen_toolsapi_python_env PYTHONPATH path_list_prepend
        "${USDGEN_TEST_USD_DIR}/lib/python" "${USDGEN_TEST_USD_DIR}/Lib/site-packages"
        "${CMAKE_CURRENT_BINARY_DIR}/python"
        "${CMAKE_CURRENT_SOURCE_DIR}/bin/launch_env")
    set_tests_properties(testUsdGenToolsExprApi PROPERTIES
        LABELS "T1;tools;expressions" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_toolsapi_plugin_env};${_usdgen_toolsapi_python_env}")

    # The Qt-free half of the brush tool -- the map/stroke twin, camera
    foreach(_playback_test PlaybackState PlaybackFlow FrameStatusApi)
        add_test(NAME testUsdGen${_playback_test}
            COMMAND "${_usdgen_test_python}"
                "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/testenv/testUsdGen${_playback_test}.py")
        set_tests_properties(testUsdGen${_playback_test} PROPERTIES
            LABELS "T1;tools;playback" TIMEOUT 60
            FAIL_REGULAR_EXPRESSION "FAIL:|Traceback")
    endforeach()

    # The Qt-free half of the brush tool -- the map/stroke twin, camera
    # maths, CPU surface pick, bind/bake authoring, session preview, the
    # gesture loop and the panel descriptors -- in a plain python process.
    add_test(NAME testUsdGenToolsBrush
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/testenv/testUsdGenToolsBrush.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/test-plane.usda")
    set_tests_properties(testUsdGenToolsBrush PROPERTIES
        LABELS "T1;tools;brush" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_toolsapi_plugin_env};${_usdgen_toolsapi_python_env}")

    # The Qt-free UI half of the brush (state, loop cadence, palette and
    # panel descriptors, hotkey logic) in the same plain interpreter.
    add_test(NAME testUsdGenToolsBrushUI
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/testenv/testUsdGenToolsBrushUI.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/test-plane.usda")
    set_tests_properties(testUsdGenToolsBrushUI PROPERTIES
        LABELS "T1;tools;brush" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_toolsapi_plugin_env};${_usdgen_toolsapi_python_env}")
endif()

# The Qt-free half of usdGenPomadeTools -- modes, tool state, math kernels --
# runs in a plain python process (numpy only), so it is a T1 and needs no
# usdview. It shares the expression tests' interpreter.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsMath
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsMath.py")
    set_tests_properties(testUsdGenPomadeToolsMath PROPERTIES
        LABELS "T1;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# The Qt-free Graph-mode half -- stroke kernels, snap conversion, HUD text,
# sub-modes -- runs in the same plain interpreter.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsGraph
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsGraph.py")
    set_tests_properties(testUsdGenPomadeToolsGraph PROPERTIES
        LABELS "T1;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# The Qt-free half of the V2 viewport controller -- camera projection, the
# plan/18 section 3.4 key table, the gizmo state machine, and GraphLoop
# driven over a fake session that records every C call. No pxr, no DLL, no
# numpy, so it runs in the barest interpreter of the lot.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsLoops
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsLoops.py")
    set_tests_properties(testUsdGenPomadeToolsLoops PROPERTIES
        LABELS "T1;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# The Qt-free half of the V4 loops (plan/18 V4): TubeLoop and FillLoop over
# a fake session -- the pick-once contract, one undo bracket per drag, the
# world-delta-to-section-chart conversion, the soft-selection anchor rule,
# and Fill's preview/full density contract. Same bare interpreter.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsLoopsTube
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsLoopsTube.py")
    set_tests_properties(testUsdGenPomadeToolsLoopsTube PROPERTIES
        LABELS "T1;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# The vendored RigExec gizmo maths (pomade_gizmo_parity G08/G09/G11/G16/G20/
# G24): AxisDragParameter's ratio-scaled floor, scaleFactor, the step and
# world-grid snaps, the per-tool settings model and the World/Screen/Tube
# frames, ported from usdRig's test_gizmo_screen/snap/settings. Pure maths.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsGizmoScreen
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsGizmoScreen.py")
    set_tests_properties(testUsdGenPomadeToolsGizmoScreen PROPERTIES
        LABELS "T0;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# The Qt-free half of the V5 loops (plan/18 V5): HierarchyLoop's actions and
# their undo brackets, the drawn-edge subdivide, SculptLoop's one shaped
# stroke per move, and the fallback ladder stepped on synthetic timings.
# Same bare interpreter as its V2 sibling, whose recording fakes it reuses.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsLoopsHier
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsLoopsHier.py")
    set_tests_properties(testUsdGenPomadeToolsLoopsHier PROPERTIES
        LABELS "T1;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# The TN-5 soak (plan/17 section 7): 30 minutes of scripted editing over a
# reference-fanout model. Registered so it can be RUN -- `ctest -L T4` --
# but never by the default suite: it is a wall-clock test, which is what
# the T4 label means here (the audit's G15 row: "TN-5 soak not registered
# in ctest"). --smoke cuts it to a minute for a spot check.
if(_usdgen_test_python)
    add_test(NAME testPomadeSoak
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testPomadeSoak.py"
            --minutes 30)
    # The soak imports the prefix's pxr through pomadeLib; bin/launch_env
    # first so its sitecustomize scrubs a nanousd editable install's shadow.
    usdgen_test_path_list_ops(_usdgen_pomadesoak_python_env PYTHONPATH path_list_prepend
        "${USDGEN_TEST_USD_DIR}/lib/python" "${USDGEN_TEST_USD_DIR}/Lib/site-packages"
        "${CMAKE_CURRENT_SOURCE_DIR}/bin/launch_env")
    set_tests_properties(testPomadeSoak PROPERTIES
        LABELS "T4;pomade;tools;gate:TN-5" TIMEOUT 2400
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT "USDGENPOMADE_DLL=$<TARGET_FILE:usdGenPomade>"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomadesoak_python_env}"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    # The smoke is 60 seconds and DOES run in the default suite, so it is
    # not T4: T4 here means "wall clock, excluded by -LE T4", and the two
    # 30-minute soaks (testPomadeSoak, testPomadeSoakController) are the
    # only tests that earn it.
    add_test(NAME testPomadeSoakSmoke
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testPomadeSoak.py"
            --smoke)
    set_tests_properties(testPomadeSoakSmoke PROPERTIES
        LABELS "T1;pomade;tools;gate:TN-5" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT "USDGENPOMADE_DLL=$<TARGET_FILE:usdGenPomade>"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomadesoak_python_env}"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# The Qt-free Tube/Fill-mode half -- gizmo previews, fill counts, ramps,
# sub-modes -- runs in the same plain interpreter.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsTube
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsTube.py")
    set_tests_properties(testUsdGenPomadeToolsTube PROPERTIES
        LABELS "T1;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# The Qt-free Hierarchy/Sculpt-mode half -- level focus, breadcrumb, locks,
# visibility, subdivide/merge params, brush falloff -- same plain interpreter.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsHierarchy
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsHierarchy.py")
    set_tests_properties(testUsdGenPomadeToolsHierarchy PROPERTIES
        LABELS "T1;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# The Qt-free workspace-dock half (plan/18 V3): pomadePanels' per-mode
# parameter/action descriptors and pomadeHud's status-strip/warnings
# formatting, exercised against a fake session (no ctypes DLL, no Qt) --
# same plain interpreter as its siblings above.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsPanels
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsPanels.py")
    set_tests_properties(testUsdGenPomadeToolsPanels PROPERTIES
        LABELS "T1;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# Icon loader + name-table coverage (IC-01): every name pomadeModes.ICONS /
# pomadePanels.ACTION_ICONS references resolves, when a PNG exists for it,
# to a square RGBA glyph pomadeIcons can tint. Plain python + PIL, no Qt.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsIcons
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsIcons.py")
    set_tests_properties(testUsdGenPomadeToolsIcons PROPERTIES
        LABELS "T1;pomade;tools;icons" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# DOC-01: docs/pomade-tool.md names every hotkey the tool binds -- the
# pomadeModes shelf tables, HotkeyAction, the dock's HINTS lines and the
# viewport's J/X holds and +/- keys, all read from the code. Plain python,
# no pxr or Qt.
if(_usdgen_test_python)
    add_test(NAME testPomadeDocsHotkeys
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/checks/check_pomade_docs.py")
    set_tests_properties(testPomadeDocsHotkeys PROPERTIES
        LABELS "T0;pomade;tools;docs" TIMEOUT 60
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback")
endif()

# TS-01: the Qt-free half of pomadeT3.py, the shared T3 driver module --
# typeKey's key table against pomadeViewport.keyName's own source, and the
# check/info/statusRecorder helpers -- same plain interpreter as its
# siblings above (no pxr, no Qt).
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsT3Helpers
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsT3Helpers.py")
    set_tests_properties(testUsdGenPomadeToolsT3Helpers PROPERTIES
        LABELS "T1;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# The P5 bridge half -- center-curve export, swept/curve L3 import, the
# geodesic checks -- runs headless but needs pxr (an in-memory stage) and
# the usdGenPomade DLL, so unlike its Qt-free siblings it takes the USD
# python environment plus USDGENPOMADE_DLL. Part B is strict here (the
# variable's presence is what makes it strict) and a skip for direct
# runs without them.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsBridge
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsBridge.py")
    usdgen_test_path_list_ops(_usdgen_pomadebridge_python_env PYTHONPATH path_list_prepend
        "${USDGEN_TEST_USD_DIR}/lib/python" "${USDGEN_TEST_USD_DIR}/Lib/site-packages"
        "${CMAKE_CURRENT_SOURCE_DIR}/bin/launch_env")
    set_tests_properties(testUsdGenPomadeToolsBridge PROPERTIES
        LABELS "T1;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT "USDGENPOMADE_DLL=$<TARGET_FILE:usdGenPomade>"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomadebridge_python_env}")
endif()

# The V0b stage-contract binding (plan/18 V0b): pomadeLibStage.py over the
# real DLL -- the per-tube operations, the per-tube fill params, and a
# commit/hydrate round trip through the binding. Same environment as the
# bridge test above (USD python + USDGENPOMADE_DLL), same skip for direct
# runs without them.
if(_usdgen_test_python)
    add_test(NAME testUsdGenPomadeToolsStage
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsStage.py")
    usdgen_test_path_list_ops(_usdgen_pomadestage_python_env PYTHONPATH path_list_prepend
        "${USDGEN_TEST_USD_DIR}/lib/python" "${USDGEN_TEST_USD_DIR}/Lib/site-packages"
        "${CMAKE_CURRENT_SOURCE_DIR}/bin/launch_env")
    set_tests_properties(testUsdGenPomadeToolsStage PROPERTIES
        LABELS "T1;pomade;tools" TIMEOUT 120
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT "USDGENPOMADE_DLL=$<TARGET_FILE:usdGenPomade>"
        ENVIRONMENT_MODIFICATION "${_usdgen_pomadestage_python_env}"
        SKIP_RETURN_CODE "77")
endif()
