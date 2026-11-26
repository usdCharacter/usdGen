# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Tests: strand hair checks and Storm tests. Included from the root CMakeLists.txt inside if(USDGEN_BUILD_TESTS).

# ---------------------------------------------------------------------------
# Strand hair checks (plan/16-ue-hair-parity.md WS1). Three python checks,
# all cross-platform, all sharing the interpreter the expression tests found.
#
#   testUsdGenHairEnergy        integrates the ported BSDF over the sphere on
#                               the CPU and asserts it creates no energy. No
#                               pxr, no GL -- numpy only.
#   testUsdGenHairStrandsDefs   Sdr parse + registry discovery of the two new
#                               defs, and that the three C5 defs survive. The
#                               counterpart of usdGenShaders/test/checkC5.py,
#                               kept separate so C5 keeps asserting exactly
#                               three files with exactly 21 inputs.
#   testUsdGenHairStrandsRender a real Storm draw: differential checks that
#                               the glslfx COMPILED (a failed compile falls
#                               back silently) and that lights, the baked hair
#                               count and coverage all reach the pixels.
#
# Each exits 77 when its prerequisites are missing, which ctest reports as a
# skip rather than a failure.
# ---------------------------------------------------------------------------
if(_usdgen_test_python)
    usdgen_test_pluginpath_ops(_usdgen_hair_plugin_env
        "${_usdgen_schema_resources}" "${_usdgen_imaging_resources}"
        "${_usdgen_shader_resources}"
        "${USDGEN_TEST_USD_DIR}/plugin/usd" "${USDGEN_TEST_USD_DIR}/lib/usd")
    usdgen_test_path_list_ops(_usdgen_hair_python_env PYTHONPATH path_list_prepend
        "${USDGEN_TEST_USD_DIR}/lib/python" "${USDGEN_TEST_USD_DIR}/Lib/site-packages"
        "${CMAKE_CURRENT_BINARY_DIR}/python"
        "${CMAKE_CURRENT_SOURCE_DIR}/bin/launch_env")

    add_test(NAME testUsdGenHairEnergy
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/test/check_hair_energy.py")
    set_tests_properties(testUsdGenHairEnergy PROPERTIES
        LABELS "T1;shaders;gate:L-2" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

    # fur_gen.py MoonRay fallback (docs/moonray-fur.md): the generated
    # PointInstancer fur must carry a universal UsdPreviewSurface terminal
    # with a displayColor primvar reader beside its Storm glslfx terminal.
    # Stdlib only, no pxr -- nothing to skip on.
    add_test(NAME testUsdGenFurGenMaterial
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/checks/check_fur_gen_material.py")
    set_tests_properties(testUsdGenFurGenMaterial PROPERTIES
        LABELS "T1;fur" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback")

    # launch_usdview.ps1 MoonRay wiring (docs/moonray-fur.md): the
    # launcher must place the MoonRay build's hd_moonray plugInfos ahead
    # of the OpenUSD prefix, its bin/ ahead of the prefix on PATH, and
    # set ARRAS_SESSION_PATH/RDL2_DSO_PATH for interactive rendering.
    # Skips (77) off Windows, without powershell, or without a MoonRay
    # build beside the repo.
    add_test(NAME testUsdGenUsdviewEnv
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/checks/check_usdview_env.py")
    set_tests_properties(testUsdGenUsdviewEnv PROPERTIES
        LABELS "T1;usdview" TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

    # The Marschner model's multiple-scattering table: rebuilds it from the
    # NumPy reference and asserts the glslfx carries it. Runs whichever model
    # the build selects -- the table ships in both.
    add_test(NAME testUsdGenHairMarschnerTable
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/test/tabulate_hair_marschner.py"
            --check)
    set_tests_properties(testUsdGenHairMarschnerTable PROPERTIES
        LABELS "T1;shaders" TIMEOUT 600
        FAIL_REGULAR_EXPRESSION "FAIL|Traceback"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

    add_test(NAME testUsdGenHairStrandsDefs
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/test/check_hair_strands_defs.py"
            --model "${_usdgen_hair_model}"
            --built "${_usdgen_shader_resources}/shaders")
    set_tests_properties(testUsdGenHairStrandsDefs PROPERTIES
        LABELS "T1;shaders;gate:L-2" TIMEOUT 180
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_hair_plugin_env};${_usdgen_hair_python_env}"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

    add_test(NAME testUsdGenScalpShadowDef
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/test/check_scalp_shadow.py"
            --no-render)
    set_tests_properties(testUsdGenScalpShadowDef PROPERTIES
        LABELS "T1;shaders;shadows" TIMEOUT 180
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_hair_plugin_env};${_usdgen_hair_python_env}"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

    # GPU-bound, so T2 and RUN_SERIAL like the other Storm tests.
    add_test(NAME testUsdGenScalpShadowRender
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/test/check_scalp_shadow.py"
            --usdrecord "${USD_INSTALL_DIR}/bin/usdrecord")
    set_tests_properties(testUsdGenScalpShadowRender PROPERTIES
        LABELS "T2;shaders;shadows" TIMEOUT 600 RUN_SERIAL TRUE
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_hair_plugin_env};${_usdgen_hair_python_env}"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

    # GPU-bound, so T2 and RUN_SERIAL like the other Storm tests.
    add_test(NAME testUsdGenHairRtxcrNumerics
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/test/check_hair_rtxcr.py"
            --usdrecord "${USD_INSTALL_DIR}/bin/usdrecord")
    set_tests_properties(testUsdGenHairRtxcrNumerics PROPERTIES
        LABELS "T2;shaders" TIMEOUT 180 RUN_SERIAL TRUE
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_hair_plugin_env};${_usdgen_hair_python_env}"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

    if(_usdgen_hair_model STREQUAL "rtxcr")
        add_test(NAME testUsdGenHairRtxcrViewport
            COMMAND "${_usdgen_test_python}"
                "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/test/check_hair_rtxcr_viewport.py"
                --testusdview "${USD_INSTALL_DIR}/bin/testusdview")
        set_tests_properties(testUsdGenHairRtxcrViewport PROPERTIES
            LABELS "T2;shaders;usdview" TIMEOUT 300 RUN_SERIAL TRUE
            FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
            ENVIRONMENT_MODIFICATION "${_usdgen_hair_plugin_env};${_usdgen_hair_python_env}"
            SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    endif()

    add_test(NAME testUsdGenHairStrandsRender
        COMMAND "${_usdgen_test_python}"
            "${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/test/check_hair_strands_render.py"
            --usdrecord "${USD_INSTALL_DIR}/bin/usdrecord"
            --model "${_usdgen_hair_model}")
    set_tests_properties(testUsdGenHairStrandsRender PROPERTIES
        LABELS "T2;shaders;gate:L-2" TIMEOUT 600 RUN_SERIAL TRUE
        FAIL_REGULAR_EXPRESSION "FAIL:|Traceback"
        ENVIRONMENT_MODIFICATION "${_usdgen_hair_plugin_env};${_usdgen_hair_python_env}"
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endif()

# M1 T2 Storm tests (EGL harness; the S-8/S-9/L-1 PW decisions get confirmed
# on shipped code here, 11 §2.2 exit-gate table).
if(NOT WIN32)
    add_executable(testUsdGenStormLook tests/testUsdGenStormLook.cpp)
    target_link_libraries(testUsdGenStormLook PRIVATE usdGenTestUtilsHd)
    target_include_directories(testUsdGenStormLook PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/test)
endif()
usdgen_add_unix_only_test(testUsdGenStormLook testUsdGenStormLook ${PROJECT_SOURCE_DIR})
set_tests_properties(testUsdGenStormLook PROPERTIES
    LABELS "T2;gate:L-2" ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

if(NOT WIN32)
    add_executable(benchUsdGenStorm tests/perf/benchUsdGenStorm.cpp)
    target_link_libraries(benchUsdGenStorm PRIVATE usdGenTestUtilsHd)
    target_include_directories(benchUsdGenStorm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/test)
endif()
usdgen_add_unix_only_test(benchUsdGenStorm benchUsdGenStorm
    --scene ${CMAKE_CURRENT_SOURCE_DIR}/usd/fixtures/g3_head100k.usda
    --static --refine 2)
# --scene is required (bare bench exits 2): the S-1 scene is the G3
# canonical groom at refineLevel 2 (the S-1 gate condition, 09 §5.3).
# The fixture is generated (usdGenFixtures below; gitignored build
# artifact like the binaries); a missing file fails loudly in OpenScene,
# never silently. HD_ENABLE_PERFLOG=1 feeds the drawBatches/drawCalls/
# itemsDrawn counters the S-5/S-6/S-12 assertions read.
set_tests_properties(benchUsdGenStorm PROPERTIES
    LABELS "T2;perf;gate:S-1;gate:S-5;gate:S-6;gate:S-12"
    ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}" ENVIRONMENT "HD_ENABLE_PERFLOG=1" SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

# Groom fixtures G1-G5 (S-1/S-5/S-6/S-12/E-5 scenes): pure-text output of
# usd/fixtures/make_fixtures.py (stdlib only, no pxr), so T2 stays
# self-contained wherever python3 exists. Regeneration is byte-identical
# (short decimals, fixed emission order); the files stay gitignored.
if(NOT WIN32)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/usd/fixtures/g3_head100k.usda
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/usd/fixtures/make_fixtures.py
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/usd/fixtures/make_fixtures.py
        COMMENT "Generating usdGen groom fixtures G1-G5 (gitignored)"
        VERBATIM)
    add_custom_target(usdGenFixtures ALL
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/usd/fixtures/g3_head100k.usda)
    add_dependencies(benchUsdGenStorm usdGenFixtures)
endif()

# N-4 (09 §5.3/§5.4, 10 §5.6 + §6.5): the remaining M1 Storm decision gates.
# S-8 (tangent strategy PW decision) needs both the variant A/B timing
# comparison (`testUsdGenStormTangent`) and the
# HDST_ENABLE_HGI_RESOURCE_GENERATION=1 compile (`testUsdGenStormHgiResource`);
# S-9 is the refineLevel live switch, L-1 the preview-material resolution.
# All sources except testUsdGenStormSurgery have landed (plan/13 §1, §3.3);
# the else() echo branch below is retained for missing sources so
# `ctest -N -L 'gate:'` still resolves every M1 gate. GPU-bound tests:
# RUN_SERIAL keeps one Storm context per ctest slot.
set(_usdgen_storm_labels_testUsdGenStormTangent  "T2;gate:S-8")
set(_usdgen_storm_labels_testUsdGenStormHgiResource "T2;gate:S-8")
set(_usdgen_storm_labels_testUsdGenStormRefine   "T2;gate:S-9")
set(_usdgen_storm_labels_testUsdGenStormMaterial "T2;gate:L-1")
set(_usdgen_storm_labels_testUsdGenStormSurgery  "T2")
foreach(_t testUsdGenStormTangent testUsdGenStormHgiResource
           testUsdGenStormRefine  testUsdGenStormMaterial
           testUsdGenStormSurgery)
    if(NOT WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/${_t}.cpp")
        add_executable(${_t} tests/${_t}.cpp)
        target_link_libraries(${_t} PRIVATE usdGenTestUtilsHd)
        target_include_directories(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/usdGenShaders/test)
        set(_usdgen_storm_cmd ${_t} ${PROJECT_SOURCE_DIR})
        set(_usdgen_storm_props SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
    elseif(WIN32)
        set(_usdgen_storm_cmd
            ${CMAKE_COMMAND} -E echo "SKIP ${_t} (Unix-only EGL/Storm harness)")
        set(_usdgen_storm_props SKIP_REGULAR_EXPRESSION "SKIP")
    else()
        set(_usdgen_storm_cmd
            ${CMAKE_COMMAND} -E echo "M1-STUB ${_t} (pending wave-3 source)")
        set(_usdgen_storm_props SKIP_REGULAR_EXPRESSION "M1-STUB")
    endif()
    add_test(NAME ${_t} COMMAND ${_usdgen_storm_cmd})
    set_tests_properties(${_t} PROPERTIES
        LABELS "${_usdgen_storm_labels_${_t}}"
        RUN_SERIAL TRUE
        ENVIRONMENT_MODIFICATION "${_usdgen_m1_env}"
        ${_usdgen_storm_props})
endforeach()
