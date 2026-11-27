# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Tests: build-rule and restructure gates. Included from the root CMakeLists.txt inside if(USDGEN_BUILD_TESTS).

# M0: build-rule tests (gate B-1) — see cmake/CheckNoStage*.cmake
set(USDGEN_FORBIDDEN_LIB_REGEX "^libusd_(usd|hd|hgi|glf|garch)")
set(USDGEN_FORBIDDEN_INC_REGEX
    "pxr/(usd|imaging|usdImaging)/(usd|hd|hgi|glf|garch)[A-Za-z0-9_]*\\/")

find_program(USDGEN_CXXFILT c++filt)

foreach(_t usdGen usdGenMath usdGenTestUtils)
    # Gate B-1 link rule inspects ELF magic and runs objdump/nm; Unix-only.
    usdgen_add_unix_only_test(testUsdGenLinkRule_${_t}
        ${CMAKE_COMMAND}
            -DTARGET_FILE=$<TARGET_FILE:${_t}>
            -DOBJDUMP=${CMAKE_OBJDUMP}
            -DCXXFILT=${USDGEN_CXXFILT}
            -DFORBIDDEN=${USDGEN_FORBIDDEN_LIB_REGEX}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNoStageLink.cmake)
endforeach()
# All three link-rule tests are the B-1 gate (core libs must not link the
# OpenUSD stage family). Label every foreach-generated test so the canonical
# `ctest -L '^T[01]$'` selector covers all of them (sol C-3, gatekeeper close).
foreach(_t usdGen usdGenMath usdGenTestUtils)
    set_tests_properties(testUsdGenLinkRule_${_t}
        PROPERTIES LABELS "T0;build;gate:B-1")
endforeach()

add_test(NAME testUsdGenIncludeRule
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DFORBIDDEN=${USDGEN_FORBIDDEN_INC_REGEX}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNoStageInclude.cmake)
set_tests_properties(testUsdGenIncludeRule
    PROPERTIES LABELS "T0;build;gate:B-1")

# ELF symbol-table inspection (nm); Unix-only.
usdgen_add_unix_only_test(testUsdGenNoThirdPartyExports
    ${CMAKE_COMMAND}
        -DTARGET_FILE=$<TARGET_FILE:usdGen>
        -DCXXFILT=${USDGEN_CXXFILT}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNoThirdPartyExports.cmake)
# sol X-1: this is a B-1 gate check and must carry the gate:B-1 label so
# `ctest -L '^gate:B-1$'` selects all five B-1 tests.
set_tests_properties(testUsdGenNoThirdPartyExports
    PROPERTIES LABELS "T0;build;gate:B-1")

# sol S-1/S-2: negative/positive fixtures proving the B-1 gates actually fail
# on forbidden input and pass on clean input (no silent false passes).
usdgen_add_unix_only_test(testUsdGenLinkRuleNegativeFixture
    ${CMAKE_COMMAND}
        -DTARGET_FILE=${USD_INSTALL_DIR}/lib/libusd_hd.so
        -DOBJDUMP=${CMAKE_OBJDUMP}
        -DCXXFILT=${USDGEN_CXXFILT}
        -DFORBIDDEN=^libusd_tf
        -DEXPECT=fail
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gateLinkDriver.cmake)
usdgen_add_unix_only_test(testUsdGenLinkRulePositiveFixture
    ${CMAKE_COMMAND}
        -DTARGET_FILE=$<TARGET_FILE:usdGen>
        -DOBJDUMP=${CMAKE_OBJDUMP}
        -DCXXFILT=${USDGEN_CXXFILT}
        -DFORBIDDEN=${USDGEN_FORBIDDEN_LIB_REGEX}
        -DEXPECT=pass
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gateLinkDriver.cmake)
add_test(NAME testUsdGenIncludeRuleNegativeFixture
    COMMAND ${CMAKE_COMMAND}
        -DFIXTURE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gateIncludeForbidden
        -DFORBIDDEN=${USDGEN_FORBIDDEN_INC_REGEX}
        -DEXPECT=fail
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gateIncludeDriver.cmake)
add_test(NAME testUsdGenIncludeRulePositiveFixture
    COMMAND ${CMAKE_COMMAND}
        -DFIXTURE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gateIncludeClean
        -DFORBIDDEN=${USDGEN_FORBIDDEN_INC_REGEX}
        -DEXPECT=pass
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gateIncludeDriver.cmake)
# Note: the fixture tests verify the gate *mechanism* but are not themselves
# gate checks, so they carry only T0;build. This keeps
# `ctest -L '^gate:B-1$'` selecting exactly the five B-1 gate checks
# (sol X-1; plan 10-build-dependencies-testing.md §1.4).
foreach(_f testUsdGenLinkRuleNegativeFixture testUsdGenLinkRulePositiveFixture
           testUsdGenIncludeRuleNegativeFixture testUsdGenIncludeRulePositiveFixture)
    set_tests_properties(${_f}
        PROPERTIES LABELS "T0;build")
endforeach()

# Restructure gates: no locks, no pxr below the hosts, no warm allocations.
include(cmake/gates/Gates.cmake)

# M2: staging-rule tests (gate B-2) — the Hydra-sourced staging fence
# (13-codebase-alignment.md §7 V2-8). Mirrors the B-1 gate pattern: an
# include-regex half over the fenced staging TUs plus an nm
# undefined-symbol half over their compiled objects, with negative/positive
# fixtures proving the mechanism bites both ways. Fenced TUs are the
# Hydra-builder and _CommitNow staging path (09 §5.1 B-2); the renamed
# stage builder (BuildGraphDescFromStage, V2-11) lives outside this list
# by design as the fenced-off T0/offline path.
set(USDGEN_FORBIDDEN_STAGING_INC_REGEX
    "pxr/usd/(usd/(stage|prim|attribute|relationship|primRange)\\.h|usdGeom/|usdShade/)")
set(USDGEN_FORBIDDEN_STAGING_SYM_REGEX
    "Usd(Stage|Prim|Attribute|Relationship|Geom|Shade)")
find_program(USDGEN_NM nm)
set(_usdgen_staging_tus
    "libs/hosts/usdGenHydra/usdGenImaging/usdGenGraphDescBuilder.h|libs/hosts/usdGenHydra/usdGenImaging/usdGenGraphDescBuilder.cpp|libs/hosts/usdGenHydra/usdGenImaging/groomSceneIndexPlugin.h|libs/hosts/usdGenHydra/usdGenImaging/groomSceneIndexPlugin.cpp")
set(_usdgen_staging_objs
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/usdGenImaging.dir/libs/hosts/usdGenHydra/usdGenImaging/usdGenGraphDescBuilder.cpp.o|${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/usdGenImaging.dir/libs/hosts/usdGenHydra/usdGenImaging/groomSceneIndexPlugin.cpp.o")
usdgen_add_unix_only_test(testUsdGenStagingRule
    ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DSTAGING_TUS=${_usdgen_staging_tus}
        -DOBJECTS=${_usdgen_staging_objs}
        -DFORBIDDEN_INC=${USDGEN_FORBIDDEN_STAGING_INC_REGEX}
        -DFORBIDDEN_SYM=${USDGEN_FORBIDDEN_STAGING_SYM_REGEX}
        -DNM=${USDGEN_NM}
        -DEXPECT=pass
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNoStageStaging.cmake)
set_tests_properties(testUsdGenStagingRule
    PROPERTIES LABELS "T0;build;gate:B-2")
add_test(NAME testUsdGenStagingRuleNegativeFixture
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gateStagingForbidden
        -DSTAGING_TUS=staging_bad.cpp
        -DOBJECTS=
        -DFORBIDDEN_INC=${USDGEN_FORBIDDEN_STAGING_INC_REGEX}
        -DFORBIDDEN_SYM=${USDGEN_FORBIDDEN_STAGING_SYM_REGEX}
        -DNM=${USDGEN_NM}
        -DEXPECT=fail
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNoStageStaging.cmake)
add_test(NAME testUsdGenStagingRulePositiveFixture
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gateStagingClean
        -DSTAGING_TUS=staging_clean.cpp
        -DOBJECTS=
        -DFORBIDDEN_INC=${USDGEN_FORBIDDEN_STAGING_INC_REGEX}
        -DFORBIDDEN_SYM=${USDGEN_FORBIDDEN_STAGING_SYM_REGEX}
        -DNM=${USDGEN_NM}
        -DEXPECT=pass
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNoStageStaging.cmake)
# Fixture tests verify the gate *mechanism*, not the tree: T0;build only,
# so `ctest -L '^gate:B-2$'` selects exactly testUsdGenStagingRule.
foreach(_f testUsdGenStagingRuleNegativeFixture testUsdGenStagingRulePositiveFixture)
    set_tests_properties(${_f}
        PROPERTIES LABELS "T0;build")
endforeach()
