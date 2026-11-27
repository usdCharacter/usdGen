# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Tests: install-tree tests. Included from the root CMakeLists.txt inside if(USDGEN_BUILD_TESTS).

# ---------------------------------------------------------------------------
# M0 install-tree tests (plan §6.2, sol S-5/X-4)
#
# Both tests read the install prefix from the USDGEN_INSTALL_PREFIX env var.
# When unset they print a note and return 0 (clean skip). CI sets
# USDGEN_INSTALL_PREFIX to the scratch prefix after running cmake --install.
# ---------------------------------------------------------------------------
if(NOT WIN32)
    add_executable(testUsdGenInstallTree tests/hosts/usdGenUsd/testUsdGenInstallTree.cpp)
    target_link_libraries(testUsdGenInstallTree PRIVATE
        usd usdGeom tf arch gf vt sdf trace)
    target_compile_definitions(testUsdGenInstallTree PRIVATE
        USDGEN_TEST_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/scenes/scene_empty_groom.usda"
        USDGEN_TEST_PLUGIN_DIR="${USDGEN_INSTALL_PLUGINDIR}"
        USDGEN_TEST_LIB_DIR="${USDGEN_INSTALL_LIBDIR}"
        USDGEN_TEST_CMAKE_DIR="${USDGEN_INSTALL_CMAKEDIR}"
        USDGEN_TEST_USD_INSTALL_DIR="${USD_INSTALL_DIR}")

    add_executable(testUsdGenConsumer tests/api/testUsdGenConsumer.cpp)
    target_compile_definitions(testUsdGenConsumer PRIVATE
        USDGEN_TEST_CONSUMER_SRC="${CMAKE_CURRENT_SOURCE_DIR}/tests/consumer"
        USDGEN_TEST_CMAKE_COMMAND="${CMAKE_COMMAND}"
        USDGEN_TEST_LIB_DIR="${USDGEN_INSTALL_LIBDIR}")
endif()
# Both assert the .so/.a install layout and drive POSIX execv/setenv or a
# POSIX shell, so they cannot be built for Windows.
usdgen_add_unix_only_test(testUsdGenInstallTree testUsdGenInstallTree)
set_tests_properties(testUsdGenInstallTree PROPERTIES
    LABELS "T0;install")
usdgen_add_unix_only_test(testUsdGenConsumer testUsdGenConsumer)
set_tests_properties(testUsdGenConsumer PROPERTIES
    LABELS "T0;install")
