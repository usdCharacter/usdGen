# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
# Standalone offline documentation tools. Configure against the existing
# usdGen build; these targets do not enter the main project's build graph.
cmake_minimum_required(VERSION 3.24)
project(usdGenDocsBaker LANGUAGES CXX)

if(NOT USD_INSTALL_DIR OR NOT USDGEN_BUILD_DIR)
    message(FATAL_ERROR "Set USD_INSTALL_DIR and USDGEN_BUILD_DIR")
endif()
if(NOT EXISTS "${USDGEN_BUILD_DIR}/CMakeCache.txt")
    message(FATAL_ERROR "USDGEN_BUILD_DIR must point to a configured usdGen build")
endif()
if(WIN32)
    set(_test_utils "${USDGEN_BUILD_DIR}/usdGenTestUtilsHd.lib")
else()
    set(_test_utils "${USDGEN_BUILD_DIR}/libusdGenTestUtilsHd.a")
endif()
if(NOT EXISTS "${_test_utils}")
    message(FATAL_ERROR "Offline baker needs ${_test_utils}; build usdGenTestUtilsHd in the main usdGen build first")
endif()

get_filename_component(USDGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(usdGenBakeGroom bake_groom.cpp)
add_executable(usdGenCollideFarOracle collide_far_oracle.cpp)
target_compile_features(usdGenCollideFarOracle PRIVATE cxx_std_17)
target_compile_definitions(usdGenCollideFarOracle PRIVATE NOMINMAX)
target_include_directories(usdGenCollideFarOracle PRIVATE
    "${USDGEN_SOURCE_DIR}/libs/core/usdGen"
    "${USDGEN_SOURCE_DIR}/libs/core/primitives/usdGenCurves"
    "${USDGEN_SOURCE_DIR}/libs/core/primitives/usdGenSurface"
    "${USDGEN_SOURCE_DIR}/libs/core/primitives/usdGenExpr"
    "${USDGEN_SOURCE_DIR}/libs/core/backends/usdGenCuda"
    "${USDGEN_SOURCE_DIR}/libs/core/backends/usdGenVulkan"
    "${USDGEN_BUILD_DIR}/shim"
    "${USD_INSTALL_DIR}/include")
target_link_directories(usdGenCollideFarOracle PRIVATE
    "${USDGEN_BUILD_DIR}"
    "${USD_INSTALL_DIR}/lib")
if(WIN32)
    file(GLOB _oracle_usd_libs "${USD_INSTALL_DIR}/lib/usd_*.lib")
    target_link_libraries(usdGenCollideFarOracle PRIVATE
        usdGen ${_oracle_usd_libs} "${USD_INSTALL_DIR}/lib/tbb.lib"
        Shlwapi Ws2_32 Dbghelp)
else()
    file(GLOB _oracle_usd_libs "${USD_INSTALL_DIR}/lib/libusd_*.so*")
    target_link_libraries(usdGenCollideFarOracle PRIVATE usdGen ${_oracle_usd_libs})
endif()
target_compile_features(usdGenBakeGroom PRIVATE cxx_std_17)
target_compile_definitions(usdGenBakeGroom PRIVATE NOMINMAX)
target_include_directories(usdGenBakeGroom PRIVATE
    "${USDGEN_SOURCE_DIR}/libs/core/usdGen"
    "${USDGEN_SOURCE_DIR}/libs/core/primitives/usdGenCurves"
    "${USDGEN_SOURCE_DIR}/libs/core/primitives/usdGenSurface"
    "${USDGEN_SOURCE_DIR}/libs/core/primitives/usdGenExpr"
    "${USDGEN_SOURCE_DIR}/libs/core/backends/usdGenCuda"
    "${USDGEN_SOURCE_DIR}/libs/core/backends/usdGenVulkan"
    "${USDGEN_BUILD_DIR}/shim"
    "${USDGEN_SOURCE_DIR}/libs/hosts/usdGenHydra"
    "${USD_INSTALL_DIR}/include")
file(STRINGS "${USDGEN_BUILD_DIR}/CMakeCache.txt" _python_include_line
    REGEX "^_Python3_INCLUDE_DIR:INTERNAL=")
if(_python_include_line)
    string(REGEX REPLACE "^[^=]*=" "" _python_include "${_python_include_line}")
    target_include_directories(usdGenBakeGroom PRIVATE "${_python_include}")
    target_include_directories(usdGenCollideFarOracle PRIVATE "${_python_include}")
    get_filename_component(_python_root "${_python_include}" DIRECTORY)
    target_link_directories(usdGenBakeGroom PRIVATE "${_python_root}/libs")
    target_link_directories(usdGenCollideFarOracle PRIVATE "${_python_root}/libs")
endif()

# Native operator-plane fixtures for the reference manual. Keep this in the
# standalone documentation build, alongside the offline groom baker.
add_executable(usdGenUtilityFixtures utility_fixtures.cpp)
target_compile_features(usdGenUtilityFixtures PRIVATE cxx_std_17)
target_compile_definitions(usdGenUtilityFixtures PRIVATE NOMINMAX)
target_include_directories(usdGenUtilityFixtures PRIVATE
    "${USDGEN_SOURCE_DIR}/libs/core/usdGen"
    "${USDGEN_SOURCE_DIR}/libs/core/primitives/usdGenCurves"
    "${USDGEN_SOURCE_DIR}/libs/core/primitives/usdGenSurface"
    "${USDGEN_SOURCE_DIR}/libs/core/primitives/usdGenExpr"
    "${USDGEN_SOURCE_DIR}/libs/core/backends/usdGenCuda"
    "${USDGEN_SOURCE_DIR}/libs/core/backends/usdGenVulkan"
    "${USDGEN_BUILD_DIR}/shim"
    "${USDGEN_SOURCE_DIR}/libs/hosts/usdGenHydra"
    "${USD_INSTALL_DIR}/include")
if(_python_include_line)
    target_include_directories(usdGenUtilityFixtures PRIVATE "${_python_include}")
    target_link_directories(usdGenUtilityFixtures PRIVATE "${_python_root}/libs")
endif()
target_link_directories(usdGenUtilityFixtures PRIVATE
    "${USDGEN_BUILD_DIR}"
    "${USD_INSTALL_DIR}/lib")
if(WIN32)
    file(GLOB _usd_libs "${USD_INSTALL_DIR}/lib/usd_*.lib")
    target_link_libraries(usdGenUtilityFixtures PRIVATE
        usdGenTestUtilsHd usdGenImaging usdGen usdGenExecutionResources
        ${_usd_libs} "${USD_INSTALL_DIR}/lib/tbb.lib" Shlwapi Ws2_32 Dbghelp)
else()
    file(GLOB _usd_libs "${USD_INSTALL_DIR}/lib/libusd_*.so*")
    target_link_libraries(usdGenUtilityFixtures PRIVATE
        usdGenTestUtilsHd usdGenImaging usdGen usdGenExecutionResources ${_usd_libs})
endif()
target_link_directories(usdGenBakeGroom PRIVATE
    "${USDGEN_BUILD_DIR}"
    "${USD_INSTALL_DIR}/lib")
if(WIN32)
    file(GLOB _usd_libs "${USD_INSTALL_DIR}/lib/usd_*.lib")
    target_link_libraries(usdGenBakeGroom PRIVATE
        usdGenTestUtilsHd usdGenImaging usdGen usdGenExecutionResources
        ${_usd_libs} "${USD_INSTALL_DIR}/lib/tbb.lib" Shlwapi Ws2_32 Dbghelp)
else()
    file(GLOB _usd_libs "${USD_INSTALL_DIR}/lib/libusd_*.so*")
    target_link_libraries(usdGenBakeGroom PRIVATE
        usdGenTestUtilsHd usdGenImaging usdGen usdGenExecutionResources ${_usd_libs})
endif()
