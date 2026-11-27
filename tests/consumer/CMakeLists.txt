# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# usdGen consumer probe (plan §6.2; 12.5). A downstream CMake project built
# against an install prefix: find_package(usdGen) alone, no OpenUSD on the
# include or link line. testUsdGenConsumer (tests/api/testUsdGenConsumer.cpp)
# configures and builds it against USDGEN_INSTALL_PREFIX and runs it.
cmake_minimum_required(VERSION 3.16)
project(usdGenConsumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(usdGen REQUIRED CONFIG)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE usdGen::usdGen)
