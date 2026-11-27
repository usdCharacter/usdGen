# usdGen M0 consumer probe (plan §6.2; sol S-5/X-4).
#
# A minimal downstream CMake project proving the installed package config is
# usable. testUsdGenConsumer (tests/api/testUsdGenConsumer.cpp) configures and
# builds this directory against a scratch install prefix
# (USDGEN_INSTALL_PREFIX) and runs the resulting binary.
#
# Order: find_package(pxr) FIRST, then find_package(usdGen). pxrConfig is
# single-pass — a second find_package(pxr CONFIG) re-runs it and detonates at
# add_library(TBB::tbb IMPORTED) ("target already exists"). usdGenConfig
# guards its own find_dependency(pxr) with if(NOT TARGET usd), so once pxr is
# imported here, the usdGen find is safe in either order — but the consumer
# must not find pxr a second time.
cmake_minimum_required(VERSION 3.26)
project(usdGenConsumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pxr REQUIRED CONFIG)
find_package(usdGen REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE usdGen::usdGen)
