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
# Below the hosts: everything that becomes base/, core/, backends/ and
# primitives/. Today all of it lives in usdGen and usdGenMath.
set(USDGEN_BELOW_HOSTS_DIRS "libs/usdGenMath;libs/usdGen;libs/base;libs/core")
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

foreach(_t usdGen usdGenMath)
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

add_executable(testUsdGenWarmAllocations tests/testUsdGenWarmAllocations.cpp)
target_link_libraries(testUsdGenWarmAllocations PRIVATE usdGen)
add_test(NAME testUsdGenWarmAllocations COMMAND testUsdGenWarmAllocations)
set_tests_properties(testUsdGenWarmAllocations PROPERTIES
    LABELS "T0;perf;gate;gate:zero-alloc" RUN_SERIAL TRUE)
