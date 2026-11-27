cmake_minimum_required(VERSION 3.26)
project(usdGen VERSION 0.1.0 LANGUAGES C CXX)

# Keep the Windows SDK's min/max macros out of C++ standard-library and
# OpenUSD headers. The definition is a no-op on other platforms.
if(WIN32)
    add_compile_definitions(NOMINMAX)
endif()

include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Test options must exist before backend dependency discovery, including the
# Vulkan test-only configuration on a fresh cache.
if(WIN32)
    option(USDGEN_BUILD_TESTS "Build the usdGen test and benchmark targets" OFF)
else()
    option(USDGEN_BUILD_TESTS "Build the usdGen test and benchmark targets" ON)
endif()

# The USD install the tests RUN against: its bin/lib go on the test
# loader path and its plugin/python/tool paths feed the test
# environment. This is USD_INSTALL_DIR unless that is a CMake shim
# rather than a real install (as with the Houdini build, where the
# shim lives in cmake/houdini but the binaries come from HOUDINI_ROOT).
set(USDGEN_TEST_USD_DIR "${USD_INSTALL_DIR}"
    CACHE PATH "USD install whose binaries the tests run against")

# ---------------------------------------------------------------------------
# OpenUSD discovery
# ---------------------------------------------------------------------------
set(USD_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../OpenUSD_26_08"
    CACHE PATH "OpenUSD installation prefix")

# rigExec is optional (S44).  It is expected to be *before* pxr so that the
# rigExec config does not re-enter pxrConfig a second time (CMake 3.28
# double-pxrConfig trap, plan §3.2).
set(RIGEXEC_INSTALL_DIR "" CACHE PATH "rigExec install prefix (optional)")
set(USDGEN_WITH_RIGEXEC OFF CACHE BOOL "Link rigExec::rigExecMath (optional)")
if(USDGEN_WITH_RIGEXEC AND RIGEXEC_INSTALL_DIR)
    find_package(rigExec CONFIG QUIET PATHS "${RIGEXEC_INSTALL_DIR}")
endif()

if(NOT TARGET usd)
    # OpenUSD's pxrTargets reference OpenGL::GL (via garch's interface). Find
    # it first so the imported target exists before pxrConfig resolves.
    find_package(OpenGL REQUIRED)
    # build_usd.py co-installs OpenSubdiv (and, on Unix, OpenSubdivConfig.cmake)
    # into the same prefix as OpenUSD. find_package(pxr PATHS ...) does not put
    # that prefix on CMAKE_PREFIX_PATH, so pxrConfig's find_dependency(OpenSubdiv
    # CONFIG) cannot see the package unless we point at it. Prepending the
    # prefix is the relocatable equivalent of how OpenUSD itself was configured.
    list(PREPEND CMAKE_PREFIX_PATH "${USD_INSTALL_DIR}")
    if(NOT OpenSubdiv_DIR)
        foreach(_usdgen_osd_cmake
                "${USD_INSTALL_DIR}/lib/cmake/OpenSubdiv"
                "${USD_INSTALL_DIR}/lib64/cmake/OpenSubdiv")
            if(EXISTS "${_usdgen_osd_cmake}/OpenSubdivConfig.cmake")
                set(OpenSubdiv_DIR "${_usdgen_osd_cmake}")
                break()
            endif()
        endforeach()
        unset(_usdgen_osd_cmake)
    endif()
    # Windows (and any other prefix that ships OSD binaries without a CMake
    # package): skip pxrConfig's rediscovery and reconstruct the imported
    # targets that pxrTargets names. Linux pxrTargets use OpenSubdiv::osdCPU;
    # Windows uses OpenSubdiv::osdCPU_static.
    if(NOT OpenSubdiv_DIR)
        if(NOT DEFINED PXR_FIND_OPENSUBDIV_IN_CONFIG)
            set(PXR_FIND_OPENSUBDIV_IN_CONFIG OFF)
        endif()
        foreach(_usdgen_osd_name osdCPU osdGPU)
            if(NOT TARGET OpenSubdiv::${_usdgen_osd_name}_static AND
               EXISTS "${USD_INSTALL_DIR}/lib/${_usdgen_osd_name}.lib")
                add_library(OpenSubdiv::${_usdgen_osd_name}_static STATIC IMPORTED)
                set_target_properties(OpenSubdiv::${_usdgen_osd_name}_static
                    PROPERTIES
                        IMPORTED_LOCATION "${USD_INSTALL_DIR}/lib/${_usdgen_osd_name}.lib"
                        INTERFACE_INCLUDE_DIRECTORIES "${USD_INSTALL_DIR}/include"
                        INTERFACE_COMPILE_DEFINITIONS
                            "OPENSUBDIV_HAS_PATCH_SHADER_SOURCE_GLSL;OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK;OPENSUBDIV_HAS_GLSL_COMPUTE;OPENSUBDIV_HAS_OPENGL")
            endif()
            if(NOT TARGET OpenSubdiv::${_usdgen_osd_name})
                foreach(_usdgen_osd_file
                        "lib${_usdgen_osd_name}${CMAKE_SHARED_LIBRARY_SUFFIX}"
                        "lib${_usdgen_osd_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
                    if(EXISTS "${USD_INSTALL_DIR}/lib/${_usdgen_osd_file}")
                        if(_usdgen_osd_file STREQUAL
                           "lib${_usdgen_osd_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
                            add_library(OpenSubdiv::${_usdgen_osd_name} SHARED IMPORTED)
                        else()
                            add_library(OpenSubdiv::${_usdgen_osd_name} STATIC IMPORTED)
                        endif()
                        set_target_properties(OpenSubdiv::${_usdgen_osd_name}
                            PROPERTIES
                                IMPORTED_LOCATION
                                    "${USD_INSTALL_DIR}/lib/${_usdgen_osd_file}"
                                INTERFACE_INCLUDE_DIRECTORIES
                                    "${USD_INSTALL_DIR}/include")
                        break()
                    endif()
                endforeach()
            endif()
        endforeach()
        unset(_usdgen_osd_name)
        unset(_usdgen_osd_file)
    endif()
    find_package(pxr REQUIRED CONFIG PATHS "${USD_INSTALL_DIR}" NO_DEFAULT_PATH)
endif()

# The OpenSubdiv CPU library the pxr prefix provides: OpenSubdiv::osdCPU
# (Linux, shared) or OpenSubdiv::osdCPU_static (Windows). Libraries that call
# Far/Bfr link it directly instead of through pxOsd.
function(usdgen_opensubdiv_target _out)
    if(TARGET OpenSubdiv::osdCPU)
        set(${_out} OpenSubdiv::osdCPU PARENT_SCOPE)
    elseif(TARGET OpenSubdiv::osdCPU_static)
        set(${_out} OpenSubdiv::osdCPU_static PARENT_SCOPE)
    else()
        message(FATAL_ERROR "usdGen: no OpenSubdiv::osdCPU or OpenSubdiv::osdCPU_static "
            "target; OpenUSD must be built with OpenSubdiv (USD_INSTALL_DIR=${USD_INSTALL_DIR})")
    endif()
endfunction()

# Threads are used by the expression frontend. No system zlib is required:
# requiring one made a standard Windows build_usd.py OpenUSD prefix unusable
# because that prefix intentionally omits zlib when no USD feature needs it.
# The only zlib consumer, the vendored Ptex, links the private vendored copy
# (usdGen_zlib, thirdparty/zlib).
find_package(Threads REQUIRED)
# tbb/task.h, reached through the OpenUSD headers, prints a deprecation
# banner on every translation unit unless told not to.
add_compile_definitions(TBB_SUPPRESS_DEPRECATED_MESSAGES=1)

# ---------------------------------------------------------------------------
# Compile hardening
# ---------------------------------------------------------------------------
# Vendored sources (thirdparty/) are compiled as-is and carry the
# USDGEN_VENDORED target property, which keeps the warning level off them;
# the generator expressions below read that property on the consuming target.
set(_usdgen_first_party "$<NOT:$<BOOL:$<TARGET_PROPERTY:USDGEN_VENDORED>>>")
if(MSVC)
    # C4251 fires for every std:: member of an exported class; the DSOs here
    # are built and consumed with one toolchain, which is the case the warning
    # is not about. /external:W0 keeps third-party (SYSTEM) headers quiet.
    add_compile_options(
        "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,${_usdgen_first_party}>:/W4;/wd4251;/external:W0>"
        "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<NOT:${_usdgen_first_party}>>:/W0>")
    # The tests use getenv/fopen; the "_s" variants are not portable.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(
        "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,${_usdgen_first_party}>:-Wall;-Wextra;-Wno-cpp>")
endif()

# Release adds NDEBUG through the per-config flag strings. The CUDA oracle
# tests rely on assert, and undefining it again (-UNDEBUG) makes cl warn D9025
# on each of their translation units. Strip it from the flag strings instead
# and re-add it per target, except where USDGEN_KEEP_ASSERTS is set.
foreach(_config RELEASE RELWITHDEBINFO MINSIZEREL)
    foreach(_lang C CXX)
        string(REGEX REPLACE "[/-]DNDEBUG" "" CMAKE_${_lang}_FLAGS_${_config}
            "${CMAKE_${_lang}_FLAGS_${_config}}")
    endforeach()
endforeach()
add_compile_definitions(
    "$<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<BOOL:$<TARGET_PROPERTY:USDGEN_KEEP_ASSERTS>>>>:NDEBUG>")

# $ORIGIN is ELF-only.  Use the native loader spelling where applicable and
# leave Windows without an RPATH.
if(UNIX AND NOT APPLE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
elseif(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

# The host's dynamic-loader variable, which the test registrations extend.
# The three hosts disagree on the name, and on Windows it is PATH itself.
# No companion path-list separator is defined on purpose: test environments
# are built with CTest's ENVIRONMENT_MODIFICATION path_list_* operations,
# which join using the host's native separator, so nothing in this build
# hard-codes ';' or ':' (see the helpers in the USDGEN_BUILD_TESTS block).
if(WIN32)
    set(USDGEN_RUNTIME_LIBRARY_PATH_ENV "PATH")
elseif(APPLE)
    set(USDGEN_RUNTIME_LIBRARY_PATH_ENV "DYLD_LIBRARY_PATH")
else()
    set(USDGEN_RUNTIME_LIBRARY_PATH_ENV "LD_LIBRARY_PATH")
endif()

# ---------------------------------------------------------------------------
# Per-target cache variables (plan §3.5.2)
# ---------------------------------------------------------------------------
set(USDGEN_FP_CONTRACT off CACHE STRING
    "Floating-point contraction for groom math and host expression evaluation (GPU parity) (off|fast)")
set_property(CACHE USDGEN_FP_CONTRACT PROPERTY STRINGS off fast)

set(USDGEN_INSTALL_LIBDIR     "lib"         CACHE PATH "lib dir")
set(USDGEN_INSTALL_PLUGINDIR  "lib/usd"     CACHE PATH "usd plugin dir")
set(USDGEN_INSTALL_PYTHONDIR  "lib/python"  CACHE PATH "python pkg dir")
set(USDGEN_INSTALL_CMAKEDIR   "lib/cmake/usdGen" CACHE PATH "cmake config dir")

# Step 12.3f: every shipped library is built hidden; what another DSO reaches
# carries the library's export macro (export.h). Private classes that tests
# drive carry the *_INTERNAL_API macros, exported only with this option (a
# PUBLIC build-tree definition of every library, so importers agree). OFF is
# the release surface: api/ and api/ext/ only; most test targets then fail to
# link.
option(USDGEN_EXPORT_INTERNALS "Export the private classes tests drive" ${USDGEN_BUILD_TESTS})
if(USDGEN_BUILD_TESTS AND NOT USDGEN_EXPORT_INTERNALS)
    message(STATUS "usdGen: USDGEN_EXPORT_INTERNALS=OFF; white-box test targets do not link (build the libraries, hosts and api tests)")
endif()
function(usdgen_hidden_library _target)
    set_target_properties(${_target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    if(USDGEN_EXPORT_INTERNALS)
        target_compile_definitions(${_target} PUBLIC $<BUILD_INTERFACE:USDGEN_EXPORT_INTERNALS>)
    endif()
    if(MSVC)
        # STL members and bases of exported classes.
        target_compile_options(${_target} PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:/wd4251> $<$<COMPILE_LANGUAGE:CXX>:/wd4275>)
    endif()
    # ELF resolves undefined symbols at dlopen, not at link: fail the link.
    if(NOT WIN32 AND NOT APPLE)
        target_link_options(${_target} PRIVATE "LINKER:--no-undefined")
    endif()
endfunction()

# Targets, one file per area, in dependency order. include() keeps
# CMAKE_CURRENT_SOURCE_DIR at the root, so relative paths resolve from here.
include(cmake/targets/thirdparty.cmake)
include(cmake/targets/math.cmake)
include(cmake/targets/engine.cmake)
include(cmake/targets/primitives.cmake)
include(cmake/targets/hosts.cmake)
include(cmake/targets/shaders.cmake)
include(cmake/targets/python.cmake)
include(cmake/targets/cuda.cmake)
include(cmake/targets/vulkan.cmake)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(USDGEN_BUILD_TESTS)
include(cmake/targets/tests/core.cmake)
include(cmake/targets/tests/vulkan.cmake)
include(cmake/targets/tests/session.cmake)
include(cmake/targets/tests/cuda.cmake)
include(cmake/targets/tests/gates.cmake)
include(cmake/targets/tests/engine.cmake)
include(cmake/targets/tests/hosts.cmake)
include(cmake/targets/tests/operators.cmake)
include(cmake/targets/tests/pomade.cmake)
include(cmake/targets/tests/storm.cmake)
include(cmake/targets/tests/install.cmake)

# ---------------------------------------------------------------------------
# tests/internal: white-box tests of private classes (USDGEN_*_INTERNAL_API).
# Labelled "internal"; without USDGEN_EXPORT_INTERNALS they are not built and
# their tests are disabled, so the api/hosts tests are what an OFF tree runs.
# ---------------------------------------------------------------------------
get_property(_usdgen_all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
set(_usdgen_internal_targets "")
set(_usdgen_api_targets "")
foreach(_tgt IN LISTS _usdgen_all_targets)
    get_target_property(_type ${_tgt} TYPE)
    if(NOT _type STREQUAL "EXECUTABLE")
        continue()
    endif()
    get_target_property(_srcs ${_tgt} SOURCES)
    foreach(_src IN LISTS _srcs)
        # tests/api: what the hidden (internals OFF) build must pass; label "api".
        if(_src MATCHES "(^|/)tests/api/")
            list(APPEND _usdgen_api_targets ${_tgt})
            break()
        endif()
        if(_src MATCHES "(^|/)tests/internal/")
            list(APPEND _usdgen_internal_targets ${_tgt})
            if(NOT USDGEN_EXPORT_INTERNALS)
                set_target_properties(${_tgt} PROPERTIES EXCLUDE_FROM_ALL ON)
            endif()
            break()
        endif()
    endforeach()
endforeach()
get_property(_usdgen_all_tests DIRECTORY PROPERTY TESTS)
foreach(_t IN LISTS _usdgen_all_tests)
    get_property(_exe GLOBAL PROPERTY USDGEN_TEST_EXE_${_t})
    if(_exe IN_LIST _usdgen_api_targets)
        set_property(TEST ${_t} APPEND PROPERTY LABELS api)
    endif()
    if(_exe IN_LIST _usdgen_internal_targets)
        set_property(TEST ${_t} APPEND PROPERTY LABELS internal)
        if(NOT USDGEN_EXPORT_INTERNALS)
            set_tests_properties(${_t} PROPERTIES DISABLED ON)
        endif()
    endif()
endforeach()

# ---------------------------------------------------------------------------
# Loader search path for every registered test.
#
# This is a sweep rather than a per-registration property because *all* tests
# need it, including the T0 engine tests that carry no other environment: they
# link usdGen, which is a shared library, and OpenUSD's libraries come along
# with it. On Windows, where there is no RPATH, omitting this is what made the
# whole suite abort with STATUS_DLL_NOT_FOUND (0xc0000135) before the test body
# ever ran. Sweeping the registered list also keeps tests added later correct
# without every registration having to remember.
# ---------------------------------------------------------------------------
usdgen_test_path_list_ops(_usdgen_runtime_ops
    "${USDGEN_RUNTIME_LIBRARY_PATH_ENV}" path_list_prepend
    ${USDGEN_TEST_RUNTIME_DIRS})
get_property(_usdgen_registered_tests DIRECTORY PROPERTY TESTS)
foreach(_t IN LISTS _usdgen_registered_tests)
    set_property(TEST ${_t} APPEND PROPERTY
        ENVIRONMENT_MODIFICATION "${_usdgen_runtime_ops}")
endforeach()

endif() # USDGEN_BUILD_TESTS

include(cmake/targets/install.cmake)
