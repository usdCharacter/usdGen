# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Third-party targets: nanoflann, SeExpr noise, vendored Ptex + zlib. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# Third-party: nanoflann (header-only, INTERFACE target)
# ---------------------------------------------------------------------------
add_library(nanoflann INTERFACE)
target_include_directories(nanoflann SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/nanoflann>
    $<INSTALL_INTERFACE:include>)
# N-7 (plan §1.2/§2.3): nanoflann's header is never installed. It is a
# private dependency of the usdGenMath archive (compiled hidden, stripped
# out of libusdGen.so by --exclude-libs,ALL), the install-tree test asserts
# no include/nanoflann* exists in the prefix, and the target is not in the
# export set either — so no install rule is emitted at all (supersedes the
# sol S-6 header-layout fix; the INSTALL_INTERFACE root above is unused by
# design).

# ---------------------------------------------------------------------------
# Third-party: SeExpr noise (M0 = Noise.cpp + headers only, S38)
# ---------------------------------------------------------------------------
add_library(usdGen_seexpr STATIC
    thirdparty/seexpr/SeExpr2/Noise.cpp)
target_include_directories(usdGen_seexpr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/seexpr>
    $<INSTALL_INTERFACE:include/seexpr>)
target_link_libraries(usdGen_seexpr PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(usdGen_seexpr PROPERTIES
    USDGEN_VENDORED ON
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
# GPU parity: Noise.cpp must not contract; the CUDA kernel pins its bit patterns with --fmad=false (mirrors usdGenMath).
# N-7 (plan §1.2): SeExpr2 headers are never installed — usdGen_seexpr only
# exists to compile Noise.cpp into the hidden archives, is never installed
# or exported, so no header rule is emitted (supersedes the sol S-6
# double-nesting fix; the INSTALL_INTERFACE root above is unused by design).

# --- BEGIN vendored ptex (usdGen_ptex) ---
# ---------------------------------------------------------------------------
# Third-party: Ptex 2.4.3 + zlib 1.3.1 (plan/07-look-maps-expressions.md §6)
# ---------------------------------------------------------------------------
# The OpenUSD prefix has no Ptex, and Storm's Ptex path is mesh-only, so
# usdGen samples .ptx files itself (usdGen/maps/ptexMap.h). Both archives are
# private: linked into the usdGen DLL only, compiled hidden, never installed
# or exported, and their headers are build-tree includes only. zlib exists
# solely for Ptex, so only usdGen_ptex sees its include directory.
set(_usdgen_zlib_dir "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/zlib")
add_library(usdGen_zlib STATIC
    "${_usdgen_zlib_dir}/adler32.c" "${_usdgen_zlib_dir}/compress.c"
    "${_usdgen_zlib_dir}/crc32.c" "${_usdgen_zlib_dir}/deflate.c"
    "${_usdgen_zlib_dir}/gzclose.c" "${_usdgen_zlib_dir}/gzlib.c"
    "${_usdgen_zlib_dir}/gzread.c" "${_usdgen_zlib_dir}/gzwrite.c"
    "${_usdgen_zlib_dir}/infback.c" "${_usdgen_zlib_dir}/inffast.c"
    "${_usdgen_zlib_dir}/inflate.c" "${_usdgen_zlib_dir}/inftrees.c"
    "${_usdgen_zlib_dir}/trees.c" "${_usdgen_zlib_dir}/uncompr.c"
    "${_usdgen_zlib_dir}/zutil.c")
target_include_directories(usdGen_zlib PRIVATE "${_usdgen_zlib_dir}")
# zconf.h is vendored as shipped; HAVE_UNISTD_H is what upstream's configure
# step would have written into it on POSIX hosts (gzlib.c needs lseek/read).
if(MSVC)
    target_compile_definitions(usdGen_zlib PRIVATE
        _CRT_SECURE_NO_DEPRECATE _CRT_NONSTDC_NO_DEPRECATE)
elseif(NOT WIN32)
    target_compile_definitions(usdGen_zlib PRIVATE HAVE_UNISTD_H)
endif()
set_target_properties(usdGen_zlib PROPERTIES
    USDGEN_VENDORED ON
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden)

set(_usdgen_ptex_dir "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ptex/ptex")
add_library(usdGen_ptex STATIC
    "${_usdgen_ptex_dir}/PtexCache.cpp"
    "${_usdgen_ptex_dir}/PtexFilters.cpp"
    "${_usdgen_ptex_dir}/PtexHalf.cpp"
    "${_usdgen_ptex_dir}/PtexReader.cpp"
    "${_usdgen_ptex_dir}/PtexSeparableFilter.cpp"
    "${_usdgen_ptex_dir}/PtexSeparableKernel.cpp"
    "${_usdgen_ptex_dir}/PtexTriangleFilter.cpp"
    "${_usdgen_ptex_dir}/PtexTriangleKernel.cpp"
    "${_usdgen_ptex_dir}/PtexUtils.cpp"
    "${_usdgen_ptex_dir}/PtexWriter.cpp")
# SYSTEM keeps Ptexture.h (which pulls in <Windows.h>) at the consumer's
# external warning level; BUILD_INTERFACE only, because nothing installs it.
target_include_directories(usdGen_ptex SYSTEM PUBLIC
    "$<BUILD_INTERFACE:${_usdgen_ptex_dir}>")
target_include_directories(usdGen_ptex PRIVATE "${_usdgen_zlib_dir}")
# PTEX_STATIC drops the dllimport/visibility attributes from PTEXAPI; it and
# PTEX_VENDOR (namespace Ptex::v2_4_usdGen, so a site's own Ptex in the same
# process cannot collide) must match in every TU that includes Ptexture.h.
target_compile_definitions(usdGen_ptex PUBLIC PTEX_STATIC PTEX_VENDOR=usdGen)
if(MSVC)
    target_compile_definitions(usdGen_ptex PRIVATE
        _CRT_SECURE_NO_DEPRECATE _CRT_NONSTDC_NO_DEPRECATE)
elseif(NOT WIN32)
    target_compile_definitions(usdGen_ptex PRIVATE HAVE_UNISTD_H)
endif()
target_link_libraries(usdGen_ptex
    PUBLIC Threads::Threads
    PRIVATE usdGen_zlib)
set_target_properties(usdGen_ptex PROPERTIES
    USDGEN_VENDORED ON
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
# --- END vendored ptex (usdGen_ptex) ---
