# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
# pxrConfig.cmake -- find_package(pxr) shim over a Houdini install's vendored
# OpenUSD.
#
# Houdini ships USD headers (toolkit/include), import libraries
# (custom/houdini/dsolib/libpxr_*.lib) and DLLs (bin/libpxr_*.dll) but no
# pxrConfig.cmake. Point USD_INSTALL_DIR at this directory and set HOUDINI_ROOT
# to configure usdGen against it:
#
#   cmake -S <usdgen> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release
#       -DUSD_INSTALL_DIR=<usdgen>/cmake/houdini
#       -DHOUDINI_ROOT=<houdini-root> ...
#
# Every libpxr_* import library becomes an imported target with the stock
# config's bare name (usd, usdGeom, tf, ...), with the stock config's
# INTERFACE_LINK_LIBRARIES edges remapped onto the Houdini equivalents
# (tbb12, python313, hboost_python313, osdCPU/osdGPU, opengl32), so consumer
# CMakeLists need no Houdini-specific branches. The default stock-OpenUSD
# configure path is untouched: this file is only read when USD_INSTALL_DIR
# points here.
#
# HOUDINI_ROOT resolution: the HOUDINI_ROOT cache variable first, then the
# HOUDINI_ROOT environment variable, then HFS.

if (NOT HOUDINI_ROOT AND DEFINED ENV{HOUDINI_ROOT})
    set(HOUDINI_ROOT "$ENV{HOUDINI_ROOT}")
endif()
if (NOT HOUDINI_ROOT AND DEFINED ENV{HFS})
    set(HOUDINI_ROOT "$ENV{HFS}")
endif()
if (NOT HOUDINI_ROOT)
    message(FATAL_ERROR
        "HOUDINI_ROOT is not set. Point it at the Houdini install whose "
        "vendored OpenUSD should be used, e.g. "
        "-DHOUDINI_ROOT=<houdini-root>.")
endif()
file(TO_CMAKE_PATH "${HOUDINI_ROOT}" _usdgen_hfs)

set(_usdgen_hfs_include "${_usdgen_hfs}/toolkit/include")
set(_usdgen_hfs_dsolib  "${_usdgen_hfs}/custom/houdini/dsolib")
set(_usdgen_hfs_bin     "${_usdgen_hfs}/bin")
foreach (_dir IN ITEMS "${_usdgen_hfs_include}" "${_usdgen_hfs_dsolib}"
                       "${_usdgen_hfs_bin}")
    if (NOT IS_DIRECTORY "${_dir}")
        message(FATAL_ERROR
            "HOUDINI_ROOT=${_usdgen_hfs} does not look like a Houdini "
            "install: ${_dir} is missing.")
    endif()
endforeach()
if (NOT EXISTS "${_usdgen_hfs_include}/pxr/pxr.h")
    message(FATAL_ERROR
        "No vendored OpenUSD headers under ${_usdgen_hfs_include}/pxr.")
endif()

# Version straight from the vendored headers, so #if PXR_VERSION guards and
# versioned messages stay honest (Houdini 22.0.459 carries 0.26.5).
file(STRINGS "${_usdgen_hfs_include}/pxr/pxr.h" _usdgen_major
     REGEX "#define PXR_MAJOR_VERSION [0-9]+")
file(STRINGS "${_usdgen_hfs_include}/pxr/pxr.h" _usdgen_minor
     REGEX "#define PXR_MINOR_VERSION [0-9]+")
file(STRINGS "${_usdgen_hfs_include}/pxr/pxr.h" _usdgen_patch
     REGEX "#define PXR_PATCH_VERSION [0-9]+")
string(REGEX REPLACE ".* " "" _usdgen_major "${_usdgen_major}")
string(REGEX REPLACE ".* " "" _usdgen_minor "${_usdgen_minor}")
string(REGEX REPLACE ".* " "" _usdgen_patch "${_usdgen_patch}")
set(pxr_VERSION
    "${_usdgen_major}.${_usdgen_minor}.${_usdgen_patch}")

# pxr headers pull Python.h (tf/pySafePython.h via vt/value.h), so every
# translation unit needs the Python includes, exactly as the stock config
# provides them. Houdini keeps the HDK-blessed copy under toolkit/include and
# the interpreter's own copy beside it; take whichever exists.
set(_usdgen_python_includes "")
foreach (_pyinc IN ITEMS "${_usdgen_hfs_include}/python3.13"
                         "${_usdgen_hfs}/python313/include")
    if (IS_DIRECTORY "${_pyinc}")
        list(APPEND _usdgen_python_includes "${_pyinc}")
    endif()
endforeach()
if (_usdgen_python_includes STREQUAL "")
    message(FATAL_ERROR
        "No Python includes found under ${_usdgen_hfs} "
        "(toolkit/include/python3.13 nor python313/include).")
endif()

set(_usdgen_pxr_include_dirs
    "${_usdgen_hfs_include}" ${_usdgen_python_includes})

# Transitive external dependencies, remapped from the stock config's
# find_dependency targets onto the Houdini files.
set(_usdgen_tbb_lib "${_usdgen_hfs_dsolib}/tbb12.lib")
set(_usdgen_python_lib "${_usdgen_hfs}/python313/libs/python313.lib")
set(_usdgen_boost_python_lib
    "${_usdgen_hfs_dsolib}/hboost_python313-mt-x64.lib")
set(_usdgen_osd_cpu_lib "${_usdgen_hfs_dsolib}/libosdCPU_md.lib")
set(_usdgen_osd_gpu_lib "${_usdgen_hfs_dsolib}/libosdGPU_md.lib")
foreach (_dep IN ITEMS _usdgen_tbb_lib _usdgen_python_lib
                      _usdgen_boost_python_lib)
    if (NOT EXISTS "${${_dep}}")
        message(FATAL_ERROR
            "Houdini pxr shim: ${${_dep}} is missing; cannot map the stock "
            "dependency graph.")
    endif()
endforeach()
# The OpenSubdiv statics are only pulled when a translation unit references
# OSD symbols; their absence merely drops those edges.
foreach (_dep IN ITEMS _usdgen_osd_cpu_lib _usdgen_osd_gpu_lib)
    if (NOT EXISTS "${${_dep}}")
        message(STATUS "Houdini pxr shim: ${${_dep}} not found; "
                       "dropping OpenSubdiv edges.")
        set(${_dep} "")
    endif()
endforeach()

# Pass 1: one imported target per vendored library.
file(GLOB _usdgen_implibs "${_usdgen_hfs_dsolib}/libpxr_*.lib")
if (_usdgen_implibs STREQUAL "")
    message(FATAL_ERROR
        "No libpxr_*.lib import libraries under ${_usdgen_hfs_dsolib}.")
endif()
list(SORT _usdgen_implibs)
set(_usdgen_pxr_targets "")
foreach (_implib IN LISTS _usdgen_implibs)
    get_filename_component(_stem "${_implib}" NAME_WE)
    string(REGEX REPLACE "^libpxr_" "" _name "${_stem}")
    if (TARGET ${_name})
        continue()
    endif()
    set(_dll "${_usdgen_hfs_bin}/libpxr_${_name}.dll")
    if (EXISTS "${_dll}")
        add_library(${_name} SHARED IMPORTED GLOBAL)
        set_target_properties(${_name} PROPERTIES
            IMPORTED_LOCATION "${_dll}"
            IMPORTED_IMPLIB "${_implib}")
    else()
        # A static-only helper: linking the archive directly is correct
        # whether it is a real static library or an import stub.
        add_library(${_name} STATIC IMPORTED GLOBAL)
        set_target_properties(${_name} PROPERTIES
            IMPORTED_LOCATION "${_implib}")
    endif()
    set_target_properties(${_name} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_usdgen_pxr_include_dirs}")
    list(APPEND _usdgen_pxr_targets "${_name}")
endforeach()

# Pass 2: the stock config's dependency edges, transcribed from the 26.08
# pxrTargets.cmake (target|dep|dep...; "|" stands in for the CMake list
# separator so each edge stays one list item). External find_dependency
# targets are spelled as the stock config spells them and remapped below.
set(_usdgen_pxr_edges
    "boost|Python3::Python"
    "arch|Ws2_32.lib|Dbghelp.lib"
    "tf|arch|python|Shlwapi.lib|TBB::tbb|Python3::Python"
    "gf|arch|tf|python|Python3::Python"
    "pegtl|arch"
    "js|tf"
    "trace|arch|js|tf|TBB::tbb"
    "work|tf|trace|TBB::tbb"
    "plug|arch|tf|js|trace|work|TBB::tbb"
    "vt|arch|tf|gf|trace|python|TBB::tbb|Python3::Python"
    "ts|vt|gf|tf"
    "ar|arch|js|tf|plug|vt|python|TBB::tbb|Python3::Python"
    "kind|tf|plug"
    "sdf|arch|tf|gf|pegtl|trace|ts|vt|work|ar|python|TBB::tbb|Python3::Python"
    "sdr|arch|plug|trace|tf|vt|work|ar|sdf"
    "pcp|tf|trace|vt|sdf|work|ar|python|TBB::tbb|Python3::Python"
    "usd|arch|kind|pcp|sdf|ar|plug|tf|trace|ts|vt|work|python|TBB::tbb|Python3::Python"
    "usdGeom|js|tf|plug|vt|sdf|trace|usd|work|TBB::tbb"
    "usdVol|tf|usd|usdGeom"
    "usdMedia|tf|vt|sdf|usd|usdGeom"
    "usdShade|tf|vt|js|sdf|sdr|usd|usdGeom|TBB::tbb"
    "usdLod|tf|gf|vt|sdf|usd|usdGeom"
    "usdLux|tf|vt|sdf|sdr|usd|usdGeom|usdShade"
    "usdProc|tf|usd|usdGeom"
    "usdProfiles|arch|tf|plug|vt|js|sdf|usd"
    "usdRender|gf|tf|usd|usdGeom|usdShade"
    "usdHydra|tf|usd|usdShade"
    "usdRi|tf|vt|sdf|usd|usdShade|usdGeom"
    "usdSemantics|tf|vt|usd"
    "usdSkel|arch|gf|tf|trace|vt|work|sdf|usd|usdGeom|TBB::tbb"
    "usdUI|tf|vt|sdf|usd"
    "usdUtils|arch|tf|gf|sdf|usd|usdGeom|usdShade|usdUI|TBB::tbb"
    "usdPhysics|tf|plug|vt|sdf|trace|usd|usdGeom|usdShade|work"
    "vdf|arch|gf|tf|trace|vt|work|TBB::tbb"
    "ef|vdf|arch|tf|trace|usd|work|TBB::tbb"
    "esf|arch|sdf|tf|vt|usd"
    "esfUsd|arch|esf|tf|sdf|usd"
    "exec|ef|esf|tf|trace|ts|sdf|usd|vdf|vt|TBB::tbb"
    "execUsd|esf|esfUsd|exec|tf|trace|sdf|usd"
    "execGeom|gf|tf|execUsd|usdGeom"
    "execIr|execGeom|execUsd|gf|sdf|tf|usd|vt"
    "usdValidation|sdf|plug|tf|gf|usd|work"
    "usdGeomValidators|tf|plug|sdf|usd|usdGeom|usdValidation"
    "usdLuxValidators|tf|plug|sdf|usd|sdr|usdShade|usdLux|usdValidation"
    "usdPhysicsValidators|tf|plug|sdf|usd|usdGeom|usdPhysics|usdValidation"
    "usdShadeValidators|tf|plug|sdf|usd|sdr|usdShade|usdValidation"
    "usdSkelValidators|tf|plug|sdf|usd|usdSkel|usdValidation"
    "usdUtilsValidators|tf|plug|sdf|usd|usdUtils|usdValidation"
    "garch|arch|tf|OpenGL::GL"
    "hf|plug|tf|trace"
    "hio|arch|js|plug|tf|vt|trace|ar|hf"
    "cameraUtil|tf|gf"
    "pxOsd|tf|gf|vt|OpenSubdiv::osdCPU_static"
    "geomUtil|arch|gf|tf|vt|pxOsd"
    "glf|ar|arch|garch|gf|hf|hio|plug|tf|trace|sdf"
    "hgi|gf|plug|tf|hio"
    "hgiGL|arch|garch|hf|hgi|tf|trace"
    "hgiInterop|gf|tf|hgi|vt|garch"
    "hd|plug|tf|trace|vt|work|sdf|cameraUtil|hf|pxOsd|sdr|TBB::tbb"
    "hdar|hd|ar"
    "hdGp|hd|hf|TBB::tbb"
    "hdsi|plug|tf|trace|vt|work|sdf|cameraUtil|usdShade|usdVol|geomUtil|hf|hd|pxOsd|TBB::tbb"
    "hdSt|hio|garch|glf|hd|hdsi|hgiGL|hgiInterop|sdr|tf|trace|TBB::tbb|OpenSubdiv::osdCPU_static|OpenSubdiv::osdGPU_static"
    "hdx|plug|tf|vt|gf|work|garch|glf|pxOsd|hd|hdSt|hgi|hgiInterop|cameraUtil|sdf"
    "usdImaging|gf|tf|plug|trace|vt|work|geomUtil|hd|hdar|hdsi|hio|pxOsd|sdf|usd|usdGeom|usdLux|usdRender|usdShade|usdSkel|usdVol|ar|TBB::tbb"
    "usdExecImaging|execGeom|execIr|execUsd|hd|sdf|tf|trace|usd|usdGeom|vt"
    "usdImagingGL|gf|tf|plug|trace|vt|work|hio|garch|glf|hd|hdsi|hdx|pxOsd|sdf|sdr|usd|usdGeom|usdHydra|usdShade|usdImaging|usdExecImaging|ar"
    "usdProcImaging|usdImaging|usdProc"
    "usdSkelImaging|hio|hd|usdImaging|usdSkel"
    "usdVolImaging|usdImaging"
    "usdIrImaging|execIr|gf|hd|tf|usd|usdGeom|usdImaging"
    "usdAppUtils|garch|gf|hio|sdf|tf|usd|usdGeom|usdImagingGL"
    "usdviewq|tf|usd|usdGeom"
)
set(_usdgen_skipped_edges "")
foreach (_edge IN LISTS _usdgen_pxr_edges)
    string(REPLACE "|" ";" _parts "${_edge}")
    list(GET _parts 0 _target)
    list(REMOVE_AT _parts 0)
    if (NOT TARGET ${_target})
        continue()
    endif()
    set(_mapped "")
    foreach (_dep IN LISTS _parts)
        if (_dep STREQUAL _target)
            # The stock export's self-edge on boost; a real one would be a
            # configure error, and it carries no information anyway.
            continue()
        elseif (_dep STREQUAL "TBB::tbb")
            list(APPEND _mapped "${_usdgen_tbb_lib}")
        elseif (_dep STREQUAL "Python3::Python")
            list(APPEND _mapped "${_usdgen_python_lib}")
        elseif (_dep STREQUAL "OpenGL::GL")
            list(APPEND _mapped "opengl32.lib")
        elseif (_dep STREQUAL "OpenSubdiv::osdCPU_static")
            if (NOT _usdgen_osd_cpu_lib STREQUAL "")
                list(APPEND _mapped "${_usdgen_osd_cpu_lib}")
            endif()
        elseif (_dep STREQUAL "OpenSubdiv::osdGPU_static")
            if (NOT _usdgen_osd_gpu_lib STREQUAL "")
                list(APPEND _mapped "${_usdgen_osd_gpu_lib}")
            endif()
        elseif (TARGET ${_dep})
            list(APPEND _mapped "${_dep}")
        elseif (_dep MATCHES "\\.lib$")
            # A Windows system library (Ws2_32, Dbghelp, Shlwapi): the
            # linker resolves it from its default search path.
            list(APPEND _mapped "${_dep}")
        else()
            # A 26.08-only library with no 26.05 counterpart (or vice
            # versa); dropping the edge only matters to dependents that
            # reference its symbols, which then fail loudly at link time.
            list(APPEND _usdgen_skipped_edges "${_target}|${_dep}")
        endif()
    endforeach()
    if (TARGET boost AND _target STREQUAL "boost")
        # The vendored Boost.Python: pxr_boost::python symbols live here,
        # not in libpxr_boost.
        list(APPEND _mapped "${_usdgen_boost_python_lib}")
    endif()
    set_target_properties(${_target} PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_mapped}")
endforeach()
if (NOT _usdgen_skipped_edges STREQUAL "")
    list(REMOVE_DUPLICATES _usdgen_skipped_edges)
    message(STATUS
        "Houdini pxr shim: dropped edges onto libraries this USD has no "
        "import target for: ${_usdgen_skipped_edges}")
endif()

list(LENGTH _usdgen_pxr_targets _usdgen_pxr_target_count)

# usdGen names TBB::tbb directly, so provide the
# imported target instead of only threading the library through edges. Inert
# for consumers that never name it.
if (NOT TARGET TBB::tbb)
    add_library(TBB::tbb SHARED IMPORTED GLOBAL)
    set_target_properties(TBB::tbb PROPERTIES
        IMPORTED_LOCATION "${_usdgen_hfs_bin}/tbb12.dll"
        IMPORTED_IMPLIB "${_usdgen_tbb_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${_usdgen_hfs_include}")
endif()

set(pxr_FOUND TRUE)
message(STATUS
    "pxr ${pxr_VERSION} (Houdini vendored): "
    "${_usdgen_pxr_target_count} imported targets from ${_usdgen_hfs}")

unset(_usdgen_hfs)
unset(_usdgen_hfs_include)
unset(_usdgen_hfs_dsolib)
unset(_usdgen_hfs_bin)
unset(_usdgen_major)
unset(_usdgen_minor)
unset(_usdgen_patch)
unset(_usdgen_python_includes)
unset(_usdgen_pxr_include_dirs)
unset(_usdgen_tbb_lib)
unset(_usdgen_python_lib)
unset(_usdgen_boost_python_lib)
unset(_usdgen_osd_cpu_lib)
unset(_usdgen_osd_gpu_lib)
unset(_usdgen_implibs)
unset(_implib)
unset(_stem)
unset(_name)
unset(_dll)
unset(_dep)
unset(_dir)
unset(_pyinc)
unset(_usdgen_pxr_targets)
unset(_usdgen_pxr_target_count)
unset(_usdgen_pxr_edges)
unset(_edge)
unset(_parts)
unset(_target)
unset(_mapped)
unset(_usdgen_skipped_edges)
