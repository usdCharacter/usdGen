# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Install rules and the package config. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
# 12.5 (D16): usdGen is the one exported target (usdGen::usdGen, no pxr in
# its interface). usdGenMath, usdGen_seexpr and nanoflann are private
# archives and are not installed (N-7).
install(TARGETS usdGen
    EXPORT usdGenTargets
    RUNTIME DESTINATION "${USDGEN_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${USDGEN_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${USDGEN_INSTALL_LIBDIR}")
# Everything else ships as a runtime library without an export: the
# extensions usdGen loads from its own directory (usdGenExtensions.txt), the
# hosts Plug loads, and Pomade. Their import libraries are not installed.
set(_usdgen_runtime_targets usdGenSurface usdGenExpr usdGenCurves usdGenCuda
    usdGenUsd usdGenHydraCurves usdGenImaging usdGenSchema usdGenPomade)
if(TARGET usdGenVulkan)
    list(APPEND _usdgen_runtime_targets usdGenVulkan)
endif()
install(TARGETS ${_usdgen_runtime_targets}
    RUNTIME DESTINATION "${USDGEN_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${USDGEN_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${USDGEN_INSTALL_LIBDIR}"
        COMPONENT usdGenImportLibraries EXCLUDE_FROM_ALL)
unset(_usdgen_runtime_targets)


# Schema + imaging plugin resources.
#
# The codeless schema declares implementsComputeExtent on UsdGenDescription, so
# Plug must be able to dlopen the registering library. A "resource" plugin is
# treated as already-loaded and never dlopen'd, so the build-tree copy is
# emitted as a "library" plugin carrying a LibraryPath and the
# implementsComputeExtent flag (mirrors usdRig, CMakeLists ~410-458). The
# checked-in copy under plugin/ stays data-only; generation happens in the
# "Build-tree plugin resources" section above so one file serves both the build
# tree and the install tree.
install(FILES
    "${_usdgen_schema_pluginfo}"
    "${_usdgen_schema_resources}/generatedSchema.usda"
    DESTINATION "${USDGEN_INSTALL_PLUGINDIR}/usdGenSchema/resources")

set(_usdgen_imaging_pluginfo
    "${_usdgen_imaging_resources}/plugInfo.json")
install(FILES "${_usdgen_imaging_pluginfo}"
    DESTINATION "${USDGEN_INSTALL_PLUGINDIR}/usdGenImaging/resources")

# usdGenShaders plugin resources (10 §3.4 install tree; C5 contract).
# shaderDefs.usda lives inside shaders/ and the .glslfx files are its siblings,
# so the relative @./...@ citations resolve in the installed layout too.
install(FILES
    "${_usdgen_shader_pluginfo}"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs/hosts/usdGenHydra/primitives/curves/shaders/resources/hairLook.usda"
    DESTINATION "${USDGEN_INSTALL_PLUGINDIR}/usdGenShaders/resources")
# N-6: a trailing-slash source copies the *contents* of shaders/, so the
# destination must be .../resources/shaders — the old rule flattened
# shaderDefs.usda and the .glslfx files straight into resources/, where
# plugInfo's "ShaderResources": "shaders" never looks (Sdr discovery found
# zero shaders in the install tree).
install(DIRECTORY libs/hosts/usdGenHydra/primitives/curves/shaders/resources/shaders/
    DESTINATION "${USDGEN_INSTALL_PLUGINDIR}/usdGenShaders/resources/shaders"
    PATTERN "usdGenHairModel.glslfx" EXCLUDE)
# The strand-model selector the build generated (USDGEN_HAIR_SHADING_MODEL).
install(FILES "${_usdgen_hair_model_file}"
    DESTINATION "${USDGEN_INSTALL_PLUGINDIR}/usdGenShaders/resources/shaders")

# usdGenTools usdview plugin: plugInfo beside the other plugin resources,
# the package where the install tree's python path already points.
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/resources/plugInfo.json"
    DESTINATION "${USDGEN_INSTALL_PLUGINDIR}/usdGenTools/resources")
install(DIRECTORY plugin/usdGenTools/python/usdGenTools
    DESTINATION "${USDGEN_INSTALL_PYTHONDIR}"
    FILES_MATCHING PATTERN "*.py")
install(DIRECTORY plugin/usdGenTools/resources/expressions
    DESTINATION "${USDGEN_INSTALL_PLUGINDIR}/usdGenTools/resources"
    FILES_MATCHING PATTERN "*.se")

# usdGenPomade scene index + usdGenPomadeTools shelf (plan/17 P0).
install(FILES "${_usdgen_pomade_pluginfo}"
    DESTINATION "${USDGEN_INSTALL_PLUGINDIR}/usdGenPomade/resources")
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/resources/plugInfo.json"
    DESTINATION "${USDGEN_INSTALL_PLUGINDIR}/usdGenPomadeTools/resources")
install(DIRECTORY plugin/usdGenPomadeTools/python/usdGenPomadeTools
    DESTINATION "${USDGEN_INSTALL_PYTHONDIR}"
    FILES_MATCHING PATTERN "*.py")
install(DIRECTORY plugin/usdGenPomadeTools/resources/icons
    DESTINATION "${USDGEN_INSTALL_PLUGINDIR}/usdGenPomadeTools/resources")

# Export targets so the package config's include(usdGenTargets.cmake) resolves.
install(EXPORT usdGenTargets
    FILE usdGenTargets.cmake
    NAMESPACE usdGen::
    DESTINATION "${USDGEN_INSTALL_CMAKEDIR}")

# CMake package config
# sol S-5/X-4: PATH_VARS makes @PACKAGE_USDGEN_INSTALL_PLUGINDIR@ and
# @PACKAGE_USDGEN_INSTALL_LIBDIR@ expand to real installed paths in the
# generated usdGenConfig.cmake (previously set_and_check(""), which made
# the package unusable for consumers).
configure_package_config_file(
    cmake/usdGenConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/usdGenConfig.cmake"
    INSTALL_DESTINATION "${USDGEN_INSTALL_CMAKEDIR}"
    PATH_VARS USDGEN_INSTALL_PLUGINDIR USDGEN_INSTALL_LIBDIR)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/usdGenConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/usdGenConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/usdGenConfigVersion.cmake"
    DESTINATION "${USDGEN_INSTALL_CMAKEDIR}")

# Headers (12.5, D16): api/ (with the extension SDK, api/ext/) and the
# usdGenMath headers it includes. Nothing else ships; testUsdGenInstallOnlyApi
# (cmake/gates/CheckInstallApiOnly.cmake) fails on any other installed header.
# The .cpp files under api/ are usdGen's own sources.
install(DIRECTORY libs/core/usdGen/usdGen/api/
    DESTINATION include/usdGen/api
    COMPONENT usdGenHeaders
    FILES_MATCHING PATTERN "*.h")
# usdGenMath is not an installed target; its compiled members live in usdGen
# (USDGEN_MATH_API). The api headers include math.h and rbf.h too.
install(FILES
    libs/base/usdGenMath/usdGenMath/half.h
    libs/base/usdGenMath/usdGenMath/vec.h
    libs/base/usdGenMath/usdGenMath/matrix.h
    libs/base/usdGenMath/usdGenMath/quat.h
    libs/base/usdGenMath/usdGenMath/range.h
    libs/base/usdGenMath/usdGenMath/math.h
    libs/base/usdGenMath/usdGenMath/rbf.h
    DESTINATION include/usdGenMath
    COMPONENT usdGenHeaders)

# NOTICE
install(FILES NOTICE DESTINATION "share/usdGen")
