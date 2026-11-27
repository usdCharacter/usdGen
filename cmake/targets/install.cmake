# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Install rules and the package config. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
# N-7 (plan §1.2/§1.6): only the shipped targets are installed and
# exported. usdGenMath, usdGen_seexpr and nanoflann are private dependency
# archives (linked hidden, --exclude-libs,ALL, symbols verified out of the
# shipped objects) — installing them would put libusdGenMath.a /
# libusdGen_seexpr.a into lib/ and drag their unprefixed INTERFACE include
# roots into the usdGenTargets export set.
install(TARGETS usdGen usdGenSurface usdGenExpr usdGenCurves usdGenWithPrimitives
        usdGenUsd usdGenHydraCurves usdGenImaging usdGenSchema usdGenShaders usdGenPomade
    EXPORT usdGenTargets
    RUNTIME DESTINATION "${USDGEN_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${USDGEN_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${USDGEN_INSTALL_LIBDIR}")
# usdGen loads the backends from its own directory (USDGEN_DEFAULT_EXTENSIONS).
install(TARGETS usdGenCuda
    EXPORT usdGenTargets
    RUNTIME DESTINATION "${USDGEN_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${USDGEN_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${USDGEN_INSTALL_LIBDIR}")
if(TARGET usdGenVulkan)
    install(TARGETS usdGenVulkan usdGenVulkanNative usdGenVulkanFactory
        EXPORT usdGenTargets
        RUNTIME DESTINATION "${USDGEN_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${USDGEN_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${USDGEN_INSTALL_LIBDIR}")
endif()


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

# Install public headers for the exported targets' $<INSTALL_INTERFACE:include>
# dirs. The build-tree include roots are:
#   usdGen        libs/core/usdGen     -> #include "usdGen/{api,registries,compile,exec}/..."
#   usdGenCurves, usdGenSurface, usdGenExpr (libs/core/primitives/<name>),
#   usdGenCuda, usdGenVulkan (libs/core/backends/<name>) -> #include "<name>/..."
#   the include shim (<build>/shim) -> the pre-move "usdGen/..." spellings
#   usdGenSchema  libs/hosts/usdGenUsd/schema -> #include "usdGenSchema/..."
#   usdGenImaging libs/hosts/usdGenHydra -> #include "usdGenImaging/..."
#   usdGenHydraCurves libs/hosts/usdGenHydra/primitives/curves -> #include "usdGenHydraCurves/..."
#   usdGenPomade   libs/usdGenPomade     -> #include "usdGenPomade/..."
# so the install mirrors the include spellings directly under
# <prefix>/include. Trailing source slashes + FILES_MATCHING ship headers
# only (sol S-6: the previous install(DIRECTORY) rules appended directory
# basenames a second time — include/usdGen/usdGen/*.h — and installed .cpp
# files; duplicate rules for schema/imaging are removed).
# N-7 (plan §1.2): usdGenMath is not an installed target (its archive stays
# private). Step 11.4: the api headers name the engine's own math types, so
# the math headers they include are installed (include/usdGenMath).
install(FILES
    libs/base/usdGenMath/usdGenMath/half.h
    libs/base/usdGenMath/usdGenMath/vec.h
    libs/base/usdGenMath/usdGenMath/matrix.h
    libs/base/usdGenMath/usdGenMath/quat.h
    libs/base/usdGenMath/usdGenMath/range.h
    libs/base/usdGenMath/usdGenMath/math.h
    libs/base/usdGenMath/usdGenMath/rbf.h
    DESTINATION include/usdGenMath)
# The Session routing bridge deliberately carries cooker/private COW
# mechanics and is not part of the installed adapter surface.
install(DIRECTORY libs/core/usdGen/usdGen/
    DESTINATION include/usdGen
    FILES_MATCHING PATTERN "*.h"
    PATTERN "sessionBackendExecutor.h" EXCLUDE)
foreach(_usdgen_root primitives/usdGenCurves primitives/usdGenSurface
        primitives/usdGenExpr backends/usdGenCuda backends/usdGenVulkan)
    get_filename_component(_usdgen_name "${_usdgen_root}" NAME)
    install(DIRECTORY libs/core/${_usdgen_root}/${_usdgen_name}/
        DESTINATION include/${_usdgen_name}
        FILES_MATCHING PATTERN "*.h")
endforeach()
install(DIRECTORY "${USDGEN_SHIM_DIR}/usdGen/"
    DESTINATION include/usdGen
    FILES_MATCHING PATTERN "*.h"
    PATTERN "sessionBackendExecutor.h" EXCLUDE)
install(DIRECTORY libs/hosts/usdGenUsd/usdGenUsd/
    DESTINATION include/usdGenUsd
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY libs/hosts/usdGenUsd/schema/usdGenSchema/
    DESTINATION include/usdGenSchema
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY libs/hosts/usdGenHydra/usdGenImaging/
    DESTINATION include/usdGenImaging
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY libs/hosts/usdGenHydra/primitives/curves/usdGenHydraCurves/
    DESTINATION include/usdGenHydraCurves
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY libs/usdGenPomade/usdGenPomade/
    DESTINATION include/usdGenPomade
    FILES_MATCHING PATTERN "*.h")

# NOTICE
install(FILES NOTICE DESTINATION "share/usdGen")
