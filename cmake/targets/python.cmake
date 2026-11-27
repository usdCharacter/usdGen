# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
#
# Python packages and usdview plugins. Included from the root CMakeLists.txt.

# ---------------------------------------------------------------------------
# usdGenTools: usdview plugin (SeExpr expression editor). Pure Python. The
# resources dir carries the Type "python" plugInfo that usdview discovers on
# PXR_PLUGINPATH_NAME; the package itself is staged into build/python, which
# the launchers put on PYTHONPATH (plan/08 §5).
# ---------------------------------------------------------------------------
set(_usdgen_tools_resources "${CMAKE_CURRENT_BINARY_DIR}/usd/usdGenTools/resources")
file(GLOB_RECURSE _usdgen_tools_python CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/python/*.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/resources/plugInfo.json"
    ${_usdgen_tools_python})
file(COPY
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/resources/plugInfo.json"
    DESTINATION "${_usdgen_tools_resources}")
file(COPY
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/python/usdGenTools"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python"
    FILES_MATCHING PATTERN "*.py")
# The shipped expression presets the editor's library browser reads. They sit
# beside plugInfo.json so exprLibrary finds them by walking up from the staged
# package, in the build tree and in the install tree alike.
file(GLOB _usdgen_tools_expressions
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/resources/expressions/*.se")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${_usdgen_tools_expressions})
file(COPY
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/resources/expressions"
    DESTINATION "${_usdgen_tools_resources}"
    FILES_MATCHING PATTERN "*.se")
# The brush palette's icons (256x256 RGBA PNGs), resolved the same way as the
# expression presets: beside plugInfo.json under resources/icons.
file(GLOB _usdgen_tools_icons CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenTools/resources/icons/*.png")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${_usdgen_tools_icons})
if(_usdgen_tools_icons)
    file(COPY ${_usdgen_tools_icons}
        DESTINATION "${_usdgen_tools_resources}/icons")
endif()

# ---------------------------------------------------------------------------
# usdGenPomade scene-index plugin resources + usdGenPomadeTools usdview plugin
# (plan/17 P0). The scene index registers through build-tree plugInfo like
# the groom index; the Python shelf stages into build/python like usdGenTools.
# ---------------------------------------------------------------------------
set(_usdgen_pomade_resources "${CMAKE_CURRENT_BINARY_DIR}/usd/usdGenPomade/resources")
set(_usdgen_pomade_pluginfo "${_usdgen_pomade_resources}/plugInfo.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomade/resources/plugInfo.json.in")
file(READ
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomade/resources/plugInfo.json.in"
    _usdgen_pomade_pluginfo_content)
file(GENERATE OUTPUT "${_usdgen_pomade_pluginfo}"
    CONTENT "${_usdgen_pomade_pluginfo_content}")

set(_usdgen_pomade_tools_resources "${CMAKE_CURRENT_BINARY_DIR}/usd/usdGenPomadeTools/resources")
# CONFIGURE_DEPENDS: the package is staged by file(COPY) at CONFIGURE time,
# and the CMAKE_CONFIGURE_DEPENDS list below is this glob's result from the
# PREVIOUS configure -- so editing an existing module re-stages, but adding
# a module never does, and the tests keep importing a package without it.
# That cost V4 and V5 a false T3 failure each; the flag re-evaluates the
# glob at build time and re-configures when the file set changes.
file(GLOB_RECURSE _usdgen_pomade_tools_python CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/python/*.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/resources/plugInfo.json"
    ${_usdgen_pomade_tools_python})
file(COPY
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/resources/plugInfo.json"
    DESTINATION "${_usdgen_pomade_tools_resources}")
file(COPY
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/python/usdGenPomadeTools"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python"
    FILES_MATCHING PATTERN "*.py")
# Pomade's shelf/toolbar icons (IC-01), staged the same way as the brush
# palette's under usdGenTools (L604-612 above): white PNGs beside
# plugInfo.json under resources/icons, resolved by pomadeIcons.iconDir().
file(GLOB _usdgen_pomade_tools_icons CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin/usdGenPomadeTools/resources/icons/*.png")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${_usdgen_pomade_tools_icons})
if(_usdgen_pomade_tools_icons)
    file(COPY ${_usdgen_pomade_tools_icons}
        DESTINATION "${_usdgen_pomade_tools_resources}/icons")
endif()

# Localized Noodles usdview editor. Its native bindings need OpenUSD's Python
# and graphics libraries; the editor remains optional for headless builds.
if(NOT CMAKE_CROSSCOMPILING AND TARGET python AND TARGET garch AND TARGET glf)
    set(_usdgen_noodles_default ON)
else()
    set(_usdgen_noodles_default OFF)
endif()
option(USDGEN_BUILD_USDNOODLES "Build the Noodles usdview editor"
    ${_usdgen_noodles_default})
if(USDGEN_BUILD_USDNOODLES)
    if(NOT TARGET python OR NOT TARGET garch OR NOT TARGET glf)
        message(FATAL_ERROR
            "USDGEN_BUILD_USDNOODLES requires OpenUSD Python and graphics libraries")
    endif()
    add_subdirectory(plugin/usdNoodles)
endif()
