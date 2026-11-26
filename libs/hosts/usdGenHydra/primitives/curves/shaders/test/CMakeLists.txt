# M1 shaders-lane verification harness (standalone; NOT part of the main
# build). Run from a shell with `source bin/_env.sh` (plan/10 §3.6):
#   cmake -S usdGenShaders/test -B usdGenShaders/test/build-m1-shaders -G Ninja \
#         -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$USD
#   cmake --build usdGenShaders/test/build-m1-shaders
#   ctest --test-dir usdGenShaders/test/build-m1-shaders -R Shader
#
# Gate L-2, both halves:
#   testUsdGenShaderC5            Sdr parse of the three glslfx + registry
#                                 discovery of the three defs (no GL needed).
#   testUsdGenShaderRender*       EGL headless Storm draws (render_check).
# Labels mirror the root CMakeLists' convention for shader/Storm tests
# (testUsdGenStormLook: LABELS "T2;gate:L-2", SKIP_RETURN_CODE "77" +
# NOT_RUN_RETURN_VALUES "77"); the Sdr-only check is tier T1 (no Hydra/GL).
# render_check exits 77 with a printed reason when this host offers no
# usable EGL_EXT_platform_device context — ctest then reports Skipped.
cmake_minimum_required(VERSION 3.20)
project(usdGenShadersTest CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenGL_GL_PREFERENCE GLVND)

find_package(pxr CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

# Interpreter: PY from bin/_env.sh wins, then python3 on PATH.
if(DEFINED ENV{PY} AND EXISTS "$ENV{PY}")
    set(USDGEN_PY "$ENV{PY}")
else()
    find_program(USDGEN_PY python3)
endif()
if(NOT USDGEN_PY)
    message(FATAL_ERROR "no python3 found; source bin/_env.sh first")
endif()

# Defaults mirroring bin/_env.sh, used only when the variable is not already
# in the environment (a shell that sourced _env.sh passes everything through).
get_filename_component(USDGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
if(DEFINED ENV{GENBUILD})
    set(USDGEN_BUILD "$ENV{GENBUILD}")
else()
    set(USDGEN_BUILD "${USDGEN_ROOT}/build")
endif()
if(DEFINED ENV{USD})
    set(USDGEN_USD "$ENV{USD}")
else()
    set(USDGEN_USD "${USDGEN_ROOT}/../OpenUSD_26_08")
endif()

set(_usdgen_shader_pluginpath
    "${USDGEN_BUILD}/usd/usdGenSchema/resources;${USDGEN_BUILD}/usd/usdGenImaging/resources;${USDGEN_BUILD}/usd/usdGenShaders/resources;${USDGEN_USD}/plugin/usd;${USDGEN_USD}/lib/usd")
file(GLOB _usdgen_shader_pysite "${USDGEN_USD}/lib/python*/site-packages")
if(_usdgen_shader_pysite)
    list(GET _usdgen_shader_pysite 0 _usdgen_shader_pysite)
endif()

if(UNIX)
    add_executable(render_check render_check.cpp)
    target_include_directories(render_check PRIVATE
        # vendored copy of the PW-egl probe header (docs/prework/probes/PW-egl/eglctx.h)
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(render_check PRIVATE
        usdAppUtils usdImagingGL usdImaging usdGeom usd garch hd hdSt hgiGL
        tf gf sdf hio OpenGL::GL ${CMAKE_DL_LIBS}
    )
endif()

# ENVIRONMENT entries are baked at configure time so the suite behaves the
# same from any runner shell: the configurator's value wins (a shell that
# sourced bin/_env.sh keeps its exact GENBUILD), else the default computed
# with bin/_env.sh's own scheme. PATH is left inherited (tool discovery).
set(_usdgen_def_PXR_PLUGINPATH_NAME
    "${USDGEN_BUILD}/usd/usdGenSchema/resources:${USDGEN_BUILD}/usd/usdGenImaging/resources:${USDGEN_BUILD}/usd/usdGenShaders/resources:${USDGEN_USD}/plugin/usd:${USDGEN_USD}/lib/usd")
set(_usdgen_def_PYTHONPATH "${USDGEN_BUILD}/python")
if(_usdgen_shader_pysite)
    string(APPEND _usdgen_def_PYTHONPATH ":${_usdgen_shader_pysite}")
endif()
set(_usdgen_def_LD_LIBRARY_PATH "${USDGEN_USD}/lib:${USDGEN_BUILD}")

function(usdgen_test_env out)
    set(_e "")
    foreach(_v PXR_PLUGINPATH_NAME PYTHONPATH LD_LIBRARY_PATH)
        if(DEFINED ENV{${_v}} AND NOT "$ENV{${_v}}" STREQUAL "")
            list(APPEND _e "${_v}=$ENV{${_v}}")
        else()
            list(APPEND _e "${_v}=${_usdgen_def_${_v}}")
        endif()
    endforeach()
    set(${out} "${_e}" PARENT_SCOPE)
endfunction()

usdgen_test_env(_usdgen_shader_env)

# --- L-2, Sdr half: glslfx parse + C5 equality + registry discovery -------
add_test(NAME testUsdGenShaderC5
    COMMAND "${USDGEN_PY}" "${CMAKE_CURRENT_SOURCE_DIR}/checkC5.py")
set_tests_properties(testUsdGenShaderC5 PROPERTIES
    LABELS "T1;gate:L-2"
    ENVIRONMENT "${_usdgen_shader_env}"
    SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

# --- L-2, render half: EGL headless Storm draws over make_scene.py scenes --
set(_scenes "${CMAKE_CURRENT_BINARY_DIR}/scenes")

if(UNIX)
    add_test(NAME testUsdGenShaderScenes
        COMMAND "${USDGEN_PY}" "${CMAKE_CURRENT_SOURCE_DIR}/make_scene.py")
    set_tests_properties(testUsdGenShaderScenes PROPERTIES
        LABELS "T2"
        ENVIRONMENT "${_usdgen_shader_env};USDGEN_SHADER_SCENES=${_scenes}"
        FIXTURES_SETUP ShaderScenes
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")

# scene: rendered at complexity 1.2 -> refineLevel 2 (default surface
# patch repr); A_l1 at 1.1 -> refineLevel 1 (S-9: variant A must not break).
foreach(_case
        "A;sceneA_asset.usda;1.2"        # variant A, glslfx sourceAsset
        "A_id;sceneA_id.usda;1.2"        # variant A via registered identifier
        "A_l1;sceneA_level1.usda;1.1"    # variant A at refineLevel 1 (S-9)
        "T;sceneT_asset.usda;1.2"        # translucent tag (OIT path)
        "B;sceneB_asset.usda;1.2"        # variant B, glslfx sourceAsset
        "B_id;sceneB_id.usda;1.2")       # variant B via registered identifier
    list(GET _case 0 _n)
    list(GET _case 1 _scene)
    list(GET _case 2 _cx)
    add_test(NAME testUsdGenShaderRender_${_n}
        COMMAND render_check "${_scenes}/${_scene}" "${_scenes}/render_${_n}.png"
                "/World/Cam" ${_cx} 800)
    set_tests_properties(testUsdGenShaderRender_${_n} PROPERTIES
        LABELS "T2;gate:L-2"
        ENVIRONMENT "${_usdgen_shader_env}"
        FIXTURES_REQUIRED ShaderScenes
        SKIP_RETURN_CODE "77" NOT_RUN_RETURN_VALUES "77")
endforeach()
endif() # UNIX render harness
