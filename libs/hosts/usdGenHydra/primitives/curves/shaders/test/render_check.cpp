// render_check.cpp — M1 shaders-lane verification harness (gate L-2 render half).
//
// Renders ONE frame headless via EGL_EXT_platform_device through Hydra Storm.
// Proves a glslfx material bound to basisCurves compiles + draws.
//
// usage: render_check <scene.usda> <out.png> <camPath> [complexity] [width]
//   complexity: usdview preset mapping -> refineLevel (usdImagingGL/engine.cpp
//   _GetRefineLevel): 1.0 -> 0 (wire), 1.1 -> 1, 1.2 -> 2, 1.3 -> 3.
//
// Diagnostics: OpenUSD's stable C++ API in this build exposes no custom
// diagnostic handler, so the caller captures stderr: severity-tagged lines
// ("Warning:", "Error:", ...) from Storm shader compilation and material
// resolution land there. The harness itself prints GL_VERSION, the record
// result, and any GL error codes observed after the frame.
//
// Exit codes: 0 rendered clean; 77 SKIP (no usable headless EGL context on
// this host — the L-2 render half is environment-gated, the Sdr half in
// checkC5.py is not); 2..8 real failures (see inline).
//
// Mirrors docs/prework/probes/PW-egl/render_hair.cpp (known-good PW-2 harness).

#include "eglctx.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"
#include "pxr/imaging/garch/glApi.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/camera.h"
#include "pxr/usdImaging/usdAppUtils/frameRecorder.h"
#include <GL/gl.h>

#include <cstdio>
#include <cstdlib>

PXR_NAMESPACE_USING_DIRECTIVE

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s scene.usda out.png camPath [complexity] [width]\n",
                argv[0]);
        return 2;
    }
    if (!eglctx::MakeHeadlessGLContext()) {
        // No display and no EGL_EXT_platform_device device that will give us
        // a current GL context: the L-2 RENDER half cannot be judged on this
        // host. Skip (77), don't fail — the Sdr half (checkC5.py) still runs.
        fprintf(stderr, "SKIP: no usable headless EGL context on this host "
                        "(EGL_EXT_platform_device). L-2 render half is "
                        "environment-gated; see the Sdr half in checkC5.py.\n");
        return 77;
    }
    GarchGLApiLoad();
    printf("GL_VERSION=%s\n", (const char *)glGetString(GL_VERSION));

    UsdStageRefPtr stage = UsdStage::Open(argv[1]);
    if (!stage) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 4;
    }

    UsdAppUtilsFrameRecorder rec(TfToken(), /*gpuEnabled*/ true);
    if (!rec.SetRendererPlugin(TfToken("HdStormRendererPlugin"))) {
        fprintf(stderr, "SetRendererPlugin(Storm) failed\n");
        return 5;
    }
    rec.SetImageWidth(argc > 5 ? (size_t)atoi(argv[5]) : 800);
    rec.SetColorCorrectionMode(TfToken("sRGB"));
    rec.SetCameraLightEnabled(true);
    float complexity = (argc > 4) ? (float)atof(argv[4]) : 1.2f;
    rec.SetComplexity(complexity);

    UsdGeomCamera cam(stage->GetPrimAtPath(SdfPath(argv[3])));
    if (!cam) {
        fprintf(stderr, "camera %s not found\n", argv[3]);
        return 7;
    }

    bool ok = rec.Record(stage, cam, UsdTimeCode::EarliestTime(), argv[2]);
    printf("Record complexity=%.2f -> %s : %s\n", complexity, argv[2],
           ok ? "OK" : "FAILED");

    int glErrors = 0;
    GLenum glErr = glGetError();
    while (glErr != GL_NO_ERROR) {
        printf("GL_ERROR after frame: 0x%04X\n", (unsigned)glErr);
        ++glErrors;
        glErr = glGetError();
    }
    printf("glErrors=%d\n", glErrors);

    int rc = ok ? 0 : 6;
    if (glErrors > 0) {
        rc = 8;
    }
    return rc;
}
