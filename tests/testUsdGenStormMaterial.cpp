// testUsdGenStormMaterial.cpp — gate L-1 (plan/09 §5.3): Storm
// render-context resolution on shipped code.
//
// PW-4 (docs/prework/PW-4-l1-material-override.md) measured that on a
// `Material` carrying both `outputs:surface` (chiang-hair mtlx network)
// and `outputs:glslfx:surface` (usdGen glslfx), Storm renders the glslfx
// terminal (materialRenderContexts = {glslfx, mtlx},
// hdSt/renderDelegate.cpp:695-707): the dual-terminal render matched the
// glslfx-only render (meanAbsDiff 0.27/255) and differed radically from the
// mtlx-only render (37.3/255). Therefore USDGEN_STORM_MATERIAL_OVERRIDE
// keeps its OFF default and tiles bind through the glslfx terminal.
//
// This test rebuilds the three PW-4 scenes (same seed-7 groom family, same
// Key+Rim lights and camera as libs/hosts/usdGenHydra/primitives/curves/shaders/test/make_scene.py, glslfx
// leg via the shipped usdGenHairPreviewPrimvar.glslfx sourceAsset) and
// asserts: all three render clean and lit, dual-terminal matches
// glslfx-only within AA-level tolerance, dual-terminal is far from
// mtlx-only (the mtlx terminal did NOT win), and re-renders are
// bit-stable.
//
// Exit: 0 pass, 77 SKIP (no usable EGL device context), 1 fail.
#include "stormTestUtils.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int gFails = 0;
void check(bool ok, const char* msg)
{
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", msg);
    if (!ok) {
        ++gFails;
    }
}

// Embedded PW-4-style scene generator: same seed-7 groom as make_scene.py
// (hairTangent always authored — the mtlx leg reads it via
// ND_geompropvalue_vector3), three terminal wirings.
const char* kGenPy = R"PY(
import os, sys, random
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf, Gf, Vt
OUT, SHADERS = sys.argv[1], sys.argv[2]
NCURVES, NCV, LEN = 4000, 8, 0.55

def geometry(stage):
    random.seed(7)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    sph = UsdGeom.Sphere.Define(stage, "/World/Scalp")
    sph.CreateRadiusAttr(0.98)
    sph.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.10, 0.07)])
    pts, widths, hairT, counts, hid, dcol, tang, st = [], [], [], [], [], [], [], []
    for c in range(NCURVES):
        while True:
            u, v, w = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.1, 1.0)
            n = Gf.Vec3f(u, w, v)
            if n.GetLength() > 1e-3:
                break
        n = n.GetNormalized()
        root = n * 1.0
        for i in range(NCV):
            t = i / (NCV - 1.0)
            p = root + n * LEN * t + Gf.Vec3f(0.0, -0.9 * t * t, 0.0) * LEN \
                + Gf.Vec3f(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) * 0.02 * t
            pts.append(Gf.Vec3f(p))
            widths.append(0.0032 * (1.0 - 0.85 * t))
            hairT.append(t)
            tan = (n * LEN + Gf.Vec3f(0.0, -1.8 * t * LEN, 0.0)).GetNormalized()
            tang.append(Gf.Vec3f(tan))
        counts.append(NCV)
        hid.append(random.random())
        sh = 0.75 + 0.5 * random.random()
        dcol.append(Gf.Vec3f(sh, sh, sh))
        st.append(Gf.Vec2f(0.5 + 0.5 * n[0], 0.5 + 0.5 * n[2]))
    curves = UsdGeom.BasisCurves.Define(stage, "/World/Hair")
    curves.CreateTypeAttr(UsdGeom.Tokens.cubic)
    curves.CreateBasisAttr(UsdGeom.Tokens.bspline)
    curves.CreateWrapAttr(UsdGeom.Tokens.pinned)
    curves.CreateCurveVertexCountsAttr(Vt.IntArray(counts))
    curves.CreatePointsAttr(Vt.Vec3fArray(pts))
    curves.CreateWidthsAttr(Vt.FloatArray(widths))
    curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    api = UsdGeom.PrimvarsAPI(curves.GetPrim())
    api.CreatePrimvar("hairT", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(Vt.FloatArray(hairT))
    api.CreatePrimvar("hairTangent", Sdf.ValueTypeNames.Vector3fArray, UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray(tang))
    api.CreatePrimvar("hairId", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.uniform).Set(Vt.FloatArray(hid))
    api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform).Set(Vt.Vec3fArray(dcol))
    api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.uniform).Set(Vt.Vec2fArray(st))
    api.CreatePrimvar("minScreenSpaceWidths", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.constant).Set(Vt.FloatArray([1.0]))
    return curves

def glslfx(stage):
    shd = UsdShade.Shader.Define(stage, "/World/Looks/M/GlslfxSurface")
    shd.GetPrim().CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token, True).Set("sourceAsset")
    shd.GetPrim().CreateAttribute("info:glslfx:sourceAsset", Sdf.ValueTypeNames.Asset, True).Set(
        Sdf.AssetPath(os.path.join(SHADERS, "usdGenHairPreviewPrimvar.glslfx")))
    shd.CreateInput("rootColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.030, 0.014, 0.006))
    shd.CreateInput("tipColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.32, 0.17, 0.07))
    shd.CreateInput("specular1Gain", Sdf.ValueTypeNames.Float).Set(0.45)
    shd.CreateInput("specular2Gain", Sdf.ValueTypeNames.Float).Set(0.30)
    shd.CreateInput("randomValue", Sdf.ValueTypeNames.Float).Set(0.25)
    shd.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    return shd

def mtlx(stage):
    absorb = UsdShade.Shader.Define(stage, "/World/Looks/M/Absorb")
    absorb.GetPrim().CreateAttribute("info:id", Sdf.ValueTypeNames.Token, True).Set("ND_deon_hair_absorption_from_melanin")
    absorb.CreateInput("melanin_concentration", Sdf.ValueTypeNames.Float).Set(0.03)
    absorb.CreateInput("melanin_redness", Sdf.ValueTypeNames.Float).Set(0.6)
    absorb.CreateOutput("absorption", Sdf.ValueTypeNames.Float3)
    rough = UsdShade.Shader.Define(stage, "/World/Looks/M/Rough")
    rough.GetPrim().CreateAttribute("info:id", Sdf.ValueTypeNames.Token, True).Set("ND_chiang_hair_roughness")
    rough.CreateInput("azimuthal", Sdf.ValueTypeNames.Float).Set(0.22)
    rough.CreateInput("longitudinal", Sdf.ValueTypeNames.Float).Set(0.1)
    rough.CreateOutput("roughness_R", Sdf.ValueTypeNames.Float2)
    rough.CreateOutput("roughness_TRT", Sdf.ValueTypeNames.Float2)
    rough.CreateOutput("roughness_TT", Sdf.ValueTypeNames.Float2)
    bsdf = UsdShade.Shader.Define(stage, "/World/Looks/M/Bsdf")
    bsdf.GetPrim().CreateAttribute("info:id", Sdf.ValueTypeNames.Token, True).Set("ND_chiang_hair_bsdf")
    bsdf.CreateInput("absorption_coefficient", Sdf.ValueTypeNames.Float3).ConnectToSource(absorb.ConnectableAPI(), "absorption")
    bsdf.CreateInput("curve_direction", Sdf.ValueTypeNames.Float3)
    bsdf.CreateInput("cuticle_angle", Sdf.ValueTypeNames.Float).Set(0.5)
    bsdf.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.55)
    bsdf.CreateInput("roughness_R", Sdf.ValueTypeNames.Float2).ConnectToSource(rough.ConnectableAPI(), "roughness_R")
    bsdf.CreateInput("roughness_TRT", Sdf.ValueTypeNames.Float2).ConnectToSource(rough.ConnectableAPI(), "roughness_TRT")
    bsdf.CreateInput("roughness_TT", Sdf.ValueTypeNames.Float2).ConnectToSource(rough.ConnectableAPI(), "roughness_TT")
    bsdf.CreateOutput("out", Sdf.ValueTypeNames.Token)
    tang = UsdShade.Shader.Define(stage, "/World/Looks/M/TangentPV")
    tang.GetPrim().CreateAttribute("info:id", Sdf.ValueTypeNames.Token, True).Set("ND_geompropvalue_vector3")
    tang.CreateInput("geomprop", Sdf.ValueTypeNames.String).Set("hairTangent")
    tang.CreateOutput("out", Sdf.ValueTypeNames.Float3)
    bsdf.GetInput("curve_direction").ConnectToSource(tang.ConnectableAPI(), "out")
    surf = UsdShade.Shader.Define(stage, "/World/Looks/M/MxSurface")
    surf.GetPrim().CreateAttribute("info:id", Sdf.ValueTypeNames.Token, True).Set("ND_surface")
    surf.CreateInput("bsdf", Sdf.ValueTypeNames.Token).ConnectToSource(bsdf.ConnectableAPI(), "out")
    surf.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    surf.CreateOutput("out", Sdf.ValueTypeNames.Token)
    return surf

def common(stage):
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(3.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35, 25, 0))
    rim = UsdLux.DistantLight.Define(stage, "/World/Lights/Rim")
    rim.CreateIntensityAttr(4.0)
    rim.CreateColorAttr(Gf.Vec3f(0.7, 0.85, 1.0))
    UsdGeom.Xformable(rim).AddRotateXYZOp().Set(Gf.Vec3f(-10, 200, 0))
    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    UsdGeom.Xformable(cam).AddTranslateOp().Set(Gf.Vec3d(0, 0.35, 5.2))
    cam.CreateFocalLengthAttr(50.0)
    cam.CreateHorizontalApertureAttr(24.0)
    cam.CreateVerticalApertureAttr(24.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))

def build(name, terminals):
    stage = Usd.Stage.CreateNew(os.path.join(OUT, name))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    curves = geometry(stage)
    mat = UsdShade.Material.Define(stage, "/World/Looks/M")
    for ctx, net in terminals:
        outp = mat.CreateSurfaceOutput() if ctx == "universal" else mat.CreateSurfaceOutput(ctx)
        src = glslfx(stage) if net == "glslfx" else mtlx(stage)
        outp.ConnectToSource(src.ConnectableAPI(), "surface" if net == "glslfx" else "out")
    UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(mat)
    common(stage)
    stage.GetRootLayer().Save()
    print("wrote", name, terminals)

build("mat_glslfx_only.usda", [("universal", "glslfx")])
build("mat_mtlx_only.usda", [("universal", "mtlx")])
build("mat_both.usda", [("universal", "mtlx"), ("glslfx", "glslfx")])
)PY";

}  // namespace

int main(int argc, char** argv)
{
    const std::string srcDir = (argc > 1) ? argv[1] : ".";
    const std::string scratch = "/tmp/usdGenStormMaterial_scratch";
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    if (!stormtest::initGl("testUsdGenStormMaterial")) {
        return 77;
    }

    // --- generate the three L-1 scenes ---
    {
        FILE* f = std::fopen((scratch + "/gen_l1.py").c_str(), "w");
        check(f != nullptr, "generator script written");
        if (f) {
            std::fputs(kGenPy, f);
            std::fclose(f);
        }
    }
    const std::string gen = "python3 '" + scratch + "/gen_l1.py' '" +
                            scratch + "' '" + srcDir +
                            "/libs/hosts/usdGenHydra/primitives/curves/shaders/resources/shaders' >/dev/null 2>&1";
    check(std::system(gen.c_str()) == 0, "L-1 scenes generated");
    const std::string glslfxOnly = scratch + "/mat_glslfx_only.usda";
    const std::string mtlxOnly = scratch + "/mat_mtlx_only.usda";
    const std::string both = scratch + "/mat_both.usda";
    check(std::filesystem::exists(glslfxOnly), "mat_glslfx_only.usda exists");
    check(std::filesystem::exists(mtlxOnly), "mat_mtlx_only.usda exists");
    check(std::filesystem::exists(both), "mat_both.usda exists");

    // --- all three render clean through Storm ---
    const int px = 256;
    check(stormtest::renderStage(glslfxOnly, scratch + "/G.png", px, 1.2f),
          "glslfx-only renders");
    check(stormtest::renderStage(mtlxOnly, scratch + "/M.png", px, 1.2f),
          "mtlx-only renders");
    check(stormtest::renderStage(both, scratch + "/B.png", px, 1.2f),
          "dual-terminal renders");
    check(stormtest::renderStage(both, scratch + "/B2.png", px, 1.2f),
          "dual-terminal re-renders");

    int w = 0, h = 0;
    VtArray<float> g, m, b, b2;
    check(stormtest::loadRgba(scratch + "/G.png", &w, &h, &g) &&
              stormtest::loadRgba(scratch + "/M.png", &w, &h, &m) &&
              stormtest::loadRgba(scratch + "/B.png", &w, &h, &b) &&
              stormtest::loadRgba(scratch + "/B2.png", &w, &h, &b2),
          "L-1 frames load");
    check(stormtest::coverage(g) > 0.05, "glslfx frame covers hair");
    check(stormtest::coverage(m) > 0.01, "mtlx frame covers hair");
    check(stormtest::coverage(b) > 0.05, "dual-terminal frame covers hair");

    // --- L-1 decision: dual-terminal matches glslfx-only (AA-level) ---
    double mean = 1.0, frac = 1.0;
    stormtest::diffStats(b, g, &mean, &frac);
    std::printf("both-vs-glslfx meanAbs=%.5f frac8=%.5f\n", mean, frac);
    check(mean <= 2.0 / 255.0, "dual-terminal matches glslfx (mean <= 2/255)");
    check(frac < 0.05, "dual-terminal matches glslfx (< 5% pixels > 8/255)");

    // --- ... and is far from mtlx-only: the mtlx terminal did NOT win ---
    stormtest::diffStats(b, m, &mean, &frac);
    std::printf("both-vs-mtlx meanAbs=%.5f frac8=%.5f\n", mean, frac);
    check(mean > 10.0 / 255.0, "dual-terminal differs from mtlx (mtlx lost)");
    stormtest::diffStats(g, m, &mean, &frac);
    std::printf("glslfx-vs-mtlx control meanAbs=%.5f frac8=%.5f\n", mean, frac);
    check(mean > 10.0 / 255.0, "control: glslfx and mtlx looks differ");

    // --- determinism ---
    stormtest::diffStats(b, b2, &mean, &frac);
    std::printf("dual-terminal determinism meanAbs=%.5f\n", mean);
    check(mean <= 1.0 / 255.0, "dual-terminal re-renders are bit-stable");

    std::printf("testUsdGenStormMaterial: %s (%d failures)\n",
                gFails ? "FAILED" : "PASSED", gFails);
    return gFails ? 1 : 0;
}
