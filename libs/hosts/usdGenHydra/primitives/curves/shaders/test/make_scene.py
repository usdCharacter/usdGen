#!/usr/bin/env python3
"""M1 shaders-lane: build scratch render scenes for the three shipped glslfx
files (usdGenShaders/resources/shaders/), modelled on the proven PW-4 probe
scenes (docs/prework/probes/PW-egl/pw4_*.usda, 4000 cubic bspline curves x
8 CVs on a scalp sphere).

Generated into usdGenShaders/test/scenes/:
  sceneA_asset.usda   variant A (usdGenHairPreview) via glslfx sourceAsset
  sceneA_id.usda      variant A via the registered Sdr identifier
                      ("UsdGenHairPreview" from shaderDefs.usda)
  sceneA_level1.usda  variant A, same as sceneA_asset (rendered with
                      complexity 1.1 -> refineLevel 1: S-9 "must not break")
  sceneT_asset.usda   usdGenHairPreviewTranslucent (materialTag translucent)
  sceneB_asset.usda   variant B (usdGenHairPreviewPrimvar) + hairTangent
  sceneB_id.usda      variant B via the registered Sdr identifier
"""
import os
import sys

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdShade, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
SHADERS = os.path.abspath(os.path.join(HERE, "..", "resources", "shaders"))
OUT = os.environ.get("USDGEN_SHADER_SCENES") or os.path.join(HERE, "scenes")
os.makedirs(OUT, exist_ok=True)

NCURVES, NCV, LEN = 4000, 8, 0.55
CAM_COLORS = {
    "rootColor": Gf.Vec3f(0.030, 0.014, 0.006),
    "tipColor": Gf.Vec3f(0.32, 0.17, 0.07),
    "specular1Gain": 0.45,
    "specular2Gain": 0.30,
    "randomValue": 0.25,
}


def build_geometry(stage, with_tangent):
    import random
    random.seed(7)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    sphere = UsdGeom.Sphere.Define(stage, "/World/Scalp")
    sphere.CreateRadiusAttr(0.98)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.10, 0.07)])

    pts, widths, hairT, counts, hairId, dispColor, tangents, st = \
        [], [], [], [], [], [], [], []
    for c in range(NCURVES):
        while True:
            u = random.uniform(-1, 1)
            v = random.uniform(-1, 1)
            w = random.uniform(0.1, 1.0)
            n = Gf.Vec3f(u, w, v)
            if n.GetLength() > 1e-3:
                break
        n = n.GetNormalized()
        root = n * 1.0
        for i in range(NCV):
            t = i / (NCV - 1.0)
            droop = Gf.Vec3f(0.0, -0.9 * t * t, 0.0)
            jitter = Gf.Vec3f(random.uniform(-1, 1),
                              random.uniform(-1, 1),
                              random.uniform(-1, 1)) * 0.02 * t
            p = root + (n * LEN * t) + droop * LEN + jitter
            pts.append(Gf.Vec3f(p))
            widths.append(0.0032 * (1.0 - 0.85 * t))
            hairT.append(t)
            tan = n * LEN + Gf.Vec3f(0.0, -1.8 * t * LEN, 0.0)
            tangents.append(Gf.Vec3f(tan.GetNormalized()))
        counts.append(NCV)
        hairId.append(random.random())
        shade = 0.75 + 0.5 * random.random()
        dispColor.append(Gf.Vec3f(shade, shade, shade))
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
    api.CreatePrimvar("hairT", Sdf.ValueTypeNames.FloatArray,
                      UsdGeom.Tokens.vertex).Set(Vt.FloatArray(hairT))
    if with_tangent:
        api.CreatePrimvar("hairTangent", Sdf.ValueTypeNames.Vector3fArray,
                          UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray(tangents))
    api.CreatePrimvar("hairId", Sdf.ValueTypeNames.FloatArray,
                      UsdGeom.Tokens.uniform).Set(Vt.FloatArray(hairId))
    api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                      UsdGeom.Tokens.uniform).Set(Vt.Vec3fArray(dispColor))
    api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                      UsdGeom.Tokens.uniform).Set(Vt.Vec2fArray(st))
    api.CreatePrimvar("minScreenSpaceWidths", Sdf.ValueTypeNames.FloatArray,
                      UsdGeom.Tokens.constant).Set(Vt.FloatArray([1.0]))
    return curves


def add_glslfx_shaders(stage, matName, mode, glslfx_file=None, identifier=None):
    shd = UsdShade.Shader.Define(stage, "/World/Looks/%s/GlslfxSurface" % matName)
    if mode == "asset":
        shd.GetPrim().CreateAttribute(
            "info:implementationSource", Sdf.ValueTypeNames.Token,
            True).Set("sourceAsset")
        shd.GetPrim().CreateAttribute(
            "info:glslfx:sourceAsset", Sdf.ValueTypeNames.Asset,
            True).Set(Sdf.AssetPath(os.path.join(SHADERS, glslfx_file)))
    else:
        shd.GetPrim().CreateAttribute(
            "info:id", Sdf.ValueTypeNames.Token,
            True).Set(identifier)
    shd.CreateInput("rootColor", Sdf.ValueTypeNames.Color3f).Set(CAM_COLORS["rootColor"])
    shd.CreateInput("tipColor", Sdf.ValueTypeNames.Color3f).Set(CAM_COLORS["tipColor"])
    shd.CreateInput("specular1Gain", Sdf.ValueTypeNames.Float).Set(CAM_COLORS["specular1Gain"])
    shd.CreateInput("specular2Gain", Sdf.ValueTypeNames.Float).Set(CAM_COLORS["specular2Gain"])
    shd.CreateInput("randomValue", Sdf.ValueTypeNames.Float).Set(CAM_COLORS["randomValue"])
    shd.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    return shd


def add_lighting_and_camera(stage):
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(3.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35, 25, 0))

    rim = UsdLux.DistantLight.Define(stage, "/World/Lights/Rim")
    rim.GetPrim().CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.7, 0.85, 1.0))
    rim.CreateIntensityAttr(4.0)
    UsdGeom.Xformable(rim).AddRotateXYZOp().Set(Gf.Vec3f(-10, 200, 0))

    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    UsdGeom.Xformable(cam).AddTranslateOp().Set(Gf.Vec3d(0, 0.35, 5.2))
    cam.CreateFocalLengthAttr(50.0)
    cam.CreateHorizontalApertureAttr(24.0)
    cam.CreateVerticalApertureAttr(24.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))


def make_scene(name, glslfx_file=None, identifier=None, mode="asset",
               with_tangent=False, terminal="surface"):
    stage = Usd.Stage.CreateNew(os.path.join(OUT, name))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    curves = build_geometry(stage, with_tangent)
    mat = UsdShade.Material.Define(stage, "/World/Looks/M")
    outp = mat.CreateSurfaceOutput(terminal)
    shd = add_glslfx_shaders(stage, "M", mode, glslfx_file, identifier)
    outp.ConnectToSource(shd.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(mat)
    add_lighting_and_camera(stage)
    stage.GetRootLayer().Save()
    print("wrote", os.path.join(OUT, name))


make_scene("sceneA_asset.usda", glslfx_file="usdGenHairPreview.glslfx")
make_scene("sceneA_id.usda", identifier="UsdGenHairPreview", mode="id")
# same content as sceneA_asset; rendered at complexity 1.1 (refineLevel 1)
make_scene("sceneA_level1.usda", glslfx_file="usdGenHairPreview.glslfx")
make_scene("sceneT_asset.usda",
           glslfx_file="usdGenHairPreviewTranslucent.glslfx")
make_scene("sceneB_asset.usda",
           glslfx_file="usdGenHairPreviewPrimvar.glslfx", with_tangent=True)
make_scene("sceneB_id.usda", identifier="UsdGenHairPreviewPrimvar",
           mode="id", with_tangent=True)
