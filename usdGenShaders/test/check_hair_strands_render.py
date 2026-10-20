#!/usr/bin/env python3
"""Storm render check for the host renderer-parity strand-hair material.

Proves, on a real GL context, that `UsdGenHairStrands` /
`UsdGenHairStrandsTranslucent` COMPILE and that the pieces the port depends on
are actually live. A glslfx that fails to compile does not raise: Storm falls
back to a fixed shader and the hair silently goes flat, so every assertion here
is differential -- it changes one input (or the light) and requires the pixels
to move by more than noise. A fallback shader ignores all of them.

  1. the material compiles and its inputs reach it (baseColor black vs red)
  2. lights respond (the key light is rotated)
  3. the baked hair count drives dual scattering (selfShadow 0 vs 2, and more
     extinction must darken the coat)
  4. coverage compensation reaches alpha (hairCoverageScale 0 clears the hair)
  5. both materialTag variants draw
  6. (opaque variant) the matching controls: each one moves the pixels the
     way its definition says, and each "neutral" setting -- the split
     roughnesses authored equal to `roughness`, azimuthalRoughness <= 1/3,
     cuticleTilt authored as the built-in 0.035 rad -- reproduces the default
     frame. The opaque variant renders deterministically, so those
     equivalences are asserted to within one LSB.
  7. --model marschner (a USDGEN_HAIR_SHADING_MODEL=marschner build): the
     neutral settings are that model's defaults instead (azimuthalRoughness
     0.35, cuticleTilt 0.43 degrees, ior 1.55), and ior and aoIntensity (with
     the furDensity primvar) must reach the pixels.

The scene is built in memory: 2400 cubic bspline curves on a sphere with the
primvars a usdGen tile publishes (widths, hairT, hairId, st,
minScreenSpaceWidths, furTauP/furTauN). The key light sets
`inputs:normalize = 1`; without it HdStLight multiplies a distant light's
intensity by the solid angle of a 0.53-degree disc and the scene renders black
(pxr/imaging/hdSt/light.cpp:161-209).

Needs the plugin search path (bin/record_usd.ps1 / bin/_env.sh set it), a GL
context, NumPy and Pillow. Exits 77 (ctest SKIP) when any of those is missing.

Usage: python3 check_hair_strands_render.py [--usdrecord PATH] [--out DIR]
"""
import argparse
import math
import os
import random
import runpy
import sys
import tempfile


def _skip(reason):
    print("SKIP:", reason)
    sys.exit(77)


def build_stage(Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt, identifier,
                environment):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    ncurves, ncv, length = 2400, 8, 0.55
    rng = random.Random(7)
    pts, widths, hairT, counts, hairId, st = [], [], [], [], [], []
    tauP, tauN, density = [], [], []
    for _ in range(ncurves):
        while True:
            n = Gf.Vec3f(rng.uniform(-1, 1), rng.uniform(0.1, 1.0),
                         rng.uniform(-1, 1))
            if n.GetLength() > 1e-3:
                break
        n = n.GetNormalized()
        root = n * 1.0
        for i in range(ncv):
            t = i / (ncv - 1.0)
            p = root + n * (length * t) + Gf.Vec3f(0.0, -0.9 * t * t * length,
                                                   0.0)
            pts.append(Gf.Vec3f(p))
            widths.append(0.0075 * (1.0 - 0.7 * t))
            hairT.append(t)
            # A plausible optical depth: dense toward the root, thinning out
            # at the tip, and deeper downward (into the coat) than upward.
            depth = 6.0 * (1.0 - t)
            tauP.append(Gf.Vec3f(0.5 * depth, 0.15 * depth, 0.5 * depth))
            tauN.append(Gf.Vec3f(0.5 * depth, depth, 0.5 * depth))
            density.append(0.25 * depth)
        counts.append(ncv)
        hairId.append(rng.random())
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
    api.CreatePrimvar("hairId", Sdf.ValueTypeNames.FloatArray,
                      UsdGeom.Tokens.uniform).Set(Vt.FloatArray(hairId))
    api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                      UsdGeom.Tokens.uniform).Set(Vt.Vec2fArray(st))
    api.CreatePrimvar("minScreenSpaceWidths", Sdf.ValueTypeNames.FloatArray,
                      UsdGeom.Tokens.constant).Set(Vt.FloatArray([1.0]))
    api.CreatePrimvar("furTauP", Sdf.ValueTypeNames.Vector3fArray,
                      UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray(tauP))
    api.CreatePrimvar("furTauN", Sdf.ValueTypeNames.Vector3fArray,
                      UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray(tauN))
    # Local density for the Marschner model's hair occlusion (aoIntensity);
    # the Karis lobes ignore it.
    api.CreatePrimvar("furDensity", Sdf.ValueTypeNames.FloatArray,
                      UsdGeom.Tokens.vertex).Set(Vt.FloatArray(density))
    # NOTE: no displayColor -- the shader would take it as the root albedo and
    # the baseColor assertion below would then be testing nothing.

    material = UsdShade.Material.Define(stage, "/World/Looks/Hair")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Hair/Surface")
    shader.GetPrim().CreateAttribute("info:id", Sdf.ValueTypeNames.Token,
                                     True).Set(identifier)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(material)

    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(3.0)
    key.GetPrim().CreateAttribute("inputs:normalize", Sdf.ValueTypeNames.Bool,
                                  True).Set(True)
    rotate = UsdGeom.Xformable(key).AddRotateXYZOp()
    rotate.Set(Gf.Vec3f(-35, 25, 0))

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
    dome.CreateIntensityAttr(0.4)
    dome.GetPrim().CreateAttribute("inputs:texture:file",
                                   Sdf.ValueTypeNames.Asset,
                                   True).Set(Sdf.AssetPath(environment))

    camera = UsdGeom.Camera.Define(stage, "/World/Cam")
    camera.CreateFocalLengthAttr(35.0)
    camera.CreateHorizontalApertureAttr(24.0)
    camera.CreateVerticalApertureAttr(18.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
    UsdGeom.Xformable(camera).AddTranslateOp().Set(Gf.Vec3d(0, 0.35, 4.6))
    return stage, shader, rotate, camera


def matching_checks(np, Gf, Sdf, shader, render, check, tag, model="karis"):
    """Differential checks of the matching controls (docs/storm-fur.md,
    "Matching an imported look"). `render(name)` draws the current stage and
    returns RGBA; colour metrics read RGB, coverage reads alpha."""

    def author(values):
        for name, value in values.items():
            if isinstance(value, Gf.Vec3f):
                tn = Sdf.ValueTypeNames.Color3f
            elif isinstance(value, Gf.Vec4f):
                tn = Sdf.ValueTypeNames.Float4
            else:
                tn = Sdf.ValueTypeNames.Float
            shader.CreateInput(name, tn).Set(value)

    def shot(name, values):
        # Restore whatever was authored before (the caller's grey baseColor
        # and tipColor), and remove what was not.
        before = {}
        for key in values:
            attr = shader.GetPrim().GetAttribute("inputs:" + key)
            before[key] = attr.Get() if attr and attr.HasAuthoredValue() \
                else None
        author(values)
        img = render(tag + "_" + name)
        for key, value in before.items():
            if value is None:
                shader.GetPrim().RemoveProperty("inputs:" + key)
            else:
                shader.GetPrim().GetAttribute("inputs:" + key).Set(value)
        return img

    def diff(a, b):
        return float(np.mean(np.abs(a - b)))

    def lum(img):
        return float(np.mean(img[..., :3]))

    def rb(img):
        return float(np.mean(img[..., 0] - img[..., 2]))

    def same(a, b, what):
        d = np.abs(a - b)
        check(float(d.max()) <= 1.0,
              "%s: %s reproduces the default frame (max |d| %.0f/255, mean "
              "%.4f)" % (tag, what, float(d.max()), float(d.mean())))

    def moves(a, b, what, at_least=0.5):
        check(diff(a, b) > at_least,
              "%s: %s reaches the shader (mean |d| %.2f/255)"
              % (tag, what, diff(a, b)))

    red = Gf.Vec3f(0.8, 0.02, 0.02)
    blue = Gf.Vec3f(0.02, 0.02, 0.8)
    grey = Gf.Vec3f(0.5, 0.5, 0.5)
    ref = render(tag + "_matching_ref")

    # -- lobe controls ------------------------------------------------------
    same(ref, shot("split_equal", {"primaryRoughness": 0.35,
                                   "longitudinalRoughness": 0.35}),
         "primary/longitudinalRoughness authored equal to roughness")
    wide = shot("longitudinal_wide", {"longitudinalRoughness": 2.0})
    moves(ref, wide, "longitudinalRoughness")
    sharp = shot("primary_sharp", {"primaryRoughness": 0.1})
    moves(ref, sharp, "primaryRoughness")
    if model == "marschner":
        same(ref, shot("azimuthal_default", {"azimuthalRoughness": 0.35}),
             "azimuthalRoughness authored as the model's 0.35")
    else:
        same(ref, shot("azimuthal_third", {"azimuthalRoughness": 0.3}),
             "azimuthalRoughness 0.3 (s = 1)")
    moves(ref, shot("azimuthal_rough", {"azimuthalRoughness": 0.9}),
          "azimuthalRoughness", at_least=0.05)
    if model == "marschner":
        same(ref, shot("tilt_builtin", {"cuticleTilt": 0.42971835}),
             "cuticleTilt authored as the model's s = 0.03 rad")
        same(ref, shot("ior_default", {"ior": 1.55}),
             "ior authored as the default 1.55")
        moves(ref, shot("ior_high", {"ior": 2.2}), "ior", at_least=0.05)
    else:
        same(ref, shot("tilt_builtin", {"cuticleTilt": 2.0053523}),
             "cuticleTilt authored as the built-in 0.035 rad")
    moves(ref, shot("tilt_steep", {"cuticleTilt": 8.0}), "cuticleTilt",
          at_least=0.1)
    no_back = shot("backscatter_off", {"backScatter": 0.0})
    check(lum(ref) - lum(no_back) > 0.05,
          "%s: backScatter 0 darkens the coat (mean delta %.3f/255)"
          % (tag, lum(ref) - lum(no_back)))

    # -- preset gradients ---------------------------------------------------
    preset = {"usePresetColors": 1.0}
    for g in "A", "B":
        for i in "123":
            preset["presetColor%s%s" % (g, i)] = red if g == "A" else grey
    for i in "123":
        preset["presetColorC" + i] = blue

    def colour_split(img):
        # hair pixels only: anything brighter than the black background
        hair = img[..., 3] > 8.0
        n = max(int(hair.sum()), 1)
        reddish = (img[..., 0] > img[..., 2] + 8.0) & hair
        bluish = (img[..., 2] > img[..., 0] + 8.0) & hair
        return reddish.sum() / n, bluish.sum() / n

    only_a = shot("preset_a", dict(preset, presetWeights=Gf.Vec4f(1, 0, 0, 0)))
    only_c = shot("preset_c", dict(preset, presetWeights=Gf.Vec4f(0, 0, 1, 0)))
    half = shot("preset_ac", dict(preset, presetWeights=Gf.Vec4f(1, 0, 1, 0)))
    ra, ba = colour_split(only_a)
    rc, bc = colour_split(only_c)
    rh, bh = colour_split(half)
    check(ra > 0.5 and ba < 0.02 and bc > 0.5 and rc < 0.02,
          "%s: presetWeights picks gradient A or C for every strand "
          "(A: %.2f red / %.2f blue, C: %.2f red / %.2f blue)"
          % (tag, ra, ba, rc, bc))
    check(rh > 0.15 and bh > 0.15,
          "%s: presetWeights (1, 0, 1) splits the strands between A and C "
          "(%.2f red / %.2f blue)" % (tag, rh, bh))
    # Along the strand: red root stops, blue tip stop. Early points turn most
    # of each strand blue, late points keep it red.
    along = dict(preset, presetColorA3=blue, presetWeights=Gf.Vec4f(1, 0, 0, 0))
    early = shot("preset_points_early",
                 dict(along, presetPointsA=Gf.Vec4f(0.0, 0.05, 0.1, 0.2)))
    late = shot("preset_points_late",
                dict(along, presetPointsA=Gf.Vec4f(0.0, 0.05, 0.9, 0.98)))
    check(rb(late) - rb(early) > 3.0,
          "%s: presetPoints move the gradient stops along hairT (R-B late "
          "%.2f vs early %.2f)" % (tag, rb(late), rb(early)))
    grey_preset = {"usePresetColors": 1.0, "presetColorA1": grey,
                   "presetColorA2": grey, "presetColorA3": grey}
    linear = shot("preset_linear", grey_preset)
    srgb = shot("preset_srgb", dict(grey_preset, presetSrgb=1.0))
    check(lum(linear) - lum(srgb) > 1.0,
          "%s: presetSrgb linearises the stops (0.5 -> 0.214 darkens by %.2f/255)"
          % (tag, lum(linear) - lum(srgb)))

    # -- colour variation ---------------------------------------------------
    red_look = {"baseColor": red, "tipColor": red}
    red_img = shot("variation_red", red_look)
    desat = shot("variation_desaturated", dict(red_look, saturationOffset=-1.0))
    check(abs(rb(desat) - rb(ref)) < 0.25 * abs(rb(red_img) - rb(ref)),
          "%s: saturationOffset -1 turns red hair grey (R-B %.2f, grey %.2f, "
          "red %.2f)" % (tag, rb(desat), rb(ref), rb(red_img)))
    muted = {"baseColor": Gf.Vec3f(0.5, 0.4, 0.3),
             "tipColor": Gf.Vec3f(0.5, 0.4, 0.3)}
    muted_img = shot("variation_muted", muted)
    near = shot("variation_onesided_zero", dict(muted, jitterMode=1.0))
    check(diff(muted_img, near) < 0.1,
          "%s: jitterMode 1 with zero amounts is neutral (mean |d| %.3f/255)"
          % (tag, diff(muted_img, near)))
    for mode in (0.0, 1.0):
        sat = shot("variation_saturation_%d" % mode,
                   dict(muted, jitterMode=mode, randomSaturation=1.0))
        check(rb(sat) > rb(muted_img) + 1.0,
              "%s: randomSaturation (jitterMode %d) saturates the coat (R-B "
              "%.2f vs %.2f)" % (tag, mode, rb(sat), rb(muted_img)))

    # -- root/tip blend, ambient, root fade ---------------------------------
    ramp = {"baseColor": red, "tipColor": blue}
    linear_ramp = shot("blend_linear", ramp)
    fast_ramp = shot("blend_fast", dict(ramp, rootBlendRate=0.1))
    check(rb(fast_ramp) < rb(linear_ramp) - 2.0,
          "%s: rootBlendRate 0.1 reaches the tip colour early (R-B %.2f vs "
          "%.2f)" % (tag, rb(fast_ramp), rb(linear_ramp)))
    no_sky = shot("indirect_off", {"indirectMultiplier": 0.0})
    check(lum(ref) - lum(no_sky) > 0.5,
          "%s: indirectMultiplier 0 removes the dome (darker by %.2f/255)"
          % (tag, lum(ref) - lum(no_sky)))
    fill = shot("ambient_fill", {"ambientFill": 1.0})
    check(lum(fill) - lum(ref) > 0.5,
          "%s: ambientFill adds a flat fill (brighter by %.2f/255)"
          % (tag, lum(fill) - lum(ref)))
    if model == "marschner":
        occluded = shot("hair_ao", {"aoIntensity": 8.0})
        check(lum(ref) - lum(occluded) > 0.2,
              "%s: aoIntensity darkens the ambient by furDensity (darker by "
              "%.2f/255)" % (tag, lum(ref) - lum(occluded)))
    faded = shot("root_fade", {"rootFade": 1.0})
    # Fading the dense root layer can reveal brighter strands behind it, so
    # the colour may move either way; the coverage it spends cannot.
    lost = float(np.mean(ref[..., 3]) - np.mean(faded[..., 3]))
    check(lost > 0.5,
          "%s: rootFade clears coverage at the root (mean alpha down "
          "%.2f/255)" % (tag, lost))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usdrecord", default=os.environ.get(
        "USDGEN_USDRECORD",
        os.path.join(os.environ.get("USD", ""), "bin", "usdrecord")))
    parser.add_argument("--out", default="")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--model", choices=("karis", "marschner"),
                        default="karis",
                        help="the strand model the build compiled "
                             "(USDGEN_HAIR_SHADING_MODEL)")
    args = parser.parse_args()
    print("strand model:", args.model)

    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        _skip("NumPy/Pillow not importable (%s)" % e)
    try:
        from pxr import Gf, Sdf, Usd, UsdAppUtils, UsdGeom, UsdLux, UsdShade, Vt
    except ImportError as e:
        _skip("OpenUSD python bindings not importable (%s)" % e)
    if not os.path.exists(args.usdrecord):
        _skip("usdrecord not found at %r; pass --usdrecord or set USD"
              % args.usdrecord)

    out = args.out or tempfile.mkdtemp(prefix="usdGenHairStrands_")
    os.makedirs(out, exist_ok=True)
    environment = os.path.join(out, "white_environment.png")
    # A constant white environment: exercises the dome path without an HDRI
    # and without Storm's per-frame "dome light has no texture" warning.
    Image.new("RGB", (8, 4), (255, 255, 255)).save(environment)

    try:
        ctx = runpy.run_path(args.usdrecord)["_SetupOpenGLContext"]()
    except Exception as e:                                    # noqa: BLE001
        _skip("no usable GL context (%s)" % e)

    # One recorder per stage: UsdImagingGLEngine populates on its first Render
    # and keeps that stage, so a second stage handed to the same recorder is
    # never drawn and every image silently repeats the first stage's.
    def new_recorder():
        recorder = UsdAppUtils.FrameRecorder("HdStormRendererPlugin", True)
        recorder.SetImageWidth(args.width)
        recorder.SetComplexity(1.3)
        recorder.SetCameraLightEnabled(False)
        recorder.SetColorCorrectionMode("sRGB")
        return recorder

    failures = []

    def check(condition, message):
        print(("OK:   " if condition else "FAIL: ") + message)
        if not condition:
            failures.append(message)

    recorder = [None]

    def render(stage, camera, name, mode="RGB"):
        path = os.path.join(out, name + ".png")
        assert recorder[0].Record(stage, camera, Usd.TimeCode.Default(), path), \
            "FrameRecorder.Record failed for " + name
        return np.asarray(Image.open(path).convert(mode), dtype=np.float32)

    for identifier in ("UsdGenHairStrands", "UsdGenHairStrandsTranslucent"):
        tag = "opaque" if identifier == "UsdGenHairStrands" else "translucent"
        stage, shader, rotate, camera = build_stage(
            Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt, identifier,
            environment)
        recorder[0] = new_recorder()

        base = shader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f)
        tip = shader.CreateInput("tipColor", Sdf.ValueTypeNames.Color3f)
        self_shadow = shader.CreateInput("selfShadow",
                                         Sdf.ValueTypeNames.Float)
        coverage = shader.CreateInput("hairCoverageScale",
                                      Sdf.ValueTypeNames.Float)
        base.Set(Gf.Vec3f(0.5, 0.5, 0.5))
        tip.Set(Gf.Vec3f(0.5, 0.5, 0.5))
        self_shadow.Set(1.0)
        coverage.Set(1.0)

        grey = render(stage, camera, tag + "_grey")
        check(float(grey.max()) > 20.0,
              "%s: the scene renders something (max %.1f)"
              % (tag, grey.max()))

        base.Set(Gf.Vec3f(0.8, 0.02, 0.02))
        tip.Set(Gf.Vec3f(0.8, 0.02, 0.02))
        red = render(stage, camera, tag + "_red")
        # A failed glslfx compile leaves Storm's fallback shader, which knows
        # nothing about baseColor: the two images would be identical.
        channel_swing = float(np.mean(red[..., 0] - red[..., 2]) -
                              np.mean(grey[..., 0] - grey[..., 2]))
        check(channel_swing > 3.0,
              "%s: baseColor reaches the shader, so it compiled "
              "(R-B swing %.2f/255)" % (tag, channel_swing))
        base.Set(Gf.Vec3f(0.5, 0.5, 0.5))
        tip.Set(Gf.Vec3f(0.5, 0.5, 0.5))

        rotate.Set(Gf.Vec3f(-35, 155, 0))
        relit = render(stage, camera, tag + "_relit")
        check(float(np.mean(np.abs(relit - grey))) > 1.0,
              "%s: rotating the key light changes the coat (mean |d| %.2f/255)"
              % (tag, float(np.mean(np.abs(relit - grey)))))
        rotate.Set(Gf.Vec3f(-35, 25, 0))

        # Karis: selfShadow 0 is dual scattering at zero hair count. The
        # Marschner model switches dual scattering OFF at 0 (its wrap
        # fallback), so its extinction comparison starts from a small count
        # and the switch is checked on its own.
        if args.model == "marschner":
            self_shadow.Set(0.0)
            no_dual = render(stage, camera, tag + "_nodual")
            self_shadow.Set(0.25)
        else:
            self_shadow.Set(0.0)
        unshadowed = render(stage, camera, tag + "_unshadowed")
        if args.model == "marschner":
            check(float(np.mean(np.abs(no_dual - unshadowed))) > 0.5,
                  "%s: selfShadow 0 switches dual scattering off (mean |d| "
                  "%.2f/255)" % (tag, float(np.mean(np.abs(no_dual - unshadowed)))))
        self_shadow.Set(2.0)
        shadowed = render(stage, camera, tag + "_shadowed")
        delta = float(np.mean(unshadowed - shadowed))
        check(abs(delta) > 0.5,
              "%s: the baked hair count drives dual scattering "
              "(mean delta %.2f/255)" % (tag, delta))
        check(delta > 0.0,
              "%s: more extinction darkens the coat (mean delta %.2f/255)"
              % (tag, delta))
        self_shadow.Set(1.0)

        coverage.Set(0.0)
        cleared = render(stage, camera, tag + "_nocoverage")
        # With zero coverage every strand is fully transparent: the frame must
        # collapse to the (unlit, empty) background.
        check(float(np.mean(np.abs(cleared - grey))) >
              float(np.mean(np.abs(unshadowed - grey))),
              "%s: hairCoverageScale reaches alpha (clearing coverage moves "
              "the frame more than the whole extinction range)" % tag)
        coverage.Set(1.0)

        if tag == "opaque":
            matching_checks(np, Gf, Sdf, shader,
                            lambda name: render(stage, camera, name, "RGBA"),
                            check, tag, args.model)

        recorder[0] = None
        del stage

    print("wrote renders to", out)
    print("HAIR STRANDS RENDER CHECK:",
          "PASS" if not failures else "FAIL (%d)" % len(failures))
    del ctx
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
