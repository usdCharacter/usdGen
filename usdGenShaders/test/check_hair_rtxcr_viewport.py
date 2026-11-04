#!/usr/bin/env python3
# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
"""Exercise RTXCR coat transport in usdview's camera and dome light modes.

Uses testusdview rather than FrameRecorder: its default headlight and scene
lighting switches are the path that originally rendered the coat black.
The black background, disabled overlays and cleared-coverage reference keep
background pixels from passing a test for visible hair. Requires a GUI/GL
context, OpenUSD's testusdview, NumPy and Pillow. Exit 77 for missing imports.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def testUsdviewInputFunction(controller):
    import numpy as np
    from PIL import Image
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    from pxr.Usdviewq.common import RenderModes

    settings = controller._dataModel.viewSettings
    settings.showHUD = settings.showBBoxes = False
    settings.clearColorText = "Black"
    settings.colorCorrectionMode = "sRGB"
    settings.cameraPrim = controller._dataModel.stage.GetPrimAtPath("/World/Cam")
    settings.enableSceneMaterials = True
    settings.enableSceneLights = False
    settings.domeLightTexturesVisible = False
    settings.renderMode = RenderModes.SMOOTH_SHADED
    view = controller._stageView
    view.SetPhysicalWindowSize(480, 360)
    view.DrawAxis = lambda matrix: None
    shader = UsdShade.Shader(controller._dataModel.stage.GetPrimAtPath(
        "/World/Looks/Hair/Surface"))
    primvars = UsdGeom.PrimvarsAPI(controller._dataModel.stage.GetPrimAtPath(
        "/World/Hair"))
    density = primvars.GetPrimvar("furDensity")
    tau_p = primvars.GetPrimvar("furTauP")
    tau_n = primvars.GetPrimvar("furTauN")
    original_density, original_p, original_n = (
        density.Get(), tau_p.Get(), tau_n.Get())
    out = Path(os.environ["USDGEN_HAIR_VIEWPORT_OUT"])

    def set_float(name, value):
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)

    def color(value):
        for name in ("baseColor", "tipColor"):
            shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(value)

    def shot(name):
        view.updateView()
        controller._processEvents(iterations=10, waitForConvergence=True)
        path = out / (name + ".png")
        assert view.grabFrameBuffer().save(str(path)), str(path)
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)

    for identifier, tag in (("UsdGenHairStrands", "opaque"),
                            ("UsdGenHairStrandsTranslucent", "translucent")):
        shader.GetIdAttr().Set(identifier)
        for mode in ("camera", "dome"):
            settings.ambientLightOnly = mode == "camera"
            settings.domeLightEnabled = mode == "dome"
            name = tag + "_" + mode
            color(Gf.Vec3f(0.5))
            set_float("selfShadow", 1.0)
            set_float("backScatter", 1.0)
            set_float("hairCoverageScale", 1.0)
            lit = shot(name)
            set_float("hairCoverageScale", 0.0)
            cleared = shot(name + "_cleared")
            assert cleared.max() < 2, "background/overlays contaminate " + name
            set_float("hairCoverageScale", 1.0)
            assert np.mean(lit) > 2, "black coat: " + name

            # Regression: direct exp(-count) threw away scattered light and
            # ignored backScatter. Test it on actual usdview lighting, too.
            set_float("backScatter", 0.0)
            no_back = shot(name + "_no_backscatter")
            gain = float(np.mean(lit - no_back))
            assert gain > 0.2, "missing coat scattering: %s (%g)" % (name, gain)
            set_float("backScatter", 1.0)

            color(Gf.Vec3f(0.3, 0.04, 0.01))
            brown = shot(name + "_brown")
            chroma = float(np.mean(brown[..., 0] - brown[..., 2]))
            assert chroma > 0.2, "hair lost its material color: " + name
            assert np.mean(brown) < np.mean(lit), "absorption ignored: " + name
            print("OK: %s: lit %.2f, coat scatter %.2f, brown R-B %.2f" %
                  (name, np.mean(lit), gain, chroma), flush=True)

        # Imported ambient-only looks can have large local voxel density even
        # at the coat boundary. An open directional path must stay illuminated:
        # local density is not another opaque shell enclosing every strand.
        settings.ambientLightOnly = False
        settings.domeLightEnabled = True
        settings.domeLightTexturesVisible = False
        color(Gf.Vec3f(0.5, 0.25, 0.1))
        set_float("indirectMultiplier", 0.0)
        set_float("ambientFill", 1.0)
        set_float("selfShadow", 0.45)
        density.Set(Vt.FloatArray([8.0] * len(original_density)))
        tau_p.Set(Vt.Vec3fArray([Gf.Vec3f(0.0)] * len(original_p)))
        tau_n.Set(Vt.Vec3fArray([Gf.Vec3f(0.0)] * len(original_n)))
        set_float("aoIntensity", 0.0)
        open_sky = shot(tag + "_open_sky")
        set_float("aoIntensity", 8.0)
        open_ao = shot(tag + "_open_sky_ao")
        assert np.mean(open_sky) > 2, "ambient-only fixture is unlit: " + tag
        error = float(np.mean(np.abs(open_ao - open_sky)))
        assert error < 0.1, "local density blocks open sky: %s (%g)" % (tag, error)

        # Light arriving through the coat still responds to AO. Keep the
        # positive hemisphere open and place hair on the negative axes.
        tau_n.Set(Vt.Vec3fArray([Gf.Vec3f(6.0)] * len(original_n)))
        set_float("aoIntensity", 0.0)
        boundary = shot(tag + "_boundary")
        set_float("aoIntensity", 8.0)
        boundary_ao = shot(tag + "_boundary_ao")
        assert np.mean(boundary_ao) > 2, "black coat boundary: " + tag
        assert np.mean(boundary - boundary_ao) > 0.2, "AO ignored: " + tag
        print("OK: %s: open-sky AO error %.3f, boundary %.2f" %
              (tag, error, np.mean(boundary_ao)), flush=True)
        density.Set(original_density)
        tau_p.Set(original_p)
        tau_n.Set(original_n)
        set_float("indirectMultiplier", 1.0)
        set_float("ambientFill", 0.0)
        set_float("aoIntensity", 0.0)
    print("RTXCR USDVIEW CHECK: PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testusdview", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    try:
        import numpy  # noqa: F401
        from PIL import Image
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
        from pxr import Usdviewq  # noqa: F401
    except ImportError as exc:
        print("SKIP:", exc)
        return 77
    if not Path(args.testusdview).is_file():
        print("SKIP: testusdview is unavailable:", args.testusdview)
        return 77
    from check_hair_strands_render import build_stage

    out = Path(args.out or tempfile.mkdtemp(prefix="usdGenRtxcrViewport_")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    environment = out / "white_environment.png"
    Image.new("RGB", (8, 4), (255, 255, 255)).save(environment)
    stage, _, _, _ = build_stage(Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt,
                                 "UsdGenHairStrands", str(environment))
    scene = out / "coat.usda"
    stage.GetRootLayer().Export(str(scene))
    env = dict(os.environ, USDGEN_HAIR_VIEWPORT_OUT=str(out))
    # Native curves need a ribbon/round refinement from the first draw to
    # provide the tangent consumed by the strand shader.
    result = subprocess.run([sys.executable, args.testusdview, str(scene),
                             "--testScript", str(Path(__file__).resolve()),
                             "--renderer", "GL", "--complexity", "veryhigh",
                             "--defaultsettings"], env=env,
                            capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    print("wrote viewport captures to", out)
    if result.returncode or "RTXCR USDVIEW CHECK: PASS" not in result.stdout:
        return 1
    if "Failed to compile" in result.stderr or "Error:" in result.stderr:
        print("FAIL: viewport reported an error")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
