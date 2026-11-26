#!/usr/bin/env python3
"""Real Storm GL render/relighting checks and synchronized steady-frame timing.

Run with the build's USD/plugin environment. Requires Qt (as usdrecord does),
NumPy and Pillow. Outputs PNGs and JSON; no golden image is silently updated.
"""
import argparse
import ctypes
import json
from pathlib import Path
import runpy
import statistics
import time

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdAppUtils, UsdGeom, UsdImagingGL, UsdShade


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("out")
    parser.add_argument("--usdrecord", required=True)
    parser.add_argument("--width", type=int, default=1000)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = runpy.run_path(args.usdrecord)["_SetupOpenGLContext"]()
    stage = Usd.Stage.Open(args.scene)
    # A constant white irradiance source avoids Storm's per-frame warning
    # for an untextured dome, which would contaminate timing with console I/O.
    environment = out / "white_environment.png"
    Image.new("RGB", (4, 2), (255, 255, 255)).save(environment)
    for prim in stage.Traverse():
        if prim.GetTypeName() == "DomeLight":
            prim.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set(str(environment.resolve()))
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/RenderCam"))
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/FurLook/Surface"))
    strength = shader.GetInput("selfOcclusion")
    recorder = UsdAppUtils.FrameRecorder("HdStormRendererPlugin", True)
    recorder.SetImageWidth(args.width)
    recorder.SetComplexity(1.2)
    recorder.SetCameraLightEnabled(False)
    recorder.SetColorCorrectionMode("sRGB")
    images = {}
    for name, value, identifier in (
        ("shadowed", 0.5, "UsdGenHairPreview"),
        ("unshadowed", 0.0, "UsdGenHairPreview"),
        ("translucent", 0.5, "UsdGenHairPreviewTranslucent"),
        ("primvar", 0.5, "UsdGenHairPreviewPrimvar"),
    ):
        strength.Set(value)
        shader.CreateIdAttr(identifier)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.65 if name == "translucent" else 1.0)
        path = out / (name + ".png")
        assert recorder.Record(stage, camera, Usd.TimeCode.Default(), str(path))
        images[name] = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        assert np.max(images[name]) > 30, "empty render: " + name
    delta = images["unshadowed"] - images["shadowed"]
    assert np.mean(np.abs(delta)) > 0.05, "density primvars did not affect Storm pixels"
    assert np.mean(delta) > 0, "enabling optical depth should reduce average illumination"
    shader.CreateIdAttr("UsdGenHairPreview")
    strength.Set(0.5)
    key = stage.GetPrimAtPath("/World/KeyLight").GetAttribute("xformOp:rotateXYZ")
    key.Set(Gf.Vec3f(-35, 120, -30))
    assert recorder.Record(stage, camera, Usd.TimeCode.Default(), str(out / "relit.png"))
    relit = np.asarray(Image.open(out / "relit.png").convert("RGB"), dtype=np.float32)
    assert np.mean(np.abs(relit-images["shadowed"])) > 0.5, "direct-light relighting did not change the coat"
    key.Set(Gf.Vec3f(35, 0, -30))
    engine = UsdImagingGL.Engine()
    engine.SetRendererPlugin("HdStormRendererPlugin")
    engine.SetRendererAov("color")
    height = round(args.width * 0.75)
    engine.SetRenderViewport(Gf.Vec4d(0, 0, args.width, height))
    engine.SetRenderBufferSize(Gf.Vec2i(args.width, height))
    cam = camera.GetCamera(Usd.TimeCode.Default())
    engine.SetCameraState(cam.frustum.ComputeViewMatrix(), cam.frustum.ComputeProjectionMatrix())
    params = UsdImagingGL.RenderParams()
    params.frame = Usd.TimeCode.Default()
    params.complexity = 1.2
    params.enableSceneLights = True
    params.enableSceneMaterials = True
    params.colorCorrectionMode = "sRGB"
    # glFinish includes GPU completion, preventing CPU submission-only timings.
    if hasattr(ctypes, "windll"):
        finish = ctypes.windll.opengl32.glFinish
    else:
        finish = ctypes.CDLL("libGL.so.1").glFinish
    timings = {}
    for name, value in (("shadowed", 0.5), ("unshadowed", 0.0)):
        strength.Set(value)
        for _ in range(10):
            engine.Render(stage.GetPseudoRoot(), params)
        finish()
        samples = []
        for _ in range(60):
            start = time.perf_counter()
            engine.Render(stage.GetPseudoRoot(), params)
            finish()
            samples.append((time.perf_counter() - start) * 1000)
        timings[name] = {"median_ms": statistics.median(samples),
                         "p95_ms": float(np.percentile(samples, 95))}
    result = {"width": args.width, "height": height,
              "method": "10 warmup + 60 frames per mode; CPU wall time with glFinish; no readback/export in timed region",
              "instances": len(UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/Fur")).GetProtoIndicesAttr().Get()),
              "mean_shadow_delta_8bit": float(np.mean(delta)),
              "timings": timings}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    del engine, recorder
    del ctx


if __name__ == "__main__":
    main()
