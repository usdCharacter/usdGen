#!/usr/bin/env python3
"""Author repeatable live usdGen fur patch renders for the operator manual.

Run ``python tools/render_operator_examples.py --sample`` for a fast visual
check.  Generated USD stages stay under ignored renders/docs/operators/;
published PNGs are copied to docs/site/media after visual review.
"""

# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "renders" / "docs" / "operators"
SCENES = ROOT / "examples" / "docs" / "operators"
MEDIA = ROOT / "docs" / "site" / "media"
MEDIA_JSON = ROOT / "docs" / "reference" / "media.json"
DIAGRAM_SOURCES = {
    "UsdGenPart": "libs/core/primitives/usdGenCurves/usdGenCurves/ops/part.cpp",
    "UsdGenFreeze": "libs/core/primitives/usdGenCurves/usdGenCurves/ops/freeze.cpp",
    "UsdGenWidthBlend": "libs/core/primitives/usdGenCurves/usdGenCurves/ops/widthBlend.cpp",
    "UsdGenInstance": "libs/usdGenSchema/schema.usda",
}


def camera_matrix(eye=(1.75, 1.18, 1.88), target=(0.50, 0.065, 0.50)):
    def unit(v):
        d = math.sqrt(sum(x*x for x in v))
        return tuple(x/d for x in v)
    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    back = unit(tuple(eye[i]-target[i] for i in range(3)))
    right = unit(cross((0, 1, 0), back))
    up = cross(back, right)
    rows = [(*right, 0), (*up, 0), (*back, 0), (*eye, 1)]
    return "( " + ", ".join("(" + ", ".join(f"{v:.8f}" for v in row) + ")" for row in rows) + " )"


BASE_OPS = '''
            def UsdGenWidth "width" {
                float usdGen:width = 0.0026
                float2[] usdGen:width:knots = [(0, 1), (0.42, 0.72), (1, 0.045)]
                token usdGen:width:interpolation = "catmullRom"
                bool usdGen:replace = true
            }
            def UsdGenClump "clump" {
                int usdGen:seed = 23
                float usdGen:clump:amount = 0.40
                float2[] usdGen:clump:profile:knots = [(0, 0), (0.4, 0.46), (1, 0.92)]
                token usdGen:clump:method = "linearBlend"
                float usdGen:clump:size = 0.085
                int usdGen:clump:levels = 2
                float usdGen:clump:density = 50
                float usdGen:clump:sizeReduction = 0.42
                float usdGen:clump:tightnessReduction = 0.55
                float usdGen:clump:stray:rate = 0.13
                float usdGen:clump:stray:amount = 0.65
                float usdGen:preserveLength = 0.8
            }
            def UsdGenNoise "noise" {
                int usdGen:seed = 7
                float usdGen:noise:magnitude = 0.0026
                float2[] usdGen:noise:magnitude:knots = [(0, 0), (0.5, 0.4), (1, 1)]
                float usdGen:noise:frequency = 11
                float usdGen:noise:correlation = 0.65
                int usdGen:noise:octaves = 2
                float usdGen:preserveLength = 1
            }
            def UsdGenGrow "grow" {
                int usdGen:seed = 19
                int usdGen:segments = 18
                float usdGen:length = 0.23
                float2 usdGen:lengthRandom = (0.78, 1.26)
                token usdGen:direction = "vector"
                vector3f usdGen:directionVector = (0.43, 0.90, 0.19)
            }
            def UsdGenScatter "scatter" {
                int usdGen:seed = 41
                float usdGen:density = 18000
            }
'''


def scene(ops=BASE_OPS, extra_prims="", description_overrides="",
          sky_asset="../../../examples/maps/StinsonBeach.hdr") -> str:
    return f'''#usda 1.0
# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Y"
    doc = "Live usdGen macro fur patch for the operator manual."
)
def Xform "World" {{
    def Camera "Cam" {{
        matrix4d xformOp:transform = {camera_matrix()}
        uniform token[] xformOpOrder = ["xformOp:transform"]
        float2 clippingRange = (0.02, 50)
        float focalLength = 44
        float horizontalAperture = 36
        float verticalAperture = 24
    }}
    def DistantLight "Key" {{
        color3f inputs:color = (1, 0.79, 0.60)
        float inputs:intensity = 1.4
        bool inputs:normalize = true
        float3 xformOp:rotateXYZ = (-20, -35, -15)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}
    def DistantLight "Rim" {{
        color3f inputs:color = (0.59, 0.79, 1)
        float inputs:intensity = 0.45
        bool inputs:normalize = true
        float3 xformOp:rotateXYZ = (55, 148, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}
    def DistantLight "Fill" {{
        color3f inputs:color = (0.84, 0.90, 1)
        float inputs:intensity = 0.85
        bool inputs:normalize = true
        float3 xformOp:rotateXYZ = (-50, 15, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}
    def DomeLight "Sky" {{
        asset inputs:texture:file = @{sky_asset}@
        color3f inputs:color = (0.82, 0.88, 1)
        float inputs:intensity = 0.5
    }}
    def Scope "Looks" {{
        def Material "Base" {{
            token outputs:surface.connect = </World/Looks/Base/Surface.outputs:surface>
            def Shader "Surface" {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.035, 0.043, 0.051)
                float inputs:roughness = 0.92
                token outputs:surface
            }}
        }}
        def Material "Fur" {{
            token outputs:surface.connect = </World/Looks/Fur/Surface.outputs:surface>
            def Shader "Surface" {{
                uniform token info:id = "UsdGenHairStrands"
                color3f inputs:baseColor = (0.63, 0.40, 0.20)
                color3f inputs:tipColor = (0.85, 0.65, 0.36)
                float inputs:roughness = 0.55
                float inputs:specular = 0.35
                float inputs:scatter = 0.36
                float inputs:selfShadow = 0.48
                float inputs:randomRoughness = 0.12
                token outputs:surface
            }}
        }}
    }}
    def Mesh "Patch" (prepend apiSchemas = ["UsdGenRestAPI", "MaterialBindingAPI"]) {{
        rel material:binding = </World/Looks/Base>
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 3, 2, 1]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]
        point3f[] primvars:rest = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)] (interpolation = "vertex")
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (interpolation = "vertex")
    }}
    def Mesh "Backdrop" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
        rel material:binding = </World/Looks/Base>
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 3, 2, 1]
        point3f[] points = [(-3, -0.055, -3), (4, -0.055, -3), (4, -0.055, 4), (-3, -0.055, 4)]
    }}
{extra_prims}
    def Xform "Groom" {{
        def UsdGenDescription "Fur" (prepend apiSchemas = ["UsdGenLookAPI", "MaterialBindingAPI"]) {{
            rel material:binding = </World/Looks/Fur>
            rel usdGen:surface = </World/Patch>
            token usdGen:curve:basis = "bspline"
            float usdGen:width:default = 0.0012
            int usdGen:look:jitterSeed = 113
            color3f usdGen:look:rootColor = (0.63, 0.40, 0.20)
            color3f usdGen:look:tipColor = (0.85, 0.65, 0.36)
            float usdGen:look:valueJitter = 0.055
{description_overrides}
            def Scope "Ops" {{
{ops}
            }}
        }}
    }}
}}
'''


def render(path: Path, out: Path, width: int, frame: int | None = None,
           camera: str = "/World/Cam") -> None:
    args = [sys.executable, "-S", str(ROOT / "tools" / "_usdrecord_clean.py"),
            "--renderer", "GL", "--imageWidth", str(width),
            "--disableCameraLight", "--camera", camera,
            "--complexity", "veryhigh"]
    target = out
    if frame is not None:
        args += ["--frames", str(frame)]
        target = out.with_name(f"{out.stem}.###.png")
    args += [str(path), str(target)]
    subprocess.run(args, check=True, cwd=ROOT)
    if frame is not None:
        matches = list(out.parent.glob(f"{out.stem}.*.png"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one frame for {out}: {matches}")
        matches[0].replace(out)


def slug(op_id: str) -> str:
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", "-", op_id.removeprefix("UsdGen")).lower()


def generate_operators(selected: set[str] | None, width: int,
                       render_stills: bool) -> None:
    from operator_recipes import RECIPES
    reference = json.loads((ROOT / "docs" / "reference" / "operators.json").read_text(encoding="utf-8"))
    SCENES.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    updates: dict[str, dict] = {}
    for entry in reference["operators"]:
        op_id = entry["id"]
        if selected and op_id not in selected:
            continue
        if op_id in {"UsdGenCollide", "UsdGenWind"}:
            diagnostic = "collide" if op_id == "UsdGenCollide" else "wind"
            print(f"DEFER {op_id}: run python tools/render_{diagnostic}_diagnostic.py "
                  "to regenerate its comparison scene and media", file=sys.stderr)
            continue
        recipe = RECIPES.get(op_id)
        if recipe is None:
            raise KeyError(f"Missing recipe for {op_id}")
        kind = recipe.get("kind", "live")
        if kind in {"reserved", "internal", "storyboard", "data_plane"}:
            diagram_path = MEDIA / f"{slug(op_id)}.svg"
            if not diagram_path.exists():
                raise FileNotFoundError(f"Run tools/render_operator_diagrams.py first: {diagram_path}")
            updates[op_id] = {
                "status": "diagram",
                "image": diagram_path.relative_to(ROOT).as_posix(),
                "caption": recipe["caption"],
                "renderer": "Source diagram",
                "source": DIAGRAM_SOURCES[op_id],
                "generator": "tools/render_operator_diagrams.py",
                "provenance": recipe.get("provenance", recipe.get("reason", "Schematic of the source contract; this is not a rendered operator result.")),
            }
            continue
        if kind not in {"live", "comparison", "existing_scene"}:
            # A special recipe must supply an independently verifiable stage or
            # exporter. Never substitute the base coat for an unexecuted effect.
            print(f"DEFER {op_id}: {recipe.get('reason', 'special runtime setup required')}", file=sys.stderr)
            continue
        if kind == "existing_scene":
            scene_path = ROOT / recipe["scene_path"]
        else:
            scene_path = SCENES / f"{slug(op_id)}.usda"
            scene_text = scene(
                ops=recipe["ops"],
                extra_prims=recipe.get("extra_prims", ""),
                description_overrides=recipe.get("description_overrides", ""),
                sky_asset="../../maps/StinsonBeach.hdr",
            )
            scene_path.write_text(scene_text, encoding="utf-8")
        image_path = MEDIA / f"{slug(op_id)}.png"
        if render_stills:
            render(scene_path, image_path, width, recipe.get("frame"),
                   recipe.get("camera", "/World/Cam"))
        updates[op_id] = {
            "image": image_path.relative_to(ROOT).as_posix(),
            "caption": recipe["caption"],
            "renderer": "Storm / UsdGenHairStrands",
            "scene": scene_path.relative_to(ROOT).as_posix(),
            "generator": "tools/render_operator_examples.py",
            "provenance": "Captured from the live usdGen Hydra scene index; operator and parameter values are authored in the linked USD scene.",
        }
        print(f"{op_id}: {image_path.relative_to(ROOT)}", flush=True)
    MEDIA_JSON.parent.mkdir(parents=True, exist_ok=True)
    registry = json.loads(MEDIA_JSON.read_text(encoding="utf-8")) if MEDIA_JSON.exists() else {"version": 1, "operators": {}}
    registry["copyright"] = "Copyright (c) 2026 Nick Burkard"
    registry["license"] = "SPDX-License-Identifier: MIT"
    for op_id, record in updates.items():
        previous = registry["operators"].get(op_id)
        if not render_stills and previous:
            # Stage regeneration alone must not relabel an already published
            # MoonRay or native fixture result as a fresh Storm capture.
            continue
        if previous:
            for key in ("animation", "animationProvenance"):
                if key in previous:
                    record[key] = previous[key]
        registry["operators"][op_id] = record
    MEDIA_JSON.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="render the base patch")
    parser.add_argument("--all", action="store_true", help="generate all operator scenes and media")
    parser.add_argument("--op", action="append", help="generate one schema ID; may be repeated")
    parser.add_argument("--scenes-only", action="store_true", help="author USD scenes and registry without rendering")
    parser.add_argument("--width", type=int, default=1536)
    args = parser.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "sample.usda"
    path.write_text(scene(), encoding="utf-8")
    if args.sample:
        render(path, SCRATCH / "sample.png", args.width)
        print(SCRATCH / "sample.png")
    if args.all or args.op:
        generate_operators(set(args.op) if args.op else None,
                           args.width, not args.scenes_only)


if __name__ == "__main__":
    main()
