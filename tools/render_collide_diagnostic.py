#!/usr/bin/env python3
"""Bake and render a real clumped groom pressed by an animated sphere collider."""

# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("USDGEN_BUILD", ROOT / "build")).resolve()
DOCS_BUILD = Path(os.environ.get(
    "USDGEN_DOCS_BUILD", ROOT / "renders/docs/baker-build")).resolve()
SCENE = ROOT / "examples/docs/operators/collide.usda"
SCRATCH = ROOT / "renders/docs/pathtrace"
OUTPUT = ROOT / "docs/site/media/collide.png"
GIF = ROOT / "docs/site/media/collide.gif"
DESCRIPTION = "/World/Groom/Fur"
SETTLED_FRAME = 99

# A workstation pxr shim can be installed through a .pth file. The actual USD
# runtime is selected by the repository viewer launcher under python -S.
if "site" in sys.modules:
    raise RuntimeError("Run with python -S to load real OpenUSD")
sys.path.append(sysconfig.get_paths()["purelib"])
settings = json.loads(subprocess.run(
    ["powershell", "-NoProfile", "-File", str(ROOT / "bin/launch_usdview.ps1"),
     "-PrintEnv", "--renderer", "HdMoonrayRendererDebugPlugin"],
    cwd=ROOT, text=True, capture_output=True, check=True).stdout)
for key in ("PATH", "PYTHONPATH", "PXR_PLUGINPATH_NAME"):
    if settings.get(key):
        settings[key] = settings[key].replace(str(ROOT / "build"), str(BUILD))
for name in ("PATH", "PYTHONPATH", "PXR_PLUGINPATH_NAME", "ARRAS_SESSION_PATH", "RDL2_DSO_PATH"):
    if settings.get(name):
        os.environ[name] = settings[name]
sys.path[:0] = os.environ["PYTHONPATH"].split(os.pathsep)

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pxr import Gf, Usd, UsdGeom

sys.path.insert(0, str(ROOT / "tools/docs"))
from render_pathtraced import viewer_environment as _viewer_environment
from collide_limit_proof import (RadialSurface, convergence_error, read_far_grid,
                                 stitch_far_grid, strand_masks)


RENDER_COMPLEXITY = "veryhigh"
USD_PREFIX = Path(os.environ["PYTHONPATH"].split(os.pathsep)[-1]).parents[1]


def viewer_environment() -> dict[str, str]:
    env = _viewer_environment()
    for key in ("PATH", "PYTHONPATH", "PXR_PLUGINPATH_NAME"):
        env[key] = env.get(key, "").replace(str(ROOT / "build"), str(BUILD))
    return env


def docs_executable(name: str) -> Path:
    release = DOCS_BUILD / "Release" / name
    direct = DOCS_BUILD / name
    return release if release.is_file() else direct


def runtime_paths() -> dict[str, Path]:
    """Pin both the native solver and the OpenUSD/OpenSubdiv installation."""
    return {
        "scene_sha256": SCENE,
        "solver_source_sha256": ROOT / "libs/core/primitives/usdGenCurves/usdGenCurves/ops/collide.cpp",
        "limit_source_sha256": ROOT / "libs/core/primitives/usdGenSurface/usdGenSurface/limitSurface.cpp",
        "solver_sha256": BUILD / "usdGen.dll",
        "imaging_sha256": BUILD / "usdGenImaging.dll",
        "baker_sha256": baker_path(),
        "far_oracle_sha256": docs_executable("usdGenCollideFarOracle.exe"),
        "usd_sha256": USD_PREFIX / "lib/usd_usd.dll",
        "pxosd_sha256": USD_PREFIX / "lib/usd_pxOsd.dll",
        "opensubdiv_sha256": USD_PREFIX / "lib/osdCPU.lib",
    }


def render_refined(stage: Path, camera: str, target: Path, width: int,
                   env: dict[str, str]) -> None:
    """Render subdivided collider at the same explicit quality for every capture."""
    name = shutil.which("usdrecord", path=env.get("PATH"))
    if not name:
        raise FileNotFoundError("usdrecord is unavailable")
    recorder = Path(name)
    if recorder.suffix.lower() == ".cmd" and recorder.with_suffix("").exists():
        recorder = recorder.with_suffix("")
    command = [sys.executable, str(recorder), "--renderer",
               "HdMoonrayRendererDebugPlugin", "--imageWidth", str(width),
               "--complexity", RENDER_COMPLEXITY, "--disableCameraLight",
               "--camera", camera, "--defaultTime", str(stage), str(target)]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".render.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"MoonRay exited {result.returncode}: "
                           f"{target.with_suffix('.render.log')}")
    with Image.open(target) as image:
        image.load()
        if image.width != width:
            raise RuntimeError(f"MoonRay wrote an unexpected image: {target}")


def baker_path() -> Path:
    """Use the matching in-tree native baker for the editable transform scene."""
    target = docs_executable("usdGenBakeGroom.exe")
    if not target.exists():
        raise FileNotFoundError(f"Build the in-tree native baker: {target}")
    return target


def bake(scene: Path, descriptions: list[str], material: str | None,
         target: Path, env: dict[str, str], time_code: float) -> None:
    assert_runtime_unchanged()
    if descriptions != [DESCRIPTION]:
        raise RuntimeError("Collide diagnostic expects one groom description")
    command = [str(baker_path()), str(scene), DESCRIPTION, str(target),
               material or "-", str(time_code)]
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with target.with_suffix(".bake.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    if completed.returncode or not target.exists():
        raise RuntimeError(f"Native Collide baker failed at frame {time_code}: "
                           f"{target.with_suffix('.bake.log')}")
    if scene.resolve() == SCENE.resolve():
        diagnostics = target.with_suffix(".bake.log").read_text(
            encoding="utf-8", errors="replace")
        for polygon in ("/World/Patch", "/World/ScalpVolume"):
            expected = (f"collider '{polygon}' subdivisionScheme=none: "
                        "authored polygon surface; using polygon collision")
            if expected not in diagnostics:
                raise RuntimeError(f"Missing expected polygon fallback for {polygon}: "
                                   f"{target.with_suffix('.bake.log')}")
        if ("collider '/World/Shield'" in diagnostics and
                "using polygon collision" in diagnostics.split(
                    "collider '/World/Shield'", 1)[1].splitlines()[0]):
            raise RuntimeError(f"Shield fell back to polygon collision: "
                               f"{target.with_suffix('.bake.log')}")
    assert_runtime_unchanged()
    print(f"native bake frame {time_code:g}: {elapsed:.3f}s ({target.name})",
          flush=True)


def pure_render_environment(env: dict[str, str]) -> dict[str, str]:
    """Render the baked standard-USD stage without live groom scene indices."""
    pure = env.copy()
    build = str(BUILD).lower()
    def unrelated(part: str) -> bool:
        return not str(Path(part).resolve()).lower().startswith(build)
    for key in ("PATH", "PYTHONPATH", "PXR_PLUGINPATH_NAME"):
        pure[key] = os.pathsep.join(part for part in pure.get(key, "").split(os.pathsep)
                                    if part and unrelated(part))
    pure["USDGEN_ENABLE"] = "false"
    pure["USDGENPOMADE_TEST_TUBE"] = "0"
    return pure


@lru_cache(maxsize=1)
def run_identity() -> dict[str, str | int]:
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    stage = Usd.Stage.Open(str(SCENE))
    iterations = stage.GetPrimAtPath(DESCRIPTION + "/Ops/collide").GetAttribute(
        "usdGen:iterations").Get()
    return {**{name: digest(path) for name, path in runtime_paths().items()},
            "iterations": iterations}


def assert_runtime_unchanged() -> None:
    identity = run_identity()
    for key, path in runtime_paths().items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity[key]:
            raise RuntimeError(f"Native Collide runtime/source changed during this run: {path}")


def disabled_scene(target: Path) -> None:
    stage = Usd.Stage.CreateNew(str(target))
    stage.GetRootLayer().subLayerPaths.append(
        os.path.relpath(SCENE, target.parent).replace("\\", "/"))
    stage.OverridePrim(DESCRIPTION + "/Ops/collide").SetActive(False)
    stage.GetRootLayer().Save()
    if Usd.Stage.Open(str(target)).GetPrimAtPath(DESCRIPTION + "/Ops/collide"):
        raise RuntimeError("Failed to disable Collide in comparison source")


def cooked_curves(path: Path):
    stage = Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.BasisCurves):
            curves = UsdGeom.BasisCurves(prim)
            return (list(curves.GetCurveVertexCountsAttr().Get()),
                    [tuple(p) for p in curves.GetPointsAttr().Get()])
    raise RuntimeError(f"No cooked groom in {path}")


def world_mesh_points(stage, prim_path: str, time_code: float) -> np.ndarray:
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
    if not mesh:
        raise RuntimeError(f"Collider mesh missing: {prim_path}")
    matrix = UsdGeom.XformCache(Usd.TimeCode(time_code)).GetLocalToWorldTransform(
        mesh.GetPrim())
    return np.asarray([tuple(matrix.Transform(Gf.Vec3d(*point)))
                       for point in mesh.GetPointsAttr().Get(time_code)], dtype=np.float64)


def assert_subdivision_authoring(stage) -> None:
    shield = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Shield"))
    floor = UsdGeom.Mesh(stage.GetPrimAtPath("/World/ScalpVolume"))
    if (not shield or shield.GetSubdivisionSchemeAttr().Get() != "catmullClark"
            or set(shield.GetFaceVertexCountsAttr().Get()) != {4}):
        raise RuntimeError("Shield must be an authored quad Catmull-Clark mesh")
    if not floor or floor.GetSubdivisionSchemeAttr().Get() != "none":
        raise RuntimeError("ScalpVolume must retain its polygonal collision boundary")


@lru_cache(maxsize=1)
def shield_oracle() -> tuple[RadialSurface, float]:
    """Use the 64-rate Far surface, checked against the 128-rate samples."""
    assert_runtime_unchanged()
    executable = runtime_paths()["far_oracle_sha256"]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    grids = {}
    for rate in (64, 128):
        sample = SCRATCH / f"collide-shield-far-{rate}.bin"
        command = [str(executable), str(SCENE), "/World/Shield", str(sample),
                   str(rate), "27"]
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"Independent Far oracle failed: {result.stderr}")
        grids[rate] = read_far_grid(sample)
    source = Usd.Stage.Open(str(SCENE))
    shield = UsdGeom.Mesh(source.GetPrimAtPath("/World/Shield"))
    face_vertices = np.asarray(
        shield.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 4)
    stitch_shifts = {}
    for rate in (64, 128):
        grids[rate], stitch_shifts[rate] = stitch_far_grid(
            grids[rate], face_vertices)
    error = convergence_error(grids[64], grids[128])
    center = np.asarray((0.5, 0.525, 0.5), dtype=np.float64)
    authored = source.GetPrimAtPath("/World/Shield").GetAttribute(
        "xformOp:translate").Get(27)
    if np.max(np.abs(np.asarray(tuple(authored)) - center)) > 1e-9:
        raise RuntimeError("Far oracle center no longer matches authored frame 27")
    if error > 2e-4:
        raise RuntimeError(f"Far 64/128 convergence exceeded proof budget: {error}")
    surface = RadialSurface.from_far(grids[64], center)
    surface.far_stitch_max_shift_64 = stitch_shifts[64]
    surface.far_stitch_max_shift_128 = stitch_shifts[128]
    assert_runtime_unchanged()
    return surface, error


def shield_center(time_code: float) -> np.ndarray:
    stage = Usd.Stage.Open(str(SCENE))
    attr = stage.GetPrimAtPath("/World/Shield").GetAttribute("xformOp:translate")
    return np.asarray(tuple(attr.Get(time_code)), dtype=np.float64)


def shield_cage(time_code: float) -> RadialSurface:
    stage = Usd.Stage.Open(str(SCENE))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Shield"))
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 4)
    triangles = np.concatenate((faces[:, (0, 1, 2)], faces[:, (0, 2, 3)]))
    return RadialSurface(world_mesh_points(stage, "/World/Shield", time_code),
                         triangles, shield_center(time_code))


def convex_mesh_hull(stage, prim_path: str, time_code: float):
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
    points = world_mesh_points(stage, prim_path, time_code)
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    if len(points) < 8 or set(counts) != {3} or len(indices) != 3 * len(counts):
        raise RuntimeError(f"Collider must be a closed triangular mesh: {prim_path}")
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    center = points.mean(axis=0)
    if np.any(np.einsum('ij,ij->i', normals, a - center) <= 0):
        raise RuntimeError(f"Collider face normals are not outward: {prim_path}")
    if np.max(np.einsum('vfi,fi->vf', points[:, None, :] - a[None, :, :], normals)) > 1e-5:
        raise RuntimeError(f"Collider is not convex: {prim_path}")
    return a, normals


def scalp_hull(time_code: float):
    stage = Usd.Stage.Open(str(SCENE))
    return convex_mesh_hull(stage, "/World/ScalpVolume", time_code)


def sphere_pose_sha256(time_code: float) -> str:
    stage = Usd.Stage.Open(str(SCENE))
    points = world_mesh_points(stage, "/World/Shield", time_code).astype(np.float32)
    return hashlib.sha256(points.tobytes()).hexdigest()


def penetration_masks(counts, points, hull):
    plane_points, normals = hull
    positions = np.asarray(points, dtype=np.float64)
    valid_segments = np.ones(len(positions) - 1, dtype=bool)
    valid_segments[np.cumsum(counts)[:-1] - 1] = False
    interior = np.ones(len(positions), dtype=bool)
    enter = np.zeros(len(valid_segments), dtype=np.float64)
    leave = np.ones(len(valid_segments), dtype=np.float64)
    active = valid_segments.copy()
    for plane_point, normal in zip(plane_points, normals):
        distance = (positions - plane_point) @ normal
        interior &= distance < -1e-5
        a_distance = distance[:-1] + 1e-4
        b_distance = distance[1:] + 1e-4
        delta = b_distance - a_distance
        flat = np.abs(delta) < 1e-12
        active &= ~(flat & (a_distance >= 0))
        t = np.divide(-a_distance, delta, out=np.zeros_like(delta), where=~flat)
        enter = np.maximum(enter, np.where(delta < -1e-12, t, 0))
        leave = np.minimum(leave, np.where(delta > 1e-12, t, 1))
        active &= leave - enter > 1e-5
    return interior, active


def validate(disabled: Path, enabled: Path, time_code: float,
             require_contact: bool = True, report_path: Path | None = None) -> dict:
    assert_runtime_unchanged()
    source = Usd.Stage.Open(str(SCENE))
    targets = [str(path) for path in source.GetPrimAtPath(
        DESCRIPTION + "/Ops/collide").GetRelationship("usdGen:colliders").GetTargets()]
    if targets != ["/World/Shield", "/World/ScalpVolume"]:
        raise RuntimeError(f"Expected both authored closed collision targets: {targets}")
    counts_off, off = cooked_curves(disabled)
    counts_on, on = cooked_curves(enabled)
    if counts_off != counts_on or len(off) != len(on):
        raise RuntimeError("Comparison bakes have different strand topology")
    roots = []
    cursor = 0
    for count in counts_off:
        roots.append(cursor)
        cursor += count
    if cursor != len(off):
        raise RuntimeError("Invalid cooked curve counts")
    shifts = [sum((a - b) ** 2 for a, b in zip(p, q)) ** 0.5
              for p, q in zip(off, on)]
    length_error = 0.0
    cursor = 0
    for count in counts_off:
        for index in range(cursor, cursor + count - 1):
            off_len = sum((off[index][axis] - off[index + 1][axis]) ** 2
                          for axis in range(3)) ** 0.5
            on_len = sum((on[index][axis] - on[index + 1][axis]) ** 2
                         for axis in range(3)) ** 0.5
            length_error = max(length_error, abs(off_len - on_len))
        cursor += count
    assert_subdivision_authoring(source)
    shield, far_convergence = shield_oracle()
    center = shield_center(time_code)
    inside_off, crossed_off, clearance_off = strand_masks(
        counts_off, off, shield, center)
    inside_on, crossed_on, clearance_on = strand_masks(
        counts_on, on, shield, center)
    near_limit = np.abs(clearance_on) < 0.003
    cage_clearance = shield_cage(time_code).clearance(
        np.asarray(on, dtype=np.float64)[near_limit], center)
    scalp_inside_on, scalp_crossed_on = penetration_masks(
        counts_on, on, scalp_hull(time_code))
    def crossing_details(mask):
        details = []
        cursor = 0
        for curve_index, count in enumerate(counts_off):
            for cv_index in range(count - 1):
                i = cursor + cv_index
                if mask[i]:
                    details.append({
                        "curve": curve_index, "segment": cv_index,
                        "off_a": off[i], "off_b": off[i + 1],
                        "on_a": on[i], "on_b": on[i + 1],
                    })
            cursor += count
        return details
    def interior_details(mask):
        details = []
        cursor = 0
        for curve_index, count in enumerate(counts_off):
            for cv_index in range(count):
                i = cursor + cv_index
                if mask[i]:
                    details.append({"curve": curve_index, "cv": cv_index,
                                    "off": off[i], "on": on[i]})
            cursor += count
        return details
    report = {
        "frame": time_code,
        "run_identity": run_identity(),
        "disabled_bake_sha256": hashlib.sha256(disabled.read_bytes()).hexdigest(),
        "enabled_bake_sha256": hashlib.sha256(enabled.read_bytes()).hexdigest(),
        "sphere_pose_sha256": sphere_pose_sha256(time_code),
        "shield_limit_oracle": "OpenSubdiv Far, 64 samples per coarse face edge",
        "shield_limit_oracle_exhaustive_fallback": (
            "all Far triangles for centroid-KD misses; same surface and tolerance"),
        "shield_limit_oracle_exhaustive_fallback_rays": (
            shield.exhaustive_fallback_rays),
        "shield_limit_oracle_stitch_max_shift_64": (
            shield.far_stitch_max_shift_64),
        "shield_limit_oracle_stitch_max_shift_128": (
            shield.far_stitch_max_shift_128),
        "far_64_to_128_max_chord_error": far_convergence,
        "shield_limit_probe_tolerance": 2e-4,
        "shield_limit_contact_cvs_on": int(np.count_nonzero(near_limit)),
        "shield_limit_min_clearance_on": float(clearance_on.min()),
        "shield_cage_contact_clearance_median": float(np.median(cage_clearance)) if len(cage_clearance) else None,
        "shield_cage_limit_clearance_gap_max": float(np.max(np.abs(cage_clearance - clearance_on[near_limit]))) if len(cage_clearance) else None,
        "strands": len(counts_off),
        "cvs": len(off),
        "moved_cvs": sum(s > 1e-5 for s in shifts),
        "max_displacement": max(shifts, default=0),
        "max_root_displacement": max((shifts[i] for i in roots), default=0),
        "max_segment_length_error": length_error,
        "min_y_on": min((p[1] for p in on), default=0),
        "below_scalp_cvs_on": sum(p[1] < -1e-4 for p in on),
        "sphere_interior_cvs_off": int(np.count_nonzero(inside_off)),
        "sphere_interior_cvs_on": int(np.count_nonzero(inside_on)),
        "sphere_crossing_segments_off": int(np.count_nonzero(crossed_off)),
        "sphere_crossing_segments_on": int(np.count_nonzero(crossed_on)),
        "scalp_volume_interior_cvs_on": int(np.count_nonzero(scalp_inside_on)),
        "scalp_volume_crossing_segments_on": int(np.count_nonzero(scalp_crossed_on)),
        "collider_targets": targets,
        "crossing_segment_details_on": crossing_details(crossed_on),
        "interior_cv_details_on": interior_details(inside_on),
    }
    report_path = report_path or SCRATCH / "collide-validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if require_contact and report["sphere_interior_cvs_off"] < 100:
        raise RuntimeError(f"Sphere does not intersect enough groom CVs: {report}")
    if report["max_root_displacement"] > 1e-5:
        raise RuntimeError(f"Collide moved scalp roots: {report}")
    if report["below_scalp_cvs_on"]:
        raise RuntimeError(f"Collide pushed fur below the scalp: {report}")
    if report["max_segment_length_error"] > 1e-4:
        raise RuntimeError(f"Stiff Collide changed segment lengths: {report_path}")
    if report["sphere_interior_cvs_on"] != 0 or report["sphere_crossing_segments_on"] != 0:
        raise RuntimeError(f"Collide left {report['sphere_interior_cvs_on']} interior CVs "
                           f"and {report['sphere_crossing_segments_on']} crossing segments; "
                           f"details: {report_path}")
    if (report["scalp_volume_interior_cvs_on"] != 0
            or report["scalp_volume_crossing_segments_on"] != 0):
        raise RuntimeError(f"Scalp volume still intersects cooked groom: {report_path}")
    return report


def settled_pair(env, reuse: bool):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    muted_source = SCRATCH / "collide-off-source.usda"
    disabled_scene(muted_source)
    muted = SCRATCH / "collide-off-baked.usdc"
    settled = SCRATCH / "collide-settled-baked.usdc"
    prior_report = SCRATCH / "collide-validation.json"
    if reuse and muted.exists() and settled.exists() and prior_report.exists():
        prior = json.loads(prior_report.read_text(encoding="utf-8"))
        reuse = (prior.get("run_identity") == run_identity()
                 and prior.get("disabled_bake_sha256") == hashlib.sha256(
                     muted.read_bytes()).hexdigest()
                 and prior.get("enabled_bake_sha256") == hashlib.sha256(
                     settled.read_bytes()).hexdigest())
    else:
        reuse = False
    if not reuse:
        bake(muted_source, [DESCRIPTION], "/World/Looks/Fur", muted, env, SETTLED_FRAME)
        bake(SCENE, [DESCRIPTION], "/World/Looks/Fur", settled, env, SETTLED_FRAME)
    report = validate(muted, settled, SETTLED_FRAME)
    (SCRATCH / "collide-validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return settled


def sparse_preview(env, muted: Path, settled: Path) -> None:
    """Check opening/contact/slide poses before committing to 100 cooks."""
    render_env = pure_render_environment(env)
    panels = []
    for frame in (0, 27, 99):
        baked = settled if frame == 99 else SCRATCH / f"collide-probe-{frame:02d}.usdc"
        if frame != 99:
            bake(SCENE, [DESCRIPTION], "/World/Looks/Fur", baked, env, frame)
        report = validate(muted, baked, frame, require_contact=frame != 0,
                          report_path=SCRATCH / f"collide-probe-{frame:02d}-validation.json")
        if frame == 0 and (report["sphere_interior_cvs_off"] or
                           report["sphere_crossing_segments_off"] or
                           report["moved_cvs"]):
            raise RuntimeError("Opening Shield pose must be clear of the groom")
        image_path = SCRATCH / f"collide-probe-{frame:02d}.png"
        render_refined(frozen_render_stage(baked, frame), "/World/CamMotion",
                       image_path, 600, render_env)
        with Image.open(image_path) as image:
            panels.append(image.convert("RGB"))
        print(f"sparse frame {frame}: moved={report['moved_cvs']} "
              f"Far contacts={report['shield_limit_contact_cvs_on']} "
              f"min clearance={report['shield_limit_min_clearance_on']:.6g}",
              flush=True)
    sheet = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)))
    x = 0
    for frame, panel in zip((0, 27, 99), panels):
        sheet.paste(panel, (x, 0))
        ImageDraw.Draw(sheet).text((x + 12, 12), f"FRAME {frame:03d}",
                                   fill="#fff2d4", stroke_width=2,
                                   stroke_fill="#17242c")
        x += panel.width
    path = SCRATCH / "collide-sparse-contact-sheet.png"
    sheet.save(path)
    print(f"sparse preview: {path}", flush=True)


def timeline_validation(env, muted: Path, start_frame: int = 0) -> None:
    """Cook and validate every usdview playback frame before rendering media."""
    records = []
    for time_code in range(start_frame):
        prior = SCRATCH / f"collide-frame-{time_code:02d}-validation.json"
        if not prior.exists():
            raise RuntimeError(f"Missing prior frame validation: {prior}")
        report = json.loads(prior.read_text(encoding="utf-8"))
        if (report.get("frame") != time_code or report.get("run_identity") != run_identity()
                or report.get("enabled_bake_sha256") != hashlib.sha256((
                    SCRATCH / f"collide-motion-{time_code:02d}.usdc").read_bytes()).hexdigest()
                or report["sphere_interior_cvs_on"] or report["sphere_crossing_segments_on"]
                or report["scalp_volume_interior_cvs_on"]
                or report["scalp_volume_crossing_segments_on"]
                or report["below_scalp_cvs_on"] or report["max_root_displacement"] > 1e-5
                or report["max_segment_length_error"] > 1e-4):
            raise RuntimeError(f"Prior frame {time_code} did not pass: {prior}")
        records.append({"frame": time_code, "moved_cvs": report["moved_cvs"],
                        "sphere_pose_sha256": report["sphere_pose_sha256"],
                        "max_root_displacement": report["max_root_displacement"],
                        "max_segment_length_error": report["max_segment_length_error"],
                        "below_scalp_cvs": report["below_scalp_cvs_on"],
                        "sphere_interior_cvs": report["sphere_interior_cvs_on"],
                        "sphere_crossing_segments": report["sphere_crossing_segments_on"],
                        "scalp_volume_interior_cvs": report["scalp_volume_interior_cvs_on"],
                        "scalp_volume_crossing_segments": report[
                            "scalp_volume_crossing_segments_on"]})
    for time_code in range(start_frame, SETTLED_FRAME + 1):
        baked = SCRATCH / f"collide-motion-{time_code:02d}.usdc"
        bake(SCENE, [DESCRIPTION], "/World/Looks/Fur", baked, env, time_code)
        report = validate(muted, baked, time_code, require_contact=False,
                          report_path=SCRATCH / f"collide-frame-{time_code:02d}-validation.json")
        if time_code == 0 and (report["sphere_interior_cvs_off"]
                               or report["sphere_crossing_segments_off"]
                               or report["moved_cvs"]):
            raise RuntimeError("The opening sphere pose must be clear of the groom")
        records.append({"frame": time_code, "moved_cvs": report["moved_cvs"],
                        "sphere_pose_sha256": report["sphere_pose_sha256"],
                        "max_root_displacement": report["max_root_displacement"],
                        "max_segment_length_error": report["max_segment_length_error"],
                        "below_scalp_cvs": report["below_scalp_cvs_on"],
                        "sphere_interior_cvs": report["sphere_interior_cvs_on"],
                        "sphere_crossing_segments": report["sphere_crossing_segments_on"],
                        "scalp_volume_interior_cvs": report["scalp_volume_interior_cvs_on"],
                        "scalp_volume_crossing_segments": report[
                            "scalp_volume_crossing_segments_on"]})
        print(f"validated frame {time_code}/{SETTLED_FRAME}", flush=True)
    if len({record["sphere_pose_sha256"] for record in records}) != SETTLED_FRAME + 1:
        raise RuntimeError("Sphere must have a distinct evaluated mesh pose every frame")
    (SCRATCH / "collide-timeline-validation.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8")


def bake_range(env, first: int, last: int) -> None:
    """Cook a disjoint inclusive range; proof remains a serial gate."""
    for time_code in range(first, last + 1):
        baked = SCRATCH / f"collide-motion-{time_code:02d}.usdc"
        bake(SCENE, [DESCRIPTION], "/World/Looks/Fur", baked, env, time_code)
        print(f"baked range frame {time_code} ({first}-{last})", flush=True)


def prove_existing_timeline(muted: Path) -> None:
    """Serially prove all prepared bakes before the final identity check/render."""
    for time_code in range(SETTLED_FRAME + 1):
        baked = SCRATCH / f"collide-motion-{time_code:02d}.usdc"
        if not baked.is_file():
            raise RuntimeError(f"Missing prepared bake at frame {time_code}: {baked}")
        validate(muted, baked, time_code, require_contact=False,
                 report_path=SCRATCH / f"collide-frame-{time_code:02d}-validation.json")
        print(f"serial proof frame {time_code}/{SETTLED_FRAME}", flush=True)


def frozen_render_stage(baked: Path, time_code: float) -> Path:
    """Freeze the authored translate at the same time as the native groom cook."""
    stage = Usd.Stage.Open(str(baked))
    source = Usd.Stage.Open(str(SCENE))
    expected = world_mesh_points(source, "/World/Shield", time_code)
    actual = world_mesh_points(stage, "/World/Shield", time_code)
    if not np.array_equal(actual, expected) and np.max(np.abs(actual - expected)) > 1e-6:
        raise RuntimeError(f"Baked Shield world pose differs from source at frame {time_code}")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Shield"))
    points = mesh.GetPointsAttr()
    value = points.Get(time_code)
    for sample in points.GetTimeSamples():
        points.ClearAtTime(sample)
    points.Set(value)
    translate = mesh.GetPrim().GetAttribute("xformOp:translate")
    if not translate or not translate.GetTimeSamples():
        raise RuntimeError("Shield needs editable animated xformOp:translate")
    position = translate.Get(time_code)
    for sample in translate.GetTimeSamples():
        translate.ClearAtTime(sample)
    translate.Set(position)
    target = baked.with_name(baked.stem + "-render.usdc")
    if not stage.Export(str(target)):
        raise RuntimeError(f"Cannot export frozen render stage: {target}")
    frozen = Usd.Stage.Open(str(target))
    assert_subdivision_authoring(frozen)
    frozen_points = UsdGeom.Mesh(frozen.GetPrimAtPath("/World/Shield")).GetPointsAttr()
    frozen_translate = frozen.GetPrimAtPath("/World/Shield").GetAttribute("xformOp:translate")
    observed = world_mesh_points(frozen, "/World/Shield", Usd.TimeCode.Default())
    if (frozen_points.GetTimeSamples() or frozen_translate.GetTimeSamples()
            or np.max(np.abs(observed - expected)) > 1e-6):
        raise RuntimeError(f"Frozen Shield pose differs from source at frame {time_code}")
    standard = {"BasisCurves", "Camera", "DistantLight", "DomeLight", "Material",
                "Mesh", "Scope", "Shader", "Xform"}
    types = {prim.GetTypeName() for prim in frozen.Traverse()}
    if not types <= standard or not frozen.GetPrimAtPath("/World/Groom/Baked_Fur"):
        raise RuntimeError(f"Render stage contains live groom authoring: {types}")
    return target


def batch_render_stage() -> Path:
    """Time-sample only the three native fields that change across cooked frames."""
    target = SCRATCH / "collide-time-sampled.usdc"
    temporary = SCRATCH / "collide-time-sampled-next.usdc"
    if temporary.exists():
        temporary.unlink()
    first = SCRATCH / "collide-motion-00.usdc"
    stage = Usd.Stage.CreateNew(str(temporary))
    stage.GetRootLayer().subLayerPaths.append(
        os.path.relpath(first, SCRATCH).replace("\\", "/"))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(SETTLED_FRAME)
    stage.SetTimeCodesPerSecond(24)
    stage.SetFramesPerSecond(24)
    curve = stage.OverridePrim("/World/Groom/Baked_Fur")
    shield = stage.OverridePrim("/World/Shield")
    points_attr = curve.GetAttribute("points")
    time_attr = curve.GetAttribute("usdGen:bakedTime")
    translate_attr = shield.GetAttribute("xformOp:translate")
    if not points_attr or not time_attr or not translate_attr:
        raise RuntimeError("Native baked stage lacks sampled fields")
    point_hashes = {}
    pose_hashes = {}
    for frame in range(SETTLED_FRAME + 1):
        baked = SCRATCH / f"collide-motion-{frame:02d}.usdc"
        report = json.loads((SCRATCH / f"collide-frame-{frame:02d}-validation.json")
                            .read_text(encoding="utf-8"))
        if (report.get("run_identity") != run_identity()
                or report.get("enabled_bake_sha256") != hashlib.sha256(
                    baked.read_bytes()).hexdigest()):
            raise RuntimeError(f"Unproven native bake at frame {frame}")
        native = Usd.Stage.Open(str(baked))
        cooked = native.GetPrimAtPath("/World/Groom/Baked_Fur")
        value = cooked.GetAttribute("points").Get()
        points_attr.Set(value, frame)
        time_attr.Set(cooked.GetAttribute("usdGen:bakedTime").Get(), frame)
        translate_attr.Set(native.GetPrimAtPath("/World/Shield").GetAttribute(
            "xformOp:translate").Get(frame), frame)
        point_hashes[frame] = hashlib.sha256(np.asarray(
            value, dtype=np.float32).tobytes()).hexdigest()
        pose_hashes[frame] = sphere_pose_sha256(frame)
        native = None
    if not stage.GetRootLayer().Save():
        raise RuntimeError("Cannot save time-sampled baked USD")
    stage = None
    check = Usd.Stage.Open(str(temporary))
    assert_subdivision_authoring(check)
    if check.GetPrimAtPath("/World/Groom/Baked_Fur").GetAttribute(
            "points").GetTimeSamples() != list(range(100)):
        raise RuntimeError("Batch stage lacks 100 native point samples")
    for frame in range(SETTLED_FRAME + 1):
        observed = check.GetPrimAtPath("/World/Groom/Baked_Fur").GetAttribute(
            "points").Get(frame)
        if hashlib.sha256(np.asarray(observed, dtype=np.float32).tobytes()).hexdigest() != point_hashes[frame]:
            raise RuntimeError(f"Batch groom differs native frame {frame}")
        world = world_mesh_points(check, "/World/Shield", frame).astype(np.float32)
        if hashlib.sha256(world.tobytes()).hexdigest() != pose_hashes[frame]:
            raise RuntimeError(f"Batch sphere differs authored frame {frame}")
    check = None
    temporary.replace(target)
    return target


def batch_render_images(env, width: int) -> dict[int, Path]:
    """Gate a single recorder by sparse parity against separate frozen poses."""
    stage = batch_render_stage()
    render_env = pure_render_environment(env)
    name = shutil.which("usdrecord", path=render_env["PATH"])
    if not name:
        raise FileNotFoundError("usdrecord is unavailable")
    recorder = Path(name)
    if recorder.suffix.lower() == ".cmd" and recorder.with_suffix("").exists():
        recorder = recorder.with_suffix("")
    def record(frames: str, pattern: Path, log: Path, image_width: int) -> None:
        command = [sys.executable, str(recorder), "--renderer",
                   "HdMoonrayRendererDebugPlugin", "--imageWidth", str(image_width),
                   "--complexity", RENDER_COMPLEXITY,
                   "--disableCameraLight", "--camera", "/World/CamMotion",
                   "--frames", frames, str(stage), str(pattern)]
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, env=render_env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Batch MoonRay exited {result.returncode}: {log}")
    record("0,27,99", SCRATCH / "collide-batch-probe-###.###.png",
           SCRATCH / "collide-batch-probe.render.log", 480)
    for frame in (0, 27, 99):
        sampled = SCRATCH / f"collide-batch-probe-{frame:03d}.000.png"
        independent = SCRATCH / f"collide-independent-{frame:03d}-480.png"
        render_refined(frozen_render_stage(SCRATCH / f"collide-motion-{frame:02d}.usdc", frame),
                       "/World/CamMotion", independent, 480, render_env)
        with Image.open(sampled) as a_image, Image.open(independent) as b_image:
            a = np.asarray(a_image.convert("RGB"), dtype=np.int16)
            b = np.asarray(b_image.convert("RGB"), dtype=np.int16)
        if a.shape != b.shape or np.abs(a - b).mean() > 1.0:
            raise RuntimeError(f"Batch MoonRay pose/look differs independent frame {frame}")
    record("0:99", SCRATCH / f"collide-batch-{width}-###.###.png",
           SCRATCH / f"collide-batch-{width}.render.log", width)
    images = {frame: SCRATCH / f"collide-batch-{width}-{frame:03d}.000.png"
              for frame in range(SETTLED_FRAME + 1)}
    for frame, path in images.items():
        with Image.open(path) as image:
            image.load()
            if image.width != width:
                raise RuntimeError(f"Batch frame {frame} has wrong width")
    if len({hashlib.sha256(path.read_bytes()).hexdigest()
            for path in images.values()}) != SETTLED_FRAME + 1:
        raise RuntimeError("Batch MoonRay did not produce 100 distinct images")
    independent = SCRATCH / f"collide-independent-099-{width}.png"
    render_refined(frozen_render_stage(SCRATCH / "collide-motion-99.usdc", 99),
                   "/World/CamMotion", independent, width, render_env)
    with Image.open(images[99]) as a_image, Image.open(independent) as b_image:
        a = np.asarray(a_image.convert("RGB"), dtype=np.int16)
        b = np.asarray(b_image.convert("RGB"), dtype=np.int16)
    same_time_error = np.abs(a - b).mean() if a.shape == b.shape else np.inf
    if same_time_error > 0.5:
        raise RuntimeError("Settled batch render differs independent same-width capture")
    wrong_time = SCRATCH / f"collide-independent-098-{width}.png"
    render_refined(frozen_render_stage(SCRATCH / "collide-motion-98.usdc", 98),
                   "/World/CamMotion", wrong_time, width, render_env)
    with Image.open(wrong_time) as image:
        previous = np.asarray(image.convert("RGB"), dtype=np.int16)
    wrong_time_error = np.abs(a - previous).mean() if a.shape == previous.shape else np.inf
    if wrong_time_error <= same_time_error + 0.02:
        raise RuntimeError("The frame-99 batch does not distinguish the wrong-time render")
    print(f"frame-99 render parity: same={same_time_error:.4f} "
          f"wrong-frame-98={wrong_time_error:.4f} RGB", flush=True)
    return images


def animation(env, width: int, batch: bool = False) -> None:
    # Every authored time code gets a distinct native groom cook and render.
    render_env = pure_render_environment(env)
    batch_images = batch_render_images(env, width) if batch else {}
    times = list(range(0, SETTLED_FRAME + 1))
    frames = []
    try:
        label_font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        label_font = ImageFont.load_default()
    for time_code in times:
        baked = SCRATCH / f"collide-motion-{time_code:02d}.usdc"
        frame = batch_images.get(time_code, SCRATCH / f"collide-motion-{time_code:02d}.png")
        if (not batch and (not frame.exists()
                           or frame.stat().st_mtime < baked.stat().st_mtime
                           or frame.stat().st_mtime < SCENE.stat().st_mtime)):
            render_refined(frozen_render_stage(baked, time_code), "/World/CamMotion", frame,
                           width, render_env)
        with Image.open(frame) as image:
            image.load()
            annotated = image.convert("RGB")
        report = json.loads((SCRATCH / f"collide-frame-{time_code:02d}-validation.json")
                            .read_text(encoding="utf-8"))
        contact = (report["sphere_interior_cvs_off"] > 0
                   or report["sphere_crossing_segments_off"] > 0
                   or report["moved_cvs"] > 0)
        label = f"FRAME {time_code:03d}  {'CONTACT' if contact else 'CLEAR'}"
        draw = ImageDraw.Draw(annotated)
        badge_right = max(202, draw.textbbox((22, 17), label, font=label_font)[2] + 12)
        draw.rounded_rectangle((12, 12, badge_right, 43), radius=7,
                               fill="#17242c")
        draw.text((22, 17), label, font=label_font,
                  fill="#f5d894" if contact else "#d7ecf1")
        frames.append(annotated)
    thumb_width = 128
    thumb_height = round(frames[0].height * thumb_width / frames[0].width)
    contact = Image.new("RGB", (thumb_width, thumb_height * len(frames)))
    for index, frame in enumerate(frames):
        contact.paste(frame.resize((thumb_width, thumb_height)),
                      (0, index * thumb_height))
    palette = contact.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    images = [frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
              for frame in frames]
    durations = [10 * (round((i + 1) * 100 / 24) - round(i * 100 / 24))
                 for i in range(len(images))]
    if len(images) != 100 or sum(durations) != 4170:
        raise RuntimeError("Expected 100 real 24 fps frames over 4170 ms")
    GIF.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(GIF, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False)
    with Image.open(GIF) as encoded:
        decoded_hashes = set()
        for index in range(encoded.n_frames):
            encoded.seek(index)
            decoded_hashes.add(hashlib.sha256(encoded.convert("RGBA").tobytes()).hexdigest())
        if encoded.n_frames != 100 or len(decoded_hashes) != 100:
            raise RuntimeError(f"Expected 100 distinct encoded GIF frames, got "
                               f"{encoded.n_frames} frames and {len(decoded_hashes)} images")
    print(f"animation: {GIF}", flush=True)


def main() -> None:
    global SCENE, SCRATCH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, default=SCENE,
                        help="Editable Collide source; use with --scratch for an isolated preview")
    parser.add_argument("--scratch", type=Path, default=SCRATCH,
                        help="Private bake/proof/render workspace")
    parser.add_argument("--preview", action="store_true", help="Render a 600 px settled preview")
    parser.add_argument("--probe-frames", action="store_true",
                        help="Bake/prove/render frames 0, 27 and 99 as a preview sheet")
    parser.add_argument("--animation", action="store_true", help="Render 100-frame descend-and-slide GIF")
    parser.add_argument("--batch", action="store_true",
                        help="Render verified native time samples with one MoonRay recorder")
    parser.add_argument("--reuse-bake", action="store_true", help="Reuse current settled bakes")
    parser.add_argument("--bake-only", action="store_true", help="Validate geometry without rendering")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Resume validation; 100 verifies all bakes and renders only")
    parser.add_argument("--frame-range",
                        help="Bake one inclusive START:END range for parallel preparation")
    parser.add_argument("--prove-existing", action="store_true",
                        help="Serially prove prepared frames 0-99 without rebaking")
    args = parser.parse_args()
    SCENE = args.source_scene.resolve()
    SCRATCH = args.scratch.resolve()
    if not SCENE.is_file():
        parser.error(f"source scene does not exist: {SCENE}")
    if args.batch and not args.animation:
        parser.error("--batch requires --animation")
    if not 0 <= args.start_frame <= SETTLED_FRAME + 1:
        parser.error("--start-frame must be between 0 and 100")
    frame_range = None
    if args.frame_range:
        if not args.bake_only or not args.animation or args.start_frame or args.prove_existing:
            parser.error("--frame-range requires --animation --bake-only and no other range mode")
        try:
            first, last = (int(value) for value in args.frame_range.split(":", 1))
        except ValueError:
            parser.error("--frame-range must be START:END")
        if not 0 <= first <= last <= SETTLED_FRAME:
            parser.error("--frame-range must stay within 0:99")
        frame_range = (first, last)
    if args.prove_existing and (not args.bake_only or not args.animation
                                or args.start_frame or args.frame_range):
        parser.error("--prove-existing requires --animation --bake-only and no range/start mode")
    env = viewer_environment()
    settled = settled_pair(env, args.reuse_bake)
    if args.probe_frames:
        sparse_preview(env, SCRATCH / "collide-off-baked.usdc", settled)
        if not args.animation:
            return
    if args.animation:
        if frame_range:
            bake_range(env, *frame_range)
        elif args.prove_existing:
            prove_existing_timeline(SCRATCH / "collide-off-baked.usdc")
        else:
            timeline_validation(env, SCRATCH / "collide-off-baked.usdc", args.start_frame)
    if args.bake_only:
        return
    target = SCRATCH / "collide-preview.png" if args.preview else OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    render_refined(frozen_render_stage(settled, SETTLED_FRAME), "/World/CamMotion", target,
                   600 if args.preview else 1400, pure_render_environment(env))
    if args.animation:
        animation(env, 800, batch=args.batch)
    print(f"still: {target}", flush=True)


if __name__ == "__main__":
    main()
