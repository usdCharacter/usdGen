# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
"""Assemble and render only an independently accepted Collide cut sequence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

FRAME_COUNT = 100
FPS = 24
PREVIEW_WIDTH = 480
FINAL_WIDTH = 800
FUR_PATH = "/World/Groom/Baked_Fur"
COLLIDER_PATH = "/World/Shield"
CAMERA_PATH = "/World/CamMotion"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_helper(path: Path):
    """Import as a module so helper functions see the same mutable globals."""
    spec = importlib.util.spec_from_file_location("usdgen_collide_render_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Collide render helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def accepted_identity(acceptance: dict[str, Any], name: str) -> str | None:
    nested = acceptance.get("freezeIdentities", {})
    if isinstance(nested, dict) and isinstance(nested.get(name), str):
        return nested[name]
    aliases = {
        "core": ("coreSha256", "runtime", "solver_sha256"),
        "imaging": ("imagingSha256", "imaging_sha256"),
        "baker": ("bakerSha256", "baker_sha256"),
        "usd": ("usdSha256", "usd_sha256"),
        "pxosd": ("pxosdSha256", "pxosd_sha256"),
        "schema": ("schemaSha256", "schema_sha256"),
        "loadedSchema": ("loadedSchemaSha256", "buildSchemaSha256"),
        "schemaDll": ("schemaDllSha256",),
        "helper": ("helperSha256", "helper", "helper_sha256"),
        "source": ("sourceSha256", "scene", "scene_sha256"),
    }
    for key in aliases[name]:
        value = acceptance.get(key)
        if isinstance(value, str):
            return value
    return None


def require_freeze(acceptance: dict[str, Any], paths: dict[str, Path]) -> None:
    missing = [name for name in paths if not accepted_identity(acceptance, name)]
    if missing:
        raise RuntimeError("Acceptance lacks frozen identities: " + ", ".join(missing))
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Frozen {name} input is missing: {path}")
        expected = accepted_identity(acceptance, name)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen {name} identity changed: expected {expected}, got {actual}: {path}")


def require_geometry_acceptance(acceptance: dict[str, Any]) -> None:
    if acceptance.get("allGeometryMetadataFramesPassed") is not True:
        raise RuntimeError("100-frame cut geometry and metadata proof is required")
    if acceptance.get("frameCount", FRAME_COUNT) != FRAME_COUNT:
        raise RuntimeError("Geometry proof must cover exactly 100 frames")


def temporal_proof_binding(acceptance: dict[str, Any], base: Path) -> bool:
    proof = acceptance.get("temporalProof")
    if isinstance(proof, dict):
        path_value, expected = proof.get("path"), proof.get("sha256")
    else:
        path_value = acceptance.get("temporalProofPath")
        expected = acceptance.get("temporalProofSha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        return False
    path = Path(path_value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.is_file() and sha256(path) == expected


def render_is_accepted(acceptance: dict[str, Any], acceptance_path: Path) -> bool:
    temporal = acceptance.get("temporalAcceptance") == "accepted"
    render = (acceptance.get("acceptedForRendering") is True or
              acceptance.get("renderAcceptance") == "accepted")
    bake_bound = isinstance(
        acceptance.get("bakeSha256", acceptance.get("bakeSha256ByFrame")),
        (list, dict))
    proof_bound = temporal_proof_binding(acceptance, acceptance_path.parent)
    return temporal and render and bake_bound and proof_bound


def accepted_bakes(scratch: Path, acceptance: dict[str, Any]) -> list[Path]:
    accepted_hashes = acceptance.get(
        "bakeSha256", acceptance.get("bakeSha256ByFrame"))
    proof_hashes = acceptance.get(
        "frameProofSha256", acceptance.get("frameProofSha256ByFrame"))
    if not isinstance(accepted_hashes, (list, dict)):
        raise RuntimeError("Acceptance must bind all 100 bake hashes")
    if not isinstance(proof_hashes, (list, dict)):
        raise RuntimeError("Acceptance must bind all 100 frame-proof hashes")
    expected_keys = {str(frame) for frame in range(FRAME_COUNT)}
    if isinstance(accepted_hashes, list) and len(accepted_hashes) != FRAME_COUNT:
        raise RuntimeError("Acceptance must contain exactly 100 bake hashes")
    if isinstance(accepted_hashes, dict) and set(accepted_hashes) != expected_keys:
        raise RuntimeError("Acceptance bake hash keys must be exactly 0 through 99")
    if isinstance(proof_hashes, list) and len(proof_hashes) != FRAME_COUNT:
        raise RuntimeError("Acceptance must contain exactly 100 frame-proof hashes")
    if isinstance(proof_hashes, dict) and set(proof_hashes) != expected_keys:
        raise RuntimeError("Acceptance frame-proof hash keys must be exactly 0 through 99")
    bakes: list[Path] = []
    for frame in range(FRAME_COUNT):
        bake = scratch / f"collide-motion-{frame:02d}.usdc"
        proof_path = scratch / f"cut-frame-{frame:02d}-proof.json"
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        actual = sha256(bake)
        proof_actual = sha256(proof_path)
        if proof.get("frame") != frame or proof.get("bakeSha256") != actual:
            raise RuntimeError(f"Unaccepted bake/proof at frame {frame}")
        if isinstance(accepted_hashes, list):
            if accepted_hashes[frame] != actual:
                raise RuntimeError(f"Acceptance bake hash differs at frame {frame}")
        elif isinstance(accepted_hashes, dict):
            if accepted_hashes.get(str(frame)) != actual:
                raise RuntimeError(f"Acceptance bake hash differs at frame {frame}")
        if isinstance(proof_hashes, list):
            if proof_hashes[frame] != proof_actual:
                raise RuntimeError(f"Acceptance frame-proof hash differs at frame {frame}")
        elif proof_hashes.get(str(frame)) != proof_actual:
            raise RuntimeError(f"Acceptance frame-proof hash differs at frame {frame}")
        bakes.append(bake)
    return bakes


def numeric_array(value: Any, np_module):
    try:
        array = np_module.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in "biufc":
        return None
    return np_module.ascontiguousarray(array)


def values_equal(left: Any, right: Any, np_module) -> bool:
    left_array = numeric_array(left, np_module)
    right_array = numeric_array(right, np_module)
    if left_array is not None or right_array is not None:
        if left_array is None or right_array is None:
            return False
        return (type(left) is type(right) and
                left_array.dtype == right_array.dtype and
                left_array.shape == right_array.shape and
                left_array.tobytes(order="C") == right_array.tobytes(order="C"))
    try:
        return type(left) is type(right) and bool(left == right)
    except (TypeError, ValueError):
        return False


def read_attributes(prim, time_code, Usd) -> dict[str, tuple[Any, Any, bool]]:
    values = {}
    for attribute in prim.GetAttributes():
        value = attribute.Get(Usd.TimeCode(time_code))
        if value is None:
            value = attribute.Get()
        values[attribute.GetName()] = (
            attribute.GetTypeName(), value, attribute.IsCustom())
    return values


def collect_values(bakes: list[Path], helper):
    fur_frames, collider_frames, collider_world = [], [], []
    source = helper.Usd.Stage.Open(str(helper.SCENE))
    for frame, bake in enumerate(bakes):
        stage = helper.Usd.Stage.Open(str(bake))
        fur = stage.GetPrimAtPath(FUR_PATH)
        collider = stage.GetPrimAtPath(COLLIDER_PATH)
        if not fur or not collider:
            raise RuntimeError(f"Accepted bake lacks required prims at frame {frame}")
        fur_frames.append(read_attributes(fur, frame, helper.Usd))
        collider_frames.append(read_attributes(collider, frame, helper.Usd))
        baked_world = helper.world_mesh_points(stage, COLLIDER_PATH, frame)
        source_world = helper.world_mesh_points(source, COLLIDER_PATH, frame)
        if (not helper.np.array_equal(baked_world, source_world) and
                helper.np.max(helper.np.abs(baked_world - source_world)) > 1e-6):
            raise RuntimeError(
                f"Accepted bake has the wrong authored collider pose at frame {frame}")
        collider_world.append(baked_world)
    return fur_frames, collider_frames, collider_world


def validate_layout(frames, label: str) -> list[str]:
    names = sorted(frames[0])
    expected = set(names)
    for frame, values in enumerate(frames):
        if set(values) != expected:
            raise RuntimeError(f"{label} attribute layout differs at frame {frame}")
        for name in names:
            if values[name][0] != frames[0][name][0]:
                raise RuntimeError(f"{label}.{name} type differs at frame {frame}")
    return names


def author_changing(stage, prim_path: str, frames, Usd, np_module) -> list[str]:
    prim = stage.OverridePrim(prim_path)
    changing = []
    for name in validate_layout(frames, prim_path):
        first = frames[0][name][1]
        if all(values_equal(frame[name][1], first, np_module)
               for frame in frames[1:]):
            continue
        changing.append(name)
        type_name, _, custom = frames[0][name]
        attribute = prim.CreateAttribute(name, type_name, custom=custom)
        for frame, values in enumerate(frames):
            if not attribute.Set(values[name][1], Usd.TimeCode(frame)):
                raise RuntimeError(f"Cannot author {prim_path}.{name} at frame {frame}")
    return changing


def verify_values(stage, prim_path: str, frames, Usd, np_module) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    for frame, expected in enumerate(frames):
        for name, (_, value, _) in expected.items():
            actual = prim.GetAttribute(name).Get(Usd.TimeCode(frame))
            if not values_equal(actual, value, np_module):
                raise RuntimeError(
                    f"Sampled stage differs from {prim_path}.{name} at frame {frame}")


def assemble_sampled_stage(scratch: Path, bakes: list[Path], helper):
    fur_frames, collider_frames, collider_world = collect_values(bakes, helper)
    target = scratch / "collide-cut-time-sampled.usdc"
    temporary = scratch / "collide-cut-time-sampled-next.usdc"
    temporary.unlink(missing_ok=True)
    stage = helper.Usd.Stage.CreateNew(str(temporary))
    stage.GetRootLayer().subLayerPaths.append(
        os.path.relpath(bakes[0], scratch).replace("\\", "/"))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(FRAME_COUNT - 1)
    stage.SetTimeCodesPerSecond(FPS)
    stage.SetFramesPerSecond(FPS)
    changing_fur = author_changing(
        stage, FUR_PATH, fur_frames, helper.Usd, helper.np)
    changing_collider = author_changing(
        stage, COLLIDER_PATH, collider_frames, helper.Usd, helper.np)
    if not stage.GetRootLayer().Save():
        raise RuntimeError("Cannot save sampled cut stage")

    check = helper.Usd.Stage.Open(str(temporary))
    helper.assert_subdivision_authoring(check)
    verify_values(check, FUR_PATH, fur_frames, helper.Usd, helper.np)
    verify_values(check, COLLIDER_PATH, collider_frames, helper.Usd, helper.np)
    pose_hashes = []
    for frame, expected in enumerate(collider_world):
        actual = helper.world_mesh_points(check, COLLIDER_PATH, frame)
        if not helper.np.array_equal(actual, expected):
            raise RuntimeError(f"Sampled collider pose differs at frame {frame}")
        pose_hashes.append(hashlib.sha256(
            helper.np.asarray(actual, dtype=helper.np.float64).tobytes()).hexdigest())
    check = None
    temporary.replace(target)
    return target, {
        "sampledStageSha256": sha256(target),
        "changingFurAttributes": changing_fur,
        "changingColliderAttributes": changing_collider,
        "colliderWorldPoseSha256": pose_hashes,
    }


def recorder_path(env: dict[str, str]) -> Path:
    name = shutil.which("usdrecord", path=env.get("PATH"))
    if not name:
        raise FileNotFoundError("usdrecord is unavailable")
    recorder = Path(name)
    if recorder.suffix.lower() == ".cmd" and recorder.with_suffix("").exists():
        recorder = recorder.with_suffix("")
    return recorder


def record_batch(root: Path, target: Path, scratch: Path, env: dict[str, str],
                 frames: str, width: int, label: str) -> list[Path]:
    pattern = scratch / f"collide-cut-{label}-{width}-###.###.png"
    log = scratch / f"collide-cut-{label}-{width}.render.log"
    command = [sys.executable, str(recorder_path(env)), "--renderer",
               "HdMoonrayRendererDebugPlugin", "--imageWidth", str(width),
               "--complexity", "veryhigh", "--disableCameraLight",
               "--camera", CAMERA_PATH, "--frames", frames,
               str(target), str(pattern)]
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"MoonRay batch render failed: {log}")
    selected = (range(FRAME_COUNT) if frames == "0:99"
                else [int(value) for value in frames.split(",")])
    images = [scratch / f"collide-cut-{label}-{width}-{frame:03d}.000.png"
              for frame in selected]
    if any(not image.is_file() for image in images):
        raise RuntimeError(f"MoonRay batch omitted frames for {label}")
    return images


def image_distance(left: Path, right: Path, helper) -> float:
    with helper.Image.open(left) as a, helper.Image.open(right) as b:
        x = helper.np.asarray(a.convert("RGB"), dtype=helper.np.int16)
        y = helper.np.asarray(b.convert("RGB"), dtype=helper.np.int16)
    return float(helper.np.abs(x - y).mean())


def independent_gate(bakes, batch, frames, width, scratch, env, helper, label):
    independent = {}
    for frame in frames:
        output = scratch / f"collide-cut-{label}-independent-{frame:03d}-{width}.png"
        helper.render_refined(helper.frozen_render_stage(bakes[frame], frame),
                              CAMERA_PATH, output, width, env)
        independent[frame] = output
    metrics = {}
    for index, frame in enumerate(frames):
        same = image_distance(batch[index], independent[frame], helper)
        wrong_frame = frames[(index + 1) % len(frames)]
        wrong = image_distance(batch[index], independent[wrong_frame], helper)
        metrics[f"same{frame}MeanAbsRgb"] = same
        metrics[f"wrong{frame}As{wrong_frame}MeanAbsRgb"] = wrong
        if same > 0.5 or wrong <= same + 0.02:
            raise RuntimeError(
                f"Independent {frame} gate failed: same={same}, wrong={wrong}")
    return metrics


def write_gif(images: list[Path], target: Path, helper) -> dict[str, Any]:
    # Encoding is independently reusable without importing the USD runtime.
    from encode_collide_gif import write_gif as encode_gif
    return encode_gif(images, target)


def render_pipeline(root, scratch, target, bakes, helper):
    env = helper.pure_render_environment(helper.viewer_environment())
    preview_frames = [0, 27, 99]
    preview_batch = record_batch(root, target, scratch, env, "0,27,99",
                                 PREVIEW_WIDTH, "preview")
    preview_metrics = independent_gate(
        bakes, preview_batch, preview_frames, PREVIEW_WIDTH, scratch, env,
        helper, "preview")
    final_batch = record_batch(root, target, scratch, env, "0:99",
                               FINAL_WIDTH, "final")
    if len({sha256(path) for path in final_batch}) != FRAME_COUNT:
        raise RuntimeError("Expected 100 distinct final rendered poses")
    final_metrics = independent_gate(
        bakes, [final_batch[99], final_batch[98]], [99, 98], FINAL_WIDTH,
        scratch, env, helper, "final")
    gif = write_gif(final_batch, scratch / "collide-cut-final-800.gif", helper)
    return {"previewWidth": PREVIEW_WIDTH, "previewFrames": preview_frames,
            "previewIndependent": preview_metrics, "finalWidth": FINAL_WIDTH,
            "finalFrames": FRAME_COUNT, "finalUniqueFrames": FRAME_COUNT,
            "finalIndependent99": final_metrics, **gif}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    root = Path(__file__).resolve().parents[2]
    scratch = args.scratch.resolve()
    source_scene = args.source_scene.resolve()
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    require_geometry_acceptance(acceptance)
    os.environ["USDGEN_BUILD"] = str(root / "build")
    os.environ["USDGEN_DOCS_BUILD"] = str(root / "renders/docs/baker-build")
    helper_path = root / "tools/render_collide_diagnostic.py"
    helper = load_helper(helper_path)
    helper.SCENE = source_scene
    helper.SCRATCH = scratch
    runtime = helper.runtime_paths()
    freeze_paths = {
        "core": runtime["solver_sha256"],
        "imaging": runtime["imaging_sha256"],
        "baker": runtime["baker_sha256"],
        "usd": runtime["usd_sha256"],
        "pxosd": runtime["pxosd_sha256"],
        "schema": root / "libs/hosts/usdGenUsd/schema/resources/generatedSchema.usda",
        "loadedSchema": root / "build/usd/usdGenSchema/resources/generatedSchema.usda",
        "helper": helper_path,
        "source": source_scene,
    }
    if accepted_identity(acceptance, "schemaDll") is not None:
        freeze_paths["schemaDll"] = root / "build/usdGenSchema.dll"
    require_freeze(acceptance, freeze_paths)
    bakes = accepted_bakes(scratch, acceptance)
    target, qa = assemble_sampled_stage(scratch, bakes, helper)
    eligible = render_is_accepted(acceptance, args.acceptance.resolve())
    qa.update({"renderEligible": eligible,
               "temporalAcceptance": acceptance.get("temporalAcceptance"),
               "freezeIdentities": {name: sha256(path)
                                    for name, path in freeze_paths.items()}})
    if args.prepare_only:
        qa.update({"preparedOnly": True, "renderAttempted": False})
        (scratch / "collide-cut-prepare-qa.json").write_text(
            json.dumps(qa, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if not eligible:
        raise RuntimeError(
            "Rendering requires accepted temporal proof and explicit render acceptance")
    qa.update(render_pipeline(root, scratch, target, bakes, helper))
    qa.update({"preparedOnly": False, "renderAttempted": True})
    (scratch / "collide-cut-render-qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(f"rendered {FRAME_COUNT} accepted frames and {qa['gifDurationMs']}ms GIF")


if __name__ == "__main__":
    main()
