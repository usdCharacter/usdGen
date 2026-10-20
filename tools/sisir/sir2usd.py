#!/usr/bin/env python3
# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
"""sir2usd -- convert a Sisir groom scene (.sir archive or legacy .json) into a usdGen USD stage.

Usage (Python 3.10 + OpenUSD 26.08; always pass -S, see README.md):

    python -S sir2usd.py SCENE.sir OUT.usda [--sets 0 "Set 1" ...] [--no-extract]
                         [--active-ops-only] [--cards-module PATH] [--quiet]
    python -S sir2usd.py --preset PRESET.json --description /World/Groom/Set_1 STAGE.usda [--output OUT.usda]

What is authored (metres, Y up, metersPerUnit = 1):

* the scalp mesh: OBJ re-indexed to unique (v, vt, vn) corners in first-occurrence face order (the
  reference tool's vertex numbering, so its per-vertex masks map 1:1), OBJ face order kept,
  vertex ``st`` and normals, ``UsdGenRestAPI``;
* per hair set: a ``UsdGenGuideSet`` holding the guides of the layers the set uses (linear
  BasisCurves with ``skinprim`` / ``skinprimuv`` / ``usdGen:plane:mirrorSign``), one
  ``UsdGenDescription`` with the full operator stack (bottom sibling runs first), mask maps
  (``UsdGenPaintMap`` over faceVarying mesh primvars, ``UsdGenImageMap`` for image masks), trim
  curves, the hair material binding, and the raw settings in ``customData["sisir"]``;
* card spines and tile strand libraries through ``cards.py`` for card-grown sets;
* ``UsdGenHairStrands`` materials mapped from the scene's hair render materials.

Every value conversion is listed in README.md.  Stdlib + pxr only.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import io
import json
import math
import os
import shutil
import struct
import sys
import warnings
import zipfile

try:
    from pxr import Gf, Plug, Sdf, Tf, Usd, UsdGeom, UsdShade, Vt
except ImportError as _e:  # pragma: no cover - environment problem
    raise SystemExit("sir2usd: cannot import pxr (%s). Run Python with -S and PYTHONPATH set to the "
                     "OpenUSD site-packages; see tools/sisir/README.md." % _e)

__version__ = "1.0"

HERE = os.path.dirname(os.path.abspath(__file__))
INT_MIN = -(1 << 31)
GUIDE_DISPLAY_WIDTH = 0.0002   # metres, display only

# ============================================================================= format constants

ZIP_MAGIC = b"PK\x03\x04"
MAX_SCENE_JSON_BYTES = 1 << 31
MAX_ENTRY_BYTES = 1 << 32

DISTRIBUTION_NAMES = {0: "Uniform", 1: "Vertex", 2: "Even", 3: "Guides", 4: "From Cards"}
DISTRIBUTION_TOKENS = {0: "uvGrid", 1: "vertex", 2: "even", 3: "perGuide"}
WEIGHTING_TOKENS = {0: "nearest", 1: "inverseDistance", 2: "gaussian"}
CLUMP_CENTER_TOKENS = {1: "guides", 2: "count", 3: "guideSeeded"}

# HairSettings constructor defaults = value of a key absent from scene.json / a preset.
HAIR_SETTINGS_DEFAULTS = {
    "hairDensity": 20000.0, "distribution": 2, "resolutionType": 0, "hairResolution": 40,
    "resolutionDistance": 1.0, "showSteps": False, "smoothness": 0.0, "guideInterpolation": 0,
    "guideInterpolationTightness": 2.0, "guideProximityDistance": 1.0, "guideProximityMode": 0,
    "guideProximityRadius": 5.0, "guideBendFollow": False, "clumpType": 0, "clumpFlyawayPercentage": 0.0,
    "clumpFlyawayAmount": 0.0, "clumpRootTightness": 0.0, "clumpAmount": 1.0, "clumpToHairTies": False,
    "tiesClumpAmount": 1.0, "tiesClumpSpread": 0.25, "clumpMidTightness": 0.5, "clumpMidShift": 0.5,
    "clumpTipTightness": 1.0, "clumpRandomCount": 20, "subClumpsEnabled": False, "subClumpCount": 2,
    "subClumpTightness": 0.5, "subClumpAmount": 1.0, "generateSubClumpIds": False, "braidFrequency": 8.0,
    "braidWidth": 1.0, "braidFlatness": 0.55, "braidKnotTightness": 0.5, "braidFadeIn": 0.1,
    "braidFadeOut": 0.1, "braidRootSpread": 0.0, "braidTipSpread": 0.0, "braidTailLength": 0.0,
    "braidTailRelax": 0.5, "braidTailCinch": 0.6, "partingAngle": 80.0, "partingAmount": 0.0,
    "partingRadius": 50.0, "partingDistance": 1.0, "pushOutEnabled": False, "pushOutDebug": False,
    "pushOutRoot": 0.0, "pushOutTip": 1.0, "pushOutSmoothness": 0.5, "lengthMin": 0.01, "lengthMax": 1.0,
    "lengthRandomize": 0.0, "lengthUseClumpIds": False, "lengthUseInterpolation": True, "lengthBreakup": 0.0,
    "lengthBreakupSize": 1.0, "textureDepth": 0.0, "textureTileBorder": 0, "cardSourceUvId": -1,
    "cardSourceTile": -1, "cardSourceLayer": -1, "cardSourceBraid": False, "cardRootFlatten": 0.0,
    "cardSmooth": True, "cardStrandAmount": 1.0, "trimAngle": 0.0, "trimInvertAngle": False,
    "trimDistance": 0.1, "trimDebugPlane": False, "frizzAmount": 0.0, "frizzScale": 5.0, "frizzScaleZ": 5.0,
    "frizzNoise": 0.0, "frizzNoiseSize": 10.0, "frizzNoiseSharpness": 0.0, "frizzRoot": 0.0, "frizzMid": 0.5,
    "frizzMidShift": 0.5, "frizzTip": 1.0, "frizzUseClumpIds": False, "frizzPreserveBraidKnots": False,
    "twistAmount": 1.0, "twistRoot": 0.0, "twistMid": 0.0, "twistMidShift": 0.5, "twistTip": 0.0,
    "twistNoise": 0.0, "twistUseClumpIds": False, "hairThickness": 0.005, "thicknessRoot": 0.2,
    "thicknessMid": 1.0, "thicknessMidShift": 0.5, "thicknessTip": 0.02, "maskBrushRadiusPx": 64.0,
    "maskBrushIntensity": 1.0, "showPaintVertexColors": True, "distributionMaskPath": "",
    "lengthMaskPath": "",
}

# Load-time clamps of the scene reader (hairsettings_offsets.md); stage-time clamps belong to the engine.
_CLAMPS = {
    "hairResolution": (3, 1000), "resolutionDistance": (0.01, 10.0), "guideProximityRadius": (0.01, 50.0),
    "tiesClumpAmount": (0.0, 1.0), "tiesClumpSpread": (0.0, 1.0), "subClumpCount": (1, None),
    "subClumpTightness": (0.0, 1.0), "subClumpAmount": (0.1, 1.0), "braidFrequency": (0.0, 1000.0),
    "braidWidth": (0.0, 5.0), "braidFlatness": (0.0, 1.0), "braidKnotTightness": (0.0, 1.0),
    "braidFadeIn": (0.0, 1.0), "braidFadeOut": (0.0, 1.0), "braidRootSpread": (0.0, 2.0),
    "braidTipSpread": (0.0, 2.0), "braidTailLength": (0.0, 0.9), "braidTailRelax": (0.0, 1.0),
    "braidTailCinch": (0.0, 1.0), "partingAngle": (0.0, 89.0), "partingAmount": (0.0, 1.0),
    "partingRadius": (0.0, 100.0), "partingDistance": (0.0, 1.0), "pushOutRoot": (0.0, 5.0),
    "pushOutTip": (0.0, 5.0), "pushOutSmoothness": (0.0, 1.0), "lengthBreakup": (0.0, 1.0),
    "lengthBreakupSize": (0.05, 10.0), "textureDepth": (-1.0, 1.0), "textureTileBorder": (0, 1),
    "cardSourceUvId": (-1, 9), "cardSourceTile": (-1, None), "cardSourceLayer": (-2, None),
    "cardRootFlatten": (0.0, 1.0), "cardStrandAmount": (0.0, 1.0), "trimAngle": (0.0, 89.0),
    "trimDistance": (0.01, 2.0), "frizzAmount": (0.0, 100.0), "frizzScale": (0.0, 100.0),
    "frizzScaleZ": (0.0, 100.0), "frizzNoise": (0.0, 100.0), "frizzNoiseSize": (0.01, 20.0),
    "frizzNoiseSharpness": (0.0, 1.0), "frizzRoot": (0.0, 1.0), "frizzMid": (0.0, 1.0),
    "frizzMidShift": (0.0, 1.0), "frizzTip": (0.0, 1.0), "twistAmount": (0.0, 200.0),
    "twistRoot": (-180.0, 180.0), "twistMid": (-180.0, 180.0), "twistMidShift": (0.1, 0.9),
    "twistTip": (-180.0, 180.0), "twistNoise": (0.0, 1.0), "hairThickness": (0.001, None),
    "thicknessRoot": (0.01, 1.0), "thicknessMid": (0.01, 1.0), "thicknessMidShift": (0.1, 0.9),
    "thicknessTip": (0.01, 1.0), "maskBrushRadiusPx": (1.0, 512.0), "maskBrushIntensity": (0.0, 1.0),
    "hairDensity": (0.0, None),
}
_ENUMS = {"distribution": (0, 4), "resolutionType": (0, 1), "guideInterpolation": (0, 2),
          "guideProximityMode": (0, 1), "clumpType": (0, 4)}

# Hair-preset key classes: 0 always applied, 1 only when the preset module equals the current module,
# 2 never stored / applied.  Unknown keys are class 1.
PRESET_CLASS2 = {"showSteps", "pushOutDebug", "cardSourceUvId", "cardSourceTile", "cardSourceLayer",
                 "cardSourceBraid", "cardRootFlatten", "cardSmooth", "cardStrandAmount", "trimAngle",
                 "trimInvertAngle", "trimDistance", "trimDebugPlane", "maskBrushRadiusPx", "maskBrushIntensity",
                 "showPaintVertexColors", "distributionMaskPath", "lengthMaskPath"}
PRESET_CLASS1 = {"hairDensity", "guideProximityDistance", "guideProximityMode", "guideProximityRadius",
                 "clumpToHairTies", "tiesClumpAmount", "tiesClumpSpread", "partingAmount", "partingAngle",
                 "partingRadius", "partingDistance", "pushOutEnabled", "pushOutRoot", "pushOutTip",
                 "pushOutSmoothness", "lengthBreakup", "lengthBreakupSize", "textureDepth", "textureTileBorder",
                 "frizzNoiseSize"}
MODULE_IDS = {"hair": 1, "texture": 2}

# Settings with no effect on a scalp groom (UI state, debug, texture-module-only stages).
INERT_KEYS = {"showSteps", "pushOutDebug", "trimDebugPlane", "maskBrushRadiusPx", "maskBrushIntensity",
              "showPaintVertexColors", "pushOutSmoothness", "lengthBreakup", "lengthBreakupSize",
              "textureDepth", "textureTileBorder"}


class SirFormatError(Exception):
    pass


def _num(v, default):
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else float(default)


def _clamp(v, lo, hi):
    if lo is not None and v < lo:
        return lo
    if hi is not None and v > hi:
        return hi
    return v


def _clamp01(v):
    return _clamp(v, 0.0, 1.0)


def _lround(x):
    """C lroundf: half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


# ============================================================================= settings / presets

def resolve_hair_settings(raw) -> dict:
    """hairSettings JSON -> all 99 keys with absent-key defaults and legacy keys (no clamps)."""
    raw = raw if isinstance(raw, dict) else {}
    out = dict(HAIR_SETTINGS_DEFAULTS)
    for k, v in raw.items():
        if k in out:
            out[k] = v
    if "hairDensity" not in raw and isinstance(raw.get("hairCount"), (int, float)):
        out["hairDensity"] = max(0.0, float(raw["hairCount"]))
    if "clumpAmount" not in raw and isinstance(raw.get("clumpTightness"), (int, float)):
        out["clumpAmount"] = float(raw["clumpTightness"])
    if "hairThickness" not in raw and any(k in raw for k in ("rootThickness", "midThickness", "tipThickness")):
        mid = _num(raw.get("midThickness"), out["hairThickness"])
        root = _num(raw.get("rootThickness"), out["thicknessRoot"])
        tip = _num(raw.get("tipThickness"), out["thicknessTip"])
        if mid > 1e-8:
            out["hairThickness"] = mid
        m = mid if mid > 1e-8 else 1.0
        out["thicknessRoot"] = _clamp01(root / m)
        out["thicknessTip"] = _clamp01(tip / m)
        out["thicknessMid"], out["thicknessMidShift"] = 1.0, 0.5
    return out


def clamp_hair_settings(st: dict) -> dict:
    """Apply the scene reader's load-time clamps / enum checks / types (the in-memory values)."""
    out = dict(st)
    for k, default in HAIR_SETTINGS_DEFAULTS.items():
        v = out.get(k, default)
        if isinstance(default, bool):
            out[k] = bool(v) if isinstance(v, (bool, int, float)) else default
        elif isinstance(default, int):
            iv = int(_num(v, default))
            if k in _ENUMS:
                lo, hi = _ENUMS[k]
                iv = iv if lo <= iv <= hi else default
            elif k in _CLAMPS:
                lo, hi = _CLAMPS[k]
                iv = int(_clamp(iv, lo, hi))
            out[k] = iv
        elif isinstance(default, float):
            fv = _num(v, default)
            if k in _CLAMPS:
                lo, hi = _CLAMPS[k]
                fv = float(_clamp(fv, lo, hi))
            out[k] = fv
        else:
            out[k] = v if isinstance(v, str) else default
    return out


def settings_for(raw) -> dict:
    return clamp_hair_settings(resolve_hair_settings(raw))


def read_hair_preset(path: str) -> dict:
    """{"version": 2, "module": "hair"|"texture", "settings": {...}}."""
    with open(path, "rb") as f:
        data = json.loads(f.read().decode("utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("settings"), dict):
        raise SirFormatError("%s: not a hair preset (expected {'version','module','settings'})" % path)
    return {"name": os.path.splitext(os.path.basename(path))[0], "version": data.get("version"),
            "module": MODULE_IDS.get(data.get("module"), 0), "settings": data["settings"]}


def apply_hair_preset(current_raw: dict, preset: dict, current_module: int = 1):
    """Preset key filtering of the reference tool.  Returns (new full settings dict, applied, skipped)."""
    cur = resolve_hair_settings(current_raw)
    out = dict(cur)
    same = preset["module"] != 0 and preset["module"] == current_module
    applied, skipped = [], []
    for k, v in preset["settings"].items():
        cls = 2 if k in PRESET_CLASS2 else (1 if (k in PRESET_CLASS1 or k not in HAIR_SETTINGS_DEFAULTS) else 0)
        if (cls == 0 or (cls == 1 and same)) and k in HAIR_SETTINGS_DEFAULTS:
            out[k] = v
            applied.append(k)
        else:
            skipped.append(k)
    if current_module == 2 and out.get("distribution") == 1:
        out["distribution"] = cur.get("distribution")
    if (out.get("distribution") == 4) != (cur.get("distribution") == 4):
        out["distribution"] = cur.get("distribution")
        skipped.append("distribution(from-cards switch refused)")
    return out, applied, skipped


# ============================================================================= container

def is_unsafe_entry(name: str):
    """Reference rules (reject '..' anywhere, leading slash/backslash) plus drive/colon/NUL."""
    if ".." in name:
        return "contains '..'"
    if name.startswith("/") or name.startswith("\\"):
        return "absolute path"
    if ":" in name:
        return "drive letter / colon"
    if "\x00" in name:
        return "NUL byte"
    return None


class SisirScene:
    """A parsed scene.  ``scene`` is the raw scene.json dict; ``archive`` the open ZIP (or None)."""

    def __init__(self, scene, source, archive, entries, rejected):
        self.scene = scene if isinstance(scene, dict) else {}
        self.source = source
        self.archive = archive
        self.entries = entries
        self.rejected = rejected

    @classmethod
    def open(cls, path: str) -> "SisirScene":
        with open(path, "rb") as f:
            head = f.read(4)
        if head == ZIP_MAGIC:
            try:
                zf = zipfile.ZipFile(path, "r")
            except zipfile.BadZipFile as e:
                raise SirFormatError("not a readable ZIP: %s" % e)
            entries, rejected = {}, []
            for info in zf.infolist():
                name = info.filename
                reason = is_unsafe_entry(name)
                if reason:
                    rejected.append((name, reason))
                    continue
                if name.endswith("/") or info.is_dir():
                    continue
                if info.file_size > MAX_ENTRY_BYTES:
                    rejected.append((name, "too large"))
                    continue
                entries[name] = info
            info = entries.get("scene.json")
            if info is None:
                raise SirFormatError("scene.json not found in archive")
            if info.file_size > MAX_SCENE_JSON_BYTES:
                raise SirFormatError("scene.json too large")
            return cls(cls._parse(cls._read(zf, info, MAX_SCENE_JSON_BYTES)), path, zf, entries, rejected)
        if os.path.getsize(path) > MAX_SCENE_JSON_BYTES:
            raise SirFormatError("scene file too large")
        with open(path, "rb") as f:
            return cls(cls._parse(f.read()), path, None, {}, [])

    @staticmethod
    def _read(zf, info, limit):
        out = io.BytesIO()
        with zf.open(info, "r") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                if out.tell() > limit:
                    raise SirFormatError("%s: decompressed size exceeds limit" % info.filename)
        return out.getvalue()

    @staticmethod
    def _parse(raw):
        try:
            data = json.loads(raw.decode("utf-8-sig", errors="replace"))
        except json.JSONDecodeError as e:
            raise SirFormatError("failed to parse scene.json: %s" % e)
        if not isinstance(data, dict):
            raise SirFormatError("scene.json root is not an object")
        return data

    def read_asset(self, rel_path):
        """Bytes of an embedded asset ('mesh/x.obj'), or of a legacy scene's file path."""
        if not rel_path:
            return None
        if self.archive is not None:
            info = self.entries.get(rel_path.replace("\\", "/"))
            return self._read(self.archive, info, MAX_ENTRY_BYTES) if info is not None else None
        p = self._legacy_path(rel_path)
        if p:
            with open(p, "rb") as f:
                return f.read()
        return None

    def _legacy_path(self, value):
        if not value:
            return None
        cands = [value] if os.path.isabs(value) else [os.path.join(os.path.dirname(os.path.abspath(self.source)), value)]
        for c in cands:
            if os.path.isfile(c):
                return os.path.abspath(c)
        return None

    def extract(self, dest_dir):
        """Extract every safe archive entry below dest_dir.  Returns {entry name: absolute path}."""
        out = {}
        if self.archive is None:
            return out
        root = os.path.abspath(dest_dir)
        for name, info in self.entries.items():
            target = os.path.abspath(os.path.join(root, *name.split("/")))
            if os.path.commonpath([root, target]) != root:
                warnings.warn("sir2usd: skipped entry escaping the asset folder: %r" % name)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with self.archive.open(info, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            out[name] = target
        return out

    # --- typed accessors -----------------------------------------------------------------------
    @property
    def mesh_import_scale(self):
        v = self.scene.get("meshImportScale", 0.01)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.01

    def list(self, key):
        v = self.scene.get(key)
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    def hair_sets(self):
        v = self.scene.get("hairSets")
        if isinstance(v, list) and v:
            return [h if isinstance(h, dict) else {} for h in v]
        hs = self.scene.get("hairSettings")
        if isinstance(hs, dict):
            return [{"name": "Set 0", "color": [0.9, 0.8, 0.65], "visible": True,
                     "renderMaterialId": 0, "settings": hs, "_legacy": True}]
        return []


def decode_vertex_mask(values):
    """Per-vertex mask (ints 0..255, legacy floats) -> floats in [0,1]; [] = no mask."""
    if not isinstance(values, list):
        return []
    out = []
    for v in values:
        if isinstance(v, bool):
            f = 1.0
        elif isinstance(v, int):
            f = v / 255.0
        elif isinstance(v, float):
            f = v
        else:
            f = 1.0
        out.append(min(1.0, max(0.0, f)))
    return out


# Path keys rewritten by the reference loader (sir_format.md §1.4); value = archive-relative in a .sir.
def _path_slots(scene):
    """Yield (container, key) for every asset path key of a scene dict (and its texture-module scene)."""
    def top(sc, nested):
        for k in ("meshPath", "blendshapePath", "refMeshPath", "meshTexturePath", "refMeshTexturePath"):
            if isinstance(sc.get(k), str):
                yield sc, k
        rm = sc.get("renderMaterial")
        if isinstance(rm, dict) and isinstance(rm.get("ColorMap"), str):
            yield rm, "ColorMap"
        for m in sc.get("renderMaterials") or []:
            if isinstance(m, dict) and isinstance(m.get("ColorMap"), str):
                yield m, "ColorMap"
        mm = sc.get("renderMeshMaterial")
        if isinstance(mm, dict):
            for k in ("albedoTexturePath", "roughnessTexturePath", "metalnessTexturePath", "cavityTexturePath",
                      "normalTexturePath", "detailNormalTexturePath"):
                if isinstance(mm.get(k), str):
                    yield mm, k
        rl = sc.get("renderLight")
        if isinstance(rl, dict) and isinstance(rl.get("iblPath"), str):
            yield rl, "iblPath"
        for u in sc.get("cardUvSets") or []:
            if isinstance(u, dict):
                for k in ("colorPath", "maskPath"):
                    if isinstance(u.get(k), str):
                        yield u, k
        for layer in sc.get("layers") or []:
            if isinstance(layer, dict):
                for k in ("cardColorTexturePath", "cardMaskTexturePath", "cardDistributionMask", "cardLengthMask"):
                    if isinstance(layer.get(k), str):
                        yield layer, k
        for m in sc.get("renderCardsMaterials") or []:
            if isinstance(m, dict):
                for k in ("colorMapOverridePath", "maskMapOverridePath", "tangentMapPath", "normalMapPath",
                          "depthMapPath", "aoMapPath", "roughnessMapPath", "idMapPath"):
                    if isinstance(m.get(k), str):
                        yield m, k
        hs_legacy = sc.get("hairSettings")
        if isinstance(hs_legacy, dict):
            for k in ("distributionMaskPath", "lengthMaskPath"):
                if isinstance(hs_legacy.get(k), str):
                    yield hs_legacy, k
        for h in sc.get("hairSets") or []:
            if isinstance(h, dict):
                if isinstance(h.get("generationDistributionMaskPath"), str):
                    yield h, "generationDistributionMaskPath"
                s = h.get("settings")
                if isinstance(s, dict):
                    for k in ("distributionMaskPath", "lengthMaskPath"):
                        if isinstance(s.get(k), str):
                            yield s, k
        if not nested:
            tm = sc.get("textureModule")
            if isinstance(tm, dict) and isinstance(tm.get("scene"), dict):
                yield from top(tm["scene"], True)
    yield from top(scene, False)


# ============================================================================= OBJ mesh

class ObjMesh:
    """OBJ re-indexed to unique (v, vt, vn) corners in first-occurrence face order."""

    def __init__(self):
        self.points = []          # scaled positions per re-indexed vertex
        self.uvs = []             # (u, v) per vertex (OBJ convention == USD st)
        self.normals = []         # per vertex (normalised)
        self.face_counts = []
        self.face_indices = []    # re-indexed vertex per face corner
        self.face_starts = []
        self.tri_face = []        # fan triangle t -> polygon index
        self.tri_k = []           # fan triangle t -> k (corners f0, fk, fk+1)
        self.has_uv = False
        self.has_vn = False
        self.n_positions = 0
        self.value_unique = 0

    @classmethod
    def parse(cls, data: bytes, scale: float) -> "ObjMesh":
        V, VT, VN = [], [], []
        faces = []
        for raw in data.splitlines():
            line = raw.strip()
            if not line or line[:1] == b"#":
                continue
            tag, _, rest = line.partition(b" ")
            if tag == b"v":
                p = rest.split()
                V.append((float(p[0]), float(p[1]), float(p[2])))
            elif tag == b"vt":
                p = rest.split()
                VT.append((float(p[0]), float(p[1]) if len(p) > 1 else 0.0))
            elif tag == b"vn":
                p = rest.split()
                VN.append((float(p[0]), float(p[1]), float(p[2])))
            elif tag == b"f":
                corners = []
                for tok in rest.split():
                    parts = tok.split(b"/")
                    vi = int(parts[0])
                    vi = vi - 1 if vi > 0 else len(V) + vi
                    ti = ni = -1
                    if len(parts) > 1 and parts[1]:
                        ti = int(parts[1])
                        ti = ti - 1 if ti > 0 else len(VT) + ti
                    if len(parts) > 2 and parts[2]:
                        ni = int(parts[2])
                        ni = ni - 1 if ni > 0 else len(VN) + ni
                    corners.append((vi, ti, ni))
                if len(corners) >= 3:
                    faces.append(corners)
        m = cls()
        m.n_positions = len(V)
        key_to_id = {}
        corner_keys = []
        for corners in faces:
            m.face_starts.append(len(m.face_indices))
            m.face_counts.append(len(corners))
            for key in corners:
                vid = key_to_id.get(key)
                if vid is None:
                    vid = len(corner_keys)
                    key_to_id[key] = vid
                    corner_keys.append(key)
                m.face_indices.append(vid)
        for fi, corners in enumerate(faces):
            for k in range(1, len(corners) - 1):
                m.tri_face.append(fi)
                m.tri_k.append(k)
        m.has_uv = bool(VT) and all(k[1] >= 0 for k in corner_keys)
        m.has_vn = bool(VN) and all(k[2] >= 0 for k in corner_keys)
        for vi, ti, ni in corner_keys:
            x, y, z = V[vi]
            m.points.append((x * scale, y * scale, z * scale))
            m.uvs.append(VT[ti] if (ti >= 0 and ti < len(VT)) else (0.0, 0.0))
        m.value_unique = len({(V[a], VT[b] if 0 <= b < len(VT) else None, VN[c] if 0 <= c < len(VN) else None)
                              for a, b, c in corner_keys})
        if m.has_vn:
            for _vi, _ti, ni in corner_keys:
                m.normals.append(_normalize(VN[ni]))
        else:
            # smooth normals per position (area weighted) for corners that share a position
            acc = [[0.0, 0.0, 0.0] for _ in V]
            for fi, s in enumerate(m.face_starts):
                c = m.face_counts[fi]
                ids = m.face_indices[s:s + c]
                p0 = m.points[ids[0]]
                for k in range(1, c - 1):
                    n = _cross(_sub(m.points[ids[k]], p0), _sub(m.points[ids[k + 1]], p0))
                    for vid in (ids[0], ids[k], ids[k + 1]):
                        a = acc[corner_keys[vid][0]]
                        a[0] += n[0]; a[1] += n[1]; a[2] += n[2]
            for vi, _ti, _ni in corner_keys:
                m.normals.append(_normalize(acc[vi]))
        return m

    def tri_corners(self, t):
        f = self.tri_face[t]
        k = self.tri_k[t]
        s = self.face_starts[f]
        fi = self.face_indices
        return f, k, (fi[s], fi[s + k], fi[s + k + 1])

    def face_points(self, f):
        s = self.face_starts[f]
        return [self.points[i] for i in self.face_indices[s:s + self.face_counts[f]]]

    def area(self):
        total = 0.0
        for t in range(len(self.tri_face)):
            _f, _k, (a, b, c) = self.tri_corners(t)
            pa = self.points[a]
            total += 0.5 * _len(_cross(_sub(self.points[b], pa), _sub(self.points[c], pa)))
        return total

    def bbox(self):
        xs = [p[0] for p in self.points]; ys = [p[1] for p in self.points]; zs = [p[2] for p in self.points]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _len(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    n = _len(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-20 else (0.0, 1.0, 0.0)


def polyline_length(pts):
    return sum(math.dist(pts[i], pts[i - 1]) for i in range(1, len(pts)))


# ============================================================================= surface bindings

def _clean_bary(bary):
    b = [_clamp01(_num(x, 0.0)) for x in (list(bary) + [0, 0, 0])[:3]]
    s = sum(b)
    return [x / s for x in b] if s > 1e-12 else [1.0, 0.0, 0.0]


def fan_point(mesh: ObjMesh, tri, bary):
    _f, _k, (a, b, c) = mesh.tri_corners(tri)
    pa, pb, pc = mesh.points[a], mesh.points[b], mesh.points[c]
    return tuple(bary[0] * pa[i] + bary[1] * pb[i] + bary[2] * pc[i] for i in range(3))


def fan_uv(mesh: ObjMesh, tri, bary):
    _f, _k, (a, b, c) = mesh.tri_corners(tri)
    ua, ub, uc = mesh.uvs[a], mesh.uvs[b], mesh.uvs[c]
    return (bary[0] * ua[0] + bary[1] * ub[0] + bary[2] * uc[0], bary[0] * ua[1] + bary[1] * ub[1] + bary[2] * uc[1])


def fan_normal(mesh: ObjMesh, tri, bary):
    _f, _k, (a, b, c) = mesh.tri_corners(tri)
    na, nb, nc = mesh.normals[a], mesh.normals[b], mesh.normals[c]
    return _normalize(tuple(bary[0] * na[i] + bary[1] * nb[i] + bary[2] * nc[i] for i in range(3)))


def _bilinear(p, u, v):
    a = (1 - u) * (1 - v); b = u * (1 - v); c = u * v; d = (1 - u) * v
    return tuple(p[0][i] * a + p[1][i] * b + p[2][i] * c + p[3][i] * d for i in range(3))


def _inverse_bilinear(p, target, u, v):
    """Gauss-Newton projection of target onto the bilinear patch p[0..3] (corners (0,0),(1,0),(1,1),(0,1))."""
    for _ in range(30):
        q = _bilinear(p, u, v)
        r = _sub(target, q)
        du = _add(_mul(_sub(p[1], p[0]), 1 - v), _mul(_sub(p[2], p[3]), v))
        dv = _add(_mul(_sub(p[3], p[0]), 1 - u), _mul(_sub(p[2], p[1]), u))
        a, b, c = _dot(du, du), _dot(du, dv), _dot(dv, dv)
        det = a * c - b * b
        if abs(det) < 1e-30:
            break
        ru, rv = _dot(du, r), _dot(dv, r)
        su = (c * ru - b * rv) / det
        sv = (a * rv - b * ru) / det
        nu, nv = _clamp01(u + su), _clamp01(v + sv)
        done = abs(nu - u) + abs(nv - v) < 1e-14
        u, v = nu, nv
        if done:
            break
    return u, v


def fan_to_face_param(mesh: ObjMesh, tri, bary):
    """(rootTri, rootBary) on the fan triangle (f0, fk, fk+1) -> (face, skinprimuv) in usdGen's face
    parametrisation: triangle barycentric (u, v), quad bilinear (u, v), n-gon u = fan + bary_u."""
    f, k, _ = mesh.tri_corners(tri)
    n = mesh.face_counts[f]
    b0, b1, b2 = bary
    if n == 3:
        u, v = b1, b2
        while _f32(u) + _f32(v) > 1.0:
            v = math.nextafter(_f32(v), -math.inf)
        return f, (u, v)
    if n == 4:
        u, v = ((b1 + b2, b2) if k == 1 else (b1, b1 + b2))
        u, v = _inverse_bilinear(mesh.face_points(f), fan_point(mesh, tri, bary), _clamp01(u), _clamp01(v))
        return f, (_clamp01(u), _clamp01(v))
    fan = k - 1
    u = min(b1, 1.0 - 1e-6)
    v = b2
    while _f32(u) + _f32(v) > 1.0:
        v = math.nextafter(_f32(v), -math.inf)
    return f, (fan + u, v)


def eval_face_param(mesh: ObjMesh, face, uv):
    """Position of (face, skinprimuv) with the engine's convention (surfaceRootFrames.cpp)."""
    pts = mesh.face_points(face)
    n = len(pts)
    u, v = uv
    if n == 3:
        return tuple(pts[0][i] * (1 - u - v) + pts[1][i] * u + pts[2][i] * v for i in range(3))
    if n == 4:
        return _bilinear(pts, u, v)
    fan = int(math.floor(u))
    uu = u - fan
    a, b, c = pts[0], pts[fan + 1], pts[fan + 2]
    return tuple(a[i] * (1 - uu - v) + b[i] * uu + c[i] * v for i in range(3))


def _closest_on_triangle(p, a, b, c):
    """Ericson 5.1.5: closest point to p on triangle abc -> (point, (wa, wb, wc))."""
    ab, ac, ap = _sub(b, a), _sub(c, a), _sub(p, a)
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a, (1.0, 0.0, 0.0)
    bp = _sub(p, b)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b, (0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        t = d1 / (d1 - d3)
        return _add(a, _mul(ab, t)), (1 - t, t, 0.0)
    cp = _sub(p, c)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c, (0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        t = d2 / (d2 - d6)
        return _add(a, _mul(ac, t)), (1 - t, 0.0, t)
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return _add(b, _mul(_sub(c, b), t)), (0.0, 1 - t, t)
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return _add(a, _add(_mul(ab, v), _mul(ac, w))), (1 - v - w, v, w)


def closest_fan_binding(mesh: ObjMesh, p):
    """Brute-force closest point over all fan triangles -> (tri, bary, distance)."""
    best = (-1, (1.0, 0.0, 0.0), math.inf)
    for t in range(len(mesh.tri_face)):
        _f, _k, (a, b, c) = mesh.tri_corners(t)
        pa, pb, pc = mesh.points[a], mesh.points[b], mesh.points[c]
        lo = [min(pa[i], pb[i], pc[i]) for i in range(3)]
        hi = [max(pa[i], pb[i], pc[i]) for i in range(3)]
        gap = sum(max(lo[i] - p[i], 0.0, p[i] - hi[i]) ** 2 for i in range(3))
        if gap >= best[2] * best[2]:
            continue
        q, w = _closest_on_triangle(p, pa, pb, pc)
        d = math.dist(p, q)
        if d < best[2]:
            best = (t, w, d)
    return best


class Binding:
    __slots__ = ("tri", "bary", "face", "uv", "mesh_uv", "err_fan", "err_param", "rebound")


def bind_root(mesh: ObjMesh, tri, bary, root, tol):
    """Convert a reference binding; validate against root (points[0]); rebind by closest point if bad."""
    b = Binding()
    b.rebound = False
    ok = isinstance(tri, int) and not isinstance(tri, bool) and 0 <= tri < len(mesh.tri_face)
    if ok:
        bary = _clean_bary(bary)
        b.err_fan = math.dist(fan_point(mesh, tri, bary), root)
        ok = b.err_fan <= tol
    else:
        b.err_fan = math.inf
    if not ok:
        tri, w, _d = closest_fan_binding(mesh, root)
        bary = list(w)
        b.rebound = True
    b.tri, b.bary = tri, bary
    b.face, b.uv = fan_to_face_param(mesh, tri, bary)
    b.mesh_uv = fan_uv(mesh, tri, bary)
    b.err_param = math.dist(eval_face_param(mesh, b.face, b.uv), root)
    return b


# ============================================================================= materials

def srgb_to_linear(c):
    c = _clamp01(_num(c, 0.0))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin3(v, default=(0.0, 0.0, 0.0)):
    if not (isinstance(v, list) and len(v) >= 3):
        v = default
    return tuple(srgb_to_linear(x) for x in v[:3])


def _smoothstep(e0, e1, x):
    if e1 <= e0:
        return 1.0 if x >= e1 else 0.0
    t = _clamp01((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def sisir_gradient(c1, c2, c3, pts, t):
    """Reference preset gradient (sRGB space): lerp(lerp(c1,c2,ss(p.x,p.y,t)), c3, ss(p.z,p.w,t))."""
    a = _smoothstep(pts[0], pts[1], t)
    b = _smoothstep(pts[2], pts[3], t)
    m = [c1[i] + (c2[i] - c1[i]) * a for i in range(3)]
    return [m[i] + (c3[i] - m[i]) * b for i in range(3)]


def fit_color_ramp(c1, c2, c3, pts):
    """Fit usdGen's mix(root, tip, t^k) to the reference 3-stop gradient (linear space)."""
    samples = [i / 64.0 for i in range(65)]
    target = [tuple(srgb_to_linear(x) for x in sisir_gradient(c1, c2, c3, pts, t)) for t in samples]
    root, tip = target[0], target[-1]

    def err(lk):
        k = math.exp(lk)
        e = 0.0
        for t, g in zip(samples, target):
            w = t ** k
            e += sum((root[i] + (tip[i] - root[i]) * w - g[i]) ** 2 for i in range(3))
        return e
    lo, hi = math.log(0.05), math.log(20.0)
    gr = (math.sqrt(5) - 1) / 2
    a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fa, fb = err(a), err(b)
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - gr * (hi - lo)
            fa = err(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + gr * (hi - lo)
            fb = err(b)
    return root, tip, math.exp((lo + hi) / 2)


def chiang_v(beta):
    beta = _clamp01(beta)
    return (0.726 * beta + 0.812 * beta * beta + 3.7 * beta ** 20) ** 2


# Extended UsdGenHairStrands inputs (docs/storm-fur.md, "Imported-groom inputs"). Each entry: candidate
# input names (the first one the shader declares wins), value, Sdf type. Inputs the shader in use does not
# declare are kept in the material's customData["sisir"]["pendingShaderInputs"].
def pending_shader_inputs(m: dict):
    def f(key, default):
        return _num(m.get(key), default)

    def pts(key):
        v = m.get(key)
        return tuple(_num(x, 0.0) for x in v[:4]) if isinstance(v, list) and len(v) >= 4 else (0.0, 0.33, 0.66, 1.0)

    def raw3(key):
        v = m.get(key)
        return tuple(_clamp01(_num(x, 0.0)) for x in v[:3]) if isinstance(v, list) and len(v) >= 3 else (0.0, 0.0, 0.0)
    preset = bool(m.get("UsePresetColorSettings"))
    v = chiang_v(f("LongitudialRoughness", 0.5))
    out = [
        # Lobe widths: the R lobe from Roughness, TT/TRT from the longitudinal band (shader_comparison.md §2).
        (("primaryRoughness",), math.sqrt(_clamp(2.0 * v * f("Roughness", 0.05), 1.0 / 255.0, 1.0)), "float"),
        (("longitudinalRoughness",), math.sqrt(max(0.0, v)), "float"),
        (("azimuthalRoughness",), _clamp01(f("AzimuthalRoughness", 0.35)), "float"),
        (("cuticleTilt",), math.degrees(f("CuticleTiltAngle", 0.03)) / 4.0, "float"),
        (("backScatter",), max(0.0, f("BackScatterScale", 1.0)), "float"),
        # Per-strand A/B/C three-stop gradients chosen by hairId (sRGB values, converted in the shader).
        (("usePresetColors",), 1.0 if preset else 0.0, "float"),
        (("presetSrgb",), 1.0 if preset else 0.0, "float"),
        (("jitterMode",), 1.0 if preset else 0.0, "float"),
    ]
    for g in "ABC":
        for i in (1, 2, 3):
            out.append((("presetColor%s%d" % (g, i),), raw3("Color%s_%d" % (g, i)), "color3f"))
        out.append((("presetPoints%s" % g,), pts("ColorPoint%s" % g), "float4"))
    out += [
        (("presetWeights",), (f("ColorA_Weight", 1.0), f("ColorB_Weight", 0.0), f("ColorC_Weight", 0.0),
                              f("RandomColor_Offset", 0.0)), "float4"),
        (("randomSaturation",), _clamp01(f("Saturate_Random", 0.0)), "float"),
        (("rootBlendRate",), _clamp01(f("RootBlendRate", 0.0)), "float"),
        (("rootFade",), _clamp01(f("RootScalpBlend", 0.5)), "float"),
        # The reference's strand ambient is albedo x ambient x IndirectLightingMultiplier, no environment term.
        (("indirectMultiplier",), 0.0, "float"),
        (("ambientFill",), max(0.0, f("IndirectLightingMultiplier", 1.0)), "float"),
        # Read by the Marschner strand model only (USDGEN_HAIR_SHADING_MODEL=marschner); the Karis lobes keep
        # n = 1.55 and have no hair occlusion.
        (("ior",), max(1.0, f("IOR", 1.55)), "float"),
        (("aoIntensity",), max(0.0, f("HairAOIntensity", 8.0)) if m.get("ShowHairAO", True) else 0.0, "float"),
    ]
    return out


def map_hair_material(m: dict):
    """Reference hair material -> UsdGenHairStrands inputs available today (shader_comparison.md §3)."""
    m = m if isinstance(m, dict) else {}
    beta = _num(m.get("LongitudialRoughness"), 0.5)
    prim_r = _num(m.get("Roughness"), 0.05)
    r = math.sqrt(max(0.0, 2.0 * chiang_v(beta) * prim_r))
    inputs = {
        "roughness": ("float", _clamp(r, 1.0 / 255.0, 1.0)),
        "specular": ("float", _clamp(_num(m.get("SpecularScale"), 1.0) / 2.0, 0.0, 1.0)),
        "opacity": ("float", _clamp01(_num(m.get("Opacity"), 1.0))),
        "randomHue": ("float", _clamp01(_num(m.get("Hue_Random"), 0.0))),
        "randomValue": ("float", 0.0),
        "randomRoughness": ("float", 0.0),
        "backlit": ("float", 1.0),
    }
    ds = "dual" in str(m.get("MultipleScattering", "DualWithIBLOptimized")).lower()
    inputs["scatter"] = ("float", 0.0 if ds else _clamp01(_num(m.get("FakeMultipleScatteringFactor"), 1.0)))
    inputs["selfShadow"] = ("float", 0.45 if ds else 0.0)
    look = {}
    if m.get("UsePresetColorSettings", False):
        weights = [_num(m.get("Color%s_Weight" % g), 1.0 if g == "A" else 0.0) for g in "ABC"]
        g = "ABC"[max(range(3), key=lambda i: (weights[i], -i))]
        c = [[_clamp01(_num(x, 0.0)) for x in (m.get("Color%s_%d" % (g, i)) or [0, 0, 0])[:3]] for i in (1, 2, 3)]
        p = m.get("ColorPoint%s" % g)
        p = [_num(x, 0.0) for x in p[:4]] if isinstance(p, list) and len(p) >= 4 else [0.0, 0.33, 0.66, 1.0]
        root, tip, k = fit_color_ramp(c[0], c[1], c[2], p)
        inputs.update({"baseColor": ("color3f", root), "tipColor": ("color3f", tip),
                       "colorRamp": ("float", k), "useMelanin": ("float", 0.0)})
        look = {"rootColor": root, "tipColor": tip, "rampExponent": k, "gradient": g}
    else:
        inputs.update({"useMelanin": ("float", 1.0),
                       "melanin": ("float", _clamp01(_num(m.get("Melanin"), 0.25))),
                       "redness": ("float", _clamp01(_num(m.get("MelaninRedness"), 0.5)))})
    return inputs, look


def load_shader_inputs(path):
    """Input names/types of UsdGenHairStrands from a shaderDefs.usda (None if unavailable)."""
    if not path or not os.path.isfile(path):
        return None
    layer = Sdf.Layer.FindOrOpen(path)
    spec = layer.GetPrimAtPath("/UsdGenHairStrands") if layer else None
    if not spec:
        return None
    return {p.name[len("inputs:"):]: str(p.typeName) for p in spec.properties
            if p.name.startswith("inputs:") and hasattr(p, "typeName")}


# ============================================================================= op derivation

def resolution_steps(st, cap, lmax):
    if st["resolutionType"] == 0:
        return int(_clamp(int(st["hairResolution"]), 3, cap))
    d = _clamp(st["resolutionDistance"], 0.01, 10.0) * 0.01
    return int(min(max(2 * math.ceil(lmax / d) + 1, 8), cap))


def bezier_profile(t, root, mid, mid_shift, tip, shift_lo=0.1, shift_hi=0.9):
    """The reference quadratic Bezier profile (control points (0,root), (midShift,mid), (1,tip))."""
    m = _clamp(mid_shift, shift_lo, shift_hi)
    t = _clamp01(t)
    a, b = 2 * m, 1 - 2 * m
    if abs(b) <= 1e-6:
        u = t if a <= 1e-6 else _clamp01(t / a)
    else:
        d = math.sqrt(max(0.0, 4 * b * t + a * a))
        u1, u2 = (d - a) / (2 * b), (-a - d) / (2 * b)
        u = u1 if 0 <= u1 <= 1 else (u2 if 0 <= u2 <= 1 else _clamp01(u1))
    return (1 - u) ** 2 * root + 2 * (1 - u) * u * mid + u * u * tip


class OpSpec:
    def __init__(self, name, type_name, enabled):
        self.name = name
        self.type_name = type_name
        self.enabled = bool(enabled)
        self.attrs = []     # (attr name, type, value, kind) kind: 'setting' (re-derived by presets) | 'data'
        self.rels = {}      # rel name -> [paths]
        self.custom = None  # callable(stage, path) -> prim, for externally authored generators

    def a(self, name, type_name, value, kind="setting"):
        self.attrs.append((name, type_name, value, kind))
        return self


def derive_ops(st: dict, ctx: dict, data: dict | None):
    """Operator stack in EXECUTION order for one hair set.

    st   : clamped settings (settings_for()).
    ctx  : scene facts that gate stages (guide count, trim curves, ties, steps inputs, ...).
    data : scene-derived payloads and relationship targets; None when re-deriving for a preset
           (only 'setting' attributes and enabled flags are rewritten then).
    """
    d = data or {}
    is_card = ctx["isCard"]
    n_guides = ctx["guideCount"]
    lmax = max(ctx.get("fallbackLength", 0.0), ctx.get("maxGuideLength", 0.0))
    if is_card:
        lmax = ctx.get("cardMaxLength", lmax)
    steps = resolution_steps(st, 1000, lmax)
    steps100 = resolution_steps(st, 100, lmax)
    r = steps / steps100 if steps100 > 0 else 1.0
    # usdGen:guides is authored only where the stage consumes guides under these settings: a node with a
    # reference must match its operator's declared slot count, a node without one always compiles.
    guides = ctx.get("guidesPath") or None
    ops = []

    # --- generators ---------------------------------------------------------------------------------
    if is_card:
        op = OpSpec("cards", "UsdGenCardSource", True)
        op.a("usdGen:cvCount", "int", steps)
        op.custom = d.get("cardSourceAuthor")
        ops.append(op)
    else:
        dist = st["distribution"]
        op = OpSpec("scatter", "UsdGenScatter", True)
        op.a("usdGen:seed", "int", 1337)
        op.a("usdGen:distribution", "token", DISTRIBUTION_TOKENS.get(dist, "even"))
        op.a("usdGen:countMode", "token", "surface")
        op.a("usdGen:density", "float", max(0.0, st["hairDensity"]))
        coverage = _clamp01(st["guideProximityDistance"])
        prox = dist not in (3, 4) and n_guides > 0 and (st["guideProximityMode"] == 1 or coverage < 0.9999)
        mode = "none" if not prox else ("distance" if st["guideProximityMode"] == 1 else "coverage")
        op.a("usdGen:proximity:mode", "token", mode)
        op.a("usdGen:proximity:distance", "float", max(0.0, st["guideProximityRadius"]) * 0.01)
        op.a("usdGen:proximity:coverage", "float", coverage)
        if guides and (dist == 3 or mode != "none"):
            op.rels["usdGen:guides"] = [guides]
        if data is not None:
            op.rels["usdGen:densityMap"] = list(d.get("densityMaps", []))
        ops.append(op)

        op = OpSpec("interp", "UsdGenGuideInterpolate", True)
        op.a("usdGen:cvCount", "int", steps)
        op.a("usdGen:weighting", "token", "nearest" if dist == 3 else WEIGHTING_TOKENS[st["guideInterpolation"]])
        op.a("usdGen:tightness", "float", st["guideInterpolationTightness"])
        op.a("usdGen:bendFollow", "bool", st["guideBendFollow"])
        op.a("usdGen:defaultLength", "float", ctx.get("fallbackLength", 0.01))
        op.a("usdGen:part:amount", "float", st["partingAmount"])
        op.a("usdGen:part:angle", "float", st["partingAngle"])
        op.a("usdGen:part:radius", "float", st["partingRadius"] / 100.0)
        op.a("usdGen:part:distance", "float", st["partingDistance"])
        part = d.get("parting")
        if part:
            op.a("usdGen:part:points", "point3f[]", part["points"], "data")
            op.a("usdGen:part:counts", "int[]", part["counts"], "data")
            if part.get("normals"):
                op.a("usdGen:part:normals", "normal3f[]", part["normals"], "data")
        if guides:
            op.rels["usdGen:guides"] = [guides]
        ops.append(op)

    # --- clumps / braid -------------------------------------------------------------------------------
    ct = st["clumpType"]
    op = OpSpec("clump", "UsdGenClump", ct == 2 or ct == 3 or (ct == 1 and n_guides > 0))
    op.a("usdGen:seed", "int", 1337)
    op.a("usdGen:clump:centers", "token", CLUMP_CENTER_TOKENS.get(ct, "count"))
    op.a("usdGen:clump:count", "int", max(0, int(st["clumpRandomCount"])))
    op.a("usdGen:clump:method", "token", "linearBlend")
    op.a("usdGen:preserveLength", "float", 0.0)
    op.a("usdGen:clump:amount", "float", st["clumpAmount"])
    op.a("usdGen:clump:profile:mode", "token", "bezier")
    op.a("usdGen:clump:root", "float", st["clumpRootTightness"])
    op.a("usdGen:clump:mid", "float", st["clumpMidTightness"])
    op.a("usdGen:clump:midShift", "float", st["clumpMidShift"])
    op.a("usdGen:clump:tip", "float", st["clumpTipTightness"])
    op.a("usdGen:clump:flyaway:fraction", "float", st["clumpFlyawayPercentage"])
    op.a("usdGen:clump:flyaway:amount", "float", st["clumpFlyawayAmount"])
    op.a("usdGen:clump:sub:enabled", "bool", st["subClumpsEnabled"])
    op.a("usdGen:clump:sub:count", "int", st["subClumpCount"])
    op.a("usdGen:clump:sub:spread", "float", st["subClumpTightness"])
    op.a("usdGen:clump:sub:amount", "float", st["subClumpAmount"])
    op.a("usdGen:clump:sub:ids", "bool", st["generateSubClumpIds"])
    ov = d.get("clumpOverrides")
    if ov:
        op.a("usdGen:clump:overrides", "int[]", ov[0], "data")
        op.a("usdGen:clump:overrideCount", "int", ov[1], "data")
    if guides and op.enabled and ct in (1, 3):
        op.rels["usdGen:guides"] = [guides]
    ops.append(op)

    op = OpSpec("braid", "UsdGenBraid", ct == 4 and n_guides > 0 and steps > 1)
    for key, attr in (("braidFrequency", "frequency"), ("braidWidth", "width"), ("braidFlatness", "flatness"),
                      ("braidKnotTightness", "knotTightness"), ("braidFadeIn", "fadeIn"),
                      ("braidFadeOut", "fadeOut"), ("braidRootSpread", "rootSpread"),
                      ("braidTipSpread", "tipSpread"), ("braidTailLength", "tailLength"),
                      ("braidTailRelax", "tailRelax"), ("braidTailCinch", "tailCinch")):
        op.a("usdGen:braid:" + attr, "float", st[key])
    op.a("usdGen:braid:referenceLength", "float", 0.0)
    op.a("usdGen:braid:root", "float", st["clumpRootTightness"])
    op.a("usdGen:braid:mid", "float", st["clumpMidTightness"])
    op.a("usdGen:braid:midShift", "float", st["clumpMidShift"])
    op.a("usdGen:braid:tip", "float", st["clumpTipTightness"])
    op.a("usdGen:braid:amount", "float", st["clumpAmount"])
    op.a("usdGen:braid:flyaway:fraction", "float", st["clumpFlyawayPercentage"])
    op.a("usdGen:braid:flyaway:amount", "float", st["clumpFlyawayAmount"])
    op.a("usdGen:braid:knotMask", "bool", bool(st["frizzPreserveBraidKnots"] and st["frizzAmount"] > 1e-4))
    if guides and op.enabled:
        op.rels["usdGen:guides"] = [guides]
    ops.append(op)

    # --- twist ----------------------------------------------------------------------------------------
    tw_on = st["twistAmount"] > 0 and (st["twistRoot"] != 0 or st["twistMid"] != 0 or st["twistTip"] != 0) and steps >= 2
    op = OpSpec("twist", "UsdGenTwist", tw_on)
    op.a("usdGen:twist:amount", "float", st["twistAmount"])
    op.a("usdGen:twist:root", "float", st["twistRoot"])
    op.a("usdGen:twist:mid", "float", st["twistMid"])
    op.a("usdGen:twist:midShift", "float", st["twistMidShift"])
    op.a("usdGen:twist:tip", "float", st["twistTip"])
    op.a("usdGen:twist:noise", "float", st["twistNoise"])
    op.a("usdGen:twist:useClumpIds", "bool", st["twistUseClumpIds"])
    op.a("usdGen:twist:resnapRoot", "bool", (not is_card) and st["twistRoot"] != 0)
    ops.append(op)

    # --- length ---------------------------------------------------------------------------------------
    op = OpSpec("length", "UsdGenLength", True)
    op.a("usdGen:length:mode", "token", "randomRange")
    op.a("usdGen:length:min", "float", st["lengthMin"])
    op.a("usdGen:length:max", "float", st["lengthMax"])
    op.a("usdGen:length:randomize", "float", st["lengthRandomize"])
    op.a("usdGen:length:shareByClump", "bool", st["lengthUseClumpIds"])
    op.a("usdGen:length:followClumpMean", "bool", st["lengthUseInterpolation"])
    if data is not None:
        op.rels["usdGen:lengthMap"] = list(d.get("lengthMaps", []))
    ops.append(op)

    # --- trim -----------------------------------------------------------------------------------------
    op = OpSpec("trim", "UsdGenTrim", ctx.get("trimCurveCount", 0) > 0)
    op.a("usdGen:trim:angle", "float", st["trimAngle"])
    op.a("usdGen:trim:invert", "bool", st["trimInvertAngle"])
    op.a("usdGen:trim:distance", "float", st["trimDistance"])
    op.a("usdGen:trim:normalSearch", "float", 0.08)
    if d.get("trimCurves"):
        op.rels["usdGen:trim:curves"] = [d["trimCurves"]]
    ops.append(op)

    # --- frizz ----------------------------------------------------------------------------------------
    op = OpSpec("frizz", "UsdGenFrizz", st["frizzAmount"] > 1e-4)
    op.a("usdGen:frizz:amplitude", "float", st["frizzAmount"] / 100.0 * 0.25)
    op.a("usdGen:frizz:wavesX", "float", st["frizzScale"] / 10.0)
    op.a("usdGen:frizz:wavesZ", "float", st["frizzScaleZ"] / 10.0)
    op.a("usdGen:frizz:noise", "float", st["frizzNoise"] / 100.0)
    op.a("usdGen:frizz:noiseSize", "float", st["frizzNoiseSize"] * 0.01)
    op.a("usdGen:frizz:sharpness", "float", st["frizzNoiseSharpness"])
    op.a("usdGen:frizz:root", "float", st["frizzRoot"])
    op.a("usdGen:frizz:mid", "float", st["frizzMid"])
    op.a("usdGen:frizz:midShift", "float", st["frizzMidShift"])
    op.a("usdGen:frizz:tip", "float", st["frizzTip"])
    op.a("usdGen:frizz:useClumpIds", "bool", st["frizzUseClumpIds"])
    op.a("usdGen:frizz:preserveKnots", "bool", st["frizzPreserveBraidKnots"])
    ops.append(op)

    # --- tie-clump ------------------------------------------------------------------------------------
    op = OpSpec("tieClump", "UsdGenTieClump",
                st["clumpToHairTies"] and st["tiesClumpAmount"] > 1e-4 and ctx.get("tieCount", 0) > 0)
    op.a("usdGen:tie:amount", "float", st["tiesClumpAmount"])
    op.a("usdGen:tie:spread", "float", st["tiesClumpSpread"])
    ties = d.get("ties")
    if ties:
        op.a("usdGen:tie:positions", "point3f[]", ties["positions"], "data")
        op.a("usdGen:tie:entryCounts", "int[]", ties["entryCounts"], "data")
        op.a("usdGen:tie:guideIndices", "int[]", ties["guideIndices"], "data")
        op.a("usdGen:tie:vertexIndices", "int[]", ties["vertexIndices"], "data")
    if guides and op.enabled:
        op.rels["usdGen:guides"] = [guides]
    ops.append(op)

    # --- push-out -------------------------------------------------------------------------------------
    op = OpSpec("pushOut", "UsdGenCollide", st["pushOutEnabled"])
    op.a("usdGen:resolveType", "token", "relax")
    op.a("usdGen:clearance:root", "float", max(1e-4, st["pushOutRoot"] * 0.01))
    op.a("usdGen:clearance:tip", "float", max(1e-4, st["pushOutTip"] * 0.01))
    op.a("usdGen:relax:iterations", "int", 24)
    ops.append(op)

    # --- smooth ---------------------------------------------------------------------------------------
    op = OpSpec("smooth", "UsdGenSmooth", st["smoothness"] > 1e-4)
    op.a("usdGen:mode", "token", "box")
    op.a("usdGen:strength", "float", _clamp01(st["smoothness"]))
    op.a("usdGen:window", "int", max(2, _lround(2 * r)))
    op.a("usdGen:iterations", "int", 1)
    op.a("usdGen:lockRoot", "bool", True)
    ops.append(op)

    # --- resample (distance resolution only) ----------------------------------------------------------
    op = OpSpec("resample", "UsdGenResample", st["resolutionType"] == 1)
    op.a("usdGen:spacing", "float", _clamp(st["resolutionDistance"], 0.01, 10.0) * 0.01)
    ops.append(op)

    # --- width ----------------------------------------------------------------------------------------
    op = OpSpec("width", "UsdGenWidth", True)
    op.a("usdGen:width", "float", 2.0 * st["hairThickness"])
    op.a("usdGen:width:profile:mode", "token", "bezier")
    op.a("usdGen:width:root", "float", st["thicknessRoot"])
    op.a("usdGen:width:mid", "float", st["thicknessMid"])
    op.a("usdGen:width:midShift", "float", st["thicknessMidShift"])
    op.a("usdGen:width:tip", "float", st["thicknessTip"])
    op.a("usdGen:replace", "bool", True)
    ops.append(op)
    return ops, {"steps": steps, "steps100": steps100}


# ============================================================================= USD helpers

_SDF_TYPES = {
    "float": Sdf.ValueTypeNames.Float, "int": Sdf.ValueTypeNames.Int, "bool": Sdf.ValueTypeNames.Bool,
    "token": Sdf.ValueTypeNames.Token, "int[]": Sdf.ValueTypeNames.IntArray,
    "float[]": Sdf.ValueTypeNames.FloatArray, "point3f[]": Sdf.ValueTypeNames.Point3fArray,
    "normal3f[]": Sdf.ValueTypeNames.Normal3fArray, "color3f": Sdf.ValueTypeNames.Color3f,
    "float4": Sdf.ValueTypeNames.Float4, "float2": Sdf.ValueTypeNames.Float2,
    "string": Sdf.ValueTypeNames.String,
}


def register_usdgen_schema(extra=None):
    """Register the generated usdGen schema so authored ops get their declared types (best effort)."""
    if Plug.Registry().GetPluginWithName("usdGenSchema") is None:   # not on PXR_PLUGINPATH_NAME already
        cands = [extra, os.environ.get("USDGEN_SCHEMA_PLUGIN"),
                 os.path.join(HERE, "..", "..", "plugin", "usdGenSchema", "resources")]
        for c in cands:
            if c and os.path.isfile(os.path.join(c, "plugInfo.json")):
                try:
                    Plug.Registry().RegisterPlugins(os.path.abspath(c))
                except Exception:  # pragma: no cover
                    pass
                break
    return Usd.SchemaRegistry().FindConcretePrimDefinition("UsdGenDescription") is not None


def _schema_attr(type_name, attr):
    pd = Usd.SchemaRegistry().FindConcretePrimDefinition(type_name)
    if pd is None:
        return None
    try:
        a = pd.GetAttributeDefinition(attr)
    except Exception:  # pragma: no cover
        return None
    return a if a else None


_UNIFORM_FALLBACK = {"usdGen:seed", "usdGen:curve:basis"}


def set_attr(prim, name, type_str, value, warn_list=None):
    """Create/set an attribute using the schema's declared type/variability when known."""
    sdf_type = _SDF_TYPES[type_str]
    uniform = type_str == "token" or name in _UNIFORM_FALLBACK
    d = _schema_attr(str(prim.GetTypeName()), name)
    if d is not None:
        declared = d.GetTypeName()
        if declared != sdf_type and warn_list is not None:
            warn_list.append("%s.%s: schema declares %s, authored %s" % (prim.GetPath(), name, declared, sdf_type))
        sdf_type = declared
        uniform = d.GetVariability() == Sdf.VariabilityUniform
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, sdf_type, False,
                                    Sdf.VariabilityUniform if uniform else Sdf.VariabilityVarying)
    if attr.GetTypeName() == Sdf.ValueTypeNames.Float and isinstance(value, bool):
        value = float(value)
    attr.Set(value)
    return attr


def to_vt(v):
    """JSON value -> something customData can hold (VtDictionary-compatible)."""
    if isinstance(v, dict):
        return {str(k): to_vt(x) for k, x in v.items() if x is not None}
    if isinstance(v, bool) or isinstance(v, str):
        return v
    if isinstance(v, int):
        return v if -(1 << 63) <= v < (1 << 63) else str(v)
    if isinstance(v, float):
        return v
    if isinstance(v, list):
        if not v:
            return Vt.IntArray()
        if all(isinstance(x, bool) for x in v):
            return Vt.BoolArray(v)
        if all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return Vt.IntArray(v) if all(-(1 << 31) <= x < (1 << 31) for x in v) else Vt.Int64Array(v)
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            return Vt.DoubleArray([float(x) for x in v])
        if all(isinstance(x, str) for x in v):
            return Vt.StringArray(v)
        return json.dumps(v)
    return str(v)


def from_vt(v):
    if isinstance(v, dict):
        return {k: from_vt(x) for k, x in v.items()}
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and type(v).__module__.startswith("pxr"):
        return [from_vt(x) for x in v]
    return v


def _valid_name(name, used):
    base = Tf.MakeValidIdentifier(str(name) if name else "set") or "set"
    n, i = base, 2
    while n in used:
        n = "%s_%d" % (base, i)
        i += 1
    used.add(n)
    return n


def _rel_path(target, layer_dir):
    rel = os.path.relpath(target, layer_dir).replace("\\", "/")
    return rel if rel.startswith(".") else "./" + rel


# ============================================================================= converter

class ConvertOptions:
    def __init__(self, sets=None, extract=True, active_ops_only=False, cards_module=None,
                 include_hidden_layers=False, shader_defs=None, quiet=False, bind_tol=1e-4):
        self.sets = sets
        self.extract = extract
        self.active_ops_only = active_ops_only
        self.cards_module = cards_module
        self.include_hidden_layers = include_hidden_layers
        self.shader_defs = shader_defs
        self.quiet = quiet
        self.bind_tol = bind_tol


def import_cards_module(path=None):
    """cards.py: next to this file, else an explicit path / $SISIR_CARDS_MODULE."""
    for cand in (path, os.environ.get("SISIR_CARDS_MODULE")):
        if cand:
            p = cand if cand.endswith(".py") else os.path.join(cand, "cards.py")
            if os.path.isfile(p):
                spec = importlib.util.spec_from_file_location("sisir_cards_ext", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod, p
    try:
        if __package__:
            mod = importlib.import_module(__package__ + ".cards")
        else:
            if HERE not in sys.path:
                sys.path.insert(0, HERE)
            mod = importlib.import_module("cards")
        return mod, getattr(mod, "__file__", "cards")
    except ImportError:
        return None, None


class Converter:
    def __init__(self, scene_path, out_path, options: ConvertOptions):
        self.scene_path = scene_path
        self.out_path = os.path.abspath(out_path)
        self.out_dir = os.path.dirname(self.out_path)
        self.opt = options
        self.log_lines = []
        self.summary = {"sets": [], "warnings": []}
        self.schema_warnings = []

    # --- logging ------------------------------------------------------------------------------------
    def log(self, msg):
        self.log_lines.append(msg)
        if not self.opt.quiet:
            print(msg)

    def warn(self, msg):
        self.summary["warnings"].append(msg)
        self.log("warning: " + msg)

    # --- main ---------------------------------------------------------------------------------------
    def run(self):
        self.schema_ok = register_usdgen_schema()
        self.ss = SisirScene.open(self.scene_path)
        sc = self.ss.scene
        for name, reason in self.ss.rejected:
            self.warn("archive entry rejected (%s): %r" % (reason, name))
        os.makedirs(self.out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.out_path))[0]
        self.asset_dir = os.path.join(self.out_dir, stem + "_assets")
        self.extracted = {}
        if self.opt.extract and self.ss.archive is not None:
            self.extracted = self.ss.extract(self.asset_dir)
            self.log("extracted %d assets -> %s" % (len(self.extracted), self.asset_dir))
        self.scene = copy.deepcopy(sc)
        self._resolve_paths(self.scene)

        layer = Sdf.Layer.CreateAnonymous(".usda")
        self.stage = stage = Usd.Stage.Open(layer)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world)
        layer.documentation = ("Converted by tools/sisir/sir2usd.py %s from %s" %
                               (__version__, os.path.basename(self.scene_path)))

        self._author_mesh()
        groom = UsdGeom.Xform.Define(stage, "/World/Groom").GetPrim()   # plain Xform: CPU reference lane
        UsdGeom.Scope.Define(stage, "/World/Groom/Guides")
        self._author_camera()
        self.cards_mod, self.cards_src = import_cards_module(self.opt.cards_module)
        self.card_prims = None
        self.materials = {}
        self.shader_inputs = load_shader_inputs(
            self.opt.shader_defs or os.path.join(HERE, "..", "..", "usdGenShaders", "resources", "shaders",
                                                 "shaderDefs.usda"))
        sets = self.ss.hair_sets()
        chosen = self._select_sets(sets)
        used_names, skipped = set(), []
        self.mask_primvars = {}
        for i in chosen:
            h = sets[i]
            st = settings_for(h.get("settings"))
            if st["distribution"] == 4 and self.cards_mod is None:
                self.warn("hair set %d %r is card-grown but cards.py is not available: skipped" % (i, h.get("name")))
                skipped.append({"index": i, "name": str(h.get("name", "")), "reason": "cards.py unavailable",
                                "settings": to_vt(h.get("settings") or {})})
                continue
            try:
                self._author_set(i, h, st, used_names)
            except Exception as e:  # keep converting the other sets
                self.warn("hair set %d %r failed: %s: %s" % (i, h.get("name"), type(e).__name__, e))
                skipped.append({"index": i, "name": str(h.get("name", "")), "reason": "%s: %s" % (type(e).__name__, e),
                                "settings": to_vt(h.get("settings") or {})})
        groom_cd = {"source": os.path.basename(self.scene_path), "sceneVersion": to_vt(sc.get("version")),
                    "sirVersion": to_vt(sc.get("sirVersion")), "converter": "sir2usd " + __version__,
                    "unsupported": Vt.StringArray(self._scene_unsupported())}
        if isinstance(sc.get("guideSettings"), dict):
            groom_cd["guideSettings"] = to_vt(sc["guideSettings"])
        if skipped:
            groom_cd["skippedHairSets"] = {"set%d" % s["index"]: s for s in skipped}
        groom.SetCustomDataByKey("sisir", groom_cd)
        layer.Export(self.out_path)
        self.log("wrote %s" % self.out_path)
        return self.summary

    def _select_sets(self, sets):
        if not self.opt.sets:
            return list(range(len(sets)))
        out = []
        for s in self.opt.sets:
            if isinstance(s, int) or (isinstance(s, str) and s.lstrip("-").isdigit()):
                i = int(s)
                if 0 <= i < len(sets):
                    out.append(i)
                    continue
            hits = [j for j, h in enumerate(sets) if str(h.get("name", "")) == str(s)]
            if not hits:
                raise SirFormatError("--sets: no hair set %r" % (s,))
            out.extend(hits)
        return sorted(set(out))

    def _resolve_paths(self, scene):
        """Rewrite asset path keys to absolute local files ('' when missing), like the reference loader."""
        copied = {}
        for cont, key in _path_slots(scene):
            v = cont.get(key)
            if not v:
                continue
            local = ""
            if self.ss.archive is not None:
                local = self.extracted.get(v.replace("\\", "/"), "")   # absolute paths in an archive: ignored
            else:
                src = self.ss._legacy_path(v)
                if src:
                    if src not in copied:
                        dst = os.path.join(self.asset_dir, os.path.basename(src))
                        n = 2
                        while os.path.exists(dst) and os.path.abspath(dst) != src:
                            s, e = os.path.splitext(os.path.basename(src))
                            dst = os.path.join(self.asset_dir, "%s_%d%s" % (s, n, e))
                            n += 1
                        if self.opt.extract:
                            os.makedirs(self.asset_dir, exist_ok=True)
                            if os.path.abspath(dst) != src:
                                shutil.copyfile(src, dst)
                            copied[src] = os.path.abspath(dst)
                        else:
                            copied[src] = src
                    local = copied[src]
            cont[key] = local

    def _scene_unsupported(self):
        sc = self.ss.scene
        out = []
        if sc.get("blendshapePath"):
            out.append("blendshapePath/blendshapeWeight: blend-shape mesh not converted (rest mesh only)")
        if sc.get("refMeshPath"):
            out.append("refMeshPath: reference (attract/collide) mesh not converted")
        if sc.get("mannequinEnabled"):
            out.append("mannequinEnabled: procedural mannequin head not converted")
        for key in ("combCurves", "cardCombCurves", "transplantMarkers"):
            if sc.get(key):
                out.append("%s: authoring-only data, not converted (%d)" % (key, len(sc[key])))
        if sc.get("cardTrimCurves"):
            out.append("cardTrimCurves: card trim surfaces not transported (%d)" % len(sc["cardTrimCurves"]))
        out.append("renderLight/renderPostProcess/renderVoxelAo/renderSettings*: viewer state, not converted")
        if sc.get("textureModule"):
            out.append("textureModule: card-texture authoring scene, not converted (its baked strands are "
                       "imported through cardHairSources)")
        return out

    # --- mesh ---------------------------------------------------------------------------------------
    def _author_mesh(self):
        sc = self.ss.scene
        mesh_path = sc.get("meshPath", "")
        data = self.ss.read_asset(mesh_path)
        if data is None:
            raise SirFormatError("scene mesh %r is not embedded / readable" % mesh_path)
        if not str(mesh_path).lower().endswith(".obj"):
            raise SirFormatError("scene mesh %r is not an OBJ (only OBJ meshes are supported)" % mesh_path)
        scale = self.ss.mesh_import_scale
        self.mesh = m = ObjMesh.parse(data, scale)
        name = Tf.MakeValidIdentifier(os.path.splitext(os.path.basename(mesh_path))[0]) or "Scalp"
        self.mesh_path = Sdf.Path("/World").AppendChild(name)
        mesh = UsdGeom.Mesh.Define(self.stage, self.mesh_path)
        prim = mesh.GetPrim()
        prim.AddAppliedSchema("UsdGenRestAPI")
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreatePointsAttr(Vt.Vec3fArray(m.points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(m.face_counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(m.face_indices))
        mesh.CreateNormalsAttr(Vt.Vec3fArray(m.normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        if m.has_uv:
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray(m.uvs))
        lo, hi = m.bbox()
        mesh.CreateExtentAttr(Vt.Vec3fArray([lo, hi]))
        self.mesh_area = m.area()
        self.bbox_diag = math.dist(lo, hi)
        prim.SetCustomDataByKey("sisir", {
            "meshPath": str(mesh_path), "meshImportScale": scale, "objPositions": m.n_positions,
            "vertexOrder": "unique (v, vt, vn) OBJ corners, first occurrence in face order",
            "faceOrder": "OBJ file order; reference triangle t = fan (f0, fk, fk+1) in this order"})
        self.log("mesh %s: %d vertices (%d OBJ positions, %d value-unique corners), %d faces, %d fan triangles, "
                 "area %.7f m^2" % (self.mesh_path, len(m.points), m.n_positions, m.value_unique,
                                    len(m.face_counts), len(m.tri_face), self.mesh_area))
        if m.value_unique != len(m.points):
            self.warn("OBJ has %d index-unique but %d value-unique corners; vertex masks follow the index-unique "
                      "numbering" % (len(m.points), m.value_unique))
        self._author_mesh_material(prim, name)

    def _author_mesh_material(self, prim, name):
        """The mesh's render material (renderMeshMaterial) as a UsdPreviewSurface.

        Reference mesh shader (mesh_pbr_standalone.hlsl, PS_Mesh); a texture that is not set reads as white:
          albedo    = srgb_to_linear(tex.rgb) * albedoColor
          roughness = max(tex.r * roughness, 0.045);   metalness = tex.r * metalness
          normal    = normalize((2 tex - 1) * (strength, strength * ySign, 1)), tangent space
        The albedo texture is the scene mesh texture when useSceneMeshTextureAsAlbedo is set, else albedoTexturePath;
        an empty slot falls back to the other ("Defaults to the mesh's own texture").
        Normal Y: the reference flips the mesh v at import, and normalFlipY marks an OpenGL-style (+Y up) map, which
        is USD's convention. Kristina.sir's normal and detail-normal maps are both +Y up (curl test) with the flag set.
        So a flagged map is used as is and an unflagged (DirectX-style) map gets Y negated.
        The detail normal map, cavity (it darkens direct specular only) and subsurface scattering have no
        UsdPreviewSurface input; they are listed as unsupported on the material."""
        sc = self.scene
        raw = self.ss.scene.get("renderMeshMaterial")
        mm = sc.get("renderMeshMaterial")
        mm = mm if isinstance(mm, dict) else {}
        scene_tex = sc.get("meshTexturePath", "") or ""
        if mm:
            own = mm.get("albedoTexturePath", "") or ""
            first, second = (scene_tex, own) if mm.get("useSceneMeshTextureAsAlbedo") else (own, scene_tex)
            albedo_tex = first or second
        else:
            albedo_tex = scene_tex
        if not mm and not albedo_tex:
            return
        look = UsdShade.Material.Define(self.stage, "/World/Looks/%s_skin" % name)
        lp = look.GetPath()
        pbs = UsdShade.Shader.Define(self.stage, lp.AppendChild("Surface"))
        pbs.CreateIdAttr("UsdPreviewSurface")
        reader = None

        def texture(child, path, color_space, scale, bias=None):
            nonlocal reader
            if reader is None:
                reader = UsdShade.Shader.Define(self.stage, lp.AppendChild("stReader"))
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
                reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            t = UsdShade.Shader.Define(self.stage, lp.AppendChild(child))
            t.CreateIdAttr("UsdUVTexture")
            t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(_rel_path(path, self.out_dir))
            t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
            t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            for wrap in ("wrapS", "wrapT"):                     # the reference samples with BilinearWrap
                t.CreateInput(wrap, Sdf.ValueTypeNames.Token).Set("repeat")
            if scale is not None:
                t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            if bias is not None:
                t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
            return t

        color = [_num(x, 1.0) for x in ((mm.get("albedoColor") or []) + [1.0, 1.0, 1.0])[:3]]
        if albedo_tex:
            t = texture("diffuse", albedo_tex, "sRGB", None if color == [1.0, 1.0, 1.0] else color + [1.0])
            t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            pbs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
        else:
            pbs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        if not mm:                                      # older scene without a render material: viewport texture only
            pbs.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)

        if mm:
            used = {"albedo": os.path.basename(albedo_tex) or "albedoColor"}
            for key, inp, tex_key, default in (("roughness", "roughness", "roughnessTexturePath", 0.5),
                                               ("metalness", "metallic", "metalnessTexturePath", 0.0)):
                f = _num(mm.get(key), default)
                tex = mm.get(tex_key, "") or ""
                if tex:
                    t = texture(key + "Tex", tex, "raw", (f, f, f, 1.0))
                    t.CreateOutput("r", Sdf.ValueTypeNames.Float)
                    pbs.CreateInput(inp, Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(), "r")
                    used[key] = "%s x %g" % (os.path.basename(tex), f)
                else:
                    v = _clamp01(f)
                    pbs.CreateInput(inp, Sdf.ValueTypeNames.Float).Set(max(v, 0.045) if key == "roughness" else v)
                    used[key] = "%g" % v
            normal = mm.get("normalTexturePath", "") or ""
            if normal:
                s = _num(mm.get("normalStrength"), 1.0)
                y = 1.0 if mm.get("normalFlipY", True) else -1.0
                t = texture("normalTex", normal, "raw", (2 * s, 2 * s * y, 2.0, 1.0), (-s, -s * y, -1.0, 0.0))
                t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                pbs.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(t.ConnectableAPI(), "rgb")
                used["normal"] = "%s x %g%s" % (os.path.basename(normal), s, "" if y > 0 else ", Y negated")
            unsupported = []
            if (mm.get("detailNormalTexturePath") or "") and _num(mm.get("detailNormalStrength"), 1.0) != 0.0:
                unsupported.append("detail normal map (%s, strength %g, tiling %g x %g): UsdPreviewSurface has one "
                                   "normal input" % (os.path.basename(mm["detailNormalTexturePath"]),
                                                     _num(mm.get("detailNormalStrength"), 1.0),
                                                     _num(mm.get("detailNormalTileU"), 1.0),
                                                     _num(mm.get("detailNormalTileV"), 1.0)))
            if (mm.get("cavityTexturePath") or "") and _num(mm.get("cavityIntensity"), 1.0) > 0.0:
                unsupported.append("cavity map (%s): darkens direct specular only, no UsdPreviewSurface input"
                                   % os.path.basename(mm["cavityTexturePath"]))
            if mm.get("subsurfaceEnabled") and _num(mm.get("subsurfaceWeight"), 0.0) > 0.0:
                unsupported.append("subsurface scattering (weight %g, radius %g): no UsdPreviewSurface input"
                                   % (_num(mm.get("subsurfaceWeight"), 0.0), _num(mm.get("subsurfaceRadius"), 0.0)))
            look.GetPrim().SetCustomDataByKey("sisir", {
                "renderMeshMaterial": to_vt(raw if isinstance(raw, dict) else {}),
                "note": "albedoColor is applied as a linear factor (the reference shader multiplies it after "
                        "decoding the sRGB texture; host-side conversion not verified)",
                "unsupported": Vt.StringArray(unsupported)})
            self.log("mesh material %s: %s" % (lp, ", ".join("%s=%s" % kv for kv in used.items())))
            for u in unsupported:
                self.log("     unsupported: %s" % u)
        pbs.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        look.CreateSurfaceOutput().ConnectToSource(pbs.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(look)

    def _author_camera(self):
        cam = self.ss.scene.get("camera")
        if not isinstance(cam, dict) or not isinstance(cam.get("target"), list):
            return
        t = tuple(_num(x, 0.0) for x in cam["target"][:3])
        dist = _num(cam.get("distance"), 1.0)
        yaw, pitch = math.radians(_num(cam.get("yaw"), 0.0)), math.radians(_num(cam.get("pitch"), 0.0))
        fwd_to_eye = (math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw))
        eye = _add(t, _mul(fwd_to_eye, dist))
        z = _normalize(_sub(eye, t))
        x = _normalize(_cross((0.0, 1.0, 0.0), z))
        y = _cross(z, x)
        c = UsdGeom.Camera.Define(self.stage, "/World/Camera")
        c.AddTransformOp().Set(Gf.Matrix4d(x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0,
                                           eye[0], eye[1], eye[2], 1))
        fov = _num(cam.get("fovY"), math.pi / 4)
        c.CreateVerticalApertureAttr(20.25)
        c.CreateHorizontalApertureAttr(36.0)
        c.CreateFocalLengthAttr(10.125 / math.tan(max(1e-3, fov) / 2))
        c.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        c.GetPrim().SetCustomDataByKey("sisir:note", "eye from target/yaw/pitch/distance; yaw/pitch convention inferred")

    # --- per hair set -------------------------------------------------------------------------------
    def _layer_visible(self, li):
        raw = self.ss.scene.get("layers")
        raw = raw if isinstance(raw, list) else []
        if not (0 <= li < len(raw)) or not isinstance(raw[li], dict):
            return False
        return self.opt.include_hidden_layers or bool(raw[li].get("visible", True))

    def _layer_bits(self, h, st):
        n = len(self.ss.scene.get("layers") or [])
        if st["distribution"] == 4:
            return [True] * n
        lv = h.get("layerVisibility")
        if not isinstance(lv, list) or (not lv and h.get("_legacy")):
            return [True] * n if h.get("_legacy") else []
        return [bool(x) for x in lv]

    def _set_guides(self, bits):
        out = []
        for gi, g in enumerate(self.ss.scene.get("curves") or []):
            if not isinstance(g, dict):
                continue
            li = g.get("layer", 0)
            li = li if isinstance(li, int) and 0 <= li < len(self.ss.scene.get("layers") or []) else 0
            pts = g.get("points")
            if not (li < len(bits) and bits[li] and self._layer_visible(li)):
                continue
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            try:
                pts = [tuple(float(c) for c in p[:3]) for p in pts]
            except (TypeError, ValueError):
                continue
            if polyline_length(pts) <= 1e-6:
                continue
            out.append((gi, g, pts))
        return out

    def _author_guides(self, desc_name, guides, unsupported):
        if not guides:
            return None, {}
        gs_path = Sdf.Path("/World/Groom/Guides").AppendChild(desc_name)
        self.stage.DefinePrim(gs_path, "UsdGenGuideSet")
        cpath = gs_path.AppendChild("curves")
        curves = UsdGeom.BasisCurves.Define(self.stage, cpath)
        prim = curves.GetPrim()
        prim.AddAppliedSchema("UsdGenCurveAPI")
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curves.CreatePurposeAttr(UsdGeom.Tokens.guide)
        pts, counts, skinprim, skinuv, meshuv, ids, mirror, layer_ids, src_idx = [], [], [], [], [], [], [], [], []
        used_ids, id_to_index = set(), {}
        errs_fan, errs_param, rebound = [], [], 0
        for k, (gi, g, gpts) in enumerate(guides):
            counts.append(len(gpts))
            pts.extend(gpts)
            b = bind_root(self.mesh, g.get("rootTri", -1), g.get("rootBary"), gpts[0], self.opt.bind_tol)
            if b.rebound:
                rebound += 1
            else:
                errs_fan.append(b.err_fan)
            errs_param.append(b.err_param)
            skinprim.append(b.face)
            skinuv.append(b.uv)
            meshuv.append(b.mesh_uv)
            gid = g.get("id", 0)
            gid = gid if isinstance(gid, int) and not isinstance(gid, bool) and 0 < gid < (1 << 64) else 0
            if gid == 0 or gid in used_ids:
                gid = (1 << 40) | (gi + 1)
            used_ids.add(gid)
            ids.append(gid)
            gid_raw = g.get("id", 0)
            if isinstance(gid_raw, int):
                id_to_index[gid_raw] = k
            twin = g.get("mirrorTwinId", 0)
            twin = twin if isinstance(twin, int) and not isinstance(twin, bool) else 0
            mirror.append(-1 if (twin != 0 and isinstance(gid_raw, int) and twin < gid_raw) else 1)
            layer_ids.append(int(g.get("layer", 0)) if isinstance(g.get("layer", 0), int) else 0)
            src_idx.append(gi)
        curves.CreateCurveVertexCountsAttr(Vt.IntArray(counts))
        curves.CreatePointsAttr(Vt.Vec3fArray(pts))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        curves.CreateExtentAttr(Vt.Vec3fArray([(min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))]))
        pv = UsdGeom.PrimvarsAPI(prim)
        # per-CV widths: the compiler requires reference-set widths to be empty or one per CV
        pv.CreatePrimvar("widths", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(
            Vt.FloatArray([GUIDE_DISPLAY_WIDTH] * len(pts)))
        pv.CreatePrimvar("skinprim", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform).Set(Vt.IntArray(skinprim))
        pv.CreatePrimvar("skinprimuv", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.uniform).Set(
            Vt.Vec2fArray(skinuv))
        pv.CreatePrimvar("usdGen:curveId", Sdf.ValueTypeNames.UInt64Array, UsdGeom.Tokens.uniform).Set(
            Vt.UInt64Array(ids))
        pv.CreatePrimvar("usdGen:role", Sdf.ValueTypeNames.Token, UsdGeom.Tokens.constant).Set("guide")
        pv.CreatePrimvar("usdGen:plane:mirrorSign", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform).Set(
            Vt.IntArray(mirror))
        pv.CreatePrimvar("sisir:layer", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform).Set(Vt.IntArray(layer_ids))
        pv.CreatePrimvar("sisir:curveIndex", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform).Set(
            Vt.IntArray(src_idx))
        pv.CreatePrimvar("sisir:rootUV", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.uniform).Set(
            Vt.Vec2fArray(meshuv))
        pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant).Set(
            Vt.Vec3fArray([(0.95, 0.35, 0.1)]))
        stats = {"count": len(guides), "rebound": rebound,
                 "maxFanError": max(errs_fan) if errs_fan else 0.0,
                 "maxParamError": max(errs_param) if errs_param else 0.0,
                 "maxLength": max(polyline_length(p) for _gi, _g, p in guides)}
        prim.SetCustomDataByKey("sisir:binding", to_vt(stats))
        if rebound:
            unsupported.append("guides: %d root bindings disagreed with points[0] (> %g m) and were rebound by "
                               "closest point" % (rebound, self.opt.bind_tol))
        return cpath, {"stats": stats, "idToIndex": id_to_index}

    def _mask_maps(self, index, desc_name, h, st, maps_scope, unsupported, approximations, is_card):
        """Paint / image maps for the generation, distribution and length masks."""
        nverts = len(self.mesh.points)
        density, length = [], []

        def paint(kind, values):
            dec = decode_vertex_mask(values)
            if not dec or all(v >= 1.0 for v in dec):
                return None
            if len(dec) != nverts:
                unsupported.append("%s vertex mask has %d entries but the mesh has %d vertices: ignored"
                                   % (kind, len(dec), nverts))
                return None
            key = tuple(dec)
            primvar = self.mask_primvars.get(key)
            if primvar is None:
                primvar = "mask:%s:%s" % (desc_name, kind)
                fv = Vt.FloatArray([dec[i] for i in self.mesh.face_indices])
                UsdGeom.PrimvarsAPI(self.stage.GetPrimAtPath(self.mesh_path)).CreatePrimvar(
                    primvar, Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.faceVarying).Set(fv)
                self.mask_primvars[key] = primvar
            p = maps_scope.AppendChild(kind + "Paint")
            prim = self.stage.DefinePrim(p, "UsdGenPaintMap")
            prim.CreateRelationship("usdGen:paint:surface").SetTargets([self.mesh_path])
            set_attr(prim, "usdGen:paint:primvar", "token", primvar)
            set_attr(prim, "usdGen:paint:interpolation", "token", "faceVarying")
            set_attr(prim, "usdGen:map:channel", "token", "r")
            set_attr(prim, "usdGen:map:default", "float", 1.0)
            prim.SetCustomDataByKey("sisir:nonZeroVertices", sum(1 for v in dec if v > 0))
            return p

        def image(kind, path_value):
            if not path_value:
                return None
            p = maps_scope.AppendChild(kind + "Image")
            prim = self.stage.DefinePrim(p, "UsdGenImageMap")
            prim.CreateAttribute("usdGen:map:file", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(_rel_path(path_value, self.out_dir)))
            set_attr(prim, "usdGen:map:filter", "token", "nearest")
            set_attr(prim, "usdGen:map:wrap", "token", "clamp")
            set_attr(prim, "usdGen:map:colorSpace", "token", "raw")
            set_attr(prim, "usdGen:map:channel", "token", "luminance")
            set_attr(prim, "usdGen:map:default", "float", 1.0)
            # No flip: the reference stores mesh UVs as (u, 1 - v) at import (FUN_14003dec0) and reads image row
            # v * (h - 1) from the top, i.e. exactly USD's bottom-origin lookup on the OBJ/USD st.
            prim.SetCustomDataByKey("sisir:source", os.path.basename(path_value))
            return p

        gen_ok = h.get("generationDistributionMaskInitialized",
                       "generationDistributionVertexMask" in h or "generationDistributionMaskPath" in h)
        settings_res = self._resolved_set(index)
        if gen_ok:
            for p in (paint("generation", h.get("generationDistributionVertexMask")),
                      image("generation", (settings_res or {}).get("generationDistributionMaskPath", ""))):
                if p:
                    density.append(p)
        for p in (paint("distribution", h.get("distributionVertexMask")),
                  image("distribution", ((settings_res or {}).get("settings") or {}).get("distributionMaskPath", ""))):
            if p:
                density.append(p)
        for p in (paint("length", h.get("lengthVertexMask")),
                  image("length", ((settings_res or {}).get("settings") or {}).get("lengthMaskPath", ""))):
            if p:
                length.append(p)
        if len(density) > 1:
            approximations.append("generation and distribution masks are multiplied into one scatter density map "
                                  "(reference: a root survives both tests with one hash, i.e. min, not product)")
        if density and not is_card:
            approximations.append("distribution mask thins roots at scatter; the reference keeps masked roots in "
                                  "the strand numbering and filters them at output (affects index-keyed edits and "
                                  "hash-keyed flyaways only)")
        if density and is_card:
            unsupported.append("generation/distribution masks on a card-grown set: CardSource has no density-map "
                                "input (masks authored as maps but not bound)")
            density = []
        return density, length

    def _resolved_set(self, index):
        """The hair set dict with resolved (absolute local) asset paths."""
        sets = self.scene.get("hairSets")
        if isinstance(sets, list) and sets:
            return sets[index] if 0 <= index < len(sets) and isinstance(sets[index], dict) else None
        hs = self.scene.get("hairSettings")
        return {"settings": hs} if isinstance(hs, dict) else None

    def _trim_curves(self, desc_path, set_index, bits, st):
        out = []
        for tc in self.ss.scene.get("trimCurves") or []:
            if not isinstance(tc, dict):
                continue
            pts = tc.get("points")
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            hs = tc.get("hairSet", -1)
            hs = hs if isinstance(hs, int) else -1
            if hs >= 0 and hs != set_index:
                continue
            trim = tc.get("trim") if isinstance(tc.get("trim"), dict) else {}
            li = tc.get("layer", 0)
            li = li if isinstance(li, int) else 0
            if not trim.get("allLayers", False):
                if li < 0 or not (li < len(bits) and bits[li]) or not self._layer_visible(li):
                    continue
            if hs >= 0:
                ang = _clamp(_num(trim.get("angle"), 0.0), 0.0, 89.0)
                inv = bool(trim.get("invertAngle", False))
                dist = _clamp(_num(trim.get("distance"), 0.1), 0.01, 2.0)
            else:
                ang, inv, dist = st["trimAngle"], st["trimInvertAngle"], st["trimDistance"]
            out.append(([tuple(float(c) for c in p[:3]) for p in pts], ang, inv, dist, hs >= 0))
        if not out:
            return None, 0
        path = desc_path.AppendChild("TrimCurves")
        curves = UsdGeom.BasisCurves.Define(self.stage, path)
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curves.CreatePurposeAttr(UsdGeom.Tokens.guide)
        curves.CreateCurveVertexCountsAttr(Vt.IntArray([len(c[0]) for c in out]))
        curves.CreatePointsAttr(Vt.Vec3fArray([p for c in out for p in c[0]]))
        pv = UsdGeom.PrimvarsAPI(curves.GetPrim())
        npts = sum(len(c[0]) for c in out)
        pv.CreatePrimvar("widths", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(
            Vt.FloatArray([2.5 * GUIDE_DISPLAY_WIDTH] * npts))
        if any(c[4] for c in out):
            pv.CreatePrimvar("usdGen:plane:trimAngle", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.uniform).Set(
                Vt.FloatArray([c[1] for c in out]))
            pv.CreatePrimvar("usdGen:plane:trimInvert", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform).Set(
                Vt.IntArray([1 if c[2] else 0 for c in out]))
            pv.CreatePrimvar("usdGen:plane:trimDistance", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.uniform).Set(
                Vt.FloatArray([c[3] for c in out]))
        return path, len(out)

    def _parting(self, bits):
        pts, counts, normals, all_bound = [], [], [], True
        for pc in self.ss.scene.get("partingCurves") or []:
            if not isinstance(pc, dict):
                continue
            cp = pc.get("points")
            li = pc.get("layer", 0)
            li = li if isinstance(li, int) else 0
            if not isinstance(cp, list) or len(cp) < 2:
                continue
            if not (0 <= li < len(bits) and bits[li]) or not self._layer_visible(li):
                continue
            tris, barys = pc.get("pointTris"), pc.get("pointBarys")
            ok = isinstance(tris, list) and isinstance(barys, list) and len(tris) == len(cp) == len(barys)
            for k, p in enumerate(cp):
                pts.append(tuple(float(c) for c in p[:3]))
                if ok and isinstance(tris[k], int) and 0 <= tris[k] < len(self.mesh.tri_face):
                    normals.append(fan_normal(self.mesh, tris[k], _clean_bary(barys[k])))
                else:
                    all_bound = False
            counts.append(len(cp))
        if not counts:
            return None
        return {"points": pts, "counts": counts, "normals": normals if all_bound else None}

    def _ties(self, id_to_index):
        positions, entry_counts, gidx, vidx = [], [], [], []
        for tie in self.ss.scene.get("hairTies") or []:
            if not isinstance(tie, dict) or not isinstance(tie.get("position"), list):
                continue
            n = 0
            for b in tie.get("bindings") or []:
                if not isinstance(b, dict):
                    continue
                k = id_to_index.get(b.get("curveId"))
                if k is None:
                    continue
                gidx.append(k)
                vidx.append(int(_num(b.get("pointIndex"), 0)))
                n += 1
            if n:
                positions.append(tuple(_num(x, 0.0) for x in tie["position"][:3]))
                entry_counts.append(n)
        if not positions:
            return None
        return {"positions": positions, "entryCounts": entry_counts, "guideIndices": gidx, "vertexIndices": vidx}

    def _material(self, mat_id):
        if mat_id in self.materials:
            return self.materials[mat_id]
        mats = self.ss.scene.get("renderMaterials")
        if isinstance(mats, list) and 0 <= mat_id < len(mats) and isinstance(mats[mat_id], dict):
            raw = mats[mat_id]
            res = (self.scene.get("renderMaterials") or [])[mat_id]
        else:
            raw = self.ss.scene.get("renderMaterial") if isinstance(self.ss.scene.get("renderMaterial"), dict) else {}
            res = self.scene.get("renderMaterial") or {}
        path = Sdf.Path("/World/Looks/Hair_%d" % mat_id)
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, path.AppendChild("HairStrands"))
        sh.CreateIdAttr("UsdGenHairStrands")
        inputs, look = map_hair_material(raw)
        declared = self.shader_inputs
        for name, (t, v) in inputs.items():
            sh.CreateInput(name, _SDF_TYPES[t]).Set(Gf.Vec3f(*v) if t == "color3f" else v)
        cmap = res.get("ColorMap", "") if isinstance(res, dict) else ""
        if cmap:
            reader = UsdShade.Shader.Define(self.stage, path.AppendChild("stReader"))
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex = UsdShade.Shader.Define(self.stage, path.AppendChild("colorMap"))
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(_rel_path(cmap, self.out_dir))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("baseColorMap", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        pending, authored_k = {}, []
        for names, value, t in pending_shader_inputs(raw):
            hit = next((n for n in names if declared and n in declared), None)
            if hit:
                v = Gf.Vec3f(*value) if t == "color3f" else (Gf.Vec4f(*value) if t == "float4" else value)
                sh.CreateInput(hit, _SDF_TYPES[t]).Set(v)
                authored_k.append(hit)
            else:
                pending[names[0]] = to_vt(list(value)) if isinstance(value, tuple) else value
        sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        unsupported = ["MultipleScattering/DS*: dual scattering runs on usdGen's baked hair count (selfShadow); "
                       "the density-volume settings are not converted",
                       "LOD*/Shadow*/ShadingQuality*/BeautyMask*/DepthWriteThreshold: renderer settings, not converted",
                       "MinimumHairWidth/MaximumHairWidth/HairWidthScale: display clamps, not converted"]
        if raw.get("UsePresetColorSettings") and "usePresetColors" not in authored_k:
            unsupported.append("ColorA/B/C gradients: only the highest-weight gradient (%s) is mapped to "
                               "baseColor/tipColor/colorRamp; per-strand A/B/C selection pending shader support"
                               % look.get("gradient"))
        mat.GetPrim().SetCustomDataByKey("sisir", {
            "renderMaterialId": mat_id, "material": to_vt(raw), "pendingShaderInputs": pending,
            "authoredShaderInputs": Vt.StringArray(authored_k), "unsupported": Vt.StringArray(unsupported),
            "note": "roughness = sqrt(2 v Roughness), v = Chiang(LongitudialRoughness); specular = SpecularScale/2; "
                    "colours sRGB->linear"})
        self.materials[mat_id] = (mat, look)
        return self.materials[mat_id]

    def _author_set(self, index, h, st, used_names):
        stage = self.stage
        name = _valid_name(h.get("name") or "Set_%d" % index, used_names)
        desc_path = Sdf.Path("/World/Groom").AppendChild(name)
        is_card = st["distribution"] == 4
        unsupported, approximations = [], []
        bits = self._layer_bits(h, st)
        guides = self._set_guides(bits)
        gpath, ginfo = self._author_guides(name, guides, unsupported)
        fallback_len = _num((self.ss.scene.get("guideSettings") or {}).get("defaultLength"), 0.01)

        desc = stage.DefinePrim(desc_path, "UsdGenDescription")
        desc.AddAppliedSchema("UsdGenLookAPI")
        desc.CreateRelationship("usdGen:surface").SetTargets([self.mesh_path])
        if gpath:
            desc.CreateRelationship("usdGen:guides").SetTargets([gpath])
        set_attr(desc, "usdGen:curve:basis", "token", "linear", self.schema_warnings)
        set_attr(desc, "usdGen:width:default", "float", 2.0 * st["hairThickness"], self.schema_warnings)
        if not h.get("visible", True):
            UsdGeom.Imageable(desc).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        maps_scope = desc_path.AppendChild("Maps")
        UsdGeom.Scope.Define(stage, maps_scope)
        density_maps, length_maps = self._mask_maps(index, name, h, st, maps_scope, unsupported,
                                                    approximations, is_card)
        if not stage.GetPrimAtPath(maps_scope).GetChildren():
            stage.RemovePrim(maps_scope)
        trim_path, trim_count = self._trim_curves(desc_path, index, bits, st)
        parting = None if is_card else self._parting(bits)
        ties = self._ties(ginfo.get("idToIndex", {})) if gpath else None

        # index-keyed edits
        overrides = None
        co = h.get("clumpOverride")
        co_n = h.get("clumpOverrideHairCount", 0)
        co_n = co_n if isinstance(co_n, int) else 0
        if isinstance(co, list) and co and co_n > 0:
            arr = [INT_MIN] * co_n
            for e in co:
                if isinstance(e, dict) and isinstance(e.get("i"), int) and 0 <= e["i"] < co_n:
                    arr[e["i"]] = int(_num(e.get("v"), INT_MIN))
            overrides = (arr, co_n)
            if st["clumpType"] == 4:
                unsupported.append("clumpOverride (%d entries) on a braid set: the Braid operator has no override "
                                   "slot" % len(co))
            elif st["clumpType"] == 0:
                approximations.append("clumpOverride present but clumpType is 0: inert (as in the reference)")
        del_n = h.get("deletedStrandsHairCount", 0)
        dels = h.get("deletedStrands")
        if isinstance(dels, list) and dels and isinstance(del_n, int) and del_n > 0:
            unsupported.append("deletedStrands: %d of %d generated strands deleted by index; no engine slot, the "
                               "strands are NOT removed" % (len(dels), del_n))
        if st["clumpType"] == 4 and st["subClumpsEnabled"] and st["braidTailLength"] > 1e-4:
            unsupported.append("sub-clumps of the braid tail: the Braid operator has no sub-clump slot")
        if st["clumpType"] == 3:
            approximations.append("clumpType 3 (random experimental) -> clump:centers guideSeeded, decoded only "
                                  "for guideInterpolation 0 / single guide")
        if st["clumpType"] in (1, 3, 4) and not gpath:
            approximations.append("clumpType %d needs guides; the set has none" % st["clumpType"])
        if st["pushOutEnabled"]:
            approximations.append("push-out: exact closest point instead of the reference 48^3 SDF grid")
        if is_card:
            approximations.append("card set: the reference re-snaps drifted roots to the scalp at output; no "
                                  "engine equivalent after the last operator")
            if st["resolutionType"] == 1:
                approximations.append("card set distance resolution: reference length from the longest spine "
                                      "(the reference uses a card-texture length)")
        inert = sorted(k for k in INERT_KEYS if st.get(k) != HAIR_SETTINGS_DEFAULTS.get(k))
        if st["lengthBreakup"] > 0:
            approximations.append("lengthBreakup %.3g: active only in texture-module tile builds; inert here"
                                  % st["lengthBreakup"])

        ctx = {"setIndex": index, "isCard": is_card, "guideCount": len(guides), "guidesPath": str(gpath or ""),
               "maxGuideLength": ginfo.get("stats", {}).get("maxLength", 0.0), "fallbackLength": fallback_len,
               "trimCurveCount": trim_count, "tieCount": len(ties["positions"]) if ties else 0,
               "partingCurveCount": len(parting["counts"]) if parting else 0}
        card_params = None
        if is_card:
            if self.card_prims is None:
                self.card_prims = self.cards_mod.author_cards(stage, self.scene, "/World/Groom/Cards",
                                                              str(self.mesh_path), 1.0)
                self.log("cards.py (%s): authored %s" % (self.cards_src, ", ".join(sorted(self.card_prims.values()))))
            card_params = self.cards_mod.card_source_params(h, self.scene, 1.0)
            ctx["cardMaxLength"] = self._card_lmax(st)
            key = self.cards_mod.card_source_key(h) if hasattr(self.cards_mod, "card_source_key") else h.get("cardSource")
            if not key or key not in self.card_prims:
                unsupported.append("cardSource %r has no strand library in cardHairSources" % key)

            def author_gen(stage_, path, h=h):
                return self.cards_mod.author_card_source(stage_, str(path), h, self.scene, self.card_prims, 1.0)
        else:
            author_gen = None
        data = {"densityMaps": density_maps, "lengthMaps": length_maps, "trimCurves": trim_path,
                "parting": parting, "ties": ties, "clumpOverrides": overrides, "cardSourceAuthor": author_gen}
        ops, res = derive_ops(st, ctx, data)
        if is_card and card_params and "usdGen:cvCount" in card_params and card_params["usdGen:cvCount"] != res["steps"]:
            approximations.append("cvCount: cards.py %d vs importer %d (cards.py value kept)"
                                  % (card_params["usdGen:cvCount"], res["steps"]))

        ops_scope = desc_path.AppendChild("Ops")
        UsdGeom.Scope.Define(stage, ops_scope)
        authored = []
        for op in reversed(ops):          # bottom sibling runs first: author the last stage first
            if self.opt.active_ops_only and not op.enabled:
                continue
            self._author_op(ops_scope.AppendChild(op.name), op)
            authored.append(op)

        mat, look = self._material(int(_clamp(int(_num(h.get("renderMaterialId"), 0)), 0, 9)))
        UsdShade.MaterialBindingAPI.Apply(desc).Bind(mat)
        if look:
            set_attr(desc, "usdGen:look:rootColor", "color3f", Gf.Vec3f(*look["rootColor"]))
            set_attr(desc, "usdGen:look:tipColor", "color3f", Gf.Vec3f(*look["tipColor"]))
            set_attr(desc, "usdGen:look:rampExponent", "float", float(look["rampExponent"]))

        set_keys = {k: v for k, v in h.items() if k not in ("settings", "distributionVertexMask", "lengthVertexMask",
                                                           "generationDistributionVertexMask", "_legacy")}
        cd = {"index": index, "name": str(h.get("name", "")), "settings": to_vt(h.get("settings") or {}),
              "hairSet": to_vt(set_keys), "context": to_vt(ctx), "unsupported": Vt.StringArray(unsupported),
              "approximations": Vt.StringArray(approximations), "inertSettings": Vt.StringArray(inert),
              "steps": res["steps"]}
        desc.SetCustomDataByKey("sisir", cd)

        target = None
        if not is_card:
            if st["distribution"] == 3:
                target = len(guides)
            elif self.mesh_area > 1e-8 and st["hairDensity"] > 0:
                target = int(_clamp(_lround(_f32(self.mesh_area * st["hairDensity"])), 1, 2000000))
        summ = {"index": index, "name": str(h.get("name", "")), "path": str(desc_path),
                "distribution": DISTRIBUTION_NAMES.get(st["distribution"]), "guides": len(guides),
                "guideBinding": ginfo.get("stats"), "steps": res["steps"], "rootTarget": target,
                "ops": [(o.name, o.type_name, o.enabled) for o in ops if o in authored],
                "densityMaps": [str(p) for p in density_maps], "lengthMaps": [str(p) for p in length_maps],
                "unsupported": unsupported, "approximations": approximations,
                "cardParams": card_params, "clumpCount": st["clumpRandomCount"] if st["clumpType"] in (2, 3) else None}
        self.summary["sets"].append(summ)
        self._print_set(summ, st)

    def _card_lmax(self, st):
        """Longest spine x its params.length.max among the set's cards (distance-mode resolution only)."""
        layer = st["cardSourceLayer"]
        lmax = 0.0
        for c in self.ss.scene.get("cards") or []:
            if not isinstance(c, dict) or (layer != -1 and c.get("layer") != layer):
                continue
            pts = c.get("points")
            if isinstance(pts, list) and len(pts) >= 2:
                try:
                    L = polyline_length([tuple(float(x) for x in p[:3]) for p in pts])
                except (TypeError, ValueError):
                    continue
                lm = _clamp(_num(((c.get("params") or {}).get("length") or {}).get("max"), 1.0), 0.01, 2.0)
                lmax = max(lmax, L * lm)
        return lmax

    def _author_op(self, path, op: OpSpec):
        if op.custom is not None:
            prim = op.custom(self.stage, path)
        else:
            prim = self.stage.DefinePrim(path, op.type_name)
        set_attr(prim, "usdGen:enabled", "bool", op.enabled, self.schema_warnings)
        for name, t, value, _kind in op.attrs:
            if op.custom is not None and prim.GetAttribute(name).HasAuthoredValue():
                continue   # the external author (cards.py) owns its values
            set_attr(prim, name, t, value, self.schema_warnings)
        for rel, targets in op.rels.items():
            if targets:
                prim.CreateRelationship(rel).SetTargets([Sdf.Path(str(t)) for t in targets])
        return prim

    def _print_set(self, s, st):
        b = s["guideBinding"] or {}
        ops = " -> ".join(n + ("" if e else "(off)") for n, _t, e in s["ops"])
        self.log("[%d] %-12s -> %s" % (s["index"], s["name"], s["path"]))
        self.log("     distribution=%s guides=%d cv=%d%s%s" % (
            s["distribution"], s["guides"], s["steps"],
            (" rootTarget=%d" % s["rootTarget"]) if s["rootTarget"] is not None else "",
            (" clumpCount=%d" % s["clumpCount"]) if s["clumpCount"] is not None else ""))
        if b:
            self.log("     guide roots: max |fan bary - points[0]| = %.3g m, max |skinprim/uv - points[0]| = %.3g m, "
                     "rebound=%d" % (b.get("maxFanError", 0), b.get("maxParamError", 0), b.get("rebound", 0)))
        self.log("     ops (exec order): %s" % ops)
        if s["densityMaps"] or s["lengthMaps"]:
            self.log("     maps: density=%s length=%s" % ([p.rsplit("/", 1)[-1] for p in s["densityMaps"]],
                                                         [p.rsplit("/", 1)[-1] for p in s["lengthMaps"]]))
        for u in s["unsupported"]:
            self.log("     unsupported: " + u)


# ============================================================================= preset application

def apply_preset_to_stage(stage_path, preset_path, desc_path, output=None, quiet=False):
    """Apply a hair preset to an existing converted description (re-derives every op attribute)."""
    register_usdgen_schema()
    stage = Usd.Stage.Open(stage_path)
    if stage is None:
        raise SirFormatError("cannot open %s" % stage_path)
    if ":" in desc_path or "\\" in desc_path:
        raise SirFormatError("--description %r is not a prim path (Git Bash rewrites /World/...: set "
                             "MSYS_NO_PATHCONV=1)" % desc_path)
    desc = stage.GetPrimAtPath(desc_path)
    if not desc:
        raise SirFormatError("no prim at %s" % desc_path)
    cd = from_vt(desc.GetCustomDataByKey("sisir") or {})
    if "settings" not in cd or "context" not in cd:
        raise SirFormatError("%s has no customData['sisir'] settings/context (not converted by sir2usd)" % desc_path)
    preset = read_hair_preset(preset_path)
    new_raw, applied, skipped = apply_hair_preset(cd["settings"], preset, current_module=1)
    st = settings_for(new_raw)
    ctx = cd["context"]
    ops, res = derive_ops(st, ctx, None)
    ops_scope = desc.GetPath().AppendChild("Ops")
    missing, warn = [], []
    for op in ops:
        prim = stage.GetPrimAtPath(ops_scope.AppendChild(op.name))
        if not prim:
            if op.enabled:
                missing.append(op.name)
            continue
        set_attr(prim, "usdGen:enabled", "bool", op.enabled, warn)
        for name, t, value, kind in op.attrs:
            if kind == "setting":
                set_attr(prim, name, t, value, warn)
        if "usdGen:guides" in op.rels:
            prim.CreateRelationship("usdGen:guides").SetTargets([Sdf.Path(p) for p in op.rels["usdGen:guides"]])
        elif prim.GetRelationship("usdGen:guides").HasAuthoredTargets():
            prim.RemoveProperty("usdGen:guides")
    set_attr(desc, "usdGen:width:default", "float", 2.0 * st["hairThickness"], warn)
    history = list(cd.get("presetHistory", []) or [])
    history.append("%s (applied %d keys, skipped %d)" % (preset["name"], len(applied), len(skipped)))
    desc.SetCustomDataByKey("sisir:settings", to_vt(new_raw))
    desc.SetCustomDataByKey("sisir:presetHistory", Vt.StringArray(history))
    desc.SetCustomDataByKey("sisir:steps", res["steps"])
    if missing:
        desc.SetCustomDataByKey("sisir:presetMissingOps", Vt.StringArray(missing))
    if output:
        stage.GetRootLayer().Export(output)
    else:
        stage.GetRootLayer().Save()
    if not quiet:
        print("preset %r -> %s: applied %d keys, skipped %d (%s)%s" % (
            preset["name"], desc_path, len(applied), len(skipped), ", ".join(skipped) or "-",
            ("; enabled stages missing from the stack: " + ", ".join(missing)) if missing else ""))
    return {"applied": applied, "skipped": skipped, "missing": missing}


# ============================================================================= CLI

def convert(scene_path, out_path, **kw):
    """Library entry point: convert(scene, out, sets=None, extract=True, ...) -> summary dict."""
    return Converter(scene_path, out_path, ConvertOptions(**kw)).run()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert a Sisir .sir/.json scene into a usdGen USD stage, or apply "
                                             "a hair preset to a converted description.")
    ap.add_argument("paths", nargs="+", help="SCENE OUT  (convert)  |  STAGE  (with --preset)")
    ap.add_argument("--sets", nargs="+", help="hair sets to convert (index or name); default all")
    ap.add_argument("--no-extract", action="store_true", help="do not extract/copy archive assets")
    ap.add_argument("--active-ops-only", action="store_true",
                    help="omit stages the reference skips for this set (default: author them disabled)")
    ap.add_argument("--include-hidden-layers", action="store_true",
                    help="let guides/trim/parting curves of hidden layers drive hair")
    ap.add_argument("--cards-module", help="path to cards.py (default: next to this file, or $SISIR_CARDS_MODULE)")
    ap.add_argument("--shader-defs", help="shaderDefs.usda declaring UsdGenHairStrands inputs")
    ap.add_argument("--summary-json", help="write the per-set conversion summary as JSON")
    ap.add_argument("--preset", help="hair preset JSON to apply (needs --description)")
    ap.add_argument("--description", help="description prim path for --preset")
    ap.add_argument("--output", help="with --preset: write here instead of saving in place")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.preset:
            if not a.description or len(a.paths) != 1:
                ap.error("--preset needs --description and exactly one STAGE path")
            apply_preset_to_stage(a.paths[0], a.preset, a.description, a.output, a.quiet)
            return 0
        if len(a.paths) != 2:
            ap.error("convert needs SCENE and OUT")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summ = convert(a.paths[0], a.paths[1], sets=a.sets, extract=not a.no_extract,
                           active_ops_only=a.active_ops_only, cards_module=a.cards_module,
                           include_hidden_layers=a.include_hidden_layers, shader_defs=a.shader_defs,
                           quiet=a.quiet)
        for w in caught:
            summ["warnings"].append(str(w.message))
            if not a.quiet:
                print("warning: %s" % w.message)
        if a.summary_json:
            with open(a.summary_json, "w", encoding="utf-8") as f:
                json.dump(summ, f, indent=2, default=str)
        return 0
    except (OSError, SirFormatError) as e:
        print("sir2usd: error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
