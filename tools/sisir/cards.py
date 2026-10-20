# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
"""Card authoring for Sisir scenes imported into usdGen.

Authors the two reference curve sets a ``UsdGenCardSource`` operator reads:

* one card spine prim (``<root>/cards``): every Sisir card as a linear
  BasisCurves curve whose per-card parameters travel as uniform
  ``primvars:usdGen:plane:<name>`` primvars;
* one tile strand library prim per ``cardHairSources`` key
  (``<root>/cardStrands_<key>``): the baked tile-space strands, tagged with
  their ``(uvId, tile)`` entry.

The exact plane list (names, types, units, clamps) is the contract in
``card-planes.md`` (next to this file); keep the two in sync.
All Sisir cm / percent values are converted here: the engine reads scene
units, degrees and fractions only.

Stdlib + ``pxr`` only (the importer runs Python with ``-S``).

Public API
----------
author_cards(stage, scene, root_path, mesh_path, scale=1.0) -> dict
card_source_params(hair_set, scene, scale=1.0) -> dict
card_source_key(hair_set) -> str | None
author_card_source(stage, op_path, hair_set, scene, card_prims, scale=1.0) -> Usd.Prim
CARD_SOURCE_ATTR_TYPES: dict attr name -> Sdf value type name
"""

from __future__ import annotations

import base64
import math
import struct
import warnings

from pxr import Sdf, Usd, UsdGeom, Vt

__all__ = [
    "author_cards",
    "card_source_params",
    "card_source_key",
    "author_card_source",
    "decode_card_hair_source",
    "card_records",
    "CARD_SOURCE_ATTR_TYPES",
    "PLANE_PREFIX",
]

PLANE_PREFIX = "usdGen:plane:"

# cardParams constructor defaults (sir_format.md §2.7.1) for keys a file omits.
_PARAM_DEFAULTS = {
    "cardType": 0, "orientation": 2, "widthSegments": 1, "sink": 0.0,
    "shape": {"width": 0.05, "root": 0.5, "mid": 0.5, "midShift": 0.5, "tip": 0.5,
              "crossWidth": 1.0, "tubeFlatten": 0.0, "centerRidge": 0.0,
              "ridgeRootFlatten": 0.0},
    "braid": {"resolution": 64, "frequency": 30.0, "width": 0.3, "flatness": 0.2,
              "knotTightness": 1.0, "fadeIn": 0.0, "fadeOut": 0.0, "rootSpread": 0.0,
              "tipSpread": 0.0, "tailLength": 0.1, "tailRelax": 1.0, "tailCinch": 0.5},
    "twist": {"root": 0.0, "mid": 0.0, "midShift": 0.5, "tip": 0.0},
    "length": {"min": 0.01, "max": 1.0, "randomize": 0.0},
    "frizz": {"preserveBraidKnots": False, "amount": 0.0, "scale": 5.0, "scaleZ": 5.0,
              "noise": 0.0, "root": 0.0, "mid": 0.5, "midShift": 0.5, "tip": 1.0},
    "pushOut": {"enabled": False, "root": 0.0, "tip": 1.0, "smoothness": 0.5},
}

# hairSettings defaults (hairsettings_offsets.md) for the keys read here.
_HS_DEFAULTS = {
    "resolutionType": 0, "hairResolution": 40, "resolutionDistance": 1.0,
    "cardSourceUvId": -1, "cardSourceTile": -1, "cardSourceLayer": -1,
    "cardSourceBraid": False, "cardRootFlatten": 0.0, "cardSmooth": True,
    "cardStrandAmount": 1.0, "braidWidth": 1.0, "braidFlatness": 0.55,
}

CARD_SOURCE_ATTR_TYPES = {
    "usdGen:cardSource:layer": "int",
    "usdGen:cardSource:uvId": "int",
    "usdGen:cardSource:tile": "int",
    "usdGen:cardSource:braid": "bool",
    "usdGen:cardSource:amount": "float",
    "usdGen:cardSource:rootFlatten": "float",
    "usdGen:cardSource:smooth": "bool",
    "usdGen:cardSource:maxStrands": "int",
    "usdGen:cardSource:braidWidth": "float",
    "usdGen:cardSource:braidFlatness": "float",
    "usdGen:cvCount": "int",
}

_SDF_TYPES = {"int": Sdf.ValueTypeNames.Int, "bool": Sdf.ValueTypeNames.Bool,
              "float": Sdf.ValueTypeNames.Float}


# ----------------------------------------------------------------------------- helpers

def _num(v, default):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return v
    return default


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _int32(u):
    """Low 32 bits of an unsigned value as a signed int32 bit pattern."""
    u &= 0xFFFFFFFF
    return u - (1 << 32) if u >= (1 << 31) else u


def _u64(v):
    try:
        return int(v) & 0xFFFFFFFFFFFFFFFF
    except (TypeError, ValueError):
        return 0


def _param(params, group, key=None):
    """params[group][key] with the constructor default when absent."""
    if key is None:
        v = params.get(group) if isinstance(params, dict) else None
        return _num(v, _PARAM_DEFAULTS[group])
    sub = params.get(group) if isinstance(params, dict) else None
    v = sub.get(key) if isinstance(sub, dict) else None
    d = _PARAM_DEFAULTS[group][key]
    if isinstance(d, bool):
        return bool(v) if isinstance(v, (bool, int)) else d
    return _num(v, d)


def _hs(settings, key):
    v = settings.get(key) if isinstance(settings, dict) else None
    d = _HS_DEFAULTS[key]
    if isinstance(d, bool):
        return bool(v) if isinstance(v, (bool, int)) else d
    return _num(v, d)


def _wrap(t, n):
    return ((int(t) % n) + n) % n if n > 0 else -1


def _uv_tiles(scene, uv_id):
    """cardUvSets[uvId].uvTiles (the builder reads cardUvSets, card_geometry.md §9)."""
    sets = scene.get("cardUvSets")
    if not isinstance(sets, list) or not (0 <= uv_id < min(10, len(sets))):
        return []
    s = sets[uv_id]
    tiles = s.get("uvTiles") if isinstance(s, dict) else None
    return [t for t in tiles if isinstance(t, dict)] if isinstance(tiles, list) else []


def _bias(tiles, t):
    if not (0 <= t < len(tiles)):
        return 0.5
    b = tiles[t].get("longitudinalBias", 0.5)
    if not isinstance(b, (int, float)) or not math.isfinite(float(b)):
        return 0.5
    return _clamp(float(b), 0.05, 0.95)


def _vertex_mask(values):
    if not isinstance(values, list) or not values:
        return None
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
        out.append(_clamp(f, 0.0, 1.0))
    return out


def _fan_triangles(stage, mesh_path):
    """Fan triangles (point indices) of the bound mesh in face order, or None."""
    if stage is None or not mesh_path:
        return None
    prim = stage.GetPrimAtPath(Sdf.Path(str(mesh_path)))
    mesh = UsdGeom.Mesh(prim) if prim else None
    if not mesh:
        return None
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    tris = []
    base = 0
    for k in counts:
        k = int(k)
        for j in range(1, k - 1):
            if base + j + 1 < len(indices):
                tris.append((int(indices[base]), int(indices[base + j]), int(indices[base + j + 1])))
        base += max(k, 0)
    return tris


def _mask_at_root(mask, tris, card):
    """Barycentric blend of a per-vertex mask at the card root (1 when unknown)."""
    if mask is None or tris is None:
        return 1.0
    tri = card.get("rootTri", -1)
    if not isinstance(tri, int) or not (0 <= tri < len(tris)):
        return 1.0
    bary = card.get("rootBary") or [1.0, 0.0, 0.0]
    b = [_clamp(float(_num(x, 0.0)), 0.0, 1.0) for x in list(bary)[:3]] + [0.0] * (3 - len(bary[:3]))
    s = sum(b)
    b = [x / s for x in b] if s > 1e-8 else [1.0, 0.0, 0.0]
    v = 0.0
    for w, vi in zip(b, tris[tri]):
        v += w * (mask[vi] if 0 <= vi < len(mask) else 1.0)
    return _clamp(v, 0.0, 1.0)


def decode_card_hair_source(entry):
    """cardHairSources[<key>] -> {'counts', 'points' (flat xyz floats), 'offsets', 'tiles'}."""
    def b64(s):
        if not isinstance(s, str):
            return b""
        try:
            return base64.b64decode(s, validate=False)
        except Exception:
            return b""
    cb, pb = b64(entry.get("counts")), b64(entry.get("points"))
    counts = list(struct.unpack("<%di" % (len(cb) // 4), cb[: len(cb) // 4 * 4]))
    flat = struct.unpack("<%df" % (len(pb) // 4), pb[: len(pb) // 4 * 4])
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + max(int(c), 0))
    tiles = [[int(x) for x in t] for t in entry.get("tiles", []) or []
             if isinstance(t, list) and len(t) == 4]
    return {"counts": counts, "points": flat, "offsets": offsets, "tiles": tiles}


def card_source_key(hair_set):
    """The hair set's cardHairSources key (16 hex), or None."""
    key = hair_set.get("cardSource") if isinstance(hair_set, dict) else None
    return key if isinstance(key, str) and key else None


def _settings(hair_set):
    s = hair_set.get("settings") if isinstance(hair_set, dict) else None
    if not isinstance(s, dict):
        s = hair_set.get("hairSettings") if isinstance(hair_set, dict) else None
    return s if isinstance(s, dict) else {}


# ----------------------------------------------------------------------------- card records

# (plane name, kind) in authoring order; kind: "int" | "float" | "float3".
_CARD_PLANES = [
    ("sceneIndex", "int"), ("cardIdLo", "int"), ("cardIdHi", "int"),
    ("twinIdLo", "int"), ("twinIdHi", "int"), ("mirrorPlane", "int"),
    ("layer", "int"), ("uvId", "int"), ("tile", "int"), ("tile2", "int"), ("tile3", "int"),
    ("vBias", "float"), ("vBias2", "float"), ("vBias3", "float"),
    ("cardType", "int"), ("orientation", "int"), ("widthSegments", "int"),
    ("rootTri", "int"), ("rootBary", "float3"), ("erased", "int"),
    ("distributionMask", "float"), ("lengthMask", "float"),
    ("width", "float"), ("widthRoot", "float"), ("widthMid", "float"),
    ("widthMidShift", "float"), ("widthTip", "float"), ("crossWidth", "float"),
    ("tubeFlatten", "float"), ("centerRidge", "float"), ("ridgeRootFlatten", "float"),
    ("sink", "float"),
    ("twistRoot", "float"), ("twistMid", "float"), ("twistMidShift", "float"), ("twistTip", "float"),
    ("lengthMin", "float"), ("lengthMax", "float"), ("lengthRandomize", "float"),
    ("frizzAmount", "float"), ("frizzFrequency", "float"), ("frizzFrequencyZ", "float"),
    ("frizzNoise", "float"), ("frizzRoot", "float"), ("frizzMid", "float"),
    ("frizzMidShift", "float"), ("frizzTip", "float"), ("frizzPreserveBraidKnots", "int"),
    ("pushOutEnabled", "int"), ("pushOutRoot", "float"), ("pushOutTip", "float"),
    ("pushOutSmoothness", "float"),
    ("braidResolution", "int"), ("braidFrequency", "float"), ("braidWidth", "float"),
    ("braidFlatness", "float"), ("braidKnotTightness", "float"), ("braidFadeIn", "float"),
    ("braidFadeOut", "float"), ("braidRootSpread", "float"), ("braidTipSpread", "float"),
    ("braidTailLength", "float"), ("braidTailRelax", "float"), ("braidTailCinch", "float"),
]


def card_records(scene, tris=None, scale=1.0):
    """[(points [(x,y,z)], {plane: value})] for every card with >= 2 points, scene order."""
    layers = scene.get("layers") if isinstance(scene.get("layers"), list) else []
    out = []
    warned_image_mask = False
    for ci, card in enumerate(scene.get("cards") or []):
        if not isinstance(card, dict):
            continue
        pts = [p for p in (card.get("points") or []) if isinstance(p, list) and len(p) >= 3]
        if len(pts) < 2:
            continue
        li = card.get("layer", 0)
        li = li if isinstance(li, int) and 0 <= li < len(layers) else 0
        layer = layers[li] if layers and isinstance(layers[li], dict) else {}
        uv_id = layer.get("cardUvId", -1)
        uv_id = uv_id if isinstance(uv_id, int) else -1
        tiles = _uv_tiles(scene, uv_id) if 0 <= uv_id <= 9 else []
        n = len(tiles)
        t1 = _wrap(card.get("uvTile", 0) if isinstance(card.get("uvTile", 0), int) else 0, n)
        def extra(key):
            v = card.get(key, -1)
            return _wrap(v, n) if isinstance(v, int) and v >= 0 and n > 0 else t1
        t2, t3 = extra("uvTile2"), extra("uvTile3")
        if not warned_image_mask and (layer.get("cardDistributionMask") or layer.get("cardLengthMask")):
            warnings.warn("cards.py: image card masks are not evaluated (vertex masks are); using 1")
            warned_image_mask = True
        P = card.get("params") if isinstance(card.get("params"), dict) else {}
        cid, twin = _u64(card.get("id", 0)), _u64(card.get("mirrorTwinId", 0))
        bary = card.get("rootBary") or [0.0, 0.0, 0.0]
        bary = [float(_num(x, 0.0)) for x in list(bary)[:3]] + [0.0] * max(0, 3 - len(bary))
        rt = card.get("rootTri", -1)
        f = {
            "sceneIndex": ci,
            "cardIdLo": _int32(cid), "cardIdHi": _int32(cid >> 32),
            "twinIdLo": _int32(twin), "twinIdHi": _int32(twin >> 32),
            "mirrorPlane": int(_num(card.get("mirrorPairPlane", 0), 0)),
            "layer": li, "uvId": uv_id if 0 <= uv_id <= 9 else -1,
            "tile": t1, "tile2": t2, "tile3": t3,
            "vBias": _bias(tiles, t1), "vBias2": _bias(tiles, t2), "vBias3": _bias(tiles, t3),
            "cardType": int(_clamp(int(_param(P, "cardType")), 0, 3)),
            "orientation": int(_clamp(int(_param(P, "orientation")), 0, 3)),
            "widthSegments": int(_clamp(int(_param(P, "widthSegments")), 1, 32)),
            "rootTri": rt if isinstance(rt, int) else -1,
            "rootBary": tuple(bary),
            "erased": 1 if card.get("erased") else 0,
            "distributionMask": _mask_at_root(_vertex_mask(layer.get("cardDistributionVertexMask")), tris, card),
            "lengthMask": _mask_at_root(_vertex_mask(layer.get("cardLengthVertexMask")), tris, card),
            "width": _clamp(float(_param(P, "shape", "width")), 0.001, 0.5) * scale,
            "widthRoot": _clamp(float(_param(P, "shape", "root")), 0.0, 1.0),
            "widthMid": _clamp(float(_param(P, "shape", "mid")), 0.0, 1.0),
            "widthMidShift": _clamp(float(_param(P, "shape", "midShift")), 0.1, 0.9),
            "widthTip": _clamp(float(_param(P, "shape", "tip")), 0.0, 1.0),
            "crossWidth": _clamp(float(_param(P, "shape", "crossWidth")), 0.05, 1.0),
            "tubeFlatten": _clamp(float(_param(P, "shape", "tubeFlatten")), 0.0, 0.98),
            "centerRidge": _clamp(float(_param(P, "shape", "centerRidge")), -1.0, 1.0),
            "ridgeRootFlatten": _clamp(float(_param(P, "shape", "ridgeRootFlatten")), 0.0, 1.0),
            "sink": _clamp(float(_param(P, "sink")), 0.0, 0.05) * scale,
            "twistRoot": _clamp(float(_param(P, "twist", "root")), -180.0, 180.0),
            "twistMid": _clamp(float(_param(P, "twist", "mid")), -180.0, 180.0),
            "twistMidShift": _clamp(float(_param(P, "twist", "midShift")), 0.1, 0.9),
            "twistTip": _clamp(float(_param(P, "twist", "tip")), -180.0, 180.0),
            "lengthMin": _clamp(float(_param(P, "length", "min")), 0.01, 2.0),
            "lengthMax": _clamp(float(_param(P, "length", "max")), 0.01, 2.0),
            "lengthRandomize": _clamp(float(_param(P, "length", "randomize")), 0.0, 1.0),
            "frizzAmount": _clamp(float(_param(P, "frizz", "amount")), 0.0, 100.0) / 100.0,
            "frizzFrequency": _clamp(float(_param(P, "frizz", "scale")), 0.0, 100.0) / 100.0 * 10.0,
            "frizzFrequencyZ": _clamp(float(_param(P, "frizz", "scaleZ")), 0.0, 100.0) / 100.0 * 10.0,
            "frizzNoise": _clamp(float(_param(P, "frizz", "noise")), 0.0, 100.0) / 100.0,
            "frizzRoot": _clamp(float(_param(P, "frizz", "root")), 0.0, 1.0),
            "frizzMid": _clamp(float(_param(P, "frizz", "mid")), 0.0, 1.0),
            "frizzMidShift": _clamp(float(_param(P, "frizz", "midShift")), 0.1, 0.9),
            "frizzTip": _clamp(float(_param(P, "frizz", "tip")), 0.0, 1.0),
            "frizzPreserveBraidKnots": 1 if _param(P, "frizz", "preserveBraidKnots") else 0,
            "pushOutEnabled": 1 if _param(P, "pushOut", "enabled") else 0,
            "pushOutRoot": _clamp(float(_param(P, "pushOut", "root")), 0.0, 10.0) * 0.01 * scale,
            "pushOutTip": _clamp(float(_param(P, "pushOut", "tip")), 0.0, 10.0) * 0.01 * scale,
            "pushOutSmoothness": _clamp(float(_param(P, "pushOut", "smoothness")), 0.0, 1.0),
            "braidResolution": int(_clamp(int(_param(P, "braid", "resolution")), 16, 16384)),
            "braidFrequency": _clamp(float(_param(P, "braid", "frequency")), 0.0, 1000.0),
            "braidWidth": _clamp(float(_param(P, "braid", "width")), 0.0, 5.0),
            "braidFlatness": _clamp(float(_param(P, "braid", "flatness")), 0.0, 1.0),
            "braidKnotTightness": _clamp(float(_param(P, "braid", "knotTightness")), 0.0, 1.0),
            "braidFadeIn": _clamp(float(_param(P, "braid", "fadeIn")), 0.0, 1.0),
            "braidFadeOut": _clamp(float(_param(P, "braid", "fadeOut")), 0.0, 1.0),
            "braidRootSpread": _clamp(float(_param(P, "braid", "rootSpread")), 0.0, 2.0),
            "braidTipSpread": _clamp(float(_param(P, "braid", "tipSpread")), 0.0, 2.0),
            "braidTailLength": _clamp(float(_param(P, "braid", "tailLength")), 0.0, 0.9),
            "braidTailRelax": _clamp(float(_param(P, "braid", "tailRelax")), 0.0, 1.0),
            "braidTailCinch": _clamp(float(_param(P, "braid", "tailCinch")), 0.0, 1.0),
        }
        pts = [(float(p[0]) * scale, float(p[1]) * scale, float(p[2]) * scale) for p in pts]
        out.append((pts, f))
    return out


# ----------------------------------------------------------------------------- authoring

def _curves(stage, path):
    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curves.CreatePurposeAttr(UsdGeom.Tokens.guide)
    return curves


def _uniform(prim, name, kind, values):
    types = {"int": Sdf.ValueTypeNames.IntArray, "float": Sdf.ValueTypeNames.FloatArray,
             "float3": Sdf.ValueTypeNames.Float3Array}
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(PLANE_PREFIX + name, types[kind],
                                                 UsdGeom.Tokens.uniform)
    if kind == "int":
        pv.Set(Vt.IntArray([int(v) for v in values]))
    elif kind == "float":
        pv.Set(Vt.FloatArray([float(v) for v in values]))
    else:
        pv.Set(Vt.Vec3fArray([tuple(float(x) for x in v) for v in values]))


def _set_curves(curves, strands):
    counts = [len(s) for s in strands]
    pts = [p for s in strands for p in s]
    curves.CreateCurveVertexCountsAttr(Vt.IntArray(counts))
    curves.CreatePointsAttr(Vt.Vec3fArray(pts))
    if pts:
        lo = [min(p[k] for p in pts) for k in range(3)]
        hi = [max(p[k] for p in pts) for k in range(3)]
        curves.CreateExtentAttr(Vt.Vec3fArray([tuple(lo), tuple(hi)]))


def _synth_tiles(scene, key, src):
    """Legacy sources without tiles: [cardSourceUvId, cardSourceTile, 0, n] of the first user."""
    if src["tiles"]:
        return src["tiles"]
    for h in scene.get("hairSets") or []:
        if card_source_key(h) == key:
            s = _settings(h)
            return [[int(_hs(s, "cardSourceUvId")), int(_hs(s, "cardSourceTile")), 0, len(src["counts"])]]
    return []


def author_cards(stage, scene, root_path, mesh_path, scale=1.0):
    """Author the card spine prim and one strand library prim per cardHairSources key.

    stage     : Usd.Stage to author into (the current edit target).
    scene     : the scene.json dict.
    root_path : parent prim path (created as an Xform/Scope if missing).
    mesh_path : the scalp mesh (re-indexed to Sisir vertices, reference face order);
                used to evaluate per-vertex card masks at the card roots. May be None.
    scale     : scene-unit multiplier for lengths (1 when the importer authors metres).

    Returns {'cards': spine prim path, <key>: library prim path, ...} (paths as strings).
    """
    root = Sdf.Path(str(root_path))
    if not stage.GetPrimAtPath(root):
        UsdGeom.Scope.Define(stage, root)
    if scene.get("cardTrimCurves"):
        warnings.warn("cards.py: cardTrimCurves are not transported (trim surfaces not decoded)")
    result = {}

    # --- spine prim: every card, scene order -----------------------------------------
    records = card_records(scene, _fan_triangles(stage, mesh_path), scale)
    spine_path = root.AppendChild("cards")
    spine = _curves(stage, spine_path)
    _set_curves(spine, [r[0] for r in records])
    for name, kind in _CARD_PLANES:
        _uniform(spine.GetPrim(), name, kind, [r[1][name] for r in records])
    result["cards"] = str(spine_path)

    # --- strand libraries ----------------------------------------------------------------
    sources = scene.get("cardHairSources")
    if isinstance(sources, dict):
        for key in sorted(sources):
            entry = sources[key]
            if not isinstance(entry, dict):
                continue
            src = decode_card_hair_source(entry)
            counts, flat, offsets = src["counts"], src["points"], src["offsets"]
            strands, uv, tl, idx, tsc = [], [], [], [], []
            seen = set()
            for uv_id, tile, first, count in _synth_tiles(scene, key, src):
                if (uv_id, tile) in seen:
                    continue
                seen.add((uv_id, tile))
                for j in range(max(first, 0), min(first + count, len(counts))):
                    c = counts[j]
                    if c <= 1 or 3 * offsets[j + 1] > len(flat):
                        continue
                    o = 3 * offsets[j]
                    strands.append([(flat[o + 3 * k], flat[o + 3 * k + 1], flat[o + 3 * k + 2])
                                    for k in range(c)])
                    uv.append(uv_id); tl.append(tile); idx.append(j - first); tsc.append(count)
            lib_path = root.AppendChild("cardStrands_" + key)
            lib = _curves(stage, lib_path)
            _set_curves(lib, strands)
            prim = lib.GetPrim()
            prim.SetCustomDataByKey("usdGen:cardSourceKey", key)
            _uniform(prim, "uvId", "int", uv)
            _uniform(prim, "tile", "int", tl)
            _uniform(prim, "index", "int", idx)
            _uniform(prim, "tileStrandCount", "int", tsc)
            result[key] = str(lib_path)
    return result


def _set_cards(scene, s):
    """Records (points, planes) of the cards a hair set's filters select (no geometry built)."""
    layer, uv, tile = int(_hs(s, "cardSourceLayer")), int(_hs(s, "cardSourceUvId")), int(_hs(s, "cardSourceTile"))
    out = []
    for pts, f in card_records(scene):
        if (layer != -1 and f["layer"] != layer) or (uv != -1 and f["uvId"] != uv) or \
           (tile != -1 and f["tile"] != tile):
            continue
        out.append((pts, f))
    return out


def card_source_params(hair_set, scene, scale=1.0):
    """CardSource attribute values for one hair set: {attribute name: value}."""
    s = _settings(hair_set)
    if int(_hs(s, "resolutionType")) == 0:
        steps = int(_clamp(int(_hs(s, "hairResolution")), 3, 1000))
    else:
        # Distance mode: the reference uses a card-texture length (not decoded);
        # use the longest spine of the selected cards x its lengthMax [INF].
        d = _clamp(float(_hs(s, "resolutionDistance")), 0.01, 10.0) * 0.01 * scale
        lmax = 0.0
        for pts, f in _set_cards(scene, s):
            L = sum(math.dist(pts[i], pts[i - 1]) for i in range(1, len(pts))) * scale
            lmax = max(lmax, L * f["lengthMax"])
        steps = min(max(2 * math.ceil(lmax / d) + 1 if d > 0 else 8, 8), 1000)
    return {
        "usdGen:cardSource:layer": int(_hs(s, "cardSourceLayer")),
        "usdGen:cardSource:uvId": int(_hs(s, "cardSourceUvId")),
        "usdGen:cardSource:tile": int(_hs(s, "cardSourceTile")),
        "usdGen:cardSource:braid": bool(_hs(s, "cardSourceBraid")),
        "usdGen:cardSource:amount": float(_clamp(float(_hs(s, "cardStrandAmount")), 0.0, 1.0)),
        "usdGen:cardSource:rootFlatten": float(_clamp(float(_hs(s, "cardRootFlatten")), 0.0, 1.0)),
        "usdGen:cardSource:smooth": bool(_hs(s, "cardSmooth")),
        "usdGen:cardSource:maxStrands": 2000000,
        "usdGen:cardSource:braidWidth": float(_clamp(float(_hs(s, "braidWidth")), 0.0, 5.0)),
        "usdGen:cardSource:braidFlatness": float(_clamp(float(_hs(s, "braidFlatness")), 0.0, 1.0)),
        "usdGen:cvCount": int(steps),
    }


def author_card_source(stage, op_path, hair_set, scene, card_prims, scale=1.0):
    """Define a UsdGenCardSource prim for a hair set and wire both relationships."""
    prim = stage.DefinePrim(Sdf.Path(str(op_path)), "UsdGenCardSource")
    for name, value in card_source_params(hair_set, scene, scale).items():
        prim.CreateAttribute(name, _SDF_TYPES[CARD_SOURCE_ATTR_TYPES[name]]).Set(value)
    prim.CreateRelationship("usdGen:cards").SetTargets([Sdf.Path(card_prims["cards"])])
    key = card_source_key(hair_set)
    if key and key in card_prims:
        prim.CreateRelationship("usdGen:cardStrands").SetTargets([Sdf.Path(card_prims[key])])
    else:
        warnings.warn("cards.py: hair set %r has no card strand library" % hair_set.get("name"))
    return prim
