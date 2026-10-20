#!/usr/bin/env python3
# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
"""Self-test for sir2usd (stdlib + pxr; run with python -S, see README.md).

    python -S selftest.py                       # synthetic scene + preset round trip
    python -S selftest.py --sample Autumn.sir   # also convert the sample and check its known numbers
    python -S selftest.py --sample Autumn.sir --out D:/tmp/out --cards-module PATH/cards.py

The synthetic part builds a tiny legacy .json scene (quads, a triangle, a pentagon, a UV seam, guides on every
face kind, masks, trim/parting curves, a hair tie, index-keyed edits) and checks the authored structure and the
binding math.  The sample part asserts the published Autumn facts (19131 mask vertices, 5716 / 2287 root targets,
6 + 6 lash guides, 38 / 28 painted vertices, ...).  usdchecker is run when it is on PATH (failures reported, not
fatal: new operator types are unknown until the schema plugin is on PXR_PLUGINPATH_NAME).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sir2usd as S  # noqa: E402
from pxr import Usd, UsdGeom, UsdShade  # noqa: E402

FAILS = []


def check(cond, msg):
    if cond:
        print("  ok   " + msg)
    else:
        print("  FAIL " + msg)
        FAILS.append(msg)


# ----------------------------------------------------------------------------- synthetic scene

OBJ = """# synthetic test mesh, centimetres
v 0 0 0
v 10 0 0
v 10 0 10
v 0 0 10
v 20 0 0
v 20 0 10
v 30 0 5
v 15 0 20
v 25 0 24
v 22 0 32
v 8 0 32
v 5 0 24
vt 0 0
vt 0.25 0
vt 0.25 0.25
vt 0 0.25
vt 0.5 0
vt 0.5 0.25
vt 0.75 0.125
vt 0.4 0.5
vt 0.6 0.6
vt 0.55 0.8
vt 0.2 0.8
vt 0.12 0.6
vt 0.3 0
vn 0 1 0
f 1/1/1 4/4/1 3/3/1 2/2/1
f 2/13/1 3/3/1 6/6/1 5/5/1
f 5/5/1 6/6/1 7/7/1
f 8/8/1 12/12/1 11/11/1 10/10/1 9/9/1
"""
# faces: quad A (t0,t1), quad B (t2,t3), triangle (t4), pentagon (t5,t6,t7)
# unique (v,vt,vn) corners: 4 + 3 (seam vertex 2/13, 5, 6) + 1 + 5 = 13


def _fan_root(mesh, tri, bary):
    return S.fan_point(mesh, tri, S._clean_bary(bary))


def write_png_gray(path, w, h, value):
    """Minimal 8-bit greyscale PNG (stdlib zlib)."""
    import struct
    import zlib

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    raw = b"".join(bytes([0]) + bytes([value]) * w for _ in range(h))
    signature = bytes([0x89]) + b"PNG" + bytes([13, 10, 26, 10])
    with open(path, "wb") as f:
        f.write(signature + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def build_synthetic(tmp):
    obj_path = os.path.join(tmp, "head.obj")
    with open(obj_path, "w") as f:
        f.write(OBJ)
    png_path = os.path.join(tmp, "length_mask.png")
    write_png_gray(png_path, 4, 4, 200)
    skin = {}
    for slot, value in (("albedo", 180), ("normal", 128), ("detail", 128)):
        skin[slot] = os.path.join(tmp, "skin_%s.png" % slot)
        write_png_gray(skin[slot], 4, 4, value)
    mesh = S.ObjMesh.parse(OBJ.encode(), 0.01)

    def guide(gid, layer, tri, bary, twin=0, length=0.03, root=None):
        r = root if root is not None else _fan_root(mesh, tri, bary)
        pts = [list(r)] + [[r[0], r[1] + length * k / 3.0, r[2] + 0.001 * k] for k in (1, 2, 3)]
        return {"id": gid, "layer": layer, "mirrorTwinId": twin, "mirrorPairPlane": 0, "points": pts,
                "rootTri": tri, "rootBary": bary}

    curves = [guide(1, 0, 1, [0.2, 0.3, 0.5]),
              guide(2, 0, 4, [0.5, 0.25, 0.25], twin=1),
              guide(3, 0, 6, [0.3, 0.3, 0.4]),
              guide(4, 0, -1, [0, 0, 0], root=(0.15, 0.0, 0.05)),        # unbound -> closest-point rebind
              guide(5, 1, 0, [0.4, 0.3, 0.3])]                            # layer 1 (hidden)
    nv = 13
    dist_mask = [255] * nv
    dist_mask[0] = 0
    dist_mask[5] = 128
    len_mask = [255] * nv
    len_mask[2] = 64
    base = dict(S.HAIR_SETTINGS_DEFAULTS)
    s0 = dict(base, distribution=2, hairDensity=5000.0, clumpType=1, frizzAmount=40.0, frizzScale=20.0,
              frizzNoiseSize=4.0, resolutionType=1, resolutionDistance=0.5, clumpToHairTies=True,
              hairThickness=0.0012, guideProximityMode=0, guideProximityDistance=0.5, partingAmount=0.6,
              partingRadius=30.0, trimAngle=20.0, lengthMaskPath=png_path)
    s1 = dict(base, distribution=1, clumpType=4, pushOutEnabled=True, pushOutRoot=0.2, pushOutTip=1.5,
              smoothness=0.5, hairResolution=150, guideInterpolation=1, twistRoot=15.0)
    s2 = dict(base, distribution=4, cardSourceLayer=0)
    scene = {
        "version": 19, "meshPath": obj_path, "meshImportScale": 0.01,
        "camera": {"target": [0.1, 0.0, 0.1], "yaw": 30.0, "pitch": 10.0, "distance": 0.5, "fovY": 0.6},
        "renderMeshMaterial": {"albedoColor": [0.5, 1.0, 1.0], "albedoTexturePath": skin["albedo"],
                               "useSceneMeshTextureAsAlbedo": False, "roughness": 0.3, "metalness": 0.0,
                               "normalTexturePath": skin["normal"], "normalStrength": 0.5, "normalFlipY": False,
                               "detailNormalTexturePath": skin["detail"], "detailNormalStrength": 0.5,
                               "subsurfaceEnabled": True, "subsurfaceWeight": 0.5, "subsurfaceRadius": 0.07},
        "guideSettings": {"defaultLength": 0.004},
        "layers": [{"name": "Front", "module": "guides", "visible": True},
                   {"name": "Hidden", "module": "guides", "visible": False}],
        "curves": curves,
        "trimCurves": [{"layer": 0, "hairSet": -1, "points": [[0.0, 0.01, 0.02], [0.3, 0.01, 0.02]]},
                       {"layer": 0, "hairSet": 0, "points": [[0.0, 0.01, 0.06], [0.3, 0.01, 0.06]],
                        "trim": {"angle": 10.0, "invertAngle": True, "distance": 0.05}},
                       {"layer": 0, "hairSet": 1, "points": [[0.0, 0.01, 0.07], [0.3, 0.01, 0.07]]}],
        "partingCurves": [{"layer": 0, "points": [[0.02, 0.0, 0.02], [0.08, 0.0, 0.08]],
                           "pointTris": [0, 1], "pointBarys": [[0.6, 0.2, 0.2], [0.2, 0.2, 0.6]]}],
        "hairTies": [{"id": 77, "position": [0.05, 0.02, 0.05],
                      "bindings": [{"curveId": 1, "pointIndex": 2, "localOffset": [0, 0, 0]},
                                   {"curveId": 999, "pointIndex": 1}]}],
        "renderMaterials": [
            {"UsePresetColorSettings": False, "Melanin": 0.4, "MelaninRedness": 0.7, "Roughness": 0.05,
             "LongitudialRoughness": 0.5, "SpecularScale": 1.0, "MultipleScattering": "None"},
            {"UsePresetColorSettings": True, "ColorA_1": [0.149, 0.086, 0.078], "ColorA_2": [0.247, 0.137, 0.133],
             "ColorA_3": [0.298, 0.184, 0.180], "ColorPointA": [0.0, 0.2965, 0.6505, 0.8901], "ColorA_Weight": 1.0,
             "ColorB_Weight": 0.0, "ColorC_Weight": 0.49, "Roughness": 0.015, "LongitudialRoughness": 0.916,
             "SpecularScale": 0.3, "Saturate_Random": 0.3, "MultipleScattering": "DualWithIBLOptimized"}],
        "hairSets": [
            {"name": "Front Set", "renderMaterialId": 1, "visible": True, "layerVisibility": [True, False],
             "settings": s0, "distributionVertexMask": dist_mask, "lengthVertexMask": len_mask,
             "generationDistributionVertexMask": [255] * nv, "generationDistributionMaskInitialized": True,
             "clumpOverrideHairCount": 3, "clumpOverride": [{"i": 1, "v": 0}, {"i": 2, "v": -1}],
             "deletedStrandsHairCount": 10, "deletedStrands": [3, 4]},
            {"name": "Hidden Set", "renderMaterialId": 0, "visible": False, "layerVisibility": [False, True],
             "settings": s1},
            {"name": "Cards", "renderMaterialId": 1, "layerVisibility": [True, True], "settings": s2,
             "cardSource": "0123456789abcdef"}],
    }
    path = os.path.join(tmp, "synthetic.json")
    with open(path, "w") as f:
        json.dump(scene, f)
    return path, mesh, scene


def test_math():
    print("math:")
    mesh = S.ObjMesh.parse(OBJ.encode(), 0.01)
    check(len(mesh.points) == 13 and mesh.n_positions == 12, "re-index: 12 positions -> 13 (v,vt,vn) corners (UV seam)")
    check(mesh.face_counts == [4, 4, 3, 5] and len(mesh.tri_face) == 8, "face order kept, 8 fan triangles")
    for tri, bary in ((0, (0.2, 0.3, 0.5)), (1, (0.2, 0.3, 0.5)), (4, (0.5, 0.25, 0.25)), (5, (0.1, 0.6, 0.3)),
                      (7, (0.3, 0.3, 0.4))):
        p = S.fan_point(mesh, tri, bary)
        face, uv = S.fan_to_face_param(mesh, tri, bary)
        q = S.eval_face_param(mesh, face, uv)
        check(math.dist(p, q) < 1e-9, "fan tri %d -> face %d uv (%.4f, %.4f) reproduces the point" % (tri, face, uv[0], uv[1]))
    t, w, d = S.closest_fan_binding(mesh, (0.15, 0.01, 0.05))
    check(mesh.tri_face[t] == 1 and abs(d - 0.01) < 1e-9, "closest-point rebind lands on quad B, distance 1 cm")
    check(abs(S.bezier_profile(0.25, 0, 0.5, 0.5, 1) - 0.25) < 1e-9, "Bezier profile r=0,m=.5,t=1 is linear")
    raw = {"hairResolution": 2, "frizzAmount": 300, "distribution": 9, "twistMidShift": 0.0, "hairThickness": 0.0}
    st = S.settings_for(raw)
    check(st["hairResolution"] == 3 and st["frizzAmount"] == 100 and st["distribution"] == 2 and
          st["twistMidShift"] == 0.1 and st["hairThickness"] == 0.001, "reader clamps / enum fallback")
    st = S.settings_for({"rootThickness": 0.002, "midThickness": 0.004, "tipThickness": 0.001})
    check(abs(st["hairThickness"] - 0.004) < 1e-12 and abs(st["thicknessRoot"] - 0.5) < 1e-12, "legacy thickness keys")
    root, tip, k = S.fit_color_ramp([0.149, 0.086, 0.078], [0.247, 0.137, 0.133], [0.298, 0.184, 0.180],
                                    [0.0, 0.2965, 0.6505, 0.8901])
    check(abs(root[0] - 0.0194) < 2e-4 and abs(tip[0] - 0.0723) < 2e-4 and 0.5 < k < 1.2,
          "preset gradient A -> root %.4f tip %.4f colorRamp %.3f" % (root[0], tip[0], k))
    inputs, _ = S.map_hair_material({"Roughness": 0.015, "LongitudialRoughness": 0.916, "SpecularScale": 0.3})
    check(abs(inputs["roughness"][1] - 0.344) < 2e-3 and abs(inputs["specular"][1] - 0.15) < 1e-9,
          "material roughness sqrt(2 v R) = %.3f, specular = %.2f" % (inputs["roughness"][1], inputs["specular"][1]))
    pre = {"name": "p", "module": 1, "settings": {"frizzAmount": 50, "hairDensity": 9, "trimAngle": 5,
                                                   "distribution": 4, "bogus": 1}}
    new, applied, skipped = S.apply_hair_preset({"distribution": 2}, pre, 1)
    check(new["frizzAmount"] == 50 and new["hairDensity"] == 9 and new["trimAngle"] == 0.0 and
          new["distribution"] == 2 and "bogus" in skipped, "preset classes: 0/1 applied, 2 + unknown skipped, "
                                                           "from-cards switch refused")
    pre["module"] = 2
    new, _a, _s = S.apply_hair_preset({}, pre, 1)
    check(new["hairDensity"] == 20000.0 and new["frizzAmount"] == 50, "texture-module preset: class 1 not applied")


def test_synthetic(tmp, cards_module):
    print("synthetic scene:")
    scene_path, mesh, scene = build_synthetic(tmp)
    out = os.path.join(tmp, "out", "synthetic.usda")
    summ = S.convert(scene_path, out, quiet=True, cards_module=cards_module)
    stage = Usd.Stage.Open(out)
    check(stage is not None, "stage opens")
    check(UsdGeom.GetStageMetersPerUnit(stage) == 1.0 and UsdGeom.GetStageUpAxis(stage) == "Y", "metersPerUnit 1, Y up")
    m = UsdGeom.Mesh(stage.GetPrimAtPath("/World/head"))
    check(len(m.GetPointsAttr().Get()) == 13 and list(m.GetFaceVertexCountsAttr().Get()) == [4, 4, 3, 5],
          "mesh: 13 vertices, faces kept in order")
    st_pv = UsdGeom.PrimvarsAPI(m).GetPrimvar("st")
    check(st_pv and st_pv.GetInterpolation() == "vertex" and len(st_pv.Get()) == 13, "vertex st")
    check(os.path.isfile(os.path.join(tmp, "out", "synthetic_assets", "head.obj")), "legacy assets copied next to output")

    d0 = stage.GetPrimAtPath("/World/Groom/Front_Set")
    check(bool(d0) and d0.GetTypeName() == "UsdGenDescription", "description per hair set")
    check(d0.GetAttribute("usdGen:curve:basis").Get() == "linear", "curve basis linear")
    check(abs(d0.GetAttribute("usdGen:width:default").Get() - 0.0024) < 1e-9, "width default = 2 * hairThickness")
    ops = [c.GetName() for c in stage.GetPrimAtPath("/World/Groom/Front_Set/Ops").GetChildren()]
    check(ops == ["width", "resample", "smooth", "pushOut", "tieClump", "frizz", "trim", "length", "twist", "braid",
                  "clump", "interp", "scatter"], "ops authored in reverse execution order: %s" % ops)
    cd = d0.GetCustomDataByKey("sisir")
    check(cd["settings"]["frizzAmount"] == 40.0 and len(cd["settings"]) == 99, "raw settings kept (99 keys)")
    check(any("deletedStrands" in u for u in cd["unsupported"]), "deletedStrands recorded as unsupported")

    gc = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Groom/Guides/Front_Set/curves"))
    check(gc and len(gc.GetCurveVertexCountsAttr().Get()) == 4, "guide set holds the 4 layer-0 guides (layer 1 hidden)")
    pv = UsdGeom.PrimvarsAPI(gc)
    faces = list(pv.GetPrimvar("skinprim").Get())
    uvs = list(pv.GetPrimvar("skinprimuv").Get())
    check(faces == [0, 2, 3, 1], "skinprim = polygon index (quad A, triangle, pentagon, rebound quad B): %s" % faces)
    pts = gc.GetPointsAttr().Get()
    counts = gc.GetCurveVertexCountsAttr().Get()
    off, errs = 0, []
    for k, n in enumerate(counts):
        errs.append(math.dist(S.eval_face_param(mesh, faces[k], tuple(uvs[k])), tuple(pts[off])))
        off += n
    check(max(errs) < 1e-6, "skinprim/skinprimuv reproduce points[0] (max %.2g m)" % max(errs))
    check(abs(uvs[2][0] - 1.3) < 1e-6, "pentagon skinprimuv encodes fan index in u (%.4f)" % uvs[2][0])
    check(list(pv.GetPrimvar("usdGen:plane:mirrorSign").Get()) == [1, -1, 1, 1], "mirrorSign plane from twin ids")
    check(len(pv.GetPrimvar("widths").Get()) == len(pts), "guide widths per CV (compiler size rule)")

    o = lambda name: stage.GetPrimAtPath("/World/Groom/Front_Set/Ops/" + name)
    check(o("scatter").GetAttribute("usdGen:distribution").Get() == "even", "distribution 2 -> even")
    check(o("scatter").GetAttribute("usdGen:proximity:mode").Get() == "coverage" and
          abs(o("scatter").GetAttribute("usdGen:proximity:coverage").Get() - 0.5) < 1e-7,
          "proximity coverage from guideProximityDistance")
    dm = o("scatter").GetRelationship("usdGen:densityMap").GetTargets()
    check([p.name for p in dm] == ["distributionPaint"], "distribution mask -> PaintMap on scatter densityMap")
    pm = stage.GetPrimAtPath(dm[0])
    pvname = pm.GetAttribute("usdGen:paint:primvar").Get()
    fv = UsdGeom.PrimvarsAPI(m).GetPrimvar(pvname)
    check(fv.GetInterpolation() == "faceVarying" and len(fv.Get()) == 16 and fv.Get()[0] == 0.0,
          "mask as faceVarying float primvar (16 face-vertices)")
    lm = o("length").GetRelationship("usdGen:lengthMap").GetTargets()
    check([p.name for p in lm] == ["lengthPaint", "lengthImage"], "length vertex + image masks -> Length lengthMap")
    img = stage.GetPrimAtPath(lm[1]) if len(lm) > 1 else None
    f = img.GetAttribute("usdGen:map:file").Get() if img else None
    check(img is not None and img.GetTypeName() == "UsdGenImageMap" and f is not None and
          f.path == "./synthetic_assets/length_mask.png" and os.path.isfile(f.resolvedPath),
          "image mask -> UsdGenImageMap, file copied and layer-relative")
    lmax = max(S.polyline_length([tuple(p) for p in c["points"]]) for c in scene["curves"][:4])
    want = min(max(2 * math.ceil(lmax / 0.005) + 1, 8), 1000)
    check(o("interp").GetAttribute("usdGen:cvCount").Get() == want,
          "distance resolution: steps = 2*ceil(Lmax/d)+1 = %d" % want)
    check(o("resample").GetAttribute("usdGen:enabled").Get() and
          abs(o("resample").GetAttribute("usdGen:spacing").Get() - 0.005) < 1e-9, "resample spacing = cm * 0.01")
    check(o("clump").GetAttribute("usdGen:clump:centers").Get() == "guides" and
          o("clump").GetRelationship("usdGen:guides").HasAuthoredTargets(), "clumpType 1 -> centers guides + guides rel")
    ovr = list(o("clump").GetAttribute("usdGen:clump:overrides").Get())
    check(ovr == [S.INT_MIN, 0, -1] and o("clump").GetAttribute("usdGen:clump:overrideCount").Get() == 3,
          "clumpOverride -> clump:overrides indexed by strand")
    check(abs(o("frizz").GetAttribute("usdGen:frizz:amplitude").Get() - 0.1) < 1e-7 and
          abs(o("frizz").GetAttribute("usdGen:frizz:wavesX").Get() - 2.0) < 1e-7 and
          abs(o("frizz").GetAttribute("usdGen:frizz:noiseSize").Get() - 0.04) < 1e-7, "frizz conversions")
    check(len(o("interp").GetAttribute("usdGen:part:points").Get()) == 2 and
          len(o("interp").GetAttribute("usdGen:part:normals").Get()) == 2 and
          abs(o("interp").GetAttribute("usdGen:part:radius").Get() - 0.3) < 1e-7, "parting curve -> part:* attributes")
    tc = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Groom/Front_Set/TrimCurves"))
    check(o("trim").GetAttribute("usdGen:enabled").Get() and len(tc.GetCurveVertexCountsAttr().Get()) == 2,
          "trim curves for this set (shared + own, not the other set's)")
    ang = list(UsdGeom.PrimvarsAPI(tc).GetPrimvar("usdGen:plane:trimAngle").Get())
    check(abs(ang[0] - 20.0) < 1e-6 and abs(ang[1] - 10.0) < 1e-6, "per-curve trim overrides resolved into planes")
    check(o("tieClump").GetAttribute("usdGen:enabled").Get() and
          list(o("tieClump").GetAttribute("usdGen:tie:guideIndices").Get()) == [0] and
          list(o("tieClump").GetAttribute("usdGen:tie:vertexIndices").Get()) == [2], "hair tie -> tie:* (guide 0, cv 2)")
    mat = d0.GetRelationship("material:binding").GetTargets()
    check(mat and str(mat[0]) == "/World/Looks/Hair_1", "material binding")
    skin = stage.GetPrimAtPath("/World/Looks/head_skin")
    bound = UsdShade.MaterialBindingAPI(m.GetPrim()).GetDirectBinding().GetMaterialPath()
    check(bool(skin) and str(bound) == "/World/Looks/head_skin", "mesh bound to its render material")
    sh = lambda n: UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/head_skin/" + n))
    check(sh("diffuse").GetInput("file").Get().path.endswith("skin_albedo.png") and
          tuple(sh("diffuse").GetInput("scale").Get()) == (0.5, 1.0, 1.0, 1.0), "skin: albedo texture x albedoColor")
    check(abs(sh("Surface").GetInput("roughness").Get() - 0.3) < 1e-6, "skin: roughness without a texture = slider")
    check(tuple(sh("normalTex").GetInput("scale").Get()) == (1.0, -1.0, 2.0, 1.0) and
          tuple(sh("normalTex").GetInput("bias").Get()) == (-0.5, 0.5, -1.0, 0.0),
          "skin: normal strength 0.5, unflagged (DirectX-style) map gets Y negated")
    unsup = list(skin.GetCustomDataByKey("sisir")["unsupported"])
    check(len(unsup) == 2 and "detail normal" in unsup[0] and "subsurface" in unsup[1],
          "skin: detail normal and subsurface listed as unsupported")

    d1 = stage.GetPrimAtPath("/World/Groom/Hidden_Set")
    o1 = lambda name: stage.GetPrimAtPath("/World/Groom/Hidden_Set/Ops/" + name)
    check(UsdGeom.Imageable(d1).GetVisibilityAttr().Get() == "invisible", "invisible hair set")
    check(not stage.GetPrimAtPath("/World/Groom/Guides/Hidden_Set"), "hidden layer drives no guides")
    check(not o1("braid").GetAttribute("usdGen:enabled").Get() and
          not o1("braid").GetRelationship("usdGen:guides").HasAuthoredTargets(),
          "braid without guides: disabled, no guides rel")
    check(o1("pushOut").GetAttribute("usdGen:enabled").Get() and
          abs(o1("pushOut").GetAttribute("usdGen:clearance:tip").Get() - 0.015) < 1e-7, "push-out clearance cm -> m")
    check(o1("smooth").GetAttribute("usdGen:enabled").Get() and o1("smooth").GetAttribute("usdGen:window").Get() == 3,
          "smooth window max(2, round(2 r)), r = 150/100")
    check(o1("twist").GetAttribute("usdGen:twist:resnapRoot").Get(), "twist root re-snap when twistRoot != 0")
    check(o1("scatter").GetAttribute("usdGen:distribution").Get() == "vertex", "distribution 1 -> vertex")

    groom_cd = stage.GetPrimAtPath("/World/Groom").GetCustomDataByKey("sisir")
    cards_prim = stage.GetPrimAtPath("/World/Groom/Cards")
    check(bool(cards_prim) or "skippedHairSets" in groom_cd, "card set authored via cards.py or recorded as skipped")

    # preset round trip
    preset = os.path.join(tmp, "preset.json")
    with open(preset, "w") as f:
        json.dump({"version": 2, "module": "hair", "settings": {
            "frizzAmount": 80.0, "clumpType": 2, "clumpRandomCount": 7, "hairDensity": 1234.0, "trimAngle": 45.0,
            "distribution": 4, "twistRoot": 30.0, "hairThickness": 0.002}}, f)
    out2 = os.path.join(tmp, "out", "preset.usda")
    res = S.apply_preset_to_stage(out, preset, "/World/Groom/Front_Set", out2, quiet=True)
    st2 = Usd.Stage.Open(out2)
    p = lambda name: st2.GetPrimAtPath("/World/Groom/Front_Set/Ops/" + name)
    check(abs(p("frizz").GetAttribute("usdGen:frizz:amplitude").Get() - 0.2) < 1e-7, "preset: frizz re-derived")
    check(p("clump").GetAttribute("usdGen:clump:centers").Get() == "count" and
          p("clump").GetAttribute("usdGen:clump:count").Get() == 7 and
          not p("clump").GetRelationship("usdGen:guides").HasAuthoredTargets(),
          "preset: clump mode switch updates the guides rel")
    check(p("scatter").GetAttribute("usdGen:density").Get() == 1234.0, "preset: class-1 key applied (same module)")
    check(p("trim").GetAttribute("usdGen:trim:angle").Get() == 20.0, "preset: class-2 key (trimAngle) ignored")
    check(p("scatter").GetAttribute("usdGen:distribution").Get() == "even", "preset: from-cards switch refused")
    check(p("twist").GetAttribute("usdGen:enabled").Get(), "preset: twist gate re-evaluated")
    check(len(p("interp").GetAttribute("usdGen:part:points").Get()) == 2, "preset: scene data attributes kept")
    check(abs(st2.GetPrimAtPath("/World/Groom/Front_Set").GetAttribute("usdGen:width:default").Get() - 0.004) < 1e-9,
          "preset: width default updated")
    check("trimAngle" in res["skipped"], "preset: skipped keys reported")
    return out


AUTUMN = {"vertices": 19131, "faces": 18922, "targets": {"Set_1": 5716, "Set_1_Copy": 2287},
          "guides": {"Set_1": 6, "Set_1_Copy": 6}, "painted": {"Set_1": 38, "Set_1_Copy": 28},
          "clumpCount": {"Set_1": 52, "Set_1_Copy": 35}}


def test_sample(sample, out_dir, cards_module):
    print("sample %s:" % sample)
    out = os.path.join(out_dir, "Autumn_selftest.usda")
    summ = S.convert(sample, out, quiet=True, cards_module=cards_module)
    stage = Usd.Stage.Open(out)
    m = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Autumn_head"))
    check(len(m.GetPointsAttr().Get()) == AUTUMN["vertices"], "mesh re-indexed to 19131 vertices = mask length")
    check(len(m.GetFaceVertexCountsAttr().Get()) == AUTUMN["faces"], "18922 faces in OBJ order")
    by_name = {s["path"].rsplit("/", 1)[-1]: s for s in summ["sets"]}
    for name, target in AUTUMN["targets"].items():
        s = by_name.get(name)
        check(s is not None and s["rootTarget"] == target, "%s root target %s (log %d)" % (name, s and s["rootTarget"], target))
        check(s is not None and s["guides"] == AUTUMN["guides"][name], "%s guides %d" % (name, AUTUMN["guides"][name]))
        b = (s or {}).get("guideBinding") or {}
        check(b.get("maxFanError", 1) < 1e-7 and b.get("rebound", 1) == 0, "%s rootTri/rootBary match points[0] (%.2g m)"
              % (name, b.get("maxFanError", -1)))
        check(b.get("maxParamError", 1) < 2e-4, "%s skinprim/skinprimuv within %.2g m of points[0] (quad warp)"
              % (name, b.get("maxParamError", -1)))
        pm = stage.GetPrimAtPath("/World/Groom/%s/Maps/distributionPaint" % name)
        check(pm and pm.GetCustomDataByKey("sisir:nonZeroVertices") == AUTUMN["painted"][name],
              "%s distribution mask painted vertices %d" % (name, AUTUMN["painted"][name]))
        clump = stage.GetPrimAtPath("/World/Groom/%s/Ops/clump" % name)
        check(clump.GetAttribute("usdGen:clump:count").Get() == AUTUMN["clumpCount"][name],
              "%s clump count %d" % (name, AUTUMN["clumpCount"][name]))
        scat = stage.GetPrimAtPath("/World/Groom/%s/Ops/scatter" % name)
        check(abs(scat.GetAttribute("usdGen:proximity:distance").Get() - 0.01) < 1e-9 and
              scat.GetAttribute("usdGen:proximity:mode").Get() == "distance", "%s proximity 1 cm -> 0.01" % name)
    cd = stage.GetPrimAtPath("/World/Groom/Set_1_Copy").GetCustomDataByKey("sisir")
    check(any("7 of 2287" in u for u in cd["unsupported"]), "Set 1 Copy: 7 deletedStrands recorded")
    card_sets = [s for s in summ["sets"] if s["distribution"] == "From Cards"]
    check(len(card_sets) in (0, 8), "card-grown sets: %d authored (0 when cards.py is absent)" % len(card_sets))
    return out


def run_usdchecker(path):
    exe = shutil.which("usdchecker")
    if not exe:
        print("usdchecker: not on PATH, skipped")
        return
    r = subprocess.run([exe, path], capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "TBB Warning" not in l]
    print("usdchecker %s: rc=%d" % (os.path.basename(path), r.returncode))
    for l in lines[:20]:
        print("    " + l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=os.environ.get("SISIR_SAMPLE"), help="Autumn.sir (or $SISIR_SAMPLE)")
    ap.add_argument("--out", help="output folder (default: a temp folder)")
    ap.add_argument("--cards-module", help="cards.py to use for card-grown sets")
    ap.add_argument("--keep", action="store_true", help="keep the temp folder")
    a = ap.parse_args()
    tmp = a.out or tempfile.mkdtemp(prefix="sir2usd_selftest_")
    os.makedirs(tmp, exist_ok=True)
    test_math()
    synth = test_synthetic(tmp, a.cards_module)
    run_usdchecker(synth)
    if a.sample:
        run_usdchecker(test_sample(a.sample, tmp, a.cards_module))
    else:
        print("sample: skipped (pass --sample Autumn.sir or set SISIR_SAMPLE)")
    print("\n%s: %d failure(s)%s" % ("FAILED" if FAILS else "PASSED", len(FAILS),
                                     "" if (a.out or a.keep) else " (temp output removed)"))
    if not (a.out or a.keep):
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
