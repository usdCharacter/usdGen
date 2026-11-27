# testPomadeAbiTube -- T3: Tube + Fill modes through the C ABI
# (plan/17 P3 exit).
#
# ABI-level by name since plan/18 V2: this script drives the C ABI
# directly and never touches the viewport, so it is a regression
# test for the model, not a proof that the tool works. The gesture
# proof for this mode is the matching testUsdviewPomade* script.
#
#   testusdview --testScript plugin/usdGenPomadeTools/testenv/testPomadeAbiTube.py \
#               examples/pomade-graph-scalp.usda
#
# with the same PXR_PLUGINPATH_NAME / PYTHONPATH the T2 editor test uses,
# plus the staged usdGenPomadeTools package and the usdGenPomade DLL (via
# USDGENPOMADE_DLL or the build tree). The scalp is the 4x4 quad grid in XZ:
# face f = ix*4 + iz covers x in [ix, ix+1], z in [iz, iz+1], and (x, z) on
# that face is (u = z - iz, v = x - ix).
#
# Proven here (the ABI path; per-kernel semantics stay in
# tests/hosts/usdGenPomade/testUsdGenPomadeTubes.cpp):
#   * a stroked region rasterises and auto-tubes (5 center CVs, >= 2 rings
#     of 8);
#   * center edits move the CV, bump the version, and insert/delete round
#     trips the count;
#   * section twist/scale/per-CV edits read back, and add/remove round
#     trips the ring count;
#   * fill params round-trip, a full refill yields guides x 8 CVs, and the
#     root census matches;
#   * K11 hits a guide CV through the identity projection, misses far
#     away, and the empty kind mask misses everywhere.
import ctypes
import os
import sys

failures = 0

# PomadePickKind (pomadeTube.h): TubeVert=1, CenterCV=2, SectionCV=4,
# GraphNode=8, Guide=16, All=0x1F.
PICK_GUIDE = 16
PICK_ALL = 0x1F


def check(ok, what):
    global failures
    if ok:
        print("ok:   %s" % what)
    else:
        failures += 1
        print("FAIL: %s" % what)


def locate(x, z):
    """(face, u, v) for a scalp position (x, z)."""
    import math
    ix = min(max(int(math.floor(x)), 0), 3)
    iz = min(max(int(math.floor(z)), 0), 3)
    return (ix * 4 + iz, float(z - iz), float(x - ix))


def strokeRect(dll, model, corners, snap):
    """Stroke a closed rectangle; returns (rc, chain, closed, ...)."""
    samples = []
    per = 5
    for k in range(len(corners)):
        x0, z0 = corners[k]
        x1, z1 = corners[(k + 1) % len(corners)]
        for i in range(per):
            t = float(i) / float(per)
            samples.append((x0 + (x1 - x0) * t, z0 + (z1 - z0) * t))
    samples.append(corners[0])
    faces = (ctypes.c_int * len(samples))()
    uvs = (ctypes.c_float * (2 * len(samples)))()
    for i, (x, z) in enumerate(samples):
        face, u, v = locate(x, z)
        faces[i] = face
        uvs[2 * i] = u
        uvs[2 * i + 1] = v
    out = (ctypes.c_int * 4096)()
    count = ctypes.c_int(0)
    closed = ctypes.c_int(0)
    weldStart = ctypes.c_int(0)
    weldEnd = ctypes.c_int(0)
    rc = dll.Pomade_GraphStroke(model, faces, uvs, len(samples),
                               ctypes.c_float(snap), ctypes.c_float(0.05),
                               out, 4096, ctypes.byref(count),
                               ctypes.byref(closed), ctypes.byref(weldStart),
                               ctypes.byref(weldEnd))
    return (rc, [out[i] for i in range(count.value)], bool(closed.value))


def readSection(dll, model, ring):
    t = ctypes.c_float(0)
    uv = (ctypes.c_float * 64)()
    count = ctypes.c_int(0)
    scale = ctypes.c_float(0)
    twist = ctypes.c_float(0)
    rc = dll.Pomade_GetSection(model, ring, ctypes.byref(t), uv, 64,
                              ctypes.byref(count), ctypes.byref(scale),
                              ctypes.byref(twist))
    return (rc, t.value, [uv[i] for i in range(2 * count.value)],
            count.value, scale.value, twist.value)


def run(stage):
    global failures
    # Under testusdview the script is exec'd (no __file__) but the staged
    # package is already on PYTHONPATH; the insert only helps direct runs.
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = ""
    if here:
        sys.path.insert(0, os.path.normpath(os.path.join(
            here, "..", "python")))
    try:
        from usdGenPomadeTools import pomadeLib
    except ImportError as exc:
        print("FAIL: cannot import usdGenPomadeTools: %s" % exc)
        return 1
    try:
        lib = pomadeLib.Library()
    except OSError as exc:
        print("FAIL: %s" % exc)
        return 1
    dll = lib.dll

    scalp = stage.GetPrimAtPath("/Scalp")
    check(scalp, "the stage carries /Scalp")
    if not scalp:
        return 1
    points = scalp.GetAttribute("points").Get()
    counts = scalp.GetAttribute("faceVertexCounts").Get()
    indices = scalp.GetAttribute("faceVertexIndices").Get()
    flat = [float(c) for p in points for c in (p[0], p[1], p[2])]
    pts = (ctypes.c_float * len(flat))(*flat)
    cnt = (ctypes.c_int * len(counts))(*[int(c) for c in counts])
    idx = (ctypes.c_int * len(indices))(*[int(i) for i in indices])

    model = ctypes.c_void_p(None)
    check(dll.Pomade_Create(ctypes.byref(model)) == 0, "model creates")
    check(dll.Pomade_BindScalp(model, pts, len(flat), cnt, len(counts),
                              idx, len(indices)) == 0, "scalp binds")

    # One interior region: x in [0, 4], z in [1, 3] (8 faces of roots).
    rc, chain, closed = strokeRect(
        dll, model, [(0.0, 1.0), (4.0, 1.0), (4.0, 3.0), (0.0, 3.0)], 0.1)
    check(rc == 0 and closed, "the region stroke closes")
    check(dll.Pomade_Rasterise(model) == 0, "K3 rasterises")
    regions = ctypes.c_int(0)
    uncovered = ctypes.c_int(0)
    intersected = ctypes.c_int(0)
    dll.Pomade_GetRegionStats(model, ctypes.byref(regions),
                             ctypes.byref(uncovered), ctypes.byref(intersected))
    check(regions.value == 1, "one region rasterised")

    # -- Tube mode ------------------------------------------------------
    check(dll.Pomade_BuildTubeFromRegion(model, 0, 5, 8, ctypes.c_float(2.0))
          == 0, "auto-tube builds from region 0")
    check(dll.Pomade_GetCenterCVCount(model) == 5,
          "the tube holds 5 center CVs")
    rings = dll.Pomade_GetSectionCount(model)
    check(rings >= 2, "the tube holds >= 2 rings (got %d)" % rings)
    check(dll.Pomade_GetTubeRegionId(model) == 0,
          "the tube is rooted in region 0")

    before = (ctypes.c_float * 3)()
    check(dll.Pomade_GetCenterCV(model, 2, before) == 0,
          "center CV 2 reads")
    v0 = dll.Pomade_GetVersion(model)
    check(dll.Pomade_MoveCenterCV(model, 2, ctypes.c_float(0.1),
                                 ctypes.c_float(0.0), ctypes.c_float(0.2))
          == 0, "center CV 2 moves")
    after = (ctypes.c_float * 3)()
    dll.Pomade_GetCenterCV(model, 2, after)
    check(dll.Pomade_GetVersion(model) == v0 + 1,
          "the move bumps the version")
    check(abs(after[0] - before[0] - 0.1) < 1e-5
          and abs(after[2] - before[2] - 0.2) < 1e-5,
          "center CV 2 sits at the moved position")
    check(dll.Pomade_InsertCenterCV(model, 2) == 0
          and dll.Pomade_GetCenterCVCount(model) == 6,
          "insert grows the center to 6")
    check(dll.Pomade_DeleteCenterCV(model, 2) == 0
          and dll.Pomade_GetCenterCVCount(model) == 5,
          "delete shrinks the center back to 5")
    check(dll.Pomade_SetTubeLength(model, ctypes.c_float(3.0)) == 0,
          "tube length sets")

    rc, t, uv, n, scale, twist = readSection(dll, model, 0)
    check(rc == 0 and n == 8, "ring 0 reads with 8 CVs (got %d)" % n)
    check(dll.Pomade_TwistSectionRing(model, 0, ctypes.c_float(0.5)) == 0,
          "ring 0 twists")
    rc, _, _, _, _, twist2 = readSection(dll, model, 0)
    check(rc == 0 and abs(twist2 - twist - 0.5) < 1e-5,
          "the twist reads back (+0.5)")
    check(dll.Pomade_ScaleSectionRing(model, 0, ctypes.c_float(1.5)) == 0,
          "ring 0 scales")
    rc, _, _, _, scale2, _ = readSection(dll, model, 0)
    check(rc == 0 and abs(scale2 - scale * 1.5) < 1e-5,
          "the scale reads back (x1.5)")
    slotU, slotV = uv[2 * 3], uv[2 * 3 + 1]
    check(dll.Pomade_MoveSectionCV(model, 0, 3, ctypes.c_float(0.1),
                                  ctypes.c_float(-0.2)) == 0,
          "ring 0 slot 3 moves")
    rc, _, uv2, _, _, _ = readSection(dll, model, 0)
    check(rc == 0 and abs(uv2[2 * 3] - slotU - 0.1) < 1e-5
          and abs(uv2[2 * 3 + 1] - slotV + 0.2) < 1e-5,
          "the section CV reads back moved")
    newRings = ctypes.c_int(0)
    check(dll.Pomade_AddSectionRing(model, ctypes.c_float(0.33),
                                   ctypes.byref(newRings)) == 0
          and newRings.value == rings + 1,
          "add ring grows to %d" % (rings + 1))
    added = -1
    for r in range(newRings.value):
        rc, rt, _, _, _, _ = readSection(dll, model, r)
        if rc == 0 and abs(rt - 0.33) < 1e-4:
            added = r
    check(added >= 0, "the added ring sorts by t")
    check(added < 0 or dll.Pomade_RemoveSectionRing(model, added) == 0
          and dll.Pomade_GetSectionCount(model) == rings,
          "remove shrinks back to %d" % rings)
    check(dll.Pomade_SetSoftSelection(model, ctypes.c_float(0.5),
                                     ctypes.c_float(0.25)) == 0,
          "soft selection sets")
    sc = ctypes.c_float(0)
    sr = ctypes.c_float(0)
    dll.Pomade_GetSoftSelection(model, ctypes.byref(sc), ctypes.byref(sr))
    check(abs(sc.value - 0.5) < 1e-6 and abs(sr.value - 0.25) < 1e-6,
          "soft selection reads back")
    check(dll.Pomade_SetDisplaySegments(model, 3) == 0
          and dll.Pomade_GetDisplaySegments(model) == 3,
          "display segments round-trip")

    # -- Fill mode ------------------------------------------------------
    check(dll.Pomade_SetFillParams(model, ctypes.c_float(50.0), 8, 11,
                                  ctypes.c_float(0.25), None, 0) == 0,
          "fill params set")
    density = ctypes.c_float(0)
    cvCount = ctypes.c_int(0)
    seed = ctypes.c_int(0)
    edgeBias = ctypes.c_float(0)
    profile = (ctypes.c_float * 16)()
    floats = ctypes.c_int(0)
    dll.Pomade_GetFillParams(model, ctypes.byref(density), ctypes.byref(cvCount),
                            ctypes.byref(seed), ctypes.byref(edgeBias),
                            profile, 16, ctypes.byref(floats))
    check(abs(density.value - 50.0) < 1e-6 and cvCount.value == 8
          and seed.value == 11 and abs(edgeBias.value - 0.25) < 1e-6,
          "fill params read back")
    check(dll.Pomade_SetPreviewFraction(model, ctypes.c_float(0.25)) == 0
          and abs(dll.Pomade_GetPreviewFraction(model) - 0.25) < 1e-6,
          "preview fraction round-trips")
    check(dll.Pomade_RefillGuides(model, ctypes.c_float(1.0)) == 0,
          "full-density refill runs")
    guides = ctypes.c_int(0)
    cv = ctypes.c_int(0)
    dll.Pomade_GetGuideCounts(model, ctypes.byref(guides), ctypes.byref(cv))
    check(guides.value > 0 and cv.value == 8,
          "the refill yields guides x 8 CVs (got %d)" % guides.value)
    xyz = (ctypes.c_float * max(3 * guides.value * cv.value, 3))()
    gcounts = (ctypes.c_int * max(guides.value, 1))()
    got = ctypes.c_int(0)
    check(dll.Pomade_ReadGuidePreview(model, xyz, len(xyz), gcounts,
                                     len(gcounts), ctypes.byref(got)) == 0
          and got.value == guides.value,
          "the guide preview reads %d guides" % guides.value)
    rootFaces = (ctypes.c_int * max(guides.value, 1))()
    rootXYZ = (ctypes.c_float * max(3 * guides.value, 3))()
    rootRU = (ctypes.c_float * max(2 * guides.value, 2))()
    roots = ctypes.c_int(0)
    check(dll.Pomade_ReadGuideRoots(model, rootFaces, rootXYZ, rootRU,
                                   max(guides.value, 1),
                                   ctypes.byref(roots)) == 0
          and roots.value == guides.value,
          "the root census matches (%d)" % guides.value)

    # -- K11 pick ---------------------------------------------------------
    # Identity projection over a 400x400 view: NDC (x, y) -> pixels
    # ((x/2+1/2)*400, (1-(y/2+1/2))*400). Guide 0 CV 0 is the target.
    viewProj = (ctypes.c_float * 16)(1, 0, 0, 0, 0, 1, 0, 0,
                                     0, 0, 1, 0, 0, 0, 0, 1)
    gx, gy = xyz[0], xyz[1]
    px = (gx * 0.5 + 0.5) * 400.0
    py = (1.0 - (gy * 0.5 + 0.5)) * 400.0
    hit = ctypes.c_int(0)
    kind = ctypes.c_uint(0)
    index = ctypes.c_int(0)
    sub = ctypes.c_int(0)
    dist = ctypes.c_float(0)
    depth = ctypes.c_float(0)
    check(dll.Pomade_Pick(model, viewProj, 400, 400, ctypes.c_float(px),
                         ctypes.c_float(py), ctypes.c_float(5.0), PICK_GUIDE,
                         ctypes.byref(hit), ctypes.byref(kind),
                         ctypes.byref(index), ctypes.byref(sub),
                         ctypes.byref(dist), ctypes.byref(depth)) == 0
          and hit.value == 1 and kind.value == PICK_GUIDE,
          "K11 hits guide 0 at its projection")
    check(dll.Pomade_Pick(model, viewProj, 400, 400, ctypes.c_float(-1000.0),
                         ctypes.c_float(-1000.0), ctypes.c_float(5.0), PICK_ALL,
                         ctypes.byref(hit), ctypes.byref(kind),
                         ctypes.byref(index), ctypes.byref(sub),
                         ctypes.byref(dist), ctypes.byref(depth)) == 0
          and hit.value == 0,
          "K11 misses far from the groom")
    check(dll.Pomade_Pick(model, viewProj, 400, 400, ctypes.c_float(px),
                         ctypes.c_float(py), ctypes.c_float(5.0), 0,
                         ctypes.byref(hit), ctypes.byref(kind),
                         ctypes.byref(index), ctypes.byref(sub),
                         ctypes.byref(dist), ctypes.byref(depth)) == 0
          and hit.value == 0,
          "K11 with the empty mask misses everywhere")

    dll.Pomade_Destroy(model)
    print("testPomadeAbiTube: %d failure(s)" % failures)
    return 1 if failures else 0


def testUsdviewInputFunction(appController):
    model = getattr(appController, "_dataModel", appController)
    return run(model.stage)


if __name__ == "__main__":
    # Direct stage run (no usdview): python testPomadeAbiTube.py stage.usda
    from pxr import Usd
    if len(sys.argv) != 2:
        print("usage: testPomadeAbiTube.py <stage.usda>")
        sys.exit(2)
    sys.exit(run(Usd.Stage.Open(sys.argv[1])))
