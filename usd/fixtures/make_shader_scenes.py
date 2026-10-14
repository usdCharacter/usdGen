#!/usr/bin/env python3
"""Shader/viewport/render benchmark scenes (autoresearch shaderrender round 1).

Eight scenes under usd/fixtures/, the benchmark matrix:

    {sr,ss,gr,gs}{1m,10m}.usda

    s/g   scatter-grow groom vs guide-interpolated groom
    r/s   deformed via RBF driver curves vs by following a deforming surface
    1m    1M hairs (1000x1000 unit-quad scalp, density 1)
    10m   10M hairs (same scalp, density 10)

Every scene: a UsdGenGroom (backend=cpu: the CUDA lane has no stock-Storm
device-resident display handoff, so only CPU cooks reach the viewport) with
one UsdGenDescription. The op chain mirrors the M1 fixtures (Scatter ->
Grow/GuideInterpolate -> Noise -> Length -> Width) with a terminal UsdGenDeform:

    rbf      8x8 animated driver curves + Deform(guides, rbfSamples=100,
             lockRoots=true), 12 frames of a traveling wave
    surface  time-sampled scalp points (4 samples of a standing wave over a
             gently curved rest pose: the RBF binds at rest and rejects a
             flat rest as coplanar) + UsdGenRestAPI/primvars:rest + Deform
             (no guides, rbfSamples=100, lockRoots=false so roots follow
             the skin). Curvature keeps face areas within ~5% of 1.0, so
             surface scenes hold ~1-2% over nominal count (actuals reported
             per scene); rbf scenes are exact.

Guides scenes author 32x32 static guide curves (purpose=guide, so Storm
skips them) and interpolate with maxGuides=3, influenceRadius=60.

Unlike the G1-G5 fixtures these scenes carry lights (key + fill
DistantLight: hair-black-on-black would make pixel checksums insensitive)
and frame the whole scalp (Storm clamps strands to >=1px via the published
minScreenSpaceWidths constant, so every hair shades).

--small emits the same matrix on a 100x100 grid (10k/100k hairs) for fast
iteration: {sr,ss,gr,gs}{010k,100k}.usda.

Regenerate: python3 usd/fixtures/make_shader_scenes.py [--small]
"""

import math
import os
import sys

SEGMENTS = 8
CHAIN_SEED = 7
GUIDE_GRID = 32
DRIVER_GRID = 8

# (filename stem, curves, nx, ny, density)
FULL = [
    ("1m", 1000000, 1000, 1000, 1),
    ("10m", 10000000, 1000, 1000, 10),
]
SMALL = [
    ("010k", 10000, 100, 100, 1),
    ("100k", 100000, 100, 100, 10),
]


def _fmt(f):
    s = "%.4g" % f
    return "0" if s == "-0" else s


def _rest_height(x, z):
    # Curved rest pose for surface-deform scenes: the RBF must bind at rest,
    # and a flat rest is coplanar (rbfField.cpp rejects it). Two mixed modes
    # keep every sample neighborhood full-rank. Amplitude ~5 over a 1000-unit
    # scalp: face areas stay within ~5% of 1.0, so density-1 counts stay
    # within a few percent of nominal (actuals are reported per scene).
    return (4.0 * math.sin(2.0 * math.pi * x / 250.0)
            * math.cos(2.0 * math.pi * z / 170.0)
            + 1.5 * math.sin(2.0 * math.pi * x / 90.0 + 1.3)
            * math.cos(2.0 * math.pi * z / 60.0 + 0.7))


def _mesh_points(nx, ny, wave=None, rest=None):
    # wave: (amp, wavelength, phase) added as y += amp*sin(2*pi*x/wl + phase).
    # rest: when True, the base pose is the curved _rest_height instead of
    # flat (surface scenes: rest + wave; Default points and primvars:rest
    # both carry the curved rest).
    pts = []
    for j in range(ny + 1):
        z = j - ny / 2.0
        for i in range(nx + 1):
            x = i - nx / 2.0
            y = _rest_height(x, z) if rest else 0.0
            if wave is not None:
                amp, wl, ph = wave
                y += amp * math.sin(2.0 * math.pi * x / wl + ph)
            pts.append("(%s, %s, %s)" % (_fmt(x), _fmt(y), _fmt(z)))
    return pts


def _mesh_block(nx, ny, time_samples=None, rest_api=False):
    # time_samples: list of wave tuples, one per 1-based frame; when given,
    # Default points carry the curved rest and each sample is rest + wave,
    # authored under points.timeSamples. rest_api adds primvars:rest (the
    # curved rest, for the deform bind and the scatter UV bind).
    lines = []
    lines.append("        int[] faceVertexCounts = [%s]"
                 % ", ".join(["4"] * (nx * ny)))
    idx = []
    npx = nx + 1
    for j in range(ny):
        for i in range(nx):
            v = j * npx + i
            idx += [v, v + npx, v + npx + 1, v + 1]  # CCW from +Y (normal up)
    lines.append("        int[] faceVertexIndices = [%s]"
                 % ", ".join(str(k) for k in idx))
    curved = bool(time_samples)
    lines.append("        point3f[] points = [%s]"
                 % ", ".join(_mesh_points(nx, ny, rest=curved)))
    if time_samples:
        lines.append("        point3f[] points.timeSamples = {")
        for f, wave in enumerate(time_samples, start=1):
            lines.append("            %d: [%s]," % (
                f, ", ".join(_mesh_points(nx, ny, wave, rest=True))))
        lines.append("        }")
    if rest_api:
        lines.append("        point3f[] primvars:rest = [%s] ("
                     % ", ".join(_mesh_points(nx, ny, rest=True)))
        lines.append('            interpolation = "vertex"')
        lines.append("        )")
    st = []
    for j in range(ny + 1):
        for i in range(npx):
            st.append("(%s, %s)" % (_fmt(i / nx), _fmt(j / ny)))
    lines.append("        texCoord2f[] primvars:st = [%s] ("
                 % ", ".join(st))
    lines.append('            interpolation = "vertex"')
    lines.append("        )")
    return "\n".join(lines + [""])


def _guides_block(nx, guides=GUIDE_GRID, surface=False):
    # Static upright guides with deterministic lean; resampled to
    # SEGMENTS CVs by the interpolator, so author SEGMENTS directly.
    # surface: plant roots on the curved rest (gs scenes).
    n = guides
    counts = ", ".join([str(SEGMENTS)] * (n * n))
    span = float(nx)
    step = span / n
    rows = []
    for gj in range(n):
        for gi in range(n):
            gx = -span / 2.0 + (gi + 0.5) * step
            gz = -span / 2.0 + (gj + 0.5) * step
            gy = _rest_height(gx, gz) if surface else 0.0
            lean = 0.25 * math.sin(gi * 0.7) * math.cos(gj * 0.9)
            lean_z = 0.25 * math.cos(gi * 0.5) * math.sin(gj * 1.1)
            pts = []
            for c in range(SEGMENTS):
                t = c / (SEGMENTS - 1.0)
                pts.append("(%s, %s, %s)" % (
                    _fmt(gx + lean * t), _fmt(gy + 1.2 * t),
                    _fmt(gz + lean_z * t)))
            rows.append(", ".join(pts))
    return ("    def BasisCurves \"Guides\" (\n"
            '        prepend apiSchemas = ["UsdGenCurveAPI"]\n'
            "    )\n"
            "    {\n"
            '        token purpose = "guide"\n'
            '        uniform token type = "linear"\n'
            '        uniform token wrap = "nonperiodic"\n'
            "        int[] curveVertexCounts = [%s]\n"
            "        point3f[] points = [\n"
            "            %s\n"
            "        ]\n"
            "    }\n") % (counts, ",\n            ".join(rows))


def _drivers_block(nx, frames=12):
    # Upright driver grid animated by a traveling +X wave (tip moves most).
    n = DRIVER_GRID
    span = float(nx)
    step = span / n
    cvs = 5
    counts = ", ".join([str(cvs)] * (n * n))

    def pose(t):
        rows = []
        for gj in range(n):
            for gi in range(n):
                gx = -span / 2.0 + (gi + 0.5) * step
                gz = -span / 2.0 + (gj + 0.5) * step
                w = 5.0 * math.sin(2.0 * math.pi * gx / 400.0
                                   - 2.0 * math.pi * t / frames)
                pts = []
                for c in range(cvs):
                    t01 = c / (cvs - 1.0)
                    pts.append("(%s, %s, %s)" % (
                        _fmt(gx + 0.3 * w * t01 * t01),
                        _fmt(2.0 * t01 + w * t01 * t01),
                        _fmt(gz)))
                rows.append(", ".join(pts))
        return ",\n            ".join(rows)

    out = ["    def BasisCurves \"Drivers\" ("]
    out.append('        prepend apiSchemas = ["UsdGenCurveAPI"]')
    out.append("    )")
    out.append("    {")
    out.append('        uniform token type = "linear"')
    out.append('        uniform token wrap = "nonperiodic"')
    out.append("        int[] curveVertexCounts = [%s]" % counts)
    out.append("        point3f[] points = [")
    out.append("            %s" % pose(0))
    out.append("        ]")
    out.append("        point3f[] points.timeSamples = {")
    for f in range(1, frames + 1):
        out.append("            %d: [" % f)
        out.append("            %s" % pose(f))
        out.append("            ],")
    out.append("        }")
    out.append("    }")
    return "\n".join(out) + "\n"


def render(stem, mode, deform, nx, ny, density):
    label = "%s%s%s" % (mode, deform, stem)
    rbf = deform == "r"
    frames = 12 if rbf else 4
    out = []
    out.append("#usda 1.0")
    out.append("(")
    out.append('    defaultPrim = "Groom"')
    out.append("    metersPerUnit = 1")
    out.append('    upAxis = "Y"')
    out.append("    startTimeCode = 1")
    out.append("    endTimeCode = %d" % frames)
    out.append('    documentation = "shader/render benchmark %s. '
               'Generated by usd/fixtures/make_shader_scenes.py; do not '
               'hand-edit."' % label)
    out.append(")")
    out.append("")
    out.append('def UsdGenGroom "Groom"')
    out.append("{")
    out.append('    uniform string usdGen:sessionId = "%s"' % label)
    out.append('    uniform token usdGen:execution:backend = "cpu"')
    out.append("")
    if rbf:
        out.append('    def Mesh "Scalp"')
    else:
        out.append('    def Mesh "Scalp" (')
        out.append('        prepend apiSchemas = ["UsdGenRestAPI"]')
        out.append("    )")
    out.append("    {")
    out.append('        uniform token subdivisionScheme = "none"')
    if rbf:
        out.append(_mesh_block(nx, ny))
    else:
        # Standing wave: frames 0,+A,0,-A at 1..4 (USD linearly interpolates
        # between authored samples; the bench cycles all four).
        waves = [(0.0, 200.0, 0.0), (5.0, 200.0, 0.0),
                 (0.0, 200.0, 0.0), (5.0, 200.0, math.pi)]
        out.append(_mesh_block(nx, ny, time_samples=waves, rest_api=True))
    out.append("    }")
    out.append("")
    out.append('    def UsdGenDescription "Description"')
    out.append("    {")
    out.append("        uniform int usdGen:tileTarget = 64")
    out.append("        rel usdGen:surface = </Groom/Scalp>")
    out.append("")
    out.append('        def Scope "Ops"')
    out.append("        {")
    # Top-to-bottom authoring; evaluator runs bottom-to-top, so Deform
    # (top) is the terminal.
    ops = [
        ("Deform", CHAIN_SEED, [
            "int usdGen:rbfSamples = 100",
            "bool usdGen:lockRoots = %s" % ("true" if rbf else "false"),
        ] + (["rel usdGen:guides = </Drivers>"] if rbf else [])),
        ("Width", CHAIN_SEED, ["float usdGen:width = 0.01"]),
        ("Length", CHAIN_SEED, ["float usdGen:length:value = 1.0"]),
        ("Noise", CHAIN_SEED, ["float usdGen:noise:magnitude = 0.05",
                               "float usdGen:noise:frequency = 3.0"]),
    ]
    if mode == "s":
        ops.append(("Grow", CHAIN_SEED, ["int usdGen:segments = 8",
                                         "float usdGen:length = 1.0"]))
    else:
        ops.append(("GuideInterpolate", CHAIN_SEED, [
            "rel usdGen:guides = </Guides>",
            "int usdGen:cvCount = 8",
            "int usdGen:maxGuides = 3",
            "float usdGen:influenceRadius = 60.0",
            "float usdGen:influenceDecay = 2.0",
            'uniform token usdGen:blendMethod = "extrudeAndBlend"',
        ]))
    ops.append(("Scatter", 21, ["float usdGen:density = %s" % float(density)]))
    for op, seed, params in ops:
        out.append('            def UsdGen%s "%s"' % (op, op))
        out.append("            {")
        out.append("                uniform int usdGen:seed = %d" % seed)
        for p in params:
            if p.startswith("rel "):
                out.append("                %s" % p)
            else:
                out.append("                %s" % p)
        out.append("            }")
    out.append("        }")
    out.append("    }")
    out.append("}")
    out.append("")
    if mode == "g":
        out.append(_guides_block(nx, surface=(not rbf)))
        out.append("")
    if rbf:
        out.append(_drivers_block(nx))
        out.append("")
    # Lights: key + fill so hair shading is visible (checksums sensitive).
    out.append('def DistantLight "Key"')
    out.append("{")
    out.append("    float inputs:intensity = 10")
    out.append("    bool inputs:normalize = 1")
    out.append("    color3f inputs:color = (1, 0.98, 0.94)")
    out.append("    float3 xformOp:rotateXYZ = (-10, 40, 0)")
    out.append('    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]')
    out.append("}")
    out.append('def DistantLight "Fill"')
    out.append("{")
    out.append("    float inputs:intensity = 2")
    out.append("    bool inputs:normalize = 1")
    out.append("    color3f inputs:color = (0.9, 0.95, 1)")
    out.append("    float3 xformOp:rotateXYZ = (-30, -50, 0)")
    out.append('    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]')
    out.append("}")
    # Whole-scalp framing (G1-G5 formula).
    extent = float(max(nx, ny))
    dist = extent * 1.35
    height = dist * 0.45
    pitch = -math.degrees(math.atan2(height, dist))
    out.append('def Camera "Cam"')
    out.append("{")
    out.append("    float focalLength = 35")
    out.append("    float horizontalAperture = 36")
    out.append("    float2 clippingRange = (%s, %s)"
               % (_fmt(dist / 100.0), _fmt(dist * 20.0)))
    out.append("    double3 xformOp:translate = (0, %s, %s)"
               % (_fmt(height), _fmt(dist)))
    out.append("    float3 xformOp:rotateXYZ = (%s, 0, 0)" % _fmt(pitch))
    out.append('    uniform token[] xformOpOrder = ["xformOp:translate", '
               '"xformOp:rotateXYZ"]')
    out.append("}")
    out.append("")
    return "\n".join(out)


def main():
    small = "--small" in sys.argv[1:]
    table = SMALL if small else FULL
    root = os.path.dirname(os.path.abspath(__file__))
    for stem, curves, nx, ny, density in table:
        for mode in ("s", "g"):
            for deform in ("r", "s"):
                label = "%s%s%s" % (mode, deform, stem)
                path = os.path.join(root, label + ".usda")
                with open(path, "w") as f:
                    f.write(render(stem, mode, deform, nx, ny, density))
                print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))


if __name__ == "__main__":
    sys.exit(main())
