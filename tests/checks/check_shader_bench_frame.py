#!/usr/bin/env python3
# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
"""benchUsdGenShaderRender must draw a non-empty frame (r7 framing guard).

The bench once rendered every scene as an all-clear image: it never called
SetFraming (empty data window) and it rooted the render at the Groom prim
instead of the stage, excluding the Key/Fill lights. Every checksum then
verified a single artifact pixel, so no pixel comparison could catch a
shading or geometry regression.

This check renders the small sr010k scene once and fails when the dumped
frame is (near-)clear. Stdlib only, no pxr. Skips (77) when the bench
binary is missing or the host offers no headless EGL context.

Usage: check_shader_bench_frame.py <bench-binary> <scene.usda>
"""

import collections
import os
import subprocess
import sys
import tempfile

# sr010k at 320x180 draws ~39% non-clear pixels in ~140 unique colors; the
# thresholds below sit an order of magnitude under that, and three orders
# above the all-clear failure mode (1 artifact pixel).
_MIN_NONCLEAR_FRAC = 0.05
_MIN_UNIQUE_COLORS = 20


def skip(reason):
    print("SKIP: %s" % reason)
    return 77


def main():
    if len(sys.argv) != 3:
        print("usage: check_shader_bench_frame.py <bench-binary> <scene.usda>")
        return 2
    bench, scene = sys.argv[1], sys.argv[2]
    if not (os.path.isfile(bench) and os.access(bench, os.X_OK)):
        return skip("bench binary missing: %s" % bench)
    if not os.path.isfile(scene):
        return skip("scene missing: %s" % scene)
    tmp = tempfile.mkdtemp(prefix="usdgen_bench_frame_")
    ppm = os.path.join(tmp, "frame.ppm")
    js = os.path.join(tmp, "bench.json")
    proc = subprocess.run(
        [bench, "--scene", scene, "--static", "--warmup", "0", "--frames",
         "1", "--repeats", "1", "--res", "320x180", "--dump-ppm", ppm,
         "--json", js],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout or ""
    if "no headless EGL" in out:
        return skip("no headless EGL context on this host")
    if proc.returncode != 0 or not os.path.isfile(ppm):
        print("FAIL: bench run failed (rc=%d):\n%s" % (proc.returncode, out))
        return 1
    with open(ppm, "rb") as f:
        data = f.read()
    try:
        px = data[data.index(b"255\n") + 4:]
    except ValueError:
        print("FAIL: not a binary PPM dump")
        return 1
    if len(px) % 3:
        print("FAIL: truncated PPM dump (%d bytes)" % len(px))
        return 1
    colors = collections.Counter(
        px[i:i + 3] for i in range(0, len(px), 3))
    n = len(px) // 3
    nonclear = sum(v for k, v in colors.items() if k != b"\x00\x00\x00")
    frac = nonclear / n if n else 0.0
    print("frame: %d px, %d unique colors, %.3f non-clear"
          % (n, len(colors), frac))
    if frac < _MIN_NONCLEAR_FRAC:
        print("FAIL: frame is (near-)clear: %.4f non-clear pixels "
              "(want >= %.2f)" % (frac, _MIN_NONCLEAR_FRAC))
        return 1
    if len(colors) < _MIN_UNIQUE_COLORS:
        print("FAIL: frame has %d unique colors (want >= %d)"
              % (len(colors), _MIN_UNIQUE_COLORS))
        return 1
    print("ok: bench frame is lit geometry, not an empty data window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
