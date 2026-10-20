#!/usr/bin/env python3
# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
"""Reference implementation and table generator for the Marschner strand model.

usdGenShaders/resources/shaders/usdGenHairStrandsMarschner.glslfx is the Storm
port of an absorption-based Marschner strand model (the one the groom importer
targets; see the importer README, "Hair shading"). This script is
its CPU reference. It does three things:

* ``reference_bsdf()`` evaluates the R / TT / TRT model in NumPy with the same
  expressions the shader uses, so the shader's constants have one readable
  source.
* It rebuilds the model's multiple-scattering table: the hemispherical
  averages a_f, a_b (forward / backward scattering of the single-scattering
  BSDF) and the two forward columns the forward-scatter term reads, for the
  32 x 16 x 16 lattice (|sin theta_L|, roughness, absorption coordinate) the
  model defines, half-float rounded and clamped to [0, 1] like the texture it
  replaces. The integrals are done by Gauss-Legendre quadrature instead of the
  reference's golden-ratio Monte Carlo.
* It resamples that lattice onto the much smaller table the shader carries
  (12 incidence nodes uniform in angle x 5 roughness nodes x the 16 absorption
  nodes) and prints it as GLSL, or with --check verifies that the glslfx
  holds exactly that table and that its trilinear lookup stays within the
  stated error of the full lattice.

Conventions mirrored from the model (each one is deliberate, not a slip):
  * The table is built at IOR 1.55, cuticle shift 0.03 rad and azimuthal
    roughness 0.35 -- the model's defaults. The model builds it from its first
    material, not the shaded one, so a material's own IOR, tilt and azimuthal
    roughness reach the direct lobes but not multiple scattering.
  * The absorption axis is BUILT at sigma = coord^2 but READ at coord = sigma
    (per channel, clamped to [0, 1]). That brightens multiple scattering for
    dark hair; the port keeps it.
  * The roughness axis is read at clamp(v, 0.18, 0.6), v the longitudinal
    variance; the table's lobes are B = {beta, beta/2, 2 beta} at beta = coord.
  * The TT azimuthal term of the table is the fixed fit exp(-3.65 cos(phi)
    - 3.98); the direct lobe uses the model's 64 x 64 table instead
    (``ntt_table``), which is closed-form and needs no data.

Usage:
    python tabulate_hair_marschner.py            # print the GLSL table
    python tabulate_hair_marschner.py --check [glslfx]
NumPy only. Exits 77 (ctest SKIP) when NumPy is missing.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    print("SKIP: NumPy is not available")
    sys.exit(77)

HERE = os.path.dirname(os.path.abspath(__file__))
GLSLFX = os.path.join(HERE, "..", "resources", "shaders",
                      "usdGenHairStrandsMarschner.glslfx")

# The model's defaults, which are also what its table is built at.
IOR = 1.55
TILT = 0.03          # cuticle shift s, radians: R at -s, TT at s/2, TRT at 1.5 s
AZIMUTHAL = 0.35

# The texture the shader's table replaces: 32 x 16 x 16, texel centres at
# (i + 0.5) / n, sampled with clamp.
LAT_T, LAT_R, LAT_A = 32, 16, 16
# Rows of the roughness axis a lookup at clamp(v, 0.18, 0.6) can touch.
LAT_ROWS = range(2, 11)

# The shader's table.
TAB_T, TAB_R, TAB_A = 12, 5, 16
T_LO, T_HI = 1.0 / 64.0, 63.0 / 64.0
R_LO, R_HI = 0.18, 0.6

# Stated accuracy of the shader table against the full lattice (checked).
MAX_ABS = {"af": 0.012, "ab": 0.003, "nr": 1e-4, "ntt": 0.002}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def f0_of(ior):
    return ((1.0 - ior) / (1.0 + ior)) ** 2


def fresnel(f0, cos_theta):
    return f0 + (1.0 - f0) * (1.0 - cos_theta) ** 5


def strand_mp(width, x):
    """Normalised longitudinal Gaussian; `width` is a standard deviation in
    sum-of-sines units (theta_h = sin(theta_L) + sin(theta_V))."""
    width = np.maximum(width, 1e-5)
    return np.exp(-0.5 * (x / width) ** 2) * 0.39894228 / width


def ntt_table(sin_theta_l, azimuthal):
    """The direct TT azimuthal term: a 64 x 64 two-channel table sampled
    bilinearly with clamp at (0.5 - 0.5 sin(theta_L), sqrt(azimuthal)).
    Row y holds R = 1 / (max(0.05, 2 (y/63)^2) sqrt(2 pi)), column x holds
    G = 0.5 - x/63, and the lobe is R exp(-(cos(phi) + G)^2)."""
    u = 0.5 - 0.5 * np.asarray(sin_theta_l, dtype=float)
    x = np.clip(64.0 * u - 0.5, 0.0, 63.0)
    g = 0.5 - x / 63.0
    y = np.clip(64.0 * math.sqrt(azimuthal) - 0.5, 0.0, 63.0)
    y0 = math.floor(y)
    y1 = min(y0 + 1.0, 63.0)

    def row(r):
        return 1.0 / (max(0.05, 2.0 * (r / 63.0) ** 2) * 2.5066311)
    r = row(y0) + (row(y1) - row(y0)) * (y - y0)
    return r, g


def chiang_d(beta_n=0.3):
    b2 = beta_n * beta_n
    return (5.969 - 0.215 * beta_n + 2.532 * b2 - 10.73 * b2 * beta_n
            + 5.574 * b2 * b2 + 0.245 * b2 * b2 * beta_n)


def albedo_to_absorption(albedo):
    l = np.log(np.maximum(albedo, 1e-6)) / chiang_d()
    return l * l


def reference_bsdf(V, L, T, sigma, B, ior=IOR, azimuthal=AZIMUTHAL,
                   tilt=TILT, specular=1.0, tt_scale=1.0, table_tt=False):
    """S, nr, ntt for unit V, T (3,) and L (..., 3). `T` runs root -> tip.
    `table_tt` selects the fixed TT azimuthal fit the scattering table uses
    instead of the direct ntt_table()."""
    f0 = f0_of(ior)
    azimuthal = max(azimuthal, 0.02)
    VoL = L @ V
    sl = L @ T
    sv = float(V @ T)
    c1 = math.sqrt(max(0.0, 1.0 - sv * sv))
    c2 = np.sqrt(np.clip(1.0 - sl * sl, 0.0, 1.0))
    cd = np.sqrt(0.5 + (c1 * c2 + sv * sl) * 0.5)
    th = sl + sv
    lp = L - sl[..., None] * T
    vp = V - sv * T
    cos_phi = (lp @ vp) / np.sqrt((lp * lp).sum(-1) * (vp @ vp) + 1e-4)
    cos_half = np.sqrt(np.clip(0.5 + 0.5 * cos_phi, 0.0, 1.0))
    iorp = np.sqrt(ior * ior - (1.0 - cd * cd)) / cd

    # R
    fp = fresnel(f0, np.sqrt(np.clip(0.5 + 0.5 * VoL, 0.0, 1.0)))
    np_r = 0.25 * cos_half
    R = fp * strand_mp(B[0], th + tilt) * np_r * specular
    nr = fp * np_r

    # TT
    a = 1.0 / iorp
    m = 1.0 - cos_half
    h_tt = (1.0 - m) * (1.0 - 0.5 * m * m + a * m + a * m * m - 0.15 * a)
    x = np.clip(h_tt / iorp, -1.0, 1.0)
    tp = np.exp(-2.0 * sigma * (2.0 - 2.0 * x * x))
    f = fresnel(f0, cd * np.sqrt(np.clip(1.0 - h_tt * h_tt, 0.0, 1.0)))
    if table_tt:
        distr = np.exp(-3.65 * cos_phi - 3.98)
    else:
        r, g = ntt_table(sl, azimuthal)
        distr = r * np.exp(-(cos_phi + g) ** 2)
    ntt = distr * (1.0 - f) ** 2 * tp * tt_scale
    TT = strand_mp(B[1], th - 0.5 * tilt) * ntt

    # TRT
    f = fresnel(f0, cd * 0.5)
    x = np.clip(0.866025403784439 / iorp, -1.0, 1.0)
    tr = np.exp(-2.0 * sigma * (2.0 - 2.0 * x * x))
    sr = min(max(1.5 * (1.0 - azimuthal), 0.0), 1.0)
    TRT = (sr * np.exp(sr * (17.0 * cos_phi - 16.78))
           * strand_mp(B[2], th - 1.5 * tilt) * (1.0 - f) ** 2 * f * tr * tr)

    return np.maximum(R + TT + TRT, 0.0), nr, ntt


# ---------------------------------------------------------------------------
# The multiple-scattering table
# ---------------------------------------------------------------------------
def _gauss(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def sphere_quadrature(nz=160, nphi=96):
    """Directions split by the sign of cos(phi) (the eye sits at phi = 0),
    with weights for the solid-angle measure dz dphi, z = T.L."""
    z, wz = [], []
    edges = np.linspace(-1.0, 1.0, 9)
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = _gauss(nz // 8, a, b)
        z.append(x)
        wz.append(w)
    z = np.concatenate(z)
    wz = np.concatenate(wz)
    halves = {}
    for half, (a, b) in (("back", (0.0, 0.5 * math.pi)),
                         ("front", (0.5 * math.pi, math.pi))):
        phi, wphi = [], []
        pe = np.linspace(a, b, 5)
        for p0, p1 in zip(pe[:-1], pe[1:]):
            x, w = _gauss(nphi // 8, p0, p1)
            phi.append(x)
            wphi.append(w)
        phi = np.concatenate(phi)
        wphi = 2.0 * np.concatenate(wphi)   # +phi and -phi
        Z, P = np.meshgrid(z, phi, indexing="ij")
        s = np.sqrt(np.clip(1.0 - Z * Z, 0.0, 1.0))
        L = np.stack([s * np.cos(P), s * np.sin(P), Z], -1).reshape(-1, 3)
        halves[half] = (L, np.outer(wz, wphi).reshape(-1))
    return halves


def table_entry(sin_theta, roughness, coord, quad):
    """(a_f, a_b, nr, ntt) at one lattice point, before rounding.
    a_f / a_b are the solid-angle integrals of S over the forward
    (cos(phi) < 0) and backward halves; nr / ntt are the forward-half means
    of the R and TT azimuthal-times-Fresnel(-times-transmittance) factors,
    halved, exactly as the model accumulates them."""
    c = math.sqrt(max(0.0, 1.0 - sin_theta * sin_theta))
    V = np.array([c, 0.0, sin_theta])
    T = np.array([0.0, 0.0, 1.0])
    beta = min(max(roughness, 1.0 / 255.0), 1.0)
    B = (beta, 0.5 * beta, 2.0 * beta)
    sigma = coord * coord
    Lf, wf = quad["front"]
    Lb, wb = quad["back"]
    Sf, nrf, nttf = reference_bsdf(V, Lf, T, sigma, B, table_tt=True)
    Sb, _, _ = reference_bsdf(V, Lb, T, sigma, B, table_tt=True)
    return ((Sf * wf).sum(), (Sb * wb).sum(),
            (nrf * wf).sum() / (4.0 * math.pi),
            (nttf * wf).sum() / (4.0 * math.pi))


def build_lattice(quad=None):
    quad = quad or sphere_quadrature()
    lat = np.zeros((LAT_T, LAT_R, LAT_A, 4))
    for i in range(LAT_T):
        for j in LAT_ROWS:
            for k in range(LAT_A):
                lat[i, j, k] = table_entry((i + 0.5) / LAT_T, (j + 0.5) / LAT_R,
                                           (k + 0.5) / LAT_A, quad)
    # The texture is half-float, written through saturate().
    return np.clip(lat, 0.0, 1.0).astype(np.float16).astype(np.float64)


def lattice_lookup(lat, t, r, a):
    """The texture fetch: trilinear between texel centres, clamp."""
    def axis(c, n):
        x = np.clip(np.asarray(c, dtype=float) * n - 0.5, 0.0, n - 1.0)
        i0 = np.floor(x).astype(int)
        return i0, np.minimum(i0 + 1, n - 1), x - i0
    i0, i1, ft = axis(t, LAT_T)
    j0, j1, fr = axis(r, LAT_R)
    k0, k1, fa = axis(a, LAT_A)
    out = 0.0
    for ii, wi in ((i0, 1 - ft), (i1, ft)):
        for jj, wj in ((j0, 1 - fr), (j1, fr)):
            for kk, wk in ((k0, 1 - fa), (k1, fa)):
                out = out + (wi * wj * wk)[..., None] * lat[ii, jj, kk]
    return out


def table_nodes():
    th = np.linspace(math.asin(T_LO), math.asin(T_HI), TAB_T)
    return (np.sin(th), np.linspace(R_LO, R_HI, TAB_R),
            (np.arange(TAB_A) + 0.5) / TAB_A)


def build_table(lat):
    """af/ab on (t, r, a); ntt on (t, a); nr on t. ntt and nr do not depend on
    the roughness (no longitudinal term in them), so they are read from the
    first roughness node; the check asserts that independence."""
    tn, rn, an = table_nodes()
    T3, R3, A3 = np.meshgrid(tn, rn, an, indexing="ij")
    full = lattice_lookup(lat, T3, R3, A3)
    return {"afab": full[..., :2], "ntt": full[:, 0, :, 3],
            "nr": full[:, 0, 0, 2], "full": full}


def shader_lookup(tab, t, r, sigma):
    """What the GLSL does, in NumPy (vectorised over the queries)."""
    th = np.arcsin(np.clip(t, T_LO, T_HI))
    th0, th1 = math.asin(T_LO), math.asin(T_HI)
    ft = (th - th0) / (th1 - th0) * (TAB_T - 1)
    t0 = np.minimum(ft.astype(int), TAB_T - 2)
    wt = ft - t0
    fr = (np.clip(r, R_LO, R_HI) - R_LO) / (R_HI - R_LO) * (TAB_R - 1)
    r0 = np.minimum(fr.astype(int), TAB_R - 2)
    wr = fr - r0
    fa = np.clip(np.asarray(sigma) * TAB_A - 0.5, 0.0, TAB_A - 1.0)
    a0 = np.minimum(fa.astype(int), TAB_A - 2)
    wa = fa - a0
    afab = 0.0
    ntt = 0.0
    for dt, w1 in ((0, 1 - wt), (1, wt)):
        for da, w3 in ((0, 1 - wa), (1, wa)):
            ntt = ntt + w1 * w3 * tab["ntt"][t0 + dt, a0 + da]
            for dr, w2 in ((0, 1 - wr), (1, wr)):
                afab = afab + (w1 * w2 * w3)[..., None] * \
                    tab["afab"][t0 + dt, r0 + dr, a0 + da]
    nr = (1 - wt) * tab["nr"][t0] + wt * tab["nr"][t0 + 1]
    return afab[..., 0], afab[..., 1], nr, ntt


# ---------------------------------------------------------------------------
# GLSL emission and parsing
# ---------------------------------------------------------------------------
def _fmt(v):
    s = "%.5g" % v
    if "e" in s:
        s = "%.7f" % v
    if "." not in s:
        s += ".0"
    return s


def _rows(values, per_line):
    vals = [_fmt(v) for v in values]
    return ",\n".join("    " + ", ".join(vals[i:i + per_line])
                      for i in range(0, len(vals), per_line))


def emit_glsl(tab):
    afab = tab["afab"].reshape(-1)
    th0, th1 = math.asin(T_LO), math.asin(T_HI)
    out = []
    out.append("// GENERATED by usdGenShaders/test/tabulate_hair_marschner.py -- do not")
    out.append("// edit by hand; re-run the script and paste its output.")
    out.append("#define USDGEN_MARSCHNER_DS_NT %d" % TAB_T)
    out.append("#define USDGEN_MARSCHNER_DS_NR %d" % TAB_R)
    out.append("#define USDGEN_MARSCHNER_DS_NA %d" % TAB_A)
    out.append("#define USDGEN_MARSCHNER_DS_TH0 %.9f" % th0)
    out.append("#define USDGEN_MARSCHNER_DS_TH1 %.9f" % th1)
    out.append("")
    out.append("// (a_f, a_b) pairs, index ((t * NR) + r) * NA + a.")
    out.append("const float UsdGenMarschnerDsAfAb[%d] = float[%d](" % (afab.size, afab.size))
    out.append(_rows(afab, 8))
    out.append(");")
    out.append("")
    out.append("// Forward TT column, index t * NA + a.")
    ntt = tab["ntt"].reshape(-1)
    out.append("const float UsdGenMarschnerDsNtt[%d] = float[%d](" % (ntt.size, ntt.size))
    out.append(_rows(ntt, 8))
    out.append(");")
    out.append("")
    out.append("// Forward R column, index t.")
    nr = tab["nr"]
    out.append("const float UsdGenMarschnerDsNr[%d] = float[%d](" % (nr.size, nr.size))
    out.append(_rows(nr, 6))
    out.append(");")
    return "\n".join(out)


def parse_glsl_array(text, name):
    m = re.search(r"const\s+float\s+%s\[(\d+)\]\s*=\s*float\[\d+\]\((.*?)\);" % name,
                  text, re.S)
    if not m:
        return None
    vals = [float(v) for v in re.split(r"[,\s]+", m.group(2).strip()) if v]
    return np.array(vals) if len(vals) == int(m.group(1)) else None


def parse_define(text, name):
    m = re.search(r"#define\s+%s\s+([-0-9.eE]+)" % name, text)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
def check(path):
    ok = True
    quad = sphere_quadrature()
    fine = sphere_quadrature(320, 192)
    # Quadrature convergence at a narrow-lobe point.
    a = np.array(table_entry(0.5, 0.18, 0.3, quad))
    b = np.array(table_entry(0.5, 0.18, 0.3, fine))
    if np.abs(a - b).max() > 1e-4:
        print("FAIL: quadrature not converged:", a, b)
        ok = False
    else:
        print("OK: quadrature converged (max diff %.2g)" % np.abs(a - b).max())

    lat = build_lattice(quad)
    # nr / ntt carry no longitudinal term, so they must not vary along the
    # roughness axis -- what lets the shader store them without it.
    rows = lat[:, list(LAT_ROWS), :, :]
    spread = np.abs(rows[..., 2:] - rows[:, :1, :, 2:]).max()
    if spread > 1e-3:
        print("FAIL: nr/ntt vary with roughness by %.3g" % spread)
        ok = False
    else:
        print("OK: nr/ntt are independent of roughness (spread %.2g)" % spread)

    tab = build_table(lat)

    # The shader table against the full lattice, between nodes.
    rng = np.random.default_rng(7)
    n = 40000
    qt = rng.uniform(0.0, 1.0, n)
    qr = rng.uniform(R_LO, R_HI, n)
    qa = rng.uniform(0.0, 1.0, n)
    ref = lattice_lookup(lat, qt, qr, qa)
    got = shader_lookup(tab, qt, qr, qa)
    for idx, key in enumerate(("af", "ab", "nr", "ntt")):
        err = np.abs(got[idx] - ref[..., idx]).max()
        status = "OK" if err <= MAX_ABS[key] else "FAIL"
        ok = ok and err <= MAX_ABS[key]
        print("%s: table %-3s max abs error %.4f (limit %.4f)"
              % (status, key, err, MAX_ABS[key]))

    # The glslfx must hold exactly this table.
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        print("FAIL: cannot read %s (%s)" % (path, e))
        return False
    want = {"UsdGenMarschnerDsAfAb": tab["afab"].reshape(-1),
            "UsdGenMarschnerDsNtt": tab["ntt"].reshape(-1),
            "UsdGenMarschnerDsNr": tab["nr"]}
    for name, values in want.items():
        have = parse_glsl_array(text, name)
        if have is None or have.shape != values.shape:
            print("FAIL: %s missing or wrong size in %s" % (name, os.path.basename(path)))
            ok = False
            continue
        err = np.abs(have - values).max()
        if err > 5e-5 + 1e-4 * np.abs(values).max():
            print("FAIL: %s differs from the regenerated table by %.3g "
                  "(re-run this script and paste its output)" % (name, err))
            ok = False
        else:
            print("OK: %s matches (%d values, max diff %.2g)" % (name, values.size, err))
    for name, value in (("USDGEN_MARSCHNER_DS_NT", TAB_T), ("USDGEN_MARSCHNER_DS_NR", TAB_R),
                        ("USDGEN_MARSCHNER_DS_NA", TAB_A),
                        ("USDGEN_MARSCHNER_DS_TH0", math.asin(T_LO)),
                        ("USDGEN_MARSCHNER_DS_TH1", math.asin(T_HI))):
        have = parse_define(text, name)
        if have is None or abs(have - value) > 1e-7:
            print("FAIL: #define %s is %s, want %s" % (name, have, value))
            ok = False

    # The direct TT table: closed form, so pin a few values the shader's
    # comment quotes.
    r, g = ntt_table(0.0, 0.35)
    if abs(g) > 1e-6 or abs(r - 0.5674) > 1e-3:
        print("FAIL: ntt_table(0, 0.35) = (%.4f, %.4f)" % (r, g))
        ok = False
    else:
        print("OK: direct TT table R(0.35) = %.4f, G(0) = %.1g" % (r, g))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", nargs="?", const=GLSLFX, metavar="GLSLFX",
                    help="verify the table in the glslfx instead of printing it")
    args = ap.parse_args()
    if args.check:
        ok = check(args.check)
        print("MARSCHNER TABLE CHECK:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    print(emit_glsl(build_table(build_lattice())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
