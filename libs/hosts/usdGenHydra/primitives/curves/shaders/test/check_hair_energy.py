#!/usr/bin/env python3
"""Energy sanity check for the host renderer hair BSDF ported to OpenUSD Storm.

What this checks
----------------
The BSDF is the host renderer's ``HairShading()`` from ``HairBsdf.ush`` -- Brian
Karis, "Physically Based Hair Shading in a host renderer", SIGGRAPH 2016 Physically
Based Shading in Theory and Practice course -- transcribed in
fit_hair_scattering.py, which this module imports (both the reference
integrator and the fitted dual-scattering closed forms).

With ``sin(theta) = dot(N, L)`` around the strand tangent, the solid-angle
measure on the fibre is ``cos(theta) d(theta) d(phi)`` and the directional
albedo is

    A(V) = INT_{-pi/2}^{pi/2} INT_0^{2pi} S(V, L(theta,phi)) cos(theta) dtheta dphi

The script integrates that exactly (Gauss-Legendre, converged to ~1e-4) and
checks:

  1. A(V) stays within the model's measured headroom for a spread of
     roughness / baseColor / view angles.
  2. White furnace (baseColor -> 1): A is close to but not above 1 in the
     cos(theta)-weighted average over incidence, and never grows with
     roughness.
  3. The R / TT / TRT split matches the behaviour Karis describes.
  4. The fitted ``HairAvgForward`` + ``HairAvgBackward`` never sum above 1.

Karis's model is *not* strictly energy conserving per view direction: each
lobe's azimuthal profile is separately normalised to about one, and the R
lobe picks up near-total grazing Fresnel, so A(V) overshoots unity by up to
about 15% for white hair at grazing incidence and the lowest roughness.  That
is a property of the reference model, not of this port; the tolerances below
are the measured headroom and the printed tables show the actual numbers.

usdGenShaders/resources/shaders/usdGenHairStrands.glslfx MUST STAY IN SYNC
with this file and with fit_hair_scattering.py.

Exits 0 when every check passes, 1 otherwise.  NumPy only, a second or two.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_hair_scattering as hair  # noqa: E402

# Tolerances.  Each is the measured headroom of the reference model, not a
# knob: see the tables this script prints.
TOL_VIEW = 0.16     # per view direction; worst measured A(V) is 1.151
TOL_NORMAL = 0.01   # at theta_v = 0 the model sits within 0.2% of unity
TOL_SPHERE = 2e-3   # cos-weighted over theta_i it never exceeds 1
TOL_MONO = 1e-4     # slack on "A must not grow with roughness"
TOL_FIT = 0.01      # fitted a_f + a_b, covering the fits' own error

VIEW_ANGLES_DEG = (0.0, 20.0, 45.0, 60.0, 75.0, -60.0)
CASES = (  # (roughness, baseColor channel)
    (0.05, 0.02), (0.30, 0.02), (0.30, 0.20), (0.30, 0.50),
    (0.60, 0.50), (0.10, 0.98), (0.50, 0.98), (1.00, 0.98),
)

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def theta_quadrature(n=9):
    """Gauss nodes in u = sin(theta); the weights are the cos(theta) measure."""
    u, w = np.polynomial.legendre.leggauss(n)
    return np.arcsin(u), w / 2.0


# ---------------------------------------------------------------------------
# 1. Directional albedo
# ---------------------------------------------------------------------------
def check_albedo():
    print("[1] directional albedo A(V) = INT S cos(theta) dtheta dphi")
    print("    rough  colour |" + "".join("  %6.1f" % d for d in VIEW_ANGLES_DEG)
          + " |   R      TT     TRT   (shares at theta_v = 0)")
    worst = (0.0, None)
    ok_all = True
    for rough, colour in CASES:
        cols = np.array([colour])
        row = []
        for deg in VIEW_ANGLES_DEG:
            a = float(hair.albedo(np.radians(deg), rough, cols)[0])
            row.append(a)
            if a > worst[0]:
                worst = (a, (rough, colour, deg))
        lobes = hair.lobe_albedo(0.0, rough, cols).sum(axis=2)[:, 0]
        share = lobes / lobes.sum()
        bad = max(row) > 1.0 + TOL_VIEW
        ok_all = ok_all and not bad
        print("    %5.2f  %5.2f  |" % (rough, colour)
              + "".join("  %6.4f" % a for a in row)
              + " |  %.3f  %.3f  %.3f%s"
              % (share[0], share[1], share[2], "   <== OVER" if bad else ""))
    print("    worst A(V) = %.4f at roughness %.2f, colour %.2f, theta_v %.0f deg"
          % (worst[0], worst[1][0], worst[1][1], worst[1][2]))
    record("A(V) <= 1 + %.2f for every probe" % TOL_VIEW, ok_all,
           "worst %.4f" % worst[0])
    return ok_all


# ---------------------------------------------------------------------------
# 2. White furnace
# ---------------------------------------------------------------------------
def check_white_furnace():
    print("\n[2] white furnace, baseColor -> 1")
    thetas, weights = theta_quadrature(9)
    roughs = np.linspace(0.05, 1.0, 10)
    cols = np.array([1.0])
    mean = np.zeros(roughs.size)
    peak = np.zeros(roughs.size)
    normal = np.zeros(roughs.size)
    for ri, rg in enumerate(roughs):
        a = np.array([float(hair.albedo(t, rg, cols)[0]) for t in thetas])
        mean[ri] = float((a * weights).sum())
        peak[ri] = float(a.max())
        normal[ri] = float(hair.albedo(0.0, rg, cols)[0])
    print("    rough | A over theta_i | A at theta_i=0 | max over theta_i")
    for ri, rg in enumerate(roughs):
        print("    %5.2f |     %8.5f   |    %8.5f    |    %8.5f"
              % (rg, mean[ri], normal[ri], peak[ri]))

    ok = record("white A (cos-weighted over theta_i) <= 1 + %.0e" % TOL_SPHERE,
                mean.max() <= 1.0 + TOL_SPHERE, "max %.5f" % mean.max())
    ok &= record("white A at theta_i = 0 <= 1 + %.2f" % TOL_NORMAL,
                 normal.max() <= 1.0 + TOL_NORMAL, "max %.5f" % normal.max())
    ok &= record("white A close to 1 at the smoothest roughness",
                 mean[0] > 0.95, "A = %.5f at r = %.2f" % (mean[0], roughs[0]))
    d = np.diff(mean)
    ok &= record("white A does not grow with roughness (cos-weighted)",
                 d.max() <= TOL_MONO, "largest increase %.2e" % d.max())
    dp = np.diff(peak)
    ok &= record("white A does not grow with roughness (worst view angle)",
                 dp.max() <= TOL_MONO, "largest increase %.2e" % dp.max())
    return ok


# ---------------------------------------------------------------------------
# 3. Lobe split vs. published Karis behaviour
# ---------------------------------------------------------------------------
def check_lobes():
    print("\n[3] R / TT / TRT split (absorption is c^~0.5 for TT, c^~0.8 for TRT)")
    dark, white = 0.02, 0.98
    cols = np.array([dark, white])
    probes = [(r, d) for r in (0.05, 0.3, 0.6, 1.0) for d in (0.0, 45.0)]
    print("    rough  theta |          dark c=0.02          |         white c=0.98")
    print("                 |     R      TT     TRT   share_R |     R      TT     TRT"
          "   share_R")
    rows = []
    for rough, deg in probes:
        m = hair.lobe_albedo(np.radians(deg), rough, cols)
        tot = m.sum(axis=2)                       # (lobe, colour)
        d_lobes, w_lobes = tot[:, 0], tot[:, 1]
        rows.append((rough, deg, m, d_lobes, w_lobes))
        print("    %5.2f  %5.1f |  %.4f  %.4f  %.4f   %.3f |  %.4f  %.4f  %.4f   %.3f"
              % (rough, deg,
                 d_lobes[0], d_lobes[1], d_lobes[2], d_lobes[0] / d_lobes.sum(),
                 w_lobes[0], w_lobes[1], w_lobes[2], w_lobes[0] / w_lobes.sum()))

    ok = True
    # R carries no colour at all, so darkening the hair can only raise its share.
    r_same = max(abs(r[3][0] - r[4][0]) for r in rows)
    ok &= record("R lobe is exactly colour independent", r_same < 1e-12,
                 "max |R(dark) - R(white)| = %.2e" % r_same)

    ratio = min(r[3][0] / r[3].sum() / (r[4][0] / r[4].sum()) for r in rows)
    ok &= record("dark hair lifts the R share at least 3x over white",
                 ratio >= 3.0, "smallest lift %.2fx" % ratio)

    # TRT travels a longer path (0.8/CosThetaD vs ~0.5/CosThetaD), so absorption
    # kills it faster than TT -- the classic "dark hair loses its secondary
    # highlight first" behaviour.
    worst = min((r[3][1] / r[4][1]) - (r[3][2] / r[4][2]) for r in rows)
    ok &= record("absorption suppresses TRT faster than TT", worst > 0.0,
                 "smallest (TT - TRT) survival margin %.3f" % worst)

    tt_share = min(r[2][1, 1, hair.HALF_FORWARD] /
                   r[2][:, 1, hair.HALF_FORWARD].sum() for r in rows)
    ok &= record("white hair: forward (transmission) half is TT dominated",
                 tt_share > 0.8, "smallest TT share of a_f %.3f" % tt_share)

    r_share = min(r[2][0, 0, hair.HALF_BACKWARD] /
                  r[2][:, 0, hair.HALF_BACKWARD].sum() for r in rows)
    ok &= record("dark hair: backward (reflection) half is R dominated",
                 r_share > 0.8, "smallest R share of a_b %.3f" % r_share)

    trt_over_r = min(r[2][2, 1, hair.HALF_BACKWARD] -
                     r[2][0, 1, hair.HALF_BACKWARD] for r in rows)
    ok &= record("white hair: TRT outshines R in the reflection half",
                 trt_over_r > 0.0, "smallest TRT - R margin %.4f" % trt_over_r)
    print("    note: for dark hair TT still carries more absolute energy than R at")
    print("          head-on incidence (c^0.5 beats c^0.8), but R's share of A rises")
    print("          several-fold and R owns the reflection half -- that is what")
    print("          'dark hair is R dominated' means for the visible highlight.")
    return ok


# ---------------------------------------------------------------------------
# 4. The fitted closed forms
# ---------------------------------------------------------------------------
def check_fits():
    print("\n[4] fitted closed forms from fit_hair_scattering.py")
    # The full grid, i.e. the constants the GLSL snippet actually ships.
    sc = hair.solve(fast=False)
    fits = hair.fit_all(sc)
    a_f, a_b = fits["a_f"], fits["a_b"]
    print("    a_f fit: max rel err %.2f%%  rms %.2f%%"
          % (100 * a_f.max_rel, 100 * a_f.rms_rel))
    print("    a_b fit: max rel err %.2f%%  rms %.2f%%"
          % (100 * a_b.max_rel, 100 * a_b.rms_rel))

    # Evaluated past the fit domain, at the full range the shader can feed in.
    rr = np.linspace(1.0 / 255.0, 1.0, 41)
    cc = np.linspace(0.0, 1.0, 41)
    R, C = np.meshgrid(rr, cc, indexing="ij")
    tot = a_f(R, C) + a_b(R, C)
    print("    fitted a_f + a_b over a %dx%d (roughness, colour) grid: "
          "min %.4f  max %.4f" % (rr.size, cc.size, tot.min(), tot.max()))
    ok = record("fitted a_f + a_b <= 1 + %.2f everywhere" % TOL_FIT,
                tot.max() <= 1.0 + TOL_FIT, "max %.4f" % tot.max())
    ok &= record("fitted a_f, a_b stay non-negative",
                 a_f(R, C).min() >= 0.0 and a_b(R, C).min() >= 0.0,
                 "min %.4f / %.4f" % (a_f(R, C).min(), a_b(R, C).min()))

    # The fits must also bracket the integrated truth on the solve grid.
    err_f = np.abs(a_f(*np.meshgrid(sc.roughs, sc.colors, indexing="ij"))
                   / sc.a_bar[:, :, hair.HALF_FORWARD] - 1.0).max()
    err_b = np.abs(a_b(*np.meshgrid(sc.roughs, sc.colors, indexing="ij"))
                   / sc.a_bar[:, :, hair.HALF_BACKWARD] - 1.0).max()
    ok &= record("a_f / a_b fits within 5% of the integrated albedo",
                 max(err_f, err_b) < 0.05,
                 "max rel err %.2f%% / %.2f%%" % (100 * err_f, 100 * err_b))

    v_f, v_b = fits["beta2_f"], fits["beta2_b"]
    ok &= record("beta_f^2 / beta_b^2 fits within 10% of the target",
                 max(v_f.max_rel, v_b.max_rel) < 0.10,
                 "max rel err %.2f%% / %.2f%%"
                 % (100 * v_f.max_rel, 100 * v_b.max_rel))
    ok &= record("beta_f^2, beta_b^2 stay positive over roughness",
                 min(v_f(rr).min(), v_b(rr).min()) > 0.0,
                 "min %.5f / %.5f" % (v_f(rr).min(), v_b(rr).min()))
    return ok


def main():
    print("UE hair BSDF (HairBsdf.ush / Karis 2016) -- CPU energy check")
    print("Specular=%.2f  Backlit=%.2f  Area=%.2f  n=%.2f\n"
          % (hair.SPECULAR, hair.BACKLIT, hair.AREA, hair.IOR))
    check_albedo()
    check_white_furnace()
    check_lobes()
    check_fits()

    width = max(len(n) for n, _, _ in RESULTS)
    print("\n" + "=" * (width + 30))
    failed = 0
    for name, ok, detail in RESULTS:
        failed += not ok
        print("  %-4s %-*s  %s" % ("PASS" if ok else "FAIL", width, name, detail))
    print("=" * (width + 30))
    print("  %d checks, %d failed" % (len(RESULTS), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
