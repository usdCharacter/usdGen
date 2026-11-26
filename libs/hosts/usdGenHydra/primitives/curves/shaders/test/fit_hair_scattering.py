#!/usr/bin/env python3
"""Derive Zinke-2008 dual-scattering parameters for the host-renderer hair BSDF.

What this computes
------------------
The BSDF is a literal transcription of the host renderer's ``HairShading()`` from
``HairBsdf.ush`` -- Brian Karis, "Physically Based Hair Shading in a host renderer",
SIGGRAPH 2016 Physically Based Shading in Theory and Practice course.  ``N`` is
the strand tangent, ``V`` points at the eye, ``L`` at the light; the R / TT /
TRT lobes are summed with the Gaussian longitudinal term ``Hair_g`` and the
Schlick-style ``Hair_F`` (n = 1.55).

Parametrising ``L`` by ``(theta, phi)`` around the fibre with
``sin(theta) = dot(N, L)``, the solid-angle measure on the cylinder is
``cos(theta) d(theta) d(phi)`` and the directional albedo is

    A(V) = INT_{-pi/2}^{pi/2} INT_0^{2pi} S(V, L(theta,phi)) cos(theta) dtheta dphi

Splitting the azimuth in two gives the dual-scattering pair:

  * forward  half, ``CosPhi < 0``  -- light behind the fibre, TT dominated;
    this is Zinke's forward scattering ``a_f`` (the TT azimuthal profile
    ``Np = exp(-3.65*CosPhi - 3.98)`` peaks at ``CosPhi = -1``).
  * backward half, ``CosPhi > 0``  -- light and eye on the same side, R + TRT;
    Zinke's backward scattering ``a_b``.

``alpha_f``/``alpha_b`` and ``beta_f^2``/``beta_b^2`` are the energy-weighted
mean and variance of the longitudinal scattering angle
``x = SinThetaL + SinThetaV`` over each half.

Those are then fitted as compact closed forms so the GLSL shader can evaluate
them without a lookup table, and the constants are printed as a ready-to-paste
snippet in two variants:

  * theta-free: a(roughness, colour), 9 coefficients, the incidence angle
    averaged out (cos-weighted).  Cheap, and accurate in (r, c).
  * theta-resolved: a(roughness, colour, sinThetaV), 39 coefficients, chosen
    from a fixed function library by orthogonal matching pursuit under an ALU
    budget, because this is evaluated per light per fragment.

Everything the script prints about a fit is measured on the constants at the
precision it prints them, on the fitted grid AND on nodes strictly between it.

usdGenShaders/resources/shaders/usdGenHairStrands.glslfx MUST STAY IN SYNC
with this file.  If the lobe constants, the Fresnel IOR, the shift/roughness
mapping or the fitted dual-scattering helpers change in one place, re-run this
script and paste the new snippet into the other.

Usage
-----
    python usdGenShaders/test/fit_hair_scattering.py            # full grid
    python usdGenShaders/test/fit_hair_scattering.py --fast     # coarse grid

Deterministic, NumPy only.  Companion energy check: check_hair_energy.py.
"""
from __future__ import annotations

import argparse
import functools

import numpy as np

# ---------------------------------------------------------------------------
# BSDF constants, verbatim from HairBsdf.ush :: HairShading()
# ---------------------------------------------------------------------------
IOR = 1.55
F0 = ((1.0 - IOR) / (1.0 + IOR)) ** 2
SHIFT = 0.035
ALPHA = (-2.0 * SHIFT, SHIFT, 4.0 * SHIFT)
AREA = 0.0  # extra longitudinal variance from light size / pixel footprint
SPECULAR = 0.5
BACKLIT = 1.0
SQRT_2PI = float(np.sqrt(2.0 * np.pi))

# Azimuthal halves.  CosPhi is the cosine of the angle between L and V
# projected off the tangent, so its sign is the sign of cos(phi).
HALF_REFL = 0   # CosPhi > 0, light and eye on the same side (R + TRT)
HALF_TRANS = 1  # CosPhi < 0, light behind the fibre (TT)

# Zinke 2008 "forward scattering" is light continuing past the fibre, i.e. the
# transmission half.
HALF_FORWARD = HALF_TRANS
HALF_BACKWARD = HALF_REFL

# Quadrature.  phi uses composite Gauss-Legendre on [0, pi/2] and [pi/2, pi]
# (the integrand depends on phi only through cos(phi), so [0, pi] doubled
# covers the circle).  The longitudinal direction is integrated in
# u = sin(theta), where cos(theta) dtheta = du exactly, with a per-lobe
# Gauss-Legendre rule laid out in units of that lobe's Gaussian width.
QUAD_FULL = dict(phi_panels=6, phi_order=16, nt=40, tmax=8.0)
QUAD_FAST = dict(phi_panels=3, phi_order=10, nt=24, tmax=7.0)
QUAD_REF = dict(phi_panels=12, phi_order=24, nt=80, tmax=9.0)

GRID_FULL = dict(nrough=20, ncolor=16, ntheta=17)
GRID_FAST = dict(nrough=8, ncolor=6, ntheta=17)

# The theta-resolved fits need a denser incidence grid than the cos-weighted
# average does: with only 17 nodes a 22-term longitudinal basis very nearly
# interpolates, and the fit then oscillates by ~70% *between* the outermost
# nodes while looking excellent on them.
TFIT_NTHETA = 33

ROUGH_RANGE = (0.05, 1.0)
COLOR_RANGE = (0.02, 0.98)

# the host renderer's own dual-scattering spread inputs (HairStrandsLUT.usf), for cross-check:
# Beta_R = r^2, Beta_TT = (r/2)^2, Beta_TRT = (2r)^2 with r clamped to this band.
UE_BETA_CLAMP = (0.18, 0.6)


# ---------------------------------------------------------------------------
# BSDF helpers
# ---------------------------------------------------------------------------
def hair_fresnel(c):
    """Hair_F: Schlick with the n = 1.55 hair IOR."""
    return F0 + (1.0 - F0) * (1.0 - c) ** 5


@functools.lru_cache(maxsize=None)
def _leggauss(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def phi_nodes(a, b, panels, order):
    """Composite Gauss-Legendre nodes/weights on [a, b]."""
    x, w = _leggauss(order)
    edges = np.linspace(a, b, panels + 1)
    lo = edges[:-1][:, None]
    hi = edges[1:][:, None]
    nodes = (0.5 * (hi - lo) * x[None, :] + 0.5 * (hi + lo)).ravel()
    wts = (0.5 * (hi - lo) * w[None, :]).ravel()
    return nodes, wts


def half_phi_nodes(half, quad):
    a, b = (0.0, 0.5 * np.pi) if half == HALF_REFL else (0.5 * np.pi, np.pi)
    return phi_nodes(a, b, quad["phi_panels"], quad["phi_order"])


def half_moments(theta_v, rough, colors, phi, wphi, nt, tmax):
    """Per-lobe moments of S over one azimuthal half.

    Returns an array of shape (3 lobes, ncolor, 3) holding

        m0 = INT S cos(theta) dtheta dphi
        m1 = INT S x   cos(theta) dtheta dphi
        m2 = INT S x^2 cos(theta) dtheta dphi     with x = SinThetaL + SinThetaV

    The substitution u = sin(theta) turns cos(theta) dtheta into du, and each
    lobe's Mp = Hair_g(B, u - mu) is then exactly a Gaussian of standard
    deviation B in u.  Integrating u over [-1, 1] therefore *truncates* that
    Gaussian, which is where the model loses energy at high roughness and at
    grazing incidence.
    """
    s_v = float(np.sin(theta_v))
    c_v = float(np.cos(theta_v))
    rr = min(max(float(rough), 1.0 / 255.0), 1.0)
    r2 = rr * rr
    b_lobe = (AREA + r2, AREA + 0.5 * r2, AREA + 2.0 * r2)

    cos_phi0 = np.cos(phi)                                        # (P,)
    chp0 = np.sqrt(np.clip(0.5 + 0.5 * cos_phi0, 0.0, 1.0))
    sa = np.sin(ALPHA[0])
    ca = np.cos(ALPHA[0])
    # sqrt(1 - SinThetaV^2) == cos(theta_v) for theta_v in (-pi/2, pi/2)
    shift_r = 2.0 * sa * (ca * chp0 * c_v + sa * s_v)             # (P,)

    mus = (shift_r - s_v,
           np.full_like(cos_phi0, ALPHA[1] - s_v),
           np.full_like(cos_phi0, ALPHA[2] - s_v))
    # Widths used only to *place* the quadrature nodes.  They use cos(phi)
    # rather than the 1e-4-regularised CosPhi; the two differ by <0.3% and only
    # within 1e-2 of the poles, and node placement cannot bias the result.
    bws = (np.maximum(b_lobe[0] * np.sqrt(2.0) * chp0, 1e-12),
           np.full_like(cos_phi0, b_lobe[1]),
           np.full_like(cos_phi0, b_lobe[2]))

    xg, wg = _leggauss(nt)
    log_c = np.log(colors)[None, None, :]
    out = np.zeros((3, colors.size, 3))

    for li in range(3):
        mu = mus[li]
        bw = bws[li]
        t_lo = np.clip((-1.0 - mu) / bw, -tmax, tmax)
        t_hi = np.clip((1.0 - mu) / bw, -tmax, tmax)
        hlf = np.maximum(0.5 * (t_hi - t_lo), 0.0)
        mid = 0.5 * (t_hi + t_lo)
        t = hlf[:, None] * xg[None, :] + mid[:, None]             # (P, T)
        w_t = hlf[:, None] * wg[None, :]
        u = np.clip(mu[:, None] + bw[:, None] * t, -1.0, 1.0)
        # Gaussian pdf x du, with the 1/B and the du = B dt cancelling.
        weight = 2.0 * wphi[:, None] * w_t * np.exp(-0.5 * t * t) / SQRT_2PI

        c_l = np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
        cp = cos_phi0[:, None]
        vol = u * s_v + c_l * c_v * cp
        cos_phi = c_l * c_v * cp / np.sqrt(c_l * c_l * c_v * c_v + 1e-4)
        cos_half_phi = np.sqrt(np.clip(0.5 + 0.5 * cos_phi, 0.0, 1.0))
        theta_l = np.arcsin(np.clip(u, -1.0, 1.0))
        cos_theta_d = np.cos(0.5 * np.abs(theta_v - theta_l))
        x = u + s_v

        if li == 0:  # R
            fp = hair_fresnel(np.sqrt(np.clip(0.5 + 0.5 * vol, 0.0, 1.0)))
            val = (0.25 * cos_half_phi * fp * (SPECULAR * 2.0)
                   * (1.0 + (BACKLIT - 1.0) * np.clip(-vol, 0.0, 1.0)))
            wv = weight * val
            out[li, :, 0] = wv.sum()
            out[li, :, 1] = (wv * x).sum()
            out[li, :, 2] = (wv * x * x).sum()
            continue

        if li == 1:  # TT
            n_prime = 1.19 / cos_theta_d + 0.36 * cos_theta_d
            a = 1.0 / n_prime
            h = cos_half_phi * (1.0 + a * (0.6 - 0.8 * cos_phi))
            f = hair_fresnel(cos_theta_d * np.sqrt(np.clip(1.0 - h * h, 0.0, 1.0)))
            base = np.exp(-3.65 * cos_phi - 3.98) * (1.0 - f) ** 2 * BACKLIT
            k = 0.5 * np.sqrt(np.clip(1.0 - (h * a) ** 2, 0.0, 1.0)) / cos_theta_d
        else:        # TRT
            f = hair_fresnel(cos_theta_d * 0.5)
            base = np.exp(17.0 * cos_phi - 16.78) * (1.0 - f) ** 2 * f
            k = 0.8 / cos_theta_d

        val = base[:, :, None] * np.exp(k[:, :, None] * log_c)     # (P, T, C)
        wv = weight[:, :, None] * val
        out[li, :, 0] = wv.sum(axis=(0, 1))
        out[li, :, 1] = (wv * x[:, :, None]).sum(axis=(0, 1))
        out[li, :, 2] = (wv * (x * x)[:, :, None]).sum(axis=(0, 1))

    return out


def lobe_albedo(theta_v, rough, colors, quad=None):
    """(3 lobes, ncolor, 2 halves) of m0 for one view angle / roughness."""
    quad = quad or QUAD_FULL
    out = np.zeros((3, np.size(colors), 2))
    for half in (HALF_REFL, HALF_TRANS):
        ph, wp = half_phi_nodes(half, quad)
        out[:, :, half] = half_moments(theta_v, rough, np.atleast_1d(colors),
                                       ph, wp, quad["nt"], quad["tmax"])[:, :, 0]
    return out


def albedo(theta_v, rough, colors, quad=None):
    """A(V) for one view angle / roughness, per colour."""
    return lobe_albedo(theta_v, rough, colors, quad).sum(axis=(0, 2))


def compute_moments(thetas, roughs, colors, quad):
    """Moments over the (theta, roughness, colour, half) grid.

    Shape: (3 lobes, ntheta, nrough, ncolor, 2 halves, 3 moments).
    """
    halves = [half_phi_nodes(h, quad) for h in (HALF_REFL, HALF_TRANS)]
    out = np.zeros((3, thetas.size, roughs.size, colors.size, 2, 3))
    for ti, th in enumerate(thetas):
        for ri, rg in enumerate(roughs):
            for hi, (ph, wp) in enumerate(halves):
                out[:, ti, ri, :, hi, :] = half_moments(
                    th, rg, colors, ph, wp, quad["nt"], quad["tmax"])
    return out


# ---------------------------------------------------------------------------
# Derived dual-scattering quantities
# ---------------------------------------------------------------------------
class Scattering:
    """Dual-scattering quantities on a (theta, roughness, colour) grid."""

    def __init__(self, thetas, theta_w, roughs, colors, mom):
        self.thetas = thetas
        self.theta_w = theta_w          # cos(theta)-weighted, sums to 1
        self.roughs = roughs
        self.colors = colors
        self.mom = mom
        tot = mom.sum(axis=0)           # (ntheta, nr, nc, 2, 3)
        self.a = tot[..., 0]            # a per half
        with np.errstate(invalid="ignore", divide="ignore"):
            self.alpha = tot[..., 1] / tot[..., 0]
            self.beta2 = tot[..., 2] / tot[..., 0] - self.alpha ** 2
        self.beta2 = np.maximum(self.beta2, 0.0)
        self.albedo = self.a.sum(axis=-1)          # A(V) = a_refl + a_trans
        w = theta_w[:, None, None, None]
        self.a_bar = (self.a * w).sum(axis=0)      # (nr, nc, 2)

    def target(self, field, half):
        """Energy- and cos(theta)-weighted average over theta and colour."""
        vals = getattr(self, field)[:, :, :, half]
        wts = self.a[:, :, :, half] * self.theta_w[:, None, None]
        return (vals * wts).sum(axis=(0, 2)) / wts.sum(axis=(0, 2))


def make_grid(fast=False):
    g = GRID_FAST if fast else GRID_FULL
    roughs = np.linspace(ROUGH_RANGE[0], ROUGH_RANGE[1], g["nrough"])
    colors = np.linspace(COLOR_RANGE[0], COLOR_RANGE[1], g["ncolor"])
    # cos(theta) d(theta) = d(sin theta), so Gauss-Legendre nodes in
    # u = sin(theta) make the cos-weighted theta average a plain weighted mean.
    uv, wv = _leggauss(g["ntheta"])
    return np.arcsin(uv), wv / 2.0, roughs, colors


def solve(fast=False, quad=None):
    thetas, theta_w, roughs, colors = make_grid(fast)
    quad = quad or (QUAD_FAST if fast else QUAD_FULL)
    return Scattering(thetas, theta_w, roughs, colors,
                      compute_moments(thetas, roughs, colors, quad))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _lstsq_rel(design, y):
    """Least squares that minimises relative error against y."""
    inv = 1.0 / y
    coef, *_ = np.linalg.lstsq(design * inv[:, None], np.ones_like(y), rcond=None)
    return coef


def _fmt(v):
    """A GLSL float literal -- never an int literal, which some drivers reject."""
    s = "%.7g" % v
    return s if any(ch in s for ch in ".eE") else s + ".0"


def _wrap_sum(prefix, parts, width=84):
    """'prefix a - b + c;' as GLSL lines, wrapped and aligned under the '='."""
    if not parts:
        return [prefix + "0.0;"]
    sign, first = parts[0]
    cur = prefix + ("-" if sign == "-" else "") + first
    pad = " " * (len(prefix) - 2)
    out = []
    for sign, term in parts[1:]:
        piece = " %s %s" % (sign, term)
        if len(cur) + len(piece) > width:
            out.append(cur)
            cur = pad + piece
        else:
            cur += piece
    out.append(cur + ";")
    return out


def _horner(coef, var):
    out = _fmt(coef[-1])
    for k in coef[-2::-1]:
        out = "%s + %s*(%s)" % (_fmt(k), var, out)
    return out


def design_rc(r, c):
    """[1, r, r^2] + sqrt(c)*[1, r, r^2] + c*[1, r, r^2]  (9 terms)."""
    s = np.sqrt(c)
    base = np.stack([np.ones_like(r), r, r * r], axis=-1)
    return np.concatenate([base, s[..., None] * base, c[..., None] * base], axis=-1)


# beta^2 and alpha are even-dominated in r (the lobe widths B go as r^2), so a
# polynomial in r^2 is both the cheapest and the most accurate low-order form.
NR2 = 5  # [1, r^2, r^4, r^6, r^8]


def design_r2(r):
    return np.stack([r ** (2 * k) for k in range(NR2)], axis=-1)


class FitRC:
    """a(r, c) ~ biquadratic in (r, sqrt(c))."""

    def __init__(self, coef, values, pred):
        self.coef = coef
        rel = pred / values - 1.0
        self.max_rel = float(np.max(np.abs(rel)))
        self.rms_rel = float(np.sqrt(np.mean(rel ** 2)))

    @classmethod
    def fit(cls, roughs, colors, values):
        rr, cc = np.meshgrid(roughs, colors, indexing="ij")
        d = design_rc(rr.ravel(), cc.ravel())
        coef = _lstsq_rel(d, values.ravel())
        return cls(coef, values.ravel(), d @ coef)

    def __call__(self, r, c):
        r = np.asarray(r, dtype=float)
        c = np.asarray(c, dtype=float)
        return np.clip(design_rc(r, c) @ self.coef, 0.0, 1.0)

    def glsl(self, name):
        k = self.coef
        return ("float %s(float r, float c)\n"
                "{\n"
                "    float s = sqrt(c);\n"
                "    return clamp((%s)\n"
                "               + s*(%s)\n"
                "               + c*(%s), 0.0, 1.0);\n"
                "}" % (name, _horner(k[0:3], "r"),
                       _horner(k[3:6], "r"), _horner(k[6:9], "r")))


class FitR:
    """v(r) ~ polynomial in r^2, fitted to the theta/colour-averaged target."""

    def __init__(self, coef, roughs, target, signed):
        self.coef = coef
        self.target = target
        self.signed = signed
        self.peak = float(np.max(np.abs(target)))
        err = design_r2(roughs) @ coef - target
        self.max_abs = float(np.max(np.abs(err)))
        denom = np.abs(target) if signed else np.full_like(target, self.peak)
        self.max_rel = float(np.max(np.abs(err / denom)))
        self.rms_rel = float(np.sqrt(np.mean((err / denom) ** 2)))

    @classmethod
    def fit(cls, roughs, target):
        # Sign-definite and bounded away from zero -> a true relative error is
        # meaningful; otherwise (alpha_b crosses zero) report against the peak.
        signed = bool(np.all(target > 0) or np.all(target < 0))
        d = design_r2(roughs)
        coef = (_lstsq_rel(d, target) if signed
                else np.linalg.lstsq(d, target, rcond=None)[0])
        return cls(coef, roughs, target, signed)

    def __call__(self, r):
        return design_r2(np.asarray(r, dtype=float)) @ self.coef

    def glsl(self, name, lo, hi):
        body = _horner(self.coef, "q")
        return ("float %s(float r)\n"
                "{\n"
                "    float q = r*r;\n"
                "    return clamp(%s, %s, %s);\n"
                "}" % (name, body, _fmt(lo), _fmt(hi)))


# ---------------------------------------------------------------------------
# Theta-resolved (3-D) fits
#
# The cos-weighted theta average is cheap but costs a lot (see the report the
# script prints), because a_f/a_b are not separable in theta: at grazing
# incidence the lobe Gaussians are truncated by the |sin theta_L| <= 1 boundary,
# which turns a_f from TT-dominated (strongly coloured) into R-dominated
# (colour free).  A plain polynomial in t = sin(theta_v) cannot follow that
# edge, so the longitudinal basis also carries odd powers of
# K = cos(theta_v) = sqrt(1 - t^2), which has the right square-root behaviour
# at |t| -> 1.  Coefficients are then solved by IRLS, which targets the worst
# relative error rather than the mean.
# ---------------------------------------------------------------------------
# The form has an ALU budget: it is evaluated per light per fragment, plus once
# per dome-quadrature direction, on grooms of ~1e5 strands, so it must stay in
# the tens of FMAs -- an unbudgeted fit reaches ~8% but needs 264 coefficients,
# which is more arithmetic than the whole BSDF it feeds.  COMPACT_NTERMS is the
# ceiling, and the basis functions are picked from COMPACT_FUNCS by orthogonal
# matching pursuit rather than by hand.
COMPACT_NTERMS = 39
COMPACT_RPOW = 3            # roughness powers rho^0..rho^2 offered per function
IRLS_ROUNDS = 24
IRLS_GAMMAS = (None, 0.2, 0.3, 0.5, 0.8)   # None = plain relative least squares
COMPACT_RMS_BUDGET = 0.05   # prefer the smallest max error whose rms fits this

# (Alpha shift, B/r^2 factor) of each lobe, matching HairShading's Alpha[]
# and B[].  Used to rebuild each lobe's truncated-Gaussian mass.
EDGE_LOBES = ((ALPHA[1], 0.5), (ALPHA[0], 1.0), (ALPHA[2], 2.0))
EDGE_SPAN = 3.0     # the smoothstep spans +-EDGE_SPAN standard deviations
K_FLOOR = 0.12      # 1/cos(theta_v) is clamped here (~83 deg)


def _sigmoid(x):
    """Cheap stand-in for the normal CDF: smoothstep(-3, 3, x)."""
    u = np.clip(x / (2.0 * EDGE_SPAN) + 0.5, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def edge_term(t, r, alpha, bfac):
    """Mass of a lobe Gaussian of width bfac*r^2 left inside |sin theta_L| <= 1.

    This is the one feature of a_f/a_b that a polynomial in t simply cannot
    represent: the lobe centre sits at x = alpha - sin(theta_v), so once
    |sin theta_v| passes 1 - alpha the peak leaves the sphere and the lobe
    switches off over a window of width B = bfac*r^2, which is 0.00125 at
    roughness 0.05.  The exact mass is a difference of two normal CDFs;
    smoothstep is as accurate here as erf once the rest of the basis absorbs the
    shape difference, and it costs no transcendental.
    """
    b = np.maximum(bfac * r * r, 1e-9)
    return _sigmoid((1.0 - alpha + t) / b) + _sigmoid((1.0 + alpha - t) / b) - 1.0


# Longitudinal basis functions: name -> (numpy, GLSL expression, setup vars).
# t = sin(theta_v), k = cos(theta_v), ik = 1/max(k, K_FLOOR) (the grazing rise
# of the R lobe's Fresnel really does go as 1/cos), e0/e1/e2 = the lobe edges.
def _compact_funcs():
    f = {}

    def add(name, fn, expr, deps=()):
        f[name] = (fn, expr, deps)

    add("1", lambda t, k, ik, e: np.ones_like(t), "1.0")
    add("t", lambda t, k, ik, e: t, "t")
    add("t2", lambda t, k, ik, e: t * t, "t2", ("t2",))
    add("t3", lambda t, k, ik, e: t ** 3, "t3", ("t2", "t3"))
    add("t4", lambda t, k, ik, e: t ** 4, "t4", ("t2", "t4"))
    add("K", lambda t, k, ik, e: k, "k", ("k",))
    add("Kt", lambda t, k, ik, e: k * t, "k*t", ("k",))
    add("Kt2", lambda t, k, ik, e: k * t * t, "k*t2", ("k", "t2"))
    add("iK", lambda t, k, ik, e: ik, "ik", ("k", "ik"))
    add("iKt", lambda t, k, ik, e: ik * t, "ik*t", ("k", "ik"))
    add("iK2", lambda t, k, ik, e: ik * ik, "ik2", ("k", "ik", "ik2"))
    add("iK2t", lambda t, k, ik, e: ik * ik * t, "ik2*t", ("k", "ik", "ik2"))
    for li in range(len(EDGE_LOBES)):
        d = ("rr", "e%d" % li)
        add("e%d" % li, (lambda i: lambda t, k, ik, e: e[i])(li), "e%d" % li, d)
        add("e%dt" % li, (lambda i: lambda t, k, ik, e: e[i] * t)(li),
            "e%d*t" % li, d)
        add("e%dt2" % li, (lambda i: lambda t, k, ik, e: e[i] * t * t)(li),
            "e%d*t2" % li, d + ("t2",))
        add("e%dK" % li, (lambda i: lambda t, k, ik, e: e[i] * k)(li),
            "e%d*k" % li, d + ("k",))
    return f


COMPACT_FUNCS = _compact_funcs()

# Setup lines, in dependency order; only the ones a fit actually needs are emitted.
SETUP_GLSL = (
    ("t2", "float t2 = t*t;"),
    ("t3", "float t3 = t2*t;"),
    ("t4", "float t4 = t2*t2;"),
    ("k", "float k = sqrt(max(1.0 - t*t, 0.0));"),
    ("ik", "float ik = 1.0/max(k, %s);" % _fmt(K_FLOOR)),
    ("ik2", "float ik2 = ik*ik;"),
    ("rr", "float rr = max(r*r, 1e-9);"),
) + tuple(("e%d" % li, "float e%d = HairLobeEdge(t, %s*rr, %s);"
           % (li, _fmt(bf), _fmt(al)))
          for li, (al, bf) in enumerate(EDGE_LOBES))


def compact_basis(names, t, r):
    """Evaluate the named longitudinal functions."""
    k = np.sqrt(np.clip(1.0 - t * t, 0.0, 1.0))
    ik = 1.0 / np.maximum(k, K_FLOOR)
    e = [edge_term(t, r, al, bf) for al, bf in EDGE_LOBES]
    return [COMPACT_FUNCS[n][0](t, k, ik, e) for n in names]


HAIR_EDGE_GLSL = """float HairLobeEdge(float t, float b, float alpha)
{
    // Mass of a lobe Gaussian of width b left inside |sin thetaL| <= 1, once
    // its centre x = alpha - sinThetaV starts to leave the sphere at grazing
    // incidence.  smoothstep(-3, 3, .) stands in for the normal CDF.
    return smoothstep(-3.0, 3.0, (1.0 - alpha + t)/b)
         + smoothstep(-3.0, 3.0, (1.0 + alpha - t)/b) - 1.0;
}"""


# t and K already live on [-1, 1]; roughness is remapped to the same range so
# the fitted coefficients stay O(1) instead of O(1000).  Without this the
# monomials are so collinear that the shader would evaluate a ~0.5 result as a
# cancelling sum of terms near 1e3, losing fp32 precision for nothing.
R_MID = 0.5 * (ROUGH_RANGE[0] + ROUGH_RANGE[1])
R_HALF = 0.5 * (ROUGH_RANGE[1] - ROUGH_RANGE[0])


def compact_design(terms, t, r, c):
    """terms = [(colour index 0..2, function name, roughness power), ...]."""
    names = sorted({n for _, n, _ in terms})
    tf = dict(zip(names, compact_basis(names, t, r)))
    rho = (r - R_MID) / R_HALF
    cf = (np.ones_like(c), np.sqrt(c), c)
    return np.stack([(cf[ci] * tf[n] * rho ** p).ravel() for ci, n, p in terms], -1)


def _irls(design, y, gamma=None, rounds=IRLS_ROUNDS):
    """Relative least squares, optionally reweighted towards the worst case.

    gamma=None is plain relative least squares (best RMS); a positive gamma
    trades RMS for a lower maximum.  Both are wanted: at this basis size a hard
    gamma collapses the RMS badly, so the caller sweeps and picks.
    """
    w_design = design / y[:, None]
    target = np.ones_like(y)
    if gamma is None:
        return np.linalg.lstsq(w_design, target, rcond=None)[0]
    weights = np.ones_like(y)
    best = (np.inf, None)
    for _ in range(rounds):
        sw = np.sqrt(weights)
        coef, *_ = np.linalg.lstsq(w_design * sw[:, None], target * sw, rcond=None)
        resid = np.abs(w_design @ coef - target)
        worst = resid.max()
        if worst < best[0]:
            best = (worst, coef)
        weights = weights * (0.05 + resid / worst) ** gamma
        weights /= weights.mean()
    return best[1]


def _select_terms(t, r, c, values, nterms):
    """Orthogonal matching pursuit over COMPACT_FUNCS x colour x roughness power."""
    pool = [(ci, n, p) for ci in range(3) for n in COMPACT_FUNCS
            for p in range(COMPACT_RPOW)]
    y = values.ravel()
    w = compact_design(pool, t, r, c) / y[:, None]
    chosen, basis, resid = [], np.zeros((y.size, 0)), np.ones_like(y)
    for _ in range(min(nterms, len(pool))):
        ortho = w - basis @ (basis.T @ w)
        norm = np.einsum("ij,ij->j", ortho, ortho)
        norm[chosen] = 0.0
        gain = np.where(norm > 1e-10, (ortho.T @ resid) ** 2 / np.maximum(norm, 1e-30), 0.0)
        j = int(np.argmax(gain))
        chosen.append(j)
        q = ortho[:, j] / np.sqrt(norm[j])
        basis = np.concatenate([basis, q[:, None]], 1)
        resid = resid - q * (q @ resid)
    return [pool[j] for j in chosen]


class FitCompactT:
    """a(sin theta_v, r, c) = p0(r,t) + sqrt(c)*p1(r,t) + c*p2(r,t).

    The three p_w depend only on (roughness, sinThetaV), so the caller evaluates
    them once and reuses them for all three colour channels.
    """

    def __init__(self, terms, coef, gamma):
        self.terms = terms
        self.coef = coef
        self.gamma = gamma

    @classmethod
    def fit(cls, thetas, roughs, colors, values, nterms=COMPACT_NTERMS):
        tt, rr, cc = np.meshgrid(np.sin(thetas), roughs, colors, indexing="ij")
        return cls.fit_fixed(thetas, roughs, colors, values,
                             _select_terms(tt, rr, cc, values, nterms))

    @classmethod
    def fit_fixed(cls, thetas, roughs, colors, values, terms):
        """Same solve, but on a basis the caller names (for the comparison table)."""
        tt, rr, cc = np.meshgrid(np.sin(thetas), roughs, colors, indexing="ij")
        design = compact_design(terms, tt, rr, cc)
        best = None
        for gamma in IRLS_GAMMAS:
            coef = _irls(design, values.ravel(), gamma)
            rel = (design @ coef).reshape(values.shape) / values - 1.0
            m = np.abs(np.degrees(thetas)) <= 75.0
            worst = float(np.max(np.abs(rel[m])))
            rms = float(np.sqrt(np.mean(rel[m] ** 2)))
            key = (rms > COMPACT_RMS_BUDGET, worst)
            if best is None or key < best[0]:
                best = (key, coef, gamma)
        return cls(terms, best[1], best[2])

    def errors(self, thetas, roughs, colors, values):
        tt, rr, cc = np.meshgrid(np.sin(thetas), roughs, colors, indexing="ij")
        rel = self(tt, rr, cc) / values - 1.0
        out = dict(max=float(np.max(np.abs(rel))),
                   rms=float(np.sqrt(np.mean(rel ** 2))))
        for lim in (75, 60):
            m = np.abs(np.degrees(thetas)) <= lim
            out["max%d" % lim] = float(np.max(np.abs(rel[m])))
            out["rms%d" % lim] = float(np.sqrt(np.mean(rel[m] ** 2)))
        return out

    def __call__(self, t, r, c, coef=None):
        t, r, c = np.broadcast_arrays(np.asarray(t, dtype=float),
                                      np.asarray(r, dtype=float),
                                      np.asarray(c, dtype=float))
        flat = compact_design(self.terms, t, r, c) @ (
            self.coef if coef is None else coef)
        return np.clip(flat.reshape(t.shape), 0.0, 1.0)

    def rounded(self):
        """The coefficients exactly as the emitted GLSL text spells them."""
        return np.array([float(_fmt(v)) for v in self.coef])

    def _groups(self):
        """{colour index: [(function name, roughness power, coefficient), ...]}."""
        out = {0: [], 1: [], 2: []}
        for (ci, n, p), v in zip(self.terms, self.coef):
            out[ci].append((n, p, v))
        return out

    def glsl(self, stem):
        groups = self._groups()
        needed = []
        for _ci, n, _p in self.terms:
            for dep in COMPACT_FUNCS[n][2]:
                if dep not in needed:
                    needed.append(dep)
        lines = ["vec3 %sRT(float r, float sinThetaV)" % stem, "{",
                 "    float t = sinThetaV;",
                 "    float q = (r - %s)*%s;" % (_fmt(R_MID), _fmt(1.0 / R_HALF))]
        lines += ["    " + line for key, line in SETUP_GLSL if key in needed]
        for ci in (0, 1, 2):
            parts = []
            for n, p, v in sorted(groups[ci], key=lambda x: (x[0], x[1])):
                expr = COMPACT_FUNCS[n][1]
                q = {0: "", 1: "*q", 2: "*q*q"}[p]
                tail = ("%s*%s%s" % (_fmt(abs(v)), expr, q) if expr != "1.0"
                        else "%s%s" % (_fmt(abs(v)), q))
                parts.append(("-" if v < 0 else "+", tail))
            lines += _wrap_sum("    float p%d = " % ci, parts)
        lines += ["    return vec3(p0, p1, p2);", "}",
                  "float %s(vec3 p, float c)" % stem, "{",
                  "    return clamp(p.x + sqrt(c)*p.y + c*p.z, 0.0, 1.0);", "}",
                  "float %s(float r, float c, float sinThetaV)" % stem, "{",
                  "    return %s(%sRT(r, sinThetaV), c);" % (stem, stem), "}"]
        return "\n".join(lines)


FIT_SPECS = (
    # key,       field,   half,          glsl name,             clamp
    ("a_f",     "a",     HALF_FORWARD,  "HairAvgForward",      (0.0, 1.0)),
    ("a_b",     "a",     HALF_BACKWARD, "HairAvgBackward",     (0.0, 1.0)),
    ("beta2_f", "beta2", HALF_FORWARD,  "HairForwardVariance", (0.0, 4.0)),
    ("beta2_b", "beta2", HALF_BACKWARD, "HairBackwardVariance", (0.0, 4.0)),
    ("alpha_b", "alpha", HALF_BACKWARD, "HairBackwardShift",   (-1.0, 1.0)),
    ("alpha_f", "alpha", HALF_FORWARD,  "HairForwardShift",    (-1.0, 1.0)),
)


def fit_all(sc):
    """Fit the theta-averaged quantities.  Returns {key: Fit*} plus diagnostics."""
    fits = {}
    for key, field, half, _name, _clamp in FIT_SPECS:
        if field == "a":
            fits[key] = FitRC.fit(sc.roughs, sc.colors, sc.a_bar[:, :, half])
        else:
            fits[key] = FitR.fit(sc.roughs, sc.target(field, half))
    fits["diag"] = theta_diagnostics(sc, fits)
    return fits


# Deliberately off every fit node, so the reported error cannot be flattered by
# a form that only behaves where it was constrained.
OFFGRID_DEG = (-81.0, -78.0, -74.0, -66.0, -51.0, -33.0, 7.0, 29.0, 52.0,
               68.0, 76.0, 79.5)
OFFGRID_ROUGH = (0.08, 0.27, 0.44, 0.71, 0.93)
OFFGRID_COLOR = (0.05, 0.31, 0.55, 0.77, 0.91)


def offgrid_errors(fit, t_off, r_off, c_off, degs, ref, coef=None):
    rel = fit(t_off, r_off, c_off, coef) / ref - 1.0
    out = dict(max=float(np.max(np.abs(rel))),
               rms=float(np.sqrt(np.mean(rel ** 2))))
    for lim in (75, 60):
        m = np.abs(degs) <= lim
        out["max%d" % lim] = float(np.max(np.abs(rel[m])))
        out["rms%d" % lim] = float(np.sqrt(np.mean(rel[m] ** 2)))
    return out


def fit_theta(fast=False, quad=None):
    """Fit the theta-resolved forms, on their own denser incidence grid."""
    quad = quad or (QUAD_FAST if fast else QUAD_FULL)
    g = GRID_FAST if fast else GRID_FULL
    roughs = np.linspace(*ROUGH_RANGE, num=g["nrough"])
    colors = np.linspace(*COLOR_RANGE, num=g["ncolor"])
    uv, wv = _leggauss(TFIT_NTHETA)
    thetas = np.arcsin(uv)
    sc = Scattering(thetas, wv / 2.0, roughs, colors,
                    compute_moments(thetas, roughs, colors, quad))

    degs = np.array(OFFGRID_DEG)
    r_off = np.array(OFFGRID_ROUGH)
    c_off = np.array(OFFGRID_COLOR)
    ref = np.zeros((2, degs.size, r_off.size, c_off.size))
    for i, th in enumerate(np.radians(degs)):
        for j, rg in enumerate(r_off):
            ref[:, i, j, :] = lobe_albedo(th, rg, c_off, quad).sum(axis=0).T
    t_off, rr_off, cc_off = np.meshgrid(np.sin(np.radians(degs)), r_off, c_off,
                                        indexing="ij")

    out = {"grid": sc, "ref": ref, "off_nodes": (t_off, rr_off, cc_off, degs)}
    for key, half in (("a_f", HALF_FORWARD), ("a_b", HALF_BACKWARD)):
        fit = FitCompactT.fit(sc.thetas, sc.roughs, sc.colors, sc.a[:, :, :, half])
        fit.err = fit.errors(sc.thetas, sc.roughs, sc.colors, sc.a[:, :, :, half])
        fit.off = offgrid_errors(fit, t_off, rr_off, cc_off, degs, ref[half])
        # Round-trip: the numbers reported next to the snippet must be the
        # numbers the *emitted text* produces, not the full-precision fit.
        fit.off_rounded = offgrid_errors(fit, t_off, rr_off, cc_off, degs,
                                         ref[half], fit.rounded())
        out[key + "_t"] = fit
    # What the pair sums to, against the reference model's own headroom: the
    # true A(V) reaches 1.15 for white hair at grazing, so this is not bounded
    # by 1 (see check_hair_energy.py).
    tt = np.sin(np.linspace(-0.5 * np.pi, 0.5 * np.pi, 91))
    rr = np.linspace(1.0 / 255.0, 1.0, 33)
    ccg = np.linspace(0.0, 1.0, 33)
    t3, r3, c3 = np.meshgrid(tt, rr, ccg, indexing="ij")
    out["albedo_max"] = float(np.max(out["a_f_t"](t3, r3, c3)
                                     + out["a_b_t"](t3, r3, c3)))
    return out


def ue_beta_comparison(sc):
    """the host renderer's blended-Beta shortcut vs. the measured weighted variances.

    the host renderer's dual scattering does not integrate the variance: it takes per-lobe
    constants Beta_R = r^2, Beta_TT = (r/2)^2, Beta_TRT = (2r)^2 with r clamped
    to UE_BETA_CLAMP, and blends them with the same per-lobe a_f / a_b weights.
    Beta_R matches HairShading's own B[0] = r^2 exactly, so the Betas read as
    longitudinal *standard deviations*; Beta_TT and Beta_TRT do not match
    B[1] = r^2/2 and B[2] = 2 r^2, so both readings are reported.
    """
    rc = np.clip(sc.roughs, *UE_BETA_CLAMP)
    ue = np.stack([rc ** 2, (0.5 * rc) ** 2, (2.0 * rc) ** 2], 0)     # (3, nr)
    exact = np.stack([sc.roughs ** 2, 0.5 * sc.roughs ** 2,
                      2.0 * sc.roughs ** 2], 0)                       # HairShading B
    w_theta = sc.theta_w[:, None, None]
    out = {}
    for key, half in (("beta2_f", HALF_FORWARD), ("beta2_b", HALF_BACKWARD)):
        a_lobe = sc.mom[:, :, :, :, half, 0]               # (lobe, theta, r, c)
        tot = a_lobe.sum(axis=0)
        models = {
            "UE Beta as variance": np.einsum("lr,ltrc->trc", ue, a_lobe) / tot,
            "UE Beta as std dev": (np.einsum("lr,ltrc->trc", ue, a_lobe) / tot) ** 2,
            "blend of HairShading B^2":
                np.einsum("lr,ltrc->trc", exact ** 2, a_lobe) / tot,
        }
        wts = sc.a[:, :, :, half] * w_theta

        def avg(x):
            return (x * wts).sum(axis=(0, 2)) / wts.sum(axis=(0, 2))

        meas = sc.beta2[:, :, :, half]
        rows = {}
        for name, model in models.items():
            ratio = avg(meas) / avg(model)
            point = meas / np.maximum(model, 1e-12)
            rows[name] = dict(curve=avg(model), ratio=ratio,
                              lo=float(point.min()), hi=float(point.max()),
                              within2x=float(np.mean((point > 0.5) & (point < 2.0))))
        out[key] = dict(measured=avg(meas), models=rows)
    return out


def theta_diagnostics(sc, fits):
    """What the theta-averaged (theta-free) closed forms cost."""
    deg = np.degrees(sc.thetas)
    mid = np.abs(deg) <= 60.0
    out = {}
    for key, half in (("a_f", HALF_FORWARD), ("a_b", HALF_BACKWARD)):
        dev = sc.a[:, :, :, half] / sc.a_bar[None, :, :, half] - 1.0
        out[key] = dict(
            max=float(np.max(np.abs(dev))),
            rms=float(np.sqrt(np.mean(dev ** 2))),
            max60=float(np.max(np.abs(dev[mid]))),
            rms60=float(np.sqrt(np.mean(dev[mid] ** 2))),
        )
        # Could a cheap separable tilt g(sin theta_v) recover it?  Only if the
        # ratio a(theta)/a_bar is (nearly) independent of roughness and colour.
        ratio = sc.a[:, :, :, half] / sc.a_bar[None, :, :, half]
        g = ratio.mean(axis=(1, 2))
        g = g / (g * sc.theta_w).sum()
        res = ratio / g[:, None, None] - 1.0
        out[key]["tilt_max"] = float(np.max(np.abs(res)))
        out[key]["tilt_rms"] = float(np.sqrt(np.mean(res ** 2)))
        out[key]["colour_spread"] = float(
            np.max(ratio.max(axis=2) / ratio.min(axis=2) - 1.0))
    for key, field, half in (("beta2_f", "beta2", HALF_FORWARD),
                             ("beta2_b", "beta2", HALF_BACKWARD),
                             ("alpha_f", "alpha", HALF_FORWARD),
                             ("alpha_b", "alpha", HALF_BACKWARD)):
        vals = getattr(sc, field)[:, :, :, half]
        spread = vals - fits[key](sc.roughs)[None, :, None]
        out[key] = dict(
            max=float(np.max(np.abs(spread))),
            rms=float(np.sqrt(np.mean(spread ** 2))),
            max60=float(np.max(np.abs(spread[mid]))),
            rms60=float(np.sqrt(np.mean(spread[mid] ** 2))),
        )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def convergence_check(fast=False):
    """Re-integrate probes at ~2x resolution; report the worst relative drift."""
    probes = [(0.0, 0.10, 0.5), (0.35, 0.35, 0.9), (0.9, 0.70, 0.2),
              (1.2, 1.00, 0.98), (-1.2, 0.05, 0.98), (-0.7, 0.05, 0.05)]
    base = QUAD_FAST if fast else QUAD_FULL
    worst, rows = 0.0, []
    for th, rg, cc in probes:
        cols = np.array([cc])
        v0 = float(albedo(th, rg, cols, base)[0])
        v1 = float(albedo(th, rg, cols, QUAD_REF)[0])
        rel = abs(v0 - v1) / v1
        worst = max(worst, rel)
        rows.append((th, rg, cc, v0, v1, rel))
    return worst, rows


def theta_report(tf, fits):
    """The (a)-(d) comparison, then what the shipped compact form delivers."""
    sc = tf["grid"]
    t_off, r_off, c_off, degs = tf["off_nodes"]
    print("\n[theta-resolved forms] within an ALU budget of %d coefficients"
          % COMPACT_NTERMS)
    print("   %d Gauss nodes in sin(theta_i).  t = sinThetaV, K = cos(theta_v),"
          % TFIT_NTHETA)
    print("   iK = 1/max(K, %g), e0/e1/e2 = the TT/R/TRT lobe edge terms."
          % K_FLOOR)
    print("   form                              coef |  ON GRID <=60      <=75   "
          "|  OFF GRID <=60      <=75")

    def stats(rel, mask_deg):
        out = {}
        for lim in (75, 60):
            m = np.abs(mask_deg) <= lim
            out["max%d" % lim] = float(np.max(np.abs(rel[m])))
            out["rms%d" % lim] = float(np.sqrt(np.mean(rel[m] ** 2)))
        return out

    def show(label, n, on, off):
        print("   %-33s %4d | %5.1f%% %5.2f%% %5.1f%% %5.2f%% | %5.1f%% %5.2f%% %5.1f%% %5.2f%%"
              % (label, n, 100 * on["max60"], 100 * on["rms60"],
                 100 * on["max75"], 100 * on["rms75"],
                 100 * off["max60"], 100 * off["rms60"],
                 100 * off["max75"], 100 * off["rms75"]))

    on_deg = np.degrees(sc.thetas)
    for key, half in (("a_f", HALF_FORWARD), ("a_b", HALF_BACKWARD)):
        print("  -- %s" % key)
        # (a) the theta-free constant, evaluated against every incidence angle
        base = fits[key]
        flat = base(*np.meshgrid(sc.roughs, sc.colors, indexing="ij"))[None]
        show("(a) theta-free baseline", base.coef.size,
             stats(flat / sc.a[:, :, :, half] - 1.0, on_deg),
             stats(base(r_off, c_off) / tf["ref"][half] - 1.0, degs))
        # (b), (c) the tensor forms
        for tag, names in (("(b) [1,s,c]x[1,r,r2]x[1,t2,t4]", ("1", "t2", "t4")),
                           ("(c)   ...         x [1,t,t2,t3]",
                            ("1", "t", "t2", "t3"))):
            terms = [(ci, n, p) for ci in range(3) for n in names
                     for p in range(COMPACT_RPOW)]
            fit = FitCompactT.fit_fixed(sc.thetas, sc.roughs, sc.colors,
                                        sc.a[:, :, :, half], terms)
            show(tag, fit.coef.size,
                 fit.errors(sc.thetas, sc.roughs, sc.colors, sc.a[:, :, :, half]),
                 offgrid_errors(fit, t_off, r_off, c_off, degs, tf["ref"][half]))
        # (d) what this script ships
        fit = tf[key + "_t"]
        show("(d) matching pursuit -- SHIPPED", fit.coef.size, fit.err, fit.off)
        print("       uses: %s" % ", ".join(sorted({n for _, n, _ in fit.terms})))
    print("   (max / rms relative error.  'off grid' samples strictly between the")
    print("    fitted nodes -- on a sparse grid a rich basis nearly interpolates and")
    print("    the on-grid column flatters it, so trust the off-grid column.)")
    print("   fitted a_f + a_b peaks at %.3f over the whole (theta, r, c) domain"
          % tf["albedo_max"])
    print("   emitted-precision round trip: re-evaluating the rounded constants")
    for label, key in (("a_f", "a_f_t"), ("a_b", "a_b_t")):
        f = tf[key]
        print("      %s off-grid <=75  full %.2f%% -> rounded %.2f%% (max)"
              % (label, 100 * f.off["max75"], 100 * f.off_rounded["max75"]))


def raw_table(quad=None):
    """Raw integrated a_f / a_b at the shader's default roughness."""
    rough = 0.35
    cols = np.array([0.02, 0.1, 0.3, 0.6, 0.9])
    print("\n[raw integrated values] roughness %.2f, no fit involved" % rough)
    print("   these are the integrals themselves, for checking the shader's")
    print("   transmittance against numbers you can read")
    print("   colour |" + "".join("   a_f     a_b   @ %2.0f deg" % d
                                  for d in (0, 30, 60)))
    per_angle = [lobe_albedo(np.radians(d), rough, cols, quad)
                 for d in (0.0, 30.0, 60.0)]
    for ci, cval in enumerate(cols):
        line = "   %5.2f  |" % cval
        for m in per_angle:
            line += "  %6.4f  %6.4f        " % (
                m[:, ci, HALF_FORWARD].sum(), m[:, ci, HALF_BACKWARD].sum())
        print(line)


def glsl_snippet_theta(tf):
    """The theta-resolved variants, which keep the incidence dependence."""
    f, b = tf["a_f_t"], tf["a_b_t"]
    out = [
        "// --------------------------------------------------------------------------",
        "// Theta-RESOLVED dual scattering for the UE hair BSDF -- the same integrals",
        "// as HairAvgForward/HairAvgBackward above, but keeping the incidence angle",
        "// instead of averaging it away.  sinThetaV = dot(strandTangent, V).",
        "// Same axes as UE's LUT (sin view angle, roughness, sqrt baseColor), except",
        "// that UE indexes |sin| while this keeps the sign: the Alpha cuticle shifts",
        "// make a_f markedly asymmetric in theta (a_f at -72 deg differs from +72 deg",
        "// by up to 45%).  Generated by usdGenShaders/test/fit_hair_scattering.py.",
        "//",
        "// Relative error of THESE constants at the precision printed below:",
        "//        coefficients   |theta|<=60      |theta|<=75",
    ]
    for label, fit in (("a_f", f), ("a_b", b)):
        out.append("// %s   on grid   %4d    %5.1f%% %5.2f%%    %5.1f%% %5.2f%%"
                   % (label, fit.coef.size, 100 * fit.err["max60"],
                      100 * fit.err["rms60"], 100 * fit.err["max75"],
                      100 * fit.err["rms75"]))
        out.append("// %s   off grid          %5.1f%% %5.2f%%    %5.1f%% %5.2f%%"
                   % (label, 100 * fit.off_rounded["max60"],
                      100 * fit.off_rounded["rms60"],
                      100 * fit.off_rounded["max75"],
                      100 * fit.off_rounded["rms75"]))
    out += [
        "// 'off grid' is sampled strictly between the fitted nodes and re-uses the",
        "// rounded constants below, so it is what this text actually delivers.",
        "//",
        "// Cost: %d + %d FMAs, plus a setup (t powers, k, 1/k, three HairLobeEdge)"
        % (f.coef.size, b.coef.size),
        "// that both RT functions share -- inline them in one scope and the compiler",
        "// folds it.  HairAvgForwardRT returns (p0, p1, p2), which depend only on",
        "// (roughness, sinThetaV), so evaluate it ONCE per fragment and feed it to",
        "// the per-channel HairAvgForward(vec3, float).",
        "// --------------------------------------------------------------------------",
        HAIR_EDGE_GLSL,
        f.glsl("HairAvgForward"),
        b.glsl("HairAvgBackward"),
        "// --------------------------------------------------------------------------",
    ]
    return "\n".join(out)


def glsl_snippet(fits):
    a_f, a_b = fits["a_f"], fits["a_b"]
    out = [
        "// --------------------------------------------------------------------------",
        "// Dual-scattering fits for the UE hair BSDF (HairBsdf.ush; Brian Karis,",
        "// \"Physically Based Hair Shading in Unreal\", SIGGRAPH 2016 PBS course).",
        "// Generated by usdGenShaders/test/fit_hair_scattering.py -- re-run that",
        "// script instead of editing these constants by hand, and keep",
        "// usdGenShaders/resources/shaders/usdGenHairStrands.glslfx in sync with it.",
        "//",
        "// forward  = transmission half (CosPhi < 0, light behind the fibre, TT)",
        "// backward = reflection   half (CosPhi > 0, light and eye same side, R+TRT)",
        "// r = clamped roughness in [1/255, 1]; c = one RGB channel of baseColor.",
        "// Longitudinal quantities are in units of x = SinThetaL + SinThetaV.",
        "// Constants are averaged over incidence theta_i (cos-weighted); see the",
        "// script output for what that costs.",
        "//",
        "// a_f(r,c): max rel err %.2f%%, rms %.2f%%" % (100 * a_f.max_rel,
                                                        100 * a_f.rms_rel),
        "// a_b(r,c): max rel err %.2f%%, rms %.2f%%" % (100 * a_b.max_rel,
                                                        100 * a_b.rms_rel),
        "// beta_f^2: max rel err %.2f%%   beta_b^2: max rel err %.2f%%"
        % (100 * fits["beta2_f"].max_rel, 100 * fits["beta2_b"].max_rel),
        "// alpha_b : max abs err %.5f (%.1f%% of peak)   alpha_f: max rel err %.2f%%"
        % (fits["alpha_b"].max_abs, 100 * fits["alpha_b"].max_rel,
           100 * fits["alpha_f"].max_rel),
        "// --------------------------------------------------------------------------",
    ]
    for key, _field, _half, name, clamp in FIT_SPECS:
        fit = fits[key]
        out.append(fit.glsl(name) if isinstance(fit, FitRC)
                   else fit.glsl(name, *clamp))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true",
                    help="coarse grid and quadrature (finishes in a few seconds)")
    ap.add_argument("--no-convergence", action="store_true")
    args = ap.parse_args(argv)

    print("UE hair BSDF (HairBsdf.ush / Karis 2016) -- dual-scattering fit")
    print("  Specular=%.2f  Backlit=%.2f  Area=%.2f  n=%.2f  Shift=%.3f"
          % (SPECULAR, BACKLIT, AREA, IOR, SHIFT))
    print("  forward  half = CosPhi < 0 (light behind the fibre, TT dominated)")
    print("  backward half = CosPhi > 0 (light and eye same side, R + TRT)")

    if not args.no_convergence:
        worst, rows = convergence_check(args.fast)
        print("\n[convergence] grid quadrature vs. a ~2x reference rule")
        print("   theta_v  rough  colour        A(grid)         A(ref)   rel.diff")
        for th, rg, cc, v0, v1, rel in rows:
            print("   %7.3f  %5.2f  %5.2f  %13.6f  %13.6f  %9.2e"
                  % (th, rg, cc, v0, v1, rel))
        print("   worst relative drift %.2e -- %s the 1e-3 target"
              % (worst, "meets" if worst < 1e-3 else "MISSES"))

    sc = solve(args.fast)
    fits = fit_all(sc)
    diag = fits["diag"]
    ue_cmp = ue_beta_comparison(sc)

    print("\n[grid] roughness %g..%g (%d), colour %g..%g (%d), %d Gauss nodes in "
          "sin(theta_i)" % (sc.roughs[0], sc.roughs[-1], sc.roughs.size,
                            sc.colors[0], sc.colors[-1], sc.colors.size,
                            sc.thetas.size))

    step = max(1, sc.colors.size // 6)
    cols = list(range(0, sc.colors.size, step))
    print("\n[albedo] A = a_f + a_b, cos-weighted over theta_i")
    print("   rough |" + "".join("  c=%.2f" % sc.colors[ci] for ci in cols))
    for ri in range(0, sc.roughs.size, max(1, sc.roughs.size // 8)):
        tot = sc.a_bar[ri].sum(axis=-1)
        print("   %5.2f |" % sc.roughs[ri]
              + "".join(" %6.3f" % tot[ci] for ci in cols))

    print("\n[fit quality] relative error of each closed form over the grid")
    print("   quantity   form                             max      rms     max abs")
    for key, label in (("a_f", "a_f     "), ("a_b", "a_b     ")):
        fit = fits[key]
        print("   %s   [1,sqrt c,c] x [1,r,r^2] (9)  %6.2f%%  %6.2f%%"
              % (label, 100 * fit.max_rel, 100 * fit.rms_rel))
    for key, label in (("beta2_f", "beta_f^2"), ("beta2_b", "beta_b^2"),
                       ("alpha_f", "alpha_f "), ("alpha_b", "alpha_b ")):
        fit = fits[key]
        note = "" if fit.signed else "   <- vs peak |target|, which crosses 0"
        print("   %s   poly in r^2 (%d coef)         %6.2f%%  %6.2f%%   %9.6f%s"
              % (label, NR2, 100 * fit.max_rel, 100 * fit.rms_rel,
                 fit.max_abs, note))

    print("\n[cost of the theta_i average] a(theta_i) vs the fitted constant")
    for key in ("a_f", "a_b"):
        d = diag[key]
        print("   %s : max %5.1f%%  rms %5.1f%%   |   |theta_i|<=60 deg: "
              "max %5.1f%%  rms %5.1f%%"
              % (key, 100 * d["max"], 100 * d["rms"],
                 100 * d["max60"], 100 * d["rms60"]))
        print("        a separable tilt g(sin theta_v) would leave max %.0f%% "
              "rms %.0f%%; the ratio a(theta)/a_bar spans %.0f%% across colour"
              % (100 * d["tilt_max"], 100 * d["tilt_rms"],
                 100 * d["colour_spread"]))
    print("   => the theta_i dependence is NOT separable from (r, c): restoring it")
    print("      needs a genuine 3-D fit or a LUT, which is what this fit avoids.")
    print("      The error is concentrated beyond +-60 deg, where the lobe")
    print("      Gaussians are truncated by the |sin theta_L| <= 1 boundary.")
    print("   longitudinal spread about the fitted constants (absolute, x units):")
    for key in ("beta2_f", "beta2_b", "alpha_f", "alpha_b"):
        d = diag[key]
        print("   %-8s max %.4f  rms %.4f   |   |theta_i|<=60 deg: max %.4f  "
              "rms %.4f" % (key, d["max"], d["rms"], d["max60"], d["rms60"]))

    tf = fit_theta(args.fast)
    theta_report(tf, fits)
    raw_table(QUAD_FAST if args.fast else QUAD_FULL)

    print("\n[UE Beta shortcut] measured beta^2 vs UE's blended per-lobe constants")
    for key, label in (("beta2_f", "beta_f^2"), ("beta2_b", "beta_b^2")):
        cmp_ = ue_cmp[key]
        print("   %s, energy- and cos-weighted over theta and colour" % label)
        show = list(range(0, sc.roughs.size, max(1, sc.roughs.size // 7)))
        print("      roughness         " + "".join("%8.2f" % sc.roughs[i] for i in show))
        print("      measured          "
              + "".join("%8.4f" % cmp_["measured"][i] for i in show))
        for name, row in cmp_["models"].items():
            print("      %-18s" % name
                  + "".join("%8.4f" % row["curve"][i] for i in show))
            print("        ratio measured/it"
                  + "".join("%8.2f" % row["ratio"][i] for i in show)
                  + "   | per point %.3f..%.1f, within 2x at %.0f%% of the grid"
                  % (row["lo"], row["hi"], 100 * row["within2x"]))

    print("\n" + glsl_snippet(fits))
    print("\n" + glsl_snippet_theta(tf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
