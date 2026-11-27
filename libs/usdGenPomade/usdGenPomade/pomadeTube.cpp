// usdGenPomade — P3 CPU twins (K4/K5/K8–K11). Same operation order as the
// CUDA kernels in pomadeKernels.cu; TN-6 parity is asserted in T0.
#include "usdGenPomade/pomadeTube.h"
#include "usdGenPomade/pomadeHierarchy.h"
#include "usdGenPomade/pomadeRegion.h"
#include "usdGenPomade/pomadeScalp.h"
#include "usdGenSurface/concaveMaterialRemap.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace usdGenPomade {

namespace {

bool _ProperRotation(float const *r)
{
    if (!r) {
        return false;
    }
    for (int i = 0; i < 9; ++i) {
        if (!std::isfinite(r[i])) {
            return false;
        }
    }
    // Rows must form a proper orthonormal basis.  The tolerance allows the
    // single-precision support-plane fit but rejects scale, shear and mirror.
    for (int row = 0; row < 3; ++row) {
        float length2 = 0.0f;
        for (int col = 0; col < 3; ++col) {
            length2 += r[row * 3 + col] * r[row * 3 + col];
        }
        if (std::fabs(length2 - 1.0f) > 2e-4f) {
            return false;
        }
        for (int other = row + 1; other < 3; ++other) {
            float dot = 0.0f;
            for (int col = 0; col < 3; ++col) {
                dot += r[row * 3 + col] * r[other * 3 + col];
            }
            if (std::fabs(dot) > 2e-4f) {
                return false;
            }
        }
    }
    float const determinant =
        r[0] * (r[4] * r[8] - r[5] * r[7]) -
        r[1] * (r[3] * r[8] - r[5] * r[6]) +
        r[2] * (r[3] * r[7] - r[4] * r[6]);
    return std::fabs(determinant - 1.0f) <= 3e-4f;
}

bool _RigidTransformValid(PomadeRigidTransform const &transform)
{
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(transform.translation[i])) {
            return false;
        }
    }
    return _ProperRotation(transform.rotation);
}

bool _ReferenceIsIdentity(std::array<float, 9> const &q)
{
    static float const identity[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 9; ++i) {
        if (q[size_t(i)] != identity[i]) {
            return false;
        }
    }
    return true;
}

void _RotateVector(float const *r, float const in[3], float out[3])
{
    float const x = in[0], y = in[1], z = in[2];
    out[0] = r[0] * x + r[1] * y + r[2] * z;
    out[1] = r[3] * x + r[4] * y + r[5] * z;
    out[2] = r[6] * x + r[7] * y + r[8] * z;
}

void _TransformVector(PomadeRigidTransform const &transform,
                      float const in[3], float out[3])
{
    _RotateVector(transform.rotation, in, out);
}

void _TransformPoint(PomadeRigidTransform const &transform,
                     float const in[3], float out[3])
{
    _TransformVector(transform, in, out);
    out[0] += transform.translation[0];
    out[1] += transform.translation[1];
    out[2] += transform.translation[2];
}

void _TransformFrame(PomadeRigidTransform const &transform, PomadeFrame *frame)
{
    float t[3] = {frame->tx, frame->ty, frame->tz};
    float n[3] = {frame->nx, frame->ny, frame->nz};
    float b[3] = {frame->bx, frame->by, frame->bz};
    _TransformVector(transform, t, t);
    _TransformVector(transform, n, n);
    _TransformVector(transform, b, b);
    frame->tx = t[0]; frame->ty = t[1]; frame->tz = t[2];
    frame->nx = n[0]; frame->ny = n[1]; frame->nz = n[2];
    frame->bx = b[0]; frame->by = b[1]; frame->bz = b[2];
}

}  // namespace

bool PomadeCenterFramesCpu(float const *cx, float const *cy, float const *cz,
                          int nCv, std::vector<PomadeFrame> *frames,
                          std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    if (!cx || !cy || !cz || !frames || nCv < 2) {
        return fail("PomadeCenterFramesCpu: need >= 2 center CVs");
    }
    frames->resize(size_t(nCv));
    float tPrev[3];
    PomadeCenterTangent(cx, cy, cz, nCv, 0, tPrev);
    float n[3];
    PomadePerp3(tPrev, n);
    for (int i = 0; i < nCv; ++i) {
        float t[3];
        PomadeCenterTangent(cx, cy, cz, nCv, i, t);
        if (i > 0) {
            float p0[3] = {cx[i - 1], cy[i - 1], cz[i - 1]};
            float p1[3] = {cx[i], cy[i], cz[i]};
            PomadeReflectNormal(p0, p1, tPrev, t, n);
            tPrev[0] = t[0];
            tPrev[1] = t[1];
            tPrev[2] = t[2];
        }
        float b[3];
        PomadeCross3(t, n, b);
        PomadeFrame f;
        f.tx = t[0];
        f.ty = t[1];
        f.tz = t[2];
        f.nx = n[0];
        f.ny = n[1];
        f.nz = n[2];
        f.bx = b[0];
        f.by = b[1];
        f.bz = b[2];
        (*frames)[size_t(i)] = f;
    }
    return true;
}

bool PomadeTubeFramesCpu(PomadeTubeDesc const &tube,
                        std::vector<PomadeFrame> *frames, std::string *err)
{
    int const nCv = int(tube.centerX.size());
    if (nCv < 2 || tube.centerY.size() != size_t(nCv) ||
        tube.centerZ.size() != size_t(nCv)) {
        if (err) {
            *err = "PomadeTubeFramesCpu: malformed center column";
        }
        return false;
    }
    float const *q = tube.frameReference.data();
    if (!_ProperRotation(q)) {
        if (err) {
            *err = "PomadeTubeFramesCpu: frame reference is not a proper "
                   "rotation";
        }
        return false;
    }
    if (_ReferenceIsIdentity(tube.frameReference)) {
        // Keep legacy descriptors bit-for-bit on their historical K4 path.
        if (!PomadeCenterFramesCpu(tube.centerX.data(), tube.centerY.data(),
                                  tube.centerZ.data(), nCv, frames, err)) {
            return false;
        }
    } else {
        // K4's initial perpendicular is chosen from a world axis. Evaluate
        // the center in the descriptor's material frame instead, then map
        // every resulting axis back to world.  Translation by center[0]
        // makes the conversion well-conditioned without changing tangents.
        std::vector<float> lx(static_cast<size_t>(nCv), 0.0f);
        std::vector<float> ly(static_cast<size_t>(nCv), 0.0f);
        std::vector<float> lz(static_cast<size_t>(nCv), 0.0f);
        float const root[3] = {tube.centerX[0], tube.centerY[0],
                               tube.centerZ[0]};
        for (int i = 0; i < nCv; ++i) {
            float const delta[3] = {tube.centerX[size_t(i)] - root[0],
                                    tube.centerY[size_t(i)] - root[1],
                                    tube.centerZ[size_t(i)] - root[2]};
            // Q^-1 is Q^T because frameReference is a proper rotation.
            lx[size_t(i)] = q[0] * delta[0] + q[3] * delta[1] +
                            q[6] * delta[2];
            ly[size_t(i)] = q[1] * delta[0] + q[4] * delta[1] +
                            q[7] * delta[2];
            lz[size_t(i)] = q[2] * delta[0] + q[5] * delta[1] +
                            q[8] * delta[2];
        }
        if (!PomadeCenterFramesCpu(lx.data(), ly.data(), lz.data(), nCv,
                                  frames, err)) {
            return false;
        }
        for (PomadeFrame &frame : *frames) {
            float t[3] = {frame.tx, frame.ty, frame.tz};
            float n[3] = {frame.nx, frame.ny, frame.nz};
            float b[3] = {frame.bx, frame.by, frame.bz};
            _RotateVector(q, t, t);
            _RotateVector(q, n, n);
            _RotateVector(q, b, b);
            frame.tx = t[0]; frame.ty = t[1]; frame.tz = t[2];
            frame.nx = n[0]; frame.ny = n[1]; frame.nz = n[2];
            frame.bx = b[0]; frame.by = b[1]; frame.bz = b[2];
        }
    }
    if (frames->empty()) {
        return false;
    }
    if (!tube.rootFramePinned || frames->empty()) {
        return true;
    }

    // The stored frame comes from the region's fitted support plane.  Repair
    // its in-plane axis defensively so deserialised data cannot make K5/K9
    // non-orthonormal; its tangent remains the authored plane normal.
    PomadeFrame f = tube.rootFrame;
    float t[3] = {f.tx, f.ty, f.tz};
    float const tl = PomadeLen3(t);
    if (!(tl > 1e-12f)) {
        if (err) {
            *err = "PomadeTubeFramesCpu: pinned root has no plane normal";
        }
        return false;
    }
    t[0] /= tl;
    t[1] /= tl;
    t[2] /= tl;
    float n[3] = {f.nx, f.ny, f.nz};
    float const ndot = PomadeDot3(n, t);
    n[0] -= ndot * t[0];
    n[1] -= ndot * t[1];
    n[2] -= ndot * t[2];
    float const nl = PomadeLen3(n);
    if (!(nl > 1e-12f)) {
        PomadePerp3(t, n);
    } else {
        n[0] /= nl;
        n[1] /= nl;
        n[2] /= nl;
    }
    float b[3];
    PomadeCross3(t, n, b);
    f.tx = t[0];
    f.ty = t[1];
    f.tz = t[2];
    f.nx = n[0];
    f.ny = n[1];
    f.nz = n[2];
    f.bx = b[0];
    f.by = b[1];
    f.bz = b[2];
    (*frames)[0] = f;
    return true;
}

bool PomadeRigidTransformTubeCpu(PomadeTubeDesc const &source,
                                PomadeRigidTransform const &transform,
                                PomadeTubeDesc *out, std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    if (!out || !_RigidTransformValid(transform)) {
        return fail("PomadeRigidTransformTubeCpu: expected a proper rigid "
                    "transform and output");
    }
    int const nCv = int(source.centerX.size());
    if (nCv < 2 || source.centerY.size() != size_t(nCv) ||
        source.centerZ.size() != size_t(nCv)) {
        return fail("PomadeRigidTransformTubeCpu: malformed center column");
    }
    for (PomadeTubeSection const &section : source.sections) {
        if (section.u.size() != section.v.size() || section.u.empty() ||
            !std::isfinite(section.t) || !std::isfinite(section.scale) ||
            !std::isfinite(section.twist)) {
            return fail("PomadeRigidTransformTubeCpu: malformed section");
        }
    }

    // Keep a local source copy so callers may transform a descriptor in
    // place.  A frame evaluation also validates the stored material frame.
    PomadeTubeDesc const before = source;
    std::vector<PomadeFrame> beforeFrames;
    std::string frameErr;
    if (!PomadeTubeFramesCpu(before, &beforeFrames, &frameErr)) {
        return fail(frameErr.c_str());
    }

    PomadeTubeDesc after = before;
    for (int i = 0; i < nCv; ++i) {
        float point[3] = {before.centerX[size_t(i)], before.centerY[size_t(i)],
                          before.centerZ[size_t(i)]};
        _TransformPoint(transform, point, point);
        after.centerX[size_t(i)] = point[0];
        after.centerY[size_t(i)] = point[1];
        after.centerZ[size_t(i)] = point[2];
    }
    if (after.rootFramePinned) {
        _TransformFrame(transform, &after.rootFrame);
    }
    for (size_t i = 0; i + 2 < after.rootSurfaceOffsets.size(); i += 3)
        _TransformVector(transform, before.rootSurfaceOffsets.data() + i,
                          after.rootSurfaceOffsets.data() + i);
    // Q maps the K4 material frame to world.  Under a rigid pose change the
    // local curve is unchanged: Q' = R*Q.  Leaving all chart controls alone
    // preserves every interpolated K5 sample, rather than merely matching
    // the sparse authored section knots.
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            after.frameReference[size_t(row * 3 + col)] =
                transform.rotation[row * 3 + 0] *
                    before.frameReference[size_t(0 * 3 + col)] +
                transform.rotation[row * 3 + 1] *
                    before.frameReference[size_t(1 * 3 + col)] +
                transform.rotation[row * 3 + 2] *
                    before.frameReference[size_t(2 * 3 + col)];
        }
    }
    *out = std::move(after);
    return true;
}

bool PomadeSampleCenterCpu(PomadeTubeDesc const &tube,
                          std::vector<PomadeFrame> const &frames, float t,
                          float *px, float *py, float *pz, PomadeFrame *frame,
                          std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    int const n = int(tube.centerX.size());
    if (n < 2 || !px || !py || !pz || !frame ||
        tube.centerY.size() != size_t(n) ||
        tube.centerZ.size() != size_t(n) || frames.size() != size_t(n)) {
        return fail("PomadeSampleCenterCpu: bad tube or frame count");
    }
    float cp[3];
    PomadeEvalCenter(tube.centerX.data(), tube.centerY.data(),
                    tube.centerZ.data(), n, t, cp);
    *px = cp[0];
    *py = cp[1];
    *pz = cp[2];
    PomadeNlerpFrame(frames.data(), n, t, frame);
    return true;
}

bool PomadeSampleTubeRingCpu(PomadeTubeDesc const &tube,
                            std::vector<PomadeFrame> const &frames, float t,
                            std::vector<float> *positions, std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    int const nCv = int(tube.centerX.size());
    int const nSec = int(tube.sections.size());
    int const rv = tube.ringVerts;
    if (!positions || !std::isfinite(t) || nCv < 2 || nSec < 2 || rv < 3 || rv > 32 ||
        frames.size() != size_t(nCv) || tube.centerY.size() != size_t(nCv) ||
        tube.centerZ.size() != size_t(nCv)) {
        return fail("PomadeSampleTubeRingCpu: malformed tube or frames");
    }
    float previousT = -std::numeric_limits<float>::infinity();
    for (PomadeTubeSection const &section : tube.sections) {
        if (int(section.u.size()) != rv || int(section.v.size()) != rv) {
            return fail("PomadeSampleTubeRingCpu: section ring mismatch");
        }
        if (!std::isfinite(section.t) || section.t < previousT ||
            !std::isfinite(section.scale) || !std::isfinite(section.twist)) {
            return fail("PomadeSampleTubeRingCpu: invalid section parameters");
        }
        previousT = section.t;
        for (int slot = 0; slot < rv; ++slot) {
            if (!std::isfinite(section.u[size_t(slot)]) ||
                !std::isfinite(section.v[size_t(slot)])) {
                return fail("PomadeSampleTubeRingCpu: non-finite section CV");
            }
        }
    }
    float const domainStart = tube.sections.front().t;
    float const domainEnd = tube.sections.back().t;
    float const domainTolerance =
        std::max(1e-6f, std::fabs(domainEnd - domainStart) * 1e-6f);
    if (t < domainStart - domainTolerance ||
        t > domainEnd + domainTolerance) {
        return fail("PomadeSampleTubeRingCpu: t outside section domain");
    }
    // Stations are often reconstructed as begin + span*i/(count-1). Clamp
    // its one-ulp endpoint overshoot, but never accept a materially invalid
    // parameter from an external caller.
    t = t < domainStart ? domainStart : (t > domainEnd ? domainEnd : t);
    int k = 0;
    while (k + 1 < nSec - 1 && tube.sections[size_t(k + 1)].t < t) {
        ++k;
    }
    PomadeTubeSection const &s0 = tube.sections[size_t(k)];
    PomadeTubeSection const &s1 = tube.sections[size_t(k + 1)];
    PomadeTubeSection const &sP =
        tube.sections[size_t(k > 0 ? k - 1 : k)];
    PomadeTubeSection const &sN =
        tube.sections[size_t(k + 2 < nSec ? k + 2 : k + 1)];
    float const dt = s1.t > s0.t ? s1.t - s0.t : 1.0f;
    float f = (t - s0.t) / dt;
    f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
    bool const first = (k == 0);
    bool const last = (k + 1 == nSec - 1);
    float const m0sc = first ? s1.scale - s0.scale
                             : 0.5f * (s1.scale - sP.scale);
    float const m1sc = last ? s1.scale - s0.scale
                            : 0.5f * (sN.scale - s0.scale);
    float const m0tw = first ? s1.twist - s0.twist
                             : 0.5f * (s1.twist - sP.twist);
    float const m1tw = last ? s1.twist - s0.twist
                            : 0.5f * (sN.twist - s0.twist);
    float scale = PomadeHermite(s0.scale, s1.scale, m0sc, m1sc, f);
    if (!(scale > 1e-6f)) {
        scale = 1e-6f;
    }
    float const twist = PomadeHermite(s0.twist, s1.twist, m0tw, m1tw, f);
    float const ct = std::cos(twist), st = std::sin(twist);
    float cp[3];
    PomadeFrame fr;
    std::string sampleErr;
    if (!PomadeSampleCenterCpu(tube, frames, t, &cp[0], &cp[1], &cp[2], &fr,
                              &sampleErr)) {
        return fail(sampleErr.c_str());
    }
    positions->assign(size_t(rv) * 3, 0.0f);
    for (int slot = 0; slot < rv; ++slot) {
        float const m0u = first ? s1.u[size_t(slot)] - s0.u[size_t(slot)]
                                : 0.5f * (s1.u[size_t(slot)] -
                                          sP.u[size_t(slot)]);
        float const m1u = last ? s1.u[size_t(slot)] - s0.u[size_t(slot)]
                               : 0.5f * (sN.u[size_t(slot)] -
                                         s0.u[size_t(slot)]);
        float const m0v = first ? s1.v[size_t(slot)] - s0.v[size_t(slot)]
                                : 0.5f * (s1.v[size_t(slot)] -
                                          sP.v[size_t(slot)]);
        float const m1v = last ? s1.v[size_t(slot)] - s0.v[size_t(slot)]
                               : 0.5f * (sN.v[size_t(slot)] -
                                         s0.v[size_t(slot)]);
        float u = PomadeHermite(s0.u[size_t(slot)], s1.u[size_t(slot)], m0u,
                               m1u, f) * scale;
        float v = PomadeHermite(s0.v[size_t(slot)], s1.v[size_t(slot)], m0v,
                               m1v, f) * scale;
        float const ru = u * ct - v * st;
        float const rvv = u * st + v * ct;
        size_t const out = size_t(slot) * 3;
        (*positions)[out + 0] = cp[0] + fr.nx * ru + fr.bx * rvv;
        (*positions)[out + 1] = cp[1] + fr.ny * ru + fr.by * rvv;
        (*positions)[out + 2] = cp[2] + fr.nz * ru + fr.bz * rvv;
    }
    for (float value : *positions) {
        if (!std::isfinite(value)) {
            return fail("PomadeSampleTubeRingCpu: non-finite result");
        }
    }
    return true;
}

bool
PomadeCenterHandlePointCpu(PomadeTubeDesc const &tube, int centerCV, float *px,
                          float *py, float *pz, std::string *err)
{
    std::vector<PomadeFrame> frames;
    if (!PomadeTubeFramesCpu(tube, &frames, err)) return false;
    return PomadeCenterHandlePointCpu(tube, frames, centerCV, px, py, pz, err);
}

bool
PomadeCenterHandlePointCpu(PomadeTubeDesc const &tube,
                          std::vector<PomadeFrame> const &frames, int centerCV,
                          float *px, float *py, float *pz, std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    int const nCv = int(tube.centerX.size());
    int const nSec = int(tube.sections.size());
    int const rv = tube.ringVerts;
    if (!px || !py || !pz || centerCV < 0 || centerCV >= nCv || nCv < 2 ||
        nSec < 2 || rv < 3 ||
        tube.centerY.size() != size_t(nCv) ||
        tube.centerZ.size() != size_t(nCv) || frames.size() != size_t(nCv)) {
        return fail("PomadeCenterHandlePointCpu: malformed tube or CV");
    }
    for (PomadeTubeSection const &section : tube.sections) {
        if (int(section.u.size()) != rv || int(section.v.size()) != rv) {
            return fail("PomadeCenterHandlePointCpu: section ring mismatch");
        }
    }

    float const t = float(centerCV) / float(nCv - 1);
    int k = 0;
    while (k + 1 < nSec - 1 && tube.sections[size_t(k + 1)].t < t) {
        ++k;
    }
    PomadeTubeSection const &s0 = tube.sections[size_t(k)];
    PomadeTubeSection const &s1 = tube.sections[size_t(k + 1)];
    PomadeTubeSection const &sP =
        tube.sections[size_t(k > 0 ? k - 1 : k)];
    PomadeTubeSection const &sN =
        tube.sections[size_t(k + 2 < nSec ? k + 2 : k + 1)];
    float const dt = s1.t > s0.t ? s1.t - s0.t : 1.0f;
    float f = (t - s0.t) / dt;
    f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
    bool const first = (k == 0);
    bool const last = (k + 1 == nSec - 1);
    float const m0sc = first ? s1.scale - s0.scale
                             : 0.5f * (s1.scale - sP.scale);
    float const m1sc = last ? s1.scale - s0.scale
                            : 0.5f * (sN.scale - s0.scale);
    float const m0tw = first ? s1.twist - s0.twist
                             : 0.5f * (s1.twist - sP.twist);
    float const m1tw = last ? s1.twist - s0.twist
                            : 0.5f * (sN.twist - s0.twist);
    float scale = PomadeHermite(s0.scale, s1.scale, m0sc, m1sc, f);
    if (!(scale > 1e-6f)) {
        scale = 1e-6f;
    }
    float const twist = PomadeHermite(s0.twist, s1.twist, m0tw, m1tw, f);
    float const ct = std::cos(twist);
    float const st = std::sin(twist);

    // The signed shoelace centroid is affine-covariant, so calculate it in
    // the evaluated (twisted and scaled) ring chart then map it through the
    // same K4 frame used by the visible K5 section. It handles concavity and
    // winding without treating a sparse clipped ring as a vertex average.
    double twiceArea = 0.0;
    double centroidU = 0.0;
    double centroidV = 0.0;
    double meanU = 0.0;
    double meanV = 0.0;
    auto point = [&](int slot, float *u, float *v) {
        float const m0u = first ? s1.u[size_t(slot)] - s0.u[size_t(slot)]
                                : 0.5f * (s1.u[size_t(slot)] -
                                          sP.u[size_t(slot)]);
        float const m1u = last ? s1.u[size_t(slot)] - s0.u[size_t(slot)]
                               : 0.5f * (sN.u[size_t(slot)] -
                                         s0.u[size_t(slot)]);
        float const m0v = first ? s1.v[size_t(slot)] - s0.v[size_t(slot)]
                                : 0.5f * (s1.v[size_t(slot)] -
                                          sP.v[size_t(slot)]);
        float const m1v = last ? s1.v[size_t(slot)] - s0.v[size_t(slot)]
                               : 0.5f * (sN.v[size_t(slot)] -
                                         s0.v[size_t(slot)]);
        float const baseU = PomadeHermite(s0.u[size_t(slot)],
                                         s1.u[size_t(slot)], m0u, m1u, f) *
                            scale;
        float const baseV = PomadeHermite(s0.v[size_t(slot)],
                                         s1.v[size_t(slot)], m0v, m1v, f) *
                            scale;
        *u = baseU * ct - baseV * st;
        *v = baseU * st + baseV * ct;
    };
    float previousU = 0.0f, previousV = 0.0f;
    point(rv - 1, &previousU, &previousV);
    for (int slot = 0; slot < rv; ++slot) {
        float u = 0.0f, v = 0.0f;
        point(slot, &u, &v);
        double const cross = double(previousU) * double(v) -
                             double(u) * double(previousV);
        twiceArea += cross;
        centroidU += (double(previousU) + double(u)) * cross;
        centroidV += (double(previousV) + double(v)) * cross;
        meanU += u;
        meanV += v;
        previousU = u;
        previousV = v;
    }

    float cp[3];
    PomadeFrame frame;
    std::string sampleErr;
    if (!PomadeSampleCenterCpu(tube, frames, t, &cp[0], &cp[1], &cp[2],
                              &frame, &sampleErr)) {
        return fail(sampleErr.c_str());
    }
    // A degenerate but offset section still has a visible core. Its vertex
    // mean gives that handle a stable position; the raw center cage is only
    // a fallback for malformed descriptors above.
    float const u = std::fabs(twiceArea) <= 1e-12
                        ? float(meanU / double(rv))
                        : float(centroidU / (3.0 * twiceArea));
    float const v = std::fabs(twiceArea) <= 1e-12
                        ? float(meanV / double(rv))
                        : float(centroidV / (3.0 * twiceArea));
    *px = cp[0] + frame.nx * u + frame.bx * v;
    *py = cp[1] + frame.ny * u + frame.by * v;
    *pz = cp[2] + frame.nz * u + frame.bz * v;
    return std::isfinite(*px) && std::isfinite(*py) && std::isfinite(*pz)
               ? true
               : fail("PomadeCenterHandlePointCpu: non-finite handle");
}

float PomadeSectionMeanRadius(PomadeTubeSection const &section)
{
    if (section.u.empty()) {
        return 0.0f;
    }
    double acc = 0.0;
    size_t const n = std::min(section.u.size(), section.v.size());
    for (size_t i = 0; i < n; ++i) {
        double const uu = double(section.u[i]) * double(section.scale);
        double const vv = double(section.v[i]) * double(section.scale);
        acc += std::sqrt(uu * uu + vv * vv);
    }
    return n > 0 ? float(acc / double(n)) : 0.0f;
}

int PomadeTessellatedRingCount(PomadeTubeDesc const &tube, int segmentsPerSpan)
{
    int const nSec = int(tube.sections.size());
    if (nSec < 2 || segmentsPerSpan < 1) {
        return 0;
    }
    return (nSec - 1) * segmentsPerSpan + 1;
}

bool PomadeTessellateCpu(PomadeTubeDesc const &tube,
                        std::vector<PomadeFrame> const &frames,
                        int segmentsPerSpan, std::vector<float> *positions,
                        std::vector<float> *normals,
                        std::vector<float> *ringT, std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    int const nCv = int(tube.centerX.size());
    int const nSec = int(tube.sections.size());
    int const rv = tube.ringVerts;
    if (nCv < 2 || nSec < 2 || rv < 3 || rv > 32 || segmentsPerSpan < 1 ||
        !positions || !normals || !ringT ||
        frames.size() != size_t(nCv)) {
        return fail("PomadeTessellateCpu: bad tube, frames or span count");
    }
    for (auto const &s : tube.sections) {
        if (int(s.u.size()) != rv || int(s.v.size()) != rv) {
            return fail("PomadeTessellateCpu: section ring size mismatch");
        }
    }
    for (int i = 1; i < nSec; ++i) {
        if (!(tube.sections[size_t(i)].t >= tube.sections[size_t(i - 1)].t)) {
            return fail("PomadeTessellateCpu: section t must ascend");
        }
    }
    int const nRings = PomadeTessellatedRingCount(tube, segmentsPerSpan);
    positions->resize(size_t(nRings) * size_t(rv) * 3);
    normals->resize(size_t(nRings) * size_t(rv) * 3);
    ringT->resize(size_t(nRings));
    for (int r = 0; r < nRings; ++r) {
        // Subdivide each authored span, not the entire parameter domain:
        // the sparse rings must remain exact rows even with uneven knots.
        int const k = std::min(r / segmentsPerSpan, nSec - 2);
        PomadeTubeSection const &s0 = tube.sections[size_t(k)];
        PomadeTubeSection const &s1 = tube.sections[size_t(k + 1)];
        PomadeTubeSection const &sP = tube.sections[size_t(k > 0 ? k - 1 : k)];
        PomadeTubeSection const &sN =
            tube.sections[size_t(k + 2 < nSec ? k + 2 : k + 1)];
        int const localRow = r - k * segmentsPerSpan;
        float const sample = PomadeDisplaySpanFraction(
            localRow, segmentsPerSpan, k, nSec - 1);
        float const t = sample == 1.0f ? s1.t
            : s0.t + (s1.t - s0.t) * sample;
        bool const rootHold = segmentsPerSpan >= 3 && k == 0 && localRow == 1;
        bool const tipHold = segmentsPerSpan >= 3 && k == nSec - 2 &&
                             localRow == segmentsPerSpan - 1;
        float const f = rootHold ? 0.0f : (tipHold ? 1.0f : sample);
        (*ringT)[size_t(r)] = t;
        // Central differences inside, one-sided at the ends (matches
        // PomadeEvalCenter, except for the explicit end holding rows).
        bool const first = (k == 0);
        bool const last = (k + 1 == nSec - 1);
        float const m0sc = first ? s1.scale - s0.scale
                                 : 0.5f * (s1.scale - sP.scale);
        float const m1sc = last ? s1.scale - s0.scale
                                : 0.5f * (sN.scale - s0.scale);
        float const m0tw = first ? s1.twist - s0.twist
                                 : 0.5f * (s1.twist - sP.twist);
        float const m1tw = last ? s1.twist - s0.twist
                                : 0.5f * (sN.twist - s0.twist);
        float sc = PomadeHermite(s0.scale, s1.scale, m0sc, m1sc, f);
        float tw = PomadeHermite(s0.twist, s1.twist, m0tw, m1tw, f);
        if (!(sc > 1e-6f)) {
            sc = 1e-6f;
        }
        float const ct = std::cos(tw), st = std::sin(tw);
        float cp[3];
        PomadeFrame fr;
        std::string derr;
        if (!PomadeSampleCenterCpu(tube, frames, t, &cp[0], &cp[1], &cp[2],
                                 &fr, &derr)) {
            return fail(derr.c_str());
        }
        float nA[3] = {fr.nx, fr.ny, fr.nz};
        float bA[3] = {fr.bx, fr.by, fr.bz};
        for (int s = 0; s < rv; ++s) {
            float const m0u =
                first ? s1.u[size_t(s)] - s0.u[size_t(s)]
                      : 0.5f * (s1.u[size_t(s)] - sP.u[size_t(s)]);
            float const m1u =
                last ? s1.u[size_t(s)] - s0.u[size_t(s)]
                     : 0.5f * (sN.u[size_t(s)] - s0.u[size_t(s)]);
            float const m0v =
                first ? s1.v[size_t(s)] - s0.v[size_t(s)]
                      : 0.5f * (s1.v[size_t(s)] - sP.v[size_t(s)]);
            float const m1v =
                last ? s1.v[size_t(s)] - s0.v[size_t(s)]
                     : 0.5f * (sN.v[size_t(s)] - s0.v[size_t(s)]);
            float uu =
                PomadeHermite(s0.u[size_t(s)], s1.u[size_t(s)], m0u, m1u, f);
            float vv =
                PomadeHermite(s0.v[size_t(s)], s1.v[size_t(s)], m0v, m1v, f);
            uu *= sc;
            vv *= sc;
            float const ru = uu * ct - vv * st;
            float const rvv = uu * st + vv * ct;
            size_t const o = (size_t(r) * size_t(rv) + size_t(s)) * 3;
            (*positions)[o + 0] = cp[0] + nA[0] * ru + bA[0] * rvv;
            (*positions)[o + 1] = cp[1] + nA[1] * ru + bA[1] * rvv;
            (*positions)[o + 2] = cp[2] + nA[2] * ru + bA[2] * rvv;
            if (tube.rootSurfaceOffsets.size() == size_t(rv) * 3) {
                float const weight = rootHold ? 1.0f :
                    PomadeRootAttachmentWeight(t - tube.sections.front().t,
                        tube.sections[1].t - tube.sections.front().t);
                for (int axis = 0; axis < 3; ++axis)
                    (*positions)[o + size_t(axis)] += weight *
                        tube.rootSurfaceOffsets[size_t(s) * 3 + size_t(axis)];
            }
            float nn[3] = {nA[0] * ru + bA[0] * rvv, nA[1] * ru + bA[1] * rvv,
                           nA[2] * ru + bA[2] * rvv};
            float const nl = PomadeLen3(nn);
            if (nl > 1e-12f) {
                (*normals)[o + 0] = nn[0] / nl;
                (*normals)[o + 1] = nn[1] / nl;
                (*normals)[o + 2] = nn[2] / nl;
            } else {
                (*normals)[o + 0] = fr.tx;
                (*normals)[o + 1] = fr.ty;
                (*normals)[o + 2] = fr.tz;
            }
        }
    }
    return true;
}

namespace {

// Dart-throwing state shared by the disc and mesh samplers: candidates are
// drawn in hash order and kept when farther than minDist from every kept
// root (a Poisson-disk approximation; deterministic per (tubeId, seed)).
bool _FarEnough(float x, float y, std::vector<float> const &keptX,
                std::vector<float> const &keptY, float minDist2)
{
    for (size_t i = 0; i < keptX.size(); ++i) {
        float const dx = x - keptX[i], dy = y - keptY[i];
        if (dx * dx + dy * dy < minDist2) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool PomadeRootSampleDiscCpu(int tubeId, int seed, float radius,
                            float const center[3], PomadeFrame const &frame,
                            int count, std::vector<PomadeGuideRoot> *roots,
                            int frozen, std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    if (!center || !roots || count < 0 || frozen < 0 || frozen > count ||
        !(radius > 0.0f)) {
        return fail("PomadeRootSampleDiscCpu: bad count, radius or buffers");
    }
    if (frozen > int(roots->size())) {
        return fail("PomadeRootSampleDiscCpu: frozen exceeds stored roots");
    }
    roots->resize(size_t(count));
    if (count == 0) {
        return true;
    }
    // Kept-root rejection works in the normalised disc so frozen roots
    // (stored with their ru/rv) seed the stream exactly.
    std::vector<float> keptX, keptY;
    keptX.reserve(size_t(count));
    keptY.reserve(size_t(count));
    for (int i = 0; i < frozen; ++i) {
        keptX.push_back((*roots)[size_t(i)].ru);
        keptY.push_back((*roots)[size_t(i)].rv);
    }
    float const twoPi = 6.28318530717958647692f;
    // Target spacing for `count` blue-noise roots in the unit disc.
    float const minDist =
        count > 1 ? 0.9f * std::sqrt(3.14159265f / float(count)) : 0.0f;
    float const minDist2 = minDist * minDist;
    uint64_t const key =
        (uint64_t(uint32_t(tubeId)) << 32) | uint64_t(uint32_t(seed));
    int const budget = count * 40 + 64;
    int accepted = frozen;
    for (int cand = 0; cand < budget && accepted < count; ++cand) {
        float const u1 = PomadeHash01(uint64_t(cand) * 2 + key, kPomadeSaltRoot);
        float const u2 =
            PomadeHash01(uint64_t(cand) * 2 + 1 + (key ^ 0x85ebca6bu),
                        kPomadeSaltRoot);
        float const rr = std::sqrt(u1);
        float const th = twoPi * u2;
        float const ru = rr * std::cos(th), rv = rr * std::sin(th);
        if (minDist2 > 0.0f && !_FarEnough(ru, rv, keptX, keptY, minDist2)) {
            continue;
        }
        keptX.push_back(ru);
        keptY.push_back(rv);
        PomadeGuideRoot root;
        root.faceId = -1;
        root.u = ru;
        root.v = rv;
        root.ru = ru;
        root.rv = rv;
        root.px = center[0] + (frame.nx * ru + frame.bx * rv) * radius;
        root.py = center[1] + (frame.ny * ru + frame.by * rv) * radius;
        root.pz = center[2] + (frame.nz * ru + frame.bz * rv) * radius;
        (*roots)[size_t(accepted)] = root;
        ++accepted;
    }
    // Best-effort tail: the budget is generous, but a degenerate (tiny)
    // count must still fill exactly `count` roots.
    for (int i = accepted; i < count; ++i) {
        float const u1 =
            PomadeHash01(uint64_t(budget + i) * 2 + key, kPomadeSaltRoot);
        float const u2 = PomadeHash01(
            uint64_t(budget + i) * 2 + 1 + (key ^ 0x85ebca6bu),
            kPomadeSaltRoot);
        float const rr = std::sqrt(u1);
        float const th = twoPi * u2;
        PomadeGuideRoot root;
        root.faceId = -1;
        root.ru = rr * std::cos(th);
        root.rv = rr * std::sin(th);
        root.u = root.ru;
        root.v = root.rv;
        root.px = center[0] + (frame.nx * root.ru + frame.bx * root.rv) *
                                  radius;
        root.py = center[1] + (frame.ny * root.ru + frame.by * root.rv) *
                                  radius;
        root.pz = center[2] + (frame.nz * root.ru + frame.bz * root.rv) *
                                  radius;
        (*roots)[size_t(i)] = root;
    }
    return true;
}

bool PomadeRootSampleMeshCpu(float const *points, int const *faceCounts,
                            int const *faceIndices, int const *faceOffsets,
                            int faceCount, int const *regionFaces,
                            int regionFaceCount, float density, int tubeId,
                            int seed, float const rootCenter[3],
                            PomadeFrame const &rootFrame, float rootRadius,
                            std::vector<PomadeGuideRoot> *roots, int frozen,
                            std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    if (!points || !faceCounts || !faceIndices || !faceOffsets ||
        !regionFaces || !rootCenter || !roots || faceCount <= 0 ||
        regionFaceCount <= 0 || !(density >= 0.0f) || frozen < 0 ||
        !(rootRadius > 0.0f)) {
        return fail("PomadeRootSampleMeshCpu: bad mesh, density or buffers");
    }
    auto triArea = [&](float const a[3], float const b[3], float const c[3]) {
        float e1[3], e2[3], cr[3];
        PomadeSub3(b, a, e1);
        PomadeSub3(c, a, e2);
        PomadeCross3(e1, e2, cr);
        return 0.5f * PomadeLen3(cr);
    };
    uint64_t const key =
        (uint64_t(uint32_t(tubeId)) << 32) | uint64_t(uint32_t(seed));
    std::vector<PomadeGuideRoot> fresh;
    for (int rf = 0; rf < regionFaceCount; ++rf) {
        int const f = regionFaces[rf];
        if (f < 0 || f >= faceCount) {
            return fail("PomadeRootSampleMeshCpu: region face out of range");
        }
        int const nv = faceCounts[f];
        int const off = faceOffsets[f];
        if (nv < 3) {
            continue;
        }
        float const *v0 = points + size_t(faceIndices[off]) * 3;
        float area = 0.0f;
        for (int k = 1; k + 1 < nv; ++k) {
            float const *v1 = points + size_t(faceIndices[off + k]) * 3;
            float const *v2 = points + size_t(faceIndices[off + k + 1]) * 3;
            area += triArea(v0, v1, v2);
        }
        float const expected = density * area;
        int take = int(expected);
        float const frac = expected - float(take);
        if (PomadeHash01(key ^ (uint64_t(f) * 0x9E3779B1u), kPomadeSaltRoot) <
            frac) {
            ++take;
        }
        // Per-face dart throwing in barycentric space, rejection in the
        // normalised root plane (shared with the disc path).
        std::vector<float> keptX, keptY;
        float const minDist =
            take > 1 ? 0.9f * std::sqrt(3.14159265f / float(take)) *
                           std::sqrt(area / 3.14159265f) / rootRadius
                     : 0.0f;
        float const minDist2 = minDist * minDist;
        int const budget = take * 40 + 16;
        int accepted = 0;
        auto emit = [&](float ba, float bb) {
            float const bc = 1.0f - ba - bb;
            int tri = int(ba * float(nv - 2));
            if (tri < 0) {
                tri = 0;
            }
            if (tri > nv - 3) {
                tri = nv - 3;
            }
            float const *t1 = points + size_t(faceIndices[off + tri]) * 3;
            float const *t2 =
                points + size_t(faceIndices[off + tri + 1]) * 3;
            float const *t3 =
                points + size_t(faceIndices[off + tri + 2]) * 3;
            float const w1 = std::fabs(bb), w2 = std::fabs(bc);
            float const wsum = std::fabs(ba) + w1 + w2;
            float const i0 = wsum > 0.0f ? std::fabs(ba) / wsum : 1.0f / 3.0f;
            float const i1 = wsum > 0.0f ? w1 / wsum : 1.0f / 3.0f;
            float const i2 = wsum > 0.0f ? w2 / wsum : 1.0f / 3.0f;
            float p[3] = {t1[0] * i0 + t2[0] * i1 + t3[0] * i2,
                          t1[1] * i0 + t2[1] * i1 + t3[1] * i2,
                          t1[2] * i0 + t2[2] * i1 + t3[2] * i2};
            float d[3];
            PomadeSub3(p, rootCenter, d);
            float nA[3] = {rootFrame.nx, rootFrame.ny, rootFrame.nz};
            float bA[3] = {rootFrame.bx, rootFrame.by, rootFrame.bz};
            PomadeGuideRoot root;
            root.faceId = f;
            root.u = i1;
            root.v = i2;
            root.px = p[0];
            root.py = p[1];
            root.pz = p[2];
            root.ru = PomadeDot3(d, nA) / rootRadius;
            root.rv = PomadeDot3(d, bA) / rootRadius;
            return root;
        };
        for (int cand = 0; cand < budget && accepted < take; ++cand) {
            uint64_t const ck = key ^ (uint64_t(f) * 0x9E3779B1u) ^
                                (uint64_t(cand) * 0x85EBCA77u);
            float const ba = PomadeHash01(ck, kPomadeSaltRoot);
            float const bb = PomadeHash01(ck ^ 0xC2B2AE35u, kPomadeSaltRoot);
            PomadeGuideRoot root = emit(ba, bb);
            float rl = std::sqrt(root.ru * root.ru + root.rv * root.rv);
            if (rl > 1.0f) {
                root.ru /= rl;
                root.rv /= rl;
            }
            if (minDist2 > 0.0f &&
                !_FarEnough(root.ru, root.rv, keptX, keptY, minDist2)) {
                continue;
            }
            keptX.push_back(root.ru);
            keptY.push_back(root.rv);
            fresh.push_back(root);
            ++accepted;
        }
        for (int i = accepted; i < take; ++i) {
            uint64_t const ck = key ^ (uint64_t(f) * 0x9E3779B1u) ^
                                (uint64_t(budget + i) * 0x85EBCA77u);
            PomadeGuideRoot root =
                emit(PomadeHash01(ck, kPomadeSaltRoot),
                     PomadeHash01(ck ^ 0xC2B2AE35u, kPomadeSaltRoot));
            float rl = std::sqrt(root.ru * root.ru + root.rv * root.rv);
            if (rl > 1.0f) {
                root.ru /= rl;
                root.rv /= rl;
            }
            fresh.push_back(root);
        }
    }
    if (frozen > int(roots->size())) {
        return fail("PomadeRootSampleMeshCpu: frozen exceeds stored roots");
    }
    std::vector<PomadeGuideRoot> out;
    out.reserve(size_t(frozen) + fresh.size());
    for (int i = 0; i < frozen; ++i) {
        out.push_back((*roots)[size_t(i)]);
    }
    for (auto const &r : fresh) {
        out.push_back(r);
    }
    *roots = std::move(out);
    return true;
}

bool PomadeRootSampleRegionMeshCpu(
    PomadeScalpMesh const &scalp, PomadeRegionLoops const &loops,
    int sourceRegionId, float density, int tubeId, int seed,
    float const rootCenter[3], PomadeFrame const &rootFrame, float rootRadius,
    std::vector<PomadeGuideRoot> *roots, int frozen, std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    if (!scalp.finalized || !rootCenter || !roots || frozen < 0 ||
        frozen > int(roots->size()) || !(density >= 0.0f) ||
        !(rootRadius > 0.0f)) {
        return fail("PomadeRootSampleRegionMeshCpu: bad region or buffers");
    }
    int loop = -1;
    for (size_t i = 0; i < loops.regionIds.size(); ++i) {
        if (loops.regionIds[i] == sourceRegionId) {
            loop = int(i);
            break;
        }
    }
    if (loop < 0 || size_t(loop) >= loops.loopCount.size() ||
        loops.loopCount[size_t(loop)] < 3) {
        return fail("PomadeRootSampleRegionMeshCpu: unknown region loop");
    }
    // The graph loop is the support. Its projected area is a stable density
    // estimate on a chart-local scalp; candidates below still have to pass
    // the exact point-in-polygon test before a root can escape this function.
    int const interp = loops.interpIds[size_t(loop)];
    auto loopArea = [&](int loopIndex) {
        float const *loopP = &loops.planeP[size_t(loopIndex) * 3];
        float const *loopU = &loops.basisU[size_t(loopIndex) * 3];
        float const *loopV = &loops.basisV[size_t(loopIndex) * 3];
        int const loopBegin = loops.loopBegin[size_t(loopIndex)];
        int const loopCount = loops.loopCount[size_t(loopIndex)];
        double twiceArea = 0.0;
        for (int i = 0; i < loopCount; ++i) {
            auto project = [&](int k, float *x, float *y) {
                float const *q =
                    &loops.points[size_t(loopBegin + k) * 3];
                float const dx = q[0] - loopP[0], dy = q[1] - loopP[1],
                            dz = q[2] - loopP[2];
                *x = dx * loopU[0] + dy * loopU[1] + dz * loopU[2];
                *y = dx * loopV[0] + dy * loopV[1] + dz * loopV[2];
            };
            float ax, ay, bx, by;
            project(i, &ax, &ay);
            project((i + 1) % loopCount, &bx, &by);
            twiceArea += double(ax) * by - double(ay) * bx;
        }
        return float(0.5 * std::fabs(twiceArea));
    };
    // Linked graph regions deliberately share one interpolation id and one
    // fill support. Unlinked overlaps follow the classifier's lowest-id
    // rule below, exactly as the Ptex map does.
    float area = 0.0f;
    std::vector<int> supportLoops;
    for (size_t i = 0; i < loops.loopCount.size(); ++i) {
        if (loops.interpIds[i] == interp) {
            float const loopSupportArea = loopArea(int(i));
            if (loopSupportArea > 0.0f) {
                area += loopSupportArea;
                supportLoops.push_back(int(i));
            }
        }
    }
    uint64_t const key =
        (uint64_t(uint32_t(tubeId)) << 32) | uint64_t(uint32_t(seed));
    float const expected = density * area;
    int target = int(expected);
    if (PomadeHash01(key ^ uint64_t(uint32_t(sourceRegionId)),
                    kPomadeSaltRoot) < expected - float(target)) {
        ++target;
    }
    int const keepCount = std::min(frozen, target);
    if (target == 0) {
        roots->clear();
        return true;
    }
    // Draw within a linked member's chart box, chosen area-weighted, then project accepted candidates to
    // the scalp. This has bounded work for a tiny graph polygon on one huge
    // coarse face, where whole-scalp rejection could miss it entirely.
    auto bounds = [&](int which, float *minX, float *maxX, float *minY,
                      float *maxY) {
        for (int i = 0; i < loops.loopCount[size_t(which)]; ++i) {
            float const *q = &loops.points[size_t(loops.loopBegin[size_t(which)] + i) * 3];
            float const dx = q[0] - loops.planeP[size_t(which) * 3 + 0];
            float const dy = q[1] - loops.planeP[size_t(which) * 3 + 1];
            float const dz = q[2] - loops.planeP[size_t(which) * 3 + 2];
            float const x = dx * loops.basisU[size_t(which) * 3 + 0] +
                            dy * loops.basisU[size_t(which) * 3 + 1] +
                            dz * loops.basisU[size_t(which) * 3 + 2];
            float const y = dx * loops.basisV[size_t(which) * 3 + 0] +
                            dy * loops.basisV[size_t(which) * 3 + 1] +
                            dz * loops.basisV[size_t(which) * 3 + 2];
            if (i == 0) {
                *minX = *maxX = x;
                *minY = *maxY = y;
            } else {
                *minX = std::min(*minX, x); *maxX = std::max(*maxX, x);
                *minY = std::min(*minY, y); *maxY = std::max(*maxY, y);
            }
        }
    };
    // Choose proposal charts by their bounding-box area. After the
    // point-in-polygon rejection, accepted mass is proportional to polygon
    // area instead of polygonArea^2 / boxArea for mixed linked shapes.
    std::vector<float> supportMinX, supportMaxX, supportMinY, supportMaxY;
    std::vector<float> proposalCumulative;
    float proposalArea = 0.0f;
    for (int support : supportLoops) {
        float minX = 0, maxX = 0, minY = 0, maxY = 0;
        bounds(support, &minX, &maxX, &minY, &maxY);
        supportMinX.push_back(minX); supportMaxX.push_back(maxX);
        supportMinY.push_back(minY); supportMaxY.push_back(maxY);
        proposalArea += std::max((maxX - minX) * (maxY - minY), 1e-12f);
        proposalCumulative.push_back(proposalArea);
    }
    std::vector<PomadeGuideRoot> out;
    out.reserve(size_t(target));
    for (int i = 0; i < keepCount; ++i) {
        // Frozen roots are accepted only while still inside this exact
        // support. A moved graph boundary must never preserve an escaped
        // root into the next fill.
        float q[3] = {(*roots)[size_t(i)].px, (*roots)[size_t(i)].py,
                      (*roots)[size_t(i)].pz};
        if (PomadeClassifyPointCpu(loops, q) == interp &&
            PomadeScalpFaceActive(scalp, (*roots)[size_t(i)].faceId)) {
            out.push_back((*roots)[size_t(i)]);
        }
    }
    size_t const frozenKept = out.size();
    int const budget = std::max(512, target * 128);
    for (int cand = 0; cand < budget && int(out.size()) < target; ++cand) {
        uint64_t const ck = key ^ (uint64_t(cand) * 0x9E3779B185EBCA87ull);
        float const choose = PomadeHash01(ck ^ 0x27D4EB2Fu, kPomadeSaltRoot) * proposalArea;
        size_t selected = size_t(std::lower_bound(proposalCumulative.begin(),
                                                   proposalCumulative.end(), choose) -
                                 proposalCumulative.begin());
        if (selected >= supportLoops.size()) selected = supportLoops.size() - 1;
        int const sampleLoop = supportLoops[selected];
        float const minX = supportMinX[selected], maxX = supportMaxX[selected];
        float const minY = supportMinY[selected], maxY = supportMaxY[selected];
        float const px = minX + (maxX - minX) * PomadeHash01(ck, kPomadeSaltRoot);
        float const py = minY + (maxY - minY) *
                                      PomadeHash01(ck ^ 0xC2B2AE35u, kPomadeSaltRoot);
        float candidate[3] = {
            loops.planeP[size_t(sampleLoop) * 3 + 0] +
                loops.basisU[size_t(sampleLoop) * 3 + 0] * px +
                loops.basisV[size_t(sampleLoop) * 3 + 0] * py,
            loops.planeP[size_t(sampleLoop) * 3 + 1] +
                loops.basisU[size_t(sampleLoop) * 3 + 1] * px +
                loops.basisV[size_t(sampleLoop) * 3 + 1] * py,
            loops.planeP[size_t(sampleLoop) * 3 + 2] +
                loops.basisU[size_t(sampleLoop) * 3 + 2] * px +
                loops.basisV[size_t(sampleLoop) * 3 + 2] * py};
        if (!PomadePointInRegionCpu(loops, sampleLoop, candidate)) {
            continue;
        }
        // Project over EVERY parent face, then drop a candidate that lands
        // outside a bound face subset: projecting onto the subset alone
        // would pile the rejected area's roots up along its border.
        PomadeHit const hit =
            PomadeClosestPointCpu(scalp, candidate, /*activeOnly*/ false);
        if (!hit.hit || !PomadeScalpFaceActive(scalp, hit.faceId)) {
            continue;
        }
        float q[3] = {hit.px, hit.py, hit.pz};
        if (PomadeClassifyPointCpu(loops, q) != interp) {
            continue;
        }
        PomadeGuideRoot root;
        root.faceId = hit.faceId;
        root.px = q[0];
        root.py = q[1];
        root.pz = q[2];
        root.u = hit.u;
        root.v = hit.v;
        float d[3] = {q[0] - rootCenter[0], q[1] - rootCenter[1],
                      q[2] - rootCenter[2]};
        float const nA[3] = {rootFrame.nx, rootFrame.ny, rootFrame.nz};
        float const bA[3] = {rootFrame.bx, rootFrame.by, rootFrame.bz};
        root.ru = PomadeDot3(d, nA) / rootRadius;
        root.rv = PomadeDot3(d, bA) / rootRadius;
        bool duplicate = false;
        // The deterministic fresh stream has distinct hash candidates. Only
        // compare the preserved prefix, which prevents a freeze/refill from
        // replaying its old samples without turning ordinary preview fills
        // into O(n^2) work.
        for (size_t i = 0; i < frozenKept; ++i) {
            PomadeGuideRoot const &prior = out[i];
            float const dx = prior.px - root.px, dy = prior.py - root.py,
                        dz = prior.pz - root.pz;
            if (dx * dx + dy * dy + dz * dz < 1e-12f) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            out.push_back(root);
        }
    }
    *roots = std::move(out);
    return true;
}

void PomadeFilterGuideRootsByOwnershipCells(
    std::vector<PomadeRootOwnershipCell> const &cells,
    std::vector<PomadeGuideRoot> *roots)
{
    if (!roots || cells.empty()) {
        return;
    }
    std::vector<PomadeGuideRoot> kept;
    kept.reserve(roots->size());
    for (PomadeGuideRoot const &root : *roots) {
        float const p[3] = {root.px, root.py, root.pz};
        bool owns = true;
        for (PomadeRootOwnershipCell const &cell : cells) {
            int const count = int(cell.childCenters.size() / 3);
            if (cell.childIndex < 0 || cell.childIndex >= count ||
                PomadeOwningChildCell(cell.rootCenter, cell.frame,
                                     cell.childCenters.data(), count, p) !=
                    cell.childIndex) {
                owns = false;
                break;
            }
        }
        if (owns) {
            kept.push_back(root);
        }
    }
    *roots = std::move(kept);
}

namespace {

struct _K9Point {
    float u = 0.0f;
    float v = 0.0f;
};

float _K9Cross(_K9Point const &a, _K9Point const &b, _K9Point const &c)
{
    return (b.u - a.u) * (c.v - a.v) - (b.v - a.v) * (c.u - a.u);
}

bool _K9Bary(_K9Point const &p, _K9Point const &a, _K9Point const &b,
             _K9Point const &c, float *w0, float *w1, float *w2)
{
    float const d = _K9Cross(a, b, c);
    if (std::fabs(d) <= 1e-12f) {
        return false;
    }
    *w0 = _K9Cross(p, b, c) / d;
    *w1 = _K9Cross(p, c, a) / d;
    *w2 = 1.0f - *w0 - *w1;
    return true;
}

bool _K9Inside(float w0, float w1, float w2)
{
    constexpr float eps = 2e-5f;
    return w0 >= -eps && w1 >= -eps && w2 >= -eps;
}

_K9Point _K9ClosestSegment(_K9Point const &p, _K9Point const &a,
                            _K9Point const &b)
{
    float const du = b.u - a.u, dv = b.v - a.v;
    float const d2 = du * du + dv * dv;
    float f = d2 > 1e-20f ? ((p.u - a.u) * du + (p.v - a.v) * dv) / d2
                           : 0.0f;
    f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
    return {a.u + du * f, a.v + dv * f};
}

bool _K9Triangulate(std::vector<_K9Point> const &polygon,
                    std::vector<std::array<int, 3>> *triangles)
{
    if (!triangles || polygon.size() < 3) {
        return false;
    }
    std::vector<usdGen::UsdGenVec2f> actual;
    actual.reserve(polygon.size());
    for (_K9Point const &point : polygon) {
        actual.emplace_back(point.u, point.v);
    }
    return usdGen::UsdGenTriangulateConcaveMaterialSlots(actual, triangles);
}

bool _K9Bind(std::vector<_K9Point> const &polygon,
             std::vector<std::array<int, 3>> const &triangles,
             _K9Point const &p, PomadeGuideMaterialBinding *out)
{
    if (!out) {
        return false;
    }
    float best2 = std::numeric_limits<float>::infinity();
    PomadeGuideMaterialBinding best;
    for (std::array<int, 3> const &tri : triangles) {
        _K9Point const &a = polygon[size_t(tri[0])];
        _K9Point const &b = polygon[size_t(tri[1])];
        _K9Point const &c = polygon[size_t(tri[2])];
        float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f;
        if (!_K9Bary(p, a, b, c, &w0, &w1, &w2)) {
            continue;
        }
        _K9Point q = p;
        if (!_K9Inside(w0, w1, w2)) {
            _K9Point const ab = _K9ClosestSegment(p, a, b);
            _K9Point const bc = _K9ClosestSegment(p, b, c);
            _K9Point const ca = _K9ClosestSegment(p, c, a);
            q = ab;
            auto nearer = [&]( _K9Point const &candidate) {
                float const du = candidate.u - p.u;
                float const dv = candidate.v - p.v;
                float const cu = q.u - p.u;
                float const cv = q.v - p.v;
                if (du * du + dv * dv < cu * cu + cv * cv) {
                    q = candidate;
                }
            };
            nearer(bc);
            nearer(ca);
            if (!_K9Bary(q, a, b, c, &w0, &w1, &w2)) {
                continue;
            }
        }
        // _K9Inside deliberately accepts a small negative boundary residue.
        // Canonical material lookup is stricter, so make that accepted root
        // point an exact convex barycentric tuple before carrying it forward.
        // A genuinely outside point still reaches this only through the
        // closest triangle edge above.
        if (!_K9Inside(w0, w1, w2) || !std::isfinite(w0) ||
            !std::isfinite(w1) || !std::isfinite(w2)) {
            continue;
        }
        w0 = std::max(0.0f, w0);
        w1 = std::max(0.0f, w1);
        w2 = std::max(0.0f, w2);
        float const weightSum = w0 + w1 + w2;
        if (!(weightSum > 0.0f) || !std::isfinite(weightSum)) {
            continue;
        }
        w0 /= weightSum;
        w1 /= weightSum;
        w2 /= weightSum;
        float const du = q.u - p.u, dv = q.v - p.v;
        float const d2 = du * du + dv * dv;
        if (d2 < best2) {
            best2 = d2;
            best.slot0 = tri[0]; best.slot1 = tri[1]; best.slot2 = tri[2];
            best.w0 = w0; best.w1 = w1; best.w2 = w2;
        }
    }
    if (!std::isfinite(best2)) {
        return false;
    }
    *out = best;
    return true;
}

bool _K9PolygonCenterRadius(std::vector<_K9Point> const &polygon,
                            _K9Point *center, float *radius)
{
    if (!center || !radius || polygon.size() < 3) {
        return false;
    }
    double area2 = 0.0, cu = 0.0, cv = 0.0;
    for (size_t i = 0; i < polygon.size(); ++i) {
        _K9Point const &a = polygon[i];
        _K9Point const &b = polygon[(i + 1) % polygon.size()];
        double const cross = double(a.u) * b.v - double(b.u) * a.v;
        area2 += cross;
        cu += (double(a.u) + b.u) * cross;
        cv += (double(a.v) + b.v) * cross;
    }
    if (std::fabs(area2) > 1e-12) {
        center->u = float(cu / (3.0 * area2));
        center->v = float(cv / (3.0 * area2));
    } else {
        center->u = center->v = 0.0f;
        for (_K9Point const &p : polygon) {
            center->u += p.u; center->v += p.v;
        }
        center->u /= float(polygon.size());
        center->v /= float(polygon.size());
    }
    double sum = 0.0;
    for (_K9Point const &p : polygon) {
        float const du = p.u - center->u, dv = p.v - center->v;
        sum += std::sqrt(double(du) * du + double(dv) * dv);
    }
    *radius = float(sum / double(polygon.size()));
    return *radius > 1e-12f;
}

}  // namespace

bool PomadeTriangulateSectionSlotsCpu(
    PomadeTubeSection const &section,
    std::vector<std::array<int, 3>> *triangles, std::string *err)
{
    if (!triangles || section.u.size() < 3 || section.u.size() != section.v.size()) {
        if (err) {
            *err = "PomadeTriangulateSectionSlotsCpu: malformed section";
        }
        return false;
    }
    std::vector<_K9Point> polygon(section.u.size());
    for (size_t i = 0; i < polygon.size(); ++i) {
        if (!std::isfinite(section.u[i]) || !std::isfinite(section.v[i])) {
            if (err) {
                *err = "PomadeTriangulateSectionSlotsCpu: non-finite CV";
            }
            return false;
        }
        polygon[i] = {section.u[i], section.v[i]};
    }
    if (!_K9Triangulate(polygon, triangles)) {
        if (err) {
            *err = "PomadeTriangulateSectionSlotsCpu: invalid polygon";
        }
        return false;
    }
    return true;
}

bool PomadeBuildGuideMaterialBindingsCpu(
    PomadeTubeDesc const &tube, std::vector<PomadeFrame> const &frames,
    std::vector<PomadeGuideRoot> const &roots, PomadeFillDesc const &fill,
    std::vector<PomadeGuideMaterialBinding> *bindings, std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    int const rv = tube.ringVerts;
    if (!bindings || tube.sections.size() < 2 || rv < 3 ||
        fill.cvCount < 2 || frames.size() != tube.centerX.size()) {
        return fail("PomadeBuildGuideMaterialBindingsCpu: malformed input");
    }
    if (fill.edgeBias < -1.0f || fill.edgeBias > 1.0f ||
        fill.lengthProfile.size() % 2 != 0) {
        return fail("PomadeBuildGuideMaterialBindingsCpu: invalid fill params");
    }
    PomadeTubeSection const &rootSection = tube.sections.front();
    if (int(rootSection.u.size()) != rv || int(rootSection.v.size()) != rv) {
        return fail("PomadeBuildGuideMaterialBindingsCpu: root ring mismatch");
    }
    float cp[3];
    PomadeFrame frame;
    std::string sampleErr;
    if (!PomadeSampleCenterCpu(tube, frames, rootSection.t, &cp[0], &cp[1],
                              &cp[2], &frame, &sampleErr)) {
        return fail(sampleErr.c_str());
    }
    float const ct = std::cos(rootSection.twist);
    float const st = std::sin(rootSection.twist);
    std::vector<_K9Point> polygon(static_cast<size_t>(rv));
    for (int i = 0; i < rv; ++i) {
        float const u = rootSection.u[size_t(i)] * rootSection.scale;
        float const v = rootSection.v[size_t(i)] * rootSection.scale;
        polygon[size_t(i)] = {u * ct - v * st, u * st + v * ct};
    }
    std::vector<std::array<int, 3>> triangles;
    std::string triangulateErr;
    if (!PomadeTriangulateSectionSlotsCpu(rootSection, &triangles,
                                         &triangulateErr)) {
        return fail(triangulateErr.c_str());
    }
    _K9Point polygonCenter;
    float polygonRadius = 0.0f;
    if (!_K9PolygonCenterRadius(polygon, &polygonCenter, &polygonRadius)) {
        return fail("PomadeBuildGuideMaterialBindingsCpu: degenerate root polygon");
    }
    bindings->assign(roots.size() * size_t(fill.cvCount), {});
    float const t0 = tube.sections.front().t;
    float const t1 = tube.sections.back().t;
    float const span = t1 > t0 ? t1 - t0 : 1.0f;
    // The usual full-length fill has identical stations for every root.
    // Cache those evaluated charts: their K5 sampling is otherwise repeated
    // for every guide before the same CPU/CUDA binding bytes are consumed.
    std::vector<std::vector<usdGen::UsdGenVec2f>> commonStationCharts;
    std::vector<std::vector<std::array<int, 3>>> commonStationTriangles;
    std::vector<bool> commonStationChartReady;
    std::vector<bool> commonStationChartCollapsed;
    if (fill.lengthProfile.empty()) {
        commonStationCharts.resize(size_t(fill.cvCount));
        commonStationTriangles.resize(size_t(fill.cvCount));
        commonStationChartReady.assign(size_t(fill.cvCount), false);
        commonStationChartCollapsed.assign(size_t(fill.cvCount), false);
    }
    auto evaluatedChart = [&](float t, int c, std::vector<usdGen::UsdGenVec2f> *chart,
                              std::vector<std::array<int, 3>> *chartTriangles,
                              bool *collapsed, std::string *chartErr) {
        if (!chart || !chartTriangles || !collapsed) {
            if (chartErr) *chartErr = "missing evaluated chart output";
            return false;
        }
        if (!commonStationCharts.empty() &&
            commonStationChartReady[size_t(c)]) {
            *chart = commonStationCharts[size_t(c)];
            *chartTriangles = commonStationTriangles[size_t(c)];
            *collapsed = commonStationChartCollapsed[size_t(c)];
            return true;
        }
        std::vector<float> ring;
        PomadeFrame currentFrame;
        float currentCenter[3] = {0.0f, 0.0f, 0.0f};
        std::string sampleError;
        if (!PomadeSampleTubeRingCpu(tube, frames, t, &ring, &sampleError) ||
            !PomadeSampleCenterCpu(tube, frames, t, &currentCenter[0],
                                  &currentCenter[1], &currentCenter[2],
                                  &currentFrame, &sampleError) ||
            ring.size() != size_t(rv) * 3) {
            if (chartErr) *chartErr = sampleError;
            return false;
        }
        chart->resize(size_t(rv));
        float maxRadius2 = 0.0f;
        float maxExtent2 = 0.0f;
        for (int slot = 0; slot < rv; ++slot) {
            size_t const at = size_t(slot) * 3;
            float const dx = ring[at + 0] - currentCenter[0];
            float const dy = ring[at + 1] - currentCenter[1];
            float const dz = ring[at + 2] - currentCenter[2];
            (*chart)[size_t(slot)] = usdGen::UsdGenVec2f(
                dx * currentFrame.nx + dy * currentFrame.ny +
                    dz * currentFrame.nz,
                dx * currentFrame.bx + dy * currentFrame.by +
                    dz * currentFrame.bz);
            float const u = (*chart)[size_t(slot)][0];
            float const v = (*chart)[size_t(slot)][1];
            maxRadius2 = std::max(maxRadius2, u * u + v * v);
            float const du = u - (*chart)[0][0];
            float const dv = v - (*chart)[0][1];
            maxExtent2 = std::max(maxExtent2, du * du + dv * dv);
        }
        // A complete point taper is a valid terminal material ring. Its
        // exact common point is emitted through slot 0; a merely collinear
        // or self-intersecting nonzero ring remains an invalid material
        // polygon and must not be hidden as a taper.
        float const epsilon = std::numeric_limits<float>::epsilon();
        *collapsed = maxExtent2 == 0.0f ||
            (maxRadius2 > 0.0f &&
             maxExtent2 <= 64.0f * epsilon * epsilon * maxRadius2);
        chartTriangles->clear();
        if (!*collapsed) {
            std::string triangulateError;
            if (!usdGen::UsdGenTriangulateConcaveMaterialSlots(
                    *chart, chartTriangles, &triangulateError)) {
                if (chartErr) *chartErr = triangulateError;
                return false;
            }
        }
        if (!commonStationCharts.empty()) {
            commonStationCharts[size_t(c)] = *chart;
            commonStationTriangles[size_t(c)] = *chartTriangles;
            commonStationChartReady[size_t(c)] = true;
            commonStationChartCollapsed[size_t(c)] = *collapsed;
        }
        return true;
    };
    float const biasExp = 1.0f - 0.5f * fill.edgeBias;
    for (size_t g = 0; g < roots.size(); ++g) {
        PomadeGuideRoot const &root = roots[g];
        _K9Point const source = {
            (root.px - cp[0]) * frame.nx + (root.py - cp[1]) * frame.ny +
                (root.pz - cp[2]) * frame.nz,
            (root.px - cp[0]) * frame.bx + (root.py - cp[1]) * frame.by +
                (root.pz - cp[2]) * frame.bz};
        PomadeGuideMaterialBinding base;
        if (!_K9Bind(polygon, triangles, source, &base)) {
            return fail("PomadeBuildGuideMaterialBindingsCpu: root bind failed");
        }
        _K9Point const rail = {
            polygon[size_t(base.slot0)].u * base.w0 +
                polygon[size_t(base.slot1)].u * base.w1 +
                polygon[size_t(base.slot2)].u * base.w2,
            polygon[size_t(base.slot0)].v * base.w0 +
                polygon[size_t(base.slot1)].v * base.w1 +
                polygon[size_t(base.slot2)].v * base.w2};
        float const railWorld[3] = {
            cp[0] + frame.nx * rail.u + frame.bx * rail.v,
            cp[1] + frame.ny * rail.u + frame.by * rail.v,
            cp[2] + frame.nz * rail.u + frame.bz * rail.v};
        float const distance = std::sqrt(
            (source.u - polygonCenter.u) * (source.u - polygonCenter.u) +
            (source.v - polygonCenter.v) * (source.v - polygonCenter.v));
        float r = distance / polygonRadius;
        r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
        float const rb = std::pow(r, biasExp);
        float const minW = std::min(base.w0, std::min(base.w1, base.w2));
        float const rho = std::max(0.0f, 1.0f - 3.0f * minW);
        float const rhoB = std::pow(rho, biasExp);
        float const factor = rho > 1e-8f ? rhoB / rho : 0.0f;
        float const targetW0 = 1.0f / 3.0f + (base.w0 - 1.0f / 3.0f) * factor;
        float const targetW1 = 1.0f / 3.0f + (base.w1 - 1.0f / 3.0f) * factor;
        float const targetW2 = 1.0f / 3.0f + (base.w2 - 1.0f / 3.0f) * factor;
        int const pairCount = int(fill.lengthProfile.size() / 2);
        float length = PomadeEvalLengthProfile(fill.lengthProfile.data(),
                                              pairCount, rb);
        length = length < 0.0f ? 0.0f : (length > 1.0f ? 1.0f : length);
        for (int c = 0; c < fill.cvCount; ++c) {
            float const s = float(c) / float(fill.cvCount - 1);
            PomadeGuideMaterialBinding binding = base;
            // Guide length is evaluated by K9 after this bind.  Use the
            // geometric progress here; an abbreviated guide simply stops at
            // its sampled tube t, preserving a continuous material path.
            float const progress = s * length;
            binding.w0 += (targetW0 - base.w0) * progress;
            binding.w1 += (targetW1 - base.w1) * progress;
            binding.w2 += (targetW2 - base.w2) * progress;
            // The root bind is expressed in its actual section-0 triangle.
            // At later K5 stations, remap that biased material point through
            // a canonical regular polygon into the evaluated section. This
            // preserves slot identity without letting a concave ring turn a
            // root-triangle affine interpolation outside its boundary.
            if (c != 0) {
                double const step = 2.0 * std::acos(-1.0) / double(rv);
                auto canonicalSlot = [&](int slot) {
                    double const angle = step * double(slot);
                    return usdGen::UsdGenVec2f(float(std::cos(angle)),
                                           float(std::sin(angle)));
                };
                usdGen::UsdGenVec2f const p0 = canonicalSlot(binding.slot0);
                usdGen::UsdGenVec2f const p1 = canonicalSlot(binding.slot1);
                usdGen::UsdGenVec2f const p2 = canonicalSlot(binding.slot2);
                usdGen::UsdGenVec2f const canonicalPoint =
                    p0 * binding.w0 + p1 * binding.w1 + p2 * binding.w2;
                float const t = t0 + span * progress;
                std::vector<usdGen::UsdGenVec2f> chart;
                std::vector<std::array<int, 3>> chartTriangles;
                bool collapsed = false;
                std::string chartError;
                if (!evaluatedChart(t, c, &chart, &chartTriangles, &collapsed,
                                    &chartError)) {
                    if (err) {
                        *err = "PomadeBuildGuideMaterialBindingsCpu: " +
                            chartError;
                    }
                    return false;
                }
                if (collapsed) {
                    binding.slot0 = binding.slot1 = binding.slot2 = 0;
                    binding.w0 = 1.0f;
                    binding.w1 = binding.w2 = 0.0f;
                } else {
                    std::array<int, 3> slots;
                    usdGen::UsdGenVec3f weights;
                    std::string remapError;
                    if (!usdGen::UsdGenLocateConcaveMaterialPoint(
                            chart.size(), chartTriangles, canonicalPoint,
                            &slots, &weights, &remapError)) {
                        if (err) {
                            *err = "PomadeBuildGuideMaterialBindingsCpu: " +
                                remapError;
                        }
                        return false;
                    }
                    binding.slot0 = slots[0];
                    binding.slot1 = slots[1];
                    binding.slot2 = slots[2];
                    binding.w0 = weights[0];
                    binding.w1 = weights[1];
                    binding.w2 = weights[2];
                }
            }
            binding.edgeRadius = rb;
            binding.rootDx = root.px - railWorld[0];
            binding.rootDy = root.py - railWorld[1];
            binding.rootDz = root.pz - railWorld[2];
            (*bindings)[g * size_t(fill.cvCount) + size_t(c)] = binding;
        }
    }
    return true;
}

namespace {

bool _PomadeGuideFillLegacyCpu(PomadeTubeDesc const &tube,
                       std::vector<PomadeFrame> const &frames,
                       std::vector<PomadeGuideRoot> const &roots,
                       PomadeFillDesc const &fill, std::vector<float> *points,
                       std::vector<float> *lengthScales, std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    int const nCv = int(tube.centerX.size());
    int const nSec = int(tube.sections.size());
    if (nCv < 2 || nSec < 2 || frames.size() != size_t(nCv) || !points ||
        !lengthScales || fill.cvCount < 2 || fill.cvCount > 64) {
        return fail("PomadeGuideFillCpu: bad tube, frames or cvCount");
    }
    if (fill.edgeBias < -1.0f || fill.edgeBias > 1.0f) {
        return fail("PomadeGuideFillCpu: edgeBias must be in [-1, 1]");
    }
    if (fill.lengthProfile.size() % 2 != 0) {
        return fail("PomadeGuideFillCpu: lengthProfile must hold pairs");
    }
    int const guideCount = int(roots.size());
    int const cvCount = fill.cvCount;
    points->assign(size_t(guideCount) * size_t(cvCount) * 3, 0.0f);
    lengthScales->assign(size_t(guideCount), 1.0f);
    if (guideCount == 0) {
        return true;
    }
    float const rootRadius = PomadeSectionMeanRadius(tube.sections.front());
    if (!(rootRadius > 0.0f)) {
        return fail("PomadeGuideFillCpu: degenerate root section");
    }
    float const t0 = tube.sections.front().t;
    float const t1 = tube.sections.back().t;
    float const span = t1 > t0 ? t1 - t0 : 1.0f;
    float const biasExp = 1.0f - 0.5f * fill.edgeBias;
    int const pairCount = int(fill.lengthProfile.size() / 2);
    for (int g = 0; g < guideCount; ++g) {
        PomadeGuideRoot const &root = roots[size_t(g)];
        float r = std::sqrt(root.ru * root.ru + root.rv * root.rv);
        float th = std::atan2(root.rv, root.ru);
        if (r > 1.0f) {
            r = 1.0f;
        }
        float const rb = std::pow(r, biasExp);
        float const ru = rb * std::cos(th), rv = rb * std::sin(th);
        float length = PomadeEvalLengthProfile(fill.lengthProfile.data(),
                                              pairCount, rb);
        if (!(length > 0.0f)) {
            length = 0.0f;
        }
        if (length > 1.0f) {
            length = 1.0f;
        }
        (*lengthScales)[size_t(g)] = length;
        for (int c = 0; c < cvCount; ++c) {
            float const s = float(c) / float(cvCount - 1);
            float const t = t0 + span * s * length;
            float cp[3];
            PomadeFrame fr;
            std::string derr;
            if (!PomadeSampleCenterCpu(tube, frames, t, &cp[0], &cp[1], &cp[2],
                                     &fr, &derr)) {
                return fail(derr.c_str());
            }
            // The root offset rides up the tube through the section
            // interpolation: scale by the local mean radius so guides
            // expand and contract with the tube.
            int k = 0;
            while (k + 1 < nSec - 1 && tube.sections[size_t(k + 1)].t < t) {
                ++k;
            }
            PomadeTubeSection const &s0 = tube.sections[size_t(k)];
            PomadeTubeSection const &s1 = tube.sections[size_t(k + 1)];
            float f = (s1.t > s0.t) ? (t - s0.t) / (s1.t - s0.t) : 0.0f;
            f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
            float const r0 = PomadeSectionMeanRadius(s0);
            float const r1 = PomadeSectionMeanRadius(s1);
            float const rr = (r0 + (r1 - r0) * f) / rootRadius;
            size_t const o = (size_t(g) * size_t(cvCount) + size_t(c)) * 3;
            (*points)[o + 0] =
                cp[0] + (fr.nx * ru + fr.bx * rv) * rootRadius * rr;
            (*points)[o + 1] =
                cp[1] + (fr.ny * ru + fr.by * rv) * rootRadius * rr;
            (*points)[o + 2] =
                cp[2] + (fr.nz * ru + fr.bz * rv) * rootRadius * rr;
        }
    }
    return true;
}

}  // namespace

bool PomadeGuideFillCpu(PomadeTubeDesc const &tube,
                       std::vector<PomadeFrame> const &frames,
                       std::vector<PomadeGuideRoot> const &roots,
                       PomadeFillDesc const &fill, std::vector<float> *points,
                       std::vector<float> *lengthScales, std::string *err)
{
    if (fill.sampler == PomadeGuideSampler::Legacy) {
        return _PomadeGuideFillLegacyCpu(tube, frames, roots, fill, points,
                                        lengthScales, err);
    }
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    int const nCv = int(tube.centerX.size());
    int const nSec = int(tube.sections.size());
    int const cvCount = fill.cvCount;
    if (nCv < 2 || nSec < 2 || frames.size() != size_t(nCv) || !points ||
        !lengthScales || cvCount < 2 || cvCount > 64) {
        return fail("PomadeGuideFillCpu: bad tube, frames or cvCount");
    }
    if (fill.edgeBias < -1.0f || fill.edgeBias > 1.0f) {
        return fail("PomadeGuideFillCpu: edgeBias must be in [-1, 1]");
    }
    if (fill.lengthProfile.size() % 2 != 0) {
        return fail("PomadeGuideFillCpu: lengthProfile must hold pairs");
    }
    int const guideCount = int(roots.size());
    points->assign(size_t(guideCount) * size_t(cvCount) * 3, 0.0f);
    lengthScales->assign(size_t(guideCount), 1.0f);
    if (guideCount == 0) {
        return true;
    }
    std::vector<PomadeGuideMaterialBinding> bindings;
    std::string bindErr;
    if (!PomadeBuildGuideMaterialBindingsCpu(tube, frames, roots, fill,
                                            &bindings, &bindErr)) {
        return fail(bindErr.c_str());
    }
    float const t0 = tube.sections.front().t;
    float const t1 = tube.sections.back().t;
    float const span = t1 > t0 ? t1 - t0 : 1.0f;
    int const pairCount = int(fill.lengthProfile.size() / 2);
    for (int g = 0; g < guideCount; ++g) {
        float length = PomadeEvalLengthProfile(
            fill.lengthProfile.data(), pairCount,
            bindings[size_t(g) * size_t(cvCount)].edgeRadius);
        length = length < 0.0f ? 0.0f : (length > 1.0f ? 1.0f : length);
        (*lengthScales)[size_t(g)] = length;
        for (int c = 0; c < cvCount; ++c) {
            size_t const out =
                (size_t(g) * size_t(cvCount) + size_t(c)) * 3;
            if (c == 0) {
                // Root attachment is physical, never the clamped material
                // binding used by later samples.
                (*points)[out + 0] = roots[size_t(g)].px;
                (*points)[out + 1] = roots[size_t(g)].py;
                (*points)[out + 2] = roots[size_t(g)].pz;
                continue;
            }
            float const s = float(c) / float(cvCount - 1);
            float const t = t0 + span * s * length;
            std::vector<float> ring;
            std::string ringErr;
            if (!PomadeSampleTubeRingCpu(tube, frames, t, &ring, &ringErr)) {
                return fail(ringErr.c_str());
            }
            PomadeGuideMaterialBinding const &binding =
                bindings[size_t(g) * size_t(cvCount) + size_t(c)];
            if (binding.slot0 < 0 || binding.slot1 < 0 || binding.slot2 < 0 ||
                binding.slot0 >= tube.ringVerts || binding.slot1 >= tube.ringVerts ||
                binding.slot2 >= tube.ringVerts) {
                return fail("PomadeGuideFillCpu: invalid material binding");
            }
            size_t const a = size_t(binding.slot0) * 3;
            size_t const b = size_t(binding.slot1) * 3;
            size_t const d = size_t(binding.slot2) * 3;
            for (int axis = 0; axis < 3; ++axis) {
                (*points)[out + size_t(axis)] =
                    ring[a + size_t(axis)] * binding.w0 +
                    ring[b + size_t(axis)] * binding.w1 +
                    ring[d + size_t(axis)] * binding.w2;
            }
            float const progress = s * length;
            float const remain = 1.0f - progress;
            (*points)[out + 0] += binding.rootDx * remain;
            (*points)[out + 1] += binding.rootDy * remain;
            (*points)[out + 2] += binding.rootDz * remain;
        }
    }
    return true;
}

bool PomadeGuideResampleCpu(float const *points, int const *counts,
                           int guideCount, int cvCount,
                           float const rootDirs[9], std::vector<float> *out,
                           std::vector<int> *outCounts,
                           std::vector<double> *outFrames, std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    if (!points || !counts || !rootDirs || !out || !outCounts || !outFrames ||
        guideCount < 0 || cvCount < 2 || cvCount > 64) {
        return fail("PomadeGuideResampleCpu: bad counts or buffers");
    }
    out->assign(size_t(guideCount) * size_t(cvCount) * 3, 0.0f);
    outCounts->assign(size_t(guideCount), cvCount);
    outFrames->assign(size_t(guideCount) * 16, 0.0);
    size_t cursor = 0;
    for (int g = 0; g < guideCount; ++g) {
        int const n = counts[g];
        if (n < 2) {
            return fail("PomadeGuideResampleCpu: guide needs >= 2 CVs");
        }
        float const *src = points + cursor * 3;
        cursor += size_t(n);
        std::vector<float> cum(size_t(n), 0.0f);
        for (int i = 1; i < n; ++i) {
            float dx = src[size_t(i) * 3 + 0] - src[size_t(i - 1) * 3 + 0];
            float dy = src[size_t(i) * 3 + 1] - src[size_t(i - 1) * 3 + 1];
            float dz = src[size_t(i) * 3 + 2] - src[size_t(i - 1) * 3 + 2];
            cum[size_t(i)] =
                cum[size_t(i - 1)] + std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        float const total = cum.back();
        for (int c = 0; c < cvCount; ++c) {
            float const s = float(c) / float(cvCount - 1) * total;
            int seg = 0;
            while (seg + 1 < n - 1 && cum[size_t(seg + 1)] < s) {
                ++seg;
            }
            float const c0 = cum[size_t(seg)], c1 = cum[size_t(seg + 1)];
            float f = (c1 > c0) ? (s - c0) / (c1 - c0) : 0.0f;
            f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
            size_t const o = (size_t(g) * size_t(cvCount) + size_t(c)) * 3;
            (*out)[o + 0] = src[size_t(seg) * 3 + 0] +
                             (src[size_t(seg + 1) * 3 + 0] -
                              src[size_t(seg) * 3 + 0]) *
                                 f;
            (*out)[o + 1] = src[size_t(seg) * 3 + 1] +
                             (src[size_t(seg + 1) * 3 + 1] -
                              src[size_t(seg) * 3 + 1]) *
                                 f;
            (*out)[o + 2] = src[size_t(seg) * 3 + 2] +
                             (src[size_t(seg + 1) * 3 + 2] -
                              src[size_t(seg) * 3 + 2]) *
                                 f;
        }
        // Root frame (UsdGenCurveAPI row-major): rows 0/1/2 are the tube
        // root N/B/T directions, row 3 is the root position.
        size_t const f = size_t(g) * 16;
        (*outFrames)[f + 0] = rootDirs[0];
        (*outFrames)[f + 1] = rootDirs[1];
        (*outFrames)[f + 2] = rootDirs[2];
        (*outFrames)[f + 3] = 0.0;
        (*outFrames)[f + 4] = rootDirs[3];
        (*outFrames)[f + 5] = rootDirs[4];
        (*outFrames)[f + 6] = rootDirs[5];
        (*outFrames)[f + 7] = 0.0;
        (*outFrames)[f + 8] = rootDirs[6];
        (*outFrames)[f + 9] = rootDirs[7];
        (*outFrames)[f + 10] = rootDirs[8];
        (*outFrames)[f + 11] = 0.0;
        (*outFrames)[f + 12] = src[0];
        (*outFrames)[f + 13] = src[1];
        (*outFrames)[f + 14] = src[2];
        (*outFrames)[f + 15] = 1.0;
    }
    return true;
}

PomadePickHit PomadePickCpu(PomadePickSets const &sets, uint32_t kindMask,
                          float const viewProj[16], int w, int h, float x,
                          float y, float radiusPx)
{
    PomadePickHit best;
    if (!viewProj || w <= 0 || h <= 0 || !(radiusPx >= 0.0f)) {
        return best;
    }
    float const r2 = radiusPx * radiusPx;
    auto consider = [&](uint32_t kind, float const *p, int index,
                        int subIndex) {
        if (!(kindMask & kind) || !p) {
            return;
        }
        float px, py, ndcZ;
        if (!PomadeProjectPoint(p, viewProj, w, h, &px, &py, &ndcZ)) {
            return;
        }
        float const dx = px - x, dy = py - y;
        float const d2 = dx * dx + dy * dy;
        if (d2 > r2) {
            return;
        }
        float const dist = std::sqrt(d2);
        if (!best.hit || dist < best.distPx ||
            (dist == best.distPx && ndcZ < best.depth)) {
            best.hit = true;
            best.kind = kind;
            best.index = index;
            best.subIndex = subIndex;
            best.distPx = dist;
            best.depth = ndcZ;
        }
    };
    for (int i = 0; i < sets.tubeVertCount; ++i) {
        consider(PomadePick_TubeVert, sets.tubeVerts + size_t(i) * 3, i, -1);
    }
    for (int i = 0; i < sets.centerCVCount; ++i) {
        consider(PomadePick_CenterCV, sets.centerCVs + size_t(i) * 3, i, -1);
    }
    for (int i = 0; i < sets.sectionCVCount; ++i) {
        int sub = -1;
        if (sets.sectionRingVerts > 0) {
            sub = i % sets.sectionRingVerts;
            consider(PomadePick_SectionCV, sets.sectionCVs + size_t(i) * 3,
                     i / sets.sectionRingVerts, sub);
        } else {
            consider(PomadePick_SectionCV, sets.sectionCVs + size_t(i) * 3, i,
                     -1);
        }
    }
    for (int i = 0; i < sets.graphNodeCount; ++i) {
        consider(PomadePick_GraphNode, sets.graphNodes + size_t(i) * 3, i, -1);
    }
    for (int g = 0; g < sets.guideCount; ++g) {
        for (int c = 0; c < sets.guideCvCount; ++c) {
            consider(PomadePick_Guide,
                     sets.guideCVs +
                         (size_t(g) * size_t(sets.guideCvCount) + size_t(c)) *
                             3,
                     g, c);
        }
    }
    return best;
}

}  // namespace usdGenPomade
