// testUsdGenPomadeTubes -- T0: P3 tubes + fill (plan/17 K4/K5/K8-K11, §5.2/§5.3).
//
// No Hydra, no stage, no files: fixtures are built in code (a straight
// 3-CV tube with circle sections, a 4x4 quad grid scalp).
// Proven here:
//   * K4: straight tube frames (T=+Y, orthonormal), bent tube frames vary
//     smoothly and stay orthonormal, coincident CVs keep the normal;
//   * K5: section rings land on their circles, twist rotates, scale
//     scales, Hermite mid-spans interpolate, ring counts follow the span
//     rule, degenerate input fails loudly;
//   * K8: disc counts are exact, roots sit in the unit disc, the stream
//     is deterministic per (tubeId, seed), freeze keeps the prefix;
//   * K9: root CVs sit on their roots, edgeBias pushes/pulls the radius,
//     lengthProfile shortens guides, bias/profile validation fails;
//   * K10: straight guides resample evenly, root frames carry the root
//     dirs + root position, degenerate input fails;
//   * K11: hits/misses/kind masks/depth tie-breaks on a known projection;
//   * the model: the legacy test-tube positions survive untouched, center
//     CV ops (move/soft/insert/delete/length/match), section ops
//     (move/scale/twist/per-CV/add/remove/copy), soft selection,
//     relax, root snap, display segments, region rooting;
//   * Fill mode: param validation, preview/full refills, freeze roots,
//     guide + root census reads;
//   * Per-tube fill: subdivided parents suspend, leaves fill from their
//     own params into an ascending merged set with per-guide tube
//     attribution, freeze is per tube, merge resumes the parent;
//   * auto-tube from a graph region (root fit + normal seeding);
//   * the P3 C ABI drives the same tube (plus its error paths);
//   * TN-6 parity (CUDA builds with a device only): all six device lanes
//     match their CPU twins (bit-exact where arithmetic-only).
// TN-1/TN-2 probes print measured move/pick times at scale (informational:
// the gates run on the reference scene, not this fixture).

#include "usdGenPomade/pomadeApi.h"
#include "usdGenPomade/pomadeApiStage.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeTube.h"

#ifdef USDGEN_POMADE_HAS_CUDA
#include "usdGenPomade/pomadeKernels.h"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
    std::fflush(stdout);
}

bool Near(float a, float b, float eps = 1e-4f)
{
    return std::fabs(a - b) <= eps;
}

bool NearD(double a, double b, double eps = 1e-9)
{
    return std::fabs(a - b) <= eps;
}

using usdGenPomade::PomadeFrame;
using usdGenPomade::PomadeRigidTransform;
using usdGenPomade::PomadeTubeDesc;
using usdGenPomade::PomadeTubeSection;

// A straight 3-CV tube along +Y (length 4) with two circle sections
// (R = 0.5, 8 CVs) at t = 0 and t = 1.
PomadeTubeDesc MakeTube(int ringVerts = 8)
{
    PomadeTubeDesc tube;
    tube.centerX = {0.0f, 0.0f, 0.0f};
    tube.centerY = {0.0f, 2.0f, 4.0f};
    tube.centerZ = {0.0f, 0.0f, 0.0f};
    tube.ringVerts = ringVerts;
    float const twoPi = 6.28318530717958647692f;
    for (int k = 0; k < 2; ++k) {
        PomadeTubeSection s;
        s.t = float(k);
        s.u.resize(size_t(ringVerts));
        s.v.resize(size_t(ringVerts));
        for (int i = 0; i < ringVerts; ++i) {
            float const a = twoPi * float(i) / float(ringVerts);
            s.u[size_t(i)] = 0.5f * std::cos(a);
            s.v[size_t(i)] = 0.5f * std::sin(a);
        }
        tube.sections.push_back(std::move(s));
    }
    return tube;
}

std::vector<PomadeFrame> FramesFor(PomadeTubeDesc const &tube)
{
    using namespace usdGenPomade;
    std::vector<PomadeFrame> frames;
    std::string err;
    if (!PomadeCenterFramesCpu(tube.centerX.data(), tube.centerY.data(),
                              tube.centerZ.data(), int(tube.centerX.size()),
                              &frames, &err)) {
        Check(false, std::string("fixture frames: ") + err);
    }
    return frames;
}

bool SectionSlotWorld(PomadeTubeDesc const &tube, int section, int slot,
                      float out[3])
{
    using namespace usdGenPomade;
    if (!out || section < 0 || section >= int(tube.sections.size())) {
        return false;
    }
    PomadeTubeSection const &ring = tube.sections[size_t(section)];
    if (slot < 0 || slot >= int(ring.u.size()) ||
        ring.u.size() != ring.v.size()) {
        return false;
    }
    std::vector<PomadeFrame> frames;
    std::string error;
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    PomadeFrame frame;
    if (!PomadeTubeFramesCpu(tube, &frames, &error) ||
        !PomadeSampleCenterCpu(tube, frames, ring.t, &cx, &cy, &cz, &frame,
                              &error)) {
        return false;
    }
    float const u = ring.u[size_t(slot)] * ring.scale;
    float const v = ring.v[size_t(slot)] * ring.scale;
    float const ct = std::cos(ring.twist), st = std::sin(ring.twist);
    float const ru = u * ct - v * st;
    float const rv = u * st + v * ct;
    out[0] = cx + frame.nx * ru + frame.bx * rv;
    out[1] = cy + frame.ny * ru + frame.by * rv;
    out[2] = cz + frame.nz * ru + frame.bz * rv;
    return true;
}

bool SameSections(std::vector<PomadeTubeSection> const &a,
                  std::vector<PomadeTubeSection> const &b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].t != b[i].t || a[i].u != b[i].u || a[i].v != b[i].v ||
            a[i].scale != b[i].scale || a[i].twist != b[i].twist) {
            return false;
        }
    }
    return true;
}

bool Orthonormal(PomadeFrame const &f, float eps = 1e-5f)
{
    float const lt =
        std::sqrt(f.tx * f.tx + f.ty * f.ty + f.tz * f.tz);
    float const ln =
        std::sqrt(f.nx * f.nx + f.ny * f.ny + f.nz * f.nz);
    float const lb =
        std::sqrt(f.bx * f.bx + f.by * f.by + f.bz * f.bz);
    float const tn = f.tx * f.nx + f.ty * f.ny + f.tz * f.nz;
    float const tb = f.tx * f.bx + f.ty * f.by + f.tz * f.bz;
    float const nb = f.nx * f.bx + f.ny * f.by + f.nz * f.bz;
    return Near(lt, 1.0f, eps) && Near(ln, 1.0f, eps) &&
           Near(lb, 1.0f, eps) && Near(tn, 0.0f, eps) &&
           Near(tb, 0.0f, eps) && Near(nb, 0.0f, eps);
}

void CheckK4()
{
    using namespace usdGenPomade;
    PomadeTubeDesc const tube = MakeTube();
    std::vector<PomadeFrame> frames = FramesFor(tube);
    Check(frames.size() == 3, "K4: one frame per center CV");
    bool tangentY = true, ortho = true;
    for (auto const &f : frames) {
        tangentY = tangentY && Near(f.tx, 0.0f) && Near(f.ty, 1.0f) &&
                   Near(f.tz, 0.0f);
        ortho = ortho && Orthonormal(f);
    }
    Check(tangentY, "K4: straight-tube tangents are +Y");
    Check(ortho, "K4: straight-tube frames are orthonormal");
    // The RMF of a straight line is twist-free: all normals agree.
    bool sameN = Near(frames[0].nx, frames[1].nx, 1e-6f) &&
                 Near(frames[0].ny, frames[1].ny, 1e-6f) &&
                 Near(frames[0].nz, frames[1].nz, 1e-6f) &&
                 Near(frames[1].nx, frames[2].nx, 1e-6f);
    Check(sameN, "K4: straight-tube normals do not twist");
    // A bent tube: frames follow the bend, stay orthonormal.
    PomadeTubeDesc bent = tube;
    bent.centerX[2] = 2.0f;
    std::vector<PomadeFrame> bf = FramesFor(bent);
    bool bOrtho = bf.size() == 3;
    for (auto const &f : bf) {
        bOrtho = bOrtho && Orthonormal(f);
    }
    Check(bOrtho, "K4: bent-tube frames stay orthonormal");
    Check(bf[2].tx > 0.2f, "K4: bent-tube tip tangent follows the bend");
    // Catmull-Rom reproduces the straight line at any t.
    float px = 0, py = 0, pz = 0;
    PomadeFrame fr;
    std::string err;
    bool ok = PomadeSampleCenterCpu(tube, frames, 0.37f, &px, &py, &pz, &fr,
                                   &err);
    Check(ok && Near(px, 0.0f, 1e-5f) && Near(py, 0.37f * 4.0f, 1e-4f) &&
              Near(pz, 0.0f, 1e-5f) && Orthonormal(fr),
          "K4: center sampling reproduces the straight line");
    // Coincident CVs keep the previous normal (no NaN).
    PomadeTubeDesc pinch = tube;
    pinch.centerX[1] = pinch.centerX[0];
    pinch.centerY[1] = pinch.centerY[0];
    pinch.centerZ[1] = pinch.centerZ[0];
    std::vector<PomadeFrame> pf = FramesFor(pinch);
    bool finite = pf.size() == 3;
    for (auto const &f : pf) {
        finite = finite && std::isfinite(f.nx + f.ny + f.nz) && Orthonormal(f);
    }
    Check(finite, "K4: coincident CVs keep a finite orthonormal frame");
    Check(!PomadeCenterFramesCpu(nullptr, nullptr, nullptr, 0, nullptr,
                                nullptr),
          "K4: null input fails");
}

void CheckK5()
{
    using namespace usdGenPomade;
    PomadeTubeDesc const tube = MakeTube();
    std::vector<PomadeFrame> frames = FramesFor(tube);
    std::vector<float> pos, nrm, ringT;
    std::string err;
    bool ok = PomadeTessellateCpu(tube, frames, 1, &pos, &nrm, &ringT, &err);
    Check(ok, "K5: straight tube tessellates");
    Check(pos.size() == 2 * 8 * 3 && nrm.size() == pos.size() &&
              ringT.size() == 2,
          "K5: (nSec-1)*seg+1 rings x ringVerts verts");
    // Ring 0 sits in the y=0 plane on the R=0.5 circle; normals are radial.
    bool ring = true, radial = true;
    for (int s = 0; s < 8; ++s) {
        float const x = pos[size_t(s) * 3 + 0];
        float const y = pos[size_t(s) * 3 + 1];
        float const z = pos[size_t(s) * 3 + 2];
        ring = ring && Near(y, 0.0f, 1e-5f) &&
               Near(std::sqrt(x * x + z * z), 0.5f, 1e-4f);
        float const nx = nrm[size_t(s) * 3 + 0];
        float const ny = nrm[size_t(s) * 3 + 1];
        float const nz = nrm[size_t(s) * 3 + 2];
        float const rl = std::sqrt(nx * nx + nz * nz);
        radial = radial && Near(ny, 0.0f, 1e-4f) && Near(rl, 1.0f, 1e-4f) &&
                 Near(nx / rl, x / 0.5f, 1e-4f);
    }
    Check(ring, "K5: root ring lands on the section circle");
    Check(radial, "K5: normals are radial");
    Check(Near(ringT[0], 0.0f) && Near(ringT[1], 1.0f),
          "K5: ring parameters span the section range");
    // Twist rotates the ring; scale scales it.
    PomadeTubeDesc twisted = tube;
    twisted.sections[1].twist = 1.57079632679f;
    twisted.sections[1].scale = 2.0f;
    std::vector<float> p2, n2, t2;
    Check(PomadeTessellateCpu(twisted, frames, 1, &p2, &n2, &t2, &err),
          "K5: twisted/scaled tip tessellates");
    bool tipR = true;
    for (int s = 0; s < 8; ++s) {
        float const x = p2[(8 + size_t(s)) * 3 + 0];
        float const z = p2[(8 + size_t(s)) * 3 + 2];
        tipR = tipR && Near(std::sqrt(x * x + z * z), 1.0f, 1e-3f);
    }
    Check(tipR, "K5: scale doubles the tip radius");
    // Slot 0 starts at (u, v) = (0.5, 0); a pi/2 twist moves it to the
    // frame-B axis (x ~= 0, |z| ~= 1 after scaling).
    float const tx = p2[8 * 3 + 0];
    float const tz = p2[8 * 3 + 2];
    Check(Near(tx, 0.0f, 1e-3f) && Near(std::fabs(tz), 1.0f, 1e-3f),
          "K5: twist rotates the tip ring by pi/2");
    // Mid-span Hermite between different radii interpolates.
    PomadeTubeDesc taper = tube;
    taper.sections[1].scale = 0.5f;
    std::vector<float> p3, n3, t3;
    Check(PomadeTessellateCpu(taper, frames, 2, &p3, &n3, &t3, &err),
          "K5: two spans tessellate");
    Check(p3.size() == 3 * 8 * 3, "K5: segments double the ring count");
    float const mx = p3[8 * 3 + 0];
    float const mz = p3[8 * 3 + 2];
    float const mid = std::sqrt(mx * mx + mz * mz);
    Check(mid > 0.25f && mid < 0.5f, "K5: mid-span radius interpolates");
    // Degenerate input fails loudly.
    PomadeTubeDesc bad = tube;
    bad.sections[0].u.pop_back();
    Check(!PomadeTessellateCpu(bad, frames, 1, &p3, &n3, &t3, &err),
          "K5: ragged rings fail");
}

void CheckSparseProfile()
{
    using namespace usdGenPomade;
    PomadeTubeDesc tube = MakeTube();
    auto const section = tube.sections.front();
    tube.sections.assign(4, section);
    for (int i = 0; i < 4; ++i) {
        tube.sections[size_t(i)].t = float(i) / 3.0f;
        tube.sections[size_t(i)].scale = i % 2 ? 0.3f : 1.0f;
    }
    auto const frames = FramesFor(tube);
    std::vector<float> p, n, t;
    std::string err;
    int const spans = 8;
    Check(PomadeTessellateCpu(tube, frames, spans, &p, &n, &t, &err),
          "profile: wide/thin/wide/thin tessellates");
    auto radius = [&](int row) {
        size_t const o = size_t(row * tube.ringVerts) * 3;
        return std::sqrt(p[o] * p[o] + p[o + 2] * p[o + 2]);
    };
    Check(t.size() == 25 && Near(radius(0), 0.5f) &&
              Near(radius(8), 0.15f) && Near(radius(16), 0.5f) &&
              Near(radius(24), 0.15f),
          "profile: four sparse rings are exact, with eight segments per span");
    Check(Near(radius(1), radius(0)) && Near(radius(23), radius(24)) &&
              t[1] < 0.02f && t[23] > 0.98f,
          "profile: close holding edges retain the root and tip footprints");
    Check(Near(radius(10), 0.15f + 0.35f * 0.15625f) &&
              Near(radius(14), 0.15f + 0.35f * 0.84375f) &&
              radius(9) - radius(8) < radius(12) - radius(11) &&
              radius(16) - radius(15) < radius(12) - radius(11),
          "profile: the inner span eases out of the pinch and into the bulge");
    tube.sections[1].t = 0.17f;
    tube.sections[2].t = 0.71f;
    Check(PomadeTessellateCpu(tube, frames, spans, &p, &n, &t, &err),
          "profile: uneven sparse knots tessellate");
    bool exact = true;
    for (int i = 0; i < 4; ++i) {
        float const ti = tube.sections[size_t(i)].t;
        exact &= Near(t[size_t(i * spans)], ti) &&
                 Near(p[size_t(i * spans * tube.ringVerts) * 3 + 1], 4 * ti) &&
                 Near(radius(i * spans), i % 2 ? 0.15f : 0.5f);
    }
    Check(exact, "profile: uneven control rings remain on their surface rows");
}

void CheckCenterCoreHandle()
{
    using namespace usdGenPomade;
    // An off-centre cross section models the clipped/asymmetric child case:
    // the authored K4 center stays on the cage, while the artist's handle
    // must be at the area-centred visible core.
    PomadeTubeDesc tube = MakeTube();
    for (PomadeTubeSection &section : tube.sections) {
        for (size_t slot = 0; slot < section.u.size(); ++slot) {
            section.u[slot] += 0.37f;
            section.v[slot] -= 0.19f;
        }
    }
    std::vector<PomadeFrame> frames = FramesFor(tube);
    float raw[3] = {0.0f, 0.0f, 0.0f};
    PomadeFrame frame;
    std::string err;
    Check(PomadeSampleCenterCpu(tube, frames, 0.5f, &raw[0], &raw[1],
                               &raw[2], &frame, &err),
          "core handle: sample the authored midpoint");
    float handle[3] = {0.0f, 0.0f, 0.0f};
    Check(PomadeCenterHandlePointCpu(tube, frames, 1, &handle[0], &handle[1],
                                    &handle[2], &err),
          "core handle: off-centre section resolves with reused frames");
    float const expected[3] = {
        raw[0] + frame.nx * 0.37f + frame.bx * -0.19f,
        raw[1] + frame.ny * 0.37f + frame.by * -0.19f,
        raw[2] + frame.nz * 0.37f + frame.bz * -0.19f};
    Check(Near(handle[0], expected[0]) && Near(handle[1], expected[1]) &&
              Near(handle[2], expected[2]) &&
              (!Near(handle[0], raw[0]) || !Near(handle[1], raw[1]) ||
               !Near(handle[2], raw[2])),
          "core handle: polygon area centroid is inside the shifted tube, "
          "not its raw center cage");
    Check(!PomadeCenterHandlePointCpu(tube, std::vector<PomadeFrame>{}, 1,
                                      &handle[0], &handle[1], &handle[2], &err),
          "core handle: an incomplete frame column is refused");

    // A zero-area but offset ring is still visible as a line; use its mean
    // rather than snapping the handle back to an unrelated raw center.
    PomadeTubeDesc collapsed = tube;
    for (PomadeTubeSection &section : collapsed.sections) {
        for (size_t slot = 0; slot < section.u.size(); ++slot) {
            section.u[slot] = 0.25f + 0.1f * float(slot);
            section.v[slot] = -0.4f;
        }
    }
    float degenerate[3] = {0.0f, 0.0f, 0.0f};
    Check(PomadeCenterHandlePointCpu(collapsed, 1, &degenerate[0],
                                    &degenerate[1], &degenerate[2], &err) &&
              (!Near(degenerate[0], raw[0]) || !Near(degenerate[1], raw[1]) ||
               !Near(degenerate[2], raw[2])),
          "core handle: a collapsed offset section keeps an offset handle");
}

void CheckRigidTubeTransport()
{
    using namespace usdGenPomade;
    // A bent, asymmetric and pinned groom exercises the exact failure mode
    // from a region move: the K4 world-axis bootstrap would otherwise roll
    // its unpinned frames while its root support frame rotates correctly.
    PomadeTubeDesc source = MakeTube();
    source.centerX = {0.0f, 0.35f, 1.10f};
    source.centerY = {0.0f, 1.55f, 3.20f};
    source.centerZ = {0.0f, -0.20f, 0.45f};
    source.sections.resize(3);
    source.sections[0].t = 0.0f;
    source.sections[1] = source.sections[0];
    source.sections[1].t = 0.5f;
    source.sections[1].scale = 1.35f;
    source.sections[1].twist = 0.41f;
    source.sections[2] = source.sections[0];
    source.sections[2].t = 1.0f;
    source.sections[2].scale = 0.72f;
    source.sections[2].twist = -0.27f;
    for (size_t i = 0; i < source.sections[1].u.size(); ++i) {
        source.sections[1].u[i] += 0.07f * float(i % 3);
        source.sections[1].v[i] -= 0.03f * float(i % 2);
        source.sections[2].u[i] -= 0.04f * float(i % 2);
    }
    source.rootFramePinned = true;
    source.rootFrame.tx = 0.0f; source.rootFrame.ty = 0.0f;
    source.rootFrame.tz = 1.0f;
    source.rootFrame.nx = 1.0f; source.rootFrame.ny = 0.0f;
    source.rootFrame.nz = 0.0f;
    source.rootFrame.bx = 0.0f; source.rootFrame.by = 1.0f;
    source.rootFrame.bz = 0.0f;
    source.rootSurfaceOffsets.resize(size_t(source.ringVerts) * 3);
    for (size_t i = 0; i < source.rootSurfaceOffsets.size(); ++i)
        source.rootSurfaceOffsets[i] = 0.03f * float(i % 5);

    std::vector<PomadeFrame> beforeFrames;
    std::vector<float> before, normals, ringT;
    std::string err;
    Check(PomadeTubeFramesCpu(source, &beforeFrames, &err) &&
              PomadeTessellateCpu(source, beforeFrames, 9, &before, &normals,
                                 &ringT, &err),
          "transport: bent pinned source tessellates");

    PomadeRigidTransform rigid;
    // +90 degrees about Y, then translate.  This moves the curve across the
    // K4 Perp3 world-axis choice and catches a frame roll at interior K5 t.
    rigid.rotation[0] = 0.0f;  rigid.rotation[1] = 0.0f;
    rigid.rotation[2] = 1.0f;
    rigid.rotation[3] = 0.0f;  rigid.rotation[4] = 1.0f;
    rigid.rotation[5] = 0.0f;
    rigid.rotation[6] = -1.0f; rigid.rotation[7] = 0.0f;
    rigid.rotation[8] = 0.0f;
    rigid.translation[0] = 3.0f;
    rigid.translation[1] = -2.0f;
    rigid.translation[2] = 5.0f;
    PomadeTubeDesc moved;
    Check(PomadeRigidTransformTubeCpu(source, rigid, &moved, &err),
          "transport: rigid tube descriptor succeeds");
    bool authored = moved.sections.size() == source.sections.size() &&
                    moved.ringVerts == source.ringVerts;
    for (size_t i = 0; authored && i < source.sections.size(); ++i) {
        PomadeTubeSection const &a = source.sections[i];
        PomadeTubeSection const &b = moved.sections[i];
        authored = a.t == b.t && a.scale == b.scale && a.twist == b.twist &&
                   a.u == b.u && a.v == b.v;
    }
    Check(authored, "transport: counts, UV, scale and twist stay authored");
    Check(moved.frameReference != source.frameReference,
          "transport: material K4 reference composes the rotation");

    std::vector<PomadeFrame> movedFrames;
    std::vector<float> after, afterNormals, afterT;
    Check(PomadeTubeFramesCpu(moved, &movedFrames, &err) &&
              PomadeTessellateCpu(moved, movedFrames, 9, &after, &afterNormals,
                                 &afterT, &err),
          "transport: moved tube tessellates");
    bool rigidMesh = before.size() == after.size();
    for (size_t i = 0; rigidMesh && i < before.size(); i += 3) {
        float const x = before[i + 0], y = before[i + 1], z = before[i + 2];
        float const want[3] = {z + rigid.translation[0],
                               y + rigid.translation[1],
                               -x + rigid.translation[2]};
        rigidMesh = Near(after[i + 0], want[0], 3e-4f) &&
                    Near(after[i + 1], want[1], 3e-4f) &&
                    Near(after[i + 2], want[2], 3e-4f);
    }
    Check(rigidMesh,
          "transport: every interpolated K5 section remains a rigid pose");

    PomadeRigidTransform reflected = rigid;
    reflected.rotation[0] = -1.0f;
    Check(!PomadeRigidTransformTubeCpu(source, reflected, &moved, &err),
          "transport: reflection is refused instead of flipping a groom");
}

void CheckK8()
{
    using namespace usdGenPomade;
    PomadeTubeDesc const tube = MakeTube();
    std::vector<PomadeFrame> frames = FramesFor(tube);
    float const center[3] = {0.0f, 0.0f, 0.0f};
    std::string err;
    std::vector<PomadeGuideRoot> a, b;
    Check(PomadeRootSampleDiscCpu(0, 7, 0.5f, center, frames[0], 16, &a, 0,
                                 &err),
          "K8: disc sampling fills the count");
    Check(a.size() == 16, "K8: disc count is exact");
    Check(PomadeRootSampleDiscCpu(0, 7, 0.5f, center, frames[0], 16, &b, 0,
                                 &err),
          "K8: disc sampling repeats");
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i) {
        same = a[i].ru == b[i].ru && a[i].rv == b[i].rv &&
               a[i].px == b[i].px && a[i].faceId == -1;
    }
    Check(same, "K8: disc stream is bit-deterministic per (tubeId, seed)");
    bool inDisc = true;
    for (auto const &r : a) {
        inDisc = inDisc && r.ru * r.ru + r.rv * r.rv <= 1.0f + 1e-6f;
    }
    Check(inDisc, "K8: disc roots sit in the unit disc");
    std::vector<PomadeGuideRoot> c;
    Check(PomadeRootSampleDiscCpu(0, 8, 0.5f, center, frames[0], 16, &c, 0,
                                 &err),
          "K8: a second seed samples");
    bool differs = false;
    for (size_t i = 0; i < a.size(); ++i) {
        differs = differs || a[i].ru != c[i].ru || a[i].rv != c[i].rv;
    }
    Check(differs, "K8: seeds draw different roots");
    // Freeze keeps the prefix bit-exactly and extends the tail.
    std::vector<PomadeGuideRoot> grown = a;
    Check(PomadeRootSampleDiscCpu(0, 7, 0.5f, center, frames[0], 24, &grown,
                                 16, &err),
          "K8: frozen refill extends the tail");
    bool kept = grown.size() == 24;
    for (size_t i = 0; kept && i < a.size(); ++i) {
        kept = grown[i].ru == a[i].ru && grown[i].rv == a[i].rv;
    }
    Check(kept, "K8: frozen roots survive the refill bit-exactly");
    // Blue-noise spacing: the closest pair is well above zero.
    float minD2 = 1e30f;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = i + 1; j < a.size(); ++j) {
            float const dx = a[i].ru - a[j].ru;
            float const dy = a[i].rv - a[j].rv;
            minD2 = std::min(minD2, dx * dx + dy * dy);
        }
    }
    Check(minD2 > 1e-4f, "K8: dart throwing spaces the roots");
    Check(!PomadeRootSampleDiscCpu(0, 0, 0.0f, center, frames[0], 4, &a, 0,
                                  &err),
          "K8: zero radius fails");
}

void CheckK9()
{
    using namespace usdGenPomade;
    PomadeTubeDesc const tube = MakeTube();
    std::vector<PomadeFrame> frames = FramesFor(tube);
    float const center[3] = {0.0f, 0.0f, 0.0f};
    std::string err;
    std::vector<PomadeGuideRoot> roots;
    Check(PomadeRootSampleDiscCpu(0, 7, 0.5f, center, frames[0], 16, &roots,
                                 0, &err),
          "K9: fixture roots sample");
    PomadeFillDesc fill;
    fill.density = 16.0f;
    fill.cvCount = 8;
    fill.seed = 7;
    std::vector<float> points, scales;
    Check(PomadeGuideFillCpu(tube, frames, roots, fill, &points, &scales,
                            &err),
          "K9: fill runs");
    Check(points.size() == 16 * 8 * 3 && scales.size() == 16,
          "K9: guideCount x cvCount points + scales");
    // Root CVs sit on their roots.
    bool onRoot = true;
    for (int g = 0; g < 16; ++g) {
        onRoot = onRoot &&
                 Near(points[size_t(g) * 24 + 0], roots[size_t(g)].px,
                      1e-4f) &&
                 Near(points[size_t(g) * 24 + 1], roots[size_t(g)].py,
                      1e-4f) &&
                 Near(points[size_t(g) * 24 + 2], roots[size_t(g)].pz, 1e-4f);
    }
    Check(onRoot, "K9: root CVs sit on their roots");
    std::vector<float> endpointRing;
    err.clear();
    Check(PomadeSampleTubeRingCpu(
              tube, frames, std::nextafter(tube.sections.back().t, 2.0f),
              &endpointRing, &err) &&
              endpointRing.size() == size_t(tube.ringVerts) * 3,
          "K5: one-ULP generated endpoint station clamps to the last ring");
    // Tips reach the tube tip on the uniform profile.
    bool tipTop = true;
    for (int g = 0; g < 16; ++g) {
        tipTop = tipTop &&
                 Near(points[size_t(g) * 24 + 7 * 3 + 1], 4.0f, 1e-3f);
    }
    Check(tipTop, "K9: uniform tips reach the tube tip");
    // Edge bias pushes the mid-guide radius outward / inward.
    auto meanMidR = [&](float bias) {
        PomadeFillDesc f = fill;
        f.edgeBias = bias;
        std::vector<float> p, s;
        std::string e;
        if (!PomadeGuideFillCpu(tube, frames, roots, f, &p, &s, &e)) {
            return -1.0f;
        }
        double acc = 0.0;
        for (int g = 0; g < 16; ++g) {
            float const x = p[size_t(g) * 24 + 3 * 3 + 0];
            float const z = p[size_t(g) * 24 + 3 * 3 + 2];
            acc += std::sqrt(double(x * x + z * z));
        }
        return float(acc / 16.0);
    };
    float const r0 = meanMidR(0.0f);
    float const rPlus = meanMidR(1.0f);
    float const rMinus = meanMidR(-1.0f);
    Check(r0 > 0.0f && rPlus > r0 && rMinus < r0,
          "K9: edgeBias pushes/pulls the mid-guide radius");
    // The length profile shortens edge guides.
    PomadeFillDesc prof = fill;
    prof.lengthProfile = {0.0f, 1.0f, 1.0f, 0.25f};
    std::vector<float> pp, ps;
    Check(PomadeGuideFillCpu(tube, frames, roots, prof, &pp, &ps, &err),
          "K9: profiled fill runs");
    double tipY = 0.0;
    for (int g = 0; g < 16; ++g) {
        tipY += pp[size_t(g) * 24 + 7 * 3 + 1];
    }
    Check(tipY / 16.0 < 4.0, "K9: lengthProfile shortens the tips");

    // K9 material sampling follows the actual K5 polygon, not its mean
    // radius. Keep a collinear edge split and a concave root so the binding
    // must retain the authored slot all the way to a scaled/twisted tip.
    PomadeTubeDesc shaped = MakeTube(6);
    shaped.sections[0].u = {0.0f, 1.0f, 2.0f, 2.0f, 0.0f, 0.6f};
    shaped.sections[0].v = {0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f};
    shaped.sections[1] = shaped.sections[0];
    shaped.sections[1].t = 1.0f;
    shaped.sections[1].scale = 1.35f;
    shaped.sections[1].twist = 0.37f;
    shaped.sections[1].u[1] = 3.25f;
    shaped.sections[1].v[1] = -0.55f;
    std::vector<PomadeFrame> shapedFrames = FramesFor(shaped);
    std::vector<std::array<int, 3>> shapedTriangles;
    Check(PomadeTriangulateSectionSlotsCpu(shaped.sections[0],
                                          &shapedTriangles, &err),
          "K9: concave root triangulates");
    bool allSlotsIncident = shapedTriangles.size() == 4;
    for (int slotIndex = 0; allSlotsIncident && slotIndex < shaped.ringVerts;
         ++slotIndex) {
        bool found = false;
        for (std::array<int, 3> const &tri : shapedTriangles) {
            found = found || tri[0] == slotIndex || tri[1] == slotIndex ||
                    tri[2] == slotIndex;
        }
        allSlotsIncident = found;
    }
    Check(allSlotsIncident,
          "K9: root triangulation retains every boundary slot");
    bool nondegenerateTriangles = !shapedTriangles.empty();
    for (std::array<int, 3> const &tri : shapedTriangles) {
        float const ax = shaped.sections[0].u[size_t(tri[0])];
        float const ay = shaped.sections[0].v[size_t(tri[0])];
        float const bx = shaped.sections[0].u[size_t(tri[1])];
        float const by = shaped.sections[0].v[size_t(tri[1])];
        float const cx = shaped.sections[0].u[size_t(tri[2])];
        float const cy = shaped.sections[0].v[size_t(tri[2])];
        nondegenerateTriangles = nondegenerateTriangles &&
            std::fabs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) >
                1e-8f;
    }
    Check(nondegenerateTriangles,
          "K9: root triangulation emits no degenerate rail triangle");
    float shapedRootP[3] = {0.0f, 0.0f, 0.0f};
    Check(SectionSlotWorld(shaped, 0, 1, shapedRootP),
          "K9: asymmetric fixture root slot resolves");
    PomadeGuideRoot shapedRoot;
    shapedRoot.px = shapedRootP[0];
    shapedRoot.py = shapedRootP[1];
    shapedRoot.pz = shapedRootP[2];
    PomadeFillDesc shapedFill;
    shapedFill.cvCount = 3;
    std::vector<float> shapedPoints, shapedScales, midRing, tipRing;
    err.clear();
    bool const shapedFillOk = PomadeGuideFillCpu(
        shaped, shapedFrames, {shapedRoot}, shapedFill, &shapedPoints,
        &shapedScales, &err);
    std::string const shapedFillErr = err;
    err.clear();
    bool const midRingOk = PomadeSampleTubeRingCpu(
        shaped, shapedFrames, 0.5f, &midRing, &err);
    std::string const midRingErr = err;
    err.clear();
    bool const tipRingOk = PomadeSampleTubeRingCpu(
        shaped, shapedFrames, 1.0f, &tipRing, &err);
    Check(shapedFillOk && midRingOk && tipRingOk,
          "K9: asymmetric full-polygon fill evaluates: fill=" +
              shapedFillErr + " mid=" + midRingErr + " tip=" + err);
    size_t const mid = 3;
    size_t const tip = 6;
    size_t const slot = 3;
    bool const fullShape = shapedPoints.size() >= 9 && midRing.size() >= 6 &&
                           tipRing.size() >= 6 &&
          Near(shapedPoints[mid + 0], midRing[slot + 0], 2e-4f) &&
              Near(shapedPoints[mid + 1], midRing[slot + 1], 2e-4f) &&
              Near(shapedPoints[mid + 2], midRing[slot + 2], 2e-4f) &&
              Near(shapedPoints[tip + 0], tipRing[slot + 0], 2e-4f) &&
              Near(shapedPoints[tip + 1], tipRing[slot + 1], 2e-4f) &&
              Near(shapedPoints[tip + 2], tipRing[slot + 2], 2e-4f);
    Check(fullShape,
          "K9: interior and terminal guides retain full asymmetric slot");

    // A material point inside the concave root must remain inside an
    // independently deformed concave terminal section. The old root-triangle
    // affine weights can cross the terminal re-entrant edge here; K9 now
    // transfers the biased point through the canonical slot polygon.
    std::vector<PomadeGuideRoot> concaveRoots;
    std::vector<float> rootRing;
    float rootCenter[3] = {0.0f, 0.0f, 0.0f};
    PomadeFrame rootFrame;
    err.clear();
    bool const concaveRootFixture =
        PomadeSampleTubeRingCpu(shaped, shapedFrames, 0.0f, &rootRing, &err) &&
        PomadeSampleCenterCpu(shaped, shapedFrames, 0.0f, &rootCenter[0],
                             &rootCenter[1], &rootCenter[2], &rootFrame,
                             &err);
    if (concaveRootFixture) {
        for (std::array<int, 3> const &tri : shapedTriangles) {
            PomadeGuideRoot root;
            root.px = (rootRing[size_t(tri[0]) * 3 + 0] +
                       rootRing[size_t(tri[1]) * 3 + 0] +
                       rootRing[size_t(tri[2]) * 3 + 0]) / 3.0f;
            root.py = (rootRing[size_t(tri[0]) * 3 + 1] +
                       rootRing[size_t(tri[1]) * 3 + 1] +
                       rootRing[size_t(tri[2]) * 3 + 1]) / 3.0f;
            root.pz = (rootRing[size_t(tri[0]) * 3 + 2] +
                       rootRing[size_t(tri[1]) * 3 + 2] +
                       rootRing[size_t(tri[2]) * 3 + 2]) / 3.0f;
            concaveRoots.push_back(root);
        }
    }
    PomadeFillDesc concaveFill = shapedFill;
    concaveFill.cvCount = 5;
    concaveFill.edgeBias = 1.0f;
    std::vector<float> concavePoints, concaveScales;
    err.clear();
    bool const concaveFillOk = concaveRootFixture && PomadeGuideFillCpu(
        shaped, shapedFrames, concaveRoots, concaveFill, &concavePoints,
        &concaveScales, &err);
    float tipCenter[3] = {0.0f, 0.0f, 0.0f};
    PomadeFrame tipFrame;
    bool const concaveTipFrame = PomadeSampleCenterCpu(
        shaped, shapedFrames, 1.0f, &tipCenter[0], &tipCenter[1],
        &tipCenter[2], &tipFrame, &err);
    auto insideTip = [&](float px, float py, float pz) {
        std::vector<std::array<float, 2>> polygon;
        polygon.reserve(size_t(shaped.ringVerts));
        for (int slotIndex = 0; slotIndex < shaped.ringVerts; ++slotIndex) {
            size_t const at = size_t(slotIndex) * 3;
            float const dx = tipRing[at + 0] - tipCenter[0];
            float const dy = tipRing[at + 1] - tipCenter[1];
            float const dz = tipRing[at + 2] - tipCenter[2];
            polygon.push_back({{dx * tipFrame.nx + dy * tipFrame.ny +
                                     dz * tipFrame.nz,
                                 dx * tipFrame.bx + dy * tipFrame.by +
                                     dz * tipFrame.bz}});
        }
        float const dx = px - tipCenter[0];
        float const dy = py - tipCenter[1];
        float const dz = pz - tipCenter[2];
        float const u = dx * tipFrame.nx + dy * tipFrame.ny + dz * tipFrame.nz;
        float const v = dx * tipFrame.bx + dy * tipFrame.by + dz * tipFrame.bz;
        bool inside = false;
        for (size_t a = 0, b = polygon.size() - 1; a < polygon.size(); b = a++) {
            float const ax = polygon[a][0], ay = polygon[a][1];
            float const bx = polygon[b][0], by = polygon[b][1];
            float const cross = (bx - ax) * (v - ay) - (by - ay) * (u - ax);
            float const dot = (u - ax) * (u - bx) + (v - ay) * (v - by);
            if (std::fabs(cross) <= 1e-4f && dot <= 1e-4f) return true;
            bool const crosses = (ay > v) != (by > v);
            if (crosses && u < (bx - ax) * (v - ay) / (by - ay) + ax)
                inside = !inside;
        }
        return inside;
    };
    bool concaveTipsInside = concaveFillOk && concaveTipFrame &&
                             tipRing.size() == size_t(shaped.ringVerts) * 3;
    for (size_t g = 0; concaveTipsInside && g < concaveRoots.size(); ++g) {
        size_t const at = (g * size_t(concaveFill.cvCount) +
                           size_t(concaveFill.cvCount - 1)) * 3;
        concaveTipsInside = insideTip(concavePoints[at + 0],
                                      concavePoints[at + 1],
                                      concavePoints[at + 2]);
    }
    Check(concaveTipsInside,
          "K9: biased concave material points remain inside terminal ring");

    // With the same concave section at both ends, canonical transfer is an
    // identity for every non-boundary material sample. This catches a root
    // bind/target ear-order mismatch, including the retained collinear slot.
    PomadeTubeDesc identityConcave = shaped;
    identityConcave.sections[1] = identityConcave.sections[0];
    identityConcave.sections[1].t = 1.0f;
    std::vector<PomadeFrame> identityFrames = FramesFor(identityConcave);
    PomadeFillDesc identityFill;
    identityFill.cvCount = 5;
    std::vector<float> identityPoints, identityScales;
    err.clear();
    bool const identityFillOk = concaveRootFixture && PomadeGuideFillCpu(
        identityConcave, identityFrames, concaveRoots, identityFill,
        &identityPoints, &identityScales, &err);
    bool identityMaterial = identityFillOk;
    for (size_t g = 0; identityMaterial && g < concaveRoots.size(); ++g) {
        float const rootDx = concaveRoots[g].px - rootCenter[0];
        float const rootDy = concaveRoots[g].py - rootCenter[1];
        float const rootDz = concaveRoots[g].pz - rootCenter[2];
        float const rootU = rootDx * rootFrame.nx + rootDy * rootFrame.ny +
                            rootDz * rootFrame.nz;
        float const rootV = rootDx * rootFrame.bx + rootDy * rootFrame.by +
                            rootDz * rootFrame.bz;
        for (int c = 1; identityMaterial && c < identityFill.cvCount; ++c) {
            float centerAt[3] = {0.0f, 0.0f, 0.0f};
            PomadeFrame frameAt;
            float const t = float(c) / float(identityFill.cvCount - 1);
            identityMaterial = PomadeSampleCenterCpu(
                identityConcave, identityFrames, t, &centerAt[0],
                &centerAt[1], &centerAt[2], &frameAt, &err);
            size_t const at =
                (g * size_t(identityFill.cvCount) + size_t(c)) * 3;
            float const dx = identityPoints[at + 0] - centerAt[0];
            float const dy = identityPoints[at + 1] - centerAt[1];
            float const dz = identityPoints[at + 2] - centerAt[2];
            identityMaterial = identityMaterial &&
                Near(dx * frameAt.nx + dy * frameAt.ny + dz * frameAt.nz,
                     rootU, 2e-4f) &&
                Near(dx * frameAt.bx + dy * frameAt.by + dz * frameAt.bz,
                     rootV, 2e-4f);
        }
    }
    Check(identityMaterial,
          "K9: unchanged concave and collinear material stays exact");

    // A complete point taper is valid material geometry. It bypasses only
    // polygon remapping at that station; a nonzero collinear terminal still
    // fails rather than being mistaken for a taper.
    PomadeTubeDesc pointTaper = shaped;
    for (int slotIndex = 0; slotIndex < pointTaper.ringVerts; ++slotIndex) {
        pointTaper.sections.back().u[size_t(slotIndex)] = 0.37f;
        pointTaper.sections.back().v[size_t(slotIndex)] = -0.21f;
    }
    std::vector<PomadeFrame> pointTaperFrames = FramesFor(pointTaper);
    std::vector<float> pointTaperPoints, pointTaperScales, pointTaperRing;
    err.clear();
    bool const pointTaperFill = PomadeGuideFillCpu(
        pointTaper, pointTaperFrames, {shapedRoot}, shapedFill,
        &pointTaperPoints, &pointTaperScales, &err) &&
        PomadeSampleTubeRingCpu(pointTaper, pointTaperFrames, 1.0f,
                               &pointTaperRing, &err);
    size_t const pointTip = size_t(shapedFill.cvCount - 1) * 3;
    bool const pointTaperExact = pointTaperFill && pointTaperRing.size() >= 3 &&
        Near(pointTaperPoints[pointTip + 0], pointTaperRing[0], 2e-4f) &&
        Near(pointTaperPoints[pointTip + 1], pointTaperRing[1], 2e-4f) &&
        Near(pointTaperPoints[pointTip + 2], pointTaperRing[2], 2e-4f);
    Check(pointTaperExact, "K9: point taper emits its common terminal point");
    PomadeTubeDesc lineTaper = pointTaper;
    for (int slotIndex = 0; slotIndex < lineTaper.ringVerts; ++slotIndex) {
        lineTaper.sections.back().u[size_t(slotIndex)] = float(slotIndex);
        lineTaper.sections.back().v[size_t(slotIndex)] = 0.0f;
    }
    std::vector<float> lineTaperPoints, lineTaperScales;
    Check(!PomadeGuideFillCpu(lineTaper, FramesFor(lineTaper), {shapedRoot},
                             shapedFill, &lineTaperPoints, &lineTaperScales,
                             &err),
          "K9: nonzero collinear taper remains invalid material geometry");

    PomadeGuideRoot detachedRoot = shapedRoot;
    detachedRoot.px += 0.17f;
    PomadeFillDesc stoppedFill = shapedFill;
    stoppedFill.lengthProfile = {0.0f, 0.0f, 1.0f, 0.0f};
    std::vector<float> stoppedPoints, stoppedScales;
    Check(PomadeGuideFillCpu(shaped, shapedFrames, {detachedRoot}, stoppedFill,
                            &stoppedPoints, &stoppedScales, &err),
          "K9: detached zero-length fill evaluates");
    bool staysAttached = stoppedPoints.size() == 9;
    for (int c = 0; staysAttached && c < 3; ++c) {
        size_t const at = size_t(c) * 3;
        staysAttached = Near(stoppedPoints[at + 0], detachedRoot.px, 2e-4f) &&
                        Near(stoppedPoints[at + 1], detachedRoot.py, 2e-4f) &&
                        Near(stoppedPoints[at + 2], detachedRoot.pz, 2e-4f);
    }
    Check(staysAttached,
          "K9: zero-length detached guide remains physically attached");
    Check(!PomadeGuideFillCpu(tube, frames, roots, fill, nullptr, nullptr,
                             &err),
          "K9: null outputs fail");
    PomadeFillDesc badBias = fill;
    badBias.edgeBias = 2.0f;
    Check(!PomadeGuideFillCpu(tube, frames, roots, badBias, &pp, &ps, &err),
          "K9: bias outside [-1, 1] fails");
    PomadeFillDesc badProf = fill;
    badProf.lengthProfile = {0.0f};
    Check(!PomadeGuideFillCpu(tube, frames, roots, badProf, &pp, &ps, &err),
          "K9: odd profile floats fail");
}

void CheckK10()
{
    using namespace usdGenPomade;
    // Two straight guides of uneven CV spacing resample evenly.
    float points[2 * 4 * 3] = {
        0, 0, 0, 0, 1, 0, 0, 1.5f, 0, 0, 4, 0,
        1, 0, 0, 1, 3, 0, 1, 3.5f, 0, 1, 4, 0,
    };
    int counts[2] = {4, 4};
    float rootDirs[9] = {1, 0, 0, 0, 0, 1, 0, 1, 0};
    std::vector<float> out;
    std::vector<int> outCounts;
    std::vector<double> outFrames;
    std::string err;
    Check(PomadeGuideResampleCpu(points, counts, 2, 5, rootDirs, &out,
                                &outCounts, &outFrames, &err),
          "K10: resample runs");
    Check(out.size() == 2 * 5 * 3 && outCounts == std::vector<int>({5, 5}) &&
              outFrames.size() == 32,
          "K10: fixed cvCount + one 16-double frame per guide");
    bool even = true;
    for (int g = 0; g < 2; ++g) {
        for (int c = 0; c < 5; ++c) {
            float const y =
                out[(size_t(g) * 5 + size_t(c)) * 3 + 1];
            even = even && Near(y, float(c), 1e-4f);
        }
    }
    Check(even, "K10: straight guides resample arc-length evenly");
    bool frame = NearD(outFrames[0], 1.0) && NearD(outFrames[5], 0.0) &&
                 NearD(outFrames[6], 1.0) && NearD(outFrames[10], 0.0) &&
                 NearD(outFrames[12], 0.0) && NearD(outFrames[15], 1.0);
    Check(frame, "K10: root frames carry dirs + root position");
    Check(!PomadeGuideResampleCpu(nullptr, counts, 2, 5, rootDirs, &out,
                                 &outCounts, &outFrames, &err),
          "K10: null points fail");
    int badCounts[1] = {1};
    Check(!PomadeGuideResampleCpu(points, badCounts, 1, 5, rootDirs, &out,
                                 &outCounts, &outFrames, &err),
          "K10: single-CV guides fail");
}

void CheckK11()
{
    using namespace usdGenPomade;
    // Identity projection: NDC == world, w = 800x600.
    float vp[16] = {0};
    vp[0] = vp[5] = vp[10] = vp[15] = 1.0f;
    float verts[9] = {
        0.0f, 0.0f, 0.0f,  // center -> (400, 300)
        0.5f, 0.5f, 0.0f,  // -> (600, 150)
        -0.5f, -0.5f, 0.5f,  // -> (200, 450), nearer depth wins ties
    };
    PomadePickSets sets;
    sets.tubeVerts = verts;
    sets.tubeVertCount = 3;
    PomadePickHit hit = PomadePickCpu(sets, PomadePick_TubeVert, vp, 800, 600,
                                    400.0f, 300.0f, 5.0f);
    Check(hit.hit && hit.kind == PomadePick_TubeVert && hit.index == 0 &&
              Near(hit.distPx, 0.0f, 1e-3f),
          "K11: the projected pixel hits its vertex");
    hit = PomadePickCpu(sets, PomadePick_TubeVert, vp, 800, 600, 10.0f, 10.0f,
                       5.0f);
    Check(!hit.hit, "K11: empty pixels miss");
    hit = PomadePickCpu(sets, PomadePick_CenterCV, vp, 800, 600, 400.0f,
                       300.0f, 5.0f);
    Check(!hit.hit, "K11: the kind mask excludes absent kinds");
    // Depth breaks pixel ties toward the viewer.
    float tie[6] = {0.0f, 0.0f, 0.9f, 0.0f, 0.0f, -0.9f};
    PomadePickSets tieSets;
    tieSets.tubeVerts = tie;
    tieSets.tubeVertCount = 2;
    hit = PomadePickCpu(tieSets, PomadePick_TubeVert, vp, 800, 600, 400.0f,
                       300.0f, 5.0f);
    Check(hit.hit && hit.index == 1 && Near(hit.depth, -0.9f, 1e-5f),
          "K11: depth breaks pixel ties toward the viewer");
    // Section CVs report (ring, slot); guides report (guide, cv).
    float sections[12] = {0.0f, 0.0f, 0.0f, 0.25f, 0.0f, 0.0f,
                          0.0f, 0.25f, 0.0f, 0.25f, 0.25f, 0.0f};
    PomadePickSets secSets;
    secSets.sectionCVs = sections;
    secSets.sectionCVCount = 4;
    secSets.sectionRingVerts = 2;
    hit = PomadePickCpu(secSets, PomadePick_SectionCV, vp, 800, 600, 400.0f,
                       300.0f, 5.0f);
    Check(hit.hit && hit.index == 0 && hit.subIndex == 0,
          "K11: section hits report (ring, slot)");
    hit = PomadePickCpu(secSets, PomadePick_SectionCV, vp, 800, 600, 500.0f,
                       300.0f, 5.0f);
    Check(hit.hit && hit.index == 0 && hit.subIndex == 1,
          "K11: section hits find the right slot");
    PomadePickSets guideSets;
    guideSets.guideCVs = sections;
    guideSets.guideCount = 2;
    guideSets.guideCvCount = 2;
    hit = PomadePickCpu(guideSets, PomadePick_Guide, vp, 800, 600, 400.0f,
                       225.0f, 5.0f);
    Check(hit.hit && hit.index == 1 && hit.subIndex == 0,
          "K11: guide hits report (guide, cv)");
    // Behind the camera never hits (w <= 0).
    float behind[16] = {0};
    behind[0] = behind[5] = behind[10] = 1.0f;
    behind[15] = -1.0f;  // w = -1 for every point
    hit = PomadePickCpu(sets, PomadePick_TubeVert, behind, 800, 600, 400.0f,
                       300.0f, 1000.0f);
    Check(!hit.hit, "K11: points behind the camera miss");
}

void CheckModelTubeMode()
{
    using namespace usdGenPomade;
    PomadeModel model;
    Check(model.BuildTestTube(), "tube: the test tube builds");
    // The P0/P1 position contract survives P3 untouched.
    PomadeModel::HostTubeMesh const &mesh = model.GetHostMesh();
    Check(Near(mesh.positions[0], 0.5f, 1e-6f) &&
              Near(mesh.positions[1], 0.0f, 1e-6f) &&
              Near(mesh.positions[2], 0.0f, 1e-6f),
          "tube: ring-0 slot-0 is still (0.5, 0, 0)");
    Check(model.GetSectionCount() == 5,
          "tube: default sections match the ring count");
    PomadeTubeSection sec;
    Check(model.GetSection(0, &sec) && sec.u.size() == 8 &&
              Near(sec.u[0], 0.5f, 1e-6f) && Near(sec.v[0], 0.0f, 1e-6f),
          "tube: default section 0 is the R=0.5 circle");
    Check(model.GetTubeRegionId() == -1,
          "tube: the test tube starts unrooted");
    // Center CV ops.
    Check(model.GetCenterCVCount() == 5, "tube: five center CVs");
    Check(model.MoveCenterCV(2, 1.0f, 0.0f, 0.0f),
          "tube: center CV moves");
    float x = 0, y = 0, z = 0;
    Check(model.GetCenterCV(2, &x, &y, &z) && Near(x, 1.0f) &&
              Near(y, 2.0f),
          "tube: the moved CV lands on (1, 2, 0)");
    Check(!model.MoveCenterCV(9, 1.0f, 0.0f, 0.0f),
          "tube: out-of-range CV moves fail");
    Check(model.InsertCenterCV(2), "tube: center CV inserts");
    Check(model.GetCenterCVCount() == 6, "tube: insert grows the census");
    Check(model.DeleteCenterCV(2), "tube: center CV deletes");
    Check(model.GetCenterCVCount() == 5, "tube: delete shrinks the census");
    {
        PomadeModel floor;
        floor.BuildTestTube();
        floor.DeleteCenterCV(0);
        floor.DeleteCenterCV(0);
        floor.DeleteCenterCV(0);
        Check(floor.GetCenterCVCount() == 2 &&
                  !floor.DeleteCenterCV(0),
              "tube: delete keeps >= 2 CVs");
    }
    // Soft selection spreads the move with the smoothstep falloff.
    Check(model.SetSoftSelection(0.5f, 0.5f), "tube: soft selection sets");
    float sc = 0, sr = 0;
    model.GetSoftSelection(&sc, &sr);
    Check(Near(sc, 0.5f) && Near(sr, 0.5f),
          "tube: soft selection reads back");
    Check(!model.SetSoftSelection(0.0f, -1.0f),
          "tube: negative soft radius fails");
    {
        PomadeModel soft;
        soft.BuildTestTube();
        soft.SetSoftSelection(0.5f, 0.5f);
        soft.MoveCenterCV(2, 1.0f, 0.0f, 0.0f);
        float xa = 0, xb = 0, xc = 0, ya = 0, yb = 0, yc = 0, za = 0, zb = 0,
              zc = 0;
        soft.GetCenterCV(2, &xb, &yb, &zb);
        soft.GetCenterCV(1, &xa, &ya, &za);
        soft.GetCenterCV(3, &xc, &yc, &zc);
        Check(Near(xb, 1.0f) && xa > 0.0f && xa < 1.0f &&
                  Near(xa, xc, 1e-5f),
              "tube: soft moves peak at the CV and spread symmetrically");
    }
    {
        // V6: soft selection is a property of the MOVE, not of tube 0.
        // MoveTubeCenterCV on a subdivided child used to snap one CV while
        // the same drag on the root feathered (plan/17 §5.2, the V0b note
        // there says the per-tube spellings keep tube 0's behaviour "soft
        // selection included" — before V6 they did not).
        PomadeModel child;
        child.BuildTestTube();
        std::vector<int> kids;
        Check(child.SubdivideTube(0, 2, "kmeans", 3, &kids) &&
                  kids.size() == 2,
              "tube: the soft-selection child model subdivides");
        if (kids.size() == 2) {
            PomadeTubeDesc before;
            child.GetTubeDesc(kids[0], &before);
            child.SetSoftSelection(0.5f, 0.5f);
            Check(child.MoveTubeCenterCV(kids[0], 2, 1.0f, 0.0f, 0.0f),
                  "tube: a child center CV moves with soft selection on");
            PomadeTubeDesc after;
            child.GetTubeDesc(kids[0], &after);
            bool shaped = after.centerX.size() == before.centerX.size() &&
                          before.centerX.size() >= 5;
            if (shaped) {
                float const d1 = after.centerX[1] - before.centerX[1];
                float const d2 = after.centerX[2] - before.centerX[2];
                float const d3 = after.centerX[3] - before.centerX[3];
                Check(Near(d2, 1.0f) && d1 > 0.0f && d1 < 1.0f &&
                          Near(d1, d3, 1e-5f),
                      "tube: a per-tube move feathers exactly as tube 0's "
                      "does");
            } else {
                Check(false, "tube: the child has a comparable center");
            }
            child.SetSoftSelection(0.0f, 0.0f);
        }
    }
    Check(model.SetSoftSelection(0.0f, 0.0f), "tube: soft selection clears");
    // Length + sections.
    Check(model.SetTubeLength(8.0f), "tube: length sets");
    {
        // On a straight tube the tip scales exactly with the length (the
        // shared model above is bent, so lengthen a fresh one).
        PomadeModel straight;
        straight.BuildTestTube();
        straight.SetTubeLength(8.0f);
        float tx = 0, ty = 0, tz = 0;
        Check(straight.GetCenterCV(4, &tx, &ty, &tz) &&
                  Near(tx, 0.0f, 1e-5f) && Near(ty, 8.0f, 1e-4f) &&
                  Near(tz, 0.0f, 1e-5f),
              "tube: the tip doubles with the length");
    }
    Check(!model.SetTubeLength(0.0f), "tube: zero length fails");
    Check(model.MoveSectionRing(1, 0.25f, 0.0f), "tube: ring moves");
    Check(model.GetSection(1, &sec) && Near(sec.u[0], 0.75f, 1e-5f),
          "tube: the ring offset lands on its CVs");
    Check(model.ScaleSectionRing(1, 2.0f), "tube: ring scales");
    Check(model.GetSection(1, &sec) && Near(sec.scale, 2.0f, 1e-6f),
          "tube: ring scale reads back");
    Check(!model.ScaleSectionRing(1, 0.0f),
          "tube: non-positive ring scale fails");
    Check(model.TwistSectionRing(1, 0.5f), "tube: ring twists");
    Check(model.GetSection(1, &sec) && Near(sec.twist, 0.5f, 1e-6f),
          "tube: ring twist reads back");
    Check(model.MoveSectionCV(1, 3, 0.1f, -0.1f), "tube: section CV drags");
    Check(model.GetSection(1, &sec) &&
              Near(sec.u[3], 0.25f + 0.5f * std::cos(3.0f * 0.78539816339f) +
                                 0.1f,
                   1e-4f),
          "tube: the dragged CV lands on its offset");
    Check(!model.MoveSectionCV(1, 99, 0.1f, 0.0f),
          "tube: out-of-range slot drags fail");
    Check(model.AddSectionRing(0.5f), "tube: rings insert at t");
    Check(model.GetSectionCount() == 6, "tube: insert grows the rings");
    Check(!model.AddSectionRing(0.0f) && !model.AddSectionRing(2.0f),
          "tube: inserts outside the open range fail");
    Check(model.CopySectionRing(1, 2), "tube: rings copy");
    {
        PomadeTubeSection src, dst;
        model.GetSection(1, &src);
        model.GetSection(2, &dst);
        Check(src.u == dst.u && src.v == dst.v &&
                  Near(src.scale, dst.scale) && Near(src.twist, dst.twist) &&
                  !Near(src.t, dst.t),
              "tube: copies carry CVs/scale/twist, keep t");
    }
    Check(model.RemoveSectionRing(2), "tube: rings remove");
    Check(model.GetSectionCount() == 5, "tube: remove shrinks the rings");
    // Relax smooths the kink; root snap needs a scalp (fails cleanly).
    {
        PomadeModel kink;
        kink.BuildTestTube();
        kink.MoveCenterCV(2, 2.0f, 0.0f, 0.0f);
        Check(kink.RelaxCenter(1.0f, 1), "tube: relax runs");
        kink.GetCenterCV(2, &x, &y, &z);
        Check(Near(x, 0.0f, 1e-4f),
              "tube: full relax pulls the kink onto the chord");
        Check(!kink.RelaxCenter(2.0f, 1),
              "tube: strength outside [0, 1] fails");
    }
    Check(!model.SnapRootToScalp() && !model.MatchSurface(),
          "tube: scalp ops fail cleanly unbound");
    // Display segments re-tessellate the sections path.
    Check(model.SetDisplaySegments(2), "tube: display segments set");
    Check(model.GetDisplaySegments() == 2, "tube: segments read back");
    Check(!model.SetDisplaySegments(0), "tube: zero segments fail");
    {
        size_t const verts = model.GetHostMesh().positions.size() / 3;
        Check(verts == size_t((5 - 1) * 2 + 1) * 8,
              "tube: segments follow (nSec-1)*seg+1 x ringVerts");
    }
    // K4 frames are cached by Sync.
    Check(model.GetFrames().size() == 5, "tube: K4 frames cache the center");
    // Region rooting round-trips.
    model.SetTubeRegionId(3);
    Check(model.GetTubeRegionId() == 3, "tube: region id round-trips");
    Check(model.Snapshot().sections.size() == 5,
          "tube: snapshots carry the sections");
}

void CheckModelFillMode()
{
    using namespace usdGenPomade;
    PomadeModel model;
    model.BuildTestTube();
    PomadeModel::FillParams params;
    params.density = 16.0f;
    params.cvCount = 8;
    params.seed = 7;
    params.edgeBias = 0.25f;
    Check(model.SetFillParams(params), "fill: params validate");
    params.edgeBias = 2.0f;
    Check(!model.SetFillParams(params), "fill: bias outside [-1, 1] fails");
    params.edgeBias = 0.25f;
    params.lengthProfile = {0.0f};
    Check(!model.SetFillParams(params), "fill: odd profile floats fail");
    params.lengthProfile.clear();
    Check(model.SetFillParams(params), "fill: params reset");
    Check(model.GetPreviewFraction() == 0.25f,
          "fill: preview defaults to 25%");
    Check(!model.GetFreezeRoots(), "fill: roots start unfrozen");
    Check(model.SetPreviewFraction(0.5f), "fill: preview fraction sets");
    Check(!model.SetPreviewFraction(2.0f),
          "fill: fractions outside [0, 1] fail");
    Check(model.SetPreviewFraction(0.25f), "fill: preview resets to 25%");
    // Full refill, then the 25% drag refill, then release.
    Check(model.RefillGuides(1.0f), "fill: full refill runs");
    Check(model.GetGuides().guideCount == 16 &&
              model.GetGuides().cvCount == 8,
          "fill: full refill holds round(density) x cvCount");
    Check(model.GetRoots().size() == 16, "fill: roots census matches");
    Check(model.RefillGuides(-1.0f), "fill: drag refill runs");
    Check(model.GetGuides().guideCount == 4,
          "fill: drag refill holds 25% of the guides");
    {
        PomadeModel::GuidePreview preview = model.GetGuidePreview();
        Check(preview.guideCount == 4 && preview.points.size() == 4 * 8 * 3,
              "fill: the preview stages points + counts");
    }
    Check(model.RefillGuides(1.0f), "fill: release refill runs");
    Check(model.GetGuides().guideCount == 16,
          "fill: release restores full density");
    // Freeze keeps the prefix bit-exactly across a density change.
    model.SetFreezeRoots(true);
    std::vector<float> before = model.GetGuides().points;
    params.density = 20.0f;
    Check(model.SetFillParams(params), "fill: density raises under freeze");
    Check(model.RefillGuides(1.0f), "fill: frozen refill runs");
    Check(model.GetGuides().guideCount == 20,
          "fill: frozen refill grows the count");
    std::vector<float> const &after = model.GetGuides().points;
    bool kept = after.size() == 20 * 8 * 3;
    for (size_t i = 0; kept && i < before.size(); ++i) {
        kept = after[i] == before[i];
    }
    Check(kept, "fill: frozen guides survive bit-exactly");
    model.SetFreezeRoots(false);
    // Guide generation is a pure snapshot function (the commit/hydrate
    // contract): same snapshot, same guides.
    PomadeModel::TubeSnapshot shot = model.Snapshot();
    PomadeGuideSet g1 = PomadeGenerateGuides(shot);
    PomadeGuideSet g2 = PomadeGenerateGuides(shot);
    Check(g1.points == g2.points && g1.frames == g2.frames &&
              g1.ids == g2.ids,
          "fill: snapshot guides are bit-deterministic");
    Check(g1.guideCount == 20, "fill: snapshot guides follow the density");
}

void CheckModelPerTubeFill()
{
    using namespace usdGenPomade;
    PomadeModel model;
    Check(model.BuildTestTube(), "per-tube fill: the tube builds");
    PomadeModel::FillParams params;
    params.density = 16.0f;
    params.cvCount = 8;
    Check(model.SetFillParams(params), "per-tube fill: global params set");
    Check(model.RefillGuides(1.0f), "per-tube fill: one tube refills");
    Check(model.GetGuides().guideCount == 16,
          "per-tube fill: one tube holds round(density)");
    // Subdivide: the parent suspends, the leaves produce from the params
    // they inherited.
    std::vector<int> kids;
    Check(model.SubdivideTube(0, 3, "kmeans", 2, &kids) && kids.size() == 3,
          "per-tube fill: the tube subdivides into three");
    Check(model.IsTubeFillSuspended(0),
          "per-tube fill: the subdivided parent suspends");
    Check(model.RefillGuides(1.0f), "per-tube fill: the leaves refill");
    Check(model.GetGuides().guideCount == 48,
          "per-tube fill: three leaves x round(density), parent silent");
    Check(model.GetRoots().size() == 48,
          "per-tube fill: the merged roots census matches");
    std::vector<int> order = kids;
    std::sort(order.begin(), order.end());
    // One leaf's density moves the merged total by exactly its delta.
    PomadeModel::FillParams leaf;
    Check(model.GetTubeFillParams(order[0], &leaf),
          "per-tube fill: the leaf params read");
    leaf.density = 32.0f;
    Check(model.SetTubeFillParams(order[0], leaf),
          "per-tube fill: one leaf takes density 32");
    Check(model.RefillGuides(1.0f), "per-tube fill: refill after the edit");
    Check(model.GetGuides().guideCount == 64,
          "per-tube fill: the merged total follows one leaf (48 + 16)");
    // The suspended parent's own density is irrelevant.
    params.density = 200.0f;
    Check(model.SetFillParams(params),
          "per-tube fill: the parent density raises");
    Check(model.RefillGuides(1.0f), "per-tube fill: refill after the raise");
    Check(model.GetGuides().guideCount == 64,
          "per-tube fill: the suspended parent contributes nothing");
    // Attribution: the merge runs ascending, each guide naming its tube.
    {
        PomadeModel::GuidePreview preview = model.GetGuidePreview();
        Check(preview.tubeIds.size() == preview.counts.size(),
              "per-tube fill: every guide names its tube");
        bool ascending = true;
        for (size_t i = 1; i < preview.tubeIds.size(); ++i) {
            ascending = ascending &&
                        preview.tubeIds[i] >= preview.tubeIds[i - 1];
        }
        Check(ascending, "per-tube fill: the merge runs ascending");
        size_t n0 = 0, n1 = 0, n2 = 0;
        for (size_t i = 0; i < preview.tubeIds.size(); ++i) {
            n0 += (preview.tubeIds[i] == order[0]);
            n1 += (preview.tubeIds[i] == order[1]);
            n2 += (preview.tubeIds[i] == order[2]);
        }
        Check(n0 == 32 && n1 == 16 && n2 == 16,
              "per-tube fill: each leaf's share follows its own density");
    }
    // Siblings that share fill params still draw distinct root patterns:
    // the disc stream is keyed by (tubeId, seed).
    {
        std::vector<PomadeGuideRoot> const &roots = model.GetRoots();
        // order[1] and order[2] both hold 16 guides at density 16; their
        // blocks start after order[0]'s 32.
        bool differ = false;
        for (int i = 0; i < 16; ++i) {
            PomadeGuideRoot const &a = roots[size_t(32 + i)];
            PomadeGuideRoot const &b = roots[size_t(48 + i)];
            if (a.px != b.px || a.py != b.py || a.pz != b.pz) {
                differ = true;
                break;
            }
        }
        Check(differ, "per-tube fill: sibling root patterns differ");
    }
    // Freeze is per tube: raising one leaf grows its own tail while the
    // siblings' guides survive bit-exactly.
    model.SetFreezeRoots(true);
    std::vector<float> before = model.GetGuides().points;
    Check(model.GetTubeFillParams(order[1], &leaf),
          "per-tube fill: the middle leaf params read");
    leaf.density = 20.0f;
    Check(model.SetTubeFillParams(order[1], leaf),
          "per-tube fill: the middle leaf takes density 20");
    Check(model.RefillGuides(1.0f), "per-tube fill: frozen refill runs");
    Check(model.GetGuides().guideCount == 68,
          "per-tube fill: the frozen refill grows the edited leaf");
    {
        std::vector<float> const &after = model.GetGuides().points;
        // order[0]: 32 guides kept at the head; order[2]: 16 guides kept
        // after order[1]'s grown block of 20.
        size_t const cv = 8, head = size_t(32) * cv * 3;
        size_t const midBefore = size_t(16) * cv * 3;
        size_t const midAfter = size_t(20) * cv * 3;
        bool kept = after.size() == size_t(68) * cv * 3 &&
                    before.size() == size_t(64) * cv * 3;
        for (size_t i = 0; kept && i < head; ++i) {
            kept = after[i] == before[i];
        }
        for (size_t i = 0; kept && i < size_t(16) * cv * 3; ++i) {
            kept = after[head + midAfter + i] == before[head + midBefore + i];
        }
        Check(kept, "per-tube fill: frozen siblings survive bit-exactly");
    }
    model.SetFreezeRoots(false);
    // Merge back: tube 0 resumes its own fill, and undo restores the
    // leaves with their per-tube root stores.
    Check(model.MergeChildren(0), "per-tube fill: the children merge back");
    Check(!model.IsTubeFillSuspended(0),
          "per-tube fill: the fill resumes once the children are gone");
    Check(model.RefillGuides(1.0f), "per-tube fill: refill after the merge");
    Check(model.GetGuides().guideCount == 200,
          "per-tube fill: the parent resumes at its own density");
    Check(model.Undo(), "per-tube fill: undo restores the leaves");
    Check(model.IsTubeFillSuspended(0),
          "per-tube fill: undo re-suspends the parent");
    Check(model.RefillGuides(1.0f), "per-tube fill: refill after undo");
    Check(model.GetGuides().guideCount == 68,
          "per-tube fill: the leaves refill from their restored stores");
}

// The 4x4 quad grid in the XZ plane (y = 0), faces x-major: face
// (ix, iz) = ix * 4 + iz, verts row-major over the 5x5 lattice.
struct Grid {
    std::vector<float> points;
    std::vector<int> counts;
    std::vector<int> indices;
};

Grid MakeGrid()
{
    Grid grid;
    for (int iz = 0; iz < 5; ++iz) {
        for (int ix = 0; ix < 5; ++ix) {
            grid.points.push_back(float(ix));
            grid.points.push_back(0.0f);
            grid.points.push_back(float(iz));
        }
    }
    for (int ix = 0; ix < 4; ++ix) {
        for (int iz = 0; iz < 4; ++iz) {
            grid.counts.push_back(4);
            // +Y winding (the pomade-graph-scalp.usda convention).
            int const v00 = iz * 5 + ix;
            grid.indices.push_back(v00);
            grid.indices.push_back(v00 + 5);
            grid.indices.push_back(v00 + 6);
            grid.indices.push_back(v00 + 1);
        }
    }
    return grid;
}

void CheckAutoTube()
{
    using namespace usdGenPomade;
    Grid grid = MakeGrid();
    PomadeModel model;
    Check(!model.BuildTubeFromRegion(0, 4, 8, 3.0f),
          "auto-tube: unbound scalp fails cleanly");
    Check(model.BindScalp(grid.points, grid.counts, grid.indices),
          "auto-tube: the grid scalp binds");
    Check(!model.BuildTubeFromRegion(0, 4, 8, 3.0f),
          "auto-tube: unknown regions fail cleanly");
    // A square loop around the middle 2x2 faces via downward raycasts.
    float down[3] = {0.0f, -1.0f, 0.0f};
    int ids[4] = {-1, -1, -1, -1};
    float corners[4][3] = {
        {1.2f, 2.0f, 1.2f}, {2.8f, 2.0f, 1.2f},
        {2.8f, 2.0f, 2.8f}, {1.2f, 2.0f, 2.8f},
    };
    for (int i = 0; i < 4; ++i) {
        PomadeHit hit = model.Raycast(corners[i], down);
        if (hit.hit) {
            ids[i] = model.GraphAddNode(hit);
        }
    }
    bool nodes = ids[0] >= 0 && ids[1] >= 0 && ids[2] >= 0 && ids[3] >= 0;
    Check(nodes, "auto-tube: four loop nodes place");
    if (nodes) {
        for (int i = 0; i < 4; ++i) {
            model.GraphConnect(ids[i], ids[(i + 1) % 4]);
        }
        Check(model.Rasterise(), "auto-tube: K3 rasterises the loop");
        std::vector<int> faces = model.TubeRegionFaces();
        Check(faces.empty(), "auto-tube: no faces before rooting");
        Check(model.BuildTubeFromRegion(0, 4, 8, 3.0f),
              "auto-tube: the region seeds a tube");
        Check(model.GetTubeRegionId() == 0,
              "auto-tube: the tube roots in region 0");
        Check(model.GetCenterCVCount() == 4,
              "auto-tube: center count follows the request");
        faces = model.TubeRegionFaces();
        Check(!faces.empty(), "auto-tube: the tube claims region faces");
        float x0 = 0, y0 = 0, z0 = 0, x3 = 0, y3 = 0, z3 = 0;
        model.GetCenterCV(0, &x0, &y0, &z0);
        model.GetCenterCV(3, &x3, &y3, &z3);
        Check(Near(x0, 2.0f, 0.6f) && Near(z0, 2.0f, 0.6f),
              "auto-tube: the root sits over the region centroid");
        Check(y3 > y0 + 2.0f && Near(x3, x0, 1e-3f) &&
                  Near(z3, z0, 1e-3f),
              "auto-tube: the center rises along +Y (the grid normal)");
        // Mesh fill roots on the claimed faces.
        PomadeModel::FillParams params;
        params.density = 4.0f;
        params.cvCount = 6;
        params.seed = 11;
        Check(model.SetFillParams(params), "auto-tube: fill params set");
        Check(model.RefillGuides(1.0f), "auto-tube: mesh refill runs");
        Check(model.GetGuides().guideCount > 0,
              "auto-tube: mesh roots fill the region");
        bool onFace = true;
        for (auto const &r : model.GetRoots()) {
            onFace = onFace && r.faceId >= 0;
        }
        Check(onFace, "auto-tube: every root names its scalp face");
        // The pure exact-region twin agrees with the model refill. The
        // legacy face-list stream is deliberately too coarse for two small
        // polygons on the same quad.
        PomadeModel::TubeSnapshot shot = model.Snapshot();
        PomadeModel::GraphSnapshot graphShot = model.SnapshotGraph();
        PomadeScalpGraph graph;
        PomadeRegionLoops loops;
        std::string loopErr;
        bool const loopsOk = graph.Restore(*model.GetScalp(), graphShot.nodes,
                                           graphShot.edges, graphShot.linked,
                                           &loopErr) &&
                             PomadeFlattenLoops(graph, &loops, &loopErr);
        PomadeGuideSet guides = loopsOk
                                   ? PomadeGenerateGuidesOnRegionForTube(
                                         shot, *model.GetScalp(), loops, 0, 0)
                                   : PomadeGuideSet();
        Check(guides.points == model.GetGuides().points,
              "auto-tube: model + pure exact-region fill agree bit-exactly");
        // Root snap + match-surface run against the live scalp.
        Check(model.SnapRootToScalp(), "auto-tube: root snap runs");
        model.GetCenterCV(0, &x0, &y0, &z0);
        Check(Near(y0, 0.0f, 1e-3f),
              "auto-tube: root snap seats the root on the scalp");
        Check(model.MatchSurface(), "auto-tube: match-surface runs");
    }
    Check(!model.BuildTubeFromRegion(0, 1, 8, 3.0f),
          "auto-tube: centerCount < 2 fails");
    Check(!model.BuildTubeFromRegion(0, 4, 2, 3.0f),
          "auto-tube: ringVerts < 3 fails");
}

void CheckCApi()
{
    PomadeModelContext *ctx = nullptr;
    Check(Pomade_Create(&ctx) == POMADE_OK && ctx != nullptr,
          "C ABI: context creates");
    Check(Pomade_BuildTestTube(ctx, 5, 8, 0.5f, 4.0f) == POMADE_OK,
          "C ABI: the test tube builds");
    float rawCenter[3] = {0, 0, 0};
    float handleCenter[3] = {0, 0, 0};
    Check(Pomade_GetTubeCenterCV(ctx, 0, 2, rawCenter) == POMADE_OK &&
              Pomade_GetTubeCenterHandle(ctx, 0, 2, handleCenter) == POMADE_OK &&
              Near(rawCenter[0], handleCenter[0]) &&
              Near(rawCenter[1], handleCenter[1]) &&
              Near(rawCenter[2], handleCenter[2]),
          "C ABI: symmetric center core reads without changing raw data");
    Check(Pomade_MoveCenterCV(ctx, 2, 1.0f, 0.0f, 0.0f) == POMADE_OK,
          "C ABI: center CV moves");
    float xyz[3] = {0, 0, 0};
    Check(Pomade_GetCenterCV(ctx, 2, xyz) == POMADE_OK &&
              Near(xyz[0], 1.0f) && Near(xyz[1], 2.0f),
          "C ABI: center CV reads");
    Check(Pomade_GetCenterCVCount(ctx) == 5, "C ABI: center census");
    Check(Pomade_InsertCenterCV(ctx, 2) == POMADE_OK &&
              Pomade_GetCenterCVCount(ctx) == 6,
          "C ABI: center CV inserts");
    Check(Pomade_DeleteCenterCV(ctx, 2) == POMADE_OK &&
              Pomade_GetCenterCVCount(ctx) == 5,
          "C ABI: center CV deletes");
    Check(Pomade_SetTubeLength(ctx, 8.0f) == POMADE_OK,
          "C ABI: tube length sets");
    Check(Pomade_GetSectionCount(ctx) == 5, "C ABI: section census");
    float uv[16] = {0};
    float t = 0, sc = 0, tw = 0;
    int count = 0;
    Check(Pomade_GetSection(ctx, 0, &t, uv, 16, &count, &sc, &tw) ==
                  POMADE_OK &&
              count == 8 && Near(uv[0], 0.5f),
          "C ABI: sections read");
    Check(Pomade_GetSection(ctx, 0, nullptr, uv, 4, &count, nullptr,
                           nullptr) == POMADE_ERROR,
          "C ABI: short section outputs fail");
    Check(Pomade_MoveSectionRing(ctx, 1, 0.25f, 0.0f) == POMADE_OK &&
              Pomade_ScaleSectionRing(ctx, 1, 2.0f) == POMADE_OK &&
              Pomade_TwistSectionRing(ctx, 1, 0.5f) == POMADE_OK &&
              Pomade_MoveSectionCV(ctx, 1, 3, 0.1f, 0.0f) == POMADE_OK,
          "C ABI: ring + CV gizmos run");
    int rings = 0;
    Check(Pomade_AddSectionRing(ctx, 0.5f, &rings) == POMADE_OK && rings == 6,
          "C ABI: rings insert");
    Check(Pomade_CopySectionRing(ctx, 1, 2) == POMADE_OK &&
              Pomade_RemoveSectionRing(ctx, 2) == POMADE_OK &&
              Pomade_GetSectionCount(ctx) == 5,
          "C ABI: rings copy + remove");
    Check(Pomade_SetSoftSelection(ctx, 0.5f, 0.5f) == POMADE_OK,
          "C ABI: soft selection sets");
    float c = 0, r = 0;
    Check(Pomade_GetSoftSelection(ctx, &c, &r) == POMADE_OK &&
              Near(c, 0.5f) && Near(r, 0.5f),
          "C ABI: soft selection reads");
    Check(Pomade_RelaxCenter(ctx, 0.5f, 2) == POMADE_OK,
          "C ABI: relax runs");
    Check(Pomade_SetDisplaySegments(ctx, 2) == POMADE_OK &&
              Pomade_GetDisplaySegments(ctx) == 2,
          "C ABI: display segments round-trip");
    Check(Pomade_SetTubeRegionId(ctx, 3) == POMADE_OK &&
              Pomade_GetTubeRegionId(ctx) == 3,
          "C ABI: region id round-trips");
    Check(Pomade_SetFillParams(ctx, 16.0f, 8, 7, 0.25f, nullptr, 0) ==
              POMADE_OK,
          "C ABI: fill params set");
    Check(Pomade_SetFillParams(ctx, 16.0f, 8, 7, 2.0f, nullptr, 0) ==
              POMADE_ERROR,
          "C ABI: fill bias validates");
    float density = 0, edgeBias = 0;
    int cvCount = 0, seed = 0, floats = -1;
    Check(Pomade_GetFillParams(ctx, &density, &cvCount, &seed, &edgeBias,
                              nullptr, 0, &floats) == POMADE_OK &&
              Near(density, 16.0f) && cvCount == 8 && seed == 7 &&
              Near(edgeBias, 0.25f) && floats == 0,
          "C ABI: fill params read");
    Check(Pomade_GetPreviewFraction(ctx) == 0.25f, "C ABI: preview default");
    Check(Pomade_SetPreviewFraction(ctx, 0.5f) == POMADE_OK &&
              Pomade_GetPreviewFraction(ctx) == 0.5f,
          "C ABI: preview fraction round-trips");
    Check(Pomade_SetFreezeRoots(ctx, 1) == POMADE_OK &&
              Pomade_GetFreezeRoots(ctx) == 1,
          "C ABI: freeze roots round-trips");
    Check(Pomade_SetFreezeRoots(ctx, 0) == POMADE_OK,
          "C ABI: freeze clears");
    Check(Pomade_RefillGuides(ctx, 1.0f) == POMADE_OK,
          "C ABI: refill runs");
    int guides = 0, cv = 0;
    Check(Pomade_GetGuideCounts(ctx, &guides, &cv) == POMADE_OK &&
              guides == 16 && cv == 8,
          "C ABI: guide census reads");
    float pts[16 * 8 * 3];
    int counts[16];
    int got = 0;
    Check(Pomade_ReadGuidePreview(ctx, pts, 16 * 8 * 3, counts, 16, &got) ==
                  POMADE_OK &&
              got == 16,
          "C ABI: guide preview reads");
    Check(Pomade_ReadGuidePreview(ctx, pts, 8, counts, 16, &got) ==
              POMADE_ERROR,
          "C ABI: short preview outputs fail");
    int faceIds[16];
    float rpos[16 * 3], rru[16 * 2];
    Check(Pomade_ReadGuideRoots(ctx, faceIds, rpos, rru, 16, &got) ==
                  POMADE_OK &&
              got == 16,
          "C ABI: guide roots read");
    float vp[16] = {0};
    vp[0] = vp[5] = vp[10] = vp[15] = 1.0f;
    int hit = 0, index = -1, sub = -1;
    unsigned kind = 0;
    float dist = 0, depth = 0;
    Check(Pomade_Pick(ctx, vp, 800, 600, 400.0f, 300.0f, 5.0f, 0x1Fu, &hit,
                     &kind, &index, &sub, &dist, &depth) == POMADE_OK,
          "C ABI: pick runs");
    Check(Pomade_Pick(nullptr, vp, 800, 600, 400.0f, 300.0f, 5.0f, 0x1Fu,
                     &hit, &kind, &index, &sub, &dist, &depth) ==
              POMADE_ERROR,
          "C ABI: null pick fails");
    Check(Pomade_Destroy(ctx) == POMADE_OK, "C ABI: context destroys");
}

void CheckProbes()
{
    using namespace usdGenPomade;
    // TN-1 probe (informational): a center move + preview refill on a
    // 200-section tube; the gate runs on the reference scene, not here.
    {
        PomadeModel model;
        model.BuildTestTube();
        for (int i = 0; i < 195; ++i) {
            model.InsertCenterCV(2);
        }
        PomadeModel::FillParams params;
        params.density = 4000.0f;
        params.cvCount = 16;
        Check(model.SetFillParams(params), "probe: scale params set");
        auto t0 = std::chrono::steady_clock::now();
        bool ok = model.MoveCenterCV(100, 0.1f, 0.0f, 0.0f);
        ok = ok && model.RefillGuides(0.25f);
        auto t1 = std::chrono::steady_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("info: move + 25%% refill on 200 CVs / %d guides: "
                    "%.2f ms\n",
                    model.GetGuides().guideCount, ms);
        Check(ok, "probe: the scaled move + refill completes");
    }
    // TN-2 probe (informational): CPU pick over 770K tube-vert
    // candidates; the gate is the K11 device lane below (TN-2) plus the
    // production wiring gate (pick-scale).
    {
        int const n = 770000;
        std::vector<float> verts(size_t(n) * 3);
        for (int i = 0; i < n; ++i) {
            verts[size_t(i) * 3 + 0] = float(i % 1000) / 500.0f - 1.0f;
            verts[size_t(i) * 3 + 1] = float((i / 1000) % 770) / 385.0f -
                                       1.0f;
            verts[size_t(i) * 3 + 2] = 0.0f;
        }
        float vp[16] = {0};
        vp[0] = vp[5] = vp[10] = vp[15] = 1.0f;
        PomadePickSets sets;
        sets.tubeVerts = verts.data();
        sets.tubeVertCount = n;
        auto t0 = std::chrono::steady_clock::now();
        PomadePickHit hit = PomadePickCpu(sets, PomadePick_TubeVert, vp, 800,
                                        600, 400.0f, 300.0f, 5.0f);
        auto t1 = std::chrono::steady_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("info: CPU pick over %d candidates: %.2f ms (hit=%d)\n",
                    n, ms, hit.hit ? 1 : 0);
        Check(true, "probe: the scaled CPU pick completes");
    }
}

// The K9 chart of `tube` at t, built exactly as
// PomadeBuildGuideMaterialBindingsCpu builds it (ring minus center on the
// frame's normal/binormal), as a section PomadeTriangulateSectionSlotsCpu
// takes.
bool MaterialChartAt(PomadeTubeDesc const &tube,
                     std::vector<PomadeFrame> const &frames, float t,
                     PomadeTubeSection *chart)
{
    using namespace usdGenPomade;
    std::vector<float> ring;
    std::string err;
    float c[3] = {0.0f, 0.0f, 0.0f};
    PomadeFrame f;
    if (!PomadeSampleTubeRingCpu(tube, frames, t, &ring, &err) ||
        !PomadeSampleCenterCpu(tube, frames, t, &c[0], &c[1], &c[2], &f,
                              &err) ||
        ring.size() != size_t(tube.ringVerts) * 3) {
        return false;
    }
    chart->t = t;
    chart->scale = 1.0f;
    chart->twist = 0.0f;
    chart->u.resize(size_t(tube.ringVerts));
    chart->v.resize(size_t(tube.ringVerts));
    for (int i = 0; i < tube.ringVerts; ++i) {
        float const dx = ring[size_t(i) * 3 + 0] - c[0];
        float const dy = ring[size_t(i) * 3 + 1] - c[1];
        float const dz = ring[size_t(i) * 3 + 2] - c[2];
        chart->u[size_t(i)] = dx * f.nx + dy * f.ny + dz * f.nz;
        chart->v[size_t(i)] = dx * f.bx + dy * f.by + dz * f.bz;
    }
    return true;
}

// Summed squared slot distance between two same-size rings after cyclic
// shift k of `b`, both taken about their own vertex means and normalised
// by their mean radius (the shape K14's alignment compares).
double ShiftCost(PomadeTubeSection const &a, PomadeTubeSection const &b, int k)
{
    auto normalised = [](PomadeTubeSection const &s, std::vector<double> *u,
                         std::vector<double> *v) {
        size_t const n = s.u.size();
        double mu = 0.0, mv = 0.0;
        for (size_t i = 0; i < n; ++i) {
            mu += s.u[i];
            mv += s.v[i];
        }
        mu /= double(n);
        mv /= double(n);
        double r = 0.0;
        for (size_t i = 0; i < n; ++i) {
            r += std::hypot(s.u[i] - mu, s.v[i] - mv);
        }
        r = r > 0.0 ? r / double(n) : 1.0;
        u->resize(n);
        v->resize(n);
        for (size_t i = 0; i < n; ++i) {
            (*u)[i] = (s.u[i] - mu) / r;
            (*v)[i] = (s.v[i] - mv) / r;
        }
    };
    std::vector<double> au, av, bu, bv;
    normalised(a, &au, &av);
    normalised(b, &bu, &bv);
    size_t const n = au.size();
    double cost = 0.0;
    for (size_t i = 0; i < n; ++i) {
        size_t const j = (i + size_t(k)) % n;
        cost += (bu[j] - au[i]) * (bu[j] - au[i]) +
                (bv[j] - av[i]) * (bv[j] - av[i]);
    }
    return cost;
}

// BUG-SUBDIVIDE-RING-ALIGNMENT (the testPomadeSoak regression from 28a4c0e).
// A K7-merged parent carries a different ring at every station, so each
// child's Voronoi cell starts on a different parent corner per section.
// K14 must keep the child's slot numbering aligned section to section --
// K5 interpolates per slot, and a rotated neighbour twists the ring into a
// figure-eight the concave material triangulator (rightly) refuses at a
// fill station -- the model must refuse a split it could not fill, and a
// refill that skips a tube must say so instead of shedding its guides.
void CheckSubdivideRingAlignment()
{
    using namespace usdGenPomade;
    // Parent: 5 CVs along +Y, a 16-slot ring at t = 0, .25, .5, .75, 1.
    // Each station turns the ring geometry a further 45 degrees (slot
    // order fixed, so the parent itself only twists smoothly) and breathes
    // its radius: the per-station drift a merge of edited children leaves.
    // (Without the K14 alignment the two slot checks below fail on it.)
    PomadeTubeDesc parent;
    parent.tubeId = 1;
    parent.level = 1;
    parent.centerX = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    parent.centerY = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    parent.centerZ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    parent.ringVerts = 16;
    float const twoPi = 6.28318530717958647692f;
    for (int k = 0; k < 5; ++k) {
        PomadeTubeSection s;
        s.t = float(k) * 0.25f;
        float const turn = float(k) * (45.0f / 360.0f) * twoPi;
        float const radius = 0.5f * (1.0f + 0.12f * std::sin(float(k)));
        for (int i = 0; i < parent.ringVerts; ++i) {
            float const a = twoPi * float(i) / float(parent.ringVerts) + turn;
            s.u.push_back(radius * std::cos(a));
            s.v.push_back(0.8f * radius * std::sin(a));
        }
        parent.sections.push_back(std::move(s));
    }
    std::vector<PomadeFrame> const parentFrames = FramesFor(parent);
    PomadeSubdivideDesc params;
    params.count = 4;
    params.seed = 3;
    params.splitMode = PomadeSplit_KMeans;
    std::vector<PomadeTubeDesc> kids;
    std::string err;
    Check(PomadeSubdivideTubeCpu(parent, parentFrames, params, &kids, &err) &&
              kids.size() == 4,
          "ring alignment: the drifting parent subdivides (" + err + ")");

    int const cvCount = 11;
    std::string misaligned, jumped, untriangulable, unbindable;
    for (PomadeTubeDesc const &kid : kids) {
        std::string const who = "child " + std::to_string(kid.tubeId);
        // Slot numbering: shift 0 is the best cyclic match between every
        // pair of adjacent sections, and slot 0 itself moves continuously.
        for (size_t s = 0; s + 1 < kid.sections.size(); ++s) {
            PomadeTubeSection const &a = kid.sections[s];
            PomadeTubeSection const &b = kid.sections[s + 1];
            double best = ShiftCost(a, b, 0);
            for (int k = 1; k < kid.ringVerts; ++k) {
                best = std::min(best, ShiftCost(a, b, k));
            }
            if (ShiftCost(a, b, 0) > best + 1e-4 * double(kid.ringVerts)) {
                misaligned += " " + who + " s" + std::to_string(s);
            }
            float const radius = PomadeSectionMeanRadius(a);
            if (std::hypot(b.u[0] - a.u[0], b.v[0] - a.v[0]) >
                0.5f * radius) {
                jumped += " " + who + " s" + std::to_string(s);
            }
        }
        // Every ring K9 triangulates: each section t and each fill station.
        std::vector<PomadeFrame> frames;
        if (!PomadeTubeFramesCpu(kid, &frames, &err)) {
            untriangulable += " " + who + " (frames: " + err + ")";
            continue;
        }
        std::vector<float> stations;
        for (PomadeTubeSection const &s : kid.sections) {
            stations.push_back(s.t);
        }
        float const t0 = kid.sections.front().t;
        float const t1 = kid.sections.back().t;
        for (int c = 1; c + 1 < cvCount; ++c) {
            stations.push_back(t0 + (t1 - t0) * float(c) / float(cvCount - 1));
        }
        for (float t : stations) {
            PomadeTubeSection chart;
            std::vector<std::array<int, 3>> triangles;
            std::string terr;
            if (!MaterialChartAt(kid, frames, t, &chart) ||
                !PomadeTriangulateSectionSlotsCpu(chart, &triangles, &terr)) {
                untriangulable += " " + who + " t=" + std::to_string(t);
            }
        }
        // And the real consumer: K9 binds a root disc through every station.
        float const rootCenter[3] = {kid.centerX[0], kid.centerY[0],
                                     kid.centerZ[0]};
        std::vector<PomadeGuideRoot> roots;
        PomadeFillDesc fill;
        fill.cvCount = cvCount;
        std::vector<PomadeGuideMaterialBinding> bindings;
        if (!PomadeRootSampleDiscCpu(kid.tubeId, 7,
                                    PomadeSectionMeanRadius(kid.sections[0]),
                                    rootCenter, frames[0], 24, &roots, 0,
                                    &err) ||
            !PomadeBuildGuideMaterialBindingsCpu(kid, frames, roots, fill,
                                                &bindings, &err)) {
            unbindable += " " + who + " (" + err + ")";
        }
        // K14 provenance rotates with the slots: two inherited corners of
        // one section sit as far apart in the child as in the parent.
        auto const &bb = kid.inheritedBoundaryBindings;
        for (size_t i = 0; i + 1 < bb.size(); ++i) {
            PomadeParentBoundaryBinding const &p = bb[i];
            PomadeParentBoundaryBinding const &q = bb[i + 1];
            if (p.section != q.section) {
                continue;
            }
            PomadeTubeSection const &cs = kid.sections[size_t(p.section)];
            PomadeTubeSection const &ps = parent.sections[size_t(p.section)];
            float const childD = std::hypot(
                cs.u[size_t(p.childSlot)] - cs.u[size_t(q.childSlot)],
                cs.v[size_t(p.childSlot)] - cs.v[size_t(q.childSlot)]);
            float const parentD = std::hypot(
                ps.u[size_t(p.parentSlot)] - ps.u[size_t(q.parentSlot)],
                ps.v[size_t(p.parentSlot)] - ps.v[size_t(q.parentSlot)]);
            if (!Near(childD, parentD, 1e-4f)) {
                misaligned += " " + who + " binding s" +
                              std::to_string(p.section);
            }
        }
    }
    Check(misaligned.empty(),
          "ring alignment: child slots stay aligned section to section and "
          "K14 bindings follow them" + misaligned);
    Check(jumped.empty(),
          "ring alignment: child slot 0 moves continuously between adjacent "
          "sections" + jumped);
    Check(untriangulable.empty(),
          "ring alignment: every child ring triangulates at every section t "
          "and fill station" + untriangulable);
    Check(unbindable.empty(),
          "ring alignment: K9 binds every child's roots through every "
          "station" + unbindable);

    // Deterministic: hydrate re-derives the stored children bit-exactly.
    std::vector<PomadeTubeDesc> again;
    bool same = PomadeSubdivideTubeCpu(parent, parentFrames, params, &again,
                                      &err) &&
                again.size() == kids.size();
    for (size_t i = 0; same && i < kids.size(); ++i) {
        same = SameSections(kids[i].sections, again[i].sections) &&
               kids[i].inheritedBoundaryBindings.size() ==
                   again[i].inheritedBoundaryBindings.size();
    }
    Check(same, "ring alignment: the aligned split is deterministic");

    // The model: a refill that skips a tube reports it, and a leaf whose
    // rings K9 cannot triangulate cannot be split into more of them.
    PomadeModel model;
    Check(model.BuildTestTube(), "refill drops: the test tube builds");
    std::vector<int> leaves;
    Check(model.SubdivideTube(0, 4, "kmeans", 2, &leaves) &&
              leaves.size() == 4,
          "refill drops: the tube subdivides into four children");
    Check(model.RefillGuides(1.0f) && model.RefillDrops().empty(),
          "refill drops: a healthy refill reports no dropped tube");
    int const bent = leaves.empty() ? -1 : leaves[0];
    // Bow-tie every non-root section of one child: drag slot 0 through
    // the ring past its opposite corner.
    bool edited = bent >= 0;
    int const nSec = model.GetTubeSectionCount(bent);
    for (int s = 1; edited && s < nSec; ++s) {
        PomadeTubeSection sec;
        edited = model.GetTubeSection(bent, s, &sec) && sec.u.size() >= 4;
        if (edited) {
            size_t const far = sec.u.size() / 2;
            edited = model.MoveTubeSectionCV(
                bent, s, 0, 1.6f * (sec.u[far] - sec.u[0]),
                1.6f * (sec.v[far] - sec.v[0]));
        }
    }
    Check(edited, "refill drops: one child's sections bow-tie (" +
                      std::string(model.GetDiagnostic()) + ")");
    std::vector<std::pair<int, std::string>> drops;
    Check(model.RefillGuides(1.0f),
          "refill drops: the siblings still fill (" +
              std::string(model.GetDiagnostic()) + ")");
    drops = model.RefillDrops();
    Check(drops.size() == 1 && drops[0].first == bent &&
              !drops[0].second.empty(),
          "refill drops: the bow-tied child is reported with its reason");
    int const tubesBefore = model.GetTubeCount();
    std::vector<int> grandkids;
    bool const split = model.SubdivideTube(bent, 2, "kmeans", 1, &grandkids);
    std::string const splitWhy = model.GetDiagnostic();
    Check(!split && model.GetTubeCount() == tubesBefore,
          "refill drops: splitting the bow-tied child is refused, model "
          "untouched (" + splitWhy + ")");

    // The C ABI reads the same list (two-call probe), empty after a clean
    // refill, and rejects a null count.
    PomadeModelContext *ctx = nullptr;
    Check(Pomade_Create(&ctx) == POMADE_OK &&
              Pomade_BuildTestTube(ctx, 5, 8, 0.5f, 4.0f) == POMADE_OK &&
              Pomade_RefillGuides(ctx, 1.0f) == POMADE_OK,
          "refill drops ABI: a test tube refills");
    int dropCount = -1;
    Check(Pomade_ReadRefillDrops(ctx, nullptr, 0, &dropCount) == POMADE_OK &&
              dropCount == 0 &&
              std::string(Pomade_GetRefillDropReason(ctx, 0)).empty(),
          "refill drops ABI: a clean refill reports no dropped tube");
    Check(Pomade_ReadRefillDrops(ctx, nullptr, 0, nullptr) == POMADE_ERROR,
          "refill drops ABI: a null count is an error");
    Pomade_Destroy(ctx);
}

#ifdef USDGEN_POMADE_HAS_CUDA
bool HaveCudaDevice()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void CheckCudaParity()
{
    using namespace usdGenPomade;
    if (!HaveCudaDevice()) {
        std::printf("skip: no CUDA device for the TN-6 parity checks\n");
        return;
    }
    Check(sizeof(PomadeDeviceRoot) == sizeof(int) + 7 * sizeof(float),
          "TN-6: the device root is tightly packed");
    char err[256] = {0};
    PomadeTubeDesc tube = MakeTube();
    tube.sections[1].twist = 0.3f;
    tube.sections[1].scale = 1.5f;
    // Uneven knots exercise per-span sampling on the GPU as well.
    tube.sections.push_back(tube.sections[0]);
    tube.sections[1].t = 0.23f;
    tube.sections[2].t = 1.0f;
    int const nCv = int(tube.centerX.size());
    int const nSec = int(tube.sections.size());
    int const rv = tube.ringVerts;
    cudaStream_t stream = nullptr;
    Check(cudaStreamCreate(&stream) == cudaSuccess,
          "TN-6: a stream creates");
    // Upload the center + sections once for K4/K5/K9.
    float *dCx = nullptr, *dCy = nullptr, *dCz = nullptr;
    PomadeFrame *dFrames = nullptr;
    float *dSecT = nullptr, *dSecU = nullptr, *dSecV = nullptr,
          *dSecS = nullptr, *dSecTw = nullptr;
    cudaMalloc(&dCx, sizeof(float) * size_t(nCv));
    cudaMalloc(&dCy, sizeof(float) * size_t(nCv));
    cudaMalloc(&dCz, sizeof(float) * size_t(nCv));
    cudaMalloc(&dFrames, sizeof(PomadeFrame) * size_t(nCv));
    cudaMalloc(&dSecT, sizeof(float) * size_t(nSec));
    cudaMalloc(&dSecU, sizeof(float) * size_t(nSec) * size_t(rv));
    cudaMalloc(&dSecV, sizeof(float) * size_t(nSec) * size_t(rv));
    cudaMalloc(&dSecS, sizeof(float) * size_t(nSec));
    cudaMalloc(&dSecTw, sizeof(float) * size_t(nSec));
    cudaMemcpy(dCx, tube.centerX.data(), sizeof(float) * size_t(nCv),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dCy, tube.centerY.data(), sizeof(float) * size_t(nCv),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dCz, tube.centerZ.data(), sizeof(float) * size_t(nCv),
               cudaMemcpyHostToDevice);
    // (Sized by assign throughout: `X(size_t(n))` parses as a function
    // declaration — the most vexing parse.)
    std::vector<float> secT, secU, secV, secS, secTw;
    secT.assign(size_t(nSec), 0.0f);
    secU.assign(size_t(nSec) * size_t(rv), 0.0f);
    secV.assign(size_t(nSec) * size_t(rv), 0.0f);
    secS.assign(size_t(nSec), 0.0f);
    secTw.assign(size_t(nSec), 0.0f);
    for (int i = 0; i < nSec; ++i) {
        secT[size_t(i)] = tube.sections[size_t(i)].t;
        secS[size_t(i)] = tube.sections[size_t(i)].scale;
        secTw[size_t(i)] = tube.sections[size_t(i)].twist;
        for (int k = 0; k < rv; ++k) {
            secU[size_t(i) * size_t(rv) + size_t(k)] =
                tube.sections[size_t(i)].u[size_t(k)];
            secV[size_t(i) * size_t(rv) + size_t(k)] =
                tube.sections[size_t(i)].v[size_t(k)];
        }
    }
    cudaMemcpy(dSecT, secT.data(), sizeof(float) * size_t(nSec),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dSecU, secU.data(), sizeof(float) * secU.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dSecV, secV.data(), sizeof(float) * secV.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dSecS, secS.data(), sizeof(float) * size_t(nSec),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dSecTw, secTw.data(), sizeof(float) * size_t(nSec),
               cudaMemcpyHostToDevice);
    // K4: bit-exact (arithmetic only).
    std::vector<PomadeFrame> hFrames = FramesFor(tube);
    Check(PomadeLaunchCenterFrames(dCx, dCy, dCz, nCv, dFrames, stream, err,
                                  sizeof(err)),
          "TN-6: K4 launches");
    std::vector<PomadeFrame> dFramesOut;
    dFramesOut.resize(size_t(nCv));
    cudaMemcpy(dFramesOut.data(), dFrames, sizeof(PomadeFrame) * size_t(nCv),
               cudaMemcpyDeviceToHost);
    bool k4 = true;
    for (int i = 0; i < nCv && k4; ++i) {
        float const *a = &hFrames[size_t(i)].tx;
        float const *b = &dFramesOut[size_t(i)].tx;
        for (int k = 0; k < 9; ++k) {
            k4 = k4 && a[k] == b[k];
        }
    }
    Check(k4, "TN-6: K4 frames match bit-exactly");
    // K5: tolerance (twist trigonometry).
    int const seg = 8;
    int const nRings = (nSec - 1) * seg + 1;
    float *dPos = nullptr, *dNrm = nullptr, *dRingT = nullptr,
          *dRootOffsets = nullptr;
    tube.rootSurfaceOffsets.resize(size_t(rv) * 3);
    for (size_t i = 0; i < tube.rootSurfaceOffsets.size(); ++i)
        tube.rootSurfaceOffsets[i] = -0.025f * float(i % 7);
    cudaMalloc(&dRootOffsets, sizeof(float) * tube.rootSurfaceOffsets.size());
    cudaMemcpy(dRootOffsets, tube.rootSurfaceOffsets.data(),
               sizeof(float) * tube.rootSurfaceOffsets.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dPos, sizeof(float) * size_t(nRings) * size_t(rv) * 3);
    cudaMalloc(&dNrm, sizeof(float) * size_t(nRings) * size_t(rv) * 3);
    cudaMalloc(&dRingT, sizeof(float) * size_t(nRings));
    Check(PomadeLaunchTubeTessellate(
              dCx, dCy, dCz, nCv, dFrames, dSecT, dSecU, dSecV, dSecS,
              dSecTw, nSec, rv, seg, dPos, dNrm, dRingT, stream, err,
              sizeof(err), dRootOffsets),
          "TN-6: K5 launches with root attachment and end holding rows");
    std::vector<float> hPos, hNrm, hRingT;
    std::string herr;
    Check(PomadeTessellateCpu(tube, hFrames, seg, &hPos, &hNrm, &hRingT,
                             &herr),
          "TN-6: K5 CPU twin runs");
    std::vector<float> dPosOut(hPos.size()), dNrmOut(hNrm.size());
    std::vector<float> dRingTOut;
    dRingTOut.resize(size_t(nRings));
    cudaMemcpy(dPosOut.data(), dPos, sizeof(float) * hPos.size(),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dNrmOut.data(), dNrm, sizeof(float) * hNrm.size(),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dRingTOut.data(), dRingT, sizeof(float) * size_t(nRings),
               cudaMemcpyDeviceToHost);
    float worstP = 0, worstN = 0;
    for (size_t i = 0; i < hPos.size(); ++i) {
        worstP = std::max(worstP, std::fabs(hPos[i] - dPosOut[i]));
        worstN = std::max(worstN, std::fabs(hNrm[i] - dNrmOut[i]));
    }
    std::printf("info: K5 worst |dp|=%.3g worst |dn|=%.3g\n", worstP,
                worstN);
    Check(worstP < 1e-4f && worstN < 1e-4f,
          "TN-6: K5 tessellation matches within 1e-4");
    Check(hRingT == dRingTOut, "TN-6: authored and holding row parameters match");
    cudaFree(dRootOffsets);
    // K8 disc: tolerance (sqrt/trigonometry in the draws; the rejection
    // boundary is a float comparison, so borderline candidates could in
    // principle diverge — with 32 roots the chance is ~1e-5 and the
    // check below would name the seed).
    int const roots = 32;
    PomadeDeviceRoot *dRoots = nullptr;
    cudaMalloc(&dRoots, sizeof(PomadeDeviceRoot) * size_t(roots));
    float const rootRadius =
        PomadeSectionMeanRadius(tube.sections.front());
    float const rc[3] = {tube.centerX[0], tube.centerY[0], tube.centerZ[0]};
    Check(PomadeLaunchRootSampleDisc(0, 7, rootRadius, rc[0], rc[1], rc[2],
                                    hFrames[0], roots, dRoots, 0, stream,
                                    err, sizeof(err)),
          "TN-6: K8 launches");
    std::vector<PomadeGuideRoot> hRoots;
    Check(PomadeRootSampleDiscCpu(0, 7, rootRadius, rc, hFrames[0], roots,
                                 &hRoots, 0, &herr),
          "TN-6: K8 CPU twin runs");
    std::vector<PomadeDeviceRoot> dRootsOut;
    dRootsOut.resize(size_t(roots));
    cudaMemcpy(dRootsOut.data(), dRoots,
               sizeof(PomadeDeviceRoot) * size_t(roots),
               cudaMemcpyDeviceToHost);
    float worstR = 0;
    for (int i = 0; i < roots; ++i) {
        worstR = std::max(
            worstR, std::fabs(hRoots[size_t(i)].ru - dRootsOut[size_t(i)].ru));
        worstR = std::max(
            worstR, std::fabs(hRoots[size_t(i)].rv - dRootsOut[size_t(i)].rv));
    }
    std::printf("info: K8 worst |dr|=%.3g\n", worstR);
    Check(worstR < 1e-3f, "TN-6: K8 roots match within 1e-3");
    // K9: full K5 material bindings, including U/V, scale and twist.
    int const cvCount = 8;
    PomadeFillDesc fill;
    fill.cvCount = cvCount;
    fill.edgeBias = 0.25f;
    std::vector<PomadeGuideMaterialBinding> hBindings;
    Check(PomadeBuildGuideMaterialBindingsCpu(tube, hFrames, hRoots, fill,
                                             &hBindings, &herr),
          "TN-6: K9 material binds build");
    PomadeGuideMaterialBinding *dBindings = nullptr;
    cudaMalloc(&dBindings,
               sizeof(PomadeGuideMaterialBinding) * hBindings.size());
    cudaMemcpy(dBindings, hBindings.data(),
               sizeof(PomadeGuideMaterialBinding) * hBindings.size(),
               cudaMemcpyHostToDevice);
    float *dGuides = nullptr, *dLens = nullptr;
    cudaMalloc(&dGuides, sizeof(float) * size_t(roots) * size_t(cvCount) * 3);
    cudaMalloc(&dLens, sizeof(float) * size_t(roots));
    Check(PomadeLaunchGuideFill(
              dCx, dCy, dCz, nCv, dFrames, dSecT, dSecU, dSecV, dSecS,
              dSecTw, nSec, rv, dRoots, dBindings, roots, fill.edgeBias,
              nullptr, 0, dGuides, dLens, cvCount, stream, err, sizeof(err)),
          "TN-6: K9 launches");
    std::vector<float> hGuides, hLens;
    Check(PomadeGuideFillCpu(tube, hFrames, hRoots, fill, &hGuides, &hLens,
                            &herr),
          "TN-6: K9 CPU twin runs");
    std::vector<float> dGuidesOut(hGuides.size());
    std::vector<float> dLensOut;
    dLensOut.resize(size_t(roots));
    cudaMemcpy(dGuidesOut.data(), dGuides, sizeof(float) * hGuides.size(),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dLensOut.data(), dLens, sizeof(float) * size_t(roots),
               cudaMemcpyDeviceToHost);
    float worstG = 0;
    for (size_t i = 0; i < hGuides.size(); ++i) {
        worstG = std::max(worstG, std::fabs(hGuides[i] - dGuidesOut[i]));
    }
    std::printf("info: K9 worst |dg|=%.3g\n", worstG);
    // NOTE: K9 parity rides on K8 parity (the device lane fills the
    // device roots, the CPU twin the host roots); a K8 divergence would
    // cascade here, so this check is only meaningful with K8 green.
    Check(worstG < 1e-2f, "TN-6: K9 guides match within 1e-2");
    // K10: tolerance (sqrt/division chains).
    float *dIn = nullptr;
    int *dOff = nullptr, *dCnt = nullptr, *dOutC = nullptr;
    double *dOutF = nullptr;
    float *dOut = nullptr;
    int const k10Guides = 4;
    cudaMalloc(&dIn, sizeof(float) * size_t(k10Guides) * size_t(cvCount) * 3);
    cudaMalloc(&dOff, sizeof(int) * size_t(k10Guides));
    cudaMalloc(&dCnt, sizeof(int) * size_t(k10Guides));
    cudaMalloc(&dOut, sizeof(float) * size_t(k10Guides) * 5 * 3);
    cudaMalloc(&dOutC, sizeof(int) * size_t(k10Guides));
    cudaMalloc(&dOutF, sizeof(double) * size_t(k10Guides) * 16);
    std::vector<float> k10In(size_t(k10Guides) * size_t(cvCount) * 3);
    std::vector<int> k10Off, k10Cnt;
    k10Off.resize(size_t(k10Guides));
    k10Cnt.resize(size_t(k10Guides));
    for (int g = 0; g < k10Guides; ++g) {
        k10Off[size_t(g)] = g * cvCount;
        k10Cnt[size_t(g)] = cvCount;
        for (int c = 0; c < cvCount; ++c) {
            k10In[(size_t(g) * size_t(cvCount) + size_t(c)) * 3 + 0] =
                float(g);
            k10In[(size_t(g) * size_t(cvCount) + size_t(c)) * 3 + 1] =
                float(c) * float(c) / float(cvCount);
            k10In[(size_t(g) * size_t(cvCount) + size_t(c)) * 3 + 2] = 0.0f;
        }
    }
    cudaMemcpy(dIn, k10In.data(), sizeof(float) * k10In.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dOff, k10Off.data(), sizeof(int) * size_t(k10Guides),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dCnt, k10Cnt.data(), sizeof(int) * size_t(k10Guides),
               cudaMemcpyHostToDevice);
    float rootDirs[9] = {1, 0, 0, 0, 0, 1, 0, 1, 0};
    Check(PomadeLaunchGuideResample(dIn, dOff, dCnt, k10Guides, 5, rootDirs,
                                   dOut, dOutC, dOutF, stream, err,
                                   sizeof(err)),
          "TN-6: K10 launches");
    std::vector<float> hOut;
    std::vector<int> hOutC;
    std::vector<double> hOutF;
    Check(PomadeGuideResampleCpu(k10In.data(), k10Cnt.data(), k10Guides, 5,
                                rootDirs, &hOut, &hOutC, &hOutF, &herr),
          "TN-6: K10 CPU twin runs");
    std::vector<float> dOutOut(hOut.size());
    cudaMemcpy(dOutOut.data(), dOut, sizeof(float) * hOut.size(),
               cudaMemcpyDeviceToHost);
    float worstRe = 0;
    for (size_t i = 0; i < hOut.size(); ++i) {
        worstRe = std::max(worstRe, std::fabs(hOut[i] - dOutOut[i]));
    }
    std::printf("info: K10 worst |dr|=%.3g\n", worstRe);
    Check(worstRe < 1e-4f, "TN-6: K10 resample matches within 1e-4");
    // K11: exact winner on a unique scene, exact distance on noise.
    {
        float vp[16] = {0};
        vp[0] = vp[5] = vp[10] = vp[15] = 1.0f;
        float cands[9] = {0.0f, 0.0f,  0.0f, 0.5f, 0.5f, 0.0f,
                          -0.5f, -0.5f, 0.5f};
        float *dCands = nullptr;
        cudaMalloc(&dCands, sizeof(cands));
        cudaMemcpy(dCands, cands, sizeof(cands), cudaMemcpyHostToDevice);
        PomadeDevicePickBest *dBest = nullptr, *dScratch = nullptr;
        cudaMalloc(&dBest, sizeof(PomadeDevicePickBest));
        cudaMalloc(&dScratch, sizeof(PomadeDevicePickBest));
        Check(PomadeLaunchPickReduce(dCands, 3, vp, 800, 600, 400.0f, 300.0f,
                                    5.0f, dBest, dScratch, stream, err,
                                    sizeof(err)),
              "TN-6: K11 launches");
        PomadeDevicePickBest best;
        cudaMemcpy(&best, dBest, sizeof(best), cudaMemcpyDeviceToHost);
        PomadePickSets sets;
        sets.tubeVerts = cands;
        sets.tubeVertCount = 3;
        PomadePickHit cpu = PomadePickCpu(sets, PomadePick_TubeVert, vp, 800,
                                        600, 400.0f, 300.0f, 5.0f);
        Check(best.hit == 1 && best.index == cpu.index &&
                  best.distPx == cpu.distPx && best.depth == cpu.depth,
              "TN-6: K11 winners agree bit-exactly");
        // A 770K-candidate reduction times the TN-2 device lane.
        int const n = 770000;
        float *dBig = nullptr;
        cudaMalloc(&dBig, sizeof(float) * size_t(n) * 3);
        std::vector<float> big(size_t(256) * 3, 0.0f);
        for (int i = 0; i < 256; ++i) {
            big[size_t(i) * 3 + 0] = float(i % 16) / 8.0f - 1.0f;
            big[size_t(i) * 3 + 1] = float(i / 16) / 8.0f - 1.0f;
        }
        for (int base = 0; base < n; base += 256) {
            int const m = std::min(256, n - base);
            cudaMemcpy(dBig + size_t(base) * 3, big.data(),
                       sizeof(float) * size_t(m) * 3,
                       cudaMemcpyHostToDevice);
        }
        int const blocks = (n + 255) / 256;
        PomadeDevicePickBest *dScratchBig = nullptr;
        cudaMalloc(&dScratchBig,
                   sizeof(PomadeDevicePickBest) * size_t(blocks + 16));
        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventRecord(ev0, stream);
        bool pok = PomadeLaunchPickReduce(dBig, n, vp, 800, 600, 400.0f,
                                         300.0f, 5.0f, dBest, dScratchBig,
                                         stream, err, sizeof(err));
        cudaEventRecord(ev1, stream);
        cudaEventSynchronize(ev1);
        float pickMs = 0;
        cudaEventElapsedTime(&pickMs, ev0, ev1);
        std::printf("info: K11 device reduction over %d candidates: %.3f "
                    "ms\n",
                    n, pickMs);
        Check(pok, "TN-6: K11 scales to 770K candidates");
        Check(pok && pickMs < 0.5f,
              "TN-2: the K11 lane beats 0.5 ms at 770K candidates");
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaFree(dCands);
        cudaFree(dBest);
        cudaFree(dScratch);
        cudaFree(dBig);
        cudaFree(dScratchBig);
    }
    cudaFree(dCx);
    cudaFree(dCy);
    cudaFree(dCz);
    cudaFree(dFrames);
    cudaFree(dSecT);
    cudaFree(dSecU);
    cudaFree(dSecV);
    cudaFree(dSecS);
    cudaFree(dSecTw);
    cudaFree(dPos);
    cudaFree(dNrm);
    cudaFree(dRingT);
    cudaFree(dRoots);
    cudaFree(dBindings);
    cudaFree(dGuides);
    cudaFree(dLens);
    cudaFree(dIn);
    cudaFree(dOff);
    cudaFree(dCnt);
    cudaFree(dOut);
    cudaFree(dOutC);
    cudaFree(dOutF);
    cudaStreamDestroy(stream);
}

void CheckProductionPickScale()
{
    using namespace usdGenPomade;
    if (!HaveCudaDevice()) {
        std::printf("skip: no CUDA device for the production pick gate\n");
        return;
    }
    // TN-2 production wiring gate: model Pick over reference-count
    // guide CVs (12K x 16) plus tube verts must take the K11 device
    // lane. (770K tube verts are gated at lane level above; a single
    // fat legacy tube would also inflate the CPU small-kind sets with
    // 24K centers, which no reference-shaped scene does. CPU needs
    // ~8 ms here; the gate at 1 ms proves the wiring.)
    PomadeModel model;
    if (!model.HasCudaMirror()) {
        std::printf("skip: no device mirror for the production pick gate\n");
        return;
    }
    PomadeTubeShape shape;
    shape.rings = 200;  // 200 x 8 = 1600 tube verts, 200 centers
    shape.ringVerts = 8;
    Check(model.BuildTestTube(shape), "pick-scale: the tube builds");
    PomadeModel::FillParams params;
    params.density = 12000.0f;
    params.cvCount = 16;
    Check(model.SetFillParams(params), "pick-scale: scale fill params set");
    bool const scaleRefill = model.RefillGuides(1.0f);
    std::string const scaleRefillDiagnostic = model.GetDiagnostic();
    Check(scaleRefill,
          "pick-scale: 12K x 16 refill runs: " + scaleRefillDiagnostic);
    Check(model.GetGuides().guideCount == 12000,
          "pick-scale: 192K guide CVs present");
    float vp[16] = {0};
    vp[0] = vp[5] = vp[10] = vp[15] = 1.0f;
    // Legacy vert 0 sits at (radius, 0, 0) = (0.5, 0, 0).
    float const px = (0.5f * 0.5f + 0.5f) * 400.0f;
    float const py = (1.0f - (0.0f * 0.5f + 0.5f)) * 400.0f;
    PomadePickHit d0 =
        model.Pick(vp, 400, 400, px, py, 5.0f, PomadePick_All);
    Check(d0.hit, "pick-scale: the production pick hits");
    PomadePickHit d1 =
        model.Pick(vp, 400, 400, px, py, 5.0f, PomadePick_All);
    Check(d1.hit == d0.hit && d1.kind == d0.kind && d1.index == d0.index &&
              d1.subIndex == d0.subIndex && d1.distPx == d0.distPx &&
              d1.depth == d0.depth,
          "pick-scale: device picks are bit-deterministic");
    int const reps = 50;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        model.Pick(vp, 400, 400, px, py, 5.0f, PomadePick_All);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    std::printf("info: production pick at 1600 + 192K: %.3f ms "
                "(wiring gate 1 ms; TN-2 budget 0.5 ms)\n",
                ms);
    Check(ms < 1.0, "pick-scale: production pick under 1 ms (TN-2)");
#ifdef _WIN32
    _putenv("USDGEN_POMADE_FORCE_CPU_PICK=1");
#else
    setenv("USDGEN_POMADE_FORCE_CPU_PICK", "1", 1);
#endif
    PomadePickHit cpu =
        model.Pick(vp, 400, 400, px, py, 5.0f, PomadePick_All);
#ifdef _WIN32
    _putenv("USDGEN_POMADE_FORCE_CPU_PICK=");
#else
    unsetenv("USDGEN_POMADE_FORCE_CPU_PICK");
#endif
    Check(cpu.hit && d0.hit, "pick-scale: forced CPU pick hits too");
    std::printf("info: CPU winner kind=%u index=%d dist=%.4g; device "
                "kind=%u index=%d dist=%.4g\n",
                cpu.kind, cpu.index, cpu.distPx, d0.kind, d0.index,
                d0.distPx);
    Check(std::abs(cpu.distPx - d0.distPx) < 1e-3f &&
              std::abs(cpu.depth - d0.depth) < 1e-5f,
          "pick-scale: CPU/device agree within tessellation wobble");
}
#endif

// V0b (plan/18 §7 G2): every §5.2/§5.3 operation takes a tube id, so a
// child can be section-edited and filled on its own. The tube-0 spellings
// stay exactly what they were.
void CheckPerTubeOps()
{
    using namespace usdGenPomade;
    PomadeModel model;
    Check(model.BuildTestTube(), "per-tube: the L1 tube builds");
    std::vector<int> kids;
    Check(model.SubdivideTube(0, 4, "kmeans", 2, &kids) && kids.size() == 4,
          "per-tube: the tube subdivides into four children");
    int const child = kids[0];

    // Section ops on a child move that child and nothing else.
    PomadeTubeSection before;
    Check(model.GetTubeSection(child, 1, &before),
          "per-tube: a child section reads back");
    PomadeTubeSection siblingBefore;
    Check(model.GetTubeSection(kids[1], 1, &siblingBefore),
          "per-tube: a sibling section reads back");
    uint64_t const v0 = model.GetVersion();
    Check(model.MoveTubeSectionRing(child, 1, 0.05f, -0.02f),
          "per-tube: MoveTubeSectionRing edits a child");
    Check(model.GetVersion() > v0, "per-tube: the edit bumps the version");
    PomadeTubeSection after;
    Check(model.GetTubeSection(child, 1, &after) &&
              after.u[0] != before.u[0] && after.v[0] != before.v[0],
          "per-tube: the child's ring moved");
    PomadeTubeSection siblingAfter;
    Check(model.GetTubeSection(kids[1], 1, &siblingAfter) &&
              siblingAfter.u == siblingBefore.u &&
              siblingAfter.v == siblingBefore.v,
          "per-tube: the sibling is untouched");

    Check(model.ScaleTubeSectionRing(child, 1, 1.5f),
          "per-tube: ScaleTubeSectionRing edits a child");
    Check(model.TwistTubeSectionRing(child, 1, 0.1f),
          "per-tube: TwistTubeSectionRing edits a child");
    Check(model.MoveTubeSectionCV(child, 1, 0, 0.01f, 0.01f),
          "per-tube: MoveTubeSectionCV edits a child");
    Check(model.CopyTubeSectionRing(child, 0, 1),
          "per-tube: CopyTubeSectionRing edits a child");
    Check(model.RelaxTubeCenter(child, 0.5f, 2),
          "per-tube: RelaxTubeCenter edits a child");
    Check(model.SetTubeLengthFor(child, 3.5f),
          "per-tube: SetTubeLengthFor rescales a child");
    Check(!model.MoveTubeSectionRing(99, 1, 0.0f, 0.0f),
          "per-tube: an unknown tube is refused");
    Check(!model.MoveTubeSectionRing(child, 99, 0.0f, 0.0f),
          "per-tube: an out-of-range ring is refused");

    // A derived child's LAYOUT belongs to the parent's subdivision, so the
    // two count-changing ops refuse rather than desync the deltas.
    int const secBefore = model.GetTubeSectionCount(child);
    Check(!model.AddTubeSectionRing(child, 0.5f),
          "per-tube: adding a ring to a derived child is refused");
    Check(!model.RemoveTubeSectionRing(child, 1),
          "per-tube: removing a ring from a derived child is refused");
    Check(!model.InsertTubeCenterCV(child, 1),
          "per-tube: inserting a center CV in a derived child is refused");
    Check(model.GetTubeSectionCount(child) == secBefore,
          "per-tube: the refused ops left the layout alone");

    // This intentionally accumulated workflow has bent/resized the child.
    // Its K7 writeback can take the non-planar projection path, so retain a
    // narrow assertion here: the selected child remains editable and its
    // siblings remain authored state, without imposing a planar cage rule.
    {
        PomadeTubeDesc parentBefore, childBefore, inheritedSiblingBefore;
        Check(model.GetTubeDesc(0, &parentBefore) &&
                  model.GetTubeDesc(child, &childBefore) &&
                  model.GetTubeDesc(kids[1], &inheritedSiblingBefore),
              "per-tube: parent, edited child and sibling snapshot");
        Check(model.MoveTubeCenterCV(child, 2, 0.4f, 0.0f, 0.0f),
              "per-tube: a child center CV moves");
        PomadeTubeDesc parentAfter, childAfter, inheritedSiblingAfter;
        Check(model.GetTubeDesc(0, &parentAfter) &&
                  model.GetTubeDesc(child, &childAfter) &&
                  model.GetTubeDesc(kids[1], &inheritedSiblingAfter),
              "per-tube: K7 descriptors read back");
        bool const childMoved = childAfter.centerX.size() > 2 &&
                                childBefore.centerX.size() > 2 &&
                                childAfter.centerX[2] != childBefore.centerX[2];
        bool const siblingStable =
            inheritedSiblingAfter.centerX == inheritedSiblingBefore.centerX &&
            inheritedSiblingAfter.centerY == inheritedSiblingBefore.centerY &&
            inheritedSiblingAfter.centerZ == inheritedSiblingBefore.centerZ &&
            SameSections(inheritedSiblingAfter.sections,
                         inheritedSiblingBefore.sections);
        Check(childMoved && siblingStable,
              "per-tube: complex child K7 edit remains local to its sibling family");
    }

    // This fresh straight parent isolates the planar K7 branch.  A direct
    // move of one retained outer slot may update that parent UV target, but
    // must not move its raw cage/Q frame or any outer corner owned by another
    // child.  It also verifies a later parent K6 retains the child residual.
    {
        PomadeModel holdingModel;
        std::vector<int> holdingKids;
        bool const holdingSetup = holdingModel.BuildTestTube() &&
            holdingModel.SubdivideTube(0, 4, "kmeans", 83, &holdingKids) &&
            holdingKids.size() == 4;
        int holdingChild = -1;
        PomadeParentBoundaryBinding holdingBinding;
        if (holdingSetup) {
            for (int childId : holdingKids) {
                PomadeTubeDesc candidate;
                if (!holdingModel.GetTubeDesc(childId, &candidate)) {
                    continue;
                }
                for (PomadeParentBoundaryBinding const &binding :
                     candidate.inheritedBoundaryBindings) {
                    if (binding.section > 0 &&
                        binding.section + 1 < int(candidate.sections.size())) {
                        holdingChild = childId;
                        holdingBinding = binding;
                        break;
                    }
                }
                if (holdingChild >= 0) {
                    break;
                }
            }
        }
        Check(holdingSetup && holdingChild >= 0,
              "per-tube: planar fixture finds an inherited outer holding slot");
        if (holdingChild >= 0) {
        PomadeTubeDesc holdingParentBefore;
        Check(holdingModel.GetTubeDesc(0, &holdingParentBefore),
              "per-tube: planar holding parent reads before K7");
        bool const holdingEdit = holdingModel.MoveTubeSectionCV(
            holdingChild, holdingBinding.section, holdingBinding.childSlot,
            0.055f, -0.035f);
        PomadeTubeDesc holdingParentAfter, holdingChildAfter;
        Check(holdingEdit && holdingModel.GetTubeDesc(0, &holdingParentAfter) &&
                  holdingModel.GetTubeDesc(holdingChild, &holdingChildAfter),
              "per-tube: planar holding-slot edit writes K7");
        bool const rawCageExact =
            holdingParentAfter.centerX == holdingParentBefore.centerX &&
            holdingParentAfter.centerY == holdingParentBefore.centerY &&
            holdingParentAfter.centerZ == holdingParentBefore.centerZ &&
            holdingParentAfter.ringVerts == holdingParentBefore.ringVerts &&
            holdingParentAfter.rootFramePinned ==
                holdingParentBefore.rootFramePinned &&
            holdingParentAfter.frameReference ==
                holdingParentBefore.frameReference;
        float editedChildPoint[3] = {}, editedParentPoint[3] = {};
        bool const holdingTargetExact =
            SectionSlotWorld(holdingChildAfter, holdingBinding.section,
                             holdingBinding.childSlot, editedChildPoint) &&
            SectionSlotWorld(holdingParentAfter, holdingBinding.section,
                             holdingBinding.parentSlot, editedParentPoint) &&
            Near(editedChildPoint[0], editedParentPoint[0], 3e-4f) &&
            Near(editedChildPoint[1], editedParentPoint[1], 3e-4f) &&
            Near(editedChildPoint[2], editedParentPoint[2], 3e-4f);
        bool unrelatedOuterExact = true;
        int unrelatedOuterCount = 0;
        for (int siblingId : holdingKids) {
            if (siblingId == holdingChild) {
                continue;
            }
            PomadeTubeDesc sibling;
            if (!holdingModel.GetTubeDesc(siblingId, &sibling)) {
                continue;
            }
            for (PomadeParentBoundaryBinding const &binding :
                 sibling.inheritedBoundaryBindings) {
                if (binding.section != holdingBinding.section ||
                    binding.parentSlot == holdingBinding.parentSlot) {
                    continue;
                }
                float beforePoint[3] = {}, afterPoint[3] = {};
                unrelatedOuterExact =
                    SectionSlotWorld(holdingParentBefore, binding.section,
                                     binding.parentSlot, beforePoint) &&
                    SectionSlotWorld(holdingParentAfter, binding.section,
                                     binding.parentSlot, afterPoint) &&
                    Near(beforePoint[0], afterPoint[0], 3e-4f) &&
                    Near(beforePoint[1], afterPoint[1], 3e-4f) &&
                    Near(beforePoint[2], afterPoint[2], 3e-4f) &&
                    unrelatedOuterExact;
                ++unrelatedOuterCount;
            }
        }
        Check(rawCageExact,
              "per-tube: planar K7 keeps the parent cage and Q frame exact");
        Check(holdingTargetExact,
              "per-tube: planar K7 writes the edited inherited outer slot");
        Check(unrelatedOuterCount > 0 && unrelatedOuterExact,
              "per-tube: planar K7 keeps sibling-owned outer slots exact");

        PomadeModel::TubeRecord residualBefore, residualAfter;
        bool const parentK6 = holdingModel.GetTubeRecord(holdingChild,
                                                           &residualBefore) &&
            holdingModel.MoveTubeSectionRing(0, holdingBinding.section,
                                              0.012f, -0.007f) &&
            holdingModel.GetTubeRecord(holdingChild, &residualAfter) &&
            holdingBinding.section < int(residualBefore.deltas.sections.size()) &&
            holdingBinding.section < int(residualAfter.deltas.sections.size()) &&
            holdingBinding.childSlot <
                int(residualBefore.deltas.sections[
                    size_t(holdingBinding.section)].u.size()) &&
            holdingBinding.childSlot <
                int(residualAfter.deltas.sections[
                    size_t(holdingBinding.section)].u.size()) &&
            Near(residualBefore.deltas.sections[size_t(holdingBinding.section)].u[
                     size_t(holdingBinding.childSlot)],
                 residualAfter.deltas.sections[size_t(holdingBinding.section)].u[
                     size_t(holdingBinding.childSlot)], 2e-5f) &&
            Near(residualBefore.deltas.sections[size_t(holdingBinding.section)].v[
                     size_t(holdingBinding.childSlot)],
                 residualAfter.deltas.sections[size_t(holdingBinding.section)].v[
                     size_t(holdingBinding.childSlot)], 2e-5f);
        Check(parentK6,
              "per-tube: a later parent K6 preserves the holding-slot sculpt residual");
        }
    }

    // Tube 0 still goes through the single-tube spelling unchanged.
    PomadeTubeSection root0;
    Check(model.GetSection(0, &root0), "per-tube: tube 0 sections read back");
    Check(model.MoveTubeSectionRing(0, 0, 0.01f, 0.0f),
          "per-tube: tubeId 0 routes to the tube-0 operation");
    PomadeTubeSection root1;
    Check(model.GetSection(0, &root1) && root1.u[0] != root0.u[0],
          "per-tube: tube 0's own ring moved");
    Check(model.AddSectionRing(0.42f),
          "per-tube: tube 0 still accepts a new ring (no derivation)");

    // Fill params are per tube, and a subdivided parent's fill is suspended.
    PomadeModel::FillParams childFill;
    childFill.density = 12.0f;
    childFill.cvCount = 5;
    childFill.seed = 21;
    childFill.edgeBias = 0.25f;
    Check(model.SetTubeFillParams(child, childFill),
          "per-tube: a child takes its own fill params");
    PomadeModel::FillParams readBack;
    Check(model.GetTubeFillParams(child, &readBack) &&
              readBack.density == 12.0f && readBack.cvCount == 5 &&
              readBack.seed == 21 && readBack.edgeBias == 0.25f,
          "per-tube: the child's fill params read back");
    PomadeModel::FillParams siblingFill;
    Check(model.GetTubeFillParams(kids[1], &siblingFill) &&
              siblingFill.density != 12.0f,
          "per-tube: a sibling keeps the inherited params");
    Check(!model.SetTubeFillParams(child, [] {
              PomadeModel::FillParams bad;
              bad.cvCount = 1;
              return bad;
          }()),
          "per-tube: a bad cvCount is refused");
    Check(model.IsTubeFillSuspended(0),
          "per-tube: the subdivided parent's fill is suspended");
    Check(!model.IsTubeFillSuspended(child),
          "per-tube: a leaf child still fills");
    Check(model.MergeChildren(0), "per-tube: the children merge back");
    Check(!model.IsTubeFillSuspended(0),
          "per-tube: the fill resumes once the children are gone");
    // The child above was edited far enough (scale 1.5, twist, a dragged
    // CV, a copied ring) that the K7 aggregate may not triangulate at every
    // station. Whatever the refill makes of the merged tube, it is never
    // silent: tube 0 either produces guides or is named in RefillDrops().
    bool const refilled = model.RefillGuides(1.0f);
    std::vector<std::pair<int, std::string>> const drops =
        model.RefillDrops();
    bool const reported = drops.size() == 1 && drops[0].first == 0 &&
                          !drops[0].second.empty();
    Check((refilled && model.GetGuides().guideCount > 0 && drops.empty()) ||
              (!refilled && reported),
          "per-tube: the merged tube fills or its refill drop is reported");
}

// V0b: the per-tube C ABI (pomadeApiStage.h) drives the same operations.
void CheckPerTubeAbi()
{
    PomadeModelContext *ctx = nullptr;
    Check(Pomade_Create(&ctx) == POMADE_OK, "per-tube ABI: the model creates");
    Check(Pomade_BuildTestTube(ctx, 0, 0, 0.0f, 0.0f) == POMADE_OK,
          "per-tube ABI: the tube builds");
    int kids[4] = {0, 0, 0, 0};
    int count = 0;
    Check(Pomade_SubdivideTube(ctx, 0, 4, "kmeans", 2, kids, 4, &count) ==
                  POMADE_OK &&
              count == 4,
          "per-tube ABI: the tube subdivides");
    int const child = kids[0];
    Check(Pomade_MoveTubeSectionRing(ctx, child, 1, 0.05f, -0.02f) == POMADE_OK,
          "per-tube ABI: MoveTubeSectionRing");
    Check(Pomade_ScaleTubeSectionRing(ctx, child, 1, 1.25f) == POMADE_OK,
          "per-tube ABI: ScaleTubeSectionRing");
    Check(Pomade_TwistTubeSectionRing(ctx, child, 1, 0.05f) == POMADE_OK,
          "per-tube ABI: TwistTubeSectionRing");
    Check(Pomade_MoveTubeSectionCV(ctx, child, 1, 0, 0.01f, 0.0f) == POMADE_OK,
          "per-tube ABI: MoveTubeSectionCV");
    Check(Pomade_CopyTubeSectionRing(ctx, child, 0, 1) == POMADE_OK,
          "per-tube ABI: CopyTubeSectionRing");
    Check(Pomade_RelaxTubeCenter(ctx, child, 0.5f, 1) == POMADE_OK,
          "per-tube ABI: RelaxTubeCenter");
    Check(Pomade_SetTubeLengthFor(ctx, child, 3.0f) == POMADE_OK,
          "per-tube ABI: SetTubeLengthFor");
    Check(Pomade_AddTubeSectionRing(ctx, child, 0.5f) == POMADE_ERROR,
          "per-tube ABI: a layout change on a derived child is refused");
    Check(std::strlen(Pomade_StageGetLastError()) > 0,
          "per-tube ABI: the refusal carries a message");
    float const profile[4] = {0.0f, 1.0f, 1.0f, 0.5f};
    Check(Pomade_SetTubeFillParams(ctx, child, 9.0f, 6, 3, 0.1f, profile, 4) ==
              POMADE_OK,
          "per-tube ABI: SetTubeFillParams");
    float density = 0.0f, edgeBias = 0.0f;
    int cvCount = 0, seed = 0, profileCount = 0;
    Check(Pomade_GetTubeFillParams(ctx, child, &density, &cvCount, &seed,
                                  &edgeBias, &profileCount) == POMADE_OK &&
              density == 9.0f && cvCount == 6 && seed == 3 &&
              profileCount == 4,
          "per-tube ABI: GetTubeFillParams round-trips");
    Check(Pomade_IsTubeFillSuspended(ctx, 0) == 1,
          "per-tube ABI: the parent's fill is suspended");
    Check(Pomade_IsTubeFillSuspended(ctx, child) == 0,
          "per-tube ABI: the child's fill is live");
    int parent = 99, childIndex = 99;
    Check(Pomade_GetTubeParent(ctx, child, &parent, &childIndex) == POMADE_OK &&
              parent == 0 && childIndex == 0,
          "per-tube ABI: GetTubeParent reports the link");
    Check(Pomade_IsTubePersistent(ctx, child) == 0,
          "per-tube ABI: a subdivided child is not an on-the-fly parent");
    Check(Pomade_MoveTubeSectionRing(nullptr, 0, 0, 0.0f, 0.0f) == POMADE_ERROR,
          "per-tube ABI: a null context is refused");
    Check(Pomade_Destroy(ctx) == POMADE_OK, "per-tube ABI: the model destroys");
}

}  // namespace

int main()
{
    CheckK4();
    CheckK5();
    CheckSparseProfile();
    CheckCenterCoreHandle();
    CheckRigidTubeTransport();
    CheckK8();
    CheckK9();
    CheckK10();
    CheckK11();
    CheckModelTubeMode();
    CheckModelFillMode();
    CheckModelPerTubeFill();
    CheckAutoTube();
    CheckCApi();
    CheckProbes();
    CheckPerTubeOps();
    CheckPerTubeAbi();
    CheckSubdivideRingAlignment();
#ifdef USDGEN_POMADE_HAS_CUDA
    CheckCudaParity();
    CheckProductionPickScale();
#else
    std::printf("skip: CUDA off, TN-6 parity checks compiled out\n");
#endif
    std::printf("%d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
