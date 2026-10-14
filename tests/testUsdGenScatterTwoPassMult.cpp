// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenScatterTwoPassMult.cpp — T0: the two-pass Capture path honors a
// valid per-face densityMultiplier. Pass 1 mirrors the one-pass count
// spelling (expected = density * areaRest * mult[f], whole + hash01 draw),
// but no suite case feeds a valid multiplier past the two-pass threshold —
// Density pins small one-pass captures, LimitScatter the subdivision path,
// and TwoPass only the malformed-size error — so a pass-1 mult divergence
// (dropped scale, wrong face) survives the suite. 20000 unit quads (area
// exactly 1.0) with a {2.0, 0.5, 1.0, 0.0} mult cycle pin per-face counts
// (scale-up exact, scale-down by the pinned draw, identity, zero skip),
// cross-worker bit-identity, and the mirrored negative/NaN error branch.
#include "usdGen/opRegistry.h"
#include "usdGen/ops/scatter.h"
#include "usdGenMath/usdGenMath/hash.h"

#include "tbb/task_arena.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

// 200x100 unit quads: 20000 faces clears the two-pass threshold with up
// to 5 emission ranges, and every quad's fan area is exactly 1.0
// (axis-aligned unit edges: two 0.5 weights, exact in float).
static UsdGenSurfaceDesc Surface()
{
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/Scalp");
    int const w = 200, h = 100;
    for (int y = 0; y <= h; ++y)
        for (int x = 0; x <= w; ++x) {
            s.restPoints.push_back(GfVec3f(float(x), float(y), 0.0f));
            s.uv.push_back(GfVec2f(float(x) / float(w), float(y) / float(h)));
        }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            s.faceVertexCounts.push_back(4);
            int const i0 = y * (w + 1) + x;
            for (int i : {i0, i0 + 1, i0 + w + 2, i0 + w + 1})
                s.faceVertexIndices.push_back(i);
        }
    s.points = s.restPoints;
    return s;
}

static UsdGenGraphDesc Desc(UsdGenSurfaceDesc const &s, double density,
                            int seed)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom");
    d.surfaces.push_back(s);
    UsdGenNodeDesc n;
    n.path = SdfPath("/Scatter");
    n.type = TfToken("UsdGenScatter");
    n.surfaces.push_back(s.path);
    n.seed = seed;
    n.params.push_back({TfToken("density"), VtValue(density), false});
    d.nodes.push_back(n);
    d.terminal = n.path;
    return d;
}

static bool RunDirect(UsdGenGraphDesc const &d, UsdGenCurveBuffer *out,
                      std::string *error = nullptr)
{
    UsdGenNodeDesc const &node = d.nodes[0];
    UsdGenParamView params{&d, &node};
    UsdGenScatterOp op;
    UsdGenDiagnostics diag;
    if (!op.Bind(params, &diag))
        return false;
    auto cap = op.CreateCapture();
    if (!cap)
        return false;
    UsdGenCaptureContext ctx;
    ctx.desc = &d;
    ctx.params = &params;
    ctx.surface = 0;
    ctx.seed = uint32_t(node.seed);
    ctx.diag = &diag;
    UsdGenCurveBuffer empty;
    if (!op.Capture(ctx, empty, cap.get(), &diag)) {
        if (error && !diag.errors.empty())
            *error = diag.errors.front();
        return false;
    }
    *out = cap->Buffer();
    return true;
}

static bool SameRoots(UsdGenCurveBuffer const &a, UsdGenCurveBuffer const &b)
{
    return a.totalCurves == b.totalCurves && a.totalCvs == b.totalCvs &&
        a.px == b.px && a.py == b.py && a.pz == b.pz &&
        a.curveId == b.curveId && a.rootPrim == b.rootPrim &&
        a.rootUV == b.rootUV && a.rootT == b.rootT && a.rootN == b.rootN &&
        a.rootB == b.rootB && a.hairT == b.hairT;
}

// Expected per-face root count from the pinned hash family (04 :634-635)
// with the multiplier scale: expected = density * area * mult.
static uint64_t TestNfMult(int seed, int f, double density, double area,
                           double mult)
{
    uint64_t const hSeed =
        UsdGenHash64(uint64_t(uint32_t(seed)), kSaltScatter);
    uint64_t const faceKey =
        UsdGenHash64(hSeed ^ uint64_t(uint32_t(f)), kSaltScatter);
    double const expected = density * area * mult;
    double const whole = std::floor(expected);
    double const frac = expected - whole;
    return uint64_t(whole) +
        (UsdGenHash01(faceKey, kSaltScatter) < float(frac) ? 1u : 0u);
}

int main()
{
    usdGenRegisterM1Operators();
    UsdGenSurfaceDesc s = Surface();
    size_t const nFaces = s.faceVertexCounts.size();
    CHECK(nFaces > 4096);  // premise: this capture takes the two-pass path
    double const density = 1.0;
    int const seed = 42;

    // Per-face multiplier cycle: exact scale-up (2.0 -> nf == 2, frac 0),
    // scale-down by the pinned draw (0.5 -> nf in {0, 1}), identity, skip.
    float const cyc[4] = {2.0f, 0.5f, 1.0f, 0.0f};
    s.densityMultiplier.resize(nFaces);
    for (size_t f = 0; f < nFaces; ++f)
        s.densityMultiplier[f] = cyc[f % 4];

    UsdGenCurveBuffer out;
    CHECK(RunDirect(Desc(s, density, seed), &out));

    // Premise: every quad's fan area is exactly 1.0, asserted from the
    // engine's own fan spelling (not assumed from the coordinates).
    for (size_t f = 0; f < nFaces; ++f) {
        size_t const cbase = f * 4;
        GfVec3f const p0 = s.restPoints[s.faceVertexIndices[cbase]];
        double area = 0.0;
        for (int t = 1; t + 1 < 4; ++t) {
            GfVec3f const pb = s.restPoints[s.faceVertexIndices[cbase + size_t(t)]];
            GfVec3f const pc = s.restPoints[s.faceVertexIndices[cbase + size_t(t) + 1]];
            GfVec3f const x = GfCross(pb - p0, pc - p0);
            area += double(0.5f * std::sqrt(
                x[0] * x[0] + x[1] * x[1] + x[2] * x[2]));
        }
        CHECK(area == 1.0);
    }

    // Per-face counts from the pinned spelling; the histogram pins the
    // pass-1 mirror face by face (a dropped scale or wrong-face read
    // moves counts).
    std::vector<uint64_t> want(nFaces);
    uint64_t total = 0;
    for (size_t f = 0; f < nFaces; ++f) {
        want[f] = TestNfMult(seed, int(f), density, 1.0,
                             double(cyc[f % 4]));
        total += want[f];
    }
    CHECK(out.totalCurves == total);
    std::vector<uint64_t> got(nFaces, 0);
    for (size_t i = 0; i < out.totalCurves; ++i) {
        CHECK(out.rootPrim[i] >= 0 && size_t(out.rootPrim[i]) < nFaces);
        got[size_t(out.rootPrim[i])] += 1;
    }
    for (size_t f = 0; f < nFaces; ++f)
        CHECK(got[f] == want[f]);

    // Seed-independent constants: scale-up faces emit exactly 2, zero
    // faces emit nothing.
    for (size_t f = 0; f < nFaces; ++f) {
        if (f % 4 == 0)
            CHECK(got[f] == 2);
        if (f % 4 == 3)
            CHECK(got[f] == 0);
    }

    // Thread-count stability: 1..8-worker arenas agree bit-for-bit (one
    // range vs up to five; the multiplier read never shows).
    for (int workers = 1; workers <= 8; ++workers) {
        tbb::task_arena arena(workers);
        UsdGenCurveBuffer wide;
        bool ok = false;
        arena.execute([&] { ok = RunDirect(Desc(s, density, seed), &wide); });
        CHECK(ok);
        CHECK(SameRoots(out, wide));
    }

    // Error parity: negative and NaN multipliers fail closed on the
    // two-pass path with the one-pass message.
    {
        UsdGenSurfaceDesc bad = s;
        bad.densityMultiplier[7] = -1.0f;
        UsdGenCurveBuffer tmp;
        std::string error;
        CHECK(!RunDirect(Desc(bad, density, seed), &tmp, &error));
        CHECK(error.find("finite and >= 0") != std::string::npos);
    }
    {
        UsdGenSurfaceDesc bad = s;
        bad.densityMultiplier[12345] =
            std::numeric_limits<float>::quiet_NaN();
        UsdGenCurveBuffer tmp;
        std::string error;
        CHECK(!RunDirect(Desc(bad, density, seed), &tmp, &error));
        CHECK(error.find("finite and >= 0") != std::string::npos);
    }

    std::printf("ok: scatter two-pass multiplier\n");
    return 0;
}
