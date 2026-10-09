// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenScatterDegenerateFan.cpp — T0: the Capture fan's
// degenerate-normal fallback. A bowtie quad (self-canceling fan: two
// unit-area triangles with opposite normals) has positive area but an
// exactly-zero normal accumulator, so Normalize3 takes its l <= 1e-12
// fallback (0,1,0) while still emitting roots. Both emission paths (the
// one-pass emitRange fan and the two-pass fanFace+spill fan) must publish
// the fallback on every rootN.
#include "usdGen/opRegistry.h"
#include "usdGen/ops/scatter.h"
#include "usdGenMath/usdGenMath/hash.h"
#include "usdGenMath/usdGenMath/kernels.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

// Bowtie quad at integer offset x0: (x0,0,0)-(x0+1,1,0)-(x0+1,0,0)-(x0,1,0).
// Fan tri 0 is (p0,p1,p2) with cross (0,0,-1); fan tri 1 is (p0,p2,p3)
// with cross (0,0,1): integer-exact, so the accumulator cancels exactly
// while each weight is exactly 0.5 (area exactly 1.0). Integer offsets
// keep every corner difference Sterbenz-exact.
static void AddBowtie(UsdGenSurfaceDesc *s, float x0)
{
    int const vb = int(s->restPoints.size());
    s->restPoints.push_back(GfVec3f(x0, 0.0f, 0.0f));
    s->restPoints.push_back(GfVec3f(x0 + 1.0f, 1.0f, 0.0f));
    s->restPoints.push_back(GfVec3f(x0 + 1.0f, 0.0f, 0.0f));
    s->restPoints.push_back(GfVec3f(x0, 1.0f, 0.0f));
    s->faceVertexCounts.push_back(4);
    for (int k = 0; k < 4; ++k)
        s->faceVertexIndices.push_back(vb + k);
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

static bool RunDirect(UsdGenGraphDesc const &d, UsdGenCurveBuffer *out)
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
    if (!op.Capture(ctx, empty, cap.get(), &diag))
        return false;
    *out = cap->Buffer();
    return true;
}

// Reference fan of face f, spelled as the engine's (fanFace in
// ops/scatter.cpp): per-triangle weights, area sum, normal accumulator.
static void TestFan(UsdGenSurfaceDesc const &s, size_t f, double *area,
                    GfVec3f *nAcc)
{
    size_t cbase = 0;
    for (size_t g = 0; g < f; ++g)
        cbase += size_t(s.faceVertexCounts[g]);
    int const nc = s.faceVertexCounts[f];
    GfVec3f const p0 = s.restPoints[s.faceVertexIndices[cbase]];
    *area = 0.0;
    *nAcc = GfVec3f(0.0f, 0.0f, 0.0f);
    for (int t = 1; t + 1 < nc; ++t) {
        GfVec3f const pb =
            s.restPoints[s.faceVertexIndices[cbase + size_t(t)]];
        GfVec3f const pc =
            s.restPoints[s.faceVertexIndices[cbase + size_t(t) + 1]];
        GfVec3f const x = GfCross(pb - p0, pc - p0);
        float const w = 0.5f * std::sqrt(
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        *area += double(w);
        *nAcc += x;
    }
}

static bool CheckBowties(UsdGenSurfaceDesc const &s, int seed,
                         UsdGenCurveBuffer const &b)
{
    size_t const nFaces = s.faceVertexCounts.size();
    // Premises, asserted from the reference fan (not assumed): every
    // face has exactly unit area and an exactly-zero accumulator.
    for (size_t f = 0; f < nFaces; ++f) {
        double area = 0.0;
        GfVec3f nAcc(0.0f);
        TestFan(s, f, &area, &nAcc);
        if (area != 1.0 || nAcc != GfVec3f(0.0f, 0.0f, 0.0f))
            return false;
    }
    // Density 1.0 over unit area: expected exactly 1.0, frac 0, so each
    // face emits exactly root k == 0 on any seed.
    if (b.totalCurves != nFaces)
        return false;
    std::unordered_map<uint64_t, uint32_t> faceOf;
    faceOf.reserve(nFaces * 2);
    for (uint32_t f = 0; f < uint32_t(nFaces); ++f)
        faceOf[UsdGenCurveId(seed, f, 0)] = f;
    GfVec3f const fallback(0.0f, 1.0f, 0.0f);
    for (uint32_t i = 0; i < b.totalCurves; ++i) {
        auto it = faceOf.find(b.curveId[i]);
        if (it == faceOf.end())
            return false;
        // Roots are morton-sorted, so match by id, not position.
        if (b.rootPrim[i] != int(it->second))
            return false;
        if (b.rootN[i] != fallback)
            return false;
    }
    return true;
}

int main()
{
    usdGenRegisterM1Operators();
    int const seed = 42;

    // One-pass path (faceCount <= 4096): one bowtie, one root, fallback N.
    {
        UsdGenSurfaceDesc s;
        s.path = SdfPath("/Scalp");
        AddBowtie(&s, 0.0f);
        UsdGenGraphDesc const d = Desc(s, 1.0, seed);
        UsdGenCurveBuffer b;
        CHECK(RunDirect(d, &b));
        CHECK(CheckBowties(s, seed, b));
    }

    // Two-pass path (faceCount > 4096): 5000 translated bowties, one root
    // each, every rootN the fallback. Integer offsets keep the fan exact.
    {
        UsdGenSurfaceDesc s;
        s.path = SdfPath("/Scalp");
        for (int k = 0; k < 5000; ++k)
            AddBowtie(&s, float(k));
        UsdGenGraphDesc const d = Desc(s, 1.0, seed);
        UsdGenCurveBuffer b;
        CHECK(RunDirect(d, &b));
        CHECK(CheckBowties(s, seed, b));
    }

    std::printf("testUsdGenScatterDegenerateFan PASS\n");
    return 0;
}
