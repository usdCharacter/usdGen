// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenScatterSortSkips.cpp — T0: the Capture morton sort's skipped-digit
// paths. The radix skips constant digits (an identity pass), so captures land
// in one of four sort shapes: N <= 1 (no pass runs), all-constant keys (every
// pass skipped), a skipped pass 0 with surviving later passes, and the common
// all-varying sort. The sort must return the stable key order in every shape:
// each case pins its shape premise from the reference morton keys plus the
// full sort contract (sorted, stable, exact permutation).
#include "usdGen/opRegistry.h"
#include "usdGen/ops/scatter.h"
#include "usdGenMath/usdGenMath/hash.h"
#include "usdGenMath/usdGenMath/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

// One axis-aligned quad (x0,0,0)-(x0+s,0,0)-(x0+s,s,0)-(x0,s,0): fan area
// exactly s*s for exact s (translation only shifts Sterbenz-exact corner
// differences), so an integral density*s*s emits an exact root count.
static void AddQuad(UsdGenSurfaceDesc *s, float x0, float w)
{
    int const vb = int(s->restPoints.size());
    s->restPoints.push_back(GfVec3f(x0, 0.0f, 0.0f));
    s->restPoints.push_back(GfVec3f(x0 + w, 0.0f, 0.0f));
    s->restPoints.push_back(GfVec3f(x0 + w, w, 0.0f));
    s->restPoints.push_back(GfVec3f(x0, w, 0.0f));
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

// Expected per-face root count from the pinned hash family (04 :634-635).
static uint64_t TestNf(int seed, int f, double density, double area)
{
    uint64_t const hSeed =
        UsdGenHash64(uint64_t(uint32_t(seed)), kSaltScatter);
    uint64_t const faceKey =
        UsdGenHash64(hSeed ^ uint64_t(uint32_t(f)), kSaltScatter);
    double const expected = density * area;
    double const whole = std::floor(expected);
    double const frac = expected - whole;
    return uint64_t(whole) +
        (UsdGenHash01(faceKey, kSaltScatter) < float(frac) ? 1u : 0u);
}

// Reference morton keys of the captured roots (kernels.cpp canonical).
static std::vector<uint64_t> TestKeys(UsdGenCurveBuffer const &b)
{
    std::vector<uint64_t> keys(b.totalCurves);
    for (uint32_t i = 0; i < b.totalCurves; ++i)
        keys[i] = UsdGenMortonKey3(b.px[i], b.py[i], b.pz[i], 64.0f);
    return keys;
}

// Full sort contract over a capture: keys non-decreasing (sorted), the
// curve ids an exact permutation of the per-face (f, k) emission set, and
// equal-key runs in face-major emission order (stable: (face, k) ordered).
static bool CheckSort(UsdGenGraphDesc const &d, int seed,
                      UsdGenCurveBuffer const &b, std::vector<uint64_t> *keys)
{
    *keys = TestKeys(b);
    for (size_t i = 1; i < keys->size(); ++i)
        if ((*keys)[i] < (*keys)[i - 1])
            return false;
    UsdGenSurfaceDesc const &s = d.surfaces[0];
    std::vector<size_t> cornerOff(s.faceVertexCounts.size() + 1, 0);
    for (size_t f = 0; f < s.faceVertexCounts.size(); ++f)
        cornerOff[f + 1] = cornerOff[f] + size_t(s.faceVertexCounts[f]);
    std::vector<uint64_t> want;
    for (size_t f = 0; f < s.faceVertexCounts.size(); ++f) {
        // Fan area spelled as the engine's (kernels.cpp canonical).
        GfVec3f const p0 = s.restPoints[s.faceVertexIndices[cornerOff[f]]];
        double area = 0.0;
        for (int t = 1; t + 1 < s.faceVertexCounts[f]; ++t) {
            GfVec3f const pb =
                s.restPoints[s.faceVertexIndices[cornerOff[f] + size_t(t)]];
            GfVec3f const pc =
                s.restPoints[s.faceVertexIndices[cornerOff[f] + size_t(t) + 1]];
            GfVec3f const x = GfCross(pb - p0, pc - p0);
            area += double(0.5f * std::sqrt(
                x[0] * x[0] + x[1] * x[1] + x[2] * x[2]));
        }
        double const density = d.nodes[0].params[0].value.UncheckedGet<double>();
        uint64_t const nf = TestNf(seed, int(f), density, area);
        for (uint64_t k = 0; k < nf; ++k)
            want.push_back(UsdGenCurveId(seed, uint32_t(f), uint32_t(k)));
    }
    if (want.size() != b.totalCurves)
        return false;
    std::vector<uint64_t> got(b.curveId.begin(), b.curveId.end());
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    if (want != got)
        return false;
    // Stability: equal keys keep emission order. Emission is face-major
    // with k-minor per face; curve ids are unique per (f, k), so invert
    // each output id to its emission rank once and require non-decreasing
    // ranks within each equal-key run.
    std::unordered_map<uint64_t, size_t> rank;
    rank.reserve(b.totalCurves * 2);
    {
        size_t r = 0;
        for (size_t f = 0; f < s.faceVertexCounts.size(); ++f) {
            GfVec3f const p0 = s.restPoints[s.faceVertexIndices[cornerOff[f]]];
            double area = 0.0;
            for (int t = 1; t + 1 < s.faceVertexCounts[f]; ++t) {
                GfVec3f const pb =
                    s.restPoints[s.faceVertexIndices[cornerOff[f] + size_t(t)]];
                GfVec3f const pc = s.restPoints[s.faceVertexIndices[cornerOff[f] +
                                                                  size_t(t) + 1]];
                GfVec3f const x = GfCross(pb - p0, pc - p0);
                area += double(0.5f * std::sqrt(
                    x[0] * x[0] + x[1] * x[1] + x[2] * x[2]));
            }
            double const density =
                d.nodes[0].params[0].value.UncheckedGet<double>();
            uint64_t const nf = TestNf(seed, int(f), density, area);
            for (uint64_t k = 0; k < nf; ++k)
                rank[UsdGenCurveId(seed, uint32_t(f), uint32_t(k))] = r++;
        }
        if (r != b.totalCurves)
            return false;
    }
    for (size_t i = 1; i < keys->size(); ++i) {
        if ((*keys)[i] != (*keys)[i - 1])
            continue;
        auto a = rank.find(b.curveId[i - 1]);
        auto e = rank.find(b.curveId[i]);
        if (a == rank.end() || e == rank.end() || e->second < a->second)
            return false;
    }
    return true;
}

int main()
{
    usdGenRegisterM1Operators();
    int const seed = 42;

    // N == 1: no pass runs; the lone root passes through. Quad area 0.25,
    // density 4: expected exactly 1.0, frac 0, so nf == 1 on any seed.
    {
        UsdGenSurfaceDesc s;
        s.path = SdfPath("/Scalp");
        AddQuad(&s, 0.0f, 0.5f);
        UsdGenGraphDesc const d = Desc(s, 4.0, seed);
        UsdGenCurveBuffer b;
        CHECK(RunDirect(d, &b));
        CHECK(b.totalCurves == 1);
        CHECK(b.curveId[0] == UsdGenCurveId(seed, 0, 0));
        CHECK(b.rootPrim[0] == 0);
        std::vector<uint64_t> keys;
        CHECK(CheckSort(d, seed, b, &keys));
    }

    // All-constant keys (vary == 0): every pass skipped, emission order
    // preserved. One 2^-10 quad at the origin (area exactly 2^-20),
    // density 2^21: expected exactly 2.0, so nf == 2; both roots sit in
    // morton cell 0 whatever the draws do.
    {
        UsdGenSurfaceDesc s;
        s.path = SdfPath("/Scalp");
        AddQuad(&s, 0.0f, 0.0009765625f);  // 2^-10
        UsdGenGraphDesc const d = Desc(s, 2097152.0, seed);  // 2^21
        UsdGenCurveBuffer b;
        CHECK(RunDirect(d, &b));
        CHECK(b.totalCurves == 2);
        std::vector<uint64_t> keys;
        CHECK(CheckSort(d, seed, b, &keys));
        CHECK(keys[0] == keys[1]);  // the all-skipped premise
        CHECK(b.curveId[0] == UsdGenCurveId(seed, 0, 0));
        CHECK(b.curveId[1] == UsdGenCurveId(seed, 0, 1));
    }

    // Skipped pass 0, surviving later passes (serial sort): two 2^-10
    // quads at x = 0 and x = 1, one root each (density 2^20, expected
    // exactly 1.0). Both quants share x-bits 0-5 (0 vs 64) and all of
    // y/z, so the low 16 morton bits match while bit 18 differs: pass 0
    // is constant, pass 1 varies. Seed-independent by confinement.
    {
        UsdGenSurfaceDesc s;
        s.path = SdfPath("/Scalp");
        AddQuad(&s, 0.0f, 0.0009765625f);
        AddQuad(&s, 1.0f, 0.0009765625f);
        UsdGenGraphDesc const d = Desc(s, 1048576.0, seed);  // 2^20
        UsdGenCurveBuffer b;
        CHECK(RunDirect(d, &b));
        CHECK(b.totalCurves == 2);
        std::vector<uint64_t> keys;
        CHECK(CheckSort(d, seed, b, &keys));
        CHECK(keys[0] != keys[1]);
        CHECK(((keys[0] ^ keys[1]) & 0xffffull) == 0);  // pass 0 skipped
        CHECK(((keys[0] ^ keys[1]) >> 16) != 0);  // later passes survive
    }

    // Skipped pass 0 (parallel sort): 200 2^-10 quads at x = 0..199, 200
    // roots each (density 200*2^20, expected exactly 200.0). Every root
    // shares the low 16 morton bits (x-quant bits 0-5 all clear: 64k is
    // a multiple of 64; y/z quants all equal), while high bits vary with
    // k; N = 40000 clears the parallel-sort threshold.
    {
        UsdGenSurfaceDesc s;
        s.path = SdfPath("/Scalp");
        for (int k = 0; k < 200; ++k)
            AddQuad(&s, float(k), 0.0009765625f);
        UsdGenGraphDesc const d = Desc(s, 209715200.0, seed);  // 200*2^20
        UsdGenCurveBuffer b;
        CHECK(RunDirect(d, &b));
        CHECK(b.totalCurves == 40000);
        std::vector<uint64_t> keys;
        CHECK(CheckSort(d, seed, b, &keys));
        uint64_t orKeys = 0, andKeys = ~uint64_t(0);
        for (uint64_t k : keys) {
            orKeys |= k;
            andKeys &= k;
        }
        uint64_t const vary = orKeys ^ andKeys;
        CHECK((vary & 0xffffull) == 0);  // pass 0 skipped
        CHECK((vary >> 16) != 0);  // later passes survive
    }

    std::printf("testUsdGenScatterSortSkips PASS\n");
    return 0;
}
