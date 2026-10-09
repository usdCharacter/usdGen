// testUsdGenScatterTwoPass.cpp — T0: the two-pass Capture path (big level-0
// captures past the threading threshold) emits exactly the roots the
// one-pass emission specifies: per-face counts from the pinned hash
// family, stable curve ids, morton order with face-major ties, and
// bit-exact positions/uv/frames, including the overflow-fan recompute
// (ntri > 2), restricted capture, flip, and error parity.
#include "usdGen/opRegistry.h"
#include "usdGen/ops/scatter.h"
#include "usdGenMath/usdGenMath/hash.h"
#include "usdGenMath/usdGenMath/kernels.h"

#include "tbb/task_arena.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

// 71x71 unit quads (area exactly 1.0) plus one axis-aligned octagon
// (convex, area exactly 7.0, ntri = 6 > 2 so pass 2 recomputes its fan),
// three trapezoids with unequal fan weights (1.0 vs 0.5, area 1.5) and
// two scalene triangles (area 3.0, ntri = 1 exercises the w1 pad).
// 5047 faces clears the two-pass threshold on every worker count.
static UsdGenSurfaceDesc Surface()
{
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/Scalp");
    int const w = 71, h = 71;
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
    int const base = int(s.restPoints.size());
    int const ox[8] = {100, 101, 102, 102, 101, 100, 99, 99};
    int const oy[8] = {0, 0, 1, 2, 3, 3, 2, 1};
    for (int k = 0; k < 8; ++k) {
        s.restPoints.push_back(
            GfVec3f(float(ox[k]), float(oy[k]), 0.0f));
        s.uv.push_back(GfVec2f(0.5f, 0.5f));
    }
    s.faceVertexCounts.push_back(8);
    for (int k = 0; k < 8; ++k)
        s.faceVertexIndices.push_back(base + k);
    // Trapezoids (10+t*10,0),(12+t*10,0),(11+t*10,1),(10+t*10,1): fan
    // weights exactly 1.0 vs 0.5 (unequal, so a spill swap moves roots).
    for (int t = 0; t < 3; ++t) {
        int const vb = int(s.restPoints.size());
        float const ox = float(10 + t * 10);
        s.restPoints.push_back(GfVec3f(ox, 0.0f, 0.0f));
        s.restPoints.push_back(GfVec3f(ox + 2.0f, 0.0f, 0.0f));
        s.restPoints.push_back(GfVec3f(ox + 1.0f, 1.0f, 0.0f));
        s.restPoints.push_back(GfVec3f(ox, 1.0f, 0.0f));
        for (int k = 0; k < 4; ++k)
            s.uv.push_back(GfVec2f(0.25f, 0.75f));
        s.faceVertexCounts.push_back(4);
        for (int k = 0; k < 4; ++k)
            s.faceVertexIndices.push_back(vb + k);
    }
    // Scalene triangles (t*10,0),(3+t*10,0),(0,2): area exactly 3.0.
    for (int t = 0; t < 2; ++t) {
        int const vb = int(s.restPoints.size());
        float const ox = float(t * 10);
        s.restPoints.push_back(GfVec3f(ox, 0.0f, 0.0f));
        s.restPoints.push_back(GfVec3f(ox + 3.0f, 0.0f, 0.0f));
        s.restPoints.push_back(GfVec3f(0.0f, 2.0f, 0.0f));
        for (int k = 0; k < 3; ++k)
            s.uv.push_back(GfVec2f(0.75f, 0.25f));
        s.faceVertexCounts.push_back(3);
        for (int k = 0; k < 3; ++k)
            s.faceVertexIndices.push_back(vb + k);
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

// Fan triangulation spelled as the engine's (kernels.cpp canonical):
// weight w[t] = 0.5*|cross|, areaRest = sum of weights, Nrest from the
// accumulated normal. Independent loop, identical float ops.
static void TestFan(GfVec3f const *rest, int const *fvi, size_t cbase, int nc,
                    std::vector<float> *triW, GfVec3f *nRest, double *area)
{
    GfVec3f const p0 = rest[fvi[cbase]];
    double areaRest = 0.0;
    GfVec3f nAcc(0.0f, 0.0f, 0.0f);
    triW->resize(size_t(nc) - 2);
    for (int t = 1; t + 1 < nc; ++t) {
        GfVec3f const pb = rest[fvi[cbase + size_t(t)]];
        GfVec3f const pc = rest[fvi[cbase + size_t(t) + 1]];
        GfVec3f const eb = pb - p0;
        GfVec3f const ec = pc - p0;
        GfVec3f const x = GfCross(eb, ec);
        float const w = 0.5f * std::sqrt(
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        (*triW)[size_t(t) - 1] = w;
        areaRest += double(w);
        nAcc += x;
    }
    float const l = std::sqrt(nAcc[0] * nAcc[0] + nAcc[1] * nAcc[1] +
                              nAcc[2] * nAcc[2]);
    *nRest = l > 1e-12f ? nAcc / l : GfVec3f(0.0f, 1.0f, 0.0f);
    *area = areaRest;
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

// Expected root values for (face f, index k), spelled as the engine's
// level-0 root loop: the same three curveId-keyed draws, triangle pick,
// blends and frame tests. Returns false for degenerate test setup.
static bool TestRoot(GfVec3f const *rest, GfVec2f const *uv, int const *fvi,
                     size_t cbase, int nc, float const *triW, double area,
                     GfVec3f const &nRest, int seed, int f, uint64_t k,
                     bool flip, GfVec3f *pos, GfVec2f *puv, GfVec3f *T,
                     GfVec3f *N, GfVec3f *B)
{
    uint64_t const hSeed =
        UsdGenHash64(uint64_t(uint32_t(seed)), kSaltScatter);
    uint64_t const faceKey =
        UsdGenHash64(hSeed ^ uint64_t(uint32_t(f)), kSaltScatter);
    uint64_t const curveId =
        UsdGenHash64(faceKey ^ uint64_t(uint32_t(k)), kSaltScatter);
    uint64_t const hB0 =
        UsdGenHash64(uint64_t(uint32_t(seed)), kSaltScatterBary);
    uint64_t const hB1 =
        UsdGenHash64(uint64_t(uint32_t(seed)), kSaltScatterBary + 1u);
    uint64_t const hB2 =
        UsdGenHash64(uint64_t(uint32_t(seed)), kSaltScatterBary + 2u);
    float const u0 = UsdGenHash01(hB0 ^ curveId, kSaltScatterBary);
    float const u1 = UsdGenHash01(hB1 ^ curveId, kSaltScatterBary + 1u);
    float const u2 = UsdGenHash01(hB2 ^ curveId, kSaltScatterBary + 2u);
    size_t const ntri = size_t(nc) - 2;
    size_t ti = ntri - 1;
    double cum = 0.0;
    double const target = double(u0) * area;
    for (size_t t = 0; t < ntri; ++t) {
        cum += double(triW[t]);
        if (target < cum) {
            ti = t;
            break;
        }
    }
    GfVec3f const p0 = rest[fvi[cbase]];
    size_t const ib = cbase + ti + 1;
    size_t const ic = ib + 1;
    GfVec3f const pb = rest[fvi[ib]];
    GfVec3f const pc = rest[fvi[ic]];
    float const r1 = std::sqrt(u1);
    *pos = p0 * (1.0f - r1) + pb * (u2 * r1) + pc * (r1 * (1.0f - u2));
    *puv = GfVec2f(1.0f / 3.0f, 1.0f / 3.0f);
    if (uv)
        *puv = uv[fvi[cbase]] * (1.0f - r1) + uv[fvi[ib]] * (u2 * r1) +
            uv[fvi[ic]] * (r1 * (1.0f - u2));
    *N = nRest;
    GfVec3f e0 = pb - p0;
    float const e0d = GfDot(e0, nRest);
    float const e0l2 = GfDot(e0, e0);
    if (e0d * e0d > 0.81f * e0l2) {
        e0 = pc - p0;
        *T = e0 - nRest * GfDot(e0, nRest);
    } else {
        *T = e0 - nRest * e0d;
    }
    float tLen = std::sqrt(GfDot(*T, *T));
    if (tLen < 1e-9f) {
        *T = std::abs(nRest[0]) > 0.9f ? GfVec3f(0.0f, 1.0f, 0.0f)
                                       : GfVec3f(1.0f, 0.0f, 0.0f);
        *T = *T - nRest * GfDot(*T, nRest);
        float const l = std::sqrt((*T)[0] * (*T)[0] + (*T)[1] * (*T)[1] +
                                  (*T)[2] * (*T)[2]);
        *T = l > 1e-12f ? *T / l : GfVec3f(0.0f, 1.0f, 0.0f);
    } else {
        *T = *T / tLen;
    }
    GfVec3f const x = GfCross(*N, *T);
    float const bl = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    *B = bl > 1e-12f ? x / bl : GfVec3f(0.0f, 1.0f, 0.0f);
    if (flip) {
        *T = -*T;
        *B = -*B;
    }
    return true;
}

int main()
{
    usdGenRegisterM1Operators();
    UsdGenSurfaceDesc const s = Surface();
    CHECK(s.faceVertexCounts.size() == 5047);
    int const seed = 7;
    double const density = 4.5;

    // Per-face corner bases (faces are FACE-RELATIVE indexed, 02 §2.2).
    std::vector<size_t> cornerOff(s.faceVertexCounts.size() + 1, 0);
    for (size_t f = 0; f < s.faceVertexCounts.size(); ++f)
        cornerOff[f + 1] = cornerOff[f] + size_t(s.faceVertexCounts[f]);

    // Independent per-face expectations: fan, area, count.
    size_t const nFaces = s.faceVertexCounts.size();
    std::vector<std::vector<float>> faceW(nFaces);
    std::vector<GfVec3f> faceN(nFaces);
    std::vector<double> faceArea(nFaces);
    std::vector<uint64_t> faceNf(nFaces);
    size_t wantTotal = 0;
    for (size_t f = 0; f < nFaces; ++f) {
        int const nc = s.faceVertexCounts[f];
        TestFan(s.restPoints.cdata(), s.faceVertexIndices.cdata(),
                cornerOff[f], nc, &faceW[f], &faceN[f], &faceArea[f]);
        faceNf[f] = TestNf(seed, int(f), density, faceArea[f]);
        wantTotal += faceNf[f];
    }
    // Analytic anchors: unit quads take exactly 4/5 roots at density 4.5
    // (area exactly 1.0, frac exactly 0.5), the area-7 octagon 31/32,
    // the area-1.5 trapezoids 6/7, the area-3 triangles 13/14.
    for (size_t f = 0; f < 5041; ++f)
        CHECK(faceNf[f] == 4 || faceNf[f] == 5);
    CHECK(faceNf[5041] == 31 || faceNf[5041] == 32);
    for (size_t f = 5042; f < 5045; ++f)
        CHECK(faceNf[f] == 6 || faceNf[f] == 7);
    for (size_t f = 5045; f < 5047; ++f)
        CHECK(faceNf[f] == 13 || faceNf[f] == 14);

    UsdGenCurveBuffer out;
    CHECK(RunDirect(Desc(s, density, seed), &out));
    CHECK(out.totalCurves == wantTotal);
    CHECK(out.totalCvs == wantTotal);

    // Curve ids are exactly the per-face (f, k) sets: every captured id
    // is a valid CurveId of its own prim, with multiplicity one.
    std::map<uint64_t, std::pair<int, uint64_t>> idFace;
    for (size_t f = 0; f < nFaces; ++f)
        for (uint64_t k = 0; k < faceNf[f]; ++k)
            idFace[UsdGenCurveId(seed, uint32_t(f), uint32_t(k))] =
                std::make_pair(int(f), k);
    CHECK(idFace.size() == wantTotal);
    for (size_t i = 0; i < out.totalCurves; ++i) {
        auto it = idFace.find(out.curveId[i]);
        CHECK(it != idFace.end());
        CHECK(out.rootPrim[i] == it->second.first);
        idFace.erase(it);
    }
    CHECK(idFace.empty());

    // Rebuild the id map for the value check below.
    for (size_t f = 0; f < nFaces; ++f)
        for (uint64_t k = 0; k < faceNf[f]; ++k)
            idFace[UsdGenCurveId(seed, uint32_t(f), uint32_t(k))] =
                std::make_pair(int(f), k);

    // Bit-exact positions, uvs and frames for every captured root.
    for (size_t i = 0; i < out.totalCurves; ++i) {
        auto it = idFace.find(out.curveId[i]);
        CHECK(it != idFace.end());
        int const f = it->second.first;
        uint64_t const k = it->second.second;
        GfVec3f pos, T, N, B;
        GfVec2f puv;
        CHECK(TestRoot(s.restPoints.cdata(), s.uv.cdata(),
                       s.faceVertexIndices.cdata(), cornerOff[size_t(f)],
                       s.faceVertexCounts[size_t(f)], faceW[size_t(f)].data(),
                       faceArea[size_t(f)], faceN[size_t(f)], seed, f, k,
                       false, &pos, &puv, &T, &N, &B));
        CHECK(pos == GfVec3f(out.px[i], out.py[i], out.pz[i]));
        CHECK(puv == out.rootUV[i]);
        CHECK(T == out.rootT[i] && N == out.rootN[i] && B == out.rootB[i]);
        CHECK(out.hairT[i] == 0.0f);
    }

    // Morton order with face-major ties (stable sort: primary key, ties
    // in emission order).
    uint64_t prevKey = 0;
    bool first = true;
    int prevPrim = -1;
    uint64_t prevK = 0;
    for (size_t i = 0; i < out.totalCurves; ++i) {
        uint64_t const key = UsdGenMortonKey3(out.px[i], out.py[i], out.pz[i],
                                              64.0f);
        auto it = idFace.find(out.curveId[i]);
        CHECK(it != idFace.end());
        if (!first) {
            CHECK(prevKey <= key);
            if (prevKey == key) {
                CHECK(it->second.first > prevPrim ||
                      (it->second.first == prevPrim &&
                       it->second.second > prevK));
            }
        }
        first = false;
        prevKey = key;
        prevPrim = it->second.first;
        prevK = it->second.second;
    }

    // Thread-count stability: 1..8-worker arenas agree bit-for-bit
    // (one range vs many ranges; the partition count never shows).
    for (int workers = 1; workers <= 8; ++workers) {
        tbb::task_arena arena(workers);
        UsdGenCurveBuffer wide;
        bool ok = false;
        arena.execute([&] { ok = RunDirect(Desc(s, density, seed), &wide); });
        CHECK(ok);
        CHECK(SameRoots(out, wide));
    }

    // Flip negates exactly the tangent/bitangent planes.
    {
        UsdGenGraphDesc d = Desc(s, density, seed);
        d.nodes[0].params.push_back({TfToken("flip"), VtValue(true), false});
        UsdGenCurveBuffer flipped;
        CHECK(RunDirect(d, &flipped));
        CHECK(flipped.totalCurves == out.totalCurves);
        CHECK(flipped.px == out.px && flipped.py == out.py &&
              flipped.pz == out.pz && flipped.curveId == out.curveId &&
              flipped.rootPrim == out.rootPrim &&
              flipped.rootUV == out.rootUV && flipped.rootN == out.rootN);
        for (size_t i = 0; i < out.totalCurves; ++i)
            CHECK(flipped.rootT[i] == -out.rootT[i] &&
                  flipped.rootB[i] == -out.rootB[i]);
    }

    // Restricted capture past the threshold takes the two-pass path too.
    {
        UsdGenSurfaceDesc r = s;
        for (int f = 0; f < 4500; ++f)
            r.subsetFaces.push_back(f);
        UsdGenCurveBuffer sub;
        CHECK(RunDirect(Desc(r, density, seed), &sub));
        size_t wantSub = 0;
        for (int f = 0; f < 4500; ++f)
            wantSub += faceNf[size_t(f)];
        CHECK(sub.totalCurves == wantSub);
        for (size_t i = 0; i < sub.totalCurves; ++i)
            CHECK(sub.rootPrim[i] >= 0 && sub.rootPrim[i] < 4500);
    }

    // Error parity: malformed multipliers and count overflow fail closed
    // with the one-pass messages.
    {
        UsdGenGraphDesc d = Desc(s, density, seed);
        VtFloatArray badMult(nFaces - 1);
        for (size_t i = 0; i < badMult.size(); ++i)
            badMult[i] = 1.0f;
        d.surfaces[0].densityMultiplier = badMult;
        UsdGenCurveBuffer tmp;
        std::string error;
        CHECK(!RunDirect(d, &tmp, &error));
        CHECK(error.find("wrong face count") != std::string::npos);
    }
    {
        UsdGenGraphDesc d = Desc(s, 5.0e9, seed);
        UsdGenCurveBuffer tmp;
        std::string error;
        CHECK(!RunDirect(d, &tmp, &error));
        CHECK(error.find("exceeds uint32 cardinality") != std::string::npos);
    }

    std::printf("ok: scatter two-pass capture\n");
    return 0;
}
