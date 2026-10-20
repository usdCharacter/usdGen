// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// Strand helpers shared by the whole-strand operators (ops/strandUtil.h):
// reference random streams, the root/mid/tip Bezier profile, arc-length
// sampling, transport frames, root re-snap, and the triangle BVH against a
// brute-force closest-point scan.
#include "usdGen/ops/strandUtil.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::strand;

static int failures = 0;
static void Check(bool v, std::string const &s)
{
    if (!v) { ++failures; std::printf("FAIL: %s\n", s.c_str()); }
    else std::printf("ok: %s\n", s.c_str());
}
static bool Near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol; }

int main()
{
    // splitmix64 / U01: reference flyaway vector (strand index s, no seed).
    float const expected[8] = {0.883311f, 0.566562f, 0.591190f, 0.113450f,
                               0.431456f, 0.386768f, 0.739817f, 0.389830f};
    bool vec = true;
    for (int s = 0; s < 8; ++s) vec = vec && Near(Hash01(uint64_t(s)), expected[s], 1e-6f);
    Check(vec, "splitmix64 U01 reference vector");

    // std::mt19937(1337) is the standard engine on every platform.
    std::mt19937 rng(1337);
    uint32_t const a = rng(), b = rng(), c = rng();
    Check(a == 1125387415u && b == 2407456957u && c == 681542492u, "mt19937(1337) stream");

    // MSVC-compatible shuffle is a permutation and deterministic.
    std::vector<int> v1(100), v2(100);
    for (int i = 0; i < 100; ++i) v1[size_t(i)] = v2[size_t(i)] = i;
    std::mt19937 r1(42), r2(42);
    MsvcShuffle(&v1, r1);
    MsvcShuffle(&v2, r2);
    std::vector<int> sorted = v1;
    std::sort(sorted.begin(), sorted.end());
    bool perm = true;
    for (int i = 0; i < 100; ++i) perm = perm && sorted[size_t(i)] == i;
    Check(perm && v1 == v2, "MsvcShuffle is a deterministic permutation");
    bool bounded = true;
    std::mt19937 r3(7);
    for (int i = 0; i < 1000; ++i) bounded = bounded && MsvcUniformIndex(r3, 6u) <= 6u;
    Check(bounded, "MsvcUniformIndex stays in [0, bound]");

    // Bezier profile: collinear controls with midShift 0.5 are linear.
    Check(Near(BezierProfile(0.0f, 0, 0.5f, 0.5f, 1), 0.0f) &&
          Near(BezierProfile(0.25f, 0, 0.5f, 0.5f, 1), 0.25f) &&
          Near(BezierProfile(1.0f, 0, 0.5f, 0.5f, 1), 1.0f), "Bezier profile linear case");
    // Endpoints are root and tip for any shift; mid is a control value.
    Check(Near(BezierProfile(0.0f, 0.2f, 1.0f, 0.1f, 0.7f), 0.2f) &&
          Near(BezierProfile(1.0f, 0.2f, 1.0f, 0.9f, 0.7f), 0.7f), "Bezier endpoints");
    float const mid = BezierProfile(0.5f, 0.0f, 1.0f, 0.5f, 0.0f);
    Check(Near(mid, 0.5f), "Bezier mid is a control (0.25r + 0.5m + 0.25t)");

    // Arc length, sampling and extrapolation along a polyline.
    float const line[9] = {0, 0, 0, 0, 1, 0, 0, 3, 0};
    std::vector<float> cum;
    float const L = ArcLengths(line, 3, &cum);
    GfVec3f const p1 = SampleArc(line, 3, cum, 2.0f);
    GfVec3f const p2 = SampleArc(line, 3, cum, 4.0f);
    Check(Near(L, 3.0f) && Near(p1[1], 2.0f) && Near(p2[1], 4.0f), "SampleArc interpolate/extrapolate");
    float res[3 * 4];
    ResampleArc(line, 3, cum, 0.0f, 3.0f, 4, res);
    Check(Near(res[4], 1.0f) && Near(res[7], 2.0f) && Near(res[10], 3.0f), "ResampleArc uniform");

    // Transport frames stay orthonormal along a bent curve.
    std::vector<GfVec3f> T = {GfVec3f(0, 1, 0), Normalize(GfVec3f(1, 1, 0), GfVec3f(0, 1, 0)),
                              GfVec3f(1, 0, 0)};
    std::vector<GfVec3f> N, B;
    TransportFrames(T, &N, &B);
    bool ortho = true;
    for (size_t i = 0; i < T.size(); ++i)
        ortho = ortho && std::fabs(GfDot(N[i], T[i])) < 1e-5f && Near(N[i].GetLength(), 1.0f) &&
                std::fabs(GfDot(B[i], N[i])) < 1e-5f;
    Check(ortho, "TransportFrames orthonormal");

    // Root re-snap moves the root fully and leaves far CVs alone.
    float strandPts[3 * 5] = {0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0};
    ResnapRoot(strandPts, 5, GfVec3f(0.1f, 0, 0));
    Check(Near(strandPts[0], 0.1f) && Near(strandPts[3], 0.0f) && Near(strandPts[12], 0.0f),
          "ResnapRoot falloff within 4|delta|");

    // BVH closest point equals brute force on a bumpy grid.
    int const G = 12;
    VtVec3fArray pts;
    VtIntArray counts, indices;
    for (int j = 0; j <= G; ++j)
        for (int i = 0; i <= G; ++i)
            pts.push_back(GfVec3f(float(i), 0.3f * std::sin(float(i + 2 * j)), float(j)));
    for (int j = 0; j < G; ++j)
        for (int i = 0; i < G; ++i) {
            int const v = j * (G + 1) + i;
            counts.push_back(4);
            indices.push_back(v); indices.push_back(v + 1);
            indices.push_back(v + G + 2); indices.push_back(v + G + 1);
        }
    std::vector<FanTriangle> tris;
    FanTriangulate(counts, indices, &tris);
    Check(tris.size() == size_t(2 * G * G) && tris[1].face == 0 && tris[1].v[1] == indices[2],
          "FanTriangulate order (f0, fk, fk+1)");
    TriangleBvh bvh;
    bvh.Build(pts, tris);
    std::mt19937 qr(3);
    bool same = true;
    for (int q = 0; q < 300; ++q) {
        GfVec3f const p(Mt01(qr) * 14 - 1, Mt01(qr) * 4 - 2, Mt01(qr) * 14 - 1);
        ClosestHit const hit = bvh.Closest(p);
        float best = 3e38f;
        for (size_t t = 0; t < tris.size(); ++t) {
            // Brute force: a one-triangle hierarchy is an exact triangle query.
            TriangleBvh single;
            single.Build(pts, std::vector<FanTriangle>{tris[t]});
            best = std::min(best, single.Closest(p).distanceSq);
        }
        same = same && hit.hit && std::fabs(hit.distanceSq - best) <= 1e-5f * (1.0f + best);
    }
    Check(same, "TriangleBvh closest point equals per-triangle minimum");

    std::printf(failures ? "FAILED (%d)\n" : "strand util tests passed\n", failures);
    return failures ? 1 : 0;
}
