// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenScatter generation-order distribution modes. See
// scatterModes.h for the parameter set.
//
// Reference behaviour reproduced here (float32 arithmetic in the reference
// order, std::mt19937 draws uniform01 = (mt() >> 8) * 2^-24):
//   * area CDF over the fan triangles of the bound surface in face order;
//     pickTri(u) = first index whose CDF value is >= u, clamped;
//   * makeRoot(tri, a, b): fold a + b > 1 -> (1 - a, 1 - b), barycentrics
//     (1 - a - b, a, b), position / normal / uv blended with those weights;
//   * guide proximity: a root is accepted when some guide root lies within
//     sqrt(cutoff2); Distance mode builds a triangle domain around the guides
//     and calibrates the effective area with mt19937(90210) x 4096 samples,
//     Coverage mode takes the cutoff from 2000 mt19937(42) area samples;
//   * target = clamp(roundf(A * density), 1, 2e6) over the whole bound
//     surface; proximity confines roots, it never lowers the target;
//   * uvGrid: deterministic UV-space grid (first containing triangle);
//     vertex: shuffled vertices, duplicate positions once; even: Poisson-disk
//     dart throwing with relaxation; perGuide: one root per guide root;
//   * density maps thin roots AFTER generation with the root existence hash,
//     so a painted mask never renumbers strands (strandIndex keeps the
//     generation index).
#include "usdGen/ops/scatterModes.h"

#include "usdGen/limitSurface.h"

#include "usdGen/ops/opUtil.h"
#include "usdGen/ops/strandUtil.h"
#include "usdGenMath/usdGenMath/hash.h"
#include "usdGenMath/usdGenMath/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace scatter {

namespace {

constexpr uint32_t kSaltScatterModes = 0x53634D64u;  // "ScMd": generation-order curve ids
constexpr uint32_t kSaltScatterThin = 0x5363546Eu;   // "ScTn": random-path thinning
constexpr uint32_t kMaxTarget = 2000000u;
constexpr size_t kSamplerBatch = 16384;

TfToken const &DensityMapRelationship()
{
    static TfToken const t("usdGen:densityMap");
    return t;
}

inline uint32_t FloatBits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// (P0 w0 + P1 w1) + P2 w2 per component: the reference's float order.
inline GfVec3f Blend3(GfVec3f const &p0, GfVec3f const &p1, GfVec3f const &p2,
                      float w0, float w1, float w2)
{
    return GfVec3f(p0[0] * w0 + p1[0] * w1 + p2[0] * w2,
                   p0[1] * w0 + p1[1] * w1 + p2[1] * w2,
                   p0[2] * w0 + p1[2] * w1 + p2[2] * w2);
}

inline GfVec2f Blend2(GfVec2f const &p0, GfVec2f const &p1, GfVec2f const &p2,
                      float w0, float w1, float w2)
{
    return GfVec2f(p0[0] * w0 + p1[0] * w1 + p2[0] * w2,
                   p0[1] * w0 + p1[1] * w1 + p2[1] * w2);
}

inline float Dist2(GfVec3f const &a, GfVec3f const &b)
{
    float const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

inline GfVec3f Unit(GfVec3f const &v, GfVec3f const &fallback)
{
    float const l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return l > 1e-12f ? v / l : fallback;
}

/// First index whose value is >= u, clamped to the last element.
size_t LowerBound(std::vector<float> const &cdf, float u)
{
    if (cdf.empty()) return 0;
    size_t const k = size_t(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
    return std::min(k, cdf.size() - 1);
}

inline uint64_t CellKey(int64_t x, int64_t y, int64_t z)
{
    return (uint64_t(x) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(y) * 0xC2B2AE3D27D4EB4Full) ^
           (uint64_t(z) * 0x165667B19E3779F9ull);
}

// ---------------------------------------------------------------------------
// Bound surface: fan triangles in face order, float32 areas and CDF, corner
// normals / UVs.
// ---------------------------------------------------------------------------

struct Root
{
    GfVec3f pos{0.0f};
    GfVec3f normal{0.0f, 1.0f, 0.0f};
    GfVec2f uv{0.0f};
    int tri = -1;                    // local triangle: mask sampling and frame
    GfVec3f sampleBary{1.0f, 0.0f, 0.0f};
    int hashTri = -1;                // whole-mesh fan index for the root hash
    int vertex = -1;
    GfVec3f hashBary{0.0f};
    int face = -1;
    uint64_t uid = 0;                // nonzero: guide id (perGuide)
};

struct Surface
{
    VtVec3fArray points;
    std::vector<strand::FanTriangle> tris;
    std::vector<int> globalTri;
    std::vector<float> area;
    float areaTotal = 0.0f;
    std::vector<float> cdf;              // running sum of area / areaTotal
    std::vector<GfVec3f> cornerNormal;   // 3 per triangle
    std::vector<GfVec2f> cornerUv;       // 3 per triangle, empty without UVs
    std::vector<GfVec3f> vertexNormal;   // per point
    std::vector<GfVec2f> vertexUv;       // per point, empty without UVs
    std::vector<int> vertexTri;          // first incident triangle, -1 none
    std::vector<int8_t> vertexCorner;    // its corner slot
    strand::TriangleBvh bvh;
    bool bvhBuilt = false;

    bool HasUv() const { return !cornerUv.empty(); }
    GfVec3f const &P(size_t t, int k) const { return points[size_t(tris[t].v[k])]; }

    void Build(UsdGenSurfaceDesc const &s)
    {
        points = s.restPoints;
        VtIntArray const &counts = s.faceVertexCounts;
        VtIntArray const &indices = s.faceVertexIndices;
        bool const restricted = UsdGenSurfaceRestricted(s);
        std::vector<strand::FanTriangle> all;
        strand::FanTriangulate(counts, indices, &all, restricted ? &s.subsetFaces : nullptr);
        // Whole-mesh fan index (the reference numbers triangles over the
        // full mesh in face order, whatever subset the roots come from).
        std::vector<size_t> faceStart(counts.size() + 1, 0), triStart(counts.size() + 1, 0);
        for (size_t f = 0; f < counts.size(); ++f) {
            int const k = counts[f];
            faceStart[f + 1] = faceStart[f] + size_t(std::max(0, k));
            bool const ok = k >= 3 && faceStart[f] + size_t(k) <= indices.size();
            triStart[f + 1] = triStart[f] + (ok ? size_t(k - 2) : 0);
        }
        size_t const np = points.size();
        tris.clear();
        globalTri.clear();
        for (strand::FanTriangle const &t : all) {
            bool ok = true;
            for (int k = 0; k < 3; ++k) ok = ok && t.v[k] >= 0 && size_t(t.v[k]) < np;
            if (!ok) continue;
            tris.push_back(t);
            globalTri.push_back(int(triStart[size_t(t.face)] +
                                    size_t(t.corner[1]) - faceStart[size_t(t.face)] - 1));
        }
        size_t const nt = tris.size();
        area.assign(nt, 0.0f);
        float total = 0.0f;
        for (size_t i = 0; i < nt; ++i) {
            GfVec3f const e1 = P(i, 1) - P(i, 0);
            GfVec3f const e2 = P(i, 2) - P(i, 0);
            GfVec3f const x = GfCross(e1, e2);
            area[i] = 0.5f * std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
            total += area[i];
        }
        areaTotal = total;
        cdf.assign(nt, 0.0f);
        float acc = 0.0f;
        for (size_t i = 0; i < nt; ++i) {
            acc += total > 0.0f ? area[i] / total : 0.0f;
            cdf[i] = acc;
        }

        vertexTri.assign(np, -1);
        vertexCorner.assign(np, 0);
        for (size_t i = 0; i < nt; ++i)
            for (int k = 0; k < 3; ++k) {
                size_t const v = size_t(tris[i].v[k]);
                if (vertexTri[v] < 0) { vertexTri[v] = int(i); vertexCorner[v] = int8_t(k); }
            }

        // Normals: the authored rest normals (vertex or faceVarying), else
        // area-weighted vertex normals.
        VtVec3fArray const &rn = s.restNormals;
        bool const nVertex = s.restNormalDomain == UsdGenSurfaceNormalDomain::Vertex &&
                             rn.size() == np;
        bool const nFaceVarying = s.restNormalDomain == UsdGenSurfaceNormalDomain::FaceVarying &&
                                  rn.size() == indices.size();
        cornerNormal.assign(3 * nt, GfVec3f(0.0f, 1.0f, 0.0f));
        vertexNormal.assign(np, GfVec3f(0.0f, 1.0f, 0.0f));
        if (nVertex || nFaceVarying) {
            for (size_t i = 0; i < nt; ++i)
                for (int k = 0; k < 3; ++k)
                    cornerNormal[3 * i + size_t(k)] =
                        rn[size_t(nVertex ? tris[i].v[k] : tris[i].corner[k])];
            for (size_t v = 0; v < np; ++v) {
                if (vertexTri[v] < 0) continue;
                vertexNormal[v] = cornerNormal[3 * size_t(vertexTri[v]) + size_t(vertexCorner[v])];
            }
        } else {
            std::vector<GfVec3d> vn(np, GfVec3d(0.0));
            for (size_t i = 0; i < nt; ++i) {
                GfVec3d const a(P(i, 0)), b(P(i, 1)), c(P(i, 2));
                GfVec3d const fn = GfCross(b - a, c - a);
                for (int k = 0; k < 3; ++k) vn[size_t(tris[i].v[k])] += fn;
            }
            for (size_t v = 0; v < np; ++v) {
                double const l = vn[v].GetLength();
                if (l > 1e-300) vertexNormal[v] = GfVec3f(vn[v] / l);
            }
            for (size_t i = 0; i < nt; ++i)
                for (int k = 0; k < 3; ++k)
                    cornerNormal[3 * i + size_t(k)] = vertexNormal[size_t(tris[i].v[k])];
        }

        // UVs: vertex-interpolated (the imported layout) or faceVarying.
        cornerUv.clear();
        vertexUv.clear();
        VtVec2fArray const &uv = s.uv;
        bool const uvVertex = !uv.empty() && uv.size() == np;
        bool const uvFaceVarying = !uv.empty() && !uvVertex && uv.size() == indices.size();
        if (uvVertex || uvFaceVarying) {
            cornerUv.resize(3 * nt);
            for (size_t i = 0; i < nt; ++i)
                for (int k = 0; k < 3; ++k)
                    cornerUv[3 * i + size_t(k)] =
                        uv[size_t(uvVertex ? tris[i].v[k] : tris[i].corner[k])];
            vertexUv.assign(np, GfVec2f(0.0f));
            for (size_t v = 0; v < np; ++v) {
                if (uvVertex) vertexUv[v] = uv[v];
                else if (vertexTri[v] >= 0)
                    vertexUv[v] = cornerUv[3 * size_t(vertexTri[v]) + size_t(vertexCorner[v])];
            }
        }
    }

    void EnsureBvh()
    {
        if (bvhBuilt) return;
        bvh.Build(points, tris, cornerNormal, cornerUv);
        bvhBuilt = true;
    }

    GfVec3f FaceNormal(size_t t) const
    {
        GfVec3f const x = GfCross(P(t, 1) - P(t, 0), P(t, 2) - P(t, 0));
        return Unit(x, GfVec3f(0.0f, 1.0f, 0.0f));
    }

    GfVec3f InterpNormal(size_t t, float w0, float w1, float w2) const
    {
        GfVec3f const n = Blend3(cornerNormal[3 * t], cornerNormal[3 * t + 1],
                                 cornerNormal[3 * t + 2], w0, w1, w2);
        float const l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        return l > 1e-12f ? n / l : FaceNormal(t);
    }

    /// The reference root at barycentrics (w0, w1, w2) of triangle t.
    Root RootAt(size_t t, float w0, float w1, float w2) const
    {
        Root r;
        r.tri = int(t);
        r.hashTri = globalTri[t];
        r.sampleBary = r.hashBary = GfVec3f(w0, w1, w2);
        r.pos = Blend3(P(t, 0), P(t, 1), P(t, 2), w0, w1, w2);
        if (HasUv())
            r.uv = Blend2(cornerUv[3 * t], cornerUv[3 * t + 1], cornerUv[3 * t + 2], w0, w1, w2);
        r.normal = InterpNormal(t, w0, w1, w2);
        r.face = tris[t].face;
        return r;
    }

    /// makeRoot(tri, a, b): fold, then barycentrics (1 - a - b, a, b).
    Root MakeRoot(size_t t, float a, float b) const
    {
        if (a + b > 1.0f) {
            a = 1.0f - a;
            b = 1.0f - b;
        }
        float const w0 = 1.0f - a - b;
        return RootAt(t, w0, a, b);
    }

    GfVec3f SamplePoint(size_t t, float a, float b) const
    {
        if (a + b > 1.0f) {
            a = 1.0f - a;
            b = 1.0f - b;
        }
        return Blend3(P(t, 0), P(t, 1), P(t, 2), 1.0f - a - b, a, b);
    }
};

// ---------------------------------------------------------------------------
// Guides and proximity
// ---------------------------------------------------------------------------

struct Guides
{
    std::vector<GfVec3f> roots;      // rest root of each eligible guide, in order
    std::vector<uint64_t> ids;
    std::vector<int> skinPrim;
    std::vector<GfVec2f> skinUv;
};

UsdGenCurveBuffer const *GuideBuffer(UsdGenCaptureContext const &ctx)
{
    if (ctx.referenceCount == 0) return nullptr;
    if (ctx.resolvedReferences && ctx.resolvedReferences[0] && ctx.resolvedReferences[0]->value)
        return &ctx.resolvedReferences[0]->value->buffer;
    if (ctx.references && ctx.references[0]) return &ctx.references[0]->buffer;
    return nullptr;
}

/// Eligible guides (>= 2 points, polyline length > 1e-6), in curve order.
bool LoadGuides(UsdGenCaptureContext const &ctx, Guides *out, std::string *error)
{
    UsdGenCurveBuffer const *b = GuideBuffer(ctx);
    if (!b || b->totalCurves == 0) return true;
    std::vector<uint32_t> spans;
    if (!opUtil::CurveSpans(*b, &spans, error)) return false;
    bool const hasRest = b->rest.size() == b->totalCvs;
    if (!hasRest && (b->px.size() != b->totalCvs || b->py.size() != b->totalCvs ||
                     b->pz.size() != b->totalCvs)) {
        if (error) *error = "guide curve set has no point planes";
        return false;
    }
    auto point = [&](size_t cv) { return hasRest ? b->rest[cv] : opUtil::Point(*b, cv); };
    for (uint32_t c = 0; c < b->totalCurves; ++c) {
        size_t const first = spans[c], n = size_t(spans[c + 1] - spans[c]);
        if (n < 2) continue;
        float length = 0.0f;
        for (size_t k = 1; k < n; ++k) length += std::sqrt(Dist2(point(first + k), point(first + k - 1)));
        if (!(length > 1e-6f)) continue;
        out->roots.push_back(point(first));
        out->ids.push_back(b->curveId.size() == b->totalCurves ? b->curveId[c] : uint64_t(c));
        out->skinPrim.push_back(b->rootPrim.size() == b->totalCurves ? b->rootPrim[c] : -1);
        out->skinUv.push_back(b->rootUV.size() == b->totalCurves ? b->rootUV[c] : GfVec2f(0.0f));
    }
    return true;
}

struct Proximity
{
    bool active = false;
    float cutoff2 = -1.0f;
    std::vector<GfVec3f> roots;
    double inv = 0.0;
    std::unordered_map<uint64_t, std::vector<uint32_t>> grid;

    void BuildGrid()
    {
        double const cell = std::max(1e-9, std::sqrt(double(cutoff2)) * 1.0625);
        inv = 1.0 / cell;
        grid.clear();
        for (size_t i = 0; i < roots.size(); ++i) {
            GfVec3f const &g = roots[i];
            grid[CellKey(int64_t(std::floor(g[0] * inv)), int64_t(std::floor(g[1] * inv)),
                         int64_t(std::floor(g[2] * inv)))].push_back(uint32_t(i));
        }
    }

    /// Some guide root within sqrt(cutoff2): the 27 cells around p.
    bool Within(GfVec3f const &p) const
    {
        int64_t const ix = int64_t(std::floor(p[0] * inv));
        int64_t const iy = int64_t(std::floor(p[1] * inv));
        int64_t const iz = int64_t(std::floor(p[2] * inv));
        for (int64_t dz = -1; dz <= 1; ++dz)
            for (int64_t dy = -1; dy <= 1; ++dy)
                for (int64_t dx = -1; dx <= 1; ++dx) {
                    auto const it = grid.find(CellKey(ix + dx, iy + dy, iz + dz));
                    if (it == grid.end()) continue;
                    for (uint32_t g : it->second)
                        if (Dist2(p, roots[g]) <= cutoff2) return true;
                }
        return false;
    }

    bool Accept(GfVec3f const &p) const
    {
        if (!active || cutoff2 < 0.0f) return true;
        if (roots.empty()) return false;
        return Within(p);
    }

    /// Some guide root g with |g - c| <= R (the domain test).
    bool AnyWithin(GfVec3f const &c, float R) const
    {
        double const span = 2.0 * double(R) * inv + 3.0;
        if (span * span * span > double(std::max<size_t>(roots.size(), 27))) {
            for (GfVec3f const &g : roots)
                if (std::sqrt(Dist2(g, c)) <= R) return true;
            return false;
        }
        int64_t lo[3], hi[3];
        for (int k = 0; k < 3; ++k) {
            lo[k] = int64_t(std::floor((c[k] - R) * inv)) - 1;
            hi[k] = int64_t(std::floor((c[k] + R) * inv)) + 1;
        }
        for (int64_t z = lo[2]; z <= hi[2]; ++z)
            for (int64_t y = lo[1]; y <= hi[1]; ++y)
                for (int64_t x = lo[0]; x <= hi[0]; ++x) {
                    auto const it = grid.find(CellKey(x, y, z));
                    if (it == grid.end()) continue;
                    for (uint32_t g : it->second)
                        if (std::sqrt(Dist2(roots[g], c)) <= R) return true;
                }
        return false;
    }
};

/// Coverage cutoff^2: the `coverage` quantile (linear interpolation) of the
/// squared distance to the nearest guide root over 2000 area-uniform samples
/// drawn with mt19937(42). This CDF divides the running SUM by the total.
float CoverageCutoff2(Surface const &s, std::vector<GfVec3f> const &roots, float coverage)
{
    size_t const nt = s.tris.size();
    if (nt == 0 || roots.empty()) return 0.0f;
    std::vector<float> cdf(nt);
    float run = 0.0f;
    for (size_t i = 0; i < nt; ++i) {
        run += s.area[i];
        cdf[i] = s.areaTotal > 0.0f ? run / s.areaTotal : 0.0f;
    }
    std::mt19937 rng(42u);
    std::vector<float> d2(2000);
    for (size_t k = 0; k < 2000; ++k) {
        float const u = strand::Mt01(rng);
        size_t const t = LowerBound(cdf, u);
        float const a = strand::Mt01(rng);
        float const b = strand::Mt01(rng);
        GfVec3f const p = s.SamplePoint(t, a, b);
        float best = std::numeric_limits<float>::max();
        for (GfVec3f const &g : roots) best = std::min(best, Dist2(p, g));
        d2[k] = best;
    }
    std::sort(d2.begin(), d2.end());
    float const x = coverage * 1999.0f;
    int const i = std::clamp(int(std::floor(x)), 0, 1999);
    int const j = std::min(i + 1, 1999);
    float const f = x - float(i);
    return d2[size_t(i)] * (1.0f - f) + d2[size_t(j)] * f;
}

void SetupProximity(Surface const &s, Guides const &guides, ModeConfig const &c,
                    Proximity *p)
{
    bool const enabled = c.proximity != ProximityMode::None &&
                         c.distribution != Distribution::PerGuide && !guides.roots.empty() &&
                         (c.proximity == ProximityMode::Distance || c.proximityCoverage < 0.9999f);
    p->active = enabled;
    p->cutoff2 = -1.0f;
    if (!enabled) return;
    p->roots = guides.roots;
    if (c.proximity == ProximityMode::Distance) {
        float const d = std::max(0.0f, c.proximityDistance);
        p->cutoff2 = d * d;
    } else {
        p->cutoff2 = CoverageCutoff2(s, guides.roots,
                                     std::min(std::max(c.proximityCoverage, 0.0f), 1.0f));
    }
    p->BuildGrid();
}

/// Triangle sampler: the proximity domain when one exists, else every
/// triangle of the bound surface.
struct Picker
{
    Surface const *s = nullptr;
    std::vector<int> domain;
    std::vector<float> cdfDom;

    size_t operator()(float u) const
    {
        if (!domain.empty()) return size_t(domain[LowerBound(cdfDom, u)]);
        return LowerBound(s->cdf, u);
    }
};

/// Distance-mode domain: triangles of positive area whose bounding sphere,
/// grown by the cutoff radius, reaches a guide root.
float BuildDomain(Surface const &s, Proximity const &prox, Picker *pick)
{
    float const Rg = std::sqrt(prox.cutoff2);
    GfVec3f lo(std::numeric_limits<float>::max()), hi(-std::numeric_limits<float>::max());
    for (GfVec3f const &g : prox.roots)
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], g[k]);
            hi[k] = std::max(hi[k], g[k]);
        }
    float aDom = 0.0f;
    for (size_t i = 0; i < s.tris.size(); ++i) {
        if (!(s.area[i] > 0.0f)) continue;
        GfVec3f const c = (s.P(i, 0) + s.P(i, 1) + s.P(i, 2)) / 3.0f;
        float rad = 0.0f;
        for (int k = 0; k < 3; ++k) rad = std::max(rad, std::sqrt(Dist2(s.P(i, k), c)));
        float const R = rad + Rg;
        // Sphere / guide-bounds overlap: a fast reject implied by the test below.
        float box = 0.0f;
        for (int k = 0; k < 3; ++k) {
            float const d = c[k] < lo[k] ? lo[k] - c[k] : (c[k] > hi[k] ? c[k] - hi[k] : 0.0f);
            box += d * d;
        }
        if (box > R * R) continue;
        if (!prox.AnyWithin(c, R)) continue;
        pick->domain.push_back(int(i));
        aDom += s.area[i];
    }
    if (aDom <= 1e-9f) {
        pick->domain.clear();
        return 0.0f;
    }
    pick->cdfDom.resize(pick->domain.size());
    float acc = 0.0f;
    for (size_t k = 0; k < pick->domain.size(); ++k) {
        acc += s.area[size_t(pick->domain[k])] / aDom;
        pick->cdfDom[k] = acc;
    }
    return aDom;
}

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------

/// Area-weighted random darts (uvGrid without UVs, random + surface count):
/// per attempt draw u, a, b; up to max(200 * target, 5000) attempts.
void DartScatter(Surface const &s, Picker const &pick, Proximity const &prox, uint32_t target,
                 std::mt19937 &rng, std::vector<Root> *out)
{
    if (s.tris.empty()) return;
    uint64_t const attempts = std::max<uint64_t>(200ull * target, 5000ull);
    for (uint64_t k = 0; k < attempts && out->size() < target; ++k) {
        float const u = strand::Mt01(rng);
        size_t const t = pick(u);
        float const a = strand::Mt01(rng);
        float const b = strand::Mt01(rng);
        Root r = s.MakeRoot(t, a, b);
        if (prox.Accept(r.pos)) out->push_back(r);
    }
}

/// Deterministic UV-space grid: n = ceil(sqrt(target)) points per side,
/// row-major (v outer); each lands on the FIRST triangle (in triangle
/// order) whose UV barycentrics are all >= -1e-4. Triangles are bucketed on
/// a G x G grid (G = clamp(sqrt(#tris), 16, 256)) by their UV bounds; each
/// triangle is shifted by the integer tile of its lowest corner so UDIM
/// tiles fold onto [0,1)^2 (inferred detail).
void UvGridScatter(Surface const &s, Proximity const &prox, uint32_t target,
                   std::vector<Root> *out)
{
    size_t const nt = s.tris.size();
    if (target == 0 || nt == 0) return;
    uint32_t const n = uint32_t(std::ceil(std::sqrt(double(target))));
    int const G = std::clamp(int(std::sqrt(double(nt))), 16, 256);
    std::vector<std::vector<uint32_t>> cells(size_t(G) * size_t(G));
    std::vector<GfVec2f> tuv(3 * nt);
    auto cellOf = [G](float x) {
        float const c = std::floor(float(G) * x);
        return c < 0.0f ? 0 : (c >= float(G) ? G - 1 : int(c));
    };
    for (size_t t = 0; t < nt; ++t) {
        GfVec2f const a = s.cornerUv[3 * t], b = s.cornerUv[3 * t + 1], c = s.cornerUv[3 * t + 2];
        float const su = std::floor(std::min({a[0], b[0], c[0]}));
        float const sv = std::floor(std::min({a[1], b[1], c[1]}));
        GfVec2f const shift(su, sv);
        tuv[3 * t] = a - shift;
        tuv[3 * t + 1] = b - shift;
        tuv[3 * t + 2] = c - shift;
        float u0 = std::numeric_limits<float>::max(), v0 = u0, u1 = -u0, v1 = -u0;
        for (int k = 0; k < 3; ++k) {
            GfVec2f const &q = tuv[3 * t + size_t(k)];
            u0 = std::min(u0, q[0]); u1 = std::max(u1, q[0]);
            v0 = std::min(v0, q[1]); v1 = std::max(v1, q[1]);
        }
        if (!std::isfinite(u0) || !std::isfinite(u1) || !std::isfinite(v0) || !std::isfinite(v1))
            continue;
        int const x0 = cellOf(u0), x1 = cellOf(u1), y0 = cellOf(v0), y1 = cellOf(v1);
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
                cells[size_t(y) * size_t(G) + size_t(x)].push_back(uint32_t(t));
    }
    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            if (out->size() >= target) return;
            GfVec2f const q((float(col) + 0.5f) / float(n), (float(row) + 0.5f) / float(n));
            int const cx = cellOf(q[0] - std::floor(q[0]));
            int const cy = cellOf(q[1] - std::floor(q[1]));
            for (uint32_t t : cells[size_t(cy) * size_t(G) + size_t(cx)]) {
                GfVec2f const &a = tuv[3 * size_t(t)];
                GfVec2f const v0 = tuv[3 * size_t(t) + 1] - a;
                GfVec2f const v1 = tuv[3 * size_t(t) + 2] - a;
                GfVec2f const v2 = q - a;
                float const d00 = v0[0] * v0[0] + v0[1] * v0[1];
                float const d01 = v0[0] * v1[0] + v0[1] * v1[1];
                float const d11 = v1[0] * v1[0] + v1[1] * v1[1];
                float const d20 = v2[0] * v0[0] + v2[1] * v0[1];
                float const d21 = v2[0] * v1[0] + v2[1] * v1[1];
                float const den = d00 * d11 - d01 * d01;
                if (!(std::fabs(den) >= 1e-12f)) continue;
                float const w1 = (d11 * d20 - d01 * d21) / den;
                float const w2 = (d00 * d21 - d01 * d20) / den;
                float const w0 = 1.0f - w1 - w2;
                if (w0 >= -1e-4f && w1 >= -1e-4f && w2 >= -1e-4f) {
                    Root r = s.RootAt(t, w0, w1, w2);
                    if (prox.Accept(r.pos)) out->push_back(r);
                    break;  // first containing triangle only
                }
            }
        }
    }
}

/// Roots exactly at mesh vertices, in MSVC-shuffled vertex order; positions
/// that repeat (seam duplicates) are emitted once.
void VertexScatter(Surface const &s, Proximity const &prox, uint32_t target, std::mt19937 &rng,
                   std::vector<Root> *out)
{
    size_t const nv = s.points.size();
    std::vector<uint32_t> order(nv);
    std::iota(order.begin(), order.end(), 0u);
    strand::MsvcShuffle(&order, rng);
    std::unordered_set<uint64_t> seen;
    for (uint32_t v : order) {
        if (out->size() >= target) break;
        int const t = s.vertexTri[v];
        if (t < 0) continue;  // not on the bound surface
        GfVec3f const &p = s.points[v];
        if (!prox.Accept(p)) continue;
        uint64_t key = (uint64_t(FloatBits(p[0])) ^ 0x14650fb0739d0383ull) * 0x100000001b3ull;
        key = (key ^ uint64_t(FloatBits(p[1]))) * 0x100000001b3ull;
        key = (key ^ uint64_t(FloatBits(p[2]))) * 0x100000001b3ull;
        if (!seen.insert(key).second) continue;
        Root r;
        r.pos = p;
        r.vertex = int(v);
        r.tri = t;
        r.sampleBary = GfVec3f(0.0f);
        r.sampleBary[s.vertexCorner[v]] = 1.0f;
        r.hashTri = -1;
        r.hashBary = GfVec3f(0.0f);  // inferred: unset record field
        r.normal = Unit(s.vertexNormal[v], GfVec3f(0.0f, 1.0f, 0.0f));
        if (s.HasUv()) r.uv = s.vertexUv[v];
        r.face = s.tris[size_t(t)].face;
        out->push_back(r);
    }
}

/// Poisson-disk dart throwing with relaxation (reference "Even").
void EvenScatter(Surface const &s, Picker const &pick, Proximity const &prox, uint32_t target,
                 float r0, std::mt19937 &rng, std::vector<Root> *out, ModeStats *st)
{
    if (target == 0 || s.tris.empty()) return;
    GfVec3f lo(std::numeric_limits<float>::max()), hi(-std::numeric_limits<float>::max());
    for (GfVec3f const &p : s.points)
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], p[k]);
            hi[k] = std::max(hi[k], p[k]);
        }
    float const diag = std::sqrt(Dist2(hi, lo));
    float const minSp = std::max(1e-7f, 1e-7f * diag);

    // The sampler draws u, a, b for every candidate of a batch first, then
    // evaluates them; `fn` returns false to stop (the batch's remaining draws
    // are consumed regardless).
    std::vector<float> draws;
    auto sample = [&](uint64_t count, auto &&fn) {
        uint64_t remaining = count;
        while (remaining > 0) {
            size_t const batch = size_t(std::min<uint64_t>(kSamplerBatch, remaining));
            draws.resize(3 * batch);
            for (size_t j = 0; j < 3 * batch; ++j) draws[j] = strand::Mt01(rng);
            remaining -= batch;
            for (size_t j = 0; j < batch; ++j) {
                Root const r = s.MakeRoot(pick(draws[3 * j]), draws[3 * j + 1], draws[3 * j + 2]);
                if (!fn(r, prox.Accept(r.pos))) return;
            }
        }
    };

    uint64_t const nc = std::max<uint64_t>(2ull * target, 4000ull);
    std::vector<Root> pool;
    pool.reserve(size_t(nc));
    sample(nc, [&](Root const &r, bool valid) {
        if (valid) pool.push_back(r);
        return true;
    });
    st->candidates = uint32_t(nc);
    st->accepted = uint32_t(pool.size());
    std::vector<uint32_t> order(pool.size());
    std::iota(order.begin(), order.end(), 0u);
    strand::MsvcShuffle(&order, rng);

    struct Key {
        long long x, y, z;
        bool operator==(Key const &o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct KeyHash {
        size_t operator()(Key const &k) const
        {
            return size_t(CellKey(int64_t(k.x), int64_t(k.y), int64_t(k.z)));
        }
    };
    std::unordered_set<Key, KeyHash> dedup;
    dedup.reserve(size_t(target) * 2);
    std::unordered_map<uint64_t, std::vector<uint32_t>> grid;
    double inv = 0.0;
    float r = r0;
    auto cellIndex = [&](float x) { return int64_t(std::floor(double(x) * inv)); };
    auto rebuild = [&]() {
        inv = 1.0 / double(r);
        grid.clear();
        for (size_t i = 0; i < out->size(); ++i) {
            GfVec3f const &p = (*out)[i].pos;
            grid[CellKey(cellIndex(p[0]), cellIndex(p[1]), cellIndex(p[2]))].push_back(uint32_t(i));
        }
    };
    auto tryInsert = [&](Root const &c, bool checkDistance) {
        Key const key{std::llround(c.pos[0] / minSp), std::llround(c.pos[1] / minSp),
                      std::llround(c.pos[2] / minSp)};
        if (dedup.count(key)) return;
        int64_t const ix = cellIndex(c.pos[0]), iy = cellIndex(c.pos[1]), iz = cellIndex(c.pos[2]);
        if (checkDistance) {
            float const r2 = r * r;
            for (int64_t dz = -1; dz <= 1; ++dz)
                for (int64_t dy = -1; dy <= 1; ++dy)
                    for (int64_t dx = -1; dx <= 1; ++dx) {
                        auto const it = grid.find(CellKey(ix + dx, iy + dy, iz + dz));
                        if (it == grid.end()) continue;
                        for (uint32_t q : it->second)
                            if (Dist2(c.pos, (*out)[q].pos) < r2) return;
                    }
        }
        grid[CellKey(ix, iy, iz)].push_back(uint32_t(out->size()));
        dedup.insert(key);
        out->push_back(c);
    };

    rebuild();
    for (uint32_t i : order) {
        if (out->size() >= target) break;
        tryInsert(pool[i], true);
    }
    st->firstPass = uint32_t(out->size());
    if (out->size() < target) {
        r *= 0.75f;
        rebuild();
        for (uint32_t i : order) {
            if (out->size() >= target) break;
            tryInsert(pool[i], true);
        }
    }
    uint32_t relax = 0, barren = 0;
    while (out->size() < target) {
        if (r <= 4.0f * minSp || relax >= 64) break;
        size_t const p0 = out->size();
        uint64_t const darts = std::clamp<uint64_t>((uint64_t(target) - p0) * 16ull, 4096ull, 250000ull);
        sample(darts, [&](Root const &c, bool valid) {
            ++st->relaxationDarts;
            if (valid) tryInsert(c, true);
            return out->size() < target;
        });
        if (out->size() >= target) break;
        float rn = 0.85f * r;
        if (out->size() == p0) {
            rn *= 0.85f;
            ++barren;
            if (barren > 2) break;
        } else {
            barren = 0;
        }
        r = std::max(minSp, rn);
        rebuild();
        ++relax;
    }
    st->relaxations = relax;
    st->barren = barren;
    if (out->size() < target && (barren < 3 || !out->empty())) {
        st->lastResort = true;
        sample(std::max<uint64_t>(200ull * target, 5000ull), [&](Root const &c, bool valid) {
            if (valid) tryInsert(c, false);
            return out->size() < target;
        });
    }
    st->finalRadius = r;
}

/// One root per eligible guide, in guide order, at the guide's rest root;
/// the surface triangle / barycentrics / normal / UV come from the closest
/// point on the bound surface.
void GuideScatter(Surface &s, Guides const &guides, std::vector<Root> *out)
{
    if (!s.tris.empty()) s.EnsureBvh();
    for (size_t g = 0; g < guides.roots.size(); ++g) {
        Root r;
        r.pos = guides.roots[g];
        r.uid = guides.ids[g];
        strand::ClosestHit const hit = s.tris.empty() ? strand::ClosestHit() : s.bvh.Closest(r.pos);
        if (hit.hit) {
            size_t const t = size_t(hit.triangle);
            r.tri = int(t);
            r.sampleBary = r.hashBary = hit.bary;
            r.hashTri = s.globalTri[t];
            r.normal = s.InterpNormal(t, hit.bary[0], hit.bary[1], hit.bary[2]);
            if (s.HasUv())
                r.uv = Blend2(s.cornerUv[3 * t], s.cornerUv[3 * t + 1], s.cornerUv[3 * t + 2],
                              hit.bary[0], hit.bary[1], hit.bary[2]);
            r.face = s.tris[t].face;
        } else {
            r.face = guides.skinPrim[g];
            r.uv = guides.skinUv[g];
        }
        out->push_back(r);
    }
}

/// Tangent frame at a root: N the root normal, T the root triangle's first
/// edge Gram-Schmidt'd against N (second edge when nearly parallel), the
/// same construction the random path uses.
void RootFrame(Surface const &s, Root const &r, bool flip, GfVec3f *T, GfVec3f *N, GfVec3f *B)
{
    GfVec3f const n = Unit(r.normal, GfVec3f(0.0f, 1.0f, 0.0f));
    GfVec3f t(0.0f);
    float tLen = 0.0f;
    if (r.tri >= 0) {
        size_t const ti = size_t(r.tri);
        GfVec3f e0 = s.P(ti, 1) - s.P(ti, 0);
        float const e0d = GfDot(e0, n);
        if (e0d * e0d > 0.81f * GfDot(e0, e0)) {
            e0 = s.P(ti, 2) - s.P(ti, 0);
            t = e0 - n * GfDot(e0, n);
        } else {
            t = e0 - n * e0d;
        }
        tLen = std::sqrt(GfDot(t, t));
    }
    if (tLen < 1e-9f) {
        t = std::fabs(n[0]) > 0.9f ? GfVec3f(0.0f, 1.0f, 0.0f) : GfVec3f(1.0f, 0.0f, 0.0f);
        t = Unit(t - n * GfDot(t, n), GfVec3f(1.0f, 0.0f, 0.0f));
    } else {
        t = t / tLen;
    }
    GfVec3f b = Unit(GfCross(n, t), GfVec3f(0.0f, 0.0f, 1.0f));
    if (flip) {
        t = -t;
        b = -b;
    }
    *T = t;
    *N = n;
    *B = b;
}

}  // namespace

// ---------------------------------------------------------------------------

float RootHash01(uint64_t index, int tri, int vertex, GfVec3f const &pos, GfVec3f const &bary)
{
    uint64_t h = 0x14650fb0739d0383ull;
    auto mix = [&h](uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
    mix(index);
    mix(uint64_t(int64_t(tri) + 1));
    mix(uint64_t(int64_t(vertex) + 1));
    for (int k = 0; k < 3; ++k) mix(uint64_t(FloatBits(pos[k])));
    for (int k = 0; k < 3; ++k) mix(uint64_t(FloatBits(bary[k])));
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    uint32_t const r = uint32_t((h >> 33) ^ h) & 0xffffffu;
    return float(r) * (1.0f / 16777216.0f);
}

bool ReadModeConfig(UsdGenParamView const *p, ModeConfig *out, std::string *error)
{
    ModeConfig c;
    auto fail = [error](std::string const &m) {
        if (error) *error = m;
        return false;
    };
    if (p) {
        TfToken const dist = p->GetToken(TfToken("distribution"), TfToken("random"));
        if (dist == TfToken("random")) c.distribution = Distribution::Random;
        else if (dist == TfToken("uvGrid")) c.distribution = Distribution::UvGrid;
        else if (dist == TfToken("vertex")) c.distribution = Distribution::Vertex;
        else if (dist == TfToken("even")) c.distribution = Distribution::Even;
        else if (dist == TfToken("perGuide")) c.distribution = Distribution::PerGuide;
        else return fail("unknown distribution '" + dist.GetString() + "'");
        TfToken const count = p->GetToken(TfToken("countMode"), TfToken("perFace"));
        if (count == TfToken("perFace")) c.countMode = CountMode::PerFace;
        else if (count == TfToken("surface")) c.countMode = CountMode::Surface;
        else return fail("unknown countMode '" + count.GetString() + "'");
        TfToken const prox = p->GetToken(TfToken("proximity:mode"), TfToken("none"));
        if (prox == TfToken("none")) c.proximity = ProximityMode::None;
        else if (prox == TfToken("distance")) c.proximity = ProximityMode::Distance;
        else if (prox == TfToken("coverage")) c.proximity = ProximityMode::Coverage;
        else return fail("unknown proximity:mode '" + prox.GetString() + "'");
        double const dist2 = p->GetDouble(TfToken("proximity:distance"), 0.01);
        double const cov = p->GetDouble(TfToken("proximity:coverage"), 1.0);
        double const density = p->GetDouble(TfToken("density"), 100.0);
        if (!std::isfinite(dist2)) return fail("proximity:distance must be finite");
        if (!std::isfinite(cov)) return fail("proximity:coverage must be finite");
        if (!std::isfinite(density) || density < 0.0) return fail("density must be finite and >= 0");
        c.proximityDistance = float(dist2);
        c.proximityCoverage = float(cov);
        c.density = float(density);
        c.flip = p->GetBool(TfToken("flip"), false);
        c.subdivisionLevel = p->GetInt(TfToken("subdivisionLevel"), 0);
        if (c.subdivisionLevel < 0 || c.subdivisionLevel > 6)
            return fail("subdivisionLevel must be in [0,6]");
    }
    // The reference modes always count over the whole surface.
    if (c.distribution != Distribution::Random) c.countMode = CountMode::Surface;
    *out = c;
    return true;
}

void SetStrandPlanes(UsdGenCurveBuffer *buf, VtIntArray index, VtIntArray seed)
{
    auto insert = [buf](TfToken const &name, VtIntArray values) {
        UsdGenPlane plane;
        plane.name = name;
        plane.interpolation = TfToken("uniform");
        plane.type = TfToken("int");
        plane.arity = 1;
        plane.i = std::move(values);
        auto &planes = buf->extraCurve;
        auto it = std::lower_bound(planes.begin(), planes.end(), name,
            [](UsdGenPlane const &a, TfToken const &b) { return a.name < b; });
        if (it != planes.end() && it->name == name) *it = std::move(plane);
        else planes.insert(it, std::move(plane));
    };
    insert(TfToken("strandIndex"), std::move(index));
    insert(TfToken("strandSeed"), std::move(seed));
}

bool CaptureModes(UsdGenCaptureContext const &ctx, ModeConfig const &config,
                  UsdGenSurfaceDesc const &surface, UsdGenCurveBuffer *buf,
                  UsdGenDiagnostics *diag, ModeStats *stats)
{
    ModeStats local;
    ModeStats &st = stats ? *stats : local;
    st = ModeStats();
    auto fail = [diag](std::string const &m) {
        if (diag) diag->Error("UsdGenScatter: " + m);
        return false;
    };

    Surface s;
    s.Build(surface);
    Guides guides;
    std::string error;
    if (!LoadGuides(ctx, &guides, &error)) return fail(error);
    if (config.distribution == Distribution::PerGuide && guides.roots.empty() && diag)
        diag->Warn("UsdGenScatter: distribution 'perGuide' has no usable guides; 0 roots");

    Proximity prox;
    SetupProximity(s, guides, config, &prox);
    st.proximity = prox.active;
    st.cutoff2 = prox.cutoff2;
    st.areaMesh = s.areaTotal;

    Picker pick;
    pick.s = &s;
    float aDom = 0.0f;
    if (prox.active && config.proximity == ProximityMode::Distance &&
        config.distribution != Distribution::Vertex)
        aDom = BuildDomain(s, prox, &pick);
    st.areaDomain = aDom;
    float aEff = s.areaTotal;
    if (!pick.domain.empty()) {
        std::mt19937 rng2(90210u);
        uint32_t hits = 0;
        for (int k = 0; k < 4096; ++k) {
            size_t const t = size_t(pick.domain[LowerBound(pick.cdfDom, strand::Mt01(rng2))]);
            float const a = strand::Mt01(rng2);
            float const b = strand::Mt01(rng2);
            GfVec3f const p = s.SamplePoint(t, a, b);
            if (!prox.active || prox.cutoff2 < 0.0f || prox.Within(p)) ++hits;
        }
        st.calibrationHits = hits;
        aEff = float(hits) / 4096.0f * aDom;
        if (aEff <= 1e-9f) aEff = aDom;
    }
    st.areaEffective = aEff;

    uint32_t target = 0;
    if (config.distribution == Distribution::PerGuide) {
        target = uint32_t(guides.roots.size());
    } else {
        float const density = std::max(0.0f, config.density);
        if (s.areaTotal > 1e-8f && density > 0.0f) {
            float const want = std::round(s.areaTotal * density);
            target = want >= float(kMaxTarget) ? kMaxTarget
                     : (want <= 1.0f ? 1u : uint32_t(want));
        }
    }
    st.target = target;

    std::mt19937 rng(uint32_t(ctx.seed));
    std::vector<Root> roots;
    roots.reserve(target);
    switch (config.distribution) {
    case Distribution::Random:
        DartScatter(s, pick, prox, target, rng, &roots);
        break;
    case Distribution::UvGrid:
        if (s.HasUv()) UvGridScatter(s, prox, target, &roots);
        else DartScatter(s, pick, prox, target, rng, &roots);
        break;
    case Distribution::Vertex:
        VertexScatter(s, prox, target, rng, &roots);
        break;
    case Distribution::Even: {
        float const spacing = std::max(1e-5f, std::sqrt(aEff / float(std::max(1u, target))));
        st.spacing = spacing;
        EvenScatter(s, pick, prox, target, spacing, rng, &roots, &st);
        break;
    }
    case Distribution::PerGuide:
        GuideScatter(s, guides, &roots);
        break;
    }
    st.placed = uint32_t(roots.size());

    // Density-map thinning (emit filter): never renumbers strands.
    std::vector<UsdGenMapDesc const *> const maps = strand::BoundMaps(
        ctx.desc, ctx.mapBindings, ctx.mapBindingCount, DensityMapRelationship());
    std::vector<uint32_t> survivors;
    survivors.reserve(roots.size());
    if (!maps.empty() && !s.tris.empty()) s.EnsureBvh();
    for (size_t i = 0; i < roots.size(); ++i) {
        Root const &r = roots[i];
        if (!maps.empty()) {
            float m = 1.0f;
            for (UsdGenMapDesc const *map : maps)
                m *= s.bvhBuilt ? strand::SampleMaskMap(*map, s.bvh, r.tri, r.sampleBary, 1.0f)
                                : 1.0f;
            bool const keep = m > 0.0f &&
                (m >= 0.999f || RootHash01(i, r.hashTri, r.vertex, r.pos, r.hashBary) <= m);
            if (!keep) continue;
        }
        survivors.push_back(uint32_t(i));
    }

    // Catmull-Clark surfaces with usdGen:subdivisionLevel: the roots above were
    // distributed over the cage (the reference generator's draws, thinning and
    // identities are unchanged); each now moves to the limit point under it,
    // with the limit normal, so hair sits on the smooth surface the renderer
    // draws. rootPrim/rootUV keep their cage meaning (face, texture st). Roots
    // on hole faces are dropped; roots off the surface stay where they are.
    if (config.subdivisionLevel > 0 && !survivors.empty()) {
        UsdGenLimitSurface limit;
        if (!limit.Build(surface, config.subdivisionLevel, &error)) return fail(error);
        std::vector<int> faceStart(surface.faceVertexCounts.size() + 1, 0);
        for (size_t f = 0; f < surface.faceVertexCounts.size(); ++f)
            faceStart[f + 1] = faceStart[f] + surface.faceVertexCounts[f];
        bool const leftHanded = surface.orientation == TfToken("leftHanded");
        std::vector<uint32_t> onLimit;
        onLimit.reserve(survivors.size());
        for (uint32_t i : survivors) {
            Root &r = roots[i];
            if (r.tri < 0) {
                onLimit.push_back(i);
                continue;
            }
            strand::FanTriangle const &t = s.tris[size_t(r.tri)];
            if (limit.IsHole(t.face)) continue;
            int const corners[3] = {t.corner[0] - faceStart[size_t(t.face)],
                                    t.corner[1] - faceStart[size_t(t.face)],
                                    t.corner[2] - faceStart[size_t(t.face)]};
            GfVec3f pos, du, dv;
            if (!limit.EvaluateCagePoint(t.face, corners, r.sampleBary, &pos, &du, &dv))
                return fail("limit patch evaluation failed");
            GfVec3f n = GfCross(du, dv);
            float const len = std::sqrt(GfDot(n, n));
            if (len > 1e-12f) {
                n /= len;
                r.normal = leftHanded ? -n : n;
            }
            r.pos = pos;
            onLimit.push_back(i);
        }
        survivors.swap(onLimit);
    }
    size_t const N = survivors.size();
    st.emitted = uint32_t(N);

    // Morton order over rest roots (cellScale 64, ties in generation order),
    // exactly the random path's order rule.
    std::vector<uint64_t> keys(N);
    std::vector<uint32_t> order(N);
    for (size_t k = 0; k < N; ++k) {
        GfVec3f const &p = roots[survivors[k]].pos;
        keys[k] = UsdGenMortonKey3(p[0], p[1], p[2], 64.0f);
        order[k] = uint32_t(k);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&keys](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });

    uint64_t const seedFold = UsdGenHash64(uint64_t(uint32_t(ctx.seed)), kSaltScatterModes);
    buf->px = VtFloatArray(N);
    buf->py = VtFloatArray(N);
    buf->pz = VtFloatArray(N);
    buf->hairT = VtFloatArray(N, 0.0f);
    buf->curveId = VtArray<uint64_t>(N);
    buf->rootPrim = VtIntArray(N);
    buf->rootUV = VtVec2fArray(N);
    buf->rootT = VtVec3fArray(N);
    buf->rootN = VtVec3fArray(N);
    buf->rootB = VtVec3fArray(N);
    VtIntArray strandIndex(N), strandSeed(N);
    for (size_t k = 0; k < N; ++k) {
        uint32_t const gen = survivors[order[k]];
        Root const &r = roots[gen];
        buf->px[k] = r.pos[0];
        buf->py[k] = r.pos[1];
        buf->pz[k] = r.pos[2];
        buf->curveId[k] = r.uid ? r.uid : UsdGenHash64(seedFold ^ uint64_t(gen), kSaltScatterModes);
        buf->rootPrim[k] = r.face;
        buf->rootUV[k] = r.uv;
        RootFrame(s, r, config.flip, &buf->rootT[k], &buf->rootN[k], &buf->rootB[k]);
        strandIndex[k] = int(gen);
        // Reference rootHash: splitmix64(uid) & 0x7fffffff for roots that
        // carry an id (guide roots), else the strand index.
        strandSeed[k] = r.uid ? int(uint32_t(strand::SplitMix64(r.uid)) & 0x7fffffffu) : int(gen);
    }
    buf->totalCurves = uint32_t(N);
    buf->totalCvs = uint32_t(N);
    SetStrandPlanes(buf, std::move(strandIndex), std::move(strandSeed));
    return true;
}

// ---------------------------------------------------------------------------

struct RandomPathFilter::Impl
{
    Surface surface;
    Proximity prox;
    std::vector<UsdGenMapDesc const *> maps;
    bool active = false;
};

RandomPathFilter::RandomPathFilter() : _impl(new Impl) {}
RandomPathFilter::~RandomPathFilter() = default;

bool RandomPathFilter::Init(UsdGenCaptureContext const &ctx, ModeConfig const &config,
                            UsdGenSurfaceDesc const &surface, UsdGenDiagnostics *diag)
{
    Impl &im = *_impl;
    im = Impl();
    im.maps = strand::BoundMaps(ctx.desc, ctx.mapBindings, ctx.mapBindingCount,
                                DensityMapRelationship());
    Guides guides;
    if (config.proximity != ProximityMode::None) {
        std::string error;
        if (!LoadGuides(ctx, &guides, &error)) {
            if (diag) diag->Error("UsdGenScatter: " + error);
            return false;
        }
    }
    bool const wantProximity = config.proximity != ProximityMode::None && !guides.roots.empty();
    if (im.maps.empty() && !wantProximity) return true;
    im.surface.Build(surface);
    SetupProximity(im.surface, guides, config, &im.prox);
    if (!im.maps.empty() && !im.surface.tris.empty()) im.surface.EnsureBvh();
    im.active = !im.maps.empty() || im.prox.active;
    return true;
}

bool RandomPathFilter::Active() const { return _impl->active; }

bool RandomPathFilter::Keep(GfVec3f const &pos, uint64_t curveId) const
{
    Impl const &im = *_impl;
    if (!im.active) return true;
    if (!im.prox.Accept(pos)) return false;
    if (im.maps.empty() || !im.surface.bvhBuilt) return true;
    strand::ClosestHit const hit = im.surface.bvh.Closest(pos);
    if (!hit.hit) return true;
    float m = 1.0f;
    for (UsdGenMapDesc const *map : im.maps)
        m *= strand::SampleMaskMap(*map, im.surface.bvh, hit.triangle, hit.bary, 1.0f);
    return m > 0.0f && (m >= 0.999f || UsdGenHash01(curveId, kSaltScatterThin) <= m);
}

}  // namespace scatter
}  // namespace usdGen
