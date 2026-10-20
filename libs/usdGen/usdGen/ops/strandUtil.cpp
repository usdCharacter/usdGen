// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — strand math shared by the whole-strand operators. See strandUtil.h.
#include "usdGen/ops/strandUtil.h"

#include "usdGen/imagePayload.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace strand {

float ArcLengths(float const *p, size_t n, std::vector<float> *out)
{
    out->assign(n, 0.0f);
    float s = 0.0f;
    for (size_t i = 1; i < n; ++i) {
        float const dx = p[3 * i] - p[3 * (i - 1)];
        float const dy = p[3 * i + 1] - p[3 * (i - 1) + 1];
        float const dz = p[3 * i + 2] - p[3 * (i - 1) + 2];
        s += std::sqrt(dx * dx + dy * dy + dz * dz);
        (*out)[i] = s;
    }
    return s;
}

GfVec3f SampleArc(float const *p, size_t n, std::vector<float> const &cum, float s)
{
    if (n == 0) return GfVec3f(0.0f);
    if (n == 1 || s <= 0.0f) return Load(p, 0);
    float const L = cum[n - 1];
    if (!(L > 1e-12f)) return Load(p, 0);
    if (s >= L) {
        // Extrapolate along the last non-degenerate segment.
        size_t k = n - 1;
        while (k > 0 && cum[k] - cum[k - 1] <= 1e-12f) --k;
        if (k == 0) return Load(p, n - 1);
        GfVec3f const a = Load(p, k - 1), b = Load(p, k);
        GfVec3f const dir = (b - a) / (cum[k] - cum[k - 1]);
        return Load(p, n - 1) + dir * (s - L);
    }
    // First segment whose end reaches s.
    size_t const hi = static_cast<size_t>(
        std::upper_bound(cum.begin(), cum.begin() + static_cast<std::ptrdiff_t>(n), s) - cum.begin());
    size_t const k = std::min(std::max<size_t>(hi, 1), n - 1);
    float const seg = cum[k] - cum[k - 1];
    float const w = seg > 1e-12f ? (s - cum[k - 1]) / seg : 0.0f;
    GfVec3f const a = Load(p, k - 1), b = Load(p, k);
    return a + (b - a) * w;
}

void ResampleArc(float const *p, size_t n, std::vector<float> const &cum,
                 float s0, float s1, size_t m, float *dst)
{
    for (size_t j = 0; j < m; ++j) {
        float const t = m > 1 ? float(j) / float(m - 1) : 0.0f;
        Store(dst, j, SampleArc(p, n, cum, s0 + (s1 - s0) * t));
    }
}

void TransportFrames(std::vector<GfVec3f> const &T, std::vector<GfVec3f> *N,
                     std::vector<GfVec3f> *B)
{
    size_t const n = T.size();
    N->assign(n, GfVec3f(1.0f, 0.0f, 0.0f));
    B->assign(n, GfVec3f(0.0f, 0.0f, 1.0f));
    if (n == 0) return;
    GfVec3f const up = std::fabs(T[0][1]) >= 0.95f ? GfVec3f(1.0f, 0.0f, 0.0f)
                                                   : GfVec3f(0.0f, 1.0f, 0.0f);
    GfVec3f prev = Normalize(GfCross(T[0], up), GfVec3f(1.0f, 0.0f, 0.0f));
    for (size_t i = 0; i < n; ++i) {
        GfVec3f const t = T[i];
        GfVec3f const proj = prev - t * GfDot(prev, t);
        GfVec3f const nrm = Normalize(proj, prev);
        (*N)[i] = nrm;
        (*B)[i] = GfCross(t, nrm);
        prev = nrm;
    }
}

void PolylineTangents(float const *p, size_t n, std::vector<GfVec3f> *T)
{
    T->assign(n, GfVec3f(0.0f, 1.0f, 0.0f));
    GfVec3f last(0.0f, 1.0f, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        size_t const a = i > 0 ? i - 1 : 0;
        size_t const b = i + 1 < n ? i + 1 : n - 1;
        GfVec3f const d = Load(p, b) - Load(p, a);
        float const l = d.GetLength();
        if (l > 1e-12f) last = d / l;
        (*T)[i] = last;
    }
}

void ResnapRoot(float *p, size_t n, GfVec3f const &delta)
{
    float const dl = delta.GetLength();
    if (n == 0 || !(dl > 1e-9f)) return;
    std::vector<float> cum;
    float const L = ArcLengths(p, n, &cum);
    float const F = std::min(4.0f * dl, 0.5f * L);
    Store(p, 0, Load(p, 0) + delta);
    if (!(F > 0.0f)) return;
    for (size_t k = 1; k < n; ++k) {
        if (cum[k] >= F) break;
        float const w = 1.0f - SmoothStep01(cum[k] / F);
        Store(p, k, Load(p, k) + delta * w);
    }
}

// ---------------------------------------------------------------------------

void FanTriangulate(VtIntArray const &counts, VtIntArray const &indices,
                    std::vector<FanTriangle> *out, VtIntArray const *faces)
{
    out->clear();
    std::vector<int> faceStart(counts.size() + 1, 0);
    for (size_t f = 0; f < counts.size(); ++f)
        faceStart[f + 1] = faceStart[f] + std::max(0, counts[f]);
    auto emit = [&](int f) {
        if (f < 0 || size_t(f) >= counts.size()) return;
        int const k = counts[f];
        int const c0 = faceStart[size_t(f)];
        if (k < 3 || size_t(c0 + k) > indices.size()) return;
        for (int j = 1; j + 1 < k; ++j) {
            FanTriangle t;
            t.face = f;
            t.corner[0] = c0; t.corner[1] = c0 + j; t.corner[2] = c0 + j + 1;
            t.v[0] = indices[size_t(t.corner[0])];
            t.v[1] = indices[size_t(t.corner[1])];
            t.v[2] = indices[size_t(t.corner[2])];
            out->push_back(t);
        }
    };
    if (faces) {
        for (int f : *faces) emit(f);
    } else {
        for (size_t f = 0; f < counts.size(); ++f) emit(int(f));
    }
}

namespace {

// Ericson 5.1.5, double precision. Returns barycentrics (u,v,w) of the
// closest point on triangle abc to p.
GfVec3d ClosestBary(GfVec3d const &p, GfVec3d const &a, GfVec3d const &b, GfVec3d const &c)
{
    GfVec3d const ab = b - a, ac = c - a, ap = p - a;
    double const d1 = GfDot(ab, ap), d2 = GfDot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return GfVec3d(1, 0, 0);
    GfVec3d const bp = p - b;
    double const d3 = GfDot(ab, bp), d4 = GfDot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return GfVec3d(0, 1, 0);
    double const vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        double const v = d1 / (d1 - d3);
        return GfVec3d(1.0 - v, v, 0.0);
    }
    GfVec3d const cp = p - c;
    double const d5 = GfDot(ab, cp), d6 = GfDot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return GfVec3d(0, 0, 1);
    double const vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        double const w = d2 / (d2 - d6);
        return GfVec3d(1.0 - w, 0.0, w);
    }
    double const va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        double const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return GfVec3d(0.0, 1.0 - w, w);
    }
    double const denom = va + vb + vc;
    if (!(std::fabs(denom) > 1e-300)) return GfVec3d(1, 0, 0);
    double const v = vb / denom, w = vc / denom;
    return GfVec3d(1.0 - v - w, v, w);
}

double BoxDistanceSq(float const lo[3], float const hi[3], GfVec3f const &p)
{
    double d = 0.0;
    for (int k = 0; k < 3; ++k) {
        double const v = p[k];
        if (v < lo[k]) d += (lo[k] - v) * double(lo[k] - v);
        else if (v > hi[k]) d += (v - hi[k]) * double(v - hi[k]);
    }
    return d;
}

}  // namespace

void TriangleBvh::Build(VtVec3fArray const &points, std::vector<FanTriangle> const &triangles,
                        std::vector<GfVec3f> cornerNormals, std::vector<GfVec2f> cornerUvs)
{
    _points = points;
    _tris.clear();
    _tris.reserve(triangles.size());
    for (FanTriangle const &t : triangles) {
        bool ok = true;
        for (int k = 0; k < 3; ++k)
            ok = ok && t.v[k] >= 0 && size_t(t.v[k]) < points.size();
        if (ok) _tris.push_back(t);
    }
    // Corner normals: authored when complete, else area-weighted vertex normals.
    if (cornerNormals.size() == 3 * triangles.size() && _tris.size() == triangles.size()) {
        _normals = std::move(cornerNormals);
        for (GfVec3f &nrm : _normals) nrm = Normalize(nrm, GfVec3f(0.0f, 1.0f, 0.0f));
    } else {
        std::vector<GfVec3d> vn(points.size(), GfVec3d(0.0));
        for (FanTriangle const &t : _tris) {
            GfVec3d const a(points[size_t(t.v[0])]), b(points[size_t(t.v[1])]), c(points[size_t(t.v[2])]);
            GfVec3d const fn = GfCross(b - a, c - a);  // length = 2 * area
            for (int k = 0; k < 3; ++k) vn[size_t(t.v[k])] += fn;
        }
        _normals.resize(3 * _tris.size());
        for (size_t i = 0; i < _tris.size(); ++i)
            for (int k = 0; k < 3; ++k) {
                GfVec3d const v = vn[size_t(_tris[i].v[k])];
                double const l = v.GetLength();
                _normals[3 * i + size_t(k)] = l > 1e-300 ? GfVec3f(v / l) : FaceNormal(int(i));
            }
    }
    if (cornerUvs.size() == 3 * triangles.size() && _tris.size() == triangles.size())
        _uvs = std::move(cornerUvs);
    else
        _uvs.clear();

    _nodes.clear();
    _order.resize(_tris.size());
    std::iota(_order.begin(), _order.end(), 0);
    if (_tris.empty()) return;
    std::vector<GfVec3f> centroids(_tris.size());
    for (size_t i = 0; i < _tris.size(); ++i)
        centroids[i] = (points[size_t(_tris[i].v[0])] + points[size_t(_tris[i].v[1])] +
                        points[size_t(_tris[i].v[2])]) / 3.0f;
    _nodes.reserve(2 * _tris.size());
    BuildRange(0, int(_tris.size()), centroids);
}

int TriangleBvh::BuildRange(int first, int count, std::vector<GfVec3f> const &centroids)
{
    int const index = int(_nodes.size());
    _nodes.push_back(Node());
    Node node;
    for (int k = 0; k < 3; ++k) { node.lo[k] = 3.0e38f; node.hi[k] = -3.0e38f; }
    float clo[3] = {3.0e38f, 3.0e38f, 3.0e38f}, chi[3] = {-3.0e38f, -3.0e38f, -3.0e38f};
    for (int i = first; i < first + count; ++i) {
        FanTriangle const &t = _tris[size_t(_order[size_t(i)])];
        for (int c = 0; c < 3; ++c) {
            GfVec3f const &q = _points[size_t(t.v[c])];
            for (int k = 0; k < 3; ++k) {
                node.lo[k] = std::min(node.lo[k], q[k]);
                node.hi[k] = std::max(node.hi[k], q[k]);
            }
        }
        GfVec3f const &cc = centroids[size_t(_order[size_t(i)])];
        for (int k = 0; k < 3; ++k) { clo[k] = std::min(clo[k], cc[k]); chi[k] = std::max(chi[k], cc[k]); }
    }
    if (count <= 4) {
        node.first = first;
        node.count = count;
        _nodes[size_t(index)] = node;
        return index;
    }
    int axis = 0;
    for (int k = 1; k < 3; ++k)
        if (chi[k] - clo[k] > chi[axis] - clo[axis]) axis = k;
    int const mid = first + count / 2;
    std::nth_element(_order.begin() + first, _order.begin() + mid, _order.begin() + first + count,
        [&](int a, int b) {
            float const ca = centroids[size_t(a)][axis], cb = centroids[size_t(b)][axis];
            return ca < cb || (ca == cb && a < b);
        });
    node.left = BuildRange(first, mid - first, centroids);
    node.right = BuildRange(mid, first + count - mid, centroids);
    _nodes[size_t(index)] = node;
    return index;
}

ClosestHit TriangleBvh::Closest(GfVec3f const &p, float maxDistanceSq) const
{
    ClosestHit best;
    if (_nodes.empty()) return best;
    double bestD = double(maxDistanceSq);
    int bestTri = std::numeric_limits<int>::max();
    GfVec3d const pd(p);
    int stack[128];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        Node const &node = _nodes[size_t(stack[--top])];
        if (BoxDistanceSq(node.lo, node.hi, p) > bestD) continue;
        if (node.left < 0) {
            for (int i = node.first; i < node.first + node.count; ++i) {
                int const ti = _order[size_t(i)];
                FanTriangle const &t = _tris[size_t(ti)];
                GfVec3d const a(_points[size_t(t.v[0])]), b(_points[size_t(t.v[1])]),
                    c(_points[size_t(t.v[2])]);
                GfVec3d const bc = ClosestBary(pd, a, b, c);
                GfVec3d const q = a * bc[0] + b * bc[1] + c * bc[2];
                double const d = (q - pd).GetLengthSq();
                if (d < bestD || (d == bestD && ti < bestTri)) {
                    bestD = d;
                    bestTri = ti;
                    best.hit = true;
                    best.triangle = ti;
                    best.bary = GfVec3f(bc);
                    best.point = GfVec3f(q);
                    best.distanceSq = float(d);
                }
            }
            continue;
        }
        // Visit the nearer child first.
        Node const &l = _nodes[size_t(node.left)];
        Node const &r = _nodes[size_t(node.right)];
        double const dl = BoxDistanceSq(l.lo, l.hi, p), dr = BoxDistanceSq(r.lo, r.hi, p);
        if (top + 2 > 128) continue;  // depth guard (median split keeps depth ~log2)
        if (dl <= dr) { stack[top++] = node.right; stack[top++] = node.left; }
        else { stack[top++] = node.left; stack[top++] = node.right; }
    }
    return best;
}

GfVec3f TriangleBvh::FaceNormal(int triangle) const
{
    if (triangle < 0 || size_t(triangle) >= _tris.size()) return GfVec3f(0.0f, 1.0f, 0.0f);
    FanTriangle const &t = _tris[size_t(triangle)];
    GfVec3f const a = _points[size_t(t.v[0])], b = _points[size_t(t.v[1])], c = _points[size_t(t.v[2])];
    return Normalize(GfCross(b - a, c - a), GfVec3f(0.0f, 1.0f, 0.0f), 1e-20f);
}

GfVec3f TriangleBvh::Normal(int triangle, GfVec3f const &bary) const
{
    if (triangle < 0 || size_t(triangle) >= _tris.size()) return GfVec3f(0.0f, 1.0f, 0.0f);
    size_t const b = 3 * size_t(triangle);
    GfVec3f const n = _normals[b] * bary[0] + _normals[b + 1] * bary[1] + _normals[b + 2] * bary[2];
    return Normalize(n, FaceNormal(triangle));
}

GfVec2f TriangleBvh::Uv(int triangle, GfVec3f const &bary) const
{
    if (_uvs.empty() || triangle < 0 || size_t(triangle) >= _tris.size()) return GfVec2f(0.0f);
    size_t const b = 3 * size_t(triangle);
    return _uvs[b] * bary[0] + _uvs[b + 1] * bary[1] + _uvs[b + 2] * bary[2];
}

GfVec3f TriangleBvh::Position(int triangle, GfVec3f const &bary) const
{
    if (triangle < 0 || size_t(triangle) >= _tris.size()) return GfVec3f(0.0f);
    FanTriangle const &t = _tris[size_t(triangle)];
    return _points[size_t(t.v[0])] * bary[0] + _points[size_t(t.v[1])] * bary[1] +
           _points[size_t(t.v[2])] * bary[2];
}

void BuildSurfaceBvh(UsdGenSurfaceDesc const &surface, bool rest, TriangleBvh *bvh)
{
    VtVec3fArray const &pts = (rest && !surface.restPoints.empty()) ? surface.restPoints
                                                                    : surface.points;
    std::vector<FanTriangle> tris;
    bool const restricted = UsdGenSurfaceRestricted(surface);
    FanTriangulate(surface.faceVertexCounts, surface.faceVertexIndices, &tris,
                   restricted ? &surface.subsetFaces : nullptr);
    std::vector<GfVec3f> normals;
    if (rest && !surface.restNormals.empty()) {
        VtVec3fArray const &rn = surface.restNormals;
        bool const vertex = surface.restNormalDomain == UsdGenSurfaceNormalDomain::Vertex &&
                            rn.size() == pts.size();
        bool const faceVarying = surface.restNormalDomain == UsdGenSurfaceNormalDomain::FaceVarying &&
                                 rn.size() == surface.faceVertexIndices.size();
        if (vertex || faceVarying) {
            normals.resize(3 * tris.size());
            for (size_t i = 0; i < tris.size(); ++i)
                for (int k = 0; k < 3; ++k)
                    normals[3 * i + size_t(k)] =
                        rn[size_t(vertex ? tris[i].v[k] : tris[i].corner[k])];
        }
    }
    std::vector<GfVec2f> uvs;
    VtVec2fArray const &uv = surface.uv;
    if (!uv.empty()) {
        bool const vertex = uv.size() == pts.size();
        bool const faceVarying = uv.size() == surface.faceVertexIndices.size();
        if (vertex || faceVarying) {
            uvs.resize(3 * tris.size());
            for (size_t i = 0; i < tris.size(); ++i)
                for (int k = 0; k < 3; ++k) {
                    int const idx = faceVarying && !vertex ? tris[i].corner[k] : tris[i].v[k];
                    uvs[3 * i + size_t(k)] = (idx >= 0 && size_t(idx) < uv.size()) ? uv[size_t(idx)]
                                                                                 : GfVec2f(0.0f);
                }
        }
    }
    bvh->Build(pts, tris, std::move(normals), std::move(uvs));
}

float SampleMaskMap(UsdGenMapDesc const &map, TriangleBvh const &bvh, int triangle,
                    GfVec3f const &bary, float fallback)
{
    if (triangle < 0 || size_t(triangle) >= bvh.Triangles().size()) return fallback;
    FanTriangle const &t = bvh.Triangles()[size_t(triangle)];
    if (map.type == TfToken("UsdGenPaintMap")) {
        VtFloatArray const &v = map.paintValues;
        for (int k = 0; k < 3; ++k)
            if (t.corner[k] < 0 || size_t(t.corner[k]) >= v.size()) return fallback;
        float const value = v[size_t(t.corner[0])] * bary[0] + v[size_t(t.corner[1])] * bary[1] +
                            v[size_t(t.corner[2])] * bary[2];
        return Clamp01(std::isfinite(value) ? value : 0.0f);
    }
    if (map.type == TfToken("UsdGenImageMap") && map.imagePayload &&
        map.imagePayload->IsValid()) {
        GfVec2f const uv = bvh.Uv(triangle, bary);
        UsdGenImageSampleOptions options;
        options.filter = UsdGenImageFilter::Nearest;
        options.wrap = UsdGenImageWrap::Clamp;
        UsdGenImagePayload const &image = *map.imagePayload;
        float value;
        if (image.Channels() >= 3) {
            options.channel = UsdGenImageChannel::R;
            float const r = UsdGenImageSampler::Sample(image, uv[0], uv[1], options);
            options.channel = UsdGenImageChannel::G;
            float const g = UsdGenImageSampler::Sample(image, uv[0], uv[1], options);
            options.channel = UsdGenImageChannel::B;
            float const b = UsdGenImageSampler::Sample(image, uv[0], uv[1], options);
            value = 0.299f * r + 0.587f * g + 0.114f * b;
        } else {
            options.channel = UsdGenImageChannel::R;
            value = UsdGenImageSampler::Sample(image, uv[0], uv[1], options);
        }
        return Clamp01(std::isfinite(value) ? value : 0.0f);
    }
    return fallback;
}

std::vector<UsdGenMapDesc const *> BoundMaps(UsdGenGraphDesc const *desc,
                                             UsdGenMapBindingDesc const *bindings,
                                             uint32_t bindingCount,
                                             TfToken const &relationship)
{
    std::vector<UsdGenMapDesc const *> out;
    if (!desc || !bindings) return out;
    for (uint32_t i = 0; i < bindingCount; ++i) {
        if (bindings[i].relationship != relationship) continue;
        for (UsdGenMapDesc const &m : desc->maps)
            if (m.path == bindings[i].map) { out.push_back(&m); break; }
    }
    return out;
}

}  // namespace strand
}  // namespace usdGen
