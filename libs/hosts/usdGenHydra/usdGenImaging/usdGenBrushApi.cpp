// usdGen imaging — the attribute-brush C ABI (see usdGenBrushApi.h).
#include "usdGenImaging/usdGenBrushApi.h"

#include "usdGenImaging/attributePreviewSceneIndex.h"
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenImaging/testHook.h"

#include "usdGen/maps/attributeMap.h"

#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/types.h"
#include "pxr/usd/sdf/path.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

using usdGen::UsdGenAttributeMap;
using usdGen::UsdGenAttributeMapSpec;
using usdGen::UsdGenBrushDab;
using usdGen::UsdGenBrushFalloff;
using usdGen::UsdGenBrushMode;

using Vec3 = std::array<double, 3>;

Vec3 Sub(Vec3 const &a, Vec3 const &b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double Dot(Vec3 const &a, Vec3 const &b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 Cross(Vec3 const &a, Vec3 const &b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double DistSq(Vec3 const &a, Vec3 const &b)
{
    Vec3 const d = Sub(a, b);
    return Dot(d, d);
}
bool Finite3(Vec3 const &a)
{
    return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]);
}

// Möller–Trumbore, both windings: true + (dist, w1, w2) on a hit in front.
bool RayTriangle(Vec3 const &o, Vec3 const &d, Vec3 const &v0, Vec3 const &v1,
                 Vec3 const &v2, double *dist, double *w1, double *w2)
{
    Vec3 const e1 = Sub(v1, v0);
    Vec3 const e2 = Sub(v2, v0);
    Vec3 const p = Cross(d, e2);
    double const det = Dot(e1, p);
    if (std::fabs(det) < 1e-20) return false;
    double const inv = 1.0 / det;
    Vec3 const t = Sub(o, v0);
    double const b1 = Dot(t, p) * inv;
    if (b1 < 0.0 || b1 > 1.0) return false;
    Vec3 const q = Cross(t, e1);
    double const b2 = Dot(d, q) * inv;
    if (b2 < 0.0 || b1 + b2 > 1.0) return false;
    double const s = Dot(e2, q) * inv;
    if (!(s > 0.0)) return false;
    *dist = s;
    *w1 = b1;
    *w2 = b2;
    return true;
}

// Closest point on triangle abc (Ericson, Real-Time Collision Detection
// 5.1.5): squared distance and the barycentric weights of b and c.
double ClosestOnTriangle(Vec3 const &p, Vec3 const &a, Vec3 const &b, Vec3 const &c,
                         double *wb, double *wc)
{
    Vec3 const ab = Sub(b, a), ac = Sub(c, a), ap = Sub(p, a);
    double const d1 = Dot(ab, ap), d2 = Dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { *wb = 0; *wc = 0; return DistSq(p, a); }
    Vec3 const bp = Sub(p, b);
    double const d3 = Dot(ab, bp), d4 = Dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { *wb = 1; *wc = 0; return DistSq(p, b); }
    double const vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        double const den = d1 - d3;
        double const v = den != 0.0 ? d1 / den : 0.0;
        *wb = v; *wc = 0;
        Vec3 const q{a[0] + ab[0] * v, a[1] + ab[1] * v, a[2] + ab[2] * v};
        return DistSq(p, q);
    }
    Vec3 const cp = Sub(p, c);
    double const d5 = Dot(ab, cp), d6 = Dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { *wb = 0; *wc = 1; return DistSq(p, c); }
    double const vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        double const den = d2 - d6;
        double const w = den != 0.0 ? d2 / den : 0.0;
        *wb = 0; *wc = w;
        Vec3 const q{a[0] + ac[0] * w, a[1] + ac[1] * w, a[2] + ac[2] * w};
        return DistSq(p, q);
    }
    double const va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        double const den = (d4 - d3) + (d5 - d6);
        double const w = den != 0.0 ? (d4 - d3) / den : 0.0;
        *wb = 1.0 - w; *wc = w;
        Vec3 const q{b[0] + (c[0] - b[0]) * w, b[1] + (c[1] - b[1]) * w,
                     b[2] + (c[2] - b[2]) * w};
        return DistSq(p, q);
    }
    double const den = va + vb + vc;
    if (den == 0.0) {
        double const da = DistSq(p, a), db = DistSq(p, b), dc = DistSq(p, c);
        if (da <= db && da <= dc) { *wb = 0; *wc = 0; return da; }
        if (db <= dc) { *wb = 1; *wc = 0; return db; }
        *wb = 0; *wc = 1; return dc;
    }
    double const v = vb / den, w = vc / den;
    *wb = v; *wc = w;
    Vec3 const q{a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w,
                 a[2] + ab[2] * v + ac[2] * w};
    return DistSq(p, q);
}

float Clamp01f(float x) { return std::min(1.0f, std::max(0.0f, x)); }

// ---------------------------------------------------------------------------
// Mesh: snapshot + uniform-grid face index + vertex adjacency
// ---------------------------------------------------------------------------

struct MeshData {
    std::vector<Vec3> points;
    std::vector<int> faces;       // 4 per face
    std::vector<float> edges;     // longest edge per face
    std::vector<float> meanEdges; // mean edge per face
    std::vector<Vec3> bmin, bmax; // per face
    // Face mask (a GeomSubset's faces): empty = every face paintable.
    std::vector<uint8_t> allowed;
    // Grid.
    bool hasGrid = false;
    int cells[3] = {1, 1, 1};
    Vec3 origin{0, 0, 0};
    double cellSize = 1.0;
    std::vector<uint32_t> cellStart;  // CSR
    std::vector<int> cellFaces;
    // Adjacency: vertex -> face-corners (face*4+i) and neighbour vertices.
    std::vector<uint32_t> cornerStart;
    std::vector<int> cornerList;
    std::vector<uint32_t> nbrStart;
    std::vector<int> nbrList;
    // Query dedupe stamps (a handle is not re-entrant across threads).
    mutable std::vector<uint32_t> stamp;
    mutable uint32_t stampId = 0;

    int NumFaces() const { return int(faces.size() / 4); }

    bool Allowed(int f) const { return allowed.empty() || allowed[size_t(f)] != 0; }

    uint32_t NextStamp() const
    {
        if (++stampId == 0) {
            std::fill(stamp.begin(), stamp.end(), 0u);
            stampId = 1;
        }
        return stampId;
    }

    int CellIndex(double value, int axis) const
    {
        double const at = std::floor((value - origin[axis]) / cellSize);
        if (!(at >= 0.0)) return 0;
        return std::min(cells[axis] - 1, int(at));
    }
    size_t Flat(int x, int y, int z) const
    {
        return (size_t(z) * size_t(cells[1]) + size_t(y)) * size_t(cells[0]) + size_t(x);
    }

    void Build()
    {
        int const F = NumFaces();
        edges.assign(size_t(F), 0.0f);
        meanEdges.assign(size_t(F), 0.0f);
        bmin.assign(size_t(F), Vec3{0, 0, 0});
        bmax.assign(size_t(F), Vec3{0, 0, 0});
        Vec3 lo{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()};
        Vec3 hi{-lo[0], -lo[1], -lo[2]};
        for (int f = 0; f < F; ++f) {
            Vec3 fl = points[size_t(faces[size_t(f) * 4])];
            Vec3 fh = fl;
            double longest = 0.0, sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                Vec3 const &a = points[size_t(faces[size_t(f) * 4 + k])];
                Vec3 const &b = points[size_t(faces[size_t(f) * 4 + (k + 1) % 4])];
                double const len = std::sqrt(DistSq(a, b));
                longest = std::max(longest, len);
                sum += len;
                for (int ax = 0; ax < 3; ++ax) {
                    fl[ax] = std::min(fl[ax], a[ax]);
                    fh[ax] = std::max(fh[ax], a[ax]);
                }
            }
            edges[size_t(f)] = float(longest);
            meanEdges[size_t(f)] = float(sum / 4.0);
            bmin[size_t(f)] = fl;
            bmax[size_t(f)] = fh;
            for (int ax = 0; ax < 3; ++ax) {
                lo[ax] = std::min(lo[ax], fl[ax]);
                hi[ax] = std::max(hi[ax], fh[ax]);
            }
        }
        stamp.assign(size_t(F), 0u);
        BuildAdjacency();
        BuildGrid(lo, hi);
    }

    // Over paintable faces only: a masked mesh smooths like the subset's own
    // submesh, so paint outside the mask never bleeds in at its border.
    void BuildAdjacency()
    {
        size_t const P = points.size();
        int const F = NumFaces();
        std::vector<uint32_t> count(P + 1, 0);
        for (int f = 0; f < F; ++f)
            if (Allowed(f))
                for (int i = 0; i < 4; ++i) ++count[size_t(faces[size_t(f) * 4 + size_t(i)]) + 1];
        cornerStart.assign(P + 1, 0);
        for (size_t p = 0; p < P; ++p) cornerStart[p + 1] = cornerStart[p] + count[p + 1];
        cornerList.assign(cornerStart[P], 0);
        std::vector<uint32_t> fill(cornerStart.begin(), cornerStart.end() - 1);
        for (int f = 0; f < F; ++f) {
            if (!Allowed(f)) continue;
            for (int i = 0; i < 4; ++i) {
                int const v = faces[size_t(f) * 4 + size_t(i)];
                cornerList[fill[size_t(v)]++] = f * 4 + i;
            }
        }
        // Edge neighbours: corner i's previous and next vertex on each face.
        std::vector<std::vector<int>> nbrs(P);
        for (int f = 0; f < F; ++f) {
            if (!Allowed(f)) continue;
            for (int i = 0; i < 4; ++i) {
                int const v = faces[size_t(f) * 4 + size_t(i)];
                nbrs[size_t(v)].push_back(faces[size_t(f) * 4 + size_t((i + 1) % 4)]);
                nbrs[size_t(v)].push_back(faces[size_t(f) * 4 + size_t((i + 3) % 4)]);
            }
        }
        nbrStart.assign(P + 1, 0);
        nbrList.clear();
        for (size_t p = 0; p < P; ++p) {
            auto &n = nbrs[p];
            std::sort(n.begin(), n.end());
            n.erase(std::unique(n.begin(), n.end()), n.end());
            n.erase(std::remove(n.begin(), n.end(), int(p)), n.end());
            nbrList.insert(nbrList.end(), n.begin(), n.end());
            nbrStart[p + 1] = uint32_t(nbrList.size());
        }
    }

    void BuildGrid(Vec3 const &lo, Vec3 const &hi)
    {
        int const F = NumFaces();
        hasGrid = false;
        if (F <= 0 || !Finite3(lo) || !Finite3(hi)) return;
        double extent = 0.0;
        for (int ax = 0; ax < 3; ++ax) extent = std::max(extent, hi[ax] - lo[ax]);
        double const pad = std::max(1e-6, extent * 1e-7);
        for (int ax = 0; ax < 3; ++ax) origin[ax] = lo[ax] - pad;
        double const span = extent + 2.0 * pad;
        // Cells along the longest axis ~ sqrt(F) (a surface fills ~N^2 of
        // N^3 cells, so that is ~1 face per occupied cell), capped so the
        // whole grid stays under 4M cells.
        int n = std::max(1, std::min(1024, int(std::ceil(std::sqrt(double(F))))));
        for (;;) {
            cellSize = span / n;
            size_t total = 1;
            for (int ax = 0; ax < 3; ++ax) {
                double const axSpan = (hi[ax] - lo[ax]) + 2.0 * pad;
                cells[ax] = std::max(1, int(std::ceil(axSpan / cellSize)));
                total *= size_t(cells[ax]);
            }
            if (total <= (size_t(1) << 22) || n == 1) break;
            n = std::max(1, n * 3 / 4);
        }
        size_t const nCells = size_t(cells[0]) * size_t(cells[1]) * size_t(cells[2]);
        size_t const budget = size_t(64) * size_t(F) + (size_t(1) << 20);
        double const eps = pad;
        // Count pass.
        std::vector<uint32_t> count(nCells + 1, 0);
        size_t entries = 0;
        auto range = [&](int f, int *l, int *h) {
            for (int ax = 0; ax < 3; ++ax) {
                l[ax] = CellIndex(bmin[size_t(f)][ax] - eps, ax);
                h[ax] = CellIndex(bmax[size_t(f)][ax] + eps, ax);
            }
        };
        for (int f = 0; f < F; ++f) {
            int l[3], h[3];
            range(f, l, h);
            size_t const c = size_t(h[0] - l[0] + 1) * size_t(h[1] - l[1] + 1) *
                             size_t(h[2] - l[2] + 1);
            entries += c;
            if (entries > budget) return;  // pathological: brute force
            for (int z = l[2]; z <= h[2]; ++z)
                for (int y = l[1]; y <= h[1]; ++y)
                    for (int x = l[0]; x <= h[0]; ++x) ++count[Flat(x, y, z) + 1];
        }
        cellStart.assign(nCells + 1, 0);
        for (size_t c = 0; c < nCells; ++c) cellStart[c + 1] = cellStart[c] + count[c + 1];
        cellFaces.assign(entries, 0);
        std::vector<uint32_t> fill(cellStart.begin(), cellStart.end() - 1);
        for (int f = 0; f < F; ++f) {
            int l[3], h[3];
            range(f, l, h);
            for (int z = l[2]; z <= h[2]; ++z)
                for (int y = l[1]; y <= h[1]; ++y)
                    for (int x = l[0]; x <= h[0]; ++x) cellFaces[fill[Flat(x, y, z)]++] = f;
        }
        hasGrid = true;
    }

    // (dist, u, v) of the ray against quad f; false on a miss.
    bool TestFace(int f, Vec3 const &o, Vec3 const &d, double *dist, float *u, float *v) const
    {
        Vec3 const &v0 = points[size_t(faces[size_t(f) * 4 + 0])];
        Vec3 const &v1 = points[size_t(faces[size_t(f) * 4 + 1])];
        Vec3 const &v2 = points[size_t(faces[size_t(f) * 4 + 2])];
        Vec3 const &v3 = points[size_t(faces[size_t(f) * 4 + 3])];
        double w1 = 0, w2 = 0, s = 0, uu = 0, vv = 0;
        if (RayTriangle(o, d, v0, v1, v2, &s, &w1, &w2)) {
            uu = w1 + w2;
            vv = w2;
        } else if (RayTriangle(o, d, v0, v2, v3, &s, &w1, &w2)) {
            uu = w1;
            vv = w1 + w2;
        } else {
            return false;
        }
        *dist = s;
        *u = Clamp01f(float(uu));
        *v = Clamp01f(float(vv));
        return true;
    }

    // Nearest hit: (face, u, v, dist). Ties keep the lower face id. Every
    // face occludes; a nearest hit on a masked face is a miss.
    bool Pick(Vec3 const &o, Vec3 const &d, int *outFace, float *outU, float *outV,
              double *outDist) const
    {
        int best = -1;
        double bestDist = std::numeric_limits<double>::infinity();
        float bu = 0, bv = 0;
        auto consider = [&](int f) {
            double dist;
            float u, v;
            if (!TestFace(f, o, d, &dist, &u, &v)) return;
            if (dist < bestDist || (dist == bestDist && f < best)) {
                best = f;
                bestDist = dist;
                bu = u;
                bv = v;
            }
        };
        int const F = NumFaces();
        if (!hasGrid) {
            for (int f = 0; f < F; ++f) consider(f);
        } else {
            // Clip the ray to the grid box, then walk cells in t order.
            double tmin = 0.0, tmax = std::numeric_limits<double>::infinity();
            for (int ax = 0; ax < 3; ++ax) {
                double const blo = origin[ax];
                double const bhi = origin[ax] + cellSize * cells[ax];
                if (d[ax] == 0.0) {
                    if (o[ax] < blo || o[ax] > bhi) return false;
                } else {
                    double t0 = (blo - o[ax]) / d[ax];
                    double t1 = (bhi - o[ax]) / d[ax];
                    if (t0 > t1) std::swap(t0, t1);
                    tmin = std::max(tmin, t0);
                    tmax = std::min(tmax, t1);
                }
            }
            if (tmin > tmax) return false;
            int cell[3], step[3];
            double tnext[3], tdelta[3];
            for (int ax = 0; ax < 3; ++ax) {
                cell[ax] = CellIndex(o[ax] + d[ax] * tmin, ax);
                if (d[ax] > 0.0) {
                    step[ax] = 1;
                    tnext[ax] = (origin[ax] + cellSize * (cell[ax] + 1) - o[ax]) / d[ax];
                    tdelta[ax] = cellSize / d[ax];
                } else if (d[ax] < 0.0) {
                    step[ax] = -1;
                    tnext[ax] = (origin[ax] + cellSize * cell[ax] - o[ax]) / d[ax];
                    tdelta[ax] = -cellSize / d[ax];
                } else {
                    step[ax] = 0;
                    tnext[ax] = std::numeric_limits<double>::infinity();
                    tdelta[ax] = std::numeric_limits<double>::infinity();
                }
            }
            uint32_t const id = NextStamp();
            int guard = 3 * (cells[0] + cells[1] + cells[2]) + 8;
            while (guard-- > 0) {
                if (cell[0] < 0 || cell[1] < 0 || cell[2] < 0 || cell[0] >= cells[0] ||
                    cell[1] >= cells[1] || cell[2] >= cells[2])
                    break;
                size_t const c = Flat(cell[0], cell[1], cell[2]);
                for (uint32_t k = cellStart[c]; k < cellStart[c + 1]; ++k) {
                    int const f = cellFaces[k];
                    if (stamp[size_t(f)] == id) continue;
                    stamp[size_t(f)] = id;
                    consider(f);
                }
                int ax = 0;
                if (tnext[0] <= tnext[1] && tnext[0] <= tnext[2]) ax = 0;
                else if (tnext[1] <= tnext[2]) ax = 1;
                else ax = 2;
                double const exitT = tnext[ax];
                // Every hit at t <= exitT lies in a cell already walked.
                if (best >= 0 && bestDist <= exitT) break;
                if (exitT > tmax) break;
                tnext[ax] += tdelta[ax];
                cell[ax] += step[ax];
            }
        }
        if (best < 0 || !Allowed(best)) return false;
        *outFace = best;
        *outU = bu;
        *outV = bv;
        *outDist = bestDist;
        return true;
    }

    // Faces whose grid cells overlap the sphere (sorted, unique), or every
    // face without a grid.
    void Nearby(Vec3 const &p, double r, std::vector<int> *out) const
    {
        out->clear();
        int const F = NumFaces();
        if (!hasGrid) {
            out->reserve(size_t(F));
            for (int f = 0; f < F; ++f) out->push_back(f);
            return;
        }
        int l[3], h[3];
        for (int ax = 0; ax < 3; ++ax) {
            l[ax] = CellIndex(p[ax] - r, ax);
            h[ax] = CellIndex(p[ax] + r, ax);
        }
        uint32_t const id = NextStamp();
        for (int z = l[2]; z <= h[2]; ++z)
            for (int y = l[1]; y <= h[1]; ++y)
                for (int x = l[0]; x <= h[0]; ++x) {
                    size_t const c = Flat(x, y, z);
                    for (uint32_t k = cellStart[c]; k < cellStart[c + 1]; ++k) {
                        int const f = cellFaces[k];
                        if (stamp[size_t(f)] == id) continue;
                        stamp[size_t(f)] = id;
                        out->push_back(f);
                    }
                }
        std::sort(out->begin(), out->end());
    }

    Vec3 FacePoint(int f, double u, double v) const
    {
        Vec3 const &v0 = points[size_t(faces[size_t(f) * 4 + 0])];
        Vec3 const &v1 = points[size_t(faces[size_t(f) * 4 + 1])];
        Vec3 const &v2 = points[size_t(faces[size_t(f) * 4 + 2])];
        Vec3 const &v3 = points[size_t(faces[size_t(f) * 4 + 3])];
        double const iu = 1.0 - u, iv = 1.0 - v;
        Vec3 out;
        for (int k = 0; k < 3; ++k)
            out[k] = v0[k] * iu * iv + v1[k] * u * iv + v2[k] * u * v + v3[k] * iu * v;
        return out;
    }

    // The dab centre in quad f's own (u, v) frame, NOT clamped to the face:
    // the plane coordinates of p over the triangle the closest point fell
    // in, extended linearly past the border. A spill stamp must be centred
    // here: centring it at the clamped closest point (the face border) put
    // the disc's full-weight core on every neighbour inside the radius, so
    // the falloff vanished across faces. Exact on planar quads.
    static bool PlaneUV(Vec3 const &p, Vec3 const &o, Vec3 const &e1, Vec3 const &e2,
                        double *w1, double *w2)
    {
        Vec3 const d = Sub(p, o);
        double const a = Dot(e1, e1), b = Dot(e1, e2), c = Dot(e2, e2);
        double const det = a * c - b * b;
        if (!(std::fabs(det) > 1e-30)) return false;
        double const r1 = Dot(e1, d), r2 = Dot(e2, d);
        *w1 = (r1 * c - r2 * b) / det;
        *w2 = (r2 * a - r1 * b) / det;
        return std::isfinite(*w1) && std::isfinite(*w2);
    }

    // (dist, u, v) of the closest point on quad f; extU/extV (optional) get
    // the unclamped centre (see PlaneUV).
    double QuadClosest(int f, Vec3 const &p, float *u, float *v, float *extU = nullptr,
                       float *extV = nullptr) const
    {
        Vec3 const &v0 = points[size_t(faces[size_t(f) * 4 + 0])];
        Vec3 const &v1 = points[size_t(faces[size_t(f) * 4 + 1])];
        Vec3 const &v2 = points[size_t(faces[size_t(f) * 4 + 2])];
        Vec3 const &v3 = points[size_t(faces[size_t(f) * 4 + 3])];
        double aw1, aw2, bw1, bw2;
        double const dA = ClosestOnTriangle(p, v0, v1, v2, &aw1, &aw2);
        double const dB = ClosestOnTriangle(p, v0, v2, v3, &bw1, &bw2);
        double uu, vv, dist;
        if (dA <= dB) {
            uu = aw1 + aw2;
            vv = aw2;
            dist = std::sqrt(dA);
        } else {
            uu = bw1;
            vv = bw1 + bw2;
            dist = std::sqrt(dB);
        }
        *u = Clamp01f(float(uu));
        *v = Clamp01f(float(vv));
        if (extU && extV) {
            double w1 = 0.0, w2 = 0.0;
            *extU = *u;
            *extV = *v;
            if (dA <= dB) {
                if (PlaneUV(p, v0, Sub(v1, v0), Sub(v2, v0), &w1, &w2)) {
                    *extU = float(w1 + w2);
                    *extV = float(w2);
                }
            } else if (PlaneUV(p, v0, Sub(v2, v0), Sub(v3, v0), &w1, &w2)) {
                *extU = float(w1);
                *extV = float(w1 + w2);
            }
        }
        return dist;
    }

    struct Stamp {
        int face;
        float u, v, radiusUV;
    };

    // The dab's footprint: primary first (exact u, v), then every other
    // paintable face within the world radius, ascending face id, each with
    // the dab centre in its own unclamped (u, v) frame (may lie outside
    // [0, 1]).
    void Footprint(int face, float u, float v, Vec3 const &point, double worldR,
                   bool primaryRadiusGiven, float primaryRadiusUV,
                   std::vector<Stamp> *out) const
    {
        out->clear();
        double const pe = edges[size_t(face)];
        float const primR = primaryRadiusGiven
                                ? primaryRadiusUV
                                : float(pe > 1e-12 ? worldR / pe : worldR);
        out->push_back({face, u, v, primR});
        if (!(worldR >= 0.0) || !std::isfinite(worldR) || !Finite3(point)) return;
        double const limit = worldR + 1e-9;
        double const limitSq = limit * limit;
        std::vector<int> candidates;
        Nearby(point, limit, &candidates);
        for (int other : candidates) {
            if (other == face || !Allowed(other)) continue;
            double const edge = edges[size_t(other)];
            if (edge <= 1e-12) continue;
            double box = 0.0;
            for (int ax = 0; ax < 3; ++ax) {
                if (point[ax] < bmin[size_t(other)][ax])
                    box += (bmin[size_t(other)][ax] - point[ax]) *
                           (bmin[size_t(other)][ax] - point[ax]);
                else if (point[ax] > bmax[size_t(other)][ax])
                    box += (point[ax] - bmax[size_t(other)][ax]) *
                           (point[ax] - bmax[size_t(other)][ax]);
            }
            if (box > limitSq) continue;
            float cu, cv, eu, ev;
            double const dist = QuadClosest(other, point, &cu, &cv, &eu, &ev);
            if (dist > limit) continue;
            out->push_back({other, eu, ev, float(worldR / edge)});
        }
    }
};

struct MeshHandle {
    std::shared_ptr<MeshData> data;
};

MeshData *MeshOf(void *mesh)
{
    return mesh ? static_cast<MeshHandle *>(mesh)->data.get() : nullptr;
}

// ---------------------------------------------------------------------------
// Stroke
// ---------------------------------------------------------------------------

int CornerS(int corner, int res) { return (corner == 1 || corner == 2) ? res - 1 : 0; }
int CornerT(int corner, int res) { return (corner >= 2) ? res - 1 : 0; }

// One map being painted non-accumulatively (UsdGenBrushAccumulator's
// max-weight model): the texel accumulator for set / erase / add, plus the
// corner-smooth segment (per face-corner max weight over a snapshot of the
// corners taken when the segment began). Working and Commit each own one,
// so both replay the same dabs through the same state machine.
struct Painter {
    std::shared_ptr<UsdGenAttributeMap> map;
    std::unique_ptr<usdGen::UsdGenBrushAccumulator> texels;
    bool cornerActive = false;
    int cornerChannel = -1;
    std::vector<float> cornerBase;  // numFaces * 4 * channels
    std::vector<float> cornerW;     // numFaces * 4

    explicit Painter(std::shared_ptr<UsdGenAttributeMap> m)
        : map(std::move(m)),
          texels(std::make_unique<usdGen::UsdGenBrushAccumulator>(map.get()))
    {
    }
};

struct StrokeData {
    std::shared_ptr<MeshData> mesh;  // may be null
    int numFaces = 0, res = 1, channels = 1;
    float defaultValue = 0.0f;
    std::shared_ptr<UsdGenAttributeMap> base, working;
    std::unique_ptr<Painter> painter;  // paints `working`
    std::vector<UsdGenBrushDab> dabs;  // recorded primaries
    std::vector<uint8_t> touchedFlag;
    std::vector<int> touched;

    size_t Index(int face, int s, int t, int c) const
    {
        return ((size_t(face) * size_t(res) + size_t(t)) * size_t(res) + size_t(s)) *
                   size_t(channels) +
               size_t(c);
    }

    float Corner(UsdGenAttributeMap const &map, int face, int corner, int c) const
    {
        return map.Data()[Index(face, CornerS(corner, res), CornerT(corner, res), c)];
    }

    // Bilinear corners -> every texel of `face`, channels [c0, c1).
    void Upsample(UsdGenAttributeMap *map, int face, float const *corners /*4*channels*/,
                  int c0, int c1) const
    {
        float *data = map->MutableData();
        for (int t = 0; t < res; ++t) {
            double const v = res > 1 ? double(t) / double(res - 1) : 0.5;
            for (int s = 0; s < res; ++s) {
                double const u = res > 1 ? double(s) / double(res - 1) : 0.5;
                for (int c = c0; c < c1; ++c) {
                    double const top = corners[0 * channels + c] * (1.0 - u) +
                                       corners[1 * channels + c] * u;
                    double const bottom = corners[3 * channels + c] * (1.0 - u) +
                                          corners[2 * channels + c] * u;
                    data[Index(face, s, t, c)] = float(top * (1.0 - v) + bottom * v);
                }
            }
        }
    }

    void Touch(int face)
    {
        if (face < 0 || face >= numFaces) return;
        if (!touchedFlag[size_t(face)]) {
            touchedFlag[size_t(face)] = 1;
            touched.push_back(face);
        }
    }

    // Corner-level cross-face smooth (see the header), non-accumulating:
    // every face-corner keeps the max dab weight it received during the
    // segment, and its value is segmentBase + (target - segmentBase) * wmax,
    // target being the vertex-ring mean of the SEGMENT-START corners. So a
    // stroke's overlapping stamps relax a corner once, by its strongest
    // weight, instead of iterating it to the ring mean.
    void SmoothCorners(Painter &painter, UsdGenBrushDab const &dab,
                       std::vector<MeshData::Stamp> const &stamps, bool track)
    {
        MeshData const &m = *mesh;
        UsdGenAttributeMap *map = painter.map.get();
        if (!painter.cornerActive || painter.cornerChannel != dab.channel) {
            painter.texels->Flush();
            painter.cornerActive = true;
            painter.cornerChannel = dab.channel;
            painter.cornerBase.resize(size_t(numFaces) * 4 * size_t(channels));
            ExtractCorners(*map, painter.cornerBase.data());
            painter.cornerW.assign(size_t(numFaces) * 4, 0.0f);
        }
        auto baseCorner = [&](int face, int corner, int c) {
            return painter.cornerBase[(size_t(face) * 4 + size_t(corner)) * size_t(channels) +
                                      size_t(c)];
        };
        int const c0 = dab.channel < 0 ? 0 : dab.channel;
        int const c1 = dab.channel < 0 ? channels : dab.channel + 1;
        struct Hit {
            int face, corner;
            float k;
        };
        std::vector<Hit> hits;
        for (auto const &st : stamps) {
            if (!(st.radiusUV > 0.0f)) continue;
            for (int i = 0; i < 4; ++i) {
                double const cu = (i == 1 || i == 2) ? 1.0 : 0.0;
                double const cv = (i >= 2) ? 1.0 : 0.0;
                double const dist = std::sqrt((cu - st.u) * (cu - st.u) + (cv - st.v) * (cv - st.v));
                float const w = usdGen::UsdGenBrushWeight(dab.falloff, dab.hardness, dist,
                                                          st.radiusUV);
                float const k = dab.strength * w;
                float &wmax = painter.cornerW[size_t(st.face) * 4 + size_t(i)];
                if (k > wmax) {
                    wmax = k;
                    hits.push_back({st.face, i, k});
                }
            }
        }
        if (hits.empty()) return;
        // Vertex means, memoised per (vertex) for this dab. A hash map keeps
        // lookups O(1) (a 1024-primary move smooths many hits) and its value
        // references stay valid across inserts.
        std::unordered_map<int, std::vector<double>> memo;
        memo.reserve(hits.size() * 2);
        auto vertexMean = [&](int p) -> std::vector<double> const & {
            auto const found = memo.find(p);
            if (found != memo.end()) return found->second;
            std::vector<double> mean(size_t(channels), 0.0);
            uint32_t const a = m.cornerStart[size_t(p)], b = m.cornerStart[size_t(p) + 1];
            for (uint32_t k = a; k < b; ++k) {
                int const fc = m.cornerList[k];
                for (int c = c0; c < c1; ++c) mean[size_t(c)] += baseCorner(fc / 4, fc % 4, c);
            }
            if (b > a)
                for (double &x : mean) x /= double(b - a);
            return memo.emplace(p, std::move(mean)).first->second;
        };
        std::vector<float> next(hits.size() * size_t(channels), 0.0f);
        for (size_t h = 0; h < hits.size(); ++h) {
            int const p = m.faces[size_t(hits[h].face) * 4 + size_t(hits[h].corner)];
            std::vector<double> target(size_t(channels), 0.0);
            int n = 0;
            auto add = [&](int q) {
                std::vector<double> const &mean = vertexMean(q);
                for (int c = c0; c < c1; ++c) target[size_t(c)] += mean[size_t(c)];
                ++n;
            };
            add(p);
            for (uint32_t k = m.nbrStart[size_t(p)]; k < m.nbrStart[size_t(p) + 1]; ++k)
                add(m.nbrList[k]);
            for (int c = c0; c < c1; ++c) {
                double const cur = baseCorner(hits[h].face, hits[h].corner, c);
                double const tgt = target[size_t(c)] / double(n);
                float value = float(cur + (tgt - cur) * hits[h].k);
                if (map->Clamp01()) value = Clamp01f(value);
                next[h * size_t(channels) + size_t(c)] = value;
            }
        }
        // Write corners, then re-upsample every affected face once.
        std::vector<int> faces;
        std::unordered_map<int, size_t> slotOf;
        std::vector<float> corners;  // per affected face: 4*channels
        for (size_t h = 0; h < hits.size(); ++h) {
            int const f = hits[h].face;
            size_t slot = 0;
            auto const it = slotOf.find(f);
            if (it == slotOf.end()) {
                faces.push_back(f);
                slot = faces.size() - 1;
                slotOf.emplace(f, slot);
                for (int i = 0; i < 4; ++i)
                    for (int c = 0; c < channels; ++c) corners.push_back(Corner(*map, f, i, c));
            } else {
                slot = it->second;
            }
            for (int c = c0; c < c1; ++c)
                corners[slot * 4 * size_t(channels) + size_t(hits[h].corner) * size_t(channels) +
                        size_t(c)] = next[h * size_t(channels) + size_t(c)];
        }
        for (size_t slot = 0; slot < faces.size(); ++slot) {
            Upsample(map, faces[slot], &corners[slot * 4 * size_t(channels)], c0, c1);
            if (track) Touch(faces[slot]);
        }
    }

    void ApplyPrimary(Painter &painter, UsdGenBrushDab const &dab, bool track,
                      int *stampCount)
    {
        std::vector<MeshData::Stamp> stamps;
        if (mesh && dab.radius > 0.0f) {
            double const edge = mesh->edges[size_t(dab.face)];
            double const worldR = edge > 1e-12 ? double(dab.radius) * edge : double(dab.radius);
            Vec3 const point = mesh->FacePoint(dab.face, dab.u, dab.v);
            mesh->Footprint(dab.face, dab.u, dab.v, point, worldR, true, dab.radius, &stamps);
        } else {
            stamps.push_back({dab.face, dab.u, dab.v, dab.radius});
        }
        if (stampCount) *stampCount += int(stamps.size());
        if (dab.mode == UsdGenBrushMode::Smooth && mesh) {
            if (dab.strength > 0.0f) SmoothCorners(painter, dab, stamps, track);
            return;
        }
        // Any texel-level dab ends a corner-smooth segment.
        painter.cornerActive = false;
        for (auto const &st : stamps) {
            UsdGenBrushDab stamp = dab;
            stamp.face = st.face;
            stamp.u = st.u;
            stamp.v = st.v;
            stamp.radius = st.radiusUV;
            painter.texels->Apply(stamp);
            if (track && dab.strength > 0.0f) Touch(st.face);
        }
    }

    // A fresh painter over a clone of the base.
    std::unique_ptr<Painter> FreshPainter() const
    {
        return std::make_unique<Painter>(base->Clone());
    }

    // Rebuild working = base + recorded dabs after a failed apply, and mark
    // every face touched so the caller repaints the rolled-back grid.
    // Never throws; on failure working stays as it was.
    void Resync() noexcept
    {
        try {
            auto fresh = FreshPainter();
            for (auto const &dab : dabs) ApplyPrimary(*fresh, dab, false, nullptr);
            working = fresh->map;
            painter = std::move(fresh);
            for (int f = 0; f < numFaces; ++f) Touch(f);
        } catch (...) {
        }
    }

    void ExtractCorners(UsdGenAttributeMap const &map, float *out) const
    {
        for (int f = 0; f < numFaces; ++f)
            for (int i = 0; i < 4; ++i)
                for (int c = 0; c < channels; ++c)
                    out[(size_t(f) * 4 + size_t(i)) * size_t(channels) + size_t(c)] =
                        Corner(map, f, i, c);
    }
};

StrokeData *StrokeOf(void *stroke) { return static_cast<StrokeData *>(stroke); }

void HeatColor(float t, float *rgb)
{
    t = Clamp01f(t);
    if (t < 0.5f) {
        rgb[0] = t * 2.0f;
        rgb[1] = 0.0f;
        rgb[2] = 0.0f;
    } else if (t < 0.75f) {
        rgb[0] = 1.0f;
        rgb[1] = (t - 0.5f) * 4.0f;
        rgb[2] = 0.0f;
    } else {
        rgb[0] = 1.0f;
        rgb[1] = 1.0f;
        rgb[2] = (t - 0.75f) * 4.0f;
    }
}

float Normalise(float value, float lo, float hi)
{
    if (hi <= lo) return value < lo ? 0.0f : 1.0f;
    return Clamp01f((value - lo) / (hi - lo));
}

}  // namespace

extern "C" {
size_t usdGenImaging_copy_playback_status_json(
    const char* selectorJson, double requestedFrame, char* buffer, size_t capacity) {
    try {
        std::string const status = UsdGenGroomSceneIndex::PlaybackStatusJson(
            selectorJson ? selectorJson : "", requestedFrame);
        size_t const required = status.size() + 1;
        if (buffer && capacity >= required) std::memcpy(buffer, status.c_str(), required);
        return required;
    } catch (...) { return 0; }
}

int UsdGenBrush_ApiVersion(void) { return USDGEN_BRUSH_API_VERSION; }

void *UsdGenBrush_MeshCreate(const double *worldPoints, int numPoints,
                             const int *faceVertexIndices, int numFaces)
{
    return UsdGenBrush_MeshCreateMasked(worldPoints, numPoints, faceVertexIndices, numFaces,
                                        nullptr, 0);
}

void *UsdGenBrush_MeshCreateMasked(const double *worldPoints, int numPoints,
                                   const int *faceVertexIndices, int numFaces,
                                   const int *maskFaces, int maskCount)
{
    try {
        if (!worldPoints || !faceVertexIndices || numPoints <= 0 || numFaces <= 0) return nullptr;
        auto data = std::make_shared<MeshData>();
        data->points.resize(size_t(numPoints));
        for (int p = 0; p < numPoints; ++p) {
            Vec3 const v{worldPoints[size_t(p) * 3], worldPoints[size_t(p) * 3 + 1],
                         worldPoints[size_t(p) * 3 + 2]};
            if (!Finite3(v)) return nullptr;
            data->points[size_t(p)] = v;
        }
        data->faces.assign(faceVertexIndices, faceVertexIndices + size_t(numFaces) * 4);
        for (int v : data->faces)
            if (v < 0 || v >= numPoints) return nullptr;
        if (maskFaces) {
            if (maskCount < 0) return nullptr;
            data->allowed.assign(size_t(numFaces), 0);
            for (int k = 0; k < maskCount; ++k) {
                int const f = maskFaces[k];
                if (f < 0 || f >= numFaces) return nullptr;
                data->allowed[size_t(f)] = 1;
            }
        }
        data->Build();
        auto *handle = new MeshHandle;
        handle->data = std::move(data);
        return handle;
    } catch (...) {
        return nullptr;
    }
}

void UsdGenBrush_MeshDestroy(void *mesh)
{
    try {
        delete static_cast<MeshHandle *>(mesh);
    } catch (...) {
    }
}

int UsdGenBrush_MeshFaceCount(void *mesh)
{
    MeshData const *m = MeshOf(mesh);
    return m ? m->NumFaces() : -1;
}

int UsdGenBrush_MeshFaceInMask(void *mesh, int face)
{
    MeshData const *m = MeshOf(mesh);
    if (!m || face < 0 || face >= m->NumFaces()) return -1;
    return m->Allowed(face) ? 1 : 0;
}

int UsdGenBrush_MeshPick(void *mesh, const double *rayOrigin3, const double *rayDir3,
                         int *outFace, float *outU, float *outV, double *outPoint3)
{
    try {
        MeshData const *m = MeshOf(mesh);
        if (!m || !rayOrigin3 || !rayDir3) return -1;
        Vec3 const o{rayOrigin3[0], rayOrigin3[1], rayOrigin3[2]};
        Vec3 const d{rayDir3[0], rayDir3[1], rayDir3[2]};
        if (!Finite3(o) || !Finite3(d) || (d[0] == 0.0 && d[1] == 0.0 && d[2] == 0.0))
            return -1;
        int face = -1;
        float u = 0, v = 0;
        double dist = 0;
        if (!m->Pick(o, d, &face, &u, &v, &dist)) return 0;
        if (outFace) *outFace = face;
        if (outU) *outU = u;
        if (outV) *outV = v;
        if (outPoint3)
            for (int k = 0; k < 3; ++k) outPoint3[k] = o[k] + d[k] * dist;
        return 1;
    } catch (...) {
        return -1;
    }
}

int UsdGenBrush_MeshFootprint(void *mesh, int face, float u, float v, const double *point3,
                              float worldRadius, int *outFaces, float *outU, float *outV,
                              float *outRadiusUV, int cap)
{
    try {
        MeshData const *m = MeshOf(mesh);
        if (!m || face < 0 || face >= m->NumFaces() || cap <= 0) return -1;
        if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(worldRadius) ||
            worldRadius < 0.0f)
            return -1;
        Vec3 const point = point3 ? Vec3{point3[0], point3[1], point3[2]}
                                  : m->FacePoint(face, u, v);
        std::vector<MeshData::Stamp> stamps;
        m->Footprint(face, u, v, point, worldRadius, false, 0.0f, &stamps);
        int const n = std::min(cap, int(stamps.size()));
        for (int i = 0; i < n; ++i) {
            if (outFaces) outFaces[i] = stamps[size_t(i)].face;
            if (outU) outU[i] = stamps[size_t(i)].u;
            if (outV) outV[i] = stamps[size_t(i)].v;
            if (outRadiusUV) outRadiusUV[i] = stamps[size_t(i)].radiusUV;
        }
        // The TOTAL count (like StrokeTakeTouched): > cap means truncated.
        return int(stamps.size());
    } catch (...) {
        return -1;
    }
}

float UsdGenBrush_MeshFaceEdgeLen(void *mesh, int face)
{
    MeshData const *m = MeshOf(mesh);
    if (!m || face < 0 || face >= m->NumFaces()) return 0.0f;
    float const e = m->edges[size_t(face)];
    return std::isfinite(e) ? e : 0.0f;
}

int UsdGenBrush_MeshSuggestResolution(void *mesh, int budgetTexels, char *infoBuf, int infoLen)
{
    try {
        MeshData const *m = MeshOf(mesh);
        if (!m) return -1;
        int const F = m->NumFaces();
        if (budgetTexels <= 0) budgetTexels = 4000000;
        std::vector<float> e(m->meanEdges);
        std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
        double const median = e.empty() ? 0.0 : e[e.size() / 2];
        // Only four corners per face persist, so texels only buy in-stroke
        // detail: ~32 across the median (every) face, lowered to the budget.
        int res = 32;
        while (res > 4 && double(F) * res * res > double(budgetTexels)) res /= 2;
        if (infoBuf && infoLen > 0) {
            double const texels = double(F) * res * res;
            std::snprintf(infoBuf, size_t(infoLen), "%d px/face (median edge %.3g) -> %.1f MTexel",
                          res, median, texels / 1.0e6);
        }
        return res;
    } catch (...) {
        return -1;
    }
}

void *UsdGenBrush_StrokeCreate(void *mesh, int numFaces, int resolution, int channels,
                               const float *baseCorners, float defaultValue)
{
    try {
        MeshHandle *mh = static_cast<MeshHandle *>(mesh);
        if (mh && mh->data->NumFaces() != numFaces) return nullptr;
        UsdGenAttributeMapSpec spec;
        spec.numFaces = numFaces;
        spec.resolution = resolution;
        spec.channels = channels;
        spec.defaultValue = defaultValue;
        std::string error;
        auto base = UsdGenAttributeMap::Create(spec, &error);
        if (!base) return nullptr;
        auto *s = new StrokeData;
        if (mh) s->mesh = mh->data;
        s->numFaces = numFaces;
        s->res = resolution;
        s->channels = channels;
        s->defaultValue = base->DefaultValue();
        if (baseCorners) {
            std::vector<float> corners(size_t(4) * size_t(channels));
            for (int f = 0; f < numFaces; ++f) {
                bool finite = true;
                for (size_t k = 0; k < corners.size(); ++k) {
                    corners[k] = baseCorners[size_t(f) * corners.size() + k];
                    finite = finite && std::isfinite(corners[k]);
                }
                // A non-finite corner leaves the face at the default
                // (brushAuthor's upsample rule).
                if (finite) s->Upsample(base.get(), f, corners.data(), 0, channels);
            }
        }
        s->base = base;
        s->painter = s->FreshPainter();
        s->working = s->painter->map;
        s->touchedFlag.assign(size_t(numFaces), 0);
        return s;
    } catch (...) {
        return nullptr;
    }
}

void UsdGenBrush_StrokeDestroy(void *stroke)
{
    try {
        delete StrokeOf(stroke);
    } catch (...) {
    }
}

int UsdGenBrush_StrokeDab(void *stroke, int face, float u, float v, float radiusUV,
                          float hardness, float strength, float value, int channel, int mode,
                          int falloff, int isMove, float spacing)
{
    try {
        StrokeData *s = StrokeOf(stroke);
        if (!s) return -1;
        if (face < 0 || face >= s->numFaces) return -2;
        if (!std::isfinite(u) || !std::isfinite(v)) return -3;
        if (!std::isfinite(radiusUV) || radiusUV < 0.0f) return -4;
        if (!std::isfinite(strength) || strength < 0.0f || strength > 1.0f) return -5;
        if (!std::isfinite(value)) return -6;
        if (!std::isfinite(hardness) || hardness < 0.0f || hardness > 1.0f) return -7;
        if (channel < -1 || channel >= s->channels) return -8;
        if (mode < 0 || mode > 3 || falloff < 0 || falloff > 2) return -9;
        if (isMove && (!std::isfinite(spacing) || spacing <= 0.0f || spacing > 1.0f)) return -10;
        if (s->mesh && !s->mesh->Allowed(face)) return -11;
        UsdGenBrushDab dab;
        dab.face = face;
        dab.u = u;
        dab.v = v;
        dab.radius = radiusUV;
        dab.hardness = hardness;
        dab.strength = strength;
        dab.value = value;
        dab.channel = channel;
        dab.mode = mode == 0   ? UsdGenBrushMode::Set
                   : mode == 1 ? UsdGenBrushMode::Add
                   : mode == 2 ? UsdGenBrushMode::Smooth
                               : UsdGenBrushMode::Erase;
        dab.falloff = falloff == 0   ? UsdGenBrushFalloff::Constant
                      : falloff == 1 ? UsdGenBrushFalloff::Linear
                                     : UsdGenBrushFalloff::Smooth;
        std::vector<UsdGenBrushDab> primaries;
        if (isMove && !s->dabs.empty() && s->dabs.back().face == face && radiusUV > 0.0f) {
            UsdGenBrushDab const prev = s->dabs.back();
            double const dx = double(u) - prev.u, dy = double(v) - prev.v;
            double const dist = std::sqrt(dx * dx + dy * dy);
            double const step = double(spacing) * radiusUV;
            int64_t steps = step > 0.0 ? int64_t(std::ceil(dist / step)) : 1;
            steps = std::max<int64_t>(1, std::min<int64_t>(1024, steps));
            for (int64_t i = 1; i <= steps; ++i) {
                double const t = double(i) / double(steps);
                UsdGenBrushDab stamp = dab;
                stamp.u = float(prev.u + dx * t);
                stamp.v = float(prev.v + dy * t);
                primaries.push_back(stamp);
            }
        } else {
            primaries.push_back(dab);
        }
        // Record only after every primary applied: a throw mid-loop must not
        // leave dabs recorded that the working grid never got (commit would
        // then differ from what the viewport showed). A partial apply is
        // rolled back by rebuilding working from base + the recorded dabs.
        int stamps = 0;
        try {
            for (auto const &p : primaries) s->ApplyPrimary(*s->painter, p, true, &stamps);
        } catch (...) {
            s->Resync();
            return -100;
        }
        s->dabs.insert(s->dabs.end(), primaries.begin(), primaries.end());
        return stamps;
    } catch (...) {
        return -100;
    }
}

int UsdGenBrush_StrokeTakeTouched(void *stroke, int *outFaces, int cap)
{
    try {
        StrokeData *s = StrokeOf(stroke);
        if (!s) return -1;
        std::sort(s->touched.begin(), s->touched.end());
        int const total = int(s->touched.size());
        int const n = outFaces ? std::min(std::max(cap, 0), total) : 0;
        for (int i = 0; i < n; ++i) outFaces[i] = s->touched[size_t(i)];
        if (outFaces && n == total) {
            for (int f : s->touched) s->touchedFlag[size_t(f)] = 0;
            s->touched.clear();
        }
        return total;
    } catch (...) {
        return -1;
    }
}

int UsdGenBrush_StrokeDabCount(void *stroke)
{
    StrokeData *s = StrokeOf(stroke);
    return s ? int(s->dabs.size()) : -1;
}

int UsdGenBrush_StrokeFaceCount(void *stroke)
{
    StrokeData *s = StrokeOf(stroke);
    return s ? s->numFaces : -1;
}

int UsdGenBrush_StrokeChannels(void *stroke)
{
    StrokeData *s = StrokeOf(stroke);
    return s ? s->channels : -1;
}

int UsdGenBrush_StrokeWorkingCorners(void *stroke, float *out)
{
    try {
        StrokeData *s = StrokeOf(stroke);
        if (!s || !out) return -1;
        s->ExtractCorners(*s->working, out);
        return s->numFaces * 4 * s->channels;
    } catch (...) {
        return -1;
    }
}

int UsdGenBrush_StrokeCommitCorners(void *stroke, float *out)
{
    try {
        StrokeData *s = StrokeOf(stroke);
        if (!s || !out) return -1;
        auto painter = s->FreshPainter();
        for (auto const &dab : s->dabs) s->ApplyPrimary(*painter, dab, false, nullptr);
        s->ExtractCorners(*painter->map, out);
        return s->numFaces * 4 * s->channels;
    } catch (...) {
        return -1;
    }
}

void UsdGenBrush_StrokeAbort(void *stroke)
{
    try {
        StrokeData *s = StrokeOf(stroke);
        if (!s) return;
        s->dabs.clear();
        s->painter = s->FreshPainter();
        s->working = s->painter->map;
        for (int f = 0; f < s->numFaces; ++f) s->Touch(f);
    } catch (...) {
    }
}

int UsdGenBrush_StrokePreviewColors(void *stroke, int channel, int colorMap, float lo, float hi,
                                    float *outRGB)
{
    try {
        StrokeData *s = StrokeOf(stroke);
        if (!s || !outRGB || channel < 0 || channel >= s->channels) return -1;
        if (colorMap != USDGEN_BRUSH_COLORMAP_HEAT && colorMap != USDGEN_BRUSH_COLORMAP_GRAY)
            return -1;
        if (!std::isfinite(lo) || !std::isfinite(hi)) return -1;
        for (int f = 0; f < s->numFaces; ++f)
            for (int i = 0; i < 4; ++i) {
                float const t = Normalise(s->Corner(*s->working, f, i, channel), lo, hi);
                float *rgb = outRGB + (size_t(f) * 4 + size_t(i)) * 3;
                if (colorMap == USDGEN_BRUSH_COLORMAP_GRAY) {
                    rgb[0] = rgb[1] = rgb[2] = t;
                } else {
                    HeatColor(t, rgb);
                }
            }
        return s->numFaces * 4;
    } catch (...) {
        return -1;
    }
}

int UsdGenBrush_PreviewSet(const char *primPath, const float *rgbFaceVarying, int count)
{
    try {
        if (!primPath || !rgbFaceVarying || count <= 0) return 0;
        SdfPath const path(primPath);
        if (path.IsEmpty() || !path.IsAbsolutePath() || !path.IsPrimPath()) return 0;
        VtVec3fArray colors;
        colors.resize(static_cast<size_t>(count));
        GfVec3f *dst = colors.data();
        for (int i = 0; i < count; ++i)
            dst[i] = GfVec3f(rgbFaceVarying[size_t(i) * 3], rgbFaceVarying[size_t(i) * 3 + 1],
                             rgbFaceVarying[size_t(i) * 3 + 2]);
        return usdGenImaging::UsdGenAttributePreviewRegistry::Get().Set(path, colors) ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

int UsdGenBrush_PreviewClear(const char *primPath)
{
    try {
        if (!primPath) return 0;
        SdfPath const path(primPath);
        if (path.IsEmpty()) return 0;
        return usdGenImaging::UsdGenAttributePreviewRegistry::Get().Clear(path) ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

int UsdGenBrush_PreviewClearAll(void)
{
    try {
        return int(usdGenImaging::UsdGenAttributePreviewRegistry::Get().ClearAll());
    } catch (...) {
        return -1;
    }
}

int UsdGenBrush_PreviewIndexCount(void)
{
    try {
        return int(usdGenImaging::UsdGenAttributePreviewRegistry::Get().IndexCount());
    } catch (...) {
        return -1;
    }
}

long long UsdGenBrush_GroomCookCount(void)
{
    return (long long)PXR_NS::UsdGenGroomSceneIndex::ProcessCookCount();
}

long long UsdGenBrush_GroomPublishCount(void)
{
    return (long long)PXR_NS::UsdGenGroomSceneIndex::ProcessPublishCount();
}

int UsdGenBrush_GroomCurveStats(const char *groomPath, long long *curves,
                                double *totalLength)
{
    if (curves) *curves = 0;
    if (totalLength) *totalLength = 0.0;
    if (!groomPath) return -1;
    try {
        PXR_NS::SdfPath const groom(groomPath);
        if (!groom.IsAbsolutePath() || !groom.IsPrimPath()) return -1;
        uint64_t count = 0;
        double length = 0.0;
        bool const found = PXR_NS::UsdGenImagingTestHook::publishedCurveStats(
            groom, &count, &length);
        if (curves) *curves = (long long)count;
        if (totalLength) *totalLength = length;
        return found ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

}  // extern "C"
