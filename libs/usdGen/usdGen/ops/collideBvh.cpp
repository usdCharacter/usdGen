// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenCollide's exact closest-point index. See collideBvh.h.
#include "usdGen/ops/collideBvh.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

constexpr uint32_t kLeafSize = 4;
constexpr int kMaxDepth = 48;     // median split: depth ~log2(N/4) << 48
constexpr int kStackSize = 128;   // >= kMaxDepth + 2

// Pad relative to the coordinate magnitude: ~16 float ulps, many orders of
// magnitude above the double rounding of a triangle distance, and far below
// any detail a float mesh can carry, so pruning stays effective.
constexpr double kPadScale = 1e-6;

bool Finite(double v) { return std::isfinite(v); }

}  // namespace

bool UsdGenCollideClosestOnTriangle(GfVec3d const &p, GfVec3d const &a,
                                    GfVec3d const &b, GfVec3d const &c,
                                    GfVec3d *closest, double *dist2)
{
    GfVec3d const n = GfCross(b - a, c - a);
    double const len = n.GetLength();
    if (!(len > 0.0) || !Finite(len)) return false;
    GfVec3d const ab = b - a, ac = c - a, ap = p - a;
    double const d1 = GfDot(ab, ap), d2 = GfDot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        *closest = a;
    } else {
        GfVec3d const bp = p - b;
        double const d3 = GfDot(ab, bp), d4 = GfDot(ac, bp);
        if (d3 >= 0.0 && d4 <= d3) {
            *closest = b;
        } else {
            double const vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
                *closest = a + ab * (d1 / (d1 - d3));
            } else {
                GfVec3d const cp = p - c;
                double const d5 = GfDot(ab, cp), d6 = GfDot(ac, cp);
                if (d6 >= 0.0 && d5 <= d6) {
                    *closest = c;
                } else {
                    double const vb = d5 * d2 - d1 * d6;
                    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
                        *closest = a + ac * (d2 / (d2 - d6));
                    } else {
                        double const va = d3 * d6 - d5 * d4;
                        double const d43 = d4 - d3, d56 = d5 - d6;
                        if (va <= 0.0 && d43 >= 0.0 && d56 >= 0.0) {
                            *closest = b + (c - b) * (d43 / (d43 + d56));
                        } else {
                            double const denom = 1.0 / (va + vb + vc);
                            *closest = a + ab * (vb * denom) + ac * (vc * denom);
                        }
                    }
                }
            }
        }
    }
    GfVec3d const d = p - *closest;
    *dist2 = GfDot(d, d);
    return true;
}

void UsdGenCollideExactBvh::Clear()
{
    _tris.clear();
    _nodes.clear();
    _nextSequence = 0;
    _magnitude = 0.0;
}

void UsdGenCollideExactBvh::Add(GfVec3f const &a, GfVec3f const &b, GfVec3f const &c)
{
    uint32_t const sequence = _nextSequence++;
    // The scan's degeneracy test, on the same double-converted corners.
    GfVec3d const ad(a), bd(b), cd(c);
    double const len = GfCross(bd - ad, cd - ad).GetLength();
    if (!(len > 0.0) || !Finite(len)) return;
    Tri t;
    t.a = a;
    t.b = b;
    t.c = c;
    t.sequence = sequence;
    _tris.push_back(t);
    for (GfVec3f const *v : {&a, &b, &c})
        for (int k = 0; k < 3; ++k)
            _magnitude = std::max(_magnitude, std::fabs(double((*v)[k])));
}

void UsdGenCollideExactBvh::Build()
{
    _nodes.clear();
    if (_tris.empty()) return;
    std::vector<GfVec3f> centroids(_tris.size());
    for (size_t i = 0; i < _tris.size(); ++i)
        centroids[i] = (_tris[i].a + _tris[i].b + _tris[i].c) / 3.0f;
    std::vector<uint32_t> order(_tris.size());
    std::iota(order.begin(), order.end(), 0u);
    _nodes.reserve(2 * _tris.size() / kLeafSize + 2);
    BuildRange(0, uint32_t(_tris.size()), centroids, order, 0);
    // Leaf-contiguous triangle storage for the query loop.
    std::vector<Tri> sorted(_tris.size());
    for (size_t i = 0; i < order.size(); ++i) sorted[i] = _tris[order[i]];
    _tris.swap(sorted);
}

int UsdGenCollideExactBvh::BuildRange(uint32_t first, uint32_t count,
                                      std::vector<GfVec3f> const &centroids,
                                      std::vector<uint32_t> &order, int depth)
{
    int const index = int(_nodes.size());
    _nodes.push_back(Node());
    Node node;
    float clo[3], chi[3];
    for (int k = 0; k < 3; ++k) {
        node.lo[k] = clo[k] = std::numeric_limits<float>::max();
        node.hi[k] = chi[k] = -std::numeric_limits<float>::max();
    }
    for (uint32_t i = first; i < first + count; ++i) {
        Tri const &t = _tris[order[i]];
        for (GfVec3f const *v : {&t.a, &t.b, &t.c})
            for (int k = 0; k < 3; ++k) {
                node.lo[k] = std::min(node.lo[k], (*v)[k]);
                node.hi[k] = std::max(node.hi[k], (*v)[k]);
            }
        GfVec3f const &cc = centroids[order[i]];
        for (int k = 0; k < 3; ++k) {
            clo[k] = std::min(clo[k], cc[k]);
            chi[k] = std::max(chi[k], cc[k]);
        }
    }
    if (count <= kLeafSize || depth >= kMaxDepth) {
        node.first = first;
        node.count = count;
        _nodes[size_t(index)] = node;
        return index;
    }
    int axis = 0;
    for (int k = 1; k < 3; ++k)
        if (chi[k] - clo[k] > chi[axis] - clo[axis]) axis = k;
    uint32_t const mid = first + count / 2;
    std::nth_element(order.begin() + first, order.begin() + mid, order.begin() + first + count,
        [&](uint32_t x, uint32_t y) {
            float const cx = centroids[x][axis], cy = centroids[y][axis];
            return cx < cy || (cx == cy && x < y);
        });
    node.left = BuildRange(first, mid - first, centroids, order, depth + 1);
    node.right = BuildRange(mid, first + count - mid, centroids, order, depth + 1);
    _nodes[size_t(index)] = node;
    return index;
}

UsdGenCollideHit UsdGenCollideExactBvh::Closest(GfVec3d const &p, double limitDist2) const
{
    UsdGenCollideHit best;
    if (_nodes.empty()) return best;
    double const pmag = std::max(std::fabs(p[0]), std::max(std::fabs(p[1]), std::fabs(p[2])));
    double const pad = kPadScale * (_magnitude + pmag);
    auto boxDist2 = [&](Node const &node) {
        double d = 0.0;
        for (int k = 0; k < 3; ++k) {
            double const lo = double(node.lo[k]) - pad, hi = double(node.hi[k]) + pad;
            double const v = p[k];
            if (v < lo) d += (lo - v) * (lo - v);
            else if (v > hi) d += (v - hi) * (v - hi);
        }
        return d;
    };
    double bestD2 = limitDist2;
    int stack[kStackSize];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        Node const &node = _nodes[size_t(stack[--top])];
        if (boxDist2(node) > bestD2) continue;
        if (node.left < 0) {
            for (uint32_t i = node.first; i < node.first + node.count; ++i) {
                Tri const &t = _tris[i];
                GfVec3d const a(t.a), b(t.b), c(t.c);
                GfVec3d q;
                double d2 = 0.0;
                if (!UsdGenCollideClosestOnTriangle(p, a, b, c, &q, &d2)) continue;
                if (!(d2 >= 0.0)) continue;
                if (d2 < bestD2 ||
                    (best.found && d2 == bestD2 && t.sequence < best.sequence)) {
                    best.found = true;
                    best.q = q;
                    best.a = a;
                    best.b = b;
                    best.c = c;
                    best.dist2 = d2;
                    best.sequence = t.sequence;
                    bestD2 = d2;
                }
            }
            continue;
        }
        if (top + 2 > kStackSize) continue;  // unreachable: depth <= kMaxDepth
        Node const &l = _nodes[size_t(node.left)];
        Node const &r = _nodes[size_t(node.right)];
        if (boxDist2(l) <= boxDist2(r)) {
            stack[top++] = node.right;
            stack[top++] = node.left;
        } else {
            stack[top++] = node.left;
            stack[top++] = node.right;
        }
    }
    return best;
}

}  // namespace usdGen
