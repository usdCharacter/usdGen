// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — private closest-point index of UsdGenCollide's flexible/stiff
// resolve types.
//
// It returns exactly what the former brute-force scan returned: the same
// per-triangle arithmetic (Ericson 5.1.5 in double precision over the float
// corners), degenerate triangles skipped, and the result is the
// lexicographic minimum of (squared distance, triangle sequence) where the
// sequence is collider order, then ascending face, then fan triangle. A
// linear scan that keeps the FIRST minimum produces that same pair, so the
// pushes are bit-identical; only the visiting order changes.
//
// Box pruning is conservative: every node box is padded by a margin far
// larger than the double rounding of the distance computation, and a node
// is skipped only when its padded box distance is STRICTLY greater than the
// best distance so far, so no triangle that could win or tie is skipped.
// strand::TriangleBvh is not used here because its hit carries float
// results computed through barycentrics and keeps degenerate triangles,
// which would change these resolve types' bits.
#ifndef USDGEN_OPS_COLLIDE_BVH_H
#define USDGEN_OPS_COLLIDE_BVH_H

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"

#include <cstdint>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// Closest point on triangle abc (Ericson 5.1.5, double precision). False
/// when the triangle is degenerate (zero or non-finite cross product).
bool UsdGenCollideClosestOnTriangle(GfVec3d const &p, GfVec3d const &a,
                                    GfVec3d const &b, GfVec3d const &c,
                                    GfVec3d *closest, double *dist2);

struct UsdGenCollideHit
{
    bool found = false;
    GfVec3d q, a, b, c;   // closest point + the winning triangle's corners
    double dist2 = 0.0;
    uint32_t sequence = 0;
};

class UsdGenCollideExactBvh
{
public:
    /// Starts an empty index. Add triangles in sequence order, then Build().
    void Clear();
    /// Appends triangle (a, b, c) as the next sequence number. Degenerate
    /// triangles are dropped (the scan skipped them) but still consume a
    /// sequence number, so the numbering stays the scan order.
    void Add(GfVec3f const &a, GfVec3f const &b, GfVec3f const &c);
    void Build();
    bool Empty() const { return _tris.empty(); }
    size_t TriangleCount() const { return _tris.size(); }

    /// The (dist2, sequence)-minimal triangle among those with
    /// dist2 < limitDist2 (found == false when none). Allocation-free.
    UsdGenCollideHit Closest(GfVec3d const &p, double limitDist2) const;

private:
    struct Tri
    {
        GfVec3f a, b, c;
        uint32_t sequence = 0;
    };
    struct Node
    {
        float lo[3], hi[3];
        int left = -1, right = -1;
        uint32_t first = 0, count = 0;
    };
    int BuildRange(uint32_t first, uint32_t count, std::vector<GfVec3f> const &centroids,
                   std::vector<uint32_t> &order, int depth);

    std::vector<Tri> _tris;       // leaf-contiguous after Build()
    std::vector<Node> _nodes;
    uint32_t _nextSequence = 0;
    double _magnitude = 0.0;      // max |coordinate| over every corner
};

}  // namespace usdGen

#endif  // USDGEN_OPS_COLLIDE_BVH_H
