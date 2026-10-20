#ifndef USDGEN_LIMIT_SURFACE_H
#define USDGEN_LIMIT_SURFACE_H
#include "usdGen/graphDesc.h"
#include "usdGen/export.h"
#include <memory>
#include <string>
namespace usdGen {
// Patch evaluation, not interpolation of a finitely subdivided polygon cage.
// Build accepts any Catmull-Clark polygon topology. Evaluate addresses a quad
// face by its coarse (u,v), which the random per-face Scatter path keeps as
// rootPrim/rootUV; EvaluateCagePoint addresses a point of any face by the
// barycentrics of one of its cage triangles, for roots placed on the cage.
class USDGEN_CORE_API UsdGenLimitSurface {
public:
    UsdGenLimitSurface();
    ~UsdGenLimitSurface();
    bool Build(UsdGenSurfaceDesc const&, int isolationLevel, std::string* error);
    // Quad faces only: (u,v) in [0,1]^2 from corner 0, u toward corner 1.
    bool Evaluate(int face, float u, float v, pxr::GfVec3f* p,
                  pxr::GfVec3f* du, pxr::GfVec3f* dv) const;
    // The limit point under a cage point: `corners` are the face-local corner
    // numbers (0..n-1) of a triangle inside face `face`, `bary` the point's
    // barycentrics on it. A quad maps affinely onto its patch; a triangle or
    // n-gon goes through its regular-polygon layout to the per-corner Ptex
    // sub-patch around the nearest corner, so shared edges and corners land
    // on the same limit point from every face. du/dv span that patch with the
    // face's own winding.
    bool EvaluateCagePoint(int face, int const corners[3], pxr::GfVec3f const& bary,
                           pxr::GfVec3f* p, pxr::GfVec3f* du, pxr::GfVec3f* dv) const;
    bool IsHole(int face) const;
private:
    struct State;
    std::unique_ptr<State> _state;
};
// Internal cache keys (usdGen/digest.h): determinism + input sensitivity,
// never golden values, never persisted.
// UsdGenSubdivisionDigest covers the whole subdivision topology:
// scheme/orientation/tags, holes, creases, corners AND faceVertexCounts /
// faceVertexIndices. UsdGenSubdivisionTagsDigest covers everything EXCEPT
// the two face arrays, for callers that already mix those arrays into the
// same key (it must never be the sole topology cover).
USDGEN_CORE_API uint64_t UsdGenSubdivisionDigest(UsdGenSurfaceDesc const&);
USDGEN_CORE_API uint64_t UsdGenSubdivisionTagsDigest(UsdGenSurfaceDesc const&);
}
#endif
