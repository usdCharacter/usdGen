#ifndef USDGEN_LIMIT_SURFACE_H
#define USDGEN_LIMIT_SURFACE_H
#include "usdGen/graphDesc.h"
#include "usdGen/export.h"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec3d.h"
#include <array>
#include <memory>
#include <string>
#include <vector>
namespace usdGen {
// Patch evaluation, not interpolation of a finitely subdivided polygon cage.
// Current Scatter admission is quad Catmull-Clark, preserving rootPrim/rootUV
// as coarse-face/Ptex coordinates without changing the binding ABI.
class USDGEN_CORE_API UsdGenLimitSurface {
public:
    UsdGenLimitSurface();
    ~UsdGenLimitSurface();
    bool Build(UsdGenSurfaceDesc const&, int isolationLevel, std::string* error);
    bool Evaluate(int face, float u, float v, pxr::GfVec3f* p,
                  pxr::GfVec3f* du, pxr::GfVec3f* dv) const;
    bool IsHole(int face) const;
private:
    struct State;
    std::unique_ptr<State> _state;
};

// Posed vertex-limit evaluator for collision. Unlike Scatter's rest-surface
// helper above, this accepts Catmull-Clark, Loop and bilinear base faces,
// including non-quad Catmull-Clark parameterizations. It owns its topology
// and prepared posed patch points; no Stage or Hydra objects enter the core.
class USDGEN_CORE_API UsdGenCollisionLimitSurface {
public:
    UsdGenCollisionLimitSurface();
    ~UsdGenCollisionLimitSurface();
    UsdGenCollisionLimitSurface(UsdGenCollisionLimitSurface&&) noexcept;
    UsdGenCollisionLimitSurface& operator=(UsdGenCollisionLimitSurface&&) noexcept;
    bool Build(UsdGenSurfaceDesc const&, std::string* error);
    bool HasFace(int face) const;
    bool Evaluate(int face, double u, double v, pxr::GfVec3d* p,
                  pxr::GfVec3d* du, pxr::GfVec3d* dv) const;
    bool TessellateFace(int face, int rate,
        std::vector<pxr::GfVec2d>* uv,
        std::vector<std::array<int, 3>>* facets) const;
    bool FaceControlBounds(int face, pxr::GfVec3d* lo,
                           pxr::GfVec3d* hi) const;
    bool FaceIsTriangle(int face) const;
    bool FaceHasSubFaces(int face) const;
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
