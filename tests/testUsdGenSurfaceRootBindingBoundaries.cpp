#include "usdGen/surfaceRootBindings.h"
#include "usdGenUsd/lower.h"
#include "usdGen/surfaceRootFrames.h"

#include <cmath>
#include <cstdio>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {

bool Near(GfVec3d const &a, GfVec3f const &b)
{
    GfVec3d const d = a - GfVec3d(b[0], b[1], b[2]);
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) <= 1.0e-5;
}

UsdGenSurfaceDesc Ngon()
{
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Ngon");
    surface.restPoints = {{0,0,0}, {2,0,0}, {3,1,0}, {1,3,0}, {-1,1,0}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {5};
    surface.faceVertexIndices = {0,1,2,3,4};
    return surface;
}

bool Bind(UsdGenSurfaceDesc const &surface, VtVec3fArray const &roots,
          UsdGenSurfaceRootBindingResult *result, std::string *error)
{
    return UsdGenBuildRestSurfaceRootBindings(usdGenUsd::Lower(surface), roots, GfMatrix4d(1.0), result, error);
}

bool Reconstructs(UsdGenSurfaceDesc const &surface, VtVec3fArray const &roots,
                  UsdGenSurfaceRootBindingResult const &bindings, std::string *error)
{
    UsdGenSurfaceRootFrameResult frames;
    if (!UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(surface), bindings.rootPrim, bindings.rootUV,
                                          &frames, error)) return false;
    if (frames.valid.size() != roots.size()) return false;
    for (size_t i = 0; i != roots.size(); ++i)
        if (!bindings.valid[i] || !frames.valid[i] || !Near(frames.frames[i].GetRow3(3), roots[i]))
            return false;
    return true;
}

} // namespace

int main()
{
    auto surface = Ngon();
    std::string error;
    UsdGenSurfaceRootBindingResult result;

    // Every exact n-gon vertex must retain a decodable binding. In particular,
    // vertex one is fan zero's u==1 boundary, which cannot be packed as 1.0.
    CHECK(Bind(surface, surface.restPoints, &result, &error));
    CHECK(result.unresolvedCount == 0 && Reconstructs(surface, surface.restPoints, result, &error));
    CHECK(result.rootPrim == VtIntArray({0,0,0,0,0}));
    CHECK(result.rootUV[1][0] < 1.0f && result.rootUV[1][0] > .9999998f &&
          result.rootUV[1][1] == 0.0f);

    // Probe each fan's radial, outer, and diagonal edge. They may have an
    // equivalent adjacent fan representation, but all must reconstruct the
    // exact closest point under the root-frame n-gon convention.
    VtVec3fArray edges;
    for (int fan = 0; fan != 3; ++fan) {
        GfVec3f const a = surface.restPoints[0];
        GfVec3f const b = surface.restPoints[fan + 1];
        GfVec3f const c = surface.restPoints[fan + 2];
        edges.push_back(a * .6f + b * .4f);       // v=0 radial
        edges.push_back(b * .4f + c * .6f);       // u+v=1 outer edge
        edges.push_back(a * .6f + c * .4f);       // u=0 radial
    }
    CHECK(Bind(surface, edges, &result, &error));
    CHECK(result.unresolvedCount == 0 && Reconstructs(surface, edges, result, &error));

    // The fan-zero u==1 fallback is only admitted if its float encoding
    // reconstructs to the documented 1e-5 tolerance. A long first edge makes
    // that impossible; a farther, representable parent face must not be used
    // in its place and falsely advertised as the closest binding.
    auto longEdge = surface;
    longEdge.restPoints = {{0,0,0}, {10000,0,0}, {10001,1,0},
                           {1,3,0}, {-1,1,0}, {10000,100,0},
                           {10001,100,0}, {9999,101,0}};
    longEdge.faceVertexCounts = {5,3};
    longEdge.faceVertexIndices = {0,1,2,3,4, 5,6,7};
    CHECK(Bind(longEdge, {{10000,0,0}}, &result, &error));
    CHECK(result.unresolvedCount == 1 && !result.valid[0] && result.unresolved[0] &&
          result.rootPrim[0] == -1);

    std::puts("surface root binding boundary tests passed");
    return 0;
}
