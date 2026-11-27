#include "utils/pxrInterop.h"
#include "usdGenSurface/surfaceRootBindings.h"
#include "usdGenUsd/lower.h"
#include "usdGenSurface/surfaceRootFrames.h"
#include "usdGenCurves/curveRootCapture.h"

#include <cmath>
#include <cstdio>
#include <limits>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c); return 1; } } while (false)
static bool Near(double a, double b) { return std::abs(a-b) < 1.e-5; }

// Most numerical fixtures express roots in the same local space as the mesh.
static bool UsdGenBuildRestSurfaceRootBindings(UsdGenSurfaceDesc const& surface,
    usdGen::UsdGenVec3fArray const& roots, UsdGenSurfaceRootBindingResult* result, std::string* error) {
    return usdGen::UsdGenBuildRestSurfaceRootBindings(usdGenUsd::Lower(surface),roots,surface.worldMatrix,result,error);
}

static UsdGenSurfaceDesc OneParentFace(UsdGenSurfaceDesc const& surface, size_t face)
{
    UsdGenSurfaceDesc single = surface;
    size_t begin = 0;
    for (size_t i = 0; i != face; ++i) begin += size_t(surface.faceVertexCounts[i]);
    size_t const count = size_t(surface.faceVertexCounts[face]);
    single.faceVertexCounts = {surface.faceVertexCounts[face]};
    single.faceVertexIndices.resize(count);
    for (size_t i = 0; i != count; ++i)
        single.faceVertexIndices[i] = surface.faceVertexIndices[begin + i];
    return single;
}

static double Distance2(usdGen::UsdGenVec3d const& a, usdGen::UsdGenVec3f const& b)
{
    usdGen::UsdGenVec3d const d = a - usdGen::UsdGenVec3d(b[0], b[1], b[2]);
    return d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
}

// A deliberately slow parent-face exhaustive oracle.  Fixtures keep <=8
// faces, so the production k=8 candidate pass considers this same set.
static bool MatchesExhaustiveParents(UsdGenSurfaceDesc const& surface,
                                     usdGen::UsdGenVec3fArray const& roots, std::string* error)
{
    UsdGenSurfaceRootBindingResult combined;
    if (!UsdGenBuildRestSurfaceRootBindings(surface, roots, &combined, error)) return false;
    UsdGenSurfaceRootFrameResult combinedFrames;
    if (!UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(surface), combined.rootPrim, combined.rootUV,
                                          &combinedFrames, error)) return false;
    for (size_t root = 0; root != roots.size(); ++root) {
        double bestDistance2 = std::numeric_limits<double>::infinity();
        int bestFace = -1;
        usdGen::UsdGenVec2f bestUv(0.0f);
        for (size_t face = 0; face != surface.faceVertexCounts.size(); ++face) {
            auto single = OneParentFace(surface, face);
            UsdGenSurfaceRootBindingResult one;
            UsdGenSurfaceRootFrameResult oneFrames;
            if (!UsdGenBuildRestSurfaceRootBindings(single, {roots[root]}, &one, error) ||
                !one.valid[0] ||
                !UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(single), one.rootPrim, one.rootUV,
                                                  &oneFrames, error) || !oneFrames.valid[0])
                continue;
            double const d2 = Distance2(oneFrames.frames[0].GetRow3(3), roots[root]);
            if (d2 < bestDistance2 || (d2 == bestDistance2 && int(face) < bestFace)) {
                bestDistance2 = d2;
                bestFace = int(face);
                bestUv = one.rootUV[0];
            }
        }
        // This is a robust exact/clear-separation fixture oracle, not a claim
        // about arbitrary near ties: per-face reconstruction is float-encoded
        // while production selects with a raw-double candidate distance.
        if (bestFace < 0 || !combined.valid[root] || combined.rootPrim[root] != bestFace ||
            combined.rootUV[root] != bestUv ||
            Distance2(combinedFrames.frames[root].GetRow3(3), roots[root]) >
                bestDistance2 + 1.e-10) return false;
    }
    return true;
}

int main() {
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Scalp");
    surface.restPoints = {{0,0,0},{1,0,0},{0,1,0}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {3}; surface.faceVertexIndices = {0,1,2};
    usdGen::UsdGenVec3fArray roots = {{.2f,.3f,2}, {-2,0,1}, {2,2,0}};
    auto retainedRoots = roots;
    UsdGenSurfaceRootBindingResult result;
    std::string error;
    CHECK(UsdGenBuildRestSurfaceRootBindings(surface,roots,&result,&error));
    CHECK(result.unresolvedCount == 0 && result.rootPrim == VtIntArray({0,0,0}));
    CHECK(Near(result.rootUV[0][0],.2) && Near(result.rootUV[0][1],.3));
    CHECK(Near(result.rootUV[1][0],0) && Near(result.rootUV[1][1],0));
    CHECK(Near(result.rootUV[2][0],.5) && Near(result.rootUV[2][1],.5));
    CHECK(roots == retainedRoots && roots.cdata() == retainedRoots.cdata());
    UsdGenSurfaceRootFrameResult frames;
    CHECK(UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(surface),result.rootPrim,result.rootUV,&frames,&error));
    CHECK(Near(frames.frames[0][3][0],.2) && Near(frames.frames[0][3][1],.3) &&
          Near(frames.frames[0][3][2],0));

    auto prior = result;
    auto reject = [&](UsdGenSurfaceDesc const& s, usdGen::UsdGenVec3fArray const& p) {
        return !UsdGenBuildRestSurfaceRootBindings(s,p,&result,&error) && !error.empty() &&
            result.rootPrim == prior.rootPrim && result.rootUV == prior.rootUV;
    };
    auto invalid = surface; invalid.restFromCurrentPoints = true;
    CHECK(reject(invalid,roots));
    invalid = surface; invalid.faceVertexIndices[0] = 99;
    CHECK(reject(invalid,roots));
    invalid = surface; invalid.faceVertexCounts[0] = 2;
    CHECK(reject(invalid,roots));
    invalid = surface; invalid.restPoints[0][0] = std::numeric_limits<float>::infinity();
    CHECK(reject(invalid,roots));
    CHECK(reject(surface,{{std::numeric_limits<float>::quiet_NaN(),0,0}}));

    // Equal-distance candidates choose the lower parent face ID, independent
    // of Scatter subsets. No source or surface transform is baked into UVs.
    auto duplicate = surface;
    duplicate.faceVertexCounts = {3,3}; duplicate.faceVertexIndices = {0,1,2,0,1,2};
    duplicate.subsetFaces = {1}; duplicate.worldMatrix.SetTranslate(usdGen::UsdGenVec3d(10,20,30));
    CHECK(UsdGenBuildRestSurfaceRootBindings(duplicate,roots,&result,&error));
    CHECK(result.rootPrim == VtIntArray({0,0,0}));
    CHECK(MatchesExhaustiveParents(duplicate, roots, &error));
    // A differently transformed source is converted to surface-local before
    // projection. Translation is not silently ignored or applied twice.
    usdGen::UsdGenMatrix4d sourceTransform(1.0);
    sourceTransform.SetTranslate(usdGen::UsdGenVec3d(10.2,20.3,32));
    CHECK(usdGen::UsdGenBuildRestSurfaceRootBindings(usdGenUsd::Lower(duplicate),{{0,0,0}},
        sourceTransform,&result,&error));
    CHECK(Near(result.rootUV[0][0],.2) && Near(result.rootUV[0][1],.3));

    // Face zero's centroid is closer to the origin, but face one contains
    // it. This catches any accidental "nearest centroid is the answer"
    // shortcut. The third face is a warped bilinear quad; all three are
    // within k=8, so compare production selection to the exhaustive oracle.
    UsdGenSurfaceDesc adversarial;
    adversarial.path = SdfPath("/Adversarial");
    adversarial.restPoints = {{.1f,-1,0},{.1f,.5f,.8660254f},{.1f,.5f,-.8660254f},
                              {0,0,0},{.3f,0,0},{0,.3f,0},
                              {2,0,0},{3,0,0},{3,1,1},{2,1,0}};
    adversarial.points = adversarial.restPoints;
    adversarial.faceVertexCounts = {3,3,4};
    adversarial.faceVertexIndices = {0,1,2, 3,4,5, 6,7,8,9};
    usdGen::UsdGenVec3fArray adversarialRoots = {{0,0,0}, {2.25f,.75f,.1875f},
                                     {-2,-2,1}, {3,1,1}};
    CHECK(UsdGenBuildRestSurfaceRootBindings(adversarial,adversarialRoots,&result,&error));
    CHECK(result.rootPrim[0] == 1 && result.rootPrim[1] == 2);
    CHECK(MatchesExhaustiveParents(adversarial, adversarialRoots, &error));
    auto translatedAdversarial = adversarial;
    translatedAdversarial.worldMatrix.SetTranslate(usdGen::UsdGenVec3d(1.e6, -2.e6, 3.e6));
    CHECK(MatchesExhaustiveParents(translatedAdversarial, adversarialRoots, &error));

    // Coincident bilinear quads exercise exact geometry ties after the AABB
    // lower-bound pass; face zero must win, never a later equal candidate.
    auto coincidentQuads = surface;
    coincidentQuads.restPoints = {{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
    coincidentQuads.points = coincidentQuads.restPoints;
    coincidentQuads.faceVertexCounts = {4,4};
    coincidentQuads.faceVertexIndices = {0,1,2,3, 0,1,2,3};
    CHECK(UsdGenBuildRestSurfaceRootBindings(coincidentQuads,{{.4f,.6f,0}},&result,&error));
    CHECK(result.rootPrim == VtIntArray({0}) && MatchesExhaustiveParents(
        coincidentQuads, {{.4f,.6f,0}}, &error));

    // A warped quad is not its diagonal triangulation. Bind actual bilinear
    // points and reconstruct them through the same frame convention.
    auto quad = surface;
    quad.restPoints = {{0,0,0},{1,0,0},{1,1,1},{0,1,0}};
    quad.faceVertexCounts = {4}; quad.faceVertexIndices = {0,1,2,3};
    usdGen::UsdGenVec3fArray quadRoots;
    for (int i=0; i!=7; ++i) {
        float const u = float(i+1)/8, v = float(7-i)/8;
        quadRoots.push_back(usdGen::UsdGenVec3f(u,v,u*v));
    }
    CHECK(UsdGenBuildRestSurfaceRootBindings(quad,quadRoots,&result,&error));
    CHECK(result.unresolvedCount == 0 && result.rootPrim.size() == quadRoots.size());
    CHECK(UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(quad),result.rootPrim,result.rootUV,&frames,&error));
    for (size_t i=0; i!=quadRoots.size(); ++i) {
        CHECK(result.valid[i]);
        for (int axis=0; axis!=3; ++axis)
            CHECK(Near(frames.frames[i][3][axis],quadRoots[i][axis]));
    }
    CHECK(UsdGenBuildRestSurfaceRootBindings(quad,{{-2,-2,-1},{2,2,2}},&result,&error));
    CHECK(result.unresolvedCount == 0 && Near(result.rootUV[0][0],0) &&
          Near(result.rootUV[0][1],0) && Near(result.rootUV[1][0],1) &&
          Near(result.rootUV[1][1],1));
    usdGen::UsdGenVec3fArray quadBoundary = {{0,0,0},{1,0,0},{1,1,1},{0,1,0},
                                  {.5f,0,0},{1,.5f,.5f},{.5f,1,.5f},{0,.5f,0}};
    CHECK(UsdGenBuildRestSurfaceRootBindings(quad,quadBoundary,&result,&error));
    CHECK(MatchesExhaustiveParents(quad, quadBoundary, &error));

    auto ngon = surface;
    ngon.restPoints = {{0,0,0},{2,0,0},{3,1,0},{1,3,0},{-1,1,0}};
    ngon.faceVertexCounts = {5}; ngon.faceVertexIndices = {0,1,2,3,4};
    usdGen::UsdGenVec3fArray fanRoots;
    for (int fan=0; fan!=3; ++fan)
        fanRoots.push_back(ngon.restPoints[fan+1]*.2f + ngon.restPoints[fan+2]*.3f);
    CHECK(UsdGenBuildRestSurfaceRootBindings(ngon,fanRoots,&result,&error));
    CHECK(result.unresolvedCount == 0);
    CHECK(UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(ngon),result.rootPrim,result.rootUV,&frames,&error));
    for (size_t i=0; i!=fanRoots.size(); ++i) {
        CHECK(Near(result.rootUV[i][0],double(i)+.2));
        for (int axis=0; axis!=3; ++axis)
            CHECK(Near(frames.frames[i][3][axis],fanRoots[i][axis]));
    }
    auto collapsed = surface; collapsed.restPoints.assign(3,usdGen::UsdGenVec3f(0,0,0));
    CHECK(UsdGenBuildRestSurfaceRootBindings(collapsed,roots,&result,&error));
    CHECK(result.unresolvedCount == 3 && result.rootPrim == VtIntArray(3,-1));
    for (size_t i=0; i!=3; ++i) CHECK(!result.valid[i] && result.unresolved[i]);
    CHECK(UsdGenBuildRestSurfaceRootBindings(surface,{},&result,&error));
    CHECK(result.rootPrim.empty() && result.rootUV.empty() && result.valid.empty() &&
          result.unresolved.empty() && result.unresolvedCount == 0);
    {
        UsdGenCurveSetDesc curves;
        curves.curveVertexCounts = {2,2};
        curves.points = {{.8f,.1f,0},{.8f,.1f,1},{.7f,.2f,0},{.7f,.2f,1}};
        curves.rest = {{.1f,.2f,0},{.1f,.2f,1},{.3f,.2f,0},{.3f,.2f,1}};
        curves.skinPrim = {0,999}; curves.skinPrimUv = {{.6f,.1f},{.9f,.9f}};
        UsdGenCurveRootCaptureResult captured;
        CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(curves),usdGenUsd::Addr(usdGenUsd::View(surface)),TfToken("onError"),&captured,&error));
        CHECK(captured.rebound == 1 && captured.dropped == 0 &&
              captured.rootPrim == VtIntArray({0,0}) &&
              captured.rootUV[0] == curves.skinPrimUv[0] &&
              Near(captured.rootUV[1][0],.3) && Near(captured.rootUV[1][1],.2));
        auto old = captured;
        CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(curves),usdGenUsd::Addr(usdGenUsd::View(surface)),TfToken("always"),&captured,&error));
        CHECK(captured.rebound == 2 && captured.dropped == 0 &&
              Near(captured.rootUV[0][0],.1) && Near(captured.rootUV[0][1],.2));
        CHECK(old.rootUV[0] == usdGen::UsdGenVec2f(.6f,.1f) && curves.skinPrim[1] == 999);
        CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(curves),usdGenUsd::Addr(usdGenUsd::View(surface)),TfToken("never"),&captured,&error));
        CHECK(captured.dropped == 1 && captured.valid[0] && !captured.valid[1] &&
              captured.rootPrim[1] == -1 && captured.dropMask[1]);
        // Failed always-rebind must not retain the previous valid flags.
        CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(curves),usdGenUsd::Addr(usdGenUsd::View(collapsed)),TfToken("always"),&captured,&error));
        CHECK(captured.dropped == 2 && captured.rebound == 0);
        curves.rootFrame.assign(2,usdGen::UsdGenMatrix4d(1.0));
        curves.skinPrim.clear(); curves.skinPrimUv.clear();
        CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(curves),usdGenUsd::Addr(usdGenUsd::View(surface)),TfToken("never"),&captured,&error));
        CHECK(captured.dropped == 2 && !captured.valid[0] && !captured.valid[1]);
        CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(curves),usdGenUsd::Addr(usdGenUsd::View(surface)),TfToken("onError"),&captured,&error));
        CHECK(captured.rebound == 2 && captured.dropped == 0);
        auto transformed = surface;
        transformed.restPoints = {{0,0,0},{0,0,1},{1,-1,0}};
        transformed.worldMatrix.SetScale(usdGen::UsdGenVec3d(2,1,1));
        curves.curveVertexCounts = {2};
        curves.points = {{0,0,0},{0,0,1}}; curves.rest = curves.points;
        curves.rootFrame.clear(); curves.skinPrim = {0}; curves.skinPrimUv = {{.2f,.3f}};
        CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(curves),usdGenUsd::Addr(usdGenUsd::View(transformed)),TfToken("never"),&captured,&error));
        CHECK(captured.valid[0] && UsdGenValidateAuthoredRootFrame(captured.frames[0]));
        CHECK(Near(captured.frames[0][2][0],1/std::sqrt(5.0)) &&
              Near(captured.frames[0][2][1],2/std::sqrt(5.0)) &&
              Near(captured.frames[0][3][0],.6) && Near(captured.frames[0][3][1],-.3) &&
              Near(captured.frames[0][3][2],.2));
        auto empty = UsdGenCurveSetDesc{};
        CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(empty),nullptr,TfToken("always"),&captured,&error));
        CHECK(captured.valid.empty() && captured.dropped == 0);
    }
    std::puts("surface root binding tests passed");
    return 0;
}
