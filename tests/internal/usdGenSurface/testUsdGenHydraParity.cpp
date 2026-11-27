// testUsdGenHydraParity — gate SI-12 (T1): BuildGraphDescFromHydra agrees
// exactly with BuildGraphDescFromStage on the G1–G4 fixtures (V2-11 item 4:
// node set, edges, structural digests) before the stage path is unplugged.
// RED until the Hydra builder is implemented; the stage builder is the oracle.
//
// Digests need no separate assertion: the Merkle structural digest is a pure
// function of the compared desc content, so field-exact equality implies
// digest equality. Node order compares exactly (S26 namespace tie-break must
// match); params compare as name-sorted sets (engine keys params by name).
//
// Requires the plugin path at runtime (adapters publish the Hydra values):
// run under ctest (ENVIRONMENT carries it) or sourced bin/_env.sh by hand.
// Fixture dir arrives as USDGEN_TEST_FIXTURE_DIR (absolute source path).

#include "utils/pxrInterop.h"
#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "usdGenUsd/lower.h"
#include "usdGenUsd/walk/usdGenGraphDescBuilderStage.h"
#include "usdGenSurface/imagePayload.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/timeCode.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;
using namespace usdGenImaging;

namespace {

int g_failures = 0;
int g_checks = 0;
int g_diffCap = 0;

void Check(bool ok, std::string const &what)
{
    ++g_checks;
    if (ok) {
        return;
    }
    ++g_failures;
    if (g_diffCap < 25) {
        ++g_diffCap;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

template <class T>
void CheckEq(T const &a, T const &b, std::string const &what)
{
    Check(a == b, what);
}

void CheckParams(std::vector<UsdGenParamValue> a,
                 std::vector<UsdGenParamValue> b, std::string const &ctx)
{
    auto byName = [](UsdGenParamValue const &x,
                     UsdGenParamValue const &y) {
        return x.name < y.name;
    };
    std::sort(a.begin(), a.end(), byName);
    std::sort(b.begin(), b.end(), byName);
    Check(a.size() == b.size(), ctx + " param count");
    size_t const n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        std::string const c = ctx + " param[" + a[i].name.GetString() + "]";
        Check(a[i].name == b[i].name, c + " name");
        Check(a[i].value == b[i].value, c + " value");
        Check(a[i].animated == b[i].animated, c + " animated");
    }
}

void CheckNode(UsdGenNodeDesc const &a, UsdGenNodeDesc const &b, size_t i)
{
    std::string const ctx =
        "node[" + std::to_string(i) + " " + a.path.GetString() + "]";
    CheckEq(a.path, b.path, ctx + " path");
    CheckEq(a.type, b.type, ctx + " type");
    CheckEq(a.mode, b.mode, ctx + " mode");
    CheckEq(a.enabled, b.enabled, ctx + " enabled");
    CheckEq(a.seed, b.seed, ctx + " seed");
    CheckEq(a.inputs, b.inputs, ctx + " inputs");
    CheckEq(a.references, b.references, ctx + " references");
    CheckEq(a.curves, b.curves, ctx + " curves");
    CheckEq(a.surfaces, b.surfaces, ctx + " surfaces");
    CheckEq(a.maps, b.maps, ctx + " maps");
    Check(a.mapBindings.size() == b.mapBindings.size(),
          ctx + " map binding count");
    for (size_t binding = 0;
         binding < std::min(a.mapBindings.size(), b.mapBindings.size());
         ++binding) {
        CheckEq(a.mapBindings[binding].map, b.mapBindings[binding].map,
                ctx + " map binding target");
        CheckEq(a.mapBindings[binding].relationship,
                b.mapBindings[binding].relationship,
                ctx + " map binding relationship");
    }
    CheckParams(a.params, b.params, ctx);
    Check(a.ramps.size() == b.ramps.size(), ctx + " ramps count");
}

void CheckCurveSet(UsdGenCurveSetDesc const &a, UsdGenCurveSetDesc const &b,
                   size_t i)
{
    std::string const ctx =
        "curveSet[" + std::to_string(i) + " " + a.path.GetString() + "]";
    CheckEq(a.path, b.path, ctx + " path");
    CheckEq(a.role, b.role, ctx + " role");
    CheckEq(a.curveRole, b.curveRole, ctx + " curveRole");
    CheckEq(a.curveVertexCounts, b.curveVertexCounts, ctx + " counts");
    CheckEq(a.points, b.points, ctx + " points");
    CheckEq(a.rest, b.rest, ctx + " rest");
    CheckEq(a.widths, b.widths, ctx + " widths");
    CheckEq(a.type, b.type, ctx + " type");
    CheckEq(a.basis, b.basis, ctx + " basis");
    CheckEq(a.wrap, b.wrap, ctx + " wrap");
    CheckEq(a.widthsInterpolation, b.widthsInterpolation,
            ctx + " widthsInterpolation");
    CheckEq(a.skinPrim, b.skinPrim, ctx + " skinPrim");
    CheckEq(a.curveId, b.curveId, ctx + " curveId");
    // Authored planes are captured twice, once per builder -- the source
    // curves' own displayColor rides here -- so they are exactly the kind of
    // field that can drift between the two paths.
    Check(a.authoredPlanes.size() == b.authoredPlanes.size(),
          ctx + " authoredPlane count");
    for (size_t p = 0; p < std::min(a.authoredPlanes.size(),
                                    b.authoredPlanes.size()); ++p) {
        std::string const at = ctx + " authoredPlane[" +
            a.authoredPlanes[p].name.GetString() + "]";
        CheckEq(a.authoredPlanes[p].name, b.authoredPlanes[p].name, at + " name");
        Check(a.authoredPlanes[p].domain == b.authoredPlanes[p].domain,
              at + " domain");
        Check(a.authoredPlanes[p].type == b.authoredPlanes[p].type, at + " type");
        Check(a.authoredPlanes[p].arity == b.authoredPlanes[p].arity, at + " arity");
        CheckEq(a.authoredPlanes[p].floatValues, b.authoredPlanes[p].floatValues,
                at + " floatValues");
        CheckEq(a.authoredPlanes[p].intValues, b.authoredPlanes[p].intValues,
                at + " intValues");
    }
    CheckEq(a.skinPrimUv, b.skinPrimUv, ctx + " skinPrimUv");
    CheckEq(a.rootFrame, b.rootFrame, ctx + " rootFrame");
    CheckEq(a.frozenEpoch, b.frozenEpoch, ctx + " frozenEpoch");
    CheckEq(a.curveGeneration, b.curveGeneration, ctx + " curveGeneration");
}

void CheckSurface(UsdGenSurfaceDesc const &a, UsdGenSurfaceDesc const &b,
                  size_t i)
{
    std::string const ctx =
        "surface[" + std::to_string(i) + " " + a.path.GetString() + "]";
    CheckEq(a.path, b.path, ctx + " path");
    CheckEq(a.faceVertexCounts, b.faceVertexCounts, ctx + " counts");
    CheckEq(a.faceVertexIndices, b.faceVertexIndices, ctx + " indices");
    CheckEq(a.subdivisionScheme,b.subdivisionScheme,ctx+" scheme");
    CheckEq(a.orientation,b.orientation,ctx+" orientation");
    CheckEq(a.interpolateBoundary,b.interpolateBoundary,ctx+" boundary");
    CheckEq(a.faceVaryingLinearInterpolation,b.faceVaryingLinearInterpolation,ctx+" face varying");
    CheckEq(a.triangleSubdivisionRule,b.triangleSubdivisionRule,ctx+" triangle rule");
    CheckEq(a.holeIndices,b.holeIndices,ctx+" holes");
    CheckEq(a.creaseIndices,b.creaseIndices,ctx+" crease indices");
    CheckEq(a.creaseLengths,b.creaseLengths,ctx+" crease lengths");
    CheckEq(a.creaseSharpnesses,b.creaseSharpnesses,ctx+" crease sharpness");
    CheckEq(a.cornerIndices,b.cornerIndices,ctx+" corner indices");
    CheckEq(a.cornerSharpnesses,b.cornerSharpnesses,ctx+" corner sharpness");
    CheckEq(a.restPoints, b.restPoints, ctx + " restPoints");
    CheckEq(a.restNormals, b.restNormals, ctx + " restNormals");
    CheckEq(a.restNormalDomain, b.restNormalDomain,
            ctx + " restNormalDomain");
    CheckEq(a.points, b.points, ctx + " points");
    Check(a.samples.size() == b.samples.size(), ctx + " samples count");
    for (size_t s = 0; s < std::min(a.samples.size(), b.samples.size());
         ++s) {
        CheckEq(a.samples[s].time, b.samples[s].time, ctx + " sample time");
        CheckEq(a.samples[s].points, b.samples[s].points,
                ctx + " sample points");
    }
    CheckEq(a.velocities, b.velocities, ctx + " velocities");
    CheckEq(a.uv, b.uv, ctx + " uv");
    CheckEq(a.subsetFaces, b.subsetFaces, ctx + " subsetFaces");
    CheckEq(a.isSubset, b.isSubset, ctx + " isSubset");
    CheckEq(a.worldMatrix, b.worldMatrix, ctx + " worldMatrix");
    CheckEq(a.surfaceGeneration, b.surfaceGeneration,
            ctx + " surfaceGeneration");
}

void CheckMap(UsdGenMapDesc const &a, UsdGenMapDesc const &b, size_t i)
{
    std::string const ctx =
        "map[" + std::to_string(i) + " " + a.path.GetString() + "]";
    CheckEq(a.path, b.path, ctx + " path");
    CheckEq(a.type, b.type, ctx + " type");
    CheckEq(a.resolvedAssetPath, b.resolvedAssetPath, ctx + " asset");
    CheckEq(a.textureGeneration, b.textureGeneration, ctx + " generation");
    Check(static_cast<bool>(a.imagePayload) == static_cast<bool>(b.imagePayload),
          ctx + " decoded payload presence");
    if (a.imagePayload && b.imagePayload) {
        Check(a.imagePayload == b.imagePayload,
              ctx + " shares the immutable decoded COW payload");
        CheckEq(a.imagePayload->Width(), b.imagePayload->Width(), ctx + " width");
        CheckEq(a.imagePayload->Height(), b.imagePayload->Height(), ctx + " height");
        CheckEq(a.imagePayload->Channels(), b.imagePayload->Channels(), ctx + " channels");
    }
    CheckParams(a.params, b.params, ctx);
}

int CheckDesc(UsdGenGraphDesc const &a, UsdGenGraphDesc const &b,
              std::string const &fixture)
{
    int const before = g_failures;
    std::string const ctx = "[" + fixture + "] ";
    CheckEq(a.description, b.description, ctx + "description");
    CheckEq(a.terminal, b.terminal, ctx + "terminal");
    CheckEq(a.time, b.time, ctx + "time");
    Check(a.nodes.size() == b.nodes.size(), ctx + "node count");
    for (size_t i = 0; i < std::min(a.nodes.size(), b.nodes.size()); ++i) {
        CheckNode(a.nodes[i], b.nodes[i], i);
    }
    Check(a.curveSets.size() == b.curveSets.size(), ctx + "curveSet count");
    for (size_t i = 0;
         i < std::min(a.curveSets.size(), b.curveSets.size()); ++i) {
        CheckCurveSet(a.curveSets[i], b.curveSets[i], i);
    }
    Check(a.surfaces.size() == b.surfaces.size(), ctx + "surface count");
    for (size_t i = 0;
         i < std::min(a.surfaces.size(), b.surfaces.size()); ++i) {
        CheckSurface(a.surfaces[i], b.surfaces[i], i);
    }
    Check(a.maps.size() == b.maps.size(), ctx + "map count");
    for (size_t i = 0; i < std::min(a.maps.size(), b.maps.size()); ++i) {
        CheckMap(a.maps[i], b.maps[i], i);
    }
    CheckEq(a.look.rootColor, b.look.rootColor, ctx + "look.rootColor");
    CheckEq(a.look.tipColor, b.look.tipColor, ctx + "look.tipColor");
    CheckEq(a.look.rampColors, b.look.rampColors, ctx + "look.rampColors");
    CheckEq(a.look.rampPositions, b.look.rampPositions,
            ctx + "look.rampPositions");
    CheckEq(a.look.rampInterpolation, b.look.rampInterpolation,
            ctx + "look.rampInterpolation");
    CheckEq(a.look.rampExponent, b.look.rampExponent,
            ctx + "look.rampExponent");
    CheckEq(a.look.bakeMode, b.look.bakeMode, ctx + "look.bakeMode");
    CheckEq(a.look.bakeTarget, b.look.bakeTarget, ctx + "look.bakeTarget");
    CheckEq(a.look.bakePrimvar, b.look.bakePrimvar, ctx + "look.bakePrimvar");
    CheckEq(a.look.hueJitter, b.look.hueJitter, ctx + "look.hueJitter");
    CheckEq(a.look.valueJitter, b.look.valueJitter, ctx + "look.valueJitter");
    CheckEq(a.look.jitterSeed, b.look.jitterSeed, ctx + "look.jitterSeed");
    CheckEq(a.preview.source, b.preview.source, ctx + "preview.source");
    CheckEq(a.preview.colorMap, b.preview.colorMap, ctx + "preview.colorMap");
    CheckEq(a.preview.range, b.preview.range, ctx + "preview.range");
    CheckEq(a.preview.evaluation, b.preview.evaluation, ctx + "preview.evaluation");
    CheckEq(a.preview.shading, b.preview.shading, ctx + "preview.shading");
    CheckEq(a.defaultWidth, b.defaultWidth, ctx + "defaultWidth");
    CheckEq(a.tileTarget, b.tileTarget, ctx + "tileTarget");
    CheckEq(a.curveBasis, b.curveBasis, ctx + "curveBasis");
    CheckEq(a.purpose, b.purpose, ctx + "purpose");
    CheckEq(a.visibility, b.visibility, ctx + "visibility");
    CheckEq(a.materialPath, b.materialPath, ctx + "materialPath");
    CheckEq(a.xformMatrix, b.xformMatrix, ctx + "xformMatrix");
    return g_failures - before;
}

}  // namespace

int
main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    char const *kFixtures[] = {
        "g1_lash4k.usda",
        "g2_brow40k.usda",
        "g3_head100k.usda",
        "g4_fur200k.usda",
    };
    int totalNew = 0;
    int opened = 0;
    for (char const *name : kFixtures) {
        g_diffCap = 0;
        std::string const path =
            std::string(USDGEN_TEST_FIXTURE_DIR) + "/" + name;
        UsdStageRefPtr stage = UsdStage::Open(path);
        if (!stage) {
            std::printf("SKIP fixture missing: %s\n", path.c_str());
            continue;
        }
        ++opened;
        SdfPath descPath;
        for (UsdPrim const &child :
             stage->GetDefaultPrim().GetChildren()) {
            if (child.GetPrimTypeInfo().GetTypeName() ==
                TfToken("UsdGenDescription")) {
                descPath = child.GetPath();
                break;
            }
        }
        if (descPath.IsEmpty()) {
            std::printf("FAIL: no UsdGenDescription in %s\n", name);
            ++g_failures;
            continue;
        }
        UsdGenGraphDescBuildOptions opts;
        opts.time = 0.0;
        UsdGenGraphDesc const fromStage =
            BuildGraphDescFromStage(stage, descPath, opts);
        UsdImagingCreateSceneIndicesInfo info;
        info.stage = stage;
        UsdImagingSceneIndices const sis =
            UsdImagingCreateSceneIndices(info);
        if (!sis.finalSceneIndex) {
            std::printf("FAIL: no final scene index for %s\n", name);
            ++g_failures;
            continue;
        }
        UsdGenGraphDesc const fromHydra =
            BuildGraphDescFromHydra(*sis.finalSceneIndex, descPath, opts);
        int const n = CheckDesc(fromStage, fromHydra, name);
        totalNew += n;
        std::printf("%s: %s (%d new diffs)\n", name,
                    n == 0 ? "PARITY" : "MISMATCH", n);
    }
    {
        auto stage=UsdStage::CreateInMemory();
        Check(stage->GetRootLayer()->ImportFromString(R"USD(#usda 1.0
def Xform "World" {
    def Mesh "Skin" (prepend apiSchemas=["UsdGenRestAPI"]) {
        point3f[] points=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
        int[] faceVertexCounts=[4]
        int[] faceVertexIndices=[0,1,2,3]
        uniform token subdivisionScheme="catmullClark"
        uniform token orientation="leftHanded"
        uniform token interpolateBoundary="edgeOnly"
        uniform token faceVaryingLinearInterpolation="all"
        uniform token triangleSubdivisionRule="smooth"
        int[] holeIndices=[0]
        int[] creaseIndices=[0,1]
        int[] creaseLengths=[2]
        float[] creaseSharpnesses=[2.5]
        int[] cornerIndices=[2]
        float[] cornerSharpnesses=[3]
    }
    def UsdGenDescription "Coat" {
        rel usdGen:surface=</World/Skin>
        def Scope "Ops" {
            def UsdGenScatter "scatter" {
                int usdGen:subdivisionLevel=3
            }
        }
    }
}
)USD"),"load subdivision opinions fixture");
        UsdGenGraphDescBuildOptions opts;opts.time=0;
        auto a=BuildGraphDescFromStage(stage,SdfPath("/World/Coat"),opts);
        UsdImagingCreateSceneIndicesInfo info;info.stage=stage;
        auto sis=UsdImagingCreateSceneIndices(info);
        auto b=BuildGraphDescFromHydra(*sis.finalSceneIndex,SdfPath("/World/Coat"),opts);
        CheckDesc(a,b,"authored subdivision opinions");
        Check(a.surfaces.size()==1 && a.surfaces[0].creaseSharpnesses==VtFloatArray{2.5f},"authored sharpness captured");
        ++opened;
    }
    {
        auto stage = UsdStage::CreateInMemory();
        Check(stage->GetRootLayer()->ImportFromString(R"USD(#usda 1.0
def Xform "World" {
    double3 xformOp:translate = (4, 5, 6)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "Skin" {
        point3f[] points = [(0,0,0),(1,0,0),(1,0,1),(0,0,1)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0,1,2,3]
    }
    def Xform "Moving" {
        double3 xformOp:translate.timeSamples = {
            0: (0, 1.45, 0),
            1: (0, 0.45, 0)
        }
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Mesh "Shield" {
            point3f[] points = [(-0.5,-0.5,-0.5),(0.5,-0.5,-0.5),
                                (0.5,0.5,-0.5),(-0.5,0.5,-0.5)]
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0,1,2,3]
        }
    }
    def UsdGenDescription "Coat" {
        rel usdGen:surface = </World/Skin>
        def Scope "Ops" {
            def UsdGenScatter "scatter" {
                float usdGen:density = 1
            }
            def UsdGenCollide "collide" {
                rel usdGen:colliders = </World/Moving/Shield>
            }
        }
    }
}
)USD"), "load animated transformed collider parity fixture");
        UsdImagingCreateSceneIndicesInfo info; info.stage = stage;
        auto sis = UsdImagingCreateSceneIndices(info);
        usdGen::UsdGenMatrix4d previous(1.0);
        for (double time : {0.0, 1.0}) {
            UsdGenGraphDescBuildOptions opts; opts.time = time;
            auto const a = BuildGraphDescFromStage(stage, SdfPath("/World/Coat"), opts);
            auto const b = BuildGraphDescFromHydra(*sis.finalSceneIndex,
                                                   SdfPath("/World/Coat"), opts);
            CheckDesc(a, b, "animated collider xform");
            auto const found = std::find_if(a.surfaces.begin(), a.surfaces.end(),
                [](UsdGenSurfaceDesc const &s) {
                    return s.path == SdfPath("/World/Moving/Shield");
                });
            Check(found != a.surfaces.end(), "animated collider surface is resolved");
            if (found != a.surfaces.end()) {
                usdGen::UsdGenMatrix4d const relative = found->worldMatrix *
                    a.xformMatrix.GetInverse();
                Check(std::abs(relative.Transform(usdGen::UsdGenVec3d(0))[1] -
                    (time == 0.0 ? 1.45 : 0.45)) < 1e-6,
                    "collider xform samples editable parent translate at graph time");
                if (time != 0.0)
                    Check(previous != found->worldMatrix &&
                          a.surfaces[0].surfaceGeneration != 0,
                          "animated collider matrix changes capture input");
                previous = found->worldMatrix;
            }
        }
        ++opened;
    }
    {
        auto stage = UsdStage::CreateInMemory();
        Check(stage->GetRootLayer()->ImportFromString(R"USD(#usda 1.0
def Xform "World" {
    def BasisCurves "Imported" {
        int[] curveVertexCounts = [3, 3]
        point3f[] points = [(0,0,0),(0,0.5,0),(0,1,0),
                            (2,0,0),(2,0.5,0),(2,1,0)]
        uniform token type = "linear"
        custom uniform int[] primvars:clumpId_10 = [7, 7] (
            interpolation = "uniform")
        custom uniform float3[] primvars:clumpCenter_10 = [(1,0,0),(1,0,0)] (
            interpolation = "uniform")
        custom uniform int2[] primvars:clumpCenterId_10 = [(-1,1073741824),(-1,1073741824)] (
            interpolation = "uniform")
        custom float[] primvars:clumpWeight_10 = [0,0.25,1,0,0.5,1] (
            interpolation = "vertex")
        custom float2[] primvars:extraF2 = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12)] (
            interpolation = "vertex")
        custom uniform int4[] primvars:extraI4 = [(1,2,3,4),(5,6,7,8)] (
            interpolation = "uniform")
        custom float4[] primvars:extraF4 = [(0.1,0.2,0.3,0.4)] (
            interpolation = "constant")
    }
    def UsdGenDescription "Coat" {
        def Scope "Ops" {
            def UsdGenCurveSource "source" {
                rel usdGen:curves = </World/Imported>
            }
        }
    }
}
)USD"), "load typed native clump parity fixture");
        UsdGenGraphDescBuildOptions opts; opts.time = 0;
        UsdImagingCreateSceneIndicesInfo info; info.stage = stage;
        auto sis = UsdImagingCreateSceneIndices(info);
        auto compare = [&](char const *label) {
            auto a = BuildGraphDescFromStage(stage, SdfPath("/World/Coat"), opts);
            auto b = BuildGraphDescFromHydra(*sis.finalSceneIndex,
                                              SdfPath("/World/Coat"), opts);
            CheckDesc(a, b, label);
            return std::make_pair(a, b);
        };
        auto valid = compare("typed native clump planes");
        Check(valid.first.curveSets.size() == 1 &&
              valid.first.curveSets.front().authoredPlanes.size() == 7,
              "typed native clump and generic planes imported");
        if (!valid.first.curveSets.empty() && !valid.second.curveSets.empty()) {
            auto const &planes = valid.first.curveSets.front().authoredPlanes;
            auto const &hydraPlanes = valid.second.curveSets.front().authoredPlanes;
            for (char const *name : {"clumpId_10", "clumpCenter_10",
                                     "clumpCenterId_10", "clumpWeight_10"}) {
                auto const stagePlane = std::find_if(planes.begin(), planes.end(),
                    [&](auto const &plane) { return plane.name == TfToken(name); });
                auto const hydraPlane = std::find_if(hydraPlanes.begin(), hydraPlanes.end(),
                    [&](auto const &plane) { return plane.name == TfToken(name); });
                bool equal = stagePlane != planes.end() && hydraPlane != hydraPlanes.end();
                if (equal)
                    equal = stagePlane->type == hydraPlane->type &&
                            stagePlane->domain == hydraPlane->domain &&
                            stagePlane->arity == hydraPlane->arity &&
                            stagePlane->floatValues == hydraPlane->floatValues &&
                            stagePlane->intValues == hydraPlane->intValues;
                Check(equal, std::string("native clump quartet Stage/Hydra parity: ") + name);
            }
            auto centerId = std::find_if(planes.begin(), planes.end(), [](auto const &plane) {
                return plane.name == TfToken("clumpCenterId_10");
            });
            Check(centerId != planes.end() && centerId->arity == 2 &&
                  centerId->intValues == VtIntArray({-1,1073741824,-1,1073741824}),
                  "clump center identity retains both exact signed words");
        }
        UsdAttribute badWeight = stage->GetPrimAtPath(SdfPath("/World/Imported"))
            .GetAttribute(TfToken("primvars:clumpWeight_10"));
        UsdAttribute center = stage->GetPrimAtPath(SdfPath("/World/Imported"))
            .GetAttribute(TfToken("primvars:clumpCenter_10"));
        VtFloatArray const editedWeights{0.0f, 0.1f, 0.9f, 0.0f, 0.4f, 0.8f};
        Check(badWeight && badWeight.Set(editedWeights) &&
              center && center.Set(VtVec3fArray{GfVec3f(1.25f,0,0),
                                                 GfVec3f(1.25f,0,0)}),
              "edit native clump weight and anchor on live stage");
        auto edited = compare("edited native clump planes on live Hydra index");
        auto const findPlane = [](UsdGenGraphDesc const &desc, TfToken const &name)
            -> UsdGenAuthoredPlaneDesc const * {
            if (desc.curveSets.empty()) return nullptr;
            for (auto const &plane : desc.curveSets.front().authoredPlanes)
                if (plane.name == name) return &plane;
            return nullptr;
        };
        auto const *editedHydraWeight = findPlane(edited.second, TfToken("clumpWeight_10"));
        auto const *editedHydraAnchor = findPlane(edited.second, TfToken("clumpCenter_10"));
        Check(editedHydraWeight && editedHydraWeight->floatValues == editedWeights &&
              editedHydraAnchor && editedHydraAnchor->floatValues ==
                  VtFloatArray({1.25f,0,0,1.25f,0,0}),
              "Hydra notice rebuild reads edited clump weight and anchor");
        Check(badWeight && badWeight.Set(VtFloatArray{0.5f}),
              "author malformed native clump weight");
        auto malformed = compare("malformed native clump planes");
        auto hasMalformed = [](UsdGenGraphDesc const &desc) {
            return std::any_of(desc.validationErrors.begin(),
                               desc.validationErrors.end(),
                [](std::string const &error) {
                    return error.find("malformed native Clump primvar") !=
                        std::string::npos;
                });
        };
        Check(hasMalformed(malformed.first) && hasMalformed(malformed.second),
              "both builders fail closed on malformed native clump plane");
        ++opened;
    }
    if (opened == 0) {
        std::printf("SKIP: G1–G4 fixtures not present under %s "
                    "(generate with usd/fixtures/make_fixtures.py)\n",
                    USDGEN_TEST_FIXTURE_DIR);
        return 77;
    }
    std::printf("hydraParity: %d checks, %d failures\n", g_checks,
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
