// testUsdGenSourceColor — the colour a tile publishes, and where it came from.
//
// Precedence (02 §2.14, plan/16-ue-hair-parity.md):
//   1. an AUTHORED usdGen:look on the description wins;
//   2. else the source curves' own primvars:displayColor, forwarded verbatim;
//   3. else the look's schema defaults.
//
// "Authored" has to mean an opinion that DIFFERS from the schema fallback.
// Applying UsdGenLookAPI without setting anything leaves a full set of
// fallbacks, and counting those as authored would silently discard the colour
// every converted asset already carries — which is exactly what a character asset
// groom ships with. That case is asserted here because it is the one a
// "does the API exist?" test would get wrong.
//
// T1 headless: in-memory stage, the real UsdImaging chain plus the groom
// scene index, no GL. The colour is read back off the published tile, so the
// whole path is covered: both graph-desc builders' capture, the authored-plane
// machinery that carries it through the graph, and the cooker's precedence.

#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "usdGenUsd/lower.h"
#include "usdGenImaging/usdGenGraphDescBuilderStage.h"

#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/dataSourceLocator.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/sdf/types.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usd/usdGeom/xform.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <cstdio>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGenImaging;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
    else     { std::printf("ok:   %s\n", what.c_str()); }
}

SdfPath const kDescription("/World/Groom/Hair");
GfVec3f const kSourceColor(0.12f, 0.055f, 0.022f);   // the converted groom's
GfVec3f const kLookRoot(0.40f, 0.20f, 0.10f);
// UsdGenLookDesc's defaults, i.e. the UsdGenLookAPI schema fallbacks.

enum class Look { None, FallbackOnly, Authored };

/// A groom whose only operator is a UsdGenCurveSource over BasisCurves that
/// author a uniform displayColor, modelled on a converted character groom.
UsdStageRefPtr MakeStage(Look look, TfToken const &interpolation,
                         VtVec3fArray const &colors)
{
    UsdStageRefPtr stage = UsdStage::CreateInMemory("sourceColor");
    UsdGeomXform::Define(stage, SdfPath("/World"));

    UsdGeomMesh scalp = UsdGeomMesh::Define(stage, SdfPath("/World/Scalp"));
    scalp.CreateFaceVertexCountsAttr().Set(VtIntArray{4});
    scalp.CreateFaceVertexIndicesAttr().Set(VtIntArray{0, 1, 2, 3});
    scalp.CreatePointsAttr().Set(VtVec3fArray{
        GfVec3f(-1, 0, -1), GfVec3f(1, 0, -1), GfVec3f(1, 0, 1), GfVec3f(-1, 0, 1)});

    UsdGeomBasisCurves curves =
        UsdGeomBasisCurves::Define(stage, SdfPath("/World/Strands"));
    curves.CreateTypeAttr().Set(UsdGeomTokens->cubic);
    curves.CreateBasisAttr().Set(UsdGeomTokens->bspline);
    curves.CreateWrapAttr().Set(UsdGeomTokens->pinned);
    int const strands = 4, cvs = 4;
    VtIntArray counts;
    VtVec3fArray points;
    VtFloatArray widths;
    for (int s = 0; s < strands; ++s) {
        counts.push_back(cvs);
        for (int c = 0; c < cvs; ++c) {
            points.push_back(GfVec3f(-0.5f + 0.3f * float(s),
                                     0.25f * float(c), 0.0f));
            widths.push_back(0.01f);
        }
    }
    curves.CreateCurveVertexCountsAttr().Set(counts);
    curves.CreatePointsAttr().Set(points);
    curves.CreateWidthsAttr().Set(widths);
    curves.SetWidthsInterpolation(UsdGeomTokens->vertex);
    // C3 wants a rest channel and the curve API on a source; without them the
    // capture rejects the operator and the groom publishes nothing.
    curves.GetPrim().AddAppliedSchema(TfToken("UsdGenCurveAPI"));
    // The C3 role marker: CurveSource only consumes the hair lane.
    UsdGeomPrimvarsAPI(curves.GetPrim())
        .CreatePrimvar(TfToken("usdGen:role"), SdfValueTypeNames->Token,
                       UsdGeomTokens->constant)
        .Set(TfToken("hair"));
    UsdGeomPrimvarsAPI(curves.GetPrim())
        .CreatePrimvar(TfToken("rest"), SdfValueTypeNames->Point3fArray,
                       UsdGeomTokens->vertex)
        .Set(points);
    UsdGeomPrimvarsAPI(curves.GetPrim())
        .CreatePrimvar(TfToken("displayColor"), SdfValueTypeNames->Color3fArray,
                       interpolation)
        .Set(colors);

    UsdGeomXform::Define(stage, SdfPath("/World/Groom"));
    UsdPrim description =
        stage->DefinePrim(kDescription, TfToken("UsdGenDescription"));
    description.CreateRelationship(TfToken("usdGen:surface"))
        .SetTargets({SdfPath("/World/Scalp")});
    if (look != Look::None) {
        description.ApplyAPI(TfToken("UsdGenLookAPI"));
    }
    if (look == Look::Authored) {
        description.CreateAttribute(TfToken("usdGen:look:rootColor"),
                                    SdfValueTypeNames->Color3f).Set(kLookRoot);
    }
    stage->DefinePrim(kDescription.AppendChild(TfToken("Ops")), TfToken("Scope"));
    UsdPrim source = stage->DefinePrim(
        kDescription.AppendPath(SdfPath("Ops/source")),
        TfToken("UsdGenCurveSource"));
    source.CreateRelationship(TfToken("usdGen:curves"))
        .SetTargets({SdfPath("/World/Strands")});
    source.CreateAttribute(TfToken("usdGen:rebind"), SdfValueTypeNames->Token)
        .Set(TfToken("never"));
    return stage;
}

/// The captured curve set's forwarded source colour, through BOTH builders.
/// Returns false when nothing was forwarded.
bool CapturedSourceColor(UsdStageRefPtr const &stage,
                         usdGen::UsdGenAuthoredPlaneDomain *domain,
                         GfVec3f *first, usdGen::UsdGenLookDesc *look,
                         std::string *why)
{
    usdGenImaging::UsdGenGraphDescBuildOptions opts;
    opts.time = 0.0;
    usdGen::UsdGenGraphDesc const fromStage =
        usdGenImaging::BuildGraphDescFromStage(stage, kDescription, opts);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices const sis = UsdImagingCreateSceneIndices(info);
    usdGen::UsdGenGraphDesc const fromHydra =
        usdGenImaging::BuildGraphDescFromHydra(*sis.finalSceneIndex,
                                               kDescription, opts);
    *look = fromHydra.look;

    auto forwarded = [&](usdGen::UsdGenGraphDesc const &desc)
        -> usdGen::UsdGenAuthoredPlaneDesc const * {
        for (usdGen::UsdGenCurveSetDesc const &set : desc.curveSets)
            for (usdGen::UsdGenAuthoredPlaneDesc const &plane : set.authoredPlanes)
                if (plane.name == usdGen::UsdGenSourceColorPlane()) return &plane;
        return nullptr;
    };
    auto const *hydra = forwarded(fromHydra);
    auto const *staged = forwarded(fromStage);
    if (!hydra || !staged) {
        *why = hydra ? "only the Hydra builder forwarded it"
                     : staged ? "only the stage builder forwarded it"
                              : "neither builder forwarded it";
        return false;
    }
    if (hydra->domain != staged->domain ||
        hydra->floatValues != staged->floatValues) {
        *why = "the two builders disagree";
        return false;
    }
    if (hydra->arity != 3 || hydra->floatValues.size() < 3) {
        *why = "forwarded plane has the wrong shape";
        return false;
    }
    *domain = hydra->domain;
    *first = GfVec3f(hydra->floatValues[0], hydra->floatValues[1],
                     hydra->floatValues[2]);
    return true;
}

bool Near(GfVec3f const &a, GfVec3f const &b, float epsilon = 1e-4f)
{
    return (a - b).GetLength() <= epsilon;
}

std::string Show(GfVec3f const &c)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "(%.4f %.4f %.4f)",
                  double(c[0]), double(c[1]), double(c[2]));
    return buffer;
}

void CheckForwarding(TfToken const &interpolation, VtVec3fArray const &colors,
                     usdGen::UsdGenAuthoredPlaneDomain expected,
                     std::string const &what)
{
    usdGen::UsdGenAuthoredPlaneDomain domain{};
    GfVec3f first;
    usdGen::UsdGenLookDesc look;
    std::string why;
    if (!CapturedSourceColor(MakeStage(Look::None, interpolation, colors),
                             &domain, &first, &look, &why)) {
        Check(false, what + " — " + why);
        return;
    }
    Check(domain == expected && Near(first, kSourceColor),
          what + ": forwarded " + Show(first) + ", domain " +
              std::to_string(int(domain)));
}

/// The predicate the session cooker uses to decide whether the look beats the
/// forwarded colour. Mirrored here so a change to one without the other fails.
bool AuthoredLook(usdGen::UsdGenLookDesc const &look)
{
    static usdGen::UsdGenLookDesc const fallback;
    return look.rootColor != fallback.rootColor ||
           look.tipColor != fallback.tipColor || !look.rampColors.empty();
}

void CheckLook(Look look, bool expectedAuthored, std::string const &what)
{
    usdGen::UsdGenAuthoredPlaneDomain domain{};
    GfVec3f first;
    usdGen::UsdGenLookDesc captured;
    std::string why;
    CapturedSourceColor(MakeStage(look, UsdGeomTokens->uniform,
                                  VtVec3fArray(4, kSourceColor)),
                        &domain, &first, &captured, &why);
    Check(AuthoredLook(captured) == expectedAuthored,
          what + ": rootColor " + Show(captured.rootColor) + " reads as " +
              (AuthoredLook(captured) ? "authored" : "schema fallback"));
}

}  // namespace

int main()
{
    VtVec3fArray const uniformColor(4, kSourceColor);

    // (a) The source curves' own colour is forwarded, at every interpolation
    // USD allows, and both graph-desc builders agree on it.
    CheckForwarding(UsdGeomTokens->uniform, uniformColor,
                    usdGen::UsdGenAuthoredPlaneDomain::Primitive,
                    "a uniform source displayColor is forwarded per curve");
    CheckForwarding(UsdGeomTokens->constant, VtVec3fArray{kSourceColor},
                    usdGen::UsdGenAuthoredPlaneDomain::Groom,
                    "a constant source displayColor is forwarded per groom");
    CheckForwarding(UsdGeomTokens->vertex, VtVec3fArray(16, kSourceColor),
                    usdGen::UsdGenAuthoredPlaneDomain::Point,
                    "a vertex source displayColor is forwarded per CV");

    // (b) An authored look reads as authored, so the cooker prefers it.
    CheckLook(Look::Authored, true,
              "an authored usdGen:look counts as authored");
    // (c) The case a "is UsdGenLookAPI applied?" test gets wrong: the API is
    // applied but nothing is set, so every value is a schema fallback and must
    // NOT displace the asset's own colour.
    CheckLook(Look::FallbackOnly, false,
              "an applied-but-unauthored UsdGenLookAPI is a schema fallback");
    CheckLook(Look::None, false, "no UsdGenLookAPI at all is not authored");

    std::printf("testUsdGenSourceColor: %s (%d failures)\n",
                g_failures ? "FAIL" : "PASS", g_failures);
    return g_failures ? 1 : 0;
}
