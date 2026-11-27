// Live C3 rest adapter regression.  The adapter must read Default-time curve
// values dynamically, preserve whether primvars:rest was authored, and dirty
// its dedicated root when either source changes.
#include "utils/pxrInterop.h"
#include "usdGenImaging/curveApiSchemaAdapter.h"

#include "pxr/pxr.h"
#include "pxr/base/vt/array.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/sdf/valueTypeName.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/timeCode.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/primvar.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <cstdio>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int Fail(char const *message)
{
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

class Notice final : public HdSceneIndexObserver
{
public:
    bool seen = false;

    void Reset() { seen = false; }
    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &e) override
    { for (auto const& entry : e) seen |= entry.primPath == SdfPath("/curves"); }
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &) override {}
    void PrimsDirtied(HdSceneIndexBase const &, DirtiedPrimEntries const &e) override
    {
        for (auto const& entry : e)
            seen |= entry.primPath == SdfPath("/curves") &&
                entry.dirtyLocators.Intersects(HdDataSourceLocator(TfToken("usdGenCurveRest")));
    }
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &) override {}
};

HdContainerDataSourceHandle Root(HdSceneIndexBase &sceneIndex,
                                 SdfPath const &path)
{
    HdSceneIndexPrim const prim = sceneIndex.GetPrim(path);
    if (!prim.dataSource) return nullptr;
    return HdContainerDataSource::Cast(
        prim.dataSource->Get(TfToken("usdGenCurveRest")));
}

HdContainerDataSourceHandle Cage(HdContainerDataSourceHandle const &root)
{
    return root ? HdContainerDataSource::Cast(root->Get(TfToken("surfaceCage")))
                : nullptr;
}

VtValue Leaf(HdContainerDataSourceHandle const &root, char const *name)
{
    if (!root) return VtValue();
    HdSampledDataSourceHandle const source =
        HdSampledDataSource::Cast(root->Get(TfToken(name)));
    return source ? source->GetValue(0.0f) : VtValue();
}

}  // namespace

int main()
{
    UsdStageRefPtr const stage = UsdStage::CreateInMemory("curve-rest-live");
    SdfPath const path("/curves");
    UsdGeomBasisCurves const curves = UsdGeomBasisCurves::Define(stage, path);
    VtVec3fArray restDefault{GfVec3f(0, 0, 0), GfVec3f(1, 0, 0)};
    VtVec3fArray pointsAt24{GfVec3f(0, 0, 5), GfVec3f(1, 0, 5)};
    VtVec3fArray editedDefault{GfVec3f(0, 0, 2), GfVec3f(1, 0, 2)};
    VtVec3fArray authoredRest{GfVec3f(0, 0, -1), GfVec3f(1, 0, -1)};
    VtVec3fArray editedRest{GfVec3f(0, 0, -2), GfVec3f(1, 0, -2)};
    curves.CreateCurveVertexCountsAttr(VtValue(VtIntArray{2}));
    curves.CreatePointsAttr(VtValue(restDefault));
    curves.GetPointsAttr().Set(VtValue(pointsAt24), UsdTimeCode(24.0));
    UsdAttribute const ownerIds = curves.GetPrim().CreateAttribute(
        TfToken("usdGen:surfaceCage:ownerIds"),
        SdfValueTypeNames->IntArray, false);
    UsdAttribute const normalizedT = curves.GetPrim().CreateAttribute(
        TfToken("usdGen:surfaceCage:normalizedT"),
        SdfValueTypeNames->FloatArray, false);
    ownerIds.Set(VtIntArray{7, 8});
    normalizedT.Set(VtFloatArray{0.0f, 0.37f, 1.0f});
    curves.GetPrim().AddAppliedSchema(TfToken("UsdGenCurveAPI"));

    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices const indices = UsdImagingCreateSceneIndices(info);
    Notice notice;
    HdSceneIndexObserverPtr const observer = TfCreateWeakPtr(&notice);
    indices.finalSceneIndex->AddObserver(observer);

    HdContainerDataSourceHandle const firstRoot =
        Root(*indices.finalSceneIndex, path);
    if (!firstRoot) return Fail("curve rest root was not published");
    VtValue const firstPoints = Leaf(firstRoot, "points");
    if (!firstPoints.IsHolding<VtVec3fArray>() ||
        firstPoints.UncheckedGet<VtVec3fArray>() != restDefault)
        return Fail("missing rest did not use Default points");
    VtValue const firstProvenance = Leaf(firstRoot, "hasAuthoredRest");
    if (!firstProvenance.IsHolding<bool>() || firstProvenance.UncheckedGet<bool>())
        return Fail("missing rest provenance was not false");
    HdContainerDataSourceHandle const firstCage = Cage(firstRoot);
    if (!firstCage ||
        !Leaf(firstCage, "ownerIds").IsHolding<VtIntArray>() ||
        Leaf(firstCage, "ownerIds").UncheckedGet<VtIntArray>() !=
            VtIntArray{7, 8} ||
        !Leaf(firstCage, "normalizedT").IsHolding<VtFloatArray>() ||
        Leaf(firstCage, "normalizedT").UncheckedGet<VtFloatArray>() !=
            VtFloatArray{0.0f, 0.37f, 1.0f})
        return Fail("surface cage fields were not exposed through CurveAPI");

    // A Default points edit must invalidate the root and update the already
    // acquired dynamic source; the time-sample at 24 is intentionally ignored.
    notice.Reset();
    curves.GetPointsAttr().Set(VtValue(editedDefault));
    indices.stageSceneIndex->ApplyPendingUpdates();
    if (!notice.seen) return Fail("Default points edit emitted no notice");
    VtValue const editedPoints = Leaf(firstRoot, "points");
    if (!editedPoints.IsHolding<VtVec3fArray>() ||
        editedPoints.UncheckedGet<VtVec3fArray>() != editedDefault)
        return Fail("dynamic fallback source retained stale points");

    // An authored rest primvar changes both the points value and provenance.
    notice.Reset();
    UsdGeomPrimvar const rest = UsdGeomPrimvarsAPI(stage->GetPrimAtPath(path))
        .CreatePrimvar(TfToken("rest"), SdfValueTypeNames->Point3fArray,
                       TfToken("vertex"));
    rest.Set(VtValue(authoredRest));
    indices.stageSceneIndex->ApplyPendingUpdates();
    if (!notice.seen) return Fail("authoring rest did not invalidate its root");
    VtValue const authoredPoints = Leaf(firstRoot, "points");
    VtValue const authoredProvenance = Leaf(firstRoot, "hasAuthoredRest");
    if (!authoredPoints.IsHolding<VtVec3fArray>() ||
        authoredPoints.UncheckedGet<VtVec3fArray>() != authoredRest ||
        !authoredProvenance.IsHolding<bool>() ||
        !authoredProvenance.UncheckedGet<bool>())
        return Fail("authored rest was not live or provenance was wrong");

    notice.Reset();
    rest.Set(VtValue(editedRest));
    indices.stageSceneIndex->ApplyPendingUpdates();
    if (!notice.seen) return Fail("rest edit emitted no notice");
    VtValue const rereadRest = Leaf(firstRoot, "points");
    if (!rereadRest.IsHolding<VtVec3fArray>() ||
        rereadRest.UncheckedGet<VtVec3fArray>() != editedRest)
        return Fail("dynamic authored rest source retained stale points");

    notice.Reset();
    rest.Set(VtValue(VtVec3fArray{}));
    indices.stageSceneIndex->ApplyPendingUpdates();
    if (!notice.seen) return Fail("empty rest edit did not invalidate its root");
    auto emptyRest = Leaf(firstRoot, "points");
    auto emptyProvenance = Leaf(firstRoot, "hasAuthoredRest");
    if (!emptyRest.IsHolding<VtVec3fArray>() || !emptyRest.UncheckedGet<VtVec3fArray>().empty() ||
        !emptyProvenance.IsHolding<bool>() || !emptyProvenance.UncheckedGet<bool>())
        return Fail("explicit empty rest was replaced by fallback points");

    notice.Reset();
    ownerIds.Set(VtIntArray{9, 10});
    indices.stageSceneIndex->ApplyPendingUpdates();
    if (!notice.seen) return Fail("surface cage edit did not invalidate its root");
    if (Leaf(Cage(firstRoot), "ownerIds").UncheckedGet<VtIntArray>() !=
        VtIntArray{9, 10})
        return Fail("surface cage source retained stale owner ids");

    indices.finalSceneIndex->RemoveObserver(observer);
    std::printf("testUsdGenCurveRestAdapter: PASS\n");
    return 0;
}
