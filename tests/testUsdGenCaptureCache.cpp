// Exact read-count, value, topology and diagnostic oracles for node capture.
#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/retainedSceneIndex.h"
#include "pxr/base/gf/vec3f.h"
#include <cstdio>
#include <map>
#include <vector>
PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGenImaging;

namespace {
int failures = 0;
void Check(bool value, char const* message) {
    std::printf("%s: %s\n", value ? "ok" : "FAIL", message);
    if (!value) ++failures;
}
HdContainerDataSourceHandle Flat(std::initializer_list<std::pair<TfToken, HdDataSourceBaseHandle>> fields) {
    TfTokenVector names;
    std::vector<HdDataSourceBaseHandle> values;
    for (auto const& field : fields) { names.push_back(field.first); values.push_back(field.second); }
    return HdRetainedContainerDataSource::New(names.size(), names.data(), values.data());
}
template<class T> HdDataSourceBaseHandle Value(T value) {
    return HdRetainedTypedSampledDataSource<T>::New(value);
}
class CountedValue final : public HdSampledDataSource {
public:
    HD_DECLARE_DATASOURCE(CountedValue);
    explicit CountedValue(float value) : value(value) {}
    float value;
    unsigned reads = 0;
    VtValue GetValue(Time) override { ++reads; return VtValue(value); }
    bool GetContributingSampleTimesForInterval(Time, Time, std::vector<Time>*) override { return false; }
};
class Input final : public HdSceneIndexBase {
public:
    HdRetainedSceneIndexRefPtr data = HdRetainedSceneIndex::New();
    mutable std::map<SdfPath, unsigned> reads;
    HdSceneIndexPrim GetPrim(SdfPath const& path) const override { ++reads[path]; return data->GetPrim(path); }
    SdfPathVector GetChildPrimPaths(SdfPath const& path) const override { return data->GetChildPrimPaths(path); }
};
float Param(usdGen::UsdGenNodeDesc const& node, char const* name) {
    for (auto const& param : node.params)
        if (param.name == TfToken(name) && param.value.IsHolding<float>())
            return param.value.UncheckedGet<float>();
    return -1;
}
}
int main() {
    auto input = TfCreateRefPtr(new Input);
    const SdfPath root("/desc"), source("/desc/source"), width("/desc/width"), curves("/curves");
    auto unused = CountedValue::New(3.0f), widthValue = CountedValue::New(1.0f);
    auto description = [&](SdfPathVector order, SdfPath surface) {
        input->data->AddPrims({{root, TfToken("UsdGenDescription"), Flat({
            {TfToken("operatorOrder"), Value(order)},
            {TfToken("surface"), Value(SdfPathVector{surface})}})}});
    };
    auto sourceData = [&](bool bad, bool guide) {
        input->data->AddPrims({{source, TfToken("UsdGenCurveSource"), Flat({
            {TfToken("unused"), unused},
            {TfToken("input"), Value(SdfPathVector{SdfPath("/hostileLegacyEdge")})},
            {TfToken("curves"), Value(SdfPathVector{curves})},
            {TfToken("guides"), Value(guide ? SdfPathVector{curves} : SdfPathVector{})},
            {TfToken("__usdGenValidationErrors"), Value(bad ? VtStringArray{"bad source"} : VtStringArray{})}})}});
    };
    description({source, width}, SdfPath("/mesh"));
    sourceData(false, true);
    input->data->AddPrims({
        {width, TfToken("UsdGenWidth"), Flat({{TfToken("width"), widthValue}})},
        {SdfPath("/mesh"), TfToken("mesh"), HdRetainedContainerDataSource::New()},
        {SdfPath("/mesh2"), TfToken("mesh"), HdRetainedContainerDataSource::New()},
        {curves, TfToken("basisCurves"), HdRetainedContainerDataSource::New()}});
    auto reset = [&] { input->reads.clear(); unused->reads = widthValue->reads = 0; };

    auto first = CaptureGraphDescFromHydra(*input, root);
    Check(first.cache && first.desc.nodes.size() == 2, "first capture creates two-node cache");
    if (first.desc.nodes.size() != 2) return 1;
    Check(input->reads[source] == 1 && input->reads[width] == 1 && unused->reads > 0,
          "first topology pull reads every node and unused mapped property");
    Check(first.desc.nodes[0].inputs.empty() && first.desc.nodes[1].inputs == SdfPathVector{source} &&
          first.desc.terminal == width, "hierarchy overrides hostile legacy edge");
    Check(Param(first.desc.nodes[0], "unused") == 3 && Param(first.desc.nodes[1], "width") == 1,
          "first capture owns exact parameter values");

    UsdGenGraphDescBuildOptions options;
    options.reuseNodes = true;
    options.previousCache = first.cache;
    options.dirtyPrimPaths = {width};
    unused->value = 9; // no source dirty yet: the captured source must remain 3
    widthValue->value = 2;
    reset();
    auto updated = CaptureGraphDescFromHydra(*input, root, options);
    Check(input->reads[source] == 0 && input->reads[width] == 1 &&
          unused->reads == 0 && widthValue->reads > 0, "only dirty node reads upstream values");
    Check(Param(updated.desc.nodes[0], "unused") == 3 && Param(updated.desc.nodes[1], "width") == 2,
          "dirty width updates while unchanged node retains its captured value");
    Check(Param(first.desc.nodes[1], "width") == 1, "older descriptor remains immutable");

    options.previousCache = updated.cache;
    options.dirtyPrimPaths = {root};
    reset();
    auto parent = CaptureGraphDescFromHydra(*input, root, options);
    Check(input->reads[source] == 1 && input->reads[width] == 1 && unused->reads > 0 &&
          Param(parent.desc.nodes[0], "unused") == 9, "parent dirty rereads every operator");

    options.previousCache = parent.cache;
    options.dirtyPrimPaths.clear();
    description({source, width}, SdfPath("/mesh2"));
    reset();
    auto inherited = CaptureGraphDescFromHydra(*input, root, options);
    Check(input->reads[source] == 0 && input->reads[width] == 0 &&
          inherited.desc.nodes[0].surfaces == SdfPathVector{SdfPath("/mesh2")} &&
          inherited.desc.nodes[1].surfaces == SdfPathVector{SdfPath("/mesh2")},
          "surface inheritance reassembles from current description, not cached inheritance");

    sourceData(true, true);
    auto invalid = CaptureGraphDescFromHydra(*input, root);
    Check(invalid.desc.validationErrors == std::vector<std::string>{"bad source"},
          "per-node validation is captured exactly");
    sourceData(false, false);
    options.previousCache = invalid.cache;
    options.dirtyPrimPaths = {width};
    auto stillInvalid = CaptureGraphDescFromHydra(*input, root, options);
    Check(stillInvalid.desc.validationErrors == std::vector<std::string>{"bad source"},
          "unrelated edits cannot erase cached validation errors");
    options.previousCache = stillInvalid.cache;
    options.dirtyPrimPaths = {source};
    auto corrected = CaptureGraphDescFromHydra(*input, root, options);
    Check(corrected.desc.validationErrors.empty() && !invalid.desc.validationErrors.empty(),
          "corrected dirty node clears its error, preserving old descriptor");
    Check(corrected.desc.curveSets.size() == 1 &&
          corrected.desc.curveSets[0].role == usdGen::UsdGenRole::Curves,
          "removing last guide role recomputes shared curve pool role");

    options.previousCache = corrected.cache;
    options.dirtyPrimPaths.clear();
    description({width, source}, SdfPath("/mesh2"));
    reset();
    auto reordered = CaptureGraphDescFromHydra(*input, root, options);
    Check(input->reads[source] == 1 && input->reads[width] == 1 &&
          reordered.desc.nodes[0].path == width && reordered.desc.nodes[0].inputs.empty() &&
          reordered.desc.nodes[1].inputs == SdfPathVector{width} && reordered.desc.terminal == source,
          "changed composed order forces all-node pull and fresh hierarchy inputs");

    options.previousCache = reordered.cache;
    options.time = 1;
    unused->value = 11;
    reset();
    auto timed = CaptureGraphDescFromHydra(*input, root, options);
    Check(input->reads[source] == 1 && input->reads[width] == 1 &&
          Param(timed.desc.nodes[1], "unused") == 11, "changed sample time disables node reuse");
    reset();
    options.previousCache = timed.cache;
    options.reuseNodes = false;
    auto explicitFull = CaptureGraphDescFromHydra(*input, root, options);
    Check(input->reads[source] == 1 && input->reads[width] == 1 && explicitFull.cache,
          "explicit topology/frame rebuild never reuses nodes");

    // A foreign/retained Hydra source can carry malformed rest leaves even
    // though authored UsdGeomMesh normals are schema typed.  The builder
    // must preserve that distinction as Invalid rather than treating it as
    // the valid geometric-normal fallback.
    auto const rawRoot = SdfPath("/normalDesc");
    auto const rawSource = SdfPath("/normalDesc/source");
    auto const rawMesh = SdfPath("/normalMesh");
    VtVec3fArray const rawPoints{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    VtIntArray const rawCounts{3};
    VtIntArray const rawIndices{0, 1, 2};
    auto const captureMalformedRest = [&](HdContainerDataSourceHandle const &rest) {
        auto raw = TfCreateRefPtr(new Input);
        HdContainerDataSourceHandle const topology = Flat({
            {TfToken("faceVertexCounts"), Value(rawCounts)},
            {TfToken("faceVertexIndices"), Value(rawIndices)}});
        HdContainerDataSourceHandle const meshSchema = Flat({
            {TfToken("topology"), topology}});
        HdContainerDataSourceHandle const restSchema = Flat({
            {TfToken("rest"), rest}});
        HdContainerDataSourceHandle const meshData = Flat({
            {TfToken("points"), Value(rawPoints)},
            {TfToken("mesh"), meshSchema},
            {TfToken("usdGen"), restSchema}});
        raw->data->AddPrims({
            {rawRoot, TfToken("UsdGenDescription"), Flat({
                {TfToken("operatorOrder"), Value(SdfPathVector{rawSource})},
                {TfToken("surface"), Value(SdfPathVector{rawMesh})}})},
            {rawSource, TfToken("UsdGenCurveSource"),
             HdRetainedContainerDataSource::New()},
            {rawMesh, TfToken("mesh"), meshData}});
        return CaptureGraphDescFromHydra(*raw, rawRoot).desc;
    };
    auto const malformedNormal = captureMalformedRest(Flat({
        {TfToken("points"), Value(rawPoints)},
        {TfToken("faceVertexCounts"), Value(rawCounts)},
        {TfToken("faceVertexIndices"), Value(rawIndices)},
        {TfToken("normals"), Value(VtFloatArray{1.0f})},
        {TfToken("normalsInterpolation"), Value(TfToken("vertex"))}}));
    Check(malformedNormal.surfaces.size() == 1 &&
              malformedNormal.surfaces[0].restNormalDomain ==
                  usdGen::UsdGenSurfaceNormalDomain::Invalid &&
              !malformedNormal.validationErrors.empty(),
          "wrong-typed nonempty rest normals remain Invalid");
    auto const missingInterpolation = captureMalformedRest(Flat({
        {TfToken("points"), Value(rawPoints)},
        {TfToken("faceVertexCounts"), Value(rawCounts)},
        {TfToken("faceVertexIndices"), Value(rawIndices)},
        {TfToken("normals"), Value(VtVec3fArray{{0, 0, 1}, {0, 0, 1},
                                                   {0, 0, 1}})}}));
    Check(missingInterpolation.surfaces.size() == 1 &&
              missingInterpolation.surfaces[0].restNormalDomain ==
                  usdGen::UsdGenSurfaceNormalDomain::Invalid &&
              !missingInterpolation.validationErrors.empty(),
          "nonempty rest normals without interpolation remain Invalid");

    // ---- geometry reuse across captures ----------------------------------
    // A pooled desc survives while its prim stays clean: no upstream read,
    // identical values, generations and diagnostics. A dirty on the prim
    // (or an ancestor) re-reads; a changed sample time disables reuse.
    auto const geoRoot = SdfPath("/geoDesc");
    auto const geoSource = SdfPath("/geoDesc/source");
    auto const geoMesh = SdfPath("/geoMesh");
    auto const geoCurves = SdfPath("/geoCurves");
    VtVec3fArray const triPoints{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    VtVec3fArray const triPointsMoved{{0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
    auto geoMeshData = [&](VtVec3fArray const &pts) {
        return Flat({
            {TfToken("points"), Value(pts)},
            {TfToken("mesh"), Flat({
                {TfToken("topology"), Flat({
                    {TfToken("faceVertexCounts"), Value(VtIntArray{3})},
                    {TfToken("faceVertexIndices"), Value(VtIntArray{0, 1, 2})}})}})},
            // Mismatched rest topology: the error below proves diagnostics
            // ride the cache entry, not the rebuild.
            {TfToken("usdGen"), Flat({
                {TfToken("rest"), Flat({
                    {TfToken("points"), Value(pts)},
                    {TfToken("faceVertexCounts"), Value(VtIntArray{4})},
                    {TfToken("faceVertexIndices"), Value(VtIntArray{0, 1, 2})}})}})}});
    };
    auto geoCurvesData = [&]() {
        return Flat({
            {TfToken("points"), Value(VtVec3fArray{{0, 0, 0}, {0, 1, 0}})},
            {TfToken("basisCurves"), Flat({
                {TfToken("topology"), Flat({
                    {TfToken("curveVertexCounts"), Value(VtIntArray{2})}})}})}});
    };
    auto geo = TfCreateRefPtr(new Input);
    auto geoSourceData = [&](bool guide) {
        geo->data->AddPrims({{geoSource, TfToken("UsdGenCurveSource"), Flat({
            {TfToken("curves"), Value(SdfPathVector{geoCurves})},
            {TfToken("guides"), Value(guide ? SdfPathVector{geoCurves}
                                            : SdfPathVector{})}})}});
    };
    geo->data->AddPrims({
        {geoRoot, TfToken("UsdGenDescription"), Flat({
            {TfToken("operatorOrder"), Value(SdfPathVector{geoSource})},
            {TfToken("surface"), Value(SdfPathVector{geoMesh})}})},
        {geoMesh, TfToken("mesh"), geoMeshData(triPoints)},
        {geoCurves, TfToken("basisCurves"), geoCurvesData()}});
    geoSourceData(true);
    auto geoBase = CaptureGraphDescFromHydra(*geo, geoRoot);
    Check(geoBase.desc.surfaces.size() == 1 &&
              geoBase.desc.curveSets.size() == 1 &&
              !geoBase.desc.validationErrors.empty() &&
              geoBase.desc.surfaces[0].surfaceGeneration != 0 &&
              geoBase.desc.curveSets[0].curveGeneration != 0 &&
              geoBase.desc.curveSets[0].role == usdGen::UsdGenRole::Reference,
          "geometry capture pools a surface and a reference curve set");
    UsdGenGraphDescBuildOptions gopts;
    gopts.reuseGeometry = true;
    gopts.previousCache = geoBase.cache;
    geo->reads.clear();
    auto geoReused = CaptureGraphDescFromHydra(*geo, geoRoot, gopts);
    Check(geo->reads[geoMesh] == 0 && geo->reads[geoCurves] == 0,
          "clean surface and curve prims are not re-read");
    Check(geoReused.desc.surfaces[0].points == triPoints &&
              geoReused.desc.surfaces[0].surfaceGeneration ==
                  geoBase.desc.surfaces[0].surfaceGeneration &&
              geoReused.desc.curveSets[0].curveGeneration ==
                  geoBase.desc.curveSets[0].curveGeneration &&
              geoReused.desc.validationErrors ==
                  geoBase.desc.validationErrors,
          "reused desc equals the fresh capture: values, generations, errors");
    Check(geoReused.reusedInputs.size() == 2,
          "reused capture reports its unread inputs for dependencies");

    geo->data->AddPrims({{geoMesh, TfToken("mesh"), geoMeshData(triPointsMoved)}});
    gopts.previousCache = geoReused.cache;
    gopts.dirtyPrimPaths = {geoMesh};
    geo->reads.clear();
    auto geoMoved = CaptureGraphDescFromHydra(*geo, geoRoot, gopts);
    Check(geo->reads[geoMesh] > 0 && geo->reads[geoCurves] == 0,
          "only the dirtied surface prim re-reads");
    Check(geoMoved.desc.surfaces[0].points == triPointsMoved &&
              geoMoved.desc.surfaces[0].surfaceGeneration !=
                  geoBase.desc.surfaces[0].surfaceGeneration &&
              geoMoved.desc.curveSets[0].curveGeneration ==
                  geoBase.desc.curveSets[0].curveGeneration &&
              geoMoved.reusedInputs == SdfPathVector{geoCurves},
          "dirtied surface refreshes its values and generation; curves keep theirs");

    gopts.previousCache = geoMoved.cache;
    gopts.dirtyPrimPaths = {SdfPath::AbsoluteRootPath()};
    geo->reads.clear();
    auto geoAncestor = CaptureGraphDescFromHydra(*geo, geoRoot, gopts);
    Check(geo->reads[geoMesh] > 0 && geo->reads[geoCurves] > 0,
          "ancestor dirty re-reads pooled geometry");
    Check(geoAncestor.desc.surfaces[0].points == triPointsMoved &&
              geoAncestor.desc.surfaces[0].surfaceGeneration ==
                  geoMoved.desc.surfaces[0].surfaceGeneration,
          "ancestor re-read restores identical values and generation");

    gopts.previousCache = geoAncestor.cache;
    gopts.dirtyPrimPaths.clear();
    gopts.time = 1;
    geo->reads.clear();
    auto geoTimed = CaptureGraphDescFromHydra(*geo, geoRoot, gopts);
    Check(geo->reads[geoMesh] > 0 && geo->reads[geoCurves] > 0,
          "changed sample time disables geometry reuse");
    gopts.time = 0;

    // Dropping the guide relationship re-claims the pool in the Curves
    // lane: the role term moves, so the carried generation must go and
    // the prim re-reads even with no dirty.
    geoSourceData(false);
    gopts.previousCache = geoTimed.cache;
    geo->reads.clear();
    auto geoDerolled = CaptureGraphDescFromHydra(*geo, geoRoot, gopts);
    Check(geoDerolled.desc.curveSets[0].role == usdGen::UsdGenRole::Curves &&
              geo->reads[geoCurves] > 0,
          "lane change re-reads the curve set without a dirty");
    gopts.previousCache = geoDerolled.cache;
    geo->reads.clear();
    auto geoSettled = CaptureGraphDescFromHydra(*geo, geoRoot, gopts);
    Check(geo->reads[geoCurves] == 0 &&
              geoSettled.desc.curveSets[0].curveGeneration ==
                  geoDerolled.desc.curveSets[0].curveGeneration,
          "re-claimed lane reuses once its role is stable");

    // ---- invariant sub-hash memo across deform cooks ----------------------
    // Moving points with identical topology/rest shares keeps the invariant
    // sub-hash while the generation moves (the memo hits); a topology edit
    // moves both (the memo misses). Shared arrays across captures model
    // Hydra's static topology shares under a deform. The memo is
    // value-identical to a fresh hash either way, so these pin the
    // sensitivity contract a stale memo would break, not the memo itself.
    VtIntArray const memoCounts{3}, memoIndices{0, 1, 2};
    VtVec3fArray const memoRest{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    auto memoMeshData = [&](VtVec3fArray const &pts, VtIntArray const &counts,
                            VtIntArray const &indices) {
        return Flat({
            {TfToken("points"), Value(pts)},
            {TfToken("mesh"), Flat({
                {TfToken("topology"), Flat({
                    {TfToken("faceVertexCounts"), Value(counts)},
                    {TfToken("faceVertexIndices"), Value(indices)}})}})},
            {TfToken("usdGen"), Flat({
                {TfToken("rest"), Flat({
                    {TfToken("points"), Value(memoRest)},
                    {TfToken("faceVertexCounts"), Value(counts)},
                    {TfToken("faceVertexIndices"), Value(indices)}})}})}});
    };
    UsdGenGraphDescBuildOptions mopts;
    mopts.reuseGeometry = true;
    geo->data->AddPrims(
        {{geoMesh, TfToken("mesh"), memoMeshData(triPoints, memoCounts, memoIndices)}});
    mopts.previousCache = geoSettled.cache;
    mopts.dirtyPrimPaths = {geoMesh};
    auto memoBase = CaptureGraphDescFromHydra(*geo, geoRoot, mopts);
    Check(memoBase.desc.surfaces.size() == 1 &&
              memoBase.desc.surfaces[0].points == triPoints &&
              memoBase.desc.surfaces[0].surfaceGeneration != 0,
          "memo baseline captures values and a generation");
    // Points move, shares hold: generation moves, invariant sub-hash holds.
    geo->data->AddPrims(
        {{geoMesh, TfToken("mesh"), memoMeshData(triPointsMoved, memoCounts, memoIndices)}});
    mopts.previousCache = memoBase.cache;
    auto memoMoved = CaptureGraphDescFromHydra(*geo, geoRoot, mopts);
    Check(memoMoved.desc.surfaces[0].points == triPointsMoved &&
              memoMoved.desc.surfaces[0].surfaceGeneration !=
                  memoBase.desc.surfaces[0].surfaceGeneration &&
              memoMoved.desc.surfaces[0].surfaceInvariantGeneration ==
                  memoBase.desc.surfaces[0].surfaceInvariantGeneration,
          "points-only move refreshes the generation, holds the invariant sub-hash");
    // The memo is value-identical to a fresh hash: the same data captured
    // without a cache must produce both generations exactly. A stale memo
    // (any dropped key term) fails here.
    auto memoMovedFresh = CaptureGraphDescFromHydra(*geo, geoRoot);
    Check(memoMovedFresh.desc.surfaces[0].surfaceGeneration ==
                  memoMoved.desc.surfaces[0].surfaceGeneration &&
              memoMovedFresh.desc.surfaces[0].surfaceInvariantGeneration ==
                  memoMoved.desc.surfaces[0].surfaceInvariantGeneration,
          "memoized capture equals the fresh hash after a points-only move");
    // Topology edit, indices only (counts shares hold): both move. Single
    // term, so a memo key that dropped the indices term would stale-hit.
    VtIntArray const memoIndices2{0, 2, 1};
    geo->data->AddPrims(
        {{geoMesh, TfToken("mesh"), memoMeshData(triPointsMoved, memoCounts, memoIndices2)}});
    mopts.previousCache = memoMoved.cache;
    auto memoRetopo = CaptureGraphDescFromHydra(*geo, geoRoot, mopts);
    Check(memoRetopo.desc.surfaces[0].surfaceGeneration !=
                  memoMoved.desc.surfaces[0].surfaceGeneration &&
              memoRetopo.desc.surfaces[0].surfaceInvariantGeneration !=
                  memoMoved.desc.surfaces[0].surfaceInvariantGeneration,
          "topology edit refreshes the generation and the invariant sub-hash");
    auto memoRetopoFresh = CaptureGraphDescFromHydra(*geo, geoRoot);
    Check(memoRetopoFresh.desc.surfaces[0].surfaceGeneration ==
                  memoRetopo.desc.surfaces[0].surfaceGeneration &&
              memoRetopoFresh.desc.surfaces[0].surfaceInvariantGeneration ==
                  memoRetopo.desc.surfaces[0].surfaceInvariantGeneration,
          "memoized capture equals the fresh hash after a topology edit");
    std::printf("testUsdGenCaptureCache: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
