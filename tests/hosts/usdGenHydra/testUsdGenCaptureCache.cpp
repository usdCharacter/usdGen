// Exact read-count, value, topology and diagnostic oracles for node capture.
#include "utils/pxrInterop.h"
#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "usdGenUsd/lower.h"
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
          corrected.desc.curveSets[0].role == usdGen::UsdGenLane::Chunked,
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
    std::printf("testUsdGenCaptureCache: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
