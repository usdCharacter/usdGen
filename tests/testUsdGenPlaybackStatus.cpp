// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenImaging/usdGenBrushApi.h"
#include "usdGenImaging/testHook.h"
#include "usdGen/opRegistry.h"
#include "pxr/base/js/json.h"
#include "pxr/imaging/hd/retainedSceneIndex.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/systemMessages.h"
#include "pxr/imaging/hd/sceneGlobalsSchema.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
#include <stdexcept>
PXR_NAMESPACE_USING_DIRECTIVE
namespace {
int failures = 0;
char const* empty = "{\"renderer\":\"HdStormRendererPlugin\",\"roots\":[]}";
JsObject Status(char const* selector = empty) {
    return JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(selector, 0)).GetJsObject();
}
void Check(bool value, char const* message) {
    std::printf("%s: %s\n", value ? "ok" : "FAIL", message);
    if (!value) ++failures;
}
bool State(char const* expected, char const* selector = empty) {
    return Status(selector).at("state").GetString() == expected;
}
std::atomic<bool> holdCapture{false}, captureEntered{false}, releaseCapture{true};
std::atomic<uint64_t> captureEpoch{0};
std::atomic<bool> holdTile{false}, tileEntered{false}, releaseTile{true};
class TemporalOp final : public usdGen::UsdGenOp {
public:
    TfToken Type() const override { return TfToken("UsdGenPlaybackTestTemporal"); }
    bool ReadsTime() const override { return true; }
    TfSpan<const TfToken> TopologyParameters() const override { return {}; }
    TfSpan<const TfToken> ValueParameters() const override { return {}; }
    TfSpan<const TfToken> ReferenceInputs() const override {
        static TfToken const guides("guides");
        return {&guides, 1};
    }
    bool Bind(usdGen::UsdGenParamView const&, usdGen::UsdGenDiagnostics*) override { return true; }
    usdGen::UsdGenEpoch CaptureDigest(usdGen::UsdGenCaptureContext const&) const override { return {1, captureEpoch.load()}; }
    bool Capture(usdGen::UsdGenCaptureContext const&, usdGen::UsdGenCurveBuffer const&,
        usdGen::UsdGenCapture*, usdGen::UsdGenDiagnostics*) override {
        if (holdCapture.exchange(false)) {
            captureEntered.store(true);
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!releaseCapture.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            return releaseCapture.load();
        }
        return true;
    }
    void Evaluate(usdGen::UsdGenEvalContext const&, usdGen::UsdGenCapture const&, usdGen::UsdGenChunkView* view) const override {
        if (holdTile.load() && view->desc->tile == 1) {
            tileEntered.store(true);
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!releaseTile.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            if (!releaseTile.load()) throw std::runtime_error("playback tile gate timed out");
        }
    }
};
HdContainerDataSourceHandle EmptyGraph(bool temporal = false) {
    TfToken names[] = {TfToken("operatorOrder"), TfToken("surface"), TfToken("tileTarget")};
    HdDataSourceBaseHandle values[] = {
        HdRetainedTypedSampledDataSource<SdfPathVector>::New(temporal ?
            SdfPathVector{SdfPath("/empty/op"), SdfPath("/empty/grow"), SdfPath("/empty/temporal")} :
            SdfPathVector{SdfPath("/empty/op"), SdfPath("/empty/grow")}),
        HdRetainedTypedSampledDataSource<SdfPathVector>::New({SdfPath("/surface")}),
        HdRetainedTypedSampledDataSource<int>::New(64)};
    return HdRetainedContainerDataSource::New(3, names, values);
}
class PointsValue final : public HdTypedSampledDataSource<VtVec3fArray> {
public:
    HD_DECLARE_DATASOURCE(PointsValue);
    VtVec3fArray value{{0,0,0},{1,0,0},{2,0,0},{0,0,1},{1,0,1},{2,0,1},{0,0,2},{1,0,2},{2,0,2}};
    VtVec3fArray GetTypedValue(Time) override { return value; }
    VtValue GetValue(Time) override { return VtValue(value); }
    bool GetContributingSampleTimesForInterval(Time, Time, std::vector<Time>*) override { return false; }
};
auto surfacePoints = PointsValue::New();
HdContainerDataSourceHandle Surface() {
    TfToken topologyNames[] = {TfToken("faceVertexCounts"), TfToken("faceVertexIndices")};
    HdDataSourceBaseHandle topologyValues[] = {
        HdRetainedTypedSampledDataSource<VtIntArray>::New(VtIntArray{4,4,4,4}),
        HdRetainedTypedSampledDataSource<VtIntArray>::New(VtIntArray{0,1,4,3,1,2,5,4,3,4,7,6,4,5,8,7})};
    TfToken restNames[] = {TfToken("faceVertexCounts"), TfToken("faceVertexIndices"), TfToken("points")};
    HdDataSourceBaseHandle restValues[] = {topologyValues[0], topologyValues[1],
        HdRetainedTypedSampledDataSource<VtVec3fArray>::New(surfacePoints->value)};
    TfToken names[] = {TfToken("mesh"), TfToken("points"), TfToken("usdGen")};
    HdDataSourceBaseHandle values[] = {
        HdRetainedContainerDataSource::New(TfToken("topology"), HdRetainedContainerDataSource::New(2, topologyNames, topologyValues)),
        surfacePoints,
        HdRetainedContainerDataSource::New(TfToken("rest"), HdRetainedContainerDataSource::New(3, restNames, restValues))};
    return HdRetainedContainerDataSource::New(3, names, values);
}
class FrameValue final : public HdTypedSampledDataSource<double> {
public:
    HD_DECLARE_DATASOURCE(FrameValue);
    double value = 0;
    double GetTypedValue(Time) override { return value; }
    VtValue GetValue(Time) override { return VtValue(value); }
    bool GetContributingSampleTimesForInterval(Time, Time, std::vector<Time>*) override { return false; }
};
class MapValue final : public HdTypedSampledDataSource<float> {
public:
    HD_DECLARE_DATASOURCE(MapValue);
    float value = 1;
    float GetTypedValue(Time) override { return value; }
    VtValue GetValue(Time) override { return VtValue(value); }
    bool GetContributingSampleTimesForInterval(Time, Time, std::vector<Time>*) override { return false; }
};
HdContainerDataSourceHandle Frame(HdDataSourceBaseHandle value) {
    return HdRetainedContainerDataSource::New(TfToken("sceneGlobals"),
        HdRetainedContainerDataSource::New(TfToken("currentFrame"),
            value));
}
HdContainerDataSourceHandle Guide() {
    TfToken const names[] = {TfToken("ownerIds"), TfToken("ownerDensities"),
        TfToken("ownerSeeds"), TfToken("ownerCvCounts"), TfToken("ownerEdgeBias"),
        TfToken("ownerLengthProfileOffsets"), TfToken("ownerLengthProfile"),
        TfToken("normalizedT"), TfToken("triangles"), TfToken("triangleOwnerIndices"),
        TfToken("triangleRootCharts"), TfToken("ownerChartCentroids"), TfToken("ownerChartMeanRadii")};
    auto ints = HdRetainedTypedSampledDataSource<VtIntArray>::New({});
    auto floats = HdRetainedTypedSampledDataSource<VtFloatArray>::New({});
    auto vec2s = HdRetainedTypedSampledDataSource<VtVec2fArray>::New({});
    HdDataSourceBaseHandle values[] = {ints, floats, ints, ints, floats, ints, vec2s, floats,
        HdRetainedTypedSampledDataSource<VtVec3iArray>::New({}), ints, vec2s, vec2s, floats};
    return HdRetainedContainerDataSource::New(
        TfToken("surfaceCage"), HdRetainedContainerDataSource::New(13, names, values),
        TfToken("points"), HdRetainedTypedSampledDataSource<VtVec3fArray>::New({{0,0,0},{0,1,0}}),
        TfToken("basisCurves"), HdRetainedContainerDataSource::New(TfToken("topology"),
            HdRetainedContainerDataSource::New(TfToken("curveVertexCounts"),
                HdRetainedTypedSampledDataSource<VtIntArray>::New({2}))));
}
}
int main() {
    usdGen::UsdGenOpRegistry::Get().Register(TfToken("UsdGenPlaybackTestTemporal"), [] { return std::make_unique<TemporalOp>(); });
    Check(State("unavailable"), "missing viewer index fails closed");
    auto input = HdRetainedSceneIndex::New();
    auto sceneTime = FrameValue::New();
    input->AddPrims({{SdfPath::AbsoluteRootPath(), TfToken(), Frame(sceneTime)}});
    auto index = UsdGenGroomSceneIndex::New(input, 8901);
    UsdGenImagingTestHook::groomSceneServiceCommandBarrier();
    Check(State("ready"), "completed empty scene population is ready");
    auto signature = Status().at("signature").GetString();
    Check(!signature.empty(), "ready publication has a signature");
    index->SystemMessage(HdSystemMessageTokens->asyncAllow, nullptr);
    input->AddPrims({{SdfPath("/ordinary"), TfToken("Scope"), HdRetainedContainerDataSource::New()}});
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    Check(State("pending"), "owner completion cannot stand in for Hydra publication");
    index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
    Check(State("ready") && signature != Status().at("signature").GetString(),
        "frontend publication restores readiness with changed signature");
    auto second = UsdGenGroomSceneIndex::New(HdRetainedSceneIndex::New(), 8902);
    Check(!State("ready"), "second index cannot be ignored before registry publication");
    UsdGenImagingTestHook::groomSceneServiceCommandBarrier();
    Check(State("ambiguous"), "multiple viewer indexes fail closed");
    second.Reset();
    UsdGenGroomSceneIndex::DrainRetired();
    Check(State("ready"), "index retirement restores unique binding");
    Check(State("unavailable", "{\"renderer\":\"GL\",\"roots\":[\"/missing\"]}"),
        "expected roots must match the actual viewer population");
    Check(State("unavailable", "{}"), "malformed selector fails closed");
    char shortBuffer[2] = {'x', 'y'};
    size_t required = usdGenImaging_copy_playback_status_json(empty, 0, shortBuffer, 2);
    Check(required > 2 && shortBuffer[0] == 'x' && shortBuffer[1] == 'y',
        "short ABI buffer receives no partial write");
    std::vector<char> buffer(required);
    size_t copied = usdGenImaging_copy_playback_status_json(empty, 0, buffer.data(), buffer.size());
    Check(copied == required && buffer.back() == '\0' && JsParseString(buffer.data()).IsObject(),
        "ABI sizing and NUL termination are coherent");
    input->AddPrims({{SdfPath("/sleeping"), TfToken("UsdGenGroom"), HdRetainedContainerDataSource::New()}});
    dynamic_cast<UsdGenGroomSceneIndex*>(index.operator->())->Synchronize();
    Check(State("ready", "{\"renderer\":\"GL\",\"roots\":[\"/sleeping\"]}"),
        "dormant groom needs no generation after coherent publication");
    input->AddPrims({{SdfPath("/empty"), TfToken("UsdGenDescription"), EmptyGraph()},
        {SdfPath("/surface"), TfToken("mesh"), Surface()},
        {SdfPath("/empty/grow"), TfToken("UsdGenGrow"), HdRetainedContainerDataSource::New()},
        {SdfPath("/empty/op"), TfToken("UsdGenScatter"), HdRetainedContainerDataSource::New(TfToken("density"),
            HdRetainedTypedSampledDataSource<double>::New(0.0))}});
    dynamic_cast<UsdGenGroomSceneIndex*>(index.operator->())->Synchronize();
    auto activeStatus = Status("{\"renderer\":\"GL\",\"roots\":[\"/sleeping\",\"/empty\"]}");
    Check(activeStatus.at("state").GetString() == "ready", "successful empty groom is ready");
    char const* active = "{\"renderer\":\"GL\",\"roots\":[\"/sleeping\",\"/empty\"]}";
    auto beforeStatic = UsdGenImagingTestHook::publishedGeneration(SdfPath("/empty"));
    sceneTime->value = 1;
    input->DirtyPrims({{SdfPath::AbsoluteRootPath(), HdDataSourceLocatorSet{HdSceneGlobalsSchema::GetDefaultLocator()}}});
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    Check(JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(active, 1)).GetJsObject().at("state").GetString() == "pending",
        "changed scene time remains pending until frontend publication");
    index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
    auto reused = JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(active, 1)).GetJsObject();
    Check(reused.at("state").GetString() == "ready" &&
        beforeStatic == UsdGenImagingTestHook::publishedGeneration(SdfPath("/empty")),
        "static groom reuses its successful generation at a changed captured scene time");
    Check(JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(active, 2)).GetJsObject().at("state").GetString() == "pending",
        "caller requested frame cannot replace actual captured scene time");
    input->AddPrims({{SdfPath("/empty"), TfToken("UsdGenDescription"), EmptyGraph(true)},
        {SdfPath("/empty/temporal"), TfToken("UsdGenPlaybackTestTemporal"), HdRetainedContainerDataSource::New()}});
    dynamic_cast<UsdGenGroomSceneIndex*>(index.operator->())->Synchronize();
    Check(JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(active, 1)).GetJsObject().at("state").GetString() == "ready",
        "time-dependent groom begins with complete current publication");
    captureEpoch.fetch_add(1);
    holdCapture.store(true); releaseCapture.store(false); captureEntered.store(false);
    sceneTime->value = 2;
    input->DirtyPrims({{SdfPath::AbsoluteRootPath(), HdDataSourceLocatorSet{HdSceneGlobalsSchema::GetDefaultLocator()}}});
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!captureEntered.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    Check(captureEntered.load(), "time-dependent cook is held deterministically");
    index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
    Check(JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(active, 2)).GetJsObject().at("state").GetString() == "pending",
        "current in-flight cook cannot expose older complete geometry as ready");
    sceneTime->value = 3;
    input->DirtyPrims({{SdfPath::AbsoluteRootPath(), HdDataSourceLocatorSet{HdSceneGlobalsSchema::GetDefaultLocator()}}});
    UsdGenImagingTestHook::groomOwnerCommandBarrier(*index);
    index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
    Check(JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(active, 3)).GetJsObject().at("state").GetString() == "pending",
        "deferred latest frame remains pending behind an in-flight cook");
    releaseCapture.store(true);
    dynamic_cast<UsdGenGroomSceneIndex*>(index.operator->())->Synchronize();
    Check(JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(active, 3)).GetJsObject().at("state").GetString() == "ready",
        "latest temporal frame becomes ready after complete publication");
    holdTile.store(true); releaseTile.store(false); tileEntered.store(false);
    input->AddPrims({{SdfPath("/empty/op"), TfToken("UsdGenScatter"), HdRetainedContainerDataSource::New(TfToken("density"),
        HdRetainedTypedSampledDataSource<double>::New(256.0))}});
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!tileEntered.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    Check(tileEntered.load(), "a nonterminal tile is held while sibling tiles finish");
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (UsdGenImagingTestHook::publishedTileCount(SdfPath("/empty")) == 0 &&
        std::chrono::steady_clock::now() < deadline) {
        index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
        std::this_thread::yield();
    }
    Check(UsdGenImagingTestHook::publishedTileCount(SdfPath("/empty")) > 0 &&
        JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(active, 3)).GetJsObject().at("state").GetString() == "pending",
        "visible progressive sibling tiles remain pending until the held tile completes");
    releaseTile.store(true); holdTile.store(false);
    dynamic_cast<UsdGenGroomSceneIndex*>(index.operator->())->Synchronize();
    Check(JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(active, 3)).GetJsObject().at("state").GetString() == "ready",
        "complete groom publication replaces progressive tiles and becomes ready");
    // Interactive rig previews dirty their surface at the SAME frame. Hold
    // one cook while mouse samples arrive: they must coalesce, not cancel it
    // or queue one cook per sample. This uses the real async owner boundary.
    auto mapDefault = MapValue::New();
    input->AddPrims({
        {SdfPath("/guides"), TfToken("basisCurves"), Guide()},
        {SdfPath("/empty/temporal"), TfToken("UsdGenPlaybackTestTemporal"),
            HdRetainedContainerDataSource::New(TfToken("guides"),
                HdRetainedTypedSampledDataSource<SdfPathVector>::New({SdfPath("/guides")}))},
        {SdfPath("/density"), TfToken("UsdGenPaintMap"),
            HdRetainedContainerDataSource::New(
                TfToken("paint"), HdRetainedContainerDataSource::New(
                    TfToken("surface"), HdRetainedTypedSampledDataSource<SdfPathVector>::New({SdfPath("/surface")}),
                    TfToken("primvar"), HdRetainedTypedSampledDataSource<TfToken>::New(TfToken("mask"))),
                TfToken("map"), HdRetainedContainerDataSource::New(TfToken("default"), mapDefault))},
        {SdfPath("/empty/op"), TfToken("UsdGenScatter"),
            HdRetainedContainerDataSource::New(
                TfToken("density"), HdRetainedTypedSampledDataSource<double>::New(256.0),
                TfToken("densityMap"), HdRetainedTypedSampledDataSource<SdfPathVector>::New({SdfPath("/density")}))}});
    dynamic_cast<UsdGenGroomSceneIndex*>(index.operator->())->Synchronize();
    beforeStatic = UsdGenImagingTestHook::publishedGeneration(SdfPath("/empty"));
    auto const tilesBeforeDrag = UsdGenImagingTestHook::publishedTileCount(SdfPath("/empty"));
    auto const cooksBeforeDrag = UsdGenImagingTestHook::groomCookCount(*index);
    captureEpoch.fetch_add(1);
    holdCapture.store(true); releaseCapture.store(false); captureEntered.store(false);
    auto poseDirty = [&] {
        for (auto &point : surfacePoints->value) point[0] += 0.01f;
        input->DirtyPrims({{SdfPath("/surface"), HdDataSourceLocatorSet{
            HdDataSourceLocator(TfToken("primvars"), TfToken("points"))}}});
    };
    poseDirty();
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!captureEntered.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    Check(captureEntered.load(), "same-frame pose cook is held deterministically");
    for (int sample = 0; sample != 20; ++sample) {
        poseDirty();
        UsdGenImagingTestHook::groomOwnerCommandBarrier(*index);
        index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
        Check(UsdGenImagingTestHook::publishedTileCount(SdfPath("/empty")) == tilesBeforeDrag,
            "in-flight drag retains every displayed hair tile");
    }
    Check(UsdGenImagingTestHook::groomCookCount(*index) == cooksBeforeDrag + 1,
        "same-frame drag samples coalesce behind the admitted cook");
    releaseCapture.store(true);
    dynamic_cast<UsdGenGroomSceneIndex*>(index.operator->())->Synchronize();
    Check(UsdGenImagingTestHook::groomCookCount(*index) == cooksBeforeDrag + 2 &&
        UsdGenImagingTestHook::publishedGeneration(SdfPath("/empty")) > beforeStatic &&
        UsdGenImagingTestHook::publishedTileCount(SdfPath("/empty")) == tilesBeforeDrag,
        "drag completes admitted and latest poses without removing hair tiles");
    // A map edit outside the groom namespace must still supersede an old
    // pose. Equal map payloads permit coalescing; changed ones do not.
    captureEpoch.fetch_add(1);
    holdCapture.store(true); releaseCapture.store(false); captureEntered.store(false);
    poseDirty();
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!captureEntered.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    Check(captureEntered.load(), "pose before external map edit is held");
    auto const cooksBeforeMap = UsdGenImagingTestHook::groomCookCount(*index);
    mapDefault->value = 0.5f;
    input->DirtyPrims({{SdfPath("/density"), HdDataSourceLocatorSet::UniversalSet()}});
    UsdGenImagingTestHook::groomOwnerCommandBarrier(*index);
    Check(UsdGenImagingTestHook::groomCookCount(*index) == cooksBeforeMap + 1,
        "changed external map immediately supersedes the in-flight pose");
    releaseCapture.store(true);
    dynamic_cast<UsdGenGroomSceneIndex*>(index.operator->())->Synchronize();
    beforeStatic = UsdGenImagingTestHook::publishedGeneration(SdfPath("/empty"));
    input->AddPrims({{SdfPath("/empty"), TfToken("UsdGenDescription"), HdRetainedContainerDataSource::New()}});
    dynamic_cast<UsdGenGroomSceneIndex*>(index.operator->())->Synchronize();
    Check(JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(active, 3)).GetJsObject().at("state").GetString() == "failed" &&
        beforeStatic == UsdGenImagingTestHook::publishedGeneration(SdfPath("/empty")),
        "failed current request cannot make an older successful generation ready");
    input->AddPrims({{SdfPath("/empty"), TfToken("UsdGenDescription"), HdRetainedContainerDataSource::New(
        TfToken("visibility"), HdRetainedContainerDataSource::New(TfToken("visibility"),
            HdRetainedTypedSampledDataSource<bool>::New(false)))}});
    dynamic_cast<UsdGenGroomSceneIndex*>(index.operator->())->Synchronize();
    Check(JsParseString(UsdGenGroomSceneIndex::PlaybackStatusJson(active, 3)).GetJsObject().at("state").GetString() == "ready",
        "disabled invisible groom is excluded after its hidden publication is coherent");
    index.Reset();
    UsdGenGroomSceneIndex::DrainRetired();
    return failures ? 1 : 0;
}



