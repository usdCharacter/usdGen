// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Native time-driven scene-index publication regressions.
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenImaging/testHook.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"

#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/overlayContainerDataSource.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/filteringSceneIndex.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"
#include "pxr/imaging/hd/systemMessages.h"
#include "pxr/imaging/hdsi/sceneGlobalsSceneIndex.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/timeCode.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {
int failures = 0;

void Check(bool value, char const *message)
{
    if (!value) {
        ++failures;
        std::printf("FAIL: %s\n", message);
    } else {
        std::printf("ok:   %s\n", message);
    }
}

char const *kFixture = R"USDA(#usda 1.0
def Mesh "Scalp" (prepend apiSchemas = ["UsdGenRestAPI"])
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(-1,0,-1), (1,0,-1), (1,0,1), (-1,0,1)]
    point3f[] primvars:rest = [(-1,0,-1), (1,0,-1), (1,0,1), (-1,0,1)] (interpolation = "vertex")
    texCoord2f[] primvars:st = [(0,0), (1,0), (1,1), (0,1)] (interpolation = "vertex")
}
# Legacy CPU reference fixture: an untyped container has no CUDA backend
# schema fallback. Production UsdGenGroom remains CUDA-only.
def Scope "Groom"
{
    def UsdGenDescription "hair"
    {
        def Scope "Ops"
        {
            def UsdGenWidth "width"
            {
                float usdGen:width = 0.02
                bool usdGen:replace = true
                uniform token usdGen:width:interpolation = "linear"
                float2[] usdGen:width:knots = [(0, 1), (1, 1)]
            }
            def UsdGenCurveSource "source"
            {
                rel usdGen:surface = </Scalp>
                rel usdGen:curves = </Groom/hair/curves>
                # Keep a uniform publication fixture; ragged source handling
                # is covered separately by the engine/CUDA source tests.
                int usdGen:resampleTo = 4
                uniform token usdGen:rebind = "never"
            }
        }
        def BasisCurves "curves" (
            prepend apiSchemas = ["UsdGenCurveAPI"]
        )
        {
            uniform token type = "cubic"
            uniform token basis = "bspline"
            uniform token wrap = "pinned"
            int[] curveVertexCounts = [4]
            point3f[] points = [(0,0,0), (0,1,0), (0,2,0), (0,3,0)]
            uniform token primvars:usdGen:role = "hair"
            uint64[] primvars:usdGen:curveId = [1] (interpolation = "uniform")
            # Explicit never-rebind input: these bindings are authored C3
            # data, rather than relying on the old scalp-as-curves surrogate.
            int[] primvars:skinprim = [0] (interpolation = "uniform")
            texCoord2f[] primvars:skinprimuv = [(0.5,0.5)] (interpolation = "uniform")
            matrix4d[] primvars:usdGen:rootFrame = [((1,0,0,0),(0,0,1,0),(0,-1,0,0),(0,0,0,1))] (interpolation = "uniform")
        }
    }
}
)USDA";

struct Scene {
    UsdStageRefPtr stage;
    UsdImagingSceneIndices indices;
    HdsiSceneGlobalsSceneIndexRefPtr globals;
    HdSceneIndexBaseRefPtr groom;
    UsdGenGroomSceneIndex *owner = nullptr;
};

Scene MakeScene(char const *name, char const *fixture = kFixture,
                bool withGlobals = false)
{
    Scene out;
    out.stage = UsdStage::CreateInMemory(name);
    out.stage->GetRootLayer()->ImportFromString(fixture);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = out.stage;
    out.indices = UsdImagingCreateSceneIndices(info);
    HdSceneIndexBaseRefPtr input = out.indices.finalSceneIndex;
    if (withGlobals) {
        out.globals = HdsiSceneGlobalsSceneIndex::New(input);
        input = out.globals;
    }
    out.groom = UsdGenGroomSceneIndex::New(input);
    out.owner = dynamic_cast<UsdGenGroomSceneIndex *>(out.groom.operator->());
    return out;
}

SdfPath FirstTile(Scene const &scene)
{
    SdfPath const render("/Groom/hair/__usdGenRender");
    SdfPathVector children = scene.groom->GetChildPrimPaths(render);
    return children.empty() ? SdfPath() : children.front();
}

float FirstWidth(Scene const &scene, SdfPath const &tile)
{
    HdSceneIndexPrim prim = scene.groom->GetPrim(tile);
    HdDataSourceBaseHandle value = HdContainerDataSource::Get(
        prim.dataSource, HdDataSourceLocator(TfToken("primvars"),
        TfToken("widths"), TfToken("primvarValue")));
    HdSampledDataSourceHandle sampled = HdSampledDataSource::Cast(value);
    if (!sampled) return -1.0f;
    VtValue result = sampled->GetValue(0.0f);
    if (!result.IsHolding<VtFloatArray>() || result.UncheckedGet<VtFloatArray>().empty())
        return -1.0f;
    return result.UncheckedGet<VtFloatArray>()[0];
}

std::vector<GfVec3f> TilePoints(Scene const &scene, SdfPath const &tile)
{
    HdSceneIndexPrim const prim = scene.groom->GetPrim(tile);
    auto const sampled = HdSampledDataSource::Cast(HdContainerDataSource::Get(
        prim.dataSource, HdDataSourceLocator(TfToken("primvars"),
        TfToken("points"), TfToken("primvarValue"))));
    if (!sampled) return {};
    VtValue const value = sampled->GetValue(0.0f);
    if (!value.IsHolding<VtVec3fArray>()) return {};
    auto const& points = value.UncheckedGet<VtVec3fArray>();
    return {points.begin(), points.end()};
}

uint64_t PublishedPointsHash(Scene const &scene, SdfPath const &render)
{
    uint64_t hash = 1469598103934665603ull;
    for (SdfPath const& tile : scene.groom->GetChildPrimPaths(render)) {
        auto points = TilePoints(scene, tile);
        for (GfVec3f const& point : points) {
            auto const* bytes = reinterpret_cast<unsigned char const*>(&point);
            for (size_t i = 0; i != sizeof(point); ++i)
                hash = (hash ^ bytes[i]) * 1099511628211ull;
        }
    }
    return hash;
}

Scene MakeCanonicalCollide(char const* name)
{
    Scene out;
    out.stage = UsdStage::Open(std::string(USDGEN_SOURCE_DIR) +
        "/examples/docs/operators/collide.usda", UsdStage::LoadAll);
    Check(bool(out.stage), name);
    if (!out.stage) return out;
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = out.stage;
    out.indices = UsdImagingCreateSceneIndices(info);
    out.indices.stageSceneIndex->SetTime(UsdTimeCode(0.0));
    out.globals = HdsiSceneGlobalsSceneIndex::New(out.indices.finalSceneIndex);
    out.globals->SetCurrentFrame(0.0);
    out.groom = UsdGenGroomSceneIndex::New(out.globals);
    out.owner = dynamic_cast<UsdGenGroomSceneIndex*>(out.groom.operator->());
    return out;
}

void SetCanonicalFrame(Scene& scene, double frame)
{
    scene.indices.stageSceneIndex->SetTime(UsdTimeCode(frame));
    scene.globals->SetCurrentFrame(frame);
    scene.indices.stageSceneIndex->ApplyPendingUpdates();
}

void CheckCanonicalAsyncPlayback()
{
    Scene live = MakeCanonicalCollide("Canonical Collide playback stage opens");
    if (!live.owner) return;
    SdfPath const root("/World/Groom/Fur");
    SdfPath const render("/World/Groom/Fur/__usdGenRender");
    live.owner->Synchronize();
    uint64_t const initialHash = PublishedPointsHash(live, render);
    int64_t const initialGeneration =
        UsdGenImagingTestHook::publishedGeneration(root);
    Check(initialGeneration >= 0, "Canonical Collide publishes initial tiles");
    live.groom->SystemMessage(HdSystemMessageTokens->asyncAllow, nullptr);

    // Continue changing stage time faster than the canonical dual-collider
    // cook. The first frame must still publish while newer times are
    // arriving, and the most recent request must drain after playback stops.
    bool publishedDuringPlayback = false;
    bool changedDuringPlayback = false;
    auto const start = std::chrono::steady_clock::now();
    unsigned step = 0;
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
        double const frame = double((++step * 2) % 100);
        SetCanonicalFrame(live, frame);
        live.groom->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
        if (UsdGenImagingTestHook::publishedGeneration(root) > initialGeneration) {
            publishedDuringPlayback = true;
            changedDuringPlayback |= PublishedPointsHash(live, render) != initialHash;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(35));
    }
    Check(publishedDuringPlayback,
          "Continuous Collide playback publishes while later frames arrive");
    Check(changedDuringPlayback,
          "Continuous Collide playback changes native points before Play stops");
    SetCanonicalFrame(live, 99.0);
    live.owner->Synchronize();
    uint64_t const stoppedHash = PublishedPointsHash(live, render);
    Scene fresh = MakeCanonicalCollide("Fresh canonical Collide comparison opens");
    if (!fresh.owner) return;
    SetCanonicalFrame(fresh, 99.0);
    fresh.owner->Synchronize();
    Check(stoppedHash == PublishedPointsHash(fresh, render),
          "Latest deferred Collide frame drains after playback stops");

    // A topology edit in the same stage-time turn must cancel the temporal
    // fast path. Keeping only one complete authored Shield face leaves most
    // of its contact absent, so stale full-Shield output is observable in the
    // native points while preserving valid subdivision topology.
    SetCanonicalFrame(fresh, 27.0);
    fresh.owner->Synchronize();
    uint64_t const uneditedHash = PublishedPointsHash(fresh, render);
    uint64_t const baseline = stoppedHash;
    uint64_t const cooksBeforeEdit =
        UsdGenImagingTestHook::groomCookCount(*live.groom);
    SetCanonicalFrame(live, 27.0);
    auto const beganDeadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(2);
    while (UsdGenImagingTestHook::groomCookCount(*live.groom) <= cooksBeforeEdit &&
           std::chrono::steady_clock::now() < beganDeadline) {
        live.groom->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(UsdGenImagingTestHook::groomCookCount(*live.groom) > cooksBeforeEdit,
          "Unedited temporal cook starts before same-tick topology edit");
    UsdGeomMesh shield(live.stage->GetPrimAtPath(SdfPath("/World/Shield")));
    Check(bool(shield), "Canonical external collider exists for topology edit");
    if (!shield) return;
    VtIntArray originalIndices;
    VtIntArray originalCounts;
    shield.GetFaceVertexIndicesAttr().Get(&originalIndices);
    shield.GetFaceVertexCountsAttr().Get(&originalCounts);
    int const retainedCount = originalCounts.empty() ? 0 : originalCounts[0];
    Check(retainedCount >= 3 && originalIndices.size() >= size_t(retainedCount),
          "Canonical collider has a complete authored face to retain");
    if (retainedCount < 3 || originalIndices.size() < size_t(retainedCount)) return;
    VtIntArray const oneFace(originalIndices.begin(),
                             originalIndices.begin() + retainedCount);
    shield.GetFaceVertexCountsAttr().Set(VtIntArray{retainedCount});
    shield.GetFaceVertexIndicesAttr().Set(oneFace);
    live.indices.stageSceneIndex->ApplyPendingUpdates();
    live.globals->SetCurrentFrame(27.0);
    UsdGenImagingTestHook::groomOwnerCommandBarrier(*live.groom);
    Check(UsdGenImagingTestHook::groomCookCount(*live.groom) > cooksBeforeEdit + 1,
          "External collider topology edit enters immediate cook path");
    bool sawUneditedAfterEdit = false;
    auto const observeUntil = std::chrono::steady_clock::now() +
        std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < observeUntil) {
        live.groom->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
        sawUneditedAfterEdit |= PublishedPointsHash(live, render) == uneditedHash;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    live.owner->Synchronize();
    uint64_t const editedHash = PublishedPointsHash(live, render);
    Scene freshEdited = MakeCanonicalCollide("Fresh topology-edited Collide comparison opens");
    if (!freshEdited.owner) return;
    UsdGeomMesh freshShield(freshEdited.stage->GetPrimAtPath(SdfPath("/World/Shield")));
    freshShield.GetFaceVertexCountsAttr().Set(VtIntArray{retainedCount});
    freshShield.GetFaceVertexIndicesAttr().Set(oneFace);
    SetCanonicalFrame(freshEdited, 27.0);
    freshEdited.owner->Synchronize();
    std::printf("canonical topology hashes: baseline %016llx unedited %016llx edited %016llx fresh %016llx staleSeen %d\n",
        static_cast<unsigned long long>(baseline),
        static_cast<unsigned long long>(uneditedHash),
        static_cast<unsigned long long>(editedHash),
        static_cast<unsigned long long>(PublishedPointsHash(freshEdited, render)),
        int(sawUneditedAfterEdit));
    Check(editedHash == PublishedPointsHash(freshEdited, render) &&
              editedHash != baseline && editedHash != uneditedHash &&
              !sawUneditedAfterEdit,
          "Same-tick external topology edit fences an already-running old cook");
    Check(UsdGenImagingTestHook::staleGroomProgressRejected(*live.groom, root),
          "Late pre-edit tile progress stays fenced after edited terminal publication");

    // A hole opinion changes topology without changing face counts or index
    // array length. It must still bypass the temporal deferral path.
    uint64_t const beforeHoleFrame =
        UsdGenImagingTestHook::groomCookCount(*live.groom);
    SetCanonicalFrame(live, 28.0);
    auto const holeBeginDeadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(2);
    while (UsdGenImagingTestHook::groomCookCount(*live.groom) <= beforeHoleFrame &&
           std::chrono::steady_clock::now() < holeBeginDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Check(UsdGenImagingTestHook::groomCookCount(*live.groom) > beforeHoleFrame,
          "Temporal cook starts before collider hole edit");
    shield.GetHoleIndicesAttr().Set(VtIntArray{0});
    live.indices.stageSceneIndex->ApplyPendingUpdates();
    UsdGenImagingTestHook::groomOwnerCommandBarrier(*live.groom);
    Check(UsdGenImagingTestHook::groomCookCount(*live.groom) > beforeHoleFrame + 1,
          "Same-shape collider hole edit enters immediate cook path");
    live.owner->Synchronize();
}

void CheckNativeWindPlayback()
{
    std::string fixture(kFixture);
    size_t const begin = fixture.find("def UsdGenWidth \"width\"");
    size_t const end = fixture.find("def UsdGenCurveSource \"source\"", begin);
    Check(begin != std::string::npos && end != std::string::npos,
          "Native Wind fixture has the expected operator section");
    if (begin == std::string::npos || end == std::string::npos) return;
    fixture.replace(begin, end - begin, R"USDA(def UsdGenWind "wind"
            {
                vector3f usdGen:direction = (0, 0, 1)
                float usdGen:billowLowStrength = 0.2
                float usdGen:billowLowFrequency = 0.75
                float usdGen:billowLowRate = 0.24
            }
            )USDA");
    Scene scene = MakeScene("native-wind-live-time", fixture.c_str(), true);
    scene.indices.stageSceneIndex->SetTime(UsdTimeCode(0.0));
    scene.globals->SetCurrentFrame(0.0);
    scene.indices.stageSceneIndex->ApplyPendingUpdates();
    scene.owner->Synchronize();
    SdfPath const tile = FirstTile(scene);
    Check(!tile.IsEmpty(), "Native Wind fixture publishes a tile");
    if (tile.IsEmpty()) return;
    auto const frameZero = TilePoints(scene, tile);
    scene.indices.stageSceneIndex->SetTime(UsdTimeCode(12.0));
    scene.globals->SetCurrentFrame(12.0);
    scene.indices.stageSceneIndex->ApplyPendingUpdates();
    scene.owner->Synchronize();
    auto const frameTwelve = TilePoints(scene, tile);
    Check(!frameZero.empty() && frameZero.size() == frameTwelve.size() &&
              frameZero != frameTwelve,
          "Scene-global frame dirt publishes native Wind motion without authored samples");
}

void CheckNativeCollidePlayback()
{
    std::string fixture(kFixture);
    size_t const begin = fixture.find("def UsdGenWidth \"width\"");
    size_t const end = fixture.find("def UsdGenCurveSource \"source\"", begin);
    Check(begin != std::string::npos && end != std::string::npos,
          "Native Collide fixture has the expected operator section");
    if (begin == std::string::npos || end == std::string::npos) return;
    fixture.replace(begin, end - begin, R"USDA(def UsdGenCollide "collide"
            {
                rel usdGen:colliders = </Collider>
                float usdGen:offset = 0.2
                float usdGen:pushAmount = 1
            }
            )USDA");
    size_t const groom = fixture.find("def Scope \"Groom\"");
    Check(groom != std::string::npos,
          "Native Collide fixture has a groom insertion point");
    if (groom == std::string::npos) return;
    fixture.insert(groom, R"USDA(def Mesh "Collider"
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points.timeSamples = {
        0: [(-1,-1,2), (1,-1,2), (1,4,2), (-1,4,2)],
        12: [(-1,-1,0.05), (1,-1,0.05), (1,4,0.05), (-1,4,0.05)]
    }
    double3 xformOp:translate.timeSamples = {
        0: (0,0,0),
        12: (0,0,0),
        24: (0,0,-0.1)
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
)USDA");
    Scene live = MakeScene("native-collide-live-time", fixture.c_str(), true);
    auto sample = [](Scene& scene, double frame) {
        scene.indices.stageSceneIndex->SetTime(UsdTimeCode(frame));
        scene.globals->SetCurrentFrame(frame);
        scene.indices.stageSceneIndex->ApplyPendingUpdates();
        scene.owner->Synchronize();
        return TilePoints(scene, FirstTile(scene));
    };
    auto const clear = sample(live, 0.0);
    auto const contact = sample(live, 12.0);
    auto const translated = sample(live, 24.0);
    Check(!clear.empty() && clear.size() == contact.size() &&
              contact.size() == translated.size() &&
              clear != contact && contact != translated,
          "Live Collide publishes clear, contact, and translated collider results");
    bool matchesFresh = !clear.empty();
    {
        Scene fresh = MakeScene("native-collide-fresh-clear", fixture.c_str(), true);
        matchesFresh &= clear == sample(fresh, 0.0);
    }
    {
        Scene fresh = MakeScene("native-collide-fresh-contact", fixture.c_str(), true);
        matchesFresh &= contact == sample(fresh, 12.0);
    }
    {
        Scene fresh = MakeScene("native-collide-fresh-translated", fixture.c_str(), true);
        matchesFresh &= translated == sample(fresh, 24.0);
    }
    Check(matchesFresh,
          "Live Collide clear/contact/translated match fresh current-frame cooks");

    // UsdImagingStageSceneIndex dirties animated collider points before
    // HdsiSceneGlobals advances currentFrame. A graph with both native Wind
    // and Collide must use one frame for its sampled input and motion even at
    // that intermediate caller boundary.
    std::string mixedFixture = fixture;
    size_t const collide = mixedFixture.find("def UsdGenCollide \"collide\"");
    Check(collide != std::string::npos, "Mixed native-time fixture has Collide");
    if (collide != std::string::npos) {
        mixedFixture.insert(collide, R"USDA(def UsdGenWind "wind"
            {
                vector3f usdGen:direction = (0, 0, 1)
                float usdGen:billowLowStrength = 0.2
                float usdGen:billowLowFrequency = 0.75
                float usdGen:billowLowRate = 0.24
            }
            )USDA");
        Scene split = MakeScene("native-mixed-split-notice", mixedFixture.c_str(), true);
        (void)sample(split, 0.0);
        split.indices.stageSceneIndex->SetTime(UsdTimeCode(12.0));
        split.indices.stageSceneIndex->ApplyPendingUpdates();
        split.owner->Synchronize();
        auto const beforeGlobals = TilePoints(split, FirstTile(split));
        Scene freshMixed = MakeScene("native-mixed-fresh-frame", mixedFixture.c_str(), true);
        auto const current = sample(freshMixed, 12.0);
        Check(!beforeGlobals.empty() && beforeGlobals == current,
              "Collider sample arriving before scene globals uses the same native Wind frame");
    }
}

} // namespace

int main()
{
    CheckNativeWindPlayback();
    CheckNativeCollidePlayback();
    CheckCanonicalAsyncPlayback();
    std::printf("testUsdGenNativeTimePublication: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
