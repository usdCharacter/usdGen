// A groom that is hidden (its Description resolves invisible, through its own
// opinion or an ancestor's) or disabled (deactivated, or a UsdGenGroom whose
// every Description is deactivated) is dormant: stage events that would
// re-capture and re-cook it are no-ops until it is shown or enabled again,
// and waking it cooks the edits it slept through before its tiles reappear.
#include "utils/pxrInterop.h"
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenImaging/testHook.h"
#include "usdGen/api/ext/op.h"
#include "usdGen/api/ext/opRegistry.h"

#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"
#include "pxr/imaging/hd/systemMessages.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

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

// The CPU publication fixture of testUsdGenScenePublication: an untyped
// container keeps the description on the CPU reference lane.
char const *kFixture = R"USDA(#usda 1.0
def Mesh "Scalp" (prepend apiSchemas = ["UsdGenRestAPI"])
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(-1,0,-1), (1,0,-1), (1,0,1), (-1,0,1)]
    point3f[] primvars:rest = [(-1,0,-1), (1,0,-1), (1,0,1), (-1,0,1)] (interpolation = "vertex")
    texCoord2f[] primvars:st = [(0,0), (1,0), (1,1), (0,1)] (interpolation = "vertex")
}
def Scope "Groom"
{
    def UsdGenDescription "hair"
    {
%s        def Scope "Ops"
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
            int[] primvars:skinprim = [0] (interpolation = "uniform")
            texCoord2f[] primvars:skinprimuv = [(0.5,0.5)] (interpolation = "uniform")
            matrix4d[] primvars:usdGen:rootFrame = [((1,0,0,0),(0,0,1,0),(0,-1,0,0),(0,0,0,1))] (interpolation = "uniform")
        }
    }
}
)USDA";

// A UsdGenGroom whose only Description is deactivated: nothing is left to
// cook, so the groom root must not be captured in the Description's place.
char const *kDisabledDescription = R"USDA(#usda 1.0
def UsdGenGroom "G"
{
    def UsdGenDescription "hair" (active = false)
    {
    }
}
)USDA";

SdfPath const kDescription("/Groom/hair");
SdfPath const kRender("/Groom/hair/__usdGenRender");

struct Scene {
    UsdStageRefPtr stage;
    UsdImagingSceneIndices indices;
    HdSceneIndexBaseRefPtr groom;
    UsdGenGroomSceneIndex *owner = nullptr;

    void Apply() { indices.stageSceneIndex->ApplyPendingUpdates(); }
    void Sync() { Apply(); owner->Synchronize(); }
    uint64_t Cooks() const { return UsdGenImagingTestHook::groomCookCount(*groom); }
    uint64_t Ingresses() const { return UsdGenImagingTestHook::groomCaptureCount(*groom); }
    UsdAttribute Attr(char const *path) const {
        return stage->GetAttributeAtPath(SdfPath(path));
    }
};

Scene MakeScene(char const *name, std::string const &fixture)
{
    Scene out;
    out.stage = UsdStage::CreateInMemory(name);
    out.stage->GetRootLayer()->ImportFromString(fixture);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = out.stage;
    out.indices = UsdImagingCreateSceneIndices(info);
    out.groom = UsdGenGroomSceneIndex::New(out.indices.finalSceneIndex);
    out.owner = dynamic_cast<UsdGenGroomSceneIndex *>(out.groom.operator->());
    return out;
}

// `descriptionBody` is authored at the top of the Description's body.
std::string Fixture(char const *descriptionBody = "")
{
    char buffer[8192];
    std::snprintf(buffer, sizeof(buffer), kFixture, descriptionBody);
    return buffer;
}

SdfPath FirstTile(Scene const &scene)
{
    for (SdfPath const &child : scene.groom->GetChildPrimPaths(kRender))
        if (scene.groom->GetPrim(child).primType == TfToken("basisCurves"))
            return child;
    return SdfPath();
}

float FirstWidth(Scene const &scene, SdfPath const &tile)
{
    HdSampledDataSourceHandle const sampled = HdSampledDataSource::Cast(
        HdContainerDataSource::Get(scene.groom->GetPrim(tile).dataSource,
            HdDataSourceLocator(TfToken("primvars"), TfToken("widths"),
                                TfToken("primvarValue"))));
    if (!sampled) return -1.0f;
    VtValue const value = sampled->GetValue(0.0f);
    if (!value.IsHolding<VtFloatArray>() || value.UncheckedGet<VtFloatArray>().empty())
        return -1.0f;
    return value.UncheckedGet<VtFloatArray>()[0];
}

// The visibility Hydra reads off a published tile: a tile without an
// opinion is visible.
bool Visible(Scene const &scene, SdfPath const &tile)
{
    HdBoolDataSourceHandle const visibility = HdVisibilitySchema::GetFromParent(
        scene.groom->GetPrim(tile).dataSource).GetVisibility();
    return !visibility || visibility->GetTypedValue(0.0f);
}

bool Near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

bool SetVisibility(Scene const &scene, char const *prim, char const *value)
{
    UsdAttribute const visibility = scene.stage->GetPrimAtPath(SdfPath(prim))
        .GetAttribute(TfToken("visibility"));
    return visibility && visibility.Set(TfToken(value));
}

bool MoveScalp(Scene const &scene, float y)
{
    VtVec3fArray const points{GfVec3f(-1, y, -1), GfVec3f(1, y, -1),
                              GfVec3f(1, y, 1), GfVec3f(-1, y, 1)};
    return scene.Attr("/Scalp.points").Set(points);
}

// Records whether a tile's visibility was dirtied.
class VisibilityDirtyObserver final : public HdSceneIndexObserver {
public:
    std::atomic<bool> dirtied{false};
    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &) override {}
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &) override {}
    void PrimsDirtied(HdSceneIndexBase const &, DirtiedPrimEntries const &entries) override {
        for (auto const &entry : entries)
            if (entry.primPath.HasPrefix(kRender) &&
                entry.dirtyLocators.Intersects(HdVisibilitySchema::GetDefaultLocator()))
                dirtied.store(true, std::memory_order_release);
    }
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &) override {}
};
}  // namespace

int main()
{

    // --- Hiding the Description -------------------------------------------
    Scene scene = MakeScene("dormant-hidden", Fixture());
    if (!scene.owner) {
        std::printf("FAIL: groom scene index is available\n");
        return 1;
    }
    scene.owner->Synchronize();
    SdfPath const tile = FirstTile(scene);
    Check(!tile.IsEmpty() && Near(FirstWidth(scene, tile), 0.02f) && Visible(scene, tile),
          "a visible groom publishes a visible tile of width 0.02");
    uint64_t const initialCooks = scene.Cooks();
    Check(initialCooks > 0, "the visible groom cooked");

    VisibilityDirtyObserver visibilityNotices;
    scene.groom->AddObserver(TfCreateWeakPtr(&visibilityNotices));
    Check(SetVisibility(scene, "/Groom/hair", "invisible"), "the Description is made invisible");
    scene.Sync();
    Check(scene.Cooks() == initialCooks, "hiding a groom does not cook it");
    Check(!Visible(scene, tile), "the hidden groom's tile reads invisible");
    Check(visibilityNotices.dirtied.load(std::memory_order_acquire),
          "hiding dirties the tile's visibility");

    uint64_t ingress = scene.Ingresses();
    Check(scene.Attr("/Groom/hair/Ops/width.usdGen:width").Set(0.08f),
          "a width edit authors while hidden");
    scene.Sync();
    Check(MoveScalp(scene, 0.5f), "the bound surface deforms while hidden");
    scene.Sync();
    Check(scene.Ingresses() > ingress, "the edits reached the groom scene index");
    Check(scene.Cooks() == initialCooks,
          "operator and surface edits on a hidden groom are no-ops (no cook)");
    Check(Near(FirstWidth(scene, tile), 0.02f) && !Visible(scene, tile),
          "the hidden groom keeps its last-good tile, still hidden");

    visibilityNotices.dirtied.store(false);
    Check(SetVisibility(scene, "/Groom/hair", "inherited"), "the Description is shown again");
    scene.Sync();
    Check(scene.Cooks() > initialCooks, "showing the groom cooks it");
    Check(Near(FirstWidth(scene, tile), 0.08f) && Visible(scene, tile),
          "the shown groom publishes the edit it slept through, visible");
    Check(visibilityNotices.dirtied.load(std::memory_order_acquire),
          "showing dirties the tile's visibility");

    // --- Hiding an ancestor -------------------------------------------------
    uint64_t cooks = scene.Cooks();
    Check(SetVisibility(scene, "/Groom", "invisible"), "the groom's ancestor is made invisible");
    scene.Sync();
    Check(scene.Attr("/Groom/hair/Ops/width.usdGen:width").Set(0.1f),
          "a width edit authors under a hidden ancestor");
    scene.Sync();
    Check(scene.Cooks() == cooks && !Visible(scene, tile) &&
              Near(FirstWidth(scene, tile), 0.08f),
          "an invisible ancestor makes the groom dormant: no cook, tile hidden");
    Check(SetVisibility(scene, "/Groom", "inherited"), "the ancestor is shown again");
    scene.Sync();
    Check(scene.Cooks() > cooks && Visible(scene, tile) && Near(FirstWidth(scene, tile), 0.1f),
          "showing the ancestor wakes and cooks the groom");

    // --- Animated inputs while hidden ----------------------------------------
    UsdAttribute width = scene.Attr("/Groom/hair/Ops/width.usdGen:width");
    Check(width.Set(0.04f, UsdTimeCode(1.0)) && width.Set(0.12f, UsdTimeCode(2.0)),
          "animated width samples author");
    scene.indices.stageSceneIndex->SetTime(UsdTimeCode(1.0));
    scene.Sync();
    Check(Near(FirstWidth(scene, tile), 0.04f), "frame 1 samples the animated width");
    Check(SetVisibility(scene, "/Groom/hair", "invisible"), "hidden again before playback");
    scene.Sync();
    cooks = scene.Cooks();
    scene.indices.stageSceneIndex->SetTime(UsdTimeCode(2.0));
    scene.Sync();
    Check(scene.Cooks() == cooks, "a frame change does not cook a hidden groom");
    Check(SetVisibility(scene, "/Groom/hair", "inherited"), "shown at frame 2");
    scene.Sync();
    Check(scene.Cooks() > cooks && Visible(scene, tile) && Near(FirstWidth(scene, tile), 0.12f),
          "showing the groom at frame 2 cooks the frame-2 width");

    // --- Waking in async mode never shows stale strands ----------------------
    scene.groom->SystemMessage(HdSystemMessageTokens->asyncAllow, nullptr);
    Check(SetVisibility(scene, "/Groom/hair", "invisible"), "async: hidden");
    scene.Apply();
    // Apply queues async ingress. Observe its hidden publication before
    // editing and waking, so the first-visible check starts from dormancy.
    bool actuallyHidden = false;
    auto const hideDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < hideDeadline) {
        scene.groom->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
        auto const prim = scene.groom->GetPrim(tile);
        auto const visibility = HdVisibilitySchema::GetFromParent(prim.dataSource).GetVisibility();
        if (prim.dataSource && visibility && !visibility->GetTypedValue(0.0f)) {
            actuallyHidden = true; break;
        }
        std::this_thread::yield();
    }
    Check(actuallyHidden, "async: hide is published before edit and wake");
    Check(width.Set(0.2f, UsdTimeCode(2.0)), "async: width edit while hidden");
    scene.Apply();
    Check(SetVisibility(scene, "/Groom/hair", "inherited"), "async: shown");
    scene.Apply();
    bool revealed = false, staleWhileVisible = false;
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        scene.groom->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
        auto const prim = scene.groom->GetPrim(tile);
        auto const visibility = HdVisibilitySchema::GetFromParent(prim.dataSource).GetVisibility();
        if (prim.dataSource && (!visibility || visibility->GetTypedValue(0.0f))) {
            revealed = true;
            auto const sampled = HdSampledDataSource::Cast(HdContainerDataSource::Get(
                prim.dataSource, HdDataSourceLocator(TfToken("primvars"),
                    TfToken("widths"), TfToken("primvarValue"))));
            auto const value = sampled ? sampled->GetValue(0.0f) : VtValue();
            float const observedWidth = value.IsHolding<VtFloatArray>() &&
                !value.UncheckedGet<VtFloatArray>().empty()
                ? value.UncheckedGet<VtFloatArray>()[0] : -1.0f;
            staleWhileVisible = !Near(observedWidth, 0.2f);
            break;
        }
        std::this_thread::yield();
    }
    Check(revealed && !staleWhileVisible,
          "async: the woken tile reappears only with the edit it slept through");
    scene.groom->RemoveObserver(TfCreateWeakPtr(&visibilityNotices));

    // --- Initially invisible ------------------------------------------------
    Scene hidden = MakeScene("dormant-initially-hidden",
                             Fixture("        token visibility = \"invisible\"\n"));
    hidden.owner->Synchronize();
    Check(hidden.Cooks() == 0 && FirstTile(hidden).IsEmpty(),
          "an initially invisible groom is never cooked and publishes no tile");
    Check(MoveScalp(hidden, 0.25f), "the surface deforms under the never-shown groom");
    hidden.Sync();
    Check(hidden.Cooks() == 0, "the never-shown groom stays uncooked");
    Check(SetVisibility(hidden, "/Groom/hair", "inherited"), "the never-shown groom is shown");
    hidden.Sync();
    SdfPath const shownTile = FirstTile(hidden);
    Check(hidden.Cooks() > 0 && !shownTile.IsEmpty() && Visible(hidden, shownTile) &&
              Near(FirstWidth(hidden, shownTile), 0.02f),
          "showing it cooks and publishes a visible tile");

    // --- Deactivated Description (CPU lane) -----------------------------------
    Scene inactive = MakeScene("dormant-deactivated", Fixture());
    inactive.owner->Synchronize();
    Check(!FirstTile(inactive).IsEmpty(), "the active groom publishes");
    inactive.stage->GetPrimAtPath(kDescription).SetActive(false);
    inactive.Sync();
    cooks = inactive.Cooks();
    Check(FirstTile(inactive).IsEmpty(), "deactivating the Description retires its tiles");
    ingress = inactive.Ingresses();
    Check(MoveScalp(inactive, 0.75f), "the surface deforms under the deactivated groom");
    inactive.Sync();
    Check(inactive.Ingresses() > ingress && inactive.Cooks() == cooks,
          "a deactivated groom does not cook on its surface's edits");
    inactive.stage->GetPrimAtPath(kDescription).SetActive(true);
    inactive.Sync();
    SdfPath const reactivatedTile = FirstTile(inactive);
    Check(inactive.Cooks() > cooks && !reactivatedTile.IsEmpty() &&
              Visible(inactive, reactivatedTile),
          "reactivating the Description cooks and publishes it again");

    // --- UsdGenGroom whose only Description is deactivated --------------------
    Scene disabled = MakeScene("dormant-disabled-description", kDisabledDescription);
    disabled.owner->Synchronize();
    Check(disabled.Cooks() == 0,
          "a UsdGenGroom with no active Description is not cooked in its place");
    ingress = disabled.Ingresses();
    Check(disabled.Attr("/G.usdGen:sessionId").Set(std::string("dormant")),
          "a groom-root edit authors");
    disabled.Sync();
    Check(disabled.Ingresses() > ingress && disabled.Cooks() == 0,
          "groom-root edits do not cook a groom whose Description is deactivated");

    std::printf("testUsdGenDormantGroom: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
