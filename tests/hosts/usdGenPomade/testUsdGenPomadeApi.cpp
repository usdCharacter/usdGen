// testUsdGenPomadeApi — T1: the pomadeApi C ABI over PomadeModel.
//
// No stage, no Hydra: pure C ABI over the model, like testUsdGenToolsApi.
// Proves create/destroy, test-tube build defaults, sculpt stub versioning,
// staged point reads, honest errors on bad arguments, the V0 registry and
// level display, and the V1 surface: selection round trips per kind,
// marquee and lasso agreeing with the point pick on a subdivided model,
// hover, the gizmo/brush records and the geometry they build, the gesture
// bracket (one undo step per drag, bit-exact cancel), redo, undo labels
// and graph edits on the same undo stack.
#include "usdGenPomade/pomadeApi.h"
#include "usdGenPomade/pomadeApiStage.h"
#include "usdGenPomade/pomadeGizmo.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeSelection.h"
#include "usdGenPomade/pomadeRegistry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
    // Unbuffered: a crash mid-run must still name the last check that ran.
    std::fflush(stdout);
}

} // namespace

int
main()
{
    Check(Pomade_Create(nullptr) == POMADE_ERROR,
          "Pomade_Create rejects null out-param");
    PomadeModelContext *ctx = nullptr;
    Check(Pomade_Create(&ctx) == POMADE_OK && ctx != nullptr,
          "Pomade_Create succeeds");
    Check(Pomade_Destroy(nullptr) == POMADE_OK,
          "Pomade_Destroy(null) is a no-op success");
    if (!ctx) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    Check(Pomade_GetVersion(ctx) == 0, "fresh model version is 0");
    Check(Pomade_GetVertexCount(ctx) == 0, "no tube before build");
    Check(Pomade_BuildTestTube(nullptr, 0, 0, 0.0f, 0.0f) == POMADE_ERROR,
          "Pomade_BuildTestTube rejects null context");

    // Non-positive arguments select the defaults (5 x 8, r=0.5, len=4).
    Check(Pomade_BuildTestTube(ctx, 0, 0, 0.0f, 0.0f) == POMADE_OK,
          "Pomade_BuildTestTube with defaults succeeds");
    Check(Pomade_GetVersion(ctx) == 1, "build bumps version to 1");
    Check(Pomade_GetVertexCount(ctx) == 40, "default tube has 40 verts");
    Check(Pomade_GetQuadCount(ctx) == 32, "default tube has 32 quads");
    int dirty = Pomade_TakeDirty(ctx);
    Check((dirty & usdGenPomade::PomadeDirty_Topology) != 0,
          "build marks topology dirty");
    Check(Pomade_TakeDirty(ctx) == 0, "TakeDirty clears");

    Check(Pomade_MoveCenterRing(ctx, 99, 1.0f, 0.0f) == POMADE_ERROR,
          "MoveCenterRing rejects out-of-range ring");
    Check(Pomade_MoveCenterRing(ctx, 2, 1.0f, 0.0f) == POMADE_OK,
          "MoveCenterRing applies");
    Check(Pomade_GetVersion(ctx) == 2, "sculpt bumps version to 2");
    Check(Pomade_TakeDirty(ctx) == int(usdGenPomade::PomadeDirty_Points),
          "sculpt marks exactly points dirty");

    // Output settings are authored commit policy, deliberately independent
    // of both the mesh and Fill's transient preview fraction.
    {
        int outputEnabled = -1;
        float outputMultiplier = -1.0f, outputWidth = -1.0f;
        Check(Pomade_GetOutputSettings(ctx, &outputEnabled, &outputMultiplier,
                                      &outputWidth) == POMADE_OK &&
                  outputEnabled == 0 && outputMultiplier == 1.0f &&
                  outputWidth == 0.01f,
              "output defaults are disabled, full density and .01 width");
        unsigned long long const outputVersion = Pomade_GetVersion(ctx);
        Check(Pomade_SetOutputSettings(ctx, 1, 2.5f, 0.04f) == POMADE_OK &&
                  Pomade_GetVersion(ctx) == outputVersion + 1 &&
                  Pomade_GetOutputSettings(ctx, &outputEnabled,
                                          &outputMultiplier, &outputWidth) ==
                      POMADE_OK &&
                  outputEnabled == 1 && outputMultiplier == 2.5f &&
                  outputWidth == 0.04f,
              "output settings update atomically and version the commit state");
        Check(Pomade_SetOutputSettings(ctx, 2, 1.0f, 0.01f) == POMADE_ERROR &&
                  Pomade_SetOutputSettings(
                      ctx, 1, std::numeric_limits<float>::quiet_NaN(),
                      0.01f) == POMADE_ERROR &&
                  Pomade_SetOutputSettings(ctx, 1, 1.0f, -0.01f) ==
                      POMADE_ERROR &&
                  Pomade_GetVersion(ctx) == outputVersion + 1 &&
                  Pomade_GetOutputSettings(ctx, &outputEnabled,
                                          &outputMultiplier, &outputWidth) ==
                      POMADE_OK &&
                  outputEnabled == 1 && outputMultiplier == 2.5f &&
                  outputWidth == 0.04f,
              "invalid output settings leave the prior state exact");
        unsigned int outputUndoDirty = 0;
        Check(Pomade_Undo(ctx, &outputUndoDirty) == POMADE_OK &&
                  Pomade_GetOutputSettings(ctx, &outputEnabled,
                                          &outputMultiplier, &outputWidth) ==
                      POMADE_OK &&
                  outputEnabled == 0 && outputMultiplier == 1.0f &&
                  outputWidth == 0.01f &&
                  Pomade_Redo(ctx, &outputUndoDirty) == POMADE_OK &&
                  Pomade_GetOutputSettings(ctx, &outputEnabled,
                                          &outputMultiplier, &outputWidth) ==
                      POMADE_OK &&
                  outputEnabled == 1 && outputMultiplier == 2.5f &&
                  outputWidth == 0.04f,
              "output settings survive one undo and redo atomically");
    }

    std::vector<float> xyz(size_t(Pomade_GetVertexCount(ctx)) * 3);
    Check(Pomade_ReadTubePoints(ctx, nullptr, 0) == POMADE_ERROR,
          "ReadTubePoints rejects null output");
    Check(Pomade_ReadTubePoints(ctx, xyz.data(), 3) == POMADE_ERROR,
          "ReadTubePoints rejects a short buffer");
    Check(Pomade_ReadTubePoints(ctx, xyz.data(), int(xyz.size())) == POMADE_OK,
          "ReadTubePoints stages the tube");
    Check(std::abs(xyz[0] - 0.5f) < 1e-6f && std::abs(xyz[1]) < 1e-6f &&
              std::abs(xyz[2]) < 1e-6f,
          "staged ring-0 slot-0 is (0.5, 0, 0)");
    Check(std::abs(xyz[16 * 3] - 1.5f) < 1e-5f &&
              std::abs(xyz[16 * 3 + 1] - 2.0f) < 1e-6f,
          "staged ring-2 slot-0 moved to (1.5, 2, 0)");
    Check(std::string(Pomade_GetLastError()).size() >= 0,
          "GetLastError returns a string");

    // A 1-ring build is rejected; the model keeps the previous tube.
    unsigned long long const versionBeforeFailedBuild = Pomade_GetVersion(ctx);
    Check(Pomade_BuildTestTube(ctx, 1, 8, 0.5f, 4.0f) == POMADE_ERROR,
          "BuildTestTube rejects rings < 2");
    Check(Pomade_GetVersion(ctx) == versionBeforeFailedBuild,
          "failed build keeps the version");
    Check(Pomade_GetVertexCount(ctx) == 40, "failed build keeps the tube");

    Check(Pomade_Destroy(ctx) == POMADE_OK, "Pomade_Destroy succeeds");

    // -- P6 undo ----------------------------------------------------------
    Check(Pomade_Undo(nullptr, nullptr) == POMADE_ERROR, "undo rejects null context");
    Check(Pomade_GetUndoDepth(nullptr) == 0, "undo depth is 0 on null");
    Check(Pomade_GetUndoBytes(nullptr) == 0, "undo bytes are 0 on null");
    Check(Pomade_SetUndoBudget(nullptr, 4, 1024) == POMADE_ERROR,
          "undo budget rejects null context");
    Check(Pomade_ClearUndo(nullptr) == POMADE_ERROR,
          "undo clear rejects null context");
    {
        PomadeModelContext *uc = nullptr;
        Check(Pomade_Create(&uc) == POMADE_OK && uc != nullptr,
              "undo: model creates");
        if (uc) {
            Check(Pomade_GetUndoDepth(uc) == 0, "undo: fresh depth is 0");
            Check(Pomade_GetUndoBytes(uc) == 0, "undo: fresh bytes are 0");
            Check(Pomade_Undo(uc, nullptr) == POMADE_OK, "undo: empty stack is a no-op");
            Check(Pomade_BuildTestTube(uc, 0, 0, 0.0f, 0.0f) == POMADE_OK,
                  "undo: build succeeds");
            Check(Pomade_GetUndoDepth(uc) == 0, "undo: build holds no steps");
            std::vector<float> before(
                size_t(Pomade_GetVertexCount(uc)) * 3);
            Check(Pomade_ReadTubePoints(uc, before.data(),
                                       int(before.size())) == POMADE_OK,
                  "undo: pre-move points read");
            unsigned long long v0 = Pomade_GetVersion(uc);
            Check(Pomade_MoveCenterRing(uc, 2, 1.0f, 0.0f) == POMADE_OK,
                  "undo: move succeeds");
            Check(Pomade_GetUndoDepth(uc) == 1, "undo: move pushes one step");
            Check(Pomade_GetUndoBytes(uc) > 0, "undo: step accounts bytes");
            std::vector<float> moved = before;
            Check(Pomade_ReadTubePoints(uc, moved.data(),
                                       int(moved.size())) == POMADE_OK,
                  "undo: post-move points read");
            Check(moved != before, "undo: the move changed the tube");
            Check(Pomade_Undo(uc, nullptr) == POMADE_OK, "undo: undo succeeds");
            Check(Pomade_GetUndoDepth(uc) == 0, "undo: stack drains");
            Check(Pomade_GetUndoBytes(uc) == 0, "undo: bytes drain too");
            std::vector<float> restored = moved;
            Check(Pomade_ReadTubePoints(uc, restored.data(),
                                       int(restored.size())) == POMADE_OK,
                  "undo: post-undo points read");
            Check(restored == before, "undo: points restore bit-exactly");
            Check(Pomade_GetVersion(uc) == v0 + 2,
                  "undo: move + undo bump the version twice");
            // Failed validation pushes nothing.
            Check(Pomade_MoveCenterRing(uc, 99, 1.0f, 0.0f) == POMADE_ERROR,
                  "undo: bad move still fails");
            Check(Pomade_GetUndoDepth(uc) == 0,
                  "undo: failed moves push nothing");
            // Depth budget evicts oldest first.
            Check(Pomade_SetUndoBudget(uc, 2, 1ULL << 40) == POMADE_OK,
                  "undo: depth budget sets");
            Check(Pomade_MoveCenterRing(uc, 1, 0.5f, 0.0f) == POMADE_OK &&
                      Pomade_MoveCenterRing(uc, 2, 0.5f, 0.0f) == POMADE_OK &&
                      Pomade_MoveCenterRing(uc, 3, 0.5f, 0.0f) == POMADE_OK,
                  "undo: three moves run");
            Check(Pomade_GetUndoDepth(uc) == 2,
                  "undo: depth budget holds two");
            Check(Pomade_Undo(uc, nullptr) == POMADE_OK && Pomade_Undo(uc, nullptr) == POMADE_OK,
                  "undo: two undos run");
            Check(Pomade_GetUndoDepth(uc) == 0, "undo: budget stack drains");
            // Byte budget evicts just as eagerly.
            Check(Pomade_SetUndoBudget(uc, 50, 1) == POMADE_OK,
                  "undo: byte budget sets");
            Check(Pomade_MoveCenterRing(uc, 1, 0.5f, 0.0f) == POMADE_OK,
                  "undo: move runs under a byte budget");
            Check(Pomade_GetUndoDepth(uc) == 0 &&
                      Pomade_GetUndoBytes(uc) == 0,
                  "undo: a 1-byte budget evicts instantly");
            Check(Pomade_SetUndoBudget(uc, 50, 1ULL << 40) == POMADE_OK,
                  "undo: budget restores");
            Check(Pomade_SetUndoBudget(uc, -1, 1ULL << 40) == POMADE_ERROR,
                  "undo: negative depth rejected");
            // Disabled + explicit clear + clear-on-build.
            Check(Pomade_SetUndoBudget(uc, 0, 1ULL << 40) == POMADE_OK,
                  "undo: depth 0 disables");
            Check(Pomade_MoveCenterRing(uc, 1, 0.5f, 0.0f) == POMADE_OK,
                  "undo: moves run while disabled");
            Check(Pomade_GetUndoDepth(uc) == 0,
                  "undo: disabled pushes nothing");
            Check(Pomade_SetUndoBudget(uc, 50, 1ULL << 40) == POMADE_OK &&
                      Pomade_MoveCenterRing(uc, 1, 0.5f, 0.0f) == POMADE_OK,
                  "undo: re-enable + move");
            Check(Pomade_GetUndoDepth(uc) == 1, "undo: step held again");
            Check(Pomade_ClearUndo(uc) == POMADE_OK, "undo: explicit clear");
            Check(Pomade_GetUndoDepth(uc) == 0 &&
                      Pomade_GetUndoBytes(uc) == 0,
                  "undo: clear drains depth and bytes");
            Check(Pomade_MoveCenterRing(uc, 1, 0.5f, 0.0f) == POMADE_OK,
                  "undo: move runs before rebuild");
            Check(Pomade_BuildTestTube(uc, 0, 0, 0.0f, 0.0f) == POMADE_OK,
                  "undo: rebuild succeeds");
            Check(Pomade_GetUndoDepth(uc) == 0,
                  "undo: rebuild clears the stack");
            // Topology undo keeps the active-cut frontier usable. Expansion
            // is viewport state, so undoing a subdivision prunes only ids
            // that became leaves and retains an independent expanded branch.
            std::vector<int> kids(4);
            std::vector<int> grandA(3), grandB(2);
            int kidCount = 0, grandACount = 0, grandBCount = 0;
            Check(Pomade_SubdivideTube(uc, 0, 4, "kmeans", 7, kids.data(),
                                      int(kids.size()), &kidCount) == POMADE_OK &&
                      kidCount == 4 &&
                      Pomade_SubdivideTube(uc, kids[0], 3, "kmeans", 3,
                                          grandA.data(), int(grandA.size()),
                                          &grandACount) == POMADE_OK &&
                      grandACount == 3 &&
                      Pomade_SubdivideTube(uc, kids[1], 2, "kmeans", 5,
                                          grandB.data(), int(grandB.size()),
                                          &grandBCount) == POMADE_OK &&
                      grandBCount == 2 &&
                      Pomade_SetActiveCutEnabled(uc, 1) == POMADE_OK &&
                      Pomade_SetTubeExpanded(uc, 0, 1) == POMADE_OK &&
                      Pomade_SetTubeExpanded(uc, kids[0], 1) == POMADE_OK &&
                      Pomade_SetTubeExpanded(uc, kids[1], 1) == POMADE_OK &&
                      Pomade_IsTubeVisible(uc, grandA[0]) == 1 &&
                      Pomade_IsTubeVisible(uc, grandB[0]) == 1,
                  "undo: expanded sibling branches build a mixed-depth cut");
            Check(Pomade_Undo(uc, nullptr) == POMADE_OK &&
                      Pomade_GetTubeCount(uc) == 8 &&
                      Pomade_GetTubeExpanded(uc, 0) == 1 &&
                      Pomade_GetTubeExpanded(uc, kids[0]) == 1 &&
                      Pomade_GetTubeExpanded(uc, kids[1]) == 0 &&
                      Pomade_IsTubeVisible(uc, grandA[0]) == 1 &&
                      Pomade_IsTubeVisible(uc, kids[1]) == 1,
                  "undo: stale expanded leaf prunes while sibling cut stays live");
            Check(Pomade_Redo(uc, nullptr) == POMADE_OK &&
                      Pomade_GetTubeCount(uc) == 10 &&
                      Pomade_GetTubeExpanded(uc, 0) == 1 &&
                      Pomade_GetTubeExpanded(uc, kids[0]) == 1 &&
                      Pomade_GetTubeExpanded(uc, kids[1]) == 0 &&
                      Pomade_IsTubeVisible(uc, grandA[0]) == 1 &&
                      Pomade_IsTubeVisible(uc, kids[1]) == 1 &&
                      Pomade_IsTubeVisible(uc, grandB[0]) == 0,
                  "redo: restored branch is collapsed and frontier remains visible");
            Check(Pomade_Undo(uc, nullptr) == POMADE_OK &&
                      Pomade_Undo(uc, nullptr) == POMADE_OK &&
                      Pomade_Undo(uc, nullptr) == POMADE_OK &&
                      Pomade_GetTubeCount(uc) == 1 &&
                      Pomade_IsTubeVisible(uc, 0) == 1,
                  "undo: restoring the root cannot leave it hidden");
            Check(Pomade_Destroy(uc) == POMADE_OK, "undo: model destroys");
        }
    }

    // -- P6 OOM fallback --------------------------------------------------
    // A healthy model reports no reason (null context too). The forced
    // injection drops the mirror deterministically on CUDA builds while
    // the op still succeeds on the host mirror.
    Check(std::string(Pomade_GetDeviceFallbackReason(nullptr)).empty(),
          "fallback reason is empty on null context");
    {
        PomadeModelContext *healthy = nullptr;
        Check(Pomade_Create(&healthy) == POMADE_OK && healthy != nullptr,
              "P6: fresh model creates");
        if (healthy) {
            Check(Pomade_BuildTestTube(healthy, 0, 0, 0.0f, 0.0f) == POMADE_OK,
                  "P6: healthy build succeeds");
            Check(std::string(Pomade_GetDeviceFallbackReason(healthy)).empty(),
                  "P6: healthy model reports no fallback");
            Check(Pomade_Destroy(healthy) == POMADE_OK,
                  "P6: healthy model destroys");
        }
    }
#ifdef USDGEN_POMADE_HAS_CUDA
    {
#ifdef _WIN32
        _putenv("USDGEN_POMADE_FORCE_DEVICE_FALLBACK=1");
#else
        setenv("USDGEN_POMADE_FORCE_DEVICE_FALLBACK", "1", 1);
#endif
        PomadeModelContext *forced = nullptr;
        Check(Pomade_Create(&forced) == POMADE_OK && forced != nullptr,
              "P6: forced model creates");
        if (forced) {
            Check(Pomade_HasCudaMirror(forced) == 1,
                  "P6: mirror exists before the first sync");
            Check(Pomade_BuildTestTube(forced, 0, 0, 0.0f, 0.0f) == POMADE_OK,
                  "P6: build succeeds despite the forced drop");
            Check(Pomade_HasCudaMirror(forced) == 0,
                  "P6: the forced drop clears the mirror");
            std::string const reason =
                Pomade_GetDeviceFallbackReason(forced);
            Check(reason.find("USDGEN_POMADE_FORCE_DEVICE_FALLBACK") !=
                      std::string::npos,
                  "P6: the reason names the injection");
            std::vector<float> fxyz(
                size_t(Pomade_GetVertexCount(forced)) * 3);
            Check(Pomade_ReadTubePoints(forced, fxyz.data(),
                                       int(fxyz.size())) == POMADE_OK &&
                      std::abs(fxyz[0] - 0.5f) < 1e-6f,
                  "P6: staged reads serve the host mirror after fallback");
            Check(Pomade_MoveCenterRing(forced, 2, 1.0f, 0.0f) == POMADE_OK,
                  "P6: later edits keep working CPU-only");
            Check(std::string(Pomade_GetDeviceFallbackReason(forced)) == reason,
                  "P6: the reason is sticky");
            Check(Pomade_Destroy(forced) == POMADE_OK,
                  "P6: forced model destroys");
        }
#ifdef _WIN32
        _putenv("USDGEN_POMADE_FORCE_DEVICE_FALLBACK=");
#else
        unsetenv("USDGEN_POMADE_FORCE_DEVICE_FALLBACK");
#endif
    }
#else
    std::printf("skip: no CUDA for the forced device fallback\n");
#endif


    // -- V0 viewport publication (plan/18 sections 2.1, 2.2) ---------------
    //
    // No scene index is attached in this process, so Publish reaches zero of
    // them - which is exactly the contract: the count IS the result, and a
    // headless tool gets an honest 0 rather than a silent success. The
    // registry bookkeeping, the activation handshake and the level display
    // state are all provable here; testUsdGenPomadeIndex proves the same ABI
    // against a live index, and testUsdviewPomadePublish against Hydra.
    {
        usdGenPomade::PomadeRegistry &registry =
            usdGenPomade::PomadeRegistry::Get();
        Check(Pomade_GetModelId(nullptr) == 0,
              "V0: Pomade_GetModelId(null) is 0");
        Check(Pomade_Activate(nullptr) == POMADE_ERROR,
              "V0: Pomade_Activate rejects null");
        Check(Pomade_Publish(nullptr, 0) == -1,
              "V0: Pomade_Publish reports -1 on null, not a count");
        PomadeModelContext *a = nullptr;
        PomadeModelContext *b = nullptr;
        Check(Pomade_Create(&a) == POMADE_OK && Pomade_Create(&b) == POMADE_OK,
              "V0: two models create");
        if (a && b) {
            int const idA = Pomade_GetModelId(a);
            int const idB = Pomade_GetModelId(b);
            Check(idA > 0 && idB > 0 && idA != idB,
                  "V0: every model gets its own registry id");
            Check(registry.ModelCount() == 2,
                  "V0: the registry holds both models");
            Check(registry.GetActiveId() == 0,
                  "V0: creating a model does not activate it");
            Check(Pomade_Activate(a) == POMADE_OK, "V0: activate the first");
            Check(registry.GetActiveId() == idA,
                  "V0: the registry records the active model");
            Check(Pomade_Publish(a, 0) == 0,
                  "V0: publish with no attached index refreshes none");
            Check(Pomade_Activate(b) == POMADE_OK,
                  "V0: activating the second takes over");
            Check(registry.GetActiveId() == idB,
                  "V0: only one model is active at a time");
            Check(Pomade_Deactivate(b) == POMADE_OK, "V0: deactivate");
            Check(registry.GetActiveId() == 0, "V0: nothing is active");

            // Level display: defaults, round trip, rejection, focus.
            int visible = -1;
            int xray = -1;
            Check(Pomade_GetLevelDisplay(a, 3, &visible, &xray) == POMADE_OK &&
                      visible == 1 && xray == 0,
                  "V0: an untouched level reads visible and opaque");
            Check(Pomade_SetLevelDisplay(a, 2, 0, 1) == POMADE_OK,
                  "V0: hide level 2 in x-ray");
            Check(Pomade_GetLevelDisplay(a, 2, &visible, &xray) == POMADE_OK &&
                      visible == 0 && xray == 1,
                  "V0: level display round-trips");
            Check(Pomade_SetLevelDisplay(a, 0, 1, 0) == POMADE_ERROR,
                  "V0: level 0 is rejected");
            Check(Pomade_GetLevelDisplay(a, 1, &visible, &xray) == POMADE_OK &&
                      visible == 1 && xray == 0,
                  "V0: setting one level leaves the others alone");
            unsigned long long const before = Pomade_GetVersion(a);
            Check(Pomade_SetLevelDisplay(a, 2, 0, 1) == POMADE_OK &&
                      Pomade_GetVersion(a) == before,
                  "V0: a no-op display set does not bump the version");
            Check(Pomade_SetFocusLevel(a, 2) == POMADE_OK &&
                      Pomade_GetFocusLevel(a) == 2,
                  "V0: focus level round-trips");
            Check(Pomade_GetVersion(a) == before + 1,
                  "V0: a real display change bumps the version");
            Check((Pomade_TakeDirty(a) &
                   usdGenPomade::PomadeDirty_Display) != 0,
                  "V0: a display change marks Display dirty");
            Check(Pomade_SetFocusLevel(a, -1) == POMADE_ERROR,
                  "V0: a negative focus level is rejected");
            Check(Pomade_SetFocusLevel(a, 0) == POMADE_OK &&
                      Pomade_GetFocusLevel(a) == 0,
                  "V0: focus level 0 focuses nothing");

            // V8 (plan/18 §2.4a): world units per screen pixel, which is
            // what turns the overlay pixel targets into the world widths
            // Storm draws points and curves with.
            {
                float scale = -1.0f;
                Check(Pomade_GetDisplayScale(a, &scale) == POMADE_OK &&
                          scale == 0.0f,
                      "V8: a fresh model has no display scale");
                unsigned long long const v = Pomade_GetVersion(a);
                Check(Pomade_SetDisplayScale(a, 0.0125f) == POMADE_OK &&
                          Pomade_GetDisplayScale(a, &scale) == POMADE_OK &&
                          scale == 0.0125f,
                      "V8: the display scale round-trips");
                Check(Pomade_GetVersion(a) == v + 1 &&
                          (Pomade_TakeDirty(a) &
                           usdGenPomade::PomadeDirty_Display) != 0,
                      "V8: setting it bumps the version and marks Display "
                      "dirty");
                Check(Pomade_SetDisplayScale(a, 0.0125f) == POMADE_OK &&
                          Pomade_GetVersion(a) == v + 1,
                      "V8: setting the same scale again publishes nothing");
                Check(Pomade_SetDisplayScale(a, -0.5f) == POMADE_ERROR,
                      "V8: a negative scale is refused");
                Check(Pomade_SetDisplayScale(nullptr, 0.01f) == POMADE_ERROR &&
                          Pomade_GetDisplayScale(a, nullptr) == POMADE_ERROR,
                      "V8: null arguments are refused, not crashed on");
                Check(Pomade_SetDisplayScale(a, 0.0f) == POMADE_OK &&
                          Pomade_GetDisplayScale(a, &scale) == POMADE_OK &&
                          scale == 0.0f,
                      "V8: 0 puts it back to the no-camera sizing");
            }

            // With no index attached there is nothing published to query.
            int faces = 0;
            Check(Pomade_GetPublishedLevelInfo(a, 1, &faces, nullptr,
                                              nullptr) == POMADE_ERROR,
                  "V0: no attached index means no published level");

            Check(Pomade_Destroy(a) == POMADE_OK, "V0: first model destroys");
            Check(registry.ModelCount() == 1,
                  "V0: destroying unregisters exactly one model");
            Check(Pomade_Activate(b) == POMADE_OK, "V0: reactivate the second");
            Check(Pomade_Destroy(b) == POMADE_OK, "V0: second model destroys");
            Check(registry.ModelCount() == 0 && registry.GetActiveId() == 0,
                  "V0: destroying the active model clears the registry");
        }
    }

    // -- V1 selection, overlays, gestures (plan/18 §2.3-§2.5) -------------
    //
    // The pick candidate sets are the model's own (tube 0's surface,
    // centers and rings; every graph node, edge and region; the guide
    // preview), so a marquee and a click can only ever disagree if the
    // projection disagrees — which is the point of testing them against
    // each other here rather than against hand-computed pixels.
    {
        PomadeModelContext *sc = nullptr;
        Check(Pomade_Create(&sc) == POMADE_OK && sc != nullptr,
              "V1: selection model creates");
        Check(Pomade_BuildTestTube(sc, 0, 0, 0.0f, 0.0f) == POMADE_OK,
              "V1: test tube builds");

        // A whole-tube press is a surface pick, not a proximity test against
        // its tessellation vertices.  Pick the centroid of a visible quad
        // triangle: it is deliberately many pixels from every vertex while
        // still inside the rendered wall.
        {
            int const pickW = 800, pickH = 600;
            float faceView[16] = {0.0f};
            faceView[0] = 1.0f;   // screen x <- world x
            faceView[5] = 0.5f;   // world y [0,4] -> NDC [-1,1]
            faceView[10] = 1.0f;  // depth <- world z
            faceView[13] = -1.0f;
            faceView[15] = 1.0f;
            std::vector<float> points(
                size_t(Pomade_GetVertexCount(sc)) * 3);
            bool const read = Pomade_ReadTubePoints(
                sc, points.data(), int(points.size())) == POMADE_OK;
            bool broadFace = false;
            bool rawVertexMiss = false;
            if (read) {
                // Ring 0, slots 0/1 and ring 1, slot 1 form a large,
                // non-degenerate side triangle on the default 5x8 tube.
                float sx[3] = {0.0f}, sy[3] = {0.0f}, depth = 0.0f;
                bool projected = true;
                int const tri[3] = {0, 1, 9};
                for (int i = 0; i < 3; ++i) {
                    projected = projected && usdGenPomade::PomadeProjectPoint(
                        points.data() + size_t(tri[i]) * 3, faceView, pickW,
                        pickH, &sx[i], &sy[i], &depth);
                }
                float const px = (sx[0] + sx[1] + sx[2]) / 3.0f;
                float const py = (sy[0] + sy[1] + sy[2]) / 3.0f;
                float nearest = 1e30f;
                for (size_t i = 0; projected && i < points.size() / 3; ++i) {
                    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
                    if (!usdGenPomade::PomadeProjectPoint(
                            points.data() + i * 3, faceView, pickW, pickH,
                            &vx, &vy, &vz)) {
                        projected = false;
                        break;
                    }
                    float const dx = vx - px, dy = vy - py;
                    nearest = std::min(nearest, std::sqrt(dx * dx + dy * dy));
                }
                int hit = 0, id = -1, sub = -1, subSub = -1;
                unsigned int kind = 0;
                broadFace = projected && nearest > 4.0f &&
                    Pomade_PickItem(sc, faceView, pickW, pickH, px, py, 4.0f,
                                   usdGenPomade::PomadePick_TubeVert, &hit,
                                   &kind, &id, &sub, &subSub) == POMADE_OK &&
                    hit == 1 && kind == usdGenPomade::PomadePick_TubeVert &&
                    id == 0 && sub == -1 && subSub == -1;
                float dist = -1.0f, rawDepth = -1.0f;
                rawVertexMiss = projected && nearest > 4.0f &&
                    Pomade_Pick(sc, faceView, pickW, pickH, px, py, 4.0f,
                               usdGenPomade::PomadePick_TubeVert, &hit, &kind,
                               &id, &sub, &dist, &rawDepth) == POMADE_OK &&
                    hit == 0;
            }
            Check(broadFace,
                  "V1: a large quad interior picks its tube beyond vertex snap radius");
            Check(rawVertexMiss,
                  "V1: raw K11 remains a vertex-distance pick at quad interior");

            // A component handle keeps priority even when it lies exactly on
            // a surface vertex, so sculpt can select one center CV.
            float center[3] = {0.0f, 0.0f, 0.0f};
            float cx = 0.0f, cy = 0.0f, cz = 0.0f;
            int hit = 0, id = -1, sub = -1, subSub = -1;
            unsigned int kind = 0;
            Check(Pomade_GetTubeCenterHandle(sc, 0, 2, center) == POMADE_OK &&
                      usdGenPomade::PomadeProjectPoint(center, faceView, pickW,
                                                      pickH, &cx, &cy, &cz) &&
                      Pomade_PickItem(sc, faceView, pickW, pickH, cx, cy, 4.0f,
                                     usdGenPomade::PomadePick_CenterCV |
                                         usdGenPomade::PomadePick_TubeVert,
                                     &hit, &kind, &id, &sub, &subSub) ==
                          POMADE_OK &&
                      hit == 1 && kind == usdGenPomade::PomadePick_CenterCV &&
                      id == 0 && sub == 2,
                  "V1: a center CV beats an overlapping tube surface");
            Check(Pomade_PickItem(sc, faceView, pickW, pickH, 10.0f, 10.0f,
                                 4.0f, usdGenPomade::PomadePick_TubeVert, &hit,
                                 &kind, &id, &sub, &subSub) == POMADE_OK &&
                      hit == 0,
                  "V1: empty screen space remains a tube-surface miss");
        }

        // -- round trips, one kind at a time ------------------------------
        int ids[4] = {0, 0, 0, 0};
        int subIds[4] = {1, 3, -1, -1};
        int count = -1;
        Check(Pomade_GetSelectionCount(sc, 0) == 0,
              "V1: a fresh model has nothing selected");
        Check(Pomade_SelectSet(sc, usdGenPomade::PomadePick_CenterCV, ids, subIds, nullptr,
                              2) == POMADE_OK,
              "V1: select two center CVs");
        Check(Pomade_GetSelectionCount(sc, usdGenPomade::PomadePick_CenterCV) == 2,
              "V1: two center CVs are selected");
        {
            int outIds[8] = {0};
            int outSubs[8] = {0};
            Check(Pomade_ReadSelection(sc, usdGenPomade::PomadePick_CenterCV, outIds,
                                      outSubs, nullptr, 8,
                                      &count) == POMADE_OK &&
                      count == 2 && outIds[0] == 0 && outIds[1] == 0 &&
                      outSubs[0] == 1 && outSubs[1] == 3,
                  "V1: center CVs read back ascending");
            Check(Pomade_ReadSelection(sc, usdGenPomade::PomadePick_CenterCV, nullptr,
                                      nullptr, nullptr, 0,
                                      &count) == POMADE_OK && count == 2,
                  "V1: a null-array read sizes the buffer");
            Check(Pomade_ReadSelection(sc, usdGenPomade::PomadePick_CenterCV, outIds, nullptr,
                                      nullptr, 1, &count) == POMADE_ERROR,
                  "V1: a short buffer is an honest error");
        }
        // Set replaces its own kind and leaves the others alone.
        int nodeIds[2] = {7, 9};
        Check(Pomade_SelectSet(sc, usdGenPomade::PomadePick_GraphNode, nodeIds, nullptr,
                              nullptr, 2) == POMADE_OK,
              "V1: select two graph nodes");
        Check(Pomade_GetSelectionCount(sc, usdGenPomade::PomadePick_CenterCV) == 2 &&
                  Pomade_GetSelectionCount(sc, usdGenPomade::PomadePick_GraphNode) == 2 &&
                  Pomade_GetSelectionCount(sc, 0) == 4,
              "V1: setting one kind leaves the others selected");
        Check(Pomade_SelectSet(sc, usdGenPomade::PomadePick_CenterCV, ids, subIds, nullptr,
                              1) == POMADE_OK &&
                  Pomade_GetSelectionCount(sc, usdGenPomade::PomadePick_CenterCV) == 1,
              "V1: Set replaces the kind it names");
        Check(Pomade_SelectAdd(sc, usdGenPomade::PomadePick_CenterCV, ids, subIds + 1,
                              nullptr, 1) == POMADE_OK &&
                  Pomade_GetSelectionCount(sc, usdGenPomade::PomadePick_CenterCV) == 2,
              "V1: Add unions");
        Check(Pomade_SelectToggle(sc, usdGenPomade::PomadePick_CenterCV, ids, subIds + 1,
                                 nullptr, 1) == POMADE_OK &&
                  Pomade_GetSelectionCount(sc, usdGenPomade::PomadePick_CenterCV) == 1,
              "V1: Toggle removes what was there");
        Check(Pomade_SelectClear(sc, usdGenPomade::PomadePick_GraphNode) == POMADE_OK &&
                  Pomade_GetSelectionCount(sc, usdGenPomade::PomadePick_GraphNode) == 0 &&
                  Pomade_GetSelectionCount(sc, usdGenPomade::PomadePick_CenterCV) == 1,
              "V1: Clear takes exactly the kinds in its mask");
        // A section CV needs all three ids to round-trip.
        {
            int sid[1] = {0};
            int ring[1] = {2};
            int slot[1] = {5};
            int oid[4] = {0}, osub[4] = {0}, osubsub[4] = {0};
            Check(Pomade_SelectSet(sc, usdGenPomade::PomadePick_SectionCV, sid, ring, slot,
                                  1) == POMADE_OK,
                  "V1: select one section CV");
            Check(Pomade_ReadSelection(sc, usdGenPomade::PomadePick_SectionCV, oid, osub,
                                      osubsub, 4, &count) == POMADE_OK &&
                      count == 1 && oid[0] == 0 && osub[0] == 2 &&
                      osubsub[0] == 5,
                  "V1: a section CV round-trips (tube, ring, slot)");
        }
        Check(Pomade_SelectSet(sc, usdGenPomade::PomadePick_CenterCV | usdGenPomade::PomadePick_GraphNode,
                              ids, nullptr, nullptr, 1) == POMADE_ERROR,
              "V1: a select call takes exactly one kind");
        Check(Pomade_SelectClear(sc, 0) == POMADE_OK &&
                  Pomade_GetSelectionCount(sc, 0) == 0,
              "V1: mask 0 clears everything");

        // -- hover is separate from the selection -------------------------
        unsigned int hoverKind = 0;
        int hoverId = -1, hoverSub = -1, hoverSubSub = -1;
        Check(Pomade_SetHover(sc, usdGenPomade::PomadePick_CenterCV, 0, 3, -1) == POMADE_OK,
              "V1: hover a center CV");
        Check(Pomade_GetHover(sc, &hoverKind, &hoverId, &hoverSub,
                             &hoverSubSub) == POMADE_OK &&
                  hoverKind == usdGenPomade::PomadePick_CenterCV && hoverId == 0 &&
                  hoverSub == 3,
              "V1: hover reads back");
        Check(Pomade_GetSelectionCount(sc, 0) == 0,
              "V1: hover is not a selection");
        Check(Pomade_SetHover(sc, 0, -1, -1, -1) == POMADE_OK &&
                  Pomade_GetHover(sc, &hoverKind, nullptr, nullptr,
                                 nullptr) == POMADE_OK && hoverKind == 0,
              "V1: kind 0 clears the hover");

        // -- the version only moves on a real change ----------------------
        {
            unsigned long long const v = Pomade_GetVersion(sc);
            Check(Pomade_SelectSet(sc, usdGenPomade::PomadePick_CenterCV, ids, subIds,
                                  nullptr, 1) == POMADE_OK &&
                      Pomade_GetVersion(sc) == v + 1,
                  "V1: a real selection change bumps the version");
            Check(Pomade_SelectSet(sc, usdGenPomade::PomadePick_CenterCV, ids, subIds,
                                  nullptr, 1) == POMADE_OK &&
                      Pomade_GetVersion(sc) == v + 1,
                  "V1: re-selecting the same CV changes nothing");
            Check((Pomade_TakeDirty(sc) &
                   usdGenPomade::PomadeDirty_Selection) != 0,
                  "V1: a selection change marks Selection dirty");
        }

        // -- selection bounds ---------------------------------------------
        {
            float mn[3] = {9.0f, 9.0f, 9.0f};
            float mx[3] = {-9.0f, -9.0f, -9.0f};
            float cv[3] = {0.0f, 0.0f, 0.0f};
            Check(Pomade_GetTubeCenterHandle(sc, 0, 1, cv) == POMADE_OK,
                  "V1: center CV 1 display handle reads");
            Check(Pomade_GetSelectionBounds(sc, mn, mx) == POMADE_OK &&
                      std::abs(mn[1] - cv[1]) < 1e-6f &&
                      std::abs(mx[1] - cv[1]) < 1e-6f,
                  "V1: one selected CV bounds exactly itself");
            Check(Pomade_SelectClear(sc, 0) == POMADE_OK &&
                      Pomade_GetSelectionBounds(sc, mn, mx) == POMADE_ERROR,
                  "V1: an empty selection has no bounds");
        }

        // -- marquee and lasso agree with the point pick ------------------
        //
        // The model is subdivided first: the children are real tubes in
        // the hierarchy, and the candidate sets span every live tube
        // (centers and surface; sections stay tube 0's per the V0b G2
        // note, since section edits on derived children are refused by
        // design). What this proves is that the rubber band and the
        // click read the SAME candidates -- and name the tube each one
        // belongs to.
        {
            int kids[4] = {0};
            int kidCount = 0;
            Check(Pomade_SubdivideTube(sc, 0, 4, "kmeans", 7, kids, 4,
                                      &kidCount) == POMADE_OK &&
                      kidCount == 4,
                  "V1: the tube subdivides for the marquee test");
            int const w = 800, h = 600;
            // Row-major, row-vector (the gpu/picking.cu convention): y in
            // [0, 4] maps to NDC [-1, 1], x and z pass through.
            float viewProj[16] = {0.0f};
            viewProj[0] = 1.0f;
            viewProj[5] = 0.5f;
            viewProj[10] = 1.0f;
            viewProj[13] = -1.0f;
            viewProj[15] = 1.0f;

            // Section candidates are flattened while being picked, but must
            // recover their owning child tube, ring and slot.  Derive a
            // child's actual world-space section CV through the public
            // section/frame contract, then require point pick, marquee and
            // selection bounds to agree on that same child item.  Trying
            // every slot keeps this independent of the deterministic split
            // polygon's vertex ordering while still requiring one isolated
            // child CV to survive all three paths.
            bool childSectionRoundTrip = false;
            bool childSectionPointPick = false;
            bool childSectionRect = false;
            bool childSectionBounds = false;
            int childLastHit = -1, childLastId = -1, childLastRing = -1,
                childLastSlot = -1;
            // Use an oblique view for this test: the earlier side view is
            // ideal for center CV spacing, but collapses a section ring's
            // z coordinate. Looking down the tube instead collapses its
            // ring heights. This projection keeps both dimensions apart.
            float sectionViewProj[16] = {0.0f};
            sectionViewProj[0] = 0.75f;
            sectionViewProj[8] = 0.25f;  // screen x <- x + z
            sectionViewProj[1] = 0.20f;
            sectionViewProj[5] = 0.40f;
            sectionViewProj[9] = 0.30f;  // screen y <- x + y + z
            sectionViewProj[6] = 0.10f;  // depth <- world y
            sectionViewProj[15] = 1.0f;
            int const childId = kids[0];
            int const childRing = 1;
            int childSlots = 0;
            float childT = 0.0f, childScale = 1.0f, childTwist = 0.0f;
            if (Pomade_GetTubeSection(sc, childId, childRing, &childT,
                                     nullptr, 0, &childSlots, &childScale,
                                     &childTwist) == POMADE_OK &&
                childSlots > 0) {
                std::vector<float> childUv(size_t(childSlots) * 2);
                float childOrigin[3] = {0.0f, 0.0f, 0.0f};
                float childFrame[9] = {0.0f};
                bool const readChild =
                    Pomade_GetTubeSection(sc, childId, childRing, nullptr,
                                         childUv.data(), int(childUv.size()),
                                         &childSlots, &childScale,
                                         &childTwist) == POMADE_OK &&
                    Pomade_GetTubeSectionFrame(sc, childId, childRing,
                                              childOrigin, childFrame,
                                              &childScale, &childTwist) ==
                        POMADE_OK;
                if (readChild) {
                    float meanU = 0.0f, meanV = 0.0f;
                    float const ct = std::cos(childTwist);
                    float const st = std::sin(childTwist);
                    for (int slot = 0; slot < childSlots; ++slot) {
                        float const u = childUv[size_t(slot) * 2] * childScale;
                        float const v = childUv[size_t(slot) * 2 + 1] * childScale;
                        meanU += u * ct - v * st;
                        meanV += u * st + v * ct;
                    }
                    meanU /= float(childSlots);
                    meanV /= float(childSlots);
                    for (int slot = 0; slot < childSlots &&
                                       !childSectionRoundTrip; ++slot) {
                        float const u = childUv[size_t(slot) * 2] * childScale;
                        float const v = childUv[size_t(slot) * 2 + 1] * childScale;
                        float const ru = u * ct - v * st;
                        float const rv = u * st + v * ct;
                        float point[3] = {
                            childOrigin[0] + childFrame[0] * (ru - meanU) +
                                childFrame[3] * (rv - meanV),
                            childOrigin[1] + childFrame[1] * (ru - meanU) +
                                childFrame[4] * (rv - meanV),
                            childOrigin[2] + childFrame[2] * (ru - meanU) +
                                childFrame[5] * (rv - meanV)};
                        float sx = 0.0f, sy = 0.0f, depth = 0.0f;
                        int childHit = 0, childPickId = -1,
                            childPickRing = -1, childPickSlot = -1;
                        unsigned int childKind = 0;
                        if (!usdGenPomade::PomadeProjectPoint(point, sectionViewProj,
                                                            w, h, &sx, &sy,
                                                            &depth) ||
                            Pomade_PickItem(sc, sectionViewProj, w, h, sx, sy, 0.25f,
                                           usdGenPomade::PomadePick_SectionCV,
                                           &childHit, &childKind, &childPickId,
                                           &childPickRing, &childPickSlot) !=
                                POMADE_OK) {
                            continue;
                        }
                        childLastHit = childHit;
                        childLastId = childPickId;
                        childLastRing = childPickRing;
                        childLastSlot = childPickSlot;
                        if (
                            childHit != 1 ||
                            childKind != usdGenPomade::PomadePick_SectionCV ||
                            childPickId != childId ||
                            childPickRing != childRing ||
                            childPickSlot != slot) {
                            continue;
                        }
                        childSectionPointPick = true;
                        int pickedCount = 0;
                        float mn[3] = {0.0f}, mx[3] = {0.0f};
                        childSectionRect =
                            Pomade_SelectRect(sc, sectionViewProj, w, h,
                                             sx - 0.25f, sy - 0.25f,
                                             sx + 0.25f, sy + 0.25f,
                                             usdGenPomade::PomadePick_SectionCV,
                                             POMADE_SELECT_SET) == POMADE_OK &&
                            Pomade_ReadSelection(sc,
                                                usdGenPomade::PomadePick_SectionCV,
                                                nullptr, nullptr, nullptr, 0,
                                                &pickedCount) == POMADE_OK &&
                            pickedCount > 0;
                        if (childSectionRect) {
                            std::vector<int> pickedIds(size_t(pickedCount), 0);
                            std::vector<int> pickedRings(size_t(pickedCount), 0);
                            std::vector<int> pickedSlots(size_t(pickedCount), 0);
                            childSectionRect =
                                Pomade_ReadSelection(
                                    sc, usdGenPomade::PomadePick_SectionCV,
                                    pickedIds.data(), pickedRings.data(),
                                    pickedSlots.data(), pickedCount,
                                    &pickedCount) == POMADE_OK;
                            bool found = false;
                            for (int i = 0; i < pickedCount; ++i) {
                                found = found ||
                                    (pickedIds[size_t(i)] == childId &&
                                     pickedRings[size_t(i)] == childRing &&
                                     pickedSlots[size_t(i)] == slot);
                            }
                            childSectionRect = childSectionRect && found;
                        }
                        if (!childSectionRect) {
                            continue;
                        }
                        int const selectedId = childId;
                        int const selectedRing = childRing;
                        int const selectedSlot = slot;
                        childSectionBounds =
                            Pomade_SelectSet(sc,
                                            usdGenPomade::PomadePick_SectionCV,
                                            &selectedId, &selectedRing,
                                            &selectedSlot, 1) == POMADE_OK &&
                            Pomade_GetSelectionBounds(sc, mn, mx) == POMADE_OK &&
                            std::abs(mn[0] - point[0]) < 1e-4f &&
                            std::abs(mn[1] - point[1]) < 1e-4f &&
                            std::abs(mn[2] - point[2]) < 1e-4f &&
                            std::abs(mx[0] - point[0]) < 1e-4f &&
                            std::abs(mx[1] - point[1]) < 1e-4f &&
                            std::abs(mx[2] - point[2]) < 1e-4f;
                        childSectionRoundTrip = childSectionBounds;
                    }
                }
            }
            std::printf("info: child section point=%d rect=%d bounds=%d last=(%d,%d,%d,%d)\n",
                        childSectionPointPick, childSectionRect,
                        childSectionBounds, childLastHit, childLastId,
                        childLastRing, childLastSlot);
            Check(childSectionRoundTrip,
                  "V1: child section CV point pick, marquee and bounds agree");
            float px[8] = {0.0f}, py[8] = {0.0f};
            int const cvCount = Pomade_GetCenterCVCount(sc);
            Check(cvCount == 5, "V1: the test tube has five center CVs");
            bool projected = cvCount > 0;
            for (int i = 0; i < cvCount && i < 8; ++i) {
                float cv[3] = {0.0f, 0.0f, 0.0f};
                float z = 0.0f;
                projected = projected &&
                            Pomade_GetTubeCenterHandle(sc, 0, i, cv) == POMADE_OK &&
                            usdGenPomade::PomadeProjectPoint(
                                cv, viewProj, w, h, &px[i], &py[i], &z);
            }
            Check(projected, "V1: every center CV projects into the frame");
            // A three-pixel box around CV 2 catches that CV and no other
            // (the CVs are 150 px apart in this frame).
            Check(Pomade_SelectRect(sc, viewProj, w, h, px[2] - 3.0f,
                                   py[2] - 3.0f, px[2] + 3.0f, py[2] + 3.0f,
                                   usdGenPomade::PomadePick_CenterCV,
                                   POMADE_SELECT_SET) == POMADE_OK,
                  "V1: marquee over one CV");
            int oid[64] = {0}, osub[64] = {0};
            Check(Pomade_ReadSelection(sc, usdGenPomade::PomadePick_CenterCV, oid, osub,
                                      nullptr, 64, &count) == POMADE_OK &&
                      count == 1 && oid[0] == 0 && osub[0] == 2,
                  "V1: the marquee caught exactly center CV 2");
            int hit = 0, pid = -1, psub = -1, pss = -1;
            unsigned int pkind = 0;
            Check(Pomade_PickItem(sc, viewProj, w, h, px[2], py[2], 4.0f,
                                 usdGenPomade::PomadePick_CenterCV, &hit, &pkind, &pid,
                                 &psub, &pss) == POMADE_OK &&
                      hit == 1 && pkind == usdGenPomade::PomadePick_CenterCV && pid == 0 &&
                      psub == 2,
                  "V1: a click at that pixel names the same item");
            // The lasso over the same box has to agree with the marquee.
            float const lasso[8] = {px[2] - 3.0f, py[2] - 3.0f,
                                    px[2] + 3.0f, py[2] - 3.0f,
                                    px[2] + 3.0f, py[2] + 3.0f,
                                    px[2] - 3.0f, py[2] + 3.0f};
            Check(Pomade_SelectPolygon(sc, viewProj, w, h, lasso, 4,
                                      usdGenPomade::PomadePick_CenterCV,
                                      POMADE_SELECT_SET) == POMADE_OK,
                  "V1: lasso over the same box");
            Check(Pomade_ReadSelection(sc, usdGenPomade::PomadePick_CenterCV, oid, osub,
                                      nullptr, 64, &count) == POMADE_OK &&
                      count == 1 && osub[0] == 2,
                  "V1: the lasso agrees with the marquee");
            Check(Pomade_SelectPolygon(sc, viewProj, w, h, lasso, 2,
                                      usdGenPomade::PomadePick_CenterCV,
                                      POMADE_SELECT_SET) == POMADE_ERROR,
                  "V1: a two-point lasso is rejected");
            // A band over the whole frame catches every CV of every
            // tube, and each one picks back to an item that names its
            // own tube.
            int allTubes[8] = {0, kids[0], kids[1], kids[2], kids[3],
                               -1, -1, -1};
            int allCount = 0;
            for (int t = 0; t < 5; ++t) {
                allCount += Pomade_GetTubeCenterCount(sc, allTubes[t]);
            }
            Check(Pomade_SelectRect(sc, viewProj, w, h, -1e4f, -1e4f, 1e4f,
                                   1e4f, usdGenPomade::PomadePick_CenterCV,
                                   POMADE_SELECT_SET) == POMADE_OK &&
                      Pomade_ReadSelection(sc, usdGenPomade::PomadePick_CenterCV, oid, osub,
                                          nullptr, 64, &count) == POMADE_OK &&
                      count == allCount,
                  "V1: a full-frame marquee catches every center CV");
            bool agree = true;
            for (int t = 0; t < 5 && agree; ++t) {
                int const nCv = Pomade_GetTubeCenterCount(sc, allTubes[t]);
                for (int i = 0; i < nCv && agree; ++i) {
                    float ccv[3] = {0.0f, 0.0f, 0.0f};
                    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
                    agree = agree &&
                            Pomade_GetTubeCenterHandle(sc, allTubes[t], i, ccv) ==
                                POMADE_OK &&
                            usdGenPomade::PomadeProjectPoint(
                                ccv, viewProj, w, h, &cx, &cy, &cz);
                    hit = 0;
                    pid = -1;
                    psub = -1;
                    agree = agree &&
                            Pomade_PickItem(sc, viewProj, w, h, cx, cy, 4.0f,
                                           usdGenPomade::PomadePick_CenterCV, &hit, &pkind, &pid,
                                           &psub, &pss) == POMADE_OK &&
                            hit == 1 && pid == allTubes[t] && psub == i;
                }
            }
            Check(agree,
                  "V1: every marqueed CV is what a click there would pick");
            // Many surface candidates collapse into one item per tube.
            int surfIds[8] = {0};
            Check(Pomade_SelectRect(sc, viewProj, w, h, -1e4f, -1e4f, 1e4f,
                                   1e4f, usdGenPomade::PomadePick_TubeVert,
                                   POMADE_SELECT_SET) == POMADE_OK &&
                      Pomade_ReadSelection(sc, usdGenPomade::PomadePick_TubeVert, surfIds,
                                          nullptr, nullptr, 8,
                                          &count) == POMADE_OK &&
                      count == 5 && surfIds[0] == 0 &&
                      surfIds[1] == kids[0] && surfIds[2] == kids[1] &&
                      surfIds[3] == kids[2] && surfIds[4] == kids[3],
                  "V1: a band over the surface selects every tube once");
            // An empty band in SET mode deselects.
            Check(Pomade_SelectRect(sc, viewProj, w, h, 1e4f, 1e4f, 1e4f + 1.0f,
                                   1e4f + 1.0f, usdGenPomade::PomadePick_CenterCV,
                                   POMADE_SELECT_SET) == POMADE_OK &&
                      Pomade_GetSelectionCount(sc, usdGenPomade::PomadePick_CenterCV) == 0,
                  "V1: an empty band clears the kind it was asked for");

            // -- a subdivide moves the selection to the children ----------
            Check(Pomade_SelectSet(sc, usdGenPomade::PomadePick_TubeVert, ids, nullptr,
                                  nullptr, 1) == POMADE_OK,
                  "V1: select the parent tube");
            int grandKids[4] = {0};
            int grandCount = 0;
            Check(Pomade_SubdivideTube(sc, kids[0], 3, "kmeans", 3, grandKids,
                                      4, &grandCount) == POMADE_OK &&
                      grandCount == 3,
                  "V1: subdivide a child (the parent stays selected)");
            Check(Pomade_GetSelectionCount(sc, usdGenPomade::PomadePick_TubeVert) == 1,
                  "V1: subdividing another tube leaves the selection alone");

            // The per-branch active cut is opt-in: old callers still see
            // every live descriptor, while hierarchy navigation replaces an
            // expanded parent by its direct children and leaves sibling
            // branches alone. It must ignore the legacy numeric focus level
            // because this frontier intentionally mixes L2 and L3 tubes.
            float hiddenChild[3] = {0.0f, 0.0f, 0.0f};
            float hiddenX = 0.0f, hiddenY = 0.0f, hiddenDepth = 0.0f;
            int activeHit = 0, activeId = -1, activeSub = -1,
                activeSubSub = -1;
            unsigned int activeKind = 0;
            bool const activeCut =
                Pomade_GetActiveCutEnabled(sc) == 0 &&
                Pomade_IsTubeVisible(sc, 0) == 1 &&
                Pomade_IsTubeVisible(sc, kids[0]) == 1 &&
                Pomade_GetTubeCenterHandle(sc, kids[0], 0, hiddenChild) == POMADE_OK &&
                usdGenPomade::PomadeProjectPoint(hiddenChild, viewProj, w, h,
                                                &hiddenX, &hiddenY,
                                                &hiddenDepth) &&
                Pomade_SetActiveCutEnabled(sc, 1) == POMADE_OK &&
                Pomade_GetActiveCutEnabled(sc) == 1 &&
                Pomade_IsTubeVisible(sc, 0) == 1 &&
                Pomade_IsTubeVisible(sc, kids[0]) == 0 &&
                Pomade_IsTubeVisible(sc, kids[1]) == 0 &&
                Pomade_PickItem(sc, viewProj, w, h, hiddenX, hiddenY, 4.0f,
                               usdGenPomade::PomadePick_CenterCV, &activeHit,
                               &activeKind, &activeId, &activeSub,
                               &activeSubSub) == POMADE_OK &&
                !(activeHit == 1 && activeKind ==
                                         usdGenPomade::PomadePick_CenterCV &&
                  activeId == kids[0]) &&
                Pomade_SetTubeExpanded(sc, 0, 1) == POMADE_OK &&
                Pomade_GetTubeExpanded(sc, 0) == 1 &&
                Pomade_IsTubeVisible(sc, 0) == 0 &&
                Pomade_IsTubeVisible(sc, kids[0]) == 1 &&
                Pomade_IsTubeVisible(sc, kids[1]) == 1 &&
                Pomade_PickItem(sc, viewProj, w, h, hiddenX, hiddenY, 4.0f,
                               usdGenPomade::PomadePick_CenterCV, &activeHit,
                               &activeKind, &activeId, &activeSub,
                               &activeSubSub) == POMADE_OK &&
                activeHit == 1 &&
                activeKind == usdGenPomade::PomadePick_CenterCV &&
                activeId == kids[0] &&
                Pomade_SetFocusLevel(sc, 2) == POMADE_OK &&
                Pomade_SetTubeExpanded(sc, kids[0], 1) == POMADE_OK &&
                Pomade_IsTubeVisible(sc, kids[0]) == 0 &&
                Pomade_IsTubeVisible(sc, kids[1]) == 1 &&
                Pomade_IsTubeVisible(sc, grandKids[0]) == 1 &&
                Pomade_SetTubeExpanded(sc, 0, 0) == POMADE_OK &&
                Pomade_GetTubeExpanded(sc, 0) == 0 &&
                Pomade_GetTubeExpanded(sc, kids[0]) == 0 &&
                Pomade_IsTubeVisible(sc, 0) == 1 &&
                Pomade_IsTubeVisible(sc, kids[0]) == 0 &&
                Pomade_IsTubeVisible(sc, grandKids[0]) == 0 &&
                Pomade_SetTubeExpanded(sc, grandKids[0], 1) == POMADE_ERROR &&
                Pomade_SetActiveCutEnabled(sc, 0) == POMADE_OK &&
                Pomade_GetActiveCutEnabled(sc) == 0 &&
                Pomade_IsTubeVisible(sc, 0) == 1 &&
                Pomade_IsTubeVisible(sc, kids[0]) == 1 &&
                Pomade_IsTubeVisible(sc, grandKids[0]) == 1 &&
                Pomade_SetFocusLevel(sc, 0) == POMADE_OK;
            Check(activeCut,
                  "V1: active cut isolates one expanded branch and filters picks");
            int selKids[8] = {0};
            Check(Pomade_SelectSet(sc, usdGenPomade::PomadePick_TubeVert, &kids[1], nullptr,
                                  nullptr, 1) == POMADE_OK,
                  "V1: select the tube about to be subdivided");
            int more[4] = {0};
            int moreCount = 0;
            Check(Pomade_SubdivideTube(sc, kids[1], 2, "kmeans", 5, more, 4,
                                      &moreCount) == POMADE_OK &&
                      moreCount == 2,
                  "V1: subdivide the selected tube");
            Check(Pomade_ReadSelection(sc, usdGenPomade::PomadePick_TubeVert, selKids,
                                      nullptr, nullptr, 8,
                                      &count) == POMADE_OK && count == 2 &&
                      selKids[0] == more[0] && selKids[1] == more[1],
                  "V1: the selection moved to the children");
            // A merge drops the ids that no longer exist.
            Check(Pomade_MergeChildren(sc, kids[1]) == POMADE_OK,
                  "V1: merge those children back");
            Check(Pomade_GetSelectionCount(sc, usdGenPomade::PomadePick_TubeVert) == 0,
                  "V1: the selection drops tubes that are gone");
        }
        Check(Pomade_Destroy(sc) == POMADE_OK, "V1: selection model destroys");
    }

    // -- gizmo and brush records ------------------------------------------
    {
        PomadeModelContext *gc = nullptr;
        Check(Pomade_Create(&gc) == POMADE_OK && gc != nullptr,
              "V1: gizmo model creates");
        float const origin[3] = {1.0f, 2.0f, 3.0f};
        float const frame[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f};
        int kind = -1;
        float outOrigin[3] = {0.0f};
        float outFrame[9] = {0.0f};
        float size = 0.0f;
        int active = -99;
        Check(Pomade_GetGizmo(gc, &kind, outOrigin, outFrame, &size,
                             &active) == POMADE_OK &&
                  kind == 0 && active == -1,
              "V1: a fresh model has no gizmo");
        unsigned long long v = Pomade_GetVersion(gc);
        Check(Pomade_SetGizmo(gc, 1, origin, frame, 2.5f, 1) == POMADE_OK &&
                  Pomade_GetVersion(gc) == v + 1,
              "V1: setting a translate gizmo bumps the version");
        Check(Pomade_SetGizmo(gc, 1, origin, frame, 2.5f, 1) == POMADE_OK &&
                  Pomade_GetVersion(gc) == v + 1,
              "V1: re-setting the same gizmo changes nothing");
        Check(Pomade_GetGizmo(gc, &kind, outOrigin, outFrame, &size,
                             &active) == POMADE_OK &&
                  kind == 1 && outOrigin[1] == 2.0f && outFrame[4] == 1.0f &&
                  size == 2.5f && active == 1,
              "V1: the gizmo record round-trips");
        Check(Pomade_SetGizmo(gc, 9, origin, frame, 1.0f, -1) == POMADE_ERROR,
              "V1: an unknown gizmo kind is rejected");
        Check((Pomade_TakeDirty(gc) & usdGenPomade::PomadeDirty_Gizmo) != 0,
              "V1: the gizmo marks its own dirty bit");
        float const center[3] = {0.0f, 1.0f, 0.0f};
        float const normal[3] = {0.0f, 1.0f, 0.0f};
        int brushActive = -1;
        float outCenter[3] = {0.0f};
        float outNormal[3] = {0.0f};
        float radius = -1.0f;
        Check(Pomade_SetBrushRing(gc, center, normal, 0.75f) == POMADE_OK &&
                  Pomade_GetBrushRing(gc, &brushActive, outCenter, outNormal,
                                     &radius) == POMADE_OK &&
                  brushActive == 1 && radius == 0.75f &&
                  outCenter[1] == 1.0f,
              "V1: the brush ring round-trips");
        Check((Pomade_TakeDirty(gc) & usdGenPomade::PomadeDirty_Brush) != 0,
              "V1: the brush marks its own dirty bit");
        Check(Pomade_SetBrushRing(gc, center, normal, 0.0f) == POMADE_OK &&
                  Pomade_GetBrushRing(gc, &brushActive, nullptr, nullptr,
                                     &radius) == POMADE_OK &&
                  brushActive == 0 && radius == 0.0f,
              "V1: a zero radius clears the brush ring");
        // The geometry the index will publish, straight from the record.
        {
            usdGenPomade::PomadeGizmoRecord record;
            record.kind = usdGenPomade::PomadeGizmo_Translate;
            record.sizeWorld = 1.0f;
            record.activeHandle = 2;
            usdGenPomade::PomadeOverlayCurves curves;
            Check(usdGenPomade::PomadeBuildGizmoCurves(record, &curves) &&
                      curves.CurveCount() == 19 &&
                      curves.points.size() == 50 * 3,
                  "V1: a translate gizmo has Maya axes, arrows and move squares");
            bool activeMatches = true;
            bool foundActive = false;
            for (size_t i = 0; i < curves.handleIds.size(); ++i) {
                bool const want = curves.handleIds[i] == record.activeHandle;
                activeMatches = activeMatches &&
                    (curves.active[i] == (want ? 1 : 0));
                foundActive = foundActive || want;
            }
            Check(foundActive && activeMatches,
                  "V1: every curve of the requested handle is active");
            Check(curves.colors[0] > 0.8f && curves.colors[1] < 0.2f,
                  "V1: the u axis is red");
            // GZ-02 / parity G06: the headless fallback uses RigExec's
            // palette, the same one the Qt overlay draws with (pure
            // primaries, the dragged handle pure yellow).
            {
                // Curve colour per handle id, first curve of each wins.
                auto colourOf = [&](int handleId, float const want[3]) {
                    for (size_t i = 0; i < curves.handleIds.size(); ++i) {
                        if (curves.handleIds[i] == handleId) {
                            return curves.colors[i * 3 + 0] == want[0] &&
                                   curves.colors[i * 3 + 1] == want[1] &&
                                   curves.colors[i * 3 + 2] == want[2];
                        }
                    }
                    return false;
                };
                float const red[3] = {1.0f, 0.0f, 0.0f};
                float const green[3] = {0.0f, 1.0f, 0.0f};
                float const blue[3] = {0.0f, 0.0f, 1.0f};
                float const pureYellow[3] = {1.0f, 1.0f, 0.0f};
                bool const primaries =
                    colourOf(0, red) && colourOf(1, green) &&
                    colourOf(usdGenPomade::PomadeGizmoHandle_PlaneXY, blue);
                bool const yellow = colourOf(2, pureYellow);
                Check(primaries && yellow,
                      "GZ-02: axes are pure primaries and the active handle "
                      "is pure yellow");
            }
            // GZ-02 / parity G05: Scale is capped by cubes (six curves each:
            // two faces and four joining edges) and has the three planes.
            record.kind = usdGenPomade::PomadeGizmo_Scale;
            record.activeHandle = -1;
            {
                bool const built =
                    usdGenPomade::PomadeBuildGizmoCurves(record, &curves);
                int perAxis[3] = {0, 0, 0};
                int planes = 0;
                int centres = 0;
                for (size_t i = 0; i < curves.handleIds.size(); ++i) {
                    int const id = curves.handleIds[i];
                    if (id >= 0 && id < 3) {
                        ++perAxis[id];
                    } else if (id >= usdGenPomade::PomadeGizmoHandle_PlaneYZ &&
                               id <= usdGenPomade::PomadeGizmoHandle_PlaneXY) {
                        ++planes;
                    } else if (id == usdGenPomade::PomadeGizmoHandle_Center) {
                        ++centres;
                    }
                }
                Check(built && curves.CurveCount() == 25 &&
                          curves.points.size() == 80 * 3 &&
                          perAxis[0] == 7 && perAxis[1] == 7 &&
                          perAxis[2] == 7 && planes == 3 && centres == 1,
                      "GZ-02: a scale gizmo has three axes with cube caps, "
                      "three planes and the centre (" +
                          std::to_string(curves.CurveCount()) + " curves)");
            }
            record.activeHandle = 2;
            record.kind = usdGenPomade::PomadeGizmo_RingTRS;
            Check(usdGenPomade::PomadeBuildGizmoCurves(record, &curves) &&
                      curves.CurveCount() == 18 &&
                      curves.vertexCounts[0] ==
                          usdGenPomade::PomadeGizmoCircleSegments() + 1,
                  "V1: a ring gizmo is a closed circle plus three axes");
            // GZ-06: the tool's handle whitelist reaches the headless
            // fallback.  A mask of {U, V, Center, PlaneXY} draws no scale
            // ring and no W axis; every surviving curve is an allowed one.
            {
                using namespace usdGenPomade;
                unsigned int const mask =
                    (1u << PomadeGizmoHandle_AxisU) |
                    (1u << PomadeGizmoHandle_AxisV) |
                    (1u << PomadeGizmoHandle_Center) |
                    (1u << PomadeGizmoHandle_PlaneXY);
                record.allowedMask = mask;
                bool const built = PomadeBuildGizmoCurves(record, &curves);
                bool ring = false, w = false, foreign = false;
                size_t points = 0;
                for (int c = 0; c < curves.CurveCount(); ++c) {
                    int const id = curves.handleIds[c];
                    ring = ring || id == PomadeGizmoHandle_Ring;
                    w = w || id == PomadeGizmoHandle_AxisW;
                    foreign = foreign || ((mask >> id) & 1u) == 0;
                    points += size_t(curves.vertexCounts[c]);
                }
                Check(built && curves.CurveCount() > 0 && !ring && !w &&
                          !foreign && points * 3 == curves.points.size() &&
                          curves.colors.size() ==
                              size_t(curves.CurveCount()) * 3,
                      "GZ-06: a RingTRS record masked to {U,V,Center,"
                      "PlaneXY} yields no ring curve and no W axis");
                record.allowedMask = 0u;
                Check(!PomadeBuildGizmoCurves(record, &curves) &&
                          curves.CurveCount() == 0,
                      "GZ-06: a mask that hides every handle is no gizmo");
                record.allowedMask = PomadeGizmoAllHandles;
                float const o[3] = {0.0f, 0.0f, 0.0f};
                float const f[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                    0.0f, 0.0f, 1.0f};
                unsigned int got = 0;
                unsigned long long const before = Pomade_GetVersion(gc);
                Check(Pomade_SetGizmoEx(gc, 2, o, f, 1.0f, -1, mask) ==
                              POMADE_OK &&
                          Pomade_GetGizmoAllowedMask(gc, &got) == POMADE_OK &&
                          got == mask && Pomade_GetVersion(gc) == before + 1,
                      "GZ-06: Pomade_SetGizmoEx stores the allowed mask");
                Check(Pomade_SetGizmo(gc, 2, o, f, 1.0f, -1) == POMADE_OK &&
                          Pomade_GetGizmoAllowedMask(gc, &got) == POMADE_OK &&
                          got == PomadeGizmoAllHandles &&
                          Pomade_GetVersion(gc) == before + 2,
                      "GZ-06: plain Pomade_SetGizmo allows every handle, and "
                      "a mask change alone is a new record");
            }
            record.kind = usdGenPomade::PomadeGizmo_None;
            Check(!usdGenPomade::PomadeBuildGizmoCurves(record, &curves) &&
                      curves.CurveCount() == 0,
                  "V1: no gizmo builds no curves");
            record.kind = usdGenPomade::PomadeGizmo_Translate;
            record.frame[0] = 0.0f;  // a degenerate u axis
            Check(!usdGenPomade::PomadeBuildGizmoCurves(record, &curves),
                  "V1: a degenerate frame draws nothing");
            usdGenPomade::PomadeBrushRingRecord brush;
            brush.active = true;
            brush.radiusWorld = 0.5f;
            Check(usdGenPomade::PomadeBuildBrushRingCurves(brush, &curves) &&
                      curves.CurveCount() == 1,
                  "V1: the brush ring is one closed circle");
        }
        Check(Pomade_Destroy(gc) == POMADE_OK, "V1: gizmo model destroys");
    }

    // -- the gesture bracket, redo, and undo labels ------------------------
    {
        PomadeModelContext *bc = nullptr;
        Check(Pomade_Create(&bc) == POMADE_OK && bc != nullptr,
              "V1: gesture model creates");
        Check(Pomade_BuildTestTube(bc, 0, 0, 0.0f, 0.0f) == POMADE_OK,
              "V1: gesture tube builds");
        size_t const floats = size_t(Pomade_GetVertexCount(bc)) * 3;
        std::vector<float> base(floats), dragged(floats), probe(floats);
        Check(Pomade_ReadTubePoints(bc, base.data(), int(floats)) == POMADE_OK,
              "V1: press-time points read");
        Check(Pomade_GetGestureDepth(bc) == 0, "V1: no gesture is open");
        Check(Pomade_EndGesture(bc) == POMADE_ERROR,
              "V1: End without Begin is an error");
        Check(Pomade_CancelGesture(bc, nullptr) == POMADE_ERROR,
              "V1: Cancel without Begin is an error");
        Check(Pomade_BeginGesture(bc, "sculpt") == POMADE_OK &&
                  Pomade_GetGestureDepth(bc) == 1,
              "V1: Begin opens the bracket");
        Check(Pomade_BeginGesture(bc, "sculpt") == POMADE_ERROR &&
                  Pomade_GetGestureDepth(bc) == 1,
              "V1: a nested Begin is an error, not a counter");
        Check(Pomade_GetUndoDepth(bc) == 1,
              "V1: Begin pushes exactly one step");
        for (int i = 0; i < 50; ++i) {
            Pomade_MoveCenterRing(bc, 2, 0.01f, 0.0f);
        }
        Check(Pomade_GetUndoDepth(bc) == 1,
              "V1: fifty moves inside the bracket are still one step");
        Check(Pomade_EndGesture(bc) == POMADE_OK &&
                  Pomade_GetGestureDepth(bc) == 0,
              "V1: End seals the bracket");
        Check(Pomade_ReadTubePoints(bc, dragged.data(), int(floats)) ==
                  POMADE_OK && dragged != base,
              "V1: the drag moved the tube");
        {
            char label[32] = {0};
            Check(Pomade_GetUndoLabel(bc, 0, label, sizeof(label)) ==
                      POMADE_OK && std::string(label) == "sculpt",
                  "V1: the step carries the gesture's label");
            Check(Pomade_GetUndoLabel(bc, 7, label, sizeof(label)) ==
                      POMADE_ERROR,
                  "V1: there is no step at depth 7");
        }
        unsigned int coreDirty = 0;
        Check(Pomade_Undo(bc, &coreDirty) == POMADE_OK &&
                  (coreDirty & usdGenPomade::PomadeDirty_Points) != 0,
              "V1: undo reports the dirty bits to publish");
        Check(Pomade_ReadTubePoints(bc, probe.data(), int(floats)) ==
                  POMADE_OK && probe == base,
              "V1: one undo takes back the whole drag");
        Check(Pomade_GetRedoDepth(bc) == 1, "V1: the drag is on the redo stack");
        {
            char label[32] = {0};
            Check(Pomade_GetUndoLabel(bc, -1, label, sizeof(label)) ==
                      POMADE_OK && std::string(label) == "sculpt",
                  "V1: a negative depth reads the redo label");
        }
        coreDirty = 0;
        Check(Pomade_Redo(bc, &coreDirty) == POMADE_OK &&
                  (coreDirty & usdGenPomade::PomadeDirty_Points) != 0,
              "V1: redo reports its dirty bits too");
        Check(Pomade_ReadTubePoints(bc, probe.data(), int(floats)) ==
                  POMADE_OK && probe == dragged,
              "V1: redo restores the drag bit-exactly");
        Check(Pomade_GetRedoDepth(bc) == 0 && Pomade_GetUndoDepth(bc) == 1,
              "V1: redo moves the step back to the undo stack");
        // A new edit branches the history.
        Check(Pomade_Undo(bc, nullptr) == POMADE_OK &&
                  Pomade_GetRedoDepth(bc) == 1,
              "V1: undo again to arm the redo stack");
        Check(Pomade_MoveCenterRing(bc, 1, 0.2f, 0.0f) == POMADE_OK &&
                  Pomade_GetRedoDepth(bc) == 0,
              "V1: a new mutation clears the redo stack");
        Check(Pomade_Redo(bc, nullptr) == POMADE_OK &&
                  Pomade_GetRedoDepth(bc) == 0,
              "V1: redo on an empty stack is a no-op success");

        // -- cancel restores the press-time base bit-exactly --------------
        Check(Pomade_ReadTubePoints(bc, base.data(), int(floats)) == POMADE_OK,
              "V1: second press-time points read");
        int const depthBefore = Pomade_GetUndoDepth(bc);
        unsigned long long const versionBefore = Pomade_GetVersion(bc);
        Check(Pomade_BeginGesture(bc, "drag") == POMADE_OK,
              "V1: open a gesture to cancel");
        for (int i = 0; i < 12; ++i) {
            Pomade_MoveCenterRing(bc, 3, 0.05f, 0.02f);
        }
        Check(Pomade_ReadTubePoints(bc, probe.data(), int(floats)) ==
                  POMADE_OK && probe != base,
              "V1: the cancelled drag did move the tube first");
        {
            // Ctrl+Z mid-drag (a dock slider holds the bracket with the
            // keyboard free) must refuse like Redo does: popping Begin's
            // step lost the base and made Cancel pop an older step.
            std::vector<float> mid(floats);
            int const redoBefore = Pomade_GetRedoDepth(bc);
            Check(Pomade_Undo(bc, nullptr) == POMADE_ERROR &&
                      Pomade_GetUndoDepth(bc) == depthBefore + 1 &&
                      Pomade_GetRedoDepth(bc) == redoBefore &&
                      Pomade_GetGestureDepth(bc) == 1,
                  "V1: Undo inside an open bracket is refused, stacks intact");
            Check(Pomade_ReadTubePoints(bc, mid.data(), int(floats)) ==
                      POMADE_OK && mid == probe,
                  "V1: the refused Undo leaves the dragged shape alone");
            Check(Pomade_Redo(bc, nullptr) == POMADE_ERROR,
                  "V1: Redo inside an open bracket is refused too");
        }
        coreDirty = 0;
        Check(Pomade_CancelGesture(bc, &coreDirty) == POMADE_OK &&
                  (coreDirty & usdGenPomade::PomadeDirty_Points) != 0,
              "V1: Cancel reports what to publish");
        Check(Pomade_ReadTubePoints(bc, probe.data(), int(floats)) ==
                  POMADE_OK && probe == base,
              "V1: Cancel restores the press-time base bit-exactly");
        Check(Pomade_GetUndoDepth(bc) == depthBefore,
              "V1: Cancel leaves no step behind");
        Check(Pomade_GetGestureDepth(bc) == 0, "V1: Cancel closes the bracket");
        Check(Pomade_GetVersion(bc) > versionBefore,
              "V1: the moves bumped the version; Cancel does not bump again");
        Check(Pomade_Destroy(bc) == POMADE_OK, "V1: gesture model destroys");
    }

    // Grab keeps its press-time screen footprint for the whole gesture. A
    // second positive increment must still select the CV after the first
    // increment has moved it farther than the brush radius from the anchor.
    {
        PomadeModelContext *sc = nullptr;
        Check(Pomade_Create(&sc) == POMADE_OK && sc != nullptr,
              "V1: frozen-grab model creates");
        if (sc) {
            Check(Pomade_BuildTestTube(sc, 0, 0, 0.0f, 0.0f) == POMADE_OK,
                  "V1: frozen-grab tube builds");
            auto readCenters = [&](std::vector<float> *out) {
                int const count = Pomade_GetTubeCenterCount(sc, 0);
                if (count < 2 || !out) {
                    return false;
                }
                out->assign(size_t(count) * 3, 0.0f);
                for (int i = 0; i < count; ++i) {
                    if (Pomade_GetTubeCenterCV(sc, 0, i,
                                               out->data() + size_t(i) * 3) !=
                        POMADE_OK) {
                        return false;
                    }
                }
                return true;
            };
            // y=2 projects to (500, 250); adjacent straight-spine CVs are
            // 125 pixels apart, so a 30-pixel brush isolates CV 2.
            float const viewProj[16] = {
                0.25f, 0.0f,  0.0f, 0.0f,
                0.0f,  0.25f, 0.0f, 0.0f,
                0.0f,  0.0f,  1.0f, 0.0f,
                0.0f,  0.0f,  0.0f, 1.0f};
            float const delta[3] = {0.4f, 0.0f, 0.0f};
            std::vector<float> base, afterFirst, afterDrag, restored;
            int firstTouched = 0, secondTouched = 0;
            bool const setup = readCenters(&base) &&
                Pomade_BeginGesture(sc, "frozen grab") == POMADE_OK &&
                Pomade_SculptStrokeShaped(sc, 0, "grab", viewProj, 1000,
                                         1000, 500.0f, 250.0f, 30.0f, delta,
                                         0.0f, 0.5f, 0.0f, 0, 0,
                                         &firstTouched) == POMADE_OK &&
                readCenters(&afterFirst) &&
                Pomade_SculptStrokeShaped(sc, 0, "grab", viewProj, 1000,
                                         1000, 500.0f, 250.0f, 30.0f, delta,
                                         0.0f, 0.5f, 0.0f, 0, 0,
                                         &secondTouched) == POMADE_OK &&
                readCenters(&afterDrag) && Pomade_EndGesture(sc) == POMADE_OK;
            bool firstOnly = setup && firstTouched == 1 &&
                std::fabs(afterFirst[2 * 3] - (base[2 * 3] + 0.4f)) < 1e-6f;
            for (int i = 0; firstOnly && i < 5; ++i) {
                if (i != 2) {
                    firstOnly = afterFirst[size_t(i) * 3] ==
                                    base[size_t(i) * 3] &&
                                afterFirst[size_t(i) * 3 + 1] ==
                                    base[size_t(i) * 3 + 1] &&
                                afterFirst[size_t(i) * 3 + 2] ==
                                    base[size_t(i) * 3 + 2];
                }
            }
            Check(firstOnly,
                  "V1: frozen grab keeps the initial one-CV footprint");
            Check(setup && secondTouched == 1 &&
                      std::fabs(afterDrag[2 * 3] -
                                (base[2 * 3] + 0.8f)) < 1e-6f,
                  "V1: a long frozen grab keeps applying incremental deltas");
            Check(Pomade_Undo(sc, nullptr) == POMADE_OK &&
                      readCenters(&restored) && restored == base,
                  "V1: frozen grab remains one undoable gesture");
            Check(Pomade_Redo(sc, nullptr) == POMADE_OK &&
                      readCenters(&restored) && restored == afterDrag,
                  "V1: frozen-grab redo restores the full drag");
            int cancelTouched = 0;
            Check(Pomade_BeginGesture(sc, "frozen cancel") == POMADE_OK &&
                      Pomade_SculptStrokeShaped(
                          // Redo left CV 2 at x=.8, which projects to x=600.
                          // A new gesture captures that current press point;
                          // 500 is its original pre-drag point and must miss.
                          sc, 0, "grab", viewProj, 1000, 1000, 600.0f,
                          250.0f, 30.0f, delta, 0.0f, 0.5f, 0.0f, 0, 1,
                          &cancelTouched) == POMADE_OK && cancelTouched == 1 &&
                      Pomade_CancelGesture(sc, nullptr) == POMADE_OK &&
                      readCenters(&restored) && restored == afterDrag,
                  "V1: mirrored frozen grab cancels to its gesture base");
            Check(Pomade_Destroy(sc) == POMADE_OK,
                  "V1: frozen-grab model destroys");
        }
    }

    // -- the marquee's device lane agrees with its CPU twin ---------------
    //
    // Guide CVs are the heavyweight kind a lasso actually discriminates
    // (a tube's 40 verts collapse to one item; 600 guides do not), so the
    // K11b mask kernel is proven against the twin on the same rectangle,
    // with USDGEN_POMADE_FORCE_CPU_PICK deciding which lane runs — the same
    // switch the point pick's parity test uses.
    {
        PomadeModelContext *dc = nullptr;
        Check(Pomade_Create(&dc) == POMADE_OK && dc != nullptr,
              "V1: marquee parity model creates");
        Check(Pomade_BuildTestTube(dc, 0, 0, 0.0f, 0.0f) == POMADE_OK,
              "V1: parity tube builds");
        Check(Pomade_SetFillParams(dc, 600.0f, 8, 11, 0.0f, nullptr, 0) ==
                  POMADE_OK,
              "V1: 600 guides of 8 CVs (4800 candidates, past the "
              "device threshold)");
        Check(Pomade_RefillGuides(dc, 1.0f) == POMADE_OK,
              "V1: the guides fill");
        int guides = 0, cvs = 0;
        Check(Pomade_GetGuideCounts(dc, &guides, &cvs) == POMADE_OK &&
                  guides * cvs > 4096,
              "V1: the candidate count clears the device threshold");
#ifdef USDGEN_POMADE_HAS_CUDA
        // Without a mirror there is no device lane and the parity below
        // would be two CPU runs agreeing with themselves. Say so loudly
        // rather than passing vacuously.
        Check(Pomade_HasCudaMirror(dc) == 1,
              "V1: this build has the device mirror the parity needs");
#else
        std::printf("skip: no CUDA, the marquee parity runs CPU vs CPU\n");
#endif
        int const w = 800, h = 600;
        float viewProj[16] = {0.0f};
        viewProj[0] = 1.0f;
        viewProj[5] = 0.5f;
        viewProj[10] = 1.0f;
        viewProj[13] = -1.0f;
        viewProj[15] = 1.0f;
        std::vector<int> device;
        std::vector<int> cpu;
        int count = 0;
        std::vector<int> buffer(size_t(guides) + 8);
        auto readGuides = [&](std::vector<int> *out) {
            count = 0;
            bool const ok =
                Pomade_ReadSelection(dc, usdGenPomade::PomadePick_Guide,
                                    buffer.data(), nullptr, nullptr,
                                    int(buffer.size()), &count) == POMADE_OK;
            out->assign(buffer.begin(), buffer.begin() + count);
            return ok;
        };
        // The right half of the frame: the tube's guides span x in
        // [-0.5, 0.5], which is px 200 to 600 here, so a band from px 400
        // splits them.
        Check(Pomade_SelectRect(dc, viewProj, w, h, 400.0f, -1e4f, 1e4f, 1e4f,
                               usdGenPomade::PomadePick_Guide,
                               POMADE_SELECT_SET) == POMADE_OK &&
                  readGuides(&device),
              "V1: marquee half the guides (device lane where there is one)");
        int const halfCount = int(device.size());
#ifdef _WIN32
        _putenv("USDGEN_POMADE_FORCE_CPU_PICK=1");
#else
        setenv("USDGEN_POMADE_FORCE_CPU_PICK", "1", 1);
#endif
        Check(Pomade_SelectRect(dc, viewProj, w, h, 400.0f, -1e4f, 1e4f, 1e4f,
                               usdGenPomade::PomadePick_Guide,
                               POMADE_SELECT_SET) == POMADE_OK &&
                  readGuides(&cpu),
              "V1: the same marquee on the CPU twin");
        Check(device == cpu,
              "V1: the device mask and the CPU scan select the same guides");
        // A lasso over the same half has to agree with the rectangle, and
        // both lanes have to agree about it.
        float const lasso[8] = {400.0f, -1e4f, 1e4f, -1e4f,
                                1e4f,   1e4f,  400.0f, 1e4f};
        std::vector<int> cpuLasso;
        Check(Pomade_SelectPolygon(dc, viewProj, w, h, lasso, 4,
                                  usdGenPomade::PomadePick_Guide,
                                  POMADE_SELECT_SET) == POMADE_OK &&
                  readGuides(&cpuLasso) && cpuLasso == cpu,
              "V1: a box-shaped lasso is the box (CPU lane)");
#ifdef _WIN32
        _putenv("USDGEN_POMADE_FORCE_CPU_PICK=");
#else
        unsetenv("USDGEN_POMADE_FORCE_CPU_PICK");
#endif
        std::vector<int> deviceLasso;
        Check(Pomade_SelectPolygon(dc, viewProj, w, h, lasso, 4,
                                  usdGenPomade::PomadePick_Guide,
                                  POMADE_SELECT_SET) == POMADE_OK &&
                  readGuides(&deviceLasso) && deviceLasso == cpuLasso,
              "V1: the device lasso mask agrees with the CPU twin");
        Check(Pomade_SelectRect(dc, viewProj, w, h, -1e4f, -1e4f, 1e4f, 1e4f,
                               usdGenPomade::PomadePick_Guide,
                               POMADE_SELECT_SET) == POMADE_OK &&
                  readGuides(&device) && count == guides,
              "V1: a full-frame marquee catches every guide");
        Check(halfCount > 0 && halfCount < guides,
              "V1: the half-frame band caught a real subset (%d of %d)");
        Check(Pomade_Destroy(dc) == POMADE_OK, "V1: parity model destroys");
    }

    // -- the kinds V1 adds to the pick: edge, region, section ring --------
    //
    // They need a real groom: a stroked region on a scalp gives the graph
    // its edges and regions, and building the tube from that region is
    // what puts the tube on the K5 sections path, which is where its rings
    // become pick candidates at all.
    {
        PomadeModelContext *kc = nullptr;
        Check(Pomade_Create(&kc) == POMADE_OK && kc != nullptr,
              "V1: kinds model creates");
        // The 4 x 4 quad grid in XZ the T3 scripts use: face f = ix * 4 + iz
        // covers x in [ix, ix+1], z in [iz, iz+1], and (u, v) on that face
        // is (z - iz, x - ix).
        std::vector<float> points;
        for (int ix = 0; ix < 5; ++ix) {
            for (int iz = 0; iz < 5; ++iz) {
                points.push_back(float(ix));
                points.push_back(0.0f);
                points.push_back(float(iz));
            }
        }
        std::vector<int> counts(16, 4);
        std::vector<int> indices;
        for (int ix = 0; ix < 4; ++ix) {
            for (int iz = 0; iz < 4; ++iz) {
                int const a = ix * 5 + iz;
                indices.push_back(a);
                indices.push_back(a + 5);
                indices.push_back(a + 6);
                indices.push_back(a + 1);
            }
        }
        Check(Pomade_BindScalp(kc, points.data(), int(points.size()),
                              counts.data(), int(counts.size()),
                              indices.data(), int(indices.size())) ==
                  POMADE_OK,
              "V1: the 4x4 scalp binds");
        Pomade_SetSnapRadius(kc, 0.1f);
        // Stroke the rectangle (0, 1) -> (2, 1) -> (2, 3) -> (0, 3) closed.
        float const corners[4][2] = {{0.0f, 1.0f},
                                     {2.0f, 1.0f},
                                     {2.0f, 3.0f},
                                     {0.0f, 3.0f}};
        std::vector<int> faces;
        std::vector<float> uvs;
        for (int c = 0; c < 4; ++c) {
            for (int s = 0; s < 5; ++s) {
                float const t = float(s) / 5.0f;
                float const x = corners[c][0] +
                                (corners[(c + 1) % 4][0] - corners[c][0]) * t;
                float const z = corners[c][1] +
                                (corners[(c + 1) % 4][1] - corners[c][1]) * t;
                int const ix = std::min(std::max(int(std::floor(x)), 0), 3);
                int const iz = std::min(std::max(int(std::floor(z)), 0), 3);
                faces.push_back(ix * 4 + iz);
                uvs.push_back(z - float(iz));
                uvs.push_back(x - float(ix));
            }
        }
        faces.push_back(0 * 4 + 1);
        uvs.push_back(0.0f);
        uvs.push_back(0.0f);
        std::vector<int> chain(256);
        int chainCount = 0, closed = 0, weldStart = 0, weldEnd = 0;
        Check(Pomade_GraphStroke(kc, faces.data(), uvs.data(),
                                int(faces.size()), 0.1f, 0.05f, chain.data(),
                                int(chain.size()), &chainCount, &closed,
                                &weldStart, &weldEnd) == POMADE_OK &&
                  closed == 1,
              "V1: the stroke closes into a region");
        Check(Pomade_Rasterise(kc) == POMADE_OK, "V1: K3 rasterises");
        Check(Pomade_BuildTubeFromRegion(kc, 0, 5, 8, 3.0f) == POMADE_OK,
              "V1: the tube builds from the region");
        Check(Pomade_GetSectionCount(kc) == 5,
              "V1: the region-built tube carries five authored rings");
        int nodes = 0, edges = 0, regions = 0;
        Check(Pomade_GetGraphCounts(kc, &nodes, &edges, &regions) ==
                  POMADE_OK && edges > 0 && regions > 0,
              "V1: the graph has edges and a region");
        int const w = 800, h = 600;
        // Top view: x and z map to the frame, so the scalp's graph spreads
        // out and the tube (which rises in +Y) does not overlap itself.
        float top[16] = {0.0f};
        top[0] = 0.5f;
        top[12] = -1.0f;
        top[9] = 0.5f;
        top[13] = -1.0f;
        top[10] = 0.0f;
        top[15] = 1.0f;
        top[6] = 1.0f;  // depth from y, so the pick has something to order
        int count = 0;
        Check(Pomade_SelectRect(kc, top, w, h, -1e4f, -1e4f, 1e4f, 1e4f,
                               usdGenPomade::PomadePick_GraphEdge,
                               POMADE_SELECT_SET) == POMADE_OK &&
                  Pomade_ReadSelection(kc, usdGenPomade::PomadePick_GraphEdge,
                                      nullptr, nullptr, nullptr, 0,
                                      &count) == POMADE_OK && count == edges,
              "V1: a full-frame marquee selects every graph edge once");
        Check(Pomade_SelectRect(kc, top, w, h, -1e4f, -1e4f, 1e4f, 1e4f,
                               usdGenPomade::PomadePick_Region,
                               POMADE_SELECT_SET) == POMADE_OK &&
                  Pomade_ReadSelection(kc, usdGenPomade::PomadePick_Region,
                                      nullptr, nullptr, nullptr, 0,
                                      &count) == POMADE_OK && count == regions,
              "V1: a full-frame marquee selects every region once");
        // Side view for the rings: they sit at different heights along the
        // tube, so they only separate in a view that shows +Y.
        float side[16] = {0.0f};
        side[0] = 0.5f;
        side[12] = -1.0f;
        side[5] = 0.4f;
        side[13] = -1.0f;
        side[10] = 1.0f;
        side[15] = 1.0f;
        Check(Pomade_SelectRect(kc, side, w, h, -1e4f, -1e4f, 1e4f, 1e4f,
                               usdGenPomade::PomadePick_SectionRing,
                               POMADE_SELECT_SET) == POMADE_OK &&
                  Pomade_ReadSelection(kc,
                                      usdGenPomade::PomadePick_SectionRing,
                                      nullptr, nullptr, nullptr, 0,
                                      &count) == POMADE_OK && count == 5,
              "V1: a full-frame marquee selects all five section rings");
        // Point picks, at points whose place on the scalp is known: the
        // midpoint of the stroke's bottom run lies on an edge, and the
        // centre of the rectangle lies in the region.
        {
            float const onEdge[3] = {1.0f, 0.0f, 1.0f};
            float const inRegion[3] = {1.0f, 0.0f, 2.0f};
            float px = 0.0f, py = 0.0f, z = 0.0f;
            Check(usdGenPomade::PomadeProjectPoint(onEdge, top, w, h, &px, &py,
                                                 &z),
                  "V1: the edge midpoint projects");
            int hit = 0, id = -1, sub = -1, subsub = -1;
            unsigned int kind = 0;
            Check(Pomade_PickItem(kc, top, w, h, px, py, 30.0f,
                                 usdGenPomade::PomadePick_GraphEdge, &hit,
                                 &kind, &id, &sub, &subsub) == POMADE_OK &&
                      hit == 1 && kind == usdGenPomade::PomadePick_GraphEdge &&
                      id >= 0,
                  "V1: a click on the boundary picks a graph edge");
            // Not the same edge id as the snap query, necessarily: the
            // pick wins on the nearest polyline SAMPLE in pixels and a
            // shared node belongs to two edges, so the two lookups can
            // name either side of a junction. What must hold is that both
            // find an edge there at all.
            Check(Pomade_GraphSnapEdge(kc, onEdge, 0.3f) >= 0,
                  "V1: the snap query agrees an edge runs through there");
            hit = 1;
            Check(Pomade_PickItem(kc, top, w, h, px, py, 30.0f,
                                 usdGenPomade::PomadePick_GraphEdge, &hit,
                                 &kind, &id, &sub, &subsub) == POMADE_OK,
                  "V1: re-pick the boundary");
            Check(usdGenPomade::PomadeProjectPoint(inRegion, top, w, h, &px,
                                                 &py, &z),
                  "V1: the region centre projects");
            hit = 0;
            id = -1;
            Check(Pomade_PickItem(kc, top, w, h, px, py, 60.0f,
                                 usdGenPomade::PomadePick_Region, &hit, &kind,
                                 &id, &sub, &subsub) == POMADE_OK &&
                      hit == 1 && kind == usdGenPomade::PomadePick_Region &&
                      id >= 0,
                  "V1: a click inside the region picks the region");
            hit = 1;
            Check(Pomade_PickItem(kc, top, w, h, px, py, 30.0f,
                                 usdGenPomade::PomadePick_GraphEdge, &hit,
                                 &kind, &id, &sub, &subsub) == POMADE_OK &&
                      hit == 0,
                  "V1: no edge is within 30 px of the region centre");
        }
        Check(Pomade_Destroy(kc) == POMADE_OK, "V1: kinds model destroys");
    }

    // -- exact surface region query --------------------------------------
    // A single coarse quad can hold several graph loops.  Its K3 face map
    // has one winning id, so region picking must classify the actual K1
    // (face, uv) hit instead of approximating from a region centre or the
    // face id.
    {
        PomadeModelContext *rc = nullptr;
        Check(Pomade_Create(&rc) == POMADE_OK && rc != nullptr,
              "surface region: model creates");
        float const points[] = {0.0f, 0.0f, 0.0f,
                                1.0f, 0.0f, 0.0f,
                                1.0f, 0.0f, 1.0f,
                                0.0f, 0.0f, 1.0f};
        int const counts[] = {4};
        int const indices[] = {0, 1, 2, 3};
        Check(rc && Pomade_BindScalp(rc, points, 12, counts, 1, indices, 4) ==
                        POMADE_OK,
              "surface region: one-quad scalp binds");
        auto addLoop = [&](float u0, float v0, float u1, float v1) {
            int node[4] = {-1, -1, -1, -1};
            float const uv[4][2] = {{u0, v0}, {u1, v0},
                                    {u1, v1}, {u0, v1}};
            for (int i = 0; i < 4; ++i) {
                if (Pomade_GraphAddNode(rc, 0, uv[i][0], uv[i][1],
                                       &node[i]) != POMADE_OK) {
                    return false;
                }
            }
            for (int i = 0; i < 4; ++i) {
                int edge = -1;
                if (Pomade_GraphConnect(rc, node[i], node[(i + 1) % 4],
                                       &edge) != POMADE_OK) {
                    return false;
                }
            }
            return true;
        };
        Check(rc && addLoop(0.10f, 0.15f, 0.35f, 0.85f) &&
                        addLoop(0.65f, 0.15f, 0.90f, 0.85f),
              "surface region: two disjoint subface loops author");
        Check(Pomade_Rasterise(rc) == POMADE_OK,
              "surface region: coarse map rasterises");
        int const left = Pomade_RegionAtSurface(rc, 0, 0.20f, 0.50f);
        int const right = Pomade_RegionAtSurface(rc, 0, 0.80f, 0.50f);
        Check(left >= 0 && right >= 0 && left != right,
              "surface region: same face picks its distinct subface regions");
        Check(Pomade_RegionAtSurface(rc, 0, 0.50f, 0.50f) == -1,
              "surface region: same-face gap is not assigned a region");
        Check(Pomade_RegionAtSurface(rc, 99, 0.20f, 0.50f) == -1 &&
                  Pomade_RegionAtSurface(nullptr, 0, 0.20f, 0.50f) == -1,
              "surface region: invalid surface hits miss");
        Check(Pomade_Destroy(rc) == POMADE_OK,
              "surface region: model destroys");
    }

    // -- atomic shared-CV region creation ---------------------------------
    {
        PomadeModelContext *cc = nullptr;
        Check(Pomade_Create(&cc) == POMADE_OK && cc != nullptr,
              "shared CV: model creates");
        float const points[] = {0.0f, 0.0f, 0.0f,
                                1.0f, 0.0f, 0.0f,
                                1.0f, 0.0f, 1.0f,
                                0.0f, 0.0f, 1.0f};
        int const counts[] = {4};
        int const indices[] = {0, 1, 2, 3};
        Check(cc && Pomade_BindScalp(cc, points, 12, counts, 1, indices, 4) ==
                        POMADE_OK,
              "shared CV: one-quad scalp binds");
        int const newNodes[] = {-1, -1, -1, -1};
        int const faces[] = {0, 0, 0, 0};
        float const leftUV[] = {0.10f, 0.20f, 0.45f, 0.20f,
                                0.45f, 0.80f, 0.10f, 0.80f};
        int leftRegion = -1;
        Check(Pomade_GraphCreateRegion(cc, newNodes, faces, leftUV, 4,
                                      &leftRegion) == POMADE_OK &&
                  leftRegion >= 0,
              "shared CV: atomically creates the first region");
        int nodeCount = 0, edgeCount = 0, regionCount = 0;
        Check(Pomade_GetGraphCounts(cc, &nodeCount, &edgeCount, &regionCount) ==
                        POMADE_OK &&
                  nodeCount == 4 && edgeCount == 4 && regionCount == 1,
              "shared CV: first closed chain has one region");
        // Leave a dead-id gap. The next call must reuse ids 1 and 2, never
        // their compact draw-order positions.
        int loose = -1;
        Check(Pomade_GraphAddNode(cc, 0, 0.70f, 0.10f, &loose) == POMADE_OK &&
                  loose == 4 && Pomade_GraphDeleteNode(cc, loose) == POMADE_OK,
              "shared CV: a deleted-node gap exists");
        int face = -1;
        float uv[2] = {}, p[3] = {};
        Check(Pomade_GraphGetNode(cc, 1, &face, uv, p) == POMADE_OK &&
                  face == 0 && uv[0] == 0.45f && uv[1] == 0.20f &&
                  Pomade_GraphGetNode(cc, loose, &face, uv, p) == POMADE_ERROR,
              "shared CV: stable node getter rejects the dead gap");
        // Reposition moves a whole edge as one graph sample: both stable
        // endpoint ids move together, regions re-extract once, and the
        // existing gesture bracket owns undo, redo and Escape cancellation.
        int edgeNodes[2] = {-1, -1};
        int const edgeFaces[2] = {0, 0};
        float const movedEdgeUV[4] = {0.50f, 0.20f, 0.50f, 0.80f};
        float const cancelledEdgeUV[4] = {0.55f, 0.20f, 0.55f, 0.80f};
        float const collapsedEdgeUV[4] = {0.50f, 0.50f, 0.50f, 0.50f};
        float p1[3] = {}, p2[3] = {};
        float uv1[2] = {}, uv2[2] = {};
        uint32_t cancelDirty = 0;
        uint64_t const mapBeforeMove = Pomade_GetMapVersion(cc);
        Check(Pomade_GraphGetEdge(cc, 1, edgeNodes) == POMADE_OK &&
                  edgeNodes[0] == 1 && edgeNodes[1] == 2 &&
                  Pomade_BeginGesture(cc, "move edge") == POMADE_OK &&
                  Pomade_GraphMoveNodes(cc, edgeNodes, edgeFaces, movedEdgeUV,
                                       2) == POMADE_OK &&
                  Pomade_GraphGetNode(cc, 1, &face, uv1, p1) == POMADE_OK &&
                  Pomade_GraphGetNode(cc, 2, &face, uv2, p2) == POMADE_OK &&
                  uv1[0] == 0.50f && uv1[1] == 0.20f &&
                  uv2[0] == 0.50f && uv2[1] == 0.80f &&
                  Pomade_GetMapVersion(cc) > mapBeforeMove &&
                  Pomade_EndGesture(cc) == POMADE_OK &&
                  Pomade_Undo(cc, nullptr) == POMADE_OK &&
                  Pomade_GraphGetNode(cc, 1, &face, uv1, p1) == POMADE_OK &&
                  Pomade_GraphGetNode(cc, 2, &face, uv2, p2) == POMADE_OK &&
                  uv1[0] == 0.45f && uv2[0] == 0.45f &&
                  Pomade_Redo(cc, nullptr) == POMADE_OK &&
                  Pomade_BeginGesture(cc, "cancel edge") == POMADE_OK &&
                  Pomade_GraphMoveNodes(cc, edgeNodes, edgeFaces,
                                       cancelledEdgeUV, 2) == POMADE_OK &&
                  Pomade_CancelGesture(cc, &cancelDirty) == POMADE_OK &&
                  cancelDirty != usdGenPomade::PomadeDirty_Clean &&
                  Pomade_GraphGetNode(cc, 1, &face, uv1, p1) == POMADE_OK &&
                  Pomade_GraphGetNode(cc, 2, &face, uv2, p2) == POMADE_OK &&
                  uv1[0] == 0.50f && uv2[0] == 0.50f &&
                  Pomade_GraphMoveNodes(cc, edgeNodes, edgeFaces,
                                       collapsedEdgeUV, 2) == POMADE_ERROR &&
                  Pomade_GraphGetNode(cc, 1, &face, uv1, p1) == POMADE_OK &&
                  Pomade_GraphGetNode(cc, 2, &face, uv2, p2) == POMADE_OK &&
                  uv1[0] == 0.50f && uv1[1] == 0.20f &&
                  uv2[0] == 0.50f && uv2[1] == 0.80f &&
                  Pomade_GraphGetEdge(cc, 1, edgeNodes) == POMADE_OK &&
                  edgeNodes[0] == 1 && edgeNodes[1] == 2,
              "shared CV: atomic edge reposition preserves ids and cancels safely");
        // The graph draws a node above the scalp tint. At an oblique,
        // high-zoom view that normal lift is many pixels, so K11 selection
        // must use the displayed point while raw Pomade_Pick keeps its
        // canonical candidate contract. This is the exact existing endpoint
        // the adjacent region reuses below.
        Check(Pomade_GraphGetNode(cc, 1, &face, uv, p) == POMADE_OK,
              "shared CV: the moved endpoint remains a canonical graph node");
        float rawNode[3] = {p[0], p[1], p[2]};
        float displayNode[3] = {};
        float oblique[16] = {};
        oblique[0] = 1.0f;
        oblique[1] = 0.10f;
        oblique[5] = 20.0f;  // high zoom makes the normal lift observable
        oblique[9] = 0.50f;
        oblique[10] = 1.0f;
        oblique[13] = -0.20f;
        oblique[15] = 1.0f;
        float rawX = 0.0f, rawY = 0.0f, displayX = 0.0f, displayY = 0.0f,
              depth = 0.0f, rawDistance = 0.0f, rawDepth = 0.0f;
        int hit = 0, pickId = -1, sub = -1, subSub = -1;
        unsigned int kind = 0;
        Check(Pomade_GraphGetNodeDisplayPosition(cc, 1, displayNode) ==
                      POMADE_OK &&
                  Pomade_GraphGetNodeDisplayPosition(cc, loose, displayNode) ==
                      POMADE_ERROR &&
                  usdGenPomade::PomadeProjectPoint(rawNode, oblique, 1024, 1024,
                                                  &rawX, &rawY, &depth) &&
                  usdGenPomade::PomadeProjectPoint(displayNode, oblique, 1024,
                                                  1024, &displayX, &displayY,
                                                  &depth) &&
                  std::hypot(displayX - rawX, displayY - rawY) > 4.0f &&
                  Pomade_Pick(cc, oblique, 1024, 1024, displayX, displayY,
                             2.0f, usdGenPomade::PomadePick_GraphNode, &hit,
                             &kind, &pickId, &sub, &rawDistance, &rawDepth) ==
                      POMADE_OK &&
                  hit == 0 &&
                  Pomade_PickItem(cc, oblique, 1024, 1024, displayX, displayY,
                                 2.0f, usdGenPomade::PomadePick_GraphNode, &hit,
                                 &kind, &pickId, &sub, &subSub) == POMADE_OK &&
                  hit == 1 && kind == usdGenPomade::PomadePick_GraphNode &&
                  pickId == 1 &&
                  Pomade_SelectRect(cc, oblique, 1024, 1024, displayX - 1.0f,
                                   displayY - 1.0f, displayX + 1.0f,
                                   displayY + 1.0f,
                                   usdGenPomade::PomadePick_GraphNode,
                                   usdGenPomade::PomadeSelect_Set) == POMADE_OK &&
                  Pomade_GetSelectionCount(cc,
                                          usdGenPomade::PomadePick_GraphNode) == 1,
              "shared CV: rendered endpoint picks and boxes independently of raw K11");
        // That first oblique camera moves the normal lift along this
        // vertical edge's projected line. Turn the camera so the lift is
        // perpendicular to the line; raw K11 must then miss the rendered
        // edge while the display-aware item picker finds its stable id.
        float edgeView[16] = {};
        edgeView[0] = 1.0f;
        edgeView[4] = 20.0f;  // screen x <- x + lifted normal
        edgeView[9] = 1.0f;   // screen y <- z (the raw edge direction)
        edgeView[10] = 1.0f;
        edgeView[13] = -0.50f;
        edgeView[15] = 1.0f;
        float rawNode2[3] = {}, displayNode2[3] = {}, edgeDisplay[3] = {};
        Check(Pomade_GraphGetNode(cc, 2, &face, uv2, rawNode2) == POMADE_OK &&
                  Pomade_GraphGetNodeDisplayPosition(cc, 2, displayNode2) ==
                      POMADE_OK,
              "shared CV: the second endpoint supplies the edge midpoint");
        for (int axis = 0; axis != 3; ++axis) {
            edgeDisplay[axis] = 0.5f *
                (displayNode[axis] + displayNode2[axis]);
        }
        float edgeX = 0.0f, edgeY = 0.0f;
        hit = 0;
        Check(usdGenPomade::PomadeProjectPoint(edgeDisplay, edgeView, 1024,
                                              1024, &edgeX, &edgeY, &depth) &&
                  Pomade_Pick(cc, edgeView, 1024, 1024, edgeX, edgeY, 2.0f,
                         usdGenPomade::PomadePick_GraphEdge, &hit, &kind,
                         &pickId, &sub, &rawDistance, &rawDepth) == POMADE_OK &&
                  hit == 0 &&
                  Pomade_PickItem(cc, edgeView, 1024, 1024, edgeX, edgeY,
                                 2.0f, usdGenPomade::PomadePick_GraphEdge, &hit,
                                 &kind, &pickId, &sub, &subSub) == POMADE_OK &&
                  hit == 1 && kind == usdGenPomade::PomadePick_GraphEdge &&
                  pickId == 1,
              "shared CV: rendered edge candidates align without changing raw K11");
        int const shared[] = {1, -1, -1, 2};
        float const rightUV[] = {0.0f, 0.0f, 0.85f, 0.20f,
                                 0.85f, 0.80f, 0.0f, 0.0f};
        int sharedRegion = -1;
        int const undoBefore = Pomade_GetUndoDepth(cc);
        Check(Pomade_GraphCreateRegion(cc, shared, faces, rightUV, 4,
                                      &sharedRegion) == POMADE_OK &&
                  sharedRegion >= 0 && sharedRegion != leftRegion &&
                  Pomade_GetUndoDepth(cc) == undoBefore + 1,
              "shared CV: one atomic undo step creates the adjacent region");
        Check(Pomade_GetGraphCounts(cc, &nodeCount, &edgeCount, &regionCount) ==
                        POMADE_OK &&
                  nodeCount == 6 && edgeCount == 7 && regionCount == 2,
              "shared CV: the existing boundary edge is reused once");
        int const invalid[] = {1, -1, 999, 2};
        int invalidRegion = -1;
        Check(Pomade_GraphCreateRegion(cc, invalid, faces, rightUV, 4,
                                      &invalidRegion) == POMADE_ERROR &&
                  Pomade_GetGraphCounts(cc, &nodeCount, &edgeCount, &regionCount) ==
                      POMADE_OK &&
                  nodeCount == 6 && edgeCount == 7 && regionCount == 2 &&
                  Pomade_GetUndoDepth(cc) == undoBefore + 1,
              "shared CV: an invalid stable id rolls back without an undo step");
        Check(Pomade_Undo(cc, nullptr) == POMADE_OK &&
                  Pomade_GetGraphCounts(cc, &nodeCount, &edgeCount, &regionCount) ==
                      POMADE_OK &&
                  nodeCount == 4 && edgeCount == 4 && regionCount == 1 &&
                  Pomade_Redo(cc, nullptr) == POMADE_OK,
              "shared CV: undo and redo preserve the whole shared chain");
        Check(Pomade_Destroy(cc) == POMADE_OK, "shared CV: model destroys");
    }

    // -- graph edits are on the same undo stack (plan/18 §2.5) -------------
    {
        PomadeModelContext *gc = nullptr;
        Check(Pomade_Create(&gc) == POMADE_OK && gc != nullptr,
              "V1: graph undo model creates");
        // A 2 x 2 quad grid in XZ at y = 0: face f = ix * 2 + iz.
        std::vector<float> points;
        for (int ix = 0; ix < 3; ++ix) {
            for (int iz = 0; iz < 3; ++iz) {
                points.push_back(float(ix));
                points.push_back(0.0f);
                points.push_back(float(iz));
            }
        }
        std::vector<int> counts(4, 4);
        std::vector<int> indices;
        for (int ix = 0; ix < 2; ++ix) {
            for (int iz = 0; iz < 2; ++iz) {
                int const a = ix * 3 + iz;
                indices.push_back(a);
                indices.push_back(a + 3);
                indices.push_back(a + 4);
                indices.push_back(a + 1);
            }
        }
        Check(Pomade_BindScalp(gc, points.data(), int(points.size()),
                              counts.data(), int(counts.size()),
                              indices.data(), int(indices.size())) ==
                  POMADE_OK,
              "V1: the scalp binds");
        int nodeA = -1, nodeB = -1;
        Check(Pomade_GraphAddNode(gc, 0, 0.25f, 0.25f, &nodeA) == POMADE_OK &&
                  nodeA >= 0,
              "V1: first graph node");
        Check(Pomade_GetUndoDepth(gc) == 1,
              "V1: a graph edit pushes an undo step");
        Check(Pomade_GraphAddNode(gc, 3, 0.75f, 0.75f, &nodeB) == POMADE_OK &&
                  nodeB >= 0,
              "V1: second graph node");
        int nodes = 0, edges = 0, regions = 0;
        Check(Pomade_GetGraphCounts(gc, &nodes, &edges, &regions) ==
                  POMADE_OK && nodes == 2,
              "V1: two nodes are in the graph");
        {
            char label[32] = {0};
            Check(Pomade_GetUndoLabel(gc, 0, label, sizeof(label)) ==
                      POMADE_OK && std::string(label) == "add node",
                  "V1: the graph step names itself");
        }
        Check(Pomade_Undo(gc, nullptr) == POMADE_OK &&
                  Pomade_GetGraphCounts(gc, &nodes, nullptr, nullptr) ==
                      POMADE_OK && nodes == 1,
              "V1: undoing a graph edit removes the node");
        Check(Pomade_Redo(gc, nullptr) == POMADE_OK &&
                  Pomade_GetGraphCounts(gc, &nodes, nullptr, nullptr) ==
                      POMADE_OK && nodes == 2,
              "V1: redoing a graph edit puts it back");
        int edgeId = -1;
        Check(Pomade_GraphConnect(gc, nodeA, nodeB, &edgeId) == POMADE_OK &&
                  edgeId >= 0,
              "V1: the two nodes connect");
        Check(Pomade_GetGraphCounts(gc, nullptr, &edges, nullptr) ==
                  POMADE_OK && edges == 1,
              "V1: one edge exists");
        Check(Pomade_Undo(gc, nullptr) == POMADE_OK &&
                  Pomade_GetGraphCounts(gc, nullptr, &edges, nullptr) ==
                      POMADE_OK && edges == 0,
              "V1: undoing the connect removes the edge");
        Check(Pomade_Destroy(gc) == POMADE_OK, "V1: graph undo model destroys");
    }

    // -- SS-05: committer failures reach the caller ----------------------
    // A bound scalp with no tube is a real model version the committer
    // cannot build ("snapshot holds no tubes"): the worker parks the
    // version as failed and leaves the reason for TakeDiagnostic. No stage
    // is needed for any of it, and the swap-error half needs no layer.
    {
        Check(Pomade_CommitterTakeDiagnostic(nullptr, nullptr, 0) == -1,
              "SS-05: TakeDiagnostic rejects a null committer");
        Check(Pomade_CommitterFailedVersion(nullptr) == 0,
              "SS-05: FailedVersion(null) answers 0");
        Check(Pomade_CommitterSwap(nullptr, "x", 0) == PomadeCommitter_Error,
              "SS-05: Swap(null) answers PomadeCommitter_Error");
        Check(PomadeCommitter_Error != PomadeCommitter_PartialProgress &&
                  PomadeCommitter_Error < 0,
              "SS-05: the swap error code cannot read as partial progress");

        PomadeModelContext *fc = nullptr;
        Check(Pomade_Create(&fc) == POMADE_OK && fc != nullptr,
              "SS-05: failing-commit model creates");
        float const quad[] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                              1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
        int const counts[] = {4};
        int const indices[] = {0, 3, 2, 1};
        Check(fc && Pomade_BindScalp(fc, quad, 12, counts, 1, indices, 4) ==
                        POMADE_OK && Pomade_GetVersion(fc) > 0,
              "SS-05: a one-quad scalp binds (a version with no tube)");
        PomadeCommitterContext *fcc = nullptr;
        Check(fc && Pomade_CommitterCreate(fc, "/PomadeGroom", nullptr, &fcc) ==
                        POMADE_OK && fcc != nullptr,
              "SS-05: the committer creates");
        if (fc && fcc) {
            Check(Pomade_CommitterFailedVersion(fcc) == 0,
                  "SS-05: a fresh committer has no failed version");
            char none[8] = {'x', 0};
            Check(Pomade_CommitterTakeDiagnostic(fcc, none, sizeof(none)) ==
                      0 && none[0] == '\0',
                  "SS-05: nothing to take yet (empty string, length 0)");
            Check(Pomade_CommitterEnqueue(fcc, 0, 0, 0, nullptr) == POMADE_OK,
                  "SS-05: enqueue of the tubeless version succeeds");
            unsigned long long const version = Pomade_GetVersion(fc);
            auto const deadline = std::chrono::steady_clock::now() +
                                  std::chrono::milliseconds(10000);
            while (Pomade_CommitterFailedVersion(fcc) != version &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            Check(Pomade_CommitterFailedVersion(fcc) == version,
                  "SS-05: FailedVersion names the version the worker "
                  "could not build");
            char text[512] = {0};
            int const length =
                Pomade_CommitterTakeDiagnostic(fcc, text, sizeof(text));
            Check(length > 0 && size_t(length) == std::strlen(text) &&
                      std::strstr(text, "no tubes") != nullptr,
                  std::string("SS-05: TakeDiagnostic returns the build "
                              "failure (") + text + ")");
            Check(Pomade_CommitterTakeDiagnostic(fcc, text, sizeof(text)) ==
                      0 && text[0] == '\0',
                  "SS-05: and the take cleared it (reported once)");
            Check(Pomade_CommitterEnqueue(fcc, 0, 0, 0, nullptr) == POMADE_OK,
                  "SS-05: a retry enqueues the same version again");
            auto const retryDeadline = std::chrono::steady_clock::now() +
                                       std::chrono::milliseconds(10000);
            char retry[4] = {0};
            int retryLength = 0;
            while (retryLength == 0 &&
                   std::chrono::steady_clock::now() < retryDeadline) {
                retryLength =
                    Pomade_CommitterTakeDiagnostic(fcc, retry, sizeof(retry));
                if (retryLength == 0) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(1));
                }
            }
            Check(retryLength >= int(sizeof(retry)) &&
                      std::strlen(retry) == sizeof(retry) - 1,
                  "SS-05: the retry rebuilds (and fails again); a short "
                  "buffer truncates with a NUL and reports the full length");
            Check(Pomade_CommitterSwap(fcc, "no-such-layer", 0) ==
                      PomadeCommitter_Error &&
                      std::strstr(Pomade_GetLastError(), "no-such-layer") !=
                          nullptr,
                  "SS-05: a swap into an unknown live layer answers "
                  "PomadeCommitter_Error and names the layer");
            Check(Pomade_CommitterCommittedVersion(fcc) == 0,
                  "SS-05: nothing was committed");
        }
        Check(Pomade_CommitterDestroy(fcc) == POMADE_OK,
              "SS-05: the committer destroys");
        Check(Pomade_Destroy(fc) == POMADE_OK, "SS-05: the model destroys");
    }

    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
