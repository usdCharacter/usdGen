// Session-key and lifecycle contract checks (stage-free imaging store).
#include "usdGenImaging/usdGenImagingSession.h"
#include "usdGenUsd/resolve/imageMapCache.h"

#include <cstdio>

using namespace usdGenImaging;

namespace {
int failures = 0;
void Check(bool value, char const *what)
{
    if (!value) { ++failures; std::printf("FAIL: %s\n", what); }
    else std::printf("ok: %s\n", what);
}
}

int main()
{
    UsdGenSessionStore &store = UsdGenSessionStore::GetInstance();
    const SdfPath pathA("/__sessionIsolationA");
    const SdfPath pathB("/__sessionIsolationB");

    UsdGenSessionKey emptyA{"", pathA, 1001};
    UsdGenSessionKey emptyB{"", pathA, 1002};
    Check(!(emptyA == emptyB), "empty session ids use render instance fallback");
    auto a = store.Attach(emptyA);
    auto b = store.Attach(emptyB);
    Check(a && b && a != b, "independent empty-id sessions do not merge");
    Check(a->AttachedIndices() == 1 && b->AttachedIndices() == 1,
          "independent sessions have independent attach counts");

    UsdGenSessionKey explicitA{"shared-session", pathA, 2001};
    UsdGenSessionKey explicitB{"shared-session", pathA, 2002};
    Check(explicitA == explicitB, "explicit id ignores render instance");
    Check(UsdGenSessionKeyHash()(explicitA) == UsdGenSessionKeyHash()(explicitB),
          "explicit equivalent keys hash equally");
    auto sharedA = store.Attach(explicitA);
    auto sharedB = store.Attach(explicitB);
    Check(sharedA == sharedB, "explicit equivalent sessions merge");
    Check(sharedA->AttachedIndices() == 2, "merged session counts both attachments");

    // The same authored id remains shareable for CPU publication, but a
    // renderer-local CUDA/GL session owns registry-specific GPU resources and
    // must isolate each renderer instance.
    UsdGenSessionKey gpuA{"shared-session", pathA, 3001, true};
    UsdGenSessionKey gpuB{"shared-session", pathA, 3002, true};
    Check(!(gpuA == gpuB), "renderer-local GPU keys include render instance");
    auto gpuSessionA = store.Attach(gpuA);
    auto gpuSessionB = store.Attach(gpuB);
    Check(gpuSessionA && gpuSessionB && gpuSessionA != gpuSessionB &&
              gpuSessionA != sharedA && gpuSessionB != sharedA,
          "same authored GPU session id never shares across renderer registries");
    UsdGenSessionKey cpuAfterGpu{"shared-session", pathA, 4001, false};
    auto sharedCpu = store.Attach(cpuAfterGpu);
    Check(cpuAfterGpu == explicitA && sharedCpu == sharedA,
          "CPU sharing resumes after a renderer-local GPU session");
    Check(sharedA->AttachedIndices() == 3,
          "CPU reattachment is independent of isolated GPU attachments");

    uint64_t const imageGeneration = CurrentUsdGenImageMapGeneration();
    Check(store.ReloadMaps(), "store accepts global image-map cache invalidation");
    store.Drain();
    // This synchronous command follows the reload command already posted by
    // the store and therefore observes its session-local descriptor action.
    sharedA->Commit(usdGen::UsdGenCommitReason::LiveOverride);
    Check(CurrentUsdGenImageMapGeneration() > imageGeneration &&
              sharedA->NeedsDesc(),
          "store reload advances image generation and marks unstaged sessions");
    Check(sharedA->ConsumeNeedsDesc(),
          "store reload description mark is consumable");

    UsdGenSessionKey differentPath{"shared-session", pathB, 2001};
    UsdGenSessionKey differentId{"other-session", pathA, 2001};
    Check(!(explicitA == differentPath) && !(explicitA == differentId),
          "path and explicit-id changes isolate sessions");
    auto pathSession = store.Attach(differentPath);
    auto idSession = store.Attach(differentId);
    Check(pathSession != sharedA && idSession != sharedA,
          "path and id variants do not share state");

    // A mark arriving after the stager consumed an earlier mark belongs to the
    // next commit and must not be cleared by Commit itself.
    sharedA->MarkNeedsDesc();
    Check(sharedA->ConsumeNeedsDesc(), "first description mark is consumable");
    sharedA->MarkNeedsDesc();
    Check(sharedA->NeedsDesc(), "second description mark is pending");
    sharedA->Commit(usdGen::UsdGenCommitReason::LiveOverride);
    Check(sharedA->NeedsDesc(), "pending description mark survives Commit");
    Check(sharedA->ConsumeNeedsDesc(), "pending mark can be consumed next cycle");
    Check(!sharedA->ConsumeNeedsDesc(), "repeated consume is idempotent");

    // Detach each shared CPU attachment. Keep the local strong pointer to
    // verify the object remains safe after the store drops its entry.
    store.Detach(explicitA);
    Check(sharedB->AttachedIndices() == 2, "detach decrements merged count");
    store.Detach(explicitB);
    Check(sharedB->AttachedIndices() == 1, "second CPU detach retains recovery attachment");
    store.Detach(cpuAfterGpu);
    Check(!store.Find(explicitA), "final detach removes store entry");
    Check(sharedB->AttachedIndices() == 0, "final detach reaches zero");

    store.Detach(emptyA);
    store.Detach(emptyB);
    store.Detach(differentPath);
    store.Detach(differentId);
    store.Detach(gpuA);
    store.Detach(gpuB);
    Check(!store.Find(emptyA) && !store.Find(emptyB), "empty-id sessions detach independently");
    Check(store.LiveSessions().empty(), "all test sessions detached");
    std::printf("testUsdGenSessionIsolation: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
