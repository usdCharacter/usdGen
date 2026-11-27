// Ownership regression coverage for immutable imaging registry reads and
// scheduled bridge mutation.
#include "usdGen/api/pipeline.h"
#include "usdGen/api/stats.h"
#include "usdGenImaging/testHook.h"
#include "usdGenImaging/usdGenEngineBridge.h"

#include "pxr/imaging/hd/retainedSceneIndex.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {
int failures = 0;

void Check(bool value, char const *message)
{
    if (!value) {
        ++failures;
        std::printf("FAIL: %s\n", message);
    }
}

usdGen::UsdGenStats Stats(size_t head, int64_t firstGeneration)
{
    usdGen::UsdGenStats stats;
    stats.ringHead = head;
    for (size_t i = 0; i != stats.ring.size(); ++i)
        stats.ring[i].generation = firstGeneration + int64_t(i);
    return stats;
}

void TestBridgeOrderingAndReentry()
{
    std::vector<int64_t> seen;
    auto bridge = std::make_unique<usdGenImaging::UsdGenEngineBridge>();
    usdGen::UsdGenStats followUp = Stats(3, 0);

    bridge->SetTimingHook([&](usdGen::UsdGenCommitTiming const &timing) {
        seen.push_back(timing.generation);
        if (timing.generation == 0) bridge->DrainStats(followUp);
        if (timing.generation == 2) bridge.reset();
    });
    // First drain establishes the cursor without replaying ring history.
    bridge->DrainStats(Stats(0, 0));
    bridge->DrainStats(Stats(2, 0));

    // This unrelated instance provides the external-only drain boundary once
    // the callback has destroyed the bridge which submitted the messages.
    usdGenImaging::UsdGenEngineBridge flush;
    flush.Drain();

    Check(!bridge, "bridge can be destroyed from its timing callback");
    Check(seen == std::vector<int64_t>({0, 1, 2}),
          "hook ordering and callback re-entry preserve the stats cursor");
}

void TestQueuedRegistryMutationDoesNotBorrowIndex()
{
    size_t const baseline = UsdGenImagingTestHook::registeredIndexCount();
    auto index = HdRetainedSceneIndex::New();
    usdGen::UsdGenExecutionRuntime runtime(1);
    usdGen::UsdGenExecutionPipeline callback(runtime);
    callback.InvokeOwner([&] {
        // IsExecuting() makes registration asynchronous.  The registry must
        // retain only weak-control identity, not this raw address.
        UsdGenImagingTestHook::_RegisterIndex(index.operator->());
    });
    callback.Drain();
    UsdGenImagingTestHook::Drain();
    Check(UsdGenImagingTestHook::registeredIndexCount() == baseline + 1,
          "async live registration is visible after the hook owner drains");

    callback.InvokeOwner([&] {
        UsdGenImagingTestHook::_UnregisterIndex(index.operator->());
    });
    callback.Drain();
    UsdGenImagingTestHook::Drain();
    Check(UsdGenImagingTestHook::registeredIndexCount() == baseline,
          "async unregister removes exactly its weak-control identity");

    callback.InvokeOwner([&] {
        UsdGenImagingTestHook::_RegisterIndex(index.operator->());
        index = nullptr;
    });
    callback.Drain();
    UsdGenImagingTestHook::Drain();
    Check(UsdGenImagingTestHook::registeredIndexCount() == baseline + 1,
          "expired queued weak entry is published without borrowing its index");
    Check(UsdGenImagingTestHook::publishedGeneration(SdfPath("/missing")) == 0,
          "queued expired registry entry is safe to read after index retirement");

    auto replacement = HdRetainedSceneIndex::New();
    UsdGenImagingTestHook::_RegisterIndex(replacement.operator->());
    UsdGenImagingTestHook::Drain();
    Check(UsdGenImagingTestHook::registeredIndexCount() == baseline + 1,
          "next mutation prunes expired identity before adding a live index");
    UsdGenImagingTestHook::_UnregisterIndex(replacement.operator->());
    UsdGenImagingTestHook::Drain();
    Check(UsdGenImagingTestHook::registeredIndexCount() == baseline,
          "replacement unregister restores the registry baseline");
}
} // namespace

int main()
{
    TestBridgeOrderingAndReentry();
    TestQueuedRegistryMutationDoesNotBorrowIndex();
    return failures == 0 ? 0 : 1;
}
