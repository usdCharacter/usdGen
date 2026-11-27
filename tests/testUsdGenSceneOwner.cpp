// Scene-owner lifecycle tests: adoption is explicit-owner work, while scene
// reads remain snapshot-only and removal cannot leave stale ownership behind.
#include "utils/pxrInterop.h"
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenImaging/testHook.h"

#include "pxr/imaging/hd/retainedSceneIndex.h"
#include "pxr/imaging/hd/retainedDataSource.h"

#include <cstdio>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {
int failures = 0;
void Check(bool value, char const *what)
{
    if (!value) { ++failures; std::printf("FAIL: %s\n", what); }
    else std::printf("ok:   %s\n", what);
}

class Observer final : public HdSceneIndexObserver {
public:
    std::vector<SdfPath> added, removed, dirtied;
    UsdGenGroomSceneIndex *owner = nullptr;
    bool synchronizeRejected = false;
    bool callbackReadSafe = false;
    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &entries) override {
        for (auto const &entry : entries) added.push_back(entry.primPath);
        if (owner) {
            callbackReadSafe = owner->GetPrim(SdfPath("/__sceneOwnerGroom")).primType ==
                               TfToken("UsdGenGroom");
            try { owner->Synchronize(); }
            catch (std::logic_error const &) { synchronizeRejected = true; }
        }
    }
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &entries) override {
        for (auto const &entry : entries) removed.push_back(entry.primPath);
    }
    void PrimsDirtied(HdSceneIndexBase const &, DirtiedPrimEntries const &entries) override {
        for (auto const &entry : entries) dirtied.push_back(entry.primPath);
    }
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &) override {}
};

class DropIndexObserver final : public HdSceneIndexObserver {
public:
    HdSceneIndexBaseRefPtr *slot = nullptr;
    std::atomic<bool> dropped{false};
    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &entries) override {
        for (auto const &entry : entries) {
            if (entry.primPath.GetString().find("__usdGenRender") == std::string::npos)
                continue;
            if (slot) slot->Reset();
            dropped.store(true, std::memory_order_release);
            break;
        }
    }
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &) override {}
    void PrimsDirtied(HdSceneIndexBase const &, DirtiedPrimEntries const &) override {}
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &) override {}
};

bool Has(std::vector<SdfPath> const &paths, SdfPath const &path)
{
    for (auto const &candidate : paths)
        if (candidate == path) return true;
    return false;
}

template <class Predicate>
bool WaitFor(Predicate predicate, int milliseconds = 10000)
{
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(milliseconds);
    while (!predicate() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    return predicate();
}

template <class Predicate>
void WaitOrAbort(Predicate predicate, char const *what)
{
    if (!WaitFor(predicate)) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::abort();
    }
}
} // namespace

int main()
{
    auto input = HdRetainedSceneIndex::New();
    const SdfPath root("/__sceneOwnerGroom");
    const int renderInstance = 4801;
    const usdGenImaging::UsdGenSessionKey key{"", root, renderInstance};
    auto index = UsdGenGroomSceneIndex::New(input, renderInstance);
    auto *owner = dynamic_cast<UsdGenGroomSceneIndex *>(index.operator->());
    Check(owner != nullptr, "scene owner index has concrete owner type");
    if (!owner) return 1;
    Observer observer;
    observer.owner = owner;
    index->AddObserver(TfCreateWeakPtr(&observer));

    input->AddPrims({{root, TfToken("UsdGenGroom"),
                      HdRetainedContainerDataSource::New()}});
    owner->Synchronize();
    auto first = usdGenImaging::UsdGenSessionStore::GetInstance().Find(key);
    Check(first && first->AttachedIndices() == 1,
          "owner synchronizes one adopted groom attachment");
    Check(index->GetPrim(root).primType == TfToken("UsdGenGroom"),
          "GetPrim reads adopted groom snapshot");
    Check(observer.callbackReadSafe && observer.synchronizeRejected,
          "observer callback can read snapshots but cannot synchronously resync");

    // A same-path type replacement must be treated as a removal from the
    // owner registry even when no explicit PrimsRemoved arrives first.
    input->AddPrims({{root, TfToken("Scope"),
                      HdRetainedContainerDataSource::New()}});
    owner->Synchronize();
    auto &store = usdGenImaging::UsdGenSessionStore::GetInstance();
    Check(!store.Find(key), "root resync to Scope removes session ownership");
    Check(index->GetChildPrimPaths(root).empty(),
          "root resync to Scope removes synthetic render children");

    // Removal must erase ownership before forwarding its notice; the old
    // handle remains safe as a strong snapshot but cannot publish stale data.
    input->RemovePrims({{root}});
    owner->Synchronize();
    Check(!usdGenImaging::UsdGenSessionStore::GetInstance().Find(key),
          "removal drops session ownership");
    Check(Has(observer.removed, root), "removal notice reaches downstream observer");
    Check(first && first->AttachedIndices() == 0,
          "retained old handle observes balanced detach");

    // Re-adoption at the same path gets a fresh identity and does not revive
    // the retired snapshot or duplicate ownership.
    observer.removed.clear();
    input->AddPrims({{root, TfToken("UsdGenGroom"),
                      HdRetainedContainerDataSource::New()}});
    owner->Synchronize();
    auto replacement = usdGenImaging::UsdGenSessionStore::GetInstance().Find(key);
    Check(replacement && replacement != first && replacement->AttachedIndices() == 1,
          "re-adoption installs one fresh session identity");
    Check(index->GetPrim(root).primType == TfToken("UsdGenGroom"),
          "re-adopted groom is readable without a synchronous cook");

    // Registry test hook reads are external snapshot boundaries and remain
    // valid while the scene index is being retired.
    const size_t registered = UsdGenImagingTestHook::registeredIndexCount();
    index.Reset();
    UsdGenImagingTestHook::Drain();
    Check(UsdGenImagingTestHook::registeredIndexCount() < registered,
          "destroyed scene index is removed from owner registry");
    Check(WaitFor([&] {
              return !replacement || replacement->AttachedIndices() == 0;
          }), "scene owner retirement balances replacement detach");
    replacement.Reset();
    first.Reset();

    // Destroy the last external scene-index reference from a synthetic
    // render-scope callback.  The owner state must retain itself until its
    // queued detach completes; no callback may dereference the destroyed
    // index afterward.
    {
        auto dropInput = HdRetainedSceneIndex::New();
        const SdfPath dropRoot("/__sceneOwnerDrop");
        const usdGenImaging::UsdGenSessionKey dropKey{"", dropRoot, 4902};
        HdSceneIndexBaseRefPtr dropIndex =
            UsdGenGroomSceneIndex::New(dropInput, 4902);
        auto *dropOwner = dynamic_cast<UsdGenGroomSceneIndex *>(dropIndex.operator->());
        Check(dropOwner != nullptr, "drop callback owner cast succeeds");
        if (dropOwner) dropOwner->Synchronize();
        DropIndexObserver dropObserver;
        dropObserver.slot = &dropIndex;
        dropIndex->AddObserver(TfCreateWeakPtr(&dropObserver));
        dropInput->AddPrims({{dropRoot, TfToken("UsdGenGroom"),
                              HdRetainedContainerDataSource::New()}});
        WaitOrAbort([&] { return dropObserver.dropped.load(std::memory_order_acquire); },
                    "synthetic render callback releases final index reference");
        UsdGenGroomSceneIndex::DrainRetired();
        UsdGenImagingTestHook::Drain();
        auto &dropStore = usdGenImaging::UsdGenSessionStore::GetInstance();
        WaitOrAbort([&] { return !dropStore.Find(dropKey); },
                    "callback-destroyed scene owner balances attachment");
    }

    // Completed final-reference cleanup must compact the actual SceneService
    // registry without an optional external DrainRetired reclamation pass.
    UsdGenImagingTestHook::groomSceneServiceCommandBarrier();
    WaitOrAbort([] {
        return UsdGenImagingTestHook::retainedGroomSceneStateCount() == 0 &&
            UsdGenImagingTestHook::liveGroomSceneStateCount() == 0 &&
            UsdGenImagingTestHook::groomRetirementRecordCount() == 0;
    }, "scene-service baseline cleanup completes without DrainRetired");
    size_t const stateBaseline = UsdGenImagingTestHook::retainedGroomSceneStateCount();
    size_t const liveStateBaseline = UsdGenImagingTestHook::liveGroomSceneStateCount();
    size_t const recordBaseline = UsdGenImagingTestHook::groomRetirementRecordCount();
    constexpr unsigned kChurn = 256;
    for (unsigned i = 0; i != kChurn; ++i) {
        auto churnInput = HdRetainedSceneIndex::New();
        auto churnIndex = UsdGenGroomSceneIndex::New(churnInput, 5000 + int(i));
        churnIndex.Reset();
    }
    std::vector<std::thread> concurrentChurn;
    for (unsigned worker = 0; worker != 4; ++worker) {
        concurrentChurn.emplace_back([worker] {
            for (unsigned i = 0; i != 32; ++i) {
                auto churnInput = HdRetainedSceneIndex::New();
                auto churnIndex = UsdGenGroomSceneIndex::New(
                    churnInput, 6000 + int(worker * 32 + i));
                churnIndex.Reset();
            }
        });
    }
    for (auto &worker : concurrentChurn) worker.join();
    UsdGenImagingTestHook::groomSceneServiceCommandBarrier();
    Check(WaitFor([&] {
              return UsdGenImagingTestHook::retainedGroomSceneStateCount() == stateBaseline &&
                  UsdGenImagingTestHook::liveGroomSceneStateCount() == liveStateBaseline &&
                  UsdGenImagingTestHook::groomRetirementRecordCount() == recordBaseline;
          }),
          "completed scene churn reclaims registry states and retirement records without DrainRetired");

    // A second index can repeatedly detach/re-attach a groom that remains
    // anchored by a first index.  Each target re-attach is real, yet shares
    // the same store session identity and must not duplicate usedSessions.
    {
        auto anchorInput = HdRetainedSceneIndex::New();
        auto targetInput = HdRetainedSceneIndex::New();
        SdfPath const dedupRoot("/__sceneOwnerDedup");
        constexpr int dedupRenderInstance = 5101;
        anchorInput->AddPrims({{dedupRoot, TfToken("UsdGenGroom"),
                               HdRetainedContainerDataSource::New()}});
        targetInput->AddPrims({{dedupRoot, TfToken("UsdGenGroom"),
                               HdRetainedContainerDataSource::New()}});
        auto anchorIndex = UsdGenGroomSceneIndex::New(anchorInput, dedupRenderInstance);
        auto targetIndex = UsdGenGroomSceneIndex::New(targetInput, dedupRenderInstance);
        auto *anchorOwner = dynamic_cast<UsdGenGroomSceneIndex *>(anchorIndex.operator->());
        auto *targetOwner = dynamic_cast<UsdGenGroomSceneIndex *>(targetIndex.operator->());
        Check(anchorOwner != nullptr && targetOwner != nullptr,
              "shared-session dedup fixture creates both owners");
        if (anchorOwner && targetOwner) {
            anchorOwner->Synchronize();
            targetOwner->Synchronize();
            usdGenImaging::UsdGenSessionKey const dedupKey{
                "", dedupRoot, dedupRenderInstance};
            auto shared = usdGenImaging::UsdGenSessionStore::GetInstance().Find(dedupKey);
            bool attachedTwice = shared && shared->AttachedIndices() == 2;
            // Create an expired tail B between two uses of live shared A.
            // The next A re-attach must both prune B and avoid appending A.
            SdfPath const expiredTail("/__sceneOwnerDedupExpiredTail");
            targetInput->RemovePrims({{dedupRoot}});
            targetOwner->Synchronize();
            targetInput->AddPrims({{expiredTail, TfToken("UsdGenGroom"),
                                    HdRetainedContainerDataSource::New()}});
            targetOwner->Synchronize();
            targetInput->RemovePrims({{expiredTail}});
            targetOwner->Synchronize();
            Check(WaitFor([&] {
                      return UsdGenImagingTestHook::groomUsedSessionWeakCount(*targetIndex) == 2 &&
                          UsdGenImagingTestHook::groomUsedSessionLiveUniqueCount(*targetIndex) == 1;
                  }),
                  "expired tail remains observable before the first shared-session re-attach");
            targetInput->AddPrims({{dedupRoot, TfToken("UsdGenGroom"),
                                    HdRetainedContainerDataSource::New()}});
            targetOwner->Synchronize();
            attachedTwice = attachedTwice && shared && shared->AttachedIndices() == 2;
            Check(UsdGenImagingTestHook::groomUsedSessionWeakCount(*targetIndex) == 1 &&
                      UsdGenImagingTestHook::groomUsedSessionLiveUniqueCount(*targetIndex) == 1,
                  "first shared-session re-attach prunes expired tail and avoids duplicate live identity");
            for (unsigned i = 0; i != 16; ++i) {
                targetInput->RemovePrims({{dedupRoot}});
                targetOwner->Synchronize();
                targetInput->AddPrims({{dedupRoot, TfToken("UsdGenGroom"),
                                        HdRetainedContainerDataSource::New()}});
                targetOwner->Synchronize();
                attachedTwice = attachedTwice && shared && shared->AttachedIndices() == 2;
            }
            Check(attachedTwice &&
                      UsdGenImagingTestHook::groomUsedSessionWeakCount(*targetIndex) == 1 &&
                      UsdGenImagingTestHook::groomUsedSessionLiveUniqueCount(*targetIndex) == 1,
                  "repeated shared-session re-attachments deduplicate used-session retention");
        }
        targetIndex.Reset();
        anchorIndex.Reset();
        UsdGenImagingTestHook::groomSceneServiceCommandBarrier();
        Check(WaitFor([&] {
                  return UsdGenImagingTestHook::retainedGroomSceneStateCount() == stateBaseline &&
                      UsdGenImagingTestHook::liveGroomSceneStateCount() == liveStateBaseline &&
                      UsdGenImagingTestHook::groomRetirementRecordCount() == recordBaseline;
              }),
              "deduplicated session fixture reclaims records without external drain");
    }

    // Preserve the final-reference -> deleter-enqueue gap contract.  The
    // registration record is present before the deleter pauses; DrainRetired
    // must wait for that record until the deleter is released.
    UsdGenImagingTestHook::armGroomFinalDeleterPauseForTesting();
    auto gapInput = HdRetainedSceneIndex::New();
    HdSceneIndexBaseRefPtr gapIndex = UsdGenGroomSceneIndex::New(gapInput, 5201);
    UsdGenImagingTestHook::groomSceneServiceCommandBarrier();
    std::atomic<bool> finalResetDone{false};
    std::thread finalReset([&] {
        gapIndex.Reset();
        finalResetDone.store(true, std::memory_order_release);
    });
    UsdGenImagingTestHook::waitGroomFinalDeleterPauseForTesting();
    Check(UsdGenImagingTestHook::retainedGroomSceneStateCount() == stateBaseline + 1 &&
              UsdGenImagingTestHook::liveGroomSceneStateCount() == liveStateBaseline &&
              UsdGenImagingTestHook::groomRetirementRecordCount() == recordBaseline + 1,
          "paused final deleter leaves an expired state weak entry protected by its record");
    std::atomic<bool> drainReturned{false};
    UsdGenImagingTestHook::armGroomDrainWaitForTesting();
    std::thread gapDrain([&] {
        UsdGenGroomSceneIndex::DrainRetired();
        drainReturned.store(true, std::memory_order_release);
    });
    // The test hook below signals that DrainRetired has found this expired
    // record and is about to wait; it avoids using scheduler timing as proof.
    UsdGenImagingTestHook::waitGroomDrainWaitForTesting();
    Check(!drainReturned.load(std::memory_order_acquire),
          "DrainRetired waits across the registered final-deleter enqueue gap");
    UsdGenImagingTestHook::releaseGroomFinalDeleterPauseForTesting();
    finalReset.join();
    gapDrain.join();
    UsdGenImagingTestHook::releaseGroomDrainWaitForTesting();
    Check(finalResetDone.load(std::memory_order_acquire) &&
              drainReturned.load(std::memory_order_acquire),
          "released final deleter completes the waiting retirement drain");
    UsdGenImagingTestHook::groomSceneServiceCommandBarrier();
    Check(WaitFor([&] {
              return UsdGenImagingTestHook::retainedGroomSceneStateCount() == stateBaseline &&
                  UsdGenImagingTestHook::liveGroomSceneStateCount() == liveStateBaseline &&
                  UsdGenImagingTestHook::groomRetirementRecordCount() == recordBaseline;
          }),
          "retirement-gap drain leaves no completed registry records behind");
    std::printf("testUsdGenSceneOwner: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
