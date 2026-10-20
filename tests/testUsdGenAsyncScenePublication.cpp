// Hydra notices belong to the serialized frontend, never a scene worker.
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenImaging/testHook.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/retainedSceneIndex.h"
#include "pxr/imaging/hd/systemMessages.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
PXR_NAMESPACE_USING_DIRECTIVE
namespace {
int failures=0;
void Check(bool okay,char const* message) {
    std::printf("%s: %s\n",okay?"ok":"FAIL",message);
    if(!okay) ++failures;
}
class Observer final : public HdSceneIndexObserver {
public:
    std::thread::id frontend=std::this_thread::get_id();
    std::atomic<bool> wrongThread{false};
    std::vector<std::string> events; // frontend only, even on a failing test
    UsdGenGroomSceneIndex* owner=nullptr;
    bool synchronizeRejected=false;
    std::function<void()> onAdd;
    bool OnFrontend() {
        if(std::this_thread::get_id()==frontend) return true;
        wrongThread.store(true); return false;
    }
    void PrimsAdded(HdSceneIndexBase const&,AddedPrimEntries const& entries) override {
        if(!OnFrontend()) return;
        for(auto const& entry:entries) events.push_back("A:"+entry.primPath.GetString());
        if (checkReentry) {
            try { owner->Synchronize(); }
            catch(std::logic_error const&) { synchronizeRejected=true; }
        }
        auto action=std::move(onAdd); onAdd={};
        if(action) action();
    }
    void PrimsRemoved(HdSceneIndexBase const&,RemovedPrimEntries const& entries) override {
        if(!OnFrontend()) return;
        for(auto const& entry:entries) events.push_back("R:"+entry.primPath.GetString());
    }
    void PrimsDirtied(HdSceneIndexBase const&,DirtiedPrimEntries const& entries) override {
        if(!OnFrontend()) return;
        for(auto const& entry:entries) {
            // Preserve locator precision in the compact event trace: only a
            // universal dirty can stand in for an omitted final-path add.
            events.push_back((entry.dirtyLocators == HdDataSourceLocatorSet::UniversalSet()
                ? "D:" : "d:") + entry.primPath.GetString());
        }
    }
    void PrimsRenamed(HdSceneIndexBase const&,RenamedPrimEntries const&) override {}
    bool checkReentry = true;
};
bool PollUntil(HdSceneIndexBase& index,std::function<bool()> const& predicate) {
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    do {
        index.SystemMessage(HdSystemMessageTokens->asyncPoll,nullptr);
        if(predicate()) return true;
        std::this_thread::yield();
    } while(std::chrono::steady_clock::now()<deadline);
    return false;
}
bool HasNetResync(std::vector<std::string> const& events, std::string const& path) {
    for (std::string const& event : events) {
        // Coalescing intentionally discards historical R/A pairs.  The one
        // delivered packet must either re-add the final prim or send a
        // universal dirty for that same prim.  Hydra dirties are not
        // hierarchical, so a root dirty cannot resync a child path.
        if (event == "A:" + path || event == "D:" + path) return true;
    }
    return false;
}
bool HasNoUnrelatedUniversalDirties(std::vector<std::string> const& events,
                                    std::string const& path) {
    for (std::string const& event : events)
        if (event.rfind("D:", 0) == 0 && event != "D:" + path) return false;
    return true;
}
}
int main() {
    auto input=HdRetainedSceneIndex::New();
    auto index=UsdGenGroomSceneIndex::New(input,6101);
    auto* owner=dynamic_cast<UsdGenGroomSceneIndex*>(index.operator->());
    if(!owner) return 1;
    Observer observer; observer.owner=owner;
    index->AddObserver(TfCreateWeakPtr(&observer));
    auto add=[&](char const* path,char const* type="Scope") {
        input->AddPrims({{SdfPath(path),TfToken(type),HdRetainedContainerDataSource::New()}});
    };
    index->SystemMessage(HdSystemMessageTokens->asyncPoll,nullptr);
    Check(observer.events.empty(),"poll before opt-in is harmless");
    add("/ordinary");
    Check(observer.events==std::vector<std::string>{"A:/ordinary"} && !observer.wrongThread,
          "filtered default notice arrives synchronously on original caller thread");
    Check(observer.synchronizeRejected,"explicit Synchronize re-entry is rejected");
    observer.events.clear();
    observer.onAdd=[&] { add("/nested"); };
    add("/outer");
    Check(observer.events==std::vector<std::string>{"A:/outer","A:/nested"},
          "default reentrant edit is delivered after outer packet and before return");
    index->SystemMessage(HdSystemMessageTokens->asyncAllow,nullptr);
    observer.events.clear();
    add("/deferred");
    Check(observer.events.empty() && !observer.wrongThread,
          "async ingress does not emit notices before polling");
    Check(PollUntil(*index,[&] { return !observer.events.empty(); }),"asyncPoll delivers completed work");
    Check(observer.events==std::vector<std::string>{"A:/deferred"} && !observer.wrongThread,
          "async notice is delivered once on polling thread");
    // Match usdview startup: AOVs are created after asyncAllow and used before
    // the next poll. Buffer resync and resize must also work in that interval.
    observer.checkReentry = false;
    observer.events.clear();
    add("/viewport/color", "renderBuffer");
    add("/viewport/depth", "renderBuffer");
    Check(observer.events == std::vector<std::string>{
              "A:/viewport/color", "A:/viewport/depth"},
          "color and depth buffers are announced before the first async poll");
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
    Check(observer.events.size() == 2, "poll does not replay AOV creation");
    observer.events.clear();
    input->RemovePrims({{SdfPath("/viewport")}});
    add("/viewport/color", "renderBuffer");
    input->DirtyPrims({{SdfPath("/viewport/color"),
        HdDataSourceLocatorSet{HdDataSourceLocator(TfToken("renderBuffer"))}}});
    std::vector<std::string> const bufferEvents{
        "R:/viewport/color", "R:/viewport/depth", "A:/viewport/color", "d:/viewport/color"};
    Check(observer.events == bufferEvents,
          "AOV subtree removal, replacement and resize are immediate");
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
    Check(observer.events == bufferEvents,
          "coalesced snapshots do not remove or replay the replacement AOV");
    // A deferred removal of an actual ancestor must re-add a surviving AOV.
    add("/viewport");
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
    observer.events.clear();
    input->RemovePrims({{SdfPath("/viewport")}});
    add("/viewport/color", "renderBuffer");
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
    Check(observer.events == std::vector<std::string>{
              "R:/viewport/color", "A:/viewport/color", "R:/viewport", "A:/viewport/color"},
          "deferred ancestor removal re-adds the live replacement buffer");
    observer.events.clear();
    add("/viewport/color", "Scope");
    Check(observer.events == std::vector<std::string>{"R:/viewport/color"},
          "changing a buffer to an ordinary prim removes the AOV immediately");
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
    Check(observer.events == std::vector<std::string>{"R:/viewport/color", "A:/viewport/color"},
          "the replacement ordinary prim is published on the next poll");
    input->RemovePrims({{SdfPath("/viewport")}});
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    index->SystemMessage(HdSystemMessageTokens->asyncPoll, nullptr);
    observer.checkReentry = true;
    observer.events.clear();
    input->RemovePrims({{SdfPath("/deferred")}});
    add("/deferred");
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    Check(observer.events.empty() &&
              UsdGenImagingTestHook::pendingGroomPublicationCount(*index) <= 1,
          "unpolled replacement coalesces to one pending frontend snapshot");
    index->SystemMessage(HdSystemMessageTokens->asyncPoll,nullptr);
    Check(index->GetPrim(SdfPath("/deferred")).dataSource &&
              HasNetResync(observer.events,"/deferred") &&
              HasNoUnrelatedUniversalDirties(observer.events,"/deferred"),
          "one poll exposes final replacement through a conservative net resync");

    // Keep the frontend idle while more than the former FIFO capacity of
    // source notices complete.  The owner-only drain makes every accepted
    // batch finish without granting a frontend delivery turn.
    observer.events.clear();
    uint64_t const capturesBeforeBurst = UsdGenImagingTestHook::groomCaptureCount(*index);
    uint64_t const cooksBeforeBurst = UsdGenImagingTestHook::groomCookCount(*index);
    constexpr unsigned kSourceBurst = 4098;
    SdfPath const transient("/transient");
    bool allSourceBatchesBounded = true;
    for (unsigned i = 0; i != kSourceBurst; ++i) {
        if ((i & 1u) == 0)
            add("/transient");
        else
            input->RemovePrims({{transient}});
        if ((i & 63u) == 63u) {
            UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
            allSourceBatchesBounded =
                UsdGenImagingTestHook::pendingGroomPublicationCount(*index) <= 1 &&
                allSourceBatchesBounded;
        }
    }
    // Exercise final-state replacement separately from the net-absent path.
    add("/netFinal");
    input->RemovePrims({{SdfPath("/netFinal")}});
    add("/netFinal");
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    uint64_t const capturesBeforePoll = UsdGenImagingTestHook::groomCaptureCount(*index);
    uint64_t const cooksBeforePoll = UsdGenImagingTestHook::groomCookCount(*index);
    Check(observer.events.empty() &&
              UsdGenImagingTestHook::pendingGroomPublicationCount(*index) <= 1 &&
              allSourceBatchesBounded &&
              capturesBeforePoll > capturesBeforeBurst && cooksBeforePoll == cooksBeforeBurst,
          "4098 completed source notices retain one net snapshot without cooking or frontend delivery");
    index->SystemMessage(HdSystemMessageTokens->asyncPoll,nullptr);
    Check(index->GetPrim(transient).primType.IsEmpty() &&
              index->GetPrim(SdfPath("/netFinal")).dataSource &&
              HasNetResync(observer.events,"/netFinal") &&
              HasNoUnrelatedUniversalDirties(observer.events,"/netFinal") &&
              UsdGenImagingTestHook::groomCaptureCount(*index) == capturesBeforePoll &&
              UsdGenImagingTestHook::groomCookCount(*index) == cooksBeforePoll,
          "one asyncPoll installs final net namespace without capture or cook work");
    // Async pass-through: an upstream dirty never waits for a poll.  It
    // reaches the observer at once, on the caller's thread, with the input's
    // own locators, and the next poll does not send it a second time.  The
    // absolute root (the scene globals' current frame) passes the same way.
    observer.events.clear();
    HdDataSourceLocatorSet const pointsOnly{HdDataSourceLocator(TfToken("points"))};
    input->DirtyPrims({{SdfPath("/netFinal"), pointsOnly},
                       {SdfPath::AbsoluteRootPath(), pointsOnly}});
    Check(observer.events == std::vector<std::string>{"d:/netFinal", "d:/"} && !observer.wrongThread,
          "async upstream dirties are forwarded immediately, before any poll");
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    index->SystemMessage(HdSystemMessageTokens->asyncPoll,nullptr);
    Check(observer.events == std::vector<std::string>{"d:/netFinal", "d:/"},
          "a poll never re-sends a dirty the pass-through already forwarded");
    // A namespace change still waits for the poll (its snapshot ordering).
    observer.events.clear();
    add("/afterPassThrough");
    Check(observer.events.empty(), "async namespace notices still wait for a poll");
    Check(PollUntil(*index,[&] { return !observer.events.empty(); }) &&
              observer.events == std::vector<std::string>{"A:/afterPassThrough"},
          "the poll delivers the deferred namespace notice once");
    // Authored roots are live upstream; synthetic namespaces advance only
    // with the corresponding frontend publication packet.
    observer.events.clear();
    add("/asyncGroom","UsdGenGroom");
    Check(index->GetChildPrimPaths(SdfPath("/asyncGroom")).empty(),
          "synthetic render scope is invisible before async delivery");
    Check(PollUntil(*index,[&] {
        return index->GetChildPrimPaths(SdfPath("/asyncGroom"))==
            SdfPathVector{SdfPath("/asyncGroom/__usdGenRender")};
    }),"poll installs matching synthetic snapshot");
    // An authored child at the virtual render path remains authoritative over
    // the synthetic scope in the final coalesced snapshot.
    observer.events.clear();
    SdfPath const collision("/asyncGroom/__usdGenRender");
    input->AddPrims({{collision,TfToken("Mesh"),HdRetainedContainerDataSource::New()}});
    UsdGenImagingTestHook::drainGroomOwnersWithoutFrontend(*index);
    index->SystemMessage(HdSystemMessageTokens->asyncPoll,nullptr);
    HdSceneIndexPrim const authoredCollision=input->GetPrim(collision);
    HdSceneIndexPrim const publishedCollision=index->GetPrim(collision);
    Check(publishedCollision.dataSource==authoredCollision.dataSource &&
              publishedCollision.primType==authoredCollision.primType &&
              HasNetResync(observer.events,"/asyncGroom/__usdGenRender") &&
              HasNoUnrelatedUniversalDirties(observer.events,"/asyncGroom/__usdGenRender"),
          "authored collision wins over the synthetic render scope after coalescing");
    observer.events.clear();
    input->RemovePrims({{SdfPath("/asyncGroom")}});
    owner->Synchronize();
    Check(index->GetChildPrimPaths(SdfPath("/asyncGroom")).empty(),
          "explicit frontend Synchronize installs completed removal");
    Check(!observer.wrongThread,"no notification escaped onto a worker");
    index->RemoveObserver(TfCreateWeakPtr(&observer));
    index.Reset();
    UsdGenGroomSceneIndex::DrainRetired();
    // Final-reference producers may arrive concurrently, while retirement
    // itself remains a single external drain boundary.
    std::atomic<bool> startRetirement{false};
    std::atomic<bool> releaseRetirement{false};
    std::atomic<unsigned> enteredRetirement{0};
    std::atomic<unsigned> completedRetirement{0};
    std::vector<std::thread> retirementProducers;
    for(int worker=0; worker!=4; ++worker) {
        retirementProducers.emplace_back([&,worker] {
            while(!startRetirement.load(std::memory_order_acquire)) std::this_thread::yield();
            auto localInput=HdRetainedSceneIndex::New();
            for(int cycle=0; cycle!=8; ++cycle) {
                auto transient=UsdGenGroomSceneIndex::New(localInput,7000+worker*16+cycle);
                enteredRetirement.fetch_add(1,std::memory_order_release);
                while(!releaseRetirement.load(std::memory_order_acquire)) std::this_thread::yield();
                transient.Reset();
                completedRetirement.fetch_add(1,std::memory_order_release);
            }
        });
    }
    startRetirement.store(true,std::memory_order_release);
    while(enteredRetirement.load(std::memory_order_acquire)!=4) std::this_thread::yield();
    releaseRetirement.store(true,std::memory_order_release);
    // The main thread is the sole external drainer while producers continue
    // releasing final references; completion is monotonic, never transient.
    while(completedRetirement.load(std::memory_order_acquire)!=32) {
        UsdGenGroomSceneIndex::DrainRetired();
        std::this_thread::yield();
    }
    for(auto& producer:retirementProducers) producer.join();
    UsdGenGroomSceneIndex::DrainRetired();
    Check(completedRetirement.load(std::memory_order_acquire)==32,
          "concurrent final-reference producers complete through retirement records");
    return failures?1:0;
}
