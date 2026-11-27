// Session-store owner serialization and snapshot/lifetime contract tests.
#include "utils/pxrInterop.h"
#include "usdGenHydraCurves/publication.h"
#include "usdGenTestUtils/hydraPublishers.h"
#include "usdGenImaging/usdGenImagingSession.h"
#include "usdGenUsd/lower.h"
#include "usdGen/exec/generationStore.h"

#include <atomic>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace usdGenImaging;

namespace {

std::atomic<int> failures{0};

struct ExitCounters {
    std::array<std::atomic<int>, 64> perRequest;
    ExitCounters() { for (auto& count : perRequest) count.store(0); }
    std::atomic<int> accepted{0};
    std::atomic<int> completed{0};
    std::atomic<bool> firstClaimed{false};
    std::atomic<bool> firstEntered{false};
    std::atomic<bool> releaseFirst{false};
};

// This object is initialized before GetInstance() first constructs the store
// singleton.  Its destructor runs on the process's own exit path, while the
// framework workers are still alive, and verifies that returning from main
// with an accepted batch outstanding loses no owner completion: every one is
// still delivered exactly once.  The store itself is process-lifetime and is
// never destroyed (a library static destructor would run on Windows only
// after ExitProcess has killed every worker), so nothing drains on the
// test's behalf; the in-flight tail is given a bounded wait instead.
struct ExitCheck {
    std::shared_ptr<ExitCounters> counters = std::make_shared<ExitCounters>();
    std::atomic<bool> armed{false};
    std::thread releaser;

    ~ExitCheck()
    {
        if (releaser.joinable()) releaser.join();
        if (!armed.load(std::memory_order_acquire)) return;
        auto const deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(10);
        while (counters->completed.load(std::memory_order_acquire) <
                   counters->accepted.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        if (counters->accepted.load(std::memory_order_acquire) !=
            counters->completed.load(std::memory_order_acquire)) {
            std::fprintf(stderr, "pending-exit callback loss: accepted=%d completed=%d\n",
                         counters->accepted.load(), counters->completed.load());
            std::abort();
        }
        for (size_t i = 0; i < counters->perRequest.size(); ++i) {
            int const count = counters->perRequest[i].load();
            if (count != 1) {
                std::fprintf(stderr, "pending-exit request %zu completed %d times\n",
                             i, count);
                std::abort();
            }
        }
    }
};

ExitCheck g_exitCheck;

void Check(bool value, char const *what)
{
    if (!value) {
        failures.fetch_add(1, std::memory_order_relaxed);
        std::printf("FAIL: %s\n", what);
    } else {
        std::printf("ok:   %s\n", what);
    }
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
        Check(false, what);
        // Callback lambdas commonly capture stack state.  Continuing after a
        // timeout would let a late owner callback access dead stack objects.
        std::abort();
    }
}

UsdGenSessionKey Key(int id)
{
    UsdGenSessionKey key;
    key.sessionId = "store-owner-" + std::to_string(id);
    key.groomRoot = SdfPath("/__storeOwner");
    key.renderInstanceId = id;
    return key;
}

usdGen::UsdGenGraphDesc MakeFrameProbe()
{
    usdGen::UsdGenGraphDesc desc;
    desc.description = SdfPath("/__storeOwner");
    desc.terminal = SdfPath("/__storeOwner/grow");
    usdGen::UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/__storeOwner/surface");
    surface.restPoints = usdGen::UsdGenVec3fArray{usdGen::UsdGenVec3f(0, 0, 0), usdGen::UsdGenVec3f(1, 0, 0),
                                    usdGen::UsdGenVec3f(0, 1, 0), usdGen::UsdGenVec3f(1, 1, 0)};
    surface.points = surface.restPoints;
    surface.uv = usdGen::UsdGenVec2fArray{usdGen::UsdGenVec2f(0, 0), usdGen::UsdGenVec2f(1, 0), usdGen::UsdGenVec2f(0, 1), usdGen::UsdGenVec2f(1, 1)};
    surface.faceVertexCounts = VtIntArray{4};
    surface.faceVertexIndices = VtIntArray{0, 1, 3, 2};
    desc.surfaces.push_back(std::move(surface));
    usdGen::UsdGenNodeDesc scatter;
    scatter.path = SdfPath("/__storeOwner/scatter");
    scatter.type = TfToken("UsdGenScatter");
    scatter.surfaces = {SdfPath("/__storeOwner/surface")};
    desc.nodes.push_back(std::move(scatter));
    usdGen::UsdGenNodeDesc grow;
    grow.path = desc.terminal;
    grow.type = TfToken("UsdGenGrow");
    grow.inputs = {SdfPath("/__storeOwner/scatter")};
    grow.params.push_back({TfToken("segments"), VtValue(4), false});
    grow.params.push_back({TfToken("length"), VtValue(1.0), false});
    desc.nodes.push_back(std::move(grow));
    return desc;
}

} // namespace

int main(int argc, char **argv)
{
    // Preserve the last completed contract check if an intermittent native
    // crash terminates this process before stdio would flush its CTest pipe.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    usdGenTest::EnsureHydraPublishers();
    UsdGenSessionStore &store = UsdGenSessionStore::GetInstance();

    if (argc > 1 && std::string(argv[1]) == "--pending-exit") {
        // Deliberately leave the accepted batch outstanding and return.  The
        // process exit path must still deliver every accepted completion;
        // g_exitCheck verifies that after main returns.
        auto session = store.Attach(Key(90));
        (void)session;
        // Initialize the cook path before CRT teardown starts. On Windows,
        // lazy static initialization on a worker can wait for the exit lock
        // held by ExitCheck's destructor. The batch below must still remain
        // outstanding when main returns; only runtime startup is warmed here.
        store.SetTime(0.0);
        constexpr int pendingCount = 64;
        auto counters = g_exitCheck.counters;
        for (int i = 0; i < pendingCount; ++i) {
            bool accepted = store.SetTimeAsync(
                100.0 + i,
                [counters, i] {
                    bool expected = false;
                    if (counters->firstClaimed.compare_exchange_strong(
                            expected, true, std::memory_order_acq_rel)) {
                        counters->firstEntered.store(true, std::memory_order_release);
                        WaitOrAbort([&] { return counters->releaseFirst.load(std::memory_order_acquire); },
                                    "pending-exit releaser opens the completion gate");
                    }
                    counters->perRequest[i].fetch_add(1);
                    counters->completed.fetch_add(1, std::memory_order_release);
                });
            if (accepted)
                counters->accepted.fetch_add(1, std::memory_order_release);
            else {
                std::fprintf(stderr, "pending-exit SetTimeAsync rejected at %d\n", i);
                std::abort();
            }
        }
        WaitOrAbort([&] {
            return counters->firstEntered.load(std::memory_order_acquire);
        }, "pending-exit first completion enters gate");
        Check(counters->completed.load(std::memory_order_acquire) <
                  counters->accepted.load(std::memory_order_acquire),
              "pending-exit retains an incomplete accepted batch");
        g_exitCheck.releaser = std::thread([counters] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            counters->releaseFirst.store(true, std::memory_order_release);
        });
        g_exitCheck.armed.store(true, std::memory_order_release);
        return 0;
    }

    // All concurrent attach commands for one key must return one canonical
    // object, with one attach count per accepted command.
    UsdGenSessionKey same = Key(1);
    constexpr int threadCount = 8;
    constexpr int attachesPerThread = 8;
    constexpr int attachCount = threadCount * attachesPerThread;
    std::vector<UsdGenSessionHandle> handles(attachCount);
    std::atomic<int> attachCallbacks{0};
    std::atomic<int> attachAccepted{0};
    std::vector<std::thread> attachers;
    for (int t = 0; t < threadCount; ++t) {
        attachers.emplace_back([&, t] {
            for (int i = 0; i < attachesPerThread; ++i) {
                int slot = t * attachesPerThread + i;
                if (store.AttachAsync(same, [&, slot](UsdGenSessionHandle handle) {
                        handles[slot] = std::move(handle);
                        ++attachCallbacks;
                    }))
                    ++attachAccepted;
            }
        });
    }
    for (auto &thread : attachers) thread.join();
    Check(attachAccepted == attachCount, "all concurrent attaches accepted");
    WaitOrAbort([&] { return attachCallbacks == attachCount; },
                "all attach callbacks complete");
    bool oneCanonical = true;
    auto canonical = handles.empty() ? UsdGenSessionHandle() : handles.front();
    for (auto const &handle : handles)
        oneCanonical = oneCanonical && handle && canonical && handle == canonical;
    Check(oneCanonical, "same-key attaches return one canonical handle");
    Check(canonical && canonical->AttachedIndices() == attachCount,
          "concurrent attaches are balanced in the canonical count");

    std::atomic<int> detachCallbacks{0};
    for (int i = 0; i < attachCount; ++i) {
        Check(store.DetachAsync(same, [&] { ++detachCallbacks; }),
              "detach command accepted");
    }
    WaitOrAbort([&] { return detachCallbacks == attachCount; },
                "all detach callbacks complete");
    Check(!store.Find(same), "balanced async detaches remove the store entry");
    Check(canonical && canonical->AttachedIndices() == 0,
          "balanced async detaches reach zero exactly");

    // A callback may queue another owner command.  The re-adopted session is
    // created after the detach and is again observable through Find.
    UsdGenSessionKey readopt = Key(2);
    auto adopted = store.Attach(readopt);
    (void)adopted;
    std::atomic<bool> readoptDetached{false};
    std::atomic<bool> readoptAttached{false};
    Check(store.DetachAsync(readopt, [&] {
        readoptDetached = true;
        Check(store.AttachAsync(readopt, [&](UsdGenSessionHandle replacement) {
            readoptAttached = replacement != nullptr;
        }), "callback can queue re-adoption");
    }), "detach/re-adopt command accepted");
    WaitOrAbort([&] { return readoptAttached.load(); },
                "callback re-adopts asynchronously");
    Check(readoptDetached && store.Find(readopt),
          "re-adopted session is visible after callback ordering");
    store.Detach(readopt);

    // A snapshot keeps its strong references after the registry drops its
    // membership, and an individual Find handle has the same property.
    UsdGenSessionKey lifetime = Key(3);
    auto held = store.Attach(lifetime);
    auto snapshot = store.LiveSessions();
    auto found = store.Find(lifetime);
    Check(snapshot.size() == 1 && found == held,
          "Find and LiveSessions return immutable strong snapshots");
    store.Detach(lifetime);
    Check(!store.Find(lifetime) && held && found && !snapshot.empty() &&
              snapshot.front(),
          "snapshot handles outlive store membership");
    Check(held->AttachedIndices() == 0,
          "snapshot lifetime does not retain an attachment count");

    // Synchronous legacy mutation is forbidden from an owner callback: it
    // must reject before changing membership or counts.
    UsdGenSessionKey callbackKey = Key(4);
    std::atomic<bool> attachRejected{false};
    std::atomic<bool> detachRejected{false};
    std::atomic<bool> callbackQueriesSafe{false};
    std::atomic<bool> callbackDone{false};
    Check(store.AttachAsync(callbackKey, [&](UsdGenSessionHandle handle) {
        try {
            (void)store.Attach(callbackKey);
        } catch (std::logic_error const &) {
            attachRejected = true;
        }
        try {
            store.Detach(callbackKey);
        } catch (std::logic_error const &) {
            detachRejected = true;
        }
        auto callbackFind = store.Find(callbackKey);
        auto callbackSnapshot = store.LiveSessions();
        callbackQueriesSafe = callbackFind == handle && !callbackSnapshot.empty();
        callbackDone = handle != nullptr;
    }), "callback-reentry attach command accepted");
    WaitOrAbort([&] { return callbackDone.load(); },
                "callback-reentry test callback completes");
    auto callbackFound = store.Find(callbackKey);
    Check(attachRejected && detachRejected && callbackQueriesSafe && callbackFound &&
              callbackFound->AttachedIndices() == 1,
          "callback queries are safe and synchronous mutations are rejected");
    store.Detach(callbackKey);

    // An old handle cannot detach a replacement session at the same key.
    UsdGenSessionKey identity = Key(5);
    auto oldHandle = store.Attach(identity);
    store.Detach(identity, oldHandle);
    auto replacement = store.Attach(identity);
    Check(replacement && replacement != oldHandle,
          "same-key replacement gets a distinct handle");
    std::atomic<bool> staleDone{false};
    store.DetachAsync(identity, oldHandle, [&] { staleDone = true; });
    WaitOrAbort([&] { return staleDone.load(); },
                "stale expected-handle callback completes");
    Check(store.Find(identity) == replacement,
          "stale expected handle cannot remove replacement");
    Check(replacement->AttachedIndices() == 1,
          "stale expected handle cannot decrement replacement count");
    store.Detach(identity, replacement);

    // Reserve both source actions before filling the store owner. The normal
    // AttachAsync path must reject at capacity, while the pre-admitted attach
    // and its matching detach still complete exactly once.
    UsdGenSessionKey lifecycleKey = Key(51);
    auto attachTicket = store.ReserveLifecycleCommand();
    auto detachTicket = store.ReserveLifecycleCommand();
    std::vector<usdGen::UsdGenExecutionPipeline::CommandTicket> lifecycleFillers;
    for (;;) {
        auto ticket = store.ReserveLifecycleCommand();
        if (!ticket) break;
        lifecycleFillers.emplace_back(std::move(ticket));
    }
    Check(attachTicket && detachTicket && !lifecycleFillers.empty(),
          "store lifecycle tickets reserve before source-owner saturation");
    Check(!store.AttachAsync(lifecycleKey, [](UsdGenSessionHandle) {}),
          "ordinary store attach rejects at full source owner");
    std::atomic<bool> lifecycleAttached{false}, lifecycleDetached{false};
    UsdGenSessionHandle lifecycleHandle;
    Check(store.AttachAsync(std::move(attachTicket), lifecycleKey,
        [&](UsdGenSessionHandle handle) {
            lifecycleHandle = std::move(handle);
            lifecycleAttached.store(true, std::memory_order_release);
        }), "reserved store attach is accepted at full source owner");
    WaitOrAbort([&] { return lifecycleAttached.load(std::memory_order_acquire); },
                "reserved store attach completion runs");
    auto replacementFiller = store.ReserveLifecycleCommand();
    Check(static_cast<bool>(replacementFiller),
          "store source credit released after reserved attach");
    if (replacementFiller) lifecycleFillers.emplace_back(std::move(replacementFiller));
    Check(store.DetachAsync(std::move(detachTicket), lifecycleKey, lifecycleHandle,
        [&] { lifecycleDetached.store(true, std::memory_order_release); }),
          "reserved store detach is accepted at full source owner");
    WaitOrAbort([&] { return lifecycleDetached.load(std::memory_order_acquire); },
                "reserved store detach completion runs");
    Check(!store.Find(lifecycleKey),
          "reserved store detach removes its pre-admitted lifetime");
    lifecycleFillers.clear(); // never carry held credits across a wait boundary
    lifecycleHandle.Reset();

    // Registry-wide state is also owner-serialized.  SetTime marks the
    // registry app-driven; a session attached by its completion callback
    // inherits that state, while synchronous re-entry is rejected.
    Check(store.SetContext(usdGen::UsdGenContext::Render),
          "store context request is admitted");
    UsdGenSessionKey inheritedKey = Key(6);
    std::atomic<bool> setTimeDone{false};
    std::atomic<bool> setTimeRejected{false};
    std::atomic<bool> inheritedDone{false};
    UsdGenSessionHandle inherited;
    Check(store.SetTimeAsync(42.0, [&] {
        try {
            store.SetTime(43.0);
        } catch (std::logic_error const &) {
            setTimeRejected = true;
        }
        setTimeDone = true;
        Check(store.AttachAsync(inheritedKey, [&](UsdGenSessionHandle handle) {
            inherited = std::move(handle);
            inheritedDone = true;
        }), "SetTime callback can queue inherited attach");
    }), "SetTimeAsync accepted");
    WaitOrAbort([&] { return inheritedDone.load(); },
                "SetTime inherited attach completes");
    Check(setTimeDone && setTimeRejected && inherited &&
              inherited->HasAppDriver(),
          "SetTime marks callback-attached session app-driven");
    if (inherited) {
        inherited->StageAndCommit(MakeFrameProbe(), usdGen::UsdGenCommitReason::NoticeBatchEnd);
        auto generation = inherited->LatestGeneration();
        Check(generation && generation->frame == 42.0 && !usdGen::UsdGenCurvePublicationOf(*generation).tiles.empty(),
              "new attachment executes at the inherited store frame");
    }
    store.Detach(inheritedKey);

    // Empty-registry forwarding still completes asynchronously and can be
    // chained by clients without a synchronous owner wait.
    std::atomic<bool> commitDone{false};
    Check(store.CommitAsync(usdGen::UsdGenCommitReason::NoticeBatchEnd,
                            [&] { commitDone = true; }),
          "CommitAsync accepted for an empty registry");
    WaitOrAbort([&] { return commitDone.load(); },
                "empty-registry CommitAsync completion runs");

    // -- cook cancellation token (plan/17 §3.2 rule 2, plan/18 §7 G7) -----
    //
    // An authoring tool whose newer edit has invalidated a cook bumps the
    // token; a request accepted before the bump is dropped as Superseded on
    // the owner, without reaching the engine. A request accepted after it
    // runs normally, so cancelling is not a latch.
    {
        UsdGenSessionKey const cookKey = Key(11);
        auto cooking = store.Attach(cookKey);
        Check(bool(cooking), "cook-token session attaches");
        if (cooking) {
            Check(cooking->CookToken() == 0, "the token starts at 0");
            Check(cooking->CancelCooks() == 1, "CancelCooks returns the new token");
            Check(cooking->CookToken() == 1, "and the session reports it");

            // Stale: stamped under token 0, accepted after the bump above.
            std::atomic<bool> staleDone{false};
            std::atomic<int> staleOutcome{-1};
            UsdGenImagingSession::CommitRequest stale;
            stale.reason = usdGen::UsdGenCommitReason::NoticeBatchEnd;
            stale.desc = std::make_shared<const usdGen::UsdGenGraphDesc>(
                MakeFrameProbe());
            stale.cookToken = 0;
            Check(cooking->CommitAsync(stale,
                      [&](UsdGenImagingSession::CommitPayload const &payload,
                          usdGen::UsdGenExecutionPipeline::Outcome outcome) {
                          staleOutcome = int(outcome);
                          staleDone = !payload.published;
                      }),
                  "a stale-token commit is still accepted");
            WaitOrAbort([&] { return staleDone.load(); },
                        "the stale cook completes without publishing");
            Check(staleOutcome ==
                      int(usdGen::UsdGenExecutionPipeline::Outcome::Superseded),
                  "the stale cook is abandoned as Superseded");

            // Fresh: no stamp, so CommitAsync stamps the current token.
            std::atomic<bool> freshDone{false};
            std::atomic<bool> freshPublished{false};
            UsdGenImagingSession::CommitRequest fresh;
            fresh.reason = usdGen::UsdGenCommitReason::NoticeBatchEnd;
            fresh.desc = std::make_shared<const usdGen::UsdGenGraphDesc>(
                MakeFrameProbe());
            Check(cooking->CommitAsync(fresh,
                      [&](UsdGenImagingSession::CommitPayload const &payload,
                          usdGen::UsdGenExecutionPipeline::Outcome) {
                          freshPublished = payload.published;
                          freshDone = true;
                      }),
                  "a fresh commit is accepted after the cancel");
            WaitOrAbort([&] { return freshDone.load(); },
                        "the fresh cook completes");
            Check(freshPublished.load(),
                  "cancelling does not latch: the next cook publishes");

            // Store-level: only the sessions rooted at the named groom move.
            uint64_t const before = cooking->CookToken();
            Check(store.CancelCooks(SdfPath("/__someOtherGroom")) == 0 &&
                      cooking->CookToken() == before,
                  "the store cancels by groom root, not wholesale");
            Check(store.CancelCooks(SdfPath("/__storeOwner")) >= 1 &&
                      cooking->CookToken() == before + 1,
                  "the store cancels every session under the groom root");
        }
        store.Detach(cookKey);
    }

    std::printf("testUsdGenSessionStoreOwner: %s\n",
                failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}