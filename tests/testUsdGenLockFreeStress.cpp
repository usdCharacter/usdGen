// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenLockFreeStress — multi-thread stress for the lock-free engine
// pieces: shared-pointer cells, the execution cache store and domain
// coalescing, the memory reservation word, and a 64-thread Submit/Visible
// run against one Session.
#include "usdGen/epoch.h"
#include "usdGen/executionCache.h"
#include "usdGen/executionResources.h"
#include "usdGen/opRegistry.h"
#include "usdGen/session.h"
#include "usdGen/sharedCell.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int failures = 0;
void Check(bool ok, char const *what)
{
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

std::atomic<int64_t> gAlive{0};
struct Tracked {
    explicit Tracked(int64_t v) : value(v), mirror(~v) { gAlive.fetch_add(1); }
    ~Tracked() { mirror = value; gAlive.fetch_sub(1); }
    bool Intact() const { return mirror == ~value; }
    int64_t value;
    int64_t mirror;
};

void DrainEpochs()
{
    for (int i = 0; i != 8; ++i) UsdGenEpochDrain();
}

template <class F>
void RunThreads(int n, F &&body)
{
    std::vector<std::thread> threads;
    threads.reserve(n);
    for (int i = 0; i != n; ++i) threads.emplace_back([&body, i] { body(i); });
    for (auto &t : threads) t.join();
}

void SharedCellStress()
{
    {
        UsdGenSharedCell<Tracked const> cell;
        cell.Store(std::make_shared<const Tracked>(0));
        std::atomic<int64_t> next{1};
        std::atomic<int> writersDone{0};
        std::atomic<int> torn{0}, nulls{0};
        int const writers = 4, readers = 28;
        RunThreads(writers + readers, [&](int i) {
            if (i < writers) {
                for (int k = 0; k != 20000; ++k) {
                    int64_t const v = next.fetch_add(1);
                    auto old = cell.Exchange(std::make_shared<const Tracked>(v));
                    if (!old || !old->Intact()) torn.fetch_add(1);
                }
                writersDone.fetch_add(1);
                return;
            }
            while (writersDone.load() != writers) {
                auto v = cell.Load();
                if (!v) { nulls.fetch_add(1); continue; }
                if (!v->Intact()) torn.fetch_add(1);
            }
        });
        Check(torn.load() == 0, "shared cell: no torn or freed value is ever read");
        Check(nulls.load() == 0, "shared cell: a published cell never reads empty");
        int64_t const alive = gAlive.load();
        Check(alive == 1, "shared cell: displaced values die with their last reference");
    }
    DrainEpochs();
    Check(gAlive.load() == 0, "shared cell: destroying the cell releases the value");

    // Prepared publication allocates nothing at the exchange.
    UsdGenSharedCell<Tracked const> cell;
    auto prepared = cell.Prepare(std::make_shared<const Tracked>(7));
    Check(prepared && prepared->value == 7 && !cell.Load(), "shared cell: prepare does not publish");
    auto displaced = cell.Exchange(std::move(prepared));
    Check(!displaced && cell.Load() && cell.Load()->value == 7, "shared cell: prepared value publishes");
}

UsdGenExecutionCacheKey BaseKey()
{
    UsdGenExecutionInputVersions inputs;
    inputs.sources.push_back({SdfPath("/stress/hair"), 1});
    UsdGenExecutionContext context{UsdGenDeviceBackend::CpuReference, 0, -1, 0,
                                   UsdGenContext::Interactive};
    return UsdGenExecutionCacheKey::Make(SdfPath("/stress"), {1, 2}, inputs, context, 0, 1.0);
}

void CacheStoreStress()
{
    {
        UsdGenExecutionCacheStore<Tracked> store(64);
        UsdGenExecutionCacheKey const base = BaseKey();
        std::atomic<int> overBudget{0}, wrong{0}, inserted{0};
        RunThreads(16, [&](int t) {
            std::mt19937 rng(1234u + unsigned(t));
            for (int k = 0; k != 4000; ++k) {
                UsdGenExecutionCacheKey key = base;
                key.layoutDigest = rng() % 256;
                unsigned const op = rng() % 100;
                if (op < 45) {
                    if (store.Insert(key, std::make_shared<const Tracked>(int64_t(key.layoutDigest)),
                                     1 + key.layoutDigest % 3))
                        inserted.fetch_add(1);
                } else if (op < 99) {
                    if (auto lease = store.Lookup(key))
                        if (!(*lease)->Intact() || (*lease)->value != int64_t(key.layoutDigest))
                            wrong.fetch_add(1);
                } else {
                    store.Clear();
                }
                if (store.Bytes() > store.MaxBytes()) overBudget.fetch_add(1);
            }
        });
        Check(overBudget.load() == 0, "cache store: bytes never exceed the budget");
        Check(wrong.load() == 0, "cache store: a lease always reads its own key's payload");
        Check(inserted.load() > 0, "cache store: concurrent inserts are admitted");
    }
    DrainEpochs();
    Check(gAlive.load() == 0, "cache store: every payload is released");
}

void DomainCoalescingStress()
{
    using Domain = UsdGenExecutionCacheDomain;
    auto domain = std::make_shared<Domain>(
        UsdGenExecutionCacheDomainKey{UsdGenDeviceBackend::CpuReference, -1, 0x5354524553u},
        1u << 20);
    int const threads = 16, rounds = 300;
    std::atomic<int> badCalls{0}, missing{0};
    std::atomic<bool> stop{false};
    std::thread invalidator([&] {
        while (!stop.load()) {
            domain->Invalidate();
            std::this_thread::sleep_for(std::chrono::microseconds(300));
        }
    });
    // Each round's counters outlive the test: an Invalidate on another
    // thread may still be delivering Stale to it after the round's threads
    // joined, exactly as before the lock-free rewrite.
    struct Round {
        explicit Round(int n) : calls(size_t(n)) {}
        std::vector<std::atomic<int>> calls;
    };
    std::vector<std::shared_ptr<Round>> keep;
    for (int round = 0; round != rounds; ++round) {
        UsdGenExecutionCacheKey key = BaseKey();
        key.layoutDigest = uint64_t(round);
        key.context.deviceGeneration = domain->Epoch();
        auto state = std::make_shared<Round>(threads);
        keep.push_back(state);
        auto& calls = state->calls;
        std::vector<int> registered(threads, 0);  // 1 leader, 2 follower, 3 withdrawn
        std::atomic<int> joined{0};
        RunThreads(threads, [&](int t) {
            auto reg = domain->BeginCoalesced(key, [&badCalls, state, t](Domain::CoalescedStatus status,
                                                          std::optional<Domain::Lease> lease) {
                if ((status == Domain::CoalescedStatus::Success ||
                     status == Domain::CoalescedStatus::Resident) != bool(lease))
                    badCalls.fetch_add(1);
                state->calls[size_t(t)].fetch_add(1);
            });
            if (reg) {
                registered[size_t(t)] = reg->IsLeader() ? 1 : 2;
                if (!reg->IsLeader() && t % 4 == 0 && domain->CancelCoalesced(*reg))
                    registered[size_t(t)] = 3;
            }
            joined.fetch_add(1);
            if (reg && reg->IsLeader()) {
                while (joined.load() != threads) std::this_thread::yield();
                auto generation = std::make_shared<UsdGenGeneration>();
                generation->id = round;
                domain->ResolveCoalesced(*reg, std::move(generation), 64);
            }
        });
        // A Stale delivery may still be in flight on the invalidating thread.
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (int t = 0; t != threads; ++t)
            while ((registered[size_t(t)] == 1 || registered[size_t(t)] == 2) &&
                   calls[size_t(t)].load() == 0 && std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
        for (int t = 0; t != threads; ++t) {
            int const c = calls[size_t(t)].load();
            if (c > 1) badCalls.fetch_add(1);
            if ((registered[size_t(t)] == 1 || registered[size_t(t)] == 2) && c != 1)
                missing.fetch_add(1);
            if (registered[size_t(t)] == 3 && c != 0) badCalls.fetch_add(1);
        }
    }
    stop.store(true);
    invalidator.join();
    Check(badCalls.load() == 0, "domain: no waiter is called twice, or after withdrawing");
    Check(missing.load() == 0, "domain: every registered waiter gets exactly one terminal call");
    Check(domain->Bytes() <= domain->MaxBytes(), "domain: accounting stays bounded");
}

void ReservationStress()
{
    UsdGenExecutionResourcePool pool(1u << 20);
    auto reservation = pool.TryReserveMemory(4096);
    Check(bool(reservation), "reservation: a precharged balance is admitted");
    std::atomic<int> overdraw{0};
    std::atomic<bool> closed{false};
    RunThreads(64, [&](int t) {
        for (int k = 0; k != 2000; ++k) {
            if (t == 0 && k == 1000) {
                reservation->Release();
                closed.store(true);
            }
            auto permit = reservation->Consume(1 + size_t(k % 7), UsdGenExecutionResourceKind::Scratch);
            if (permit && closed.load() && t == 0) overdraw.fetch_add(1);
            auto direct = pool.TryReserve(1 + size_t(t % 5), UsdGenExecutionResourceKind::Active);
            if (pool.Snapshot().usedBytes > pool.Snapshot().usableBytes) overdraw.fetch_add(1);
        }
    });
    auto const snapshot = pool.Snapshot();
    size_t kinds = 0;
    for (size_t v : snapshot.byKind) kinds += v;
    Check(overdraw.load() == 0, "reservation: never consumes after release or past the budget");
    Check(snapshot.usedBytes == 0 && kinds == 0, "reservation: conservation holds after every release");
}

UsdGenGraphDesc SessionDesc()
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/stress");
    d.terminal = SdfPath("/stress/width");
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/stress/surface");
    surface.id = 0;
    surface.restPoints = VtVec3fArray{GfVec3f(0, 0, 0), GfVec3f(1, 0, 0),
                                      GfVec3f(0, 1, 0), GfVec3f(1, 1, 0)};
    surface.points = surface.restPoints;
    surface.uv = VtVec2fArray{GfVec2f(0, 0), GfVec2f(1, 0), GfVec2f(0, 1), GfVec2f(1, 1)};
    surface.faceVertexCounts = VtIntArray{4};
    surface.faceVertexIndices = VtIntArray{0, 1, 3, 2};
    d.surfaces.push_back(std::move(surface));
    UsdGenNodeDesc scatter;
    scatter.path = SdfPath("/stress/scatter");
    scatter.type = TfToken("UsdGenScatter");
    scatter.surfaces = {SdfPath("/stress/surface")};
    d.nodes.push_back(std::move(scatter));
    UsdGenNodeDesc grow;
    grow.path = SdfPath("/stress/grow");
    grow.type = TfToken("UsdGenGrow");
    grow.inputs = {SdfPath("/stress/scatter")};
    grow.params.push_back({TfToken("segments"), VtValue(4), false});
    grow.params.push_back({TfToken("length"), VtValue(1.0), false});
    d.nodes.push_back(std::move(grow));
    UsdGenNodeDesc width;
    width.path = SdfPath("/stress/width");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {SdfPath("/stress/grow")};
    width.params.push_back({TfToken("width"), VtValue(0.01f), false});
    d.nodes.push_back(std::move(width));
    return d;
}

void SessionSubmitVisibleStress()
{
    UsdGenSession session;
    session.SetGraphDesc(SessionDesc());
    auto first = session.CommitSnapshot(0.0, UsdGenCommitReason::NoticeBatchEnd);
    Check(first && first->generation && !first->diagnostics.HasErrors(),
          "session: the baseline cook publishes");
    int const submitters = 32, readers = 32, perSubmitter = 40;
    std::atomic<int> accepted{0}, completed{0}, doubled{0};
    std::atomic<int> emptied{0}, regressed{0};
    std::atomic<int> submittersDone{0};
    RunThreads(submitters + readers, [&](int t) {
        if (t < submitters) {
            for (int k = 0; k != perSubmitter; ++k) {
                auto once = std::make_shared<std::atomic<int>>(0);
                double const frame = double(1 + (t * perSubmitter + k) % 8);
                if (session.CommitAsync(frame, UsdGenCommitReason::NoticeBatchEnd,
                        [&, once](UsdGenSession::SnapshotPtr, UsdGenExecutionPipeline::Outcome) {
                            if (once->fetch_add(1) != 0) doubled.fetch_add(1);
                            completed.fetch_add(1);
                        }))
                    accepted.fetch_add(1);
            }
            submittersDone.fetch_add(1);
            return;
        }
        int64_t lastId = -1;
        while (submittersDone.load() != submitters) {
            auto snapshot = session.Snapshot();
            if (!snapshot || !snapshot->generation) { emptied.fetch_add(1); continue; }
            int64_t const id = snapshot->generation->id;
            if (id < lastId) regressed.fetch_add(1);
            lastId = id;
            auto generation = session.Generation();
            if (!generation) emptied.fetch_add(1);
        }
    });
    session.Drain();
    Check(accepted.load() > 0, "session: concurrent submits are accepted");
    Check(completed.load() == accepted.load() && doubled.load() == 0,
          "session: every accepted submit completes exactly once");
    Check(emptied.load() == 0, "session: Visible never reads an empty snapshot");
    Check(regressed.load() == 0, "session: a reader never sees the generation go back");
    auto last = session.Snapshot();
    Check(last && last->generation && last->generation->id >= first->generation->id,
          "session: the final snapshot is current");
}

}  // namespace

int main()
{
    usdGenRegisterM1Operators();
    SharedCellStress();
    CacheStoreStress();
    DomainCoalescingStress();
    ReservationStress();
    SessionSubmitVisibleStress();
    DrainEpochs();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::puts("PASS");
    return 0;
}
