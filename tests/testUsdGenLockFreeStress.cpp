// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenLockFreeStress — multi-thread stress for the lock-free engine
// pieces: shared-pointer cells, the execution cache store (leases against
// eviction) and domain coalescing (Invalidate against settling leaders and
// publications), the pipeline's newest-wins mailbox, the memory reservation
// word, and a 64-thread Submit/Visible run against one Session. Tracked
// values live in a never-reused poisoned arena, so reading a freed one fails.
#include "utils/pxrInterop.h"
#include "usdGen/api/epoch.h"
#include "usdGenUsd/lower.h"
#include "usdGen/exec/executionCache.h"
#include "usdGen/api/pipeline.h"
#include "usdGen/api/ext/device/executionResources.h"
#include "usdGen/api/ext/op.h"
#include "usdGen/api/ext/opRegistry.h"
#include "usdGen/api/session.h"
#include "usdGen/api/sharedCell.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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

// A bump arena that never hands memory out twice and poisons what is freed,
// so a value (or its control block) read after its last release is never a
// fresh object at the same address: Intact() fails on the poison.
struct PoisonArena {
    static constexpr size_t kBytes = size_t(128) << 20;
    unsigned char *base = static_cast<unsigned char *>(std::malloc(kBytes));
    std::atomic<size_t> used{0};
    void *Allocate(size_t n)
    {
        n = (n + 63) & ~size_t(63);
        size_t const at = used.fetch_add(n);
        if (!base || at + n > kBytes) {
            std::puts("FAIL: poison arena exhausted");
            std::abort();
        }
        return base + at;
    }
    static void Free(void *p, size_t n) { std::memset(p, 0xdd, n); }
};
PoisonArena gArena;

template <class T>
struct PoisonAllocator {
    using value_type = T;
    PoisonAllocator() = default;
    template <class U>
    PoisonAllocator(PoisonAllocator<U> const &) noexcept {}
    T *allocate(size_t n) { return static_cast<T *>(gArena.Allocate(n * sizeof(T))); }
    void deallocate(T *p, size_t n) noexcept { PoisonArena::Free(p, n * sizeof(T)); }
    template <class U>
    bool operator==(PoisonAllocator<U> const &) const noexcept { return true; }
    template <class U>
    bool operator!=(PoisonAllocator<U> const &) const noexcept { return false; }
};

std::shared_ptr<const Tracked> MakeTracked(int64_t v)
{
    return std::allocate_shared<Tracked>(PoisonAllocator<Tracked>(), v);
}

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
        cell.Store(MakeTracked(0));
        std::atomic<int64_t> next{1};
        std::atomic<int> writersDone{0};
        std::atomic<int> torn{0}, nulls{0};
        int const writers = 4, readers = 28;
        RunThreads(writers + readers, [&](int i) {
            if (i < writers) {
                for (int k = 0; k != 20000; ++k) {
                    int64_t const v = next.fetch_add(1);
                    auto old = cell.Exchange(MakeTracked(v));
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
    auto prepared = cell.Prepare(MakeTracked(7));
    Check(prepared && prepared->value == 7 && !cell.Load(), "shared cell: prepare does not publish");
    auto displaced = cell.Exchange(std::move(prepared));
    Check(!displaced && cell.Load() && cell.Load()->value == 7, "shared cell: prepared value publishes");
}

UsdGenExecutionCacheKey BaseKey()
{
    UsdGenExecutionInputVersions inputs;
    inputs.sources.push_back({usdGenUsd::KeyOf(SdfPath("/stress/hair")), 1});
    UsdGenExecutionContext context{kUsdGenBackendCpu, 0, -1, 0,
                                   UsdGenContext::Interactive};
    return UsdGenExecutionCacheKey::Make(usdGenUsd::KeyOf(SdfPath("/stress")), {1, 2}, inputs, context, 0, 1.0);
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
                    if (store.Insert(key, MakeTracked(int64_t(key.layoutDigest)),
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
        UsdGenExecutionCacheDomainKey{kUsdGenBackendCpu, -1, 0x5354524553u},
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


// A held lease keeps its entry resident and charged: an inserter that needs
// room never evicts it, so a second lookup under the lease always hits.
void LeasedEvictionStress()
{
    {
        UsdGenExecutionCacheStore<Tracked> store(8);
        UsdGenExecutionCacheKey const base = BaseKey();
        std::atomic<int> evictedUnderLease{0}, wrong{0}, leased{0};
        std::atomic<int> inserters{0};
        int const nInserters = 6, nLeasers = 10;
        RunThreads(nInserters + nLeasers, [&](int t) {
            std::mt19937 rng(777u + unsigned(t));
            if (t < nInserters) {
                for (int k = 0; k != 30000; ++k) {
                    UsdGenExecutionCacheKey key = base;
                    key.layoutDigest = rng() % 32;
                    store.Insert(key, MakeTracked(int64_t(key.layoutDigest)), 1);
                }
                inserters.fetch_add(1);
                return;
            }
            while (inserters.load() != nInserters) {
                UsdGenExecutionCacheKey key = base;
                key.layoutDigest = rng() % 32;
                auto lease = store.Lookup(key);
                if (!lease) continue;
                leased.fetch_add(1);
                for (int y = 0; y != 4; ++y) std::this_thread::yield();
                if (!(*lease)->Intact() || (*lease)->value != int64_t(key.layoutDigest))
                    wrong.fetch_add(1);
                if (!store.Lookup(key)) evictedUnderLease.fetch_add(1);
            }
        });
        Check(leased.load() > 0, "leased eviction: lookups lease entries under insert pressure");
        Check(evictedUnderLease.load() == 0, "leased eviction: a leased entry is never evicted");
        Check(wrong.load() == 0, "leased eviction: a lease reads its own intact payload");
        Check(store.Bytes() <= store.MaxBytes(), "leased eviction: accounting stays bounded");
    }
    DrainEpochs();
    Check(gAlive.load() == 0, "leased eviction: every payload is released");
}

using Domain = UsdGenExecutionCacheDomain;

std::shared_ptr<Domain> MakeDomain(uint64_t identity)
{
    return std::make_shared<Domain>(
        UsdGenExecutionCacheDomainKey{kUsdGenBackendCpu, -1, identity}, 1u << 20);
}

UsdGenExecutionCacheKey DomainKey(Domain const &domain, uint64_t layout)
{
    UsdGenExecutionCacheKey key = BaseKey();
    key.layoutDigest = layout;
    key.context.deviceGeneration = domain.Epoch();
    return key;
}

uint64_t EpochNow() { return UsdGenEpochGetStats().epoch; }

// Invalidate walks flights it took from the list while a leader that won a
// flight's settled CAS before the take retires it. The walk must stay pinned:
// with the leader's retire done and drained, the global epoch may advance at
// most once (to free the flight it must advance twice).
void InvalidateWalkPinned()
{
    auto domain = MakeDomain(0x57414c4bu);
    struct Ctx {
        std::atomic<int> phase{0};
        uint64_t advanced = 0;
    } ctx;
    auto reg = domain->BeginCoalesced(DomainKey(*domain, 1),
                                      [](Domain::CoalescedStatus, std::optional<Domain::Lease>) {});
    Check(reg && reg->IsLeader(), "invalidate walk: a leader registers");
    auto wait = [](std::atomic<int> &phase, int value) {
        while (phase.load() < value) std::this_thread::yield();
    };
    Domain::SetTestHook([](Domain::TestPoint point, void *p) {
        auto &c = *static_cast<Ctx *>(p);
        auto wait = [&c](int value) { while (c.phase.load() < value) std::this_thread::yield(); };
        if (point == Domain::TestPoint::SettleWonFlight && c.phase.load() == 0) {
            c.phase.store(1);
            wait(2);
        } else if (point == Domain::TestPoint::InvalidateTookFlights && c.phase.load() == 1) {
            c.phase.store(2);
            wait(3);
        }
    }, &ctx);
    bool rejected = false;
    std::thread leader([&] {
        rejected = domain->RejectCoalesced(*reg, Domain::CoalescedStatus::Failed);
        uint64_t const before = EpochNow();
        DrainEpochs();
        ctx.advanced = EpochNow() - before;
        ctx.phase.store(3);
    });
    wait(ctx.phase, 1);
    std::thread invalidator([&] { domain->Invalidate(); });
    leader.join();
    invalidator.join();
    Domain::SetTestHook(nullptr, nullptr);
    Check(ctx.phase.load() == 3 && !rejected,
          "invalidate walk: the leader settled its flight before Invalidate walked it (Stale)");
    Check(ctx.advanced <= 1, "invalidate walk: a flight Invalidate still walks cannot be reclaimed");
}

// BeginCoalesced withdraws a flight an Invalidate made stale before it was
// published. The leader's callback dies there and must die unpinned.
void StaleWithdrawUnpinned()
{
    auto domain = MakeDomain(0x5354414cu);
    struct Ctx {
        Domain *domain = nullptr;
        std::atomic<int> fired{0};
    } ctx{domain.get()};
    struct Probe {
        std::shared_ptr<std::atomic<int>> unpinned = std::make_shared<std::atomic<int>>(-1);
        ~Probe()
        {
            if (!unpinned) return;
            // Drain does nothing on a pinned thread; unpinned, nothing else
            // in this process holds a pin, so the epoch moves.
            uint64_t const before = EpochNow();
            UsdGenEpochDrain();
            unpinned->store(EpochNow() > before ? 1 : 0);
        }
        Probe() = default;
        Probe(Probe &&o) noexcept : unpinned(std::move(o.unpinned)) {}
    };
    Probe probe;
    auto result = probe.unpinned;
    UsdGenExecutionCacheKey const key = DomainKey(*domain, 2);
    Domain::SetTestHook([](Domain::TestPoint point, void *p) {
        auto &c = *static_cast<Ctx *>(p);
        if (point == Domain::TestPoint::BeginCoalescedBeforePublish && !c.fired.exchange(1))
            c.domain->Invalidate();
    }, &ctx);
    auto holder = std::make_shared<Probe>(std::move(probe));
    bool called = false;
    auto reg = domain->BeginCoalesced(key, [holder = std::move(holder), &called](
                                               Domain::CoalescedStatus, std::optional<Domain::Lease>) {
        called = true;
    });
    Domain::SetTestHook(nullptr, nullptr);
    Check(ctx.fired.load() == 1 && !reg && !called,
          "stale withdraw: a flight made stale before publication is withdrawn silently");
    Check(result->load() == 1, "stale withdraw: the leader's callback is destroyed outside a guard");
}

// No publication that passed its epoch check may land after Invalidate has
// returned: Invalidate waits for the action that is already running.
void PublishVersusInvalidate()
{
    auto domain = MakeDomain(0x50554255u);
    int violations = 0, published = 0;
    for (int round = 0; round != 8; ++round) {
        auto fence = domain->CapturePublicationFence(DomainKey(*domain, 3));
        if (!fence) { ++violations; continue; }
        uint64_t const epoch = fence->Epoch();
        std::atomic<int> entered{0}, returned{0};
        std::thread publisher([&] {
            bool const ok = domain->PublishIfCurrent(*fence, [&] {
                entered.store(1);
                auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                while (domain->Epoch() == epoch && std::chrono::steady_clock::now() < deadline)
                    std::this_thread::yield();
                auto const linger = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
                while (std::chrono::steady_clock::now() < linger) std::this_thread::yield();
                if (returned.load()) ++violations;
                ++published;
                return std::shared_ptr<const void>();
            });
            if (!ok) ++violations;
        });
        while (!entered.load()) std::this_thread::yield();
        domain->Invalidate();
        returned.store(1);
        publisher.join();
    }
    Check(published == 8, "publish fence: an action that passed its check completes");
    Check(violations == 0, "publish fence: no publication lands after Invalidate returns");

    // Many publishers against a repeating Invalidate: an action with an old
    // fence never runs after the Invalidate that retired its epoch returned.
    std::atomic<uint64_t> invalidatedThrough{0};  // epochs < this are retired
    std::atomic<int> late{0}, ran{0};
    std::atomic<bool> stop{false};
    std::thread invalidator([&] {
        for (int i = 0; i != 400; ++i) {
            uint64_t const before = domain->Epoch();
            domain->Invalidate();
            invalidatedThrough.store(before + 1);
            std::this_thread::yield();
        }
        stop.store(true);
    });
    RunThreads(8, [&](int) {
        while (!stop.load()) {
            auto fence = domain->CapturePublicationFence(DomainKey(*domain, 4));
            if (!fence) continue;
            uint64_t const epoch = fence->Epoch();
            domain->PublishIfCurrent(*fence, [&] {
                for (int y = 0; y != 3; ++y) std::this_thread::yield();
                if (invalidatedThrough.load() > epoch) late.fetch_add(1);
                ran.fetch_add(1);
                return std::shared_ptr<const void>();
            });
        }
    });
    invalidator.join();
    Check(ran.load() > 0, "publish fence: concurrent publications run");
    Check(late.load() == 0, "publish fence: no old-epoch action outlives its Invalidate");
}

// PostLatestCommand against the owner's recheck: the newest command always
// runs (store-load ordering on scheduled/latest on both sides).
void MailboxNewestRuns()
{
    UsdGenExecutionRuntime runtime(4);
    UsdGenExecutionPipeline pipeline(runtime, 4, 64);
    auto mailbox = pipeline.ReserveCommandMailbox();
    std::atomic<int> last{-1};
    int stranded = 0;
    int const rounds = 20000;
    for (int round = 0; round != rounds; ++round) {
        pipeline.PostLatestCommand(mailbox, [&last, round] { last.store(2 * round, std::memory_order_relaxed); });
        for (int y = 0; y != round % 16; ++y) std::this_thread::yield();
        pipeline.PostLatestCommand(mailbox, [&last, round] { last.store(2 * round + 1, std::memory_order_relaxed); });
        pipeline.Drain();
        if (last.load() != 2 * round + 1) ++stranded;
    }
    Check(stranded == 0, "mailbox: the newest command always runs");
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
            // Read before Consume: a Release that already happened must win.
            bool const released = closed.load();
            auto permit = reservation->Consume(1 + size_t(k % 7), UsdGenExecutionResourceKind::Scratch);
            if (permit && released) overdraw.fetch_add(1);
            auto direct = pool.TryReserve(1 + size_t(t % 5), UsdGenExecutionResourceKind::Active);
            auto const snapshot = pool.Snapshot();
            if (snapshot.usedBytes > snapshot.usableBytes) overdraw.fetch_add(1);
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
    surface.restPoints = usdGen::UsdGenVec3fArray{usdGen::UsdGenVec3f(0, 0, 0), usdGen::UsdGenVec3f(1, 0, 0),
                                      usdGen::UsdGenVec3f(0, 1, 0), usdGen::UsdGenVec3f(1, 1, 0)};
    surface.points = surface.restPoints;
    surface.uv = usdGen::UsdGenVec2fArray{usdGen::UsdGenVec2f(0, 0), usdGen::UsdGenVec2f(1, 0), usdGen::UsdGenVec2f(0, 1), usdGen::UsdGenVec2f(1, 1)};
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
    usdGenUsd::SetGraphDesc(session, SessionDesc());
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
    SharedCellStress();
    CacheStoreStress();
    LeasedEvictionStress();
    InvalidateWalkPinned();
    StaleWithdrawUnpinned();
    PublishVersusInvalidate();
    DomainCoalescingStress();
    MailboxNewestRuns();
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
