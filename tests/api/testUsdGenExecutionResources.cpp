#include "usdGen/api/ext/device/executionResources.h"
#include "testBackend.h"

#include <atomic>
#include <limits>
#include <cstdio>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>


using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {

using Kind = UsdGenExecutionResourceKind;

size_t KindBytes(UsdGenExecutionResourceSnapshot const& snapshot, Kind kind) {
    return snapshot.byKind[static_cast<size_t>(kind)];
}

bool SameAccounting(UsdGenExecutionResourceSnapshot const& left,
                    UsdGenExecutionResourceSnapshot const& right) {
    return left.usedBytes == right.usedBytes &&
        left.byKind == right.byKind;
}

int TestMemoryReservations()
{
    // A precharged reservation saturates the usable pool atomically, while
    // headroom remains unavailable to both ordinary and memory reservations.
    auto pool = std::make_shared<UsdGenExecutionResourcePool>(100, 20);
    auto reservation = pool->TryReserveMemory(80);
    CHECK(reservation && *reservation && reservation->RemainingBytes() == 80 &&
          reservation->PendingKind() == Kind::Pending);
    CHECK(pool->Snapshot().usedBytes == 80 &&
          KindBytes(pool->Snapshot(), Kind::Pending) == 80);
    CHECK(!pool->TryReserveMemory(1) &&
          !pool->TryReserve(1, Kind::Active));

    // Partial and full Consume operations transfer category ownership only;
    // they never change total used bytes. Invalid and over-sized requests are
    // no-ops, including their category counters and remaining balance.
    auto scratch = reservation->Consume(25, Kind::Scratch);
    CHECK(scratch && scratch->Bytes() == 25 && scratch->Kind() == Kind::Scratch &&
          reservation->RemainingBytes() == 55 && pool->Snapshot().usedBytes == 80);
    CHECK(KindBytes(pool->Snapshot(), Kind::Pending) == 55 &&
          KindBytes(pool->Snapshot(), Kind::Scratch) == 25);
    auto beforeInvalid = pool->Snapshot();
    CHECK(!reservation->Consume(56, Kind::Pinned));
    CHECK(!reservation->Consume(1, Kind::Pending));
    CHECK(!reservation->Consume(1, static_cast<Kind>(255)));
    CHECK(!pool->TryReserve(1, Kind::Pending) &&
          !pool->TryReserveMemory(std::numeric_limits<size_t>::max()));
    CHECK(reservation->RemainingBytes() == 55 &&
          SameAccounting(beforeInvalid, pool->Snapshot()));
    // Releasing a child while its reservation is live recycles the child
    // charge into Pending without changing total usage. The restored credit
    // can fund a later phase, so cumulative allocations may exceed the peak.
    scratch->Release();
    CHECK(!*scratch && reservation->RemainingBytes() == 80 &&
          pool->Snapshot().usedBytes == 80 &&
          KindBytes(pool->Snapshot(), Kind::Pending) == 80 &&
          KindBytes(pool->Snapshot(), Kind::Scratch) == 0);
    auto laterScratch = reservation->Consume(30, Kind::Scratch);
    CHECK(laterScratch && reservation->RemainingBytes() == 50 &&
          pool->Snapshot().usedBytes == 80);
    laterScratch->Release();
    CHECK(!*laterScratch && reservation->RemainingBytes() == 80 &&
          pool->Snapshot().usedBytes == 80);
    auto active = reservation->Consume(80, Kind::Active);
    CHECK(active && active->Bytes() == 80 && active->Kind() == Kind::Active &&
          reservation->RemainingBytes() == 0 && pool->Snapshot().usedBytes == 80);
    CHECK(KindBytes(pool->Snapshot(), Kind::Pending) == 0 &&
          KindBytes(pool->Snapshot(), Kind::Active) == 80);
    active->Release();
    CHECK(!*active && reservation->RemainingBytes() == 80 &&
          KindBytes(pool->Snapshot(), Kind::Pending) == 80 &&
          pool->Snapshot().usedBytes == 80);
    reservation->Release();
    reservation->Release();
    CHECK(!*reservation && reservation->RemainingBytes() == 0 &&
          pool->Snapshot().usedBytes == 0);

    // Zero is a real, inert reservation amount. Estimate availability—not a
    // sentinel byte value—decides whether a graph is eligible for admission.
    auto zeroReservation = pool->TryReserveMemory(0);
    CHECK(zeroReservation && *zeroReservation &&
          zeroReservation->RemainingBytes() == 0 &&
          pool->Snapshot().usedBytes == 0);
    auto zeroPermit = zeroReservation->Consume(0, Kind::Scratch);
    CHECK(zeroPermit && zeroPermit->Bytes() == 0);
    zeroPermit->Release();
    zeroReservation->Release();

    // Destroying a reservation releases only its unconsumed remainder. A
    // transferred permit remains charged and independently releasable after
    // the reservation owner has gone away.
    std::optional<UsdGenExecutionResourcePermit> retainedPermit;
    {
        auto scoped = pool->TryReserveMemory(50);
        CHECK(scoped);
        retainedPermit = scoped->Consume(20, Kind::Pinned);
        CHECK(retainedPermit && scoped->RemainingBytes() == 30 &&
              pool->Snapshot().usedBytes == 50);
    }
    CHECK(retainedPermit && pool->Snapshot().usedBytes == 20 &&
          KindBytes(pool->Snapshot(), Kind::Pinned) == 20 &&
          KindBytes(pool->Snapshot(), Kind::Pending) == 0 &&
          !pool->TryReserve(81, Kind::Active));
    retainedPermit->Release();
    CHECK(pool->Snapshot().usedBytes == 0);

    // Move construction and assignment transfer the reservation balance;
    // moved-from handles become inert and assignment releases the old charge.
    auto movedFrom = pool->TryReserveMemory(10);
    auto movedTo = pool->TryReserveMemory(20);
    CHECK(movedFrom && movedTo && pool->Snapshot().usedBytes == 30);
    auto moved = std::move(*movedFrom);
    CHECK(!*movedFrom && moved && moved.RemainingBytes() == 10 &&
          pool->Snapshot().usedBytes == 30);
    moved = std::move(*movedTo);
    CHECK(!*movedTo && moved && moved.RemainingBytes() == 20 &&
          moved.PendingKind() == Kind::Pending && pool->Snapshot().usedBytes == 20);
    moved.Release();
    CHECK(pool->Snapshot().usedBytes == 0);
    auto emptyReservation = pool->TryReserveMemory(1);
    CHECK(emptyReservation && emptyReservation->RemainingBytes() == 1);
    emptyReservation->Release();
    CHECK(pool->Snapshot().usedBytes == 0 && moved.RemainingBytes() == 0);

    // Competing precharges saturate exactly at usable capacity; no admission
    // can overdraw the pool even when all callers race on the same counter.
    auto saturated = std::make_shared<UsdGenExecutionResourcePool>(257);
    std::vector<UsdGenExecutionMemoryReservation> heldReservations;
    heldReservations.reserve(16);
    std::mutex reservationMutex;
    std::atomic<int> admittedReservations{0};
    std::vector<std::thread> reservers;
    for (int i = 0; i != 16; ++i) reservers.emplace_back([&] {
        auto candidate = saturated->TryReserveMemory(17);
        if (!candidate) return;
        std::lock_guard<std::mutex> lock(reservationMutex);
        heldReservations.push_back(std::move(*candidate));
        admittedReservations.fetch_add(1, std::memory_order_relaxed);
    });
    for (auto& reserver : reservers) reserver.join();
    CHECK(admittedReservations == 15 && saturated->Snapshot().usedBytes == 255 &&
          KindBytes(saturated->Snapshot(), Kind::Pending) == 255 &&
          !saturated->TryReserveMemory(3));
    heldReservations.clear();
    CHECK(saturated->Snapshot().usedBytes == 0);

    // Concurrent Consume calls share one balance, so successful permits can
    // never account for more than the original reservation. Holding every
    // permit until the workers finish makes the unchanged total explicit.
    auto concurrent = std::make_shared<UsdGenExecutionResourcePool>(100);
    auto concurrentReservation = concurrent->TryReserveMemory(100);
    CHECK(concurrentReservation);
    std::vector<UsdGenExecutionResourcePermit> permits;
    permits.reserve(128);
    std::mutex permitsMutex;
    std::atomic<size_t> consumed{0};
    std::vector<std::thread> consumers;
    for (int worker = 0; worker != 8; ++worker) consumers.emplace_back([&] {
        for (int i = 0; i != 40; ++i) {
            auto permit = concurrentReservation->Consume(3, Kind::Active);
            if (!permit) continue;
            consumed.fetch_add(3, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(permitsMutex);
            permits.push_back(std::move(*permit));
        }
    });
    for (auto& consumer : consumers) consumer.join();
    CHECK(consumed.load() <= 100 && concurrentReservation->RemainingBytes() +
              consumed.load() == 100 && concurrent->Snapshot().usedBytes == 100);
    permits.clear();
    CHECK(concurrent->Snapshot().usedBytes == concurrentReservation->RemainingBytes());
    concurrentReservation->Release();
    CHECK(concurrent->Snapshot().usedBytes == 0);

    // Closing a reservation and releasing one of its children concurrently
    // must linearize to exactly one final charge removal, regardless of which
    // side wins the reservation mutex.
    auto raced = std::make_shared<UsdGenExecutionResourcePool>(64);
    auto racedReservation = raced->TryReserveMemory(64);
    CHECK(racedReservation);
    auto racedChild = racedReservation->Consume(32, Kind::Active);
    CHECK(racedChild && racedReservation->RemainingBytes() == 32 &&
          raced->Snapshot().usedBytes == 64);
    std::atomic<int> raceReady{0};
    std::atomic<bool> raceGo{false};
    std::thread closeReservation([&] {
        raceReady.fetch_add(1, std::memory_order_release);
        while (!raceGo.load(std::memory_order_acquire)) std::this_thread::yield();
        racedReservation->Release();
    });
    std::thread releaseChild([&] {
        raceReady.fetch_add(1, std::memory_order_release);
        while (!raceGo.load(std::memory_order_acquire)) std::this_thread::yield();
        racedChild->Release();
    });
    while (raceReady.load(std::memory_order_acquire) != 2)
        std::this_thread::yield();
    raceGo.store(true, std::memory_order_release);
    closeReservation.join();
    releaseChild.join();
    CHECK(!*racedReservation && !*racedChild &&
          racedReservation->RemainingBytes() == 0 &&
          raced->Snapshot().usedBytes == 0 &&
          KindBytes(raced->Snapshot(), Kind::Pending) == 0 &&
          KindBytes(raced->Snapshot(), Kind::Active) == 0);

    // Reservations are isolated: one job cannot consume another job's
    // precharged balance, nor can a later reservation steal its remainder.
    auto isolated = std::make_shared<UsdGenExecutionResourcePool>(100);
    auto first = isolated->TryReserveMemory(60);
    auto second = isolated->TryReserveMemory(20);
    CHECK(first && second && first->RemainingBytes() == 60 &&
          second->RemainingBytes() == 20 && !isolated->TryReserveMemory(21));
    auto secondPermit = second->Consume(20, Kind::Scratch);
    CHECK(secondPermit && second->RemainingBytes() == 0 &&
          first->RemainingBytes() == 60 && isolated->Snapshot().usedBytes == 80);
    secondPermit->Release();
    CHECK(isolated->Snapshot().usedBytes == 80 && first->RemainingBytes() == 60 &&
          second->RemainingBytes() == 20 &&
          KindBytes(isolated->Snapshot(), Kind::Pending) == 80);
    // The released second-job credit is reusable by that job, but remains
    // isolated from the first reservation's balance.
    auto secondAgain = second->Consume(20, Kind::Active);
    CHECK(secondAgain && second->RemainingBytes() == 0 &&
          first->RemainingBytes() == 60 && isolated->Snapshot().usedBytes == 80);
    secondAgain->Release();
    CHECK(second->RemainingBytes() == 20 && isolated->Snapshot().usedBytes == 80);
    auto firstPermit = first->Consume(60, Kind::Active);
    CHECK(firstPermit && isolated->Snapshot().usedBytes == 80);
    firstPermit->Release();
    CHECK(first->RemainingBytes() == 60 && isolated->Snapshot().usedBytes == 80);
    first->Release();
    second->Release();
    CHECK(isolated->Snapshot().usedBytes == 0);

    // Abandon preserves a transferred charge permanently, while the
    // reservation destructor still returns its unused pending remainder.
    auto abandonedPool = std::make_shared<UsdGenExecutionResourcePool>(50);
    std::optional<UsdGenExecutionResourcePermit> abandonedPermit;
    {
        auto pending = abandonedPool->TryReserveMemory(30);
        CHECK(pending);
        abandonedPermit = pending->Consume(12, Kind::Cache);
        CHECK(abandonedPermit && pending->RemainingBytes() == 18);
        abandonedPermit->Abandon();
        CHECK(!*abandonedPermit && pending->RemainingBytes() == 18 &&
              abandonedPool->Snapshot().usedBytes == 30 &&
              KindBytes(abandonedPool->Snapshot(), Kind::Pending) == 18 &&
              KindBytes(abandonedPool->Snapshot(), Kind::Cache) == 12);
    }
    CHECK(abandonedPool->Snapshot().usedBytes == 12 &&
          KindBytes(abandonedPool->Snapshot(), Kind::Cache) == 12 &&
          !abandonedPool->TryReserve(39, Kind::Active));
    auto remaining = abandonedPool->TryReserve(38, Kind::Scratch);
    CHECK(remaining);
    remaining->Release();
    CHECK(abandonedPool->Snapshot().usedBytes == 12);
    return 0;
}

} // namespace

int main() {
    if (TestMemoryReservations() != 0) return 1;
    using Kind = UsdGenExecutionResourceKind;
    auto pool = std::make_shared<UsdGenExecutionResourcePool>(100, 20);
    CHECK(pool->Snapshot().usableBytes == 80);
    CHECK(!pool->TryReserve(81, Kind::Active));
    auto zero = pool->TryReserve(0, Kind::Scratch);
    CHECK(zero && zero->Bytes() == 0 && pool->Snapshot().usedBytes == 0);
    CHECK(!pool->TryReserve(std::numeric_limits<size_t>::max(), Kind::Active));
    auto active = pool->TryReserve(60, Kind::Active);
    CHECK(active && active->Bytes() == 60);
    CHECK(!pool->TryReserve(21, Kind::Scratch));
    active->Reclassify(Kind::Pinned);
    active->Reclassify(static_cast<Kind>(255));
    auto snapshot = pool->Snapshot();
    CHECK(snapshot.usedBytes == 60 && snapshot.byKind[1] == 60 && snapshot.byKind[0] == 0);
    auto retained = std::move(*active);
    CHECK(!*active && retained);
    retained.Release();
    CHECK(pool->Snapshot().usedBytes == 0);

    // A retained allocation consumes budget until its actual lifetime ends.
    auto cache = pool->TryReserve(50, Kind::Cache);
    CHECK(cache && !pool->TryReserve(31, Kind::Active));
    cache.reset();
    CHECK(pool->TryReserve(80, Kind::Scratch));

    // Concurrent admissions never exceed usable capacity. Each worker holds
    // one permit at a time; this bounded test intentionally needs no barrier.
    auto concurrent = std::make_shared<UsdGenExecutionResourcePool>(4096, 96);
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (int t = 0; t != 8; ++t) workers.emplace_back([&, t] {
        for (int i = 0; i != 3000; ++i) {
            size_t bytes = 1 + static_cast<size_t>((i * 17 + t * 13) % 251);
            auto permit = concurrent->TryReserve(bytes, (i & 1) ? Kind::Active : Kind::Scratch);
            if (permit) {
                auto state = concurrent->Snapshot();
                if (state.usedBytes > state.usableBytes) failed.store(true);
                if ((i % 7) == 0) permit->Reclassify(Kind::Pinned);
            }
        }
    });
    for (auto& worker : workers) worker.join();
    auto final = concurrent->Snapshot();
    CHECK(!failed && final.usedBytes == 0);
    for (auto bytes : final.byKind) CHECK(bytes == 0);

    // Held permits force contention at a known capacity. This bounded test
    // uses a rendezvous solely to observe admission, not a production lock.
    auto exhausted = std::make_shared<UsdGenExecutionResourcePool>(192);
    std::atomic<int> attempted{0}, admitted{0};
    std::atomic<bool> release{false};
    workers.clear();
    for (int i = 0; i != 8; ++i) workers.emplace_back([&] {
        auto permit = exhausted->TryReserve(96, Kind::Active);
        if (permit) ++admitted;
        ++attempted;
        while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
    });
    for (int spin = 0; spin != 100000 && attempted.load() != 8; ++spin)
        std::this_thread::yield();
    release.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    CHECK(attempted == 8 && admitted == 2 && exhausted->Snapshot().usedBytes == 0);

    UsdGenExecutionResourceDevice device{usdGenTestUtils::kTestBackendA, 92741};
    auto registered = GetOrCreateUsdGenExecutionResourcePool(device, {91, 11});
    CHECK(registered && FindUsdGenExecutionResourcePool(device) == registered);
    CHECK(GetOrCreateUsdGenExecutionResourcePool(device, {91, 11}) == registered);
    CHECK(!GetOrCreateUsdGenExecutionResourcePool(device, {92, 11}));
    auto abandonedPool = std::make_shared<UsdGenExecutionResourcePool>(16);
    auto abandoned = abandonedPool->TryReserve(7, Kind::Cache);
    CHECK(abandoned);
    abandoned->Abandon();
    auto abandonedSnapshot = abandonedPool->Snapshot();
    CHECK(abandonedSnapshot.usedBytes == 7 && abandonedSnapshot.byKind[2] == 7);
    CHECK(!abandonedPool->TryReserve(10, Kind::Active));
    return 0;
}
