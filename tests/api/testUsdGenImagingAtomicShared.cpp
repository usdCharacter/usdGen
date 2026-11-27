// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenImagingAtomicShared — T0: lock-free shared_ptr publication used by
// the Hydra host (usdGen/sharedCell.h).
//
//   0. Single thread: Load/Store/Exchange/CompareExchange; a displaced value
//      is released at once when no reader is pinned.
//   1. A reader pinned on another thread never delays a displaced value: it
//      dies with its last reference; the holder recycles after a drain.
//   2. 64 threads Load/Store/Exchange/CAS on one cell; every value carries a
//      canary, and every value is destroyed exactly once.
//   3. UsdGenCowUpdate from 64 threads loses no update.
//   4. UsdGenLazyLeaked from 64 threads constructs exactly one object (the
//      racers park until it is published) and destroys none; a throwing
//      make reopens the slot for the next caller.

#include "usdGen/api/sharedCell.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace usdGen;

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
}

std::atomic<int64_t> g_live{0};
std::atomic<int64_t> g_made{0};
constexpr uint64_t kCanary = 0x5eedf00dcafebeefull;

struct Value {
    explicit Value(uint64_t v) : value(v) { g_live.fetch_add(1); g_made.fetch_add(1); }
    ~Value() { canary = 0; g_live.fetch_sub(1); }
    uint64_t canary = kCanary;
    uint64_t value;
};

void SingleThread()
{
    UsdGenSharedCell<Value const> cell;
    Check(!cell.Load(), "empty cell loads null");
    cell.Store(std::make_shared<Value const>(1));
    Check(cell.Load() && cell.Load()->value == 1, "store then load");
    std::weak_ptr<Value const> first = cell.Load();
    cell.Store(std::make_shared<Value const>(2));
    Check(first.expired(), "displaced value released at once when unpinned");
    auto old = cell.Exchange(std::make_shared<Value const>(3));
    Check(old && old->value == 2 && cell.Load()->value == 3, "exchange returns the old value");
    auto cur = cell.Load();
    Check(!cell.CompareExchange(old, std::make_shared<Value const>(4)), "CAS fails on a stale expected");
    Check(cell.CompareExchange(cur, std::make_shared<Value const>(5)) && cell.Load()->value == 5,
          "CAS succeeds on the current value");
    Check(cell.CompareExchange(cell.Load(), nullptr) && !cell.Load(), "CAS to null");
    Check(cell.CompareExchange(nullptr, std::make_shared<Value const>(6)) && cell.Load()->value == 6,
          "CAS from null");
}

void PinnedReader()
{
    UsdGenSharedCell<Value const> cell(std::make_shared<Value const>(10));
    std::weak_ptr<Value const> first = cell.Load();
    std::atomic<int> phase{0};
    std::thread reader([&] {
        usdGen::UsdGenEpochGuard guard;
        phase.store(1);
        while (phase.load() != 2) std::this_thread::yield();
    });
    while (phase.load() != 1) std::this_thread::yield();
    cell.Store(std::make_shared<Value const>(11));
    Check(first.expired(), "a pinned reader does not hold the displaced value");
    auto const pinned = usdGen::UsdGenEpochGetStats();
    phase.store(2);
    reader.join();
    for (int i = 0; i != 4; ++i) usdGen::UsdGenEpochDrain();
    auto const drained = usdGen::UsdGenEpochGetStats();
    Check(drained.freed > pinned.freed, "the displaced holder recycles after the reader leaves and a drain");
    Check(cell.Load() && cell.Load()->value == 11, "the new value stays published");
}

void Stress()
{
    usdGen::UsdGenEpochShutdown();  // earlier sections' cells
    int64_t const before = g_live.load();
    {
        UsdGenSharedCell<Value const> cell(std::make_shared<Value const>(0));
        std::atomic<bool> stop{false};
        std::atomic<uint64_t> bad{0}, loads{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < 64; ++t) {
            threads.emplace_back([&, t] {
                uint64_t n = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    switch ((t + n) % 4) {
                    case 0:
                    case 1: {
                        auto v = cell.Load();
                        if (!v || v->canary != kCanary) bad.fetch_add(1);
                        loads.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                    case 2: cell.Store(std::make_shared<Value const>(n)); break;
                    default: {
                        auto cur = cell.Load();
                        if (t & 1) {
                            auto old = cell.Exchange(std::make_shared<Value const>(n));
                            if (!old || old->canary != kCanary) bad.fetch_add(1);
                        } else {
                            cell.CompareExchange(cur, std::make_shared<Value const>(n));
                        }
                    }
                    }
                    ++n;
                }
            });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        stop.store(true);
        for (auto &th : threads) th.join();
        Check(bad.load() == 0, "64 threads: no torn or freed value observed (" +
                                   std::to_string(loads.load()) + " loads)");
    }
    size_t const pending = usdGen::UsdGenEpochShutdown();
    auto const stats = usdGen::UsdGenEpochGetStats();
    std::printf("pending %zu live %lld retired %llu freed %llu\n", pending,
                (long long)(g_live.load() - before), (unsigned long long)stats.retired,
                (unsigned long long)stats.freed);
    Check(g_live.load() == before, "every value destroyed exactly once (" +
                                       std::to_string(g_made.load()) + " made)");
}

void CowUpdate()
{
    UsdGenSharedCell<std::vector<int> const> cell;
    constexpr int kThreads = 64, kEach = 200;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
        threads.emplace_back([&, t] {
            for (int i = 0; i < kEach; ++i)
                UsdGenCowUpdate(cell, [&](std::vector<int> &v) { v.push_back(t); });
        });
    for (auto &th : threads) th.join();
    auto v = cell.Load();
    Check(v && v->size() == size_t(kThreads * kEach), "CoW update from 64 threads loses nothing");
}

struct Lazy {
    Lazy()
    {
        made.fetch_add(1);
        // Widen the race: a slow constructor (a thread pool, say) is where
        // a speculative build would start a second one.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ~Lazy() { freed.fetch_add(1); }
    static std::atomic<int> made, freed;
};
std::atomic<int> Lazy::made{0};
std::atomic<int> Lazy::freed{0};
std::atomic<Lazy *> g_lazy{nullptr};

void LazyLeaked()
{
    std::atomic<int> go{0};
    std::vector<Lazy *> seen(64);
    std::vector<std::thread> threads;
    for (int t = 0; t < 64; ++t)
        threads.emplace_back([&, t] {
            while (!go.load()) std::this_thread::yield();
            seen[t] = &UsdGenLazyLeaked(g_lazy);
        });
    go.store(1);
    for (auto &th : threads) th.join();
    bool same = true;
    for (Lazy *p : seen) same = same && p == seen[0];
    Check(same && seen[0] == g_lazy.load(), "every racer gets the installed object");
    Check(Lazy::made.load() == 1 && Lazy::freed.load() == 0,
          "exactly one object is constructed and none destroyed (made " +
              std::to_string(Lazy::made.load()) + ")");

    std::atomic<Lazy *> slot{nullptr};
    bool threw = false;
    try {
        UsdGenLazyLeaked(slot, []() -> Lazy * { throw std::runtime_error("make failed"); });
    } catch (std::runtime_error const &) {
        threw = true;
    }
    Check(threw && slot.load() == nullptr, "a throwing make propagates and reopens the slot");
    Lazy &second = UsdGenLazyLeaked(slot);
    Check(&second == slot.load() && Lazy::made.load() == 2, "the next caller builds the object");
}

}  // namespace

int main()
{
    SingleThread();
    PinnedReader();
    Stress();
    CowUpdate();
    LazyLeaked();
    std::printf("%s\n", g_failures ? "FAILED" : "PASSED");
    return g_failures ? 1 : 0;
}
