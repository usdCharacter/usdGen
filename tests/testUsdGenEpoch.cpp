// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenEpoch — T0: epoch reclamation and atomic publication (epoch.h).
//
//   0. Single thread: refcounts, Load/Store/Exchange/CompareExchange, guard
//      nesting, a pinned thread's Drain does nothing.
//   1. Retire/drain accounting: a pin on another thread holds reclamation;
//      after it leaves, retired == freed. Counted re-retire of one node.
//   2. 64 threads Load/Store/Exchange/CAS on shared cells. Objects come from
//      a pool that is never reused, so a use after free reads the poisoned
//      canary. Every object is destroyed exactly once.
//   3. Treiber stack, continuation stack, newest-wins slot under threads.
//   4. Slot exhaustion: more threads than slots still pin correctly through
//      the overflow counters; slots come back at thread exit.
//   5. Use from a thread_local destroyed after the epoch's thread state
//      claims no slot and is still reclaimed.

#include "usdGen/exec/epoch.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
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

std::atomic<int> g_bad{0};

constexpr uint32_t kAlive = 0xC0FFEE11u, kDead = 0xDEADDEADu;
constexpr size_t kPoolObjects = 400000;

// Canary object from a never-reused pool.
struct Obj : UsdGenRefCounted {
    explicit Obj(uint64_t v) : value(v), check(~v) {}
    ~Obj() override
    {
        if (canary != kAlive) g_bad.fetch_add(1);
        canary = kDead;
        destroyed.fetch_add(1);
    }
    bool Valid() const { return canary == kAlive && check == ~value; }

    static void *operator new(size_t n)
    {
        size_t const i = next.fetch_add(1);
        if (i >= kPoolObjects || n > sizeof(Cell)) std::abort();
        return &pool[i];
    }
    static void operator delete(void *) noexcept {}  // keep it poisoned

    uint32_t canary = kAlive;
    uint64_t value;
    uint64_t check;

    struct alignas(64) Cell { unsigned char b[128]; };
    static Cell *pool;
    static std::atomic<size_t> next;
    static std::atomic<size_t> destroyed;
};
Obj::Cell *Obj::pool = nullptr;
std::atomic<size_t> Obj::next{0};
std::atomic<size_t> Obj::destroyed{0};

size_t Live() { return Obj::next.load() - Obj::destroyed.load(); }

uint64_t Pending()
{
    UsdGenEpochStats const s = UsdGenEpochGetStats();
    return s.retired - s.freed;
}

void DrainAll()
{
    for (int i = 0; i < 64 && Pending(); ++i) UsdGenEpochDrain();
}

struct Plain : UsdGenEpochNode {
    static std::atomic<int> deleted;
    ~Plain() { deleted.fetch_add(1); }
};
std::atomic<int> Plain::deleted{0};

void TestSingleThread()
{
    size_t const live0 = Live();
    {
        UsdGenRef<Obj> a = UsdGenMakeRef<Obj>(1);
        Check(a->RefCount() == 1, "MakeRef holds one reference");
        UsdGenRef<Obj> b = a;
        Check(a->RefCount() == 2, "copy adds a reference");
        UsdGenAtomicRef<Obj> cell(a);
        Check(a->RefCount() == 3, "cell holds a reference");
        Check(cell.Load() == a, "Load returns the stored value");
        cell.Store(UsdGenMakeRef<Obj>(2));
        Check(cell.Load()->value == 2, "Store publishes");
        Check(a->RefCount() == 3, "displaced reference is released only through epochs");
        DrainAll();
        Check(a->RefCount() == 2, "drain releases the displaced reference");
        UsdGenRef<Obj> old = cell.Exchange(a);
        Check(old && old->value == 2, "Exchange returns the old value");
        Check(!cell.CompareExchange(old, UsdGenMakeRef<Obj>(3)), "CAS fails on a stale expected");
        Check(cell.CompareExchange(a, UsdGenMakeRef<Obj>(4)), "CAS succeeds on the current value");
        Check(cell.Load()->value == 4, "CAS published");
        {
            UsdGenEpochGuard g1;
            UsdGenEpochGuard g2;  // nests
            Obj *p = cell.Peek();
            cell.Store(nullptr);
            Check(UsdGenEpochDrain() == 0, "Drain on a pinned thread does nothing");
            Check(p->Valid(), "Peek stays valid under the guard");
        }
        cell.Store(UsdGenMakeRef<Obj>(5));
    }
    DrainAll();
    Check(Live() == live0, "every object freed after the cells and refs go");

    int const d0 = Plain::deleted.load();
    UsdGenEpochRetireDelete(new Plain);
    DrainAll();
    Check(Plain::deleted.load() == d0 + 1, "plain node deleted once through Retire");
}

void TestAccounting()
{
    DrainAll();
    UsdGenEpochStats const s0 = UsdGenEpochGetStats();
    std::atomic<int> phase{0};
    std::thread reader([&] {
        UsdGenEpochGuard g;
        phase.store(1);
        while (phase.load() != 2) std::this_thread::yield();
    });
    while (phase.load() != 1) std::this_thread::yield();
    UsdGenRef<Obj> held = UsdGenMakeRef<Obj>(7);
    constexpr int kN = 1000;
    for (int i = 0; i < kN; ++i) {
        held->AddRef();  // the reference each retire owes
        UsdGenEpochRetire(held.Get(), &UsdGenEpochReleaseNode);
    }
    for (int i = 0; i < kN; ++i) UsdGenEpochRetireDelete(new Plain);
    for (int i = 0; i < 16; ++i) UsdGenEpochDrain();
    UsdGenEpochStats const s1 = UsdGenEpochGetStats();
    Check(s1.retired - s0.retired == 2 * kN, "retired counts every Retire");
    Check(s1.freed == s0.freed, "a pinned reader holds every retire");
    Check(s1.epoch <= s0.epoch + 1, "epoch advances at most once past a pin");
    Check(held->RefCount() == kN + 1, "owed releases not yet run");
    phase.store(2);
    reader.join();
    DrainAll();
    UsdGenEpochStats const s2 = UsdGenEpochGetStats();
    Check(s2.retired == s2.freed, "retired == freed after the reader leaves");
    Check(held->RefCount() == 1, "counted re-retire releases once per retire");
}

void TestStress()
{
    constexpr int kThreads = 64, kWriters = 16, kCells = 4, kWrites = 5000, kReads = 40000;
    size_t const live0 = Live();
    UsdGenAtomicRef<Obj> cells[kCells];
    for (int c = 0; c < kCells; ++c) cells[c].Store(UsdGenMakeRef<Obj>(c));
    std::atomic<int> go{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            while (!go.load()) std::this_thread::yield();
            uint64_t x = 0x9E3779B97F4A7C15ull * (t + 1);
            auto rnd = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
            if (t < kWriters) {
                for (int i = 0; i < kWrites; ++i) {
                    UsdGenAtomicRef<Obj> &c = cells[rnd() % kCells];
                    switch (i % 3) {
                    case 0: c.Store(UsdGenMakeRef<Obj>(rnd())); break;
                    case 1: if (!c.Exchange(UsdGenMakeRef<Obj>(rnd()))->Valid()) g_bad.fetch_add(1); break;
                    default: {
                        UsdGenRef<Obj> e = c.Load();
                        c.CompareExchange(e, UsdGenMakeRef<Obj>(rnd()));
                    }
                    }
                    if (t == 0 && i % 64 == 0) UsdGenEpochDrain();  // the cook lane
                }
            } else {
                for (int i = 0; i < kReads; ++i) {
                    UsdGenAtomicRef<Obj> &c = cells[rnd() % kCells];
                    if (i & 1) {
                        UsdGenRef<Obj> r = c.Load();
                        if (!r || !r->Valid()) g_bad.fetch_add(1);
                    } else {
                        UsdGenEpochGuard g;
                        Obj *p = c.Peek();
                        if (!p || !p->Valid()) g_bad.fetch_add(1);
                    }
                }
            }
        });
    }
    go.store(1);
    for (auto &th : threads) th.join();
    std::printf("stress: %zu objects, epoch %llu\n", Obj::next.load() - live0,
                static_cast<unsigned long long>(UsdGenEpochGetStats().epoch));
    Check(g_bad.load() == 0, "64 threads: no use after free, no double destroy");
    for (auto &c : cells) c.Store(nullptr);
    DrainAll();
    Check(Pending() == 0, "stress: retired == freed");
    Check(Live() == live0, "stress: every object destroyed exactly once");
}

struct Item {
    Item *next = nullptr;
    int v = 0;
};

void TestHelpers()
{
    constexpr int kProducers = 8, kPer = 20000;
    std::vector<Item> items(kProducers * kPer);
    UsdGenTreiberStack<Item, &Item::next> stack;
    std::atomic<int> done{0};
    std::atomic<long long> popped{0};
    std::vector<std::thread> ts;
    for (int p = 0; p < kProducers; ++p)
        ts.emplace_back([&, p] {
            for (int i = 0; i < kPer; ++i) stack.Push(&items[p * kPer + i]);
            done.fetch_add(1);
        });
    std::thread consumer([&] {
        for (;;) {
            bool const last = done.load() == kProducers;
            for (Item *n = stack.PopAll(); n; n = n->next) ++n->v, popped.fetch_add(1);
            if (last) break;
        }
    });
    for (auto &t : ts) t.join();
    consumer.join();
    bool once = true;
    for (auto &i : items) once = once && i.v == 1;
    Check(popped.load() == kProducers * kPer && once, "Treiber stack: every push popped exactly once");

    // Push vs Close: many short rounds, each closing after a different
    // number of successful pushes, so Close lands while producers push.
    constexpr int kRounds = 200, kPerRound = 250;
    long long closedTotal = 0, selfTotal = 0;
    bool contOnce = true;
    for (int round = 0; round < kRounds; ++round) {
        UsdGenContinuationStack<Item, &Item::next> cont;
        std::vector<Item> conts(kProducers * kPerRound);
        std::atomic<int> ranSelf{0}, start{0}, pushed{0}, finished{0};
        int const closeAfter = (round * 7919) % (kProducers * kPerRound);
        ts.clear();
        for (int p = 0; p < kProducers; ++p)
            ts.emplace_back([&, p] {
                while (!start.load()) std::this_thread::yield();
                for (int i = 0; i < kPerRound; ++i) {
                    Item &it = conts[p * kPerRound + i];
                    if (cont.Push(&it)) pushed.fetch_add(1);
                    else ++it.v, ranSelf.fetch_add(1);
                }
                finished.fetch_add(1);
            });
        start.store(1);
        while (pushed.load() < closeAfter && finished.load() != kProducers) std::this_thread::yield();
        int closed = 0;
        for (Item *n = cont.Close(); n; n = n->next) ++n->v, ++closed;
        for (auto &t : ts) t.join();
        contOnce = contOnce && cont.IsClosed() && cont.Close() == nullptr &&
                   closed + ranSelf.load() == kProducers * kPerRound;
        for (auto &i : conts) contOnce = contOnce && i.v == 1;
        closedTotal += closed;
        selfTotal += ranSelf.load();
    }
    std::printf("continuation stack: %lld by the leader, %lld by themselves\n", closedTotal, selfTotal);
    Check(contOnce, "continuation stack: each continuation runs once, by the leader or itself");
    Check(closedTotal > 0 && selfTotal > 0, "continuation stack: both the leader and the late-push paths ran");

    size_t const live0 = Live();
    {
        UsdGenNewestWins<Obj> slot;
        constexpr uint64_t kN = 20000;
        std::thread producer([&] {
            for (uint64_t i = 1; i <= kN; ++i) slot.Publish(UsdGenMakeRef<Obj>(i));
        });
        uint64_t last = 0;
        bool monotonic = true;
        while (last != kN) {
            if (UsdGenRef<Obj> r = slot.Take()) {
                monotonic = monotonic && r->Valid() && r->value > last;
                last = r->value;
            }
        }
        producer.join();
        Check(monotonic && last == kN, "newest-wins: values only move forward and the last arrives");
    }
    Check(Live() == live0, "newest-wins: displaced values released");
}

// A thread_local whose destructor runs after the epoch's own thread state
// (constructed first, so destroyed last) still pins and retires. It must not
// claim a slot then: nobody would ever free it.
struct LateUser {
    UsdGenAtomicRef<Obj> *cell = nullptr;
    ~LateUser()
    {
        if (!cell) return;
        UsdGenRef<Obj> r = cell->Load();
        if (!r || !r->Valid()) g_bad.fetch_add(1);
        UsdGenEpochRetireDelete(new Plain);
    }
};
thread_local LateUser tLate;

void TestLateThreadLocal()
{
    DrainAll();
    UsdGenAtomicRef<Obj> cell(UsdGenMakeRef<Obj>(42));
    uint32_t const slots0 = UsdGenEpochGetStats().slotsInUse;
    int const d0 = Plain::deleted.load();
    int const bad0 = g_bad.load();
    for (int i = 0; i < 16; ++i)
        std::thread([&] {
            tLate.cell = &cell;          // constructed first: destroyed last
            UsdGenRef<Obj> r = cell.Load();  // now the epoch's state exists
        }).join();
    Check(UsdGenEpochGetStats().slotsInUse == slots0, "late thread_local use claims no slot");
    Check(g_bad.load() == bad0, "late thread_local reads a valid value");
    DrainAll();
    Check(Plain::deleted.load() == d0 + 16, "late thread_local retires are reclaimed");
}

void TestExhaustion()
{
    DrainAll();
    UsdGenEpochStats const s0 = UsdGenEpochGetStats();
    constexpr int kExtra = 8, kN = int(kUsdGenEpochSlots) + kExtra;
    std::atomic<int> ready{0}, release{0};
    std::vector<std::thread> ts;
    for (int i = 0; i < kN; ++i)
        ts.emplace_back([&] {
            UsdGenEpochGuard g;
            ready.fetch_add(1);
            while (!release.load()) std::this_thread::yield();
        });
    while (ready.load() != kN) std::this_thread::yield();
    UsdGenEpochStats const s1 = UsdGenEpochGetStats();
    Check(s1.slotsInUse == kUsdGenEpochSlots, "slot table full");
    Check(s1.overflowPins - s0.overflowPins >= uint64_t(kExtra),
          "threads past the table pin through the overflow counters");
    int const d0 = Plain::deleted.load();
    // This thread may itself be slotless now; retire from a fresh thread too.
    UsdGenEpochRetireDelete(new Plain);
    std::thread([&] { UsdGenEpochRetireDelete(new Plain); }).join();
    for (int i = 0; i < 16; ++i) UsdGenEpochDrain();
    Check(Plain::deleted.load() == d0, "overflow pins hold reclamation");
    release.store(1);
    for (auto &t : ts) t.join();
    DrainAll();
    Check(Plain::deleted.load() == d0 + 2, "reclaimed after the overflow pins leave");
    Check(UsdGenEpochGetStats().slotsInUse == s0.slotsInUse, "slots return at thread exit");
}

}  // namespace

int main()
{
    Obj::pool = static_cast<Obj::Cell *>(::operator new(sizeof(Obj::Cell) * kPoolObjects));
    TestSingleThread();
    TestAccounting();
    TestStress();
    TestHelpers();
    TestLateThreadLocal();
    TestExhaustion();
    Check(UsdGenEpochShutdown() == 0, "shutdown leaves nothing pending");
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
