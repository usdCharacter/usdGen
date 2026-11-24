// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// Epoch reclamation. Global state is constant-initialized and trivially
// destructible, so nothing here runs in a static destructor or atexit. The
// orderings that make a pin visible to the advancer are seq_cst (the pin store
// is the store-load fence the scheme needs).
#include "usdGen/epoch.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#if defined(_MSC_VER)
#  pragma warning(disable : 4324)  // padding from alignas is the point
#endif

namespace usdGen {

using Node = UsdGenEpochNode;

struct UsdGenEpochInternal {
    static Node *&Next(Node *n) noexcept { return n->epochNext_; }
    static uint64_t &Retired(Node *n) noexcept { return n->epochRetired_; }
    static UsdGenEpochDeleter &Deleter(Node *n) noexcept { return n->epochDeleter_; }
    static std::atomic<uint32_t> &Pending(Node *n) noexcept { return n->epochPending_; }
    static uint32_t &Batch(Node *n) noexcept { return n->epochBatch_; }
};
using I = UsdGenEpochInternal;

namespace {

constexpr auto kSeq = std::memory_order_seq_cst;

struct alignas(64) Slot {
    std::atomic<uint32_t> owner{0};
    std::atomic<uint64_t> pinned{0};  // 0 idle, else the pinned epoch
    std::atomic<Node *> retired{nullptr};
};

struct alignas(64) Counter {
    std::atomic<uint64_t> v{0};
};

struct State {
    alignas(64) std::atomic<uint64_t> epoch{1};
    alignas(64) std::atomic<uint32_t> slotsHigh{0};
    std::atomic<uint32_t> slotsInUse{0};
    Counter overflow[2];  // pins without a slot, by epoch parity
    alignas(64) std::atomic<uint64_t> retired{0};
    std::atomic<uint64_t> freed{0};
    std::atomic<uint64_t> overflowPins{0};
    std::atomic<Node *> orphans{nullptr};  // lists of threads without a slot
    Slot slots[kUsdGenEpochSlots];
};
State g;

struct ThreadState {
    Slot *slot = nullptr;
    Slot *pinnedSlot = nullptr;
    Counter *pinnedCounter = nullptr;
    uint32_t depth = 0;
    uint32_t claimBackoff = 0;
    uint32_t sinceDrain = 0;
    bool draining = false;
    ~ThreadState();
};
thread_local ThreadState t;

void PushChain(std::atomic<Node *> &head, Node *first, Node *last) noexcept
{
    Node *h = head.load(std::memory_order_relaxed);
    do { I::Next(last) = h; } while (!head.compare_exchange_weak(h, first, std::memory_order_release,
                                                                    std::memory_order_relaxed));
}

ThreadState::~ThreadState()
{
    // Thread exit (also at DLL detach): hand the list over, free the slot.
    if (!slot) return;
    if (Node *first = slot->retired.exchange(nullptr, std::memory_order_acquire)) {
        Node *last = first;
        while (I::Next(last)) last = I::Next(last);
        PushChain(g.orphans, first, last);
    }
    slot->pinned.store(0, std::memory_order_release);
    slot->owner.store(0, std::memory_order_release);
    g.slotsInUse.fetch_sub(1, std::memory_order_relaxed);
    slot = nullptr;
}

Slot *OwnSlot() noexcept
{
    if (t.slot) return t.slot;
    if (t.claimBackoff) {
        --t.claimBackoff;
        return nullptr;
    }
    for (uint32_t i = 0; i < kUsdGenEpochSlots; ++i) {
        Slot &s = g.slots[i];
        uint32_t free = 0;
        if (s.owner.load(std::memory_order_relaxed) != 0 ||
            !s.owner.compare_exchange_strong(free, 1, std::memory_order_acq_rel))
            continue;
        uint32_t h = g.slotsHigh.load(kSeq);
        while (h < i + 1 && !g.slotsHigh.compare_exchange_weak(h, i + 1, kSeq)) {}
        g.slotsInUse.fetch_add(1, std::memory_order_relaxed);
        return t.slot = &s;
    }
    t.claimBackoff = 64;  // table full: retry the claim later
    return nullptr;
}

bool ProcessDetaching() noexcept
{
#if defined(_WIN32)
    using Fn = BOOLEAN(NTAPI *)();
    auto fn = reinterpret_cast<Fn>(reinterpret_cast<void *>(
        ::GetProcAddress(::GetModuleHandleW(L"ntdll.dll"), "RtlDllShutdownInProgress")));
    return fn && fn();
#else
    return false;
#endif
}

// Advance from e to e+1 when every pin is at e. Lock-free: a losing CAS
// means someone else advanced.
bool TryAdvance() noexcept
{
    uint64_t e = g.epoch.load(kSeq);
    uint32_t const hi = g.slotsHigh.load(kSeq);
    for (uint32_t i = 0; i < hi; ++i) {
        uint64_t const p = g.slots[i].pinned.load(kSeq);
        if (p && p != e) return false;
    }
    if (g.overflow[(e + 1) & 1].v.load(kSeq)) return false;  // pins at e-1
    return g.epoch.compare_exchange_strong(e, e + 1, kSeq);
}

size_t DrainList(std::atomic<Node *> &head) noexcept
{
    Node *list = head.exchange(nullptr, std::memory_order_acquire);
    Node *keep = nullptr, *keepLast = nullptr;
    size_t n = 0;
    uint64_t const now = g.epoch.load(kSeq);
    while (list) {
        Node *node = list;
        list = I::Next(node);
        bool requeue = I::Retired(node) + 2 > now;
        if (!requeue) {
            UsdGenEpochDeleter const fn = I::Deleter(node);
            uint32_t const k = I::Batch(node);
            // Retires that landed while queued are newer: requeue them at now.
            uint32_t const left = I::Pending(node).fetch_sub(k, kSeq) - k;
            if (left) {
                I::Batch(node) = left;
                I::Retired(node) = g.epoch.load(kSeq);
                requeue = true;
            }
            for (uint32_t i = 0; i < k; ++i) fn(node);  // the last may free node
            n += k;
        }
        if (requeue) {
            I::Next(node) = keep;
            if (!keep) keepLast = node;
            keep = node;
        }
    }
    if (keep) PushChain(head, keep, keepLast);
    return n;
}

}  // namespace

void UsdGenEpochEnter() noexcept
{
    if (t.depth++) return;
    if (Slot *s = OwnSlot()) {
        s->pinned.store(g.epoch.load(kSeq), kSeq);
        t.pinnedSlot = s;
        return;
    }
    for (;;) {  // retries only if the epoch moved meanwhile
        uint64_t const e = g.epoch.load(kSeq);
        Counter &c = g.overflow[e & 1];
        c.v.fetch_add(1, kSeq);
        if (g.epoch.load(kSeq) == e) {
            t.pinnedCounter = &c;
            break;
        }
        c.v.fetch_sub(1, std::memory_order_release);
    }
    g.overflowPins.fetch_add(1, std::memory_order_relaxed);
}

void UsdGenEpochLeave() noexcept
{
    if (--t.depth) return;
    if (t.pinnedSlot) {
        t.pinnedSlot->pinned.store(0, std::memory_order_release);
        t.pinnedSlot = nullptr;
    } else if (t.pinnedCounter) {
        t.pinnedCounter->v.fetch_sub(1, std::memory_order_release);
        t.pinnedCounter = nullptr;
    }
}

void UsdGenEpochRetire(Node *node, UsdGenEpochDeleter deleter) noexcept
{
    if (!node) return;
    g.retired.fetch_add(1, std::memory_order_relaxed);
    if (I::Pending(node).fetch_add(1, kSeq) != 0) return;  // already queued
    I::Deleter(node) = deleter;
    I::Batch(node) = 1;
    I::Retired(node) = g.epoch.load(kSeq);
    Slot *s = OwnSlot();
    PushChain(s ? s->retired : g.orphans, node, node);
    if (++t.sinceDrain >= kUsdGenEpochAutoDrain && !t.depth && !t.draining) {
        t.sinceDrain = 0;
        if (!ProcessDetaching()) UsdGenEpochDrain();
    }
}

size_t UsdGenEpochDrain() noexcept
{
    if (t.depth || t.draining) return 0;  // a pinned drainer would block itself
    t.draining = true;
    if (TryAdvance()) TryAdvance();
    size_t n = 0;
    uint32_t const hi = g.slotsHigh.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < hi; ++i) {
        if (g.slots[i].retired.load(std::memory_order_relaxed)) n += DrainList(g.slots[i].retired);
    }
    if (g.orphans.load(std::memory_order_relaxed)) n += DrainList(g.orphans);
    g.freed.fetch_add(n, std::memory_order_relaxed);
    t.draining = false;
    return n;
}

size_t UsdGenEpochShutdown() noexcept
{
    if (!ProcessDetaching()) {
        // Bounded passes: deleters that retire more objects get two epochs each.
        for (int i = 0; i < 4; ++i) UsdGenEpochDrain();
    }
    UsdGenEpochStats const s = UsdGenEpochGetStats();
    return size_t(s.retired - s.freed);
}

UsdGenEpochStats UsdGenEpochGetStats() noexcept
{
    UsdGenEpochStats s;
    s.epoch = g.epoch.load(std::memory_order_acquire);
    s.freed = g.freed.load(std::memory_order_acquire);
    s.retired = g.retired.load(std::memory_order_acquire);
    s.overflowPins = g.overflowPins.load(std::memory_order_relaxed);
    s.slotsInUse = g.slotsInUse.load(std::memory_order_relaxed);
    return s;
}

void UsdGenRefCountedDestroy(UsdGenRefCounted const *p) noexcept { delete p; }

void UsdGenEpochReleaseNode(Node *n) noexcept { static_cast<UsdGenRefCounted *>(n)->Release(); }

}  // namespace usdGen
