// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen imaging — lock-free publication helpers for the Hydra host.
//
// UsdGenAtomicShared<T> replaces std::atomic_load/store on a shared_ptr
// (a lock inside MSVC and libstdc++). The cell holds an atomic raw pointer to
// an intrusively counted box carrying the shared_ptr; readers pin an epoch,
// copy the box's shared_ptr and leave; writers exchange in a new box and
// retire the old one through the epochs. Readers and writers never wait.
// Writers drain the epochs right after a swap, so a displaced value is
// released at once when no reader is pinned.
//
// UsdGenLazyLeaked installs a process-lifetime object into a constant-
// initialized atomic pointer by CAS (no function-local static, no guard
// lock). Nothing it makes is ever destroyed, so no destructor that might
// wait runs at DLL detach or on a racing loser: a loser's candidate (only
// when two threads race the first call) is leaked like the winner.
#ifndef USDGEN_IMAGING_ATOMIC_SHARED_H
#define USDGEN_IMAGING_ATOMIC_SHARED_H

#include "usdGen/epoch.h"

#include <atomic>
#include <memory>
#include <utility>

namespace usdGenImaging {

template <class T>
class UsdGenAtomicShared {
public:
    using Ptr = std::shared_ptr<T>;

    constexpr UsdGenAtomicShared() noexcept = default;
    explicit UsdGenAtomicShared(Ptr v) : p_(v ? new Box(std::move(v)) : nullptr)
    {
        if (Box *b = p_.load(std::memory_order_relaxed)) b->AddRef();
    }
    ~UsdGenAtomicShared() { _Retire(p_.load(std::memory_order_relaxed)); }
    UsdGenAtomicShared(UsdGenAtomicShared const &) = delete;
    UsdGenAtomicShared &operator=(UsdGenAtomicShared const &) = delete;

    /// Any thread; never waits.
    Ptr Load() const noexcept
    {
        usdGen::UsdGenEpochGuard guard;
        Box const *b = p_.load(std::memory_order_seq_cst);
        return b ? b->value : Ptr();
    }
    explicit operator bool() const noexcept { return p_.load(std::memory_order_acquire) != nullptr; }

    void Store(Ptr v) { _Swap(_Make(std::move(v))); }

    Ptr Exchange(Ptr v)
    {
        Box *old = p_.exchange(_Make(std::move(v)), std::memory_order_seq_cst);
        Ptr r;
        if (old) {
            usdGen::UsdGenEpochGuard guard;  // other readers may still copy it
            r = old->value;
        }
        _Retire(old);
        usdGen::UsdGenEpochDrain();
        return r;
    }

    /// Replaces `expected` (compared by pointer) with `desired`; false if
    /// the cell changed first. Never waits; callers loop.
    bool CompareExchange(Ptr const &expected, Ptr desired)
    {
        Box *nb = _Make(std::move(desired));
        Box *cur;
        {
            usdGen::UsdGenEpochGuard guard;
            cur = p_.load(std::memory_order_seq_cst);
            T *const seen = cur ? cur->value.get() : nullptr;
            if (seen != expected.get() ||
                !p_.compare_exchange_strong(cur, nb, std::memory_order_seq_cst)) {
                if (nb) nb->Release();
                return false;
            }
        }
        _Retire(cur);
        usdGen::UsdGenEpochDrain();
        return true;
    }

private:
    struct Box final : usdGen::UsdGenRefCounted {
        explicit Box(Ptr v) : value(std::move(v)) {}
        Ptr const value;
    };
    static Box *_Make(Ptr v)
    {
        if (!v) return nullptr;
        Box *b = new Box(std::move(v));
        b->AddRef();  // the cell's reference
        return b;
    }
    void _Swap(Box *nb)
    {
        _Retire(p_.exchange(nb, std::memory_order_seq_cst));
        usdGen::UsdGenEpochDrain();
    }
    static void _Retire(Box *b) noexcept
    {
        if (b) usdGen::UsdGenEpochRetire(b, &usdGen::UsdGenEpochReleaseNode);
    }
    std::atomic<Box *> p_{nullptr};
};

/// Copy-on-write update of a published value: `edit(T&)` runs on a copy of
/// the current value (or a default one) until the CAS lands. `edit` may run
/// more than once and must only touch the copy. Returns the value installed.
template <class T, class Edit>
std::shared_ptr<T const> UsdGenCowUpdate(UsdGenAtomicShared<T const> &cell, Edit &&edit)
{
    for (;;) {
        std::shared_ptr<T const> current = cell.Load();
        auto next = current ? std::make_shared<T>(*current) : std::make_shared<T>();
        edit(*next);
        std::shared_ptr<T const> installed = std::move(next);
        if (cell.CompareExchange(current, installed)) return installed;
    }
}

/// As UsdGenCowUpdate, but `edit(T&)` returns false to leave the cell
/// unchanged; returns whether an edited copy was installed.
template <class T, class Edit>
bool UsdGenCowUpdateIf(UsdGenAtomicShared<T const> &cell, Edit &&edit)
{
    for (;;) {
        std::shared_ptr<T const> current = cell.Load();
        auto next = current ? std::make_shared<T>(*current) : std::make_shared<T>();
        if (!edit(*next)) return false;
        if (cell.CompareExchange(current, std::shared_ptr<T const>(std::move(next)))) return true;
    }
}

template <class T, class Make>
T &UsdGenLazyLeaked(std::atomic<T *> &slot, Make &&make)
{
    T *p = slot.load(std::memory_order_acquire);
    if (p) return *p;
    T *candidate = make();
    if (slot.compare_exchange_strong(p, candidate, std::memory_order_acq_rel,
                                     std::memory_order_acquire))
        return *candidate;
    return *p;  // lost the race: p holds the winner; candidate stays leaked
}

template <class T>
T &UsdGenLazyLeaked(std::atomic<T *> &slot)
{
    return UsdGenLazyLeaked(slot, [] { return new T; });
}

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_ATOMIC_SHARED_H
