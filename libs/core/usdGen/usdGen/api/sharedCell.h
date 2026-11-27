// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — lock-free publication of a std::shared_ptr, the one helper for
// every module (engine, device backends, Hydra host). Replaces
// std::atomic_load/store/exchange on shared_ptr and atomic<shared_ptr> (a
// hidden lock in MSVC and libstdc++):
//
//   UsdGenSharedCell<T const> cell;
//   cell.Store(p);                           // writer: one exchange
//   std::shared_ptr<T const> old = cell.Exchange(p);
//   auto prepared = cell.Prepare(p);         // allocate outside a fence,
//   cell.Exchange(std::move(prepared));      // then publish allocation-free
//   cell.CompareExchange(expected, p);       // by pointer; false if changed
//   std::shared_ptr<T const> v = cell.Load();  // any thread, never waits
//
// - The cell's reference moves to the writer that displaces it, so the old
//   value dies exactly when the last reference drops, as with atomic_exchange;
//   a pinned reader never delays it.
// - Readers reach the value through an epoch-protected holder and take their
//   reference with weak_ptr::lock (a CAS that never resurrects a dead count);
//   a reader that loses to a writer reloads the newer holder. Lock-free.
// - Holders are recycled through a free stack: no allocation after warm-up.
// - T may be const or not; Load hands back std::shared_ptr<T>.
//
// UsdGenCowUpdate edits a published value copy-on-write, and UsdGenLazyLeaked
// installs a process-lifetime object without a function-local static.
#pragma once

#include "usdGen/api/epoch.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

namespace usdGen {

struct UsdGenSharedCellHolder : UsdGenEpochNode {
    std::shared_ptr<const void> strong;  // the cell's reference
    std::weak_ptr<const void> weak;      // what readers lock
    void const *raw = nullptr;           // the published address
    UsdGenSharedCellHolder *freeNext = nullptr;
};

/// A holder from the free stack, or a new one (may throw bad_alloc).
UsdGenSharedCellHolder *UsdGenSharedCellAcquire(std::shared_ptr<const void> value);
/// Returns a never-published holder to the free stack.
void UsdGenSharedCellRecycle(UsdGenSharedCellHolder *h) noexcept;
/// Recycles a displaced holder once no reader can still see it.
void UsdGenSharedCellRetire(UsdGenSharedCellHolder *h) noexcept;

class UsdGenSharedCellBase {
public:
    constexpr UsdGenSharedCellBase() noexcept = default;
    UsdGenSharedCellBase(UsdGenSharedCellBase const &) = delete;
    UsdGenSharedCellBase &operator=(UsdGenSharedCellBase const &) = delete;

    explicit operator bool() const noexcept { return cur_.load(std::memory_order_acquire) != nullptr; }

protected:
    std::shared_ptr<const void> LoadVoid() const noexcept
    {
        for (;;) {
            UsdGenEpochGuard guard;
            UsdGenSharedCellHolder *h = cur_.load(std::memory_order_seq_cst);
            if (!h) return {};
            if (auto value = h->weak.lock()) return value;
            // Displaced and released meanwhile: the cell holds a newer one.
        }
    }
    std::shared_ptr<const void> ExchangeHolder(UsdGenSharedCellHolder *h) noexcept
    {
        return Displace(cur_.exchange(h, std::memory_order_seq_cst));
    }
    /// Installs h if the cell still publishes `expected`; h stays the
    /// caller's on failure. The displaced value is returned through *out.
    bool CompareExchangeHolder(void const *expected, UsdGenSharedCellHolder *h,
                               std::shared_ptr<const void> *out) noexcept
    {
        UsdGenSharedCellHolder *cur;
        {
            UsdGenEpochGuard guard;  // a holder seen here is not recycled (no ABA)
            cur = cur_.load(std::memory_order_seq_cst);
            if ((cur ? cur->raw : nullptr) != expected ||
                !cur_.compare_exchange_strong(cur, h, std::memory_order_seq_cst))
                return false;
        }
        *out = Displace(cur);
        return true;
    }
    static std::shared_ptr<const void> Displace(UsdGenSharedCellHolder *old) noexcept
    {
        if (!old) return {};
        std::shared_ptr<const void> displaced = std::move(old->strong);
        UsdGenSharedCellRetire(old);
        return displaced;
    }
    std::atomic<UsdGenSharedCellHolder *> cur_{nullptr};
};

template <class T>
class UsdGenSharedCell : public UsdGenSharedCellBase {
public:
    using Ptr = std::shared_ptr<T>;

    /// A value with its holder, ready to publish without allocating.
    class Prepared {
    public:
        Prepared() noexcept = default;
        Prepared(Prepared &&o) noexcept : h_(std::exchange(o.h_, nullptr)) {}
        Prepared &operator=(Prepared &&o) noexcept
        {
            if (this != &o) {
                Reset();
                h_ = std::exchange(o.h_, nullptr);
            }
            return *this;
        }
        ~Prepared() { Reset(); }
        T *Get() const noexcept { return h_ ? const_cast<T *>(static_cast<T const *>(h_->raw)) : nullptr; }
        T *operator->() const noexcept { return Get(); }
        explicit operator bool() const noexcept { return Get() != nullptr; }
        Ptr Value() const noexcept { return h_ ? Cast(h_->strong) : nullptr; }

    private:
        friend class UsdGenSharedCell;
        explicit Prepared(UsdGenSharedCellHolder *h) noexcept : h_(h) {}
        void Reset() noexcept
        {
            if (h_) UsdGenSharedCellRecycle(std::exchange(h_, nullptr));
        }
        UsdGenSharedCellHolder *h_ = nullptr;
    };

    constexpr UsdGenSharedCell() noexcept = default;
    explicit UsdGenSharedCell(Ptr v) { Store(std::move(v)); }
    ~UsdGenSharedCell() { (void)ExchangeHolder(nullptr); }

    /// Only a non-null value may allocate (and throw).
    static Prepared Prepare(Ptr v)
    {
        return Prepared(v ? UsdGenSharedCellAcquire(std::move(v)) : nullptr);
    }
    /// Any thread; never waits.
    Ptr Load() const noexcept { return Cast(LoadVoid()); }
    /// Publishes and returns the displaced value (the caller may drop it last).
    Ptr Exchange(Prepared p) noexcept { return Cast(ExchangeHolder(std::exchange(p.h_, nullptr))); }
    Ptr Exchange(Ptr v) { return Exchange(Prepare(std::move(v))); }
    void Store(Ptr v) { (void)Exchange(std::move(v)); }
    /// Clears the cell; never allocates.
    void Reset() noexcept { (void)ExchangeHolder(nullptr); }
    /// Clears the cell and returns what it held; never allocates.
    Ptr Take() noexcept { return Cast(ExchangeHolder(nullptr)); }
    /// Publishes `desired` if the cell still holds `expected` (compared by
    /// pointer); false if it changed first. Never waits; callers loop.
    bool CompareExchange(Ptr const &expected, Ptr desired)
    {
        Prepared p = Prepare(std::move(desired));
        std::shared_ptr<const void> displaced;
        if (!CompareExchangeHolder(expected.get(), p.h_, &displaced)) return false;
        p.h_ = nullptr;
        return true;
    }

private:
    static Ptr Cast(std::shared_ptr<const void> const &v) noexcept
    {
        return std::const_pointer_cast<T>(std::static_pointer_cast<T const>(v));
    }
};

/// Copy-on-write update of a published value: `edit(T&)` runs on a copy of
/// the current value (or a default one) until the CAS lands. `edit` may run
/// more than once and must only touch the copy. Returns the value installed.
template <class T, class Edit>
std::shared_ptr<T const> UsdGenCowUpdate(UsdGenSharedCell<T const> &cell, Edit &&edit)
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
bool UsdGenCowUpdateIf(UsdGenSharedCell<T const> &cell, Edit &&edit)
{
    for (;;) {
        std::shared_ptr<T const> current = cell.Load();
        auto next = current ? std::make_shared<T>(*current) : std::make_shared<T>();
        if (!edit(*next)) return false;
        if (cell.CompareExchange(current, std::shared_ptr<T const>(std::move(next)))) return true;
    }
}

/// Sequence word UsdGenLazyLeaked waiters park on (sharedCell.cpp).
uint32_t UsdGenLazyLeakedSequence() noexcept;
void UsdGenLazyLeakedPark(uint32_t seen) noexcept;
void UsdGenLazyLeakedPublished() noexcept;

/// Installs a process-lifetime object into a constant-initialized atomic
/// pointer (no function-local static, no guard lock). Exactly one caller
/// runs `make`: it claims the slot with a sentinel by CAS, and racing first
/// callers park on a sequence word until it publishes (a thread that owns
/// workers is never built twice). Nothing it makes is destroyed, so no
/// destructor that might wait runs at DLL detach. If `make` throws, the slot
/// is reopened and the exception propagates. `make` must not re-enter the
/// same slot. Read the slot only through this function.
template <class T, class Make>
T &UsdGenLazyLeaked(std::atomic<T *> &slot, Make &&make)
{
    T *const busy = reinterpret_cast<T *>(uintptr_t(1));
    T *p = slot.load(std::memory_order_acquire);
    if (p && p != busy) return *p;
    for (;;) {
        p = nullptr;
        if (slot.compare_exchange_strong(p, busy, std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
            T *made = nullptr;
            try {
                made = make();
            } catch (...) {
                slot.store(nullptr, std::memory_order_seq_cst);
                UsdGenLazyLeakedPublished();
                throw;
            }
            slot.store(made, std::memory_order_seq_cst);
            UsdGenLazyLeakedPublished();
            return *made;
        }
        if (p != busy) return *p;
        // Read the word before re-checking the slot: a publish in between
        // changes the word, so the park returns at once.
        uint32_t const seen = UsdGenLazyLeakedSequence();
        p = slot.load(std::memory_order_seq_cst);
        if (p == busy) {
            UsdGenLazyLeakedPark(seen);
        } else if (p) {
            return *p;
        }
    }
}

template <class T>
T &UsdGenLazyLeaked(std::atomic<T *> &slot)
{
    return UsdGenLazyLeaked(slot, [] { return new T; });
}

}  // namespace usdGen
