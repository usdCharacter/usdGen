// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — lock-free publication of a std::shared_ptr. Replaces
// std::atomic_load/store/exchange on shared_ptr (a hidden lock in MSVC and
// libstdc++) where the published type must stay a shared_ptr:
//
//   UsdGenSharedCell<T> cell;
//   cell.Store(p);                           // writer: one exchange
//   std::shared_ptr<const T> old = cell.Exchange(p);
//   auto prepared = cell.Prepare(p);         // allocate outside a fence,
//   cell.Exchange(std::move(prepared));      // then publish allocation-free
//   std::shared_ptr<const T> v = cell.Load();  // any thread, never waits
//
// - The cell's reference moves to the writer that displaces it, so the old
//   value dies exactly when the last reference drops, as with atomic_exchange.
// - Readers reach the value through an epoch-protected holder and take their
//   reference with weak_ptr::lock (a CAS that never resurrects a dead count);
//   a reader that loses to a writer reloads the newer holder. Lock-free.
// - Holders are recycled through a free stack: no allocation after warm-up.
#pragma once

#include "usdGen/epoch.h"

#include <atomic>
#include <memory>
#include <utility>

namespace usdGen {

struct UsdGenSharedCellHolder : UsdGenEpochNode {
    std::shared_ptr<const void> strong;  // the cell's reference
    std::weak_ptr<const void> weak;      // what readers lock
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
        UsdGenSharedCellHolder *old = cur_.exchange(h, std::memory_order_seq_cst);
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
    using Ptr = std::shared_ptr<const T>;

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
        T const *Get() const noexcept { return h_ ? static_cast<T const *>(h_->strong.get()) : nullptr; }
        T const *operator->() const noexcept { return Get(); }
        explicit operator bool() const noexcept { return Get() != nullptr; }
        Ptr Value() const noexcept
        {
            return h_ ? std::static_pointer_cast<const T>(h_->strong) : nullptr;
        }

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

    static Prepared Prepare(Ptr v)
    {
        return Prepared(v ? UsdGenSharedCellAcquire(std::move(v)) : nullptr);
    }
    Ptr Load() const noexcept { return std::static_pointer_cast<const T>(LoadVoid()); }
    /// Publishes and returns the displaced value (the caller may drop it last).
    Ptr Exchange(Prepared p) noexcept
    {
        return std::static_pointer_cast<const T>(ExchangeHolder(std::exchange(p.h_, nullptr)));
    }
    Ptr Exchange(Ptr v) { return Exchange(Prepare(std::move(v))); }
    void Store(Ptr v) { (void)Exchange(std::move(v)); }
};

/// For namespace-scope cells: constant-initialized, no destructor, so nothing
/// runs at DLL detach. The last value is leaked with the process.
template <class T>
class UsdGenStaticSharedCell : public UsdGenSharedCellBase {
public:
    using Ptr = std::shared_ptr<const T>;
    constexpr UsdGenStaticSharedCell() noexcept = default;
    Ptr Load() const noexcept { return std::static_pointer_cast<const T>(LoadVoid()); }
    Ptr Exchange(Ptr v)
    {
        return std::static_pointer_cast<const T>(
            ExchangeHolder(v ? UsdGenSharedCellAcquire(std::move(v)) : nullptr));
    }
    void Store(Ptr v) { (void)Exchange(std::move(v)); }
};

}  // namespace usdGen
