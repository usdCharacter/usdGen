// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — epoch reclamation, intrusive refcounts and atomic publication.
// The replacement for atomic shared_ptr loads and stores:
//
//   struct Snapshot : UsdGenRefCounted { ... };
//   UsdGenAtomicRef<Snapshot> cell;                // one per published value
//   cell.Store(UsdGenMakeRef<Snapshot>(...));      // writer: exchange; the old
//                                                  // value is retired via epochs
//   UsdGenRef<Snapshot> s = cell.Load();           // any thread, never waits
//   { UsdGenEpochGuard g; Snapshot *p = cell.Peek(); }  // no refcount traffic
//   UsdGenEpochDrain();                            // cook lane / quiescent points
//
// - A pointer read under a guard stays valid until the guard ends; hold a
//   UsdGenRef for anything longer. Guards nest, never block, and must not span
//   a callback or host code.
// - Retired objects are freed by Drain once the global epoch is two past their
//   retire epoch. Drain never waits and does nothing on a pinned thread or
//   during process detach. Every kUsdGenEpochAutoDrain retires, an unpinned
//   thread also frees its own due retires (never other threads'). Idle
//   scheduler workers run a full Drain before they park.
// - A thread_local destroyed after the epoch's own thread state may still pin
//   and retire; it pins through the overflow counters and claims no slot.
// - A thread claims one of kUsdGenEpochSlots slots on first use (CAS, no
//   allocation) and frees it at thread exit. With the table full, threads pin
//   through two shared counters instead: correct, just contended.
// - Plain objects derive from UsdGenEpochNode and are retired exactly once with
//   UsdGenEpochRetire(node, deleter) or UsdGenEpochRetireDelete(p).
// - UsdGenEpochShutdown never blocks; during DLL process detach it runs no
//   deleters and leaks whatever is still pending.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace usdGen {

constexpr uint32_t kUsdGenEpochSlots = 256;
constexpr uint32_t kUsdGenEpochAutoDrain = 128;

class UsdGenEpochNode;
using UsdGenEpochDeleter = void (*)(UsdGenEpochNode *);

/// Intrusive retire-list link; derive from it. Copies start unlinked.
class UsdGenEpochNode {
public:
    UsdGenEpochNode() noexcept = default;
    UsdGenEpochNode(UsdGenEpochNode const &) noexcept {}
    UsdGenEpochNode &operator=(UsdGenEpochNode const &) noexcept { return *this; }

private:
    friend struct UsdGenEpochInternal;
    UsdGenEpochNode *epochNext_ = nullptr;
    uint64_t epochRetired_ = 0;
    UsdGenEpochDeleter epochDeleter_ = nullptr;
    std::atomic<uint32_t> epochPending_{0};  // deleter calls owed; >0 = queued
    uint32_t epochBatch_ = 0;                // calls covered by epochRetired_
};

void UsdGenEpochEnter() noexcept;
void UsdGenEpochLeave() noexcept;

/// Queue `deleter(node)` for when no guard can still see `node`. A node
/// retired again while queued runs its deleter once per retire (refcounts).
void UsdGenEpochRetire(UsdGenEpochNode *node, UsdGenEpochDeleter deleter) noexcept;

/// Try to advance the epoch and run every due deleter. Returns deleters run.
/// Returns 0 at once on a pinned thread, with nothing pending, or during
/// process detach.
size_t UsdGenEpochDrain() noexcept;

/// Drain what is safe without waiting; returns retires still pending.
size_t UsdGenEpochShutdown() noexcept;

struct UsdGenEpochStats {
    uint64_t epoch = 0;
    uint64_t retired = 0;       // UsdGenEpochRetire calls
    uint64_t freed = 0;         // deleters run
    uint64_t overflowPins = 0;  // pins taken without a slot
    uint32_t slotsInUse = 0;
};
UsdGenEpochStats UsdGenEpochGetStats() noexcept;

/// RAII pin of the current thread.
class UsdGenEpochGuard {
public:
    UsdGenEpochGuard() noexcept { UsdGenEpochEnter(); }
    ~UsdGenEpochGuard() { UsdGenEpochLeave(); }
    UsdGenEpochGuard(UsdGenEpochGuard const &) = delete;
    UsdGenEpochGuard &operator=(UsdGenEpochGuard const &) = delete;
};

template <class T>
void UsdGenEpochRetireDelete(T *p) noexcept
{
    UsdGenEpochRetire(p, [](UsdGenEpochNode *n) { delete static_cast<T *>(n); });
}

class UsdGenRefCounted;
/// Exported: the last release always frees through usdGen.
void UsdGenRefCountedDestroy(UsdGenRefCounted const *p) noexcept;
/// The deferred release AtomicRef retires with.
void UsdGenEpochReleaseNode(UsdGenEpochNode *n) noexcept;

/// Intrusive refcount base. The count starts at 0; UsdGenRef owns references.
class UsdGenRefCounted : public UsdGenEpochNode {
public:
    void AddRef() const noexcept { refs_.fetch_add(1, std::memory_order_relaxed); }
    void Release() const noexcept
    {
        if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) UsdGenRefCountedDestroy(this);
    }
    uint32_t RefCount() const noexcept { return refs_.load(std::memory_order_relaxed); }

protected:
    UsdGenRefCounted() noexcept = default;
    UsdGenRefCounted(UsdGenRefCounted const &) noexcept : UsdGenEpochNode() {}
    UsdGenRefCounted &operator=(UsdGenRefCounted const &) noexcept { return *this; }
    virtual ~UsdGenRefCounted() = default;

private:
    friend void UsdGenRefCountedDestroy(UsdGenRefCounted const *) noexcept;
    mutable std::atomic<uint32_t> refs_{0};
};

/// Owning intrusive pointer.
template <class T>
class UsdGenRef {
public:
    UsdGenRef() noexcept = default;
    UsdGenRef(std::nullptr_t) noexcept {}
    explicit UsdGenRef(T *p) noexcept : p_(p) { if (p_) p_->AddRef(); }
    UsdGenRef(UsdGenRef const &o) noexcept : UsdGenRef(o.p_) {}
    UsdGenRef(UsdGenRef &&o) noexcept : p_(o.Detach()) {}
    template <class U>
    UsdGenRef(UsdGenRef<U> o) noexcept : p_(o.Detach()) {}
    ~UsdGenRef() { if (p_) p_->Release(); }
    UsdGenRef &operator=(UsdGenRef o) noexcept { std::swap(p_, o.p_); return *this; }

    /// Takes over a reference the caller already holds.
    static UsdGenRef Adopt(T *p) noexcept { UsdGenRef r; r.p_ = p; return r; }
    /// Gives up the reference without releasing it.
    T *Detach() noexcept { T *p = p_; p_ = nullptr; return p; }
    void Reset() noexcept { UsdGenRef().Swap(*this); }
    void Swap(UsdGenRef &o) noexcept { std::swap(p_, o.p_); }

    T *Get() const noexcept { return p_; }
    T *operator->() const noexcept { return p_; }
    T &operator*() const noexcept { return *p_; }
    explicit operator bool() const noexcept { return p_ != nullptr; }
    friend bool operator==(UsdGenRef const &a, UsdGenRef const &b) noexcept { return a.p_ == b.p_; }
    friend bool operator!=(UsdGenRef const &a, UsdGenRef const &b) noexcept { return a.p_ != b.p_; }

private:
    T *p_ = nullptr;
};

template <class T, class... A>
UsdGenRef<T> UsdGenMakeRef(A &&...a)
{
    return UsdGenRef<T>(new T(std::forward<A>(a)...));
}

/// Atomically published reference. Readers never wait; writers never wait;
/// the reference a writer displaces is released through the epochs.
template <class T>
class UsdGenAtomicRef {
public:
    UsdGenAtomicRef() noexcept = default;
    explicit UsdGenAtomicRef(UsdGenRef<T> v) noexcept : p_(v.Detach()) {}
    ~UsdGenAtomicRef() { _Retire(p_.load(std::memory_order_relaxed)); }
    UsdGenAtomicRef(UsdGenAtomicRef const &) = delete;
    UsdGenAtomicRef &operator=(UsdGenAtomicRef const &) = delete;

    UsdGenRef<T> Load() const noexcept
    {
        UsdGenEpochGuard guard;
        return UsdGenRef<T>(p_.load(std::memory_order_seq_cst));
    }
    /// Raw read; valid only while the caller holds a UsdGenEpochGuard.
    T *Peek() const noexcept { return p_.load(std::memory_order_seq_cst); }

    void Store(UsdGenRef<T> v) noexcept { _Retire(p_.exchange(v.Detach(), std::memory_order_seq_cst)); }
    UsdGenRef<T> Exchange(UsdGenRef<T> v) noexcept
    {
        T *old = p_.exchange(v.Detach(), std::memory_order_seq_cst);
        UsdGenRef<T> r(old);  // the cell's reference is still owed here
        _Retire(old);
        return r;
    }
    /// Publishes `desired` if the cell still holds `expected` (held by the
    /// caller, so its address cannot be reused: no ABA).
    bool CompareExchange(UsdGenRef<T> const &expected, UsdGenRef<T> desired) noexcept
    {
        T *e = expected.Get();
        if (!p_.compare_exchange_strong(e, desired.Get(), std::memory_order_seq_cst)) return false;
        desired.Detach();
        _Retire(expected.Get());
        return true;
    }

private:
    static void _Retire(T *p) noexcept
    {
        if (p) UsdGenEpochRetire(const_cast<UsdGenRefCounted *>(static_cast<UsdGenRefCounted const *>(p)),
                                 &UsdGenEpochReleaseNode);
    }
    std::atomic<T *> p_{nullptr};
};

/// Newest-wins handoff: producers publish, one consumer takes the latest.
/// Values never read in place, so displaced ones are released directly.
template <class T>
class UsdGenNewestWins {
public:
    UsdGenNewestWins() noexcept = default;
    ~UsdGenNewestWins() { if (T *p = p_.load(std::memory_order_acquire)) p->Release(); }
    UsdGenNewestWins(UsdGenNewestWins const &) = delete;
    UsdGenNewestWins &operator=(UsdGenNewestWins const &) = delete;

    void Publish(UsdGenRef<T> v) noexcept
    {
        if (T *old = p_.exchange(v.Detach(), std::memory_order_acq_rel)) old->Release();
    }
    UsdGenRef<T> Take() noexcept { return UsdGenRef<T>::Adopt(p_.exchange(nullptr, std::memory_order_acq_rel)); }
    bool HasValue() const noexcept { return p_.load(std::memory_order_acquire) != nullptr; }

private:
    std::atomic<T *> p_{nullptr};
};

/// Lock-free intrusive LIFO through `T::*Next`. Push and PopAll only, so
/// there is no ABA.
template <class T, T *T::*Next>
class UsdGenTreiberStack {
public:
    void Push(T *n) noexcept { PushChain(n, n); }
    /// Pushes first..last, already linked through Next.
    void PushChain(T *first, T *last) noexcept
    {
        T *h = head_.load(std::memory_order_relaxed);
        do { last->*Next = h; } while (!head_.compare_exchange_weak(h, first, std::memory_order_release,
                                                                      std::memory_order_relaxed));
    }
    T *PopAll() noexcept { return head_.exchange(nullptr, std::memory_order_acquire); }
    /// Installs the chain at first (already linked) only if the stack is
    /// empty; false leaves the chain with the caller.
    bool InstallIfEmpty(T *first) noexcept
    {
        T *expected = nullptr;
        return head_.compare_exchange_strong(expected, first, std::memory_order_release,
                                             std::memory_order_relaxed);
    }
    bool Empty() const noexcept { return head_.load(std::memory_order_acquire) == nullptr; }

private:
    std::atomic<T *> head_{nullptr};
};

/// Continuation stack for coalescing: followers Push until the leader Closes;
/// Close returns everything pushed, and a later Push fails (run it yourself).
template <class T, T *T::*Next>
class UsdGenContinuationStack {
public:
    bool Push(T *n) noexcept
    {
        T *h = head_.load(std::memory_order_acquire);
        do {
            if (h == _Closed()) return false;
            n->*Next = h;
        } while (!head_.compare_exchange_weak(h, n, std::memory_order_acq_rel, std::memory_order_acquire));
        return true;
    }
    T *Close() noexcept
    {
        T *h = head_.exchange(_Closed(), std::memory_order_acq_rel);
        return h == _Closed() ? nullptr : h;
    }
    bool IsClosed() const noexcept { return head_.load(std::memory_order_acquire) == _Closed(); }
    /// The newest pushed node, or null once closed. Links below it never
    /// change until Close; walk them under a UsdGenEpochGuard.
    T *Peek() const noexcept
    {
        T *h = head_.load(std::memory_order_acquire);
        return h == _Closed() ? nullptr : h;
    }
    /// Owner only, after Close: accept pushes again.
    void Reopen() noexcept { head_.store(nullptr, std::memory_order_release); }

private:
    static T *_Closed() noexcept { return reinterpret_cast<T *>(uintptr_t(1)); }
    std::atomic<T *> head_{nullptr};
};

}  // namespace usdGen
