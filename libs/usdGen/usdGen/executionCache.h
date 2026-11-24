// Backend-neutral immutable input versions and execution cache identity.
//
// These are value contracts only.  They deliberately contain no native
// allocation, queue, event, or backend handle, so a cache entry can be
// compared before a device is acquired and shared by CUDA, Metal, and Vulkan
// adapters.
#ifndef USDGEN_EXECUTION_CACHE_H
#define USDGEN_EXECUTION_CACHE_H

#include "usdGen/deviceGeneration.h"
#include "usdGen/graphDesc.h"
#include "usdGen/generationStore.h"

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <functional>
#include <list>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

struct UsdGenExecutionInputGeneration {
    SdfPath identity;
    uint64_t generation = 0;

    bool operator==(UsdGenExecutionInputGeneration const& rhs) const noexcept {
        return identity == rhs.identity && generation == rhs.generation;
    }
    bool operator!=(UsdGenExecutionInputGeneration const& rhs) const noexcept {
        return !(*this == rhs);
    }
};

/// Canonical, immutable-after-publication input version tuple.  Source and
/// reference entries are curve-set identities; map and surface entries are
/// their corresponding descriptor identities.  Relationship-only references
/// with no payload generation are retained at generation zero so a changed
/// target still changes the plan/cache identity.
struct UsdGenExecutionInputVersions {
    std::vector<UsdGenExecutionInputGeneration> sources;
    std::vector<UsdGenExecutionInputGeneration> references;
    std::vector<UsdGenExecutionInputGeneration> maps;
    std::vector<UsdGenExecutionInputGeneration> surfaces;
    // False means two descriptor records named the same identity with
    // contradictory nonzero generations. Such a tuple must not be admitted
    // to an execution cache, even though it remains inspectable for
    // diagnostics.
    bool valid = true;

    static UsdGenExecutionInputVersions FromDescription(
        UsdGenGraphDesc const& desc);

    bool operator==(UsdGenExecutionInputVersions const& rhs) const noexcept {
        return valid == rhs.valid && sources == rhs.sources &&
            references == rhs.references && maps == rhs.maps &&
            surfaces == rhs.surfaces;
    }
    bool operator!=(UsdGenExecutionInputVersions const& rhs) const noexcept {
        return !(*this == rhs);
    }

    /// Stable process-independent FNV-1a digest over canonical path and
    /// generation terms.  Equality remains authoritative; this is for hash
    /// tables and cheap cache prefiltering only.
    uint64_t Hash() const noexcept;
    bool IsValid() const noexcept { return valid; }

    // Dirty routing updates are intentionally explicit and return whether an
    // existing identity was found. Unknown targets can still be represented
    // by callers by appending a generation-zero entry before publication.
    bool UpdateCurve(SdfPath const& identity, uint64_t generation) noexcept;
    bool UpdateMap(SdfPath const& identity, uint64_t generation) noexcept;
    bool UpdateSurface(SdfPath const& identity, uint64_t generation) noexcept;
};

/// Backend/device/capability context is deliberately separate from the
/// semantic input tuple. A CPU cache entry can therefore be compared without
/// pretending it is portable across incompatible device contexts.
struct UsdGenExecutionContext {
    UsdGenDeviceBackend backend = UsdGenDeviceBackend::Unknown;
    uint32_t capabilityVersion = 0;
    int32_t deviceIndex = -1;
    // Compatibility/context epoch for the selected device runtime. This is
    // not the published geometry generation and should only change when the
    // adapter's compatible execution context changes.
    uint64_t deviceGeneration = 0;
    // Interactive and render evaluation may select different kernels,
    // defaults, or quality policy even on the same backend/device.
    UsdGenContext evaluationContext = UsdGenContext::Interactive;

    bool operator==(UsdGenExecutionContext const& rhs) const noexcept {
        return backend == rhs.backend &&
            capabilityVersion == rhs.capabilityVersion &&
            deviceIndex == rhs.deviceIndex &&
            deviceGeneration == rhs.deviceGeneration &&
            evaluationContext == rhs.evaluationContext;
    }
    bool operator!=(UsdGenExecutionContext const& rhs) const noexcept {
        return !(*this == rhs);
    }
};

/// Immutable cache identity.  `frameBits` preserves exact floating-point
/// sample identity, including signed zero and NaN payloads, without making
/// floating-point equality part of the contract.
struct UsdGenExecutionCacheKey {
    SdfPath description;
    UsdGenEpoch planDigest{};
    uint64_t layoutDigest = 0;
    uint64_t frameBits = 0;
    UsdGenExecutionContext context;
    UsdGenExecutionInputVersions inputs;

    static UsdGenExecutionCacheKey Make(
        SdfPath description, UsdGenEpoch planDigest,
        UsdGenExecutionInputVersions inputs,
        UsdGenExecutionContext context,
        uint64_t layoutDigest, double frame) noexcept;

    bool operator==(UsdGenExecutionCacheKey const& rhs) const noexcept {
        return description == rhs.description && planDigest == rhs.planDigest &&
            layoutDigest == rhs.layoutDigest && frameBits == rhs.frameBits &&
            context == rhs.context && inputs == rhs.inputs;
    }
    bool operator!=(UsdGenExecutionCacheKey const& rhs) const noexcept {
        return !(*this == rhs);
    }
    bool IsValid() const noexcept {
        return context.backend != UsdGenDeviceBackend::Unknown && inputs.IsValid();
    }
    uint64_t Hash() const noexcept;
};

struct UsdGenExecutionCacheKeyHasher {
    size_t operator()(UsdGenExecutionCacheKey const& key) const noexcept {
        return static_cast<size_t>(key.Hash());
    }
};

class UsdGenExecutionCacheDomain;

/// Thread-safe immutable-snapshot cache for backend execution artifacts.
///
/// The current state is copy-on-write: every mutation builds a complete next
/// state and publishes it only after admission and all allocations succeed.
/// A Lease retains its immutable snapshot independently of the store, while
/// LRU eviction skips actively leased entries. Hasher is injectable so tests
/// and adapters can exercise hash collisions; key equality is always the
/// authoritative match after hashing.
template<class Payload, class Hasher = UsdGenExecutionCacheKeyHasher>
class UsdGenExecutionCacheStore {
private:
    struct State;

public:
    struct Snapshot {
        Snapshot(UsdGenExecutionCacheKey cacheKey,
                 std::shared_ptr<const Payload> cachePayload,
                 size_t byteCharge)
            : key(std::move(cacheKey)), payload(std::move(cachePayload)),
              bytes(byteCharge) {}

        UsdGenExecutionCacheKey key;
        std::shared_ptr<const Payload> payload;
        size_t bytes = 0;
        // State generations also share Snapshot objects, so shared_ptr's
        // reference count cannot distinguish a caller lease from a retired
        // COW state. Track only externally-visible leases explicitly.
        mutable std::atomic<size_t> activeLeases{0};
    };

    /// Holds references displaced by one store transaction. Domain wrappers
    /// use this to extend their destruction beyond an enclosing serialization
    /// lock; native payload destructors may perform backend work or re-enter
    /// cache inspection. The type is intentionally opaque to callers.
    class DeferredRetirement {
    private:
        DeferredRetirement() = default;
        DeferredRetirement(DeferredRetirement const&) = delete;
        DeferredRetirement& operator=(DeferredRetirement const&) = delete;
        DeferredRetirement(DeferredRetirement&&) noexcept = default;
        DeferredRetirement& operator=(DeferredRetirement&&) noexcept = default;

        std::shared_ptr<const State> state_;
        std::shared_ptr<const Snapshot> candidate_;
        friend class UsdGenExecutionCacheStore;
        friend class UsdGenExecutionCacheDomain;
    };

    class Lease {
    public:
        Lease() = default;
        Lease(Lease const& rhs) : snapshot_(rhs.snapshot_) { _Acquire(); }
        Lease(Lease&&) noexcept = default;
        Lease& operator=(Lease const& rhs) {
            if (this != &rhs) {
                Lease copy(rhs);
                swap(copy);
            }
            return *this;
        }
        Lease& operator=(Lease&& rhs) noexcept {
            if (this != &rhs) {
                _Release();
                snapshot_ = std::move(rhs.snapshot_);
            }
            return *this;
        }
        ~Lease() { _Release(); }
        explicit operator bool() const noexcept { return bool(snapshot_); }
        UsdGenExecutionCacheKey const& Key() const noexcept {
            return snapshot_->key;
        }
        std::shared_ptr<const Payload> SharedPayload() const noexcept {
            return snapshot_ ? snapshot_->payload : nullptr;
        }
        Payload const* operator->() const noexcept {
            return snapshot_ ? snapshot_->payload.get() : nullptr;
        }
        Payload const& operator*() const { return *snapshot_->payload; }
        size_t Bytes() const noexcept {
            return snapshot_ ? snapshot_->bytes : 0;
        }

    private:
        explicit Lease(std::shared_ptr<const Snapshot> snapshot)
            : snapshot_(std::move(snapshot)) { _Acquire(); }
        void _Acquire() noexcept {
            if (snapshot_)
                snapshot_->activeLeases.fetch_add(1, std::memory_order_relaxed);
        }
        void _Release() noexcept {
            if (snapshot_)
                snapshot_->activeLeases.fetch_sub(1, std::memory_order_relaxed);
        }
        void swap(Lease& rhs) noexcept { snapshot_.swap(rhs.snapshot_); }
        std::shared_ptr<const Snapshot> snapshot_;
        friend class UsdGenExecutionCacheStore;
    };

    explicit UsdGenExecutionCacheStore(size_t maxBytes, Hasher hasher = {})
        : maxBytes_(maxBytes), state_(std::make_shared<const State>(hasher)) {}

    UsdGenExecutionCacheStore(UsdGenExecutionCacheStore const&) = delete;
    UsdGenExecutionCacheStore& operator=(UsdGenExecutionCacheStore const&) = delete;

    /// Returns a lease and promotes the key to MRU. A missing key is a normal
    /// miss. Failure to allocate the MRU copy leaves the old state intact but
    /// still returns the valid snapshot lease.
    std::optional<Lease> Lookup(
        UsdGenExecutionCacheKey const& key,
        DeferredRetirement* retirement = nullptr);

    /// Atomically inserts a new key. An exact duplicate with the same byte
    /// charge coalesces to the already-published immutable snapshot; a byte
    /// mismatch is rejected as an identity/accounting contradiction. If
    /// admission fails, published state and accounting remain unchanged.
    bool Insert(UsdGenExecutionCacheKey key,
                std::shared_ptr<const Payload> payload,
                size_t bytes, std::string* reason = nullptr,
                DeferredRetirement* retirement = nullptr);

    /// Drops all resident entries as one immutable-state publication. Existing
    /// caller leases remain valid, and payload destruction occurs after the
    /// cache lock is released. Used when a backend context epoch is replaced.
    bool Clear(DeferredRetirement* retirement = nullptr) noexcept;

    size_t Size() const noexcept;
    size_t Bytes() const noexcept;
    size_t MaxBytes() const noexcept { return maxBytes_; }

private:
    struct Record {
        std::shared_ptr<const Snapshot> snapshot;
    };
    struct State {
        explicit State(Hasher const& hasher) : entries(0, hasher) {}
        std::unordered_map<UsdGenExecutionCacheKey, Record, Hasher> entries;
        std::list<UsdGenExecutionCacheKey> lru; // front = most recently used
        size_t bytes = 0;
    };

    static void Touch(State* state, UsdGenExecutionCacheKey const& key);
    static bool EvictOne(State* state);

    size_t maxBytes_ = 0;
    mutable std::mutex mutex_;
    std::shared_ptr<const State> state_;
};

/// Shared completed-result cache domain.  The domain is the only state that
/// may cross SessionCooker ownership boundaries: graph/compiler/workspace
/// state remains private to each cooker.  A context epoch makes an external
/// backend reset a logical miss immediately; Clear() then retires the old COW
/// state without destroying active leases.
struct UsdGenExecutionCacheDomainKey {
    UsdGenDeviceBackend backend = UsdGenDeviceBackend::Unknown;
    int32_t deviceIndex = -1;
    // Stable adapter/context identity. CUDA's current primary-context path
    // uses zero; Metal/Vulkan adapters must provide their logical-device or
    // context identity when they create a shared domain.
    uint64_t contextIdentity = 0;

    bool operator==(UsdGenExecutionCacheDomainKey const& rhs) const noexcept {
        return backend == rhs.backend && deviceIndex == rhs.deviceIndex &&
            contextIdentity == rhs.contextIdentity;
    }
    bool operator!=(UsdGenExecutionCacheDomainKey const& rhs) const noexcept {
        return !(*this == rhs);
    }
};

struct UsdGenExecutionCacheDomainKeyHasher {
    size_t operator()(UsdGenExecutionCacheDomainKey const& key) const noexcept {
        size_t result = static_cast<size_t>(key.backend);
        result ^= std::hash<int32_t>{}(key.deviceIndex) << 1;
        result ^= std::hash<uint64_t>{}(key.contextIdentity) << 2;
        return result;
    }
};

class UsdGenExecutionCacheDomain
    : public std::enable_shared_from_this<UsdGenExecutionCacheDomain> {
public:
    using Store = UsdGenExecutionCacheStore<UsdGenGeneration>;
    using Lease = Store::Lease;

    /// Terminal state delivered to a nonblocking exact-key execution flight.
    /// Resident is used only when a caller races a cache hit between its
    /// ordinary lookup and BeginCoalesced; it does not count as coalescing.
    enum class CoalescedStatus {
        Success,
        Resident,
        Failed,
        Cancelled,
        Stale
    };

    using CoalescedCallback =
        std::function<void(CoalescedStatus, std::optional<Lease>)>;

    /// A registration is a move-free value naming one waiter in one exact
    /// cache-key flight. The leader alone may resolve the flight; a follower
    /// may cancel its own callback without disturbing the leader or siblings.
    struct CoalescedRegistration {
        UsdGenExecutionCacheKey key;
        uint64_t flight = 0;
        uint64_t waiter = 0;
        bool leader = false;

        explicit operator bool() const noexcept {
            return flight != 0 && waiter != 0;
        }
        bool IsLeader() const noexcept { return leader; }
    };

    // A publication fence is a linearization point shared by cache lookup,
    // execution completion, and context/domain invalidation.  It intentionally
    // carries no native resource: it only proves that a caller's exact cache
    // context was current when it asked to publish.  PublishIfCurrent holds
    // the domain serialization lock while it validates and runs the tiny
    // local-publication action, so Invalidate cannot make that action become
    // visible after the invalidation's epoch bump.
    class PublicationFence {
    public:
        PublicationFence() = default;
        uint64_t Epoch() const noexcept { return _epoch; }
        explicit operator bool() const noexcept { return _domain != nullptr; }

    private:
        friend class UsdGenExecutionCacheDomain;
        PublicationFence(UsdGenExecutionCacheDomain const* domain,
                         uint64_t epoch) noexcept
            : _domain(domain), _epoch(epoch) {}
        UsdGenExecutionCacheDomain const* _domain = nullptr;
        uint64_t _epoch = 0;
    };

    UsdGenExecutionCacheDomain(UsdGenExecutionCacheDomainKey key,
                               size_t maxBytes)
        : key_(key), store_(maxBytes) {}

    UsdGenExecutionCacheDomain(UsdGenExecutionCacheDomain const&) = delete;
    UsdGenExecutionCacheDomain& operator=(UsdGenExecutionCacheDomain const&) = delete;

    UsdGenExecutionCacheDomainKey const& Key() const noexcept { return key_; }
    size_t MaxBytes() const noexcept { return store_.MaxBytes(); }
    uint64_t Epoch() const noexcept { return epoch_.load(std::memory_order_acquire); }
    /// Reserve a device-generation identity which is at least the caller's
    /// next private publication id and newer than every device generation
    /// previously admitted to this shared domain. Zero means the uint64
    /// identity space is exhausted.
    uint64_t AllocatePublicationGeneration(uint64_t minimum) noexcept {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_publicationWatermark == std::numeric_limits<uint64_t>::max())
            return 0;
        uint64_t const afterWatermark = _publicationWatermark + 1;
        _publicationWatermark = std::max(afterWatermark, minimum);
        // Device publication zero is valid for an executed first generation,
        // but this allocator is used only for a cache republish and must
        // return a strictly newer identity.
        if (_publicationWatermark == 0) _publicationWatermark = 1;
        return _publicationWatermark;
    }
    size_t Size() const noexcept { return store_.Size(); }
    size_t Bytes() const noexcept { return store_.Bytes(); }

    /// Register a nonblocking exact-key execution flight. Callers should do
    /// their ordinary Lookup first. A racing resident result is delivered to
    /// `callback` as Resident and returns no registration. Otherwise the
    /// first registration is leader and later registrations become followers.
    /// The callback is always invoked outside the domain mutex.
    std::optional<CoalescedRegistration> BeginCoalesced(
        UsdGenExecutionCacheKey const& key, CoalescedCallback callback,
        bool notifyLeader = true) {
        if (!callback || !key.IsValid()) return std::nullopt;
        std::optional<Lease> resident;
        std::optional<CoalescedRegistration> registration;
        CoalescedCallback immediate;
        Store::DeferredRetirement retirement;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_clearFailed.load(std::memory_order_acquire) ||
                key.context.backend != key_.backend ||
                key.context.deviceIndex != key_.deviceIndex ||
                key.context.deviceGeneration !=
                    epoch_.load(std::memory_order_acquire))
                return std::nullopt;
            resident = store_.Lookup(key, &retirement);
            if (resident) {
                immediate = std::move(callback);
            } else {
                auto found = _inFlight.find(key);
                try {
                    if (found == _inFlight.end()) {
                        if (_nextFlight == std::numeric_limits<uint64_t>::max() ||
                            _nextWaiter == std::numeric_limits<uint64_t>::max())
                            return std::nullopt;
                        uint64_t const flight = ++_nextFlight;
                        uint64_t const waiter = ++_nextWaiter;
                        InFlight state;
                        state.epoch = epoch_.load(std::memory_order_relaxed);
                        state.leaderWaiter = waiter;
                        state.notifyLeader = notifyLeader;
                        state.waiters.emplace(waiter, std::move(callback));
                        auto inserted = _inFlight.emplace(key, std::move(state));
                        if (!inserted.second) return std::nullopt;
                        registration = CoalescedRegistration{
                            key, flight, waiter, true};
                        inserted.first->second.flight = flight;
                    } else {
                        if (found->second.epoch !=
                            epoch_.load(std::memory_order_relaxed) ||
                            _nextWaiter == std::numeric_limits<uint64_t>::max())
                            return std::nullopt;
                        uint64_t const waiter = ++_nextWaiter;
                        found->second.waiters.emplace(waiter, std::move(callback));
                        ++_coalesced;
                        registration = CoalescedRegistration{
                            key, found->second.flight, waiter, false};
                    }
                } catch (...) {
                    return std::nullopt;
                }
            }
        }
        if (resident && immediate) {
            try { immediate(CoalescedStatus::Resident, std::move(resident)); }
            catch (...) {}
        }
        return registration;
    }

    /// Resolve the leader's exact-key flight. The payload is inserted into
    /// the immutable COW store once, then every waiter receives its own Lease
    /// over that one immutable Snapshot/payload. No callback runs under the
    /// domain mutex. A false result means stale/invalid registration or failed
    /// cache admission; callbacks still receive the corresponding terminal
    /// status.
    bool ResolveCoalesced(CoalescedRegistration const& registration,
                          std::shared_ptr<const UsdGenGeneration> payload,
                          size_t bytes, std::string* reason = nullptr) {
        std::vector<std::pair<CoalescedCallback, std::optional<Lease>>> notify;
        CoalescedStatus status = CoalescedStatus::Failed;
        bool resolved = false;
        Store::DeferredRetirement retirement;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            auto found = _FindFlight(registration);
            if (found == _inFlight.end() || !registration.leader) return false;
            auto state = std::move(found->second);
            _inFlight.erase(found);
            if (state.epoch != epoch_.load(std::memory_order_acquire) ||
                _clearFailed.load(std::memory_order_acquire)) {
                status = CoalescedStatus::Stale;
                if (reason) *reason = "execution cache coalesced flight is stale";
            } else {
                bool const inserted = store_.Insert(
                    registration.key, std::move(payload), bytes, reason,
                    &retirement);
                if (inserted) {
                    auto lease = store_.Lookup(registration.key, &retirement);
                    if (lease) {
                        status = CoalescedStatus::Success;
                        resolved = true;
                        for (auto& waiter : state.waiters) {
                            if (waiter.first == state.leaderWaiter &&
                                !state.notifyLeader) continue;
                            notify.emplace_back(waiter.second, *lease);
                        }
                    } else if (reason && reason->empty()) {
                        *reason = "execution cache coalesced result lookup failed";
                    }
                }
            }
            if (notify.empty())
                for (auto& waiter : state.waiters) {
                    if (waiter.first == state.leaderWaiter &&
                        !state.notifyLeader) continue;
                    notify.emplace_back(waiter.second, std::nullopt);
                }
        }
        _Notify(std::move(notify), status);
        return resolved;
    }

    /// Complete a leader's flight without inserting a result. Failure and
    /// cancellation are intentionally explicit so waiters cannot mistake a
    /// missing result for a cache hit.
    bool RejectCoalesced(CoalescedRegistration const& registration,
                         CoalescedStatus status = CoalescedStatus::Failed,
                         std::string* reason = nullptr) {
        if (status == CoalescedStatus::Success ||
            status == CoalescedStatus::Resident)
            return false;
        std::vector<std::pair<CoalescedCallback, std::optional<Lease>>> notify;
        bool accepted = false;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            auto found = _FindFlight(registration);
            if (found == _inFlight.end() || !registration.leader) return false;
            auto state = std::move(found->second);
            _inFlight.erase(found);
            accepted = state.epoch == epoch_.load(std::memory_order_acquire) &&
                !_clearFailed.load(std::memory_order_acquire);
            if (!accepted) status = CoalescedStatus::Stale;
            if (reason && !accepted)
                *reason = "execution cache coalesced flight is stale";
            for (auto& waiter : state.waiters) {
                if (waiter.first == state.leaderWaiter &&
                    !state.notifyLeader) continue;
                notify.emplace_back(waiter.second, std::nullopt);
            }
        }
        _Notify(std::move(notify), status);
        return accepted;
    }

    bool CancelCoalesced(CoalescedRegistration const& registration,
                         std::string* reason = nullptr) {
        if (registration.leader)
            return RejectCoalesced(registration, CoalescedStatus::Cancelled,
                                   reason);
        std::lock_guard<std::mutex> lock(_mutex);
        auto found = _FindFlight(registration);
        if (found == _inFlight.end()) return false;
        auto waiter = found->second.waiters.find(registration.waiter);
        if (waiter == found->second.waiters.end()) return false;
        found->second.waiters.erase(waiter);
        return true;
    }

    /// Withdraw a follower while still settling an owner-bound asynchronous
    /// continuation. Unlike CancelCoalesced, this retains the callback long
    /// enough to notify it, so a parked pipeline work item can release its
    /// graph wait during shutdown or supersession.
    bool CancelCoalescedAndNotify(CoalescedRegistration const& registration,
                                  std::string* reason = nullptr) {
        if (registration.leader)
            return RejectCoalesced(registration, CoalescedStatus::Cancelled,
                                   reason);
        CoalescedCallback callback;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            auto found = _FindFlight(registration);
            if (found == _inFlight.end()) return false;
            auto waiter = found->second.waiters.find(registration.waiter);
            if (waiter == found->second.waiters.end()) return false;
            callback = std::move(waiter->second);
            found->second.waiters.erase(waiter);
        }
        if (callback) {
            try { callback(CoalescedStatus::Cancelled, std::nullopt); }
            catch (...) {}
        }
        return true;
    }

    /// Number of followers which joined an existing exact-key flight. This is
    /// a deterministic neutral seam for Session statistics; it does not count
    /// resident hits or duplicate completed-cache insertions.
    uint64_t CoalescedCount() const noexcept {
        std::lock_guard<std::mutex> lock(_mutex);
        return _coalesced;
    }

    /// Capture a fence for `key` only if its exact backend/device/epoch is
    /// current.  A caller keeps normal shared ownership of the domain while
    /// retaining the fence; the fence itself deliberately does not extend the
    /// domain lifetime.
    std::optional<PublicationFence> CapturePublicationFence(
        UsdGenExecutionCacheKey const& key) const noexcept {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_clearFailed.load(std::memory_order_acquire) ||
            key.context.backend != key_.backend ||
            key.context.deviceIndex != key_.deviceIndex ||
            key.context.deviceGeneration != epoch_.load(std::memory_order_acquire))
            return std::nullopt;
        return PublicationFence(this, key.context.deviceGeneration);
    }

    /// Run an allocation-free, non-blocking local publication exchange iff
    /// `fence` still names this domain's epoch. The action must return its
    /// displaced owner (or another retirement token); that value is destroyed
    /// only after the domain mutex is released. Do not run GPU work, allocate,
    /// wait, or re-enter this domain from `publish`.
    template<class Publish>
    bool PublishIfCurrent(PublicationFence const& fence, Publish&& publish) {
        using Retirement = std::invoke_result_t<Publish>;
        static_assert(!std::is_void_v<Retirement>,
            "publication must return displaced ownership for retirement");
        std::optional<Retirement> retirement;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (fence._domain != this ||
                _clearFailed.load(std::memory_order_acquire) ||
                fence._epoch != epoch_.load(std::memory_order_acquire))
                return false;
            retirement.emplace(std::invoke(std::forward<Publish>(publish)));
        }
        return true;
    }

    std::optional<Lease> Lookup(UsdGenExecutionCacheKey const& key) {
        Store::DeferredRetirement retirement;
        std::optional<Lease> result;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_clearFailed.load(std::memory_order_acquire)) return std::nullopt;
            if (key.context.backend != key_.backend ||
                key.context.deviceIndex != key_.deviceIndex ||
                key.context.deviceGeneration != Epoch()) return std::nullopt;
            result = store_.Lookup(key, &retirement);
        }
        return result;
    }

    bool Insert(UsdGenExecutionCacheKey key,
                std::shared_ptr<const UsdGenGeneration> payload,
                size_t bytes, std::string* reason = nullptr) {
        Store::DeferredRetirement retirement;
        bool inserted = false;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_clearFailed.load(std::memory_order_acquire)) {
                if (reason) *reason = "execution cache domain cleanup is pending";
                return false;
            }
            if (key.context.backend != key_.backend ||
                key.context.deviceIndex != key_.deviceIndex ||
                key.context.deviceGeneration != Epoch()) {
                if (reason) *reason = "execution cache domain epoch or identity is stale";
                return false;
            }
            uint64_t const generation = payload && payload->device
                ? payload->device->Identity().generation : 0;
            inserted = store_.Insert(std::move(key), std::move(payload), bytes,
                                     reason, &retirement);
            if (inserted && generation > _publicationWatermark)
                _publicationWatermark = generation;
        }
        return inserted;
    }

    /// Bumps the logical epoch before clearing. Domain serialization prevents
    /// stale insertion from racing the clear; a failed clear leaves the
    /// domain fail-closed until a later Invalidate retry succeeds.
    bool Invalidate() noexcept {
        std::unordered_map<UsdGenExecutionCacheKey, InFlight,
            UsdGenExecutionCacheKeyHasher> stale;
        Store::DeferredRetirement retirement;
        bool cleared = false;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_clearFailed.load(std::memory_order_acquire)) {
                cleared = store_.Clear(&retirement);
                if (cleared)
                    _clearFailed.store(false, std::memory_order_release);
            } else {
                uint64_t value = epoch_.load(std::memory_order_relaxed);
                for (;;) {
                    if (value == std::numeric_limits<uint64_t>::max()) return false;
                    if (epoch_.compare_exchange_weak(value, value + 1,
                            std::memory_order_acq_rel, std::memory_order_relaxed)) break;
                }
                stale.swap(_inFlight);
                cleared = store_.Clear(&retirement);
                if (!cleared) _clearFailed.store(true, std::memory_order_release);
            }
        }
        for (auto& flight : stale)
            for (auto& waiter : flight.second.waiters) {
                if (waiter.first == flight.second.leaderWaiter &&
                    !flight.second.notifyLeader) continue;
                try { waiter.second(CoalescedStatus::Stale, std::nullopt); }
                catch (...) {}
            }
        return cleared;
    }

    /// Obtain a process-local shared domain for an explicitly identical
    /// backend/device/context and byte budget. Weak registry entries allow the
    /// domain (and its native owners) to retire when the last Session goes
    /// away; no process-global cache payload is leaked by the registry.
    static std::shared_ptr<UsdGenExecutionCacheDomain> AcquireShared(
        UsdGenExecutionCacheDomainKey key, size_t maxBytes) {
        if (key.backend == UsdGenDeviceBackend::Unknown || maxBytes == 0)
            return {};
        // One registry per process: an inline function-local static would
        // give usdGen.dll and every caller module its own copy on Windows.
        SharedRegistry* registry = _SharedRegistry();
        std::lock_guard<std::mutex> lock(registry->mutex);
        auto found = registry->domains.find(key);
        if (found != registry->domains.end()) {
            if (auto domain = found->second.lock())
                return domain->MaxBytes() == maxBytes ? domain : nullptr;
            registry->domains.erase(found);
        }
        auto domain = std::make_shared<UsdGenExecutionCacheDomain>(key, maxBytes);
        registry->domains.emplace(key, domain);
        return domain;
    }

    struct SharedRegistry {
        std::mutex mutex;
        std::unordered_map<UsdGenExecutionCacheDomainKey,
            std::weak_ptr<UsdGenExecutionCacheDomain>,
            UsdGenExecutionCacheDomainKeyHasher> domains;
    };
    static SharedRegistry* _SharedRegistry() noexcept;

private:
    struct InFlight {
        uint64_t flight = 0;
        uint64_t epoch = 0;
        uint64_t leaderWaiter = 0;
        bool notifyLeader = true;
        std::unordered_map<uint64_t, CoalescedCallback> waiters;
    };

    using InFlightMap = std::unordered_map<UsdGenExecutionCacheKey, InFlight,
        UsdGenExecutionCacheKeyHasher>;

    InFlightMap::iterator _FindFlight(
        CoalescedRegistration const& registration) {
        if (!registration || (!registration.leader && registration.flight == 0))
            return _inFlight.end();
        auto found = _inFlight.find(registration.key);
        if (found == _inFlight.end() || found->second.flight != registration.flight)
            return _inFlight.end();
        return found;
    }

    static void _Notify(
        std::vector<std::pair<CoalescedCallback, std::optional<Lease>>> notify,
        CoalescedStatus status) noexcept {
        for (auto& item : notify) {
            try { item.first(status, std::move(item.second)); }
            catch (...) {}
        }
    }

    UsdGenExecutionCacheDomainKey key_;
    mutable std::mutex _mutex;
    std::atomic<uint64_t> epoch_{0};
    std::atomic<bool> _clearFailed{false};
    uint64_t _publicationWatermark = 0;
    uint64_t _nextFlight = 0;
    uint64_t _nextWaiter = 0;
    uint64_t _coalesced = 0;
    InFlightMap _inFlight;
    Store store_;
};

template<class Payload, class Hasher>
void UsdGenExecutionCacheStore<Payload, Hasher>::Touch(
    State* state, UsdGenExecutionCacheKey const& key)
{
    auto found = std::find(state->lru.begin(), state->lru.end(), key);
    if (found != state->lru.end())
        state->lru.splice(state->lru.begin(), state->lru, found);
}

template<class Payload, class Hasher>
bool UsdGenExecutionCacheStore<Payload, Hasher>::EvictOne(State* state)
{
    // Retired COW states may outlive the cache lock while payload destructors
    // run. Their shared ownership is not a caller lease and must not pin an
    // entry, so eviction consults the explicit external-lease count.
    for (auto reverse = state->lru.rbegin(); reverse != state->lru.rend();
         ++reverse) {
        auto found = state->entries.find(*reverse);
        if (found == state->entries.end()) continue;
        if (found->second.snapshot->activeLeases.load(
                std::memory_order_relaxed) != 0) continue;
        size_t const bytes = found->second.snapshot->bytes;
        auto listIt = std::prev(reverse.base());
        state->entries.erase(found);
        state->lru.erase(listIt);
        state->bytes -= bytes;
        return true;
    }
    return false;
}

template<class Payload, class Hasher>
std::optional<typename UsdGenExecutionCacheStore<Payload, Hasher>::Lease>
UsdGenExecutionCacheStore<Payload, Hasher>::Lookup(
    UsdGenExecutionCacheKey const& key,
    DeferredRetirement* retirement)
{
    if (!key.IsValid()) return std::nullopt;
    Lease lease;
    // Keep the retired immutable state alive until after unlocking. A payload
    // destructor is user code and may re-enter cache inspection.
    std::shared_ptr<const State> retired;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        std::shared_ptr<const State> current = state_;
        auto found = current->entries.find(key);
        if (found == current->entries.end()) return std::nullopt;
        lease = Lease(found->second.snapshot);
        State candidate(*current);
        Touch(&candidate, key);
        auto next = std::make_shared<const State>(std::move(candidate));
        retired = std::move(state_);
        state_ = std::move(next);
    } catch (...) {
        // LRU promotion is an optimization. The immutable hit remains valid
        // even when a transient allocation failure prevents promotion.
    }
    if (retirement) retirement->state_ = std::move(retired);
    else retired.reset();
    if (!lease) return std::nullopt;
    return lease;
}

template<class Payload, class Hasher>
bool UsdGenExecutionCacheStore<Payload, Hasher>::Insert(
    UsdGenExecutionCacheKey key, std::shared_ptr<const Payload> payload,
    size_t bytes, std::string* reason, DeferredRetirement* retirement)
{
    auto fail = [reason](char const* message) {
        if (reason) *reason = message;
        return false;
    };
    if (!key.IsValid()) return fail("execution cache key is invalid");
    if (!payload) return fail("execution cache payload is null");
    if (bytes == 0) return fail("execution cache byte charge is zero");
    if (maxBytes_ == 0 || bytes > maxBytes_)
        return fail("execution cache entry exceeds its byte budget");

    std::shared_ptr<const State> retired;
    try {
        auto snapshot = std::make_shared<const Snapshot>(
            std::move(key), std::move(payload), bytes);
        // Keep a transaction-local candidate alive until the caller-selected
        // retirement boundary. This covers duplicate/rejected candidates and
        // allocation unwinds as well as successfully published snapshots.
        if (retirement) retirement->candidate_ = snapshot;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::shared_ptr<const State> current = state_;
            State candidate(*current);
            auto existing = candidate.entries.find(snapshot->key);
            if (existing != candidate.entries.end()) {
                if (existing->second.snapshot->bytes != bytes)
                    return fail("execution cache duplicate key has a conflicting byte charge");
                Touch(&candidate, snapshot->key);
            } else {
                while (candidate.bytes > maxBytes_ - bytes) {
                    if (!EvictOne(&candidate))
                        return fail("execution cache admission blocked by active lease");
                }
                candidate.lru.push_front(snapshot->key);
                auto inserted = candidate.entries.emplace(
                    snapshot->key, Record{snapshot});
                if (!inserted.second)
                    return fail("execution cache duplicate insertion failed");
                candidate.bytes += bytes;
            }
            // Only this final publication changes the live store. Retain the
            // former state across unlock so evicted payload destruction can
            // safely re-enter this cache.
            auto next = std::make_shared<const State>(std::move(candidate));
            retired = std::move(state_);
            state_ = std::move(next);
        }
        if (retirement) retirement->state_ = std::move(retired);
        else retired.reset();
        return true;
    } catch (...) {
        return fail("execution cache transaction allocation failed");
    }
}

template<class Payload, class Hasher>
bool UsdGenExecutionCacheStore<Payload, Hasher>::Clear(
    DeferredRetirement* retirement) noexcept
{
    std::shared_ptr<const State> retired;
    try {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto next = std::make_shared<const State>(
                state_->entries.hash_function());
            retired = std::move(state_);
            state_ = std::move(next);
        }
        if (retirement) retirement->state_ = std::move(retired);
        else retired.reset();
        return true;
    } catch (...) {
        return false;
    }
}

template<class Payload, class Hasher>
size_t UsdGenExecutionCacheStore<Payload, Hasher>::Size() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return state_->entries.size();
}

template<class Payload, class Hasher>
size_t UsdGenExecutionCacheStore<Payload, Hasher>::Bytes() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return state_->bytes;
}

} // namespace usdGen

#endif // USDGEN_EXECUTION_CACHE_H
