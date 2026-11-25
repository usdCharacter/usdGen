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
#include "usdGen/epoch.h"

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

struct UsdGenGeneration;   // generationStore.h; the Store holds it by shared_ptr

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
    /// The same, refilling *out in place (its vectors keep their capacity).
    static void FromDescription(UsdGenGraphDesc const& desc,
                                UsdGenExecutionInputVersions* out);

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
    bool UpdateInput(SdfPath const& identity, uint64_t generation) noexcept;
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

/// Lock-free immutable-snapshot cache for backend execution artifacts.
///
/// The entry index is an immutable State published by atomic pointer; an
/// insert or clear builds the next State and publishes it with one CAS,
/// retrying on contention. Lookups never wait: they read the State under an
/// epoch guard, lock the entry's weak snapshot, take a lease and set its CLOCK
/// reference bit. Eviction is CLOCK over insertion order, skipping referenced
/// and leased entries; the inserter that needs room advances the hand. A lease
/// and an eviction claim exclude each other on the snapshot's lease word, so
/// a leased entry is never evicted and its bytes stay charged. The
/// publisher that removes an entry drops the store's reference before it
/// returns, so an unleased payload dies there; entry and State shells retire
/// through epochs and own nothing.
/// Hasher is injectable so tests and adapters can exercise hash collisions;
/// key equality is always the authoritative match after hashing.
template<class Payload, class Hasher = UsdGenExecutionCacheKeyHasher>
class UsdGenExecutionCacheStore {
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
        // Low bits: inserters claiming the entry for eviction; above them,
        // caller leases (the store's own references do not count). A lease
        // is taken only with no claim, a claim only with no lease; the
        // inserter whose State CAS removes the entry keeps its claim.
        mutable std::atomic<uint64_t> leaseWord{0};
        static constexpr uint64_t kClaimMask = 0xffff;
        static constexpr uint64_t kLease = kClaimMask + 1;

        bool TryLease() const noexcept {
            uint64_t word = leaseWord.load(std::memory_order_relaxed);
            do {
                if (word & kClaimMask) return false;
            } while (!leaseWord.compare_exchange_weak(word, word + kLease,
                         std::memory_order_acquire, std::memory_order_relaxed));
            return true;
        }
        bool TryClaim() const noexcept {
            uint64_t word = leaseWord.load(std::memory_order_relaxed);
            do {
                if (word >= kLease) return false;
            } while (!leaseWord.compare_exchange_weak(word, word + 1,
                         std::memory_order_acq_rel, std::memory_order_relaxed));
            return true;
        }
        void Unclaim() const noexcept {
            leaseWord.fetch_sub(1, std::memory_order_release);
        }
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
        struct Adopt {};
        // Takes over a lease count the caller already added.
        Lease(std::shared_ptr<const Snapshot> snapshot, Adopt) noexcept
            : snapshot_(std::move(snapshot)) {}
        // A copy comes from a held lease, so no claim can be present.
        void _Acquire() noexcept {
            if (snapshot_)
                snapshot_->leaseWord.fetch_add(Snapshot::kLease, std::memory_order_relaxed);
        }
        void _Release() noexcept {
            if (snapshot_) {
                snapshot_->leaseWord.fetch_sub(Snapshot::kLease, std::memory_order_release);
                snapshot_.reset();
            }
        }
        void swap(Lease& rhs) noexcept { snapshot_.swap(rhs.snapshot_); }
        std::shared_ptr<const Snapshot> snapshot_;
        friend class UsdGenExecutionCacheStore;
    };

    explicit UsdGenExecutionCacheStore(size_t maxBytes, Hasher hasher = {})
        : maxBytes_(maxBytes), hasher_(std::move(hasher)), state_(new State) {}
    ~UsdGenExecutionCacheStore();

    UsdGenExecutionCacheStore(UsdGenExecutionCacheStore const&) = delete;
    UsdGenExecutionCacheStore& operator=(UsdGenExecutionCacheStore const&) = delete;

    /// Returns a lease and sets the key's reference bit. A missing key is a
    /// normal miss.
    std::optional<Lease> Lookup(UsdGenExecutionCacheKey const& key) {
        return _Lookup(key, nullptr);
    }

    /// Inserts a new key. An exact duplicate with the same byte charge
    /// coalesces to the already-published immutable snapshot; a byte
    /// mismatch is rejected as an identity/accounting contradiction. If
    /// admission fails, published state and accounting remain unchanged.
    bool Insert(UsdGenExecutionCacheKey key,
                std::shared_ptr<const Payload> payload,
                size_t bytes, std::string* reason = nullptr) {
        return _Insert(std::move(key), std::move(payload), bytes, reason, nullptr,
                       nullptr);
    }

    /// Drops all resident entries as one State publication. Existing caller
    /// leases remain valid; unleased payloads are destroyed before return.
    bool Clear() noexcept { return _Clear(nullptr); }

    size_t Size() const noexcept;
    size_t Bytes() const noexcept;
    size_t MaxBytes() const noexcept { return maxBytes_; }

private:
    friend class UsdGenExecutionCacheDomain;

    // Readers touch only key, hash, weak and referenced; "strong" belongs
    // to the State that lists the entry and is moved out by its remover.
    struct Entry : UsdGenEpochNode {
        Entry(std::shared_ptr<const Snapshot> value, size_t keyHash)
            : key(value->key), hash(keyHash), bytes(value->bytes), weak(value),
              strong(std::move(value)) {}
        UsdGenExecutionCacheKey key;
        size_t hash = 0;
        size_t bytes = 0;
        std::weak_ptr<const Snapshot> weak;
        std::shared_ptr<const Snapshot> strong;
        mutable std::atomic<uint8_t> referenced{0};
    };
    struct State : UsdGenEpochNode {
        std::vector<Entry*> order;  // insertion order: the CLOCK ring
        std::vector<Entry*> table;  // open addressing, power of two or empty
        size_t bytes = 0;
        uint64_t epoch = 0;         // owning domain epoch; 0 standalone
    };

    // A null epoch means "any"; otherwise the published State must carry it.
    std::optional<Lease> _Lookup(UsdGenExecutionCacheKey const& key,
                                 uint64_t const* epoch);
    // A non-null lease receives a lease on the admitted (or resident
    // duplicate) snapshot, taken before it is published.
    bool _Insert(UsdGenExecutionCacheKey key,
                 std::shared_ptr<const Payload> payload, size_t bytes,
                 std::string* reason, uint64_t const* epoch,
                 std::optional<Lease>* lease);
    // Publish an empty State carrying *epoch (or the current epoch). With an
    // epoch, a State already at or past it counts as done (a helper won).
    bool _Clear(uint64_t const* epoch) noexcept;
    uint64_t _StateEpoch() const noexcept {
        UsdGenEpochGuard guard;
        return state_.load(std::memory_order_seq_cst)->epoch;
    }

    static Entry* _Find(State const* state, UsdGenExecutionCacheKey const& key,
                        size_t hash) noexcept {
        size_t const n = state->table.size();
        if (!n) return nullptr;
        for (size_t i = hash & (n - 1);; i = (i + 1) & (n - 1)) {
            Entry* entry = state->table[i];
            if (!entry) return nullptr;
            if (entry->hash == hash && entry->key == key) return entry;
        }
    }
    static void _Index(State* state) {
        size_t n = 0;
        if (!state->order.empty()) {
            n = 4;
            while (n < state->order.size() * 2) n <<= 1;
        }
        state->table.assign(n, nullptr);
        for (Entry* entry : state->order) {
            size_t i = entry->hash & (n - 1);
            while (state->table[i]) i = (i + 1) & (n - 1);
            state->table[i] = entry;
        }
    }
    static void _Retire(Entry* entry) noexcept { UsdGenEpochRetireDelete(entry); }
    static void _Retire(State* state) noexcept { UsdGenEpochRetireDelete(state); }

    size_t maxBytes_ = 0;
    Hasher hasher_;
    std::atomic<size_t> hand_{0};
    std::atomic<State*> state_;
};

/// Shared completed-result cache domain.  The domain is the only state that
/// may cross SessionCooker ownership boundaries: graph/compiler/workspace
/// state remains private to each cooker.  A context epoch makes an external
/// backend reset a logical miss immediately; the store's State carries the
/// epoch, so a stale insert can never land after the clear.
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

    // A publication fence proves that a caller's exact cache context was
    // current when it asked to publish. It carries no native resource.
    // PublishIfCurrent announces itself, checks the epoch and runs the tiny
    // local publication action; Invalidate bumps the epoch and then parks
    // until every announced action has finished. So an action either sees
    // the bump and does nothing, or completes before Invalidate returns: no
    // old-epoch publication becomes visible after Invalidate.
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
    ~UsdGenExecutionCacheDomain();

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
        uint64_t value = _publicationWatermark.load(std::memory_order_acquire);
        for (;;) {
            if (value == std::numeric_limits<uint64_t>::max()) return 0;
            // Device publication zero is valid for an executed first
            // generation, but this allocator is used only for a cache
            // republish and must return a strictly newer identity.
            uint64_t next = std::max(value + 1, minimum);
            if (next == 0) next = 1;
            if (_publicationWatermark.compare_exchange_weak(
                    value, next, std::memory_order_acq_rel, std::memory_order_acquire))
                return next;
        }
    }
    size_t Size() const noexcept { return store_.Size(); }
    size_t Bytes() const noexcept { return store_.Bytes(); }

    /// Register a nonblocking exact-key execution flight. Callers should do
    /// their ordinary Lookup first. A racing resident result is delivered to
    /// "callback" as Resident and returns no registration. Otherwise the
    /// first registration is leader and later registrations become followers
    /// on the flight's continuation stack. Callbacks never run under a guard.
    std::optional<CoalescedRegistration> BeginCoalesced(
        UsdGenExecutionCacheKey const& key, CoalescedCallback callback,
        bool notifyLeader = true);

    /// Resolve the leader's exact-key flight. The payload is inserted into
    /// the store once, then every waiter receives its own Lease over that one
    /// immutable Snapshot/payload. A false result means stale/invalid
    /// registration or failed cache admission; callbacks still receive the
    /// corresponding terminal status.
    bool ResolveCoalesced(CoalescedRegistration const& registration,
                          std::shared_ptr<const UsdGenGeneration> payload,
                          size_t bytes, std::string* reason = nullptr) {
        return _Settle(registration, std::move(payload), bytes,
                       CoalescedStatus::Success, reason);
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
        return _Settle(registration, nullptr, 0, status, reason);
    }

    bool CancelCoalesced(CoalescedRegistration const& registration,
                         std::string* reason = nullptr) {
        if (registration.leader)
            return RejectCoalesced(registration, CoalescedStatus::Cancelled,
                                   reason);
        CoalescedCallback callback;
        return _Withdraw(registration, &callback);
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
        if (!_Withdraw(registration, &callback)) return false;
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
        return _coalesced.load(std::memory_order_acquire);
    }

    /// Capture a fence for "key" only if its exact backend/device/epoch is
    /// current.  A caller keeps normal shared ownership of the domain while
    /// retaining the fence; the fence itself deliberately does not extend the
    /// domain lifetime.
    std::optional<PublicationFence> CapturePublicationFence(
        UsdGenExecutionCacheKey const& key) const noexcept {
        if (!_Current(key)) return std::nullopt;
        return PublicationFence(this, key.context.deviceGeneration);
    }

    /// Run an allocation-free, non-blocking local publication exchange iff
    /// "fence" still names this domain's epoch. The action must return its
    /// displaced owner (or another retirement token); that value is destroyed
    /// after the action returns. Do not run GPU work, allocate or wait in
    /// "publish".
    template<class Publish>
    bool PublishIfCurrent(PublicationFence const& fence, Publish&& publish) {
        using Retirement = std::invoke_result_t<Publish>;
        static_assert(!std::is_void_v<Retirement>,
            "publication must return displaced ownership for retirement");
        if (fence._domain != this) return false;
        struct Call {
            std::remove_reference_t<Publish>* publish;
            std::optional<Retirement> retirement;
        } call{&publish, std::nullopt};
        // The displaced owner dies with "call", after the window has closed.
        return _PublishIfCurrent(fence._epoch, [](void* context) {
            Call& c = *static_cast<Call*>(context);
            c.retirement.emplace(std::invoke(std::forward<Publish>(*c.publish)));
        }, &call);
    }

    std::optional<Lease> Lookup(UsdGenExecutionCacheKey const& key) {
        if (!_Current(key)) return std::nullopt;
        uint64_t const epoch = key.context.deviceGeneration;
        return store_._Lookup(key, &epoch);
    }

    bool Insert(UsdGenExecutionCacheKey key,
                std::shared_ptr<const UsdGenGeneration> payload,
                size_t bytes, std::string* reason = nullptr);

    /// Bumps the logical epoch, fails every outstanding flight as Stale and
    /// clears the store. A failed clear (allocation) leaves the domain
    /// fail-closed until a later Invalidate retry succeeds.
    bool Invalidate() noexcept;

    /// Test seam: points inside the domain where a stress test may inject a
    /// delay. Null disables; never set outside tests.
    enum class TestPoint {
        InvalidateTookFlights,       // Invalidate, before it walks the list
        BeginCoalescedBeforePublish, // BeginCoalesced, before its guard
        SettleWonFlight              // a leader's _Settle, before _Unlist
    };
    using TestHook = void (*)(TestPoint, void*);
    static void SetTestHook(TestHook hook, void* context) noexcept;

    /// Obtain a process-local shared domain for an explicitly identical
    /// backend/device/context and byte budget. Weak registry entries allow the
    /// domain (and its native owners) to retire when the last Session goes
    /// away; no process-global cache payload is leaked by the registry.
    static std::shared_ptr<UsdGenExecutionCacheDomain> AcquireShared(
        UsdGenExecutionCacheDomainKey key, size_t maxBytes);

private:
    struct Waiter : UsdGenEpochNode {
        CoalescedCallback callback;
        uint64_t id = 0;
        std::atomic<uint32_t> claimed{0};  // 1: notified or withdrawn
        Waiter* next = nullptr;
    };
    using WaiterStack = UsdGenContinuationStack<Waiter, &Waiter::next>;
    // A flight is settled exactly once: by its leader or by Invalidate,
    // whichever wins the state CAS. The winner closes the waiter stack.
    struct Flight : UsdGenEpochNode {
        UsdGenExecutionCacheKey key;
        uint64_t id = 0;
        uint64_t epoch = 0;
        uint64_t leaderWaiter = 0;
        bool notifyLeader = true;
        std::atomic<uint32_t> settled{0};
        WaiterStack waiters;
        Flight* settleNext = nullptr;
    };
    // Copy-on-write flight list published by atomic pointer.
    struct FlightList : UsdGenEpochNode {
        std::vector<Flight*> flights;
    };

    bool _Current(UsdGenExecutionCacheKey const& key) const noexcept {
        return !_clearFailed.load(std::memory_order_acquire) &&
            key.context.backend == key_.backend &&
            key.context.deviceIndex == key_.deviceIndex &&
            key.context.deviceGeneration == epoch_.load(std::memory_order_acquire);
    }
    static Flight* _FindFlight(FlightList const* list,
                               UsdGenExecutionCacheKey const& key) noexcept {
        if (list)
            for (Flight* flight : list->flights)
                if (flight->key == key) return flight;
        return nullptr;
    }
    bool _Settle(CoalescedRegistration const& registration,
                 std::shared_ptr<const UsdGenGeneration> payload, size_t bytes,
                 CoalescedStatus status, std::string* reason);
    bool _Withdraw(CoalescedRegistration const& registration,
                   CoalescedCallback* callback) noexcept;
    // True once the flight is off the current list (false: allocation failed).
    bool _Unlist(Flight* flight) noexcept;
    // Close a settled flight and notify every unclaimed waiter.
    static void _Notify(Flight* flight, CoalescedStatus status,
                        std::optional<Lease> const& lease) noexcept;
    // Announce, check the epoch, run(context), withdraw. False: not current.
    bool _PublishIfCurrent(uint64_t epoch, void (*run)(void*), void* context);
    void _EndPublish(uint32_t unit) noexcept;
    // Park until every publication announced before the epoch bump ended.
    void _AwaitPublications(uint64_t epoch) noexcept;

    UsdGenExecutionCacheDomainKey key_;
    std::atomic<uint64_t> epoch_{0};
    std::atomic<bool> _clearFailed{false};
    std::atomic<uint64_t> _publicationWatermark{0};
    std::atomic<uint64_t> _nextFlight{0};
    std::atomic<uint64_t> _nextWaiter{0};
    std::atomic<uint64_t> _coalesced{0};
    std::atomic<FlightList*> _flights{nullptr};
    // Publications in flight, by epoch parity (executionCache.cpp);
    // kPublishWaiter is set by a parked Invalidate.
    static constexpr uint32_t kPublishWaiter = 0x80000000u;
    std::atomic<uint32_t> _publishing{0};
    Store store_;
};

template<class Payload, class Hasher>
UsdGenExecutionCacheStore<Payload, Hasher>::~UsdGenExecutionCacheStore()
{
    // No reader can remain: the owner is destroying the store.
    State* state = state_.load(std::memory_order_acquire);
    for (Entry* entry : state->order) delete entry;
    delete state;
}

template<class Payload, class Hasher>
std::optional<typename UsdGenExecutionCacheStore<Payload, Hasher>::Lease>
UsdGenExecutionCacheStore<Payload, Hasher>::_Lookup(
    UsdGenExecutionCacheKey const& key, uint64_t const* epoch)
{
    if (!key.IsValid()) return std::nullopt;
    size_t const hash = hasher_(key);
    std::shared_ptr<const Snapshot> snapshot;
    {
        UsdGenEpochGuard guard;
        State const* state = state_.load(std::memory_order_seq_cst);
        if (epoch && state->epoch != *epoch) return std::nullopt;
        Entry* entry = _Find(state, key, hash);
        if (!entry) return std::nullopt;
        // Fails when a remover already dropped the last reference, or when
        // an inserter has claimed the entry for eviction.
        snapshot = entry->weak.lock();
        if (!snapshot || !snapshot->TryLease()) return std::nullopt;
        entry->referenced.store(1, std::memory_order_relaxed);
    }
    return Lease(std::move(snapshot), typename Lease::Adopt{});
}

template<class Payload, class Hasher>
bool UsdGenExecutionCacheStore<Payload, Hasher>::_Insert(
    UsdGenExecutionCacheKey key, std::shared_ptr<const Payload> payload,
    size_t bytes, std::string* reason, uint64_t const* epoch,
    std::optional<Lease>* lease)
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

    // Victims hold a claim and a pin on their snapshot until the State CAS
    // settles; a claim keeps lookups from leasing the entry meanwhile.
    struct Victim {
        Entry* entry;
        std::shared_ptr<const Snapshot> pin;
    };
    std::vector<Victim> victims;
    std::vector<std::shared_ptr<const Snapshot>> dropped;  // after the guard
    auto unclaim = [&victims]() noexcept {
        for (Victim& victim : victims) victim.pin->Unclaim();
        victims.clear();
    };
    auto chosen = [&victims](Entry const* entry) noexcept {
        for (Victim const& victim : victims)
            if (victim.entry == entry) return true;
        return false;
    };
    try {
        size_t const hash = hasher_(key);
        auto snapshot = std::make_shared<const Snapshot>(
            std::move(key), std::move(payload), bytes);
        // The caller's lease is taken before publication, so no inserter can
        // claim the new entry before the caller holds it.
        if (lease) snapshot->leaseWord.store(Snapshot::kLease, std::memory_order_relaxed);
        std::shared_ptr<const Snapshot> mine = lease ? snapshot : nullptr;
        std::unique_ptr<Entry> entry(new Entry(std::move(snapshot), hash));
        std::unique_ptr<State> next;
        for (;;) {
            UsdGenEpochGuard guard;
            State* current = state_.load(std::memory_order_seq_cst);
            if (epoch && current->epoch != *epoch)
                return fail("execution cache domain epoch or identity is stale");
            if (Entry* existing = _Find(current, entry->key, hash)) {
                if (existing->bytes != bytes)
                    return fail("execution cache duplicate key has a conflicting byte charge");
                existing->referenced.store(1, std::memory_order_relaxed);
                if (lease) {
                    auto resident = existing->weak.lock();
                    if (resident && resident->TryLease())
                        lease->emplace(Lease(std::move(resident), typename Lease::Adopt{}));
                }
                return true;
            }
            // CLOCK: a referenced entry gets a second chance, a leased or
            // claimed one is skipped. Two turns of the ring bound the sweep.
            size_t total = current->bytes;
            size_t const n = current->order.size();
            victims.reserve(n);  // a claim is never left behind by a throw
            for (size_t step = 0; total > maxBytes_ - bytes && step < 2 * n; ++step) {
                Entry* candidate =
                    current->order[hand_.fetch_add(1, std::memory_order_relaxed) % n];
                if (chosen(candidate)) continue;
                if (candidate->referenced.exchange(0, std::memory_order_relaxed)) continue;
                // The weak reference keeps the control block alive, so the
                // lock is safe even if a racing remover dropped the strong one.
                auto pin = candidate->weak.lock();
                if (!pin || !pin->TryClaim()) continue;
                victims.push_back(Victim{candidate, std::move(pin)});
                total -= candidate->bytes;
            }
            if (total > maxBytes_ - bytes) {
                unclaim();
                return fail("execution cache admission blocked by active lease");
            }
            if (!next) next.reset(new State);
            next->order.clear();
            next->order.reserve(n + 1 - victims.size());
            for (Entry* resident : current->order)
                if (!chosen(resident)) next->order.push_back(resident);
            next->order.push_back(entry.get());
            next->bytes = total + bytes;
            next->epoch = current->epoch;
            _Index(next.get());
            dropped.reserve(2 * victims.size());  // nothing throws after the CAS
            if (!state_.compare_exchange_strong(current, next.get(),
                                                std::memory_order_seq_cst)) {
                unclaim();
                continue;
            }
            next.release();
            entry.release();
            _Retire(current);
            for (Victim& victim : victims) {
                // The claim stays: the removed entry can never be leased.
                dropped.push_back(std::move(victim.entry->strong));
                dropped.push_back(std::move(victim.pin));
                _Retire(victim.entry);
            }
            victims.clear();
            if (lease) lease->emplace(Lease(std::move(mine), typename Lease::Adopt{}));
            break;
        }
    } catch (...) {
        unclaim();
        return fail("execution cache transaction allocation failed");
    }
    // Evicted payloads die here, outside any guard: a destructor may re-enter.
    return true;
}

template<class Payload, class Hasher>
bool UsdGenExecutionCacheStore<Payload, Hasher>::_Clear(uint64_t const* epoch) noexcept
{
    std::vector<std::shared_ptr<const Snapshot>> dropped;  // after the guard
    try {
        std::unique_ptr<State> next(new State);
        UsdGenEpochGuard guard;
        State* current = state_.load(std::memory_order_seq_cst);
        for (;;) {
            if (epoch && current->epoch >= *epoch) return true;  // a helper won
            next->epoch = epoch ? *epoch : current->epoch;
            if (state_.compare_exchange_weak(current, next.get(),
                                             std::memory_order_seq_cst))
                break;
        }
        next.release();
        dropped.reserve(current->order.size());
        for (Entry* entry : current->order) {
            dropped.push_back(std::move(entry->strong));
            _Retire(entry);
        }
        _Retire(current);
    } catch (...) {
        return false;
    }
    return true;
}

template<class Payload, class Hasher>
size_t UsdGenExecutionCacheStore<Payload, Hasher>::Size() const noexcept
{
    UsdGenEpochGuard guard;
    return state_.load(std::memory_order_seq_cst)->order.size();
}

template<class Payload, class Hasher>
size_t UsdGenExecutionCacheStore<Payload, Hasher>::Bytes() const noexcept
{
    UsdGenEpochGuard guard;
    return state_.load(std::memory_order_seq_cst)->bytes;
}

} // namespace usdGen

#endif // USDGEN_EXECUTION_CACHE_H
