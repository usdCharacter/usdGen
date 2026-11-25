#include "usdGen/executionCache.h"
#include "usdGen/exec/generationStore.h"

#include "usdGen/executionRuntimeInternal.h"
#include "usdGen/parkingWord.h"

#include <algorithm>
#include <array>
#include <limits>
#include <new>

namespace usdGen {
namespace {

uint64_t Mix(uint64_t hash, void const* data, size_t size) noexcept
{
    auto const* bytes = static_cast<unsigned char const*>(data);
    for (size_t i = 0; i != size; ++i) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

uint64_t MixU64(uint64_t hash, uint64_t value) noexcept
{
    // Fix the byte order explicitly so cache identities remain stable across
    // little- and big-endian hosts.
    for (unsigned shift = 0; shift != 64; shift += 8) {
        unsigned char const byte =
            static_cast<unsigned char>((value >> shift) & 0xffu);
        hash = Mix(hash, &byte, sizeof(byte));
    }
    return hash;
}

uint64_t MixPath(uint64_t hash, SdfPath const& path) noexcept
{
    std::string const text = path.GetString();
    hash = MixU64(hash, static_cast<uint64_t>(text.size()));
    return Mix(hash, text.data(), text.size());
}

void Canonicalize(std::vector<UsdGenExecutionInputGeneration>* values)
{
    std::sort(values->begin(), values->end(),
        [](auto const& a, auto const& b) { return a.identity < b.identity; });
}

void Coalesce(std::vector<UsdGenExecutionInputGeneration>* values, bool* valid)
{
    Canonicalize(values);
    // In place: the sorted run of each identity folds into its first entry.
    size_t kept = 0;
    for (size_t i = 0; i != values->size(); ++i) {
        UsdGenExecutionInputGeneration const& value = (*values)[i];
        if (kept == 0 || (*values)[kept - 1].identity != value.identity) {
            if (kept != i) (*values)[kept] = value;
            ++kept;
            continue;
        }
        UsdGenExecutionInputGeneration& unique = (*values)[kept - 1];
        if (unique.generation != 0 && value.generation != 0 &&
            unique.generation != value.generation)
            *valid = false;
        // A resolved payload generation is stronger than a relationship-only
        // zero. On contradictory nonzero records, retain the newest value so
        // an accidental stale alias cannot be treated as current.
        unique.generation = std::max(unique.generation, value.generation);
    }
    values->erase(values->begin() + std::ptrdiff_t(kept), values->end());
}

uint64_t HashValues(uint64_t hash,
                    std::vector<UsdGenExecutionInputGeneration> const& values,
                    uint64_t tag) noexcept
{
    hash = MixU64(hash, tag);
    hash = MixU64(hash, static_cast<uint64_t>(values.size()));
    for (auto const& value : values) {
        hash = MixPath(hash, value.identity);
        hash = MixU64(hash, value.generation);
    }
    return hash;
}

bool Update(std::vector<UsdGenExecutionInputGeneration>* values,
            SdfPath const& identity, uint64_t generation) noexcept
{
    bool found = false;
    for (auto& value : *values) {
        if (value.identity == identity) {
            value.generation = generation;
            found = true;
        }
    }
    return found;
}

} // namespace

UsdGenExecutionInputVersions UsdGenExecutionInputVersions::FromDescription(
    UsdGenGraphDesc const& desc)
{
    UsdGenExecutionInputVersions result;
    FromDescription(desc, &result);
    return result;
}

void UsdGenExecutionInputVersions::FromDescription(
    UsdGenGraphDesc const& desc, UsdGenExecutionInputVersions* out)
{
    // Refilled in place: a recompile keeps the vectors' capacity.
    UsdGenExecutionInputVersions& result = *out;
    result.sources.clear();
    result.references.clear();
    result.maps.clear();
    result.surfaces.clear();
    result.valid = true;
    for (auto const& curve : desc.curveSets) {
        auto& bucket = curve.role == UsdGenRole::Reference
            ? result.references : result.sources;
        bucket.push_back({curve.path, curve.curveGeneration});
    }
    for (auto const& node : desc.nodes)
        for (auto const& reference : node.references)
            result.references.push_back({reference, 0});
    for (auto const& map : desc.maps)
        result.maps.push_back({map.path, map.textureGeneration});
    for (auto const& surface : desc.surfaces)
        result.surfaces.push_back({surface.path, surface.surfaceGeneration});
    Coalesce(&result.sources, &result.valid);
    Coalesce(&result.references, &result.valid);
    Coalesce(&result.maps, &result.valid);
    Coalesce(&result.surfaces, &result.valid);
}

uint64_t UsdGenExecutionInputVersions::Hash() const noexcept
{
    uint64_t hash = 1469598103934665603ULL;
    hash = HashValues(hash, sources, 1);
    hash = HashValues(hash, references, 2);
    hash = HashValues(hash, maps, 3);
    hash = HashValues(hash, surfaces, 4);
    return MixU64(hash, valid ? 1 : 0);
}

bool UsdGenExecutionInputVersions::UpdateCurve(
    SdfPath const& identity, uint64_t generation) noexcept
{
    return Update(&sources, identity, generation) |
        Update(&references, identity, generation);
}

bool UsdGenExecutionInputVersions::UpdateMap(
    SdfPath const& identity, uint64_t generation) noexcept
{
    return Update(&maps, identity, generation);
}

bool UsdGenExecutionInputVersions::UpdateSurface(
    SdfPath const& identity, uint64_t generation) noexcept
{
    return Update(&surfaces, identity, generation);
}

UsdGenExecutionCacheKey UsdGenExecutionCacheKey::Make(
    SdfPath description, UsdGenEpoch planDigest,
    UsdGenExecutionInputVersions inputs,
    UsdGenExecutionContext context,
    uint64_t layoutDigest, double frame) noexcept
{
    UsdGenExecutionCacheKey result;
    result.description = std::move(description);
    result.planDigest = planDigest;
    result.inputs = std::move(inputs);
    result.context = context;
    result.layoutDigest = layoutDigest;
    std::memcpy(&result.frameBits, &frame, sizeof(frame));
    return result;
}

uint64_t UsdGenExecutionCacheKey::Hash() const noexcept
{
    uint64_t hash = MixPath(1469598103934665603ULL, description);
    hash = MixU64(hash, planDigest[0]);
    hash = MixU64(hash, planDigest[1]);
    hash = MixU64(hash, layoutDigest);
    hash = MixU64(hash, frameBits);
    hash = MixU64(hash, static_cast<uint64_t>(context.backend));
    hash = MixU64(hash, context.capabilityVersion);
    hash = MixU64(hash, static_cast<uint64_t>(context.deviceIndex));
    hash = MixU64(hash, context.deviceGeneration);
    hash = MixU64(hash, static_cast<uint64_t>(context.evaluationContext));
    return MixU64(hash, inputs.Hash());
}


namespace {

// Test seam; constant-initialized, read with one acquire load per point.
std::atomic<UsdGenExecutionCacheDomain::TestHook> gTestHook{nullptr};
std::atomic<void*> gTestHookContext{nullptr};

void RunTestHook(UsdGenExecutionCacheDomain::TestPoint point) noexcept
{
    if (auto hook = gTestHook.load(std::memory_order_acquire))
        hook(point, gTestHookContext.load(std::memory_order_acquire));
}

// The domain (if any) whose publication action this thread is running, and
// how deep (packed by epoch parity, like _publishing): an Invalidate reached
// from inside that action must not wait for its own caller.
thread_local UsdGenExecutionCacheDomain const* tPublishDomain = nullptr;
thread_local uint32_t tPublishDepth = 0;

// _publishing counts publications by the parity of the epoch they were
// checked against, 15 bits each, so an Invalidate waits only for those of
// the epoch it retired, never for publications of the new epoch that keep
// starting meanwhile.
constexpr uint32_t PublishUnit(uint64_t epoch) noexcept
{
    return 1u << (15u * uint32_t(epoch & 1u));
}
constexpr uint32_t PublishCount(uint32_t word, uint64_t epoch) noexcept
{
    return (word >> (15u * uint32_t(epoch & 1u))) & 0x7fffu;
}

}  // namespace

void UsdGenExecutionCacheDomain::SetTestHook(TestHook hook, void* context) noexcept
{
    gTestHookContext.store(context, std::memory_order_release);
    gTestHook.store(hook, std::memory_order_release);
}

UsdGenExecutionCacheDomain::~UsdGenExecutionCacheDomain()
{
    // No caller remains. Unsettled waiters are dropped without a callback.
    FlightList* list = _flights.load(std::memory_order_acquire);
    if (!list) return;
    for (Flight* flight : list->flights) {
        for (Waiter* waiter = flight->waiters.Close(); waiter;) {
            Waiter* next = waiter->next;
            delete waiter;
            waiter = next;
        }
        delete flight;
    }
    delete list;
}

std::optional<UsdGenExecutionCacheDomain::CoalescedRegistration>
UsdGenExecutionCacheDomain::BeginCoalesced(
    UsdGenExecutionCacheKey const& key, CoalescedCallback callback,
    bool notifyLeader)
{
    if (!callback || !key.IsValid()) return std::nullopt;
    uint64_t const epoch = key.context.deviceGeneration;
    std::unique_ptr<Waiter> waiter;
    for (;;) {
        if (!_Current(key)) return std::nullopt;
        if (auto resident = store_._Lookup(key, &epoch)) {
            CoalescedCallback immediate =
                waiter ? std::move(waiter->callback) : std::move(callback);
            try { immediate(CoalescedStatus::Resident, std::move(resident)); }
            catch (...) {}
            return std::nullopt;
        }
        try {
            if (!waiter) {
                waiter.reset(new Waiter);
                waiter->callback = std::move(callback);
                waiter->id = _nextWaiter.fetch_add(1, std::memory_order_relaxed) + 1;
            }
            std::unique_ptr<Flight> fresh;
            std::unique_ptr<FlightList> nextList;
            RunTestHook(TestPoint::BeginCoalescedBeforePublish);
            // Set only on the withdraw path, and used after the guard ends:
            // callbacks never run or die inside it.
            Flight* published = nullptr;
            CoalescedCallback unused;
            {
                UsdGenEpochGuard guard;
                FlightList* list = _flights.load(std::memory_order_seq_cst);
                Flight* flight = _FindFlight(list, key);
                if (flight && !flight->settled.load(std::memory_order_acquire)) {
                    if (flight->epoch != epoch_.load(std::memory_order_acquire))
                        return std::nullopt;
                    if (flight->waiters.Push(waiter.get())) {
                        Waiter* joined = waiter.release();
                        _coalesced.fetch_add(1, std::memory_order_acq_rel);
                        return CoalescedRegistration{key, flight->id, joined->id, false};
                    }
                    continue;  // closed meanwhile: it is settled now
                }
                // No open flight: publish one, dropping a settled leftover.
                fresh.reset(new Flight);
                fresh->key = key;
                fresh->id = _nextFlight.fetch_add(1, std::memory_order_relaxed) + 1;
                fresh->epoch = epoch;
                fresh->leaderWaiter = waiter->id;
                fresh->notifyLeader = notifyLeader;
                waiter->next = nullptr;
                fresh->waiters.Push(waiter.get());
                nextList.reset(new FlightList);
                if (list) {
                    nextList->flights.reserve(list->flights.size() + 1);
                    for (Flight* other : list->flights)
                        if (other != flight) nextList->flights.push_back(other);
                }
                nextList->flights.push_back(fresh.get());
                if (!_flights.compare_exchange_strong(list, nextList.get(),
                                                      std::memory_order_seq_cst))
                    continue;
                nextList.release();
                Flight* candidate = fresh.release();
                Waiter* leader = waiter.release();
                if (list) UsdGenEpochRetireDelete(list);
                // An Invalidate that bumped the epoch before it could see this
                // flight: withdraw it unless Invalidate already settled it (then
                // Invalidate owns it, and only the guard keeps it readable here).
                uint32_t open = 0;
                if (epoch_.load(std::memory_order_seq_cst) == epoch ||
                    !candidate->settled.compare_exchange_strong(
                        open, 1, std::memory_order_acq_rel))
                    return CoalescedRegistration{key, candidate->id, leader->id, true};
                // Not registered: the leader's callback never runs.
                leader->claimed.store(1, std::memory_order_release);
                unused = std::move(leader->callback);
                published = candidate;
            }
            // Winning the settled CAS made this thread the flight's only
            // retirer, so it stays valid without a guard.
            bool const unlisted = _Unlist(published);
            // Followers that joined meanwhile are registered: Stale.
            _Notify(published, CoalescedStatus::Stale, std::nullopt);
            if (unlisted) UsdGenEpochRetireDelete(published);
            return std::nullopt;
        } catch (...) {
            return std::nullopt;
        }
    }
}

bool UsdGenExecutionCacheDomain::_Unlist(Flight* flight) noexcept
{
    UsdGenEpochGuard guard;
    FlightList* current = _flights.load(std::memory_order_seq_cst);
    for (;;) {
        if (!current || std::find(current->flights.begin(), current->flights.end(),
                                  flight) == current->flights.end())
            return true;
        FlightList* next = nullptr;
        if (current->flights.size() > 1) {
            next = new (std::nothrow) FlightList;
            if (!next) return false;  // left listed: a joiner replaces it
            try {
                next->flights.reserve(current->flights.size() - 1);
                for (Flight* other : current->flights)
                    if (other != flight) next->flights.push_back(other);
            } catch (...) {
                delete next;
                return false;
            }
        }
        if (_flights.compare_exchange_strong(current, next, std::memory_order_seq_cst)) {
            UsdGenEpochRetireDelete(current);
            return true;
        }
        delete next;
    }
}

void UsdGenExecutionCacheDomain::_Notify(Flight* flight, CoalescedStatus status,
                                         std::optional<Lease> const& lease) noexcept
{
    // Newest first; links are not rewritten, so a concurrent canceller's
    // walk stays valid. A waiter claimed by its canceller is skipped.
    for (Waiter* waiter = flight->waiters.Close(); waiter;) {
        Waiter* next = waiter->next;
        uint32_t open = 0;
        if (waiter->claimed.compare_exchange_strong(open, 1, std::memory_order_acq_rel)) {
            // Released here, never by the epoch deleter: captures may own
            // native state.
            CoalescedCallback callback = std::move(waiter->callback);
            if (callback && !(waiter->id == flight->leaderWaiter && !flight->notifyLeader)) {
                try {
                    callback(status, lease ? std::optional<Lease>(*lease) : std::nullopt);
                } catch (...) {}
            }
        }
        UsdGenEpochRetireDelete(waiter);
        waiter = next;
    }
}

bool UsdGenExecutionCacheDomain::_Settle(
    CoalescedRegistration const& registration,
    std::shared_ptr<const UsdGenGeneration> payload, size_t bytes,
    CoalescedStatus status, std::string* reason)
{
    if (!registration || !registration.leader) return false;
    Flight* flight = nullptr;
    {
        UsdGenEpochGuard guard;
        Flight* found = _FindFlight(_flights.load(std::memory_order_seq_cst),
                                    registration.key);
        if (!found || found->id != registration.flight) return false;
        uint32_t open = 0;
        if (!found->settled.compare_exchange_strong(open, 1, std::memory_order_acq_rel))
            return false;
        flight = found;
    }
    RunTestHook(TestPoint::SettleWonFlight);
    bool const listed = !_Unlist(flight);
    auto stale = [&] {
        return flight->epoch != epoch_.load(std::memory_order_acquire) ||
            _clearFailed.load(std::memory_order_acquire);
    };
    std::optional<Lease> lease;
    bool result = false;
    if (stale()) {
        status = CoalescedStatus::Stale;
        if (reason) *reason = "execution cache coalesced flight is stale";
    } else if (status == CoalescedStatus::Success) {
        status = CoalescedStatus::Failed;
        bool inserted = false;
        try {
            // The lease is taken before the entry is published, so a racing
            // inserter cannot evict it between admission and delivery.
            inserted = store_._Insert(registration.key, std::move(payload), bytes,
                                      reason, &flight->epoch, &lease);
        } catch (...) {}
        if (inserted) {
            if (lease) {
                status = CoalescedStatus::Success;
                result = true;
            } else if (reason && reason->empty()) {
                *reason = "execution cache coalesced result lookup failed";
            }
        }
        if (!result && stale()) {
            status = CoalescedStatus::Stale;
            if (reason) *reason = "execution cache coalesced flight is stale";
        }
    } else {
        result = true;
    }
    _Notify(flight, status, lease);
    if (!listed) UsdGenEpochRetireDelete(flight);  // else leaked: still reachable
    return result;
}

bool UsdGenExecutionCacheDomain::_Withdraw(
    CoalescedRegistration const& registration, CoalescedCallback* callback) noexcept
{
    if (!registration) return false;
    UsdGenEpochGuard guard;
    Flight* flight = _FindFlight(_flights.load(std::memory_order_seq_cst),
                                 registration.key);
    if (!flight || flight->id != registration.flight) return false;
    for (Waiter* waiter = flight->waiters.Peek(); waiter; waiter = waiter->next) {
        if (waiter->id != registration.waiter) continue;
        uint32_t open = 0;
        if (!waiter->claimed.compare_exchange_strong(open, 1, std::memory_order_acq_rel))
            return false;
        *callback = std::move(waiter->callback);
        return true;
    }
    return false;
}

bool UsdGenExecutionCacheDomain::Insert(UsdGenExecutionCacheKey key,
                                        std::shared_ptr<const UsdGenGeneration> payload,
                                        size_t bytes, std::string* reason)
{
    if (_clearFailed.load(std::memory_order_acquire)) {
        if (reason) *reason = "execution cache domain cleanup is pending";
        return false;
    }
    if (!_Current(key)) {
        if (reason) *reason = "execution cache domain epoch or identity is stale";
        return false;
    }
    uint64_t const epoch = key.context.deviceGeneration;
    uint64_t const generation = payload && payload->device
        ? payload->device->Identity().generation : 0;
    // A key of the new epoch can arrive between Invalidate's bump and its
    // clear: finish the clear instead of failing.
    if (store_._StateEpoch() < epoch) store_._Clear(&epoch);
    bool const inserted = store_._Insert(std::move(key), std::move(payload), bytes,
                                         reason, &epoch, nullptr);
    if (inserted) {
        uint64_t mark = _publicationWatermark.load(std::memory_order_acquire);
        while (generation > mark &&
               !_publicationWatermark.compare_exchange_weak(
                   mark, generation, std::memory_order_acq_rel, std::memory_order_acquire)) {}
    }
    return inserted;
}

bool UsdGenExecutionCacheDomain::Invalidate() noexcept
{
    if (_clearFailed.load(std::memory_order_acquire)) {
        uint64_t const epoch = epoch_.load(std::memory_order_acquire);
        bool const cleared = store_._Clear(&epoch);
        if (cleared) _clearFailed.store(false, std::memory_order_release);
        return cleared;
    }
    uint64_t value = epoch_.load(std::memory_order_relaxed);
    for (;;) {
        if (value == std::numeric_limits<uint64_t>::max()) return false;
        if (epoch_.compare_exchange_weak(value, value + 1, std::memory_order_seq_cst,
                                         std::memory_order_relaxed))
            break;
    }
    uint64_t const next = value + 1;
    // Every flight of the old epoch goes Stale. Taking the whole list makes
    // this thread the only one that can retire it, but not its flights: a
    // leader (or a withdrawing BeginCoalesced) that wins a flight's settled
    // CAS retires it, so the walk stays pinned. Pinned before the exchange,
    // so every such retire comes after the pin.
    Flight* stale = nullptr;
    {
        UsdGenEpochGuard guard;
        if (FlightList* list = _flights.exchange(nullptr, std::memory_order_seq_cst)) {
            RunTestHook(TestPoint::InvalidateTookFlights);
            for (Flight* flight : list->flights) {
                uint32_t open = 0;
                if (flight->settled.compare_exchange_strong(open, 1, std::memory_order_acq_rel)) {
                    flight->settleNext = stale;
                    stale = flight;
                }
            }
            UsdGenEpochRetireDelete(list);
        }
    }
    // No old-epoch publication may land after this returns.
    _AwaitPublications(value);
    bool const cleared = store_._Clear(&next);
    if (!cleared) _clearFailed.store(true, std::memory_order_release);
    while (stale) {
        Flight* flight = stale;
        stale = flight->settleNext;
        _Notify(flight, CoalescedStatus::Stale, std::nullopt);
        UsdGenEpochRetireDelete(flight);
    }
    return cleared;
}

bool UsdGenExecutionCacheDomain::_PublishIfCurrent(
    uint64_t epoch, void (*run)(void*), void* context)
{
    // Store-load against Invalidate (seq_cst on both sides): either this
    // announcement is visible to its count load, or its bump is visible here.
    // Counted under the fence's epoch: the one an Invalidate retiring it
    // waits for.
    uint32_t const unit = PublishUnit(epoch);
    _publishing.fetch_add(unit, std::memory_order_seq_cst);
    struct Scope {
        UsdGenExecutionCacheDomain* domain;
        UsdGenExecutionCacheDomain const* previousDomain;
        uint32_t previousDepth;
        uint32_t unit;
        ~Scope() {
            tPublishDomain = previousDomain;
            tPublishDepth = previousDepth;
            domain->_EndPublish(unit);
        }
    } scope{this, tPublishDomain, tPublishDepth, unit};
    tPublishDepth = tPublishDomain == this ? tPublishDepth + unit : unit;
    tPublishDomain = this;
    if (_clearFailed.load(std::memory_order_seq_cst) ||
        epoch != epoch_.load(std::memory_order_seq_cst))
        return false;
    run(context);
    return true;
}

void UsdGenExecutionCacheDomain::_EndPublish(uint32_t unit) noexcept
{
    uint32_t const before = _publishing.fetch_sub(unit, std::memory_order_seq_cst);
    if (!(before & kPublishWaiter)) return;
    // A parked Invalidate: the last publication clears the flag; every
    // decrement wakes, so a waiter re-entered from an action sees its count.
    if (before == (kPublishWaiter | unit)) {
        uint32_t expected = kPublishWaiter;
        _publishing.compare_exchange_strong(expected, 0, std::memory_order_seq_cst);
    }
    UsdGenUnparkAll(_publishing);
}

void UsdGenExecutionCacheDomain::_AwaitPublications(uint64_t epoch) noexcept
{
    // Never wait at process exit: the publishers may already be gone.
    if (UsdGenProcessShutdownInProgress()) return;
    uint32_t const own = tPublishDomain == this ? PublishCount(tPublishDepth, epoch) : 0;
    uint32_t word = _publishing.load(std::memory_order_seq_cst);
    while (PublishCount(word, epoch) > own) {
        if (!(word & kPublishWaiter)) {
            if (!_publishing.compare_exchange_weak(word, word | kPublishWaiter,
                                                   std::memory_order_seq_cst))
                continue;
            word |= kPublishWaiter;
        }
        UsdGenParkWhileEqual(_publishing, word);
        word = _publishing.load(std::memory_order_seq_cst);
    }
}

namespace {

// Shared-domain registry: a copy-on-write list published by atomic pointer.
// An acquire that adds a domain CASes a new list (dropping expired entries),
// so the key check and the insert are one step. Old lists retire through
// epochs. Constant-initialized and trivially destructible.
struct DomainEntry {
    UsdGenExecutionCacheDomainKey key;
    std::weak_ptr<UsdGenExecutionCacheDomain> domain;
};
struct DomainRegistry : UsdGenEpochNode {
    std::vector<DomainEntry> entries;
};
std::atomic<DomainRegistry*> gDomainRegistry{nullptr};

}  // namespace

std::shared_ptr<UsdGenExecutionCacheDomain> UsdGenExecutionCacheDomain::AcquireShared(
    UsdGenExecutionCacheDomainKey key, size_t maxBytes)
{
    if (key.backend == UsdGenDeviceBackend::Unknown || maxBytes == 0)
        return {};
    std::shared_ptr<UsdGenExecutionCacheDomain> fresh;
    for (;;) {
        std::unique_ptr<DomainRegistry> next(new DomainRegistry);
        {
            UsdGenEpochGuard guard;
            DomainRegistry* current = gDomainRegistry.load(std::memory_order_seq_cst);
            if (current) {
                for (DomainEntry const& entry : current->entries) {
                    if (entry.key != key) continue;
                    if (auto live = entry.domain.lock())
                        return live->MaxBytes() == maxBytes ? live : nullptr;
                }
                next->entries.reserve(current->entries.size() + 1);
                for (DomainEntry const& entry : current->entries)
                    if (!entry.domain.expired()) next->entries.push_back(entry);
            }
            if (!fresh) fresh = std::make_shared<UsdGenExecutionCacheDomain>(key, maxBytes);
            next->entries.push_back(DomainEntry{key, fresh});
            if (gDomainRegistry.compare_exchange_strong(current, next.get(),
                                                        std::memory_order_seq_cst)) {
                next.release();
                if (current) UsdGenEpochRetireDelete(current);
                return fresh;
            }
        }
    }
}

} // namespace usdGen

// The cache store is header-only so callers can select a payload type and an
// injectable hasher without adding backend-specific explicit instantiations.
