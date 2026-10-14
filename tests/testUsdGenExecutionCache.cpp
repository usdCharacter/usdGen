#include "usdGen/compiler.h"
#include "usdGen/executionCache.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/sessionCooker.h"

#include "pxr/base/gf/vec3f.h"

#include <cstdio>
#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {
int failures = 0;
void Check(bool value, char const* message)
{
    if (!value) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

UsdGenGraphDesc Description()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/cache/groomA");
    desc.terminal = SdfPath("/cache/groomA/width");
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/cache/groomA/scalp");
    surface.surfaceGeneration = 41;
    surface.faceVertexCounts = VtIntArray{3};
    surface.faceVertexIndices = VtIntArray{0, 1, 2};
    surface.restPoints = VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(0, 1, 0)};
    surface.points = surface.restPoints;
    desc.surfaces.push_back(surface);
    UsdGenCurveSetDesc hair;
    hair.path = SdfPath("/cache/groomA/hair");
    hair.role = UsdGenRole::Curves;
    hair.curveRole = TfToken("hair");
    hair.curveGeneration = 7;
    // The publication tests need actual nonempty C3 data, not the legacy
    // CurveSource surrogate that treated scalp vertices as curve points.
    hair.curveVertexCounts = {3};
    hair.points = {{0,0,0}, {0,0,1}, {0,0,2}};
    hair.rest = hair.points;
    hair.curveId = {7};
    hair.skinPrim = {0};
    hair.skinPrimUv = {{0,0}};
    hair.rootFrame = {GfMatrix4d(1.0)};
    desc.curveSets.push_back(std::move(hair));
    UsdGenCurveSetDesc secondHair;
    secondHair.path = SdfPath("/cache/groomA/hair2");
    secondHair.role = UsdGenRole::Curves;
    secondHair.curveRole = TfToken("hair");
    secondHair.curveGeneration = 8;
    desc.curveSets.push_back(std::move(secondHair));
    UsdGenCurveSetDesc guides;
    guides.path = SdfPath("/cache/groomA/guides");
    guides.role = UsdGenRole::Reference;
    guides.curveRole = TfToken("guide");
    guides.curveGeneration = 9;
    desc.curveSets.push_back(std::move(guides));
    UsdGenMapDesc map;
    map.path = SdfPath("/cache/groomA/mask");
    map.type = TfToken("image");
    map.resolvedAssetPath = "/assets/mask.exr";
    map.textureGeneration = 13;
    desc.maps.push_back(std::move(map));
    UsdGenNodeDesc source;
    source.path = SdfPath("/cache/groomA/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {SdfPath("/cache/groomA/hair")};
    source.maps = {SdfPath("/cache/groomA/mask")};
    source.surfaces = {surface.path};
    UsdGenNodeDesc width;
    width.path = desc.terminal;
    width.type = TfToken("UsdGenWidth");
    width.inputs = {source.path};
    desc.nodes = {source, width};
    return desc;
}

struct ConstantHasher {
    size_t operator()(UsdGenExecutionCacheKey const&) const noexcept { return 7; }
};

struct ReentrantPayload {
    std::function<void()> onDestroy;
    ReentrantPayload() = default;
    explicit ReentrantPayload(std::function<void()> callback)
        : onDestroy(std::move(callback)) {}
    ~ReentrantPayload() { if (onDestroy) onDestroy(); }
};

struct ReentrantDomainOwner final : UsdGenDeviceOwner {
    explicit ReentrantDomainOwner(std::function<void()> callback)
        : onDestroy(std::move(callback)) {}
    ~ReentrantDomainOwner() override { if (onDestroy) onDestroy(); }
    bool ProducerReady() const noexcept override { return true; }
    size_t ExclusiveRetainedBytes() const noexcept override { return 1; }
    std::unique_ptr<UsdGenDeviceConsumer> AcquireConsumer(
        UsdGenDeviceStream) const noexcept override { return {}; }
    std::function<void()> onDestroy;
};

std::shared_ptr<const UsdGenGeneration> DevicePayload(
    int64_t id, std::shared_ptr<const UsdGenDeviceOwner> owner)
{
    UsdGenDeviceGeneration::CreateInfo info;
    info.identity = {UsdGenDeviceBackend::CpuReference, 0,
                     static_cast<uint64_t>(id + 1)};
    info.owner = std::move(owner);
    auto generation = std::make_shared<UsdGenGeneration>();
    generation->id = id;
    generation->device = UsdGenDeviceGeneration::Create(std::move(info));
    return generation->device
        ? std::shared_ptr<const UsdGenGeneration>(std::move(generation))
        : nullptr;
}
}

int main()
{
    usdGenRegisterM1Operators();
    UsdGenGraphDesc desc = Description();
    UsdGenExecutionInputVersions versions =
        UsdGenExecutionInputVersions::FromDescription(desc);
    Check(versions.sources.size() == 2 && versions.references.size() == 1 &&
              versions.maps.size() == 1 && versions.surfaces.size() == 1,
          "description produces exact source/reference/map/surface buckets");
    Check(versions.sources.front().generation == 7 &&
              versions.references.front().generation == 9 &&
              versions.maps.front().generation == 13 &&
              versions.surfaces.front().generation == 41,
          "input tuple preserves every authored generation");
    UsdGenGraphDesc aliased = desc;
    aliased.nodes[1].references = {SdfPath("/cache/groomA/guides")};
    auto aliasedVersions =
        UsdGenExecutionInputVersions::FromDescription(aliased);
    Check(aliasedVersions.references.size() == 1 &&
              aliasedVersions.references.front().generation == 9,
          "relationship-only reference aliases coalesce with payload generation");
    UsdGenGraphDesc contradictory = desc;
    contradictory.maps.push_back(contradictory.maps.front());
    contradictory.maps.back().textureGeneration = 99;
    Check(!UsdGenExecutionInputVersions::FromDescription(contradictory).IsValid(),
          "contradictory duplicate identities are rejected from cache admission");
    UsdGenGraphDesc reorderedDesc = desc;
    std::reverse(reorderedDesc.curveSets.begin(), reorderedDesc.curveSets.end());
    UsdGenExecutionInputVersions reordered =
        UsdGenExecutionInputVersions::FromDescription(reorderedDesc);
    Check(reordered == versions && reordered.Hash() == versions.Hash(),
          "equal canonical tuples have equal hashes");
    Check(versions.UpdateMap(SdfPath("/cache/groomA/mask"), 14) &&
              versions != UsdGenExecutionInputVersions::FromDescription(desc),
          "map generation change changes tuple identity");

    UsdGenExecutionContext cuda{
        UsdGenDeviceBackend::Cuda, 8, 2, 17};
    auto key = UsdGenExecutionCacheKey::Make(
        desc.description, {11, 22}, versions, cuda, 99, 24.0);
    Check(key.IsValid(), "valid input tuple produces an admissible cache key");
    auto unknownContext = key;
    unknownContext.context.backend = UsdGenDeviceBackend::Unknown;
    Check(!unknownContext.IsValid(), "unknown backend context is not cache-admissible");
    auto same = key;
    Check(same == key && same.Hash() == key.Hash(),
          "identical cache keys compare and hash equally");
    same.context.capabilityVersion++;
    Check(same != key, "capability version partitions cache keys");
    same = key; same.context.deviceIndex++;
    Check(same != key, "device identity partitions cache keys");
    same = key; same.context.backend = UsdGenDeviceBackend::Metal;
    Check(same != key, "backend identity partitions cache keys");
    same = key; same.context.evaluationContext = UsdGenContext::Render;
    Check(same != key, "interactive and render contexts partition cache keys");
    same = key; same.inputs.UpdateSurface(desc.surfaces.front().path, 42);
    Check(same != key, "surface generation partitions cache keys");
    same = key; same.frameBits = key.frameBits ^ 1;
    Check(same != key, "sample identity partitions cache keys");

    using Store = UsdGenExecutionCacheStore<int>;
    auto payload = [](int value) {
        return std::make_shared<const int>(value);
    };
    Store store(8);
    UsdGenExecutionCacheKey keyA = key;
    keyA.layoutDigest = 1;
    UsdGenExecutionCacheKey keyB = key;
    keyB.layoutDigest = 2;
    UsdGenExecutionCacheKey keyC = key;
    keyC.layoutDigest = 3;
    std::string reason;
    Check(store.Insert(keyA, payload(11), 4, &reason) && store.Bytes() == 4,
          "cache inserts immutable payload and charges bytes");
    auto hitA = store.Lookup(keyA);
    Check(hitA && **hitA == 11, "cache lookup returns an immutable lease");
    Check(store.Insert(keyB, payload(22), 4, &reason),
          "cache admits a second bounded entry");
    hitA.reset();
    auto promoteA = store.Lookup(keyA);
    Check(promoteA && store.Insert(keyC, payload(33), 4, &reason),
          "cache promotes MRU and evicts deterministic LRU");
    promoteA.reset();
    Check(store.Lookup(keyB) == std::nullopt && store.Lookup(keyA) &&
              store.Lookup(keyC),
          "LRU eviction retains the promoted entry and removes the oldest");

    Store replacement(8);
    Check(replacement.Insert(keyA, payload(1), 4, &reason),
          "duplicate coalescing has an initial value");
    auto oldLease = replacement.Lookup(keyA);
    Check(oldLease && replacement.Insert(keyA, payload(2), 4, &reason) &&
              replacement.Bytes() == 4 && replacement.Size() == 1,
          "exact duplicate key coalesces without replacing immutable state");
    auto coalescedLease = replacement.Lookup(keyA);
    Check(oldLease && coalescedLease && **oldLease == 1 && **coalescedLease == 1,
          "duplicate publication retains the first accepted immutable payload");
    Check(!replacement.Insert(keyA, payload(3), 6, &reason) &&
              replacement.Bytes() == 4 && replacement.Size() == 1,
          "duplicate key with conflicting byte charge fails transactionally");

    Store pinned(8);
    Check(pinned.Insert(keyA, payload(4), 4, &reason),
          "pinned-cache setup succeeds");
    auto pinnedLease = pinned.Lookup(keyA);
    auto copiedPinnedLease = *pinnedLease;
    pinnedLease.reset();
    size_t const beforeBytes = pinned.Bytes();
    size_t const beforeSize = pinned.Size();
    Check(!pinned.Insert(keyB, payload(5), 8, &reason) &&
              pinned.Bytes() == beforeBytes && pinned.Size() == beforeSize,
          "a copied lease pins COW data and failed admission preserves accounting");
    copiedPinnedLease = Store::Lease{};
    Check(pinned.Insert(keyB, payload(5), 8, &reason) &&
              pinned.Bytes() == 8 && pinned.Size() == 1,
          "releasing the final copied lease permits deterministic eviction");

    Store cleared(16);
    Check(cleared.Insert(keyA, payload(61), 4, &reason),
          "clear fixture inserts a resident immutable payload");
    auto clearLease = cleared.Lookup(keyA);
    auto copiedClearLease = clearLease ? *clearLease : Store::Lease{};
    Check(clearLease && copiedClearLease && cleared.Clear() &&
              cleared.Size() == 0 && cleared.Bytes() == 0 &&
              **clearLease == 61 && *copiedClearLease == 61,
          "clear drops resident accounting while copied leases retain readable COW data");
    clearLease.reset();
    copiedClearLease = Store::Lease{};

    // A cache domain is the COW boundary shared by otherwise independent
    // cookers.  Its epoch is part of the exact tuple: context replacement
    // immediately hides older entries, while an acquired consumer lease keeps
    // the old immutable payload readable until that consumer releases it.
    using Domain = UsdGenExecutionCacheDomain;
    auto domain = std::make_shared<Domain>(
        UsdGenExecutionCacheDomainKey{
            UsdGenDeviceBackend::CpuReference, -1, 91},
        2 * sizeof(UsdGenGeneration));
    auto domainPayload = [](int id) {
        auto generation = std::make_shared<UsdGenGeneration>();
        generation->id = id;
        return std::shared_ptr<const UsdGenGeneration>(std::move(generation));
    };
    UsdGenExecutionCacheKey domainKey = keyA;
    domainKey.context = UsdGenExecutionContext{
        UsdGenDeviceBackend::CpuReference, 0, -1, domain->Epoch(),
        UsdGenContext::Interactive};
    auto firstDomainPayload = domainPayload(71);
    Check(domain->Insert(domainKey, firstDomainPayload,
                         sizeof(UsdGenGeneration), &reason) &&
              domain->Size() == 1 && domain->Bytes() == sizeof(UsdGenGeneration),
          "shared domain admits a completed CPU result under its exact context tuple");
    auto domainLease = domain->Lookup(domainKey);
    auto incompatibleDomainKey = domainKey;
    incompatibleDomainKey.context.capabilityVersion = 1;
    Check(domainLease && (**domainLease).id == 71 &&
              !domain->Lookup(incompatibleDomainKey),
          "domain partitions completed results by the full execution context");
    uint64_t const oldDomainEpoch = domain->Epoch();
    auto stalePublicationFence = domain->CapturePublicationFence(domainKey);
    bool stalePublicationRan = false;
    Check(stalePublicationFence && domain->Invalidate() &&
              !domain->PublishIfCurrent(*stalePublicationFence, [&] {
                  stalePublicationRan = true;
                  return std::shared_ptr<const void>{};
              }) && !stalePublicationRan &&
              domain->Epoch() == oldDomainEpoch + 1 &&
              domain->Size() == 0 && domain->Bytes() == 0 &&
              (**domainLease).id == 71,
          "a captured publication cannot run after deterministic matching-domain invalidation");
    Check(!domain->Insert(domainKey, domainPayload(72),
                          sizeof(UsdGenGeneration), &reason) &&
              reason == "execution cache domain epoch or identity is stale" &&
              domain->Size() == 0,
          "a candidate minted before a domain epoch bump cannot repopulate the cache");
    UsdGenExecutionCacheKey refreshedDomainKey = domainKey;
    refreshedDomainKey.context.deviceGeneration = domain->Epoch();
    auto refreshedDomainPayload = domainPayload(73);
    Check(domain->Insert(refreshedDomainKey, refreshedDomainPayload,
                         sizeof(UsdGenGeneration), &reason) &&
              domain->Size() == 1 && domain->Lookup(refreshedDomainKey),
          "a completed result minted in the replacement context is independently resident");
    bool currentPublicationRan = false;
    auto currentPublicationFence = domain->CapturePublicationFence(refreshedDomainKey);
    Check(currentPublicationFence && domain->PublishIfCurrent(
              *currentPublicationFence, [&] {
                  currentPublicationRan = true;
                  return std::shared_ptr<const void>{};
              }) &&
              currentPublicationRan,
          "a current-domain publication fence runs its bounded local publication action");

    // Exact-key miss coalescing is a neutral, nonblocking seam.  The first
    // waiter is the only resolver; every follower receives a distinct Lease
    // object over the same immutable COW generation/payload.
    UsdGenExecutionCacheKey coalescedKey = refreshedDomainKey;
    coalescedKey.layoutDigest += 100;
    auto coalescedLeaderPayload = DevicePayload(
        74, std::make_shared<ReentrantDomainOwner>(std::function<void()>{}));
    std::vector<Domain::CoalescedStatus> coalescedLeaderStatuses;
    std::vector<Domain::CoalescedStatus> coalescedFollowerStatuses;
    std::optional<Domain::Lease> coalescedLeaderLease;
    std::optional<Domain::Lease> coalescedFollowerLease;
    bool coalescedCallbackReentered = false;
    auto leaderRegistration = domain->BeginCoalesced(
        coalescedKey, [&](Domain::CoalescedStatus status,
                          std::optional<Domain::Lease> lease) {
            coalescedLeaderStatuses.push_back(status);
            coalescedLeaderLease = std::move(lease);
            coalescedCallbackReentered =
                status == Domain::CoalescedStatus::Success &&
                bool(domain->Lookup(coalescedKey));
        });
    auto followerRegistration = domain->BeginCoalesced(
        coalescedKey, [&](Domain::CoalescedStatus status,
                          std::optional<Domain::Lease> lease) {
            coalescedFollowerStatuses.push_back(status);
            coalescedFollowerLease = std::move(lease);
        });
    Check(leaderRegistration && followerRegistration &&
              leaderRegistration->IsLeader() && !followerRegistration->IsLeader() &&
              domain->CoalescedCount() == 1 &&
              coalescedLeaderStatuses.empty() && coalescedFollowerStatuses.empty(),
          "same-key misses register one leader and one nonblocking follower");
    Check(domain->ResolveCoalesced(
              *leaderRegistration, coalescedLeaderPayload, 1, &reason) &&
              coalescedLeaderStatuses ==
                  std::vector<Domain::CoalescedStatus>{Domain::CoalescedStatus::Success} &&
              coalescedFollowerStatuses ==
                  std::vector<Domain::CoalescedStatus>{Domain::CoalescedStatus::Success} &&
              coalescedLeaderLease && coalescedFollowerLease &&
              coalescedLeaderLease->SharedPayload() ==
                  coalescedFollowerLease->SharedPayload() &&
              (**coalescedLeaderLease).device == (**coalescedFollowerLease).device &&
              coalescedCallbackReentered,
          "leader success fulfills all waiters with shared immutable COW data");
    Check(!domain->ResolveCoalesced(
              *leaderRegistration, DevicePayload(
                  75, std::make_shared<ReentrantDomainOwner>(std::function<void()>{})),
              1, &reason),
          "a completed coalesced leader registration cannot resolve twice");

    UsdGenExecutionCacheKey failedKey = coalescedKey;
    failedKey.layoutDigest += 1;
    std::vector<Domain::CoalescedStatus> failedStatuses;
    auto failedLeader = domain->BeginCoalesced(
        failedKey, [&](Domain::CoalescedStatus status,
                       std::optional<Domain::Lease> lease) {
            failedStatuses.push_back(status);
            Check(!lease, "failed coalesced resolution has no immutable payload");
        });
    auto failedFollower = domain->BeginCoalesced(
        failedKey, [&](Domain::CoalescedStatus status,
                       std::optional<Domain::Lease> lease) {
            failedStatuses.push_back(status);
            Check(!lease, "failed follower resolution has no immutable payload");
        });
    Check(failedLeader && failedFollower &&
              domain->RejectCoalesced(*failedLeader) &&
              failedStatuses == std::vector<Domain::CoalescedStatus>{
                  Domain::CoalescedStatus::Failed,
                  Domain::CoalescedStatus::Failed} &&
              !domain->Lookup(failedKey),
          "leader failure wakes every waiter without seeding the cache");

    // A follower may withdraw independently; cancelling the leader resolves
    // the remaining callbacks without inserting a cache entry.
    UsdGenExecutionCacheKey cancelledKey = coalescedKey;
    cancelledKey.layoutDigest += 1;
    int cancelledFollowerCallbacks = 0;
    auto cancelledLeader = domain->BeginCoalesced(
        cancelledKey, [&](Domain::CoalescedStatus status,
                          std::optional<Domain::Lease> lease) {
            Check(status == Domain::CoalescedStatus::Cancelled && !lease,
                  "leader cancellation reports an explicit terminal status");
        });
    auto cancelledFollower = domain->BeginCoalesced(
        cancelledKey, [&](Domain::CoalescedStatus,
                          std::optional<Domain::Lease>) {
            ++cancelledFollowerCallbacks;
        });
    Check(cancelledLeader && cancelledFollower &&
              domain->CancelCoalesced(*cancelledFollower) &&
              domain->RejectCoalesced(*cancelledLeader,
                                      Domain::CoalescedStatus::Cancelled) &&
              cancelledFollowerCallbacks == 0 && !domain->Lookup(cancelledKey),
          "follower cancellation and leader cancellation do not seed the cache");

    // Epoch invalidation rejects every outstanding callback and makes the old
    // leader token unusable.  The callback is delivered after the domain lock
    // is released, so it is safe to re-enter the domain here.
    UsdGenExecutionCacheKey staleKey = coalescedKey;
    staleKey.layoutDigest += 2;
    bool staleCallbackReentered = false;
    auto staleRegistration = domain->BeginCoalesced(
        staleKey, [&](Domain::CoalescedStatus status,
                      std::optional<Domain::Lease> lease) {
            staleCallbackReentered = status == Domain::CoalescedStatus::Stale &&
                !lease && !domain->Lookup(staleKey);
        });
    Check(staleRegistration && domain->Invalidate() && staleCallbackReentered &&
              !domain->ResolveCoalesced(
                  *staleRegistration, DevicePayload(
                      76, std::make_shared<ReentrantDomainOwner>(std::function<void()>{})),
                  1, &reason),
          "epoch invalidation rejects stale coalesced fulfillment and wakes waiters");

    // The preceding invalidation deliberately retired the earlier fence;
    // refresh it before the existing reentrant-publication coverage below.
    refreshedDomainKey.context.deviceGeneration = domain->Epoch();
    currentPublicationFence = domain->CapturePublicationFence(refreshedDomainKey);

    // A race with an already-resident result is a synchronous callback, not a
    // new flight and not a coalesced counter increment.
    auto residentKey = refreshedDomainKey;
    residentKey.context.deviceGeneration = domain->Epoch();
    Check(domain->Insert(residentKey, DevicePayload(
              77, std::make_shared<ReentrantDomainOwner>(std::function<void()>{})),
              1, &reason),
          "resident-race fixture inserts a current immutable result");
    bool residentCallback = false;
    auto residentRegistration = domain->BeginCoalesced(
        residentKey, [&](Domain::CoalescedStatus status,
                         std::optional<Domain::Lease> lease) {
            residentCallback = status == Domain::CoalescedStatus::Resident &&
                bool(lease) && (**lease).id == 77;
        });
    Check(!residentRegistration && residentCallback &&
              domain->CoalescedCount() == 3,
          "resident race bypasses flight registration and coalescing count");

    // A generation can own arbitrary native/COW resources. Replacing it under
    // the domain mutex must return the old immutable owner and retire it only
    // after unlocking, or an owner destructor that re-enters the domain would
    // deadlock deterministically here.
    bool publicationOwnerReentered = false;
    UsdGenGenerationStore publicationStore;
    {
        auto old = DevicePayload(80, std::make_shared<ReentrantDomainOwner>([&] {
            publicationOwnerReentered =
                static_cast<bool>(domain->Lookup(refreshedDomainKey));
        }));
        UsdGenGeneration mutableOld = *old;
        old.reset();
        publicationStore.Publish(std::move(mutableOld));
    }
    UsdGenGeneration replacementGeneration;
    replacementGeneration.frame = 92.0;
    auto preparedReplacement =
        publicationStore.PreparePublication(std::move(replacementGeneration));
    Check(domain->PublishIfCurrent(*currentPublicationFence,
              [&] {
                  return publicationStore.PublishPrepared(
                      std::move(preparedReplacement));
              }) && publicationOwnerReentered && publicationStore.Get() &&
              publicationStore.Get()->id == 1,
          "fenced generation publication retires the displaced COW owner after the domain lock");
    domainLease.reset();

    auto acquiredDomain = Domain::AcquireShared(
        {UsdGenDeviceBackend::CpuReference, -1, 0x5245474953545259ull}, 128);
    auto sameAcquiredDomain = Domain::AcquireShared(
        {UsdGenDeviceBackend::CpuReference, -1, 0x5245474953545259ull}, 128);
    auto conflictingBudgetDomain = Domain::AcquireShared(
        {UsdGenDeviceBackend::CpuReference, -1, 0x5245474953545259ull}, 64);
    auto otherContextDomain = Domain::AcquireShared(
        {UsdGenDeviceBackend::CpuReference, -1, 0x524547495354525aull}, 128);
    Check(acquiredDomain && acquiredDomain == sameAcquiredDomain &&
              !conflictingBudgetDomain && otherContextDomain &&
              otherContextDomain != acquiredDomain,
          "shared-domain registry coalesces only exact backend/device/context and budget identity");

    // The immutable store already retires COW states outside its own mutex.
    // A shared domain adds another serialization mutex around epoch checks;
    // displaced native owners must outlive that outer lock as well because an
    // owner destructor may inspect the same domain during backend retirement.
    auto reentrantDomain = std::make_shared<Domain>(
        UsdGenExecutionCacheDomainKey{
            UsdGenDeviceBackend::CpuReference, 0, 0x5245454e5452414eull}, 1);
    UsdGenExecutionCacheKey reentrantDomainKey = keyA;
    reentrantDomainKey.context = {
        UsdGenDeviceBackend::CpuReference, 1, 0, reentrantDomain->Epoch(),
        UsdGenContext::Interactive};
    std::weak_ptr<Domain> weakReentrantDomain = reentrantDomain;
    std::atomic<int> reentrantDomainDestructions{0};
    auto firstReentrantGeneration = DevicePayload(80,
        std::make_shared<ReentrantDomainOwner>(
            [weakReentrantDomain, reentrantDomainKey,
             &reentrantDomainDestructions] {
                if (auto locked = weakReentrantDomain.lock())
                    (void)locked->Lookup(reentrantDomainKey);
                ++reentrantDomainDestructions;
            }));
    Check(firstReentrantGeneration &&
              reentrantDomain->Insert(reentrantDomainKey,
                  firstReentrantGeneration, 1, &reason),
          "shared-domain retirement fixture admits a native COW owner");
    firstReentrantGeneration.reset();
    auto replacementDomainKey = reentrantDomainKey;
    replacementDomainKey.layoutDigest += 1;
    Check(reentrantDomain->Insert(replacementDomainKey,
              DevicePayload(81, std::make_shared<ReentrantDomainOwner>(
                  std::function<void()>{})), 1, &reason) &&
              reentrantDomainDestructions.load() == 1,
          "domain eviction destroys a reentrant native owner outside the outer lock");
    auto invalidatedDomainKey = replacementDomainKey;
    invalidatedDomainKey.layoutDigest += 1;
    auto invalidatedGeneration = DevicePayload(82,
        std::make_shared<ReentrantDomainOwner>(
            [weakReentrantDomain, invalidatedDomainKey,
             &reentrantDomainDestructions] {
                if (auto locked = weakReentrantDomain.lock())
                    (void)locked->Lookup(invalidatedDomainKey);
                ++reentrantDomainDestructions;
            }));
    Check(invalidatedGeneration && reentrantDomain->Insert(
              invalidatedDomainKey, invalidatedGeneration, 1, &reason),
          "domain invalidation fixture replaces the eviction payload");
    invalidatedGeneration.reset();
    Check(reentrantDomain->Invalidate() &&
              reentrantDomainDestructions.load() == 2,
          "domain invalidation destroys a reentrant native owner outside the outer lock");

    UsdGenExecutionCacheStore<int, ConstantHasher> collisionStore(
        16, ConstantHasher{});
    Check(collisionStore.Insert(keyA, payload(41), 4, &reason) &&
              collisionStore.Insert(keyB, payload(42), 4, &reason),
          "colliding hashes admit distinct full keys");
    auto collisionHit = collisionStore.Lookup(keyB);
    Check(collisionHit && **collisionHit == 42,
          "full key equality resolves hash collisions authoritatively");
    Check(!store.Insert(UsdGenExecutionCacheKey{}, payload(0), 1, &reason) &&
              reason == "execution cache key is invalid",
          "invalid cache keys are rejected precisely");
    Check(!store.Insert(keyA, payload(0), 0, &reason) &&
              reason == "execution cache byte charge is zero",
          "zero-byte entries cannot bypass the bounded-store contract");

    std::atomic<int> reentrantDestructions{0};
    UsdGenExecutionCacheStore<ReentrantPayload> reentrant(1);
    auto reentrantPayload = std::make_shared<const ReentrantPayload>(
        [&] {
            (void)reentrant.Bytes();
            ++reentrantDestructions;
        });
    Check(reentrant.Insert(keyA, reentrantPayload, 1, &reason),
          "reentrant-destruction fixture inserts first payload");
    reentrantPayload.reset();
    Check(reentrant.Insert(keyB,
              std::make_shared<const ReentrantPayload>(), 1, &reason) &&
              reentrantDestructions.load() == 1,
          "evicted payload destruction occurs after releasing the cache lock");

    auto stale = keyA;
    stale.inputs.UpdateMap(desc.maps.front().path, 999);
    Check(!replacement.Lookup(stale),
          "stale input-generation tuple does not hit a cached value");
    auto otherContext = keyA;
    otherContext.context.deviceIndex = keyA.context.deviceIndex + 1;
    Check(!replacement.Lookup(otherContext),
          "incompatible device context does not hit a cached value");

    Store concurrent(128);
    std::vector<std::thread> workers;
    std::atomic<int> inserts{0};
    for (int thread = 0; thread != 4; ++thread) {
        workers.emplace_back([&, thread] {
            for (int i = 0; i != 64; ++i) {
                auto concurrentKey = keyA;
                concurrentKey.layoutDigest =
                    static_cast<uint64_t>(thread * 1000 + i);
                if (concurrent.Insert(concurrentKey, payload(i), 1)) ++inserts;
                (void)concurrent.Lookup(concurrentKey);
            }
        });
    }
    for (auto& worker : workers) worker.join();
    Check(inserts == 256 && concurrent.Bytes() <= concurrent.MaxBytes(),
          "concurrent lookup/insert preserves bounded cache accounting");

    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = compiler.Compile(desc, &graph);
    Check(compiled.ok, "cache dirty graph compiles");
    if (!compiled.ok) {
        for (auto const& error : compiled.errors)
            std::fprintf(stderr, "compile error: %s\n", error.c_str());
        return failures ? 1 : 0;
    }
    Check(graph.InputVersions().maps.front().generation == 13,
          "compiled graph captures immutable input versions");
    graph.DirtyMap(desc.maps.front().path, 14);
    auto sourceNode = graph.NodeIdForPath(
        SdfPath("/cache/groomA/source"));
    auto mapNode = graph.NodeIdForPath(desc.terminal);
    Check(graph.InputVersions().maps.front().generation == 14 &&
              graph.Node(sourceNode).captureNeeded &&
              graph.Node(mapNode).captureNeeded,
          "map dirty updates version and invalidates consumer and descendant captures");
    Check(graph.AnyDirty(), "map dirty remains observable");
    graph.DirtyCurves(desc.curveSets[2].path, 10);
    Check(graph.InputVersions().references.front().generation == 10 &&
              graph.Node(mapNode).captureNeeded,
          "curve dirty updates reference generation without clearing existing dirt");
    graph.DirtySurface(0, UsdGenDirtySurfacePoints, 42);
    Check(graph.InputVersions().surfaces.front().generation == 42,
          "surface dirty updates exact surface generation");

    // The cooker integration defers admission until its owner accepts the
    // publication. A direct cooker fixture makes that boundary observable
    // without depending on a renderer callback or a CUDA device.
    UsdGenSessionCooker cooker(2, 1024 * 1024);
    auto cookedDesc = std::make_shared<const UsdGenGraphDesc>(desc);
    UsdGenGenerationConstPtr first = cooker.Cook(
        cookedDesc, UsdGenContext::Interactive, false, {}, 1.0,
        UsdGenCommitReason::NoticeBatchEnd, {}, {}, false, 0, 1, -1);
    Check(first && first->id == 0 &&
              cooker.Stats().executionCacheMisses == 1,
          "CPU cooker publishes an initial cache miss");
    auto firstCandidate = cooker.TakeCacheCandidate();
    Check(firstCandidate && cooker.ExecutionCacheSize() == 0,
          "successful cooker output remains deferred before owner acceptance");
    if (firstCandidate)
        Check(cooker.CommitCacheCandidate(*firstCandidate) &&
                  cooker.ExecutionCacheSize() == 1 && cooker.ExecutionCacheBytes() > 0 &&
                  cooker.Stats().executionCacheAdmissions == 1,
              "accepted cooker publication seeds bounded immutable cache");
    UsdGenStats const afterAdmission = cooker.Stats();
    UsdGenGenerationConstPtr second = cooker.Cook(
        cookedDesc, UsdGenContext::Interactive, false, {}, 1.0,
        UsdGenCommitReason::NoticeBatchEnd, first, afterAdmission, false, 1, 2, -1);
    Check(second && second->id == 1 && !cooker.TakeCacheCandidate() &&
              cooker.Stats().executionCacheHits == 1 &&
              cooker.Stats().executionCacheMisses == 1,
          "exact CPU cache hit republishes a fresh generation without reseeding");

    // Independent cookers can share only a completed-result domain.  The
    // follower must publish a fresh host generation, but the VtArray payload
    // remains the immutable COW data retained by the leader's accepted entry.
    auto sharedCpuDomain = std::make_shared<UsdGenExecutionCacheDomain>(
        UsdGenExecutionCacheDomainKey{
            UsdGenDeviceBackend::CpuReference, -1, 0x435055434f57ull},
        1024 * 1024);
    UsdGenSessionCooker leaderCooker(2, 1024 * 1024, sharedCpuDomain);
    UsdGenSessionCooker followerCooker(2, 1024 * 1024, sharedCpuDomain);
    auto sharedLeader = leaderCooker.Cook(
        cookedDesc, UsdGenContext::Interactive, false, {}, 6.0,
        UsdGenCommitReason::NoticeBatchEnd, {}, {}, false, 0, 1, -1);
    auto sharedLeaderCandidate = leaderCooker.TakeCacheCandidate();
    Check(sharedLeader && !sharedLeader->tiles.empty() && sharedLeaderCandidate &&
              leaderCooker.CommitCacheCandidate(*sharedLeaderCandidate) &&
              sharedCpuDomain->Size() == 1,
          "leader accepted publication seeds its explicit shared CPU domain");
    if (!sharedLeader || sharedLeader->tiles.empty() || !sharedLeaderCandidate)
        return 1;
    auto sharedFollower = followerCooker.Cook(
        cookedDesc, UsdGenContext::Interactive, false, {}, 6.0,
        UsdGenCommitReason::NoticeBatchEnd, {}, {}, false, 0, 1, -1);
    Check(sharedFollower && sharedFollower != sharedLeader &&
              sharedFollower->tiles.size() == sharedLeader->tiles.size() &&
              sharedFollower->tiles.front().points.IsIdentical(
                  sharedLeader->tiles.front().points) &&
              !followerCooker.TakeCacheCandidate() &&
              followerCooker.Stats().executionCacheHits == 1 &&
              followerCooker.Stats().executionCacheMisses == 0,
          "exact shared CPU tuple creates a fresh wrapper over immutable COW payload data");
    auto oldEpochOutput = leaderCooker.Cook(
        cookedDesc, UsdGenContext::Interactive, false, {}, 7.0,
        UsdGenCommitReason::NoticeBatchEnd, sharedLeader, leaderCooker.Stats(),
        false, 1, 2, -1);
    auto oldEpochCandidate = leaderCooker.TakeCacheCandidate();
    auto oldEpochPublication = leaderCooker.TakePublicationFence();
    bool staleSessionPublicationRan = false;
    Check(oldEpochOutput && oldEpochCandidate && oldEpochPublication &&
              sharedCpuDomain->Invalidate() &&
              !leaderCooker.PublishIfCurrent(*oldEpochPublication, [&] {
                  staleSessionPublicationRan = true;
                  return std::shared_ptr<const void>{};
              }) && !staleSessionPublicationRan &&
              !leaderCooker.CommitCacheCandidate(*oldEpochCandidate) &&
              sharedCpuDomain->Size() == 0,
          "domain invalidation rejects both a completed cooker's deferred session action and cache admission");
    auto freshEpochOutput = leaderCooker.Cook(
        cookedDesc, UsdGenContext::Interactive, false, {}, 7.0,
        UsdGenCommitReason::NoticeBatchEnd, oldEpochOutput, leaderCooker.Stats(),
        false, 1, 3, -1);
    auto freshEpochPublication = leaderCooker.TakePublicationFence();
    bool freshSessionPublicationRan = false;
    Check(freshEpochOutput && freshEpochPublication &&
              leaderCooker.PublishIfCurrent(*freshEpochPublication, [&] {
                  freshSessionPublicationRan = true;
                  return std::shared_ptr<const void>{};
              }) && freshSessionPublicationRan,
          "a fresh cooker handoff fence permits the bounded session publication action");
    leaderCooker.TakeCacheCandidate();

    // The hit above did not evaluate the cooker's mutable graph buffers.  A
    // following cook may hit again, but preparation must first rebuild the
    // graph baseline so a later miss cannot incrementally reuse stale data.
    UsdGenStats afterFirstHit = cooker.Stats();
    UsdGenGenerationConstPtr rebuiltHit = cooker.Cook(
        cookedDesc, UsdGenContext::Interactive, false, {}, 1.0,
        UsdGenCommitReason::NoticeBatchEnd, second, afterFirstHit, false,
        2, 3, -1);
    Check(rebuiltHit && rebuiltHit->id == 2 &&
              cooker.Stats().recompiles == afterFirstHit.recompiles + 1 &&
              !cooker.TakeCacheCandidate(),
          "cache-hit publication detaches and rebuilds mutable graph baseline");

    UsdGenGraphDesc valueChanged = desc;
    valueChanged.nodes[1].params.push_back(
        {TfToken("width"), VtValue(0.02f), false});
    auto valueChangedDesc = std::make_shared<const UsdGenGraphDesc>(
        std::move(valueChanged));
    UsdGenGenerationConstPtr third = cooker.Cook(
        valueChangedDesc, UsdGenContext::Interactive, false, {}, 1.0,
        UsdGenCommitReason::NoticeBatchEnd, rebuiltHit, {}, false, 3, 4, -1);
    auto changedCandidate = cooker.TakeCacheCandidate();
    Check(third && changedCandidate && cooker.ExecutionCacheSize() == 1,
          "value-parameter change misses exact plan identity and defers new entry");
    Check(firstCandidate && !cooker.CommitCacheCandidate(*firstCandidate) &&
              cooker.ExecutionCacheSize() == 1 &&
              cooker.Stats().executionCacheAdmissionFailures == 1,
          "superseded generation cannot publish an older cache candidate");

    UsdGenGraphDesc mapChanged = desc;
    mapChanged.maps.front().params.push_back(
        {TfToken("gain"), VtValue(2.0f), false});
    auto mapChangedDesc = std::make_shared<const UsdGenGraphDesc>(
        std::move(mapChanged));
    UsdGenGenerationConstPtr mapResult = cooker.Cook(
        mapChangedDesc, UsdGenContext::Interactive, false, {}, 1.0,
        UsdGenCommitReason::NoticeBatchEnd, third, {}, false, 4, 5, -1);
    Check(mapResult && cooker.TakeCacheCandidate(),
          "resolved map parameter change misses exact cache identity");

    UsdGenGraphDesc densityChanged = desc;
    densityChanged.surfaces.front().densityMultiplier = VtFloatArray{0.0f};
    auto densityChangedDesc = std::make_shared<const UsdGenGraphDesc>(
        std::move(densityChanged));
    UsdGenGenerationConstPtr densityResult = cooker.Cook(
        densityChangedDesc, UsdGenContext::Interactive, false, {}, 1.0,
        UsdGenCommitReason::NoticeBatchEnd, mapResult, {}, false, 5, 6, -1);
    Check(densityResult && cooker.TakeCacheCandidate(),
          "paint-density change misses exact cache identity (no stale roots)");

    UsdGenGraphDesc paintBase = desc;
    UsdGenMapDesc paint;
    paint.path = SdfPath("/cache/groomA/lengthPaint");
    paint.type = TfToken("UsdGenPaintMap");
    paint.paintSurface = paintBase.surfaces.front().path;
    paint.paintPrimvar = TfToken("usdGen:paint:length");
    paint.paintValues = VtFloatArray{1.0f, 1.0f};
    paintBase.maps.push_back(paint);
    auto paintBaseDesc = std::make_shared<const UsdGenGraphDesc>(paintBase);
    UsdGenGenerationConstPtr paintResult = cooker.Cook(
        paintBaseDesc, UsdGenContext::Interactive, false, {}, 1.0,
        UsdGenCommitReason::NoticeBatchEnd, densityResult, {}, false,
        6, 7, -1);
    Check(paintResult && cooker.TakeCacheCandidate(),
          "paint-map snapshot addition misses exact cache identity");

    UsdGenGraphDesc paintChanged = paintBase;
    paintChanged.maps.back().paintValues = VtFloatArray{0.0f, 0.0f};
    auto paintChangedDesc = std::make_shared<const UsdGenGraphDesc>(
        std::move(paintChanged));
    UsdGenGenerationConstPtr paintChangedResult = cooker.Cook(
        paintChangedDesc, UsdGenContext::Interactive, false, {}, 1.0,
        UsdGenCommitReason::NoticeBatchEnd, paintResult, {}, false,
        7, 8, -1);
    Check(paintChangedResult && cooker.TakeCacheCandidate(),
          "paint-map repaint misses exact cache identity (no stale groom)");

    // Big-key plan identity: a surface whose hashed arrays exceed the 1MB
    // threaded-digest gate must still hit on an exact recook and miss when
    // a single hashed index flips (the parallel segment digests fold back
    // in walk order; dropping or misordering a term breaks one of these).
    {
        int const nx = 300, ny = 300;
        UsdGenGraphDesc big;
        big.description = SdfPath("/cache/bigKey");
        big.terminal = SdfPath("/cache/bigKey/scatter");
        UsdGenSurfaceDesc surf;
        surf.path = SdfPath("/cache/bigKey/scalp");
        surf.restPoints = VtVec3fArray((nx + 1) * (ny + 1));
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i)
                surf.restPoints[j * (nx + 1) + i] =
                    GfVec3f(0.1f * i, 0.1f * j, 0.0f);
        surf.faceVertexCounts = VtIntArray(nx * ny, 4);
        surf.faceVertexIndices = VtIntArray(nx * ny * 4);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int const a = j * (nx + 1) + i;
                int const o = (j * nx + i) * 4;
                surf.faceVertexIndices[o + 0] = a;
                surf.faceVertexIndices[o + 1] = a + 1;
                surf.faceVertexIndices[o + 2] = a + nx + 2;
                surf.faceVertexIndices[o + 3] = a + nx + 1;
            }
        big.surfaces.push_back(std::move(surf));
        UsdGenNodeDesc scatter;
        scatter.path = SdfPath("/cache/bigKey/scatter");
        scatter.type = TfToken("UsdGenScatter");
        scatter.enabled = true;
        scatter.seed = 7;
        scatter.surfaces.push_back(SdfPath("/cache/bigKey/scalp"));
        big.nodes.push_back(std::move(scatter));
        auto bigDesc =
            std::make_shared<const UsdGenGraphDesc>(std::move(big));
        UsdGenSessionCooker bigCooker(0, 256u * 1024u * 1024u);
        UsdGenGenerationConstPtr bigFirst = bigCooker.Cook(
            bigDesc, UsdGenContext::Interactive, false, {}, 0.0,
            UsdGenCommitReason::NoticeBatchEnd, {}, {}, false, 0, 1, -1);
        auto bigCandidate = bigCooker.TakeCacheCandidate();
        Check(bigFirst && !bigFirst->tiles.empty() && bigCandidate,
              "big-key cook publishes a cache candidate");
        if (bigCandidate)
            bigCooker.CommitCacheCandidate(*bigCandidate);
        // The first cook's full compile moves the key before publication
        // (the lookup key predates the compiled node digests), so
        // admission lands on the second identical cook; the third hits.
        UsdGenGenerationConstPtr bigSecond = bigCooker.Cook(
            bigDesc, UsdGenContext::Interactive, false, {}, 0.0,
            UsdGenCommitReason::NoticeBatchEnd, bigFirst, bigCooker.Stats(),
            false, 1, 2, -1);
        auto bigCandidate2 = bigCooker.TakeCacheCandidate();
        Check(bigSecond && bigCandidate2 &&
                  bigCooker.CommitCacheCandidate(*bigCandidate2) &&
                  bigCooker.ExecutionCacheSize() == 1,
              "second big-key cook admits its cache entry");
        UsdGenGenerationConstPtr bigThird = bigCooker.Cook(
            bigDesc, UsdGenContext::Interactive, false, {}, 0.0,
            UsdGenCommitReason::NoticeBatchEnd, bigSecond, bigCooker.Stats(),
            false, 2, 3, -1);
        Check(bigThird && !bigCooker.TakeCacheCandidate() &&
                  bigCooker.Stats().executionCacheHits == 1,
              "exact big-key recook hits (threaded digest is deterministic)");
        UsdGenGraphDesc bigFlipped = *bigDesc;
        bigFlipped.surfaces.front().faceVertexIndices[12345] ^= 1;
        auto bigFlippedDesc =
            std::make_shared<const UsdGenGraphDesc>(std::move(bigFlipped));
        UsdGenGenerationConstPtr bigFourth = bigCooker.Cook(
            bigFlippedDesc, UsdGenContext::Interactive, false, {}, 0.0,
            UsdGenCommitReason::NoticeBatchEnd, bigThird, bigCooker.Stats(),
            false, 3, 4, -1);
        Check(bigFourth && bigCooker.TakeCacheCandidate() &&
                  bigCooker.Stats().executionCacheHits == 1,
              "single-index flip misses big-key identity (no stale groom)");
    }
    // An entry larger than the whole domain budget can never be admitted,
    // so the cooker refuses it up front instead of re-hashing the key — and
    // a coalesced leader flight still settles (followers woken Failed with
    // no payload) rather than leaking an open flight.
    auto tinyDomain = std::make_shared<Domain>(
        UsdGenExecutionCacheDomainKey{
            UsdGenDeviceBackend::CpuReference, -1, 0x54494e59ull},
        1);
    UsdGenSessionCooker tinyLeader(2, 1, tinyDomain);
    UsdGenSessionCooker tinyFollower(2, 1, tinyDomain);
    Domain::CoalescedStatus tinyFollowerStatus =
        Domain::CoalescedStatus::Success;
    bool tinyFollowerLease = true;
    int tinyFollowerCallbacks = 0;
    UsdGenSessionCooker::CoalescedHooks tinyFollowerHooks;
    tinyFollowerHooks.callback = [&](
        Domain::CoalescedStatus status,
        std::optional<UsdGenSessionCooker::CoalescedLease> lease,
        UsdGenSessionCooker::ExecutionPublicationFence) {
        tinyFollowerStatus = status;
        tinyFollowerLease = bool(lease);
        ++tinyFollowerCallbacks;
    };
    UsdGenSessionCooker::CoalescedHooks tinyLeaderHooks;
    tinyLeaderHooks.callback = [](
        Domain::CoalescedStatus,
        std::optional<UsdGenSessionCooker::CoalescedLease>,
        UsdGenSessionCooker::ExecutionPublicationFence) {};
    auto tinyOutput = tinyLeader.Cook(
        cookedDesc, UsdGenContext::Interactive, false, {}, 9.0,
        UsdGenCommitReason::NoticeBatchEnd, {}, {}, false, 0, 1, -1,
        tinyLeaderHooks);
    tinyFollower.Cook(
        cookedDesc, UsdGenContext::Interactive, false, {}, 9.0,
        UsdGenCommitReason::NoticeBatchEnd, {}, {}, false, 0, 1, -1,
        tinyFollowerHooks);
    auto tinyCandidate = tinyLeader.TakeCacheCandidate();
    Check(tinyOutput && tinyCandidate && tinyCandidate->bytes > 1,
          "tiny-budget leader still cooks and defers an over-budget candidate");
    uint64_t tinyFailuresBefore =
        tinyLeader.Stats().executionCacheAdmissionFailures;
    bool const tinyCommitted = tinyCandidate &&
        tinyLeader.CommitCacheCandidate(*tinyCandidate);
    Check(!tinyCommitted && tinyDomain->Size() == 0 &&
              tinyLeader.Stats().executionCacheAdmissionFailures ==
                  tinyFailuresBefore + 1,
          "over-budget candidate is refused without seeding the cache");
    Check(tinyFollowerCallbacks == 1 &&
              tinyFollowerStatus == Domain::CoalescedStatus::Failed &&
              !tinyFollowerLease,
          "over-budget leader flight settles followers Failed with no payload");
    auto tinyAgain = tinyLeader.Cook(
        cookedDesc, UsdGenContext::Interactive, false, {}, 9.0,
        UsdGenCommitReason::NoticeBatchEnd, tinyOutput,
        tinyLeader.Stats(), false, 1, 2, -1, tinyLeaderHooks);
    Check(tinyAgain && tinyLeader.TakeCacheCandidate(),
          "settled over-budget flight leaves no zombie for the next cook");

    if (failures) return 1;
    std::puts("ok");
    return 0;
}
