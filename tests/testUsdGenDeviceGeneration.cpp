#include "utils/pxrInterop.h"
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "testBackend.h"
#include "usdGen/exec/device/deviceGeneration.h"

#include <cstdio>
#include <memory>
#include <string>

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

class FakeOwner;

class FakeConsumer final : public usdGen::UsdGenDeviceConsumer
{
public:
    FakeConsumer(FakeOwner const *owner, usdGen::UsdGenDeviceStream stream)
        : owner_(owner), stream_(stream) {}
    ~FakeConsumer() override { Complete(); }
    usdGen::UsdGenDeviceStream Stream() const noexcept override { return stream_; }
    usdGen::UsdGenDeviceStatus WaitUntilReady() const noexcept override;
    void Complete() noexcept override;
private:
    FakeOwner const *owner_;
    usdGen::UsdGenDeviceStream stream_;
    bool completed_ = false;
};

class FakeOwner final : public usdGen::UsdGenDeviceOwner
{
public:
    bool ProducerReady() const noexcept override { return ready; }
    size_t ExclusiveRetainedBytes() const noexcept override { return retainedBytes; }
    std::unique_ptr<usdGen::UsdGenDeviceConsumer> AcquireConsumer(
        usdGen::UsdGenDeviceStream stream) const noexcept override;

    bool ready = false;
    bool reject = false;
    bool wrongStream = false;
    size_t retainedBytes = 4096;
    mutable int active = 0;
    mutable int acquired = 0;
    mutable int rejected = 0;
    mutable int waits = 0;
    mutable int released = 0;
};

std::unique_ptr<usdGen::UsdGenDeviceConsumer>
FakeOwner::AcquireConsumer(usdGen::UsdGenDeviceStream stream) const noexcept
{
    if (reject) {
        ++rejected;
        return {};
    }
    try {
        auto consumer = std::make_unique<FakeConsumer>(
            this, wrongStream ? stream + 1 : stream);
        ++active;
        ++acquired;
        return consumer;
    } catch (...) {
        ++rejected;
        return {};
    }
}

usdGen::UsdGenDeviceStatus FakeConsumer::WaitUntilReady() const noexcept
{
    if (!owner_ || completed_) return usdGen::UsdGenDeviceStatus::InvalidLease;
    ++owner_->waits;
    return owner_->ready ? usdGen::UsdGenDeviceStatus::Ok
                         : usdGen::UsdGenDeviceStatus::ProducerNotReady;
}

void FakeConsumer::Complete() noexcept
{
    if (completed_) return;
    completed_ = true;
    if (owner_ && owner_->active > 0) {
        --owner_->active;
        ++owner_->released;
    }
}

int Check(bool condition, char const *message, int *failures)
{
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++*failures;
    }
    return condition ? 0 : 1;
}

} // namespace

int main()
{
    int failures = 0;
    auto owner = std::make_shared<FakeOwner>();
    std::weak_ptr<FakeOwner> weakOwner = owner;

    usdGen::UsdGenDeviceGeneration::CreateInfo info;
    info.identity = {usdGenTestUtils::kTestBackendA, 2, 17};
    info.geometry.topologyVersion = 4;
    info.geometry.valueVersion = 9;
    info.geometry.elementCount = 3;
    info.geometry.pointCount = 12;
    info.geometry.tiles.push_back({7, 0, 3, 0, 12});
    info.tool = {"brushA", 21, 17};
    info.channels.push_back({TfToken("points"), usdGen::UsdGenDeviceValueType::Float32x3,
                             usdGen::UsdGenDeviceDomain::Point, 12, 3, 12, true,
                             usdGen::UsdGenDeviceChannelSemantic::Points});
    info.channels.push_back({TfToken("width"), usdGen::UsdGenDeviceValueType::Float32,
                             usdGen::UsdGenDeviceDomain::Point, 12, 1, 4, true,
                             usdGen::UsdGenDeviceChannelSemantic::Widths});
    info.channels.push_back({TfToken("curveOffsets"), usdGen::UsdGenDeviceValueType::UInt32,
                             usdGen::UsdGenDeviceDomain::Topology, 4, 1, 4, true,
                             usdGen::UsdGenDeviceChannelSemantic::ElementOffsets});
    info.owner = owner;

    std::string reason;
    Check(usdGen::ValidateUsdGenDeviceMetadata(info.geometry, info.channels, &reason),
          "backend preflight validates metadata without a native owner", &failures);
    {
        auto channel = info.channels[0];
        size_t bytes = 17;
        Check(usdGen::UsdGenDeviceChannelStorageBytes(channel, &bytes, &reason) && bytes == 144,
              "packed channel byte span is exact", &failures);
        channel.strideBytes = 16;
        Check(usdGen::UsdGenDeviceChannelStorageBytes(channel, &bytes, &reason) && bytes == 188,
              "strided byte span excludes unused trailing padding", &failures);
        channel.elementCount = 0;
        Check(usdGen::UsdGenDeviceChannelStorageBytes(channel, &bytes, &reason) && bytes == 0,
              "empty typed channel has zero byte span", &failures);
        channel.elementCount = UINT64_MAX; bytes = 17;
        Check(!usdGen::UsdGenDeviceChannelStorageBytes(channel, &bytes, &reason) && bytes == 17,
              "overflow rejects without changing byte-span output", &failures);
        auto shape = info.geometry; shape.pointCount = UINT64_MAX;
        Check(!usdGen::ValidateUsdGenDeviceMetadata(shape, {channel}, &reason),
              "metadata admission rejects unrepresentable payload storage", &failures);
        channel.elementCount = 1; channel.arity = 0;
        Check(!usdGen::UsdGenDeviceChannelStorageBytes(channel, &bytes, &reason),
              "zero arity rejects before span arithmetic", &failures);
        channel.arity = 3; channel.type = static_cast<usdGen::UsdGenDeviceValueType>(255);
        Check(!usdGen::UsdGenDeviceChannelStorageBytes(channel, &bytes, &reason),
              "unknown value type is not silently coerced", &failures);
    }
    std::shared_ptr<const usdGen::UsdGenDeviceGeneration> generation =
        usdGen::UsdGenDeviceGeneration::Create(std::move(info), &reason);
    Check(bool(generation), "valid device generation creates", &failures);
    Check(generation && generation->Identity().deviceIndex == 2,
          "identity is retained", &failures);
    Check(generation && generation->Geometry().tiles.size() == 1,
          "tile metadata is retained", &failures);
    Check(generation && generation->Tool().toolId == "brushA",
          "tool snapshot is retained", &failures);

    // The immutable handoff contract names portable backend identities; native
    // synchronization remains entirely inside each backend owner.
    for (auto backend : {usdGenTestUtils::kTestBackendB,
                         usdGenTestUtils::kTestBackendA}) {
        usdGen::UsdGenDeviceGeneration::CreateInfo portable;
        portable.identity = {backend, 0, 1};
        portable.geometry = generation->Geometry();
        portable.tool = generation->Tool();
        portable.channels = generation->Channels();
        portable.owner = owner;
        auto candidate = usdGen::UsdGenDeviceGeneration::Create(
            std::move(portable), &reason);
        Check(candidate && candidate->Identity().backend == backend,
              "Metal/Vulkan identities use the backend-neutral generation contract",
              &failures);
    }

    // Legacy range-only snapshots retain the all-Unknown topology contract;
    // a partially supplied topology is never ambiguous for a device renderer.
    usdGen::UsdGenDeviceGeneration::CreateInfo legacyInfo;
    legacyInfo.identity = generation->Identity();
    legacyInfo.geometry = generation->Geometry();
    legacyInfo.tool = generation->Tool();
    legacyInfo.channels = generation->Channels();
    legacyInfo.owner = generation->Owner();
    Check(bool(usdGen::UsdGenDeviceGeneration::Create(std::move(legacyInfo), &reason)),
          "all-Unknown curve topology preserves legacy generation admission", &failures);
    usdGen::UsdGenDeviceGeneration::CreateInfo genericArrays;
    genericArrays.identity = generation->Identity();
    genericArrays.geometry = generation->Geometry();
    genericArrays.tool = generation->Tool();
    genericArrays.channels = generation->Channels();
    genericArrays.channels.push_back(
        {TfToken("guideIndex"), usdGen::UsdGenDeviceValueType::Int32,
         usdGen::UsdGenDeviceDomain::Primitive, 3, 3, 12, true,
         usdGen::UsdGenDeviceChannelSemantic::Generic});
    genericArrays.channels.push_back(
        {TfToken("guideWeight"), usdGen::UsdGenDeviceValueType::Float32,
         usdGen::UsdGenDeviceDomain::Primitive, 3, 3, 12, true,
         usdGen::UsdGenDeviceChannelSemantic::Generic});
    genericArrays.owner = generation->Owner();
    auto genericGeneration = usdGen::UsdGenDeviceGeneration::Create(
        std::move(genericArrays), &reason);
    Check(genericGeneration && genericGeneration->Channels().size() == 5 &&
              genericGeneration->Channels()[3].arity == 3 &&
              genericGeneration->Channels()[4].arity == 3,
          "generic scalar arrays preserve elementSize-three metadata", &failures);
    usdGen::UsdGenDeviceGeneration::CreateInfo badGenericStride;
    badGenericStride.identity = generation->Identity();
    badGenericStride.geometry = generation->Geometry();
    badGenericStride.tool = generation->Tool();
    badGenericStride.channels = generation->Channels();
    badGenericStride.channels.push_back(
        {TfToken("guideIndex"), usdGen::UsdGenDeviceValueType::Int32,
         usdGen::UsdGenDeviceDomain::Primitive, 3, 3, 8, true,
         usdGen::UsdGenDeviceChannelSemantic::Generic});
    badGenericStride.owner = generation->Owner();
    Check(!usdGen::UsdGenDeviceGeneration::Create(
              std::move(badGenericStride), &reason),
          "generic scalar arrays reject stride below elementSize", &failures);
    usdGen::UsdGenDeviceGeneration::CreateInfo mixedTopology;
    mixedTopology.identity = generation->Identity();
    mixedTopology.geometry = generation->Geometry();
    mixedTopology.geometry.layoutMetadata.kind = uint8_t(usdGen::UsdGenCurveLayoutType::Cubic);
    mixedTopology.tool = generation->Tool();
    mixedTopology.channels = generation->Channels();
    mixedTopology.owner = generation->Owner();
    Check(!usdGen::UsdGenDeviceGeneration::Create(std::move(mixedTopology), &reason),
          "mixed Unknown and concrete curve topology is rejected", &failures);

    auto invalid = usdGen::UsdGenDeviceGeneration::Create(
        usdGen::UsdGenDeviceGeneration::CreateInfo{}, &reason);
    Check(!invalid && !reason.empty(), "invalid generation is rejected", &failures);

    usdGen::UsdGenDeviceGeneration::CreateInfo badShape;
    badShape.identity = {usdGenTestUtils::kTestBackendA, 2, 18};
    badShape.geometry.elementCount = 3;
    badShape.geometry.pointCount = 12;
    badShape.owner = owner;
    badShape.channels.push_back(
        {TfToken("width"), usdGen::UsdGenDeviceValueType::Float32,
         usdGen::UsdGenDeviceDomain::Point, 11, 1, 4, true,
         usdGen::UsdGenDeviceChannelSemantic::Widths});
    Check(!usdGen::UsdGenDeviceGeneration::Create(std::move(badShape), &reason),
          "semantic channel count is validated", &failures);

    usdGen::UsdGenDeviceLease lease = generation->AcquireLease(0x1234);
    Check(lease.IsValid() && owner->active == 1,
          "consumer lease acquires owner use", &failures);
    usdGen::UsdGenDeviceLease secondLease = lease;
    lease.Complete();
    Check(!lease.IsValid() && secondLease.IsValid() && owner->active == 1,
          "completing one copy does not release shared backend use", &failures);
    Check(secondLease.WaitUntilReady() == usdGen::UsdGenDeviceStatus::ProducerNotReady,
          "consumer cannot use an unfinished producer", &failures);
    owner->ready = true;
    Check(owner->ProducerReady(), "owner exposes producer readiness", &failures);
    Check(generation->ExclusiveRetainedBytes() == owner->retainedBytes,
          "generation exposes backend-owner-exclusive cache bytes", &failures);
    auto republished = generation->Republish(18, &reason);
    Check(republished && republished->Identity().generation == 18 &&
              republished->Geometry().valueVersion == 18 &&
              republished->Geometry().topologyVersion ==
                  generation->Geometry().topologyVersion &&
              republished->Owner() == generation->Owner(),
          "cache republish creates a fresh identity over shared immutable COW owner",
          &failures);
    Check(!generation->Republish(17, &reason),
          "cache republish rejects a recycled device generation identity", &failures);
    auto immutableAlias = generation->RepublishPreservingRevisions(19, &reason);
    Check(immutableAlias && immutableAlias->Identity().generation == 19 &&
              immutableAlias->Geometry().valueVersion == generation->Geometry().valueVersion &&
              immutableAlias->Geometry().topologyVersion == generation->Geometry().topologyVersion &&
              immutableAlias->Owner() == generation->Owner(),
          "immutable republish preserves payload revisions and COW owner", &failures);
    Check(!generation->RepublishPreservingRevisions(17, &reason),
          "immutable republish rejects recycled publication identity", &failures);
    Check(secondLease.WaitUntilReady() == usdGen::UsdGenDeviceStatus::Ok,
          "lease inserts producer ordering after readiness", &failures);

    owner->reject = true;
    usdGen::UsdGenDeviceLease rejectedLease = generation->AcquireLease(0x4321);
    Check(!rejectedLease.IsValid() && owner->rejected == 1,
          "owner rejection does not create a lease", &failures);
    owner->reject = false;

    owner->wrongStream = true;
    auto wrongStreamLease = generation->AcquireLease(0x4321);
    Check(!wrongStreamLease.IsValid() && owner->active == 1 &&
              owner->released == 1,
          "wrong-stream consumer is completed and rejected", &failures);
    owner->wrongStream = false;

    genericGeneration.reset();
    republished.reset();
    immutableAlias.reset();
    generation.reset();
    owner.reset();
    Check(!weakOwner.expired(), "lease retains owner after generation release", &failures);
    secondLease.Complete();
    secondLease.Complete();
    Check(weakOwner.expired() &&
              secondLease.WaitUntilReady() == usdGen::UsdGenDeviceStatus::InvalidLease,
          "owner releases after asynchronous use completes exactly once", &failures);

    std::printf(failures ? "testUsdGenDeviceGeneration: FAILED (%d)\n"
                         : "testUsdGenDeviceGeneration: PASS\n",
                failures);
    return failures ? 1 : 0;
}
