#include "usdGen/deviceGeneration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace usdGen {
namespace {

size_t ScalarBytes(UsdGenDeviceValueType type)
{
    switch (type) {
    case UsdGenDeviceValueType::Float32:
    case UsdGenDeviceValueType::Float32x2:
    case UsdGenDeviceValueType::Float32x3:
    case UsdGenDeviceValueType::Float32x4:
    case UsdGenDeviceValueType::Int32:
    case UsdGenDeviceValueType::UInt32:
        return 4;
    case UsdGenDeviceValueType::UInt64:
        return 8;
    }
    return 0;
}

uint32_t TypeArity(UsdGenDeviceValueType type)
{
    switch (type) {
    case UsdGenDeviceValueType::Float32x2: return 2;
    case UsdGenDeviceValueType::Float32x3: return 3;
    case UsdGenDeviceValueType::Float32x4: return 4;
    default: return 1;
    }
}

bool ValidArity(UsdGenDeviceChannelMetadata const &channel)
{
    // Generic scalar arrays use arity as renderer-neutral elementSize (R24),
    // e.g. uniform int[3]/float[3] guide data. Vector value types already
    // encode their width and must retain that exact arity.
    if (channel.semantic == UsdGenDeviceChannelSemantic::Generic &&
        (channel.type == UsdGenDeviceValueType::Float32 ||
         channel.type == UsdGenDeviceValueType::Int32 ||
         channel.type == UsdGenDeviceValueType::UInt32 ||
         channel.type == UsdGenDeviceValueType::UInt64)) {
        return channel.arity >= 1 && channel.arity <= 16;
    }
    return channel.arity == TypeArity(channel.type);
}

bool ValidValueType(UsdGenDeviceValueType type)
{
    switch (type) {
    case UsdGenDeviceValueType::Float32:
    case UsdGenDeviceValueType::Float32x2:
    case UsdGenDeviceValueType::Float32x3:
    case UsdGenDeviceValueType::Float32x4:
    case UsdGenDeviceValueType::Int32:
    case UsdGenDeviceValueType::UInt32:
    case UsdGenDeviceValueType::UInt64:
        return true;
    }
    return false;
}

bool ValidDomain(UsdGenDeviceDomain domain)
{
    switch (domain) {
    case UsdGenDeviceDomain::Groom:
    case UsdGenDeviceDomain::Primitive:
    case UsdGenDeviceDomain::Point:
    case UsdGenDeviceDomain::Tile:
    case UsdGenDeviceDomain::Topology:
    case UsdGenDeviceDomain::Tool:
        return true;
    }
    return false;
}

bool ValidSemantic(UsdGenDeviceChannelSemantic semantic)
{
    switch (semantic) {
    case UsdGenDeviceChannelSemantic::Generic:
    case UsdGenDeviceChannelSemantic::Points:
    case UsdGenDeviceChannelSemantic::RestPoints:
    case UsdGenDeviceChannelSemantic::Widths:
    case UsdGenDeviceChannelSemantic::HairT:
    case UsdGenDeviceChannelSemantic::CurveOffsets:
    case UsdGenDeviceChannelSemantic::StableIds:
    case UsdGenDeviceChannelSemantic::RootPrim:
    case UsdGenDeviceChannelSemantic::RootUV:
        return true;
    }
    return false;
}

bool ValidCurveTopology(UsdGenDeviceCurveTopologyMetadata const &topology)
{
    using Type = UsdGenDeviceCurveType;
    using Basis = UsdGenDeviceCurveBasis;
    using Wrap = UsdGenDeviceCurveWrap;
    if (topology.type == Type::Unknown || topology.basis == Basis::Unknown ||
        topology.wrap == Wrap::Unknown) {
        return topology.type == Type::Unknown &&
            topology.basis == Basis::Unknown && topology.wrap == Wrap::Unknown;
    }
    return topology.type == Type::Cubic &&
        (topology.basis == Basis::BSpline || topology.basis == Basis::CatmullRom) &&
        topology.wrap == Wrap::Pinned;
}

bool ValidBounds(UsdGenDeviceTileMetadata const &tile)
{
    for (size_t axis = 0; axis != tile.extentMin.size(); ++axis) {
        if (!std::isfinite(tile.extentMin[axis]) ||
            !std::isfinite(tile.extentMax[axis]) ||
            tile.extentMin[axis] > tile.extentMax[axis]) {
            return false;
        }
        if (tile.curveCount == 0 && tile.pointCount == 0 &&
            (tile.extentMin[axis] != 0.0f || tile.extentMax[axis] != 0.0f)) {
            return false;
        }
    }
    return true;
}

bool ValidSemanticShape(UsdGenDeviceChannelMetadata const &channel,
                        UsdGenDeviceGeometryMetadata const &geometry)
{
    using S = UsdGenDeviceChannelSemantic;
    switch (channel.semantic) {
    case S::Generic:
        switch (channel.domain) {
        case UsdGenDeviceDomain::Groom:
        case UsdGenDeviceDomain::Tool:
            return channel.elementCount == 1;
        case UsdGenDeviceDomain::Primitive:
            return channel.elementCount == geometry.curveCount;
        case UsdGenDeviceDomain::Point:
            return channel.elementCount == geometry.pointCount;
        case UsdGenDeviceDomain::Tile:
            return channel.elementCount == geometry.tiles.size();
        case UsdGenDeviceDomain::Topology:
            return false;
        }
        return false;
    case S::Points:
    case S::RestPoints:
        return channel.type == UsdGenDeviceValueType::Float32x3 &&
               channel.domain == UsdGenDeviceDomain::Point &&
               channel.elementCount == geometry.pointCount;
    case S::Widths:
    case S::HairT:
        return channel.type == UsdGenDeviceValueType::Float32 &&
               channel.domain == UsdGenDeviceDomain::Point &&
               channel.elementCount == geometry.pointCount;
    case S::CurveOffsets:
        return channel.type == UsdGenDeviceValueType::UInt32 &&
               channel.domain == UsdGenDeviceDomain::Topology &&
               geometry.curveCount != UINT64_MAX &&
               channel.elementCount == geometry.curveCount + 1;
    case S::StableIds:
        return channel.type == UsdGenDeviceValueType::UInt64 &&
               channel.domain == UsdGenDeviceDomain::Primitive &&
               channel.elementCount == geometry.curveCount;
    case S::RootPrim:
        return channel.type == UsdGenDeviceValueType::Int32 &&
               channel.domain == UsdGenDeviceDomain::Primitive &&
               channel.elementCount == geometry.curveCount;
    case S::RootUV:
        return channel.type == UsdGenDeviceValueType::Float32x2 &&
               channel.domain == UsdGenDeviceDomain::Primitive &&
               channel.elementCount == geometry.curveCount;
    }
    return false;
}

bool Fail(std::string *reason, char const *message)
{
    if (reason) *reason = message;
    return false;
}

} // namespace

bool UsdGenDeviceChannelStorageBytes(UsdGenDeviceChannelMetadata const& channel,
    size_t* bytes, std::string* reason)
{
    if (!bytes || !ValidValueType(channel.type) || !ValidSemantic(channel.semantic) ||
        !ValidArity(channel)) return Fail(reason, "device channel has invalid storage type or arity");
    size_t const elementBytes = ScalarBytes(channel.type) * channel.arity;
    if (channel.strideBytes < elementBytes)
        return Fail(reason, "device channel has invalid stride");
    if (!channel.elementCount) { *bytes = 0; return true; }
    size_t const limit = std::numeric_limits<size_t>::max();
    if (channel.elementCount - 1 > (limit - elementBytes) / channel.strideBytes)
        return Fail(reason, "device channel byte span overflows this platform");
    *bytes = size_t(channel.elementCount - 1) * channel.strideBytes + elementBytes;
    return true;
}

bool ValidateUsdGenDeviceMetadata(UsdGenDeviceGeometryMetadata const& geometry,
    std::vector<UsdGenDeviceChannelMetadata> const& channels, std::string* reason)
{
    if ((geometry.pointCount == 0 && geometry.curveCount != 0) ||
        (geometry.curveCount == 0 && geometry.pointCount != 0)) {
        Fail(reason, "device geometry has mismatched curve and point counts");
        return false;
    }
    if (!ValidCurveTopology(geometry.curveTopology)) {
        Fail(reason, "device geometry has invalid curve topology metadata");
        return false;
    }

    std::vector<std::string> names;
    names.reserve(channels.size());
    for (UsdGenDeviceChannelMetadata const &channel : channels) {
        if (channel.name.empty() || channel.arity == 0 ||
            !ValidValueType(channel.type) || !ValidDomain(channel.domain) ||
            !ValidSemantic(channel.semantic) || !channel.readOnly ||
            !ValidArity(channel)) {
            Fail(reason, "device channel has invalid name or arity");
            return false;
        }
        size_t storageBytes = 0;
        if (!UsdGenDeviceChannelStorageBytes(channel, &storageBytes, reason)) return false;
        if (!ValidSemanticShape(channel, geometry)) {
            Fail(reason, "device channel has invalid semantic shape or domain");
            return false;
        }
        if (std::find(names.begin(), names.end(), channel.name) != names.end()) {
            Fail(reason, "device generation has duplicate channel");
            return false;
        }
        names.push_back(channel.name);
    }

    std::vector<uint32_t> tileIds;
    tileIds.reserve(geometry.tiles.size());
    for (UsdGenDeviceTileMetadata const &tile : geometry.tiles) {
        if (tile.firstCurve > geometry.curveCount ||
            tile.curveCount > geometry.curveCount - tile.firstCurve ||
            tile.firstPoint > geometry.pointCount ||
            tile.pointCount > geometry.pointCount - tile.firstPoint) {
            Fail(reason, "device tile range exceeds geometry");
            return false;
        }
        if (std::find(tileIds.begin(), tileIds.end(), tile.tile) != tileIds.end()) {
            Fail(reason, "device generation has duplicate tile");
            return false;
        }
        if (tile.boundsValid && !ValidBounds(tile)) {
            Fail(reason, "device tile has invalid conservative bounds");
            return false;
        }
        tileIds.push_back(tile.tile);
    }
    return true;
}

struct UsdGenDeviceLease::State
{
    State(std::shared_ptr<const UsdGenDeviceOwner> owner_,
          std::unique_ptr<UsdGenDeviceConsumer> consumer_)
        : owner(std::move(owner_)), consumer(std::move(consumer_)) {}

    ~State() { if (consumer) consumer->Complete(); }

    // Declaration order is intentional: State destroys consumer before owner.
    // Backends that complete asynchronously must also retain a strong owner
    // from the consumer through their terminal retirement callback.
    std::shared_ptr<const UsdGenDeviceOwner> owner;
    std::unique_ptr<UsdGenDeviceConsumer> consumer;
};

UsdGenDeviceLease::~UsdGenDeviceLease() = default;

UsdGenDeviceStream UsdGenDeviceLease::Stream() const noexcept
{
    return _state && _state->consumer ? _state->consumer->Stream()
                                      : UsdGenDeviceStream(0);
}

UsdGenDeviceStatus UsdGenDeviceLease::WaitUntilReady() const noexcept
{
    if (!_state || !_state->consumer) return UsdGenDeviceStatus::InvalidLease;
    return _state->consumer->WaitUntilReady();
}

void UsdGenDeviceLease::Complete() noexcept
{
    // Do not complete the shared state here: another copied lease may still
    // be using the backend handle.  State::~State completes exactly once when
    // this is the final lease copy.
    _state.reset();
}

std::shared_ptr<const UsdGenDeviceGeneration>
UsdGenDeviceGeneration::Create(CreateInfo info, std::string *reason)
{
    if (!info.owner) {
        Fail(reason, "device generation requires an owner");
        return {};
    }
    if (info.identity.backend == UsdGenDeviceBackend::Unknown ||
        info.identity.deviceIndex < 0) {
        Fail(reason, "device generation has invalid backend identity");
        return {};
    }
    if (!ValidateUsdGenDeviceMetadata(info.geometry, info.channels, reason)) return {};

    std::shared_ptr<UsdGenDeviceGeneration> generation(
        new UsdGenDeviceGeneration());
    generation->_identity = std::move(info.identity);
    generation->_geometry = std::move(info.geometry);
    generation->_tool = std::move(info.tool);
    generation->_channels = std::move(info.channels);
    generation->_owner = std::move(info.owner);
    return std::shared_ptr<const UsdGenDeviceGeneration>(std::move(generation));
}

UsdGenDeviceIdentity const &UsdGenDeviceGeneration::Identity() const noexcept
{
    return _identity;
}

UsdGenDeviceGeometryMetadata const &UsdGenDeviceGeneration::Geometry() const noexcept
{
    return _geometry;
}

UsdGenDeviceToolMetadata const &UsdGenDeviceGeneration::Tool() const noexcept
{
    return _tool;
}

std::vector<UsdGenDeviceChannelMetadata> const &
UsdGenDeviceGeneration::Channels() const noexcept
{
    return _channels;
}

std::shared_ptr<const UsdGenDeviceOwner>
UsdGenDeviceGeneration::Owner() const noexcept
{
    return _owner;
}

size_t
UsdGenDeviceGeneration::ExclusiveRetainedBytes() const noexcept
{
    return _owner ? _owner->ExclusiveRetainedBytes() : 0;
}

size_t
UsdGenDeviceGeneration::InclusiveRetainedBytes() const noexcept
{
    return _owner ? _owner->InclusiveRetainedBytes() : 0;
}

std::shared_ptr<const UsdGenDeviceGeneration>
UsdGenDeviceGeneration::Republish(uint64_t generation, std::string *reason) const
{
    return _Republish(generation, false, reason);
}

std::shared_ptr<const UsdGenDeviceGeneration>
UsdGenDeviceGeneration::RepublishPreservingRevisions(
    uint64_t generation, std::string *reason) const
{
    return _Republish(generation, true, reason);
}

std::shared_ptr<const UsdGenDeviceGeneration>
UsdGenDeviceGeneration::_Republish(
    uint64_t generation, bool preserveRevisions, std::string *reason) const
{
    if (generation <= _identity.generation || !_owner ||
        !_owner->ProducerReady()) {
        if (reason) *reason = "device cache republish requires a ready owner and newer generation";
        return {};
    }
    CreateInfo info;
    info.identity = _identity;
    info.identity.generation = generation;
    info.geometry = _geometry;
    if (!preserveRevisions) info.geometry.valueVersion = generation;
    info.tool = _tool;
    info.channels = _channels;
    info.owner = _owner;
    return Create(std::move(info), reason);
}

UsdGenDeviceLease
UsdGenDeviceGeneration::AcquireLease(UsdGenDeviceStream stream) const noexcept
{
    if (!_owner) return {};
    std::unique_ptr<UsdGenDeviceConsumer> consumer =
        _owner->AcquireConsumer(stream);
    if (!consumer) return {};
    if (consumer->Stream() != stream) {
        consumer->Complete();
        return {};
    }
    try {
        return UsdGenDeviceLease(
            std::make_shared<UsdGenDeviceLease::State>(
                _owner, std::move(consumer)));
    } catch (...) {
        // Allocation can fail before State takes ownership. Do not require
        // backend destructors to implement the explicit completion contract.
        if (consumer) consumer->Complete();
        return {};
    }
}

} // namespace usdGen
