// usdGen engine — immutable device-generation handoff.
//
// This header deliberately contains no native, Hydra, or VtArray types.  A
// backend owner supplies the native allocation and synchronization details;
// the generation is only the immutable identity/shape contract shared by the
// engine, tools, and a renderer adapter.
#ifndef USDGEN_API_DEVICE_LEASE_H
#define USDGEN_API_DEVICE_LEASE_H

#include "usdGen/api/export.h"
#include "usdGen/api/token.h"
#include "usdGen/api/ids.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace usdGen {

enum class UsdGenDeviceValueType : uint8_t {
    Float32 = 0,
    Float32x2,
    Float32x3,
    Float32x4,
    Int32,
    UInt32,
    UInt64
};

enum class UsdGenDeviceDomain : uint8_t {
    Groom = 0,
    Primitive,
    Point,
    Tile,
    Topology,
    Tool
};

enum class UsdGenDeviceChannelSemantic : uint8_t {
    Generic = 0,
    Points,
    RestPoints,
    Widths,
    HairT,
    ElementOffsets,
    StableIds,
    RootPrim,
    RootUV
};

enum class UsdGenDeviceStatus : uint8_t {
    Ok = 0,
    InvalidArgument,
    InvalidLease,
    ProducerNotReady,
    ConsumerRejected,
    SynchronizationFailed
};

/// Opaque layout descriptor for a device generation: three bytes the owning
/// primitive interprets (its published type, basis and wrap).  All zero
/// means unspecified, the
/// state for consumers that do not opt into device rendering.
struct UsdGenDeviceLayoutMetadata
{
    uint8_t kind = 0;
    uint8_t variant = 0;
    uint8_t wrap = 0;
};

constexpr bool operator==(UsdGenDeviceLayoutMetadata const& a,
                          UsdGenDeviceLayoutMetadata const& b) noexcept {
    return a.kind == b.kind && a.variant == b.variant && a.wrap == b.wrap;
}
constexpr bool operator!=(UsdGenDeviceLayoutMetadata const& a,
                          UsdGenDeviceLayoutMetadata const& b) noexcept {
    return !(a == b);
}

/// An opaque native stream/queue token.  Zero denotes the backend default
/// queue.  The concrete owner interprets this value; this API never casts or
/// dereferences it and does not claim ownership of the native queue.
using UsdGenDeviceStream = uintptr_t;

struct UsdGenDeviceIdentity
{
    UsdGenBackendId backend = UsdGenBackendId::Invalid;
    int32_t deviceIndex = -1;
    uint64_t generation = 0;
};

struct UsdGenDeviceChannelMetadata
{
    UsdGenToken name;   // interned when the plan compiles
    UsdGenDeviceValueType type = UsdGenDeviceValueType::Float32;
    UsdGenDeviceDomain domain = UsdGenDeviceDomain::Point;
    uint64_t elementCount = 0;
    uint32_t arity = 1;
    uint32_t strideBytes = 0;
    bool readOnly = true;
    UsdGenDeviceChannelSemantic semantic =
        UsdGenDeviceChannelSemantic::Generic;
};

/// A range into the backend-owned geometry channels.  The range is metadata;
/// it is not a host copy of element offsets or points.
struct UsdGenDeviceTileMetadata
{
    uint32_t tile = 0;
    uint64_t firstElement = 0;
    uint64_t elementCount = 0;
    uint64_t firstPoint = 0;
    uint64_t pointCount = 0;
    // Scalar conservative render bounds paired with this immutable tile
    // range.  False preserves range-only legacy consumers and is required
    // after a point revision until a producer recomputes bounds.
    std::array<float, 3> extentMin{};
    std::array<float, 3> extentMax{};
    bool boundsValid = false;
};

struct UsdGenDeviceGeometryMetadata
{
    uint64_t topologyVersion = 0;
    uint64_t valueVersion = 0;
    uint64_t elementCount = 0;
    uint64_t pointCount = 0;
    std::vector<UsdGenDeviceTileMetadata> tiles;
    // Already-deformed C3 caches must not have scalp motion applied again.
    bool alreadyDeformed = false;
    // Appended to preserve existing aggregate initializers above.
    UsdGenDeviceLayoutMetadata layoutMetadata;
};

/// Validate immutable shape metadata before a backend allocates or submits.
/// Uses the same contract as generation publication, without requiring a dummy
/// native owner. This does not validate device data or readiness.
USDGEN_API bool ValidateUsdGenDeviceMetadata(UsdGenDeviceGeometryMetadata const& geometry,
    std::vector<UsdGenDeviceChannelMetadata> const& channels,
    std::string* reason = nullptr);

/// Minimum byte span for a typed strided channel, with checked arithmetic.
/// Zero elements require zero bytes. The output is unchanged on failure.
USDGEN_API bool UsdGenDeviceChannelStorageBytes(UsdGenDeviceChannelMetadata const& channel,
    size_t* bytes, std::string* reason = nullptr);

/// A channel's device storage for interop. Read it only while a lease
/// acquired on the reading stream is held.
struct UsdGenDeviceChannelAddress
{
    uint64_t address = 0;       // the backend's device address of the first element
    uint64_t bytes = 0;
    uint32_t strideBytes = 0;
};

/// Tool identity is retained with a geometry snapshot so a tool cannot
/// accidentally consume a later graph/device generation.
struct UsdGenDeviceToolMetadata
{
    std::string toolId;
    uint64_t graphVersion = 0;
    uint64_t snapshotVersion = 0;
};

class UsdGenDeviceOwner;

/// Shared lease state makes copies safe and keeps the owner alive until the
/// asynchronous consumer has completed.  It carries no raw allocation
/// pointer and is valid independently of the generation snapshot.
class USDGEN_API UsdGenDeviceLease
{
public:
    UsdGenDeviceLease() = default;
    ~UsdGenDeviceLease();
    UsdGenDeviceLease(UsdGenDeviceLease const &) = default;
    UsdGenDeviceLease &operator=(UsdGenDeviceLease const &) = default;
    UsdGenDeviceLease(UsdGenDeviceLease &&) noexcept = default;
    UsdGenDeviceLease &operator=(UsdGenDeviceLease &&) noexcept = default;

    explicit operator bool() const noexcept { return bool(_state); }
    bool IsValid() const noexcept { return bool(_state); }
    UsdGenDeviceStream Stream() const noexcept;

    /// Performs the producer-to-consumer ordering operation.  A false/failed
    /// result leaves the lease held so the caller can report and release it.
    UsdGenDeviceStatus WaitUntilReady() const noexcept;

    /// Releases this lease's shared use state.  Other copies, if any, retain
    /// the use until they are also destroyed.
    void Complete() noexcept;

private:
    struct State;
    explicit UsdGenDeviceLease(std::shared_ptr<State> state)
        : _state(std::move(state)) {}
    std::shared_ptr<State> _state;

    friend class UsdGenDeviceGeneration;
};

class USDGEN_API UsdGenDeviceGeneration
{
public:
    struct CreateInfo
    {
        UsdGenDeviceIdentity identity;
        UsdGenDeviceGeometryMetadata geometry;
        UsdGenDeviceToolMetadata tool;
        std::vector<UsdGenDeviceChannelMetadata> channels;
        std::shared_ptr<const UsdGenDeviceOwner> owner;
    };

    /// Returns null and writes a short reason when the immutable snapshot is
    /// malformed or has no owner.  The owner is the only backend payload
    /// retained by this neutral interface.
    static std::shared_ptr<const UsdGenDeviceGeneration> Create(
        CreateInfo info, std::string *reason = nullptr);

    UsdGenDeviceIdentity const &Identity() const noexcept;
    UsdGenDeviceGeometryMetadata const &Geometry() const noexcept;
    UsdGenDeviceToolMetadata const &Tool() const noexcept;
    std::vector<UsdGenDeviceChannelMetadata> const &Channels() const noexcept;
    std::shared_ptr<const UsdGenDeviceOwner> Owner() const noexcept;
    size_t ExclusiveRetainedBytes() const noexcept;
    size_t InclusiveRetainedBytes() const noexcept;

    /// Creates a new immutable publication identity over the same backend
    /// owner. No channel is copied or made writable; this is a COW cache-hit
    /// wrapper used to keep Session and device generation ids coherent.
    std::shared_ptr<const UsdGenDeviceGeneration> Republish(
        uint64_t generation, std::string *reason = nullptr) const;
    /// Metadata-only publication alias for backends whose immutable payload
    /// revisions are independent of publication identity. Keeps all geometry
    /// and channel revisions; legacy Republish still advances valueVersion.
    std::shared_ptr<const UsdGenDeviceGeneration> RepublishPreservingRevisions(
        uint64_t generation, std::string *reason = nullptr) const;

    /// Acquires a lifetime/fence lease for a concrete consumer stream.  An
    /// invalid lease means the owner rejected the asynchronous use.
    UsdGenDeviceLease AcquireLease(UsdGenDeviceStream stream) const noexcept;

    /// The owner's storage of 'semantic' sliced to the tile with id 'tile'
    /// (UINT32_MAX: the whole channel) by the channel's domain: Point by the
    /// tile's points, Primitive by its elements, Topology by its elements
    /// plus one. Valid while a lease on the reading stream is held.
    bool ChannelAddress(UsdGenDeviceChannelSemantic semantic, uint32_t tile,
                        UsdGenDeviceChannelAddress *out) const noexcept;

private:
    std::shared_ptr<const UsdGenDeviceGeneration> _Republish(
        uint64_t generation, bool preserveRevisions, std::string *reason) const;
    UsdGenDeviceGeneration() = default;

    UsdGenDeviceIdentity _identity;
    UsdGenDeviceGeometryMetadata _geometry;
    UsdGenDeviceToolMetadata _tool;
    std::vector<UsdGenDeviceChannelMetadata> _channels;
    std::shared_ptr<const UsdGenDeviceOwner> _owner;
};

} // namespace usdGen

#endif  // USDGEN_API_DEVICE_LEASE_H
