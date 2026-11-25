// usdGen engine — immutable device-generation handoff.
//
// This header deliberately contains no CUDA, Hydra, or VtArray types.  A
// backend owner supplies the native allocation and synchronization details;
// the generation is only the immutable identity/shape contract shared by the
// engine, tools, and a renderer adapter.
#ifndef USDGEN_DEVICE_GENERATION_H
#define USDGEN_DEVICE_GENERATION_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace usdGen {

enum class UsdGenDeviceBackend : uint8_t {
    Unknown = 0,
    Cuda = 1,
    Metal = 2,
    Vulkan = 3,
    CpuReference = 4,
    Other = 255
};

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
    CurveOffsets,
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

/// Renderer-neutral curve topology for a device generation.  Unknown is the
/// backwards-compatible state for consumers that do not opt into device
/// rendering; CUDA C3 publication admits only the concrete values below.
enum class UsdGenDeviceCurveType : uint8_t {
    Unknown = 0,
    Cubic = 1
};

enum class UsdGenDeviceCurveBasis : uint8_t {
    Unknown = 0,
    BSpline = 1,
    CatmullRom = 2
};

enum class UsdGenDeviceCurveWrap : uint8_t {
    Unknown = 0,
    Pinned = 1
};

struct UsdGenDeviceCurveTopologyMetadata
{
    UsdGenDeviceCurveType type = UsdGenDeviceCurveType::Unknown;
    UsdGenDeviceCurveBasis basis = UsdGenDeviceCurveBasis::Unknown;
    UsdGenDeviceCurveWrap wrap = UsdGenDeviceCurveWrap::Unknown;
};

/// An opaque native stream/queue token.  Zero denotes the backend default
/// queue.  The concrete owner interprets this value; this API never casts or
/// dereferences it and does not claim ownership of the native queue.
using UsdGenDeviceStream = uintptr_t;

struct UsdGenDeviceIdentity
{
    UsdGenDeviceBackend backend = UsdGenDeviceBackend::Unknown;
    int32_t deviceIndex = -1;
    uint64_t generation = 0;
};

struct UsdGenDeviceChannelMetadata
{
    std::string name;
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
/// it is not a host copy of curve offsets or points.
struct UsdGenDeviceTileMetadata
{
    uint32_t tile = 0;
    uint64_t firstCurve = 0;
    uint64_t curveCount = 0;
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
    uint64_t curveCount = 0;
    uint64_t pointCount = 0;
    std::vector<UsdGenDeviceTileMetadata> tiles;
    // Already-deformed C3 caches must not have scalp motion applied again.
    bool alreadyDeformed = false;
    // Appended to preserve existing aggregate initializers above.
    UsdGenDeviceCurveTopologyMetadata curveTopology;
};

/// Validate immutable shape metadata before a backend allocates or submits.
/// Uses the same contract as generation publication, without requiring a dummy
/// native owner. This does not validate device data or readiness.
bool ValidateUsdGenDeviceMetadata(UsdGenDeviceGeometryMetadata const& geometry,
    std::vector<UsdGenDeviceChannelMetadata> const& channels,
    std::string* reason = nullptr);

/// Minimum byte span for a typed strided channel, with checked arithmetic.
/// Zero elements require zero bytes. The output is unchanged on failure.
bool UsdGenDeviceChannelStorageBytes(UsdGenDeviceChannelMetadata const& channel,
    size_t* bytes, std::string* reason = nullptr);

/// Tool identity is retained with a geometry snapshot so a tool cannot
/// accidentally consume a later graph/device generation.
struct UsdGenDeviceToolMetadata
{
    std::string toolId;
    uint64_t graphVersion = 0;
    uint64_t snapshotVersion = 0;
};

/// A single consumer use of a backend generation.  The handle is owned by a
/// lease state, so a backend can keep all state needed for fencing locally
/// instead of maintaining a process-wide token registry.
class UsdGenDeviceConsumer
{
public:
    virtual ~UsdGenDeviceConsumer() = default;
    virtual UsdGenDeviceStream Stream() const noexcept = 0;

    /// Orders the producer's work before the consumer dereferences channels.
    /// A failed wait does not complete the handle; the owner remains alive
    /// until Complete (or destruction) fences and releases it.
    virtual UsdGenDeviceStatus WaitUntilReady() const noexcept = 0;

    /// Arranges terminal retirement of the consumer stream and releases the
    /// use. It may return before native work completes; the backend must keep
    /// the owner alive until that completion is proven. Implementations must
    /// be idempotent and must not throw.
    virtual void Complete() noexcept = 0;
};

/// Backend-specific implementation of allocation ownership and fencing.
/// Implementations may add typed accessors in a private/backend header, but
/// the generation and lease only depend on this synchronization contract.
class UsdGenDeviceOwner
{
public:
    virtual ~UsdGenDeviceOwner() = default;

    /// Return true only when all producer work needed by the generation has
    /// completed or can safely be waited on.
    virtual bool ProducerReady() const noexcept = 0;

    /// Exact bytes owned exclusively by this owner. Shared COW predecessors
    /// are excluded so a cache can charge each physical delta only once.
    /// A backend returning zero is ineligible for device-cache admission.
    virtual size_t ExclusiveRetainedBytes() const noexcept { return 0; }

    /// Conservative physical footprint reachable through this owner,
    /// including immutable COW predecessors it retains. This is the
    /// cache-charge view: it may count a shared predecessor again when more
    /// than one independent cache entry references it, but it cannot
    /// undercharge an entry whose predecessor outlives its cache record.
    /// Backends with no COW predecessor inherit the exclusive value.
    virtual size_t InclusiveRetainedBytes() const noexcept {
        return ExclusiveRetainedBytes();
    }

    /// Acquire a consumer use.  The owner may reject a stream that cannot be
    /// fenced safely.  The returned handle owns all per-use backend state and
    /// is released by UsdGenDeviceLease destruction/completion.
    virtual std::unique_ptr<UsdGenDeviceConsumer> AcquireConsumer(
        UsdGenDeviceStream stream) const noexcept = 0;
};

/// Shared lease state makes copies safe and keeps the owner alive until the
/// asynchronous consumer has completed.  It carries no raw allocation
/// pointer and is valid independently of the generation snapshot.
class UsdGenDeviceLease
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

class UsdGenDeviceGeneration
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

#endif // USDGEN_DEVICE_GENERATION_H
