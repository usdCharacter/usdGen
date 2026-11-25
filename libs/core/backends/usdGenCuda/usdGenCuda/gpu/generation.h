#ifndef USDGEN_GPU_GENERATION_H
#define USDGEN_GPU_GENERATION_H

#include "usdGen/deviceGeneration.h"
#include "curveSource.h"
#include "curveResample.h"
#include "curveGrow.h"
#include "scatterGrow.h"

namespace usdGen::gpu {

class CudaCurveCompaction;

// Generic, byte-addressed extension channel owned by a CUDA generation. The
// metadata remains backend-neutral; the byte buffer is only the CUDA storage
// implementation. Planes may replace an inherited Generic channel by name,
// but cannot replace C3 geometry semantics.
struct CudaNamedChannelPlane {
    UsdGenDeviceChannelMetadata metadata;
    std::unique_ptr<DeviceBuffer<unsigned char>> bytes;
};

// Takes exclusive ownership of a completed source. Each published generation
// gets distinct mutable storage; it is never replaced underneath a reader.
std::shared_ptr<const UsdGenDeviceGeneration> MakeSourceGeneration(
    std::unique_ptr<CudaCurveSource> source, uint64_t generation,
    std::string* reason = nullptr, bool alreadyDeformed = false,
    std::unique_ptr<DeviceBuffer<float>> widths = {},
    std::unique_ptr<DeviceBuffer<float3>> points = {},
    uint64_t topologyVersion = UINT64_MAX,
    std::vector<CudaNamedChannelPlane> namedChannels = {});

// Owns both the immutable uploaded source and its completed device-only
// resample. The source remains alive because the resampler borrowed it until
// Finish; consumers expose only the resampled channels.
std::shared_ptr<const UsdGenDeviceGeneration> MakeResampledGeneration(
    std::unique_ptr<CudaCurveSource> source, std::unique_ptr<CudaCurveResample> resampled,
    uint64_t generation, std::string* reason = nullptr, bool alreadyDeformed = false,
    std::unique_ptr<DeviceBuffer<float>> widths = {},
    std::unique_ptr<DeviceBuffer<float3>> points = {},
    uint64_t topologyVersion = UINT64_MAX,
    std::vector<CudaNamedChannelPlane> namedChannels = {});

// Owns a completed GPU topology revision, including every reordered channel.
std::shared_ptr<const UsdGenDeviceGeneration> MakeCompactedGeneration(
    std::unique_ptr<CudaCurveCompaction> geometry, uint64_t generation,
    std::string* reason = nullptr, bool alreadyDeformed = false,
    std::unique_ptr<DeviceBuffer<float>> widths = {},
    std::unique_ptr<DeviceBuffer<float3>> points = {},
    uint64_t topologyVersion = UINT64_MAX,
    std::vector<CudaNamedChannelPlane> namedChannels = {});

// Owns a completed Scatter-capture/Grow topology expansion.  The producer
// carries the immutable copied roots and T/B/N frame planes through its
// completion proof; only its fresh expanded storage is transferred here.
std::shared_ptr<const UsdGenDeviceGeneration> MakeScatterGrowGeneration(
    std::unique_ptr<CudaScatterGrow> geometry, uint64_t generation,
    std::string* reason = nullptr, bool alreadyDeformed = false,
    uint64_t topologyVersion = UINT64_MAX,
    std::vector<CudaNamedChannelPlane> namedChannels = {},
    // Fresh point/width revisions remain private COW planes of this output;
    // Scatter/Grow's expanded topology and root-frame owner are retained.
    std::unique_ptr<DeviceBuffer<float>> widths = {},
    std::unique_ptr<DeviceBuffer<float3>> points = {});

// Owns a completed C3-native Grow topology expansion. The producer has
// already copied every C3 geometry/binding channel into fresh output storage;
// named planes are the executor-owned, freshly transformed counterparts.
std::shared_ptr<const UsdGenDeviceGeneration> MakeCurveGrowGeneration(
    std::unique_ptr<CudaCurveGrow> geometry, uint64_t generation,
    std::string* reason = nullptr, bool alreadyDeformed = false,
    uint64_t topologyVersion = UINT64_MAX,
    std::vector<CudaNamedChannelPlane> namedChannels = {},
    std::unique_ptr<DeviceBuffer<float>> widths = {},
    std::unique_ptr<DeviceBuffer<float3>> points = {});

// Completed device-only point edit of an immutable snapshot. All other
// channels and the topology version are shared with base. A null points
// revision republishes the unmodified base (gesture cancellation).
std::shared_ptr<const UsdGenDeviceGeneration> MakePointRevisionGeneration(
    std::shared_ptr<const UsdGenDeviceGeneration> base, uint64_t generation,
    std::unique_ptr<DeviceBuffer<float3>> points,
    UsdGenDeviceToolMetadata tool = {}, std::string* reason = nullptr);

std::shared_ptr<const UsdGenDeviceGeneration> MakeNamedChannelRevisionGeneration(
    std::shared_ptr<const UsdGenDeviceGeneration> base, uint64_t generation,
    std::vector<CudaNamedChannelPlane> planes, UsdGenDeviceToolMetadata tool = {},
    std::string* reason = nullptr);

class CudaNamedChannelLease {
public:
    explicit operator bool() const noexcept { return bool(lease_); }
    UsdGenDeviceChannelMetadata const* Metadata() const noexcept {
        return lease_ ? &metadata_ : nullptr;
    }
    DeviceView<const unsigned char> Bytes() const noexcept {
        return lease_ ? bytes_ : DeviceView<const unsigned char>{};
    }
private:
    UsdGenDeviceLease lease_;
    UsdGenDeviceChannelMetadata metadata_;
    DeviceView<const unsigned char> bytes_;
    friend CudaNamedChannelLease AcquireNamedChannel(
        std::shared_ptr<const UsdGenDeviceGeneration> const&, std::string const&, cudaStream_t);
};

CudaNamedChannelLease AcquireNamedChannel(
    std::shared_ptr<const UsdGenDeviceGeneration> const&, std::string const&, cudaStream_t);

// Read-only geometry access for CUDA tools/consumers. The native stream must
// remain alive until every copy of the lease is released. Destruction arranges
// backend retirement after queued consumer work before releasing the last
// storage reference. CUDA lease acquisition rejects a stream under graph
// capture; Complete must run outside capture. Acquire/release run on ordinary
// host threads, never CUDA host callbacks
// (CUDA forbids runtime API calls from those callbacks).
class CudaGeometryLease {
public:
    explicit operator bool() const noexcept { return bool(lease_); }
    DeviceCurveGeometryView Geometry() const noexcept { return lease_ ? geometry_ : DeviceCurveGeometryView{}; }
    DeviceView<const float> HairT() const noexcept { return lease_ ? hairT_ : DeviceView<const float>{}; }
    DeviceView<const int32_t> RootPrim() const noexcept { return lease_ ? rootPrim_ : DeviceView<const int32_t>{}; }
    DeviceView<const float2> RootUV() const noexcept { return lease_ ? rootUV_ : DeviceView<const float2>{}; }
    // Optional immutable rest-frame planes carried by generated topology.
    DeviceView<const float3> RootT() const noexcept { return lease_ ? rootT_ : DeviceView<const float3>{}; }
    DeviceView<const float3> RootB() const noexcept { return lease_ ? rootB_ : DeviceView<const float3>{}; }
    DeviceView<const float3> RootN() const noexcept { return lease_ ? rootN_ : DeviceView<const float3>{}; }
private:
    UsdGenDeviceLease lease_;
    DeviceCurveGeometryView geometry_{};
    DeviceView<const float> hairT_{};
    DeviceView<const int32_t> rootPrim_{};
    DeviceView<const float2> rootUV_{};
    DeviceView<const float3> rootT_{}, rootB_{}, rootN_{};
    friend CudaGeometryLease AcquireGeometry(
        std::shared_ptr<const UsdGenDeviceGeneration> const&, cudaStream_t);
};

/// A zero-copy tile view over a geometry lease. curveOffsets retain their
/// global values (the slice has curveCount + 1 entries); use CurveIndexSpan
/// when passing this view to the index builder.
struct CudaGeometryTileView {
    DeviceView<const float3> points;
    DeviceView<const float3> restPoints;
    DeviceView<const float> widths;
    DeviceView<const float> hairT;
    DeviceView<const uint32_t> curveOffsets;
    DeviceView<const uint64_t> stableIds;
    DeviceView<const int32_t> rootPrim;
    DeviceView<const float2> rootUV;
    UsdGenDeviceTileMetadata range;
    uint64_t generation = 0;
    uint64_t topologyVersion = 0;
    uint64_t valueVersion = 0;
};

/// Owns the parent geometry lease while exposing one zero-copy tile slice.
/// This deliberately does not expose DeviceCurveGeometryView, whose offsets
/// are implicitly zero-based and would misrepresent a tile slice.
class CudaGeometryTileLease {
public:
    explicit operator bool() const noexcept { return bool(parent_); }
    CudaGeometryTileView View() const noexcept {
        return parent_ ? view_ : CudaGeometryTileView{};
    }
private:
    CudaGeometryLease parent_;
    CudaGeometryTileView view_{};
    friend CudaGeometryTileLease AcquireGeometryTile(
        std::shared_ptr<const UsdGenDeviceGeneration> const&, uint32_t,
        uint64_t, cudaStream_t);
};

CudaGeometryLease AcquireGeometry(
    std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
    cudaStream_t stream);

CudaGeometryTileLease AcquireGeometryTile(
    std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
    uint32_t tileId, uint64_t expectedGeneration, cudaStream_t stream);

/// Attach immutable tile-range metadata without copying or replacing any
/// backend channels/owner. Returns a new immutable wrapper on success.
std::shared_ptr<const UsdGenDeviceGeneration> WithTileMetadata(
    std::shared_ptr<const UsdGenDeviceGeneration> const& candidate,
    std::vector<UsdGenDeviceTileMetadata> tiles,
    std::string* reason = nullptr,
    UsdGenDeviceCurveTopologyMetadata curveTopology = {});

} // namespace usdGen::gpu
#endif
