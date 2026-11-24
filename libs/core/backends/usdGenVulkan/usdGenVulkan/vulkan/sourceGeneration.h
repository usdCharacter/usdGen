// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_SOURCE_GENERATION_H
#define USDGEN_VULKAN_SOURCE_GENERATION_H

#include "chargedBuffer.h"
#include "widthPipeline.h"
#include "widthBlendPipeline.h"
#include "nonWidthComparePipeline.h"
#include "lengthScalePipeline.h"
#include "lengthCompactionPipeline.h"
#include "noisePipeline.h"
#include "deformPipeline.h"
#include "rbfVkDeformPipeline.h"
#include "resampleVkPipeline.h"
#include "curveGrowPipeline.h"
#include "exprVkApply.h"
#include "usdGen/curveBuffer.h"
#include "usdGen/deviceGeneration.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace usdGen::vulkan {
enum class SourceGenerationStatus { InvalidInput, AllocationFailed, Created, QueueRejected, Submitted, NotReady, Ready, LostProof };

// Extra native schema fields not represented by UsdGenCurveBuffer::UsdGenPlane.
// `bytes` is copied at Create; metadata must be a valid Generic neutral field.
struct VulkanSourceNamedChannel {
    UsdGenDeviceChannelMetadata metadata;
    std::vector<uint8_t> bytes;
    // Executor-owned primitive float3 data, retained/gathered with private
    // source frames and omitted from renderer-visible channel enumeration.
    bool privateStructural = false;
};

struct VulkanSourceGenerationCreateInfo {
    std::shared_ptr<DeviceContext> context;
    UsdGenCurveBuffer source;
    // Explicit renderer-neutral capture state; CurveBuffer does not encode
    // curve basis/wrap, alreadyDeformed, or tile bounds metadata.
    UsdGenDeviceGeometryMetadata geometry;
    std::vector<VulkanSourceNamedChannel> additionalNamed;
};

// CPU-only source capture boundary. It deliberately carries no DeviceContext
// or native ownership, so validation/packing can be shared before a queue
// owner admits the resulting immutable upload.
struct VulkanSourcePrepareInfo {
    UsdGenCurveBuffer source;
    UsdGenDeviceGeometryMetadata geometry;
    std::vector<VulkanSourceNamedChannel> additionalNamed;
};

class VulkanPreparedSource final {
public:
    struct Plane {
        UsdGenDeviceChannelMetadata metadata;
        std::vector<uint8_t> bytes;
        bool privateFrame = false;
    };
    static std::shared_ptr<const VulkanPreparedSource> Prepare(VulkanSourcePrepareInfo,
        std::string* reason = nullptr);
    uint64_t topologyVersion() const noexcept { return topology_; }
    uint64_t valueVersion() const noexcept { return value_; }
    uint32_t curveCount() const noexcept { return curves_; }
    uint32_t pointCount() const noexcept { return points_; }
    UsdGenDeviceGeometryMetadata const& geometry() const noexcept { return geometry_; }
    std::vector<UsdGenChunkDesc> const& chunks() const noexcept { return chunks_; }
    // Complete upload packet, including privateFrame-tagged T/B/N planes. It
    // is not the public native channel enumeration exposed by a generation.
    std::vector<Plane> const& planes() const noexcept { return planes_; }
private:
    friend class VulkanSourceUpload;
    VulkanPreparedSource() = default;
    std::vector<Plane> planes_;
    std::vector<UsdGenChunkDesc> chunks_;
    UsdGenDeviceGeometryMetadata geometry_;
    uint64_t topology_ = 0, value_ = 0;
    uint32_t curves_ = 0, points_ = 0;
};

class VulkanSourceGeneration;

// Mutable only until fence proof. Queue-owner confinement applies to Submit
// and Poll; no native plane is exposed from this pending candidate.
class VulkanSourceUpload final {
public:
    struct Storage;
    static std::shared_ptr<VulkanSourceUpload> Create(VulkanSourceGenerationCreateInfo,
        VkResult* = nullptr, std::string* reason = nullptr);
    static std::shared_ptr<VulkanSourceUpload> Create(std::shared_ptr<DeviceContext>,
        std::shared_ptr<const VulkanPreparedSource>, VkResult* = nullptr,
        std::string* reason = nullptr);
    ~VulkanSourceUpload();
    VulkanSourceUpload(VulkanSourceUpload const&) = delete;
    VulkanSourceUpload& operator=(VulkanSourceUpload const&) = delete;
    SourceGenerationStatus Submit() noexcept;
    SourceGenerationStatus Poll() noexcept;
    void Quarantine() noexcept;
    std::shared_ptr<const VulkanSourceGeneration> TakeReady() noexcept;
private:
    friend class VulkanSourceGeneration;
    explicit VulkanSourceUpload(std::shared_ptr<Storage>);
    std::shared_ptr<Storage> storage_;
};

// Immutable native source snapshot. It is intentionally not a neutral device
// generation/owner adapter and cannot make Vulkan production-available.
class VulkanSourceGeneration final {
public:
    // `buffer == VK_NULL_HANDLE && bytes == 0` is the canonical empty plane.
    struct PlaneView { UsdGenDeviceChannelMetadata metadata; VkBuffer buffer = VK_NULL_HANDLE; VkDeviceSize bytes = 0; };
    uint64_t topologyVersion() const noexcept;
    uint64_t valueVersion() const noexcept;
    uint32_t curveCount() const noexcept;
    uint32_t pointCount() const noexcept;
    // Per-generation native footprint: a COW child's exclusive footprint is
    // its width delta; inclusive bytes also cover retained ancestors. These
    // values are not additive across aliases/siblings: repeated TakeReady or
    // adoption of one candidate can share allocations. Cache-wide physical
    // accounting must deduplicate owners or use the resource pool ledger.
    size_t ExclusiveRetainedBytes() const noexcept;
    size_t InclusiveRetainedBytes() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    UsdGenDeviceGeometryMetadata const& geometry() const noexcept;
    std::vector<UsdGenChunkDesc> const& chunks() const noexcept;
    std::vector<PlaneView> const& planes() const noexcept;
    // Captured source T/B/N stay private and never appear in neutral planes.
    std::vector<PlaneView> const& sourceFrames() const noexcept;
    // Empty descriptors have no owner. Public plane names only are accepted.
    std::shared_ptr<const ChargedBuffer> PlaneOwner(std::string const& name) const noexcept;
    std::shared_ptr<const ChargedBuffer> SourceFrameOwner(std::string const& name) const noexcept;
    // Stable identity of the immutable non-width bundle. Width-only COW
    // children delegate to their base; a point edit establishes a new bundle.
    void const* NonWidthIdentity() const noexcept;
    // Metadata-only alias for a newly published value revision. All planes,
    // topology, private source frames and native allocation owners are shared.
    static std::shared_ptr<const VulkanSourceGeneration> WithValueVersion(
        std::shared_ptr<const VulkanSourceGeneration> const&, uint64_t newValueVersion);
    // Identity layout of a dynamic topology stage. Native owners are shared;
    // the declared immutable topology and value revisions are both published.
    static std::shared_ptr<const VulkanSourceGeneration> WithTopologyVersion(
        std::shared_ptr<const VulkanSourceGeneration> const&, uint64_t newValueVersion);
    // A statically disabled Deform shares native planes but still publishes
    // animated-space provenance, matching the CUDA operator's disabled path.
    static std::shared_ptr<const VulkanSourceGeneration> WithDeformAlias(
        std::shared_ptr<const VulkanSourceGeneration> const&, uint64_t newValueVersion);
    static std::shared_ptr<const VulkanSourceGeneration> WithWidth(
        std::shared_ptr<const VulkanSourceGeneration> const&, WidthPipeline::Candidate const&,
        uint64_t newValueVersion, std::string* reason = nullptr);
    static std::shared_ptr<const VulkanSourceGeneration> WithExpressionWidth(
        std::shared_ptr<const VulkanSourceGeneration> const&, ExprVkWidthApplyPipeline::Candidate const&,
        uint64_t newValueVersion, std::string* reason = nullptr);
    // Grow geometry is GPU-produced; the parallel resample supplies named
    // point planes and primitive structural fields without host readback.
    static std::shared_ptr<const VulkanSourceGeneration> WithGrown(
        std::shared_ptr<const VulkanSourceGeneration> const&, CurveGrowPipeline::Candidate const&,
        ResampleVkPipeline::Candidate const&, uint32_t target, uint64_t newValueVersion,
        std::string* reason = nullptr);
    static std::shared_ptr<const VulkanSourceGeneration> WithPoints(
        std::shared_ptr<const VulkanSourceGeneration> const&, LengthScalePipeline::Candidate const&,
        uint64_t newValueVersion, std::string* reason = nullptr);
    static std::shared_ptr<const VulkanSourceGeneration> WithNoise(
        std::shared_ptr<const VulkanSourceGeneration> const&,
        NoisePipeline::Candidate const&, uint64_t newValueVersion,
        std::string* reason = nullptr);
    static std::shared_ptr<const VulkanSourceGeneration> WithDeform(
        std::shared_ptr<const VulkanSourceGeneration> const&,
        DeformPipeline::Candidate const&, uint64_t newValueVersion,
        std::string* reason = nullptr);
    // Phase-2 RBF device path (rbfVk hook): identical contract.
    static std::shared_ptr<const VulkanSourceGeneration> WithDeform(
        std::shared_ptr<const VulkanSourceGeneration> const&,
        RbfVkDeformPipeline::Candidate const&, uint64_t newValueVersion,
        std::string* reason = nullptr);
    static std::shared_ptr<const VulkanSourceGeneration> WithCompacted(
        std::shared_ptr<const VulkanSourceGeneration> const&,
        LengthCompactionPipeline::Candidate const&, uint64_t newValueVersion,
        std::string* reason = nullptr, bool emptyGrow = false);
    static std::shared_ptr<const VulkanSourceGeneration> WithResampled(
        std::shared_ptr<const VulkanSourceGeneration> const&,
        ResampleVkPipeline::Candidate const&, uint32_t target, uint64_t newValueVersion,
        std::string* reason = nullptr);
    static std::shared_ptr<const VulkanSourceGeneration> WithWidthBlend(
        std::shared_ptr<const VulkanSourceGeneration> const& left,
        std::shared_ptr<const VulkanSourceGeneration> const& right,
        WidthBlendPipeline::Candidate const&, uint64_t newValueVersion,
        std::string* reason = nullptr,
        NonWidthComparePipeline::Candidate const* nonWidthProof = nullptr);
    // Tile-finalize attach: metadata-only alias child replacing geometry
    // tiles with finalized spans/bounds (same contiguous-coverage contract
    // as gpu::WithTileMetadata, plus validated conservative bounds). Planes,
    // chunks, topology and value versions are inherited unchanged.
    static std::shared_ptr<const VulkanSourceGeneration> WithTiles(
        std::shared_ptr<const VulkanSourceGeneration> const&,
        std::vector<UsdGenDeviceTileMetadata>, std::string* reason = nullptr);
private:
    friend class VulkanSourceUpload;
    VulkanSourceGeneration(std::shared_ptr<const VulkanSourceGeneration>, uint64_t);
    explicit VulkanSourceGeneration(std::shared_ptr<const VulkanSourceUpload::Storage>);
    VulkanSourceGeneration(std::shared_ptr<const VulkanSourceGeneration>,
                           std::shared_ptr<const ChargedBuffer>, uint64_t);
    VulkanSourceGeneration(std::shared_ptr<const VulkanSourceGeneration>,
                           std::shared_ptr<const ChargedBuffer>, uint64_t, bool pointsOverride);
    VulkanSourceGeneration(std::shared_ptr<const VulkanSourceGeneration> const&,
                           LengthCompactionPipeline::Candidate const&, uint64_t);
    VulkanSourceGeneration(std::shared_ptr<const VulkanSourceGeneration> const&,
                           ResampleVkPipeline::Candidate const&, uint32_t, uint64_t);
    VulkanSourceGeneration(std::shared_ptr<const VulkanSourceGeneration> const&,
                           std::vector<UsdGenDeviceTileMetadata>);
    std::shared_ptr<const VulkanSourceUpload::Storage> storage_;
    std::shared_ptr<const VulkanSourceGeneration> base_;
    std::shared_ptr<const ChargedBuffer> widthOverride_;
    std::shared_ptr<const ChargedBuffer> pointsOverride_;
    std::vector<std::pair<std::string, std::shared_ptr<const ChargedBuffer>>> compactOwners_;
    std::vector<PlaneView> compactFrames_;
    std::vector<UsdGenChunkDesc> compactChunks_;
    std::vector<UsdGenDeviceTileMetadata> compactTiles_;
    uint32_t compactCurves_ = 0, compactPoints_ = 0;
    bool compacted_ = false;
    std::shared_ptr<const void> nonWidthIdentity_;
    uint64_t valueVersion_ = 0;
    UsdGenDeviceGeometryMetadata childGeometry_;
    std::vector<PlaneView> planes_;
    std::vector<PlaneView> sourceFrames_;
};
} // namespace usdGen::vulkan
#endif
