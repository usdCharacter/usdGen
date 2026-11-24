// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// PicktileVk: Vulkan port of the device-only picking + tile spans/bounds/index
// streams (gpu/picking.cu, gpu/curveTiles.cu, gpu/curveTileBounds.cu,
// gpu/curveIndices.cu). CUDA twins have no graph operator: they run at
// executor finalization (tile spans/bounds populate geometry metadata) and in
// the session tool layer (CudaToolSession::Pick/Footprint). This module ports
// that same surface to Vulkan with no new graph operator:
//
//   * five compute pipelines (pick, footprint, tiles, bounds, indices) with
//     the repo-standard Create/Begin->Candidate/Poll lifecycle;
//   * Vulkan-identity geometry/tile leases plus a backend-neutral tile
//     metadata attach (the AcquireGeometry/AcquireGeometryTile Vulkan
//     support; the CUDA entry points keep failing closed for Vulkan state);
//   * host requirements/admission helpers mirroring the CUDA scalar cores;
//   * a blocking tool session (Pick/Footprint) mirroring CudaToolSession.
//
// This header is CUDA-free so Vulkan-only builds can consume it. Enum
// ordinals intentionally match their CUDA twins; the parity test asserts
// that where CUDA is available.
#ifndef USDGEN_VULKAN_PICKTILE_VK_H
#define USDGEN_VULKAN_PICKTILE_VK_H

#include "chargedBuffer.h"
#include "deviceGenerationAdapter.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace usdGen::vulkan {

// Ordinals match gpu/curveTileBounds.h, gpu/curveTiles.h, gpu/curveIndices.h.
enum class PicktileVkTileBoundsBasis : uint8_t { Linear = 0, BSpline = 1, CatmullRom = 2 };
enum class PicktileVkTileOrder : uint8_t {
    SortedSurvivorSubset = 0,
    IdentityCaptureOrder = 1,
    CaptureOrderSurvivorSubset = 2
};
enum class PicktileVkIndexBasis : uint8_t {
    Linear = 0, Bezier = 1, BSpline = 2, CatmullRom = 3, CentripetalCatmullRom = 4
};
enum class PicktileVkIndexWrap : uint8_t {
    Nonperiodic = 0, Periodic = 1, Pinned = 2, Segmented = 3
};
enum class PicktileVkIndexMode : uint8_t {
    Curves = 0, Hull = 1, Points = 2
};

// Layout matches gpu::CurveTileSpan (5xu32, 20 bytes).
struct PicktileVkTileSpan {
    uint32_t tile = 0;
    uint32_t firstCurve = 0;
    uint32_t curveCount = 0;
    uint32_t firstPoint = 0;
    uint32_t pointCount = 0;
};
static_assert(sizeof(PicktileVkTileSpan) == 20, "tile span ABI");

// Row-major viewProj with USD row-vector convention, top-left pixel origin.
// Mirrors gpu::PickQuery field-for-field.
struct PicktileVkQuery {
    float viewProj[16]{};
    uint32_t width = 0;
    uint32_t height = 0;
    float x = 0.0f;
    float y = 0.0f;
    float radiusPx = 0.0f;
};

// Mirrors gpu::PickResult. UINT32_MAX denotes an unset curve/CV/flat index.
struct PicktileVkResult {
    bool hit = false;
    uint32_t curve = UINT32_MAX;
    uint32_t cv = UINT32_MAX;
    uint32_t flatIndex = UINT32_MAX;
    uint64_t stableId = 0;
    float distancePx = 0.0f;
};

// CUDA CurveTileRequirements scalar core, sans device/stream binding (a Vulkan
// candidate binds its context at Begin instead).
struct PicktileVkTileRequirements {
    size_t chunkCount = 0;
    size_t tileCount = 0;
    size_t chunksPerTile = 0;
    size_t curvesPerTile = 0;
    uint32_t chunkSize = 0;
    uint32_t tileTarget = 0;
};

// CUDA CurveIndexRequirements scalar core. scanBytes is always 0: scan scratch
// is pipeline-owned on Vulkan rather than caller-provided.
struct PicktileVkIndexRequirements {
    size_t maxRecords = 0;
    uint32_t indexArity = 0;
};

// Bit-exact ports of the CUDA requirements scalar arithmetic
// (GetCurveTileRequirements/GetCurveIndexRequirements minus device/stream
// binding). Zero selects the 512/64 tile defaults; other values clamp
// chunkSize to [128,1024] and tileTarget to [32,256].
bool GetPicktileVkTileRequirements(uint32_t chunkSize, uint32_t tileTarget,
    size_t captureCurveCount, size_t survivorCurveCount, size_t survivorPointCount,
    PicktileVkTileRequirements* result) noexcept;
bool GetPicktileVkIndexRequirements(PicktileVkIndexBasis basis, PicktileVkIndexWrap wrap,
    PicktileVkIndexMode mode, size_t curveCount, size_t pointCount,
    PicktileVkIndexRequirements* result) noexcept;

// ---------------------------------------------------------------------------
// Pipelines. Each mirrors TopologyPipeline's lifecycle: Create(context, spirv)
// / Begin(...)->unique_ptr<Candidate> / Poll / succeeded / output /
// inputOwner / context, with Quarantine on dropped pending work and an
// owner-side BeforeSubmit admission checkpoint before the first native
// submit. All owned buffers are host-visible/coherent (tools path); borrowed
// inputs may live anywhere. No method blocks except the documented
// bounds-host-finalize inside Bounds Poll (pure host compute, <= 256 tiles).
// ---------------------------------------------------------------------------

class PicktileVkPickPipeline final : public std::enable_shared_from_this<PicktileVkPickPipeline> {
public:
    class Candidate;
    // Mirrors gpu::PickingStatus ordering for the reachable codes.
    enum class Semantic : uint32_t { Ok = 0, InvalidArgument = 1, NonFiniteInput = 2 };
    using BeforeSubmit = std::function<bool()>;
    // SPIR-V must be the trusted, externally spirv-val-validated
    // picktileVkPick.comp module (single "main" entry, phase selector in the
    // push constant) with the seven-binding/16-byte push-constant ABI from
    // the shader header comment.
    static std::shared_ptr<PicktileVkPickPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& spirv,
        VkResult* result = nullptr);
    ~PicktileVkPickPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    // Host Begin checks mirror CudaPicking::validateGeometry/validateQuery
    // (minus CUDA stream/device binding, plus ChargedBuffer context/usage
    // checks): pointCount <= INT32_MAX, exact plane bytes, width/height >= 1,
    // finite x/y/radius with 0 <= radius <= sqrt(FLT_MAX), finite viewProj.
    // stableIds may be null (index fallback, like a null CUDA id view).
    // Vulkan admission bound (documented): pointCount <= UINT32_MAX/12, so
    // flat-plane 3*i indexing never wraps u32.
    std::unique_ptr<Candidate> Begin(
        std::shared_ptr<const ChargedBuffer> points,
        std::shared_ptr<const ChargedBuffer> offsets,
        std::shared_ptr<const ChargedBuffer> stableIds,
        uint32_t curveCount, uint32_t pointCount,
        PicktileVkQuery const& query,
        VkResult* result = nullptr, BeforeSubmit beforeSubmit = {});
private:
    struct Native;
    explicit PicktileVkPickPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

class PicktileVkPickPipeline::Candidate final {
public:
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;
    // VK_SUCCESS means device proof obtained. *status carries the Semantic.
    // VK_NOT_READY preserves the candidate.
    VkResult Poll(Semantic* status = nullptr);
    // Valid after a proved Poll with semantic == Ok: miss or hit record.
    PicktileVkResult result() const noexcept;
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    void Quarantine() noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class PicktileVkPickPipeline;
};

class PicktileVkFootprintPipeline final : public std::enable_shared_from_this<PicktileVkFootprintPipeline> {
public:
    class Candidate;
    enum class Semantic : uint32_t { Ok = 0, InvalidArgument = 1, NonFiniteInput = 2 };
    using BeforeSubmit = std::function<bool()>;
    // Trusted, externally spirv-val-validated picktileVkFootprint.comp with
    // the eight-binding/20-byte push-constant ABI.
    static std::shared_ptr<PicktileVkFootprintPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& spirv,
        VkResult* result = nullptr);
    ~PicktileVkFootprintPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    std::unique_ptr<Candidate> Begin(
        std::shared_ptr<const ChargedBuffer> points,
        std::shared_ptr<const ChargedBuffer> offsets,
        uint32_t curveCount, uint32_t pointCount,
        PicktileVkQuery const& query,
        VkResult* result = nullptr, BeforeSubmit beforeSubmit = {});
private:
    struct Native;
    explicit PicktileVkFootprintPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

class PicktileVkFootprintPipeline::Candidate final {
public:
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;
    VkResult Poll(Semantic* status = nullptr);
    // Device-resident ascending flat indices; only [0, footprintCount()) are
    // defined, and only after a proved Ok poll. Mirrors CudaPicking's
    // footprint()/footprintCount() publication.
    std::shared_ptr<const ChargedBuffer> footprint() const noexcept;
    uint32_t footprintCount() const noexcept;
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    void Quarantine() noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class PicktileVkFootprintPipeline;
};

class PicktileVkTilesPipeline final : public std::enable_shared_from_this<PicktileVkTilesPipeline> {
public:
    class Candidate;
    using BeforeSubmit = std::function<bool()>;
    // Trusted, externally spirv-val-validated picktileVkTiles.comp with the
    // seven-binding/36-byte push-constant ABI.
    static std::shared_ptr<PicktileVkTilesPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& spirv,
        VkResult* result = nullptr);
    ~PicktileVkTilesPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    // Host Begin checks mirror BuildCurveTiles enqueue validation (order,
    // requirements identity, exact plane bytes, identity cardinality, sorted
    // lookup presence for CaptureOrderSurvivorSubset). sortedIds/sortedOrdinals
    // may be null unless order is CaptureOrderSurvivorSubset.
    std::unique_ptr<Candidate> Begin(
        std::shared_ptr<const ChargedBuffer> captureIds,
        std::shared_ptr<const ChargedBuffer> survivorIds,
        std::shared_ptr<const ChargedBuffer> survivorOffsets,
        std::shared_ptr<const ChargedBuffer> sortedIds,
        std::shared_ptr<const ChargedBuffer> sortedOrdinals,
        uint32_t captureCurveCount, uint32_t survivorCurveCount, uint32_t survivorPointCount,
        PicktileVkTileOrder order, PicktileVkTileRequirements const& requirements,
        VkResult* result = nullptr, BeforeSubmit beforeSubmit = {});
private:
    struct Native;
    explicit PicktileVkTilesPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

class PicktileVkTilesPipeline::Candidate final {
public:
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;
    // *deviceStatus carries the GPU scalar (0 valid, 1 invalid input).
    VkResult Poll(uint32_t* deviceStatus = nullptr);
    // tileCount spans; valid only after a proved poll with status 0.
    std::shared_ptr<const ChargedBuffer> spans() const noexcept;
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    void Quarantine() noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class PicktileVkTilesPipeline;
};

class PicktileVkBoundsPipeline final : public std::enable_shared_from_this<PicktileVkBoundsPipeline> {
public:
    class Candidate;
    using BeforeSubmit = std::function<bool()>;
    // Trusted, externally spirv-val-validated picktileVkBounds.comp with the
    // seven-binding/12-byte push-constant ABI.
    static std::shared_ptr<PicktileVkBoundsPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& spirv,
        VkResult* result = nullptr);
    ~PicktileVkBoundsPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    // Host Begin checks mirror BuildCurveTileBounds enqueue validation
    // (basis, points/widths domain match, tileCount <= INT32_MAX). Scratch
    // and outputs are pipeline-owned and exactly sized.
    // Vulkan admission bound (documented): pointDomain <= UINT32_MAX/12, so
    // flat-plane 3*i indexing never wraps u32.
    std::unique_ptr<Candidate> Begin(
        std::shared_ptr<const ChargedBuffer> points,
        std::shared_ptr<const ChargedBuffer> widths,
        std::shared_ptr<const ChargedBuffer> spans,
        uint32_t pointDomain, uint32_t tileCount,
        PicktileVkTileBoundsBasis basis,
        VkResult* result = nullptr, BeforeSubmit beforeSubmit = {});
private:
    struct Native;
    explicit PicktileVkBoundsPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

class PicktileVkBoundsPipeline::Candidate final {
public:
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;
    // First proved poll also runs the host directed-rounding finalize; a
    // finalize failure (unrepresentable bound) reports status 1 with outputs
    // unpublished, exactly like the CUDA finalize-then-gated-publish.
    // *deviceStatus carries the GPU/host scalar (0 valid, 1 invalid).
    VkResult Poll(uint32_t* deviceStatus = nullptr);
    std::shared_ptr<const ChargedBuffer> minimums() const noexcept;
    std::shared_ptr<const ChargedBuffer> maximums() const noexcept;
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    void Quarantine() noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class PicktileVkBoundsPipeline;
};

class PicktileVkIndicesPipeline final : public std::enable_shared_from_this<PicktileVkIndicesPipeline> {
public:
    class Candidate;
    using BeforeSubmit = std::function<bool()>;
    // Trusted, externally spirv-val-validated picktileVkIndices.comp with the
    // eight-binding/48-byte push-constant ABI.
    static std::shared_ptr<PicktileVkIndicesPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& spirv,
        VkResult* result = nullptr);
    ~PicktileVkIndicesPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    // Host Begin checks mirror BuildCurveIndices/Requirements validation
    // (enums, segmented-linear rule, (!curves && points), points < curves,
    // curves >= INT32_MAX, points > INT32_MAX, capacity overflow, exact plane
    // bytes, requirements identity). A pointBase + pointCount overflow is NOT
    // a host failure: it enqueues a spanInvalid candidate that fails closed
    // on device (status 1, recordCount 0), mirroring InvalidSpanKernel.
    // Vulkan admission bound (documented): maxRecords * arity must fit u32;
    // beyond that Begin fails rather than emit truncated device indices.
    std::unique_ptr<Candidate> Begin(
        std::shared_ptr<const ChargedBuffer> offsets,
        uint32_t curveCount, uint32_t pointCount, uint32_t pointBase,
        PicktileVkIndexBasis basis, PicktileVkIndexWrap wrap, PicktileVkIndexMode mode,
        PicktileVkIndexRequirements const& requirements,
        VkResult* result = nullptr, BeforeSubmit beforeSubmit = {});
private:
    struct Native;
    explicit PicktileVkIndicesPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

class PicktileVkIndicesPipeline::Candidate final {
public:
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;
    // *deviceStatus: 0 valid, 1 bad offsets, 2 invalid segmented count.
    VkResult Poll(uint32_t* deviceStatus = nullptr);
    // Packed records (recordCount * arity int32) and owning curves; only
    // [0, recordCount) are defined, after a proved status-0 poll.
    std::shared_ptr<const ChargedBuffer> indices() const noexcept;
    std::shared_ptr<const ChargedBuffer> primitiveParam() const noexcept;
    uint64_t recordCount() const noexcept;
    // Native indirect-draw count (PackCurveDrawCount parity): 0 unless
    // status == 0 with a representable count.
    uint32_t drawCount() const noexcept;
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    void Quarantine() noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class PicktileVkIndicesPipeline;
};

// ---------------------------------------------------------------------------
// Vulkan-identity geometry/tile leases: the AcquireGeometry/AcquireGeometryTile
// Vulkan support. The CUDA entry points keep failing closed for Vulkan state
// (backend check); these acquire the same data through AcquireVulkanGeneration
// planes. Views are zero-copy (buffer, byteOffset, elementCount); curveOffsets
// slices retain global values like CudaGeometryTileView.
// ---------------------------------------------------------------------------

struct PicktileVkPlaneSlice {
    VkBuffer buffer = VK_NULL_HANDLE;
    uint64_t byteOffset = 0;
    uint64_t elementCount = 0;
    uint32_t elementBytes = 0;
};

struct PicktileVkGeometryView {
    PicktileVkPlaneSlice points;
    PicktileVkPlaneSlice restPoints;
    PicktileVkPlaneSlice widths;
    PicktileVkPlaneSlice curveOffsets; // curveCount + 1 global offsets
    PicktileVkPlaneSlice stableIds;
    PicktileVkPlaneSlice hairT;
    PicktileVkPlaneSlice rootPrim;
    PicktileVkPlaneSlice rootUV;
    uint64_t curveCount = 0;
    uint64_t pointCount = 0;
    bool hasStableIds = false;
};

class PicktileVkGeometryLease final {
public:
    PicktileVkGeometryLease() = default;
    explicit operator bool() const noexcept { return valid_; }
    PicktileVkGeometryView const& View() const noexcept { return view_; }
    UsdGenDeviceStatus WaitUntilReady() const noexcept;
    void Complete() noexcept;
private:
    PicktileVkGeometryLease(VulkanGenerationLease lease, PicktileVkGeometryView view) noexcept;
    VulkanGenerationLease lease_;
    PicktileVkGeometryView view_{};
    bool valid_ = false;
    friend PicktileVkGeometryLease AcquirePicktileVkGeometry(
        std::shared_ptr<const UsdGenDeviceGeneration> const&, UsdGenDeviceStream) noexcept;
};

// Acquires a validated Vulkan-native geometry view. Fails closed (invalid
// lease) for null/foreign-backend generations, stream-token mismatch, missing
// or misshapen required planes (points/curveOffsets), or unreadable state.
PicktileVkGeometryLease AcquirePicktileVkGeometry(
    std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
    UsdGenDeviceStream queueStream) noexcept;

struct PicktileVkTileView {
    PicktileVkPlaneSlice points;
    PicktileVkPlaneSlice restPoints;
    PicktileVkPlaneSlice widths;
    PicktileVkPlaneSlice hairT;
    PicktileVkPlaneSlice curveOffsets; // curveCount + 1 entries, global values
    PicktileVkPlaneSlice stableIds;
    PicktileVkPlaneSlice rootPrim;
    PicktileVkPlaneSlice rootUV;
    UsdGenDeviceTileMetadata range;
    uint64_t generation = 0;
    uint64_t topologyVersion = 0;
    uint64_t valueVersion = 0;
};

class PicktileVkTileLease final {
public:
    PicktileVkTileLease() = default;
    explicit operator bool() const noexcept { return valid_; }
    PicktileVkTileView const& View() const noexcept { return view_; }
private:
    PicktileVkTileLease(PicktileVkGeometryLease parent, PicktileVkTileView view) noexcept;
    PicktileVkGeometryLease parent_;
    PicktileVkTileView view_{};
    bool valid_ = false;
    friend PicktileVkTileLease AcquirePicktileVkTile(
        std::shared_ptr<const UsdGenDeviceGeneration> const&,
        uint32_t, uint64_t, UsdGenDeviceStream) noexcept;
};

// Owns the parent geometry lease while exposing one zero-copy tile slice.
// Validation mirrors AcquireGeometryTile: Vulkan backend, expected-generation
// match, tile membership, contiguity/u32-range checks, and the rootPrim/rootUV
// pairing rule.
PicktileVkTileLease AcquirePicktileVkTile(
    std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
    uint32_t tileId, uint64_t expectedGeneration, UsdGenDeviceStream queueStream) noexcept;

// Backend-neutral tile-range attach for Vulkan publications (the WithTileMetadata
// Vulkan support; the gpu entry point lives in a CUDA-gated file but is itself
// backend-neutral). Same contiguous-coverage validation, same re-publication
// of identity/owner/channels with replaced geometry tiles/topology.
std::shared_ptr<const UsdGenDeviceGeneration> PicktileVkWithTileMetadata(
    std::shared_ptr<const UsdGenDeviceGeneration> const& candidate,
    std::vector<UsdGenDeviceTileMetadata> tiles,
    std::string* reason = nullptr,
    UsdGenDeviceCurveTopologyMetadata curveTopology = {});

// Session-layer admission for Vulkan tools: the generation exposes the exact
// channels a tool needs (points/curveOffsets required with exact bytes;
// stableIds/widths/rest/hairT/root channels validated when present).
// Host-only, no device work; mirrors the CudaToolSession lease checks.
bool ValidatePicktileVkToolChannels(UsdGenDeviceGeometryMetadata const& geometry,
    std::vector<UsdGenDeviceChannelMetadata> const& channels,
    std::string* reason = nullptr);

// Converts device tile spans + finalized bounds into publication metadata,
// validating conservativeness (finite, min <= max per axis) exactly like
// CudaToolSession::RefreshTileBounds. Pure host function for the executor
// finalization hook (see phase2 plan fragment).
bool PicktileVkPublishTileMetadata(PicktileVkTileSpan const* spans,
    float const* minimums, float const* maximums, size_t tileCount,
    std::vector<UsdGenDeviceTileMetadata>* tiles);

// ---------------------------------------------------------------------------
// Blocking tool session: the CudaToolSession::Pick/Footprint Vulkan support.
// Retains the last successful pick result and footprint across failed ops
// (CudaPicking::Finish parity). All methods block on a bounded fence wait;
// async composition through the completion service is the documented
// integration follow-up (see phase2 plan fragment).
// ---------------------------------------------------------------------------

struct PicktileVkChargedGeometry {
    std::shared_ptr<const ChargedBuffer> points;
    std::shared_ptr<const ChargedBuffer> offsets;
    std::shared_ptr<const ChargedBuffer> stableIds; // may be null
    uint32_t curveCount = 0;
    uint32_t pointCount = 0;
};

class PicktileVkToolSession final {
public:
    static std::shared_ptr<PicktileVkToolSession> Create(std::shared_ptr<DeviceContext>,
        std::vector<uint32_t> const& pickSpirv, std::vector<uint32_t> const& footprintSpirv,
        std::string* reason = nullptr);
    ~PicktileVkToolSession();
    PicktileVkToolSession(PicktileVkToolSession const&) = delete;
    PicktileVkToolSession& operator=(PicktileVkToolSession const&) = delete;
    bool Pick(PicktileVkChargedGeometry const& geometry, PicktileVkQuery const& query,
        PicktileVkResult* result, std::string* reason = nullptr);
    bool Footprint(PicktileVkChargedGeometry const& geometry, PicktileVkQuery const& query,
        std::string* reason = nullptr);
    // Last successful results (preserved across failures, like CudaPicking).
    PicktileVkResult result() const noexcept { return result_; }
    std::shared_ptr<const ChargedBuffer> footprint() const noexcept;
    uint32_t footprintCount() const noexcept { return footprintCount_; }
    char const* diagnostic() const noexcept;
private:
    explicit PicktileVkToolSession(std::shared_ptr<DeviceContext>);
    bool Fail(std::string const& message);
    std::shared_ptr<DeviceContext> context_;
    std::shared_ptr<PicktileVkPickPipeline> pick_;
    std::shared_ptr<PicktileVkFootprintPipeline> footprint_;
    PicktileVkResult result_{};
    std::shared_ptr<const ChargedBuffer> footprintBuffer_;
    uint32_t footprintCount_ = 0;
    std::string diagnostic_;
};

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_PICKTILE_VK_H
