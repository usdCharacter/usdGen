#ifdef USDGEN_ENABLE_CUDA
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "usdGenCurves/cuda/tools.h"
#include "usdGen/gpu/cudaCompletionLane.h"
#include <cmath>
#include <cstdint>
#include <limits>

namespace usdGen::gpu {

CudaToolSession::~CudaToolSession() { Close(); }

bool CudaToolSession::Fail(char const* message) {
    diagnostic_ = message;
    return false;
}

bool CudaToolSession::CheckCurrent(UsdGenGenerationConstPtr const& current) {
    if (!base_) return Fail("no active device stroke");
    if (!current || (current != expected_ && current->device != latest_))
        return Fail("device stroke is stale; close it before using a new graph snapshot");
    if (current->id == std::numeric_limits<int64_t>::max())
        return Fail("device publication identity exhausted");
    return true;
}

bool CudaToolSession::RefreshTileBounds(CudaGeometryLease const& lease,
    DeviceView<const float3> revisedPoints,
    std::vector<UsdGenDeviceTileMetadata>* tiles, bool* boundsReady) {
    if (!tiles || !boundsReady || !base_) return Fail("missing point-edit bounds state");
    *boundsReady = false;
    auto const& geometry = base_->device->Geometry();
    auto const& topology = geometry.layoutMetadata;
    // Range-only legacy generations are legal, but cannot be declared render
    // ready without a concrete shader topology.
    if (topology == UsdGenDeviceLayoutMetadata{})
        return true;
    CurveTileBoundsBasis basis;
    if (UsdGenCurveLayoutTypeOf(topology) != UsdGenCurveLayoutType::Cubic ||
        UsdGenCurveLayoutWrapOf(topology) != UsdGenCurveLayoutWrap::Pinned) {
        return Fail("point-edit bounds require concrete pinned C3 topology");
    }
    if (UsdGenCurveLayoutBasisOf(topology) == UsdGenCurveLayoutBasis::BSpline)
        basis = CurveTileBoundsBasis::BSpline;
    else if (UsdGenCurveLayoutBasisOf(topology) == UsdGenCurveLayoutBasis::CatmullRom)
        basis = CurveTileBoundsBasis::CatmullRom;
    else
        return Fail("point-edit bounds require B-spline or Catmull-Rom topology");
    DeviceCurveGeometryView const baseGeometry = lease.Geometry();
    if (revisedPoints.size != geometry.pointCount ||
        baseGeometry.widths.size != geometry.pointCount ||
        (geometry.pointCount && (!revisedPoints.data || !baseGeometry.widths.data))) {
        return Fail("point-edit bounds have incompatible device channels");
    }
    if (geometry.tiles.size() > 256)
        return Fail("point-edit bounds exceed the 256-tile readback contract");

    std::vector<CurveTileSpan> hostSpans;
    hostSpans.reserve(geometry.tiles.size());
    uint64_t curveEnd = 0, pointEnd = 0, previousId = 0;
    bool first = true;
    for (UsdGenDeviceTileMetadata const& tile : geometry.tiles) {
        if ((!first && tile.tile <= previousId) || tile.firstElement != curveEnd ||
            tile.firstPoint != pointEnd || tile.elementCount > tile.pointCount ||
            (tile.elementCount == 0 && tile.pointCount != 0) ||
            tile.firstElement > UINT32_MAX || tile.elementCount > UINT32_MAX - tile.firstElement ||
            tile.firstPoint > UINT32_MAX || tile.pointCount > UINT32_MAX - tile.firstPoint ||
            tile.elementCount > geometry.elementCount - curveEnd ||
            tile.pointCount > geometry.pointCount - pointEnd) {
            return Fail("point-edit bounds found malformed tile metadata");
        }
        hostSpans.push_back({tile.tile, static_cast<uint32_t>(tile.firstElement),
            static_cast<uint32_t>(tile.elementCount), static_cast<uint32_t>(tile.firstPoint),
            static_cast<uint32_t>(tile.pointCount)});
        curveEnd += tile.elementCount;
        pointEnd += tile.pointCount;
        previousId = tile.tile;
        first = false;
    }
    if ((hostSpans.empty() && (geometry.elementCount || geometry.pointCount)) ||
        (!hostSpans.empty() &&
         (curveEnd != geometry.elementCount || pointEnd != geometry.pointCount))) {
        return Fail("point-edit bounds do not cover base geometry");
    }
    size_t const count = hostSpans.size();
    if (spans_.reset(count) != cudaSuccess || boundsScratch_.reset(count) != cudaSuccess ||
        boundsMinimums_.reset(count) != cudaSuccess || boundsMaximums_.reset(count) != cudaSuccess ||
        boundsStatus_.reset(1) != cudaSuccess) {
        return Fail("point-edit bounds workspace allocation failed");
    }
    if (!(hostSpans.empty() || cudaMemcpyAsync(spans_.data(), hostSpans.data(),
            hostSpans.size() * sizeof(CurveTileSpan), cudaMemcpyHostToDevice,
            stream_) == cudaSuccess)) {
        return Fail("point-edit bounds span upload failed");
    }
    DeviceView<const CurveTileSpan> const deviceSpans =
        static_cast<DeviceBuffer<CurveTileSpan> const&>(spans_).view();
    CurveTileBoundsInput const input{revisedPoints, baseGeometry.widths, deviceSpans};
    CurveTileBoundsWorkspace const workspace{boundsScratch_.view()};
    CurveTileBoundsOutput const output{boundsMinimums_.view(), boundsMaximums_.view(),
                                       boundsStatus_.view()};
    cudaError_t const launch = BuildCurveTileBounds({basis}, input, workspace, output, stream_);
    if (launch != cudaSuccess) {
        // The helper can have queued work before reporting a later launch
        // error. Retire the same-stream H2D span copy before hostSpans dies.
        gpu::CudaFenceStream(stream_);
        return Fail("point-edit bounds launch failed");
    }
    uint32_t status = 0;
    cudaError_t const statusCopy = cudaMemcpyAsync(&status, boundsStatus_.data(),
        sizeof(status), cudaMemcpyDeviceToHost, stream_);
    // Always retire every queued launch even if status-copy enqueue failed.
    cudaError_t const statusFence = gpu::CudaFenceStream(stream_);
    if (statusCopy != cudaSuccess || statusFence != cudaSuccess || status != 0)
        return Fail("point-edit bounds validation failed");

    std::vector<float3> minimums(count), maximums(count);
    cudaError_t minimumCopy = cudaSuccess, maximumCopy = cudaSuccess;
    if (count) {
        minimumCopy = cudaMemcpyAsync(minimums.data(), boundsMinimums_.data(),
            count * sizeof(float3), cudaMemcpyDeviceToHost, stream_);
        maximumCopy = cudaMemcpyAsync(maximums.data(), boundsMaximums_.data(),
            count * sizeof(float3), cudaMemcpyDeviceToHost, stream_);
    }
    cudaError_t const boundsFence = gpu::CudaFenceStream(stream_);
    if (minimumCopy != cudaSuccess || maximumCopy != cudaSuccess ||
        boundsFence != cudaSuccess)
        return Fail("point-edit bounds readback failed");
    *tiles = geometry.tiles;
    for (size_t i = 0; i != count; ++i) {
        float const minimum[3] = {minimums[i].x, minimums[i].y, minimums[i].z};
        float const maximum[3] = {maximums[i].x, maximums[i].y, maximums[i].z};
        for (size_t axis = 0; axis != 3; ++axis) {
            if (!std::isfinite(minimum[axis]) || !std::isfinite(maximum[axis]) ||
                minimum[axis] > maximum[axis])
                return Fail("point-edit bounds readback is not conservative");
            (*tiles)[i].extentMin[axis] = minimum[axis];
            (*tiles)[i].extentMax[axis] = maximum[axis];
        }
        (*tiles)[i].boundsValid = true;
    }
    *boundsReady = true;
    return true;
}

bool CudaToolSession::Begin(uint64_t graphVersion) {
    if (base_) return Fail("a device stroke is already active");
    auto snapshot = engine_.Generation();
    if (!snapshot || !snapshot->device || engine_.NeedsCommit())
        return Fail("device stroke requires a clean published GPU snapshot");
    auto lease = AcquireGeometry(snapshot->device, stream_);
    if (!lease) return Fail("cannot acquire the device stroke base on this stream");
    if (!engine_.BeginDeviceEdit(snapshot, &editToken_, &diagnostic_)) return false;
    base_ = std::move(snapshot);
    expected_ = base_;
    latest_ = base_->device;
    graphVersion_ = graphVersion;
    diagnostic_.clear();
    return true;
}

bool CudaToolSession::UpdateIndexed(DeviceView<const int32_t> indices,
                                   DeviceView<const float3> absolutePositions) {
    auto current = engine_.Generation();
    if (!CheckCurrent(current)) return false;
    auto lease = AcquireGeometry(base_->device, stream_);
    if (!lease) return Fail("cannot acquire press-time geometry");
    auto status = override_.Apply(lease.Geometry().points, indices, absolutePositions, stream_);
    if (status == PointOverrideStatus::Ok) status = override_.Finish(stream_);
    if (status != PointOverrideStatus::Ok) return Fail("invalid or failed sparse GPU point edit");
    std::vector<UsdGenDeviceTileMetadata> tiles;
    bool boundsReady = false;
    if (!RefreshTileBounds(lease, override_.view(), &tiles, &boundsReady)) return false;
    auto points = override_.releaseCompleted();
    if (!points) return Fail("GPU point edit did not produce a completed allocation");
    auto revision = MakePointRevisionGeneration(base_->device, uint64_t(current->id + 1),
        std::move(points), {"pointEdit", graphVersion_, base_->device->Identity().generation}, &diagnostic_);
    if (revision && boundsReady)
        revision = WithTileMetadata(revision, std::move(tiles), &diagnostic_);
    if (!revision || !engine_.StageDeviceRevision(current, revision, editToken_, &diagnostic_)) return false;
    latest_ = staged_ = std::move(revision);
    expected_ = current;
    diagnostic_.clear();
    return true;
}

bool CudaToolSession::Cancel() {
    auto current = engine_.Generation();
    if (!CheckCurrent(current)) return false;
    if (current != base_) {
        auto revision = MakePointRevisionGeneration(base_->device, uint64_t(current->id + 1), {},
            {"pointEdit.cancel", graphVersion_, base_->device->Identity().generation}, &diagnostic_);
        if (!revision || !engine_.StageDeviceRevision(current, revision, editToken_, &diagnostic_)) return false;
        // The engine owns the staged restoration. Close/destruction must not
        // remove it before the app's subsequent LiveOverride commit.
    }
    const bool ended = engine_.EndDeviceEdit(editToken_, current, current == base_);
    editToken_ = 0;
    staged_.reset(); latest_.reset(); base_.reset(); expected_.reset();
    if (!ended) return Fail("device stroke was superseded before cancellation");
    diagnostic_.clear();
    return true;
}

std::shared_ptr<const UsdGenDeviceGeneration> CudaToolSession::Close() {
    auto current = engine_.Generation();
    bool stale = base_ && !CheckCurrent(current);
    if (editToken_) {
        if (!engine_.EndDeviceEdit(editToken_, current)) stale = true;
        editToken_ = 0;
    }
    auto result = std::move(latest_);
    staged_.reset(); base_.reset(); expected_.reset();
    return stale ? nullptr : result;
}

bool CudaToolSession::Pick(PickQuery query, PickResult* result) {
    if (!result) return Fail("pick result pointer is null");
    auto current = engine_.Generation();
    if (base_ && !CheckCurrent(current)) return false;
    auto lease = current ? AcquireGeometry(base_ ? base_->device : current->device, stream_) : CudaGeometryLease{};
    if (!lease) return Fail("pick requires a published GPU snapshot on this stream");
    auto status = picking_.ApplyPick(lease.Geometry(), query, stream_);
    if (status == PickingStatus::Ok) status = picking_.Finish(stream_);
    if (status != PickingStatus::Ok) return Fail(picking_.diagnostic());
    *result = picking_.result();
    diagnostic_.clear();
    return true;
}

bool CudaToolSession::Footprint(PickQuery query) {
    auto current = engine_.Generation();
    if (base_ && !CheckCurrent(current)) return false;
    auto lease = current ? AcquireGeometry(base_ ? base_->device : current->device, stream_) : CudaGeometryLease{};
    if (!lease) return Fail("footprint requires a published GPU snapshot on this stream");
    auto status = picking_.ApplyFootprint(lease.Geometry(), query, stream_);
    if (status == PickingStatus::Ok) status = picking_.Finish(stream_);
    if (status != PickingStatus::Ok) return Fail(picking_.diagnostic());
    diagnostic_.clear();
    return true;
}

} // namespace usdGen::gpu
#endif
