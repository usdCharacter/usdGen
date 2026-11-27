#include "usdGenExpr/cuda/expressionContext.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenCuda/gpu/cudaCompat.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <iterator>
#include <limits>
#include <utility>

namespace usdGen::gpu {
namespace {
using expr::Domain;
using expr::Variable;

constexpr int kBadGeometry = 1;
constexpr int kBadChannel = 2;

ExpressionContextStatus FinishStatus(int error) {
    return error == kBadChannel ? ExpressionContextStatus::InvalidChannel
                                : ExpressionContextStatus::InvalidGeometry;
}

__device__ void Fail(int* error, int code) { atomicCAS(error, 0, code); }
__device__ int Error(int const* error) {
    return atomicAdd(const_cast<int*>(error), 0);
}
__device__ bool Finite(float3 const& p) {
    return isfinite(p.x) && isfinite(p.y) && isfinite(p.z);
}
__device__ bool Finite(float2 const& p) { return isfinite(p.x) && isfinite(p.y); }

__global__ void Validate(DeviceCurveGeometryView g,
                         ExpressionGeometryChannels channels, int* error) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (!g.curveOffsets.data ||
            g.curveOffsets.size != g.curveCount + 1 ||
            g.curveOffsets.data[0] != 0 ||
            g.curveOffsets.data[g.curveCount] != g.pointCount)
            Fail(error, kBadGeometry);
    }
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < g.curveCount) {
        uint32_t a = g.curveOffsets.data[i], b = g.curveOffsets.data[i + 1];
        if (a >= b || b > g.pointCount) Fail(error, kBadGeometry);
        if (!g.stableIds.data || i >= g.stableIds.size) Fail(error, kBadGeometry);
    }
    if (i < g.pointCount) {
        if (!g.points.data || i >= g.points.size || !Finite(g.points.data[i]))
            Fail(error, kBadGeometry);
        if (g.restPoints.data && (i >= g.restPoints.size || !Finite(g.restPoints.data[i])))
            Fail(error, kBadGeometry);
        if (g.widths.data && (i >= g.widths.size || !isfinite(g.widths.data[i])))
            Fail(error, kBadGeometry);
        if (channels.hairT.data && (i >= channels.hairT.size ||
                                    !isfinite(channels.hairT.data[i]) ||
                                    channels.hairT.data[i] < 0 || channels.hairT.data[i] > 1))
            Fail(error, kBadChannel);
    }
    if (channels.hairT.data && i < g.curveCount) {
        uint32_t begin = g.curveOffsets.data[i], end = g.curveOffsets.data[i + 1];
        if (begin < end && end <= g.pointCount && end > begin + 1 &&
            (channels.hairT.data[begin] != 0.0f ||
                                channels.hairT.data[end - 1] != 1.0f))
            Fail(error, kBadChannel);
    }
    if (i < g.curveCount && channels.rootUV.data &&
        (i >= channels.rootUV.size || !Finite(channels.rootUV.data[i])))
        Fail(error, kBadChannel);
    // The rest root frame planes. rootPrim is an integer face index and is
    // deliberately not checked for finiteness.
    if (i < g.curveCount) {
        if (channels.rootN.data &&
            (i >= channels.rootN.size || !Finite(channels.rootN.data[i])))
            Fail(error, kBadChannel);
        if (channels.rootT.data &&
            (i >= channels.rootT.size || !Finite(channels.rootT.data[i])))
            Fail(error, kBadChannel);
        if (channels.rootB.data &&
            (i >= channels.rootB.size || !Finite(channels.rootB.data[i])))
            Fail(error, kBadChannel);
    }
}

__global__ void BuildOwnersAndArc(DeviceCurveGeometryView g,
                                  uint32_t* owners, double* arc, int* error) {
    size_t c = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.curveCount || Error(error) != 0) return;
    uint32_t begin = g.curveOffsets.data[c], end = g.curveOffsets.data[c + 1];
    double total = 0.0;
    for (uint32_t p = begin; p < end; ++p) {
        owners[p] = static_cast<uint32_t>(c);
        if (p > begin) {
            float3 a = g.points.data[p - 1], b = g.points.data[p];
            double dx = double(b.x) - a.x, dy = double(b.y) - a.y, dz = double(b.z) - a.z;
            total += sqrt(dx * dx + dy * dy + dz * dz);
        }
        arc[p] = total;
    }
}

__device__ double T(DeviceCurveGeometryView g, ExpressionGeometryChannels ch,
                    size_t c, size_t p, double const* arc) {
    if (ch.hairT.data) return ch.hairT.data[p];
    uint32_t end = g.curveOffsets.data[c + 1];
    const double total = arc[end - 1];
    return total > 0.0 ? arc[p] / total : 0.0;
}
__device__ void Set3(ExpressionInputs const& out, Variable v, size_t i, float3 p) {
    double* dst = const_cast<double*>(out.fields[(unsigned)v].data);
    dst[i * 3 + 0] = p.x; dst[i * 3 + 1] = p.y; dst[i * 3 + 2] = p.z;
}
__device__ void Set1(ExpressionInputs const& out, Variable v, size_t i, double value) {
    const_cast<double*>(out.fields[(unsigned)v].data)[i] = value;
}

__global__ void BuildFields(DeviceCurveGeometryView g,
                            ExpressionGeometryChannels ch, Domain domain,
                            double const* arc, ExpressionInputs out, int* error) {
    if (Error(error) != 0) return;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (domain == Domain::Primitive) {
        if (i >= g.curveCount) return;
        uint32_t p = g.curveOffsets.data[i];
        Set3(out, Variable::P, i, g.points.data[p]);
        Set3(out, Variable::RootP, i, g.points.data[p]);
        if (g.restPoints.data) { Set3(out, Variable::PRef, i, g.restPoints.data[p]); Set3(out, Variable::RootPRef, i, g.restPoints.data[p]); }
        if (g.widths.data) Set1(out, Variable::CWidth, i, g.widths.data[p]);
        Set1(out, Variable::T, i, 0.0);
        Set1(out, Variable::PrimIndex, i, double(i));
        Set1(out, Variable::PrimCount, i, double(g.curveCount));
        Set1(out, Variable::CLength, i, arc[g.curveOffsets.data[i + 1] - 1]);
        uint64_t id = g.stableIds.data[i];
        Set1(out, Variable::IdLo, i, double(uint32_t(id)));
        Set1(out, Variable::IdHi, i, double(uint32_t(id >> 32)));
        Set1(out, Variable::Id, i, double(id));
        if (ch.rootUV.data) { Set1(out, Variable::U, i, ch.rootUV.data[i].x); Set1(out, Variable::V, i, ch.rootUV.data[i].y); }
        if (ch.rootPrim.data) Set1(out, Variable::FaceId, i, double(ch.rootPrim.data[i]));
        // usdGen keeps ONE rest root frame: the animated and the rest spelling
        // of each variable read the same plane.
        if (ch.rootN.data) { Set3(out, Variable::N, i, ch.rootN.data[i]); Set3(out, Variable::NRef, i, ch.rootN.data[i]); }
        if (ch.rootT.data) { Set3(out, Variable::DPdu, i, ch.rootT.data[i]); Set3(out, Variable::DPduRef, i, ch.rootT.data[i]); }
        if (ch.rootB.data) { Set3(out, Variable::DPdv, i, ch.rootB.data[i]); Set3(out, Variable::DPdvRef, i, ch.rootB.data[i]); }
    } else if (domain == Domain::Point) {
        if (i >= g.pointCount) return;
        uint32_t c = out.pointToPrimitive.data[i];
        uint32_t root = g.curveOffsets.data[c];
        Set3(out, Variable::P, i, g.points.data[i]); Set3(out, Variable::RootP, i, g.points.data[root]);
        if (g.restPoints.data) { Set3(out, Variable::PRef, i, g.restPoints.data[i]); Set3(out, Variable::RootPRef, i, g.restPoints.data[root]); }
        if (g.widths.data) Set1(out, Variable::CWidth, i, g.widths.data[i]);
        Set1(out, Variable::T, i, T(g, ch, c, i, arc));
        Set1(out, Variable::PrimIndex, i, double(c));
        Set1(out, Variable::PrimCount, i, double(g.curveCount));
        Set1(out, Variable::CLength, i, arc[g.curveOffsets.data[c + 1] - 1]);
        Set1(out, Variable::PointIndex, i, double(i - root));
        Set1(out, Variable::PointCount, i, double(g.curveOffsets.data[c + 1] - root));
        if (ch.rootUV.data) { Set1(out, Variable::U, i, ch.rootUV.data[c].x); Set1(out, Variable::V, i, ch.rootUV.data[c].y); }
        if (ch.rootPrim.data) Set1(out, Variable::FaceId, i, double(ch.rootPrim.data[c]));
        // Per-curve root data at point rate reads the CV's owning curve, the
        // same addressing $rootP and $u/$v already use.
        if (ch.rootN.data) { Set3(out, Variable::N, i, ch.rootN.data[c]); Set3(out, Variable::NRef, i, ch.rootN.data[c]); }
        if (ch.rootT.data) { Set3(out, Variable::DPdu, i, ch.rootT.data[c]); Set3(out, Variable::DPduRef, i, ch.rootT.data[c]); }
        if (ch.rootB.data) { Set3(out, Variable::DPdv, i, ch.rootB.data[c]); Set3(out, Variable::DPdvRef, i, ch.rootB.data[c]); }
        uint64_t id = g.stableIds.data[c];
        Set1(out, Variable::IdLo, i, double(uint32_t(id)));
        Set1(out, Variable::IdHi, i, double(uint32_t(id >> 32)));
        Set1(out, Variable::Id, i, double(id));
    }
}
} // namespace

CudaExpressionContext::~CudaExpressionContext() {
    if (freshPending_ || unprovenWork_) {
        // Fresh status/callback work is later than ready_.  Do not query or
        // free any CUDA allocation without its terminal proof.
        for (auto& field : fields_) field.quarantine();
        owners_.quarantine(); arcLength_.quarantine(); error_.quarantine();
        ready_ = nullptr;
        freshHostError_ = nullptr;
        freshHostErrorPermit_.Abandon();
        return;
    }
    const bool owns = ready_ || freshHostError_ || error_.size() || owners_.size() ||
        arcLength_.size() || std::any_of(std::begin(fields_), std::end(fields_),
            [](DeviceBuffer<double> const& field) { return field.size() != 0; });
    int previous = -1;
    const bool selected = !owns ||
        (cudaGetDevice(&previous) == cudaSuccess && deviceIndex_ >= 0 &&
         cudaSetDevice(deviceIndex_) == cudaSuccess);
    // shutdown wait: destructor.
    if (!selected || (ready_ && cudaEventSynchronize(ready_) != cudaSuccess)) {
        for (auto& field : fields_) field.quarantine();
        owners_.quarantine(); arcLength_.quarantine(); error_.quarantine();
        ready_ = nullptr;
        freshHostError_ = nullptr;
        freshHostErrorPermit_.Abandon();
        if (selected && previous >= 0 && previous != deviceIndex_)
            cudaSetDevice(previous);
        return;
    }
    if (ready_) { cudaEventDestroy(ready_); ready_ = nullptr; }
    // Release while the producing device remains selected. Member destruction
    // happens after this body and must not perform those frees after restoring
    // the caller's selected device.
    for (auto& field : fields_) field.reset(0);
    owners_.reset(0); arcLength_.reset(0); error_.reset(0);
    if (freshHostError_) {
        if (cudaFreeHost(freshHostError_) == cudaSuccess) freshHostErrorPermit_.Release();
        else freshHostErrorPermit_.Abandon();
        freshHostError_ = nullptr;
    }
    if (selected && previous >= 0 && previous != deviceIndex_) cudaSetDevice(previous);
}

namespace {
ExpressionInputs MakeEmptyInputs() noexcept {
    ExpressionInputs result;
    result.count = 0;
    result.context.domain = Domain::None;
    result.context.count = 0;
    return result;
}
// Namespace scope: initialized at load, no function-local static guard.
ExpressionInputs const g_emptyInputs = MakeEmptyInputs();
}  // namespace

ExpressionInputs const& CudaExpressionContext::Inputs() const {
    return usable_ ? inputs_ : g_emptyInputs;
}

ExpressionContextStatus CudaExpressionContext::Build(
    DeviceCurveGeometryView geometry, ExpressionGeometryChannels channels,
    expr::Context context, cudaStream_t stream,
    UsdGenExecutionMemoryReservation* reservation,
    UsdGenExecutionResourceKind kind) {
    return BuildImpl(geometry, channels, context, stream, false, reservation, kind);
}

ExpressionContextStatus CudaExpressionContext::BuildFresh(
    DeviceCurveGeometryView geometry, ExpressionGeometryChannels channels,
    expr::Context context, cudaStream_t stream,
    UsdGenExecutionMemoryReservation* reservation,
    UsdGenExecutionResourceKind kind) {
    // Capture must be checked before stream-device queries or allocation.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return ExpressionContextStatus::InvalidArgument;
    return BuildImpl(geometry, channels, context, stream, true, reservation, kind);
}

ExpressionContextStatus CudaExpressionContext::BuildImpl(
    DeviceCurveGeometryView geometry, ExpressionGeometryChannels channels,
    expr::Context context, cudaStream_t stream, bool fresh,
    UsdGenExecutionMemoryReservation* reservation,
    UsdGenExecutionResourceKind kind) {
    if (pending_ || freshPending_ || unprovenWork_) return ExpressionContextStatus::InvalidArgument;
    usable_ = false;
    if (context.domain != Domain::Groom && context.domain != Domain::Primitive && context.domain != Domain::Point)
        return ExpressionContextStatus::InvalidArgument;
    auto validView = [](auto view) { return view.data || view.size == 0; };
    if (!validView(geometry.points) || !validView(geometry.restPoints) || !validView(geometry.widths) ||
        !validView(geometry.curveOffsets) || !validView(geometry.stableIds) ||
        !validView(channels.hairT) || !validView(channels.rootUV) ||
        !validView(channels.rootN) || !validView(channels.rootT) ||
        !validView(channels.rootB) || !validView(channels.rootPrim))
        return ExpressionContextStatus::InvalidArgument;
    if (!std::isfinite(context.frame) || !std::isfinite(context.time))
        return ExpressionContextStatus::InvalidArgument;
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) return ExpressionContextStatus::CudaError;
    if (stream) {
        int streamDevice = -1;
        if (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess || streamDevice != current)
            return ExpressionContextStatus::InvalidArgument;
    }
    if (deviceIndex_ >= 0 && deviceIndex_ != current) return ExpressionContextStatus::InvalidArgument;
    deviceIndex_ = current;
    pendingInputs_ = ExpressionInputs{}; pendingInputs_.context = context; pendingInputs_.count = 1;
    if (context.domain == Domain::Groom) {
        if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess) return ExpressionContextStatus::CudaError;
        if (fresh && !freshHostError_) {
            auto permit = TryReserveCudaExecutionBytes(sizeof(int), kind, reservation);
            int* status = nullptr;
            if (!permit || cudaHostAlloc(reinterpret_cast<void**>(&status), sizeof(int), cudaHostAllocDefault) != cudaSuccess)
                return ExpressionContextStatus::CudaError;
            freshHostError_ = status; freshHostErrorPermit_ = std::move(*permit);
        }
        if (error_.reset(1, reservation, kind) != cudaSuccess) return ExpressionContextStatus::CudaError;
        if (fresh) {
            // All allocations are complete. From the first stream operation
            // onward failed terminal installation must quarantine this state.
            unprovenWork_ = true; freshFailed_ = false; freshStatusEnqueued_ = false;
            *freshHostError_ = std::numeric_limits<int>::min();
        }
        if (cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess ||
            cudaEventRecord(ready_, stream) != cudaSuccess) {
            if (fresh) freshFailed_ = true;
            return ExpressionContextStatus::CudaError;
        }
        if (fresh && cudaGetLastError() != cudaSuccess) {
            freshFailed_ = true;
            return ExpressionContextStatus::CudaError;
        }
        pending_ = true; freshPending_ = fresh;
        return ExpressionContextStatus::Ok;
    }
    if ((geometry.curveCount == 0) != (geometry.pointCount == 0) ||
        geometry.curveCount > UINT32_MAX - 1 || geometry.pointCount > UINT32_MAX ||
        !geometry.curveOffsets.data ||
        geometry.curveOffsets.size != geometry.curveCount + 1 ||
        (geometry.pointCount && !geometry.points.data) ||
        geometry.points.size != geometry.pointCount ||
        (geometry.curveCount && !geometry.stableIds.data) ||
        geometry.stableIds.size != geometry.curveCount)
        return ExpressionContextStatus::InvalidGeometry;
    if (geometry.restPoints.data && geometry.restPoints.size != geometry.pointCount)
        return ExpressionContextStatus::InvalidGeometry;
    if (geometry.widths.data && geometry.widths.size != geometry.pointCount)
        return ExpressionContextStatus::InvalidGeometry;
    if (channels.hairT.data && channels.hairT.size != geometry.pointCount)
        return ExpressionContextStatus::InvalidChannel;
    if (channels.rootUV.data && channels.rootUV.size != geometry.curveCount)
        return ExpressionContextStatus::InvalidChannel;
    if ((channels.rootN.data && channels.rootN.size != geometry.curveCount) ||
        (channels.rootT.data && channels.rootT.size != geometry.curveCount) ||
        (channels.rootB.data && channels.rootB.size != geometry.curveCount) ||
        (channels.rootPrim.data && channels.rootPrim.size != geometry.curveCount))
        return ExpressionContextStatus::InvalidChannel;
    size_t n = context.domain == Domain::Groom ? 1 :
               context.domain == Domain::Primitive ? geometry.curveCount : geometry.pointCount;
    pendingInputs_.count = n; pendingInputs_.primitiveCount = geometry.curveCount;
    pendingInputs_.context.count = static_cast<uint32_t>(n);
    const unsigned u = static_cast<unsigned>(expr::Variable::CountVariables);
    for (unsigned v = 0; v < u; ++v) {
        bool vec = v == unsigned(Variable::P) || v == unsigned(Variable::PRef) || v == unsigned(Variable::RootP) || v == unsigned(Variable::RootPRef) ||
            v == unsigned(Variable::N) || v == unsigned(Variable::NRef) ||
            v == unsigned(Variable::DPdu) || v == unsigned(Variable::DPduRef) ||
            v == unsigned(Variable::DPdv) || v == unsigned(Variable::DPdvRef);
        if (vec || v == unsigned(Variable::T) || v == unsigned(Variable::CWidth) || v == unsigned(Variable::PointIndex) || v == unsigned(Variable::PointCount) || v == unsigned(Variable::PrimIndex) || v == unsigned(Variable::PrimCount) || v == unsigned(Variable::CLength) || v == unsigned(Variable::U) || v == unsigned(Variable::V) || v == unsigned(Variable::FaceId) || v == unsigned(Variable::IdLo) || v == unsigned(Variable::IdHi) || v == unsigned(Variable::Id)) {
            if (v == unsigned(Variable::PRef) || v == unsigned(Variable::RootPRef)) { if (!geometry.restPoints.data) continue; }
            if (v == unsigned(Variable::CWidth) && !geometry.widths.data) continue;
            if ((v == unsigned(Variable::U) || v == unsigned(Variable::V)) && !channels.rootUV.data) continue;
            if ((v == unsigned(Variable::N) || v == unsigned(Variable::NRef)) && !channels.rootN.data) continue;
            if ((v == unsigned(Variable::DPdu) || v == unsigned(Variable::DPduRef)) && !channels.rootT.data) continue;
            if ((v == unsigned(Variable::DPdv) || v == unsigned(Variable::DPdvRef)) && !channels.rootB.data) continue;
            if (v == unsigned(Variable::FaceId) && !channels.rootPrim.data) continue;
        if (vec && n > std::numeric_limits<size_t>::max() / 3) return ExpressionContextStatus::InvalidArgument;
            if (context.domain == Domain::Primitive && (v == unsigned(Variable::PointIndex) || v == unsigned(Variable::PointCount))) continue;
            if (fields_[v].reset(n * (vec ? 3 : 1), reservation, kind) != cudaSuccess) return ExpressionContextStatus::CudaError;
            pendingInputs_.fields[v] = {fields_[v].data(), n, context.domain, vec ? 3u : 1u};
        }
    }
    if (owners_.reset(geometry.pointCount, reservation, kind) != cudaSuccess ||
        arcLength_.reset(geometry.pointCount, reservation, kind) != cudaSuccess ||
        error_.reset(1, reservation, kind) != cudaSuccess) return ExpressionContextStatus::CudaError;
    pendingInputs_.pointToPrimitive = {owners_.data(), geometry.pointCount};
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess) return ExpressionContextStatus::CudaError;
    if (fresh && !freshHostError_) {
        auto permit = TryReserveCudaExecutionBytes(sizeof(int), kind, reservation);
        int* status = nullptr;
        if (!permit || cudaHostAlloc(reinterpret_cast<void**>(&status), sizeof(int), cudaHostAllocDefault) != cudaSuccess)
            return ExpressionContextStatus::CudaError;
        freshHostError_ = status; freshHostErrorPermit_ = std::move(*permit);
    }
    if (fresh) {
        unprovenWork_ = true; freshFailed_ = false; freshStatusEnqueued_ = false;
        *freshHostError_ = std::numeric_limits<int>::min();
    }
    if (cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess) {
        if (fresh) freshFailed_ = true;
        return ExpressionContextStatus::CudaError;
    }
    const unsigned blocks = std::max(1u, static_cast<unsigned>((std::max(geometry.curveCount, geometry.pointCount) + 127) / 128));
    Validate<<<blocks,128,0,stream>>>(geometry, channels, error_.data());
    if (cudaGetLastError() != cudaSuccess) {
        if (fresh) freshFailed_ = true;
        return ExpressionContextStatus::CudaError;
    }
    if (geometry.curveCount)
        BuildOwnersAndArc<<<(geometry.curveCount + 127) / 128,128,0,stream>>>(geometry, owners_.data(), arcLength_.data(), error_.data());
    if (geometry.curveCount && cudaGetLastError() != cudaSuccess) {
        if (fresh) freshFailed_ = true;
        return ExpressionContextStatus::CudaError;
    }
    BuildFields<<<blocks,128,0,stream>>>(geometry, channels, context.domain,
                                          arcLength_.data(), pendingInputs_, error_.data());
    if (cudaGetLastError() != cudaSuccess || cudaEventRecord(ready_, stream) != cudaSuccess) {
        if (fresh) freshFailed_ = true;
        return ExpressionContextStatus::CudaError;
    }
    pending_ = true; freshPending_ = fresh;
    return ExpressionContextStatus::Ok;
}

ExpressionContextStatus CudaExpressionContext::Wait(cudaStream_t stream) const {
    return ready_ && cudaStreamWaitEvent(stream, ready_, 0) == cudaSuccess ? ExpressionContextStatus::Ok : ExpressionContextStatus::CudaError;
}

ExpressionContextStatus CudaExpressionContext::EnqueueFreshStatus(cudaStream_t stream) {
    if (!pending_ || !freshPending_ || freshStatusEnqueued_ || freshFailed_ ||
        !ready_ || !freshHostError_)
        return ExpressionContextStatus::InvalidArgument;
    // As for BuildFresh, reject capture before a stream-device query.  A
    // failed terminal installation leaves the already-submitted buffers
    // unproven and therefore quarantined by destruction.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        freshFailed_ = true;
        return ExpressionContextStatus::InvalidArgument;
    }
    int current = -1, streamDevice = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != deviceIndex_ ||
        (stream && (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess ||
                    streamDevice != deviceIndex_)) ||
        cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(freshHostError_, error_.data(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        freshFailed_ = true;
        return ExpressionContextStatus::CudaError;
    }
    freshStatusEnqueued_ = true;
    return ExpressionContextStatus::Ok;
}

ExpressionContextStatus CudaExpressionContext::CommitFreshFinish() {
    if (!pending_ || !freshPending_ || !freshStatusEnqueued_ || freshFailed_)
        return ExpressionContextStatus::InvalidArgument;
    // The sentinel detects an uncompleted D2H status copy. The parent native
    // callback and successful launcher return remain mandatory proof even if
    // that copy happens to have completed before the callback runs.
    if (!freshHostError_ || *freshHostError_ == std::numeric_limits<int>::min())
        return ExpressionContextStatus::InvalidArgument;

    const int error = *freshHostError_;
    pending_ = false;
    freshPending_ = false;
    freshStatusEnqueued_ = false;
    freshFailed_ = false;
    unprovenWork_ = false;
    usable_ = false;
    if (error != 0) {
        ExpressionInputs cleared;
        pendingInputs_ = cleared;
        return FinishStatus(error);
    }
    inputs_ = pendingInputs_;
    usable_ = true;
    return ExpressionContextStatus::Ok;
}

ExpressionContextStatus CudaExpressionContext::Finish(cudaStream_t stream) {
    // A fresh operation has a later D2H/callback than ready_.  Letting the
    // synchronous path consume it would free or publish before that terminal
    // lifetime proof.
    if (freshPending_ || unprovenWork_ || freshFailed_)
        return ExpressionContextStatus::InvalidArgument;
    // The stream waits on ready_; the fence below proves both.
    if (!pending_ || !ready_ || cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess) return ExpressionContextStatus::CudaError;
    int error = 0;
    if (cudaMemcpyAsync(&error, error_.data(), sizeof(error), cudaMemcpyDeviceToHost, stream) != cudaSuccess || CudaFenceStream(stream) != cudaSuccess) return ExpressionContextStatus::CudaError;
    pending_ = false;
    if (error != 0) {
        ExpressionInputs cleared;
        pendingInputs_ = cleared;
        return FinishStatus(error);
    }
    inputs_ = pendingInputs_;
    usable_ = true;
    return ExpressionContextStatus::Ok;
}
} // namespace usdGen::gpu
