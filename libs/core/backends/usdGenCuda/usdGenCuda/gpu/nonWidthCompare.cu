#include "nonWidthCompare.h"
#include "cudaCompat.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace usdGen::gpu {
namespace {
constexpr int kPending = std::numeric_limits<int>::min();

__device__ void Bad(int* p) { atomicCAS(p, 0, 1); }
__device__ void Different(int* p) { atomicExch(p, 0); }
__global__ void Init(CurveFullNonWidthCompareResult* r, int equal) {
    if (!blockIdx.x && !threadIdx.x) { r->error = 0; r->equal = equal; }
}
__global__ void Topology(DeviceCurveGeometryView a, DeviceCurveGeometryView b,
                         CurveFullNonWidthCompareResult* r) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t c = i; c < a.curveCount; c += stride) {
        uint32_t x = a.curveOffsets.data[c], y = a.curveOffsets.data[c + 1];
        if (y <= x || y > a.pointCount) Bad(&r->error);
    }
    for (size_t c = i; c < b.curveCount; c += stride) {
        uint32_t x = b.curveOffsets.data[c], y = b.curveOffsets.data[c + 1];
        if (y <= x || y > b.pointCount) Bad(&r->error);
    }
    if (!i) {
        if (a.curveOffsets.data[0] != 0 || a.curveOffsets.data[a.curveCount] != a.pointCount ||
            b.curveOffsets.data[0] != 0 || b.curveOffsets.data[b.curveCount] != b.pointCount)
            Bad(&r->error);
        if (a.curveCount != b.curveCount || a.pointCount != b.pointCount) Different(&r->equal);
    }
    for (size_t c = i, n = a.curveCount < b.curveCount ? a.curveCount : b.curveCount; c < n; c += stride)
        if (a.stableIds.data[c] != b.stableIds.data[c] ||
            a.curveOffsets.data[c] != b.curveOffsets.data[c] ||
            a.curveOffsets.data[c + 1] != b.curveOffsets.data[c + 1]) Different(&r->equal);
}
template<class T> __global__ void Exact(DeviceView<const T> a, DeviceView<const T> b, size_t n, int* equal) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x, stride = size_t(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) if (!(a.data[i] == b.data[i])) Different(equal);
}
__global__ void F2(DeviceView<const float2> a, DeviceView<const float2> b, size_t n, int* equal) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x, stride = size_t(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) if (!(a.data[i].x == b.data[i].x && a.data[i].y == b.data[i].y)) Different(equal);
}
__global__ void F3(DeviceView<const float3> a, DeviceView<const float3> b, size_t n, int* equal) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x, stride = size_t(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) if (!(a.data[i].x == b.data[i].x && a.data[i].y == b.data[i].y && a.data[i].z == b.data[i].z)) Different(equal);
}
__global__ void ValidateFrameIds(DeviceView<const uint64_t> frameIds,
                                 DeviceView<const uint64_t> geometryIds,
                                 size_t n, int* error) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x, stride = size_t(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) if (frameIds.data[i] != geometryIds.data[i]) Bad(error);
}
unsigned Blocks(size_t n) {
    size_t const blocks = n / 256 + (n % 256 != 0);
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(blocks, 65535)));
}
cudaError_t Pointer(void const* p, size_t n, int d) {
    if (!n) return cudaSuccess;
    if (!p) return cudaErrorInvalidValue;
    cudaPointerAttributes a{}; cudaError_t s = cudaPointerGetAttributes(&a, p);
    return s != cudaSuccess ? s : (a.type == cudaMemoryTypeDevice && a.device == d ? cudaSuccess : cudaErrorInvalidDevicePointer);
}
template<class T> cudaError_t Pointer(DeviceView<const T> p, int d) { return Pointer(p.data, p.size, d); }
bool Mul(size_t a, size_t b, size_t* o) { if (a && b > std::numeric_limits<size_t>::max()/a) return false; *o = a*b; return true; }
bool Type(UsdGenDeviceValueType x) { return x >= UsdGenDeviceValueType::Float32 && x <= UsdGenDeviceValueType::UInt64; }
bool Domain(UsdGenDeviceDomain x) { return x >= UsdGenDeviceDomain::Groom && x <= UsdGenDeviceDomain::Tool; }
bool Semantic(UsdGenDeviceChannelSemantic x) { return x >= UsdGenDeviceChannelSemantic::Generic && x <= UsdGenDeviceChannelSemantic::RootUV; }
uint32_t Components(UsdGenDeviceValueType x) { return x == UsdGenDeviceValueType::Float32x2 ? 2 : x == UsdGenDeviceValueType::Float32x3 ? 3 : x == UsdGenDeviceValueType::Float32x4 ? 4 : 1; }
size_t Scalar(UsdGenDeviceValueType x) { return x == UsdGenDeviceValueType::UInt64 ? 8 : 4; }
cudaError_t Frames(RestRootFrames const& f, size_t curves, int d) {
    bool any = f.tangent.size || f.binormal.size || f.normal.size || f.stableIds.size;
    if (!any) return cudaSuccess;
    if (f.tangent.size != curves || f.binormal.size != curves || f.normal.size != curves ||
        (f.stableIds.size && f.stableIds.size != curves)) return cudaErrorInvalidValue;
    cudaError_t s = Pointer(f.tangent,d); if (s == cudaSuccess) s = Pointer(f.binormal,d);
    if (s == cudaSuccess) s = Pointer(f.normal,d); if (s == cudaSuccess) s = Pointer(f.stableIds,d); return s;
}
cudaError_t Validate(CurveFullNonWidthInput const& x, int d) {
    auto const& g = x.geometry;
    if (g.curveCount == std::numeric_limits<size_t>::max() || g.curveCount > UINT32_MAX || g.pointCount > UINT32_MAX ||
        g.curveOffsets.size != g.curveCount + 1 || g.stableIds.size != g.curveCount ||
        g.points.size != g.pointCount || (g.restPoints.size && g.restPoints.size != g.pointCount)) return cudaErrorInvalidValue;
    cudaError_t s = Pointer(g.curveOffsets,d); if(s==cudaSuccess)s=Pointer(g.stableIds,d); if(s==cudaSuccess)s=Pointer(g.points,d); if(s==cudaSuccess)s=Pointer(g.restPoints,d); if(s!=cudaSuccess)return s;
    if ((x.hairT.size && x.hairT.size != g.pointCount) || (x.rootPrim.size && x.rootPrim.size != g.curveCount) ||
        (x.rootUV.size && x.rootUV.size != g.curveCount)) return cudaErrorInvalidValue;
    if ((s=Pointer(x.hairT,d))!=cudaSuccess || (s=Pointer(x.rootPrim,d))!=cudaSuccess || (s=Pointer(x.rootUV,d))!=cudaSuccess ||
        (s=Frames(x.frames,g.curveCount,d))!=cudaSuccess) return s;
    for (auto const& chunk : x.chunks) {
        if (chunk.firstCurve > g.curveCount ||
            chunk.curveCount > g.curveCount - chunk.firstCurve ||
            chunk.firstCv > g.pointCount || chunk.liveCount > chunk.curveCount ||
            (chunk.curveCount && chunk.cvCount > (g.pointCount - chunk.firstCv) / chunk.curveCount))
            return cudaErrorInvalidValue;
    }
    std::vector<std::string> names;
    names.reserve(x.named.size());
    for (auto const& p:x.named) { auto const&m=p.metadata; size_t bytes=0, packedStride=0, components=0;
        uint64_t expected = m.domain == UsdGenDeviceDomain::Point ? g.pointCount :
            m.domain == UsdGenDeviceDomain::Primitive ? g.curveCount :
            m.domain == UsdGenDeviceDomain::Groom ? 1 : 0;
        if(m.name.empty() || !Type(m.type) || !Domain(m.domain) || !Semantic(m.semantic) ||
           m.semantic != UsdGenDeviceChannelSemantic::Generic || !m.arity ||
           (m.domain != UsdGenDeviceDomain::Point && m.domain != UsdGenDeviceDomain::Primitive && m.domain != UsdGenDeviceDomain::Groom) ||
           m.elementCount != expected || std::find(names.begin(), names.end(), m.name) != names.end() ||
           !Mul(static_cast<size_t>(m.arity), Components(m.type), &components) ||
           !Mul(components, Scalar(m.type), &packedStride) || m.strideBytes != packedStride || m.elementCount>SIZE_MAX ||
           !Mul(static_cast<size_t>(m.elementCount),m.strideBytes,&bytes) || p.bytes.size!=bytes) return cudaErrorInvalidValue;
        if((s=Pointer(p.bytes,d))!=cudaSuccess || (p.bytes.size && reinterpret_cast<uintptr_t>(p.bytes.data) % Scalar(m.type))) return s == cudaSuccess ? cudaErrorInvalidValue : s;
        names.push_back(m.name);
    } return cudaSuccess;
}
bool SameMeta(UsdGenDeviceChannelMetadata const&a,UsdGenDeviceChannelMetadata const&b) { return a.name==b.name&&a.type==b.type&&a.domain==b.domain&&a.elementCount==b.elementCount&&a.arity==b.arity&&a.strideBytes==b.strideBytes&&a.semantic==b.semantic; }
bool SameDesc(CurveFullNonWidthInput const& a, CurveFullNonWidthInput const& b) {
    if (a.geometry.restPoints.size != b.geometry.restPoints.size || a.hairT.size != b.hairT.size ||
        a.rootPrim.size != b.rootPrim.size || a.rootUV.size != b.rootUV.size ||
        a.frames.tangent.size != b.frames.tangent.size ||
        a.frames.binormal.size != b.frames.binormal.size || a.frames.normal.size != b.frames.normal.size ||
        a.chunks.size() != b.chunks.size() || a.named.size() != b.named.size()) return false;
    for (size_t i = 0; i < a.chunks.size(); ++i) { auto const& x = a.chunks[i]; auto const& y = b.chunks[i]; if (x.firstCurve != y.firstCurve || x.curveCount != y.curveCount || x.liveCount != y.liveCount || x.firstCv != y.firstCv || x.cvCount != y.cvCount || x.tile != y.tile || x.surface != y.surface) return false; }
    for (size_t i = 0; i < a.named.size(); ++i) if (!SameMeta(a.named[i].metadata, b.named[i].metadata)) return false;
    return true;
}
template<class T> cudaError_t Run(cudaStream_t s,DeviceView<const T>a,DeviceView<const T>b,size_t n,int*e){if(!n)return cudaSuccess;Exact<<<Blocks(n),256,0,s>>>(a,b,n,e);return cudaGetLastError();}
cudaError_t Run3(cudaStream_t s,DeviceView<const float3>a,DeviceView<const float3>b,size_t n,int*e){if(!n)return cudaSuccess;F3<<<Blocks(n),256,0,s>>>(a,b,n,e);return cudaGetLastError();}
cudaError_t Run2(cudaStream_t s,DeviceView<const float2>a,DeviceView<const float2>b,size_t n,int*e){if(!n)return cudaSuccess;F2<<<Blocks(n),256,0,s>>>(a,b,n,e);return cudaGetLastError();}
cudaError_t RunNamed(cudaStream_t s,NonWidthNamedPlaneView const&a,NonWidthNamedPlaneView const&b,int*e){size_t n=a.bytes.size/Scalar(a.metadata.type);auto A=a.bytes.data;auto B=b.bytes.data;switch(a.metadata.type){case UsdGenDeviceValueType::Float32:case UsdGenDeviceValueType::Float32x2:case UsdGenDeviceValueType::Float32x3:case UsdGenDeviceValueType::Float32x4:return Run<float>(s,{reinterpret_cast<float const*>(A),n},{reinterpret_cast<float const*>(B),n},n,e);case UsdGenDeviceValueType::Int32:return Run<int32_t>(s,{reinterpret_cast<int32_t const*>(A),n},{reinterpret_cast<int32_t const*>(B),n},n,e);case UsdGenDeviceValueType::UInt32:return Run<uint32_t>(s,{reinterpret_cast<uint32_t const*>(A),n},{reinterpret_cast<uint32_t const*>(B),n},n,e);case UsdGenDeviceValueType::UInt64:return Run<uint64_t>(s,{reinterpret_cast<uint64_t const*>(A),n},{reinterpret_cast<uint64_t const*>(B),n},n,e);}return cudaErrorInvalidValue;}
}

CudaCurveFullNonWidthCompare::~CudaCurveFullNonWidthCompare(){if(unprovenWork_){Quarantine();return;}if(!result_.size()&&!ready_&&!hostResult_)return;int old=-1;bool got=cudaGetDevice(&old)==cudaSuccess;if(device_<0||cudaSetDevice(device_)!=cudaSuccess){Quarantine();return;}if(ready_)cudaEventDestroy(ready_);result_.release();if(hostResult_){if(cudaFreeHost(hostResult_)==cudaSuccess)hostPermit_.Release();else hostPermit_.Abandon();}if(got&&old!=device_)cudaSetDevice(old);}
void CudaCurveFullNonWidthCompare::Quarantine() noexcept {unprovenWork_=true;result_.quarantine();if(quarantineLifetime_)quarantineLifetime_.release();lifetime_.reset();ready_=nullptr;hostResult_=nullptr;hostPermit_.Abandon();}
size_t CudaCurveFullNonWidthCompare::ExclusiveRetainedBytes() const noexcept{return result_.bytes()+hostPermit_.Bytes();}

cudaError_t CudaCurveFullNonWidthCompare::BeginFresh(
    CurveFullNonWidthInput const& a, CurveFullNonWidthInput const& b,
    std::shared_ptr<const void> life, cudaStream_t stream,
    UsdGenExecutionMemoryReservation* reserve) {
    if (running_ || statusEnqueued_ || failed_ || unprovenWork_ || !life)
        return cudaErrorInvalidValue;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return cudaErrorInvalidValue;
    int device = -1, streamDevice = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    if (stream && ((status = cudaStreamGetDevice(stream, &streamDevice)) != cudaSuccess ||
                   streamDevice != device))
        return status == cudaSuccess ? cudaErrorInvalidDevice : status;
    if (device_ >= 0 && device_ != device) return cudaErrorInvalidDevice;
    if ((status = Validate(a, device)) != cudaSuccess ||
        (status = Validate(b, device)) != cudaSuccess)
        return status;
    device_ = device;
    if (!ready_ && (status = cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming)) != cudaSuccess)
        return status;
    if ((status = result_.reset(sizeof(CurveFullNonWidthCompareResult), reserve,
                                UsdGenExecutionResourceKind::Scratch)) != cudaSuccess)
        return status;
    if (!hostResult_) {
        auto permit = TryReserveCudaExecutionBytes(sizeof(CurveFullNonWidthCompareResult),
            UsdGenExecutionResourceKind::Scratch, reserve);
        if (!permit) return cudaErrorMemoryAllocation;
        CurveFullNonWidthCompareResult* host = nullptr;
        status = cudaHostAlloc(reinterpret_cast<void**>(&host), sizeof(*host), cudaHostAllocDefault);
        if (status != cudaSuccess) {
            if (host && cudaFreeHost(host) != cudaSuccess) permit->Abandon();
            return status;
        }
        hostResult_ = host;
        hostPermit_ = std::move(*permit);
    }
    try {
        quarantineLifetime_ = std::make_unique<std::shared_ptr<const void>>(life);
    } catch (...) {
        return cudaErrorMemoryAllocation;
    }
    lifetime_ = std::move(life);
    hostResult_->error = kPending;
    hostResult_->equal = 0;
    unprovenWork_ = true;
    auto failed = [&](cudaError_t error) {
        failed_ = true;
        return error;
    };
    for (auto const* input : {&a, &b})
        for (auto const& plane : input->named)
            if (plane.readyBuffer &&
                (status = plane.readyBuffer->waitOn(stream)) != cudaSuccess)
                return failed(status);

    auto* result = reinterpret_cast<CurveFullNonWidthCompareResult*>(result_.data());
    Init<<<1, 1, 0, stream>>>(result, SameDesc(a, b));
    if ((status = cudaGetLastError()) != cudaSuccess) return failed(status);
    Topology<<<Blocks(std::max(a.geometry.curveCount, b.geometry.curveCount)), 256, 0, stream>>>(
        a.geometry, b.geometry, result);
    if ((status = cudaGetLastError()) != cudaSuccess) return failed(status);
    // Alignment is an input-validity proof, independent of the other operand
    // and of whether their transported values compare equal.
    for (auto const* input : {&a, &b}) {
        auto ids = input->frames.stableIds;
        if (ids.size) {
            ValidateFrameIds<<<Blocks(ids.size), 256, 0, stream>>>(
                ids, input->geometry.stableIds, ids.size, &result->error);
            if ((status = cudaGetLastError()) != cudaSuccess) return failed(status);
        }
    }
    if (a.geometry.points.size == b.geometry.points.size &&
        (status = Run3(stream, a.geometry.points, b.geometry.points,
                       a.geometry.points.size, &result->equal)) != cudaSuccess)
        return failed(status);
    if (a.geometry.restPoints.size == b.geometry.restPoints.size &&
        (status = Run3(stream, a.geometry.restPoints, b.geometry.restPoints,
                       a.geometry.restPoints.size, &result->equal)) != cudaSuccess)
        return failed(status);
    if (a.hairT.size == b.hairT.size &&
        (status = Run(stream, a.hairT, b.hairT, a.hairT.size, &result->equal)) != cudaSuccess)
        return failed(status);
    if (a.rootPrim.size == b.rootPrim.size &&
        (status = Run(stream, a.rootPrim, b.rootPrim, a.rootPrim.size, &result->equal)) != cudaSuccess)
        return failed(status);
    if (a.rootUV.size == b.rootUV.size &&
        (status = Run2(stream, a.rootUV, b.rootUV, a.rootUV.size, &result->equal)) != cudaSuccess)
        return failed(status);
    if (a.frames.tangent.size == b.frames.tangent.size &&
        (status = Run3(stream, a.frames.tangent, b.frames.tangent,
                       a.frames.tangent.size, &result->equal)) != cudaSuccess)
        return failed(status);
    if (a.frames.binormal.size == b.frames.binormal.size &&
        (status = Run3(stream, a.frames.binormal, b.frames.binormal,
                       a.frames.binormal.size, &result->equal)) != cudaSuccess)
        return failed(status);
    if (a.frames.normal.size == b.frames.normal.size &&
        (status = Run3(stream, a.frames.normal, b.frames.normal,
                       a.frames.normal.size, &result->equal)) != cudaSuccess)
        return failed(status);
    if (a.named.size() == b.named.size())
        for (size_t i = 0; i < a.named.size(); ++i)
            if (SameMeta(a.named[i].metadata, b.named[i].metadata) &&
                (status = RunNamed(stream, a.named[i], b.named[i], &result->equal)) != cudaSuccess)
                return failed(status);
    for (auto const* input : {&a, &b})
        for (auto const& plane : input->named)
            if (plane.readyBuffer &&
                (status = plane.readyBuffer->recordUse(stream)) != cudaSuccess)
                return failed(status);
    if ((status = result_.recordUse(stream)) != cudaSuccess ||
        (status = cudaEventRecord(ready_, stream)) != cudaSuccess)
        return failed(status);
    running_ = true;
    return cudaSuccess;
}
cudaError_t CudaCurveFullNonWidthCompare::EnqueueFreshStatus(cudaStream_t stream){if(!running_||statusEnqueued_||failed_||!hostResult_)return cudaErrorInvalidValue;cudaStreamCaptureStatus cap=cudaStreamCaptureStatusNone;if(cudaStreamIsCapturing(stream,&cap)!=cudaSuccess||cap!=cudaStreamCaptureStatusNone){failed_=true;return cudaErrorInvalidValue;}int dev=-1,sd=-1;cudaError_t s=cudaGetDevice(&dev);if(s!=cudaSuccess||dev!=device_||(stream&&((s=cudaStreamGetDevice(stream,&sd))!=cudaSuccess||sd!=device_))||(s=cudaStreamWaitEvent(stream,ready_,0))!=cudaSuccess||(s=cudaMemcpyAsync(hostResult_,result_.data(),sizeof(*hostResult_),cudaMemcpyDeviceToHost,stream))!=cudaSuccess){failed_=true;return s==cudaSuccess?cudaErrorInvalidDevice:s;}statusEnqueued_=true;return cudaSuccess;}
cudaError_t CudaCurveFullNonWidthCompare::CommitFreshFinish(bool*equal){if(!equal||!running_||!statusEnqueued_||failed_||!hostResult_||hostResult_->error==kPending)return cudaErrorInvalidValue;running_=statusEnqueued_=false;unprovenWork_=false;lifetime_.reset();quarantineLifetime_.reset();if(hostResult_->error)return cudaErrorInvalidValue;*equal=hostResult_->equal!=0;return cudaSuccess;}
} // namespace usdGen::gpu
