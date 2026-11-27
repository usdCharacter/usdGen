#include "usdGenSurface/cuda/rootFrameGather.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenCuda/gpu/cudaCompat.h"

#include <algorithm>
#include <utility>

namespace usdGen::gpu {
namespace {
constexpr int kBadIds=2, kMissingId=3;
__device__ void Error(int* error, int code) { atomicCAS(error, 0, code); }
__global__ void ValidateIds(DeviceView<const uint64_t> sourceIds,
    DeviceView<const uint64_t> survivors, uint32_t* count, int* error) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *count = static_cast<uint32_t>(survivors.size);
    size_t const work = sourceIds.size > survivors.size ? sourceIds.size : survivors.size;
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<work; i+=size_t(blockDim.x)*gridDim.x) {
        if (i < sourceIds.size && i && sourceIds.data[i-1] >= sourceIds.data[i]) Error(error,kBadIds);
        if (i < survivors.size && i && survivors.data[i-1] >= survivors.data[i]) Error(error,kBadIds);
    }
}
__global__ void Gather(DeviceView<const uint64_t> sourceIds, RestRootFrames source,
    DeviceView<const float3> sourceOrigins, DeviceView<const uint8_t> sourceValid,
    DeviceView<const uint8_t> sourceDrop, DeviceView<const uint64_t> survivors,
    DeviceView<float3> origins, DeviceView<float3> tangent, DeviceView<float3> binormal,
    DeviceView<float3> normal, DeviceView<uint8_t> valid, DeviceView<uint8_t> drop,
    int* error) {
    if (atomicAdd(error, 0)) return;
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<survivors.size; i+=size_t(blockDim.x)*gridDim.x) {
        uint64_t const wanted=survivors.data[i]; size_t lo=0, hi=sourceIds.size;
        while (lo < hi) { size_t mid=lo+(hi-lo)/2; if (sourceIds.data[mid] < wanted) lo=mid+1; else hi=mid; }
        if (lo == sourceIds.size || sourceIds.data[lo] != wanted) { Error(error,kMissingId); continue; }
        origins.data[i]=sourceOrigins.data[lo]; tangent.data[i]=source.tangent.data[lo];
        binormal.data[i]=source.binormal.data[lo]; normal.data[i]=source.normal.data[lo];
        valid.data[i]=sourceValid.data[lo]; drop.data[i]=sourceDrop.data[lo];
    }
}
unsigned Blocks(size_t n) { return static_cast<unsigned>(n ? std::min<size_t>((n+255)/256,65535) : 1); }
cudaError_t CheckStream(int device, cudaStream_t stream) {
    int current=-1; if (cudaGetDevice(&current)!=cudaSuccess) return cudaErrorUnknown;
    if (device>=0 && current!=device) return cudaErrorInvalidDevice;
    if (stream) { int streamDevice=-1; if (cudaStreamGetDevice(stream,&streamDevice)!=cudaSuccess || streamDevice!=current) return cudaErrorInvalidDevice; }
    return cudaSuccess;
}
}
void CudaRootFrameGather::Storage::clear() noexcept { origins.reset(0); tangent.reset(0); binormal.reset(0); normal.reset(0); valid.reset(0); drop.reset(0); count=0; }
void CudaRootFrameGather::Storage::quarantine() noexcept { origins.quarantine(); tangent.quarantine(); binormal.quarantine(); normal.quarantine(); valid.quarantine(); drop.quarantine(); count=0; }
void CudaRootFrameGather::Storage::swap(Storage& o) noexcept { using std::swap; swap(origins,o.origins); swap(tangent,o.tangent); swap(binormal,o.binormal); swap(normal,o.normal); swap(valid,o.valid); swap(drop,o.drop); swap(count,o.count); }
cudaError_t CudaRootFrameGather::Storage::recordUse(cudaStream_t s) { cudaError_t e=origins.recordUse(s); if(e==cudaSuccess)e=tangent.recordUse(s); if(e==cudaSuccess)e=binormal.recordUse(s); if(e==cudaSuccess)e=normal.recordUse(s); if(e==cudaSuccess)e=valid.recordUse(s); if(e==cudaSuccess)e=drop.recordUse(s); return e; }
cudaError_t CudaRootFrameGather::Storage::waitOn(cudaStream_t s) const { cudaError_t e=origins.waitOn(s); if(e==cudaSuccess)e=tangent.waitOn(s); if(e==cudaSuccess)e=binormal.waitOn(s); if(e==cudaSuccess)e=normal.waitOn(s); if(e==cudaSuccess)e=valid.waitOn(s); if(e==cudaSuccess)e=drop.waitOn(s); return e; }
CudaRootFrameGather::~CudaRootFrameGather() {
    int previous = -1;
    bool const owns = ready_ || active_.count || pendingStorage_.count || error_.size();
    bool const selected = !owns || (cudaGetDevice(&previous) == cudaSuccess &&
        deviceIndex_ >= 0 && cudaSetDevice(deviceIndex_) == cudaSuccess);
    // shutdown wait: destructor.
    bool const done = !owns || (selected && (!ready_ || cudaEventSynchronize(ready_) == cudaSuccess) &&
        (!active_.count || (active_.waitOn(nullptr) == cudaSuccess && cudaStreamSynchronize(nullptr) == cudaSuccess)));
    if (!selected || !done) {
        active_.quarantine(); pendingStorage_.quarantine(); error_.quarantine(); pendingCount_.quarantine(); ready_ = nullptr;
        if (selected && previous >= 0 && previous != deviceIndex_) cudaSetDevice(previous);
        return;
    }
    if (ready_) cudaEventDestroy(ready_);
    active_.clear(); pendingStorage_.clear(); error_.reset(0); pendingCount_.reset(0);
    if (previous >= 0 && previous != deviceIndex_) cudaSetDevice(previous);
}
RootFrameGatherStatus CudaRootFrameGather::fail(RootFrameGatherStatus s,char const* m){diagnostic_=m;return s;} cudaError_t CudaRootFrameGather::validateStream(cudaStream_t s)const{return CheckStream(deviceIndex_,s);} void CudaRootFrameGather::discardPending()noexcept{pendingStorage_.clear();pendingCount_.reset(0);pending_=false;} void CudaRootFrameGather::quarantinePending()noexcept{pendingStorage_.quarantine();pendingCount_.quarantine();error_.quarantine();pending_=false;}
RootFrameGatherStatus CudaRootFrameGather::Apply(DeviceView<const uint64_t> sourceIds, RestRootFrames source, DeviceView<const float3> sourceOrigins, DeviceView<const uint8_t> sourceValid, DeviceView<const uint8_t> sourceDrop, DeviceView<const uint64_t> survivors, cudaStream_t stream) {
    if(pending_)return fail(RootFrameGatherStatus::InvalidArgument,"Finish is required before Apply"); if(validateStream(stream)!=cudaSuccess)return fail(RootFrameGatherStatus::InvalidArgument,"wrong CUDA stream device"); if(deviceIndex_<0&&cudaGetDevice(&deviceIndex_)!=cudaSuccess)return fail(RootFrameGatherStatus::CudaError,"cannot identify CUDA device");
    size_t n=sourceIds.size,m=survivors.size; if((n&&!sourceIds.data)||(m&&!survivors.data)||source.tangent.size!=n||source.binormal.size!=n||source.normal.size!=n||sourceOrigins.size!=n||sourceValid.size!=n||sourceDrop.size!=n||(n&&!sourceOrigins.data)||(n&&!source.tangent.data)||(n&&!source.binormal.data)||(n&&!source.normal.data)||(n&&!sourceValid.data)||(n&&!sourceDrop.data)||m>UINT32_MAX)return fail(RootFrameGatherStatus::InvalidArgument,"invalid root-frame gather view");
    if(!ready_&&cudaEventCreateWithFlags(&ready_,cudaEventDisableTiming)!=cudaSuccess)return fail(RootFrameGatherStatus::CudaError,"gather event allocation failed"); pendingStorage_.clear();pendingStorage_.count=m;
    if(error_.reset(1)!=cudaSuccess||pendingCount_.reset(1)!=cudaSuccess||pendingStorage_.origins.reset(m)!=cudaSuccess||pendingStorage_.tangent.reset(m)!=cudaSuccess||pendingStorage_.binormal.reset(m)!=cudaSuccess||pendingStorage_.normal.reset(m)!=cudaSuccess||pendingStorage_.valid.reset(m)!=cudaSuccess||pendingStorage_.drop.reset(m)!=cudaSuccess||cudaMemsetAsync(error_.data(),0,sizeof(int),stream)!=cudaSuccess){quarantinePending();return fail(RootFrameGatherStatus::CudaError,"gather allocation failed");}
    ValidateIds<<<Blocks(n>m?n:m),256,0,stream>>>(sourceIds,survivors,pendingCount_.data(),error_.data());
    if(cudaGetLastError()!=cudaSuccess){quarantinePending();return fail(RootFrameGatherStatus::CudaError,"gather ID validation launch failed");}
    Gather<<<Blocks(m),256,0,stream>>>(sourceIds,source,sourceOrigins,sourceValid,sourceDrop,survivors,pendingStorage_.origins.view(),pendingStorage_.tangent.view(),pendingStorage_.binormal.view(),pendingStorage_.normal.view(),pendingStorage_.valid.view(),pendingStorage_.drop.view(),error_.data());
    if(cudaGetLastError()!=cudaSuccess||cudaEventRecord(ready_,stream)!=cudaSuccess){quarantinePending();return fail(RootFrameGatherStatus::CudaError,"gather launch failed");} pending_=true;diagnostic_.clear();return RootFrameGatherStatus::Ok;
}
RootFrameGatherStatus CudaRootFrameGather::Finish(cudaStream_t stream) {
    if (validateStream(stream) != cudaSuccess)
        return fail(RootFrameGatherStatus::InvalidArgument, "wrong completion stream device");
    if (!pending_) return RootFrameGatherStatus::NoPendingUpdate;
    int error = 0; uint32_t count = 0;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(&error, error_.data(), sizeof(error), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(&count, pendingCount_.data(), sizeof(count), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess) {
        quarantinePending(); return fail(RootFrameGatherStatus::CudaError, "gather completion failed");
    }
    if (error) { discardPending(); return fail(error == kMissingId ? RootFrameGatherStatus::MissingStableId : RootFrameGatherStatus::InvalidArgument, "invalid stable-ID gather input"); }
    if (count != pendingStorage_.count) { discardPending(); return fail(RootFrameGatherStatus::CudaError, "gathered count mismatch"); }
    if (active_.count && (active_.waitOn(stream) != cudaSuccess || CudaFenceStream(stream) != cudaSuccess)) { quarantinePending(); return fail(RootFrameGatherStatus::CudaError, "active gather consumer fence failed"); }
    active_.swap(pendingStorage_); pendingStorage_.clear(); pendingCount_.reset(0); pending_ = false; ++generation_; diagnostic_.clear();
    return RootFrameGatherStatus::Ok;
}
RestRootFrames CudaRootFrameGather::frames()const noexcept{return{active_.tangent.view(),active_.binormal.view(),active_.normal.view()};} DeviceView<const float3> CudaRootFrameGather::origins()const noexcept{return active_.origins.view();} DeviceView<const uint8_t> CudaRootFrameGather::valid()const noexcept{return active_.valid.view();} DeviceView<const uint8_t> CudaRootFrameGather::dropMask()const noexcept{return active_.drop.view();} cudaError_t CudaRootFrameGather::recordUse(cudaStream_t s){return validateStream(s)==cudaSuccess?active_.recordUse(s):cudaErrorInvalidDevice;} cudaError_t CudaRootFrameGather::waitOn(cudaStream_t s)const{return validateStream(s)==cudaSuccess?active_.waitOn(s):cudaErrorInvalidDevice;}
} // namespace usdGen::gpu
