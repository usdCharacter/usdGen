#include "curveGrow.h"
#include "cudaCompat.h"

#include "usdGen/executionResources.h"

#include <cmath>
#include <limits>
#include <utility>

namespace usdGen::gpu {
namespace {
constexpr int kInvalidTopology = 1;
constexpr int kNonFinite = 2;
constexpr int kPendingStatus = std::numeric_limits<int>::min();
CurveGrowStatus Status(cudaError_t e) { return e == cudaSuccess ? CurveGrowStatus::Ok : CurveGrowStatus::CudaError; }
bool Finite(float v) { return std::isfinite(v); }
bool Finite(double v) { return std::isfinite(v); }
bool Finite(float3 v) { return Finite(v.x) && Finite(v.y) && Finite(v.z); }

__device__ uint64_t Hash64(uint64_t key, uint32_t salt) {
    uint64_t z = (key ^ (uint64_t(salt) * 0x9E3779B97F4A7C15ull)) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ float DrawGrow(int seed, uint64_t id, uint32_t salt = 0x47726F77u) {
    uint64_t key = Hash64(uint64_t(uint32_t(seed)), salt) ^ id;
    return float(uint32_t(Hash64(key, salt) >> 32) >> 8) * 0x1.0p-24f;
}
__device__ float3 Normalize(float3 v) {
    float l2 = v.x*v.x + v.y*v.y + v.z*v.z;
    if (!(l2 > 1.e-24f) || !isfinite(l2)) return make_float3(0, 1, 0);
    // Keep CPU Grow's `sqrt` then component division, rather than using the
    // faster reciprocal-sqrt approximation (the latter changes oracle bits).
    float length = sqrtf(l2);
    return make_float3(v.x/length, v.y/length, v.z/length);
}
__device__ float3 RotateAroundB(float3 direction, float3 axis, float degrees) {
    if (degrees == 0.0f) return direction;
    axis = Normalize(axis);
    constexpr float pi = 3.14159265358979323846f;
    float const radians = degrees * (pi / 180.0f);
    float const c = cosf(radians), s = sinf(radians);
    float const dot = axis.x * direction.x + axis.y * direction.y + axis.z * direction.z;
    float3 const cross = make_float3(
        axis.y * direction.z - axis.z * direction.y,
        axis.z * direction.x - axis.x * direction.z,
        axis.x * direction.y - axis.y * direction.x);
    float const oneMinusC = 1.0f - c;
    return make_float3(
        direction.x * c + cross.x * s + axis.x * dot * oneMinusC,
        direction.y * c + cross.y * s + axis.y * dot * oneMinusC,
        direction.z * c + cross.z * s + axis.z * dot * oneMinusC);
}
__device__ bool FiniteDevice(float x) { return isfinite(x); }
__device__ bool FiniteDevice(float2 v) { return isfinite(v.x) && isfinite(v.y); }
__device__ bool FiniteDevice(float3 v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z); }

__device__ bool FrameIndex(CurveGrowInput const& input, uint64_t stableId,
                           uint32_t* result) {
    if (!input.frameStableIds.size) { *result = 0; return true; }
    size_t lo = 0, hi = input.frameStableIds.size;
    while (lo < hi) {
        size_t const mid = lo + (hi - lo) / 2;
        uint64_t const value = input.frameStableIds.data[mid];
        if (value < stableId) lo = mid + 1; else hi = mid;
    }
    if (lo >= input.frameStableIds.size || input.frameStableIds.data[lo] != stableId)
        return false;
    *result = static_cast<uint32_t>(lo);
    return true;
}

__global__ void GrowKernel(CurveGrowInput input, uint32_t curves, uint32_t cvCount,
    CurveGrowControls controls, float3* points, float3* rest, float* widths,
    float* hairT, uint32_t* offsets, uint64_t* ids, int32_t* rootPrim,
    float2* rootUV, float3* rootT, float3* rootB, float3* rootN,
    int* error) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= curves) return;
    uint32_t first = input.geometry.curveOffsets.data[c];
    uint32_t last = input.geometry.curveOffsets.data[c + 1];
    if (first >= last || last > input.geometry.pointCount ||
        (c == 0 && first != 0) || (c == curves - 1 && last != input.geometry.pointCount) ||
        last - first < 2) {
        atomicCAS(error, 0, kInvalidTopology); return;
    }
    // The executor's C3 upload already performs this validation.  Retaining
    // it here keeps this public producer fail-closed for direct callers too,
    // rather than making a malformed non-root CV silently disappear.
    for (uint32_t i = first; i < last; ++i) {
        if (!FiniteDevice(input.geometry.points.data[i]) ||
            !FiniteDevice(input.geometry.restPoints.data[i]) ||
            (input.geometry.widths.size &&
             (!FiniteDevice(input.geometry.widths.data[i]) || input.geometry.widths.data[i] < 0))) {
            atomicCAS(error, 0, kNonFinite); return;
        }
    }
    uint64_t id = input.geometry.stableIds.data[c];
    uint32_t frame = c;
    if (input.frameStableIds.size && !FrameIndex(input, id, &frame)) {
        atomicCAS(error, 0, kInvalidTopology); return;
    }
    float3 currentRoot = input.geometry.points.data[first];
    float3 restRoot = input.geometry.restPoints.data[first];
    float3 t = input.rootT.data[frame], b = input.rootB.data[frame], n = input.rootN.data[frame];
    if (!FiniteDevice(currentRoot) || !FiniteDevice(restRoot) || !FiniteDevice(t) ||
        !FiniteDevice(b) || !FiniteDevice(n) || !FiniteDevice(input.rootUV.data[c])) {
        atomicCAS(error, 0, kNonFinite); return;
    }
    float3 direction = controls.direction == CurveGrowDirection::RootNormal ? n :
        controls.direction == CurveGrowDirection::RootTangent ? t : controls.literalDirection;
    direction = Normalize(direction);
    direction = RotateAroundB(direction, b, controls.lift);
    float const azimuth = controls.azimuth + controls.azimuthRandom * 360.0f *
        (DrawGrow(controls.seed, id, 0x4772417Au) - 0.5f); // kSaltGrowAzimuth
    direction = RotateAroundB(direction, n, azimuth);
    // Match CPU Capture: authored length/random math happens in double and
    // the captured per-curve target is then narrowed to float.
    double targetDouble = controls.length * (controls.randomLo +
        double(DrawGrow(controls.seed, id)) * (controls.randomHi - controls.randomLo));
    float target = float(targetDouble);
    if (!isfinite(target)) { atomicCAS(error, 0, kNonFinite); return; }
    uint32_t outputFirst = c * cvCount;
    offsets[c] = outputFirst;
    if (c == curves - 1) offsets[curves] = curves * cvCount;
    uint32_t inputCount = last - first;
    for (uint32_t i = 0; i < cvCount; ++i) {
        float h = float(i) / float(cvCount - 1);
        float distance = target * h;
        if (!isfinite(distance)) { atomicCAS(error, 0, kNonFinite); return; }
        points[outputFirst+i] = make_float3(currentRoot.x + direction.x*distance,
            currentRoot.y + direction.y*distance, currentRoot.z + direction.z*distance);
        rest[outputFirst+i] = make_float3(restRoot.x + direction.x*distance,
            restRoot.y + direction.y*distance, restRoot.z + direction.z*distance);
        if (!FiniteDevice(points[outputFirst+i]) || !FiniteDevice(rest[outputFirst+i])) {
            atomicCAS(error, 0, kNonFinite); return;
        }
        if (input.geometry.widths.size) {
            float position = h * float(inputCount - 1);
            uint32_t lo = detail::CurveGrowWidthLowerIndex(h, inputCount);
            uint32_t hi = min(lo + 1, inputCount - 1);
            float a = position - float(lo);
            float w0 = input.geometry.widths.data[first + lo];
            float w1 = input.geometry.widths.data[first + hi];
            if (!FiniteDevice(w0) || !FiniteDevice(w1) || w0 < 0 || w1 < 0) {
                atomicCAS(error, 0, kNonFinite); return;
            }
            widths[outputFirst+i] = w0 + (w1 - w0) * a;
        } else widths[outputFirst+i] = controls.fallbackWidth;
        if (!isfinite(widths[outputFirst+i])) {
            atomicCAS(error, 0, kNonFinite); return;
        }
        hairT[outputFirst+i] = h;
    }
    ids[c] = id; rootPrim[c] = input.rootPrim.data[c]; rootUV[c] = input.rootUV.data[c];
    rootT[c] = t; rootB[c] = b; rootN[c] = n;
}
__global__ void ValidateEmptyOffset(uint32_t const* input, uint32_t* output, int* error) {
    if (input[0] != 0) atomicCAS(error, 0, kInvalidTopology);
    output[0] = 0;
}
__global__ void ValidateFrameStableIds(uint64_t const* ids, size_t count, int* error) {
    size_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (i && ids[i - 1] >= ids[i]) atomicCAS(error, 0, kInvalidTopology);
}
}

CurveGrowStatus GetCurveGrowRequirements(size_t curves, uint32_t cvs,
                                         CurveGrowRequirements* result) {
    if (!result || cvs < 2 || cvs > 64 || curves > uint64_t(UINT32_MAX) / cvs)
        return CurveGrowStatus::InvalidArgument;
    CurveGrowRequirements r; r.pointCount = curves * size_t(cvs);
    auto add = [](size_t* a, size_t b) { if (b > std::numeric_limits<size_t>::max() - *a) return false; *a += b; return true; };
    auto mul = [](size_t a, size_t b, size_t* out) { if (a && b > std::numeric_limits<size_t>::max() / a) return false; *out=a*b; return true; };
    size_t b = 0;
    if (!mul(r.pointCount, 2*sizeof(float3)+2*sizeof(float), &b) || !add(&r.outputBytes,b) ||
        !mul(curves, 3*sizeof(float3)+sizeof(uint64_t)+sizeof(int32_t)+sizeof(float2)+sizeof(uint32_t),&b) ||
        !add(&r.outputBytes,b) || !add(&r.outputBytes,sizeof(uint32_t))) return CurveGrowStatus::InvalidArgument;
    // One device status plus the pinned host status read by Commit.  Both are
    // retained by a published owner, so requirements include both explicitly.
    r.statusBytes = 2 * sizeof(int); r.peakBytes = r.outputBytes + r.statusBytes;
    *result = r; return CurveGrowStatus::Ok;
}

void CudaCurveGrow::Storage::quarantine() noexcept {
    points.quarantine(); restPoints.quarantine(); widths.quarantine(); hairT.quarantine();
    offsets.quarantine(); stableIds.quarantine(); rootPrim.quarantine(); rootUV.quarantine();
    rootT.quarantine(); rootB.quarantine(); rootN.quarantine();
}
void CudaCurveGrow::Storage::reclassify(UsdGenExecutionResourceKind k) noexcept {
    points.Reclassify(k); restPoints.Reclassify(k); widths.Reclassify(k); hairT.Reclassify(k);
    offsets.Reclassify(k); stableIds.Reclassify(k); rootPrim.Reclassify(k); rootUV.Reclassify(k);
    rootT.Reclassify(k); rootB.Reclassify(k); rootN.Reclassify(k);
}
size_t CudaCurveGrow::Storage::bytes() const noexcept {
    return points.bytes()+restPoints.bytes()+widths.bytes()+hairT.bytes()+offsets.bytes()+
        stableIds.bytes()+rootPrim.bytes()+rootUV.bytes()+rootT.bytes()+rootB.bytes()+rootN.bytes();
}
CurveGrowStatus CudaCurveGrow::Storage::recordUse(cudaStream_t s) {
    cudaError_t e=points.recordUse(s); if(e==cudaSuccess)e=restPoints.recordUse(s);
    if(e==cudaSuccess)e=widths.recordUse(s); if(e==cudaSuccess)e=hairT.recordUse(s);
    if(e==cudaSuccess)e=offsets.recordUse(s); if(e==cudaSuccess)e=stableIds.recordUse(s);
    if(e==cudaSuccess)e=rootPrim.recordUse(s); if(e==cudaSuccess)e=rootUV.recordUse(s);
    if(e==cudaSuccess)e=rootT.recordUse(s); if(e==cudaSuccess)e=rootB.recordUse(s);
    if(e==cudaSuccess)e=rootN.recordUse(s); return Status(e);
}
CurveGrowStatus CudaCurveGrow::Storage::waitOn(cudaStream_t s) const {
    cudaError_t e=points.waitOn(s); if(e==cudaSuccess)e=restPoints.waitOn(s);
    if(e==cudaSuccess)e=widths.waitOn(s); if(e==cudaSuccess)e=hairT.waitOn(s);
    if(e==cudaSuccess)e=offsets.waitOn(s); if(e==cudaSuccess)e=stableIds.waitOn(s);
    if(e==cudaSuccess)e=rootPrim.waitOn(s); if(e==cudaSuccess)e=rootUV.waitOn(s);
    if(e==cudaSuccess)e=rootT.waitOn(s); if(e==cudaSuccess)e=rootB.waitOn(s);
    if(e==cudaSuccess)e=rootN.waitOn(s); return Status(e);
}
CurveGrowStatus CudaCurveGrow::Storage::synchronizeUse() const {
    cudaError_t e=points.synchronizeUse(); if(e==cudaSuccess)e=restPoints.synchronizeUse();
    if(e==cudaSuccess)e=widths.synchronizeUse(); if(e==cudaSuccess)e=hairT.synchronizeUse();
    if(e==cudaSuccess)e=offsets.synchronizeUse(); if(e==cudaSuccess)e=stableIds.synchronizeUse();
    if(e==cudaSuccess)e=rootPrim.synchronizeUse(); if(e==cudaSuccess)e=rootUV.synchronizeUse();
    if(e==cudaSuccess)e=rootT.synchronizeUse(); if(e==cudaSuccess)e=rootB.synchronizeUse();
    if(e==cudaSuccess)e=rootN.synchronizeUse(); return Status(e);
}

CudaCurveGrow::~CudaCurveGrow() {
    if (unprovenWork_) {
        active_.quarantine(); pending_.quarantine(); error_.quarantine();
        (void)inputQuarantineOwner_.release(); hostError_=nullptr;
        hostErrorPermit_.Abandon(); ready_=nullptr; return;
    }
    int prior=-1;
    bool owns=active_.offsets.size()||pending_.offsets.size()||error_.size()||
              ready_||hostError_;
    bool selected=!owns || (cudaGetDevice(&prior)==cudaSuccess&&deviceIndex_>=0&&
                            cudaSetDevice(deviceIndex_)==cudaSuccess);
    bool proved=!owns || (selected&&(!ready_||cudaEventSynchronize(ready_)==cudaSuccess)&&
                          active_.synchronizeUse()==CurveGrowStatus::Ok);
    if (!proved) {
        active_.quarantine(); pending_.quarantine(); error_.quarantine();
        (void)inputQuarantineOwner_.release(); hostError_=nullptr;
        hostErrorPermit_.Abandon(); ready_=nullptr;
    } else {
        if(ready_) cudaEventDestroy(ready_);
        if(hostError_) {
            if(cudaFreeHost(hostError_)==cudaSuccess) hostErrorPermit_.Release();
            else hostErrorPermit_.Abandon();
            hostError_ = nullptr;
        }
    }
    if(selected&&prior>=0&&prior!=deviceIndex_) cudaSetDevice(prior);
}

CurveGrowStatus CudaCurveGrow::validateStream(cudaStream_t stream) const {
    int d=-1;
    if(cudaGetDevice(&d)!=cudaSuccess) return CurveGrowStatus::CudaError;
    if(deviceIndex_>=0&&d!=deviceIndex_) return CurveGrowStatus::InvalidArgument;
    cudaStreamCaptureStatus capture;
    if(cudaStreamIsCapturing(stream,&capture)!=cudaSuccess) return CurveGrowStatus::CudaError;
    if(capture!=cudaStreamCaptureStatusNone) return CurveGrowStatus::InvalidArgument;
    if(stream) {
        int sd=-1;
        if(cudaStreamGetDevice(stream,&sd)!=cudaSuccess) return CurveGrowStatus::CudaError;
        if(sd!=d) return CurveGrowStatus::InvalidArgument;
    }
    return CurveGrowStatus::Ok;
}
CurveGrowStatus CudaCurveGrow::validate(CurveGrowInput const& in,
    std::shared_ptr<const void> const& owner, CurveGrowControls const& c,
    size_t* total) const {
    // Plan lift is an angular root-B rotation, not a translation.  Reject
    // angles outside the schema range before any device allocation.
    if(!owner||!total||c.cvCount<2||c.cvCount>64||!Finite(c.length)||
       !Finite(c.randomLo)||!Finite(c.randomHi)||!Finite(c.lift)||
       !Finite(c.fallbackWidth)||c.length<0||c.randomLo<0||c.randomHi<0||
       c.fallbackWidth<0||c.lift < -90.0f||c.lift > 90.0f||
       !Finite(c.azimuth)||c.azimuth < -360.0f||c.azimuth > 360.0f||
       !Finite(c.azimuthRandom)||c.azimuthRandom < 0.0f||c.azimuthRandom > 1.0f||
       c.direction>CurveGrowDirection::Literal||
       (c.direction==CurveGrowDirection::Literal&&!Finite(c.literalDirection)))
        return CurveGrowStatus::InvalidArgument;
    auto const& g=in.geometry;
    bool const mappedFrames = in.frameStableIds.data != nullptr;
    if(g.curveCount>UINT32_MAX||g.pointCount>UINT32_MAX||
       (g.curveCount==0)!=(g.pointCount==0)||!g.curveOffsets.data||
       g.curveOffsets.size!=g.curveCount+1||
       (g.pointCount&&(!g.points.data||!g.restPoints.data||
                         g.points.size!=g.pointCount||g.restPoints.size!=g.pointCount))||
       (!g.pointCount&&(g.points.size||g.restPoints.size))||
       (g.widths.data&&g.widths.size!=g.pointCount)||(!g.widths.data&&g.widths.size)||
       (g.curveCount&&(!g.stableIds.data||!in.rootPrim.data||!in.rootUV.data||
                         !in.rootT.data||!in.rootB.data||!in.rootN.data))||
       g.stableIds.size!=g.curveCount||in.rootPrim.size!=g.curveCount||
       in.rootUV.size!=g.curveCount||
       in.frameStableIds.size > UINT32_MAX ||
       (mappedFrames
            ? (!in.frameStableIds.size || !in.rootT.data || !in.rootB.data || !in.rootN.data ||
               in.rootT.size != in.frameStableIds.size ||
               in.rootB.size != in.frameStableIds.size ||
               in.rootN.size != in.frameStableIds.size)
            : (in.frameStableIds.size || in.rootT.size != g.curveCount ||
               in.rootB.size != g.curveCount || in.rootN.size != g.curveCount)))
        return CurveGrowStatus::InvalidTopology;
    CurveGrowRequirements r;
    auto s=GetCurveGrowRequirements(g.curveCount,c.cvCount,&r);
    if(s!=CurveGrowStatus::Ok) return s;
    *total=r.pointCount;
    return CurveGrowStatus::Ok;
}
void CudaCurveGrow::discardPending() noexcept {
    pending_=Storage{}; error_.reset(0); inputOwner_.reset();
    inputQuarantineOwner_.reset(); input_={}; pendingWork_=finishScheduled_=false;
    pendingCurves_=pendingPoints_=0;
}
CurveGrowStatus CudaCurveGrow::BeginFresh(CurveGrowInput input,
    std::shared_ptr<const void> owner, CurveGrowControls controls,
    cudaStream_t stream, UsdGenExecutionMemoryReservation* reservation) {
    if(pendingWork_||generation_) return CurveGrowStatus::InvalidArgument;
    auto s=validateStream(stream);
    if(s!=CurveGrowStatus::Ok) return s;
    if(controls.randomLo>controls.randomHi) std::swap(controls.randomLo,controls.randomHi);
    size_t total=0;
    s=validate(input,owner,controls,&total);
    if(s!=CurveGrowStatus::Ok) return s;
    int d=-1;
    if(cudaGetDevice(&d)!=cudaSuccess) return CurveGrowStatus::CudaError;
    if(deviceIndex_<0) deviceIndex_=d;
    constexpr auto active=UsdGenExecutionResourceKind::Active;
    cudaError_t e=pending_.points.reset(total,reservation,active);
    if(e==cudaSuccess)e=pending_.restPoints.reset(total,reservation,active);
    if(e==cudaSuccess)e=pending_.widths.reset(total,reservation,active);
    if(e==cudaSuccess)e=pending_.hairT.reset(total,reservation,active);
    if(e==cudaSuccess)e=pending_.offsets.reset(input.geometry.curveCount+1,reservation,active);
    if(e==cudaSuccess)e=pending_.stableIds.reset(input.geometry.curveCount,reservation,active);
    if(e==cudaSuccess)e=pending_.rootPrim.reset(input.geometry.curveCount,reservation,active);
    if(e==cudaSuccess)e=pending_.rootUV.reset(input.geometry.curveCount,reservation,active);
    if(e==cudaSuccess)e=pending_.rootT.reset(input.geometry.curveCount,reservation,active);
    if(e==cudaSuccess)e=pending_.rootB.reset(input.geometry.curveCount,reservation,active);
    if(e==cudaSuccess)e=pending_.rootN.reset(input.geometry.curveCount,reservation,active);
    if(e==cudaSuccess)e=error_.reset(1,reservation,UsdGenExecutionResourceKind::Scratch);
    if(e!=cudaSuccess) { discardPending(); return Status(e); }
    if(!hostError_) {
        auto permit=TryReserveCudaExecutionBytes(sizeof(int),UsdGenExecutionResourceKind::Scratch,reservation);
        if(!permit||cudaHostAlloc(reinterpret_cast<void**>(&hostError_),sizeof(int),cudaHostAllocDefault)!=cudaSuccess) {
            discardPending(); return CurveGrowStatus::CudaError;
        }
        hostErrorPermit_=std::move(*permit);
    }
    try { inputQuarantineOwner_=std::make_unique<std::shared_ptr<const void>>(owner); }
    catch (...) { discardPending(); return CurveGrowStatus::CudaError; }
    inputOwner_=std::move(owner); input_=input; pendingCurves_=input.geometry.curveCount;
    pendingPoints_=total; pendingWork_=true; producerStream_=stream; unprovenWork_=true;
    *hostError_=kPendingStatus;
    if(cudaMemsetAsync(error_.data(),0,sizeof(int),stream)!=cudaSuccess)return CurveGrowStatus::CudaError;
    if (input_.frameStableIds.size) {
        ValidateFrameStableIds<<<uint32_t((input_.frameStableIds.size + 127u) / 128u),128,0,stream>>>(
            input_.frameStableIds.data, input_.frameStableIds.size, error_.data());
        if(cudaGetLastError()!=cudaSuccess) return CurveGrowStatus::CudaError;
    }
    if(!pendingCurves_) {
        ValidateEmptyOffset<<<1,1,0,stream>>>(input_.geometry.curveOffsets.data,
            pending_.offsets.data(),error_.data());
    } else {
        GrowKernel<<<(unsigned(pendingCurves_)+127)/128,128,0,stream>>>(input_,
            uint32_t(pendingCurves_),controls.cvCount,controls,pending_.points.data(),
            pending_.restPoints.data(),pending_.widths.data(),pending_.hairT.data(),
            pending_.offsets.data(),pending_.stableIds.data(),pending_.rootPrim.data(),
            pending_.rootUV.data(),pending_.rootT.data(),pending_.rootB.data(),
            pending_.rootN.data(),error_.data());
    }
    if(cudaGetLastError()!=cudaSuccess) return CurveGrowStatus::CudaError;
    return CurveGrowStatus::Ok;
}
CurveGrowStatus CudaCurveGrow::FinishFreshAsync(cudaStream_t stream,
    void(*callback)(cudaStream_t,cudaError_t,void*) noexcept,void* userdata) {
    if(!pendingWork_||finishScheduled_||!callback||stream!=producerStream_)
        return CurveGrowStatus::NoPendingUpdate;
    auto s=validateStream(stream);
    if(s!=CurveGrowStatus::Ok) return s;
    if(!ready_&&cudaEventCreateWithFlags(&ready_,cudaEventDisableTiming)!=cudaSuccess)
        return CurveGrowStatus::CudaError;
    if(pending_.recordUse(stream)!=CurveGrowStatus::Ok ||
       Status(error_.recordUse(stream))!=CurveGrowStatus::Ok ||
       cudaEventRecord(ready_,stream)!=cudaSuccess ||
       cudaMemcpyAsync(hostError_,error_.data(),sizeof(int),cudaMemcpyDeviceToHost,stream)!=cudaSuccess)
        return CurveGrowStatus::CudaError;
    auto e=cudaStreamAddCallback(stream,callback,userdata,0);
    if(e==cudaSuccess) finishScheduled_=true;
    return Status(e);
}
CurveGrowStatus CudaCurveGrow::CommitFreshFinish() {
    if(!pendingWork_||!finishScheduled_) return CurveGrowStatus::NoPendingUpdate;
    if(!hostError_) return CurveGrowStatus::CudaError;
    if(*hostError_==kPendingStatus) return CurveGrowStatus::NoPendingUpdate;
    if(*hostError_!=0) {
        CurveGrowStatus const status=*hostError_==kNonFinite
            ? CurveGrowStatus::NonFiniteInput : CurveGrowStatus::InvalidTopology;
        unprovenWork_=false;
        discardPending();
        return status;
    }
    active_=std::move(pending_); curves_=pendingCurves_; points_=pendingPoints_;
    input_={}; inputOwner_.reset(); inputQuarantineOwner_.reset();
    pendingCurves_=pendingPoints_=0; pendingWork_=finishScheduled_=false;
    unprovenWork_=false; ++generation_;
    return CurveGrowStatus::Ok;
}
DeviceCurveGeometryView CudaCurveGrow::view() const noexcept {
    return {active_.points.view(),active_.restPoints.view(),active_.widths.view(),
            active_.offsets.view(),active_.stableIds.view(),curves_,points_};
}
DeviceView<const float> CudaCurveGrow::hairT() const noexcept { return active_.hairT.view(); }
DeviceView<const int32_t> CudaCurveGrow::rootPrim() const noexcept { return active_.rootPrim.view(); }
DeviceView<const float2> CudaCurveGrow::rootUV() const noexcept { return active_.rootUV.view(); }
DeviceView<const float3> CudaCurveGrow::rootT() const noexcept { return active_.rootT.view(); }
DeviceView<const float3> CudaCurveGrow::rootB() const noexcept { return active_.rootB.view(); }
DeviceView<const float3> CudaCurveGrow::rootN() const noexcept { return active_.rootN.view(); }
CurveGrowStatus CudaCurveGrow::recordUse(cudaStream_t s) {
    auto x=validateStream(s);
    return x==CurveGrowStatus::Ok ? active_.recordUse(s) : x;
}
CurveGrowStatus CudaCurveGrow::waitOn(cudaStream_t s) const {
    auto x=validateStream(s);
    return x==CurveGrowStatus::Ok ? active_.waitOn(s) : x;
}
void CudaCurveGrow::ReclassifyPublishedGeneration() noexcept {
    active_.reclassify(UsdGenExecutionResourceKind::Pinned);
    error_.Reclassify(UsdGenExecutionResourceKind::Pinned);
    hostErrorPermit_.Reclassify(UsdGenExecutionResourceKind::Pinned);
}
size_t CudaCurveGrow::ExclusiveRetainedBytes() const noexcept {
    return active_.bytes()+error_.bytes()+hostErrorPermit_.Bytes();
}
} // namespace usdGen::gpu
