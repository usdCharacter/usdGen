#include "curveIndices.h"
#include "cudaCompat.h"

#include <cub/device/device_scan.cuh>

#include <climits>
#include <limits>

namespace usdGen::gpu {
namespace {

constexpr int kThreads = 128;

__host__ __device__ bool Valid(CurveIndexOptions o) {
    return o.basis >= CurveIndexBasis::Linear && o.basis <= CurveIndexBasis::CentripetalCatmullRom &&
        o.wrap >= CurveIndexWrap::Nonperiodic && o.wrap <= CurveIndexWrap::Segmented &&
        o.mode >= CurveIndexMode::Curves && o.mode <= CurveIndexMode::Points;
}
__host__ __device__ bool Cubic(CurveIndexOptions o) { return o.basis != CurveIndexBasis::Linear && o.mode == CurveIndexMode::Curves; }
__host__ __device__ bool Cat(CurveIndexBasis b) { return b == CurveIndexBasis::CatmullRom || b == CurveIndexBasis::CentripetalCatmullRom; }
__device__ int ClampIndex(uint64_t i, uint32_t count) {
    return static_cast<int>(i < count ? i : count - 1);
}

__device__ uint64_t Count(CurveIndexOptions o, uint32_t n) {
    if (o.mode == CurveIndexMode::Points) return n;
    if (!Cubic(o)) {
        if (n < 2) return o.wrap == CurveIndexWrap::Periodic ? 1 : 0;
        if (o.wrap == CurveIndexWrap::Segmented) return n / 2;
        bool skip = o.mode == CurveIndexMode::Hull && Cat(o.basis) && o.wrap != CurveIndexWrap::Pinned;
        if (skip) {
            if (n <= 3) return o.wrap == CurveIndexWrap::Periodic ? 1 : 0;
            return o.wrap == CurveIndexWrap::Periodic ? n - 2 : n - 3;
        }
        return o.wrap == CurveIndexWrap::Periodic ? n : n - 1;
    }
    if (n < 2) return 0;
    uint64_t step = o.basis == CurveIndexBasis::Bezier ? 3 : 1;
    uint64_t core = o.wrap == CurveIndexWrap::Periodic ? max(uint64_t(n) / step, uint64_t(1))
        : (max(int(n) - 4, 0) / step) + 1;
    if (o.wrap == CurveIndexWrap::Pinned && step == 1)
        core += o.basis == CurveIndexBasis::BSpline ? 4 : 2;
    return core;
}

__global__ void CountKernel(CurveIndexOptions o, size_t curves,
    uint32_t pointBase, uint32_t pointEnd, const uint32_t* offsets,
    uint64_t* counts, uint32_t* status) {
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= curves) return;
    uint32_t a = offsets[c], b = offsets[c + 1];
    if ((c == 0 && a != pointBase) || a < pointBase || a >= pointEnd ||
        b <= a || b > pointEnd ||
        (c + 1 == curves && b != pointEnd)) {
        atomicCAS(status, 0u, 1u); counts[c] = 0; return;
    }
    uint32_t n = b - a;
    if (o.wrap == CurveIndexWrap::Segmented && o.basis == CurveIndexBasis::Linear && (n & 1)) {
        atomicCAS(status, 0u, 2u); counts[c] = 0; return;
    }
    counts[c] = Count(o, n);
}
__global__ void TerminalKernel(uint64_t* counts, size_t curves,
    uint32_t pointBase, uint32_t pointEnd, const uint32_t* offsets,
    uint32_t* status) {
    if (threadIdx.x || blockIdx.x) return;
    counts[curves] = 0;
    if (curves == 0 && (offsets[0] != pointBase || pointEnd != pointBase))
        atomicCAS(status, 0u, 1u);
}
__global__ void InvalidSpanKernel(uint64_t* total, uint32_t* status) {
    if (threadIdx.x || blockIdx.x) return;
    *status = 1;
    *total = 0;
}
__global__ void TotalKernel(size_t curves, const uint64_t* offsets, uint64_t* total,
    uint32_t* status, size_t maxRecords) {
    if (threadIdx.x || blockIdx.x) return;
    if (*status) { *total = 0; return; }
    uint64_t n = offsets[curves];
    if (n > maxRecords || n > INT_MAX) { *status = 1; *total = 0; return; }
    *total = n;
}

__device__ void Put(int32_t* dst, uint64_t r, int arity, int a, int b, int c, int d) {
    int32_t* p = dst + r * arity;
    p[0] = a; if (arity > 1) p[1] = b; if (arity > 2) { p[2] = c; p[3] = d; }
}
__global__ void EmitKernel(CurveIndexOptions o, size_t curves, uint32_t pointBase,
    const uint32_t* cv,
    const uint64_t* recordOffsets, const uint32_t* status, int32_t* indices, int32_t* prim) {
    size_t curve = blockIdx.x;
    if (curve >= curves || *status) return;
    uint32_t first = cv[curve] - pointBase, n = cv[curve + 1] - cv[curve];
    uint64_t out = recordOffsets[curve];
    if (o.mode == CurveIndexMode::Points) {
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) { Put(indices, out+i, 1, first+i,0,0,0); prim[out+i]=curve; }
        return;
    }
    if (!Cubic(o)) {
        if (n < 2) {
            if (o.wrap == CurveIndexWrap::Periodic && threadIdx.x == 0) {
                Put(indices, out, 2, first, first, 0, 0); prim[out] = curve;
            }
            return;
        }
        bool segmented = o.wrap == CurveIndexWrap::Segmented;
        bool skip = o.mode == CurveIndexMode::Hull && Cat(o.basis) && o.wrap != CurveIndexWrap::Pinned;
        uint64_t records = Count(o,n);
        for (uint64_t i=threadIdx.x;i<records;i+=blockDim.x) {
            uint32_t x, y;
            if (segmented) { x=2*i; y=x+1; }
            else if (skip) {
                if (o.wrap == CurveIndexWrap::Periodic && i + 1 == records) { x=n-1; y=0; }
                else { x=i+1; y=x+1; }
            }
            else { x=i; y=(o.wrap==CurveIndexWrap::Periodic) ? ((i+1)%n) : (i+1); }
            Put(indices,out+i,2,first+x,first+y,0,0); prim[out+i]=curve;
        }
        return;
    }
    if (n < 2) return;
    bool periodic=o.wrap==CurveIndexWrap::Periodic;
    int step=o.basis==CurveIndexBasis::Bezier ? 3 : 1;
    bool pinned=o.wrap==CurveIndexWrap::Pinned && step==1;
    uint64_t core=periodic ? max(uint64_t(n)/step,uint64_t(1)) : (max(int(n)-4,0)/step)+1;
    uint64_t pre=pinned ? (o.basis==CurveIndexBasis::BSpline ? 2 : 1) : 0;
    uint64_t post=pre;
    for(uint64_t r=threadIdx.x;r<pre+core+post;r+=blockDim.x) {
        int a,b,c,d;
        if(r<pre) {
            int v1=min(1u,n-1), v2=min(2u,n-1);
            if(o.basis==CurveIndexBasis::BSpline && r==0) a=0,b=0,c=0,d=v1;
            else a=0,b=0,c=v1,d=v2;
        } else if(r<pre+core) {
            uint64_t i=r-pre, base=i*step;
            a=periodic ? base%n : ClampIndex(base,n);
            b=periodic ? (base+1)%n : ClampIndex(base+1,n);
            c=periodic ? (base+2)%n : ClampIndex(base+2,n);
            d=periodic ? (base+3)%n : ClampIndex(base+3,n);
        } else {
            uint64_t base=(core-1)*step;
            int q1=ClampIndex(base+1,n);
            int q2=ClampIndex(base+2,n);
            int q3=ClampIndex(base+3,n);
            if(o.basis==CurveIndexBasis::BSpline && r==pre+core+1) a=q2,b=q3,c=q3,d=q3;
            else a=q1,b=q2,c=q3,d=q3;
        }
        Put(indices,out+r,4,first+a,first+b,first+c,first+d); prim[out+r]=curve;
    }
}

__global__ void PackDrawCountKernel(const uint64_t* recordCount,
    const uint32_t* status, uint32_t arity, uint64_t maxRecords,
    uint32_t* drawCount) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t records = *recordCount;
    if (*status != 0 || records > maxRecords ||
        records > uint64_t(UINT32_MAX) / uint64_t(arity)) {
        *drawCount = 0;
        return;
    }
    *drawCount = static_cast<uint32_t>(records * uint64_t(arity));
}

bool Same(CurveIndexRequirements const&a, CurveIndexRequirements const&b) {
    return a.maxRecords==b.maxRecords && a.indexArity==b.indexArity &&
        a.scanBytes==b.scanBytes && a.deviceIndex==b.deviceIndex && a.stream==b.stream;
}

cudaError_t Requirements(CurveIndexOptions o, size_t curves, size_t points,
    CurveIndexRequirements* out, cudaStream_t stream, bool prepare) {
    if (!out || !Valid(o) || (o.wrap == CurveIndexWrap::Segmented && o.basis != CurveIndexBasis::Linear) ||
        (!curves && points) || points < curves || curves >= size_t(INT_MAX) ||
        points > size_t(INT_MAX)) return cudaErrorInvalidValue;
    size_t max = points;
    if (Cubic(o) && o.wrap==CurveIndexWrap::Pinned && o.basis!=CurveIndexBasis::Bezier) {
        const size_t extra = o.basis==CurveIndexBasis::BSpline ? 3 : 1;
        if (curves > (size_t(INT_MAX)-points)/extra) return cudaErrorInvalidValue;
        max=points+extra*curves;
    }
    if (max > size_t(INT_MAX)) return cudaErrorInvalidValue;
    int device = -1;
    cudaError_t e = cudaGetDevice(&device);
    if (e != cudaSuccess) return e;
    if (prepare && stream) {
        int streamDevice = -1;
        e = cudaStreamGetDevice(stream, &streamDevice);
        if (e != cudaSuccess) return e;
        if (streamDevice != device) return cudaErrorInvalidResourceHandle;
    }
    size_t bytes=0;
    e=cub::DeviceScan::ExclusiveSum(nullptr,bytes,(uint64_t*)nullptr,(uint64_t*)nullptr,curves+1,stream);
    if(e!=cudaSuccess) return e;
    *out={max, o.mode==CurveIndexMode::Points?1u:(Cubic(o)?4u:2u), bytes, device, stream};
    return cudaSuccess;
}
} // namespace

cudaError_t GetCurveIndexRequirements(CurveIndexOptions o, size_t curves,
    size_t points, CurveIndexRequirements* out, cudaStream_t stream) {
    return Requirements(o,curves,points,out,stream,true);
}

cudaError_t BuildCurveIndices(CurveIndexOptions o, CurveIndexSpan span,
 CurveIndexRequirements const& req,CurveIndexWorkspace ws,CurveIndexOutput out,cudaStream_t stream) {
    size_t const curves = span.curveCount;
    size_t const points = span.pointCount;
    DeviceView<const uint32_t> const offsets = span.curveOffsets;
    CurveIndexRequirements expected;
    // cudaStreamGetDevice rejects an actively captured stream. Validate it
    // during preparation, then bind execution to that live stream and device.
    cudaError_t e = Requirements(o,curves,points,&expected,stream,false);
    if (e != cudaSuccess) return e;
    if(!Same(req,expected)||
      offsets.size!=curves+1||ws.recordCounts.size!=curves+1||ws.recordOffsets.size!=curves+1||ws.scan.size<req.scanBytes||
      !offsets.data || !ws.recordCounts.data || !ws.recordOffsets.data || (req.scanBytes && !ws.scan.data) ||
      (req.maxRecords && (!out.indices.data || !out.primitiveParam.data)) || !out.recordCount.data || !out.status.data ||
      out.indices.size<req.maxRecords*req.indexArity||out.primitiveParam.size<req.maxRecords||out.recordCount.size<1||out.status.size<1) return cudaErrorInvalidValue;
    if((e=cudaMemsetAsync(out.status.data,0,sizeof(uint32_t),stream))!=cudaSuccess|| (e=cudaMemsetAsync(out.recordCount.data,0,sizeof(uint64_t),stream))!=cudaSuccess) return e;
    if (points > size_t(UINT32_MAX) - span.pointBase) {
        InvalidSpanKernel<<<1,1,0,stream>>>(out.recordCount.data,out.status.data);
        return cudaGetLastError();
    }
    uint32_t const pointEnd = span.pointBase + static_cast<uint32_t>(points);
    if (curves) {
        dim3 grid((curves+kThreads-1)/kThreads);
        CountKernel<<<grid,kThreads,0,stream>>>(o,curves,span.pointBase,pointEnd,
            offsets.data,ws.recordCounts.data,out.status.data);
        if ((e=cudaGetLastError())!=cudaSuccess) return e;
    }
    TerminalKernel<<<1,1,0,stream>>>(ws.recordCounts.data,curves,span.pointBase,
        pointEnd,offsets.data,out.status.data);
    if ((e=cudaGetLastError())!=cudaSuccess) return e;
    size_t scanBytes = req.scanBytes;
    if((e=cub::DeviceScan::ExclusiveSum(ws.scan.data,scanBytes,ws.recordCounts.data,ws.recordOffsets.data,curves+1,stream))!=cudaSuccess) return e;
    TotalKernel<<<1,1,0,stream>>>(curves,ws.recordOffsets.data,out.recordCount.data,out.status.data,req.maxRecords);
    if ((e=cudaGetLastError())!=cudaSuccess) return e;
    if(curves) EmitKernel<<<curves,kThreads,0,stream>>>(o,curves,span.pointBase,
        offsets.data,ws.recordOffsets.data,out.status.data,out.indices.data,
        out.primitiveParam.data);
    return cudaGetLastError();
}

cudaError_t BuildCurveIndices(CurveIndexOptions o, size_t curves, size_t points,
    DeviceView<const uint32_t> offsets, CurveIndexRequirements const& req,
    CurveIndexWorkspace ws, CurveIndexOutput out, cudaStream_t stream) {
    return BuildCurveIndices(o, CurveIndexSpan{offsets, curves, points, 0},
        req, ws, out, stream);
}

cudaError_t PackCurveDrawCount(DeviceView<const uint64_t> recordCount,
    DeviceView<const uint32_t> status, uint32_t indexArity,
    size_t maxRecords, DeviceView<uint32_t> drawCount,
    cudaStream_t stream) {
    if ((indexArity != 1 && indexArity != 2 && indexArity != 4) ||
        recordCount.size != 1 || status.size != 1 || drawCount.size != 1 ||
        !recordCount.data || !status.data || !drawCount.data) {
        return cudaErrorInvalidValue;
    }
    PackDrawCountKernel<<<1, 1, 0, stream>>>(recordCount.data, status.data,
        indexArity, static_cast<uint64_t>(maxRecords), drawCount.data);
    return cudaGetLastError();
}
} // namespace usdGen::gpu
