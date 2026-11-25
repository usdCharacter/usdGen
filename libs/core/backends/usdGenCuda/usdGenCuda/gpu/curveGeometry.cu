#include "curveGeometry.h"
#include <limits>
namespace usdGen { namespace gpu {
cudaError_t CudaCurveGeometry::Set(DeviceCurveGeometryView s,cudaStream_t stream) {
 if(!s.points.data||!s.restPoints.data||!s.widths.data||!s.curveOffsets.data||!s.stableIds.data||!s.curveCount||!s.pointCount||s.curveCount==std::numeric_limits<size_t>::max()||s.points.size!=s.pointCount||s.restPoints.size!=s.pointCount||s.widths.size!=s.pointCount||s.curveOffsets.size!=s.curveCount+1||s.stableIds.size!=s.curveCount)return cudaErrorInvalidValue;
 cudaError_t e;if((e=waitOn(stream))!=cudaSuccess)return e;
 DeviceBuffer<float3> np,nr;DeviceBuffer<float> nw;DeviceBuffer<uint32_t> no;DeviceBuffer<uint64_t> ni;
 if((e=np.reset(s.pointCount))||(e=nr.reset(s.pointCount))||(e=nw.reset(s.pointCount))||(e=no.reset(s.curveCount+1))||(e=ni.reset(s.curveCount)))return e;
 if((e=cudaMemcpyAsync(np.data(),s.points.data,s.pointCount*sizeof(float3),cudaMemcpyDeviceToDevice,stream))||(e=cudaMemcpyAsync(nr.data(),s.restPoints.data,s.pointCount*sizeof(float3),cudaMemcpyDeviceToDevice,stream))||(e=cudaMemcpyAsync(nw.data(),s.widths.data,s.pointCount*sizeof(float),cudaMemcpyDeviceToDevice,stream))||(e=cudaMemcpyAsync(no.data(),s.curveOffsets.data,(s.curveCount+1)*sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream))||(e=cudaMemcpyAsync(ni.data(),s.stableIds.data,s.curveCount*sizeof(uint64_t),cudaMemcpyDeviceToDevice,stream)))return e;
 if((e=np.recordUse(stream))||(e=nr.recordUse(stream))||(e=nw.recordUse(stream))||(e=no.recordUse(stream))||(e=ni.recordUse(stream)))return e;
 points_=std::move(np);rest_=std::move(nr);widths_=std::move(nw);offsets_=std::move(no);ids_=std::move(ni);curves_=s.curveCount;pointsCount_=s.pointCount;++generation_;return cudaSuccess;
}
DeviceCurveGeometryView CudaCurveGeometry::view()const{return {points_.view(),rest_.view(),widths_.view(),offsets_.view(),ids_.view(),curves_,pointsCount_};}
cudaError_t CudaCurveGeometry::recordUse(cudaStream_t s){cudaError_t e;if((e=points_.recordUse(s))||(e=rest_.recordUse(s))||(e=widths_.recordUse(s))||(e=offsets_.recordUse(s))||(e=ids_.recordUse(s)))return e;return cudaSuccess;}
cudaError_t CudaCurveGeometry::waitOn(cudaStream_t s)const{cudaError_t e;if((e=points_.waitOn(s))||(e=rest_.waitOn(s))||(e=widths_.waitOn(s))||(e=offsets_.waitOn(s))||(e=ids_.waitOn(s)))return e;return cudaSuccess;}
}}
