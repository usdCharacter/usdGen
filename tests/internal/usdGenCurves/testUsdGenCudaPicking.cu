#include "usdGenCurves/cuda/picking.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); return 1; } } while (false)

template <class T>
static bool Upload(DeviceBuffer<T>& d, std::vector<T> const& v, cudaStream_t s) {
    return cudaMemcpyAsync(d.data(), v.data(), v.size()*sizeof(T),
                           cudaMemcpyHostToDevice, s) == cudaSuccess;
}
template <class T>
static bool Download(DeviceView<const T> v, std::vector<T>& out, cudaStream_t s) {
    if (!out.empty() && cudaMemcpyAsync(out.data(), v.data,
                                        out.size()*sizeof(T),
                                        cudaMemcpyDeviceToHost, s) != cudaSuccess)
        return false;
    return cudaStreamSynchronize(s) == cudaSuccess;
}
static PickQuery IdentityQuery(float x, float y, float radius) {
    PickQuery q{};
    q.viewProj[0] = q.viewProj[5] = q.viewProj[10] = q.viewProj[15] = 1.0f;
    q.width = q.height = 100; q.x = x; q.y = y; q.radiusPx = radius;
    return q;
}

int main() {
    cudaStream_t producer = nullptr, consumer = nullptr;
    CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) == cudaSuccess);
    DeviceBuffer<float3> points;
    DeviceBuffer<uint32_t> offsets;
    DeviceBuffer<uint64_t> ids;
    CHECK(points.reset(5) == cudaSuccess && offsets.reset(3) == cudaSuccess &&
          ids.reset(2) == cudaSuccess);
    const std::vector<float3> source{{0,0,0}, {.2f,0,0},
                                     {-.5f,0,0}, {.5f,0,0}, {0,0,0}};
    CHECK(Upload(points, source, producer) && Upload(offsets, {0u,2u,5u}, producer) &&
          Upload(ids, {100u,200u}, producer) && cudaStreamSynchronize(producer) == cudaSuccess);
    DeviceCurveGeometryView geometry{{points.data(),5}, {}, {},
                                     {offsets.data(),3}, {ids.data(),2}, 2, 5};
    CudaPicking picker;

    // Row-vector identity projection, top-left pixel conversion, exact tie
    // selection by lowest flat index, and cross-stream Finish.
    CHECK(picker.ApplyPick(geometry, IdentityQuery(50,50,1), producer) == PickingStatus::Ok);
    CHECK(picker.Finish(consumer) == PickingStatus::Ok);
    PickResult result = picker.result();
    CHECK(result.hit && result.flatIndex == 0 && result.curve == 0 && result.cv == 0 &&
          result.stableId == 100 && std::fabs(result.distancePx) < 1e-6f);

    // Non-identity row-vector translation and a perspective-style varying w.
    PickQuery translated = IdentityQuery(60,50,1);
    translated.viewProj[12] = .2f; // clip.x = x + 0.2 for row vectors
    CHECK(picker.ApplyPick(geometry, translated, producer) == PickingStatus::Ok &&
          picker.Finish(consumer) == PickingStatus::Ok && picker.result().hit &&
          picker.result().flatIndex == 0);
    DeviceBuffer<float3> perspectivePoints;
    DeviceBuffer<uint32_t> perspectiveOffsets;
    CHECK(perspectivePoints.reset(2) == cudaSuccess && perspectiveOffsets.reset(2) == cudaSuccess);
    CHECK(Upload(perspectivePoints, {make_float3(0,0,0), make_float3(.5f,0,2)}, producer) &&
          Upload(perspectiveOffsets, {0u,2u}, producer) && cudaStreamSynchronize(producer) == cudaSuccess);
    DeviceCurveGeometryView perspective{{perspectivePoints.data(),2}, {}, {},
                                        {perspectiveOffsets.data(),2}, {}, 1, 2};
    PickQuery perspectiveQuery = IdentityQuery(50,50,1);
    perspectiveQuery.viewProj[11] = .5f; // row-vector clip.w = 1 + .5*z
    CHECK(picker.ApplyPick(perspective, perspectiveQuery, producer) == PickingStatus::Ok &&
          picker.Finish(consumer) == PickingStatus::Ok && picker.result().hit &&
          picker.result().flatIndex == 0);
    CHECK(picker.ApplyPick(perspective, IdentityQuery(75,50,1), producer) == PickingStatus::Ok &&
          picker.Finish(consumer) == PickingStatus::Ok && !picker.result().hit);

    // Footprint output is device-resident and sorted by flat index.
    CHECK(picker.ApplyFootprint(geometry, IdentityQuery(50,50,11), producer) == PickingStatus::Ok);
    CHECK(picker.Finish(consumer) == PickingStatus::Ok && picker.footprintCount() == 3);
    std::vector<int32_t> indices(3);
    CHECK(Download(picker.footprint(), indices, consumer));
    CHECK(indices == std::vector<int32_t>({0,1,4}));

    // A large all-hit footprint exercises uint32 flags/prefixes (a byte flag
    // would wrap at 256) while preserving exact ascending flat-index order.
    DeviceBuffer<float3> manyPoints;
    DeviceBuffer<uint32_t> manyOffsets;
    CHECK(manyPoints.reset(1025) == cudaSuccess && manyOffsets.reset(2) == cudaSuccess);
    std::vector<float3> allHit(1025, make_float3(0,0,0));
    CHECK(Upload(manyPoints, allHit, producer) && Upload(manyOffsets, {0u,1025u}, producer) &&
          cudaStreamSynchronize(producer) == cudaSuccess);
    DeviceCurveGeometryView many{{manyPoints.data(),1025}, {}, {},
                                 {manyOffsets.data(),2}, {}, 1, 1025};
    CHECK(picker.ApplyFootprint(many, IdentityQuery(50,50,1), producer) == PickingStatus::Ok &&
          picker.Finish(consumer) == PickingStatus::Ok && picker.footprintCount() == 1025);
    std::vector<int32_t> manyIndices(1025);
    CHECK(Download(picker.footprint(), manyIndices, consumer));
    for (int32_t i = 0; i < 1025; ++i) CHECK(manyIndices[size_t(i)] == i);

    // Behind-camera and outside-depth points are ignored, while a finite
    // matrix/query is accepted.
    PickQuery clipped = IdentityQuery(50,50,100);
    clipped.viewProj[15] = -1.0f;
    CHECK(picker.ApplyPick(geometry, clipped, producer) == PickingStatus::Ok &&
          picker.Finish(consumer) == PickingStatus::Ok && !picker.result().hit);
    PickQuery badQuery = IdentityQuery(0,0,1);
    badQuery.viewProj[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(picker.ApplyPick(geometry, badQuery, producer) == PickingStatus::NonFiniteInput &&
          !picker.pending());

    // Malformed device offsets fail before any offset-derived point access and
    // preserve the prior successful footprint.
    CHECK(Upload(offsets, {0u,2u,99u}, producer) && cudaStreamSynchronize(producer) == cudaSuccess);
    CHECK(picker.ApplyFootprint(geometry, IdentityQuery(50,50,11), producer) == PickingStatus::Ok &&
          picker.Finish(consumer) == PickingStatus::InvalidArgument &&
          picker.footprintCount() == 1025);
    manyIndices.resize(1025);
    CHECK(Download(picker.footprint(), manyIndices, consumer));
    for (int32_t i = 0; i < 1025; ++i) CHECK(manyIndices[size_t(i)] == i);
    CHECK(Upload(offsets, {0u,2u,5u}, producer));

    // Nonfinite source geometry is diagnosed on device and does not publish a
    // new scalar pick result.
    std::vector<float3> badSource = source; badSource[1].x = NAN;
    CHECK(Upload(points, badSource, producer) && cudaStreamSynchronize(producer) == cudaSuccess);
    CHECK(picker.ApplyPick(geometry, IdentityQuery(50,50,1), producer) == PickingStatus::Ok &&
          picker.Finish(consumer) == PickingStatus::NonFiniteInput &&
          !picker.result().hit);
    CHECK(Upload(points, source, producer));

    // Empty geometry is valid, with no footprint allocation or host geometry
    // readback. The offsets sentinel is still required.
    DeviceBuffer<uint32_t> emptyOffsets;
    CHECK(emptyOffsets.reset(1) == cudaSuccess && Upload(emptyOffsets, {0u}, producer) &&
          cudaStreamSynchronize(producer) == cudaSuccess);
    DeviceCurveGeometryView empty{{}, {}, {}, {emptyOffsets.data(),1}, {}, 0, 0};
    CudaPicking emptyPicker;
    CHECK(emptyPicker.ApplyPick(empty, IdentityQuery(1,1,2), producer) == PickingStatus::Ok &&
          emptyPicker.Finish(consumer) == PickingStatus::Ok && !emptyPicker.result().hit);
    CHECK(emptyPicker.ApplyFootprint(empty, IdentityQuery(1,1,2), producer) == PickingStatus::Ok &&
          emptyPicker.Finish(consumer) == PickingStatus::Ok && emptyPicker.footprintCount() == 0);

    CHECK(cudaStreamDestroy(producer) == cudaSuccess && cudaStreamDestroy(consumer) == cudaSuccess);
    std::puts("testUsdGenCudaPicking: PASS");
    return 0;
}
