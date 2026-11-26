#include "usdGenCurves/cuda/curveCompaction.h"
#include "usdGenCuda/backendId.h"
#include "usdGen/executionResources.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(X) do { if (!(X)) { \
    std::fprintf(stderr, "check failed at %d: %s\n", __LINE__, #X); return 1; \
} } while (0)

static float3 P(float x, float y, float z) { return make_float3(x, y, z); }
static bool Near(float3 a, float3 b) {
    return std::fabs(a.x-b.x) < 1.e-5f && std::fabs(a.y-b.y) < 1.e-5f &&
           std::fabs(a.z-b.z) < 1.e-5f;
}
struct AsyncStatus { std::atomic<int> calls{0}; std::atomic<cudaError_t> status{cudaErrorUnknown}; };
static void CUDART_CB Done(cudaStream_t, cudaError_t status, void* data) noexcept {
    auto* result = static_cast<AsyncStatus*>(data);
    result->status.store(status, std::memory_order_release);
    result->calls.fetch_add(1, std::memory_order_release);
}

template<class T>
static bool Upload(DeviceBuffer<T>& buffer, std::vector<T> const& values) {
    return buffer.reset(values.size()) == cudaSuccess &&
        (values.empty() || cudaMemcpy(buffer.data(), values.data(),
                                      values.size() * sizeof(T),
                                      cudaMemcpyHostToDevice) == cudaSuccess);
}

template<class T>
static std::vector<T> Read(DeviceView<const T> view, cudaStream_t stream) {
    std::vector<T> result(view.size);
    if (view.size && (cudaMemcpyAsync(result.data(), view.data,
                                      view.size * sizeof(T),
                                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                      cudaStreamSynchronize(stream) != cudaSuccess))
        result.clear();
    return result;
}

static bool Same3(std::vector<float3> const& actual, std::vector<float3> const& expected) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i != actual.size(); ++i)
        if (!Near(actual[i], expected[i])) return false;
    return true;
}
static bool Same2(std::vector<float2> const& actual, std::vector<float2> const& expected) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i != actual.size(); ++i)
        if (std::fabs(actual[i].x - expected[i].x) >= 1.e-5f ||
            std::fabs(actual[i].y - expected[i].y) >= 1.e-5f) return false;
    return true;
}

int main() {
    cudaStream_t producer = nullptr, consumer = nullptr;
    CHECK(cudaStreamCreate(&producer) == cudaSuccess);
    CHECK(cudaStreamCreate(&consumer) == cudaSuccess);
    {
        const std::vector<float3> points = {
            P(0,0,0), P(1,0,0), P(0,1,0), P(0,0,1), P(1,1,1)};
        const std::vector<float3> rest = {
            P(10,0,0), P(11,0,0), P(10,1,0), P(10,0,1), P(11,1,1)};
        const std::vector<float> widths = {1,2,3,4,5};
        const std::vector<float> hairT = {0,1,0,.5f,1};
        const std::vector<uint32_t> offsets = {0,2,5};
        const std::vector<uint64_t> ids = {30,70};
        const std::vector<int32_t> roots = {4,8};
        const std::vector<float2> rootUV = {{.1f,.2f},{.7f,.8f}};
        DeviceBuffer<float3> dPoints, dRest;
        DeviceBuffer<float> dWidths, dHairT;
        DeviceBuffer<uint32_t> dOffsets;
        DeviceBuffer<uint64_t> dIds;
        DeviceBuffer<int32_t> dRoots;
        DeviceBuffer<unsigned char> dRootUV;
        DeviceBuffer<uint8_t> dKeep;
        CHECK(Upload(dPoints, points) && Upload(dRest, rest) &&
              Upload(dWidths, widths) && Upload(dHairT, hairT) &&
              Upload(dOffsets, offsets) && Upload(dIds, ids) &&
              Upload(dRoots, roots) &&
              dRootUV.reset(rootUV.size() * sizeof(float2)) == cudaSuccess &&
              cudaMemcpy(dRootUV.data(), rootUV.data(),
                         rootUV.size() * sizeof(float2), cudaMemcpyHostToDevice) == cudaSuccess &&
              Upload(dKeep, std::vector<uint8_t>{0,1}));
        DeviceCurveGeometryView input = {
            {dPoints.data(), points.size()}, {dRest.data(), rest.size()},
            {dWidths.data(), widths.size()}, {dOffsets.data(), offsets.size()},
            {dIds.data(), ids.size()}, 2, 5};

        // The production compactor uses the larger of the two CUB scans for
        // its one shared temporary buffer.  Keep the query tied to the same
        // two-curve Apply allocation ledger: all-cull makes the output only
        // the required one-element offset sentinel, so every reserved byte is
        // attributable to the query, fixed scratch, and that sentinel.
        CHECK(Upload(dKeep, std::vector<uint8_t>{0,0}));
        size_t scanBytes = 0;
        CHECK(GetCudaCurveCompactionScanTemporaryBytes(2, &scanBytes, producer) ==
              cudaSuccess && scanBytes > 0);
        size_t zeroScanBytes = 17;
        CHECK(GetCudaCurveCompactionScanTemporaryBytes(0, &zeroScanBytes, producer) ==
              cudaSuccess && zeroScanBytes == 0);
        int resourceDevice = -1;
        CHECK(cudaGetDevice(&resourceDevice) == cudaSuccess);
        auto resourcePool = FindUsdGenExecutionResourcePool(
            {usdGen::UsdGenCudaBackendId(), resourceDevice});
        CHECK(resourcePool);
        auto const scanBaseline = resourcePool->Snapshot();
        constexpr size_t fixedScratch = sizeof(int) + 2 * sizeof(uint32_t) +
            4 * 2 * sizeof(uint32_t);
        constexpr size_t emptyOutputOffsets = sizeof(uint32_t);
        auto scanReservation = resourcePool->TryReserveMemory(
            scanBytes + fixedScratch + emptyOutputOffsets);
        CHECK(scanReservation);
        CudaCurveCompaction queriedScan;
        CHECK(queriedScan.Apply(input, {dHairT.data(), hairT.size()},
                                {dRoots.data(), roots.size()},
                                {reinterpret_cast<float2 const*>(dRootUV.data()), 2},
                                {dKeep.data(), 2}, producer, &*scanReservation) ==
                  CurveCompactionStatus::Ok &&
              scanReservation->RemainingBytes() == 0);
        auto const scanCharged = resourcePool->Snapshot();
        CHECK(scanCharged.usedBytes == scanBaseline.usedBytes + scanBytes +
                  fixedScratch + emptyOutputOffsets);
        CHECK(queriedScan.Finish(consumer) == CurveCompactionStatus::Ok);
        auto const scanFinished = resourcePool->Snapshot();
        CHECK(scanFinished.usedBytes == scanCharged.usedBytes);

        CudaCurveCompaction compact;
        CHECK(Upload(dKeep, std::vector<uint8_t>{0,1}));
        CHECK(compact.Apply(input, {dHairT.data(), hairT.size()},
                            {dRoots.data(), roots.size()},
                            {reinterpret_cast<float2 const*>(dRootUV.data()), 2},
                            {dKeep.data(), 2}, producer) == CurveCompactionStatus::Ok);
        CHECK(compact.pending());
        CHECK(compact.Finish(consumer) == CurveCompactionStatus::Ok);
        CHECK(!compact.pending() && compact.generation() == 1);
        auto view = compact.view();
        CHECK(view.curveCount == 1 && view.pointCount == 3);
        auto outOffsets = Read(view.curveOffsets, consumer);
        auto outPoints = Read(view.points, consumer);
        auto outRest = Read(view.restPoints, consumer);
        auto outWidths = Read(view.widths, consumer);
        auto outHairT = Read(compact.hairT(), consumer);
        auto outIds = Read(view.stableIds, consumer);
        auto outRoots = Read(compact.rootPrim(), consumer);
        auto outUV = Read(compact.rootUV(), consumer);
        CHECK((outOffsets == std::vector<uint32_t>{0,3}));
        CHECK(outPoints.size() == 3 && Near(outPoints[0], points[2]) &&
              Near(outPoints[2], points[4]));
        CHECK(outRest.size() == 3 && Near(outRest[0], rest[2]) &&
              Near(outRest[2], rest[4]));
        CHECK(outWidths == std::vector<float>({3,4,5}));
        CHECK(outHairT == std::vector<float>({0,.5f,1}));
        CHECK((outIds == std::vector<uint64_t>{70}));
        CHECK((outRoots == std::vector<int32_t>{8}));
        CHECK(outUV.size() == 1 && std::fabs(outUV[0].x-.7f) < 1.e-5f);

        // Optional Grow frames follow surviving stable IDs, never the old
        // curve ordinal. Legacy and fresh publication must agree, including
        // all-cull and replacement of a previous framed output.
        {
            std::vector<float3> const tangents{P(1,0,0),P(0,1,0)};
            std::vector<float3> const binormals{P(0,1,0),P(0,0,1)};
            std::vector<float3> const normals{P(0,0,1),P(1,0,0)};
            DeviceBuffer<float3> dT, dB, dN;
            DeviceBuffer<uint64_t> dFrameIds;
            CHECK(Upload(dT,tangents)&&Upload(dB,binormals)&&
                  Upload(dN,normals)&&Upload(dFrameIds,ids));
            RestRootFrames frames{{dT.data(),2},{dB.data(),2},
                                  {dN.data(),2},{dFrameIds.data(),2}};
            CudaCurveCompaction framed;
            CHECK(framed.Apply(input,{dHairT.data(),hairT.size()},
                {dRoots.data(),2},{reinterpret_cast<float2 const*>(dRootUV.data()),2},
                {dKeep.data(),2},producer,nullptr,frames)==CurveCompactionStatus::Ok);
            CHECK(framed.Finish(consumer)==CurveCompactionStatus::Ok);
            CHECK(framed.ExclusiveRetainedBytes()==compact.ExclusiveRetainedBytes()+
                  3*sizeof(float3));
            CHECK(Same3(Read(framed.frames().tangent,consumer),{tangents[1]})&&
                  Same3(Read(framed.frames().binormal,consumer),{binormals[1]})&&
                  Same3(Read(framed.frames().normal,consumer),{normals[1]})&&
                  Read(framed.frames().stableIds,consumer)==std::vector<uint64_t>{70});
            CHECK(framed.recordUse(producer)==cudaSuccess&&
                  framed.waitOn(consumer)==cudaSuccess);
            auto const publishedT=framed.frames().tangent.data;
            auto malformedFrames=frames; malformedFrames.normal={};
            CHECK(framed.Apply(input,{}, {}, {},{dKeep.data(),2},producer,nullptr,
                malformedFrames)==CurveCompactionStatus::InvalidArgument);
            CHECK(framed.frames().tangent.data==publishedT);
            // Same shape, wrong ID association is a device-side rejection.
            CHECK(cudaMemcpy(dFrameIds.data(),std::vector<uint64_t>{70,30}.data(),
                             2*sizeof(uint64_t),cudaMemcpyHostToDevice)==cudaSuccess);
            CHECK(framed.Apply(input,{}, {}, {},{dKeep.data(),2},producer,nullptr,
                frames)==CurveCompactionStatus::InvalidArgument);
            CHECK(framed.frames().tangent.data==publishedT&&
                  Same3(Read(framed.frames().tangent,consumer),{tangents[1]}));
            CHECK(cudaMemcpy(dFrameIds.data(),ids.data(),2*sizeof(uint64_t),
                             cudaMemcpyHostToDevice)==cudaSuccess);
            auto nonFiniteT=tangents; nonFiniteT[1].x=NAN;
            CHECK(cudaMemcpy(dT.data(),nonFiniteT.data(),2*sizeof(float3),
                             cudaMemcpyHostToDevice)==cudaSuccess);
            CHECK(framed.Apply(input,{}, {}, {},{dKeep.data(),2},producer,nullptr,
                frames)==CurveCompactionStatus::InvalidArgument);
            CHECK(framed.frames().tangent.data==publishedT);
            CHECK(cudaMemcpy(dT.data(),tangents.data(),2*sizeof(float3),
                             cudaMemcpyHostToDevice)==cudaSuccess);
            // Aligned frames do not require a duplicate input ID plane.
            auto alignedFrames=frames; alignedFrames.stableIds={};
            CudaCurveCompaction aligned;
            CHECK(aligned.Apply(input,{}, {}, {},{dKeep.data(),2},producer,nullptr,
                alignedFrames)==CurveCompactionStatus::Ok&&
                aligned.Finish(consumer)==CurveCompactionStatus::Ok);
            CHECK(Same3(Read(aligned.frames().tangent,consumer),{tangents[1]})&&
                  Read(aligned.frames().stableIds,consumer)==std::vector<uint64_t>{70});
            for (auto const& keep : {std::vector<uint8_t>{0,1},
                                    std::vector<uint8_t>{1,1},
                                    std::vector<uint8_t>{0,0}}) {
                CHECK(Upload(dKeep,keep));
                CudaCurveCompaction candidate;
                CHECK(candidate.ApplyFreshCounts(input,{dHairT.data(),hairT.size()},
                    {dRoots.data(),2},{reinterpret_cast<float2 const*>(dRootUV.data()),2},
                    {dKeep.data(),2},producer,nullptr,frames)==CurveCompactionStatus::Ok);
                AsyncStatus countsDone, scatterDone;
                CHECK(candidate.FinishFreshCountsAsync(producer,Done,&countsDone)==CurveCompactionStatus::Ok&&
                      cudaStreamSynchronize(producer)==cudaSuccess&&
                      countsDone.calls.load()==1&&countsDone.status.load()==cudaSuccess&&
                      candidate.CommitFreshCounts()==CurveCompactionStatus::Ok);
                CHECK(candidate.ApplyFreshScatter(producer)==CurveCompactionStatus::Ok&&
                      candidate.frames().tangent.size==0);
                CHECK(candidate.FinishFreshScatterAsync(producer,Done,&scatterDone)==CurveCompactionStatus::Ok&&
                      cudaStreamSynchronize(producer)==cudaSuccess&&
                      scatterDone.calls.load()==1&&scatterDone.status.load()==cudaSuccess&&
                      candidate.CommitFreshFinish()==CurveCompactionStatus::Ok);
                std::vector<float3> expectedT,expectedB,expectedN;
                std::vector<uint64_t> expectedIds;
                for(size_t c=0;c<keep.size();++c) if(keep[c]) {
                    expectedT.push_back(tangents[c]);expectedB.push_back(binormals[c]);
                    expectedN.push_back(normals[c]);expectedIds.push_back(ids[c]);
                }
                CHECK(Same3(Read(candidate.frames().tangent,consumer),expectedT)&&
                      Same3(Read(candidate.frames().binormal,consumer),expectedB)&&
                      Same3(Read(candidate.frames().normal,consumer),expectedN)&&
                      Read(candidate.frames().stableIds,consumer)==expectedIds);
                CHECK(Same3(Read(framed.frames().tangent,consumer),{tangents[1]}));
            }
            CHECK(Upload(dKeep,std::vector<uint8_t>{0,1}));
        }

        // Cross-stream use is explicitly fenced by the owned output channels.
        CHECK(compact.recordUse(producer) == cudaSuccess);
        CHECK(compact.waitOn(consumer) == cudaSuccess);
        CHECK(Read(compact.view().points, consumer).size() == 3);

        // Keep-all retains exact source order and topology.
        CHECK(Upload(dKeep, std::vector<uint8_t>{1,1}));
        CHECK(compact.Apply(input, {dHairT.data(), hairT.size()},
                            {dRoots.data(), roots.size()},
                            {reinterpret_cast<float2 const*>(dRootUV.data()), 2},
                            {dKeep.data(), 2}, producer) == CurveCompactionStatus::Ok);
        CHECK(compact.Finish(consumer) == CurveCompactionStatus::Ok);
        CHECK(compact.view().curveCount == 2 && compact.view().pointCount == 5);
        CHECK(Read(compact.view().curveOffsets, consumer) == offsets);
        CHECK(Read(compact.view().stableIds, consumer) == ids);

        // Keep-none retains a zero sentinel offset and empty optional roots.
        CHECK(Upload(dKeep, std::vector<uint8_t>{0,0}));
        CHECK(compact.Apply(input, {dHairT.data(), hairT.size()},
                            {dRoots.data(), roots.size()},
                            {reinterpret_cast<float2 const*>(dRootUV.data()), 2},
                            {dKeep.data(), 2}, producer) == CurveCompactionStatus::Ok);
        CHECK(compact.Finish(consumer) == CurveCompactionStatus::Ok);
        CHECK(compact.view().curveCount == 0 && compact.view().pointCount == 0 &&
              compact.view().curveOffsets.size == 1 && compact.rootPrim().size == 0 &&
              (Read(compact.view().curveOffsets, consumer) == std::vector<uint32_t>{0}));

        // Invalid input never replaces the completed output.
        CHECK(Upload(dKeep, std::vector<uint8_t>{2,1}));
        CHECK(compact.Apply(input, {dHairT.data(), hairT.size()},
                            {dRoots.data(), roots.size()},
                            {reinterpret_cast<float2 const*>(dRootUV.data()), 2},
                            {dKeep.data(), 2}, producer) == CurveCompactionStatus::InvalidArgument);
        CHECK((compact.view().curveCount == 0 && Read(compact.view().curveOffsets, consumer) ==
              std::vector<uint32_t>{0}));
        CHECK(Upload(dKeep, std::vector<uint8_t>{1,1}));
        DeviceBuffer<uint32_t> dBadOffsets;
        CHECK(Upload(dBadOffsets, std::vector<uint32_t>{1,3,5}));
        // Shape validation is device-side but scalar-only and rejects before
        // any scan or pending publication is started.  The first offset must
        // be zero; this malformed buffer must not replace the empty output.
        DeviceCurveGeometryView malformed = input;
        malformed.curveOffsets = {dBadOffsets.data(), 3};
        CHECK(compact.Apply(malformed, {dHairT.data(), hairT.size()},
                            {dRoots.data(), roots.size()},
                            {reinterpret_cast<float2 const*>(dRootUV.data()), 2},
                            {dKeep.data(), 2}, producer) == CurveCompactionStatus::InvalidArgument);
        CHECK(compact.view().curveCount == 0);
        auto badPoints = points; badPoints[3].x = NAN;
        CHECK(Upload(dPoints, badPoints));
        CHECK(compact.Apply(input, {dHairT.data(), hairT.size()},
                            {dRoots.data(), roots.size()},
                            {reinterpret_cast<float2 const*>(dRootUV.data()), 2},
                            {dKeep.data(), 2}, producer) == CurveCompactionStatus::NonFiniteInput);
        CHECK(compact.view().curveCount == 0);
        CHECK(Upload(dPoints, points));

        auto badWidths = widths;
        badWidths[0] = -1.0f;
        CHECK(Upload(dWidths, badWidths));
        CHECK(compact.Apply(input, {dHairT.data(), hairT.size()},
                            {dRoots.data(), roots.size()},
                            {reinterpret_cast<float2 const*>(dRootUV.data()), 2},
                            {dKeep.data(), 2}, producer) == CurveCompactionStatus::InvalidArgument);
        CHECK(Upload(dWidths, widths));
        auto badRoots = roots;
        badRoots[0] = -1;
        CHECK(Upload(dRoots, badRoots));
        CHECK(compact.Apply(input, {dHairT.data(), hairT.size()},
                            {dRoots.data(), roots.size()},
                            {reinterpret_cast<float2 const*>(dRootUV.data()), 2},
                            {dKeep.data(), 2}, producer) == CurveCompactionStatus::InvalidArgument);
        CHECK(Upload(dRoots, roots));

        // Empty input still has a required one-element offset sentinel.
        DeviceBuffer<uint32_t> emptyOffsets;
        CHECK(Upload(emptyOffsets, std::vector<uint32_t>{0}));
        DeviceCurveGeometryView empty = {{}, {}, {}, {emptyOffsets.data(),1}, {}, 0, 0};
        CHECK(compact.Apply(empty, {}, {}, {}, {}, producer) == CurveCompactionStatus::Ok);
        CHECK(compact.Finish(consumer) == CurveCompactionStatus::Ok);
        CHECK(compact.generation() == 4 && compact.view().curveCount == 0 &&
              compact.view().curveOffsets.size == 1);

        // Fresh compaction has a counts proof before it allocates/scatters
        // exact survivor storage.  Inputs remain unpublished until the second
        // native terminal callback and host-only commit.
        CudaCurveCompaction fresh;
        CHECK(Upload(dKeep, std::vector<uint8_t>{0,1}));
        CHECK(fresh.ApplyFreshCounts(input, {dHairT.data(),hairT.size()},
                                     {dRoots.data(),roots.size()},
                                     {reinterpret_cast<float2 const*>(dRootUV.data()),2},
              {dKeep.data(),2}, producer) == CurveCompactionStatus::Ok &&
              fresh.HasUnprovenWork() && fresh.view().curveOffsets.size == 0 &&
              fresh.CommitFreshCounts() == CurveCompactionStatus::NoPendingUpdate &&
              fresh.Apply(input, {}, {}, {}, {dKeep.data(),2}, producer) == CurveCompactionStatus::InvalidArgument);
        AsyncStatus countStatus;
        CHECK(fresh.FinishFreshCountsAsync(producer, Done, &countStatus) == CurveCompactionStatus::Ok &&
              cudaStreamSynchronize(producer) == cudaSuccess && countStatus.calls.load() == 1 &&
              countStatus.status.load() == cudaSuccess && fresh.CommitFreshCounts() == CurveCompactionStatus::Ok &&
              !fresh.HasUnprovenWork() && fresh.view().curveOffsets.size == 0);
        failNextCudaCurveCompactionFreshScatterAllocationForTesting();
        CHECK(fresh.ApplyFreshScatter(producer) == CurveCompactionStatus::CudaError &&
              !fresh.HasUnprovenWork() && fresh.ApplyFreshScatter(producer) == CurveCompactionStatus::Ok &&
              fresh.HasUnprovenWork() && fresh.CommitFreshFinish() == CurveCompactionStatus::NoPendingUpdate);
        AsyncStatus scatterStatus;
        CHECK(fresh.FinishFreshScatterAsync(producer, Done, &scatterStatus) == CurveCompactionStatus::Ok &&
              fresh.FinishFreshScatterAsync(producer, Done, &scatterStatus) == CurveCompactionStatus::NoPendingUpdate &&
              cudaStreamSynchronize(producer) == cudaSuccess && scatterStatus.calls.load() == 1 &&
              scatterStatus.status.load() == cudaSuccess && fresh.CommitFreshFinish() == CurveCompactionStatus::Ok &&
              !fresh.HasUnprovenWork());
        CHECK(fresh.view().curveCount == 1 && fresh.view().pointCount == 3 &&
              Read(fresh.view().curveOffsets,consumer) == std::vector<uint32_t>({0,3}) &&
              Same3(Read(fresh.view().points,consumer), {points[2],points[3],points[4]}) &&
              Same3(Read(fresh.view().restPoints,consumer), {rest[2],rest[3],rest[4]}) &&
              Read(fresh.view().widths,consumer) == std::vector<float>({widths[2],widths[3],widths[4]}) &&
              Read(fresh.view().stableIds,consumer) == std::vector<uint64_t>({ids[1]}) &&
              Read(fresh.hairT(),consumer) == std::vector<float>({hairT[2],hairT[3],hairT[4]}) &&
              Read(fresh.rootPrim(),consumer) == std::vector<int32_t>({roots[1]}) &&
              Same2(Read(fresh.rootUV(),consumer), {rootUV[1]}));

        // A completed fresh generation can return to the legacy API, while a
        // fresh API remains replacement-only so it cannot silently discard the
        // just-published result.
        CHECK(fresh.ApplyFreshCounts(input, {dHairT.data(),hairT.size()},
                                     {dRoots.data(),roots.size()},
                                     {reinterpret_cast<float2 const*>(dRootUV.data()),2},
                                     {dKeep.data(),2}, producer) == CurveCompactionStatus::InvalidArgument &&
              Upload(dKeep, std::vector<uint8_t>{1,1}) &&
              fresh.Apply(input,{dHairT.data(),hairT.size()},{dRoots.data(),roots.size()},
                          {reinterpret_cast<float2 const*>(dRootUV.data()),2},{dKeep.data(),2},producer) == CurveCompactionStatus::Ok &&
              fresh.Finish(consumer) == CurveCompactionStatus::Ok &&
              fresh.view().curveCount == 2 && fresh.view().pointCount == 5);

        // Device validation completes through the counts callback and never
        // allocates or publishes scatter output for malformed offsets/keep.
        CudaCurveCompaction malformedFresh;
        CHECK(Upload(dKeep,std::vector<uint8_t>{2,1}) &&
              malformedFresh.ApplyFreshCounts(input,{dHairT.data(),hairT.size()},
                                               {dRoots.data(),roots.size()},
                                               {reinterpret_cast<float2 const*>(dRootUV.data()),2},
                                               {dKeep.data(),2},producer)==CurveCompactionStatus::Ok);
        AsyncStatus malformedStatus;
        CHECK(malformedFresh.FinishFreshCountsAsync(producer,Done,&malformedStatus)==CurveCompactionStatus::Ok &&
              cudaStreamSynchronize(producer)==cudaSuccess && malformedStatus.status.load()==cudaSuccess &&
              malformedFresh.CommitFreshCounts()==CurveCompactionStatus::InvalidArgument &&
              !malformedFresh.HasUnprovenWork() && malformedFresh.view().curveOffsets.size==0);

        auto badFreshHairT = hairT;
        badFreshHairT[1] = NAN;
        CudaCurveCompaction malformedChannelFresh;
        AsyncStatus malformedChannelStatus;
        CHECK(Upload(dHairT,badFreshHairT) && Upload(dKeep,std::vector<uint8_t>{1,1}) &&
              malformedChannelFresh.ApplyFreshCounts(input,{dHairT.data(),hairT.size()},
                                                      {dRoots.data(),roots.size()},
                                                      {reinterpret_cast<float2 const*>(dRootUV.data()),2},
                                                      {dKeep.data(),2},producer)==CurveCompactionStatus::Ok &&
              malformedChannelFresh.FinishFreshCountsAsync(producer,Done,&malformedChannelStatus)==CurveCompactionStatus::Ok &&
              cudaStreamSynchronize(producer)==cudaSuccess && malformedChannelStatus.status.load()==cudaSuccess &&
              malformedChannelFresh.CommitFreshCounts()==CurveCompactionStatus::InvalidArgument &&
              !malformedChannelFresh.HasUnprovenWork() && Upload(dHairT,hairT));

        // Offset validation makes a malformed range inert before scatter even
        // if the keep channel would otherwise select it.
        DeviceBuffer<uint32_t> dFreshBadOffsets;
        DeviceCurveGeometryView freshMalformedOffsets = input;
        CHECK(Upload(dFreshBadOffsets, std::vector<uint32_t>{0,UINT32_MAX,5}) &&
              Upload(dKeep,std::vector<uint8_t>{1,1}));
        freshMalformedOffsets.curveOffsets = {dFreshBadOffsets.data(),3};
        CudaCurveCompaction malformedOffsetsFresh;
        AsyncStatus malformedOffsetsStatus;
        CHECK(malformedOffsetsFresh.ApplyFreshCounts(freshMalformedOffsets,{dHairT.data(),hairT.size()},
                                                      {dRoots.data(),roots.size()},
                                                      {reinterpret_cast<float2 const*>(dRootUV.data()),2},
                                                      {dKeep.data(),2},producer)==CurveCompactionStatus::Ok &&
              malformedOffsetsFresh.FinishFreshCountsAsync(producer,Done,&malformedOffsetsStatus)==CurveCompactionStatus::Ok &&
              cudaStreamSynchronize(producer)==cudaSuccess && malformedOffsetsStatus.calls.load()==1 &&
              malformedOffsetsFresh.CommitFreshCounts()==CurveCompactionStatus::InvalidArgument &&
              malformedOffsetsFresh.ApplyFreshScatter(producer)==CurveCompactionStatus::InvalidArgument &&
              malformedOffsetsFresh.view().curveOffsets.size==0);

        // All-cull still produces the required zero offset, but only after its
        // own counts and scatter terminal proofs.
        CudaCurveCompaction allCullFresh;
        AsyncStatus allCullCounts, allCullScatter;
        CHECK(Upload(dKeep,std::vector<uint8_t>{0,0}) &&
              allCullFresh.ApplyFreshCounts(input,{dHairT.data(),hairT.size()},
                                            {dRoots.data(),roots.size()},
                                            {reinterpret_cast<float2 const*>(dRootUV.data()),2},
                                            {dKeep.data(),2},producer)==CurveCompactionStatus::Ok &&
              allCullFresh.FinishFreshCountsAsync(producer,Done,&allCullCounts)==CurveCompactionStatus::Ok &&
              cudaStreamSynchronize(producer)==cudaSuccess &&
              allCullFresh.CommitFreshCounts()==CurveCompactionStatus::Ok &&
              allCullFresh.ApplyFreshScatter(producer)==CurveCompactionStatus::Ok &&
              allCullFresh.FinishFreshScatterAsync(producer,Done,&allCullScatter)==CurveCompactionStatus::Ok &&
              cudaStreamSynchronize(producer)==cudaSuccess && allCullScatter.calls.load()==1 &&
              allCullFresh.CommitFreshFinish()==CurveCompactionStatus::Ok &&
              allCullFresh.view().curveCount==0 && allCullFresh.view().pointCount==0 &&
              Read(allCullFresh.view().curveOffsets,consumer)==std::vector<uint32_t>({0}));

        // Empty geometry takes the separate zero-count memset path rather
        // than the scan path, but still requires both terminal proofs.
        CudaCurveCompaction emptyFresh;
        AsyncStatus emptyCounts, emptyScatter;
        CHECK(emptyFresh.ApplyFreshCounts(empty,{},{},{},{},producer)==CurveCompactionStatus::Ok &&
              emptyFresh.FinishFreshCountsAsync(producer,Done,&emptyCounts)==CurveCompactionStatus::Ok &&
              cudaStreamSynchronize(producer)==cudaSuccess && emptyCounts.calls.load()==1 &&
              emptyFresh.CommitFreshCounts()==CurveCompactionStatus::Ok &&
              emptyFresh.ApplyFreshScatter(producer)==CurveCompactionStatus::Ok &&
              emptyFresh.FinishFreshScatterAsync(producer,Done,&emptyScatter)==CurveCompactionStatus::Ok &&
              cudaStreamSynchronize(producer)==cudaSuccess && emptyScatter.calls.load()==1 &&
              emptyFresh.CommitFreshFinish()==CurveCompactionStatus::Ok &&
              emptyFresh.view().curveCount==0 && emptyFresh.view().pointCount==0 &&
              Read(emptyFresh.view().curveOffsets,consumer)==std::vector<uint32_t>({0}));

        // Capture must reject before stream-device queries or fresh allocation.
        cudaStream_t capture = nullptr;
        cudaGraph_t graph = nullptr;
        CudaCurveCompaction capturedFresh;
        CHECK(cudaStreamCreate(&capture)==cudaSuccess &&
              cudaStreamBeginCapture(capture,cudaStreamCaptureModeGlobal)==cudaSuccess &&
              capturedFresh.ApplyFreshCounts(input,{dHairT.data(),hairT.size()},
                                             {dRoots.data(),roots.size()},
                                             {reinterpret_cast<float2 const*>(dRootUV.data()),2},
                                             {dKeep.data(),2},capture)==CurveCompactionStatus::InvalidArgument &&
              !capturedFresh.HasUnprovenWork() &&
              cudaStreamEndCapture(capture,&graph)==cudaSuccess &&
              cudaGraphDestroy(graph)==cudaSuccess && cudaStreamDestroy(capture)==cudaSuccess);
    }
    CHECK(cudaStreamDestroy(consumer) == cudaSuccess);
    CHECK(cudaStreamDestroy(producer) == cudaSuccess);
    std::puts("testUsdGenCudaCurveCompaction: PASS");
    return 0;
}
