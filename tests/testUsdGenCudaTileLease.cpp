#include "usdGenCurves/types/curveDeviceLayout.h"
#include "cudaGlFixture.h"
#include "usdGenCurves/cuda/generation.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (false)

template <class T>
static std::vector<T> Copy(DeviceView<const T> view, cudaStream_t stream = nullptr)
{
    std::vector<T> result(view.size);
    if (!result.empty() && cudaMemcpyAsync(result.data(), view.data,
                                      result.size() * sizeof(T),
                                      cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        result.clear();
    if (!result.empty() && cudaStreamSynchronize(stream) != cudaSuccess)
        result.clear();
    return result;
}

static std::shared_ptr<const UsdGenDeviceGeneration> MakeNoRootsGeneration()
{
    auto source = std::make_unique<CudaCurveSource>();
    const int32_t counts[] = {2, 3};
    const float3 points[] = {
        make_float3(-.5f, -.6f, 0), make_float3(-.5f, .6f, 0),
        make_float3(.5f, -.6f, 0), make_float3(.5f, 0, 0),
        make_float3(.5f, .6f, 0)};
    CurveSourceInput input;
    input.curveVertexCounts = {counts, 2};
    input.points = {points, 5};
    input.restPoints = {points, 5};
    input.fallbackWidth = .2f;
    if (source->Set(input, nullptr) != CurveSourceStatus::Ok ||
        source->Finish(nullptr) != CurveSourceStatus::Ok) return {};
    std::string reason;
    return MakeSourceGeneration(std::move(source), 0, &reason);
}

int main()
{
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    auto generation = usdGenTest::MakeCudaGlFixture(.2f, 0);
    CHECK(generation);

    std::vector<UsdGenDeviceTileMetadata> tiles{
        {1, 0, 0, 0, 0}, {3, 0, 1, 0, 2},
        {9, 1, 1, 2, 3}, {12, 2, 0, 5, 0}};
    tiles[0].boundsValid = true;
    tiles[1].extentMin = {-.6f, -.7f, -.1f};
    tiles[1].extentMax = {-.4f, .7f, .1f};
    tiles[1].boundsValid = true;
    tiles[2].extentMin = {.4f, -.7f, -.1f};
    tiles[2].extentMax = {.6f, .7f, .1f};
    tiles[2].boundsValid = true;
    tiles[3].boundsValid = true;
    UsdGenDeviceLayoutMetadata const topology = UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic, UsdGenCurveLayoutBasis::BSpline, UsdGenCurveLayoutWrap::Pinned);
    std::string reason;
    auto tiled = WithTileMetadata(generation, tiles, &reason, topology);
    CHECK(tiled);
    CHECK(tiled->Geometry().topologyVersion == generation->Geometry().topologyVersion);
    CHECK(tiled->Geometry().valueVersion == generation->Geometry().valueVersion);
    CHECK(UsdGenCurveLayoutTypeOf(tiled->Geometry().layoutMetadata) == UsdGenCurveLayoutType::Cubic &&
          UsdGenCurveLayoutBasisOf(tiled->Geometry().layoutMetadata) == UsdGenCurveLayoutBasis::BSpline &&
          UsdGenCurveLayoutWrapOf(tiled->Geometry().layoutMetadata) == UsdGenCurveLayoutWrap::Pinned &&
          tiled->Geometry().tiles[1].boundsValid &&
          tiled->Geometry().tiles[1].extentMin[0] == -.6f &&
          tiled->Geometry().tiles[2].extentMax[1] == .7f);

    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    auto first = AcquireGeometryTile(tiled, 3, 0, stream);
    auto second = AcquireGeometryTile(tiled, 9, 0, stream);
    auto emptyTile = AcquireGeometryTile(tiled, 1, 0, stream);
    CHECK(first && second);
    CHECK(emptyTile && emptyTile.View().curveOffsets.size == 1);
    CHECK(emptyTile.View().points.size == 0 && emptyTile.View().stableIds.size == 0);
    CHECK(first.View().curveOffsets.size == 2 && first.View().points.size == 2);
    CHECK(second.View().curveOffsets.size == 2 && second.View().points.size == 3);
    CHECK(first.View().stableIds.size == 1 && second.View().stableIds.size == 1);
    CHECK(first.View().rootPrim.size == 1 && first.View().rootUV.size == 1);
    CHECK(first.View().curveOffsets.data != second.View().curveOffsets.data);
    CHECK(first.View().range.firstPoint == 0 && second.View().range.firstPoint == 2);
    auto firstOffsets = Copy(first.View().curveOffsets, stream);
    auto secondOffsets = Copy(second.View().curveOffsets, stream);
    auto firstIds = Copy(first.View().stableIds, stream);
    auto secondIds = Copy(second.View().stableIds, stream);
    auto firstRoots = Copy(first.View().rootPrim, stream);
    auto firstUvs = Copy(first.View().rootUV, stream);
    auto firstWidths = Copy(first.View().widths, stream);
    auto firstHairT = Copy(first.View().hairT, stream);
    CHECK(firstOffsets == std::vector<uint32_t>({0, 2}));
    CHECK(secondOffsets == std::vector<uint32_t>({2, 5}));
    CHECK(firstIds == std::vector<uint64_t>({0x123456780000005bULL}));
    CHECK(secondIds == std::vector<uint64_t>({0xabcdef0100000025ULL}));
    CHECK(firstRoots == std::vector<int32_t>({4}));
    CHECK(firstUvs.size() == 1 && firstUvs[0].x == .1f && firstUvs[0].y == .2f);
    CHECK(firstWidths.size() == 2 && std::isfinite(firstWidths[0]) &&
          std::isfinite(firstWidths[1]) && std::fabs(firstWidths[0] - .2f) < 1e-6f &&
          std::fabs(firstWidths[1] - .2f) < 1e-6f);
    CHECK(firstHairT.size() == 2 && firstHairT[0] == 0.0f && firstHairT[1] == 1.0f);

    CHECK(!AcquireGeometryTile(tiled, 3, 1, stream));
    CHECK(!AcquireGeometryTile(tiled, 77, 0, stream));

    auto empty = WithTileMetadata(generation, {}, &reason);
    CHECK(!empty);
    std::vector<UsdGenDeviceTileMetadata> gap{{3, 0, 1, 0, 2}, {9, 1, 0, 2, 0}};
    CHECK(!WithTileMetadata(generation, gap, &reason));
    std::vector<UsdGenDeviceTileMetadata> overflow{{3, 0, UINT64_MAX, 0, UINT64_MAX}};
    CHECK(!WithTileMetadata(generation, overflow, &reason));
    std::vector<UsdGenDeviceTileMetadata> pointOverflow{{3, 0, 1, UINT32_MAX, 1}, {9, 1, 1, 0, 4}};
    CHECK(!WithTileMetadata(generation, pointOverflow, &reason));
    std::vector<UsdGenDeviceTileMetadata> duplicate{{3, 0, 1, 0, 2}, {3, 1, 1, 2, 3}};
    CHECK(!WithTileMetadata(generation, duplicate, &reason));
    auto invalidBounds = tiles;
    invalidBounds[1].extentMin[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!WithTileMetadata(generation, invalidBounds, &reason, topology));

    // The tile lease retains the parent owner independently of both handles.
    auto retained = AcquireGeometryTile(tiled, 9, 0, stream);
    CHECK(retained);
    // Release all other leases before dropping the generation wrappers.
    first = {};
    second = {};
    emptyTile = {};
    tiled.reset();
    generation.reset();
    CHECK(retained.View().points.size == 3);

    DeviceBuffer<float3> copiedPoints;
    CHECK(copiedPoints.reset(retained.View().points.size) == cudaSuccess);
    CHECK(cudaMemcpyAsync(copiedPoints.data(), retained.View().points.data,
                          copiedPoints.size() * sizeof(float3),
                          cudaMemcpyDeviceToDevice, stream) == cudaSuccess);
    std::vector<float3> hostPoints(copiedPoints.size());
    CHECK(cudaMemcpyAsync(hostPoints.data(), copiedPoints.data(),
                          hostPoints.size() * sizeof(float3),
                          cudaMemcpyDeviceToHost, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(hostPoints[0].x == .5f && hostPoints[0].y == -.6f);
    CHECK(hostPoints[1].x == .5f && hostPoints[1].y == 0.0f);
    CHECK(hostPoints[2].x == .5f && hostPoints[2].y == .6f);
    auto moved = std::move(retained);
    CHECK(moved && moved.View().points.size == 3);
    CHECK(!retained && retained.View().points.size == 0);
    DeviceBuffer<float3> fenceCopy;
    CHECK(fenceCopy.reset(3) == cudaSuccess);
    std::vector<float3> fenceHost(3);
    CHECK(cudaMemcpyAsync(fenceCopy.data(), moved.View().points.data,
                          3 * sizeof(float3), cudaMemcpyDeviceToDevice,
                          stream) == cudaSuccess);
    CHECK(cudaMemcpyAsync(fenceHost.data(), fenceCopy.data(),
                          3 * sizeof(float3), cudaMemcpyDeviceToHost,
                          stream) == cudaSuccess);
    cudaEvent_t fence = nullptr;
    CHECK(cudaEventCreateWithFlags(&fence, cudaEventDisableTiming) == cudaSuccess);
    CHECK(cudaEventRecord(fence, stream) == cudaSuccess);
    moved = {};
    CHECK(cudaEventQuery(fence) == cudaSuccess);
    CHECK(fenceHost[0].x == .5f && fenceHost[2].y == .6f);
    CHECK(cudaEventDestroy(fence) == cudaSuccess);
    fenceCopy = {};
    copiedPoints = {};

    auto noRoots = MakeNoRootsGeneration();
    CHECK(noRoots);
    UsdGenDeviceTileMetadata noRootsRange{2, 0, 2, 0, 5};
    noRootsRange.extentMin = {-.6f, -.7f, -.1f};
    noRootsRange.extentMax = {.6f, .7f, .1f};
    noRootsRange.boundsValid = true;
    auto noRootsTiled = WithTileMetadata(noRoots, {noRootsRange}, &reason,
        topology);
    CHECK(noRootsTiled);
    auto noRootsTile = AcquireGeometryTile(noRootsTiled, 2, 0, stream);
    CHECK(noRootsTile && noRootsTile.View().rootPrim.size == 0 &&
          noRootsTile.View().rootUV.size == 0);
    DeviceBuffer<float3> revisedPoints;
    CHECK(revisedPoints.reset(5) == cudaSuccess);
    const float3 changed[] = {
        make_float3(-.75f, -.6f, 0), make_float3(-.5f, .6f, 0),
        make_float3(.5f, -.6f, 0), make_float3(.5f, 0, 0),
        make_float3(.5f, .6f, 0)};
    CHECK(cudaMemcpyAsync(revisedPoints.data(), changed, sizeof(changed),
                          cudaMemcpyHostToDevice, stream) == cudaSuccess);
    CHECK(revisedPoints.recordUse(stream) == cudaSuccess);
    auto revision = MakePointRevisionGeneration(noRootsTiled, 1,
        std::make_unique<DeviceBuffer<float3>>(std::move(revisedPoints)), {}, &reason);
    CHECK(revision);
    CHECK(revision->Geometry().tiles.size() == 1 &&
          revision->Geometry().tiles[0].elementCount == 2);
    CHECK(revision->Geometry().topologyVersion == noRootsTiled->Geometry().topologyVersion);
    CHECK(!revision->Geometry().tiles[0].boundsValid &&
          UsdGenCurveLayoutTypeOf(revision->Geometry().layoutMetadata) == UsdGenCurveLayoutType::Cubic);
    auto oldRest = Copy(noRootsTile.View().restPoints, stream);
    auto revisionTile = AcquireGeometryTile(revision, 2, 1, stream);
    CHECK(revisionTile && revisionTile.View().points.size == 5);
    auto revisedValues = Copy(revisionTile.View().points, stream);
    auto revisedRest = Copy(revisionTile.View().restPoints, stream);
    CHECK(revisedValues.size() == 5 && revisedValues[0].x == -.75f);
    CHECK(oldRest.size() == 5 && revisedRest.size() == 5 &&
          oldRest[0].x == -.5f && revisedRest[0].x == -.5f);
    noRootsTile = {};
    noRootsTiled.reset();
    noRoots.reset();
    revision.reset();
    CHECK(revisionTile.View().points.size == 5);
    revisionTile = {};
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    return 0;
}
