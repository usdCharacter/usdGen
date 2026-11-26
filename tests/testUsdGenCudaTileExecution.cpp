// End-to-end CUDA session tile publication: Source -> Length(cull) -> Width.
#include "usdGenCurves/publish/curvePublication.h"
#include "testBackend.h"
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "usdGenUsd/lower.h"
#include "usdGen/session.h"
#include "usdGenCurves/cuda/generation.h"

#include <cuda_runtime.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(c) do { if (!(c)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; \
} } while (false)

template <class T>
static std::vector<T> Copy(DeviceView<const T> view, cudaStream_t stream)
{
    if (view.size == 0 || !view.data) return {};
    std::vector<T> result(view.size);
    cudaError_t const copy = cudaMemcpyAsync(result.data(), view.data,
        result.size() * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaError_t const fence = cudaStreamSynchronize(stream);
    if (copy != cudaSuccess || fence != cudaSuccess) return {};
    return result;
}

static UsdGenGraphDesc MakeDesc(size_t curves, float threshold, float width,
                                int tileTarget = 64)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom/Description");
    d.executionBackend = usdGenTestUtils::Backend("cuda");
    d.defaultWidth = width;
    d.tileTarget = tileTarget;

    // A real scalp binding makes the CurveSource fixture valid C3 input.
    UsdGenSurfaceDesc scalp;
    scalp.path = SdfPath("/Scalp");
    scalp.worldMatrix.SetIdentity();
    scalp.restPoints = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    scalp.points = scalp.restPoints;
    scalp.faceVertexCounts = {3};
    scalp.faceVertexIndices = {0, 1, 2};
    d.surfaces.push_back(scalp);

    UsdGenCurveSetDesc hair;
    hair.path = SdfPath("/Hair");
    hair.role = UsdGenLane::Chunked;
    hair.curveRole = TfToken("hair");
    hair.type = TfToken("cubic");
    hair.basis = TfToken("bspline");
    hair.wrap = TfToken("pinned");
    hair.curveVertexCounts.assign(curves, 2);
    hair.curveId.resize(curves);
    hair.skinPrim.assign(curves, 0);
    hair.skinPrimUv.assign(curves, GfVec2f(.25f, .25f));
    hair.points.resize(curves * 2);
    for (size_t curve = 0; curve != curves; ++curve) {
        float const length = curves == 1100
            ? (curve < 512 ? 2.0f : (curve < 1024 ? .2f : 3.0f))
            : 2.0f;
        float const x = float(curve) * .001f;
        hair.curveId[curve] = curve;
        hair.points[2 * curve] = GfVec3f(x, 0, 0);
        hair.points[2 * curve + 1] = GfVec3f(x, length, 0);
    }
    hair.rest = hair.points;
    d.curveSets.push_back(hair);

    UsdGenNodeDesc source;
    source.path = SdfPath("/Ops/Source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {hair.path};
    source.surfaces = {scalp.path};
    UsdGenNodeDesc length;
    length.path = SdfPath("/Ops/Length");
    length.type = TfToken("UsdGenLength");
    length.inputs = {source.path};
    length.params.push_back({TfToken("length:mode"), VtValue(TfToken("cull")), false});
    length.params.push_back({TfToken("cullThreshold"), VtValue(threshold), false});
    UsdGenNodeDesc widthNode;
    widthNode.path = SdfPath("/Ops/Width");
    widthNode.type = TfToken("UsdGenWidth");
    widthNode.inputs = {length.path};
    widthNode.params.push_back({TfToken("width"), VtValue(width), false});
    d.nodes = {source, length, widthNode};
    d.terminal = widthNode.path;
    return d;
}

static bool SameTiles(std::vector<UsdGenDeviceTileMetadata> const& a,
                      std::vector<UsdGenDeviceTileMetadata> const& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i != a.size(); ++i) {
        if (a[i].tile != b[i].tile || a[i].firstElement != b[i].firstElement ||
            a[i].elementCount != b[i].elementCount || a[i].firstPoint != b[i].firstPoint ||
            a[i].pointCount != b[i].pointCount) return false;
    }
    return true;
}

static bool CoversEveryCurve(std::vector<UsdGenDeviceTileMetadata> const& tiles,
                             uint64_t curveCount, uint64_t pointCount)
{
    if (tiles.empty() || tiles.front().firstElement != 0 ||
        tiles.front().firstPoint != 0) return false;
    uint64_t curves = 0;
    uint64_t points = 0;
    for (UsdGenDeviceTileMetadata const& tile : tiles) {
        if (tile.firstElement != curves || tile.firstPoint != points ||
            tile.elementCount == 0 || tile.pointCount != tile.elementCount * 2) return false;
        curves += tile.elementCount;
        points += tile.pointCount;
    }
    return curves == curveCount && points == pointCount;
}

int main()
{
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

    UsdGenSession session;
    session.SetDevicePublicationEnabled(true);
    auto desc = MakeDesc(1100, 1.0f, .1f);
    usdGenUsd::SetGraphDesc(session, desc);
    auto first = session.Commit(1, UsdGenCommitReason::SetTime);
    CHECK(first && first->device && usdGen::UsdGenCurvePublicationOf(*first).tiles.empty() &&
          !session.LastDiagnostics().HasErrors());
    auto const& initialTiles = first->device->Geometry().tiles;
    CHECK(SameTiles(initialTiles, {{0, 0, 512, 0, 1024},
                                   {1, 512, 0, 1024, 0},
                                   {2, 512, 76, 1024, 152}}));
    CHECK(UsdGenCurveLayoutTypeOf(first->device->Geometry().layoutMetadata) == UsdGenCurveLayoutType::Cubic &&
          UsdGenCurveLayoutBasisOf(first->device->Geometry().layoutMetadata) == UsdGenCurveLayoutBasis::BSpline &&
          UsdGenCurveLayoutWrapOf(first->device->Geometry().layoutMetadata) == UsdGenCurveLayoutWrap::Pinned);
    for (UsdGenDeviceTileMetadata const &tile : initialTiles) {
        CHECK(tile.boundsValid);
        for (size_t axis = 0; axis != tile.extentMin.size(); ++axis)
            CHECK(std::isfinite(tile.extentMin[axis]) && std::isfinite(tile.extentMax[axis]) &&
                  tile.extentMin[axis] <= tile.extentMax[axis]);
    }
    CHECK((initialTiles[1].extentMin == std::array<float, 3>{} &&
           initialTiles[1].extentMax == std::array<float, 3>{}));
    auto const firstBounds = initialTiles[0];
    uint64_t const topology = first->device->Geometry().topologyVersion;

    auto firstTile = AcquireGeometryTile(first->device, 0,
                                         first->device->Identity().generation, stream);
    auto emptyTile = AcquireGeometryTile(first->device, 1,
                                         first->device->Identity().generation, stream);
    auto lastTile = AcquireGeometryTile(first->device, 2,
                                        first->device->Identity().generation, stream);
    CHECK(firstTile && emptyTile && lastTile);
    auto firstView = firstTile.View();
    auto emptyView = emptyTile.View();
    auto lastView = lastTile.View();
    CHECK(firstView.range.elementCount == 512 && firstView.range.pointCount == 1024 &&
          lastView.range.firstElement == 512 && lastView.range.elementCount == 76 &&
          lastView.range.firstPoint == 1024 && lastView.range.pointCount == 152);
    auto firstOffsets = Copy(firstView.curveOffsets, stream);
    auto firstIds = Copy(firstView.stableIds, stream);
    auto firstWidths = Copy(firstView.widths, stream);
    auto firstPoints = Copy(firstView.points, stream);
    auto emptyOffsets = Copy(emptyView.curveOffsets, stream);
    auto lastOffsets = Copy(lastView.curveOffsets, stream);
    auto lastIds = Copy(lastView.stableIds, stream);
    auto lastWidths = Copy(lastView.widths, stream);
    auto lastPoints = Copy(lastView.points, stream);
    CHECK(firstOffsets.size() == 513 && firstOffsets.front() == 0 &&
          firstOffsets[1] == 2 && firstOffsets.back() == 1024);
    CHECK(firstIds.size() == 512 && firstIds.front() == 0 && firstIds.back() == 511 &&
          firstWidths.size() == 1024 && firstPoints.size() == 1024);
    CHECK(emptyOffsets == std::vector<uint32_t>({1024}) && emptyView.stableIds.size == 0 &&
          emptyView.points.size == 0 && emptyView.widths.size == 0);
    CHECK(lastOffsets.size() == 77 && lastOffsets.front() == 1024 &&
          lastOffsets.back() == 1176 && lastIds.size() == 76 &&
          lastIds.front() == 1024 && lastIds.back() == 1099 &&
          lastWidths.size() == 152 && lastPoints.size() == 152);
    for (float value : firstWidths) CHECK(std::fabs(value - .1f) < 1e-6f);
    for (float value : lastWidths) CHECK(std::fabs(value - .1f) < 1e-6f);
    for (size_t i = 0; i != firstPoints.size(); ++i) {
        float const radius = firstWidths[i] * .5f;
        CHECK(firstBounds.extentMin[0] <= firstPoints[i].x - radius &&
              firstBounds.extentMax[0] >= firstPoints[i].x + radius &&
              firstBounds.extentMin[1] <= firstPoints[i].y - radius &&
              firstBounds.extentMax[1] >= firstPoints[i].y + radius &&
              firstBounds.extentMin[2] <= firstPoints[i].z - radius &&
              firstBounds.extentMax[2] >= firstPoints[i].z + radius);
    }
    CHECK(std::fabs(firstPoints[1].y - 2.0f) < 1e-6f &&
          std::fabs(lastPoints[1].y - 3.0f) < 1e-6f);

    auto wider = desc;
    wider.nodes[2].params[0].value = VtValue(.2f);
    usdGenUsd::SetGraphDesc(session, wider);
    auto second = session.Commit(2, UsdGenCommitReason::SetTime);
    CHECK(second && second != first && second->device && usdGen::UsdGenCurvePublicationOf(*second).tiles.empty() &&
          !session.LastDiagnostics().HasErrors());
    CHECK(second->device->Geometry().topologyVersion == topology &&
          SameTiles(second->device->Geometry().tiles, initialTiles));
    auto const &widerBounds = second->device->Geometry().tiles[0];
    CHECK(widerBounds.boundsValid && widerBounds.extentMin[0] < firstBounds.extentMin[0] &&
          widerBounds.extentMax[0] > firstBounds.extentMax[0]);
    auto widerTile = AcquireGeometryTile(second->device, 0,
                                         second->device->Identity().generation, stream);
    CHECK(widerTile);
    auto widerWidths = Copy(widerTile.View().widths, stream);
    CHECK(widerWidths.size() == 1024);
    for (float value : widerWidths) CHECK(std::fabs(value - .2f) < 1e-6f);
    // The old owning lease must retain the first generation's device width buffer.
    auto retainedWidths = Copy(firstTile.View().widths, stream);
    CHECK(retainedWidths.size() == 1024);
    for (float value : retainedWidths) CHECK(std::fabs(value - .1f) < 1e-6f);

    auto catmull = wider;
    catmull.curveSets[0].basis = TfToken("catmullRom");
    usdGenUsd::SetGraphDesc(session, catmull);
    auto third = session.Commit(3, UsdGenCommitReason::SetTime);
    CHECK(third && third->device &&
          UsdGenCurveLayoutBasisOf(third->device->Geometry().layoutMetadata) == UsdGenCurveLayoutBasis::CatmullRom &&
          third->device->Geometry().topologyVersion != topology);
    auto catmullWidth = catmull;
    catmullWidth.nodes[2].params[0].value = VtValue(.3f);
    usdGenUsd::SetGraphDesc(session, catmullWidth);
    auto fourth = session.Commit(4, UsdGenCommitReason::SetTime);
    CHECK(fourth && fourth->device &&
          fourth->device->Geometry().topologyVersion ==
              third->device->Geometry().topologyVersion &&
          fourth->device->Geometry().tiles[0].extentMin[0] <
              third->device->Geometry().tiles[0].extentMin[0]);

    auto allCulled = catmullWidth;
    allCulled.nodes[1].params[1].value = VtValue(4.0f);
    usdGenUsd::SetGraphDesc(session, allCulled);
    auto empty = session.Commit(5, UsdGenCommitReason::SetTime);
    CHECK(empty && empty != second && empty->device && usdGen::UsdGenCurvePublicationOf(*empty).tiles.empty() &&
          !session.LastDiagnostics().HasErrors());
    auto const& emptyTiles = empty->device->Geometry().tiles;
    CHECK(emptyTiles.size() == 3);
    for (size_t tile = 0; tile != emptyTiles.size(); ++tile) {
        CHECK(emptyTiles[tile].tile == tile && emptyTiles[tile].elementCount == 0 &&
              emptyTiles[tile].pointCount == 0 && emptyTiles[tile].firstElement == 0 &&
              emptyTiles[tile].firstPoint == 0 && emptyTiles[tile].boundsValid);
        CHECK((emptyTiles[tile].extentMin == std::array<float, 3>{} &&
               emptyTiles[tile].extentMax == std::array<float, 3>{}));
    }
    auto emptyLease = AcquireGeometryTile(empty->device, 2,
                                          empty->device->Identity().generation, stream);
    CHECK(emptyLease && emptyLease.View().curveOffsets.size == 1 &&
          Copy(emptyLease.View().curveOffsets, stream) == std::vector<uint32_t>({0}));

    auto invalid = allCulled;
    invalid.nodes[1].params[0].value = VtValue(TfToken("not-a-length-mode"));
    usdGenUsd::SetGraphDesc(session, invalid);
    CHECK(session.Commit(6, UsdGenCommitReason::SetTime) == empty);
    CHECK(session.LastDiagnostics().HasErrors() && session.NeedsCommit() &&
          SameTiles(empty->device->Geometry().tiles, emptyTiles));

    // Exercise clamp-floor partitioning through the session publication path,
    // rather than calling the tile builder or WithTileMetadata directly.
    UsdGenSession denseSession;
    denseSession.SetDevicePublicationEnabled(true);
    auto dense = MakeDesc(17000, .1f, .1f, 64);
    usdGenUsd::SetGraphDesc(denseSession, dense);
    auto dense64 = denseSession.Commit(5, UsdGenCommitReason::SetTime);
    CHECK(dense64 && dense64->device && usdGen::UsdGenCurvePublicationOf(*dense64).tiles.empty() &&
          !denseSession.LastDiagnostics().HasErrors());
    auto const& tiles64 = dense64->device->Geometry().tiles;
    CHECK(tiles64.size() == 34 && CoversEveryCurve(tiles64, 17000, 34000));
    uint64_t const denseTopology = dense64->device->Geometry().topologyVersion;
    auto dense32Desc = dense;
    dense32Desc.tileTarget = 32;
    usdGenUsd::SetGraphDesc(denseSession, dense32Desc);
    auto dense32 = denseSession.Commit(6, UsdGenCommitReason::SetTime);
    CHECK(dense32 && dense32 != dense64 && dense32->device && usdGen::UsdGenCurvePublicationOf(*dense32).tiles.empty() &&
          !denseSession.LastDiagnostics().HasErrors());
    auto const& tiles32 = dense32->device->Geometry().tiles;
    CHECK(tiles32.size() == 32 && CoversEveryCurve(tiles32, 17000, 34000) &&
          dense32->device->Geometry().topologyVersion != denseTopology);

    dense32.reset();
    dense64.reset();
    emptyLease = {};
    widerTile = {};
    emptyTile = {};
    lastTile = {};
    firstTile = {};
    empty.reset();
    second.reset();
    first.reset();
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::puts("CUDA tile session: PASS");
    return 0;
}