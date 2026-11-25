// CUDA generation presentation snapshot: descriptor scalar state must travel
// with the accepted device generation, including tool point revisions.
#include "usdGen/gpu/deviceBuffer.h"
#include "usdGen/gpu/tools.h"
#include "usdGen/session.h"

#include <cuda_runtime.h>

#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>

#include <cstdio>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(c) do { if (!(c)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; \
} } while (false)

static UsdGenGraphDesc MakeDesc()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom/Presentation");
    desc.executionBackend = UsdGenExecutionBackend::Cuda;
    desc.xformMatrix.SetTranslate(GfVec3d(1.0, 2.0, 3.0));
    desc.purpose = TfToken("render");
    desc.visibility = TfToken("invisible");
    desc.materialPath = SdfPath("/Looks/HairA");

    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Scalp");
    surface.worldMatrix.SetIdentity();
    surface.restPoints = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {3};
    surface.faceVertexIndices = {0, 1, 2};
    desc.surfaces.push_back(surface);

    UsdGenCurveSetDesc hair;
    hair.path = SdfPath("/Hair");
    hair.role = UsdGenLane::Chunked;
    hair.curveRole = TfToken("hair");
    hair.type = TfToken("cubic");
    hair.basis = TfToken("bspline");
    hair.wrap = TfToken("pinned");
    hair.curveVertexCounts = {2, 2};
    hair.curveId = {10, 20};
    hair.widths = {.2f, .2f, .2f, .2f};
    hair.widthsInterpolation = TfToken("vertex");
    hair.skinPrim = {0, 0};
    hair.skinPrimUv = {GfVec2f(.25f, .25f), GfVec2f(.5f, .25f)};
    hair.points = {{0, 0, 0}, {0, 1, 0}, {.5f, 0, 0}, {.5f, 1, 0}};
    hair.rest = hair.points;
    desc.curveSets.push_back(hair);

    UsdGenNodeDesc source;
    source.path = SdfPath("/Ops/Source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {hair.path};
    source.surfaces = {surface.path};
    desc.nodes = {source};
    desc.terminal = source.path;
    return desc;
}

static bool SamePresentation(UsdGenDevicePresentationMetadata const &a,
                             UsdGenDevicePresentationMetadata const &b)
{
    return a.description == b.description &&
        a.renderNamespace == b.renderNamespace &&
        a.xformMatrix == b.xformMatrix &&
        a.purpose == b.purpose && a.visibility == b.visibility &&
        a.materialPath == b.materialPath &&
        a.materialPurpose == b.materialPurpose &&
        a.refineLevel == b.refineLevel && a.primOrigin == b.primOrigin &&
        a.dependencySurface == b.dependencySurface;
}

static bool Read(CudaGeometryLease const& lease, std::vector<float3>* points,
                 cudaStream_t stream)
{
    points->resize(lease.Geometry().pointCount);
    return cudaMemcpyAsync(points->data(), lease.Geometry().points.data,
        points->size() * sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}

int main()
{
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

    UsdGenSession session;
    session.SetDevicePublicationEnabled(true);
    auto desc = MakeDesc();
    session.SetGraphDesc(desc);
    auto first = session.Commit(1.0, UsdGenCommitReason::SetTime);
    CHECK(first && first->device && first->devicePresentation &&
          first->tiles.empty() && !session.LastDiagnostics().HasErrors());
    auto const initial = *first->devicePresentation;
    CHECK(initial.description == desc.description &&
          initial.renderNamespace == SdfPath("/Groom/Presentation/__usdGenRender") &&
          initial.xformMatrix == desc.xformMatrix &&
          initial.purpose == TfToken("render") && initial.visibility == TfToken("invisible") &&
          initial.materialPath == SdfPath("/Looks/HairA") &&
          initial.materialPurpose == TfToken("allPurpose") && initial.refineLevel == 2 &&
          initial.primOrigin == desc.description &&
          initial.dependencySurface == SdfPath("/Scalp"));

    auto changed = desc;
    changed.xformMatrix.SetTranslate(GfVec3d(-4.0, 5.0, 6.0));
    changed.purpose = TfToken("proxy");
    changed.visibility = TfToken("inherited");
    changed.materialPath = SdfPath("/Looks/HairB");
    session.SetGraphDesc(changed);
    auto second = session.Commit(2.0, UsdGenCommitReason::SetTime);
    CHECK(second && second != first && second->device && second->devicePresentation &&
          !session.LastDiagnostics().HasErrors());
    auto const changedPresentation = *second->devicePresentation;
    CHECK(changedPresentation.description == changed.description &&
          changedPresentation.renderNamespace == initial.renderNamespace &&
          changedPresentation.xformMatrix == changed.xformMatrix &&
          changedPresentation.purpose == changed.purpose &&
          changedPresentation.visibility == changed.visibility &&
          changedPresentation.materialPath == changed.materialPath &&
          changedPresentation.materialPurpose == TfToken("allPurpose") &&
          changedPresentation.refineLevel == 2 &&
          changedPresentation.primOrigin == changed.description &&
          changedPresentation.dependencySurface == SdfPath("/Scalp"));
    // Retained immutable generations must retain their own descriptor epoch.
    CHECK(SamePresentation(*first->devicePresentation, initial));

    auto const baseGeometry = second->device->Geometry();
    CHECK(baseGeometry.tiles.size() == 1 && baseGeometry.tiles[0].boundsValid);
    auto const oldTile = baseGeometry.tiles[0];
    auto oldLease = AcquireGeometry(second->device, stream);
    std::vector<float3> oldPoints, oldAfter;
    CHECK(oldLease && Read(oldLease, &oldPoints, stream) && oldPoints.size() == 4 &&
          oldPoints[1].x == 0.0f && oldPoints[1].y == 1.0f);

    // Exercise the production point-revision path: it replaces only the
    // device geometry/tool snapshot, so presentation must copy from its base.
    DeviceBuffer<int32_t> indices;
    DeviceBuffer<float3> positions;
    CHECK(indices.reset(1) == cudaSuccess && positions.reset(1) == cudaSuccess);
    int32_t const index = 1;
    // This is beyond the base tile's width-expanded x bound (.6).
    float3 const position{1.0f, 1.0f, .0f};
    CHECK(cudaMemcpyAsync(indices.data(), &index, sizeof(index), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
          cudaMemcpyAsync(positions.data(), &position, sizeof(position), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess);
    CudaToolSession tool(session, stream);
    CHECK(tool.Begin(99));
    CHECK(tool.UpdateIndexed({indices.data(), 1}, {positions.data(), 1}));
    auto revised = session.Commit(2.0, UsdGenCommitReason::LiveOverride);
    CHECK(revised && revised != second && revised->device && revised->devicePresentation &&
          !session.LastDiagnostics().HasErrors());
    CHECK(SamePresentation(*revised->devicePresentation, changedPresentation));
    CHECK(revised->device->Tool().toolId == "pointEdit" &&
          revised->device->Tool().graphVersion == 99);
    auto const& revisedGeometry = revised->device->Geometry();
    CHECK(revisedGeometry.tiles.size() == 1);
    auto const& revisedTile = revisedGeometry.tiles[0];
    CHECK(revisedTile.boundsValid &&
          revisedTile.extentMin[0] <= .9f && revisedTile.extentMax[0] >= 1.1f &&
          revisedTile.extentMin[1] <= .9f && revisedTile.extentMax[1] >= 1.1f &&
          revisedTile.extentMin[2] <= -.1f && revisedTile.extentMax[2] >= .1f);
    CHECK(position.x > oldTile.extentMax[0] &&
          revisedTile.extentMax[0] > oldTile.extentMax[0]);
    CHECK(revisedGeometry.topologyVersion == baseGeometry.topologyVersion &&
          SamePresentation(*revised->devicePresentation, *second->devicePresentation));
    CHECK(second->device->Geometry().tiles[0].extentMin == oldTile.extentMin &&
          second->device->Geometry().tiles[0].extentMax == oldTile.extentMax &&
          Read(oldLease, &oldAfter, stream) && oldAfter.size() == oldPoints.size() &&
          oldAfter[1].x == oldPoints[1].x && oldAfter[1].y == oldPoints[1].y &&
          oldAfter[1].z == oldPoints[1].z);
    oldLease = {};
    CHECK(tool.Close());

    positions.release();
    indices.release();
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::puts("CUDA device presentation: PASS");
    return 0;
}
