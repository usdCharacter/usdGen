#include "utils/pxrInterop.h"
#include "usdGenCurves/cuda/tools.h"
#include "usdGenTestUtils/hydraPublishers.h"
#include "api/testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGenImaging/usdGenImagingSession.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <limits>
#include <thread>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c); return 1; } } while(false)

static UsdGenGraphDesc Desc() {
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom/Hair");
    d.executionBackend = usdGenTestUtils::Backend("cuda");
    UsdGenSurfaceDesc surface; surface.path = SdfPath("/Scalp");
    surface.restPoints = {{0,0,0},{1,0,0},{0,1,0}}; surface.points = surface.restPoints;
    surface.faceVertexCounts = {3}; surface.faceVertexIndices = {0,1,2};
    d.surfaces.push_back(surface);
    UsdGenCurveSetDesc curves; curves.path = SdfPath("/Hair");
    curves.role = UsdGenLane::Chunked; curves.curveRole = TfToken("hair");
    curves.type = TfToken("cubic"); curves.basis = TfToken("catmullRom");
    curves.wrap = TfToken("pinned");
    curves.curveVertexCounts = {2,3}; curves.curveId = {20,10};
    curves.points = {{0,.5f,0},{0,1,0},{-.5f,0,0},{0,0,0},{.5f,0,0}};
    curves.rest = curves.points; curves.skinPrim = {0,0};
    curves.skinPrimUv = {{.2f,.2f},{.2f,.2f}};
    d.curveSets.push_back(curves);
    UsdGenNodeDesc source; source.path = SdfPath("/Groom/Hair/Ops/Source");
    source.type = TfToken("UsdGenCurveSource"); source.curves = {curves.path};
    source.surfaces = {surface.path}; d.nodes = {source}; d.terminal = source.path;
    return d;
}

static UsdGenGraphDesc EmptyDesc() {
    auto d = Desc();
    auto& curves = d.curveSets.front();
    curves.curveVertexCounts.clear(); curves.curveId.clear();
    curves.points.clear(); curves.rest.clear(); curves.widths.clear();
    curves.skinPrim.clear(); curves.skinPrimUv.clear();
    return d;
}

static bool Read(CudaGeometryLease const& lease, std::vector<float3>* points, cudaStream_t stream) {
    points->resize(lease.Geometry().pointCount);
    return cudaMemcpyAsync(points->data(), lease.Geometry().points.data,
        points->size()*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}
static bool Equal(std::vector<float3> const& a, std::vector<float3> const& b) {
    return a.size() == b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float3)) == 0;
}
static bool ValidBounds(UsdGenDeviceGeometryMetadata const& geometry) {
    for (auto const& tile : geometry.tiles) {
        if (!tile.boundsValid) return false;
        for (size_t axis = 0; axis != 3; ++axis)
            if (!std::isfinite(tile.extentMin[axis]) || !std::isfinite(tile.extentMax[axis]) ||
                tile.extentMin[axis] > tile.extentMax[axis]) return false;
    }
    return true;
}
static bool Contains(UsdGenDeviceGeometryMetadata const& geometry, uint64_t point,
                     float3 value) {
    for (auto const& tile : geometry.tiles) {
        if (point < tile.firstPoint || point - tile.firstPoint >= tile.pointCount) continue;
        return tile.boundsValid && value.x >= tile.extentMin[0] && value.x <= tile.extentMax[0] &&
            value.y >= tile.extentMin[1] && value.y <= tile.extentMax[1] &&
            value.z >= tile.extentMin[2] && value.z <= tile.extentMax[2];
    }
    return false;
}
static bool SameBounds(UsdGenDeviceGeometryMetadata const& a,
                       UsdGenDeviceGeometryMetadata const& b) {
    if (a.tiles.size() != b.tiles.size()) return false;
    for (size_t i = 0; i != a.tiles.size(); ++i) {
        if (a.tiles[i].boundsValid != b.tiles[i].boundsValid ||
            a.tiles[i].extentMin != b.tiles[i].extentMin ||
            a.tiles[i].extentMax != b.tiles[i].extentMax) return false;
    }
    return true;
}

int main() {
    usdGenTest::EnsureHydraPublishers();
    cudaStream_t stream = nullptr, reader = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) == cudaSuccess);
    CHECK(cudaStreamCreateWithFlags(&reader,cudaStreamNonBlocking) == cudaSuccess);
    CudaGeometryLease retained;
    std::shared_ptr<const UsdGenDeviceGeneration> retainedGeneration;
    {
        auto engine = std::make_shared<UsdGenSession>();
        engine->SetDevicePublicationEnabled(true);
        usdGenImaging::UsdGenSessionKey key{"gpu-tool-test",SdfPath("/Groom"),0};
        auto imaging = TfCreateRefPtr(new usdGenImaging::UsdGenImagingSession(key,engine,1));
        int publications = 0;
        UsdGenGenerationConstPtr callbackGeneration;
        imaging->RegisterRepublishCallback([&](auto const& payload) {
            ++publications; callbackGeneration = payload.generation;
        });
        auto desc = Desc();
        imaging->StageAndCommit(desc,UsdGenCommitReason::SetTime);
        auto first = engine->Generation();
        if (!first) for (auto const& e : engine->LastDiagnostics().errors) std::fprintf(stderr,"%s\n",e.c_str());
        CHECK(first && first->device && publications == 1 && callbackGeneration == first);
        auto const firstGeometry = first->device->Geometry();
        CHECK(ValidBounds(firstGeometry));
        auto baseLease = AcquireGeometry(first->device,reader); CHECK(baseLease);
        std::vector<float3> original, values, edited;
        CHECK(Read(baseLease,&original,reader) && original.size() == 5 && original[1].x == 0);
        const auto topology = first->device->Geometry().topologyVersion;
        CudaToolSession tool(*engine,stream);
        PickQuery query; for (int i=0;i<4;++i) query.viewProj[i*4+i] = 1;
        query.width=200; query.height=100; query.x=100; query.y=50; query.radiusPx=10;
        PickResult hit;
        CHECK(tool.Pick(query,&hit) && hit.hit && hit.flatIndex == 1 && hit.cv == 1 &&
              hit.curve == 0 && hit.stableId == 10 && hit.distancePx == 0);
        CHECK(tool.Footprint(query) && tool.FootprintIndices().size == 1);
        int32_t selected = -1;
        CHECK(cudaMemcpyAsync(&selected,tool.FootprintIndices().data,sizeof(selected),
                              cudaMemcpyDeviceToHost,stream) == cudaSuccess);
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess && selected == 1);
        CHECK(tool.Begin(77) && !tool.Begin());
        CudaToolSession rival(*engine,stream);
        CHECK(!rival.Begin()); // one reservation per engine, not per wrapper
        DeviceBuffer<int32_t> indices; DeviceBuffer<float3> replacements;
        CHECK(indices.reset(2) == cudaSuccess && replacements.reset(2) == cudaSuccess);
        int32_t hostIndices[2] = {1,1};
        float3 hostPositions[2] = {{.1f,.2f,0},{.3f,.4f,0}};
        auto upload = [&]() {
            return cudaMemcpyAsync(indices.data(),hostIndices,sizeof(hostIndices),cudaMemcpyHostToDevice,stream) == cudaSuccess &&
                cudaMemcpyAsync(replacements.data(),hostPositions,sizeof(hostPositions),cudaMemcpyHostToDevice,stream) == cudaSuccess;
        };
        CHECK(upload());
        CHECK(tool.UpdateIndexed({indices.data(),1},{replacements.data(),1}));
        CHECK(engine->Generation() == first && publications == 1 && engine->NeedsCommit());
        imaging->Commit(UsdGenCommitReason::LiveOverride);
        auto move1 = engine->Generation(); CHECK(move1 != first && publications == 2 && callbackGeneration == move1);
        retainedGeneration = move1->device;
        CHECK(move1->device->Geometry().topologyVersion == topology && move1->device->Geometry().valueVersion == uint64_t(move1->id));
        CHECK(move1->device->Tool().graphVersion == 77 && move1->device->Tool().snapshotVersion == first->device->Identity().generation);
        CHECK(ValidBounds(move1->device->Geometry()) &&
              Contains(move1->device->Geometry(), 1, {.1f, .2f, 0}));
        retained = AcquireGeometry(move1->device,reader); CHECK(retained && Read(retained,&edited,reader));
        CHECK(edited[1].x == .1f && edited[1].y == .2f && edited[4].y == original[4].y);
        CHECK(retained.Geometry().restPoints.data == baseLease.Geometry().restPoints.data &&
              retained.Geometry().stableIds.data == baseLease.Geometry().stableIds.data &&
              retained.Geometry().widths.data == baseLease.Geometry().widths.data);
        CHECK(Read(baseLease,&values,reader) && Equal(values,original));
        CHECK(tool.Pick(query,&hit) && hit.hit && hit.flatIndex == 1 && hit.distancePx == 0);
        // Same move is bitwise idempotent; duplicate updates fail without staging.
        CHECK(!tool.UpdateIndexed({indices.data(),2},{replacements.data(),2}));
        CHECK(!engine->NeedsCommit() && engine->Generation() == move1);
        hostPositions[0].x = std::numeric_limits<float>::quiet_NaN();
        CHECK(upload());
        CHECK(!tool.UpdateIndexed({indices.data(),1},{replacements.data(),1}) &&
              !engine->NeedsCommit() && engine->Generation() == move1);
        // A finite override can still fail only after the conservative
        // Catmull-Rom bounds stage overflows.  That failure must leave the
        // preceding, uncommitted staged edit intact for publication.
        hostIndices[0] = 4; hostPositions[0] = {.6f,.6f,0}; CHECK(upload());
        CHECK(tool.UpdateIndexed({indices.data(),1},{replacements.data(),1}) &&
              engine->NeedsCommit() && engine->Generation() == move1);
        hostPositions[0] = {std::numeric_limits<float>::max(),0,0}; CHECK(upload());
        CHECK(!tool.UpdateIndexed({indices.data(),1},{replacements.data(),1}) &&
              engine->NeedsCommit() && engine->Generation() == move1);
        imaging->Commit(UsdGenCommitReason::LiveOverride);
        auto boundsFailureRetained = engine->Generation();
        CHECK(boundsFailureRetained != move1 &&
              Read(AcquireGeometry(boundsFailureRetained->device,reader),&values,reader) &&
              values[4].x == .6f &&
              ValidBounds(boundsFailureRetained->device->Geometry()) &&
              Contains(boundsFailureRetained->device->Geometry(), 4, {.6f,.6f,0}));
        hostPositions[0] = {.1f,.2f,0};
        hostIndices[0] = 1;
        CHECK(upload());
        CHECK(tool.UpdateIndexed({indices.data(),1},{replacements.data(),1}));
        imaging->Commit(UsdGenCommitReason::LiveOverride);
        auto repeated = engine->Generation();
        CHECK(repeated != boundsFailureRetained &&
              Read(AcquireGeometry(repeated->device,reader),&values,reader) && Equal(values,edited));
        // Two moves before publication replace the staged move; neither
        // accumulates the previous move's sparse values.
        hostIndices[0] = 4; hostPositions[0] = {.8f,.8f,0}; CHECK(upload());
        CHECK(tool.UpdateIndexed({indices.data(),1},{replacements.data(),1}));
        hostPositions[0] = {.7f,.7f,0}; CHECK(upload());
        CHECK(tool.UpdateIndexed({indices.data(),1},{replacements.data(),1}));
        imaging->Commit(UsdGenCommitReason::LiveOverride);
        auto move2 = engine->Generation();
        CHECK(Read(AcquireGeometry(move2->device,reader),&values,reader));
        CHECK(values[1].x == original[1].x && values[1].y == original[1].y && values[4].x == .7f);
        CHECK(ValidBounds(move2->device->Geometry()) &&
              Contains(move2->device->Geometry(), 4, {.7f, .7f, 0}) &&
              !Contains(firstGeometry, 4, {.7f, .7f, 0}));
        CHECK(tool.Cancel() && !tool.active() && engine->Generation() == move2);
        imaging->Commit(UsdGenCommitReason::LiveOverride);
        auto cancelled = engine->Generation(); CHECK(cancelled != move2);
        CHECK(Read(AcquireGeometry(cancelled->device,reader),&values,reader) && Equal(values,original));
        CHECK(cancelled->device->Geometry().topologyVersion == topology);
        CHECK(SameBounds(cancelled->device->Geometry(), firstGeometry));
        CHECK(rival.Begin() && rival.Close());
        // Release closes without publishing; the completed edit can be
        // retained for a later explicit authoring/bake boundary.
        CHECK(tool.Begin());
        CHECK(tool.UpdateIndexed({indices.data(),1},{replacements.data(),1}));
        int beforeClose = publications;
        auto closed = tool.Close(); CHECK(closed && !tool.active() && !engine->NeedsCommit());
        CHECK(engine->Generation() == cancelled && publications == beforeClose);
        CHECK(Read(AcquireGeometry(closed,reader),&values,reader) && values[4].x == .7f);
        // A new graph/frame cook supersedes a staged press snapshot.
        CHECK(tool.Begin());
        CHECK(tool.UpdateIndexed({indices.data(),1},{replacements.data(),1}));
        desc.curveSets[0].points[0][0] += .05f;
        imaging->SetTime(2); imaging->StageAndCommit(desc,UsdGenCommitReason::SetTime);
        auto frame2 = engine->Generation(); CHECK(frame2 != cancelled && frame2->frame == 2);
        CHECK(frame2->device->Geometry().topologyVersion == topology);
        CHECK(Read(AcquireGeometry(frame2->device,reader),&values,reader) && values[3].x == .05f);
        CHECK(!tool.UpdateIndexed({indices.data(),1},{replacements.data(),1}) && !tool.Cancel());
        CHECK(!tool.Close());
        CHECK(engine->Generation() == frame2 && !engine->NeedsCommit());
        CHECK(tool.Begin());
        desc.curveSets[0].curveId[0] = 21;
        imaging->StageAndCommit(desc,UsdGenCommitReason::NoticeBatchEnd);
        auto changed = engine->Generation(); CHECK(changed != frame2);
        CHECK(changed->device->Geometry().topologyVersion != topology);
        CHECK(!tool.UpdateIndexed({indices.data(),1},{replacements.data(),1})); tool.Close();
        CHECK(tool.Begin());
        usdGenUsd::SetGraphDesc(*engine, desc); // pending edits alone invalidate release
        CHECK(!tool.Close() && engine->NeedsCommit());
        imaging->Commit(UsdGenCommitReason::NoticeBatchEnd);
        CHECK(rival.Begin() && rival.Cancel());
        CHECK(rival.Begin());
        CHECK(rival.UpdateIndexed({indices.data(),1},{replacements.data(),1}));
        engine->InvalidateAllValues();
        CHECK(!rival.Close() && engine->NeedsCommit());
        imaging->Commit(UsdGenCommitReason::LiveOverride);
        CHECK(!engine->NeedsCommit());
        CHECK(Read(AcquireGeometry(engine->Generation()->device,reader),&values,reader) &&
              values[4].x == original[4].x);
        CHECK(Read(retained,&values,reader) && Equal(values,edited));
        baseLease = {};
    }
    // Empty C3 geometry has no tile bounds to allocate/read back, yet an
    // empty point edit remains a valid device publication.
    {
        UsdGenSession empty;
        empty.SetDevicePublicationEnabled(true);
        usdGenUsd::SetGraphDesc(empty, EmptyDesc());
        auto firstEmpty = empty.Commit(0, UsdGenCommitReason::SetTime);
        CHECK(firstEmpty && firstEmpty->device &&
              firstEmpty->device->Geometry().pointCount == 0 &&
              firstEmpty->device->Geometry().tiles.empty());
        CudaToolSession emptyTool(empty, stream);
        CHECK(emptyTool.Begin() &&
              emptyTool.UpdateIndexed(DeviceView<const int32_t>{},
                                      DeviceView<const float3>{}));
        auto revisedEmpty = empty.Commit(0, UsdGenCommitReason::LiveOverride);
        CHECK(revisedEmpty && revisedEmpty != firstEmpty && revisedEmpty->device &&
              revisedEmpty->device->Geometry().pointCount == 0 &&
              revisedEmpty->device->Geometry().tiles.empty() &&
              emptyTool.Close());
    }
    // Independently acquired consumers share immutable point/base storage
    // after the engine and tool have been destroyed.
    CHECK(retainedGeneration);
    const int leaseDevice = retainedGeneration->Identity().deviceIndex;
    std::atomic<int> concurrentFailures{0};
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 4; ++worker) {
        workers.emplace_back([&]() {
            if (cudaSetDevice(leaseDevice) != cudaSuccess) {
                ++concurrentFailures;
                return;
            }
            cudaStream_t consumerStream = nullptr;
            if (cudaStreamCreateWithFlags(&consumerStream, cudaStreamNonBlocking) != cudaSuccess) {
                ++concurrentFailures;
                return;
            }
            for (int iteration = 0; iteration < 8; ++iteration) {
                CudaGeometryLease lease =
                    AcquireGeometry(retainedGeneration, consumerStream);
                if (!lease || lease.Geometry().pointCount != 5) {
                    ++concurrentFailures;
                    break;
                }
                CudaGeometryLease copyA = lease;
                CudaGeometryLease copyB = copyA;
                DeviceBuffer<float3> copy;
                if (copy.reset(lease.Geometry().pointCount) != cudaSuccess ||
                    cudaMemcpyAsync(copy.data(), lease.Geometry().points.data,
                                    copy.size() * sizeof(float3), cudaMemcpyDeviceToDevice,
                                    consumerStream) != cudaSuccess) {
                    cudaStreamSynchronize(consumerStream);
                    ++concurrentFailures;
                    break;
                }
                float3 editedPoint{};
                if (cudaMemcpyAsync(&editedPoint, copy.data() + 1, sizeof(editedPoint),
                                    cudaMemcpyDeviceToHost, consumerStream) != cudaSuccess) {
                    cudaStreamSynchronize(consumerStream);
                    ++concurrentFailures;
                    break;
                }
                // All copies share one backend consumer; only the final copy
                // fences/releases it, exercising the lease State refcount.
                lease = {};
                copyA = {};
                copyB = {};
                // Final lease release must finish queued consumer work before
                // returning, without a caller-side synchronization first.
                if (editedPoint.x != .1f || editedPoint.y != .2f ||
                    cudaStreamSynchronize(consumerStream) != cudaSuccess) {
                    ++concurrentFailures;
                    break;
                }
            }
            if (cudaStreamDestroy(consumerStream) != cudaSuccess)
                ++concurrentFailures;
        });
    }
    for (std::thread &worker : workers) worker.join();
    CHECK(concurrentFailures.load() == 0);
    // The final lease alone now retains private points and parent channels.
    retainedGeneration.reset();
    std::vector<float3> stillAlive;
    CHECK(Read(retained,&stillAlive,reader) && stillAlive[1].x == .1f);
    retained = {};
    CHECK(cudaStreamDestroy(reader) == cudaSuccess && cudaStreamDestroy(stream) == cudaSuccess);
    return 0;
}
