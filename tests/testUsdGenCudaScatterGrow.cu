#include "gpu/scatterGrow.h"
#include "usdGen/gpu/generation.h"
#include "usdGen/executionRetirement.h"
#include "usdGen/executionResources.h"
#include "usdGenMath/hash.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <memory>
#include <limits>
#include <vector>

using namespace usdGen::gpu;
using namespace usdGen;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)
static float3 V(float x,float y,float z) { return make_float3(x,y,z); }
static bool Near(float a,float b) { return std::fabs(a-b)<2e-5f; }
static bool Near(float3 a,float3 b) { return Near(a.x,b.x)&&Near(a.y,b.y)&&Near(a.z,b.z); }
struct Relay { std::atomic<int> status{-1}; };
static void Done(cudaStream_t, cudaError_t status, void* p) noexcept { static_cast<Relay*>(p)->status.store(int(status)); }
template <class T> static bool Get(DeviceView<const T> v,std::vector<T>& o,cudaStream_t s) { return cudaMemcpyAsync(o.data(),v.data,o.size()*sizeof(T),cudaMemcpyDeviceToHost,s)==cudaSuccess && cudaStreamSynchronize(s)==cudaSuccess; }
static size_t KindBytes(UsdGenExecutionResourceSnapshot const& s, UsdGenExecutionResourceKind k) { return s.byKind[static_cast<size_t>(k)]; }

int main() {
    int count=0, device=-1; CHECK(cudaGetDeviceCount(&count)==cudaSuccess); if(!count) return 77; CHECK(cudaGetDevice(&device)==cudaSuccess);
    CHECK(ConfigureCudaExecutionResources(device, {size_t{8} << 20, 0}));
    auto resources=FindUsdGenExecutionResourcePool({UsdGenExecutionResourceBackend::Cuda,device}); CHECK(resources);
    auto const baseline=resources->Snapshot();
    cudaStream_t stream=nullptr; CHECK(cudaStreamCreate(&stream)==cudaSuccess);
    {
    auto roots=std::make_shared<ScatterGrowRoots>();
    roots->positions={V(1,2,3),V(-2,0,1)}; roots->stableIds={17,29}; roots->rootPrim={4,8};
    roots->rootUV={make_float2(.1f,.2f),make_float2(.3f,.4f)};
    roots->rootT={V(1,0,0),V(0,1,0)}; roots->rootB={V(0,1,0),V(-1,0,0)}; roots->rootN={V(0,0,2),V(0,3,0)};
    std::weak_ptr<const ScatterGrowRoots> weak=roots;
    ScatterGrowControls controls; controls.cvCount=3; controls.seed=11; controls.length=2; controls.randomLo=.5f; controls.randomHi=1.5f; controls.lift=0.0f; controls.fallbackWidth=.125f;
    CudaScatterGrow grow;
    ScatterGrowRequirements requirements;
    CHECK(GetScatterGrowRequirements(2, controls.cvCount, &requirements) == ScatterGrowStatus::Ok);
    CHECK(grow.BeginFresh(roots,controls,stream)==ScatterGrowStatus::Ok);
    CHECK(resources->Snapshot().usedBytes - baseline.usedBytes == requirements.peakBytes);
    CHECK(grow.view().curveCount==0 && grow.view().points.data==nullptr); // pending work is not visible
    roots.reset(); CHECK(!weak.expired()); // producer owns captured roots through callback proof
    Relay relay; CHECK(grow.FinishFreshAsync(stream,Done,&relay)==ScatterGrowStatus::Ok);
    CHECK(cudaStreamSynchronize(stream)==cudaSuccess && relay.status.load()==int(cudaSuccess));
    CHECK(grow.CommitFreshFinish()==ScatterGrowStatus::Ok && weak.expired());
    auto g=grow.view(); CHECK(g.curveCount==2 && g.pointCount==6 && grow.generation()==1);
    CHECK(g.restPoints.data == g.points.data && g.restPoints.size == g.points.size); // rest aliases points: no D2D copy (r33)
    CHECK(grow.ExclusiveRetainedBytes() == requirements.outputBytes + requirements.statusBytes &&
          resources->Snapshot().usedBytes - baseline.usedBytes ==
              requirements.outputBytes + requirements.statusBytes);
    float3 const* const publishedPoints=g.points.data;
    CHECK(publishedPoints && grow.BeginFresh(std::make_shared<ScatterGrowRoots>(),controls,stream)==ScatterGrowStatus::InvalidArgument && grow.view().points.data==publishedPoints);
    std::vector<float3> p(6),rest(6),t(2),b(2),n(2); std::vector<float> widths(6),hair(6); std::vector<uint32_t> offsets(3); std::vector<uint64_t> ids(2); std::vector<int32_t> prim(2); std::vector<float2> uv(2);
    CHECK(Get(g.points,p,stream)&&Get(g.restPoints,rest,stream)&&Get(g.widths,widths,stream)&&Get(grow.hairT(),hair,stream)&&Get(g.curveOffsets,offsets,stream)&&Get(g.stableIds,ids,stream)&&Get(grow.rootPrim(),prim,stream)&&Get(grow.rootUV(),uv,stream)&&Get(grow.rootT(),t,stream)&&Get(grow.rootB(),b,stream)&&Get(grow.rootN(),n,stream));
    CHECK(offsets[0]==0&&offsets[1]==3&&offsets[2]==6&&ids[0]==17&&ids[1]==29&&prim[0]==4&&prim[1]==8);
    CHECK(Near(uv[1].x,.3f)&&Near(uv[1].y,.4f)&&Near(t[0],V(1,0,0))&&Near(b[1],V(-1,0,0))&&Near(n[0],V(0,0,2)));
    for(unsigned c=0;c<2;++c) for(unsigned i=0;i<3;++i) { float q=float(i)/2; float3 root=c?V(-2,0,1):V(1,2,3); float3 dir=c?V(0,1,0):V(0,0,1); uint64_t id=c?29:17; float target=static_cast<float>(controls.length*(controls.randomLo+double(UsdGenDraw01(controls.seed,id,kSaltGrow))*(controls.randomHi-controls.randomLo))); float3 expected=V(root.x+dir.x*(target*q),root.y+dir.y*(target*q),root.z+dir.z*(target*q)); CHECK(Near(p[c*3+i],expected)&&Near(rest[c*3+i],expected)&&Near(hair[c*3+i],q)&&Near(widths[c*3+i],.125f)); }
    ScatterGrowControls bad=controls; bad.direction=static_cast<ScatterGrowDirection>(99); auto invalid=std::make_shared<ScatterGrowRoots>(); *invalid=ScatterGrowRoots{}; CHECK(grow.BeginFresh(invalid,bad,stream)==ScatterGrowStatus::InvalidArgument);
    grow.ReclassifyPublishedGeneration();
    auto pinned=resources->Snapshot();
    CHECK(KindBytes(pinned,UsdGenExecutionResourceKind::Pinned)==KindBytes(baseline,UsdGenExecutionResourceKind::Pinned)+grow.ExclusiveRetainedBytes());
    CHECK(KindBytes(pinned,UsdGenExecutionResourceKind::Active)==KindBytes(baseline,UsdGenExecutionResourceKind::Active));

    // An empty captured Scatter result is still a valid topology: one zero
    // offset makes downstream ragged consumers distinguish it from no owner.
    CudaScatterGrow empty; auto emptyRoots=std::make_shared<ScatterGrowRoots>(); Relay emptyRelay;
    CHECK(empty.BeginFresh(emptyRoots,controls,stream)==ScatterGrowStatus::Ok);
    CHECK(empty.FinishFreshAsync(stream,Done,&emptyRelay)==ScatterGrowStatus::Ok);
    CHECK(cudaStreamSynchronize(stream)==cudaSuccess && emptyRelay.status.load()==int(cudaSuccess));
    CHECK(empty.CommitFreshFinish()==ScatterGrowStatus::Ok && empty.curveCount()==0 && empty.pointCount()==0);
    std::vector<uint32_t> emptyOffsets(1); CHECK(Get(empty.view().curveOffsets,emptyOffsets,stream)&&emptyOffsets[0]==0);
    }
    CHECK(resources->Snapshot().usedBytes==baseline.usedBytes && resources->Snapshot().byKind==baseline.byKind);

    // Lift is an angular Rodrigues rotation around the captured root-frame B,
    // not a distance offset.  A Z axis makes this catch a hard-coded world-Y
    // implementation while the two signs exercise the full schema range.
    {
        auto roots = std::make_shared<ScatterGrowRoots>();
        roots->positions = {V(0, 0, 0)}; roots->stableIds = {77};
        roots->rootPrim = {-1}; roots->rootUV = {make_float2(0, 0)};
        roots->rootT = {V(0, 1, 0)}; roots->rootB = {V(0, 0, 1)};
        roots->rootN = {V(1, 0, 0)};
        ScatterGrowControls controls; controls.cvCount = 3; controls.length = 2.0;
        controls.direction = ScatterGrowDirection::RootNormal;
        for (float lift : {90.0f, -90.0f}) {
            controls.lift = lift;
            CudaScatterGrow grow; Relay relay;
            CHECK(grow.BeginFresh(roots, controls, stream) == ScatterGrowStatus::Ok &&
                  grow.FinishFreshAsync(stream, Done, &relay) == ScatterGrowStatus::Ok &&
                  cudaStreamSynchronize(stream) == cudaSuccess &&
                  relay.status.load() == int(cudaSuccess) &&
                  grow.CommitFreshFinish() == ScatterGrowStatus::Ok);
            std::vector<float3> points(3);
            CHECK(Get(grow.view().points, points, stream));
            float const expectedY = lift > 0 ? 2.0f : -2.0f;
        CHECK(Near(points[0], V(0, 0, 0)) && Near(points[1], V(0, expectedY * .5f, 0)) &&
                  Near(points[2], V(0, expectedY, 0)));
        }

        for(float random : {0.f,1.f}) {
            controls.lift=90.f; controls.azimuth=90.f; controls.azimuthRandom=random;
            CudaScatterGrow azimuthGrow; Relay relay;
            CHECK(azimuthGrow.BeginFresh(roots,controls,stream)==ScatterGrowStatus::Ok &&
                  azimuthGrow.FinishFreshAsync(stream,Done,&relay)==ScatterGrowStatus::Ok &&
                  cudaStreamSynchronize(stream)==cudaSuccess && relay.status.load()==int(cudaSuccess) &&
                  azimuthGrow.CommitFreshFinish()==ScatterGrowStatus::Ok);
            float const angle=(90.f+random*360.f*(UsdGenDraw01(controls.seed,77,kSaltGrowAzimuth)-.5f))*3.14159265358979323846f/180.f;
            std::vector<float3> points(3);
            CHECK(Get(azimuthGrow.view().points,points,stream) &&
                  Near(points[2],V(0,2*std::cos(angle),2*std::sin(angle))));
        }
        CudaScatterGrow invalidAzimuth;
        controls.azimuth=361.f;
        CHECK(invalidAzimuth.BeginFresh(roots,controls,stream)==ScatterGrowStatus::InvalidArgument);
        controls.azimuth=0; controls.azimuthRandom=1.01f;
        CHECK(invalidAzimuth.BeginFresh(roots,controls,stream)==ScatterGrowStatus::InvalidArgument);
        controls.azimuthRandom=0;

        // Both target narrowing overflow and endpoint arithmetic overflow are
        // rejected before any device allocation or stream submission.
        auto badTarget = std::make_shared<ScatterGrowRoots>(*roots);
        ScatterGrowControls huge = controls; huge.lift = 0; huge.length = std::numeric_limits<double>::max();
        CudaScatterGrow rejectedTarget;
        CHECK(rejectedTarget.BeginFresh(badTarget, huge, stream) == ScatterGrowStatus::NonFiniteInput);
        auto badOutput = std::make_shared<ScatterGrowRoots>(*roots);
        badOutput->positions[0] = V(std::numeric_limits<float>::max(), 0, 0);
        ScatterGrowControls endpoint = controls; endpoint.lift = 0;
        endpoint.length = std::numeric_limits<float>::max();
        CudaScatterGrow rejectedOutput;
        CHECK(rejectedOutput.BeginFresh(badOutput, endpoint, stream) == ScatterGrowStatus::NonFiniteInput);
    }
    CHECK(resources->Snapshot().usedBytes==baseline.usedBytes && resources->Snapshot().byKind==baseline.byKind);

    // Publication transfers the complete topology owner into the same lease
    // and retirement protocol used by other CUDA generations. Keep a lease
    // after dropping the first wrapper while a second output is published.
    auto retirement = GetOrCreateUsdGenExecutionRetirementService(
        {UsdGenExecutionResourceBackend::Cuda, device}, {32});
    CHECK(retirement);
    auto makeGeneration = [&](float length, uint64_t version, bool zero = false) {
        auto roots = std::make_shared<ScatterGrowRoots>();
        if (!zero) {
            roots->positions = {V(1, 2, 3)};
            roots->stableIds = {17};
            roots->rootPrim = {4};
            roots->rootUV = {make_float2(.1f, .2f)};
            roots->rootT = {V(1, 0, 0)};
            roots->rootB = {V(0, 1, 0)};
            roots->rootN = {V(0, 0, 1)};
        }
        ScatterGrowControls controls;
        controls.cvCount = 3;
        controls.length = length;
        auto owner = std::make_unique<CudaScatterGrow>();
        Relay done;
        if (owner->BeginFresh(roots, controls, stream) != ScatterGrowStatus::Ok ||
            owner->FinishFreshAsync(stream, Done, &done) != ScatterGrowStatus::Ok ||
            cudaStreamSynchronize(stream) != cudaSuccess ||
            done.status.load() != int(cudaSuccess) ||
            owner->CommitFreshFinish() != ScatterGrowStatus::Ok)
            return std::shared_ptr<const UsdGenDeviceGeneration>{};
        std::string reason;
        auto result = MakeScatterGrowGeneration(std::move(owner), version, &reason);
        if (!result) std::fprintf(stderr, "publication failed: %s\n", reason.c_str());
        return result;
    };
    {
        auto first = makeGeneration(2.0f, 1);
        CHECK(first && first->Owner()->ProducerReady());
        auto retained = AcquireGeometry(first, stream);
        CHECK(retained);
        auto const* firstPoints = retained.Geometry().points.data;
        first.reset();
        auto second = makeGeneration(4.0f, 2);
        CHECK(second && second->Geometry().pointCount == 3);
        auto current = AcquireGeometry(second, stream);
        CHECK(current && current.Geometry().points.data != firstPoints);
        std::vector<float3> oldPoints(3), newPoints(3);
        CHECK(Get(retained.Geometry().points, oldPoints, stream));
        CHECK(Get(current.Geometry().points, newPoints, stream));
        CHECK(Near(oldPoints.back(), V(1, 2, 5)) &&
              Near(newPoints.back(), V(1, 2, 7)));
        std::vector<float3> frame(1);
        CHECK(Get(retained.RootT(), frame, stream) && Near(frame[0], V(1, 0, 0)));
        CHECK(Get(retained.RootB(), frame, stream) && Near(frame[0], V(0, 1, 0)));
        CHECK(Get(retained.RootN(), frame, stream) && Near(frame[0], V(0, 0, 1)));
        auto revision = MakePointRevisionGeneration(second, 4, {});
        CHECK(revision);
        auto revised = AcquireGeometry(revision, stream);
        CHECK(revised && revised.RootN().data == current.RootN().data);
        CHECK(second->Owner()->ExclusiveRetainedBytes() > 0);
        auto empty = makeGeneration(1.0f, 3, true);
        CHECK(empty && empty->Owner()->ProducerReady() &&
              empty->Geometry().curveCount == 0 && empty->Geometry().pointCount == 0);
        auto emptyLease = AcquireGeometry(empty, stream);
        CHECK(emptyLease && emptyLease.Geometry().curveOffsets.size == 1);
        std::vector<uint32_t> offsets(1, 99);
        CHECK(Get(emptyLease.Geometry().curveOffsets, offsets, stream) && offsets[0] == 0);
    }

    // Settle the preceding lease retirements before comparing whole-pool
    // category deltas; their asynchronous cleanup is unrelated to this owner.
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == baseline.usedBytes);

    // Scatter/Grow may publish fresh COW values while retaining its expanded
    // topology owner.  The overrides replace only points/widths; rest data
    // and root frames must continue to come from that immutable owner.
    {
        auto roots = std::make_shared<ScatterGrowRoots>();
        roots->positions = {V(3, 4, 5)};
        roots->stableIds = {101};
        roots->rootPrim = {12};
        roots->rootUV = {make_float2(.6f, .7f)};
        roots->rootT = {V(1, 0, 0)};
        roots->rootB = {V(0, 1, 0)};
        roots->rootN = {V(0, 0, 1)};
        ScatterGrowControls controls;
        controls.cvCount = 3;
        controls.length = 2.0f;
        controls.fallbackWidth = .125f;
        ScatterGrowRequirements requirements;
        CHECK(GetScatterGrowRequirements(1, controls.cvCount, &requirements) ==
              ScatterGrowStatus::Ok);
        auto owner = std::make_unique<CudaScatterGrow>();
        Relay done;
        CHECK(owner->BeginFresh(roots, controls, stream) == ScatterGrowStatus::Ok);
        CHECK(owner->FinishFreshAsync(stream, Done, &done) == ScatterGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess &&
              done.status.load() == int(cudaSuccess));
        CHECK(owner->CommitFreshFinish() == ScatterGrowStatus::Ok);
        auto const basePoints = owner->view().points;
        auto const baseRest = owner->view().restPoints;
        auto const baseT = owner->rootT();
        auto const baseB = owner->rootB();
        auto const baseN = owner->rootN();
        std::vector<float3> expectedPoints(3), expectedRest(3), expectedT(1),
            expectedB(1), expectedN(1);
        CHECK(Get(basePoints, expectedPoints, stream) && Get(baseRest, expectedRest, stream) &&
              Get(baseT, expectedT, stream) && Get(baseB, expectedB, stream) &&
              Get(baseN, expectedN, stream));

        auto widths = std::make_unique<DeviceBuffer<float>>();
        auto points = std::make_unique<DeviceBuffer<float3>>();
        CHECK(widths->reset(3) == cudaSuccess && points->reset(3) == cudaSuccess);
        float const replacementWidths[] = {.25f, .5f, .75f};
        float3 const replacementPoints[] = {V(9, 8, 7), V(6, 5, 4), V(3, 2, 1)};
        CHECK(cudaMemcpyAsync(widths->data(), replacementWidths, sizeof(replacementWidths),
                              cudaMemcpyHostToDevice, stream) == cudaSuccess &&
              cudaMemcpyAsync(points->data(), replacementPoints, sizeof(replacementPoints),
                              cudaMemcpyHostToDevice, stream) == cudaSuccess &&
              widths->recordUse(stream) == cudaSuccess && points->recordUse(stream) == cudaSuccess &&
              cudaStreamSynchronize(stream) == cudaSuccess);
        auto const* replacementWidthPtr = widths->data();
        auto const* replacementPointPtr = points->data();
        size_t const overrideBytes = 3 * (sizeof(float) + sizeof(float3));
        size_t const retainedBytes = requirements.outputBytes + requirements.statusBytes + overrideBytes;
        auto const beforePublication = resources->Snapshot();
        std::string reason;
        auto generation = MakeScatterGrowGeneration(std::move(owner), 9, &reason, false,
            UINT64_MAX, {}, std::move(widths), std::move(points));
        CHECK(generation && !owner && !widths && !points &&
              generation->Owner()->ProducerReady());
        auto const afterPublication = resources->Snapshot();
        CHECK(generation->Owner()->ExclusiveRetainedBytes() == retainedBytes &&
              afterPublication.usedBytes == beforePublication.usedBytes &&
              KindBytes(afterPublication, UsdGenExecutionResourceKind::Pinned) ==
                  KindBytes(beforePublication, UsdGenExecutionResourceKind::Pinned) + retainedBytes &&
              KindBytes(afterPublication, UsdGenExecutionResourceKind::Scratch) ==
                  KindBytes(beforePublication, UsdGenExecutionResourceKind::Scratch) -
                      requirements.statusBytes &&
              KindBytes(afterPublication, UsdGenExecutionResourceKind::Active) ==
                  KindBytes(beforePublication, UsdGenExecutionResourceKind::Active) -
                      requirements.outputBytes - overrideBytes);

        auto retained = AcquireGeometry(generation, stream);
        CHECK(retained && retained.Geometry().points.data == replacementPointPtr &&
              retained.Geometry().widths.data == replacementWidthPtr &&
              retained.Geometry().restPoints.data == baseRest.data &&
              retained.RootT().data == baseT.data && retained.RootB().data == baseB.data &&
              retained.RootN().data == baseN.data);
        std::vector<float3> gotPoints(3), gotRest(3), gotBasePoints(3), gotT(1), gotB(1), gotN(1);
        std::vector<float> gotWidths(3);
        CHECK(Get(retained.Geometry().points, gotPoints, stream) &&
              Get(retained.Geometry().widths, gotWidths, stream) &&
              Get(retained.Geometry().restPoints, gotRest, stream) &&
              Get(basePoints, gotBasePoints, stream) && Get(retained.RootT(), gotT, stream) &&
              Get(retained.RootB(), gotB, stream) && Get(retained.RootN(), gotN, stream));
        for (size_t i = 0; i != 3; ++i) {
            CHECK(Near(gotPoints[i], replacementPoints[i]) && Near(gotWidths[i], replacementWidths[i]) &&
                  Near(gotRest[i], expectedRest[i]) && Near(gotBasePoints[i], expectedPoints[i]));
        }
        CHECK(Near(gotT[0], expectedT[0]) && Near(gotB[0], expectedB[0]) &&
              Near(gotN[0], expectedN[0]));
        generation.reset();
        // The lease owns the generation's retirement consumer, so the COW
        // override and immutable Scatter/Grow topology remain valid after all
        // factory inputs and the public generation wrapper are gone.
        CHECK(Get(retained.Geometry().points, gotPoints, stream) &&
              Near(gotPoints[2], replacementPoints[2]) &&
              Get(retained.RootN(), gotN, stream) && Near(gotN[0], expectedN[0]));
        retained = {};
    }
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == baseline.usedBytes);
    CHECK(cudaStreamDestroy(stream)==cudaSuccess); std::puts("testUsdGenCudaScatterGrow: PASS"); return 0;
}
