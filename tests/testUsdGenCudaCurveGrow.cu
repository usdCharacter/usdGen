#include "usdGenCurves/cuda/curveGrow.h"
#include "usdGenCuda/backendId.h"
#include "usdGen/executionResources.h"
#include "usdGenMath/hash.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static float3 V(float x, float y, float z) { return make_float3(x, y, z); }
static bool Near(float a, float b) { return std::fabs(a-b) < 3.e-5f; }
static bool Near(float3 a, float3 b) { return Near(a.x,b.x) && Near(a.y,b.y) && Near(a.z,b.z); }
struct Relay { std::atomic<int> status{-1}; };
static void Done(cudaStream_t, cudaError_t status, void* data) noexcept { static_cast<Relay*>(data)->status.store(int(status)); }
template <class T> static bool Upload(DeviceBuffer<T>& dst, std::vector<T> const& values, cudaStream_t stream) {
    return dst.reset(values.size()) == cudaSuccess &&
        (values.empty() || (cudaMemcpyAsync(dst.data(), values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                            cudaStreamSynchronize(stream) == cudaSuccess));
}
template <class T> static bool Download(DeviceView<const T> input, std::vector<T>* output, cudaStream_t stream) {
    return output->size() == input.size && (!input.size ||
        cudaMemcpyAsync(output->data(), input.data, input.size*sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}

struct InputOwner {
    DeviceBuffer<float3> points, rest, rootT, rootB, rootN;
    DeviceBuffer<float> widths;
    DeviceBuffer<uint32_t> offsets;
    DeviceBuffer<uint64_t> ids;
    DeviceBuffer<uint64_t> frameStableIds;
    DeviceBuffer<int32_t> rootPrim;
    DeviceBuffer<float2> rootUV;
    CurveGrowInput Input(size_t curves, size_t pointsCount) const {
        return {{points.view(), rest.view(), widths.view(), offsets.view(), ids.view(), curves, pointsCount},
                rootPrim.view(), rootUV.view(), rootT.view(), rootB.view(), rootN.view(), frameStableIds.view()};
    }
};

int main(int argc, char** argv) {
    for (uint32_t span : {2u, 16777218u, UINT32_MAX}) {
        CHECK(usdGen::gpu::detail::CurveGrowWidthLowerIndex(0.f, span) == 0);
        CHECK(usdGen::gpu::detail::CurveGrowWidthLowerIndex(1.f, span) == span - 1);
        CHECK(usdGen::gpu::detail::CurveGrowWidthLowerIndex(.5f, span) < span);
    }
    int count=0, device=-1;
    CHECK(cudaGetDeviceCount(&count)==cudaSuccess); if (!count) return 77;
    CHECK(cudaGetDevice(&device)==cudaSuccess);
    CHECK(ConfigureCudaExecutionResources(device, {size_t{8} << 20, 0}));
    auto pool=FindUsdGenExecutionResourcePool({usdGen::UsdGenCudaBackendId(),device}); CHECK(pool);
    auto const baseline=pool->Snapshot();
    cudaStream_t stream=nullptr; CHECK(cudaStreamCreate(&stream)==cudaSuccess);
    if (argc == 2 && std::string(argv[1]) == "--quarantine") {
        auto input=std::make_shared<InputOwner>();
        CHECK(Upload(input->points,{V(0,0,0),V(0,0,1)},stream));
        CHECK(Upload(input->rest,{V(0,0,0),V(0,0,1)},stream));
        CHECK(Upload(input->offsets,{0u,2u},stream)&&Upload(input->ids,{1ull},stream));
        CHECK(Upload(input->rootPrim,{0},stream)&&Upload(input->rootUV,{make_float2(0,0)},stream));
        CHECK(Upload(input->rootT,{V(1,0,0)},stream)&&Upload(input->rootB,{V(0,1,0)},stream)&&Upload(input->rootN,{V(0,0,1)},stream));
        auto const inputBaseline=pool->Snapshot(); std::weak_ptr<InputOwner> weak=input;
        { CudaCurveGrow grow; CurveGrowControls c;
          CHECK(grow.BeginFresh(input->Input(1,2),input,c,stream)==CurveGrowStatus::Ok);
          input.reset(); grow.MarkUnprovenWork(); }
        CHECK(!weak.expired() && pool->Snapshot().usedBytes > inputBaseline.usedBytes);
        // The stream is synchronized only so process teardown has no racing
        // device work; the intentionally abandoned owner remains charged.
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess);
        CHECK(cudaStreamDestroy(stream)==cudaSuccess); return 0;
    }
    {
        auto input=std::make_shared<InputOwner>();
        CHECK(Upload(input->points,{V(120,0,0),V(121,0,0),V(122,0,0),V(110,0,0),V(111,0,0)},stream));
        CHECK(Upload(input->rest,{V(20,0,0),V(21,0,0),V(22,0,0),V(10,0,0),V(11,0,0)},stream));
        CHECK(Upload(input->widths,{.1f,.2f,.4f,1.f,.5f},stream));
        CHECK(Upload(input->offsets,{0u,3u,5u},stream));
        CHECK(Upload(input->ids,{40ull,7ull},stream));
        CHECK(Upload(input->rootPrim,{2,1},stream));
        CHECK(Upload(input->rootUV,{make_float2(.2f,.3f),make_float2(.4f,.5f)},stream));
        CHECK(Upload(input->rootT,{V(1,0,0),V(0,1,0)},stream));
        CHECK(Upload(input->rootB,{V(0,1,0),V(-1,0,0)},stream));
        CHECK(Upload(input->rootN,{V(0,0,1),V(0,0,2)},stream));
        CurveGrowControls controls; controls.cvCount=4; controls.seed=19; controls.length=2.0;
        controls.randomLo=.5; controls.randomHi=1.5;
        CurveGrowRequirements requirements;
        CHECK(GetCurveGrowRequirements(2,4,&requirements)==CurveGrowStatus::Ok);
        auto const inputBaseline=pool->Snapshot();
        std::weak_ptr<InputOwner> weak=input;
        CudaCurveGrow grow;
        CHECK(grow.BeginFresh(input->Input(2,5),input,controls,stream)==CurveGrowStatus::Ok);
        CHECK(pool->Snapshot().usedBytes-inputBaseline.usedBytes==requirements.peakBytes);
        input.reset(); CHECK(!weak.expired());
        Relay relay; CHECK(grow.FinishFreshAsync(stream,Done,&relay)==CurveGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess && relay.status.load()==int(cudaSuccess));
        CHECK(grow.CommitFreshFinish()==CurveGrowStatus::Ok && weak.expired());
        CHECK(pool->Snapshot().usedBytes==baseline.usedBytes+requirements.peakBytes);
        auto geometry=grow.view(); CHECK(geometry.curveCount==2 && geometry.pointCount==8 && grow.generation()==1);
        std::vector<float3> points(8), rest(8), t(2), b(2), n(2); std::vector<float> widths(8), hair(8);
        std::vector<uint32_t> offsets(3); std::vector<uint64_t> ids(2); std::vector<int32_t> prim(2); std::vector<float2> uv(2);
        CHECK(Download(geometry.points,&points,stream)&&Download(geometry.restPoints,&rest,stream)&&Download(geometry.widths,&widths,stream)&&Download(grow.hairT(),&hair,stream)&&Download(geometry.curveOffsets,&offsets,stream)&&Download(geometry.stableIds,&ids,stream)&&Download(grow.rootPrim(),&prim,stream)&&Download(grow.rootUV(),&uv,stream)&&Download(grow.rootT(),&t,stream)&&Download(grow.rootB(),&b,stream)&&Download(grow.rootN(),&n,stream));
        CHECK(offsets==std::vector<uint32_t>({0,4,8}) && ids==std::vector<uint64_t>({40,7}) && prim==std::vector<int32_t>({2,1}));
        CHECK(Near(t[0],V(1,0,0))&&Near(b[1],V(-1,0,0))&&Near(n[1],V(0,0,2))&&Near(uv[1].x,.4f));
        float const rootsX[]={120.f,110.f}, restsX[]={20.f,10.f};
        float const sourceWidths[][3]={{.1f,.2f,.4f},{1.f,.5f,0.f}};
        uint32_t const sourceCount[]={3,2}; uint64_t const sourceId[]={40,7};
        for(uint32_t c=0;c<2;++c) for(uint32_t i=0;i<4;++i) {
            float h=float(i)/3.f;
            float target=static_cast<float>(controls.length*(controls.randomLo+
                double(UsdGenDraw01(controls.seed,sourceId[c],kSaltGrow))*(controls.randomHi-controls.randomLo)));
            float d=target*h;
            CHECK(Near(points[c*4+i],V(rootsX[c],0,d)) && Near(rest[c*4+i],V(restsX[c],0,d)) && Near(hair[c*4+i],h));
            float pos=h*float(sourceCount[c]-1); uint32_t lo=uint32_t(pos), hi=std::min(lo+1,sourceCount[c]-1);
            CHECK(Near(widths[c*4+i],sourceWidths[c][lo]+(sourceWidths[c][hi]-sourceWidths[c][lo])*(pos-float(lo))));
        }
        auto const* published=geometry.points.data;
        CHECK(grow.BeginFresh({}, {}, controls, stream)==CurveGrowStatus::InvalidArgument && grow.view().points.data==published);
        grow.ReclassifyPublishedGeneration();
    }
    CHECK(pool->Snapshot().usedBytes==baseline.usedBytes);

    // Empty C3 still carries a one-element offsets owner and never launches a
    // zero-grid kernel.
    {
        auto input=std::make_shared<InputOwner>(); CHECK(Upload(input->offsets,{0u},stream));
        CudaCurveGrow empty; CurveGrowControls c; Relay relay;
        CHECK(empty.BeginFresh(input->Input(0,0),input,c,stream)==CurveGrowStatus::Ok);
        CHECK(empty.FinishFreshAsync(stream,Done,&relay)==CurveGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess&&relay.status.load()==int(cudaSuccess));
        CHECK(empty.CommitFreshFinish()==CurveGrowStatus::Ok&&empty.curveCount()==0&&empty.pointCount()==0);
        std::vector<uint32_t> offsets(1); CHECK(Download(empty.view().curveOffsets,&offsets,stream)&&offsets[0]==0);
        CHECK(Upload(input->frameStableIds,{7ull},stream));
        auto malformed = input->Input(0,0);
        malformed.rootT = {nullptr,1}; malformed.rootB = {nullptr,1};
        malformed.rootN = {nullptr,1};
        CudaCurveGrow rejected;
        CHECK(rejected.BeginFresh(malformed,input,c,stream) == CurveGrowStatus::InvalidTopology);
        // An empty active topology may retain a structurally valid map domain.
        CHECK(Upload(input->rootT,{V(1,0,0)},stream) &&
              Upload(input->rootB,{V(0,1,0)},stream) &&
              Upload(input->rootN,{V(0,0,1)},stream));
        CudaCurveGrow mappedEmpty; Relay mappedRelay;
        CHECK(mappedEmpty.BeginFresh(input->Input(0,0),input,c,stream) == CurveGrowStatus::Ok &&
              mappedEmpty.FinishFreshAsync(stream,Done,&mappedRelay) == CurveGrowStatus::Ok &&
              cudaStreamSynchronize(stream) == cudaSuccess &&
              mappedRelay.status.load() == int(cudaSuccess) &&
              mappedEmpty.CommitFreshFinish() == CurveGrowStatus::Ok &&
              mappedEmpty.curveCount() == 0 && mappedEmpty.pointCount() == 0);
    }
    {
        auto malformed=std::make_shared<InputOwner>(); CHECK(Upload(malformed->offsets,{1u},stream));
        CudaCurveGrow empty; CurveGrowControls c; Relay relay;
        CHECK(empty.BeginFresh(malformed->Input(0,0),malformed,c,stream)==CurveGrowStatus::Ok);
        CHECK(empty.FinishFreshAsync(stream,Done,&relay)==CurveGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess&&relay.status.load()==int(cudaSuccess));
        CHECK(empty.CommitFreshFinish()==CurveGrowStatus::InvalidTopology&&!empty.HasUnprovenWork());
    }
    CHECK(pool->Snapshot().usedBytes==baseline.usedBytes);

    // Absent input widths get the explicit fallback. RootTangent and literal
    // directions share the same fresh topology path; reversed random bounds
    // are canonicalized before the device draw.
    {
        auto input=std::make_shared<InputOwner>();
        CHECK(Upload(input->points,{V(1,2,3),V(1,2,4)},stream)&&Upload(input->rest,{V(9,8,7),V(9,8,8)},stream));
        CHECK(Upload(input->offsets,{0u,2u},stream)&&Upload(input->ids,{99ull},stream)&&Upload(input->rootPrim,{3},stream));
        CHECK(Upload(input->rootUV,{make_float2(.3f,.4f)},stream)&&Upload(input->rootT,{V(1,0,0)},stream)&&Upload(input->rootB,{V(0,1,0)},stream)&&Upload(input->rootN,{V(0,0,1)},stream));
        CurveGrowControls c; c.cvCount=2; c.length=2; c.randomLo=1.5; c.randomHi=.5; c.fallbackWidth=.25f; c.direction=CurveGrowDirection::RootTangent;
        CudaCurveGrow tangent; Relay relay;
        CHECK(tangent.BeginFresh(input->Input(1,2),input,c,stream)==CurveGrowStatus::Ok);
        // No terminal callback has been installed: commit must retain the
        // pending owner. Do not block a stream callback while issuing CUDA
        // calls on that stream; instrumentation may make those calls wait.
        CHECK(tangent.CommitFreshFinish()==CurveGrowStatus::NoPendingUpdate&&tangent.HasUnprovenWork());
        CHECK(tangent.FinishFreshAsync(stream,Done,&relay)==CurveGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess&&relay.status.load()==int(cudaSuccess)&&tangent.CommitFreshFinish()==CurveGrowStatus::Ok);
        std::vector<float3> points(2),rest(2); std::vector<float> widths(2); CHECK(Download(tangent.view().points,&points,stream)&&Download(tangent.view().restPoints,&rest,stream)&&Download(tangent.view().widths,&widths,stream));
        float target=float(2.0*(.5+double(UsdGenDraw01(0,99,kSaltGrow))));
        CHECK(Near(points[1],V(1+target,2,3))&&Near(rest[1],V(9+target,8,7))&&Near(widths[0],.25f)&&Near(widths[1],.25f));
        CurveGrowControls literal=c; literal.direction=CurveGrowDirection::Literal; literal.literalDirection=V(0,1,0);
        CudaCurveGrow literalGrow; Relay literalRelay;
        CHECK(literalGrow.BeginFresh(input->Input(1,2),input,literal,stream)==CurveGrowStatus::Ok&&literalGrow.FinishFreshAsync(stream,Done,&literalRelay)==CurveGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess&&literalRelay.status.load()==int(cudaSuccess)&&literalGrow.CommitFreshFinish()==CurveGrowStatus::Ok);
        points.assign(2,{}); CHECK(Download(literalGrow.view().points,&points,stream)&&Near(points[1],V(1,2+target,3)));
        // Use a non-world-Y frame axis so the test detects hard-coded
        // rotation axes in the native lowering.
        CHECK(Upload(input->rootB,{V(0,0,1)},stream) &&
              Upload(input->rootN,{V(1,0,0)},stream));
        CurveGrowControls angular=c; angular.length=1.0; angular.randomLo=1.0; angular.randomHi=1.0;
        angular.direction=CurveGrowDirection::RootNormal; angular.lift=90.0f;
        CudaCurveGrow lifted; Relay liftedRelay;
        CHECK(lifted.BeginFresh(input->Input(1,2),input,angular,stream)==CurveGrowStatus::Ok &&
              lifted.FinishFreshAsync(stream,Done,&liftedRelay)==CurveGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess &&
              liftedRelay.status.load()==int(cudaSuccess) &&
              lifted.CommitFreshFinish()==CurveGrowStatus::Ok);
        points.assign(2,{}); rest.assign(2,{});
        CHECK(Download(lifted.view().points,&points,stream) &&
              Download(lifted.view().restPoints,&rest,stream) &&
              Near(points[1],V(1,3,3)) && Near(rest[1],V(9,9,7)));
        angular.lift=-90.0f;
        CudaCurveGrow negativeLift; Relay negativeRelay;
        CHECK(negativeLift.BeginFresh(input->Input(1,2),input,angular,stream)==CurveGrowStatus::Ok &&
              negativeLift.FinishFreshAsync(stream,Done,&negativeRelay)==CurveGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess &&
              negativeRelay.status.load()==int(cudaSuccess) &&
              negativeLift.CommitFreshFinish()==CurveGrowStatus::Ok);
        points.assign(2,{}); CHECK(Download(negativeLift.view().points,&points,stream) &&
              Near(points[1],V(1,1,3)));

        CurveGrowControls lift=c; lift.lift=90.1f; CudaCurveGrow rejected;
        for (float random : {0.0f, 1.0f}) {
            angular.lift=90.f; angular.azimuth=90.f; angular.azimuthRandom=random;
            CudaCurveGrow azimuthGrow; Relay azimuthRelay;
            CHECK(azimuthGrow.BeginFresh(input->Input(1,2),input,angular,stream)==CurveGrowStatus::Ok &&
                  azimuthGrow.FinishFreshAsync(stream,Done,&azimuthRelay)==CurveGrowStatus::Ok &&
                  cudaStreamSynchronize(stream)==cudaSuccess && azimuthRelay.status.load()==int(cudaSuccess) &&
                  azimuthGrow.CommitFreshFinish()==CurveGrowStatus::Ok);
            float const angle=(90.f+random*360.f*(UsdGenDraw01(angular.seed,99,kSaltGrowAzimuth)-.5f))*3.14159265358979323846f/180.f;
            points.assign(2,{}); rest.assign(2,{});
            CHECK(Download(azimuthGrow.view().points,&points,stream) &&
                  Download(azimuthGrow.view().restPoints,&rest,stream) &&
                  Near(points[1],V(1,2+std::cos(angle),3+std::sin(angle))) &&
                  Near(rest[1],V(9,8+std::cos(angle),7+std::sin(angle))));
        }
        CurveGrowControls badAzimuth=c; badAzimuth.azimuth=361.f;
        CHECK(rejected.BeginFresh(input->Input(1,2),input,badAzimuth,stream)==CurveGrowStatus::InvalidArgument);
        badAzimuth.azimuth=0; badAzimuth.azimuthRandom=1.01f;
        CHECK(rejected.BeginFresh(input->Input(1,2),input,badAzimuth,stream)==CurveGrowStatus::InvalidArgument);
        CHECK(rejected.BeginFresh(input->Input(1,2),input,lift,stream)==CurveGrowStatus::InvalidArgument);
        lift.lift=std::numeric_limits<float>::infinity();
        CudaCurveGrow nonFiniteLift;
        CHECK(nonFiniteLift.BeginFresh(input->Input(1,2),input,lift,stream)==CurveGrowStatus::InvalidArgument);
        CurveGrowControls overflow=c; overflow.length=std::numeric_limits<double>::max(); CudaCurveGrow nonfinite; Relay overflowRelay;
        CHECK(nonfinite.BeginFresh(input->Input(1,2),input,overflow,stream)==CurveGrowStatus::Ok&&nonfinite.FinishFreshAsync(stream,Done,&overflowRelay)==CurveGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess&&overflowRelay.status.load()==int(cudaSuccess));
        CHECK(nonfinite.CommitFreshFinish()==CurveGrowStatus::NonFiniteInput&&!nonfinite.HasUnprovenWork());
    }
    CHECK(pool->Snapshot().usedBytes==baseline.usedBytes);

    // Structural rejection is entirely pre-submit; malformed ragged offsets
    // are detected by the device status and leave no committed output.
    {
        auto input=std::make_shared<InputOwner>();
        CHECK(Upload(input->points,{V(0,0,0)},stream)&&Upload(input->rest,{V(0,0,0)},stream)&&Upload(input->widths,{.1f},stream));
        CHECK(Upload(input->offsets,{0u,1u},stream)&&Upload(input->ids,{1ull},stream)&&Upload(input->rootPrim,{0},stream));
        CHECK(Upload(input->rootUV,{make_float2(0,0)},stream)&&Upload(input->rootT,{V(1,0,0)},stream)&&Upload(input->rootB,{V(0,1,0)},stream)&&Upload(input->rootN,{V(0,0,1)},stream));
        CudaCurveGrow invalid; CurveGrowControls c; c.cvCount=1;
        CHECK(invalid.BeginFresh(input->Input(1,1),input,c,stream)==CurveGrowStatus::InvalidArgument);
        c.cvCount=2; Relay relay;
        CHECK(invalid.BeginFresh(input->Input(1,1),input,c,stream)==CurveGrowStatus::Ok);
        CHECK(invalid.FinishFreshAsync(stream,Done,&relay)==CurveGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess&&relay.status.load()==int(cudaSuccess));
        CHECK(invalid.CommitFreshFinish()==CurveGrowStatus::InvalidTopology&&invalid.view().points.data==nullptr&&!invalid.HasUnprovenWork());
    }
    CHECK(pool->Snapshot().usedBytes==baseline.usedBytes);
    CHECK(cudaStreamDestroy(stream)==cudaSuccess);
    return 0;
}
