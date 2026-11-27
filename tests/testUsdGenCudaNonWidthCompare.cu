#include "usdGenCurves/cuda/nonWidthCompare.h"
#include "usdGenCuda/backendId.h"
#include "usdGen/executionResources.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

template <class T> static bool Upload(DeviceBuffer<T>& b, std::vector<T> const& v,
                                      cudaStream_t s) {
    if (b.reset(v.size()) != cudaSuccess) return false;
    return v.empty() || (cudaMemcpyAsync(b.data(), v.data(), v.size()*sizeof(T),
                                         cudaMemcpyHostToDevice, s) == cudaSuccess &&
                         cudaStreamSynchronize(s) == cudaSuccess);
}

template <class T> static std::vector<unsigned char> Bytes(std::vector<T> const& v) {
    std::vector<unsigned char> out(v.size() * sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), v.data(), out.size());
    return out;
}

struct Owner {
    DeviceBuffer<float3> points, rest, tangent, binormal, normal;
    DeviceBuffer<float> widths, hairT;
    DeviceBuffer<uint32_t> offsets;
    DeviceBuffer<uint64_t> ids;
    DeviceBuffer<int32_t> prim;
    DeviceBuffer<float2> uv;
    DeviceBuffer<unsigned char> namedFloat, namedInt;
    DeviceView<const float3> view(DeviceBuffer<float3> const& b) const { return b.view(); }
    CurveFullNonWidthInput input(size_t curves, size_t pointsCount) const {
        CurveFullNonWidthInput x;
        x.geometry = {points.view(), rest.view(), widths.view(), offsets.view(), ids.view(), curves, pointsCount};
        x.hairT = hairT.view(); x.rootPrim = prim.view(); x.rootUV = uv.view();
        x.frames = {tangent.view(), binormal.view(), normal.view(), {}};
        x.chunks = {{0, curves, curves, 0, pointsCount, 0, 0}};
        if (!namedFloat.size() && !namedInt.size()) return x;
        UsdGenDeviceChannelMetadata fm; fm.name=TfToken("testFloat"); fm.type=UsdGenDeviceValueType::Float32;
        fm.domain=UsdGenDeviceDomain::Point; fm.elementCount=pointsCount; fm.arity=1;
        fm.strideBytes=sizeof(float); x.named.push_back({fm,namedFloat.view(),nullptr});
        UsdGenDeviceChannelMetadata im; im.name=TfToken("testInt"); im.type=UsdGenDeviceValueType::Int32;
        im.domain=UsdGenDeviceDomain::Point; im.elementCount=pointsCount; im.arity=1;
        im.strideBytes=sizeof(int32_t); x.named.push_back({im,namedInt.view(),nullptr});
        return x;
    }
};

static bool Compare(CurveFullNonWidthInput const& a, CurveFullNonWidthInput const& b,
                    std::shared_ptr<const void> lifetime, cudaStream_t stream, bool want,
                    UsdGenExecutionMemoryReservation* reservation = nullptr) {
    CudaCurveFullNonWidthCompare proof;
    cudaError_t e = proof.BeginFresh(a, b, std::move(lifetime), stream, reservation);
    if (e != cudaSuccess) return false;
    if (proof.EnqueueFreshStatus(stream) != cudaSuccess) return false;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
    bool equal = false;
    if (proof.CommitFreshFinish(&equal) != cudaSuccess) return false;
    return equal == want && !proof.HasUnprovenWork();
}

int main() {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count) == cudaSuccess);
    if (!count) return 77;
    int device = -1; CHECK(cudaGetDevice(&device) == cudaSuccess);
    CHECK(ConfigureCudaExecutionResources(device, {size_t{8} << 20, 0}));
    cudaStream_t stream = nullptr; CHECK(cudaStreamCreate(&stream) == cudaSuccess);

    auto a = std::make_shared<Owner>();
    auto b = std::make_shared<Owner>();
    CHECK(Upload(a->points, {{0,0,0},{1,0,0},{2,0,0}}, stream));
    CHECK(Upload(b->points, {{0,0,0},{1,0,0},{2,0,0}}, stream));
    CHECK(Upload(a->rest, {{0,0,0},{1,0,0},{2,0,0}}, stream));
    CHECK(Upload(b->rest, {{0,0,0},{1,0,0},{2,0,0}}, stream));
    CHECK(Upload(a->widths, {1.f,2.f,3.f}, stream) && Upload(b->widths, {9.f,8.f,7.f}, stream));
    CHECK(Upload(a->hairT, {0.f,.5f,1.f}, stream) && Upload(b->hairT, {0.f,.5f,1.f}, stream));
    CHECK(Upload(a->offsets, {0u,3u}, stream) && Upload(b->offsets, {0u,3u}, stream));
    CHECK(Upload(a->ids, {17ull}, stream) && Upload(b->ids, {17ull}, stream));
    CHECK(Upload(a->prim, {2}, stream) && Upload(b->prim, {2}, stream));
    CHECK(Upload(a->uv, {make_float2(.2f,.3f)}, stream) && Upload(b->uv, {make_float2(.2f,.3f)}, stream));
    CHECK(Upload(a->tangent, {{1,0,0}}, stream) && Upload(b->tangent, {{1,0,0}}, stream));
    CHECK(Upload(a->binormal, {{0,1,0}}, stream) && Upload(b->binormal, {{0,1,0}}, stream));
    CHECK(Upload(a->normal, {{0,0,1}}, stream) && Upload(b->normal, {{0,0,1}}, stream));
    CHECK(Upload(a->namedFloat, Bytes(std::vector<float>{0.f,1.f,2.f}), stream) &&
          Upload(b->namedFloat, Bytes(std::vector<float>{0.f,1.f,2.f}), stream));
    CHECK(Upload(a->namedInt, Bytes(std::vector<int32_t>{1,2,3}), stream) &&
          Upload(b->namedInt, Bytes(std::vector<int32_t>{1,2,3}), stream));
    auto lifetime = std::make_shared<int>(1);
    auto base = a->input(1,3); auto same = b->input(1,3);
    CHECK(Compare(base, same, lifetime, stream, true));
    CHECK(Upload(b->namedFloat, Bytes(std::vector<float>{0.f,1.f,2.1f}), stream));
    CHECK(Compare(base, b->input(1,3), std::make_shared<int>(10), stream, false));
    CHECK(Upload(b->namedFloat, Bytes(std::vector<float>{0.f,1.f,2.f}), stream));
    CHECK(Upload(b->namedInt, Bytes(std::vector<int32_t>{1,2,4}), stream));
    CHECK(Compare(base, b->input(1,3), std::make_shared<int>(11), stream, false));
    CHECK(Upload(b->namedInt, Bytes(std::vector<int32_t>{1,2,3}), stream));

    // Width storage exists in the geometry view but is excluded from proof.
    CHECK(Compare(base, b->input(1,3), lifetime, stream, true));
    auto noWidths = b->input(1,3); noWidths.geometry.widths = {};
    CHECK(Compare(base, noWidths, lifetime, stream, true));
    CHECK(Upload(b->points, {{0,0,0},{1,0,0},{2,1,0}}, stream));
    CHECK(Compare(base, b->input(1,3), std::make_shared<int>(3), stream, false));
    CHECK(Upload(b->points, {{0,0,0},{1,0,0},{2,0,0}}, stream));
    CHECK(Upload(b->ids, {18ull}, stream));
    CHECK(Compare(base, b->input(1,3), std::make_shared<int>(4), stream, false));
    CHECK(Upload(b->ids, {17ull}, stream));
    CHECK(Upload(b->hairT, {0.f, .25f, 1.f}, stream));
    CHECK(Compare(base, b->input(1,3), std::make_shared<int>(5), stream, false));

    CHECK(Upload(b->hairT, {0.f, .5f, 1.f}, stream));
    CHECK(Upload(b->offsets, {0u, 2u}, stream));
    {
        CudaCurveFullNonWidthCompare malformedOffsets;
        CHECK(malformedOffsets.BeginFresh(base,b->input(1,3),lifetime,stream)==cudaSuccess &&
              malformedOffsets.EnqueueFreshStatus(stream)==cudaSuccess &&
              cudaStreamSynchronize(stream)==cudaSuccess);
        bool unchanged=true;
        CHECK(malformedOffsets.CommitFreshFinish(&unchanged)==cudaErrorInvalidValue &&
              unchanged && !malformedOffsets.HasUnprovenWork());
    }
    CHECK(Upload(b->offsets, {0u, 3u}, stream));
    CHECK(Upload(b->normal, {{0,1,0}}, stream));
    CHECK(Compare(base, b->input(1,3), std::make_shared<int>(13), stream, false));
    CHECK(Upload(b->normal, {{0,0,1}}, stream));
    CHECK(Upload(b->rest, {{0,0,0},{1,1,0},{2,0,0}},stream));
    CHECK(Compare(base,b->input(1,3),lifetime,stream,false));
    CHECK(Upload(b->rest, {{0,0,0},{1,0,0},{2,0,0}},stream));
    CHECK(Upload(b->prim,{3},stream));
    CHECK(Compare(base,b->input(1,3),lifetime,stream,false));
    CHECK(Upload(b->prim,{2},stream));
    CHECK(Upload(b->uv,{make_float2(.25f,.3f)},stream));
    CHECK(Compare(base,b->input(1,3),lifetime,stream,false));
    CHECK(Upload(b->uv,{make_float2(.2f,.3f)},stream));
    CHECK(Upload(b->tangent,{{0,1,0}},stream));
    CHECK(Compare(base,b->input(1,3),lifetime,stream,false));
    CHECK(Upload(b->tangent,{{1,0,0}},stream));
    CHECK(Upload(b->binormal,{{1,0,0}},stream));
    CHECK(Compare(base,b->input(1,3),lifetime,stream,false));
    CHECK(Upload(b->binormal,{{0,1,0}},stream));
    CHECK(Upload(b->hairT,{0.f,.25f,1.f},stream));
    CHECK(Compare(base,b->input(1,3),lifetime,stream,false));
    CHECK(Upload(b->hairT,{0.f,.5f,1.f},stream));
    auto absentRest=b->input(1,3); absentRest.geometry.restPoints={};
    CHECK(Compare(base,absentRest,lifetime,stream,false));
    auto absentFrames=b->input(1,3); absentFrames.frames={};
    CHECK(Compare(base,absentFrames,lifetime,stream,false));
    {
        auto aligned=b->input(1,3);aligned.frames.stableIds=aligned.geometry.stableIds;
        CHECK(Compare(base,aligned,lifetime,stream,true));
        DeviceBuffer<uint64_t> wrongFrameIds;CHECK(Upload(wrongFrameIds,{18ull},stream));
        aligned.frames.stableIds={wrongFrameIds.data(),wrongFrameIds.size()};
        // A descriptor mismatch must not hide a malformed operand's frame map.
        aligned.chunks[0].tile=1;
        CudaCurveFullNonWidthCompare proof;
        CHECK(proof.BeginFresh(base,aligned,lifetime,stream)==cudaSuccess &&
              proof.EnqueueFreshStatus(stream)==cudaSuccess && cudaStreamSynchronize(stream)==cudaSuccess);
        bool unchanged=true;
        CHECK(proof.CommitFreshFinish(&unchanged)==cudaErrorInvalidValue && unchanged);
    }
    {
        auto malformed=b->input(1,3);malformed.frames.binormal={};
        CudaCurveFullNonWidthCompare rejected;
        CHECK(rejected.BeginFresh(base,malformed,lifetime,stream)==cudaErrorInvalidValue);
    }
    {
        auto malformed=b->input(1,3);malformed.named[0].metadata.strideBytes=3;
        CudaCurveFullNonWidthCompare rejected;
        CHECK(rejected.BeginFresh(base,malformed,lifetime,stream)==cudaErrorInvalidValue);
    }
    {
        auto malformed=b->input(1,3);++malformed.named[0].bytes.data;
        CudaCurveFullNonWidthCompare rejected;
        CHECK(rejected.BeginFresh(base,malformed,lifetime,stream)==cudaErrorInvalidValue);
    }
    {
        DeviceBuffer<uint32_t> shortOffsets;
        CHECK(Upload(shortOffsets,{0u,2u},stream));
        auto shorter=b->input(1,3);
        shorter.geometry.pointCount=2;shorter.geometry.points.size=2;
        shorter.geometry.restPoints.size=2;shorter.hairT.size=2;
        shorter.geometry.curveOffsets={shortOffsets.data(),shortOffsets.size()};shorter.chunks[0].cvCount=2;
        for(auto& plane:shorter.named){plane.metadata.elementCount=2;plane.bytes.size=8;}
        CHECK(Compare(base,shorter,lifetime,stream,false));
    }
    for (int field=0;field<3;++field) {
        auto changed=b->input(1,3);
        if(field==0) changed.chunks[0].tile=1;
        if(field==1) changed.chunks[0].surface=1;
        if(field==2) changed.chunks[0].liveCount=0;
        CHECK(Compare(base,changed,lifetime,stream,false));
    }
    auto renamed=b->input(1,3); renamed.named[0].metadata.name=TfToken("other");
    CHECK(Compare(base,renamed,lifetime,stream,false));
    auto reordered=b->input(1,3); std::swap(reordered.named[0],reordered.named[1]);
    CHECK(Compare(base,reordered,lifetime,stream,false));
    // Primitive and Groom named domains carry their own cardinalities.
    for (auto domain : {UsdGenDeviceDomain::Primitive,UsdGenDeviceDomain::Groom}) {
        auto left=a->input(1,3), right=b->input(1,3);
        for(auto* input : {&left,&right}) for(auto& plane:input->named) {
            plane.metadata.domain=domain;plane.metadata.elementCount=1;
            plane.bytes.size=sizeof(float);
        }
        CHECK(Compare(left,right,lifetime,stream,true));
        right.named[0].metadata.name=TfToken("domainMismatch");
        CHECK(Compare(left,right,lifetime,stream,false));
    }

    // Signed zero compares numerically equal; NaN is unequal.
    CHECK(Upload(a->hairT, {0.f, -0.f, 1.f}, stream));
    base = a->input(1,3);
    CHECK(Upload(b->hairT, {0.f, 0.f, 1.f}, stream));
    CHECK(Compare(base, b->input(1,3), std::make_shared<int>(6), stream, true));
    CHECK(Upload(b->hairT, {0.f, std::nanf(""), 1.f}, stream));
    CHECK(Compare(base, b->input(1,3), std::make_shared<int>(7), stream, false));
    CHECK(Upload(b->hairT,{0.f,0.f,1.f},stream));
    CHECK(Upload(a->namedFloat,Bytes(std::vector<float>{-0.f,1.f,2.f}),stream) &&
          Upload(b->namedFloat,Bytes(std::vector<float>{0.f,1.f,2.f}),stream));
    CHECK(Compare(a->input(1,3),b->input(1,3),lifetime,stream,true));
    CHECK(Upload(a->namedFloat,Bytes(std::vector<float>{std::nanf(""),1.f,2.f}),stream) &&
          Upload(b->namedFloat,Bytes(std::vector<float>{std::nanf(""),1.f,2.f}),stream));
    CHECK(Compare(a->input(1,3),b->input(1,3),lifetime,stream,false));

    // Canonical empty views are valid; a malformed non-empty view is rejected.
    Owner emptyA, emptyB;
    CHECK(Upload(emptyA.offsets, {0u}, stream) && Upload(emptyB.offsets, {0u}, stream));
    CHECK(Compare(emptyA.input(0,0), emptyB.input(0,0), std::make_shared<int>(8), stream, true));
    auto malformed = emptyB.input(0,0); malformed.geometry.curveOffsets = {nullptr, 1};
    CudaCurveFullNonWidthCompare invalid;
    CHECK(invalid.BeginFresh(emptyA.input(0,0), malformed, std::make_shared<int>(9), stream) == cudaErrorInvalidValue);
    CHECK(Compare(a->input(1,3),emptyA.input(0,0),lifetime,stream,false));
    {
        auto token=std::make_shared<int>(42);std::weak_ptr<int> weak=token;
        CudaCurveFullNonWidthCompare proof;
        CHECK(proof.BeginFresh(emptyA.input(0,0),emptyB.input(0,0),token,stream)==cudaSuccess);
        token.reset();CHECK(!weak.expired() && proof.ExclusiveRetainedBytes()>0);
        CHECK(proof.EnqueueFreshStatus(stream)==cudaSuccess && cudaStreamSynchronize(stream)==cudaSuccess);
        bool equal=false;
        CHECK(proof.CommitFreshFinish(&equal)==cudaSuccess && equal && weak.expired());
    }

    // Admission rejection must not submit work or perturb the full ledger.
    auto pool=FindUsdGenExecutionResourcePool({usdGen::UsdGenCudaBackendId(),device});
    CHECK(pool);
    // The comparison may run on a Width branch stream while scalar proof is
    // returned on another stream. EnqueueFreshStatus must order its copy after
    // the comparison event, without a host wait on the producer stream.
    {
        cudaStream_t reply = nullptr;
        CHECK(cudaStreamCreateWithFlags(&reply,cudaStreamNonBlocking)==cudaSuccess);
        auto before=pool->Snapshot();
        auto retained=std::make_shared<std::vector<std::shared_ptr<Owner>>>(
            std::initializer_list<std::shared_ptr<Owner>>{a,b});
        std::weak_ptr<std::vector<std::shared_ptr<Owner>>> weak=retained;
        {
            auto reservation=pool->TryReserveMemory(2*sizeof(CurveFullNonWidthCompareResult));
            CHECK(reservation);
            CudaCurveFullNonWidthCompare proof;
            auto left=a->input(1,3), right=b->input(1,3);
            // Both current named float packets contain NaN. This is a valid
            // unequal result, not malformed input or a lost completion proof.
            CHECK(proof.BeginFresh(left,right,retained,stream,&*reservation)==cudaSuccess);
            retained.reset(); CHECK(!weak.expired());
            bool unchanged=true;
            CHECK(proof.CommitFreshFinish(&unchanged)==cudaErrorInvalidValue && unchanged);
            CHECK(proof.EnqueueFreshStatus(reply)==cudaSuccess);
            CHECK(cudaStreamSynchronize(reply)==cudaSuccess);
            CHECK(proof.CommitFreshFinish(&unchanged)==cudaSuccess && !unchanged);
            CHECK(!proof.HasUnprovenWork() && weak.expired());
        }
        CHECK(pool->Snapshot().usedBytes==before.usedBytes);
        CHECK(pool->Snapshot().byKind==before.byKind);
        CHECK(cudaStreamDestroy(reply)==cudaSuccess);
    }
    {
        auto before=pool->Snapshot();
        auto filler=pool->TryReserve(before.usableBytes-before.usedBytes,UsdGenExecutionResourceKind::Active);
        CHECK(filler);
        auto full=pool->Snapshot();
        CudaCurveFullNonWidthCompare rejected;
        CHECK(rejected.BeginFresh(emptyA.input(0,0),emptyB.input(0,0),lifetime,stream)==cudaErrorMemoryAllocation);
        CHECK(!rejected.HasUnprovenWork());
        auto after=pool->Snapshot();
        CHECK(full.usedBytes==after.usedBytes && full.byKind==after.byKind);
    }
    // Simulate a parent losing completion proof: the borrowed native owner
    // and comparator charges must survive candidate destruction, even after
    // an unrelated later stream fence succeeds. Quarantine is process-long.
    {
        auto retained=std::make_shared<Owner>();
        CHECK(Upload(retained->offsets,{0u},stream));
        std::weak_ptr<Owner> weak=retained;
        auto before=pool->Snapshot();
        size_t retainedProofBytes=0;
        {
            CudaCurveFullNonWidthCompare proof;
            auto input=retained->input(0,0);
            CHECK(proof.BeginFresh(input,input,retained,stream)==cudaSuccess);
            retainedProofBytes=proof.ExclusiveRetainedBytes();
            CHECK(retainedProofBytes>0);
            retained.reset();
            proof.Quarantine();
            CHECK(proof.HasUnprovenWork() && !weak.expired());
        }
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess);
        CHECK(!weak.expired());
        CHECK(pool->Snapshot().usedBytes==before.usedBytes+retainedProofBytes);
    }
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    return 0;
}
