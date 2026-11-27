#include "utils/pxrInterop.h"
#include "usdGenCurves/cuda/cudaExecution.h"
#include "cudaPlanRunner.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "cudaExecutionQueueFixture.h"
#include "usdGenCurves/cuda/generation.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

using namespace usdGen;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (false)

namespace {
struct Packet {
    std::vector<float3> points, rest, t, b, n;
    std::vector<float> widths, hairT;
    std::vector<uint32_t> offsets;
    std::vector<uint64_t> ids;
    std::vector<int> prim;
    std::vector<float2> uv;
    bool deformed = false;
};
template<class T> bool Read(gpu::DeviceView<const T> view, std::vector<T>* out, cudaStream_t stream) {
    out->resize(view.size);
    return (!view.size || cudaMemcpyAsync(out->data(), view.data, view.size * sizeof(T),
        cudaMemcpyDeviceToHost, stream) == cudaSuccess) && cudaStreamSynchronize(stream) == cudaSuccess;
}
bool Capture(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
             Packet* out, cudaStream_t stream) {
    auto lease = gpu::AcquireGeometry(generation, stream);
    if (!lease) return false;
    out->deformed = generation->Geometry().alreadyDeformed;
    return Read(lease.Geometry().points, &out->points, stream) &&
        Read(lease.Geometry().restPoints, &out->rest, stream) &&
        Read(lease.Geometry().widths, &out->widths, stream) &&
        Read(lease.Geometry().curveOffsets, &out->offsets, stream) &&
        Read(lease.Geometry().stableIds, &out->ids, stream) &&
        Read(lease.HairT(), &out->hairT, stream) && Read(lease.RootPrim(), &out->prim, stream) &&
        Read(lease.RootUV(), &out->uv, stream) && Read(lease.RootT(), &out->t, stream) &&
        Read(lease.RootB(), &out->b, stream) && Read(lease.RootN(), &out->n, stream);
}
bool Near(float a, float b) { return std::fabs(a - b) <= 3e-5f; }
bool Near(float3 a, float3 b) { return Near(a.x,b.x) && Near(a.y,b.y) && Near(a.z,b.z); }
bool SameFrameBytes(Packet const& a, Packet const& b) {
    auto same=[](auto const& x, auto const& y) {
        return x.size()==y.size() && (x.empty() || std::memcmp(x.data(),y.data(),x.size()*sizeof(x[0]))==0);
    };
    return same(a.t,b.t) && same(a.b,b.b) && same(a.n,b.n);
}
bool Same(Packet const& a, Packet const& b) {
    if (a.ids.size() != b.ids.size() || a.points.size() != b.points.size() ||
        a.offsets.size() != a.ids.size()+1 || b.offsets.size() != b.ids.size()+1 ||
        a.rest.size() != a.points.size() || b.rest.size() != b.points.size() ||
        a.widths.size() != a.points.size() || b.widths.size() != b.points.size() ||
        a.hairT.size() != a.points.size() || b.hairT.size() != b.points.size() ||
        a.prim.size() != a.ids.size() || b.prim.size() != b.ids.size() ||
        a.uv.size() != a.ids.size() || b.uv.size() != b.ids.size() || a.deformed != b.deformed) return false;
    std::map<uint64_t,size_t> byId;
    for (size_t i=0;i<b.ids.size();++i) if (!byId.emplace(b.ids[i],i).second) return false;
    for (size_t i=0;i<a.ids.size();++i) {
        auto found=byId.find(a.ids[i]); if(found==byId.end()) return false;
        size_t j=found->second, count=a.offsets[i+1]-a.offsets[i];
        if(count!=b.offsets[j+1]-b.offsets[j] || a.prim[i]!=b.prim[j] ||
           !Near(a.uv[i].x,b.uv[j].x) || !Near(a.uv[i].y,b.uv[j].y)) return false;
        for(size_t k=0;k<count;++k) {
            size_t ai=a.offsets[i]+k, bi=b.offsets[j]+k;
            if(!Near(a.points[ai],b.points[bi]) || !Near(a.rest[ai],b.rest[bi]) ||
               !Near(a.widths[ai],b.widths[bi]) || !Near(a.hairT[ai],b.hairT[bi])) {
                std::fprintf(stderr,"mismatch id=%llu cv=%zu points=(%g,%g,%g)/(%g,%g,%g) rest=(%g,%g,%g)/(%g,%g,%g) width=%g/%g hairT=%g/%g\n",
                    static_cast<unsigned long long>(a.ids[i]),k,a.points[ai].x,a.points[ai].y,a.points[ai].z,
                    b.points[bi].x,b.points[bi].y,b.points[bi].z,a.rest[ai].x,a.rest[ai].y,a.rest[ai].z,
                    b.rest[bi].x,b.rest[bi].y,b.rest[bi].z,a.widths[ai],b.widths[bi],a.hairT[ai],b.hairT[bi]);
                return false;
            }
        }
    }
    return true;
}
UsdGenNodeDesc Node(char const* path, char const* type, SdfPath input) {
    UsdGenNodeDesc n; n.path=SdfPath(path); n.type=TfToken(type); n.inputs={input}; return n;
}
UsdGenGraphDesc Source() {
    UsdGenGraphDesc d; d.description=SdfPath("/GeneratedDag");
    d.executionBackend=usdGenTestUtils::Backend("cuda"); d.defaultWidth=.025f;
    UsdGenSurfaceDesc s; s.path=SdfPath("/Scalp");
    s.restPoints={{0,0,0},{1,0,0},{1,1,0},{0,1,0}}; s.points=s.restPoints;
    s.faceVertexCounts={4}; s.faceVertexIndices={0,1,2,3}; s.uv={{0,0},{1,0},{1,1},{0,1}};
    d.surfaces={s};
    UsdGenNodeDesc scatter; scatter.path=SdfPath("/Ops/Scatter"); scatter.type=TfToken("UsdGenScatter");
    scatter.seed=41; scatter.surfaces={s.path}; scatter.params={{TfToken("density"),VtValue(80.f),false}};
    auto grow=Node("/Ops/Grow","UsdGenGrow",scatter.path); grow.seed=19;
    grow.params={{TfToken("segments"),VtValue(5),false},{TfToken("length"),VtValue(2.f),false},
                 {TfToken("lengthRandom"),VtValue(GfVec2f(.5f,1.5f)),false}};
    d.nodes={grow,scatter}; d.terminal=grow.path; return d;
}
// Test-only native readback supplies an independent ordinary C3 CUDA oracle.
// Production source/operator execution never takes this host geometry route.
UsdGenGraphDesc AsC3(UsdGenGraphDesc d, Packet const& p) {
    UsdGenCurveSetDesc c; c.path=SdfPath("/Captured"); c.role=UsdGenLane::Chunked; c.curveRole=TfToken("hair");
    for(size_t i=0;i<p.ids.size();++i) {
        c.curveId.push_back(p.ids[i]); c.curveVertexCounts.push_back(int(p.offsets[i+1]-p.offsets[i]));
        c.skinPrim.push_back(p.prim[i]); c.skinPrimUv.push_back(GfVec2f(p.uv[i].x,p.uv[i].y));
        GfMatrix4d frame(1.0);
        // Native float axes need double re-orthonormalization to satisfy the
        // authored frame's stricter 1e-8 contract, not a production math change.
        GfVec3d t(p.t[i].x,p.t[i].y,p.t[i].z), n(p.n[i].x,p.n[i].y,p.n[i].z);
        n.Normalize(); t -= GfDot(t,n)*n; t.Normalize();
        GfVec3d axes[3]={t,GfCross(n,t),n};
        for(int r=0;r<3;++r) for(int col=0;col<3;++col) frame[r][col]=axes[r][col];
        auto root=p.rest[p.offsets[i]]; frame[3][0]=root.x; frame[3][1]=root.y; frame[3][2]=root.z;
        c.rootFrame.push_back(frame);
    }
    for(auto v:p.points) c.points.push_back(GfVec3f(v.x,v.y,v.z));
    for(auto v:p.rest) c.rest.push_back(GfVec3f(v.x,v.y,v.z));
    for(float v:p.widths) c.widths.push_back(v);
    UsdGenNodeDesc source; source.path=SdfPath("/Ops/Grow"); source.type=TfToken("UsdGenCurveSource");
    source.curves={c.path}; source.surfaces={d.surfaces[0].path};
    source.params={{TfToken("useRest"),VtValue(!p.deformed),false},{TfToken("rebind"),VtValue(TfToken("never")),false}};
    d.curveSets={c}; d.nodes={source}; d.terminal=source.path; return d;
}
UsdGenGraphDesc Descendants(UsdGenGraphDesc d, int variant) {
    auto root=d.terminal;
    auto length=Node("/Ops/Length","UsdGenLength",root);
    length.params={{TfToken("length:mode"),VtValue(TfToken(variant==2||variant==3?"cull":"scale")),false},
                   {TfToken("length:value"),VtValue(.6f),false}};
    if(variant==2||variant==3) length.params.push_back({TfToken("cullThreshold"),VtValue(2.f),false});
    auto noise=Node("/Ops/Noise","UsdGenNoise",variant==1?root:length.path);
    noise.params={{TfToken("noise:frequency"),VtValue(1.f),false},{TfToken("noise:magnitude"),VtValue(.1f),false},
                  {TfToken("noise:seed"),VtValue(7),false},{TfToken("noise:octaves"),VtValue(2),false}};
    if(variant!=1) d.nodes.push_back(length);
    if(variant==1||variant==3||variant==4) d.nodes.push_back(noise);
    auto input=(variant==1||variant==3||variant==4)?noise.path:length.path;
    auto width=Node("/Ops/Width","UsdGenWidth",input); width.params={{TfToken("width"),VtValue(.2f),false}};
    d.nodes.push_back(width); d.terminal=width.path;
    if(variant==4) d.terminal=root;
    if(variant==5||variant==6) {
        auto right=length; right.path=SdfPath("/Ops/RightLength");
        if(variant==6) right.params[1].value=VtValue(.7f);
        auto rw=width; rw.path=SdfPath("/Ops/RightWidth"); rw.inputs={right.path}; rw.params[0].value=VtValue(.8f);
        auto blend=Node("/Ops/Blend","UsdGenWidthBlend",width.path); blend.inputs.push_back(rw.path);
        blend.params.push_back({TfToken("widthBlend:weight"), VtValue(.25f), false});
        d.nodes.push_back(right); d.nodes.push_back(rw); d.nodes.push_back(blend); d.terminal=blend.path;
    }
    std::reverse(d.nodes.begin(),d.nodes.end()); return d;
}
}

static int RbfCases(cudaStream_t stream) {
    auto sourceDesc=Source();
    auto& surface=sourceDesc.surfaces.front();
    surface.restPoints.push_back(GfVec3f(0,0,1));
    surface.points=surface.restPoints;
    for(auto& p:surface.points) p[0]+=.25f;
    surface.faceVertexCounts.push_back(3);
    surface.faceVertexIndices.insert(surface.faceVertexIndices.end(),{0,1,4});
    surface.uv.push_back(GfVec2f(0,1));
    UsdGenDiagnostics diagnostics;
    auto sourcePlan=CompileCudaGraph(usdGenUsd::Lower(sourceDesc),&diagnostics); CHECK(sourcePlan);
    auto sourceWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(sourceWorkspace);
    auto source=ExecuteCudaPlanForTesting(sourcePlan,*sourceWorkspace,1.,1,&diagnostics); CHECK(source);
    Packet original; CHECK(Capture(source,&original,stream));
    CHECK(!original.ids.empty() && !std::is_sorted(original.ids.begin(),original.ids.end()));
    auto compose=[](UsdGenGraphDesc d,int variant) {
        auto const grow=d.terminal;
        auto input=grow;
        if(variant==1 || variant==2 || variant==3) {
            auto predecessor=Node("/Ops/Predecessor",variant==3?"UsdGenNoise":"UsdGenLength",grow);
            if(variant==3) predecessor.params={{TfToken("noise:magnitude"),VtValue(.1f),false}};
            else predecessor.params={{TfToken("length:mode"),VtValue(TfToken(variant==2?"cull":"scale")),false},
                {TfToken("length:value"),VtValue(.6f),false}};
            if(variant==2) predecessor.params.push_back({TfToken("cullThreshold"),VtValue(2.f),false});
            input=predecessor.path; d.nodes.push_back(predecessor);
        }
        auto deform=Node("/Ops/Deform","UsdGenDeform",input);
        deform.surfaces={d.surfaces.front().path};
        deform.params={{TfToken("rbfSamples"),VtValue(5),false}};
        d.nodes.push_back(deform);
        auto width=Node("/Ops/Width","UsdGenWidth",deform.path);
        width.params={{TfToken("width"),VtValue(.2f),false}};
        d.nodes.push_back(width); d.terminal=variant==4?grow:width.path;
        if(variant>=5) {
            auto right=deform; right.path=SdfPath("/Ops/RightDeform");
            if(variant==6) {
                auto other=d.surfaces.front(); other.path=SdfPath("/OtherScalp");
                for(auto& p:other.points) p[0]+=.25f;
                d.surfaces.push_back(other); right.surfaces={other.path};
            }
            if(variant==7) right.inputs={deform.path};
            auto rw=width; rw.path=SdfPath("/Ops/RightWidth"); rw.inputs={right.path}; rw.params[0].value=VtValue(.8f);
            auto blend=Node("/Ops/Blend","UsdGenWidthBlend",width.path); blend.inputs.push_back(rw.path);
        blend.params.push_back({TfToken("widthBlend:weight"), VtValue(.25f), false});
            d.nodes.push_back(right); d.nodes.push_back(rw); d.nodes.push_back(blend); d.terminal=blend.path;
        }
        std::reverse(d.nodes.begin(),d.nodes.end()); return d;
    };
    auto sameStats=[](auto const& a,auto const& b) {
        if(a.size()!=b.size()) return false;
        for(size_t i=0;i<a.size();++i) if(a[i].key!=b[i].key || a[i].identity!=b[i].identity ||
            a[i].bindCount!=b[i].bindCount || a[i].solveCount!=b[i].solveCount || a[i].sampleCount!=b[i].sampleCount) return false;
        return true;
    };
    for(int variant=0;variant<8;++variant) {
        std::fprintf(stderr,"ScatterGrow RBF variant %d\n",variant); diagnostics={};
        auto desc=compose(sourceDesc,variant); auto plan=CompileCudaGraph(usdGenUsd::Lower(desc),&diagnostics);
        if(variant==7) { CHECK(!plan && diagnostics.HasErrors()); continue; }
        for(auto const& e:diagnostics.errors) std::fprintf(stderr,"%s\n",e.c_str());
        CHECK(plan && !diagnostics.HasErrors());
        auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(workspace);
        auto actual=ExecuteCudaPlanForTesting(plan,*workspace,1.,10+variant,&diagnostics);
        if(variant==6) {
            CHECK(!actual && diagnostics.HasErrors()); diagnostics={};
            UsdGenExecutionRuntime runtime(4); UsdGenCudaExecutionQueue queue(runtime,-1,&diagnostics);
            CHECK(queue.Valid() && queue.Submit(sourcePlan,1.)); queue.Drain(); auto good=queue.Snapshot();
            CHECK(good && good->generation);
            std::atomic<unsigned> callbacks{0}; std::atomic<bool> failed{false};
            CHECK(queue.Submit(plan,2.,[&](uint64_t,auto outcome,auto const&) {
                ++callbacks; failed=outcome==UsdGenExecutionPipeline::Outcome::Failed;
            }));
            queue.Drain(); CHECK(callbacks==1 && failed && queue.Snapshot()==good);
            Packet retained; CHECK(Capture(good->generation,&retained,stream) && Same(original,retained));
            CHECK(queue.Submit(sourcePlan,3.)); queue.Drain();
            CHECK(queue.Snapshot()!=good && queue.Snapshot()->generation &&
                Capture(queue.Snapshot()->generation,&retained,stream) && Same(original,retained));
            continue;
        }
        for(auto const& e:diagnostics.errors) std::fprintf(stderr,"%s\n",e.c_str());
        CHECK(actual && !diagnostics.HasErrors());
        auto oraclePlan=CompileCudaGraph(usdGenUsd::Lower(compose(AsC3(sourceDesc,original),variant)),&diagnostics); CHECK(oraclePlan);
        auto oracleWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(oracleWorkspace);
        auto expected=ExecuteCudaPlanForTesting(oraclePlan,*oracleWorkspace,1.,100,&diagnostics); CHECK(expected);
        Packet a,b,old; CHECK(Capture(actual,&a,stream) && Capture(expected,&b,stream) && Same(a,b));
        if(variant==2) CHECK(!a.ids.empty() && a.ids.size()<original.ids.size());
        else CHECK(a.ids.size()==original.ids.size());
        auto stats=GetCudaBindingStats(*workspace); CHECK(stats.size()==(variant==5?2u:1u));
        for(auto const& item:stats) {
            std::fprintf(stderr,"RBF accepted stats identity=%llu bind=%llu solve=%llu samples=%zu\n",
                static_cast<unsigned long long>(item.identity),static_cast<unsigned long long>(item.bindCount),
                static_cast<unsigned long long>(item.solveCount),item.sampleCount);
            CHECK(item.identity!=0 && item.bindCount==1 && item.solveCount==1 && item.sampleCount==5);
        }
        auto const fieldAccepted=cudaRbfFieldAcceptAttemptCountForTesting();
        auto const fieldRolledBack=cudaRbfFieldRollbackAttemptCountForTesting();
        auto const surfaceAccepted=cudaRbfSurfaceAcceptAttemptCountForTesting();
        auto const surfaceRolledBack=cudaRbfSurfaceRollbackAttemptCountForTesting();
        failNextCudaRbfFinalizationForTesting(); diagnostics={};
        CHECK(!ExecuteCudaPlanForTesting(plan,*workspace,2.,200,&diagnostics,actual) && diagnostics.HasErrors());
        CHECK(sameStats(stats,GetCudaBindingStats(*workspace)) && Capture(actual,&b,stream) && Same(a,b));
        CHECK(cudaRbfFieldAcceptAttemptCountForTesting()==fieldAccepted &&
            cudaRbfSurfaceAcceptAttemptCountForTesting()==surfaceAccepted &&
            cudaRbfFieldRollbackAttemptCountForTesting()==fieldRolledBack+stats.size() &&
            cudaRbfSurfaceRollbackAttemptCountForTesting()==surfaceRolledBack+stats.size());
        diagnostics={}; auto retry=ExecuteCudaPlanForTesting(plan,*workspace,3.,201,&diagnostics,actual);
        CHECK(retry && Capture(retry,&b,stream) && Same(a,b));
        auto after=GetCudaBindingStats(*workspace); CHECK(after.size()==stats.size());
        for(size_t i=0;i<after.size();++i) CHECK(after[i].identity==stats[i].identity &&
            after[i].bindCount==stats[i].bindCount && after[i].solveCount==stats[i].solveCount+1);
        UsdGenExecutionRuntime runtime(4); UsdGenCudaExecutionQueue queue(runtime,-1,&diagnostics);
        CHECK(queue.Valid() && queue.Submit(plan,1.)); queue.Drain(); auto snapshot=queue.Snapshot();
        CHECK(snapshot && snapshot->generation && Capture(snapshot->generation,&b,stream) && Same(a,b));
        failNextCudaFinalizationRelayAllocationForTesting();
        std::atomic<unsigned> callbacks{0}; std::atomic<bool> failed{false};
        CHECK(queue.Submit(plan,2.,[&](uint64_t,auto outcome,auto const&) {
            ++callbacks; failed=outcome==UsdGenExecutionPipeline::Outcome::Failed;
        }));
        queue.Drain(); CHECK(callbacks==1 && failed && queue.Snapshot()==snapshot);
        CHECK(queue.Submit(plan,3.)); queue.Drain();
        CHECK(queue.Snapshot()!=snapshot && queue.Snapshot()->generation &&
            Capture(queue.Snapshot()->generation,&b,stream) && Same(a,b));
        CHECK(Capture(source,&old,stream) && Same(original,old) && SameFrameBytes(original,old));
    }
    std::fprintf(stderr,"ScatterGrow RBF empty source\n");
    for(auto& node:sourceDesc.nodes) if(node.type==TfToken("UsdGenScatter")) node.params[0].value=VtValue(0.f);
    diagnostics={}; auto emptyPlan=CompileCudaGraph(usdGenUsd::Lower(compose(sourceDesc,0)),&diagnostics); CHECK(emptyPlan);
    auto emptyWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(emptyWorkspace);
    auto empty=ExecuteCudaPlanForTesting(emptyPlan,*emptyWorkspace,1.,300,&diagnostics);
    CHECK(empty && empty->Geometry().alreadyDeformed && empty->Geometry().pointCount==0 && empty->Geometry().elementCount==0);
    UsdGenExecutionRuntime runtime(4); UsdGenCudaExecutionQueue queue(runtime,-1,&diagnostics);
    CHECK(queue.Valid() && queue.Submit(emptyPlan,1.)); queue.Drain();
    CHECK(queue.Snapshot() && queue.Snapshot()->generation && queue.Snapshot()->generation->Geometry().alreadyDeformed &&
        queue.Snapshot()->generation->Geometry().pointCount==0 && queue.Snapshot()->generation->Geometry().elementCount==0);
    return 0;
}

int main(int argc,char** argv) {
    int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    cudaStream_t stream=nullptr; CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
    if(argc==2 && std::strcmp(argv[1],"--rbf")==0) {
        int const result=RbfCases(stream); CHECK(cudaStreamDestroy(stream)==cudaSuccess); return result;
    }
    {
        auto sourceDesc=Source(); UsdGenDiagnostics diagnostics;
        auto sourcePlan=CompileCudaGraph(usdGenUsd::Lower(sourceDesc),&diagnostics); CHECK(sourcePlan);
        auto sourceWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(sourceWorkspace);
        auto source=ExecuteCudaPlanForTesting(sourcePlan,*sourceWorkspace,1.,1,&diagnostics); CHECK(source);
        Packet original; CHECK(Capture(source,&original,stream));
        CHECK(!original.ids.empty() && !std::is_sorted(original.ids.begin(),original.ids.end()) &&
              original.t.size()==original.ids.size() && original.b.size()==original.ids.size() && original.n.size()==original.ids.size());
        for(int variant=0;variant!=7;++variant) {
            std::fprintf(stderr,"ScatterGrowValueDag variant %d\n",variant);
            diagnostics={}; auto desc=Descendants(sourceDesc,variant);
            auto plan=CompileCudaGraph(usdGenUsd::Lower(desc),&diagnostics);
            for(auto const& e:diagnostics.errors) std::fprintf(stderr,"%s\n",e.c_str());
            CHECK(plan && !diagnostics.HasErrors());
            auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(workspace);
            auto actual=ExecuteCudaPlanForTesting(plan,*workspace,1.,uint64_t(10+variant),&diagnostics);
            if(variant==6) {
                CHECK(!actual && diagnostics.HasErrors()); diagnostics={};
                UsdGenExecutionRuntime runtime(4); UsdGenCudaExecutionQueue queue(runtime,-1,&diagnostics);
                CHECK(queue.Valid() && queue.Submit(sourcePlan,1.)); queue.Drain();
                auto good=queue.Snapshot(); CHECK(good && good->generation);
                std::atomic<unsigned> callbacks{0}; std::atomic<bool> failed{false};
                CHECK(queue.Submit(plan,2.,[&](uint64_t,auto outcome,auto const&) {
                    ++callbacks; failed=outcome==UsdGenExecutionPipeline::Outcome::Failed;
                }));
                queue.Drain(); CHECK(callbacks==1 && failed && queue.Snapshot()==good);
                Packet retained; CHECK(Capture(good->generation,&retained,stream) && Same(original,retained));
                CHECK(queue.Submit(sourcePlan,3.)); queue.Drain();
                CHECK(queue.Snapshot()!=good && queue.Snapshot()->generation &&
                      Capture(queue.Snapshot()->generation,&retained,stream) && Same(original,retained));
                continue;
            }
            for(auto const& e:diagnostics.errors) std::fprintf(stderr,"%s\n",e.c_str());
            CHECK(actual && !diagnostics.HasErrors());
            auto referenceDesc=Descendants(AsC3(sourceDesc,original),variant);
            auto referencePlan=CompileCudaGraph(usdGenUsd::Lower(referenceDesc),&diagnostics); CHECK(referencePlan);
            auto referenceWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(referenceWorkspace);
            auto expected=ExecuteCudaPlanForTesting(referencePlan,*referenceWorkspace,1.,100,&diagnostics);
            for(auto const& e:diagnostics.errors) std::fprintf(stderr,"oracle: %s\n",e.c_str());
            CHECK(expected);
            Packet a,b,old; CHECK(Capture(actual,&a,stream) && Capture(expected,&b,stream) && Same(a,b));
            if(variant==2||variant==3) CHECK(!a.ids.empty() && a.ids.size()<original.ids.size());
            CHECK(Capture(source,&old,stream) && Same(original,old) && SameFrameBytes(original,old));
            UsdGenExecutionRuntime runtime(4); UsdGenCudaExecutionQueue queue(runtime,-1,&diagnostics);
            CHECK(queue.Valid() && queue.Submit(plan,1.)); queue.Drain(); auto snapshot=queue.Snapshot();
            CHECK(snapshot && snapshot->generation && Capture(snapshot->generation,&b,stream) && Same(a,b));
            CHECK(Capture(actual,&b,stream) && Same(a,b));
            CHECK(Capture(source,&old,stream) && Same(original,old) && SameFrameBytes(original,old));
            if(variant==3) {
                failNextCudaFinalizationRelayAllocationForTesting();
                std::atomic<unsigned> callbacks{0}; std::atomic<bool> failed{false};
                CHECK(queue.Submit(plan,2.,[&](uint64_t,auto outcome,auto const&) {
                    ++callbacks; failed=outcome==UsdGenExecutionPipeline::Outcome::Failed;
                }));
                queue.Drain(); CHECK(callbacks==1 && failed && queue.Snapshot()==snapshot);
                CHECK(Capture(snapshot->generation,&b,stream) && Same(a,b));
                CHECK(queue.Submit(plan,3.)); queue.Drain();
                CHECK(queue.Snapshot()!=snapshot && queue.Snapshot()->generation &&
                      Capture(queue.Snapshot()->generation,&b,stream) && Same(a,b));
            }
        }
    }
    CHECK(cudaStreamDestroy(stream)==cudaSuccess); return 0;
}
