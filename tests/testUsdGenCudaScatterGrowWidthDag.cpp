#include "usdGen/compiler.h"
#include "nodeSets.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/executionResources.h"
#include "usdGen/executionRetirement.h"
#include "usdGen/graph.h"
#include "usdGen/imagePayload.h"
#include "usdGen/scheduler.h"
#include "usdGen/session.h"
#include "usdGen/gpu/generation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

static UsdGenNodeDesc Width(char const* path, char const* input, float value) {
    UsdGenNodeDesc n;
    n.path = SdfPath(path); n.type = TfToken("UsdGenWidth");
    n.inputs = {SdfPath(input)};
    n.params = {{TfToken("width"), VtValue(value), false}};
    return n;
}

static UsdGenGraphDesc Fixture(bool branching) {
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom/Generated");
    d.executionBackend = UsdGenExecutionBackend::Cuda;
    d.defaultWidth = .025f;
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/Scalp");
    s.restPoints = {{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
    s.points = s.restPoints;
    s.uv = {{0,0},{1,0},{1,1},{0,1}};
    s.faceVertexCounts = {4}; s.faceVertexIndices = {0,1,2,3};
    d.surfaces = {s};
    UsdGenNodeDesc scatter;
    scatter.path = SdfPath("/Ops/Scatter"); scatter.type = TfToken("UsdGenScatter");
    scatter.seed = 41; scatter.surfaces = {s.path};
    scatter.params = {{TfToken("density"), VtValue(300.0f), false}};
    UsdGenNodeDesc grow;
    grow.path = SdfPath("/Ops/Grow"); grow.type = TfToken("UsdGenGrow");
    grow.seed = 19; grow.inputs = {scatter.path};
    grow.params = {{TfToken("segments"),VtValue(5),false},
                  {TfToken("length"),VtValue(2.0f),false},
                  {TfToken("lengthRandom"),VtValue(GfVec2f(.5f,1.5f)),false}};
    auto left = Width("/Ops/Left", "/Ops/Grow", .2f);
    if (!branching) {
        // Authored ordinals deliberately differ from execution order.
        d.nodes = {left, grow, scatter}; d.terminal = left.path;
        return d;
    }
    auto right = Width("/Ops/Right", "/Ops/Grow", .8f);
    UsdGenNodeDesc blend;
    blend.path = SdfPath("/Ops/Blend"); blend.type = TfToken("UsdGenWidthBlend");
    blend.inputs = {left.path, right.path};
    blend.params.push_back({TfToken("widthBlend:weight"), VtValue(.25f), false});
    d.nodes = {blend, right, scatter, left, grow}; d.terminal = blend.path;
    return d;
}

static bool Reference(UsdGenGraphDesc d, UsdGenCurveBuffer* result) {
    d.executionBackend = UsdGenExecutionBackend::CpuReference;
    UsdGenCompiler compiler; UsdGenGraph graph;
    if (!compiler.Compile(d, &graph).ok) return false;
    UsdGenScheduler scheduler(2); UsdGenEvalContext ctx; ctx.desc = &graph.Desc();
    auto evaluated = scheduler.Run(graph, ctx, 1);
    if (evaluated.diagnostics.HasErrors()) return false;
    *result = NodeBuffer(graph, graph.NodeIdForPath(d.terminal));
    return true;
}

template<class T> static bool Read(gpu::DeviceView<const T> v, std::vector<T>* out,
                                   cudaStream_t stream) {
    out->resize(v.size);
    return (!v.size || cudaMemcpyAsync(out->data(), v.data, v.size*sizeof(T),
        cudaMemcpyDeviceToHost, stream) == cudaSuccess) && cudaStreamSynchronize(stream) == cudaSuccess;
}
static bool Near(float a, float b) { return std::fabs(a-b) < 2e-5f; }

int main() {
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
    cudaStream_t stream=nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
    for (int variant : {0,1,2,3}) {
        std::fprintf(stderr,"ScatterGrowWidthDag variant %d\n",variant);
        bool const branching=variant!=0;
        auto desc=Fixture(branching);
        // Execute the descendant branches but publish their predecessor.
        // Any in-place Width write would corrupt this selected source value.
        if(variant==3) desc.terminal=SdfPath("/Ops/Grow");
        CHECK(desc.curveSets.empty());
        UsdGenCurveBuffer reference;
        CHECK(Reference(desc,&reference));
        UsdGenDiagnostics diagnostics;
        auto plan=CompileCudaGraph(desc,&diagnostics);
        for(auto const& e:diagnostics.errors) std::fprintf(stderr,"%s\n",e.c_str());
        CHECK(plan && !diagnostics.HasErrors());
        auto metadata=GetCudaExecutionPlanMetadata(*plan);
        CHECK(metadata && metadata->Tasks().size()==(branching?5:3));
        CHECK(metadata->TerminalTask()==metadata->Tasks().size()-1);
        CHECK(metadata->FindTask(metadata->TerminalTask())->kind==UsdGenExecutionTaskKind::Publication);
        CHECK(metadata->Operators().size()==desc.nodes.size());
        for(auto const& task:metadata->Tasks()) {
            if(task.type!=TfToken("UsdGenWidth")) continue;
            CHECK(!task.exclusiveWorkspace);
            for(auto kind:{UsdGenExecutionDataKind::StableIds,UsdGenExecutionDataKind::RootBindings}) {
                auto use=std::find_if(task.resources.begin(),task.resources.end(),[&](auto const& r) {
                    return r.resource==kind;
                });
                CHECK(use!=task.resources.end() && use->access==UsdGenExecutionResourceAccess::Read && use->producerTask==0);
            }
        }
        CHECK(metadata->MemoryEstimate().memoryAvailable && metadata->MemoryEstimate().conservativeUpperBound);
        UsdGenSession session; session.SetDevicePublicationEnabled(true); session.SetGraphDesc(desc);
        auto first=session.Commit(1,UsdGenCommitReason::SetTime);
        for(auto const& e:session.LastDiagnostics().errors) std::fprintf(stderr,"%s\n",e.c_str());
        CHECK(first && first->device && !session.LastDiagnostics().HasErrors());
        auto old=gpu::AcquireGeometry(first->device,stream);
        CHECK(old);
        std::vector<float3> points;
        std::vector<float> widths;
        std::vector<uint64_t> ids;
        CHECK(Read(old.Geometry().points,&points,stream) && Read(old.Geometry().widths,&widths,stream));
        CHECK(Read(old.Geometry().stableIds,&ids,stream));
        CHECK(points.size()==reference.totalCvs && widths.size()==points.size());
        CHECK(!std::is_sorted(ids.begin(),ids.end()));
        for(size_t i=0;i<ids.size();++i) CHECK(ids[i]==reference.curveId[i]);
        for(size_t i=0;i<points.size();++i) {
            CHECK(Near(points[i].x,reference.px[i]) && Near(points[i].y,reference.py[i]) && Near(points[i].z,reference.pz[i]));
            if(!Near(widths[i],reference.width[i]))
                std::fprintf(stderr,"variant %d width[%zu]: GPU %.9g CPU %.9g\n",variant,i,widths[i],reference.width[i]);
            CHECK(Near(widths[i],reference.width[i]));
        }
        auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(workspace);
        auto direct=ExecuteCudaGraph(*plan,*workspace,1,100,&diagnostics);
        CHECK(direct && !diagnostics.HasErrors());
        auto directLease=gpu::AcquireGeometry(direct,stream); CHECK(directLease);
        std::vector<float> directWidths;
        CHECK(Read(directLease.Geometry().widths,&directWidths,stream) && directWidths==widths);
        CHECK(direct->Geometry().tiles.size()==first->device->Geometry().tiles.size());
        // Leave exactly peak-1 bytes available in the ledger. Neither entry
        // point may submit work or alter any category when admission fails.
        int device=-1; CHECK(cudaGetDevice(&device)==cudaSuccess);
        UsdGenExecutionResourceDevice key{UsdGenExecutionResourceBackend::Cuda,device};
        if(auto retirement=FindUsdGenExecutionRetirementService(key)) retirement->Drain();
        auto pool=FindUsdGenExecutionResourcePool(key); CHECK(pool);
        auto baseline=pool->Snapshot();
        auto peak=metadata->MemoryEstimate().concurrentPeakBytes;
        CHECK(peak>0 && baseline.usedBytes<=baseline.usableBytes &&
              peak<=baseline.usableBytes-baseline.usedBytes);
        auto filler=pool->TryReserve(baseline.usableBytes-baseline.usedBytes-peak+1,
            UsdGenExecutionResourceKind::Active);
        CHECK(filler);
        auto saturated=pool->Snapshot();
        CHECK(!ExecuteCudaGraph(*plan,*workspace,1,101,&diagnostics,first->device));
        CHECK(!CreateCudaExecutionJob(plan,*workspace,1,102,&diagnostics,first->device));
        CHECK(diagnostics.HasErrors());
        auto rejected=pool->Snapshot();
        CHECK(rejected.usedBytes==saturated.usedBytes && rejected.byKind==saturated.byKind);
        CHECK(!workspace->IsPoisoned());
        filler->Release();
        diagnostics={};
        auto originalWidths=widths;
        auto originalPoints=points;
        if(variant==3) desc.defaultWidth=.6f;
        else if(branching) std::swap(desc.nodes[0].inputs[0],desc.nodes[0].inputs[1]);
        else desc.nodes[0].params[0].value=VtValue(.6f);
        CHECK(Reference(desc,&reference));
        session.SetGraphDesc(desc);
        auto second=session.Commit(2,UsdGenCommitReason::SetTime);
        CHECK(second && second!=first && second->device && !session.LastDiagnostics().HasErrors());
        auto current=gpu::AcquireGeometry(second->device,stream); CHECK(current);
        CHECK(current.Geometry().widths.data!=old.Geometry().widths.data);
        CHECK(second->device->Geometry().topologyVersion==first->device->Geometry().topologyVersion);
        CHECK(Read(current.Geometry().widths,&widths,stream));
        for(size_t i=0;i<widths.size();++i) CHECK(Near(widths[i],reference.width[i]));
        CHECK(widths!=originalWidths);
        CHECK(Read(old.Geometry().widths,&widths,stream) && widths==originalWidths);
        CHECK(Read(old.Geometry().points,&points,stream));
        CHECK(std::memcmp(points.data(),originalPoints.data(),points.size()*sizeof(float3))==0);
        if(variant==2) {
            desc.defaultWidth=.125f;
            CHECK(Reference(desc,&reference));
            session.SetGraphDesc(desc);
            auto fallbackEdit=session.Commit(3,UsdGenCommitReason::SetTime);
            CHECK(fallbackEdit && fallbackEdit!=second && fallbackEdit->device && !session.LastDiagnostics().HasErrors());
            auto edited=gpu::AcquireGeometry(fallbackEdit->device,stream); CHECK(edited);
            CHECK(Read(edited.Geometry().widths,&widths,stream));
            CHECK(widths.size()==reference.width.size());
            for(size_t i=0;i<widths.size();++i) CHECK(Near(widths[i],reference.width[i]));
        }
        // Generated cardinality, including empty output, must drive both
        // Width allocation and finalization budgets without a C3 source.
        for (float density : {1.0f,0.0f}) {
            auto small=desc;
            auto scatter=std::find_if(small.nodes.begin(),small.nodes.end(),[](auto const& n) {
                return n.type==TfToken("UsdGenScatter");
            });
            CHECK(scatter!=small.nodes.end());
            scatter->params[0].value=VtValue(density);
            CHECK(Reference(small,&reference));
            session.SetGraphDesc(small);
            auto smallGeneration=session.Commit(density==0?5:4,UsdGenCommitReason::SetTime);
            for(auto const& e:session.LastDiagnostics().errors) std::fprintf(stderr,"%s\n",e.c_str());
            CHECK(smallGeneration && smallGeneration!=second && smallGeneration->device && !session.LastDiagnostics().HasErrors());
            auto smallLease=gpu::AcquireGeometry(smallGeneration->device,stream); CHECK(smallLease);
            CHECK(smallLease.Geometry().curveCount==reference.totalCurves);
            CHECK(Read(smallLease.Geometry().widths,&widths,stream));
            CHECK(widths.size()==reference.totalCvs);
            for(size_t i=0;i<widths.size();++i) CHECK(Near(widths[i],reference.width[i]));
        }
    }
    CHECK(cudaStreamDestroy(stream)==cudaSuccess);
    std::puts("testUsdGenCudaScatterGrowWidthDag: PASS");
    return 0;
}
