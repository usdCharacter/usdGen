#include "usdGen/compiler.h"
#include "usdGenOpFixture.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGen/session.h"
#include "usdGen/gpu/generation.h"
#include "usdGen/imagePayload.h"
#include "usdGenMath/usdGenMath/hash.h"
#include "SeExpr2/Noise.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <functional>
#include <limits>
#include <map>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {
bool Near(float a, float b) { return std::fabs(a-b) < 2.e-5f; }
bool Near(float3 a, GfVec3f b) { return Near(a.x,b[0])&&Near(a.y,b[1])&&Near(a.z,b[2]); }

template<class T> bool Read(gpu::DeviceView<const T> input, std::vector<T>* output,
                            cudaStream_t stream) {
    output->resize(input.size);
    return (!input.size || cudaMemcpyAsync(output->data(),input.data,
        input.size*sizeof(T),cudaMemcpyDeviceToHost,stream)==cudaSuccess) &&
        cudaStreamSynchronize(stream)==cudaSuccess;
}

UsdGenNodeDesc Width(char const* path, char const* input, float width) {
    UsdGenNodeDesc node;
    node.path=SdfPath(path); node.type=TfToken("UsdGenWidth");
    node.inputs={SdfPath(input)};
    node.params={{TfToken("width"),VtValue(width),false}};
    return node;
}

bool CpuReference(UsdGenGraphDesc desc, UsdGenCurveBuffer* output);

// The CPU operator implementation currently projects a scalar FBM result;
// this test oracle follows the native CUDA vector FBM/TBN contract instead.
bool GrowNoiseReference(UsdGenGraphDesc desc, UsdGenCurveBuffer* output,
                        bool transformOnly = false) {
    auto noise = std::find_if(desc.nodes.begin(), desc.nodes.end(),
        [](auto const& node) { return node.type == TfToken("UsdGenNoise"); });
    if (noise == desc.nodes.end() || !output) return false;
    auto withoutNoise = desc;
    auto muted = std::find_if(withoutNoise.nodes.begin(), withoutNoise.nodes.end(),
        [&](auto const& node) { return node.path == noise->path; });
    if (muted == withoutNoise.nodes.end()) return false;
    bool mutedMagnitude = false;
    for (auto& parameter : muted->params) {
        if (parameter.name == TfToken("noise:magnitude")) {
            parameter.value = VtValue(0.0f);
            mutedMagnitude = true;
        }
    }
    if (!mutedMagnitude) muted->params.push_back(
        {TfToken("noise:magnitude"), VtValue(0.0f), false});
    if (!noise->expressionBindings.empty() || !noise->ramps.empty()) return false;
    for (auto const& parameter : noise->params) {
        if (parameter.name != TfToken("enabled") &&
            parameter.name != TfToken("noise:magnitude") &&
            parameter.name != TfToken("noise:frequency") &&
            parameter.name != TfToken("noise:correlation") &&
            parameter.name != TfToken("noise:seed") &&
            parameter.name != TfToken("noise:octaves") &&
            parameter.name != TfToken("preserveLength")) return false;
    }
    if (!transformOnly && !CpuReference(withoutNoise, output)) return false;
    usdGenTest::NodeParams values(*noise);
    if (!values.GetBool(TfToken("enabled"), true)) return false;
    if (values.GetDouble(TfToken("preserveLength"), 1.0) != 0.0 ||
        values.GetBool(TfToken("cumulative"), false)) return false;
    float const magnitude = float(values.GetDouble(TfToken("noise:magnitude"), .05));
    float const frequency = float(values.GetDouble(TfToken("noise:frequency"), 3.0));
    float const correlation = float(values.GetDouble(TfToken("noise:correlation"), .5));
    int const seed = values.GetInt(TfToken("noise:seed"), 0);
    int const octaves = values.GetInt(TfToken("noise:octaves"), 1);
    float const amplitude = magnitude;
    for (size_t curve = 0; curve != output->totalCurves; ++curve) {
        uint32_t begin = output->cvOffsets.empty()
            ? uint32_t(curve * output->totalCvs / output->totalCurves)
            : uint32_t(output->cvOffsets[curve]);
        uint32_t end = output->cvOffsets.empty()
            ? uint32_t((curve + 1) * output->totalCvs / output->totalCurves)
            : uint32_t(output->cvOffsets[curve + 1]);
        auto const root = output->rest[begin];
        auto const id = output->curveId[curve];
        auto const t = output->rootT[curve];
        auto const b = output->rootB[curve];
        auto const n = output->rootN[curve];
        float3 hash{UsdGenDraw01(seed, id, kSaltNoise),
                    UsdGenDraw01(seed, id, kSaltNoise + 1u),
                    UsdGenDraw01(seed, id, kSaltNoise + 2u)};
        for (uint32_t point = begin; point != end; ++point) {
            float const hairT = output->hairT[point];
            double const input[3] = {
                double(root[0] * correlation + hash.x * (1 - correlation)),
                double(root[1] * correlation + hash.y * (1 - correlation)),
                double(root[2] * correlation + hash.z * (1 - correlation) +
                       hairT * frequency)};
            double value[3]{};
            SeExpr2::FBM<3, 3, false, double>(input, value, octaves, 2.0, .5);
            float3 local{float(value[0]) * amplitude, float(value[1]) * amplitude,
                         float(value[2]) * amplitude};
            output->px[point] += t[0] * local.x + b[0] * local.y + n[0] * local.z;
            output->py[point] += t[1] * local.x + b[1] * local.y + n[1] * local.z;
            output->pz[point] += t[2] * local.x + b[2] * local.y + n[2] * local.z;
        }
    }
    return true;
}

UsdGenGraphDesc Desc(bool resampleAndLift = false, bool namedPlanes = true,
                     bool emptySource = false, bool withNoise = false) {
    UsdGenGraphDesc desc;
    desc.description=SdfPath("/Groom/C3Grow");
    desc.executionBackend=UsdGenExecutionBackend::Cuda;
    desc.defaultWidth=.125f;
    UsdGenCurveSetDesc curves;
    curves.path=SdfPath("/Hair"); curves.role=UsdGenRole::Curves;
    curves.curveRole=TfToken("hair");
    if (!emptySource) {
        curves.curveVertexCounts={3,2};
        curves.rest={{10,0,0},{11,0,0},{12,0,0},{20,0,0},{21,0,0}};
        curves.points=curves.rest;
        for (auto& point : curves.points) point[0]+=100;
        curves.widths={.1f,.2f,.3f,.4f,.5f};
        curves.curveId={40,7}; curves.skinPrim={2,1};
        curves.skinPrimUv={{.2f,.3f},{.4f,.5f}};
        curves.rootFrame={GfMatrix4d(1.0),GfMatrix4d(1.0)};
        if (withNoise) {
            curves.rootFrame[1][0][0] = 0; curves.rootFrame[1][0][1] = 1;
            curves.rootFrame[1][1][0] = 0; curves.rootFrame[1][1][1] = 0;
            curves.rootFrame[1][1][2] = 1;
            curves.rootFrame[1][2][0] = 1; curves.rootFrame[1][2][1] = 0;
            curves.rootFrame[1][2][2] = 0;
        }
        if (namedPlanes) {
            UsdGenAuthoredPlaneDesc vertex;
            vertex.name=TfToken("sourceValue"); vertex.type=UsdGenAuthoredPlaneType::Float32;
            vertex.domain=UsdGenAuthoredPlaneDomain::Point; vertex.arity=1;
            vertex.floatValues={0,1,2,3,4}; curves.authoredPlanes.push_back(vertex);
            UsdGenAuthoredPlaneDesc uniform;
            uniform.name=TfToken("sourceGroup"); uniform.type=UsdGenAuthoredPlaneType::Int32;
            uniform.domain=UsdGenAuthoredPlaneDomain::Primitive; uniform.arity=1;
            uniform.intValues={40,7}; curves.authoredPlanes.push_back(uniform);
        }
    }
    desc.curveSets={curves};
    UsdGenNodeDesc source;
    source.path=SdfPath("/Ops/Source"); source.type=TfToken("UsdGenCurveSource");
    source.curves={curves.path};
    source.params={{TfToken("rebind"),VtValue(TfToken("never")),false},
                   {TfToken("useRest"),VtValue(false),false}};
    if(resampleAndLift)
        source.params.push_back({TfToken("resampleTo"),VtValue(3),false});
    UsdGenNodeDesc grow;
    grow.path=SdfPath("/Ops/Grow"); grow.type=TfToken("UsdGenGrow");
    grow.seed=19; grow.inputs={source.path};
    grow.params={{TfToken("segments"),VtValue(4),false},
                 {TfToken("length"),VtValue(2.0),false},
                 {TfToken("lengthRandom"),VtValue(GfVec2f(.5f,1.5f)),false},
                 {TfToken("direction"),VtValue(TfToken("surfaceNormal")),false}};
    if(resampleAndLift) {
        grow.params.push_back({TfToken("lift"),VtValue(30.0f),false});
        grow.params.push_back({TfToken("azimuth"),VtValue(35.0f),false});
        grow.params.push_back({TfToken("azimuthRandom"),VtValue(1.0f),false});
    }
    auto left=Width("/Ops/Left","/Ops/Grow",.2f);
    auto right=Width("/Ops/Right","/Ops/Grow",.8f);
    UsdGenNodeDesc noise;
    noise.path=SdfPath("/Ops/Noise"); noise.type=TfToken("UsdGenNoise");
    noise.inputs={grow.path};
    noise.params={{TfToken("enabled"),VtValue(true),false},
                 {TfToken("noise:frequency"),VtValue(1.3f),false},
                 {TfToken("noise:magnitude"),VtValue(.052f),false},
                 {TfToken("noise:correlation"),VtValue(.5f),false},
                 {TfToken("noise:seed"),VtValue(23),false},
                 {TfToken("noise:octaves"),VtValue(2),false},
                 {TfToken("preserveLength"),VtValue(0.0f),false}};
    if (withNoise) {
        left.inputs={noise.path}; right.inputs={noise.path};
    }
    UsdGenNodeDesc blend;
    blend.path=SdfPath("/Ops/Blend"); blend.type=TfToken("UsdGenWidthBlend");
    blend.inputs={left.path,right.path};
    blend.params.push_back({TfToken("widthBlend:weight"),VtValue(.25f),false});
    // Deliberately not execution order: C3 canonicalization and value DAG
    // lowering must not depend on descriptor ordinal.
    // Noise is currently exercised in the linear native suffix.  Keep the
    // pre-existing shuffled two-branch WidthBlend fixture intact when the
    // optional Noise stage is disabled.
    desc.nodes=withNoise ? std::vector<UsdGenNodeDesc>{source,grow,noise,left} :
        std::vector<UsdGenNodeDesc>{blend,right,source,left,grow};
    desc.terminal=withNoise ? left.path : blend.path;
    return desc;
}

UsdGenGraphDesc GrowLengthDesc(TfToken mode, float value, bool cutExtend = false) {
    auto desc = Desc();
    // Distinct frames make survivor-ID/frame association observable.
    GfMatrix4d rotated(1.0);
    rotated[0][0]=0;rotated[0][1]=1;
    rotated[1][1]=0;rotated[1][2]=1;
    rotated[2][2]=0;rotated[2][0]=1;
    desc.curveSets.front().rootFrame.front()=rotated;
    auto source = std::find_if(desc.nodes.begin(), desc.nodes.end(),
        [](auto const& n){ return n.type == TfToken("UsdGenCurveSource"); });
    auto grow = std::find_if(desc.nodes.begin(), desc.nodes.end(),
        [](auto const& n){ return n.type == TfToken("UsdGenGrow"); });
    if (source == desc.nodes.end() || grow == desc.nodes.end()) return {};
    UsdGenNodeDesc length;
    length.path = SdfPath("/Ops/Length"); length.type = TfToken("UsdGenLength");
    length.inputs = {grow->path};
    length.params = {{TfToken("length:mode"), VtValue(mode), false},
                     {TfToken(mode == TfToken("cull") ? "cullThreshold" : "length:value"),
                      VtValue(value), false}};
    if (cutExtend) length.params.push_back(
        {TfToken("length:method"), VtValue(TfToken("cutExtend")), false});
    auto width = Width("/Ops/LengthWidth", "/Ops/Length", .2f);
    desc.nodes = {*source, *grow, length, width};
    desc.terminal = width.path;
    return desc;
}

bool CpuReference(UsdGenGraphDesc desc, UsdGenCurveBuffer* output) {
    desc.executionBackend=UsdGenExecutionBackend::CpuReference;
    UsdGenCompiler compiler; UsdGenGraph graph;
    auto compiled=compiler.Compile(desc,&graph);
    if(!compiled.ok) {
        std::fprintf(stderr,"CPU oracle compile failed terminal %s\n",desc.terminal.GetText());
        return false;
    }
    UsdGenScheduler scheduler(2); UsdGenEvalContext context; context.desc=&graph.Desc();
    auto result=scheduler.Run(graph,context,1);
    if(result.diagnostics.HasErrors()) {
        for(auto const& error:result.diagnostics.errors) std::fprintf(stderr,"CPU oracle: %s\n",error.c_str());
        return false;
    }
    *output=graph.Node(graph.NodeIdForPath(desc.terminal)).buffer;
    return true;
}

bool GrowLengthScaleReference(UsdGenGraphDesc const& desc, float scale,
                              UsdGenCurveBuffer* output, bool cutExtend = false,
                              bool transformOnly = false) {
    auto base = desc;
    auto length = std::find_if(base.nodes.begin(), base.nodes.end(),
        [](auto const& n){ return n.type == TfToken("UsdGenLength"); });
    if (length == base.nodes.end()) return false;
    usdGenTest::NodeParams parameters(*length);
    auto const mode=parameters.GetToken(TfToken("length:mode"),TfToken("scale"));
    auto const method=parameters.GetToken(TfToken("length:method"),TfToken("scale"));
    if ((mode!=TfToken("scale")&&mode!=TfToken("cull")) ||
        method!=(cutExtend?TfToken("cutExtend"):TfToken("scale"))) return false;
    for(auto const& parameter:length->params)
        if(parameter.name!=TfToken("length:mode")&&parameter.name!=TfToken("length:method")&&
           parameter.name!=TfToken("length:value")&&parameter.name!=TfToken("cullThreshold")) return false;
    bool const cull=mode==TfToken("cull");
    float const threshold=float(parameters.GetDouble(TfToken("cullThreshold"),0));
    if (!length->expressionBindings.empty() || !length->ramps.empty() ||
        (!cull && parameters.GetDouble(TfToken("length:value"),1)!=double(scale)) ||
        (cutExtend && (scale<0 || scale>1))) return false;
    if (!transformOnly) {
    base.nodes.erase(length);
    auto width = std::find_if(base.nodes.begin(), base.nodes.end(),
        [](auto const& n){ return n.type == TfToken("UsdGenWidth"); });
    if (width == base.nodes.end()) return false;
    width->inputs = {SdfPath("/Ops/Grow")};
    base.terminal = width->path;
    if (!CpuReference(base, output)) return false;
    }
    for (size_t curve = 0; curve != output->totalCurves; ++curve) {
        if (cull) continue;
        uint32_t begin = output->cvOffsets.empty()
            ? uint32_t(curve * output->totalCvs / output->totalCurves)
            : uint32_t(output->cvOffsets[curve]);
        uint32_t end = output->cvOffsets.empty()
            ? uint32_t((curve + 1) * output->totalCvs / output->totalCurves)
            : uint32_t(output->cvOffsets[curve + 1]);
        GfVec3f root{output->px[begin], output->py[begin], output->pz[begin]};
        if (!cutExtend) for (uint32_t i = begin; i != end; ++i) {
            output->px[i] = root[0] + (output->px[i] - root[0]) * scale;
            output->py[i] = root[1] + (output->py[i] - root[1]) * scale;
            output->pz[i] = root[2] + (output->pz[i] - root[2]) * scale;
        } else {
            std::vector<GfVec3f> original;
            float total = 0;
            for (uint32_t i = begin; i != end; ++i) {
                original.push_back({output->px[i],output->py[i],output->pz[i]});
                if (i > begin) total += (original.back()-original[original.size()-2]).GetLength();
            }
            float const target = total * scale;
            std::vector<float> arc(original.size(),0);
            for(size_t j=1;j<original.size();++j)
                arc[j]=arc[j-1]+(original[j]-original[j-1]).GetLength();
            for (uint32_t i = begin; i != end; ++i) {
                float const distance=std::min(target,arc[i-begin]);
                size_t j=1;
                while(j+1<arc.size() && arc[j]<distance) ++j;
                float const span=arc[j]-arc[j-1];
                auto const point=original[j-1]+(original[j]-original[j-1])*
                    (span>0 ? (distance-arc[j-1])/span : 0);
                output->px[i]=point[0]; output->py[i]=point[1]; output->pz[i]=point[2];
            }
        }
    }
    if (cull) {
        if (scale!=1 || cutExtend) return false;
        UsdGenCurveBuffer const input=*output;
        UsdGenCurveBuffer compact;
        compact.cvOffsets.push_back(0);
        std::vector<uint32_t> survivors;
        for(uint32_t c=0;c<input.totalCurves;++c) {
            uint32_t const begin=input.cvOffsets.empty()?c*input.totalCvs/input.totalCurves:input.cvOffsets[c];
            uint32_t const end=input.cvOffsets.empty()?(c+1)*input.totalCvs/input.totalCurves:input.cvOffsets[c+1];
            float arc=0;
            for(uint32_t i=begin+1;i<end;++i)
                arc+=GfVec3f(input.px[i]-input.px[i-1],input.py[i]-input.py[i-1],
                            input.pz[i]-input.pz[i-1]).GetLength();
            if(arc<threshold) continue;
            survivors.push_back(c);
            for(uint32_t i=begin;i<end;++i) {
                compact.px.push_back(input.px[i]);compact.py.push_back(input.py[i]);
                compact.pz.push_back(input.pz[i]);compact.rest.push_back(input.rest[i]);
                compact.width.push_back(input.width[i]);compact.hairT.push_back(input.hairT[i]);
            }
            compact.curveId.push_back(input.curveId[c]);
            compact.rootPrim.push_back(input.rootPrim[c]);compact.rootUV.push_back(input.rootUV[c]);
            compact.rootT.push_back(input.rootT[c]);compact.rootB.push_back(input.rootB[c]);
            compact.rootN.push_back(input.rootN[c]);
            compact.cvOffsets.push_back(int(compact.px.size()));
        }
        compact.totalCurves=uint32_t(survivors.size());compact.totalCvs=uint32_t(compact.px.size());
        if(!UsdGenCompactExtraPlanes(input,survivors,&compact)) return false;
        *output=std::move(compact);
    }
    return true;
}

// Start from independently evaluated Source/Grow/constant Width, then apply
// each bounded host transform in graph order. Never invoke CPU Length/Noise
// for the expected mixed result (their semantics are known to differ).
bool MixedGrowReference(UsdGenGraphDesc const& desc, UsdGenCurveBuffer* output) {
    auto base=desc;
    base.nodes.erase(std::remove_if(base.nodes.begin(),base.nodes.end(),[](auto const& n) {
        return n.type==TfToken("UsdGenLength")||n.type==TfToken("UsdGenNoise");
    }),base.nodes.end());
    if(base.nodes.size()!=3||base.nodes.back().type!=TfToken("UsdGenWidth")) return false;
    base.nodes.back().inputs={base.nodes[1].path};
    if(!CpuReference(base,output)) return false;
    for(auto const& node:desc.nodes) {
        UsdGenGraphDesc stage=desc;stage.nodes={node};
        if(node.type==TfToken("UsdGenNoise")) {
            if(!GrowNoiseReference(stage,output,true)) return false;
        } else if(node.type==TfToken("UsdGenLength")) {
            usdGenTest::NodeParams p(node);
            float const scale=float(p.GetDouble(TfToken("length:value"),1));
            bool const cut=p.GetToken(TfToken("length:method"),TfToken("scale"))==TfToken("cutExtend");
            if(!GrowLengthScaleReference(stage,scale,output,cut,true)) return false;
        }
    }
    return true;
}

UsdGenGraphDesc NoiseDagDesc() {
    auto desc=Desc(false,true,false,true);
    auto source=desc.nodes.front();
    auto noise=desc.nodes[2];noise.path=SdfPath("/Dag/Noise");
    auto pre=Width("/Dag/Pre",source.path.GetText(),.2f);
    noise.inputs={pre.path};
    auto left=Width("/Dag/Left",noise.path.GetText(),.3f);
    auto right=Width("/Dag/Right",noise.path.GetText(),.9f);
    auto other=noise;other.path=SdfPath("/Dag/Other");other.inputs={source.path};
    auto tail=noise;tail.path=SdfPath("/Dag/Tail");tail.inputs={left.path};
    for(auto& p:other.params) if(p.name==TfToken("noise:magnitude")) p.value=VtValue(.104f);
    for(auto& p:tail.params) if(p.name==TfToken("noise:magnitude")) p.value=VtValue(.026f);
    UsdGenNodeDesc blend;blend.path=SdfPath("/Dag/Blend");blend.type=TfToken("UsdGenWidthBlend");
    blend.inputs={left.path,right.path};
    blend.params.push_back({TfToken("widthBlend:weight"),VtValue(.25f),false});
    desc.nodes={blend,other,right,tail,source,noise,left,pre};desc.terminal=blend.path;
    return desc;
}

UsdGenGraphDesc TopologyNoiseDagDesc(bool grow, TfToken mode=TfToken("scale"),
                                    float value=.5f, bool cut=false) {
    auto desc=NoiseDagDesc();
    desc.curveSets.front().basis=TfToken("catmullRom");
    auto templateDesc=grow ? Desc() : GrowLengthDesc(mode,value,cut);
    auto trunk=std::find_if(templateDesc.nodes.begin(),templateDesc.nodes.end(),[&](auto const& node) {
        return node.type==TfToken(grow?"UsdGenGrow":"UsdGenLength");
    });
    if(trunk==templateDesc.nodes.end()) return {};
    auto node=*trunk;node.path=SdfPath("/Dag/Topology");node.inputs={SdfPath("/Ops/Source")};
    for(auto& descendant:desc.nodes) for(auto& input:descendant.inputs)
        if(input==SdfPath("/Ops/Source")) input=node.path;
    desc.nodes.insert(desc.nodes.begin()+2,node);
    return desc;
}

UsdGenGraphDesc SourceBranchNoiseDagDesc(bool grow, TfToken mode=TfToken("scale"),float value=.5f) {
    auto desc=TopologyNoiseDagDesc(grow,mode,value);
    auto other=std::find_if(desc.nodes.begin(),desc.nodes.end(),[](auto const& n){return n.path==SdfPath("/Dag/Other");});
    other->inputs={SdfPath("/Ops/Source")};
    auto tail=*other;tail.path=SdfPath("/Dag/SideTail");tail.inputs={SdfPath("/Dag/SideLeft")};
    for(auto& p:tail.params) if(p.name==TfToken("noise:magnitude")) p.value=VtValue(.0585f);
    desc.nodes.push_back(Width("/Dag/SideLeft","/Dag/Other",.4f));
    desc.nodes.push_back(Width("/Dag/SideRight","/Dag/Other",.8f));
    desc.nodes.push_back(Width("/Dag/SideRawWidth","/Ops/Source",.6f));
    UsdGenNodeDesc blend;blend.path=SdfPath("/Dag/SideBlend");blend.type=TfToken("UsdGenWidthBlend");
    blend.inputs={SdfPath("/Dag/SideLeft"),SdfPath("/Dag/SideRight")};
    blend.params.push_back({TfToken("widthBlend:weight"),VtValue(.6f),false});
    desc.nodes.push_back(blend);desc.nodes.push_back(tail);desc.terminal=blend.path;
    return desc;
}

UsdGenGraphDesc MultipleLengthNoiseDagDesc(bool nested=false,
                                         TfToken mode=TfToken("scale"),float value=.75f) {
    auto desc=SourceBranchNoiseDagDesc(false);
    auto templateDesc=GrowLengthDesc(mode,value);
    auto length=std::find_if(templateDesc.nodes.begin(),templateDesc.nodes.end(),[](auto const& node) {
        return node.type==TfToken("UsdGenLength");
    });
    if(length==templateDesc.nodes.end()) return {};
    auto side=*length;side.path=SdfPath("/Dag/SideTopology");
    side.inputs={SdfPath(nested?"/Dag/Tail":"/Ops/Source")};
    for(auto& node:desc.nodes) if(node.path==SdfPath("/Dag/Other")) node.inputs={side.path};
    desc.nodes.insert(desc.nodes.begin(),side);
    return desc;
}

UsdGenGraphDesc MultipleGrowNoiseDagDesc(bool withLengths=false,
                                       TfToken mode=TfToken("scale"),float value=.5f) {
    auto desc=SourceBranchNoiseDagDesc(true);
    auto first=std::find_if(desc.nodes.begin(),desc.nodes.end(),[](auto const& node) {
        return node.type==TfToken("UsdGenGrow");
    });
    if(first==desc.nodes.end()) return {};
    auto side=*first;side.path=SdfPath("/Dag/SideTopology");
    for(auto& p:side.params) {
        if(p.name==TfToken("segments")) p.value=VtValue(6);
        if(p.name==TfToken("length")) p.value=VtValue(.7);
    }
    side.params.push_back({TfToken("lift"),VtValue(-20.f),false});
    for(auto& node:desc.nodes) if(node.path==SdfPath("/Dag/Other")) node.inputs={side.path};
    desc.nodes.insert(desc.nodes.begin(),side);
    if(withLengths) {
        auto templateDesc=GrowLengthDesc(mode,value);
        auto length=std::find_if(templateDesc.nodes.begin(),templateDesc.nodes.end(),[](auto const& node) {
            return node.type==TfToken("UsdGenLength");
        });
        if(length==templateDesc.nodes.end()) return {};
        auto mainLength=*length;mainLength.path=SdfPath("/Dag/MainLength");mainLength.inputs={SdfPath("/Dag/Tail")};
        auto sideLength=*length;sideLength.path=SdfPath("/Dag/SideLength");sideLength.inputs={SdfPath("/Dag/SideTail")};
        desc.nodes.push_back(sideLength);desc.nodes.push_back(mainLength);
        desc.terminal=sideLength.path;
    }
    return desc;
}

UsdGenGraphDesc RegrowNoiseDagDesc(bool grownInput,char const* input,
                                 TfToken mode=TfToken("scale"),float value=.5f) {
    auto desc=grownInput?MultipleGrowNoiseDagDesc(true,mode,value):SourceBranchNoiseDagDesc(false,mode,value);
    auto templateDesc=Desc();
    auto grow=std::find_if(templateDesc.nodes.begin(),templateDesc.nodes.end(),[](auto const& node) {
        return node.type==TfToken("UsdGenGrow");
    });
    if(grow==templateDesc.nodes.end()) return {};
    auto regrow=*grow;regrow.path=SdfPath("/Dag/Regrow");regrow.inputs={SdfPath(input)};
    for(auto& p:regrow.params) {
        if(p.name==TfToken("segments")) p.value=VtValue(5);
        if(p.name==TfToken("length")) p.value=VtValue(1.1);
    }
    regrow.params.push_back({TfToken("lift"),VtValue(25.f),false});
    auto noise=std::find_if(desc.nodes.begin(),desc.nodes.end(),[](auto const& node) {
        return node.path==SdfPath("/Dag/Noise");
    });
    if(noise==desc.nodes.end()) return {};
    auto tail=*noise;tail.path=SdfPath("/Dag/RegrowNoise");tail.inputs={SdfPath("/Dag/RegrowBlend")};
    desc.nodes.insert(desc.nodes.begin(),regrow);
    desc.nodes.push_back(Width("/Dag/RegrowLeft","/Dag/Regrow",.2f));
    desc.nodes.push_back(Width("/Dag/RegrowRight","/Dag/Regrow",.6f));
    UsdGenNodeDesc blend;blend.path=SdfPath("/Dag/RegrowBlend");blend.type=TfToken("UsdGenWidthBlend");
    blend.inputs={SdfPath("/Dag/RegrowLeft"),SdfPath("/Dag/RegrowRight")};
    blend.params.push_back({TfToken("widthBlend:weight"),VtValue(.4f),false});
    desc.nodes.push_back(blend);desc.nodes.push_back(tail);desc.terminal=tail.path;
    return desc;
}

bool TerminalUsesOperator(UsdGenGraphDesc const& desc,TfToken type) {
    SdfPath path=desc.terminal;
    for(size_t depth=0;depth<=desc.nodes.size();++depth) {
        auto node=std::find_if(desc.nodes.begin(),desc.nodes.end(),[&](auto const& n){return n.path==path;});
        if(node==desc.nodes.end()) return false;
        if(node->type==type) return true;
        if(node->inputs.empty()) return false;
        path=node->inputs.front();
    }
    return false;
}

// Feed only the independent CPU Grow implementation with an already-computed
// host predecessor. Noise and Length remain the bounded host oracles below;
// rebuilding this synthetic C3 input must preserve their exact data, not
// rerun the CPU versions of those operators.
bool GrowFromHostValueReference(UsdGenGraphDesc const& desc,UsdGenNodeDesc const& grow,
                               UsdGenCurveBuffer const& input,UsdGenCurveBuffer* output) {
    size_t const curves=input.totalCurves,points=input.totalCvs;
    if(input.px.size()!=points||input.py.size()!=points||input.pz.size()!=points||
       input.rest.size()!=points||input.width.size()!=points||input.curveId.size()!=curves||
       input.rootT.size()!=curves||input.rootB.size()!=curves||input.rootN.size()!=curves||
       input.rootPrim.size()!=curves||input.rootUV.size()!=curves) {
        std::fprintf(stderr,"Grow host shape C=%zu P=%zu xyz=%zu/%zu/%zu rest=%zu width=%zu ids=%zu TBN=%zu/%zu/%zu primUV=%zu/%zu\n",
            curves,points,input.px.size(),input.py.size(),input.pz.size(),input.rest.size(),input.width.size(),input.curveId.size(),
            input.rootT.size(),input.rootB.size(),input.rootN.size(),input.rootPrim.size(),input.rootUV.size());
        return false;
    }
    UsdGenCurveSetDesc sourceData;sourceData.path=SdfPath("/Oracle/Input");
    sourceData.role=UsdGenRole::Curves;sourceData.curveRole=TfToken("hair");
    sourceData.points.resize(points);sourceData.curveVertexCounts.resize(curves);
    if(input.cvOffsets.empty()) {
        if(curves ? points%curves!=0 : points!=0) return false;
        for(size_t c=0;c<curves;++c) sourceData.curveVertexCounts[c]=int(points/curves);
    } else {
        if(input.cvOffsets.size()!=curves+1||input.cvOffsets.front()!=0||
           input.cvOffsets.back()!=int(points)) return false;
        for(size_t c=0;c<curves;++c) {
            if(input.cvOffsets[c+1]<=input.cvOffsets[c]) return false;
            sourceData.curveVertexCounts[c]=input.cvOffsets[c+1]-input.cvOffsets[c];
        }
    }
    for(size_t p=0;p<points;++p) sourceData.points[p]=GfVec3f(input.px[p],input.py[p],input.pz[p]);
    sourceData.rest=input.rest;sourceData.widths=input.width;sourceData.curveId=input.curveId;
    sourceData.skinPrim=input.rootPrim;sourceData.skinPrimUv=input.rootUV;
    sourceData.rootFrame.resize(curves);
    for(size_t c=0;c<curves;++c) {
        GfMatrix4d frame(1.0);
        frame.SetRow3(0,GfVec3d(input.rootT[c]));frame.SetRow3(1,GfVec3d(input.rootB[c]));
        frame.SetRow3(2,GfVec3d(input.rootN[c]));sourceData.rootFrame[c]=frame;
    }
    for(auto const* planes:{&input.extraCv,&input.extraCurve}) for(auto const& plane:*planes) {
        UsdGenAuthoredPlaneDesc authored;authored.name=plane.name;authored.arity=plane.arity;
        if(plane.interpolation==TfToken("vertex")) authored.domain=UsdGenAuthoredPlaneDomain::Point;
        else if(plane.interpolation==TfToken("uniform")) authored.domain=UsdGenAuthoredPlaneDomain::Primitive;
        else if(plane.interpolation==TfToken("constant")) authored.domain=UsdGenAuthoredPlaneDomain::Groom;
        else return false;
        if(plane.type==TfToken("float")) {authored.type=UsdGenAuthoredPlaneType::Float32;authored.floatValues=plane.f;}
        else if(plane.type==TfToken("int")) {authored.type=UsdGenAuthoredPlaneType::Int32;authored.intValues=plane.i;}
        else return false;
        sourceData.authoredPlanes.push_back(std::move(authored));
    }
    auto stage=desc;stage.curveSets={sourceData};
    UsdGenNodeDesc source;source.path=SdfPath("/Oracle/Source");source.type=TfToken("UsdGenCurveSource");
    source.curves={sourceData.path};source.params={{TfToken("useRest"),VtValue(false),false},
                                                  {TfToken("rebind"),VtValue(TfToken("never")),false}};
    auto node=grow;node.inputs={source.path};stage.nodes={source,node};stage.terminal=node.path;
    return CpuReference(stage,output);
}

// Memoized host values follow explicit edges, not authored vector order.
// Noise uses the independent vector/TBN oracle; WidthBlend requires matching
// non-width provenance in this fixture, so its point data is shared exactly.
bool NoiseDagReference(UsdGenGraphDesc const& desc,UsdGenCurveBuffer* output) {
    auto source=std::find_if(desc.nodes.begin(),desc.nodes.end(),[](auto const& n) {
        return n.type==TfToken("UsdGenCurveSource");
    });
    if(source==desc.nodes.end()) return false;
    auto sourceDesc=desc;sourceDesc.nodes={*source};sourceDesc.terminal=source->path;
    UsdGenCurveBuffer sourceValue;if(!CpuReference(sourceDesc,&sourceValue)) return false;
    std::map<SdfPath,UsdGenCurveBuffer> values;values[source->path]=sourceValue;
    std::function<bool(SdfPath const&,size_t)> evaluate=[&](SdfPath const& path,size_t depth) {
        if(values.count(path)) return true;
        if(depth>desc.nodes.size()) return false;
        auto node=std::find_if(desc.nodes.begin(),desc.nodes.end(),[&](auto const& n){return n.path==path;});
        if(node==desc.nodes.end()||node->inputs.empty()||!evaluate(node->inputs[0],depth+1)) return false;
        auto value=values.at(node->inputs[0]);
        if(node->type==TfToken("UsdGenNoise")) {
            auto stage=desc;stage.nodes={*node};
            if(!GrowNoiseReference(stage,&value,true)) {
                std::fprintf(stderr,"Noise host oracle failed at %s\n",node->path.GetText());
                return false;
            }
        } else if(node->type==TfToken("UsdGenGrow")) {
            if(node->inputs.size()!=1) return false;
            if(node->inputs[0]==source->path) {
                auto stage=desc;stage.nodes={*source,*node};stage.terminal=node->path;
                if(!CpuReference(stage,&value)) return false;
            } else {
                UsdGenCurveBuffer grown;
                if(!GrowFromHostValueReference(desc,*node,value,&grown)) {
                    std::fprintf(stderr,"Grow host oracle failed at %s offsets=%zu C=%u P=%u\n",
                        node->path.GetText(),value.cvOffsets.size(),value.totalCurves,value.totalCvs);
                    return false;
                }
                value=std::move(grown);
            }
        } else if(node->type==TfToken("UsdGenLength")) {
            auto stage=desc;stage.nodes={*node};usdGenTest::NodeParams p(stage.nodes.front());
            bool const cut=p.GetToken(TfToken("length:method"),TfToken("scale"))==TfToken("cutExtend");
            if(!GrowLengthScaleReference(stage,float(p.GetDouble(TfToken("length:value"),1)),&value,cut,true)) {
                std::fprintf(stderr,"Length host oracle failed at %s\n",node->path.GetText());
                return false;
            }
        } else if(node->type==TfToken("UsdGenWidth")) {
            usdGenTest::NodeParams p(*node);
            std::fill(value.width.begin(),value.width.end(),float(p.GetDouble(TfToken("width"),0)));
        } else if(node->type==TfToken("UsdGenWidthBlend")) {
            if(node->inputs.size()!=2||!evaluate(node->inputs[1],depth+1)) return false;
            auto const& right=values.at(node->inputs[1]);
            if(value.px!=right.px||value.py!=right.py||value.pz!=right.pz||value.width.size()!=right.width.size()) return false;
            usdGenTest::NodeParams const wb(*node);
            float const weight=float(wb.GetDouble(TfToken("widthBlend:weight"),1.0));
            for(size_t i=0;i<value.width.size();++i)
                value.width[i]=value.width[i]+(right.width[i]-value.width[i])*weight;
        } else return false;
        values[path]=std::move(value);return true;
    };
    if(!evaluate(desc.terminal,0)) return false;
    *output=values.at(desc.terminal);return true;
}

bool CheckNamed(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                UsdGenCurveBuffer const& reference, cudaStream_t stream) {
    for (auto const* planes : {&reference.extraCv,&reference.extraCurve}) for(auto const& plane : *planes) {
        auto channel=gpu::AcquireNamedChannel(generation,plane.name.GetString(),stream);
        if(!channel) return false;
        auto bytes=channel.Bytes();
        if(plane.type==TfToken("float")) {
            std::vector<float> values(plane.f.size());
            if(bytes.size!=values.size()*sizeof(float) ||
               (bytes.size&&cudaMemcpyAsync(values.data(),bytes.data,bytes.size,
                   cudaMemcpyDeviceToHost,stream)!=cudaSuccess) ||
               cudaStreamSynchronize(stream)!=cudaSuccess) return false;
            for(size_t i=0;i<values.size();++i) if(!Near(values[i],plane.f[i])) return false;
        } else {
            std::vector<int> values(plane.i.size());
            if(bytes.size!=values.size()*sizeof(int) ||
               (bytes.size&&cudaMemcpyAsync(values.data(),bytes.data,bytes.size,cudaMemcpyDeviceToHost,stream)!=cudaSuccess) ||
               cudaStreamSynchronize(stream)!=cudaSuccess) return false;
            if(values!=std::vector<int>(plane.i.begin(),plane.i.end())) return false;
        }
    }
    return true;
}

bool CheckGeneration(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                     UsdGenCurveBuffer const& reference, cudaStream_t stream,
                     bool sourceTerminal = false) {
    auto lease=gpu::AcquireGeometry(generation,stream);
    if(!lease||lease.Geometry().curveCount!=reference.totalCurves||
       lease.Geometry().pointCount!=reference.totalCvs) {
        std::fprintf(stderr,"generation shape: lease=%d curves=%zu/%u points=%zu/%u\n",
            int(bool(lease)),lease.Geometry().curveCount,reference.totalCurves,
            lease.Geometry().pointCount,reference.totalCvs);
        return false;
    }
    std::vector<float3> points,rest,t,b,n;
    std::vector<float> widths,hair;
    std::vector<uint64_t> ids;
    std::vector<uint32_t> offsets;
    std::vector<int32_t> prim;
    std::vector<float2> uv;
    if(!Read(lease.Geometry().points,&points,stream)||!Read(lease.Geometry().restPoints,&rest,stream)||
       !Read(lease.Geometry().widths,&widths,stream)||!Read(lease.HairT(),&hair,stream)||
       !Read(lease.Geometry().stableIds,&ids,stream)||!Read(lease.Geometry().curveOffsets,&offsets,stream)||
       !Read(lease.RootPrim(),&prim,stream)||!Read(lease.RootUV(),&uv,stream)||
       !Read(lease.RootT(),&t,stream)||!Read(lease.RootB(),&b,stream)||!Read(lease.RootN(),&n,stream)) {
        std::fprintf(stderr,"generation read failed\n"); return false;
    }
    if(ids.size()!=reference.curveId.size()||offsets.size()!=reference.totalCurves+1||
       prim.size()!=reference.rootPrim.size()||uv.size()!=reference.rootUV.size()||
       t.size()!=(sourceTerminal?0:reference.rootT.size())||
       b.size()!=(sourceTerminal?0:reference.rootB.size())||
       n.size()!=(sourceTerminal?0:reference.rootN.size())) {
        std::fprintf(stderr,"generation channel cardinality failed\n"); return false;
    }
    std::vector<uint32_t> expectedOffsets(reference.totalCurves+1,0);
    if (!reference.cvOffsets.empty()) {
        if (reference.cvOffsets.size()!=expectedOffsets.size()) return false;
        for(size_t c=0;c<expectedOffsets.size();++c) {
            if(reference.cvOffsets[c]<0) return false;
            expectedOffsets[c]=uint32_t(reference.cvOffsets[c]);
        }
    } else if (reference.totalCurves) {
        if(reference.totalCvs%reference.totalCurves) return false;
        auto const cvsPerCurve=reference.totalCvs/reference.totalCurves;
        for(size_t c=0;c<expectedOffsets.size();++c) expectedOffsets[c]=uint32_t(c*cvsPerCurve);
    }
    for(size_t c=0;c<ids.size();++c) {
        if(ids[c]!=reference.curveId[c]||offsets[c]!=expectedOffsets[c]||prim[c]!=reference.rootPrim[c]||
           !Near(uv[c].x,reference.rootUV[c][0])||!Near(uv[c].y,reference.rootUV[c][1])||
           (!sourceTerminal && (!Near(t[c],reference.rootT[c])||!Near(b[c],reference.rootB[c])||!Near(n[c],reference.rootN[c])))) {
            std::fprintf(stderr,"generation curve mismatch %zu\n",c); return false;
        }
    }
    if(offsets!=expectedOffsets||offsets.back()!=points.size()) {
        std::fprintf(stderr,"generation offsets failed\n"); return false;
    }
    for(size_t i=0;i<points.size();++i)
        if(!Near(points[i],GfVec3f(reference.px[i],reference.py[i],reference.pz[i]))||
           !Near(rest[i],reference.rest[i])||!Near(widths[i],reference.width[i])||
           !Near(hair[i],reference.hairT[i])) {
            std::fprintf(stderr,"generation CV mismatch %zu point=(%.7g %.7g %.7g) expected=(%.7g %.7g %.7g) width=%.7g expected=%.7g\n",
                i,points[i].x,points[i].y,points[i].z,reference.px[i],reference.py[i],reference.pz[i],widths[i],reference.width[i]);
            return false;
        }
    bool const named=CheckNamed(generation,reference,stream);
    if(!named) std::fprintf(stderr,"generation named plane mismatch\n");
    return named;
}

bool CheckEndpointLengths(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                          UsdGenCurveBuffer const& reference, cudaStream_t stream) {
    auto lease=gpu::AcquireGeometry(generation,stream);
    std::vector<float3> points;
    std::vector<uint32_t> offsets;
    if(!lease||!Read(lease.Geometry().points,&points,stream)||
       !Read(lease.Geometry().curveOffsets,&offsets,stream)||
       offsets.size()!=size_t(reference.totalCurves)+1) return false;
    for(uint32_t curve=0;curve!=reference.totalCurves;++curve) {
        uint32_t const begin=offsets[curve], end=offsets[curve+1];
        if(begin>=end||end>points.size()) return false;
        uint32_t refBegin=0,refEnd=0;
        if(reference.cvOffsets.empty()) {
            if(!reference.totalCurves||reference.totalCvs%reference.totalCurves) return false;
            uint32_t const perCurve=reference.totalCvs/reference.totalCurves;
            refBegin=curve*perCurve; refEnd=refBegin+perCurve;
        } else {
            if(reference.cvOffsets.size()!=size_t(reference.totalCurves)+1||
               reference.cvOffsets[curve]<0||reference.cvOffsets[curve+1]<=reference.cvOffsets[curve]) return false;
            refBegin=uint32_t(reference.cvOffsets[curve]); refEnd=uint32_t(reference.cvOffsets[curve+1]);
        }
        auto length=[](float x,float y,float z) { return std::sqrt(x*x+y*y+z*z); };
        auto const first=points[begin], last=points[end-1];
        float const actual=length(last.x-first.x,last.y-first.y,last.z-first.z);
        float const expected=length(reference.px[refEnd-1]-reference.px[refBegin],
            reference.py[refEnd-1]-reference.py[refBegin],reference.pz[refEnd-1]-reference.pz[refBegin]);
        if(!Near(actual,expected)) return false;
    }
    return true;
}

bool CheckNoNamed(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                  cudaStream_t stream) {
    return !gpu::AcquireNamedChannel(generation,"sourceValue",stream) &&
           !gpu::AcquireNamedChannel(generation,"sourceGroup",stream);
}

bool CudaRejected(UsdGenGraphDesc const& desc) {
    UsdGenDiagnostics diagnostics;
    auto plan=CompileCudaGraph(desc,&diagnostics);
    if (plan || !diagnostics.HasErrors()) {
        std::fprintf(stderr,"invalid CUDA composition was admitted\n");
        return false;
    }
    return true;
}

// Cross-origin joins are structurally admitted; unequal non-width payloads
// must instead fail the execution-time GPU proof without publishing a value.
bool CudaNonWidthRejected(UsdGenGraphDesc const& desc) {
    UsdGenDiagnostics diagnostics;
    auto plan=CompileCudaGraph(desc,&diagnostics);
    if(!plan || diagnostics.HasErrors()) return false;
    UsdGenSession session;
    session.SetDevicePublicationEnabled(true);
    session.SetGraphDesc(desc);
    auto result=session.Commit(1,UsdGenCommitReason::SetTime);
    return (!result || !result->device) && session.LastDiagnostics().HasErrors();
}
}

int main() {
    int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess||!count) return 77;
    cudaStream_t stream=nullptr; CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
    auto desc=Desc(); UsdGenCurveBuffer reference; CHECK(CpuReference(desc,&reference));
    CHECK(reference.curveId==VtArray<uint64_t>({7,40})&&reference.totalCvs==8);
    CHECK(reference.rest[0][0]!=reference.px[0]);
    UsdGenDiagnostics diagnostics; auto plan=CompileCudaGraph(desc,&diagnostics);
    for(auto const& error:diagnostics.errors) std::fprintf(stderr,"%s\n",error.c_str());
    CHECK(plan&&!diagnostics.HasErrors());
    auto metadata=GetCudaExecutionPlanMetadata(*plan);
    CHECK(metadata&&metadata->Tasks().size()>=4&&metadata->Operators().size()==desc.nodes.size());
    UsdGenSession session; session.SetDevicePublicationEnabled(true); session.SetGraphDesc(desc);
    auto first=session.Commit(1,UsdGenCommitReason::SetTime);
    for(auto const& error:session.LastDiagnostics().errors) std::fprintf(stderr,"%s\n",error.c_str());
    CHECK(first&&first->device&&!session.LastDiagnostics().HasErrors());
    CHECK(CheckGeneration(first->device,reference,stream));
    auto old=gpu::AcquireGeometry(first->device,stream); CHECK(old);
    std::vector<float3> oldPoints; std::vector<float> oldWidths;
    CHECK(Read(old.Geometry().points,&oldPoints,stream)&&Read(old.Geometry().widths,&oldWidths,stream));

    auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(workspace);
    auto direct=ExecuteCudaGraph(*plan,*workspace,1,100,&diagnostics);
    CHECK(direct&&!diagnostics.HasErrors()&&CheckGeneration(direct,reference,stream));

    // Public callers are allowed to use an asynchronous native source then
    // synchronous operator dispatch. Keep this path distinct from the fully
    // asynchronous relay test below: both must finalize the same generation.
    diagnostics={};
    auto mixedWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(mixedWorkspace);
    auto mixedJob=CreateCudaExecutionJob(plan,*mixedWorkspace,1,101,&diagnostics,first->device); CHECK(mixedJob);
    std::promise<bool> mixedSourcePromise; auto mixedSourceFuture=mixedSourcePromise.get_future();
    CHECK(ExecuteCudaJobSourceAsync(mixedJob,[&](bool ok){mixedSourcePromise.set_value(ok);}));
    CHECK(mixedSourceFuture.wait_for(std::chrono::seconds(10))==std::future_status::ready);
    bool const mixedSourceOk=mixedSourceFuture.get();
    if(!mixedSourceOk) for(auto const& error:diagnostics.errors) std::fprintf(stderr,"mixed source: %s\n",error.c_str());
    CHECK(mixedSourceOk&&!diagnostics.HasErrors());
    for(size_t i=0;i<CudaExecutionJobOperatorCount(*mixedJob);++i) {
        bool const operatorOk=ExecuteCudaJobOperator(*mixedJob,i);
        if(!operatorOk) for(auto const& error:diagnostics.errors) std::fprintf(stderr,"mixed operator %zu: %s\n",i,error.c_str());
        CHECK(operatorOk&&!diagnostics.HasErrors());
    }
    std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> mixedFinalPromise;
    auto mixedFinalFuture=mixedFinalPromise.get_future();
    bool const mixedFinalStarted=FinalizeCudaExecutionJobAsync(mixedJob,[&](auto result) {
        mixedFinalPromise.set_value(std::move(result));
    });
    if(!mixedFinalStarted) for(auto const& error:diagnostics.errors) std::fprintf(stderr,"mixed final start: %s\n",error.c_str());
    CHECK(mixedFinalStarted&&mixedFinalFuture.wait_for(std::chrono::seconds(10))==std::future_status::ready);
    auto mixed=mixedFinalFuture.get();
    if(!mixed) {
        std::fprintf(stderr,"mixed final null: poisoned=%d\n",int(mixedWorkspace->IsPoisoned()));
        for(auto const& error:diagnostics.errors) std::fprintf(stderr,"mixed final: %s\n",error.c_str());
    }
    CHECK(mixed&&CheckGeneration(mixed,reference,stream));

    diagnostics={};
    auto asyncWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(asyncWorkspace);
    auto job=CreateCudaExecutionJob(plan,*asyncWorkspace,1,102,&diagnostics,first->device); CHECK(job);
    std::promise<bool> sourcePromise; auto sourceFuture=sourcePromise.get_future();
    CHECK(ExecuteCudaJobSourceAsync(job,[&](bool ok){sourcePromise.set_value(ok);}));
    CHECK(sourceFuture.wait_for(std::chrono::seconds(10))==std::future_status::ready);
    bool const sourceOk=sourceFuture.get();
    if(!sourceOk) for(auto const& error:diagnostics.errors) std::fprintf(stderr,"async source: %s\n",error.c_str());
    CHECK(sourceOk&&!diagnostics.HasErrors());
    for(size_t i=0;i<CudaExecutionJobOperatorCount(*job);++i) {
        std::promise<bool> operatorPromise;
        auto operatorFuture=operatorPromise.get_future();
        bool const started=ExecuteCudaJobOperatorAsync(job,i,[&](bool ok) {
            operatorPromise.set_value(ok);
        });
        if(!started || operatorFuture.wait_for(std::chrono::seconds(10))!=std::future_status::ready)
            for(auto const& error:diagnostics.errors) std::fprintf(stderr,"async operator %zu: %s\n",i,error.c_str());
        CHECK(started&&operatorFuture.wait_for(std::chrono::seconds(0))==std::future_status::ready);
        bool const operatorOk=operatorFuture.get();
        if(!operatorOk) for(auto const& error:diagnostics.errors) std::fprintf(stderr,"async operator %zu: %s\n",i,error.c_str());
        CHECK(operatorOk&&!diagnostics.HasErrors());
    }
    std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> finalPromise;
    auto finalFuture=finalPromise.get_future();
    CHECK(FinalizeCudaExecutionJobAsync(job,[&](auto result){finalPromise.set_value(std::move(result));}));
    CHECK(finalFuture.wait_for(std::chrono::seconds(10))==std::future_status::ready);
    auto async=finalFuture.get();
    if(!async) for(auto const& error:diagnostics.errors) std::fprintf(stderr,"async final: %s\n",error.c_str());
    CHECK(async&&CheckGeneration(async,reference,stream));

    auto edited=desc;
    auto left=std::find_if(edited.nodes.begin(),edited.nodes.end(),[](auto const& node){return node.path==SdfPath("/Ops/Left");});
    CHECK(left!=edited.nodes.end()); left->params[0].value=VtValue(.6f);
    CHECK(CpuReference(edited,&reference)); session.SetGraphDesc(edited);
    auto second=session.Commit(2,UsdGenCommitReason::SetTime);
    CHECK(second&&second->device&&!session.LastDiagnostics().HasErrors()&&CheckGeneration(second->device,reference,stream));
    auto current=gpu::AcquireGeometry(second->device,stream); CHECK(current);
    CHECK(current.Geometry().widths.data!=old.Geometry().widths.data);
    std::vector<float3> points; std::vector<float> widths;
    CHECK(Read(old.Geometry().points,&points,stream)&&
          points.size()==oldPoints.size()&&
          std::memcmp(points.data(),oldPoints.data(),points.size()*sizeof(float3))==0);
    CHECK(Read(old.Geometry().widths,&widths,stream)&&widths==oldWidths);

    // Grow is a native topology producer, while Noise is a value writer on
    // the generated C3.  Keep the authored storage shuffled and verify the
    // complete suffix against the CPU oracle through direct and Session
    // publication routes.  The posed source points remain distinct from
    // rest, so Noise must use the retained rest snapshot rather than silently
    // treating the generated current points as its rest input.
    auto growNoise=Desc(false,true,false,true);
    auto importedClumpNoise = growNoise;
    UsdGenAuthoredPlaneDesc clumpWeight;
    clumpWeight.name = TfToken("clumpWeight_2");
    clumpWeight.type = UsdGenAuthoredPlaneType::Float32;
    clumpWeight.domain = UsdGenAuthoredPlaneDomain::Point;
    clumpWeight.arity = 1;
    clumpWeight.floatValues = {0.f, 1.f, 1.f, 0.f, 1.f};
    importedClumpNoise.curveSets.front().authoredPlanes.push_back(clumpWeight);
    UsdGenDiagnostics importedClumpDiagnostics;
    CHECK(!CompileCudaGraph(importedClumpNoise, &importedClumpDiagnostics));
    CHECK(std::any_of(importedClumpDiagnostics.errors.begin(),
                      importedClumpDiagnostics.errors.end(),
        [](std::string const &error) {
            return error.find("CUDA Noise does not support imported clump motion planes") !=
                std::string::npos;
        }));
    UsdGenCurveBuffer growNoiseReference;
    CHECK(GrowNoiseReference(growNoise,&growNoiseReference));
    UsdGenDiagnostics growNoiseDiagnostics;
    auto growNoisePlan=CompileCudaGraph(growNoise,&growNoiseDiagnostics);
    for(auto const& error:growNoiseDiagnostics.errors)
        std::fprintf(stderr,"grow-noise compile: %s\n",error.c_str());
    CHECK(growNoisePlan&&!growNoiseDiagnostics.HasErrors());
    auto growNoiseWorkspace=CreateCudaExecutionWorkspace(-1,&growNoiseDiagnostics);
    CHECK(growNoiseWorkspace);
    auto growNoiseDirect=ExecuteCudaGraph(*growNoisePlan,*growNoiseWorkspace,1,150,
                                          &growNoiseDiagnostics);
    for(auto const& error:growNoiseDiagnostics.errors)
        std::fprintf(stderr,"grow-noise direct: %s\n",error.c_str());
    CHECK(growNoiseDirect&&!growNoiseDiagnostics.HasErrors()&&
          CheckGeneration(growNoiseDirect,growNoiseReference,stream));
    // Exercise the staged relay API as well: Grow's topology completion must
    // hand its generated rest/frame inputs to Noise before the final Width
    // publication, without a host-side synchronization between operators.
    auto growNoiseAsyncWorkspace=CreateCudaExecutionWorkspace(-1,
        &growNoiseDiagnostics);
    CHECK(growNoiseAsyncWorkspace);
    auto growNoiseJob=CreateCudaExecutionJob(growNoisePlan,
        *growNoiseAsyncWorkspace,1,151,&growNoiseDiagnostics);
    CHECK(growNoiseJob);
    std::promise<bool> growNoiseSourcePromise;
    auto growNoiseSourceFuture=growNoiseSourcePromise.get_future();
    CHECK(ExecuteCudaJobSourceAsync(growNoiseJob,[&](bool ok) {
        growNoiseSourcePromise.set_value(ok);
    }));
    CHECK(growNoiseSourceFuture.wait_for(std::chrono::seconds(10))==
          std::future_status::ready&&growNoiseSourceFuture.get());
    for(size_t i=0;i<CudaExecutionJobOperatorCount(*growNoiseJob);++i) {
        std::promise<bool> operatorPromise;
        auto operatorFuture=operatorPromise.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(growNoiseJob,i,[&](bool ok) {
            operatorPromise.set_value(ok);
        }));
        CHECK(operatorFuture.wait_for(std::chrono::seconds(10))==
              std::future_status::ready&&operatorFuture.get());
    }
    std::promise<std::shared_ptr<const UsdGenDeviceGeneration>>
        growNoiseFinalPromise;
    auto growNoiseFinalFuture=growNoiseFinalPromise.get_future();
    CHECK(FinalizeCudaExecutionJobAsync(growNoiseJob,[&](auto result) {
        growNoiseFinalPromise.set_value(std::move(result));
    }));
    CHECK(growNoiseFinalFuture.wait_for(std::chrono::seconds(10))==
          std::future_status::ready);
    auto growNoiseAsync=growNoiseFinalFuture.get();
    CHECK(growNoiseAsync&&!growNoiseDiagnostics.HasErrors()&&
          CheckGeneration(growNoiseAsync,growNoiseReference,stream));
    auto resampledGrowNoise=Desc(true,true,false,true);
    UsdGenCurveBuffer resampledGrowNoiseReference;
    CHECK(GrowNoiseReference(resampledGrowNoise,&resampledGrowNoiseReference));
    growNoiseDiagnostics={};
    auto resampledGrowNoisePlan=CompileCudaGraph(resampledGrowNoise,
                                                  &growNoiseDiagnostics);
    CHECK(resampledGrowNoisePlan&&!growNoiseDiagnostics.HasErrors());
    auto resampledGrowNoiseDirect=ExecuteCudaGraph(*resampledGrowNoisePlan,
        *growNoiseWorkspace,1,152,&growNoiseDiagnostics);
    CHECK(resampledGrowNoiseDirect&&!growNoiseDiagnostics.HasErrors()&&
          CheckGeneration(resampledGrowNoiseDirect,resampledGrowNoiseReference,
                          stream));
    auto generatedRestGrowNoise=Desc(false,true,false,true);
    generatedRestGrowNoise.curveSets.front().rest.clear();
    UsdGenCurveBuffer generatedRestReference;
    CHECK(GrowNoiseReference(generatedRestGrowNoise,&generatedRestReference));
    growNoiseDiagnostics={};
    auto generatedRestPlan=CompileCudaGraph(generatedRestGrowNoise,
                                             &growNoiseDiagnostics);
    CHECK(generatedRestPlan&&!growNoiseDiagnostics.HasErrors());
    auto generatedRestDirect=ExecuteCudaGraph(*generatedRestPlan,
        *growNoiseWorkspace,1,153,&growNoiseDiagnostics);
    CHECK(generatedRestDirect&&!growNoiseDiagnostics.HasErrors()&&
          CheckGeneration(generatedRestDirect,generatedRestReference,stream));
    UsdGenSession generatedRestSession;
    generatedRestSession.SetDevicePublicationEnabled(true);
    generatedRestSession.SetGraphDesc(generatedRestGrowNoise);
    auto generatedRestPublished=generatedRestSession.Commit(1,
        UsdGenCommitReason::SetTime);
    CHECK(generatedRestPublished&&generatedRestPublished->device&&
          !generatedRestSession.LastDiagnostics().HasErrors()&&
          CheckGeneration(generatedRestPublished->device,generatedRestReference,
                          stream));
    // Exercise both source resampling directions through the same complete
    // Grow→Noise suffix, and retain the Session result independently of the
    // direct execution result.
    for (int resampleTo : {3, 2}) {
        auto resampled=Desc(false,true,false,true);
        auto resampleNode=std::find_if(resampled.nodes.begin(),resampled.nodes.end(),
            [](auto const& node){return node.type==TfToken("UsdGenCurveSource");});
        CHECK(resampleNode!=resampled.nodes.end());
        resampleNode->params.push_back({TfToken("resampleTo"),VtValue(resampleTo),false});
        UsdGenCurveBuffer expected;
        CHECK(GrowNoiseReference(resampled,&expected));
        growNoiseDiagnostics={};
        auto resampledPlan=CompileCudaGraph(resampled,&growNoiseDiagnostics);
        CHECK(resampledPlan&&!growNoiseDiagnostics.HasErrors());
        auto directResampled=ExecuteCudaGraph(*resampledPlan,*growNoiseWorkspace,
            1,154+resampleTo,&growNoiseDiagnostics);
        CHECK(directResampled&&!growNoiseDiagnostics.HasErrors()&&
              CheckGeneration(directResampled,expected,stream));
        UsdGenSession resampledSession;
        resampledSession.SetDevicePublicationEnabled(true);
        resampledSession.SetGraphDesc(resampled);
        auto publishedResampled=resampledSession.Commit(1,UsdGenCommitReason::SetTime);
        CHECK(publishedResampled&&publishedResampled->device&&
              !resampledSession.LastDiagnostics().HasErrors()&&
              CheckGeneration(publishedResampled->device,expected,stream));
    }
    auto emptyGrowNoise=Desc(false,false,true,true);
    UsdGenCurveBuffer emptyGrowNoiseReference;
    CHECK(GrowNoiseReference(emptyGrowNoise,&emptyGrowNoiseReference));
    growNoiseDiagnostics={};
    auto emptyGrowNoisePlan=CompileCudaGraph(emptyGrowNoise,&growNoiseDiagnostics);
    CHECK(emptyGrowNoisePlan&&!growNoiseDiagnostics.HasErrors());
    auto emptyGrowNoiseDirect=ExecuteCudaGraph(*emptyGrowNoisePlan,
        *growNoiseWorkspace,1,158,&growNoiseDiagnostics);
    CHECK(emptyGrowNoiseDirect&&!growNoiseDiagnostics.HasErrors()&&
          CheckGeneration(emptyGrowNoiseDirect,emptyGrowNoiseReference,stream));
    UsdGenSession emptyGrowNoiseSession;
    emptyGrowNoiseSession.SetDevicePublicationEnabled(true);
    emptyGrowNoiseSession.SetGraphDesc(emptyGrowNoise);
    auto emptyGrowNoisePublished=emptyGrowNoiseSession.Commit(1,
        UsdGenCommitReason::SetTime);
    CHECK(emptyGrowNoisePublished&&emptyGrowNoisePublished->device&&
          !emptyGrowNoiseSession.LastDiagnostics().HasErrors()&&
          CheckGeneration(emptyGrowNoisePublished->device,
                          emptyGrowNoiseReference,stream));
    UsdGenSession growNoiseSession;
    growNoiseSession.SetDevicePublicationEnabled(true);
    growNoiseSession.SetGraphDesc(growNoise);
    auto growNoisePublished=growNoiseSession.Commit(1,UsdGenCommitReason::SetTime);
    for(auto const& error:growNoiseSession.LastDiagnostics().errors)
        std::fprintf(stderr,"grow-noise session: %s\n",error.c_str());
    CHECK(growNoisePublished&&growNoisePublished->device&&
          !growNoiseSession.LastDiagnostics().HasErrors()&&
          CheckGeneration(growNoisePublished->device,growNoiseReference,stream));
    auto retainedGrowNoise=growNoisePublished->device;
    auto retainedGrowNoiseLease=gpu::AcquireGeometry(retainedGrowNoise,stream);
    CHECK(retainedGrowNoiseLease);
    std::vector<float3> retainedGrowNoisePoints, retainedGrowNoiseRest;
    std::vector<float> retainedGrowNoiseWidths;
    CHECK(Read(retainedGrowNoiseLease.Geometry().points,&retainedGrowNoisePoints,stream)&&
          Read(retainedGrowNoiseLease.Geometry().restPoints,&retainedGrowNoiseRest,stream)&&
          Read(retainedGrowNoiseLease.Geometry().widths,&retainedGrowNoiseWidths,stream));
    auto editedGrowNoise=growNoise;
    auto editedNoise=std::find_if(editedGrowNoise.nodes.begin(),editedGrowNoise.nodes.end(),
        [](auto const& node){return node.path==SdfPath("/Ops/Noise");});
    CHECK(editedNoise!=editedGrowNoise.nodes.end());
    editedNoise->params[3].value=VtValue(.16f);
    UsdGenCurveBuffer editedGrowNoiseReference;
    CHECK(GrowNoiseReference(editedGrowNoise,&editedGrowNoiseReference));
    growNoiseSession.SetGraphDesc(editedGrowNoise);
    auto editedGrowNoisePublished=growNoiseSession.Commit(2,UsdGenCommitReason::SetTime);
    CHECK(editedGrowNoisePublished&&editedGrowNoisePublished->device&&
          !growNoiseSession.LastDiagnostics().HasErrors()&&
          CheckGeneration(editedGrowNoisePublished->device,editedGrowNoiseReference,stream));
    auto editedGrowNoiseLease=gpu::AcquireGeometry(editedGrowNoisePublished->device,stream);
    CHECK(editedGrowNoiseLease&&editedGrowNoiseLease.Geometry().points.data !=
          retainedGrowNoiseLease.Geometry().points.data&&
          editedGrowNoiseLease.Geometry().restPoints.data !=
          retainedGrowNoiseLease.Geometry().restPoints.data&&
          editedGrowNoiseLease.Geometry().widths.data !=
          retainedGrowNoiseLease.Geometry().widths.data);
    std::vector<float3> editedGrowNoisePoints, editedGrowNoiseRest;
    std::vector<float> editedGrowNoiseWidths;
    std::vector<float3> retainedAfterEditPoints, retainedAfterEditRest;
    std::vector<float> retainedAfterEditWidths;
    CHECK(Read(editedGrowNoiseLease.Geometry().points,&editedGrowNoisePoints,stream)&&
          Read(editedGrowNoiseLease.Geometry().restPoints,&editedGrowNoiseRest,stream)&&
          Read(editedGrowNoiseLease.Geometry().widths,&editedGrowNoiseWidths,stream)&&
          Read(retainedGrowNoiseLease.Geometry().points,&retainedAfterEditPoints,stream)&&
          Read(retainedGrowNoiseLease.Geometry().restPoints,&retainedAfterEditRest,stream)&&
          Read(retainedGrowNoiseLease.Geometry().widths,&retainedAfterEditWidths,stream));
    CHECK(std::memcmp(retainedGrowNoisePoints.data(),editedGrowNoisePoints.data(),
                      retainedGrowNoisePoints.size()*sizeof(float3)) != 0 &&
          std::memcmp(retainedGrowNoisePoints.data(),retainedAfterEditPoints.data(),
                      retainedGrowNoisePoints.size()*sizeof(float3)) == 0 &&
          std::memcmp(retainedGrowNoiseRest.data(),retainedAfterEditRest.data(),
                      retainedGrowNoiseRest.size()*sizeof(float3)) == 0 &&
          std::memcmp(retainedGrowNoiseWidths.data(),retainedAfterEditWidths.data(),
                      retainedGrowNoiseWidths.size()*sizeof(float)) == 0);
    CHECK(CheckGeneration(retainedGrowNoise,growNoiseReference,stream));

    // Noise before Grow now consumes the exact transformed predecessor.
    auto noiseBeforeGrow=growNoise;
    auto badGrow=std::find_if(noiseBeforeGrow.nodes.begin(),noiseBeforeGrow.nodes.end(),
        [](auto const& node){return node.path==SdfPath("/Ops/Grow");});
    auto badNoise=std::find_if(noiseBeforeGrow.nodes.begin(),noiseBeforeGrow.nodes.end(),
        [](auto const& node){return node.path==SdfPath("/Ops/Noise");});
    auto badLeft=std::find_if(noiseBeforeGrow.nodes.begin(),noiseBeforeGrow.nodes.end(),
        [](auto const& node){return node.path==SdfPath("/Ops/Left");});
    CHECK(badGrow!=noiseBeforeGrow.nodes.end()&&badNoise!=noiseBeforeGrow.nodes.end());
    CHECK(badLeft!=noiseBeforeGrow.nodes.end());
    auto badSourceNode=*std::find_if(noiseBeforeGrow.nodes.begin(),noiseBeforeGrow.nodes.end(),
        [](auto const& node){return node.path==SdfPath("/Ops/Source");});
    auto badNoiseNode=*badNoise; badNoiseNode.inputs={badSourceNode.path};
    auto badGrowNode=*badGrow; badGrowNode.inputs={badNoiseNode.path};
    auto badLeftNode=*badLeft; badLeftNode.inputs={badGrowNode.path};
    noiseBeforeGrow.nodes={badSourceNode,badNoiseNode,badGrowNode,badLeftNode};
    noiseBeforeGrow.terminal=badLeftNode.path;
    {
        UsdGenCurveBuffer expected;
        CHECK(NoiseDagReference(noiseBeforeGrow,&expected));
        UsdGenDiagnostics reorderDiagnostics;
        auto reorderPlan=CompileCudaGraph(noiseBeforeGrow,&reorderDiagnostics);
        auto reorderWorkspace=CreateCudaExecutionWorkspace(-1,&reorderDiagnostics);
        CHECK(reorderPlan&&reorderWorkspace&&!reorderDiagnostics.HasErrors());
        auto generation=ExecuteCudaGraph(*reorderPlan,*reorderWorkspace,1,169,&reorderDiagnostics);
        CHECK(generation&&!reorderDiagnostics.HasErrors()&&
              CheckGeneration(generation,expected,stream));
    }
    auto growLength=growNoise;
    auto growLengthSource=std::find_if(growLength.nodes.begin(),growLength.nodes.end(),
        [](auto const& node){return node.type==TfToken("UsdGenCurveSource");});
    auto growLengthGrow=std::find_if(growLength.nodes.begin(),growLength.nodes.end(),
        [](auto const& node){return node.type==TfToken("UsdGenGrow");});
    auto growLengthNoise=std::find_if(growLength.nodes.begin(),growLength.nodes.end(),
        [](auto const& node){return node.type==TfToken("UsdGenNoise");});
    CHECK(growLengthSource!=growLength.nodes.end()&&growLengthGrow!=growLength.nodes.end()&&
          growLengthNoise!=growLength.nodes.end());
    UsdGenNodeDesc afterGrowLength;
    afterGrowLength.path=SdfPath("/Ops/AfterGrowLength");
    afterGrowLength.type=TfToken("UsdGenLength");
    afterGrowLength.inputs={growLengthNoise->path};
    afterGrowLength.params={{TfToken("length:mode"),VtValue(TfToken("scale")),false},
                            {TfToken("length:value"),VtValue(.5f),false}};
    auto afterGrowWidth=Width("/Ops/AfterGrowWidth","/Ops/AfterGrowLength",.2f);
    growLength.nodes={*growLengthSource,*growLengthGrow,*growLengthNoise,
                      afterGrowLength,afterGrowWidth};
    growLength.terminal=afterGrowWidth.path;
    // This previously rejected mixed suffix is exercised below.

    // Positive ordered Source→Grow→Length→Width coverage. The host oracle
    // removes Length from the CPU graph, then applies root-locked scale; CPU
    // Length's noncompacting cull behavior is intentionally not used.
    auto positiveLength=GrowLengthDesc(TfToken("scale"),.5f);
    UsdGenCurveBuffer positiveLengthReference;
    CHECK(GrowLengthScaleReference(positiveLength,.5f,&positiveLengthReference));
    UsdGenDiagnostics positiveLengthDiagnostics;
    auto positiveLengthPlan=CompileCudaGraph(positiveLength,&positiveLengthDiagnostics);
    CHECK(positiveLengthPlan&&!positiveLengthDiagnostics.HasErrors());
    auto positiveLengthWorkspace=CreateCudaExecutionWorkspace(-1,&positiveLengthDiagnostics);
    CHECK(positiveLengthWorkspace);
    auto positiveLengthDirect=ExecuteCudaGraph(*positiveLengthPlan,*positiveLengthWorkspace,
        1,170,&positiveLengthDiagnostics);
    CHECK(positiveLengthDirect&&!positiveLengthDiagnostics.HasErrors()&&
          CheckGeneration(positiveLengthDirect,positiveLengthReference,stream));
    UsdGenSession positiveLengthSession;
    positiveLengthSession.SetDevicePublicationEnabled(true);
    positiveLengthSession.SetGraphDesc(positiveLength);
    auto positiveLengthPublished=positiveLengthSession.Commit(1,UsdGenCommitReason::SetTime);
    CHECK(positiveLengthPublished&&positiveLengthPublished->device&&
          !positiveLengthSession.LastDiagnostics().HasErrors()&&
          CheckGeneration(positiveLengthPublished->device,positiveLengthReference,stream));
    auto positiveLengthJob=CreateCudaExecutionJob(positiveLengthPlan,*positiveLengthWorkspace,
        1,171,&positiveLengthDiagnostics);
    CHECK(positiveLengthJob);
    std::promise<bool> positiveSourcePromise; auto positiveSourceFuture=positiveSourcePromise.get_future();
    CHECK(ExecuteCudaJobSourceAsync(positiveLengthJob,[&](bool ok){positiveSourcePromise.set_value(ok);}));
    CHECK(positiveSourceFuture.wait_for(std::chrono::seconds(10))==std::future_status::ready&&positiveSourceFuture.get());
    for(size_t i=0;i<CudaExecutionJobOperatorCount(*positiveLengthJob);++i) {
        std::promise<bool> p; auto f=p.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(positiveLengthJob,i,[&](bool ok){p.set_value(ok);}));
        CHECK(f.wait_for(std::chrono::seconds(10))==std::future_status::ready&&f.get());
    }
    std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> positiveFinalPromise;
    auto positiveFinalFuture=positiveFinalPromise.get_future();
    CHECK(FinalizeCudaExecutionJobAsync(positiveLengthJob,[&](auto result){positiveFinalPromise.set_value(std::move(result));}));
    CHECK(positiveFinalFuture.wait_for(std::chrono::seconds(10))==std::future_status::ready);
    auto positiveLengthAsync=positiveFinalFuture.get();
    CHECK(positiveLengthAsync&&!positiveLengthDiagnostics.HasErrors()&&
          CheckGeneration(positiveLengthAsync,positiveLengthReference,stream));

    // Cut/extend is arc-length sampling of the generated current geometry,
    // not a radial root scale (the source current roots are deliberately
    // displaced from rest in this fixture).
    auto cutExtend=GrowLengthDesc(TfToken("scale"),.5f,true);
    UsdGenCurveBuffer cutExtendReference;
    CHECK(GrowLengthScaleReference(cutExtend,.5f,&cutExtendReference,true));
    positiveLengthDiagnostics={};
    auto cutExtendPlan=CompileCudaGraph(cutExtend,&positiveLengthDiagnostics);
    CHECK(cutExtendPlan&&!positiveLengthDiagnostics.HasErrors());
    auto cutExtendDirect=ExecuteCudaGraph(*cutExtendPlan,*positiveLengthWorkspace,1,172,&positiveLengthDiagnostics);
    CHECK(cutExtendDirect&&!positiveLengthDiagnostics.HasErrors()&&
          CheckGeneration(cutExtendDirect,cutExtendReference,stream));
    UsdGenSession cutExtendSession;
    cutExtendSession.SetDevicePublicationEnabled(true);
    cutExtendSession.SetGraphDesc(cutExtend);
    auto cutExtendPublished=cutExtendSession.Commit(1,UsdGenCommitReason::SetTime);
    CHECK(cutExtendPublished&&cutExtendPublished->device&&
          !cutExtendSession.LastDiagnostics().HasErrors()&&
          CheckGeneration(cutExtendPublished->device,cutExtendReference,stream));
    auto allCull=GrowLengthDesc(TfToken("cull"),1000.0f);
    UsdGenCurveBuffer allCullReference;
    CHECK(GrowLengthScaleReference(allCull,1,&allCullReference));
    UsdGenDiagnostics allCullDiagnostics;
    auto allCullPlan=CompileCudaGraph(allCull,&allCullDiagnostics);
    CHECK(allCullPlan&&!allCullDiagnostics.HasErrors());
    auto allCullDirect=ExecuteCudaGraph(*allCullPlan,*positiveLengthWorkspace,1,173,&allCullDiagnostics);
    CHECK(allCullDirect&&!allCullDiagnostics.HasErrors()&&
          allCullDirect->Geometry().curveCount==0&&allCullDirect->Geometry().pointCount==0&&
          CheckGeneration(allCullDirect,allCullReference,stream));
    // Midpoint threshold is derived from the two deterministic generated
    // strands, proving survivor selection is data-driven rather than a fixed
    // authored-count shortcut.
    UsdGenCurveBuffer grownForCull;
    CHECK(CpuReference(Desc(),&grownForCull));
    float const strand0 = std::hypot(grownForCull.px[3]-grownForCull.px[0],
        std::hypot(grownForCull.py[3]-grownForCull.py[0],grownForCull.pz[3]-grownForCull.pz[0]));
    float const strand1 = std::hypot(grownForCull.px[7]-grownForCull.px[4],
        std::hypot(grownForCull.py[7]-grownForCull.py[4],grownForCull.pz[7]-grownForCull.pz[4]));
    auto partialCull=GrowLengthDesc(TfToken("cull"),(strand0+strand1)*.5f);
    UsdGenCurveBuffer partialReference;
    CHECK(GrowLengthScaleReference(partialCull,1,&partialReference)&&partialReference.totalCurves==1);
    UsdGenDiagnostics partialDiagnostics;
    auto partialPlan=CompileCudaGraph(partialCull,&partialDiagnostics);
    CHECK(partialPlan&&!partialDiagnostics.HasErrors());
    auto partialDirect=ExecuteCudaGraph(*partialPlan,*positiveLengthWorkspace,1,176,&partialDiagnostics);
    CHECK(partialDirect&&!partialDiagnostics.HasErrors()&&partialDirect->Geometry().curveCount==1&&
          partialDirect->Geometry().pointCount==4&&CheckGeneration(partialDirect,partialReference,stream));
    {
        // Retain the full pre-cull generation while the same Session changes
        // cardinality twice. Old COW points/rest/widths must be byte-identical.
        auto held=positiveLengthPublished->device;
        auto heldLease=gpu::AcquireGeometry(held,stream); CHECK(heldLease);
        std::vector<float3> beforePoints,beforeRest;
        std::vector<float> beforeWidths;
        CHECK(Read(heldLease.Geometry().points,&beforePoints,stream)&&
              Read(heldLease.Geometry().restPoints,&beforeRest,stream)&&
              Read(heldLease.Geometry().widths,&beforeWidths,stream));
        positiveLengthSession.SetGraphDesc(partialCull);
        auto partial=positiveLengthSession.Commit(2,UsdGenCommitReason::SetTime);
        CHECK(partial&&partial->device&&!positiveLengthSession.LastDiagnostics().HasErrors()&&
              CheckGeneration(partial->device,partialReference,stream));
        auto partialLease=gpu::AcquireGeometry(partial->device,stream); CHECK(partialLease);
        CHECK(partialLease.Geometry().points.data!=heldLease.Geometry().points.data);
        positiveLengthSession.SetGraphDesc(allCull);
        auto empty=positiveLengthSession.Commit(3,UsdGenCommitReason::SetTime);
        CHECK(empty&&empty->device&&!positiveLengthSession.LastDiagnostics().HasErrors()&&
              CheckGeneration(empty->device,allCullReference,stream)&&
              CheckGeneration(partial->device,partialReference,stream)&&
              CheckGeneration(held,positiveLengthReference,stream));
        std::vector<float3> afterPoints,afterRest;
        std::vector<float> afterWidths;
        CHECK(Read(heldLease.Geometry().points,&afterPoints,stream)&&
              Read(heldLease.Geometry().restPoints,&afterRest,stream)&&
              Read(heldLease.Geometry().widths,&afterWidths,stream));
        CHECK(afterPoints.size()==beforePoints.size()&&afterRest.size()==beforeRest.size()&&
              afterWidths.size()==beforeWidths.size()&&
              std::memcmp(beforePoints.data(),afterPoints.data(),beforePoints.size()*sizeof(float3))==0&&
              std::memcmp(beforeRest.data(),afterRest.data(),beforeRest.size()*sizeof(float3))==0&&
              std::memcmp(beforeWidths.data(),afterWidths.data(),beforeWidths.size()*sizeof(float))==0);
    }
    {
        // A second Length reads survivor-owned frames and keeps the previous
        // compactor alive until its own proof; exercise that admission peak.
        auto repeated=positiveLength;
        auto secondLength=repeated.nodes[2]; secondLength.path=SdfPath("/Ops/SecondLength");
        secondLength.inputs={repeated.nodes[2].path};
        auto terminal=repeated.nodes.back();terminal.inputs={secondLength.path};
        repeated.nodes.pop_back();repeated.nodes.push_back(secondLength);repeated.nodes.push_back(terminal);
        auto quarter=GrowLengthDesc(TfToken("scale"),.25f);
        UsdGenCurveBuffer expected;
        CHECK(GrowLengthScaleReference(quarter,.25f,&expected));
        UsdGenDiagnostics repeatDiagnostics;
        auto repeatPlan=CompileCudaGraph(repeated,&repeatDiagnostics);CHECK(repeatPlan&&!repeatDiagnostics.HasErrors());
        auto repeatDirect=ExecuteCudaGraph(*repeatPlan,*positiveLengthWorkspace,1,180,&repeatDiagnostics);
        CHECK(repeatDirect&&!repeatDiagnostics.HasErrors()&&CheckGeneration(repeatDirect,expected,stream));
        UsdGenSession repeatSession;repeatSession.SetDevicePublicationEnabled(true);repeatSession.SetGraphDesc(repeated);
        auto published=repeatSession.Commit(1,UsdGenCommitReason::SetTime);
        CHECK(published&&published->device&&!repeatSession.LastDiagnostics().HasErrors()&&
              CheckGeneration(published->device,expected,stream));
    }
    for (int resampleTo : {3, 2}) {
        auto resampledLength=positiveLength;
        auto resampledSource=std::find_if(resampledLength.nodes.begin(),resampledLength.nodes.end(),
            [](auto const& n){return n.type==TfToken("UsdGenCurveSource");});
        CHECK(resampledSource!=resampledLength.nodes.end());
        resampledSource->params.push_back({TfToken("resampleTo"),VtValue(resampleTo),false});
        UsdGenCurveBuffer resampledReference;
        CHECK(GrowLengthScaleReference(resampledLength,.5f,&resampledReference));
        UsdGenDiagnostics resampledDiagnostics;
        auto resampledPlan=CompileCudaGraph(resampledLength,&resampledDiagnostics);
        CHECK(resampledPlan&&!resampledDiagnostics.HasErrors());
        auto resampledDirect=ExecuteCudaGraph(*resampledPlan,*positiveLengthWorkspace,
            1,174+resampleTo,&resampledDiagnostics);
        CHECK(resampledDirect&&!resampledDiagnostics.HasErrors()&&
              CheckGeneration(resampledDirect,resampledReference,stream));
        UsdGenSession resampledLengthSession;
        resampledLengthSession.SetDevicePublicationEnabled(true);
        resampledLengthSession.SetGraphDesc(resampledLength);
        auto resampledPublished=resampledLengthSession.Commit(1,UsdGenCommitReason::SetTime);
        CHECK(resampledPublished&&resampledPublished->device&&
              !resampledLengthSession.LastDiagnostics().HasErrors()&&
              CheckGeneration(resampledPublished->device,resampledReference,stream));
    }

    {
        auto makeMixed=[&](bool noiseFirst,TfToken mode,float value) {
            auto desc=GrowLengthDesc(mode,value);
            auto noise=*std::find_if(growNoise.nodes.begin(),growNoise.nodes.end(),
                [](auto const& node){return node.type==TfToken("UsdGenNoise");});
            auto length=desc.nodes[2];auto width=desc.nodes[3];
            if(noiseFirst) {
                noise.inputs={desc.nodes[1].path};length.inputs={noise.path};
                width.inputs={length.path};
                desc.nodes={desc.nodes[0],desc.nodes[1],noise,length,width};
            } else {
                noise.inputs={length.path};width.inputs={noise.path};
                desc.nodes={desc.nodes[0],desc.nodes[1],length,noise,width};
            }
            return desc;
        };
        std::vector<UsdGenGraphDesc> cases{
            makeMixed(false,TfToken("scale"),.5f),
            makeMixed(true,TfToken("scale"),.5f),
            makeMixed(false,TfToken("cull"),(strand0+strand1)*.5f),
            makeMixed(false,TfToken("cull"),1000.f)};
        auto emptyRepeated=makeMixed(true,TfToken("cull"),1000.f);
        auto extraNoise=emptyRepeated.nodes[2];extraNoise.path=SdfPath("/Ops/SecondNoise");
        extraNoise.inputs={emptyRepeated.nodes[3].path};
        emptyRepeated.nodes.back().inputs={extraNoise.path};
        emptyRepeated.nodes.insert(emptyRepeated.nodes.end()-1,extraNoise);
        cases.push_back(emptyRepeated);
        auto repeated=cases.front();
        auto extraLength=repeated.nodes[2];extraLength.path=SdfPath("/Ops/SecondMixedLength");
        extraLength.inputs={repeated.nodes[3].path};repeated.nodes.back().inputs={extraLength.path};
        repeated.nodes.insert(repeated.nodes.end()-1,extraLength);cases.push_back(repeated);
        auto emptySource=cases.front();emptySource.curveSets=Desc(false,false,true).curveSets;
        cases.push_back(emptySource);
        auto repeatedNoise=cases[1];
        auto secondNoise=repeatedNoise.nodes[2];secondNoise.path=SdfPath("/Ops/NonemptySecondNoise");
        secondNoise.inputs={repeatedNoise.nodes[2].path};repeatedNoise.nodes[3].inputs={secondNoise.path};
        repeatedNoise.nodes.insert(repeatedNoise.nodes.begin()+3,secondNoise);
        cases.push_back(repeatedNoise);
        UsdGenSession mixedCaseSession;mixedCaseSession.SetDevicePublicationEnabled(true);
        std::shared_ptr<const UsdGenDeviceGeneration> held;
        UsdGenCurveBuffer heldReference;
        std::vector<float3> mixedCaseOldPoints,mixedCaseOldRest;
        std::vector<float> mixedCaseOldWidths;
        for(size_t variant=0;variant<cases.size();++variant) {
            auto const& mixedCaseDesc=cases[variant];UsdGenCurveBuffer expected;
            CHECK(MixedGrowReference(mixedCaseDesc,&expected));
            if(variant==2) CHECK(expected.totalCurves==1);
            if(variant==3||variant==4||variant==6) CHECK(expected.totalCurves==0);
            UsdGenDiagnostics mixedCaseDiagnostics;
            auto mixedCasePlan=CompileCudaGraph(mixedCaseDesc,&mixedCaseDiagnostics);
            for(auto const& error:mixedCaseDiagnostics.errors) std::fprintf(stderr,"mixed compile: %s\n",error.c_str());
            CHECK(mixedCasePlan&&!mixedCaseDiagnostics.HasErrors());
            auto mixedCaseWorkspace=CreateCudaExecutionWorkspace(-1,&mixedCaseDiagnostics);CHECK(mixedCaseWorkspace);
            auto mixedCaseDirect=ExecuteCudaGraph(*mixedCasePlan,*mixedCaseWorkspace,1,300+variant,&mixedCaseDiagnostics);
            for(auto const& error:mixedCaseDiagnostics.errors) std::fprintf(stderr,"mixed execute: %s\n",error.c_str());
            CHECK(mixedCaseDirect&&!mixedCaseDiagnostics.HasErrors()&&CheckGeneration(mixedCaseDirect,expected,stream));
            mixedCaseSession.SetGraphDesc(mixedCaseDesc);
            auto published=mixedCaseSession.Commit(static_cast<double>(1+variant),UsdGenCommitReason::SetTime);
            CHECK(published&&published->device&&!mixedCaseSession.LastDiagnostics().HasErrors()&&
                  CheckGeneration(published->device,expected,stream));
            if(variant==2||variant==4) {
                auto mixedCaseJob=CreateCudaExecutionJob(mixedCasePlan,*mixedCaseWorkspace,1,350+variant,&mixedCaseDiagnostics);CHECK(mixedCaseJob);
                std::promise<bool> sourceDone;auto ready=sourceDone.get_future();
                CHECK(ExecuteCudaJobSourceAsync(mixedCaseJob,[&](bool ok){sourceDone.set_value(ok);})&&
                      ready.wait_for(std::chrono::seconds(10))==std::future_status::ready&&ready.get());
                for(size_t i=0;i<CudaExecutionJobOperatorCount(*mixedCaseJob);++i) {
                    std::promise<bool> done;auto completed=done.get_future();
                    CHECK(ExecuteCudaJobOperatorAsync(mixedCaseJob,i,[&](bool ok){done.set_value(ok);})&&
                          completed.wait_for(std::chrono::seconds(10))==std::future_status::ready&&completed.get());
                }
                std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> finalDone;
                auto finalReady=finalDone.get_future();
                CHECK(FinalizeCudaExecutionJobAsync(mixedCaseJob,[&](auto result){finalDone.set_value(std::move(result));})&&
                      finalReady.wait_for(std::chrono::seconds(10))==std::future_status::ready);
                auto result=finalReady.get();
                CHECK(result&&!mixedCaseDiagnostics.HasErrors()&&CheckGeneration(result,expected,stream));
            }
            if(!held) {
                held=published->device;heldReference=expected;
                auto lease=gpu::AcquireGeometry(held,stream);CHECK(lease);
                CHECK(Read(lease.Geometry().points,&mixedCaseOldPoints,stream)&&Read(lease.Geometry().restPoints,&mixedCaseOldRest,stream)&&
                      Read(lease.Geometry().widths,&mixedCaseOldWidths,stream));
            }
            CHECK(CheckGeneration(held,heldReference,stream));
            auto lease=gpu::AcquireGeometry(held,stream);CHECK(lease);
            std::vector<float3> heldPoints,heldRest;std::vector<float> heldWidths;
            CHECK(Read(lease.Geometry().points,&heldPoints,stream)&&Read(lease.Geometry().restPoints,&heldRest,stream)&&
                  Read(lease.Geometry().widths,&heldWidths,stream)&&heldPoints.size()==mixedCaseOldPoints.size()&&heldRest.size()==mixedCaseOldRest.size()&&
                  heldWidths.size()==mixedCaseOldWidths.size()&&std::memcmp(heldPoints.data(),mixedCaseOldPoints.data(),heldPoints.size()*sizeof(float3))==0&&
                  std::memcmp(heldRest.data(),mixedCaseOldRest.data(),heldRest.size()*sizeof(float3))==0&&
                  std::memcmp(heldWidths.data(),mixedCaseOldWidths.data(),heldWidths.size()*sizeof(float))==0);
        }
    }

    // Branch values must follow declared edges, including inherited point and
    // width owners. Retain an older publication throughout all terminal edits.
    {
        std::vector<UsdGenGraphDesc> cases;
        for(auto const* terminal:{"/Dag/Blend","/Dag/Noise","/Dag/Left",
                                  "/Dag/Tail","/Dag/Other","/Ops/Source"}) {
            auto terminalDesc=NoiseDagDesc();terminalDesc.terminal=SdfPath(terminal);cases.push_back(terminalDesc);
        }
        auto reversed=cases.front();
        std::swap(reversed.nodes.front().inputs[0],reversed.nodes.front().inputs[1]);
        cases.push_back(reversed);
        for(int resampleCount:{3,2}) {
            auto resampledDesc=cases[3];
            for(auto& node:resampledDesc.nodes) if(node.type==TfToken("UsdGenCurveSource"))
                node.params.push_back({TfToken("resampleTo"),VtValue(resampleCount),false});
            cases.push_back(resampledDesc);
        }
        auto empty=cases.front();empty.curveSets=Desc(false,false,true).curveSets;cases.push_back(empty);
        auto inherited=cases.front();inherited.nodes.front().inputs[0]=SdfPath("/Dag/Noise");
        cases.push_back(inherited);
        std::swap(inherited.nodes.front().inputs[0],inherited.nodes.front().inputs[1]);
        cases.push_back(inherited);
        auto repeated=cases[3];
        for(auto& node:repeated.nodes) if(node.path==SdfPath("/Dag/Tail"))
            node.inputs={SdfPath("/Dag/Noise")};
        cases.push_back(repeated);
        size_t const topologyCasesBegin=cases.size();
        for(bool grow:{true,false}) {
            for(auto const* terminal:{"/Dag/Blend","/Dag/Noise","/Dag/Left",
                                      "/Dag/Tail","/Dag/Other","/Dag/Topology","/Ops/Source"}) {
                auto topologyDesc=TopologyNoiseDagDesc(grow);topologyDesc.terminal=SdfPath(terminal);cases.push_back(topologyDesc);
            }
            for(int resampleCount:{3,2}) {
                auto resampledTopologyDesc=TopologyNoiseDagDesc(grow);resampledTopologyDesc.terminal=SdfPath("/Dag/Tail");
                for(auto& node:resampledTopologyDesc.nodes) if(node.type==TfToken("UsdGenCurveSource"))
                    node.params.push_back({TfToken("resampleTo"),VtValue(resampleCount),false});
                cases.push_back(resampledTopologyDesc);
            }
            auto emptyTopology=TopologyNoiseDagDesc(grow);emptyTopology.curveSets=Desc(false,false,true).curveSets;
            cases.push_back(emptyTopology);
        }
        for(float threshold:{1.5f,100.f}) {
            auto culledTopologyDesc=TopologyNoiseDagDesc(false,TfToken("cull"),threshold);
            culledTopologyDesc.terminal=SdfPath("/Dag/Tail");cases.push_back(culledTopologyDesc);
        }
        cases.push_back(TopologyNoiseDagDesc(false,TfToken("scale"),.5f,true));
        auto manyNoise=TopologyNoiseDagDesc(false);
        auto noiseTemplate=std::find_if(manyNoise.nodes.begin(),manyNoise.nodes.end(),[](auto const& node) {
            return node.path==SdfPath("/Dag/Tail");
        });
        auto nextNoise=*noiseTemplate;SdfPath previous=nextNoise.path;
        for(int i=0;i<4;++i) {
            nextNoise.path=SdfPath("/Dag/ExtraNoise"+std::to_string(i));nextNoise.inputs={previous};
            manyNoise.nodes.push_back(nextNoise);previous=nextNoise.path;
        }
        manyNoise.terminal=previous;cases.push_back(manyNoise);
        for(bool grow:{true,false}) {
            for(auto const* terminal:{"/Dag/Other","/Dag/SideLeft","/Dag/SideBlend","/Dag/SideTail",
                                      "/Dag/SideRawWidth","/Dag/Blend","/Dag/Topology","/Ops/Source"}) {
                auto branchDesc=SourceBranchNoiseDagDesc(grow);branchDesc.terminal=SdfPath(terminal);cases.push_back(branchDesc);
            }
            for(int resampleCount:{3,2}) {
                auto resampledBranchDesc=SourceBranchNoiseDagDesc(grow);
                for(auto& node:resampledBranchDesc.nodes) if(node.type==TfToken("UsdGenCurveSource"))
                    node.params.push_back({TfToken("resampleTo"),VtValue(resampleCount),false});
                cases.push_back(resampledBranchDesc);
            }
            auto emptyBranch=SourceBranchNoiseDagDesc(grow);emptyBranch.curveSets=Desc(false,false,true).curveSets;
            cases.push_back(emptyBranch);
        }
        for(float threshold:{1.5f,100.f}) {
            auto culledBranchDesc=SourceBranchNoiseDagDesc(false,TfToken("cull"),threshold);
            cases.push_back(culledBranchDesc);culledBranchDesc.terminal=SdfPath("/Dag/Blend");cases.push_back(culledBranchDesc);
        }
        auto removeNoise=[](UsdGenGraphDesc desc) {
            std::map<SdfPath,SdfPath> inputs;
            for(auto const& node:desc.nodes) if(node.type==TfToken("UsdGenNoise")) inputs[node.path]=node.inputs.front();
            for(auto& node:desc.nodes) for(auto& input:node.inputs)
                for(size_t depth=0;depth<desc.nodes.size()&&inputs.count(input);++depth) input=inputs.at(input);
            for(size_t depth=0;depth<desc.nodes.size()&&inputs.count(desc.terminal);++depth)
                desc.terminal=inputs.at(desc.terminal);
            desc.nodes.erase(std::remove_if(desc.nodes.begin(),desc.nodes.end(),[](auto const& node) {
                return node.type==TfToken("UsdGenNoise");
            }),desc.nodes.end());
            return desc;
        };
        for(bool grow:{true,false}) for(auto const* terminal:{"/Dag/SideBlend","/Dag/Blend","/Dag/SideRawWidth"}) {
            auto noiselessDesc=removeNoise(SourceBranchNoiseDagDesc(grow));noiselessDesc.terminal=SdfPath(terminal);cases.push_back(noiselessDesc);
        }
        cases.push_back(removeNoise(SourceBranchNoiseDagDesc(false,TfToken("cull"),100.f)));
        for(bool nested:{false,true}) {
            for(auto const* terminal:{"/Dag/SideTopology","/Dag/SideBlend","/Dag/SideTail",
                                      "/Dag/Blend","/Dag/Topology","/Dag/SideRawWidth","/Ops/Source"}) {
                auto multiLengthDesc=MultipleLengthNoiseDagDesc(nested);multiLengthDesc.terminal=SdfPath(terminal);cases.push_back(multiLengthDesc);
            }
            for(int resampleCount:{3,2}) {
                auto resampledMultiLengthDesc=MultipleLengthNoiseDagDesc(nested);
                for(auto& node:resampledMultiLengthDesc.nodes) if(node.type==TfToken("UsdGenCurveSource"))
                    node.params.push_back({TfToken("resampleTo"),VtValue(resampleCount),false});
                cases.push_back(resampledMultiLengthDesc);
            }
            auto emptyMultiLength=MultipleLengthNoiseDagDesc(nested);emptyMultiLength.curveSets=Desc(false,false,true).curveSets;
            cases.push_back(emptyMultiLength);
            cases.push_back(removeNoise(MultipleLengthNoiseDagDesc(nested)));
            for(float threshold:{1.5f,100.f}) {
                auto culledMultiLengthDesc=MultipleLengthNoiseDagDesc(nested,TfToken("cull"),threshold);
                cases.push_back(culledMultiLengthDesc);culledMultiLengthDesc.terminal=SdfPath("/Dag/Blend");cases.push_back(culledMultiLengthDesc);
            }
        }
        // A Length consumes a source-side Width snapshot, while another
        // Length consumes its Noise-descended value with independently owned widths.
        auto widthBeforeLength=MultipleLengthNoiseDagDesc(true);
        for(auto& node:widthBeforeLength.nodes) if(node.path==SdfPath("/Dag/Topology"))
            node.inputs={SdfPath("/Dag/SideRawWidth")};
        cases.push_back(widthBeforeLength);
        // Retain more than two compaction/named bundles through the join.
        auto manyLengths=MultipleLengthNoiseDagDesc(true);
        auto nextLength=manyLengths.nodes.front();SdfPath previousLength("/Dag/SideTail");
        for(int i=0;i<4;++i) {
            nextLength.path=SdfPath("/Dag/ExtraLength"+std::to_string(i));nextLength.inputs={previousLength};
            manyLengths.nodes.push_back(nextLength);previousLength=nextLength.path;
        }
        manyLengths.terminal=previousLength;cases.push_back(manyLengths);
        for(bool noiseInput:{false,true}) {
            auto topologyInputDesc=SourceBranchNoiseDagDesc(false);
            for(auto& node:topologyInputDesc.nodes) if(node.path==SdfPath("/Dag/Topology"))
                node.inputs={SdfPath(noiseInput?"/Dag/SideTail":"/Dag/SideRawWidth")};
            topologyInputDesc.terminal=SdfPath("/Dag/Blend");cases.push_back(topologyInputDesc);
            if(!noiseInput) cases.push_back(removeNoise(topologyInputDesc));
            topologyInputDesc.terminal=SdfPath("/Dag/SideBlend");cases.push_back(topologyInputDesc);
            if(!noiseInput) cases.push_back(removeNoise(topologyInputDesc));
        }
        for(bool withLengths:{false,true}) {
            for(auto const* terminal:{"/Dag/Topology","/Dag/SideTopology","/Dag/Blend",
                                      "/Dag/SideBlend","/Dag/Tail","/Dag/SideTail",
                                      "/Dag/SideRawWidth","/Ops/Source"}) {
                auto multiGrowDesc=MultipleGrowNoiseDagDesc(withLengths);multiGrowDesc.terminal=SdfPath(terminal);cases.push_back(multiGrowDesc);
            }
            if(withLengths) for(auto const* terminal:{"/Dag/MainLength","/Dag/SideLength"}) {
                auto lengthTerminalDesc=MultipleGrowNoiseDagDesc(true);lengthTerminalDesc.terminal=SdfPath(terminal);cases.push_back(lengthTerminalDesc);
            }
            for(int resampleCount:{3,2}) {
                auto resampledMultiGrowDesc=MultipleGrowNoiseDagDesc(withLengths);
                for(auto& node:resampledMultiGrowDesc.nodes) if(node.type==TfToken("UsdGenCurveSource"))
                    node.params.push_back({TfToken("resampleTo"),VtValue(resampleCount),false});
                cases.push_back(resampledMultiGrowDesc);
            }
            auto emptyMultiGrow=MultipleGrowNoiseDagDesc(withLengths);emptyMultiGrow.curveSets=Desc(false,false,true).curveSets;
            cases.push_back(emptyMultiGrow);
            cases.push_back(removeNoise(MultipleGrowNoiseDagDesc(withLengths)));
        }
        for(float threshold:{1.5f,100.f}) for(auto const* terminal:{"/Dag/MainLength","/Dag/SideLength","/Dag/SideBlend"}) {
            auto culledMultiGrowDesc=MultipleGrowNoiseDagDesc(true,TfToken("cull"),threshold);
            culledMultiGrowDesc.terminal=SdfPath(terminal);cases.push_back(culledMultiGrowDesc);
        }
        auto manyGrow=MultipleGrowNoiseDagDesc(true);
        auto extraGrow=manyGrow.nodes.front();
        for(int i=0;i<4;++i) {
            extraGrow.path=SdfPath("/Dag/ExtraGrow"+std::to_string(i));
            for(auto& p:extraGrow.params) if(p.name==TfToken("segments")) p.value=VtValue(8+i*2);
            manyGrow.nodes.push_back(extraGrow);
        }
        manyGrow.terminal=extraGrow.path;cases.push_back(manyGrow);
        auto manyGrowStatic=manyGrow;
        manyGrowStatic.nodes.erase(std::remove_if(manyGrowStatic.nodes.begin(),manyGrowStatic.nodes.end(),[](auto const& node) {
            return node.type==TfToken("UsdGenLength");
        }),manyGrowStatic.nodes.end());
        cases.push_back(manyGrowStatic);
        for(auto const* input:{"/Dag/SideRawWidth","/Dag/Other","/Dag/Topology","/Dag/Pre","/Dag/Noise","/Dag/Tail"})
            for(auto const* terminal:{"/Dag/Regrow","/Dag/RegrowNoise","/Dag/SideBlend"}) {
                auto regrowDesc=RegrowNoiseDagDesc(false,input);regrowDesc.terminal=SdfPath(terminal);cases.push_back(regrowDesc);
            }
        for(float threshold:{1.5f,100.f}) for(auto const* input:{"/Dag/Topology","/Dag/Tail"})
            for(auto const* terminal:{"/Dag/RegrowNoise","/Dag/SideBlend"}) {
                auto culledRegrowDesc=RegrowNoiseDagDesc(false,input,TfToken("cull"),threshold);
                culledRegrowDesc.terminal=SdfPath(terminal);cases.push_back(culledRegrowDesc);
            }
        for(bool grownInput:{false,true}) {
            for(int resampleCount:{3,2}) {
                auto resampledRegrowDesc=RegrowNoiseDagDesc(grownInput,grownInput?"/Dag/SideLength":"/Dag/Tail");
                for(auto& node:resampledRegrowDesc.nodes) if(node.type==TfToken("UsdGenCurveSource"))
                    node.params.push_back({TfToken("resampleTo"),VtValue(resampleCount),false});
                cases.push_back(resampledRegrowDesc);
            }
            auto emptyRegrow=RegrowNoiseDagDesc(grownInput,grownInput?"/Dag/SideLength":"/Dag/Tail");
            emptyRegrow.curveSets=Desc(false,false,true).curveSets;cases.push_back(emptyRegrow);
        }
        cases.push_back(removeNoise(RegrowNoiseDagDesc(false,"/Dag/Pre")));
        for(auto const* input:{"/Dag/Topology","/Dag/SideTopology","/Dag/MainLength","/Dag/SideLength","/Dag/SideTail"})
            for(auto const* terminal:{"/Dag/Regrow","/Dag/RegrowNoise"}) {
                auto grownRegrowDesc=RegrowNoiseDagDesc(true,input);grownRegrowDesc.terminal=SdfPath(terminal);cases.push_back(grownRegrowDesc);
            }
        for(float threshold:{1.5f,100.f}) cases.push_back(RegrowNoiseDagDesc(true,"/Dag/SideLength",TfToken("cull"),threshold));
        std::fprintf(stderr,"Grow DAG matrix variants: %zu\n",cases.size());
        UsdGenSession dagSession;dagSession.SetDevicePublicationEnabled(true);
        UsdGenDiagnostics dagDiagnostics;
        auto dagWorkspace=CreateCudaExecutionWorkspace(-1,&dagDiagnostics);CHECK(dagWorkspace);
        std::shared_ptr<const UsdGenDeviceGeneration> held;
        UsdGenCurveBuffer heldReference;
        std::vector<float3> dagOldPoints,dagOldRest;std::vector<float> dagOldWidths;
        for(size_t variant=0;variant<cases.size();++variant) {
            auto const& dagDesc=cases[variant];UsdGenCurveBuffer expected;
            bool const oracleOk=NoiseDagReference(dagDesc,&expected);
            if(!oracleOk) {
                std::fprintf(stderr,"oracle failed variant %zu terminal %s\n",variant,dagDesc.terminal.GetText());
                for(auto const& node:dagDesc.nodes) {
                    std::fprintf(stderr,"  %s %s",node.path.GetText(),node.type.GetText());
                    for(auto const& input:node.inputs) std::fprintf(stderr," <- %s",input.GetText());
                    std::fprintf(stderr,"\n");
                }
            }
            CHECK(oracleOk);dagDiagnostics={};
            auto const lengthNodes=std::count_if(dagDesc.nodes.begin(),dagDesc.nodes.end(),[](auto const& node) {
                return node.type==TfToken("UsdGenLength");
            });
            for(auto const& node:dagDesc.nodes) if(lengthNodes==1&&node.type==TfToken("UsdGenLength")) {
                usdGenTest::NodeParams parameters(node);
                if(parameters.GetToken(TfToken("length:mode"),TfToken("scale"))==TfToken("cull")&&
                   TerminalUsesOperator(dagDesc,TfToken("UsdGenLength")))
                    CHECK(expected.totalCurves==(parameters.GetDouble(TfToken("cullThreshold"),0)==1.5?1u:0u));
            }
            bool const sourceRootContract=!TerminalUsesOperator(dagDesc,TfToken("UsdGenGrow"));
            auto const expectedBasis=sourceRootContract&&dagDesc.curveSets.front().basis==TfToken("catmullRom")
                ? UsdGenDeviceCurveBasis::CatmullRom : UsdGenDeviceCurveBasis::BSpline;
            auto dagPlan=CompileCudaGraph(dagDesc,&dagDiagnostics);
            for(auto const& error:dagDiagnostics.errors) std::fprintf(stderr,"Noise DAG compile: %s\n",error.c_str());
            CHECK(dagPlan&&!dagDiagnostics.HasErrors());
            auto dagMetadata=GetCudaExecutionPlanMetadata(*dagPlan);CHECK(dagMetadata);
            CHECK(dagMetadata->Shape()==UsdGenExecutionPlanShape::SourceRootedValueDag);
            auto const& tasks=dagMetadata->Tasks();
            auto pre=std::find_if(tasks.begin(),tasks.end(),[](auto const& task){return task.path==SdfPath("/Dag/Pre");});
            auto noise=std::find_if(tasks.begin(),tasks.end(),[](auto const& task){return task.path==SdfPath("/Dag/Noise");});
            CHECK(pre!=tasks.end());
            if(noise!=tasks.end()) {
            CHECK(noise->exclusiveWorkspace);
            CHECK(std::any_of(noise->resources.begin(),noise->resources.end(),[&](auto const& use) {
                return use.resource==UsdGenExecutionDataKind::Widths&&
                       use.access==UsdGenExecutionResourceAccess::Read&&use.producerTask==pre->id;
            }));
            }
            auto dagDirect=ExecuteCudaGraph(*dagPlan,*dagWorkspace,1,400+variant,&dagDiagnostics,held);
            for(auto const& error:dagDiagnostics.errors) std::fprintf(stderr,"Noise DAG execute variant=%zu terminal=%s: %s\n",variant,dagDesc.terminal.GetText(),error.c_str());
            CHECK(dagDirect&&!dagDiagnostics.HasErrors()&&CheckGeneration(dagDirect,expected,stream,sourceRootContract));
            CHECK(dagDirect->Geometry().curveTopology.basis==expectedBasis);
            dagSession.SetGraphDesc(dagDesc);
            auto published=dagSession.Commit(static_cast<double>(1+variant),UsdGenCommitReason::SetTime);
            if(!published||!published->device||dagSession.LastDiagnostics().HasErrors()) {
                std::fprintf(stderr,"Noise DAG Session variant=%zu terminal=%s\n",variant,dagDesc.terminal.GetText());
                for(auto const& error:dagSession.LastDiagnostics().errors)
                    std::fprintf(stderr,"Noise DAG Session: %s\n",error.c_str());
            }
            CHECK(published&&published->device&&!dagSession.LastDiagnostics().HasErrors()&&
                  CheckGeneration(published->device,expected,stream,sourceRootContract));
            CHECK(published->device->Geometry().curveTopology.basis==expectedBasis);
            if(variant==1||variant==3||variant==8||variant==9||variant>=topologyCasesBegin) {
                auto dagJob=CreateCudaExecutionJob(dagPlan,*dagWorkspace,1,450+variant,&dagDiagnostics);CHECK(dagJob);
                std::promise<bool> sourceDone;auto sourceReady=sourceDone.get_future();
                CHECK(ExecuteCudaJobSourceAsync(dagJob,[&](bool ok){sourceDone.set_value(ok);})&&
                      sourceReady.wait_for(std::chrono::seconds(10))==std::future_status::ready&&sourceReady.get());
                for(size_t i=0;i<CudaExecutionJobOperatorCount(*dagJob);++i) {
                    std::promise<bool> done;auto ready=done.get_future();
                    CHECK(ExecuteCudaJobOperatorAsync(dagJob,i,[&](bool ok){done.set_value(ok);})&&
                          ready.wait_for(std::chrono::seconds(10))==std::future_status::ready&&ready.get());
                }
                bool unexpectedCallback=false;
                setCudaFinalizationRelayCapacityForTesting(0);
                bool admitted=FinalizeCudaExecutionJobAsync(dagJob,[&](auto){unexpectedCallback=true;});
                setCudaFinalizationRelayCapacityForTesting(1024);
                CHECK(!admitted&&!unexpectedCallback);
                std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> done;auto ready=done.get_future();
                CHECK(FinalizeCudaExecutionJobAsync(dagJob,[&](auto result){done.set_value(std::move(result));})&&
                      ready.wait_for(std::chrono::seconds(10))==std::future_status::ready);
                auto result=ready.get();CHECK(result&&CheckGeneration(result,expected,stream,sourceRootContract));
                CHECK(result->Geometry().curveTopology.basis==expectedBasis);
            }
            if(!held) {
                held=published->device;heldReference=expected;
                auto lease=gpu::AcquireGeometry(held,stream);CHECK(lease);
                CHECK(Read(lease.Geometry().points,&dagOldPoints,stream)&&Read(lease.Geometry().restPoints,&dagOldRest,stream)&&
                      Read(lease.Geometry().widths,&dagOldWidths,stream));
            }
            CHECK(CheckGeneration(held,heldReference,stream,true));
            auto lease=gpu::AcquireGeometry(held,stream);CHECK(lease);
            std::vector<float3> heldPoints,heldRest;std::vector<float> heldWidths;
            CHECK(Read(lease.Geometry().points,&heldPoints,stream)&&Read(lease.Geometry().restPoints,&heldRest,stream)&&
                  Read(lease.Geometry().widths,&heldWidths,stream)&&heldPoints.size()==dagOldPoints.size()&&heldRest.size()==dagOldRest.size()&&
                  heldWidths.size()==dagOldWidths.size()&&std::memcmp(heldPoints.data(),dagOldPoints.data(),heldPoints.size()*sizeof(float3))==0&&
                  std::memcmp(heldRest.data(),dagOldRest.data(),heldRest.size()*sizeof(float3))==0&&
                  std::memcmp(heldWidths.data(),dagOldWidths.data(),heldWidths.size()*sizeof(float))==0);
        }
        {
            // Hold a Grow-origin terminal (including TBN/named planes) while
            // selecting source-side and transformed sibling branches.
            auto growOwnerDesc=SourceBranchNoiseDagDesc(true);
            growOwnerDesc.terminal=SdfPath("/Dag/Topology");
            UsdGenCurveBuffer growOwnerReference;
            CHECK(NoiseDagReference(growOwnerDesc,&growOwnerReference));
            UsdGenSession ownerSession; ownerSession.SetDevicePublicationEnabled(true);
            ownerSession.SetGraphDesc(growOwnerDesc);
            auto growOwner=ownerSession.Commit(200,UsdGenCommitReason::SetTime);
            CHECK(growOwner&&growOwner->device&&!ownerSession.LastDiagnostics().HasErrors()&&
                  CheckGeneration(growOwner->device,growOwnerReference,stream));
            auto heldGrowLease=gpu::AcquireGeometry(growOwner->device,stream); CHECK(heldGrowLease);
            std::vector<float3> heldGrowPoints,heldGrowRest,heldGrowT,heldGrowB,heldGrowN;
            std::vector<float> heldGrowWidths;
            CHECK(Read(heldGrowLease.Geometry().points,&heldGrowPoints,stream)&&
                  Read(heldGrowLease.Geometry().restPoints,&heldGrowRest,stream)&&
                  Read(heldGrowLease.Geometry().widths,&heldGrowWidths,stream)&&
                  Read(heldGrowLease.RootT(),&heldGrowT,stream)&&
                  Read(heldGrowLease.RootB(),&heldGrowB,stream)&&
                  Read(heldGrowLease.RootN(),&heldGrowN,stream));
            auto sourceSideDesc=growOwnerDesc;
            sourceSideDesc.terminal=SdfPath("/Dag/SideRawWidth");
            UsdGenCurveBuffer sourceSideReference;
            CHECK(NoiseDagReference(sourceSideDesc,&sourceSideReference));
            ownerSession.SetGraphDesc(sourceSideDesc);
            auto sourceSide=ownerSession.Commit(201,UsdGenCommitReason::SetTime);
            CHECK(sourceSide&&sourceSide->device&&!ownerSession.LastDiagnostics().HasErrors()&&
                  CheckGeneration(sourceSide->device,sourceSideReference,stream,true)&&
                  CheckGeneration(growOwner->device,growOwnerReference,stream));
            auto transformedDesc=growOwnerDesc;
            transformedDesc.terminal=SdfPath("/Dag/Blend");
            UsdGenCurveBuffer transformedReference;
            CHECK(NoiseDagReference(transformedDesc,&transformedReference));
            ownerSession.SetGraphDesc(transformedDesc);
            auto transformed=ownerSession.Commit(202,UsdGenCommitReason::SetTime);
            CHECK(transformed&&transformed->device&&!ownerSession.LastDiagnostics().HasErrors()&&
                  CheckGeneration(transformed->device,transformedReference,stream));
            CHECK(CheckGeneration(growOwner->device,growOwnerReference,stream));
            std::vector<float3> afterPoints,afterRest,afterT,afterB,afterN;
            std::vector<float> afterWidths;
            CHECK(Read(heldGrowLease.Geometry().points,&afterPoints,stream)&&
                  Read(heldGrowLease.Geometry().restPoints,&afterRest,stream)&&
                  Read(heldGrowLease.Geometry().widths,&afterWidths,stream)&&
                  Read(heldGrowLease.RootT(),&afterT,stream)&&
                  Read(heldGrowLease.RootB(),&afterB,stream)&&
                  Read(heldGrowLease.RootN(),&afterN,stream)&&
                  afterPoints.size()==heldGrowPoints.size()&&
                  afterRest.size()==heldGrowRest.size()&&
                  afterWidths.size()==heldGrowWidths.size()&&
                  afterT.size()==heldGrowT.size()&&afterB.size()==heldGrowB.size()&&
                  afterN.size()==heldGrowN.size()&&
                  std::memcmp(afterPoints.data(),heldGrowPoints.data(),afterPoints.size()*sizeof(float3))==0&&
                  std::memcmp(afterRest.data(),heldGrowRest.data(),afterRest.size()*sizeof(float3))==0&&
                  std::memcmp(afterWidths.data(),heldGrowWidths.data(),afterWidths.size()*sizeof(float))==0&&
                  std::memcmp(afterT.data(),heldGrowT.data(),afterT.size()*sizeof(float3))==0&&
                  std::memcmp(afterB.data(),heldGrowB.data(),afterB.size()*sizeof(float3))==0&&
                  std::memcmp(afterN.data(),heldGrowN.data(),afterN.size()*sizeof(float3))==0);
        }
        auto differentOrigins=NoiseDagDesc();
        differentOrigins.nodes.front().inputs[1]=SdfPath("/Dag/Other");
        CHECK(CudaNonWidthRejected(differentOrigins));
        // An unrelated or downstream Grow cannot supply rest to a Noise
        // node that actually reads the source's missing rest channel.
        auto missingRestBeforeGrow=RegrowNoiseDagDesc(false,"/Dag/Other");
        missingRestBeforeGrow.curveSets.front().rest.clear();
        CHECK(CudaRejected(missingRestBeforeGrow));
        auto unrelatedGrowRest=MultipleGrowNoiseDagDesc(false);
        unrelatedGrowRest.curveSets.front().rest.clear();
        for(auto& node:unrelatedGrowRest.nodes) if(node.path==SdfPath("/Dag/Other"))
            node.inputs={SdfPath("/Ops/Source")};
        CHECK(CudaRejected(unrelatedGrowRest));
        for(bool grow:{true,false}) {
            auto unequal=SourceBranchNoiseDagDesc(grow);
            unequal.nodes.front().inputs[1]=SdfPath("/Dag/SideLeft");
            CHECK(CudaNonWidthRejected(unequal));
        }
        // Previously ordered Noise-to-Length chains now use value snapshots;
        // their selected payload and cull semantics must remain unchanged.
        for(int variant=0;variant<4;++variant) {
            auto chainDesc=TopologyNoiseDagDesc(false,variant==1||variant==2?TfToken("cull"):TfToken("scale"),
                variant==1?1.5f:variant==2?100.f:.5f);
            auto findNode=[&](TfToken type) {
                return *std::find_if(chainDesc.nodes.begin(),chainDesc.nodes.end(),[&](auto const& node){return node.type==type;});
            };
            auto source=findNode(TfToken("UsdGenCurveSource"));
            auto noise=findNode(TfToken("UsdGenNoise"));
            auto length=findNode(TfToken("UsdGenLength"));
            auto width=Width("/Dag/LinearWidth",length.path.GetText(),.2f);
            noise.inputs={source.path};length.inputs={noise.path};
            chainDesc.nodes={source,noise,length,width};chainDesc.terminal=width.path;
            if(variant==3) {
                auto pre=Width("/Dag/LinearPre",source.path.GetText(),.4f);
                chainDesc.nodes[1].inputs={pre.path};chainDesc.nodes.insert(chainDesc.nodes.begin()+1,pre);
            }
            UsdGenCurveBuffer expected;CHECK(NoiseDagReference(chainDesc,&expected));
            dagDiagnostics={};auto chainPlan=CompileCudaGraph(chainDesc,&dagDiagnostics);CHECK(chainPlan&&!dagDiagnostics.HasErrors());
            CHECK(GetCudaExecutionPlanMetadata(*chainPlan)->Shape()==UsdGenExecutionPlanShape::SourceRootedUnaryDag);
            auto chainDirect=ExecuteCudaGraph(*chainPlan,*dagWorkspace,1,550+variant,&dagDiagnostics);
            CHECK(chainDirect&&!dagDiagnostics.HasErrors()&&CheckGeneration(chainDirect,expected,stream,true));
            dagSession.SetGraphDesc(chainDesc);auto result=dagSession.Commit(100+variant,UsdGenCommitReason::SetTime);
            CHECK(result&&result->device&&!dagSession.LastDiagnostics().HasErrors()&&
                  CheckGeneration(result->device,expected,stream,true));
        }
    }

    // Source resampling is itself a topology transform before Grow. The
    // 30-degree root-B lift exercises the native angular convention rather
    // than the historical CPU additive-distance shortcut; named point and
    // primitive planes must survive both topology owners.
    auto lifted=Desc(true); UsdGenCurveBuffer liftedReference;
    CHECK(CpuReference(lifted,&liftedReference)&&liftedReference.totalCvs==8);
    UsdGenDiagnostics liftedDiagnostics;
    auto liftedPlan=CompileCudaGraph(lifted,&liftedDiagnostics);
    for(auto const& error:liftedDiagnostics.errors) std::fprintf(stderr,"lifted compile: %s\n",error.c_str());
    CHECK(liftedPlan&&!liftedDiagnostics.HasErrors());
    UsdGenSession liftedSession; liftedSession.SetDevicePublicationEnabled(true); liftedSession.SetGraphDesc(lifted);
    auto liftedSessionResult=liftedSession.Commit(3,UsdGenCommitReason::SetTime);
    for(auto const& error:liftedSession.LastDiagnostics().errors) std::fprintf(stderr,"lifted session: %s\n",error.c_str());
    CHECK(liftedSessionResult&&liftedSessionResult->device&&!liftedSession.LastDiagnostics().HasErrors());
    CHECK(CheckGeneration(liftedSessionResult->device,liftedReference,stream));
    auto liftedWorkspace=CreateCudaExecutionWorkspace(-1,&liftedDiagnostics); CHECK(liftedWorkspace);
    auto liftedDirect=ExecuteCudaGraph(*liftedPlan,*liftedWorkspace,3,103,&liftedDiagnostics);
    CHECK(liftedDirect&&!liftedDiagnostics.HasErrors()&&CheckGeneration(liftedDirect,liftedReference,stream));

    // No authored rest is valid for a deformed/current C3 source.  The
    // loader must materialize its current-points fallback, retain the
    // authored root frame for native Grow, and skip named-plane phase 3
    // entirely.  Exercise both publication paths without relying on an
    // empty-special-case producer.
    auto currentOnly=Desc(false,false);
    currentOnly.curveSets[0].rest.clear();
    UsdGenCurveBuffer currentOnlyReference;
    CHECK(CpuReference(currentOnly,&currentOnlyReference));
    UsdGenDiagnostics currentOnlyDiagnostics;
    auto currentOnlyPlan=CompileCudaGraph(currentOnly,&currentOnlyDiagnostics);
    CHECK(currentOnlyPlan&&!currentOnlyDiagnostics.HasErrors());
    UsdGenSession currentOnlySession;
    currentOnlySession.SetDevicePublicationEnabled(true);
    currentOnlySession.SetGraphDesc(currentOnly);
    auto currentOnlySessionResult=currentOnlySession.Commit(4,UsdGenCommitReason::SetTime);
    CHECK(currentOnlySessionResult&&currentOnlySessionResult->device&&
          !currentOnlySession.LastDiagnostics().HasErrors()&&
          CheckGeneration(currentOnlySessionResult->device,currentOnlyReference,stream)&&
          CheckNoNamed(currentOnlySessionResult->device,stream));
    auto currentOnlyWorkspace=CreateCudaExecutionWorkspace(-1,&currentOnlyDiagnostics);
    CHECK(currentOnlyWorkspace);
    auto currentOnlyDirect=ExecuteCudaGraph(*currentOnlyPlan,*currentOnlyWorkspace,4,104,
                                             &currentOnlyDiagnostics);
    CHECK(currentOnlyDirect&&!currentOnlyDiagnostics.HasErrors()&&
          CheckGeneration(currentOnlyDirect,currentOnlyReference,stream)&&
          CheckNoNamed(currentOnlyDirect,stream));

    // Empty C3 input is still a valid fresh topology: Grow and the complete
    // Width/WidthBlend suffix publish one zero offset rather than failing or
    // using an uninitialized zero-grid path.
    auto empty=Desc(false,false,true); UsdGenCurveBuffer emptyReference;
    CHECK(CpuReference(empty,&emptyReference)&&emptyReference.totalCurves==0&&
          emptyReference.totalCvs==0);
    UsdGenDiagnostics emptyDiagnostics;
    auto emptyPlan=CompileCudaGraph(empty,&emptyDiagnostics);
    CHECK(emptyPlan&&!emptyDiagnostics.HasErrors());
    UsdGenSession emptySession;
    emptySession.SetDevicePublicationEnabled(true); emptySession.SetGraphDesc(empty);
    auto emptySessionResult=emptySession.Commit(5,UsdGenCommitReason::SetTime);
    CHECK(emptySessionResult&&emptySessionResult->device&&
          !emptySession.LastDiagnostics().HasErrors()&&
          CheckGeneration(emptySessionResult->device,emptyReference,stream)&&
          CheckNoNamed(emptySessionResult->device,stream));
    auto emptyWorkspace=CreateCudaExecutionWorkspace(-1,&emptyDiagnostics); CHECK(emptyWorkspace);
    auto emptyDirect=ExecuteCudaGraph(*emptyPlan,*emptyWorkspace,5,105,&emptyDiagnostics);
    CHECK(emptyDirect&&!emptyDiagnostics.HasErrors()&&
          CheckGeneration(emptyDirect,emptyReference,stream)&&CheckNoNamed(emptyDirect,stream));

    // A source/Grow join with unequal payloads must fail the runtime proof.
    // Publication itself may select the complete immutable Source snapshot.
    auto bypass=Desc();
    auto bypassRight=std::find_if(bypass.nodes.begin(),bypass.nodes.end(),[](auto const& node) {
        return node.path==SdfPath("/Ops/Right");
    });
    CHECK(bypassRight!=bypass.nodes.end());
    bypassRight->inputs={SdfPath("/Ops/Source")};
    CHECK(CudaNonWidthRejected(bypass));
    {
        UsdGenDiagnostics sourceDiagnostics;
        auto sourceWorkspace=CreateCudaExecutionWorkspace(-1,&sourceDiagnostics); CHECK(sourceWorkspace);
        std::shared_ptr<const UsdGenDeviceGeneration> retainedSource;
        std::shared_ptr<const UsdGenDeviceGeneration> retainedSessionSource;
        UsdGenCurveBuffer retainedReference;
        UsdGenSession sourceSession;
        sourceSession.SetDevicePublicationEnabled(true);
        for(int variant=0;variant<4;++variant) {
            auto terminalSource=Desc(variant==1||variant==2,true,variant==3);
            terminalSource.terminal=SdfPath("/Ops/Source");
            terminalSource.curveSets.front().basis=TfToken("catmullRom");
            if(variant==2) for(auto& node:terminalSource.nodes)
                if(node.type==TfToken("UsdGenCurveSource")) for(auto& param:node.params)
                    if(param.name==TfToken("resampleTo")) param.value=VtValue(2);
            UsdGenCurveBuffer expected; CHECK(CpuReference(terminalSource,&expected));
            sourceDiagnostics={};
            auto sourcePlan=CompileCudaGraph(terminalSource,&sourceDiagnostics);
            CHECK(sourcePlan&&!sourceDiagnostics.HasErrors());
            auto sourceDirect=ExecuteCudaGraph(*sourcePlan,*sourceWorkspace,1,200+variant,&sourceDiagnostics,retainedSource);
            for(auto const& error:sourceDiagnostics.errors) std::fprintf(stderr,"%s\n",error.c_str());
            CHECK(sourceDirect&&!sourceDiagnostics.HasErrors()&&CheckGeneration(sourceDirect,expected,stream,true));
            CHECK(sourceDirect->Geometry().curveTopology.basis==UsdGenDeviceCurveBasis::CatmullRom);
            sourceSession.SetGraphDesc(terminalSource);
            auto published=sourceSession.Commit(1,UsdGenCommitReason::SetTime);
            for(auto const& error:sourceSession.LastDiagnostics().errors) std::fprintf(stderr,"%s\n",error.c_str());
            CHECK(published&&published->device&&!sourceSession.LastDiagnostics().HasErrors()&&
                CheckGeneration(published->device,expected,stream,true)&&
                published->device->Geometry().curveTopology.basis==UsdGenDeviceCurveBasis::CatmullRom);
            if(variant==2) {
                auto sourceJob=CreateCudaExecutionJob(sourcePlan,*sourceWorkspace,1,210,&sourceDiagnostics);
                CHECK(sourceJob);
                std::promise<bool> sourceDone; auto sourceReady=sourceDone.get_future();
                CHECK(ExecuteCudaJobSourceAsync(sourceJob,[&](bool ok){sourceDone.set_value(ok);}));
                CHECK(sourceReady.wait_for(std::chrono::seconds(10))==std::future_status::ready&&sourceReady.get());
                for(size_t i=0;i<CudaExecutionJobOperatorCount(*sourceJob);++i) {
                    std::promise<bool> done;auto ready=done.get_future();
                    CHECK(ExecuteCudaJobOperatorAsync(sourceJob,i,[&](bool ok){done.set_value(ok);}));
                    CHECK(ready.wait_for(std::chrono::seconds(10))==std::future_status::ready&&ready.get());
                }
                bool unexpectedCallback=false;
                setCudaFinalizationRelayCapacityForTesting(0);
                bool const admitted=FinalizeCudaExecutionJobAsync(sourceJob,[&](auto){unexpectedCallback=true;});
                setCudaFinalizationRelayCapacityForTesting(1024);
                CHECK(!admitted&&!unexpectedCallback);
                std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> done;
                auto ready=done.get_future();
                CHECK(FinalizeCudaExecutionJobAsync(sourceJob,[&](auto result){done.set_value(std::move(result));}));
                CHECK(ready.wait_for(std::chrono::seconds(10))==std::future_status::ready);
                auto retried=ready.get();
                CHECK(retried&&!sourceDiagnostics.HasErrors()&&CheckGeneration(retried,expected,stream,true));
            }
            if(!retainedSource) {retainedSource=sourceDirect;retainedSessionSource=published->device;retainedReference=expected;}
            CHECK(CheckGeneration(retainedSource,retainedReference,stream,true)&&
                CheckGeneration(retainedSessionSource,retainedReference,stream,true));
        }
    }

    // Geometry leases enqueue their retirement event on the caller's stream.
    // They must complete before that external stream is destroyed.
    old={}; current={};
    retainedGrowNoiseLease={};
    editedGrowNoiseLease={};
    CHECK(cudaStreamDestroy(stream)==cudaSuccess);
    return 0;
}
