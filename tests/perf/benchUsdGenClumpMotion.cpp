// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Informational timings; correctness failures are always fatal. Run on a quiet
// host, never alongside the documentation renderer. No absolute timing gates.
#include "usdGen/clumpMotion.h"
#include "usdGen/ops/wind.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace usdGen;
namespace {
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
double Median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end()); return samples[samples.size()/2];
}
UsdGenPlane Plane(char const* prefix, int level, char const* interp,
                  char const* type, int arity) {
    UsdGenPlane p; p.name=TfToken(std::string(prefix)+std::to_string(level));
    p.interpolation=TfToken(interp); p.type=TfToken(type); p.arity=uint8_t(arity);
    return p;
}
UsdGenCurveBuffer Fixture(uint32_t curves, int levels, float cohesion) {
    constexpr uint32_t cvs=8, chunkSize=256;
    UsdGenCurveBuffer b; b.totalCurves=curves; b.totalCvs=curves*cvs;
    b.px.resize(b.totalCvs); b.py.resize(b.totalCvs); b.pz.resize(b.totalCvs);
    b.hairT.resize(b.totalCvs); b.rest.resize(b.totalCvs); b.curveId.resize(curves);
    for(uint32_t c=0;c<curves;++c) {
        b.curveId[c]=uint64_t(c)+0x100000000ULL;
        for(uint32_t i=0;i<cvs;++i) {
            size_t k=size_t(c)*cvs+i;
            b.px[k]=float(c%128)*.01f; b.py[k]=float(i)*.1f;
            b.pz[k]=float(c/128)*.01f; b.hairT[k]=float(i)/(cvs-1);
            b.rest[k]=GfVec3f(b.px[k],b.py[k],b.pz[k]);
        }
    }
    for(uint32_t first=0;first<curves;first+=chunkSize) {
        UsdGenChunkDesc ch; ch.firstCurve=first;
        ch.curveCount=ch.liveCount=std::min(chunkSize,curves-first);
        ch.firstCv=first*cvs; ch.cvCount=cvs; b.chunks.push_back(ch);
    }
    for(int l=0;l<levels;++l) {
        auto ids=Plane("clumpId_",l,"uniform","int",1);
        auto center=Plane("clumpCenter_",l,"uniform","float",3);
        auto exact=Plane("clumpCenterId_",l,"uniform","int",2);
        auto weight=Plane("clumpWeight_",l,"vertex","float",1);
        ids.i.resize(curves); center.f.resize(size_t(curves)*3);
        exact.i.resize(size_t(curves)*2); weight.f.resize(b.totalCvs);
        // Every group spans many chunks, exercising global capture deduplication.
        for(uint32_t c=0;c<curves;++c) {
            int group=int(c%uint32_t(8*(l+1))); ids.i[c]=group;
            center.f[c*3]=float(group)*.03f; center.f[c*3+1]=0;
            center.f[c*3+2]=float(l)*.05f;
            uint64_t id=(uint64_t(0x80000001U+uint32_t(l))<<32)|uint32_t(group);
            uint32_t low=uint32_t(id), high=uint32_t(id>>32);
            std::memcpy(&exact.i[c*2],&low,4); std::memcpy(&exact.i[c*2+1],&high,4);
            for(uint32_t i=0;i<cvs;++i) weight.f[c*cvs+i]=i?cohesion:0.f;
        }
        b.extraCurve.push_back(std::move(ids)); b.extraCurve.push_back(std::move(center));
        b.extraCurve.push_back(std::move(exact)); b.extraCv.push_back(std::move(weight));
    }
    return b;
}
size_t PlaneBytes(UsdGenCurveBuffer const& b) {
    size_t bytes=0;
    for(auto const& p:b.extraCurve) bytes+=p.f.size()*sizeof(float)+p.i.size()*sizeof(int);
    for(auto const& p:b.extraCv) bytes+=p.f.size()*sizeof(float)+p.i.size()*sizeof(int);
    return bytes;
}
bool Equal(UsdGenCurveBuffer const& a,UsdGenCurveBuffer const& b) {
    return a.px==b.px && a.py==b.py && a.pz==b.pz;
}
void Evaluate(UsdGenWindOp const& op, UsdGenCapture const& capture,
              UsdGenEvalContext const& ctx, UsdGenCurveBuffer const& input,
              UsdGenCurveBuffer* output, bool reverse=false) {
    for(size_t n=0;n<input.chunks.size();++n) {
        auto const& ch=input.chunks[reverse?input.chunks.size()-1-n:n];
        UsdGenChunkView v{}; v.desc=&ch; v.curveCount=ch.curveCount; v.cvCount=ch.cvCount;
        v.inCvCount=ch.cvCount; v.inFirstCv=ch.firstCv;
        v.inPx=input.px.cdata()+ch.firstCv; v.inPy=input.py.cdata()+ch.firstCv;
        v.inPz=input.pz.cdata()+ch.firstCv; v.px=output->px.data()+ch.firstCv;
        v.py=output->py.data()+ch.firstCv; v.pz=output->pz.data()+ch.firstCv;
        v.hairT=output->hairT.data()+ch.firstCv; v.curveId=input.curveId.cdata()+ch.firstCurve;
        op.Evaluate(ctx,capture,&v);
    }
}
}
int main(int argc,char** argv) {
    uint32_t curves=argc>1?uint32_t(std::strtoul(argv[1],nullptr,10)):10000;
    if(curves<512 || curves>1000000) { std::fprintf(stderr,"curves must be 512..1000000\n"); return 2; }
    UsdGenGraphDesc desc; desc.timeCodesPerSecond=24;
    UsdGenNodeDesc node; node.params={
        {TfToken("direction"),VtValue(GfVec3f(0,0,1)),false},
        {TfToken("constStrength"),VtValue(.2),false},
        {TfToken("gustStrength"),VtValue(.3),false},
        {TfToken("billowLowStrength"),VtValue(.1),false},
        {TfToken("billowHighStrength"),VtValue(.05),false}};
    UsdGenOpRegistry const& registry=UsdGenOpRegistry::Get();
    UsdGenOpRecord const& windRecord=registry.Record(registry.Find(TfToken("UsdGenWind")));
    UsdGenStackParams stack(desc,node,windRecord); UsdGenParamView const& params=stack.View();
    UsdGenCaptureContext cc; cc.desc=&desc; cc.params=&params; cc.seed=42;
    UsdGenEvalContext ec; ec.desc=&desc; ec.params=&params; ec.seed=42; ec.time=17;
    UsdGenWindOp op(windRecord); UsdGenCurveBuffer legacy;
    int failures=0;
    struct Case { char const* name; int levels; float weight; };
    for(auto const& test:std::vector<Case>{{"none",0,0},{"zero",2,0},{"one",1,1},{"two",2,.7f}}) {
        auto input=Fixture(curves,test.levels,test.weight);
        auto capture=op.CreateCapture(); UsdGenDiagnostics diag;
        std::vector<double> cold,warm,recapture,clone;
        for(int run=0;run<7;++run) {
            capture=op.CreateCapture();
            auto start=Clock::now();
            if(!op.Capture(cc,input,capture.get(),&diag)) { std::fprintf(stderr,"capture failed %s\n",test.name); return 1; }
            cold.push_back(Milliseconds(start));
        }
        // Moving upstream points invalidate Wind's value-version capture while
        // the clump quartet retains exactly the same COW payload identities.
        auto animated=input;
        for(int run=0;run<7;++run) {
            animated.valueVersion=uint64_t(run+1);
            animated.py[animated.totalCvs-1]=input.py[input.totalCvs-1]+float(run+1)*.001f;
            auto start=Clock::now();
            if(!op.Capture(cc,animated,capture.get(),&diag)) return 1;
            recapture.push_back(Milliseconds(start));
        }
        if(!op.Capture(cc,input,capture.get(),&diag)) return 1;
        for(int run=0;run<11;++run) {
            auto start=Clock::now(); auto copy=capture->Clone();
            clone.push_back(Milliseconds(start));
            if(!copy) return 1;
        }
        auto output=input; output.px=VtFloatArray(input.px.begin(),input.px.end());
        output.py=VtFloatArray(input.py.begin(),input.py.end());
        output.pz=VtFloatArray(input.pz.begin(),input.pz.end());
        Evaluate(op,*capture,ec,input,&output); // warm thread-local caches and COW storage
        for(int run=0;run<11;++run) {
            auto start=Clock::now(); Evaluate(op,*capture,ec,input,&output);
            warm.push_back(Milliseconds(start));
        }
        auto forward=output; Evaluate(op,*capture,ec,input,&output,true);
        if(!Equal(forward,output)) { ++failures; std::fprintf(stderr,"chunk order mismatch %s\n",test.name); }
        if(test.levels==0) legacy=output;
        if(test.weight==0 && !Equal(legacy,output)) { ++failures; std::fprintf(stderr,"legacy bit mismatch %s\n",test.name); }
        UsdGenClumpMotion motion; std::string error;
        if(!UsdGenBuildClumpMotion(input,&motion,&error)) return 1;
        size_t denseBytes=0,groups=0,sharedWeightBytes=0;
        for(auto const& l:motion.levels) {
            denseBytes+=l.groupForCurve.capacity()*sizeof(uint32_t)+l.groups.capacity()*sizeof(UsdGenClumpMotionGroup);
            groups+=l.groups.size(); sharedWeightBytes+=l.weight.size()*sizeof(float);
            if(l.groups.size()!=size_t(8*(l.level+1)) || l.groupForCurve[0]!=l.groupForCurve[256]) ++failures;
            if(!l.groups.empty() && uint32_t(l.groups.front().centerId>>32)!=0x80000001U+uint32_t(l.level)) ++failures;
        }
        std::printf("%s curves=%u cvs=%u cold_capture_ms=%.3f value_recapture_ms=%.3f clone_ms=%.3f warm_evaluate_ms=%.3f metadata_bytes=%zu capture_dense_bytes=%zu shared_weight_bytes=%zu groups=%zu\n",
            test.name,curves,input.totalCvs,Median(cold),Median(recapture),Median(clone),Median(warm),PlaneBytes(input),denseBytes,sharedWeightBytes,groups);
    }
    std::printf("correctness_failures=%d (memory excludes common Wind rest arrays and allocator overhead)\n",failures);
    return failures?1:0;
}
