// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// benchUsdGenRestructure: before/after numbers for the engine restructure.
// Informational (never fails on timing). G3: 100k curves x 8 segments,
// scatter -> grow -> noise -> length -> width.
//
//   compile.cold        fresh Compile of G3
//   cook.cold           first Run after Compile (capture + full eval)
//   cook.warm           full-dirty Run, no topology change
//   cook.paramEdit      Recompile with one value edit, then Run
//   hydra.publish       stage edit -> scene index Synchronize (cooked, published)
//   hydra.pull          GetPrim + every tile's points/widths/topology GetValue
//   allocs.*            heap allocations during the measured region, from the
//                       usdGen counter (allocCounter.h): CRT heap calls of every
//                       module on Windows, operator new calls on ELF
//
// Usage: benchUsdGenRestructure [--threads N] [--runs N] [--no-hydra]

#include "usdGen/allocCounter.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGen/types.h"

#include "usdGenImaging/groomSceneIndexPlugin.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

// ---- allocation counter ----------------------------------------------------
// The count lives in usdGen (allocCounter.h). On ELF this replacement
// interposes on every module and feeds it; on Windows usdGen wraps every
// module's CRT heap imports instead.
#ifndef _WIN32
void *operator new(std::size_t n)
{
    usdGen::UsdGenEngineAllocNote(n);
    if (void *p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t n) { return operator new(n); }
void *operator new(std::size_t n, std::align_val_t a)
{
    usdGen::UsdGenEngineAllocNote(n);
    size_t const al = static_cast<size_t>(a);
    if (void *p = std::aligned_alloc(al, (n + al - 1) / al * al)) return p;
    throw std::bad_alloc();
}
inline void AlignedFree(void *p) noexcept { std::free(p); }
void *operator new[](std::size_t n, std::align_val_t a) { return operator new(n, a); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
void operator delete(void *p, std::align_val_t) noexcept { AlignedFree(p); }
void operator delete[](void *p, std::align_val_t) noexcept { AlignedFree(p); }
void operator delete(void *p, std::size_t, std::align_val_t) noexcept { AlignedFree(p); }
void operator delete[](void *p, std::size_t, std::align_val_t) noexcept { AlignedFree(p); }
#endif

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

using Clock = std::chrono::steady_clock;

// Counting stays on for the whole run; regions read deltas. Restarting picks
// up modules loaded since (Windows) and resets the count.
uint64_t Allocs() { return UsdGenEngineAllocCount(); }
double Ms(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Stat {
    std::vector<double> v;
    double Median() const
    {
        if (v.empty()) return 0.0;
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        return s[s.size() / 2];
    }
    double Min() const { return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); }
};

void Report(char const *name, Stat const &s, char const *unit = "ms")
{
    std::printf("RESULT %-22s median=%10.3f min=%10.3f %s (n=%zu)\n", name,
                s.Median(), s.Min(), unit, s.v.size());
}

constexpr int kNX = 400, kNY = 250;   // 1000 area units at density 100: ~100k roots

UsdGenGraphDesc MakeG3(double noiseMagnitude)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/width");
    d.time = 0.0;
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.id = 0;
    s.restPoints = VtVec3fArray((kNX + 1) * (kNY + 1));
    s.uv = VtVec2fArray((kNX + 1) * (kNY + 1));
    for (int j = 0; j <= kNY; ++j)
        for (int i = 0; i <= kNX; ++i) {
            int const k = j * (kNX + 1) + i;
            s.restPoints[k] = GfVec3f(i * 0.1f, j * 0.1f, 0.0f);
            s.uv[k] = GfVec2f(float(i) / kNX, float(j) / kNY);
        }
    s.faceVertexCounts = VtIntArray(kNX * kNY, 4);
    s.faceVertexIndices = VtIntArray(kNX * kNY * 4);
    for (int j = 0; j < kNY; ++j)
        for (int i = 0; i < kNX; ++i) {
            int const a = j * (kNX + 1) + i, o = (j * kNX + i) * 4;
            s.faceVertexIndices[o + 0] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + kNX + 2;
            s.faceVertexIndices[o + 3] = a + kNX + 1;
        }
    d.surfaces.push_back(std::move(s));
    auto addNode = [&](std::string const &name, char const *type,
                       std::string const &input, int seed) {
        UsdGenNodeDesc n;
        n.path = SdfPath("/groom/" + name);
        n.type = TfToken(type);
        n.enabled = true;
        n.seed = seed;
        if (!input.empty()) n.inputs.push_back(SdfPath("/groom/" + input));
        if (n.type == TfToken("UsdGenScatter")) n.surfaces.push_back(SdfPath("/groom/surface"));
        d.nodes.push_back(std::move(n));
    };
    addNode("scatter", "UsdGenScatter", "", 42);
    addNode("grow", "UsdGenGrow", "scatter", 43);
    addNode("noise", "UsdGenNoise", "grow", 44);
    addNode("length", "UsdGenLength", "noise", 45);
    addNode("width", "UsdGenWidth", "length", 46);
    auto setp = [&](std::string const &name, char const *p, VtValue v) {
        for (auto &n : d.nodes)
            if (n.path == SdfPath("/groom/" + name))
                n.params.push_back(UsdGenParamValue{TfToken(p), v, false});
    };
    setp("grow", "segments", VtValue(8));
    setp("grow", "length", VtValue(1.0));
    setp("noise", "noise:magnitude", VtValue(noiseMagnitude));
    setp("noise", "noise:frequency", VtValue(3.0));
    setp("noise", "noise:octaves", VtValue(2));
    setp("noise", "noise:correlation", VtValue(0.5));
    setp("length", "length:mode", VtValue(TfToken("scale")));
    setp("length", "length:value", VtValue(1.2));
    setp("width", "width", VtValue(0.02));
    return d;
}

void ForceFullDirty(UsdGenGraph &g)
{
    for (int i = 0; i < g.NodeCount(); ++i)
        for (auto &word : g.Node(i).chunkDirty) word = UsdGenDirtyParameter;
}

void EngineBench(int threads, int runs)
{
    Stat compile, coldCook, warm, edit, warmAllocs, editAllocs;
    UsdGenGraphDesc const a = MakeG3(0.05), b = MakeG3(0.06);
    UsdGenScheduler sched(threads);
    UsdGenEngineAllocCounting(true);
    uint64_t gen = 0;
    size_t curves = 0, cvs = 0;
    for (int r = 0; r < runs; ++r) {
        UsdGenCompiler c;
        UsdGenGraph g;
        auto t0 = Clock::now();
        UsdGenCompileResult const cr = c.Compile(a, &g);
        auto t1 = Clock::now();
        if (!cr.ok) {
            std::printf("compile failed:");
            for (auto const &e : cr.errors) std::printf(" %s", e.c_str());
            std::printf("\n");
            std::exit(1);
        }
        compile.v.push_back(Ms(t0, t1));
        UsdGenEvalContext ctx;
        ctx.desc = &g.Desc();
        t0 = Clock::now();
        UsdGenRunResult const first = sched.Run(g, ctx, ++gen);
        t1 = Clock::now();
        coldCook.v.push_back(Ms(t0, t1));
        if (first.terminalOutput) {
            curves = first.terminalOutput->totalCurves;
            cvs = first.terminalOutput->totalCvs;
        }
        // Warm: same topology, every chunk dirty.
        for (int i = 0; i < 3; ++i) {
            ForceFullDirty(g);
            uint64_t const a0 = Allocs();
            t0 = Clock::now();
            sched.Run(g, ctx, ++gen);
            t1 = Clock::now();
            warmAllocs.v.push_back(double(Allocs() - a0));
            warm.v.push_back(Ms(t0, t1));
        }
        // Parameter edit: one value-class change, incremental recompile + run.
        for (int i = 0; i < 3; ++i) {
            UsdGenGraphDesc const &next = (i % 2 == 0) ? b : a;
            uint64_t const a0 = Allocs();
            t0 = Clock::now();
            UsdGenCompileResult const rr = c.Recompile(next, &g);
            ctx.desc = &g.Desc();
            if (rr.ok) sched.Run(g, ctx, ++gen);
            t1 = Clock::now();
            editAllocs.v.push_back(double(Allocs() - a0));
            edit.v.push_back(Ms(t0, t1));
        }
    }
    std::printf("engine: threads=%d curves=%zu cvs=%zu\n", threads, curves, cvs);
    Report("compile.cold", compile);
    Report("cook.cold", coldCook);
    Report("cook.warm", warm);
    Report("cook.paramEdit", edit);
    Report("allocs.cook.warm", warmAllocs, "allocs");
    Report("allocs.cook.paramEdit", editAllocs, "allocs");
}

// ---- Hydra -----------------------------------------------------------------

std::string MakeStageText()
{
    std::string s = "#usda 1.0\n";
    s += "def Mesh \"Scalp\"\n{\n    int[] faceVertexCounts = [";
    for (int f = 0; f < kNX * kNY; ++f) s += f ? ", 4" : "4";
    s += "]\n    int[] faceVertexIndices = [";
    bool firstIndex = true;
    for (int j = 0; j < kNY; ++j)
        for (int i = 0; i < kNX; ++i) {
            int const a = j * (kNX + 1) + i;
            int const q[4] = {a, a + 1, a + kNX + 2, a + kNX + 1};
            for (int k = 0; k < 4; ++k) {
                if (!firstIndex) s += ", ";
                firstIndex = false;
                s += std::to_string(q[k]);
            }
        }
    s += "]\n    point3f[] points = [";
    char buf[64];
    for (int j = 0; j <= kNY; ++j)
        for (int i = 0; i <= kNX; ++i) {
            std::snprintf(buf, sizeof(buf), "%s(%g, 0, %g)",
                          (i || j) ? ", " : "", i * 0.1, j * 0.1);
            s += buf;
        }
    s += "]\n}\n";
    s += R"USDA(
def Scope "Groom"
{
    def UsdGenDescription "Description"
    {
        rel usdGen:surface = </Scalp>
        def Scope "Ops"
        {
            def UsdGenWidth "width"
            {
                float usdGen:width = 0.02
            }
            def UsdGenLength "length"
            {
                uniform token usdGen:length:mode = "scale"
                float usdGen:length:value = 1.2
            }
            def UsdGenNoise "noise"
            {
                float usdGen:noise:magnitude = 0.05
                float usdGen:noise:frequency = 3
                int usdGen:noise:octaves = 2
                float usdGen:noise:correlation = 0.5
                uniform int usdGen:seed = 44
            }
            def UsdGenGrow "grow"
            {
                float usdGen:length = 1
                int usdGen:segments = 8
                uniform int usdGen:seed = 43
            }
            def UsdGenScatter "scatter"
            {
                float usdGen:density = 100
                uniform int usdGen:seed = 42
            }
        }
    }
}
)USDA";
    return s;
}

struct PullCounts { size_t tiles = 0, curves = 0, points = 0; };

PullCounts Pull(HdSceneIndexBaseRefPtr const &groom)
{
    PullCounts out;
    SdfPath const render("/Groom/Description/__usdGenRender");
    HdDataSourceLocator const pointsLoc(TfToken("primvars"), TfToken("points"), TfToken("primvarValue"));
    HdDataSourceLocator const widthsLoc(TfToken("primvars"), TfToken("widths"), TfToken("primvarValue"));
    HdDataSourceLocator const countsLoc(TfToken("basisCurves"), TfToken("topology"), TfToken("curveVertexCounts"));
    for (SdfPath const &tile : groom->GetChildPrimPaths(render)) {
        HdSceneIndexPrim const prim = groom->GetPrim(tile);
        ++out.tiles;
        auto value = [&](HdDataSourceLocator const &loc) {
            HdSampledDataSourceHandle ds = HdSampledDataSource::Cast(
                HdContainerDataSource::Get(prim.dataSource, loc));
            return ds ? ds->GetValue(0.0f) : VtValue();
        };
        VtValue const p = value(pointsLoc);
        VtValue const w = value(widthsLoc);
        VtValue const n = value(countsLoc);
        if (p.IsHolding<VtVec3fArray>()) out.points += p.UncheckedGet<VtVec3fArray>().size();
        if (n.IsHolding<VtIntArray>()) out.curves += n.UncheckedGet<VtIntArray>().size();
        (void)w;
    }
    return out;
}

void HydraBench(int runs)
{
    UsdStageRefPtr stage = UsdStage::CreateInMemory("benchRestructure.usda");
    auto tb = Clock::now();
    if (!stage->GetRootLayer()->ImportFromString(MakeStageText())) {
        std::printf("hydra: stage import failed\n");
        return;
    }
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices indices = UsdImagingCreateSceneIndices(info);
    HdSceneIndexBaseRefPtr groom = UsdGenGroomSceneIndex::New(indices.finalSceneIndex);
    auto *owner = dynamic_cast<UsdGenGroomSceneIndex *>(groom.operator->());
    if (!owner) { std::printf("hydra: no groom scene index\n"); return; }
    auto t0 = Clock::now();
    owner->Synchronize();
    auto t1 = Clock::now();
    std::printf("hydra: stage build %.1f ms, first synchronize %.1f ms\n", Ms(tb, t0), Ms(t0, t1));
    PullCounts const first = Pull(groom);
    std::printf("hydra: tiles=%zu curves=%zu points=%zu\n", first.tiles, first.curves, first.points);
    if (!first.curves) { std::printf("hydra: groom published nothing\n"); return; }

    Stat publish, pull, pullAllocs, publishAllocs;
    UsdGenEngineAllocCounting(true);
    UsdAttribute length = stage->GetPrimAtPath(SdfPath("/Groom/Description/Ops/grow"))
                              .GetAttribute(TfToken("usdGen:length"));
    for (int r = 0; r < runs * 3; ++r) {
        uint64_t a0 = Allocs();
        t0 = Clock::now();
        length.Set(r % 2 ? 1.0f : 1.1f);
        indices.stageSceneIndex->ApplyPendingUpdates();
        owner->Synchronize();
        t1 = Clock::now();
        publishAllocs.v.push_back(double(Allocs() - a0));
        publish.v.push_back(Ms(t0, t1));
        a0 = Allocs();
        t0 = Clock::now();
        PullCounts const c = Pull(groom);
        t1 = Clock::now();
        pullAllocs.v.push_back(double(Allocs() - a0));
        pull.v.push_back(Ms(t0, t1));
        if (c.curves != first.curves) std::printf("hydra: curve count moved %zu -> %zu\n", first.curves, c.curves);
    }
    Report("hydra.publish", publish);
    Report("hydra.pull", pull);
    Report("allocs.hydra.publish", publishAllocs, "allocs");
    Report("allocs.hydra.pull", pullAllocs, "allocs");
}

}  // namespace

int main(int argc, char **argv)
{
    int threads = 0, runs = 5;
    bool hydra = true;
    for (int i = 1; i < argc; ++i) {
        std::string const a = argv[i];
        if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--runs" && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
        else if (a == "--no-hydra") hydra = false;
    }
    if (threads <= 0) threads = int(std::max(1u, std::thread::hardware_concurrency()));
    EngineBench(threads, runs);
    if (hydra) HydraBench(runs);
    return 0;
}
