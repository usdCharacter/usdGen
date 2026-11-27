// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenWarmAllocations: the allocation-after-warm-up gate.
// After two warm-up cooks, a steady-state cook (same topology, every chunk
// dirty) and a value-only parameter edit must not allocate more than their
// budgets. The budgets ratchet down to zero; lower them when a change frees
// allocations, never raise them. Neither may intern a token (step-4 gate 4).
// Two graphs at 10k curves: the bench's G3 chain (scatter -> grow -> noise ->
// length -> width) and a re-capture chain (scatter -> grow -> noise ->
// resample -> twist -> width) whose noise edit re-captures Resample (an
// owning capture) and Twist (a whole-strand op) because their input moved.
// Clump, GuideInterpolate and Collide are not in it: their capture bodies
// still allocate (PLAN.md leftovers). Every measured cook must also
// reproduce a fresh compile's output bit for bit, and recycling must never
// write into a buffer something still holds.

#include "utils/pxrInterop.h"
#include "usdGen/allocCounter.h"
#include "nodeSets.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGenUsd/lower.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/token.h"
#include "usdGen/scheduler.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// The count lives in usdGen (allocCounter.h). On ELF this replacement
// interposes on every module and feeds it; on Windows usdGen wraps every
// module's CRT heap imports instead, so no replacement is needed here.
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

// Budgets measured on the restructure branch; ratchet only downward.
// Target: 0 for both (plan: zero heap allocation per steady-state cook).
// Re-captures refill the node's spare capture and retired planes, and a
// value-only edit keeps the compiled node, so both reach zero.
constexpr uint64_t kWarmCookBudget = 0;   // BUDGET_WARM
constexpr uint64_t kParamEditBudget = 0;  // BUDGET_EDIT

void AddSurface(UsdGenGraphDesc *d)
{
    constexpr int NX = 100, NY = 100;   // 100 area units at density 100: ~10k roots
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.restPoints = usdGen::UsdGenVec3fArray((NX + 1) * (NY + 1));
    s.uv = usdGen::UsdGenVec2fArray((NX + 1) * (NY + 1));
    for (int j = 0; j <= NY; ++j)
        for (int i = 0; i <= NX; ++i) {
            int const k = j * (NX + 1) + i;
            s.restPoints[k] = usdGen::UsdGenVec3f(i * 0.1f, j * 0.1f, 0.0f);
            s.uv[k] = usdGen::UsdGenVec2f(float(i) / NX, float(j) / NY);
        }
    s.faceVertexCounts = VtIntArray(NX * NY, 4);
    s.faceVertexIndices = VtIntArray(NX * NY * 4);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            int const a = j * (NX + 1) + i, o = (j * NX + i) * 4;
            s.faceVertexIndices[o + 0] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + NX + 2;
            s.faceVertexIndices[o + 3] = a + NX + 1;
        }
    d->surfaces.push_back(std::move(s));
}

UsdGenNodeDesc &AddNode(UsdGenGraphDesc *d, char const *name, char const *type, char const *input)
{
    UsdGenNodeDesc n;
    n.path = SdfPath(std::string("/groom/") + name);
    n.type = TfToken(type);
    n.enabled = true;
    if (*input) n.inputs.push_back(SdfPath(std::string("/groom/") + input));
    else n.surfaces.push_back(SdfPath("/groom/surface"));
    d->nodes.push_back(std::move(n));
    return d->nodes.back();
}

UsdGenGraphDesc MakeDesc(double magnitude)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/width");
    AddSurface(&d);
    auto add = [&](char const *name, char const *type, char const *input) {
        AddNode(&d, name, type, input);
    };
    add("scatter", "UsdGenScatter", "");
    add("grow", "UsdGenGrow", "scatter");
    add("noise", "UsdGenNoise", "grow");
    add("length", "UsdGenLength", "noise");
    add("width", "UsdGenWidth", "length");
    for (auto &n : d.nodes) {
        if (n.type == TfToken("UsdGenGrow"))
            n.params.push_back({TfToken("segments"), VtValue(8), false});
        if (n.type == TfToken("UsdGenNoise"))
            n.params.push_back({TfToken("noise:magnitude"), VtValue(magnitude), false});
        if (n.type == TfToken("UsdGenLength")) {
            n.params.push_back({TfToken("length:mode"), VtValue(TfToken("scale")), false});
            n.params.push_back({TfToken("length:value"), VtValue(1.2), false});
        }
        if (n.type == TfToken("UsdGenWidth"))
            n.params.push_back({TfToken("width"), VtValue(0.02), false});
    }
    return d;
}

// The re-capture chain over the same surface. Twist keeps its defaults (no
// twist angles), so its Compute returns at its gate and the chain measures
// the whole-strand capture itself.
UsdGenGraphDesc MakeRecaptureDesc(double magnitude)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/width");
    AddSurface(&d);
    AddNode(&d, "scatter", "UsdGenScatter", "");
    AddNode(&d, "grow", "UsdGenGrow", "scatter")
        .params.push_back({TfToken("segments"), VtValue(8), false});
    AddNode(&d, "noise", "UsdGenNoise", "grow")
        .params.push_back({TfToken("noise:magnitude"), VtValue(magnitude), false});
    AddNode(&d, "resample", "UsdGenResample", "noise")
        .params.push_back({TfToken("cvCount"), VtValue(6), false});
    AddNode(&d, "twist", "UsdGenTwist", "resample");
    AddNode(&d, "width", "UsdGenWidth", "twist")
        .params.push_back({TfToken("width"), VtValue(0.02), false});
    return d;
}

// USDGEN_ALLOC_TRACE=1 prints each counted allocation's stack as
// module+offset (resolve with the module's map or pdb).
void PrintTrace(char const *what, uint64_t count)
{
    for (int slot = 0; slot < int(count) && slot < kUsdGenAllocTraceSlots; ++slot) {
        void *frames[kUsdGenAllocTraceFrames];
        size_t bytes = 0;
        int const n = UsdGenEngineAllocTrace(slot, frames, kUsdGenAllocTraceFrames, &bytes);
        std::printf("%s alloc %d: %zu bytes\n", what, slot, bytes);
        for (int i = 0; i < n; ++i) {
            char name[260] = "?";
            uintptr_t base = 0;
#ifdef _WIN32
            HMODULE mod = nullptr;
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   static_cast<char const *>(frames[i]), &mod)) {
                GetModuleFileNameA(mod, name, sizeof(name));
                base = reinterpret_cast<uintptr_t>(mod);
            }
#else
            Dl_info info;
            if (dladdr(frames[i], &info) && info.dli_fname) {
                std::snprintf(name, sizeof(name), "%s", info.dli_fname);
                base = reinterpret_cast<uintptr_t>(info.dli_fbase);
            }
#endif
            char const *leaf = name;
            for (char const *c = name; *c; ++c)
                if (*c == '/' || *c == '\\') leaf = c + 1;
            std::printf("    %s+0x%llx\n", leaf,
                        (unsigned long long)(reinterpret_cast<uintptr_t>(frames[i]) - base));
        }
    }
}

// The terminal output of a fresh compile and cook of `desc`: what the
// recycled graph must reproduce bit for bit.
UsdGenCurveBuffer FreshCook(UsdGenGraphDesc const &desc)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    if (!usdGenUsd::Compile(compiler, desc, &graph).ok) return {};
    UsdGenScheduler scheduler(4);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Spec();
    scheduler.Run(graph, ctx, 1);
    return OutputBuffer(graph);
}

template <class T>
bool SameBits(VtArray<T> const &a, VtArray<T> const &b)
{
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.cdata(), b.cdata(), a.size() * sizeof(T)) == 0);
}

template <class T>
bool SameBits(usdGen::UsdGenCowArray<T> const &a, usdGen::UsdGenCowArray<T> const &b)
{
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.cdata(), b.cdata(), a.size() * sizeof(T)) == 0);
}

bool SameOutput(UsdGenCurveBuffer const &a, UsdGenCurveBuffer const &b)
{
    return a.totalCurves == b.totalCurves && a.totalCvs == b.totalCvs &&
        SameBits(a.px, b.px) && SameBits(a.py, b.py) && SameBits(a.pz, b.pz) &&
        SameBits(a.width, b.width) && SameBits(a.hairT, b.hairT) &&
        SameBits(a.rest, b.rest) && SameBits(a.curveId, b.curveId);
}

template <class Fn>
uint64_t Counted(char const *what, Fn &&fn)
{
    bool const trace = std::getenv("USDGEN_ALLOC_TRACE") != nullptr;
    UsdGenEngineAllocTracing(trace);
    UsdGenEngineAllocCounting(true);
    fn();
    UsdGenEngineAllocCounting(false);
    uint64_t const n = UsdGenEngineAllocCount();
    if (trace) PrintTrace(what, n);
    return n;
}

}  // namespace

void RegisterLateOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenLateOp";
    d.displayName = "LateOp";
    d.create = [](UsdGenOpRecord const&, void*) -> std::unique_ptr<UsdGenOp> { return nullptr; };
    r.RegisterOp(d);
}

namespace {

// Warm-up, then the measured warm cook and value edit of a -> b, the
// fresh-cook comparisons and the held-output check on node 'heldNode'.
// 'recaptured' nodes (paths) must re-capture in the measured param edit.
int Gate(char const *name, UsdGenGraphDesc const &a, UsdGenGraphDesc const &b, int heldNode,
         std::vector<char const *> const &recaptured = {})
{
    std::printf("== %s\n", name);
    // The host lowers once; the measured edit is the engine's Recompile.
    UsdGenGraphSpec const specA = usdGenUsd::Lower(a), specB = usdGenUsd::Lower(b);
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    if (!usdGenUsd::Compile(compiler, a, &graph).ok) {
        std::printf("FAIL: compile\n");
        return 1;
    }
    UsdGenScheduler scheduler(4);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Spec();
    uint64_t gen = 0;
    auto fullDirty = [&] {
        for (int i = 0; i < graph.NodeCount(); ++i)
            for (auto &word : graph.Node(i).chunkDirty) word = UsdGenDirtyParameter;
    };
    // Warm up: first cook, then two steady cooks and two edit round trips.
    scheduler.Run(graph, ctx, ++gen);
    for (int i = 0; i < 2; ++i) {
        fullDirty();
        scheduler.Run(graph, ctx, ++gen);
        compiler.Recompile(i % 2 ? specA : specB, &graph);
        ctx.desc = &graph.Spec();
        scheduler.Run(graph, ctx, ++gen);
    }
    compiler.Recompile(specA, &graph);
    ctx.desc = &graph.Spec();
    scheduler.Run(graph, ctx, ++gen);

    fullDirty();
    uint64_t warmInterns = 0, editInterns = 0;
    uint64_t const warm = Counted("warm", [&] {
        uint64_t const before = UsdGenTotalInternCount();
        scheduler.Run(graph, ctx, ++gen);
        warmInterns = UsdGenTotalInternCount() - before;
    });
    std::vector<UsdGenCapture const *> captures;
    for (char const *path : recaptured)
        captures.push_back(graph.Node(usdGenUsd::NodeIdForPath(graph, SdfPath(path))).capture.get());
    uint64_t const edit = Counted("edit", [&] {
        uint64_t const before = UsdGenTotalInternCount();
        compiler.Recompile(specB, &graph);
        ctx.desc = &graph.Spec();
        scheduler.Run(graph, ctx, ++gen);
        editInterns = UsdGenTotalInternCount() - before;
    });
    int failures = 0;
    for (size_t i = 0; i < recaptured.size(); ++i) {
        UsdGenCompiledNode const &node = graph.Node(usdGenUsd::NodeIdForPath(graph, SdfPath(recaptured[i])));
        if (!node.capture || node.capture.get() == captures[i]) {
            std::printf("FAIL: %s did not re-capture in the param edit\n", recaptured[i]);
            ++failures;
        }
    }
    std::printf("warm cook interns: %llu, param edit interns: %llu (target 0)\n",
                (unsigned long long)warmInterns, (unsigned long long)editInterns);
    if (warmInterns != 0) { std::printf("FAIL: the warm cook interned a token\n"); ++failures; }
    if (editInterns != 0) { std::printf("FAIL: the param edit interned a token\n"); ++failures; }
    // Recycled storage must not change a single bit of the output.
    UsdGenCurveBuffer const out = OutputBuffer(graph);
    if (out.totalCurves == 0 || !SameOutput(out, FreshCook(b))) {
        std::printf("FAIL: param edit output differs from a fresh cook\n");
        ++failures;
    }
    fullDirty();
    scheduler.Run(graph, ctx, ++gen);
    if (!SameOutput(OutputBuffer(graph), FreshCook(b))) {
        std::printf("FAIL: warm cook output differs from a fresh cook\n");
        ++failures;
    }
    // A buffer something still holds (a published generation) is never a
    // recycling candidate: two more re-captures of Grow must leave it intact.
    {
        // The node's output set as a published generation would hold it
        // (Export marks its arenas shared outside the cook).
        UsdGenRef<const UsdGenElementSet> const heldSet = graph.Node(heldNode).output;
        if (heldSet) heldSet->Export();
        UsdGenCurveBuffer const copy = heldSet ? UsdGenCurveBufferFromSet(*heldSet) : UsdGenCurveBuffer();
        for (int i = 0; i < 2; ++i) {
            fullDirty();
            scheduler.Run(graph, ctx, ++gen);
        }
        UsdGenCurveBuffer const held = heldSet ? UsdGenCurveBufferFromSet(*heldSet) : UsdGenCurveBuffer();
        if (held.px.empty() || !SameOutput(held, copy)) {
            std::printf("FAIL: recycling wrote into a held buffer\n");
            ++failures;
        }
    }
    usdGenUsd::Recompile(compiler, a, &graph);
    ctx.desc = &graph.Spec();
    scheduler.Run(graph, ctx, ++gen);
    if (!SameOutput(OutputBuffer(graph), FreshCook(a)) ||
        SameOutput(FreshCook(a), FreshCook(b))) {
        std::printf("FAIL: edit back output differs from a fresh cook\n");
        ++failures;
    }
    std::printf("warm cook allocations: %llu (budget %llu, target 0)\n",
                (unsigned long long)warm, (unsigned long long)kWarmCookBudget);
    std::printf("param edit allocations: %llu (budget %llu, target 0)\n",
                (unsigned long long)edit, (unsigned long long)kParamEditBudget);
    if (warm > kWarmCookBudget) { std::printf("FAIL: warm cook over budget\n"); ++failures; }
    if (edit > kParamEditBudget) { std::printf("FAIL: param edit over budget\n"); ++failures; }
    return failures;
}

}  // namespace

int main()
{
    (void)UsdGenOpRegistry::Get();   // built-ins register on first use
    UsdGenFreezeRegistries();   // compile and exec read frozen registries
    if (UsdGenRegisterStatic(&RegisterLateOp)) {
        std::printf("FAIL: registration after freeze\n");
        return 1;
    }
    int failures = Gate("G3 chain", MakeDesc(0.05), MakeDesc(0.06), 1);
    failures += Gate("re-capture chain", MakeRecaptureDesc(0.05), MakeRecaptureDesc(0.06), 3,
                     {"/groom/resample", "/groom/twist"});
    return failures ? 1 : 0;
}
