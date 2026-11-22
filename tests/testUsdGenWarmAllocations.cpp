// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenWarmAllocations: the allocation-after-warm-up gate.
// After two warm-up cooks, a steady-state cook (same topology, every chunk
// dirty) and a value-only parameter edit must not allocate more than their
// budgets. The budgets ratchet down to zero; lower them when a change frees
// allocations, never raise them.

#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>

namespace {
std::atomic<uint64_t> g_allocs{0};
std::atomic<bool> g_counting{false};
inline void Count()
{
    if (g_counting.load(std::memory_order_relaxed))
        g_allocs.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

void *operator new(std::size_t n)
{
    Count();
    if (void *p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t n) { return operator new(n); }
void *operator new(std::size_t n, std::align_val_t a)
{
    Count();
    size_t const al = static_cast<size_t>(a);
    if (void *p = std::aligned_alloc(al, (n + al - 1) / al * al)) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t n, std::align_val_t a) { return operator new(n, a); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void *p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t, std::align_val_t) noexcept { std::free(p); }

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

// Budgets measured on the restructure branch; ratchet only downward.
// Target: 0 for both (plan: zero heap allocation per steady-state cook).
constexpr uint64_t kWarmCookBudget = 79;   // BUDGET_WARM
constexpr uint64_t kParamEditBudget = 115;  // BUDGET_EDIT

UsdGenGraphDesc MakeDesc(double magnitude)
{
    constexpr int NX = 100, NY = 100;   // 100 area units at density 100: ~10k roots
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/width");
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.restPoints = VtVec3fArray((NX + 1) * (NY + 1));
    s.uv = VtVec2fArray((NX + 1) * (NY + 1));
    for (int j = 0; j <= NY; ++j)
        for (int i = 0; i <= NX; ++i) {
            int const k = j * (NX + 1) + i;
            s.restPoints[k] = GfVec3f(i * 0.1f, j * 0.1f, 0.0f);
            s.uv[k] = GfVec2f(float(i) / NX, float(j) / NY);
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
    d.surfaces.push_back(std::move(s));
    auto add = [&](char const *name, char const *type, char const *input) {
        UsdGenNodeDesc n;
        n.path = SdfPath(std::string("/groom/") + name);
        n.type = TfToken(type);
        n.enabled = true;
        if (*input) n.inputs.push_back(SdfPath(std::string("/groom/") + input));
        else n.surfaces.push_back(SdfPath("/groom/surface"));
        d.nodes.push_back(std::move(n));
    };
    add("scatter", "UsdGenScatter", "");
    add("grow", "UsdGenGrow", "scatter");
    add("noise", "UsdGenNoise", "grow");
    add("width", "UsdGenWidth", "noise");
    for (auto &n : d.nodes) {
        if (n.type == TfToken("UsdGenGrow"))
            n.params.push_back({TfToken("segments"), VtValue(8), false});
        if (n.type == TfToken("UsdGenNoise"))
            n.params.push_back({TfToken("noise:magnitude"), VtValue(magnitude), false});
        if (n.type == TfToken("UsdGenWidth"))
            n.params.push_back({TfToken("width"), VtValue(0.02), false});
    }
    return d;
}

template <class Fn>
uint64_t Counted(Fn &&fn)
{
    g_allocs.store(0);
    g_counting.store(true);
    fn();
    g_counting.store(false);
    return g_allocs.load();
}

}  // namespace

int main()
{
    usdGenRegisterM1Operators();
    UsdGenGraphDesc const a = MakeDesc(0.05), b = MakeDesc(0.06);
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    if (!compiler.Compile(a, &graph).ok) {
        std::printf("FAIL: compile\n");
        return 1;
    }
    UsdGenScheduler scheduler(4);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
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
        compiler.Recompile(i % 2 ? a : b, &graph);
        ctx.desc = &graph.Desc();
        scheduler.Run(graph, ctx, ++gen);
    }
    compiler.Recompile(a, &graph);
    ctx.desc = &graph.Desc();
    scheduler.Run(graph, ctx, ++gen);

    fullDirty();
    uint64_t const warm = Counted([&] { scheduler.Run(graph, ctx, ++gen); });
    uint64_t const edit = Counted([&] {
        compiler.Recompile(b, &graph);
        ctx.desc = &graph.Desc();
        scheduler.Run(graph, ctx, ++gen);
    });
    int failures = 0;
    std::printf("warm cook allocations: %llu (budget %llu, target 0)\n",
                (unsigned long long)warm, (unsigned long long)kWarmCookBudget);
    std::printf("param edit allocations: %llu (budget %llu, target 0)\n",
                (unsigned long long)edit, (unsigned long long)kParamEditBudget);
    if (warm > kWarmCookBudget) { std::printf("FAIL: warm cook over budget\n"); ++failures; }
    if (edit > kParamEditBudget) { std::printf("FAIL: param edit over budget\n"); ++failures; }
    return failures ? 1 : 0;
}
