// testUsdGenScatterDensity.cpp — T0: scatter honors the per-face density
// multiplier (UsdGenSurfaceDesc::densityMultiplier), the brush's
// usdGen:paint:density channel. Empty == all 1.0; a zero face emits
// nothing; scales multiply; malformed multipliers fail closed; and the
// multiplier joins the capture digest so a repaint recooks instead of
// reading back cached pre-stroke roots.
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/ops/scatter.h"
#include "usdGen/scheduler.h"
#include "usdGen/sessionCooker.h"

#include <cstdio>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

static UsdGenGraphDesc Desc() {
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom");
    // Two unit right triangles (area 0.5 each): density 20 -> exactly 10
    // roots per face (integral expectation, no stochastic rounding).
    UsdGenSurfaceDesc scalp;
    scalp.path = SdfPath("/Scalp");
    scalp.restPoints = {{0,0,0},{1,0,0},{0,1,0},
                        {2,0,0},{3,0,0},{2,1,0}};
    scalp.points = scalp.restPoints;
    scalp.faceVertexCounts = {3,3};
    scalp.faceVertexIndices = {0,1,2,3,4,5};
    d.surfaces.push_back(scalp);
    UsdGenNodeDesc node;
    node.path = SdfPath("/Scatter");
    node.type = TfToken("UsdGenScatter");
    node.surfaces = {scalp.path};
    node.params = {{TfToken("density"), VtValue(20.0), false}};
    d.nodes.push_back(node);
    d.terminal = node.path;
    return d;
}

static bool Run(UsdGenGraphDesc const& d, UsdGenCurveBuffer* result) {
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    auto compiled = compiler.Compile(d, &graph);
    if (!compiled.ok) return false;
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    auto run = scheduler.Run(graph, ctx, 1);
    for (auto const& e : run.diagnostics.errors)
        std::fprintf(stderr, "%s\n", e.c_str());
    if (run.diagnostics.HasErrors()) return false;
    *result = graph.Output();
    return true;
}

static void AddGrow(UsdGenGraphDesc* d) {
    UsdGenNodeDesc grow;
    grow.path = SdfPath("/Grow");
    grow.type = TfToken("UsdGenGrow");
    grow.surfaces = {SdfPath("/Scalp")};
    grow.inputs = {SdfPath("/Scatter")};
    grow.params = {{TfToken("seed"), VtValue(19), false},
                   {TfToken("segments"), VtValue(5), false},
                   {TfToken("length"), VtValue(0.25f), false}};
    d->nodes.push_back(grow);
    d->terminal = grow.path;
}

static bool ReRun(UsdGenGraphDesc const& d0, UsdGenGraphDesc const& d1,
                  UsdGenCurveBuffer* result, bool* plansEmpty) {
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    if (!compiler.Compile(d0, &graph).ok) return false;
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    auto run0 = scheduler.Run(graph, ctx, 1);
    if (run0.diagnostics.HasErrors()) return false;
    if (!compiler.Recompile(d1, &graph).ok) return false;
    ctx.desc = &graph.Desc();
    auto run1 = scheduler.Run(graph, ctx, 1);
    for (auto const& e : run1.diagnostics.errors)
        std::fprintf(stderr, "%s\n", e.c_str());
    if (run1.diagnostics.HasErrors()) return false;
    *result = graph.Output();
    *plansEmpty = true;
    for (size_t i = 0; i < graph.NodeCount(); ++i)
        *plansEmpty = *plansEmpty &&
            graph.Node(static_cast<UsdGenNodeId>(i)).chunks.empty();
    return true;
}

int main() {
    usdGenRegisterM1Operators();

    {
        // Production cook loop: sequential novel density recooks
        // (the live drag) must neither crash nor serve stale roots.
        auto d = Desc();
        AddGrow(&d);
        UsdGenSessionCooker cooker(2, 1024 * 1024);
        UsdGenGenerationConstPtr prev;
        UsdGenStats stats;
        uint64_t epoch = 0;
        auto cook = [&](float m0, float m1, size_t want) {
            d.surfaces[0].densityMultiplier = {m0, m1};
            auto shared = std::make_shared<const UsdGenGraphDesc>(d);
            UsdGenGenerationConstPtr gen = cooker.Cook(shared,
                UsdGenContext::Interactive, false, {}, 0.0,
                UsdGenCommitReason::NoticeBatchEnd, prev, stats,
                false, epoch, epoch + 1, -1);
            if (!gen) return false;
            size_t curves = 0;
            for (auto const& tile : gen->tiles)
                curves += tile.curveVertexCounts.size();
            auto candidate = cooker.TakeCacheCandidate();
            if (candidate) cooker.CommitCacheCandidate(*candidate);
            prev = gen;
            stats = cooker.Stats();
            ++epoch;
            return curves == want;
        };
        CHECK(cook(1.0f, 1.0f, 20));
        CHECK(cook(1.0f, 0.0f, 10));
        CHECK(cook(0.0f, 0.0f, 0));
        CHECK(cook(1.0f, 1.0f, 20));
    }
    UsdGenCurveBuffer out;
    CHECK(Run(Desc(), &out));
    CHECK(out.totalCurves == 20);
    {
        auto d = Desc();
        d.surfaces[0].densityMultiplier = {1.0f, 0.0f};
        CHECK(Run(d, &out));
        CHECK(out.totalCurves == 10);
    }
    {
        auto d = Desc();
        d.surfaces[0].densityMultiplier = {0.0f, 0.0f};
        CHECK(Run(d, &out));
        CHECK(out.totalCurves == 0);
    }
    {
        auto d = Desc();
        d.surfaces[0].densityMultiplier = {2.0f, 0.5f};
        CHECK(Run(d, &out));
        CHECK(out.totalCurves == 25);
    }
    {
        auto d = Desc();
        d.surfaces[0].densityMultiplier = {1.0f};
        CHECK(!Run(d, &out));
    }
    {
        auto d = Desc();
        d.surfaces[0].densityMultiplier = {1.0f, -1.0f};
        CHECK(!Run(d, &out));
    }
    {
        UsdGenScatterOp op;
        auto a = Desc(), b = Desc();
        b.surfaces[0].densityMultiplier = {1.0f, 0.5f};
        UsdGenCaptureContext ctxA, ctxB;
        ctxA.desc = &a; ctxB.desc = &b;
        CHECK(op.CaptureDigest(ctxA) != op.CaptureDigest(ctxB));
        auto c = Desc();
        UsdGenCaptureContext ctxC;
        ctxC.desc = &c;
        CHECK(op.CaptureDigest(ctxA) == op.CaptureDigest(ctxC));
    }
    {
        auto d = Desc();
        d.surfaces[0].densityMultiplier = {0.0f, 0.0f};
        UsdGenNodeDesc grow;
        grow.path = SdfPath("/Grow");
        grow.type = TfToken("UsdGenGrow");
        grow.surfaces = {SdfPath("/Scalp")};
        grow.inputs = {SdfPath("/Scatter")};
        grow.params = {{TfToken("seed"), VtValue(19), false},
                       {TfToken("segments"), VtValue(5), false},
                       {TfToken("length"), VtValue(0.25f), false}};
        d.nodes.push_back(grow);
        d.terminal = grow.path;
        CHECK(Run(d, &out));
        CHECK(out.totalCurves == 0);
    }
    {
        // Sweeping the stale plan against empty buffers reads out of
        // bounds (the zero-density paint segfault).
        auto full = Desc();
        full.surfaces[0].densityMultiplier = {1.0f, 1.0f};
        AddGrow(&full);
        auto empty = full;
        empty.surfaces[0].densityMultiplier = {0.0f, 0.0f};
        bool plansEmpty = false;
        CHECK(ReRun(full, empty, &out, &plansEmpty));
        CHECK(out.totalCurves == 0);
        CHECK(plansEmpty);
    }
    {
        // Incremental recook 20 roots -> 10 (partial density): the
        // chunk plan follows the shrunken layout.
        auto full = Desc();
        full.surfaces[0].densityMultiplier = {1.0f, 1.0f};
        AddGrow(&full);
        auto part = full;
        part.surfaces[0].densityMultiplier = {1.0f, 0.0f};
        bool plansEmpty = true;
        CHECK(ReRun(full, part, &out, &plansEmpty));
        CHECK(out.totalCurves == 10);
        CHECK(!plansEmpty);
    }
    std::printf("ok: scatter density multiplier\n");
    return 0;
}
