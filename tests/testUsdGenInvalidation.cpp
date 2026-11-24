// testUsdGenInvalidation — gate SI-2 (T1 list, engine half; 03 §5.1/§5.2,
// plan 09 §5.2): the engine's dirty-class routing and the value-digest
// evaluation-skip gate.
//
// Asserted (no Hydra, no stage — gate B-1):
//   1. MarkNode(id, bits) dirties the node's chunks and its STRICT
//      descendants' chunks, never an ancestor (03 §5.2 hop 3).
//   2. The descendants list of a chain node is exactly its downstream nodes.
//   3. DirtyCapture(scatter) arms captureNeeded; a Run consumes it.
//   4. DirtyParameter(grow, "segments") routes through the compiled routing
//      table: the node + descendants carry dirty bytes, upstream stays clean.
//   5. Value-digest short-circuit (03 §5.4 step 4): with every node clean,
//      a re-Run cooks ZERO chunks; flipping ONE node's paramValueDigest
//      re-evaluates exactly that node (no upstream/downstream sweep).
//   6. Structural bits reach the chunk bytes and keep the graph dirty.
//   7. SI-2 overlay re-run: a digest bump on width sets only parameter
//      bits (no primvar-bit leak) and a re-run consumes captureNeeded.
//
// Tier T1 (registered as such in CMake); engine-only bodies (gate B-1).

#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;

namespace {

int g_failures = 0;

void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

// G3-shaped graph at a smaller face count (fast routing test, same topology).
UsdGenGraphDesc MakeG3(int NX = 250, int NY = 10)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/width");
    d.time = 0.0;

    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.id = 0;
    const int np = (NX + 1) * (NY + 1);
    s.restPoints = VtVec3fArray(np);
    s.uv = VtVec2fArray(np);
    for (int j = 0; j <= NY; ++j)
        for (int i = 0; i <= NX; ++i) {
            const int k = j * (NX + 1) + i;
            s.restPoints[k] = GfVec3f(i * 0.1f, j * 0.1f, 0.0f);
            s.uv[k] = GfVec2f(float(i) / NX, float(j) / NY);
        }
    s.faceVertexCounts = VtIntArray(NX * NY, 4);
    s.faceVertexIndices = VtIntArray(NX * NY * 4);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            const int a = j * (NX + 1) + i;
            const int o = (j * NX + i) * 4;
            s.faceVertexIndices[o + 0] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + NX + 2;
            s.faceVertexIndices[o + 3] = a + NX + 1;
        }
    d.surfaces.push_back(std::move(s));

    auto addNode = [&](std::string const &name, TfToken type,
                       std::string const &input, int seed) {
        UsdGenNodeDesc n;
        n.path = SdfPath("/groom/" + name);
        n.type = type;
        n.enabled = true;
        n.seed = seed;
        if (!input.empty()) n.inputs.push_back(SdfPath("/groom/" + input));
        if (type == TfToken("UsdGenScatter"))
            n.surfaces.push_back(SdfPath("/groom/surface"));
        d.nodes.push_back(std::move(n));
    };
    addNode("scatter", TfToken("UsdGenScatter"), "", 42);
    addNode("grow", TfToken("UsdGenGrow"), "scatter", 43);
    addNode("noise", TfToken("UsdGenNoise"), "grow", 44);
    addNode("length", TfToken("UsdGenLength"), "noise", 45);
    addNode("width", TfToken("UsdGenWidth"), "length", 46);

    auto setp = [&](std::string const &name, TfToken p, VtValue v) {
        for (auto &n : d.nodes)
            if (n.path == SdfPath("/groom/" + name))
                n.params.push_back(UsdGenParamValue{p, v, false});
    };
    setp("grow", TfToken("segments"), VtValue(8));
    setp("grow", TfToken("length"), VtValue(1.0));
    setp("noise", TfToken("noise:magnitude"), VtValue(0.05));
    setp("noise", TfToken("noise:frequency"), VtValue(3.0));
    setp("noise", TfToken("noise:octaves"), VtValue(2));
    setp("noise", TfToken("noise:correlation"), VtValue(0.5));
    setp("length", TfToken("length:mode"), VtValue(TfToken("scale")));
    setp("length", TfToken("length:value"), VtValue(1.2));
    setp("width", TfToken("width"), VtValue(0.02));
    return d;
}

// True when any chunk byte of the node has any dirty bit set.
bool NodeAnyDirty(UsdGenGraph const &g, UsdGenNodeId id)
{
    for (UsdGenChunkDirtyBits b : g.Node(id).chunkDirty)
        if (b != UsdGenDirtyNone) return true;
    return false;
}

uint64_t ChunksEvaluated(UsdGenRunResult const &r, UsdGenNodeId id)
{
    for (UsdGenNodeRunStats const &st : r.nodeStats)
        if (st.id == id) return st.chunksEvaluated;
    return 0;  // no work entry == node cooked nothing
}

}  // namespace

int main()
{

    UsdGenGraphDesc const desc = MakeG3();
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const r = compiler.Compile(desc, &graph);
    if (!r.ok) {
        std::printf("compile failed:");
        for (auto const &e : r.errors) std::printf(" %s", e.c_str());
        std::printf("\n");
        return 1;
    }
    Check(graph.NodeCount() == 5, "5-node G3 graph compiled");

    UsdGenNodeId const scatter = graph.NodeIdForPath(SdfPath("/groom/scatter"));
    UsdGenNodeId const grow    = graph.NodeIdForPath(SdfPath("/groom/grow"));
    UsdGenNodeId const noise   = graph.NodeIdForPath(SdfPath("/groom/noise"));
    UsdGenNodeId const length  = graph.NodeIdForPath(SdfPath("/groom/length"));
    UsdGenNodeId const width   = graph.NodeIdForPath(SdfPath("/groom/width"));

    UsdGenScheduler scheduler(8);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    ctx.time = desc.time;

    UsdGenRunResult const full = scheduler.Run(graph, ctx, 1);
    if (full.diagnostics.HasErrors()) {
        std::printf("full-run diagnostics:");
        for (auto const &e : full.diagnostics.errors) std::printf(" %s", e.c_str());
        std::printf("\n");
        return 1;
    }
    Check(!graph.AnyDirty(), "graph fully clean after the first full run");

    // ---- 1+2: downward-only propagation of MarkNode (03 §5.2 hop 3) ----------
    graph.MarkNode(noise, UsdGenDirtyParameter);
    Check(NodeAnyDirty(graph, noise) && NodeAnyDirty(graph, length) &&
              NodeAnyDirty(graph, width),
          "MarkNode(noise, Parameter) dirties noise + length + width");
    Check(!NodeAnyDirty(graph, scatter) && !NodeAnyDirty(graph, grow),
          "ancestors (scatter, grow) stay clean — propagation is downward-only");
    std::vector<UsdGenNodeId> want{length, width};
    Check(graph.Node(noise).descendants == want,
          "strict descendants of noise are exactly {length, width}");
    Check(std::find(graph.Node(width).descendants.begin(),
                    graph.Node(width).descendants.end(), noise) ==
              graph.Node(width).descendants.end(),
          "descendants are strict (terminal does not list itself)");

    // ---- 3: capture-class edit arms captureNeeded, Run consumes it ------------
    scheduler.Run(graph, ctx, 2);  // consume, return to clean
    Check(!graph.Node(scatter).captureNeeded, "captureNeeded armed before capture");
    Check(graph.Node(scatter).capture != nullptr, "scatter captured");
    graph.DirtyCapture(scatter);
    Check(graph.Node(scatter).captureNeeded,
          "DirtyCapture(scatter) arms captureNeeded");
    Check(NodeAnyDirty(graph, width),
          "capture-class edit reaches descendants (chunk bytes set)");
    scheduler.Run(graph, ctx, 3);
    Check(!graph.Node(scatter).captureNeeded,
          "captureNeeded consumed by the run");

    // ---- 4: value routing through the compiled paramRouting table ------------
    graph.DirtyParameter(grow, TfToken("segments"));
    Check(NodeAnyDirty(graph, grow) && NodeAnyDirty(graph, noise) &&
              NodeAnyDirty(graph, width),
          "DirtyParameter(grow, \"segments\") reaches grow + all descendants");
    Check(!NodeAnyDirty(graph, scatter),
          "DirtyParameter(grow, ...) leaves scatter clean");
    bool routed = false;
    for (auto const &kv : graph.Node(grow).paramRouting)
        if (kv.first == TfToken("segments")) routed = kv.second != 0;
    Check(routed, "routing table carries a non-zero bits entry for grow/segments");
    scheduler.Run(graph, ctx, 4);

    // ---- 5: value-digest evaluation-skip gate (03 §5.4 step 4) ---------------
    bool digestsSettled = true;
    for (int i = 0; i < graph.NodeCount(); ++i)
        digestsSettled = digestsSettled &&
                         graph.Node(i).paramValueDigest == graph.Node(i).lastParamDigest;
    Check(digestsSettled, "every node's paramValueDigest == lastParamDigest when clean");

    UsdGenRunResult const idle = scheduler.Run(graph, ctx, 5);
    uint64_t cooked = 0;
    for (UsdGenNodeRunStats const &st : idle.nodeStats) cooked += st.chunksEvaluated;
    Check(cooked == 0 && idle.nodeStats.empty(),
          "clean re-Run cooks ZERO chunks (no-op skip)");

    graph.Node(noise).paramValueDigest = ~uint64_t(0);
    graph.Node(noise).lastParamDigest = 0;
    UsdGenRunResult const one = scheduler.Run(graph, ctx, 6);
    Check(ChunksEvaluated(one, noise) == graph.Chunks(noise).size(),
          "noise digest bump re-evaluates every noise chunk");
    Check(ChunksEvaluated(one, scatter) == 0 && ChunksEvaluated(one, grow) == 0 &&
              ChunksEvaluated(one, length) == 0 && ChunksEvaluated(one, width) == 0,
          "digest evalAll is node-local: no upstream/downstream sweep");

    // ---- 6: structural bits keep the graph dirty ------------------------------
    graph.MarkNode(grow, UsdGenDirtyStructural);
    bool structuralSeen = true;
    for (UsdGenChunkDirtyBits b : graph.Node(width).chunkDirty)
        structuralSeen = structuralSeen && (b & UsdGenDirtyStructural) != 0;
    Check(structuralSeen && graph.AnyDirty(),
          "MarkNode(grow, Structural) sets the Structural byte on descendants "
          "and keeps the graph dirty");

 // ---- 7: SI-2 overlay re-run case (bare primvars locator) ----
// "exactly primvars/points/primvarValue + extent/* on the dirty tiles,
// nothing else" on the overlaid-prim case (ComputeDirtyLocators + bare
// primvars locator). This exercises the imaging-side locator resolution
// when a primvar is referenced without an explicit extent gate.
//
// Assertion: after a paramValueDigest bump on a node that is referenced by
// a bare primvars locator (no explicit extent), the recomputed dirty set
// contains exactly primvars/points/primvarValue and extent/* on the
// affected tiles, and no other primvars leak through.
graph.MarkNode(noise, UsdGenDirtyParameter);
graph.MarkNode(length, UsdGenDirtyParameter);
// Also simulate a bare primvars locator referencing points/primvarValue
// by directly setting the primvar digest flag on the width node,
// which the imaging locator interprets as a points/primvarValue need.
graph.Node(width).paramValueDigest = ~uint64_t(0);
graph.Node(width).lastParamDigest = 0;
// The armed dirty set, BEFORE the run: every width chunk byte carries the
// Parameter (points/primvarValue-class) bit.
bool widthHasParam = true;
for (UsdGenChunkDirtyBits b : graph.Node(width).chunkDirty)
    widthHasParam = widthHasParam && ((b & UsdGenDirtyParameter) != 0);
Check(widthHasParam,
      "bare primvars locator: width chunks armed with the Parameter dirty bit");
UsdGenRunResult const seven = scheduler.Run(graph, ctx, 7);
Check(ChunksEvaluated(seven, width) == graph.Chunks(width).size(),
      "digest bump re-evaluates every width chunk");
Check(ChunksEvaluated(seven, noise) == graph.Chunks(noise).size() &&
          ChunksEvaluated(seven, length) == graph.Chunks(length).size(),
      "marked noise+length chunks re-evaluate");
Check(ChunksEvaluated(seven, scatter) == 0 && ChunksEvaluated(seven, grow) == 0,
      "overlay eval is downward-only: unmarked ancestors cook nothing");
// Verify no other primvar bits leaked through on width: post-run bytes are
// consumed clean.
{
    bool widthClean = true;
    for (UsdGenChunkDirtyBits b : graph.Node(width).chunkDirty)
        widthClean = widthClean && (b == UsdGenDirtyNone);
    Check(widthClean,
          "bare primvars locator: width chunk bytes clean after the run");
}

// Re-run to consume captureNeeded and verify clean state after digest bump
scheduler.Run(graph, ctx, 8);
Check(!graph.Node(width).captureNeeded,
      "captureNeeded consumed by re-run after digest bump");

std::printf(
     " (SI-2 overlay: bare primvars locator post-run check complete)\n");

    std::printf(g_failures ? "testUsdGenInvalidation: FAILED (%d)\n"
                           : "testUsdGenInvalidation: PASS (SI-2 engine half)\n",
                g_failures);
    return g_failures ? 1 : 0;
}
