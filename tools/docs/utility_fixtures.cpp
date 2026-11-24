// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Native, source-grounded operator fixtures for the reference manual.
// The values below are authored inputs; all displayed output planes come from
// the usdGen compiler and CPU scheduler, including Part's partId channel.
#include "usdGen/compiler.h"
#include "usdGen/curveBuffer.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"

#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usd/usdGeom/xform.h"
#include "pxr/usd/sdf/types.h"
#include "pxr/base/vt/types.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

void Require(bool condition, char const *message) {
    if (!condition) throw std::runtime_error(message);
}

UsdGenNodeDesc Source(SdfPath path, SdfPath curves) {
    UsdGenNodeDesc node;
    node.path = path;
    node.type = TfToken("UsdGenCurveSource");
    node.curves = {curves};
    node.params = {{TfToken("rebind"), VtValue(TfToken("never")), false},
                   {TfToken("useRest"), VtValue(true), false}};
    return node;
}

UsdGenCurveSetDesc Strands(SdfPath path, int nx, int ny, float spacing,
                           float step, float width) {
    UsdGenCurveSetDesc set;
    set.path = path;
    set.role = UsdGenRole::Curves;
    set.curveRole = TfToken("hair");
    int const n = nx * ny, cvs = 6;
    set.curveVertexCounts.assign(n, cvs);
    set.points.resize(n * cvs);
    set.widths.assign(n * cvs, width);
    set.curveId.resize(n);
    set.skinPrim.resize(n);
    set.skinPrimUv.assign(n, GfVec2f(0.2f, 0.3f));
    set.rootFrame.assign(n, GfMatrix4d(1.0));
    for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
        int const c = y * nx + x;
        float const rx = (x - (nx - 1) * 0.5f) * spacing;
        float const ry = (y - (ny - 1) * 0.5f) * spacing;
        set.curveId[c] = uint64_t(c);
        set.skinPrim[c] = c;
        for (int i = 0; i < cvs; ++i) {
            float const t = float(i) / float(cvs - 1);
            set.points[c * cvs + i] = GfVec3f(rx + 0.012f * t * t,
                                               ry + 0.007f * t * t,
                                               step * i);
        }
    }
    set.rest = set.points;
    return set;
}

UsdGenCurveBuffer Cook(UsdGenGraphDesc const &desc, UsdGenGraph *graph) {
    UsdGenCompiler compiler;
    UsdGenCompileResult const compiled = compiler.Compile(desc, graph);
    if (!compiled.ok) {
        for (auto const &e : compiled.errors) std::fprintf(stderr, "compile: %s\n", e.c_str());
        throw std::runtime_error("native fixture graph did not compile");
    }
    UsdGenScheduler scheduler(4);
    UsdGenEvalContext context;
    context.desc = &graph->Desc();
    UsdGenRunResult const run = scheduler.Run(*graph, context, 1);
    if (run.diagnostics.HasErrors()) {
        for (auto const &e : run.diagnostics.errors) std::fprintf(stderr, "cook: %s\n", e.c_str());
        throw std::runtime_error("native fixture graph did not cook");
    }
    return graph->Output();
}

void WriteCurves(UsdStageRefPtr const &stage, SdfPath path,
                 UsdGenCurveBuffer const &buffer, GfVec3f color,
                 GfVec3f translate = GfVec3f(0)) {
    Require(buffer.totalCurves > 0 && buffer.totalCvs > 0,
            "fixture produced empty curve buffer");
    Require(buffer.px.size() == buffer.totalCvs &&
            buffer.py.size() == buffer.totalCvs &&
            buffer.pz.size() == buffer.totalCvs,
            "fixture produced incomplete positions");
    UsdGeomBasisCurves curves = UsdGeomBasisCurves::Define(stage, path);
    VtVec3fArray points(buffer.totalCvs);
    for (size_t i = 0; i < points.size(); ++i)
        points[i] = GfVec3f(buffer.px[i], buffer.py[i], buffer.pz[i]) + translate;
    VtIntArray counts(buffer.totalCurves);
    if (buffer.cvOffsets.empty()) {
        Require(buffer.totalCvs % buffer.totalCurves == 0, "invalid uniform topology");
        counts.assign(buffer.totalCurves, int(buffer.totalCvs / buffer.totalCurves));
    } else {
        Require(buffer.cvOffsets.size() == size_t(buffer.totalCurves) + 1,
                "invalid ragged topology");
        for (size_t i = 0; i < counts.size(); ++i)
            counts[i] = buffer.cvOffsets[i + 1] - buffer.cvOffsets[i];
    }
    curves.CreatePointsAttr().Set(points);
    curves.CreateCurveVertexCountsAttr().Set(counts);
    curves.CreateTypeAttr().Set(UsdGeomTokens->linear);
    curves.CreateWrapAttr().Set(UsdGeomTokens->nonperiodic);
    if (buffer.width.size() == buffer.totalCvs) {
        curves.CreateWidthsAttr().Set(VtFloatArray(buffer.width.begin(), buffer.width.end()));
        curves.SetWidthsInterpolation(UsdGeomTokens->vertex);
    }
    VtVec3fArray colors(buffer.totalCvs, color);
    UsdGeomPrimvarsAPI(curves).CreatePrimvar(
        TfToken("displayColor"), SdfValueTypeNames->Color3fArray,
        UsdGeomTokens->vertex).Set(colors);
}

void ExportWidthBlend(UsdStageRefPtr const &stage) {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/WidthBlendGraph");
    desc.defaultWidth = .001f;
    desc.curveSets.push_back(Strands(SdfPath("/WidthBlendGraph/Hair"),
                                    19, 19, .008f, .035f, .001f));
    UsdGenNodeDesc source = Source(SdfPath("/WidthBlendGraph/Source"),
                                   desc.curveSets[0].path);
    UsdGenNodeDesc a;
    a.path = SdfPath("/WidthBlendGraph/WidthA");
    a.type = TfToken("UsdGenWidth");
    a.inputs = {source.path};
    a.params = {{TfToken("width"), VtValue(.0014f), false}};
    UsdGenNodeDesc b = a;
    b.path = SdfPath("/WidthBlendGraph/WidthB");
    b.params[0].value = VtValue(.0060f);
    UsdGenNodeDesc blend;
    blend.path = SdfPath("/WidthBlendGraph/Blend");
    blend.type = TfToken("UsdGenWidthBlend");
    blend.inputs = {a.path, b.path};
    blend.params = {{TfToken("widthBlend:weight"), VtValue(.5f), false}};
    desc.terminal = blend.path;
    desc.nodes = {blend, b, source, a}; // authored out of topological order on purpose
    UsdGenGraph graph;
    UsdGenCurveBuffer const merged = Cook(desc, &graph);
    UsdGenCurveBuffer const left = graph.Node(graph.NodeIdForPath(a.path)).buffer;
    UsdGenCurveBuffer const right = graph.Node(graph.NodeIdForPath(b.path)).buffer;
    Require(left.width.size() == merged.totalCvs &&
            right.width.size() == merged.totalCvs &&
            merged.width.size() == merged.totalCvs,
            "WidthBlend width planes incomplete");
    for (size_t i = 0; i < merged.width.size(); ++i)
        Require(std::fabs(merged.width[i] - .5f * (left.width[i] + right.width[i])) < 1e-6f,
                "WidthBlend cooked value differs from ordered midpoint");
    WriteCurves(stage, SdfPath("/World/Width_A"), left,
                GfVec3f(.48f, .32f, .23f), GfVec3f(-.27f, 0, 0));
    WriteCurves(stage, SdfPath("/World/Width_Blend"), merged,
                GfVec3f(.82f, .55f, .27f));
    WriteCurves(stage, SdfPath("/World/Width_B"), right,
                GfVec3f(.43f, .28f, .62f), GfVec3f(.27f, 0, 0));
    std::printf("WidthBlend native cook: %u curves/group, widths %.6f / %.6f / %.6f\n",
                merged.totalCurves, left.width[0], merged.width[0], right.width[0]);
}

void ExportPart(UsdStageRefPtr const &stage) {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/PartGraph");
    desc.defaultWidth = .0018f;
    desc.curveSets.push_back(Strands(SdfPath("/PartGraph/Hair"),
                                    41, 25, .006f, .033f, .0018f));
    UsdGenCurveSetDesc parting;
    parting.path = SdfPath("/PartGraph/PartingCurve");
    parting.role = UsdGenRole::Reference;
    parting.curveRole = TfToken("guide");
    parting.curveVertexCounts = {3};
    parting.points = {GfVec3f(-.16f, 0, 0), GfVec3f(0, 0, 0),
                      GfVec3f(.16f, 0, 0)};
    parting.rest = parting.points;
    desc.curveSets.push_back(parting);
    UsdGenNodeDesc source = Source(SdfPath("/PartGraph/Source"),
                                   desc.curveSets[0].path);
    UsdGenNodeDesc part;
    part.path = SdfPath("/PartGraph/Part");
    part.type = TfToken("UsdGenPart");
    part.inputs = {source.path};
    part.references = {parting.path};
    part.params = {{TfToken("part:radius"), VtValue(.085f), false},
                   {TfToken("part:strength"), VtValue(1.f), false}};
    desc.nodes = {source, part};
    desc.terminal = part.path;
    UsdGenGraph graph;
    UsdGenCurveBuffer const output = Cook(desc, &graph);
    UsdGenPlane const *ids = nullptr;
    for (auto const &plane : output.extraCurve)
        if (plane.name == TfToken("partId")) ids = &plane;
    Require(ids && ids->type == TfToken("int") &&
            ids->interpolation == TfToken("uniform") &&
            ids->i.size() == output.totalCurves,
            "Part did not publish a uniform int partId per strand");
    UsdGenCurveBuffer const &upstream = graph.Node(graph.NodeIdForPath(source.path)).buffer;
    Require(output.px == upstream.px && output.py == upstream.py &&
            output.pz == upstream.pz, "Part unexpectedly moved strand points");
    WriteCurves(stage, SdfPath("/World/Part_ID_Diagnostic"), output,
                GfVec3f(.5f));
    UsdGeomBasisCurves const curves(stage->GetPrimAtPath(
        SdfPath("/World/Part_ID_Diagnostic")));
    VtVec3fArray color(output.totalCurves);
    VtIntArray partIds(output.totalCurves);
    size_t side0 = 0, side1 = 0, outside = 0;
    for (size_t c = 0; c < color.size(); ++c) {
        int const id = ids->i[c];
        partIds[c] = id;
        if (id == 0) { color[c] = GfVec3f(.30f, .72f, .88f); ++side0; }
        else if (id == 1) { color[c] = GfVec3f(.97f, .44f, .26f); ++side1; }
        else { color[c] = GfVec3f(.46f, .45f, .44f); ++outside; }
    }
    Require(side0 && side1 && outside, "Part fixture does not cover all three partId states");
    UsdGeomPrimvarsAPI api(curves);
    api.CreatePrimvar(TfToken("displayColor"), SdfValueTypeNames->Color3fArray,
                      UsdGeomTokens->uniform).Set(color);
    api.CreatePrimvar(TfToken("partId"), SdfValueTypeNames->IntArray,
                      UsdGeomTokens->uniform).Set(partIds);
    std::printf("Part native cook: %zu side-0, %zu side-1, %zu outside; points unchanged\n",
                side0, side1, outside);
}

void ExportFreeze(UsdStageRefPtr const &stage) {
    // The same compiled graph runs twice. Its first frozen capture owns an
    // independent copy; then we edit the committed upstream buffer and dirty
    // only Freeze. This mirrors testUsdGenFreeze::CheckImmunitySameGraph.
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/FreezeGraph");
    desc.defaultWidth = .002f;
    UsdGenCurveSetDesc live = Strands(SdfPath("/FreezeGraph/Live"),
                                     21, 9, .009f, .036f, .002f);
    desc.curveSets = {live};
    UsdGenNodeDesc source = Source(SdfPath("/FreezeGraph/Source"), live.path);
    UsdGenNodeDesc freeze;
    freeze.path = SdfPath("/FreezeGraph/Freeze");
    freeze.type = TfToken("UsdGenFreeze");
    freeze.inputs = {source.path};
    freeze.params = {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}};
    desc.nodes = {source, freeze};
    desc.terminal = freeze.path;
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = compiler.Compile(desc, &graph);
    Require(compiled.ok, "Freeze fixture did not compile");
    UsdGenScheduler scheduler(4);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    UsdGenRunResult const run1 = scheduler.Run(graph, context, 1);
    Require(!run1.diagnostics.HasErrors(), "Freeze first cook failed");
    UsdGenNodeId const sourceId = graph.NodeIdForPath(source.path);
    UsdGenNodeId const freezeId = graph.NodeIdForPath(freeze.path);
    Require(sourceId != UsdGenGraph::InvalidNode &&
            freezeId != UsdGenGraph::InvalidNode, "Freeze nodes did not resolve");
    UsdGenCompiledNode &upstreamNode = graph.Node(sourceId);
    UsdGenCurveBuffer const &first = graph.Output();
    std::vector<float> firstX(first.px.begin(), first.px.end());
    std::vector<float> firstY(first.py.begin(), first.py.end());
    std::vector<float> firstZ(first.pz.begin(), first.pz.end());
    float *editedX = const_cast<float *>(upstreamNode.buffer.px.cdata());
    for (size_t i = 0; i < upstreamNode.buffer.px.size(); ++i) {
        float const t = float(i % 6) / 5.f;
        editedX[i] += .09f * t * t;
    }
    upstreamNode.buffer.valueVersion += 100;
    graph.MarkNode(freezeId, UsdGenDirtyParameter);
    UsdGenRunResult const run2 = scheduler.Run(graph, context, 2);
    Require(!run2.diagnostics.HasErrors(), "Freeze second cook failed");
    UsdGenCurveBuffer const &output = graph.Output();
    for (size_t i = 0; i < firstX.size(); ++i)
        Require(output.px[i] == firstX[i] && output.py[i] == firstY[i] &&
                output.pz[i] == firstZ[i],
                "Freeze changed its captured output after upstream edit");
    UsdGenCurveBuffer const &upstream = upstreamNode.buffer;
    Require(output.px != upstream.px && output.totalCvs == upstream.totalCvs,
            "Freeze live input did not differ from captured output");
    WriteCurves(stage, SdfPath("/World/Freeze_Updated_Live_Input"), upstream,
                GfVec3f(.36f, .57f, .82f), GfVec3f(-.19f, 0, 0));
    WriteCurves(stage, SdfPath("/World/Freeze_Held_Output"), output,
                GfVec3f(.91f, .58f, .32f), GfVec3f(.19f, 0, 0));
    std::printf("Freeze native two-cook: %u held curves; edited upstream differs\n",
                output.totalCurves);
}

} // namespace

int main(int argc, char **argv) try {
    if (argc != 3) {
        std::fprintf(stderr,
                     "usage: usdGenUtilityFixtures width-blend|part|freeze output.usda\n");
        return 2;
    }
    usdGenRegisterM1Operators();
    UsdStageRefPtr const stage = UsdStage::CreateNew(argv[2]);
    Require(bool(stage), "cannot create output stage");
    UsdGeomXform::Define(stage, SdfPath("/World"));
    stage->SetDefaultPrim(stage->GetPrimAtPath(SdfPath("/World")));
    std::string const mode(argv[1]);
    if (mode == "width-blend") ExportWidthBlend(stage);
    else if (mode == "part") ExportPart(stage);
    else if (mode == "freeze") ExportFreeze(stage);
    else throw std::runtime_error("unknown fixture mode");
    Require(stage->GetRootLayer()->Save(), "cannot save fixture stage");
    return 0;
} catch (std::exception const &e) {
    std::fprintf(stderr, "usdGenUtilityFixtures: %s\n", e.what());
    return 1;
}
