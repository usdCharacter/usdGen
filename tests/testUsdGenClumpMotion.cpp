// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/clumpMotion.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include "utils/pxrInterop.h"

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {
int failures = 0;
void Check(bool value, char const *message) {
    if (!value) { ++failures; std::printf("FAIL: %s\n", message); }
}

UsdGenPlane IntPlane(char const *name, uint8_t arity, VtIntArray values) {
    UsdGenPlane p; p.name = TfToken(name); p.type = TfToken("int");
    p.interpolation = TfToken("uniform"); p.arity = arity;
    p.i = std::move(values); return p;
}
UsdGenPlane FloatPlane(char const *name, uint8_t arity,
                       TfToken interp, VtFloatArray values) {
    UsdGenPlane p; p.name = TfToken(name); p.type = TfToken("float");
    p.interpolation = interp; p.arity = arity;
    p.f = std::move(values); return p;
}
void AddLevel(UsdGenCurveBuffer *b, int level, VtIntArray membership,
              uint64_t center, float weight) {
    std::string const suffix = std::to_string(level);
    uint32_t words[2] = {uint32_t(center), uint32_t(center >> 32)};
    int32_t packed[2]; std::memcpy(packed, words, sizeof(packed));
    b->extraCurve.push_back(IntPlane(("clumpId_" + suffix).c_str(), 1,
                                     std::move(membership)));
    b->extraCurve.push_back(FloatPlane(("clumpCenter_" + suffix).c_str(), 3,
        TfToken("uniform"), {1,2,3, 1,2,3, 1,2,3}));
    b->extraCurve.push_back(IntPlane(("clumpCenterId_" + suffix).c_str(), 2,
        {packed[0],packed[1], packed[0],packed[1], packed[0],packed[1]}));
    b->extraCv.push_back(FloatPlane(("clumpWeight_" + suffix).c_str(), 1,
        TfToken("vertex"), VtFloatArray(b->totalCvs, weight)));
}

UsdGenGraphDesc Chain(int firstLevels, int secondPinned = -1) {
    UsdGenGraphDesc d; d.description = SdfPath("/clump");
    UsdGenNodeDesc source; source.path = SdfPath("/clump/source");
    source.type = TfToken("UsdGenScatter"); source.enabled = true;
    source.surfaces.push_back(SdfPath("/clump/surface"));
    d.nodes.push_back(source);
    UsdGenNodeDesc a; a.path = SdfPath("/clump/a");
    a.type = TfToken("UsdGenClump"); a.enabled = true;
    a.inputs = {source.path};
    a.params.push_back({TfToken("clump:levels"), VtValue(firstLevels), false});
    d.nodes.push_back(a);
    UsdGenNodeDesc b; b.path = SdfPath("/clump/b");
    b.type = TfToken("UsdGenClump"); b.enabled = true;
    b.inputs = {a.path};
    if (secondPinned >= 0)
        b.params.push_back({TfToken("clump:level"), VtValue(secondPinned), false});
    d.nodes.push_back(b);
    d.terminal = b.path;
    UsdGenSurfaceDesc surface; surface.path = SdfPath("/clump/surface");
    surface.id = 0; d.surfaces.push_back(surface);
    return d;
}
}

int main() {
    UsdGenCurveBuffer b; b.totalCurves = 3; b.totalCvs = 6;
    AddLevel(&b, 10, {2,2,-1}, 0xfedcba9876543210ULL, 0.75f);
    AddLevel(&b, 2, {1,1,-1}, 0xfedcba9876543210ULL, 0.5f);
    auto byName = [](UsdGenPlane const &a, UsdGenPlane const &c) { return a.name < c.name; };
    std::sort(b.extraCurve.begin(), b.extraCurve.end(), byName);
    std::sort(b.extraCv.begin(), b.extraCv.end(), byName);
    UsdGenClumpMotion motion; std::string error;
    Check(UsdGenBuildClumpMotion(b, &motion, &error), "native level capture");
    Check(motion.levels.size() == 2 && motion.levels[0].level == 2 &&
          motion.levels[1].level == 10, "numeric level order");
    Check(motion.levels[0].groups.size() == 1 &&
          motion.levels[0].groups[0].centerId == 0xfedcba9876543210ULL,
          "exact full-width center identity");
    Check(motion.levels[0].groupForCurve[0] == 0 &&
          motion.levels[0].groupForCurve[1] == 0 &&
          motion.levels[0].groupForCurve[2] == UINT32_MAX,
          "group refs and absent member");
    UsdGenCurveBuffer valueOnly = b;
    for (UsdGenPlane &plane : valueOnly.extraCv)
        if (plane.name == TfToken("clumpWeight_2")) plane.f[0] = 0.25f;
    UsdGenClumpMotion reused;
    Check(UsdGenBuildClumpMotion(valueOnly, &reused, &error, &motion) &&
          &reused.levels[0].groupForCurve[0] ==
              &motion.levels[0].groupForCurve[0] &&
          reused.levels[0].weight[0] == 0.25f,
          "value-only recapture shares immutable dense binding");
    UsdGenCurveBuffer zeroWeights = valueOnly;
    for (UsdGenPlane &plane : zeroWeights.extraCv)
        if (plane.name == TfToken("clumpWeight_2"))
            plane.f = VtFloatArray(zeroWeights.totalCvs, 0.0f);
    UsdGenClumpMotion zeroMotion, restoredMotion;
    Check(UsdGenBuildClumpMotion(zeroWeights, &zeroMotion, &error, &reused) &&
          UsdGenBuildClumpMotion(valueOnly, &restoredMotion, &error, &zeroMotion) &&
          &zeroMotion.levels[0].groupForCurve[0] ==
              &restoredMotion.levels[0].groupForCurve[0] &&
          restoredMotion.levels[0].weight[0] == 0.25f,
          "zero-to-positive cohesion retains immutable group binding");
    UsdGenCurveBuffer changedMembership = valueOnly;
    for (UsdGenPlane &plane : changedMembership.extraCurve)
        if (plane.name == TfToken("clumpId_2")) {
            plane.i[0] = 9; plane.i[1] = 9;
        }
    UsdGenClumpMotion rebuilt;
    Check(UsdGenBuildClumpMotion(changedMembership, &rebuilt, &error, &reused) &&
          &rebuilt.levels[0].groupForCurve[0] !=
              &reused.levels[0].groupForCurve[0] &&
          rebuilt.levels[0].groups.size() == 1,
          "membership mutation invalidates dense binding");
    UsdGenCurveBuffer changedAnchor = valueOnly;
    for (UsdGenPlane &plane : changedAnchor.extraCurve)
        if (plane.name == TfToken("clumpCenter_2")) {
            plane.f[0] = 4.0f; plane.f[3] = 4.0f;
        }
    UsdGenClumpMotion anchorMotion;
    Check(UsdGenBuildClumpMotion(changedAnchor, &anchorMotion, &error, &reused) &&
          &anchorMotion.levels[0].groupForCurve[0] !=
              &reused.levels[0].groupForCurve[0] &&
          anchorMotion.levels[0].groups[0].restAnchor[0] == 4.0f,
          "anchor mutation invalidates dense binding");
    UsdGenCurveBuffer changedCenterId = valueOnly;
    for (UsdGenPlane &plane : changedCenterId.extraCurve)
        if (plane.name == TfToken("clumpCenterId_2")) {
            plane.i[0] = 17; plane.i[2] = 17;
        }
    UsdGenClumpMotion centerMotion;
    Check(UsdGenBuildClumpMotion(changedCenterId, &centerMotion, &error, &reused) &&
          &centerMotion.levels[0].groupForCurve[0] !=
              &reused.levels[0].groupForCurve[0] &&
          centerMotion.levels[0].groups[0].centerId !=
              reused.levels[0].groups[0].centerId,
          "center identity mutation invalidates dense binding");
    UsdGenCurveBuffer changedLayout = valueOnly;
    for (UsdGenPlane &plane : changedLayout.extraCurve)
        if (plane.name == TfToken("clumpCenterId_2")) plane.arity = 3;
    UsdGenClumpMotion rejectedLayout;
    Check(!UsdGenBuildClumpMotion(changedLayout, &rejectedLayout, &error, &reused),
          "layout mutation fails closed despite matching CoW payload");
    UsdGenClumpMotion retained;
    {
        UsdGenCurveBuffer ephemeral = b;
        Check(UsdGenBuildClumpMotion(ephemeral, &retained, &error),
              "temporary source captures native binding");
    }
    Check(retained.levels[0].sourceMembership.size() == 3 &&
          retained.levels[0].groups[0].centerId == 0xfedcba9876543210ULL,
          "binding retains source CoW lifetime after source destruction");

    UsdGenCurveBuffer resampled; resampled.totalCurves = 3; resampled.totalCvs = 9;
    Check(UsdGenResampleExtraPlanes(b, &resampled, &error), "topology transform");
    UsdGenClumpMotion resampledMotion;
    Check(UsdGenBuildClumpMotion(resampled, &resampledMotion, &error) &&
          resampledMotion.levels[0].groups[0].centerId ==
              motion.levels[0].groups[0].centerId &&
          resampledMotion.levels[0].weight.size() == 9,
          "motion planes survive resampling");
    UsdGenCurveBuffer compacted; compacted.totalCurves = 2; compacted.totalCvs = 4;
    Check(UsdGenCompactExtraPlanes(b, {0,2}, &compacted, &error), "stable compaction");
    UsdGenClumpMotion compactMotion;
    Check(UsdGenBuildClumpMotion(compacted, &compactMotion, &error) &&
          compactMotion.levels[0].groups[0].centerId ==
              motion.levels[0].groups[0].centerId,
          "stable center survives compaction");

    UsdGenCurveBuffer legacy; legacy.totalCurves = 3; legacy.totalCvs = 6;
    legacy.extraCurve.push_back(IntPlane("clumpId_4", 1, {1,1,2}));
    UsdGenClumpMotion legacyMotion;
    Check(UsdGenBuildClumpMotion(legacy, &legacyMotion, &error) &&
          legacyMotion.levels.size() == 1 && legacyMotion.levels[0].weight.empty() &&
          legacyMotion.levels[0].groups.empty(), "legacy IDs do not invent motion");
    UsdGenCurveBuffer orphan = legacy;
    orphan.extraCurve.clear();
    orphan.extraCv.push_back(FloatPlane("clumpWeight_4", 1,
        TfToken("vertex"), VtFloatArray(6, 1.0f)));
    Check(!UsdGenBuildClumpMotion(orphan, &legacyMotion, &error),
          "orphan native plane fails closed");
    UsdGenCurveBuffer wrongDomain = legacy;
    wrongDomain.extraCurve.clear();
    auto misplacedId = IntPlane("clumpId_4", 1, {1,1,2});
    misplacedId.interpolation = TfToken("vertex");
    wrongDomain.extraCv.push_back(misplacedId);
    Check(!UsdGenBuildClumpMotion(wrongDomain, &legacyMotion, &error),
          "vertex-domain clumpId fails closed");
    wrongDomain = legacy;
    wrongDomain.extraCurve.push_back(FloatPlane("clumpWeight_4", 1,
        TfToken("uniform"), VtFloatArray(3, 1.0f)));
    Check(!UsdGenBuildClumpMotion(wrongDomain, &legacyMotion, &error),
          "uniform-domain clumpWeight fails closed");
    UsdGenCurveBuffer noncanonical = legacy;
    noncanonical.extraCurve[0].name = TfToken("clumpId_04");
    Check(!UsdGenBuildClumpMotion(noncanonical, &legacyMotion, &error),
          "noncanonical level name fails closed");
    UsdGenCurveBuffer split = b;
    for (UsdGenPlane &plane : split.extraCurve) {
        if (plane.name != TfToken("clumpCenterId_2")) continue;
        plane.i[2] = 77;
    }
    Check(!UsdGenBuildClumpMotion(split, &legacyMotion, &error),
          "one membership cannot split across center IDs");

    UsdGenCompiler compiler; UsdGenGraph graph;
    auto valid = usdGenUsd::Compile(compiler, Chain(2), &graph);
    Check(valid.ok, "multi-level auto Clumps compile without overlap");
    if (valid.ok) {
        Check(graph.Node(1).outputPrimvars.front() ==
              TfToken("clumpId_0") && graph.Node(2).outputPrimvars.front() ==
              TfToken("clumpId_2"), "second auto Clump follows all prior levels");
    }
    UsdGenGraph invalidGraph;
    auto invalid = usdGenUsd::Compile(compiler, Chain(2, 1), &invalidGraph);
    Check(!invalid.ok && !invalid.errors.empty(), "pinned overlapping levels fail compile");
    UsdGenGraphDesc imported = Chain(1);
    imported.nodes[0].type = TfToken("UsdGenCurveSource");
    imported.nodes[0].surfaces.clear();
    imported.nodes[0].curves = {SdfPath("/clump/imported")};
    UsdGenCurveSetDesc sourceCurves;
    sourceCurves.path = imported.nodes[0].curves.front();
    sourceCurves.role = UsdGenLane::Chunked;
    sourceCurves.curveRole = TfToken("hair");
    sourceCurves.curveVertexCounts = {2};
    sourceCurves.points = {{0,0,0}, {0,1,0}};
    sourceCurves.rest = sourceCurves.points;
    sourceCurves.skinPrim = {0};
    sourceCurves.skinPrimUv = {{0,0}};
    sourceCurves.rootFrame = {GfMatrix4d(1.0)};
    imported.nodes[0].params = {
        {TfToken("rebind"), VtValue(TfToken("never")), false},
        {TfToken("useRest"), VtValue(true), false}};
    auto authored = [](char const *name, UsdGenAuthoredPlaneType type,
                       UsdGenAuthoredPlaneDomain domain, uint8_t arity) {
        UsdGenAuthoredPlaneDesc p;
        p.name = TfToken(name); p.type = type; p.domain = domain; p.arity = arity;
        return p;
    };
    auto id = authored("clumpId_0", UsdGenAuthoredPlaneType::Int32,
                       UsdGenAuthoredPlaneDomain::Primitive, 1);
    id.intValues = {7};
    auto center = authored("clumpCenter_0", UsdGenAuthoredPlaneType::Float32,
                           UsdGenAuthoredPlaneDomain::Primitive, 3);
    center.floatValues = {0,0,0};
    auto centerId = authored("clumpCenterId_0", UsdGenAuthoredPlaneType::Int32,
                             UsdGenAuthoredPlaneDomain::Primitive, 2);
    centerId.intValues = {1,0};
    auto weight = authored("clumpWeight_0", UsdGenAuthoredPlaneType::Float32,
                           UsdGenAuthoredPlaneDomain::Point, 1);
    weight.floatValues = {1,1};
    sourceCurves.authoredPlanes = {id, center, centerId, weight};
    imported.curveSets.push_back(std::move(sourceCurves));
    UsdGenGraph importedGraph;
    auto importedResult = usdGenUsd::Compile(compiler, imported, &importedGraph);
    Check(importedResult.ok, "imported quartet and auto Clumps compile");
    if (importedResult.ok)
        Check(importedGraph.Node(1).outputPrimvars.front() == TfToken("clumpId_1") &&
              importedGraph.Node(2).outputPrimvars.front() == TfToken("clumpId_2"),
              "auto Clumps preserve imported level zero");
    if (importedResult.ok) {
        UsdGenScheduler scheduler(2);
        UsdGenEvalContext context;
        context.desc = &importedGraph.Spec();
        auto run = scheduler.Run(importedGraph, context, 1);
        for (auto const &diagnostic : run.diagnostics.errors)
            std::printf("  imported cook: %s\n", diagnostic.c_str());
        Check(!run.diagnostics.HasErrors(), "imported quartet survives Clump cook");
        if (!run.diagnostics.HasErrors()) {
            UsdGenClumpMotion importedMotion;
            Check(UsdGenBuildClumpMotion(OutputView(importedGraph),
                                         &importedMotion, &error) &&
                  importedMotion.levels.size() == 3 &&
                  importedMotion.levels[0].level == 0 &&
                  importedMotion.levels[0].groups.size() == 1 &&
                  importedMotion.levels[0].groups[0].centerId == 1,
                  "imported native group remains intact after auto Clumps");
        }
    }
    imported.nodes[1].params.push_back({TfToken("clump:level"), VtValue(0), false});
    UsdGenGraph importedCollision;
    auto collision = usdGenUsd::Compile(compiler, imported, &importedCollision);
    Check(!collision.ok && !collision.errors.empty(),
          "pinned Clump cannot replace an imported level");
    int const top = std::numeric_limits<int>::max();
    UsdGenGraphDesc boundary = Chain(1);
    boundary.nodes[1].params.push_back(
        {TfToken("clump:level"), VtValue(top - 5), false});
    boundary.nodes[2].params.push_back(
        {TfToken("clump:levels"), VtValue(4), false});
    UsdGenGraph boundaryGraph;
    auto boundaryValid = usdGenUsd::Compile(compiler, boundary, &boundaryGraph);
    Check(boundaryValid.ok && boundaryGraph.Node(2).outputPrimvars.front() ==
          TfToken("clumpId_" + std::to_string(top - 4)),
          "highest representable auto Clump range compiles");
    UsdGenCurveSetDesc reserved;
    reserved.path = SdfPath("/clump/highImported");
    auto highId = authored(("clumpId_" + std::to_string(top - 4)).c_str(),
                           UsdGenAuthoredPlaneType::Int32,
                           UsdGenAuthoredPlaneDomain::Primitive, 1);
    reserved.authoredPlanes.push_back(std::move(highId));
    boundary.curveSets.push_back(std::move(reserved));
    UsdGenGraph overflowGraph;
    auto overflow = usdGenUsd::Compile(compiler, boundary, &overflowGraph);
    Check(!overflow.ok && !overflow.errors.empty(),
          "auto Clump rejects occupied range at integer boundary");
    return failures ? 1 : 0;
}
