// An attribute -> prim connection is valid USD, so a connectable usdGen
// parameter may name either the expression PRIM or one of its outputs. Both
// spellings must reach the descriptor as the SAME binding, and both graph-desc
// builders (stage reference path and Hydra production path) must agree.
//
// Covers usdGen:mask specifically, because that is the operator envelope the
// UsdGenMaskAPI block was replaced with, but the resolution is generic.
#include "utils/pxrInterop.h"
#include "usdGen/compile/compiler.h"
#include "api/testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGen/compile/graph.h"
#include "usdGen/api/ext/op.h"
#include "usdGen/api/ext/opRegistry.h"
#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "usdGenUsd/walk/usdGenGraphDescBuilderStage.h"

#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"
#include "pxr/usd/sdf/types.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <cstdio>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_failures = 0;

void Check(bool ok, std::string const &what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

UsdStageRefPtr BuildStage()
{
    UsdStageRefPtr stage = UsdStage::CreateInMemory("exprPrimConnection");
    UsdGeomMesh mesh = UsdGeomMesh::Define(stage, SdfPath("/Plane"));
    mesh.CreatePointsAttr(VtValue(VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(1, 0, 1), GfVec3f(0, 0, 1)}));
    mesh.CreateFaceVertexCountsAttr(VtValue(VtIntArray{4}));
    mesh.CreateFaceVertexIndicesAttr(VtValue(VtIntArray{0, 1, 2, 3}));

    stage->DefinePrim(SdfPath("/Groom"), TfToken("Xform"));
    UsdPrim description =
        stage->DefinePrim(SdfPath("/Groom/Fur"), TfToken("UsdGenDescription"));
    description.CreateRelationship(TfToken("usdGen:surface"))
        .SetTargets({SdfPath("/Plane")});
    stage->DefinePrim(SdfPath("/Groom/Fur/Ops"), TfToken("Scope"));

    // Namespace order is reversed at execution: scatter, then grow, then the
    // two stylers whose usdGen:mask is connected two different ways.
    UsdPrim byOutput =
        stage->DefinePrim(SdfPath("/Groom/Fur/Ops/widthByOutput"), TfToken("UsdGenWidth"));
    UsdPrim byPrim =
        stage->DefinePrim(SdfPath("/Groom/Fur/Ops/widthByPrim"), TfToken("UsdGenWidth"));
    UsdPrim grow =
        stage->DefinePrim(SdfPath("/Groom/Fur/Ops/grow"), TfToken("UsdGenGrow"));
    grow.CreateAttribute(TfToken("usdGen:segments"), SdfValueTypeNames->Int).Set(4);
    UsdPrim scatter =
        stage->DefinePrim(SdfPath("/Groom/Fur/Ops/scatter"), TfToken("UsdGenScatter"));
    scatter.CreateAttribute(TfToken("usdGen:density"), SdfValueTypeNames->Float).Set(4.0f);

    stage->DefinePrim(SdfPath("/Groom/Fur/Expressions"), TfToken("Scope"));
    UsdPrim expression = stage->DefinePrim(
        SdfPath("/Groom/Fur/Expressions/styleMask"), TfToken("UsdGenExpression"));
    expression.CreateAttribute(TfToken("usdGen:expr:source"),
                               SdfValueTypeNames->String, true)
        .Set(std::string("$value * $u"));
    expression.CreateAttribute(TfToken("outputs:result"),
                               SdfValueTypeNames->Float, true);

    auto connectMask = [&](UsdPrim const &prim, SdfPath const &target) {
        UsdAttribute mask = prim.CreateAttribute(
            TfToken("usdGen:mask"), SdfValueTypeNames->Float);
        mask.Set(1.0f);
        mask.SetConnections(SdfPathVector{target});
    };
    connectMask(byOutput,
                SdfPath("/Groom/Fur/Expressions/styleMask.outputs:result"));
    connectMask(byPrim, SdfPath("/Groom/Fur/Expressions/styleMask"));
    return stage;
}

UsdGenExpressionBinding const *MaskBinding(UsdGenGraphDesc const &desc,
                                           char const *nodePath)
{
    for (UsdGenNodeDesc const &node : desc.nodes) {
        if (node.path != SdfPath(nodePath)) continue;
        for (UsdGenExpressionBinding const &binding : node.expressionBindings) {
            std::string destination = binding.destination.GetString();
            if (destination == "usdGen:mask" || destination == "mask")
                return &binding;
        }
    }
    return nullptr;
}

void CheckBinding(UsdGenExpressionBinding const *binding, std::string const &what)
{
    Check(binding != nullptr, what + ": binding exists");
    if (!binding) return;
    Check(binding->expression == SdfPath("/Groom/Fur/Expressions/styleMask"),
          what + ": binding names the expression prim");
    Check(binding->output == TfToken("result"),
          what + ": prim-path and outputs:result resolve to output 'result'");
    Check(binding->domain == expr::Domain::Primitive,
          what + ": usdGen:mask defaults to primitive evaluation");
    Check(binding->nativeType == TfToken("float"),
          what + ": destination keeps its authored float type");
}

}  // namespace

int main()
{
    UsdStageRefPtr stage = BuildStage();
    Check(static_cast<bool>(stage), "fixture stage opens");
    if (!stage) return 1;

    usdGenImaging::UsdGenGraphDescBuildOptions options;
    UsdGenGraphDesc stageDesc = usdGenImaging::BuildGraphDescFromStage(
        stage, SdfPath("/Groom/Fur"), options);
    Check(stageDesc.expressions.size() == 1 &&
              stageDesc.expressions[0].outputs.size() == 1,
          "stage builder pools the one expression and its single output");
    CheckBinding(MaskBinding(stageDesc, "/Groom/Fur/Ops/widthByOutput"),
                 "stage lane, outputs:result connection");
    CheckBinding(MaskBinding(stageDesc, "/Groom/Fur/Ops/widthByPrim"),
                 "stage lane, prim-path connection");

    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    auto indices = UsdImagingCreateSceneIndices(info);
    indices.stageSceneIndex->SetTime(UsdTimeCode::Default());
    UsdGenGraphDesc hydraDesc = usdGenImaging::BuildGraphDescFromHydra(
        *indices.finalSceneIndex, SdfPath("/Groom/Fur"));
    CheckBinding(MaskBinding(hydraDesc, "/Groom/Fur/Ops/widthByOutput"),
                 "hydra lane, outputs:result connection");
    UsdGenExpressionBinding const *hydraPrim =
        MaskBinding(hydraDesc, "/Groom/Fur/Ops/widthByPrim");
    CheckBinding(hydraPrim, "hydra lane, prim-path connection");

    UsdGenExpressionBinding const *stageOutput =
        MaskBinding(stageDesc, "/Groom/Fur/Ops/widthByOutput");
    UsdGenExpressionBinding const *stagePrim =
        MaskBinding(stageDesc, "/Groom/Fur/Ops/widthByPrim");
    if (stageOutput && stagePrim) {
        Check(stageOutput->output == stagePrim->output &&
                  stageOutput->expression == stagePrim->expression &&
                  stageOutput->domain == stagePrim->domain &&
                  stageOutput->nativeType == stagePrim->nativeType,
              "both connection spellings produce the same binding");
    }
    if (stagePrim && hydraPrim) {
        Check(stagePrim->output == hydraPrim->output &&
                  stagePrim->expression == hydraPrim->expression &&
                  stagePrim->domain == hydraPrim->domain &&
                  stagePrim->nativeType == hydraPrim->nativeType,
              "stage and hydra builders agree on the prim-path binding");
    }

    // An ambiguous prim-path connection fails CLOSED, with a diagnostic that
    // names the prim: two outputs and no outputs:result cannot be resolved.
    {
        UsdGenGraphDesc ambiguous = stageDesc;
        ambiguous.executionBackend = usdGenTestUtils::Backend("cuda");
        Check(ambiguous.expressions.size() == 1, "ambiguity fixture has one expression");
        if (ambiguous.expressions.size() == 1) {
            UsdGenExpressionOutputDesc first;
            first.name = TfToken("a");
            UsdGenExpressionOutputDesc second = first;
            second.name = TfToken("b");
            ambiguous.expressions[0].outputs = {first, second};
        }
        for (UsdGenNodeDesc &node : ambiguous.nodes)
            for (UsdGenExpressionBinding &binding : node.expressionBindings)
                binding.output = TfToken();
        UsdGenCompiler compiler;
        UsdGenGraph graph;
        UsdGenCompileResult const result = usdGenUsd::Compile(compiler, ambiguous, &graph);
        bool named = false;
        for (std::string const &error : result.errors)
            named = named ||
                error.find("/Groom/Fur/Expressions/styleMask") != std::string::npos;
        Check(!result.ok && named,
              "an unresolvable prim-path connection fails closed naming the prim");
    }

    std::printf(g_failures ? "testUsdGenExpressionPrimConnection: FAILED\n"
                           : "testUsdGenExpressionPrimConnection: PASS\n");
    return g_failures ? 1 : 0;
}
