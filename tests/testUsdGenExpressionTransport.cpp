#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGenUsd/walk/usdGenGraphDescBuilderStage.h"

#include "pxr/pxr.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {
int Fail(char const *s) { std::fprintf(stderr, "FAIL: %s\n", s); return 1; }
class Notice final : public HdSceneIndexObserver {
public:
 explicit Notice(SdfPath path = SdfPath("/Character/Groom/hair")) : prefix(std::move(path)) {}
 SdfPath prefix;
 bool seen=false;
 void Reset() { seen=false; }
 void PrimsAdded(HdSceneIndexBase const&, AddedPrimEntries const&e) override { Hit(e); }
 void PrimsRemoved(HdSceneIndexBase const&, RemovedPrimEntries const&e) override { Hit(e); }
 void PrimsDirtied(HdSceneIndexBase const&, DirtiedPrimEntries const&e) override { Hit(e); }
 void PrimsRenamed(HdSceneIndexBase const&, RenamedPrimEntries const&) override {}
 template<class E> void Hit(E const&e) { for(auto const&x:e) seen|=x.primPath.HasPrefix(prefix); }
};

bool Same(usdGen::UsdGenGraphDesc const &a, usdGen::UsdGenGraphDesc const &b)
{
    if (a.executionBackend != b.executionBackend ||
        a.expressions.size() != b.expressions.size() || a.nodes.size() != b.nodes.size()) return false;
    for (size_t i=0; i<a.expressions.size(); ++i) {
        auto const &x=a.expressions[i], &y=b.expressions[i];
        if (x.path!=y.path || x.source!=y.source || x.outputs.size()!=y.outputs.size()) return false;
        for (size_t j=0;j<x.outputs.size();++j)
            if (x.outputs[j].name!=y.outputs[j].name || x.outputs[j].nativeType!=y.outputs[j].nativeType ||
                x.outputs[j].shape.scalar!=y.outputs[j].shape.scalar ||
                x.outputs[j].shape.elementCount!=y.outputs[j].shape.elementCount ||
                x.outputs[j].shape.components!=y.outputs[j].shape.components ||
                x.outputs[j].shape.rows!=y.outputs[j].shape.rows ||
                x.outputs[j].shape.columns!=y.outputs[j].shape.columns ||
                x.outputs[j].shape.isArray!=y.outputs[j].shape.isArray) return false;
    }
    for (size_t i=0;i<a.nodes.size();++i) {
        auto const &x=a.nodes[i].expressionBindings, &y=b.nodes[i].expressionBindings;
        if (x.size()!=y.size()) return false;
        for (size_t j=0;j<x.size();++j)
            if (x[j].expression!=y[j].expression || x[j].output!=y[j].output ||
                x[j].destination!=y[j].destination || x[j].nativeType!=y[j].nativeType ||
                x[j].destinationShape.scalar!=y[j].destinationShape.scalar ||
                x[j].destinationShape.elementCount!=y[j].destinationShape.elementCount ||
                x[j].destinationShape.components!=y[j].destinationShape.components ||
                x[j].destinationShape.rows!=y[j].destinationShape.rows ||
                x[j].destinationShape.columns!=y[j].destinationShape.columns ||
                x[j].destinationShape.isArray!=y[j].destinationShape.isArray ||
                x[j].domain!=y[j].domain || x[j].literal!=y[j].literal) return false;
    }
    return true;
}

bool CheckShapeParity()
{
    SdfLayerRefPtr const layer = SdfLayer::CreateAnonymous("expression-shapes.usda");
    if (!layer || !layer->ImportFromString(R"usda(
#usda 1.0
def UsdGenDescription "D"
{
    def Scope "Ops"
    {
        def UsdGenWidth "op"
        {
            custom bool boolValue
            bool boolValue.connect = </D/Expressions/bool.outputs:result>
            custom int64 int64Value
            int64 int64Value.connect = </D/Expressions/int64.outputs:result>
            custom half halfValue
            half halfValue.connect = </D/Expressions/half.outputs:result>
            custom vector3f vectorValue (
                customData = { dictionary usdGen = { string evaluation = "primitive" } }
            )
            vector3f vectorValue.connect = </D/Expressions/vector.outputs:result>
            custom float[] scalarArrayValue = [1, 2, 3]
            float[] scalarArrayValue.connect = </D/Expressions/scalarArray.outputs:result>
            custom vector3f[] vectorArrayValue = [(1, 2, 3), (4, 5, 6)]
            vector3f[] vectorArrayValue.connect = </D/Expressions/vectorArray.outputs:result>
        }
    }
    def Scope "Expressions"
    {
        def UsdGenExpression "bool" {
            custom string usdGen:expr:source = "0"
            custom bool outputs:result = false
        }
        def UsdGenExpression "int64" {
            custom string usdGen:expr:source = "0"
            custom int64 outputs:result = 7
        }
        def UsdGenExpression "half" {
            custom string usdGen:expr:source = "0"
            custom half outputs:result = 0.5
        }
        def UsdGenExpression "vector" {
            custom string usdGen:expr:source = "0"
            custom vector3f outputs:result
        }
        def UsdGenExpression "point" {
            custom string usdGen:expr:source = "0"
            custom point3f outputs:result
        }
        def UsdGenExpression "quat" {
            custom string usdGen:expr:source = "0"
            custom quatf outputs:result
        }
        def UsdGenExpression "halfQuat" {
            custom string usdGen:expr:source = "0"
            custom quath outputs:result
        }
        def UsdGenExpression "matrix" {
            custom string usdGen:expr:source = "0"
            custom matrix4d outputs:result
        }
        def UsdGenExpression "scalarArray" {
            custom string usdGen:expr:source = "0"
            custom float[] outputs:result = [1, 2, 3]
        }
        def UsdGenExpression "vectorArray" {
            custom string usdGen:expr:source = "0"
            custom vector3f[] outputs:result = [(1, 2, 3), (4, 5, 6)]
        }
        def UsdGenExpression "emptyArray" {
            custom string usdGen:expr:source = "0"
            custom float[] outputs:result = []
        }
        def UsdGenExpression "undeclaredArray" {
            custom string usdGen:expr:source = "0"
            custom float[] outputs:result
        }
    }
}
)usda")) return false;
    UsdStageRefPtr const stage = UsdStage::Open(layer);
    if (!stage) return false;
    UsdImagingCreateSceneIndicesInfo info; info.stage = stage;
    UsdImagingSceneIndices const indices = UsdImagingCreateSceneIndices(info);
    auto const stageDesc = usdGenImaging::BuildGraphDescFromStage(stage, SdfPath("/D"));
    auto const hydraDesc = usdGenImaging::BuildGraphDescFromHydra(
        *indices.finalSceneIndex, SdfPath("/D"));
    if (!Same(stageDesc, hydraDesc)) return false;

    auto output = [](usdGen::UsdGenGraphDesc const& desc, char const* name) {
        for (auto const& expression : desc.expressions)
            if (expression.path.GetName() == TfToken(name) && !expression.outputs.empty())
                return expression.outputs.front().shape;
        return usdGen::expr::ValueShape{};
    };
    auto expect = [&](char const* name, usdGen::expr::ScalarType scalar,
                      uint32_t elements, uint32_t components, uint32_t rows,
                      uint32_t columns, bool array) {
        auto const shape = output(stageDesc, name);
        return shape.scalar == scalar && shape.elementCount == elements &&
            shape.components == components && shape.rows == rows &&
            shape.columns == columns && shape.isArray == array;
    };
    if (!expect("bool", usdGen::expr::ScalarType::Bool, 1, 1, 1, 1, false) ||
        !expect("int64", usdGen::expr::ScalarType::Int64, 1, 1, 1, 1, false) ||
        !expect("half", usdGen::expr::ScalarType::Float16, 1, 1, 1, 1, false) ||
        !expect("vector", usdGen::expr::ScalarType::Float32, 1, 3, 1, 1, false) ||
        !expect("point", usdGen::expr::ScalarType::Float32, 1, 3, 1, 1, false) ||
        !expect("quat", usdGen::expr::ScalarType::Float32, 1, 4, 1, 1, false) ||
        !expect("halfQuat", usdGen::expr::ScalarType::Float16, 1, 4, 1, 1, false) ||
        !expect("matrix", usdGen::expr::ScalarType::Float64, 1, 1, 4, 4, false) ||
        !expect("scalarArray", usdGen::expr::ScalarType::Float32, 3, 1, 1, 1, true) ||
        !expect("vectorArray", usdGen::expr::ScalarType::Float32, 2, 3, 1, 1, true) ||
        !expect("emptyArray", usdGen::expr::ScalarType::Float32, 0, 1, 1, 1, true) ||
        !expect("undeclaredArray", usdGen::expr::ScalarType::Invalid, 0, 1, 1, 1, true))
        return false;

    if (stageDesc.nodes.size() != 1 || stageDesc.nodes.front().expressionBindings.size() != 6)
        return false;
    for (auto const& binding : stageDesc.nodes.front().expressionBindings) {
        if (binding.destination == TfToken("vectorValue") &&
            binding.domain != usdGen::expr::Domain::Primitive) return false;
        if (binding.destination == TfToken("scalarArrayValue") &&
            (binding.destinationShape.elementCount != 3 || !binding.destinationShape.isArray)) return false;
        if (binding.destination == TfToken("vectorArrayValue") &&
            (binding.destinationShape.elementCount != 2 ||
             binding.destinationShape.components != 3 || !binding.destinationShape.isArray)) return false;
    }
    return true;
}
}

int main()
{
    if (!CheckShapeParity()) return Fail("typed expression shape parity");
    std::string const file = std::string(USDGEN_TEST_SOURCE_DIR) + "/plan/examples/operator-network.usda";
    UsdStageRefPtr const stage = UsdStage::Open(file);
    if (!stage) return Fail("open operator-network mock");
    SdfPath const descPath("/Character/Groom/hair");
    auto stageDesc = usdGenImaging::BuildGraphDescFromStage(stage, descPath);
    UsdImagingCreateSceneIndicesInfo info; info.stage = stage;
    UsdImagingSceneIndices const indices = UsdImagingCreateSceneIndices(info);
    Notice notice; HdSceneIndexObserverPtr observer = TfCreateWeakPtr(&notice);
    indices.finalSceneIndex->AddObserver(observer);
    auto hydraDesc = usdGenImaging::BuildGraphDescFromHydra(*indices.finalSceneIndex, descPath);
    if (!Same(stageDesc, hydraDesc)) return Fail("Stage/Hydra expression descriptor parity");
    if (stageDesc.executionBackend != usdGenTestUtils::Backend("cuda")) return Fail("cuda backend transport");
    for (auto const &curve : hydraDesc.curveSets) {
        if (curve.path.GetName() == TfToken("curves")) {
            if (curve.restFromCurrentPoints)
                return Fail("valid Default-time C3 rest was marked current-frame");
        }
    }

    // The description metadata source is dynamic rather than a retained
    // first-pull value.  Exercise it through the same live scene index.
    double const originalRate = stage->GetTimeCodesPerSecond();
    notice.Reset();
    stage->SetTimeCodesPerSecond(30.0);
    indices.stageSceneIndex->ApplyPendingUpdates();
    if (!notice.seen) return Fail("timeCodesPerSecond edit emitted no notice");
    auto rateStage = usdGenImaging::BuildGraphDescFromStage(stage, descPath);
    auto rateHydra = usdGenImaging::BuildGraphDescFromHydra(
        *indices.finalSceneIndex, descPath);
    if (rateStage.timeCodesPerSecond != 30.0 ||
        rateHydra.timeCodesPerSecond != 30.0)
        return Fail("live timeCodesPerSecond metadata was not transported");
    stage->SetTimeCodesPerSecond(originalRate);
    indices.stageSceneIndex->ApplyPendingUpdates();

    // USD resolves framesPerSecond when timeCodesPerSecond is absent, then
    // falls back to 24 when both are absent.
    SdfLayerRefPtr const rootLayer = stage->GetRootLayer();
    rootLayer->ClearTimeCodesPerSecond();
    rootLayer->SetFramesPerSecond(30.0);
    indices.stageSceneIndex->ApplyPendingUpdates();
    auto fpsStage = usdGenImaging::BuildGraphDescFromStage(stage, descPath);
    auto fpsHydra = usdGenImaging::BuildGraphDescFromHydra(
        *indices.finalSceneIndex, descPath);
    if (fpsStage.timeCodesPerSecond != 30.0 ||
        fpsHydra.timeCodesPerSecond != 30.0)
        return Fail("framesPerSecond fallback was not transported");
    rootLayer->ClearFramesPerSecond();
    indices.stageSceneIndex->ApplyPendingUpdates();
    auto defaultRateStage = usdGenImaging::BuildGraphDescFromStage(stage, descPath);
    auto defaultRateHydra = usdGenImaging::BuildGraphDescFromHydra(
        *indices.finalSceneIndex, descPath);
    if (defaultRateStage.timeCodesPerSecond != 24.0 ||
        defaultRateHydra.timeCodesPerSecond != 24.0)
        return Fail("USD default time rate was not transported");
    rootLayer->SetFramesPerSecond(24.0);
    rootLayer->SetTimeCodesPerSecond(originalRate);
    indices.stageSceneIndex->ApplyPendingUpdates();

    // Invalid authored stage rates are transported as diagnostics in both
    // builders; neither may silently reinstate the 24 fallback.
    rootLayer->SetTimeCodesPerSecond(0.0);
    indices.stageSceneIndex->ApplyPendingUpdates();
    auto zeroRateStage = usdGenImaging::BuildGraphDescFromStage(stage, descPath);
    auto zeroRateHydra = usdGenImaging::BuildGraphDescFromHydra(
        *indices.finalSceneIndex, descPath);
    if (zeroRateStage.validationErrors.empty() ||
        zeroRateHydra.validationErrors.empty())
        return Fail("non-positive time rate was silently accepted");
    rootLayer->SetTimeCodesPerSecond(
        std::numeric_limits<double>::quiet_NaN());
    indices.stageSceneIndex->ApplyPendingUpdates();
    auto nanRateStage = usdGenImaging::BuildGraphDescFromStage(stage, descPath);
    auto nanRateHydra = usdGenImaging::BuildGraphDescFromHydra(
        *indices.finalSceneIndex, descPath);
    if (nanRateStage.validationErrors.empty() ||
        nanRateHydra.validationErrors.empty())
        return Fail("non-finite time rate was silently accepted");
    rootLayer->SetTimeCodesPerSecond(originalRate);
    indices.stageSceneIndex->ApplyPendingUpdates();

    // An authored but unknown backend is not a request for the legacy CPU
    // path.  Check both transports while they share this live scene index.
    UsdAttribute backend = stage->GetAttributeAtPath(
        SdfPath("/Character/Groom.usdGen:execution:backend"));
    if (!backend) return Fail("backend attribute lookup");
    backend.Set(TfToken("not-a-backend"));
    indices.stageSceneIndex->ApplyPendingUpdates();
    auto invalidStage = usdGenImaging::BuildGraphDescFromStage(stage, descPath);
    auto invalidHydra = usdGenImaging::BuildGraphDescFromHydra(
        *indices.finalSceneIndex, descPath);
    if (invalidStage.executionBackend != usdGen::UsdGenBackendId::Invalid ||
        invalidHydra.executionBackend != usdGen::UsdGenBackendId::Invalid)
        return Fail("unknown backend did not fail closed");
    auto namesBackend = [](std::vector<std::string> const &errors) {
        return std::find(errors.begin(), errors.end(),
            "no backend named 'not-a-backend' is registered") != errors.end();
    };
    if (!namesBackend(invalidStage.validationErrors) ||
        !namesBackend(invalidHydra.validationErrors))
        return Fail("unknown backend was not reported by name");

    // Clearing the authored opinion exposes the schema's resolved cuda
    // fallback, which is still a valid backend request.
    backend.Clear();
    indices.stageSceneIndex->ApplyPendingUpdates();
    auto absentStage = usdGenImaging::BuildGraphDescFromStage(stage, descPath);
    auto absentHydra = usdGenImaging::BuildGraphDescFromHydra(
        *indices.finalSceneIndex, descPath);
    if (absentStage.executionBackend != usdGenTestUtils::Backend("cuda") ||
        absentHydra.executionBackend != usdGenTestUtils::Backend("cuda"))
        return Fail("schema backend fallback was not transported");

    // A genuinely undeclared property remains the legacy CPU baseline.  Use
    // an untyped parent Scope so no UsdGenGroom schema fallback can supply a
    // backend value.
    SdfLayerRefPtr const noBackendLayer =
        SdfLayer::CreateAnonymous("expression-transport-no-backend.usda");
    if (!noBackendLayer->ImportFromString(R"usda(
#usda 1.0
def Scope "Groom"
{
    def UsdGenDescription "hair"
    {
        string usdGen:width:default = "bad"
        string usdGen:tileTarget = "bad"
        def Scope "Ops"
        {
            def UsdGenWidth "width"
            {
                # Deliberately malformed dedicated scalars: both builders
                # must preserve each type error instead of using defaults.
                string usdGen:enabled = "bad"
                string usdGen:seed = "bad"
            }
        }
    }
}
)usda")) return Fail("open no-backend fixture");
    UsdStageRefPtr const noBackendStage = UsdStage::Open(noBackendLayer);
    if (!noBackendStage) return Fail("open no-backend stage");
    UsdImagingCreateSceneIndicesInfo noBackendInfo;
    noBackendInfo.stage = noBackendStage;
    UsdImagingSceneIndices const noBackendIndices =
        UsdImagingCreateSceneIndices(noBackendInfo);
    auto noBackendStageDesc = usdGenImaging::BuildGraphDescFromStage(
        noBackendStage, SdfPath("/Groom/hair"));
    auto noBackendHydraDesc = usdGenImaging::BuildGraphDescFromHydra(
        *noBackendIndices.finalSceneIndex, SdfPath("/Groom/hair"));
    if (noBackendStageDesc.executionBackend !=
            usdGen::kUsdGenBackendCpu ||
        noBackendHydraDesc.executionBackend !=
            usdGen::kUsdGenBackendCpu)
        return Fail("undeclared backend did not retain CPU baseline");
    if (noBackendStageDesc.validationErrors.size() < 4 ||
        noBackendHydraDesc.validationErrors.size() < 4) {
        for (auto const& e : noBackendStageDesc.validationErrors) std::fprintf(stderr, "Stage: %s\n", e.c_str());
        for (auto const& e : noBackendHydraDesc.validationErrors) std::fprintf(stderr, "Hydra: %s\n", e.c_str());
        std::fprintf(stderr, "Hydra nodes: %zu\n", noBackendHydraDesc.nodes.size());
        return Fail("malformed dedicated scalar was silently defaulted");
    }
    Notice recoveryNotice(SdfPath("/Groom/hair"));
    HdSceneIndexObserverPtr recoveryObserver = TfCreateWeakPtr(&recoveryNotice);
    noBackendIndices.finalSceneIndex->AddObserver(recoveryObserver);
    auto widthPrim = noBackendStage->GetPrimAtPath(SdfPath("/Groom/hair/Ops/width"));
    auto descriptionPrim = noBackendStage->GetPrimAtPath(SdfPath("/Groom/hair"));
    if (!widthPrim.GetAttribute(TfToken("usdGen:enabled")).Set(true) ||
        !widthPrim.GetAttribute(TfToken("usdGen:seed")).Set(0) ||
        !descriptionPrim.GetAttribute(TfToken("usdGen:width:default")).Set(.01f) ||
        !descriptionPrim.GetAttribute(TfToken("usdGen:tileTarget")).Set(64))
        return Fail("repair malformed dedicated values");
    noBackendIndices.stageSceneIndex->ApplyPendingUpdates();
    if (!recoveryNotice.seen) return Fail("dedicated value repair emitted no notice");
    auto repaired = usdGenImaging::BuildGraphDescFromHydra(
        *noBackendIndices.finalSceneIndex, SdfPath("/Groom/hair"));
    if (!repaired.validationErrors.empty()) return Fail("stale dedicated type diagnostics survived repair");
    noBackendIndices.finalSceneIndex->RemoveObserver(recoveryObserver);

    // Restore the fixture's authored request for the remaining transport
    // assertions below.
    backend.Set(TfToken("cuda"));
    indices.stageSceneIndex->ApplyPendingUpdates();
    if (stageDesc.expressions.size()!=6) return Fail("six expression programs");
    size_t bindings=0; for (auto const &n:stageDesc.nodes) bindings += n.expressionBindings.size();
    if (bindings!=7) return Fail("seven expression connections");

    // Each edit uses the same live scene index and its own observer window:
    // a notice from a different edit must not mask a missing invalidation.
    UsdAttribute source = stage->GetAttributeAtPath(SdfPath("/Character/Groom/hair/Expressions/strandWidth.usdGen:expr:source"));
    source.Set(std::string("$value * 2"));
    indices.stageSceneIndex->ApplyPendingUpdates();
    if (!notice.seen) return Fail("source edit emitted no scene-index notice");
    auto sourceLive = usdGenImaging::BuildGraphDescFromHydra(*indices.finalSceneIndex, descPath);
    bool changed=false; for (auto const &e:sourceLive.expressions) if (e.path.GetName()==TfToken("strandWidth")) changed = e.source == "$value * 2";
    if (!changed) return Fail("live source edit was not transported");

    notice.Reset();
    UsdAttribute width = stage->GetAttributeAtPath(SdfPath("/Character/Groom/hair/Ops/width.usdGen:width"));
    width.ClearConnections();
    indices.stageSceneIndex->ApplyPendingUpdates();
    if (!notice.seen) return Fail("disconnect emitted no scene-index notice");
    auto disconnectLive = usdGenImaging::BuildGraphDescFromHydra(*indices.finalSceneIndex, descPath);
    size_t disconnectBindings=0; for (auto const &n:disconnectLive.nodes) disconnectBindings += n.expressionBindings.size();
    if (disconnectBindings != 6) return Fail("live disconnect was not transported");

    notice.Reset();
    UsdAttribute magnitude = stage->GetAttributeAtPath(
        SdfPath("/Character/Groom/hair/Ops/style/frizz.usdGen:noise:magnitude"));
    magnitude.SetCustomDataByKey(TfToken("usdGen:evaluation"), VtValue("groom"));
    indices.stageSceneIndex->ApplyPendingUpdates();
    if (!notice.seen) return Fail("metadata edit emitted no scene-index notice");
    auto live = usdGenImaging::BuildGraphDescFromHydra(*indices.finalSceneIndex, descPath);
    bool point=false; for (auto const &n:live.nodes) for (auto const &b:n.expressionBindings)
        if (b.destination == TfToken("usdGen:noise:magnitude")) point |= b.domain == usdGen::expr::Domain::Groom;
    if (!point) return Fail("live evaluation metadata edit was not transported");
    indices.finalSceneIndex->RemoveObserver(observer);
    return 0;
}
