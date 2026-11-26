#include "usdGenCurves/publish/curvePublication.h"
#include "usdGen/session.h"
#include "usdGenUsd/lower.h"
#include "usdGen/cudaExecution.h"
#include "usdGenCuda/bindingReport.h"
#include "usdGen/gpu/generation.h"
#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "usdGenUsd/walk/usdGenGraphDescBuilderStage.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/sdf/types.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"
#include <cmath>
#include <cstdio>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); return 1; } } while (false)

class RestNotice final : public HdSceneIndexObserver {
public:
    bool rest = false;
    void PrimsAdded(HdSceneIndexBase const&, AddedPrimEntries const&) override {}
    void PrimsRemoved(HdSceneIndexBase const&, RemovedPrimEntries const&) override {}
    void PrimsRenamed(HdSceneIndexBase const&, RenamedPrimEntries const&) override {}
    void PrimsDirtied(HdSceneIndexBase const&, DirtiedPrimEntries const& entries) override {
        for (auto const& e : entries)
            rest |= e.primPath == SdfPath("/Scalp") && e.dirtyLocators.Intersects(
                HdDataSourceLocator(TfToken("usdGen"), TfToken("rest"), TfToken("points")));
    }
};

class LengthNotice final : public HdSceneIndexObserver {
public:
    bool seen = false;
    void PrimsAdded(HdSceneIndexBase const&, AddedPrimEntries const&) override {}
    void PrimsRemoved(HdSceneIndexBase const&, RemovedPrimEntries const&) override {}
    void PrimsRenamed(HdSceneIndexBase const&, RenamedPrimEntries const&) override {}
    void PrimsDirtied(HdSceneIndexBase const&, DirtiedPrimEntries const& entries) override {
        for (auto const& e : entries)
            seen |= e.primPath == SdfPath("/Character/Groom/hair/Ops/length") &&
                e.dirtyLocators.Intersects(HdDataSourceLocator(TfToken("usdGen"), TfToken("cullThreshold")));
    }
};

int main() {
    auto stage = UsdStage::Open(std::string(USDGEN_TEST_SOURCE_DIR) +
        "/examples/cuda-width-network.usda");
    CHECK(stage);
    UsdImagingCreateSceneIndicesInfo info; info.stage = stage;
    auto indices = UsdImagingCreateSceneIndices(info);
    indices.stageSceneIndex->SetTime(UsdTimeCode(24));
    auto desc = usdGenImaging::BuildGraphDescFromHydra(
        *indices.finalSceneIndex, SdfPath("/Groom/hair"));
    CHECK(desc.validationErrors.empty());
    CHECK(desc.nodes.size() == 2 && desc.nodes[0].type == TfToken("UsdGenCurveSource"));
    CHECK(desc.nodes[0].inputs.empty() && desc.nodes[1].inputs == SdfPathVector{desc.nodes[0].path});
    CHECK(desc.nodes[1].expressionBindings.size() == 1);
    CHECK(desc.nodes[1].expressionBindings[0].domain == expr::Domain::Point);
    CHECK(desc.curveSets.size() == 1 && !desc.curveSets[0].restFromCurrentPoints);
    CHECK(desc.curveSets[0].rest.size() == 5 && desc.curveSets[0].rest[0][2] == 0);
    // Canonical authored useRest connection: the bool groom expression
    // overrides a false literal without replacing loaded points/rest channels.
    UsdPrim const useRestSourcePrim = stage->GetPrimAtPath(SdfPath("/Groom/hair/Ops/source"));
    UsdPrim const useRestExpression = stage->DefinePrim(
        SdfPath("/Groom/hair/Expressions/useRest"), TfToken("UsdGenExpression"));
    CHECK(useRestSourcePrim && useRestExpression);
    CHECK(useRestExpression.GetAttribute(TfToken("usdGen:expr:source")).Set(std::string("$frame < 25")));
    CHECK(useRestExpression.CreateAttribute(TfToken("outputs:result"), SdfValueTypeNames->Bool, true));
    UsdAttribute const useRestConnect = useRestSourcePrim.CreateAttribute(TfToken("usdGen:useRest"), SdfValueTypeNames->Bool, true);
    CHECK(useRestConnect.Set(false));
    CHECK(useRestConnect.SetConnections(SdfPathVector{SdfPath("/Groom/hair/Expressions/useRest.outputs:result")}));
    useRestConnect.SetCustomDataByKey(TfToken("usdGen:evaluation"), VtValue(std::string("groom")));
    indices.stageSceneIndex->ApplyPendingUpdates();
    auto useRestDesc = usdGenImaging::BuildGraphDescFromHydra(*indices.finalSceneIndex, SdfPath("/Groom/hair"));
    CHECK(useRestDesc.validationErrors.empty() && useRestDesc.nodes[0].expressionBindings.size() == 1);
    UsdGenSession session;
    session.SetDevicePublicationEnabled(true);
    usdGenUsd::SetGraphDesc(session, useRestDesc);
    auto generation = session.Commit(24, UsdGenCommitReason::SetTime);
    if (session.LastDiagnostics().HasErrors())
        for (auto const& error : session.LastDiagnostics().errors) std::fprintf(stderr, "%s\n", error.c_str());
    CHECK(generation && generation->device && usdGen::UsdGenCurvePublicationOf(*generation).tiles.empty());
    CHECK(!generation->device->Geometry().alreadyDeformed);
    auto lease = gpu::AcquireGeometry(generation->device, nullptr);
    CHECK(lease && lease.Geometry().pointCount == 5);
    float widths[5]{};
    float3 points[5]{};
    float3 rest[5]{};
    CHECK(cudaMemcpy(widths, lease.Geometry().widths.data, sizeof(widths), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(points, lease.Geometry().points.data, sizeof(points), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(rest, lease.Geometry().restPoints.data, sizeof(rest), cudaMemcpyDeviceToHost) == cudaSuccess);
    float const expected[5]{.04f, .03f, .02f, .04f, .02f};
    for (size_t i = 0; i < 5; ++i) {
        CHECK(std::abs(widths[i] - expected[i]) < 1e-6f);
        // CurveSource keeps loaded points and the Default-time rest channel
        // separate; useRest declares space, it does not overwrite points.
        CHECK(points[i].z == 5 && rest[i].z == 0);
    }
    auto currentGeneration = session.Commit(25, UsdGenCommitReason::SetTime);
    CHECK(currentGeneration && currentGeneration != generation && currentGeneration->device &&
          currentGeneration->device->Geometry().alreadyDeformed);
    auto rbfStage = UsdStage::Open(std::string(USDGEN_TEST_SOURCE_DIR) +
        "/examples/cuda-rbf-network.usda");
    CHECK(rbfStage);
    UsdImagingCreateSceneIndicesInfo rbfInfo; rbfInfo.stage = rbfStage;
    auto rbfIndices = UsdImagingCreateSceneIndices(rbfInfo);
    rbfIndices.stageSceneIndex->SetTime(UsdTimeCode(24));
    auto rbfDesc = usdGenImaging::BuildGraphDescFromHydra(
        *rbfIndices.finalSceneIndex, SdfPath("/Groom/hair"));
    CHECK(rbfDesc.nodes.size() == 3 && rbfDesc.nodes[1].type == TfToken("UsdGenDeform"));
    CHECK(rbfDesc.validationErrors.empty());
    CHECK(rbfDesc.surfaces.size() == 1 && rbfDesc.surfaces[0].restPoints.size() == 5 &&
          rbfDesc.surfaces[0].restPoints[0][0] == 0 && !rbfDesc.surfaces[0].restFromCurrentPoints);
    usdGenUsd::SetGraphDesc(session, rbfDesc);
    auto deformed = session.Commit(24, UsdGenCommitReason::SetTime);
    for (auto const& error : session.LastDiagnostics().errors) std::fprintf(stderr, "%s\n", error.c_str());
    CHECK(!session.LastDiagnostics().HasErrors());
    CHECK(deformed && deformed != currentGeneration && deformed->device && usdGen::UsdGenCurvePublicationOf(*deformed).tiles.empty());
    CHECK(deformed->device->Geometry().alreadyDeformed);
    auto deformedLease = gpu::AcquireGeometry(deformed->device, nullptr);
    CHECK(deformedLease && deformedLease.Geometry().pointCount == 5);
    CHECK(cudaMemcpy(points, deformedLease.Geometry().points.data, sizeof(points), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(widths, deformedLease.Geometry().widths.data, sizeof(widths), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (size_t i = 0; i < 5; ++i) {
        float x = i < 3 ? 1.25f : 1.75f;
        CHECK(std::abs(points[i].x - x) < 1e-5f);
        CHECK(std::abs(widths[i] - .02f * (1 + x)) < 1e-6f);
    }
    auto bindings = UsdGenCudaBindingStatsOf(session.Snapshot());
    CHECK(bindings.size() == 1 && bindings[0].bindCount == 1 && bindings[0].sampleCount == 5);
    RestNotice notice;
    auto observer = TfCreateWeakPtr(&notice);
    rbfIndices.finalSceneIndex->AddObserver(observer);
    // A Default-time rest edit must invalidate/rebuild the persistent bind,
    // even though the animated current pose remains at frame 24.
    auto newRest = rbfDesc.surfaces[0].restPoints;
    newRest[0][0] = -.1f;
    UsdGeomMesh scalp(rbfStage->GetPrimAtPath(SdfPath("/Scalp")));
    CHECK(scalp.GetPointsAttr().Set(newRest, UsdTimeCode::Default()));
    rbfIndices.stageSceneIndex->ApplyPendingUpdates();
    CHECK(notice.rest);
    auto editedDesc = usdGenImaging::BuildGraphDescFromHydra(
        *rbfIndices.finalSceneIndex, SdfPath("/Groom/hair"));
    CHECK(editedDesc.validationErrors.empty() && editedDesc.surfaces.size() == 1);
    CHECK(editedDesc.surfaces[0].restPoints == newRest &&
          editedDesc.surfaces[0].points == rbfDesc.surfaces[0].points);
    usdGenUsd::SetGraphDesc(session, editedDesc);
    auto edited = session.Commit(24, UsdGenCommitReason::SetTime);
    CHECK(edited && edited != deformed && !session.LastDiagnostics().HasErrors());
    auto editedBinding = UsdGenCudaBindingStatsOf(session.Snapshot());
    CHECK(editedBinding.size() == 1 && editedBinding[0].identity != bindings[0].identity);

    // Rest normals are a separate Default-time surface binding: authored
    // Default normals/interpolation must agree through Stage and Hydra,
    // while a current-time normal sample cannot replace them.
    VtVec3fArray defaultRestNormals(rbfDesc.surfaces[0].restPoints.size(),
                                    GfVec3f(0, 0, 1));
    VtVec3fArray posedNormals(rbfDesc.surfaces[0].restPoints.size(),
                              GfVec3f(0, 1, 0));
    CHECK(scalp.CreateNormalsAttr(VtValue(defaultRestNormals)));
    CHECK(scalp.SetNormalsInterpolation(TfToken("vertex")));
    CHECK(scalp.GetNormalsAttr().Set(VtValue(posedNormals), UsdTimeCode(24)));
    rbfIndices.stageSceneIndex->ApplyPendingUpdates();
    usdGenImaging::UsdGenGraphDescBuildOptions normalOptions;
    normalOptions.time = 24.0;
    auto const normalStageDesc = usdGenImaging::BuildGraphDescFromStage(
        rbfStage, SdfPath("/Groom/hair"), normalOptions);
    auto const normalHydraDesc = usdGenImaging::BuildGraphDescFromHydra(
        *rbfIndices.finalSceneIndex, SdfPath("/Groom/hair"), normalOptions);
    CHECK(normalStageDesc.surfaces.size() == 1 &&
          normalHydraDesc.surfaces.size() == 1);
    CHECK(normalStageDesc.surfaces[0].restNormals == defaultRestNormals &&
          normalHydraDesc.surfaces[0].restNormals == defaultRestNormals &&
          normalStageDesc.surfaces[0].restNormalDomain ==
              UsdGenSurfaceNormalDomain::Vertex &&
          normalHydraDesc.surfaces[0].restNormalDomain ==
              UsdGenSurfaceNormalDomain::Vertex);
    CHECK(scalp.SetNormalsInterpolation(TfToken("varying")));
    rbfIndices.stageSceneIndex->ApplyPendingUpdates();
    auto const invalidNormalsStage = usdGenImaging::BuildGraphDescFromStage(
        rbfStage, SdfPath("/Groom/hair"), normalOptions);
    auto const invalidNormalsHydra = usdGenImaging::BuildGraphDescFromHydra(
        *rbfIndices.finalSceneIndex, SdfPath("/Groom/hair"), normalOptions);
    CHECK(invalidNormalsStage.surfaces.size() == 1 &&
          invalidNormalsHydra.surfaces.size() == 1 &&
          invalidNormalsStage.surfaces[0].restNormalDomain ==
              UsdGenSurfaceNormalDomain::Invalid &&
          invalidNormalsHydra.surfaces[0].restNormalDomain ==
              UsdGenSurfaceNormalDomain::Invalid);
    CHECK(scalp.SetNormalsInterpolation(TfToken("vertex")));
    rbfIndices.stageSceneIndex->ApplyPendingUpdates();

    // A failed edit must not replace the last published immutable generation.
    CHECK(scalp.GetFaceVertexIndicesAttr().Set(VtIntArray{99,1,2,0,1,3,1,2,4}));
    rbfIndices.stageSceneIndex->ApplyPendingUpdates();
    auto invalidDesc = usdGenImaging::BuildGraphDescFromHydra(
        *rbfIndices.finalSceneIndex, SdfPath("/Groom/hair"));
    usdGenUsd::SetGraphDesc(session, invalidDesc);
    CHECK(session.Commit(24, UsdGenCommitReason::SetTime) == edited);
    CHECK(session.LastDiagnostics().HasErrors());
    rbfIndices.finalSceneIndex->RemoveObserver(observer);
    auto lengthStage = UsdStage::Open(std::string(USDGEN_TEST_SOURCE_DIR) +
        "/examples/cuda-length-network.usda");
    CHECK(lengthStage);
    UsdImagingCreateSceneIndicesInfo lengthInfo; lengthInfo.stage = lengthStage;
    auto lengthIndices = UsdImagingCreateSceneIndices(lengthInfo);
    lengthIndices.stageSceneIndex->SetTime(UsdTimeCode(1));
    auto lengthDesc = usdGenImaging::BuildGraphDescFromHydra(
        *lengthIndices.finalSceneIndex, SdfPath("/Character/Groom/hair"));
    CHECK(lengthDesc.validationErrors.empty() && lengthDesc.nodes.size() == 3);
    CHECK(lengthDesc.nodes[0].type == TfToken("UsdGenCurveSource") &&
          lengthDesc.nodes[1].type == TfToken("UsdGenLength"));

    // Exercise the authored transport, not a hand-built graph description:
    // canonical source connect carries a groom-native Int32 expression.
    UsdPrim const sourcePrim = lengthStage->GetPrimAtPath(
        SdfPath("/Character/Groom/hair/Ops/source"));
    UsdPrim const resampleExpression = lengthStage->DefinePrim(
        SdfPath("/Character/Groom/hair/Expressions/resampleTo"), TfToken("UsdGenExpression"));
    CHECK(sourcePrim && resampleExpression);
    CHECK(resampleExpression.GetAttribute(TfToken("usdGen:expr:source")).Set(
        std::string("$frame > 1 ? 4 : 0")));
    CHECK(resampleExpression.CreateAttribute(TfToken("outputs:result"), SdfValueTypeNames->Int, true));
    UsdAttribute const resampleConnect = sourcePrim.CreateAttribute(
        TfToken("usdGen:resampleTo"), SdfValueTypeNames->Int, true);
    CHECK(resampleConnect.SetConnections(SdfPathVector{
        SdfPath("/Character/Groom/hair/Expressions/resampleTo.outputs:result")}));
    resampleConnect.SetCustomDataByKey(TfToken("usdGen:evaluation"), VtValue(std::string("groom")));
    lengthIndices.stageSceneIndex->ApplyPendingUpdates();
    auto resampleDesc = usdGenImaging::BuildGraphDescFromHydra(
        *lengthIndices.finalSceneIndex, SdfPath("/Character/Groom/hair"));
    CHECK(resampleDesc.validationErrors.empty() && resampleDesc.nodes[0].expressionBindings.size() == 1);
    usdGenUsd::SetGraphDesc(session, resampleDesc);
    auto resampleRagged = session.Commit(1, UsdGenCommitReason::SetTime);
    CHECK(resampleRagged && resampleRagged->device);
    auto resampleUniform = session.Commit(2, UsdGenCommitReason::SetTime);
    CHECK(resampleUniform && resampleUniform != resampleRagged && resampleUniform->device);
    auto resampleLease = gpu::AcquireGeometry(resampleUniform->device, nullptr);
    CHECK(resampleLease && resampleLease.Geometry().pointCount == 8);
    uint64_t resampleIds[2]{};
    CHECK(cudaMemcpy(resampleIds, resampleLease.Geometry().stableIds.data,
                     sizeof(resampleIds), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(resampleIds[0] == 10 && resampleIds[1] == 20);
    CHECK(resampleConnect.ClearConnections());
    CHECK(lengthStage->RemovePrim(resampleExpression.GetPath()));
    lengthIndices.stageSceneIndex->ApplyPendingUpdates();

    // Author the supported primitive-domain float2 random binding through
    // USD, then take both Stage and Hydra transport paths into actual CUDA
    // compilation/execution.  This is specifically not a hand-built graph.
    SdfPath const randomExpressionPath(
        "/Character/Groom/hair/Expressions/lengthRandom");
    UsdPrim const randomExpression = lengthStage->DefinePrim(
        randomExpressionPath, TfToken("UsdGenExpression"));
    CHECK(randomExpression);
    CHECK(randomExpression.GetAttribute(TfToken("usdGen:expr:source")).Set(
        std::string("[$primIndex * 0 + 2, $primIndex * 0 + 2]")));
    CHECK(randomExpression.CreateAttribute(
        TfToken("outputs:result"), SdfValueTypeNames->Float2, true));
    UsdAttribute const randomAttribute = lengthStage->GetAttributeAtPath(
        SdfPath("/Character/Groom/hair/Ops/length.usdGen:length:random"));
    CHECK(randomAttribute);
    auto hasRandomTransport = [](UsdGenGraphDesc const& graph) {
        bool output = false;
        for (auto const& expression : graph.expressions)
            if (expression.path.GetName() == TfToken("lengthRandom") &&
                expression.outputs.size() == 1) {
                auto const& shape = expression.outputs.front().shape;
                output = expression.outputs.front().nativeType == TfToken("float2") &&
                    shape.scalar == expr::ScalarType::Float32 &&
                    shape.components == 2 && shape.elementCount == 1 &&
                    !shape.isArray;
            }
        bool binding = false;
        for (auto const& node : graph.nodes) if (node.type == TfToken("UsdGenLength"))
            for (auto const& value : node.expressionBindings)
                if (value.destination == TfToken("usdGen:length:random")) {
                    auto const& shape = value.destinationShape;
                    binding = value.nativeType == TfToken("float2") &&
                        value.domain == expr::Domain::Primitive &&
                        shape.scalar == expr::ScalarType::Float32 &&
                        shape.components == 2 && shape.elementCount == 1 &&
                        !shape.isArray;
                }
        return output && binding;
    };
    usdGenUsd::SetGraphDesc(session, lengthDesc);
    auto culled = session.Commit(1, UsdGenCommitReason::SetTime);
    for (auto const& error : session.LastDiagnostics().errors) std::fprintf(stderr, "%s\n", error.c_str());
    CHECK(culled && culled != edited && culled->device && !session.LastDiagnostics().HasErrors());
    auto culledLease = gpu::AcquireGeometry(culled->device, nullptr);
    CHECK(culledLease && culledLease.Geometry().curveCount == 2 && culledLease.Geometry().pointCount == 7);
    uint64_t lengthIds[2]{};
    float lengthWidths[7]{};
    CHECK(cudaMemcpy(lengthIds, culledLease.Geometry().stableIds.data, sizeof(lengthIds), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(lengthIds[0] == 10 && lengthIds[1] == 20);
    CHECK(cudaMemcpy(lengthWidths, culledLease.Geometry().widths.data, sizeof(lengthWidths), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (size_t i = 0; i < 7; ++i) CHECK(std::abs(lengthWidths[i] - (i < 3 ? .03f : .06f)) < 1e-6f);
    // Culling intentionally uses the current curve length and does not apply
    // the random multiplier.  Switch to the supported scale path so the
    // connected primitive float2 has an observable geometry effect.
    UsdPrim const lengthPrim = lengthStage->GetPrimAtPath(
        SdfPath("/Character/Groom/hair/Ops/length"));
    UsdAttribute const lengthMode = lengthPrim.GetAttribute(
        TfToken("usdGen:length:mode"));
    UsdAttribute const lengthValue = lengthPrim.GetAttribute(
        TfToken("usdGen:length:value"));
    UsdAttribute const cullThreshold = lengthPrim.GetAttribute(
        TfToken("usdGen:cullThreshold"));
    CHECK(lengthMode.Set(TfToken("scale")) && lengthValue.Set(.5f) &&
          cullThreshold.Set(0.0f));
    lengthIndices.stageSceneIndex->ApplyPendingUpdates();
    auto const scaledLiteralHydraDesc = usdGenImaging::BuildGraphDescFromHydra(
        *lengthIndices.finalSceneIndex, SdfPath("/Character/Groom/hair"));
    CHECK(scaledLiteralHydraDesc.validationErrors.empty());
    usdGenUsd::SetGraphDesc(session, scaledLiteralHydraDesc);
    auto scaledLength = session.Commit(1, UsdGenCommitReason::SetTime);
    CHECK(scaledLength && scaledLength != culled && scaledLength->device &&
          !session.LastDiagnostics().HasErrors());
    auto scaledLengthLease = gpu::AcquireGeometry(scaledLength->device, nullptr);
    CHECK(scaledLengthLease && scaledLengthLease.Geometry().curveCount == 3 &&
          scaledLengthLease.Geometry().pointCount == 9);
    float3 scaledLengthPoints[9]{};
    CHECK(cudaMemcpy(scaledLengthPoints, scaledLengthLease.Geometry().points.data,
                     sizeof(scaledLengthPoints), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(std::abs(scaledLengthPoints[8].y - .2f) < 2e-3f);

    CHECK(randomAttribute.SetConnections(SdfPathVector{
        SdfPath("/Character/Groom/hair/Expressions/lengthRandom.outputs:result")}));
    lengthIndices.stageSceneIndex->ApplyPendingUpdates();
    auto const randomStageDesc = usdGenImaging::BuildGraphDescFromStage(
        lengthStage, SdfPath("/Character/Groom/hair"));
    auto const randomHydraDesc = usdGenImaging::BuildGraphDescFromHydra(
        *lengthIndices.finalSceneIndex, SdfPath("/Character/Groom/hair"));
    for (auto const &e : randomStageDesc.validationErrors)
        std::printf("stage validation: %s\n", e.c_str());
    for (auto const &e : randomHydraDesc.validationErrors)
        std::printf("hydra validation: %s\n", e.c_str());
    CHECK(randomStageDesc.validationErrors.empty() &&
          randomHydraDesc.validationErrors.empty());
    CHECK(hasRandomTransport(randomStageDesc) && hasRandomTransport(randomHydraDesc));
    usdGenUsd::SetGraphDesc(session, randomHydraDesc);
    auto randomLength = session.Commit(1, UsdGenCommitReason::SetTime);
    CHECK(randomLength && randomLength != scaledLength && randomLength->device &&
          !session.LastDiagnostics().HasErrors());
    auto randomLengthLease = gpu::AcquireGeometry(randomLength->device, nullptr);
    CHECK(randomLengthLease && randomLengthLease.Geometry().curveCount == 3 &&
          randomLengthLease.Geometry().pointCount == 9);
    uint64_t randomLengthIds[3]{};
    CHECK(cudaMemcpy(randomLengthIds, randomLengthLease.Geometry().stableIds.data,
                     sizeof(randomLengthIds), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(randomLengthIds[0] == 10 && randomLengthIds[1] == 20 &&
          randomLengthIds[2] == 30);
    float3 randomLengthPoints[9]{};
    CHECK(cudaMemcpy(randomLengthPoints, randomLengthLease.Geometry().points.data,
                     sizeof(randomLengthPoints), cudaMemcpyDeviceToHost) == cudaSuccess);
    // The primitive [2,2] field doubles the .5 scale field, restoring the
    // third sorted curve's .4-unit tip rather than only changing metadata.
    CHECK(std::abs(randomLengthPoints[8].y - .4f) < 2e-3f &&
          std::abs(randomLengthPoints[8].y - scaledLengthPoints[8].y) > .1f);

    // Restore the fixture's original cull configuration before exercising the
    // independent high-threshold all-cull path below.
    CHECK(randomAttribute.ClearConnections());
    CHECK(lengthMode.Set(TfToken("cull")) && lengthValue.Set(1.0f));
    lengthIndices.stageSceneIndex->ApplyPendingUpdates();
    LengthNotice lengthNotice;
    auto lengthObserver = TfCreateWeakPtr(&lengthNotice);
    lengthIndices.finalSceneIndex->AddObserver(lengthObserver);
    CHECK(lengthStage->GetPrimAtPath(SdfPath("/Character/Groom/hair/Ops/length"))
        .GetAttribute(TfToken("usdGen:cullThreshold")).Set(.9f));
    lengthIndices.stageSceneIndex->ApplyPendingUpdates();
    CHECK(lengthNotice.seen);
    auto emptyLengthDesc = usdGenImaging::BuildGraphDescFromHydra(
        *lengthIndices.finalSceneIndex, SdfPath("/Character/Groom/hair"));
    usdGenUsd::SetGraphDesc(session, emptyLengthDesc);
    auto emptyLength = session.Commit(1, UsdGenCommitReason::SetTime);
    CHECK(emptyLength && emptyLength != culled && emptyLength->device && !session.LastDiagnostics().HasErrors());
    auto emptyLengthLease = gpu::AcquireGeometry(emptyLength->device, nullptr);
    CHECK(emptyLengthLease && emptyLengthLease.Geometry().curveCount == 0 && emptyLengthLease.Geometry().pointCount == 0);
    CHECK(culledLease.Geometry().curveCount == 2);
    lengthIndices.finalSceneIndex->RemoveObserver(lengthObserver);
    std::puts("testUsdGenCudaHierarchy: PASS");
}