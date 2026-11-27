#include "utils/pxrInterop.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/half.h"
#include "usdGenCurves/cuda/cudaParameters.h"
#include "usdGen/executionResources.h"
#include "usdGenUsd/lower.h"
#include "usdGenCurves/cuda/curveSource.h"
#include <atomic>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
using namespace usdGen;
namespace {
struct FreshCallbackStatus {
    std::atomic<int> calls{0};
    std::atomic<cudaError_t> status{cudaErrorUnknown};
};
void CUDART_CB FreshCallback(cudaStream_t, cudaError_t status, void* userdata) {
    auto* result = static_cast<FreshCallbackStatus*>(userdata);
    result->status.store(status, std::memory_order_release);
    result->calls.fetch_add(1, std::memory_order_release);
}
bool CompleteFreshBatch(cudaStream_t stream, FreshCallbackStatus* status) {
    return cudaStreamAddCallback(stream, FreshCallback, status, 0) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess &&
        status->calls.load(std::memory_order_acquire) == 1 &&
        status->status.load(std::memory_order_acquire) == cudaSuccess;
}
} // namespace
int main() {
    int failures = 0;
    auto check = [&](bool condition, char const* message) {
        if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
    };
    UsdGenGraphDesc graph;
    UsdGenExpressionDesc expression;
    expression.path = SdfPath("/Expr"); expression.source = "$value * 2";
    expression.outputs.push_back({TfToken("result"), TfToken("float"), {expr::ScalarType::Float32,1,1,1,1,false}});
    graph.expressions.push_back(expression);
    UsdGenNodeDesc node;
    UsdGenExpressionBinding binding;
    binding.expression = expression.path; binding.destination = TfToken("width");
    binding.destinationShape.scalar = expr::ScalarType::Float32;
    binding.destinationShape.components = 1; binding.literal = VtValue(0.5f);
    node.expressionBindings.push_back(binding);
    CudaParameterPlan plan; std::vector<std::string> diagnostics;
    auto status = CudaParameterPlan::Compile(usdGenUsd::Lower(graph), usdGenUsd::Lower(node), &plan, &diagnostics);
    if (status != CudaParameterStatus::Ok || plan.Bindings().size() != 1 || !plan.Fields().empty() || plan.Find(TfToken("width"))) {
        std::fprintf(stderr, "parameter plan compile failed (%d)\n", int(status));
        for (auto const& message : diagnostics) std::fprintf(stderr, "  %s\n", message.c_str());
        return 1;
    }
    // Compilation remains CPU-only and may contain all three consumer rates.
    for (auto domain : {expr::Domain::Groom, expr::Domain::Primitive, expr::Domain::Point}) {
        UsdGenExpressionBinding b = binding;
        b.destination = TfToken(domain == expr::Domain::Groom ? "enabled" : domain == expr::Domain::Primitive ? "seed" : "widthPoint");
        b.domain = domain;
        node.expressionBindings.push_back(b);
    }
    CudaParameterPlan mixed;
    if (CudaParameterPlan::Compile(usdGenUsd::Lower(graph), usdGenUsd::Lower(node), &mixed, &diagnostics) != CudaParameterStatus::Ok ||
        mixed.Bindings().size() != 4) {
        std::fprintf(stderr, "mixed-domain parameter plan compile failed\n"); return 1;
    }
    auto malformed = node;
    malformed.expressionBindings.back().destinationShape.isArray = true;
    if (CudaParameterPlan::Compile(usdGenUsd::Lower(graph), usdGenUsd::Lower(malformed), &mixed, &diagnostics) != CudaParameterStatus::UnsupportedType) {
        std::fprintf(stderr, "malformed parameter shape was accepted\n"); return 1;
    }
    auto wrongLiteral = node;
    wrongLiteral.expressionBindings[0].literal = VtValue(double(.5));
    check(CudaParameterPlan::Compile(usdGenUsd::Lower(graph), usdGenUsd::Lower(wrongLiteral), &mixed, &diagnostics) == CudaParameterStatus::UnsupportedType,
          "reject non-native float literal");
    auto hugeInteger = binding;
    hugeInteger.destination = TfToken("largeInt");
    hugeInteger.nativeType = TfToken("int64");
    hugeInteger.destinationShape.scalar = expr::ScalarType::Int64;
    hugeInteger.literal = VtValue(int64_t(1) << 54);
    UsdGenExpressionDesc intExpression;
    intExpression.path = SdfPath("/IntExpr"); intExpression.source = "$value";
    intExpression.outputs.push_back({TfToken("result"), TfToken("int64"),
        {expr::ScalarType::Int64, 1, 1, 1, 1, false}});
    graph.expressions.push_back(intExpression);
    hugeInteger.expression = intExpression.path;
    UsdGenNodeDesc intNode; intNode.expressionBindings.push_back(hugeInteger);
    check(CudaParameterPlan::Compile(usdGenUsd::Lower(graph), usdGenUsd::Lower(intNode), &mixed, &diagnostics) == CudaParameterStatus::UnsupportedType,
          "reject int64 literal beyond exact double range");
    UsdGenNodeDesc duplicateNode;
    auto duplicateA = binding; duplicateA.destination = TfToken("width");
    auto duplicateB = binding; duplicateB.destination = TfToken("usdGen:width");
    duplicateNode.expressionBindings = {duplicateA, duplicateB};
    check(CudaParameterPlan::Compile(usdGenUsd::Lower(graph), usdGenUsd::Lower(duplicateNode), &mixed, &diagnostics) == CudaParameterStatus::InvalidArgument,
          "reject canonical duplicate destinations");
    auto badOutputGraph = graph;
    badOutputGraph.expressions.front().outputs.front().shape.elementCount = 2;
    check(CudaParameterPlan::Compile(usdGenUsd::Lower(badOutputGraph), usdGenUsd::Lower(node), &mixed, &diagnostics) == CudaParameterStatus::InvalidArgument,
          "reject output element-count mismatch");

    auto addExpression = [&](char const* path, char const* source) {
        UsdGenExpressionDesc e; e.path = SdfPath(path); e.source = source;
        e.outputs.push_back({TfToken("result"), TfToken("float"),
                             {expr::ScalarType::Float32, 1, 1, 1, 1, false}});
        graph.expressions.push_back(std::move(e));
    };
    addExpression("/Frame", "$frame + $value");
    addExpression("/Prim", "$primIndex + $value");
    addExpression("/Point", "$P[0] + $value");
    addExpression("/Bool", "$frame >= 2");
    graph.expressions.back().outputs[0].nativeType = TfToken("bool");
    graph.expressions.back().outputs[0].shape.scalar = expr::ScalarType::Bool;
    UsdGenNodeDesc runtimeNode;
    auto addBinding = [&](char const* expression, char const* destination,
                          expr::Domain domain, expr::ScalarType scalar, float literal) {
        UsdGenExpressionBinding b; b.expression = SdfPath(expression);
        b.destination = TfToken(destination); b.domain = domain;
        b.destinationShape = {scalar, 1, 1, 1, 1, false}; b.literal = VtValue(literal);
        runtimeNode.expressionBindings.push_back(std::move(b));
    };
    addBinding("/Frame", "frameValue", expr::Domain::Groom, expr::ScalarType::Float32, .5f);
    addBinding("/Prim", "primValue", expr::Domain::Primitive, expr::ScalarType::Float32, 1.f);
    addBinding("/Point", "pointValue", expr::Domain::Point, expr::ScalarType::Float32, 2.f);
    addBinding("/Bool", "frameBool", expr::Domain::Groom, expr::ScalarType::Bool, 0.f);
    runtimeNode.expressionBindings.back().nativeType = TfToken("bool");
    runtimeNode.expressionBindings.back().literal = VtValue(false);
    CudaParameterPlan runtimePlan;
    if (CudaParameterPlan::Compile(usdGenUsd::Lower(graph), usdGenUsd::Lower(runtimeNode), &runtimePlan, &diagnostics) != CudaParameterStatus::Ok) {
        std::fprintf(stderr, "runtime parameter plan compile failed\n");
        for (auto const& message : diagnostics) std::fprintf(stderr, "  %s\n", message.c_str());
        return 1;
    }
    cudaStream_t stream = nullptr;
    auto streamStatus = cudaStreamCreate(&stream);
    if (streamStatus != cudaSuccess) {
        std::fprintf(stderr, "CUDA stream creation failed: %s\n", cudaGetErrorString(streamStatus));
        return 2;
    }
    std::vector<int32_t> counts{2, 3};
    std::vector<float3> points{{1,0,0},{2,0,0},{10,0,0},{11,0,0},{12,0,0}};
    std::vector<float> widths(points.size(), 1.f);
    std::vector<uint64_t> ids{4, 9};
    gpu::CurveSourceInput sourceInput;
    sourceInput.curveVertexCounts = {counts.data(), counts.size()};
    sourceInput.points = {points.data(), points.size()};
    sourceInput.widths = {widths.data(), widths.size()};
    sourceInput.stableIds = {ids.data(), ids.size()};
    gpu::CudaCurveSource curveSource;
    if (curveSource.Set(sourceInput, stream) != gpu::CurveSourceStatus::Ok ||
        curveSource.Finish(stream) != gpu::CurveSourceStatus::Ok) {
        std::fprintf(stderr, "curve source setup failed\n"); return 2;
    }
    std::shared_ptr<const CudaParameterProgram> immutableProgram;
    check(CudaParameterProgram::Compile(usdGenUsd::Lower(graph), usdGenUsd::Lower(runtimeNode), &immutableProgram,
                                        &diagnostics) == CudaParameterStatus::Ok &&
              immutableProgram && immutableProgram->Bindings().size() == 4,
          "compile immutable CUDA parameter program");
    CudaParameterEvaluator evaluatorA;
    CudaParameterEvaluator evaluatorB;
    expr::Context independentControls;
    independentControls.domain = expr::Domain::Point;
    independentControls.frame = 1;
    check(immutableProgram &&
              evaluatorA.Evaluate(*immutableProgram, curveSource.view(), {},
                                  independentControls, stream, &diagnostics) ==
                  CudaParameterStatus::Ok,
          "evaluate immutable program in first workspace");
    float firstWorkspaceValue = 0.0f;
    auto const* firstWorkspaceField = evaluatorA.Find(TfToken("frameValue"));
    check(firstWorkspaceField &&
              cudaMemcpyAsync(&firstWorkspaceValue, firstWorkspaceField->data,
                              sizeof(firstWorkspaceValue), cudaMemcpyDeviceToHost,
                              stream) == cudaSuccess &&
              cudaStreamSynchronize(stream) == cudaSuccess &&
              std::fabs(firstWorkspaceValue - 1.5f) < 1e-5f,
          "first evaluator publishes independent frame result");
    independentControls.frame = 2;
    check(immutableProgram &&
              evaluatorB.Evaluate(*immutableProgram, curveSource.view(), {},
                                  independentControls, stream, &diagnostics) ==
                  CudaParameterStatus::Ok,
          "evaluate immutable program in second workspace");
    float secondWorkspaceValue = 0.0f;
    auto const* secondWorkspaceField = evaluatorB.Find(TfToken("frameValue"));
    check(secondWorkspaceField &&
              cudaMemcpyAsync(&secondWorkspaceValue, secondWorkspaceField->data,
                              sizeof(secondWorkspaceValue), cudaMemcpyDeviceToHost,
                              stream) == cudaSuccess &&
              cudaStreamSynchronize(stream) == cudaSuccess &&
              std::fabs(secondWorkspaceValue - 2.5f) < 1e-5f,
          "second evaluator publishes a different frame result");
    float firstWorkspaceAfter = 0.0f;
    check(firstWorkspaceField &&
              cudaMemcpyAsync(&firstWorkspaceAfter, firstWorkspaceField->data,
                              sizeof(firstWorkspaceAfter), cudaMemcpyDeviceToHost,
                              stream) == cudaSuccess &&
              cudaStreamSynchronize(stream) == cudaSuccess &&
              std::fabs(firstWorkspaceAfter - firstWorkspaceValue) < 1e-5f,
          "second evaluator leaves first workspace unmodified");
    for (int frame = 1; frame <= 2; ++frame) {
        expr::Context controls; controls.domain = expr::Domain::Point;
        controls.frame = frame;
        if (runtimePlan.Evaluate(curveSource.view(), {}, controls, stream, &diagnostics) != CudaParameterStatus::Ok) {
            std::fprintf(stderr, "runtime parameter evaluation failed at frame %d\n", frame); return 1;
        }
        auto const* frameField = runtimePlan.Find(TfToken("frameValue"));
        auto const* primField = runtimePlan.Find(TfToken("primValue"));
        auto const* pointField = runtimePlan.Find(TfToken("pointValue"));
        auto const* boolField = runtimePlan.Find(TfToken("frameBool"));
        check(frameField && primField && pointField && boolField, "publish all mixed-domain fields");
        if (!frameField || !primField || !pointField || !boolField) continue;
        float frameValue = 0, primValue[2]{}, pointValue[5]{}; unsigned char boolValue = 0;
        check(cudaMemcpy(&frameValue, frameField->data, sizeof(frameValue), cudaMemcpyDeviceToHost) == cudaSuccess, "read groom result");
        check(cudaMemcpy(primValue, primField->data, sizeof(primValue), cudaMemcpyDeviceToHost) == cudaSuccess, "read primitive results");
        check(cudaMemcpy(pointValue, pointField->data, sizeof(pointValue), cudaMemcpyDeviceToHost) == cudaSuccess, "read point results");
        check(cudaMemcpy(&boolValue, boolField->data, sizeof(boolValue), cudaMemcpyDeviceToHost) == cudaSuccess, "read boolean result");
        check(std::fabs(frameValue - (frame + .5f)) < 1e-5f, "groom frame reevaluation");
        check(std::fabs(primValue[0]-1.f)<1e-5f && std::fabs(primValue[1]-2.f)<1e-5f, "primitive values/count");
        for (unsigned i=0;i<5;++i) check(std::fabs(pointValue[i]-(points[i].x+2.f))<1e-5f, "point values/count");
        check(boolValue == (frame >= 2), "typed bool output");
    }

    // The fresh candidate estimator covers every context, literal, output,
    // expression program, and terminal status allocation. An exact isolated
    // reservation must fully charge the candidate while preserving the warm
    // published fields until commit; destroying the evaluator returns the
    // child permits to the still-live reservation.
    CudaParameterGeometryMemoryShape candidateShape;
    candidateShape.pointCount = points.size();
    candidateShape.curveCount = counts.size();
    candidateShape.hasRestPoints = true;
    candidateShape.hasWidths = true;
    uint64_t estimatedCandidateBytes = 0;
    check(immutableProgram &&
              immutableProgram->EstimateFreshCandidateBytes(candidateShape,
                                                             &estimatedCandidateBytes) &&
              estimatedCandidateBytes > 1,
          "estimate complete fresh expression candidate");
    CudaParameterGeometryMemoryShape overflowShape;
    overflowShape.pointCount = UINT64_MAX;
    overflowShape.curveCount = UINT64_MAX;
    uint64_t overflowCandidateBytes = 0;
    check(immutableProgram &&
              !immutableProgram->EstimateFreshCandidateBytes(overflowShape,
                                                               &overflowCandidateBytes),
          "reject overflowing fresh expression candidate estimate");
    if (immutableProgram && estimatedCandidateBytes > 1) {
        size_t const exactCandidateBytes = static_cast<size_t>(estimatedCandidateBytes);
        {
            UsdGenExecutionResourcePool exactCandidatePool(exactCandidateBytes);
            auto exactCandidateReservation =
                exactCandidatePool.TryReserveMemory(exactCandidateBytes);
            auto const exactCandidateBefore = exactCandidatePool.Snapshot();
            {
            CudaParameterEvaluator exactCandidate;
            expr::Context exactControls;
            exactControls.domain = expr::Domain::Point;
            exactControls.frame = 1;
            check(exactCandidateReservation &&
                      exactCandidate.Evaluate(*immutableProgram, curveSource.view(), {},
                                              exactControls, stream, &diagnostics) ==
                          CudaParameterStatus::Ok,
                  "establish warm fields before exact fresh reservation");
            auto const* oldCandidateField = exactCandidate.Find(TfToken("frameValue"));
            void const* const oldCandidateData =
                oldCandidateField ? oldCandidateField->data : nullptr;
            float oldCandidateValue = 0;
            check(oldCandidateField &&
                      cudaMemcpy(&oldCandidateValue, oldCandidateField->data,
                                 sizeof(oldCandidateValue), cudaMemcpyDeviceToHost) ==
                          cudaSuccess,
                  "read warm field before exact fresh reservation");

            exactControls.frame = 2;
            FreshCallbackStatus exactContextProof, exactProgramProof;
            bool exactContextReady = exactCandidateReservation &&
                exactCandidate.BeginFreshContexts(
                    immutableProgram, curveSource.view(), {}, exactControls, stream,
                    &diagnostics, &*exactCandidateReservation,
                    UsdGenExecutionResourceKind::Cache) == CudaParameterStatus::Ok;
            check(exactContextReady, "exact fresh reservation admits contexts");
            bool exactProgramsReady = false;
            if (exactContextReady) {
                check(exactCandidate.EnqueueFreshContextStatus(stream) ==
                          CudaParameterStatus::Ok &&
                          CompleteFreshBatch(stream, &exactContextProof) &&
                          exactCandidate.CommitFreshContexts() == CudaParameterStatus::Ok,
                      "exact fresh reservation proves contexts");
                exactProgramsReady =
                    exactCandidate.BeginFreshPrograms(
                        stream, &*exactCandidateReservation,
                        UsdGenExecutionResourceKind::Cache) == CudaParameterStatus::Ok;
                check(exactProgramsReady, "exact fresh reservation admits programs");
            }
            if (exactProgramsReady) {
                check(exactCandidate.EnqueueFreshProgramStatus(stream) ==
                          CudaParameterStatus::Ok &&
                          CompleteFreshBatch(stream, &exactProgramProof) &&
                          exactCandidate.CommitFreshPrograms() == CudaParameterStatus::Ok &&
                          !exactCandidate.HasUnprovenWork(),
                      "exact fresh reservation commits candidate");
                auto const* freshCandidateField = exactCandidate.Find(TfToken("frameValue"));
                float freshCandidateValue = 0;
                check(freshCandidateField && freshCandidateField->data != oldCandidateData &&
                          cudaMemcpy(&freshCandidateValue, freshCandidateField->data,
                                     sizeof(freshCandidateValue), cudaMemcpyDeviceToHost) ==
                              cudaSuccess &&
                          std::fabs(freshCandidateValue - 2.5f) < 1e-5f,
                      "fresh commit publishes a private replacement field");
                auto const exactCandidateHeld = exactCandidatePool.Snapshot();
                check(exactCandidateReservation->RemainingBytes() == 0 &&
                          exactCandidateHeld.usedBytes == exactCandidateBefore.usedBytes &&
                          exactCandidateHeld.byKind[static_cast<size_t>(
                              UsdGenExecutionResourceKind::Pending)] == 0 &&
                          exactCandidateHeld.byKind[static_cast<size_t>(
                              UsdGenExecutionResourceKind::Cache)] == exactCandidateBytes,
                      "exact fresh candidate charge transfers without double charging");
            }
            check(oldCandidateField && exactCandidate.Find(TfToken("frameValue")) &&
                      oldCandidateValue == 1.5f,
                  "warm published field remains valid through fresh candidate work");
            }
        auto const exactCandidateReturned = exactCandidatePool.Snapshot();
        check(exactCandidateReservation->RemainingBytes() == exactCandidateBytes &&
                  exactCandidateReturned.usedBytes == exactCandidateBefore.usedBytes &&
                  exactCandidateReturned.byKind[static_cast<size_t>(
                      UsdGenExecutionResourceKind::Pending)] == exactCandidateBytes &&
                  exactCandidateReturned.byKind[static_cast<size_t>(
                      UsdGenExecutionResourceKind::Cache)] == 0,
              "fresh candidate destruction returns its reservation credit");
        exactCandidateReservation->Release();
        check(exactCandidatePool.Snapshot().usedBytes == 0,
              "fresh candidate reservation releases without a leak");
        }

        // One byte short is rejected by the reservation admission itself,
        // before the evaluator can submit a candidate. The warm published
        // field remains untouched and the failed full-size reservation does
        // not alter the short pool's accounting.
        size_t const shortCandidateBytes = exactCandidateBytes - 1;
        UsdGenExecutionResourcePool shortCandidatePool(shortCandidateBytes);
        auto shortCandidateReservation =
            shortCandidatePool.TryReserveMemory(shortCandidateBytes);
        auto const shortCandidateBefore = shortCandidatePool.Snapshot();
        {
            CudaParameterEvaluator shortCandidate;
            expr::Context shortControls;
            shortControls.domain = expr::Domain::Point;
            shortControls.frame = 1;
            check(shortCandidate.Evaluate(*immutableProgram, curveSource.view(), {},
                                          shortControls, stream, &diagnostics) ==
                          CudaParameterStatus::Ok,
                  "establish warm fields before short fresh reservation");
            auto const* shortOldField = shortCandidate.Find(TfToken("frameValue"));
            void const* const shortOldData = shortOldField ? shortOldField->data : nullptr;
            float shortOldValue = 0;
            check(shortOldField &&
                      cudaMemcpy(&shortOldValue, shortOldField->data,
                                 sizeof(shortOldValue), cudaMemcpyDeviceToHost) == cudaSuccess,
                  "read warm field before short fresh reservation");
            auto rejectedFullReservation =
                shortCandidatePool.TryReserveMemory(exactCandidateBytes);
            check(shortCandidateReservation && !rejectedFullReservation && shortOldField &&
                      shortCandidate.Find(TfToken("frameValue")) &&
                      shortCandidate.Find(TfToken("frameValue"))->data == shortOldData &&
                      std::fabs(shortOldValue - 1.5f) < 1e-5f,
                  "one-byte-short fresh admission preserves warm publication");
        }
        auto const shortCandidateAfter = shortCandidatePool.Snapshot();
        check(shortCandidateReservation->RemainingBytes() == shortCandidateBytes &&
                  shortCandidateAfter.usedBytes == shortCandidateBefore.usedBytes &&
                  shortCandidateAfter.byKind[static_cast<size_t>(
                      UsdGenExecutionResourceKind::Pending)] == shortCandidateBytes &&
                  shortCandidateAfter.byKind[static_cast<size_t>(
                      UsdGenExecutionResourceKind::Cache)] == 0,
              "one-byte-short fresh candidate returns partial charge");
        shortCandidateReservation->Release();
        check(shortCandidatePool.Snapshot().usedBytes == 0,
              "one-byte-short fresh reservation releases without a leak");
    }

    // The staged evaluator uses one parent-owned callback for each batch:
    // contexts first, then expression programs. It retains the legacy fields
    // until the program batch is proven and committed.
    CudaParameterEvaluator staged;
    expr::Context stagedControls;
    stagedControls.domain = expr::Domain::Point;
    stagedControls.frame = 1;
    check(immutableProgram &&
              staged.Evaluate(*immutableProgram, curveSource.view(), {}, stagedControls,
                              stream, &diagnostics) == CudaParameterStatus::Ok,
          "establish legacy fields before staged replacement");
    auto const* oldFrame = staged.Find(TfToken("frameValue"));
    float oldFrameValue = 0;
    check(oldFrame && cudaMemcpy(&oldFrameValue, oldFrame->data, sizeof(oldFrameValue),
                                 cudaMemcpyDeviceToHost) == cudaSuccess &&
          std::fabs(oldFrameValue - 1.5f) < 1e-5f,
          "read legacy field retained during staged work");
    void const* const oldFrameData = oldFrame ? oldFrame->data : nullptr;

    stagedControls.frame = 2;
    check(immutableProgram &&
              staged.BeginFreshContexts(immutableProgram, curveSource.view(), {}, stagedControls,
                                        stream, &diagnostics) == CudaParameterStatus::Ok &&
          staged.HasUnprovenWork(),
          "begin staged groom primitive and point contexts");
    check(staged.BeginFreshContexts(immutableProgram, curveSource.view(), {}, stagedControls,
                                    stream, &diagnostics) == CudaParameterStatus::InvalidArgument,
          "reject mixed second fresh context phase");
    check(staged.BeginFreshPrograms(stream) == CudaParameterStatus::InvalidArgument,
          "programs cannot evaluate before context commit");
    auto const* beforeContextCommit = staged.Find(TfToken("frameValue"));
    check(beforeContextCommit && beforeContextCommit->data == oldFrameData,
          "fields remain old while contexts are pending");
    FreshCallbackStatus contextBatch;
    check(staged.EnqueueFreshContextStatus(stream) == CudaParameterStatus::Ok &&
          CompleteFreshBatch(stream, &contextBatch),
          "one external callback proves all fresh contexts");
    check(staged.CommitFreshContexts() == CudaParameterStatus::Ok && staged.HasUnprovenWork() == false,
          "commit fresh contexts after callback proof");
    check(staged.BeginFreshPrograms(stream) == CudaParameterStatus::Ok && staged.HasUnprovenWork(),
          "begin staged programs after contexts publish");
    auto const* beforeProgramCommit = staged.Find(TfToken("frameValue"));
    check(beforeProgramCommit && beforeProgramCommit->data == oldFrameData,
          "fields remain old while programs are pending");
    FreshCallbackStatus programBatch;
    check(staged.EnqueueFreshProgramStatus(stream) == CudaParameterStatus::Ok &&
          CompleteFreshBatch(stream, &programBatch),
          "one external callback proves all staged programs");
    check(staged.CommitFreshPrograms() == CudaParameterStatus::Ok && !staged.HasUnprovenWork(),
          "commit staged program results after callback proof");
    auto const* stagedFrame = staged.Find(TfToken("frameValue"));
    auto const* stagedPrim = staged.Find(TfToken("primValue"));
    auto const* stagedPoint = staged.Find(TfToken("pointValue"));
    auto const* stagedBool = staged.Find(TfToken("frameBool"));
    float stagedFrameValue = 0, stagedPrimValues[2]{}, stagedPointValues[5]{};
    unsigned char stagedBoolValue = 0;
    check(stagedFrame && stagedPrim && stagedPoint && stagedBool &&
          cudaMemcpy(&stagedFrameValue, stagedFrame->data, sizeof(stagedFrameValue), cudaMemcpyDeviceToHost) == cudaSuccess &&
          cudaMemcpy(stagedPrimValues, stagedPrim->data, sizeof(stagedPrimValues), cudaMemcpyDeviceToHost) == cudaSuccess &&
          cudaMemcpy(stagedPointValues, stagedPoint->data, sizeof(stagedPointValues), cudaMemcpyDeviceToHost) == cudaSuccess &&
          cudaMemcpy(&stagedBoolValue, stagedBool->data, sizeof(stagedBoolValue), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read committed staged mixed-domain fields");
    check(std::fabs(stagedFrameValue - 2.5f) < 1e-5f &&
          std::fabs(stagedPrimValues[0] - 1.f) < 1e-5f &&
          std::fabs(stagedPrimValues[1] - 2.f) < 1e-5f &&
          stagedBoolValue == 1,
          "staged groom primitive and typed bool match legacy evaluation");
    for (unsigned i = 0; i < 5; ++i)
        check(std::fabs(stagedPointValues[i] - (points[i].x + 2.f)) < 1e-5f,
              "staged point fields match legacy evaluation");

    // A proved context batch may be abandoned before program submission, but
    // an unproved context/program batch must retain its candidate for the
    // terminal callback. Published fields stay stable in every discard path.
    CudaParameterEvaluator discard;
    stagedControls.frame = 1;
    check(immutableProgram &&
          discard.Evaluate(*immutableProgram, curveSource.view(), {}, stagedControls,
                           stream, &diagnostics) == CudaParameterStatus::Ok,
          "establish fields before fresh discard coverage");
    auto const* discardOld = discard.Find(TfToken("frameValue"));
    void const* const discardOldData = discardOld ? discardOld->data : nullptr;
    stagedControls.frame = 2;
    check(immutableProgram &&
          discard.BeginFreshContexts(immutableProgram, curveSource.view(), {}, stagedControls,
                                     stream, &diagnostics) == CudaParameterStatus::Ok &&
          discard.HasUnprovenWork() && !discard.DiscardFreshProven() && discardOld &&
          discard.Find(TfToken("frameValue")) &&
          discard.Find(TfToken("frameValue"))->data == discardOldData,
          "reject discard while fresh contexts remain unproven");
    FreshCallbackStatus discardContextBatch;
    check(discard.EnqueueFreshContextStatus(stream) == CudaParameterStatus::Ok &&
          CompleteFreshBatch(stream, &discardContextBatch) &&
          discard.CommitFreshContexts() == CudaParameterStatus::Ok &&
          !discard.HasUnprovenWork() && discard.DiscardFreshProven() && discardOld &&
          discard.Find(TfToken("frameValue")) &&
          discard.Find(TfToken("frameValue"))->data == discardOldData,
          "discard proved contexts preserves published fields");
    check(immutableProgram &&
          discard.BeginFreshContexts(immutableProgram, curveSource.view(), {}, stagedControls,
                                     stream, &diagnostics) == CudaParameterStatus::Ok &&
          discard.EnqueueFreshContextStatus(stream) == CudaParameterStatus::Ok,
          "restart fresh contexts after proved discard");
    FreshCallbackStatus discardRestartContextBatch;
    check(CompleteFreshBatch(stream, &discardRestartContextBatch) &&
          discard.CommitFreshContexts() == CudaParameterStatus::Ok &&
          discard.BeginFreshPrograms(stream) == CudaParameterStatus::Ok &&
          discard.HasUnprovenWork() && !discard.DiscardFreshProven() && discardOld &&
          discard.Find(TfToken("frameValue")) &&
          discard.Find(TfToken("frameValue"))->data == discardOldData,
          "reject discard while fresh programs remain unproven");
    FreshCallbackStatus discardProgramBatch;
    check(discard.EnqueueFreshProgramStatus(stream) == CudaParameterStatus::Ok &&
          CompleteFreshBatch(stream, &discardProgramBatch) &&
          discard.CommitFreshPrograms() == CudaParameterStatus::Ok &&
          !discard.HasUnprovenWork(),
          "finish discarded-program coverage terminally before reuse");

    CudaParameterEvaluator allocationRetry;
    check(immutableProgram &&
          allocationRetry.BeginFreshContexts(immutableProgram, curveSource.view(), {}, stagedControls,
                                             stream, &diagnostics) == CudaParameterStatus::Ok &&
          allocationRetry.EnqueueFreshContextStatus(stream) == CudaParameterStatus::Ok,
          "begin allocation-failure retry context batch");
    FreshCallbackStatus allocationContextBatch;
    check(CompleteFreshBatch(stream, &allocationContextBatch) &&
          allocationRetry.CommitFreshContexts() == CudaParameterStatus::Ok,
          "prove allocation-failure retry contexts");
    failNextCudaParameterFreshProgramAllocationForTesting();
    check(allocationRetry.BeginFreshPrograms(stream) == CudaParameterStatus::CudaError &&
          !allocationRetry.HasUnprovenWork() && allocationRetry.DiscardFreshProven(),
          "pre-submit program allocation failure discards without quarantine");
    check(immutableProgram &&
          allocationRetry.BeginFreshContexts(immutableProgram, curveSource.view(), {}, stagedControls,
                                             stream, &diagnostics) == CudaParameterStatus::Ok &&
          allocationRetry.EnqueueFreshContextStatus(stream) == CudaParameterStatus::Ok,
          "restart cleanly after pre-submit allocation failure");
    FreshCallbackStatus allocationRestartContextBatch;
    check(CompleteFreshBatch(stream, &allocationRestartContextBatch) &&
          allocationRetry.CommitFreshContexts() == CudaParameterStatus::Ok &&
          allocationRetry.DiscardFreshProven() && !allocationRetry.HasUnprovenWork(),
          "discard restarted proved candidate without poisoning evaluator");

    // Device semantic geometry failure reaches the context batch callback,
    // clears its proof state on commit, and never starts programs.
    uint32_t malformedOffsets[] = {0, UINT32_MAX, 5};
    check(cudaMemcpy(const_cast<uint32_t*>(curveSource.view().curveOffsets.data), malformedOffsets,
                     sizeof(malformedOffsets), cudaMemcpyHostToDevice) == cudaSuccess,
          "inject malformed staged context offsets");
    CudaParameterEvaluator malformedContexts;
    FreshCallbackStatus malformedContextBatch;
    check(immutableProgram &&
          malformedContexts.BeginFreshContexts(immutableProgram, curveSource.view(), {}, stagedControls,
                                               stream, &diagnostics) == CudaParameterStatus::Ok &&
          malformedContexts.EnqueueFreshContextStatus(stream) == CudaParameterStatus::Ok &&
          CompleteFreshBatch(stream, &malformedContextBatch),
          "malformed context reaches native terminal proof");
    check(malformedContexts.CommitFreshContexts() == CudaParameterStatus::InvalidArgument &&
          !malformedContexts.HasUnprovenWork() && malformedContexts.Fields().empty() &&
          malformedContexts.BeginFreshPrograms(stream) == CudaParameterStatus::InvalidArgument,
          "semantic malformed context stops programs without unproven work");
    uint32_t validOffsets[] = {0, 2, 5};
    check(cudaMemcpy(const_cast<uint32_t*>(curveSource.view().curveOffsets.data), validOffsets,
                     sizeof(validOffsets), cudaMemcpyHostToDevice) == cudaSuccess,
          "restore source offsets after staged context failure");

    // Runtime expression values are validated on device. A terminally proven
    // invalid value leaves the old fields untouched and does not require
    // quarantine.
    auto invalidGraph = graph;
    UsdGenExpressionDesc invalidExpression;
    invalidExpression.path = SdfPath("/FreshInvalid"); invalidExpression.source = "1 / 0";
    invalidExpression.outputs.push_back({TfToken("result"), TfToken("float"),
        {expr::ScalarType::Float32, 1, 1, 1, 1, false}});
    invalidGraph.expressions.push_back(invalidExpression);
    UsdGenNodeDesc invalidNode;
    UsdGenExpressionBinding invalidBinding;
    invalidBinding.expression = invalidExpression.path; invalidBinding.destination = TfToken("invalidFresh");
    invalidBinding.destinationShape = {expr::ScalarType::Float32, 1, 1, 1, 1, false};
    invalidBinding.literal = VtValue(0.f);
    invalidNode.expressionBindings.push_back(invalidBinding);
    std::shared_ptr<const CudaParameterProgram> invalidProgram;
    check(CudaParameterProgram::Compile(usdGenUsd::Lower(invalidGraph), usdGenUsd::Lower(invalidNode), &invalidProgram, &diagnostics) ==
              CudaParameterStatus::Ok,
          "compile runtime-invalid staged expression");
    CudaParameterEvaluator invalidValue;
    FreshCallbackStatus invalidContextBatch, invalidProgramBatch;
    check(invalidProgram &&
          invalidValue.BeginFreshContexts(invalidProgram, curveSource.view(), {}, stagedControls,
                                          stream, &diagnostics) == CudaParameterStatus::Ok &&
          invalidValue.EnqueueFreshContextStatus(stream) == CudaParameterStatus::Ok &&
          CompleteFreshBatch(stream, &invalidContextBatch) &&
          invalidValue.CommitFreshContexts() == CudaParameterStatus::Ok &&
          invalidValue.BeginFreshPrograms(stream) == CudaParameterStatus::Ok &&
          invalidValue.EnqueueFreshProgramStatus(stream) == CudaParameterStatus::Ok &&
          CompleteFreshBatch(stream, &invalidProgramBatch),
          "runtime-invalid staged program reaches terminal proof");
    check(invalidValue.CommitFreshPrograms() == CudaParameterStatus::InvalidValue &&
          !invalidValue.HasUnprovenWork() && invalidValue.Fields().empty(),
          "invalid staged expression value never publishes fields");

    // Invalid pre-submission and capture paths reject before an unsafe fresh
    // operation exists; starting a second fresh phase is likewise rejected.
    CudaParameterEvaluator rejected;
    check(rejected.BeginFreshContexts({}, curveSource.view(), {}, stagedControls, stream, &diagnostics) ==
              CudaParameterStatus::InvalidArgument && !rejected.HasUnprovenWork(),
          "reject fresh pre-submit null program without unproven work");
    cudaStream_t captureStream = nullptr; cudaGraph_t captureGraph = nullptr;
    check(cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking) == cudaSuccess &&
          cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal) == cudaSuccess,
          "begin staged capture rejection");
    check(immutableProgram &&
          rejected.BeginFreshContexts(immutableProgram, curveSource.view(), {}, stagedControls,
                                      captureStream, &diagnostics) == CudaParameterStatus::InvalidArgument &&
          !rejected.HasUnprovenWork(),
          "capture rejects staged contexts before unsafe submission");
    check(cudaStreamEndCapture(captureStream, &captureGraph) == cudaSuccess,
          "end staged capture rejection");
    if (captureGraph) check(cudaGraphDestroy(captureGraph) == cudaSuccess, "destroy staged capture graph");
    check(cudaStreamDestroy(captureStream) == cudaSuccess, "destroy staged capture stream");

    auto addTypedExpression = [&](char const* path, char const* source,
                                  char const* nativeType, expr::ValueShape shape) {
        UsdGenExpressionDesc e; e.path = SdfPath(path); e.source = source;
        e.outputs.push_back({TfToken("result"), TfToken(nativeType), shape});
        graph.expressions.push_back(std::move(e));
    };
    addTypedExpression("/Half", "$value * 2", "half",
        {expr::ScalarType::Float16, 1, 1, 1, 1, false});
    addTypedExpression("/Double3", "$value * 2", "double3",
        {expr::ScalarType::Float64, 1, 3, 1, 1, false});
    addTypedExpression("/Float2", "$value * 2", "float2",
        {expr::ScalarType::Float32, 1, 2, 1, 1, false});
    UsdGenNodeDesc typedNode;
    UsdGenExpressionBinding halfBinding;
    halfBinding.expression = SdfPath("/Half"); halfBinding.destination = TfToken("halfValue");
    halfBinding.nativeType = TfToken("half");
    halfBinding.destinationShape = {expr::ScalarType::Float16, 1, 1, 1, 1, false};
    halfBinding.literal = VtValue(GfHalf(.75f));
    typedNode.expressionBindings.push_back(halfBinding);
    UsdGenExpressionBinding doubleBinding;
    doubleBinding.expression = SdfPath("/Double3"); doubleBinding.destination = TfToken("doubleValue");
    doubleBinding.nativeType = TfToken("double3");
    doubleBinding.destinationShape = {expr::ScalarType::Float64, 1, 3, 1, 1, false};
    doubleBinding.literal = VtValue(GfVec3d(1.25, -2.5, 4.0));
    typedNode.expressionBindings.push_back(doubleBinding);
    UsdGenExpressionBinding floatBinding;
    floatBinding.expression = SdfPath("/Float2"); floatBinding.destination = TfToken("floatValue");
    floatBinding.nativeType = TfToken("float2");
    floatBinding.destinationShape = {expr::ScalarType::Float32, 1, 2, 1, 1, false};
    floatBinding.literal = VtValue(GfVec2f(2.f, -3.f));
    typedNode.expressionBindings.push_back(floatBinding);
    CudaParameterPlan typedPlan;
    check(CudaParameterPlan::Compile(usdGenUsd::Lower(graph), usdGenUsd::Lower(typedNode), &typedPlan, &diagnostics) == CudaParameterStatus::Ok,
          "compile half and vector literal plan");
    if (typedPlan.Bindings().size() == 3 &&
        typedPlan.Evaluate(curveSource.view(), {}, {}, stream, &diagnostics) == CudaParameterStatus::Ok) {
        auto const* halfField = typedPlan.Find(TfToken("halfValue"));
        auto const* doubleField = typedPlan.Find(TfToken("doubleValue"));
        auto const* floatField = typedPlan.Find(TfToken("floatValue"));
        check(halfField && doubleField && floatField, "publish typed literal fields");
        if (halfField && doubleField && floatField) {
            __half halfResult{}; double doubleResult[3]{}; float floatResult[2]{};
            check(cudaMemcpy(&halfResult, halfField->data, sizeof(halfResult), cudaMemcpyDeviceToHost) == cudaSuccess,
                  "read half result");
            check(cudaMemcpy(doubleResult, doubleField->data, sizeof(doubleResult), cudaMemcpyDeviceToHost) == cudaSuccess,
                  "read double3 result");
            check(cudaMemcpy(floatResult, floatField->data, sizeof(floatResult), cudaMemcpyDeviceToHost) == cudaSuccess,
                  "read float2 result");
            check(std::fabs(__half2float(halfResult) - 1.5f) < 1e-3f, "half literal precision");
            check(std::fabs(doubleResult[0] - 2.5) < 1e-12 &&
                  std::fabs(doubleResult[1] + 5.0) < 1e-12 &&
                  std::fabs(doubleResult[2] - 8.0) < 1e-12, "double3 literal precision");
            check(std::fabs(floatResult[0] - 4.f) < 1e-5f &&
                  std::fabs(floatResult[1] + 6.f) < 1e-5f, "float2 literal precision");
        }
    } else {
        check(false, "evaluate half and vector literal plan");
    }

    // Source structural controls are read only after their evaluator's
    // program-status proof.  The helper owns compact pinned D2H staging but
    // never installs that proof callback itself.
    std::array<expr::ScalarType, CudaGroomScalarReadback::Capacity> scalarTypes{{
        expr::ScalarType::Bool, expr::ScalarType::Float16,
        expr::ScalarType::Int32, expr::ScalarType::UInt32,
        expr::ScalarType::Float32, expr::ScalarType::Int64,
        expr::ScalarType::UInt64, expr::ScalarType::Float64}};
    std::array<std::array<unsigned char, 8>, CudaGroomScalarReadback::Capacity> scalarBytes{};
    uint8_t scalarBool = 1; uint16_t scalarHalf = 0x3e00; int32_t scalarInt = -7;
    uint32_t scalarUInt = 9; float scalarFloat = 3.25f; int64_t scalarInt64 = -19;
    uint64_t scalarUInt64 = 23; double scalarDouble = 6.5;
    auto setScalarBytes = [&](size_t index, auto value) {
        std::memcpy(scalarBytes[index].data(), &value, sizeof(value));
    };
    setScalarBytes(0, scalarBool); setScalarBytes(1, scalarHalf);
    setScalarBytes(2, scalarInt); setScalarBytes(3, scalarUInt);
    setScalarBytes(4, scalarFloat); setScalarBytes(5, scalarInt64);
    setScalarBytes(6, scalarUInt64); setScalarBytes(7, scalarDouble);
    std::array<gpu::DeviceBuffer<unsigned char>, CudaGroomScalarReadback::Capacity> scalarDevice;
    std::vector<CudaParameterField> scalarFields;
    std::vector<CudaGroomScalarRequest> scalarRequests;
    for (size_t i = 0; i != scalarDevice.size(); ++i) {
        check(scalarDevice[i].reset(8) == cudaSuccess &&
              cudaMemcpyAsync(scalarDevice[i].data(), scalarBytes[i].data(), 8,
                  cudaMemcpyHostToDevice, stream) == cudaSuccess,
              "allocate/upload scalar readback input");
        std::string const destinationName = "control" + std::to_string(i);
        TfToken const destination(destinationName);
        scalarFields.push_back({destination, scalarDevice[i].data(), 1, scalarTypes[i], 1,
                                expr::Domain::Groom});
        scalarRequests.push_back({destination, scalarTypes[i]});
    }
    CudaGroomScalarReadback scalarReadback;
    FreshCallbackStatus scalarProof;
    check(scalarReadback.BeginFresh(scalarFields, scalarRequests, stream) == CudaParameterStatus::Ok &&
          scalarReadback.HasUnprovenWork() && CompleteFreshBatch(stream, &scalarProof) &&
          scalarReadback.CommitFreshFinish() == CudaParameterStatus::Ok &&
          !scalarReadback.HasUnprovenWork(),
          "native proof commits bounded groom scalar readback");

    // The pinned packet is fixed at Capacity * 8 bytes. An explicit job
    // reservation transfers that exact charge from Pending to Scratch
    // without changing total usage, then returns it to Pending when the
    // readback owner is destroyed. This isolated pool proof is independent
    // of the process-wide CUDA pool and physical VRAM state.
    {
        UsdGenExecutionResourcePool exactPool(64);
        auto exactReservation = exactPool.TryReserveMemory(64);
        check(exactReservation && exactReservation->RemainingBytes() == 64,
              "obtain exact scalar readback reservation");
        auto const exactBefore = exactPool.Snapshot();
        {
            CudaGroomScalarReadback exactReadback;
            FreshCallbackStatus exactProof;
            check(exactReadback.BeginFresh({scalarFields.front()}, {scalarRequests.front()},
                                           stream, &*exactReservation) == CudaParameterStatus::Ok &&
                      exactReservation->RemainingBytes() == 0 &&
                      CompleteFreshBatch(stream, &exactProof) &&
                      exactReadback.CommitFreshFinish() == CudaParameterStatus::Ok &&
                      exactReadback.Find(scalarRequests.front().destination),
                  "exact scalar readback reservation transfers and commits");
            auto const exactHeld = exactPool.Snapshot();
            check(exactHeld.usedBytes == exactBefore.usedBytes &&
                      exactHeld.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Scratch)] == 64 &&
                      exactHeld.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] == 0,
                  "exact scalar readback reservation preserves total accounting");
        }
        auto const exactReturned = exactPool.Snapshot();
        check(exactReservation->RemainingBytes() == 64 &&
                  exactReturned.usedBytes == exactBefore.usedBytes &&
                  exactReturned.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] == 64,
              "scalar readback destruction returns charge to live reservation");
        exactReservation->Release();
        auto const exactReleased = exactPool.Snapshot();
        check(exactReleased.usedBytes == 0 &&
                  exactReleased.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] == 0,
              "exact scalar readback reservation releases without a leak");
    }
    {
        UsdGenExecutionResourcePool shortPool(63);
        auto shortReservation = shortPool.TryReserveMemory(63);
        check(shortReservation && shortReservation->RemainingBytes() == 63,
              "obtain one-byte-short scalar readback reservation");
        auto const shortBefore = shortPool.Snapshot();
        CudaGroomScalarReadback shortReadback;
        check(shortReadback.BeginFresh({scalarFields.front()}, {scalarRequests.front()},
                                       stream, &*shortReservation) == CudaParameterStatus::CudaError &&
                  !shortReadback.HasUnprovenWork() &&
                  shortReservation->RemainingBytes() == 63,
              "one-byte-short scalar readback reservation rejects before submission");
        auto const shortAfter = shortPool.Snapshot();
        check(shortAfter.usedBytes == shortBefore.usedBytes &&
                  shortAfter.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] ==
                      shortBefore.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] &&
                  shortAfter.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Scratch)] ==
                      shortBefore.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Scratch)],
              "one-byte-short scalar readback leaves pool categories unchanged");
        shortReservation->Release();
        check(shortPool.Snapshot().usedBytes == 0,
              "one-byte-short scalar readback reservation releases cleanly");
    }
    for (size_t i = 0; i != scalarRequests.size(); ++i) {
        auto const* value = scalarReadback.Find(scalarRequests[i].destination);
        size_t const bytes = scalarTypes[i] == expr::ScalarType::Bool ||
            scalarTypes[i] == expr::ScalarType::Float16 ? (scalarTypes[i] == expr::ScalarType::Bool ? 1 : 2) :
            scalarTypes[i] == expr::ScalarType::Int32 || scalarTypes[i] == expr::ScalarType::UInt32 ||
            scalarTypes[i] == expr::ScalarType::Float32 ? 4 : 8;
        check(value && value->type == scalarTypes[i] &&
              std::memcmp(value->bytes.data(), scalarBytes[i].data(), bytes) == 0,
              "read back native groom scalar type");
    }

    CudaGroomScalarValue retainedScalar{};
    if (auto const* value = scalarReadback.Find(scalarRequests[4].destination))
        retainedScalar = *value;
    else
        check(false, "scalar value exists before preserving it across rejection");
    auto badFields = scalarFields;
    badFields[0].domain = expr::Domain::Primitive;
    check(scalarReadback.BeginFresh(badFields, scalarRequests, stream) == CudaParameterStatus::InvalidArgument &&
          scalarReadback.Find(scalarRequests[4].destination) &&
          std::memcmp(scalarReadback.Find(scalarRequests[4].destination)->bytes.data(),
                      retainedScalar.bytes.data(), retainedScalar.bytes.size()) == 0,
          "wrong domain preserves committed scalar values");
    badFields = scalarFields; badFields[0].components = 2;
    check(scalarReadback.BeginFresh(badFields, scalarRequests, stream) == CudaParameterStatus::InvalidArgument,
          "wrong scalar component shape rejected");
    badFields = scalarFields; badFields[0].count = 2;
    check(scalarReadback.BeginFresh(badFields, scalarRequests, stream) == CudaParameterStatus::InvalidArgument,
          "wrong scalar count shape rejected");
    badFields = scalarFields; badFields[0].type = expr::ScalarType::Float32;
    check(scalarReadback.BeginFresh(badFields, scalarRequests, stream) == CudaParameterStatus::InvalidArgument,
          "wrong scalar type rejected");
    badFields = scalarFields; badFields[0].data = nullptr;
    check(scalarReadback.BeginFresh(badFields, scalarRequests, stream) == CudaParameterStatus::InvalidArgument,
          "null scalar data rejected");
    badFields = scalarFields; badFields[0].data = scalarBytes[0].data();
    check(scalarReadback.BeginFresh(badFields, scalarRequests, stream) == CudaParameterStatus::InvalidArgument,
          "host scalar provenance rejected");
    auto duplicateRequests = scalarRequests;
    duplicateRequests.push_back(scalarRequests.front());
    check(scalarReadback.BeginFresh(scalarFields, duplicateRequests, stream) == CudaParameterStatus::InvalidArgument,
          "duplicate scalar destination rejected");

    std::vector<CudaGroomScalarRequest> duplicateMissing{{TfToken("missingControl"), expr::ScalarType::Bool},
                                                           {TfToken("usdGen:missingControl"), expr::ScalarType::Bool}};
    check(scalarReadback.BeginFresh(scalarFields, duplicateMissing, stream) == CudaParameterStatus::InvalidArgument,
          "duplicate missing scalar destination rejected");
    auto duplicateFields = scalarFields;
    duplicateFields.push_back(scalarFields.front());
    check(scalarReadback.BeginFresh(duplicateFields, scalarRequests, stream) == CudaParameterStatus::InvalidArgument,
          "ambiguous published scalar field rejected");

    FreshCallbackStatus missingProof;
    std::vector<CudaGroomScalarRequest> missingRequest{
        {TfToken("missingFirst"), expr::ScalarType::Bool},
        scalarRequests[0],
        {TfToken("missingMiddle"), expr::ScalarType::Int32},
        scalarRequests[1]};
    check(scalarReadback.BeginFresh(scalarFields, missingRequest, stream) == CudaParameterStatus::Ok &&
          scalarReadback.HasUnprovenWork() && CompleteFreshBatch(stream, &missingProof) &&
          scalarReadback.CommitFreshFinish() == CudaParameterStatus::Ok &&
          !scalarReadback.HasUnprovenWork() &&
          !scalarReadback.Find(TfToken("missingFirst")) &&
          !scalarReadback.Find(TfToken("missingMiddle")) &&
          scalarReadback.Find(scalarRequests[0].destination) &&
          scalarReadback.Find(scalarRequests[1].destination) &&
          std::memcmp(scalarReadback.Find(scalarRequests[0].destination)->bytes.data(),
                      scalarBytes[0].data(), 1) == 0 &&
          std::memcmp(scalarReadback.Find(scalarRequests[1].destination)->bytes.data(),
                      scalarBytes[1].data(), 2) == 0 &&
          !scalarReadback.Find(scalarRequests[2].destination),
          "missing first/middle controls compact packet and replace old overrides");
    FreshCallbackStatus scalarRetryProof;
    check(scalarReadback.BeginFresh(scalarFields, scalarRequests, stream) == CudaParameterStatus::Ok &&
          CompleteFreshBatch(stream, &scalarRetryProof) &&
          scalarReadback.CommitFreshFinish() == CudaParameterStatus::Ok &&
          scalarReadback.Find(scalarRequests[0].destination),
          "proved scalar readback reuses pinned staging");

    {
        CudaGroomScalarReadback allocationFailure;
        failNextCudaGroomScalarReadbackAllocationForTesting();
        check(allocationFailure.BeginFresh(scalarFields, scalarRequests, stream) == CudaParameterStatus::CudaError &&
              !allocationFailure.HasUnprovenWork(),
              "pre-copy scalar allocation failure has no speculative work");
        FreshCallbackStatus allocationRetryProof;
        check(allocationFailure.BeginFresh(scalarFields, scalarRequests, stream) == CudaParameterStatus::Ok &&
              CompleteFreshBatch(stream, &allocationRetryProof) &&
              allocationFailure.CommitFreshFinish() == CudaParameterStatus::Ok &&
              allocationFailure.Find(scalarRequests[0].destination),
              "pre-copy scalar allocation failure retries cleanly");
    }

    {
        CudaGroomScalarReadback failingReadback;
        std::vector<CudaGroomScalarRequest> twoRequests{scalarRequests[0], scalarRequests[1]};
        FreshCallbackStatus initialProof;
        check(failingReadback.BeginFresh(scalarFields, twoRequests, stream) == CudaParameterStatus::Ok &&
              CompleteFreshBatch(stream, &initialProof) &&
              failingReadback.CommitFreshFinish() == CudaParameterStatus::Ok,
              "establish scalar readback value before partial-copy failure");
        CudaGroomScalarValue retainedAfterFailure{};
        if (auto const* value = failingReadback.Find(twoRequests[0].destination))
            retainedAfterFailure = *value;
        else
            check(false, "scalar value exists before partial-copy failure");
        failNextCudaGroomScalarReadbackCopyAfterOneForTesting();
        check(failingReadback.BeginFresh(scalarFields, twoRequests, stream) == CudaParameterStatus::CudaError &&
              failingReadback.HasUnprovenWork() &&
              failingReadback.CommitFreshFinish() == CudaParameterStatus::CudaError &&
              failingReadback.Find(twoRequests[0].destination) &&
              std::memcmp(failingReadback.Find(twoRequests[0].destination)->bytes.data(),
                          retainedAfterFailure.bytes.data(), retainedAfterFailure.bytes.size()) == 0 &&
              failingReadback.BeginFresh(scalarFields, twoRequests, stream) == CudaParameterStatus::InvalidArgument,
              "partial scalar copy fails closed without publishing or reuse");
        check(cudaStreamSynchronize(stream) == cudaSuccess,
              "finish accepted partial scalar copy before quarantined teardown");
    }

    cudaStream_t scalarCaptureStream = nullptr; cudaGraph_t scalarCaptureGraph = nullptr;
    check(cudaStreamCreateWithFlags(&scalarCaptureStream, cudaStreamNonBlocking) == cudaSuccess &&
          cudaStreamBeginCapture(scalarCaptureStream, cudaStreamCaptureModeGlobal) == cudaSuccess &&
          scalarReadback.BeginFresh(scalarFields, scalarRequests, scalarCaptureStream) == CudaParameterStatus::InvalidArgument &&
          !scalarReadback.HasUnprovenWork() &&
          cudaStreamEndCapture(scalarCaptureStream, &scalarCaptureGraph) == cudaSuccess,
          "scalar readback capture rejects before submission");
    if (scalarCaptureGraph) check(cudaGraphDestroy(scalarCaptureGraph) == cudaSuccess,
                                  "destroy scalar readback capture graph");
    check(cudaStreamDestroy(scalarCaptureStream) == cudaSuccess,
          "destroy scalar readback capture stream");
    cudaStreamDestroy(stream);
    return failures;
}
