#include "usdGenExpr/cuda/expressionContext.h"
#include "usdGenExpr/expressions/frontend.h"

#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace usdGen;
using namespace usdGen::gpu;

namespace {
struct AsyncStatus {
    std::atomic<int> calls{0};
    std::atomic<cudaError_t> status{cudaErrorUnknown};
};
void CUDART_CB FreshDone(cudaStream_t, cudaError_t status, void* userdata) {
    auto* result = static_cast<AsyncStatus*>(userdata);
    result->status.store(status, std::memory_order_release);
    result->calls.fetch_add(1, std::memory_order_release);
}
} // namespace

int main() {
    cudaStream_t stream = nullptr, consumer = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess || cudaStreamCreate(&consumer) != cudaSuccess) return 2;
    int failures = 0;
    auto check = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; } };

    DeviceBuffer<float3> points, rest;
    DeviceBuffer<float> widths, hairT;
    DeviceBuffer<uint32_t> offsets;
    DeviceBuffer<uint64_t> ids;
    check(points.reset(5) == cudaSuccess && rest.reset(5) == cudaSuccess && widths.reset(5) == cudaSuccess &&
          hairT.reset(5) == cudaSuccess && offsets.reset(3) == cudaSuccess && ids.reset(2) == cudaSuccess,
          "allocate geometry channels");
    float3 p[] = {{1,2,3},{2,3,4},{10,20,30},{10,20,32},{10,24,32}};
    float3 r[] = {{0,1,2},{1,2,3},{9,19,29},{10,20,30},{11,21,31}};
    float w[] = {1,2,3,4,5}, t[] = {0,1,0,.5,1};
    uint32_t o[] = {0,2,5}; uint64_t stable[] = {7, (uint64_t(1)<<40) + 9};
    check(cudaMemcpy(points.data(), p, sizeof(p), cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(rest.data(), r, sizeof(r), cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(widths.data(), w, sizeof(w), cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(hairT.data(), t, sizeof(t), cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(offsets.data(), o, sizeof(o), cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(ids.data(), stable, sizeof(stable), cudaMemcpyHostToDevice) == cudaSuccess,
          "upload geometry channels");

    DeviceCurveGeometryView geometry{{points.data(),5},{rest.data(),5},{widths.data(),5},
                                      {offsets.data(),3},{ids.data(),2},2,5};
    CudaExpressionContext context;
    check(context.Inputs().count == 0 && context.Inputs().context.domain == expr::Domain::None,
          "unbuilt context is not a groom invocation");
    expr::Context controls; controls.domain = expr::Domain::Point; controls.frame = 12; controls.time = 0.5;
    check(context.Build(geometry, {{hairT.data(),5},{}}, controls, stream) == ExpressionContextStatus::Ok,
          "build point context on device");
    check(context.Finish(stream) == ExpressionContextStatus::Ok,
          "publish validated point context");
    check(context.Inputs().fields[unsigned(expr::Variable::N)].data == nullptr &&
          context.Inputs().fields[unsigned(expr::Variable::U)].data == nullptr,
          "missing surface channels remain unset");
    check(context.Wait(consumer) == ExpressionContextStatus::Ok, "context cross-stream wait");
    auto program = expr::Frontend::Compile("$P[0] + $P[1] + $t + $pointIndex + $idLo",
        {expr::Domain::Point, expr::ScalarType::Float32, 1});
    check(program.ok, "compile point context expression");
    DeviceBuffer<float> output;
    check(output.reset(5) == cudaSuccess, "allocate output");
    if (program.ok) {
        CudaExpressionProgram evaluator;
        check(evaluator.Upload(program.program.IR(), consumer) == ExpressionStatus::Ok, "upload context expression");
        check(evaluator.Evaluate(context.Inputs(), {output.data(),5,expr::ScalarType::Float32}, consumer) == ExpressionStatus::Ok,
              "evaluate context expression");
        check(evaluator.Finish(stream) == ExpressionStatus::Ok, "finish context expression");
        float actual[5]{};
        check(cudaMemcpy(actual, output.data(), sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess, "read context result");
        for (unsigned i = 0; i < 5; ++i) {
            unsigned c = i < 2 ? 0 : 1; unsigned root = o[c];
            double expected = p[i].x + p[i].y + t[i] + (i-root) + double(uint32_t(stable[c]));
            check(std::fabs(actual[i] - expected) < 1e-5, "point/root/stable-id values");
        }
    }
    uint32_t ownersHost[5]{};
    check(cudaMemcpy(ownersHost, context.Inputs().pointToPrimitive.data,
                     sizeof(ownersHost), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read complete owner map");
    check(ownersHost[0] == 0 && ownersHost[1] == 0 && ownersHost[2] == 1 &&
          ownersHost[3] == 1 && ownersHost[4] == 1, "ragged owner map is complete");

    CudaExpressionContext fallback;
    check(fallback.Build(geometry, {}, controls, stream) == ExpressionContextStatus::Ok,
          "build arc-length fallback context");
    check(fallback.Finish(stream) == ExpressionContextStatus::Ok, "validate fallback context");
    auto tProgram = expr::Frontend::Compile("$t", {expr::Domain::Point, expr::ScalarType::Float32, 1});
    check(tProgram.ok, "compile fallback t expression");
    if (tProgram.ok) {
        CudaExpressionProgram evaluator;
        check(evaluator.Upload(tProgram.program.IR(), stream) == ExpressionStatus::Ok, "upload fallback t expression");
        check(evaluator.Evaluate(fallback.Inputs(), {output.data(),5,expr::ScalarType::Float32}, stream) == ExpressionStatus::Ok,
              "evaluate arc-length fallback");
        check(evaluator.Finish(stream) == ExpressionStatus::Ok, "finish arc-length fallback");
        float actual[5]{};
        check(cudaMemcpy(actual, output.data(), sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess, "read fallback t");
        check(std::fabs(actual[0] - 0.0f) < 1e-6f && std::fabs(actual[1] - 1.0f) < 1e-6f,
              "first curve arc endpoints");
        check(std::fabs(actual[2] - 0.0f) < 1e-6f && std::fabs(actual[3] - 1.0f/3.0f) < 1e-5f &&
              std::fabs(actual[4] - 1.0f) < 1e-6f, "uneven curve arc-length parameter");
    }

    // Groom contexts intentionally carry no geometry fields and can be built
    // before strands exist; primitive contexts use one invocation per curve.
    CudaExpressionContext groom;
    expr::Context groomControls; groomControls.domain = expr::Domain::Groom; groomControls.frame = 3;
    check(groom.Build({}, {}, groomControls, stream) == ExpressionContextStatus::Ok,
          "build geometry-free groom context");
    check(groom.Finish(stream) == ExpressionContextStatus::Ok, "finish groom context");
    CudaExpressionContext primitive;
    expr::Context primitiveControls; primitiveControls.domain = expr::Domain::Primitive;
    check(primitive.Build(geometry, {}, primitiveControls, stream) == ExpressionContextStatus::Ok,
          "build primitive context");
    check(primitive.Finish(stream) == ExpressionContextStatus::Ok, "finish primitive context");
    check(primitive.Inputs().fields[unsigned(expr::Variable::PointIndex)].data == nullptr &&
          primitive.Inputs().fields[unsigned(expr::Variable::PointCount)].data == nullptr,
          "primitive context omits point-only fields");
    check(primitive.Build(geometry, {{nullptr, 1}, {}}, primitiveControls, stream) ==
              ExpressionContextStatus::InvalidArgument,
          "reject null optional channel with nonzero size");

    // Fresh contexts stage device fields but cannot publish an invocation
    // until the shared terminal callback has run and the host relay commits.
    CudaExpressionContext fresh;
    AsyncStatus freshStatus;
    check(fresh.BuildFresh(geometry, {{hairT.data(),5},{}}, controls, stream) ==
              ExpressionContextStatus::Ok && fresh.HasUnprovenWork(),
          "fresh build submits unproven context work");
    check(fresh.Inputs().count == 0 && fresh.Inputs().context.domain == expr::Domain::None,
          "fresh inputs remain unpublished before terminal proof");
    check(fresh.Build(geometry, {{hairT.data(),5},{}}, controls, stream) ==
              ExpressionContextStatus::InvalidArgument &&
          fresh.Finish(stream) == ExpressionContextStatus::InvalidArgument,
          "legacy entrypoints cannot consume a fresh pending context");
    check(fresh.EnqueueFreshStatus(stream) == ExpressionContextStatus::Ok,
          "enqueue fresh context status before shared callback");
    check(fresh.EnqueueFreshStatus(stream) == ExpressionContextStatus::InvalidArgument,
          "reject duplicate fresh status copy");
    check(fresh.CommitFreshFinish() == ExpressionContextStatus::InvalidArgument,
          "status sentinel rejects commit before terminal callback");
    check(cudaStreamAddCallback(stream, FreshDone, &freshStatus, 0) == cudaSuccess,
          "install batch terminal callback after context status");
    check(cudaStreamSynchronize(stream) == cudaSuccess &&
          freshStatus.calls.load(std::memory_order_acquire) == 1 &&
          freshStatus.status.load(std::memory_order_acquire) == cudaSuccess,
          "fresh terminal callback completes exactly once");
    check(fresh.CommitFreshFinish() == ExpressionContextStatus::Ok && !fresh.HasUnprovenWork() &&
          fresh.Inputs().count == 5 && fresh.Inputs().context.domain == expr::Domain::Point,
          "fresh host commit publishes only proven inputs");

    cudaStream_t captureStream = nullptr;
    cudaGraph_t graph = nullptr;
    check(cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking) == cudaSuccess &&
          cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal) == cudaSuccess,
          "begin fresh context capture rejection test");
    CudaExpressionContext captured;
    check(captured.BuildFresh(geometry, {{hairT.data(),5},{}}, controls, captureStream) ==
              ExpressionContextStatus::InvalidArgument && !captured.HasUnprovenWork(),
          "fresh capture rejects before any allocation or submission");
    check(cudaStreamEndCapture(captureStream, &graph) == cudaSuccess,
          "end fresh context capture rejection test");
    if (graph) check(cudaGraphDestroy(graph) == cudaSuccess, "destroy empty capture graph");
    check(cudaStreamDestroy(captureStream) == cudaSuccess, "destroy capture stream");

    // Fresh semantic failures carry a guarded status through the same shared
    // callback path. Invalid offsets must not reach dependent field kernels.
    o[1] = UINT32_MAX;
    check(cudaMemcpy(offsets.data(), o, sizeof(o), cudaMemcpyHostToDevice) == cudaSuccess,
          "upload fresh malformed offsets");
    CudaExpressionContext malformedFresh;
    AsyncStatus malformedStatus;
    check(malformedFresh.BuildFresh(geometry, {{hairT.data(),5},{}}, controls, stream) ==
              ExpressionContextStatus::Ok &&
          malformedFresh.EnqueueFreshStatus(stream) == ExpressionContextStatus::Ok &&
          cudaStreamAddCallback(stream, FreshDone, &malformedStatus, 0) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess &&
          malformedStatus.status.load(std::memory_order_acquire) == cudaSuccess,
          "fresh malformed offsets complete guarded validation");
    check(malformedFresh.CommitFreshFinish() == ExpressionContextStatus::InvalidGeometry &&
          malformedFresh.Inputs().count == 0 && !malformedFresh.HasUnprovenWork(),
          "fresh malformed offsets never publish fields");
    o[1] = 2;
    check(cudaMemcpy(offsets.data(), o, sizeof(o), cudaMemcpyHostToDevice) == cudaSuccess,
          "restore offsets after fresh malformed validation");

    float nanT[] = {0, 1, 0, std::numeric_limits<float>::quiet_NaN(), 1};
    check(cudaMemcpy(hairT.data(), nanT, sizeof(nanT), cudaMemcpyHostToDevice) == cudaSuccess,
          "upload fresh nonfinite channel");
    CudaExpressionContext invalidChannelFresh;
    AsyncStatus invalidChannelStatus;
    check(invalidChannelFresh.BuildFresh(geometry, {{hairT.data(),5},{}}, controls, stream) ==
              ExpressionContextStatus::Ok &&
          invalidChannelFresh.EnqueueFreshStatus(stream) == ExpressionContextStatus::Ok &&
          cudaStreamAddCallback(stream, FreshDone, &invalidChannelStatus, 0) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess,
          "fresh nonfinite channel reaches terminal callback");
    check(invalidChannelFresh.CommitFreshFinish() == ExpressionContextStatus::InvalidChannel &&
          invalidChannelFresh.Inputs().count == 0 && !invalidChannelFresh.HasUnprovenWork(),
          "fresh invalid channel remains unpublished");
    check(cudaMemcpy(hairT.data(), t, sizeof(t), cudaMemcpyHostToDevice) == cudaSuccess,
          "restore valid channel after fresh semantic failure");

    // Invalid offsets are rejected after the device validation pass, before
    // any offset-derived field can be consumed.
    o[2] = 4;
    check(cudaMemcpy(offsets.data(), o, sizeof(o), cudaMemcpyHostToDevice) == cudaSuccess, "upload malformed offsets");
    check(context.Build(geometry, {{hairT.data(),5},{}}, controls, stream) == ExpressionContextStatus::Ok,
          "queue malformed geometry validation");
    check(context.Inputs().count == 0 && context.Inputs().context.domain == expr::Domain::None,
          "invalid pending context is not publishable");
    check(context.Finish(stream) == ExpressionContextStatus::InvalidGeometry, "reject malformed offsets");
    check(context.Inputs().count == 0 && context.Inputs().context.domain == expr::Domain::None,
          "rejected context remains unusable");
    o[2] = UINT32_MAX;
    check(cudaMemcpy(offsets.data(), o, sizeof(o), cudaMemcpyHostToDevice) == cudaSuccess,
          "upload extreme malformed offsets");
    check(context.Build(geometry, {{hairT.data(),5},{}}, controls, stream) == ExpressionContextStatus::Ok,
          "queue extreme malformed validation");
    check(context.Finish(stream) == ExpressionContextStatus::InvalidGeometry,
          "extreme offset never indexes hairT outside allocation");

    o[2] = 5;
    check(cudaMemcpy(offsets.data(), o, sizeof(o), cudaMemcpyHostToDevice) == cudaSuccess,
          "restore valid offsets");
    float badT[] = {0, .5, .1, .5, 1};
    check(cudaMemcpy(hairT.data(), badT, sizeof(badT), cudaMemcpyHostToDevice) == cudaSuccess,
          "upload non-normalized hairT");
    check(context.Build(DeviceCurveGeometryView{{points.data(),5},{rest.data(),5},{widths.data(),5},
                                                 {offsets.data(),3},{ids.data(),2},2,5},
                         {{hairT.data(),5},{}}, controls, stream) == ExpressionContextStatus::Ok,
          "queue hairT endpoint validation");
    check(context.Finish(stream) == ExpressionContextStatus::InvalidChannel,
          "reject hairT without root-tip endpoints");

    // Empty primitive/point domains are valid zero-invocation contexts, not
    // a groom invocation and not malformed geometry. Keep the offset sentinel.
    DeviceCurveGeometryView emptyGeometry;
    emptyGeometry.curveOffsets = {offsets.data(), 1};
    for (auto domain : {expr::Domain::Primitive, expr::Domain::Point}) {
        auto emptyControls = controls;
        emptyControls.domain = domain;
        check(context.Build(emptyGeometry, {}, emptyControls, stream) == ExpressionContextStatus::Ok,
              "queue empty-domain context");
        check(context.Finish(stream) == ExpressionContextStatus::Ok,
              "validate empty-domain context");
        check(context.Inputs().count == 0 && context.Inputs().context.domain == domain,
              "empty domain has zero invocations and retains declared rate");
    }
    uint32_t badSentinel = 1;
    check(cudaMemcpy(offsets.data(), &badSentinel, sizeof(badSentinel), cudaMemcpyHostToDevice) == cudaSuccess,
          "upload invalid empty sentinel");
    check(context.Build(emptyGeometry, {}, controls, stream) == ExpressionContextStatus::Ok,
          "queue invalid empty sentinel");
    check(context.Finish(stream) == ExpressionContextStatus::InvalidGeometry,
          "reject invalid empty sentinel");

    cudaStreamDestroy(consumer); cudaStreamDestroy(stream);
    return failures;
}
