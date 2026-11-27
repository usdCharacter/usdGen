#include "usdGenExpr/cuda/expression.h"
#include "usdGenCurves/cuda/styleOps.h"
#include "usdGenExpr/expressions/frontend.h"
#include "api/usdGenSeExprOracle.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>

using namespace usdGen;
using namespace usdGen::gpu;

int main() {
    int failures = 0;
    auto check = [&](bool ok, char const* message) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
    };
    cudaStream_t producer = nullptr, consumer = nullptr;
    if (cudaStreamCreate(&producer) != cudaSuccess || cudaStreamCreate(&consumer) != cudaSuccess)
        return 2;
    {
        DeviceBuffer<double> literal, u, t;
        DeviceBuffer<uint32_t> owners;
        DeviceBuffer<float> output;
        check(literal.reset(1) == cudaSuccess && u.reset(2) == cudaSuccess &&
              t.reset(5) == cudaSuccess && owners.reset(5) == cudaSuccess &&
              output.reset(5) == cudaSuccess, "allocate test fields");
        if (failures) return failures;
        double literalHost = 0.1, uHost[]{0.2, 0.8}, tHost[]{0., 1., 0., .5, 1.};
        uint32_t ownersHost[]{0, 0, 1, 1, 1};
        check(cudaMemcpy(literal.data(), &literalHost, sizeof(literalHost), cudaMemcpyHostToDevice) == cudaSuccess,
              "upload literal test fixture");
        check(cudaMemcpy(u.data(), uHost, sizeof(uHost), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(t.data(), tHost, sizeof(tHost), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(owners.data(), ownersHost, sizeof(ownersHost), cudaMemcpyHostToDevice) == cudaSuccess,
              "upload domain test fixture");
        ExpressionInputs inputs;
        inputs.context.domain = expr::Domain::Point;
        inputs.context.frame = 2.0;
        inputs.count = 5; inputs.primitiveCount = 2;
        inputs.pointToPrimitive = {owners.data(), owners.size()};
        inputs.fields[unsigned(expr::Variable::Value)] = {literal.data(), 1, expr::Domain::Groom};
        inputs.fields[unsigned(expr::Variable::U)] = {u.data(), 2, expr::Domain::Primitive};
        inputs.fields[unsigned(expr::Variable::T)] = {t.data(), 5, expr::Domain::Point};
        struct Case { const char* source; float expected[5]; } cases[]{
            {"$value * (0.5 + 0.5 * $u)", {.06f, .06f, .09f, .09f, .09f}},
            {"$value * $t", {0.f, .1f, 0.f, .05f, .1f}},
            {"$value * (1 - 0.95 * $t)", {.1f, .005f, .1f, .0525f, .005f}},
            {"$value * (0.25 + 0.75 * $u)", {.04f, .04f, .085f, .085f, .085f}},
        };
        for (auto const& test : cases) {
            auto compiled = expr::Frontend::Compile(test.source,
                {expr::Domain::Point, expr::ScalarType::Float64, 1});
            check(compiled.ok, test.source);
            if (!compiled.ok) {
                for (auto const& error : compiled.diagnostics) std::fprintf(stderr, "  %s\n", error.c_str());
                continue;
            }
            CudaExpressionProgram program;
            check(program.Upload(compiled.program.IR(), producer) == ExpressionStatus::Ok, "upload checked IR");
            for (int frame = 0; frame != 2; ++frame) {
                inputs.context.frame = frame + 1;
                check(program.Evaluate(inputs, {output.data(), 5, expr::ScalarType::Float32}, producer) ==
                      ExpressionStatus::Ok, "execute without geometry readback");
                check(program.Finish(consumer) == ExpressionStatus::Ok, "cross-stream completion");
                float actual[5]{};
                check(cudaMemcpy(actual, output.data(), sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess,
                      "explicit test oracle readback");
                for (unsigned i = 0; i != 5; ++i)
                    check(std::fabs(actual[i] - test.expected[i]) < 1.e-6f, "domain-broadcast expression result");
            }
        }

        // Vector fields remain interleaved on device; a scalar variable
        // broadcasts while vector $value retains distinct components.
        DeviceBuffer<double> positions, vectorLiteral;
        DeviceBuffer<float> vectorOutput;
        check(positions.reset(15) == cudaSuccess && vectorLiteral.reset(3) == cudaSuccess &&
              vectorOutput.reset(15) == cudaSuccess, "allocate vector fixtures");
        double positionsHost[]{1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15};
        double vectorLiteralHost[]{2,3,4};
        check(cudaMemcpy(positions.data(), positionsHost, sizeof(positionsHost), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(vectorLiteral.data(), vectorLiteralHost, sizeof(vectorLiteralHost), cudaMemcpyHostToDevice) == cudaSuccess,
              "upload vector fixtures");
        ExpressionInputs vectorInputs = inputs;
        vectorInputs.fields[unsigned(expr::Variable::P)] = {positions.data(), 5, expr::Domain::Point, 3};
        vectorInputs.fields[unsigned(expr::Variable::Value)] = {vectorLiteral.data(), 1, expr::Domain::Groom, 3};
        const char* vectors[]{"[1,2,3]", "$P * $t", "$value * 2",
                              "[$P[0], $P[1], $P[2]]",
                              "$frame > 1 ? [1,2,3] : [4,5,6]",
                              "abs(-$P)", "sin($P) + cos($P)", "pow($value, 2)",
                              "clamp($P, 2, 8)", "min($P, $value) + max($P, $value)"};
        for (unsigned test = 0; test < sizeof(vectors)/sizeof(vectors[0]); ++test) {
            auto compiled = expr::Frontend::Compile(vectors[test],
                {expr::Domain::Point, expr::ScalarType::Float64, 3});
            check(compiled.ok, vectors[test]);
            if (!compiled.ok) {
                for (auto const& error : compiled.diagnostics) std::fprintf(stderr,"  %s\n",error.c_str());
                continue;
            }
            CudaExpressionProgram program;
            check(program.Upload(compiled.program.IR(), producer) == ExpressionStatus::Ok, "upload vector IR");
            for (int frame = 1; frame <= 2; ++frame) {
                vectorInputs.context.frame = frame;
                check(program.Evaluate(vectorInputs, {vectorOutput.data(), 5, expr::ScalarType::Float32, 3}, producer) ==
                      ExpressionStatus::Ok, "execute vector expression");
                check(program.Finish(consumer) == ExpressionStatus::Ok, "vector expression completion");
                float actual[15]{};
                check(cudaMemcpy(actual, vectorOutput.data(), sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess,
                      "vector test oracle readback");
                for (unsigned i = 0; i < 5; ++i) {
                    auto reference = usdGenTest::EvaluateSeExpr(vectors[test], 3, {
                        {"P", {positionsHost[i*3],positionsHost[i*3+1],positionsHost[i*3+2]}},
                        {"t", {tHost[i]}}, {"value", {2,3,4}}, {"frame", {double(frame)}}});
                    check(reference.ok && reference.values.size() == 3, "upstream SeExpr vector reference");
                    if (!reference.ok || reference.values.size() != 3) {
                        std::fprintf(stderr, "%s: %s\n", vectors[test], reference.diagnostic.c_str());
                        continue;
                    }
                    for (unsigned c = 0; c < 3; ++c) {
                        check(std::fabs(actual[i*3+c] - reference.values[c]) < 1.e-6,
                              "GPU numeric result matches actual SeExpr interpreter");
                        if (test < 5) {
                            const double expected = test == 0 ? c + 1 :
                                test == 1 ? positionsHost[i*3+c] * tHost[i] :
                                test == 2 ? vectorLiteralHost[c] * 2 :
                                test == 3 ? positionsHost[i*3+c] : c + (frame > 1 ? 1 : 4);
                            check(std::fabs(actual[i*3+c] - expected) < 1.e-6, "vector component numeric result");
                        }
                    }
                }
            }
        }

        // Consume an evaluated parameter directly in a CUDA style primitive.
        // The only result download occurs after both device computations.
        DeviceBuffer<uint32_t> curveOffsets;
        DeviceBuffer<float> styledWidths;
        check(curveOffsets.reset(3) == cudaSuccess && styledWidths.reset(5) == cudaSuccess,
              "allocate expression-to-operator fixture");
        const uint32_t offsetsHost[]{0,2,5};
        check(cudaMemcpy(curveOffsets.data(), offsetsHost, sizeof(offsetsHost), cudaMemcpyHostToDevice) == cudaSuccess,
              "upload operator topology fixture");
        auto widthExpression = expr::Frontend::Compile("$frame * $value * (1 - 0.95 * $t)",
            {expr::Domain::Point,expr::ScalarType::Float32,1});
        check(widthExpression.ok, "compile consuming-operator parameter");
        if (widthExpression.ok) {
            CudaExpressionProgram parameter;
            CudaStyleOps style;
            check(parameter.Upload(widthExpression.program.IR(), producer) == ExpressionStatus::Ok,
                  "compile-time parameter IR upload");
            DeviceCurveGeometryView geometry{{},{},{},{curveOffsets.data(),3},{},2,5};
            for (int frame=1; frame<=2; ++frame) {
                inputs.context.frame = frame;
                check(parameter.Evaluate(inputs, {output.data(),5,expr::ScalarType::Float32}, producer) ==
                      ExpressionStatus::Ok, "runtime parameter evaluation");
                check(parameter.Finish(consumer) == ExpressionStatus::Ok, "validate parameter scalar diagnostic");
                ScalarField field = ScalarField::Device({output.data(),5},expr::Domain::Point);
                check(style.WidthRamp(geometry,field,field,styledWidths.view(),consumer) == StyleStatus::Ok,
                      "operator consumes device parameter buffer directly");
                check(style.Finish(producer) == StyleStatus::Ok, "validate operator scalar diagnostic");
                float actual[5]{};
                check(cudaMemcpy(actual,styledWidths.data(),sizeof(actual),cudaMemcpyDeviceToHost) == cudaSuccess,
                      "final test-only expression/operator result download");
                for (unsigned i=0;i<5;++i)
                    check(std::fabs(actual[i] - frame*literalHost*(1-0.95*tHost[i])) < 1.e-6,
                          "frame-dependent device parameter reaches consuming operator");
            }
        }

        // Typed topology/control outputs are validated on device, not by
        // silently rounding a double or accepting arbitrary boolean values.
        DeviceBuffer<int> integer;
        DeviceBuffer<unsigned char> boolean;
        check(integer.reset(1) == cudaSuccess && boolean.reset(1) == cudaSuccess, "allocate control outputs");
        struct Control { const char* source; expr::ScalarType type; bool valid; } controls[]{
            {"4 * 2", expr::ScalarType::Int32, true},
            {"$frame >= 1", expr::ScalarType::Bool, true},
            {"!0", expr::ScalarType::Bool, true},
            {"!1", expr::ScalarType::Bool, true},
            {"0.5", expr::ScalarType::Int32, false},
            {"2", expr::ScalarType::Bool, false},
            {"1 / 0", expr::ScalarType::Float32, false},
            {"$frame > 0 ? 3 : 0 / 0", expr::ScalarType::Int32, true},
        };
        ExpressionInputs control;
        control.context.domain = expr::Domain::Groom; control.context.frame = 2;
        for (auto const& test : controls) {
            auto compiled = expr::Frontend::Compile(test.source,
                {expr::Domain::Groom, test.type, 1});
            check(compiled.ok, test.source);
            if (!compiled.ok) continue;
            CudaExpressionProgram program;
            check(program.Upload(compiled.program.IR(), producer) == ExpressionStatus::Ok, "upload control IR");
            void* destination = test.type == expr::ScalarType::Bool
                ? static_cast<void*>(boolean.data()) : static_cast<void*>(integer.data());
            check(program.Evaluate(control, {destination, 1, test.type}, producer) == ExpressionStatus::Ok,
                  "launch typed control expression");
            check(program.Finish(consumer) == (test.valid ? ExpressionStatus::Ok : ExpressionStatus::InvalidValue),
                  "strict numeric conversion and selected-branch validity");
            if (test.valid && test.type == expr::ScalarType::Int32) {
                int actual = 0;
                check(cudaMemcpy(&actual, integer.data(), sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess,
                      "control oracle readback");
                check(actual == (test.source[0] == '4' ? 8 : 3), "integer output value");
            }
            if (test.valid && test.type == expr::ScalarType::Bool) {
                unsigned char actual = 99;
                check(cudaMemcpy(&actual, boolean.data(), sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess,
                      "boolean oracle readback");
                check(actual == (std::string(test.source) == "!1" ? 0 : 1), "boolean output value");
            }
        }
        for (double value : {0.5,65504.0,70000.0}) {
            auto compiled = expr::Frontend::Compile(std::to_string(value),
                {expr::Domain::Groom, expr::ScalarType::Float16, 1});
            check(compiled.ok, "compile half output");
            if (!compiled.ok) continue;
            CudaExpressionProgram halfProgram;
            check(halfProgram.Upload(compiled.program.IR(), producer) == ExpressionStatus::Ok,
                  "upload half IR");
            check(halfProgram.Evaluate(control, {integer.data(),1,expr::ScalarType::Float16}, producer) ==
                  ExpressionStatus::Ok, "launch half conversion");
            check(halfProgram.Finish(consumer) == (value > 65504 ? ExpressionStatus::InvalidValue : ExpressionStatus::Ok),
                  "half conversion rejects overflow");
            if (value <= 65504) {
                __half actual;
                check(cudaMemcpy(&actual, integer.data(),sizeof(actual),cudaMemcpyDeviceToHost) == cudaSuccess,
                      "half oracle readback");
                check(__half2float(actual) == value, "half numeric value");
            }
        }
        expr::IRProgram malformed;
        malformed.instructions.push_back({expr::IROp::Add, 0, 0, 0});
        malformed.registerCount = 1;
        CudaExpressionProgram program;
        check(program.Upload(malformed, producer) == ExpressionStatus::InvalidProgram,
              "reject reads of uninitialized registers");
        auto constant = expr::Frontend::Compile("1", {expr::Domain::Groom, expr::ScalarType::Float64, 1});
        check(constant.ok, "malformed-IR baseline");
        if (constant.ok) {
            auto invalid = constant.program.IR();
            invalid.instructions.push_back(invalid.instructions[0]);
            check(program.Upload(invalid, producer) == ExpressionStatus::InvalidProgram,
                  "reject duplicate SSA destinations");
            invalid = constant.program.IR();
            invalid.outputCount = 2; invalid.valueComponents = 3;
            check(program.Upload(invalid, producer) == ExpressionStatus::InvalidProgram,
                  "reject inconsistent vector output shape");
            invalid.valueComponents = 2; invalid.output[1] = invalid.registerCount;
            check(program.Upload(invalid, producer) == ExpressionStatus::InvalidProgram,
                  "reject undefined output register");
        }
        auto mapped = expr::Frontend::Compile("$u", {expr::Domain::Point, expr::ScalarType::Float64, 1});
        check(mapped.ok, "mapped-variable baseline");
        if (mapped.ok) {
            check(program.Upload(mapped.program.IR(), producer) == ExpressionStatus::Ok,
                  "upload mapped-variable IR");
            auto invalid = inputs;
            invalid.pointToPrimitive = {};
            check(program.Evaluate(invalid, {output.data(),5,expr::ScalarType::Float32}, producer) ==
                  ExpressionStatus::InvalidArgument, "reject missing primitive ownership map");
            invalid.pointToPrimitive = {owners.data(),4};
            check(program.Evaluate(invalid, {output.data(),5,expr::ScalarType::Float32}, producer) ==
                  ExpressionStatus::InvalidArgument, "reject short primitive ownership map");
            invalid = inputs;
            invalid.fields[unsigned(expr::Variable::U)] = {};
            check(program.Evaluate(invalid, {output.data(),5,expr::ScalarType::Float32}, producer) ==
                  ExpressionStatus::Ok, "missing used variable is deferred device diagnostic");
            check(program.Finish(consumer) == ExpressionStatus::InvalidValue,
                  "missing used variable cannot publish a valid field");
            const uint32_t badOwner = 2;
            check(cudaMemcpy(owners.data(), &badOwner, sizeof(badOwner), cudaMemcpyHostToDevice) == cudaSuccess,
                  "upload invalid ownership fixture");
            check(program.Evaluate(inputs, {output.data(),5,expr::ScalarType::Float32}, producer) ==
                  ExpressionStatus::Ok, "launch bounds-checked ownership lookup");
            check(program.Finish(consumer) == ExpressionStatus::InvalidValue,
                  "out-of-range primitive owner fails without out-of-bounds read");
        }
    }

    // Fresh execution owns its IR through the terminal D2H status proof.  It
    // never permits a legacy synchronous Finish to consume mixed state, and
    // semantic device failure clears the proof requirement without publishing
    // a valid parameter field.
    {
        DeviceBuffer<float> freshOutput;
        check(freshOutput.reset(1) == cudaSuccess, "allocate fresh expression output");
        auto compiled = expr::Frontend::Compile("2", {expr::Domain::Groom,
            expr::ScalarType::Float64, 1});
        check(compiled.ok, "compile fresh expression");
        if (compiled.ok) {
            ExpressionInputs freshInputs;
            freshInputs.context.domain = expr::Domain::Groom;
            freshInputs.count = 1;
            CudaExpressionProgram fresh;

            cudaGraph_t graph = nullptr;
            check(cudaStreamBeginCapture(producer, cudaStreamCaptureModeGlobal) == cudaSuccess,
                  "begin fresh-expression capture");
            check(fresh.UploadFresh(compiled.program.IR(), producer) == ExpressionStatus::InvalidArgument,
                  "fresh upload rejects capture before allocation/query");
            check(cudaStreamEndCapture(producer, &graph) == cudaSuccess && graph,
                  "fresh capture remains usable after rejection");
            if (graph) cudaGraphDestroy(graph);

            auto ir = compiled.program.IR();
            check(fresh.UploadFresh(ir, producer) == ExpressionStatus::Ok,
                  "fresh upload owns immutable IR");
            ir.instructions.clear();
            check(fresh.HasUnprovenWork(), "fresh upload marks owned IR unproven");
            check(fresh.Evaluate(freshInputs, {freshOutput.data(), 1,
                  expr::ScalarType::Float32}, producer) == ExpressionStatus::InvalidArgument,
                  "legacy evaluate rejects fresh upload state");
            check(fresh.EvaluateFresh(freshInputs, {freshOutput.data(), 1,
                  expr::ScalarType::Float32}, producer) == ExpressionStatus::Ok,
                  "fresh evaluate queues after fresh upload");
            check(fresh.CommitFreshFinish() == ExpressionStatus::InvalidArgument,
                  "fresh commit rejects before status enqueue");
            check(fresh.EnqueueFreshStatus(producer) == ExpressionStatus::Ok,
                  "fresh status queues pinned D2H result");
            check(fresh.Finish(consumer) == ExpressionStatus::InvalidArgument,
                  "legacy finish rejects fresh pending state");
            check(fresh.CommitFreshFinish() == ExpressionStatus::InvalidArgument,
                  "fresh commit rejects speculative pre-terminal use");
            check(cudaStreamSynchronize(producer) == cudaSuccess,
                  "explicit test-only fresh terminal proof");
            check(fresh.CommitFreshFinish() == ExpressionStatus::Ok,
                  "fresh commit consumes successful terminal status");
            check(!fresh.HasUnprovenWork(), "fresh success clears proof retention");
            float actual = 0;
            check(cudaMemcpy(&actual, freshOutput.data(), sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess &&
                  actual == 2.0f, "fresh IR remains valid after caller destroys IR bytes");

            auto invalid = expr::Frontend::Compile("$frame / 0", {expr::Domain::Groom,
                expr::ScalarType::Float64, 1});
            check(invalid.ok, "compile invalid fresh expression");
            if (invalid.ok) {
                check(fresh.UploadFresh(invalid.program.IR(), producer) == ExpressionStatus::Ok,
                      "fresh upload permits a new proved instance");
                check(fresh.EvaluateFresh(freshInputs, {freshOutput.data(), 1,
                      expr::ScalarType::Float32}, producer) == ExpressionStatus::Ok,
                      "fresh invalid value still queues device status");
                check(fresh.EnqueueFreshStatus(producer) == ExpressionStatus::Ok,
                      "fresh invalid value queues status");
                check(cudaStreamSynchronize(producer) == cudaSuccess,
                      "explicit test-only invalid terminal proof");
                check(fresh.CommitFreshFinish() == ExpressionStatus::InvalidValue,
                      "fresh commit reports semantic device failure");
                check(!fresh.HasUnprovenWork(), "semantic status does not quarantine fresh storage");
            }
        }
    }
    cudaStreamDestroy(consumer); cudaStreamDestroy(producer);
    std::printf("testUsdGenCudaExpression: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
