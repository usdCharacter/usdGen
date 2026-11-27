// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// usdGen — unfused CPU oracle for the exprVk parity test.
//
// This translation unit MUST be compiled with contraction disabled
// (-ffp-contract=off -fno-fast-math): it re-instantiates the shared
// interpreter templates (expressions/irExec.h) with strict IEEE order so the
// Vulkan shader's `precise` evaluation has a bit-meaningful host twin. The
// library build of cpuEvaluator.cpp uses the toolchain default contraction,
// which fuses x+y*z inside noise/voronoi into FMA and drifts 1 ulp from any
// unfused lane. The admission below is a verbatim copy of
// expr::EvaluateProgram's (cpuEvaluator.cpp); the semantics still have one
// definition in irExec.h, this file only re-hosts the per-element loop.
#include "usdGenExpr/expressions/cpuEvaluator.h"
#include "usdGenExpr/expressions/irExec.h"

namespace usdGen::expr {

CpuExpressionStatus EvaluateProgramUnfused(IRProgram const &program,
                                           CpuExpressionInputs const &inputs,
                                           CpuExpressionOutput const &output)
{
    if (!ValidProgram(program)) return CpuExpressionStatus::InvalidProgram;
    const unsigned outputCount = program.outputCount ? program.outputCount : 1u;
    // The same agreement gpu::ValidInputs enforces before a launch.
    if (!ExprIsDomain(inputs.context.domain) || inputs.count != output.count ||
        (inputs.count && !output.data) || output.type == ScalarType::Invalid ||
        output.components != outputCount ||
        (inputs.context.domain == Domain::Groom && inputs.count != 1))
        return CpuExpressionStatus::InvalidArgument;
    for (auto const &field : inputs.fields) {
        if (!field.data && field.count == 0) continue;
        if (!field.data || !ExprIsDomain(field.domain) || field.components == 0 ||
            field.components > 4)
            return CpuExpressionStatus::InvalidArgument;
        size_t expected = field.domain == Domain::Groom ? 1 : inputs.count;
        if (field.domain == Domain::Primitive && inputs.context.domain == Domain::Point) {
            expected = inputs.primitiveCount;
            if (!inputs.pointToPrimitive.data || inputs.pointToPrimitive.size != inputs.count)
                return CpuExpressionStatus::InvalidArgument;
        } else if (field.domain != Domain::Groom && field.domain != inputs.context.domain)
            return CpuExpressionStatus::InvalidArgument;
        if (field.count != expected) return CpuExpressionStatus::InvalidArgument;
    }
    uint16_t results[4]{};
    for (unsigned c = 0; c < outputCount; ++c)
        results[c] = program.outputCount ? program.output[c] : program.result;

    double registers[kExprRegisters];
    bool valid = true;
    for (size_t i = 0; i < inputs.count; ++i) {
        ExecuteElement(program.instructions.data(), program.instructions.size(),
                       registers, inputs, i);
        for (unsigned c = 0; c < outputCount; ++c)
            if (!Store(registers[results[c]], output, i * outputCount + c)) valid = false;
    }
    return valid ? CpuExpressionStatus::Ok : CpuExpressionStatus::InvalidValue;
}

} // namespace usdGen::expr
