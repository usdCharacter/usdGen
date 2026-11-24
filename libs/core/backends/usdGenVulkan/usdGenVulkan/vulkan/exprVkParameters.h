// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// usdGen — Vulkan connected-parameter evaluator.
//
// The Vulkan counterpart of CudaParameterEvaluator::Evaluate (synchronous
// path): for each compiled binding it builds a domain-specialized field set
// on device (ExprVkContextPipeline), runs the program's exact prefixes on
// device (ExprVkEvaluatePipeline), evaluates transcendental steps on the host
// with expressions/exprMath.h and <cmath> (bit-identical to the CPU reference
// lane), and publishes one device field per binding (8-byte slots, the
// exprVkEvaluate output ABI). Fields are borrowed until the next Evaluate or
// destruction.
//
// One context serves one binding (CUDA shares contexts across bindings of a
// domain; sharing is a future optimization, not a correctness need). All
// waits are bounded (30s per dispatch); a timeout quarantines and reports
// DeviceError.
#ifndef USDGEN_VULKAN_EXPR_VK_PARAMETERS_H
#define USDGEN_VULKAN_EXPR_VK_PARAMETERS_H

#include "exprVkContext.h"
#include "exprVkEvaluate.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace usdGen::vulkan {

enum class ExprVkParameterStatus {
    Ok,
    InvalidArgument,
    InvalidGeometry,
    InvalidChannel,
    InvalidValue,
    DeviceError
};

struct ExprVkParameterField {
    TfToken destination;
    // 8-byte slots, count*components of them; null only when count is 0.
    std::shared_ptr<const ChargedBuffer> data;
    size_t count = 0;
    expr::ScalarType type = expr::ScalarType::Invalid;
    uint32_t components = 1;
    expr::Domain domain = expr::Domain::Groom;
};

class ExprVkParameterEvaluator {
public:
    ExprVkParameterEvaluator(std::shared_ptr<ExprVkContextPipeline> contextPipeline,
                             std::shared_ptr<ExprVkEvaluatePipeline> evaluatePipeline);
    ~ExprVkParameterEvaluator();
    ExprVkParameterEvaluator(ExprVkParameterEvaluator const&) = delete;
    ExprVkParameterEvaluator& operator=(ExprVkParameterEvaluator const&) = delete;

    // Evaluates every binding over device geometry. controls carries the
    // frame/time/seed/descId; each binding's own domain selects n. On any
    // failure the previously published fields are cleared so no operator
    // reads a stale connected value (the CUDA lane's contract).
    ExprVkParameterStatus Evaluate(
        std::vector<ExprVkCompiledBinding> const& bindings,
        ExprVkCurveGeometry const& geometry, ExprVkGeometryChannels const& channels,
        expr::Context controls, std::vector<std::string>* diagnostics = nullptr);

    std::vector<ExprVkParameterField> const& Fields() const noexcept { return fields_; }
    ExprVkParameterField const* Find(TfToken const& destination) const;

private:
    std::shared_ptr<ExprVkContextPipeline> context_;
    std::shared_ptr<ExprVkEvaluatePipeline> evaluate_;
    std::vector<ExprVkParameterField> fields_;
    // Retained device state behind the borrowed fields.
    std::vector<std::shared_ptr<void>> retained_;
};

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_EXPR_VK_PARAMETERS_H
