// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// usdGen — Vulkan expression plan data and validation.
//
// Native-free per-node validation for expression-bound Width and UsdGenExprOp,
// plus the compiled stage data consumed by the executor.
// All diagnostics are plain strings; the fragment routes them through
// UsdGenDiagnostics::Error with the "Vulkan source-width plan: " prefix.
#ifndef USDGEN_VULKAN_EXPR_VK_PLAN_H
#define USDGEN_VULKAN_EXPR_VK_PLAN_H

#include "usdGenCurves/vulkan/executionPlan.h"
#include "exprVkProgram.h"

#include <string>
#include <vector>

namespace usdGen::vulkan {

// One compiled binding in plan data. For UsdGenExprOp's source (a plain
// string, not a binding) expression/output are empty, destination is
// "expr:source", and the literal is zeros (the program cannot read $value).
struct ExprVkStageBinding {
    UsdGenToken destination;
    expr::Domain domain = expr::Domain::Groom;
    expr::ScalarType type = expr::ScalarType::Float32;
    uint32_t components = 1;
    expr::IRProgram ir;
    std::vector<double> literal;
    UsdGenNodeKey expression;
    UsdGenToken output;
};

struct ExprVkWidthData {
    VulkanLiteralWidthControls literal;
    std::vector<ExprVkStageBinding> bindings;
};

struct ExprVkExprOpData {
    bool modeCv = true;
    bool displacement = true;
    float mask = 1.0f;
    std::string source;
    // [0] is always the compiled source; [1] is the connected mask when the
    // node binds one (a connected enabled is compiled for error parity with
    // the CPU lane, which compiles every binding, but not stored: Evaluate
    // ignores it on every lane).
    std::vector<ExprVkStageBinding> bindings;
};

// Compiles node.expressionBindings after the op record's admission
// (UsdGenValidateExpressionBindings, the one rule for every lane).
bool ExprVkCompileStageBindings(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                                std::vector<ExprVkStageBinding> *out,
                                std::vector<std::string> *diagnostics);

// Width with optional expression bindings: literal controls validate exactly
// as the literal lane, and width/mask floats plus replace/enabled bools may
// be connected (the op record's admission; enabled is groom-only).
bool ExprVkValidateWidthNode(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                             ExprVkWidthData *out, std::vector<std::string> *diagnostics);

// UsdGenExprOp: expr:source/expr:returnType/mode validate as Bind does,
// node.mode agrees with the mode param when both are present, the source
// compiles sampler-free without $value/$frame/$time (the Capture refusals),
// and mask/enabled accept connections (the ValidateExprOp allowlist).
// expr:maps still rejects: maps are the image-map gap's surface.
bool ExprVkValidateExprOpNode(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                              ExprVkExprOpData *out, std::vector<std::string> *diagnostics);

// Frame/time controls for expression evaluation, derived from the
// descriptor: frame is desc.time (timeCodes), time is seconds.
expr::Context ExprVkPlanControls(UsdGenGraphSpec const &desc, int32_t seed, expr::Domain domain);

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_EXPR_VK_PLAN_H
