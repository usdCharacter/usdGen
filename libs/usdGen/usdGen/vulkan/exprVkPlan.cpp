// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "exprVkPlan.h"

#include "usdGen/expressionTargets.h"
#include "usdGen/expressions/frontend.h"
#include "usdGen/expressions/irExec.h"
#include "usdGen/ops/width.h"
#include "usdGenMath/usdGenMath/ramp.h"

#include <cctype>
#include <cmath>
#include <set>

namespace usdGen::vulkan {
namespace {
void Error(std::vector<std::string> *diagnostics, std::string const &message) {
    if (diagnostics)
        diagnostics->push_back(message);
}
bool IsBlank(std::string const &text) {
    for (char c : text)
        if (!std::isspace(static_cast<unsigned char>(c)))
            return false;
    return true;
}
bool ReadFloatParam(UsdGenParamValue const &param, float *value) {
    if (!param.value.IsHolding<float>())
        return false;
    *value = param.value.UncheckedGet<float>();
    return std::isfinite(*value);
}
ExprVkCompileStatus ToBindings(UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
                               std::vector<ExprVkStageBinding> *out,
                               std::vector<std::string> *diagnostics) {
    std::vector<ExprVkCompiledBinding> compiled;
    ExprVkCompileStatus status =
        ExprVkCompileNodeBindings(desc, node, true, true, &compiled, diagnostics);
    if (status != ExprVkCompileStatus::Ok)
        return status;
    for (auto const &item : compiled) {
        ExprVkStageBinding binding;
        binding.destination = item.binding.destination;
        binding.domain = item.binding.domain;
        binding.type = item.binding.destinationShape.scalar;
        binding.components = item.binding.destinationShape.components;
        binding.ir = item.ir;
        binding.literal = item.literal;
        binding.expression = item.binding.expression;
        binding.output = item.binding.output;
        out->push_back(std::move(binding));
    }
    return ExprVkCompileStatus::Ok;
}
} // namespace

bool ExprVkCompileStageBindings(UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
                                std::vector<ExprVkStageBinding> *out,
                                std::vector<std::string> *diagnostics) {
    if (!out) {
        Error(diagnostics, "null stage bindings");
        return false;
    }
    std::vector<std::string> errors;
    if (!UsdGenValidateExpressionTargets(node, &errors)) {
        for (auto const &message : errors)
            Error(diagnostics, message);
        return false;
    }
    if (ToBindings(desc, node, out, diagnostics) != ExprVkCompileStatus::Ok)
        return false;
    return true;
}

bool ExprVkValidateWidthNode(UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
                             ExprVkWidthData *out, std::vector<std::string> *diagnostics) {
    if (!out || !node.mode.IsEmpty() || !node.references.empty() || !node.curves.empty()) {
        Error(diagnostics, "Width has unsupported mode/input configuration");
        return false;
    }
    ExprVkWidthData data;
    std::set<TfToken> seen;
    VtVec2fArray knots;
    TfToken interpolation("catmullRom");
    for (auto const &param : node.params) {
        if (!seen.insert(param.name).second) {
            Error(diagnostics, "duplicate Width parameter");
            return false;
        }
        auto const &name = param.name.GetString();
        if (name == "width" || name == "mask") {
            float value;
            if (!ReadFloatParam(param, &value)) {
                Error(diagnostics, "Width scalar must be finite float");
                return false;
            }
            if (name == "width")
                data.literal.width = value;
            else
                data.literal.mask = value;
        } else if (name == "replace") {
            if (!param.value.IsHolding<bool>()) {
                Error(diagnostics, "Width replace must be bool");
                return false;
            }
            data.literal.replace = param.value.UncheckedGet<bool>();
        } else if (name == "width:knots") {
            if (!param.value.IsHolding<VtVec2fArray>()) {
                Error(diagnostics, "Width knots must be float2 array");
                return false;
            }
            knots = param.value.UncheckedGet<VtVec2fArray>();
            float previous = -1;
            for (auto const &k : knots) {
                if (!std::isfinite(k[0]) || !std::isfinite(k[1]) || k[0] < 0 || k[0] > 1 ||
                    k[0] < previous) {
                    Error(diagnostics, "malformed Width knots");
                    return false;
                }
                previous = k[0];
            }
        } else if (name == "width:interpolation") {
            if (!param.value.IsHolding<TfToken>()) {
                Error(diagnostics, "Width interpolation must be token");
                return false;
            }
            interpolation = param.value.UncheckedGet<TfToken>();
        } else if (name == "width:profile:mode") {
            if (param.value != VtValue(TfToken("ramp")) && param.value != VtValue(TfToken("bezier"))) {
                Error(diagnostics, "Width profile mode must be ramp or bezier");
                return false;
            }
        } else if (name == "width:root" || name == "width:mid" || name == "width:midShift" ||
                   name == "width:tip") {
            if (!param.value.IsHolding<float>() || !std::isfinite(param.value.UncheckedGet<float>())) {
                Error(diagnostics, "Width profile controls must be finite float literals");
                return false;
            }
        } else {
            Error(diagnostics, "unsupported Width parameter " + name);
            return false;
        }
    }
    for (auto const &ramp : node.ramps)
        if (!ramp.knots.empty() || !ramp.positions.empty() || !ramp.colors.empty()) {
            Error(diagnostics, "anonymous Width ramp requires named parameter");
            return false;
        }
    if (!knots.empty())
        UsdGenBuildRampLut(knots, interpolation, data.literal.profile.data(), 257);
    // Bezier profile mode replaces the ramp LUT with the profile factor.
    UsdGenParamView const params{&desc, &node};
    bool const bezier = UsdGenWidthBezierLut(params, data.literal.profile.data(), 257);
    if (!ExprVkCompileStageBindings(desc, node, &data.bindings, diagnostics))
        return false;
    data.literal.extended = !knots.empty() || bezier || data.literal.mask != 1.0f ||
        !data.bindings.empty();
    *out = std::move(data);
    return true;
}

bool ExprVkValidateExprOpNode(UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
                              ExprVkExprOpData *out, std::vector<std::string> *diagnostics) {
    if (!out) {
        Error(diagnostics, "null ExprOp data");
        return false;
    }
    // expr:maps (node.maps) still rejects: sampler calls have no Vulkan
    // transport and maps are the image-map gap's surface.
    if (!node.references.empty() || !node.curves.empty() || !node.surfaces.empty() ||
        !node.maps.empty() || !node.mapBindings.empty() || !node.ramps.empty()) {
        Error(diagnostics, "UsdGenExprOp has unsupported input configuration");
        return false;
    }
    ExprVkExprOpData data;
    std::string modeParam;
    std::set<TfToken> seen;
    for (auto const &param : node.params) {
        if (!seen.insert(param.name).second) {
            Error(diagnostics, "duplicate UsdGenExprOp parameter " + param.name.GetString());
            return false;
        }
        auto const &name = param.name.GetString();
        if (name == "mode") {
            if (!param.value.IsHolding<TfToken>()) {
                Error(diagnostics, "UsdGenExprOp: unknown usdGen:mode");
                return false;
            }
            modeParam = param.value.UncheckedGet<TfToken>().GetString();
        } else if (name == "expr:returnType") {
            if (!param.value.IsHolding<TfToken>()) {
                Error(diagnostics, "UsdGenExprOp: unknown usdGen:expr:returnType");
                return false;
            }
            auto const value = param.value.UncheckedGet<TfToken>().GetString();
            if (value != "displacement" && value != "width" && value != "color") {
                Error(diagnostics, "UsdGenExprOp: unknown usdGen:expr:returnType '" + value + "'");
                return false;
            }
            if (value == "color") {
                Error(diagnostics, "UsdGenExprOp: usdGen:expr:returnType 'color' is not supported: "
                                   "the operator-emitted primvar transport carries no vec3 color "
                                   "plane, so a color field has nowhere to publish");
                return false;
            }
            data.displacement = value == "displacement";
        } else if (name == "expr:source") {
            if (param.value.IsHolding<std::string>())
                data.source = param.value.UncheckedGet<std::string>();
            else if (param.value.IsHolding<TfToken>())
                data.source = param.value.UncheckedGet<TfToken>().GetString();
            else {
                Error(diagnostics, "UsdGenExprOp: usdGen:expr:source must be a string");
                return false;
            }
        } else if (name == "mask") {
            if (!ReadFloatParam(param, &data.mask)) {
                Error(diagnostics, "UsdGenExprOp: mask must be a finite float");
                return false;
            }
        } else if (name == "enabled") {
            if (!param.value.IsHolding<bool>()) {
                Error(diagnostics, "UsdGenExprOp: enabled must be bool");
                return false;
            }
        } else if (name == "label") {
            if (!param.value.IsHolding<std::string>()) {
                Error(diagnostics, "UsdGenExprOp: label must be a string");
                return false;
            }
        } else {
            Error(diagnostics, "unsupported UsdGenExprOp parameter " + name);
            return false;
        }
    }
    // The builder puts usdGen:mode in both node.mode and params; a direct
    // client that disagrees with itself fails closed instead of silently
    // following one spelling.
    std::string const nodeMode = node.mode.IsEmpty() ? "" : node.mode.GetString();
    if (!nodeMode.empty() && nodeMode != "cv" && nodeMode != "curve") {
        Error(diagnostics, "UsdGenExprOp: unknown usdGen:mode '" + nodeMode + "'");
        return false;
    }
    if (!modeParam.empty() && modeParam != "cv" && modeParam != "curve") {
        Error(diagnostics, "UsdGenExprOp: unknown usdGen:mode '" + modeParam + "'");
        return false;
    }
    if (!nodeMode.empty() && !modeParam.empty() && nodeMode != modeParam) {
        Error(diagnostics, "UsdGenExprOp: node mode and mode parameter disagree");
        return false;
    }
    std::string const mode = !modeParam.empty() ? modeParam : !nodeMode.empty() ? nodeMode : "cv";
    data.modeCv = mode == "cv";
    if (IsBlank(data.source)) {
        Error(diagnostics, "UsdGenExprOp: usdGen:expr:source is empty");
        return false;
    }
    expr::Domain const domain = data.modeCv ? expr::Domain::Point : expr::Domain::Primitive;
    uint32_t const components = data.displacement ? 3u : 1u;
    expr::FrontendOptions options;
    options.domain = domain;
    options.destination = expr::ScalarType::Float32;
    options.components = components;
    expr::CompileResult compiled = expr::Frontend::Compile(data.source, options);
    if (!compiled.ok) {
        if (compiled.diagnostics.empty())
            Error(diagnostics, "UsdGenExprOp: expression compilation failed");
        for (auto const &message : compiled.diagnostics)
            Error(diagnostics, "UsdGenExprOp: " + message);
        return false;
    }
    expr::IRProgram const ir = compiled.program.IR();
    if (!ir.samplers.empty()) {
        expr::IRSampler const &slot = ir.samplers.front();
        std::string const call = slot.kind == expr::SamplerKind::Geometry
                                     ? "geoSampler(\"" + slot.input + "\")"
                                     : "ptex(\"" + slot.input + "\")";
        Error(diagnostics, "UsdGenExprOp: expression calls " + call +
                               ", but a bare usdGen:expr:source has no input:<name> bindings to "
                               "resolve it against; usdGen:expr:maps names no maps");
        return false;
    }
    if (!expr::ValidProgram(ir)) {
        Error(diagnostics, "UsdGenExprOp: compiler produced an invalid program");
        return false;
    }
    for (auto const &instruction : ir.instructions) {
        if (instruction.op != expr::IROp::LoadVariable)
            continue;
        if (instruction.variable == expr::Variable::Value) {
            Error(diagnostics, "UsdGenExprOp: expression reads $value, which a bare "
                               "usdGen:expr:source does not define (width authors read "
                               "the upstream width as $cWidth)");
            return false;
        }
        if (instruction.variable == expr::Variable::Frame ||
            instruction.variable == expr::Variable::Time) {
            Error(diagnostics,
                  "UsdGenExprOp: expression reads " +
                      std::string(instruction.variable == expr::Variable::Frame ? "$frame"
                                                                                : "$time") +
                      ", but the operator evaluates once per capture and cannot see "
                      "time");
            return false;
        }
    }
    ExprVkStageBinding source;
    source.destination = TfToken("expr:source");
    source.domain = domain;
    source.type = expr::ScalarType::Float32;
    source.components = components;
    source.ir = ir;
    source.literal.assign(components, 0.0);
    data.bindings.push_back(std::move(source));
    if (!node.expressionBindings.empty()) {
        std::vector<std::string> errors;
        if (!UsdGenValidateExpressionTargets(node, &errors)) {
            for (auto const &message : errors)
                Error(diagnostics, message);
            return false;
        }
        std::vector<ExprVkCompiledBinding> compiledBindings;
        if (ExprVkCompileNodeBindings(desc, node, true, true, &compiledBindings, diagnostics) !=
            ExprVkCompileStatus::Ok)
            return false;
        for (auto const &item : compiledBindings) {
            auto name = item.binding.destination.GetString();
            if (name.compare(0, 7, "usdGen:") == 0)
                name.erase(0, 7);
            if (name != "mask")
                continue; // enabled: compiled, then ignored (all lanes)
            ExprVkStageBinding binding;
            binding.destination = item.binding.destination;
            binding.domain = item.binding.domain;
            binding.type = item.binding.destinationShape.scalar;
            binding.components = item.binding.destinationShape.components;
            binding.ir = item.ir;
            binding.literal = item.literal;
            binding.expression = item.binding.expression;
            binding.output = item.binding.output;
            data.bindings.push_back(std::move(binding));
        }
    }
    *out = std::move(data);
    return true;
}

expr::Context ExprVkPlanControls(UsdGenGraphDesc const &desc, int32_t seed, expr::Domain domain) {
    expr::Context controls;
    controls.frame = desc.time;
    double const tcps = desc.timeCodesPerSecond > 0.0 ? desc.timeCodesPerSecond : 24.0;
    controls.time = desc.time / tcps;
    controls.seed = seed;
    controls.descId = expr::DescriptionId(desc.description.GetText());
    controls.domain = domain;
    return controls;
}

} // namespace usdGen::vulkan
