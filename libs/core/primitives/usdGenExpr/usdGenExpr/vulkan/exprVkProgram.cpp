// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "exprVkProgram.h"
#include "usdGen/api/value.h"
#include "usdGen/api/token.h"
#include "usdGenExpr/exprInput.h"

#include "usdGen/expressions/exprMath.h"
#include "usdGen/expressions/frontend.h"
#include "usdGen/expressions/irExec.h"

#include "usdGenMath/math.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>

namespace usdGen::vulkan {
namespace {
using Type = expr::ScalarType;
using Domain = expr::Domain;

// The shader hardcodes these numerics; an enum reorder must break the build.
static_assert(static_cast<unsigned>(expr::IROp::Const) == 0);
static_assert(static_cast<unsigned>(expr::IROp::LoadVariable) == 1);
static_assert(static_cast<unsigned>(expr::IROp::Add) == 2);
static_assert(static_cast<unsigned>(expr::IROp::Sub) == 3);
static_assert(static_cast<unsigned>(expr::IROp::Mul) == 4);
static_assert(static_cast<unsigned>(expr::IROp::Div) == 5);
static_assert(static_cast<unsigned>(expr::IROp::Neg) == 6);
static_assert(static_cast<unsigned>(expr::IROp::Compare) == 7);
static_assert(static_cast<unsigned>(expr::IROp::Select) == 8);
static_assert(static_cast<unsigned>(expr::IROp::Min) == 9);
static_assert(static_cast<unsigned>(expr::IROp::Max) == 10);
static_assert(static_cast<unsigned>(expr::IROp::Clamp) == 11);
static_assert(static_cast<unsigned>(expr::IROp::Abs) == 12);
static_assert(static_cast<unsigned>(expr::IROp::Sin) == 13);
static_assert(static_cast<unsigned>(expr::IROp::Cos) == 14);
static_assert(static_cast<unsigned>(expr::IROp::Pow) == 15);
static_assert(static_cast<unsigned>(expr::IROp::Sqrt) == 16);
static_assert(static_cast<unsigned>(expr::IROp::Exp) == 17);
static_assert(static_cast<unsigned>(expr::IROp::Log) == 18);
static_assert(static_cast<unsigned>(expr::IROp::Floor) == 19);
static_assert(static_cast<unsigned>(expr::IROp::Ceil) == 20);
static_assert(static_cast<unsigned>(expr::IROp::Fmod) == 21);
static_assert(static_cast<unsigned>(expr::IROp::Tan) == 22);
static_assert(static_cast<unsigned>(expr::IROp::Atan2) == 23);
static_assert(static_cast<unsigned>(expr::IROp::Move) == 24);
static_assert(static_cast<unsigned>(expr::IROp::Asin) == 25);
static_assert(static_cast<unsigned>(expr::IROp::Acos) == 26);
static_assert(static_cast<unsigned>(expr::IROp::Atan) == 27);
static_assert(static_cast<unsigned>(expr::IROp::Sinh) == 28);
static_assert(static_cast<unsigned>(expr::IROp::Cosh) == 29);
static_assert(static_cast<unsigned>(expr::IROp::Tanh) == 30);
static_assert(static_cast<unsigned>(expr::IROp::Log10) == 31);
static_assert(static_cast<unsigned>(expr::IROp::Cbrt) == 32);
static_assert(static_cast<unsigned>(expr::IROp::Round) == 33);
static_assert(static_cast<unsigned>(expr::IROp::Trunc) == 34);
static_assert(static_cast<unsigned>(expr::IROp::Hypot) == 35);
static_assert(static_cast<unsigned>(expr::IROp::Call) == 36);
static_assert(static_cast<unsigned>(expr::IROp::Sample) == 37);
static_assert(static_cast<unsigned>(expr::IRFunc::Invalid) == 0);
static_assert(static_cast<unsigned>(expr::IRFunc::Expand) == 1);
static_assert(static_cast<unsigned>(expr::IRFunc::Bias) == 2);
static_assert(static_cast<unsigned>(expr::IRFunc::Contrast) == 3);
static_assert(static_cast<unsigned>(expr::IRFunc::BoxStep) == 4);
static_assert(static_cast<unsigned>(expr::IRFunc::LinearStep) == 5);
static_assert(static_cast<unsigned>(expr::IRFunc::SmoothStep) == 6);
static_assert(static_cast<unsigned>(expr::IRFunc::GaussStep) == 7);
static_assert(static_cast<unsigned>(expr::IRFunc::Remap) == 8);
static_assert(static_cast<unsigned>(expr::IRFunc::Cycle) == 9);
static_assert(static_cast<unsigned>(expr::IRFunc::Hash) == 10);
static_assert(static_cast<unsigned>(expr::IRFunc::Noise) == 11);
static_assert(static_cast<unsigned>(expr::IRFunc::SNoise) == 12);
static_assert(static_cast<unsigned>(expr::IRFunc::VNoise) == 13);
static_assert(static_cast<unsigned>(expr::IRFunc::CNoise) == 14);
static_assert(static_cast<unsigned>(expr::IRFunc::PNoise) == 15);
static_assert(static_cast<unsigned>(expr::IRFunc::CellNoise) == 16);
static_assert(static_cast<unsigned>(expr::IRFunc::CCellNoise) == 17);
static_assert(static_cast<unsigned>(expr::IRFunc::Fbm) == 18);
static_assert(static_cast<unsigned>(expr::IRFunc::VFbm) == 19);
static_assert(static_cast<unsigned>(expr::IRFunc::CFbm) == 20);
static_assert(static_cast<unsigned>(expr::IRFunc::Turbulence) == 21);
static_assert(static_cast<unsigned>(expr::IRFunc::VTurbulence) == 22);
static_assert(static_cast<unsigned>(expr::IRFunc::CTurbulence) == 23);
static_assert(static_cast<unsigned>(expr::IRFunc::Voronoi) == 24);
static_assert(static_cast<unsigned>(expr::IRFunc::CVoronoi) == 25);
static_assert(static_cast<unsigned>(expr::IRFunc::PVoronoi) == 26);
static_assert(static_cast<unsigned>(expr::IRFunc::Dist) == 27);
static_assert(static_cast<unsigned>(expr::IRFunc::Length) == 28);
static_assert(static_cast<unsigned>(expr::IRFunc::Dot) == 29);
static_assert(static_cast<unsigned>(expr::IRFunc::Cross) == 30);
static_assert(static_cast<unsigned>(expr::IRFunc::Norm) == 31);
static_assert(static_cast<unsigned>(expr::IRFunc::Angle) == 32);
static_assert(static_cast<unsigned>(expr::IRFunc::Ortho) == 33);
static_assert(static_cast<unsigned>(expr::IRFunc::Rotate) == 34);
static_assert(static_cast<unsigned>(expr::IRFunc::Up) == 35);
static_assert(static_cast<unsigned>(expr::IRFunc::RgbToHsl) == 36);
static_assert(static_cast<unsigned>(expr::IRFunc::HslToRgb) == 37);
static_assert(static_cast<unsigned>(expr::IRFunc::Saturate) == 38);
static_assert(static_cast<unsigned>(expr::IRFunc::Hsi) == 39);
static_assert(static_cast<unsigned>(expr::IRFunc::MidHsi) == 40);
static_assert(static_cast<unsigned>(expr::IRFunc::Spline) == 41);
static_assert(static_cast<unsigned>(expr::IRFunc::Choose) == 42);
static_assert(static_cast<unsigned>(expr::IRFunc::WChoose) == 43);
static_assert(static_cast<unsigned>(expr::IRFunc::Pick) == 44);
static_assert(static_cast<unsigned>(expr::IRFunc::Curve) == 45);
static_assert(static_cast<unsigned>(expr::IRFunc::CCurve) == 46);
static_assert(static_cast<unsigned>(expr::Variable::Invalid) == 0);
static_assert(static_cast<unsigned>(expr::Variable::Value) == 1);
static_assert(static_cast<unsigned>(expr::Variable::Frame) == 2);
static_assert(static_cast<unsigned>(expr::Variable::Time) == 3);
static_assert(static_cast<unsigned>(expr::Variable::Index) == 4);
static_assert(static_cast<unsigned>(expr::Variable::Count) == 5);
static_assert(static_cast<unsigned>(expr::Variable::Seed) == 6);
static_assert(static_cast<unsigned>(expr::Variable::DescId) == 7);
static_assert(static_cast<unsigned>(expr::Variable::PrimIndex) == 8);
static_assert(static_cast<unsigned>(expr::Variable::PrimCount) == 9);
static_assert(static_cast<unsigned>(expr::Variable::IdLo) == 10);
static_assert(static_cast<unsigned>(expr::Variable::IdHi) == 11);
static_assert(static_cast<unsigned>(expr::Variable::Id) == 12);
static_assert(static_cast<unsigned>(expr::Variable::U) == 13);
static_assert(static_cast<unsigned>(expr::Variable::V) == 14);
static_assert(static_cast<unsigned>(expr::Variable::FaceId) == 15);
static_assert(static_cast<unsigned>(expr::Variable::P) == 16);
static_assert(static_cast<unsigned>(expr::Variable::PRef) == 17);
static_assert(static_cast<unsigned>(expr::Variable::RootP) == 18);
static_assert(static_cast<unsigned>(expr::Variable::RootPRef) == 19);
static_assert(static_cast<unsigned>(expr::Variable::N) == 20);
static_assert(static_cast<unsigned>(expr::Variable::NRef) == 21);
static_assert(static_cast<unsigned>(expr::Variable::DPdu) == 22);
static_assert(static_cast<unsigned>(expr::Variable::DPdv) == 23);
static_assert(static_cast<unsigned>(expr::Variable::DPduRef) == 24);
static_assert(static_cast<unsigned>(expr::Variable::DPdvRef) == 25);
static_assert(static_cast<unsigned>(expr::Variable::T) == 26);
static_assert(static_cast<unsigned>(expr::Variable::PointIndex) == 27);
static_assert(static_cast<unsigned>(expr::Variable::PointCount) == 28);
static_assert(static_cast<unsigned>(expr::Variable::CLength) == 29);
static_assert(static_cast<unsigned>(expr::Variable::CWidth) == 30);
static_assert(static_cast<unsigned>(expr::Variable::Q) == 31);
static_assert(static_cast<unsigned>(expr::Variable::QDist) == 32);
static_assert(static_cast<unsigned>(expr::Variable::CountVariables) == 33);
static_assert(static_cast<unsigned>(Domain::None) == 0);
static_assert(static_cast<unsigned>(Domain::Groom) == 1);
static_assert(static_cast<unsigned>(Domain::Primitive) == 2);
static_assert(static_cast<unsigned>(Domain::Point) == 4);
static_assert(static_cast<unsigned>(Type::Invalid) == 0);
static_assert(static_cast<unsigned>(Type::Bool) == 1);
static_assert(static_cast<unsigned>(Type::Int32) == 2);
static_assert(static_cast<unsigned>(Type::UInt32) == 3);
static_assert(static_cast<unsigned>(Type::Int64) == 4);
static_assert(static_cast<unsigned>(Type::UInt64) == 5);
static_assert(static_cast<unsigned>(Type::Float16) == 6);
static_assert(static_cast<unsigned>(Type::Float32) == 7);
static_assert(static_cast<unsigned>(Type::Float64) == 8);

void Error(std::vector<std::string>* diagnostics, std::string const& message) {
    if (diagnostics) diagnostics->push_back(message);
}
std::string CanonicalName(UsdGenToken const& name) {
    auto value = name.GetString();
    if (value.compare(0, 7, "usdGen:") == 0) value.erase(0, 7);
    return value;
}
size_t ScalarBytes(Type type) {
    switch (type) {
    case Type::Bool: return 1;
    case Type::Float16: return 2;
    case Type::Int32: case Type::UInt32: case Type::Float32: return 4;
    case Type::Int64: case Type::UInt64: case Type::Float64: return 8;
    default: return 0;
    }
}
bool SupportedShape(expr::ValueShape const& shape) {
    return ScalarBytes(shape.scalar) && !shape.isArray &&
        shape.elementCount == 1 && shape.rows == 1 && shape.columns == 1 &&
        shape.components >= 1 && shape.components <= 4;
}
bool SameShape(expr::ValueShape const& a, expr::ValueShape const& b) {
    return a.scalar == b.scalar && a.elementCount == b.elementCount &&
        a.components == b.components && a.rows == b.rows &&
        a.columns == b.columns && a.isArray == b.isArray;
}
template<class T>
bool ScalarLiteral(UsdGenValue const& value, std::vector<double>* output) {
    if (!value.IsHolding<T>()) return false;
    output->assign(1, static_cast<double>(value.UncheckedGet<T>()));
    return std::isfinite((*output)[0]);
}
template<class T>
bool VectorLiteral(UsdGenValue const& value, unsigned count, std::vector<double>* output) {
    if (!value.IsHolding<T>() || count != T::dimension) return false;
    output->resize(count);
    auto const& vector = value.UncheckedGet<T>();
    for (unsigned i = 0; i < count; ++i) {
        (*output)[i] = static_cast<double>(vector[i]);
        if (!std::isfinite((*output)[i])) return false;
    }
    return true;
}
// Exact copy of the CUDA lane's literal rule: the exact finite native type,
// int64/uint64 limited to 2^53.
bool Literal(UsdGenValue const& value, expr::ValueShape const& shape,
             std::vector<double>* output) {
    unsigned n = shape.components;
    if (n == 1) {
        switch (shape.scalar) {
        case Type::Bool: return ScalarLiteral<bool>(value, output);
        case Type::Int32: return ScalarLiteral<int32_t>(value, output);
        case Type::UInt32: return ScalarLiteral<uint32_t>(value, output);
        case Type::Int64:
            if (!value.IsHolding<int64_t>() ||
                value.UncheckedGet<int64_t>() < -(int64_t(1) << 53) ||
                value.UncheckedGet<int64_t>() > (int64_t(1) << 53)) return false;
            return ScalarLiteral<int64_t>(value, output);
        case Type::UInt64:
            if (!value.IsHolding<uint64_t>() ||
                value.UncheckedGet<uint64_t>() > (uint64_t(1) << 53)) return false;
            return ScalarLiteral<uint64_t>(value, output);
        case Type::Float16: return ScalarLiteral<UsdGenHalf>(value, output);
        case Type::Float32: return ScalarLiteral<float>(value, output);
        case Type::Float64: return ScalarLiteral<double>(value, output);
        default: return false;
        }
    }
    switch (shape.scalar) {
    case Type::Float16:
        return VectorLiteral<UsdGenVec2h>(value, n, output) ||
            VectorLiteral<UsdGenVec3h>(value, n, output) || VectorLiteral<UsdGenVec4h>(value, n, output);
    case Type::Float32:
        return VectorLiteral<UsdGenVec2f>(value, n, output) ||
            VectorLiteral<UsdGenVec3f>(value, n, output) || VectorLiteral<UsdGenVec4f>(value, n, output);
    case Type::Float64:
        return VectorLiteral<UsdGenVec2d>(value, n, output) ||
            VectorLiteral<UsdGenVec3d>(value, n, output) || VectorLiteral<UsdGenVec4d>(value, n, output);
    case Type::Int32:
        return VectorLiteral<UsdGenVec2i>(value, n, output) ||
            VectorLiteral<UsdGenVec3i>(value, n, output) || VectorLiteral<UsdGenVec4i>(value, n, output);
    default: return false;
    }
}
} // namespace

ExprVkCompileStatus ExprVkCompileBinding(
    UsdGenGraphSpec const& graph, UsdGenParamBinding const& binding,
    bool allowValue, bool allowTime, ExprVkCompiledBinding* out,
    std::vector<std::string>* diagnostics) {
    if (!out) {
        Error(diagnostics, "null compiled binding");
        return ExprVkCompileStatus::InvalidArgument;
    }
    auto fail = [&](ExprVkCompileStatus status, std::string message) {
        Error(diagnostics, binding.destination.GetString() + ": " + message);
        return status;
    };
    if (binding.destination.IsEmpty() || !binding.expression.IsValid())
        return fail(ExprVkCompileStatus::InvalidArgument, "empty expression destination/path");
    UsdGenInputRecord const* expressionRecord = UsdGenFindExpressionInput(graph, binding.expression);
    if (!expressionRecord)
        return fail(ExprVkCompileStatus::CompileError, "missing expression");
    UsdGenExpressionInputView const expressionView = UsdGenExpressionInputView::Of(*expressionRecord);
    UsdGenExpressionInputView const* expression = &expressionView;
    UsdGenExpressionOutputDesc const* result =
        UsdGenFindExpressionOutput(*expression, binding.output);
    if (!result)
        return fail(ExprVkCompileStatus::CompileError, binding.output.IsEmpty()
            ? "connection to prim " + graph.Label(binding.expression) +
                  " declaring no outputs:result and no single outputs:* attribute"
            : "missing expression output");
    if (binding.domain != Domain::Groom && binding.domain != Domain::Primitive &&
        binding.domain != Domain::Point)
        return fail(ExprVkCompileStatus::InvalidArgument, "invalid evaluation domain");
    if (!SupportedShape(binding.destinationShape))
        return fail(ExprVkCompileStatus::UnsupportedType, "unsupported array/matrix/type shape");
    if (binding.nativeType.IsEmpty() || result->nativeType != binding.nativeType ||
        !SameShape(result->shape, binding.destinationShape))
        return fail(ExprVkCompileStatus::InvalidArgument, "output native type/shape mismatch");
    ExprVkCompiledBinding item;
    item.binding = binding;
    if (!Literal(binding.literal, binding.destinationShape, &item.literal))
        return fail(ExprVkCompileStatus::UnsupportedType,
            "literal must have the exact finite native type; int64/uint64 currently limited to 2^53");
    auto compiled = expr::Frontend::Compile(std::string(expression->source),
        {binding.domain, binding.destinationShape.scalar, binding.destinationShape.components});
    if (!compiled.ok) {
        for (auto const& message : compiled.diagnostics)
            Error(diagnostics, binding.destination.GetString() + ": " + message);
        return fail(ExprVkCompileStatus::CompileError, "SeExpr compilation failed");
    }
    if (!compiled.program.IR().samplers.empty())
        return fail(ExprVkCompileStatus::UnsupportedType,
            "geoSampler() and ptex() are only available on the CPU lane");
    item.ir = compiled.program.IR();
    if (!expr::ValidProgram(item.ir))
        return fail(ExprVkCompileStatus::CompileError, "compiler produced an invalid program");
    for (auto const& instruction : item.ir.instructions) {
        if (instruction.op != expr::IROp::LoadVariable) continue;
        if (!allowValue && instruction.variable == expr::Variable::Value)
            return fail(ExprVkCompileStatus::CompileError,
                "expression reads $value, which a bare usdGen:expr:source does not define "
                "(width authors read the upstream width as $cWidth)");
        if (!allowTime && (instruction.variable == expr::Variable::Frame ||
                            instruction.variable == expr::Variable::Time))
            return fail(ExprVkCompileStatus::CompileError,
                std::string("expression reads ") +
                    (instruction.variable == expr::Variable::Frame ? "$frame" : "$time") +
                    ", but the operator evaluates once per capture and cannot see time");
    }
    *out = std::move(item);
    return ExprVkCompileStatus::Ok;
}

ExprVkCompileStatus ExprVkCompileNodeBindings(
    UsdGenGraphSpec const& graph, UsdGenOpNode const& node,
    bool allowValue, bool allowTime,
    std::vector<ExprVkCompiledBinding>* out,
    std::vector<std::string>* diagnostics) {
    if (!out) {
        Error(diagnostics, "null compiled bindings");
        return ExprVkCompileStatus::InvalidArgument;
    }
    std::vector<ExprVkCompiledBinding> items;
    std::set<std::string> destinations;
    for (auto const& binding : node.expressionBindings) {
        if (!destinations.insert(CanonicalName(binding.destination)).second) {
            Error(diagnostics, binding.destination.GetString() + ": duplicate expression destination");
            return ExprVkCompileStatus::InvalidArgument;
        }
        ExprVkCompiledBinding item;
        ExprVkCompileStatus status = ExprVkCompileBinding(graph, binding, allowValue,
                                                           allowTime, &item, diagnostics);
        if (status != ExprVkCompileStatus::Ok) return status;
        items.push_back(std::move(item));
    }
    *out = std::move(items);
    return ExprVkCompileStatus::Ok;
}

bool ExprVkIsHostInstruction(expr::IRInstruction const& instruction) {
    using Op = expr::IROp;
    switch (instruction.op) {
    case Op::Sin: case Op::Cos: case Op::Pow: case Op::Exp: case Op::Log:
    case Op::Tan: case Op::Atan2: case Op::Asin: case Op::Acos: case Op::Atan:
    case Op::Sinh: case Op::Cosh: case Op::Tanh: case Op::Log10: case Op::Cbrt:
        return true;
    case Op::Call: {
        auto func = static_cast<expr::IRFunc>(instruction.c);
        return func == expr::IRFunc::Bias || func == expr::IRFunc::Contrast ||
            func == expr::IRFunc::GaussStep || func == expr::IRFunc::Remap ||
            func == expr::IRFunc::Angle || func == expr::IRFunc::Rotate ||
            func == expr::IRFunc::Up || func == expr::IRFunc::MidHsi;
    }
    default:
        return false;
    }
}

std::vector<size_t> ExprVkHostSteps(expr::IRProgram const& program) {
    std::vector<size_t> steps;
    for (size_t i = 0; i < program.instructions.size(); ++i)
        if (ExprVkIsHostInstruction(program.instructions[i])) steps.push_back(i);
    return steps;
}

bool ExprVkEvaluateHostStep(expr::IRInstruction const& instruction,
                            double const* const* inputs, size_t count,
                            double* outputs) {
    using Op = expr::IROp;
    auto unary = [&](double (*f)(double)) {
        for (size_t e = 0; e < count; ++e) outputs[e] = f(inputs[0][e]);
        return true;
    };
    switch (instruction.op) {
    case Op::Sin: return unary(std::sin);
    case Op::Cos: return unary(std::cos);
    case Op::Tan: return unary(std::tan);
    case Op::Asin: return unary(std::asin);
    case Op::Acos: return unary(std::acos);
    case Op::Atan: return unary(std::atan);
    case Op::Sinh: return unary(std::sinh);
    case Op::Cosh: return unary(std::cosh);
    case Op::Tanh: return unary(std::tanh);
    case Op::Exp: return unary(std::exp);
    case Op::Log: return unary(std::log);
    case Op::Log10: return unary(std::log10);
    case Op::Cbrt: return unary(std::cbrt);
    case Op::Pow:
        for (size_t e = 0; e < count; ++e)
            outputs[e] = std::pow(inputs[0][e], inputs[1][e]);
        return true;
    case Op::Atan2:
        for (size_t e = 0; e < count; ++e)
            outputs[e] = std::atan2(inputs[0][e], inputs[1][e]);
        return true;
    case Op::Call: {
        auto func = static_cast<expr::IRFunc>(instruction.c);
        if (func == expr::IRFunc::Bias || func == expr::IRFunc::Contrast ||
            func == expr::IRFunc::GaussStep || func == expr::IRFunc::Remap ||
            func == expr::IRFunc::Angle || func == expr::IRFunc::Rotate ||
            func == expr::IRFunc::Up || func == expr::IRFunc::MidHsi) {
            std::vector<double> block(instruction.b);
            for (size_t e = 0; e < count; ++e) {
                for (unsigned k = 0; k < instruction.b; ++k)
                    block[k] = inputs[k][e];
                outputs[e] = expr::ExprCall(func, block.data(), int(instruction.b),
                                            instruction.component);
            }
            return true;
        }
        return false;
    }
    default:
        return false;
    }
}

ExprVkFieldLayout ExprVkLayoutFields(expr::Domain domain, size_t n,
                                     unsigned literalComponents,
                                     unsigned injectionCount, bool hasRest,
                                     bool hasWidths, bool hasRootUV,
                                     bool hasRootN, bool hasRootT,
                                     bool hasRootB, bool hasRootPrim) {
    using V = expr::Variable;
    ExprVkFieldLayout layout;
    unsigned const base = ExprVkEncoding::kBaseVariables;
    layout.varCount = base + injectionCount;
    layout.offsets.assign(layout.varCount, UINT64_MAX);
    layout.counts.assign(layout.varCount, 0);
    layout.domains.assign(layout.varCount, Domain::None);
    layout.components.assign(layout.varCount, 0);
    uint64_t next = 0;
    auto place = [&](unsigned v, unsigned components) {
        layout.offsets[v] = next;
        layout.counts[v] = uint32_t(n);
        layout.domains[v] = domain;
        layout.components[v] = components;
        next += uint64_t(n) * components;
    };
    auto vec = [&](V v) { place(static_cast<unsigned>(v), 3); };
    auto scalar = [&](V v) { place(static_cast<unsigned>(v), 1); };
    if (domain == Domain::Primitive || domain == Domain::Point) {
        vec(V::P);
        vec(V::RootP);
        if (hasRest) {
            vec(V::PRef);
            vec(V::RootPRef);
        }
        if (hasWidths) scalar(V::CWidth);
        if (hasRootUV) {
            scalar(V::U);
            scalar(V::V);
        }
        if (hasRootN) {
            vec(V::N);
            vec(V::NRef);
        }
        if (hasRootT) {
            vec(V::DPdu);
            vec(V::DPduRef);
        }
        if (hasRootB) {
            vec(V::DPdv);
            vec(V::DPdvRef);
        }
        if (hasRootPrim) scalar(V::FaceId);
        scalar(V::T);
        scalar(V::PrimIndex);
        scalar(V::PrimCount);
        scalar(V::CLength);
        scalar(V::IdLo);
        scalar(V::IdHi);
        scalar(V::Id);
        // CUDA skips PointIndex/PointCount at primitive rate.
        if (domain == Domain::Point) {
            scalar(V::PointIndex);
            scalar(V::PointCount);
        }
    }
    if (literalComponents) {
        layout.offsets[static_cast<unsigned>(V::Value)] = next;
        layout.counts[static_cast<unsigned>(V::Value)] = 1;
        layout.domains[static_cast<unsigned>(V::Value)] = Domain::Groom;
        layout.components[static_cast<unsigned>(V::Value)] = literalComponents;
        next += literalComponents;
    }
    for (unsigned j = 0; j < injectionCount; ++j) {
        unsigned const v = base + j;
        layout.offsets[v] = next;
        layout.counts[v] = uint32_t(n);
        layout.domains[v] = domain;
        layout.components[v] = 1;
        next += uint64_t(n);
    }
    layout.totalDoubles = next;
    return layout;
}

bool ExprVkPackProgram(expr::IRProgram const& program, size_t endExclusive,
                       std::map<size_t, unsigned> const& rewriteSlot,
                       ExprVkSegmentOutputs const& outputs,
                       ExprVkPackOptions const& options,
                       std::vector<ExprVkFieldSlot> const& table,
                       std::vector<uint32_t>* words,
                       std::string* diagnostic) {
    auto fail = [&](std::string const& message) {
        if (diagnostic) *diagnostic = message;
        return false;
    };
    if (!words) return fail("null program words");
    if (endExclusive == 0 || endExclusive > program.instructions.size())
        return fail("segment overruns the program");
    if (outputs.raw) {
        if (outputs.regCount == 0 || outputs.regCount > expr::kExprRegisters ||
            unsigned(outputs.rawBase) + outputs.regCount > expr::kExprRegisters)
            return fail("raw segment outputs out of range");
    } else {
        if (outputs.regCount == 0 || outputs.regCount > 4 ||
            outputs.type == Type::Invalid)
            return fail("cooked segment outputs out of range");
        for (unsigned k = 0; k < outputs.regCount; ++k)
            if (outputs.regs[k] >= program.registerCount) return fail("output register out of range");
    }
    if (program.registerCount == 0 || program.registerCount > expr::kExprRegisters)
        return fail("register count out of range");
    for (auto const& [index, slot] : rewriteSlot) {
        if (index >= endExclusive || slot < ExprVkEncoding::kFirstInjectionSlot ||
            slot >= table.size())
            return fail("injection rewrite out of range");
    }
    uint64_t const codeBytes = uint64_t(endExclusive) * ExprVkEncoding::kInstructionBytes;
    uint64_t const tableBytes = uint64_t(table.size()) * ExprVkEncoding::kTableEntryBytes;
    constexpr uint64_t kG3 = 514ull * 3ull * 8ull;
    constexpr uint64_t kP514 = 514ull * 4ull;
    constexpr uint64_t kP256 = 256ull * 4ull;
    uint64_t const headerBytes = ExprVkEncoding::kHeaderBytes;
    uint64_t const codeOffset = headerBytes;
    uint64_t const tableOffset = codeOffset + codeBytes;
    uint64_t const g3Offset = tableOffset + tableBytes;
    uint64_t const p514Offset = g3Offset + kG3;
    uint64_t const p256Offset = p514Offset + kP514;
    uint64_t const total = p256Offset + kP256;
    if (total > uint64_t(UINT32_MAX))
        return fail("program image overflows 32 bits");
    std::vector<uint32_t> out(size_t(total) / 4, 0);
    auto w32 = [&](uint64_t byte, uint32_t value) { out[size_t(byte) / 4] = value; };
    auto w64 = [&](uint64_t byte, uint64_t value) {
        out[size_t(byte) / 4] = uint32_t(value);
        out[size_t(byte) / 4 + 1] = uint32_t(value >> 32);
    };
    auto wdouble = [&](uint64_t byte, double value) {
        uint64_t bits = 0;
        std::memcpy(&bits, &value, 8);
        w64(byte, bits);
    };
    w32(0, uint32_t(endExclusive));
    w32(4, program.result);
    for (unsigned k = 0; k < 4; ++k)
        w32(8 + 4 * k, outputs.raw ? (k == 0 ? outputs.rawBase : 0) : outputs.regs[k]);
    w32(24, outputs.raw ? 0u : outputs.regCount);
    w32(28, outputs.raw ? ExprVkEncoding::kRaw64OutType : static_cast<unsigned>(outputs.type));
    w32(32, outputs.regCount);
    w32(36, options.count);
    w32(40, options.primitiveCount);
    w32(44, static_cast<unsigned>(options.domain));
    w32(48, uint32_t(options.seed));
    w32(52, options.descId);
    w32(56, uint32_t(table.size()));
    w32(60, uint32_t(tableOffset));
    wdouble(64, options.frame);
    wdouble(72, options.time);
    w32(80, uint32_t(codeOffset));
    w32(84, uint32_t(g3Offset));
    w32(88, uint32_t(p514Offset));
    w32(92, uint32_t(p256Offset));
    for (size_t i = 0; i < endExclusive; ++i) {
        uint64_t const base = codeOffset + uint64_t(i) * ExprVkEncoding::kInstructionBytes;
        auto it = rewriteSlot.find(i);
        if (it != rewriteSlot.end()) {
            // Already-evaluated host step: dst = LoadVariable(injection slot).
            expr::IRInstruction const& orig = program.instructions[i];
            w32(base, static_cast<unsigned>(expr::IROp::LoadVariable));
            w32(base + 4, uint32_t(orig.dst));
            w32(base + 8, 0);
            w32(base + 12, it->second);
            wdouble(base + 16, 0.0);
            continue;
        }
        expr::IRInstruction const& in = program.instructions[i];
        if (in.dst >= program.registerCount) return fail("instruction dst out of range");
        w32(base, static_cast<unsigned>(in.op));
        w32(base + 4, uint32_t(in.dst) | (uint32_t(in.a) << 16));
        w32(base + 8, uint32_t(in.b) | (uint32_t(in.c) << 16));
        w32(base + 12, static_cast<unsigned>(in.variable) |
                           (uint32_t(uint8_t(in.compare)) << 16) |
                           (uint32_t(in.component) << 24));
        wdouble(base + 16, in.immediate);
    }
    for (size_t v = 0; v < table.size(); ++v) {
        uint64_t const base = tableOffset + uint64_t(v) * ExprVkEncoding::kTableEntryBytes;
        w64(base, table[v].offsetDoubles);
        w32(base + 8, table[v].count);
        w32(base + 12, static_cast<unsigned>(table[v].domain) |
                           (uint32_t(table[v].components) << 8));
    }
    for (int i = 0; i < 514; ++i)
        for (int k = 0; k < 3; ++k)
            wdouble(g3Offset + uint64_t(i * 3 + k) * 8ull, expr::tables::kNoiseG3Host[i][k]);
    for (int i = 0; i < 514; ++i)
        w32(p514Offset + uint64_t(i) * 4ull, uint32_t(expr::tables::kNoiseP514Host[i]));
    for (int i = 0; i < 256; ++i)
        w32(p256Offset + uint64_t(i) * 4ull, expr::tables::kHashP256Host[i]);
    *words = std::move(out);
    return true;
}

} // namespace usdGen::vulkan
