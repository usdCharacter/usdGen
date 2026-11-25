// usdGen — the ONE IR interpreter, compiled for both the host and the device.
//
// gpu/expression.cu and expressions/cpuEvaluator.cpp both include this header
// and both call ExecuteElement/Store/ReadVariable from it, so the CUDA lane and
// the CPU reference lane cannot drift apart: there is a single source for the
// NaN/inf poison rules, the Compare semantics, the lazily-predicated Select,
// the Clamp ordering rule and the destination-type range checks. The SeExpr2
// builtin library IROp::Call dispatches to lives next door in
// expressions/exprMath.h, under the same single-definition rule.
//
// The parity guarantee is exact (bit-for-bit) for every op whose result is an
// IEEE-754 correctly rounded double operation: Const, LoadVariable, Move, Add,
// Sub, Mul, Div, Neg, Compare, Select, Min, Max, Clamp, Abs, Sqrt, Floor,
// Ceil, Trunc, Round, Fmod, Hypot — and for every IRFunc built only out of
// those, which is the whole noise/hash/cellnoise/voronoi/curve/spline/choose/
// pick family, because SeExpr2's lattices are pure +-*/ over a shared table.
// It is NOT guaranteed for the libm-quality transcendentals (Sin, Cos, Tan,
// Asin, Acos, Atan, Sinh, Cosh, Tanh, Exp, Log, Log10, Pow, Cbrt, Atan2),
// where the host CRT and CUDA's libdevice are each correctly rounded to within
// a fraction of an ulp but not necessarily to the same bits; nor for the few
// IRFuncs that call one (bias, contrast, gamma, gaussstep, angle, rotate, up,
// and anything that reaches them). Tests that assert bit equality across the
// two lanes must stay inside the exact set.
#ifndef USDGEN_EXPRESSIONS_IR_EXEC_H
#define USDGEN_EXPRESSIONS_IR_EXEC_H

#include "usdGen/expressions/exprMath.h"
#include "usdGen/expressions/ir.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

namespace usdGen::expr {

namespace detail {
/// IROp::Sample dispatch. Only an input type that carries a `samplers` table
/// (the host's CpuExpressionInputs) can sample; everything else, the device
/// inputs in particular, reads NaN, which poisons the result. CUDA admission
/// refuses sampler programs before they get that far.
template <class Inputs, class = void>
struct ExprSamplerHook {
    USDGEN_EXPR_HD static double Sample(Inputs const &, unsigned, double const *, unsigned,
                                        size_t, unsigned)
    {
        return NAN;
    }
};
template <class Inputs>
struct ExprSamplerHook<Inputs, std::void_t<decltype(std::declval<Inputs const &>().samplers)>> {
    static double Sample(Inputs const &in, unsigned slot, double const *args, unsigned n,
                         size_t index, unsigned component)
    {
        return in.samplers ? SampleSlot(*in.samplers, in, slot, args, n, index, component) : NAN;
    }
};
} // namespace detail

/// Register file size. The frontend allocates one register per instruction, so
/// this also bounds IRProgram::registerCount, and it is deliberately modest:
/// the CUDA kernel spends `kExprRegisters` doubles of thread-local memory on
/// every launch whether the program needs them or not. A program that does not
/// fit is refused with a diagnostic rather than silently made slower for
/// everyone. In practice it holds a curve() of ~30 knots or a ccurve() of ~8.
constexpr unsigned kExprRegisters = 256;
/// Instruction ceiling. registerCount == instruction count, so kExprRegisters
/// is the binding limit today; this is the belt to that braces.
constexpr unsigned kExprInstructions = 4096;

USDGEN_EXPR_HD inline bool ExprIsDomain(Domain d)
{
    return d == Domain::Groom || d == Domain::Primitive || d == Domain::Point;
}

/// One SeExpr2 builtin, over a contiguous argument block. `component` selects
/// the output channel of a vector-valued builtin and is 0 for the scalar ones.
USDGEN_EXPR_HD inline double ExprCall(IRFunc func, double const *a, int n, unsigned component)
{
    double v[3] = {0, 0, 0};
    switch (func) {
    // --- blending / remapping -------------------------------------------
    case IRFunc::Expand:      return ExprExpand(a[0], a[1], a[2]);
    case IRFunc::Bias:        return ExprBias(a[0], a[1]);
    case IRFunc::Contrast:    return ExprContrast(a[0], a[1]);
    case IRFunc::BoxStep:     return ExprBoxStep(a[0], a[1]);
    case IRFunc::LinearStep:  return ExprLinearStep(a[0], a[1], a[2]);
    case IRFunc::SmoothStep:  return ExprSmoothStep(a[0], a[1], a[2]);
    case IRFunc::GaussStep:   return ExprGaussStep(a[0], a[1], a[2]);
    case IRFunc::Remap:       return ExprRemap(a[0], a[1], a[2], a[3], a[4]);
    case IRFunc::Cycle:       return ExprCycle(a[0], a[1], a[2]);
    // --- noise -----------------------------------------------------------
    case IRFunc::Hash:        return ExprHash(a, n);
    case IRFunc::Noise:       ExprNoise3(a, 1, v); return .5 * v[0] + .5;
    case IRFunc::SNoise:      ExprNoise3(a, 1, v); return v[0];
    case IRFunc::VNoise:      ExprNoise3(a, 3, v); return v[component];
    case IRFunc::CNoise:      ExprNoise3(a, 3, v); return .5 * v[component] + .5;
    case IRFunc::PNoise:      return ExprPNoise3(a, a + 3);
    case IRFunc::CellNoise:   ExprCellNoise3(a, 1, v); return v[0];
    case IRFunc::CCellNoise:  ExprCellNoise3(a, 3, v); return v[component];
    case IRFunc::Fbm:
        ExprFbm3(a, 1, false, ExprInt(ExprClamp(a[3], 1, 8)), a[4], a[5], v);
        return .5 * v[0] + .5;
    case IRFunc::VFbm:
        ExprFbm3(a, 3, false, ExprInt(ExprClamp(a[3], 1, 8)), a[4], a[5], v);
        return v[component];
    case IRFunc::CFbm:
        ExprFbm3(a, 3, false, ExprInt(ExprClamp(a[3], 1, 8)), a[4], a[5], v);
        return v[component] * .5 + .5;
    case IRFunc::Turbulence:
        ExprFbm3(a, 1, true, ExprInt(ExprClamp(a[3], 1, 8)), a[4], a[5], v);
        return .5 * v[0] + .5;
    case IRFunc::VTurbulence:
        ExprFbm3(a, 3, true, ExprInt(ExprClamp(a[3], 1, 8)), a[4], a[5], v);
        return v[component];
    case IRFunc::CTurbulence:
        ExprFbm3(a, 3, true, ExprInt(ExprClamp(a[3], 1, 8)), a[4], a[5], v);
        return v[component] * .5 + .5;
    case IRFunc::Voronoi:  return ExprVoronoi(a, a[3], a[4], a[5], a[6], a[7], a[8]);
    case IRFunc::CVoronoi: ExprCVoronoi(a, a[3], a[4], a[5], a[6], a[7], a[8], v); return v[component];
    case IRFunc::PVoronoi: ExprPVoronoi(a, a[3], a[4], a[5], a[6], a[7], v); return v[component];
    // --- vectors ---------------------------------------------------------
    case IRFunc::Dist:     return ExprDist(a, a + 3);
    case IRFunc::Length:   return ExprLength3(a);
    case IRFunc::Dot:      return ExprDot(a, a + 3);
    case IRFunc::Cross:    ExprCross(a, a + 3, v); return v[component];
    case IRFunc::Norm:     ExprNorm(a, v); return v[component];
    case IRFunc::Angle:    return ExprAngle(a, a + 3);
    case IRFunc::Ortho:    ExprOrtho(a, a + 3, v); return v[component];
    case IRFunc::Rotate:   ExprRotate(a, a + 3, a[6], v); return v[component];
    case IRFunc::Up:       ExprUp(a, a + 3, v); return v[component];
    // --- colour ----------------------------------------------------------
    case IRFunc::RgbToHsl: ExprRgbToHsl(a, v); return v[component];
    case IRFunc::HslToRgb: ExprHslToRgb(a, v); return v[component];
    case IRFunc::Saturate: ExprSaturate(a, a[3], v); return v[component];
    case IRFunc::Hsi:      ExprHsi(a, a[3], a[4], a[5], a[6], v); return v[component];
    case IRFunc::MidHsi:   ExprMidHsi(a, a[3], a[4], a[5], a[6], a[7], a[8], v); return v[component];
    // --- variations and control curves -----------------------------------
    case IRFunc::Spline:   return ExprSpline(a, n);
    case IRFunc::Choose:   return ExprChoose(a, n);
    case IRFunc::WChoose:  return ExprWChoose(a, n);
    case IRFunc::Pick:     return ExprPick(a, n);
    case IRFunc::Curve:    return ExprCurveEval(a, n, 4, 0);
    case IRFunc::CCurve:   return ExprCurveEval(a, n, 8, int(component));
    default: return NAN;
    }
}

/// Host-side program admission. Shared so the CPU lane refuses exactly the
/// programs the device refuses, before either allocates anything.
inline bool ValidProgram(IRProgram const &program)
{
    if (program.instructions.empty() || program.instructions.size() > kExprInstructions ||
        program.registerCount == 0 || program.registerCount > kExprRegisters ||
        program.result >= program.registerCount || program.outputCount > 4 ||
        program.valueComponents == 0 || program.valueComponents > 4 ||
        (program.outputCount && program.outputCount != program.valueComponents))
        return false;
    bool written[kExprRegisters]{};
    for (auto const &op : program.instructions) {
        if (op.dst >= program.registerCount || written[op.dst]) return false;
        auto defined = [&](uint16_t r) { return r < program.registerCount && written[r]; };
        switch (op.op) {
        case IROp::Const: break;
        case IROp::LoadVariable:
            if (op.variable == Variable::Invalid || op.variable >= Variable::CountVariables ||
                op.component > 3 ||
                (op.variable >= Variable::Frame && op.variable <= Variable::DescId && op.component != 0))
                return false;
            break;
        case IROp::Neg: case IROp::Abs: case IROp::Sin: case IROp::Cos:
        case IROp::Sqrt: case IROp::Exp: case IROp::Log: case IROp::Floor:
        case IROp::Ceil: case IROp::Tan: case IROp::Move: case IROp::Asin:
        case IROp::Acos: case IROp::Atan: case IROp::Sinh: case IROp::Cosh:
        case IROp::Tanh: case IROp::Log10: case IROp::Cbrt: case IROp::Round:
        case IROp::Trunc:
            if (!defined(op.a)) return false;
            break;
        case IROp::Add: case IROp::Sub: case IROp::Mul: case IROp::Div:
        case IROp::Min: case IROp::Max: case IROp::Pow: case IROp::Compare:
        case IROp::Fmod: case IROp::Atan2: case IROp::Hypot:
            if (!defined(op.a) || !defined(op.b) ||
                (op.op == IROp::Compare && op.compare != '<' && op.compare != '>' &&
                 op.compare != 'l' && op.compare != 'g' && op.compare != '=' &&
                 op.compare != '!'))
                return false;
            break;
        case IROp::Select: case IROp::Clamp:
            if (!defined(op.a) || !defined(op.b) || !defined(op.c)) return false;
            break;
        case IROp::Call: {
            const IRFunc func = static_cast<IRFunc>(op.c);
            if (func == IRFunc::Invalid || func >= IRFunc::CountFuncs) return false;
            const IRFuncSpec spec = IRFuncSpecOf(func);
            if (spec.outComponents == 0) return false;
            if (op.b < spec.minArgs || (spec.maxArgs != 0xFFFFu && op.b > spec.maxArgs))
                return false;
            if (op.component >= spec.outComponents) return false;
            if (unsigned(op.a) + unsigned(op.b) > program.registerCount) return false;
            for (unsigned i = 0; i < op.b; ++i)
                if (!written[op.a + i]) return false;
            break;
        }
        case IROp::Sample: {
            if (op.c >= program.samplers.size()) return false;
            IRSampler const &sampler = program.samplers[op.c];
            if (sampler.components == 0 || op.component >= sampler.components) return false;
            const unsigned expected = sampler.kind == SamplerKind::Geometry ? 3u : 0u;
            if (op.b != expected) return false;
            if (sampler.kind == SamplerKind::Geometry &&
                (!sampler.element || !ValidProgram(*sampler.element) ||
                 !sampler.element->samplers.empty()))
                return false;
            if (unsigned(op.a) + unsigned(op.b) > program.registerCount) return false;
            for (unsigned i = 0; i < op.b; ++i)
                if (!written[op.a + i]) return false;
            break;
        }
        default: return false;
        }
        written[op.dst] = true;
    }
    if (!written[program.result]) return false;
    for (unsigned c = 0; c < program.outputCount; ++c)
        if (program.output[c] >= program.registerCount || !written[program.output[c]]) return false;
    return true;
}

/// One variable read. `Inputs` is gpu::ExpressionInputs on the device and
/// CpuExpressionInputs on the host; both expose the same member names, so the
/// indexing rule (groom broadcasts, a primitive field at point rate goes
/// through the ownership map, a mismatched domain poisons) has one definition.
template <class Inputs>
USDGEN_EXPR_HD double ReadVariable(Variable variable, unsigned component,
                                   Inputs const &in, size_t index)
{
    switch (variable) {
    case Variable::Frame: return in.context.frame;
    case Variable::Time: return in.context.time;
    case Variable::Index: return double(index);
    case Variable::Count: return double(in.count);
    case Variable::Seed: return double(in.context.seed);
    case Variable::DescId: return double(in.context.descId);
    default: break;
    }
    const auto v = static_cast<unsigned>(variable);
    if (v == 0 || v >= static_cast<unsigned>(Variable::CountVariables)) return NAN;
    auto const &field = in.fields[v];
    if (!field.data || component >= field.components) return NAN;
    size_t element = index;
    if (field.domain == Domain::Groom) element = 0;
    else if (field.domain == Domain::Primitive && in.context.domain == Domain::Point) {
        if (!in.pointToPrimitive.data || index >= in.pointToPrimitive.size) return NAN;
        element = in.pointToPrimitive.data[index];
    } else if (field.domain != in.context.domain) return NAN;
    return element < field.count ? field.data[element * field.components + component] : NAN;
}

#if !defined(__CUDACC__)
/// Host double -> IEEE-754 binary16, round-to-nearest-even, matching CUDA's
/// __double2half. Only finite inputs reach this (Store rejects the rest);
/// an overflow returns the infinity encoding so the caller's finiteness check
/// rejects the value exactly as the device does.
inline uint16_t DoubleToHalfBits(double value)
{
    const uint16_t sign = value < 0.0 || (value == 0.0 && std::signbit(value)) ? 0x8000u : 0u;
    const double magnitude = std::fabs(value);
    if (magnitude == 0.0) return sign;
    int exponent = std::ilogb(magnitude);
    if (exponent < -14) exponent = -15;               // subnormal, fixed exponent
    double scaled = std::nearbyint(std::ldexp(magnitude, 10 - exponent));
    if (scaled >= 2048.0) { scaled *= 0.5; ++exponent; }
    if (exponent > 15) return uint16_t(sign | 0x7C00u);   // overflow -> inf
    const uint16_t mantissa = static_cast<uint16_t>(static_cast<uint32_t>(scaled) & 0x3FFu);
    if (exponent == -15 && scaled < 1024.0) return uint16_t(sign | mantissa);
    return uint16_t(sign | uint16_t((exponent + 15) << 10) | mantissa);
}
#endif

/// Destination conversion and range admission. `Output` exposes data/type/
/// components. Returns false for any value the destination cannot represent
/// exactly; the caller turns that into the lane's InvalidValue diagnostic.
template <class Output>
USDGEN_EXPR_HD bool Store(double value, Output const &out, size_t i)
{
    if (!ExprFinite(value)) return false;
    switch (out.type) {
    case ScalarType::Bool:
        if (value != 0.0 && value != 1.0) return false;
        static_cast<unsigned char *>(out.data)[i] = value != 0; return true;
    case ScalarType::Int32:
        if (USDGEN_EXPR_FN(trunc)(value) != value || value < double(INT_MIN) ||
            value > double(INT_MAX)) return false;
        static_cast<int32_t *>(out.data)[i] = static_cast<int32_t>(value); return true;
    case ScalarType::UInt32:
        if (USDGEN_EXPR_FN(trunc)(value) != value || value < 0.0 ||
            value > double(UINT_MAX)) return false;
        static_cast<uint32_t *>(out.data)[i] = static_cast<uint32_t>(value); return true;
    case ScalarType::Int64:
        // SeExpr's double arithmetic is exact only through 2^53. Do not
        // suggest that an arbitrary USD int64 was evaluated losslessly.
        if (USDGEN_EXPR_FN(trunc)(value) != value ||
            USDGEN_EXPR_FN(fabs)(value) > 9007199254740992.0) return false;
        static_cast<int64_t *>(out.data)[i] = static_cast<int64_t>(value); return true;
    case ScalarType::UInt64:
        if (USDGEN_EXPR_FN(trunc)(value) != value || value < 0.0 ||
            value > 9007199254740992.0) return false;
        static_cast<uint64_t *>(out.data)[i] = static_cast<uint64_t>(value); return true;
    case ScalarType::Float16: {
#if defined(__CUDACC__)
        const __half converted = __double2half(value);
        if (!ExprFinite(double(__half2float(converted)))) return false;
        static_cast<__half *>(out.data)[i] = converted; return true;
#else
        const uint16_t bits = DoubleToHalfBits(value);
        if ((bits & 0x7C00u) == 0x7C00u) return false;   // inf/NaN encoding
        static_cast<uint16_t *>(out.data)[i] = bits; return true;
#endif
    }
    case ScalarType::Float32: {
        const float converted = static_cast<float>(value);
        if (!ExprFinite(double(converted))) return false;
        static_cast<float *>(out.data)[i] = converted; return true;
    }
    case ScalarType::Float64: static_cast<double *>(out.data)[i] = value; return true;
    default: return false;
    }
}

/// Runs the whole program for one element into `r` (at least registerCount
/// entries). Pure predication: a dead conditional arm may carry NaN, but only
/// the selected result is ever validated and published, and no instruction has
/// a side effect.
template <class Inputs>
USDGEN_EXPR_HD void ExecuteElement(IRInstruction const *code, size_t count,
                                   double *r, Inputs const &in, size_t index)
{
    for (size_t pc = 0; pc < count; ++pc) {
        IRInstruction const &instruction = code[pc];
        double value = NAN;
        const unsigned a = instruction.a, b = instruction.b, c = instruction.c;
        switch (instruction.op) {
        case IROp::Const: value = instruction.immediate; break;
        case IROp::LoadVariable:
            value = ReadVariable(instruction.variable, instruction.component, in, index);
            break;
        case IROp::Move: value = r[a]; break;
        case IROp::Add: value = r[a] + r[b]; break;
        case IROp::Sub: value = r[a] - r[b]; break;
        case IROp::Mul: value = r[a] * r[b]; break;
        case IROp::Div: value = r[a] / r[b]; break;
        case IROp::Neg: value = -r[a]; break;
        case IROp::Compare:
            // Invalid operands must not be turned into an apparently valid
            // boolean. Poison propagates until a conditional selects a value.
            if (!ExprFinite(r[a]) || !ExprFinite(r[b])) break;
            switch (instruction.compare) {
            case '<': value = r[a] < r[b]; break;
            case '>': value = r[a] > r[b]; break;
            case 'l': value = r[a] <= r[b]; break;
            case 'g': value = r[a] >= r[b]; break;
            case '=': value = r[a] == r[b]; break;
            case '!': value = r[a] != r[b]; break;
            }
            break;
        case IROp::Select:
            if (ExprFinite(r[a])) value = r[a] != 0.0 ? r[b] : r[c];
            break;
        case IROp::Min:
            if (ExprFinite(r[a]) && ExprFinite(r[b])) value = USDGEN_EXPR_FN(fmin)(r[a], r[b]);
            break;
        case IROp::Max:
            if (ExprFinite(r[a]) && ExprFinite(r[b])) value = USDGEN_EXPR_FN(fmax)(r[a], r[b]);
            break;
        case IROp::Clamp:
            if (ExprFinite(r[a]) && ExprFinite(r[b]) && ExprFinite(r[c]) && r[b] <= r[c])
                value = USDGEN_EXPR_FN(fmin)(r[c], USDGEN_EXPR_FN(fmax)(r[b], r[a]));
            break;
        case IROp::Abs: value = USDGEN_EXPR_FN(fabs)(r[a]); break;
        case IROp::Sin: value = USDGEN_EXPR_FN(sin)(r[a]); break;
        case IROp::Cos: value = USDGEN_EXPR_FN(cos)(r[a]); break;
        case IROp::Pow: value = USDGEN_EXPR_FN(pow)(r[a], r[b]); break;
        case IROp::Sqrt: value = USDGEN_EXPR_FN(sqrt)(r[a]); break;
        case IROp::Exp: value = USDGEN_EXPR_FN(exp)(r[a]); break;
        case IROp::Log: value = USDGEN_EXPR_FN(log)(r[a]); break;
        case IROp::Floor: value = USDGEN_EXPR_FN(floor)(r[a]); break;
        case IROp::Ceil: value = USDGEN_EXPR_FN(ceil)(r[a]); break;
        case IROp::Tan: value = USDGEN_EXPR_FN(tan)(r[a]); break;
        case IROp::Fmod: value = USDGEN_EXPR_FN(fmod)(r[a], r[b]); break;
        case IROp::Atan2: value = USDGEN_EXPR_FN(atan2)(r[a], r[b]); break;
        case IROp::Asin: value = USDGEN_EXPR_FN(asin)(r[a]); break;
        case IROp::Acos: value = USDGEN_EXPR_FN(acos)(r[a]); break;
        case IROp::Atan: value = USDGEN_EXPR_FN(atan)(r[a]); break;
        case IROp::Sinh: value = USDGEN_EXPR_FN(sinh)(r[a]); break;
        case IROp::Cosh: value = USDGEN_EXPR_FN(cosh)(r[a]); break;
        case IROp::Tanh: value = USDGEN_EXPR_FN(tanh)(r[a]); break;
        case IROp::Log10: value = USDGEN_EXPR_FN(log10)(r[a]); break;
        case IROp::Cbrt: value = USDGEN_EXPR_FN(cbrt)(r[a]); break;
        case IROp::Round: value = ExprRound(r[a]); break;
        case IROp::Trunc: value = USDGEN_EXPR_FN(trunc)(r[a]); break;
        case IROp::Hypot: value = ExprHypot(r[a], r[b]); break;
        case IROp::Call:
            value = ExprCall(static_cast<IRFunc>(c), r + a, int(b), instruction.component);
            break;
        case IROp::Sample:
            value = detail::ExprSamplerHook<Inputs>::Sample(in, c, r + a, b, index,
                                                            instruction.component);
            break;
        }
        r[instruction.dst] = value;
    }
}

} // namespace usdGen::expr
#endif
