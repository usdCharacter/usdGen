#ifndef USDGEN_EXPRESSIONS_IR_H
#define USDGEN_EXPRESSIONS_IR_H
#include "usdGen/expressions/context.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace usdGen::expr {
// Appended only: the numeric values are the uploaded device instruction
// encoding, so an existing op never changes its slot.
//
// Operand convention
//   most ops        a/b/c are source registers, dst is the destination
//   Move            dst = r[a]; used only to gather a Call's arguments into
//                   consecutive registers, and never common-subexpression
//                   eliminated, because a Call reads its block by range
//   Call            a = first register of the argument block
//                   b = number of registers in that block
//                   c = the IRFunc to run
//                   component = which component of a vector-valued builtin
//   Sample          a/b = the argument block, exactly as for Call
//                   c = the slot in IRProgram::samplers
//                   component = which component of the sampled value
//                   Host lane only: it reads external data (geometry, maps)
//                   through the evaluator's sampler table. The device has no
//                   such table, so CUDA admission refuses any program that
//                   contains one (IRProgram::samplers is non-empty).
enum class IROp : uint8_t { Const, LoadVariable, Add, Sub, Mul, Div, Neg, Compare, Select, Min, Max, Clamp, Abs, Sin, Cos, Pow,
                            Sqrt, Exp, Log, Floor, Ceil, Fmod, Tan, Atan2,
                            Move, Asin, Acos, Atan, Sinh, Cosh, Tanh, Log10, Cbrt, Round, Trunc, Hypot, Call,
                            Sample };

/// The SeExpr2 builtins that IROp::Call dispatches to. Every one of them has a
/// single `__host__ __device__` implementation in expressions/exprMath.h, so
/// the CUDA lane and the CPU reference lane run the same arithmetic.
///
/// The frontend normalises each call site to the fixed flat argument layout
/// below, filling SeExpr2's optional-argument defaults as immediates, so the
/// interpreter never has to reason about arity.
enum class IRFunc : uint16_t {
    Invalid,
    // --- blending / remapping (scalar in, scalar out) ---
    Expand,        // (x, lo, hi)
    Bias,          // (x, b)
    Contrast,      // (x, c)
    BoxStep,       // (x, a)
    LinearStep,    // (x, a, b)
    SmoothStep,    // (x, a, b)
    GaussStep,     // (x, a, b)
    Remap,         // (x, source, range, falloff, interp)
    Cycle,         // (index, lo, hi)
    // --- noise. Vector spellings carry 3 output components. ---
    Hash,          // (s0, ...)            variadic, >= 1
    Noise,         // (x, y, z)
    SNoise,        // (x, y, z)
    VNoise,        // (x, y, z)            -> 3
    CNoise,        // (x, y, z)            -> 3
    PNoise,        // (x, y, z, px, py, pz)
    CellNoise,     // (x, y, z)
    CCellNoise,    // (x, y, z)            -> 3
    Fbm,           // (x, y, z, octaves, lacunarity, gain)
    VFbm,          // same                 -> 3
    CFbm,          // same                 -> 3
    Turbulence,    // same
    VTurbulence,   // same                 -> 3
    CTurbulence,   // same                 -> 3
    Voronoi,       // (x,y,z, type, jitter, fbmScale, fbmOctaves, fbmLacunarity, fbmGain)
    CVoronoi,      // same                 -> 3
    PVoronoi,      // (x,y,z, jitter, fbmScale, fbmOctaves, fbmLacunarity, fbmGain) -> 3
    // --- vectors ---
    Dist,          // (ax,ay,az, bx,by,bz)
    Length,        // (x, y, z)
    Dot,           // (ax,ay,az, bx,by,bz)
    Cross,         // (ax,ay,az, bx,by,bz) -> 3
    Norm,          // (x, y, z)            -> 3
    Angle,         // (ax,ay,az, bx,by,bz)
    Ortho,         // (ax,ay,az, bx,by,bz) -> 3
    Rotate,        // (px,py,pz, ax,ay,az, angle) -> 3
    Up,            // (px,py,pz, ux,uy,uz) -> 3
    // --- colour ---
    RgbToHsl,      // (r, g, b)            -> 3
    HslToRgb,      // (h, s, l)            -> 3
    Saturate,      // (r, g, b, amount)    -> 3
    Hsi,           // (r,g,b, h, s, i, map) -> 3
    MidHsi,        // (r,g,b, h, s, i, map, falloff, interp) -> 3
    // --- variations and control curves ---
    Spline,        // (param, y1, y2, ...) variadic, >= 5
    Choose,        // (index, c1, c2, ...) variadic, >= 3
    WChoose,       // (index, c1, w1, ...) variadic, >= 5
    Pick,          // (index, lo, hi, w...) variadic, >= 3
    Curve,         // (param, cvCount, [pos, interp, val, deriv] * n)
    CCurve,        // (param, cvCount, [pos, interp, r,g,b, dr,dg,db] * n) -> 3
    CountFuncs
};

/// What IROp::Call admits for one IRFunc. Host-side only: it is what
/// ValidProgram checks and what the frontend normalises against.
struct IRFuncSpec {
    uint16_t minArgs = 0;
    uint16_t maxArgs = 0;
    uint8_t outComponents = 1;
};
IRFuncSpec IRFuncSpecOf(IRFunc func) noexcept;
/// The spelling used in diagnostics; "" for an out-of-range value.
const char *IRFuncName(IRFunc func) noexcept;

namespace detail {
struct Row { IRFunc func; uint16_t minArgs; uint16_t maxArgs; uint8_t out; const char *name; };
// One row per IRFunc. The frontend normalises every call site to this shape and
// ValidProgram refuses any instruction that does not match it, so a malformed
// program cannot reach either lane's interpreter.
inline constexpr uint16_t kVariadic = 0xFFFFu;
inline constexpr Row kRows[] = {
    {IRFunc::Expand,       3, 3, 1, "expand"},
    {IRFunc::Bias,         2, 2, 1, "bias"},
    {IRFunc::Contrast,     2, 2, 1, "contrast"},
    {IRFunc::BoxStep,      2, 2, 1, "boxstep"},
    {IRFunc::LinearStep,   3, 3, 1, "linearstep"},
    {IRFunc::SmoothStep,   3, 3, 1, "smoothstep"},
    {IRFunc::GaussStep,    3, 3, 1, "gaussstep"},
    {IRFunc::Remap,        5, 5, 1, "remap"},
    {IRFunc::Cycle,        3, 3, 1, "cycle"},
    {IRFunc::Hash,         1, kVariadic, 1, "hash"},
    {IRFunc::Noise,        3, 3, 1, "noise"},
    {IRFunc::SNoise,       3, 3, 1, "snoise"},
    {IRFunc::VNoise,       3, 3, 3, "vnoise"},
    {IRFunc::CNoise,       3, 3, 3, "cnoise"},
    {IRFunc::PNoise,       6, 6, 1, "pnoise"},
    {IRFunc::CellNoise,    3, 3, 1, "cellnoise"},
    {IRFunc::CCellNoise,   3, 3, 3, "ccellnoise"},
    {IRFunc::Fbm,          6, 6, 1, "fbm"},
    {IRFunc::VFbm,         6, 6, 3, "vfbm"},
    {IRFunc::CFbm,         6, 6, 3, "cfbm"},
    {IRFunc::Turbulence,   6, 6, 1, "turbulence"},
    {IRFunc::VTurbulence,  6, 6, 3, "vturbulence"},
    {IRFunc::CTurbulence,  6, 6, 3, "cturbulence"},
    {IRFunc::Voronoi,      9, 9, 1, "voronoi"},
    {IRFunc::CVoronoi,     9, 9, 3, "cvoronoi"},
    {IRFunc::PVoronoi,     8, 8, 3, "pvoronoi"},
    {IRFunc::Dist,         6, 6, 1, "dist"},
    {IRFunc::Length,       3, 3, 1, "length"},
    {IRFunc::Dot,          6, 6, 1, "dot"},
    {IRFunc::Cross,        6, 6, 3, "cross"},
    {IRFunc::Norm,         3, 3, 3, "norm"},
    {IRFunc::Angle,        6, 6, 1, "angle"},
    {IRFunc::Ortho,        6, 6, 3, "ortho"},
    {IRFunc::Rotate,       7, 7, 3, "rotate"},
    {IRFunc::Up,           6, 6, 3, "up"},
    {IRFunc::RgbToHsl,     3, 3, 3, "rgbtohsl"},
    {IRFunc::HslToRgb,     3, 3, 3, "hsltorgb"},
    {IRFunc::Saturate,     4, 4, 3, "saturate"},
    {IRFunc::Hsi,          7, 7, 3, "hsi"},
    {IRFunc::MidHsi,       9, 9, 3, "midhsi"},
    {IRFunc::Spline,       5, kVariadic, 1, "spline"},
    {IRFunc::Choose,       3, kVariadic, 1, "choose"},
    {IRFunc::WChoose,      5, kVariadic, 1, "wchoose"},
    {IRFunc::Pick,         3, kVariadic, 1, "pick"},
    {IRFunc::Curve,       10, kVariadic, 1, "curve"},
    {IRFunc::CCurve,      18, kVariadic, 3, "ccurve"},
};
inline const Row *Find(IRFunc func) noexcept
{
    for (auto const &row : kRows) if (row.func == func) return &row;
    return nullptr;
}
} // namespace detail

inline IRFuncSpec IRFuncSpecOf(IRFunc func) noexcept
{
    if (const detail::Row *row = detail::Find(func)) return {row->minArgs, row->maxArgs, row->out};
    return {};
}

inline const char *IRFuncName(IRFunc func) noexcept
{
    const detail::Row *row = detail::Find(func);
    return row ? row->name : "";
}

struct IRInstruction { IROp op=IROp::Const; uint16_t dst=0,a=0,b=0,c=0; Variable variable=Variable::Invalid; double immediate=0; char compare=0; uint8_t component=0; };

struct IRProgram;

/// What an IROp::Sample reads. Every string is a compile-time constant of the
/// call site; the evaluator resolves `input` against the expression prim's
/// `input:<name>` relationships.
enum class SamplerKind : uint8_t {
    Geometry,   // geoSampler(input, expression [, iterate [, reduce [, query]]])
    Ptex        // ptex(input): a UsdGenPtexMap read at the strand root
};
/// Which elements of the input geometry a geoSampler() visits.
enum class SampleIterate : uint8_t { Point, Prim, Geometry };
/// How a geoSampler() folds the per-element values into one.
enum class SampleReduce : uint8_t {
    Nearest,    // the element nearest the query position
    Nearest2,   // the second-nearest element (voronoi borders: f2 - f1)
    Min, Max, Sum, Mean
};
struct IRSampler {
    SamplerKind kind = SamplerKind::Geometry;
    std::string input;                     // name of the input:<name> relationship
    std::string source;                    // geometry: the per-element expression
    SampleIterate iterate = SampleIterate::Prim;
    SampleReduce reduce = SampleReduce::Nearest;
    uint8_t components = 1;                // width of the sampled value (1 or 3)
    // Geometry: the element expression, compiled at point rate with $Q/$Qdist
    // admitted. Shared, immutable.
    std::shared_ptr<const IRProgram> element;
};
const char *SampleIterateName(SampleIterate iterate) noexcept;
const char *SampleReduceName(SampleReduce reduce) noexcept;

struct IRProgram {
    std::vector<IRInstruction> instructions;
    uint16_t result=0;
    uint16_t output[4]{};
    uint8_t outputCount=0;
    uint8_t valueComponents=1;
    uint16_t registerCount=0;
    bool hasLazyBranches=false;
    // One entry per distinct sampler call site (IROp::Sample's c). Empty for
    // every program the device lane can run.
    std::vector<IRSampler> samplers;
};

inline const char *SampleIterateName(SampleIterate iterate) noexcept
{
    switch (iterate) {
    case SampleIterate::Point: return "point";
    case SampleIterate::Prim: return "prim";
    case SampleIterate::Geometry: return "geometry";
    }
    return "";
}
inline const char *SampleReduceName(SampleReduce reduce) noexcept
{
    switch (reduce) {
    case SampleReduce::Nearest: return "nearest";
    case SampleReduce::Nearest2: return "nearest2";
    case SampleReduce::Min: return "min";
    case SampleReduce::Max: return "max";
    case SampleReduce::Sum: return "sum";
    case SampleReduce::Mean: return "mean";
    }
    return "";
}
}
#endif
