// usdGen — the function library against upstream SeExpr2, which is the
// definition of "a host groomer parity" for everything except a host groomer's own rand().
//
// expressions/exprMath.h re-implements SeExpr2's builtins as `__host__
// __device__` code so the CUDA lane and the CPU reference lane can share ONE
// implementation. That re-implementation is only worth anything if it agrees
// with the original, so this test links the vendored SeExpr2 frontend
// (thirdparty/seexprFrontend, the same archive the parser comes from) and
// compares:
//
//   1. exprMath.h's noise lattices against SeExpr2's Noise.cpp templates;
//   2. a COMPILED usdGen expression, evaluated through the real IR
//      interpreter, against SeExpr2's builtin for the same arguments.
//
// (2) is the interesting one: it also proves the frontend normalises optional
// arguments to SeExpr2's defaults, lays out vector arguments the way the
// interpreter reads them, and prepares control-curve knots the way
// SeExpr2::Curve does.
//
// Both sides run on the host here, so agreement is asserted as EXACT double
// equality, not a tolerance. Anything that is only near-equal would mean the
// transcription changed an operation or its order. Where usdGen deliberately
// differs from SeExpr2 the case is listed at the bottom of main() with the
// reason rather than quietly loosened.
//
// Host/device agreement is a separate claim, proved by
// testUsdGenCudaExpressionParity.
#include "usdGenExpr/expressions/cpuEvaluator.h"
#include "usdGenExpr/expressions/exprMath.h"
#include "usdGenExpr/expressions/frontend.h"
#include "usdGenExpr/expressions/irExec.h"

#include <ExprBuiltins.h>
#include <Noise.h>
#include <Vec.h>
#include <Curve.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// Declared in ExprBuiltins.cpp but not in its header, and not file-static, so
// they link. Spelled through the SeExpr2 macro the archive was built with.
namespace SeExpr2 {
Vec3d rotate(int n, const Vec3d *args);
Vec3d saturate(int n, const Vec3d *args);
}

namespace SE = UsdGenSeExprFrontend;
using namespace usdGen::expr;

namespace {

int g_failures = 0;

void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
}

void Same(double mine, double theirs, std::string const &what)
{
    // Exact: both sides are the same arithmetic on the same host.
    if (mine != theirs) {
        ++g_failures;
        std::printf("FAIL: %s\n  usdGen   %.17g\n  SeExpr2  %.17g\n", what.c_str(), mine, theirs);
    }
}

/// Compiles `source` at groom rate into a float64 destination and runs it
/// through the production interpreter. `literal` is what $value reads.
double Evaluate(char const *source, double literal = 0.0)
{
    auto program = Frontend::Compile(source, {Domain::Groom, ScalarType::Float64, 1});
    if (!program.ok) {
        ++g_failures;
        std::printf("FAIL: does not compile: %s\n  %s\n", source,
                    program.diagnostics.empty() ? "" : program.diagnostics[0].c_str());
        return NAN;
    }
    CpuExpressionContext context;
    CpuCurveGeometryView geometry;
    if (context.Build(geometry, Context{0, 0, 0, 1, 0, 0, Domain::Groom}, nullptr) !=
        CpuExpressionStatus::Ok) {
        ++g_failures;
        std::printf("FAIL: groom context build for %s\n", source);
        return NAN;
    }
    auto inputs = context.Inputs();
    inputs.fields[unsigned(Variable::Value)] = {&literal, 1, Domain::Groom, 1};
    double out = NAN;
    if (EvaluateProgram(program.program.IR(), inputs, {&out, 1, ScalarType::Float64, 1}) !=
        CpuExpressionStatus::Ok) {
        ++g_failures;
        std::printf("FAIL: evaluation refused: %s\n", source);
        return NAN;
    }
    return out;
}

/// 17 significant digits round-trips a double exactly, so the oracle and the
/// compiled expression really do see the same number.
std::string Number(double v)
{
    std::ostringstream text;
    text << std::setprecision(17) << v;
    return text.str();
}
std::string Triple(SE::Vec3d const &v)
{
    return "[" + Number(v[0]) + ", " + Number(v[1]) + ", " + Number(v[2]) + "]";
}

// ---------------------------------------------------------------------------
// 1. The lattices, straight against Noise.cpp.
// ---------------------------------------------------------------------------
void CheckLattices()
{
    const double samples[][3] = {
        {0, 0, 0}, {0.5, 0.25, 0.125}, {-1.75, 3.25, 0.5}, {12.3, -4.7, 8.9},
        {1e-4, 1e4, -1e4}, {0.9999999, 1.0000001, -0.0000001},
    };
    for (auto const &p : samples) {
        const std::string where = " at " + Triple(SE::Vec3d(p[0], p[1], p[2]));
        double mine1 = 0, theirs1 = 0;
        ExprNoise3(p, 1, &mine1);
        SE::Noise<3, 1>(p, &theirs1);
        Same(mine1, theirs1, "Noise<3,1>" + where);

        double mine3[3] = {0, 0, 0}, theirs3[3] = {0, 0, 0};
        ExprNoise3(p, 3, mine3);
        SE::Noise<3, 3>(p, theirs3);
        for (int k = 0; k < 3; ++k)
            Same(mine3[k], theirs3[k], "Noise<3,3>[" + std::to_string(k) + "]" + where);

        double mineCell1 = 0, theirsCell1 = 0;
        ExprCellNoise3(p, 1, &mineCell1);
        SE::CellNoise<3, 1>(p, &theirsCell1);
        Same(mineCell1, theirsCell1, "CellNoise<3,1>" + where);

        double mineCell3[3] = {0, 0, 0}, theirsCell3[3] = {0, 0, 0};
        ExprCellNoise3(p, 3, mineCell3);
        SE::CellNoise<3, 3>(p, theirsCell3);
        for (int k = 0; k < 3; ++k)
            Same(mineCell3[k], theirsCell3[k], "CellNoise<3,3>[" + std::to_string(k) + "]" + where);

        const int period[3] = {4, 5, 6};
        const double periodArgs[3] = {4, 5, 6};
        double minePeriodic = ExprPNoise3(p, periodArgs), theirsPeriodic = 0;
        SE::PNoise<3, 1>(p, period, &theirsPeriodic);
        Same(minePeriodic, theirsPeriodic, "PNoise<3,1>" + where);

        for (int octaves : {1, 3, 6, 8}) {
            const double lacunarity = 2.0, gain = 0.5;
            double mineFbm = 0, theirsFbm = 0;
            ExprFbm3(p, 1, false, octaves, lacunarity, gain, &mineFbm);
            SE::FBM<3, 1, false>(p, &theirsFbm, octaves, lacunarity, gain);
            Same(mineFbm, theirsFbm,
                 "FBM<3,1,false> octaves " + std::to_string(octaves) + where);

            double mineTurb = 0, theirsTurb = 0;
            ExprFbm3(p, 1, true, octaves, lacunarity, gain, &mineTurb);
            SE::FBM<3, 1, true>(p, &theirsTurb, octaves, lacunarity, gain);
            Same(mineTurb, theirsTurb,
                 "FBM<3,1,true> octaves " + std::to_string(octaves) + where);

            double mineV[3] = {0, 0, 0}, theirsV[3] = {0, 0, 0};
            ExprFbm3(p, 3, false, octaves, lacunarity, gain, mineV);
            SE::FBM<3, 3, false>(p, theirsV, octaves, lacunarity, gain);
            for (int k = 0; k < 3; ++k)
                Same(mineV[k], theirsV[k],
                     "FBM<3,3,false>[" + std::to_string(k) + "] octaves " +
                         std::to_string(octaves) + where);

            double mineVT[3] = {0, 0, 0}, theirsVT[3] = {0, 0, 0};
            ExprFbm3(p, 3, true, octaves, lacunarity, gain, mineVT);
            SE::FBM<3, 3, true>(p, theirsVT, octaves, lacunarity, gain);
            for (int k = 0; k < 3; ++k)
                Same(mineVT[k], theirsVT[k],
                     "FBM<3,3,true>[" + std::to_string(k) + "] octaves " +
                         std::to_string(octaves) + where);
        }
    }
    // hash() is SeExpr2's own standalone hash, not the lattice hash.
    for (int n = 1; n <= 4; ++n) {
        double args[4] = {1.0, -2.5, 1e6, 0.0009765625};
        Same(ExprHash(args, n), SE::hash(n, args), "hash of " + std::to_string(n) + " seeds");
    }
}

// ---------------------------------------------------------------------------
// 2. Compiled expressions against the builtin they claim to implement.
// ---------------------------------------------------------------------------
void CheckCompiledScalars()
{
    struct Case { std::string source; double expected; std::string what; };
    const double x = 0.3125, y = 0.75, z = 2.5;
    std::vector<Case> cases;
    auto add = [&](std::string source, double expected, std::string what) {
        cases.push_back({std::move(source), expected, std::move(what)});
    };
    add("deg(" + Number(x) + ")", SE::deg(x), "deg");
    add("rad(" + Number(z) + ")", SE::rad(z), "rad");
    add("cosd(" + Number(z) + ")", SE::cosd(z), "cosd");
    add("sind(" + Number(z) + ")", SE::sind(z), "sind");
    add("tand(" + Number(z) + ")", SE::tand(z), "tand");
    add("acosd(" + Number(x) + ")", SE::acosd(x), "acosd");
    add("asind(" + Number(x) + ")", SE::asind(x), "asind");
    add("atand(" + Number(z) + ")", SE::atand(z), "atand");
    add("atan2d(" + Number(y) + ", " + Number(z) + ")", SE::atan2d(y, z), "atan2d");
    add("round(" + Number(z) + ")", SE::round(z), "round");
    add("round(-2.5)", SE::round(-2.5), "round of a negative half");
    add("invert(" + Number(x) + ")", SE::invert(x), "invert");
    add("compress(" + Number(x) + ", 2, 7)", SE::compress(x, 2, 7), "compress");
    add("expand(" + Number(z) + ", 2, 7)", SE::expand(z, 2, 7), "expand");
    add("expand(" + Number(z) + ", 2, 2)", SE::expand(z, 2, 2), "expand with lo == hi");
    add("fit(" + Number(z) + ", 0, 4, 10, 20)", SE::fit(z, 0, 4, 10, 20), "fit");
    add("gamma(" + Number(x) + ", 2.2)", SE::gamma(x, 2.2), "gamma");
    add("bias(" + Number(x) + ", 0.3)", SE::bias(x, 0.3), "bias");
    add("contrast(" + Number(x) + ", 0.7)", SE::contrast(x, 0.7), "contrast below the midpoint");
    add("contrast(" + Number(y) + ", 0.7)", SE::contrast(y, 0.7), "contrast above the midpoint");
    add("boxstep(" + Number(x) + ", 0.5)", SE::boxstep(x, 0.5), "boxstep");
    add("linearstep(" + Number(x) + ", 0, 1)", SE::linearstep(x, 0, 1), "linearstep");
    add("linearstep(" + Number(x) + ", 1, 0)", SE::linearstep(x, 1, 0), "linearstep reversed");
    add("linearstep(" + Number(x) + ", 0.5, 0.5)", SE::linearstep(x, 0.5, 0.5),
        "linearstep degenerate");
    add("smoothstep(" + Number(x) + ", 0, 1)", SE::smoothstep(x, 0, 1), "smoothstep");
    add("smoothstep(" + Number(x) + ", 1, 0)", SE::smoothstep(x, 1, 0), "smoothstep reversed");
    add("smoothstep(" + Number(x) + ", 0.5, 0.5)", SE::smoothstep(x, 0.5, 0.5),
        "smoothstep degenerate");
    add("gaussstep(" + Number(x) + ", 0, 1)", SE::gaussstep(x, 0, 1), "gaussstep");
    add("gaussstep(" + Number(x) + ", 1, 0)", SE::gaussstep(x, 1, 0), "gaussstep reversed");
    for (int interp = 0; interp <= 2; ++interp) {
        const std::string code = std::to_string(interp);
        add("remap(" + Number(z) + ", 2, 0.25, 0.75, " + code + ")",
            SE::remap(z, 2, 0.25, 0.75, interp), "remap interp " + code);
    }
    add("remap(" + Number(z) + ", 2, 0.25, 0, 0)", SE::remap(z, 2, 0.25, 0, 0),
        "remap with no falloff");
    add("mix(2, 7, " + Number(x) + ")", SE::mix(2, 7, x), "mix");
    add("cycle(7, 2, 5)", SE::cycle(7, 2, 5), "cycle");
    add("cycle(-7, 2, 5)", SE::cycle(-7, 2, 5), "cycle of a negative index");
    add("hypot(3, 4)", SE::hypot(3, 4), "hypot");

    const SE::Vec3d a(0.25, -1.5, 2.0), b(3.0, 0.5, -0.75);
    add("dist(" + Triple(a) + ", " + Triple(b) + ")",
        SE::dist(a[0], a[1], a[2], b[0], b[1], b[2]), "dist");
    add("length(" + Triple(a) + ")", SE::length(a), "length");
    add("dot(" + Triple(a) + ", " + Triple(b) + ")", SE::dot(a, b), "dot");
    add("angle(" + Triple(a) + ", " + Triple(b) + ")", SE::angle(a, b), "angle");

    {
        double params[] = {0.35, 1, 2, 3, 4, 5};
        add("spline(0.35, 1, 2, 3, 4, 5)", SE::spline(6, params), "spline");
        double atZero[] = {0, 1, 2, 3, 4, 5};
        add("spline(0, 1, 2, 3, 4, 5)", SE::spline(6, atZero), "spline at 0");
        double atOne[] = {1, 1, 2, 3, 4, 5};
        add("spline(1, 1, 2, 3, 4, 5)", SE::spline(6, atOne), "spline at 1");
    }
    {
        double params[] = {0.6, 10, 20, 30, 40};
        add("choose(0.6, 10, 20, 30, 40)", SE::choose(5, params), "choose");
    }
    {
        double params[] = {0.6, 10, 1, 20, 3, 30, 2};
        add("wchoose(0.6, 10, 1, 20, 3, 30, 2)", SE::wchoose(7, params), "wchoose");
    }
    {
        double params[] = {0.6, 3, 9};
        add("pick(0.6, 3, 9)", SE::pick(3, params), "pick");
        double weighted[] = {0.6, 3, 9, 1, 0, 4, 2};
        add("pick(0.6, 3, 9, 1, 0, 4, 2)", SE::pick(7, weighted), "pick with weights");
    }
    {
        // SeExpr2 spells the scalar noises over Vec3d argument arrays.
        const SE::Vec3d p(0.375, -2.125, 4.0);
        SE::Vec3d one[1] = {p};
        add("noise(" + Triple(p) + ")", SE::noise(1, one), "noise");
        add("snoise(" + Triple(p) + ")", SE::snoise(p), "snoise");
        add("cellnoise(" + Triple(p) + ")", SE::cellnoise(p), "cellnoise");
        const SE::Vec3d period(4, 5, 6);
        add("pnoise(" + Triple(p) + ", " + Triple(period) + ")", SE::pnoise(p, period), "pnoise");
        // Defaults first, then every optional argument spelled out.
        add("fbm(" + Triple(p) + ")", SE::fbm(1, one), "fbm with SeExpr defaults");
        add("turbulence(" + Triple(p) + ")", SE::turbulence(1, one),
            "turbulence with SeExpr defaults");
        SE::Vec3d four[4] = {p, SE::Vec3d(3.0), SE::Vec3d(2.25), SE::Vec3d(0.4)};
        add("fbm(" + Triple(p) + ", 3, 2.25, 0.4)", SE::fbm(4, four), "fbm with all arguments");
        add("turbulence(" + Triple(p) + ", 3, 2.25, 0.4)", SE::turbulence(4, four),
            "turbulence with all arguments");
    }

    for (Case const &c : cases) Same(Evaluate(c.source.c_str()), c.expected, c.what);
}

void CheckCompiledVectors()
{
    struct Case { std::string source; SE::Vec3d expected; std::string what; };
    std::vector<Case> cases;
    auto add = [&](std::string source, SE::Vec3d expected, std::string what) {
        cases.push_back({std::move(source), expected, std::move(what)});
    };
    const SE::Vec3d a(0.25, -1.5, 2.0), b(3.0, 0.5, -0.75);
    const SE::Vec3d p(0.375, -2.125, 4.0);
    const SE::Vec3d rgb(0.2, 0.6, 0.45), grey(0.5, 0.5, 0.5), hsl(0.125, 0.8, 0.4);

    add("cross(" + Triple(a) + ", " + Triple(b) + ")", SE::cross(a, b), "cross");
    add("norm(" + Triple(a) + ")", SE::norm(a), "norm");
    add("norm([0, 0, 0])", SE::norm(SE::Vec3d(0.0)), "norm of the zero vector");
    add("ortho(" + Triple(a) + ", " + Triple(b) + ")", SE::ortho(a, b), "ortho");
    {
        SE::Vec3d args[3] = {a, b, SE::Vec3d(0.6)};
        add("rotate(" + Triple(a) + ", " + Triple(b) + ", 0.6)", SE::rotate(3, args), "rotate");
    }
    add("up(" + Triple(a) + ", " + Triple(b) + ")", SE::up(a, b), "up");
    add("rgbtohsl(" + Triple(rgb) + ")", SE::rgbtohsl(rgb), "rgbtohsl");
    add("rgbtohsl(" + Triple(grey) + ")", SE::rgbtohsl(grey), "rgbtohsl of a grey");
    add("hsltorgb(" + Triple(hsl) + ")", SE::hsltorgb(hsl), "hsltorgb");
    add("hsltorgb([0.125, 0, 0.4])", SE::hsltorgb(SE::Vec3d(0.125, 0, 0.4)),
        "hsltorgb of an unsaturated colour");
    {
        SE::Vec3d args[2] = {rgb, SE::Vec3d(0.35)};
        add("saturate(" + Triple(rgb) + ", 0.35)", SE::saturate(2, args), "saturate");
    }
    {
        SE::Vec3d four[4] = {rgb, SE::Vec3d(30.0), SE::Vec3d(1.5), SE::Vec3d(0.8)};
        add("hsi(" + Triple(rgb) + ", 30, 1.5, 0.8)", SE::hsi(4, four), "hsi without a map");
        SE::Vec3d five[5] = {rgb, SE::Vec3d(30.0), SE::Vec3d(1.5), SE::Vec3d(0.8), SE::Vec3d(0.4)};
        add("hsi(" + Triple(rgb) + ", 30, 1.5, 0.8, 0.4)", SE::hsi(5, five), "hsi with a map");
        add("midhsi(" + Triple(rgb) + ", 30, 1.5, 0.8, 0.4)", SE::midhsi(5, five),
            "midhsi with its map below the midpoint");
        SE::Vec3d above[5] = {rgb, SE::Vec3d(30.0), SE::Vec3d(1.5), SE::Vec3d(0.8), SE::Vec3d(0.9)};
        add("midhsi(" + Triple(rgb) + ", 30, 1.5, 0.8, 0.9)", SE::midhsi(5, above),
            "midhsi with its map above the midpoint");
        SE::Vec3d seven[7] = {rgb, SE::Vec3d(30.0), SE::Vec3d(1.5), SE::Vec3d(0.8),
                              SE::Vec3d(0.9), SE::Vec3d(0.5), SE::Vec3d(1.0)};
        add("midhsi(" + Triple(rgb) + ", 30, 1.5, 0.8, 0.9, 0.5, 1)", SE::midhsi(7, seven),
            "midhsi with falloff and interp");
    }
    {
        SE::Vec3d one[1] = {p};
        add("vnoise(" + Triple(p) + ")", SE::vnoise(p), "vnoise");
        add("cnoise(" + Triple(p) + ")", SE::cnoise(p), "cnoise");
        add("ccellnoise(" + Triple(p) + ")", SE::ccellnoise(p), "ccellnoise");
        add("vfbm(" + Triple(p) + ")", SE::vfbm(1, one), "vfbm with SeExpr defaults");
        add("cfbm(" + Triple(p) + ")", SE::cfbm(1, one), "cfbm with SeExpr defaults");
        add("vturbulence(" + Triple(p) + ")", SE::vturbulence(1, one), "vturbulence");
        add("cturbulence(" + Triple(p) + ")", SE::cturbulence(1, one), "cturbulence");
        SE::Vec3d four[4] = {p, SE::Vec3d(5.0), SE::Vec3d(1.75), SE::Vec3d(0.6)};
        add("vfbm(" + Triple(p) + ", 5, 1.75, 0.6)", SE::vfbm(4, four), "vfbm with all arguments");
        add("cfbm(" + Triple(p) + ", 5, 1.75, 0.6)", SE::cfbm(4, four), "cfbm with all arguments");
        add("vturbulence(" + Triple(p) + ", 5, 1.75, 0.6)", SE::vturbulence(4, four),
            "vturbulence with all arguments");
        add("cturbulence(" + Triple(p) + ", 5, 1.75, 0.6)", SE::cturbulence(4, four),
            "cturbulence with all arguments");
    }

    for (Case const &c : cases)
        for (int k = 0; k < 3; ++k) {
            const std::string source = c.source + "[" + std::to_string(k) + "]";
            Same(Evaluate(source.c_str()), c.expected[k],
                 c.what + " component " + std::to_string(k));
        }
}

// ---------------------------------------------------------------------------
// 3. curve()/ccurve() against SeExpr2::Curve, including its preparation.
// ---------------------------------------------------------------------------
void CheckCurves()
{
    struct Knot { double pos; double value; int interp; };
    const std::vector<std::vector<Knot>> shapes = {
        {{0, 0, 1}, {1, 1, 1}},
        {{0, 1, 0}, {0.5, 0.25, 0}, {1, 0, 0}},
        {{0, 1, 2}, {0.5, 0.7, 2}, {1, 0, 2}},
        {{0, 1, 3}, {0.5, 0.7, 3}, {1, 0, 3}},
        {{0, 1, 4}, {0.5, 0.7, 4}, {1, 0, 4}},
        // Out of order, and with a repeated position, so the sort and the
        // sentinel handling are exercised too.
        {{1, 0, 4}, {0.25, 0.9, 3}, {0, 1, 1}, {0.25, 0.9, 2}},
    };
    for (size_t shape = 0; shape < shapes.size(); ++shape) {
        auto const &knots = shapes[shape];
        SE::Curve<double> oracle;
        std::string source = "curve($value";
        for (Knot const &knot : knots) {
            oracle.addPoint(knot.pos, knot.value,
                            SE::Curve<double>::InterpType(knot.interp));
            source += ", " + Number(knot.pos) + ", " + Number(knot.value) + ", " +
                      std::to_string(knot.interp);
        }
        source += ")";
        oracle.preparePoints();
        for (double param : {-0.5, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5}) {
            Same(Evaluate(source.c_str(), param), oracle.getValue(param),
                 "curve shape " + std::to_string(shape) + " at " + Number(param));
        }
    }
    // The colour curve keeps each channel independent, through the same
    // preparation as the scalar one.
    SE::Curve<SE::Vec3d> colour;
    colour.addPoint(0, SE::Vec3d(1.0, 0.0, 0.25), SE::Curve<SE::Vec3d>::InterpType(4));
    colour.addPoint(0.5, SE::Vec3d(0.2, 0.7, 0.5), SE::Curve<SE::Vec3d>::InterpType(4));
    colour.addPoint(1, SE::Vec3d(0.0, 1.0, 0.75), SE::Curve<SE::Vec3d>::InterpType(4));
    colour.preparePoints();
    const std::string source =
        "ccurve($value, 0, [1, 0, 0.25], 4, 0.5, [0.2, 0.7, 0.5], 4, 1, [0, 1, 0.75], 4)";
    for (double param : {0.0, 0.3, 0.5, 0.8, 1.0})
        for (int k = 0; k < 3; ++k) {
            const std::string spelling = source + "[" + std::to_string(k) + "]";
            Same(Evaluate(spelling.c_str(), param), colour.getValue(param)[k],
                 "ccurve channel " + std::to_string(k) + " at " + Number(param));
        }
}

// ---------------------------------------------------------------------------
// 4. The places usdGen deliberately differs, asserted so they stay deliberate.
// ---------------------------------------------------------------------------
void CheckDeliberateDifferences()
{
    // clamp: SeExpr2 answers with one of the bounds when hi < lo (its body is
    // `x < lo ? lo : x > hi ? hi : x`, so 0.5 clamped to [1, 0] is 1). usdGen
    // refuses the whole evaluation instead, because an inverted range is an
    // authoring mistake and a silent answer hides it. (irExec.h IROp::Clamp.)
    Check(SE::clamp(0.5, 1, 0) == 1.0, "SeExpr2 clamp with an inverted range answers anyway");
    auto inverted = Frontend::Compile("clamp(0.5, 1, 0)", {Domain::Groom, ScalarType::Float64, 1});
    Check(inverted.ok, "an inverted clamp still compiles");
    if (inverted.ok) {
        CpuExpressionContext context;
        CpuCurveGeometryView geometry;
        context.Build(geometry, Context{0, 0, 0, 1, 0, 0, Domain::Groom}, nullptr);
        auto inputs = context.Inputs();
        double literal = 0.0, out = 0.0;
        inputs.fields[unsigned(Variable::Value)] = {&literal, 1, Domain::Groom, 1};
        Check(EvaluateProgram(inverted.program.IR(), inputs,
                              {&out, 1, ScalarType::Float64, 1}) ==
              CpuExpressionStatus::InvalidValue,
              "usdGen refuses an inverted clamp rather than answering");
    }
    // rand() is a host groomer's, not SeExpr2's: SeExpr2 has no rand at all, so there is
    // nothing to compare against. It is specified in plan/07 and covered by
    // testUsdGenExpressionCpuEval (repeatable, per-strand, per-call-site).
    Check(Frontend::Compile("rand(1234)", {Domain::Groom, ScalarType::Float64, 1}).ok,
          "rand compiles even though SeExpr2 has no such builtin");
    // voronoi/cvoronoi/pvoronoi are ExprFuncSimple objects with file-static
    // implementations in ExprBuiltins.cpp, so they cannot be called from here.
    // They are built entirely out of cellnoise and vfbm, both proved above, and
    // their host/device agreement is proved by the CUDA parity test.
    Check(Frontend::Compile("voronoi([0.5, 0.5, 0.5], 2)",
                            {Domain::Groom, ScalarType::Float64, 1}).ok,
          "voronoi compiles");
    // dist: ExprBuiltins.cpp binds it as six scalars, contradicting its own
    // docstring and a host groomer's reference. usdGen rebinds it to dist(vector, vector)
    // and computes exactly SeExpr2's arithmetic, which is what the scalar cases
    // above compare against.
    Check(!Frontend::Compile("dist(1, 2, 3, 4, 5, 6)",
                             {Domain::Groom, ScalarType::Float64, 1}).ok,
          "the six-scalar spelling of dist is not accepted");
}

} // namespace

int main()
{
    CheckLattices();
    CheckCompiledScalars();
    CheckCompiledVectors();
    CheckCurves();
    CheckDeliberateDifferences();
    std::printf("testUsdGenSeExprOracle: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
