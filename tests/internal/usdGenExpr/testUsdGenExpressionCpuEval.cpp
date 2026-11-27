// The CPU reference lane's IR interpreter: every op, the new function set,
// and the NaN/inf poison rules.
//
// The interpreter under test is expressions/irExec.h, the SAME translation
// unit body the CUDA kernel compiles (gpu/expression.cu). A bug found here is
// a bug on the device too; testUsdGenCudaExpressionParity proves the two lanes
// still produce identical bits for the exactly-rounded op set.
#include "usdGenExpr/expressions/cpuEvaluator.h"
#include "usdGenExpr/expressions/frontend.h"
#include "usdGenExpr/expressions/irExec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace usdGen::expr;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
}

// Two straight strands of four CVs each, one at surface (0, 0) and one at
// (1, 0.5), so $u/$v, $t, $cLength and the id split all have distinct values.
struct Geometry {
    std::vector<float> px{0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<float> py{0.0f, 0.25f, 0.5f, 0.75f, 0.0f, 0.5f, 1.0f, 1.5f};
    std::vector<float> pz{0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> rest{0,0,0, 0,0.25f,0, 0,0.5f,0, 0,0.75f,0,
                            1,0,0, 1,0.5f,0, 1,1,0, 1,1.5f,0};
    std::vector<float> widths{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> hairT{0.0f, 1.0f / 3, 2.0f / 3, 1.0f, 0.0f, 1.0f / 3, 2.0f / 3, 1.0f};
    std::vector<float> rootUV{0.0f, 0.0f, 1.0f, 0.5f};
    std::vector<uint64_t> ids{7, (uint64_t(1) << 40) + 9};
    std::vector<uint32_t> offsets{0, 4, 8};

    CpuCurveGeometryView View() const
    {
        CpuCurveGeometryView view;
        view.px = px.data(); view.py = py.data(); view.pz = pz.data();
        view.rest = rest.data();
        view.widths = widths.data();
        view.hairT = hairT.data();
        view.rootUV = rootUV.data();
        view.stableIds = ids.data();
        view.curveOffsets = offsets.data();
        view.curveCount = 2;
        view.pointCount = 8;
        return view;
    }
};

/// Compiles `source`, evaluates it at `domain` into float32 and returns the
/// result. `status` receives the evaluation status so poison cases can assert
/// the refusal rather than a value.
std::vector<float> Run(Geometry const &geometry, char const *source, Domain domain,
                       double literal, CpuExpressionStatus *status,
                       bool *compiled = nullptr)
{
    auto program = Frontend::Compile(source, {domain, ScalarType::Float32, 1});
    if (compiled) *compiled = program.ok;
    if (!program.ok) {
        if (status) *status = CpuExpressionStatus::InvalidProgram;
        return {};
    }
    CpuExpressionContext context;
    std::string diagnostic;
    if (context.Build(geometry.View(), Context{0, 0, 0, 1, 0, 0, domain}, &diagnostic) !=
        CpuExpressionStatus::Ok) {
        std::printf("FAIL: context build (%s): %s\n", source, diagnostic.c_str());
        ++g_failures;
        if (status) *status = CpuExpressionStatus::InvalidArgument;
        return {};
    }
    auto inputs = context.Inputs();
    inputs.fields[unsigned(Variable::Value)] = {&literal, 1, Domain::Groom, 1};
    std::vector<float> out(inputs.count, 0.0f);
    CpuExpressionOutput output;
    output.data = out.data();
    output.count = inputs.count;
    output.type = ScalarType::Float32;
    output.components = 1;
    const auto result = EvaluateProgram(program.program.IR(), inputs, output);
    if (status) *status = result;
    return out;
}

/// Groom-domain scalar shorthand.
double Scalar(Geometry const &geometry, char const *source, double literal = 1.0)
{
    CpuExpressionStatus status = CpuExpressionStatus::Ok;
    auto values = Run(geometry, source, Domain::Groom, literal, &status);
    if (status != CpuExpressionStatus::Ok || values.size() != 1) {
        ++g_failures;
        std::printf("FAIL: groom evaluation of '%s'\n", source);
        return NAN;
    }
    return values[0];
}

bool Near(double a, double b) { return std::fabs(a - b) <= 1e-6 * std::max(1.0, std::fabs(b)); }

void CheckArithmeticAndComparisons(Geometry const &g)
{
    Check(Near(Scalar(g, "1 + 2"), 3.0), "Add");
    Check(Near(Scalar(g, "5 - 2"), 3.0), "Sub");
    Check(Near(Scalar(g, "3 * 4"), 12.0), "Mul");
    Check(Near(Scalar(g, "9 / 4"), 2.25), "Div");
    Check(Near(Scalar(g, "-7"), -7.0), "Neg");
    Check(Near(Scalar(g, "$value", 2.5), 2.5), "LoadVariable/$value literal");
    Check(Near(Scalar(g, "2 < 3"), 1.0) && Near(Scalar(g, "3 < 2"), 0.0), "Compare <");
    Check(Near(Scalar(g, "3 > 2"), 1.0), "Compare >");
    Check(Near(Scalar(g, "2 <= 2"), 1.0), "Compare <=");
    Check(Near(Scalar(g, "2 >= 3"), 0.0), "Compare >=");
    Check(Near(Scalar(g, "2 == 2"), 1.0), "Compare ==");
    Check(Near(Scalar(g, "2 != 2"), 0.0), "Compare !=");
    Check(Near(Scalar(g, "1 > 0 ? 10 : 20"), 10.0), "Select true arm");
    Check(Near(Scalar(g, "1 < 0 ? 10 : 20"), 20.0), "Select false arm");
    Check(Near(Scalar(g, "min(3, 7)"), 3.0), "Min");
    Check(Near(Scalar(g, "max(3, 7)"), 7.0), "Max");
    Check(Near(Scalar(g, "clamp(9, 0, 4)"), 4.0), "Clamp high");
    Check(Near(Scalar(g, "clamp(-9, 0, 4)"), 0.0), "Clamp low");
    Check(Near(Scalar(g, "abs(-3.5)"), 3.5), "Abs");
    Check(Near(Scalar(g, "pow(2, 10)"), 1024.0), "Pow");
    Check(Near(Scalar(g, "sin(0)"), 0.0) && Near(Scalar(g, "cos(0)"), 1.0), "Sin/Cos");
}

void CheckNewFunctions(Geometry const &g)
{
    Check(Near(Scalar(g, "sqrt(16)"), 4.0), "sqrt");
    Check(Near(Scalar(g, "exp(0)"), 1.0), "exp");
    Check(Near(Scalar(g, "log(1)"), 0.0), "log");
    Check(Near(Scalar(g, "floor(2.7)"), 2.0), "floor");
    Check(Near(Scalar(g, "ceil(2.1)"), 3.0), "ceil");
    Check(Near(Scalar(g, "fmod(7, 3)"), 1.0), "fmod");
    Check(Near(Scalar(g, "tan(0)"), 0.0), "tan");
    Check(Near(Scalar(g, "atan2(0, 1)"), 0.0), "atan2");
    // SeExpr argument order: smoothstep(x, a, b).
    Check(Near(Scalar(g, "smoothstep(-1, 0, 1)"), 0.0), "smoothstep below the low edge");
    Check(Near(Scalar(g, "smoothstep(2, 0, 1)"), 1.0), "smoothstep above the high edge");
    Check(Near(Scalar(g, "smoothstep(0.5, 0, 1)"), 0.5), "smoothstep midpoint");
    Check(Near(Scalar(g, "fit(0.25, 0, 1, 10, 20)"), 12.5), "fit");
    Check(Near(Scalar(g, "fit(2, 0, 1, 10, 20)"), 30.0), "fit does not clamp");
    Check(Near(Scalar(g, "mix(10, 20, 0.25)"), 12.5), "mix");
    Check(Near(Scalar(g, "mix(10, 20, 0)"), 10.0), "mix at alpha 0");
}

/// Builds the shortest call a signature admits: "vector v"/"color rgb"
/// parameters take a literal triple, an "=" default or a "..." tail ends the
/// required list. This is exactly the rule the tools browser uses, so what the
/// editor offers and what the compiler accepts are checked against each other.
std::string ShortestCall(std::string const &signature)
{
    std::string spelling = signature;
    const size_t alternative = spelling.find('|');
    if (alternative != std::string::npos) spelling.resize(alternative);
    const size_t open = spelling.find('(');
    const size_t close = spelling.rfind(')');
    if (open == std::string::npos || close == std::string::npos || close < open) return {};
    std::string call = spelling.substr(0, open) + "(";
    const std::string inside = spelling.substr(open + 1, close - open - 1);
    size_t written = 0, cursor = 0;
    while (cursor <= inside.size()) {
        const size_t comma = inside.find(',', cursor);
        std::string parameter = inside.substr(
            cursor, comma == std::string::npos ? std::string::npos : comma - cursor);
        cursor = comma == std::string::npos ? inside.size() + 1 : comma + 1;
        while (!parameter.empty() && parameter.front() == ' ') parameter.erase(parameter.begin());
        if (parameter.empty()) continue;
        if (parameter.find("...") != std::string::npos ||
            parameter.find('=') != std::string::npos || parameter.front() == '[') break;
        const bool isVector = parameter.rfind("vector ", 0) == 0 ||
                              parameter.rfind("color ", 0) == 0;
        const bool isString = parameter.rfind("string ", 0) == 0;
        if (written++) call += ", ";
        call += isString ? "\"0.5\"" : isVector ? "[1, 1, 1]" : "0.5";
    }
    return call + ")";
}

void CheckUnsupportedFunctions()
{
    // A name SeExpr knows but usdGen deliberately does not provide must say WHY
    // rather than hiding behind the generic list.
    struct Refusal { char const *source; char const *reason; };
    for (Refusal const &refusal : {
             Refusal{"printf(\"x\")", "printf"},
             Refusal{"map(0.5)", "image maps"},
             Refusal{"fbm4([1,1,1], 1)", "3D noise"},
             Refusal{"noise(1, 2)", "2D"}}) {
        auto result = Frontend::Compile(refusal.source, {Domain::Point, ScalarType::Float32, 1});
        const bool explained = std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
            [&](std::string const &message) {
                return message.find(refusal.reason) != std::string::npos;
            });
        Check(!result.ok && explained,
              std::string("refusal explains itself: ") + refusal.source);
    }
    // A name nobody provides still reports the whole supported list.
    for (char const *source : {"wobble($P)", "guidesAttr(1)"}) {
        auto result = Frontend::Compile(source, {Domain::Point, ScalarType::Float32, 1});
        const bool named = std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
            [](std::string const &message) {
                return message.find("supported functions are") != std::string::npos;
            });
        Check(!result.ok && named,
              std::string("unknown function reports the supported list: ") + source);
    }
    auto arity = Frontend::Compile("clamp(1, 2)", {Domain::Groom, ScalarType::Float32, 1});
    Check(!arity.ok, "wrong arity is rejected");
    // Every advertised function must actually lower at the arity it advertises,
    // and must be documented and categorised.
    for (auto const &info : Frontend::SupportedFunctions()) {
        const std::string call = ShortestCall(info.signature);
        // The sampling functions read at the element, never at groom rate.
        const Domain domain = info.category == "sampling" ? Domain::Primitive : Domain::Groom;
        auto result = Frontend::Compile(call, {domain, ScalarType::Float32, 1});
        if (!result.ok) {
            ++g_failures;
            std::printf("FAIL: SupportedFunctions entry does not lower as '%s': %s\n",
                        call.c_str(),
                        result.diagnostics.empty() ? "" : result.diagnostics[0].c_str());
        }
        Check(!info.signature.empty() && !info.doc.empty(),
              "SupportedFunctions entry is documented: " + info.name);
        Check(info.category == "math" || info.category == "noise" ||
              info.category == "vector" || info.category == "color" ||
              info.category == "curve" || info.category == "control" ||
              info.category == "sampling",
              "SupportedFunctions entry is categorised: " + info.name);
        Check(info.components == 1 || info.components == 3,
              "SupportedFunctions entry declares a result width: " + info.name);
        // One argument short of the minimum must be refused by name.
        if (info.minArity == 0) continue;
        std::string tooFew = info.name + "(";
        for (uint32_t i = 0; i + 1 < info.minArity; ++i) tooFew += (i ? ", 1" : "1");
        tooFew += ")";
        Check(!Frontend::Compile(tooFew, {domain, ScalarType::Float32, 1}).ok,
              "too few arguments is rejected: " + tooFew);
    }
}

void CheckVariableDocs()
{
    const auto variables = Frontend::VariableDocs();
    Check(!variables.empty(), "the variable table is not empty");
    for (auto const &variable : variables) {
        Check(!variable.doc.empty(), "variable is documented: " + variable.name);
        Check(!variable.domains.empty(), "variable lists its domains: " + variable.name);
    }
    // $patchId, $Cs and $As were declared but never written by any field
    // builder, so an expression naming one poisoned. They are gone, not silent.
    for (char const *gone : {"$patchId", "$Cs", "$As"}) {
        const bool present = std::any_of(variables.begin(), variables.end(),
            [&](VariableDoc const &v) { return v.name == gone; });
        Check(!present, std::string("removed variable is not advertised: ") + gone);
        Check(!Frontend::Compile(gone, {Domain::Point, ScalarType::Float32, 1}).ok,
              std::string("removed variable does not compile: ") + gone);
    }
    // The root-frame variables are wired now and must compile.
    for (char const *wired : {"$N[1]", "$Nref[0]", "$dPdu[0]", "$dPdv[2]",
                              "$dPduref[0]", "$dPdvref[1]", "$faceId"}) {
        auto result = Frontend::Compile(wired, {Domain::Point, ScalarType::Float32, 1});
        if (!result.ok) {
            ++g_failures;
            std::printf("FAIL: root-frame variable does not compile: %s (%s)\n", wired,
                        result.diagnostics.empty() ? "" : result.diagnostics[0].c_str());
        }
    }
}

/// Multi-statement sources, locals, if/else and comments.
void CheckLanguage(Geometry const &g)
{
    Check(Near(Scalar(g, "$a = 2;\n$a * 3"), 6.0), "a local variable is assigned and read");
    Check(Near(Scalar(g, "$a = 2; $a = $a + 1; $a * 2"), 6.0), "a local can be re-assigned");
    Check(Near(Scalar(g, "$a = 2;   # a comment\n$a + 1  # trailing\n"), 3.0),
          "# comments are ignored");
    Check(Near(Scalar(g, "$a = 1; $b = 2; $c = $a + $b; $c * $c"), 9.0),
          "several statements run in order");
    Check(Near(Scalar(g, "$a = 0; if (1 > 0) { $a = 5; } else { $a = 9; } $a"), 5.0),
          "if takes the true branch");
    Check(Near(Scalar(g, "$a = 0; if (1 < 0) { $a = 5; } else { $a = 9; } $a"), 9.0),
          "if takes the false branch");
    Check(Near(Scalar(g, "$a = 3; if (0 > 1) { $a = 5; } $a"), 3.0),
          "an if with no else keeps the prior value");
    Check(Near(Scalar(g,
        "$a = 0;"
        "if (2 > 1) { if (3 > 4) { $a = 1; } else { $a = 2; } } else { $a = 3; }"
        "$a"), 2.0), "nested ifs merge correctly");
    Check(Near(Scalar(g, "$c = [1, 0.5, 0.2]; $c[1]"), 0.5), "a vector local keeps its components");
    Check(Near(Scalar(g, "$c = [1, 2, 3]; $d = $c * 2; $d[2]"), 6.0),
          "vector arithmetic on a local is component-wise");
    // $v is a registry variable, so the local is named $vec: a local may not
    // shadow a built-in, which the sandbox check below asserts directly.
    Check(Near(Scalar(g, "$s = 4; $vec = [$s, $s + 1, $s + 2]; $vec[2]"), 6.0),
          "a vector literal may be built from locals");
    Check(Near(Scalar(g, "$a = 1; $a += 2; $a"), 3.0), "the += form assigns");
    // The sandbox stays shut.
    Check(!Frontend::Compile("$t = 0.5; $t", {Domain::Point, ScalarType::Float32, 1}).ok,
          "assigning to a built-in variable is refused");
    auto shadow = Frontend::Compile("$u = 1; $u", {Domain::Point, ScalarType::Float32, 1});
    const bool named = std::any_of(shadow.diagnostics.begin(), shadow.diagnostics.end(),
        [](std::string const &m) {
            return m.find("built-in expression variable $u") != std::string::npos;
        });
    Check(!shadow.ok && named, "the refusal names the variable");
    Check(!Frontend::Compile("def float f(float x) { x } f(1)",
                             {Domain::Groom, ScalarType::Float32, 1}).ok,
          "user-defined functions are refused");
    Check(!Frontend::Compile("\"text\"", {Domain::Groom, ScalarType::Float32, 1}).ok,
          "string literals are refused");
    // Every diagnostic carries a position so an editor can point at it.
    auto positioned = Frontend::Compile("1 + wobble(2)", {Domain::Groom, ScalarType::Float32, 1});
    Check(!positioned.ok && !positioned.diagnostics.empty() &&
          positioned.diagnostics[0].find(" at ") != std::string::npos,
          "a diagnostic carries a source position");
}

/// The a host groomer/SeExpr function library, against values worked out by hand from
/// SeExpr2's own definitions.
void CheckLibrary(Geometry const &g)
{
    Check(Near(Scalar(g, "deg(PI)"), 180.0), "deg");
    Check(Near(Scalar(g, "rad(180)"), 3.14159265358979323846), "rad");
    Check(Near(Scalar(g, "cosd(0)"), 1.0) && Near(Scalar(g, "sind(90)"), 1.0) &&
          Near(Scalar(g, "tand(45)"), 1.0), "cosd/sind/tand");
    Check(Near(Scalar(g, "acosd(0)"), 90.0) && Near(Scalar(g, "asind(1)"), 90.0) &&
          Near(Scalar(g, "atand(1)"), 45.0) && Near(Scalar(g, "atan2d(1, 1)"), 45.0),
          "acosd/asind/atand/atan2d");
    Check(Near(Scalar(g, "acos(1)"), 0.0) && Near(Scalar(g, "asin(0)"), 0.0) &&
          Near(Scalar(g, "atan(0)"), 0.0), "acos/asin/atan");
    Check(Near(Scalar(g, "cosh(0)"), 1.0) && Near(Scalar(g, "sinh(0)"), 0.0) &&
          Near(Scalar(g, "tanh(0)"), 0.0), "cosh/sinh/tanh");
    Check(Near(Scalar(g, "cbrt(27)"), 3.0), "cbrt");
    Check(Near(Scalar(g, "log10(1000)"), 3.0), "log10");
    Check(Near(Scalar(g, "hypot(3, 4)"), 5.0), "hypot");
    Check(Near(Scalar(g, "round(2.5)"), 3.0) && Near(Scalar(g, "round(-2.5)"), -3.0),
          "round takes halves away from zero");
    Check(Near(Scalar(g, "trunc(-2.7)"), -2.0), "trunc");
    Check(Near(Scalar(g, "invert(0.25)"), 0.75), "invert");
    Check(Near(Scalar(g, "compress(0.5, 2, 4)"), 3.0), "compress");
    Check(Near(Scalar(g, "expand(3, 2, 4)"), 0.5), "expand");
    Check(Near(Scalar(g, "expand(1, 2, 2)"), 0.0), "expand with a degenerate range steps");
    Check(Near(Scalar(g, "gamma(0.25, 2)"), 0.5), "gamma");
    Check(Near(Scalar(g, "bias(0.5, 0.5)"), 0.5), "bias");
    Check(Near(Scalar(g, "contrast(0.25, 0.5)"), 0.25), "contrast");
    Check(Near(Scalar(g, "boxstep(0.4, 0.5)"), 0.0) && Near(Scalar(g, "boxstep(0.6, 0.5)"), 1.0),
          "boxstep");
    Check(Near(Scalar(g, "linearstep(0.25, 0, 1)"), 0.25), "linearstep");
    Check(Near(Scalar(g, "linearstep(0.25, 1, 0)"), 0.75), "linearstep reverses when a > b");
    Check(Near(Scalar(g, "smoothstep(0.5, 1, 0)"), 0.5),
          "smoothstep reverses when a > b instead of poisoning");
    Check(Near(Scalar(g, "gaussstep(0.5, 0, 1)"), 0.25), "gaussstep");
    Check(Near(Scalar(g, "remap(1, 1, 0.5, 0, 0)"), 1.0) &&
          Near(Scalar(g, "remap(2, 1, 0.5, 0, 0)"), 0.0), "remap with no falloff is a box");
    Check(Near(Scalar(g, "cycle(5, 0, 2)"), 2.0) && Near(Scalar(g, "cycle(-1, 0, 2)"), 2.0),
          "cycle wraps in both directions");
    // Vectors.
    Check(Near(Scalar(g, "length([3, 4, 0])"), 5.0), "length");
    Check(Near(Scalar(g, "dist([0, 0, 0], [3, 4, 0])"), 5.0), "dist");
    Check(Near(Scalar(g, "dot([1, 2, 3], [4, 5, 6])"), 32.0), "dot");
    Check(Near(Scalar(g, "cross([1, 0, 0], [0, 1, 0])[2]"), 1.0), "cross");
    Check(Near(Scalar(g, "norm([3, 4, 0])[0]"), 0.6), "norm");
    Check(Near(Scalar(g, "norm([0, 0, 0])[0]"), 0.0), "norm of the zero vector stays zero");
    Check(Near(Scalar(g, "angle([1, 0, 0], [0, 1, 0])"), 3.14159265358979323846 / 2), "angle");
    Check(Near(Scalar(g, "ortho([1, 0, 0], [0, 1, 0])[2]"), 1.0), "ortho");
    Check(Near(Scalar(g, "rotate([1, 0, 0], [0, 0, 1], rad(90))[1]"), 1.0), "rotate");
    Check(Near(Scalar(g, "up([1, 0, 0], [0, 1, 0])[0]"), 1.0), "up along Y is the identity");
    // Colour.
    Check(Near(Scalar(g, "rgbtohsl([0.5, 0.5, 0.5])[2]"), 0.5), "rgbtohsl of a grey is its L");
    Check(Near(Scalar(g, "rgbtohsl([0.5, 0.5, 0.5])[1]"), 0.0), "grey has no saturation");
    Check(Near(Scalar(g, "hsltorgb([0, 0, 0.5])[0]"), 0.5), "hsltorgb of an unsaturated colour");
    Check(Near(Scalar(g, "hsltorgb(rgbtohsl([0.2, 0.6, 0.4]))[1]"), 0.6),
          "rgbtohsl and hsltorgb round-trip");
    Check(Near(Scalar(g, "saturate([1, 0, 0], 1)[0]"), 1.0), "saturate at 1 is the identity");
    Check(Near(Scalar(g, "saturate([1, 0, 0], 0)[0]"), 0.2126),
          "saturate at 0 collapses to rec709 luminance");
    Check(Near(Scalar(g, "hsi([0.2, 0.6, 0.4], 0, 1, 1)[1]"), 0.6), "hsi with no shift");
    Check(Near(Scalar(g, "midhsi([0.2, 0.6, 0.4], 0, 1, 1, 0.5)[1]"), 0.6),
          "midhsi at the mid point of its map");
    // Noise. Perlin noise is exactly zero on the integer lattice, which is the
    // one value that can be asserted rather than bounded.
    Check(Near(Scalar(g, "snoise([0, 0, 0])"), 0.0), "snoise is 0 on the lattice");
    Check(Near(Scalar(g, "noise([0, 0, 0])"), 0.5), "noise is 0.5 on the lattice");
    Check(Near(Scalar(g, "noise(0, 0, 0)"), 0.5), "three scalars spell the same 3D noise");
    Check(Near(Scalar(g, "vnoise([0, 0, 0])[0]"), 0.0), "vnoise is 0 on the lattice");
    Check(Near(Scalar(g, "cnoise([0, 0, 0])[0]"), 0.5), "cnoise is 0.5 on the lattice");
    Check(Near(Scalar(g, "fbm([0, 0, 0], 1, 2, 0.5)"), 0.5), "one octave of fbm is noise");
    Check(Near(Scalar(g, "turbulence([0, 0, 0], 1, 2, 0.5)"), 0.5), "one octave of turbulence");
    Check(Near(Scalar(g, "pnoise([0, 0, 0], [4, 4, 4])"), 0.0), "pnoise is 0 on the lattice");
    auto inUnit = [&](char const *source) {
        const double value = Scalar(g, source);
        return value >= 0.0 && value <= 1.0;
    };
    Check(inUnit("cellnoise([0.5, 0.5, 0.5])"), "cellnoise lands in [0,1]");
    Check(inUnit("ccellnoise([0.5, 0.5, 0.5])[0]"), "ccellnoise lands in [0,1]");
    Check(inUnit("hash(1)") && inUnit("hash(1, 2, 3)"), "hash lands in [0,1]");
    Check(Near(Scalar(g, "hash(1)"), Scalar(g, "hash(1)")), "hash is repeatable");
    Check(!Near(Scalar(g, "hash(1)"), Scalar(g, "hash(2)")), "hash separates its seeds");
    Check(inUnit("voronoi([0.5, 0.5, 0.5])"), "voronoi type 1 lands in [0,1]");
    Check(Scalar(g, "voronoi([0.5, 0.5, 0.5], 2)") >= 0.0, "voronoi f1 is a distance");
    Check(Scalar(g, "voronoi([0.5, 0.5, 0.5], 4)") >= 0.0, "voronoi f2-f1 is non-negative");
    Check(inUnit("cvoronoi([0.5, 0.5, 0.5])[0]"), "cvoronoi lands in [0,1]");
    Check(std::fabs(Scalar(g, "pvoronoi([0.5, 0.5, 0.5])[0]") - 0.5) < 1.5,
          "pvoronoi returns a nearby cell centre");
    // rand() is stable per call site and separated between call sites.
    Check(inUnit("rand()"), "rand lands in [0,1]");
    Check(Near(Scalar(g, "rand()"), Scalar(g, "rand()")), "rand is repeatable");
    Check(!Near(Scalar(g, "rand() - rand()"), 0.0), "two rand() call sites differ");
    Check(Near(Scalar(g, "rand(1234)"), Scalar(g, "rand(1234)")), "rand(seed) is repeatable");
    Check(!Near(Scalar(g, "rand(1) - rand(2)"), 0.0), "rand separates its seeds");
    const double ranged = Scalar(g, "rand(2, 3, 7)");
    Check(ranged >= 2.0 && ranged <= 3.0, "rand(min, max, seed) stays in range");
    // Variations.
    Check(Near(Scalar(g, "spline(0, 1, 2, 3, 4)"), 2.0), "spline at 0");
    Check(Near(Scalar(g, "spline(1, 1, 2, 3, 4)"), 3.0), "spline at 1");
    Check(Near(Scalar(g, "choose(0, 10, 20, 30)"), 10.0), "choose picks the first");
    Check(Near(Scalar(g, "choose(0.9, 10, 20, 30)"), 30.0), "choose picks the last");
    Check(Near(Scalar(g, "wchoose(0.1, 10, 1, 20, 1)"), 10.0), "wchoose picks by weight");
    const double picked = Scalar(g, "pick(0.5, 3, 7)");
    Check(picked >= 3.0 && picked <= 7.0, "pick stays inside its range");
    // Control curves.
    Check(Near(Scalar(g, "curve($value, 0, 0, 1, 1, 1, 1)", 0.5), 0.5), "a linear curve");
    Check(Near(Scalar(g, "curve($value, 0, 0, 0, 1, 1, 0)", 0.5), 0.0),
          "interpolation 0 holds the lower knot");
    Check(Near(Scalar(g, "curve($value, 0, 0, 2, 1, 1, 2)", 0.5), 0.5),
          "a smooth curve is symmetric at the midpoint");
    Check(Near(Scalar(g, "curve($value, 0, 0, 1, 1, 1, 1)", 0.0), 0.0) &&
          Near(Scalar(g, "curve($value, 0, 0, 1, 1, 1, 1)", 1.0), 1.0),
          "a curve reaches both of its end knots");
    Check(Near(Scalar(g, "curve($value, 0, 1, 4, 0.5, 0.7, 4, 1, 0, 4)", 0.0), 1.0),
          "a monotone spline curve starts at its first knot");
    Check(Near(Scalar(g, "ccurve($value, 0, [0,0,0], 1, 1, [1,0.5,0], 1)[1]", 0.5), 0.25),
          "a colour curve interpolates each channel");
    Check(!Frontend::Compile("curve($t, 0, $u, 1, 1, 1, 1)",
                             {Domain::Point, ScalarType::Float32, 1}).ok,
          "a non-constant curve knot is refused");
    Check(!Frontend::Compile("curve($t, 0, 0, 9, 1, 1, 1)",
                             {Domain::Point, ScalarType::Float32, 1}).ok,
          "an out-of-range interpolation code is refused");
}

void CheckPoison(Geometry const &g)
{
    CpuExpressionStatus status = CpuExpressionStatus::Ok;
    // 1/0 is an infinity, which no destination can represent: the whole
    // evaluation is refused rather than publishing a finite-looking value.
    Run(g, "1 / 0", Domain::Groom, 1.0, &status);
    Check(status == CpuExpressionStatus::InvalidValue, "division by zero is refused");
    Run(g, "0 / 0", Domain::Groom, 1.0, &status);
    Check(status == CpuExpressionStatus::InvalidValue, "0/0 NaN is refused");
    Run(g, "sqrt(-1)", Domain::Groom, 1.0, &status);
    Check(status == CpuExpressionStatus::InvalidValue, "sqrt of a negative is refused");
    Run(g, "log(0)", Domain::Groom, 1.0, &status);
    Check(status == CpuExpressionStatus::InvalidValue, "log(0) is refused");
    // A comparison on poison must not become a plausible boolean, and the
    // conditional that consumes it must not select an arm.
    Run(g, "(1 / 0) > 0 ? 1 : 2", Domain::Groom, 1.0, &status);
    Check(status == CpuExpressionStatus::InvalidValue,
          "a comparison on poison poisons the conditional");
    // But a dead arm may carry poison: only the selected value is published.
    auto live = Run(g, "1 > 0 ? 5 : 1 / 0", Domain::Groom, 1.0, &status);
    Check(status == CpuExpressionStatus::Ok && live.size() == 1 && Near(live[0], 5.0),
          "a dead conditional arm may be poison");
    // clamp with an inverted range is invalid, not silently reordered.
    Run(g, "clamp(0.5, 1, 0)", Domain::Groom, 1.0, &status);
    Check(status == CpuExpressionStatus::InvalidValue, "clamp with hi < lo is refused");
    Run(g, "min(1 / 0, 1)", Domain::Groom, 1.0, &status);
    Check(status == CpuExpressionStatus::InvalidValue, "min of poison is refused");
}

void CheckDestinationRanges(Geometry const &g)
{
    auto run = [&](char const *source, ScalarType type, void *data) {
        auto program = Frontend::Compile(source, {Domain::Groom, type, 1});
        Check(program.ok, std::string("compile for destination: ") + source);
        if (!program.ok) return CpuExpressionStatus::InvalidProgram;
        CpuExpressionContext context;
        context.Build(g.View(), Context{0, 0, 0, 1, 0, 0, Domain::Groom}, nullptr);
        auto inputs = context.Inputs();
        double literal = 1.0;
        inputs.fields[unsigned(Variable::Value)] = {&literal, 1, Domain::Groom, 1};
        return EvaluateProgram(program.program.IR(), inputs, {data, 1, type, 1});
    };
    unsigned char boolean = 0;
    int32_t integer = 0;
    Check(run("1", ScalarType::Bool, &boolean) == CpuExpressionStatus::Ok && boolean == 1,
          "exact 1 stores into a bool destination");
    Check(run("0.5", ScalarType::Bool, &boolean) == CpuExpressionStatus::InvalidValue,
          "a fractional value is refused by a bool destination");
    Check(run("7", ScalarType::Int32, &integer) == CpuExpressionStatus::Ok && integer == 7,
          "an integral value stores into an int destination");
    Check(run("7.5", ScalarType::Int32, &integer) == CpuExpressionStatus::InvalidValue,
          "a fractional value is refused by an int destination");
}

void CheckGeometryFields(Geometry const &g)
{
    CpuExpressionStatus status = CpuExpressionStatus::Ok;
    auto t = Run(g, "$t", Domain::Point, 1.0, &status);
    Check(status == CpuExpressionStatus::Ok && t.size() == 8 && t[0] == 0.0f &&
          Near(t[3], 1.0) && t[4] == 0.0f && Near(t[7], 1.0),
          "$t comes from the authored hairT channel, 0 at the root and 1 at the tip");
    auto u = Run(g, "$u", Domain::Primitive, 1.0, &status);
    Check(status == CpuExpressionStatus::Ok && u.size() == 2 && Near(u[0], 0.0) &&
          Near(u[1], 1.0), "$u is the strand root's surface coordinate");
    auto v = Run(g, "$v", Domain::Point, 1.0, &status);
    Check(status == CpuExpressionStatus::Ok && v.size() == 8 && Near(v[0], 0.0) &&
          Near(v[4], 0.5), "a primitive channel read at point rate goes through the owner map");
    auto length = Run(g, "$cLength", Domain::Primitive, 1.0, &status);
    Check(status == CpuExpressionStatus::Ok && Near(length[0], 0.75) && Near(length[1], 1.5),
          "$cLength is the strand's cumulative arc length");
    auto width = Run(g, "$cWidth", Domain::Point, 1.0, &status);
    Check(status == CpuExpressionStatus::Ok && Near(width[2], 3.0),
          "$cWidth is the incoming per-CV width");
    auto lo = Run(g, "$idLo", Domain::Primitive, 1.0, &status);
    auto hi = Run(g, "$idHi", Domain::Primitive, 1.0, &status);
    Check(status == CpuExpressionStatus::Ok && Near(lo[0], 7.0) && Near(lo[1], 9.0) &&
          Near(hi[0], 0.0) && Near(hi[1], 256.0),
          "$idLo/$idHi split the 64-bit stable id");
    auto index = Run(g, "$pointIndex", Domain::Point, 1.0, &status);
    auto count = Run(g, "$pointCount", Domain::Point, 1.0, &status);
    Check(status == CpuExpressionStatus::Ok && Near(index[5], 1.0) && Near(count[5], 4.0),
          "$pointIndex/$pointCount are curve-local");
    // The groom domain carries no per-element field at all: the variable is
    // rejected by the frontend before it can reach the interpreter.
    Check(!Frontend::Compile("$t", {Domain::Groom, ScalarType::Float32, 1}).ok,
          "the groom domain has no per-element variables");

    // Arc-length fallback: with no hairT channel, $t is the normalized
    // cumulative arc length. Strand 1 is evenly spaced, strand 2 is not.
    Geometry noChannel = g;
    noChannel.py = {0.0f, 0.25f, 0.5f, 0.75f, 0.0f, 0.1f, 0.2f, 1.0f};
    noChannel.hairT.clear();
    CpuCurveGeometryView view = noChannel.View();
    view.hairT = nullptr;
    CpuExpressionContext context;
    auto program = Frontend::Compile("$t", {Domain::Point, ScalarType::Float32, 1});
    Check(program.ok && context.Build(view, Context{0, 0, 0, 1, 0, 0, Domain::Point}, nullptr) ==
          CpuExpressionStatus::Ok, "build the arc-length fallback context");
    std::vector<float> arc(8, 0.0f);
    double literal = 1.0;
    auto inputs = context.Inputs();
    inputs.fields[unsigned(Variable::Value)] = {&literal, 1, Domain::Groom, 1};
    Check(EvaluateProgram(program.program.IR(), inputs,
                          {arc.data(), 8, ScalarType::Float32, 1}) == CpuExpressionStatus::Ok &&
          Near(arc[1], 1.0 / 3) && Near(arc[5], 0.1) && Near(arc[7], 1.0),
          "$t falls back to normalized arc length when no hairT channel exists");
}

void CheckMalformedGeometry(Geometry const &g)
{
    Geometry bad = g;
    bad.hairT[3] = 0.5f;   // the tip must be 1
    std::string diagnostic;
    CpuExpressionContext context;
    Check(context.Build(bad.View(), Context{0, 0, 0, 1, 0, 0, Domain::Point}, &diagnostic) ==
          CpuExpressionStatus::InvalidChannel,
          "a hairT channel that does not reach 1 at the tip is refused");
    Geometry nonFinite = g;
    nonFinite.py[2] = std::numeric_limits<float>::infinity();
    CpuExpressionContext other;
    Check(other.Build(nonFinite.View(), Context{0, 0, 0, 1, 0, 0, Domain::Point}, &diagnostic) ==
          CpuExpressionStatus::InvalidGeometry, "a non-finite point is refused");
}

} // namespace

int main()
{
    Geometry geometry;
    CheckArithmeticAndComparisons(geometry);
    CheckNewFunctions(geometry);
    CheckUnsupportedFunctions();
    CheckVariableDocs();
    CheckLanguage(geometry);
    CheckLibrary(geometry);
    CheckPoison(geometry);
    CheckDestinationRanges(geometry);
    CheckGeometryFields(geometry);
    CheckMalformedGeometry(geometry);
    std::printf("testUsdGenExpressionCpuEval: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
