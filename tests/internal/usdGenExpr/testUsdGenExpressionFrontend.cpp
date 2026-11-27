// The expression frontend: what the language accepts, what it refuses and how
// it says so.
//
// Evaluation lives in testUsdGenExpressionCpuEval (values) and
// testUsdGenSeExprOracle (agreement with upstream SeExpr2). This file is about
// compilation only: every source below either must compile or must be refused
// with a diagnostic a user can act on.
#include "usdGenExpr/expressions/cpuEvaluator.h"
#include "usdGenExpr/expressions/frontend.h"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace usdGen::expr;

namespace {

int g_failures = 0;

void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
}

CompileResult Compile(std::string const &source, Domain domain = Domain::Point,
                      ScalarType destination = ScalarType::Float32, uint32_t components = 1)
{
    return Frontend::Compile(source, {domain, destination, components});
}

void Accepts(std::string const &source, Domain domain = Domain::Point,
             uint32_t components = 1)
{
    auto result = Compile(source, domain, ScalarType::Float32, components);
    if (!result.ok) {
        ++g_failures;
        std::printf("FAIL: should compile: %s\n  %s\n", source.c_str(),
                    result.diagnostics.empty() ? "(no diagnostic)"
                                               : result.diagnostics[0].c_str());
    }
}

/// Refuses `source`, and the first diagnostic must contain `fragment` and a
/// source position, so an editor can both explain and point.
void Refuses(std::string const &source, std::string const &fragment,
             Domain domain = Domain::Point)
{
    auto result = Compile(source, domain);
    if (result.ok) {
        ++g_failures;
        std::printf("FAIL: should be refused: %s\n", source.c_str());
        return;
    }
    // A position is either usdGen's own " at <offset>" or the vendored
    // parser's "Line L Col C", depending on which stage caught it.
    bool explained = false, positioned = false;
    for (std::string const &message : result.diagnostics) {
        explained = explained || message.find(fragment) != std::string::npos;
        positioned = positioned || message.find(" at ") != std::string::npos ||
                     message.find("Col ") != std::string::npos;
    }
    if (!explained) {
        ++g_failures;
        std::printf("FAIL: refusal of '%s' does not mention '%s'\n  %s\n", source.c_str(),
                    fragment.c_str(),
                    result.diagnostics.empty() ? "(no diagnostic)"
                                               : result.diagnostics[0].c_str());
    }
    if (!positioned) {
        ++g_failures;
        std::printf("FAIL: refusal of '%s' carries no source position\n", source.c_str());
    }
}

void CheckBasics()
{
    for (char const *source : {"4 * 2", "$frame >= 1", "$value * (0.5 + 0.5 * $u)",
                               "$value * $t", "$value * (1 - 0.95 * $t)",
                               "$value * (0.25 + 0.75 * $u)"})
        Accepts(source);
    Check(!Compile("unknownFile($u)").ok, "an unknown function is refused");
    Check(!Compile("$pointIndex", Domain::Groom).ok, "$pointIndex is not a groom variable");
    Check(Compile("$frame >= 1", Domain::Groom, ScalarType::Bool).ok, "bool destination");
    Check(Compile("4*2", Domain::Groom, ScalarType::Int32).ok, "int destination");
    Check(Compile("$value*0.5", Domain::Point, ScalarType::Float16).ok, "half destination");
    Check(!Compile("1", Domain::Groom, ScalarType::Invalid).ok, "invalid destination type");
    Check(Compile("[$t, $u, $v]", Domain::Point, ScalarType::Float32, 3).ok,
          "a three-component destination takes a vector expression");
    // The operators the parser spells but the old IR did not lower.
    for (char const *source : {"$t % 0.25", "$t ^ 2", "($t > 0.5) && ($u > 0.5)",
                               "($t > 0.5) || ($u > 0.5)", "!($t > 0.5)", "~$t"})
        Accepts(source);
}

/// Goal 1: multi-statement sources, locals and if/else.
void CheckLanguage()
{
    Accepts("$a = 2;\n$a * 3", Domain::Groom);
    Accepts("$a = 2; $b = $a * $a; $b + 1", Domain::Groom);
    Accepts("# leading comment\n$a = 2;  # trailing\n$a", Domain::Groom);
    Accepts("$a = 0; if ($t > 0.5) { $a = 1; } else { $a = 2; } $a");
    Accepts("$a = 0; if ($t > 0.5) { $a = 1; } $a");
    Accepts("$a = 0; if ($t > 0.75) { $a = 1; } else if ($t > 0.5) { $a = 2; } else { $a = 3; } $a");
    Accepts("$c = [1, 0.5, 0.2]; $c[1] * $value");
    Accepts("$c = [1, 0.5, 0.2]; $c * $value", Domain::Point, 3);
    Accepts("$rootWidth = 1.0;\n"
            "$tipWidth = 0.15;\n"
            "$profile = curve($t, 0, 1, 4, 0.5, 0.7, 4, 1, 0, 4);\n"
            "$value * mix($tipWidth, $rootWidth, $profile)");
    // The sandbox.
    Refuses("$t = 1; $t", "built-in expression variable $t");
    Refuses("\"text\"", "string literals");
    // The vendored parser is built with USDGEN_SEEXPR_SANDBOX, so `def` never
    // reaches the AST walk: it is a syntax error, which is a refusal either
    // way. Assert only that it is refused and explained.
    auto declaration = Compile("def float f(float x) { x } f(1)", Domain::Groom);
    Check(!declaration.ok && !declaration.diagnostics.empty() &&
          !declaration.diagnostics[0].empty(),
          "a user-defined function is refused with a diagnostic");
}

/// Goal 2: every function the table advertises, and every refusal that carries
/// its own reason.
void CheckFunctionTable()
{
    const auto functions = Frontend::SupportedFunctions();
    Check(functions.size() > 80, "the function table covers the host-groomer/SeExpr set");
    std::set<std::string> names, categories;
    for (auto const &info : functions) {
        Check(names.insert(info.name).second, "function appears once: " + info.name);
        Check(!info.doc.empty(), "function is documented: " + info.name);
        Check(info.signature.rfind(info.name + "(", 0) == 0,
              "signature starts with the function name: " + info.name);
        Check(info.minArity <= info.arity, "advertised arity is the minimum: " + info.name);
        Check(info.maxArity == 0 || info.maxArity >= info.minArity,
              "arity range is ordered: " + info.name);
        categories.insert(info.category);
    }
    // Sorted by name, which the editor's browser relies on.
    for (size_t i = 1; i < functions.size(); ++i)
        Check(functions[i - 1].name < functions[i].name,
              "the function table is sorted: " + functions[i].name);
    for (char const *category : {"math", "noise", "vector", "color", "curve", "control",
                                 "sampling"})
        Check(categories.count(category) == 1,
              std::string("the table has a ") + category + " category");
    Check(categories.size() == 7, "the table has no category outside the documented seven");
    // Names a user coming from a host groomer or SeExpr will try, each with its own
    // reason rather than the generic list.
    Refuses("printf(\"%f\", $t)", "printf");
    Refuses("sprintf(\"%f\", $t)", "printf");
    Refuses("map(0.5)", "image maps are not yet available");
    Refuses("ptex(0.5)", "must be a string literal");
    Refuses("file(0.5)", "filesystem");
    Refuses("system(0.5)", "run a command");
    Refuses("exec(0.5)", "run a command");
    Refuses("fbm4([1,1,1], 1)", "only 3D noise is available");
    Refuses("snoise4([1,1,1], 1)", "only 3D noise is available");
    Refuses("swatch(0.5, 1, 2)", "use choose()");
    Refuses("noise(1, 2)", "2D");
    Refuses("wobble($t)", "supported functions are");
    // Arity messages name the signature.
    Refuses("clamp(1, 2)", "clamp(x, lo, hi)");
    Refuses("hash()", "at least 1");
    Refuses("curve($t, 0, 1)", "at least 4");
    Refuses("curve($t, 0, 1, 1, 0.5)", "(position, value, interpolation) triples");
}

/// Goal 2, control curves: the knots are lowered as constants, so anything the
/// frontend cannot fold is refused rather than silently evaluated per element.
void CheckCurves()
{
    Accepts("curve($t, 0, 1, 4, 0.5, 0.7, 4, 1, 0, 4)");
    Accepts("curve($t, 0, 0, 0, 0.5, 1, 1, 1, 0, 2)");
    Accepts("ccurve($t, 0, [1, 0, 0], 1, 1, [0, 0, 1], 1)", Domain::Point, 3);
    Accepts("spline($t, 0, 0.25, 0.75, 1)");
    Accepts("choose($t, 0.1, 0.5, 0.9)");
    Accepts("wchoose($t, 0.1, 1, 0.9, 3)");
    Accepts("pick($t, 1, 4, 1, 2, 1, 1)");
    // A varying knot is caught by the type checker, which already says so.
    Refuses("curve($t, 0, $u, 1, 1, 1, 1)", "constant");
    Refuses("curve($t, $t, 1, 1, 1, 1, 1)", "constant");
    // A knot that is constant to SeExpr but not foldable by the frontend is
    // caught by the lowering, which is where usdGen's own message lives: the
    // knots become immediates in the instruction stream, so they must be
    // numbers the compiler can work out.
    Refuses("curve($t, 0, sqrt(4), 1, 1, 1, 1)", "must be constants");
    Refuses("curve($t, 0, 0, 7, 1, 1, 1)", "interpolation must be");
    // A constant expression IS foldable, so arithmetic on knots is allowed.
    Accepts("curve($t, 0, 1, 1, 0.5 + 0.25, 1 - 0.3, 1, 1, 0, 1)");
}

/// The advertised limits, and that exceeding them is a diagnostic rather than a
/// program the interpreter would refuse later.
void CheckLimits()
{
    std::string huge = "$t";
    for (int i = 0; i < 400; ++i) huge += " + $t";
    auto result = Compile(huge);
    Check(!result.ok, "a program past the register file is refused");
    bool named = false;
    for (std::string const &message : result.diagnostics)
        named = named || message.find("expression IR limit exceeded") != std::string::npos;
    Check(named, "the limit refusal says what the limit is");
}

void CheckVariables()
{
    const auto variables = Frontend::VariableDocs();
    Check(variables.size() >= 30, "the variable table is complete");
    for (auto const &variable : variables) {
        Check(variable.name.size() > 1 && variable.name[0] == '$',
              "a variable is spelled with $: " + variable.name);
        Check(!variable.doc.empty(), "a variable is documented: " + variable.name);
        Check(!variable.domains.empty(), "a variable lists its domains: " + variable.name);
    }
    Check(Compile("$N[0] + $dPdu[1] + $dPdv[2] + $dPduref[0] + $dPdvref[1] + $Nref[2] + $faceId").ok,
          "the root-frame variables compile at point rate");
    for (char const *gone : {"$patchId", "$Cs", "$As"})
        Check(!Compile(gone).ok, std::string("a removed variable is refused: ") + gone);
}

/// Evaluates a variable-free groom program with hand-built inputs: no geometry
/// is needed because the program reads no fields.
bool EvaluateScalar(IRProgram const &program, float *out)
{
    CpuExpressionInputs inputs;
    inputs.context.domain = Domain::Groom;
    inputs.count = 1;
    CpuExpressionOutput output;
    output.data = out;
    output.count = 1;
    output.type = ScalarType::Float32;
    output.components = 1;
    return EvaluateProgram(program, inputs, output) == CpuExpressionStatus::Ok;
}

bool SameIR(IRProgram const &a, IRProgram const &b)
{
    if (a.instructions.size() != b.instructions.size() || a.result != b.result ||
        a.outputCount != b.outputCount || a.valueComponents != b.valueComponents ||
        a.registerCount != b.registerCount || a.hasLazyBranches != b.hasLazyBranches ||
        a.samplers.size() != b.samplers.size())
        return false;
    for (size_t i = 0; i < 4; ++i)
        if (a.output[i] != b.output[i]) return false;
    for (size_t i = 0; i < a.instructions.size(); ++i) {
        IRInstruction const &x = a.instructions[i], &y = b.instructions[i];
        if (x.op != y.op || x.dst != y.dst || x.a != y.a || x.b != y.b || x.c != y.c ||
            x.variable != y.variable || x.immediate != y.immediate ||
            x.compare != y.compare || x.component != y.component)
            return false;
    }
    return true;
}

/// Line endings must not change compilation. A USD-authored source keeps its
/// checkout's line endings, so on Windows the multi-line sources of
/// examples/expression-width-plane.usda reach the compiler with CR characters.
/// All three spellings below must compile to identical IR and evaluate alike.
void CheckLineEndings()
{
    const std::string lf = "$a = 2;  # root\n$a * 3";
    const std::string crlf = "$a = 2;  # root\r\n$a * 3";
    const std::string cr = "$a = 2;  # root\r$a * 3";
    auto base = Compile(lf, Domain::Groom);
    auto windows = Compile(crlf, Domain::Groom);
    auto classic = Compile(cr, Domain::Groom);
    Check(base.ok && windows.ok && classic.ok, "CRLF and lone CR sources compile");
    if (!base.ok || !windows.ok || !classic.ok) return;
    Check(SameIR(base.program.IR(), windows.program.IR()) &&
          SameIR(base.program.IR(), classic.program.IR()),
          "CRLF and lone CR sources lower to identical IR");
    Check(windows.program.Source() == lf && classic.program.Source() == lf,
          "the stored source is the normalized text");
    float expected = 0.0f, fromCrlf = 0.0f, fromCr = 0.0f;
    const bool ran = EvaluateScalar(base.program.IR(), &expected) &&
                     EvaluateScalar(windows.program.IR(), &fromCrlf) &&
                     EvaluateScalar(classic.program.IR(), &fromCr);
    Check(ran && expected == 6.0f && fromCrlf == expected && fromCr == expected,
          "CRLF and lone CR sources evaluate the same as the LF form");
}

} // namespace

int main()
{
    CheckBasics();
    CheckLanguage();
    CheckFunctionTable();
    CheckCurves();
    CheckLimits();
    CheckVariables();
    CheckLineEndings();
    std::printf("testUsdGenExpressionFrontend: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
