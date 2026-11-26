#include "usdGenImaging/usdGenToolsApi.h"

#include "usdGen/expressions/context.h"
#include "usdGen/expressions/frontend.h"

#include <cctype>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using usdGen::expr::Domain;
using usdGen::expr::Frontend;
using usdGen::expr::FrontendOptions;
using usdGen::expr::Registry;
using usdGen::expr::ScalarType;

// --- delimited-text plumbing ----------------------------------------------

// Common tail for the two list entry points: return the size the caller needs
// and fill `buf` only when it fits, so one protocol covers sizing and reading.
int Emit(std::string const &text, char *buf, int len)
{
    const size_t needed = text.size() + 1;
    if (needed > size_t(INT32_MAX)) return -1;
    if (buf && len > 0 && size_t(len) >= needed) {
        std::memcpy(buf, text.c_str(), needed);
    }
    return int(needed);
}

Domain DomainFromCode(int code)
{
    switch (code) {
    case USDGEN_TOOLS_DOMAIN_GROOM: return Domain::Groom;
    case USDGEN_TOOLS_DOMAIN_PRIMITIVE: return Domain::Primitive;
    case USDGEN_TOOLS_DOMAIN_POINT: return Domain::Point;
    default: return Domain::None;
    }
}

// --- diagnostics ----------------------------------------------------------

// 1-based line and column of a character offset. An offset past the end is
// clamped to the end, which is where "unexpected end of expression" points.
void LineColOf(std::string const &source, size_t offset, int *line, int *column)
{
    if (offset > source.size()) offset = source.size();
    int l = 1, c = 1;
    for (size_t i = 0; i < offset; ++i) {
        if (source[i] == '\n') { ++l; c = 1; } else { ++c; }
    }
    *line = l;
    *column = c;
}

// The vendored parser appends " near '<token>':\n    <echo of the source>" to
// every message. An editor is already showing that source with the offending
// line marked, so the echo is noise in a one-line status strip; cut it.
std::string TrimParserEcho(std::string text)
{
    const size_t p = text.find(" near '");
    return p == std::string::npos ? text : text.substr(0, p);
}

std::string Flatten(std::string text)
{
    for (char &ch : text) if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
    // The parser's context line is indented; collapse the run so the message
    // reads as one line in a status strip.
    std::string out;
    out.reserve(text.size());
    bool space = false;
    for (char ch : text) {
        if (ch == ' ') { space = true; continue; }
        if (space && !out.empty()) out.push_back(' ');
        space = false;
        out.push_back(ch);
    }
    return out;
}

// The frontend reports positions inside the diagnostic text, in two shapes:
// its own AST walk appends " at <character offset>" (sometimes followed by
// more prose, e.g. "; supported functions are ..."), while the vendored
// parser's message embeds " at line <n>" and no column. Recover whichever is
// there. Nothing structured is available across that boundary today, so this
// is deliberately a text protocol; it degrades to line 0 column 0 (meaning
// "no position") rather than guessing.
void SplitDiagnostic(std::string const &source, std::string const &diagnostic,
                     int *line, int *column, std::string *message)
{
    *line = 0;
    *column = 0;
    *message = Flatten(TrimParserEcho(diagnostic));

    const std::string atLine = " at line ";
    size_t p = diagnostic.find(atLine);
    if (p != std::string::npos) {
        size_t start = p + atLine.size();
        size_t end = start;
        while (end < diagnostic.size() &&
               std::isdigit(static_cast<unsigned char>(diagnostic[end]))) ++end;
        if (end > start) {
            *line = std::stoi(diagnostic.substr(start, end - start));
            *column = 1;
            return;
        }
    }

    const std::string at = " at ";
    p = diagnostic.rfind(at);
    if (p != std::string::npos) {
        const size_t start = p + at.size();
        size_t end = start;
        while (end < diagnostic.size() &&
               std::isdigit(static_cast<unsigned char>(diagnostic[end]))) ++end;
        if (end > start) {
            LineColOf(source, size_t(std::stoul(diagnostic.substr(start, end - start))),
                      line, column);
            // Everything past the offset (e.g. "; supported functions are
            // ...") is kept -- it is useful context, not parser noise -- with
            // just the "at <offset>" itself excised.
            *message = Flatten(TrimParserEcho(diagnostic.substr(0, p) +
                                              diagnostic.substr(end)));
            return;
        }
    }

    // The vendored parser writes " at line <n>" only when n > 1, so one of its
    // messages WITHOUT a line is on line 1 -- that is a fact about the
    // generated parser, not a guess. Running off the end of the text is the
    // one case where the position worth showing is the end of the source.
    if (diagnostic.find("Unexpected end of expression") == 0) {
        LineColOf(source, source.size(), line, column);
    } else if (diagnostic.find("Syntax error") == 0) {
        *line = 1;
        *column = 1;
    }
}

// --- the function table ---------------------------------------------------

struct FunctionInfo {
    const char *name;
    const char *signature;
    const char *doc;
    const char *insert;  // '|' marks where the cursor lands
    const char *category;
};

// The pure math functions (abs, sin, clamp, sqrt, ...) are NOT listed here:
// they come straight from usdGen::expr::Frontend::SupportedFunctions(), the
// engine's single source of truth (see frontend.h), so the editor can never
// advertise a function the engine refuses to lower. This table carries only
// what that engine table does not: the operators, the ternary and the vector
// literal, which Walk()/Lower() accept as AST node kinds rather than named
// functions and so have no FunctionInfo entry of their own.
constexpr FunctionInfo kOperators[] = {
    {"+", "a + b", "Addition.", " + |", "Operators"},
    {"-", "a - b", "Subtraction, and unary negation.", " - |", "Operators"},
    {"*", "a * b", "Multiplication.", " * |", "Operators"},
    {"/", "a / b", "Division.", " / |", "Operators"},
    {"<", "a < b", "Less than; 1 when true, 0 when false.", " < |", "Operators"},
    {"<=", "a <= b", "Less than or equal.", " <= |", "Operators"},
    {">", "a > b", "Greater than.", " > |", "Operators"},
    {">=", "a >= b", "Greater than or equal.", " >= |", "Operators"},
    {"==", "a == b", "Equal.", " == |", "Operators"},
    {"!=", "a != b", "Not equal.", " != |", "Operators"},
    {"?:", "cond ? a : b", "Conditional select.", " ? | : ", "Operators"},
    {"[]", "[x, y, z]", "Vector literal. Subscript it with a constant index.",
     "[|, 0, 0]", "Vectors"},
};

// The engine's FunctionInfo carries no insert text (Frontend has no notion of
// an editor caret), so synthesize one from the arity alone: the caret lands
// right after the opening paren and a ", " placeholder marks every argument
// after the first, matching the hand-written entries above (e.g. "pow(|, )").
std::string InsertFor(usdGen::expr::FunctionInfo const &fn)
{
    // The sampling functions take string literals, so the insertion carries
    // the quotes and places the caret inside the first.
    if (fn.name == "geoSampler") return "geoSampler(\"|\", \"$index\")";
    if (fn.name == "ptex") return "ptex(\"|\")";
    std::string insert = fn.name + "(|";
    for (uint32_t i = 1; i < fn.arity; ++i) insert += ", ";
    insert += ")";
    return insert;
}

// --- categories -----------------------------------------------------------

// The group a browser files a function under. The engine's FunctionInfo is
// expected to grow a `category` field; until it has one, classify by name so
// the ABI's record shape does not depend on which engine this is built
// against. Detected at compile time rather than guarded by a version macro:
// the field either exists in this translation unit or it does not.
template <typename T, typename = void>
struct HasCategory : std::false_type {};
template <typename T>
struct HasCategory<T, std::void_t<decltype(std::declval<T const &>().category)>>
    : std::true_type {};

struct CategoryRow { const char *name; const char *category; };

// Covers the SeExpr2/a host groomer function set, not only what this engine lowers
// today, so the classification stays right as the frontend grows into it.
constexpr CategoryRow kCategories[] = {
    {"abs", "Math"}, {"ceil", "Math"}, {"floor", "Math"}, {"round", "Math"},
    {"trunc", "Math"}, {"sign", "Math"}, {"exp", "Math"}, {"log", "Math"},
    {"log10", "Math"}, {"pow", "Math"}, {"sqrt", "Math"}, {"cbrt", "Math"},
    {"fmod", "Math"}, {"hypot", "Math"}, {"invert", "Math"}, {"cycle", "Math"},

    {"min", "Interpolation"}, {"max", "Interpolation"},
    {"clamp", "Interpolation"}, {"fit", "Interpolation"},
    {"mix", "Interpolation"}, {"lerp", "Interpolation"},
    {"smoothstep", "Interpolation"}, {"linearstep", "Interpolation"},
    {"boxstep", "Interpolation"}, {"gaussstep", "Interpolation"},
    {"remap", "Interpolation"}, {"bias", "Interpolation"},
    {"contrast", "Interpolation"}, {"gamma", "Interpolation"},
    {"compress", "Interpolation"}, {"expand", "Interpolation"},

    {"sin", "Trigonometry"}, {"cos", "Trigonometry"}, {"tan", "Trigonometry"},
    {"asin", "Trigonometry"}, {"acos", "Trigonometry"}, {"atan", "Trigonometry"},
    {"atan2", "Trigonometry"}, {"sinh", "Trigonometry"}, {"cosh", "Trigonometry"},
    {"tanh", "Trigonometry"}, {"deg", "Trigonometry"}, {"rad", "Trigonometry"},
    {"sind", "Trigonometry"}, {"cosd", "Trigonometry"}, {"tand", "Trigonometry"},
    {"asind", "Trigonometry"}, {"acosd", "Trigonometry"}, {"atand", "Trigonometry"},
    {"atan2d", "Trigonometry"},

    {"noise", "Noise"}, {"snoise", "Noise"}, {"vnoise", "Noise"},
    {"cnoise", "Noise"}, {"snoise4", "Noise"}, {"vnoise4", "Noise"},
    {"cnoise4", "Noise"}, {"turbulence", "Noise"}, {"vturbulence", "Noise"},
    {"cturbulence", "Noise"}, {"fbm", "Noise"}, {"vfbm", "Noise"},
    {"cfbm", "Noise"}, {"fbm4", "Noise"}, {"vfbm4", "Noise"}, {"cfbm4", "Noise"},
    {"cellnoise", "Noise"}, {"ccellnoise", "Noise"}, {"pnoise", "Noise"},
    {"voronoi", "Noise"}, {"svoronoi", "Noise"}, {"cvoronoi", "Noise"},
    {"pvoronoi", "Noise"}, {"hash", "Noise"}, {"rand", "Noise"},

    {"curve", "Curves"}, {"ccurve", "Curves"}, {"spline", "Curves"},

    {"dot", "Vectors"}, {"cross", "Vectors"}, {"norm", "Vectors"},
    {"length", "Vectors"}, {"dist", "Vectors"}, {"up", "Vectors"},
    {"ortho", "Vectors"}, {"rotate", "Vectors"}, {"angle", "Vectors"},
    {"comp", "Vectors"}, {"[]", "Vectors"},

    {"hsi", "Color"}, {"midhsi", "Color"}, {"rgbtohsl", "Color"},
    {"hsltorgb", "Color"}, {"ctransform", "Color"}, {"saturate", "Color"},

    {"choose", "Control"}, {"pick", "Control"}, {"wchoose", "Control"},

    {"printf", "Strings"},

    {"geoSampler", "Sampling"}, {"ptex", "Sampling"},
};

std::string CategoryByName(std::string const &name)
{
    for (auto const &row : kCategories)
        if (name == row.name) return row.category;
    return "Other";
}

template <typename T>
std::string CategoryOf(T const &fn)
{
    // The name table carries the fine-grained grouping the browser wants
    // (Trigonometry, Interpolation, ...); the engine's own category is
    // coarser (math, curve, ...) and is only a fallback for a function the
    // name table does not yet know, so a newly added engine function still
    // gets filed under something more useful than "Other".
    const std::string byName = CategoryByName(std::string(fn.name));
    if (byName != "Other") return byName;
    if constexpr (HasCategory<T>::value) {
        const std::string category(fn.category);
        if (!category.empty()) return category;
    }
    return byName;
}

// One line of prose per variable, keyed by the name in the engine's table.
// The editor's variable browser shows these; keeping them here rather than in
// the Python plugin means one place describes the language.
struct VariableDoc { const char *name; const char *doc; };
constexpr VariableDoc kVariableDocs[] = {
    {"$value", "The consumer's own literal, in the consumer's own type."},
    {"$frame", "Stage time code."},
    {"$time", "Stage time in seconds."},
    {"$index", "Element index in the evaluation domain."},
    {"$count", "Element count in the evaluation domain."},
    {"$seed", "The consuming operator's usdGen:seed."},
    {"$descId", "Stable id of the description being cooked."},
    {"$primIndex", "Strand index."},
    {"$primCount", "Strand count."},
    {"$idLo", "Low 32 bits of the stable strand id."},
    {"$idHi", "High 32 bits of the stable strand id."},
    {"$id", "Stable strand id."},
    {"$u", "Surface u at the strand root."},
    {"$v", "Surface v at the strand root."},
    {"$faceId", "Index of the surface face the strand roots on."},
    {"$patchId", "Index of the surface patch the strand roots on."},
    {"$P", "Current position."},
    {"$Pref", "Rest position."},
    {"$rootP", "Current position of the strand root."},
    {"$rootPref", "Rest position of the strand root."},
    {"$N", "Current surface normal at the root."},
    {"$Nref", "Rest surface normal at the root."},
    {"$dPdu", "Current surface tangent along u at the root."},
    {"$dPdv", "Current surface tangent along v at the root."},
    {"$dPduref", "Rest surface tangent along u at the root."},
    {"$dPdvref", "Rest surface tangent along v at the root."},
    {"$t", "Root-to-tip parameter: 0 at the root, 1 at the tip."},
    {"$pointIndex", "CV index within the strand."},
    {"$pointCount", "CV count of the strand."},
    {"$cLength", "Strand length."},
    {"$cWidth", "Strand width."},
    {"$Cs", "Surface colour at the strand root."},
    {"$As", "Surface alpha at the strand root."},
};

const char *DocFor(const char *name)
{
    for (auto const &d : kVariableDocs) if (std::strcmp(d.name, name) == 0) return d.doc;
    return "";
}

std::string DomainList(Domain domains)
{
    std::string out;
    const Domain bits[] = {Domain::Groom, Domain::Primitive, Domain::Point};
    for (Domain bit : bits) {
        if (!usdGen::expr::HasDomain(domains, bit)) continue;
        if (!out.empty()) out.push_back(',');
        out += usdGen::expr::DomainName(bit);
    }
    return out;
}

}  // namespace

extern "C" {

int UsdGenTools_ApiVersion(void) { return USDGEN_TOOLS_API_VERSION; }

int UsdGenTools_CompileExpression(const char *source, int domain, int components,
                                  char *errBuf, int errLen)
{
    if (errBuf && errLen > 0) errBuf[0] = '\0';
    const Domain d = DomainFromCode(domain);
    if (!source || d == Domain::None || components < 1 || components > 4) {
        return USDGEN_TOOLS_BAD_ARGS;
    }
    try {
        const std::string text(source);
        FrontendOptions options;
        options.domain = d;
        options.destination = ScalarType::Float32;
        options.components = uint32_t(components);
        const auto result = Frontend::Compile(text, options);
        if (result.ok && result.diagnostics.empty()) return USDGEN_TOOLS_OK;

        std::string records;
        for (auto const &diagnostic : result.diagnostics) {
            int line = 0, column = 0;
            std::string message;
            SplitDiagnostic(text, diagnostic, &line, &column, &message);
            if (!records.empty()) records.push_back('\n');
            records += std::to_string(line);
            records.push_back('\t');
            records += std::to_string(column);
            records.push_back('\t');
            records += message;
        }
        if (records.empty()) records = "0\t0\texpression did not compile";
        if (errBuf && errLen > 0) {
            const size_t copy = records.size() < size_t(errLen) - 1
                                    ? records.size()
                                    : size_t(errLen) - 1;
            std::memcpy(errBuf, records.c_str(), copy);
            errBuf[copy] = '\0';
        }
        return USDGEN_TOOLS_DIAGNOSTICS;
    } catch (...) {
        // Nothing may cross the boundary. An unexpected throw is reported as a
        // positionless diagnostic rather than as success.
        const char *fallback = "0\t0\tinternal error while compiling the expression";
        if (errBuf && errLen > 0) {
            const size_t copy = std::strlen(fallback) < size_t(errLen) - 1
                                    ? std::strlen(fallback)
                                    : size_t(errLen) - 1;
            std::memcpy(errBuf, fallback, copy);
            errBuf[copy] = '\0';
        }
        return USDGEN_TOOLS_DIAGNOSTICS;
    }
}

int UsdGenTools_ListVariables(int domain, char *buf, int len)
{
    try {
        const Domain requested = DomainFromCode(domain);
        auto const &registry = Registry::Get();
        std::string text;
        for (size_t i = 0; i < registry.Count(); ++i) {
            const auto *info = registry.At(i);
            if (!info) break;
            // $Q/$Qdist exist only inside a geoSampler() element expression.
            const bool sampler = Registry::IsSamplerVariable(info->id);
            const bool valid = !sampler &&
                (requested == Domain::None ||
                 usdGen::expr::HasDomain(info->domains, requested));
            if (!text.empty()) text.push_back('\n');
            text += info->name;
            text.push_back('\t');
            text += usdGen::expr::ScalarTypeName(info->scalar);
            text.push_back('\t');
            text += std::to_string(info->components);
            text.push_back('\t');
            text += sampler ? std::string("sampler") : DomainList(info->domains);
            text.push_back('\t');
            text += valid ? "1" : "0";
            text.push_back('\t');
            // The local table words the editor's prose; the registry's own
            // line covers anything it does not know yet.
            const char *doc = DocFor(info->name);
            text += *doc ? doc : (info->doc ? info->doc : "");
        }
        return Emit(text, buf, len);
    } catch (...) {
        return -1;
    }
}

int UsdGenTools_ListFunctions(char *buf, int len)
{
    try {
        std::string text;
        auto appendRow = [&](std::string const &name, std::string const &signature,
                             std::string const &doc, std::string const &insert,
                             std::string const &category) {
            if (!text.empty()) text.push_back('\n');
            text += name;
            text.push_back('\t');
            text += signature;
            text.push_back('\t');
            text += doc;
            text.push_back('\t');
            text += insert;
            text.push_back('\t');
            text += category;
        };
        // The pure functions, straight from the engine's single source of
        // truth, so this list can never advertise a function Lower() refuses.
        for (auto const &fn : Frontend::SupportedFunctions())
            appendRow(fn.name, fn.signature, fn.doc, InsertFor(fn), CategoryOf(fn));
        // The operators, ternary and vector literal Walk()/Lower() accept as
        // AST node kinds rather than named functions.
        for (auto const &op : kOperators)
            appendRow(op.name, op.signature, op.doc, op.insert, CategoryOf(op));
        return Emit(text, buf, len);
    } catch (...) {
        return -1;
    }
}

}  // extern "C"
