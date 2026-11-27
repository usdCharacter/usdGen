#include "usdGenExpr/expressions/frontend.h"

#include "usdGenExpr/expressions/irExec.h"

// This translation unit is compiled with SeExpr2=UsdGenSeExprFrontend so the
// complete vendored parser is private and cannot collide with the legacy noise
// subset's symbols.
#include <Expression.h>
#if defined(_MSC_VER)
// ExprFuncX.h, reached through ExprFunc.h, has an unnamed formal parameter.
// Vendored code: silence it here rather than diverge from upstream.
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
#include <ExprFunc.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#include <ExprNode.h>
#include <Vec.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace SE = UsdGenSeExprFrontend;

namespace usdGen::expr {
namespace {

// ---------------------------------------------------------------------------
// Four names the stock SeExpr2 table gets wrong for usdGen, declared through
// the per-expression resolveFunc hook rather than ExprFunc::define. The global
// table is process-wide and its own header calls the define path not
// thread-safe; resolveFunc is consulted FIRST (ExprNode.cpp, ExprFuncNode::prep)
// and touches nothing outside the expression being compiled, which is the
// pattern plan/07 §7.3 settled on.
//
//   cbrt, trunc  guarded with SEEXPR_WIN32 in ExprBuiltins.cpp because of an
//                old overload-resolution problem with ::cbrt, so on Windows
//                they are simply absent. The language must not depend on the
//                platform it was compiled for.
//   rand         a host groomer's, not SeExpr2's: the parser has never heard of it.
//   dist         bound in ExprBuiltins.cpp as six scalars even though its own
//                docstring, a host groomer's reference and every other vector builtin
//                spell it dist(vector, vector).
//
// All four are STUBS. They exist so the type checker knows each name's shape;
// nothing here is ever evaluated, because the program is lowered to usdGen's
// own IR (LowerFunction below) and SeExpr2's evaluator never runs.
// ---------------------------------------------------------------------------
double StubCbrt(double) { return 0.0; }
double StubTrunc(double) { return 0.0; }
double StubRand(int, double *) { return 0.0; }
double StubDist(const SE::Vec3d &, const SE::Vec3d &) { return 0.0; }

// geoSampler() and ptex() take string arguments, which no ExprFunc
// function-pointer shape can type-check, so they are ExprFuncSimple stubs.
// Like the four above they only describe the call's type; the call is lowered
// to IROp::Sample and SeExpr2 never evaluates it.
class SamplerStub final : public SE::ExprFuncSimple {
public:
    explicit SamplerStub(SamplerKind kind) : ExprFuncSimple(true), _kind(kind) {}
    SE::ExprType prep(SE::ExprFuncNode *node, bool scalarWanted,
                      SE::ExprVarEnvBuilder &envBuilder) const override;
    SE::ExprFuncNode::Data *evalConstant(const SE::ExprFuncNode *, ArgHandle) const override
    {
        return new SE::ExprFuncNode::Data(true);
    }
    void eval(ArgHandle) override {}
private:
    SamplerKind _kind;
};

// The stub table, made at load and read-only afterwards; never reachable
// from another SeExpr consumer.
SE::ExprFunc cbrtFunc(&StubCbrt);
SE::ExprFunc truncFunc(&StubTrunc);
SE::ExprFunc randFunc(&StubRand, 0, 3);
SE::ExprFunc distFunc(&StubDist);
SamplerStub geoSamplerStub(SamplerKind::Geometry);
SamplerStub ptexStub(SamplerKind::Ptex);
SE::ExprFunc geoSamplerFunc(geoSamplerStub, 2, 5);
SE::ExprFunc ptexFunc(ptexStub, 1, 1);

class Var final : public SE::ExprVarRef {
public:
    explicit Var(int dim) : ExprVarRef(SE::ExprType().FP(dim).Varying()) {}
    void eval(double *out) override { for (int i=0;i<type().dim();++i) out[i]=0.0; }
    void eval(const char **) override {}
};
/// Folds CRLF and lone CR to LF. A USD-authored source keeps its checkout's
/// line endings, so on Windows a multi-line expression reaches the SeExpr
/// lexer containing CR characters, which it rejects ("Syntax error ... near
/// CR"). Normalizing here -- the one place source text enters the SeExpr
/// compiler -- makes every checkout compile identically.
std::string NormalizeLineEndings(std::string const &source)
{
    if (source.find('\r') == std::string::npos) return source;
    std::string out;
    out.reserve(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        if (source[i] == '\r') {
            out.push_back('\n');
            if (i + 1 < source.size() && source[i + 1] == '\n') ++i;
        } else {
            out.push_back(source[i]);
        }
    }
    return out;
}
class CheckedExpression final : public SE::Expression {
public:
    CheckedExpression(std::string const &s, int dim, Domain domain, bool samplerElement = false) : Expression(NormalizeLineEndings(s), SE::ExprType().FP(dim), Expression::UseInterpreter), _domain(domain), _valueDim(dim), _samplerElement(samplerElement) {}
    SE::ExprFunc *resolveFunc(const std::string &name) const override {
        if (name == "cbrt") return &cbrtFunc;
        if (name == "trunc") return &truncFunc;
        if (name == "rand") return &randFunc;
        if (name == "dist") return &distFunc;
        // An element expression cannot sample again; Walk() says so first.
        if (!_samplerElement && name == "geoSampler") return &geoSamplerFunc;
        if (!_samplerElement && name == "ptex") return &ptexFunc;
        return nullptr;
    }
    SE::ExprVarRef *resolveVar(const std::string &name) const override {
        std::string canonical = name;
        if (canonical.empty() || canonical[0] != '$') canonical.insert(canonical.begin(), '$');
        auto info = Registry::Get().Find(canonical.c_str());
        if (!info) return nullptr;
        if (!Registry::Get().Validate(canonical.c_str(), _domain, nullptr, _samplerElement)) return nullptr;
        const int dim = info->components ? int(info->components) : (info->id == Variable::Value ? _valueDim : 1);
        _vars.emplace_back(new Var(dim)); return _vars.back().get();
    }
private:
    mutable std::vector<std::unique_ptr<Var>> _vars;
    Domain _domain;
    int _valueDim;
    bool _samplerElement;
};

bool Walk(SE::ExprNode const *node, std::vector<std::string> &errors, bool samplerElement);

/// The natural width of a geoSampler() element expression: 1 or 3. Returns 0
/// and fills `error` when the source does not type-check as one.
int ElementExpressionWidth(std::string const &source, std::string *error)
{
    CheckedExpression expression(source, 3, Domain::Point, /*samplerElement=*/true);
    // The AST walk first, as Compile() does, so a refused construct is named
    // rather than surfacing as an undefined-function type error.
    if (SE::ExprNode const *tree = expression.parseTree()) {
        std::vector<std::string> refusals;
        if (!Walk(tree, refusals, /*samplerElement=*/true)) {
            if (error) *error = refusals.empty() ? "refused construct" : refusals.front();
            return 0;
        }
    }
    if (!expression.isValid()) {
        if (error) {
            *error = expression.parseError();
            for (auto const &e : expression.getErrors()) {
                if (!error->empty()) *error += "; ";
                *error += e.error;
            }
            if (error->empty()) *error = "the element expression does not compile";
        }
        return 0;
    }
    SE::ExprType const type = expression.returnType();
    if (!type.isFP() || (type.dim() != 1 && type.dim() != 3)) {
        if (error) *error = "the element expression must produce a scalar or a 3-vector";
        return 0;
    }
    return type.dim();
}

SE::ExprType SamplerStub::prep(SE::ExprFuncNode *node, bool, SE::ExprVarEnvBuilder &envBuilder) const
{
    const int count = node->numChildren();
    bool valid = true;
    const SE::ExprType string = SE::ExprType().String().Constant();
    if (_kind == SamplerKind::Ptex) {
        valid = node->checkArg(0, string, envBuilder) && valid;
        return valid ? SE::ExprType().FP(1).Varying() : SE::ExprType().Error().Varying();
    }
    for (int i = 0; i < count && i < 4; ++i)
        valid = node->checkArg(i, string, envBuilder) && valid;
    if (count == 5)
        valid = node->checkArg(4, SE::ExprType().FP(3).Varying(), envBuilder) && valid;
    if (!valid) return SE::ExprType().Error().Varying();
    std::string error;
    const int width = ElementExpressionWidth(node->getStrArg(1), &error);
    if (!width) {
        node->addError("geoSampler element expression: " + error);
        return SE::ExprType().Error().Varying();
    }
    return SE::ExprType().FP(width).Varying();
}

// ---------------------------------------------------------------------------
// The ONE function table. Walk() admits exactly these names and Lower()
// implements exactly these names, so a function can never pass the AST walk
// and then fail in lowering. SupportedFunctions() is this table, so the
// editor's browser, the docs and the compiler cannot disagree.
//
// `componentWise` marks the scalar math functions that SeExpr applies per
// component when handed a vector, which is how the rest of the lowering
// already treats arithmetic. Everything else has a fixed result width.
// ---------------------------------------------------------------------------
constexpr uint32_t kVariadic = 0;   // maxArgs == 0 means "no upper bound"
struct FunctionRow {
    char const *name;
    uint32_t minArgs;
    uint32_t maxArgs;          // 0 == unbounded
    uint8_t components;        // result width
    bool componentWise;
    char const *category;
    char const *signature;
    char const *doc;
};
constexpr FunctionRow kFunctions[] = {
 // --- math -------------------------------------------------------------
 {"abs",1,1,1,true,"math","abs(x)","Absolute value."},
 {"acos",1,1,1,true,"math","acos(x)","Arc cosine, radians."},
 {"acosd",1,1,1,true,"math","acosd(x)","Arc cosine, degrees."},
 {"asin",1,1,1,true,"math","asin(x)","Arc sine, radians."},
 {"asind",1,1,1,true,"math","asind(x)","Arc sine, degrees."},
 {"atan",1,1,1,true,"math","atan(x)","Arc tangent, radians."},
 {"atan2",2,2,1,true,"math","atan2(y, x)","Arc tangent of y/x using the signs to pick the quadrant."},
 {"atan2d",2,2,1,true,"math","atan2d(y, x)","atan2 in degrees."},
 {"atand",1,1,1,true,"math","atand(x)","Arc tangent, degrees."},
 {"bias",2,2,1,true,"math","bias(x, b)","pow(x, log(b)/log(0.5)); b below 0.5 pulls the curve down."},
 {"boxstep",2,2,1,true,"math","boxstep(x, a)","0 when x < a, otherwise 1."},
 {"cbrt",1,1,1,true,"math","cbrt(x)","Cube root."},
 {"ceil",1,1,1,true,"math","ceil(x)","Smallest integer not less than x."},
 {"clamp",3,3,1,true,"math","clamp(x, lo, hi)","x limited to [lo, hi]; invalid when hi < lo."},
 {"compress",3,3,1,true,"math","compress(x, lo, hi)","Remaps x in [0,1] to [lo,hi]."},
 {"contrast",2,2,1,true,"math","contrast(x, c)","Decreases contrast for c < 0.5, increases it above."},
 {"cos",1,1,1,true,"math","cos(x)","Cosine of x in radians."},
 {"cosd",1,1,1,true,"math","cosd(x)","Cosine of x in degrees."},
 {"cosh",1,1,1,true,"math","cosh(x)","Hyperbolic cosine."},
 {"cycle",3,3,1,true,"math","cycle(index, lo, hi)","Offset modulo: cycles index through the integers lo..hi."},
 {"deg",1,1,1,true,"math","deg(angle)","Radians to degrees."},
 {"exp",1,1,1,true,"math","exp(x)","e raised to x."},
 {"expand",3,3,1,true,"math","expand(x, lo, hi)","Remaps x in [lo,hi] to [0,1]."},
 {"fit",5,5,1,true,"math","fit(x, a1, b1, a2, b2)","Linearly remaps x from [a1,b1] to [a2,b2] without clamping."},
 {"floor",1,1,1,true,"math","floor(x)","Largest integer not greater than x."},
 {"fmod",2,2,1,true,"math","fmod(x, y)","Floating-point remainder of x/y, with the sign of x; also the % operator."},
 {"gamma",2,2,1,true,"math","gamma(x, g)","pow(x, 1/g)."},
 {"gaussstep",3,3,1,true,"math","gaussstep(x, a, b)","0 at x <= a, 1 at x >= b, exponential in between."},
 {"hypot",2,2,1,true,"math","hypot(x, y)","Length of the 2D vector [x, y]."},
 {"invert",1,1,1,true,"math","invert(x)","1 - x."},
 {"linearstep",3,3,1,true,"math","linearstep(x, a, b)","0 at x <= a, 1 at x >= b, linear in between."},
 {"log",1,1,1,true,"math","log(x)","Natural logarithm."},
 {"log10",1,1,1,true,"math","log10(x)","Base 10 logarithm."},
 {"max",2,2,1,true,"math","max(a, b)","Larger of a and b."},
 {"min",2,2,1,true,"math","min(a, b)","Smaller of a and b."},
 {"mix",3,3,1,true,"math","mix(a, b, alpha)","a*(1-alpha) + b*alpha."},
 {"pow",2,2,1,true,"math","pow(x, y)","x raised to y; also the ^ operator."},
 {"rad",1,1,1,true,"math","rad(angle)","Degrees to radians."},
 {"remap",5,5,1,true,"math","remap(x, source, range, falloff, interp)","1 within +/- range of source, falling to 0 over falloff; interp 0 linear, 1 smooth, 2 gaussian."},
 {"round",1,1,1,true,"math","round(x)","Nearest integer, halves away from zero."},
 {"sin",1,1,1,true,"math","sin(x)","Sine of x in radians."},
 {"sind",1,1,1,true,"math","sind(x)","Sine of x in degrees."},
 {"sinh",1,1,1,true,"math","sinh(x)","Hyperbolic sine."},
 {"smoothstep",3,3,1,true,"math","smoothstep(x, a, b)","0 at x <= a, 1 at x >= b, cubic in between (SeExpr argument order)."},
 {"sqrt",1,1,1,true,"math","sqrt(x)","Square root; negative x is invalid."},
 {"tan",1,1,1,true,"math","tan(x)","Tangent of x in radians."},
 {"tand",1,1,1,true,"math","tand(x)","Tangent of x in degrees."},
 {"tanh",1,1,1,true,"math","tanh(x)","Hyperbolic tangent."},
 {"trunc",1,1,1,true,"math","trunc(x)","Nearest integer toward zero."},
 // --- noise ------------------------------------------------------------
 {"ccellnoise",1,1,3,false,"noise","ccellnoise(vector v)","Colour cellnoise: a field of constant-coloured unit cubes."},
 {"cellnoise",1,1,1,false,"noise","cellnoise(vector v)","A field of constant-valued unit cubes keyed on the integer location."},
 {"cfbm",1,4,3,false,"noise","cfbm(vector v, octaves=6, lacunarity=2, gain=0.5)","Colour fbm, remapped to [0,1]."},
 {"cnoise",1,1,3,false,"noise","cnoise(vector v)","Colour Perlin noise, remapped to [0,1]."},
 {"cturbulence",1,4,3,false,"noise","cturbulence(vector v, octaves=6, lacunarity=2, gain=0.5)","Colour turbulence, remapped to [0,1]."},
 {"cvoronoi",1,7,3,false,"noise","cvoronoi(vector v, type=1, jitter=0.5, fbmScale=0, fbmOctaves=4, fbmLacunarity=2, fbmGain=0.5)","Colour voronoi: a jittered variant of cellnoise."},
 {"fbm",1,4,1,false,"noise","fbm(vector v, octaves=6, lacunarity=2, gain=0.5)","Fractal Brownian motion, remapped to [0,1]."},
 {"hash",1,kVariadic,1,false,"noise","hash(s1, [s2, ...])","A repeatable random number in [0,1] from any number of seeds."},
 {"noise",1,3,1,false,"noise","noise(vector v) | noise(x, y, z)","Perlin noise remapped to [0,1]."},
 {"pnoise",2,2,1,false,"noise","pnoise(vector v, vector period)","Periodic Perlin noise; the period is a per-axis integer."},
 {"pvoronoi",1,6,3,false,"noise","pvoronoi(vector v, jitter=0.5, fbmScale=0, fbmOctaves=4, fbmLacunarity=2, fbmGain=0.5)","Centre of the voronoi cell containing v."},
 {"rand",0,3,1,false,"noise","rand() | rand(seed) | rand(min, max, seed)","Repeatable random number, stable per strand: hashed from $seed, $id and the call site."},
 {"snoise",1,1,1,false,"noise","snoise(vector v)","Signed Perlin noise in [-1,1]."},
 {"turbulence",1,4,1,false,"noise","turbulence(vector v, octaves=6, lacunarity=2, gain=0.5)","Perlin turbulence, remapped to [0,1]."},
 {"vfbm",1,4,3,false,"noise","vfbm(vector v, octaves=6, lacunarity=2, gain=0.5)","Vector fbm."},
 {"vnoise",1,1,3,false,"noise","vnoise(vector v)","Vector Perlin noise in [-1,1]."},
 {"voronoi",1,7,1,false,"noise","voronoi(vector v, type=1, jitter=0.5, fbmScale=0, fbmOctaves=4, fbmLacunarity=2, fbmGain=0.5)","Voronoi cellular noise; type 1 cell id, 2 f1, 3 f2, 4 f2-f1, 5 edges."},
 {"vturbulence",1,4,3,false,"noise","vturbulence(vector v, octaves=6, lacunarity=2, gain=0.5)","Vector turbulence."},
 // --- vector -----------------------------------------------------------
 {"angle",2,2,1,false,"vector","angle(vector a, vector b)","Angle between two vectors, radians."},
 {"cross",2,2,3,false,"vector","cross(vector a, vector b)","Vector cross product."},
 {"dist",2,2,1,false,"vector","dist(vector a, vector b)","Distance between two points."},
 {"dot",2,2,1,false,"vector","dot(vector a, vector b)","Vector dot product."},
 {"length",1,1,1,false,"vector","length(vector v)","Length of a vector."},
 {"norm",1,1,3,false,"vector","norm(vector v)","v scaled to unit length; the zero vector stays zero."},
 {"ortho",2,2,3,false,"vector","ortho(vector a, vector b)","Unit vector orthogonal to a and b."},
 {"rotate",3,3,3,false,"vector","rotate(vector v, vector axis, angle)","Rotates v around axis by angle, radians."},
 {"up",2,2,3,false,"vector","up(vector P, vector upvec)","Rotates P so that the Y axis points along upvec."},
 // --- color ------------------------------------------------------------
 {"hsi",4,5,3,false,"color","hsi(color c, h, s, i, map=1)","Shifts hue by h degrees and scales saturation and intensity."},
 {"hsltorgb",1,1,3,false,"color","hsltorgb(color hsl)","HSL to RGB."},
 {"midhsi",5,7,3,false,"color","midhsi(color c, h, s, i, map, falloff=1, interp=0)","hsi with the map centred on 0.5 so it shifts both ways."},
 {"rgbtohsl",1,1,3,false,"color","rgbtohsl(color rgb)","RGB to HSL."},
 {"saturate",2,2,3,false,"color","saturate(color c, amount)","Scales saturation around the rec709 luminance."},
 // --- curve ------------------------------------------------------------
 {"ccurve",4,kVariadic,3,false,"curve","ccurve(param, pos0, color val0, interp0, ...)","Colour control curve; positions and colours must be constants."},
 {"curve",4,kVariadic,1,false,"curve","curve(param, pos0, val0, interp0, ...)","Control curve; interp 0 none, 1 linear, 2 smooth, 3 spline, 4 monotone spline. Knots must be constants."},
 {"spline",5,kVariadic,1,false,"curve","spline(param, y1, y2, y3, y4, ...)","Catmull-Rom through values spread evenly over [0,1]."},
 // --- control ----------------------------------------------------------
 {"choose",3,kVariadic,1,false,"control","choose(index, c1, c2, ...)","Picks one choice from index in [0,1]."},
 {"pick",3,kVariadic,1,false,"control","pick(index, lo, hi, [weights...])","Random integer in lo..hi from a hashed index, distributed by weights."},
 {"wchoose",5,kVariadic,1,false,"control","wchoose(index, c1, w1, c2, w2, ...)","choose() with per-choice weights."},
 // --- sampling: external data named by the expression prim's input:<name> ---
 // geoSampler's width is its element expression's (1 or 3); Dim() asks the
 // resolved sampler rather than this row.
 {"geoSampler",2,5,1,false,"sampling","geoSampler(string input, string expression, string iterate=\"prim\", string reduce=\"nearest\", vector query=$P)","Evaluates a one-line expression over the point, prim or geometry elements of the input:<input> geometry and returns the nearest element's value (reduce: nearest, nearest2, min, max, sum, mean). Element variables: $P $Pref $N $rootP $index $count $id $primIndex $pointIndex $t $cLength, the query $Q and $Qdist."},
 {"ptex",1,1,1,false,"sampling","ptex(string input)","The UsdGenPtexMap or UsdGenPaintMap named by input:<input>, read at the strand root (channel, scale, offset and clamp come from the map prim; a paint map reads the face average of its snapshot)."},
};
constexpr size_t kFunctionCount = sizeof(kFunctions) / sizeof(kFunctions[0]);

FunctionRow const *FindFunction(char const *name)
{
    if (!name) return nullptr;
    for (auto const &row : kFunctions) if (std::strcmp(name, row.name) == 0) return &row;
    return nullptr;
}
std::string SupportedFunctionNames()
{
    std::string names;
    for (auto const &row : kFunctions) { if (!names.empty()) names += ", "; names += row.name; }
    return names;
}
std::string ArityText(FunctionRow const &row)
{
    if (row.maxArgs == kVariadic)
        return "at least " + std::to_string(row.minArgs) + " arguments";
    if (row.minArgs == row.maxArgs)
        return std::to_string(row.minArgs) + (row.minArgs == 1 ? " argument" : " arguments");
    return std::to_string(row.minArgs) + " to " + std::to_string(row.maxArgs) + " arguments";
}

/// Names a user may reasonably try that usdGen deliberately does not provide,
/// each with a reason rather than the generic "unsupported function" list.
struct RefusalRow { char const *name; char const *reason; };
constexpr RefusalRow kRefusals[] = {
 {"printf", "printf/sprintf have no output in a cooked groom"},
 {"sprintf", "printf/sprintf have no output in a cooked groom"},
 {"map", "image maps are not yet available in expressions; ptex(\"<input>\") reads a UsdGenPtexMap"},
 {"texture", "image maps are not yet available in expressions; ptex(\"<input>\") reads a UsdGenPtexMap"},
 {"file", "an expression may not read the filesystem"},
 {"system", "an expression may not run a command"},
 {"exec", "an expression may not run a command"},
 {"noise4", "only 3D noise is available; usdGen vendors the 3D gradient table"},
 {"snoise4", "only 3D noise is available; usdGen vendors the 3D gradient table"},
 {"vnoise4", "only 3D noise is available; usdGen vendors the 3D gradient table"},
 {"cnoise4", "only 3D noise is available; usdGen vendors the 3D gradient table"},
 {"fbm4", "only 3D noise is available; usdGen vendors the 3D gradient table"},
 {"vfbm4", "only 3D noise is available; usdGen vendors the 3D gradient table"},
 {"cfbm4", "only 3D noise is available; usdGen vendors the 3D gradient table"},
 {"swatch", "use choose(); swatch is an alias SeExpr keeps for colour swatches"},
};
char const *FindRefusal(char const *name)
{
    if (!name) return nullptr;
    for (auto const &row : kRefusals) if (std::strcmp(name, row.name) == 0) return row.reason;
    return nullptr;
}

std::string Canonical(std::string name)
{
    if (name.empty() || name[0] != '$') name.insert(name.begin(), '$');
    return name;
}

std::string At(SE::ExprNode const *node)
{
    return " at " + std::to_string(node ? node->startPos() : 0);
}

bool IsStatement(SE::ExprNode const *node)
{
    return dynamic_cast<SE::ExprAssignNode const *>(node) ||
           dynamic_cast<SE::ExprIfThenElseNode const *>(node);
}

// ---------------------------------------------------------------------------
// The AST walk. Everything rejected is rejected HERE, before prep, so the
// diagnostic names the construct and its position rather than surfacing a
// parser-internal type error.
// ---------------------------------------------------------------------------
bool IsSamplerCall(char const *name)
{
    return name && (std::strcmp(name, "geoSampler") == 0 || std::strcmp(name, "ptex") == 0);
}

/// Whether a sampler call's argument `index` is one of its string literals.
bool SamplerStringArg(char const *name, int index)
{
    if (std::strcmp(name, "ptex") == 0) return index == 0;
    return index < 4;   // geoSampler: input, expression, iterate, reduce
}

bool Walk(SE::ExprNode const *node, std::vector<std::string> &errors, bool samplerElement)
{
    if (!node) return true;
    if (auto fn = dynamic_cast<SE::ExprFuncNode const *>(node)) {
        char const *name = fn->name();
        if (IsSamplerCall(name)) {
            if (samplerElement) {
                errors.push_back(std::string("expression function '") + name +
                                 "' cannot be used inside a geoSampler() element expression" +
                                 At(fn));
                return false;
            }
            FunctionRow const *row = FindFunction(name);
            const int count = fn->numChildren();
            if (count < int(row->minArgs) || count > int(row->maxArgs)) {
                errors.push_back(std::string("expression function '") + name + "' takes " +
                                 ArityText(*row) + " (" + row->signature + ")" + At(fn));
                return false;
            }
            bool ok = true;
            for (int i = 0; i < count; ++i) {
                SE::ExprNode const *arg = fn->child(i);
                const bool isString = dynamic_cast<SE::ExprStrNode const *>(arg) != nullptr;
                if (SamplerStringArg(name, i)) {
                    if (!isString) {
                        errors.push_back(std::string(name) + "() argument " +
                                         std::to_string(i + 1) +
                                         " must be a string literal (" + row->signature + ")" +
                                         At(arg));
                        ok = false;
                    }
                    continue;
                }
                ok = Walk(arg, errors, samplerElement) && ok;
            }
            return ok;
        }
        if (FunctionRow const *row = FindFunction(name)) {
            const uint32_t count = uint32_t(fn->numChildren());
            if (count < row->minArgs || (row->maxArgs != kVariadic && count > row->maxArgs)) {
                errors.push_back(std::string("expression function '") + name + "' takes " +
                                 ArityText(*row) + " (" + row->signature + ")" + At(fn));
                return false;
            }
            if ((std::strcmp(name, "curve") == 0 || std::strcmp(name, "ccurve") == 0) &&
                (count - 1) % 3 != 0) {
                errors.push_back(std::string(name) +
                                 "() takes a parameter followed by (position, value, interpolation)"
                                 " triples" + At(fn));
                return false;
            }
            if (std::strcmp(name, "noise") == 0 && count == 2) {
                errors.push_back("noise(x, y) is 2D and unavailable; usdGen vendors the 3D"
                                 " gradient table, so pass a vector or three scalars" + At(fn));
                return false;
            }
        } else if (char const *reason = FindRefusal(name)) {
            errors.push_back(std::string("expression function '") + name +
                             "' is not available in usdGen expressions: " + reason + At(fn));
            return false;
        } else {
            errors.push_back(std::string("unsupported expression function '") + name + "'" +
                             At(fn) + "; supported functions are " + SupportedFunctionNames());
            return false;
        }
    }
    if (dynamic_cast<SE::ExprStrNode const *>(node)) {
        errors.push_back("string literals are not supported in numeric expressions" + At(node));
        return false;
    }
    if (dynamic_cast<SE::ExprLocalFunctionNode const *>(node) ||
        dynamic_cast<SE::ExprPrototypeNode const *>(node)) {
        errors.push_back("user-defined functions (def) are not supported in usdGen expressions" +
                         At(node));
        return false;
    }
    if (!samplerElement) {
        if (auto var = dynamic_cast<SE::ExprVarNode const *>(node)) {
            const std::string name = Canonical(var->name());
            VariableInfo const *info = Registry::Get().Find(name.c_str());
            if (info && Registry::IsSamplerVariable(info->id)) {
                errors.push_back("variable " + name +
                                 " is only available inside a geoSampler() element expression" +
                                 At(node));
                return false;
            }
        }
    }
    if (auto assign = dynamic_cast<SE::ExprAssignNode const *>(node)) {
        const std::string name = Canonical(assign->name());
        if (Registry::Get().Find(name.c_str())) {
            errors.push_back("cannot assign to the built-in expression variable " + name +
                             At(node));
            return false;
        }
    }
    bool ok = true;
    for (int i = 0; i < node->numChildren(); ++i) ok = Walk(node->child(i), errors, samplerElement) && ok;
    return ok;
}

// ---------------------------------------------------------------------------
// Lowering
// ---------------------------------------------------------------------------

/// One local variable: up to four component registers.
struct Local {
    uint16_t reg[4]{};
    uint8_t components = 1;
};
using Symbols = std::map<std::string, Local>;

/// One entry of a Call's argument block: either a value already in a register
/// or a compile-time constant that will be materialised in place.
struct Slot {
    bool isConst = false;
    double value = 0;
    uint16_t reg = 0;
};

using CseKey = std::tuple<int, uint16_t, uint16_t, uint16_t, int, uint64_t, char, uint8_t>;

class Lowerer {
public:
    Lowerer(IRProgram &ir, std::vector<std::string> &errors, Domain domain, int valueDim,
            bool samplerElement)
        : ir_(ir), errors_(errors), domain_(domain), valueDim_(valueDim),
          samplerElement_(samplerElement) {}

    /// Resolves every geoSampler()/ptex() call site into IRProgram::samplers,
    /// compiling each element expression once. Identical call sites share a
    /// slot. Runs before lowering so Dim() knows a sampler's width.
    void CollectSamplers(SE::ExprNode const *node)
    {
        if (!node) return;
        if (auto fn = dynamic_cast<SE::ExprFuncNode const *>(node))
            if (IsSamplerCall(fn->name())) ResolveSampler(fn);
        for (int i = 0; i < node->numChildren(); ++i) CollectSamplers(node->child(i));
    }

    /// Numbers every rand() call site up front so its hash seed does not depend
    /// on the order components happen to be lowered in.
    void NumberRandCalls(SE::ExprNode const *node)
    {
        if (!node) return;
        if (auto fn = dynamic_cast<SE::ExprFuncNode const *>(node))
            if (std::strcmp(fn->name(), "rand") == 0)
                randIndex_.emplace(node, uint32_t(randIndex_.size()));
        for (int i = 0; i < node->numChildren(); ++i) NumberRandCalls(node->child(i));
    }

    /// Lowers the whole module: statements once, then the result expression
    /// once per output component.
    bool LowerProgram(SE::ExprNode const *tree, uint8_t components, uint16_t *outputs)
    {
        SE::ExprNode const *node = tree;
        if (auto module = dynamic_cast<SE::ExprModuleNode const *>(node)) {
            if (module->numChildren() != 1) {
                errors_.push_back("user-defined functions (def) are not supported in usdGen"
                                  " expressions" + At(node));
                return false;
            }
            node = module->child(0);
        }
        SE::ExprNode const *value = node;
        if (auto block = dynamic_cast<SE::ExprBlockNode const *>(node)) {
            if (block->numChildren() != 2) {
                errors_.push_back("malformed expression block" + At(node));
                return false;
            }
            LowerStatements(block->child(0));
            value = block->child(1);
        }
        for (uint8_t c = 0; c < components && errors_.empty(); ++c)
            outputs[c] = Lower(value, c);
        return errors_.empty();
    }

private:
    IRProgram &ir_;
    std::vector<std::string> &errors_;
    Domain domain_;
    int valueDim_;
    bool samplerElement_;
    Symbols symbols_;
    std::map<CseKey, uint16_t> cse_;
    std::map<SE::ExprNode const *, uint32_t> randIndex_;
    std::map<SE::ExprNode const *, uint16_t> samplerSlot_;

    bool Fail(std::string const &message) { errors_.push_back(message); return false; }

    void ResolveSampler(SE::ExprFuncNode const *fn)
    {
        IRSampler sampler;
        const bool geometry = std::strcmp(fn->name(), "geoSampler") == 0;
        const int count = fn->numChildren();
        sampler.kind = geometry ? SamplerKind::Geometry : SamplerKind::Ptex;
        sampler.input = fn->getStrArg(0);
        std::string const call = std::string(fn->name()) + "(\"" + sampler.input + "\")";
        if (sampler.input.empty()) {
            Fail(std::string(fn->name()) + "() needs the name of an input:<name> relationship" + At(fn));
            return;
        }
        if (!geometry) {
            if (domain_ == Domain::Groom) {
                Fail(call + " reads the map at the strand root and needs primitive or point"
                     " evaluation" + At(fn));
                return;
            }
        } else {
            sampler.source = fn->getStrArg(1);
            if (count > 2) {
                std::string const iterate = fn->getStrArg(2);
                if (iterate == "point") sampler.iterate = SampleIterate::Point;
                else if (iterate == "prim") sampler.iterate = SampleIterate::Prim;
                else if (iterate == "geometry") sampler.iterate = SampleIterate::Geometry;
                else {
                    Fail(call + ": iterate must be \"point\", \"prim\" or \"geometry\", not \"" +
                         iterate + "\"" + At(fn));
                    return;
                }
            }
            if (count > 3) {
                std::string const reduce = fn->getStrArg(3);
                if (reduce == "nearest") sampler.reduce = SampleReduce::Nearest;
                else if (reduce == "nearest2") sampler.reduce = SampleReduce::Nearest2;
                else if (reduce == "min") sampler.reduce = SampleReduce::Min;
                else if (reduce == "max") sampler.reduce = SampleReduce::Max;
                else if (reduce == "sum") sampler.reduce = SampleReduce::Sum;
                else if (reduce == "mean") sampler.reduce = SampleReduce::Mean;
                else {
                    Fail(call + ": reduce must be one of nearest, nearest2, min, max, sum, mean;"
                         " not \"" + reduce + "\"" + At(fn));
                    return;
                }
            }
            if (count < 5 && domain_ == Domain::Groom) {
                Fail(call + " at groom rate has no $P to query from; pass the query position"
                     " as the fifth argument" + At(fn));
                return;
            }
            std::string error;
            const int width = ElementExpressionWidth(sampler.source, &error);
            if (!width) {
                Fail(call + " element expression: " + error + At(fn));
                return;
            }
            FrontendOptions options;
            options.domain = Domain::Point;
            options.destination = ScalarType::Float64;
            options.components = uint32_t(width);
            options.samplerElement = true;
            auto compiled = Frontend::Compile(sampler.source, options);
            if (!compiled.ok) {
                for (auto const &message : compiled.diagnostics)
                    Fail(call + " element expression: " + message);
                return;
            }
            sampler.components = static_cast<uint8_t>(width);
            sampler.element = std::make_shared<const IRProgram>(compiled.program.IR());
        }
        for (size_t i = 0; i < ir_.samplers.size(); ++i) {
            IRSampler const &known = ir_.samplers[i];
            if (known.kind == sampler.kind && known.input == sampler.input &&
                known.source == sampler.source && known.iterate == sampler.iterate &&
                known.reduce == sampler.reduce) {
                samplerSlot_[fn] = static_cast<uint16_t>(i);
                return;
            }
        }
        if (ir_.samplers.size() >= 0xFFFF) {
            Fail("too many sampler calls in one expression" + At(fn));
            return;
        }
        samplerSlot_[fn] = static_cast<uint16_t>(ir_.samplers.size());
        ir_.samplers.push_back(std::move(sampler));
    }

    // -- emission ----------------------------------------------------------
    uint16_t EmitRaw(IRInstruction x)
    {
        if (ir_.registerCount >= kExprRegisters || ir_.instructions.size() >= kExprInstructions) {
            if (errors_.empty() || errors_.back().rfind("expression IR limit", 0) != 0)
                errors_.push_back("expression IR limit exceeded: a program may use at most " +
                                  std::to_string(kExprRegisters) + " values");
            return 0;
        }
        x.dst = ir_.registerCount++;
        ir_.instructions.push_back(x);
        return x.dst;
    }
    /// Common subexpression elimination. Every op is pure and every register is
    /// written once, so two identical instructions always hold the same value.
    /// Move is excluded: a Call reads its arguments as a register RANGE, so its
    /// gathering moves must stay where they were emitted.
    uint16_t Emit(IRInstruction x)
    {
        if (x.op == IROp::Move) return EmitRaw(x);
        uint64_t bits = 0;
        std::memcpy(&bits, &x.immediate, sizeof(bits));
        const CseKey key{int(x.op), x.a, x.b, x.c, int(x.variable), bits, x.compare, x.component};
        auto found = cse_.find(key);
        if (found != cse_.end()) return found->second;
        const uint16_t reg = EmitRaw(x);
        if (errors_.empty()) cse_.emplace(key, reg);
        return reg;
    }
    uint16_t Constant(double v) { IRInstruction q; q.op = IROp::Const; q.immediate = v; return Emit(q); }
    uint16_t Unary(IROp op, uint16_t a) { IRInstruction q; q.op = op; q.a = a; return Emit(q); }
    uint16_t Binary(IROp op, uint16_t a, uint16_t b) { IRInstruction q; q.op = op; q.a = a; q.b = b; return Emit(q); }
    uint16_t Ternary(IROp op, uint16_t a, uint16_t b, uint16_t c) { IRInstruction q; q.op = op; q.a = a; q.b = b; q.c = c; return Emit(q); }
    uint16_t Compare(char op, uint16_t a, uint16_t b) { IRInstruction q; q.op = IROp::Compare; q.a = a; q.b = b; q.compare = op; return Emit(q); }
    uint16_t LoadVar(::usdGen::expr::Variable v, uint8_t component)
    {
        IRInstruction q; q.op = IROp::LoadVariable; q.variable = v; q.component = component;
        return Emit(q);
    }

    /// Materialises a Call's argument block in consecutive registers and emits
    /// the Call. Nothing else may be emitted between the first and last slot.
    uint16_t EmitCall(IRFunc func, std::vector<Slot> const &slots, uint8_t component)
    {
        const uint16_t base = ir_.registerCount;
        for (Slot const &slot : slots) {
            IRInstruction q;
            if (slot.isConst) { q.op = IROp::Const; q.immediate = slot.value; }
            else { q.op = IROp::Move; q.a = slot.reg; }
            EmitRaw(q);
        }
        IRInstruction call;
        call.op = IROp::Call;
        call.a = base;
        call.b = static_cast<uint16_t>(slots.size());
        call.c = static_cast<uint16_t>(func);
        call.component = component;
        return EmitRaw(call);
    }

    uint16_t EmitSample(uint16_t slot, std::vector<Slot> const &slots, uint8_t component)
    {
        const uint16_t base = ir_.registerCount;
        for (Slot const &argument : slots) {
            IRInstruction q;
            if (argument.isConst) { q.op = IROp::Const; q.immediate = argument.value; }
            else { q.op = IROp::Move; q.a = argument.reg; }
            EmitRaw(q);
        }
        IRInstruction sample;
        sample.op = IROp::Sample;
        sample.a = base;
        sample.b = static_cast<uint16_t>(slots.size());
        sample.c = slot;
        sample.component = component;
        return EmitRaw(sample);
    }

    uint16_t LowerSampler(SE::ExprFuncNode const *fn, uint8_t component)
    {
        auto const found = samplerSlot_.find(fn);
        if (found == samplerSlot_.end()) {
            Fail(std::string("unresolved ") + fn->name() + "() call" + At(fn));
            return 0;
        }
        IRSampler const &sampler = ir_.samplers[found->second];
        const uint8_t out = component < sampler.components
            ? component : static_cast<uint8_t>(sampler.components - 1);
        std::vector<Slot> slots;
        if (sampler.kind == SamplerKind::Geometry) {
            if (fn->numChildren() == 5) {
                PushVector(slots, fn->child(4));
            } else {
                for (uint8_t c = 0; c < 3; ++c) {
                    Slot slot;
                    slot.reg = LoadVar(::usdGen::expr::Variable::P, c);
                    slots.push_back(slot);
                }
            }
        }
        return EmitSample(found->second, slots, out);
    }

    // -- static shape ------------------------------------------------------
    int Dim(SE::ExprNode const *n)
    {
        if (!n) return 1;
        if (dynamic_cast<SE::ExprNumNode const *>(n)) return 1;
        if (auto x = dynamic_cast<SE::ExprVarNode const *>(n)) {
            const std::string name = Canonical(x->name());
            auto local = symbols_.find(name);
            if (local != symbols_.end()) return local->second.components;
            if (auto info = Registry::Get().Find(name.c_str()))
                return info->components ? int(info->components) : valueDim_;
            return 1;
        }
        if (auto x = dynamic_cast<SE::ExprVecNode const *>(n))
            return std::min(4, std::max(1, x->numChildren()));
        if (dynamic_cast<SE::ExprSubscriptNode const *>(n)) return 1;
        if (dynamic_cast<SE::ExprCompareNode const *>(n)) return 1;
        if (dynamic_cast<SE::ExprCompareEqNode const *>(n)) return 1;
        if (auto x = dynamic_cast<SE::ExprFuncNode const *>(n)) {
            if (IsSamplerCall(x->name())) {
                auto const found = samplerSlot_.find(x);
                return found == samplerSlot_.end() ? 1 : ir_.samplers[found->second].components;
            }
            FunctionRow const *row = FindFunction(x->name());
            if (!row) return 1;
            if (!row->componentWise) return row->components;
            int dim = 1;
            for (int i = 0; i < x->numChildren(); ++i) dim = std::max(dim, Dim(x->child(i)));
            return dim;
        }
        if (auto x = dynamic_cast<SE::ExprUnaryOpNode const *>(n)) return Dim(x->child(0));
        if (dynamic_cast<SE::ExprBinaryOpNode const *>(n))
            return std::max(Dim(n->child(0)), Dim(n->child(1)));
        if (dynamic_cast<SE::ExprCondNode const *>(n))
            return std::max(Dim(n->child(1)), Dim(n->child(2)));
        int dim = 1;
        for (int i = 0; i < n->numChildren(); ++i) dim = std::max(dim, Dim(n->child(i)));
        return dim;
    }

    // -- constant folding, for curve()/ccurve() knots -----------------------
    bool Fold(SE::ExprNode const *n, int component, double *out)
    {
        if (!n) return false;
        if (auto x = dynamic_cast<SE::ExprNumNode const *>(n)) { *out = x->value(); return true; }
        if (auto x = dynamic_cast<SE::ExprVecNode const *>(n)) {
            if (component >= x->numChildren()) return false;
            return Fold(x->child(component), 0, out);
        }
        if (auto x = dynamic_cast<SE::ExprUnaryOpNode const *>(n)) {
            double a = 0;
            if (!Fold(x->child(0), component, &a)) return false;
            if (x->_op == '-') { *out = -a; return true; }
            if (x->_op == '~') { *out = 1 - a; return true; }
            return false;
        }
        if (auto x = dynamic_cast<SE::ExprBinaryOpNode const *>(n)) {
            double a = 0, b = 0;
            if (!Fold(x->child(0), component, &a) || !Fold(x->child(1), component, &b)) return false;
            switch (x->_op) {
            case '+': *out = a + b; return true;
            case '-': *out = a - b; return true;
            case '*': *out = a * b; return true;
            case '/': *out = a / b; return true;
            case '%': *out = std::fmod(a, b); return true;
            case '^': *out = std::pow(a, b); return true;
            default: return false;
            }
        }
        return false;
    }

    // -- statements --------------------------------------------------------
    void LowerStatements(SE::ExprNode const *node)
    {
        if (!node || !errors_.empty()) return;
        if (IsStatement(node)) { LowerStatement(node); return; }
        for (int i = 0; i < node->numChildren() && errors_.empty(); ++i)
            LowerStatement(node->child(i));
    }
    void LowerStatement(SE::ExprNode const *node)
    {
        if (!node || !errors_.empty()) return;
        if (auto assign = dynamic_cast<SE::ExprAssignNode const *>(node)) {
            const std::string name = Canonical(assign->name());
            const int dim = std::min(4, std::max(1, Dim(assign->child(0))));
            Local local;
            local.components = static_cast<uint8_t>(dim);
            for (int c = 0; c < dim; ++c) local.reg[c] = Lower(assign->child(0), uint8_t(c));
            symbols_[name] = local;
            return;
        }
        if (auto branch = dynamic_cast<SE::ExprIfThenElseNode const *>(node)) {
            LowerIf(branch);
            return;
        }
        if (!IsStatement(node)) LowerStatements(node);
    }
    void LowerIf(SE::ExprIfThenElseNode const *node)
    {
        const uint16_t condition = Lower(node->child(0), 0);
        const Symbols before = symbols_;
        symbols_ = before;
        LowerStatements(node->child(1));
        const Symbols thenScope = symbols_;
        symbols_ = before;
        LowerStatements(node->child(2));
        const Symbols elseScope = symbols_;
        symbols_ = before;
        if (!errors_.empty()) return;
        std::set<std::string> touched;
        for (auto const &entry : thenScope) touched.insert(entry.first);
        for (auto const &entry : elseScope) touched.insert(entry.first);
        for (std::string const &name : touched) {
            auto t = thenScope.find(name);
            auto e = elseScope.find(name);
            auto prior = before.find(name);
            // A name assigned in only one arm keeps its value from before the
            // if. A name that did not exist before and is assigned in only one
            // arm is not defined afterwards, which is what SeExpr's own type
            // checker reports too.
            if (t == thenScope.end() && prior == before.end()) continue;
            if (e == elseScope.end() && prior == before.end()) continue;
            Local const &thenValue = t != thenScope.end() ? t->second : prior->second;
            Local const &elseValue = e != elseScope.end() ? e->second : prior->second;
            Local merged;
            merged.components = std::max(thenValue.components, elseValue.components);
            for (int c = 0; c < merged.components; ++c) {
                const uint16_t a = thenValue.reg[c < thenValue.components ? c : 0];
                const uint16_t b = elseValue.reg[c < elseValue.components ? c : 0];
                merged.reg[c] = a == b ? a : Ternary(IROp::Select, condition, a, b);
            }
            symbols_[name] = merged;
        }
        ir_.hasLazyBranches = true;
    }

    // -- expressions -------------------------------------------------------
    uint16_t Lower(SE::ExprNode const *n, uint8_t component)
    {
        if (!n || !errors_.empty()) return 0;
        if (auto x = dynamic_cast<SE::ExprNumNode const *>(n)) return Constant(x->value());
        if (auto x = dynamic_cast<SE::ExprVarNode const *>(n)) {
            const std::string name = Canonical(x->name());
            auto local = symbols_.find(name);
            if (local != symbols_.end()) {
                Local const &value = local->second;
                return value.reg[component < value.components ? component : 0];
            }
            auto info = Registry::Get().Find(name.c_str());
            if (!info) { Fail("unknown expression variable " + name + At(n)); return 0; }
            std::string diagnostic;
            if (!Registry::Get().Validate(name.c_str(), domain_, &diagnostic, samplerElement_)) {
                Fail(diagnostic + At(n));
                return 0;
            }
            return LoadVar(info->id, info->components == 1 ? 0 : component);
        }
        if (dynamic_cast<SE::ExprModuleNode const *>(n) ||
            dynamic_cast<SE::ExprBlockNode const *>(n)) {
            // A nested block only appears inside a construct we already reject,
            // but keep the single-child passthrough rather than crashing.
            if (n->numChildren() == 1) return Lower(n->child(0), component);
            if (dynamic_cast<SE::ExprBlockNode const *>(n) && n->numChildren() == 2) {
                LowerStatements(n->child(0));
                return Lower(n->child(1), component);
            }
            Fail("malformed expression block" + At(n));
            return 0;
        }
        if (auto x = dynamic_cast<SE::ExprVecNode const *>(n)) {
            if (component >= x->numChildren()) {
                Fail("vector component out of range" + At(n));
                return 0;
            }
            return Lower(x->child(component), 0);
        }
        if (auto x = dynamic_cast<SE::ExprSubscriptNode const *>(n)) {
            auto idx = dynamic_cast<SE::ExprNumNode const *>(x->child(1));
            if (!idx || idx->value() < 0 || idx->value() > 3 ||
                idx->value() != int(idx->value())) {
                Fail("only constant vector indexing is supported" + At(n));
                return 0;
            }
            return Lower(x->child(0), static_cast<uint8_t>(idx->value()));
        }
        if (auto x = dynamic_cast<SE::ExprFuncNode const *>(n)) return LowerFunction(x, component);
        if (auto x = dynamic_cast<SE::ExprUnaryOpNode const *>(n)) {
            const uint16_t a = Lower(x->child(0), component);
            if (x->_op == '+') return a;
            if (x->_op == '-') return Unary(IROp::Neg, a);
            if (x->_op == '!') return Compare('=', a, Constant(0.0));
            if (x->_op == '~') return Binary(IROp::Sub, Constant(1.0), a);
            Fail(std::string("unsupported unary operator '") + x->_op + "'" + At(n));
            return 0;
        }
        if (auto x = dynamic_cast<SE::ExprBinaryOpNode const *>(n)) {
            const uint16_t a = Lower(x->child(0), component);
            const uint16_t b = Lower(x->child(1), component);
            switch (x->_op) {
            case '+': return Binary(IROp::Add, a, b);
            case '-': return Binary(IROp::Sub, a, b);
            case '*': return Binary(IROp::Mul, a, b);
            case '/': return Binary(IROp::Div, a, b);
            case '%': return Binary(IROp::Fmod, a, b);
            case '^': return Binary(IROp::Pow, a, b);
            default:
                Fail(std::string("unsupported binary operator '") + x->_op + "'" + At(n));
                return 0;
            }
        }
        if (auto x = dynamic_cast<SE::ExprCompareNode const *>(n)) {
            // '&' and '|' are the parser's spellings of && and ||. Both sides
            // reduce to 0/1 first so the result is a boolean, and a poisoned
            // operand still poisons the result.
            if (x->_op == '&' || x->_op == '|') {
                const uint16_t a = Compare('!', Lower(x->child(0), 0), Constant(0.0));
                const uint16_t b = Compare('!', Lower(x->child(1), 0), Constant(0.0));
                return Binary(x->_op == '&' ? IROp::Min : IROp::Max, a, b);
            }
            return Compare(x->_op, Lower(x->child(0), component), Lower(x->child(1), component));
        }
        if (auto x = dynamic_cast<SE::ExprCompareEqNode const *>(n))
            return Compare(x->_op, Lower(x->child(0), component), Lower(x->child(1), component));
        if (dynamic_cast<SE::ExprCondNode const *>(n)) {
            const uint16_t c = Lower(n->child(0), 0);
            const uint16_t a = Lower(n->child(1), component);
            const uint16_t b = Lower(n->child(2), component);
            ir_.hasLazyBranches = true;
            return Ternary(IROp::Select, c, a, b);
        }
        if (IsStatement(n)) { LowerStatement(n); return 0; }
        Fail("unsupported expression construct" + At(n));
        return 0;
    }

    // -- argument gathering -------------------------------------------------
    void PushScalar(std::vector<Slot> &slots, SE::ExprNode const *node)
    {
        Slot slot; slot.reg = Lower(node, 0); slots.push_back(slot);
    }
    void PushVector(std::vector<Slot> &slots, SE::ExprNode const *node)
    {
        const int dim = Dim(node);
        for (uint8_t c = 0; c < 3; ++c) {
            Slot slot;
            slot.reg = Lower(node, dim == 1 ? 0 : c);
            slots.push_back(slot);
        }
    }
    void PushConst(std::vector<Slot> &slots, double value)
    {
        Slot slot; slot.isConst = true; slot.value = value; slots.push_back(slot);
    }
    /// Optional trailing scalar argument with a SeExpr default.
    void PushOptional(std::vector<Slot> &slots, SE::ExprFuncNode const *fn, int index, double fallback)
    {
        if (index < fn->numChildren()) PushScalar(slots, fn->child(index));
        else PushConst(slots, fallback);
    }

    uint16_t LowerFunction(SE::ExprFuncNode const *fn, uint8_t component)
    {
        const std::string name = fn->name();
        FunctionRow const *row = FindFunction(name.c_str());
        if (!row) { Fail("unsupported expression function '" + name + "'" + At(fn)); return 0; }
        const int count = fn->numChildren();
        auto arg = [&](int i) { return Lower(fn->child(i), component); };
        // A builtin with a scalar result is always asked for component 0, even
        // when the destination is a vector: only its arguments vary per lane.
        const uint8_t out = row->components == 1
            ? uint8_t(0)
            : uint8_t(component < row->components ? component : row->components - 1);
        if (IsSamplerCall(name.c_str())) return LowerSampler(fn, component);

        // --- direct opcodes -------------------------------------------------
        if (name == "abs")   return Unary(IROp::Abs,   arg(0));
        if (name == "sin")   return Unary(IROp::Sin,   arg(0));
        if (name == "cos")   return Unary(IROp::Cos,   arg(0));
        if (name == "tan")   return Unary(IROp::Tan,   arg(0));
        if (name == "asin")  return Unary(IROp::Asin,  arg(0));
        if (name == "acos")  return Unary(IROp::Acos,  arg(0));
        if (name == "atan")  return Unary(IROp::Atan,  arg(0));
        if (name == "sinh")  return Unary(IROp::Sinh,  arg(0));
        if (name == "cosh")  return Unary(IROp::Cosh,  arg(0));
        if (name == "tanh")  return Unary(IROp::Tanh,  arg(0));
        if (name == "sqrt")  return Unary(IROp::Sqrt,  arg(0));
        if (name == "cbrt")  return Unary(IROp::Cbrt,  arg(0));
        if (name == "exp")   return Unary(IROp::Exp,   arg(0));
        if (name == "log")   return Unary(IROp::Log,   arg(0));
        if (name == "log10") return Unary(IROp::Log10, arg(0));
        if (name == "floor") return Unary(IROp::Floor, arg(0));
        if (name == "ceil")  return Unary(IROp::Ceil,  arg(0));
        if (name == "round") return Unary(IROp::Round, arg(0));
        if (name == "trunc") return Unary(IROp::Trunc, arg(0));
        if (name == "pow")   { auto a = arg(0); auto b = arg(1); return Binary(IROp::Pow, a, b); }
        if (name == "min")   { auto a = arg(0); auto b = arg(1); return Binary(IROp::Min, a, b); }
        if (name == "max")   { auto a = arg(0); auto b = arg(1); return Binary(IROp::Max, a, b); }
        if (name == "fmod")  { auto a = arg(0); auto b = arg(1); return Binary(IROp::Fmod, a, b); }
        if (name == "atan2") { auto a = arg(0); auto b = arg(1); return Binary(IROp::Atan2, a, b); }
        if (name == "hypot") { auto a = arg(0); auto b = arg(1); return Binary(IROp::Hypot, a, b); }
        if (name == "clamp") { auto a = arg(0); auto b = arg(1); auto c = arg(2); return Ternary(IROp::Clamp, a, b, c); }

        // --- algebraic composites, bit-identical on both lanes for free ------
        if (name == "deg")    return Binary(IROp::Mul, arg(0), Constant(180 / kExprPi));
        if (name == "rad")    return Binary(IROp::Mul, arg(0), Constant(kExprPi / 180));
        if (name == "cosd")   return Unary(IROp::Cos, Binary(IROp::Mul, arg(0), Constant(kExprPi / 180)));
        if (name == "sind")   return Unary(IROp::Sin, Binary(IROp::Mul, arg(0), Constant(kExprPi / 180)));
        if (name == "tand")   return Unary(IROp::Tan, Binary(IROp::Mul, arg(0), Constant(kExprPi / 180)));
        if (name == "acosd")  return Binary(IROp::Mul, Unary(IROp::Acos, arg(0)), Constant(180 / kExprPi));
        if (name == "asind")  return Binary(IROp::Mul, Unary(IROp::Asin, arg(0)), Constant(180 / kExprPi));
        if (name == "atand")  return Binary(IROp::Mul, Unary(IROp::Atan, arg(0)), Constant(180 / kExprPi));
        if (name == "atan2d") { auto a = arg(0); auto b = arg(1); return Binary(IROp::Mul, Binary(IROp::Atan2, a, b), Constant(180 / kExprPi)); }
        if (name == "invert") return Binary(IROp::Sub, Constant(1.0), arg(0));
        if (name == "gamma")  { auto a = arg(0); auto g = arg(1); return Binary(IROp::Pow, a, Binary(IROp::Div, Constant(1.0), g)); }
        if (name == "compress") {   // (hi - lo) * x + lo
            auto x = arg(0); auto lo = arg(1); auto hi = arg(2);
            auto span = Binary(IROp::Sub, hi, lo);
            return Binary(IROp::Add, Binary(IROp::Mul, span, x), lo);
        }
        if (name == "mix") {        // x * (1 - alpha) + y * alpha
            auto a = arg(0); auto b = arg(1); auto alpha = arg(2);
            auto inverse = Binary(IROp::Sub, Constant(1.0), alpha);
            auto left = Binary(IROp::Mul, a, inverse);
            auto right = Binary(IROp::Mul, b, alpha);
            return Binary(IROp::Add, left, right);
        }
        if (name == "fit") {        // (x*(b2-a2) - a1*b2 + b1*a2) / (b1-a1)
            auto x = arg(0); auto a1 = arg(1); auto b1 = arg(2); auto a2 = arg(3); auto b2 = arg(4);
            auto scaled = Binary(IROp::Mul, x, Binary(IROp::Sub, b2, a2));
            auto shifted = Binary(IROp::Sub, scaled, Binary(IROp::Mul, a1, b2));
            auto numerator = Binary(IROp::Add, shifted, Binary(IROp::Mul, b1, a2));
            return Binary(IROp::Div, numerator, Binary(IROp::Sub, b1, a1));
        }

        // --- scalar builtins that need SeExpr's branchy definition -----------
        auto simpleCall = [&](IRFunc func, int arity) {
            std::vector<Slot> slots;
            for (int i = 0; i < arity; ++i) PushScalar(slots, fn->child(i));
            return EmitCall(func, slots, 0);
        };
        if (name == "expand")     return simpleCall(IRFunc::Expand, 3);
        if (name == "bias")       return simpleCall(IRFunc::Bias, 2);
        if (name == "contrast")   return simpleCall(IRFunc::Contrast, 2);
        if (name == "boxstep")    return simpleCall(IRFunc::BoxStep, 2);
        if (name == "linearstep") return simpleCall(IRFunc::LinearStep, 3);
        if (name == "smoothstep") return simpleCall(IRFunc::SmoothStep, 3);
        if (name == "gaussstep")  return simpleCall(IRFunc::GaussStep, 3);
        if (name == "remap")      return simpleCall(IRFunc::Remap, 5);
        if (name == "cycle")      return simpleCall(IRFunc::Cycle, 3);

        // --- noise -----------------------------------------------------------
        if (name == "hash") {
            std::vector<Slot> slots;
            for (int i = 0; i < count; ++i) PushScalar(slots, fn->child(i));
            return EmitCall(IRFunc::Hash, slots, 0);
        }
        if (name == "rand") return LowerRand(fn);
        auto noiseVector = [&](IRFunc func) {
            std::vector<Slot> slots;
            if (count == 1) PushVector(slots, fn->child(0));
            else for (int i = 0; i < 3; ++i) PushScalar(slots, fn->child(i));
            return EmitCall(func, slots, out);
        };
        if (name == "noise")      return noiseVector(IRFunc::Noise);
        if (name == "snoise")     return noiseVector(IRFunc::SNoise);
        if (name == "vnoise")     return noiseVector(IRFunc::VNoise);
        if (name == "cnoise")     return noiseVector(IRFunc::CNoise);
        if (name == "cellnoise")  return noiseVector(IRFunc::CellNoise);
        if (name == "ccellnoise") return noiseVector(IRFunc::CCellNoise);
        if (name == "pnoise") {
            std::vector<Slot> slots;
            PushVector(slots, fn->child(0));
            PushVector(slots, fn->child(1));
            return EmitCall(IRFunc::PNoise, slots, 0);
        }
        auto fbmFamily = [&](IRFunc func) {
            std::vector<Slot> slots;
            PushVector(slots, fn->child(0));
            PushOptional(slots, fn, 1, 6);     // octaves
            PushOptional(slots, fn, 2, 2);     // lacunarity
            PushOptional(slots, fn, 3, 0.5);   // gain
            return EmitCall(func, slots, out);
        };
        if (name == "fbm")          return fbmFamily(IRFunc::Fbm);
        if (name == "vfbm")         return fbmFamily(IRFunc::VFbm);
        if (name == "cfbm")         return fbmFamily(IRFunc::CFbm);
        if (name == "turbulence")   return fbmFamily(IRFunc::Turbulence);
        if (name == "vturbulence")  return fbmFamily(IRFunc::VTurbulence);
        if (name == "cturbulence")  return fbmFamily(IRFunc::CTurbulence);
        if (name == "voronoi" || name == "cvoronoi") {
            std::vector<Slot> slots;
            PushVector(slots, fn->child(0));
            PushOptional(slots, fn, 1, 1);     // type
            PushOptional(slots, fn, 2, 0.5);   // jitter
            PushOptional(slots, fn, 3, 0);     // fbmScale
            PushOptional(slots, fn, 4, 4);     // fbmOctaves
            PushOptional(slots, fn, 5, 2);     // fbmLacunarity
            PushOptional(slots, fn, 6, 0.5);   // fbmGain
            return EmitCall(name == "voronoi" ? IRFunc::Voronoi : IRFunc::CVoronoi, slots, out);
        }
        if (name == "pvoronoi") {
            std::vector<Slot> slots;
            PushVector(slots, fn->child(0));
            PushOptional(slots, fn, 1, 0.5);
            PushOptional(slots, fn, 2, 0);
            PushOptional(slots, fn, 3, 4);
            PushOptional(slots, fn, 4, 2);
            PushOptional(slots, fn, 5, 0.5);
            return EmitCall(IRFunc::PVoronoi, slots, out);
        }

        // --- vector -----------------------------------------------------------
        auto vectorCall = [&](IRFunc func, int vectors, int trailingScalars) {
            std::vector<Slot> slots;
            for (int i = 0; i < vectors; ++i) PushVector(slots, fn->child(i));
            for (int i = 0; i < trailingScalars; ++i) PushScalar(slots, fn->child(vectors + i));
            return EmitCall(func, slots, out);
        };
        if (name == "dist")   return vectorCall(IRFunc::Dist, 2, 0);
        if (name == "length") return vectorCall(IRFunc::Length, 1, 0);
        if (name == "dot")    return vectorCall(IRFunc::Dot, 2, 0);
        if (name == "cross")  return vectorCall(IRFunc::Cross, 2, 0);
        if (name == "norm")   return vectorCall(IRFunc::Norm, 1, 0);
        if (name == "angle")  return vectorCall(IRFunc::Angle, 2, 0);
        if (name == "ortho")  return vectorCall(IRFunc::Ortho, 2, 0);
        if (name == "rotate") return vectorCall(IRFunc::Rotate, 2, 1);
        if (name == "up")     return vectorCall(IRFunc::Up, 2, 0);

        // --- colour -----------------------------------------------------------
        if (name == "rgbtohsl") return vectorCall(IRFunc::RgbToHsl, 1, 0);
        if (name == "hsltorgb") return vectorCall(IRFunc::HslToRgb, 1, 0);
        if (name == "saturate") return vectorCall(IRFunc::Saturate, 1, 1);
        if (name == "hsi" || name == "midhsi") {
            std::vector<Slot> slots;
            PushVector(slots, fn->child(0));
            PushScalar(slots, fn->child(1));
            PushScalar(slots, fn->child(2));
            PushScalar(slots, fn->child(3));
            PushOptional(slots, fn, 4, 1);     // map
            if (name == "midhsi") {
                PushOptional(slots, fn, 5, 1); // falloff
                PushOptional(slots, fn, 6, 0); // interp
            }
            return EmitCall(name == "hsi" ? IRFunc::Hsi : IRFunc::MidHsi, slots, out);
        }

        // --- variations -------------------------------------------------------
        if (name == "spline" || name == "choose" || name == "wchoose" || name == "pick") {
            std::vector<Slot> slots;
            for (int i = 0; i < count; ++i) PushScalar(slots, fn->child(i));
            const IRFunc func = name == "spline" ? IRFunc::Spline
                              : name == "choose" ? IRFunc::Choose
                              : name == "wchoose" ? IRFunc::WChoose : IRFunc::Pick;
            return EmitCall(func, slots, 0);
        }
        if (name == "curve" || name == "ccurve") return LowerCurve(fn, name == "ccurve", out);

        Fail("unsupported expression function '" + name + "'" + At(fn));
        return 0;
    }

    /// rand() is a host groomer's, not SeExpr's: deterministic per strand rather than per
    /// evaluation, so a groom looks the same every cook. The seed is the
    /// operator's $seed, the strand's $id and the call site's index, which is
    /// what makes two rand() calls in one expression independent.
    uint16_t LowerRand(SE::ExprFuncNode const *fn)
    {
        const int count = fn->numChildren();
        std::vector<Slot> slots;
        Slot seed; seed.reg = LoadVar(::usdGen::expr::Variable::Seed, 0); slots.push_back(seed);
        if (domain_ == Domain::Groom) {
            // $id is a per-strand field and does not exist at groom rate.
            PushConst(slots, 0);
        } else {
            Slot id; id.reg = LoadVar(::usdGen::expr::Variable::Id, 0); slots.push_back(id);
        }
        auto site = randIndex_.find(fn);
        PushConst(slots, site != randIndex_.end() ? double(site->second) : 0.0);
        if (count == 1) PushScalar(slots, fn->child(0));
        if (count == 3) PushScalar(slots, fn->child(2));
        const uint16_t hash = EmitCall(IRFunc::Hash, slots, 0);
        if (count != 3) return hash;
        // rand(min, max, seed) = min + (max - min) * hash
        const uint16_t lo = Lower(fn->child(0), 0);
        const uint16_t hi = Lower(fn->child(1), 0);
        return Binary(IROp::Add, lo, Binary(IROp::Mul, Binary(IROp::Sub, hi, lo), hash));
    }

    /// One prepared control-curve knot, mirroring SeExpr2's Curve<T>::CV.
    struct CV {
        double pos = 0;
        double value[3]{};
        double deriv[3]{};
        int interp = 0;
    };

    uint16_t LowerCurve(SE::ExprFuncNode const *fn, bool colour, uint8_t component)
    {
        const int channels = colour ? 3 : 1;
        const int knots = (fn->numChildren() - 1) / 3;
        std::vector<CV> cvs;
        const double sentinel = double(FLT_MAX);
        cvs.push_back(CV{-sentinel, {}, {}, 0});
        cvs.push_back(CV{sentinel, {}, {}, 0});
        for (int k = 0; k < knots; ++k) {
            CV cv;
            SE::ExprNode const *positionNode = fn->child(1 + k * 3);
            SE::ExprNode const *valueNode = fn->child(2 + k * 3);
            SE::ExprNode const *interpNode = fn->child(3 + k * 3);
            double interp = 0;
            if (!Fold(positionNode, 0, &cv.pos) || !Fold(interpNode, 0, &interp)) {
                Fail(std::string(fn->name()) +
                     "() positions and interpolation codes must be constants" + At(fn));
                return 0;
            }
            for (int c = 0; c < channels; ++c) {
                if (!Fold(valueNode, colour ? c : 0, &cv.value[c])) {
                    Fail(std::string(fn->name()) + "() values must be constants" + At(fn));
                    return 0;
                }
            }
            cv.interp = int(interp);
            if (cv.interp < 0 || cv.interp > 4) {
                Fail(std::string(fn->name()) +
                     "() interpolation must be 0 none, 1 linear, 2 smooth, 3 spline or"
                     " 4 monotone spline" + At(fn));
                return 0;
            }
            cvs.push_back(cv);
        }
        // Curve<T>::preparePoints, verbatim. stable_sort rather than sort so
        // two knots at the same position keep authoring order on every
        // toolchain; the frontend runs on the host for both lanes, so the
        // prepared table itself is what both lanes see.
        std::stable_sort(cvs.begin(), cvs.end(),
                         [](CV const &a, CV const &b) { return a.pos < b.pos; });
        const int n = int(cvs.size());
        if (n > 2) {
            for (int c = 0; c < channels; ++c) {
                cvs.front().value[c] = cvs[1].value[c];
                cvs.back().value[c] = cvs[n - 2].value[c];
                cvs.front().deriv[c] = 0;
                cvs.back().deriv[c] = 0;
            }
            cvs.front().interp = 0;
            cvs.back().interp = 0;
        } else {
            cvs.front().pos = 0;
            cvs.back().pos = 0;
        }
        for (int i = 1; i < n - 1; ++i)
            for (int c = 0; c < channels; ++c)
                cvs[i].deriv[c] = (cvs[i + 1].value[c] - cvs[i - 1].value[c]) /
                                  (cvs[i + 1].pos - cvs[i - 1].pos);
        for (int i = 0; i < n - 1; ++i) {
            if (cvs[i].interp != 4) continue;
            const double h = cvs[i + 1].pos - cvs[i].pos;
            for (int c = 0; c < channels; ++c) {
                if (h == 0) { cvs[i].deriv[c] = cvs[i + 1].deriv[c] = 0; continue; }
                const double delta = (cvs[i + 1].value[c] - cvs[i].value[c]) / h;
                if (delta == 0) { cvs[i].deriv[c] = cvs[i + 1].deriv[c] = 0; continue; }
                cvs[i].deriv[c] = ExprClamp(cvs[i].deriv[c] / delta, 0, 3) * delta;
                cvs[i + 1].deriv[c] = ExprClamp(cvs[i + 1].deriv[c] / delta, 0, 3) * delta;
            }
        }
        std::vector<Slot> slots;
        PushScalar(slots, fn->child(0));
        PushConst(slots, double(n));
        for (CV const &cv : cvs) {
            PushConst(slots, cv.pos);
            PushConst(slots, double(cv.interp));
            for (int c = 0; c < channels; ++c) PushConst(slots, cv.value[c]);
            for (int c = 0; c < channels; ++c) PushConst(slots, cv.deriv[c]);
        }
        return EmitCall(colour ? IRFunc::CCurve : IRFunc::Curve, slots, colour ? component : 0);
    }
};

std::string const kNoSource;
IRProgram const kNoProgram{};

} // namespace

struct Program::Impl { std::string source; bool valid=false; IRProgram ir; };
Program::Program() : _impl(new Impl) {}
Program::~Program() = default;
Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;
Program::Program(std::unique_ptr<Impl> p) : _impl(std::move(p)) {}
bool Program::Valid() const noexcept { return _impl && _impl->valid; }
std::string const &Program::Source() const noexcept { return _impl ? _impl->source : kNoSource; }
IRProgram const &Program::IR() const noexcept { return _impl ? _impl->ir : kNoProgram; }

CompileResult Frontend::Compile(std::string const &source, FrontendOptions const &opt)
{
    CompileResult out; out.program = Program(std::unique_ptr<Program::Impl>(new Program::Impl));
    out.program._impl->source = NormalizeLineEndings(source);
    if (opt.domain != Domain::Groom && opt.domain != Domain::Primitive && opt.domain != Domain::Point) { out.diagnostics.push_back("invalid evaluation domain"); return out; }
    switch (opt.destination) {
    case ScalarType::Bool: case ScalarType::Int32: case ScalarType::UInt32:
    case ScalarType::Int64: case ScalarType::UInt64: case ScalarType::Float16:
    case ScalarType::Float32: case ScalarType::Float64: break;
    default: out.diagnostics.push_back("invalid expression destination type"); return out;
    }
    if (opt.components == 0 || opt.components > 4) { out.diagnostics.push_back("unsupported vector dimension"); return out; }
    CheckedExpression expr(source, int(opt.components), opt.domain, opt.samplerElement);
    const auto *tree = expr.parseTree();
    if (!tree) { out.diagnostics.push_back(expr.parseError().empty() ? "expression parse failed" : expr.parseError()); return out; }
    // Parse first, then walk the actual AST before prep/binding. This catches
    // disallowed builtins even in branches and ignores comment/string text.
    if (!Walk(tree, out.diagnostics, opt.samplerElement)) return out;
    if (!expr.isValid()) { out.diagnostics.push_back(expr.parseError()); for (auto const &e:expr.getErrors()) out.diagnostics.push_back(e.error); return out; }
    IRProgram &ir = out.program._impl->ir;
    ir.result = 0;
    ir.valueComponents = static_cast<uint8_t>(opt.components);
    ir.outputCount = opt.components > 1 ? static_cast<uint8_t>(opt.components) : 0;
    Lowerer lowerer(ir, out.diagnostics, opt.domain, int(opt.components), opt.samplerElement);
    lowerer.NumberRandCalls(tree);
    lowerer.CollectSamplers(tree);
    if (!out.diagnostics.empty()) return out;
    uint16_t outputs[4]{};
    lowerer.LowerProgram(tree, static_cast<uint8_t>(opt.components), outputs);
    for (uint8_t c = 0; c < opt.components; ++c) ir.output[c] = outputs[c];
    ir.result = ir.output[0];
    ir.registerCount = static_cast<uint16_t>(ir.instructions.size());
    if (!out.diagnostics.empty()) return out;
    if (!ValidProgram(ir)) {
        out.diagnostics.push_back("expression lowered to an invalid program");
        return out;
    }
    out.program._impl->valid = true; out.ok = true; return out;
}

std::vector<FunctionInfo> Frontend::SupportedFunctions()
{
    std::vector<FunctionInfo> result;
    result.reserve(kFunctionCount);
    for (auto const &row : kFunctions) {
        FunctionInfo info;
        info.name = row.name;
        info.minArity = row.minArgs;
        info.maxArity = row.maxArgs;
        info.variadic = row.maxArgs == kVariadic || row.maxArgs != row.minArgs;
        info.arity = row.minArgs;
        info.signature = row.signature;
        info.doc = row.doc;
        info.category = row.category;
        info.components = row.components;
        result.push_back(std::move(info));
    }
    // The table is written grouped by category so it reads as documentation;
    // callers are promised name order, which is what a browser and a
    // completion list want.
    std::sort(result.begin(), result.end(),
              [](FunctionInfo const &a, FunctionInfo const &b) { return a.name < b.name; });
    return result;
}

std::vector<VariableDoc> Frontend::VariableDocs()
{
    std::vector<VariableDoc> result;
    const Registry &registry = Registry::Get();
    result.reserve(registry.Count());
    for (size_t i = 0; i < registry.Count(); ++i) {
        const VariableInfo *info = registry.At(i);
        if (!info) break;
        VariableDoc entry;
        entry.name = info->name;
        entry.type = info->scalar == ScalarType::Invalid ? "" : ScalarTypeName(info->scalar);
        entry.components = info->components;
        const Domain bits[] = {Domain::Groom, Domain::Primitive, Domain::Point};
        if (Registry::IsSamplerVariable(info->id)) entry.domains = "sampler";
        for (Domain bit : bits) {
            if (!HasDomain(info->domains, bit)) continue;
            if (!entry.domains.empty()) entry.domains.push_back(',');
            entry.domains += DomainName(bit);
        }
        entry.doc = info->doc ? info->doc : "";
        result.push_back(std::move(entry));
    }
    return result;
}
}
