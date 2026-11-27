// geoSampler() and ptex(): the expression functions that read external data
// through the expression prim's input:<name> relationships.
//
//   * the frontend admits them only with string-literal names, lowers each
//     call site to IROp::Sample and compiles the element expression once;
//   * GeometrySampler iterates points / prims / whole gprims and folds the
//     element expression with each reduction;
//   * UsdGenCpuParameters resolves the inputs against a UsdGenGraphDesc and
//     re-resolves when the sampled geometry changes;
//   * ptex() reads a UsdGenPtexMap at the strand root, or a UsdGenPaintMap (bilinear at the root's face-local uv).
#include "utils/pxrInterop.h"
#include "usdGenExpr/cpuParameters.h"
#include "usdGenUsd/lower.h"
#include "usdGenExpr/expressions/cpuEvaluator.h"
#include "usdGenExpr/expressions/frontend.h"
#include "usdGenExpr/expressions/irExec.h"
#include "usdGenExpr/expressions/samplers.h"
#include "usdGenCurves/types/curveBuffer.h"

#include <Ptexture.h>

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/tf/token.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;
using namespace usdGen::expr;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
}
bool Near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) <= eps; }

CompileResult Compile(char const *source, Domain domain = Domain::Primitive,
                      unsigned components = 1)
{
    return Frontend::Compile(source, {domain, ScalarType::Float32, components});
}

bool Mentions(CompileResult const &result, char const *text)
{
    for (auto const &message : result.diagnostics)
        if (message.find(text) != std::string::npos) return true;
    return false;
}

void Refuses(char const *source, char const *reason, Domain domain = Domain::Primitive)
{
    auto result = Compile(source, domain);
    Check(!result.ok && Mentions(result, reason),
          std::string("refused with '") + reason + "': " + source +
              (result.diagnostics.empty() ? "" : "  [" + result.diagnostics[0] + "]"));
}

void CheckFrontend()
{
    auto simple = Compile("geoSampler(\"guides\", \"$index\")");
    Check(simple.ok, "geoSampler(input, expression) compiles at primitive rate");
    if (simple.ok) {
        auto const &ir = simple.program.IR();
        Check(ir.samplers.size() == 1, "one sampler slot");
        Check(ir.samplers[0].kind == SamplerKind::Geometry && ir.samplers[0].input == "guides" &&
              ir.samplers[0].iterate == SampleIterate::Prim &&
              ir.samplers[0].reduce == SampleReduce::Nearest &&
              ir.samplers[0].components == 1 && ir.samplers[0].element,
              "the slot records input, iterate, reduce and the element program");
    }
    auto full = Compile("geoSampler('guides', '$P', 'point', 'nearest2', $rootP)", Domain::Point, 3);
    Check(full.ok, "the full form (single-quoted strings, explicit query) compiles");
    if (full.ok) {
        Check(full.program.IR().samplers[0].components == 3, "a vector element expression is 3 wide");
        Check(full.program.IR().samplers[0].reduce == SampleReduce::Nearest2, "nearest2 parses");
    }
    auto shared = Compile("geoSampler(\"g\", \"$index\") + 2 * geoSampler(\"g\", \"$index\")");
    Check(shared.ok && shared.program.IR().samplers.size() == 1, "identical call sites share a slot");
    auto border = Compile("geoSampler(\"g\", \"$Qdist\", \"prim\", \"nearest2\") -"
                          " geoSampler(\"g\", \"$Qdist\")");
    Check(border.ok && border.program.IR().samplers.size() == 2, "a voronoi border (f2 - f1) compiles");
    auto groom = Compile("geoSampler(\"g\", \"$count\", \"geometry\", \"sum\", [0, 0, 0])",
                         Domain::Groom);
    Check(groom.ok, "an explicit query admits groom-rate sampling");
    Check(Compile("ptex(\"clumpMap\") * $value").ok, "ptex(input) compiles at primitive rate");
    Check(Compile("ptex(\"clumpMap\")", Domain::Point).ok, "ptex(input) compiles at point rate");

    Refuses("geoSampler(1, \"$index\")", "must be a string literal");
    Refuses("geoSampler(\"g\", $index)", "must be a string literal");
    Refuses("geoSampler(\"g\", \"$index\", \"edges\")", "iterate must be");
    Refuses("geoSampler(\"g\", \"$index\", \"prim\", \"median\")", "reduce must be");
    Refuses("geoSampler(\"g\", \"$index\")", "fifth argument", Domain::Groom);
    Refuses("geoSampler(\"g\", \"geoSampler('h', '1')\")", "cannot be used inside");
    Refuses("geoSampler(\"g\", \"$index +\")", "element expression");
    Refuses("geoSampler(\"g\", \"[1, 2]\")", "element expression");
    Refuses("geoSampler(\"\", \"$index\")", "input:<name>");
    Refuses("geoSampler(\"g\")", "takes 2 to 5 arguments");
    Refuses("$Q[0]", "only available inside a geoSampler()");
    Refuses("ptex(\"m\")", "strand root", Domain::Groom);
    Refuses("ptex(\"m\", 1)", "takes 1 argument");
}

// Three curves along +y rooted at x = 0, 1 and 3, of lengths 1, 2 and 3.
SamplerGeometrySource Curves()
{
    SamplerGeometrySource s;
    s.kind = SamplerGeometrySource::Kind::Curves;
    s.points = {0, 0, 0, 0, 0.5f, 0, 0, 1, 0,
                1, 0, 0, 1, 1, 0, 1, 2, 0,
                3, 0, 0, 3, 1.5f, 0, 3, 3, 0};
    s.counts = {3, 3, 3};
    s.ids = {70, 71, 72};
    return s;
}

// One unit quad in the XZ plane, wound so its normal is +Y.
SamplerGeometrySource Quad(float x)
{
    SamplerGeometrySource s;
    s.kind = SamplerGeometrySource::Kind::Mesh;
    s.points = {x, 0, 0, x, 0, 1, x + 1, 0, 1, x + 1, 0, 0};
    s.counts = {4};
    s.indices = {0, 1, 2, 3};
    return s;
}

// Two unit quads side by side in the XZ plane (x in [0, 2]), normal +Y;
// face 1 is points 3, 2, 4, 5.
SamplerGeometrySource Strip()
{
    SamplerGeometrySource s;
    s.kind = SamplerGeometrySource::Kind::Mesh;
    s.points = {0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 2, 0, 1, 2, 0, 0};
    s.counts = {4, 4};
    s.indices = {0, 1, 2, 3, 3, 2, 4, 5};
    return s;
}

IRSampler Spec(char const *element, SampleIterate iterate, SampleReduce reduce,
               std::string *error = nullptr)
{
    IRSampler spec;
    spec.kind = SamplerKind::Geometry;
    spec.input = "g";
    spec.source = element;
    spec.iterate = iterate;
    spec.reduce = reduce;
    // Width from the same frontend path the call site uses.
    auto three = Frontend::Compile(element, {Domain::Point, ScalarType::Float64, 3, true});
    auto one = Frontend::Compile(element, {Domain::Point, ScalarType::Float64, 1, true});
    const bool vector = std::string(element).find("$P") != std::string::npos ||
                        std::string(element).find("$N") != std::string::npos;
    auto &compiled = vector ? three : one;
    if (!compiled.ok) {
        if (error && !compiled.diagnostics.empty()) *error = compiled.diagnostics[0];
        return spec;
    }
    spec.components = vector ? 3 : 1;
    spec.element = std::make_shared<IRProgram>(compiled.program.IR());
    return spec;
}

double Sample(std::vector<SamplerGeometrySource> const &sources, char const *element,
              SampleIterate iterate, SampleReduce reduce, double q[3],
              unsigned component = 0, std::string *error = nullptr)
{
    GeometrySampler sampler;
    std::string message;
    if (!sampler.Build(sources, Spec(element, iterate, reduce, &message), Context{}, &message)) {
        if (error) *error = message;
        else { ++g_failures; std::printf("FAIL: build '%s': %s\n", element, message.c_str()); }
        return NAN;
    }
    return sampler.Sample(q, component);
}

void CheckGeometrySampler()
{
    std::vector<SamplerGeometrySource> curves{Curves()};
    auto P = SampleIterate::Prim;
    auto Pt = SampleIterate::Point;
    auto G = SampleIterate::Geometry;
    double a[3] = {0.9, 0, 0}, b[3] = {2.1, 0, 0}, tip[3] = {3, 3, 0};
    Check(Near(Sample(curves, "$index", P, SampleReduce::Nearest, a), 1), "nearest root to x=0.9 is curve 1");
    Check(Near(Sample(curves, "$index", P, SampleReduce::Nearest, b), 2), "nearest root to x=2.1 is curve 2");
    Check(Near(Sample(curves, "$index", P, SampleReduce::Nearest2, a), 0), "second-nearest root to x=0.9 is curve 0");
    Check(Near(Sample(curves, "$Qdist", P, SampleReduce::Nearest, a), 0.1), "$Qdist at the nearest element");
    Check(Near(Sample(curves, "$Qdist", P, SampleReduce::Nearest2, a), 0.9), "$Qdist at the second-nearest element");
    Check(Near(Sample(curves, "$Qdist", P, SampleReduce::Min, a), 0.1), "min folds a query-dependent value");
    Check(Near(Sample(curves, "1", P, SampleReduce::Sum, a), 3), "sum over prims counts them");
    Check(Near(Sample(curves, "$index", P, SampleReduce::Mean, a), 1), "mean of the prim indices");
    Check(Near(Sample(curves, "$cLength", P, SampleReduce::Max, a), 3), "curves expose $cLength");
    Check(Near(Sample(curves, "$id", P, SampleReduce::Nearest, b), 72), "authored ids reach $id");
    Check(Near(Sample(curves, "$Q[0] + $Q[1]", P, SampleReduce::Nearest, b), 2.1), "$Q is the query position");
    Check(Near(Sample(curves, "1", Pt, SampleReduce::Sum, a), 9), "point iteration visits every CV");
    Check(Near(Sample(curves, "$t", Pt, SampleReduce::Nearest, tip), 1), "the tip CV reads $t = 1");
    Check(Near(Sample(curves, "$pointIndex", Pt, SampleReduce::Nearest, tip), 2), "$pointIndex of the tip");
    Check(Near(Sample(curves, "$rootP", Pt, SampleReduce::Nearest, tip, 0), 3), "$rootP of a CV is its curve root");
    Check(Near(Sample(curves, "$P", P, SampleReduce::Nearest, a, 0), 1), "a vector element value, x");
    Check(Near(Sample(curves, "$Pref", P, SampleReduce::Nearest, a, 1), 0),
          "$Pref falls back to the current points without a rest channel");
    Check(Near(Sample(curves, "$count", G, SampleReduce::Sum, a), 1), "geometry iteration: one element per gprim");
    Check(Near(Sample(curves, "$pointCount", G, SampleReduce::Sum, a), 9), "geometry iteration: $pointCount");

    std::string error;
    Check(std::isnan(Sample(curves, "$N", P, SampleReduce::Nearest, a, 0, &error)) &&
          error.find("$N") != std::string::npos,
          "curves supply no $N, and the build says so");

    std::vector<SamplerGeometrySource> quads{Quad(0), Quad(5)};
    double near5[3] = {5.4, 0, 0.5};
    Check(Near(Sample(quads, "$index", P, SampleReduce::Nearest, near5), 1), "mesh prims are faces");
    Check(Near(Sample(quads, "$N", P, SampleReduce::Nearest, near5, 1), 1), "a face normal");
    Check(Near(Sample(quads, "$N", Pt, SampleReduce::Nearest, near5, 1), 1), "a vertex normal");
    Check(Near(Sample(quads, "$P", P, SampleReduce::Nearest, near5, 0), 5.5), "a face centroid");
    Check(Near(Sample(quads, "$index", G, SampleReduce::Nearest, near5), 1), "the nearest gprim");
    Check(Near(Sample(quads, "1", Pt, SampleReduce::Sum, near5), 8), "mesh points");

    // A time-reading element expression is legal.
    Check(Near(Sample(curves, "$frame + 1", P, SampleReduce::Nearest, a), 1), "$frame in an element expression");

    // Nothing to sample: NaN, which poisons the consuming expression.
    Check(std::isnan(Sample({}, "$index", P, SampleReduce::Nearest, a)), "an empty input samples NaN");
}

// A GeomSubset input (R15): only its faces and the points they use are
// sampled, and element ids stay the parent mesh's.
void CheckSubsetGeometry()
{
    auto P = SampleIterate::Prim;
    auto Pt = SampleIterate::Point;
    auto G = SampleIterate::Geometry;
    SamplerGeometrySource strip = Strip();
    strip.subset = true;
    strip.faces = {1};
    std::vector<SamplerGeometrySource> subset{strip};
    double left[3] = {0.2, 0, 0.1};
    Check(Near(Sample({Strip()}, "1", P, SampleReduce::Sum, left), 2), "the whole strip has two faces");
    Check(Near(Sample(subset, "1", P, SampleReduce::Sum, left), 1), "a subset samples only its faces");
    Check(Near(Sample(subset, "$primIndex", P, SampleReduce::Nearest, left), 1),
          "a subset face keeps its parent face index, even nearest the other face");
    Check(Near(Sample(subset, "$primCount", P, SampleReduce::Nearest, left), 2),
          "$primCount stays the parent's, so $primIndex / $primCount keeps its range");
    Check(Near(Sample({Strip()}, "1", Pt, SampleReduce::Sum, left), 6), "the whole strip has six points");
    Check(Near(Sample(subset, "1", Pt, SampleReduce::Sum, left), 4), "a subset samples only its faces' points");
    Check(Near(Sample(subset, "$id", Pt, SampleReduce::Nearest, left), 3),
          "a subset point keeps its parent point id");
    Check(Near(Sample(subset, "$primIndex", Pt, SampleReduce::Nearest, left), 1),
          "a subset point is owned by a subset face");
    Check(Near(Sample(subset, "$pointCount", G, SampleReduce::Sum, left), 4),
          "geometry iteration counts the subset's points");
    Check(Near(Sample(subset, "$P", G, SampleReduce::Nearest, left, 0), 1.5),
          "the geometry centroid is the subset's");

    SamplerGeometrySource empty = Strip();
    empty.subset = true;
    Check(std::isnan(Sample({empty}, "$primIndex", P, SampleReduce::Nearest, left)),
          "an empty subset selects nothing rather than the whole mesh");
    Check(std::isnan(Sample({empty}, "$id", Pt, SampleReduce::Nearest, left)),
          "an empty subset samples no point");

    SamplerGeometrySource wild = Strip();
    wild.subset = true;
    wild.faces = {2};
    std::string error;
    Check(std::isnan(Sample({wild}, "1", P, SampleReduce::Sum, left, 0, &error)) &&
          error.find("subset face") != std::string::npos,
          "an out-of-range subset face fails the build");
}

void CheckEndToEnd()
{
    // Two strands rooted at x = 0.2 and x = 2.9: a voronoi id per strand.
    std::vector<float> px{0.2f, 0.2f, 2.9f, 2.9f}, py{0, 1, 0, 1}, pz{0, 0, 0, 0};
    std::vector<uint64_t> ids{5, 6};
    std::vector<uint32_t> offsets{0, 2, 4};
    CpuCurveGeometryView view;
    view.px = px.data(); view.py = py.data(); view.pz = pz.data();
    view.stableIds = ids.data();
    view.curveOffsets = offsets.data();
    view.curveCount = 2;
    view.pointCount = 4;
    auto program = Compile("geoSampler(\"g\", \"$index\")");
    if (!program.ok) { Check(false, "end-to-end program compiles"); return; }
    CpuExpressionContext context;
    Check(context.Build(view, Context{0, 0, 0, 1, 0, 0, Domain::Primitive}) == CpuExpressionStatus::Ok,
          "context builds");
    GeometrySampler sampler;
    std::string error;
    Check(sampler.Build({Curves()}, program.program.IR().samplers[0], Context{}, &error),
          "the program's own slot builds: " + error);
    CpuExpressionSamplers table;
    CpuExpressionSamplers::Slot slot;
    slot.geometry = &sampler;
    table.slots.push_back(slot);
    auto inputs = context.Inputs();
    inputs.samplers = &table;
    std::vector<float> out(2, -1.0f);
    CpuExpressionOutput output{out.data(), 2, ScalarType::Float32, 1};
    Check(EvaluateProgram(program.program.IR(), inputs, output) == CpuExpressionStatus::Ok,
          "the sampling program evaluates");
    Check(out[0] == 0.0f && out[1] == 2.0f, "each strand reads its nearest guide's index");

    // Without a sampler table the call poisons instead of reading garbage.
    inputs.samplers = nullptr;
    Check(EvaluateProgram(program.program.IR(), inputs, output) == CpuExpressionStatus::InvalidValue,
          "no sampler table poisons the result");
}

// --- UsdGenCpuParameters over a descriptor --------------------------------

UsdGenCurveBuffer Strands()
{
    UsdGenCurveBuffer b;
    b.totalCurves = 2;
    b.totalCvs = 4;
    b.px = VtFloatArray{0.2f, 0.2f, 2.9f, 2.9f};
    b.py = VtFloatArray{0, 1, 0, 1};
    b.pz = VtFloatArray{0.25f, 0.25f, 0.25f, 0.25f};
    b.curveId = VtArray<uint64_t>{5, 6};
    b.rootPrim = VtIntArray{0, 1};
    return b;
}

UsdGenGraphDesc Description(char const *source)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom/Desc");
    UsdGenGeometryDesc guides;
    guides.path = SdfPath("/Groom/Guides");
    guides.kind = UsdGenGeometryKind::Curves;
    guides.counts = VtIntArray{2, 2, 2};
    guides.points = usdGen::UsdGenVec3fArray{usdGen::UsdGenVec3f(0, 0, 0), usdGen::UsdGenVec3f(0, 1, 0), usdGen::UsdGenVec3f(1, 0, 0),
                                 usdGen::UsdGenVec3f(1, 1, 0), usdGen::UsdGenVec3f(3, 0, 0), usdGen::UsdGenVec3f(3, 1, 0)};
    guides.generation = 1;
    desc.geometries.push_back(guides);

    UsdGenExpressionDesc expression;
    expression.path = SdfPath("/Groom/Desc/Expressions/region");
    expression.source = source;
    UsdGenExpressionInputDesc input;
    input.name = TfToken("guideCurves");
    input.targets = {guides.path};
    input.geometries = {guides.path};
    expression.inputs.push_back(input);
    desc.expressions.push_back(expression);

    UsdGenNodeDesc node;
    node.path = SdfPath("/Groom/Desc/Ops/interp");
    node.type = TfToken("UsdGenGuideInterpolate");
    UsdGenExpressionBinding binding;
    binding.expression = expression.path;
    binding.output = TfToken();
    binding.nativeType = TfToken("float");
    binding.destination = TfToken("usdGen:region");
    binding.destinationShape.scalar = ScalarType::Float32;
    binding.domain = Domain::Primitive;
    binding.literal = VtValue(0.0f);
    node.expressionBindings.push_back(binding);
    UsdGenExpressionOutputDesc result;
    result.name = TfToken("result");
    result.nativeType = TfToken("float");
    result.shape.scalar = ScalarType::Float32;
    desc.expressions[0].outputs.push_back(result);
    desc.nodes.push_back(node);
    return desc;
}

void CheckParameters()
{
    UsdGenGraphDesc desc = Description("geoSampler(\"guideCurves\", \"$index\")");
    UsdGenCpuParameters parameters;
    bool changed = false;
    std::vector<std::string> errors;
    UsdGenCurveBuffer const strands = Strands();
    bool ok = parameters.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors);
    Check(ok, "a geoSampler binding evaluates" + (errors.empty() ? "" : ": " + errors[0]));
    auto const *value = parameters.Find(TfToken("region"));
    Check(value && value->values.size() == 2 && value->values[0] == 0 && value->values[1] == 2,
          "the region field is the nearest guide index per strand");

    // Moving a guide (a new generation) re-resolves the sampler.
    desc.geometries[0].points[4] = usdGen::UsdGenVec3f(0.3f, 0, 0);
    desc.geometries[0].points[5] = usdGen::UsdGenVec3f(0.3f, 1, 0);
    desc.geometries[0].generation = 2;
    errors.clear();
    ok = parameters.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors);
    value = parameters.Find(TfToken("region"));
    Check(ok && changed && value && value->values[0] == 2 && value->values[1] == 1,
          "a guide edit changes the sampled region and reports the change");

    // A world matrix on the guides moves them into the groom's space.
    desc.geometries[0].worldMatrix.SetTranslate(usdGen::UsdGenVec3d(10, 0, 0));
    ok = parameters.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors);
    value = parameters.Find(TfToken("region"));
    Check(ok && value && value->values[1] == 0, "sampled geometry follows its world matrix");

    // Naming an input the prim does not have is a diagnostic.
    UsdGenGraphDesc missing = Description("geoSampler(\"guides\", \"$index\")");
    UsdGenCpuParameters other;
    errors.clear();
    Check(!other.Evaluate(usdGenUsd::Lower(missing), usdGenUsd::Lower(missing.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors),
          "an unknown input fails the evaluation");
    bool named = false;
    for (auto const &e : errors) named = named || e.find("no input:guides relationship") != std::string::npos;
    Check(named, "and says which relationship is missing");

    // ptex() against a geometry input is refused by name.
    UsdGenGraphDesc wrongKind = Description("ptex(\"guideCurves\")");
    UsdGenCpuParameters third;
    errors.clear();
    Check(!third.Evaluate(usdGenUsd::Lower(wrongKind), usdGenUsd::Lower(wrongKind.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors),
          "ptex() on a geometry input fails");
}

void CheckPtexParameters()
{
    // Two unit quads side by side; strand 0 roots on face 0, strand 1 on face 1.
    namespace fs = std::filesystem;
    fs::path const file = fs::temp_directory_path() / "testUsdGenExpressionSamplers.ptx";
    {
        Ptex::String error;
        PtexPtr<PtexWriter> writer(PtexWriter::open(file.string().c_str(), Ptex::mt_quad,
                                                    Ptex::dt_float, 1, -1, 2, error, true));
        if (!writer) { Check(false, "write a test ptex: " + std::string(error.c_str())); return; }
        for (int f = 0; f < 2; ++f) {
            Ptex::Res res(2, 2);
            int adjfaces[4] = {-1, -1, -1, -1};
            int adjedges[4] = {0, 0, 0, 0};
            Ptex::FaceInfo info(res, adjfaces, adjedges);
            std::vector<float> texels(res.size(), f == 0 ? 0.25f : 0.75f);
            writer->writeFace(f, info, texels.data());
        }
        Check(writer->close(error), "close the test ptex");
    }

    UsdGenGraphDesc desc = Description("ptex(\"clumpMap\")");
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Groom/Skin");
    surface.faceVertexCounts = VtIntArray{4, 4};
    surface.faceVertexIndices = VtIntArray{0, 1, 2, 3, 1, 4, 5, 2};
    surface.restPoints = usdGen::UsdGenVec3fArray{usdGen::UsdGenVec3f(0, 0, 0), usdGen::UsdGenVec3f(1.5f, 0, 0), usdGen::UsdGenVec3f(1.5f, 0, 1),
                                      usdGen::UsdGenVec3f(0, 0, 1), usdGen::UsdGenVec3f(3, 0, 0), usdGen::UsdGenVec3f(3, 0, 1)};
    desc.surfaces.push_back(surface);
    desc.nodes[0].surfaces = {surface.path};
    UsdGenMapDesc map;
    map.path = SdfPath("/Groom/Desc/Maps/clumps");
    map.type = TfToken("UsdGenPtexMap");
    map.resolvedAssetPath = file.string();
    map.params.push_back({TfToken("map:filter"), VtValue(TfToken("nearest")), false});
    map.params.push_back({TfToken("map:scale"), VtValue(2.0f), false});
    map.params.push_back({TfToken("map:offset"), VtValue(1.0f), false});
    map.params.push_back({TfToken("map:clamp"), VtValue(usdGen::UsdGenVec2f(0, 0)), false});
    desc.maps.push_back(map);
    desc.expressions[0].inputs[0].name = TfToken("clumpMap");
    desc.expressions[0].inputs[0].targets = {map.path};
    desc.expressions[0].inputs[0].geometries.clear();
    desc.expressions[0].inputs[0].maps = {map.path};

    UsdGenCurveBuffer strands = Strands();
    UsdGenCpuParameters parameters;
    bool changed = false;
    std::vector<std::string> errors;
    bool const ok = parameters.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors);
    Check(ok, "a ptex binding evaluates" + (errors.empty() ? "" : ": " + errors[0]));
    auto const *value = parameters.Find(TfToken("region"));
    Check(value && value->values.size() == 2 && Near(value->values[0], 1.5) &&
              Near(value->values[1], 2.5),
          "each strand reads its root face, scaled and offset by the map prim");
    Check(parameters.TakeWarnings().empty(), "a clean read warns about nothing");

    // Limit roots carry authoritative face UVs. They must not be projected
    // back onto the cage to recover a different texel after smoothing.
    // A varying Ptex fixture is used so merely preserving the face ID cannot
    // pass this test. Put the UV and object-space projections on opposite sides.
    {
        auto limitDesc=desc;
        auto const varyingFile=fs::temp_directory_path()/"testUsdGenLimitExpression.ptx";
        Ptex::String error;
        {PtexPtr<PtexWriter> writer(PtexWriter::open(varyingFile.string().c_str(),Ptex::mt_quad,Ptex::dt_float,1,-1,2,error,true));
        Check(bool(writer),"create varying limit Ptex");
        if(writer){int adj[4]={-1,-1,-1,-1},edges[4]={0,0,0,0};
            for(int f=0;f<2;++f){float pixels[16];for(int j=0;j<4;++j)for(int i=0;i<4;++i)pixels[j*4+i]=(i+.5f)/4;
                writer->writeFace(f,Ptex::FaceInfo(Ptex::Res(2,2),adj,edges),pixels);}
            Check(writer->close(error),"close varying limit Ptex");}}
        limitDesc.maps[0].resolvedAssetPath=varyingFile.string();
        UsdGenNodeDesc scatter;scatter.path=SdfPath("/Groom/Scatter");scatter.type=TfToken("UsdGenScatter");
        scatter.params={{TfToken("subdivisionLevel"),VtValue(3),false}};
        limitDesc.nodes[0].inputs={scatter.path};limitDesc.nodes.push_back(scatter);
        auto limitStrands=strands;limitStrands.rootUV={{.875f,.375f},{.125f,.625f}};
        UsdGenCpuParameters sampled;
        Check(sampled.Evaluate(usdGenUsd::Lower(limitDesc),usdGenUsd::Lower(limitDesc.nodes[0]),UsdGenCurveSetView::Of(limitStrands),0,0,0,&changed,&errors),"evaluate exact limit UV Ptex");
        auto const* samples=sampled.Find(TfToken("region"));
        Check(samples && samples->values.size()==2 && Near(samples->values[0],2.75) && Near(samples->values[1],1.25),"Ptex uses retained limit UV, not projected cage position");
        std::error_code ignored;fs::remove(varyingFile,ignored);
    }

    // A missing file is not a broken groom: map:default, and a warning.
    desc.maps[0].resolvedAssetPath = (fs::temp_directory_path() / "noSuchMap.ptx").string();
    desc.maps[0].params.push_back({TfToken("map:default"), VtValue(0.5f), false});
    UsdGenCpuParameters fallback;
    Check(fallback.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors),
          "a missing map still evaluates");
    value = fallback.Find(TfToken("region"));
    Check(value && Near(value->values[0], 0.5) && Near(value->values[1], 0.5),
          "every strand reads usdGen:map:default");
    Check(!fallback.TakeWarnings().empty(), "and the missing file is reported");

    // Strands with no root face ids cannot be mapped.
    desc.maps[0].resolvedAssetPath = file.string();
    strands.rootPrim = VtIntArray();
    UsdGenCpuParameters noRoots;
    errors.clear();
    Check(!noRoots.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors),
          "strands without root faces fail a ptex() read");
    std::error_code ignored;
    fs::remove(file, ignored);
}

void CheckPaintParameters()
{
    // The same two quads as CheckPtexParameters; strand 0 roots on face 0,
    // strand 1 on face 1. A paint map bilinearly samples its faceVarying
    // snapshot at each strand root's face-local (u, v): strand 0 lands in
    // triangle B (u = 2/15, v = 1/4), strand 1 in triangle A of a uniform
    // face, which still reads its mean.
    UsdGenGraphDesc desc = Description("ptex(\"lengthMap\")");
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Groom/Skin");
    surface.faceVertexCounts = VtIntArray{4, 4};
    surface.faceVertexIndices = VtIntArray{0, 1, 2, 3, 1, 4, 5, 2};
    surface.restPoints = usdGen::UsdGenVec3fArray{usdGen::UsdGenVec3f(0, 0, 0), usdGen::UsdGenVec3f(1.5f, 0, 0), usdGen::UsdGenVec3f(1.5f, 0, 1),
                                      usdGen::UsdGenVec3f(0, 0, 1), usdGen::UsdGenVec3f(3, 0, 0), usdGen::UsdGenVec3f(3, 0, 1)};
    desc.surfaces.push_back(surface);
    desc.nodes[0].surfaces = {surface.path};
    UsdGenMapDesc map;
    map.path = SdfPath("/Groom/Desc/Maps/lengths");
    map.type = TfToken("UsdGenPaintMap");
    map.paintSurface = surface.path;
    map.paintPrimvar = TfToken("usdGen:paint:length");
    map.paintInterpolation = TfToken("faceVarying");
    map.paintValues = VtFloatArray{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.0f, 2.0f, 2.0f};
    map.params.push_back({TfToken("map:scale"), VtValue(2.0f), false});
    map.params.push_back({TfToken("map:offset"), VtValue(1.0f), false});
    map.params.push_back({TfToken("map:clamp"), VtValue(usdGen::UsdGenVec2f(0, 0)), false});
    desc.maps.push_back(map);
    desc.expressions[0].inputs[0].name = TfToken("lengthMap");
    desc.expressions[0].inputs[0].targets = {map.path};
    desc.expressions[0].inputs[0].geometries.clear();
    desc.expressions[0].inputs[0].maps = {map.path};

    UsdGenCurveBuffer strands = Strands();
    UsdGenCpuParameters parameters;
    bool changed = false;
    std::vector<std::string> errors;
    bool const ok = parameters.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors);
    Check(ok, "a paint binding evaluates" + (errors.empty() ? "" : ": " + errors[0]));
    auto const *value = parameters.Find(TfToken("region"));
    Check(value && value->values.size() == 2 &&
              Near(value->values[0], 1.8166667, 1e-4) && Near(value->values[1], 5.0),
          "each strand reads the bilinear root sample, scaled and offset");
    Check(parameters.TakeWarnings().empty(), "a clean paint read warns about nothing");

    // An empty payload is not a broken groom: map:default, and a warning.
    desc.maps[0].paintValues = VtFloatArray();
    desc.maps[0].params.push_back({TfToken("map:default"), VtValue(0.5f), false});
    UsdGenCpuParameters fallback;
    errors.clear();
    Check(fallback.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors),
          "an empty paint payload still evaluates");
    value = fallback.Find(TfToken("region"));
    Check(value && Near(value->values[0], 0.5) && Near(value->values[1], 0.5),
          "every strand reads usdGen:map:default");
    Check(!fallback.TakeWarnings().empty(), "and the empty payload is reported");

    // v1 roots paint on its own surface: any other mesh fails the read.
    desc.maps[0].paintValues = VtFloatArray{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.0f, 2.0f, 2.0f};
    desc.maps[0].paintSurface = SdfPath("/Groom/OtherSkin");
    UsdGenCpuParameters wrongSurface;
    errors.clear();
    Check(!wrongSurface.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors),
          "paint for another surface fails the read");
    bool named = false;
    for (auto const &e : errors) named = named || e.find("not the root surface") != std::string::npos;
    Check(named, "and says the paint misses the root surface");

    // A GeomSubset root surface (R15) carries its parent mesh's geometry and
    // face ids, so paint on that mesh reads exactly as it does on the mesh.
    {
        UsdGenGraphDesc onSubset = desc;
        onSubset.maps[0].paintSurface = surface.path;
        onSubset.surfaces[0].path = SdfPath("/Groom/Skin/crown");
        onSubset.surfaces[0].isSubset = true;
        onSubset.surfaces[0].subsetFaces = VtIntArray{0, 1};
        onSubset.nodes[0].surfaces = {onSubset.surfaces[0].path};
        UsdGenCpuParameters subsetParameters;
        errors.clear();
        Check(subsetParameters.Evaluate(usdGenUsd::Lower(onSubset), usdGenUsd::Lower(onSubset.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0,
                                        &changed, &errors),
              "paint on a subset's parent mesh evaluates" +
                  (errors.empty() ? "" : ": " + errors[0]));
        value = subsetParameters.Find(TfToken("region"));
        Check(value && value->values.size() == 2 &&
                  Near(value->values[0], 1.8166667, 1e-4) && Near(value->values[1], 5.0),
              "and reads the parent mesh's paint");
    }

    // A payload that does not cover the faces fails the read too.
    desc.maps[0].paintSurface = surface.path;
    desc.maps[0].paintValues = VtFloatArray{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.0f, 2.0f};
    UsdGenCpuParameters shortPayload;
    errors.clear();
    Check(!shortPayload.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors),
          "a short paint payload fails the read");
    named = false;
    for (auto const &e : errors) named = named || e.find("paint values for") != std::string::npos;
    Check(named, "and says how many face vertices the payload covers");

    // The snapshot is scalar-folded: g/b/a select nothing.
    desc.maps[0].paintValues = VtFloatArray{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.0f, 2.0f, 2.0f};
    desc.maps[0].params.push_back({TfToken("map:channel"), VtValue(TfToken("g")), false});
    UsdGenCpuParameters badChannel;
    errors.clear();
    Check(!badChannel.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors),
          "a paint read on channel g fails");
    named = false;
    for (auto const &e : errors) named = named || e.find("scalar paint snapshot") != std::string::npos;
    Check(named, "and says the snapshot is scalar");

    // A root face outside the mesh reads the fallback, and is reported.
    desc.maps[0].params.pop_back();
    strands.rootPrim = VtIntArray{0, 99};
    UsdGenCpuParameters strayRoot;
    errors.clear();
    Check(strayRoot.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), UsdGenCurveSetView::Of(strands), 0, 0, 0, &changed, &errors),
          "a stray root face still evaluates" + (errors.empty() ? "" : ": " + errors[0]));
    value = strayRoot.Find(TfToken("region"));
    Check(value && Near(value->values[0], 1.8166667, 1e-4) && Near(value->values[1], 0.5),
          "the stray strand reads usdGen:map:default");
    Check(!strayRoot.TakeWarnings().empty(), "and the stray root is reported");
}

} // namespace

int main()
{
    CheckFrontend();
    CheckGeometrySampler();
    CheckSubsetGeometry();
    CheckEndToEnd();
    CheckParameters();
    CheckPtexParameters();
    CheckPaintParameters();
    std::printf("testUsdGenExpressionSamplers: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
