#include "utils/pxrInterop.h"
#include "usdGen/compile/compiler.h"
#include "api/testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGenExpr/expressions/context.h"
#include <cstdio>
#include <algorithm>
#include <string>

using namespace usdGen;
static int failures = 0;
static void Check(bool v, char const *s) { if (!v) { ++failures; std::printf("FAIL: %s\n",s); } else std::printf("ok: %s\n",s); }
static expr::ValueShape FloatShape(uint32_t n=1) { return {expr::ScalarType::Float32,n,1,1,1,false}; }

int main()
{
    UsdGenCompiler compiler; UsdGenGraph graph;
    UsdGenGraphDesc baseline;
    baseline.description = SdfPath("/existing");
    baseline.defaultWidth = 0.025f;
    UsdGenNodeDesc baselineNode;
    baselineNode.path = SdfPath("/existing/width");
    baselineNode.type = TfToken("UsdGenWidth");
    baselineNode.inputs = {SdfPath("/existing/source")};
    baseline.nodes.push_back(baselineNode);
    UsdGenNodeDesc baselineSource;
    baselineSource.path = SdfPath("/existing/source");
    baselineSource.type = TfToken("UsdGenCurveSource");
    baseline.nodes.push_back(baselineSource);
    baseline.terminal = baselineNode.path;
    baseline.expressions.push_back({SdfPath("/existing/unusedExpression"), "4*2",
        {{TfToken("result"),TfToken("float"),FloatShape()}}});
    Check(usdGenUsd::Compile(compiler, baseline, &graph).ok, "seed a real existing graph");
    Check(UsdGenExpressionInputs(graph.Spec()).size() == 1 &&
          UsdGenExpressionInputView(UsdGenExpressionInputs(graph.Spec())[0]).source == "4*2",
          "compiler retains expression program descriptors");
    Check(graph.Spec().defaultWidth == baseline.defaultWidth,
          "compiler retains description width fallback");
    if (failures) return failures;
    auto preservesGraph = [&]() {
        return graph.NodeCount() == 2 &&
            graph.Spec().description == usdGenUsd::KeyOf(baseline.description) &&
            graph.Spec().terminal == usdGenUsd::KeyOf(baseline.terminal) &&
            usdGenUsd::NodeIdForPath(graph, baselineNode.path) != UsdGenGraph::InvalidNode;
    };
    UsdGenGraphDesc valid;
    valid.description = SdfPath("/groom");
    valid.expressions.push_back({SdfPath("/groom/Expressions/width"), "$value", {{TfToken("result"), TfToken("float"), FloatShape()}}});
    UsdGenNodeDesc node; node.path=SdfPath("/groom/width"); node.type=TfToken("UsdGenWidth");
    node.inputs = {SdfPath("/groom/source")};
    node.expressionBindings.push_back({SdfPath("/groom/Expressions/width"), TfToken("result"), TfToken("float"), TfToken("width"), FloatShape(), expr::Domain::Point, VtValue(0.1f)});
    valid.nodes.push_back(node);
    UsdGenNodeDesc source; source.path=SdfPath("/groom/source"); source.type=TfToken("UsdGenCurveSource");
    valid.nodes.push_back(source);
    valid.terminal = node.path;
    // The CPU reference lane now carries an expression evaluator of its own
    // (expressions/cpuEvaluator.cpp), so a well-formed connected parameter
    // compiles on EVERY backend. It is compiled into its own graph so the
    // preservesGraph() invariant below still describes the seeded graph.
    UsdGenGraph cpuGraph;
    auto first = usdGenUsd::Compile(compiler, valid, &cpuGraph);
    Check(first.ok, "connected binding compiles on the CPU reference backend");
    Check(preservesGraph(), "a separate compilation leaves the existing graph intact");
    auto unknown = baseline;
    unknown.nodes[0].type = TfToken("UsdGenUnregisteredTestOperator");
    Check(!usdGenUsd::Compile(compiler, unknown,&graph).ok && preservesGraph(),
          "failed fresh compilation leaves graph contents intact");
    Check(!usdGenUsd::Recompile(compiler, unknown,&graph).ok && preservesGraph(),
          "failed incremental compilation leaves graph contents intact");
    auto invalidBackend = baseline;
    invalidBackend.executionBackend = static_cast<UsdGenBackendId>(255);
    Check(!usdGenUsd::Compile(compiler, invalidBackend,&graph).ok && preservesGraph(),
          "invalid backend cannot become CPU reference execution");
    auto invalidTransport = baseline;
    invalidTransport.validationErrors = {"malformed authored enabled"};
    Check(!usdGenUsd::Compile(compiler, invalidTransport,&graph).ok && preservesGraph(),
          "typed transport errors cannot become default parameters");

    auto expectFail = [&](UsdGenGraphDesc d, char const *label, char const *reason) {
        UsdGenCompileResult r=usdGenUsd::Compile(compiler, d,&graph);
        bool specific = std::any_of(r.errors.begin(), r.errors.end(),
            [&](std::string const &message) { return message.find(reason) != std::string::npos; });
        Check(!r.ok && specific, label);
        Check(preservesGraph(), "failed binding validation preserves existing graph contents");
    };
    auto missing = valid; missing.nodes[0].expressionBindings[0].expression=SdfPath("/missing");
    expectFail(missing,"missing expression fails closed", "references missing expression");
    auto shape = valid; shape.nodes[0].expressionBindings[0].destinationShape=FloatShape(3);
    expectFail(shape,"shape mismatch fails closed", "type/shape");
    auto duplicate = valid; duplicate.nodes[0].expressionBindings.push_back(duplicate.nodes[0].expressionBindings[0]);
    expectFail(duplicate,"duplicate consumer fails closed", "duplicate expression consumer");
    auto stringLike = valid; stringLike.expressions[0].outputs[0].shape.scalar=expr::ScalarType::Invalid;
    expectFail(stringLike,"string-like output fails closed", "type/shape");
    auto cuda = valid; cuda.executionBackend=usdGenTestUtils::Backend("cuda");
    expectFail(cuda,"CUDA backend fails closed without runtime evaluator", "CUDA");
    auto badDomain = valid; badDomain.nodes[0].expressionBindings[0].domain=expr::Domain::All;
    expectFail(badDomain,"invalid combined domain fails closed", "invalid expression evaluation domain");
    for (char const *control : {"enabled", "seed", "segments", "cvCount"}) {
        for (char const *prefix : {"", "usdGen:"}) {
            auto topology = valid;
            topology.nodes[0].expressionBindings[0].destination = TfToken(std::string(prefix) + control);
            expectFail(topology, "native and local control names require groom evaluation",
                       "topology/control expression must evaluate at groom domain");
        }
    }
    // The per-operator allowlist is one shared table (the op record's decls)
    // that the compiler applies on every backend and the CUDA admission calls
    // instead of keeping a copy, so these rejections are lane-independent.
    auto foreignTarget = valid;
    foreignTarget.nodes[0].expressionBindings[0].destination = TfToken("noise:magnitude");
    expectFail(foreignTarget, "a destination another operator owns fails closed",
               "unsupported or incorrectly typed Width expression");
    auto wrongType = valid;
    wrongType.nodes[0].expressionBindings[0].nativeType = TfToken("double");
    wrongType.expressions[0].outputs[0].nativeType = TfToken("double");
    expectFail(wrongType, "a non-native destination type fails closed",
               "unsupported or incorrectly typed Width expression");
    auto generator = valid;
    generator.nodes[0].type = TfToken("UsdGenScatter");
    expectFail(generator, "a source generator refuses connected parameters",
               "does not accept connected (expression) parameters");
    // Grow bakes per-strand length/lift at capture, so it accepts
    // groom/primitive connections but fails closed on anything else.
    auto grow = valid;
    grow.nodes[0].type = TfToken("UsdGenGrow");
    grow.nodes[0].expressionBindings[0].destination = TfToken("lift");
    grow.nodes[0].expressionBindings[0].domain = expr::Domain::Primitive;
    Check(usdGenUsd::Compile(compiler, grow, &cpuGraph).ok,
          "a primitive-domain usdGen:lift connection compiles on Grow");
    for(char const* name : {"azimuth", "azimuthRandom"}) {
        auto azimuth = grow;
        azimuth.nodes[0].expressionBindings[0].destination = TfToken(name);
        Check(usdGenUsd::Compile(compiler, azimuth, &cpuGraph).ok,
              "a primitive-domain azimuth control connection compiles on Grow");
        azimuth.nodes[0].expressionBindings[0].domain = expr::Domain::Point;
        expectFail(azimuth, "point-domain Grow azimuth fails closed",
                   "requires groom/primitive evaluation");
    }
    auto growForeign = grow;
    growForeign.nodes[0].expressionBindings[0].destination = TfToken("width");
    expectFail(growForeign, "a destination Grow does not own fails closed",
               "unsupported or incorrectly typed Grow expression");
    auto growPoint = grow;
    growPoint.nodes[0].expressionBindings[0].domain = expr::Domain::Point;
    expectFail(growPoint, "a point-domain Grow lift fails closed",
               "requires groom/primitive evaluation");
    auto maskPrimitive = valid;
    maskPrimitive.nodes[0].expressionBindings[0].destination = TfToken("usdGen:mask");
    maskPrimitive.nodes[0].expressionBindings[0].domain = expr::Domain::Primitive;
    Check(usdGenUsd::Compile(compiler, maskPrimitive, &cpuGraph).ok,
          "a primitive-domain usdGen:mask connection compiles");

    std::printf("testUsdGenExpressionBindings: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
