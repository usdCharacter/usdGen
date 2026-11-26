// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenOpContracts: what the engine knows about each registered op,
// snapshotted. Per op: the static description (topology effect, role,
// geometry arity and order, generator/time/frame/digest/remap flags,
// reference inputs, input planes), the routing table of a compiled node
// (RoutingSnapshot), which parameters are structural digest terms (observed:
// authoring the param must move the node's structural digest), the output
// planes and PlanesTouched after Configure with the fallbacks and with each
// enum token, and the expression admission matrix: every declared name plus
// "bogus", times {bool, int, float, float2, vector3f}, times {groom,
// primitive, point}. Must equal tests/golden/opContracts.txt byte for byte;
// USDGEN_GOLDEN_WRITE=1 rewrites it (a design change, never a refactor fix).

#include "golden/goldenGraphs.h"

#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGenUsd/lower.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"

#include "pxr/pxr.h"
#include "pxr/base/vt/types.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/schema.h"
#include "pxr/usd/usd/primDefinition.h"
#include "pxr/usd/usd/schemaRegistry.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_failures = 0;
void Fail(std::string const &what)
{
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
}

std::string Hex(uint32_t v)
{
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%x", v);
    return buf;
}

std::string Join(std::vector<std::string> const &v)
{
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
    return s.empty() ? "-" : s;
}

std::vector<std::string> Names(TfSpan<const TfToken> tokens)
{
    std::vector<std::string> out;
    for (TfToken const &t : tokens) out.push_back(t.GetString());
    return out;
}

std::vector<std::string> Names(TfSpan<const UsdGenPlane> planes)
{
    std::vector<std::string> out;
    for (UsdGenPlane const &p : planes) out.push_back(p.name.GetString());
    return out;
}

// Golden spelling of countChanges (level 1 = curves, level 2 = CVs).
char const *TopoFx(uint8_t countChanges)
{
    if (countChanges == 0) return "None";
    if (countChanges == UsdGenCountsAtLevel(1)) return "CurveCount";
    if (countChanges == UsdGenCountsAtLevel(2)) return "CvCount";
    if (countChanges == (UsdGenCountsAtLevel(1) | UsdGenCountsAtLevel(2))) return "Both";
    return "?";
}

UsdGenCompiledNode const *OpNode(UsdGenGraph const &graph)
{
    UsdGenNodeId const id = usdGenUsd::NodeIdForPath(graph, usdGenGolden::OpPath());
    return id == kUsdGenInvalidNode ? nullptr : &graph.Node(id);
}

void SetParam(UsdGenGraphDesc *desc, TfToken const &name, VtValue const &value)
{
    for (UsdGenNodeDesc &n : desc->nodes) {
        if (n.path != usdGenGolden::OpPath()) continue;
        for (UsdGenParamValue &p : n.params)
            if (p.name == name) { p.value = value; return; }
        n.params.push_back({name, value, false});
    }
}

/// Configure's result: output planes and PlanesTouched of the compiled op.
std::string Configured(UsdGenGraphDesc const &desc)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    if (!usdGenUsd::Compile(compiler, desc, &graph).ok) return "compile-failed";
    UsdGenCompiledNode const *node = OpNode(graph);
    if (!node || !node->op) return "no-node";
    std::vector<std::string> outputs;
    for (TfToken const &t : node->outputPrimvars) outputs.push_back(t.GetString());
    return Join(outputs) + " planes=" + Hex(node->op->PlanesTouched());
}

bool Digest(UsdGenGraphDesc const &desc, UsdGenEpoch *out)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    if (!usdGenUsd::Compile(compiler, desc, &graph).ok) return false;
    UsdGenCompiledNode const *node = OpNode(graph);
    if (!node) return false;
    *out = node->structuralDigest;
    return true;
}

struct Native { char const *name; expr::ScalarType scalar; uint32_t components; };
Native const kNatives[] = {{"bool", expr::ScalarType::Bool, 1},
                           {"int", expr::ScalarType::Int32, 1},
                           {"float", expr::ScalarType::Float32, 1},
                           {"float2", expr::ScalarType::Float32, 2},
                           {"vector3f", expr::ScalarType::Float32, 3}};

/// Accepted domains of one single-binding node: "GPT" letters or '-'.
std::string Admission(TfToken const &type, std::string const &name, Native const &native)
{
    std::string out;
    expr::Domain const domains[] = {expr::Domain::Groom, expr::Domain::Primitive,
                                    expr::Domain::Point};
    char const letters[] = {'G', 'P', 'T'};
    for (int d = 0; d < 3; ++d) {
        UsdGenNodeDesc node;
        node.path = usdGenGolden::OpPath();
        node.type = type;
        UsdGenExpressionBinding b;
        b.expression = SdfPath("/g/expr");
        b.nativeType = TfToken(native.name);
        b.destination = TfToken(name);
        b.destinationShape = expr::ValueShape{native.scalar, 1, native.components, 1, 1, false};
        b.domain = domains[d];
        node.expressionBindings.push_back(b);
        std::vector<std::string> errors;
        UsdGenGraphDesc d1;
        d1.nodes = {node};
        UsdGenGraphSpec const spec = usdGenUsd::Lower(d1);
        out += UsdGenValidateExpressionBindings(spec.nodes[0], &spec, &errors) ? letters[d] : '-';
    }
    return out;
}

void Snapshot(TfToken const &type, std::ostringstream &os)
{
    std::string const T = type.GetString();
    UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
    std::unique_ptr<UsdGenOp> const op = registry.Create(registry.Find(type));
    if (!op) { Fail(T + ": registry cannot create it"); return; }
    UsdGenOpRecord const &rec = op->Record();
    UsdGenOpDef const &opDef = rec.Def();
    os << T << " static topoFx=" << TopoFx(opDef.countChanges)
       << " role=" << (opDef.lane == UsdGenLane::Reference ? "Reference" : "Curves")
       << " arity=" << size_t(opDef.geometryInputArity)
       << " ordered=" << rec.Has(kUsdGenOpGeometryInputsOrdered)
       << " generator=" << rec.Has(kUsdGenOpGenerator)
       << " readsTime=" << rec.Has(kUsdGenOpReadsTime)
       << " samplesFrameInputs=" << rec.Has(kUsdGenOpSamplesFrameInputs)
       << " captureDigestComplete=" << rec.Has(kUsdGenOpCaptureDigestComplete)
       << " remapsVertexPlanes=" << rec.Has(kUsdGenOpRemapsVertexPlanes) << '\n';
    os << T << " referenceInputs " << Join(Names(rec.ReferenceInputs())) << '\n';
    os << T << " inputPlanes " << Join(Names(rec.InputPlanes())) << '\n';

    UsdGenGraphDesc const base = usdGenGolden::MakeOpGraph(type, false);
    if (base.nodes.empty()) { Fail(T + ": no fixture graph"); return; }
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    if (!usdGenUsd::Compile(compiler, base, &graph).ok) { Fail(T + ": fixture graph does not compile"); return; }
    std::shared_ptr<const UsdGenGraphRoutingSnapshot> const routing = graph.RoutingSnapshot();
    UsdGenGraphRoutingNode const *rn = nullptr;
    for (UsdGenGraphRoutingNode const &n : routing->nodes)
        if (usdGenUsd::PathOf(usdGenUsd::Paths(*routing), n.key) == usdGenGolden::OpPath()) rn = &n;
    if (!rn) { Fail(T + ": no routing node"); return; }
    os << T << " routedType " << rn->type.GetString() << '\n';
    std::vector<std::string> topo, value, route;
    for (TfToken const &t : rn->topologyParameters) topo.push_back(t.GetString());
    for (TfToken const &t : rn->valueParameters) value.push_back(t.GetString());
    for (auto const &r : rn->paramRouting) route.push_back(r.first.GetString() + "=" + Hex(r.second));
    os << T << " topologyParameters " << Join(topo) << '\n';
    os << T << " valueParameters " << Join(value) << '\n';
    os << T << " routing " << Join(route) << '\n';

    // Declared names, first-seen order.
    std::vector<std::string> declared;
    for (auto const &list : {topo, value})
        for (std::string const &n : list)
            if (std::find(declared.begin(), declared.end(), n) == declared.end())
                declared.push_back(n);

    // Structural digest terms, observed.
    UsdPrimDefinition const *def =
        UsdSchemaRegistry::GetInstance().FindConcretePrimDefinition(type);
    UsdGenEpoch baseDigest{};
    std::vector<std::string> digest;
    if (!Digest(base, &baseDigest)) Fail(T + ": no base digest");
    for (std::string const &n : declared) {
        VtValue value(1);
        if (def)
            if (UsdPrimDefinition::Attribute const a =
                    def->GetAttributeDefinition(TfToken("usdGen:" + n))) {
                VtValue fallback;
                if (a.GetFallbackValue(&fallback) && !fallback.IsEmpty()) value = fallback;
            }
        UsdGenGraphDesc probe = base;
        SetParam(&probe, TfToken(n), value);
        UsdGenEpoch d{};
        if (!Digest(probe, &d)) digest.push_back(n + "(compile-failed)");
        else if (d != baseDigest) digest.push_back(n);
    }
    os << T << " digest " << Join(digest) << '\n';

    // Configure: fallbacks, then each enum token of every token attribute.
    os << T << " configure[fallbacks] " << Configured(base) << '\n';
    if (def) {
        std::vector<TfToken> props = def->GetPropertyNames();
        std::sort(props.begin(), props.end(),
                  [](TfToken const &a, TfToken const &b) { return a.GetString() < b.GetString(); });
        for (TfToken const &prop : props) {
            if (prop.GetString().compare(0, 7, "usdGen:") != 0) continue;
            UsdPrimDefinition::Attribute const a = def->GetAttributeDefinition(prop);
            if (!a) continue;
            VtTokenArray allowed;
            a.GetMetadata(SdfFieldKeys->AllowedTokens, &allowed);
            for (TfToken const &tok : allowed) {
                UsdGenGraphDesc probe = base;
                std::string const name = prop.GetString().substr(7);
                SetParam(&probe, TfToken(name), VtValue(tok));
                os << T << " configure[" << name << "=" << tok.GetString() << "] "
                   << Configured(probe) << '\n';
            }
        }
    }
    if (T == "UsdGenClump") {
        for (int level : {-1, 0, 3})
            for (int levels : {0, 1, 2, 4, 5}) {
                UsdGenGraphDesc probe = base;
                SetParam(&probe, TfToken("clump:level"), VtValue(level));
                SetParam(&probe, TfToken("clump:levels"), VtValue(levels));
                os << T << " configure[clump:level=" << level << ",clump:levels=" << levels
                   << "] " << Configured(probe) << '\n';
            }
        // Descriptor-order numbering: an earlier Clump with two levels.
        UsdGenGraphDesc probe = base;
        UsdGenNodeDesc first;
        first.path = SdfPath("/g/clumpFirst");
        first.type = type;
        first.params = {{TfToken("clump:levels"), VtValue(2), false}};
        for (UsdGenNodeDesc &n : probe.nodes)
            if (n.path == usdGenGolden::OpPath()) {
                first.inputs = n.inputs;
                n.inputs = {first.path};
            }
        probe.nodes.insert(probe.nodes.begin(), first);
        os << T << " configure[after a 2-level Clump] " << Configured(probe) << '\n';
    }

    // Expression admission.
    std::vector<std::string> names = declared;
    names.push_back("bogus");
    for (std::string const &n : names) {
        os << T << " expr " << n;
        for (Native const &native : kNatives) os << ' ' << native.name << '=' << Admission(type, n, native);
        os << '\n';
    }
}

/// The file with CRLF folded to LF (git may convert on checkout).
std::string ReadFile(std::string const &path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream os;
    os << in.rdbuf();
    std::string text = os.str();
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    return text;
}

/// Per op: which snapshot rows changed.
void Diff(std::string const &expected, std::string const &actual)
{
    auto rows = [](std::string const &text) {
        std::vector<std::string> out;
        std::istringstream is(text);
        std::string line;
        while (std::getline(is, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty() && line[0] != '#') out.push_back(line);
        }
        return out;
    };
    std::vector<std::string> const a = rows(expected), b = rows(actual);
    size_t shown = 0, diffs = 0;
    for (std::string const &line : a)
        if (std::find(b.begin(), b.end(), line) == b.end()) {
            ++diffs;
            if (shown++ < 40) std::printf("  - %s\n", line.c_str());
        }
    for (std::string const &line : b)
        if (std::find(a.begin(), a.end(), line) == a.end()) {
            ++diffs;
            if (shown++ < 80) std::printf("  + %s\n", line.c_str());
        }
    std::printf("%zu contract rows differ (- golden, + now)\n", diffs);
}

}  // namespace

int main()
{
    if (!UsdSchemaRegistry::GetInstance().FindConcretePrimDefinition(TfToken("UsdGenLength"))) {
        std::printf("FAIL: the usdGen schema is not loaded (PXR_PLUGINPATH_NAME)\n");
        return 1;
    }
    std::vector<TfToken> types;
    for (uint32_t i = 0; i < UsdGenOpRegistry::Get().Count(); ++i)
        types.push_back(UsdGenOpRegistry::Get().Record(UsdGenOpId(i)).Name());
    std::sort(types.begin(), types.end(),
              [](TfToken const &a, TfToken const &b) { return a.GetString() < b.GetString(); });
    std::ostringstream os;
    os << "# testUsdGenOpContracts: the engine's view of every registered op.\n"
          "# Never edit by hand; USDGEN_GOLDEN_WRITE=1 rewrites.\n";
    for (TfToken const &type : types) Snapshot(type, os);
    std::string const text = os.str();
    std::printf("%zu ops, %zu bytes of contract text\n", types.size(), text.size());

    std::string const path = std::string(USDGEN_GOLDEN_DIR) + "/opContracts.txt";
    char const *write = std::getenv("USDGEN_GOLDEN_WRITE");
    if (write && std::strcmp(write, "1") == 0) {
        std::ofstream(path, std::ios::binary) << text;
        std::printf("wrote %s\n", path.c_str());
    } else {
        std::string const expected = ReadFile(path);
        if (expected.empty()) Fail("missing golden file tests/golden/opContracts.txt");
        else if (expected != text) {
            std::ofstream("opContracts.actual.txt", std::ios::binary) << text;
            Diff(expected, text);
            Fail("contracts differ from tests/golden/opContracts.txt (actual: opContracts.actual.txt)");
        }
    }
    if (g_failures) std::printf("testUsdGenOpContracts: FAILED (%d)\n", g_failures);
    else std::printf("testUsdGenOpContracts: PASS\n");
    return g_failures ? 1 : 0;
}
