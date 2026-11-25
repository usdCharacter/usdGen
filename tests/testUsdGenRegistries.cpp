// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenRegistries: the step-4 registries (design-step4 S4-1). Dense
// handles, lookups that never intern, duplicate and post-freeze rejection,
// FindParam with and without "usdGen:", EnumIndex, routing derived from the
// decls (checked against the compiler's routing for every built-in op), type
// and kernel registries, the interner's counters and expression admission.

#include "golden/goldenGraphs.h"

#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/token.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (ok) return;
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
}

// ---- a test op described by a def ----------------------------------------------

class TestOp : public UsdGenOp
{
public:
    explicit TestOp(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override { return {}; }
    bool Capture(UsdGenCaptureContext const &, UsdGenCurveBuffer const &, UsdGenCapture *,
                 UsdGenDiagnostics *) override { return true; }
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &,
                  UsdGenLegacyChunkView *) const override {}
};

std::unique_ptr<UsdGenOp> CreateTestOp(UsdGenOpRecord const &rec, void *)
{
    return std::make_unique<TestOp>(rec);
}

using D = expr::Domain;
constexpr char const *kModes[] = {"a", "b", "c"};
constexpr char const *kDupModes[] = {"a", "a"};
constexpr UsdGenExprAdmission kFloatRoot{expr::ScalarType::Float32, 1, "float",
                                         D::Groom | D::Primitive};
constexpr UsdGenExprAdmission kBoolGroom{expr::ScalarType::Bool, 1, "bool", D::Groom};
UsdGenParamDecl const kParams[] = {
    {"input", UsdGenParamClass::Capture},
    {"seed", UsdGenParamClass::Capture},
    {"enabled", UsdGenParamClass::Capture, true, kBoolGroom},
    {"test:mode", UsdGenParamClass::Structural, true, {}, UsdGenSpan(kModes)},
    {"test:value", UsdGenParamClass::Capture, true, kFloatRoot},
    {"test:cull", UsdGenParamClass::Topology},
    {"test:hidden", UsdGenParamClass::Structural, false},
    {"mask", UsdGenParamClass::Value},
};
constexpr char const *kRefs[] = {"guides"};
constexpr char const *kTokens[] = {"tokA", "tokB"};
constexpr UsdGenPlaneDecl kInPlanes[] = {{"clumpId_0", UsdGenPlaneRate::Uniform,
                                          UsdGenPlaneScalar::Int, 1}};
constexpr UsdGenMapSlotDecl kMaps[] = {{"usdGen:lengthMap"}};

UsdGenOpDef TestDef(char const *name)
{
    UsdGenOpDef d;
    d.name = name;
    d.displayName = "RegTest";
    d.create = &CreateTestOp;
    d.topoFx = UsdGenTopoFx::CurveCount;
    d.traits = kUsdGenOpReadsTime;
    d.params = UsdGenSpan(kParams);
    d.referenceInputs = UsdGenSpan(kRefs);
    d.inputPlanes = UsdGenSpan(kInPlanes);
    d.mapSlots = UsdGenSpan(kMaps);
    d.tokens = UsdGenSpan(kTokens);
    return d;
}

struct Registered {
    UsdGenOpId test = UsdGenOpId::Invalid;
    UsdGenOpId plain = UsdGenOpId::Invalid;
    UsdGenOpId duplicateName = UsdGenOpId::Invalid;
    UsdGenOpId duplicateParam = UsdGenOpId::Invalid;
    UsdGenOpId duplicateEnum = UsdGenOpId::Invalid;
    UsdGenOpId noCreate = UsdGenOpId::Invalid;
    UsdGenOpId tooMany = UsdGenOpId::Invalid;
    UsdGenOpId builtinDuplicate = UsdGenOpId::Invalid;
    UsdGenTypeId type = UsdGenTypeId::Invalid;
    UsdGenTypeId typeDuplicate = UsdGenTypeId::Invalid;
    bool kernel = false, kernelDuplicate = true, kernelBadOp = true;
    UsdGenRegistrar *saved = nullptr;    // a copy, for post-freeze calls
} g_reg;

std::vector<UsdGenParamDecl> g_many;

void RegisterTestLibrary(UsdGenRegistrar &r)
{
    g_reg.test = r.RegisterOp(TestDef("UsdGenRegistriesTestOp"));
    UsdGenOpDef plain;
    plain.name = "UsdGenRegistriesPlainOp";
    plain.create = &CreateTestOp;
    g_reg.plain = r.RegisterOp(plain);
    g_reg.duplicateName = r.RegisterOp(TestDef("UsdGenRegistriesTestOp"));
    g_reg.duplicateName = g_reg.duplicateName == UsdGenOpId::Invalid
        ? r.RegisterOp(TestDef("UsdGenLength")) : g_reg.duplicateName;

    static UsdGenParamDecl const dupParams[] = {{"x", UsdGenParamClass::Capture},
                                                {"y", UsdGenParamClass::Capture},
                                                {"x", UsdGenParamClass::Value}};
    UsdGenOpDef d = TestDef("UsdGenRegistriesDupParam");
    d.params = UsdGenSpan(dupParams);
    g_reg.duplicateParam = r.RegisterOp(d);

    static UsdGenParamDecl const dupEnum[] = {
        {"m", UsdGenParamClass::Structural, true, {}, UsdGenSpan(kDupModes)}};
    d = TestDef("UsdGenRegistriesDupEnum");
    d.params = UsdGenSpan(dupEnum);
    g_reg.duplicateEnum = r.RegisterOp(d);

    d = TestDef("UsdGenRegistriesNoCreate");
    d.create = nullptr;
    g_reg.noCreate = r.RegisterOp(d);

    g_many.clear();
    static std::vector<std::string> names;
    names.clear();
    for (uint32_t i = 0; i <= kUsdGenMaxParamsPerOp; ++i) names.push_back("p" + std::to_string(i));
    for (std::string const &n : names) g_many.push_back({n.c_str(), UsdGenParamClass::Capture});
    d = TestDef("UsdGenRegistriesTooMany");
    d.params = TfSpan<const UsdGenParamDecl>(g_many.data(), g_many.size());
    g_reg.tooMany = r.RegisterOp(d);

    g_reg.builtinDuplicate = r.RegisterOp(TestDef("UsdGenWidth"));

    UsdGenTypeDef t;
    t.name = "UsdGenRegistriesTestMap";
    t.traits = kUsdGenTypeMap | kUsdGenTypeMapImage;
    g_reg.type = r.RegisterType(t);
    g_reg.typeDuplicate = r.RegisterType(t);

    static UsdGenParamSlot const paramMap[] = {4, 3};
    UsdGenKernelDef k;
    k.stage = 7;
    k.paramMap = UsdGenSpan(paramMap);
    g_reg.kernel = r.RegisterKernel(g_reg.test, UsdGenBackendId(0), k);
    g_reg.kernelDuplicate = r.RegisterKernel(g_reg.test, UsdGenBackendId(0), k);
    g_reg.kernelBadOp = r.RegisterKernel(UsdGenOpId(kUsdGenMaxOps - 1), UsdGenBackendId(0), k);
    g_reg.saved = new UsdGenRegistrar(r);
}

void TestRegistration()
{
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    Check(ops.Count() == 32, "32 built-in ops registered, got " + std::to_string(ops.Count()));
    Check(!ops.IsFrozen(), "registry is open before the first compile");
    Check(UsdGenRegisterStatic(&RegisterTestLibrary), "UsdGenRegisterStatic");
    Check(g_reg.test != UsdGenOpId::Invalid, "test op registers");
    Check(g_reg.plain != UsdGenOpId::Invalid, "plain op registers");
    Check(g_reg.duplicateName == UsdGenOpId::Invalid, "duplicate op name is rejected");
    Check(g_reg.duplicateParam == UsdGenOpId::Invalid, "duplicate param is rejected");
    Check(g_reg.duplicateEnum == UsdGenOpId::Invalid, "duplicate enumerant is rejected");
    Check(g_reg.noCreate == UsdGenOpId::Invalid, "a def without create is rejected");
    Check(g_reg.tooMany == UsdGenOpId::Invalid, "more than kUsdGenMaxParamsPerOp is rejected");
    Check(g_reg.builtinDuplicate == UsdGenOpId::Invalid, "a built-in's name is rejected");
    Check(ops.Count() == 34, "rejected defs take no id");
    Check(g_reg.type != UsdGenTypeId::Invalid && g_reg.typeDuplicate == UsdGenTypeId::Invalid,
          "type registers once");
    Check(g_reg.kernel && !g_reg.kernelDuplicate && !g_reg.kernelBadOp,
          "kernel registers once, only for a registered op");
}

void TestDenseHandles()
{
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    for (uint32_t i = 0; i < ops.Count(); ++i) {
        UsdGenOpRecord const &r = ops.Record(UsdGenOpId(i));
        Check(UsdGenIndex(r.Id()) == i, "dense id " + std::to_string(i));
        Check(ops.Find(r.Name()) == r.Id(), "Find(Name()) for " + r.Name().GetString());
        Check(r.Def().create != nullptr, "create hook " + r.Name().GetString());
        std::unique_ptr<UsdGenOp> op = ops.Create(r.Id());
        Check(op && &op->Record() == &r,
              "Create(id) makes " + r.Name().GetString());
    }
    Check(ops.Record(ops.Find(TfToken("UsdGenScatter"))).Id() == UsdGenOpId(0),
          "Scatter is registered first");
    // CurveSource declares useRest and usdGen:useRest: each finds its own slot.
    UsdGenOpRecord const &cs = ops.Record(ops.Find(TfToken("UsdGenCurveSource")));
    UsdGenParamSlot const useRest = cs.FindParam(TfToken("useRest"));
    UsdGenParamSlot const prefixedUseRest = cs.FindParam(TfToken("usdGen:useRest"));
    Check(useRest != kUsdGenNoParam && prefixedUseRest != kUsdGenNoParam &&
              useRest != prefixedUseRest && cs.ParamName(prefixedUseRest) == TfToken("usdGen:useRest"),
          "explicit usdGen: spelling wins over the alias");
    Check(ops.Find(TfToken("UsdGenNoSuchOp")) == UsdGenOpId::Invalid, "unknown op");
    Check(ops.Find(TfToken()) == UsdGenOpId::Invalid, "empty name");
    Check(!ops.Create(UsdGenOpId::Invalid), "Create(Invalid) is null");
}

void TestRecord()
{
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    UsdGenOpRecord const &r = ops.Record(g_reg.test);
    Check(r.Name() == TfToken("UsdGenRegistriesTestOp"), "record name");
    Check(r.Has(kUsdGenOpReadsTime) && !r.Has(kUsdGenOpGenerator), "traits");
    Check(r.ParamCount() == 8, "param count");
    Check(r.FindParam(TfToken("test:mode")) == 3, "FindParam plain");
    Check(r.FindParam(TfToken("usdGen:test:mode")) == 3, "FindParam with usdGen:");
    Check(r.FindParam(TfToken("usdGen:input")) == 0, "FindParam base name with usdGen:");
    Check(r.FindParam(TfToken("bogus")) == kUsdGenNoParam, "FindParam unknown");
    Check(r.FindParam(TfToken()) == kUsdGenNoParam, "FindParam empty");
    Check(r.ParamName(4) == TfToken("test:value"), "ParamName");
    Check(r.ParamName(kUsdGenNoParam).IsEmpty(), "ParamName out of range");
    Check(r.ParamClass(5) == UsdGenParamClass::Topology, "ParamClass");
    Check(r.EnabledSlot() == 2, "EnabledSlot");
    Check(ops.Record(g_reg.plain).EnabledSlot() == kUsdGenNoParam, "EnabledSlot undeclared");
    Check(r.EnumIndex(3, TfToken("b")) == 1, "EnumIndex token");
    Check(r.EnumIndex(3, std::string("c")) == 2, "EnumIndex string");
    Check(r.EnumIndex(3, TfToken("z")) == -1 && r.EnumIndex(3, std::string("z")) == -1,
          "EnumIndex no match");
    Check(r.EnumIndex(4, TfToken("a")) == -1, "EnumIndex on a non-enum param");

    // Routing: compiler.cpp ClassifyParamBits rules; routed=false is absent.
    auto const routing = r.Routing();
    std::vector<std::pair<std::string, uint32_t>> const want = {
        {"input", UsdGenDirtyStructural}, {"seed", UsdGenDirtyCapture},
        {"enabled", UsdGenDirtyTopology | UsdGenDirtyCapture},
        {"test:mode", UsdGenDirtyStructural}, {"test:value", UsdGenDirtyCapture},
        {"test:cull", UsdGenDirtyTopology}, {"mask", UsdGenDirtyParameter}};
    Check(routing.size() == want.size(), "routing size");
    for (size_t i = 0; i < want.size() && i < routing.size(); ++i)
        Check(routing[i].first.GetString() == want[i].first && routing[i].second == want[i].second,
              "routing row " + want[i].first);
    Check(r.DigestSlots().size() == 1 && r.DigestSlots()[0] == 3, "digest slots");

    Check(r.ReferenceInputs().size() == 1 && r.ReferenceInputs()[0] == TfToken("guides"),
          "reference inputs");
    Check(r.InputPlanes().size() == 1 && r.InputPlanes()[0].name == TfToken("clumpId_0") &&
              r.InputPlanes()[0].interpolation == UsdGenTokens().uniform &&
              r.InputPlanes()[0].type == UsdGenTokens().int_,
          "input plane layout");
    Check(r.OutputPlanes().empty(), "no static outputs");
    Check(r.MapRelationships().size() == 1 &&
              r.MapRelationships()[0] == TfToken("usdGen:lengthMap"), "map relationships");
    Check(r.Token(1) == TfToken("tokB") && r.Token(2).IsEmpty(), "op tokens");

    UsdGenBackendRegistry const &kernels = UsdGenBackendRegistry::Get();
    UsdGenKernelDef const *k = kernels.Find(g_reg.test, UsdGenBackendId(0));
    Check(k && k->stage == 7 && k->paramMap.size() == 2 && k->paramMap[0] == 4, "kernel row");
    Check(!kernels.Find(g_reg.test, UsdGenBackendId(1)), "no kernel on another backend");
    Check(!kernels.Find(g_reg.plain, UsdGenBackendId(0)), "no kernel for another op");
    Check(kernels.Ops(UsdGenBackendId(0)).size() == 1, "capability rows");
}

void TestNoInternLookups()
{
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    UsdGenOpRecord const &r = ops.Record(g_reg.test);
    TfToken const length("UsdGenLength"), mode("usdGen:test:mode"), b("b"), ptex("UsdGenPtexMap");
    std::string const c("c");
    uint64_t const before = UsdGenTotalInternCount();
    size_t hits = 0;
    for (int i = 0; i < 10000; ++i) {
        hits += ops.Find(length) != UsdGenOpId::Invalid;
        hits += r.FindParam(mode) != kUsdGenNoParam;
        hits += r.EnumIndex(3, b) == 1;
        hits += r.EnumIndex(3, c) == 2;
        hits += UsdGenTypeRegistry::Get().Find(ptex) != UsdGenTypeId::Invalid;
    }
    Check(hits == 50000, "lookups hit");
    Check(UsdGenTotalInternCount() == before, "10k lookups intern nothing");
}

void TestTypes()
{
    UsdGenTypeRegistry const &types = UsdGenTypeRegistry::Get();
    UsdGenTypeId const ptex = types.Find(TfToken("UsdGenPtexMap"));
    UsdGenTypeId const paint = types.Find(TfToken("UsdGenPaintMap"));
    UsdGenTypeId const image = types.Find(TfToken("UsdGenImageMap"));
    UsdGenTypeId const exprMap = types.Find(TfToken("UsdGenExprMap"));
    Check(ptex != UsdGenTypeId::Invalid && types.Has(ptex, kUsdGenTypeMapPtex) &&
              types.Has(ptex, kUsdGenTypeMapExprSampleable), "UsdGenPtexMap traits");
    Check(types.Has(paint, kUsdGenTypeMapPaintSnapshot) &&
              types.Has(paint, kUsdGenTypeMapExprSampleable), "UsdGenPaintMap traits");
    Check(types.Has(image, kUsdGenTypeMapImage) && !types.Has(image, kUsdGenTypeMapExprSampleable),
          "UsdGenImageMap traits");
    Check(types.Has(exprMap, kUsdGenTypeMap) && !types.Has(exprMap, kUsdGenTypeMapPtex),
          "UsdGenExprMap traits");
    Check(types.Name(ptex) == TfToken("UsdGenPtexMap"), "type name");
    Check(types.Find(TfToken("UsdGenNoMap")) == UsdGenTypeId::Invalid &&
              !types.Has(UsdGenTypeId::Invalid, kUsdGenTypeMap), "unknown type");
    Check(types.Find(TfToken("UsdGenRegistriesTestMap")) == g_reg.type, "registered test type");
}

void TestResolveDescIds()
{
    UsdGenGraphDesc desc;
    desc.nodes.resize(2);
    desc.nodes[0].type = TfToken("UsdGenLength");
    desc.nodes[1].type = TfToken("UsdGenNoSuchOp");
    desc.maps.resize(2);
    desc.maps[0].type = TfToken("UsdGenPaintMap");
    desc.maps[1].type = TfToken("UsdGenNoMap");
    UsdGenResolveDescIds(&desc);
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    Check(desc.nodes[0].opId == ops.Find(TfToken("UsdGenLength")) &&
              desc.nodes[0].opId != UsdGenOpId::Invalid, "node opId");
    Check(desc.nodes[1].opId == UsdGenOpId::Invalid, "unknown node opId");
    Check(desc.maps[0].typeId == UsdGenTypeRegistry::Get().Find(TfToken("UsdGenPaintMap")) &&
              desc.maps[0].typeId != UsdGenTypeId::Invalid, "map typeId");
    Check(desc.maps[1].typeId == UsdGenTypeId::Invalid, "unknown map typeId");
}

UsdGenExpressionBinding Binding(char const *destination, char const *native,
                                expr::ScalarType scalar, uint32_t components, expr::Domain domain)
{
    UsdGenExpressionBinding b;
    b.expression = SdfPath("/g/expr");
    b.nativeType = TfToken(native);
    b.destination = TfToken(destination);
    b.destinationShape = expr::ValueShape{scalar, 1, components, 1, 1, false};
    b.domain = domain;
    return b;
}

bool HasError(std::vector<std::string> const &errors, std::string const &text)
{
    for (std::string const &e : errors)
        if (e.find(text) != std::string::npos) return true;
    return false;
}

void TestExpressionAdmission()
{
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    UsdGenOpRecord const &r = ops.Record(g_reg.test);
    using S = expr::ScalarType;
    UsdGenNodeDesc node;
    node.path = SdfPath("/g/op");
    node.type = r.Name();
    std::vector<std::string> errors;
    Check(UsdGenValidateExpressionBindings(r, node, &errors), "no bindings is fine");

    node.expressionBindings = {Binding("usdGen:test:value", "float", S::Float32, 1, D::Primitive),
                               Binding("enabled", "bool", S::Bool, 1, D::Groom)};
    Check(UsdGenValidateExpressionBindings(r, node, &errors) && errors.empty(), "admitted bindings");

    node.expressionBindings = {Binding("test:value", "float", S::Float32, 1, D::Point)};
    Check(!UsdGenValidateExpressionBindings(r, node, &errors) &&
              HasError(errors, "RegTest test:value requires groom/primitive evaluation"),
          "point domain rejected");
    errors.clear();
    node.expressionBindings = {Binding("enabled", "bool", S::Bool, 1, D::Primitive)};
    Check(!UsdGenValidateExpressionBindings(r, node, &errors) &&
              HasError(errors, "RegTest enabled requires groom evaluation"),
          "groom-only rejected");
    errors.clear();
    node.expressionBindings = {Binding("test:value", "int", S::Int32, 1, D::Groom)};
    Check(!UsdGenValidateExpressionBindings(r, node, &errors) &&
              HasError(errors, "unsupported or incorrectly typed RegTest expression test:value"),
          "wrong type rejected");
    errors.clear();
    node.expressionBindings = {Binding("test:cull", "float", S::Float32, 1, D::Groom),
                               Binding("bogus", "float", S::Float32, 1, D::Groom)};
    Check(!UsdGenValidateExpressionBindings(r, node, &errors) && errors.size() == 2 &&
              HasError(errors, "expression test:cull") && HasError(errors, "expression bogus"),
          "unconnectable and unknown rejected");
    errors.clear();
    node.expressionBindings = {Binding("test:value", "float", S::Float32, 1, D::Groom),
                               Binding("usdGen:test:value", "float", S::Float32, 1, D::Groom)};
    Check(!UsdGenValidateExpressionBindings(r, node, &errors), "duplicate destination rejected");
    errors.clear();
    UsdGenOpRecord const &plain = ops.Record(g_reg.plain);
    node.type = plain.Name();
    node.expressionBindings = {Binding("x", "float", S::Float32, 1, D::Groom)};
    Check(!UsdGenValidateExpressionBindings(plain, node, &errors) &&
              HasError(errors, "operator UsdGenRegistriesPlainOp at /g/op does not accept "
                               "connected (expression) parameters"),
          "op without connectable params");
}

/// Routing derived from the decls equals the compiler's routing table for
/// every built-in op.
void TestRoutingMatchesCompiler()
{
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    int compared = 0;
    for (TfToken const &type : usdGenGolden::FixtureTypes()) {
        UsdGenOpId const id = ops.Find(type);
        Check(id != UsdGenOpId::Invalid, "fixture type registered: " + type.GetString());
        if (id == UsdGenOpId::Invalid) continue;
        UsdGenGraphDesc const desc = usdGenGolden::MakeOpGraph(type, false);
        UsdGenCompiler compiler;
        UsdGenGraph graph;
        if (!compiler.Compile(desc, &graph).ok) {
            Check(false, type.GetString() + ": fixture compiles");
            continue;
        }
        auto const snapshot = graph.RoutingSnapshot();
        UsdGenGraphRoutingNode const *rn = nullptr;
        for (UsdGenGraphRoutingNode const &n : snapshot->nodes)
            if (n.path == usdGenGolden::OpPath()) rn = &n;
        if (!rn) { Check(false, type.GetString() + ": routing node"); continue; }
        UsdGenOpRecord const &r = ops.Record(id);
        auto const routing = r.Routing();
        bool same = routing.size() == rn->paramRouting.size();
        for (size_t i = 0; same && i < routing.size(); ++i)
            same = routing[i] == rn->paramRouting[i];
        Check(same, type.GetString() + ": record routing == compiled routing");
        UsdGenNodeId const nodeId = graph.NodeIdForPath(usdGenGolden::OpPath());
        Check(nodeId != kUsdGenInvalidNode && graph.Node(nodeId).opId == id &&
                  graph.Node(nodeId).record == &r, type.GetString() + ": compiled node ids");
        ++compared;
    }
    Check(compared == 32, "every built-in compared, got " + std::to_string(compared));
}

void TestInterner()
{
    TfToken const vertex("vertex");
    Check(UsdGenTokens().vertex == vertex && UsdGenTokens().float_ == TfToken("float") &&
              UsdGenTokens().renderNamespace == TfToken("__usdGenRender"),
          "engine tokens");
    Check(UsdGenTokenChecksEnabled(), "token checks on by default");
    uint64_t const total = UsdGenTotalInternCount(), exec = UsdGenExecInternCount();
    UsdGenCookTokenCheck const check;
    Check(UsdGenInternToken("registriesTestA") == TfToken("registriesTestA"), "intern text");
    Check(UsdGenTotalInternCount() == total + 1, "total count moves");
    bool const frozen = UsdGenOpRegistry::Get().IsFrozen();
    Check(UsdGenExecInternCount() == exec + (frozen ? 1 : 0), "exec count after freeze only");
    Check(check.Violated() == frozen, "UsdGenCookTokenCheck");
    {
        UsdGenCompileTokenScope scope;
        uint64_t const e = UsdGenExecInternCount();
        UsdGenInternToken("registriesTestB");
        Check(UsdGenExecInternCount() == e, "compile scope interns are not exec interns");
    }
}

void TestFrozen()
{
    UsdGenFreezeRegistries();
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    Check(ops.IsFrozen(), "frozen");
    uint32_t const count = ops.Count();
    Check(!UsdGenRegisterStatic(&RegisterTestLibrary), "UsdGenRegisterStatic after freeze");
    Check(g_reg.saved && g_reg.saved->RegisterOp(TestDef("UsdGenRegistriesLate")) ==
                             UsdGenOpId::Invalid, "RegisterOp after freeze");
    UsdGenTypeDef t;
    t.name = "UsdGenRegistriesLateMap";
    Check(g_reg.saved->RegisterType(t) == UsdGenTypeId::Invalid, "RegisterType after freeze");
    Check(!g_reg.saved->RegisterKernel(g_reg.plain, UsdGenBackendId(0), UsdGenKernelDef{}),
          "RegisterKernel after freeze");
    Check(ops.Count() == count, "nothing registered after freeze");
    UsdGenFreezeRegistries();   // idempotent
    TestInterner();
}

}  // namespace

int main()
{
    TestRegistration();
    TestDenseHandles();
    TestRecord();
    TestNoInternLookups();
    TestTypes();
    TestResolveDescIds();
    TestExpressionAdmission();
    TestInterner();
    TestRoutingMatchesCompiler();
    TestFrozen();
    if (g_failures) {
        std::printf("testUsdGenRegistries: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("testUsdGenRegistries: OK\n");
    return 0;
}
