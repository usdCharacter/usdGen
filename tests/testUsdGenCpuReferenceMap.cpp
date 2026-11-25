// CPU-only reference/map lowering contract.  External guide and map
// descriptors become immutable compiled values, and reference-role capture
// finishes before ordinary curve work regardless of Kahn tie-break order.

#include "usdGen/compiler.h"
#include "nodeSets.h"
#include "usdGen/graph.h"
#include "usdGen/imagePayload.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/scheduler.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3f.h"

#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int gFailures = 0;
int gReferenceCaptures = 0;
int gCurveCaptures = 0;
uint64_t gReferenceGeneration = 0;
uint64_t gSecondReferenceGeneration = 0;
uint64_t gReferenceIdentity = 0;
uint64_t gMapGeneration = 0;
uint64_t gMapIdentity = 0;
std::vector<int> gOrder;

void Check(bool ok, std::string const &what)
{
    if (!ok) { ++gFailures; std::printf("FAIL: %s\n", what.c_str()); }
    else std::printf("ok:   %s\n", what.c_str());
}

bool RestFallbackMatchesPoints(UsdGenCurveBuffer const &buffer)
{
    // Recapture snapshots the current reference points, not the first cook.
    if (buffer.rest.size() != buffer.totalCvs ||
        buffer.px.size() != buffer.totalCvs ||
        buffer.py.size() != buffer.totalCvs ||
        buffer.pz.size() != buffer.totalCvs)
        return false;
    for (size_t i = 0; i != buffer.totalCvs; ++i) {
        if (buffer.rest[i] != GfVec3f(buffer.px[i], buffer.py[i], buffer.pz[i]))
            return false;
    }
    return true;
}

class ReferenceProbe final : public UsdGenOp
{
public:
    explicit ReferenceProbe(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override { return {31, 7}; }
    bool Capture(UsdGenCaptureContext const &ctx, UsdGenOpInputs const &,
                 UsdGenCapture *, UsdGenDiagnostics *diag) override
    {
        if (ctx.referenceCount != 2 || !ctx.references || !ctx.resolvedReferences ||
            !ctx.references[0] || !ctx.references[1] || !ctx.resolvedReferences[0] ||
            !ctx.resolvedReferences[1] || ctx.mapCount != 2 ||
            !ctx.maps || !ctx.maps[0] || !ctx.maps[1] ||
            ctx.mapBindingCount != 2 || !ctx.mapBindings) {
            if (diag) diag->Error("reference probe did not receive compiled external values");
            return false;
        }
        ++gReferenceCaptures;
        gOrder.push_back(1);
        gReferenceGeneration = ctx.resolvedReferences[0]->curveGeneration;
        gSecondReferenceGeneration = ctx.resolvedReferences[1]->curveGeneration;
        gReferenceIdentity = ctx.resolvedReferences[0]->identity;
        gMapGeneration = ctx.maps[0]->textureGeneration;
        gMapIdentity = ctx.maps[0]->identity;
        return ctx.resolvedReferences[0]->path == SdfPath("/referenceMap/guidesA") &&
            ctx.resolvedReferences[1]->path == SdfPath("/referenceMap/guidesB") &&
            ctx.references[0]->generation == gReferenceGeneration &&
            ctx.references[0]->buffer.totalCurves == 2 &&
            ctx.references[0]->buffer.totalCvs == 4 &&
            RestFallbackMatchesPoints(ctx.references[0]->buffer) &&
            ctx.mapBindings[0].relationship == TfToken("usdGen:map") &&
            ctx.mapBindings[1].relationship == TfToken("usdGen:map");
    }
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &, UsdGenChunkView const &) const override {}
};

class CurveProbe final : public UsdGenOp
{
public:
    explicit CurveProbe(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override { return {41, 9}; }
    bool Capture(UsdGenCaptureContext const &, UsdGenOpInputs const &,
                 UsdGenCapture *, UsdGenDiagnostics *) override
    {
        ++gCurveCaptures;
        gOrder.push_back(2);
        return true;
    }
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &, UsdGenChunkView const &) const override {}
};

class ContractProbe final : public UsdGenOp
{
public:
    ContractProbe(UsdGenOpRecord const &r, bool reference) : UsdGenOp(r), _reference(reference) {}
    bool Bind(UsdGenParamView const&, UsdGenDiagnostics*) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const&) const override { return {1,2}; }
    bool Capture(UsdGenCaptureContext const&, UsdGenOpInputs const &,
                 UsdGenCapture*, UsdGenDiagnostics*) override { return true; }
    void Evaluate(UsdGenEvalContext const&, UsdGenCapture const&, UsdGenChunkView const &) const override {}
private:
    [[maybe_unused]] bool _reference;
};

int gContractConstructions = 0;

char const *const kProbeReferences[]={"guides","centers"};
// Map-taking probes declare their map relationship (the compiler validates
// typed bindings against UsdGenOpDef::mapSlots).
constexpr UsdGenMapSlotDecl kProbeMapSlots[]={{"usdGen:map"}};

void RegisterProbes(UsdGenRegistrar &r)
{
    UsdGenOpDef ref;
    ref.name="UsdGenTestReferenceProbe";
    ref.displayName="TestReferenceProbe";
    ref.create=[](UsdGenOpRecord const &rec, void *)->std::unique_ptr<UsdGenOp>{return std::make_unique<ReferenceProbe>(rec);};
    ref.role=UsdGenRole::Reference;
    ref.geometryInputArity=0;
    ref.referenceInputs=UsdGenSpan(kProbeReferences);
    ref.mapSlots=UsdGenSpan(kProbeMapSlots);
    r.RegisterOp(ref);
    UsdGenOpDef curve;
    curve.name="UsdGenTestCurveProbe";
    curve.displayName="TestCurveProbe";
    curve.create=[](UsdGenOpRecord const &rec, void *)->std::unique_ptr<UsdGenOp>{return std::make_unique<CurveProbe>(rec);};
    curve.geometryInputArity=0;
    r.RegisterOp(curve);
}

bool gContractRegistered[3]={false,false,false};
char const *const kContractReferences[]={"first","second"};

void RegisterContractProbes(UsdGenRegistrar &r)
{
    // The contract lives in the def, so registration constructs no operator.
    auto create=[](UsdGenOpRecord const &rec, void *context)->std::unique_ptr<UsdGenOp>{
        ++gContractConstructions;
        return std::make_unique<ContractProbe>(rec, context!=nullptr);
    };
    static int reference=0;
    UsdGenOpDef source;
    source.name="UsdGenTestSourceContract";
    source.displayName="TestSourceContract";
    source.create=create;
    source.geometryInputArity=0;
    source.mapSlots=UsdGenSpan(kProbeMapSlots);
    gContractRegistered[0]=r.RegisterOp(source)!=UsdGenOpId::Invalid;
    UsdGenOpDef ref;
    ref.name="UsdGenTestReferenceContract";
    ref.displayName="TestReferenceContract";
    ref.create=create;
    ref.createContext=&reference;
    ref.role=UsdGenRole::Reference;
    ref.geometryInputArity=1;
    ref.referenceInputs=UsdGenSpan(kContractReferences);
    gContractRegistered[1]=r.RegisterOp(ref)!=UsdGenOpId::Invalid;
    UsdGenOpDef again=source;
    again.createContext=&reference;
    gContractRegistered[2]=r.RegisterOp(again)!=UsdGenOpId::Invalid;
}

bool OperatorContract(UsdGenOpRegistry const &registry,TfToken const &type,size_t *geometry,
                      size_t *references,UsdGenRole *role)
{
    UsdGenOpId const id=registry.Find(type);
    if(id==UsdGenOpId::Invalid)return false;
    UsdGenOpRecord const &record=registry.Record(id);
    *geometry=record.Def().geometryInputArity;
    *references=record.ReferenceInputs().size();
    *role=record.Def().role;
    return true;
}

void ValidateRegistryContractReuse()
{
    auto& registry=UsdGenOpRegistry::Get();
    TfToken const type("UsdGenTestSourceContract");
    TfToken const referenceType("UsdGenTestReferenceContract");
    Check(gContractRegistered[0]&&gContractRegistered[1],
          "registers one static input contract per type");
    int const probed=gContractConstructions;
    Check(probed==0,"registration constructs no operator");
    Check(!gContractRegistered[2],
          "a type has exactly one kernel: re-registration is refused");
    Check(gContractConstructions==probed,"a refused registration constructs nothing");
    size_t geometry=99,references=99;UsdGenRole role=UsdGenRole::Reference;
    Check(OperatorContract(registry,type,&geometry,&references,&role)&&
          geometry==0&&references==0&&role==UsdGenRole::Curves,
          "the source type retains its geometry/reference/role contract");
    Check(OperatorContract(registry,referenceType,&geometry,&references,&role)&&
          geometry==1&&references==2&&role==UsdGenRole::Reference,
          "the reference type has its distinct reference contract");
    Check(!OperatorContract(registry,TfToken("UsdGenMissingContract"),&geometry,&references,&role),
          "a missing type does not borrow a different contract");
    Check(gContractConstructions==probed,"contract queries do not construct operators");

    UsdGenGraphDesc desc;desc.description=SdfPath("/contract");desc.terminal=SdfPath("/contract/source");
    UsdGenNodeDesc source;source.path=desc.terminal;source.type=type;
    desc.nodes={source};
    UsdGenCompiler compiler;UsdGenGraph graph;
    Check(compiler.Compile(desc,&graph).ok,"contract probe source compiles");
    Check(gContractConstructions==probed+1,"fresh compilation creates only the executable operator");
    auto const* oldOp=graph.Node(0).op.get();
    auto const digest=graph.Node(0).structuralDigest;
    Check(compiler.Recompile(desc,&graph).ok&&graph.Node(0).op.get()==oldOp&&
          graph.Node(0).structuralDigest==digest&&gContractConstructions==probed+1,
          "unchanged recompile retains operator/digest without preflight factory calls");

    auto invalid=desc;UsdGenNodeDesc reference;
    reference.path=SdfPath("/contract/reference");reference.type=referenceType;
    reference.inputs={source.path};invalid.nodes.push_back(reference);invalid.terminal=reference.path;
    auto result=compiler.Recompile(invalid,&graph);
    Check(!result.ok&&!result.errors.empty()&&
          result.errors.front()=="UsdGenCompiler: node '/contract/reference' has 0 resolved reference values but its operator declares 2 reference slots",
          "cached Reference role still rejects missing reference slots exactly");
    Check(graph.NodeCount()==1&&graph.Node(0).op.get()==oldOp&&
          graph.Node(0).structuralDigest==digest&&gContractConstructions==probed+1,
          "failed contract preflight preserves prior graph without constructing candidates");
    UsdGenCurveSetDesc external;external.path=SdfPath("/contract/guides");
    external.role=UsdGenRole::Reference;external.curveRole=TfToken("guide");
    external.curveVertexCounts={2};external.points={{0,0,0},{0,1,0}};
    invalid.curveSets={external};invalid.nodes.back().references={external.path};
    result=compiler.Recompile(invalid,&graph);
    Check(!result.ok&&!result.errors.empty()&&
          result.errors.front()=="UsdGenCompiler: node '/contract/reference' has 1 resolved reference values but its operator declares 2 reference slots"&&
          graph.NodeCount()==1&&graph.Node(0).op.get()==oldOp&&gContractConstructions==probed+1,
          "nonempty reference mismatch uses exact cached arity without mutating prior graph");

    // Exercise the nonempty-to-empty transition as well as the always-empty
    // path. Retaining a compiled node must never retain stale map handles.
    auto mapped=desc;
    UsdGenMapDesc map;map.path=SdfPath("/contract/mask");
    map.type=TfToken("UsdGenImageMap");map.textureGeneration=1;
    mapped.maps={map};mapped.nodes[0].maps={map.path};
    UsdGenGraph mapGraph;
    Check(compiler.Compile(mapped,&mapGraph).ok&&mapGraph.Node(0).mapValues.size()==1,
          "legacy map takes the nonempty external binding path");
    mapped.nodes[0].maps.clear();
    Check(compiler.Recompile(mapped,&mapGraph).ok&&
          mapGraph.Node(0).mapValues.empty()&&mapGraph.Node(0).referenceValues.empty(),
          "removing the last external binding clears compiled handles");
    mapped.nodes[0].mapBindings={{map.path, TfToken("usdGen:map")}};
    Check(compiler.Recompile(mapped,&mapGraph).ok&&mapGraph.Node(0).mapValues.size()==1,
          "typed-only map still takes the nonempty external binding path");
    mapped.nodes[0].mapBindings.clear();
    Check(compiler.Recompile(mapped,&mapGraph).ok&&mapGraph.Node(0).mapValues.empty(),
          "removing the last typed binding clears compiled handles");
}

UsdGenGraphDesc Fixture()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/referenceMap");
    desc.terminal = SdfPath("/referenceMap/curve");

    UsdGenCurveSetDesc guide;
    guide.path = SdfPath("/referenceMap/guidesA");
    guide.role = UsdGenRole::Reference;
    guide.curveRole = TfToken("guide");
    guide.curveVertexCounts = VtIntArray{2, 2};
    guide.points = VtVec3fArray{GfVec3f(0, 0, 0), GfVec3f(0, 1, 0),
                                GfVec3f(1, 0, 0), GfVec3f(1, 1, 0)};
    guide.curveGeneration = 17;
    desc.curveSets.push_back(std::move(guide));
    UsdGenCurveSetDesc guideB = desc.curveSets.front();
    guideB.path = SdfPath("/referenceMap/guidesB");
    guideB.curveGeneration = 23;
    desc.curveSets.push_back(std::move(guideB));

    UsdGenMapDesc map;
    map.path = SdfPath("/referenceMap/mask");
    map.type = TfToken("UsdGenImageMap");
    map.resolvedAssetPath = "/resolved/mask.exr";
    map.textureGeneration = 9;
    map.params.push_back({TfToken("gain"), VtValue(0.5f), false});
    desc.maps.push_back(std::move(map));

    UsdGenMapDesc secondMap;
    secondMap.path = SdfPath("/referenceMap/tint");
    secondMap.type = TfToken("UsdGenImageMap");
    secondMap.resolvedAssetPath = "/resolved/tint.exr";
    secondMap.textureGeneration = 11;
    desc.maps.push_back(std::move(secondMap));

    // Namespace order would normally place curve before reference.  The CPU
    // lane must nevertheless run the reference capture first.
    UsdGenNodeDesc curve;
    curve.path = desc.terminal;
    curve.type = TfToken("UsdGenTestCurveProbe");
    UsdGenNodeDesc reference;
    reference.path = SdfPath("/referenceMap/reference");
    reference.type = TfToken("UsdGenTestReferenceProbe");
    reference.references = {SdfPath("/referenceMap/guidesA"),
                            SdfPath("/referenceMap/guidesB")};
    reference.maps = {SdfPath("/referenceMap/mask"),
                      SdfPath("/referenceMap/tint")};
    reference.mapBindings = {
        {SdfPath("/referenceMap/mask"), TfToken("usdGen:map")},
        {SdfPath("/referenceMap/tint"), TfToken("usdGen:map")}};
    desc.nodes = {curve, reference};
    return desc;
}

void RunReferenceMapLowering()
{
    gReferenceCaptures = gCurveCaptures = 0;
    gReferenceGeneration = gSecondReferenceGeneration = gReferenceIdentity =
        gMapGeneration = gMapIdentity = 0;
    gOrder.clear();
    UsdGenGraphDesc desc = Fixture();
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok, "reference/map CPU fixture compiles");
    Check(graph.ReferenceValues().size() == 2 && graph.MapValues().size() == 2 &&
              graph.ReferenceValues()[0].value &&
              graph.ReferenceValues()[0].curveGeneration == 17 &&
              graph.ReferenceValues()[1].curveGeneration == 23 &&
              graph.MapValues()[0].textureGeneration == 9,
          "compiler exposes immutable resolved reference and map values");
    UsdGenCompiledNode const *referenceNode = nullptr;
    for (size_t i = 0; i < static_cast<size_t>(graph.NodeCount()); ++i) {
        if (graph.Node(static_cast<UsdGenNodeId>(i)).record->Name() ==
            TfToken("UsdGenTestReferenceProbe")) {
            referenceNode = &graph.Node(static_cast<UsdGenNodeId>(i));
            break;
        }
    }
    Check(referenceNode && referenceNode->mapBindingRefs.size() == 2 &&
              referenceNode->mapBindingRefs[0].map == SdfPath("/referenceMap/mask") &&
              referenceNode->mapBindingRefs[1].map == SdfPath("/referenceMap/tint"),
          "compiler preserves every typed map binding in authored order");

    UsdGenGraphDesc legacy = Fixture();
    legacy.nodes[1].mapBindings.clear();
    legacy.nodes[1].maps = {SdfPath("/referenceMap/mask")};
    UsdGenGraph legacyGraph;
    UsdGenCompiledNode const *legacyReferenceNode = nullptr;
    if (compiler.Compile(legacy, &legacyGraph).ok) {
        for (size_t i = 0; i < static_cast<size_t>(legacyGraph.NodeCount()); ++i) {
            if (legacyGraph.Node(static_cast<UsdGenNodeId>(i)).record->Name() ==
                TfToken("UsdGenTestReferenceProbe")) {
                legacyReferenceNode =
                    &legacyGraph.Node(static_cast<UsdGenNodeId>(i));
                break;
            }
        }
    }
    Check(legacyReferenceNode && legacyReferenceNode->mapBindingRefs.size() == 1 &&
              legacyReferenceNode->mapBindingRefs[0].relationship.IsEmpty(),
          "legacy maps-only input carries no authored relationship token");

    UsdGenScheduler scheduler(1);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 1).diagnostics.HasErrors(),
          "CPU reference lane executes without diagnostics");
    Check(gReferenceCaptures == 1 && gCurveCaptures == 1 &&
              gOrder == std::vector<int>({1, 2}),
          "reference-role capture completes before curve-role capture");
    Check(gReferenceGeneration == 17 && gSecondReferenceGeneration == 23 &&
              gMapGeneration == 9 &&
              gReferenceIdentity == graph.ReferenceValues()[0].identity &&
              gMapIdentity == graph.MapValues()[0].identity,
          "capture receives deterministic immutable external identities");

    desc.curveSets[0].curveGeneration = 18;
    desc.curveSets[0].points[0] = GfVec3f(3, 0, 0);
    Check(compiler.Recompile(desc, &graph).ok, "reference generation edit recompiles");
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 2).diagnostics.HasErrors(),
          "reference generation edit executes");
    Check(gReferenceCaptures == 2 && gReferenceGeneration == 18,
          "reference generation participates in capture identity");

    desc.maps[0].textureGeneration = 10;
    Check(compiler.Recompile(desc, &graph).ok, "map generation edit recompiles");
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 3).diagnostics.HasErrors(),
          "map generation edit executes");
    Check(gReferenceCaptures == 3 && gMapGeneration == 10,
          "map generation participates in capture identity");
}

void ValidateFailures()
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenGraphDesc duplicate = Fixture();
    duplicate.maps.push_back(duplicate.maps.front());
    UsdGenCompileResult result = compiler.Compile(duplicate, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("duplicate map descriptor") != std::string::npos,
          "duplicate map values fail closed deterministically");

    UsdGenGraphDesc duplicateReference = Fixture();
    duplicateReference.curveSets.push_back(duplicateReference.curveSets.front());
    result = compiler.Compile(duplicateReference, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("duplicate reference curve set") != std::string::npos,
          "duplicate reference values fail closed deterministically");

    UsdGenGraphDesc badRestSize = Fixture();
    badRestSize.curveSets[0].rest = VtVec3fArray{GfVec3f(0, 0, 0)};
    result = compiler.Compile(badRestSize, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("inconsistent topology/value array sizes") !=
                  std::string::npos,
          "reference rest cardinality fails closed deterministically");

    UsdGenGraphDesc badRestFinite = Fixture();
    badRestFinite.curveSets[0].rest = VtVec3fArray{
        GfVec3f(std::numeric_limits<float>::quiet_NaN(), 0, 0),
        GfVec3f(0, 1, 0), GfVec3f(1, 0, 0), GfVec3f(1, 1, 0)};
    result = compiler.Compile(badRestFinite, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("non-finite rest points") != std::string::npos,
          "reference rest finiteness fails closed deterministically");

    UsdGenGraphDesc missing = Fixture();
    missing.nodes[1].references = {SdfPath("/referenceMap/missing"),
                                   SdfPath("/referenceMap/guidesB")};
    result = compiler.Compile(missing, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("unresolved reference value") != std::string::npos,
          "missing reference value fails closed deterministically");

    UsdGenGraphDesc duplicateNodeReference = Fixture();
    duplicateNodeReference.nodes[1].references = {
        SdfPath("/referenceMap/guidesA"), SdfPath("/referenceMap/guidesA")};
    result = compiler.Compile(duplicateNodeReference, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("duplicate reference value") != std::string::npos,
          "duplicate node reference slots fail closed deterministically");

    UsdGenGraphDesc duplicateBinding = Fixture();
    duplicateBinding.nodes[1].mapBindings.push_back(
        duplicateBinding.nodes[1].mapBindings.front());
    duplicateBinding.nodes[1].maps.push_back(SdfPath("/referenceMap/mask"));
    result = compiler.Compile(duplicateBinding, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("duplicate map binding") != std::string::npos,
          "ambiguous duplicate typed map binding fails closed");

    UsdGenGraphDesc disagreeingLegacy = Fixture();
    disagreeingLegacy.nodes[1].maps.pop_back();
    result = compiler.Compile(disagreeingLegacy, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("ambiguous map bindings") != std::string::npos,
          "disagreeing legacy and typed map transport fails closed");

    UsdGenGraphDesc malformedRelationship = Fixture();
    malformedRelationship.nodes[1].mapBindings[0].relationship =
        TfToken("usdGen:guides");
    result = compiler.Compile(malformedRelationship, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("is not a map relationship") != std::string::npos,
          "a non-map relationship token on a map binding fails closed");

    UsdGenGraphDesc cuda = Fixture();
    cuda.executionBackend = UsdGenExecutionBackend::Cuda;
    result = compiler.Compile(cuda, &graph);
    Check(!result.ok && !result.errors.empty(),
          "CUDA reference/map fixture remains fail-closed without device transport");

    UsdGenGraphDesc typedCuda = Fixture();
    typedCuda.executionBackend = UsdGenExecutionBackend::Cuda;
    typedCuda.nodes[1].maps.clear();
    result = compiler.Compile(typedCuda, &graph);
    Check(!result.ok && !result.errors.empty(),
          "CUDA rejects typed-only map bindings before device transport exists");
}

void ValidateMapIdentityAndRecompileTransaction()
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenGraphDesc first = Fixture();
    first.maps[0].params.push_back({TfToken("bias"), VtValue(0.25f), false});
    Check(compiler.Compile(first, &graph).ok,
          "canonical map identity fixture compiles");
    uint64_t const canonicalIdentity = graph.MapValues()[0].identity;

    UsdGenGraphDesc reordered = first;
    std::swap(reordered.maps[0].params[0], reordered.maps[0].params[1]);
    UsdGenGraph reorderedGraph;
    Check(compiler.Compile(reordered, &reorderedGraph).ok &&
              reorderedGraph.MapValues()[0].identity == canonicalIdentity,
          "map identity is independent of authored parameter order");

    UsdGenScheduler scheduler(1);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 1).diagnostics.HasErrors(),
          "valid graph executes before failed recompile");
    uint64_t const savedReferenceIdentity = graph.ReferenceValues()[0].identity;
    uint64_t const savedMapIdentity = graph.MapValues()[0].identity;
    SdfPath const savedTerminal = graph.Desc().terminal;

    UsdGenGraphDesc invalid = first;
    invalid.maps.push_back(invalid.maps.front());
    UsdGenCompileResult const failed = compiler.Recompile(invalid, &graph);
    Check(!failed.ok && !failed.errors.empty(),
          "invalid external descriptor rejects incremental recompile");
    Check(graph.Desc().terminal == savedTerminal &&
              graph.ReferenceValues().size() == 2 &&
              graph.ReferenceValues()[0].identity == savedReferenceIdentity &&
              graph.MapValues().size() == 2 &&
              graph.MapValues()[0].identity == savedMapIdentity,
          "failed external recompile preserves prior graph values");
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 2).diagnostics.HasErrors(),
          "prior graph remains executable after failed external recompile");
}

UsdGenGraphDesc ReferenceChainFixture()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/referenceChain");
    desc.terminal = SdfPath("/referenceChain/width");

    UsdGenCurveSetDesc reference;
    reference.path = SdfPath("/referenceChain/guides");
    reference.role = UsdGenRole::Reference;
    reference.curveVertexCounts = VtIntArray{2, 2};
    reference.points = VtVec3fArray{
        GfVec3f(1, 2, 3), GfVec3f(1, 3, 3),
        GfVec3f(4, 5, 6), GfVec3f(4, 6, 6)};
    reference.rest = VtVec3fArray{
        GfVec3f(11, 12, 13), GfVec3f(11, 13, 13),
        GfVec3f(14, 15, 16), GfVec3f(14, 16, 16)};
    reference.widths = VtFloatArray{0.1f, 0.2f, 0.3f, 0.4f};
    reference.curveId = VtArray<uint64_t>{101, 202};
    reference.skinPrim = VtIntArray{7, 8};
    reference.skinPrimUv = VtVec2fArray{GfVec2f(0.1f, 0.2f), GfVec2f(0.3f, 0.4f)};
    GfMatrix4d frame(1.0);
    frame.SetRow3(0, GfVec3d(1, 0, 0));
    frame.SetRow3(1, GfVec3d(0, 1, 0));
    frame.SetRow3(2, GfVec3d(0, 0, 1));
    reference.rootFrame = VtMatrix4dArray{frame, frame};
    UsdGenAuthoredPlaneDesc point;
    point.name = TfToken("referenceWeight");
    point.type = UsdGenAuthoredPlaneType::Float32;
    point.domain = UsdGenAuthoredPlaneDomain::Point;
    point.floatValues = VtFloatArray{1, 2, 3, 4};
    reference.authoredPlanes.push_back(point);
    UsdGenAuthoredPlaneDesc primitive;
    primitive.name = TfToken("referenceClass");
    primitive.type = UsdGenAuthoredPlaneType::Int32;
    primitive.domain = UsdGenAuthoredPlaneDomain::Primitive;
    primitive.intValues = VtIntArray{11, 22};
    reference.authoredPlanes.push_back(primitive);
    reference.curveGeneration = 31;
    desc.curveSets.push_back(std::move(reference));

    UsdGenNodeDesc width;
    width.path = desc.terminal;
    width.type = TfToken("UsdGenWidth");
    width.inputs = {SdfPath("/referenceChain/reference")};
    width.params.push_back({TfToken("width"), VtValue(0.5f), false});
    UsdGenNodeDesc source;
    source.path = SdfPath("/referenceChain/reference");
    source.type = TfToken("UsdGenReferenceSource");
    source.references = {desc.curveSets.front().path};
    // Namespace order puts the consumer first; dependency order must still
    // capture the reference producer before Width.
    desc.nodes = {width, source};
    return desc;
}

void ValidateReferenceProducerChain()
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenGraphDesc desc = ReferenceChainFixture();
    UsdGenCompileResult const compiled = compiler.Compile(desc, &graph);
    Check(compiled.ok, "reference source chain compiles");
    if (!compiled.ok) return;

    UsdGenScheduler scheduler(1);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    UsdGenRunResult const run = scheduler.Run(graph, context, 1);
    Check(!run.diagnostics.HasErrors(), "reference source chain executes");
    UsdGenNodeId const sourceId = graph.NodeIdForPath(SdfPath("/referenceChain/reference"));
    UsdGenNodeId const widthId = graph.NodeIdForPath(desc.terminal);
    UsdGenCurveBuffer const source = NodeBuffer(graph, sourceId);
    UsdGenCurveBuffer const output = NodeBuffer(graph, widthId);
    Check(source.totalCurves == 2 && source.totalCvs == 4 &&
              source.px == VtFloatArray{1, 1, 4, 4} &&
              source.py == VtFloatArray{2, 3, 5, 6} &&
              source.rest == VtVec3fArray{
                  GfVec3f(11, 12, 13), GfVec3f(11, 13, 13),
                  GfVec3f(14, 15, 16), GfVec3f(14, 16, 16)} &&
              source.curveId == VtArray<uint64_t>{101, 202} &&
              source.rootPrim == VtIntArray{7, 8} &&
              source.rootUV == VtVec2fArray{GfVec2f(0.1f, 0.2f), GfVec2f(0.3f, 0.4f)} &&
              source.rootT == VtVec3fArray{GfVec3f(1, 0, 0), GfVec3f(1, 0, 0)} &&
              source.rootB == VtVec3fArray{GfVec3f(0, 1, 0), GfVec3f(0, 1, 0)} &&
              source.rootN == VtVec3fArray{GfVec3f(0, 0, 1), GfVec3f(0, 0, 1)},
          "reference producer preserves topology, points, IDs, and roots");
    Check(source.hairT == VtFloatArray{0, 1, 0, 1} &&
              source.extraCv.size() == 1 && source.extraCurve.size() == 1 &&
              source.extraCv[0].f == VtFloatArray{1, 2, 3, 4} &&
              source.extraCurve[0].i == VtIntArray{11, 22},
          "reference producer preserves hairT and named planes");
    Check(output.width == VtFloatArray{0.5f, 0.5f, 0.5f, 0.5f} &&
              output.px == source.px && output.extraCv[0].f == source.extraCv[0].f,
          "downstream Width consumes reference-produced geometry");

    std::shared_ptr<const UsdGenReferenceSet> const oldValue =
        graph.ReferenceValues()[0].value;
    float const oldPoint = oldValue->buffer.px[0];
    GfVec3f const oldRest = oldValue->buffer.rest[0];
    VtFloatArray const oldPlane = oldValue->buffer.extraCv[0].f;
    desc.curveSets[0].curveGeneration = 32;
    desc.curveSets[0].points[0] = GfVec3f(9, 8, 7);
    desc.curveSets[0].rest[0] = GfVec3f(19, 18, 17);
    Check(compiler.Recompile(desc, &graph).ok,
          "reference generation edit recompiles producer chain");
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 2).diagnostics.HasErrors(),
          "reference generation edit executes producer chain");
    UsdGenNodeId const newSourceId = graph.NodeIdForPath(SdfPath("/referenceChain/reference"));
    Check(oldValue->buffer.px[0] == oldPoint && oldValue->buffer.rest[0] == oldRest &&
              oldValue->buffer.extraCv[0].f == oldPlane &&
              NodeBuffer(graph, newSourceId).px[0] == 9.0f &&
              NodeBuffer(graph, newSourceId).rest[0] == GfVec3f(19, 18, 17),
          "reference generation recapture does not mutate prior immutable COW points/rest");

    UsdGenGraphDesc invalid = desc;
    invalid.nodes[1].references = {SdfPath("/referenceChain/missing")};
    UsdGenCompileResult const failed = compiler.Recompile(invalid, &graph);
    Check(!failed.ok && !failed.errors.empty() &&
              failed.errors.front().find("unresolved reference value") != std::string::npos,
          "reference source rejects an unresolved input deterministically");
    Check(NodeBuffer(graph, newSourceId).px[0] == 9.0f,
          "failed reference recompile preserves last-good producer output");

    UsdGenGraphDesc missingSlot = desc;
    missingSlot.nodes[1].references.clear();
    UsdGenGraph rejected;
    UsdGenCompileResult const missing = compiler.Compile(missingSlot, &rejected);
    Check(!missing.ok && !missing.errors.empty() &&
              missing.errors.front().find("has 0 resolved reference values") != std::string::npos,
          "reference source requires exactly one resolved input");
}

}  // namespace

int main()
{
    Check(UsdGenRegisterStatic(&RegisterContractProbes)&&UsdGenRegisterStatic(&RegisterProbes)&&
          UsdGenOpRegistry::Get().Find(TfToken("UsdGenTestReferenceProbe"))!=UsdGenOpId::Invalid,
          "registers reference-lane probe");
    Check(UsdGenOpRegistry::Get().Find(TfToken("UsdGenTestCurveProbe"))!=UsdGenOpId::Invalid,
          "registers curve-lane probe");
    ValidateRegistryContractReuse();
    RunReferenceMapLowering();
    ValidateReferenceProducerChain();
    ValidateFailures();
    ValidateMapIdentityAndRecompileTransaction();
    std::printf(gFailures ? "testUsdGenCpuReferenceMap: FAILED (%d)\n"
                          : "testUsdGenCpuReferenceMap: PASS\n", gFailures);
    return gFailures ? 1 : 0;
}
