#include "usdGen/cudaExecution.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>

using namespace usdGen;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {
class TopologyNamedPlaneProbe final : public UsdGenOp {
public:
    explicit TopologyNamedPlaneProbe(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    bool Bind(UsdGenParamView const&, UsdGenDiagnostics*) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const&) const override { return {1, 2}; }
    bool Capture(UsdGenCaptureContext const&, UsdGenOpInputs const &,
                 UsdGenCapture*, UsdGenDiagnostics*) override { return true; }
    void Evaluate(UsdGenEvalContext const&, UsdGenCapture const&,
                  UsdGenChunkView const &) const override {}
};

class ValueNamedPlaneProbe final : public UsdGenOp {
public:
    explicit ValueNamedPlaneProbe(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    bool Bind(UsdGenParamView const&, UsdGenDiagnostics*) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const&) const override { return {1, 3}; }
    bool Capture(UsdGenCaptureContext const&, UsdGenOpInputs const &,
                 UsdGenCapture*, UsdGenDiagnostics*) override { return true; }
    void Evaluate(UsdGenEvalContext const&, UsdGenCapture const&,
                  UsdGenChunkView const &) const override {}
};

void RegisterNamedPlaneProbes(UsdGenRegistrar &r)
{
    static UsdGenPlaneDecl const kIn[] = {
        {"guideIndex", UsdGenPlaneRate::Uniform, UsdGenPlaneScalar::Int, 3}};
    static UsdGenPlaneDecl const kOut[] = {
        {"guideWeight", UsdGenPlaneRate::Uniform, UsdGenPlaneScalar::Float, 3}};
    UsdGenOpDef d;
    d.name = "UsdGenTestTopologyNamedPlane";
    d.displayName = "TestTopologyNamedPlane";
    d.create = [](UsdGenOpRecord const &rec, void *) -> std::unique_ptr<UsdGenOp> {
        return std::make_unique<TopologyNamedPlaneProbe>(rec);
    };
    d.countChanges = UsdGenCountsAtLevel(1);
    d.inputPlanes = UsdGenSpan(kIn);
    r.RegisterOp(d);
    UsdGenOpDef v;
    v.name = "UsdGenTestValueNamedPlane";
    v.displayName = "TestValueNamedPlane";
    v.create = [](UsdGenOpRecord const &rec, void *) -> std::unique_ptr<UsdGenOp> {
        return std::make_unique<ValueNamedPlaneProbe>(rec);
    };
    v.outputPlanes = UsdGenSpan(kOut);
    r.RegisterOp(v);
}
}

static UsdGenGraphDesc MakeDesc() {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom/Hair");
    desc.executionBackend = UsdGenExecutionBackend::Cuda;

    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Scalp");
    surface.restPoints = {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {1,1,1}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {3,3,3};
    surface.faceVertexIndices = {0,1,2, 0,1,3, 1,2,4};
    desc.surfaces.push_back(surface);

    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Curves");
    curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {2};
    curves.points = {{.2f,.2f,0}, {.2f,.4f,0}};
    curves.rest = curves.points;
    curves.curveId = {42};
    curves.skinPrim = {0};
    curves.skinPrimUv = {{.2f,.2f}};
    desc.curveSets.push_back(curves);

    // Deliberately non-lexical paths: semantic ordinals must follow the
    // hierarchy-derived authored vector, not paths or operator families.
    UsdGenNodeDesc source;
    source.path = SdfPath("/Groom/Hair/Ops/ZSource");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.surfaces = {surface.path};

    UsdGenNodeDesc width;
    width.path = SdfPath("/Groom/Hair/Ops/AWidth");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {source.path};
    width.params.push_back({TfToken("width"), VtValue(.03f), false});

    UsdGenNodeDesc length;
    length.path = SdfPath("/Groom/Hair/Ops/MLength");
    length.type = TfToken("UsdGenLength");
    length.inputs = {width.path};
    length.params.push_back({TfToken("length:value"), VtValue(.5f), false});

    UsdGenNodeDesc deform;
    deform.path = SdfPath("/Groom/Hair/Ops/BDeform");
    deform.type = TfToken("UsdGenDeform");
    deform.inputs = {length.path};
    deform.surfaces = {surface.path};
    deform.params.push_back({TfToken("rbfSamples"), VtValue(5), false});

    desc.nodes = {source, width, length, deform};
    desc.terminal = deform.path;
    return desc;
}

// Compiler-only Scatter->Grow value DAG. Runtime payload coverage belongs to
// the dedicated ScatterGrowValueDag fixture; this one pins logical ownership
// and cross-origin WidthBlend lowering.
static UsdGenGraphDesc MakeScatterGrowValueDagDesc() {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom/Generated");
    desc.executionBackend = UsdGenExecutionBackend::Cuda;
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Scalp");
    surface.restPoints = {{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}};
    surface.points = surface.restPoints;
    surface.uv = {{0,0}, {1,0}, {1,1}, {0,1}};
    surface.faceVertexCounts = {4}; surface.faceVertexIndices = {0,1,2,3};
    desc.surfaces = {surface};
    UsdGenNodeDesc scatter;
    scatter.path = SdfPath("/Ops/Scatter"); scatter.type = TfToken("UsdGenScatter");
    scatter.seed = 41; scatter.surfaces = {surface.path};
    scatter.params = {{TfToken("density"), VtValue(300.0f), false}};
    UsdGenNodeDesc grow;
    grow.path = SdfPath("/Ops/Grow"); grow.type = TfToken("UsdGenGrow");
    grow.seed = 19; grow.inputs = {scatter.path};
    grow.params = {{TfToken("segments"), VtValue(5), false},
                   {TfToken("length"), VtValue(2.0f), false},
                   {TfToken("lengthRandom"), VtValue(GfVec2f(.5f,1.5f)), false}};
    UsdGenNodeDesc length;
    length.path = SdfPath("/Ops/Length"); length.type = TfToken("UsdGenLength");
    length.inputs = {grow.path};
    length.params = {{TfToken("length:mode"), VtValue(TfToken("scale")), false},
                     {TfToken("length:value"), VtValue(1.f), false}};
    UsdGenNodeDesc noise;
    noise.path = SdfPath("/Ops/Noise"); noise.type = TfToken("UsdGenNoise");
    noise.inputs = {grow.path};
    UsdGenNodeDesc left;
    left.path = SdfPath("/Ops/Left"); left.type = TfToken("UsdGenWidth");
    left.inputs = {length.path}; left.params = {{TfToken("width"), VtValue(.2f), false}};
    UsdGenNodeDesc right;
    right.path = SdfPath("/Ops/Right"); right.type = TfToken("UsdGenWidth");
    right.inputs = {noise.path}; right.params = {{TfToken("width"), VtValue(.8f), false}};
    UsdGenNodeDesc blend;
    blend.path = SdfPath("/Ops/Blend"); blend.type = TfToken("UsdGenWidthBlend");
    blend.inputs = {left.path, right.path};
    blend.params.push_back({TfToken("widthBlend:weight"), VtValue(.25f), false});
    desc.nodes = {blend, right, noise, scatter, left, grow, length};
    desc.terminal = blend.path;
    return desc;
}

static bool DiagnosticIs(UsdGenDiagnostics const& diagnostics, char const* expected) {
    return diagnostics.errors.size() == 1 && diagnostics.warnings.empty() &&
        diagnostics.errors.front() == expected;
}

static bool DependenciesAre(UsdGenExecutionTaskMetadata const& task,
                            std::initializer_list<uint32_t> expected) {
    return task.dependencies == std::vector<uint32_t>(expected);
}

static bool DependencyHas(UsdGenExecutionTaskMetadata const& task,
                          uint32_t predecessor, uint8_t provenance) {
    for (auto const& edge : task.dependencyProvenance)
        if (edge.predecessor == predecessor &&
            (edge.provenance & provenance) == provenance)
            return true;
    return false;
}

static bool ResourceIs(UsdGenExecutionTaskMetadata const& task, size_t index,
                       UsdGenExecutionDataKind kind,
                       UsdGenExecutionResourceAccess access,
                       uint32_t producer) {
    if (index >= task.resources.size()) return false;
    auto const& resource = task.resources[index];
    return resource.resource == kind && resource.access == access &&
        resource.producerTask == producer;
}

static uint64_t ExpectedSourcePayloadBytes(UsdGenGraphDesc const& desc) {
    auto const& curves = desc.curveSets.front();
    bool const roots = !desc.nodes.front().surfaces.empty() ||
        !curves.skinPrim.empty() || !curves.skinPrimUv.empty();
    return static_cast<uint64_t>(curves.points.size()) * 32u +
        static_cast<uint64_t>(curves.curveVertexCounts.size()) *
            (roots ? 24u : 12u) + sizeof(uint32_t);
}

static void AddScalarExpression(UsdGenGraphDesc* desc, UsdGenNodeDesc* node,
                                char const* path, char const* source,
                                char const* destination, expr::Domain domain,
                                TfToken nativeType, expr::ScalarType scalar,
                                VtValue literal) {
    expr::ValueShape shape{scalar, 1, 1, 1, 1, false};
    UsdGenExpressionDesc expression;
    expression.path = SdfPath(path);
    expression.source = source;
    expression.outputs.push_back({TfToken("result"), nativeType, shape});
    desc->expressions.push_back(std::move(expression));
    UsdGenExpressionBinding binding;
    binding.expression = SdfPath(path);
    binding.destination = TfToken(destination);
    binding.domain = domain;
    binding.nativeType = nativeType;
    binding.destinationShape = shape;
    binding.literal = std::move(literal);
    node->expressionBindings.push_back(std::move(binding));
}

static bool ValueFlowIsConsistent(UsdGenExecutionPlanMetadata const& metadata,
                                  UsdGenExecutionTaskMetadata const& task,
                                  UsdGenExecutionResourceUse const& use) {
    bool ok = true;
    if (use.access != UsdGenExecutionResourceAccess::Write) {
        auto const* input = metadata.FindValue(use.inputValue);
        ok &= input && input->resource == use.resource &&
            input->producerTask == use.producerTask;
    } else {
        ok &= use.inputValue == UINT32_MAX;
    }
    if (use.access != UsdGenExecutionResourceAccess::Read) {
        auto const* output = metadata.FindValue(use.outputValue);
        ok &= output && output->resource == use.resource &&
            output->producerTask == task.id;
        if (use.access == UsdGenExecutionResourceAccess::ReadWrite) {
            auto const* input = metadata.FindValue(use.inputValue);
            ok &= input && output && output->version == input->version + 1;
        }
    } else {
        ok &= use.outputValue == UINT32_MAX;
    }
    return ok;
}

int main() {
    // Before the first compile: registries freeze there.
    CHECK(UsdGenRegisterStatic(&RegisterNamedPlaneProbes));
    CHECK(UsdGenOpRegistry::Get().Find(TfToken("UsdGenTestTopologyNamedPlane")) != UsdGenOpId::Invalid);
    CHECK(UsdGenOpRegistry::Get().Find(TfToken("UsdGenTestValueNamedPlane")) != UsdGenOpId::Invalid);
    {
        auto sourceNoise = MakeDesc();
        sourceNoise.nodes.resize(1);
        UsdGenNodeDesc noise;
        noise.path = SdfPath("/Noise"); noise.type = TfToken("UsdGenNoise");
        noise.inputs = {sourceNoise.nodes.front().path};
        sourceNoise.nodes.push_back(noise); sourceNoise.terminal = noise.path;
        UsdGenDiagnostics admission;
        CHECK(ValidateCudaGraph(sourceNoise,&admission));
        auto rejects = [&](UsdGenGraphDesc const& invalid) {
            UsdGenDiagnostics errors;
            return !ValidateCudaGraph(invalid,&errors) && errors.HasErrors();
        };
        for (int variant = 0; variant != 7; ++variant) {
            auto invalid = sourceNoise;
            auto& curves = invalid.curveSets[0];
            if (variant == 0) curves.rest.clear();
            if (variant == 1) curves.restFromCurrentPoints = true;
            if (variant == 2) curves.rest[0][0] = std::numeric_limits<float>::quiet_NaN();
            if (variant == 3) curves.points.pop_back();
            if (variant == 4) invalid.surfaces[0].restFromCurrentPoints = true;
            if (variant == 5) invalid.surfaces[0].faceVertexIndices[0] = 999;
            if (variant == 6) curves.rootFrame = {GfMatrix4d(2.0)};
            CHECK(rejects(invalid));
        }
        auto posed = sourceNoise;
        auto repairable = sourceNoise;
        repairable.curveSets[0].skinPrim.clear();
        repairable.curveSets[0].skinPrimUv.clear();
        CHECK(ValidateCudaGraph(repairable,&admission));
        repairable = sourceNoise;
        repairable.curveSets[0].skinPrimUv[0] = GfVec2f(.8f,.8f);
        CHECK(ValidateCudaGraph(repairable,&admission));
        posed.nodes[0].params.push_back({TfToken("useRest"),VtValue(false),false});
        CHECK(ValidateCudaGraph(posed,&admission));
        posed.curveSets[0].rest.clear();
        CHECK(rejects(posed));
        auto explicitFrames = sourceNoise;
        explicitFrames.curveSets[0].rootFrame = {GfMatrix4d(1.0)};
        explicitFrames.surfaces[0].restPoints.clear();
        CHECK(ValidateCudaGraph(explicitFrames,&admission));
        auto degenerate = sourceNoise;
        degenerate.surfaces[0].restPoints[1] = degenerate.surfaces[0].restPoints[0];
        CHECK(ValidateCudaGraph(degenerate,&admission));
        CHECK(sourceNoise.curveSets[0].rootFrame.empty() &&
              sourceNoise.curveSets[0].curveId == VtArray<uint64_t>({42}));
    }
    // Noise owns a float3 output/staging plane and retains one 9-float root
    // frame per source curve. Keep these exact fields visible in metadata so
    // the plan cannot regress to charging only a scalar width plane.
    {
        auto noiseDesc = MakeDesc();
        noiseDesc.nodes.resize(1);
        UsdGenNodeDesc noise;
        noise.path = SdfPath("/Noise");
        noise.type = TfToken("UsdGenNoise");
        noise.inputs = {noiseDesc.nodes.front().path};
        noiseDesc.nodes.push_back(noise);
        noiseDesc.terminal = noise.path;
        UsdGenDiagnostics noiseDiagnostics;
        auto noisePlan = CompileCudaGraph(noiseDesc, &noiseDiagnostics);
        CHECK(noisePlan && !noiseDiagnostics.HasErrors());
        auto noiseMetadata = GetCudaExecutionPlanMetadata(*noisePlan);
        CHECK(noiseMetadata && noiseMetadata->Tasks().size() == 3);
        auto const& noiseSource = noiseMetadata->Tasks()[0];
        auto const& noiseTask = noiseMetadata->Tasks()[1];
        CHECK(noiseMetadata->Shape() == UsdGenExecutionPlanShape::SourceRootedUnaryDag &&
              noiseTask.exclusiveWorkspace);
        CHECK(std::any_of(noiseTask.resources.begin(),noiseTask.resources.end(),[](auto const& use) {
            return use.resource == UsdGenExecutionDataKind::Widths &&
                   use.access == UsdGenExecutionResourceAccess::Read;
        }));
        uint64_t const pointCount = noiseDesc.curveSets.front().points.size();
        uint64_t const expectedFrameBytes = 9 * sizeof(float);
        uint64_t const expectedFloat3Bytes = pointCount * 3 * sizeof(float);
        CHECK(noiseSource.estimate.steadyBytes == expectedFrameBytes &&
              noiseTask.estimate.retainedOutputBytes == expectedFloat3Bytes &&
              noiseTask.estimate.producerRetentionBytes ==
                  ExpectedSourcePayloadBytes(noiseDesc) + expectedFrameBytes &&
              noiseTask.estimate.steadyBytes == 0 &&
              noiseTask.estimate.scratchPeakBytes == expectedFloat3Bytes +
                  257 * sizeof(float) + 2 * sizeof(int) + sizeof(int));
    }
    auto const& matrix = GetCudaExecutionCapabilityMatrix();
    CHECK(matrix.Backend() == "cuda" && matrix.Version() == 35 && matrix.Available());
    CHECK(matrix.Capabilities().size() == 9);
    struct ExpectedCapability { char const* type; uint32_t version; uint32_t flags; };
    ExpectedCapability const expectedCapabilities[] = {
        {"UsdGenScatter", 1, UsdGenCapabilityChangesTopology |
             UsdGenCapabilityTopologyBarrier | UsdGenCapabilityExclusiveWorkspace},
        {"UsdGenGrow", 13, UsdGenCapabilityChangesTopology |
             UsdGenCapabilityTopologyBarrier | UsdGenCapabilityExclusiveWorkspace},
        {"UsdGenCurveSource", 2, UsdGenCapabilityExpressions | UsdGenCapabilityChangesTopology |
             UsdGenCapabilityTopologyBarrier | UsdGenCapabilityExclusiveWorkspace},
        {"UsdGenWidth", 6, UsdGenCapabilityExpressions |
             UsdGenCapabilityCopyOnWriteWrites},
        {"UsdGenLength", 7, UsdGenCapabilityExpressions | UsdGenCapabilityChangesTopology |
             UsdGenCapabilityTopologyBarrier | UsdGenCapabilityExclusiveWorkspace},
        {"UsdGenNoise", 7, UsdGenCapabilityExpressions |
             UsdGenCapabilityCopyOnWriteWrites},
        {"UsdGenDeform", 1, UsdGenCapabilityExpressions | UsdGenCapabilityExclusiveWorkspace |
             UsdGenCapabilitySurfaceInput | UsdGenCapabilityTransactionalPublication},
        {"UsdGenReferenceSource", 2, UsdGenCapabilityChangesTopology |
             UsdGenCapabilityTopologyBarrier | UsdGenCapabilityExclusiveWorkspace},
        {"UsdGenWidthBlend", 6, UsdGenCapabilityCopyOnWriteWrites}
    };
    for (size_t i = 0; i < matrix.Capabilities().size(); ++i) {
        auto const& row = matrix.Capabilities()[i];
        CHECK(row.type == TfToken(expectedCapabilities[i].type));
        CHECK(row.capabilityVersion == expectedCapabilities[i].version &&
              row.minimumAlgorithmVersion == 0 &&
              row.maximumAlgorithmVersion ==
                  (row.type == TfToken("UsdGenWidthBlend") ||
                   row.type == TfToken("UsdGenScatter") ||
                   row.type == TfToken("UsdGenGrow") ? 0 : 1) &&
              row.flags == expectedCapabilities[i].flags);
        CHECK(matrix.Find(row.type) == &row);
    }
    CHECK(matrix.Find(TfToken("UsdGenNoise")) != nullptr);

    auto desc = MakeDesc();
    UsdGenDiagnostics diagnostics;
    auto plan = CompileCudaGraph(desc, &diagnostics);
    if (!plan || diagnostics.HasErrors()) for (auto const& error : diagnostics.errors)
        std::fprintf(stderr, "compile diagnostic: %s\n", error.c_str());
    CHECK(plan && !diagnostics.HasErrors());
    {
        // Stage/Hydra descriptors carry the schema's empty curveWrap-only
        // default on surface RBF nodes. It must not disable CUDA deformation.
        auto defaults = desc;
        defaults.nodes.back().params.push_back(
            {TfToken("guideRegions"), VtValue(VtIntArray{}), false});
        UsdGenDiagnostics defaultDiagnostics;
        CHECK(CompileCudaGraph(defaults, &defaultDiagnostics) &&
              !defaultDiagnostics.HasErrors());
        for (VtValue const& invalid : {VtValue(VtIntArray{7}),
                                      VtValue(VtFloatArray{}), VtValue(7)}) {
            auto malformed = defaults;
            malformed.nodes.back().params.back().value = invalid;
            UsdGenDiagnostics rejected;
            CHECK(!CompileCudaGraph(malformed, &rejected) && rejected.HasErrors());
        }
        defaults.nodes.back().mode = TfToken("curveWrap");
        UsdGenDiagnostics modeDiagnostics;
        CHECK(!CompileCudaGraph(defaults, &modeDiagnostics) &&
              modeDiagnostics.HasErrors());
    }
    auto metadata = GetCudaExecutionPlanMetadata(*plan);
    CHECK(metadata && metadata->Backend() == "cuda" &&
          metadata->CapabilityVersion() == matrix.Version() &&
          metadata->Shape() == UsdGenExecutionPlanShape::LinearAuthoredChain);
    CHECK(metadata->Operators().size() == 4 && metadata->Tasks().size() == 5 &&
          metadata->TerminalTask() == 4);
    CHECK(metadata->Values().size() == 17 && metadata->FindValue(99) == nullptr);
    for (auto const& task : metadata->Tasks())
        CHECK(!task.estimate.timeAvailable &&
              task.estimate.estimatedMicroseconds == 0);
    if (metadata->MemoryEstimate().memoryAvailable)
        CHECK(metadata->MemoryEstimate().conservativeUpperBound &&
              metadata->MemoryEstimate().immutableSharedInputBytes ==
                  ExpectedSourcePayloadBytes(desc));

    for (uint32_t i = 0; i != 4; ++i) {
        auto const& op = metadata->Operators()[i];
        CHECK(op.path == desc.nodes[i].path && op.type == desc.nodes[i].type);
        auto const* capability = matrix.Find(op.type);
        CHECK(op.authoredOrderKey == i &&
              capability && op.capabilityVersion == capability->capabilityVersion &&
              op.flags == capability->flags &&
              op.status == UsdGenExecutionCapabilityStatus::Supported);
        auto const* task = metadata->FindTask(i);
        CHECK(task && task == metadata->FindSemanticTask(i));
        CHECK(task->id == i && task->semanticNode == i && task->authoredOrderKey == i &&
              task->path == desc.nodes[i].path && task->type == desc.nodes[i].type);
        CHECK(task->kind == (i == 0 ? UsdGenExecutionTaskKind::Source :
                                     UsdGenExecutionTaskKind::Operator));
        CHECK(i == 0 ? DependenciesAre(*task, {}) :
              i == 2 ? DependenciesAre(*task, {0, 1}) :
                       DependenciesAre(*task, {i - 1}));
        // The byte fields may be populated independently of the admission
        // summary. Timing is intentionally unknown until a calibrated model
        // exists, so zero must never be interpreted as known zero cost.
        CHECK(!task->estimate.timeAvailable &&
              task->estimate.estimatedMicroseconds == 0);
        CHECK(task->cancellation == UsdGenExecutionCancellationPoint::BeforeTaskOnly);
        // Width capability v2 can be isolated for DAG lowering, but this
        // compatibility linear plan still mutates the primary geometry lane.
        CHECK(task->exclusiveWorkspace);
        CHECK(task->resourceHazards.size() == 1 &&
              task->resourceHazards.front().resource ==
                  UsdGenExecutionDataKind::ExecutionWorkspace &&
              task->resourceHazards.front().identity == 0 &&
              task->resourceHazards.front().access ==
                  UsdGenExecutionResourceHazardAccess::ReadWrite);
    }
    auto const& sourceEstimate = metadata->Tasks()[0].estimate;
    auto const& widthEstimate = metadata->Tasks()[1].estimate;
    auto const sourcePayload = ExpectedSourcePayloadBytes(desc);
    CHECK(sourceEstimate.memoryAvailable &&
          sourceEstimate.memoryConservativeUpperBound &&
          sourceEstimate.steadyBytes == 0 &&
          sourceEstimate.retainedOutputBytes == sourcePayload &&
          sourceEstimate.producerRetentionBytes == 0 &&
          sourceEstimate.scratchPeakBytes == 0);
    CHECK(widthEstimate.memoryAvailable &&
          widthEstimate.memoryConservativeUpperBound &&
          widthEstimate.steadyBytes == 0 &&
          widthEstimate.retainedOutputBytes == 2 * sizeof(float) &&
          widthEstimate.producerRetentionBytes == sourcePayload &&
          widthEstimate.scratchPeakBytes ==
              2 * sizeof(float) + 257 * sizeof(float) + 2 * sizeof(int));
    auto const& lengthEstimate = metadata->Tasks()[2].estimate;
    CHECK(!lengthEstimate.memoryAvailable &&
          lengthEstimate.retainedOutputBytes == sourcePayload &&
          lengthEstimate.producerRetentionBytes == sourcePayload &&
          lengthEstimate.scratchPeakBytes ==
              2 * 3 * sizeof(float) * 2 + 18 + 10 * sizeof(int));
    auto const& deformEstimate = metadata->Tasks()[3].estimate;
    CHECK(!deformEstimate.memoryAvailable &&
          deformEstimate.retainedOutputBytes == 2 * 3 * sizeof(float) &&
          deformEstimate.producerRetentionBytes == sourcePayload &&
          deformEstimate.scratchPeakBytes ==
              2 * 3 * sizeof(float) * 2 + sizeof(int));
    auto const& publicationEstimate = metadata->Tasks()[4].estimate;
    CHECK(publicationEstimate.memoryAvailable &&
          publicationEstimate.retainedOutputBytes == 0 &&
          publicationEstimate.producerRetentionBytes == 0 &&
          publicationEstimate.scratchPeakBytes == 32 + 116);
    CHECK(!metadata->MemoryEstimate().runtimeRefinementAvailable);
    CHECK(metadata->FindTask(99) == nullptr && metadata->FindSemanticTask(99) == nullptr);
    CHECK(metadata->Tasks()[0].topologyBarrier && !metadata->Tasks()[1].topologyBarrier &&
          metadata->Tasks()[2].topologyBarrier && !metadata->Tasks()[3].topologyBarrier);

    auto const external = UINT32_MAX;
    auto const& source = metadata->Tasks()[0];
    CHECK(source.resources.size() == 7);
    CHECK(ResourceIs(source,0,UsdGenExecutionDataKind::ElementGeometry,UsdGenExecutionResourceAccess::Write,external));
    CHECK(ResourceIs(source,1,UsdGenExecutionDataKind::ElementTopology,UsdGenExecutionResourceAccess::Write,external));
    CHECK(ResourceIs(source,2,UsdGenExecutionDataKind::StableIds,UsdGenExecutionResourceAccess::Write,external));
    CHECK(ResourceIs(source,3,UsdGenExecutionDataKind::RootBindings,UsdGenExecutionResourceAccess::Write,external));
    CHECK(ResourceIs(source,4,UsdGenExecutionDataKind::Widths,UsdGenExecutionResourceAccess::Write,external));
    CHECK(ResourceIs(source,5,UsdGenExecutionDataKind::NamedChannels,UsdGenExecutionResourceAccess::Write,external));
    CHECK(ResourceIs(source,6,UsdGenExecutionDataKind::ParameterValues,UsdGenExecutionResourceAccess::Read,external));

    auto const& width = metadata->Tasks()[1];
    CHECK(width.resources.size() == 6);
    CHECK(ResourceIs(width,4,UsdGenExecutionDataKind::StableIds,UsdGenExecutionResourceAccess::Read,0));
    CHECK(ResourceIs(width,5,UsdGenExecutionDataKind::RootBindings,UsdGenExecutionResourceAccess::Read,0));
    CHECK(ResourceIs(width,0,UsdGenExecutionDataKind::ElementGeometry,UsdGenExecutionResourceAccess::Read,0));
    CHECK(ResourceIs(width,1,UsdGenExecutionDataKind::ElementTopology,UsdGenExecutionResourceAccess::Read,0));
    CHECK(ResourceIs(width,2,UsdGenExecutionDataKind::ParameterValues,UsdGenExecutionResourceAccess::Read,external));
    CHECK(ResourceIs(width,3,UsdGenExecutionDataKind::Widths,UsdGenExecutionResourceAccess::ReadWrite,0));

    auto const& length = metadata->Tasks()[2];
    CHECK(length.resources.size() == 7);
    CHECK(ResourceIs(length,0,UsdGenExecutionDataKind::ElementGeometry,UsdGenExecutionResourceAccess::ReadWrite,0));
    CHECK(ResourceIs(length,1,UsdGenExecutionDataKind::ElementTopology,UsdGenExecutionResourceAccess::ReadWrite,0));
    CHECK(ResourceIs(length,2,UsdGenExecutionDataKind::StableIds,UsdGenExecutionResourceAccess::ReadWrite,0));
    CHECK(ResourceIs(length,3,UsdGenExecutionDataKind::RootBindings,UsdGenExecutionResourceAccess::ReadWrite,0));
    CHECK(ResourceIs(length,4,UsdGenExecutionDataKind::Widths,UsdGenExecutionResourceAccess::ReadWrite,1));
    CHECK(ResourceIs(length,5,UsdGenExecutionDataKind::NamedChannels,UsdGenExecutionResourceAccess::ReadWrite,0));
    CHECK(ResourceIs(length,6,UsdGenExecutionDataKind::ParameterValues,UsdGenExecutionResourceAccess::Read,external));

    auto const& deform = metadata->Tasks()[3];
    CHECK(deform.resources.size() == 8);
    CHECK(ResourceIs(deform,0,UsdGenExecutionDataKind::ElementGeometry,UsdGenExecutionResourceAccess::ReadWrite,2));
    CHECK(ResourceIs(deform,1,UsdGenExecutionDataKind::ElementTopology,UsdGenExecutionResourceAccess::Read,2));
    CHECK(ResourceIs(deform,2,UsdGenExecutionDataKind::StableIds,UsdGenExecutionResourceAccess::Read,2));
    CHECK(ResourceIs(deform,3,UsdGenExecutionDataKind::RootBindings,UsdGenExecutionResourceAccess::Read,2));
    CHECK(ResourceIs(deform,4,UsdGenExecutionDataKind::Widths,UsdGenExecutionResourceAccess::Read,2));
    CHECK(ResourceIs(deform,5,UsdGenExecutionDataKind::NamedChannels,UsdGenExecutionResourceAccess::Read,2));
    CHECK(ResourceIs(deform,6,UsdGenExecutionDataKind::SurfaceGeometry,UsdGenExecutionResourceAccess::Read,external));
    CHECK(ResourceIs(deform,7,UsdGenExecutionDataKind::ParameterValues,UsdGenExecutionResourceAccess::Read,external));

    auto const& publication = metadata->Tasks()[4];
    CHECK(publication.id == 4 && publication.semanticNode == UINT32_MAX &&
          publication.authoredOrderKey == 4 &&
          publication.kind == UsdGenExecutionTaskKind::Publication &&
          publication.path == desc.terminal && publication.type == TfToken("CudaPublication") &&
          DependenciesAre(publication,{0,1,2,3}) && publication.topologyBarrier &&
          publication.exclusiveWorkspace && !publication.estimate.timeAvailable &&
          publication.estimate.estimatedMicroseconds == 0 &&
          publication.cancellation == UsdGenExecutionCancellationPoint::BeforeTaskOnly);
    CHECK(publication.declaredDependencies.size() == 4 &&
          publication.dependencyProvenance.size() == 4);
    for (uint32_t i = 0; i != 4; ++i) {
        CHECK(publication.declaredDependencies[i].predecessor == i &&
              publication.declaredDependencies[i].provenance ==
                  UsdGenExecutionDependencyLifetimePublicationJoin);
        CHECK(DependencyHas(publication, i,
                            UsdGenExecutionDependencyLifetimePublicationJoin));
    }
    CHECK(DependencyHas(publication, 3,
                        UsdGenExecutionDependencySemanticData) &&
          DependencyHas(publication, 3,
                        UsdGenExecutionDependencyResourceHazard));
    CHECK(publication.resourceHazards.size() == 1 &&
          publication.resourceHazards.front().resource ==
              UsdGenExecutionDataKind::ExecutionWorkspace &&
          publication.resourceHazards.front().identity == 0 &&
          publication.resourceHazards.front().access ==
              UsdGenExecutionResourceHazardAccess::ReadWrite);
    CHECK(metadata->FindSemanticTask(publication.semanticNode) == nullptr);
    CHECK(publication.resources.size() == 7);
    CHECK(ResourceIs(publication,0,UsdGenExecutionDataKind::ElementGeometry,UsdGenExecutionResourceAccess::Read,3));
    CHECK(ResourceIs(publication,1,UsdGenExecutionDataKind::ElementTopology,UsdGenExecutionResourceAccess::Read,2));
    CHECK(ResourceIs(publication,2,UsdGenExecutionDataKind::StableIds,UsdGenExecutionResourceAccess::Read,2));
    CHECK(ResourceIs(publication,3,UsdGenExecutionDataKind::RootBindings,UsdGenExecutionResourceAccess::Read,2));
    CHECK(ResourceIs(publication,4,UsdGenExecutionDataKind::Widths,UsdGenExecutionResourceAccess::Read,2));
    CHECK(ResourceIs(publication,5,UsdGenExecutionDataKind::NamedChannels,UsdGenExecutionResourceAccess::Read,2));
    CHECK(ResourceIs(publication,6,UsdGenExecutionDataKind::TerminalGeneration,UsdGenExecutionResourceAccess::Write,external));

    // Every access names an explicit logical input/output version.  The
    // current CUDA implementation truthfully marks intermediate values as
    // exclusive-workspace aliases; only authored inputs and publication are
    // immutable.  Dependency edges alone therefore cannot legalize fan-out.
    for (auto const& task : metadata->Tasks())
        for (auto const& use : task.resources)
            CHECK(ValueFlowIsConsistent(*metadata, task, use));
    for (auto const& value : metadata->Values()) {
        CHECK(value.id < metadata->Values().size());
        if (value.producerTask == UINT32_MAX) {
            CHECK(value.version == 0 &&
                  value.storage == UsdGenExecutionValueStorage::ExternalImmutable);
        } else if (value.resource == UsdGenExecutionDataKind::TerminalGeneration) {
            CHECK(value.producerTask == publication.id && value.version == 1 &&
                  value.storage == UsdGenExecutionValueStorage::PublishedImmutable);
        } else {
            CHECK(value.storage ==
                  UsdGenExecutionValueStorage::ExclusiveWorkspaceVersion);
        }
    }
    CHECK(source.resources[0].outputValue == width.resources[0].inputValue);
    CHECK(source.resources[0].outputValue == length.resources[0].inputValue);
    CHECK(width.resources[3].outputValue == length.resources[4].inputValue);
    CHECK(length.resources[0].outputValue == deform.resources[0].inputValue);
    CHECK(deform.resources[0].outputValue == publication.resources[0].inputValue);
    CHECK(length.resources[5].outputValue == publication.resources[5].inputValue);
    CHECK(publication.resources[6].outputValue != UINT32_MAX);

    // Fused Scatter->Grow is a source value which may feed independent
    // Length and Noise point/topology descendants. A subsequent WidthBlend
    // joins those origins only after retaining the right branch's complete
    // non-width logical packet for the GPU proof.
    auto scatterGrowDag = MakeScatterGrowValueDagDesc();
    diagnostics = {};
    auto scatterGrowPlan = CompileCudaGraph(scatterGrowDag, &diagnostics);
    CHECK(scatterGrowPlan && !diagnostics.HasErrors());
    auto scatterGrowMetadata = GetCudaExecutionPlanMetadata(*scatterGrowPlan);
    auto taskAt = [&](char const* path) -> UsdGenExecutionTaskMetadata const* {
        if (!scatterGrowMetadata) return nullptr;
        auto found = std::find_if(scatterGrowMetadata->Tasks().begin(),
            scatterGrowMetadata->Tasks().end(), [&](auto const& task) {
                return task.path == SdfPath(path);
            });
        return found == scatterGrowMetadata->Tasks().end() ? nullptr : &*found;
    };
    auto const* fused = taskAt("/Ops/Grow");
    auto const* scatterLength = taskAt("/Ops/Length");
    auto const* scatterNoise = taskAt("/Ops/Noise");
    auto const* scatterLeft = taskAt("/Ops/Left");
    auto const* scatterRight = taskAt("/Ops/Right");
    auto const* scatterBlend = taskAt("/Ops/Blend");
    CHECK(scatterGrowMetadata && fused && scatterLength && scatterNoise &&
          scatterLeft && scatterRight && scatterBlend &&
          fused->type == TfToken("CudaScatterGrow") &&
          scatterGrowMetadata->Shape() == UsdGenExecutionPlanShape::SourceRootedValueDag);
    auto hasDependency = [](UsdGenExecutionTaskMetadata const& task, uint32_t id) {
        return std::find(task.dependencies.begin(), task.dependencies.end(), id) != task.dependencies.end();
    };
    // Shared workspace hazards add ordering edges; inherited planes also
    // retain dependencies on their actual producers, not only the last op.
    CHECK(hasDependency(*scatterLength, fused->id) &&
          hasDependency(*scatterNoise, fused->id) &&
          hasDependency(*scatterLeft, scatterLength->id) &&
          hasDependency(*scatterRight, scatterNoise->id) &&
          (hasDependency(*scatterLength, scatterNoise->id) ||
           hasDependency(*scatterNoise, scatterLength->id)) &&
          std::find(scatterBlend->dependencies.begin(), scatterBlend->dependencies.end(),
                    scatterLeft->id) != scatterBlend->dependencies.end() &&
          std::find(scatterBlend->dependencies.begin(), scatterBlend->dependencies.end(),
                    scatterRight->id) != scatterBlend->dependencies.end());
    auto findUse = [](UsdGenExecutionTaskMetadata const& task,
                      UsdGenExecutionDataKind kind,
                      UsdGenExecutionResourceAccess access,
                      uint32_t producer) {
        return std::find_if(task.resources.begin(), task.resources.end(),
            [&](auto const& use) { return use.resource == kind && use.access == access &&
                use.producerTask == producer; }) != task.resources.end();
    };
    CHECK(findUse(*scatterLength, UsdGenExecutionDataKind::ElementGeometry,
                  UsdGenExecutionResourceAccess::ReadWrite, fused->id) &&
          findUse(*scatterNoise, UsdGenExecutionDataKind::ElementGeometry,
                  UsdGenExecutionResourceAccess::ReadWrite, fused->id) &&
          findUse(*scatterRight, UsdGenExecutionDataKind::Widths,
                  UsdGenExecutionResourceAccess::ReadWrite, fused->id));
    // Noise owns the right point revision, while Grow remains the owner of
    // its inherited topology/IDs/root bindings/named channels.
    CHECK(findUse(*scatterBlend, UsdGenExecutionDataKind::ElementGeometry,
                  UsdGenExecutionResourceAccess::Read, scatterNoise->id));
    for (auto kind : {UsdGenExecutionDataKind::ElementTopology,
                      UsdGenExecutionDataKind::StableIds,
                      UsdGenExecutionDataKind::RootBindings,
                      UsdGenExecutionDataKind::NamedChannels})
        CHECK(findUse(*scatterBlend, kind, UsdGenExecutionResourceAccess::Read, fused->id));
    CHECK(findUse(*scatterBlend, UsdGenExecutionDataKind::Widths,
                  UsdGenExecutionResourceAccess::Read, scatterRight->id));
    // The fused Grow value may be selected for publication even when its
    // Length/Noise descendants remain present as siblings.
    auto selectedGrow = scatterGrowDag;
    selectedGrow.nodes.erase(std::remove_if(selectedGrow.nodes.begin(), selectedGrow.nodes.end(),
        [](UsdGenNodeDesc const& node) { return node.path == SdfPath("/Ops/Blend"); }),
        selectedGrow.nodes.end());
    selectedGrow.terminal = SdfPath("/Ops/Grow");
    diagnostics = {};
    auto selectedGrowPlan = CompileCudaGraph(selectedGrow, &diagnostics);
    CHECK(selectedGrowPlan && !diagnostics.HasErrors());
    auto selectedGrowMetadata = GetCudaExecutionPlanMetadata(*selectedGrowPlan);
    CHECK(selectedGrowMetadata && selectedGrowMetadata->FindTask(
        selectedGrowMetadata->TerminalTask())->kind == UsdGenExecutionTaskKind::Publication);
    auto const& selectedPublication = *selectedGrowMetadata->FindTask(
        selectedGrowMetadata->TerminalTask());
    CHECK(std::any_of(selectedPublication.resources.begin(), selectedPublication.resources.end(),
        [](auto const& use) { return use.resource == UsdGenExecutionDataKind::ElementGeometry &&
            use.access == UsdGenExecutionResourceAccess::Read && use.producerTask == 0; }));
    // A resolved surface also admits RBF Deform after a fused Grow lineage.
    // The Deform task owns only the new geometry revision; topology, roots,
    // IDs, named channels, and widths remain inherited from its input.
    auto scatterDeform = scatterGrowDag;
    UsdGenNodeDesc fusedDeform;
    fusedDeform.path = SdfPath("/Ops/Deform"); fusedDeform.type = TfToken("UsdGenDeform");
    fusedDeform.inputs = {SdfPath("/Ops/Length")}; fusedDeform.surfaces = {SdfPath("/Scalp")};
    fusedDeform.params = {{TfToken("rbfSamples"), VtValue(5), false}};
    scatterDeform.nodes.push_back(fusedDeform); scatterDeform.terminal = fusedDeform.path;
    diagnostics = {};
    auto scatterDeformPlan = CompileCudaGraph(scatterDeform, &diagnostics);
    CHECK(scatterDeformPlan && !diagnostics.HasErrors());
    auto scatterDeformMetadata = GetCudaExecutionPlanMetadata(*scatterDeformPlan);
    auto findTask = [](std::shared_ptr<const UsdGenExecutionPlanMetadata> const& metadata,
                       char const* path) -> UsdGenExecutionTaskMetadata const* {
        if (!metadata) return nullptr;
        auto found = std::find_if(metadata->Tasks().begin(), metadata->Tasks().end(),
            [&](auto const& task) { return task.path == SdfPath(path); });
        return found == metadata->Tasks().end() ? nullptr : &*found;
    };
    auto const* scatterDeformTask = findTask(scatterDeformMetadata, "/Ops/Deform");
    CHECK(scatterDeformMetadata && scatterDeformTask &&
          !scatterDeformMetadata->MemoryEstimate().memoryAvailable &&
          !scatterDeformMetadata->MemoryEstimate().conservativeUpperBound &&
          !scatterDeformMetadata->MemoryEstimate().runtimeRefinementAvailable);
    CHECK(findUse(*scatterDeformTask, UsdGenExecutionDataKind::ElementGeometry,
                  UsdGenExecutionResourceAccess::ReadWrite, scatterLength->id));
    auto const surfaceUse = std::find_if(scatterDeformTask->resources.begin(),
        scatterDeformTask->resources.end(), [](auto const& use) {
            return use.resource == UsdGenExecutionDataKind::SurfaceGeometry &&
                use.access == UsdGenExecutionResourceAccess::Read &&
                use.producerTask == UINT32_MAX;
        });
    CHECK(surfaceUse != scatterDeformTask->resources.end() &&
          surfaceUse->inputValue < scatterDeformMetadata->Values().size());
    CHECK(scatterDeformMetadata->Values()[surfaceUse->inputValue].storage ==
          UsdGenExecutionValueStorage::ExternalImmutable);

    // Two Deforms on independent fused descendants are legal and retain
    // their respective geometry producers.
    auto independentDeform = scatterDeform;
    auto secondIndependent = fusedDeform;
    secondIndependent.path = SdfPath("/Ops/NoiseDeform");
    secondIndependent.inputs = {SdfPath("/Ops/Noise")};
    independentDeform.nodes.push_back(secondIndependent);
    independentDeform.terminal = secondIndependent.path;
    diagnostics = {};
    auto independentPlan = CompileCudaGraph(independentDeform, &diagnostics);
    CHECK(independentPlan && !diagnostics.HasErrors());
    auto independentMetadata = GetCudaExecutionPlanMetadata(*independentPlan);
    auto const* noiseDeformTask = findTask(independentMetadata, "/Ops/NoiseDeform");
    CHECK(independentMetadata && noiseDeformTask &&
          findUse(*noiseDeformTask, UsdGenExecutionDataKind::ElementGeometry,
                  UsdGenExecutionResourceAccess::ReadWrite, scatterNoise->id));

    // A second Deform on the same lineage remains forbidden because it would
    // apply rest-to-animated surface motion twice.
    auto doubleDeform = scatterDeform;
    auto secondDeform = fusedDeform;
    secondDeform.path = SdfPath("/Ops/SecondDeform");
    secondDeform.inputs = {fusedDeform.path};
    doubleDeform.nodes.push_back(secondDeform); doubleDeform.terminal = secondDeform.path;
    diagnostics = {};
    CHECK(!CompileCudaGraph(doubleDeform, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unsupported composition: a second rest-to-animated deformation would apply surface motion twice"));
    // Width may not bypass Grow to consume raw Scatter output.
    auto rawScatter = selectedGrow;
    auto rawWidth = std::find_if(rawScatter.nodes.begin(), rawScatter.nodes.end(),
        [](UsdGenNodeDesc const& node) { return node.path == SdfPath("/Ops/Left"); });
    CHECK(rawWidth != rawScatter.nodes.end());
    rawWidth->inputs = {SdfPath("/Ops/Scatter")}; rawScatter.terminal = rawWidth->path;
    diagnostics = {}; CHECK(!CompileCudaGraph(rawScatter, &diagnostics) && diagnostics.HasErrors());

    // Failed admission must classify the exact reason instead of collapsing
    // unsupported operators, configurations, and compositions together.
    auto unsupportedOperator = MakeDesc();
    unsupportedOperator.nodes[1].type = TfToken("UsdGenUnknownOperator");
    diagnostics = {};
    CHECK(!CompileCudaGraph(unsupportedOperator,&diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unsupported operator: CUDA capability matrix has no implementation for UsdGenUnknownOperator"));

    auto unsupportedConfiguration = MakeDesc();
    unsupportedConfiguration.nodes[2].params.push_back(
        {TfToken("mask"), VtValue(2.0f), false});
    diagnostics = {};
    CHECK(!CompileCudaGraph(unsupportedConfiguration,&diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unsupported or malformed Length parameter mask"));

    // RBF consumes the exact immutable value on its authored predecessor
    // lineage. Independent Width/Length -> RBF branches are legal; the
    // terminal selects the Length branch while the Width branch remains a
    // sibling value, rather than forcing the historical linear composition.
    auto branchedRbf = MakeDesc();
    branchedRbf.nodes[2].inputs = {branchedRbf.nodes[0].path};
    auto rbfSibling = branchedRbf.nodes[3];
    rbfSibling.path = SdfPath("/Groom/Hair/Ops/XSiblingDeform");
    rbfSibling.inputs = {branchedRbf.nodes[1].path};
    branchedRbf.nodes.push_back(rbfSibling);
    diagnostics = {};
    auto branchedRbfPlan = CompileCudaGraph(branchedRbf, &diagnostics);
    CHECK(branchedRbfPlan && !diagnostics.HasErrors());
    auto branchedRbfMetadata = GetCudaExecutionPlanMetadata(*branchedRbfPlan);
    CHECK(branchedRbfMetadata &&
          branchedRbfMetadata->Shape() == UsdGenExecutionPlanShape::SourceRootedUnaryDag);
    auto const* branchWidth = branchedRbfMetadata->FindSemanticTask(1);
    auto const* branchLength = branchedRbfMetadata->FindSemanticTask(2);
    auto const* branchTerminal = branchedRbfMetadata->FindSemanticTask(3);
    auto const* branchSibling = branchedRbfMetadata->FindSemanticTask(4);
    auto const* branchPublication = branchedRbfMetadata->FindTask(
        branchedRbfMetadata->TerminalTask());
    CHECK(branchWidth && branchLength && branchTerminal && branchSibling &&
          branchPublication && branchPublication->kind == UsdGenExecutionTaskKind::Publication &&
          DependencyHas(*branchTerminal, branchLength->id,
                        UsdGenExecutionDependencySemanticData) &&
          DependencyHas(*branchSibling, branchWidth->id,
                        UsdGenExecutionDependencySemanticData));
    auto geometryUse = [](UsdGenExecutionTaskMetadata const& task) {
        for (auto const& use : task.resources)
            if (use.resource == UsdGenExecutionDataKind::ElementGeometry) return &use;
        return static_cast<UsdGenExecutionResourceUse const*>(nullptr);
    };
    auto const* terminalGeometry = geometryUse(*branchTerminal);
    auto const* siblingGeometry = geometryUse(*branchSibling);
    auto const* publicationGeometry = geometryUse(*branchPublication);
    CHECK(terminalGeometry && siblingGeometry && publicationGeometry &&
          terminalGeometry->producerTask == branchLength->id &&
          siblingGeometry->producerTask == branchedRbfMetadata->FindSemanticTask(0)->id &&
          publicationGeometry->producerTask == branchTerminal->id);

    // A second RBF on the same lineage would reapply rest-to-animated surface
    // motion. It remains a precise compile-time rejection, not a branch
    // restriction masquerading as a safety policy.
    auto doubleRbf = MakeDesc();
    auto secondRbf = doubleRbf.nodes[3];
    secondRbf.path = SdfPath("/Groom/Hair/Ops/SecondDeform");
    secondRbf.inputs = {doubleRbf.nodes[3].path};
    doubleRbf.nodes.push_back(secondRbf); doubleRbf.terminal = secondRbf.path;
    diagnostics = {};
    CHECK(!CompileCudaGraph(doubleRbf,&diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unsupported composition: a second rest-to-animated deformation would apply surface motion twice"));

    // Curve and surface inputs are direct CUDA source payloads and remain
    // supported.  References and maps are different: CPU lowering normally
    // materializes them as ReferenceInputs/map payloads, for which CUDA has no
    // upload or ownership contract yet.
    auto directCurveSurfaceSource = MakeDesc();
    directCurveSurfaceSource.nodes.resize(1);
    directCurveSurfaceSource.terminal = directCurveSurfaceSource.nodes.front().path;
    diagnostics = {};
    CHECK(CompileCudaGraph(directCurveSurfaceSource, &diagnostics));
    CHECK(!diagnostics.HasErrors());

    // Only the explicit surfaceCage source boundary may consume its one
    // categorical regionMap.  A normal CurveSource with a typed ordinary map
    // must still fail before CUDA lowering rather than silently ignore it.
    auto ordinarySourceMap = directCurveSurfaceSource;
    ordinarySourceMap.nodes.front().maps = {SdfPath("/Maps/ordinary")};
    ordinarySourceMap.nodes.front().mapBindings = {
        {SdfPath("/Maps/ordinary"), TfToken("usdGen:map")}};
    diagnostics = {};
    CHECK(!CompileCudaGraph(ordinarySourceMap, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unresolved ReferenceInputs/maps are not supported by CUDA at "
        "/Groom/Hair/Ops/ZSource (references=0, maps=1)"));

    auto unresolvedReference = MakeDesc();
    unresolvedReference.nodes[1].references = {SdfPath("/Guides")};
    diagnostics = {};
    CHECK(!CompileCudaGraph(unresolvedReference, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unresolved ReferenceInputs/maps are not supported by CUDA at "
        "/Groom/Hair/Ops/AWidth (references=1, maps=0)"));

    auto unresolvedMap = MakeDesc();
    unresolvedMap.nodes[2].maps = {SdfPath("/Maps/lengthMask")};
    diagnostics = {};
    CHECK(!CompileCudaGraph(unresolvedMap, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unresolved ReferenceInputs/maps are not supported by CUDA at "
        "/Groom/Hair/Ops/MLength (references=0, maps=1)"));

    // Check the combined form on an otherwise CUDA-unknown operator too.  The
    // payload boundary is intentionally checked before capability lookup, so
    // a ReferenceInputs-requiring operator can never progress to submission.
    auto unresolvedFutureReferenceOperator = MakeDesc();
    unresolvedFutureReferenceOperator.nodes[1].type = TfToken("UsdGenUnknownOperator");
    unresolvedFutureReferenceOperator.nodes[1].references = {SdfPath("/Guides")};
    unresolvedFutureReferenceOperator.nodes[1].maps = {SdfPath("/Maps/density")};
    diagnostics = {};
    CHECK(!CompileCudaGraph(unresolvedFutureReferenceOperator, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unresolved ReferenceInputs/maps are not supported by CUDA at "
        "/Groom/Hair/Ops/AWidth (references=1, maps=1)"));

    auto unsupportedNamedTopology = MakeDesc();
    unsupportedNamedTopology.nodes[1].type = TfToken("UsdGenTestTopologyNamedPlane");
    diagnostics = {};
    CHECK(!CompileCudaGraph(unsupportedNamedTopology, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: named topology planes require CUDA execution-graph transport at "
        "/Groom/Hair/Ops/AWidth"));

    auto unsupportedNamedValue = MakeDesc();
    unsupportedNamedValue.nodes[1].type = TfToken("UsdGenTestValueNamedPlane");
    diagnostics = {};
    CHECK(!CompileCudaGraph(unsupportedNamedValue, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: named primvar bindings require an explicit CUDA operator COW "
        "contract at /Groom/Hair/Ops/AWidth"));

    // A statically sized resample owns both its device error scalar and the
    // pinned host error scalar through publication. They are steady auxiliary
    // storage, not recyclable source-phase scratch. The surrounding mixed
    // graph remains unavailable because Length/RBF have dynamic workspace.
    auto staticResample = MakeDesc();
    staticResample.nodes.front().params.push_back(
        {TfToken("resampleTo"), VtValue(4), false});
    diagnostics = {};
    auto staticResamplePlan = CompileCudaGraph(staticResample, &diagnostics);
    CHECK(staticResamplePlan && !diagnostics.HasErrors());
    auto staticResampleMetadata = GetCudaExecutionPlanMetadata(*staticResamplePlan);
    CHECK(staticResampleMetadata &&
          staticResampleMetadata->Tasks().front().estimate.memoryAvailable &&
          staticResampleMetadata->Tasks().front().estimate.steadyBytes ==
              2 * sizeof(int) &&
          staticResampleMetadata->Tasks().front().estimate.scratchPeakBytes == 0 &&
          !staticResampleMetadata->MemoryEstimate().memoryAvailable &&
          !staticResampleMetadata->MemoryEstimate().conservativeUpperBound &&
          !staticResampleMetadata->MemoryEstimate().runtimeRefinementAvailable);

    // Named-plane accounting follows the target domain, rather than merely
    // charging the authored source array again after resampling.
    auto namedResample = staticResample;
    UsdGenAuthoredPlaneDesc pointPlane;
    pointPlane.name = TfToken("pointPair");
    pointPlane.type = UsdGenAuthoredPlaneType::Float32;
    pointPlane.domain = UsdGenAuthoredPlaneDomain::Point;
    pointPlane.arity = 2;
    pointPlane.floatValues = {1.f, 2.f, 3.f, 4.f};
    UsdGenAuthoredPlaneDesc primitivePlane;
    primitivePlane.name = TfToken("primitiveTriple");
    primitivePlane.type = UsdGenAuthoredPlaneType::Int32;
    primitivePlane.domain = UsdGenAuthoredPlaneDomain::Primitive;
    primitivePlane.arity = 3;
    primitivePlane.intValues = {5, 6, 7};
    UsdGenAuthoredPlaneDesc groomPlane;
    groomPlane.name = TfToken("groomQuad");
    groomPlane.type = UsdGenAuthoredPlaneType::Float32;
    groomPlane.domain = UsdGenAuthoredPlaneDomain::Groom;
    groomPlane.arity = 4;
    groomPlane.floatValues = {8.f, 9.f, 10.f, 11.f};
    namedResample.curveSets.front().authoredPlanes = {
        pointPlane, primitivePlane, groomPlane};
    diagnostics = {};
    auto namedResamplePlan = CompileCudaGraph(namedResample, &diagnostics);
    CHECK(namedResamplePlan && !diagnostics.HasErrors());
    auto namedResampleMetadata = GetCudaExecutionPlanMetadata(*namedResamplePlan);
    CHECK(namedResampleMetadata);
    auto const& plainSource = staticResampleMetadata->Tasks().front().estimate;
    auto const& namedSource = namedResampleMetadata->Tasks().front().estimate;
    CHECK(namedSource.producerRetentionBytes ==
              plainSource.producerRetentionBytes + 44 &&
          namedSource.retainedOutputBytes ==
              plainSource.retainedOutputBytes + 60);

    // A resampled Point plane is charged at target cardinality while the
    // canonical source plane remains live as producer retention. The raw
    // named-topology transaction additionally owns one device and one pinned
    // status scalar until its terminal callback is proved.
    auto namedStaticResample = staticResample;
    UsdGenAuthoredPlaneDesc density;
    density.name = TfToken("density");
    density.type = UsdGenAuthoredPlaneType::Float32;
    density.domain = UsdGenAuthoredPlaneDomain::Point;
    density.arity = 1;
    density.floatValues = {1.f, 2.f};
    namedStaticResample.curveSets.front().authoredPlanes.push_back(density);
    diagnostics = {};
    auto namedStaticPlan = CompileCudaGraph(namedStaticResample, &diagnostics);
    CHECK(namedStaticPlan && !diagnostics.HasErrors());
    auto namedStaticMetadata = GetCudaExecutionPlanMetadata(*namedStaticPlan);
    auto const& baseSourceEstimate =
        staticResampleMetadata->Tasks().front().estimate;
    auto const& namedSourceEstimate =
        namedStaticMetadata->Tasks().front().estimate;
    CHECK(namedSourceEstimate.memoryAvailable &&
          namedSourceEstimate.steadyBytes == baseSourceEstimate.steadyBytes &&
          namedSourceEstimate.producerRetentionBytes ==
              baseSourceEstimate.producerRetentionBytes + 2 * sizeof(float) &&
          namedSourceEstimate.retainedOutputBytes ==
              baseSourceEstimate.retainedOutputBytes + 4 * sizeof(float) &&
          namedSourceEstimate.scratchPeakBytes ==
              baseSourceEstimate.scratchPeakBytes + 2 * sizeof(int));

    // A runtime expression controlling CurveSource resampling changes output
    // cardinality. The compiler must not manufacture an exact byte bound from
    // the literal fallback; timing remains independently unknown as well.
    auto dynamic = MakeDesc();
    UsdGenExpressionDesc expression;
    expression.path = SdfPath("/Groom/Hair/Expressions/resampleTo");
    expression.source = "$frame < 1 ? 0 : 4";
    expression.outputs.push_back({TfToken("result"), TfToken("int"),
                                  {expr::ScalarType::Int32, 1, 1, 1, 1, false}});
    dynamic.expressions.push_back(expression);
    UsdGenExpressionBinding binding;
    binding.expression = expression.path;
    binding.destination = TfToken("resampleTo");
    binding.domain = expr::Domain::Groom;
    binding.nativeType = TfToken("int");
    binding.destinationShape = {expr::ScalarType::Int32, 1, 1, 1, 1, false};
    binding.literal = VtValue(0);
    dynamic.nodes[0].expressionBindings.push_back(binding);
    diagnostics = {};
    auto dynamicPlan = CompileCudaGraph(dynamic, &diagnostics);
    CHECK(dynamicPlan && !diagnostics.HasErrors());
    auto dynamicMetadata = GetCudaExecutionPlanMetadata(*dynamicPlan);
    CHECK(dynamicMetadata && !dynamicMetadata->MemoryEstimate().memoryAvailable &&
          !dynamicMetadata->MemoryEstimate().conservativeUpperBound &&
          !dynamicMetadata->MemoryEstimate().runtimeRefinementAvailable);
    for (auto const& task : dynamicMetadata->Tasks())
        CHECK(!task.estimate.timeAvailable &&
              task.estimate.estimatedMicroseconds == 0);

    // Fixed-cardinality expressions participate in aggregate admission. The
    // source scalar packet is scratch, while complete fresh evaluator
    // candidates are counted once and the prior published COW generation is
    // left to its existing global charge.
    auto boundedExpressions = MakeDesc();
    boundedExpressions.nodes.resize(2);
    boundedExpressions.terminal = boundedExpressions.nodes[1].path;
    AddScalarExpression(&boundedExpressions, &boundedExpressions.nodes[0],
        "/Groom/Hair/Expressions/useRest", "$frame >= 0", "useRest",
        expr::Domain::Groom, TfToken("bool"), expr::ScalarType::Bool,
        VtValue(true));
    AddScalarExpression(&boundedExpressions, &boundedExpressions.nodes[1],
        "/Groom/Hair/Expressions/width", "$value * (1 + $t)", "width",
        expr::Domain::Point, TfToken("float"), expr::ScalarType::Float32,
        VtValue(.03f));
    diagnostics = {};
    auto boundedPlan = CompileCudaGraph(boundedExpressions, &diagnostics);
    CHECK(boundedPlan && !diagnostics.HasErrors());
    auto boundedMetadata = GetCudaExecutionPlanMetadata(*boundedPlan);
    CHECK(boundedMetadata && boundedMetadata->MemoryEstimate().memoryAvailable &&
          boundedMetadata->MemoryEstimate().conservativeUpperBound &&
          !boundedMetadata->MemoryEstimate().runtimeRefinementAvailable &&
          boundedMetadata->Tasks()[0].estimate.steadyBytes > 0 &&
          boundedMetadata->Tasks()[0].estimate.scratchPeakBytes == 64 &&
          boundedMetadata->Tasks()[1].estimate.steadyBytes > 0);

    // Automatic capture creates the same root channels as a complete authored
    // binding. Its static source/output and evaluator reservation must not
    // shrink just because those arrays are absent (or only partly authored).
    for (int variant = 0; variant != 3; ++variant) {
        auto automatic = boundedExpressions;
        if (variant != 1) automatic.curveSets[0].skinPrim.clear();
        if (variant != 2) automatic.curveSets[0].skinPrimUv.clear();
        diagnostics = {};
        auto automaticPlan = CompileCudaGraph(automatic, &diagnostics);
        CHECK(automaticPlan && !diagnostics.HasErrors());
        auto automaticMetadata = GetCudaExecutionPlanMetadata(*automaticPlan);
        CHECK(automaticMetadata && automaticMetadata->MemoryEstimate().memoryAvailable);
        CHECK(automaticMetadata->MemoryEstimate().immutableSharedInputBytes ==
              boundedMetadata->MemoryEstimate().immutableSharedInputBytes);
        CHECK(automaticMetadata->Tasks()[0].estimate.retainedOutputBytes ==
              boundedMetadata->Tasks()[0].estimate.retainedOutputBytes);
        CHECK(automaticMetadata->Tasks()[1].estimate.steadyBytes ==
              boundedMetadata->Tasks()[1].estimate.steadyBytes);
    }
    auto unboundEstimate = boundedExpressions;
    unboundEstimate.nodes[0].surfaces.clear();
    unboundEstimate.nodes[0].params.push_back(
        {TfToken("rebind"), VtValue(TfToken("never")), false});
    unboundEstimate.curveSets[0].skinPrim.clear();
    unboundEstimate.curveSets[0].skinPrimUv.clear();
    diagnostics = {};
    auto unboundEstimatePlan = CompileCudaGraph(unboundEstimate, &diagnostics);
    CHECK(unboundEstimatePlan && !diagnostics.HasErrors());
    auto unboundMetadata = GetCudaExecutionPlanMetadata(*unboundEstimatePlan);
    CHECK(unboundMetadata &&
          unboundMetadata->MemoryEstimate().immutableSharedInputBytes +
              12u * unboundEstimate.curveSets[0].curveVertexCounts.size() ==
          boundedMetadata->MemoryEstimate().immutableSharedInputBytes);

    // Length expressions remain shape-bounded by the all-survivor input and
    // are admitted by the selected-device CUB refinement.
    auto boundedLength = MakeDesc();
    boundedLength.nodes.erase(boundedLength.nodes.begin() + 1);
    boundedLength.nodes.resize(2);
    boundedLength.nodes[1].inputs = {boundedLength.nodes[0].path};
    boundedLength.terminal = boundedLength.nodes[1].path;
    AddScalarExpression(&boundedLength, &boundedLength.nodes[1],
        "/Groom/Hair/Expressions/length", "$value", "length:value",
        expr::Domain::Primitive, TfToken("float"), expr::ScalarType::Float32,
        VtValue(.5f));
    diagnostics = {};
    auto boundedLengthPlan = CompileCudaGraph(boundedLength, &diagnostics);
    CHECK(boundedLengthPlan && !diagnostics.HasErrors());
    auto boundedLengthMetadata = GetCudaExecutionPlanMetadata(*boundedLengthPlan);
    CHECK(boundedLengthMetadata &&
          !boundedLengthMetadata->MemoryEstimate().memoryAvailable &&
          boundedLengthMetadata->MemoryEstimate().runtimeRefinementAvailable &&
          boundedLengthMetadata->Tasks()[1].estimate.steadyBytes > 0);

    // RBF control expressions are bounded except for rbfSamples, which
    // changes the selected-device LU dimensions and remains unavailable.
    auto boundedRbf = MakeDesc();
    boundedRbf.nodes = {boundedRbf.nodes[0], boundedRbf.nodes[3]};
    boundedRbf.nodes[1].inputs = {boundedRbf.nodes[0].path};
    boundedRbf.terminal = boundedRbf.nodes[1].path;
    AddScalarExpression(&boundedRbf, &boundedRbf.nodes[1],
        "/Groom/Hair/Expressions/maskAmount", "$value", "mask",
        expr::Domain::Groom, TfToken("float"), expr::ScalarType::Float32,
        VtValue(1.f));
    diagnostics = {};
    auto boundedRbfPlan = CompileCudaGraph(boundedRbf, &diagnostics);
    CHECK(boundedRbfPlan && !diagnostics.HasErrors());
    auto boundedRbfMetadata = GetCudaExecutionPlanMetadata(*boundedRbfPlan);
    CHECK(boundedRbfMetadata &&
          boundedRbfMetadata->MemoryEstimate().runtimeRefinementAvailable);
    AddScalarExpression(&boundedRbf, &boundedRbf.nodes[1],
        "/Groom/Hair/Expressions/rbfSamples", "5", "rbfSamples",
        expr::Domain::Groom, TfToken("int"), expr::ScalarType::Int32,
        VtValue(5));
    diagnostics = {};
    auto dynamicRbfPlan = CompileCudaGraph(boundedRbf, &diagnostics);
    CHECK(dynamicRbfPlan && !diagnostics.HasErrors());
    auto dynamicRbfMetadata = GetCudaExecutionPlanMetadata(*dynamicRbfPlan);
    CHECK(dynamicRbfMetadata &&
          !dynamicRbfMetadata->MemoryEstimate().runtimeRefinementAvailable);
    auto missingRbfRoots = boundedRbf;
    missingRbfRoots.nodes[1].expressionBindings.pop_back();
    missingRbfRoots.curveSets.front().skinPrimUv.clear();
    diagnostics = {};
    auto missingRbfRootsPlan = CompileCudaGraph(missingRbfRoots, &diagnostics);
    CHECK(missingRbfRootsPlan && !diagnostics.HasErrors());
    auto missingRbfRootsMetadata =
        GetCudaExecutionPlanMetadata(*missingRbfRootsPlan);
    CHECK(missingRbfRootsMetadata &&
          !missingRbfRootsMetadata->MemoryEstimate().runtimeRefinementAvailable);

    // Empty geometry is a valid zero-cardinality source. Its estimate may
    // retain a terminal offset, but must stay finite and must not claim time.
    auto empty = MakeDesc();
    auto& emptyCurves = empty.curveSets.front();
    emptyCurves.curveVertexCounts.clear();
    emptyCurves.points.clear();
    emptyCurves.rest.clear();
    emptyCurves.curveId.clear();
    emptyCurves.skinPrim.clear();
    emptyCurves.skinPrimUv.clear();
    diagnostics = {};
    auto emptyPlan = CompileCudaGraph(empty, &diagnostics);
    CHECK(emptyPlan && !diagnostics.HasErrors());
    auto emptyMetadata = GetCudaExecutionPlanMetadata(*emptyPlan);
    CHECK(emptyMetadata);
    for (auto const& task : emptyMetadata->Tasks())
        CHECK(!task.estimate.timeAvailable &&
              task.estimate.estimatedMicroseconds == 0);
    if (emptyMetadata->MemoryEstimate().memoryAvailable)
        CHECK(emptyMetadata->MemoryEstimate().conservativeUpperBound &&
              emptyMetadata->MemoryEstimate().immutableSharedInputBytes ==
                  ExpectedSourcePayloadBytes(empty));

    // The pure checked-arithmetic contract must reject overflow rather than
    // saturating to UINT64_MAX and presenting a false conservative estimate.
    // Exercise the same backend-neutral checked arithmetic used by CUDA
    // planning rather than duplicating a test-only implementation.
    uint64_t checked = 0;
    CHECK(!UsdGenExecutionCheckedBytes::Multiply(
              2, std::numeric_limits<uint64_t>::max(), &checked) &&
          checked == 0);
    checked = std::numeric_limits<uint64_t>::max();
    CHECK(!UsdGenExecutionCheckedBytes::Add(&checked, 1) &&
          checked == std::numeric_limits<uint64_t>::max());
    return 0;
}
