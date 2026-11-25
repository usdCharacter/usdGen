#include "usdGen/compiler.h"
#include "nodeSets.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/cudaExecutionQueue.h"
#include "usdGen/executionResources.h"
#include "usdGen/graph.h"
#include "usdGen/gpu/generation.h"
#include "usdGen/imagePayload.h"
#include "usdGen/scheduler.h"
#include "usdGen/session.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace usdGen;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {

UsdGenNodeDesc Width(char const* path, char const* input, float value) {
    UsdGenNodeDesc node;
    node.path = SdfPath(path);
    node.type = TfToken("UsdGenWidth");
    node.inputs = {SdfPath(input)};
    node.params.push_back({TfToken("width"), VtValue(value), false});
    return node;
}

UsdGenNodeDesc Length(char const* path, char const* input) {
    UsdGenNodeDesc node;
    node.path = SdfPath(path);
    node.type = TfToken("UsdGenLength");
    node.inputs = {SdfPath(input)};
    node.params.push_back(
        {TfToken("length:mode"), VtValue(TfToken("cull")), false});
    node.params.push_back(
        {TfToken("cullThreshold"), VtValue(.45f), false});
    return node;
}

UsdGenNodeDesc RbfDeform(char const* path, char const* input) {
    UsdGenNodeDesc node;
    node.path = SdfPath(path);
    node.type = TfToken("UsdGenDeform");
    node.inputs = {SdfPath(input)};
    node.surfaces = {SdfPath("/Dag/Scalp")};
    node.params.push_back({TfToken("rbfSamples"), VtValue(5), false});
    return node;
}

UsdGenGraphDesc BaseDesc() {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Dag/Hair");
    desc.executionBackend = UsdGenExecutionBackend::Cuda;

    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Dag/Scalp");
    surface.restPoints = {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {1,1,1}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {3,3,3};
    surface.faceVertexIndices = {0,1,2, 0,1,3, 1,2,4};
    desc.surfaces.push_back(surface);

    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Dag/Curves");
    curves.role = UsdGenRole::Curves;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {2};
    curves.points = {{.2f,.2f,0}, {.2f,.4f,0}};
    curves.rest = curves.points;
    curves.curveId = {42};
    curves.skinPrim = {0};
    curves.skinPrimUv = {{.2f,.2f}};
    desc.curveSets.push_back(curves);
    return desc;
}

UsdGenNodeDesc Source() {
    UsdGenNodeDesc node;
    node.path = SdfPath("/Dag/Source");
    node.type = TfToken("UsdGenCurveSource");
    node.curves = {SdfPath("/Dag/Curves")};
    node.surfaces = {SdfPath("/Dag/Scalp")};
    return node;
}

UsdGenNodeDesc ReferenceSource() {
    UsdGenNodeDesc node;
    node.path = SdfPath("/Dag/ReferenceSource");
    node.type = TfToken("UsdGenReferenceSource");
    node.references = {SdfPath("/Dag/Curves")};
    return node;
}

UsdGenGraphDesc ReferenceWidthDesc(float width = .25f) {
    auto desc = BaseDesc();
    desc.curveSets.front().role = UsdGenRole::Reference;
    desc.curveSets.front().curveRole = TfToken("guide");
    auto source = ReferenceSource();
    auto terminal = Width("/Dag/ReferenceWidth", "/Dag/ReferenceSource", width);
    desc.nodes = {source, terminal};
    desc.terminal = terminal.path;
    return desc;
}

UsdGenGraphDesc ReferenceLengthWidthDesc() {
    auto desc = ReferenceWidthDesc();
    auto length = Length("/Dag/ReferenceLength", "/Dag/ReferenceSource");
    // Keep every reference curve alive so this test exercises the literal
    // Length admission/compaction path without making its output cardinality
    // dependent on the cull threshold.
    length.params.clear();
    length.params.push_back(
        {TfToken("length:mode"), VtValue(TfToken("scale")), false});
    length.params.push_back(
        {TfToken("length:value"), VtValue(.5f), false});
    auto terminal = Width("/Dag/ReferenceLengthWidth",
                          "/Dag/ReferenceLength", .75f);
    auto sibling = Width("/Dag/ReferenceLengthSibling",
                         "/Dag/ReferenceLength", .25f);
    desc.nodes = {ReferenceSource(), length, terminal, sibling};
    desc.terminal = terminal.path;
    return desc;
}

UsdGenGraphDesc FanoutDesc() {
    auto desc = BaseDesc();
    auto terminal = Width("/Dag/Terminal", "/Dag/Source", .25f);
    auto sibling = Width("/Dag/Sibling", "/Dag/Source", .75f);
    // Deliberately shuffled. The terminal has the lower authored ordinal in
    // normalized metadata; publication must still select its snapshot while
    // the independent sibling is allowed to run concurrently.
    desc.nodes = {terminal, Source(), sibling};
    desc.terminal = terminal.path;
    return desc;
}

UsdGenGraphDesc WidthBlendDesc(float blend = .25f) {
    auto desc = BaseDesc();
    auto left = Width("/Dag/BlendLeft", "/Dag/Source", .2f);
    auto right = Width("/Dag/BlendRight", "/Dag/Source", .8f);
    UsdGenNodeDesc merge;
    merge.path = SdfPath("/Dag/Blend");
    merge.type = TfToken("UsdGenWidthBlend");
    merge.inputs = {left.path, right.path};
    merge.params.push_back({TfToken("widthBlend:weight"), VtValue(blend), false});
    // Deliberately non-topological: lowering must retain the authored input
    // order, not storage order, while both Width predecessors remain COW.
    desc.nodes = {merge, right, Source(), left};
    desc.terminal = merge.path;
    return desc;
}

// Two independently-produced, topology-preserving Length values have
// distinct executor origins while retaining the same full non-width payload.
// WidthBlend must prove that payload on the device before accepting this join.
UsdGenGraphDesc CrossOriginWidthBlendDesc(float rightScale = 1.0f) {
    auto desc = BaseDesc();
    UsdGenAuthoredPlaneDesc named;
    named.name=TfToken("crossOriginValue"); named.type=UsdGenAuthoredPlaneType::Float32;
    named.domain=UsdGenAuthoredPlaneDomain::Point; named.arity=1;
    named.floatValues={.25f,.75f}; desc.curveSets.front().authoredPlanes.push_back(named);
    auto leftLength = Length("/Dag/EqualLeftLength", "/Dag/Source");
    leftLength.params = {{TfToken("length:mode"), VtValue(TfToken("scale")), false},
                         {TfToken("length:value"), VtValue(1.0f), false}};
    auto rightLength = leftLength;
    rightLength.path = SdfPath("/Dag/EqualRightLength");
    rightLength.params[1].value = VtValue(rightScale);
    auto left = Width("/Dag/EqualLeftWidth", "/Dag/EqualLeftLength", .2f);
    auto right = Width("/Dag/EqualRightWidth", "/Dag/EqualRightLength", .8f);
    UsdGenNodeDesc merge;
    merge.path = SdfPath("/Dag/CrossOriginBlend");
    merge.type = TfToken("UsdGenWidthBlend");
    merge.inputs = {left.path, right.path};
    merge.params.push_back({TfToken("widthBlend:weight"), VtValue(.25f), false});
    desc.nodes = {merge, right, rightLength, left, leftLength, Source()};
    desc.terminal = merge.path;
    return desc;
}

UsdGenGraphDesc TopologyFanoutDesc() {
    auto desc = BaseDesc();
    auto& curves = desc.curveSets.front();
    curves.curveVertexCounts = {2, 3, 4};
    curves.curveId = {30, 10, 20};
    curves.points = {{.2f,0,.2f},{.2f,.4f,.2f},
                     {.8f,.4f,.2f},{.8f,.6f,.2f},{.8f,.6f,.5f},
                     {.2f,.2f,0},{.3f,.3f,0},{.3f,.3f,.3f},{.3f,.3f,.6f}};
    curves.rest = curves.points;
    curves.skinPrim = {1, 2, 0};
    curves.skinPrimUv = {{.2f,.2f},{.2f,.2f},{.2f,.2f}};
    auto length = Length("/Dag/Length", "/Dag/Source");
    auto terminal = Width("/Dag/TopologyTerminal", "/Dag/Length", .25f);
    auto sibling = Width("/Dag/TopologySibling", "/Dag/Length", .75f);
    // Shuffled authored storage proves lowering follows explicit input values,
    // not vector adjacency. Length is the sole topology-changing trunk and
    // dominates both private Width COW branches.
    desc.nodes = {terminal, Source(), sibling, length};
    desc.terminal = terminal.path;
    return desc;
}

// The Length node is the only topology writer.  Both Width values must see
// its compacted output before their ordered blend is evaluated; this is the
// topology/value fan-in shape that used to be rejected by the CUDA layout.
UsdGenGraphDesc TopologyWidthBlendDesc(float blend = .25f,
                                       bool scale = false,
                                       float cullThreshold = .45f,
                                       bool publishLeft = false) {
    auto desc = TopologyFanoutDesc();
    auto length = Length("/Dag/Length", "/Dag/Source");
    if (scale) {
        length.params = {
            {TfToken("length:mode"), VtValue(TfToken("scale")), false},
            {TfToken("length:value"), VtValue(.5f), false}};
    } else {
        length.params[1].value = VtValue(cullThreshold);
    }
    auto left = Width("/Dag/TopologyBlendLeft", "/Dag/Length", .2f);
    auto right = Width("/Dag/TopologyBlendRight", "/Dag/Length", .8f);
    UsdGenNodeDesc merge;
    merge.path = SdfPath("/Dag/TopologyBlend");
    merge.type = TfToken("UsdGenWidthBlend");
    merge.inputs = {left.path, right.path};
    merge.params.push_back({TfToken("widthBlend:weight"), VtValue(blend), false});
    // Keep authored storage deliberately non-topological: both the topology
    // trunk and the WidthBlend's left/right operand order are semantic.
    desc.nodes = {merge, right, Source(), left, length};
    desc.terminal = publishLeft ? left.path : merge.path;
    return desc;
}

// A blend may consume the topology trunk itself as one ordered value and a
// COW Width sibling as the other.  This is intentionally different from the
// two-Width fan-in above: it proves that the Length output's inherited width
// plane remains a valid immutable operand after a sibling has replaced its
// own plane.
UsdGenGraphDesc TopologyLengthWidthBlendDesc() {
    auto desc = TopologyFanoutDesc();
    auto length = Length("/Dag/Length", "/Dag/Source");
    auto right = Width("/Dag/TopologyBlendRight", "/Dag/Length", .8f);
    UsdGenNodeDesc merge;
    merge.path = SdfPath("/Dag/TopologyLengthBlend");
    merge.type = TfToken("UsdGenWidthBlend");
    // Keep Length on the ordered right-hand input: this is the dominance
    // case that is easiest to accidentally skip when validating fan-in.
    merge.inputs = {right.path, length.path};
    desc.nodes = {merge, right, Source(), length};
    desc.terminal = merge.path;
    return desc;
}

bool CpuReference(UsdGenGraphDesc desc, UsdGenCurveBuffer* output) {
    desc.executionBackend = UsdGenExecutionBackend::CpuReference;
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    auto const compile = compiler.Compile(desc, &graph);
    if (!compile.ok) {
        std::fprintf(stderr, "CPU reference compile failed for %s\n",
                     desc.terminal.GetText());
        for (auto const& error : compile.errors)
            std::fprintf(stderr, "  %s\n", error.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    auto result = scheduler.Run(graph, context, 1);
    if (result.diagnostics.HasErrors()) {
        std::fprintf(stderr, "CPU reference scheduler failed for %s\n",
                     desc.terminal.GetText());
        for (auto const& error : result.diagnostics.errors)
            std::fprintf(stderr, "  %s\n", error.c_str());
        return false;
    }
    *output = NodeBuffer(graph, graph.NodeIdForPath(desc.terminal));
    return true;
}

template <class T>
bool Read(gpu::DeviceView<const T> input, std::vector<T>* output,
          cudaStream_t stream) {
    output->resize(input.size);
    return (!input.size || cudaMemcpyAsync(output->data(), input.data,
               input.size * sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}

// Test-only topology oracle. CPU Length currently collapses culled curves,
// misses ragged scale CVs and lacks cutExtend evaluation. Use CPU
// Source/Width/Blend for their actual
// math, with identity Length, then independently apply the literal topology
// operation. This commutes only for these constant Width/root-map fixtures;
// it is not a production fallback or a general CPU Length parity claim.
bool LengthTopologyReference(UsdGenGraphDesc desc, UsdGenCurveBuffer* output) {
    auto length = std::find_if(desc.nodes.begin(), desc.nodes.end(), [](auto const& node) {
        return node.type == TfToken("UsdGenLength");
    });
    if (length == desc.nodes.end()) return false;
    TfToken mode("scale"), method("scale");
    float value = 1, threshold = 0;
    for (auto const& param : length->params) {
        if (param.name == TfToken("length:mode")) mode = param.value.Get<TfToken>();
        if (param.name == TfToken("length:method")) method = param.value.Get<TfToken>();
        if (param.name == TfToken("length:value")) value = param.value.Get<float>();
        if (param.name == TfToken("cullThreshold")) threshold = param.value.Get<float>();
    }
    length->params = {{TfToken("length:mode"), VtValue(TfToken("scale")), false},
                      {TfToken("length:value"), VtValue(1.f), false}};
    UsdGenCurveBuffer input;
    if (!CpuReference(desc, &input) || input.cvOffsets.size() != input.totalCurves + 1)
        return false;
    UsdGenCurveBuffer result;
    result.cvOffsets.push_back(0);
    for (uint32_t c = 0; c < input.totalCurves; ++c) {
        auto const begin = input.cvOffsets[c], end = input.cvOffsets[c + 1];
        std::vector<GfVec3f> points;
        std::vector<float> arc(1, 0);
        for (int i = begin; i < end; ++i) {
            points.emplace_back(input.px[i], input.py[i], input.pz[i]);
            if (points.size() > 1)
                arc.push_back(arc.back() + (points.back() - points[points.size()-2]).GetLength());
        }
        if (points.empty()) return false;
        if (mode == TfToken("cull") && arc.back() < threshold) continue;
        float const target = arc.back() * value;
        for (size_t j = 0; j < points.size(); ++j) {
            GfVec3f p = points[j];
            if (mode != TfToken("cull")) {
                if (method == TfToken("scale")) p = points.front() + (p - points.front()) * value;
                else {
                    // Fixture uses keepParam and shortening: preserve each
                    // original arc distance until it reaches the clipped tip.
                    if (target > arc.back()) return false;
                    float const distance = std::min(arc[j], target);
                    size_t k = 1;
                    while (k + 1 < arc.size() && arc[k] < distance) ++k;
                    float const span = arc[k] - arc[k-1];
                    p = points[k-1] + (points[k] - points[k-1]) *
                        (span > 0 ? (distance - arc[k-1]) / span : 0);
                }
            }
            size_t const i = size_t(begin) + j;
            result.px.push_back(p[0]); result.py.push_back(p[1]); result.pz.push_back(p[2]);
            result.rest.push_back(input.rest[i]); result.width.push_back(input.width[i]);
        }
        result.curveId.push_back(input.curveId[c]);
        result.cvOffsets.push_back(static_cast<int>(result.px.size()));
    }
    result.totalCurves = static_cast<uint32_t>(result.curveId.size());
    result.totalCvs = static_cast<uint32_t>(result.px.size());
    *output = std::move(result);
    return true;
}

bool Near(float a, float b) {
    return std::fabs(a - b) < 2.e-5f;
}

bool CheckGeneration(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                     UsdGenCurveBuffer const& reference,
                     cudaStream_t stream) {
    auto lease = gpu::AcquireGeometry(generation, stream);
    if (!lease || lease.Geometry().curveCount != reference.totalCurves ||
        lease.Geometry().pointCount != reference.totalCvs)
        return false;
    std::vector<float3> points, rest;
    std::vector<float> widths;
    std::vector<uint64_t> ids;
    std::vector<uint32_t> offsets;
    if (!Read(lease.Geometry().points, &points, stream) ||
        !Read(lease.Geometry().restPoints, &rest, stream) ||
        !Read(lease.Geometry().widths, &widths, stream) ||
        !Read(lease.Geometry().stableIds, &ids, stream) ||
        !Read(lease.Geometry().curveOffsets, &offsets, stream))
        return false;
    if (points.size() != reference.totalCvs || rest.size() != reference.totalCvs ||
        widths.size() != reference.totalCvs || ids.size() != reference.totalCurves ||
        offsets.size() != size_t(reference.totalCurves) + 1 ||
        ids != std::vector<uint64_t>(reference.curveId.begin(), reference.curveId.end()))
        return false;
    std::vector<uint32_t> expectedOffsets(reference.totalCurves + 1, 0);
    if (!reference.cvOffsets.empty()) {
        if (reference.cvOffsets.size() != expectedOffsets.size())
            return false;
        for (size_t i = 0; i != expectedOffsets.size(); ++i) {
            if (reference.cvOffsets[i] < 0)
                return false;
            expectedOffsets[i] = static_cast<uint32_t>(reference.cvOffsets[i]);
        }
    } else if (reference.totalCurves) {
        if (reference.totalCvs % reference.totalCurves)
            return false;
        uint32_t const perCurve = reference.totalCvs / reference.totalCurves;
        for (uint32_t curve = 0; curve <= reference.totalCurves; ++curve)
            expectedOffsets[curve] = curve * perCurve;
    }
    if (offsets != expectedOffsets)
        return false;
    for (size_t i = 0; i != points.size(); ++i) {
        if (!Near(points[i].x, reference.px[i]) ||
            !Near(points[i].y, reference.py[i]) ||
            !Near(points[i].z, reference.pz[i]) ||
            !Near(rest[i].x, reference.rest[i][0]) ||
            !Near(rest[i].y, reference.rest[i][1]) ||
            !Near(rest[i].z, reference.rest[i][2]) ||
            !Near(widths[i], reference.width[i])) {
            std::fprintf(stderr, "point %zu: GPU P=(%g,%g,%g) rest=(%g,%g,%g) w=%g; expected P=(%g,%g,%g) rest=(%g,%g,%g) w=%g\n",
                i, points[i].x, points[i].y, points[i].z, rest[i].x, rest[i].y, rest[i].z, widths[i],
                reference.px[i], reference.py[i], reference.pz[i], reference.rest[i][0], reference.rest[i][1], reference.rest[i][2], reference.width[i]);
            return false;
        }
    }
    return true;
}

bool CheckSourcePayload(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                        UsdGenCurveBuffer const& reference, cudaStream_t stream) {
    if (!CheckGeneration(generation, reference, stream)) return false;
    auto lease = gpu::AcquireGeometry(generation, stream);
    std::vector<float> hair;
    std::vector<int32_t> prim;
    std::vector<float2> uv;
    if (!lease || !Read(lease.HairT(), &hair, stream) ||
        !Read(lease.RootPrim(), &prim, stream) || !Read(lease.RootUV(), &uv, stream) ||
        hair.size() != reference.hairT.size() || uv.size() != reference.rootUV.size() ||
        prim != std::vector<int32_t>(reference.rootPrim.begin(), reference.rootPrim.end())) return false;
    for (size_t i=0; i<hair.size(); ++i) if (!Near(hair[i], reference.hairT[i])) return false;
    for (size_t i=0; i<uv.size(); ++i)
        if (!Near(uv[i].x, reference.rootUV[i][0]) || !Near(uv[i].y, reference.rootUV[i][1])) return false;
    for (auto const* planes : {&reference.extraCv, &reference.extraCurve})
        for (auto const& plane : *planes) {
            auto channel = gpu::AcquireNamedChannel(generation, plane.name.GetString(), stream);
            if (!channel || plane.type != TfToken("float")) return false;
            auto bytes = channel.Bytes();
            std::vector<float> data(plane.f.size());
            if (bytes.size != data.size()*sizeof(float) ||
                (bytes.size && cudaMemcpyAsync(data.data(), bytes.data, bytes.size,
                    cudaMemcpyDeviceToHost, stream) != cudaSuccess) ||
                cudaStreamSynchronize(stream) != cudaSuccess) return false;
            for (size_t i=0; i<data.size(); ++i) if (!Near(data[i], plane.f[i])) return false;
        }
    return true;
}

UsdGenExecutionTaskMetadata const* TaskByPath(
    UsdGenExecutionPlanMetadata const& metadata, char const* path) {
    auto found = std::find_if(metadata.Tasks().begin(), metadata.Tasks().end(),
        [&](auto const& task) { return task.path == SdfPath(path); });
    return found == metadata.Tasks().end() ? nullptr : &*found;
}

UsdGenExecutionResourceUse const* Use(
    UsdGenExecutionTaskMetadata const& task, UsdGenExecutionDataKind resource) {
    auto found = std::find_if(task.resources.begin(), task.resources.end(),
        [&](auto const& use) { return use.resource == resource; });
    return found == task.resources.end() ? nullptr : &*found;
}

bool DependenciesAre(UsdGenExecutionTaskMetadata const& task,
                     std::initializer_list<uint32_t> values) {
    return task.dependencies == std::vector<uint32_t>(values);
}

// The estimator's source formula is intentionally payload-only: two float3
// point/rest channels plus widths/hairT (32 bytes/CV), compact per-curve
// channels (12 bytes/curve without roots), and one terminal offset.  This
// keeps the COW peak assertion tied to descriptor cardinality rather than to
// a runtime allocation snapshot.
uint64_t ExpectedSourcePayloadBytes(UsdGenGraphDesc const& desc) {
    auto const& curves = desc.curveSets.front();
    bool const roots = !curves.skinPrim.empty() || !curves.skinPrimUv.empty();
    return static_cast<uint64_t>(curves.points.size()) * 32u +
        static_cast<uint64_t>(curves.curveVertexCounts.size()) *
            (roots ? 24u : 12u) + sizeof(uint32_t);
}

uint64_t ExpectedPrivateWidthBytes(UsdGenGraphDesc const& desc) {
    return static_cast<uint64_t>(desc.curveSets.front().points.size()) *
        sizeof(float);
}

size_t ResourceKindBytes(UsdGenExecutionResourceSnapshot const& snapshot,
                         UsdGenExecutionResourceKind kind) {
    return snapshot.byKind[static_cast<size_t>(kind)];
}

bool DiagnosticIs(UsdGenDiagnostics const& diagnostics, char const* expected) {
    return diagnostics.errors.size() == 1 && diagnostics.warnings.empty() &&
        diagnostics.errors.front() == expected;
}

bool ReadWidths(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                std::vector<float>* widths) {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
        return false;
    bool ok = false;
    {
        auto lease = gpu::AcquireGeometry(generation, stream);
        if (lease) {
            auto const geometry = lease.Geometry();
            widths->resize(geometry.widths.size);
            ok = (widths->empty() || cudaMemcpyAsync(widths->data(),
                    geometry.widths.data, widths->size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream) == cudaSuccess) &&
                cudaStreamSynchronize(stream) == cudaSuccess;
        }
    }
    ok = cudaStreamSynchronize(stream) == cudaSuccess && ok;
    cudaStreamDestroy(stream);
    return ok;
}

bool AllWidthsAre(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                  float expected) {
    std::vector<float> widths;
    return ReadWidths(generation, &widths) && !widths.empty() &&
        std::all_of(widths.begin(), widths.end(), [&](float value) {
            return std::fabs(value - expected) < 1e-6f;
        });
}

struct ReleaseWidthBranchGate {
    bool armed = true;
    ~ReleaseWidthBranchGate() {
        if (armed) releaseCudaOperatorAsyncWidthBranchGateForTesting();
    }
    void Release() {
        if (armed) releaseCudaOperatorAsyncWidthBranchGateForTesting();
        armed = false;
    }
};

bool WaitForWidthBranches() {
    try {
        waitCudaOperatorAsyncWidthBranchGateForTesting();
        return true;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Width branch rendezvous failed with %zu streams: %s\n",
            cudaOperatorAsyncWidthDistinctBranchStreamCountForTesting(), error.what());
        return false;
    }
}

void AddRuntimeFailure(UsdGenGraphDesc* desc, SdfPath const& nodePath) {
    UsdGenExpressionDesc expression;
    expression.path = SdfPath("/Dag/FailExpression");
    expression.source = "$frame / 0";
    expr::ValueShape const shape{expr::ScalarType::Float32, 1, 1, 1, 1, false};
    expression.outputs.push_back({TfToken("result"), TfToken("float"), shape});
    desc->expressions.push_back(expression);
    auto found = std::find_if(desc->nodes.begin(), desc->nodes.end(),
        [&](auto const& node) { return node.path == nodePath; });
    UsdGenExpressionBinding binding;
    binding.expression = expression.path;
    binding.destination = TfToken("width");
    binding.domain = expr::Domain::Groom;
    binding.nativeType = TfToken("float");
    binding.destinationShape = shape;
    binding.literal = VtValue(.75f);
    found->expressionBindings.push_back(std::move(binding));
}

// Three roots spread along u so a primitive-domain expression on
// usdGen:mask produces three different envelopes in one execution.
UsdGenGraphDesc MaskExpressionDesc(bool connected) {
    auto desc = BaseDesc();
    auto& curves = desc.curveSets.front();
    curves.curveVertexCounts = {2, 2, 2};
    curves.points = {{0,0,0}, {0,.1f,0},
                     {.5f,0,0}, {.5f,.1f,0},
                     {1,0,0}, {1,.1f,0}};
    curves.rest = curves.points;
    curves.curveId = {7, 8, 9};
    curves.skinPrim = {0, 0, 0};
    curves.skinPrimUv = {{0,0}, {.5f,0}, {1,0}};
    auto width = Width("/Dag/MaskWidth", "/Dag/Source", .5f);
    if (connected) {
        UsdGenExpressionDesc expression;
        expression.path = SdfPath("/Dag/Expressions/rootMask");
        expression.source = "$value * $u";
        expr::ValueShape const shape{expr::ScalarType::Float32, 1, 1, 1, 1, false};
        expression.outputs.push_back({TfToken("result"), TfToken("float"), shape});
        desc.expressions.push_back(expression);
        UsdGenExpressionBinding binding;
        binding.expression = expression.path;
        // An empty output is the prim-path spelling of the same connection.
        binding.output = TfToken();
        binding.destination = TfToken("usdGen:mask");
        binding.domain = expr::Domain::Primitive;
        binding.nativeType = TfToken("float");
        binding.destinationShape = shape;
        binding.literal = VtValue(1.0f);
        width.expressionBindings.push_back(std::move(binding));
    } else {
        width.params.push_back({TfToken("mask"), VtValue(0.0f), false});
    }
    desc.nodes = {Source(), width};
    desc.terminal = width.path;
    return desc;
}

} // namespace

int main() {
    UsdGenDiagnostics diagnostics;

    // usdGen:mask IS the operator envelope and accepts an expression
    // connection: the per-curve result varies with $u, and an exact-zero
    // literal leaves the incoming widths bit-identical.
    {
        auto connectedDesc = MaskExpressionDesc(true);
        auto connectedPlan = CompileCudaGraph(connectedDesc, &diagnostics);
        for (auto const& e : diagnostics.errors) std::fprintf(stderr, "%s\n", e.c_str());
        CHECK(connectedPlan && !diagnostics.HasErrors());
        auto maskWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
        CHECK(maskWorkspace);
        auto connectedGeneration = ExecuteCudaGraph(*connectedPlan, *maskWorkspace,
                                                    1.0, 4201, &diagnostics);
        CHECK(connectedGeneration && !diagnostics.HasErrors());
        std::vector<float> connectedWidths;
        CHECK(ReadWidths(connectedGeneration, &connectedWidths) &&
              connectedWidths.size() == 6);
        // envelope = 1 * u = {0, .5, 1}; replace lerps the source width
        // (the description default) toward the authored .5f target.
        float const input = connectedDesc.defaultWidth;
        float const expected[3] = {input, input + (.5f - input) * .5f, .5f};
        for (size_t curve = 0; curve < 3; ++curve)
            for (size_t cv = 0; cv < 2; ++cv)
                CHECK(std::fabs(connectedWidths[curve * 2 + cv] - expected[curve]) < 1e-6f);
        CHECK(connectedWidths[0] != connectedWidths[2] &&
              connectedWidths[2] != connectedWidths[4]);
        // The zero-envelope curve is the input bit pattern, not arithmetic.
        CHECK(std::memcmp(&connectedWidths[0], &input, sizeof(float)) == 0);

        auto zeroDesc = MaskExpressionDesc(false);
        diagnostics = {};
        auto zeroPlan = CompileCudaGraph(zeroDesc, &diagnostics);
        CHECK(zeroPlan && !diagnostics.HasErrors());
        auto zeroGeneration = ExecuteCudaGraph(*zeroPlan, *maskWorkspace, 1.0,
                                               4202, &diagnostics);
        CHECK(zeroGeneration && !diagnostics.HasErrors());
        std::vector<float> zeroWidths;
        CHECK(ReadWidths(zeroGeneration, &zeroWidths) && zeroWidths.size() == 6);
        for (float value : zeroWidths)
            CHECK(std::memcmp(&value, &input, sizeof(float)) == 0);
    }

    diagnostics = {};
    auto desc = FanoutDesc();
    diagnostics = {};
    auto plan = CompileCudaGraph(desc, &diagnostics);
    CHECK(plan && !diagnostics.HasErrors());
    auto metadata = GetCudaExecutionPlanMetadata(*plan);
    CHECK(metadata && metadata->Shape() == UsdGenExecutionPlanShape::SourceRootedUnaryDag);
    CHECK(metadata->Tasks().size() == 4 && metadata->TerminalTask() == 3);

    auto const* source = TaskByPath(*metadata, "/Dag/Source");
    auto const* terminal = TaskByPath(*metadata, "/Dag/Terminal");
    auto const* sibling = TaskByPath(*metadata, "/Dag/Sibling");
    auto const* publication = metadata->FindTask(metadata->TerminalTask());
    CHECK(source && terminal && sibling && publication);
    CHECK(source->id == 0 && source->semanticNode == 1 &&
          terminal->id == 1 && terminal->semanticNode == 0 &&
          sibling->id == 2 && sibling->semanticNode == 2);
    CHECK(DependenciesAre(*source, {}) && DependenciesAre(*terminal, {0}) &&
          DependenciesAre(*sibling, {0}) &&
          DependenciesAre(*publication, {0,1,2}));
    CHECK(source->exclusiveWorkspace && !terminal->exclusiveWorkspace &&
          !sibling->exclusiveWorkspace && publication->exclusiveWorkspace);
    CHECK(source->resourceHazards.size() == 1 &&
          source->resourceHazards.front().resource ==
              UsdGenExecutionDataKind::ExecutionWorkspace &&
          source->resourceHazards.front().identity == 0 &&
          source->resourceHazards.front().access ==
              UsdGenExecutionResourceHazardAccess::ReadWrite);
    CHECK(terminal->resourceHazards.empty() && sibling->resourceHazards.empty());
    CHECK(publication->resourceHazards.size() == 1 &&
          publication->resourceHazards.front().resource ==
              UsdGenExecutionDataKind::ExecutionWorkspace &&
          publication->resourceHazards.front().identity == 0 &&
          publication->resourceHazards.front().access ==
              UsdGenExecutionResourceHazardAccess::ReadWrite);
    CHECK(publication->declaredDependencies.size() == 3 &&
          publication->dependencyProvenance.size() == 3);
    for (uint32_t i = 0; i != 3; ++i)
        CHECK(publication->declaredDependencies[i].predecessor == i &&
              publication->declaredDependencies[i].provenance ==
                  UsdGenExecutionDependencyLifetimePublicationJoin);
    auto const* widthCapability = GetCudaExecutionCapabilityMatrix().Find(
        TfToken("UsdGenWidth"));
    CHECK(widthCapability &&
          (widthCapability->flags & UsdGenCapabilityCopyOnWriteWrites) != 0 &&
          (widthCapability->flags & UsdGenCapabilityExclusiveWorkspace) == 0);

    // Narrow ordered binary fan-in: the blend task has two real logical
    // Width inputs, so dependency lowering must wait for both branch values
    // rather than treating the right operand as an executor-private side
    // channel.  Its output is another immutable width-only COW value.
    auto blendDesc = WidthBlendDesc();
    diagnostics = {};
    auto blendPlan = CompileCudaGraph(blendDesc, &diagnostics);
    CHECK(blendPlan && !diagnostics.HasErrors());
    auto blendMetadata = GetCudaExecutionPlanMetadata(*blendPlan);
    auto const* blendTask = blendMetadata
        ? TaskByPath(*blendMetadata, "/Dag/Blend") : nullptr;
    auto const* blendSource = blendMetadata
        ? TaskByPath(*blendMetadata, "/Dag/Source") : nullptr;
    auto const* blendLeft = blendMetadata
        ? TaskByPath(*blendMetadata, "/Dag/BlendLeft") : nullptr;
    auto const* blendRight = blendMetadata
        ? TaskByPath(*blendMetadata, "/Dag/BlendRight") : nullptr;
    CHECK(blendMetadata && blendTask && blendSource && blendLeft && blendRight);
    CHECK(blendMetadata->Shape() == UsdGenExecutionPlanShape::SourceRootedValueDag);
    auto blendDependencies = blendTask->dependencies;
    std::sort(blendDependencies.begin(), blendDependencies.end());
    std::array<uint32_t, 3> expectedBlendDependencies{
        blendSource->id, blendLeft->id, blendRight->id};
    std::sort(expectedBlendDependencies.begin(), expectedBlendDependencies.end());
    CHECK(blendDependencies == std::vector<uint32_t>(
        expectedBlendDependencies.begin(), expectedBlendDependencies.end()));
    CHECK(!blendTask->exclusiveWorkspace);
    CHECK(blendTask->estimate.retainedOutputBytes == ExpectedPrivateWidthBytes(blendDesc));
    auto const blendWidthUses = std::count_if(blendTask->resources.begin(),
        blendTask->resources.end(), [](auto const& use) {
            return use.resource == UsdGenExecutionDataKind::Widths;
        });
    CHECK(blendWidthUses == 2);
    auto const blendLeftUse = std::find_if(blendTask->resources.begin(),
        blendTask->resources.end(), [](auto const& use) {
            return use.resource == UsdGenExecutionDataKind::Widths &&
                use.access == UsdGenExecutionResourceAccess::ReadWrite;
        });
    auto const blendRightUse = std::find_if(blendTask->resources.begin(),
        blendTask->resources.end(), [](auto const& use) {
            return use.resource == UsdGenExecutionDataKind::Widths &&
                use.access == UsdGenExecutionResourceAccess::Read;
        });
    CHECK(blendLeftUse != blendTask->resources.end() &&
          blendRightUse != blendTask->resources.end() &&
          blendLeftUse->producerTask == blendLeft->id &&
          blendRightUse->producerTask == blendRight->id);
    auto const* blendCapability = GetCudaExecutionCapabilityMatrix().Find(
        TfToken("UsdGenWidthBlend"));
    CHECK(blendCapability &&
          (blendCapability->flags & UsdGenCapabilityCopyOnWriteWrites) != 0);

    // A resolved reference is an immutable source payload, not a host-side
    // fallback.  The narrow CUDA contract is ReferenceSource followed by one
    // or more topology-preserving Width operators; Width owns only its
    // private overlay while the source generation remains retained.
    auto referenceDesc = ReferenceWidthDesc();
    diagnostics = {};
    auto referencePlan = CompileCudaGraph(referenceDesc, &diagnostics);
    CHECK(referencePlan && !diagnostics.HasErrors());
    auto referenceMetadata = GetCudaExecutionPlanMetadata(*referencePlan);
    CHECK(referenceMetadata &&
          referenceMetadata->Shape() == UsdGenExecutionPlanShape::SourceRootedUnaryDag &&
          referenceMetadata->Tasks().size() == 3);
    auto const* referenceSourceTask = TaskByPath(
        *referenceMetadata, "/Dag/ReferenceSource");
    auto const* referenceWidthTask = TaskByPath(
        *referenceMetadata, "/Dag/ReferenceWidth");
    CHECK(referenceSourceTask && referenceWidthTask &&
          referenceSourceTask->kind == UsdGenExecutionTaskKind::Source &&
          referenceWidthTask->kind == UsdGenExecutionTaskKind::Operator &&
          DependenciesAre(*referenceSourceTask, {}) &&
          DependenciesAre(*referenceWidthTask, {referenceSourceTask->id}));
    CHECK(referenceMetadata->MemoryEstimate().memoryAvailable &&
          referenceMetadata->MemoryEstimate().immutableSharedInputBytes ==
              ExpectedSourcePayloadBytes(referenceDesc));
    auto referenceWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(referenceWorkspace);
    auto referenceGeneration = ExecuteCudaGraph(
        *referencePlan, *referenceWorkspace, 1.0, 100, &diagnostics);
    CHECK(referenceGeneration && !diagnostics.HasErrors() &&
          AllWidthsAre(referenceGeneration, .25f) &&
          referenceGeneration->Owner() &&
          referenceGeneration->InclusiveRetainedBytes() > 0);
    auto editedReferenceDesc = ReferenceWidthDesc(.5f);
    diagnostics = {};
    auto editedReferencePlan = CompileCudaGraph(
        editedReferenceDesc, &diagnostics);
    CHECK(editedReferencePlan && !diagnostics.HasErrors());
    auto editedReferenceGeneration = ExecuteCudaGraph(
        *editedReferencePlan, *referenceWorkspace, 1.0, 101, &diagnostics,
        referenceGeneration);
    CHECK(editedReferenceGeneration && !diagnostics.HasErrors() &&
          AllWidthsAre(editedReferenceGeneration, .5f) &&
          editedReferenceGeneration->Owner() != referenceGeneration->Owner() &&
          referenceGeneration->Owner() &&
          referenceGeneration->InclusiveRetainedBytes() > 0);

    // ReferenceSource may also feed the one allowed topology trunk. The
    // reference has root channels but no local surface descriptor; literal
    // Length admission must still account for and preserve those channels
    // through compaction before Width branches run.
    auto referenceLengthDesc = ReferenceLengthWidthDesc();
    diagnostics = {};
    auto referenceLengthPlan = CompileCudaGraph(
        referenceLengthDesc, &diagnostics);
    CHECK(referenceLengthPlan && !diagnostics.HasErrors());
    auto referenceLengthMetadata = GetCudaExecutionPlanMetadata(
        *referenceLengthPlan);
    CHECK(referenceLengthMetadata &&
          referenceLengthMetadata->Shape() ==
              UsdGenExecutionPlanShape::SourceRootedUnaryDag &&
          referenceLengthMetadata->Tasks().size() == 5 &&
          referenceLengthMetadata->MemoryEstimate().runtimeRefinementAvailable);
    auto const* referenceLengthSourceTask = TaskByPath(
        *referenceLengthMetadata, "/Dag/ReferenceSource");
    auto const* referenceLengthTask = TaskByPath(
        *referenceLengthMetadata, "/Dag/ReferenceLength");
    auto const* referenceLengthWidthTask = TaskByPath(
        *referenceLengthMetadata, "/Dag/ReferenceLengthWidth");
    auto const* referenceLengthSiblingTask = TaskByPath(
        *referenceLengthMetadata, "/Dag/ReferenceLengthSibling");
    CHECK(referenceLengthSourceTask && referenceLengthTask &&
          referenceLengthWidthTask && referenceLengthSiblingTask &&
          DependenciesAre(*referenceLengthTask,
              {referenceLengthSourceTask->id}) &&
          DependenciesAre(*referenceLengthWidthTask,
              {referenceLengthTask->id}) &&
          DependenciesAre(*referenceLengthSiblingTask,
              {referenceLengthTask->id}) &&
          referenceLengthTask->exclusiveWorkspace == true &&
          !referenceLengthWidthTask->exclusiveWorkspace &&
          !referenceLengthSiblingTask->exclusiveWorkspace);
    auto referenceLengthWorkspace = CreateCudaExecutionWorkspace(
        -1, &diagnostics);
    CHECK(referenceLengthWorkspace);
    auto referenceLengthGeneration = ExecuteCudaGraph(
        *referenceLengthPlan, *referenceLengthWorkspace, 1.0, 102, &diagnostics);
    CHECK(referenceLengthGeneration && !diagnostics.HasErrors() &&
          AllWidthsAre(referenceLengthGeneration, .75f));
    {
        cudaStream_t reader = nullptr;
        CHECK(cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking) ==
              cudaSuccess);
        {
            auto lease = gpu::AcquireGeometry(referenceLengthGeneration,
                                               reader);
            CHECK(lease && lease.Geometry().curveCount == 1 &&
                  lease.Geometry().pointCount == 2 &&
                  lease.RootPrim().size == 1 && lease.RootUV().size == 1);
            std::array<int32_t, 1> rootPrim{};
            CHECK(cudaMemcpyAsync(rootPrim.data(), lease.RootPrim().data,
                                  sizeof(rootPrim), cudaMemcpyDeviceToHost,
                                  reader) == cudaSuccess &&
                  cudaStreamSynchronize(reader) == cudaSuccess &&
                  rootPrim[0] == 0);
        }
        CHECK(cudaStreamSynchronize(reader) == cudaSuccess &&
              cudaStreamDestroy(reader) == cudaSuccess);
    }

    // ReferenceSource remains constrained to Width-consuming terminals; its
    // source-side Width branch above is an immutable snapshot, not a request
    // to publish Source or Length by themselves.
    auto referenceLengthOnly = ReferenceLengthWidthDesc();
    referenceLengthOnly.nodes.resize(2);
    referenceLengthOnly.terminal = referenceLengthOnly.nodes.back().path;
    diagnostics = {};
    CHECK(!CompileCudaGraph(referenceLengthOnly, &diagnostics) &&
          diagnostics.HasErrors());
    auto referenceMultipleLengths = ReferenceLengthWidthDesc();
    auto secondReferenceLength = Length(
        "/Dag/ReferenceLength2", "/Dag/ReferenceLength");
    secondReferenceLength.params={
        {TfToken("length:mode"),VtValue(TfToken("scale")),false},
        {TfToken("length:value"),VtValue(.5f),false}};
    referenceMultipleLengths.nodes.insert(
        referenceMultipleLengths.nodes.begin() + 2,
        secondReferenceLength);
    referenceMultipleLengths.nodes[3].inputs = {
        SdfPath("/Dag/ReferenceLength2")};
    // Multiple Length values now use independent compaction bundles. Their
    // full selected payload is checked below alongside the source-side branch.
    // A source-side Width is now a first-class immutable value branch. It
    // must retain the original ReferenceSource payload while the direct
    // Length trunk owns an independently scaled topology snapshot.
    auto referenceMisplacedWidth = ReferenceLengthWidthDesc();
    auto sourceWidth = Width("/Dag/ReferenceSourceWidth",
                             "/Dag/ReferenceSource", .125f);
    referenceMisplacedWidth.nodes.push_back(sourceWidth);
    referenceMisplacedWidth.terminal = sourceWidth.path;
    diagnostics = {};
    auto referenceSourceBranchPlan = CompileCudaGraph(referenceMisplacedWidth,
                                                       &diagnostics);
    CHECK(referenceSourceBranchPlan && !diagnostics.HasErrors());
    UsdGenCurveBuffer referenceSourceBranch;
    CHECK(CpuReference(referenceMisplacedWidth, &referenceSourceBranch));
    cudaStream_t referenceBranchReader = nullptr;
    CHECK(cudaStreamCreateWithFlags(&referenceBranchReader,
                                    cudaStreamNonBlocking) == cudaSuccess);
    auto referenceSourceBranchDirect = ExecuteCudaGraph(*referenceSourceBranchPlan,
        *referenceLengthWorkspace, 1.0, 103, &diagnostics);
    CHECK(referenceSourceBranchDirect && !diagnostics.HasErrors() &&
          CheckSourcePayload(referenceSourceBranchDirect, referenceSourceBranch,
                             referenceBranchReader));
    auto hasNoPublishedFrames = [&](auto const& generation) {
        auto lease = gpu::AcquireGeometry(generation, referenceBranchReader);
        return lease && !lease.RootT().size && !lease.RootB().size &&
            !lease.RootN().size;
    };
    CHECK(hasNoPublishedFrames(referenceSourceBranchDirect));

    auto referenceLengthTerminal = referenceMisplacedWidth;
    referenceLengthTerminal.terminal = SdfPath("/Dag/ReferenceLengthWidth");
    diagnostics = {};
    auto referenceLengthBranchPlan = CompileCudaGraph(referenceLengthTerminal,
                                                       &diagnostics);
    CHECK(referenceLengthBranchPlan && !diagnostics.HasErrors());
    // This scale fixture preserves cardinality. Build the topology result
    // from the complete source-side payload so roots/hairT remain part of
    // the expected transformed generation rather than using the generic
    // topology oracle (which intentionally only models geometry planes).
    auto referenceLengthBranch = referenceSourceBranch;
    std::vector<int> referenceOffsets;
    if (referenceLengthBranch.cvOffsets.empty()) {
        CHECK(referenceLengthBranch.totalCurves > 0 &&
              referenceLengthBranch.totalCvs % referenceLengthBranch.totalCurves == 0);
        auto const perCurve = referenceLengthBranch.totalCvs /
            referenceLengthBranch.totalCurves;
        referenceOffsets.resize(referenceLengthBranch.totalCurves + 1);
        for (uint32_t curve = 0; curve <= referenceLengthBranch.totalCurves;
             ++curve)
            referenceOffsets[curve] = static_cast<int>(curve * perCurve);
    } else {
        CHECK(referenceLengthBranch.cvOffsets.size() ==
              size_t(referenceLengthBranch.totalCurves) + 1);
        referenceOffsets.assign(referenceLengthBranch.cvOffsets.begin(),
                                referenceLengthBranch.cvOffsets.end());
    }
    for (uint32_t curve = 0; curve < referenceLengthBranch.totalCurves;
         ++curve) {
        CHECK(referenceOffsets[curve] >= 0 && referenceOffsets[curve + 1] >=
              referenceOffsets[curve]);
        auto const begin = static_cast<size_t>(referenceOffsets[curve]);
        auto const end = static_cast<size_t>(referenceOffsets[curve + 1]);
        CHECK(begin < end && end <= referenceLengthBranch.px.size());
        GfVec3f const root(referenceLengthBranch.px[begin],
                           referenceLengthBranch.py[begin],
                           referenceLengthBranch.pz[begin]);
        for (size_t cv = begin; cv < end; ++cv) {
            GfVec3f const current(referenceLengthBranch.px[cv],
                                  referenceLengthBranch.py[cv],
                                  referenceLengthBranch.pz[cv]);
            auto const scaled = root + (current - root) * .5f;
            referenceLengthBranch.px[cv] = scaled[0];
            referenceLengthBranch.py[cv] = scaled[1];
            referenceLengthBranch.pz[cv] = scaled[2];
            referenceLengthBranch.width[cv] = .75f;
        }
    }
    auto referenceLengthBranchDirect = ExecuteCudaGraph(*referenceLengthBranchPlan,
        *referenceLengthWorkspace, 1.0, 104, &diagnostics,
        referenceSourceBranchDirect);
    CHECK(referenceLengthBranchDirect && !diagnostics.HasErrors() &&
          CheckSourcePayload(referenceLengthBranchDirect, referenceLengthBranch,
                             referenceBranchReader));
    CHECK(hasNoPublishedFrames(referenceLengthBranchDirect));

    // Session publication repeats the terminal swap while retaining the old
    // lease: source-side Width must still expose the original points/roots
    // after Length has published its independent topology generation.
    UsdGenSession referenceBranchSession;
    referenceBranchSession.SetDevicePublicationEnabled(true);
    referenceBranchSession.SetGraphDesc(referenceMisplacedWidth);
    auto referenceSourcePublished = referenceBranchSession.Commit(
        1, UsdGenCommitReason::SetTime);
    CHECK(referenceSourcePublished && referenceSourcePublished->device &&
          !referenceBranchSession.LastDiagnostics().HasErrors() &&
          CheckSourcePayload(referenceSourcePublished->device,
                             referenceSourceBranch, referenceBranchReader) &&
          hasNoPublishedFrames(referenceSourcePublished->device));
    referenceBranchSession.SetGraphDesc(referenceLengthTerminal);
    auto referenceLengthPublished = referenceBranchSession.Commit(
        2, UsdGenCommitReason::SetTime);
    CHECK(referenceLengthPublished && referenceLengthPublished->device &&
          !referenceBranchSession.LastDiagnostics().HasErrors() &&
          CheckSourcePayload(referenceLengthPublished->device,
                             referenceLengthBranch, referenceBranchReader) &&
          CheckSourcePayload(referenceSourcePublished->device,
                             referenceSourceBranch, referenceBranchReader) &&
          hasNoPublishedFrames(referenceLengthPublished->device) &&
          hasNoPublishedFrames(referenceSourcePublished->device));
    diagnostics = {};
    auto referenceMultiplePlan=CompileCudaGraph(referenceMultipleLengths,&diagnostics);
    CHECK(referenceMultiplePlan&&!diagnostics.HasErrors());
    auto referenceMultipleExpected=referenceLengthBranch;
    for(uint32_t curve=0;curve<referenceMultipleExpected.totalCurves;++curve) {
        size_t const begin=static_cast<size_t>(referenceOffsets[curve]);
        size_t const end=static_cast<size_t>(referenceOffsets[curve+1]);
        GfVec3f const root(referenceMultipleExpected.px[begin],referenceMultipleExpected.py[begin],
                           referenceMultipleExpected.pz[begin]);
        for(size_t cv=begin;cv<end;++cv) {
            GfVec3f const current(referenceMultipleExpected.px[cv],referenceMultipleExpected.py[cv],
                                  referenceMultipleExpected.pz[cv]);
            auto const scaled=root+(current-root)*.5f;
            referenceMultipleExpected.px[cv]=scaled[0];referenceMultipleExpected.py[cv]=scaled[1];
            referenceMultipleExpected.pz[cv]=scaled[2];
        }
    }
    auto referenceMultipleDirect=ExecuteCudaGraph(*referenceMultiplePlan,*referenceLengthWorkspace,
        1.0,105,&diagnostics,referenceSourceBranchDirect);
    CHECK(referenceMultipleDirect&&!diagnostics.HasErrors()&&
          CheckSourcePayload(referenceMultipleDirect,referenceMultipleExpected,referenceBranchReader)&&
          hasNoPublishedFrames(referenceMultipleDirect));
    referenceBranchSession.SetGraphDesc(referenceMultipleLengths);
    auto referenceMultiplePublished=referenceBranchSession.Commit(3,UsdGenCommitReason::SetTime);
    CHECK(referenceMultiplePublished&&referenceMultiplePublished->device&&
          !referenceBranchSession.LastDiagnostics().HasErrors()&&
          CheckSourcePayload(referenceMultiplePublished->device,referenceMultipleExpected,referenceBranchReader)&&
          hasNoPublishedFrames(referenceMultiplePublished->device)&&
          CheckSourcePayload(referenceSourcePublished->device,referenceSourceBranch,referenceBranchReader));
    CHECK(cudaStreamSynchronize(referenceBranchReader) == cudaSuccess &&
          cudaStreamDestroy(referenceBranchReader) == cudaSuccess);

    auto referenceLength = ReferenceWidthDesc();
    referenceLength.nodes[1].type = TfToken("UsdGenLength");
    referenceLength.nodes[1].params.clear();
    referenceLength.nodes[1].params.push_back(
        {TfToken("length:value"), VtValue(.5f), false});
    diagnostics = {};
    CHECK(!CompileCudaGraph(referenceLength, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: UsdGenReferenceSource CUDA lowering requires one or more Width consumers"));
    auto referenceTerminal = ReferenceWidthDesc();
    referenceTerminal.nodes.resize(1);
    referenceTerminal.terminal = referenceTerminal.nodes.front().path;
    diagnostics = {};
    CHECK(!CompileCudaGraph(referenceTerminal, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: UsdGenReferenceSource CUDA lowering requires one or more Width consumers"));
    auto referencePlanes = ReferenceWidthDesc();
    UsdGenAuthoredPlaneDesc plane;
    plane.name = TfToken("guideWeight");
    plane.type = UsdGenAuthoredPlaneType::Float32;
    plane.domain = UsdGenAuthoredPlaneDomain::Point;
    plane.arity = 1;
    plane.floatValues = {1.f, 1.f};
    referencePlanes.curveSets.front().authoredPlanes.push_back(plane);
    diagnostics = {};
    CHECK(!CompileCudaGraph(referencePlanes, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: UsdGenReferenceSource requires one authored reference curve set without named planes at /Dag/ReferenceSource"));
    auto referenceRootFrames = ReferenceWidthDesc();
    referenceRootFrames.curveSets.front().rootFrame.push_back(GfMatrix4d(1.0));
    diagnostics = {};
    CHECK(!CompileCudaGraph(referenceRootFrames, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: UsdGenReferenceSource rootFrame bindings are not supported by CUDA at /Dag/ReferenceSource"));

    auto const* sourceWidths = Use(*source, UsdGenExecutionDataKind::Widths);
    auto const* terminalWidths = Use(*terminal, UsdGenExecutionDataKind::Widths);
    auto const* siblingWidths = Use(*sibling, UsdGenExecutionDataKind::Widths);
    auto const* publishedWidths = Use(*publication, UsdGenExecutionDataKind::Widths);
    CHECK(sourceWidths && terminalWidths && siblingWidths && publishedWidths);
    CHECK(terminalWidths->producerTask == source->id &&
          siblingWidths->producerTask == source->id &&
          terminalWidths->inputValue == sourceWidths->outputValue &&
          siblingWidths->inputValue == sourceWidths->outputValue);
    CHECK(publishedWidths->producerTask == terminal->id &&
          publishedWidths->inputValue == terminalWidths->outputValue);
    CHECK(metadata->FindValue(terminalWidths->outputValue)->storage ==
              UsdGenExecutionValueStorage::JobOwnedImmutable &&
          metadata->FindValue(siblingWidths->outputValue)->storage ==
              UsdGenExecutionValueStorage::JobOwnedImmutable &&
          terminalWidths->outputValue != siblingWidths->outputValue);
    auto const sourcePayload = ExpectedSourcePayloadBytes(desc);
    auto const privateWidth = ExpectedPrivateWidthBytes(desc);
    auto const& graphEstimate = metadata->MemoryEstimate();
    // CUDA may decline a complete estimate (for example when a dynamic
    // expression controls output cardinality), but when it publishes one it
    // must count the immutable source once and both distinct CoW branch
    // outputs. This is metadata evidence only; it is not admission enforcement.
    CHECK(graphEstimate.memoryAvailable);
    {
        CHECK(graphEstimate.conservativeUpperBound &&
              graphEstimate.immutableSharedInputBytes == sourcePayload &&
              graphEstimate.concurrentPeakBytes ==
                  sourcePayload +
                  2 * (privateWidth + privateWidth + 257 * sizeof(float) +
                       2 * sizeof(int)));
        CHECK(source->estimate.memoryAvailable &&
              terminal->estimate.memoryAvailable &&
              sibling->estimate.memoryAvailable);
        CHECK(source->estimate.retainedOutputBytes == sourcePayload &&
              source->estimate.producerRetentionBytes == 0 &&
              source->estimate.scratchPeakBytes == 0);
        CHECK(terminal->estimate.retainedOutputBytes == privateWidth &&
              sibling->estimate.retainedOutputBytes == privateWidth &&
              terminal->estimate.producerRetentionBytes == sourcePayload &&
              sibling->estimate.producerRetentionBytes == sourcePayload);
        CHECK(terminal->estimate.scratchPeakBytes ==
                  privateWidth + 257 * sizeof(float) + 2 * sizeof(int) &&
              sibling->estimate.scratchPeakBytes ==
                  privateWidth + 257 * sizeof(float) + 2 * sizeof(int));
        CHECK(publication->estimate.memoryAvailable &&
              publication->estimate.retainedOutputBytes == 0 &&
              publication->estimate.producerRetentionBytes == 0 &&
              publication->estimate.scratchPeakBytes == 32 + 116);
    }
    for (auto const& use : source->resources) {
        if (use.outputValue != UINT32_MAX)
            CHECK(metadata->FindValue(use.outputValue)->storage ==
                  UsdGenExecutionValueStorage::JobOwnedImmutable);
    }
    auto const* terminalGeometry = Use(
        *terminal, UsdGenExecutionDataKind::ElementGeometry);
    auto const* siblingGeometry = Use(
        *sibling, UsdGenExecutionDataKind::ElementGeometry);
    CHECK(terminalGeometry && siblingGeometry &&
          terminalGeometry->inputValue == siblingGeometry->inputValue &&
          metadata->FindValue(terminalGeometry->inputValue)->storage ==
              UsdGenExecutionValueStorage::JobOwnedImmutable);

    // Recompiling the same shuffled descriptor yields the same normalized
    // task order and dependency edges.
    UsdGenDiagnostics repeatDiagnostics;
    auto repeatPlan = CompileCudaGraph(desc, &repeatDiagnostics);
    CHECK(repeatPlan && !repeatDiagnostics.HasErrors());
    auto repeatMetadata = GetCudaExecutionPlanMetadata(*repeatPlan);
    CHECK(repeatMetadata && repeatMetadata->Tasks().size() == metadata->Tasks().size());
    CHECK(repeatMetadata->MemoryEstimate().concurrentPeakBytes ==
              metadata->MemoryEstimate().concurrentPeakBytes &&
          repeatMetadata->MemoryEstimate().immutableSharedInputBytes ==
              metadata->MemoryEstimate().immutableSharedInputBytes &&
          repeatMetadata->MemoryEstimate().memoryAvailable ==
              metadata->MemoryEstimate().memoryAvailable &&
          repeatMetadata->MemoryEstimate().conservativeUpperBound ==
              metadata->MemoryEstimate().conservativeUpperBound);
    for (size_t i = 0; i != metadata->Tasks().size(); ++i) {
        CHECK(metadata->Tasks()[i].path == repeatMetadata->Tasks()[i].path &&
              metadata->Tasks()[i].dependencies == repeatMetadata->Tasks()[i].dependencies);
    }

    diagnostics = {};
    auto workspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(workspace);
    auto generation = ExecuteCudaGraph(*plan, *workspace, 1.0, 1, &diagnostics);
    CHECK(generation && !diagnostics.HasErrors() && AllWidthsAre(generation, .25f));

    // Synchronous direct execution uses the same conservative graph peak as
    // staged admission.  Fill the shared pool so exactly peak-1 bytes remain;
    // rejection must happen before Source/Width can allocate and must not
    // perturb either the total or the reservation-only category.  The
    // baseline intentionally includes the retained first generation.
    int admissionDevice = -1;
    CHECK(cudaGetDevice(&admissionDevice) == cudaSuccess);
    auto admissionPool = FindUsdGenExecutionResourcePool(
        {UsdGenExecutionResourceBackend::Cuda, admissionDevice});
    CHECK(admissionPool);
    auto const admissionEstimate = metadata->MemoryEstimate();
    CHECK(admissionEstimate.memoryAvailable &&
          admissionEstimate.conservativeUpperBound &&
          admissionEstimate.concurrentPeakBytes > 0 &&
          admissionEstimate.concurrentPeakBytes <=
              static_cast<uint64_t>(std::numeric_limits<size_t>::max()));
    auto const admissionPeak = static_cast<size_t>(
        admissionEstimate.concurrentPeakBytes);
    auto const admissionBaseline = admissionPool->Snapshot();
    CHECK(admissionBaseline.usedBytes <= admissionBaseline.usableBytes &&
          admissionBaseline.usableBytes - admissionBaseline.usedBytes >=
              admissionPeak);
    auto const fillerBytes = admissionBaseline.usableBytes -
        admissionBaseline.usedBytes - admissionPeak + 1;
    auto filler = admissionPool->TryReserve(
        fillerBytes, UsdGenExecutionResourceKind::Active);
    CHECK(filler && filler->Bytes() == fillerBytes);
    auto const saturated = admissionPool->Snapshot();
    CHECK(saturated.usedBytes == admissionBaseline.usedBytes + fillerBytes &&
          ResourceKindBytes(saturated, UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(admissionBaseline,
                                UsdGenExecutionResourceKind::Pending));
    diagnostics = {};
    auto rejectedDirect = ExecuteCudaGraph(
        *plan, *workspace, 1.0, 2, &diagnostics, generation);
    auto const afterRejected = admissionPool->Snapshot();
    CHECK(!rejectedDirect && DiagnosticIs(diagnostics,
        "CUDA: CUDA execution job memory reservation exceeds the available device budget"));
    CHECK(afterRejected.usedBytes == saturated.usedBytes &&
          ResourceKindBytes(afterRejected, UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(saturated, UsdGenExecutionResourceKind::Pending));
    filler->Release();
    CHECK(admissionPool->Snapshot().usedBytes == admissionBaseline.usedBytes);

    // Releasing only the test filler admits the same direct graph.  Pending
    // is fully retired at return, while the new generation's owner keeps its
    // source and terminal Width output charged.  The delta is bounded by the
    // one graph peak, proving COW siblings were not charged twice.
    diagnostics = {};
    auto admittedDirect = ExecuteCudaGraph(
        *plan, *workspace, 1.0, 3, &diagnostics, generation);
    CHECK(admittedDirect && !diagnostics.HasErrors() &&
          AllWidthsAre(admittedDirect, .25f) && admittedDirect->Owner() &&
          admittedDirect->Owner() != generation->Owner());
    auto const afterAdmitted = admissionPool->Snapshot();
    CHECK(ResourceKindBytes(afterAdmitted, UsdGenExecutionResourceKind::Pending) == 0 &&
          afterAdmitted.usedBytes >= admissionBaseline.usedBytes + sourcePayload + privateWidth &&
          afterAdmitted.usedBytes <= admissionBaseline.usedBytes + admissionPeak);

    // A literal resample target keeps the Source/Width graph's complete peak
    // statically known.  Reuse the existing compact fixture to prove the
    // direct reservation path for the resampled source as well.
    auto staticResampleDesc = BaseDesc();
    auto staticResampleSource = Source();
    staticResampleSource.params.push_back(
        {TfToken("resampleTo"), VtValue(4), false});
    auto staticResampleWidth = Width(
        "/Dag/ResampledWidth", "/Dag/Source", .25f);
    staticResampleDesc.nodes = {staticResampleSource, staticResampleWidth};
    staticResampleDesc.terminal = staticResampleWidth.path;
    diagnostics = {};
    auto staticResamplePlan = CompileCudaGraph(staticResampleDesc, &diagnostics);
    CHECK(staticResamplePlan && !diagnostics.HasErrors());
    auto staticResampleMetadata = GetCudaExecutionPlanMetadata(*staticResamplePlan);
    CHECK(staticResampleMetadata &&
          staticResampleMetadata->MemoryEstimate().memoryAvailable &&
          staticResampleMetadata->MemoryEstimate().conservativeUpperBound &&
          staticResampleMetadata->Tasks().front().estimate.memoryAvailable &&
          staticResampleMetadata->Tasks().front().estimate.steadyBytes ==
              2 * sizeof(int));
    auto const staticPeak = staticResampleMetadata->MemoryEstimate().concurrentPeakBytes;
    auto const staticBaseline = admissionPool->Snapshot();
    CHECK(staticPeak > 0 && staticPeak <= std::numeric_limits<size_t>::max() &&
          staticBaseline.usableBytes - staticBaseline.usedBytes >= staticPeak);
    auto const staticFillerBytes = staticBaseline.usableBytes -
        staticBaseline.usedBytes - static_cast<size_t>(staticPeak) + 1;
    auto staticFiller = admissionPool->TryReserve(
        staticFillerBytes,
        UsdGenExecutionResourceKind::Active);
    CHECK(staticFiller && staticFiller->Bytes() == staticFillerBytes);
    auto const staticSaturated = admissionPool->Snapshot();
    diagnostics = {};
    auto staticRejected = ExecuteCudaGraph(
        *staticResamplePlan, *workspace, 1.0, 4, &diagnostics, admittedDirect);
    auto const staticAfterRejected = admissionPool->Snapshot();
    CHECK(!staticRejected && DiagnosticIs(diagnostics,
        "CUDA: CUDA execution job memory reservation exceeds the available device budget") &&
          staticAfterRejected.usedBytes == staticSaturated.usedBytes &&
          ResourceKindBytes(staticAfterRejected, UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(staticSaturated, UsdGenExecutionResourceKind::Pending));
    staticFiller->Release();
    CHECK(admissionPool->Snapshot().usedBytes == staticBaseline.usedBytes);
    diagnostics = {};
    auto staticGeneration = ExecuteCudaGraph(
        *staticResamplePlan, *workspace, 1.0, 5, &diagnostics, admittedDirect);
    CHECK(staticGeneration && !diagnostics.HasErrors() &&
          AllWidthsAre(staticGeneration, .25f));
    auto const staticAfter = admissionPool->Snapshot();
    CHECK(ResourceKindBytes(staticAfter, UsdGenExecutionResourceKind::Pending) == 0 &&
          staticAfter.usedBytes > staticBaseline.usedBytes &&
          staticAfter.usedBytes <= staticBaseline.usedBytes + staticPeak);

    // Exercise the production staged queue as well as the synchronous oracle:
    // it consumes task ids/dependencies independently and must publish the
    // declared terminal snapshot after completing the metadata-later sibling.
    UsdGenExecutionRuntime runtime(2);
    UsdGenDiagnostics queueDiagnostics;
    UsdGenCudaExecutionQueue queue(runtime, -1, &queueDiagnostics);
    CHECK(queue.Valid());
    std::atomic<int> queueCalls{0};
    std::atomic<bool> queuePublished{false};
    CHECK(metadata->MemoryEstimate().memoryAvailable &&
          metadata->MemoryEstimate().conservativeUpperBound);
    int resourceDevice = -1;
    CHECK(cudaGetDevice(&resourceDevice) == cudaSuccess);
    auto resourcePool = FindUsdGenExecutionResourcePool(
        {UsdGenExecutionResourceBackend::Cuda, resourceDevice});
    CHECK(resourcePool);
    // The device overlap witness is test-only storage and is not part of the
    // compiled graph estimate. Charge it before taking the job baseline.
    armCudaOperatorAsyncWidthDeviceOverlapWitnessForTesting(2, 200000000);
    auto const beforeFanout = resourcePool->Snapshot();
    auto const fanoutPeak = metadata->MemoryEstimate().concurrentPeakBytes;
    armCudaOperatorAsyncWidthBranchGateForTesting(2);
    ReleaseWidthBranchGate releaseFanout;
    CHECK(queue.Submit(plan, 1.0, [&](uint64_t, auto outcome, auto const&) {
        queuePublished.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                             std::memory_order_release);
        queueCalls.fetch_add(1, std::memory_order_release);
    }) != 0);
    CHECK(WaitForWidthBranches());
    CHECK(cudaOperatorAsyncWidthDistinctBranchStreamCountForTesting() == 2 &&
          queueCalls.load(std::memory_order_acquire) == 0 && !queue.Snapshot());
    auto const duringFanout = resourcePool->Snapshot();
    // Both private Width children have consumed portions of one precharged
    // graph reservation. Their source/COW allocations and scratch are charged
    // exactly once through the reservation, not once again against the pool.
    CHECK(duringFanout.usedBytes == beforeFanout.usedBytes + fanoutPeak &&
          ResourceKindBytes(duringFanout, UsdGenExecutionResourceKind::Pending) > 0);
    // A device with concurrent-kernel support is the only environment in
    // which two admitted branch streams can be credited as an overlap
    // witness. Stream selection and both launcher arrivals remain required
    // on every device; capability only gates the overlap-specific assertion.
    int currentDevice = -1;
    cudaDeviceProp properties{};
    bool const canOverlap =
        cudaGetDevice(&currentDevice) == cudaSuccess &&
        cudaGetDeviceProperties(&properties, currentDevice) == cudaSuccess &&
        properties.concurrentKernels != 0;
    CHECK(cudaOperatorAsyncWidthDistinctBranchStreamCountForTesting() == 2);
    if (canOverlap)
        CHECK(cudaOperatorAsyncWidthDistinctBranchStreamCountForTesting() >= 2);
    releaseFanout.Release();
    queue.Drain();
    auto queueSnapshot = queue.Snapshot();
    CHECK(queueCalls.load(std::memory_order_acquire) == 1 &&
          queuePublished.load(std::memory_order_acquire) && queueSnapshot &&
          queueSnapshot->generation && AllWidthsAre(queueSnapshot->generation, .25f));
    auto const afterFanout = resourcePool->Snapshot();
    CHECK(ResourceKindBytes(afterFanout, UsdGenExecutionResourceKind::Pending) == 0 &&
          afterFanout.usedBytes > beforeFanout.usedBytes);
    auto const overlap =
        cudaOperatorAsyncWidthDeviceOverlapWitnessSnapshotForTesting();
    CHECK(overlap.expected == 2 && overlap.claims == 2 && overlap.arrivals == 2 &&
          overlap.status != UsdGenExecutionOverlapWitnessStatus::Error &&
          overlap.taskIds[0] != overlap.taskIds[1] &&
          overlap.laneIds[0] != overlap.laneIds[1]);
    if (canOverlap)
        CHECK(overlap.status == UsdGenExecutionOverlapWitnessStatus::Observed &&
              overlap.maxActive >= 2);
    else
        CHECK(overlap.maxActive <= 2);

    // Negative control: a dependency-ordered Width chain has two admitted
    // arrivals but cannot manufacture a two-branch overlap witness. Re-arming
    // the witness must reset its counters before this serial case.
    auto serialDesc = BaseDesc();
    auto serialParent = Width("/Dag/SerialParent", "/Dag/Source", .25f);
    auto serialChild = Width("/Dag/SerialChild", "/Dag/SerialParent", .25f);
    serialDesc.nodes = {Source(), serialParent, serialChild};
    serialDesc.terminal = serialChild.path;
    UsdGenDiagnostics serialDiagnostics;
    auto serialPlan = CompileCudaGraph(serialDesc, &serialDiagnostics);
    CHECK(serialPlan && !serialDiagnostics.HasErrors());
    UsdGenCudaExecutionQueue serialQueue(runtime, -1, &serialDiagnostics);
    CHECK(serialQueue.Valid());
    std::atomic<int> serialCalls{0};
    std::atomic<bool> serialPublished{false};
    armCudaOperatorAsyncWidthDeviceOverlapWitnessForTesting(2, 200000000);
    CHECK(serialQueue.Submit(serialPlan, 1.0,
        [&](uint64_t, auto outcome, auto const&) {
            serialPublished.store(
                outcome == UsdGenExecutionPipeline::Outcome::Published,
                std::memory_order_release);
            serialCalls.fetch_add(1, std::memory_order_release);
        }) != 0);
    serialQueue.Drain();
    CHECK(serialCalls.load(std::memory_order_acquire) == 1 &&
          serialPublished.load(std::memory_order_acquire) &&
          serialQueue.Snapshot() && serialQueue.Snapshot()->generation &&
          AllWidthsAre(serialQueue.Snapshot()->generation, .25f));
    auto const serialOverlap =
        cudaOperatorAsyncWidthDeviceOverlapWitnessSnapshotForTesting();
    CHECK(serialOverlap.expected == 2 && serialOverlap.claims == 2 &&
          serialOverlap.arrivals == 2 &&
          serialOverlap.status != UsdGenExecutionOverlapWitnessStatus::Error &&
          serialOverlap.taskIds[0] != serialOverlap.taskIds[1] &&
          serialOverlap.maxActive <= 1);
    if (canOverlap)
        CHECK(serialOverlap.maxActive < 2 &&
              serialOverlap.status != UsdGenExecutionOverlapWitnessStatus::Observed);

    // A nested branch retains declared-input value flow. The child waits for
    // its parent, while the source sibling has no false physical dependency.
    auto nestedDesc = BaseDesc();
    auto child = Width("/Dag/Child", "/Dag/Parent", .25f);
    auto parent = Width("/Dag/Parent", "/Dag/Source", .5f);
    auto nestedSibling = Width("/Dag/Sibling", "/Dag/Source", .75f);
    nestedDesc.nodes = {child, parent, Source(), nestedSibling};
    nestedDesc.terminal = child.path;
    diagnostics = {};
    auto nestedPlan = CompileCudaGraph(nestedDesc, &diagnostics);
    CHECK(nestedPlan && !diagnostics.HasErrors());
    auto nested = GetCudaExecutionPlanMetadata(*nestedPlan);
    CHECK(nested && nested->Shape() == UsdGenExecutionPlanShape::SourceRootedUnaryDag);
    auto const* nestedSource = TaskByPath(*nested, "/Dag/Source");
    auto const* nestedParent = TaskByPath(*nested, "/Dag/Parent");
    auto const* nestedChild = TaskByPath(*nested, "/Dag/Child");
    auto const* nestedOther = TaskByPath(*nested, "/Dag/Sibling");
    CHECK(nestedSource && nestedParent && nestedChild && nestedOther);
    CHECK(nestedSource->id == 0 && nestedParent->id == 1 &&
          nestedChild->id == 2 && nestedOther->id == 3);
    CHECK(DependenciesAre(*nestedParent, {0}) &&
          DependenciesAre(*nestedChild, {0,1}) &&
          DependenciesAre(*nestedOther, {0}));
    auto const* nestedPublication = nested->FindTask(nested->TerminalTask());
    CHECK(nestedPublication && DependenciesAre(*nestedPublication, {0,1,2,3}));
    auto const* parentWidths = Use(*nestedParent, UsdGenExecutionDataKind::Widths);
    auto const* childWidths = Use(*nestedChild, UsdGenExecutionDataKind::Widths);
    auto const* otherWidths = Use(*nestedOther, UsdGenExecutionDataKind::Widths);
    CHECK(parentWidths && childWidths && otherWidths);
    CHECK(childWidths->producerTask == nestedParent->id &&
          childWidths->inputValue == parentWidths->outputValue);
    CHECK(otherWidths->producerTask == nestedSource->id);
    for (auto const& task : nested->Tasks())
        CHECK(!task.estimate.timeAvailable &&
              task.estimate.estimatedMicroseconds == 0);
    CHECK(nested->MemoryEstimate().memoryAvailable);
    {
        CHECK(nested->MemoryEstimate().conservativeUpperBound &&
              nested->MemoryEstimate().immutableSharedInputBytes ==
                  ExpectedSourcePayloadBytes(nestedDesc));
        // Parent output is a private immutable value retained by Child. The
        // estimate must classify that producer lifetime instead of silently
        // dropping it from the nested path's peak.
        CHECK(nestedChild->estimate.producerRetentionBytes >=
              ExpectedSourcePayloadBytes(nestedDesc) +
              nestedParent->estimate.retainedOutputBytes);
    }
    auto nestedWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(nestedWorkspace);
    auto nestedGeneration = ExecuteCudaGraph(
        *nestedPlan, *nestedWorkspace, 1.0, 2, &diagnostics);
    CHECK(nestedGeneration && !diagnostics.HasErrors() &&
          AllWidthsAre(nestedGeneration, .25f));

    // Only Parent and the independent Sibling can reach the rendezvous. Child
    // remains behind Parent's semantic dependency until the gate is released.
    UsdGenDiagnostics nestedQueueDiagnostics;
    UsdGenCudaExecutionQueue nestedQueue(runtime, -1, &nestedQueueDiagnostics);
    CHECK(nestedQueue.Valid());
    std::atomic<int> nestedCalls{0};
    std::atomic<bool> nestedPublished{false};
    armCudaOperatorAsyncWidthBranchGateForTesting(2);
    ReleaseWidthBranchGate releaseNested;
    CHECK(nestedQueue.Submit(nestedPlan, 1.0,
        [&](uint64_t, auto outcome, auto const&) {
            nestedPublished.store(
                outcome == UsdGenExecutionPipeline::Outcome::Published,
                std::memory_order_release);
            nestedCalls.fetch_add(1, std::memory_order_release);
        }) != 0);
    CHECK(WaitForWidthBranches());
    CHECK(cudaOperatorAsyncWidthDistinctBranchStreamCountForTesting() == 2 &&
          nestedCalls.load(std::memory_order_acquire) == 0 &&
          !nestedQueue.Snapshot());
    releaseNested.Release();
    nestedQueue.Drain();
    CHECK(nestedCalls.load(std::memory_order_acquire) == 1 &&
          nestedPublished.load(std::memory_order_acquire) &&
          nestedQueue.Snapshot() && nestedQueue.Snapshot()->generation &&
          AllWidthsAre(nestedQueue.Snapshot()->generation, .25f));

    // A failing non-terminal sibling proves that it participates in execution,
    // while the previously published terminal value remains independently usable.
    auto failingDesc = FanoutDesc();
    AddRuntimeFailure(&failingDesc, SdfPath("/Dag/Sibling"));
    diagnostics = {};
    auto failingPlan = CompileCudaGraph(failingDesc, &diagnostics);
    CHECK(failingPlan && !diagnostics.HasErrors());
    auto failed = ExecuteCudaGraph(*failingPlan, *workspace, 2.0, 3,
                                   &diagnostics, generation);
    CHECK(!failed && diagnostics.HasErrors() && AllWidthsAre(generation, .25f));
    std::atomic<int> failureCalls{0};
    std::atomic<bool> queueFailed{false};
    CHECK(queue.Submit(failingPlan, 2.0,
        [&](uint64_t, auto outcome, auto const& failureDiagnostics) {
            queueFailed.store(
                outcome == UsdGenExecutionPipeline::Outcome::Failed &&
                    failureDiagnostics.HasErrors(),
                std::memory_order_release);
            failureCalls.fetch_add(1, std::memory_order_release);
        }) != 0);
    queue.Drain();
    CHECK(failureCalls.load(std::memory_order_acquire) == 1 &&
          queueFailed.load(std::memory_order_acquire) &&
          queue.Snapshot() == queueSnapshot &&
          AllWidthsAre(queue.Snapshot()->generation, .25f));

    // A source-only Width-DAG plan still makes Source an explicit ancestor of
    // publication; an empty final dependency list is not a valid task graph.
    auto sourceOnly = BaseDesc();
    sourceOnly.nodes = {Source()};
    sourceOnly.terminal = sourceOnly.nodes.front().path;
    diagnostics = {};
    auto sourceOnlyPlan = CompileCudaGraph(sourceOnly, &diagnostics);
    CHECK(sourceOnlyPlan && !diagnostics.HasErrors());
    auto sourceOnlyMetadata = GetCudaExecutionPlanMetadata(*sourceOnlyPlan);
    CHECK(sourceOnlyMetadata && sourceOnlyMetadata->Tasks().size() == 2 &&
          sourceOnlyMetadata->TerminalTask() == 1 &&
          DependenciesAre(*sourceOnlyMetadata->FindTask(1), {0}));

    // A single topology-changing Length may form an immutable trunk before
    // Width fan-out. Both Width tasks consume the compacted value family and
    // privately own only their replacement width planes.
    auto topologyDesc = TopologyFanoutDesc();
    diagnostics = {};
    auto topologyPlan = CompileCudaGraph(topologyDesc, &diagnostics);
    CHECK(topologyPlan && !diagnostics.HasErrors());
    auto topologyMetadata = GetCudaExecutionPlanMetadata(*topologyPlan);
    CHECK(topologyMetadata &&
          topologyMetadata->Shape() ==
              UsdGenExecutionPlanShape::SourceRootedUnaryDag &&
          topologyMetadata->Tasks().size() == 5);
    auto const* topologySource = TaskByPath(*topologyMetadata, "/Dag/Source");
    auto const* topologyLength = TaskByPath(*topologyMetadata, "/Dag/Length");
    auto const* topologyTerminal = TaskByPath(
        *topologyMetadata, "/Dag/TopologyTerminal");
    auto const* topologySibling = TaskByPath(
        *topologyMetadata, "/Dag/TopologySibling");
    auto const* topologyPublication =
        topologyMetadata->FindTask(topologyMetadata->TerminalTask());
    CHECK(topologySource && topologyLength && topologyTerminal &&
          topologySibling && topologyPublication);
    CHECK(DependenciesAre(*topologySource, {}) &&
          DependenciesAre(*topologyLength, {topologySource->id}) &&
          DependenciesAre(*topologyTerminal, {topologyLength->id}) &&
          DependenciesAre(*topologySibling, {topologyLength->id}) &&
          DependenciesAre(*topologyPublication,
              {topologySource->id, topologyLength->id,
               topologyTerminal->id, topologySibling->id}));
    CHECK(topologyLength->exclusiveWorkspace &&
          !topologyTerminal->exclusiveWorkspace &&
          !topologySibling->exclusiveWorkspace);
    auto const* lengthGeometry = Use(
        *topologyLength, UsdGenExecutionDataKind::ElementGeometry);
    auto const* lengthWidths = Use(
        *topologyLength, UsdGenExecutionDataKind::Widths);
    auto const* topologyTerminalGeometry = Use(
        *topologyTerminal, UsdGenExecutionDataKind::ElementGeometry);
    auto const* topologySiblingGeometry = Use(
        *topologySibling, UsdGenExecutionDataKind::ElementGeometry);
    auto const* terminalTopology = Use(
        *topologyTerminal, UsdGenExecutionDataKind::ElementTopology);
    auto const* siblingTopology = Use(
        *topologySibling, UsdGenExecutionDataKind::ElementTopology);
    auto const* topologyTerminalWidths = Use(
        *topologyTerminal, UsdGenExecutionDataKind::Widths);
    auto const* topologySiblingWidths = Use(
        *topologySibling, UsdGenExecutionDataKind::Widths);
    CHECK(lengthGeometry && lengthWidths && topologyTerminalGeometry &&
          topologySiblingGeometry && terminalTopology && siblingTopology &&
          topologyTerminalWidths && topologySiblingWidths);
    CHECK(topologyTerminalGeometry->inputValue == lengthGeometry->outputValue &&
          topologySiblingGeometry->inputValue == lengthGeometry->outputValue &&
          terminalTopology->producerTask == topologyLength->id &&
          siblingTopology->producerTask == topologyLength->id &&
          topologyTerminalWidths->inputValue == lengthWidths->outputValue &&
          topologySiblingWidths->inputValue == lengthWidths->outputValue &&
          topologyTerminalWidths->outputValue !=
              topologySiblingWidths->outputValue);
    CHECK(topologyMetadata->FindValue(lengthGeometry->outputValue)->storage ==
              UsdGenExecutionValueStorage::JobOwnedImmutable &&
          topologyMetadata->FindValue(
              topologyTerminalWidths->outputValue)->storage ==
              UsdGenExecutionValueStorage::JobOwnedImmutable &&
          topologyMetadata->FindValue(
              topologySiblingWidths->outputValue)->storage ==
              UsdGenExecutionValueStorage::JobOwnedImmutable);
    auto topologyWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(topologyWorkspace);
    auto topologyGeneration = ExecuteCudaGraph(
        *topologyPlan, *topologyWorkspace, 1.0, 20, &diagnostics);
    CHECK(topologyGeneration && !diagnostics.HasErrors() &&
          AllWidthsAre(topologyGeneration, .25f));
    {
        cudaStream_t topologyReader = nullptr;
        CHECK(cudaStreamCreateWithFlags(
            &topologyReader, cudaStreamNonBlocking) == cudaSuccess);
        {
            auto topologyLease = gpu::AcquireGeometry(
                topologyGeneration, topologyReader);
            CHECK(topologyLease && topologyLease.Geometry().curveCount == 2 &&
                  topologyLease.Geometry().pointCount == 7);
            std::vector<uint64_t> ids(2);
            uint32_t offsets[3]{};
            CHECK(cudaMemcpyAsync(ids.data(),
                      topologyLease.Geometry().stableIds.data,
                      ids.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                      topologyReader) == cudaSuccess &&
                  cudaMemcpyAsync(offsets,
                      topologyLease.Geometry().curveOffsets.data,
                      sizeof(offsets), cudaMemcpyDeviceToHost,
                      topologyReader) == cudaSuccess &&
                  cudaStreamSynchronize(topologyReader) == cudaSuccess);
            CHECK(ids == std::vector<uint64_t>({10, 20}) &&
                  offsets[0] == 0 && offsets[1] == 3 && offsets[2] == 7);
        }
        CHECK(cudaStreamSynchronize(topologyReader) == cudaSuccess);
        CHECK(cudaStreamDestroy(topologyReader) == cudaSuccess);
    }
    UsdGenDiagnostics topologyQueueDiagnostics;
    UsdGenCudaExecutionQueue topologyQueue(
        runtime, -1, &topologyQueueDiagnostics);
    CHECK(topologyQueue.Valid());
    std::atomic<int> topologyQueueCalls{0};
    std::atomic<bool> topologyQueuePublished{false};
    auto const topologyBeforeQueue = resourcePool->Snapshot();
    armCudaOperatorAsyncWidthBranchGateForTesting(2);
    ReleaseWidthBranchGate releaseTopologyBranches;
    CHECK(topologyQueue.Submit(topologyPlan, 1.0,
        [&](uint64_t, auto outcome, auto const&) {
            topologyQueuePublished.store(
                outcome == UsdGenExecutionPipeline::Outcome::Published,
                std::memory_order_release);
            topologyQueueCalls.fetch_add(1, std::memory_order_release);
        }) != 0);
    CHECK(WaitForWidthBranches());
    auto const topologyDuringBranches = resourcePool->Snapshot();
    CHECK(ResourceKindBytes(topologyDuringBranches,
              UsdGenExecutionResourceKind::Pinned) ==
              ResourceKindBytes(topologyBeforeQueue,
                  UsdGenExecutionResourceKind::Pinned) &&
          ResourceKindBytes(topologyDuringBranches,
              UsdGenExecutionResourceKind::Pending) >
              ResourceKindBytes(topologyBeforeQueue,
                  UsdGenExecutionResourceKind::Pending));
    releaseTopologyBranches.Release();
    topologyQueue.Drain();
    auto const topologyAfterQueue = resourcePool->Snapshot();
    CHECK(topologyQueueCalls.load(std::memory_order_acquire) == 1 &&
          topologyQueuePublished.load(std::memory_order_acquire) &&
          topologyQueue.Snapshot() && topologyQueue.Snapshot()->generation &&
          AllWidthsAre(topologyQueue.Snapshot()->generation, .25f) &&
          ResourceKindBytes(topologyAfterQueue,
              UsdGenExecutionResourceKind::Pinned) >
              ResourceKindBytes(topologyBeforeQueue,
                  UsdGenExecutionResourceKind::Pinned) &&
          ResourceKindBytes(topologyAfterQueue,
              UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(topologyBeforeQueue,
                  UsdGenExecutionResourceKind::Pending));

    // A failing non-terminal branch below the topology-changing trunk aborts
    // the whole transaction.  The compacted candidate and both private Width
    // planes remain job-owned until the publication join, so failure neither
    // replaces the accepted generation nor manufactures a Pinned charge.
    auto failingTopologyDesc = TopologyFanoutDesc();
    AddRuntimeFailure(
        &failingTopologyDesc, SdfPath("/Dag/TopologySibling"));
    diagnostics = {};
    auto failingTopologyPlan =
        CompileCudaGraph(failingTopologyDesc, &diagnostics);
    CHECK(failingTopologyPlan && !diagnostics.HasErrors());
    auto const beforeFailingTopology = resourcePool->Snapshot();
    auto failedTopology = ExecuteCudaGraph(*failingTopologyPlan,
        *topologyWorkspace, 2.0, 21, &diagnostics, topologyGeneration);
    auto const afterFailingTopology = resourcePool->Snapshot();
    CHECK(!failedTopology && diagnostics.HasErrors() &&
          AllWidthsAre(topologyGeneration, .25f) &&
          ResourceKindBytes(afterFailingTopology,
              UsdGenExecutionResourceKind::Pinned) ==
              ResourceKindBytes(beforeFailingTopology,
                  UsdGenExecutionResourceKind::Pinned) &&
          ResourceKindBytes(afterFailingTopology,
              UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(beforeFailingTopology,
                  UsdGenExecutionResourceKind::Pending));

    auto const acceptedTopologySnapshot = topologyQueue.Snapshot();
    auto const beforeQueuedTopologyFailure = resourcePool->Snapshot();
    std::atomic<int> topologyFailureCalls{0};
    std::atomic<bool> topologyFailed{false};
    CHECK(topologyQueue.Submit(failingTopologyPlan, 2.0,
        [&](uint64_t, auto outcome, auto const& failureDiagnostics) {
            topologyFailed.store(
                outcome == UsdGenExecutionPipeline::Outcome::Failed &&
                    failureDiagnostics.HasErrors(),
                std::memory_order_release);
            topologyFailureCalls.fetch_add(1, std::memory_order_release);
        }) != 0);
    topologyQueue.Drain();
    auto const afterQueuedTopologyFailure = resourcePool->Snapshot();
    CHECK(topologyFailureCalls.load(std::memory_order_acquire) == 1 &&
          topologyFailed.load(std::memory_order_acquire) &&
          topologyQueue.Snapshot() == acceptedTopologySnapshot &&
          AllWidthsAre(topologyQueue.Snapshot()->generation, .25f) &&
          ResourceKindBytes(afterQueuedTopologyFailure,
              UsdGenExecutionResourceKind::Pinned) ==
              ResourceKindBytes(beforeQueuedTopologyFailure,
                  UsdGenExecutionResourceKind::Pinned) &&
          ResourceKindBytes(afterQueuedTopologyFailure,
              UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(beforeQueuedTopologyFailure,
                  UsdGenExecutionResourceKind::Pending));

    // A topology-changing Length trunk may also feed a true two-input value
    // merge.  The two Width values are private COW replacements on the
    // compacted geometry; WidthBlend's operand order remains authored.
    auto topologyBlendDesc = TopologyWidthBlendDesc();
    UsdGenCurveBuffer topologyBlendReference;
    CHECK(LengthTopologyReference(topologyBlendDesc, &topologyBlendReference));
    if (topologyBlendReference.totalCurves != 2 ||
        topologyBlendReference.totalCvs != 7)
        std::fprintf(stderr,
            "Topology WidthBlend CPU cardinality: curves=%u cvs=%u (expected 2,7)\n",
            topologyBlendReference.totalCurves, topologyBlendReference.totalCvs);
    CHECK(topologyBlendReference.totalCurves == 2 &&
          topologyBlendReference.totalCvs == 7);
    diagnostics = {};
    auto topologyBlendPlan = CompileCudaGraph(topologyBlendDesc, &diagnostics);
    CHECK(topologyBlendPlan && !diagnostics.HasErrors());
    auto topologyBlendMetadata = GetCudaExecutionPlanMetadata(*topologyBlendPlan);
    auto const* topologyBlendSource = topologyBlendMetadata
        ? TaskByPath(*topologyBlendMetadata, "/Dag/Source") : nullptr;
    auto const* topologyBlendLength = topologyBlendMetadata
        ? TaskByPath(*topologyBlendMetadata, "/Dag/Length") : nullptr;
    auto const* topologyBlendLeft = topologyBlendMetadata
        ? TaskByPath(*topologyBlendMetadata, "/Dag/TopologyBlendLeft") : nullptr;
    auto const* topologyBlendRight = topologyBlendMetadata
        ? TaskByPath(*topologyBlendMetadata, "/Dag/TopologyBlendRight") : nullptr;
    auto const* topologyBlend = topologyBlendMetadata
        ? TaskByPath(*topologyBlendMetadata, "/Dag/TopologyBlend") : nullptr;
    CHECK(topologyBlendMetadata && topologyBlendSource && topologyBlendLength &&
          topologyBlendLeft && topologyBlendRight && topologyBlend &&
          topologyBlendMetadata->Tasks().size() == 6 &&
          topologyBlendMetadata->MemoryEstimate().runtimeRefinementAvailable &&
          DependenciesAre(*topologyBlendLength, {topologyBlendSource->id}) &&
          std::find(topologyBlend->dependencies.begin(),
                    topologyBlend->dependencies.end(), topologyBlendLeft->id) !=
              topologyBlend->dependencies.end() &&
          std::find(topologyBlend->dependencies.begin(),
                    topologyBlend->dependencies.end(), topologyBlendRight->id) !=
              topologyBlend->dependencies.end());
    auto topologyBlendWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(topologyBlendWorkspace);
    auto topologyBlendGeneration = ExecuteCudaGraph(*topologyBlendPlan,
        *topologyBlendWorkspace, 1.0, 30, &diagnostics);
    cudaStream_t topologyBlendReader = nullptr;
    CHECK(topologyBlendGeneration && !diagnostics.HasErrors() &&
          cudaStreamCreateWithFlags(&topologyBlendReader,
              cudaStreamNonBlocking) == cudaSuccess &&
          CheckGeneration(topologyBlendGeneration, topologyBlendReference,
              topologyBlendReader));

    // The public Session route must use the same native Length->Width->Blend
    // value DAG.  Retain the first lease across an ordered-input edit to prove
    // that COW publishes fresh widths without rewriting its compacted source.
    UsdGenSession topologyBlendSession;
    topologyBlendSession.SetDevicePublicationEnabled(true);
    topologyBlendSession.SetGraphDesc(topologyBlendDesc);
    auto topologyBlendFirst = topologyBlendSession.Commit(
        1.0, UsdGenCommitReason::SetTime);
    CHECK(topologyBlendFirst && topologyBlendFirst->device &&
          !topologyBlendSession.LastDiagnostics().HasErrors() &&
          CheckGeneration(topologyBlendFirst->device, topologyBlendReference,
              topologyBlendReader));
    auto swappedTopologyBlendDesc = topologyBlendDesc;
    auto merged = std::find_if(swappedTopologyBlendDesc.nodes.begin(),
        swappedTopologyBlendDesc.nodes.end(), [](auto const& node) {
            return node.path == SdfPath("/Dag/TopologyBlend");
        });
    CHECK(merged != swappedTopologyBlendDesc.nodes.end());
    std::swap(merged->inputs[0], merged->inputs[1]);
    UsdGenCurveBuffer swappedTopologyBlendReference;
    CHECK(LengthTopologyReference(swappedTopologyBlendDesc,
                       &swappedTopologyBlendReference));
    diagnostics = {};
    auto swappedTopologyBlendPlan = CompileCudaGraph(swappedTopologyBlendDesc,
                                                      &diagnostics);
    auto swappedTopologyBlendDirect = swappedTopologyBlendPlan
        ? ExecuteCudaGraph(*swappedTopologyBlendPlan, *topologyBlendWorkspace,
              1.0, 301, &diagnostics, topologyBlendGeneration) : nullptr;
    CHECK(swappedTopologyBlendDirect && !diagnostics.HasErrors() &&
          CheckGeneration(swappedTopologyBlendDirect,
              swappedTopologyBlendReference, topologyBlendReader));
    {
        auto oldLease = gpu::AcquireGeometry(topologyBlendFirst->device,
                                              topologyBlendReader);
        CHECK(oldLease);
        topologyBlendSession.SetGraphDesc(swappedTopologyBlendDesc);
        auto topologyBlendSecond = topologyBlendSession.Commit(
            2.0, UsdGenCommitReason::SetTime);
        CHECK(topologyBlendSecond && topologyBlendSecond->device &&
              topologyBlendSecond != topologyBlendFirst &&
              !topologyBlendSession.LastDiagnostics().HasErrors() &&
              CheckGeneration(topologyBlendSecond->device,
                  swappedTopologyBlendReference, topologyBlendReader));
        auto newLease = gpu::AcquireGeometry(topologyBlendSecond->device,
                                              topologyBlendReader);
        CHECK(newLease && newLease.Geometry().widths.data !=
              oldLease.Geometry().widths.data &&
              CheckGeneration(topologyBlendFirst->device,
                  topologyBlendReference, topologyBlendReader));
    }

    // Length's literal scale and cut/extend methods keep all ragged source
    // curves. Its cull mode can also validly compact every curve: Width and
    // WidthBlend must allocate from the post-Length cardinality, including
    // the one-zero offset empty topology contract.
    auto topologyCutExtendDesc = TopologyWidthBlendDesc(.25f, true);
    auto cutExtendLength = std::find_if(topologyCutExtendDesc.nodes.begin(),
        topologyCutExtendDesc.nodes.end(), [](auto const& node) {
            return node.path == SdfPath("/Dag/Length");
        });
    CHECK(cutExtendLength != topologyCutExtendDesc.nodes.end());
    cutExtendLength->params.push_back(
        {TfToken("length:method"), VtValue(TfToken("cutExtend")), false});
    auto lengthTerminalDesc = TopologyWidthBlendDesc();
    lengthTerminalDesc.terminal = SdfPath("/Dag/Length");
    int lengthCaseIndex = 0;
    for (auto const& lengthCase : std::array<UsdGenGraphDesc, 5>{
             TopologyWidthBlendDesc(.25f, true), topologyCutExtendDesc,
             TopologyWidthBlendDesc(.25f, false, .8f),
             TopologyLengthWidthBlendDesc(), lengthTerminalDesc}) {
        std::fprintf(stderr, "Length fan-in case %d\n", lengthCaseIndex++);
        UsdGenCurveBuffer lengthReference;
        CHECK(LengthTopologyReference(lengthCase, &lengthReference));
        diagnostics = {};
        auto lengthPlan = CompileCudaGraph(lengthCase, &diagnostics);
        CHECK(lengthPlan && !diagnostics.HasErrors());
        auto lengthGeneration = ExecuteCudaGraph(*lengthPlan,
            *topologyBlendWorkspace, 1.0, 31, &diagnostics,
            topologyBlendGeneration);
        for (auto const& error : diagnostics.errors) std::fprintf(stderr, "%s\n", error.c_str());
        CHECK(lengthGeneration && !diagnostics.HasErrors() &&
              CheckGeneration(lengthGeneration, lengthReference,
                  topologyBlendReader));
        UsdGenSession lengthSession;
        lengthSession.SetDevicePublicationEnabled(true);
        lengthSession.SetGraphDesc(lengthCase);
        auto lengthPublished = lengthSession.Commit(
            1.0, UsdGenCommitReason::SetTime);
        CHECK(lengthPublished && lengthPublished->device &&
              !lengthSession.LastDiagnostics().HasErrors() &&
              CheckGeneration(lengthPublished->device, lengthReference,
                  topologyBlendReader));
    }

    // Publication may select a Width predecessor while the sibling and its
    // WidthBlend descendant still execute.  That predecessor is immutable:
    // selecting it must not permit either descendant to write its widths.
    auto topologyPredecessorDesc = TopologyWidthBlendDesc(
        .25f, false, .45f, true);
    UsdGenCurveBuffer topologyPredecessorReference;
    CHECK(LengthTopologyReference(topologyPredecessorDesc, &topologyPredecessorReference));
    diagnostics = {};
    auto topologyPredecessorPlan = CompileCudaGraph(
        topologyPredecessorDesc, &diagnostics);
    auto topologyPredecessorMetadata = topologyPredecessorPlan
        ? GetCudaExecutionPlanMetadata(*topologyPredecessorPlan) : nullptr;
    CHECK(topologyPredecessorPlan && !diagnostics.HasErrors() &&
          topologyPredecessorMetadata && TaskByPath(
              *topologyPredecessorMetadata, "/Dag/TopologyBlend"));
    auto topologyPredecessorGeneration = ExecuteCudaGraph(
        *topologyPredecessorPlan, *topologyBlendWorkspace, 1.0, 32,
        &diagnostics, topologyBlendGeneration);
    CHECK(topologyPredecessorGeneration && !diagnostics.HasErrors() &&
          CheckGeneration(topologyPredecessorGeneration,
              topologyPredecessorReference, topologyBlendReader));

    CHECK(cudaStreamSynchronize(topologyBlendReader) == cudaSuccess &&
          cudaStreamDestroy(topologyBlendReader) == cudaSuccess);

    // Source is an immutable logical terminal even when every descendant
    // changes geometry/topology. Check complete source payloads after the
    // all-task join, including named channels and resampled source owners.
    {
        cudaStream_t reader = nullptr;
        CHECK(cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking) == cudaSuccess);
        auto sourceWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
        CHECK(sourceWorkspace);
        std::shared_ptr<const UsdGenDeviceGeneration> retained;
        UsdGenCurveBuffer retainedReference;
        for (int variant = 0; variant < 4; ++variant) {
            auto sourceDesc = TopologyWidthBlendDesc(.25f, variant == 0,
                variant == 2 ? .8f : .45f);
            sourceDesc.terminal = SdfPath("/Dag/Source");
            UsdGenAuthoredPlaneDesc named;
            named.name = TfToken("selectedSourceValue");
            named.type = UsdGenAuthoredPlaneType::Float32;
            named.domain = UsdGenAuthoredPlaneDomain::Point;
            named.arity = 1; named.floatValues = {0,1,2,3,4,5,6,7,8};
            sourceDesc.curveSets.front().authoredPlanes.push_back(named);
            if (variant == 3) for (auto& node : sourceDesc.nodes)
                if (node.type == TfToken("UsdGenCurveSource"))
                    node.params.push_back({TfToken("resampleTo"), VtValue(2), false});
            UsdGenCurveBuffer expected;
            CHECK(CpuReference(sourceDesc, &expected));
            CHECK(expected.totalCurves == 3 && expected.totalCvs == (variant == 3 ? 6u : 9u));
            diagnostics = {};
            auto sourcePlan = CompileCudaGraph(sourceDesc, &diagnostics);
            CHECK(sourcePlan && !diagnostics.HasErrors());
            auto direct = ExecuteCudaGraph(*sourcePlan, *sourceWorkspace, 1, 50 + variant,
                &diagnostics, retained);
            for (auto const& error : diagnostics.errors) std::fprintf(stderr, "%s\n", error.c_str());
            CHECK(direct && !diagnostics.HasErrors() && CheckSourcePayload(direct, expected, reader));
            UsdGenSession session;
            session.SetDevicePublicationEnabled(true); session.SetGraphDesc(sourceDesc);
            auto published = session.Commit(1, UsdGenCommitReason::SetTime);
            for (auto const& error : session.LastDiagnostics().errors) std::fprintf(stderr, "%s\n", error.c_str());
            CHECK(published && published->device && !session.LastDiagnostics().HasErrors() &&
                CheckSourcePayload(published->device, expected, reader));
            if (variant == 3) {
                // Selection happens before relay admission. A rejected
                // finalization must leave the selected named owners intact
                // for a second selection by the synchronous finalizer.
                auto job = CreateCudaExecutionJob(sourcePlan, *sourceWorkspace, 1, 60, &diagnostics);
                CHECK(job && ExecuteCudaJobSource(*job));
                for (size_t i=0; i<CudaExecutionJobOperatorCount(*job); ++i)
                    CHECK(ExecuteCudaJobOperator(*job, i));
                bool callbackCalled = false;
                setCudaFinalizationRelayCapacityForTesting(0);
                bool const admitted = FinalizeCudaExecutionJobAsync(job,
                    [&](auto) { callbackCalled = true; });
                setCudaFinalizationRelayCapacityForTesting(1024);
                CHECK(!admitted && !callbackCalled);
                auto retried = FinalizeCudaExecutionJob(*job);
                CHECK(retried && !diagnostics.HasErrors() &&
                    CheckSourcePayload(retried, expected, reader));
            }
            if (!retained) { retained = direct; retainedReference = expected; }
            CHECK(CheckSourcePayload(retained, retainedReference, reader));
        }
        CHECK(cudaStreamDestroy(reader) == cudaSuccess);
    }

    // Downsampling's authored + resampled named-plane peak is larger than
    // two resampled planes. Prove the reservation uses that phase's bound.
    {
        auto downsample = TopologyWidthBlendDesc();
        for (auto& node : downsample.nodes)
            if (node.type == TfToken("UsdGenCurveSource"))
                node.params.push_back({TfToken("resampleTo"), VtValue(2), false});
        diagnostics = {};
        auto plainPlan = CompileCudaGraph(downsample, &diagnostics);
        CHECK(plainPlan && !diagnostics.HasErrors());
        auto baseJob = CreateCudaExecutionJob(plainPlan, *topologyBlendWorkspace,
            1, 40, &diagnostics);
        CHECK(baseJob && !diagnostics.HasErrors());
        auto const plainPending = ResourceKindBytes(resourcePool->Snapshot(),
            UsdGenExecutionResourceKind::Pending);
        baseJob.reset();
        UsdGenAuthoredPlaneDesc named;
        named.name = TfToken("lengthAdmissionWitness");
        named.type = UsdGenAuthoredPlaneType::Float32;
        named.domain = UsdGenAuthoredPlaneDomain::Point;
        named.arity = 1;
        named.floatValues = {0,1,2,3,4,5,6,7,8};
        downsample.curveSets.front().authoredPlanes.push_back(named);
        auto namedPlan = CompileCudaGraph(downsample, &diagnostics);
        CHECK(namedPlan && !diagnostics.HasErrors());
        auto namedJob = CreateCudaExecutionJob(namedPlan, *topologyBlendWorkspace,
            1, 41, &diagnostics);
        CHECK(namedJob && !diagnostics.HasErrors());
        CHECK(ResourceKindBytes(resourcePool->Snapshot(), UsdGenExecutionResourceKind::Pending) ==
            plainPending + (9 + 6) * sizeof(float));
    }

    // Admission diagnostics distinguish invalid unary arity, structural root
    // errors, and branch kinds whose immutable snapshots are not implemented.
    auto twoInputs = FanoutDesc();
    twoInputs.nodes[0].inputs.push_back(SdfPath("/Dag/Sibling"));
    diagnostics = {};
    CHECK(!CompileCudaGraph(twoInputs, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unsupported composition: CUDA unary Width requires exactly one geometry input"));

    auto multipleSources = FanoutDesc();
    auto secondSource = Source();
    secondSource.path = SdfPath("/Dag/SecondSource");
    multipleSources.nodes.push_back(std::move(secondSource));
    diagnostics = {};
    CHECK(!CompileCudaGraph(multipleSources, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: CUDA graph requires exactly one CurveSource"));

    auto missingInput = FanoutDesc();
    missingInput.nodes[2].inputs = {SdfPath("/Dag/Missing")};
    diagnostics = {};
    CHECK(!CompileCudaGraph(missingInput, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: invalid authoring: CUDA Width input does not name a graph node"));

    auto disconnected = FanoutDesc();
    disconnected.nodes[2].inputs = {disconnected.nodes[2].path};
    diagnostics = {};
    CHECK(!CompileCudaGraph(disconnected, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: invalid authoring: CUDA Width graph contains a cycle or is not source-rooted"));

    auto ambiguousTerminal = FanoutDesc();
    ambiguousTerminal.terminal = SdfPath();
    diagnostics = {};
    CHECK(!CompileCudaGraph(ambiguousTerminal, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unsupported composition: branched CUDA Width graphs require an explicit terminal"));

    auto branchedLength = BaseDesc();
    auto length = Width("/Dag/Length", "/Dag/Source", .5f);
    length.type = TfToken("UsdGenLength");
    length.params.clear();
    length.params.push_back({TfToken("length:value"), VtValue(.5f), false});
    auto widthSibling = Width("/Dag/Width", "/Dag/Source", .25f);
    branchedLength.nodes = {Source(), length, widthSibling};
    branchedLength.terminal = length.path;
    diagnostics = {};
    auto branchedLengthPlan = CompileCudaGraph(branchedLength, &diagnostics);
    CHECK(branchedLengthPlan && !diagnostics.HasErrors());
    UsdGenCurveBuffer branchedLengthReference;
    CHECK(CpuReference(branchedLength, &branchedLengthReference));
    auto branchedLengthWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(branchedLengthWorkspace && !diagnostics.HasErrors());
    cudaStream_t branchedLengthReader = nullptr;
    CHECK(cudaStreamCreateWithFlags(&branchedLengthReader,
                                    cudaStreamNonBlocking) == cudaSuccess);
    auto branchedLengthGeneration = ExecuteCudaGraph(*branchedLengthPlan,
        *branchedLengthWorkspace, 1.0, 300, &diagnostics);
    CHECK(branchedLengthGeneration && !diagnostics.HasErrors() &&
          CheckSourcePayload(branchedLengthGeneration, branchedLengthReference,
                             branchedLengthReader));
    {
        auto lease = gpu::AcquireGeometry(branchedLengthGeneration,
                                          branchedLengthReader);
        CHECK(lease && !lease.RootT().size && !lease.RootB().size &&
              !lease.RootN().size);
    }
    CHECK(cudaStreamSynchronize(branchedLengthReader) == cudaSuccess &&
          cudaStreamDestroy(branchedLengthReader) == cudaSuccess);

    // RBF is a point-value DAG node: each branch consumes exactly its authored
    // Width/Length predecessor and can coexist with an independent RBF
    // sibling. The ordered WidthBlend must retain the right branch's complete
    // non-width bundle as a proof input, rather than reducing it to widths.
    auto rbfDag = BaseDesc();
    auto rbfWidth = Width("/Dag/RbfWidth", "/Dag/Source", .5f);
    auto rbfLength = Length("/Dag/RbfLength", "/Dag/Source");
    rbfLength.params = {{TfToken("length:mode"), VtValue(TfToken("scale")), false},
                        {TfToken("length:value"), VtValue(1.f), false}};
    auto rbfLeft = RbfDeform("/Dag/RbfAfterWidth", "/Dag/RbfWidth");
    auto rbfRight = RbfDeform("/Dag/RbfAfterLength", "/Dag/RbfLength");
    UsdGenNodeDesc rbfBlend;
    rbfBlend.path = SdfPath("/Dag/RbfBlend"); rbfBlend.type = TfToken("UsdGenWidthBlend");
    rbfBlend.inputs = {rbfLeft.path, rbfRight.path};
    rbfBlend.params.push_back({TfToken("widthBlend:weight"), VtValue(.5f), false});
    rbfDag.nodes = {rbfBlend, rbfRight, rbfLength, Source(), rbfLeft, rbfWidth};
    rbfDag.terminal = rbfBlend.path;
    diagnostics = {};
    auto rbfDagPlan = CompileCudaGraph(rbfDag, &diagnostics);
    CHECK(rbfDagPlan && !diagnostics.HasErrors());
    auto rbfDagMetadata = GetCudaExecutionPlanMetadata(*rbfDagPlan);
    auto rbfSourceTask = rbfDagMetadata ? TaskByPath(*rbfDagMetadata, "/Dag/Source") : nullptr;
    auto rbfWidthTask = rbfDagMetadata ? TaskByPath(*rbfDagMetadata, "/Dag/RbfWidth") : nullptr;
    auto rbfLengthTask = rbfDagMetadata ? TaskByPath(*rbfDagMetadata, "/Dag/RbfLength") : nullptr;
    auto rbfLeftTask = rbfDagMetadata ? TaskByPath(*rbfDagMetadata, "/Dag/RbfAfterWidth") : nullptr;
    auto rbfRightTask = rbfDagMetadata ? TaskByPath(*rbfDagMetadata, "/Dag/RbfAfterLength") : nullptr;
    auto rbfBlendTask = rbfDagMetadata ? TaskByPath(*rbfDagMetadata, "/Dag/RbfBlend") : nullptr;
    CHECK(rbfDagMetadata && rbfSourceTask && rbfWidthTask && rbfLengthTask &&
          rbfLeftTask && rbfRightTask && rbfBlendTask &&
          rbfDagMetadata->Shape() == UsdGenExecutionPlanShape::SourceRootedValueDag);
    CHECK(std::find(rbfLeftTask->dependencies.begin(), rbfLeftTask->dependencies.end(),
                    rbfWidthTask->id) != rbfLeftTask->dependencies.end());
    CHECK(std::find(rbfRightTask->dependencies.begin(), rbfRightTask->dependencies.end(),
                    rbfLengthTask->id) != rbfRightTask->dependencies.end());
    CHECK(std::find(rbfBlendTask->dependencies.begin(), rbfBlendTask->dependencies.end(),
                    rbfLeftTask->id) != rbfBlendTask->dependencies.end() &&
          std::find(rbfBlendTask->dependencies.begin(), rbfBlendTask->dependencies.end(),
                    rbfRightTask->id) != rbfBlendTask->dependencies.end());
    for (auto kind : {UsdGenExecutionDataKind::ElementGeometry,
                      UsdGenExecutionDataKind::ElementTopology,
                      UsdGenExecutionDataKind::StableIds,
                      UsdGenExecutionDataKind::RootBindings,
                      UsdGenExecutionDataKind::NamedChannels}) {
        // RBF creates only the right branch's point revision. The rest of
        // its non-width packet is the immutable Length-owned inheritance.
        auto const expectedProducer = kind == UsdGenExecutionDataKind::ElementGeometry
            ? rbfRightTask->id : rbfLengthTask->id;
        CHECK(std::any_of(rbfBlendTask->resources.begin(), rbfBlendTask->resources.end(),
            [&](auto const& use) { return use.resource == kind &&
                use.access == UsdGenExecutionResourceAccess::Read &&
                use.producerTask == expectedProducer; }));
    }
    auto rbfLeftTerminal = rbfDag;
    rbfLeftTerminal.nodes.erase(std::remove_if(rbfLeftTerminal.nodes.begin(),
        rbfLeftTerminal.nodes.end(), [](UsdGenNodeDesc const& node) {
            return node.path == SdfPath("/Dag/RbfBlend");
        }), rbfLeftTerminal.nodes.end());
    rbfLeftTerminal.terminal = rbfLeft.path;
    diagnostics = {};
    auto rbfLeftTerminalPlan = CompileCudaGraph(rbfLeftTerminal, &diagnostics);
    CHECK(rbfLeftTerminalPlan && !diagnostics.HasErrors());
    auto rbfLeftTerminalMetadata = GetCudaExecutionPlanMetadata(*rbfLeftTerminalPlan);
    auto rbfLeftTerminalTask = rbfLeftTerminalMetadata
        ? TaskByPath(*rbfLeftTerminalMetadata, "/Dag/RbfAfterWidth") : nullptr;
    auto rbfPublication = rbfLeftTerminalMetadata
        ? rbfLeftTerminalMetadata->FindTask(rbfLeftTerminalMetadata->TerminalTask()) : nullptr;
    CHECK(rbfLeftTerminalTask && rbfPublication &&
          rbfPublication->kind == UsdGenExecutionTaskKind::Publication &&
          std::any_of(rbfPublication->resources.begin(), rbfPublication->resources.end(),
              [&](auto const& use) { return use.resource == UsdGenExecutionDataKind::ElementGeometry &&
                  use.access == UsdGenExecutionResourceAccess::Read &&
                  use.producerTask == rbfLeftTerminalTask->id; }));

    // RBF is rest-to-animated per lineage. A true sequential second Deform is
    // not a branch and must fail closed instead of applying surface motion
    // twice.
    auto doubleRbf = BaseDesc();
    auto doubleWidth = Width("/Dag/RbfWidth", "/Dag/Source", .5f);
    auto firstRbf = RbfDeform("/Dag/RbfAfterWidth", "/Dag/RbfWidth");
    auto secondRbf = RbfDeform("/Dag/RbfSecond", "/Dag/RbfAfterWidth");
    doubleRbf.nodes = {Source(), doubleWidth, firstRbf, secondRbf};
    doubleRbf.terminal = secondRbf.path;
    diagnostics = {};
    CHECK(!CompileCudaGraph(doubleRbf, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unsupported composition: a second rest-to-animated deformation would apply surface motion twice"));

    auto missingBlendInput = WidthBlendDesc();
    missingBlendInput.nodes.front().inputs.pop_back();
    diagnostics = {};
    CHECK(!CompileCudaGraph(missingBlendInput, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: unsupported composition: CUDA WidthBlend requires exactly two ordered geometry inputs"));

    auto duplicateBlendInput = WidthBlendDesc();
    duplicateBlendInput.nodes.front().inputs[1] =
        duplicateBlendInput.nodes.front().inputs[0];
    diagnostics = {};
    CHECK(!CompileCudaGraph(duplicateBlendInput, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: invalid authoring: CUDA WidthBlend requires two distinct graph inputs"));

    auto invalidBlendWeight = WidthBlendDesc(1.25f);
    diagnostics = {};
    CHECK(!CompileCudaGraph(invalidBlendWeight, &diagnostics));
    CHECK(DiagnosticIs(diagnostics,
        "CUDA: WidthBlend requires a finite usdGen:widthBlend:weight in [0,1] and no auxiliary inputs"));

    // Execute through the task graph, which exercises cross-stream event
    // waits for both immutable predecessors and retains their owners until
    // publication.  Endpoint results must select a source plane exactly.
    diagnostics = {};
    auto blendGeneration = ExecuteCudaGraph(*blendPlan, *workspace, 1.0, 900,
        &diagnostics);
    std::vector<float> blendedWidths;
    CHECK(blendGeneration && !diagnostics.HasErrors() &&
          ReadWidths(blendGeneration, &blendedWidths) &&
          std::all_of(blendedWidths.begin(), blendedWidths.end(), [](float value) {
              return std::fabs(value - .35f) < 1e-6f;
          }) && blendGeneration->Owner());
    auto leftEndpointDesc = WidthBlendDesc(0.0f);
    diagnostics = {};
    auto leftEndpointPlan = CompileCudaGraph(leftEndpointDesc, &diagnostics);
    auto leftEndpoint = leftEndpointPlan ? ExecuteCudaGraph(*leftEndpointPlan,
        *workspace, 1.0, 901, &diagnostics, blendGeneration) : nullptr;
    CHECK(leftEndpoint && !diagnostics.HasErrors() &&
          AllWidthsAre(leftEndpoint, .2f));
    auto rightEndpointDesc = WidthBlendDesc(1.0f);
    diagnostics = {};
    auto rightEndpointPlan = CompileCudaGraph(rightEndpointDesc, &diagnostics);
    auto rightEndpoint = rightEndpointPlan ? ExecuteCudaGraph(*rightEndpointPlan,
        *workspace, 1.0, 902, &diagnostics, leftEndpoint) : nullptr;
    CHECK(rightEndpoint && !diagnostics.HasErrors() &&
          AllWidthsAre(rightEndpoint, .8f));
    auto swappedBlendDesc = WidthBlendDesc();
    swappedBlendDesc.nodes.front().inputs = {SdfPath("/Dag/BlendRight"),
                                             SdfPath("/Dag/BlendLeft")};
    diagnostics = {};
    auto swappedBlendPlan = CompileCudaGraph(swappedBlendDesc, &diagnostics);
    auto swappedBlend = swappedBlendPlan ? ExecuteCudaGraph(*swappedBlendPlan,
        *workspace, 1.0, 903, &diagnostics, rightEndpoint) : nullptr;
    CHECK(swappedBlend && !diagnostics.HasErrors() &&
          AllWidthsAre(swappedBlend, .65f));

    // The runtime graph specialization owns only its private Cache slots; it
    // never publishes one.  A repeated same-key execution replays the native
    // graph into a newly allocated COW output, while a blend change replaces
    // the idle entry without accumulating a second cache charge.
    int graphDevice = -1;
    CHECK(cudaGetDevice(&graphDevice) == cudaSuccess);
    auto graphPool = FindUsdGenExecutionResourcePool(
        {UsdGenExecutionResourceBackend::Cuda, graphDevice});
    CHECK(graphPool);
    auto const cacheBeforeGraphWorkspace = graphPool->Snapshot();
    std::shared_ptr<const UsdGenDeviceGeneration> graphFirst, graphReplay;
    {
        diagnostics = {};
        auto graphWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
        CHECK(graphWorkspace && !diagnostics.HasErrors());
        auto const cacheBefore = graphPool->Snapshot();
        auto const graphBefore = GetCudaWidthBlendGraphStats(*graphWorkspace);
        CHECK(graphBefore.captures == 0 && graphBefore.cacheBytes == 0 &&
              !graphBefore.inFlight && !graphBefore.quarantined);
        graphFirst = ExecuteCudaGraph(*blendPlan, *graphWorkspace, 1.0,
                                      910, &diagnostics);
        auto const graphFirstStats = GetCudaWidthBlendGraphStats(*graphWorkspace);
        auto const graphSlotBytes = 3 * ExpectedPrivateWidthBytes(blendDesc) +
            64u * 1024u;
        auto const cacheAfterFirst = graphPool->Snapshot();
        CHECK(graphFirst && !diagnostics.HasErrors() && AllWidthsAre(graphFirst, .35f) &&
              graphFirst->Owner() && graphFirstStats.captures == 1 &&
              graphFirstStats.replays == 0 && graphFirstStats.evictions == 0 &&
              graphFirstStats.quarantines == 0 && !graphFirstStats.inFlight &&
              !graphFirstStats.quarantined && graphFirstStats.cacheBytes == graphSlotBytes &&
              ResourceKindBytes(cacheAfterFirst, UsdGenExecutionResourceKind::Cache) >=
                  ResourceKindBytes(cacheBefore, UsdGenExecutionResourceKind::Cache) +
                      graphSlotBytes);
        diagnostics = {};
        graphReplay = ExecuteCudaGraph(*blendPlan, *graphWorkspace, 1.0,
                                       911, &diagnostics, graphFirst);
        auto const graphReplayStats = GetCudaWidthBlendGraphStats(*graphWorkspace);
        auto const cacheAfterReplay = graphPool->Snapshot();
        CHECK(graphReplay && !diagnostics.HasErrors() && AllWidthsAre(graphReplay, .35f) &&
              graphReplay->Owner() != graphFirst->Owner() &&
              graphReplayStats.captures == graphFirstStats.captures &&
              graphReplayStats.replays == graphFirstStats.replays + 1 &&
              graphReplayStats.cacheBytes == graphSlotBytes &&
              ResourceKindBytes(cacheAfterReplay, UsdGenExecutionResourceKind::Cache) ==
                  ResourceKindBytes(cacheAfterFirst, UsdGenExecutionResourceKind::Cache));
        auto graphKeyChangeDesc = WidthBlendDesc(.5f);
        diagnostics = {};
        auto graphKeyChangePlan = CompileCudaGraph(graphKeyChangeDesc, &diagnostics);
        CHECK(graphKeyChangePlan && !diagnostics.HasErrors());
        auto graphKeyChange = ExecuteCudaGraph(*graphKeyChangePlan, *graphWorkspace, 1.0,
                                                912, &diagnostics, graphReplay);
        auto const graphKeyChangeStats = GetCudaWidthBlendGraphStats(*graphWorkspace);
        auto const cacheAfterKeyChange = graphPool->Snapshot();
        CHECK(graphKeyChange && !diagnostics.HasErrors() &&
              AllWidthsAre(graphKeyChange, .5f) &&
              graphKeyChangeStats.captures == graphReplayStats.captures + 1 &&
              graphKeyChangeStats.evictions == graphReplayStats.evictions + 1 &&
              graphKeyChangeStats.cacheBytes == graphSlotBytes &&
              ResourceKindBytes(cacheAfterKeyChange, UsdGenExecutionResourceKind::Cache) ==
                  ResourceKindBytes(cacheAfterReplay, UsdGenExecutionResourceKind::Cache));
    }
    auto const cacheAfterGraphWorkspace = graphPool->Snapshot();
    CHECK(ResourceKindBytes(cacheAfterGraphWorkspace, UsdGenExecutionResourceKind::Cache) ==
              ResourceKindBytes(cacheBeforeGraphWorkspace,
                                UsdGenExecutionResourceKind::Cache) &&
          graphFirst && graphReplay && AllWidthsAre(graphFirst, .35f) &&
          AllWidthsAre(graphReplay, .35f));

    // A rejected graph entry is strictly pre-submit: the ordinary WidthBlend
    // remains legal, no cache charge/poison is retained, and the next job can
    // capture normally.
    {
        diagnostics = {};
        auto fallbackWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
        CHECK(fallbackWorkspace && !diagnostics.HasErrors());
        failNextCudaWidthBlendGraphCaptureForTesting();
        auto captureRejected = ExecuteCudaGraph(*blendPlan, *fallbackWorkspace, 1.0,
                                                913, &diagnostics);
        auto const rejectedStats = GetCudaWidthBlendGraphStats(*fallbackWorkspace);
        CHECK(captureRejected && !diagnostics.HasErrors() &&
              AllWidthsAre(captureRejected, .35f) && rejectedStats.captures == 0 &&
              rejectedStats.misses == 1 && rejectedStats.cacheBytes == 0 &&
              !rejectedStats.inFlight && !rejectedStats.quarantined);
        diagnostics = {};
        auto captureRetry = ExecuteCudaGraph(*blendPlan, *fallbackWorkspace, 1.0,
                                             914, &diagnostics, captureRejected);
        auto const retryStats = GetCudaWidthBlendGraphStats(*fallbackWorkspace);
        CHECK(captureRetry && !diagnostics.HasErrors() && AllWidthsAre(captureRetry, .35f) &&
              retryStats.captures == 1 && retryStats.misses == 2 &&
              retryStats.cacheBytes == 3 * ExpectedPrivateWidthBytes(blendDesc) +
                  64u * 1024u && !retryStats.inFlight && !retryStats.quarantined);
    }

    // Cross-origin WidthBlend: independently produced identity Length values
    // carry equal full non-width payloads and require the GPU equivalence
    // witness before the blend is admitted.
    cudaStream_t crossOriginReader=nullptr;
    CHECK(cudaStreamCreateWithFlags(&crossOriginReader,cudaStreamNonBlocking)==cudaSuccess);
    {
        auto equalDesc = CrossOriginWidthBlendDesc();
        UsdGenCurveBuffer equalReference;
        CHECK(CpuReference(equalDesc, &equalReference) &&
              equalReference.chunks.empty());
        diagnostics = {};
        auto equalPlan = CompileCudaGraph(equalDesc, &diagnostics);
        CHECK(equalPlan && !diagnostics.HasErrors());
        auto equalMetadata=GetCudaExecutionPlanMetadata(*equalPlan);CHECK(equalMetadata);
        auto mergeTask=TaskByPath(*equalMetadata,"/Dag/CrossOriginBlend");
        auto rightTask=TaskByPath(*equalMetadata,"/Dag/EqualRightLength");
        CHECK(mergeTask && rightTask && mergeTask->estimate.scratchPeakBytes==16);
        for(auto kind:{UsdGenExecutionDataKind::ElementGeometry,UsdGenExecutionDataKind::ElementTopology,
                      UsdGenExecutionDataKind::StableIds,UsdGenExecutionDataKind::RootBindings,
                      UsdGenExecutionDataKind::NamedChannels}) {
            auto rightUse=std::find_if(rightTask->resources.begin(),rightTask->resources.end(),
                [kind](auto const& use){return use.resource==kind && use.access==UsdGenExecutionResourceAccess::ReadWrite;});
            CHECK(rightUse!=rightTask->resources.end());
            CHECK(std::any_of(mergeTask->resources.begin(),mergeTask->resources.end(),[&](auto const& use){
                return use.resource==kind && use.access==UsdGenExecutionResourceAccess::Read &&
                    use.inputValue==rightUse->outputValue && use.producerTask==rightTask->id;
            }));
        }
        auto equalGeneration = ExecuteCudaGraph(*equalPlan, *workspace, 1.0,
                                                920, &diagnostics, generation);
        CHECK(equalGeneration && !diagnostics.HasErrors() &&
              CheckSourcePayload(equalGeneration, equalReference, crossOriginReader));
        UsdGenSession equalSession;
        equalSession.SetDevicePublicationEnabled(true);
        equalSession.SetGraphDesc(equalDesc);
        auto equalPublished = equalSession.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(equalPublished && equalPublished->device &&
              !equalSession.LastDiagnostics().HasErrors() &&
              CheckSourcePayload(equalPublished->device, equalReference, crossOriginReader));

        auto unequalDesc = CrossOriginWidthBlendDesc(.5f);
        diagnostics = {};
        auto unequalPlan = CompileCudaGraph(unequalDesc, &diagnostics);
        CHECK(unequalPlan && !diagnostics.HasErrors());
        auto rejected = ExecuteCudaGraph(*unequalPlan, *workspace, 1.0,
                                         921, &diagnostics, equalGeneration);
        CHECK(!rejected && diagnostics.HasErrors() &&
              CheckSourcePayload(equalGeneration, equalReference, crossOriginReader));
        CHECK(!workspace->IsPoisoned());
        diagnostics={};
        auto recovered=ExecuteCudaGraph(*equalPlan,*workspace,2.0,922,&diagnostics,equalGeneration);
        CHECK(recovered && !diagnostics.HasErrors() &&
              CheckSourcePayload(recovered,equalReference,crossOriginReader) &&
              CheckSourcePayload(equalGeneration,equalReference,crossOriginReader));
        UsdGenSession unequalSession;
        unequalSession.SetDevicePublicationEnabled(true);
        unequalSession.SetGraphDesc(equalDesc);
        auto prior = unequalSession.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(prior && prior->device && !unequalSession.LastDiagnostics().HasErrors());
        unequalSession.SetGraphDesc(unequalDesc);
        auto failedPublication = unequalSession.Commit(2, UsdGenCommitReason::SetTime);
        CHECK(failedPublication == prior && unequalSession.LastDiagnostics().HasErrors() &&
              prior->device && CheckSourcePayload(prior->device, equalReference, crossOriginReader));
    }

    // Exercise source-vs-transformed frame provenance, a frame-absent authored
    // ReferenceSource, and independently materialized Grow frame selections.
    for(unsigned variant=0;variant<3;++variant) {
        auto provenanceDesc=CrossOriginWidthBlendDesc();
        if(variant==0) {
            provenanceDesc.nodes.erase(std::remove_if(provenanceDesc.nodes.begin(),provenanceDesc.nodes.end(),
                [](auto const& node){return node.path==SdfPath("/Dag/EqualRightLength");}),provenanceDesc.nodes.end());
            for(auto& node:provenanceDesc.nodes) if(node.path==SdfPath("/Dag/EqualRightWidth"))
                node.inputs={SdfPath("/Dag/Source")};
        } else if(variant==1) {
            auto& curves=provenanceDesc.curveSets.front();
            curves.role=UsdGenRole::Reference;curves.curveRole=TfToken("guide");
            curves.authoredPlanes.clear();
            for(auto& node:provenanceDesc.nodes) if(node.path==SdfPath("/Dag/Source")) {
                node.type=TfToken("UsdGenReferenceSource");node.curves.clear();node.surfaces.clear();
                node.references={curves.path};
            }
        } else {
            for(auto& node:provenanceDesc.nodes) if(node.type==TfToken("UsdGenLength")) {
                node.type=TfToken("UsdGenGrow");
                node.params={{TfToken("segments"),VtValue(4),false},
                             {TfToken("length"),VtValue(.2f),false}};
            }
        }
        UsdGenCurveBuffer expected;
        CHECK(CpuReference(provenanceDesc,&expected) && expected.chunks.empty());
        diagnostics={};auto provenancePlan=CompileCudaGraph(provenanceDesc,&diagnostics);CHECK(provenancePlan && !diagnostics.HasErrors());
        auto localWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);CHECK(localWorkspace);
        auto result=ExecuteCudaGraph(*provenancePlan,*localWorkspace,1,930+variant,&diagnostics);
        CHECK(result && !diagnostics.HasErrors() && CheckSourcePayload(result,expected,crossOriginReader));
        UsdGenSession session;session.SetDevicePublicationEnabled(true);session.SetGraphDesc(provenanceDesc);
        auto published=session.Commit(1,UsdGenCommitReason::SetTime);
        CHECK(published && published->device && !session.LastDiagnostics().HasErrors() &&
              CheckSourcePayload(published->device,expected,crossOriginReader));
    }

    // Failure after comparison submission is not a semantic mismatch. Both
    // predecessor snapshots and the scalar proof must remain quarantined,
    // while Session continues to expose its already-proved COW publication.
    for (auto inject : {failNextCudaOperatorRelayNonWidthCallbackInstallForTesting,
                        failNextCudaOperatorRelayNonWidthNativeCallbackForTesting}) {
        auto quarantineDesc=CrossOriginWidthBlendDesc();
        UsdGenCurveBuffer reference; CHECK(CpuReference(quarantineDesc,&reference));
        UsdGenSession session; session.SetDevicePublicationEnabled(true);
        session.SetGraphDesc(quarantineDesc);
        auto prior=session.Commit(1,UsdGenCommitReason::SetTime);
        CHECK(prior && prior->device && !session.LastDiagnostics().HasErrors());
        auto changed=quarantineDesc;
        for(auto& node:changed.nodes) if(node.path==SdfPath("/Dag/EqualRightWidth"))
            node.params[0].value=VtValue(.9f);
        auto const quarantined=cudaOperatorRelayQuarantinedCountForTesting();
        session.SetGraphDesc(changed); inject();
        CHECK(session.Commit(2,UsdGenCommitReason::SetTime)==prior);
        CHECK(session.LastDiagnostics().HasErrors() &&
              cudaOperatorRelayQuarantinedCountForTesting()==quarantined+1);
        CHECK(CheckSourcePayload(prior->device,reference,crossOriginReader));
    }
    CHECK(cudaStreamDestroy(crossOriginReader)==cudaSuccess);

    return 0;
}
