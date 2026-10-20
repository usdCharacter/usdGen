#include "usdGen/cudaExecution.h"

#include "usdGen/expressionTargets.h"
#include "usdGen/opRegistry.h"
#include "usdGen/surfaceCageSource.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <numeric>
#include <array>
#include <atomic>
#include <mutex>
#include <limits>
#include <initializer_list>
#include <chrono>
#include <cstring>
#include <thread>
#include <stdexcept>

#ifdef USDGEN_ENABLE_CUDA
constexpr size_t kCudaWidthBranchStreamCount = 8;
#include "usdGen/cudaSourceInput.h"
#include "usdGen/cudaScatterInput.h"
#include "usdGen/surfaceRootBindings.h"
#include "usdGen/surfaceRootFrames.h"
#include "usdGen/curveRootCapture.h"
#include <tbb/flow_graph.h>
#include <tbb/task_arena.h>
#include "usdGen/gpu/generation.h"
#include "usdGen/gpu/namedChannelTopology.h"
#include "usdGen/cudaParameters.h"
#include "usdGen/gpu/width.h"
#include "usdGen/gpu/widthBlend.h"
#include "usdGen/gpu/nonWidthCompare.h"
#include "usdGen/gpu/noise.h"
#include "usdGen/gpu/imageSampler.h"
#include "usdGen/gpu/widthOverlapWitness.h"
#include "usdGen/cudaSurfaceInput.h"
#include "usdGen/gpu/surfaceBinding.h"
#include "usdGen/gpu/deformCurves.h"
#include "usdGen/gpu/length.h"
#include "usdGen/gpu/curveCompaction.h"
#include "usdGen/gpu/topology.h"
#include "usdGen/gpu/curveTiles.h"
#include "usdGen/gpu/curveTileBounds.h"
#include "usdGen/ops/width.h"
#include "usdGenMath/usdGenMath/ramp.h"
#endif

namespace usdGen {
namespace {
bool Fail(UsdGenDiagnostics* diagnostics, std::string message) {
    if (diagnostics) diagnostics->Error("CUDA: " + std::move(message));
    return false;
}
[[maybe_unused]] std::string LocalName(TfToken const& name) {
    auto value = name.GetString();
    if (value.compare(0, 7, "usdGen:") == 0) value.erase(0, 7);
    return value;
}
bool IsCudaWidthValueNode(TfToken const& type) {
    return type == TfToken("UsdGenWidth") || type == TfToken("UsdGenWidthBlend");
}
// UsdGenWidthBlend carries its own interpolation weight as an operator
// parameter; it is clamped to exact endpoints exactly as the CPU kernel does.
float CudaWidthBlendWeight(UsdGenGraphDesc const& desc, UsdGenNodeDesc const& node) {
    UsdGenParamView const view{&desc, &node};
    double const weight = view.GetDouble(TfToken("widthBlend:weight"), 1.0);
    if (!(weight > 0.0)) return 0.0f;
    return weight >= 1.0 ? 1.0f : static_cast<float>(weight);
}
#ifdef USDGEN_ENABLE_CUDA
// The per-operator expression allowlist is shared with the CPU lane and the
// compiler (expressionTargets.cpp). Keeping a second copy here is exactly how
// the two lanes would drift.
bool ValidateExpressionTargets(UsdGenNodeDesc const& node, UsdGenDiagnostics* diagnostics) {
    std::vector<std::string> errors;
    if (UsdGenValidateExpressionTargets(node, &errors)) return true;
    for (auto const& error : errors) Fail(diagnostics, error);
    return false;
}
bool ValidateLength(UsdGenNodeDesc const& node, UsdGenDiagnostics* diagnostics) {
    if (!node.mode.IsEmpty() ||
        !node.references.empty() || !node.curves.empty() || !node.maps.empty() ||
        !node.mapBindings.empty())
        return Fail(diagnostics, "unsupported Length mode/reference/map configuration");
    static const std::set<std::string> floats{
        "length:value", "minRemainingLength", "cullThreshold", "mask"};
    std::set<std::string> seen;
    for (auto const& param : node.params) {
        auto name = param.name.GetString();
        if (!seen.insert(name).second) return Fail(diagnostics, "duplicate Length parameter " + name);
        bool valid = false;
        if (floats.count(name))
            valid = param.value.IsHolding<float>() && std::isfinite(param.value.UncheckedGet<float>()) &&
                param.value.UncheckedGet<float>() >= 0 &&
                (name != "mask" || param.value.UncheckedGet<float>() <= 1);
        else if (name == "length:random") {
            valid = param.value.IsHolding<GfVec2f>();
            if (valid) for (int component = 0; component < 2; ++component) {
                auto value = param.value.UncheckedGet<GfVec2f>()[component];
                valid &= std::isfinite(value) && value >= 0;
            }
        } else if (name == "length:mode")
            valid = param.value == VtValue(TfToken("scale")) || param.value == VtValue(TfToken("set")) ||
                param.value == VtValue(TfToken("cull"));
        else if (name == "length:method")
            valid = param.value == VtValue(TfToken("scale")) || param.value == VtValue(TfToken("cutExtend"));
        else if (name == "rebuild")
            valid = param.value == VtValue(TfToken("keepParam")) || param.value == VtValue(TfToken("reparam"));
        // randomRange controls are inert here (that mode itself is refused
        // above); accept their schema defaults so ordinary Length nodes plan.
        else if (name == "length:min" || name == "length:max" || name == "length:randomize")
            valid = param.value.IsHolding<float>() && std::isfinite(param.value.UncheckedGet<float>());
        else if (name == "length:shareByClump" || name == "length:followClumpMean")
            valid = param.value.IsHolding<bool>();
        if (!valid) return Fail(diagnostics, "unsupported or malformed Length parameter " + name);
    }
    for (auto const& ramp : node.ramps)
        if (!ramp.knots.empty() || !ramp.colors.empty() || !ramp.positions.empty())
            return Fail(diagnostics, "Length ramps require named parameters");
    return ValidateExpressionTargets(node, diagnostics);
}
bool ValidateWidth(UsdGenNodeDesc const& node, UsdGenDiagnostics* diagnostics) {
    if (node.type != TfToken("UsdGenWidth"))
        return Fail(diagnostics, "only Width operators may follow CurveSource in this CUDA executor");
    // Surface inheritance is metadata on Width; this operator does not sample
    // it. Do not reject the Description's ordinary inherited scalp target.
    if (!node.references.empty() || !node.curves.empty() || !node.mode.IsEmpty())
        return Fail(diagnostics, "Width references/curves/mode are not supported");
    // The Bezier profile controls are literal floats; the profile itself is
    // folded into the per-node LUT (UsdGenWidthBezierLut) below.
    static const std::set<std::string> floats{
        "width", "mask", "width:root", "width:mid", "width:midShift", "width:tip"};
    std::set<std::string> seen;
    for (auto const& param : node.params) {
        auto const& name = param.name.GetString();
        if (!seen.insert(name).second) return Fail(diagnostics, "duplicate Width parameter " + name);
        bool valid = false;
        if (floats.count(name)) valid = param.value.IsHolding<float>() && std::isfinite(param.value.UncheckedGet<float>());
        else if (name == "replace") valid = param.value.IsHolding<bool>();
        else if (name == "width:interpolation") valid = param.value.IsHolding<TfToken>();
        else if (name == "width:profile:mode")
            valid = param.value == VtValue(TfToken("ramp")) || param.value == VtValue(TfToken("bezier"));
        else if (name == "width:knots") {
            valid = param.value.IsHolding<VtVec2fArray>();
            float previous = -1;
            if (valid) for (auto const& knot : param.value.UncheckedGet<VtVec2fArray>()) {
                if (!std::isfinite(knot[0]) || !std::isfinite(knot[1]) || knot[0] < 0 || knot[0] > 1 || knot[0] < previous)
                    valid = false;
                previous = knot[0];
            }
        }
        if (!valid) return Fail(diagnostics, "unsupported or malformed Width parameter " + name);
    }
    // Typed parameter ramps above are authoritative. Anonymous ramps cannot
    // be associated with a destination without guessing.
    for (auto const& ramp : node.ramps)
        if (!ramp.knots.empty() || !ramp.positions.empty() || !ramp.colors.empty())
            return Fail(diagnostics, "anonymous Width ramps require a named parameter");
    return ValidateExpressionTargets(node, diagnostics);
}

// WidthBlend is intentionally not a second configurable Width.  It consumes
// two already-proved immutable width values in authored order and owns no
// fields, maps, curves, or parameter evaluation.  Keeping the boundary this
// small prevents a generic multi-input operator from silently entering the
// CUDA COW DAG.
bool ValidateWidthBlend(UsdGenNodeDesc const& node, UsdGenDiagnostics* diagnostics) {
    bool paramsOk = true;
    for (UsdGenParamValue const& param : node.params)
        paramsOk = paramsOk && param.name == TfToken("widthBlend:weight") &&
            param.value.IsHolding<float>() &&
            std::isfinite(param.value.UncheckedGet<float>()) &&
            param.value.UncheckedGet<float>() >= 0.0f &&
            param.value.UncheckedGet<float>() <= 1.0f;
    if (node.type != TfToken("UsdGenWidthBlend") ||
        !paramsOk || !node.ramps.empty() ||
        !node.expressionBindings.empty() || !node.references.empty() ||
        !node.curves.empty() || !node.surfaces.empty() || !node.maps.empty() ||
        !node.mapBindings.empty() || !node.mode.IsEmpty() || !node.enabled)
        return Fail(diagnostics,
            "WidthBlend requires a finite usdGen:widthBlend:weight in [0,1] and no auxiliary inputs");
    return true;
}

bool ValidateNoise(UsdGenNodeDesc const& node, UsdGenDiagnostics* diagnostics) {
    // The imaging builders copy the Description's usdGen:surface onto every
    // node, so a stage-authored Noise always carries the scalp target. Noise
    // does not sample it; rejecting it would make Noise unusable from a stage
    // (the same reason ValidateWidth accepts the inherited target).
    if (node.type != TfToken("UsdGenNoise") || !node.references.empty() ||
        !node.curves.empty() || !node.maps.empty() || !node.mapBindings.empty() ||
        !node.mode.IsEmpty())
        return Fail(diagnostics, "Noise does not accept references, curves, maps, or mode");
    static const std::set<std::string> floats{
        "noise:magnitude", "noise:frequency", "noise:correlation",
        "noise:lacunarity", "noise:gain", "preserveLength", "mask"};
    std::set<std::string> seen;
    for (auto const& param : node.params) {
        auto const name = LocalName(param.name);
        if (!seen.insert(name).second) return Fail(diagnostics, "duplicate Noise parameter " + name);
        bool valid = false;
        if (floats.count(name)) {
            valid = param.value.IsHolding<float>() &&
                std::isfinite(param.value.UncheckedGet<float>());
            if (valid && (name == "noise:magnitude" || name == "noise:frequency" ||
                          name == "noise:lacunarity" || name == "preserveLength" ||
                          name == "mask"))
                valid = param.value.UncheckedGet<float>() >= 0;
            if (valid && name == "noise:frequency")
                valid = param.value.UncheckedGet<float>() > 0;
            if (valid && name == "noise:lacunarity")
                valid = param.value.UncheckedGet<float>() > 1;
            if (valid && (name == "noise:correlation" || name == "noise:gain" ||
                          name == "preserveLength" || name == "mask"))
                valid = param.value.UncheckedGet<float>() <= 1;
        } else if (name == "noise:octaves" || name == "noise:seed") {
            valid = param.value.IsHolding<int>();
            if (valid && name == "noise:octaves")
                valid = param.value.UncheckedGet<int>() >= 1 &&
                    param.value.UncheckedGet<int>() <= 6;
        } else if (name == "enabled" || name == "cumulative") {
            valid = param.value.IsHolding<bool>();
        } else if (name == "noise:magnitude:interpolation") {
            valid = param.value.IsHolding<TfToken>();
        } else if (name == "noise:magnitude:knots") {
            valid = param.value.IsHolding<VtVec2fArray>();
            float previous = -1;
            if (valid) for (auto const& knot : param.value.UncheckedGet<VtVec2fArray>()) {
                valid = std::isfinite(knot[0]) && std::isfinite(knot[1]) &&
                    knot[0] >= 0 && knot[0] <= 1 && knot[0] >= previous;
                previous = knot[0];
            }
        } else if (name == "label") {
            valid = param.value.IsHolding<std::string>();
        }
        if (!valid) return Fail(diagnostics, "unsupported or malformed Noise parameter " + name);
    }
    for (auto const& ramp : node.ramps)
        if (!ramp.knots.empty() || !ramp.colors.empty() || !ramp.positions.empty())
            return Fail(diagnostics, "Noise ramps require named parameters");
    return ValidateExpressionTargets(node, diagnostics);
}

bool ValidateDeform(UsdGenNodeDesc const& node, UsdGenDiagnostics* diagnostics) {
    if (!node.mode.IsEmpty())
        return Fail(diagnostics, "Deform mode requires the CPU lane; CUDA supports default surface RBF");
    if (node.surfaces.size() != 1 || !node.references.empty() || !node.curves.empty() ||
        !node.maps.empty() || !node.mapBindings.empty())
        return Fail(diagnostics, "RBF Deform requires exactly one surface and no guide/map inputs "
                                 "(guide-driven RBF, usdGen:guides, runs on the CPU lane)");
    std::set<std::string> seen;
    for (auto const& param : node.params) {
        auto name = param.name.GetString();
        bool valid = seen.insert(name).second;
        if (name == "rbfSamples")
            valid &= param.value.IsHolding<int>() && param.value.UncheckedGet<int>() >= 4 && param.value.UncheckedGet<int>() <= 46336;
        else if (name == "lockRoots") valid &= param.value.IsHolding<bool>();
        // The schema supplies this empty curveWrap-only default even for
        // surface RBF. Accept the inert default without admitting guide
        // partitioning, which remains a CPU capability.
        else if (name == "guideRegions")
            valid &= param.value.IsHolding<VtIntArray>() &&
                param.value.UncheckedGet<VtIntArray>().empty();
        else if (name == "mask")
            valid &= param.value.IsHolding<float>() && std::isfinite(param.value.UncheckedGet<float>()) &&
                param.value.UncheckedGet<float>() >= 0 && param.value.UncheckedGet<float>() <= 1;
        else valid = false;
        if (!valid) return Fail(diagnostics, "unsupported or malformed RBF parameter " + name);
    }
    for (auto const& ramp : node.ramps)
        if (!ramp.knots.empty() || !ramp.colors.empty() || !ramp.positions.empty())
            return Fail(diagnostics, "RBF ramps require named parameters");
    return ValidateExpressionTargets(node, diagnostics);
}

bool ValidateScatterGrow(UsdGenGraphDesc const& desc, uint32_t source, uint32_t terminal,
                         UsdGenDiagnostics* diagnostics, bool c3 = false) {
    auto const& scatter = desc.nodes[source];
    // Scatter's source program is immutable root capture, not the connected
    // CurveSource parameter evaluator. Reject fields this route cannot read.
    if (!c3 && !scatter.expressionBindings.empty())
        return Fail(diagnostics, "CUDA Scatter does not support connected expression controls");
    for(auto const& param:scatter.params)
        if(param.name==TfToken("subdivisionLevel") &&
           (!param.value.IsHolding<int>() || param.value.UncheckedGet<int>()!=0))
            return Fail(diagnostics,"OpenSubdiv limit Scatter requires the CPU reference backend");
    // Distribution, surface counting, proximity and density maps are CPU
    // reference behaviours; refuse them instead of running random scatter.
    for (auto const& param : scatter.params) {
        auto nonDefault = [&](char const* name, char const* fallback) {
            return param.name == TfToken(name) &&
                !(param.value.IsHolding<TfToken>() &&
                  param.value.UncheckedGet<TfToken>() == TfToken(fallback));
        };
        if (nonDefault("distribution", "random") || nonDefault("countMode", "perFace") ||
            nonDefault("proximity:mode", "none"))
            return Fail(diagnostics,
                "Scatter distribution/count/proximity modes require the CPU reference backend");
    }
    if (!c3 && (!scatter.mapBindings.empty() || !scatter.curves.empty()))
        return Fail(diagnostics,
            "Scatter density maps and guides require the CPU reference backend");
    auto const& grow = desc.nodes[terminal];
    if (!grow.mode.IsEmpty())
        return Fail(diagnostics, "Scatter->Grow has no Grow mode property");
    if (!std::isfinite(desc.defaultWidth) || desc.defaultWidth < 0)
        return Fail(diagnostics, "description default width must be finite and non-negative");
    // The imaging builders copy the description's usdGen:surface onto every
    // operator that authors none of its own, so a Grow reaching this slice
    // normally carries the same surface as its Scatter. That is the surface
    // the kernel grows from anyway; only a different one is unsupported.
    if (!grow.references.empty() || !grow.curves.empty() ||
        !grow.expressionBindings.empty() || !grow.ramps.empty() ||
        (!grow.surfaces.empty() && grow.surfaces != scatter.surfaces))
        return Fail(diagnostics, "Scatter->Grow does not support Grow references/maps/expressions/ramps");
    // Do not use ParamView for admission here: its convenience conversions
    // deliberately substitute defaults for malformed native values.  This
    // compact GPU slice has no alternate interpretation for an authored
    // control, so every authored value must be both unique and native-typed.
    static std::set<TfToken> const allowed{
        TfToken("segments"), TfToken("length"), TfToken("lengthRandom"),
        TfToken("lift"), TfToken("azimuth"), TfToken("azimuthRandom"),
        TfToken("direction"), TfToken("directionVector")};
    std::set<TfToken> seen;
    for (auto const& value : grow.params) {
        if (value.animated || !seen.insert(value.name).second)
            return Fail(diagnostics, "Scatter->Grow has animated or duplicate Grow control " +
                value.name.GetString());
        if (!allowed.count(value.name))
            return Fail(diagnostics, "Scatter->Grow has unsupported Grow control " +
                value.name.GetString());
        bool valid = false;
        if (value.name == TfToken("segments")) valid = value.value.IsHolding<int>();
        else if (value.name == TfToken("length") || value.name == TfToken("lift") ||
                 value.name == TfToken("azimuth") || value.name == TfToken("azimuthRandom"))
            valid = value.value.IsHolding<float>() || value.value.IsHolding<double>();
        else if (value.name == TfToken("lengthRandom")) valid = value.value.IsHolding<GfVec2f>();
        else if (value.name == TfToken("direction"))
            valid = value.value.IsHolding<TfToken>() || value.value.IsHolding<std::string>();
        else if (value.name == TfToken("directionVector")) valid = value.value.IsHolding<GfVec3f>();
        if (!valid)
            return Fail(diagnostics, "Scatter->Grow has wrong native type for " + value.name.GetString());
    }
    UsdGenParamView p{&desc, &grow};
    int const segments=p.GetInt(TfToken("segments"),8);
    double const length=p.GetDouble(TfToken("length"),1.0);
    if (segments<2 || segments>64 || !std::isfinite(length) || length<0)
        return Fail(diagnostics, "Scatter->Grow requires segments in [2,64] and finite non-negative length");
    if (!std::isfinite(float(length)))
        return Fail(diagnostics, "Scatter->Grow length is not float-representable");
    auto random=p.GetVtValue(TfToken("lengthRandom"),VtValue(GfVec2f(1,1)));
    if (!random.IsHolding<GfVec2f>()) return Fail(diagnostics,"Scatter->Grow lengthRandom must be float2");
    auto const r=random.UncheckedGet<GfVec2f>();
    if (!std::isfinite(r[0])||!std::isfinite(r[1])||r[0]<0||r[1]<0)
        return Fail(diagnostics,"Scatter->Grow lengthRandom must be finite and non-negative");
    double const lift=p.GetDouble(TfToken("lift"),0.0);
    if (!std::isfinite(lift) || lift < -90.0 || lift > 90.0)
        return Fail(diagnostics,"Grow requires lift in [-90,90] degrees");
    double const azimuth=p.GetDouble(TfToken("azimuth"),0.0);
    double const azimuthRandom=p.GetDouble(TfToken("azimuthRandom"),0.0);
    if (!std::isfinite(azimuth) || azimuth < -360.0 || azimuth > 360.0 ||
        !std::isfinite(azimuthRandom) || azimuthRandom < 0.0 || azimuthRandom > 1.0)
        return Fail(diagnostics,"Grow requires azimuth in [-360,360] degrees and azimuthRandom in [0,1]");
    TfToken direction=p.GetToken(TfToken("direction"),TfToken("surfaceNormal"));
    if (direction!=TfToken("surfaceNormal") && direction!=TfToken("vector"))
        return Fail(diagnostics,"Grow requires surfaceNormal or literal vector direction");
    auto v=p.GetVtValue(TfToken("directionVector"),VtValue(GfVec3f(0,1,0)));
    if (!v.IsHolding<GfVec3f>()) return Fail(diagnostics,"Scatter->Grow directionVector must be float3");
    auto const d=v.UncheckedGet<GfVec3f>();
    if (!std::isfinite(d[0])||!std::isfinite(d[1])||!std::isfinite(d[2]))
        return Fail(diagnostics,"Scatter->Grow directionVector must be finite");
    return true;
}

// CPU lowering resolves node.references/node.maps into the operator's
// ReferenceInputs/map payloads.  The constrained CUDA ReferenceSource path
// consumes one resolved immutable reference curve set through its own source
// upload boundary; all other unresolved references/maps remain rejected before
// layout, parameter compilation, or any device-facing work.
bool ValidateCudaResolvedInputs(UsdGenGraphDesc const& desc,
                                UsdGenDiagnostics* diagnostics) {
    for (auto const& node : desc.nodes) {
        // Width and Grow declare no map inputs of their own; every other
        // unresolved map remains a hard failure.
        if ((node.type == TfToken("UsdGenWidth") || node.type == TfToken("UsdGenGrow")) &&
            node.references.empty()) continue;
        auto const references = node.references.size();
        auto const maps = node.mapBindings.empty() ? node.maps.size()
                                                   : node.mapBindings.size();
        if (references == 0 && maps == 0) continue;
        if (node.type == TfToken("UsdGenCurveSource") &&
            references == 0 && maps == 1 && node.curves.size() == 1 &&
            node.surfaces.size() == 1 && node.inputs.empty() &&
            node.mapBindings.size() == 1 &&
            node.mapBindings.front().relationship == TfToken("usdGen:regionMap")) {
            // surfaceCage expands on the CPU source-preparation boundary
            // before ordinary C3 upload; detailed cage/map validation occurs
            // in the same shared bridge used by the CPU CurveSource lane.
            continue;
        }
        if (node.type == TfToken("UsdGenReferenceSource") &&
            references == 1 && maps == 0 && node.curves.empty() &&
            node.surfaces.empty() && node.inputs.empty()) {
            auto const found = std::find_if(desc.curveSets.begin(), desc.curveSets.end(),
                [&](auto const& curves) { return curves.path == node.references.front(); });
            if (found != desc.curveSets.end() && found->role == UsdGenRole::Reference) {
                if (found->authoredPlanes.empty() && found->rootFrame.empty())
                    continue;
                if (!found->authoredPlanes.empty())
                    return Fail(diagnostics,
                        "UsdGenReferenceSource requires one authored reference curve set without named planes at " +
                        node.path.GetString());
                return Fail(diagnostics,
                    "UsdGenReferenceSource rootFrame bindings are not supported by CUDA at " +
                    node.path.GetString());
            }
            return Fail(diagnostics,
                "UsdGenReferenceSource requires one authored reference curve set without named planes at " +
                node.path.GetString());
        }
        return Fail(diagnostics,
            "unresolved ReferenceInputs/maps are not supported by CUDA at " +
            node.path.GetString() + " (references=" +
            std::to_string(references) + ", maps=" + std::to_string(maps) + ")");
    }
    return true;
}

#endif
}

class UsdGenCudaExecutionPlan {
public:
    UsdGenGraphDesc desc;
    std::shared_ptr<const UsdGenExecutionPlanMetadata> metadata;
#ifdef USDGEN_ENABLE_CUDA
    std::shared_ptr<const CudaParameterProgram> sourceParameters;
    uint32_t sourceNode = 0;
    uint32_t terminalNode = 0;
    bool taskDag = false;
    bool widthDag = false;
    // A value DAG whose branches share immutable curve topology but may own
    // distinct point and width planes (Noise/Width/WidthBlend).
    bool sameTopologyDag = false;
    // General immutable value DAG.  Direct CurveSource Grow values and
    // authored-predecessor Length values may each publish independent
    // topology/named snapshots.
    bool geometryValueDag = false;
    // The topology-generator path has no authored CurveSource. Scatter is a
    // capture source and Grow is its single CUDA topology-expansion consumer.
    // Keep the shape explicit so source preparation, resource accounting and
    // publication never reinterpret captured roots as authored curve input.
    bool scatterGrow = false;
    // A surface-cage CurveSource expands on the host before C3 upload.  Keep
    // its descriptor-only upper bound on the plan so runtime preparation can
    // prove the actual categorical-map result stayed within the capacity that
    // was admitted during compilation.
    uint64_t surfaceCageMaxCurves = 0;
    uint64_t surfaceCageMaxCvs = 0;
    // Semantic Grow output that seeds the Width value DAG.  This is distinct
    // from sourceNode, which remains Scatter for immutable CPU capture.
    uint32_t scatterGrowOutputNode = UINT32_MAX;
    std::shared_ptr<const gpu::ScatterGrowRoots> scatterRoots;
    gpu::ScatterGrowControls scatterGrowControls;
    gpu::ScatterGrowRequirements scatterGrowRequirements;
    struct Step {
        SdfPath path;
        TfToken type;
        uint32_t semanticNode = 0;
        uint32_t inputNode = 0;
        // Ordered second predecessor for the narrow binary WidthBlend only.
        // UINT32_MAX keeps all existing unary lowering structurally intact.
        uint32_t rightInputNode = UINT32_MAX;
        bool requiresNonWidthProof = false;
        std::shared_ptr<const CudaParameterProgram> parameters;
        std::array<float, kUsdGenRampLutSize> profile{};
        CudaSurfacePrepared surface;
        gpu::CurveGrowControls grow;
    };
    std::vector<std::unique_ptr<const Step>> steps;
#endif
};

#ifdef USDGEN_ENABLE_CUDA
struct CudaRbfCache {
    struct State {
        struct Resources {
            gpu::CudaSurfaceBinding surface;
            gpu::CudaRbfBinding field;
            gpu::DeviceBuffer<float3> currentVertices;
        };
        int device = -1;
        uint32_t sampleBudget = 0;
        uint64_t identity = 0;
        std::unique_ptr<Resources> resources = std::make_unique<Resources>();
        ~State() {
            int previous = -1;
            cudaGetDevice(&previous);
            if (device < 0 || cudaSetDevice(device) != cudaSuccess) {
                resources.release(); // cannot safely free a foreign/lost context
                return;
            }
            resources.reset();
            if (previous >= 0 && previous != device) cudaSetDevice(previous);
        }
    };
    CudaSurfaceBindingKey key;
    std::unique_ptr<State> state;
    // Retired only at the next ordinary relay-launch boundary, never from a
    // host-only candidate acceptance commit.
    std::unique_ptr<State> retiredState;
    uint64_t bindCount = 0, solveCount = 0;
};

// A proved RBF operator result is still only a publication candidate.  Keep
// its surface pose, coefficients, and (for a rest change) whole cache state
// provisional until final tile/bounds/topology construction has also passed.
struct CudaRbfPendingPublication {
    CudaRbfCache* cache = nullptr;
    std::unique_ptr<CudaRbfCache::State> freshState;
    CudaRbfCache::State* state = nullptr;
    CudaSurfaceBindingKey key;
    bool rebind = false;
    bool resetCounters = false;
    bool solveCommitted = false;
    bool surfaceCommitted = false;
};

namespace {
std::string CudaStatusText(cudaError_t status) {
    return std::string(cudaGetErrorName(status)) + " (" +
        std::to_string(static_cast<int>(status)) + ": " +
        cudaGetErrorString(status) + ")";
}
std::string CurveSourceStatusText(gpu::CurveSourceStatus status) {
    char const* name = "Unknown";
    switch (status) {
    case gpu::CurveSourceStatus::Ok: name = "Ok"; break;
    case gpu::CurveSourceStatus::InvalidArgument: name = "InvalidArgument"; break;
    case gpu::CurveSourceStatus::NonFiniteInput: name = "NonFiniteInput"; break;
    case gpu::CurveSourceStatus::InvalidTopology: name = "InvalidTopology"; break;
    case gpu::CurveSourceStatus::DuplicateStableId: name = "DuplicateStableId"; break;
    case gpu::CurveSourceStatus::NoPendingUpdate: name = "NoPendingUpdate"; break;
    case gpu::CurveSourceStatus::CudaError: name = "CudaError"; break;
    }
    return std::string(name) + " (" + std::to_string(static_cast<int>(status)) + ")";
}
struct CudaDeviceScope {
    int previous = -1;
    bool selected = false;
    cudaError_t status = cudaSuccess;
    char const* operation = "cudaGetDevice";
    explicit CudaDeviceScope(int device) {
        status = cudaGetDevice(&previous);
        if (status == cudaSuccess) {
            operation = "cudaSetDevice";
            status = cudaSetDevice(device);
        }
        selected = status == cudaSuccess;
    }
    std::string FailureDetail() const { return std::string(operation) + " returned " + CudaStatusText(status); }
    void Restore() noexcept { if (selected) { cudaSetDevice(previous); selected = false; } }
    ~CudaDeviceScope() { Restore(); }
};
std::atomic<bool> s_failNextCudaWidthBlendGraphCapture{false};

// A deliberately narrow, workspace-private CUDA Graph specialization.  The
// graph never observes generation-owned COW planes: stable cache slots carry
// the leaf's inputs/output, and the caller copies the result into its fresh
// WidthValue before publication.  This makes the native graph executable
// reusable without allowing mutable workspace storage to escape as a device
// generation.
struct CudaWidthBlendGraphCache {
    static constexpr size_t kConservativeGraphChargeBytes = 64u * 1024u;

    struct Key {
        SdfPath path, left, right;
        size_t count = 0;
        uint32_t blendBits = 0;
        // Captured launch geometry is part of the executable contract even
        // though the current kernel derives its grid solely from count.
        uint32_t blockThreads = 256;
        uint64_t gridBlocks = 0;
        bool operator==(Key const& rhs) const noexcept {
            return path == rhs.path && left == rhs.left && right == rhs.right &&
                count == rhs.count && blendBits == rhs.blendBits &&
                blockThreads == rhs.blockThreads && gridBlocks == rhs.gridBlocks;
        }
    };

    class Lease {
    public:
        Lease() = default;
        ~Lease() { Release(); }
        Lease(Lease const&) = delete;
        Lease& operator=(Lease const&) = delete;
        Lease(Lease&& other) noexcept : cache_(other.cache_) { other.cache_ = nullptr; }
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) { Release(); cache_ = other.cache_; other.cache_ = nullptr; }
            return *this;
        }
        explicit operator bool() const noexcept { return cache_ != nullptr; }
        void Quarantine() noexcept { if (cache_) cache_->Quarantine(); }
        void Release() noexcept {
            if (cache_) cache_->inFlight_.store(false, std::memory_order_release);
            cache_ = nullptr;
        }
    private:
        explicit Lease(CudaWidthBlendGraphCache* cache) : cache_(cache) {}
        CudaWidthBlendGraphCache* cache_ = nullptr;
        friend struct CudaWidthBlendGraphCache;
    };

    explicit CudaWidthBlendGraphCache(int device) : device_(device) {}
    ~CudaWidthBlendGraphCache() {
        // Capture itself queues no executable work, but retain the same proof
        // discipline as the workspace streams before destroying its handle.
        if (captureStream_ && cudaStreamSynchronize(captureStream_) != cudaSuccess) {
            Quarantine();
            return;
        }
        Destroy();
        if (!quarantined_.load(std::memory_order_acquire) && captureStream_)
            cudaStreamDestroy(captureStream_);
    }
    CudaWidthBlendGraphCache(CudaWidthBlendGraphCache const&) = delete;
    CudaWidthBlendGraphCache& operator=(CudaWidthBlendGraphCache const&) = delete;

    bool Acquire(Key const& key, Lease* result) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!result || quarantined_.load(std::memory_order_acquire) || !key.count ||
            !EnsureCaptureStream()) return false;
        bool expected = false;
        if (valid_ && key_ == key && inFlight_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
            replays_.fetch_add(1, std::memory_order_relaxed);
            *result = Lease(this);
            return true;
        }
        misses_.fetch_add(1, std::memory_order_relaxed);
        if (valid_) {
            if (inFlight_.load(std::memory_order_acquire) || !EvictIdle()) return false;
        }
        // A failed cache creation is a clean miss.  It must not consume the
        // job reservation because this storage survives the job.
        auto charge = gpu::TryReserveCudaExecutionBytes(kConservativeGraphChargeBytes,
            UsdGenExecutionResourceKind::Cache);
        if (!charge || left_.reset(key.count, nullptr, UsdGenExecutionResourceKind::Cache) != cudaSuccess ||
            right_.reset(key.count, nullptr, UsdGenExecutionResourceKind::Cache) != cudaSuccess ||
            output_.reset(key.count, nullptr, UsdGenExecutionResourceKind::Cache) != cudaSuccess) {
            left_.release(); right_.release(); output_.release();
            return false;
        }
        if (s_failNextCudaWidthBlendGraphCapture.exchange(false, std::memory_order_acq_rel)) {
            left_.release(); right_.release(); output_.release(); return false;
        }
        cudaGraph_t graph = nullptr;
        // This capture touches only this cache's preallocated private slot on
        // captureStream_.  Thread-local mode preserves CUDA's same-thread
        // unsafe-call protection without rejecting unrelated relay cleanup
        // (including DeviceBuffer release) on worker threads.
        if (cudaStreamBeginCapture(captureStream_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            left_.release(); right_.release(); output_.release(); return false;
        }
        auto const left = static_cast<gpu::DeviceBuffer<float> const&>(left_).view();
        auto const right = static_cast<gpu::DeviceBuffer<float> const&>(right_).view();
        float blend = 0.0f; std::memcpy(&blend, &key.blendBits, sizeof(blend));
        bool const launched = gpu::LaunchWidthBlend(left, right, blend, output_.view(), captureStream_);
        cudaError_t const ended = cudaStreamEndCapture(captureStream_, &graph);
        cudaGraphExec_t executable = nullptr;
        if (!launched || ended != cudaSuccess || !graph ||
            cudaGraphInstantiate(&executable, graph, 0) != cudaSuccess) {
            if (executable) cudaGraphExecDestroy(executable);
            if (graph) cudaGraphDestroy(graph);
            left_.release(); right_.release(); output_.release(); return false;
        }
        try { key_ = key; }
        catch (...) {
            cudaGraphExecDestroy(executable); cudaGraphDestroy(graph);
            left_.release(); right_.release(); output_.release(); return false;
        }
        graphCharge_ = std::move(*charge); graph_ = graph; executable_ = executable;
        retainedCacheBytes_.store(3 * key.count * sizeof(float) +
                                      kConservativeGraphChargeBytes,
                                  std::memory_order_release);
        valid_ = true; captures_.fetch_add(1, std::memory_order_relaxed);
        expected = false;
        if (!inFlight_.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
            // This cache is workspace-private and construction occurs on its
            // owner lane. A concurrent claim here is still safe: keep the
            // freshly built entry resident and take the ordinary fallback.
            return false;
        }
        *result = Lease(this);
        return true;
    }

    bool CopyLaunchCopy(gpu::DeviceView<const float> left,
                        gpu::DeviceView<const float> right,
                        gpu::DeviceView<float> freshOutput,
                        cudaStream_t stream) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid_ || !stream || left.size != key_.count || right.size != key_.count ||
            freshOutput.size != key_.count || !left.data || !right.data || !freshOutput.data)
            return false;
        if (cudaMemcpyAsync(left_.data(), left.data, key_.count * sizeof(float),
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(right_.data(), right.data, key_.count * sizeof(float),
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            cudaGraphLaunch(executable_, stream) != cudaSuccess ||
            cudaMemcpyAsync(freshOutput.data, output_.data(), key_.count * sizeof(float),
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            left_.recordUse(stream) != cudaSuccess || right_.recordUse(stream) != cudaSuccess ||
            output_.recordUse(stream) != cudaSuccess)
            return false;
        return true;
    }

    bool EvictIdle() noexcept {
        if (!valid_) return true;
        if (inFlight_.load(std::memory_order_acquire)) return false;
        auto const left = left_.queryUse(), right = right_.queryUse(), output = output_.queryUse();
        if (left == cudaErrorNotReady || right == cudaErrorNotReady || output == cudaErrorNotReady)
            return false;
        if (left != cudaSuccess || right != cudaSuccess || output != cudaSuccess) {
            QuarantineLocked(); return false;
        }
        if (cudaGraphExecDestroy(executable_) != cudaSuccess || cudaGraphDestroy(graph_) != cudaSuccess) {
            executable_ = nullptr; graph_ = nullptr; QuarantineLocked(); return false;
        }
        executable_ = nullptr; graph_ = nullptr; valid_ = false;
        left_.release(); right_.release(); output_.release(); graphCharge_.Release();
        retainedCacheBytes_.store(0, std::memory_order_release);
        evictions_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void Quarantine() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        QuarantineLocked();
    }

    UsdGenCudaWidthBlendGraphStats Stats() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        UsdGenCudaWidthBlendGraphStats result;
        result.captures = captures_.load(std::memory_order_relaxed);
        result.replays = replays_.load(std::memory_order_relaxed);
        result.misses = misses_.load(std::memory_order_relaxed);
        result.evictions = evictions_.load(std::memory_order_relaxed);
        result.quarantines = quarantines_.load(std::memory_order_relaxed);
        result.inFlight = inFlight_.load(std::memory_order_acquire);
        result.quarantined = quarantined_.load(std::memory_order_acquire);
        // A quarantined entry deliberately abandons its permits and native
        // handles.  Keep that reservation visible; reporting zero here would
        // make the ledger look healthier than the conservative retirement is.
        result.cacheBytes = retainedCacheBytes_.load(std::memory_order_acquire);
        return result;
    }

private:
    void QuarantineLocked() noexcept {
        if (quarantined_.exchange(true, std::memory_order_acq_rel)) return;
        // Do not issue CUDA destroys when completion/context proof was lost.
        executable_ = nullptr; graph_ = nullptr; valid_ = false;
        left_.quarantine(); right_.quarantine(); output_.quarantine(); graphCharge_.Abandon();
        quarantines_.fetch_add(1, std::memory_order_relaxed);
    }
    bool EnsureCaptureStream() noexcept {
        return captureStream_ ||
            cudaStreamCreateWithFlags(&captureStream_, cudaStreamNonBlocking) == cudaSuccess;
    }
    void Destroy() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (quarantined_.load(std::memory_order_acquire)) return;
        if (!EvictIdle()) QuarantineLocked();
    }
    int device_ = -1;
    cudaStream_t captureStream_ = nullptr;
    gpu::DeviceBuffer<float> left_, right_, output_;
    UsdGenExecutionResourcePermit graphCharge_;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    Key key_;
    std::atomic<bool> inFlight_{false};
    std::atomic<bool> quarantined_{false};
    std::atomic<uint64_t> captures_{0}, replays_{0}, misses_{0}, evictions_{0}, quarantines_{0};
    std::atomic<size_t> retainedCacheBytes_{0};
    bool valid_ = false;
    mutable std::mutex mutex_;
};

CudaWidthBlendGraphCache::Key MakeWidthBlendGraphKey(
    UsdGenCudaExecutionPlan const& plan, UsdGenCudaExecutionPlan::Step const& step,
    size_t count) {
    CudaWidthBlendGraphCache::Key key;
    auto const& node = plan.desc.nodes[step.semanticNode];
    key.path = step.path;
    key.left = node.inputs.empty() ? SdfPath{} : node.inputs.front();
    key.right = node.inputs.size() < 2 ? SdfPath{} : node.inputs[1];
    key.count = count;
    float const weight = CudaWidthBlendWeight(plan.desc, node);
    std::memcpy(&key.blendBits, &weight, sizeof(key.blendBits));
    key.gridBlocks = count ? 1 + (count - 1) / key.blockThreads : 0;
    return key;
}
struct CudaInlineCompletion {
    std::atomic<int> status{int(cudaErrorUnknown)};
};
void CudaInlineCompletionCallback(cudaStream_t, cudaError_t status,
                                  void* userdata) noexcept {
    static_cast<CudaInlineCompletion*>(userdata)->status.store(
        int(status), std::memory_order_release);
}
}
#endif

struct UsdGenCudaExecutionWorkspace::Impl {
    int device = -1;
    SdfPath description;
    std::shared_ptr<const std::vector<UsdGenCudaBindingStats>> stats =
        std::make_shared<const std::vector<UsdGenCudaBindingStats>>();
#ifdef USDGEN_ENABLE_CUDA
    cudaStream_t stream = nullptr;
    // Width-only fan-out uses this fixed, workspace-owned pool.  Handles are
    // made with the workspace (never lazily from relay workers), so a relay
    // cannot outlive the stream it submitted to.  The primary stream remains
    // the sole lane for source, Length, Deform/RBF, and finalization.
    std::array<cudaStream_t, kCudaWidthBranchStreamCount> widthStreams{};
    std::unique_ptr<CudaWidthBlendGraphCache> widthBlendGraph;
    // Workspace-local immutable spatial index for C3 automatic bindings.
    // Replacement is candidate-based: an in-flight capture never mutates or
    // invalidates the cache used by an already accepted source generation.
    std::shared_ptr<const UsdGenRestSurfaceBindingCache> sourceBindingCache;
    std::atomic<bool> poisoned{false};
    std::unique_ptr<CudaParameterEvaluator> sourceParameters;
    struct Step {
        CudaParameterEvaluator parameters;
        std::unique_ptr<CudaRbfCache> rbf;
    };
    std::map<SdfPath, std::unique_ptr<Step>> steps;
#endif
};

#ifdef USDGEN_ENABLE_CUDA
struct CurveGrowInputOwners {
    std::unique_ptr<gpu::CudaCurveSource> source;
    std::unique_ptr<gpu::CudaCurveResample> resampled;
    std::unique_ptr<gpu::DeviceBuffer<float3>> t, b, n;
};
// CUDA copies from these immutable host vectors are asynchronous.  Keep their
// exact lifetime with the source transaction; pageable-memory staging is not
// a proof that the caller may release the vectors at cudaMemcpyAsync return.
struct ScatterGrowFrameHostPacket {
    std::vector<uint64_t> ids;
    std::vector<uint32_t> ordinals;
    std::vector<float3> t, b, n;
};
// Immutable topology ownership is deliberately distinct from point/width
// overlays.  A source-side branch can therefore continue to name the source
// generation after a sibling Grow/Length has produced a newer topology.
struct TopologyOwners {
    std::unique_ptr<gpu::CudaCurveSource> source;
    std::unique_ptr<gpu::CudaCurveResample> resampled;
    std::unique_ptr<gpu::CudaScatterGrow> scatterGrow;
    std::unique_ptr<gpu::CudaCurveGrow> grow;
    std::unique_ptr<gpu::CudaCurveCompaction> compaction;
    std::unique_ptr<gpu::DeviceBuffer<float3>> t, b, n;
    // Frame axes are addressed by stable ID, not capture ordinal.  Scatter
    // roots are permitted to arrive unsorted, so retain the independently
    // sorted key plane with the COW frame axes.
    std::unique_ptr<gpu::DeviceBuffer<uint64_t>> frameStableIds;
    std::unique_ptr<gpu::DeviceBuffer<uint32_t>> frameCaptureOrdinals;
    std::shared_ptr<const ScatterGrowFrameHostPacket> frameHostPacket;
};
struct NamedOwners {
    std::vector<gpu::CudaNamedChannelPlane> planes;
};
// A Grow producer is a value just like Length: its native work and named
// topology rewrite are private to one relay until the complete packet has
// proved.  In particular, source-side branches must never observe or borrow
// the mutable legacy `curveGrow` slot.
struct GrowValueInputLifetime {
    // The native producer reads the actual predecessor geometry/root/names,
    // while its axes and ID map remain the immutable original source frame
    // domain.  Async relays also retain their job so point/width overlays in
    // GeometryValue cannot disappear before native completion is proved.
    std::shared_ptr<TopologyOwners> topology;
    std::shared_ptr<TopologyOwners> sourceFrames;
    std::shared_ptr<NamedOwners> names;
    std::shared_ptr<const void> job;
};
struct OperatorGrowCandidate {
    std::shared_ptr<GrowValueInputLifetime> inputLifetime;
    std::unique_ptr<gpu::CudaCurveGrow> grow;
    std::unique_ptr<gpu::CudaNamedChannelTopologyCandidate> namedTopology;
    bool unsafeInput = false;
};
// The optional per-curve channels an expression context reads on top of the
// geometry: hairT and the root UVs as before, plus the rest root frame
// ($N/$Nref <- rootN, $dPdu/$dPduref <- rootT, $dPdv/$dPdvref <- rootB) and the
// surface face index ($faceId <- rootPrim).
//
// Those last four are attached only when they are already addressed by the
// current curve index: the plane must be curveCount long, and a frame set that
// carries its own stable-ID key must key on the geometry's own IDs.  A frame
// set from another curve domain would need a gather the context has no
// machinery for, so it stays null and the variable keeps poisoning rather than
// reading another curve's frame.
gpu::ExpressionGeometryChannels ExpressionChannels(
    gpu::DeviceCurveGeometryView const& geometry,
    gpu::DeviceView<const float> hairT,
    gpu::DeviceView<const float2> rootUV,
    gpu::DeviceView<const int32_t> rootPrim,
    gpu::RestRootFrames const& frames) {
    gpu::ExpressionGeometryChannels channels;
    channels.hairT = hairT;
    channels.rootUV = rootUV;
    if (rootPrim.data && rootPrim.size == geometry.curveCount)
        channels.rootPrim = rootPrim;
    if (frames.tangent.data && frames.binormal.data && frames.normal.data &&
        frames.tangent.size == geometry.curveCount &&
        frames.binormal.size == geometry.curveCount &&
        frames.normal.size == geometry.curveCount &&
        (!frames.stableIds.data || frames.stableIds.data == geometry.stableIds.data)) {
        channels.rootT = frames.tangent;
        channels.rootB = frames.binormal;
        channels.rootN = frames.normal;
    }
    return channels;
}
// Candidate-owned producer state.  Source/resample stay alive even after a
// later Length compaction: captureStableIds deliberately refers to that
// original ordered producer during terminal tile construction.
struct ExecutionState {
    // Non-owning view of the job's precharged balance. The owning job declares
    // its reservation before this state so all candidate buffers are destroyed
    // or transferred before the balance closes.
    UsdGenExecutionMemoryReservation* memoryReservation = nullptr;
    std::unique_ptr<gpu::CudaCurveSource> source;
    std::unique_ptr<gpu::CudaScatterGrow> scatterGrow;
    std::shared_ptr<CurveGrowInputOwners> growInputs;
    std::unique_ptr<gpu::CudaCurveGrow> curveGrow;
    std::unique_ptr<gpu::CudaNamedChannelTopologyCandidate> growNamedTopology;
    std::unique_ptr<gpu::CudaCurveResample> resampled;
    gpu::DeviceCurveGeometryView geometry;
    gpu::DeviceView<const uint64_t> captureStableIds;
    size_t captureCurveCount = 0;
    gpu::DeviceView<const float> hairT;
    gpu::DeviceView<const int32_t> rootPrim;
    gpu::DeviceView<const float2> rootUV;
    // Noise consumes an immutable per-curve rest frame. Source generation
    // deliberately keeps these internal, so the execution owner retains a
    // device COW copy for the lifetime of all point revisions.
    std::unique_ptr<gpu::DeviceBuffer<float3>> noiseTangent;
    std::unique_ptr<gpu::DeviceBuffer<float3>> noiseBinormal;
    std::unique_ptr<gpu::DeviceBuffer<float3>> noiseNormal;
    std::unique_ptr<gpu::DeviceBuffer<uint64_t>> frameStableIds;
    std::unique_ptr<gpu::DeviceBuffer<uint32_t>> frameCaptureOrdinals;
    std::shared_ptr<const ScatterGrowFrameHostPacket> scatterFrameHostPacket;
    // Preallocated before source-frame H2D.  If a stream loses proof, the
    // holder itself is intentionally retained with the quarantined upload.
    std::unique_ptr<std::shared_ptr<const ScatterGrowFrameHostPacket>>
        scatterFrameQuarantineOwner;
    std::unique_ptr<gpu::CudaCurveCompaction> compacted;
    std::unique_ptr<gpu::DeviceBuffer<float>> finalWidths;
    std::unique_ptr<gpu::DeviceBuffer<float3>> finalPoints;
    std::vector<gpu::CudaNamedChannelPlane> namedChannels;
    // Bundles are populated only after their native/named producer has been
    // proved complete. GeometryValue snapshots retain these shared owners;
    // mutable legacy fields remain for linear execution compatibility.
    std::vector<std::shared_ptr<TopologyOwners>> topologyOwners;
    std::shared_ptr<TopologyOwners> sourceTopologyOwners;
    std::vector<std::shared_ptr<NamedOwners>> namedOwners;
    // Immutable source views borrow the already-retained source/resampler.
    // Preserve original named owners only when publication selects Source;
    // descendants still consume their normal compacted values.
    gpu::DeviceCurveGeometryView sourceGeometry;
    gpu::DeviceView<const float> sourceHairT;
    gpu::DeviceView<const int32_t> sourceRootPrim;
    gpu::DeviceView<const float2> sourceRootUV;
    UsdGenDeviceCurveTopologyMetadata sourceCurveTopology;
    std::vector<gpu::CudaNamedChannelPlane> sourceTerminalNamedChannels;
    bool sourceNamedPreserved = false;
    bool sourceTerminalSelected = false;
    uint32_t selectedTopologyOwner = UINT32_MAX;
    bool sourceDeformed = false;
    std::vector<std::vector<uint32_t>> namedChannelHostStaging;
    std::unique_ptr<gpu::CudaNamedChannelTopologyCandidate> sourceNamedTopology;
    struct WidthValue {
        std::unique_ptr<gpu::DeviceBuffer<float>> owner;
        gpu::DeviceView<const float> view;
    };
    std::vector<WidthValue> widthValues;
    struct GeometryValue {
        std::unique_ptr<gpu::DeviceBuffer<float3>> points;
        gpu::DeviceCurveGeometryView geometry;
        gpu::DeviceView<const float> hairT;
        gpu::DeviceView<const int32_t> rootPrim;
        gpu::DeviceView<const float2> rootUV;
        UsdGenDeviceCurveTopologyMetadata curveTopology;
        bool deformed = false;
        gpu::DeviceView<const float3> pointOrigin;
        gpu::DeviceView<const float> widthOrigin;
        uint32_t pointOwnerNode = UINT32_MAX;
        uint32_t widthOwnerNode = UINT32_MAX;
        uint32_t topologyOwnerNode = UINT32_MAX;
        uint32_t namedOwnerNode = UINT32_MAX;
        std::shared_ptr<TopologyOwners> topologyOwners;
        std::shared_ptr<NamedOwners> namedOwners;
        // Original C3 frame-domain provenance.  Private source T/B/N is
        // stable-ID addressed; this token records both that owner and whether
        // the semantic frame plane is present without treating it as ordinal.
        std::shared_ptr<TopologyOwners> sourceFrameDomain;
        bool sourceFramesPresent = false;
        bool ready = false;
    };
    std::vector<GeometryValue> geometryValues;
    // Fixed at source admission, before any RBF work.  Each semantic Deform
    // owns an independent provisional cache transaction until whole-job
    // publication accepts (or rolls back) every entry together.
    std::vector<std::unique_ptr<CudaRbfPendingPublication>> pendingRbf;
    bool deformed = false;
    UsdGenDeviceCurveTopologyMetadata curveTopology;
    bool failed = false;
    bool prepared = false;
    bool sourceUploadPending = false;
    bool sourceValueRecorded = false;
    bool namedChannelsUnproven = false;
    CudaSourcePreparationOptions sourceOptions;
    bool finalized = false;
    bool NoiseFrames(gpu::DeviceBuffer<float3>** t,
                     gpu::DeviceBuffer<float3>** b,
                     gpu::DeviceBuffer<float3>** n,
                     gpu::DeviceView<const uint64_t>* ids = nullptr) const {
        auto* tangent = noiseTangent.get();
        auto* binormal = noiseBinormal.get();
        auto* normal = noiseNormal.get();
        if ((!tangent || !binormal || !normal) && growInputs) {
            tangent = growInputs->t.get();
            binormal = growInputs->b.get();
            normal = growInputs->n.get();
        }
        if (!tangent || !binormal || !normal) {
            auto const& owners = sourceTopologyOwners;
            if (owners && owners->t && owners->b && owners->n) {
                tangent = owners->t.get(); binormal = owners->b.get(); normal = owners->n.get();
            }
        }
        auto frameIds = frameStableIds
            ? gpu::DeviceView<const uint64_t>{frameStableIds->data(), frameStableIds->size()}
            : captureStableIds;
        if (!frameStableIds && sourceTopologyOwners && sourceTopologyOwners->frameStableIds)
            frameIds = {sourceTopologyOwners->frameStableIds->data(),
                        sourceTopologyOwners->frameStableIds->size()};
        if (!tangent || !binormal || !normal ||
            (frameIds.size && !frameIds.data) ||
            tangent->size() != frameIds.size ||
            binormal->size() != frameIds.size ||
            normal->size() != frameIds.size)
            return false;
        *t = tangent; *b = binormal; *n = normal;
        if (ids) *ids = frameIds;
        return true;
    }
    gpu::RestRootFrames CompactionFrames() const noexcept {
        if (compacted) return compacted->frames();
        if (curveGrow) return {curveGrow->rootT(), curveGrow->rootB(),
                                curveGrow->rootN(), geometry.stableIds};
        for (auto const& owners : topologyOwners) if (owners && owners->grow)
            return {owners->grow->rootT(), owners->grow->rootB(),
                    owners->grow->rootN(), geometry.stableIds};
        for (auto const& owners : topologyOwners) if (owners && owners->scatterGrow)
            return {owners->scatterGrow->rootT(), owners->scatterGrow->rootB(),
                    owners->scatterGrow->rootN(), owners->scatterGrow->view().stableIds};
        return {};
    }
    bool Prepare(UsdGenCudaExecutionPlan const&, UsdGenCudaExecutionWorkspace&,
                 double, UsdGenDiagnostics*, bool finishSource = true,
                 CudaSourcePreparationOptions const* provenOptions = nullptr,
                 UsdGenExecutionMemoryReservation* reservation = nullptr);
    bool PrepareScatterGrowWorkspace(UsdGenCudaExecutionPlan const&,
                                     UsdGenCudaExecutionWorkspace&,
                                     UsdGenDiagnostics*,
                                     UsdGenExecutionMemoryReservation*);
    bool PrepareScatterGrowFrames(UsdGenCudaExecutionPlan const&,
                                  UsdGenCudaExecutionWorkspace&, cudaStream_t,
                                  UsdGenDiagnostics*);
    bool UploadAuthoredPlanes(UsdGenCurveSetDesc const&, CudaSourcePrepared const&,
                              CudaSourcePreparationOptions const&,
                              cudaStream_t,
                              UsdGenDiagnostics*);
    bool PrepareNoiseFrames(CudaSourcePrepared const&, UsdGenCudaExecutionWorkspace&,
                            cudaStream_t,
                            UsdGenDiagnostics*);
    bool CommitSourceUpload(UsdGenCudaExecutionWorkspace&, UsdGenDiagnostics*);
    bool StartSourceResampleAsync(UsdGenCudaExecutionWorkspace&,
        void (*)(cudaStream_t, cudaError_t, void*) noexcept, void*,
        void (*)(cudaStream_t, cudaError_t, void*) noexcept, void*, bool, bool*);
    bool CompleteSourceResample(UsdGenCudaExecutionWorkspace&, UsdGenDiagnostics*,
                                std::shared_ptr<const void> lifetime = {});
    bool StartSourceNamedTopologyAsync(UsdGenCudaExecutionWorkspace&,
        void (*)(cudaStream_t, cudaError_t, void*) noexcept, void*, bool, bool*);
    bool CompleteSourceNamedTopology(cudaError_t, UsdGenCudaExecutionWorkspace&,
                                    UsdGenDiagnostics*);
    bool HasUnprovenAsyncWork() const noexcept;
    // Direct execution is synchronous at its public boundary. If that proof
    // fails, no sibling relay can publish a bundle, so conservatively retain
    // every selected/retained immutable owner before stack unwinding.
    void QuarantineGeometryBundles() noexcept {
        for (auto& value : geometryValues)
            if (value.points) value.points->quarantine();
        for (auto& value : widthValues)
            if (value.owner) value.owner->quarantine();
        for (auto const& owners : topologyOwners) {
            if (!owners) continue;
            if (owners->source) owners->source->MarkUnprovenUpload();
            if (owners->resampled) owners->resampled->MarkUnprovenUpload();
            if (owners->scatterGrow) owners->scatterGrow->MarkUnprovenWork();
            if (owners->grow) owners->grow->MarkUnprovenWork();
            if (owners->compaction) owners->compaction->MarkUnprovenWork();
            if (owners->t) owners->t->quarantine();
            if (owners->b) owners->b->quarantine();
            if (owners->n) owners->n->quarantine();
            if (owners->frameStableIds) owners->frameStableIds->quarantine();
            if (owners->frameCaptureOrdinals) owners->frameCaptureOrdinals->quarantine();
        }
        for (auto const& owners : namedOwners) if (owners)
            for (auto const& plane : owners->planes) if (plane.bytes) plane.bytes->quarantine();
        namedChannelsUnproven = true;
    }
    bool BeginCurveGrow(gpu::CurveGrowControls const&,
                        cudaStream_t, UsdGenDiagnostics*);
    bool BeginGrowNamed(cudaStream_t, std::shared_ptr<const void>,
                        std::shared_ptr<NamedOwners>, UsdGenDiagnostics*);
    bool AcceptCurveGrow(UsdGenCudaExecutionPlan const&, uint32_t, cudaError_t,
                         std::shared_ptr<NamedOwners>, UsdGenDiagnostics*);
    bool BeginCurveGrowValue(gpu::CurveGrowControls const&,
                             gpu::DeviceCurveGeometryView,
                             gpu::DeviceView<const int32_t>,
                             gpu::DeviceView<const float2>,
                             std::shared_ptr<TopologyOwners>,
                             std::shared_ptr<TopologyOwners>,
                             gpu::DeviceView<const uint64_t>,
                             std::shared_ptr<NamedOwners>,
                             std::shared_ptr<const void>,
                             OperatorGrowCandidate*, cudaStream_t,
                             UsdGenDiagnostics*);
    bool BeginGrowNamedValue(gpu::DeviceCurveGeometryView,
                             OperatorGrowCandidate*, cudaStream_t,
                             std::shared_ptr<const void>,
                             std::shared_ptr<NamedOwners>, UsdGenDiagnostics*);
    bool AcceptCurveGrowValue(UsdGenCudaExecutionPlan const&, uint32_t,
                              uint32_t, OperatorGrowCandidate*,
                              std::shared_ptr<NamedOwners>, UsdGenDiagnostics*);
    bool ResolvePendingRbf(bool accept) noexcept;
    bool HasPendingRbf() const noexcept {
        return std::any_of(pendingRbf.begin(), pendingRbf.end(),
            [](auto const& entry) { return bool(entry); });
    }
    void AbandonPendingRbf() noexcept {
        for (auto& entry : pendingRbf) (void)entry.release();
    }
    void RecordSourceValue(UsdGenCudaExecutionPlan const&);
    void PreserveSourceNamedChannels(UsdGenCudaExecutionPlan const&);
    // Width DAG dispatch takes a snapshot instead of rebinding `geometry`.
    // Sibling relays can therefore only read shared source geometry and write
    // their own width allocation.
    bool GetOperatorInput(UsdGenCudaExecutionPlan const&, size_t, cudaStream_t,
                          gpu::DeviceCurveGeometryView*,
                          gpu::DeviceView<const float>*, UsdGenDiagnostics*);
    bool SelectOperatorInput(UsdGenCudaExecutionPlan const&, size_t,
                             UsdGenDiagnostics*);
    bool SelectTerminal(UsdGenCudaExecutionPlan const&, cudaStream_t,
                        UsdGenDiagnostics*);
    std::shared_ptr<const UsdGenDeviceGeneration> Finalize(
        UsdGenCudaExecutionPlan const&, UsdGenCudaExecutionWorkspace&, uint64_t,
        std::shared_ptr<const UsdGenDeviceGeneration> const&, UsdGenDiagnostics*);
    std::shared_ptr<const UsdGenDeviceGeneration> PublishFinal(
        UsdGenCudaExecutionPlan const&, UsdGenCudaExecutionWorkspace&, uint64_t,
        std::shared_ptr<const UsdGenDeviceGeneration> const&,
        std::vector<UsdGenDeviceTileMetadata>, bool, UsdGenDiagnostics*);
    bool RunOperator(UsdGenCudaExecutionPlan const& plan,
                     UsdGenCudaExecutionWorkspace& workspace, double frame,
                     size_t i, UsdGenDiagnostics* diagnostics);
};

static bool BuildFullNonWidthInput(ExecutionState::GeometryValue const&,
                                   gpu::CurveFullNonWidthInput*);

bool PrepareCudaJobSource(ExecutionState& state,
                          CudaSourcePrepared const& prepared,
                          CudaSourcePreparationOptions const& options,
                          cudaStream_t stream, UsdGenDiagnostics* diagnostics,
                          bool finish = true) {
    state.source = std::make_unique<gpu::CudaCurveSource>();
    auto sourceStatus = state.source->Set(prepared.Input(), stream, state.memoryReservation);
    if (sourceStatus == gpu::CurveSourceStatus::Ok && finish)
        sourceStatus = state.source->Finish(stream);
    if (sourceStatus != gpu::CurveSourceStatus::Ok)
        return Fail(diagnostics, "source upload failed (CurveSourceStatus " +
            CurveSourceStatusText(sourceStatus) +
            "); previous generation retained");
    if (!finish) return true;
    if (options.resampleTo) {
        state.resampled = std::make_unique<gpu::CudaCurveResample>();
        if (state.resampled->Apply(state.source->view(), state.source->hairT(),
                state.source->rootPrim(), state.source->rootUV(), options.resampleTo,
                stream, state.memoryReservation) !=
                gpu::CurveResampleStatus::Ok ||
            state.resampled->Finish(stream) != gpu::CurveResampleStatus::Ok)
            return Fail(diagnostics, "CUDA source resample failed; previous generation retained");
    }
    state.geometry = state.resampled ? state.resampled->view() : state.source->view();
    state.captureStableIds = state.geometry.stableIds;
    state.captureCurveCount = state.geometry.curveCount;
    state.hairT = state.resampled ? state.resampled->hairT() : state.source->hairT();
    state.rootPrim = state.resampled ? state.resampled->rootPrim() : state.source->rootPrim();
    state.rootUV = state.resampled ? state.resampled->rootUV() : state.source->rootUV();
    state.deformed = !options.useRest;
    return true;
}

// The fused Scatter/Grow source has no authored C3 CurveSource to prepare,
// but Width runtimes are still workspace-owned state.  Keep their lifecycle
// identical to the regular source path without entering its curve-set parser.
bool ExecutionState::PrepareScatterGrowWorkspace(UsdGenCudaExecutionPlan const& plan,
                                                  UsdGenCudaExecutionWorkspace& workspace,
                                                  UsdGenDiagnostics* diagnostics,
                                                  UsdGenExecutionMemoryReservation* reservation) {
    auto& execution = *workspace.impl_;
    if (!plan.scatterGrow || execution.poisoned.load(std::memory_order_acquire) ||
        !std::isfinite(plan.desc.timeCodesPerSecond))
        return Fail(diagnostics, "invalid Scatter->Grow CUDA workspace state");
    if (!execution.description.IsEmpty() && execution.description != plan.desc.description)
        return Fail(diagnostics, "CUDA workspaces cannot be shared between descriptions");
    execution.description = plan.desc.description;
    // Every accepted Deform owns its provisional cache transaction until
    // whole-job publication. Allocate the fixed slots before fused uploads.
    try { pendingRbf.resize(plan.steps.size()); }
    catch (...) { return Fail(diagnostics, "CUDA fused RBF transaction admission failed"); }
    std::set<SdfPath> live;
    for (auto const& step : plan.steps) {
        live.insert(step->path);
        auto& runtime = execution.steps[step->path];
        if (!runtime) runtime = std::make_unique<UsdGenCudaExecutionWorkspace::Impl::Step>();
        if (step->type != TfToken("UsdGenDeform")) runtime->rbf.reset();
        else if (!runtime->rbf) runtime->rbf = std::make_unique<CudaRbfCache>();
    }
    for (auto it = execution.steps.begin(); it != execution.steps.end(); )
        if (!live.count(it->first)) it = execution.steps.erase(it); else ++it;
    memoryReservation = reservation;
    return true;
}

// Operator-declared named bindings require an explicit CUDA launcher contract.
// The aggregate NamedChannels value currently transports authored source data
// through Source/Length/publication, but generic operator launchers do not bind
// InputPrimvars()/OutputPrimvars().  Reject every such operator here, including
// topology-preserving ones, so a future capability row cannot silently ignore
// a read or mutate inherited COW storage.  CurveSource and Length have dedicated
// asynchronous topology transactions for inherited authored planes.
bool ValidateCudaNamedPlaneBindings(UsdGenGraphDesc const& desc,
                                    UsdGenDiagnostics* diagnostics) {
    usdGenRegisterM1Operators();
    bool hasAuthoredPlanes = false;
    for (auto const& curves : desc.curveSets)
        hasAuthoredPlanes = hasAuthoredPlanes || !curves.authoredPlanes.empty();
    for (auto const& node : desc.nodes) {
        std::unique_ptr<UsdGenOp> op = UsdGenOpRegistry::Get().Create(node.type);
        if (!op) continue;
        if (!op->InputPrimvars().empty() || !op->OutputPrimvars().empty()) {
            if (op->TopologyEffect() == UsdGenTopoFx::None)
                return Fail(diagnostics,
                    "named primvar bindings require an explicit CUDA operator "
                    "COW contract at " + node.path.GetString());
            return Fail(diagnostics,
                "named topology planes require CUDA execution-graph transport at " +
                node.path.GetString());
        }
        if (op->TopologyEffect() == UsdGenTopoFx::None) continue;
        if (!hasAuthoredPlanes || node.type == TfToken("UsdGenLength") ||
            node.type == TfToken("UsdGenGrow")) continue;
        if (node.type == TfToken("UsdGenCurveSource")) continue;
        return Fail(diagnostics,
            "authored named planes require asynchronous CUDA topology transport at " +
            node.path.GetString());
    }
    return true;
}
#endif

class UsdGenCudaExecutionJob {
public:
    ~UsdGenCudaExecutionJob();
    std::shared_ptr<const UsdGenCudaExecutionPlan> plan;
    UsdGenCudaExecutionWorkspace* workspace = nullptr;
    double frame = 0.0;
    uint64_t generation = 0;
    UsdGenDiagnostics* diagnostics = nullptr;
    std::shared_ptr<const UsdGenDeviceGeneration> previous;
    bool sourceDone = false, finalDone = false, sourceAsyncInFlight = false,
         finalAsyncInFlight = false;
    std::atomic<bool> failed{false};
    std::atomic<size_t> operatorAsyncInFlight{0};
    std::atomic<bool> noiseAsyncInFlight{false};
    // Noise/Length/Grow share mutable frame/named-topology bookkeeping even
    // when their point/width values are immutable DAG outputs.
    std::atomic<bool> geometryAsyncInFlight{false};
    size_t nextOperator = 0;
    std::vector<uint8_t> operatorStarted;
    std::vector<uint8_t> operatorFinished;
#ifdef USDGEN_ENABLE_CUDA
    void CloseMemoryReservation(bool detach = false) noexcept {
        if (memoryReservation) memoryReservation->Release();
        // Compatibility synchronous job stages intentionally fall back to
        // exact per-allocation admission after closing the job balance.  Do
        // not leave ExecutionState with a pointer to that closed balance.
        state.memoryReservation = nullptr;
        // A synchronous compatibility boundary has no concurrent launchers.
        // Later async stages must see null (exact admission), not a closed
        // transferable balance whose Consume would reject every allocation.
        // Failure/retirement callers retain the object for in-flight users.
        if (detach) memoryReservation.reset();
    }
    // Declared before state: member destruction runs state first, returning
    // every recyclable child permit before the reservation releases Pending.
    std::shared_ptr<UsdGenExecutionMemoryReservation> memoryReservation;
    std::unique_ptr<gpu::CudaScatterGrow> scatterGrow;
    ExecutionState state;
#endif
};

#ifdef USDGEN_ENABLE_CUDA
bool ExecutionState::ResolvePendingRbf(bool accept) noexcept {
    // Preflight every field/surface pair before changing any cache.  A
    // multi-lineage DAG must never publish a prefix of its RBF candidates.
    for (auto const& entry : pendingRbf) {
        if (!entry) continue;
        auto const& transaction = *entry;
        if (!transaction.cache || !transaction.state || !transaction.solveCommitted ||
            !transaction.surfaceCommitted || (transaction.rebind &&
            (!transaction.freshState || transaction.freshState.get() != transaction.state))) return false;
        auto& resources = *transaction.state->resources;
        if (accept ? (!resources.field.CanAcceptFreshSolve() || !resources.surface.CanAcceptFreshUpdate())
                   : (!resources.field.CanRollbackFreshSolve() || !resources.surface.CanRollbackFreshUpdate()))
            return false;
    }
    for (auto& entry : pendingRbf) {
        if (!entry) continue;
        auto& transaction = *entry;
        auto& resources = *transaction.state->resources;
        auto const surface = accept ? resources.surface.AcceptFreshUpdate()
                                    : resources.surface.RollbackFreshUpdate();
        auto const field = accept ? resources.field.AcceptFreshSolve()
                                  : resources.field.RollbackFreshSolve();
        if (surface != gpu::SurfaceBindingStatus::Ok || field != gpu::RbfStatus::Ok)
            return false; // caller quarantines the entire collection
        transaction.surfaceCommitted = false;
        transaction.solveCommitted = false;
    }
    // Only after every native state machine transitioned can the host-visible
    // cache identities/counters move.  This keeps partial phase-two failure
    // from publishing a prefix of the DAG's logical cache state.
    for (auto& entry : pendingRbf) {
        if (!entry) continue;
        auto& transaction = *entry;
        if (accept) {
            if (transaction.rebind) {
                static std::atomic<uint64_t> nextIdentity{1};
                transaction.freshState->identity = nextIdentity.fetch_add(1, std::memory_order_relaxed);
                transaction.cache->retiredState = std::move(transaction.cache->state);
                transaction.cache->state = std::move(transaction.freshState);
                transaction.cache->key = std::move(transaction.key);
                if (transaction.resetCounters) transaction.cache->bindCount = transaction.cache->solveCount = 1;
                else { ++transaction.cache->bindCount; ++transaction.cache->solveCount; }
            } else ++transaction.cache->solveCount;
        }
        entry.reset();
    }
    return true;
}

UsdGenCudaExecutionJob::~UsdGenCudaExecutionJob() {
    if (!state.HasPendingRbf() || !workspace) return;
    CudaDeviceScope selected(workspace->impl_->device);
    if (selected.selected && state.ResolvePendingRbf(false)) return;
    // The public job contract requires the borrowed workspace to outlive the
    // job.  If its device can no longer be selected, keep the transaction's
    // CUDA owners alive and make any attempted workspace reuse fail closed.
    workspace->impl_->poisoned.store(true, std::memory_order_release);
    state.AbandonPendingRbf();
}
#endif
#ifndef USDGEN_ENABLE_CUDA
UsdGenCudaExecutionJob::~UsdGenCudaExecutionJob() = default;
#endif

UsdGenCudaExecutionWorkspace::UsdGenCudaExecutionWorkspace()
    : impl_(std::make_unique<Impl>()) {}
UsdGenCudaExecutionWorkspace::~UsdGenCudaExecutionWorkspace() {
#ifdef USDGEN_ENABLE_CUDA
    if (impl_->device < 0) return;
    // A source whose terminal callback was not proved may still own pageable
    // staging used by the stream.  Do not turn a later wrapper destruction
    // into an unsafe free merely because the stream happens to synchronize.
    if (impl_->poisoned.load(std::memory_order_acquire)) {
        impl_.release();
        return;
    }
    CudaDeviceScope selected(impl_->device);
    bool drained = selected.selected &&
        cudaStreamSynchronize(impl_->stream) == cudaSuccess;
    for (auto stream : impl_->widthStreams)
        drained = drained && cudaStreamSynchronize(stream) == cudaSuccess;
    if (!drained) {
        // No completion proof: preserve all allocations/handles for context
        // teardown, including parameter evaluators and shared solver inputs.
        impl_.release();
        return;
    }
    impl_->steps.clear();
    impl_->sourceParameters.reset();
    impl_->widthBlendGraph.reset();
    for (auto& stream : impl_->widthStreams) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
    cudaStreamDestroy(impl_->stream);
#endif
}
int UsdGenCudaExecutionWorkspace::DeviceIndex() const noexcept { return impl_->device; }
bool UsdGenCudaExecutionWorkspace::IsPoisoned() const noexcept {
#ifdef USDGEN_ENABLE_CUDA
    return impl_->poisoned.load(std::memory_order_acquire);
#else
    return false;
#endif
}
void UsdGenCudaExecutionWorkspace::MarkContextLost() noexcept {
#ifdef USDGEN_ENABLE_CUDA
    impl_->poisoned.store(true, std::memory_order_release);
#endif
}

std::unique_ptr<UsdGenCudaExecutionWorkspace> CreateCudaExecutionWorkspace(
    int device, UsdGenDiagnostics* diagnostics) {
#ifdef USDGEN_ENABLE_CUDA
    if (device < -1) {
        Fail(diagnostics, "cannot capture CUDA workspace device: invalid device " + std::to_string(device)); return {};
    }
    if (device == -1) {
        auto const status = cudaGetDevice(&device);
        if (status != cudaSuccess) {
            Fail(diagnostics, "cannot capture CUDA workspace device: cudaGetDevice returned " + CudaStatusText(status)); return {};
        }
    }
    CudaDeviceScope selected(device);
    if (!selected.selected) { Fail(diagnostics, "cannot select CUDA workspace device " + std::to_string(device) + ": " + selected.FailureDetail()); return {}; }
    auto workspace = std::unique_ptr<UsdGenCudaExecutionWorkspace>(new UsdGenCudaExecutionWorkspace);
    auto const streamStatus = cudaStreamCreateWithFlags(&workspace->impl_->stream, cudaStreamNonBlocking);
    if (streamStatus != cudaSuccess) {
        Fail(diagnostics, "cannot create CUDA workspace stream: " + CudaStatusText(streamStatus)); return {};
    }
    for (auto& stream : workspace->impl_->widthStreams) {
        auto const branchStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (branchStatus == cudaSuccess)
            continue;
        for (auto& created : workspace->impl_->widthStreams) {
            if (created) cudaStreamDestroy(created);
            created = nullptr;
        }
        cudaStreamDestroy(workspace->impl_->stream);
        workspace->impl_->stream = nullptr;
        Fail(diagnostics, "cannot create CUDA Width branch stream: " + CudaStatusText(branchStatus)); return {};
    }
    try {
        workspace->impl_->widthBlendGraph = std::make_unique<CudaWidthBlendGraphCache>(device);
    } catch (...) {
        for (auto& created : workspace->impl_->widthStreams) {
            if (created) cudaStreamDestroy(created);
            created = nullptr;
        }
        cudaStreamDestroy(workspace->impl_->stream);
        workspace->impl_->stream = nullptr;
        Fail(diagnostics, "cannot allocate CUDA WidthBlend graph cache"); return {};
    }
    workspace->impl_->device = device;
    return workspace;
#else
    (void)device;
    Fail(diagnostics, "backend is not built"); return {};
#endif
}

#ifdef USDGEN_ENABLE_CUDA

namespace {
bool SameRest(CudaSurfaceBindingKey const& a, CudaSurfaceBindingKey const& b) {
    // Runtime sample-count expressions control a rebind separately. Sharing
    // this cache must not depend on a frame's evaluated count or posed points.
    return a.path == b.path && a.restPoints == b.restPoints &&
        a.restNormals == b.restNormals && a.restNormalDomain == b.restNormalDomain &&
        a.faceVertexCounts == b.faceVertexCounts && a.faceVertexIndices == b.faceVertexIndices;
}

// Synchronous transaction seam: it fires only after the fresh pose is fully
// proved, before the deformer and terminal publication are constructed.
std::atomic<bool> s_failNextSynchronousFinalization{false};

size_t AcceptedRbfSampleCount(gpu::CudaRbfBinding const& field) {
    auto const fresh = field.freshSampleCount();
    return fresh ? fresh : field.sampleCount();
}
template<class T> bool Upload(gpu::DeviceBuffer<T>& buffer, std::vector<T> const& values,
                              cudaStream_t stream,
                              UsdGenExecutionMemoryReservation* reservation = nullptr) {
    return buffer.reset(values.size(), reservation,
                        UsdGenExecutionResourceKind::Scratch) == cudaSuccess &&
        (values.empty() || cudaMemcpyAsync(buffer.data(), values.data(), values.size() * sizeof(T),
            cudaMemcpyHostToDevice, stream) == cudaSuccess);
}

} // namespace
#endif

#ifdef USDGEN_ENABLE_CUDA
// Finalization accounting below intentionally follows the concrete buffer
// sites.  Keep the compact host/device layout assumptions visible to CUDA
// builds rather than hiding them in a rounded estimate.
static_assert(sizeof(gpu::CurveTileSpan) == 5 * sizeof(uint32_t));
static_assert(sizeof(gpu::CurveTileBoundsScratch) == 2 * sizeof(float3) + sizeof(float));
static_assert(sizeof(float3) == 3 * sizeof(float));
#endif

std::vector<UsdGenCudaBindingStats> GetCudaBindingStats(UsdGenCudaExecutionWorkspace const& workspace) {
    return *std::atomic_load(&workspace.impl_->stats);
}

UsdGenCudaWidthBlendGraphStats GetCudaWidthBlendGraphStats(
    UsdGenCudaExecutionWorkspace const& workspace) noexcept {
#ifdef USDGEN_ENABLE_CUDA
    return workspace.impl_ && workspace.impl_->widthBlendGraph
        ? workspace.impl_->widthBlendGraph->Stats() : UsdGenCudaWidthBlendGraphStats{};
#else
    (void)workspace;
    return {};
#endif
}

uintptr_t getCudaRestSurfaceBindingCacheIdentityForTesting(
    UsdGenCudaExecutionWorkspace const& workspace) noexcept {
#ifdef USDGEN_ENABLE_CUDA
    return workspace.impl_ && workspace.impl_->sourceBindingCache
        ? reinterpret_cast<uintptr_t>(workspace.impl_->sourceBindingCache.get())
        : 0;
#else
    (void)workspace;
    return 0;
#endif
}

void failNextCudaWidthBlendGraphCaptureForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    s_failNextCudaWidthBlendGraphCapture.store(true, std::memory_order_release);
#endif
}

UsdGenExecutionCapabilityMatrix const& GetCudaExecutionCapabilityMatrix() noexcept {
    // Bump the matrix version whenever a row or its executor contract changes;
    // compiled-plan/cache keys can then reject stale backend assumptions.
    static const UsdGenExecutionCapabilityMatrix matrix{
        "cuda", 35,
#ifdef USDGEN_ENABLE_CUDA
        true,
#else
        false,
#endif
        {
            {TfToken("UsdGenScatter"), 1, 0, 0,
                UsdGenCapabilityChangesTopology | UsdGenCapabilityTopologyBarrier |
                UsdGenCapabilityExclusiveWorkspace},
            {TfToken("UsdGenGrow"), 13, 0, 0,
                UsdGenCapabilityChangesTopology | UsdGenCapabilityTopologyBarrier |
                UsdGenCapabilityExclusiveWorkspace},
            {TfToken("UsdGenCurveSource"), 2, 0, 1,
                UsdGenCapabilityExpressions | UsdGenCapabilityChangesTopology |
                UsdGenCapabilityTopologyBarrier | UsdGenCapabilityExclusiveWorkspace},
            {TfToken("UsdGenWidth"), 6, 0, 1,
                UsdGenCapabilityExpressions |
                UsdGenCapabilityCopyOnWriteWrites},
            {TfToken("UsdGenLength"), 7, 0, 1,
                UsdGenCapabilityExpressions | UsdGenCapabilityChangesTopology |
                UsdGenCapabilityTopologyBarrier | UsdGenCapabilityExclusiveWorkspace},
            {TfToken("UsdGenNoise"), 7, 0, 1,
                UsdGenCapabilityExpressions | UsdGenCapabilityCopyOnWriteWrites},
            {TfToken("UsdGenDeform"), 1, 0, 1,
                UsdGenCapabilityExpressions | UsdGenCapabilityExclusiveWorkspace |
                UsdGenCapabilitySurfaceInput |
                UsdGenCapabilityTransactionalPublication},
            {TfToken("UsdGenReferenceSource"), 2, 0, 1,
                UsdGenCapabilityChangesTopology | UsdGenCapabilityTopologyBarrier |
                UsdGenCapabilityExclusiveWorkspace},
            {TfToken("UsdGenWidthBlend"), 6, 0, 0,
                UsdGenCapabilityCopyOnWriteWrites}
        }};
    return matrix;
}

std::shared_ptr<const UsdGenExecutionPlanMetadata> GetCudaExecutionPlanMetadata(
    UsdGenCudaExecutionPlan const& plan) noexcept {
    return plan.metadata;
}

namespace {
struct CudaGraphLayout {
    uint32_t source = 0;
    uint32_t terminal = 0;
    bool taskDag = false;
    bool widthDag = false;
    bool topologyDag = false;
    bool sameTopologyDag = false;
    bool geometryValueDag = false;
    bool scatterGrow = false;
    uint32_t scatterGrowOutput = UINT32_MAX;
    std::vector<uint32_t> operators;
    std::vector<uint32_t> input;
    std::vector<uint32_t> rightInput;
    std::vector<bool> requiresNonWidthProof;
};

// Grow is the only value in this CUDA slice that creates an immutable rest
// geometry.  Noise may therefore omit the authored Source-rest requirement
// only when its own predecessor ancestry has passed through Grow; an
// unrelated, later, or downstream Grow cannot establish that contract.
bool NoiseNeedsAuthoredSourceRest(UsdGenGraphDesc const& desc, uint32_t source,
                                  std::vector<uint32_t> const& order,
                                  std::vector<uint32_t> const& input) {
    if (source >= desc.nodes.size() || input.size() != desc.nodes.size()) return true;
    std::vector<bool> generatedRest(desc.nodes.size(), false);
    for (uint32_t node : order) {
        if (node >= desc.nodes.size() || node >= input.size() || input[node] >= desc.nodes.size())
            return true;
        bool const inherited = generatedRest[input[node]];
        auto const& type = desc.nodes[node].type;
        if (type == TfToken("UsdGenNoise") && !inherited) return true;
        generatedRest[node] = type == TfToken("UsdGenGrow") ? true : inherited;
    }
    return false;
}

// Planning is deliberately integer-only.  These estimates describe explicit
// usdGen payload buffers, not CUDA events, driver allocations, cuSolver's
// opaque workspace, or allocator bookkeeping.  Every product/sum is checked
// while compiling; saturating to UINT64_MAX would make a later admission
// decision look safe when it is not.
bool EstimateAdd(uint64_t* total, uint64_t value) {
    return UsdGenExecutionCheckedBytes::Add(total, value);
}
bool EstimateMultiply(uint64_t count, uint64_t bytes, uint64_t* result) {
    return UsdGenExecutionCheckedBytes::Multiply(count, bytes, result);
}
bool EstimateSum(std::initializer_list<uint64_t> values, uint64_t* result) {
    uint64_t total = 0;
    for (uint64_t value : values)
        if (!EstimateAdd(&total, value)) return false;
    *result = total;
    return true;
}
#ifdef USDGEN_ENABLE_CUDA
bool HasExpressionBinding(UsdGenNodeDesc const& node, char const* destination) {
    return std::any_of(node.expressionBindings.begin(), node.expressionBindings.end(),
        [&](UsdGenExpressionBinding const& binding) {
            return LocalName(binding.destination) == destination;
        });
}
bool SourceExpressionCardinalityKnown(UsdGenNodeDesc const& source) {
    return !HasExpressionBinding(source, "resampleTo");
}
bool EstimateParameterCandidateBytes(
    std::shared_ptr<const CudaParameterProgram> const& program,
    uint64_t points, uint64_t curves, bool roots, uint64_t* result) {
    if (!result) return false;
    if (!program) { *result = 0; return true; }
    CudaParameterGeometryMemoryShape shape;
    shape.pointCount = points;
    shape.curveCount = curves;
    // Every generated geometry owns these channels. Root UV remains optional
    // and is present only when the paired root binding survived validation.
    shape.hasRestPoints = true;
    shape.hasWidths = true;
    shape.hasHairT = true;
    shape.hasRootUV = roots;
    return program->EstimateFreshCandidateBytes(shape, result);
}
#endif
bool EstimateGeometryBytes(uint64_t points, uint64_t curves, bool roots,
                           uint64_t* result) {
    // points/rest float3 + widths/hairT float = 32 bytes per point;
    // offsets + stable ids = 12 bytes per curve plus one offset. Root
    // bindings add int32 + aligned float2 bytes = 12 bytes per curve.
    uint64_t total = 0, value = 0;
    return EstimateMultiply(points, 32, &value) && EstimateAdd(&total, value) &&
        EstimateMultiply(curves, roots ? 24 : 12, &value) &&
        EstimateAdd(&total, value) && EstimateAdd(&total, sizeof(uint32_t)) &&
        ((*result = total), true);
}
bool SourceMayProduceRootBindings(UsdGenNodeDesc const& node,
                                  UsdGenCurveSetDesc const& source) {
    // Root capture can repair absent/partial bindings from the resolved
    // surface. Authoring alone is therefore not an output-shape proof.
    // Reserve the full input cardinality even when rebind=never may drop
    // roots; only a completely unbound surface-free source omits this term.
    // ReferenceSource has no local surface and transports authored bindings.
    return !node.surfaces.empty() || !source.skinPrim.empty() ||
        !source.skinPrimUv.empty();
}
bool EstimateTileScratchBytes(uint64_t curves, int tileTarget, uint64_t* result) {
    uint64_t const chunk = 512;
    uint64_t const target = static_cast<uint64_t>(std::clamp(tileTarget, 32, 256));
    uint64_t numerator = 0;
    numerator = curves;
    if (curves && !EstimateAdd(&numerator, chunk - 1)) return false;
    uint64_t chunks = curves ? numerator / chunk : 0;
    numerator = chunks;
    if (chunks && !EstimateAdd(&numerator, target - 1)) return false;
    uint64_t chunksPerTile = chunks ? std::max<uint64_t>(1, numerator / target) : 0;
    numerator = chunks;
    if (chunks && !EstimateAdd(&numerator, chunksPerTile - 1)) return false;
    uint64_t rawTiles = chunks ? numerator / chunksPerTile : 0;
    uint64_t tiles = chunks ? std::min(chunks, std::max<uint64_t>(32,
        std::min<uint64_t>(rawTiles, 256))) : 0;
    // FinalizationCandidate allocation sites, in bytes/tile:
    // device: span 20 + two status scalars amortized below + min/max 24 +
    // bounds scratch 28 = 72; pinned: span 20 + min/max 24 = 44.
    // The fixed 32 covers both device/pinned status pairs (16) and the
    // optional previous-generation topology result (device+pinned, 16).
    constexpr uint64_t kDeviceBytesPerTile = 20 + 24 + 28;
    constexpr uint64_t kPinnedBytesPerTile = 20 + 24;
    constexpr uint64_t kFixedBytes = 32;
    uint64_t value = 0;
    return EstimateMultiply(tiles, kDeviceBytesPerTile + kPinnedBytesPerTile,
                            &value) && EstimateAdd(&value, kFixedBytes) &&
        ((*result = value), true);
}

#ifdef USDGEN_ENABLE_CUDA
enum class LiteralLengthAdmission { NotEligible, Failed, Admitted };
enum class LiteralRbfAdmission { NotEligible, Failed, Admitted };

// RBF's factor/cache sizes are fully determined by the prepared literal
// surface.  Unlike expression-driven RBF, this bound can be admitted before
// source upload: N is the surface binder's clamped sample count, M=N+4, and
// W is the selected-device legacy-LU workspace query result in doubles.
// The cold-cache sum includes both the old/public candidate side of every
// COW exchange; cache permits then stay with the accepted binding after the
// job reservation itself closes.
LiteralRbfAdmission ReserveLiteralRbfExecution(
    UsdGenCudaExecutionPlan const& plan, int device, UsdGenDiagnostics* diagnostics,
    UsdGenExecutionMemoryReservation* result) {
    if (!result || !plan.metadata ||
        !plan.metadata->MemoryEstimate().runtimeRefinementAvailable ||
        plan.metadata->Shape() != UsdGenExecutionPlanShape::LinearAuthoredChain ||
        plan.sourceNode >= plan.desc.nodes.size() || plan.steps.size() != 1)
        return LiteralRbfAdmission::NotEligible;
    auto const& sourceNode = plan.desc.nodes[plan.sourceNode];
    auto const& step = *plan.steps.front();
    if (step.type != TfToken("UsdGenDeform") ||
        !SourceExpressionCardinalityKnown(sourceNode) ||
        step.semanticNode >= plan.desc.nodes.size() ||
        HasExpressionBinding(plan.desc.nodes[step.semanticNode], "rbfSamples") ||
        sourceNode.curves.empty())
        return LiteralRbfAdmission::NotEligible;
    auto source = std::find_if(plan.desc.curveSets.begin(), plan.desc.curveSets.end(),
        [&](auto const& curves) { return curves.path == sourceNode.curves.front(); });
    if (source == plan.desc.curveSets.end()) return LiteralRbfAdmission::NotEligible;
    UsdGenParamView sourceParams{&plan.desc, &sourceNode};
    // The formula below is for the literal cold Source->Deform producer; a
    // resampled input has an additional retained geometry owner.
    if (sourceParams.GetInt(TfToken("resampleTo"), 0) != 0) return LiteralRbfAdmission::NotEligible;
    UsdGenParamView deformParams{&plan.desc, &plan.desc.nodes[step.semanticNode]};
    int const budget = deformParams.GetInt(TfToken("rbfSamples"), 100);
    if (budget < 4 || budget > 46336 || step.surface.restPoints.empty() ||
        step.surface.currentPoints.size() != step.surface.restPoints.size() ||
        step.surface.faceOffsets.empty())
        return LiteralRbfAdmission::NotEligible;
    uint64_t const curves = source->curveVertexCounts.size();
    uint64_t const points = source->points.size();
    if (source->skinPrim.size() != curves || source->skinPrimUv.size() != curves)
        return LiteralRbfAdmission::NotEligible;
    uint64_t const vertices = step.surface.restPoints.size();
    uint64_t const faceOffsets = step.surface.faceOffsets.size();
    uint64_t const faceIndices = step.surface.faceVertexIndices.size();
    uint64_t const samples = std::min<uint64_t>(vertices, static_cast<uint64_t>(budget));
    uint64_t order = samples, matrixElements = 0, solverWorkElements = 0;
    uint64_t sourceBytes = 0, publicationScratch = 0, rbfBytes = 0, peak = 0;
    uint64_t sourceParameterBytes = 0, operatorParameterBytes = 0;
    if (!EstimateAdd(&order, 4) ||
        !EstimateMultiply(order, order, &matrixElements) ||
        !EstimateGeometryBytes(points, curves,
            source->skinPrim.size() == curves && source->skinPrimUv.size() == curves &&
                !sourceNode.surfaces.empty(), &sourceBytes) ||
        !EstimateTileScratchBytes(curves, plan.desc.tileTarget, &publicationScratch) ||
        !EstimateParameterCandidateBytes(plan.sourceParameters, points, curves,
            false, &sourceParameterBytes) ||
        !EstimateParameterCandidateBytes(step.parameters, points, curves,
            source->skinPrim.size() == curves &&
                source->skinPrimUv.size() == curves && !sourceNode.surfaces.empty(),
            &operatorParameterBytes)) {
        Fail(diagnostics, "CUDA literal-RBF memory estimate overflows");
        return LiteralRbfAdmission::Failed;
    }
    CudaDeviceScope selected(device);
    if (!selected.selected) {
        Fail(diagnostics, "cannot select CUDA workspace device for literal-RBF memory reservation");
        return LiteralRbfAdmission::Failed;
    }
    size_t solverWork = 0;
    if (gpu::GetCudaRbfLuWorkspaceElements(
            static_cast<size_t>(samples), &solverWork) != cudaSuccess) {
        Fail(diagnostics, "cannot query CUDA cuSOLVER workspace for literal-RBF memory reservation");
        return LiteralRbfAdmission::Failed;
    }
    solverWorkElements = static_cast<uint64_t>(solverWork);
    // Cold RBF COW peak (bytes): 32V + 8(Fo+Fi) + 36P + 64N + 12C +
    // 8M^2 + 8W + 52M + 1696. It includes the source-side upload buffers,
    // both surface current-sample generations, accepted+pending factors and
    // all three fresh pinned packets. The bind freshActual packet is released
    // before the update/solve/evaluate/deformer peak, so 1696 deliberately
    // does not double-count that mutually exclusive proof handle. Old accepted cache/output has its own
    // existing permits and remains charged through accept/rollback.
    uint64_t value = 0, surfaceIndexCounts = faceOffsets, surfaceIndices = 0, pointTerm = 0, sampleTerm = 0,
        curveTerm = 0, matrixTerm = 0, workTerm = 0, orderTerm = 0;
    if (!EstimateMultiply(vertices, 32, &value) ||
        !EstimateAdd(&rbfBytes, value) ||
        !EstimateAdd(&surfaceIndexCounts, faceIndices) ||
        !EstimateMultiply(surfaceIndexCounts, 8, &surfaceIndices) ||
        !EstimateAdd(&rbfBytes, surfaceIndices) ||
        !EstimateMultiply(points, 36, &pointTerm) || !EstimateAdd(&rbfBytes, pointTerm) ||
        !EstimateMultiply(samples, 64, &sampleTerm) || !EstimateAdd(&rbfBytes, sampleTerm) ||
        !EstimateMultiply(curves, 12, &curveTerm) || !EstimateAdd(&rbfBytes, curveTerm) ||
        !EstimateMultiply(matrixElements, 8, &matrixTerm) || !EstimateAdd(&rbfBytes, matrixTerm) ||
        !EstimateMultiply(solverWorkElements, 8, &workTerm) || !EstimateAdd(&rbfBytes, workTerm) ||
        !EstimateMultiply(order, 52, &orderTerm) || !EstimateAdd(&rbfBytes, orderTerm) ||
        !EstimateAdd(&rbfBytes, 1696) ||
        !EstimateSum({sourceBytes, rbfBytes, publicationScratch,
                      sourceParameterBytes, operatorParameterBytes,
                      plan.sourceParameters ? CudaGroomScalarReadback::Capacity * 8 : 0},
                     &peak) ||
        peak > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        Fail(diagnostics, "CUDA literal-RBF runtime memory reservation size overflows this platform");
        return LiteralRbfAdmission::Failed;
    }
    auto reservation = gpu::TryReserveCudaExecutionMemory(static_cast<size_t>(peak));
    if (!reservation) {
        Fail(diagnostics, "CUDA literal-RBF memory reservation of " + std::to_string(peak) +
            " bytes was not admitted by the device budget");
        return LiteralRbfAdmission::Failed;
    }
    *result = std::move(*reservation);
    return LiteralRbfAdmission::Admitted;
}

// Length survivor counts are device results, so the static plan deliberately
// leaves them unavailable. For a linear chain or a single Length trunk with
// a Width/WidthBlend DAG we can nevertheless admit a truthful worst case:
// every input curve/point survives. CUB's exact
// selected-device scan requirement is queried before Prepare or submission.
LiteralLengthAdmission ReserveLiteralLengthExecution(
    UsdGenCudaExecutionPlan const& plan, int device, cudaStream_t stream,
    UsdGenDiagnostics* diagnostics, UsdGenExecutionMemoryReservation* result) {
    if (!result || plan.scatterGrow || !plan.metadata ||
        !plan.metadata->MemoryEstimate().runtimeRefinementAvailable ||
        (plan.metadata->Shape() != UsdGenExecutionPlanShape::LinearAuthoredChain &&
         !(plan.taskDag && !plan.widthDag &&
           (plan.metadata->Shape() == UsdGenExecutionPlanShape::SourceRootedUnaryDag ||
            plan.metadata->Shape() == UsdGenExecutionPlanShape::SourceRootedValueDag))))
        return LiteralLengthAdmission::NotEligible;
    bool hasLength = false, hasWidth = false, hasGrow = false, hasNoise = false;
    size_t lengthCount = 0, growCount = 0, widthCount = 0, noiseCount = 0;
    for (auto const& step : plan.steps) {
        if (step->semanticNode >= plan.desc.nodes.size() ||
            (!IsCudaWidthValueNode(step->type) &&
             step->type != TfToken("UsdGenLength") &&
             step->type != TfToken("UsdGenGrow") &&
             step->type != TfToken("UsdGenNoise")))
            return LiteralLengthAdmission::NotEligible;
        if (step->type == TfToken("UsdGenLength")) {
            hasLength = true;
            ++lengthCount;
        } else if (step->type == TfToken("UsdGenGrow")) {
            hasGrow = true;
            ++growCount;
        } else if (step->type == TfToken("UsdGenNoise")) {
            hasNoise = true;
            ++noiseCount;
        } else {
            hasWidth = true;
            ++widthCount;
        }
    }
    if (!hasLength || plan.sourceNode >= plan.desc.nodes.size())
        return LiteralLengthAdmission::NotEligible;
    auto const& sourceNode = plan.desc.nodes[plan.sourceNode];
    bool const referenceSource = sourceNode.type == TfToken("UsdGenReferenceSource");
    if ((!referenceSource && sourceNode.curves.empty()) ||
        (referenceSource && sourceNode.references.size() != 1) ||
        !SourceExpressionCardinalityKnown(sourceNode))
        return LiteralLengthAdmission::NotEligible;
    SdfPath const sourcePath = referenceSource
        ? sourceNode.references.front() : sourceNode.curves.front();
    auto source = std::find_if(plan.desc.curveSets.begin(), plan.desc.curveSets.end(),
        [&](auto const& curves) { return curves.path == sourcePath; });
    if (source == plan.desc.curveSets.end())
        return LiteralLengthAdmission::NotEligible;
    UsdGenParamView params{&plan.desc, &sourceNode};
    int const resample = params.GetInt(TfToken("resampleTo"), 0);
    if (resample < 0 || resample == 1)
        return LiteralLengthAdmission::NotEligible;
    uint64_t const curves = static_cast<uint64_t>(source->curveVertexCounts.size());
    uint64_t const sourcePoints = static_cast<uint64_t>(source->points.size());
    uint64_t inputPoints = sourcePoints;
    if (resample > 0 && !EstimateMultiply(curves, static_cast<uint64_t>(resample),
                                          &inputPoints)) {
        Fail(diagnostics, "CUDA literal-Length memory estimate overflows while resampling");
        return LiteralLengthAdmission::Failed;
    }
    if (hasGrow) {
        for (auto const& step : plan.steps) if (step->type == TfToken("UsdGenGrow")) {
            UsdGenParamView grow{&plan.desc, &plan.desc.nodes[step->semanticNode]};
            uint64_t grown = 0;
            if (!EstimateMultiply(curves, static_cast<uint64_t>(grow.GetInt(TfToken("segments"), 8)), &grown)) {
                Fail(diagnostics, "CUDA literal-Length memory estimate overflows while growing");
                return LiteralLengthAdmission::Failed;
            }
            inputPoints = std::max(inputPoints, grown);
        }
    }
    uint64_t inputNamedBytes = 0, authoredNamedBytes = 0;
    for (UsdGenAuthoredPlaneDesc const& plane : source->authoredPlanes) {
        uint64_t elements = plane.domain == UsdGenAuthoredPlaneDomain::Point
            ? inputPoints : plane.domain == UsdGenAuthoredPlaneDomain::Primitive
            ? curves : 1;
        uint64_t values = 0, bytes = 0, authoredValues = 0, authoredBytes = 0;
        uint64_t const authoredElements = plane.domain == UsdGenAuthoredPlaneDomain::Point
            ? sourcePoints : plane.domain == UsdGenAuthoredPlaneDomain::Primitive
            ? curves : 1;
        if (!EstimateMultiply(elements, static_cast<uint64_t>(plane.arity), &values) ||
            !EstimateMultiply(values, sizeof(uint32_t), &bytes) ||
            !EstimateAdd(&inputNamedBytes, bytes) ||
            !EstimateMultiply(authoredElements, static_cast<uint64_t>(plane.arity), &authoredValues) ||
            !EstimateMultiply(authoredValues, sizeof(uint32_t), &authoredBytes) ||
            !EstimateAdd(&authoredNamedBytes, authoredBytes)) {
            Fail(diagnostics,
                "CUDA literal-Length named-channel memory estimate overflows");
            return LiteralLengthAdmission::Failed;
        }
    }
    // Use the possible captured output, not just authored root arrays: repair
    // may create these planes before Length/compaction consumes them.
    bool const roots = hasGrow || SourceMayProduceRootBindings(sourceNode, *source);
    uint64_t sourceBytes = 0, inputBytes = 0, scanBytes = 0, perCompaction = 0;
    uint64_t lengthScratch = 0, widthPeak = 0, publicationScratch = 0;
    if (!EstimateGeometryBytes(sourcePoints, curves, roots, &sourceBytes) ||
        !EstimateGeometryBytes(inputPoints, curves, roots, &inputBytes) ||
        !EstimateTileScratchBytes(curves, plan.desc.tileTarget, &publicationScratch)) {
        Fail(diagnostics, "CUDA literal-Length memory estimate overflows");
        return LiteralLengthAdmission::Failed;
    }
    uint64_t parameterBytes = 0, candidateBytes = 0;
    if (!EstimateParameterCandidateBytes(plan.sourceParameters, sourcePoints,
            curves, false, &candidateBytes) ||
        !EstimateAdd(&parameterBytes, candidateBytes)) {
        Fail(diagnostics, "CUDA expression evaluator memory estimate overflows");
        return LiteralLengthAdmission::Failed;
    }
    for (auto const& step : plan.steps) {
        if (!EstimateParameterCandidateBytes(step->parameters, inputPoints,
                curves, roots, &candidateBytes) ||
            !EstimateAdd(&parameterBytes, candidateBytes)) {
            Fail(diagnostics, "CUDA expression evaluator memory estimate overflows");
            return LiteralLengthAdmission::Failed;
        }
    }
    CudaDeviceScope selected(device);
    if (!selected.selected) {
        Fail(diagnostics, "cannot select CUDA workspace device for literal-Length memory reservation");
        return LiteralLengthAdmission::Failed;
    }
    size_t scan = 0;
    if (gpu::GetCudaCurveCompactionScanTemporaryBytes(
            static_cast<size_t>(curves), &scan, stream) != cudaSuccess) {
        Fail(diagnostics, "cannot query CUDA CUB scan workspace for literal-Length memory reservation");
        return LiteralLengthAdmission::Failed;
    }
    scanBytes = static_cast<uint64_t>(scan);
    uint64_t pointBytes = 0, curveBytes = 0, twoCurveBytes = 0,
        twoPointBytes = 0, prefixBytes = 0;
    bool widthEstimateOk = true;
    if (hasWidth) {
        uint64_t oneWidthPeak = 0;
        widthEstimateOk =
            EstimateMultiply(inputPoints, sizeof(float), &oneWidthPeak) &&
            EstimateMultiply(oneWidthPeak, 3, &oneWidthPeak) &&
            EstimateAdd(&oneWidthPeak, 257 * sizeof(float) + 2 * sizeof(int));
        // DAG Width outputs are immutable sibling values retained through the
        // publication join. Charge each branch's private COW plane and a
        // conservative simultaneous scratch packet. WidthBlend is charged
        // the same upper bound although its uncaptured kernel needs only its
        // private output. Optional capture storage is separately cache-charged
        // and falls back to the ordinary launch if not admitted.
        if (widthEstimateOk)
            widthEstimateOk = EstimateMultiply(oneWidthPeak,
                plan.taskDag ? static_cast<uint64_t>(widthCount) : 1,
                &widthPeak);
        // Every retained WidthBlend candidate owns its device+pinned
        // non-width comparison status packet until the owner commits it.
        for (auto const& step : plan.steps) if (step->requiresNonWidthProof)
            widthEstimateOk = widthEstimateOk &&
                EstimateAdd(&widthPeak, 4 * sizeof(int32_t));
    }
    // A compactor retains its output plus these scan/count allocations after
    // Finish. During a later Length, both old and candidate compactors live.
    if (!EstimateMultiply(inputPoints, sizeof(float3), &pointBytes) ||
        !EstimateMultiply(curves, sizeof(uint32_t), &curveBytes) ||
        !EstimateMultiply(curves, 2, &twoCurveBytes) ||
        !EstimateMultiply(pointBytes, 2, &twoPointBytes) ||
        !EstimateMultiply(curveBytes, 4, &prefixBytes) ||
        !EstimateSum({sizeof(int), 2 * sizeof(uint32_t), prefixBytes,
                      scanBytes, sizeof(int) + 2 * sizeof(uint32_t)},
                     &perCompaction) ||
        // External changed/keep plus CudaLength staging/error and its async
        // pinned diagnostic scalar.  Direct execution uses a subset.
        !EstimateSum({twoPointBytes, twoCurveBytes,
                      2 * sizeof(int)}, &lengthScratch) ||
        !widthEstimateOk) {
        Fail(diagnostics, "CUDA literal-Length runtime memory estimate overflows");
        return LiteralLengthAdmission::Failed;
    }
    uint64_t valueGeometryCount = static_cast<uint64_t>(lengthCount);
    uint64_t namedOutputCount = valueGeometryCount;
    if (plan.geometryValueDag &&
        ((!EstimateAdd(&valueGeometryCount, static_cast<uint64_t>(growCount))) ||
         !EstimateAdd(&valueGeometryCount, resample > 0 ? 1 : 0) ||
         !EstimateAdd(&namedOutputCount, static_cast<uint64_t>(growCount)) ||
         !EstimateAdd(&namedOutputCount, 1))) {
        Fail(diagnostics, "CUDA literal-Length value-DAG memory count overflows");
        return LiteralLengthAdmission::Failed;
    }
    uint64_t geometryMultiplicity = plan.geometryValueDag
        ? std::max<uint64_t>(2, valueGeometryCount) : 2;
    uint64_t survivorGeometryBytes = 0, doubledCompaction = 0,
        doubledNamedBytes = 0, sourceNamedPeak = 0,
        growFrames = 0, growMapBytes = 0, peak = 0;
    uint64_t noisePeak = 0;
    // Source is captured once. Every accepted value-DAG Grow and Length owns
    // an immutable native geometry packet; a literal resampler remains live
    // too.  Legacy linear execution retains its historical two-candidate
    // bound instead.
    if (hasGrow) geometryMultiplicity = std::max<uint64_t>(geometryMultiplicity,
        3 + (resample > 0 ? 1 : 0));
    else if (!plan.geometryValueDag && resample > 0 && lengthCount > 1)
        ++geometryMultiplicity;
    if (hasGrow) {
        uint64_t framePackets = 0, frameBytesPerCurve = 0;
        if (!EstimateAdd(&framePackets, 1) || // private source Noise frames
            !EstimateAdd(&framePackets, static_cast<uint64_t>(growCount)) ||
            !EstimateAdd(&framePackets, static_cast<uint64_t>(lengthCount)) ||
            !EstimateMultiply(framePackets, 3 * sizeof(float3), &frameBytesPerCurve) ||
            !EstimateMultiply(curves, frameBytesPerCurve, &growFrames)) {
            Fail(diagnostics, "CUDA literal-Length Grow frame estimate overflows");
            return LiteralLengthAdmission::Failed;
        }
    }
    if (hasNoise) {
        uint64_t noisePoints = 0, noiseFrames = 0;
        // A repeated Noise retains the prior overlay while the next output
        // and native staging plane are live.
        uint64_t const retainedNoisePlanes = plan.geometryValueDag
            ? std::max<uint64_t>(3, static_cast<uint64_t>(noiseCount) + 1) : 3;
        uint64_t retainedNoiseBytes = 0;
        if (!EstimateMultiply(retainedNoisePlanes, sizeof(float3), &retainedNoiseBytes) ||
            !EstimateMultiply(inputPoints, retainedNoiseBytes, &noisePoints) ||
            !EstimateMultiply(curves, 3 * sizeof(float3), &noiseFrames) ||
            !EstimateSum({noisePoints, noiseFrames, 257 * sizeof(float),
                          2 * sizeof(int)}, &noisePeak)) {
            Fail(diagnostics, "CUDA literal-Length Noise estimate overflows");
            return LiteralLengthAdmission::Failed;
        }
    }
    uint64_t const retainedCompactors = plan.geometryValueDag
        ? std::max<uint64_t>(2, static_cast<uint64_t>(lengthCount)) : 2;
    uint64_t const retainedNamedOutputs = plan.geometryValueDag
        ? namedOutputCount : 2;
    uint64_t growStatusBytes = 0;
    if (!EstimateMultiply(inputBytes, geometryMultiplicity,
                          &survivorGeometryBytes) ||
        !EstimateMultiply(perCompaction, retainedCompactors, &doubledCompaction) ||
        !EstimateMultiply(inputNamedBytes, retainedNamedOutputs, &doubledNamedBytes) ||
        !EstimateMultiply(static_cast<uint64_t>(growCount), 2 * sizeof(int),
                          &growStatusBytes) ||
        !EstimateSum({authoredNamedBytes, inputNamedBytes}, &sourceNamedPeak) ||
        !EstimateSum({sourceBytes, survivorGeometryBytes, doubledCompaction,
                      // Source resampling retains authored + resampled planes;
                      // Length retains input + compacted planes. Downsampling
                      // makes the former larger, so reserve their maximum.
                      std::max(sourceNamedPeak, doubledNamedBytes), 2 * sizeof(int), lengthScratch,
                      growFrames, growMapBytes, growStatusBytes,
                      noisePeak,
                      widthPeak, publicationScratch,
                      resample > 0 ? 2 * sizeof(int) : 0, parameterBytes,
                      plan.sourceParameters ? CudaGroomScalarReadback::Capacity * 8 : 0},
                     &peak) ||
        peak > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        Fail(diagnostics, "CUDA literal-Length runtime memory reservation size overflows this platform");
        return LiteralLengthAdmission::Failed;
    }
    auto reservation = gpu::TryReserveCudaExecutionMemory(static_cast<size_t>(peak));
    if (!reservation) {
        Fail(diagnostics, "CUDA literal-Length memory reservation of " +
            std::to_string(peak) + " bytes was not admitted by the device budget");
        return LiteralLengthAdmission::Failed;
    }
    *result = std::move(*reservation);
    return LiteralLengthAdmission::Admitted;
}
// A fused source has no authored C3 cardinality to pass to the literal C3
// estimator. Its captured roots and fixed Grow CV count nevertheless bound
// every Length descendant: compaction can remove, but never add, elements.
// Refine the known complete source/value payload subtotal with the actual
// selected-device CUB workspace before any upload or native job is accepted.
LiteralLengthAdmission ReserveScatterGrowExecution(
    UsdGenCudaExecutionPlan const& plan, int device, cudaStream_t stream,
    UsdGenDiagnostics* diagnostics, UsdGenExecutionMemoryReservation* result) {
    if (!result || !plan.scatterGrow || !plan.geometryValueDag || !plan.metadata ||
        !plan.scatterRoots) return LiteralLengthAdmission::NotEligible;
    // RBF's solver/cache shape is not represented by the fused Length CUB
    // subtotal. Preserve exact per-allocation native admission for these
    // jobs, as for generic C3 RBF DAGs, rather than precharging an incomplete
    // balance that could reject valid later solver allocations.
    if (std::any_of(plan.steps.begin(), plan.steps.end(), [](auto const& step) {
            return step->type == TfToken("UsdGenDeform"); }))
        return LiteralLengthAdmission::NotEligible;
    uint64_t lengthCount = 0;
    for (auto const& step : plan.steps)
        if (step->type == TfToken("UsdGenLength")) ++lengthCount;
    if (!lengthCount) return LiteralLengthAdmission::NotEligible;
    CudaDeviceScope selected(device);
    if (!selected.selected) {
        Fail(diagnostics, "cannot select CUDA device for ScatterGrow memory refinement");
        return LiteralLengthAdmission::Failed;
    }
    size_t scanBytes = 0;
    if (gpu::GetCudaCurveCompactionScanTemporaryBytes(
            plan.scatterRoots->stableIds.size(), &scanBytes, stream) != cudaSuccess) {
        Fail(diagnostics, "cannot query ScatterGrow descendant compaction workspace");
        return LiteralLengthAdmission::Failed;
    }
    uint64_t extra = 0;
    uint64_t peak = plan.metadata->MemoryEstimate().concurrentPeakBytes;
    // Each compactor remains an immutable topology owner until the terminal
    // join. Count every scan allocation, including independent branches.
    if (!EstimateMultiply(lengthCount, static_cast<uint64_t>(scanBytes), &extra) ||
        !EstimateAdd(&peak, extra) ||
        peak > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        Fail(diagnostics, "ScatterGrow descendant memory refinement overflow");
        return LiteralLengthAdmission::Failed;
    }
    auto reservation = gpu::TryReserveCudaExecutionMemory(static_cast<size_t>(peak));
    if (!reservation) {
        Fail(diagnostics, "ScatterGrow descendant memory reservation exceeds the device budget");
        return LiteralLengthAdmission::Failed;
    }
    *result = std::move(*reservation);
    return LiteralLengthAdmission::Admitted;
}
#endif
void SetTaskEstimate(UsdGenExecutionTaskEstimate* estimate, uint64_t steady,
                     uint64_t output, uint64_t retention, uint64_t scratch,
                     bool available) {
    if (!estimate) return;
    estimate->steadyBytes = steady;
    estimate->retainedOutputBytes = output;
    estimate->producerRetentionBytes = retention;
    estimate->scratchPeakBytes = scratch;
    estimate->memoryAvailable = available;
    estimate->memoryConservativeUpperBound = available;
    estimate->timeAvailable = false;
    estimate->estimatedMicroseconds = 0;
}

bool BuildCudaGraphLayout(UsdGenGraphDesc const& desc, CudaGraphLayout* result,
                          UsdGenDiagnostics* diagnostics) {
    if (!result || desc.nodes.empty())
        return Fail(diagnostics, "CUDA graph requires exactly one CurveSource");
    std::map<SdfPath, uint32_t> byPath;
    std::vector<uint32_t> sources;
    bool widthsOnly = true;
    uint32_t lengthNode = UINT32_MAX;
    uint32_t lengthCount = 0;
    uint32_t actualLengthCount = 0;
    uint32_t growCount = 0;
    bool widthOrLengthOnly = true;
    bool sameTopologyOnly = true;
    bool hasNoise = false;
    bool hasDeform = false;
    for (uint32_t i = 0; i != desc.nodes.size(); ++i) {
        auto const& node = desc.nodes[i];
        if (!byPath.emplace(node.path, i).second)
            return Fail(diagnostics, "invalid authoring: duplicate CUDA node path " +
                node.path.GetString());
        if (node.type == TfToken("UsdGenCurveSource") ||
            node.type == TfToken("UsdGenReferenceSource")) sources.push_back(i);
        else if (node.type != TfToken("UsdGenWidth") &&
                 node.type != TfToken("UsdGenWidthBlend")) {
            widthsOnly = false;
            if (node.type == TfToken("UsdGenLength") || node.type == TfToken("UsdGenGrow")) {
                lengthNode = i;
                ++lengthCount;
                if (node.type == TfToken("UsdGenLength")) ++actualLengthCount;
                else ++growCount;
            } else {
                widthOrLengthOnly = false;
            }
        }
        if (node.type != TfToken("UsdGenCurveSource") &&
            node.type != TfToken("UsdGenReferenceSource") &&
            node.type != TfToken("UsdGenWidth") &&
            node.type != TfToken("UsdGenWidthBlend") &&
            node.type != TfToken("UsdGenNoise")) sameTopologyOnly = false;
        if (node.type == TfToken("UsdGenNoise")) hasNoise = true;
        if (node.type == TfToken("UsdGenDeform")) hasDeform = true;
    }

    // Scatter is a captured topology source and Grow is its one fused native
    // producer.  Grow itself is deliberately not emitted as a fake no-op
    // operator. Its immutable value may feed Width, Length, and Noise
    // descendants; Deform remains intentionally excluded from this route.
    uint32_t scatter = UINT32_MAX, grow = UINT32_MAX;
    uint32_t scatterGrowCount = 0;
    for (uint32_t i = 0; i != desc.nodes.size(); ++i) {
        if (desc.nodes[i].type == TfToken("UsdGenScatter")) {
            if (scatter != UINT32_MAX) return Fail(diagnostics,
                "Scatter->Grow CUDA lowering requires exactly one Scatter");
            scatter = i;
        }
        if (desc.nodes[i].type == TfToken("UsdGenGrow")) {
            ++scatterGrowCount;
            grow = i;
        }
    }
    if (scatter != UINT32_MAX) {
        if (scatterGrowCount != 1 || grow == UINT32_MAX ||
            !desc.nodes[scatter].inputs.empty() ||
            desc.nodes[grow].inputs.size() != 1 ||
            desc.nodes[grow].inputs.front() != desc.nodes[scatter].path)
            return Fail(diagnostics,
                "Scatter->Grow CUDA lowering requires one direct Scatter input to Grow");
        result->source = scatter;
        result->scatterGrowOutput = grow;
        result->scatterGrow = true;
        result->taskDag = true;
        result->widthDag = true;
        result->input.assign(desc.nodes.size(), UINT32_MAX);
        result->rightInput.assign(desc.nodes.size(), UINT32_MAX);
        result->input[grow] = scatter;
        std::vector<std::vector<uint32_t>> children(desc.nodes.size());
        std::vector<uint32_t> indegree(desc.nodes.size(), 0);
        for (uint32_t i = 0; i != desc.nodes.size(); ++i) {
            if (i == scatter || i == grow) continue;
            auto const& node = desc.nodes[i];
            if (node.type != TfToken("UsdGenWidth") &&
                node.type != TfToken("UsdGenWidthBlend") &&
                node.type != TfToken("UsdGenLength") &&
                node.type != TfToken("UsdGenNoise") &&
                node.type != TfToken("UsdGenDeform"))
                return Fail(diagnostics,
                    "Scatter->Grow CUDA lowering supports only Width/Length/Noise/Deform descendants");
            size_t const expected = node.type == TfToken("UsdGenWidthBlend") ? 2u : 1u;
            if (node.inputs.size() != expected)
                return Fail(diagnostics, node.type == TfToken("UsdGenWidthBlend")
                    ? "Scatter->Grow WidthBlend requires exactly two ordered inputs"
                    : "Scatter->Grow Width requires exactly one input");
            auto left = byPath.find(node.inputs.front());
            if (left == byPath.end() || left->second == scatter)
                return Fail(diagnostics,
                    "Scatter->Grow descendant input must not bypass Grow");
            result->input[i] = left->second;
            children[left->second].push_back(i);
            if (expected == 2) {
                auto right = byPath.find(node.inputs[1]);
                if (right == byPath.end() || right->second == scatter ||
                    right->second == left->second)
                    return Fail(diagnostics,
                        "Scatter->Grow WidthBlend requires two distinct Width values");
                result->rightInput[i] = right->second;
                children[right->second].push_back(i);
            }
            indegree[i] = static_cast<uint32_t>(expected);
        }
        std::vector<uint32_t> ready{grow};
        std::vector<bool> emitted(desc.nodes.size(), false);
        emitted[scatter] = true;
        while (!ready.empty()) {
            auto next = *std::min_element(ready.begin(), ready.end());
            ready.erase(std::find(ready.begin(), ready.end(), next));
            if (emitted[next]) continue;
            emitted[next] = true;
            if (next != grow) result->operators.push_back(next);
            for (uint32_t child : children[next])
                if (--indegree[child] == 0) ready.push_back(child);
        }
        if (std::any_of(emitted.begin(), emitted.end(), [](bool value) { return !value; }))
            return Fail(diagnostics,
                "Scatter->Grow Width graph contains a cycle or is not Grow-rooted");
        bool hasGeometryDescendant = false;
        for (uint32_t nodeIndex : result->operators) {
            auto const& node = desc.nodes[nodeIndex];
            hasGeometryDescendant = hasGeometryDescendant ||
                node.type == TfToken("UsdGenLength") || node.type == TfToken("UsdGenNoise") ||
                node.type == TfToken("UsdGenDeform");
        }
        result->widthDag = !hasGeometryDescendant;
        result->geometryValueDag = hasGeometryDescendant;
        result->requiresNonWidthProof.assign(desc.nodes.size(), false);
        std::vector<uint32_t> pointOrigin(desc.nodes.size(), UINT32_MAX);
        std::vector<uint32_t> topologyOrigin(desc.nodes.size(), UINT32_MAX);
        std::vector<bool> deformed(desc.nodes.size(), false);
        pointOrigin[grow] = topologyOrigin[grow] = grow;
        for (uint32_t nodeIndex : result->operators) {
            auto const left = result->input[nodeIndex];
            if (left == UINT32_MAX || pointOrigin[left] == UINT32_MAX ||
                topologyOrigin[left] == UINT32_MAX)
                return Fail(diagnostics, "Scatter->Grow descendant origin is unavailable");
            auto const& node = desc.nodes[nodeIndex];
            if (node.type == TfToken("UsdGenDeform") && deformed[left])
                return Fail(diagnostics,
                    "unsupported composition: a second rest-to-animated deformation would apply surface motion twice");
            deformed[nodeIndex] = deformed[left] || node.type == TfToken("UsdGenDeform");
            if (node.type == TfToken("UsdGenLength")) {
                pointOrigin[nodeIndex] = topologyOrigin[nodeIndex] = nodeIndex;
            } else if (node.type == TfToken("UsdGenNoise") || node.type == TfToken("UsdGenDeform")) {
                pointOrigin[nodeIndex] = nodeIndex;
                topologyOrigin[nodeIndex] = topologyOrigin[left];
            } else if (node.type == TfToken("UsdGenWidth")) {
                pointOrigin[nodeIndex] = pointOrigin[left];
                topologyOrigin[nodeIndex] = topologyOrigin[left];
            } else {
                auto const right = result->rightInput[nodeIndex];
                if (right == UINT32_MAX || pointOrigin[right] == UINT32_MAX ||
                    topologyOrigin[right] == UINT32_MAX)
                    return Fail(diagnostics, "Scatter->Grow WidthBlend right origin is unavailable");
                result->requiresNonWidthProof[nodeIndex] = pointOrigin[left] != pointOrigin[right] ||
                    topologyOrigin[left] != topologyOrigin[right];
                pointOrigin[nodeIndex] = pointOrigin[left];
                topologyOrigin[nodeIndex] = topologyOrigin[left];
            }
        }
        if (!desc.terminal.IsEmpty()) {
            auto terminal = byPath.find(desc.terminal);
            if (terminal == byPath.end() || terminal->second == scatter)
                return Fail(diagnostics,
                    "Scatter->Grow terminal does not name Grow or a Width value");
            result->terminal = terminal->second;
        } else {
            std::vector<uint32_t> sinks;
            for (uint32_t i = 0; i != children.size(); ++i)
                if (i != scatter && children[i].empty()) sinks.push_back(i);
            if (sinks.size() != 1)
                return Fail(diagnostics,
                    "Scatter->Grow Width graph requires an explicit terminal");
            result->terminal = sinks.front();
        }
        return true;
    }
    if (sources.size() != 1)
        return Fail(diagnostics, "CUDA graph requires exactly one CurveSource");
    result->source = sources.front();
    result->input.assign(desc.nodes.size(), UINT32_MAX);
    result->rightInput.assign(desc.nodes.size(), UINT32_MAX);

    // Deform and multiple/topology-branching Length compositions retain the
    // established linear contract.  A single Length directly below Source is
    // handled by the immutable topology-trunk DAG path below.
    // A value DAG can have one topology-producing trunk, provided that trunk
    // is the sole non-value producer.  Noise deliberately does not disqualify
    // this form: it writes a point overlay while inheriting the trunk's
    // topology/named snapshot.
    bool topologyValueOnly = true;
    for (auto const& node : desc.nodes) {
        if (node.type == TfToken("UsdGenCurveSource") ||
            node.type == TfToken("UsdGenReferenceSource") ||
            node.type == TfToken("UsdGenGrow") ||
            node.type == TfToken("UsdGenLength") ||
            node.type == TfToken("UsdGenNoise") ||
            // Deform writes a point COW value.  Its predecessor's topology,
            // named planes, and source-frame domain remain immutable.
            node.type == TfToken("UsdGenDeform") ||
            node.type == TfToken("UsdGenWidth") ||
            node.type == TfToken("UsdGenWidthBlend")) continue;
        topologyValueOnly = false;
        break;
    }
    bool const directTopologyTrunk = lengthNode != UINT32_MAX &&
        desc.nodes[lengthNode].inputs.size() == 1 &&
        desc.nodes[lengthNode].inputs.front() == desc.nodes[result->source].path;
    bool const topologyTrunkCandidate = !widthsOnly && topologyValueOnly &&
        lengthCount == 1 && (widthOrLengthOnly || (hasNoise && directTopologyTrunk));
    // Actual Length nodes are immutable topology values when no Grow
    // participates. Each consumes its authored predecessor snapshot.
    bool const multiLengthValueCandidate = !widthsOnly && topologyValueOnly &&
        growCount == 0 && actualLengthCount >= 1;
    // Ordinary C3 Grow values retain CurveSource's original ID-addressed
    // frame domain while consuming their authored predecessor snapshot.
    bool const growValueCandidate = !widthsOnly && topologyValueOnly &&
        growCount != 0;
    // Keep the established source-index-zero, authored-linear RBF route out
    // of the value DAG: its literal-RBF reservation/cache contract depends
    // on the legacy LinearAuthoredChain shape.  Deform becomes a value-DAG
    // node only for a genuine branch, non-topological ordering, or selected
    // sibling terminal.
    bool legacyLinearDeform = hasDeform && result->source == 0 &&
        (desc.terminal.IsEmpty() || desc.terminal == desc.nodes.back().path);
    if (legacyLinearDeform) {
        for (uint32_t i = 1; i != desc.nodes.size(); ++i) {
            if (desc.nodes[i].inputs.size() != 1 ||
                desc.nodes[i].inputs.front() != desc.nodes[i - 1].path) {
                legacyLinearDeform = false;
                break;
            }
        }
    }
    // A Deform is a point-only COW value: no topology, named-channel, or
    // source-frame producer changes. Its per-lineage input is structurally
    // checked below, while sibling rest branches remain legal.
    bool const deformValueCandidate = !widthsOnly && topologyValueOnly && hasDeform &&
        !legacyLinearDeform;
    bool const sameTopologyCandidate = !widthsOnly && sameTopologyOnly;
    if (legacyLinearDeform ||
        (!widthsOnly && !topologyTrunkCandidate && !multiLengthValueCandidate &&
         !growValueCandidate && !deformValueCandidate && !sameTopologyCandidate)) {
        if (result->source != 0)
            return Fail(diagnostics,
                "unsupported composition: branched Length/Deform CUDA graphs are not supported");
        bool lineageDeformed = false;
        for (uint32_t i = 1; i != desc.nodes.size(); ++i) {
            auto const& node = desc.nodes[i];
            if (node.inputs.size() != 1 || node.inputs.front() != desc.nodes[i - 1].path)
                return Fail(diagnostics,
                    "unsupported composition: branched Length/Deform CUDA graphs are not supported");
            if (node.type == TfToken("UsdGenDeform")) {
                if (lineageDeformed)
                    return Fail(diagnostics,
                        "unsupported composition: a second rest-to-animated deformation would apply surface motion twice");
                lineageDeformed = true;
            }
            result->operators.push_back(i);
            result->input[i] = i - 1;
        }
        result->terminal = static_cast<uint32_t>(desc.nodes.size() - 1);
        if (!desc.terminal.IsEmpty() && desc.terminal != desc.nodes.back().path)
            return Fail(diagnostics, "terminal does not match the hierarchy result");
        return true;
    }

    result->taskDag = true;
    result->widthDag = widthsOnly;
    result->topologyDag = topologyTrunkCandidate || multiLengthValueCandidate ||
        growValueCandidate;
    result->sameTopologyDag = sameTopologyCandidate || deformValueCandidate;
    result->geometryValueDag = sameTopologyCandidate || topologyTrunkCandidate ||
        multiLengthValueCandidate || growValueCandidate || deformValueCandidate;
    if (!desc.nodes[result->source].inputs.empty())
        return Fail(diagnostics, "CUDA source root cannot consume an upstream geometry input");
    std::vector<std::vector<uint32_t>> children(desc.nodes.size());
    std::vector<uint32_t> indegree(desc.nodes.size(), 0);
    for (uint32_t i = 0; i != desc.nodes.size(); ++i) {
        if (i == result->source) continue;
        auto const& node = desc.nodes[i];
        size_t const expected = node.type == TfToken("UsdGenWidthBlend") ? 2u : 1u;
        if (node.inputs.size() != expected)
            return Fail(diagnostics, node.type == TfToken("UsdGenWidthBlend")
                ? "unsupported composition: CUDA WidthBlend requires exactly two ordered geometry inputs"
                : "unsupported composition: CUDA unary Width requires exactly one geometry input");
        auto left = byPath.find(node.inputs.front());
        if (left == byPath.end())
            return Fail(diagnostics,
                "invalid authoring: CUDA Width input does not name a graph node");
        result->input[i] = left->second;
        children[left->second].push_back(i);
        if (expected == 2) {
            // Duplicate inputs hide a unary operation behind a binary ABI and
            // would make ordered fan-in/lifetime tests meaningless.
            auto right = byPath.find(node.inputs[1]);
            if (right == byPath.end() || right->second == left->second)
                return Fail(diagnostics,
                    "invalid authoring: CUDA WidthBlend requires two distinct graph inputs");
            result->rightInput[i] = right->second;
            children[right->second].push_back(i);
        }
        indegree[i] = static_cast<uint32_t>(expected);
    }
    std::vector<uint32_t> ready{result->source};
    std::vector<bool> emitted(desc.nodes.size(), false);
    while (!ready.empty()) {
        auto next = *std::min_element(ready.begin(), ready.end());
        ready.erase(std::find(ready.begin(), ready.end(), next));
        if (emitted[next]) continue;
        emitted[next] = true;
        if (next != result->source) result->operators.push_back(next);
        for (auto child : children[next]) {
            if (--indegree[child] == 0) ready.push_back(child);
        }
    }
    if (std::any_of(emitted.begin(), emitted.end(), [](bool value) { return !value; }))
        return Fail(diagnostics,
            "invalid authoring: CUDA Width graph contains a cycle or is not source-rooted");

    if (topologyTrunkCandidate && !multiLengthValueCandidate && growCount == 0) {
        // The topology producer itself remains direct below Source. Other
        // immutable value branches may retain Source's independent snapshot;
        // WidthBlend origin proof below rejects any unsupported cross-topology
        // join before device submission.
        if (result->input[lengthNode] != result->source)
            return Fail(diagnostics,
                "unsupported composition: CUDA Length DAG must place its single Length directly below CurveSource");
    }
    if (result->geometryValueDag) {
        std::vector<uint32_t> pointOrigin(desc.nodes.size(), UINT32_MAX);
        std::vector<uint32_t> topologyOrigin(desc.nodes.size(), UINT32_MAX);
        // A Deform consumes rest-space source data and produces one animated
        // point revision.  Track that state per DAG lineage: sibling branches
        // may each deform their inherited rest snapshot, while a second
        // Deform below an already animated predecessor would apply surface
        // motion twice and must fail before any CUDA work is planned.
        std::vector<bool> deformed(desc.nodes.size(), false);
        result->requiresNonWidthProof.assign(desc.nodes.size(), false);
        pointOrigin[result->source] = result->source;
        topologyOrigin[result->source] = result->source;
        for (uint32_t nodeIndex : result->operators) {
            auto const& node = desc.nodes[nodeIndex];
            uint32_t const left = result->input[nodeIndex];
            if (left == UINT32_MAX || pointOrigin[left] == UINT32_MAX ||
                topologyOrigin[left] == UINT32_MAX)
                return Fail(diagnostics,
                    "CUDA geometry-value DAG predecessor origin is unavailable");
            if (node.type == TfToken("UsdGenDeform") && deformed[left])
                return Fail(diagnostics,
                    "unsupported composition: a second rest-to-animated deformation would apply surface motion twice");
            if (node.type == TfToken("UsdGenNoise")) {
                pointOrigin[nodeIndex] = nodeIndex;
                topologyOrigin[nodeIndex] = topologyOrigin[left];
                deformed[nodeIndex] = deformed[left];
            } else if (node.type == TfToken("UsdGenDeform")) {
                pointOrigin[nodeIndex] = nodeIndex;
                topologyOrigin[nodeIndex] = topologyOrigin[left];
                deformed[nodeIndex] = true;
            } else if (node.type == TfToken("UsdGenGrow") ||
                       node.type == TfToken("UsdGenLength")) {
                pointOrigin[nodeIndex] = nodeIndex;
                topologyOrigin[nodeIndex] = nodeIndex;
                deformed[nodeIndex] = deformed[left];
            } else if (node.type == TfToken("UsdGenWidth")) {
                pointOrigin[nodeIndex] = pointOrigin[left];
                topologyOrigin[nodeIndex] = topologyOrigin[left];
                deformed[nodeIndex] = deformed[left];
            } else if (node.type == TfToken("UsdGenWidthBlend")) {
                auto const right = result->rightInput[nodeIndex];
                if (right == UINT32_MAX)
                    return Fail(diagnostics,
                        "invalid authoring: CUDA WidthBlend right predecessor is unavailable");
                result->requiresNonWidthProof[nodeIndex] =
                    pointOrigin[right] != pointOrigin[left] ||
                    topologyOrigin[right] != topologyOrigin[left];
                pointOrigin[nodeIndex] = pointOrigin[left];
                topologyOrigin[nodeIndex] = topologyOrigin[left];
                deformed[nodeIndex] = deformed[left];
            }
        }
    }

    if (!desc.terminal.IsEmpty()) {
        auto terminal = byPath.find(desc.terminal);
        if (terminal == byPath.end())
            return Fail(diagnostics, "invalid authoring: CUDA terminal does not name a graph node");
        result->terminal = terminal->second;
    } else {
        std::vector<uint32_t> sinks;
        for (uint32_t i = 0; i != children.size(); ++i)
            if (children[i].empty()) sinks.push_back(i);
        if (sinks.size() != 1)
            return Fail(diagnostics,
                "unsupported composition: branched CUDA Width graphs require an explicit terminal");
        result->terminal = sinks.front();
    }
    return true;
}
} // namespace

bool ValidateCudaGraph(UsdGenGraphDesc const& desc, UsdGenDiagnostics* diagnostics) {
#ifndef USDGEN_ENABLE_CUDA
    (void)desc;
    return Fail(diagnostics, "backend is not built");
#else
    // This must precede capability and graph-layout validation.  In
    // particular, a future operator that advertises ReferenceInputs() must
    // not be reported as a generic unsupported operator after its unresolved
    // payload dependency was ignored.
    if (!ValidateCudaResolvedInputs(desc, diagnostics) ||
        !ValidateCudaNamedPlaneBindings(desc, diagnostics)) return false;
    for (auto const& node : desc.nodes) {
        if (node.type != TfToken("UsdGenCurveSource") &&
            !GetCudaExecutionCapabilityMatrix().Find(node.type))
            return Fail(diagnostics,
                "unsupported operator: CUDA capability matrix has no implementation for " +
                node.type.GetString());
    }
    CudaGraphLayout layout;
    if (!BuildCudaGraphLayout(desc, &layout, diagnostics)) return false;
    if (layout.scatterGrow) {
        if (!ValidateScatterGrow(desc, layout.source, layout.scatterGrowOutput, diagnostics))
            return false;
        for (uint32_t nodeIndex : layout.operators) {
            auto const& node = desc.nodes[nodeIndex];
            if (node.type == TfToken("UsdGenWidth")) {
                if (!ValidateWidth(node, diagnostics)) return false;
            } else if (node.type == TfToken("UsdGenWidthBlend")) {
                if (!ValidateWidthBlend(node, diagnostics)) return false;
            } else if (node.type == TfToken("UsdGenLength")) {
                if (!ValidateLength(node, diagnostics)) return false;
            } else if (node.type == TfToken("UsdGenNoise")) {
                if (!ValidateNoise(node, diagnostics)) return false;
            } else if (node.type == TfToken("UsdGenDeform")) {
                if (!ValidateDeform(node, diagnostics)) return false;
            } else {
                return Fail(diagnostics,
                    "Scatter->Grow CUDA lowering has an unsupported descendant operator");
            }
        }
        return true;
    }
    if (desc.nodes[layout.source].type == TfToken("UsdGenReferenceSource")) {
        // ReferenceSource has two deliberately narrow lowering shapes:
        // source -> Width DAG, or source -> exactly one direct Length -> one
        // or more Width branches. BuildCudaGraphLayout proves the latter's
        // directness, single-Length cardinality, and dominance before this
        // check; retain the explicit Width requirement here so a source-only
        // or source->Length graph cannot publish a reference generation.
        bool const hasWidth = std::any_of(layout.operators.begin(),
            layout.operators.end(), [&](uint32_t index) {
                return desc.nodes[index].type == TfToken("UsdGenWidth");
            });
        if ((!layout.widthDag && !layout.topologyDag) ||
            layout.operators.empty() || !hasWidth)
            return Fail(diagnostics,
                "UsdGenReferenceSource CUDA lowering requires one or more Width consumers");
    }
    if (!std::isfinite(desc.timeCodesPerSecond) || desc.timeCodesPerSecond <= 0)
        return Fail(diagnostics, "timeCodesPerSecond must be finite and positive");
    bool sawDeform = false, sawC3Grow = false;
    for (auto nodeIndex : layout.operators) {
        auto const& next = desc.nodes[nodeIndex];
        if (!GetCudaExecutionCapabilityMatrix().Find(next.type))
            return Fail(diagnostics, "unsupported operator: CUDA capability matrix has no implementation for " +
                next.type.GetString());
        if (next.type == TfToken("UsdGenDeform")) {
            sawDeform = true;
            if (!ValidateDeform(next, diagnostics)) return false;
            if (desc.nodes[layout.source].surfaces.size() != 1 ||
                desc.nodes[layout.source].surfaces.front() != next.surfaces.front())
                return Fail(diagnostics, "RBF target must match the CurveSource root-binding surface");
        } else if (next.type == TfToken("UsdGenGrow")) {
            if ((layout.taskDag && !layout.topologyDag) ||
                desc.nodes[layout.source].type != TfToken("UsdGenCurveSource") ||
                !ValidateScatterGrow(desc, layout.source, nodeIndex, diagnostics, true))
                return Fail(diagnostics, "CUDA C3 Grow requires a supported CurveSource value predecessor");
            sawC3Grow = true;
        } else if (next.type == TfToken("UsdGenLength")) {
            if (!ValidateLength(next, diagnostics)) return false;
        } else if (next.type == TfToken("UsdGenNoise")) {
            if (!ValidateNoise(next, diagnostics)) return false;
        } else if (next.type == TfToken("UsdGenWidth")) {
            if (!ValidateWidth(next, diagnostics)) return false;
        } else if (next.type == TfToken("UsdGenWidthBlend")) {
            if (!ValidateWidthBlend(next, diagnostics)) return false;
        } else {
            return Fail(diagnostics,
                "unsupported operator: CUDA Width DAG has an unknown value node");
        }
    }
    if (sawC3Grow && !layout.taskDag) {
        bool seenGrow = false;
        for (uint32_t index : layout.operators) {
            auto const& candidate = desc.nodes[index];
            if (candidate.type == TfToken("UsdGenGrow")) {
                if (seenGrow || candidate.inputs.size() != 1 ||
                    candidate.inputs.front() != desc.nodes[layout.source].path)
                    return Fail(diagnostics,
                        "CUDA linear Grow requires one direct CurveSource input");
                seenGrow = true;
            } else if (!seenGrow ||
                       (candidate.type != TfToken("UsdGenNoise") &&
                        candidate.type != TfToken("UsdGenLength") &&
                        candidate.type != TfToken("UsdGenWidth"))) {
                return Fail(diagnostics,
                    "CUDA linear Grow supports sequential Noise/Length/Width suffixes");
            }
        }
    }
    auto const& node = desc.nodes[layout.source];
    bool const referenceSource = node.type == TfToken("UsdGenReferenceSource");
    if (referenceSource) {
        if (node.references.size() != 1 || !node.curves.empty() ||
            !node.surfaces.empty() || !node.maps.empty() ||
            !node.mapBindings.empty() || !node.inputs.empty())
            return Fail(diagnostics,
                "UsdGenReferenceSource requires exactly one reference and no geometry/map inputs");
        auto const reference = std::find_if(desc.curveSets.begin(), desc.curveSets.end(),
            [&](auto const& curves) { return curves.path == node.references.front(); });
        if (reference == desc.curveSets.end() || reference->role != UsdGenRole::Reference)
            return Fail(diagnostics, "UsdGenReferenceSource requires a resolved reference curve set");
        if (!reference->authoredPlanes.empty())
            return Fail(diagnostics,
                "UsdGenReferenceSource named planes are not supported by CUDA");
        if (!reference->rootFrame.empty())
            return Fail(diagnostics,
                "UsdGenReferenceSource rootFrame bindings are not supported by CUDA");
        if (!node.params.empty() || !node.ramps.empty() ||
            !node.expressionBindings.empty())
            return Fail(diagnostics,
                "UsdGenReferenceSource accepts no parameters, ramps, or expressions");
        if (!std::isfinite(desc.defaultWidth) || desc.defaultWidth < 0)
            return Fail(diagnostics, "description default width must be finite and non-negative");
        return true;
    }
    if (!ValidateExpressionTargets(node, diagnostics)) return false;
    TfToken const interpolationMode = UsdGenParamView{&desc, &node}.GetToken(
        TfToken("interpolationMode"), TfToken("none"));
    if (interpolationMode != TfToken("none") && interpolationMode != TfToken("surfaceCage"))
        return Fail(diagnostics, "CurveSource interpolationMode must be none or surfaceCage");
    bool const surfaceCage = interpolationMode == TfToken("surfaceCage");
    if (!referenceSource && (!node.inputs.empty() || !node.references.empty() ||
        (!surfaceCage && (!node.maps.empty() || !node.mapBindings.empty()))))
        return Fail(diagnostics, "CurveSource cannot consume an upstream/reference/map in the current executor");
    if (surfaceCage && (node.maps.size() != 1 || node.mapBindings.size() != 1 ||
        node.maps.front() != node.mapBindings.front().map ||
        node.mapBindings.front().relationship != TfToken("usdGen:regionMap") ||
        node.surfaces.size() != 1))
        return Fail(diagnostics, "surfaceCage CurveSource requires one bound surface and one usdGen:regionMap");
    for (auto const& ramp : node.ramps) {
        if (!ramp.positions.empty() || !ramp.colors.empty() ||
            std::any_of(ramp.knots.begin(), ramp.knots.end(), [](auto const& k) { return k[1] != 1.0f || !std::isfinite(k[0]); }))
            return Fail(diagnostics, "non-identity source ramps are not supported");
    }
    if (node.curves.size() != 1)
        return Fail(diagnostics, "CurveSource requires exactly one C3 curve target");
    if (node.surfaces.size() > 1)
        return Fail(diagnostics, "CurveSource binding requires one resolved parent surface");
    if (!std::isfinite(desc.defaultWidth) || desc.defaultWidth < 0)
        return Fail(diagnostics, "description default width must be finite and non-negative");
    if (!node.mode.IsEmpty()) return Fail(diagnostics, "CurveSource has no mode property");
    UsdGenParamView params; params.desc = &desc; params.node = &node;
    const bool connectedUseRest = std::any_of(node.expressionBindings.begin(), node.expressionBindings.end(),
        [](UsdGenExpressionBinding const& binding) { return LocalName(binding.destination) == "useRest"; });
    if (sawDeform && !connectedUseRest && !params.GetBool(TfToken("useRest"), true))
        return Fail(diagnostics, "already-deformed CurveSource cannot feed rest-to-animated RBF Deform");
    if (sawDeform) {
        auto source = std::find_if(desc.curveSets.begin(), desc.curveSets.end(),
            [&](auto const& curves) { return curves.path == node.curves.front(); });
        if (desc.xformMatrix != GfMatrix4d(1.0) || source == desc.curveSets.end() ||
            source->worldMatrix != GfMatrix4d(1.0))
            return Fail(diagnostics, "RBF currently requires identity Description and CurveSource transforms");
    }
    // Never silently ignore an authored effect just because this is a source.
    static const std::set<std::string> supported{
        "useRest", "idSource", "expectEpoch", "staleAction",
        "resampleTo", "rebind", "interpolationMode", "densityMultiplier",
        "regionMapChannel", "expectMapGeneration"};
    std::set<TfToken> seen;
    for (auto const& param : node.params) {
        if (!seen.insert(param.name).second)
            return Fail(diagnostics, "duplicate source parameter " + param.name.GetString());
        auto const& name = param.name.GetString();
        if (!supported.count(name))
            return Fail(diagnostics, "unsupported CurveSource parameter " + param.name.GetString());
        const bool validType = name == "useRest" ? param.value.IsHolding<bool>() :
            name == "resampleTo" || name == "regionMapChannel" ? param.value.IsHolding<int>() :
            name == "densityMultiplier" ? param.value.IsHolding<float>() &&
                std::isfinite(param.value.UncheckedGet<float>()) && param.value.UncheckedGet<float>() > 0.0f :
            name == "expectMapGeneration" ? param.value.IsHolding<uint64_t>() :
            (name == "expectEpoch") ? param.value.IsHolding<std::string>() :
            param.value.IsHolding<TfToken>();
        if (!validType || (surfaceCage && param.animated))
            return Fail(diagnostics, "wrong native type for source parameter " + name);
    }
    if (surfaceCage) {
        UsdGenParamView const cageParams{&desc, &node};
        if (cageParams.GetInt(TfToken("regionMapChannel"), 0) != 0)
            return Fail(diagnostics, "surfaceCage CurveSource requires categorical regionMapChannel zero");
        auto const source = std::find_if(desc.curveSets.begin(), desc.curveSets.end(),
            [&](auto const& curves) { return curves.path == node.curves.front(); });
        auto const map = std::find_if(desc.maps.begin(), desc.maps.end(),
            [&](auto const& value) { return value.path == node.maps.front(); });
        if (source == desc.curveSets.end() || !source->surfaceCage ||
            map == desc.maps.end() || map->type != TfToken("UsdGenPtexMap") ||
            map->resolvedAssetPath.empty())
            return Fail(diagnostics, "surfaceCage CurveSource requires a captured cage and resolved UsdGenPtexMap");
    }
    if (NoiseNeedsAuthoredSourceRest(desc, layout.source, layout.operators, layout.input)) {
        auto const source = std::find_if(desc.curveSets.begin(), desc.curveSets.end(),
            [&](auto const& curves) { return curves.path == node.curves.front(); });
        if (source == desc.curveSets.end())
            return Fail(diagnostics, "CUDA Noise requires a resolved C3 source");
        // Noise always samples rest geometry, independently of a literal or
        // connected source useRest control. This invariant can be established
        // at admission without evaluating expressions or allocating CUDA data.
        if (!source->curveVertexCounts.empty()) {
            size_t total = 0;
            for (int count : source->curveVertexCounts) {
                if (count < 2 || size_t(count) > std::numeric_limits<size_t>::max() - total)
                    return Fail(diagnostics, "CUDA Noise source has invalid curve topology");
                total += size_t(count);
            }
            if (source->restFromCurrentPoints || source->rest.size() != total)
                return Fail(diagnostics, "CUDA Noise requires an authored/default-time C3 rest snapshot");
            for (auto const& point : source->rest)
                if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2]))
                    return Fail(diagnostics, "CUDA Noise source rest snapshot contains non-finite points");
            UsdGenSurfaceDesc const* surface = nullptr;
            if (!node.surfaces.empty()) {
                auto const found = std::find_if(desc.surfaces.begin(),desc.surfaces.end(),
                    [&](auto const& value) { return value.path == node.surfaces.front(); });
                if (found == desc.surfaces.end())
                    return Fail(diagnostics, "CUDA Noise root binding surface is unresolved");
                surface = &*found;
            }
            UsdGenCurveRootCaptureResult roots;
            std::string error;
            if (!UsdGenCaptureCurveRoots(*source, surface,
                    params.GetToken(TfToken("rebind"),TfToken("onError")), &roots,&error))
                return Fail(diagnostics, "CUDA Noise root admission failed: " + error);
            if (roots.frames.size() != source->curveVertexCounts.size())
                return Fail(diagnostics, "CUDA Noise cannot consume an unbound source without rest root frames");
            // Missing/out-of-range bindings can now be repaired by the shared
            // immutable capture. Degeneracy remains valid source compaction.
        }
    }
    return true;
#endif
}

std::shared_ptr<const UsdGenCudaExecutionPlan> CompileCudaGraph(
    UsdGenGraphDesc const& desc, UsdGenDiagnostics* diagnostics) {
    if (!ValidateCudaGraph(desc, diagnostics)) return {};
    CudaGraphLayout layout;
    if (!BuildCudaGraphLayout(desc, &layout, diagnostics)) return {};
    auto plan = std::make_shared<UsdGenCudaExecutionPlan>();
    plan->desc = desc;
#ifdef USDGEN_ENABLE_CUDA
    plan->sourceNode = layout.source;
    plan->terminalNode = layout.terminal;
    plan->taskDag = layout.taskDag;
    plan->widthDag = layout.widthDag;
    plan->sameTopologyDag = layout.sameTopologyDag;
    plan->geometryValueDag = layout.geometryValueDag;
    plan->scatterGrow = layout.scatterGrow;
    plan->scatterGrowOutputNode = layout.scatterGrowOutput;
    if (layout.scatterGrow) {
        std::string reason;
        if (PrepareCudaScatterInput(desc, desc.nodes[layout.source].path,
                &plan->scatterRoots, &reason) != CudaScatterInputStatus::Ok) {
            Fail(diagnostics, "Scatter capture preparation failed: " + reason);
            return {};
        }
        UsdGenParamView grow{&desc, &desc.nodes[layout.scatterGrowOutput]};
        plan->scatterGrowControls.cvCount = uint32_t(grow.GetInt(TfToken("segments"), 8));
        plan->scatterGrowControls.seed = desc.nodes[layout.scatterGrowOutput].seed;
        plan->scatterGrowControls.length = grow.GetDouble(TfToken("length"), 1.0);
        plan->scatterGrowControls.lift = float(grow.GetDouble(TfToken("lift"), 0.0));
        plan->scatterGrowControls.azimuth = float(grow.GetDouble(TfToken("azimuth"), 0.0));
        plan->scatterGrowControls.azimuthRandom = float(grow.GetDouble(TfToken("azimuthRandom"), 0.0));
        plan->scatterGrowControls.fallbackWidth = desc.defaultWidth;
        auto random = grow.GetVtValue(TfToken("lengthRandom"), VtValue(GfVec2f(1,1))).UncheckedGet<GfVec2f>();
        plan->scatterGrowControls.randomLo = random[0];
        plan->scatterGrowControls.randomHi = random[1];
        TfToken direction = grow.GetToken(TfToken("direction"), TfToken("surfaceNormal"));
        plan->scatterGrowControls.direction = direction == TfToken("vector")
            ? gpu::ScatterGrowDirection::Literal : direction == TfToken("attribute")
            ? gpu::ScatterGrowDirection::RootTangent : gpu::ScatterGrowDirection::RootNormal;
        if (direction == TfToken("vector")) {
            auto value = grow.GetVtValue(TfToken("directionVector"), VtValue(GfVec3f(0,1,0)));
            if (!value.IsHolding<GfVec3f>()) { Fail(diagnostics,"Scatter->Grow directionVector must be float3"); return {}; }
            auto const v=value.UncheckedGet<GfVec3f>();
            plan->scatterGrowControls.literalDirection=make_float3(v[0],v[1],v[2]);
        }
        if (gpu::GetScatterGrowRequirements(plan->scatterRoots->positions.size(),
                plan->scatterGrowControls.cvCount, &plan->scatterGrowRequirements)
                != gpu::ScatterGrowStatus::Ok) {
            Fail(diagnostics, "Scatter->Grow topology requirements are invalid"); return {};
        }
        // Compile descendant operators against the generated topology;
        // no Grow Step is emitted because the native source already did it.
        std::map<uint32_t, std::pair<uint32_t, uint32_t>> proofOrigins;
        proofOrigins[layout.scatterGrowOutput] = {layout.scatterGrowOutput, layout.scatterGrowOutput};
        for (uint32_t nodeIndex : layout.operators) {
            auto step = std::make_unique<UsdGenCudaExecutionPlan::Step>();
            step->path = desc.nodes[nodeIndex].path;
            step->type = desc.nodes[nodeIndex].type;
            step->semanticNode = nodeIndex;
            step->inputNode = layout.input[nodeIndex];
            step->rightInputNode = layout.rightInput[nodeIndex];
            auto const leftOrigin = proofOrigins.find(step->inputNode);
            if (leftOrigin == proofOrigins.end()) {
                Fail(diagnostics, "Scatter->Grow predecessor proof origin is unavailable"); return {};
            }
            auto origin = leftOrigin->second;
            if (step->type == TfToken("UsdGenLength")) origin = {nodeIndex, nodeIndex};
            else if (step->type == TfToken("UsdGenNoise") ||
                     step->type == TfToken("UsdGenDeform")) origin.first = nodeIndex;
            if (step->rightInputNode != UINT32_MAX) {
                auto const rightOrigin = proofOrigins.find(step->rightInputNode);
                if (rightOrigin == proofOrigins.end()) {
                    Fail(diagnostics, "Scatter->Grow right proof origin is unavailable"); return {};
                }
                step->requiresNonWidthProof = leftOrigin->second != rightOrigin->second;
            }
            proofOrigins[nodeIndex] = origin;
            std::vector<std::string> errors;
            if (CudaParameterProgram::Compile(desc, desc.nodes[nodeIndex], &step->parameters,
                                              &errors) != CudaParameterStatus::Ok) {
                for (auto const& error : errors) Fail(diagnostics, error);
                return {};
            }
            UsdGenParamView params{&desc, &desc.nodes[nodeIndex]};
            auto lut = [&](char const* knots, char const* interpolation, auto& output) {
                auto value = params.GetVtValue(TfToken(knots), VtValue(VtVec2fArray{}));
                UsdGenBuildRampLut(value.UncheckedGet<VtVec2fArray>(),
                    params.GetToken(TfToken(interpolation), TfToken("catmullRom")),
                    output.data(), output.size());
            };
            if (step->type == TfToken("UsdGenWidth")) {
                lut("width:knots", "width:interpolation", step->profile);
                UsdGenWidthBezierLut(params, step->profile.data(), step->profile.size());
            } else if (step->type == TfToken("UsdGenNoise")) {
                lut("noise:magnitude:knots", "noise:magnitude:interpolation", step->profile);
            }
            if (step->type == TfToken("UsdGenDeform")) {
                auto const& node = desc.nodes[nodeIndex];
                auto surface = std::find_if(desc.surfaces.begin(), desc.surfaces.end(),
                    [&](auto const& value) { return value.path == node.surfaces.front(); });
                if (surface == desc.surfaces.end()) {
                    Fail(diagnostics, "Scatter->Grow Deform surface is unavailable"); return {};
                }
                if (PrepareCudaSurface(*surface, params.GetInt(TfToken("rbfSamples"), 100),
                        &step->surface, &errors) != CudaSurfacePreparationStatus::Ok) {
                    for (auto const& error : errors) Fail(diagnostics, error);
                    return {};
                }
            }
            plan->steps.push_back(std::move(step));
        }
        std::vector<UsdGenCompiledOperatorCapability> caps;
        for (uint32_t i = 0; i != desc.nodes.size(); ++i) {
            auto const* row=GetCudaExecutionCapabilityMatrix().Find(desc.nodes[i].type);
            caps.push_back({desc.nodes[i].path,desc.nodes[i].type,i,row->capabilityVersion,row->flags,
                            UsdGenExecutionCapabilityStatus::Supported});
        }
        // One fused native task uploads captured Scatter roots and produces
        // Grow geometry. Do not invent a zero-work Operator task: the queue
        // must see exactly the native stages it actually dispatches.
        std::vector<UsdGenExecutionTaskMetadata> tasks;
        std::vector<UsdGenExecutionValueMetadata> values;
        uint32_t const parameterExternal=static_cast<uint32_t>(values.size());
        values.push_back({parameterExternal,UsdGenExecutionDataKind::ParameterValues,
            UINT32_MAX,0,UsdGenExecutionValueStorage::ExternalImmutable});
        // The ImageMaps external value keeps its slot: maps still reach the
        // descriptor even though no CUDA operator samples one today.
        values.push_back({static_cast<uint32_t>(values.size()),
            UsdGenExecutionDataKind::ImageMaps,
            UINT32_MAX,0,UsdGenExecutionValueStorage::ExternalImmutable});
        uint32_t const surfaceExternal=static_cast<uint32_t>(values.size());
        values.push_back({surfaceExternal,UsdGenExecutionDataKind::SurfaceGeometry,
            UINT32_MAX,0,UsdGenExecutionValueStorage::ExternalImmutable});
        UsdGenExecutionTaskMetadata producer;
        producer.id=0; producer.semanticNode=layout.scatterGrowOutput;
        producer.authoredOrderKey=layout.scatterGrowOutput;
        producer.path=desc.nodes[layout.scatterGrowOutput].path;
        producer.type=TfToken("UsdGenScatterGrow");
        producer.kind=UsdGenExecutionTaskKind::Source;
        producer.topologyBarrier=true;
        producer.exclusiveWorkspace=true;
        producer.resourceHazards.push_back({UsdGenExecutionDataKind::ExecutionWorkspace,
            0, UsdGenExecutionResourceHazardAccess::ReadWrite});
        uint64_t const producerScratch = plan->scatterGrowRequirements.inputBytes;
        SetTaskEstimate(&producer.estimate, plan->scatterGrowRequirements.statusBytes,
            plan->scatterGrowRequirements.outputBytes,
            0, producerScratch, true);
        std::array<uint32_t, 6> sourceValues{};
        std::array<UsdGenExecutionDataKind, 6> const sourceKinds{
            UsdGenExecutionDataKind::CurveGeometry, UsdGenExecutionDataKind::CurveTopology,
            UsdGenExecutionDataKind::StableIds, UsdGenExecutionDataKind::RootBindings,
            UsdGenExecutionDataKind::Widths, UsdGenExecutionDataKind::NamedChannels};
        for (size_t i=0;i<sourceKinds.size();++i) {
            sourceValues[i]=static_cast<uint32_t>(values.size());
            values.push_back({sourceValues[i],sourceKinds[i],0,1,
                UsdGenExecutionValueStorage::JobOwnedImmutable});
            producer.resources.push_back({sourceKinds[i],UsdGenExecutionResourceAccess::Write,
                UINT32_MAX,UINT32_MAX,sourceValues[i]});
        }
        // A fused Grow is the complete immutable source value, not merely
        // the initial width plane. Each descendant inherits the exact logical
        // bundle and replaces only the planes it actually produces.
        std::map<uint32_t, std::array<uint32_t, 6>> nodeValues;
        nodeValues[layout.scatterGrowOutput] = sourceValues;
        bool const hasGeometryDescendants = std::any_of(plan->steps.begin(), plan->steps.end(),
            [](auto const& step) { return step->type == TfToken("UsdGenLength") ||
                step->type == TfToken("UsdGenNoise") || step->type == TfToken("UsdGenDeform"); });
        bool const hasDeformDescendants = std::any_of(plan->steps.begin(), plan->steps.end(),
            [](auto const& step) { return step->type == TfToken("UsdGenDeform"); });
        bool const hasLengthDescendants = std::any_of(plan->steps.begin(), plan->steps.end(),
            [](auto const& step) { return step->type == TfToken("UsdGenLength"); });
        uint64_t pointWidthBytes=0, widthOutputs=0, widthScratch=0, widthSteady=0;
        if (!EstimateMultiply(plan->scatterGrowRequirements.pointCount,sizeof(float),&pointWidthBytes)) {
            Fail(diagnostics,"Scatter->Grow Width memory estimate overflow"); return {};
        }
        uint64_t frameDomainBytes = 0;
        if (hasGeometryDescendants &&
            !EstimateMultiply(plan->scatterRoots->positions.size(),
                9 * sizeof(float) + sizeof(uint64_t) + sizeof(uint32_t), &frameDomainBytes)) {
            Fail(diagnostics, "Scatter->Grow frame domain estimate overflow"); return {};
        }
        // The original fused native geometry stays in capture order; a
        // separately owned sorted frame domain supports stable-ID lookup.
        if (!EstimateAdd(&producer.estimate.steadyBytes, frameDomainBytes)) {
            Fail(diagnostics, "Scatter->Grow frame domain estimate overflow"); return {};
        }
        tasks.push_back(std::move(producer));
        for (size_t stepIndex=0; stepIndex!=plan->steps.size(); ++stepIndex) {
            auto const& step=*plan->steps[stepIndex];
            UsdGenExecutionTaskMetadata task;
            task.id=static_cast<uint32_t>(tasks.size()); task.semanticNode=step.semanticNode;
            task.authoredOrderKey=step.semanticNode; task.path=step.path; task.type=step.type;
            task.kind=UsdGenExecutionTaskKind::Operator;
            bool const length = step.type == TfToken("UsdGenLength");
            bool const noise = step.type == TfToken("UsdGenNoise");
            bool const deform = step.type == TfToken("UsdGenDeform");
            if (deform) task.resources.push_back({UsdGenExecutionDataKind::SurfaceGeometry,
                UsdGenExecutionResourceAccess::Read, UINT32_MAX, surfaceExternal, UINT32_MAX});
            task.exclusiveWorkspace = length || noise || deform;
            if (task.exclusiveWorkspace) task.resourceHazards.push_back({
                UsdGenExecutionDataKind::ExecutionWorkspace, 0,
                UsdGenExecutionResourceHazardAccess::ReadWrite});
            auto left = nodeValues.find(step.inputNode);
            if (left == nodeValues.end()) { Fail(diagnostics,"Scatter->Grow predecessor is unavailable"); return {}; }
            auto outputBundle = left->second;
            for (size_t kind = 0; kind != sourceKinds.size(); ++kind) {
                uint32_t const input = left->second[kind];
                bool const writes = length || ((noise || deform) ? kind == 0 : kind == 4);
                uint32_t output = UINT32_MAX;
                if (writes) {
                    output = static_cast<uint32_t>(values.size());
                    values.push_back({output, sourceKinds[kind], task.id,
                        values[input].version + 1, UsdGenExecutionValueStorage::JobOwnedImmutable});
                    outputBundle[kind] = output;
                }
                task.resources.push_back({sourceKinds[kind], writes
                    ? UsdGenExecutionResourceAccess::ReadWrite : UsdGenExecutionResourceAccess::Read,
                    values[input].producerTask, input, output});
            }
            task.resources.push_back({UsdGenExecutionDataKind::ParameterValues,
                UsdGenExecutionResourceAccess::Read, UINT32_MAX, parameterExternal, UINT32_MAX});
            if (step.rightInputNode!=UINT32_MAX) {
                auto right = nodeValues.find(step.rightInputNode);
                if(right == nodeValues.end()) { Fail(diagnostics,"Scatter->Grow WidthBlend predecessor is unavailable"); return {}; }
                for (size_t kind = 0; kind != sourceKinds.size(); ++kind) {
                    if (kind != 4 && !step.requiresNonWidthProof) continue;
                    uint32_t const input = right->second[kind];
                    task.resources.push_back({sourceKinds[kind], UsdGenExecutionResourceAccess::Read,
                        values[input].producerTask, input, UINT32_MAX});
                }
            }
            nodeValues[step.semanticNode] = outputBundle;
            uint64_t scratch=0,imageBytes=0;
            uint64_t evaluatorBytes=0;
            uint64_t outputBytes = pointWidthBytes;
            if ((noise || deform) && !EstimateMultiply(pointWidthBytes, 3, &outputBytes)) {
                Fail(diagnostics, "Scatter->Grow point output estimate overflow"); return {};
            }
            if (length) outputBytes = plan->scatterGrowRequirements.outputBytes;
            if (!EstimateParameterCandidateBytes(step.parameters,
                    plan->scatterGrowRequirements.pointCount,
                    plan->scatterRoots->positions.size(), true, &evaluatorBytes) ||
                !EstimateSum({outputBytes,257*sizeof(float),2*sizeof(int),imageBytes},&scratch)) {
                Fail(diagnostics,"Scatter->Grow operator memory estimate overflow"); return {};
            }
            if (length) {
                uint64_t curveScratch = 0, pointScratch = 0;
                if (!EstimateMultiply(plan->scatterRoots->positions.size(), 18, &curveScratch) ||
                    !EstimateMultiply(pointWidthBytes, 6, &pointScratch) ||
                    !EstimateAdd(&scratch, curveScratch) || !EstimateAdd(&scratch, pointScratch) ||
                    !EstimateAdd(&scratch, 10 * sizeof(int))) {
                    Fail(diagnostics,"Scatter->Grow Length scratch estimate overflow"); return {};
                }
            }
            if (noise && !EstimateAdd(&scratch, sizeof(int))) {
                Fail(diagnostics,"Scatter->Grow Noise status estimate overflow"); return {};
            }
            if (deform && !EstimateAdd(&scratch, outputBytes)) {
                Fail(diagnostics,"Scatter->Grow Deform scratch estimate overflow"); return {};
            }
            if (step.requiresNonWidthProof && !EstimateAdd(&scratch, 4 * sizeof(int32_t))) {
                Fail(diagnostics,"Scatter->Grow comparison estimate overflow"); return {};
            }
            if (!EstimateAdd(&widthOutputs,outputBytes) || !EstimateAdd(&widthScratch,scratch) ||
                !EstimateAdd(&widthSteady,evaluatorBytes)) {
                Fail(diagnostics,"Scatter->Grow Width memory estimate overflow"); return {};
            }
            SetTaskEstimate(&task.estimate,evaluatorBytes,outputBytes,0,scratch,!length && !deform);
            tasks.push_back(std::move(task));
        }
        UsdGenExecutionTaskMetadata publication; publication.id=static_cast<uint32_t>(tasks.size()); publication.kind=UsdGenExecutionTaskKind::Publication;
        publication.authoredOrderKey=desc.nodes.size();
        publication.path=desc.nodes[layout.terminal].path; publication.type=TfToken("UsdGenPublication");
        publication.topologyBarrier=true;
        publication.resourceHazards.push_back({UsdGenExecutionDataKind::ExecutionWorkspace,
            0, UsdGenExecutionResourceHazardAccess::ReadWrite});
        for (uint32_t id=0;id!=tasks.size();++id) publication.declaredDependencies.push_back({id,UsdGenExecutionDependencyLifetimePublicationJoin});
        for (size_t i=0;i<sourceKinds.size();++i) {
            uint32_t input = nodeValues.at(layout.terminal)[i];
            publication.resources.push_back({sourceKinds[i],UsdGenExecutionResourceAccess::Read,values[input].producerTask,input,UINT32_MAX});
        }
        uint32_t const terminalValue=static_cast<uint32_t>(values.size());
        values.push_back({terminalValue,UsdGenExecutionDataKind::TerminalGeneration,publication.id,1,
            UsdGenExecutionValueStorage::PublishedImmutable});
        publication.resources.push_back({UsdGenExecutionDataKind::TerminalGeneration,
            UsdGenExecutionResourceAccess::Write,UINT32_MAX,UINT32_MAX,terminalValue});
        uint64_t publicationScratch=0, peak=0;
        if (!EstimateTileScratchBytes(plan->scatterRoots->positions.size(),desc.tileTarget,&publicationScratch) ||
            !EstimateSum({plan->scatterGrowRequirements.outputBytes,
                plan->scatterGrowRequirements.statusBytes,frameDomainBytes,widthSteady,widthOutputs,
                std::max<uint64_t>(producerScratch,publicationScratch),widthScratch},&peak)) {
            Fail(diagnostics,"Scatter->Grow publication memory estimate overflow"); return {};
        }
        SetTaskEstimate(&publication.estimate,0,0,0,publicationScratch,true);
        tasks.push_back(std::move(publication));
        UsdGenExecutionGraphMemoryEstimate memory;
        memory.concurrentPeakBytes=peak;
        // Downstream tasks read the fused source's immutable output. The
        // upload staging is transient scratch, released before publication.
        memory.immutableSharedInputBytes=plan->scatterGrowRequirements.outputBytes;
        // CUB compaction workspace is queried on the selected device before
        // runtime admission; never label this known-payload subtotal a full
        // compile-time upper bound for a Length descendant.
        // RBF's selected solver/cache workspace is not a fixed source/point
        // payload. Its native allocations retain individual budget permits;
        // this known subtotal must not become an aggregate admission bound.
        memory.memoryAvailable=memory.conservativeUpperBound=
            !hasLengthDescendants && !hasDeformDescendants;
        memory.runtimeRefinementAvailable = hasLengthDescendants && !hasDeformDescendants;
        if (!UsdGenExecutionDependencyCompiler::Lower(&tasks,values,&reason) ||
            !UsdGenExecutionDependencyCompiler::Validate(tasks,values,&reason)) {
            Fail(diagnostics,"Scatter->Grow dependency lowering failed: "+reason); return {};
        }
        plan->metadata=std::make_shared<const UsdGenExecutionPlanMetadata>("cuda",GetCudaExecutionCapabilityMatrix().Version(),
            plan->steps.empty() ? UsdGenExecutionPlanShape::LinearAuthoredChain :
                std::any_of(plan->steps.begin(),plan->steps.end(),[](auto const& s){return s->type==TfToken("UsdGenWidthBlend");})
                    ? UsdGenExecutionPlanShape::SourceRootedValueDag : UsdGenExecutionPlanShape::SourceRootedUnaryDag,
            std::move(caps),std::move(tasks),std::move(values),
            static_cast<uint32_t>(plan->steps.size()+1),memory);
        // This compact source has no CurveSource parameter program or generic
        // operator lowering: its immutable CPU capture is its source program.
        return plan;
    }
    if (!desc.nodes[layout.source].expressionBindings.empty()) {
        std::vector<std::string> errors;
        if (CudaParameterProgram::Compile(desc, desc.nodes[layout.source], &plan->sourceParameters, &errors) != CudaParameterStatus::Ok) {
            for (auto const& error : errors) Fail(diagnostics, error);
            return {};
        }
    }
    std::vector<uint32_t> proofPointOrigin(desc.nodes.size(), UINT32_MAX),
        proofTopologyOrigin(desc.nodes.size(), UINT32_MAX);
    proofPointOrigin[layout.source] = proofTopologyOrigin[layout.source] = layout.source;
    for (auto nodeIndex : layout.operators) {
        auto const left = layout.input[nodeIndex];
        if (left == UINT32_MAX) { Fail(diagnostics, "CUDA proof origin is unavailable"); return {}; }
        auto const& node = desc.nodes[nodeIndex];
        if (node.type == TfToken("UsdGenNoise") ||
            node.type == TfToken("UsdGenDeform")) {
            proofPointOrigin[nodeIndex] = nodeIndex; proofTopologyOrigin[nodeIndex] = proofTopologyOrigin[left];
        } else if (node.type == TfToken("UsdGenGrow") || node.type == TfToken("UsdGenLength")) {
            proofPointOrigin[nodeIndex] = proofTopologyOrigin[nodeIndex] = nodeIndex;
        } else {
            proofPointOrigin[nodeIndex] = proofPointOrigin[left]; proofTopologyOrigin[nodeIndex] = proofTopologyOrigin[left];
        }
        auto width = std::make_unique<UsdGenCudaExecutionPlan::Step>();
        width->path = desc.nodes[nodeIndex].path;
        width->type = desc.nodes[nodeIndex].type;
        width->semanticNode = nodeIndex;
        width->inputNode = layout.input[nodeIndex];
        width->rightInputNode = layout.rightInput[nodeIndex];
        if (width->type == TfToken("UsdGenWidthBlend")) {
            auto const right = width->rightInputNode;
            width->requiresNonWidthProof = right != UINT32_MAX &&
                (proofPointOrigin[left] != proofPointOrigin[right] ||
                 proofTopologyOrigin[left] != proofTopologyOrigin[right]);
        }
        std::vector<std::string> errors;
        if (CudaParameterProgram::Compile(desc, desc.nodes[nodeIndex], &width->parameters, &errors) != CudaParameterStatus::Ok) {
            for (auto const& error : errors) Fail(diagnostics, error);
            if (errors.empty()) Fail(diagnostics, "operator parameter compilation failed");
            return {};
        }
        UsdGenParamView params; params.desc = &desc; params.node = &desc.nodes[nodeIndex];
        if (width->type == TfToken("UsdGenGrow")) {
            auto& controls = width->grow;
            controls.cvCount = uint32_t(params.GetInt(TfToken("segments"), 8));
            controls.seed = desc.nodes[nodeIndex].seed;
            controls.length = params.GetDouble(TfToken("length"), 1.0);
            auto const random = params.GetVtValue(TfToken("lengthRandom"), VtValue(GfVec2f(1,1))).UncheckedGet<GfVec2f>();
            controls.randomLo = random[0]; controls.randomHi = random[1];
            controls.lift = float(params.GetDouble(TfToken("lift"), 0.0));
            controls.azimuth = float(params.GetDouble(TfToken("azimuth"), 0.0));
            controls.azimuthRandom = float(params.GetDouble(TfToken("azimuthRandom"), 0.0));
            controls.fallbackWidth = desc.defaultWidth;
            auto const direction = params.GetToken(TfToken("direction"), TfToken("surfaceNormal"));
            controls.direction = direction == TfToken("vector") ? gpu::CurveGrowDirection::Literal :
                direction == TfToken("attribute") ? gpu::CurveGrowDirection::RootTangent : gpu::CurveGrowDirection::RootNormal;
            auto const vector = params.GetVtValue(TfToken("directionVector"), VtValue(GfVec3f(0,1,0))).UncheckedGet<GfVec3f>();
            controls.literalDirection = make_float3(vector[0], vector[1], vector[2]);
        }
        auto lut = [&](char const* knotsName, char const* interpolationName, auto& output) {
            auto value = params.GetVtValue(TfToken(knotsName), VtValue(VtVec2fArray{}));
            UsdGenBuildRampLut(value.UncheckedGet<VtVec2fArray>(),
                params.GetToken(TfToken(interpolationName), TfToken("catmullRom")), output.data(), output.size());
        };
        if (width->type == TfToken("UsdGenWidth")) {
            lut("width:knots", "width:interpolation", width->profile);
            UsdGenWidthBezierLut(params, width->profile.data(), width->profile.size());
        } else if (width->type == TfToken("UsdGenNoise"))
            lut("noise:magnitude:knots", "noise:magnitude:interpolation", width->profile);
        if (width->type == TfToken("UsdGenDeform")) {
            auto found = std::find_if(desc.surfaces.begin(), desc.surfaces.end(),
                [&](auto const& surface) { return surface.path == desc.nodes[nodeIndex].surfaces.front(); });
            if (found == desc.surfaces.end()) { Fail(diagnostics, "missing RBF surface"); return {}; }
            if (PrepareCudaSurface(*found, params.GetInt(TfToken("rbfSamples"), 100),
                    &width->surface, &errors) != CudaSurfacePreparationStatus::Ok) {
                for (auto const& error : errors) Fail(diagnostics, error);
                return {};
            }
        }
        plan->steps.push_back(std::move(width));
    }
#endif
    auto const& matrix = GetCudaExecutionCapabilityMatrix();
    std::vector<UsdGenCompiledOperatorCapability> capabilities;
    std::vector<UsdGenExecutionTaskMetadata> tasks;
    std::vector<UsdGenExecutionValueMetadata> values;
    capabilities.reserve(desc.nodes.size());
    tasks.reserve(desc.nodes.size() + 1);
    // The descriptor is the only compile-time cardinality source.  Keep this
    // accounting independent of CUDA objects so plan consumers can inspect it
    // before a device/context exists.
    auto const& sourceNodeDesc = desc.nodes[layout.source];
    SdfPath const sourcePath = sourceNodeDesc.type == TfToken("UsdGenReferenceSource")
        ? sourceNodeDesc.references.front()
        : sourceNodeDesc.curves.front();
    auto sourceIt = std::find_if(desc.curveSets.begin(), desc.curveSets.end(),
        [&](auto const& curves) { return curves.path == sourcePath; });
    if (sourceIt == desc.curveSets.end()) {
        Fail(diagnostics, "CUDA memory estimate requires the source curve set");
        return {};
    }
    UsdGenParamView sourceParams;
    sourceParams.desc = &desc;
    sourceParams.node = &desc.nodes[layout.source];
    bool const sourceSurfaceCage = sourceNodeDesc.type == TfToken("UsdGenCurveSource") &&
        sourceParams.GetToken(TfToken("interpolationMode"), TfToken("none")) ==
            TfToken("surfaceCage");
    uint64_t sourcePoints = static_cast<uint64_t>(sourceIt->points.size());
    uint64_t sourceCurves =
        static_cast<uint64_t>(sourceIt->curveVertexCounts.size());
    bool sourceHasRoots = SourceMayProduceRootBindings(sourceNodeDesc, *sourceIt);
    [[maybe_unused]] bool sourceHasCompleteRoots =
        sourceIt->skinPrim.size() == sourceCurves &&
        sourceIt->skinPrimUv.size() == sourceCurves;
    // A sparse surface cage is expanded before any CUDA upload.  Estimate its
    // worst case now, before the bridge allocates transient host arrays, so
    // the normal source/output and task-admission accounting sees dense C3
    // cardinality rather than the authored rail count.  Map clipping only
    // lowers this bound.
    if (sourceSurfaceCage) {
        std::string error;
        uint64_t denseCurves = 0, densePoints = 0;
        if (!UsdGenEstimateSurfaceCageCurveSet(desc, sourceNodeDesc, *sourceIt,
                                                &denseCurves, &densePoints, &error)) {
            Fail(diagnostics, "CUDA surfaceCage memory preflight failed: " + error);
            return {};
        }
        sourceCurves = denseCurves;
        sourcePoints = densePoints;
        sourceHasRoots = true;
        sourceHasCompleteRoots = true;
#ifdef USDGEN_ENABLE_CUDA
        plan->surfaceCageMaxCurves = denseCurves;
        plan->surfaceCageMaxCvs = densePoints;
#endif
    }
    uint64_t sourceNamedChannelBytes = 0;
    if (sourceSurfaceCage) {
        // The bridge emits uniform tubeId plus optional regionId and
        // hierarchyLevel reporting planes. Charge all three even when the
        // latter two are absent on a generic cage; payload arrays themselves
        // are source metadata and never travel with dense C3 output.
        if (!EstimateMultiply(sourceCurves, 3 * sizeof(uint32_t), &sourceNamedChannelBytes)) {
            Fail(diagnostics, "CUDA surfaceCage named-channel memory estimate overflow");
            return {};
        }
    } else {
        for (UsdGenAuthoredPlaneDesc const& plane : sourceIt->authoredPlanes) {
            uint64_t planeValues = plane.type == UsdGenAuthoredPlaneType::Float32
                ? static_cast<uint64_t>(plane.floatValues.size())
                : static_cast<uint64_t>(plane.intValues.size());
            uint64_t bytes = 0;
            if (!EstimateMultiply(planeValues, sizeof(uint32_t), &bytes) ||
                !EstimateAdd(&sourceNamedChannelBytes, bytes)) {
                Fail(diagnostics, "CUDA memory estimate overflow for authored named channels");
                return {};
            }
        }
    }
    bool sourceHasDynamicResample = false;
    for (auto const& binding : desc.nodes[layout.source].expressionBindings) {
        if (LocalName(binding.destination) == "resampleTo") {
            sourceHasDynamicResample = true;
            break;
        }
    }
    int const literalResample = sourceParams.GetInt(TfToken("resampleTo"), 0);
    uint64_t finalPoints = sourcePoints;
    bool finalCardinalityKnown = !sourceHasDynamicResample;
    if (finalCardinalityKnown && literalResample > 0 &&
        !EstimateMultiply(sourceCurves, static_cast<uint64_t>(literalResample),
                          &finalPoints)) {
        Fail(diagnostics, "CUDA memory estimate overflow for CurveSource resample output");
        return {};
    }
    bool const hasC3Grow = std::any_of(layout.operators.begin(), layout.operators.end(),
        [&](uint32_t index) { return desc.nodes[index].type == TfToken("UsdGenGrow"); });
    // C3 and Grow outputs coexist through publication. A common upper-bound
    // cardinality keeps all evaluator/Width estimates conservative even when
    // Grow increases or reduces the per-curve CV count.
    if (hasC3Grow && finalCardinalityKnown) {
        for (auto index : layout.operators) if (desc.nodes[index].type == TfToken("UsdGenGrow")) {
            UsdGenParamView controls{&desc, &desc.nodes[index]};
            uint64_t grownPoints = 0;
            if (!EstimateMultiply(sourceCurves, uint64_t(controls.GetInt(TfToken("segments"), 8)), &grownPoints)) {
                Fail(diagnostics, "CUDA Grow memory estimate overflow"); return {};
            }
            finalPoints = std::max(finalPoints, grownPoints);
        }
    }
    uint64_t finalNamedChannelBytes = 0;
    if (finalCardinalityKnown) {
        if (sourceSurfaceCage) {
            if (!EstimateMultiply(sourceCurves, 3 * sizeof(uint32_t),
                                  &finalNamedChannelBytes)) {
                Fail(diagnostics,
                    "CUDA surfaceCage resampled named-channel estimate overflow");
                return {};
            }
        } else for (UsdGenAuthoredPlaneDesc const& plane : sourceIt->authoredPlanes) {
            uint64_t elements = plane.domain == UsdGenAuthoredPlaneDomain::Point
                ? finalPoints
                : plane.domain == UsdGenAuthoredPlaneDomain::Primitive
                ? sourceCurves : 1;
            uint64_t planeValues = 0, bytes = 0;
            if (!EstimateMultiply(elements, static_cast<uint64_t>(plane.arity),
                                  &planeValues) ||
                !EstimateMultiply(planeValues, sizeof(uint32_t), &bytes) ||
                !EstimateAdd(&finalNamedChannelBytes, bytes)) {
                Fail(diagnostics,
                    "CUDA memory estimate overflow for resampled named channels");
                return {};
            }
        }
    }
    uint64_t sourceGeometryBytes = 0;
    if (!EstimateGeometryBytes(sourcePoints, sourceCurves, sourceHasRoots,
                               &sourceGeometryBytes) ||
        !EstimateAdd(&sourceGeometryBytes, sourceNamedChannelBytes)) {
        Fail(diagnostics, "CUDA memory estimate overflow for CurveSource input");
        return {};
    }
    // The bridge first owns a dense helper result and then hands an ordinary
    // C3 candidate to CurveLoader.  Charge both host packets before source
    // preparation so the dense expansion cannot allocate beyond the
    // compile-time-admitted source phase.  The CUDA reservation is a
    // conservative common budget for this CPU preparation boundary and its
    // subsequent device upload.
    uint64_t surfaceCagePreparationBytes = 0;
    if (sourceSurfaceCage &&
        (!EstimateMultiply(sourceGeometryBytes, 2, &surfaceCagePreparationBytes) ||
         !EstimateAdd(&surfaceCagePreparationBytes, sourceNamedChannelBytes))) {
        Fail(diagnostics, "CUDA surfaceCage host preparation memory estimate overflow");
        return {};
    }
    uint64_t finalGeometryBytes = 0;
    if (finalCardinalityKnown &&
        !EstimateGeometryBytes(finalPoints, sourceCurves, sourceHasRoots,
                               &finalGeometryBytes)) {
        Fail(diagnostics, "CUDA memory estimate overflow for CurveSource output");
        return {};
    }
    if (finalCardinalityKnown &&
        !EstimateAdd(&finalGeometryBytes, finalNamedChannelBytes)) {
        Fail(diagnostics, "CUDA memory estimate overflow for authored named output channels");
        return {};
    }
    for (size_t i = 0; i != desc.nodes.size(); ++i) {
        auto const& node = desc.nodes[i];
        auto const* capability = matrix.Find(node.type);
        if (!capability) {
            Fail(diagnostics, "CUDA capability metadata is missing " + node.type.GetString());
            return {};
        }
        capabilities.push_back({node.path, node.type, static_cast<uint64_t>(i),
            capability->capabilityVersion, capability->flags,
            UsdGenExecutionCapabilityStatus::Supported});
    }

    std::vector<uint32_t> normalizedNodes{layout.source};
    normalizedNodes.insert(normalizedNodes.end(), layout.operators.begin(),
                           layout.operators.end());
    std::vector<uint32_t> semanticToTask(desc.nodes.size(), UINT32_MAX);
    std::vector<std::map<UsdGenExecutionDataKind, uint32_t>> nodeValues(desc.nodes.size());
    std::map<UsdGenExecutionDataKind, uint32_t> latestResourceValue;
    std::map<UsdGenExecutionDataKind, uint32_t> resourceVersion;
    std::map<UsdGenExecutionDataKind, uint32_t> externalValues;
    auto externalValue = [&](UsdGenExecutionDataKind kind) {
        auto found = externalValues.find(kind);
        if (found != externalValues.end()) return found->second;
        uint32_t const id = static_cast<uint32_t>(values.size());
        values.push_back({id, kind, UINT32_MAX, 0,
            UsdGenExecutionValueStorage::ExternalImmutable});
        externalValues[kind] = id;
        return id;
    };
    auto bindResourceValues = [&](UsdGenExecutionTaskMetadata& task,
                                  uint32_t semanticNode) {
        for (auto& resource : task.resources) {
            if (resource.access != UsdGenExecutionResourceAccess::Write) {
                uint32_t inputValue = UINT32_MAX;
                if (layout.taskDag && task.kind == UsdGenExecutionTaskKind::Operator) {
                    auto found = nodeValues[semanticNode].find(resource.resource);
                    if (found != nodeValues[semanticNode].end()) inputValue = found->second;
                } else {
                    auto found = latestResourceValue.find(resource.resource);
                    if (found != latestResourceValue.end()) inputValue = found->second;
                }
                if (inputValue == UINT32_MAX) inputValue = externalValue(resource.resource);
                resource.inputValue = inputValue;
                auto const* input = &values[resource.inputValue];
                resource.producerTask = input->producerTask;
            }
            if (resource.access != UsdGenExecutionResourceAccess::Read) {
                uint32_t const id = static_cast<uint32_t>(values.size());
                uint32_t version = ++resourceVersion[resource.resource];
                if (layout.taskDag) {
                    version = resource.inputValue != UINT32_MAX
                        ? values[resource.inputValue].version + 1 : 1;
                }
                auto const storage = task.kind == UsdGenExecutionTaskKind::Publication
                    ? UsdGenExecutionValueStorage::PublishedImmutable
                    : layout.taskDag &&
                        (task.kind == UsdGenExecutionTaskKind::Source ||
                         task.kind == UsdGenExecutionTaskKind::Operator)
                    ? UsdGenExecutionValueStorage::JobOwnedImmutable
                    : UsdGenExecutionValueStorage::ExclusiveWorkspaceVersion;
                values.push_back({id, resource.resource, task.id, version, storage});
                resource.outputValue = id;
                latestResourceValue[resource.resource] = id;
                if (semanticNode != UINT32_MAX)
                    nodeValues[semanticNode][resource.resource] = id;
            }
        }
    };
    auto setCudaTaskEstimate = [&](UsdGenExecutionTaskMetadata* task,
                                   UsdGenNodeDesc const& node) {
        uint64_t pointFloatBytes = 0;
        if (finalCardinalityKnown &&
            !EstimateMultiply(finalPoints, sizeof(float), &pointFloatBytes)) {
            return Fail(diagnostics, "CUDA memory estimate overflow for " +
                node.type.GetString() + " point payload");
        }
        uint64_t evaluatorBytes = 0;
#ifdef USDGEN_ENABLE_CUDA
        std::shared_ptr<const CudaParameterProgram> parameterProgram;
        if (task->kind == UsdGenExecutionTaskKind::Source) {
            parameterProgram = plan->sourceParameters;
        } else {
            auto found = std::find_if(plan->steps.begin(), plan->steps.end(),
                [&](auto const& step) {
                    return step->semanticNode == task->semanticNode;
                });
            if (found != plan->steps.end()) parameterProgram = (*found)->parameters;
        }
        if (finalCardinalityKnown &&
            !EstimateParameterCandidateBytes(parameterProgram, finalPoints,
                sourceCurves, sourceHasRoots, &evaluatorBytes)) {
            return Fail(diagnostics, "CUDA expression evaluator memory estimate overflow for " +
                node.type.GetString());
        }
#endif
        if (task->kind == UsdGenExecutionTaskKind::Source) {
            uint64_t scratch = surfaceCagePreparationBytes;
            uint64_t steady = evaluatorBytes;
            uint64_t output = finalCardinalityKnown ? finalGeometryBytes : 0;
            uint64_t retention = 0;
            if (hasC3Grow || std::any_of(layout.operators.begin(), layout.operators.end(),
                    [&](uint32_t index) { return desc.nodes[index].type == TfToken("UsdGenNoise"); })) {
                uint64_t frames = 0;
                if (!EstimateMultiply(sourceCurves, 9 * sizeof(float), &frames) ||
                    !EstimateAdd(&steady, frames))
                    return Fail(diagnostics, "CUDA Grow source-frame estimate overflow");
            }
#ifdef USDGEN_ENABLE_CUDA
            if (parameterProgram && !parameterProgram->Bindings().empty() &&
                !EstimateAdd(&scratch, CudaGroomScalarReadback::Capacity * 8))
                return Fail(diagnostics, "CUDA memory estimate overflow for CurveSource scalar readback");
#endif
            if (literalResample > 0 && finalCardinalityKnown) {
                retention = sourceGeometryBytes;
                // CurveResample promotes its staging geometry to the retained
                // output on Finish. Its device and pinned-host diagnostic
                // scalars remain owned by that resampler through publication,
                // so they are steady auxiliary bytes rather than recyclable
                // source-phase scratch.
                if (!EstimateAdd(&steady, 2 * sizeof(int)))
                    return Fail(diagnostics, "CUDA memory estimate overflow for CurveSource steady storage");
                if (!sourceIt->authoredPlanes.empty() &&
                    !EstimateAdd(&scratch, 2 * sizeof(int)))
                    return Fail(diagnostics,
                        "CUDA memory estimate overflow for CurveSource named topology proof");
            }
            SetTaskEstimate(&task->estimate, steady, output, retention, scratch,
                finalCardinalityKnown);
            return true;
        }
        if (node.type == TfToken("UsdGenWidth")) {
            uint64_t scratch = 0;
            uint64_t retention = finalCardinalityKnown ? finalGeometryBytes : 0;
            // A descendant width consumes its immediate predecessor's unique
            // CoW width allocation in addition to the shared source geometry.
            if (layout.input[task->semanticNode] != layout.source &&
                !EstimateAdd(&retention, pointFloatBytes)) {
                return Fail(diagnostics, "CUDA memory estimate overflow for Width producer retention");
            }
            // CudaWidth allocates output staging plus one 257-float LUT and
            // device/pinned error scalars.
            if (!EstimateSum({pointFloatBytes, 257 * sizeof(float),
                              2 * sizeof(int)}, &scratch)) {
                return Fail(diagnostics, "CUDA memory estimate overflow for Width scratch");
            }
            SetTaskEstimate(&task->estimate, evaluatorBytes, pointFloatBytes,
                retention, scratch, finalCardinalityKnown);
            return true;
        }
        if (node.type == TfToken("UsdGenWidthBlend")) {
            uint64_t retention = finalCardinalityKnown ? finalGeometryBytes : 0;
            // Both predecessor Width values remain immutable and live until
            // the blend terminal joins the job.  Count both explicitly here;
            // graph-level accounting deduplicates them as persistent values.
            if (!EstimateAdd(&retention, pointFloatBytes) ||
                !EstimateAdd(&retention, pointFloatBytes))
                return Fail(diagnostics,
                    "CUDA memory estimate overflow for WidthBlend producer retention");
            // Cross-origin branches first prove all non-width C3 payloads on
            // device.  The comparator owns one device and one pinned result
            // scalar through its asynchronous status/commit handoff.
            // Device result is two int fields and the pinned status mirrors
            // it. Keep this estimator CUDA-header-neutral for CPU builds.
            constexpr uint64_t compareStatusBytes = 4 * sizeof(int32_t);
            auto const proof = task->semanticNode < layout.requiresNonWidthProof.size() &&
                layout.requiresNonWidthProof[task->semanticNode];
            SetTaskEstimate(&task->estimate, 0, pointFloatBytes, retention,
                proof ? compareStatusBytes : 0, finalCardinalityKnown);
            return true;
        }
        if (node.type == TfToken("UsdGenNoise")) {
            uint64_t frameBytes = 0, scratch = 0, retention =
                finalCardinalityKnown ? finalGeometryBytes : 0;
            // Frames remain an immutable source-order COW input. Noise uses
            // stable-ID lookup after an upstream Length compaction, so the
            // source frame footprint is retained even when the active curve
            // count is smaller.
            uint64_t pointFloat3Bytes = 0;
            if (!EstimateMultiply(pointFloatBytes, 3, &pointFloat3Bytes) ||
                !EstimateMultiply(sourceCurves, 9 * sizeof(float), &frameBytes) ||
                !EstimateAdd(&retention, frameBytes) ||
                !EstimateSum({pointFloat3Bytes, 257 * sizeof(float),
                              2 * sizeof(int), sizeof(int)}, &scratch))
                return Fail(diagnostics, "CUDA memory estimate overflow for Noise");
            SetTaskEstimate(&task->estimate, evaluatorBytes, pointFloat3Bytes,
                retention, scratch, finalCardinalityKnown);
            return true;
        }
        if (node.type == TfToken("UsdGenGrow")) {
            uint64_t frames = 0, output = finalGeometryBytes, steady = evaluatorBytes;
            uint64_t scratch = sourceIt->authoredPlanes.empty() ? 0 : 2 * sizeof(int);
            if (!EstimateMultiply(sourceCurves, 9 * sizeof(float), &frames) ||
                !EstimateAdd(&output, frames) || !EstimateAdd(&steady, 2 * sizeof(int)))
                return Fail(diagnostics, "CUDA Grow output estimate overflow");
            SetTaskEstimate(&task->estimate, steady, output, finalGeometryBytes,
                scratch, finalCardinalityKnown);
            return true;
        }
        if (node.type == TfToken("UsdGenLength")) {
            uint64_t changedPoints = 0, curveScratch = 0, scratch = 0;
            if (!EstimateMultiply(finalPoints, 6 * sizeof(float), &changedPoints) ||
                !EstimateMultiply(sourceCurves, 18, &curveScratch) ||
                !EstimateSum({changedPoints,
                              curveScratch, 10 * sizeof(int)}, &scratch)) {
                return Fail(diagnostics, "CUDA memory estimate overflow for Length scratch");
            }
            // These are the known external, Length and compaction bytes for
            // the async path: point/keep staging, four prefix arrays,
            // device status/counts and pinned status/counts, including the
            // named-topology device/pinned validation pair. CUB workspace
            // and survivor storage are selected-device/runtime data, so this
            // remains partial until the graph-level refinement runs.
            SetTaskEstimate(&task->estimate, evaluatorBytes,
                finalCardinalityKnown ? finalGeometryBytes : 0,
                finalCardinalityKnown ? finalGeometryBytes : 0, scratch, false);
            return true;
        }
        // RBF has explicit deformation buffers, but cuSOLVER workspace and
        // field/rebind cardinalities are runtime-dependent. Preserve the
        // known payload components while correctly marking the total absent.
        uint64_t deformScratch = 0;
        uint64_t pointFloat3Bytes = 0;
        if (!EstimateMultiply(finalPoints, 6 * sizeof(float), &deformScratch) ||
            !EstimateMultiply(pointFloatBytes, 3, &pointFloat3Bytes) ||
            !EstimateAdd(&deformScratch, sizeof(int))) {
            return Fail(diagnostics, "CUDA memory estimate overflow for Deform scratch");
        }
        SetTaskEstimate(&task->estimate, evaluatorBytes, pointFloat3Bytes,
            finalCardinalityKnown ? finalGeometryBytes : 0, deformScratch, false);
        return true;
    };
    for (uint32_t taskIndex = 0; taskIndex != normalizedNodes.size(); ++taskIndex) {
        uint32_t const semanticNode = normalizedNodes[taskIndex];
        auto const& node = desc.nodes[semanticNode];
        auto const* capability = matrix.Find(node.type);
        UsdGenExecutionTaskMetadata task;
        task.id = taskIndex;
        task.semanticNode = semanticNode;
        task.authoredOrderKey = static_cast<uint64_t>(semanticNode);
        task.kind = semanticNode == layout.source ? UsdGenExecutionTaskKind::Source :
            UsdGenExecutionTaskKind::Operator;
        task.path = node.path;
        task.type = node.type;
        semanticToTask[semanticNode] = task.id;
        if (layout.taskDag && task.kind == UsdGenExecutionTaskKind::Operator)
            nodeValues[semanticNode] = nodeValues[layout.input[semanticNode]];
        task.topologyBarrier =
            (capability->flags & UsdGenCapabilityTopologyBarrier) != 0;
        task.exclusiveWorkspace =
            (capability->flags & UsdGenCapabilityExclusiveWorkspace) != 0;
        if (!layout.taskDag && node.type == TfToken("UsdGenWidth"))
            task.exclusiveWorkspace = true;
        if (layout.taskDag && IsCudaWidthValueNode(node.type))
            task.exclusiveWorkspace = false;
        if (layout.geometryValueDag && node.type == TfToken("UsdGenNoise"))
            task.exclusiveWorkspace = true;
        if (task.kind == UsdGenExecutionTaskKind::Source) {
            task.resources = {
                {UsdGenExecutionDataKind::CurveGeometry, UsdGenExecutionResourceAccess::Write},
                {UsdGenExecutionDataKind::CurveTopology, UsdGenExecutionResourceAccess::Write},
                {UsdGenExecutionDataKind::StableIds, UsdGenExecutionResourceAccess::Write},
                {UsdGenExecutionDataKind::RootBindings, UsdGenExecutionResourceAccess::Write},
                {UsdGenExecutionDataKind::Widths, UsdGenExecutionResourceAccess::Write},
                {UsdGenExecutionDataKind::NamedChannels, UsdGenExecutionResourceAccess::Write},
                {UsdGenExecutionDataKind::ParameterValues, UsdGenExecutionResourceAccess::Read}};
        } else if (node.type == TfToken("UsdGenWidth")) {
            task.resources = {
                {UsdGenExecutionDataKind::CurveGeometry, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::CurveTopology, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::ParameterValues, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::Widths, UsdGenExecutionResourceAccess::ReadWrite}};
            if (!node.mapBindings.empty()) task.resources.push_back(
                {UsdGenExecutionDataKind::ImageMaps,
                 UsdGenExecutionResourceAccess::Read});
            task.resources.push_back({UsdGenExecutionDataKind::StableIds,
                UsdGenExecutionResourceAccess::Read});
            task.resources.push_back({UsdGenExecutionDataKind::RootBindings,
                UsdGenExecutionResourceAccess::Read});
        } else if (node.type == TfToken("UsdGenWidthBlend")) {
            task.resources = {
                {UsdGenExecutionDataKind::CurveGeometry, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::CurveTopology, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::StableIds, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::RootBindings, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::NamedChannels, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::Widths, UsdGenExecutionResourceAccess::ReadWrite},
                // A second Widths read models ordered right-hand fan-in as a
                // real logical value/dependency, never an undocumented wait.
                {UsdGenExecutionDataKind::Widths, UsdGenExecutionResourceAccess::Read}};
        } else if (node.type == TfToken("UsdGenLength") || node.type == TfToken("UsdGenGrow")) {
            task.resources = {
                {UsdGenExecutionDataKind::CurveGeometry, UsdGenExecutionResourceAccess::ReadWrite},
                {UsdGenExecutionDataKind::CurveTopology, UsdGenExecutionResourceAccess::ReadWrite},
                {UsdGenExecutionDataKind::StableIds, UsdGenExecutionResourceAccess::ReadWrite},
                {UsdGenExecutionDataKind::RootBindings, UsdGenExecutionResourceAccess::ReadWrite},
                {UsdGenExecutionDataKind::Widths, UsdGenExecutionResourceAccess::ReadWrite},
                {UsdGenExecutionDataKind::NamedChannels, UsdGenExecutionResourceAccess::ReadWrite},
                {UsdGenExecutionDataKind::ParameterValues, UsdGenExecutionResourceAccess::Read}};
        } else if (node.type == TfToken("UsdGenNoise")) {
            task.resources = {
                {UsdGenExecutionDataKind::CurveGeometry, UsdGenExecutionResourceAccess::ReadWrite},
                {UsdGenExecutionDataKind::CurveTopology, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::StableIds, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::RootBindings, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::Widths, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::NamedChannels, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::ParameterValues, UsdGenExecutionResourceAccess::Read}};
        } else if (node.type == TfToken("UsdGenDeform")) {
            // Deform establishes only a new point owner. Its exact authored
            // predecessor continues to own topology, IDs, root bindings,
            // widths, and named channels, all of which are immutable inputs
            // to the RBF operation and subsequent non-width fan-in proof.
            task.resources = {
                {UsdGenExecutionDataKind::CurveGeometry, UsdGenExecutionResourceAccess::ReadWrite},
                {UsdGenExecutionDataKind::CurveTopology, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::StableIds, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::RootBindings, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::Widths, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::NamedChannels, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::SurfaceGeometry, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::ParameterValues, UsdGenExecutionResourceAccess::Read}};
        } else {
            task.resources = {
                {UsdGenExecutionDataKind::CurveGeometry, UsdGenExecutionResourceAccess::ReadWrite},
                {UsdGenExecutionDataKind::RootBindings, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::SurfaceGeometry, UsdGenExecutionResourceAccess::Read},
                {UsdGenExecutionDataKind::ParameterValues, UsdGenExecutionResourceAccess::Read}};
        }
        if (node.type == TfToken("UsdGenGrow") && !node.mapBindings.empty())
            task.resources.push_back({UsdGenExecutionDataKind::ImageMaps,
                UsdGenExecutionResourceAccess::Read});
        bindResourceValues(task, semanticNode);
        if (node.type == TfToken("UsdGenWidthBlend")) {
            uint32_t const right = layout.rightInput[semanticNode];
            auto found = right < nodeValues.size()
                ? nodeValues[right].find(UsdGenExecutionDataKind::Widths)
                : nodeValues.front().end();
            if (right >= nodeValues.size() || found == nodeValues[right].end()) {
                Fail(diagnostics,
                    "CUDA WidthBlend right predecessor logical value is unavailable");
                return {};
            }
            auto rightUse = std::find_if(task.resources.rbegin(), task.resources.rend(),
                [](UsdGenExecutionResourceUse const& use) {
                    return use.resource == UsdGenExecutionDataKind::Widths &&
                        use.access == UsdGenExecutionResourceAccess::Read;
                });
            if (rightUse == task.resources.rend()) {
                Fail(diagnostics, "CUDA WidthBlend right resource use is missing");
                return {};
            }
            rightUse->inputValue = found->second;
            rightUse->producerTask = values[found->second].producerTask;
            // Only a cross-origin join reads the right predecessor's complete
            // non-width packet.  Preserve the tight same-origin WidthBlend
            // contract by adding these explicit immutable fan-in values only
            // when its compiled proof bit requires them.
            if (semanticNode < layout.requiresNonWidthProof.size() &&
                layout.requiresNonWidthProof[semanticNode]) {
                for (auto kind : {UsdGenExecutionDataKind::CurveGeometry,
                                  UsdGenExecutionDataKind::CurveTopology,
                                  UsdGenExecutionDataKind::StableIds,
                                  UsdGenExecutionDataKind::RootBindings,
                                  UsdGenExecutionDataKind::NamedChannels}) {
                    auto const leftUse = std::find_if(task.resources.begin(), task.resources.end(),
                        [kind](UsdGenExecutionResourceUse const& use) {
                            return use.resource == kind &&
                                use.access == UsdGenExecutionResourceAccess::Read;
                        });
                    auto const rightValue = right < nodeValues.size()
                        ? nodeValues[right].find(kind) : nodeValues.front().end();
                    if (leftUse == task.resources.end() || right >= nodeValues.size() ||
                        rightValue == nodeValues[right].end()) {
                        Fail(diagnostics, "CUDA WidthBlend right non-width resource is unavailable");
                        return {};
                    }
                    auto use = *leftUse;
                    use.inputValue = rightValue->second;
                    use.producerTask = values[use.inputValue].producerTask;
                    task.resources.push_back(std::move(use));
                }
            }
        }
        // The mutable linear workspace lane is explicit metadata, rather than
        // an implicit prior-task edge.  Width-DAG inputs/outputs are immutable
        // logical values or task-private CoW storage, so it opts out.
        if (task.exclusiveWorkspace) {
            task.resourceHazards.push_back({
                UsdGenExecutionDataKind::ExecutionWorkspace,
                0,
                UsdGenExecutionResourceHazardAccess::ReadWrite});
        }
        if (!setCudaTaskEstimate(&task, node)) return {};
        tasks.push_back(std::move(task));
    }

    UsdGenExecutionTaskMetadata terminal;
    terminal.id = static_cast<uint32_t>(normalizedNodes.size());
    terminal.semanticNode = UINT32_MAX;
    terminal.authoredOrderKey = static_cast<uint64_t>(desc.nodes.size());
    terminal.kind = UsdGenExecutionTaskKind::Publication;
    terminal.path = desc.nodes[layout.terminal].path;
    terminal.type = TfToken("UsdGenPublication");
    // Publication consumes the declared terminal value, while this explicit
    // lifetime join proves every launched relay has completed before any
    // job-owned stream/value can be moved or released.  It is not a hidden
    // serialization edge and is intentionally independent of data flow.
    for (uint32_t id = 0; id != normalizedNodes.size(); ++id)
        terminal.declaredDependencies.push_back({
            id, UsdGenExecutionDependencyLifetimePublicationJoin});
    terminal.resources = {
        {UsdGenExecutionDataKind::CurveGeometry, UsdGenExecutionResourceAccess::Read},
        {UsdGenExecutionDataKind::CurveTopology, UsdGenExecutionResourceAccess::Read},
        {UsdGenExecutionDataKind::StableIds, UsdGenExecutionResourceAccess::Read},
        {UsdGenExecutionDataKind::RootBindings, UsdGenExecutionResourceAccess::Read},
        {UsdGenExecutionDataKind::Widths, UsdGenExecutionResourceAccess::Read},
        {UsdGenExecutionDataKind::NamedChannels, UsdGenExecutionResourceAccess::Read},
        {UsdGenExecutionDataKind::TerminalGeneration, UsdGenExecutionResourceAccess::Write}};
    terminal.topologyBarrier = true;
    if (layout.taskDag) {
        for (auto& resource : terminal.resources) {
            if (resource.access == UsdGenExecutionResourceAccess::Write) {
                uint32_t const id = static_cast<uint32_t>(values.size());
                values.push_back({id, resource.resource, terminal.id,
                    ++resourceVersion[resource.resource],
                    UsdGenExecutionValueStorage::PublishedImmutable});
                resource.outputValue = id;
                continue;
            }
            auto found = nodeValues[layout.terminal].find(resource.resource);
            resource.inputValue = found == nodeValues[layout.terminal].end()
                ? externalValue(resource.resource) : found->second;
            resource.producerTask = values[resource.inputValue].producerTask;
        }
    } else {
        bindResourceValues(terminal, UINT32_MAX);
    }
    terminal.resourceHazards.push_back({
        UsdGenExecutionDataKind::ExecutionWorkspace,
        0,
        UsdGenExecutionResourceHazardAccess::ReadWrite});
    uint64_t publicationScratch = 0;
    if (!EstimateTileScratchBytes(sourceCurves, desc.tileTarget, &publicationScratch)) {
        Fail(diagnostics, "CUDA memory estimate overflow for Publication scratch");
        return {};
    }
    // Publication creates tile metadata, not another geometry allocation; it
    // consumes the already-retained terminal logical value.
    SetTaskEstimate(&terminal.estimate, 0, 0, 0, publicationScratch, true);
    uint32_t const terminalId = terminal.id;
    tasks.push_back(std::move(terminal));

    UsdGenExecutionGraphMemoryEstimate graphMemory;
    bool const hasLiteralLength = std::any_of(layout.operators.begin(),
        layout.operators.end(), [&](uint32_t semanticNode) {
            return desc.nodes[semanticNode].type == TfToken("UsdGenLength");
        });
    bool const literalLengthOperators = std::all_of(layout.operators.begin(),
        layout.operators.end(), [&](uint32_t semanticNode) {
            auto const& operatorNode = desc.nodes[semanticNode];
            return (IsCudaWidthValueNode(operatorNode.type) ||
                    operatorNode.type == TfToken("UsdGenLength") ||
                    operatorNode.type == TfToken("UsdGenGrow") ||
                    operatorNode.type == TfToken("UsdGenNoise"));
        });
    bool literalRbf = false;
#ifdef USDGEN_ENABLE_CUDA
    literalRbf = !layout.widthDag && finalCardinalityKnown &&
        sourceHasCompleteRoots &&
        literalResample == 0 && layout.operators.size() == 1 &&
        desc.nodes[layout.operators.front()].type == TfToken("UsdGenDeform") &&
        !HasExpressionBinding(desc.nodes[layout.operators.front()], "rbfSamples") &&
        plan->steps.size() == 1 && !plan->steps.front()->surface.restPoints.empty() &&
        plan->steps.front()->surface.currentPoints.size() ==
            plan->steps.front()->surface.restPoints.size() &&
        !plan->steps.front()->surface.faceOffsets.empty();
#endif
    graphMemory.runtimeRefinementAvailable = !layout.widthDag &&
        ((finalCardinalityKnown && hasLiteralLength &&
          literalLengthOperators && literalResample >= 0 && literalResample != 1) ||
         literalRbf);
    graphMemory.memoryAvailable = std::all_of(tasks.begin(), tasks.end(),
        [](UsdGenExecutionTaskMetadata const& task) {
            return task.estimate.memoryAvailable;
        });
    if (graphMemory.memoryAvailable) {
        // Do not sum producer-retention per task: Width branches all refer to
        // the same source value (and descendants refer to a value already in
        // the CoW output set). Instead model the actual execution phases.
        auto const& sourceEstimate = tasks.front().estimate;
        uint64_t persistent = 0, sourcePhase = 0, operatorScratch = 0;
        if (!EstimateSum({sourceEstimate.steadyBytes,
                          sourceEstimate.retainedOutputBytes,
                          sourceEstimate.producerRetentionBytes}, &persistent) ||
            !EstimateAdd(&sourcePhase, persistent) ||
            !EstimateAdd(&sourcePhase, sourceEstimate.scratchPeakBytes)) {
            Fail(diagnostics, "CUDA memory estimate overflow for CurveSource phase");
            return {};
        }
        for (auto const& task : tasks) {
            if (task.kind != UsdGenExecutionTaskKind::Operator) continue;
            // Every Width output owner stays live through terminal selection;
            // on the DAG they can all be live at once.  Producer-retention is
            // intentionally excluded because these identities are deduped.
            if (!EstimateAdd(&persistent, task.estimate.steadyBytes) ||
                !EstimateAdd(&persistent, task.estimate.retainedOutputBytes) ||
                !EstimateAdd(&operatorScratch, task.estimate.scratchPeakBytes)) {
                Fail(diagnostics, "CUDA memory estimate overflow for operator phase");
                return {};
            }
        }
        uint64_t operatorPhase = persistent;
        if (!EstimateAdd(&operatorPhase, operatorScratch)) {
            Fail(diagnostics, "CUDA memory estimate overflow for operator scratch phase");
            return {};
        }
        auto const& publicationEstimate = tasks.back().estimate;
        uint64_t publicationPhase = persistent;
        if (!EstimateSum({publicationEstimate.steadyBytes,
                          publicationEstimate.retainedOutputBytes,
                          publicationEstimate.scratchPeakBytes},
                         &operatorScratch) ||
            !EstimateAdd(&publicationPhase, operatorScratch)) {
            Fail(diagnostics, "CUDA memory estimate overflow for Publication phase");
            return {};
        }
        graphMemory.concurrentPeakBytes = std::max(sourcePhase,
            std::max(operatorPhase, publicationPhase));
        graphMemory.immutableSharedInputBytes = (layout.widthDag || layout.geometryValueDag)
            ? sourceEstimate.retainedOutputBytes : 0;
        graphMemory.conservativeUpperBound = true;
    }
    std::string dependencyReason;
    if (!UsdGenExecutionDependencyCompiler::Lower(
            &tasks, values, &dependencyReason) ||
        !UsdGenExecutionDependencyCompiler::Validate(
            tasks, values, &dependencyReason)) {
        Fail(diagnostics, "CUDA execution dependency lowering failed: " +
            dependencyReason);
        return {};
    }
    bool const hasValueFanIn = std::any_of(
        desc.nodes.begin(), desc.nodes.end(), [](UsdGenNodeDesc const& node) {
            return node.type == TfToken("UsdGenWidthBlend");
        });
    plan->metadata = std::make_shared<const UsdGenExecutionPlanMetadata>(
        "cuda", matrix.Version(), hasValueFanIn
            ? UsdGenExecutionPlanShape::SourceRootedValueDag
            : layout.taskDag ? UsdGenExecutionPlanShape::SourceRootedUnaryDag
                             : UsdGenExecutionPlanShape::LinearAuthoredChain,
        std::move(capabilities), std::move(tasks), std::move(values),
        terminalId, graphMemory);
    return plan;
}

std::shared_ptr<UsdGenCudaExecutionJob> CreateCudaExecutionJob(
    std::shared_ptr<const UsdGenCudaExecutionPlan> plan,
    UsdGenCudaExecutionWorkspace& workspace, double frame, uint64_t generation,
    UsdGenDiagnostics* diagnostics,
    std::shared_ptr<const UsdGenDeviceGeneration> previous) {
    if (!plan || !std::isfinite(frame)) return {};
#ifdef USDGEN_ENABLE_CUDA
    if (workspace.IsPoisoned()) {
        Fail(diagnostics, "CUDA execution workspace is poisoned");
        return {};
    }
    std::shared_ptr<UsdGenExecutionMemoryReservation> memoryReservation;
    UsdGenExecutionMemoryReservation scatterGrowReservation;
    auto const scatterGrowAdmission = ReserveScatterGrowExecution(
        *plan, workspace.impl_->device, workspace.impl_->stream, diagnostics,
        &scatterGrowReservation);
    if (scatterGrowAdmission == LiteralLengthAdmission::Failed) return {};
    if (scatterGrowAdmission == LiteralLengthAdmission::Admitted) {
        try {
            memoryReservation = std::make_shared<UsdGenExecutionMemoryReservation>(
                std::move(scatterGrowReservation));
        } catch (...) {
            Fail(diagnostics, "CUDA ScatterGrow memory reservation allocation failed");
            return {};
        }
    }
    UsdGenExecutionMemoryReservation literalLengthReservation;
    auto const literalLengthAdmission = ReserveLiteralLengthExecution(
        *plan, workspace.impl_->device, workspace.impl_->stream, diagnostics,
        &literalLengthReservation);
    if (literalLengthAdmission == LiteralLengthAdmission::Failed) return {};
    if (literalLengthAdmission == LiteralLengthAdmission::Admitted) {
        try {
            memoryReservation = std::make_shared<UsdGenExecutionMemoryReservation>(
                std::move(literalLengthReservation));
        } catch (...) {
            Fail(diagnostics, "CUDA literal-Length memory reservation allocation failed");
            return {};
        }
    } else {
        UsdGenExecutionMemoryReservation literalRbfReservation;
        auto const literalRbfAdmission = ReserveLiteralRbfExecution(
            *plan, workspace.impl_->device, diagnostics, &literalRbfReservation);
        if (literalRbfAdmission == LiteralRbfAdmission::Failed) return {};
        if (literalRbfAdmission == LiteralRbfAdmission::Admitted) {
            try {
                memoryReservation = std::make_shared<UsdGenExecutionMemoryReservation>(
                    std::move(literalRbfReservation));
            } catch (...) {
                Fail(diagnostics, "CUDA literal-RBF memory reservation allocation failed");
                return {};
            }
        }
    }
    if (!memoryReservation) if (auto const metadata = GetCudaExecutionPlanMetadata(*plan)) {
        auto const& estimate = metadata->MemoryEstimate();
        if (estimate.memoryAvailable && estimate.conservativeUpperBound) {
            if (estimate.concurrentPeakBytes >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                Fail(diagnostics, "CUDA execution job memory reservation size overflows this platform");
                return {};
            }
            CudaDeviceScope selected(workspace.impl_->device);
            if (!selected.selected) {
                Fail(diagnostics, "cannot select CUDA device for execution job memory reservation");
                return {};
            }
            auto reservation = gpu::TryReserveCudaExecutionMemory(
                static_cast<size_t>(estimate.concurrentPeakBytes));
            if (!reservation) {
                Fail(diagnostics,
                    "CUDA execution job memory reservation exceeds the available device budget");
                return {};
            }
            try {
                memoryReservation = std::make_shared<UsdGenExecutionMemoryReservation>(
                    std::move(*reservation));
            } catch (...) {
                Fail(diagnostics, "CUDA execution job memory reservation allocation failed");
                return {};
            }
        }
    }
#endif
    auto job = std::make_shared<UsdGenCudaExecutionJob>();
    job->plan = std::move(plan); job->workspace = &workspace; job->frame = frame;
    job->generation = generation; job->diagnostics = diagnostics;
    job->previous = std::move(previous);
#ifdef USDGEN_ENABLE_CUDA
    job->memoryReservation = std::move(memoryReservation);
    job->operatorStarted.assign(job->plan->steps.size(), 0);
    job->operatorFinished.assign(job->plan->steps.size(), 0);
#endif
    return job;
}

size_t CudaExecutionJobOperatorCount(UsdGenCudaExecutionJob const& job) noexcept {
#ifdef USDGEN_ENABLE_CUDA
    return job.plan ? job.plan->steps.size() : 0;
#else
    (void)job;
    return 0;
#endif
}

bool ExecuteCudaJobSource(UsdGenCudaExecutionJob& job) {
#ifdef USDGEN_ENABLE_CUDA
    if (job.finalAsyncInFlight || job.sourceAsyncInFlight ||
        job.operatorAsyncInFlight.load(std::memory_order_acquire)) return false;
    if (job.failed.load(std::memory_order_acquire) || job.sourceDone || !job.plan || !job.workspace) { job.failed = true; return false; }
    if (job.workspace->IsPoisoned()) {
        job.failed=true;
        return Fail(job.diagnostics, "CUDA execution workspace is poisoned");
    }
    if (job.plan->scatterGrow) {
        CudaDeviceScope selected(job.workspace->impl_->device);
        if (!selected.selected || !job.plan->scatterRoots) { job.failed=true; return false; }
        if (!job.state.PrepareScatterGrowWorkspace(*job.plan, *job.workspace,
                job.diagnostics, job.memoryReservation.get())) { job.failed=true; return false; }
        auto owner=std::make_unique<gpu::CudaScatterGrow>();
        auto const stream=job.workspace->impl_->stream;
        // The callback's userdata must outlive an unprovable stream failure.
        // Retain the callback state only when native proof is lost.  Clean
        // BeginFresh/Finish rejection has installed no live callback and must
        // not leak host state or clear an already-poisoned workspace.
        auto done=std::make_unique<CudaInlineCompletion>();
        auto status=owner->BeginFresh(job.plan->scatterRoots, job.plan->scatterGrowControls,
            stream, job.memoryReservation.get());
        if (status == gpu::ScatterGrowStatus::Ok &&
            !job.state.PrepareScatterGrowFrames(*job.plan, *job.workspace, stream,
                                                job.diagnostics))
            status = gpu::ScatterGrowStatus::CudaError;
        if (status == gpu::ScatterGrowStatus::Ok)
            status=owner->FinishFreshAsync(stream, CudaInlineCompletionCallback,done.get());
        if (status != gpu::ScatterGrowStatus::Ok || cudaStreamSynchronize(stream) != cudaSuccess ||
            done->status.load(std::memory_order_acquire) != int(cudaSuccess) ||
            owner->CommitFreshFinish() != gpu::ScatterGrowStatus::Ok) {
            if (owner->HasUnprovenWork()) {
                owner->MarkUnprovenWork();
                job.workspace->impl_->poisoned.store(true, std::memory_order_release);
                (void)done.release();
            }
            job.failed=true; return false;
        }
        job.state.memoryReservation=job.memoryReservation.get();
        job.state.scatterGrow=std::move(owner);
        job.state.geometry=job.state.scatterGrow->view();
        job.state.captureStableIds=job.state.geometry.stableIds;
        job.state.captureCurveCount=job.state.geometry.curveCount;
        job.state.hairT=job.state.scatterGrow->hairT();
        job.state.rootPrim=job.state.scatterGrow->rootPrim();
        job.state.rootUV=job.state.scatterGrow->rootUV();
        job.state.curveTopology={UsdGenDeviceCurveType::Cubic,UsdGenDeviceCurveBasis::BSpline,UsdGenDeviceCurveWrap::Pinned};
        job.state.prepared=true;
        job.state.RecordSourceValue(*job.plan);
        job.sourceDone=true; return true;
    }
    job.failed = true;
    if (!job.state.Prepare(*job.plan, *job.workspace, job.frame, job.diagnostics,
            true, nullptr, job.memoryReservation.get())) return false;
    job.state.RecordSourceValue(*job.plan);
    job.failed = false;
    job.sourceDone = true; return true;
#else
    job.failed = true; return false;
#endif
}

#ifdef USDGEN_ENABLE_CUDA
struct FinalizationReturnGate;
struct SourceAsyncTestGate {
    std::atomic<bool> claimed{false};
    std::atomic<bool> entered{false}, released{false}, timedOut{false}, terminal{false};
    std::atomic<std::atomic<unsigned>*> terminalState{nullptr};
};
std::shared_ptr<SourceAsyncTestGate> s_sourceAsyncTestGate;
std::shared_ptr<SourceAsyncTestGate> s_sourceAsyncResampleTestGate;
std::shared_ptr<SourceAsyncTestGate> s_sourceAsyncResampleReturnTestGate;
struct SourceScalarReturnGate {
    enum Handoff : unsigned { Ready = 1u, Released = 2u, Returned = 4u };
    std::atomic<bool> claimed{false};
    std::atomic<unsigned> handoff{0};
    std::atomic<bool> entered{false}, terminal{false};
    std::atomic<std::atomic<unsigned>*> terminalState{nullptr};
    std::atomic<void*> slot{nullptr};
};
std::shared_ptr<SourceScalarReturnGate> s_sourceAsyncScalarReturnTestGate;
std::array<std::shared_ptr<SourceAsyncTestGate>, 3> s_sourceAsyncControlGates;
void SourceAsyncGateCallback(cudaStream_t, cudaError_t, void* data) noexcept {
    auto* gate = static_cast<SourceAsyncTestGate*>(data);
    gate->entered.store(true, std::memory_order_release);
    // Native CUDA callbacks must not call flow-graph wait_for_all(): that can
    // steal and execute arbitrary application work on the CUDA callback
    // thread.  This is a test-only bounded yield gate.
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!gate->released.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            gate->timedOut.store(true, std::memory_order_release);
            return;
        }
        std::this_thread::yield();
    }
}
constexpr size_t kSourceRelayCapacity = 1024;
constexpr int kSourceRelayWorkers = 8;
constexpr unsigned kSourceRelayPhases = 6;
enum SourceRelayPhase : unsigned {
    SourceContexts, SourcePrograms, SourceScalars, SourceUpload,
    SourceResample, SourceNamedTopology
};
struct SourceRelayService;
void SourceCallback(cudaStream_t, cudaError_t, void*) noexcept;
void SourceResampleCallback(cudaStream_t, cudaError_t, void*) noexcept;
void ReturnSourceScalarWhenReleased(SourceScalarReturnGate&) noexcept;
struct SourceRelay {
    std::shared_ptr<UsdGenCudaExecutionJob> job;
    std::function<void(bool)> done;
    std::shared_ptr<SourceAsyncTestGate> gate;
    // Control gates are independently retained.  Keeping one relay-wide gate
    // is insufficient if more than one test seam is armed before this slot
    // reaches terminal proof.
    std::array<std::shared_ptr<SourceAsyncTestGate>, 3> controlGates;
    std::shared_ptr<SourceAsyncTestGate> resampleReturnGate;
    std::shared_ptr<SourceScalarReturnGate> scalarReturnGate;
    std::unique_ptr<CudaParameterEvaluator> controls;
    std::unique_ptr<CudaGroomScalarReadback> scalarPacket;
    CudaSourcePreparationOptions provenOptions;
    std::atomic<bool> quarantine{false};
    std::atomic<bool> failure{false};
    std::function<void()> poison;
    size_t slot = std::numeric_limits<size_t>::max();
};
// Slots are permanent callback identities. A CUDA callback is handed only a
// Slot::CallbackIdentity while the slot owns its relay; the worker cannot release that
// ownership before both terminal and launcher-return gates are open.
struct SourceRelayService {
    struct Slot {
        struct CallbackIdentity { Slot* slot = nullptr; unsigned phase = 0; };
        std::atomic<bool> occupied{false};
        std::shared_ptr<SourceRelay> relay;
        SourceRelayService* service = nullptr;
        size_t index = 0;
        std::array<std::atomic<unsigned>, kSourceRelayPhases> state{};
        std::array<std::atomic<int>, kSourceRelayPhases> status{};
        std::atomic<bool> quarantined{false};
        std::array<std::atomic<SourceAsyncTestGate*>, 3> controlGate{};
        std::atomic<SourceAsyncTestGate*> uploadGate{nullptr};
        std::atomic<SourceAsyncTestGate*> resampleGate{nullptr};
        std::atomic<SourceAsyncTestGate*> resampleReturnGate{nullptr};
        std::atomic<SourceScalarReturnGate*> scalarReturnGate{nullptr};
        std::array<CallbackIdentity, kSourceRelayPhases> callback{};
        void Signal(unsigned phase, cudaError_t value) noexcept;
        void ReturnFromLauncher(unsigned phase) noexcept;
    };
    std::array<Slot, kSourceRelayCapacity> slots;
    tbb::task_arena arena;
    std::unique_ptr<tbb::flow::graph> graph;
    std::unique_ptr<tbb::flow::function_node<size_t>> worker;
    std::atomic<size_t> capacity{kSourceRelayCapacity};
    std::array<std::atomic<bool>, kSourceRelayPhases> failInstall{}, failNative{};

    SourceRelayService() : arena(kSourceRelayWorkers, 0) {
        arena.execute([this] {
          graph = std::make_unique<tbb::flow::graph>();
          worker = std::make_unique<tbb::flow::function_node<size_t>>(*graph, tbb::flow::unlimited,
        [this](size_t index) {
            try { Complete(index); }
            catch (...) {
                auto const slot = index / kSourceRelayPhases;
                auto relay = slots[slot].relay;
                bool const unsafe = relay && (relay->quarantine.load(std::memory_order_acquire) ||
                    (relay->job && relay->job->state.HasUnprovenAsyncWork()) ||
                    (relay->controls && relay->controls->HasUnprovenWork()) ||
                    (relay->scalarPacket && relay->scalarPacket->HasUnprovenWork()));
                if (unsafe) Quarantine(slot); else AbortClean(slot);
            }
            return tbb::flow::continue_msg{};
        });
        });
        for (size_t i = 0; i != slots.size(); ++i) {
            slots[i].service = this;
            slots[i].index = i;
            for (unsigned phase=0; phase!=kSourceRelayPhases; ++phase)
                slots[i].callback[phase] = {&slots[i], phase};
        }
    }
    SourceRelay* Reserve(std::shared_ptr<UsdGenCudaExecutionJob> job,
                         std::function<void(bool)> done) noexcept {
        size_t const limit = std::min(capacity.load(std::memory_order_acquire), slots.size());
        for (size_t i = 0; i != limit; ++i) {
            bool expected = false;
            if (!slots[i].occupied.compare_exchange_strong(expected, true,
                    std::memory_order_acq_rel, std::memory_order_acquire)) continue;
            try {
                auto relay = std::make_shared<SourceRelay>();
                relay->job = std::move(job); relay->done = std::move(done);
                relay->slot = i;
                for (unsigned phase=0; phase!=kSourceRelayPhases; ++phase) {
                    slots[i].status[phase].store(int(cudaErrorUnknown), std::memory_order_relaxed);
                    slots[i].state[phase].store(0, std::memory_order_relaxed);
                }
                slots[i].quarantined.store(false, std::memory_order_relaxed);
                for (auto& gate : slots[i].controlGate)
                    gate.store(nullptr, std::memory_order_relaxed);
                slots[i].resampleGate.store(nullptr, std::memory_order_relaxed);
                slots[i].uploadGate.store(nullptr, std::memory_order_relaxed);
                slots[i].resampleReturnGate.store(nullptr, std::memory_order_relaxed);
                slots[i].scalarReturnGate.store(nullptr, std::memory_order_relaxed);
                slots[i].relay = std::move(relay);
                return slots[i].relay.get();
            } catch (...) {
                slots[i].occupied.store(false, std::memory_order_release);
                return nullptr;
            }
        }
        return nullptr;
    }
    void Post(size_t index, unsigned phase = 0) noexcept {
        try { if (!worker->try_put(index * kSourceRelayPhases + phase)) std::terminate(); }
        catch (...) { std::terminate(); }
    }
    void Complete(size_t message) {
        size_t const index = message / kSourceRelayPhases; unsigned const phase = unsigned(message % kSourceRelayPhases);
        auto relay = slots[index].relay;
        if (!relay) return;
        auto job = relay->job;
        if (phase <= SourceScalars) {
            bool phaseOk = !relay->failure.load(std::memory_order_acquire) &&
                !relay->quarantine.load(std::memory_order_acquire) && job && !job->failed.load(std::memory_order_acquire) &&
                job->workspace && cudaError_t(slots[index].status[phase].load(std::memory_order_acquire)) == cudaSuccess;
            if (phaseOk) {
                CudaDeviceScope selected(job->workspace->impl_->device);
                phaseOk = selected.selected;
                if (!phaseOk)
                    Fail(job->diagnostics, "cannot select CUDA device for source control phase " +
                        std::to_string(phase) + ": " + selected.FailureDetail());
                auto const stream = phaseOk ? job->workspace->impl_->stream : nullptr;
                if (phaseOk && phase == SourceContexts)
                    phaseOk = relay->controls &&
                        relay->controls->CommitFreshContexts() == CudaParameterStatus::Ok &&
                        relay->controls->BeginFreshPrograms(stream,
                            job->memoryReservation.get(),
                            UsdGenExecutionResourceKind::Scratch) == CudaParameterStatus::Ok &&
                        relay->controls->EnqueueFreshProgramStatus(stream) == CudaParameterStatus::Ok;
                else if (phaseOk && phase == SourcePrograms) {
                    phaseOk = relay->controls && relay->controls->CommitFreshPrograms() == CudaParameterStatus::Ok;
                    if (phaseOk) {
                        relay->scalarPacket = std::make_unique<CudaGroomScalarReadback>();
                        std::vector<CudaGroomScalarRequest> requests{{TfToken("useRest"),expr::ScalarType::Bool},
                                                                      {TfToken("resampleTo"),expr::ScalarType::Int32}};
                        phaseOk = relay->scalarPacket->BeginFresh(
                            relay->controls->Fields(), requests, stream,
                            job->memoryReservation.get()) == CudaParameterStatus::Ok;
                    }
                } else if (phaseOk && phase == SourceScalars) {
                    phaseOk = relay->scalarPacket && relay->scalarPacket->CommitFreshFinish() == CudaParameterStatus::Ok;
                    if (phaseOk) {
                        auto const& node=job->plan->desc.nodes[job->plan->sourceNode]; UsdGenParamView params; params.desc=&job->plan->desc; params.node=&node;
                        relay->provenOptions.useRest=params.GetBool(TfToken("useRest"),true);
                        relay->provenOptions.resampleTo=params.GetInt(TfToken("resampleTo"),0);
                        if (auto const* value=relay->scalarPacket->Find(TfToken("useRest"))) {
                            uint8_t raw=0; std::memcpy(&raw,value->bytes.data(),sizeof(raw));
                            phaseOk = raw <= 1; relay->provenOptions.useRest = raw != 0;
                        }
                        if (phaseOk) if (auto const* value=relay->scalarPacket->Find(TfToken("resampleTo"))) {
                            int raw=0; std::memcpy(&raw,value->bytes.data(),sizeof(raw));
                            phaseOk = raw >= 0 && raw != 1; relay->provenOptions.resampleTo = raw;
                        }
                        if (phaseOk)
                            phaseOk = job->state.Prepare(*job->plan, *job->workspace,
                                job->frame, job->diagnostics, false,
                                &relay->provenOptions,
                                job->memoryReservation.get());
                    }
                }
                selected.Restore();
            }
            if (!phaseOk) { relay->failure.store(true,std::memory_order_release); slots[index].Signal(phase,cudaSuccess); }
            else {
                CudaDeviceScope launcherDevice(job->workspace->impl_->device);
                if (!launcherDevice.selected) {
                    Fail(job->diagnostics, "cannot select CUDA device for source callback installation: " +
                        launcherDevice.FailureDetail());
                    relay->quarantine.store(true,std::memory_order_release);
                    relay->failure.store(true, std::memory_order_release);
                } else {
                unsigned const next=phase+1;
                auto const stream=job->workspace->impl_->stream;
                if (next == SourceUpload) if (auto gate=std::atomic_load(&s_sourceAsyncTestGate);
                    gate && !gate->claimed.exchange(true)) {
                    relay->gate=gate;
                    if (cudaStreamAddCallback(stream,SourceAsyncGateCallback,gate.get(),0)!=cudaSuccess) {
                        relay->quarantine.store(true,std::memory_order_release);
                        slots[index].Signal(next,cudaErrorUnknown);
                        slots[index].ReturnFromLauncher(next);
                        return;
                    }
                }
                if (next <= SourceScalars) if (auto gate=std::atomic_load(&s_sourceAsyncControlGates[next]);
                    gate && !gate->claimed.exchange(true)) {
                    // FreshAsync has already queued its compact proof D2H.
                    // Keep the test hold in the terminal callback, immediately
                    // before Signal, so callback installation itself never
                    // blocks behind the held predecessor callback.
                    relay->controlGates[next]=gate;
                    slots[index].controlGate[next].store(gate.get(), std::memory_order_release);
                }
                if (next == SourceScalars) if (auto gate = std::atomic_load(&s_sourceAsyncScalarReturnTestGate);
                    gate && !gate->claimed.exchange(true, std::memory_order_acq_rel)) {
                    relay->scalarReturnGate = gate;
                    gate->slot.store(&slots[index], std::memory_order_release);
                    slots[index].scalarReturnGate.store(gate.get(), std::memory_order_release);
                }
                bool installFailed=failInstall[next].exchange(false,std::memory_order_acq_rel);
                if (!installFailed && next == SourceUpload) {
                    auto const status = job->state.source->FinishFreshAsync(stream,SourceCallback,
                        &slots[index].callback[SourceUpload]);
                    installFailed = status != gpu::CurveSourceStatus::Ok;
                    if (installFailed)
                        Fail(job->diagnostics, "source async callback installation failed: CurveSourceStatus " +
                            CurveSourceStatusText(status));
                } else if (!installFailed) {
                    auto const status = cudaStreamAddCallback(stream,SourceCallback,&slots[index].callback[next],0);
                    installFailed = status != cudaSuccess;
                    if (installFailed)
                        Fail(job->diagnostics, "source control callback installation failed: " + CudaStatusText(status));
                }
                if (installFailed) {
                    relay->quarantine.store(true,std::memory_order_release);
                    slots[index].Signal(next,cudaErrorUnknown);
                }
                launcherDevice.Restore();
                if (next == SourceScalars && relay->scalarReturnGate) {
                    // Ready means this launcher has restored its device and
                    // made its final payload/slot access; only then may a
                    // released test handoff cause the slot to recycle.
                    relay->scalarReturnGate->entered.store(true, std::memory_order_release);
                    relay->scalarReturnGate->handoff.fetch_or(SourceScalarReturnGate::Ready,
                        std::memory_order_acq_rel);
                    ReturnSourceScalarWhenReleased(*relay->scalarReturnGate);
                } else {
                    slots[index].ReturnFromLauncher(next);
                }
                return;
                }
            }
        }
        bool controlGatesOK = true;
        for (auto const& gate : relay->controlGates)
            controlGatesOK = controlGatesOK && (!gate || !gate->timedOut.load(std::memory_order_acquire));
        bool ok = !relay->failure.load(std::memory_order_acquire) &&
            !relay->quarantine.load(std::memory_order_acquire) && (!relay->gate ||
                !relay->gate->timedOut.load(std::memory_order_acquire)) &&
            controlGatesOK &&
            (!relay->resampleReturnGate ||
                !relay->resampleReturnGate->timedOut.load(std::memory_order_acquire)) &&
            cudaError_t(slots[index].status[phase].load(std::memory_order_acquire)) == cudaSuccess &&
            job && !job->failed.load(std::memory_order_acquire) && job->workspace &&
            (phase == SourceUpload ? (job->plan->scatterGrow
                ? job->scatterGrow && job->scatterGrow->CommitFreshFinish() == gpu::ScatterGrowStatus::Ok
                : job->state.CommitSourceUpload(*job->workspace, job->diagnostics)) :
             phase == SourceResample ? job->state.CompleteSourceResample(*job->workspace, job->diagnostics, job) :
             phase == SourceNamedTopology ? job->state.CompleteSourceNamedTopology(
                 cudaError_t(slots[index].status[phase].load(std::memory_order_acquire)),
                 *job->workspace, job->diagnostics) : true);
        if (ok && phase == SourceUpload && job->state.sourceOptions.resampleTo) {
            auto gate = std::atomic_load(&s_sourceAsyncResampleTestGate);
            bool useGate = gate && !gate->claimed.exchange(true);
            if (useGate) {
                // The callback identity is permanent and relay ownership
                // retains this raw test-gate pointer until terminal proof.
                // Holding it in SourceResampleCallback places the test seam
                // after FreshAsync's status D2H, not before its protocol.
                relay->gate = gate;
                slots[index].resampleGate.store(gate.get(), std::memory_order_release);
            }
            // ApplyFresh and callback install are one worker-owned sequence;
            // the phase return below is its final payload-visible operation.
            bool const injectInstall = failInstall[SourceResample].exchange(false, std::memory_order_acq_rel);
            bool resampleSubmitted = false;
            auto returnGate = std::atomic_load(&s_sourceAsyncResampleReturnTestGate);
            if (returnGate && !returnGate->claimed.exchange(true)) {
                relay->resampleReturnGate = returnGate;
                returnGate->terminalState.store(&slots[index].state[SourceResample], std::memory_order_release);
                slots[index].resampleReturnGate.store(returnGate.get(), std::memory_order_release);
            }
            if (!job->state.StartSourceResampleAsync(*job->workspace,
                    nullptr, nullptr,
                    SourceResampleCallback,
                    &slots[index].callback[SourceResample], injectInstall, &resampleSubmitted)) {
                relay->failure.store(true, std::memory_order_release);
                relay->quarantine.store(resampleSubmitted || job->state.HasUnprovenAsyncWork(), std::memory_order_release);
                slots[index].Signal(SourceResample, cudaSuccess);
            }
            if (relay->resampleReturnGate) {
                auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                relay->resampleReturnGate->entered.store(true, std::memory_order_release);
                while (!relay->resampleReturnGate->released.load(std::memory_order_acquire)) {
                    if (std::chrono::steady_clock::now() >= deadline) {
                        relay->resampleReturnGate->timedOut.store(true, std::memory_order_release);
                        break;
                    }
                    std::this_thread::yield();
                }
            }
            slots[index].ReturnFromLauncher(SourceResample);
            return;
        }
        if (ok && phase == SourceUpload && job->plan->scatterGrow) {
            // Source callback proof is complete and launcher-return gated.
            // Move the fresh immutable topology into the finalization state;
            // it is never reinterpreted as a CurveSource upload.
            job->state.scatterGrow = std::move(job->scatterGrow);
            if (!job->state.scatterGrow) ok = false;
            else {
                job->state.geometry = job->state.scatterGrow->view();
                job->state.captureStableIds = job->state.geometry.stableIds;
                job->state.captureCurveCount = job->state.geometry.curveCount;
                job->state.hairT = job->state.scatterGrow->hairT();
                job->state.rootPrim = job->state.scatterGrow->rootPrim();
                job->state.rootUV = job->state.scatterGrow->rootUV();
                job->state.curveTopology = {UsdGenDeviceCurveType::Cubic,
                    UsdGenDeviceCurveBasis::BSpline, UsdGenDeviceCurveWrap::Pinned};
                job->state.prepared = true;
                job->state.RecordSourceValue(*job->plan);
            }
        }
        if (ok && phase == SourceResample && job->state.sourceNamedTopology) {
            bool const injectInstall = failInstall[SourceNamedTopology].exchange(
                false, std::memory_order_acq_rel);
            bool submitted = false;
            if (!job->state.StartSourceNamedTopologyAsync(*job->workspace,
                    SourceCallback, &slots[index].callback[SourceNamedTopology],
                    injectInstall, &submitted)) {
                relay->failure.store(true, std::memory_order_release);
                relay->quarantine.store(submitted || job->state.HasUnprovenAsyncWork(),
                                        std::memory_order_release);
                slots[index].Signal(SourceNamedTopology, cudaSuccess);
            }
            slots[index].ReturnFromLauncher(SourceNamedTopology);
            return;
        }
        if (!ok) {
            bool const jobWasFailed = job && job->failed.load(std::memory_order_acquire);
            if (job) { job->failed = true; job->sourceAsyncInFlight = false; }
            if (job && job->diagnostics) {
                static char const* const phaseNames[] = {
                    "contexts", "programs", "scalars", "upload", "resample", "named topology"};
                auto const nativeStatus = cudaError_t(slots[index].status[phase].load(std::memory_order_acquire));
                Fail(job->diagnostics, "source async upload failed; previous generation retained (phase=" +
                    std::string(phaseNames[phase]) + ", callback=" + CudaStatusText(nativeStatus) +
                    ", relayFailure=" + std::to_string(relay->failure.load(std::memory_order_acquire)) +
                    ", quarantine=" + std::to_string(relay->quarantine.load(std::memory_order_acquire)) +
                    ", jobFailed=" + std::to_string(jobWasFailed) +
                    ", workspacePoisoned=" + std::to_string(job->workspace && job->workspace->impl_->poisoned.load(std::memory_order_acquire)) +
                    ", sourceUploadPending=" + std::to_string(job->state.sourceUploadPending) +
                    ", sourcePresent=" + std::to_string(bool(job->state.source)) +
                    ", sourceUnproven=" + std::to_string(job->state.source && job->state.source->HasUnprovenUpload()) +
                    ", controlGateTimeout=" + std::to_string(!controlGatesOK) +
                    ", gateTimeout=" + std::to_string(relay->gate && relay->gate->timedOut.load(std::memory_order_acquire)) +
                    ", resampleReturnTimeout=" + std::to_string(relay->resampleReturnGate && relay->resampleReturnGate->timedOut.load(std::memory_order_acquire)) + ")");
            }
            // This lambda is invoked before user code and is created in the
            // launcher, where private workspace access is valid.
            bool const unsafe = relay->quarantine.load(std::memory_order_acquire) ||
                cudaError_t(slots[index].status[phase].load(std::memory_order_acquire)) != cudaSuccess ||
                (job && job->state.HasUnprovenAsyncWork()) ||
                (job && job->scatterGrow && job->scatterGrow->HasUnprovenWork()) ||
                (relay->controls && relay->controls->HasUnprovenWork()) ||
                (relay->scalarPacket && relay->scalarPacket->HasUnprovenWork());
            relay->quarantine.store(unsafe, std::memory_order_release);
            if (unsafe) {
                if (job) job->CloseMemoryReservation();
                slots[index].quarantined.store(true, std::memory_order_release);
                if (relay->poison) relay->poison();
            }
            relay->poison = {};
        } else {
            if (!job->plan->scatterGrow) job->state.RecordSourceValue(*job->plan);
            job->sourceDone = true;
            job->sourceAsyncInFlight = false;
        }
        auto done = std::move(relay->done);
        if (relay->quarantine.load(std::memory_order_acquire)) {
            if (job) { job->sourceAsyncInFlight = false; job->workspace = nullptr; job->diagnostics = nullptr; }
            if (done) { try { done(false); } catch (...) {} }
            // Never retain the caller's dispatcher closure with poison.
            relay->done = {};
            // Keep this payload in its fixed admission slot forever.  This is
            // bounded by kSourceRelayCapacity and has no cross-worker vector
            // mutation or teardown race.
            return;
        }
        // All normal terminal paths have native proof. Tear down CUDA-owning
        // control candidates under their workspace device before user code can
        // destroy the borrowed workspace/stream.
        if (job && job->workspace) {
            CudaDeviceScope cleanupDevice(job->workspace->impl_->device);
            if (cleanupDevice.selected) {
                relay->scalarPacket.reset();
                relay->controls.reset();
            } else {
                relay->quarantine.store(true, std::memory_order_release);
                slots[index].quarantined.store(true, std::memory_order_release);
                job->failed = true;
                job->sourceAsyncInFlight = false;
                job->CloseMemoryReservation();
                if (relay->poison) relay->poison();
                else job->workspace->impl_->poisoned.store(true, std::memory_order_release);
                relay->poison = {};
                job->workspace = nullptr;
                job->diagnostics = nullptr;
                relay->done = {};
                if (done) { try { done(false); } catch (...) {} }
                return;
            }
        }
        if (!ok && job) job->CloseMemoryReservation();
        relay->job.reset();
        slots[index].relay.reset();
        slots[index].occupied.store(false, std::memory_order_release);
        // User completion may release the final external job/workspace owner.
        // Do not keep candidate CUDA owners alive across that call.
        relay.reset();
        job.reset();
        if (done) { try { done(ok); } catch (...) {} }
    }
    void Quarantine(size_t index) noexcept {
        auto relay = slots[index].relay;
        if (!relay) return;
        relay->quarantine.store(true, std::memory_order_release);
        slots[index].quarantined.store(true, std::memory_order_release);
        if (relay->job) { relay->job->failed = true; relay->job->sourceAsyncInFlight = false; }
        if (relay->job) relay->job->CloseMemoryReservation();
        if (relay->poison) relay->poison();
        relay->poison = {};
        if (relay->job) { relay->job->workspace = nullptr; relay->job->diagnostics = nullptr; }
        auto done = std::move(relay->done);
        relay->done = {};
        if (done) { try { done(false); } catch (...) {} }
        // Permanent fixed-slot quarantine; see Complete().
    }
    void AbortClean(size_t index) noexcept {
        auto relay = slots[index].relay;
        if (!relay) return;
        auto job = relay->job;
        if (job) { job->failed = true; job->sourceAsyncInFlight = false; }
        if (job && job->workspace) {
            CudaDeviceScope selected(job->workspace->impl_->device);
            if (!selected.selected) { Quarantine(index); return; }
            relay->scalarPacket.reset();
            relay->controls.reset();
        }
        auto done = std::move(relay->done);
        relay->done = {};
        relay->job.reset();
        slots[index].relay.reset();
        slots[index].occupied.store(false, std::memory_order_release);
        relay.reset(); job.reset();
        if (done) { try { done(false); } catch (...) {} }
    }
};
// The service is intentionally process-lifetime.  CUDA may invoke a native
// callback while static session/scene teardown is under way.
SourceRelayService& SourceRelays() { static auto* service = new SourceRelayService; return *service; }
void SourceRelayService::Slot::Signal(unsigned phase, cudaError_t value) noexcept {
    // Publish status before terminal.  The arrival that observes both gates
    // owns posting; no relay is touched by the CUDA callback.
    if (state[phase].fetch_or(8u, std::memory_order_acq_rel) & 8u) return;
    status[phase].store(int(value), std::memory_order_release);
    unsigned const old = state[phase].fetch_or(1u, std::memory_order_acq_rel);
    if ((old & 2u) && !(old & 4u) && !(state[phase].fetch_or(4u, std::memory_order_acq_rel) & 4u))
        service->Post(index, phase);
}
void SourceRelayService::Slot::ReturnFromLauncher(unsigned phase) noexcept {
    unsigned const old = state[phase].fetch_or(2u, std::memory_order_acq_rel);
    if ((old & 1u) && !(old & 4u) && !(state[phase].fetch_or(4u, std::memory_order_acq_rel) & 4u))
        service->Post(index, phase);
}
void ReturnSourceScalarWhenReleased(SourceScalarReturnGate& gate) noexcept {
    unsigned state = gate.handoff.load(std::memory_order_acquire);
    for (;;) {
        if ((state & (SourceScalarReturnGate::Ready | SourceScalarReturnGate::Released)) !=
                (SourceScalarReturnGate::Ready | SourceScalarReturnGate::Released) ||
            (state & SourceScalarReturnGate::Returned)) return;
        if (gate.handoff.compare_exchange_weak(state, state | SourceScalarReturnGate::Returned,
                std::memory_order_acq_rel, std::memory_order_acquire)) break;
    }
    auto* slot = static_cast<SourceRelayService::Slot*>(gate.slot.load(std::memory_order_acquire));
    if (slot && slot->scalarReturnGate.load(std::memory_order_acquire) == &gate)
        slot->ReturnFromLauncher(SourceScalars);
}
void SourceCallback(cudaStream_t, cudaError_t status, void* data) noexcept {
    auto* identity = static_cast<SourceRelayService::Slot::CallbackIdentity*>(data);
    auto* slot = identity->slot;
    if (slot->service->failNative[identity->phase].exchange(false, std::memory_order_acq_rel))
        status = cudaErrorUnknown;
    if (identity->phase == SourceScalars)
        if (auto* gate=slot->scalarReturnGate.load(std::memory_order_acquire)) {
            gate->terminalState.store(&slot->state[SourceScalars],std::memory_order_release);
            gate->terminal.store(true,std::memory_order_release);
        }
    // Fresh phase status copies precede this callback.  Test-only holds live
    // here rather than as predecessor callbacks so stream callback insertion
    // cannot be serialized behind a hold before the launcher returns.
    if (identity->phase <= SourceScalars)
        if (auto* gate=slot->controlGate[identity->phase].load(std::memory_order_acquire))
            SourceAsyncGateCallback(nullptr, status, gate);
    if (identity->phase == SourceUpload)
        if (auto* gate=slot->uploadGate.load(std::memory_order_acquire))
            SourceAsyncGateCallback(nullptr, status, gate);
    // Signal can post a worker which recycles this slot: it is the callback's
    // final slot access.
    slot->Signal(identity->phase, status);
}
void SourceResampleCallback(cudaStream_t, cudaError_t status, void* data) noexcept {
    auto* identity = static_cast<SourceRelayService::Slot::CallbackIdentity*>(data);
    auto* slot = identity->slot;
    if (slot->service->failNative[SourceResample].exchange(false, std::memory_order_acq_rel))
        status = cudaErrorUnknown;
    if (auto* gate = slot->resampleReturnGate.load(std::memory_order_acquire))
        gate->terminal.store(true, std::memory_order_release);
    // Test-only: FreshAsync queued its status D2H before this callback.  Do
    // not add a predecessor callback, which can interfere with unrelated
    // runtime scheduling while it holds the stream.  The relay owns the gate
    // and this is immediately before Signal, the final slot access.
    if (auto* gate = slot->resampleGate.load(std::memory_order_acquire))
        SourceAsyncGateCallback(nullptr, status, gate);
    // SignalResample may post a worker which can recycle this slot.  This is
    // deliberately the callback's final slot access.
    slot->Signal(SourceResample, status);
}
#endif

bool ExecuteCudaJobSourceAsync(std::shared_ptr<UsdGenCudaExecutionJob> job,
    std::function<void(bool)> completion) {
#ifndef USDGEN_ENABLE_CUDA
    (void)job; (void)completion; return false;
#else
    if (job && (job->finalAsyncInFlight || job->sourceAsyncInFlight ||
                job->operatorAsyncInFlight.load(std::memory_order_acquire))) return false;
    if (!job || !completion || job->failed.load(std::memory_order_acquire) || job->sourceDone || !job->plan || !job->workspace ||
        job->workspace->impl_->poisoned.load(std::memory_order_acquire)) return false;
    // Capture is rejected before expression evaluation or Set can enqueue any
    // work.  FinishFreshAsync repeats this check as its low-level boundary.
    CudaDeviceScope selected(job->workspace->impl_->device);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (!selected.selected)
        return Fail(job->diagnostics, "cannot select CUDA device for source async launch: " + selected.FailureDetail());
    auto const captureStatus = cudaStreamIsCapturing(job->workspace->impl_->stream, &capture);
    if (captureStatus != cudaSuccess)
        return Fail(job->diagnostics, "cannot query CUDA source stream capture: " + CudaStatusText(captureStatus));
    if (capture != cudaStreamCaptureStatusNone)
        return Fail(job->diagnostics, "CUDA source async launch rejected stream capture status " + std::to_string(static_cast<int>(capture)));
    auto* relay = SourceRelays().Reserve(job, std::move(completion));
    if (!relay) return false;
    size_t const relaySlot = relay->slot;
    job->sourceAsyncInFlight = true;
    if (job->plan->scatterGrow) {
        bool submitted = false;
        try {
            // Match CurveSource: a relay that loses any native completion
            // proof poisons this workspace before its permanent quarantine.
            // The closure is created while workspace access is valid and is
            // never invoked by a native callback.
            relay->poison = [workspace = job->workspace] {
                workspace->impl_->poisoned.store(true, std::memory_order_release);
            };
            if (!job->state.PrepareScatterGrowWorkspace(*job->plan, *job->workspace,
                    job->diagnostics, job->memoryReservation.get())) {
                job->sourceAsyncInFlight = false;
                SourceRelays().slots[relaySlot].relay.reset();
                SourceRelays().slots[relaySlot].occupied.store(false, std::memory_order_release);
                return false;
            }
            if (!job->plan->scatterRoots) {
                job->sourceAsyncInFlight = false;
                SourceRelays().slots[relaySlot].relay.reset();
                SourceRelays().slots[relaySlot].occupied.store(false, std::memory_order_release);
                return false;
            }
            job->scatterGrow=std::make_unique<gpu::CudaScatterGrow>();
            job->state.memoryReservation=job->memoryReservation.get();
            auto const stream=job->workspace->impl_->stream;
            auto status=job->scatterGrow->BeginFresh(job->plan->scatterRoots,
                job->plan->scatterGrowControls,stream,job->memoryReservation.get());
            if (status == gpu::ScatterGrowStatus::Ok &&
                !job->state.PrepareScatterGrowFrames(*job->plan, *job->workspace, stream,
                                                     job->diagnostics))
                status = gpu::ScatterGrowStatus::CudaError;
            if (status != gpu::ScatterGrowStatus::Ok) {
                // BeginFresh is the no-work admission boundary.  Its clean
                // rejection is not an accepted async operation and must not
                // consume a fixed relay/quarantine slot.
                if (!job->scatterGrow->HasUnprovenWork()) {
                    job->scatterGrow.reset();
                    job->state.memoryReservation=nullptr;
                    job->sourceAsyncInFlight = false;
                    SourceRelays().slots[relaySlot].relay.reset();
                    SourceRelays().slots[relaySlot].occupied.store(false, std::memory_order_release);
                    return false;
                }
                relay->quarantine.store(true, std::memory_order_release);
                SourceRelays().slots[relaySlot].Signal(SourceUpload,cudaErrorUnknown);
                selected.Restore();
                SourceRelays().slots[relaySlot].ReturnFromLauncher(SourceUpload);
                return true;
            }
            submitted = true;
            if (status == gpu::ScatterGrowStatus::Ok) {
                if (auto gate=std::atomic_load(&s_sourceAsyncTestGate);
                    gate && !gate->claimed.exchange(true)) {
                    relay->gate=gate;
                    // Hold the terminal callback after the status D2H. A
                    // predecessor hold can block that copy's submission
                    // under instrumentation before this launcher returns.
                    SourceRelays().slots[relaySlot].uploadGate.store(
                        gate.get(), std::memory_order_release);
                }
                bool const injectInstall = SourceRelays().failInstall[SourceUpload].exchange(
                    false, std::memory_order_acq_rel);
                if (!injectInstall)
                    status=job->scatterGrow->FinishFreshAsync(stream,SourceCallback,
                        &SourceRelays().slots[relaySlot].callback[SourceUpload]);
                if (injectInstall) status = gpu::ScatterGrowStatus::InvalidArgument;
            }
            if (status != gpu::ScatterGrowStatus::Ok) {
                // BeginFresh has submitted H2D roots.  A missing terminal
                // callback is therefore proof loss even if a lower-level
                // status has not yet reflected it.
                relay->quarantine.store(true,std::memory_order_release);
                SourceRelays().slots[relaySlot].Signal(SourceUpload,cudaErrorUnknown);
            }
            selected.Restore(); SourceRelays().slots[relaySlot].ReturnFromLauncher(SourceUpload); return true;
        } catch (...) {
            if (!submitted && (!job->scatterGrow || !job->scatterGrow->HasUnprovenWork())) {
                job->scatterGrow.reset();
                job->state.memoryReservation=nullptr;
                job->sourceAsyncInFlight = false;
                SourceRelays().slots[relaySlot].relay.reset();
                SourceRelays().slots[relaySlot].occupied.store(false, std::memory_order_release);
                return false;
            }
            relay->quarantine.store(true,std::memory_order_release);
            SourceRelays().slots[relaySlot].Signal(SourceUpload,cudaErrorUnknown);
            selected.Restore(); SourceRelays().slots[relaySlot].ReturnFromLauncher(SourceUpload); return true;
        }
    }
    if (job->plan->sourceParameters) {
        try {
            relay->poison = [workspace = job->workspace] {
                workspace->impl_->poisoned.store(true, std::memory_order_release);
            };
            relay->controls = std::make_unique<CudaParameterEvaluator>();
            expr::Context context; context.frame=job->frame;
            context.time=job->frame/job->plan->desc.timeCodesPerSecond;
            auto const& node=job->plan->desc.nodes[job->plan->sourceNode];
            context.seed=node.seed; context.descId=expr::DescriptionId(job->plan->desc.description.GetText());
            auto const stream=job->workspace->impl_->stream;
            job->failed=false;
            bool controlsReady=relay->controls->BeginFreshContexts(
                    job->plan->sourceParameters, {}, {}, context, stream, nullptr,
                    job->memoryReservation.get(),
                    UsdGenExecutionResourceKind::Scratch) == CudaParameterStatus::Ok &&
                relay->controls->EnqueueFreshContextStatus(stream) == CudaParameterStatus::Ok;
            if (controlsReady) if (auto gate=std::atomic_load(&s_sourceAsyncControlGates[SourceContexts]);
                gate && !gate->claimed.exchange(true)) {
                relay->controlGates[SourceContexts]=gate;
                SourceRelays().slots[relaySlot].controlGate[SourceContexts].store(
                    gate.get(), std::memory_order_release);
            }
            bool const terminalInstallFailed = controlsReady &&
                (SourceRelays().failInstall[SourceContexts].exchange(false,std::memory_order_acq_rel) ||
                 cudaStreamAddCallback(stream,SourceCallback,&SourceRelays().slots[relaySlot].callback[SourceContexts],0)!=cudaSuccess);
            if (!controlsReady || terminalInstallFailed) {
                relay->failure.store(true,std::memory_order_release);
                if (terminalInstallFailed || relay->controls->HasUnprovenWork())
                    relay->quarantine.store(true,std::memory_order_release);
                SourceRelays().slots[relaySlot].Signal(SourceContexts,cudaSuccess);
            }
            // The launcher-return join can wake a worker.  Restore before it
            // does, matching each worker-side phase launcher below.
            selected.Restore();
            SourceRelays().slots[relaySlot].ReturnFromLauncher(SourceContexts);
            return true;
        } catch (...) {
            relay->failure.store(true,std::memory_order_release);
            if (relay->controls && relay->controls->HasUnprovenWork()) relay->quarantine.store(true,std::memory_order_release);
            SourceRelays().slots[relaySlot].Signal(SourceContexts,cudaSuccess);
            selected.Restore();
            SourceRelays().slots[relaySlot].ReturnFromLauncher(SourceContexts);
            return true;
        }
    }
    job->failed = true;
    bool prepared = false;
    try {
        relay->poison = [workspace = job->workspace] {
            workspace->impl_->poisoned.store(true, std::memory_order_release);
        };
        prepared = job->state.Prepare(*job->plan, *job->workspace, job->frame,
            job->diagnostics, false, nullptr,
            job->memoryReservation.get());
    } catch (...) {
        if (job->state.source && (job->state.source->HasUnprovenUpload() ||
                                  job->state.source->pending())) {
            relay->quarantine.store(true, std::memory_order_release);
            SourceRelays().slots[relaySlot].Signal(SourceUpload, cudaErrorUnknown);
            SourceRelays().slots[relaySlot].ReturnFromLauncher(SourceUpload);
            return true;
        }
        job->sourceAsyncInFlight = false;
        SourceRelays().slots[relaySlot].relay.reset();
        SourceRelays().slots[relaySlot].occupied.store(false, std::memory_order_release);
        return false;
    }
    if (!prepared) {
        if (job->state.source && job->state.source->HasUnprovenUpload()) {
            relay->quarantine.store(true, std::memory_order_release); SourceRelays().slots[relaySlot].Signal(SourceUpload, cudaErrorUnknown);
            SourceRelays().slots[relaySlot].ReturnFromLauncher(SourceUpload); return true;
        }
        job->sourceAsyncInFlight = false;
        SourceRelays().slots[relaySlot].relay.reset();
        SourceRelays().slots[relaySlot].occupied.store(false, std::memory_order_release);
        return false;
    }
    // The relay may run before FinishFreshAsync returns; publish the pending
    // state before installing any native callback.
    job->failed = false;
    auto stream = job->workspace->impl_->stream;
    if (auto gate = std::atomic_load(&s_sourceAsyncTestGate);
        gate && !gate->claimed.exchange(true)) {
        // This stream-ordered test callback is intentionally after Set's H2D
        // submissions and before the real terminal callback; launcher returns.
        relay->gate = gate;
        if (cudaStreamAddCallback(stream, SourceAsyncGateCallback, gate.get(), 0) != cudaSuccess) {
            relay->quarantine.store(true, std::memory_order_release); SourceRelays().slots[relaySlot].Signal(SourceUpload, cudaErrorUnknown);
            SourceRelays().slots[relaySlot].ReturnFromLauncher(SourceUpload); return true;
        }
    }
    bool const injectInstall = SourceRelays().failInstall[SourceUpload].exchange(false, std::memory_order_acq_rel);
    auto const uploadStatus = injectInstall ? gpu::CurveSourceStatus::Ok :
        job->state.source->FinishFreshAsync(stream, SourceCallback,
            &SourceRelays().slots[relaySlot].callback[SourceUpload]);
    if (injectInstall || uploadStatus != gpu::CurveSourceStatus::Ok) {
        if (!injectInstall)
            Fail(job->diagnostics, "source async callback installation failed: CurveSourceStatus " +
                CurveSourceStatusText(uploadStatus));
        relay->quarantine.store(true, std::memory_order_release); SourceRelays().slots[relaySlot].Signal(SourceUpload, cudaErrorUnknown);
        SourceRelays().slots[relaySlot].ReturnFromLauncher(SourceUpload);
        return true;
    }
    SourceRelays().slots[relaySlot].ReturnFromLauncher(SourceUpload);
    return true;
#endif
}

#ifdef USDGEN_ENABLE_CUDA
namespace {
constexpr size_t kOperatorRelayCapacity = 1024;
constexpr int kOperatorRelayWorkers = 8;
// Context/program proof is shared by all operators.  RBF then has a scalar
// budget readback, three rest-factor phases, posed-surface/solve proof, and
// four deformation phases.  Keep all of them in the same permanent callback
// slot: callbacks only signal a phase; relay workers own every CUDA call.
constexpr unsigned kOperatorRelayPhases = 14;
std::array<std::shared_ptr<SourceAsyncTestGate>, kOperatorRelayPhases> s_operatorAsyncGates;
std::shared_ptr<SourceAsyncTestGate> s_operatorAsyncReturnGate;
struct WidthBranchTestGate : SourceAsyncTestGate {
    explicit WidthBranchTestGate(size_t count) : expected(count) {}
    size_t const expected;
    std::atomic<size_t> claims{0};
    std::atomic<size_t> arrivals{0};
    bool Claim() noexcept {
        size_t value = claims.load(std::memory_order_acquire);
        while (value < expected) {
            if (claims.compare_exchange_weak(value, value + 1,
                    std::memory_order_acq_rel, std::memory_order_acquire))
                return true;
        }
        return false;
    }
};
std::shared_ptr<WidthBranchTestGate> s_operatorAsyncWidthBranchGate;
std::atomic<uint64_t> s_operatorAsyncWidthBranchStreamMask{0};
constexpr size_t kWidthOverlapWitnessMaxTasks =
    UsdGenExecutionOverlapWitnessSnapshot::kMaxTasks;
constexpr size_t kWidthOverlapWitnessCounters = 3;

// Retained by every submitted branch candidate as well as the test control.
// Consequently a re-arm cannot destroy state still referenced by queued CUDA
// work, and an unproven terminal keeps it through the normal relay quarantine.
struct WidthOverlapWitness {
    gpu::DeviceBuffer<uint32_t> counters;
    cudaStream_t initializeStream = nullptr;
    cudaEvent_t initialized = nullptr;
    int device = -1;
    uint32_t expected = 0;
    uint64_t dwellCycles = 0;
    std::atomic<uint32_t> claims{0}, submitted{0};
    std::array<std::atomic<uint32_t>, kWidthOverlapWitnessMaxTasks> taskIds;
    std::array<std::atomic<uint32_t>, kWidthOverlapWitnessMaxTasks> laneIds;
    std::atomic<bool> error{false}, unsafe{false};

    WidthOverlapWitness() {
        for (auto& value : taskIds) value.store(UINT32_MAX, std::memory_order_relaxed);
        for (auto& value : laneIds) value.store(UINT32_MAX, std::memory_order_relaxed);
    }
    ~WidthOverlapWitness() {
        if (unsafe.load(std::memory_order_acquire)) {
            counters.quarantine();
            // An initialization event/stream may be ordered before a branch
            // probe with no completion proof.  Deliberately leave its native
            // handles for context teardown rather than freeing them here.
            initializeStream = nullptr; initialized = nullptr;
            return;
        }
        int previous = -1;
        bool const selected = cudaGetDevice(&previous) == cudaSuccess &&
            cudaSetDevice(device) == cudaSuccess;
        if (!selected) {
            counters.quarantine(); initializeStream = nullptr; initialized = nullptr;
            return;
        }
        if (initialized) cudaEventDestroy(initialized);
        if (initializeStream) cudaStreamDestroy(initializeStream);
        if (previous >= 0 && previous != device) cudaSetDevice(previous);
    }
    void Abandon() noexcept {
        unsafe.store(true, std::memory_order_release);
    }
    bool Initialize(int selectedDevice, uint32_t count, uint64_t cycles) noexcept {
        device = selectedDevice; expected = count; dwellCycles = cycles;
        if (counters.reset(kWidthOverlapWitnessCounters) != cudaSuccess ||
            cudaStreamCreateWithFlags(&initializeStream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&initialized, cudaEventDisableTiming) != cudaSuccess)
            return false;
        if (cudaMemsetAsync(counters.data(), 0,
                kWidthOverlapWitnessCounters * sizeof(uint32_t), initializeStream) != cudaSuccess ||
            cudaEventRecord(initialized, initializeStream) != cudaSuccess) {
            Abandon();
            return false;
        }
        return true;
    }
    bool Submit(uint32_t task, uint32_t lane, cudaStream_t stream) noexcept {
        uint32_t claim = claims.load(std::memory_order_acquire);
        while (claim < expected) {
            if (claims.compare_exchange_weak(claim, claim + 1,
                    std::memory_order_acq_rel, std::memory_order_acquire)) break;
        }
        if (claim >= expected) return false;
        taskIds[claim].store(task, std::memory_order_release);
        laneIds[claim].store(lane, std::memory_order_release);
        if (cudaStreamWaitEvent(stream, initialized, 0) != cudaSuccess ||
            !gpu::LaunchWidthOverlapWitness(
                counters.view(), expected, dwellCycles, stream)) {
            error.store(true, std::memory_order_release);
            return false;
        }
        submitted.fetch_add(1, std::memory_order_release);
        return true;
    }
    UsdGenExecutionOverlapWitnessSnapshot Snapshot() noexcept {
        UsdGenExecutionOverlapWitnessSnapshot result;
        result.expected = expected;
        result.claims = claims.load(std::memory_order_acquire);
        for (size_t i = 0; i != result.taskIds.size(); ++i) {
            result.taskIds[i] = taskIds[i].load(std::memory_order_acquire);
            result.laneIds[i] = laneIds[i].load(std::memory_order_acquire);
        }
        if (error.load(std::memory_order_acquire)) {
            result.status = UsdGenExecutionOverlapWitnessStatus::Error;
            return result;
        }
        if (unsafe.load(std::memory_order_acquire)) {
            result.status = UsdGenExecutionOverlapWitnessStatus::Error;
            return result;
        }
        CudaDeviceScope selected(device);
        std::array<uint32_t, kWidthOverlapWitnessCounters> host{};
        if (!selected.selected || !counters.data() ||
            cudaMemcpy(host.data(), counters.data(), host.size() * sizeof(uint32_t),
                cudaMemcpyDeviceToHost) != cudaSuccess) {
            result.status = UsdGenExecutionOverlapWitnessStatus::Error;
            return result;
        }
        result.arrivals = host[2]; result.maxActive = host[1];
        auto const sent = submitted.load(std::memory_order_acquire);
        result.status = sent == 0 ? UsdGenExecutionOverlapWitnessStatus::Armed :
            (result.arrivals == 0 ? UsdGenExecutionOverlapWitnessStatus::Submitted :
            (result.arrivals >= expected && result.maxActive >= 2
                ? UsdGenExecutionOverlapWitnessStatus::Observed
                : UsdGenExecutionOverlapWitnessStatus::Incomplete));
        return result;
    }
};
std::shared_ptr<WidthOverlapWitness> s_operatorAsyncWidthOverlapWitness;
struct OperatorWidthCandidate {
    std::unique_ptr<gpu::DeviceBuffer<float>> profile, widths;
    std::unique_ptr<gpu::DeviceBuffer<float>> priorWidths;
    std::unique_ptr<gpu::CudaWidth> width;
    std::unique_ptr<gpu::CudaCurveFullNonWidthCompare> nonWidthCompare;
    std::shared_ptr<const void> nonWidthLifetime;
    CudaWidthBlendGraphCache::Lease widthBlendGraphLease;
    bool widthBlend = false;
    float blend = 0.0f;
    bool submitted = false;
    bool semanticRejected = false;
};
struct OperatorLengthCandidate {
    std::unique_ptr<gpu::DeviceBuffer<float3>> changedPoints;
    std::unique_ptr<gpu::DeviceBuffer<uint8_t>> keep;
    std::unique_ptr<gpu::CudaLength> length;
    std::unique_ptr<gpu::CudaCurveCompaction> compaction;
    std::unique_ptr<gpu::CudaNamedChannelTopologyCandidate> namedTopology;
    std::vector<gpu::CudaNamedChannelPlane> transformedNamedChannels;
    std::unique_ptr<gpu::CudaCurveCompaction> priorCompacted;
    std::unique_ptr<gpu::DeviceBuffer<float3>> priorPoints;
    std::unique_ptr<gpu::DeviceBuffer<float>> priorWidths;
    bool submitted = false;
    bool semanticRejected = false;
};
struct OperatorNoiseCandidate {
    std::unique_ptr<gpu::DeviceBuffer<float>> profile;
    std::unique_ptr<gpu::DeviceBuffer<float3>> points;
    std::unique_ptr<gpu::DeviceBuffer<float3>> priorPoints;
    std::unique_ptr<gpu::CudaNoise> noise;
    bool submitted = false;
    bool semanticRejected = false;
};
struct OperatorRbfCandidate {
    // A rebind is entirely private until the final deformer copy has proved.
    // Same-rest poses borrow the accepted state but retain every producer
    // buffer and the old coefficients through the solve transaction.
    CudaRbfCache* cache = nullptr;
    std::unique_ptr<CudaRbfCache::State> freshState;
    CudaRbfCache::State* state = nullptr;
    std::unique_ptr<CudaGroomScalarReadback> budgetPacket;
    std::unique_ptr<gpu::DeviceBuffer<float3>> rest, current, points;
    std::unique_ptr<gpu::DeviceBuffer<uint32_t>> offsets, indices;
    std::unique_ptr<gpu::CudaRbfCurveDeformer> deformer;
    gpu::DeformParameters parameters;
    uint32_t budget = 0;
    bool rebind = false;
    bool resetCounters = false;
    bool solveCommitted = false;
    bool surfaceCommitted = false;
};
struct OperatorRelay {
    std::shared_ptr<UsdGenCudaExecutionJob> job;
    std::function<void(bool)> done;
    std::unique_ptr<OperatorWidthCandidate> candidate;
    std::unique_ptr<OperatorLengthCandidate> length;
    std::unique_ptr<OperatorGrowCandidate> grow;
    std::unique_ptr<OperatorNoiseCandidate> noise;
    std::unique_ptr<OperatorRbfCandidate> rbf;
    std::function<void()> poison;
    std::array<std::shared_ptr<SourceAsyncTestGate>, kOperatorRelayPhases> gates;
    std::shared_ptr<WidthBranchTestGate> widthBranchGate;
    std::shared_ptr<WidthOverlapWitness> overlapWitness;
    std::shared_ptr<SourceAsyncTestGate> returnGate;
    std::atomic<bool> failed{false}, quarantine{false};
    bool geometryGuard = false;
    size_t slot = std::numeric_limits<size_t>::max(), index = 0;
    // Immutable source geometry plus the predecessor Width value captured at
    // task admission.  No sibling relay is allowed to rebind job.state.geometry.
    gpu::DeviceCurveGeometryView geometry;
    gpu::DeviceView<const float> rightWidths;
    gpu::DeviceView<const float> hairT;
    gpu::DeviceView<const int32_t> rootPrim;
    gpu::DeviceView<const float2> rootUV;
    gpu::RestRootFrames frames;
    std::shared_ptr<NamedOwners> namedOwners;
    cudaStream_t stream = nullptr;
};
}

// This is deliberately distinct from SourceRelayService: source admission is
// producer sequencing; Width uses phases 0..2 and Length extends its same
// fixed slot through count/scatter phases 3..4 and named-plane proof phase 5.
// Slots and callback identities are permanent; a native callback has no job,
// candidate, workspace, or completion ownership to destroy.
struct OperatorRelayService {
    struct Slot {
        struct CallbackIdentity { Slot* slot = nullptr; unsigned phase = 0; };
        std::atomic<bool> occupied{false}, quarantined{false};
        std::shared_ptr<OperatorRelay> relay;
        OperatorRelayService* service = nullptr;
        size_t index = 0;
        std::array<std::atomic<unsigned>, kOperatorRelayPhases> state{};
        std::array<std::atomic<int>, kOperatorRelayPhases> status{};
        std::array<std::atomic<SourceAsyncTestGate*>, kOperatorRelayPhases> gate{};
        std::atomic<SourceAsyncTestGate*> widthReturnGate{nullptr};
        std::array<CallbackIdentity, kOperatorRelayPhases> callback{};
        void Signal(unsigned, cudaError_t) noexcept;
        void Return(unsigned) noexcept;
    };
    std::array<Slot, kOperatorRelayCapacity> slots;
    tbb::task_arena arena{kOperatorRelayWorkers, 0};
    std::unique_ptr<tbb::flow::graph> graph;
    std::unique_ptr<tbb::flow::function_node<size_t>> worker;
    std::atomic<size_t> capacity{kOperatorRelayCapacity};
    std::array<std::atomic<bool>, kOperatorRelayPhases> failInstall{}, failNative{};

    OperatorRelayService() {
        arena.execute([this] {
            graph = std::make_unique<tbb::flow::graph>();
            worker = std::make_unique<tbb::flow::function_node<size_t>>(*graph,
                tbb::flow::unlimited, [this](size_t message) {
                    try { Complete(message / kOperatorRelayPhases, unsigned(message % kOperatorRelayPhases)); }
                    catch (...) { Quarantine(message / kOperatorRelayPhases); }
                    return tbb::flow::continue_msg{};
                });
        });
        for (size_t i = 0; i != slots.size(); ++i) {
            slots[i].service = this; slots[i].index = i;
            for (unsigned phase = 0; phase != kOperatorRelayPhases; ++phase)
                slots[i].callback[phase] = {&slots[i], phase};
        }
    }
    OperatorRelay* Reserve(std::shared_ptr<UsdGenCudaExecutionJob> job,
                           size_t index, std::function<void(bool)> done) noexcept {
        size_t const limit = std::min(capacity.load(std::memory_order_acquire), slots.size());
        for (size_t i = 0; i != limit; ++i) {
            bool expected = false;
            if (!slots[i].occupied.compare_exchange_strong(expected, true,
                    std::memory_order_acq_rel, std::memory_order_acquire)) continue;
            try {
                auto relay = std::make_shared<OperatorRelay>();
                relay->job = std::move(job); relay->done = std::move(done);
                relay->slot = i; relay->index = index;
                for (unsigned p = 0; p != kOperatorRelayPhases; ++p) {
                    slots[i].state[p].store(0, std::memory_order_relaxed);
                    slots[i].status[p].store(int(cudaErrorUnknown), std::memory_order_relaxed);
                }
                slots[i].quarantined.store(false, std::memory_order_relaxed);
                for (auto& gate : slots[i].gate) gate.store(nullptr, std::memory_order_relaxed);
                slots[i].widthReturnGate.store(nullptr, std::memory_order_relaxed);
                slots[i].relay = std::move(relay);
                return slots[i].relay.get();
            } catch (...) { slots[i].occupied.store(false, std::memory_order_release); return nullptr; }
        }
        return nullptr;
    }
    void Post(size_t index, unsigned phase) noexcept {
        try { if (!worker->try_put(index * kOperatorRelayPhases + phase)) std::terminate(); }
        catch (...) { std::terminate(); }
    }
    bool Install(Slot& slot, unsigned phase, cudaStream_t stream,
                 void (*callback)(cudaStream_t, cudaError_t, void*) noexcept) noexcept {
        auto relay = slot.relay;
        if (relay) {
            auto gate = std::atomic_load(&s_operatorAsyncGates[phase]);
            if (gate && !gate->claimed.exchange(true, std::memory_order_acq_rel)) {
                relay->gates[phase] = gate;
                slot.gate[phase].store(gate.get(), std::memory_order_release);
            }
        }
        return !failInstall[phase].exchange(false, std::memory_order_acq_rel) &&
            cudaStreamAddCallback(stream, callback, &slot.callback[phase], 0) == cudaSuccess;
    }
    bool InstallGate(Slot& slot, unsigned phase, cudaStream_t stream) noexcept {
        (void)stream;
        auto relay = slot.relay;
        if (!relay) return false;
        auto gate = std::atomic_load(&s_operatorAsyncGates[phase]);
        if (!gate || gate->claimed.exchange(true, std::memory_order_acq_rel)) return true;
        relay->gates[phase] = gate;
        slot.gate[phase].store(gate.get(), std::memory_order_release);
        return true;
    }
    void Finish(size_t index, bool ok, bool unsafe) noexcept {
        auto relay = slots[index].relay;
        if (!relay) return;
        auto job = relay->job;
        if (!ok && !unsafe && relay->length && relay->length->namedTopology &&
            relay->length->namedTopology->HasUnprovenWork())
            unsafe = true;
        if (!ok && !unsafe && relay->grow &&
            ((relay->grow->grow && relay->grow->grow->HasUnprovenWork()) ||
             (relay->grow->namedTopology && relay->grow->namedTopology->HasUnprovenWork())))
            unsafe = true;
        if (!ok && !unsafe && relay->candidate && relay->candidate->nonWidthCompare &&
            relay->candidate->nonWidthCompare->HasUnprovenWork()) {
            relay->candidate->nonWidthCompare->Quarantine();
            unsafe = true;
        }
        // Geometry-value producers keep their native candidate on this
        // relay.  Do not scan legacy state here: siblings may publish bundle
        // slots concurrently, and this relay has already inspected its own
        // Grow/Length packet above.
        bool const inspectState = job && job->plan && !job->plan->geometryValueDag;
        if (!ok && !unsafe && inspectState && job->state.HasUnprovenAsyncWork())
            unsafe = true;
        if (!ok && !unsafe && job && job->workspace) {
            bool needsRbfResolution = (!job->plan->taskDag && job->state.HasPendingRbf()) ||
                (relay->rbf && (relay->rbf->solveCommitted ||
                                relay->rbf->surfaceCommitted));
            CudaDeviceScope selected(job->workspace->impl_->device);
            // A failure after solve/surface proof but before the phase-13
            // publication packet still owns the same provisional transaction.
            // Route it through the one resolver rather than independently
            // rolling back resources and silently dropping a failure.
            bool const ownPending = relay->index < job->state.pendingRbf.size() &&
                bool(job->state.pendingRbf[relay->index]);
            if (ownPending && relay->rbf &&
                (relay->rbf->solveCommitted || relay->rbf->surfaceCommitted))
                unsafe = true;
            if (!ownPending && relay->rbf &&
                (relay->rbf->solveCommitted || relay->rbf->surfaceCommitted)) {
                try {
                    if (relay->index >= job->state.pendingRbf.size() ||
                        job->state.pendingRbf[relay->index]) {
                        unsafe = true;
                    } else {
                    auto& candidate = *relay->rbf;
                    auto transaction = std::make_unique<CudaRbfPendingPublication>();
                    transaction->cache = candidate.cache;
                    transaction->state = candidate.state;
                    transaction->key = job->plan->steps[relay->index]->surface.key;
                    transaction->rebind = candidate.rebind;
                    transaction->resetCounters = candidate.resetCounters;
                    transaction->solveCommitted = candidate.solveCommitted;
                    transaction->surfaceCommitted = candidate.surfaceCommitted;
                    transaction->freshState = std::move(candidate.freshState);
                    if (transaction->freshState)
                        transaction->state = transaction->freshState.get();
                    candidate.solveCommitted = false;
                    candidate.surfaceCommitted = false;
                    job->state.pendingRbf[relay->index] = std::move(transaction);
                    }
                } catch (...) {
                    // Finish is noexcept. A host allocation failure after a
                    // proved provisional mutation has no safe normal cleanup.
                    unsafe = true;
                }
            }
            if (!unsafe && needsRbfResolution && !job->plan->taskDag &&
                (!selected.selected || (job->state.HasPendingRbf() &&
                !job->state.ResolvePendingRbf(false))))
                unsafe = true;
        }
        if (!ok && job) job->failed = true;
        if (unsafe) {
            if (relay->grow && relay->grow->grow) relay->grow->grow->MarkUnprovenWork();
            if (relay->grow && relay->grow->namedTopology) relay->grow->namedTopology->Quarantine();
            if (relay->candidate && relay->candidate->widthBlendGraphLease)
                relay->candidate->widthBlendGraphLease.Quarantine();
            if (job) job->CloseMemoryReservation();
            if (relay->overlapWitness) relay->overlapWitness->Abandon();
            relay->quarantine.store(true, std::memory_order_release);
            slots[index].quarantined.store(true, std::memory_order_release);
            if (relay->poison) relay->poison();
            relay->poison = {};
            if (job) { job->workspace = nullptr; job->diagnostics = nullptr; }
        }
        auto done = std::move(relay->done);
        if (unsafe) {
            relay->done = {};
            if (done) try { done(false); } catch (...) {}
            return;
        }
        bool const geometryGuard = relay->geometryGuard;
        bool const noiseGuard = job && job->plan && relay->index < job->plan->steps.size() &&
            job->plan->steps[relay->index]->type == TfToken("UsdGenNoise");
        if (!ok && job) job->CloseMemoryReservation();
        relay->candidate.reset(); relay->length.reset(); relay->grow.reset(); relay->noise.reset(); relay->rbf.reset(); relay->job.reset();
        slots[index].relay.reset(); slots[index].occupied.store(false, std::memory_order_release);
        // Do not admit another geometry lane (or report no operators in
        // flight) until every prior candidate has released its scratch/event
        // owners and the relay slot is no longer observable.
        if (job) job->operatorAsyncInFlight.fetch_sub(1, std::memory_order_acq_rel);
        if (job && geometryGuard)
            job->geometryAsyncInFlight.store(false, std::memory_order_release);
        if (job && noiseGuard)
            job->noiseAsyncInFlight.store(false, std::memory_order_release);
        if (done) try { done(ok); } catch (...) {}
    }
    void Quarantine(size_t index) noexcept { Finish(index, false, true); }
    void Complete(size_t slotIndex, unsigned phase);
};

namespace {
OperatorRelayService& OperatorRelays() { static auto* service = new OperatorRelayService; return *service; }
void OperatorCallback(cudaStream_t, cudaError_t status, void* data) noexcept {
    auto* identity = static_cast<OperatorRelayService::Slot::CallbackIdentity*>(data);
    auto* slot = identity->slot;
    if (slot->service->failNative[identity->phase].exchange(false, std::memory_order_acq_rel))
        status = cudaErrorUnknown;
    if (identity->phase == 2) {
        if (auto* gate = slot->widthReturnGate.load(std::memory_order_acquire))
            gate->terminal.store(true, std::memory_order_release);
    }
    // Test-only phase gates are retained by the relay and reached only once
    // the low-level FreshAsync callback has observed its queued status D2H.
    // Do not put a predecessor stream callback ahead of that protocol.
    if (auto* gate = slot->gate[identity->phase].load(std::memory_order_acquire))
        SourceAsyncGateCallback(nullptr, status, gate);
    slot->Signal(identity->phase, status); // final slot access
}
}

void OperatorRelayService::Slot::Signal(unsigned phase, cudaError_t value) noexcept {
    auto& bits = state[phase];
    if (bits.fetch_or(8u, std::memory_order_acq_rel) & 8u) return;
    status[phase].store(int(value), std::memory_order_release);
    unsigned const old = bits.fetch_or(1u, std::memory_order_acq_rel);
    if ((old & 2u) && !(old & 4u) && !(bits.fetch_or(4u, std::memory_order_acq_rel) & 4u))
        service->Post(index, phase);
}
void OperatorRelayService::Slot::Return(unsigned phase) noexcept {
    auto& bits = state[phase];
    unsigned const old = bits.fetch_or(2u, std::memory_order_acq_rel);
    if ((old & 1u) && !(old & 4u) && !(bits.fetch_or(4u, std::memory_order_acq_rel) & 4u))
        service->Post(index, phase);
}

void OperatorRelayService::Complete(size_t slotIndex, unsigned phase) {
    auto relay = slots[slotIndex].relay;
    if (!relay || !relay->job || !relay->job->workspace) return;
    auto job = relay->job;
    auto& workspace = *job->workspace;
    auto& execution = *workspace.impl_;
    CudaDeviceScope selected(execution.device);
    bool const nativeOK = selected.selected &&
        cudaError_t(slots[slotIndex].status[phase].load(std::memory_order_acquire)) == cudaSuccess;
    bool gatesOK = true;
    for (auto const& gate : relay->gates)
        gatesOK = gatesOK && (!gate || !gate->timedOut.load(std::memory_order_acquire));
    gatesOK = gatesOK && (!relay->returnGate ||
        !relay->returnGate->timedOut.load(std::memory_order_acquire));
    if (!nativeOK || execution.poisoned.load(std::memory_order_acquire)) {
        selected.Restore(); Finish(slotIndex, false, true); return;
    }
    if (!gatesOK) { selected.Restore(); Finish(slotIndex, false, false); return; }
    auto const& step = *job->plan->steps[relay->index];
    auto& runtime = *execution.steps.at(step.path);
    auto const stream = relay->stream ? relay->stream : execution.stream;
    if (phase == 0) {
        if (runtime.parameters.CommitFreshContexts() != CudaParameterStatus::Ok) {
            selected.Restore(); Finish(slotIndex, false, runtime.parameters.HasUnprovenWork()); return;
        }
        if (runtime.parameters.BeginFreshPrograms(stream,
                job->memoryReservation.get(),
                UsdGenExecutionResourceKind::Cache) != CudaParameterStatus::Ok) {
            bool const unsafe = runtime.parameters.HasUnprovenWork();
            if (!unsafe) runtime.parameters.DiscardFreshProven();
            selected.Restore(); Finish(slotIndex, false, unsafe); return;
        }
        if (runtime.parameters.EnqueueFreshProgramStatus(stream) != CudaParameterStatus::Ok) {
            selected.Restore(); Finish(slotIndex, false, runtime.parameters.HasUnprovenWork()); return;
        }
        if (!Install(slots[slotIndex], 1, stream, OperatorCallback)) { selected.Restore(); Finish(slotIndex, false, true); return; }
        selected.Restore(); slots[slotIndex].Return(1); return;
    }
    if (phase == 6 && job->plan->steps[relay->index]->type == TfToken("UsdGenWidthBlend")) {
        if (!relay->candidate || !relay->candidate->nonWidthCompare) {
            selected.Restore(); Finish(slotIndex, false, false); return;
        }
        bool equal = false;
        if (relay->candidate->nonWidthCompare->CommitFreshFinish(&equal) != cudaSuccess) {
            selected.Restore(); Finish(slotIndex, false, false); return;
        }
        relay->candidate->nonWidthLifetime.reset();
        relay->candidate->nonWidthCompare.reset();
        if (!equal) { selected.Restore(); Finish(slotIndex, false, false); return; }
        auto const& blendStep = *job->plan->steps[relay->index];
        relay->candidate->submitted = true;
        bool launched = false;
        if (execution.widthBlendGraph) {
            CudaWidthBlendGraphCache::Lease lease;
            if (execution.widthBlendGraph->Acquire(MakeWidthBlendGraphKey(
                    *job->plan, blendStep, relay->geometry.pointCount), &lease)) {
                relay->candidate->widthBlendGraphLease = std::move(lease);
                launched = execution.widthBlendGraph->CopyLaunchCopy(relay->geometry.widths,
                    relay->rightWidths, relay->candidate->widths->view(), stream);
            }
        }
        if (!launched && !relay->candidate->widthBlendGraphLease)
            launched = relay->rightWidths.size == relay->geometry.pointCount &&
                gpu::LaunchWidthBlend(relay->geometry.widths, relay->rightWidths,
                    relay->candidate->blend, relay->candidate->widths->view(), stream);
        if (!launched || !Install(slots[slotIndex], 2, stream, OperatorCallback)) {
            selected.Restore(); Finish(slotIndex, false, true); return;
        }
        selected.Restore(); slots[slotIndex].Return(2); return;
    }
    if (phase == 1) {
        if (runtime.parameters.CommitFreshPrograms() != CudaParameterStatus::Ok) {
            selected.Restore(); Finish(slotIndex, false, runtime.parameters.HasUnprovenWork()); return;
        }
        if (step.type == TfToken("UsdGenGrow")) {
            if (job->plan->geometryValueDag) {
                auto const inputNode = step.inputNode;
                if (inputNode >= job->state.geometryValues.size()) {
                    selected.Restore(); Finish(slotIndex, false, false); return;
                }
                auto const& input = job->state.geometryValues[inputNode];
                relay->grow = std::make_unique<OperatorGrowCandidate>();
                if (!job->state.BeginCurveGrowValue(step.grow,
                        relay->geometry, relay->rootPrim, relay->rootUV,
                        input.topologyOwners, job->state.sourceTopologyOwners,
                        job->state.captureStableIds, relay->namedOwners, job,
                        relay->grow.get(), stream, job->diagnostics)) {
                    bool const unsafe = relay->grow->unsafeInput ||
                        (relay->grow->grow && relay->grow->grow->HasUnprovenWork());
                    selected.Restore(); Finish(slotIndex, false, unsafe); return;
                }
            } else if (!job->state.BeginCurveGrow(step.grow, stream,
                                                    job->diagnostics)) {
                selected.Restore(); Finish(slotIndex, false, job->state.HasUnprovenAsyncWork()); return;
            }
            if (!InstallGate(slots[slotIndex], 2, stream) ||
                failInstall[2].exchange(false, std::memory_order_acq_rel) ||
                (job->plan->geometryValueDag ? relay->grow->grow.get() : job->state.curveGrow.get())
                    ->FinishFreshAsync(stream, OperatorCallback,
                    &slots[slotIndex].callback[2]) != gpu::CurveGrowStatus::Ok) {
                selected.Restore(); Finish(slotIndex, false, true); return;
            }
            selected.Restore(); slots[slotIndex].Return(2); return;
        }
        if (step.type == TfToken("UsdGenDeform")) {
            if (job->plan->geometryValueDag &&
                (step.inputNode >= job->state.geometryValues.size() ||
                 job->state.geometryValues[step.inputNode].deformed)) {
                Fail(job->diagnostics,
                    "unsupported composition: a second rest-to-animated deformation would apply surface motion twice");
                selected.Restore(); Finish(slotIndex, false, false); return;
            }
            auto candidate = std::make_unique<OperatorRbfCandidate>();
            candidate->cache = runtime.rbf.get();
            candidate->budgetPacket = std::make_unique<CudaGroomScalarReadback>();
            UsdGenParamView values; values.desc = &job->plan->desc;
            values.node = &job->plan->desc.nodes[step.semanticNode];
            candidate->budget = static_cast<uint32_t>(values.GetInt(TfToken("rbfSamples"), 100));
            if (!candidate->cache || candidate->budget < 4 || candidate->budget > 46336 ||
                candidate->budgetPacket->BeginFresh(runtime.parameters.Fields(),
                    {{TfToken("rbfSamples"), expr::ScalarType::Int32}}, stream,
                    job->memoryReservation.get()) != CudaParameterStatus::Ok) {
                selected.Restore(); Finish(slotIndex, false,
                    candidate->budgetPacket && candidate->budgetPacket->HasUnprovenWork()); return;
            }
            relay->rbf = std::move(candidate);
            if (!Install(slots[slotIndex], 2, stream, OperatorCallback)) {
                selected.Restore(); Finish(slotIndex, false, true); return;
            }
            selected.Restore(); slots[slotIndex].Return(2); return;
        }
        if (step.type == TfToken("UsdGenLength")) {
            auto candidate = std::make_unique<OperatorLengthCandidate>();
            candidate->changedPoints = std::make_unique<gpu::DeviceBuffer<float3>>();
            candidate->keep = std::make_unique<gpu::DeviceBuffer<uint8_t>>();
            candidate->length = std::make_unique<gpu::CudaLength>();
            candidate->compaction = std::make_unique<gpu::CudaCurveCompaction>();
            relay->length = std::move(candidate);
            auto* const reservation = job->memoryReservation.get();
            if (relay->length->changedPoints->reset(relay->geometry.pointCount,
                    reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
                relay->length->keep->reset(relay->geometry.curveCount,
                    reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess) {
                selected.Restore(); Finish(slotIndex, false, false); return;
            }
            UsdGenParamView values; values.desc = &job->plan->desc; values.node = &job->plan->desc.nodes[step.semanticNode];
            auto field = [&](char const* name, float fallback) { if (auto const* e = runtime.parameters.Find(TfToken(name))) return gpu::ScalarField::Device({static_cast<float const*>(e->data), e->count}, e->domain); return gpu::ScalarField::Literal(static_cast<float>(values.GetDouble(TfToken(name), fallback))); };
            auto boolean = [&](char const* name, bool fallback) { if (auto const* e = runtime.parameters.Find(TfToken(name))) return gpu::BoolField::Device({static_cast<uint8_t const*>(e->data), e->count}, e->domain); return gpu::BoolField::Literal(values.GetBool(TfToken(name), fallback)); };
            gpu::LengthParameters p; auto mode = values.GetToken(TfToken("length:mode"), TfToken("scale"));
            p.mode = mode == TfToken("set") ? gpu::LengthMode::Set : mode == TfToken("cull") ? gpu::LengthMode::Cull : gpu::LengthMode::Scale;
            p.method = values.GetToken(TfToken("length:method"), TfToken("scale")) == TfToken("cutExtend") ? gpu::LengthMethod::CutExtend : gpu::LengthMethod::Scale;
            p.rebuild = values.GetToken(TfToken("rebuild"), TfToken("keepParam")) == TfToken("reparam") ? gpu::LengthRebuild::Reparam : gpu::LengthRebuild::KeepParam;
            p.value = field("length:value", 1); p.mask = field("mask", 1); p.minRemainingLength = field("minRemainingLength", 0); p.cullThreshold = field("cullThreshold", 0); p.enabled = boolean("enabled", values.node->enabled); p.seed = values.node->seed;
            if (auto e = runtime.parameters.Find(TfToken("length:random"))) p.random = gpu::Vec2Field::Device({static_cast<float2 const*>(e->data), e->count}, e->domain); else { auto v = values.GetVtValue(TfToken("length:random"), VtValue(GfVec2f(1,1))).UncheckedGet<GfVec2f>(); p.random = gpu::Vec2Field::Literal(make_float2(v[0],v[1])); }
            relay->length->submitted = true;
            if (relay->length->length->ApplyFresh(relay->geometry, relay->hairT, p, relay->length->changedPoints->view(), relay->length->keep->view(), stream, reservation) != gpu::StyleStatus::Ok) {
                if (relay->length->length->HasUnprovenWork() || !InstallGate(slots[slotIndex], 2, stream) || failInstall[2].exchange(false, std::memory_order_acq_rel) || cudaStreamAddCallback(stream, OperatorCallback, &slots[slotIndex].callback[2], 0) != cudaSuccess) { selected.Restore(); Finish(slotIndex, false, true); return; }
                relay->length->semanticRejected = true; selected.Restore(); slots[slotIndex].Return(2); return;
            }
            if (!InstallGate(slots[slotIndex], 2, stream) || failInstall[2].exchange(false, std::memory_order_acq_rel) || relay->length->length->FinishFreshAsync(stream, OperatorCallback, &slots[slotIndex].callback[2]) != gpu::StyleStatus::Ok) { selected.Restore(); Finish(slotIndex, false, true); return; }
            selected.Restore(); slots[slotIndex].Return(2); return;
        }
        if (step.type == TfToken("UsdGenWidthBlend")) {
            auto candidate = std::make_unique<OperatorWidthCandidate>();
            candidate->widthBlend = true;
            candidate->blend = CudaWidthBlendWeight(job->plan->desc,
                job->plan->desc.nodes[step.semanticNode]);
            candidate->widths = std::make_unique<gpu::DeviceBuffer<float>>();
            relay->candidate = std::move(candidate);
            auto* const reservation = job->memoryReservation.get();
            if (relay->candidate->widths->reset(relay->geometry.pointCount,
                    reservation, UsdGenExecutionResourceKind::Active) != cudaSuccess) {
                selected.Restore(); Finish(slotIndex, false, false); return;
            }
            if (job->plan->geometryValueDag && step.requiresNonWidthProof) {
                auto const& left = job->state.geometryValues[step.inputNode];
                auto const& right = job->state.geometryValues[step.rightInputNode];
                {
                    gpu::CurveFullNonWidthInput a, b;
                    if (!BuildFullNonWidthInput(left, &a) || !BuildFullNonWidthInput(right, &b) ||
                        !left.sourceFrameDomain || left.sourceFrameDomain != right.sourceFrameDomain ||
                        left.sourceFramesPresent != right.sourceFramesPresent ||
                        left.curveTopology.type != right.curveTopology.type ||
                        left.curveTopology.basis != right.curveTopology.basis ||
                        left.curveTopology.wrap != right.curveTopology.wrap) {
                        selected.Restore(); Finish(slotIndex, false, false); return;
                    }
                    relay->candidate->nonWidthCompare = std::make_unique<gpu::CudaCurveFullNonWidthCompare>();
                    relay->candidate->nonWidthLifetime = job;
                    auto status = relay->candidate->nonWidthCompare->BeginFresh(a, b,
                        relay->candidate->nonWidthLifetime, stream, reservation);
                    if (status != cudaSuccess) { selected.Restore(); Finish(slotIndex, false, false); return; }
                    status = relay->candidate->nonWidthCompare->EnqueueFreshStatus(stream);
                    if (status != cudaSuccess || !Install(slots[slotIndex], 6, stream, OperatorCallback)) {
                        selected.Restore(); Finish(slotIndex, false, true); return;
                    }
                    selected.Restore(); slots[slotIndex].Return(6); return;
                }
            }
            relay->candidate->submitted = true;
            if (execution.widthBlendGraph) {
                CudaWidthBlendGraphCache::Lease lease;
                if (execution.widthBlendGraph->Acquire(MakeWidthBlendGraphKey(
                        *job->plan, step, relay->geometry.pointCount), &lease)) {
                    relay->candidate->widthBlendGraphLease = std::move(lease);
                    if (!execution.widthBlendGraph->CopyLaunchCopy(relay->geometry.widths,
                            relay->rightWidths, relay->candidate->widths->view(), stream)) {
                        selected.Restore(); Finish(slotIndex, false, true); return;
                    }
                    if (!Install(slots[slotIndex], 2, stream, OperatorCallback)) {
                        selected.Restore(); Finish(slotIndex, false, true); return;
                    }
                    selected.Restore(); slots[slotIndex].Return(2); return;
                }
            }
            if (relay->rightWidths.size != relay->geometry.pointCount ||
                !gpu::LaunchWidthBlend(relay->geometry.widths, relay->rightWidths,
                    relay->candidate->blend, relay->candidate->widths->view(), stream)) {
                // A launch status cannot prove whether an earlier command
                // touched this fresh plane; retain/quarantine it with the
                // relay instead of letting a stack destructor free it.
                selected.Restore(); Finish(slotIndex, false, true); return;
            }
            if (!Install(slots[slotIndex], 2, stream, OperatorCallback)) {
                selected.Restore(); Finish(slotIndex, false, true); return;
            }
            selected.Restore(); slots[slotIndex].Return(2); return;
        }
        if (step.type == TfToken("UsdGenNoise")) {
            auto candidate = std::make_unique<OperatorNoiseCandidate>();
            candidate->profile = std::make_unique<gpu::DeviceBuffer<float>>();
            candidate->points = std::make_unique<gpu::DeviceBuffer<float3>>();
            candidate->noise = std::make_unique<gpu::CudaNoise>();
            auto* const reservation = job->memoryReservation.get();
            gpu::DeviceBuffer<float3> *tangent = nullptr, *binormal = nullptr,
                                      *normal = nullptr;
            gpu::DeviceView<const uint64_t> frameIds;
            if (!job->state.NoiseFrames(&tangent, &binormal, &normal, &frameIds)) {
                selected.Restore(); Finish(slotIndex, false, false); return;
            }
            if (candidate->profile->reset(step.profile.size(), reservation,
                    UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
                candidate->points->reset(relay->geometry.pointCount, reservation,
                    UsdGenExecutionResourceKind::Active) != cudaSuccess) {
                selected.Restore(); Finish(slotIndex, false, false); return;
            }
            relay->noise = std::move(candidate);
            auto& noise = *relay->noise;
            cudaError_t upload = cudaMemcpyAsync(noise.profile->data(), step.profile.data(),
                sizeof(step.profile), cudaMemcpyHostToDevice, stream);
            bool const uploaded = upload == cudaSuccess;
            if (upload != cudaSuccess) {
                bool const unsafe = uploaded && cudaStreamSynchronize(stream) != cudaSuccess;
                selected.Restore(); Finish(slotIndex, false, unsafe); return;
            }
            UsdGenParamView values; values.desc = &job->plan->desc;
            values.node = &job->plan->desc.nodes[step.semanticNode];
            auto field = [&](char const* name, float fallback) {
                if (auto const* e = runtime.parameters.Find(TfToken(name)))
                    return gpu::ScalarField::Device(
                        {static_cast<float const*>(e->data), e->count}, e->domain);
                return gpu::ScalarField::Literal(
                    static_cast<float>(values.GetDouble(TfToken(name), fallback)));
            };
            auto boolean = [&](char const* name, bool fallback) {
                if (auto const* e = runtime.parameters.Find(TfToken(name)))
                    return gpu::BoolField::Device(
                        {static_cast<uint8_t const*>(e->data), e->count}, e->domain);
                return gpu::BoolField::Literal(values.GetBool(TfToken(name), fallback));
            };
            auto integer = [&](char const* name, int fallback) {
                if (auto const* e = runtime.parameters.Find(TfToken(name)))
                    return gpu::IntField::Device(
                        {static_cast<int32_t const*>(e->data), e->count}, e->domain);
                return gpu::IntField::Literal(values.GetInt(TfToken(name), fallback));
            };
            gpu::NoiseParameters parameters;
            parameters.magnitude = field("noise:magnitude", .05f);
            parameters.frequency = field("noise:frequency", 3.0f);
            parameters.correlation = field("noise:correlation", .5f);
            parameters.octaves = integer("noise:octaves", 1);
            parameters.lacunarity = field("noise:lacunarity", 2.0f);
            parameters.gain = field("noise:gain", .5f);
            parameters.preserveLength = field("preserveLength", 1.0f);
            parameters.mask = field("mask", 1.0f);
            parameters.enabled = boolean("enabled", values.node->enabled);
            parameters.cumulative = boolean("cumulative", false);
            parameters.seed = integer("noise:seed", values.node->seed);
            parameters.magnitudeProfile = {
                noise.profile->data(), noise.profile->size()};
            noise.submitted = true;
            gpu::RestRootFrames const frames = relay->geometry.curveCount ? gpu::RestRootFrames{
                {tangent->data(), tangent->size()},
                {binormal->data(), binormal->size()},
                {normal->data(), normal->size()},
                frameIds} : gpu::RestRootFrames{};
            if (tangent->waitOn(stream) != cudaSuccess ||
                binormal->waitOn(stream) != cudaSuccess ||
                normal->waitOn(stream) != cudaSuccess) {
                selected.Restore(); Finish(slotIndex, false, true); return;
            }
            if (noise.noise->ApplyFresh(relay->geometry, relay->hairT, frames,
                    parameters, noise.points->view(), stream, reservation) != gpu::StyleStatus::Ok) {
                bool const unsafe = noise.noise->HasUnprovenWork();
                if (!unsafe) noise.semanticRejected = true;
                selected.Restore(); Finish(slotIndex, false, unsafe); return;
            }
            if (tangent->recordUse(stream) != cudaSuccess ||
                binormal->recordUse(stream) != cudaSuccess ||
                normal->recordUse(stream) != cudaSuccess ||
                noise.profile->recordUse(stream) != cudaSuccess) {
                bool const unsafe = cudaStreamSynchronize(stream) != cudaSuccess;
                selected.Restore(); Finish(slotIndex, false, unsafe); return;
            }
            if (noise.noise->FinishFreshAsync(stream, OperatorCallback,
                    &slots[slotIndex].callback[2]) != gpu::StyleStatus::Ok) {
                bool unsafe = noise.noise->HasUnprovenWork();
                if (!unsafe) unsafe = cudaStreamSynchronize(stream) != cudaSuccess;
                selected.Restore(); Finish(slotIndex, false, unsafe); return;
            }
            selected.Restore(); slots[slotIndex].Return(2); return;
        }
        auto candidate = std::make_unique<OperatorWidthCandidate>();
        candidate->profile = std::make_unique<gpu::DeviceBuffer<float>>();
        candidate->widths = std::make_unique<gpu::DeviceBuffer<float>>();
        candidate->width = std::make_unique<gpu::CudaWidth>();
        UsdGenParamView values; values.desc = &job->plan->desc; values.node = &job->plan->desc.nodes[step.semanticNode];
        auto field = [&](char const* name, float fallback) {
            if (auto const* expression = runtime.parameters.Find(TfToken(name)))
                return gpu::ScalarField::Device({static_cast<float const*>(expression->data), expression->count}, expression->domain);
            return gpu::ScalarField::Literal(static_cast<float>(values.GetDouble(TfToken(name), fallback)));
        };
        auto boolean = [&](char const* name, bool fallback) {
            if (auto const* expression = runtime.parameters.Find(TfToken(name)))
                return gpu::BoolField::Device({static_cast<uint8_t const*>(expression->data), expression->count}, expression->domain);
            return gpu::BoolField::Literal(values.GetBool(TfToken(name), fallback));
        };
        gpu::WidthParameters parameters;
        parameters.width = field("width", .01f);
        parameters.mask = field("mask", 1); parameters.enabled = boolean("enabled", values.node->enabled);
        parameters.replace = boolean("replace", true);
        relay->candidate = std::move(candidate);
        auto* const reservation = job->memoryReservation.get();
        if (relay->candidate->profile->reset(step.profile.size(), reservation,
                UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
            relay->candidate->widths->reset(relay->geometry.pointCount, reservation,
                UsdGenExecutionResourceKind::Active) != cudaSuccess) {
            selected.Restore(); Finish(slotIndex, false, false); return;
        }
        parameters.widthProfile = {relay->candidate->profile->data(), relay->candidate->profile->size()};
        relay->candidate->submitted = true;
        if (cudaMemcpyAsync(relay->candidate->profile->data(), step.profile.data(), sizeof(step.profile), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            selected.Restore(); Finish(slotIndex, false, true); return;
        }
        if (relay->candidate->width->ApplyFresh(relay->geometry, relay->hairT, parameters,
                relay->candidate->widths->view(), stream,
                reservation) != gpu::StyleStatus::Ok) {
            // The profile DMA already precedes this semantic rejection.  A
            // plain terminal callback proves that prefix without treating a
            // bad authored Width value as an unsafe workspace poison.
            if (relay->candidate->width->HasUnprovenWork()) {
                selected.Restore(); Finish(slotIndex, false, true); return;
            }
            relay->candidate->semanticRejected = true;
            if (failInstall[2].exchange(false, std::memory_order_acq_rel) ||
                cudaStreamAddCallback(stream, OperatorCallback, &slots[slotIndex].callback[2], 0) != cudaSuccess) {
                selected.Restore(); Finish(slotIndex, false, true); return;
            }
            selected.Restore(); slots[slotIndex].Return(2); return;
        }
        if (auto gate = std::atomic_load(&s_operatorAsyncGates[2]);
            gate && !gate->claimed.exchange(true, std::memory_order_acq_rel)) {
            relay->gates[2] = gate;
            slots[slotIndex].gate[2].store(gate.get(), std::memory_order_release);
        }
        if (auto gate = std::atomic_load(&s_operatorAsyncReturnGate);
            gate && !gate->claimed.exchange(true, std::memory_order_acq_rel)) {
            relay->returnGate = gate;
            slots[slotIndex].widthReturnGate.store(gate.get(), std::memory_order_release);
            gate->terminalState.store(&slots[slotIndex].state[2], std::memory_order_release);
        }
        if (failInstall[2].exchange(false, std::memory_order_acq_rel) ||
            relay->candidate->width->FinishFreshAsync(stream, OperatorCallback,
                &slots[slotIndex].callback[2]) != gpu::StyleStatus::Ok) {
            selected.Restore(); Finish(slotIndex, false, relay->candidate->width->HasUnprovenWork()); return;
        }
        if (relay->returnGate) {
            auto gate = relay->returnGate;
            gate->entered.store(true, std::memory_order_release);
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!gate->released.load(std::memory_order_acquire)) {
                if (std::chrono::steady_clock::now() >= deadline) { gate->timedOut.store(true, std::memory_order_release); break; }
                std::this_thread::yield();
            }
        }
        selected.Restore(); slots[slotIndex].Return(2); return;
    }
    if (relay->rbf) {
        auto& candidate = *relay->rbf;
        auto* const reservation = job->memoryReservation.get();
        auto failRbf = [&](bool unsafe) {
            selected.Restore(); Finish(slotIndex, false, unsafe); };
        auto installNext = [&](unsigned next) {
            if (!Install(slots[slotIndex], next, stream, OperatorCallback)) {
                failRbf(true); return false;
            }
            selected.Restore(); slots[slotIndex].Return(next); return true;
        };
        auto rbfUnsafe = [&] {
            return (candidate.state && (candidate.state->resources->surface.HasUnprovenFreshWork() ||
                candidate.state->resources->field.HasUnprovenWork())) ||
                (candidate.deformer && candidate.deformer->HasUnprovenWork()) ||
                (candidate.budgetPacket && candidate.budgetPacket->HasUnprovenWork());
        };
        if (phase == 2) {
            if (candidate.budgetPacket->CommitFreshFinish() != CudaParameterStatus::Ok) {
                failRbf(candidate.budgetPacket->HasUnprovenWork()); return;
            }
            if (auto const* value = candidate.budgetPacket->Find(TfToken("rbfSamples"))) {
                int raw = 0; std::memcpy(&raw, value->bytes.data(), sizeof(raw));
                if (raw < 4 || raw > 46336) { failRbf(false); return; }
                candidate.budget = static_cast<uint32_t>(raw);
            }
            if (!candidate.cache) { failRbf(false); return; }
            // Canonical empty geometry has no roots, CVs, samples, or
            // coefficients to publish.  Consume the operator in graph order
            // without perturbing the workspace's last accepted RBF cache.
            if (relay->geometry.pointCount == 0) {
                if (job->plan->geometryValueDag) {
                    auto const& input = job->state.geometryValues[step.inputNode];
                    auto& value = job->state.geometryValues[step.semanticNode];
                    value.geometry = input.geometry;
                    value.hairT = input.hairT;
                    value.rootPrim = input.rootPrim;
                    value.rootUV = input.rootUV;
                    value.curveTopology = input.curveTopology;
                    value.widthOrigin = input.widthOrigin;
                    value.widthOwnerNode = input.widthOwnerNode;
                    value.topologyOwnerNode = input.topologyOwnerNode;
                    value.namedOwnerNode = input.namedOwnerNode;
                    value.topologyOwners = input.topologyOwners;
                    value.namedOwners = input.namedOwners;
                    value.sourceFrameDomain = input.sourceFrameDomain;
                    value.sourceFramesPresent = input.sourceFramesPresent;
                    value.pointOrigin = input.pointOrigin;
                    value.deformed = true;
                    value.pointOwnerNode = step.semanticNode;
                    value.ready = true;
                    job->operatorFinished[relay->index] = 1;
                } else {
                    job->state.deformed = true;
                    ++job->nextOperator;
                }
                selected.Restore(); Finish(slotIndex, true, false); return;
            }
            // This is an ordinary worker launch point.  It is the only place
            // an old state is retired; final acceptance below merely parks it.
            candidate.cache->retiredState.reset();
            candidate.resetCounters = candidate.cache->state &&
                !SameRest(candidate.cache->key, step.surface.key);
            candidate.rebind = !candidate.cache->state ||
                !SameRest(candidate.cache->key, step.surface.key) ||
                candidate.cache->state->sampleBudget != candidate.budget;
            candidate.current = std::make_unique<gpu::DeviceBuffer<float3>>();
            candidate.points = std::make_unique<gpu::DeviceBuffer<float3>>();
            candidate.deformer = std::make_unique<gpu::CudaRbfCurveDeformer>();
            if (candidate.current->reset(step.surface.currentPoints.size(), reservation,
                    UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
                candidate.points->reset(relay->geometry.pointCount, reservation) != cudaSuccess) {
                failRbf(false); return;
            }
            UsdGenParamView values; values.desc = &job->plan->desc;
            values.node = &job->plan->desc.nodes[step.semanticNode];
            auto field = [&](char const* name, float fallback) {
                if (auto const* e = runtime.parameters.Find(TfToken(name)))
                    return gpu::ScalarField::Device({static_cast<float const*>(e->data), e->count}, e->domain);
                return gpu::ScalarField::Literal(static_cast<float>(values.GetDouble(TfToken(name), fallback)));
            };
            auto boolean = [&](char const* name, bool fallback) {
                if (auto const* e = runtime.parameters.Find(TfToken(name)))
                    return gpu::BoolField::Device({static_cast<uint8_t const*>(e->data), e->count}, e->domain);
                return gpu::BoolField::Literal(values.GetBool(TfToken(name), fallback));
            };
            candidate.parameters.mask = field("mask", 1);
            candidate.parameters.enabled = boolean("enabled", values.node->enabled);
            candidate.parameters.lockRoots = boolean("lockRoots", true);
            if (candidate.rebind) {
                candidate.freshState = std::make_unique<CudaRbfCache::State>();
                candidate.freshState->device = execution.device;
                candidate.freshState->sampleBudget = candidate.budget;
                candidate.state = candidate.freshState.get();
                candidate.rest = std::make_unique<gpu::DeviceBuffer<float3>>();
                candidate.offsets = std::make_unique<gpu::DeviceBuffer<uint32_t>>();
                candidate.indices = std::make_unique<gpu::DeviceBuffer<uint32_t>>();
                if (!Upload(*candidate.rest, step.surface.restPoints, stream, reservation) ||
                    !Upload(*candidate.offsets, step.surface.faceOffsets, stream, reservation) ||
                    !Upload(*candidate.indices, step.surface.faceVertexIndices, stream, reservation) ||
                    candidate.state->resources->surface.BeginFreshBind(
                        {candidate.rest->data(), candidate.rest->size()},
                        {candidate.offsets->data(), candidate.offsets->size()},
                        candidate.offsets->size() - 1,
                        {candidate.indices->data(), candidate.indices->size()}, candidate.budget, stream,
                        reservation) !=
                        gpu::SurfaceBindingStatus::Ok) { failRbf(true); return; }
                if (!installNext(3)) return;
                return;
            }
            candidate.state = candidate.cache->state.get();
            if (!Upload(*candidate.current, step.surface.currentPoints, stream, reservation) ||
                candidate.state->resources->surface.BeginFreshUpdate(
                    {candidate.current->data(), candidate.current->size()}, relay->rootPrim,
                    relay->rootUV, stream, reservation) != gpu::SurfaceBindingStatus::Ok) { failRbf(true); return; }
            if (!installNext(7)) return;
            return;
        }
        if (phase == 3) {
            if (candidate.state->resources->surface.CommitFreshBind() != gpu::SurfaceBindingStatus::Ok ||
                candidate.state->resources->field.BeginFreshBind(candidate.state->resources->surface.restSamples(), 0., stream, reservation) != gpu::RbfStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            if (!installNext(4)) return;
            return;
        }
        if (phase == 4) {
            if (candidate.state->resources->field.CommitFreshBindExtent() != gpu::RbfStatus::Ok ||
                candidate.state->resources->field.BeginFreshBindRank(stream) != gpu::RbfStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            if (!installNext(5)) return;
            return;
        }
        if (phase == 5) {
            if (candidate.state->resources->field.CommitFreshBindRank() != gpu::RbfStatus::Ok ||
                candidate.state->resources->field.BeginFreshBindLu(stream) != gpu::RbfStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            if (!installNext(6)) return;
            return;
        }
        if (phase == 6) {
            if (candidate.state->resources->field.CommitFreshBindLu() != gpu::RbfStatus::Ok ||
                !Upload(*candidate.current, step.surface.currentPoints, stream, reservation) ||
                candidate.state->resources->surface.BeginFreshUpdate(
                    {candidate.current->data(), candidate.current->size()}, relay->rootPrim,
                    relay->rootUV, stream, reservation) != gpu::SurfaceBindingStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            if (!installNext(7)) return;
            return;
        }
        if (phase == 7) {
            if (candidate.state->resources->surface.CommitFreshUpdate() != gpu::SurfaceBindingStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            candidate.surfaceCommitted = true;
            if (candidate.state->resources->field.BeginFreshSolve(
                    candidate.state->resources->surface.currentSamples(), stream, reservation) != gpu::RbfStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            if (!installNext(8)) return;
            return;
        }
        if (phase == 8) {
            if (candidate.state->resources->field.CommitFreshSolveInput() != gpu::RbfStatus::Ok ||
                candidate.state->resources->field.BeginFreshSolveFactors(stream) != gpu::RbfStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            if (!installNext(9)) return;
            return;
        }
        if (phase == 9) {
            if (candidate.state->resources->field.CommitFreshSolve() != gpu::RbfStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            candidate.solveCommitted = true;
            if (candidate.deformer->BeginFreshShape(relay->geometry,
                    candidate.state->resources->surface.rootTargets(), candidate.parameters,
                    candidate.points->view(), stream, reservation) != gpu::RbfStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            if (!installNext(10)) return;
            return;
        }
        if (phase == 10) {
            if (candidate.deformer->CommitFreshShape() != gpu::RbfStatus::Ok ||
                candidate.deformer->BeginFreshEvaluate(candidate.state->resources->field, stream, reservation) != gpu::RbfStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            if (!installNext(11)) return;
            return;
        }
        if (phase == 11) {
            if (candidate.deformer->CommitFreshEvaluate(candidate.state->resources->field) != gpu::RbfStatus::Ok ||
                candidate.deformer->BeginFreshApply(stream) != gpu::RbfStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            if (!installNext(12)) return;
            return;
        }
        if (phase == 12) {
            if (candidate.deformer->CommitFreshApply() != gpu::RbfStatus::Ok ||
                candidate.deformer->BeginFreshCopy(stream) != gpu::RbfStatus::Ok) {
                failRbf(rbfUnsafe()); return;
            }
            if (!installNext(13)) return;
            return;
        }
        if (phase == 13) {
            if (candidate.deformer->CommitFreshFinish() != gpu::RbfStatus::Ok ||
                candidate.points->recordUse(stream) != cudaSuccess) {
                failRbf(rbfUnsafe()); return;
            }
            if (relay->index >= job->state.pendingRbf.size() || job->state.pendingRbf[relay->index]) {
                selected.Restore(); Finish(slotIndex, false, true); return;
            }
            auto publication = std::make_unique<CudaRbfPendingPublication>();
            publication->cache = candidate.cache;
            publication->freshState = std::move(candidate.freshState);
            publication->state = publication->freshState ? publication->freshState.get() : candidate.state;
            publication->key = step.surface.key;
            publication->rebind = candidate.rebind;
            publication->resetCounters = candidate.resetCounters;
            publication->solveCommitted = candidate.solveCommitted;
            publication->surfaceCommitted = candidate.surfaceCommitted;
            job->state.pendingRbf[relay->index] = std::move(publication);
            candidate.solveCommitted = false;
            candidate.surfaceCommitted = false;
            if (job->plan->geometryValueDag) {
                auto const node = step.semanticNode;
                auto const& input = job->state.geometryValues[step.inputNode];
                auto& value = job->state.geometryValues[node];
                value.geometry = input.geometry;
                value.geometry.points = {candidate.points->data(), candidate.points->size()};
                value.hairT = input.hairT;
                value.rootPrim = input.rootPrim;
                value.rootUV = input.rootUV;
                value.curveTopology = input.curveTopology;
                value.deformed = true;
                value.topologyOwnerNode = input.topologyOwnerNode;
                value.topologyOwners = input.topologyOwners;
                value.namedOwnerNode = input.namedOwnerNode;
                value.namedOwners = input.namedOwners;
                value.sourceFrameDomain = input.sourceFrameDomain;
                value.sourceFramesPresent = input.sourceFramesPresent;
                value.points = std::move(candidate.points);
                value.pointOrigin = value.geometry.points;
                value.widthOrigin = input.geometry.widths;
                value.pointOwnerNode = node;
                value.widthOwnerNode = input.widthOwnerNode;
                value.ready = true;
                job->operatorFinished[relay->index] = 1;
            } else {
                job->state.finalPoints = std::move(candidate.points);
                job->state.geometry.points = {job->state.finalPoints->data(), job->state.finalPoints->size()};
                job->state.deformed = true;
                ++job->nextOperator;
            }
            selected.Restore(); Finish(slotIndex, true, false); return;
        }
        failRbf(false); return;
    }
    if (phase == 2 && relay->length) {
        if (relay->length->semanticRejected) { selected.Restore(); Finish(slotIndex, false, false); return; }
        if (relay->length->length->CommitFreshFinish() != gpu::StyleStatus::Ok) { selected.Restore(); Finish(slotIndex, false, relay->length->length->HasUnprovenWork()); return; }
        auto changed = relay->geometry; changed.points = {relay->length->changedPoints->data(), relay->length->changedPoints->size()};
        if (relay->length->compaction->ApplyFreshCounts(changed, relay->hairT,
                relay->rootPrim,
                relay->rootUV, {relay->length->keep->data(), relay->length->keep->size()}, stream,
                job->memoryReservation.get(), relay->frames) != gpu::CurveCompactionStatus::Ok) { selected.Restore(); Finish(slotIndex, false, relay->length->compaction->HasUnprovenWork()); return; }
        if (!InstallGate(slots[slotIndex], 3, stream) || failInstall[3].exchange(false, std::memory_order_acq_rel) || relay->length->compaction->FinishFreshCountsAsync(stream, OperatorCallback, &slots[slotIndex].callback[3]) != gpu::CurveCompactionStatus::Ok) { selected.Restore(); Finish(slotIndex, false, true); return; }
        selected.Restore(); slots[slotIndex].Return(3); return;
    }
    if (phase == 3 && relay->length) {
        if (relay->length->compaction->CommitFreshCounts() != gpu::CurveCompactionStatus::Ok) { selected.Restore(); Finish(slotIndex, false, relay->length->compaction->HasUnprovenWork()); return; }
        if (relay->length->compaction->ApplyFreshScatter(stream, job->memoryReservation.get()) != gpu::CurveCompactionStatus::Ok) { selected.Restore(); Finish(slotIndex, false, relay->length->compaction->HasUnprovenWork()); return; }
        if (!InstallGate(slots[slotIndex], 4, stream) || failInstall[4].exchange(false, std::memory_order_acq_rel) ||
            relay->length->compaction->FinishFreshScatterAsync(stream, OperatorCallback, &slots[slotIndex].callback[4]) != gpu::CurveCompactionStatus::Ok) { selected.Restore(); Finish(slotIndex, false, true); return; }
        selected.Restore(); slots[slotIndex].Return(4); return;
    }
    if (phase == 4 && relay->length) {
        if (relay->length->compaction->CommitFreshFinish() != gpu::CurveCompactionStatus::Ok) { selected.Restore(); Finish(slotIndex, false, relay->length->compaction->HasUnprovenWork()); return; }
        if (relay->length->compaction->recordUse(stream) != cudaSuccess) { selected.Restore(); Finish(slotIndex, false, false); return; }
        auto const* lengthNames = relay->namedOwners ? &relay->namedOwners->planes :
            &job->state.namedChannels;
        if (!lengthNames->empty()) {
            std::vector<gpu::CudaNamedChannelTopologyCandidate::RawInputPlane> inputs;
            inputs.reserve(lengthNames->size());
            for (auto const& plane : *lengthNames) {
                if (!plane.bytes || plane.bytes->waitOn(stream) != cudaSuccess) {
                    selected.Restore(); Finish(slotIndex, false, true); return;
                }
                inputs.push_back({plane.metadata,
                    {plane.bytes->data(), plane.bytes->size()}, plane.bytes.get()});
            }
            relay->length->namedTopology =
                std::make_unique<gpu::CudaNamedChannelTopologyCandidate>();
            std::string reason;
            // The accepted job state owns every raw input until this relay
            // either proves and commits phase 5 or is quarantined.  Neither
            // topology nor named storage is visible before that proof.
            if (relay->length->namedTopology->BeginRaw(
                    relay->geometry, relay->length->compaction->view(),
                    std::move(inputs), stream,
                    gpu::CudaNamedChannelTopologyMode::Compaction,
                    std::shared_ptr<const void>(job), &reason,
                    job->memoryReservation.get()) != cudaSuccess) {
                if (job->diagnostics && !reason.empty()) job->diagnostics->Error(reason);
                selected.Restore(); Finish(slotIndex, false, true); return;
            }
            if (!InstallGate(slots[slotIndex], 5, stream) ||
                failInstall[5].exchange(false, std::memory_order_acq_rel) ||
                relay->length->namedTopology->FinishAsync(
                    OperatorCallback, &slots[slotIndex].callback[5]) != cudaSuccess) {
                relay->length->namedTopology->Quarantine();
                selected.Restore(); Finish(slotIndex, false, true); return;
            }
            selected.Restore(); slots[slotIndex].Return(5); return;
        }
    }
    if ((phase == 2 || phase == 3) && step.type == TfToken("UsdGenGrow")) {
        if (job->plan->geometryValueDag) {
            if (!relay->grow || !relay->grow->grow) {
                selected.Restore(); Finish(slotIndex, false, false); return;
            }
            if (phase == 2) {
                if (relay->grow->grow->CommitFreshFinish() != gpu::CurveGrowStatus::Ok) {
                    selected.Restore(); Finish(slotIndex, false,
                        relay->grow->grow->HasUnprovenWork()); return;
                }
                if (!job->state.BeginGrowNamedValue(relay->geometry, relay->grow.get(), stream,
                        job, relay->namedOwners, job->diagnostics)) {
                    bool const unsafe = relay->grow->namedTopology &&
                        relay->grow->namedTopology->HasUnprovenWork();
                    selected.Restore(); Finish(slotIndex, false, unsafe); return;
                }
                if (relay->grow->namedTopology) {
                    if (!InstallGate(slots[slotIndex], 3, stream) ||
                        failInstall[3].exchange(false, std::memory_order_acq_rel) ||
                        relay->grow->namedTopology->FinishAsync(OperatorCallback,
                            &slots[slotIndex].callback[3]) != cudaSuccess) {
                        selected.Restore(); Finish(slotIndex, false, true); return;
                    }
                    selected.Restore(); slots[slotIndex].Return(3); return;
                }
            }
            if (!job->state.AcceptCurveGrowValue(*job->plan, step.semanticNode,
                    step.inputNode, relay->grow.get(), relay->namedOwners, job->diagnostics)) {
                bool const unsafe = (relay->grow->grow && relay->grow->grow->HasUnprovenWork()) ||
                    (relay->grow->namedTopology && relay->grow->namedTopology->HasUnprovenWork());
                selected.Restore(); Finish(slotIndex, false, unsafe); return;
            }
            job->operatorFinished[relay->index] = 1;
            selected.Restore(); Finish(slotIndex, true, false); return;
        }
        if (phase == 2) {
            if (job->state.curveGrow->CommitFreshFinish() != gpu::CurveGrowStatus::Ok) {
                selected.Restore(); Finish(slotIndex, false, job->state.HasUnprovenAsyncWork()); return;
            }
            if (!job->state.BeginGrowNamed(stream, job, relay->namedOwners, job->diagnostics)) {
                selected.Restore(); Finish(slotIndex, false, job->state.HasUnprovenAsyncWork()); return;
            }
            if (job->state.growNamedTopology) {
                if (!InstallGate(slots[slotIndex], 3, stream) ||
                    failInstall[3].exchange(false, std::memory_order_acq_rel) ||
                    job->state.growNamedTopology->FinishAsync(OperatorCallback,
                        &slots[slotIndex].callback[3]) != cudaSuccess) {
                    selected.Restore(); Finish(slotIndex, false, true); return;
                }
                selected.Restore(); slots[slotIndex].Return(3); return;
            }
        }
        if (!job->state.AcceptCurveGrow(*job->plan, step.semanticNode, cudaSuccess,
                relay->namedOwners, job->diagnostics)) {
            selected.Restore(); Finish(slotIndex, false, job->state.HasUnprovenAsyncWork()); return;
        }
        if (!job->plan->taskDag) ++job->nextOperator;
        else job->operatorFinished[relay->index] = 1;
        selected.Restore(); Finish(slotIndex, true, false); return;
    }
    if (phase == 2 && relay->noise) {
        if (relay->noise->semanticRejected) {
            selected.Restore(); Finish(slotIndex, false, false); return;
        }
        if (relay->noise->noise->CommitFreshFinish() != gpu::StyleStatus::Ok ||
            relay->noise->points->recordUse(stream) != cudaSuccess) {
            selected.Restore(); Finish(slotIndex, false,
                relay->noise->noise->HasUnprovenWork()); return;
        }
        if (job->plan->geometryValueDag) {
            auto const node = job->plan->steps[relay->index]->semanticNode;
            auto& value = job->state.geometryValues[node];
            auto const& inputValue = job->state.geometryValues[
                job->plan->steps[relay->index]->inputNode];
            value.geometry = inputValue.geometry;
            value.hairT = inputValue.hairT;
            value.rootPrim = inputValue.rootPrim;
            value.rootUV = inputValue.rootUV;
            value.curveTopology = inputValue.curveTopology;
            value.deformed = inputValue.deformed;
            value.topologyOwnerNode = inputValue.topologyOwnerNode;
            value.topologyOwners = inputValue.topologyOwners;
            value.namedOwnerNode = inputValue.namedOwnerNode;
            value.namedOwners = inputValue.namedOwners;
            value.sourceFrameDomain = inputValue.sourceFrameDomain;
            value.sourceFramesPresent = inputValue.sourceFramesPresent;
            value.points = std::move(relay->noise->points);
            value.pointOrigin = {value.points->data(), value.points->size()};
            value.widthOrigin = relay->geometry.widths;
            value.pointOwnerNode = node;
            value.widthOwnerNode = job->state.geometryValues[
                job->plan->steps[relay->index]->inputNode].widthOwnerNode;
            value.ready = true;
        } else {
            relay->noise->priorPoints = std::move(job->state.finalPoints);
            job->state.finalPoints = std::move(relay->noise->points);
            job->state.geometry.points = {
                job->state.finalPoints->data(), job->state.finalPoints->size()};
        }
        if (!job->plan->taskDag) ++job->nextOperator;
        else job->operatorFinished[relay->index] = 1;
        selected.Restore(); Finish(slotIndex, true, false); return;
    }
    if ((phase == 4 || phase == 5) && relay->length) {
        if (phase == 5) {
            std::string reason;
            auto const status = static_cast<cudaError_t>(
                slots[slotIndex].status[phase].load(std::memory_order_acquire));
            if (!relay->length->namedTopology ||
                !relay->length->namedTopology->CommitPlanes(status,
                    &relay->length->transformedNamedChannels, &reason)) {
                if (job->diagnostics && !reason.empty()) job->diagnostics->Error(reason);
                bool const unsafe = relay->length->namedTopology &&
                    relay->length->namedTopology->HasUnprovenWork();
                selected.Restore(); Finish(slotIndex, false, unsafe); return;
            }
        }
        if (!job->plan->geometryValueDag) {
            relay->length->priorCompacted = std::move(job->state.compacted);
            relay->length->priorPoints = std::move(job->state.finalPoints);
            relay->length->priorWidths = std::move(job->state.finalWidths);
            job->state.compacted = std::move(relay->length->compaction);
            job->state.geometry = job->state.compacted->view();
            job->state.hairT = job->state.compacted->hairT();
            job->state.rootPrim = job->state.compacted->rootPrim();
            job->state.rootUV = job->state.compacted->rootUV();
        }
        if (phase == 5 && !job->plan->geometryValueDag) {
            job->state.PreserveSourceNamedChannels(*job->plan);
            job->state.namedChannels =
                std::move(relay->length->transformedNamedChannels);
        }
        if (job->plan->taskDag) {
            auto const node = job->plan->steps[relay->index]->semanticNode;
            if (node >= job->state.widthValues.size() ||
                node >= job->state.geometryValues.size()) {
                selected.Restore(); Finish(slotIndex, false, false); return;
            }
            if (job->plan->geometryValueDag) {
                if (job->state.topologyOwners.size() != job->plan->desc.nodes.size() ||
                    job->state.namedOwners.size() != job->plan->desc.nodes.size()) {
                    selected.Restore(); Finish(slotIndex, false, false); return;
                }
                auto topology = std::make_shared<TopologyOwners>();
                topology->compaction = std::move(relay->length->compaction);
                job->state.topologyOwners[node] = topology;
                auto names = std::make_shared<NamedOwners>();
                if (phase == 5) names->planes = std::move(relay->length->transformedNamedChannels);
                else if (relay->namedOwners) names = relay->namedOwners;
                job->state.namedOwners[node] = std::move(names);
            }
            auto const outputGeometry = job->plan->geometryValueDag
                ? job->state.topologyOwners[node]->compaction->view()
                : job->state.geometry;
            auto const outputHairT = job->plan->geometryValueDag
                ? job->state.topologyOwners[node]->compaction->hairT() : job->state.hairT;
            auto const outputRootPrim = job->plan->geometryValueDag
                ? job->state.topologyOwners[node]->compaction->rootPrim() : job->state.rootPrim;
            auto const outputRootUV = job->plan->geometryValueDag
                ? job->state.topologyOwners[node]->compaction->rootUV() : job->state.rootUV;
            job->state.widthValues[node].view = outputGeometry.widths;
            auto& value = job->state.geometryValues[node];
            auto const& inputValue = job->state.geometryValues[
                job->plan->steps[relay->index]->inputNode];
            value.geometry = outputGeometry;
            value.hairT = outputHairT;
            value.rootPrim = outputRootPrim;
            value.rootUV = outputRootUV;
            value.curveTopology = job->plan->geometryValueDag
                ? inputValue.curveTopology : job->state.curveTopology;
            value.deformed = job->plan->geometryValueDag
                ? inputValue.deformed : job->state.deformed;
            value.pointOrigin = outputGeometry.points;
            value.widthOrigin = outputGeometry.widths;
            value.pointOwnerNode = node;
            value.widthOwnerNode = node;
            value.topologyOwnerNode = node;
            value.namedOwnerNode = node;
            value.topologyOwners = job->plan->geometryValueDag ? job->state.topologyOwners[node] : nullptr;
            value.namedOwners = job->plan->geometryValueDag ? job->state.namedOwners[node] : nullptr;
            value.sourceFrameDomain = inputValue.sourceFrameDomain;
            value.sourceFramesPresent = inputValue.sourceFramesPresent;
            value.ready = true;
            job->operatorFinished[relay->index] = 1;
        } else {
            ++job->nextOperator;
        }
        selected.Restore(); Finish(slotIndex, true, false); return;
    }
    if (!relay->candidate || relay->candidate->semanticRejected) {
        selected.Restore(); Finish(slotIndex, false, false); return;
    }
    if (!relay->candidate->widthBlend &&
        relay->candidate->width->CommitFreshFinish() != gpu::StyleStatus::Ok) {
        selected.Restore(); Finish(slotIndex, false, relay->candidate && relay->candidate->width->HasUnprovenWork()); return;
    }
    if (relay->candidate->widths->recordUse(stream) != cudaSuccess) { selected.Restore(); Finish(slotIndex, false, false); return; }
    if (job->plan->taskDag) {
        auto const node = job->plan->steps[relay->index]->semanticNode;
        auto& value = job->state.widthValues[node];
        value.owner = std::move(relay->candidate->widths);
        value.view = {value.owner->data(), value.owner->size()};
        if (job->plan->geometryValueDag) {
            auto& geometryValue = job->state.geometryValues[node];
            auto const& inputSnapshot = job->state.geometryValues[
                job->plan->steps[relay->index]->inputNode];
            geometryValue.geometry = inputSnapshot.geometry;
            geometryValue.hairT = inputSnapshot.hairT;
            geometryValue.rootPrim = inputSnapshot.rootPrim;
            geometryValue.rootUV = inputSnapshot.rootUV;
            geometryValue.curveTopology = inputSnapshot.curveTopology;
            geometryValue.deformed = inputSnapshot.deformed;
            geometryValue.topologyOwnerNode = inputSnapshot.topologyOwnerNode;
            geometryValue.topologyOwners = inputSnapshot.topologyOwners;
            geometryValue.namedOwnerNode = inputSnapshot.namedOwnerNode;
            geometryValue.namedOwners = inputSnapshot.namedOwners;
            geometryValue.sourceFrameDomain = inputSnapshot.sourceFrameDomain;
            geometryValue.sourceFramesPresent = inputSnapshot.sourceFramesPresent;
            geometryValue.pointOrigin = relay->geometry.points;
            geometryValue.widthOrigin = value.view;
            auto const& inputValue = job->state.geometryValues[
                job->plan->steps[relay->index]->inputNode];
            geometryValue.pointOwnerNode = inputValue.pointOwnerNode;
            geometryValue.widthOwnerNode = node;
            geometryValue.ready = true;
        }
    } else {
        relay->candidate->priorWidths = std::move(job->state.finalWidths);
        job->state.finalWidths = std::move(relay->candidate->widths);
        job->state.geometry.widths = {job->state.finalWidths->data(), job->state.finalWidths->size()};
    }
    if (!job->plan->taskDag) ++job->nextOperator;
    else job->operatorFinished[relay->index] = 1;
    selected.Restore(); Finish(slotIndex, true, false);
}

bool ExecuteCudaJobOperatorAsync(std::shared_ptr<UsdGenCudaExecutionJob> job, size_t index,
                                 std::function<void(bool)> completion) {
    if (job && job->finalAsyncInFlight) return false;
    if (!job || !completion || job->failed.load(std::memory_order_acquire) || job->sourceAsyncInFlight || !job->sourceDone || job->finalDone ||
        !job->plan || !job->workspace || index >= job->plan->steps.size()) return false;
#ifndef USDGEN_ENABLE_CUDA
    return false;
#else
    bool const taskDag = job->plan->taskDag;
    if ((!taskDag && (job->operatorAsyncInFlight.load(std::memory_order_acquire) ||
                       index != job->nextOperator)) ||
        (taskDag && job->operatorStarted[index])) return false;
    if (job->plan->steps[index]->type != TfToken("UsdGenWidth") &&
        job->plan->steps[index]->type != TfToken("UsdGenWidthBlend") &&
        job->plan->steps[index]->type != TfToken("UsdGenLength") &&
        job->plan->steps[index]->type != TfToken("UsdGenGrow") &&
        job->plan->steps[index]->type != TfToken("UsdGenDeform") &&
        job->plan->steps[index]->type != TfToken("UsdGenNoise")) {
        bool ok = ExecuteCudaJobOperator(*job, index);
        try { completion(ok); } catch (...) {}
        return true;
    }
    auto& execution = *job->workspace->impl_;
    if (execution.poisoned.load(std::memory_order_acquire)) return false;
    bool const widthBranch = taskDag &&
        IsCudaWidthValueNode(job->plan->steps[index]->type);
    auto const stream = widthBranch
        ? execution.widthStreams[index % execution.widthStreams.size()]
        : execution.stream;
    if (widthBranch)
        s_operatorAsyncWidthBranchStreamMask.fetch_or(
            uint64_t{1} << (index % execution.widthStreams.size()),
            std::memory_order_acq_rel);
    CudaDeviceScope selected(execution.device);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (!selected.selected || cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return false;
    gpu::DeviceCurveGeometryView geometry;
    gpu::DeviceView<const float> rightWidths;
    if (!job->state.GetOperatorInput(*job->plan, index, stream, &geometry,
                                     &rightWidths, job->diagnostics)) return false;
    bool const noiseStage = job->plan->steps[index]->type == TfToken("UsdGenNoise");
    if (noiseStage && job->noiseAsyncInFlight.exchange(true, std::memory_order_acq_rel))
        return false;
    bool const geometryStage = job->plan->geometryValueDag &&
        (noiseStage || job->plan->steps[index]->type == TfToken("UsdGenLength") ||
         job->plan->steps[index]->type == TfToken("UsdGenGrow") ||
         job->plan->steps[index]->type == TfToken("UsdGenDeform"));
    if (geometryStage && job->geometryAsyncInFlight.exchange(true,
            std::memory_order_acq_rel)) {
        if (noiseStage) job->noiseAsyncInFlight.store(false, std::memory_order_release);
        return false;
    }
    auto* relay = OperatorRelays().Reserve(job, index, std::move(completion));
    if (!relay) {
        if (geometryStage) job->geometryAsyncInFlight.store(false, std::memory_order_release);
        if (noiseStage) job->noiseAsyncInFlight.store(false, std::memory_order_release);
        return false;
    }
    size_t const slot = relay->slot;
    job->operatorStarted[index] = 1;
    relay->geometryGuard = geometryStage;
    relay->geometry = geometry;
    relay->rightWidths = rightWidths;
    if (job->plan->geometryValueDag) {
        auto const& inputValue = job->state.geometryValues[
            job->plan->steps[index]->inputNode];
        relay->hairT = inputValue.hairT;
        relay->rootPrim = inputValue.rootPrim;
        relay->rootUV = inputValue.rootUV;
        relay->namedOwners = inputValue.namedOwners;
        if (auto const& owners = inputValue.topologyOwners) {
            if (owners->compaction) relay->frames = owners->compaction->frames();
            else if (owners->grow) relay->frames = {owners->grow->rootT(), owners->grow->rootB(),
                owners->grow->rootN(), relay->geometry.stableIds};
            else if (owners->scatterGrow) relay->frames = {
                owners->scatterGrow->rootT(), owners->scatterGrow->rootB(),
                owners->scatterGrow->rootN(), relay->geometry.stableIds};
        }
    } else {
        relay->hairT = job->state.hairT;
        relay->rootPrim = job->state.rootPrim;
        relay->rootUV = job->state.rootUV;
        relay->frames = job->state.CompactionFrames();
    }
    relay->stream = stream;
    job->operatorAsyncInFlight.fetch_add(1, std::memory_order_acq_rel);
    if (widthBranch) {
        if (auto gate = std::atomic_load(&s_operatorAsyncWidthBranchGate);
            gate && gate->Claim()) {
            // This is intentionally a launcher-side rendezvous. CUDA may
            // serialize host callbacks from distinct streams, so blocking a
            // callback cannot prove stream independence. At this point both
            // semantic tasks own distinct relays and selected branch streams,
            // but neither has enqueued mutable evaluator work yet.
            relay->widthBranchGate = gate;
            gate->arrivals.fetch_add(1, std::memory_order_acq_rel);
            gate->entered.store(true, std::memory_order_release);
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!gate->released.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            if (!gate->released.load(std::memory_order_acquire)) {
                gate->timedOut.store(true, std::memory_order_release);
                selected.Restore(); OperatorRelays().Finish(slot, false, false);
                return true;
            }
        }
    }
    try {
    relay->poison = [workspace = job->workspace] { workspace->impl_->poisoned.store(true, std::memory_order_release); };
    auto const& step = *job->plan->steps[index];
    if (widthBranch) {
        if (auto witness = std::atomic_load(&s_operatorAsyncWidthOverlapWitness)) {
            // First device command admitted after the launcher rendezvous on
            // each legal Width branch. It is separate from every COW output.
            relay->overlapWitness = witness;
            if (!witness->Submit(step.semanticNode,
                    static_cast<uint32_t>(index % kCudaWidthBranchStreamCount),
                    stream) && witness->error.load(std::memory_order_acquire)) {
                selected.Restore(); OperatorRelays().Finish(slot, false, true);
                return true;
            }
        }
    }
    auto& runtime = *execution.steps.at(step.path);
    expr::Context context; context.frame = job->frame; context.time = job->frame / job->plan->desc.timeCodesPerSecond;
    auto const& op = job->plan->desc.nodes[step.semanticNode]; context.seed = op.seed;
    context.descId = expr::DescriptionId(job->plan->desc.description.GetText());
    if (runtime.parameters.BeginFreshContexts(step.parameters, relay->geometry,
            ExpressionChannels(relay->geometry, relay->hairT, relay->rootUV,
                               relay->rootPrim, relay->frames),
            context, stream, nullptr,
            job->memoryReservation.get(),
            UsdGenExecutionResourceKind::Cache) != CudaParameterStatus::Ok ||
        runtime.parameters.EnqueueFreshContextStatus(stream) != CudaParameterStatus::Ok) {
        bool unsafe = runtime.parameters.HasUnprovenWork();
        selected.Restore(); OperatorRelays().Finish(slot, false, unsafe);
        return true;
    }
    if (!OperatorRelays().Install(OperatorRelays().slots[slot], 0, stream, OperatorCallback)) {
        selected.Restore(); OperatorRelays().Finish(slot, false, true); return true;
    }
    selected.Restore(); OperatorRelays().slots[slot].Return(0);
    return true;
    } catch (...) {
        selected.Restore(); OperatorRelays().Finish(slot, false, true);
        return true;
    }
#endif
}
#endif

#ifndef USDGEN_ENABLE_CUDA
bool ExecuteCudaJobOperatorAsync(std::shared_ptr<UsdGenCudaExecutionJob>, size_t,
                                 std::function<void(bool)>) { return false; }
#endif

#ifdef USDGEN_ENABLE_CUDA
namespace {
constexpr size_t kFinalizationRelayCapacity = 1024;
constexpr int kFinalizationRelayWorkers = 8;
constexpr unsigned kFinalizationRelayPhases = 3;

// Finalization owns pinned D2H targets itself.  It is deliberately retained
// with its slot after an unproven terminal so cudaFreeHost is never attempted
// while the stream may still reference one of these addresses.
template <class T> struct FinalPinned {
    T* data = nullptr;
    UsdGenExecutionResourcePermit permit;
    bool Allocate(size_t count,
                  UsdGenExecutionMemoryReservation* reservation = nullptr) noexcept {
        if (!count) return true;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
        auto charge = gpu::TryReserveCudaExecutionBytes(count * sizeof(T),
            UsdGenExecutionResourceKind::Scratch, reservation);
        if (!charge) return false;
        T* allocation = nullptr;
        if (cudaHostAlloc(reinterpret_cast<void**>(&allocation), count * sizeof(T), cudaHostAllocDefault) != cudaSuccess) {
            // Defensively retain the charge until a non-null partial result is
            // either proved freed or deliberately abandoned.
            if (allocation) {
                data = allocation; permit = std::move(*charge);
                if (cudaFreeHost(data) == cudaSuccess) { data = nullptr; permit.Release(); }
                else { data = nullptr; permit.Abandon(); }
            }
            return false;
        }
        data = allocation; permit = std::move(*charge); return true;
    }
    void Abandon() noexcept { data = nullptr; permit.Abandon(); }
    ~FinalPinned() { if (data) { if (cudaFreeHost(data) == cudaSuccess) permit.Release(); else permit.Abandon(); } }
};
struct FinalizationCandidate {
    gpu::CurveTileRequirements requirements;
    gpu::DeviceBuffer<gpu::CurveTileSpan> deviceTiles;
    gpu::DeviceBuffer<uint32_t> tileStatus, boundsStatus;
    gpu::DeviceBuffer<float3> tileMinimums, tileMaximums;
    gpu::DeviceBuffer<gpu::CurveTileBoundsScratch> boundsScratch;
    FinalPinned<uint32_t> tileHostStatus, boundsHostStatus;
    FinalPinned<gpu::CurveTileSpan> spans;
    FinalPinned<float3> minimums, maximums;
    std::vector<UsdGenDeviceTileMetadata> tiles;
    gpu::CudaGeometryLease previousLease;
    std::unique_ptr<gpu::CudaCurveTopologyCompare> topology;
    bool topologyActive = false;
    bool submitted = false;
};
struct FinalizationRelay {
    std::shared_ptr<UsdGenCudaExecutionJob> job;
    std::function<void(std::shared_ptr<const UsdGenDeviceGeneration>)> done;
    std::unique_ptr<FinalizationCandidate> candidate;
    // A worker-side publication can already own CUDA retirement state when a
    // later cleanup device selection fails.  Retain it with the poisoned slot
    // instead of letting a local result destructor issue backend work.
    std::shared_ptr<const UsdGenDeviceGeneration> published;
    std::function<void()> poison;
    std::array<std::shared_ptr<SourceAsyncTestGate>, kFinalizationRelayPhases> gates;
    std::shared_ptr<FinalizationReturnGate> returnGate;
    std::atomic<bool> quarantine{false};
    size_t slot = std::numeric_limits<size_t>::max();
};
std::array<std::shared_ptr<SourceAsyncTestGate>, kFinalizationRelayPhases> s_finalizationGates;
std::shared_ptr<FinalizationReturnGate> s_finalizationReturnGate;
}

struct FinalizationRelayService {
    struct Slot {
        struct CallbackIdentity { Slot* slot = nullptr; unsigned phase = 0; };
        std::atomic<bool> occupied{false}, quarantined{false};
        std::shared_ptr<FinalizationRelay> relay;
        FinalizationRelayService* service = nullptr;
        size_t index = 0;
        std::array<std::atomic<unsigned>, kFinalizationRelayPhases> state{};
        std::array<std::atomic<int>, kFinalizationRelayPhases> status{};
        std::atomic<FinalizationReturnGate*> metadataReturnGate{nullptr};
        std::array<CallbackIdentity, kFinalizationRelayPhases> callback{};
        void Signal(unsigned, cudaError_t) noexcept;
        void Return(unsigned) noexcept;
    };
    std::array<Slot, kFinalizationRelayCapacity> slots;
    tbb::task_arena arena{kFinalizationRelayWorkers, 0};
    std::unique_ptr<tbb::flow::graph> graph;
    std::unique_ptr<tbb::flow::function_node<size_t>> worker;
    std::atomic<size_t> capacity{kFinalizationRelayCapacity};
    std::atomic<bool> failAllocation{false}, failTopologyAllocation{false};
    std::array<std::atomic<bool>, kFinalizationRelayPhases> failInstall{}, failNative{};

    FinalizationRelayService() {
        arena.execute([this] { graph = std::make_unique<tbb::flow::graph>();
            worker = std::make_unique<tbb::flow::function_node<size_t>>(*graph, tbb::flow::unlimited,
                [this](size_t message) { try { Complete(message / kFinalizationRelayPhases, unsigned(message % kFinalizationRelayPhases)); }
                    catch (...) { Finish(message / kFinalizationRelayPhases, {}, true); }
                    return tbb::flow::continue_msg{}; }); });
        for (size_t i = 0; i != slots.size(); ++i) { slots[i].service = this; slots[i].index = i;
            for (unsigned phase = 0; phase != kFinalizationRelayPhases; ++phase) slots[i].callback[phase] = {&slots[i], phase}; }
    }
    FinalizationRelay* Reserve(std::shared_ptr<UsdGenCudaExecutionJob> job,
        std::function<void(std::shared_ptr<const UsdGenDeviceGeneration>)> done) noexcept;
    void Post(size_t index, unsigned phase) noexcept { try { if (!worker->try_put(index * kFinalizationRelayPhases + phase)) std::terminate(); } catch (...) { std::terminate(); } }
    bool Install(Slot&, unsigned, cudaStream_t) noexcept;
    void Finish(size_t, std::shared_ptr<const UsdGenDeviceGeneration>, bool) noexcept;
    void Complete(size_t, unsigned);
};

// This test-only gate defers only the phase-return signal.  It never parks a
// relay worker: native terminal and launcher-return remain independently
// observable, while release later emits the one causal Return(2).
struct FinalizationReturnGate {
    enum Handoff : unsigned { Ready = 1u, Released = 2u, Returned = 4u };
    std::atomic<bool> claimed{false};
    std::atomic<unsigned> handoff{0};
    std::atomic<bool> entered{false};
    std::atomic<bool> terminal{false};
    std::atomic<std::atomic<unsigned>*> terminalState{nullptr};
    std::atomic<FinalizationRelayService::Slot*> slot{nullptr};
};

namespace {
FinalizationRelayService& FinalizationRelays() { static auto* service = new FinalizationRelayService; return *service; }
void FinalizationCallback(cudaStream_t, cudaError_t status, void* data) noexcept {
    auto* identity = static_cast<FinalizationRelayService::Slot::CallbackIdentity*>(data);
    auto* slot = identity->slot;
    if (slot->service->failNative[identity->phase].exchange(false, std::memory_order_acq_rel)) status = cudaErrorUnknown;
    if (identity->phase == 2) if (auto* gate = slot->metadataReturnGate.load(std::memory_order_acquire)) gate->terminal.store(true, std::memory_order_release);
    slot->Signal(identity->phase, status);
}
}
FinalizationRelay* FinalizationRelayService::Reserve(std::shared_ptr<UsdGenCudaExecutionJob> job,
    std::function<void(std::shared_ptr<const UsdGenDeviceGeneration>)> done) noexcept {
    size_t const limit = std::min(capacity.load(std::memory_order_acquire), slots.size());
    for (size_t i = 0; i != limit; ++i) { bool expected = false;
        if (!slots[i].occupied.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) continue;
        try { auto relay = std::make_shared<FinalizationRelay>(); relay->job = std::move(job); relay->done = std::move(done); relay->slot = i;
            for (unsigned p = 0; p != kFinalizationRelayPhases; ++p) { slots[i].state[p].store(0); slots[i].status[p].store(int(cudaErrorUnknown)); }
            slots[i].quarantined.store(false); slots[i].metadataReturnGate.store(nullptr); slots[i].relay = std::move(relay); return slots[i].relay.get();
        } catch (...) { slots[i].occupied.store(false, std::memory_order_release); return nullptr; }
    } return nullptr;
}
void FinalizationRelayService::Slot::Signal(unsigned phase, cudaError_t value) noexcept {
    auto& bits = state[phase]; if (bits.fetch_or(8u, std::memory_order_acq_rel) & 8u) return;
    status[phase].store(int(value), std::memory_order_release); unsigned const old = bits.fetch_or(1u, std::memory_order_acq_rel);
    if ((old & 2u) && !(old & 4u) && !(bits.fetch_or(4u, std::memory_order_acq_rel) & 4u)) service->Post(index, phase);
}
void FinalizationRelayService::Slot::Return(unsigned phase) noexcept {
    auto& bits = state[phase]; unsigned const old = bits.fetch_or(2u, std::memory_order_acq_rel);
    if ((old & 1u) && !(old & 4u) && !(bits.fetch_or(4u, std::memory_order_acq_rel) & 4u)) service->Post(index, phase);
}

namespace {
void ReturnFinalizationMetadataWhenReleased(FinalizationReturnGate& gate) noexcept {
    unsigned state = gate.handoff.load(std::memory_order_acquire);
    for (;;) {
        if ((state & (FinalizationReturnGate::Ready | FinalizationReturnGate::Released)) !=
                (FinalizationReturnGate::Ready | FinalizationReturnGate::Released) ||
            (state & FinalizationReturnGate::Returned))
            return;
        if (gate.handoff.compare_exchange_weak(state,
                state | FinalizationReturnGate::Returned,
                std::memory_order_acq_rel, std::memory_order_acquire))
            break;
    }
    // Claim the one-shot handoff before dereferencing the permanent slot.
    // A stale/reused slot is rejected by its gate identity rather than scanned.
    auto* const slot = gate.slot.load(std::memory_order_acquire);
    if (!slot || slot->metadataReturnGate.load(std::memory_order_acquire) != &gate)
        return;
    slot->Return(2);
}
}

bool FinalizationRelayService::Install(Slot& slot, unsigned phase, cudaStream_t stream) noexcept {
    auto relay = slot.relay; if (!relay) return false;
    if (auto gate = std::atomic_load(&s_finalizationGates[phase]); gate && !gate->claimed.exchange(true, std::memory_order_acq_rel)) {
        relay->gates[phase] = gate;
        if (cudaStreamAddCallback(stream, SourceAsyncGateCallback, gate.get(), 0) != cudaSuccess) return false;
    }
    return !failInstall[phase].exchange(false, std::memory_order_acq_rel) &&
        cudaStreamAddCallback(stream, FinalizationCallback, &slot.callback[phase], 0) == cudaSuccess;
}
void FinalizationRelayService::Finish(size_t index, std::shared_ptr<const UsdGenDeviceGeneration> result, bool unsafe) noexcept {
    auto relay = slots[index].relay; if (!relay) return; auto job = relay->job;
    if (!result && !unsafe && job && job->state.HasPendingRbf() && job->workspace) {
        CudaDeviceScope selected(job->workspace->impl_->device);
        if (!selected.selected || !job->state.ResolvePendingRbf(false)) unsafe = true;
    }
    if (!result && job) { job->failed = true; job->finalAsyncInFlight = false; job->finalDone = true; }
    if (unsafe) {
        if (job) job->CloseMemoryReservation();
        relay->quarantine.store(true, std::memory_order_release);
        slots[index].quarantined.store(true, std::memory_order_release);
        if (relay->poison) relay->poison();
        relay->poison = {};
        if (job) { job->workspace = nullptr; job->diagnostics = nullptr; }
    }
    auto done = std::move(relay->done);
    if (unsafe) { relay->done = {}; if (done) try { done({}); } catch (...) {} return; }
    if (job) { job->finalAsyncInFlight = false; job->finalDone = true; if (result) job->failed = false; }
    // Pinned D2H storage and the previous geometry lease have normal CUDA
    // cleanup paths.  Release a proven candidate under its workspace device,
    // before handing user code an opportunity to tear down that workspace.
    if (relay->candidate && job && job->workspace) {
        {
            CudaDeviceScope selected(job->workspace->impl_->device);
            if (selected.selected) relay->candidate.reset();
            else {
                // Even completed work must not run lease/pinned/device-buffer
                // destructors in an unselectable CUDA context.  This is a
                // terminal failure: retain both candidate and any just-made
                // publication in the fixed slot, poison the workspace, and
                // report null without dropping the caller completion.
                relay->published = std::move(result);
                relay->quarantine.store(true, std::memory_order_release);
                slots[index].quarantined.store(true, std::memory_order_release);
                if (relay->poison) relay->poison();
                relay->poison = {};
                job->failed = true; job->finalAsyncInFlight = false; job->finalDone = true;
                job->CloseMemoryReservation();
                job->workspace = nullptr; job->diagnostics = nullptr;
                relay->done = {};
                if (done) try { done({}); } catch (...) {}
                return;
            }
        }
    } else relay->candidate.reset();
    // Publication has transferred every retained allocation to its immutable
    // generation owner (and reclassified only that owner's private COW
    // planes). Seal the job reservation before user-visible completion so a
    // subsequently retired temporary cannot recycle credit back to Pending.
    // Published child permits remain charged normally after reservation close.
    if (job) job->CloseMemoryReservation();
    relay->job.reset();
    slots[index].metadataReturnGate.store(nullptr, std::memory_order_release);
    slots[index].relay.reset(); slots[index].occupied.store(false, std::memory_order_release);
    if (done) try { done(std::move(result)); } catch (...) {}
}

std::shared_ptr<const UsdGenDeviceGeneration> ExecutionState::PublishFinal(
    UsdGenCudaExecutionPlan const& plan,
    UsdGenCudaExecutionWorkspace& workspace, uint64_t generation,
    std::shared_ptr<const UsdGenDeviceGeneration> const& previous,
    std::vector<UsdGenDeviceTileMetadata> tiles, bool gpuTopologyEqual,
    UsdGenDiagnostics* diagnostics) {
    auto& execution = *workspace.impl_;
    uint64_t topologyVersion = generation;
    if (previous) {
        auto const& priorTiles = previous->Geometry().tiles;
        bool const sameTiles = priorTiles.size() == tiles.size() && std::equal(priorTiles.begin(), priorTiles.end(), tiles.begin(),
            [](auto const& a, auto const& b) { return a.tile == b.tile && a.firstCurve == b.firstCurve && a.curveCount == b.curveCount && a.firstPoint == b.firstPoint && a.pointCount == b.pointCount; });
        auto const& oldTopology = previous->Geometry().curveTopology;
        bool const sameKind = oldTopology.type == curveTopology.type && oldTopology.basis == curveTopology.basis && oldTopology.wrap == curveTopology.wrap;
        if (gpuTopologyEqual && sameTiles && sameKind) topologyVersion = previous->Geometry().topologyVersion;
    }
    // Allocate/cache stats before ownership transfer so a throwing host
    // allocation cannot leave a half-published terminal state.
    std::shared_ptr<std::vector<UsdGenCudaBindingStats>> stats;
    try { stats = std::make_shared<std::vector<UsdGenCudaBindingStats>>();
        for (auto const& step : plan.steps) { auto const& runtime = *execution.steps.at(step->path); if (!runtime.rbf) continue; auto const& cache = *runtime.rbf;
            stats->push_back({step->path, cache.state ? cache.state->identity : 0, cache.bindCount, cache.solveCount, cache.state ? AcceptedRbfSampleCount(cache.state->resources->field) : 0}); }
    } catch (...) { Fail(diagnostics, "CUDA final binding statistics allocation failed; previous generation retained"); return {}; }
    std::string reason;
    auto result = sourceTerminalSelected
        ? (resampled
            ? gpu::MakeResampledGeneration(std::move(source), std::move(resampled), generation,
                  &reason, deformed, std::move(finalWidths), std::move(finalPoints),
                  topologyVersion, std::move(namedChannels))
            : gpu::MakeSourceGeneration(std::move(source), generation, &reason, deformed,
                  std::move(finalWidths), std::move(finalPoints), topologyVersion,
                  std::move(namedChannels)))
        : compacted
        ? gpu::MakeCompactedGeneration(std::move(compacted), generation, &reason, deformed, std::move(finalWidths), std::move(finalPoints), topologyVersion, std::move(namedChannels))
        : curveGrow
        ? gpu::MakeCurveGrowGeneration(std::move(curveGrow), generation, &reason,
              deformed, topologyVersion, std::move(namedChannels),
              std::move(finalWidths), std::move(finalPoints))
        : scatterGrow
        ? gpu::MakeScatterGrowGeneration(std::move(scatterGrow), generation, &reason,
              deformed, topologyVersion, std::move(namedChannels),
              std::move(finalWidths), std::move(finalPoints))
        : resampled
        ? gpu::MakeResampledGeneration(std::move(source), std::move(resampled), generation, &reason, deformed, std::move(finalWidths), std::move(finalPoints), topologyVersion, std::move(namedChannels))
        : gpu::MakeSourceGeneration(std::move(source), generation, &reason, deformed, std::move(finalWidths), std::move(finalPoints), topologyVersion, std::move(namedChannels));
    if (!result) { Fail(diagnostics, reason); return {}; }
    result = gpu::WithTileMetadata(result, std::move(tiles), &reason, curveTopology);
    if (!result) { Fail(diagnostics, reason); return {}; }
    std::atomic_store(&execution.stats, std::shared_ptr<const std::vector<UsdGenCudaBindingStats>>(std::move(stats)));
    finalized = true; return result;
}

void FinalizationRelayService::Complete(size_t index, unsigned phase) {
    auto relay = slots[index].relay; if (!relay || !relay->job || !relay->candidate) return;
    auto job = relay->job; auto& candidate = *relay->candidate;
    auto& workspace = *job->workspace;
    CudaDeviceScope selected(workspace.impl_->device);
    if (!selected.selected || workspace.impl_->poisoned.load(std::memory_order_acquire)) { selected.Restore(); Finish(index, {}, true); return; }
    bool gatesOK = true;
    for (auto const& gate : relay->gates) gatesOK = gatesOK && (!gate || !gate->timedOut.load(std::memory_order_acquire));
    bool const nativeOK = cudaError_t(slots[index].status[phase].load(std::memory_order_acquire)) == cudaSuccess;
    if (!nativeOK) { selected.Restore(); Finish(index, {}, true); return; }
    if (!gatesOK) { selected.Restore(); Finish(index, {}, false); return; }
    auto stream = workspace.impl_->stream;
    if (phase == 0) {
        if (!candidate.tileHostStatus.data || *candidate.tileHostStatus.data != 0) { selected.Restore(); Finish(index, {}, false); return; }
        gpu::CurveTileBoundsOptions options; options.basis = job->state.curveTopology.basis == UsdGenDeviceCurveBasis::CatmullRom ? gpu::CurveTileBoundsBasis::CatmullRom : gpu::CurveTileBoundsBasis::BSpline;
        gpu::CurveTileBoundsInput input{job->state.geometry.points, job->state.geometry.widths, static_cast<gpu::DeviceBuffer<gpu::CurveTileSpan> const&>(candidate.deviceTiles).view()};
        if (gpu::BuildCurveTileBounds(options, input, {candidate.boundsScratch.view()}, {candidate.tileMinimums.view(), candidate.tileMaximums.view(), candidate.boundsStatus.view()}, stream) != cudaSuccess ||
            cudaMemcpyAsync(candidate.boundsHostStatus.data, candidate.boundsStatus.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            !Install(slots[index], 1, stream)) { selected.Restore(); Finish(index, {}, true); return; }
        selected.Restore(); slots[index].Return(1); return;
    }
    if (phase == 1) {
        if (!candidate.boundsHostStatus.data || *candidate.boundsHostStatus.data != 0) { selected.Restore(); Finish(index, {}, false); return; }
        // BeginFresh is the only phase-2 allocation/preflight boundary.  It
        // precedes every metadata D2H, so a clean allocation rejection has no
        // queued prefix and can reclaim normally.
        if (job->previous && FinalizationRelays().failTopologyAllocation.exchange(false, std::memory_order_acq_rel)) { selected.Restore(); Finish(index, {}, false); return; }
        if (job->previous && (candidate.topology->BeginFresh(
                candidate.previousLease.Geometry(), job->state.geometry, stream,
                job->memoryReservation.get()) != cudaSuccess)) {
            selected.Restore(); Finish(index, {}, candidate.topology->HasUnprovenWork()); return;
        }
        candidate.topologyActive = bool(job->previous);
        bool copiesOK = true;
        if (candidate.requirements.tileCount) copiesOK =
            cudaMemcpyAsync(candidate.spans.data, candidate.deviceTiles.data(), candidate.requirements.tileCount * sizeof(gpu::CurveTileSpan), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaMemcpyAsync(candidate.minimums.data, candidate.tileMinimums.data(), candidate.requirements.tileCount * sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaMemcpyAsync(candidate.maximums.data, candidate.tileMaximums.data(), candidate.requirements.tileCount * sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
        if (!copiesOK ||
            (candidate.topologyActive && candidate.topology->EnqueueFreshStatus(stream) != cudaSuccess)) { selected.Restore(); Finish(index, {}, true); return; }
        if (auto gate = std::atomic_load(&s_finalizationReturnGate);
            gate && !gate->claimed.exchange(true, std::memory_order_acq_rel)) {
            relay->returnGate = gate;
            gate->terminalState.store(&slots[index].state[2], std::memory_order_release);
            slots[index].metadataReturnGate.store(gate.get(), std::memory_order_release);
        }
        if (!Install(slots[index], 2, stream)) { selected.Restore(); Finish(index, {}, true); return; }
        if (auto gate = relay->returnGate) {
            // No later launcher access may touch the slot or payload after
            // this publication: release can immediately post phase 2.
            selected.Restore();
            gate->slot.store(&slots[index], std::memory_order_release);
            gate->entered.store(true, std::memory_order_release);
            gate->handoff.fetch_or(FinalizationReturnGate::Ready,
                                   std::memory_order_acq_rel);
            ReturnFinalizationMetadataWhenReleased(*gate);
            return;
        }
        selected.Restore(); slots[index].Return(2); return;
    }
    bool equal = false;
    if (candidate.topologyActive && candidate.topology->CommitFreshFinish(&equal) != cudaSuccess) { bool const unsafe = candidate.topology->HasUnprovenWork(); selected.Restore(); Finish(index, {}, unsafe); return; }
    try { for (size_t i = 0; i != candidate.requirements.tileCount; ++i) { auto const& span = candidate.spans.data[i]; auto& tile = candidate.tiles[i]; tile = {span.tile, span.firstCurve, span.curveCount, span.firstPoint, span.pointCount}; tile.extentMin = {candidate.minimums.data[i].x, candidate.minimums.data[i].y, candidate.minimums.data[i].z}; tile.extentMax = {candidate.maximums.data[i].x, candidate.maximums.data[i].y, candidate.maximums.data[i].z}; tile.boundsValid = true; }
        auto result = job->state.PublishFinal(*job->plan, workspace, job->generation, job->previous, std::move(candidate.tiles), equal, job->diagnostics);
        // PublishFinal has completed every fallible generation/metadata
        // construction step, but its existing statistics snapshot still
        // describes the previously accepted cache.  Allocate the replacement
        // snapshot before accepting the RBF transaction so allocation failure
        // can still restore the old surface pose and coefficients.
        std::shared_ptr<std::vector<UsdGenCudaBindingStats>> acceptedStats;
        if (result && job->state.HasPendingRbf()) {
            try {
                acceptedStats = std::make_shared<std::vector<UsdGenCudaBindingStats>>();
                for (auto const& step : job->plan->steps) {
                    auto const& runtime = *workspace.impl_->steps.at(step->path);
                    if (runtime.rbf) acceptedStats->push_back({step->path, 0, 0, 0, 0});
                }
            } catch (...) {
                Fail(job->diagnostics,
                    "CUDA accepted RBF statistics allocation failed; previous generation retained");
                result.reset();
            }
        }
        if (result && job->state.HasPendingRbf()) {
            if (!job->state.ResolvePendingRbf(true)) {
                result.reset();
                selected.Restore(); Finish(index, {}, true); return;
            }
            size_t output = 0;
            for (auto const& step : job->plan->steps) {
                auto const& runtime = *workspace.impl_->steps.at(step->path);
                if (!runtime.rbf) continue;
                auto const& cache = *runtime.rbf;
                auto& value = (*acceptedStats)[output++];
                value.identity = cache.state ? cache.state->identity : 0;
                value.bindCount = cache.bindCount;
                value.solveCount = cache.solveCount;
                value.sampleCount = cache.state ? AcceptedRbfSampleCount(cache.state->resources->field) : 0;
            }
            std::atomic_store(&workspace.impl_->stats,
                std::shared_ptr<const std::vector<UsdGenCudaBindingStats>>(std::move(acceptedStats)));
        }
        selected.Restore(); Finish(index, std::move(result), false);
    } catch (...) { selected.Restore(); Finish(index, {}, false); }
}

bool FinalizeCudaExecutionJobAsync(std::shared_ptr<UsdGenCudaExecutionJob> job,
    std::function<void(std::shared_ptr<const UsdGenDeviceGeneration>)> completion) {
    if (!job || !completion || job->failed.load(std::memory_order_acquire) || job->finalDone || job->finalAsyncInFlight || job->sourceAsyncInFlight ||
        job->operatorAsyncInFlight.load(std::memory_order_acquire) || !job->sourceDone || !job->plan || !job->workspace) return false;
    bool const allOperatorsFinished = !job->plan->taskDag
        ? job->nextOperator == job->plan->steps.size()
        : std::all_of(job->operatorFinished.begin(), job->operatorFinished.end(),
            [](uint8_t value) { return value != 0; });
    if (!allOperatorsFinished) return false;
    auto& workspace = *job->workspace;
    if (workspace.impl_->poisoned.load(std::memory_order_acquire)) return false;
    CudaDeviceScope selected(workspace.impl_->device); cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (!selected.selected || cudaStreamIsCapturing(workspace.impl_->stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone) return false;
    if (!job->state.SelectTerminal(*job->plan, workspace.impl_->stream,
                                   job->diagnostics)) return false;
    auto* relay = FinalizationRelays().Reserve(job, std::move(completion)); if (!relay) return false; size_t const index = relay->slot;
    job->finalAsyncInFlight = true;
    try {
        relay->poison = [raw = job->workspace] {
            raw->impl_->poisoned.store(true, std::memory_order_release);
        };
        if (FinalizationRelays().failAllocation.exchange(false, std::memory_order_acq_rel)) { selected.Restore(); FinalizationRelays().Finish(index, {}, false); return true; }
        relay->candidate = std::make_unique<FinalizationCandidate>(); auto& c = *relay->candidate; gpu::CurveTileOptions options; options.tileTarget = static_cast<uint32_t>(std::clamp(job->plan->desc.tileTarget, 32, 256));
        auto* const reservation = job->memoryReservation.get();
        if (gpu::GetCurveTileRequirements(options, job->state.captureCurveCount, job->state.geometry.curveCount, job->state.geometry.pointCount, &c.requirements, workspace.impl_->stream) != cudaSuccess ||
            c.deviceTiles.reset(c.requirements.tileCount, reservation,
                UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
            c.tileStatus.reset(1, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
            c.boundsStatus.reset(1, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
            c.tileMinimums.reset(c.requirements.tileCount, reservation,
                UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
            c.tileMaximums.reset(c.requirements.tileCount, reservation,
                UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
            c.boundsScratch.reset(c.requirements.tileCount, reservation,
                UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
            !c.tileHostStatus.Allocate(1, reservation) ||
            !c.boundsHostStatus.Allocate(1, reservation) ||
            !c.spans.Allocate(c.requirements.tileCount, reservation) ||
            !c.minimums.Allocate(c.requirements.tileCount, reservation) ||
            !c.maximums.Allocate(c.requirements.tileCount, reservation)) {
            selected.Restore(); FinalizationRelays().Finish(index, {}, false); return true;
        }
        c.tiles.resize(c.requirements.tileCount); c.topology = std::make_unique<gpu::CudaCurveTopologyCompare>();
        if (job->previous) { c.previousLease = gpu::AcquireGeometry(job->previous, workspace.impl_->stream); if (!c.previousLease) { selected.Restore(); FinalizationRelays().Finish(index, {}, false); return true; } }
        auto const frameOwners = job->state.sourceTopologyOwners;
        bool const fusedSubset = job->plan->scatterGrow && job->plan->geometryValueDag;
        if (fusedSubset && (!frameOwners || !frameOwners->frameStableIds ||
            !frameOwners->frameCaptureOrdinals ||
            frameOwners->frameStableIds->waitOn(workspace.impl_->stream) != cudaSuccess ||
            frameOwners->frameCaptureOrdinals->waitOn(workspace.impl_->stream) != cudaSuccess)) {
            selected.Restore(); FinalizationRelays().Finish(index, {}, true); return true;
        }
        gpu::CurveTileInput input{job->state.captureStableIds, job->state.geometry.stableIds,
            job->state.geometry.curveOffsets, job->state.captureCurveCount,
            job->state.geometry.curveCount, job->state.geometry.pointCount,
            fusedSubset ? gpu::CurveTileOrder::CaptureOrderSurvivorSubset :
                job->plan->scatterGrow ? gpu::CurveTileOrder::IdentityCaptureOrder :
                gpu::CurveTileOrder::SortedSurvivorSubset,
            fusedSubset ? gpu::DeviceView<const uint64_t>{frameOwners->frameStableIds->data(), frameOwners->frameStableIds->size()} : gpu::DeviceView<const uint64_t>{},
            fusedSubset ? gpu::DeviceView<const uint32_t>{frameOwners->frameCaptureOrdinals->data(), frameOwners->frameCaptureOrdinals->size()} : gpu::DeviceView<const uint32_t>{}};
        c.submitted = true;
        if (gpu::BuildCurveTiles(input, c.requirements, {c.deviceTiles.view(), c.tileStatus.view()}, workspace.impl_->stream) != cudaSuccess || cudaMemcpyAsync(c.tileHostStatus.data, c.tileStatus.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, workspace.impl_->stream) != cudaSuccess || !FinalizationRelays().Install(FinalizationRelays().slots[index], 0, workspace.impl_->stream)) { selected.Restore(); FinalizationRelays().Finish(index, {}, true); return true; }
        selected.Restore(); FinalizationRelays().slots[index].Return(0); return true;
    } catch (...) { bool const unsafe = relay->candidate && relay->candidate->submitted; selected.Restore(); FinalizationRelays().Finish(index, {}, unsafe); return true; }
}
#endif

#ifndef USDGEN_ENABLE_CUDA
bool FinalizeCudaExecutionJobAsync(std::shared_ptr<UsdGenCudaExecutionJob>,
    std::function<void(std::shared_ptr<const UsdGenDeviceGeneration>)>) { return false; }
#endif

bool ExecuteCudaJobOperator(UsdGenCudaExecutionJob& job, size_t index) {
#ifdef USDGEN_ENABLE_CUDA
    if (job.finalAsyncInFlight || job.sourceAsyncInFlight) return false;
    if (job.failed.load(std::memory_order_acquire) || job.sourceAsyncInFlight || job.operatorAsyncInFlight.load(std::memory_order_acquire) || !job.sourceDone || job.finalDone || index != job.nextOperator ||
        index >= CudaExecutionJobOperatorCount(job)) { job.failed = true; return false; }
    // The transferable peak is currently wired through relay-owned Width and
    // publication candidates. A compatibility synchronous operator must close
    // unused Pending credit before falling back to exact per-allocation charges.
    job.CloseMemoryReservation(true);
    job.failed = true;
    if (!job.state.RunOperator(*job.plan, *job.workspace, job.frame, index, job.diagnostics)) return false;
    job.failed = false;
    if (job.plan->taskDag) job.operatorFinished[index] = 1;
    ++job.nextOperator; return true;
#else
    (void)index;
    job.failed = true; return false;
#endif
}

std::shared_ptr<const UsdGenDeviceGeneration> FinalizeCudaExecutionJob(
    UsdGenCudaExecutionJob& job) {
#ifdef USDGEN_ENABLE_CUDA
    if (job.finalAsyncInFlight || job.sourceAsyncInFlight ||
        job.operatorAsyncInFlight.load(std::memory_order_acquire)) return {};
    bool const allOperatorsFinished = !job.plan->taskDag
        ? job.nextOperator == CudaExecutionJobOperatorCount(job)
        : std::all_of(job.operatorFinished.begin(), job.operatorFinished.end(),
            [](uint8_t value) { return value != 0; });
    if (job.failed.load(std::memory_order_acquire) || job.sourceAsyncInFlight || job.finalAsyncInFlight || !job.sourceDone || job.finalDone || !job.plan || !job.workspace ||
        !allOperatorsFinished) { job.failed = true; return {}; }
    // Synchronous finalization retains its established exact-allocation path.
    // Close any remaining async reservation credit before entering it.
    job.CloseMemoryReservation(true);
    job.failed = true;
    std::shared_ptr<const UsdGenDeviceGeneration> result;
    if (job.state.HasPendingRbf() &&
        s_failNextSynchronousFinalization.exchange(false, std::memory_order_acq_rel)) {
        Fail(job.diagnostics, "RBF synchronous finalization failure; previous generation retained");
    } else {
        result = job.state.Finalize(*job.plan, *job.workspace, job.generation,
            job.previous, job.diagnostics);
    }
    if (!result) {
        if (job.state.HasPendingRbf() && !job.state.ResolvePendingRbf(false))
            job.workspace->impl_->poisoned.store(true, std::memory_order_release);
        job.failed = true; return {};
    }
    if (job.state.HasPendingRbf()) {
        std::shared_ptr<std::vector<UsdGenCudaBindingStats>> acceptedStats;
        try {
            acceptedStats = std::make_shared<std::vector<UsdGenCudaBindingStats>>();
            for (auto const& step : job.plan->steps) {
                auto const& runtime = *job.workspace->impl_->steps.at(step->path);
                if (runtime.rbf) acceptedStats->push_back({step->path, 0, 0, 0, 0});
            }
        } catch (...) {
            if (!job.state.ResolvePendingRbf(false))
                job.workspace->impl_->poisoned.store(true, std::memory_order_release);
            job.failed = true; return {};
        }
        if (!job.state.ResolvePendingRbf(true)) {
            job.workspace->impl_->poisoned.store(true, std::memory_order_release);
            job.failed = true; return {};
        }
        size_t output = 0;
        for (auto const& step : job.plan->steps) {
            auto const& runtime = *job.workspace->impl_->steps.at(step->path);
            if (!runtime.rbf) continue;
            auto const& cache = *runtime.rbf;
            auto& value = (*acceptedStats)[output++];
            value.path = step->path;
            value.identity = cache.state ? cache.state->identity : 0;
            value.bindCount = cache.bindCount;
            value.solveCount = cache.solveCount;
            value.sampleCount = cache.state ? AcceptedRbfSampleCount(cache.state->resources->field) : 0;
        }
        std::atomic_store(&job.workspace->impl_->stats,
            std::shared_ptr<const std::vector<UsdGenCudaBindingStats>>(std::move(acceptedStats)));
    }
    job.failed = false;
    job.finalDone = true; return result;
#else
    job.failed = true; return {};
#endif
}

std::shared_ptr<const UsdGenDeviceGeneration> ExecuteCudaGraph(
    UsdGenCudaExecutionPlan const& plan, UsdGenCudaExecutionWorkspace& workspace, double frame,
    uint64_t generation, UsdGenDiagnostics* diagnostics,
    std::shared_ptr<const UsdGenDeviceGeneration> const& previous) {
#ifndef USDGEN_ENABLE_CUDA
    (void)plan;
    (void)generation;
    (void)frame;
    (void)previous;
    (void)workspace;
    Fail(diagnostics, "backend is not built");
    return {};
#else
    if (workspace.IsPoisoned()) {
        Fail(diagnostics, "CUDA execution workspace is poisoned");
        return {};
    }
    // A direct execution has no relay owner for a precharged balance.  Only
    // use one when planning proved the allocation shape. Literal Length and
    // literal cold RBF refine their runtime-only cardinalities before source
    // work; expression-driven paths retain exact per-allocation admission.
    UsdGenExecutionMemoryReservation memoryReservation;
    UsdGenExecutionMemoryReservation* reservation = nullptr;
    auto const scatterGrowAdmission = ReserveScatterGrowExecution(
        plan, workspace.impl_->device, workspace.impl_->stream, diagnostics,
        &memoryReservation);
    if (scatterGrowAdmission == LiteralLengthAdmission::Failed) return {};
    if (scatterGrowAdmission == LiteralLengthAdmission::Admitted)
        reservation = &memoryReservation;
    auto const literalLengthAdmission = ReserveLiteralLengthExecution(
        plan, workspace.impl_->device, workspace.impl_->stream, diagnostics,
        &memoryReservation);
    if (literalLengthAdmission == LiteralLengthAdmission::Failed) return {};
    if (literalLengthAdmission == LiteralLengthAdmission::Admitted)
        reservation = &memoryReservation;
    if (!reservation) {
        auto const literalRbfAdmission = ReserveLiteralRbfExecution(
            plan, workspace.impl_->device, diagnostics, &memoryReservation);
        if (literalRbfAdmission == LiteralRbfAdmission::Failed) return {};
        if (literalRbfAdmission == LiteralRbfAdmission::Admitted)
            reservation = &memoryReservation;
    }
    auto const metadata = GetCudaExecutionPlanMetadata(plan);
    if (!reservation && metadata) {
        auto const& estimate = metadata->MemoryEstimate();
        if (estimate.memoryAvailable && estimate.conservativeUpperBound) {
            if (estimate.concurrentPeakBytes >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                Fail(diagnostics,
                    "CUDA synchronous execution memory reservation size overflows this platform");
                return {};
            }
            CudaDeviceScope selected(workspace.impl_->device);
            if (!selected.selected) {
                Fail(diagnostics,
                    "cannot select CUDA workspace device for synchronous execution memory reservation");
                return {};
            }
            auto admitted = gpu::TryReserveCudaExecutionMemory(
                static_cast<size_t>(estimate.concurrentPeakBytes));
            if (!admitted) {
                Fail(diagnostics,
                    "CUDA execution job memory reservation exceeds the available device budget");
                return {};
            }
            memoryReservation = std::move(*admitted);
            reservation = &memoryReservation;
        }
    }
    // Keep the reservation before state: state destroys or transfers its
    // child permits first, then the reservation releases only unused Pending.
    CudaDeviceScope selected(workspace.impl_->device);
    if (!selected.selected) {
        Fail(diagnostics, "cannot select CUDA workspace device for synchronous execution");
        return {};
    }
    ExecutionState state;
    if (plan.scatterGrow) {
        if (!plan.scatterRoots) return {};
        if (!state.PrepareScatterGrowWorkspace(plan, workspace, diagnostics, reservation))
            return {};
        auto owner=std::make_unique<gpu::CudaScatterGrow>();
        auto done=std::make_unique<CudaInlineCompletion>();
        auto status=owner->BeginFresh(plan.scatterRoots, plan.scatterGrowControls,
            workspace.impl_->stream, reservation);
        if (status == gpu::ScatterGrowStatus::Ok &&
            !state.PrepareScatterGrowFrames(plan, workspace, workspace.impl_->stream,
                                            diagnostics))
            status = gpu::ScatterGrowStatus::CudaError;
        if (status == gpu::ScatterGrowStatus::Ok)
            status=owner->FinishFreshAsync(workspace.impl_->stream,
                CudaInlineCompletionCallback,done.get());
        if (status != gpu::ScatterGrowStatus::Ok ||
            cudaStreamSynchronize(workspace.impl_->stream)!=cudaSuccess ||
            done->status.load(std::memory_order_acquire)!=int(cudaSuccess) ||
            owner->CommitFreshFinish()!=gpu::ScatterGrowStatus::Ok) {
            if (owner->HasUnprovenWork()) {
                workspace.impl_->poisoned.store(true,std::memory_order_release);
                (void)done.release();
            }
            Fail(diagnostics,"CUDA Scatter->Grow execution failed; previous generation retained"); return {};
        }
        // Route the completed native producer through the same finalization
        // state as the async path.  Apart from making direct execution obey
        // the exact tile contract, this is what makes topology comparison and
        // COW publication identical between the two entry points.
        state.memoryReservation = reservation;
        state.scatterGrow = std::move(owner);
        state.geometry = state.scatterGrow->view();
        state.captureStableIds = state.geometry.stableIds;
        state.captureCurveCount = state.geometry.curveCount;
        state.hairT = state.scatterGrow->hairT();
        state.rootPrim = state.scatterGrow->rootPrim();
        state.rootUV = state.scatterGrow->rootUV();
        state.curveTopology = {UsdGenDeviceCurveType::Cubic,
                               UsdGenDeviceCurveBasis::BSpline,
                               UsdGenDeviceCurveWrap::Pinned};
        state.prepared = true;
    } else {
        if (!state.Prepare(plan, workspace, frame, diagnostics, true, nullptr,
                reservation)) return {};
    }
    // Both source routes share the all-operator transaction boundary. A
    // fused producer must neither skip failure injection nor leave accepted
    // RBF caches provisional after publishing its descendant generation.
    state.RecordSourceValue(plan);
    for (size_t i = 0; i < plan.steps.size(); ++i)
        if (!state.RunOperator(plan, workspace, frame, i, diagnostics)) return {};
    // Inject only at the all-operator join: every independent RBF lineage
    // must have parked its provisional transaction before rollback is tested.
    std::shared_ptr<const UsdGenDeviceGeneration> result;
    if (state.HasPendingRbf() &&
        s_failNextSynchronousFinalization.exchange(false, std::memory_order_acq_rel)) {
        Fail(diagnostics, "RBF synchronous finalization failure; previous generation retained");
    } else {
        result = state.Finalize(plan, workspace, generation, previous, diagnostics);
    }
    if (!result) {
        if (state.HasPendingRbf() && !state.ResolvePendingRbf(false)) {
            workspace.impl_->poisoned.store(true, std::memory_order_release);
            // A partially resolved host transaction is not safely reclaimable
            // on the synchronous owner path. Match relay quarantine: retain
            // the resource owners for context teardown instead of destructing
            // them after an impossible concrete rollback failure.
            state.AbandonPendingRbf();
        }
        return {};
    }
    if (state.HasPendingRbf()) {
        // Publish stats from the accepted cache, not the provisional state
        // observed by Finalize while it was constructing the generation.
        std::shared_ptr<std::vector<UsdGenCudaBindingStats>> acceptedStats;
        try {
            acceptedStats = std::make_shared<std::vector<UsdGenCudaBindingStats>>();
            acceptedStats->reserve(plan.steps.size());
        } catch (...) {
            if (!state.ResolvePendingRbf(false)) {
                workspace.impl_->poisoned.store(true, std::memory_order_release);
                state.AbandonPendingRbf();
            }
            Fail(diagnostics, "CUDA accepted RBF statistics allocation failed; previous generation retained");
            return {};
        }
        if (!state.ResolvePendingRbf(true)) {
            workspace.impl_->poisoned.store(true, std::memory_order_release);
            state.AbandonPendingRbf();
            return {};
        }
        for (auto const& step : plan.steps) {
            auto const& runtime = *workspace.impl_->steps.at(step->path);
            if (!runtime.rbf) continue;
            auto const& cache = *runtime.rbf;
            acceptedStats->push_back({step->path, cache.state ? cache.state->identity : 0,
                cache.bindCount, cache.solveCount,
                cache.state ? AcceptedRbfSampleCount(cache.state->resources->field) : 0});
        }
        std::atomic_store(&workspace.impl_->stats,
            std::shared_ptr<const std::vector<UsdGenCudaBindingStats>>(std::move(acceptedStats)));
    }
    return result;
#endif
}

#ifdef USDGEN_ENABLE_CUDA
std::shared_ptr<const UsdGenDeviceGeneration> ExecutionState::Finalize(
    UsdGenCudaExecutionPlan const& plan, UsdGenCudaExecutionWorkspace& workspace,
    uint64_t generation, std::shared_ptr<const UsdGenDeviceGeneration> const& previous,
    UsdGenDiagnostics* diagnostics) {
    if (!prepared || failed || finalized) return {};
    auto& execution = *workspace.impl_;
    if (execution.poisoned.load(std::memory_order_acquire)) {
        Fail(diagnostics, "CUDA workspace is poisoned by an unproven source upload"); this->failed = true; return {};
    }
    CudaDeviceScope selected(execution.device);
    if (!selected.selected) { Fail(diagnostics, "cannot select CUDA workspace device for final stage"); failed = true; return {}; }
    if (!SelectTerminal(plan, execution.stream, diagnostics)) return {};
    auto const stream = execution.stream;
    auto& geometryRef = this->geometry;
    auto const& desc = plan.desc;
    auto const capturedStableIds = this->captureStableIds;
    auto const capturedCurveCount = this->captureCurveCount;
    auto const& topologyRef = this->curveTopology;
    gpu::CurveTileRequirements tileRequirements;
    gpu::CurveTileOptions tileOptions;
    tileOptions.tileTarget = static_cast<uint32_t>(std::clamp(
        desc.tileTarget, 32, 256));
    if (gpu::GetCurveTileRequirements(tileOptions, capturedCurveCount,
            geometryRef.curveCount, geometryRef.pointCount, &tileRequirements, stream) != cudaSuccess) {
        Fail(diagnostics, "CUDA tile metadata requirements failed; previous generation retained"); return {};
    }
    gpu::DeviceBuffer<gpu::CurveTileSpan> deviceTiles;
    gpu::DeviceBuffer<uint32_t> tileStatus;
    if (deviceTiles.reset(tileRequirements.tileCount, memoryReservation,
            UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        tileStatus.reset(1, memoryReservation,
            UsdGenExecutionResourceKind::Scratch) != cudaSuccess) {
        Fail(diagnostics, "CUDA tile metadata allocation failed; previous generation retained"); return {};
    }
    auto const frameOwners = sourceTopologyOwners;
    bool const fusedSubset = plan.scatterGrow && plan.geometryValueDag;
    if (fusedSubset && (!frameOwners || !frameOwners->frameStableIds ||
        !frameOwners->frameCaptureOrdinals ||
        frameOwners->frameStableIds->waitOn(stream) != cudaSuccess ||
        frameOwners->frameCaptureOrdinals->waitOn(stream) != cudaSuccess)) {
        Fail(diagnostics, "CUDA Scatter->Grow tile frame ownership is unavailable"); return {};
    }
    gpu::CurveTileInput tileInput{capturedStableIds, geometryRef.stableIds,
        geometryRef.curveOffsets, capturedCurveCount, geometryRef.curveCount,
        geometryRef.pointCount,
        fusedSubset ? gpu::CurveTileOrder::CaptureOrderSurvivorSubset :
        plan.scatterGrow ? gpu::CurveTileOrder::IdentityCaptureOrder :
        gpu::CurveTileOrder::SortedSurvivorSubset,
        fusedSubset ? gpu::DeviceView<const uint64_t>{frameOwners->frameStableIds->data(), frameOwners->frameStableIds->size()} : gpu::DeviceView<const uint64_t>{},
        fusedSubset ? gpu::DeviceView<const uint32_t>{frameOwners->frameCaptureOrdinals->data(), frameOwners->frameCaptureOrdinals->size()} : gpu::DeviceView<const uint32_t>{}};
    gpu::CurveTileOutput tileOutput{deviceTiles.view(), tileStatus.view()};
    if (gpu::BuildCurveTiles(tileInput, tileRequirements, tileOutput, stream) != cudaSuccess) {
        Fail(diagnostics, "CUDA tile metadata launch failed; previous generation retained"); return {};
    }
    // This is the deliberate publication boundary.  Fence producer work and
    // read the status before touching spans: a rejected GPU candidate leaves
    // its span buffer untouched by contract.
    uint32_t tileError = 0;
    cudaError_t const statusCopy = cudaMemcpyAsync(&tileError, tileStatus.data(),
        sizeof(tileError), cudaMemcpyDeviceToHost, stream);
    cudaError_t const statusFence = cudaStreamSynchronize(stream);
    if (statusCopy != cudaSuccess || statusFence != cudaSuccess) {
        Fail(diagnostics, "CUDA tile metadata status readback failed; previous generation retained"); return {};
    }
    if (tileError != 0) {
        Fail(diagnostics, "CUDA tile metadata validation failed; previous generation retained"); return {};
    }
    gpu::DeviceBuffer<float3> tileMinimums, tileMaximums;
    gpu::DeviceBuffer<gpu::CurveTileBoundsScratch> tileBoundsScratch;
    gpu::DeviceBuffer<uint32_t> boundsStatus;
    if (tileMinimums.reset(tileRequirements.tileCount, memoryReservation,
            UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        tileMaximums.reset(tileRequirements.tileCount, memoryReservation,
            UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        tileBoundsScratch.reset(tileRequirements.tileCount, memoryReservation,
            UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        boundsStatus.reset(1, memoryReservation,
            UsdGenExecutionResourceKind::Scratch) != cudaSuccess) {
        Fail(diagnostics, "CUDA tile bounds allocation failed; previous generation retained"); return {};
    }
    gpu::CurveTileBoundsOptions boundsOptions;
    boundsOptions.basis = topologyRef.basis == UsdGenDeviceCurveBasis::CatmullRom
        ? gpu::CurveTileBoundsBasis::CatmullRom
        : gpu::CurveTileBoundsBasis::BSpline;
    gpu::CurveTileBoundsInput boundsInput{
        geometryRef.points, geometryRef.widths,
        static_cast<gpu::DeviceBuffer<gpu::CurveTileSpan> const&>(deviceTiles).view()};
    gpu::CurveTileBoundsWorkspace boundsWorkspace{tileBoundsScratch.view()};
    gpu::CurveTileBoundsOutput boundsOutput{
        tileMinimums.view(), tileMaximums.view(), boundsStatus.view()};
    if (gpu::BuildCurveTileBounds(boundsOptions, boundsInput, boundsWorkspace,
            boundsOutput, stream) != cudaSuccess) {
        Fail(diagnostics, "CUDA tile bounds launch failed; previous generation retained"); return {};
    }
    std::vector<gpu::CurveTileSpan> tileSpans(tileRequirements.tileCount);
    std::vector<float3> tileMins(tileRequirements.tileCount);
    std::vector<float3> tileMaxs(tileRequirements.tileCount);
    uint32_t boundsError = 0;
    cudaError_t boundsStatusCopy = cudaMemcpyAsync(&boundsError, boundsStatus.data(),
        sizeof(boundsError), cudaMemcpyDeviceToHost, stream);
    cudaError_t const boundsStatusFence = cudaStreamSynchronize(stream);
    if (boundsStatusCopy != cudaSuccess || boundsStatusFence != cudaSuccess) {
        Fail(diagnostics, "CUDA tile bounds status readback failed; previous generation retained"); return {};
    }
    if (boundsError != 0) {
        Fail(diagnostics, "CUDA tile bounds validation failed; previous generation retained"); return {};
    }

    cudaError_t spanCopy = cudaSuccess;
    cudaError_t minimumCopy = cudaSuccess;
    cudaError_t maximumCopy = cudaSuccess;
    if (tileRequirements.tileCount) {
        spanCopy = cudaMemcpyAsync(tileSpans.data(), deviceTiles.data(),
            tileSpans.size() * sizeof(gpu::CurveTileSpan), cudaMemcpyDeviceToHost, stream);
        minimumCopy = cudaMemcpyAsync(tileMins.data(), tileMinimums.data(),
            tileMins.size() * sizeof(float3), cudaMemcpyDeviceToHost, stream);
        maximumCopy = cudaMemcpyAsync(tileMaxs.data(), tileMaximums.data(),
            tileMaxs.size() * sizeof(float3), cudaMemcpyDeviceToHost, stream);
    }
    cudaError_t const spanFence = cudaStreamSynchronize(stream);
    if (spanCopy != cudaSuccess || minimumCopy != cudaSuccess ||
        maximumCopy != cudaSuccess || spanFence != cudaSuccess) {
        Fail(diagnostics, "CUDA tile bounds readback failed; previous generation retained"); return {};
    }
    std::vector<UsdGenDeviceTileMetadata> tiles;
    tiles.reserve(tileSpans.size());
    for (size_t i = 0; i != tileSpans.size(); ++i) {
        gpu::CurveTileSpan const& span = tileSpans[i];
        tiles.push_back({span.tile, span.firstCurve, span.curveCount,
                         span.firstPoint, span.pointCount});
        UsdGenDeviceTileMetadata &tile = tiles.back();
        tile.extentMin = {tileMins[i].x, tileMins[i].y, tileMins[i].z};
        tile.extentMax = {tileMaxs[i].x, tileMaxs[i].y, tileMaxs[i].z};
        tile.boundsValid = true;
    }

    bool sameTopology = false;
    if (previous) {
        auto lease = gpu::AcquireGeometry(previous, stream);
        if (!lease || gpu::CompareCurveTopology(lease.Geometry(), geometryRef,
                stream, &sameTopology, memoryReservation) != cudaSuccess) {
            Fail(diagnostics, "cannot compare GPU topology against the previous publication"); return {};
        }
    }
    auto result = PublishFinal(plan, workspace, generation,
        previous, std::move(tiles), sameTopology, diagnostics);
    if (!result) failed = true;
    return result;
}
#endif

void armCudaSourceAsyncCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    std::atomic_store(&s_sourceAsyncTestGate, std::make_shared<SourceAsyncTestGate>());
#endif
}
void waitCudaSourceAsyncCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_sourceAsyncTestGate)) {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!gate->entered.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        if (!gate->entered.load(std::memory_order_acquire))
            throw std::runtime_error("CUDA source async test gate timed out");
    }
#endif
}
void releaseCudaSourceAsyncCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_sourceAsyncTestGate))
        gate->released.store(true, std::memory_order_release);
    std::atomic_store(&s_sourceAsyncTestGate, std::shared_ptr<SourceAsyncTestGate>());
#endif
}
namespace {
void ArmSourceControlGate(unsigned phase) {
#ifdef USDGEN_ENABLE_CUDA
    std::atomic_store(&s_sourceAsyncControlGates[phase],std::make_shared<SourceAsyncTestGate>());
#else
    (void)phase;
#endif
}
void WaitSourceControlGate(unsigned phase) {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate=std::atomic_load(&s_sourceAsyncControlGates[phase])) {
        auto const deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while (!gate->entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
        if (!gate->entered.load(std::memory_order_acquire)) throw std::runtime_error("CUDA source control callback gate timed out");
    }
#else
    (void)phase;
#endif
}
void ReleaseSourceControlGate(unsigned phase) {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate=std::atomic_load(&s_sourceAsyncControlGates[phase])) gate->released.store(true,std::memory_order_release);
    std::atomic_store(&s_sourceAsyncControlGates[phase],std::shared_ptr<SourceAsyncTestGate>());
#else
    (void)phase;
#endif
}
}
void armCudaSourceAsyncContextCallbackGateForTesting() { ArmSourceControlGate(0); }
void waitCudaSourceAsyncContextCallbackGateForTesting() { WaitSourceControlGate(0); }
void releaseCudaSourceAsyncContextCallbackGateForTesting() { ReleaseSourceControlGate(0); }
void armCudaSourceAsyncProgramCallbackGateForTesting() { ArmSourceControlGate(1); }
void waitCudaSourceAsyncProgramCallbackGateForTesting() { WaitSourceControlGate(1); }
void releaseCudaSourceAsyncProgramCallbackGateForTesting() { ReleaseSourceControlGate(1); }
void armCudaSourceAsyncScalarCallbackGateForTesting() { ArmSourceControlGate(2); }
void waitCudaSourceAsyncScalarCallbackGateForTesting() { WaitSourceControlGate(2); }
void releaseCudaSourceAsyncScalarCallbackGateForTesting() { ReleaseSourceControlGate(2); }
void armCudaSourceAsyncScalarLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    std::atomic_store(&s_sourceAsyncScalarReturnTestGate,std::make_shared<SourceScalarReturnGate>());
#endif
}
void waitCudaSourceAsyncScalarLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate=std::atomic_load(&s_sourceAsyncScalarReturnTestGate)) {
        auto const deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while ((!gate->entered.load(std::memory_order_acquire) || !gate->terminal.load(std::memory_order_acquire) ||
                !(gate->terminalState.load(std::memory_order_acquire) &&
                  (gate->terminalState.load(std::memory_order_acquire)->load(std::memory_order_acquire)&1u))) &&
               std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
        if (!gate->entered.load(std::memory_order_acquire) || !gate->terminal.load(std::memory_order_acquire) ||
            !(gate->terminalState.load(std::memory_order_acquire) &&
              (gate->terminalState.load(std::memory_order_acquire)->load(std::memory_order_acquire)&1u)))
            throw std::runtime_error("CUDA source scalar launcher-return gate timed out");
    }
#endif
}
void releaseCudaSourceAsyncScalarLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate=std::atomic_load(&s_sourceAsyncScalarReturnTestGate)) {
        gate->handoff.fetch_or(SourceScalarReturnGate::Released, std::memory_order_acq_rel);
        ReturnSourceScalarWhenReleased(*gate);
    }
    std::atomic_store(&s_sourceAsyncScalarReturnTestGate,std::shared_ptr<SourceScalarReturnGate>());
#endif
}
void armCudaSourceAsyncResampleCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    std::atomic_store(&s_sourceAsyncResampleTestGate, std::make_shared<SourceAsyncTestGate>());
#endif
}
void waitCudaSourceAsyncResampleCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_sourceAsyncResampleTestGate)) {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!gate->entered.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        if (!gate->entered.load(std::memory_order_acquire))
            throw std::runtime_error("CUDA source async resample test gate timed out");
    }
#endif
}
void releaseCudaSourceAsyncResampleCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_sourceAsyncResampleTestGate))
        gate->released.store(true, std::memory_order_release);
    std::atomic_store(&s_sourceAsyncResampleTestGate, std::shared_ptr<SourceAsyncTestGate>());
#endif
}
void armCudaSourceAsyncResampleLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    std::atomic_store(&s_sourceAsyncResampleReturnTestGate, std::make_shared<SourceAsyncTestGate>());
#endif
}
void waitCudaSourceAsyncResampleLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_sourceAsyncResampleReturnTestGate)) {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while ((!gate->entered.load(std::memory_order_acquire) ||
                !gate->terminal.load(std::memory_order_acquire) ||
                !(gate->terminalState.load(std::memory_order_acquire) &&
                  (gate->terminalState.load(std::memory_order_acquire)->load(std::memory_order_acquire) & 1u))) &&
               std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        auto* terminalState = gate->terminalState.load(std::memory_order_acquire);
        if (!gate->entered.load(std::memory_order_acquire) ||
            !gate->terminal.load(std::memory_order_acquire) || !terminalState ||
            !(terminalState->load(std::memory_order_acquire) & 1u))
            throw std::runtime_error("CUDA source async resample launcher-return gate timed out");
    }
#endif
}
void releaseCudaSourceAsyncResampleLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_sourceAsyncResampleReturnTestGate))
        gate->released.store(true, std::memory_order_release);
    std::atomic_store(&s_sourceAsyncResampleReturnTestGate, std::shared_ptr<SourceAsyncTestGate>());
#endif
}
#ifdef USDGEN_ENABLE_CUDA
namespace {
void ArmOperatorGate(unsigned phase) { std::atomic_store(&s_operatorAsyncGates[phase], std::make_shared<SourceAsyncTestGate>()); }
void WaitOperatorGate(unsigned phase) { auto gate = std::atomic_load(&s_operatorAsyncGates[phase]); if (!gate) return; auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10); while (!gate->entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield(); if (!gate->entered.load(std::memory_order_acquire)) throw std::runtime_error("CUDA operator async gate timed out"); }
void ReleaseOperatorGate(unsigned phase) { if (auto gate = std::atomic_load(&s_operatorAsyncGates[phase])) gate->released.store(true, std::memory_order_release); std::atomic_store(&s_operatorAsyncGates[phase], std::shared_ptr<SourceAsyncTestGate>()); }
size_t BranchStreamCount(uint64_t mask) noexcept {
    size_t result = 0;
    while (mask) { result += size_t(mask & 1u); mask >>= 1; }
    return result;
}
}
#endif
void armCudaOperatorAsyncContextCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ArmOperatorGate(0);
#endif
}
void waitCudaOperatorAsyncContextCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    WaitOperatorGate(0);
#endif
}
void releaseCudaOperatorAsyncContextCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ReleaseOperatorGate(0);
#endif
}
void armCudaOperatorAsyncProgramCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ArmOperatorGate(1);
#endif
}
void waitCudaOperatorAsyncProgramCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    WaitOperatorGate(1);
#endif
}
void releaseCudaOperatorAsyncProgramCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ReleaseOperatorGate(1);
#endif
}
void armCudaOperatorAsyncWidthCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ArmOperatorGate(2);
#endif
}
void waitCudaOperatorAsyncWidthCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    WaitOperatorGate(2);
#endif
}
void releaseCudaOperatorAsyncWidthCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ReleaseOperatorGate(2);
#endif
}
void armCudaOperatorAsyncWidthBranchGateForTesting(size_t expected) {
#ifdef USDGEN_ENABLE_CUDA
    if (!expected) expected = 1;
    std::atomic_store(&s_operatorAsyncWidthBranchGate,
        std::make_shared<WidthBranchTestGate>(expected));
    s_operatorAsyncWidthBranchStreamMask.store(0, std::memory_order_release);
#else
    (void)expected;
#endif
}
void waitCudaOperatorAsyncWidthBranchGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_operatorAsyncWidthBranchGate)) {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (gate->arrivals.load(std::memory_order_acquire) < gate->expected &&
               std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        if (gate->arrivals.load(std::memory_order_acquire) < gate->expected)
            throw std::runtime_error("CUDA Width branch async gate timed out");
    }
#endif
}
void releaseCudaOperatorAsyncWidthBranchGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_operatorAsyncWidthBranchGate))
        gate->released.store(true, std::memory_order_release);
    std::atomic_store(&s_operatorAsyncWidthBranchGate,
        std::shared_ptr<WidthBranchTestGate>());
#endif
}
size_t cudaOperatorAsyncWidthDistinctBranchStreamCountForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    return BranchStreamCount(s_operatorAsyncWidthBranchStreamMask.load(
        std::memory_order_acquire));
#else
    return 0;
#endif
}
void armCudaOperatorAsyncWidthDeviceOverlapWitnessForTesting(
    size_t expected, uint64_t dwellCycles) {
#ifdef USDGEN_ENABLE_CUDA
    auto witness = std::make_shared<WidthOverlapWitness>();
    // A probe only establishes an overlap claim for two or more branches;
    // cap both dimensions so a test control cannot create an unbounded device
    // dwell or request more task identities than the CUDA-free snapshot holds.
    if (expected < 2 || expected > kWidthOverlapWitnessMaxTasks) {
        witness->error.store(true, std::memory_order_release);
    } else {
        int device = -1;
        constexpr uint64_t kMaximumDwellCycles = 500000000;
        auto const cycles = std::max<uint64_t>(1,
            std::min(dwellCycles, kMaximumDwellCycles));
        if (cudaGetDevice(&device) != cudaSuccess ||
            !witness->Initialize(device, static_cast<uint32_t>(expected), cycles))
            witness->error.store(true, std::memory_order_release);
    }
    std::atomic_store(&s_operatorAsyncWidthOverlapWitness, std::move(witness));
#else
    (void)expected; (void)dwellCycles;
#endif
}
UsdGenExecutionOverlapWitnessSnapshot
cudaOperatorAsyncWidthDeviceOverlapWitnessSnapshotForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto witness = std::atomic_load(&s_operatorAsyncWidthOverlapWitness))
        return witness->Snapshot();
#endif
    return {};
}
void armCudaOperatorAsyncLengthCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ArmOperatorGate(2);
#endif
}
void waitCudaOperatorAsyncLengthCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    WaitOperatorGate(2);
#endif
}
void releaseCudaOperatorAsyncLengthCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ReleaseOperatorGate(2);
#endif
}
void armCudaOperatorAsyncCountsCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ArmOperatorGate(3);
#endif
}
void waitCudaOperatorAsyncCountsCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    WaitOperatorGate(3);
#endif
}
void releaseCudaOperatorAsyncCountsCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ReleaseOperatorGate(3);
#endif
}
void armCudaOperatorAsyncScatterCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ArmOperatorGate(4);
#endif
}
void waitCudaOperatorAsyncScatterCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    WaitOperatorGate(4);
#endif
}
void releaseCudaOperatorAsyncScatterCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ReleaseOperatorGate(4);
#endif
}
void armCudaOperatorAsyncNamedTopologyCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ArmOperatorGate(5);
#endif
}
void waitCudaOperatorAsyncNamedTopologyCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    WaitOperatorGate(5);
#endif
}
void releaseCudaOperatorAsyncNamedTopologyCallbackGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    ReleaseOperatorGate(5);
#endif
}
void armCudaOperatorAsyncWidthLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    std::atomic_store(&s_operatorAsyncReturnGate, std::make_shared<SourceAsyncTestGate>());
#endif
}
void waitCudaOperatorAsyncWidthLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    auto gate = std::atomic_load(&s_operatorAsyncReturnGate); if (!gate) return;
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while ((!gate->entered.load(std::memory_order_acquire) || !gate->terminal.load(std::memory_order_acquire) ||
            !(gate->terminalState.load(std::memory_order_acquire) &&
              (gate->terminalState.load(std::memory_order_acquire)->load(std::memory_order_acquire) & 1u))) &&
           std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    auto* terminalState = gate->terminalState.load(std::memory_order_acquire);
    if (!gate->entered.load(std::memory_order_acquire) || !gate->terminal.load(std::memory_order_acquire) ||
        !terminalState || !(terminalState->load(std::memory_order_acquire) & 1u))
        throw std::runtime_error("CUDA operator width launcher-return gate timed out");
#endif
}
void releaseCudaOperatorAsyncWidthLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_operatorAsyncReturnGate)) gate->released.store(true, std::memory_order_release);
    std::atomic_store(&s_operatorAsyncReturnGate, std::shared_ptr<SourceAsyncTestGate>());
#endif
}
void setCudaOperatorRelayCapacityForTesting(size_t value) {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().capacity.store(std::min(value, kOperatorRelayCapacity), std::memory_order_release);
#else
    (void)value;
#endif
}
size_t cudaOperatorRelayOccupiedCountForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    size_t count = 0; for (auto const& slot : OperatorRelays().slots) count += slot.occupied.load(std::memory_order_acquire) ? 1 : 0; return count;
#else
    return 0;
#endif
}
size_t cudaOperatorRelayQuarantinedCountForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    size_t count = 0; for (auto const& slot : OperatorRelays().slots) count += slot.quarantined.load(std::memory_order_acquire) ? 1 : 0; return count;
#else
    return 0;
#endif
}
void failNextCudaOperatorRelayContextCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failInstall[0].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayProgramCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failInstall[1].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayWidthCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failInstall[2].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayLengthCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failInstall[2].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayCountsCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failInstall[3].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayScatterCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failInstall[4].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayNamedTopologyCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failInstall[5].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayNonWidthCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failInstall[6].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayScatterAllocationForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    // Route through the execution DSO: tests may otherwise bind a separate
    // hidden static GPU archive instance rather than the relay's instance.
    gpu::failNextCudaCurveCompactionFreshScatterAllocationForTesting();
#endif
}
void failNextCudaOperatorRelayContextNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failNative[0].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayProgramNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failNative[1].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayWidthNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failNative[2].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayLengthNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failNative[2].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayCountsNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failNative[3].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayScatterNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failNative[4].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayNamedTopologyNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failNative[5].store(true, std::memory_order_release);
#endif
}
void failNextCudaOperatorRelayNonWidthNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    OperatorRelays().failNative[6].store(true, std::memory_order_release);
#endif
}
void setCudaSourceRelayCapacityForTesting(size_t value) {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().capacity.store(std::min(value, kSourceRelayCapacity), std::memory_order_release);
#else
    (void)value;
#endif
}
size_t cudaSourceRelayOccupiedCountForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    size_t count = 0;
    for (auto const& slot : SourceRelays().slots)
        count += slot.occupied.load(std::memory_order_acquire) ? 1 : 0;
    return count;
#else
    return 0;
#endif
}
size_t cudaSourceRelayQuarantinedCountForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    size_t count = 0;
    for (auto const& slot : SourceRelays().slots)
        count += slot.quarantined.load(std::memory_order_acquire) ? 1 : 0;
    return count;
#else
    return 0;
#endif
}
void failNextCudaSourceRelayCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failInstall[SourceUpload].store(true, std::memory_order_release);
#endif
}
void failNextCudaSourceRelayNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failNative[SourceUpload].store(true, std::memory_order_release);
#endif
}
void failNextCudaSourceRelayContextCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failInstall[SourceContexts].store(true,std::memory_order_release);
#endif
}
void failNextCudaSourceRelayProgramCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failInstall[SourcePrograms].store(true,std::memory_order_release);
#endif
}
void failNextCudaSourceRelayScalarCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failInstall[SourceScalars].store(true,std::memory_order_release);
#endif
}
void failNextCudaSourceRelayContextNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failNative[SourceContexts].store(true,std::memory_order_release);
#endif
}
void failNextCudaSourceRelayProgramNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failNative[SourcePrograms].store(true,std::memory_order_release);
#endif
}
void failNextCudaSourceRelayScalarNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failNative[SourceScalars].store(true,std::memory_order_release);
#endif
}
void failNextCudaSourceRelayControlsAllocationForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    failNextCudaParameterFreshProgramAllocationForTesting();
#endif
}
void failNextCudaSourceRelayResampleCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failInstall[SourceResample].store(true, std::memory_order_release);
#endif
}
void failNextCudaSourceRelayResampleNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failNative[SourceResample].store(true, std::memory_order_release);
#endif
}

void failNextCudaSourceRelayNamedCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failInstall[SourceNamedTopology].store(
        true, std::memory_order_release);
#endif
}

void failNextCudaSourceRelayNamedNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    SourceRelays().failNative[SourceNamedTopology].store(
        true, std::memory_order_release);
#endif
}

#ifdef USDGEN_ENABLE_CUDA
bool ExecutionState::UploadAuthoredPlanes(
    UsdGenCurveSetDesc const& curves, CudaSourcePrepared const& preparedSource,
    CudaSourcePreparationOptions const& options,
    cudaStream_t stream,
    UsdGenDiagnostics* diagnostics)
{
    if (!namedChannels.empty())
        return Fail(diagnostics, "authored named channels were uploaded twice");
    std::vector<gpu::CudaNamedChannelPlane> candidate;
    try {
        candidate.reserve(curves.authoredPlanes.size());
        namedChannelHostStaging.reserve(curves.authoredPlanes.size());
        std::map<uint64_t, size_t> sourceCurveById;
        for (size_t curve = 0; curve != curves.curveVertexCounts.size(); ++curve) {
            uint64_t const id = options.idSource == CudaSourceIdSource::Index ||
                    curves.curveId.empty()
                ? static_cast<uint64_t>(curve) : curves.curveId[curve];
            sourceCurveById.emplace(id, curve);
        }
        std::vector<size_t> sourceOffsets(curves.curveVertexCounts.size() + 1, 0);
        for (size_t curve = 0; curve != curves.curveVertexCounts.size(); ++curve)
            sourceOffsets[curve + 1] = sourceOffsets[curve] +
                static_cast<size_t>(curves.curveVertexCounts[curve]);
        for (UsdGenAuthoredPlaneDesc const& authored : curves.authoredPlanes) {
            uint64_t elements = authored.domain == UsdGenAuthoredPlaneDomain::Point
                ? static_cast<uint64_t>(preparedSource.points.size())
                : authored.domain == UsdGenAuthoredPlaneDomain::Primitive
                ? static_cast<uint64_t>(preparedSource.curveVertexCounts.size()) : 1;
            if (elements > std::numeric_limits<size_t>::max() / authored.arity ||
                elements * authored.arity >
                    std::numeric_limits<size_t>::max() / sizeof(uint32_t))
                return Fail(diagnostics, "authored named channel upload size overflows");
            size_t const bytes = static_cast<size_t>(elements * authored.arity) *
                sizeof(uint32_t);
            gpu::CudaNamedChannelPlane plane;
            plane.metadata = {
                authored.name.GetString(),
                authored.type == UsdGenAuthoredPlaneType::Float32
                    ? UsdGenDeviceValueType::Float32
                    : UsdGenDeviceValueType::Int32,
                authored.domain == UsdGenAuthoredPlaneDomain::Point
                    ? UsdGenDeviceDomain::Point
                    : authored.domain == UsdGenAuthoredPlaneDomain::Primitive
                    ? UsdGenDeviceDomain::Primitive
                    : UsdGenDeviceDomain::Groom,
                elements, authored.arity,
                static_cast<uint32_t>(authored.arity * sizeof(uint32_t)), true,
                UsdGenDeviceChannelSemantic::Generic};
            plane.bytes = std::make_unique<gpu::DeviceBuffer<unsigned char>>();
            if (plane.bytes->reset(bytes, memoryReservation) != cudaSuccess)
                return Fail(diagnostics, "authored named channel device allocation failed at " +
                    authored.name.GetString());
            auto& staging = namedChannelHostStaging.emplace_back();
            staging.reserve(static_cast<size_t>(elements * authored.arity));
            auto append = [&](size_t valueIndex) {
                uint32_t bits = 0;
                if (authored.type == UsdGenAuthoredPlaneType::Float32) {
                    float const value = authored.floatValues[valueIndex];
                    std::memcpy(&bits, &value, sizeof(bits));
                } else {
                    int32_t const value = authored.intValues[valueIndex];
                    std::memcpy(&bits, &value, sizeof(bits));
                }
                staging.push_back(bits);
            };
            if (authored.domain == UsdGenAuthoredPlaneDomain::Groom) {
                for (size_t lane = 0; lane != authored.arity; ++lane) append(lane);
            } else {
                for (uint64_t id : preparedSource.curveId) {
                    auto const found = sourceCurveById.find(id);
                    if (found == sourceCurveById.end())
                        return Fail(diagnostics,
                            "authored named channel cannot follow canonical stable ids");
                    size_t const curve = found->second;
                    size_t const first = authored.domain == UsdGenAuthoredPlaneDomain::Point
                        ? sourceOffsets[curve] : curve;
                    size_t const count = authored.domain == UsdGenAuthoredPlaneDomain::Point
                        ? static_cast<size_t>(curves.curveVertexCounts[curve]) : 1;
                    for (size_t element = 0; element != count; ++element)
                        for (size_t lane = 0; lane != authored.arity; ++lane)
                            append((first + element) * authored.arity + lane);
                }
            }
            if (staging.size() * sizeof(uint32_t) != bytes)
                return Fail(diagnostics,
                    "authored named channel canonicalization changed cardinality");
            if (bytes && cudaMemcpyAsync(plane.bytes->data(), staging.data(), bytes,
                                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
                return Fail(diagnostics, "authored named channel upload failed at " +
                    authored.name.GetString());
            if (plane.bytes->recordUse(stream) != cudaSuccess) {
                plane.bytes->quarantine();
                namedChannelsUnproven = true;
                return Fail(diagnostics,
                    "authored named channel completion event failed at " +
                    authored.name.GetString());
            }
            candidate.push_back(std::move(plane));
        }
    } catch (...) {
        return Fail(diagnostics, "authored named channel upload allocation failed");
    }
    namedChannels = std::move(candidate);
    return true;
}

bool ExecutionState::PrepareNoiseFrames(CudaSourcePrepared const& preparedSource,
                                         UsdGenCudaExecutionWorkspace& workspace,
                                         cudaStream_t stream,
                                         UsdGenDiagnostics* diagnostics) {
    if (preparedSource.rootFrames.size() != preparedSource.curveVertexCounts.size())
        return Fail(diagnostics,
            "CUDA Noise requires one authored rest root frame per source curve");
    try {
        auto tangent = std::make_unique<gpu::DeviceBuffer<float3>>();
        auto binormal = std::make_unique<gpu::DeviceBuffer<float3>>();
        auto normal = std::make_unique<gpu::DeviceBuffer<float3>>();
        std::vector<float3> t, b, n;
        t.reserve(preparedSource.rootFrames.size());
        b.reserve(preparedSource.rootFrames.size());
        n.reserve(preparedSource.rootFrames.size());
        for (auto const& frame : preparedSource.rootFrames) {
            t.push_back(make_float3(float(frame[0]), float(frame[1]), float(frame[2])));
            b.push_back(make_float3(float(frame[4]), float(frame[5]), float(frame[6])));
            n.push_back(make_float3(float(frame[8]), float(frame[9]), float(frame[10])));
        }
        size_t const count = t.size();
        if (tangent->reset(count, memoryReservation,
                           UsdGenExecutionResourceKind::Active) != cudaSuccess ||
            binormal->reset(count, memoryReservation,
                            UsdGenExecutionResourceKind::Active) != cudaSuccess ||
            normal->reset(count, memoryReservation,
                          UsdGenExecutionResourceKind::Active) != cudaSuccess)
            return Fail(diagnostics, "CUDA Noise rest root-frame allocation failed");
        bool submitted = false;
        cudaError_t status = cudaSuccess;
        if (count) {
            status = cudaMemcpyAsync(tangent->data(), t.data(),
                                     count * sizeof(float3), cudaMemcpyHostToDevice, stream);
            submitted = status == cudaSuccess;
            if (status == cudaSuccess)
                status = cudaMemcpyAsync(binormal->data(), b.data(),
                                         count * sizeof(float3), cudaMemcpyHostToDevice, stream);
            submitted = submitted || status == cudaSuccess;
            if (status == cudaSuccess)
                status = cudaMemcpyAsync(normal->data(), n.data(),
                                         count * sizeof(float3), cudaMemcpyHostToDevice, stream);
            submitted = submitted || status == cudaSuccess;
        }
        if (status != cudaSuccess) {
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                tangent->quarantine(); binormal->quarantine(); normal->quarantine();
                workspace.impl_->poisoned.store(true, std::memory_order_release);
            }
            return Fail(diagnostics, "CUDA Noise rest root-frame upload failed");
        }
        if (tangent->recordUse(stream) != cudaSuccess ||
            binormal->recordUse(stream) != cudaSuccess ||
            normal->recordUse(stream) != cudaSuccess) {
            if (submitted && cudaStreamSynchronize(stream) != cudaSuccess) {
                tangent->quarantine(); binormal->quarantine(); normal->quarantine();
                workspace.impl_->poisoned.store(true, std::memory_order_release);
            }
            return Fail(diagnostics, "CUDA Noise rest root-frame proof failed");
        }
        noiseTangent = std::move(tangent);
        noiseBinormal = std::move(binormal);
        noiseNormal = std::move(normal);
        return true;
    } catch (...) {
        return Fail(diagnostics, "CUDA Noise rest root-frame allocation failed");
    }
}

// ScatterGrow's generated geometry preserves captured curve order for tile
// publication, but the root-frame ABI is stable-ID addressed.  Keep those
// two orders deliberately separate: ScatterGrowRoots can be unsorted while
// CurveGrow/Noise binary-search their frame IDs.
bool ExecutionState::PrepareScatterGrowFrames(UsdGenCudaExecutionPlan const& plan,
    UsdGenCudaExecutionWorkspace& workspace, cudaStream_t stream,
    UsdGenDiagnostics* diagnostics) {
    // Preserve the established fused Width-only fast path: it does not need
    // a stable-ID frame domain and its legacy reservation stays unchanged.
    if (!plan.geometryValueDag) return true;
    if (!plan.scatterRoots) return Fail(diagnostics,
        "CUDA Scatter->Grow root-frame source is unavailable");
    auto const& roots = *plan.scatterRoots;
    size_t const count = roots.stableIds.size();
    if (roots.rootT.size() != count || roots.rootB.size() != count ||
        roots.rootN.size() != count)
        return Fail(diagnostics,
            "CUDA Scatter->Grow root-frame cardinality is invalid");
    try {
        auto packet = std::make_shared<ScatterGrowFrameHostPacket>();
        std::vector<size_t> order(count);
        std::iota(order.begin(), order.end(), size_t{0});
        std::sort(order.begin(), order.end(), [&roots](size_t a, size_t b) {
            return roots.stableIds[a] < roots.stableIds[b];
        });
        packet->ids.resize(count); packet->ordinals.resize(count);
        packet->t.resize(count); packet->b.resize(count); packet->n.resize(count);
        for (size_t out = 0; out != count; ++out) {
            size_t const in = order[out];
            if (out && roots.stableIds[in] == packet->ids[out - 1])
                return Fail(diagnostics,
                    "CUDA Scatter->Grow root-frame stable IDs must be unique");
            packet->ids[out] = roots.stableIds[in];
            if (in > std::numeric_limits<uint32_t>::max()) return Fail(diagnostics,
                "CUDA Scatter->Grow root-frame ordinal exceeds uint32");
            packet->ordinals[out] = static_cast<uint32_t>(in);
            packet->t[out] = roots.rootT[in]; packet->b[out] = roots.rootB[in]; packet->n[out] = roots.rootN[in];
        }
        // Install before the first async H2D command.  On an error the state
        // (and, if needed, its permanent quarantine) owns this packet until
        // it is safe to retire the device buffers; never rely on pageable
        // copy staging to extend a local vector's lifetime.
        scatterFrameHostPacket = packet;
        scatterFrameQuarantineOwner =
            std::make_unique<std::shared_ptr<const ScatterGrowFrameHostPacket>>(packet);
        auto nextT = std::make_unique<gpu::DeviceBuffer<float3>>();
        auto nextB = std::make_unique<gpu::DeviceBuffer<float3>>();
        auto nextN = std::make_unique<gpu::DeviceBuffer<float3>>();
        auto nextIds = std::make_unique<gpu::DeviceBuffer<uint64_t>>();
        auto nextOrdinals = std::make_unique<gpu::DeviceBuffer<uint32_t>>();
        if (nextT->reset(count, memoryReservation, UsdGenExecutionResourceKind::Active) != cudaSuccess ||
            nextB->reset(count, memoryReservation, UsdGenExecutionResourceKind::Active) != cudaSuccess ||
            nextN->reset(count, memoryReservation, UsdGenExecutionResourceKind::Active) != cudaSuccess ||
            nextIds->reset(count, memoryReservation, UsdGenExecutionResourceKind::Active) != cudaSuccess ||
            nextOrdinals->reset(count, memoryReservation, UsdGenExecutionResourceKind::Active) != cudaSuccess)
            return Fail(diagnostics, "CUDA Scatter->Grow root-frame allocation failed");
        cudaError_t status = cudaSuccess;
        if (count) {
            status = cudaMemcpyAsync(nextT->data(), packet->t.data(), count * sizeof(float3),
                                     cudaMemcpyHostToDevice, stream);
            if (status == cudaSuccess) status = cudaMemcpyAsync(nextB->data(), packet->b.data(),
                count * sizeof(float3), cudaMemcpyHostToDevice, stream);
            if (status == cudaSuccess) status = cudaMemcpyAsync(nextN->data(), packet->n.data(),
                count * sizeof(float3), cudaMemcpyHostToDevice, stream);
            if (status == cudaSuccess) status = cudaMemcpyAsync(nextIds->data(), packet->ids.data(),
                count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
            if (status == cudaSuccess) status = cudaMemcpyAsync(nextOrdinals->data(), packet->ordinals.data(),
                count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
        }
        if (status != cudaSuccess || nextT->recordUse(stream) != cudaSuccess ||
            nextB->recordUse(stream) != cudaSuccess || nextN->recordUse(stream) != cudaSuccess ||
            nextIds->recordUse(stream) != cudaSuccess || nextOrdinals->recordUse(stream) != cudaSuccess) {
            nextT->quarantine(); nextB->quarantine(); nextN->quarantine(); nextIds->quarantine(); nextOrdinals->quarantine();
            // A failed event registration does not say whether queued H2D
            // reads have completed.  This asynchronous source owner must
            // never block to find out: poison and retain the preallocated
            // host holder with the quarantined device allocations.
            workspace.impl_->poisoned.store(true, std::memory_order_release);
            (void)scatterFrameQuarantineOwner.release();
            return Fail(diagnostics, "CUDA Scatter->Grow root-frame upload failed");
        }
        noiseTangent = std::move(nextT); noiseBinormal = std::move(nextB);
        noiseNormal = std::move(nextN); frameStableIds = std::move(nextIds);
        frameCaptureOrdinals = std::move(nextOrdinals);
        scatterFrameQuarantineOwner.reset();
        sourceOptions.hasRootFrame = true;
        return true;
    } catch (...) {
        return Fail(diagnostics, "CUDA Scatter->Grow root-frame allocation failed");
    }
}

bool ExecutionState::Prepare(UsdGenCudaExecutionPlan const& plan,
    UsdGenCudaExecutionWorkspace& workspace, double frame,
    UsdGenDiagnostics* diagnostics, bool finishSource,
    CudaSourcePreparationOptions const* provenOptions,
    UsdGenExecutionMemoryReservation* reservation) {
    if (prepared || failed) return false;
    memoryReservation = reservation;
    auto const& desc = plan.desc;
    auto& execution = *workspace.impl_;
    if (execution.poisoned.load(std::memory_order_acquire)) {
        Fail(diagnostics, "CUDA workspace is poisoned by an unproven source upload"); this->failed = true; return false;
    }
    CudaDeviceScope selected(execution.device);
    if (!selected.selected) { Fail(diagnostics, "cannot select execution workspace device"); this->failed = true; return false; }
    auto const stream = execution.stream;
    if (!std::isfinite(frame) || plan.steps.size() + 1 != desc.nodes.size()) {
        Fail(diagnostics, "invalid frame or mismatched compiled CUDA plan"); this->failed = true; return false;
    }
    // Allocate every terminal RBF transaction slot before any source upload
    // can reach the GPU.  Post-submit failure and completion paths only move
    // a transaction into their own pre-existing slot.
    try { pendingRbf.resize(plan.steps.size()); }
    catch (...) {
        Fail(diagnostics, "CUDA RBF pending-publication slot allocation failed");
        this->failed = true;
        return false;
    }
    if (!execution.description.IsEmpty() && execution.description != desc.description) {
        Fail(diagnostics, "CUDA workspaces cannot be shared between descriptions"); this->failed = true; return false;
    }
    execution.description = desc.description;
    // Reconcile runtime storage only on this description's work owner. Rest
    // identity may survive plan replacement, posed solve state never escapes.
    std::set<SdfPath> live;
    for (auto const& step : plan.steps) {
        live.insert(step->path);
        auto& runtime = execution.steps[step->path];
        if (!runtime) runtime = std::make_unique<UsdGenCudaExecutionWorkspace::Impl::Step>();
        if (step->type != TfToken("UsdGenDeform")) runtime->rbf.reset();
        // A rest-key change is itself an asynchronous candidate.  Do not
        // discard this description's accepted cache before that candidate
        // survives bind, solve, deformation, and final publication.
        else if (!runtime->rbf) runtime->rbf = std::make_unique<CudaRbfCache>();
    }
    for (auto it = execution.steps.begin(); it != execution.steps.end(); )
        if (!live.count(it->first)) it = execution.steps.erase(it); else ++it;
    auto const& node = desc.nodes[plan.sourceNode];
    bool const referenceSource = node.type == TfToken("UsdGenReferenceSource");
    SdfPath const sourcePath = referenceSource
        ? node.references.front() : node.curves.front();
    auto found = std::find_if(desc.curveSets.begin(), desc.curveSets.end(),
        [&](auto const& c) { return c.path == sourcePath; });
    if (found == desc.curveSets.end()) {
        Fail(diagnostics, "missing C3 input " + sourcePath.GetString()); this->failed = true; return false;
    }
    // surfaceCage expands sparse rails on this CPU source-preparation
    // boundary. CUDA only uploads the resulting transient ordinary C3 data;
    // no worker consults a stage or a map relationship.
    UsdGenCurveSetDesc surfaceCageSource;
    UsdGenCurveSetDesc const* preparedInput = &*found;
    TfToken const sourceInterpolationMode = UsdGenParamView{&desc, &node}.GetToken(
        TfToken("interpolationMode"), TfToken("none"));
    if (node.enabled && sourceInterpolationMode == TfToken("surfaceCage")) {
        std::string error;
        if (!UsdGenBuildSurfaceCageCurveSet(desc, node, *found,
                                             &surfaceCageSource, &error)) {
            Fail(diagnostics, "surfaceCage source preparation failed: " + error);
            this->failed = true;
            return false;
        }
        if (surfaceCageSource.curveVertexCounts.size() > plan.surfaceCageMaxCurves ||
            surfaceCageSource.points.size() > plan.surfaceCageMaxCvs) {
            Fail(diagnostics,
                "surfaceCage source preparation exceeded its admitted dense capacity");
            this->failed = true;
            return false;
        }
        preparedInput = &surfaceCageSource;
    }
    // 02 §6.3: a disabled generator publishes an EMPTY curve set. Substitute
    // an empty C3 input, keeping the authored topology metadata so the rest of
    // this preparation validates the same shape it always did; every step
    // downstream then runs over zero curves, as it does on the CPU lane.
    UsdGenCurveSetDesc emptySource;
    if (!node.enabled) {
        emptySource.path = found->path;
        emptySource.role = found->role;
        emptySource.curveRole = found->curveRole;
        emptySource.type = found->type;
        emptySource.basis = found->basis;
        emptySource.wrap = found->wrap;
        emptySource.widthsInterpolation = found->widthsInterpolation;
        emptySource.worldMatrix = found->worldMatrix;
        emptySource.frozenEpoch = found->frozenEpoch;
        emptySource.curveGeneration = found->curveGeneration;
    }
    auto const& curves = node.enabled ? *preparedInput : emptySource;
    if (curves.type != TfToken("cubic") ||
        (curves.basis != TfToken("bspline") && curves.basis != TfToken("catmullRom")) ||
        (curves.wrap != TfToken("pinned") && curves.wrap != TfToken("nonperiodic"))) {
        Fail(diagnostics, "C3 requires cubic bspline/catmullRom curves with pinned/nonperiodic wrap"); this->failed = true; return false;
    }
    if (curves.wrap == TfToken("nonperiodic") && diagnostics)
        diagnostics->Warn("CUDA CurveSource promotes nonperiodic wrap to pinned; endpoints now reach the first/last CV");
    this->curveTopology = {
        UsdGenDeviceCurveType::Cubic,
        curves.basis == TfToken("catmullRom")
            ? UsdGenDeviceCurveBasis::CatmullRom
            : UsdGenDeviceCurveBasis::BSpline,
        UsdGenDeviceCurveWrap::Pinned};
    if (!curves.widths.empty() &&
        curves.widthsInterpolation != TfToken("vertex") && curves.widthsInterpolation != TfToken("constant")) {
        Fail(diagnostics, "C3 widths must use vertex or constant interpolation"); this->failed = true; return false;
    }
    if (!curves.widths.empty() &&
        ((curves.widthsInterpolation == TfToken("constant") && curves.widths.size() != 1) ||
         (curves.widthsInterpolation == TfToken("vertex") && curves.widths.size() != curves.points.size()))) {
        Fail(diagnostics, "C3 widths cardinality does not match interpolation"); this->failed = true; return false;
    }
    if (!referenceSource && curves.curveRole != TfToken("hair")) {
        Fail(diagnostics, "CurveSource requires the C3 hair role"); this->failed = true; return false;
    }
    UsdGenParamView params; params.desc = &desc; params.node = &node;
    std::vector<uint32_t> restOrder;
    std::vector<uint32_t> restInput(desc.nodes.size(), UINT32_MAX);
    restOrder.reserve(plan.steps.size());
    for (auto const& step : plan.steps) {
        restOrder.push_back(step->semanticNode);
        if (step->semanticNode < restInput.size())
            restInput[step->semanticNode] = step->inputNode;
    }
    bool const noiseNeedsAuthoredRest = NoiseNeedsAuthoredSourceRest(
        desc, plan.sourceNode, restOrder, restInput);
    if (referenceSource) {
        CudaSourcePreparationOptions options;
        options.defaultWidth = desc.defaultWidth;
        options.useRest = false;
        options.idSource = CudaSourceIdSource::Primvar;
        options.rebind = "never";
        CudaSourcePreparationInput input;
        input.curveVertexCounts.assign(curves.curveVertexCounts.begin(),
                                       curves.curveVertexCounts.end());
        for (auto const& p : curves.points)
            input.points.push_back(make_float3(p[0], p[1], p[2]));
        input.widths.assign(curves.widths.begin(), curves.widths.end());
        input.curveId.assign(curves.curveId.begin(), curves.curveId.end());
        if (!curves.skinPrim.empty() || !curves.skinPrimUv.empty()) {
            if (curves.skinPrim.size() != curves.curveVertexCounts.size() ||
                curves.skinPrimUv.size() != curves.curveVertexCounts.size()) {
                Fail(diagnostics,
                    "UsdGenReferenceSource root bindings must provide skinprim and skinprimuv for every curve");
                this->failed = true;
                return false;
            }
            input.rootPrim.assign(curves.skinPrim.begin(), curves.skinPrim.end());
            for (auto const& uv : curves.skinPrimUv)
                input.rootUV.push_back(make_float2(uv[0], uv[1]));
        }
        CudaSourcePrepared preparedSource;
        std::vector<std::string> messages;
        if (PrepareCudaSource(input, options, &preparedSource, &messages) !=
                CudaSourcePreparationStatus::Ok) {
            for (auto const& message : messages) Fail(diagnostics, message);
            if (messages.empty()) Fail(diagnostics,
                "UsdGenReferenceSource reference validation failed");
            this->failed = true;
            return false;
        }
        if (diagnostics) for (auto const& message : messages)
            diagnostics->Warn(message);
        if (!PrepareCudaJobSource(*this, preparedSource, options, stream, diagnostics,
                                  finishSource)) {
            this->failed = true;
            return false;
        }
        if (!finishSource) {
            sourceOptions = options;
            sourceUploadPending = true;
            return true;
        }
        geometry = source->view();
        captureStableIds = geometry.stableIds;
        captureCurveCount = geometry.curveCount;
        hairT = source->hairT();
        rootPrim = source->rootPrim();
        rootUV = source->rootUV();
        deformed = false;
        this->prepared = true;
        return true;
    }
    CudaSourcePreparationOptions options;
    options.defaultWidth = desc.defaultWidth;
    options.useRest = params.GetBool(TfToken("useRest"), true);
    options.resampleTo = params.GetInt(TfToken("resampleTo"), 0);
    if (provenOptions) {
        // Async source controls have already completed their own native proof
        // and compact scalar readback.  Never evaluate/read back stale
        // workspace fields in this path.
        options.useRest = provenOptions->useRest;
        options.resampleTo = provenOptions->resampleTo;
    } else if (plan.sourceParameters) {
        expr::Context context; context.frame = frame; context.time = frame / desc.timeCodesPerSecond;
        context.seed = node.seed; context.descId = expr::DescriptionId(desc.description.GetText());
        std::vector<std::string> errors;
        if (!execution.sourceParameters)
            execution.sourceParameters = std::make_unique<CudaParameterEvaluator>();
        if (execution.sourceParameters->Evaluate(*plan.sourceParameters, {}, {},
                context, stream, &errors, memoryReservation,
                UsdGenExecutionResourceKind::Cache) != CudaParameterStatus::Ok) {
            for (auto const& error : errors) Fail(diagnostics, error);
            Fail(diagnostics, "source expression evaluation failed; previous generation retained"); this->failed = true; return false;
        }
        if (auto const* field = execution.sourceParameters->Find(TfToken("useRest"))) {
            uint8_t value = 0;
            if (field->type != expr::ScalarType::Bool || field->count != 1 ||
                cudaMemcpyAsync(&value, field->data, sizeof(value), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess) {
                Fail(diagnostics, "invalid useRest expression value; previous generation retained"); this->failed = true; return false;
            }
            if (value > 1) { Fail(diagnostics, "invalid useRest expression value; previous generation retained"); this->failed = true; return false; }
            options.useRest = value != 0;
        }
        if (auto const* field = execution.sourceParameters->Find(TfToken("resampleTo"))) {
            int value = 0;
            if (field->type != expr::ScalarType::Int32 || field->count != 1 ||
                cudaMemcpyAsync(&value, field->data, sizeof(value), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess || value < 0 || value == 1) {
                Fail(diagnostics, "invalid resampleTo expression value; previous generation retained"); this->failed = true; return false;
            }
            options.resampleTo = value;
        }
    }
    const bool hasRbfDeform = std::any_of(plan.steps.begin(), plan.steps.end(),
        [](auto const& step) { return step->type == TfToken("UsdGenDeform"); });
    if (!options.useRest && hasRbfDeform) {
        Fail(diagnostics, "effective useRest=false cannot feed CUDA RBF Deform; previous generation retained");
        this->failed = true; return false;
    }
    if (options.useRest && !curves.points.empty() &&
        (curves.rest.empty() || curves.restFromCurrentPoints)) {
        Fail(diagnostics, "useRest requires an authored/default-time C3 rest snapshot; current-frame fallback is not a rest binding");
        this->failed = true; return false;
    }
    if (noiseNeedsAuthoredRest && !options.useRest && !curves.points.empty() &&
        (curves.rest.empty() || curves.restFromCurrentPoints)) {
        Fail(diagnostics,
            "CUDA Noise requires an authored/default-time C3 rest snapshot when source useRest=false");
        this->failed = true; return false;
    }
    auto idSource = params.GetToken(TfToken("idSource"), TfToken("primvar"));
    if (idSource != TfToken("primvar") && idSource != TfToken("index")) {
        Fail(diagnostics, "invalid idSource"); this->failed = true; return false;
    }
    options.idSource = idSource == TfToken("index") ? CudaSourceIdSource::Index : CudaSourceIdSource::Primvar;
    auto staleAction = params.GetToken(TfToken("staleAction"), TfToken("warn"));
    if (staleAction != TfToken("warn") && staleAction != TfToken("ignore") && staleAction != TfToken("block")) {
        Fail(diagnostics, "invalid staleAction"); this->failed = true; return false;
    }
    options.staleAction = staleAction == TfToken("block") ? CudaSourceStaleAction::Block :
        staleAction == TfToken("ignore") ? CudaSourceStaleAction::Ignore : CudaSourceStaleAction::Warn;
    auto epoch = params.GetVtValue(TfToken("expectEpoch"), VtValue(std::string{}));
    if (!epoch.IsHolding<std::string>()) { Fail(diagnostics, "expectEpoch must be a string"); this->failed = true; return false; }
    options.expectedEpoch = epoch.UncheckedGet<std::string>();
    options.actualEpoch = curves.frozenEpoch;
    // A connected groom scalar is the effective value. Do not reject its
    // authored literal fallback before the evaluator has had a chance to
    // override it; source preparation itself never host-resamples.
    options.rebind = params.GetToken(TfToken("rebind"), TfToken("onError")).GetString();
    const UsdGenSurfaceDesc* surface = nullptr;
    if (!node.surfaces.empty()) {
        auto s = std::find_if(desc.surfaces.begin(), desc.surfaces.end(),
            [&](auto const& value) { return value.path == node.surfaces.front(); });
        if (s == desc.surfaces.end()) { Fail(diagnostics, "missing CurveSource binding surface"); this->failed = true; return false; }
        surface = &*s;
    }
    UsdGenCurveRootCaptureResult capturedRoots;
    std::string rootError;
    std::shared_ptr<const UsdGenRestSurfaceBindingCache> bindingCache =
        execution.sourceBindingCache;
    if (!UsdGenCaptureCurveRoots(curves, surface, TfToken(options.rebind),
                                  &capturedRoots, &rootError, &bindingCache)) {
        this->failed = true;
        return Fail(diagnostics, "CUDA C3 root capture failed: " + rootError);
    }
    // Commit only the validated candidate after capture succeeds. The old
    // workspace cache remains untouched if capture or binding resolution
    // fails, preserving COW reuse for the last accepted generation.
    if (bindingCache && bindingCache != execution.sourceBindingCache)
        execution.sourceBindingCache = std::move(bindingCache);
    // Keep original-order IDs until PrepareCudaSource has validated and
    // canonicalized every source channel. Dropping earlier would renumber
    // synthesized IDs or hide invalid source data.
    std::vector<std::array<double, 16>> rootFrames;
    std::vector<uint64_t> retainedStableIds;
    bool const helperFrames = capturedRoots.dropped != 0;
    rootFrames.reserve(capturedRoots.frames.size());
    retainedStableIds.reserve(capturedRoots.frames.size());
    for (size_t curve = 0; curve != capturedRoots.frames.size(); ++curve) {
        std::array<double, 16> raw{};
        for (int row = 0; row != 4; ++row)
            for (int column = 0; column != 4; ++column)
                raw[size_t(row)*4 + size_t(column)] = capturedRoots.frames[curve][row][column];
        rootFrames.push_back(raw);
        if (capturedRoots.valid[curve]) {
            if (options.idSource != CudaSourceIdSource::Index && !curves.curveId.empty() &&
                curves.curveId.size() != curves.curveVertexCounts.size()) {
                this->failed = true;
                return Fail(diagnostics, "CUDA C3 curve IDs have invalid cardinality");
            }
            retainedStableIds.push_back(options.idSource == CudaSourceIdSource::Index ||
                curves.curveId.empty() ? uint64_t(curve) : curves.curveId[curve]);
        }
    }
    if (capturedRoots.dropped && diagnostics)
        diagnostics->Warn("CUDA source dropped " + std::to_string(capturedRoots.dropped) +
            " curves with unresolved or degenerate root captures");
    options.rebind = "never"; // The shared capture has resolved policy.
    options.hasRootFrame = !rootFrames.empty();
    CudaSourcePreparationInput input;
    input.curveVertexCounts.assign(curves.curveVertexCounts.begin(), curves.curveVertexCounts.end());
    for (auto const& p : curves.points) input.points.push_back(make_float3(p[0], p[1], p[2]));
    for (auto const& p : curves.rest) input.rest.push_back(make_float3(p[0], p[1], p[2]));
    input.widths.assign(curves.widths.begin(), curves.widths.end());
    input.curveId.assign(curves.curveId.begin(), curves.curveId.end());
    input.rootFrames = std::move(rootFrames);
    input.rootPrim.assign(capturedRoots.rootPrim.begin(), capturedRoots.rootPrim.end());
    for (auto const& uv : capturedRoots.rootUV)
        input.rootUV.push_back(make_float2(uv[0],uv[1]));
    CudaSourcePrepared preparedSource;
    std::vector<std::string> messages;
    if (PrepareCudaSource(input, options, &preparedSource, &messages) != CudaSourcePreparationStatus::Ok) {
        for (auto const& message : messages) Fail(diagnostics, message);
        if (messages.empty()) Fail(diagnostics, "C3 source validation failed");
        this->failed = true; return false;
    }
    if (diagnostics) for (auto const& message : messages) diagnostics->Warn(message);
    if (helperFrames &&
        !CompactCudaSource(&preparedSource, retainedStableIds)) {
        this->failed = true;
        return Fail(diagnostics,
            "CUDA source root-frame compaction failed while retaining valid stable IDs");
    }
    if (!PrepareCudaJobSource(*this, preparedSource, options, stream, diagnostics, finishSource)) {
        this->failed = true;
        return false;
    }
    if (!UploadAuthoredPlanes(curves, preparedSource, options, stream, diagnostics)) {
        this->failed = true;
        return false;
    }
    if (std::any_of(plan.steps.begin(), plan.steps.end(),
            [](auto const& step) { return step->type == TfToken("UsdGenNoise") ||
                                        step->type == TfToken("UsdGenGrow"); }) &&
        !PrepareNoiseFrames(preparedSource, workspace, stream, diagnostics)) {
        this->failed = true;
        return false;
    }
    if (finishSource && resampled && !namedChannels.empty()) {
        std::vector<gpu::CudaNamedChannelTopologyCandidate::RawInputPlane> inputs;
        try {
            inputs.reserve(namedChannels.size());
            for (auto const& plane : namedChannels) {
                if (!plane.bytes) {
                    this->failed = true;
                    return Fail(diagnostics,
                        "CUDA source named channel has no private owner");
                }
                inputs.push_back({plane.metadata,
                    {plane.bytes->data(), plane.bytes->size()}, plane.bytes.get()});
            }
        } catch (...) {
            this->failed = true;
            return Fail(diagnostics,
                "CUDA source named topology input allocation failed");
        }
        gpu::CudaNamedChannelTopologyCandidate candidate;
        std::string reason;
        auto completion = std::make_unique<CudaInlineCompletion>();
        auto preserveRawOwnersOnProofLoss = [&] {
            // Raw candidates borrow source/resample geometry and authored
            // plane owners. Keep every owner conservative if this stream
            // cannot prove that the candidate has stopped reading them.
            if (source) source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->source) sourceTopologyOwners->source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->resampled) sourceTopologyOwners->resampled->MarkUnprovenUpload();
            if (resampled) resampled->MarkUnprovenUpload();
            for (auto const& plane : namedChannels)
                if (plane.bytes) plane.bytes->quarantine();
            namedChannelsUnproven = true;
            workspace.impl_->poisoned.store(true, std::memory_order_release);
        };
        auto proveRawOrPreserve = [&] {
            if (cudaStreamSynchronize(stream) == cudaSuccess) return true;
            preserveRawOwnersOnProofLoss();
            return false;
        };
        if (candidate.BeginRaw(source->view(), resampled->view(),
                std::move(inputs), stream,
                gpu::CudaNamedChannelTopologyMode::Resample, {}, &reason,
                memoryReservation) != cudaSuccess) {
            candidate.Quarantine();
            proveRawOrPreserve();
            this->failed = true;
            return Fail(diagnostics, reason.empty()
                ? "CUDA source named topology synchronization failed" : reason);
        }
        if (candidate.FinishAsync(CudaInlineCompletionCallback,
                                  completion.get()) != cudaSuccess) {
            candidate.Quarantine();
            proveRawOrPreserve();
            this->failed = true;
            return Fail(diagnostics,
                "CUDA source named topology callback installation failed");
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            candidate.Quarantine();
            // The callback is armed and CUDA may still dereference userdata
            // after a failed synchronization; retain it with the quarantined
            // device work instead of returning a dangling stack address.
            completion.release();
            preserveRawOwnersOnProofLoss();
            this->failed = true;
            return Fail(diagnostics,
                "CUDA source named topology synchronization failed");
        }
        std::vector<gpu::CudaNamedChannelPlane> rebuilt;
        auto const status = cudaError_t(completion->status.load(
            std::memory_order_acquire));
        if (!candidate.CommitPlanes(status, &rebuilt, &reason)) {
            this->failed = true;
            return Fail(diagnostics, reason.empty()
                ? "CUDA source named topology commit failed" : reason);
        }
        namedChannels = std::move(rebuilt);
    }
    if (!finishSource) {
        sourceOptions = options;
        sourceUploadPending = true;
        return true;
    }
    auto& sourceRef = this->source;
    auto& resampledRef = this->resampled;
    // PrepareCudaSource establishes this strictly ordered capture identity.
    // Keep the borrowed view alive through tile assignment even if a later
    // Length pass compacts the survivor geometry.
    this->hairT = resampledRef ? resampledRef->hairT() : sourceRef->hairT();
    this->rootPrim = resampledRef ? resampledRef->rootPrim() : sourceRef->rootPrim();
    this->rootUV = resampledRef ? resampledRef->rootUV() : sourceRef->rootUV();
    this->deformed = !options.useRest;
    this->prepared = true;
    return true;
}

bool ExecutionState::CommitSourceUpload(UsdGenCudaExecutionWorkspace& workspace,
    UsdGenDiagnostics* diagnostics) {
    if (!sourceUploadPending || !source || workspace.impl_->poisoned.load())
        return Fail(diagnostics, "source async commit precondition failed (sourceUploadPending=" +
            std::to_string(sourceUploadPending) + ", sourcePresent=" + std::to_string(bool(source)) +
            ", workspacePoisoned=" + std::to_string(workspace.impl_->poisoned.load()) + ")");
    CudaDeviceScope selected(workspace.impl_->device);
    if (!selected.selected) return Fail(diagnostics, "cannot select CUDA device for source completion: " + selected.FailureDetail());
    auto const status = source->CommitFreshFinish();
    if (status != gpu::CurveSourceStatus::Ok)
        return Fail(diagnostics, "source async commit failed; previous generation retained (CurveSourceStatus " +
            CurveSourceStatusText(status) + ")");
    if (sourceOptions.resampleTo) {
        return true;
    }
    geometry = resampled ? resampled->view() : source->view();
    captureStableIds = geometry.stableIds; captureCurveCount = geometry.curveCount;
    hairT = resampled ? resampled->hairT() : source->hairT();
    rootPrim = resampled ? resampled->rootPrim() : source->rootPrim();
    rootUV = resampled ? resampled->rootUV() : source->rootUV();
    deformed = !sourceOptions.useRest;
    sourceUploadPending = false; prepared = true;
    return true;
}
bool ExecutionState::StartSourceResampleAsync(UsdGenCudaExecutionWorkspace& workspace,
    void (*gate)(cudaStream_t, cudaError_t, void*) noexcept, void* gateData,
    void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata,
    bool injectInstallFailure, bool* submitted) {
    if (submitted) *submitted = false;
    if (!sourceUploadPending || !source || !sourceOptions.resampleTo ||
        workspace.impl_->poisoned.load(std::memory_order_acquire)) return false;
    CudaDeviceScope selected(workspace.impl_->device);
    if (!selected.selected) return false;
    auto const stream = workspace.impl_->stream;
    resampled = std::make_unique<gpu::CudaCurveResample>();
    if (resampled->ApplyFresh(source->view(), source->hairT(), source->rootPrim(),
            source->rootUV(), sourceOptions.resampleTo, stream,
            memoryReservation) != gpu::CurveResampleStatus::Ok) return false;
    if (submitted) *submitted = true;
    if (injectInstallFailure) return false;
    if (gate && cudaStreamAddCallback(stream, gate, gateData, 0) != cudaSuccess) return false;
    return resampled->FinishFreshAsync(stream, callback, userdata) == gpu::CurveResampleStatus::Ok;
}
bool ExecutionState::CompleteSourceResample(UsdGenCudaExecutionWorkspace& workspace,
    UsdGenDiagnostics* diagnostics, std::shared_ptr<const void> lifetime) {
    if (!sourceUploadPending || !source || !resampled || !sourceOptions.resampleTo ||
        workspace.impl_->poisoned.load(std::memory_order_acquire)) return false;
    CudaDeviceScope selected(workspace.impl_->device);
    if (!selected.selected) return Fail(diagnostics, "cannot select CUDA device for source resample completion");
    if (resampled->CommitFreshFinish() != gpu::CurveResampleStatus::Ok)
        return Fail(diagnostics, "CUDA source resample failed; previous generation retained");
    geometry = resampled->view(); captureStableIds = geometry.stableIds; captureCurveCount = geometry.curveCount;
    hairT = resampled->hairT(); rootPrim = resampled->rootPrim(); rootUV = resampled->rootUV();
    deformed = !sourceOptions.useRest;
    if (!namedChannels.empty()) {
        std::vector<gpu::CudaNamedChannelTopologyCandidate::RawInputPlane> inputs;
        try {
            inputs.reserve(namedChannels.size());
            for (auto const& plane : namedChannels) {
                if (!plane.bytes)
                    return Fail(diagnostics, "CUDA source named channel has no private owner");
                inputs.push_back({plane.metadata,
                    {plane.bytes->data(), plane.bytes->size()}, plane.bytes.get()});
            }
            sourceNamedTopology =
                std::make_unique<gpu::CudaNamedChannelTopologyCandidate>();
        } catch (...) {
            return Fail(diagnostics, "CUDA source named topology candidate allocation failed");
        }
        std::string reason;
        if (sourceNamedTopology->BeginRaw(source->view(), geometry,
                std::move(inputs), workspace.impl_->stream,
                gpu::CudaNamedChannelTopologyMode::Resample, std::move(lifetime), &reason,
                memoryReservation) != cudaSuccess) {
            // Keep the candidate (and its raw-input lifetime token) alive
            // through relay cleanup.  BeginRaw may have submitted work before
            // reporting an enqueue failure; resetting it here would let the
            // source/resample/plane owners destruct while CUDA still reads
            // their borrowed views.
            if (sourceNamedTopology->HasUnprovenWork())
                namedChannelsUnproven = true;
            return Fail(diagnostics, reason.empty()
                ? "CUDA source named topology submission failed" : reason);
        }
        // The old COW planes and source geometry remain job-owned until the
        // candidate's distinct terminal callback is proved and committed.
        return true;
    }
    sourceUploadPending = false; prepared = true;
    return true;
}

bool ExecutionState::StartSourceNamedTopologyAsync(
    UsdGenCudaExecutionWorkspace& workspace,
    void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata,
    bool injectInstallFailure, bool* submitted) {
    if (submitted) *submitted = false;
    if (!sourceUploadPending || !sourceNamedTopology ||
        workspace.impl_->poisoned.load(std::memory_order_acquire)) return false;
    CudaDeviceScope selected(workspace.impl_->device);
    if (!selected.selected) return false;
    if (submitted) *submitted = true;
    if (injectInstallFailure) {
        sourceNamedTopology->Quarantine();
        return false;
    }
    return sourceNamedTopology->FinishAsync(callback, userdata) == cudaSuccess;
}

bool ExecutionState::CompleteSourceNamedTopology(
    cudaError_t completionStatus, UsdGenCudaExecutionWorkspace& workspace,
    UsdGenDiagnostics* diagnostics) {
    if (!sourceUploadPending || !sourceNamedTopology ||
        workspace.impl_->poisoned.load(std::memory_order_acquire)) return false;
    CudaDeviceScope selected(workspace.impl_->device);
    if (!selected.selected)
        return Fail(diagnostics, "cannot select CUDA device for source named topology completion");
    std::vector<gpu::CudaNamedChannelPlane> rebuilt;
    std::string reason;
    if (!sourceNamedTopology->CommitPlanes(completionStatus, &rebuilt, &reason))
        return Fail(diagnostics, reason.empty()
            ? "CUDA source named topology completion failed" : reason);
    namedChannels = std::move(rebuilt);
    sourceNamedTopology.reset();
    sourceUploadPending = false;
    prepared = true;
    return true;
}
bool ExecutionState::BeginCurveGrow(gpu::CurveGrowControls const& controls,
    cudaStream_t stream, UsdGenDiagnostics* diagnostics) {
    std::shared_ptr<TopologyOwners> sourceOwners;
    sourceOwners = sourceTopologyOwners;
    if (curveGrow) return Fail(diagnostics, "C3 Grow is already active");
    if (!sourceOwners) {
        // Established linear execution keeps its source owners in state and
        // transfers them into the native lifetime exactly as before.
        if (!source || !noiseTangent || !noiseBinormal || !noiseNormal)
            return Fail(diagnostics, "C3 Grow requires completed source and root frame owners");
        auto inputs = std::make_shared<CurveGrowInputOwners>();
        inputs->source = std::move(source); inputs->resampled = std::move(resampled);
        inputs->t = std::move(noiseTangent); inputs->b = std::move(noiseBinormal);
        inputs->n = std::move(noiseNormal);
        growInputs = inputs;
        curveGrow = std::make_unique<gpu::CudaCurveGrow>();
        gpu::CurveGrowInput input{geometry, rootPrim, rootUV,
            {inputs->t->data(), inputs->t->size()}, {inputs->b->data(), inputs->b->size()},
            {inputs->n->data(), inputs->n->size()}, {}};
        if (curveGrow->BeginFresh(input, inputs, controls, stream,
                memoryReservation) != gpu::CurveGrowStatus::Ok)
            return Fail(diagnostics, "CUDA C3 Grow launch failed");
        return true;
    }
    if (!sourceOwners->source || !sourceOwners->t || !sourceOwners->b || !sourceOwners->n)
        return Fail(diagnostics, "C3 Grow requires completed source and root frame owners");
    auto candidate = std::make_unique<gpu::CudaCurveGrow>();
    // Preserve the source bundle for source-side branches. The Grow native
    // lifetime holds a shared alias rather than stealing those owners.
    growInputs.reset();
    curveGrow = std::move(candidate);
    gpu::CurveGrowInput input{geometry, rootPrim, rootUV,
        {sourceOwners->t->data(), sourceOwners->t->size()},
        {sourceOwners->b->data(), sourceOwners->b->size()},
        {sourceOwners->n->data(), sourceOwners->n->size()}, {}};
    if (curveGrow->BeginFresh(input, sourceOwners, controls, stream,
            memoryReservation) != gpu::CurveGrowStatus::Ok)
        return Fail(diagnostics, "CUDA C3 Grow launch failed");
    return true;
}

bool ExecutionState::BeginGrowNamed(cudaStream_t stream,
    std::shared_ptr<const void> lifetime, std::shared_ptr<NamedOwners> owners,
    UsdGenDiagnostics* diagnostics) {
    auto const* planes = owners ? &owners->planes : &namedChannels;
    if (planes->empty()) return true;
    std::vector<gpu::CudaNamedChannelTopologyCandidate::RawInputPlane> inputs;
    inputs.reserve(planes->size());
    for (auto const& plane : *planes) {
        if (!plane.bytes || plane.bytes->waitOn(stream) != cudaSuccess)
            return Fail(diagnostics, "CUDA Grow named input wait failed");
        inputs.push_back({plane.metadata, {plane.bytes->data(), plane.bytes->size()}, plane.bytes.get()});
    }
    growNamedTopology = std::make_unique<gpu::CudaNamedChannelTopologyCandidate>();
    std::string reason;
    if (growNamedTopology->BeginRaw(geometry, curveGrow->view(), std::move(inputs),
            stream, gpu::CudaNamedChannelTopologyMode::Resample, std::move(lifetime),
            &reason, memoryReservation) != cudaSuccess) {
        if (!growNamedTopology->HasUnprovenWork()) growNamedTopology.reset();
        return Fail(diagnostics, "CUDA Grow named topology failed: " + reason);
    }
    return true;
}

bool ExecutionState::AcceptCurveGrow(UsdGenCudaExecutionPlan const& plan, uint32_t semanticNode, cudaError_t status,
    std::shared_ptr<NamedOwners> inputNames, UsdGenDiagnostics* diagnostics) {
    if (growNamedTopology) {
        std::vector<gpu::CudaNamedChannelPlane> rebuilt;
        std::string reason;
        if (!growNamedTopology->CommitPlanes(status, &rebuilt, &reason)) {
            // The raw candidate retains its job while native work is live.
            // Break that self-retention after a proved semantic rejection;
            // unproved failures remain retained by the quarantine relay.
            if (!growNamedTopology->HasUnprovenWork()) growNamedTopology.reset();
            return Fail(diagnostics, "CUDA Grow named completion failed: " + reason);
        }
        PreserveSourceNamedChannels(plan);
        if (plan.geometryValueDag) {
            if (namedOwners.size() != plan.desc.nodes.size())
                return Fail(diagnostics, "CUDA Grow named bundle storage is unavailable");
            auto owners = std::make_shared<NamedOwners>();
            owners->planes = std::move(rebuilt);
            namedOwners[semanticNode] = owners;
        } else namedChannels = std::move(rebuilt);
        growNamedTopology.reset();
    }
    if (plan.geometryValueDag && semanticNode < namedOwners.size() &&
        !namedOwners[semanticNode])
        namedOwners[semanticNode] = std::move(inputNames);
    if (plan.geometryValueDag) {
        if (topologyOwners.size() != plan.desc.nodes.size())
            return Fail(diagnostics, "CUDA Grow topology bundle storage is unavailable");
        auto owners = std::make_shared<TopologyOwners>();
        owners->grow = std::move(curveGrow);
        topologyOwners[semanticNode] = owners;
        geometry = owners->grow->view(); hairT = owners->grow->hairT();
        rootPrim = owners->grow->rootPrim(); rootUV = owners->grow->rootUV();
    } else {
        geometry = curveGrow->view(); hairT = curveGrow->hairT();
        rootPrim = curveGrow->rootPrim(); rootUV = curveGrow->rootUV();
    }
    // Grow preserves source curve order/IDs. Retain growInputs until terminal
    // tile construction has finished reading captureStableIds.
    if (semanticNode < widthValues.size()) widthValues[semanticNode].view = geometry.widths;
    if (plan.taskDag && semanticNode < geometryValues.size()) {
        auto& value = geometryValues[semanticNode];
        value.geometry = geometry;
        value.hairT = hairT;
        value.rootPrim = rootPrim;
        value.rootUV = rootUV;
        value.curveTopology = {UsdGenDeviceCurveType::Cubic,
                               UsdGenDeviceCurveBasis::BSpline,
                               UsdGenDeviceCurveWrap::Pinned};
        value.deformed = deformed;
        value.pointOrigin = geometry.points;
        value.widthOrigin = geometry.widths;
        value.pointOwnerNode = semanticNode;
        value.widthOwnerNode = semanticNode;
        value.topologyOwnerNode = semanticNode;
        value.namedOwnerNode = semanticNode;
        value.topologyOwners = plan.geometryValueDag ? topologyOwners[semanticNode] : nullptr;
        value.namedOwners = plan.geometryValueDag && semanticNode < namedOwners.size()
            ? namedOwners[semanticNode] : nullptr;
        if (plan.geometryValueDag && semanticNode < geometryValues.size()) {
            value.sourceFrameDomain = sourceTopologyOwners;
            value.sourceFramesPresent = sourceOptions.hasRootFrame;
        }
        value.ready = true;
    }
    curveTopology = {UsdGenDeviceCurveType::Cubic, UsdGenDeviceCurveBasis::BSpline,
                     UsdGenDeviceCurveWrap::Pinned};
    return true;
}

bool ExecutionState::BeginCurveGrowValue(gpu::CurveGrowControls const& controls,
    gpu::DeviceCurveGeometryView inputGeometry,
    gpu::DeviceView<const int32_t> inputRootPrim, gpu::DeviceView<const float2> inputRootUV,
    std::shared_ptr<TopologyOwners> inputOwners,
    std::shared_ptr<TopologyOwners> sourceFrameOwners,
    gpu::DeviceView<const uint64_t> sourceFrameIds, std::shared_ptr<NamedOwners> inputNames,
    std::shared_ptr<const void> jobLifetime, OperatorGrowCandidate* candidate,
    cudaStream_t stream, UsdGenDiagnostics* diagnostics) {
    if (!candidate || candidate->grow || !inputOwners || !sourceFrameOwners ||
        !sourceFrameOwners->source || !sourceFrameOwners->t ||
        !sourceFrameOwners->b || !sourceFrameOwners->n)
        return Fail(diagnostics, "CUDA Grow value input owners are unavailable");
    candidate->inputLifetime = std::make_shared<GrowValueInputLifetime>();
    candidate->inputLifetime->topology = std::move(inputOwners);
    candidate->inputLifetime->sourceFrames = std::move(sourceFrameOwners);
    candidate->inputLifetime->names = std::move(inputNames);
    candidate->inputLifetime->job = std::move(jobLifetime);
    if (candidate->inputLifetime->sourceFrames->t->waitOn(stream) != cudaSuccess ||
        candidate->inputLifetime->sourceFrames->b->waitOn(stream) != cudaSuccess ||
        candidate->inputLifetime->sourceFrames->n->waitOn(stream) != cudaSuccess) {
        candidate->unsafeInput = true;
        return Fail(diagnostics, "CUDA Grow source frame wait failed");
    }
    candidate->grow = std::make_unique<gpu::CudaCurveGrow>();
    // An empty topology has no active frame lookup. Keep the original source
    // bundle alive through the candidate, but do not expose a nonempty frame
    // ABI to the native empty-path validator.
    auto const frameT = inputGeometry.curveCount
        ? gpu::DeviceView<const float3>{candidate->inputLifetime->sourceFrames->t->data(),
                                        candidate->inputLifetime->sourceFrames->t->size()}
        : gpu::DeviceView<const float3>{};
    auto const frameB = inputGeometry.curveCount
        ? gpu::DeviceView<const float3>{candidate->inputLifetime->sourceFrames->b->data(),
                                        candidate->inputLifetime->sourceFrames->b->size()}
        : gpu::DeviceView<const float3>{};
    auto const frameN = inputGeometry.curveCount
        ? gpu::DeviceView<const float3>{candidate->inputLifetime->sourceFrames->n->data(),
                                        candidate->inputLifetime->sourceFrames->n->size()}
        : gpu::DeviceView<const float3>{};
    if (!inputGeometry.curveCount) sourceFrameIds = {};
    gpu::CurveGrowInput input{inputGeometry, inputRootPrim, inputRootUV,
        frameT, frameB, frameN,
        sourceFrameIds};
    if (candidate->grow->BeginFresh(input, candidate->inputLifetime, controls, stream,
            memoryReservation) != gpu::CurveGrowStatus::Ok)
        return Fail(diagnostics, "CUDA Grow value launch failed");
    return true;
}

bool ExecutionState::BeginGrowNamedValue(gpu::DeviceCurveGeometryView inputGeometry,
    OperatorGrowCandidate* candidate, cudaStream_t stream, std::shared_ptr<const void> lifetime,
    std::shared_ptr<NamedOwners> inputNames, UsdGenDiagnostics* diagnostics) {
    if (!candidate || !candidate->grow)
        return Fail(diagnostics, "CUDA Grow value candidate is unavailable");
    auto const* planes = inputNames ? &inputNames->planes : &namedChannels;
    if (planes->empty()) return true;
    std::vector<gpu::CudaNamedChannelTopologyCandidate::RawInputPlane> inputs;
    inputs.reserve(planes->size());
    for (auto const& plane : *planes) {
        if (!plane.bytes || plane.bytes->waitOn(stream) != cudaSuccess)
            return Fail(diagnostics, "CUDA Grow value named input wait failed");
        inputs.push_back({plane.metadata, {plane.bytes->data(), plane.bytes->size()},
                          plane.bytes.get()});
    }
    candidate->namedTopology = std::make_unique<gpu::CudaNamedChannelTopologyCandidate>();
    std::string reason;
    if (candidate->namedTopology->BeginRaw(inputGeometry, candidate->grow->view(),
            std::move(inputs), stream, gpu::CudaNamedChannelTopologyMode::Resample,
            std::move(lifetime), &reason, memoryReservation) != cudaSuccess) {
        if (!candidate->namedTopology->HasUnprovenWork()) candidate->namedTopology.reset();
        return Fail(diagnostics, "CUDA Grow value named topology failed: " + reason);
    }
    return true;
}

bool ExecutionState::AcceptCurveGrowValue(UsdGenCudaExecutionPlan const& plan,
    uint32_t semanticNode, uint32_t inputNode, OperatorGrowCandidate* candidate,
    std::shared_ptr<NamedOwners> inputNames, UsdGenDiagnostics* diagnostics) {
    if (!candidate || !candidate->grow || semanticNode >= geometryValues.size() ||
        inputNode >= geometryValues.size() || topologyOwners.size() != plan.desc.nodes.size() ||
        namedOwners.size() != plan.desc.nodes.size())
        return Fail(diagnostics, "CUDA Grow value output storage is unavailable");
    std::shared_ptr<NamedOwners> outputNames = std::move(inputNames);
    if (candidate->namedTopology) {
        std::vector<gpu::CudaNamedChannelPlane> rebuilt;
        std::string reason;
        if (!candidate->namedTopology->CommitPlanes(cudaSuccess, &rebuilt, &reason))
            return Fail(diagnostics, "CUDA Grow value named completion failed: " + reason);
        outputNames = std::make_shared<NamedOwners>();
        outputNames->planes = std::move(rebuilt);
        candidate->namedTopology.reset();
    }
    auto owners = std::make_shared<TopologyOwners>();
    owners->grow = std::move(candidate->grow);
    topologyOwners[semanticNode] = owners;
    namedOwners[semanticNode] = std::move(outputNames);
    auto const& input = geometryValues[inputNode];
    auto& value = geometryValues[semanticNode];
    value.geometry = owners->grow->view();
    value.hairT = owners->grow->hairT();
    value.rootPrim = owners->grow->rootPrim();
    value.rootUV = owners->grow->rootUV();
    value.curveTopology = {UsdGenDeviceCurveType::Cubic,
                           UsdGenDeviceCurveBasis::BSpline,
                           UsdGenDeviceCurveWrap::Pinned};
    value.deformed = input.deformed;
    value.pointOrigin = value.geometry.points;
    value.widthOrigin = value.geometry.widths;
    value.pointOwnerNode = semanticNode;
    value.widthOwnerNode = semanticNode;
    value.topologyOwnerNode = semanticNode;
    value.namedOwnerNode = semanticNode;
    value.topologyOwners = std::move(owners);
    value.namedOwners = namedOwners[semanticNode];
    value.sourceFrameDomain = input.sourceFrameDomain;
    value.sourceFramesPresent = input.sourceFramesPresent;
    value.ready = true;
    if (semanticNode < widthValues.size()) widthValues[semanticNode].view = value.geometry.widths;
    return true;
}

void ExecutionState::RecordSourceValue(UsdGenCudaExecutionPlan const& plan) {
    if (!plan.taskDag || !prepared || sourceValueRecorded ||
        (plan.scatterGrow ? plan.scatterGrowOutputNode : plan.sourceNode) >=
            plan.desc.nodes.size()) return;
    if (widthValues.size() != plan.desc.nodes.size())
        widthValues.resize(plan.desc.nodes.size());
    if (geometryValues.size() != plan.desc.nodes.size())
        geometryValues.resize(plan.desc.nodes.size());
    uint32_t const sourceNode = plan.scatterGrow ? plan.scatterGrowOutputNode : plan.sourceNode;
    if (plan.geometryValueDag && topologyOwners.size() != plan.desc.nodes.size())
        topologyOwners.resize(plan.desc.nodes.size());
    if (plan.geometryValueDag && !topologyOwners[sourceNode]) {
        auto owners = std::make_shared<TopologyOwners>();
        owners->source = std::move(source);
        owners->resampled = std::move(resampled);
        owners->scatterGrow = std::move(scatterGrow);
        owners->t = std::move(noiseTangent);
        owners->b = std::move(noiseBinormal);
        owners->n = std::move(noiseNormal);
        owners->frameStableIds = std::move(frameStableIds);
        owners->frameCaptureOrdinals = std::move(frameCaptureOrdinals);
        owners->frameHostPacket = std::move(scatterFrameHostPacket);
        topologyOwners[sourceNode] = std::move(owners);
        sourceTopologyOwners = topologyOwners[sourceNode];
    }
    if (plan.geometryValueDag) {
        if (namedOwners.size() != plan.desc.nodes.size()) namedOwners.resize(plan.desc.nodes.size());
        if (!namedOwners[sourceNode]) {
            auto owners = std::make_shared<NamedOwners>();
            owners->planes = std::move(namedChannels);
            namedOwners[sourceNode] = std::move(owners);
        }
    }
    widthValues[sourceNode].view =
        geometry.widths;
    auto& sourceValue = geometryValues[sourceNode];
    sourceValue.geometry = geometry;
    sourceValue.hairT = hairT;
    sourceValue.rootPrim = rootPrim;
    sourceValue.rootUV = rootUV;
    sourceValue.curveTopology = curveTopology;
    sourceValue.deformed = deformed;
    sourceValue.pointOrigin = geometry.points;
    sourceValue.widthOrigin = geometry.widths;
    sourceValue.pointOwnerNode = sourceNode;
    sourceValue.widthOwnerNode = sourceValue.pointOwnerNode;
    sourceValue.topologyOwnerNode = sourceValue.pointOwnerNode;
    sourceValue.namedOwnerNode = sourceValue.pointOwnerNode;
    sourceValue.topologyOwners = plan.geometryValueDag ? topologyOwners[sourceNode] : nullptr;
    sourceValue.namedOwners = plan.geometryValueDag ? namedOwners[sourceNode] : nullptr;
    sourceValue.sourceFrameDomain = plan.geometryValueDag ? topologyOwners[sourceNode] : nullptr;
    sourceValue.sourceFramesPresent = sourceOptions.hasRootFrame || plan.scatterGrow;
    sourceValue.ready = true;
    sourceGeometry = geometry;
    sourceHairT = hairT; sourceRootPrim = rootPrim; sourceRootUV = rootUV;
    sourceCurveTopology = curveTopology; sourceDeformed = deformed;
    sourceValueRecorded = true;
}
void ExecutionState::PreserveSourceNamedChannels(UsdGenCudaExecutionPlan const& plan) {
    if (plan.taskDag && !plan.scatterGrow && plan.terminalNode == plan.sourceNode &&
        !sourceNamedPreserved) {
        sourceTerminalNamedChannels = std::move(namedChannels);
        sourceNamedPreserved = true;
    }
}
// Geometry-value DAGs are admitted only for the C3 CurveSource/
// ReferenceSource or fused ScatterGrow -> Grow/Length/Noise/Width family. Those producers do not
// transport CurveBuffer::chunks (scheduler chunks
// are unrelated), so their persistent values are canonically and provably
// empty.  ScatterGrow enters only through its explicit fused owner path, not
// as a generic empty fallback.
static bool BuildFullNonWidthInput(ExecutionState::GeometryValue const& value,
    gpu::CurveFullNonWidthInput* out) {
    if (!out || !value.ready || !value.topologyOwners) return false;
    gpu::CurveFullNonWidthInput input;
    input.geometry = value.geometry;
    input.geometry.points = value.pointOrigin;
    input.geometry.widths = {}; // Width is intentionally excluded.
    input.hairT = value.hairT;
    input.rootPrim = value.rootPrim;
    input.rootUV = value.rootUV;
    // Every admitted geometry-value snapshot has the one immutable original
    // C3 source-frame owner.  Grow/Length preserve those axes by stable ID;
    // the source plane is not ordinal, so neither side passes it as one.
    // Stable-ID equality below is the GPU proof that both selected axes agree.
    // Fused ScatterGrow follows the same immutable frame-domain invariant.
    if (value.namedOwners) {
        input.named.reserve(value.namedOwners->planes.size());
        for (auto const& plane : value.namedOwners->planes) {
            if (!plane.bytes) return false;
            input.named.push_back({plane.metadata,
                {plane.bytes->data(), plane.bytes->size()}, plane.bytes.get()});
        }
    }
    *out = std::move(input);
    return true;
}
bool ExecutionState::SelectOperatorInput(UsdGenCudaExecutionPlan const& plan,
    size_t index, UsdGenDiagnostics* diagnostics) {
    if (!plan.taskDag) return true;
    if (index >= plan.steps.size())
        return Fail(diagnostics, "invalid CUDA Width DAG operator index");
    auto const inputNode = plan.steps[index]->inputNode;
    if (plan.geometryValueDag) return true;
    if (inputNode >= widthValues.size())
        return Fail(diagnostics, "CUDA Width DAG predecessor value is unavailable");
    auto const input = widthValues[inputNode].view;
    if (input.size != geometry.pointCount || (input.size && !input.data))
        return Fail(diagnostics, "CUDA Width DAG predecessor value is unavailable");
    geometry.widths = input;
    return true;
}
bool ExecutionState::GetOperatorInput(UsdGenCudaExecutionPlan const& plan,
    size_t index, cudaStream_t stream, gpu::DeviceCurveGeometryView* result,
    gpu::DeviceView<const float>* rightResult, UsdGenDiagnostics* diagnostics) {
    if (!result) return false;
    if (!plan.taskDag) {
        if (!SelectOperatorInput(plan, index, diagnostics)) return false;
        *result = geometry;
        if (rightResult) *rightResult = {};
        return true;
    }
    if (index >= plan.steps.size())
        return Fail(diagnostics, "invalid CUDA Width DAG operator index");
    auto const inputNode = plan.steps[index]->inputNode;
    if (plan.geometryValueDag) {
        if (inputNode >= geometryValues.size())
            return Fail(diagnostics, "CUDA same-topology DAG predecessor is unavailable");
        auto const& inputValue = geometryValues[inputNode];
        if (!inputValue.ready || inputValue.pointOrigin.size != inputValue.geometry.pointCount ||
            inputValue.widthOrigin.size != inputValue.geometry.pointCount ||
            (inputValue.pointOrigin.size && (!inputValue.pointOrigin.data ||
                                             !inputValue.widthOrigin.data)))
            return Fail(diagnostics, "CUDA same-topology DAG predecessor is unavailable");
        if (inputValue.pointOwnerNode < geometryValues.size() &&
            geometryValues[inputValue.pointOwnerNode].points &&
            geometryValues[inputValue.pointOwnerNode].points->waitOn(stream) != cudaSuccess)
            return Fail(diagnostics, "CUDA same-topology DAG point predecessor wait failed");
        if (inputValue.pointOwnerNode == plan.sourceNode && source &&
            source->waitOn(stream) != gpu::CurveSourceStatus::Ok)
            return Fail(diagnostics, "CUDA same-topology DAG source predecessor wait failed");
        // The point overlay is not sufficient provenance for a topology trunk:
        // root bindings, named planes and rest data are owned by the selected
        // Source/Grow/Length generation and may be consumed on another stream.
        auto waitTopologyOwner = [&](uint32_t owner) {
            if (plan.scatterGrow && owner == plan.scatterGrowOutputNode) {
                auto const bundle = owner < geometryValues.size()
                    ? geometryValues[owner].topologyOwners : nullptr;
                return bundle && bundle->scatterGrow &&
                    bundle->scatterGrow->waitOn(stream) == gpu::ScatterGrowStatus::Ok;
            }
            if (owner == plan.sourceNode) {
                auto const bundle = owner < geometryValues.size()
                    ? geometryValues[owner].topologyOwners : nullptr;
                auto* selectedSource = bundle ? bundle->source.get() : source ? source.get() :
                    (growInputs ? growInputs->source.get() : nullptr);
                auto* selectedResampler = bundle ? bundle->resampled.get() : resampled ? resampled.get() :
                    (growInputs ? growInputs->resampled.get() : nullptr);
                return (!selectedSource || selectedSource->waitOn(stream) != gpu::CurveSourceStatus::Ok ||
                    (selectedResampler && selectedResampler->waitOn(stream) != cudaSuccess)) ? false : true;
            }
            if (owner < plan.desc.nodes.size() &&
                plan.desc.nodes[owner].type == TfToken("UsdGenGrow"))
                return owner < geometryValues.size() && geometryValues[owner].topologyOwners &&
                    geometryValues[owner].topologyOwners->grow &&
                    geometryValues[owner].topologyOwners->grow->waitOn(stream) == gpu::CurveGrowStatus::Ok;
            if (owner < plan.desc.nodes.size() &&
                plan.desc.nodes[owner].type == TfToken("UsdGenLength"))
                return owner < geometryValues.size() && geometryValues[owner].topologyOwners &&
                    geometryValues[owner].topologyOwners->compaction &&
                    geometryValues[owner].topologyOwners->compaction->waitOn(stream) == cudaSuccess;
            return false;
        };
        if (!waitTopologyOwner(inputValue.topologyOwnerNode))
            return Fail(diagnostics, "CUDA geometry-value DAG topology predecessor wait failed");
        if (inputValue.widthOwnerNode < widthValues.size()) {
            auto const& widthOwner = widthValues[inputValue.widthOwnerNode].owner;
            if (widthOwner && widthOwner->waitOn(stream) != cudaSuccess)
                return Fail(diagnostics, "CUDA same-topology DAG width predecessor wait failed");
        }
        *result = inputValue.geometry;
        result->points = inputValue.pointOrigin;
        result->widths = inputValue.widthOrigin;
        if (rightResult) *rightResult = {};
        auto const rightNode = plan.steps[index]->rightInputNode;
        if (rightNode == UINT32_MAX) return true;
        if (rightNode >= geometryValues.size() || !geometryValues[rightNode].ready)
            return Fail(diagnostics, "CUDA WidthBlend right predecessor is unavailable");
        auto const& rightValue = geometryValues[rightNode];
        if (rightValue.widthOrigin.size != inputValue.widthOrigin.size ||
            (rightValue.widthOrigin.size && !rightValue.widthOrigin.data))
            return Fail(diagnostics, "CUDA WidthBlend right predecessor is unavailable");
        if (rightValue.pointOwnerNode < geometryValues.size() &&
            geometryValues[rightValue.pointOwnerNode].points &&
            geometryValues[rightValue.pointOwnerNode].points->waitOn(stream) != cudaSuccess)
            return Fail(diagnostics, "CUDA WidthBlend right point predecessor wait failed");
        if (!waitTopologyOwner(rightValue.topologyOwnerNode))
            return Fail(diagnostics, "CUDA WidthBlend right topology predecessor wait failed");
        if (rightValue.namedOwners) for (auto const& plane : rightValue.namedOwners->planes)
            if (!plane.bytes || plane.bytes->waitOn(stream) != cudaSuccess)
                return Fail(diagnostics, "CUDA WidthBlend right named predecessor wait failed");
        if (rightValue.widthOwnerNode < widthValues.size()) {
            auto const& widthOwner = widthValues[rightValue.widthOwnerNode].owner;
            if (widthOwner && widthOwner->waitOn(stream) != cudaSuccess)
                return Fail(diagnostics, "CUDA WidthBlend right predecessor wait failed");
        }
        if (rightResult) *rightResult = rightValue.widthOrigin;
        return true;
    }
    if (inputNode >= widthValues.size())
        return Fail(diagnostics, "CUDA Width DAG predecessor value is unavailable");
    auto const input = widthValues[inputNode].view;
    if (input.size != geometry.pointCount || (input.size && !input.data))
        return Fail(diagnostics, "CUDA Width DAG predecessor value is unavailable");
    // The task-graph completion handoff is the host happens-before edge.  The
    // device event is the corresponding cross-stream edge for a descendant
    // Width task; it is a wait enqueue, never a host synchronization.
    if (auto const& owner = widthValues[inputNode].owner;
        owner && owner->waitOn(stream) != cudaSuccess)
        return Fail(diagnostics, "CUDA Width DAG predecessor stream wait failed");
    if (!plan.widthDag && compacted &&
        compacted->waitOn(stream) != cudaSuccess)
        return Fail(diagnostics, "CUDA topology DAG predecessor stream wait failed");
    if (curveGrow && curveGrow->waitOn(stream) != gpu::CurveGrowStatus::Ok)
        return Fail(diagnostics, "CUDA Grow predecessor stream wait failed");
    *result = geometry;
    result->widths = input;
    if (rightResult) *rightResult = {};
    auto const rightNode = plan.steps[index]->rightInputNode;
    if (rightNode == UINT32_MAX) return true;
    if (rightNode >= widthValues.size())
        return Fail(diagnostics, "CUDA WidthBlend right predecessor value is unavailable");
    auto const right = widthValues[rightNode].view;
    if (right.size != geometry.pointCount || (right.size && !right.data))
        return Fail(diagnostics, "CUDA WidthBlend right predecessor value is unavailable");
    // Both events are waited on before the blend launch.  State owns both
    // immutable predecessor planes until the publication join settles.
    if (auto const& owner = widthValues[rightNode].owner;
        owner && owner->waitOn(stream) != cudaSuccess)
        return Fail(diagnostics, "CUDA WidthBlend right predecessor stream wait failed");
    if (rightResult) *rightResult = right;
    return true;
}
bool ExecutionState::SelectTerminal(UsdGenCudaExecutionPlan const& plan,
    cudaStream_t stream, UsdGenDiagnostics* diagnostics) {
    if (!plan.taskDag) return true;
    if (!plan.geometryValueDag && !plan.scatterGrow && plan.terminalNode == plan.sourceNode) {
        auto const sourceBundle = plan.sourceNode < geometryValues.size()
            ? geometryValues[plan.sourceNode].topologyOwners : nullptr;
        auto* const selectedSource = source ? source.get() : sourceBundle ? sourceBundle->source.get() :
            growInputs ? growInputs->source.get() : nullptr;
        auto* const selectedResampler = resampled ? resampled.get() : sourceBundle ? sourceBundle->resampled.get() :
            growInputs ? growInputs->resampled.get() : nullptr;
        if (!sourceValueRecorded || !selectedSource ||
            selectedSource->waitOn(stream) != gpu::CurveSourceStatus::Ok ||
            (selectedResampler && selectedResampler->waitOn(stream) != cudaSuccess))
            return Fail(diagnostics, "CUDA source terminal snapshot is unavailable");
        if (!sourceTerminalSelected) {
            // Grow's borrowed input bundle retains source owners through its
            // native completion proof. Reclaim them only at this all-task
            // join, never while a Grow or named-topology callback can read.
            if (!source && sourceBundle) {
                source = std::move(sourceBundle->source);
                resampled = std::move(sourceBundle->resampled);
            } else if (!source && growInputs) {
                source = std::move(growInputs->source);
                resampled = std::move(growInputs->resampled);
            }
            geometry = sourceGeometry; hairT = sourceHairT;
            rootPrim = sourceRootPrim; rootUV = sourceRootUV;
            curveTopology = sourceCurveTopology; deformed = sourceDeformed;
            auto const sourceNames = plan.sourceNode < geometryValues.size()
                ? geometryValues[plan.sourceNode].namedOwners : nullptr;
            if (sourceNames)
                namedChannels = std::move(sourceNames->planes);
            else if (sourceNamedPreserved)
                namedChannels = std::move(sourceTerminalNamedChannels);
            finalWidths.reset(); finalPoints.reset();
            sourceTerminalSelected = true;
        }
        // All descendant tasks have joined, but keep their compacted owner
        // until state teardown so unselected width views never dangle during
        // a retried finalization. Publication transfers only source owners.
        return true;
    }
    if (plan.geometryValueDag) {
        if (plan.terminalNode >= geometryValues.size())
            return Fail(diagnostics, "CUDA same-topology DAG terminal value is unavailable");
        auto& terminal = geometryValues[plan.terminalNode];
        if (!terminal.ready || terminal.pointOwnerNode >= geometryValues.size() ||
            terminal.widthOwnerNode >= widthValues.size())
            return Fail(diagnostics, "CUDA same-topology DAG terminal value is unavailable");
        if (terminal.topologyOwnerNode >= plan.desc.nodes.size() ||
            terminal.namedOwnerNode != terminal.topologyOwnerNode)
            return Fail(diagnostics, "CUDA geometry-value DAG terminal snapshot ownership is unavailable");
        auto const& topologyNode = plan.desc.nodes[terminal.topologyOwnerNode];
        if (topologyNode.type == TfToken("UsdGenCurveSource") ||
            topologyNode.type == TfToken("UsdGenReferenceSource")) {
            if ((!terminal.topologyOwners || !terminal.topologyOwners->source) &&
                !(selectedTopologyOwner == terminal.topologyOwnerNode && source))
                return Fail(diagnostics, "CUDA geometry-value DAG source terminal factory is unavailable");
        } else if (topologyNode.type == TfToken("UsdGenGrow") && plan.scatterGrow) {
            if ((!terminal.topologyOwners || !terminal.topologyOwners->scatterGrow ||
                terminal.topologyOwners->scatterGrow->waitOn(stream) != gpu::ScatterGrowStatus::Ok) &&
                !(selectedTopologyOwner == terminal.topologyOwnerNode && scatterGrow))
                return Fail(diagnostics, "CUDA geometry-value DAG ScatterGrow terminal factory is unavailable");
        } else if (topologyNode.type == TfToken("UsdGenGrow")) {
            if ((!terminal.topologyOwners || !terminal.topologyOwners->grow ||
                terminal.topologyOwners->grow->waitOn(stream) != gpu::CurveGrowStatus::Ok) &&
                !(selectedTopologyOwner == terminal.topologyOwnerNode && curveGrow))
                return Fail(diagnostics, "CUDA geometry-value DAG Grow terminal factory is unavailable");
        } else if (topologyNode.type == TfToken("UsdGenLength")) {
            if ((!terminal.topologyOwners || !terminal.topologyOwners->compaction ||
                terminal.topologyOwners->compaction->waitOn(stream) != cudaSuccess) &&
                !(selectedTopologyOwner == terminal.topologyOwnerNode && compacted))
                return Fail(diagnostics, "CUDA geometry-value DAG Length terminal factory is unavailable");
        } else {
            return Fail(diagnostics, "CUDA geometry-value DAG terminal topology owner is invalid");
        }
        auto& pointOwner = geometryValues[terminal.pointOwnerNode];
        auto& widthOwner = widthValues[terminal.widthOwnerNode];
        if (pointOwner.points && pointOwner.points->waitOn(stream) != cudaSuccess)
            return Fail(diagnostics, "CUDA same-topology DAG terminal point wait failed");
        if (widthOwner.owner && widthOwner.owner->waitOn(stream) != cudaSuccess)
            return Fail(diagnostics, "CUDA same-topology DAG terminal width wait failed");
        if (pointOwner.points) finalPoints = std::move(pointOwner.points);
        if (widthOwner.owner) finalWidths = std::move(widthOwner.owner);
        // Restore the complete immutable snapshot, not the mutable latest
        // state.  Under the one-trunk admission proof the topology/named
        // generation is uniquely identified by topologyOwnerNode; the
        // corresponding factory (source, Grow, or compactor) remains owned
        // by state until PublishFinal transfers it.
        geometry = terminal.geometry;
        geometry.points = finalPoints ? gpu::DeviceView<const float3>{finalPoints->data(), finalPoints->size()} :
            terminal.pointOrigin;
        geometry.widths = finalWidths ? gpu::DeviceView<const float>{finalWidths->data(), finalWidths->size()} :
            terminal.widthOrigin;
        hairT = terminal.hairT;
        rootPrim = terminal.rootPrim;
        rootUV = terminal.rootUV;
        curveTopology = terminal.curveTopology;
        deformed = terminal.deformed;
        terminal.pointOrigin = geometry.points;
        terminal.widthOrigin = geometry.widths;
        // Publication factories retain unique native owners. Transfer only
        // the selected topology at the all-task join; retries see the normal
        // legacy state owner and never move a sibling's source bundle.
        if ((topologyNode.type == TfToken("UsdGenCurveSource") ||
             topologyNode.type == TfToken("UsdGenReferenceSource")) && !source &&
            terminal.topologyOwners) {
            source = std::move(terminal.topologyOwners->source);
            resampled = std::move(terminal.topologyOwners->resampled);
        }
        if (topologyNode.type == TfToken("UsdGenGrow") && plan.scatterGrow && !scatterGrow)
            scatterGrow = std::move(terminal.topologyOwners->scatterGrow);
        else if (topologyNode.type == TfToken("UsdGenGrow") && !plan.scatterGrow && !curveGrow)
            curveGrow = std::move(terminal.topologyOwners->grow);
        if (topologyNode.type == TfToken("UsdGenLength") && !compacted)
            compacted = std::move(terminal.topologyOwners->compaction);
        if (terminal.namedOwners && namedChannels.empty())
            namedChannels = std::move(terminal.namedOwners->planes);
        selectedTopologyOwner = terminal.topologyOwnerNode;
        return true;
    }
    if (plan.terminalNode >= widthValues.size())
        return Fail(diagnostics, "CUDA Width DAG terminal value is unavailable");
    auto& terminal = widthValues[plan.terminalNode];
    if (terminal.view.size != geometry.pointCount ||
        (terminal.view.size && !terminal.view.data))
        return Fail(diagnostics, "CUDA Width DAG terminal value is unavailable");
    if (terminal.owner && terminal.owner->waitOn(stream) != cudaSuccess)
        return Fail(diagnostics, "CUDA Width DAG terminal stream wait failed");
    if (terminal.owner) finalWidths = std::move(terminal.owner);
    if (finalWidths)
        geometry.widths = {finalWidths->data(), finalWidths->size()};
    else
        geometry.widths = terminal.view;
    terminal.view = geometry.widths;
    return true;
}
bool ExecutionState::HasUnprovenAsyncWork() const noexcept {
    if (namedChannelsUnproven ||
        (curveGrow && curveGrow->HasUnprovenWork()) ||
        (growNamedTopology && growNamedTopology->HasUnprovenWork()) ||
        (sourceNamedTopology && sourceNamedTopology->HasUnprovenWork()) ||
        (source && source->HasUnprovenUpload()) ||
        (resampled && resampled->HasUnprovenUpload())) return true;
    // Accepted branch bundles are published only after their native proof;
    // do not scan their vector here because sibling callbacks can publish a
    // distinct slot concurrently. The immutable source bundle is seeded
    // before operator launch and remains the only borrowed async input.
    if (sourceTopologyOwners &&
        ((sourceTopologyOwners->source && sourceTopologyOwners->source->HasUnprovenUpload()) ||
         (sourceTopologyOwners->resampled && sourceTopologyOwners->resampled->HasUnprovenUpload()) ||
         (sourceTopologyOwners->scatterGrow && sourceTopologyOwners->scatterGrow->HasUnprovenWork()))) return true;
    return false;
}
#endif

void setCudaFinalizationRelayCapacityForTesting(size_t value) {
#ifdef USDGEN_ENABLE_CUDA
    FinalizationRelays().capacity.store(std::min(value, kFinalizationRelayCapacity), std::memory_order_release);
#else
    (void)value;
#endif
}
size_t cudaFinalizationRelayOccupiedCountForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    size_t count = 0; for (auto const& slot : FinalizationRelays().slots) if (slot.occupied.load(std::memory_order_acquire)) ++count; return count;
#else
    return 0;
#endif
}
size_t cudaFinalizationRelayQuarantinedCountForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    size_t count = 0; for (auto const& slot : FinalizationRelays().slots) if (slot.quarantined.load(std::memory_order_acquire)) ++count; return count;
#else
    return 0;
#endif
}
void failNextCudaFinalizationRelayTileCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    FinalizationRelays().failInstall[0].store(true, std::memory_order_release);
#endif
}
void failNextCudaFinalizationRelayBoundsCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    FinalizationRelays().failInstall[1].store(true, std::memory_order_release);
#endif
}
void failNextCudaFinalizationRelayMetadataCallbackInstallForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    FinalizationRelays().failInstall[2].store(true, std::memory_order_release);
#endif
}
void failNextCudaFinalizationRelayAllocationForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    FinalizationRelays().failAllocation.store(true, std::memory_order_release);
#endif
}
void failNextCudaFinalizationRelayTopologyAllocationForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    FinalizationRelays().failTopologyAllocation.store(true, std::memory_order_release);
#endif
}
void failNextCudaRbfFieldResolvePreflightForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    gpu::TestFailNextFreshRbfResolvePreflight();
#endif
}
void failNextCudaRbfFieldResolveCommitForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    gpu::TestFailNextFreshRbfResolveCommit();
#endif
}
void failNextCudaRbfSolveAfterInputSubmitForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    // Route through libusdGen so tests affect the executor's statically
    // linked GPU implementation rather than a duplicate test-library copy.
    gpu::TestFailNextFreshRbfSolveAfterInputSubmit();
#endif
}
void failNextCudaRbfSurfaceResolvePreflightForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    gpu::TestFailNextFreshSurfaceResolvePreflight();
#endif
}
void failNextCudaRbfSurfaceResolveCommitForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    gpu::TestFailNextFreshSurfaceResolveCommit();
#endif
}
uint64_t cudaRbfFieldAcceptAttemptCountForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    return gpu::FreshRbfAcceptAttemptCountForTesting();
#else
    return 0;
#endif
}
uint64_t cudaRbfFieldRollbackAttemptCountForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    return gpu::FreshRbfRollbackAttemptCountForTesting();
#else
    return 0;
#endif
}
uint64_t cudaRbfSurfaceAcceptAttemptCountForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    return gpu::FreshSurfaceAcceptAttemptCountForTesting();
#else
    return 0;
#endif
}
uint64_t cudaRbfSurfaceRollbackAttemptCountForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    return gpu::FreshSurfaceRollbackAttemptCountForTesting();
#else
    return 0;
#endif
}
void failNextCudaSynchronousFinalizationForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    s_failNextSynchronousFinalization.store(true, std::memory_order_release);
#endif
}
void failNextCudaFinalizationRelayTileNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    FinalizationRelays().failNative[0].store(true, std::memory_order_release);
#endif
}
void failNextCudaFinalizationRelayBoundsNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    FinalizationRelays().failNative[1].store(true, std::memory_order_release);
#endif
}
void failNextCudaFinalizationRelayMetadataNativeCallbackForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    FinalizationRelays().failNative[2].store(true, std::memory_order_release);
#endif
}
namespace {
void ArmFinalGate(unsigned phase) {
#ifdef USDGEN_ENABLE_CUDA
    std::atomic_store(&s_finalizationGates[phase], std::make_shared<SourceAsyncTestGate>());
#else
    (void)phase;
#endif
}
void WaitFinalGate(unsigned phase) {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_finalizationGates[phase])) {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!gate->entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        if (!gate->entered.load(std::memory_order_acquire)) throw std::runtime_error("CUDA finalization callback gate timed out");
    }
#else
    (void)phase;
#endif
}
void ReleaseFinalGate(unsigned phase) {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_finalizationGates[phase])) gate->released.store(true, std::memory_order_release);
    std::atomic_store(&s_finalizationGates[phase], std::shared_ptr<SourceAsyncTestGate>());
#else
    (void)phase;
#endif
}
}
void armCudaFinalizationAsyncTileCallbackGateForTesting() { ArmFinalGate(0); }
void waitCudaFinalizationAsyncTileCallbackGateForTesting() { WaitFinalGate(0); }
void releaseCudaFinalizationAsyncTileCallbackGateForTesting() { ReleaseFinalGate(0); }
void armCudaFinalizationAsyncBoundsCallbackGateForTesting() { ArmFinalGate(1); }
void waitCudaFinalizationAsyncBoundsCallbackGateForTesting() { WaitFinalGate(1); }
void releaseCudaFinalizationAsyncBoundsCallbackGateForTesting() { ReleaseFinalGate(1); }
void armCudaFinalizationAsyncMetadataCallbackGateForTesting() { ArmFinalGate(2); }
void waitCudaFinalizationAsyncMetadataCallbackGateForTesting() { WaitFinalGate(2); }
void releaseCudaFinalizationAsyncMetadataCallbackGateForTesting() { ReleaseFinalGate(2); }
void armCudaFinalizationAsyncMetadataLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    std::atomic_store(&s_finalizationReturnGate, std::make_shared<FinalizationReturnGate>());
#endif
}
void waitCudaFinalizationAsyncMetadataLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_finalizationReturnGate)) {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        auto ready = [&] { auto* state = gate->terminalState.load(std::memory_order_acquire); return gate->entered.load(std::memory_order_acquire) && gate->terminal.load(std::memory_order_acquire) && state && (state->load(std::memory_order_acquire) & 1u); };
        while (!ready() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        if (!ready()) throw std::runtime_error("CUDA finalization launcher-return gate timed out");
    }
#endif
}
void releaseCudaFinalizationAsyncMetadataLauncherReturnGateForTesting() {
#ifdef USDGEN_ENABLE_CUDA
    if (auto gate = std::atomic_load(&s_finalizationReturnGate)) {
        gate->handoff.fetch_or(FinalizationReturnGate::Released,
                               std::memory_order_acq_rel);
        ReturnFinalizationMetadataWhenReleased(*gate);
    }
    std::atomic_store(&s_finalizationReturnGate, std::shared_ptr<FinalizationReturnGate>());
#endif
}

#ifdef USDGEN_ENABLE_CUDA
bool ExecutionState::RunOperator(UsdGenCudaExecutionPlan const& plan,
    UsdGenCudaExecutionWorkspace& workspace, double frame, size_t i,
    UsdGenDiagnostics* diagnostics) {
    auto& execution = *workspace.impl_;
    if (execution.poisoned.load(std::memory_order_acquire)) {
        Fail(diagnostics, "CUDA workspace is poisoned by an unproven source upload");
        failed = true;
        return false;
    }
    CudaDeviceScope selected(execution.device);
    if (!selected.selected) {
        Fail(diagnostics, "cannot select CUDA workspace device for operator stage");
        failed = true;
        return false;
    }
    auto const stream = execution.stream;
    auto const& desc = plan.desc;
    auto& geometryRef = this->geometry;
    auto& hairTRef = this->hairT;
    auto& rootPrimRef = this->rootPrim;
    auto& rootUVRef = this->rootUV;
    auto& compactedRef = this->compacted;
    auto& finalWidthsRef = this->finalWidths;
    auto& finalPointsRef = this->finalPoints;
    auto& deformedRef = this->deformed;
    if (i >= plan.steps.size()) return false;
    auto const& width = *plan.steps[i];
    gpu::DeviceCurveGeometryView operatorGeometry;
    gpu::DeviceView<const float> rightInput;
    if (!GetOperatorInput(plan, i, stream, &operatorGeometry, &rightInput, diagnostics)) return false;
    geometryRef = operatorGeometry;
    if (plan.geometryValueDag) {
        auto const& inputSnapshot = geometryValues[width.inputNode];
        hairTRef = inputSnapshot.hairT;
        rootPrimRef = inputSnapshot.rootPrim;
        rootUVRef = inputSnapshot.rootUV;
        curveTopology = inputSnapshot.curveTopology;
        deformedRef = inputSnapshot.deformed;
        if (width.type == TfToken("UsdGenDeform") && deformedRef) {
            Fail(diagnostics,
                "unsupported composition: a second rest-to-animated deformation would apply surface motion twice");
            return false;
        }
    }
    if (width.type == TfToken("UsdGenGrow")) {
        if (plan.geometryValueDag) {
            auto candidate = std::make_unique<OperatorGrowCandidate>();
            auto const& inputValue = geometryValues[width.inputNode];
            auto preserveValue = [&] {
                if (candidate->grow) candidate->grow->MarkUnprovenWork();
                if (candidate->namedTopology) candidate->namedTopology->Quarantine();
                QuarantineGeometryBundles();
                execution.poisoned.store(true, std::memory_order_release);
            };
            if (!BeginCurveGrowValue(width.grow, geometryRef, rootPrimRef,
                    rootUVRef, inputValue.topologyOwners, sourceTopologyOwners,
                    captureStableIds, inputValue.namedOwners, {}, candidate.get(), stream,
                    diagnostics)) {
                if (candidate->unsafeInput ||
                    (candidate->grow && candidate->grow->HasUnprovenWork())) preserveValue();
                return false;
            }
            auto completion = std::make_unique<CudaInlineCompletion>();
            if (candidate->grow->FinishFreshAsync(stream, CudaInlineCompletionCallback,
                    completion.get()) != gpu::CurveGrowStatus::Ok ||
                cudaStreamSynchronize(stream) != cudaSuccess ||
                completion->status.load(std::memory_order_acquire) != int(cudaSuccess)) {
                completion.release(); preserveValue();
                return Fail(diagnostics, "CUDA Grow value completion proof failed");
            }
            if (candidate->grow->CommitFreshFinish() != gpu::CurveGrowStatus::Ok) {
                if (candidate->grow->HasUnprovenWork()) preserveValue();
                return Fail(diagnostics, "CUDA Grow value rejected device input");
            }
            if (!BeginGrowNamedValue(geometryRef, candidate.get(), stream, {},
                    inputValue.namedOwners, diagnostics)) {
                if ((candidate->namedTopology && candidate->namedTopology->HasUnprovenWork()) ||
                    cudaStreamSynchronize(stream) != cudaSuccess) preserveValue();
                return false;
            }
            if (candidate->namedTopology) {
                auto namedCompletion = std::make_unique<CudaInlineCompletion>();
                if (candidate->namedTopology->FinishAsync(CudaInlineCompletionCallback,
                        namedCompletion.get()) != cudaSuccess ||
                    cudaStreamSynchronize(stream) != cudaSuccess ||
                    namedCompletion->status.load(std::memory_order_acquire) != int(cudaSuccess)) {
                    namedCompletion.release(); preserveValue();
                    return Fail(diagnostics, "CUDA Grow value named completion proof failed");
                }
            }
            if (!AcceptCurveGrowValue(plan, width.semanticNode, width.inputNode,
                    candidate.get(), inputValue.namedOwners, diagnostics)) {
                if ((candidate->grow && candidate->grow->HasUnprovenWork()) ||
                    (candidate->namedTopology && candidate->namedTopology->HasUnprovenWork()))
                    preserveValue();
                return false;
            }
            return true;
        }
        auto completion = std::make_unique<CudaInlineCompletion>();
        auto preserve = [&] {
            if (curveGrow) curveGrow->MarkUnprovenWork();
            if (growInputs) {
                if (growInputs->source) growInputs->source->MarkUnprovenUpload();
                if (growInputs->resampled) growInputs->resampled->MarkUnprovenUpload();
                if (growInputs->t) growInputs->t->quarantine();
                if (growInputs->b) growInputs->b->quarantine();
                if (growInputs->n) growInputs->n->quarantine();
            }
            for (auto const& plane : namedChannels) if (plane.bytes) plane.bytes->quarantine();
            QuarantineGeometryBundles();
            if (growNamedTopology) growNamedTopology->Quarantine();
            execution.poisoned.store(true, std::memory_order_release);
        };
        if (!BeginCurveGrow(width.grow, stream, diagnostics)) return false;
        auto status = curveGrow->FinishFreshAsync(stream, CudaInlineCompletionCallback, completion.get());
        if (status != gpu::CurveGrowStatus::Ok || cudaStreamSynchronize(stream) != cudaSuccess ||
            completion->status.load(std::memory_order_acquire) != int(cudaSuccess)) {
            completion.release(); preserve();
            return Fail(diagnostics, "CUDA C3 Grow completion proof failed");
        }
        if (curveGrow->CommitFreshFinish() != gpu::CurveGrowStatus::Ok)
            return Fail(diagnostics, "CUDA C3 Grow rejected device input");
        auto const inputNames = plan.geometryValueDag
            ? geometryValues[width.inputNode].namedOwners : nullptr;
        if (!BeginGrowNamed(stream, {}, inputNames, diagnostics)) {
            if (cudaStreamSynchronize(stream) != cudaSuccess) preserve();
            return false;
        }
        if (growNamedTopology) {
            auto namedCompletion = std::make_unique<CudaInlineCompletion>();
            if (growNamedTopology->FinishAsync(CudaInlineCompletionCallback, namedCompletion.get()) != cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess ||
                namedCompletion->status.load(std::memory_order_acquire) != int(cudaSuccess)) {
                namedCompletion.release(); preserve();
                return Fail(diagnostics, "CUDA C3 Grow named completion proof failed");
            }
        }
        return AcceptCurveGrow(plan, width.semanticNode, cudaSuccess, inputNames, diagnostics);
    }
    if (width.type == TfToken("UsdGenWidthBlend")) {
        if (!plan.geometryValueDag && width.rightInputNode >= widthValues.size())
            return Fail(diagnostics, "CUDA WidthBlend right predecessor value is unavailable");
        auto const right = plan.geometryValueDag ? rightInput : widthValues[width.rightInputNode].view;
        if (right.size != geometryRef.pointCount || (right.size && !right.data) ||
            (!plan.geometryValueDag && widthValues[width.rightInputNode].owner &&
             widthValues[width.rightInputNode].owner->waitOn(stream) != cudaSuccess))
            return Fail(diagnostics, "CUDA WidthBlend right predecessor stream wait failed");
        auto preserveDirectValueOwners = [&] {
            QuarantineGeometryBundles();
            for (auto& value : geometryValues) if (value.points) value.points->quarantine();
            for (auto& value : widthValues) if (value.owner) value.owner->quarantine();
            if (source) source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->source) sourceTopologyOwners->source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->resampled) sourceTopologyOwners->resampled->MarkUnprovenUpload();
            if (resampled) resampled->MarkUnprovenUpload();
            if (curveGrow) curveGrow->MarkUnprovenWork();
            if (compactedRef) compactedRef->MarkUnprovenWork();
            if (finalPointsRef) finalPointsRef->quarantine();
            if (finalWidthsRef) finalWidthsRef->quarantine();
            if (growInputs) {
                if (growInputs->source) growInputs->source->MarkUnprovenUpload();
                if (growInputs->resampled) growInputs->resampled->MarkUnprovenUpload();
            }
        };
        if (plan.geometryValueDag) {
            auto const& leftValue = geometryValues[width.inputNode];
            auto const& rightValue = geometryValues[width.rightInputNode];
            if (width.requiresNonWidthProof) {
                gpu::CurveFullNonWidthInput leftInput, rightFullInput;
                // The C3 geometry-value invariant is deliberately explicit:
                // CurveBuffer chunks are persistent empty values here,
                // not scheduler tiles and not a ScatterGrow fallback.
                if (!BuildFullNonWidthInput(leftValue, &leftInput) ||
                    !BuildFullNonWidthInput(rightValue, &rightFullInput) ||
                    !leftValue.sourceFrameDomain || !rightValue.sourceFrameDomain ||
                    leftValue.sourceFramesPresent != rightValue.sourceFramesPresent ||
                    leftValue.sourceFrameDomain != rightValue.sourceFrameDomain ||
                    leftValue.curveTopology.type != rightValue.curveTopology.type ||
                    leftValue.curveTopology.basis != rightValue.curveTopology.basis ||
                    leftValue.curveTopology.wrap != rightValue.curveTopology.wrap) {
                    return Fail(diagnostics, "CUDA WidthBlend non-width predecessor metadata is unavailable");
                }
                struct CompareLifetime {
                    std::shared_ptr<TopologyOwners> leftTopology, rightTopology;
                    std::shared_ptr<NamedOwners> leftNames, rightNames;
                };
                auto life = std::make_shared<CompareLifetime>(CompareLifetime{
                    leftValue.topologyOwners, rightValue.topologyOwners,
                    leftValue.namedOwners, rightValue.namedOwners});
                gpu::CudaCurveFullNonWidthCompare compare;
                bool equal = false;
                auto compareStatus = compare.BeginFresh(leftInput, rightFullInput, life, stream,
                                                        memoryReservation);
                if (compareStatus != cudaSuccess) {
                    if (compare.HasUnprovenWork()) {
                        compare.Quarantine(); preserveDirectValueOwners();
                        execution.poisoned.store(true, std::memory_order_release);
                    }
                    return Fail(diagnostics, "CUDA WidthBlend non-width comparison admission failed");
                }
                compareStatus = compare.EnqueueFreshStatus(stream);
                if (compareStatus != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
                    compare.Quarantine(); preserveDirectValueOwners();
                    execution.poisoned.store(true, std::memory_order_release);
                    return Fail(diagnostics, "CUDA WidthBlend non-width completion proof failed");
                }
                if (compare.CommitFreshFinish(&equal) != cudaSuccess)
                    return Fail(diagnostics, "CUDA WidthBlend non-width comparison rejected device input");
                if (!equal)
                    return Fail(diagnostics, "CUDA WidthBlend predecessors differ outside Width");
            }
        }
        auto next = std::make_unique<gpu::DeviceBuffer<float>>();
        if (next->reset(geometryRef.pointCount, memoryReservation,
                UsdGenExecutionResourceKind::Active) != cudaSuccess)
            return Fail(diagnostics, "CUDA WidthBlend output allocation failed at " +
                desc.nodes[width.semanticNode].path.GetString());
        CudaWidthBlendGraphCache::Lease graphLease;
        bool launched = false;
        if (execution.widthBlendGraph && execution.widthBlendGraph->Acquire(
                MakeWidthBlendGraphKey(plan, width, geometryRef.pointCount), &graphLease)) {
            launched = execution.widthBlendGraph->CopyLaunchCopy(
                geometryRef.widths, right, next->view(), stream);
        } else {
            launched = gpu::LaunchWidthBlend(geometryRef.widths, right,
                CudaWidthBlendWeight(desc, desc.nodes[width.semanticNode]),
                next->view(), stream);
        }
        if (!launched) {
            preserveDirectValueOwners();
            if (graphLease) graphLease.Quarantine();
            next->quarantine();
            if (widthValues[width.inputNode].owner)
                widthValues[width.inputNode].owner->quarantine();
            if (widthValues[width.rightInputNode].owner)
                widthValues[width.rightInputNode].owner->quarantine();
            execution.poisoned.store(true, std::memory_order_release);
            return Fail(diagnostics, "CUDA WidthBlend execution failed at " +
                desc.nodes[width.semanticNode].path.GetString());
        }
        if (next->recordUse(stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            preserveDirectValueOwners();
            if (graphLease) graphLease.Quarantine();
            next->quarantine();
            if (widthValues[width.inputNode].owner)
                widthValues[width.inputNode].owner->quarantine();
            if (widthValues[width.rightInputNode].owner)
                widthValues[width.rightInputNode].owner->quarantine();
            execution.poisoned.store(true, std::memory_order_release);
            return Fail(diagnostics,
                "CUDA WidthBlend completion proof failed at " +
                desc.nodes[width.semanticNode].path.GetString());
        }
        if (plan.taskDag) {
            auto& value = widthValues[width.semanticNode];
            value.owner = std::move(next);
            value.view = {value.owner->data(), value.owner->size()};
            if (plan.geometryValueDag) {
                auto& geometryValue = geometryValues[width.semanticNode];
                auto const& inputValue = geometryValues[width.inputNode];
                geometryValue.geometry = inputValue.geometry;
                geometryValue.hairT = inputValue.hairT;
                geometryValue.rootPrim = inputValue.rootPrim;
                geometryValue.rootUV = inputValue.rootUV;
                geometryValue.curveTopology = inputValue.curveTopology;
                geometryValue.deformed = inputValue.deformed;
                geometryValue.topologyOwnerNode = inputValue.topologyOwnerNode;
                geometryValue.topologyOwners = inputValue.topologyOwners;
                geometryValue.namedOwnerNode = inputValue.namedOwnerNode;
                geometryValue.namedOwners = inputValue.namedOwners;
                geometryValue.sourceFrameDomain = inputValue.sourceFrameDomain;
                geometryValue.sourceFramesPresent = inputValue.sourceFramesPresent;
                geometryValue.pointOrigin = inputValue.pointOrigin;
                geometryValue.widthOrigin = value.view;
                geometryValue.pointOwnerNode = inputValue.pointOwnerNode;
                geometryValue.widthOwnerNode = width.semanticNode;
                geometryValue.ready = true;
            }
            geometryRef.widths = value.view;
        } else {
            finalWidthsRef = std::move(next);
            geometryRef.widths = {finalWidthsRef->data(), finalWidthsRef->size()};
        }
        return true;
    }
    auto& runtime = *execution.steps.at(width.path);
    auto const& op = desc.nodes[width.semanticNode];
    expr::Context context;
    context.frame = frame;
    context.time = frame / desc.timeCodesPerSecond;
    context.seed = op.seed;
    context.descId = expr::DescriptionId(desc.description.GetText());
    // The rest root frame feeding $N/$Nref, $dPdu/$dPduref and $dPdv/$dPdvref,
    // resolved exactly as the Length compaction below resolves it.
    gpu::RestRootFrames expressionFrames = CompactionFrames();
    if (plan.geometryValueDag) {
        auto const& inputValue = geometryValues[width.inputNode];
        if (inputValue.topologyOwners) {
            if (inputValue.topologyOwners->compaction)
                expressionFrames = inputValue.topologyOwners->compaction->frames();
            else if (inputValue.topologyOwners->grow)
                expressionFrames = {inputValue.topologyOwners->grow->rootT(),
                                    inputValue.topologyOwners->grow->rootB(),
                                    inputValue.topologyOwners->grow->rootN(),
                                    geometryRef.stableIds};
            else if (inputValue.topologyOwners->scatterGrow)
                expressionFrames = {inputValue.topologyOwners->scatterGrow->rootT(),
                                    inputValue.topologyOwners->scatterGrow->rootB(),
                                    inputValue.topologyOwners->scatterGrow->rootN(),
                                    geometryRef.stableIds};
        }
    }
    std::vector<std::string> errors;
    if (runtime.parameters.Evaluate(*width.parameters, geometryRef,
            ExpressionChannels(geometryRef, hairTRef, rootUVRef, rootPrimRef,
                               expressionFrames),
            context,
            stream, &errors, memoryReservation,
            UsdGenExecutionResourceKind::Cache) != CudaParameterStatus::Ok) {
        for (auto const& error : errors) Fail(diagnostics, error);
        if (errors.empty()) Fail(diagnostics, "expression evaluation failed at " + op.path.GetString());
        return false;
    }
    UsdGenParamView values; values.desc = &desc; values.node = &op;
    auto field = [&](char const* name, float fallback) {
        if (auto const* expression = runtime.parameters.Find(TfToken(name)))
            return gpu::ScalarField::Device(
                {static_cast<float const*>(expression->data), expression->count}, expression->domain);
        return gpu::ScalarField::Literal(static_cast<float>(values.GetDouble(TfToken(name), fallback)));
    };
    auto boolean = [&](char const* name, bool fallback) {
        if (auto const* expression = runtime.parameters.Find(TfToken(name)))
            return gpu::BoolField::Device(
                {static_cast<uint8_t const*>(expression->data), expression->count}, expression->domain);
        return gpu::BoolField::Literal(values.GetBool(TfToken(name), fallback));
    };
    auto integer = [&](char const* name, int fallback) {
        if (auto const* expression = runtime.parameters.Find(TfToken(name)))
            return gpu::IntField::Device(
                {static_cast<int32_t const*>(expression->data), expression->count},
                expression->domain);
        return gpu::IntField::Literal(values.GetInt(TfToken(name), fallback));
    };
    if (width.type == TfToken("UsdGenNoise")) {
        gpu::DeviceBuffer<float3> *tangent = nullptr, *binormal = nullptr,
                                  *normal = nullptr;
        gpu::DeviceView<const uint64_t> frameIds;
        if (!NoiseFrames(&tangent, &binormal, &normal, &frameIds))
            return Fail(diagnostics, "CUDA Noise rest root frames are unavailable");
        auto preserveNoiseProofLoss = [&] {
            QuarantineGeometryBundles();
            tangent->quarantine(); binormal->quarantine(); normal->quarantine();
            if (growInputs) {
                if (growInputs->source) growInputs->source->MarkUnprovenUpload();
                if (growInputs->resampled) growInputs->resampled->MarkUnprovenUpload();
            } else {
                if (source) source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->source) sourceTopologyOwners->source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->resampled) sourceTopologyOwners->resampled->MarkUnprovenUpload();
                if (resampled) resampled->MarkUnprovenUpload();
            }
            if (curveGrow) curveGrow->MarkUnprovenWork();
            if (compactedRef) compactedRef->MarkUnprovenWork();
            if (finalPointsRef) finalPointsRef->quarantine();
            if (finalWidthsRef) finalWidthsRef->quarantine();
            for (auto& value : geometryValues)
                if (value.points) value.points->quarantine();
            for (auto& value : widthValues)
                if (value.owner) value.owner->quarantine();
            execution.poisoned.store(true, std::memory_order_release);
        };
        if (tangent->waitOn(stream) != cudaSuccess ||
            binormal->waitOn(stream) != cudaSuccess ||
            normal->waitOn(stream) != cudaSuccess) {
            preserveNoiseProofLoss();
            return Fail(diagnostics, "CUDA Noise frame wait failed");
        }
        auto profile = std::make_unique<gpu::DeviceBuffer<float>>();
        auto next = std::make_unique<gpu::DeviceBuffer<float3>>();
        if (profile->reset(width.profile.size(), memoryReservation,
                           UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
            next->reset(geometryRef.pointCount, memoryReservation,
                        UsdGenExecutionResourceKind::Active) != cudaSuccess)
            return Fail(diagnostics, "CUDA Noise output/profile allocation failed");
        cudaError_t upload = cudaMemcpyAsync(profile->data(), width.profile.data(),
            sizeof(width.profile), cudaMemcpyHostToDevice, stream);
        bool const uploaded = upload == cudaSuccess;
        if (upload != cudaSuccess) {
            if (uploaded && cudaStreamSynchronize(stream) != cudaSuccess) {
                profile->quarantine(); next->quarantine();
                preserveNoiseProofLoss();
                execution.poisoned.store(true, std::memory_order_release);
            }
            return Fail(diagnostics, "CUDA Noise profile upload failed");
        }
        gpu::NoiseParameters parameters;
        parameters.magnitude = field("noise:magnitude", .05f);
        parameters.frequency = field("noise:frequency", 3.0f);
        parameters.correlation = field("noise:correlation", .5f);
        parameters.octaves = integer("noise:octaves", 1);
        parameters.lacunarity = field("noise:lacunarity", 2.0f);
        parameters.gain = field("noise:gain", .5f);
        parameters.preserveLength = field("preserveLength", 1.0f);
        parameters.mask = field("mask", 1.0f);
        parameters.enabled = boolean("enabled", values.node->enabled);
        parameters.cumulative = boolean("cumulative", false);
        parameters.seed = integer("noise:seed", values.node->seed);
        parameters.magnitudeProfile = {profile->data(), profile->size()};
        auto noise = std::make_unique<gpu::CudaNoise>();
        gpu::RestRootFrames const frames = geometryRef.curveCount ? gpu::RestRootFrames{
            {tangent->data(), tangent->size()},
            {binormal->data(), binormal->size()},
            {normal->data(), normal->size()},
            frameIds} : gpu::RestRootFrames{};
        if (noise->ApplyFresh(geometryRef, hairTRef, frames,
                parameters, next->view(), stream, memoryReservation) != gpu::StyleStatus::Ok) {
            bool unsafe = noise->HasUnprovenWork();
            if (!unsafe && cudaStreamSynchronize(stream) != cudaSuccess) unsafe = true;
            if (unsafe) {
                profile->quarantine(); next->quarantine();
                preserveNoiseProofLoss();
            }
            if (unsafe) execution.poisoned.store(true, std::memory_order_release);
            return Fail(diagnostics, "CUDA Noise execution rejected");
        }
        if (tangent->recordUse(stream) != cudaSuccess ||
            binormal->recordUse(stream) != cudaSuccess ||
            normal->recordUse(stream) != cudaSuccess) {
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                profile->quarantine(); next->quarantine();
                preserveNoiseProofLoss();
                execution.poisoned.store(true, std::memory_order_release);
            }
            return Fail(diagnostics, "CUDA Noise frame completion proof failed");
        }
        auto completion = std::make_unique<CudaInlineCompletion>();
        if (noise->FinishFreshAsync(stream, CudaInlineCompletionCallback,
                                    completion.get()) != gpu::StyleStatus::Ok) {
            bool unsafe = noise->HasUnprovenWork();
            if (!unsafe) unsafe = cudaStreamSynchronize(stream) != cudaSuccess;
            if (unsafe) {
                completion.release();
                profile->quarantine(); next->quarantine();
                preserveNoiseProofLoss();
                execution.poisoned.store(true, std::memory_order_release);
            }
            return Fail(diagnostics, "CUDA Noise completion submission failed");
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess ||
            completion->status.load(std::memory_order_acquire) != int(cudaSuccess)) {
            completion.release();
            profile->quarantine(); next->quarantine();
            preserveNoiseProofLoss();
            execution.poisoned.store(true, std::memory_order_release);
            return Fail(diagnostics, "CUDA Noise completion proof failed");
        }
        if (noise->CommitFreshFinish() != gpu::StyleStatus::Ok)
            return Fail(diagnostics, "CUDA Noise execution rejected");
        // The stream was just proved complete. A last-use event failure is a
        // publication failure, but ordinary destruction is safe.
        if (next->recordUse(stream) != cudaSuccess)
            return Fail(diagnostics, "CUDA Noise output proof recording failed");
        if (plan.geometryValueDag) {
            auto& value = geometryValues[width.semanticNode];
            auto const& inputValue = geometryValues[width.inputNode];
            value.geometry = inputValue.geometry;
            value.hairT = inputValue.hairT;
            value.rootPrim = inputValue.rootPrim;
            value.rootUV = inputValue.rootUV;
            value.curveTopology = inputValue.curveTopology;
            value.deformed = inputValue.deformed;
            value.topologyOwnerNode = inputValue.topologyOwnerNode;
            value.topologyOwners = inputValue.topologyOwners;
            value.namedOwnerNode = inputValue.namedOwnerNode;
            value.namedOwners = inputValue.namedOwners;
            value.sourceFrameDomain = inputValue.sourceFrameDomain;
            value.sourceFramesPresent = inputValue.sourceFramesPresent;
            value.points = std::move(next);
            value.pointOrigin = {value.points->data(), value.points->size()};
            value.widthOrigin = geometryRef.widths;
            value.pointOwnerNode = width.semanticNode;
            value.widthOwnerNode = geometryValues[width.inputNode].widthOwnerNode;
            value.ready = true;
        } else {
            finalPointsRef = std::move(next);
            geometryRef.points = {finalPointsRef->data(), finalPointsRef->size()};
        }
        return true;
    }
    if (width.type == TfToken("UsdGenLength")) {
        gpu::LengthParameters parameters;
        auto mode = values.GetToken(TfToken("length:mode"), TfToken("scale"));
        parameters.mode = mode == TfToken("set") ? gpu::LengthMode::Set :
            mode == TfToken("cull") ? gpu::LengthMode::Cull : gpu::LengthMode::Scale;
        parameters.method = values.GetToken(TfToken("length:method"), TfToken("scale")) ==
            TfToken("cutExtend") ? gpu::LengthMethod::CutExtend : gpu::LengthMethod::Scale;
        parameters.rebuild = values.GetToken(TfToken("rebuild"), TfToken("keepParam")) ==
            TfToken("reparam") ? gpu::LengthRebuild::Reparam : gpu::LengthRebuild::KeepParam;
        parameters.value = field("length:value", 1);
        parameters.mask = field("mask", 1);
        parameters.minRemainingLength = field("minRemainingLength", 0);
        parameters.cullThreshold = field("cullThreshold", 0);
        parameters.enabled = boolean("enabled", op.enabled);
        parameters.seed = op.seed;
        if (auto expression = runtime.parameters.Find(TfToken("length:random"))) parameters.random = gpu::Vec2Field::Device({static_cast<float2 const*>(expression->data), expression->count}, expression->domain);
        else { auto random = values.GetVtValue(TfToken("length:random"), VtValue(GfVec2f(1,1))).UncheckedGet<GfVec2f>(); parameters.random = gpu::Vec2Field::Literal(make_float2(random[0], random[1])); }
        gpu::DeviceBuffer<float3> changedPoints; gpu::DeviceBuffer<uint8_t> keep;
        // These two buffers are borrowed by Length and then by the
        // compaction pass.  A failed asynchronous upload/operator launch must
        // not let their scope destructors free storage still referenced by
        // the primary stream.
        auto abandonLengthWork = [&](bool potentiallyInFlight) {
            if (!potentiallyInFlight) return;
            // A successful fence proves all uploads/operator work on this
            // primary lane, allowing ordinary destructors to release their
            // permits.  Only an unsuccessful proof requires quarantine.
            if (cudaStreamSynchronize(stream) == cudaSuccess) return;
            QuarantineGeometryBundles();
            changedPoints.quarantine();
            keep.quarantine();
            if (curveGrow) curveGrow->MarkUnprovenWork();
            if (compactedRef) compactedRef->MarkUnprovenWork();
            if (growInputs) {
                if (growInputs->source) growInputs->source->MarkUnprovenUpload();
                if (growInputs->resampled) growInputs->resampled->MarkUnprovenUpload();
                if (growInputs->t) growInputs->t->quarantine();
                if (growInputs->b) growInputs->b->quarantine();
                if (growInputs->n) growInputs->n->quarantine();
            }
            if (source) source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->source) sourceTopologyOwners->source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->resampled) sourceTopologyOwners->resampled->MarkUnprovenUpload();
            if (resampled) resampled->MarkUnprovenUpload();
            if (finalPointsRef) finalPointsRef->quarantine();
            if (finalWidthsRef) finalWidthsRef->quarantine();
            execution.poisoned.store(true, std::memory_order_release);
        };
        if (changedPoints.reset(geometryRef.pointCount, memoryReservation,
                UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
            keep.reset(geometryRef.curveCount, memoryReservation,
                UsdGenExecutionResourceKind::Scratch) != cudaSuccess) {
            abandonLengthWork(false);
            Fail(diagnostics, "Length allocation failed");
            return false;
        }
        gpu::CudaLength kernel;
        auto const applyStatus = kernel.Apply(geometryRef, hairTRef, parameters,
            changedPoints.view(), keep.view(), stream, memoryReservation);
        if (applyStatus != gpu::StyleStatus::Ok) {
            abandonLengthWork(true);
            Fail(diagnostics, "Length execution failed at " + op.path.GetString());
            return false;
        }
        auto const finishStatus = kernel.Finish(stream);
        if (finishStatus != gpu::StyleStatus::Ok) {
            // Non-CudaError statuses are device-semantic results returned
            // only after Finish's stream synchronization.  Their buffers are
            // proven complete and can be released normally.
            if (kernel.HasUnprovenWork()) abandonLengthWork(true);
            Fail(diagnostics, "Length execution failed at " + op.path.GetString());
            return false;
        }
        auto changed = geometryRef; changed.points = {changedPoints.data(), changedPoints.size()};
        auto next = std::make_unique<gpu::CudaCurveCompaction>();
        gpu::RestRootFrames frames = CompactionFrames();
        if (plan.geometryValueDag) {
            auto const& inputValue = geometryValues[width.inputNode];
            if (inputValue.topologyOwners) {
                if (inputValue.topologyOwners->compaction)
                    frames = inputValue.topologyOwners->compaction->frames();
                else if (inputValue.topologyOwners->grow)
                    frames = {inputValue.topologyOwners->grow->rootT(),
                              inputValue.topologyOwners->grow->rootB(),
                              inputValue.topologyOwners->grow->rootN(),
                              geometryRef.stableIds};
                else if (inputValue.topologyOwners->scatterGrow)
                    frames = {inputValue.topologyOwners->scatterGrow->rootT(),
                              inputValue.topologyOwners->scatterGrow->rootB(),
                              inputValue.topologyOwners->scatterGrow->rootN(),
                              geometryRef.stableIds};
            }
        }
        auto const compactApply = next->Apply(changed, hairTRef, rootPrimRef, rootUVRef,
            {keep.data(), keep.size()}, stream, memoryReservation, frames);
        if (compactApply != gpu::CurveCompactionStatus::Ok) {
            if (next->HasUnprovenWork()) abandonLengthWork(true);
            Fail(diagnostics, "Length curve compaction failed at " + op.path.GetString());
            return false;
        }
        auto const compactFinish = next->Finish(stream);
        if (compactFinish != gpu::CurveCompactionStatus::Ok) {
            if (next->HasUnprovenWork()) abandonLengthWork(true);
            Fail(diagnostics, "Length curve compaction failed at " + op.path.GetString());
            return false;
        }
        std::vector<gpu::CudaNamedChannelPlane> transformedNamedChannels;
        auto const* lengthNames = plan.geometryValueDag
            ? (geometryValues[width.inputNode].namedOwners
                ? &geometryValues[width.inputNode].namedOwners->planes : &namedChannels)
            : &namedChannels;
        if (!lengthNames->empty()) {
            // BeginRaw borrows every geometry/channel owner below.  This
            // synchronous path has no shared job lifetime token, so if a
            // later CUDA proof is lost, transfer that responsibility to all
            // possible raw owners before the stack unwinds.  COW generation
            // data remains valid through its owners' quarantine state.
            auto preserveRawOwnersOnProofLoss = [&] {
                QuarantineGeometryBundles();
                if (source) source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->source) sourceTopologyOwners->source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->resampled) sourceTopologyOwners->resampled->MarkUnprovenUpload();
                if (resampled) resampled->MarkUnprovenUpload();
                if (compactedRef) compactedRef->MarkUnprovenWork();
                if (curveGrow) curveGrow->MarkUnprovenWork();
                if (growInputs) {
                    if (growInputs->source) growInputs->source->MarkUnprovenUpload();
                    if (growInputs->resampled) growInputs->resampled->MarkUnprovenUpload();
                    if (growInputs->t) growInputs->t->quarantine();
                    if (growInputs->b) growInputs->b->quarantine();
                    if (growInputs->n) growInputs->n->quarantine();
                }
                if (finalPointsRef) finalPointsRef->quarantine();
                if (finalWidthsRef) finalWidthsRef->quarantine();
                next->MarkUnprovenWork();
                for (auto const& value : widthValues)
                    if (value.owner) value.owner->quarantine();
                for (auto const& plane : *lengthNames)
                    if (plane.bytes) plane.bytes->quarantine();
                namedChannelsUnproven = true;
                execution.poisoned.store(true, std::memory_order_release);
            };
            auto proveRawOrPreserve = [&] {
                if (cudaStreamSynchronize(stream) == cudaSuccess) return true;
                preserveRawOwnersOnProofLoss();
                return false;
            };
            std::vector<gpu::CudaNamedChannelTopologyCandidate::RawInputPlane> inputs;
            inputs.reserve(lengthNames->size());
            for (auto const& plane : *lengthNames) {
                if (!plane.bytes || plane.bytes->waitOn(stream) != cudaSuccess) {
                    proveRawOrPreserve();
                    Fail(diagnostics, "Length named-channel producer wait failed");
                    return false;
                }
                inputs.push_back({plane.metadata,
                    {plane.bytes->data(), plane.bytes->size()}, plane.bytes.get()});
            }
            gpu::CudaNamedChannelTopologyCandidate namedTopology;
            std::string reason;
            if (namedTopology.BeginRaw(geometryRef, next->view(), std::move(inputs),
                    stream, gpu::CudaNamedChannelTopologyMode::Compaction, {},
                    &reason, memoryReservation) != cudaSuccess) {
                namedTopology.Quarantine();
                proveRawOrPreserve();
                Fail(diagnostics, reason.empty()
                    ? "Length named-channel compaction submission failed" : reason);
                return false;
            }
            auto completion = std::make_unique<CudaInlineCompletion>();
            if (namedTopology.FinishAsync(CudaInlineCompletionCallback,
                                          completion.get()) != cudaSuccess) {
                namedTopology.Quarantine();
                proveRawOrPreserve();
                Fail(diagnostics, "Length named-channel callback installation failed");
                return false;
            }
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                namedTopology.Quarantine();
                completion.release();
                preserveRawOwnersOnProofLoss();
                Fail(diagnostics, "Length named-channel synchronization failed");
                return false;
            }
            if (!namedTopology.CommitPlanes(static_cast<cudaError_t>(
                    completion->status.load(std::memory_order_acquire)),
                    &transformedNamedChannels, &reason)) {
                Fail(diagnostics, reason.empty()
                    ? "Length named-channel compaction failed" : reason);
                return false;
            }
        }
        if (!plan.geometryValueDag) {
            compactedRef = std::move(next); finalPointsRef.reset(); finalWidthsRef.reset();
            geometryRef = compactedRef->view(); hairTRef = compactedRef->hairT();
            rootPrimRef = compactedRef->rootPrim(); rootUVRef = compactedRef->rootUV();
        }
        if (plan.geometryValueDag) {
            if (topologyOwners.size() != plan.desc.nodes.size() ||
                namedOwners.size() != plan.desc.nodes.size())
                return Fail(diagnostics, "CUDA Length output bundle storage is unavailable");
            auto topology = std::make_shared<TopologyOwners>();
            topology->compaction = std::move(next);
            topologyOwners[width.semanticNode] = std::move(topology);
            auto names = std::make_shared<NamedOwners>();
            names->planes = std::move(transformedNamedChannels);
            namedOwners[width.semanticNode] = std::move(names);
        }
        if (plan.taskDag) {
            auto& value = geometryValues[width.semanticNode];
            auto const outputGeometry = plan.geometryValueDag
                ? topologyOwners[width.semanticNode]->compaction->view() : geometryRef;
            auto const outputHairT = plan.geometryValueDag
                ? topologyOwners[width.semanticNode]->compaction->hairT() : hairTRef;
            auto const outputRootPrim = plan.geometryValueDag
                ? topologyOwners[width.semanticNode]->compaction->rootPrim() : rootPrimRef;
            auto const outputRootUV = plan.geometryValueDag
                ? topologyOwners[width.semanticNode]->compaction->rootUV() : rootUVRef;
            value.geometry = outputGeometry; value.hairT = outputHairT; value.rootPrim = outputRootPrim;
            value.rootUV = outputRootUV; value.curveTopology = curveTopology;
            value.deformed = deformedRef; value.pointOrigin = outputGeometry.points;
            value.widthOrigin = outputGeometry.widths; value.pointOwnerNode = width.semanticNode;
            value.widthOwnerNode = width.semanticNode; value.topologyOwnerNode = width.semanticNode;
            value.namedOwnerNode = width.semanticNode;
            value.topologyOwners = plan.geometryValueDag ? topologyOwners[width.semanticNode] : nullptr;
            value.namedOwners = plan.geometryValueDag ? namedOwners[width.semanticNode] : nullptr;
            if (plan.geometryValueDag) {
                auto const& inputValue = geometryValues[width.inputNode];
                value.sourceFrameDomain = inputValue.sourceFrameDomain;
                value.sourceFramesPresent = inputValue.sourceFramesPresent;
            }
            value.ready = true;
        }
        if (!plan.geometryValueDag && !namedChannels.empty()) {
            PreserveSourceNamedChannels(plan);
            namedChannels = std::move(transformedNamedChannels);
        }
        if (plan.taskDag) {
            if (widthValues.size() != plan.desc.nodes.size())
                widthValues.resize(plan.desc.nodes.size());
            widthValues[width.semanticNode].view =
                geometryValues[width.semanticNode].geometry.widths;
        }
        return true;
    }
    if (width.type == TfToken("UsdGenDeform")) {
        int budget = values.GetInt(TfToken("rbfSamples"), 100);
        if (auto const* expression = runtime.parameters.Find(TfToken("rbfSamples"))) if (cudaMemcpyAsync(&budget, expression->data, sizeof(budget), cudaMemcpyDeviceToHost, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) { cudaStreamSynchronize(stream); Fail(diagnostics, "cannot read expression-driven RBF sample count"); return false; }
        if (budget < 4 || budget > 46336) { Fail(diagnostics, "RBF sample count exceeds the supported solver range [4,46336]"); return false; }
        auto nextPoints = std::make_unique<gpu::DeviceBuffer<float3>>();
        if (nextPoints->reset(geometryRef.pointCount, memoryReservation) != cudaSuccess) { Fail(diagnostics, "RBF point allocation failed"); return false; }
        if (geometryRef.pointCount) {
            if (rootPrimRef.size != geometryRef.curveCount || rootUVRef.size != geometryRef.curveCount) { Fail(diagnostics, "RBF deformation requires persistent C3 root bindings"); return false; }
            auto& cache = *runtime.rbf;
            cache.retiredState.reset();
            bool const resetCounters = cache.state && !SameRest(cache.key, width.surface.key);
            bool const rebind = !cache.state || !SameRest(cache.key, width.surface.key) ||
                cache.state->sampleBudget != static_cast<uint32_t>(budget);
            auto freshState = std::unique_ptr<CudaRbfCache::State>();
            CudaRbfCache::State* rbfState = cache.state.get();
            auto current = std::make_unique<gpu::DeviceBuffer<float3>>();
            auto deformer = std::make_unique<gpu::CudaRbfCurveDeformer>();
            auto rest = std::unique_ptr<gpu::DeviceBuffer<float3>>();
            auto offsets = std::unique_ptr<gpu::DeviceBuffer<uint32_t>>();
            auto indices = std::unique_ptr<gpu::DeviceBuffer<uint32_t>>();
            bool surfaceCommitted = false, solveCommitted = false;
            bool freshWorkStarted = false, fieldSolveStarted = false;
            bool freshUploadSubmitted = false;
            auto upload = [&](auto& buffer, auto const& values) {
                if (buffer.reset(values.size(), memoryReservation,
                        UsdGenExecutionResourceKind::Scratch) != cudaSuccess)
                    return false;
                if (values.empty()) return true;
                auto const status = cudaMemcpyAsync(buffer.data(), values.data(),
                    values.size() * sizeof(values[0]), cudaMemcpyHostToDevice,
                    stream);
                if (status == cudaSuccess) freshUploadSubmitted = true;
                return status == cudaSuccess;
            };
            auto unproven = [&] {
                return (rbfState && (rbfState->resources->surface.HasUnprovenFreshWork() ||
                    rbfState->resources->field.HasUnprovenWork())) ||
                    (deformer && deformer->HasUnprovenWork()) ||
                    freshUploadSubmitted;
            };
            auto synchronizeProof = [&] {
                bool const proved = cudaStreamSynchronize(stream) == cudaSuccess;
                // Successful synchronization proves every preceding host to
                // device upload on this lane as well as the staged operation.
                if (proved) freshUploadSubmitted = false;
                return proved;
            };
            auto abandonUnproven = [&] {
                execution.poisoned.store(true, std::memory_order_release);
                // Host proof is missing: retain every owner the queued work
                // can touch.  Workspace teardown then leaks/quarantines these
                // owners rather than destroying them on this thread.
                rest.release();
                offsets.release();
                indices.release();
                current.release();
                deformer.release();
                nextPoints.release();
                freshState.release();
            };
            auto failFresh = [&](char const* message) {
                bool const unsafe = unproven();
                if (!unsafe && surfaceCommitted && solveCommitted) {
                    try {
                        if (i >= pendingRbf.size() || pendingRbf[i]) {
                            abandonUnproven();
                            Fail(diagnostics, "RBF pending transaction slot is unavailable");
                            return false;
                        }
                        auto transaction = std::make_unique<CudaRbfPendingPublication>();
                        transaction->cache = &cache;
                        transaction->state = freshState ? freshState.get() : rbfState;
                        transaction->key = width.surface.key;
                        transaction->rebind = rebind;
                        transaction->resetCounters = resetCounters;
                        transaction->surfaceCommitted = true;
                        transaction->solveCommitted = true;
                        transaction->freshState = std::move(freshState);
                        pendingRbf[i] = std::move(transaction);
                    } catch (...) {
                        // All submitted work is proved.  Avoid early cache
                        // acceptance even if the resolver carrier cannot be
                        // allocated; compensate both independent resources.
                        auto const field = rbfState->resources->field.RollbackFreshSolve();
                        auto const surface = rbfState->resources->surface.RollbackFreshUpdate();
                        if (field != gpu::RbfStatus::Ok || surface != gpu::SurfaceBindingStatus::Ok) {
                            execution.poisoned.store(true, std::memory_order_release);
                            freshState.release();
                        }
                        Fail(diagnostics, message);
                        return false;
                    }
                    if (!ResolvePendingRbf(false)) {
                        execution.poisoned.store(true, std::memory_order_release);
                        AbandonPendingRbf();
                    }
                } else if (!unsafe && surfaceCommitted && !fieldSolveStarted && rbfState &&
                           rbfState->resources->surface.RollbackFreshUpdate() != gpu::SurfaceBindingStatus::Ok) {
                    execution.poisoned.store(true, std::memory_order_release);
                    freshState.release();
                } else if (!unsafe && freshWorkStarted && rbfState) {
                    // The staged APIs have no reversible transaction before
                    // CommitFreshSolve.  A host-proved partial packet is safe
                    // to quarantine, but it must not remain in a reusable
                    // accepted cache.
                    execution.poisoned.store(true, std::memory_order_release);
                    rbfState->resources->field.AbandonFresh();
                    rbfState->resources->surface.AbandonFresh();
                    freshState.release();
                } else if (unsafe) {
                    abandonUnproven();
                }
                Fail(diagnostics, message);
                return false;
            };
            // Construct all host-owned private state before the first stream
            // submission.  An allocation exception here therefore cannot free
            // a buffer still referenced by queued CUDA work.
            if (rebind) {
                freshState = std::make_unique<CudaRbfCache::State>();
                freshState->device = execution.device;
                freshState->sampleBudget = static_cast<uint32_t>(budget);
                rbfState = freshState.get();
                rest = std::make_unique<gpu::DeviceBuffer<float3>>();
                offsets = std::make_unique<gpu::DeviceBuffer<uint32_t>>();
                indices = std::make_unique<gpu::DeviceBuffer<uint32_t>>();
            }
            if (!current || !deformer || current->reset(width.surface.currentPoints.size(), memoryReservation,
                    UsdGenExecutionResourceKind::Scratch) != cudaSuccess)
                return failFresh("RBF allocation failed");
            freshUploadSubmitted = true;
            if (rebind) {
                if (!rest || !offsets || !indices || !upload(*rest, width.surface.restPoints) ||
                    !upload(*offsets, width.surface.faceOffsets) ||
                    !upload(*indices, width.surface.faceVertexIndices) ||
                    (freshWorkStarted = true, rbfState->resources->surface.BeginFreshBind({rest->data(), rest->size()},
                        {offsets->data(), offsets->size()}, offsets->size() - 1,
                        {indices->data(), indices->size()}, static_cast<uint32_t>(budget), stream,
                        memoryReservation) !=
                        gpu::SurfaceBindingStatus::Ok) || !synchronizeProof() ||
                    rbfState->resources->surface.CommitFreshBind() != gpu::SurfaceBindingStatus::Ok ||
                    rbfState->resources->field.BeginFreshBind(rbfState->resources->surface.restSamples(), 0., stream,
                        memoryReservation) != gpu::RbfStatus::Ok ||
                    !synchronizeProof() || rbfState->resources->field.CommitFreshBindExtent() != gpu::RbfStatus::Ok ||
                    rbfState->resources->field.BeginFreshBindRank(stream) != gpu::RbfStatus::Ok ||
                    !synchronizeProof() || rbfState->resources->field.CommitFreshBindRank() != gpu::RbfStatus::Ok ||
                    rbfState->resources->field.BeginFreshBindLu(stream) != gpu::RbfStatus::Ok ||
                    !synchronizeProof() || rbfState->resources->field.CommitFreshBindLu() != gpu::RbfStatus::Ok)
                    return failFresh("RBF fresh rest binding failed; previous generation retained");
            }
            if (!rbfState || !upload(*current, width.surface.currentPoints) ||
                (freshWorkStarted = true, rbfState->resources->surface.BeginFreshUpdate({current->data(), current->size()}, rootPrimRef, rootUVRef, stream,
                    memoryReservation) != gpu::SurfaceBindingStatus::Ok) ||
                !synchronizeProof() || rbfState->resources->surface.CommitFreshUpdate() != gpu::SurfaceBindingStatus::Ok)
                return failFresh("RBF fresh surface update failed; previous generation retained");
            surfaceCommitted = true;
            if ((freshWorkStarted = true, fieldSolveStarted = true,
                    rbfState->resources->field.BeginFreshSolve(rbfState->resources->surface.currentSamples(), stream,
                        memoryReservation) != gpu::RbfStatus::Ok) ||
                !synchronizeProof() || rbfState->resources->field.CommitFreshSolveInput() != gpu::RbfStatus::Ok ||
                rbfState->resources->field.BeginFreshSolveFactors(stream) != gpu::RbfStatus::Ok ||
                !synchronizeProof() || rbfState->resources->field.CommitFreshSolve() != gpu::RbfStatus::Ok)
                return failFresh("RBF fresh solve failed; previous generation retained");
            solveCommitted = true;
            gpu::DeformParameters parameters; parameters.mask = field("mask", 1); parameters.enabled = boolean("enabled", op.enabled); parameters.lockRoots = boolean("lockRoots", true);
            if (deformer->BeginFreshShape(geometryRef, rbfState->resources->surface.rootTargets(), parameters, nextPoints->view(), stream,
                    memoryReservation) != gpu::RbfStatus::Ok ||
                !synchronizeProof() || deformer->CommitFreshShape() != gpu::RbfStatus::Ok ||
                deformer->BeginFreshEvaluate(rbfState->resources->field, stream, memoryReservation) != gpu::RbfStatus::Ok ||
                !synchronizeProof() || deformer->CommitFreshEvaluate(rbfState->resources->field) != gpu::RbfStatus::Ok ||
                deformer->BeginFreshApply(stream) != gpu::RbfStatus::Ok ||
                !synchronizeProof() || deformer->CommitFreshApply() != gpu::RbfStatus::Ok ||
                deformer->BeginFreshCopy(stream) != gpu::RbfStatus::Ok ||
                !synchronizeProof() || deformer->CommitFreshFinish() != gpu::RbfStatus::Ok ||
                nextPoints->recordUse(stream) != cudaSuccess)
                return failFresh("RBF fresh deformation failed; previous generation retained");
            try {
                if (i >= pendingRbf.size() || pendingRbf[i]) {
                    abandonUnproven();
                    Fail(diagnostics, "RBF pending transaction slot is unavailable");
                    return false;
                }
                auto transaction = std::make_unique<CudaRbfPendingPublication>();
                transaction->cache = &cache;
                transaction->state = freshState ? freshState.get() : rbfState;
                transaction->key = width.surface.key;
                transaction->rebind = rebind;
                transaction->resetCounters = resetCounters;
                transaction->surfaceCommitted = true;
                transaction->solveCommitted = true;
                transaction->freshState = std::move(freshState);
                pendingRbf[i] = std::move(transaction);
            } catch (...) {
                return failFresh("RBF publication allocation failed; previous generation retained");
            }
        }
        if (!geometryRef.pointCount && nextPoints->recordUse(stream) != cudaSuccess) {
            Fail(diagnostics, "RBF publication event failed"); return false;
        }
        if (plan.geometryValueDag) {
            auto& value = geometryValues[width.semanticNode];
            auto const& inputValue = geometryValues[width.inputNode];
            value.geometry = inputValue.geometry;
            value.hairT = inputValue.hairT;
            value.rootPrim = inputValue.rootPrim;
            value.rootUV = inputValue.rootUV;
            value.curveTopology = inputValue.curveTopology;
            value.deformed = true;
            value.topologyOwnerNode = inputValue.topologyOwnerNode;
            value.topologyOwners = inputValue.topologyOwners;
            value.namedOwnerNode = inputValue.namedOwnerNode;
            value.namedOwners = inputValue.namedOwners;
            value.sourceFrameDomain = inputValue.sourceFrameDomain;
            value.sourceFramesPresent = inputValue.sourceFramesPresent;
            value.points = std::move(nextPoints);
            value.geometry.points = {value.points->data(), value.points->size()};
            value.pointOrigin = value.geometry.points;
            value.widthOrigin = inputValue.geometry.widths;
            value.pointOwnerNode = width.semanticNode;
            value.widthOwnerNode = inputValue.widthOwnerNode;
            value.ready = true;
        } else {
            finalPointsRef = std::move(nextPoints);
            geometryRef.points = {finalPointsRef->data(), finalPointsRef->size()};
            deformedRef = true;
        }
        return true;
    }
    gpu::WidthParameters parameters; parameters.width = field("width", .01f); parameters.mask = field("mask", 1); parameters.enabled = boolean("enabled", op.enabled); parameters.replace = boolean("replace", true);
    gpu::DeviceBuffer<float> profile;
    auto nextWidths = std::make_unique<gpu::DeviceBuffer<float>>();
    // The profile upload and the Width kernels share the primary stream.
    // Once one upload has succeeded, every failure below must detach all
    // local buffers: their destructors cannot infer that the stream may still
    // reference them from a failed asynchronous operation.
    bool widthWorkSubmitted = false;
    auto abandonWidthWork = [&](bool potentiallyInFlight) {
        if (!potentiallyInFlight) return;
        // A successful primary-lane fence proves every upload and Width
        // command, so normal DeviceBuffer destructors can release permits.
        // Quarantine is reserved for the case where that proof itself fails.
        if (cudaStreamSynchronize(stream) == cudaSuccess) return;
        QuarantineGeometryBundles();
        profile.quarantine();
        for (auto& value : geometryValues) if (value.points) value.points->quarantine();
        for (auto& value : widthValues) if (value.owner) value.owner->quarantine();
        if (source) source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->source) sourceTopologyOwners->source->MarkUnprovenUpload();
            if (sourceTopologyOwners && sourceTopologyOwners->resampled) sourceTopologyOwners->resampled->MarkUnprovenUpload();
        if (resampled) resampled->MarkUnprovenUpload();
        if (curveGrow) curveGrow->MarkUnprovenWork();
        if (compactedRef) compactedRef->MarkUnprovenWork();
        if (finalPointsRef) finalPointsRef->quarantine();
        if (finalWidthsRef) finalWidthsRef->quarantine();
        if (growInputs) {
            if (growInputs->source) growInputs->source->MarkUnprovenUpload();
            if (growInputs->resampled) growInputs->resampled->MarkUnprovenUpload();
        }
        if (nextWidths) nextWidths->quarantine();
        execution.poisoned.store(true, std::memory_order_release);
    };
    if (profile.reset(width.profile.size(), memoryReservation,
            UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        nextWidths->reset(geometryRef.pointCount, memoryReservation,
            UsdGenExecutionResourceKind::Active) != cudaSuccess) {
        abandonWidthWork(false);
        Fail(diagnostics, "Width device allocation failed");
        return false;
    }
    if (cudaMemcpyAsync(profile.data(), width.profile.data(),
            sizeof(width.profile), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        abandonWidthWork(false);
        Fail(diagnostics, "Width profile upload failed");
        return false;
    }
    widthWorkSubmitted = true;
    parameters.widthProfile = {profile.data(), profile.size()}; gpu::CudaWidth kernel;
    auto const applyStatus = kernel.Apply(geometryRef, hairTRef, parameters,
        nextWidths->view(), stream, memoryReservation);
    if (applyStatus != gpu::StyleStatus::Ok) {
        // Even a host-side Apply failure follows a successful profile upload,
        // so the profile remains unproven until the stream is known idle.
        if (widthWorkSubmitted || kernel.HasUnprovenWork())
            abandonWidthWork(true);
        Fail(diagnostics, "Width execution failed at " + op.path.GetString() + "; previous generation retained");
        return false;
    }
    auto const finishStatus = kernel.Finish(stream);
    if (finishStatus != gpu::StyleStatus::Ok) {
        // CudaWidth marks its own storage unproven on a failed completion.
        // The shared profile upload needs a proof only when Width could
        // not establish one; a semantic status is already fully synchronized.
        if (kernel.HasUnprovenWork()) abandonWidthWork(true);
        Fail(diagnostics, "Width execution failed at " + op.path.GetString() + "; previous generation retained");
        return false;
    }
    if (nextWidths->recordUse(stream) != cudaSuccess) {
        // Finish synchronizes the primary stream, so no Width kernel can be
        // using the profile here.  If recordUse fails, ordinary destruction
        // is safe and preserves the normal resource-release behavior.
        Fail(diagnostics, "Width publication event failed at " + op.path.GetString() + "; previous generation retained");
        return false;
    }
    if (plan.taskDag) {
        auto& value = widthValues[width.semanticNode];
        value.owner = std::move(nextWidths);
        value.view = {value.owner->data(), value.owner->size()};
        if (plan.geometryValueDag) {
            auto& geometryValue = geometryValues[width.semanticNode];
            auto const& inputValue = geometryValues[width.inputNode];
            geometryValue.geometry = inputValue.geometry;
            geometryValue.hairT = inputValue.hairT;
            geometryValue.rootPrim = inputValue.rootPrim;
            geometryValue.rootUV = inputValue.rootUV;
            geometryValue.curveTopology = inputValue.curveTopology;
            geometryValue.deformed = inputValue.deformed;
            geometryValue.topologyOwnerNode = inputValue.topologyOwnerNode;
                geometryValue.topologyOwners = inputValue.topologyOwners;
            geometryValue.namedOwnerNode = inputValue.namedOwnerNode;
                geometryValue.namedOwners = inputValue.namedOwners;
                geometryValue.sourceFrameDomain = inputValue.sourceFrameDomain;
                geometryValue.sourceFramesPresent = inputValue.sourceFramesPresent;
            geometryValue.pointOrigin = inputValue.pointOrigin;
            geometryValue.widthOrigin = value.view;
            geometryValue.pointOwnerNode = inputValue.pointOwnerNode;
            geometryValue.widthOwnerNode = width.semanticNode;
            geometryValue.ready = true;
        }
        geometryRef.widths = value.view;
    } else {
        finalWidthsRef = std::move(nextWidths);
        geometryRef.widths = {finalWidthsRef->data(), finalWidthsRef->size()};
    }
    return true;
}
#endif
} // namespace usdGen
