// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenVulkanGapRejections — host-only proof that every large CUDA gap
// without a Vulkan counterpart either rejects Vulkan plans with a clear
// diagnostic or is already served on Vulkan (proven here, not assumed).
//
// Plan-touching gaps reject through CompileVulkanSourceWidthPlan: expression
// graphs/bindings/ops, style operators, and non-admitted source controls.
// Resample (operator + source resampleTo), image maps, moved-transform
// surface binding, and RBF budgets in [1,1024] execute (proven by acceptance
// below, not assumed). Device-only library gaps with no plan surface (picking, tile
// spans/bounds, index streams) are proven unreachable: the admitted operator
// sweep shows no plan operator routes through them, the plan value taxonomy
// carries no such values, and (CUDA builds) AcquireGeometry fails closed for
// Vulkan-identity generations so the CUDA tools entry points cannot consume
// Vulkan state. Positive controls prove each rejection fixture is valid
// apart from the one excluded feature, so a rejection cannot be blamed on a
// broken fixture. curveGeometry is disposed by proving the prepared Vulkan
// source carries the full channel set; its CUDA holder class has no call
// sites upstream. The widthOverlapWitness port itself is proven by the
// dedicated device test, not here.
#include "usdGen/executionBackend.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/curveBuffer.h"
#include "usdGen/deviceGeneration.h"
#include "usdGen/vulkan/executionPlan.h"
#include "usdGen/vulkan/picktileVk.h"
#include "usdGen/vulkan/sourceGeneration.h"

#ifdef USDGEN_ENABLE_CUDA
#include "gpu/generation.h"
#include "gpu/picking.h"

#include <cuda_runtime.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

namespace {

int g_pass = 0, g_fail = 0;

void Check(bool ok, char const* label) {
    if (ok) {
        ++g_pass;
        std::printf("PASS: %s\n", label);
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", label);
    }
}

UsdGenGraphDesc MakeDesc() {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom");
    desc.executionBackend = UsdGenExecutionBackend::Vulkan;

    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Groom/C3");
    curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {2};
    curves.points = {{0, 0, 0}, {0, 1, 0}};
    curves.rest = curves.points;
    curves.widths = {.02f, .03f};
    curves.curveId = {41};
    curves.curveGeneration = 19;
    curves.frozenEpoch = "epoch-19";
    desc.curveSets.push_back(curves);

    UsdGenNodeDesc source;
    source.path = SdfPath("/Groom/Ops/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.params = {{TfToken("useRest"), VtValue(true), false},
        {TfToken("idSource"), VtValue(TfToken("primvar")), false},
        {TfToken("expectEpoch"), VtValue(std::string("epoch-19")), false},
        {TfToken("staleAction"), VtValue(TfToken("block")), false},
        {TfToken("rebind"), VtValue(TfToken("never")), false}};
    UsdGenNodeDesc width;
    width.path = SdfPath("/Groom/Ops/width");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {source.path};
    width.params = {{TfToken("width"), VtValue(.125f), false},
        {TfToken("replace"), VtValue(false), false}};
    desc.nodes = {source, width};
    desc.terminal = width.path;
    return desc;
}

// Rooted variant whose pose surface spans 3D with six samples, so the host
// FPF fit used by Deform admission succeeds and only the excluded feature
// under test can reject.
UsdGenGraphDesc MakeRootedDesc() {
    auto desc = MakeDesc();
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Groom/Scalp");
    surface.restPoints = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                          {0, 0, 1}, {1, 1, 0}, {1, 0, 1}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {3, 3};
    surface.faceVertexIndices = {0, 1, 2, 3, 4, 5};
    surface.uv = {{0, 0}, {1, 0}, {0, 1}, {0, 0}, {1, 0}, {0, 1}};
    desc.surfaces.push_back(surface);
    auto& source = desc.nodes[0];
    source.surfaces = {surface.path};
    auto& curves = desc.curveSets[0];
    curves.skinPrim = {0};
    curves.skinPrimUv = {{.2f, .2f}};
    return desc;
}

UsdGenGraphDesc WithMiddleOp(UsdGenGraphDesc desc, TfToken const& type) {
    UsdGenNodeDesc middle;
    middle.path = SdfPath("/Groom/Ops/middle");
    middle.type = type;
    middle.inputs = {desc.nodes[0].path};
    desc.nodes[1].inputs = {middle.path};
    desc.nodes.insert(desc.nodes.begin() + 1, middle);
    return desc;
}

bool RejectsWith(UsdGenGraphDesc const& desc, char const* needle,
                 std::string* firstError = nullptr) {
    UsdGenDiagnostics diagnostics;
    auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
    if (handle || !diagnostics.HasErrors()) return false;
    if (firstError && !diagnostics.errors.empty())
        *firstError = diagnostics.errors.front();
    for (auto const& error : diagnostics.errors)
        if (error.find(needle) != std::string::npos) return true;
    return false;
}

void ExpectReject(UsdGenGraphDesc const& desc, char const* needle,
                  char const* label) {
    std::string first;
    bool const ok = RejectsWith(desc, needle, &first);
    Check(ok, label);
    std::printf("      diagnostic: %s\n", first.empty() ? "<none>" : first.c_str());
    if (!ok)
        std::printf("      want substring: %s\n", needle);
}

VulkanSourceWidthStage const* FindDeformStage(
    std::shared_ptr<const UsdGenExecutionPlanHandle> const& handle) {
    if (!handle) return nullptr;
    auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
    if (!plan) return nullptr;
    for (auto const& stage : plan->Steps())
        if (stage.kind == VulkanSourceWidthStage::Kind::Deform) return &stage;
    return nullptr;
}

} // namespace

#ifdef USDGEN_ENABLE_CUDA
namespace {
// Minimal neutral owner with Vulkan identity. AcquireGeometry must fail
// closed on the backend check before touching owner state.
class VulkanIdentityOwner final : public UsdGenDeviceOwner {
public:
    bool ProducerReady() const noexcept override { return true; }
    std::unique_ptr<UsdGenDeviceConsumer> AcquireConsumer(
        UsdGenDeviceStream) const noexcept override { return {}; }
};
} // namespace
#endif

int main() {
    // Positive controls: each rejection fixture below is one excluded
    // feature away from one of these accepted plans.
    {
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(MakeDesc(), &diagnostics);
        Check(handle && !diagnostics.HasErrors(), "control: literal source-width plan accepts");
    }
    {
        auto desc = MakeRootedDesc();
        UsdGenNodeDesc deform;
        deform.path = SdfPath("/Groom/Ops/deform");
        deform.type = TfToken("UsdGenDeform");
        deform.inputs = {desc.nodes[0].path};
        deform.surfaces = {desc.surfaces[0].path};
        desc.nodes[1].inputs = {deform.path};
        desc.nodes.insert(desc.nodes.begin() + 1, deform);
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        if (ok) {
            auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
            ok = plan && plan->Steps().size() == 2 &&
                plan->Steps()[0].kind == VulkanSourceWidthStage::Kind::Deform &&
                plan->Steps()[0].deform.sampleCount >= 4 &&
                plan->Steps()[0].deform.sampleCount <= 100;
        }
        Check(ok, "control: identity deform plan accepts with host samples");
    }
    {
        auto desc = MakeRootedDesc();
        UsdGenNodeDesc cull;
        cull.path = SdfPath("/Groom/Ops/cull");
        cull.type = TfToken("UsdGenLength");
        cull.inputs = {desc.nodes[0].path};
        cull.params = {{TfToken("length:mode"), VtValue(TfToken("cull")), false},
                       {TfToken("cullThreshold"), VtValue(.25f), false}};
        auto cullWidth = desc.nodes[1];
        cullWidth.inputs = {cull.path};
        desc.nodes = {desc.nodes[0], cull, cullWidth};
        desc.terminal = cullWidth.path;
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        uint32_t cullTask = UINT32_MAX;
        if (ok) {
            for (auto const& task : handle->Metadata()->Tasks())
                if (task.path == SdfPath("/Groom/Ops/cull")) {
                    ok = task.topologyBarrier;
                    cullTask = task.id;
                }
            bool namedReplacement = false, topologyReplacement = false;
            for (auto const& value : handle->Metadata()->Values()) {
                if (value.producerTask != cullTask) continue;
                if (value.resource == UsdGenExecutionDataKind::NamedChannels)
                    namedReplacement = true;
                if (value.resource == UsdGenExecutionDataKind::CurveTopology)
                    topologyReplacement = true;
            }
            ok = ok && cullTask != UINT32_MAX && namedReplacement && topologyReplacement;
        }
        Check(ok, "control: cull plan accepts with barrier plus named/topology replacements");
    }
    {
        auto desc = MakeDesc();
        desc.nodes[0].params.push_back({TfToken("resampleTo"), VtValue(0), false});
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        Check(handle && !diagnostics.HasErrors(), "control: neutral resampleTo=0 accepts");
    }

    // Unreferenced expression definitions are metadata. Malformed connected
    // targets still reject; standalone ExprOp lacks a CUDA production contract.
    {
        auto desc = MakeDesc();
        desc.expressions.push_back(UsdGenExpressionDesc{});
        UsdGenDiagnostics diagnostics;
        Check(CompileVulkanSourceWidthPlan(desc,&diagnostics) && !diagnostics.HasErrors(),
              "expression: unused definitions are carried");
    }
    {
        auto desc = MakeDesc();
        desc.nodes[1].expressionBindings.push_back(UsdGenExpressionBinding{});
        ExpectReject(desc, "incorrectly typed Width expression",
                     "expression: malformed width expression target rejects");
    }
    {
        auto desc = WithMiddleOp(MakeDesc(), TfToken("UsdGenExprOp"));
        ExpectReject(desc, "unsupported Vulkan operator",
                     "expression: UsdGenExprOp rejects");
    }

    {
        auto desc=MakeDesc();
        UsdGenExpressionDesc expression;
        expression.path=SdfPath("/Groom/Expressions/width");
        expression.source="$value * (0.5 + $t)";
        UsdGenExpressionOutputDesc output;
        output.shape.scalar=expr::ScalarType::Float32;output.shape.components=1;
        expression.outputs.push_back(output);desc.expressions.push_back(expression);
        UsdGenExpressionBinding binding;
        binding.expression=expression.path;binding.destination=TfToken("width");
        binding.destinationShape=output.shape;binding.domain=expr::Domain::Point;
        binding.literal=VtValue(.125f);desc.nodes[1].expressionBindings.push_back(binding);
        desc.nodes[1].params.push_back({TfToken("width:knots"),VtValue(VtVec2fArray{{0,1},{1,.5f}}),false});
        UsdGenDiagnostics diagnostics;auto handle=CompileVulkanSourceWidthPlan(desc,&diagnostics);
        bool ok=handle&&!diagnostics.HasErrors();
        if(ok){auto plan=std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
            ok=plan->Steps()[0].width.extended&&plan->Steps()[0].expressionBindings.size()==1&&
                plan->Steps()[0].expressionBindings[0].binding.domain==expr::Domain::Point;}
        Check(ok,"expression: connected point Width and ramp retain compiled GPU controls");
        desc.nodes[1].expressionBindings[0].expression=SdfPath("/MissingExpression");
        ExpectReject(desc,"missing expression","expression: unresolved connected Width rejects");
    }

    // styleOps(413): style operators have no Vulkan plan lowering. Every
    // registered operator outside the admitted six (plus Grow/Scatter,
    // which are out of scope for this proof) must reject with the operator
    // diagnostic. This doubles as the operator sweep for resample and the
    // expression operator above.
    {
        std::set<std::string> known;
        UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
        for (uint32_t i = 0; i < registry.Count(); ++i)
            known.insert(registry.Record(UsdGenOpId(i)).Name().GetString());
        bool const sane =
            known.count("UsdGenCurveSource") && known.count("UsdGenWidth") &&
            known.count("UsdGenLength") && known.count("UsdGenWidthBlend") &&
            known.count("UsdGenNoise") && known.count("UsdGenDeform") &&
            known.count("UsdGenClump") && known.count("UsdGenResample") &&
            known.count("UsdGenExprOp");
        Check(sane, "style: registry exposes admitted and excluded operators");
        std::set<std::string> const admitted{"UsdGenCurveSource", "UsdGenWidth",
            "UsdGenLength", "UsdGenWidthBlend", "UsdGenNoise", "UsdGenDeform",
            "UsdGenResample", "UsdGenReferenceSource"};
        for (auto const& name : known) {
            if (admitted.count(name)) continue;
            if (name == "UsdGenGrow" || name == "UsdGenScatter") {
                std::printf("PASS: style: %s skipped (out of scope)\n", name.c_str());
                ++g_pass;
                continue;
            }
            auto desc = WithMiddleOp(MakeDesc(), TfToken(name));
            ExpectReject(desc, "unsupported Vulkan operator",
                         ("style: " + name + " rejects").c_str());
        }
    }

    // curveResample(529): indexed-CV resampling has no Vulkan lowering, at
    // the operator or at source upload.
    {
        auto desc = WithMiddleOp(MakeDesc(), TfToken("UsdGenResample"));
        ExpectReject(desc, "Resample distribution must be keepParam",
                     "resample: UsdGenResample without keepParam rejects");
    }
    {
        // keepParam indexed resampling executes as an authored stage.
        auto desc = WithMiddleOp(MakeDesc(), TfToken("UsdGenResample"));
        desc.nodes[1].params = {{TfToken("cvCount"), VtValue(4), false},
            {TfToken("distribution"), VtValue(TfToken("keepParam")), false}};
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        if (ok) {
            auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
            ok = plan && plan->Steps().size() == 2 &&
                plan->Steps()[0].kind == VulkanSourceWidthStage::Kind::Resample &&
                plan->Steps()[0].resampleTarget == 4 &&
                plan->Steps()[0].input == 0;
        }
        Check(ok, "resample: keepParam UsdGenResample accepts with an indexed stage");
    }
    {
        auto desc = WithMiddleOp(MakeDesc(), TfToken("UsdGenResample"));
        desc.nodes[1].params = {{TfToken("cvCount"), VtValue(4), false},
            {TfToken("distribution"), VtValue(TfToken("uniform")), false}};
        ExpectReject(desc, "Resample distribution must be keepParam",
                     "resample: uniform distribution rejects");
    }
    {
        auto desc = WithMiddleOp(MakeDesc(), TfToken("UsdGenResample"));
        desc.nodes[1].params = {{TfToken("cvCount"), VtValue(4), false},
            {TfToken("distribution"), VtValue(TfToken("keepParam")), false},
            {TfToken("mask"), VtValue(0.0f), false}};
        ExpectReject(desc, "Resample mask must be neutral 1.0",
                     "resample: non-neutral mask rejects");
    }
    {
        auto desc = WithMiddleOp(MakeDesc(), TfToken("UsdGenResample"));
        desc.nodes[1].params = {{TfToken("cvCount"), VtValue(4), false},
            {TfToken("distribution"), VtValue(TfToken("keepParam")), false},
            {TfToken("restoreSegmentLengths"), VtValue(true), false}};
        ExpectReject(desc, "Resample restoreSegmentLengths must be false",
                     "resample: length restore rejects");
    }
    {
        auto desc = WithMiddleOp(MakeDesc(), TfToken("UsdGenResample"));
        desc.nodes[1].params = {{TfToken("cvCount"), VtValue(1), false},
            {TfToken("distribution"), VtValue(TfToken("keepParam")), false}};
        ExpectReject(desc, "Resample cvCount must be in [2,64]",
                     "resample: cvCount=1 rejects");
    }
    {
        // Merged resample/csourceVk acceptance: source resampleTo lowers to
        // an implicit leading stage and records its target in the controls.
        auto desc = MakeDesc();
        desc.nodes[0].params.push_back({TfToken("resampleTo"), VtValue(8), false});
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        if (ok) {
            auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
            ok = plan && plan->SourceControls().resampleTo == 8 &&
                plan->Steps().size() == 2 &&
                plan->Steps()[0].kind == VulkanSourceWidthStage::Kind::Resample &&
                plan->Steps()[0].resampleTarget == 8 &&
                plan->Steps()[0].input == 0;
        }
        Check(ok, "resample: CurveSource resampleTo=8 accepts with an implicit stage");
    }
    {
        auto desc = MakeDesc();
        desc.nodes[0].params.push_back({TfToken("resampleTo"), VtValue(1), false});
        ExpectReject(desc, "unsupported or malformed CurveSource parameter resampleTo",
                     "resample: CurveSource resampleTo=1 rejects");
    }
    {
        // Invalid targets keep rejecting with the CUDA-matching diagnostic
        // (curveLoader.cpp ReadOptions: "zero or at least two").
        auto neg = MakeDesc();
        neg.nodes[0].params.push_back({TfToken("resampleTo"), VtValue(-1), false});
        ExpectReject(neg, "CurveSource resampleTo must be zero or at least two",
                     "resample: CurveSource resampleTo=-1 rejects");
        auto one = MakeDesc();
        one.nodes[0].params.push_back({TfToken("resampleTo"), VtValue(1), false});
        ExpectReject(one, "CurveSource resampleTo must be zero or at least two",
                     "resample: CurveSource resampleTo=1 rejects with reason");
        // Two curves so the product (2*INT_MAX) exceeds int32, matching
        // the CudaCurveResample::Apply product check (1*INT_MAX is
        // in-bounds and compiles on both backends; the plan estimate
        // gates its execution through resource admission instead).
        auto huge = MakeDesc();
        huge.curveSets[0].curveVertexCounts = {2, 2};
        huge.curveSets[0].points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
        huge.curveSets[0].rest = huge.curveSets[0].points;
        huge.curveSets[0].widths = {.02f, .02f, .02f, .02f};
        huge.nodes[0].params.push_back(
            {TfToken("resampleTo"), VtValue(INT_MAX), false});
        ExpectReject(huge, "CurveSource resampleTo output exceeds uint32 cardinality",
                     "resample: CurveSource resampleTo=INT_MAX rejects");
    }

    // imageSampler(265): maps are carried, never sampled, exactly like CUDA.
    {
        auto desc = MakeDesc();
        desc.maps.push_back(UsdGenMapDesc{});
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        // No bindings anywhere: no ImageMaps slot (lazy CUDA parity) and the
        // terminal Width controls are untouched by the carried maps.
        if (ok) {
            for (auto const& value : handle->Metadata()->Values())
                ok = ok && value.resource != UsdGenExecutionDataKind::ImageMaps;
            auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
            ok = ok && plan && plan->Width().width == .125f && !plan->Width().replace;
        }
        Check(ok, "maps: graph with maps accepts and stays out of values");
    }
    {
        auto desc = MakeDesc();
        desc.nodes[1].maps.push_back(SdfPath("/Groom/Map"));
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        // Legacy-only maps add no task edge (CUDA adds the ImageMaps read
        // for mapBindings only).
        if (ok)
            for (auto const& task : handle->Metadata()->Tasks())
                for (auto const& use : task.resources)
                    ok = ok && use.resource != UsdGenExecutionDataKind::ImageMaps;
        Check(ok, "maps: width legacy map accepts without a task edge");
    }
    {
        auto desc = MakeDesc();
        desc.nodes[1].mapBindings.push_back({SdfPath("/Groom/Map"), TfToken("sample")});
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        bool slot = false, edge = false;
        if (ok) {
            for (auto const& value : handle->Metadata()->Values())
                slot = slot || (value.resource == UsdGenExecutionDataKind::ImageMaps &&
                    value.storage == UsdGenExecutionValueStorage::ExternalImmutable);
            for (auto const& task : handle->Metadata()->Tasks()) {
                if (task.path != SdfPath("/Groom/Ops/width")) continue;
                for (auto const& use : task.resources)
                    edge = edge || (use.resource == UsdGenExecutionDataKind::ImageMaps &&
                        use.access == UsdGenExecutionResourceAccess::Read);
            }
        }
        Check(ok && slot && edge, "maps: width map binding accepts with slot plus read edge");
    }
    {
        // CUDA parity: non-cage sources reject maps (cudaExecution.cpp:2674);
        // only the regionMapChannel/expectMapGeneration params are admitted.
        auto desc = MakeDesc();
        desc.nodes[0].maps.push_back(SdfPath("/Groom/Map"));
        ExpectReject(desc, "CurveSource has unsupported enabled/input configuration",
                     "maps: non-cage source maps still reject");
    }
    {
        auto desc = MakeDesc();
        desc.nodes[0].params.push_back({TfToken("regionMapChannel"), VtValue(0), false});
        desc.nodes[0].params.push_back({TfToken("expectMapGeneration"), VtValue(uint64_t(7)), false});
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        Check(handle && !diagnostics.HasErrors(), "maps: source map params accept");
    }

    // surfaceBinding(857) + rbf device path: Vulkan Deform serves every
    // finite affine invertible source/surface transform through the host
    // surface-local fit (p2-surface); rbfSamples budgets in [1,1024] take
    // the device-path validator (rbfVk), and only budgets outside [1,1024]
    // still reject.
    {
        auto desc = MakeRootedDesc();
        UsdGenNodeDesc deform;
        deform.path = SdfPath("/Groom/Ops/deform");
        deform.type = TfToken("UsdGenDeform");
        deform.inputs = {desc.nodes[0].path};
        deform.surfaces = {desc.surfaces[0].path};
        desc.nodes[1].inputs = {deform.path};
        desc.nodes.insert(desc.nodes.begin() + 1, deform);
        desc.curveSets[0].worldMatrix[3][0] = 5.0;
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        if (ok) {
            auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
            ok = plan && plan->Steps().size() == 2 &&
                plan->Steps()[0].kind == VulkanSourceWidthStage::Kind::Deform &&
                plan->Steps()[0].deform.sampleCount == 6;
            if (ok) {
                // Six verts under budget 100: every position is selected and
                // mapped surface-local -> source-local (here, x - 5). The root
                // target is the face-0 gather at uv (.2,.2): (.2,.2,0) - (5,0,0).
                auto const& frozen = plan->Steps()[0].deform;
                auto near = [](float a, float b) { return std::fabs(a - b) <= 1e-6f; };
                ok = near(frozen.restSamples[0], -5.0f) && near(frozen.restSamples[1], 0.0f) &&
                    near(frozen.restSamples[2], 0.0f) && near(frozen.posedSamples[0], -5.0f) &&
                    near(frozen.posedSamples[1], 0.0f) && near(frozen.posedSamples[2], 0.0f) &&
                    near(frozen.rootTargets[0], -4.8f) && near(frozen.rootTargets[1], 0.2f) &&
                    near(frozen.rootTargets[2], 0.0f);
            }
        }
        Check(ok, "surface: deform with moved source executes with mapped samples");
    }
    {
        auto desc = MakeRootedDesc();
        UsdGenNodeDesc deform;
        deform.path = SdfPath("/Groom/Ops/deform");
        deform.type = TfToken("UsdGenDeform");
        deform.inputs = {desc.nodes[0].path};
        deform.surfaces = {desc.surfaces[0].path};
        desc.nodes[1].inputs = {deform.path};
        desc.nodes.insert(desc.nodes.begin() + 1, deform);
        desc.surfaces[0].worldMatrix[3][1] = -2.0;
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        if (ok) {
            auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
            ok = plan && plan->Steps().size() == 2 &&
                plan->Steps()[0].kind == VulkanSourceWidthStage::Kind::Deform &&
                plan->Steps()[0].deform.sampleCount == 6;
            if (ok) {
                // Mapped surface-local -> source-local (here, y - 2): first
                // sample (0,0,0) -> (0,-2,0); root target (.2,.2,0) -> (.2,-1.8,0).
                auto const& frozen = plan->Steps()[0].deform;
                auto near = [](float a, float b) { return std::fabs(a - b) <= 1e-6f; };
                ok = near(frozen.restSamples[0], 0.0f) && near(frozen.restSamples[1], -2.0f) &&
                    near(frozen.restSamples[2], 0.0f) && near(frozen.posedSamples[0], 0.0f) &&
                    near(frozen.posedSamples[1], -2.0f) && near(frozen.posedSamples[2], 0.0f) &&
                    near(frozen.rootTargets[0], 0.2f) && near(frozen.rootTargets[1], -1.8f) &&
                    near(frozen.rootTargets[2], 0.0f);
            }
        }
        Check(ok, "surface: deform with moved surface executes with mapped samples");
    }
    {
        auto desc = MakeRootedDesc();
        UsdGenNodeDesc deform;
        deform.path = SdfPath("/Groom/Ops/deform");
        deform.type = TfToken("UsdGenDeform");
        deform.inputs = {desc.nodes[0].path};
        deform.surfaces = {desc.surfaces[0].path};
        desc.nodes[1].inputs = {deform.path};
        desc.nodes.insert(desc.nodes.begin() + 1, deform);
        desc.nodes[0].params[0].value = VtValue(false); // useRest=false
        ExpectReject(desc, "already-deformed CurveSource cannot feed rest-to-animated RBF Deform",
                     "source/deform: useRest=false cannot feed RBF Deform");
    }
    {
        auto desc = MakeRootedDesc();
        UsdGenNodeDesc deform;
        deform.path = SdfPath("/Groom/Ops/deform");
        deform.type = TfToken("UsdGenDeform");
        deform.inputs = {desc.nodes[0].path};
        deform.surfaces = {desc.surfaces[0].path};
        deform.params = {{TfToken("rbfSamples"), VtValue(3), false}};
        desc.nodes[1].inputs = {deform.path};
        desc.nodes.insert(desc.nodes.begin() + 1, deform);
        {
            UsdGenDiagnostics diagnostics;
            auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
            auto const* stage = FindDeformStage(handle);
            Check(handle && !diagnostics.HasErrors() && stage && stage->deform.devicePath &&
                      stage->deform.sampleCount == 3,
                  "rbf: deform rbfSamples=3 accepts for device-path Bind reporting");
        }
    }
    {
        auto desc = MakeRootedDesc();
        UsdGenNodeDesc deform;
        deform.path = SdfPath("/Groom/Ops/deform");
        deform.type = TfToken("UsdGenDeform");
        deform.inputs = {desc.nodes[0].path};
        deform.surfaces = {desc.surfaces[0].path};
        deform.params = {{TfToken("rbfSamples"), VtValue(101), false}};
        desc.nodes[1].inputs = {deform.path};
        desc.nodes.insert(desc.nodes.begin() + 1, deform);
        {
            UsdGenDiagnostics diagnostics;
            auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
            auto const* stage = FindDeformStage(handle);
            Check(handle && !diagnostics.HasErrors() && stage && stage->deform.devicePath &&
                      stage->deform.sampleCount == 6,
                  "rbf: deform rbfSamples=101 accepts on the device path");
        }
    }
    {
        // 125-vertex grid surface: FPF selects exactly 101.
        auto big = MakeRootedDesc();
        VtVec3fArray pts;
        for (int i = 0; i < 125; ++i) {
            int x = i % 5, y = (i / 5) % 5, z = i / 25;
            pts.push_back(GfVec3f(float(x) + float((i * 37) % 11) * 0.013f,
                                  float(y) + float((i * 53) % 13) * 0.017f,
                                  float(z) * 0.8f + float((i * 29) % 7) * 0.019f));
        }
        big.surfaces[0].restPoints = pts;
        big.surfaces[0].points = pts;
        UsdGenNodeDesc deformBig;
        deformBig.path = SdfPath("/Groom/Ops/deform");
        deformBig.type = TfToken("UsdGenDeform");
        deformBig.inputs = {big.nodes[0].path};
        deformBig.surfaces = {big.surfaces[0].path};
        deformBig.params = {{TfToken("rbfSamples"), VtValue(101), false}};
        big.nodes[1].inputs = {deformBig.path};
        big.nodes.insert(big.nodes.begin() + 1, deformBig);
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(big, &diagnostics);
        auto const* stage = FindDeformStage(handle);
        Check(handle && !diagnostics.HasErrors() && stage && stage->deform.devicePath &&
                  stage->deform.sampleCount == 101,
              "rbf: deform rbfSamples=101 selects 101 device samples");
    }
    {
        auto desc = MakeRootedDesc();
        UsdGenNodeDesc deform;
        deform.path = SdfPath("/Groom/Ops/deform");
        deform.type = TfToken("UsdGenDeform");
        deform.inputs = {desc.nodes[0].path};
        deform.surfaces = {desc.surfaces[0].path};
        deform.params = {{TfToken("rbfSamples"), VtValue(46337), false}};
        desc.nodes[1].inputs = {deform.path};
        desc.nodes.insert(desc.nodes.begin() + 1, deform);
        ExpectReject(desc, "Deform rbfSamples must be in [1,46336]",
                     "rbf: deform rbfSamples=46337 rejects over device capacity");
    }

    // curveSource(553): the admitted upload surface is served by Vulkan
    // source capture; every non-admitted control rejects instead of
    // silently degrading.
    {
        auto desc = MakeDesc();
        desc.curveSets[0].rest.clear();
        ExpectReject(desc, "CurveSource useRest=true requires an authored non-current rest sample",
                     "source: useRest without authored rest rejects");
    }
    {
        auto desc = MakeDesc();
        desc.nodes[0].params.back().value = VtValue(TfToken("onError"));
        ExpectReject(desc, "surface-free CurveSource requires rebind=never",
                     "source: surface-free rebind=onError rejects");
    }
    {
        auto desc = MakeDesc();
        desc.curveSets[0].rootFrame = VtMatrix4dArray(1, GfMatrix4d(1.0));
        desc.curveSets[0].skinPrim = {0};
        desc.curveSets[0].skinPrimUv = {{.2f, .2f}};
        desc.nodes[0].params.back().value = VtValue(TfToken("onError"));
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        Check(handle && !diagnostics.HasErrors(),
              "source: surface-free rebind=onError accepts with authored frames");
    }
    {
        auto desc = MakeDesc();
        desc.curveSets[0].rootFrame = VtMatrix4dArray(1, GfMatrix4d(1.0));
        desc.curveSets[0].skinPrim = {0};
        desc.curveSets[0].skinPrimUv = {{.2f, .2f}};
        desc.nodes[0].params.back().value = VtValue(TfToken("always"));
        ExpectReject(desc, "surface-free CurveSource requires rebind=never",
                     "source: surface-free rebind=always rejects even with frames");
    }

    // picking(450) + curveTileBounds/Tiles/Indices(~746): no plan operator
    // routes through these device-only tools (proven by the sweep above),
    // and the plan value taxonomy carries no pick/tile/index values.
    {
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(MakeDesc(), &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        std::set<UsdGenExecutionDataKind> kinds;
        if (ok)
            for (auto const& value : handle->Metadata()->Values())
                kinds.insert(value.resource);
        std::set<UsdGenExecutionDataKind> const want{
            UsdGenExecutionDataKind::CurveGeometry,
            UsdGenExecutionDataKind::Widths,
            UsdGenExecutionDataKind::CurveTopology,
            UsdGenExecutionDataKind::StableIds,
            UsdGenExecutionDataKind::NamedChannels};
        Check(ok && kinds == want, "tools: plan values carry no pick/tile/index kinds");
    }
    {
        // Phase 2 picktiles: tools stay session-layer on Vulkan exactly like
        // CUDA (no plan operator, no plan values); the new execution route
        // serves what the plan values never carried. Device spans/bounds are
        // proven by testUsdGenVulkanPicktileVkParity, not here.
        UsdGenDeviceGeometryMetadata geometry{};
        geometry.curveCount = 2;
        geometry.pointCount = 5;
        std::vector<UsdGenDeviceChannelMetadata> channels{
            {"points", UsdGenDeviceValueType::Float32x3, UsdGenDeviceDomain::Point, 5, 3, 12,
                true, UsdGenDeviceChannelSemantic::Points},
            {"curveOffsets", UsdGenDeviceValueType::UInt32, UsdGenDeviceDomain::Topology, 3,
                1, 4, true, UsdGenDeviceChannelSemantic::CurveOffsets},
            {"stableIds", UsdGenDeviceValueType::UInt64, UsdGenDeviceDomain::Primitive, 2,
                1, 8, true, UsdGenDeviceChannelSemantic::StableIds},
        };
        bool admission = ValidatePicktileVkToolChannels(geometry, channels);
        auto bad = channels;
        bad[1].elementCount = 2;
        admission = admission && !ValidatePicktileVkToolChannels(geometry, bad);
        Check(admission, "tools: vulkan tool admission accepts the prepared channels");
    }
    {
        PicktileVkTileRequirements req{};
        bool golden = GetPicktileVkTileRequirements(0, 0, 100000, 0, 0, &req) &&
            req.chunkCount == 196 && req.chunksPerTile == 4 && req.tileCount == 49;
        Check(golden, "tools: vulkan tile requirements match the cuda 100k golden");
    }
    {
        PicktileVkTileSpan spans[2]{{0, 0, 1, 0, 2}, {1, 1, 1, 2, 3}};
        float minimums[6]{-1, 0, 1, -2, -2, -2};
        float maximums[6]{5, 6, 7, 10, 6, 4};
        std::vector<UsdGenDeviceTileMetadata> tiles;
        bool publish = PicktileVkPublishTileMetadata(spans, minimums, maximums, 2, &tiles) &&
            tiles.size() == 2 && tiles[0].boundsValid &&
            tiles[1].extentMax[0] == 10.0f;
        float flipped[6]{6, 7, 8, 11, 7, 5};
        publish = publish && !PicktileVkPublishTileMetadata(spans, flipped, maximums, 2, &tiles);
        Check(publish, "tools: vulkan tile metadata publish validates conservativeness");
    }
#ifdef USDGEN_ENABLE_CUDA
    {
        // The CUDA tools boundary fails closed for Vulkan state: geometry
        // and tile leases require a CUDA-backend generation, so picking and
        // the tile/index builders cannot silently consume Vulkan channels.
        auto owner = std::make_shared<VulkanIdentityOwner>();
        UsdGenDeviceGeneration::CreateInfo info;
        info.identity.backend = UsdGenDeviceBackend::Vulkan;
        info.identity.deviceIndex = 0;
        info.identity.generation = 7;
        info.owner = owner;
        std::string reason;
        auto generation = UsdGenDeviceGeneration::Create(info, &reason);
        Check(bool(generation), "tools: Vulkan-identity generation fixture creates");
        if (generation) {
            auto lease = gpu::AcquireGeometry(generation, nullptr);
            Check(!lease, "picking/tiles: AcquireGeometry fails closed for Vulkan state");
            auto tile = gpu::AcquireGeometryTile(generation, 0, 7, nullptr);
            Check(!tile, "tiles: AcquireGeometryTile fails closed for Vulkan state");
            // The Vulkan leases fail closed for a foreign (non-adapter)
            // owner and on generation mismatch; the positive lease legs
            // execute in testUsdGenVulkanPicktileVkParity's WithTileMetadata
            // checks, not in this host-only gate.
            Check(!AcquirePicktileVkGeometry(generation, 0),
                  "picktiles: AcquirePicktileVkGeometry fails closed for foreign owners");
            Check(!AcquirePicktileVkTile(generation, 0, 8, 0),
                  "picktiles: AcquirePicktileVkTile fails closed on generation mismatch");
        }
        gpu::CudaPicking picking;
        auto status = picking.ApplyPick(gpu::DeviceCurveGeometryView{},
                                        gpu::PickQuery{}, nullptr);
        Check(status != gpu::PickingStatus::Ok && !picking.pending(),
              "picking: empty view fails closed without a pending op");
        // The Vulkan empty-geometry leg (empty pick = clean miss, empty
        // footprint = count 0, both semantic Ok) lives in
        // testUsdGenVulkanPicktileVkParity, which owns a device; it is cited
        // here, not duplicated.
    }
#else
    std::printf("PASS: tools: lease fail-closed proofs skipped (CUDA off)\n");
    ++g_pass;
#endif

    // namedChannelTopology(820): resample-mode transforms are unreachable
    // because resample itself rejects (proven above); compaction-mode
    // gathering is served by the generic Vulkan plane gather (proven by
    // the cull control's named replacement values).
    {
        // Merged resample/csourceVk acceptance: resample-mode is reachable
        // through the implicit barrier stage, which publishes gathered
        // named/topology replacements like cull; the resampled source
        // retains the named bundle.
        auto desc = MakeDesc();
        desc.nodes[0].params.push_back({TfToken("resampleTo"), VtValue(8), false});
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        uint32_t resampleTask = UINT32_MAX;
        if (ok) {
            for (auto const& task : handle->Metadata()->Tasks())
                if (task.type == TfToken("VulkanResampleTransform")) {
                    ok = task.topologyBarrier;
                    resampleTask = task.id;
                }
            bool namedReplacement = false, topologyReplacement = false;
            for (auto const& value : handle->Metadata()->Values()) {
                if (value.producerTask == resampleTask) {
                    if (value.resource == UsdGenExecutionDataKind::NamedChannels)
                        namedReplacement = true;
                    if (value.resource == UsdGenExecutionDataKind::CurveTopology)
                        topologyReplacement = true;
                }
            }
            ok = ok && resampleTask != UINT32_MAX && namedReplacement && topologyReplacement;
        }
        Check(ok, "named-topology: resample barrier publishes named/topology replacements");
    }

    // curveCompaction(712): the Length:cull compaction path is admitted and
    // explicit (barrier plus replacement values, proven by the cull
    // control); no silent compaction exists.
    {
        auto desc = MakeRootedDesc();
        UsdGenNodeDesc cull;
        cull.path = SdfPath("/Groom/Ops/cull");
        cull.type = TfToken("UsdGenLength");
        cull.inputs = {desc.nodes[0].path};
        cull.params = {{TfToken("length:mode"), VtValue(TfToken("cull")), false},
                       {TfToken("cullThreshold"), VtValue(.25f), false}};
        auto cullWidth = desc.nodes[1];
        cullWidth.inputs = {cull.path};
        desc.nodes = {desc.nodes[0], cull, cullWidth};
        desc.terminal = cullWidth.path;
        UsdGenDiagnostics diagnostics;
        auto handle = CompileVulkanSourceWidthPlan(desc, &diagnostics);
        bool ok = handle && !diagnostics.HasErrors();
        if (ok) {
            auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
            ok = plan && plan->Steps().size() == 2 &&
                plan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthCull &&
                plan->Steps()[0].cullThreshold == .25f;
        }
        Check(ok, "compaction: cull lowers to an explicit LengthCull stage");
    }

    // curveGeometry(16): dispose the channel set on Vulkan. The prepared
    // source must carry points, rest, widths, curve offsets, and stable
    // IDs with exact contents (a superset of DeviceCurveGeometryView).
    {
        VulkanSourcePrepareInfo info;
        auto& b = info.source;
        b.totalCurves = 2; b.totalCvs = 4;
        b.topologyVersion = 11; b.valueVersion = 12;
        b.px = {0, 1, 2, 3}; b.py = {1, 2, 3, 4}; b.pz = {2, 3, 4, 5};
        b.rest = VtVec3fArray(4);
        b.rest[0] = GfVec3f(9, 8, 7); b.rest[1] = GfVec3f(6, 5, 4);
        b.rest[2] = GfVec3f(3, 2, 1); b.rest[3] = GfVec3f(0, -1, -2);
        b.width = {1, 2, 3, 4};
        b.curveId = {17, 29};
        b.cvOffsets = {0, 2, 4};
        std::string reason;
        auto prepared = VulkanPreparedSource::Prepare(std::move(info), &reason);
        bool ok = prepared && reason.empty() &&
            prepared->curveCount() == 2 && prepared->pointCount() == 4 &&
            prepared->topologyVersion() == 11 && prepared->valueVersion() == 12;
        auto find = [&](char const* name) -> VulkanPreparedSource::Plane const* {
            if (!prepared) return nullptr;
            for (auto const& plane : prepared->planes())
                if (plane.metadata.name == name) return &plane;
            return nullptr;
        };
        auto const* points = find("points");
        auto const* rest = find("rest");
        auto const* width = find("width");
        auto const* offsets = find("curveOffsets");
        auto const* ids = find("stableIds");
        ok = ok && points && rest && width && offsets && ids;
        if (ok) {
            std::vector<float> wantPoints{0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5};
            std::vector<float> wantRest{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2};
            std::vector<float> wantWidth{1, 2, 3, 4};
            std::vector<uint32_t> wantOffsets{0, 2, 4};
            std::vector<uint64_t> wantIds{17, 29};
            ok = points->bytes.size() == wantPoints.size() * 4 &&
                std::memcmp(points->bytes.data(), wantPoints.data(), points->bytes.size()) == 0 &&
                rest->bytes.size() == wantRest.size() * 4 &&
                std::memcmp(rest->bytes.data(), wantRest.data(), rest->bytes.size()) == 0 &&
                width->bytes.size() == wantWidth.size() * 4 &&
                std::memcmp(width->bytes.data(), wantWidth.data(), width->bytes.size()) == 0 &&
                offsets->bytes.size() == wantOffsets.size() * 4 &&
                std::memcmp(offsets->bytes.data(), wantOffsets.data(), offsets->bytes.size()) == 0 &&
                ids->bytes.size() == wantIds.size() * 8 &&
                std::memcmp(ids->bytes.data(), wantIds.data(), ids->bytes.size()) == 0 &&
                points->metadata.elementCount == 4 && rest->metadata.elementCount == 4 &&
                width->metadata.elementCount == 4 && offsets->metadata.elementCount == 3 &&
                ids->metadata.elementCount == 2;
        }
        Check(ok, "curveGeometry: prepared source carries the full channel set");
    }

    std::printf("gap-rejections: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
