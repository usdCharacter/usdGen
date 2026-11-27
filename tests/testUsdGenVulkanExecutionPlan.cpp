// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "utils/pxrInterop.h"
#include "usdGen/api/backend.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "usdGen/compile/compiler.h"
#include "usdGenCurves/curveLoader.h"
#include "usdGen/api/ext/op.h"
#include "usdGenCurves/vulkan/executionPlan.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <set>
#include <string>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (false)

namespace {
UsdGenGraphDesc MakeDesc() {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom");
    desc.executionBackend = usdGenTestUtils::Backend("vulkan");

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

UsdGenGraphDesc MakeFanoutDesc() {
    auto desc = MakeDesc();
    auto source = desc.nodes[0];
    auto left = desc.nodes[1]; left.path = SdfPath("/Groom/Ops/widthL");
    left.params[0].value = VtValue(.5f);
    auto right = desc.nodes[1]; right.path = SdfPath("/Groom/Ops/widthR");
    right.params[0].value = VtValue(1.5f);
    auto blend = UsdGenNodeDesc{};
    blend.path = SdfPath("/Groom/Ops/blend"); blend.type = TfToken("UsdGenWidthBlend");
    blend.inputs = {left.path, right.path}; blend.params = {{TfToken("widthBlend:weight"), VtValue(.25f), false}};
    // Deliberately authored out of dependency order: the compiler must topo-sort.
    desc.nodes = {source, blend, right, left}; desc.terminal = blend.path;
    return desc;
}

UsdGenGraphDesc MakeRootedDesc(TfToken const& rebind = TfToken("never")) {
    auto desc = MakeDesc();
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Groom/Scalp");
    surface.restPoints = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {3};
    surface.faceVertexIndices = {0, 1, 2};
    surface.uv = {{0, 0}, {1, 0}, {0, 1}};
    desc.surfaces.push_back(surface);
    auto& source = desc.nodes[0];
    source.surfaces = {surface.path};
    source.params.back().value = VtValue(rebind);
    auto& curves = desc.curveSets[0];
    curves.skinPrim = {0};
    curves.skinPrimUv = {{.25f, .25f}};
    curves.rootFrame = usdGen::UsdGenMatrix4dArray(1, usdGen::UsdGenMatrix4d(1.0));
    return desc;
}

void AddAuthoredPlane(UsdGenCurveSetDesc* curves, char const* name,
                      UsdGenAuthoredPlaneType type,
                      UsdGenAuthoredPlaneDomain domain, uint8_t arity) {
    if (!curves) std::abort();
    UsdGenAuthoredPlaneDesc plane;
    plane.name = TfToken(name); plane.type = type; plane.domain = domain;
    plane.arity = arity;
    size_t const elements = domain == UsdGenAuthoredPlaneDomain::Point
        ? curves->points.size() : domain == UsdGenAuthoredPlaneDomain::Primitive
        ? curves->curveVertexCounts.size() : 1;
    size_t const values = elements * arity;
    if (type == UsdGenAuthoredPlaneType::Float32) {
        plane.floatValues.resize(values);
        for (size_t i = 0; i != values; ++i) plane.floatValues[i] = float(i) + .25f;
    } else {
        plane.intValues.resize(values);
        for (size_t i = 0; i != values; ++i) plane.intValues[i] = int(i * 3 + 1);
    }
    curves->authoredPlanes.push_back(std::move(plane));
}

bool Rejects(UsdGenGraphDesc const& desc) {
    UsdGenDiagnostics diagnostics;
    return !CompileVulkanSourceWidthPlan(usdGenUsd::Lower(desc), &diagnostics) && diagnostics.HasErrors();
}

bool HasDependency(UsdGenExecutionTaskMetadata const& task, uint32_t predecessor,
                   uint8_t provenance) {
    for (auto const& edge : task.dependencyProvenance)
        if (edge.predecessor == predecessor &&
            (edge.provenance & provenance) == provenance) return true;
    return false;
}
}

int main() {
    {
        auto noise = MakeRootedDesc();
        noise.nodes[1].type = TfToken("UsdGenNoise");
        noise.nodes[1].params.clear();
        CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(noise)));

        auto weighted = noise;
        AddAuthoredPlane(&weighted.curveSets[0], "clumpWeight_0",
                         UsdGenAuthoredPlaneType::Float32,
                         UsdGenAuthoredPlaneDomain::Point, 1);
        UsdGenDiagnostics weightedDiagnostics;
        CHECK(!CompileVulkanSourceWidthPlan(usdGenUsd::Lower(weighted), &weightedDiagnostics));
        CHECK(std::any_of(weightedDiagnostics.errors.begin(),
                          weightedDiagnostics.errors.end(),
            [](std::string const &error) {
                return error.find("Noise does not support imported clump motion planes") !=
                    std::string::npos;
            }));

        auto legacyIds = noise;
        AddAuthoredPlane(&legacyIds.curveSets[0], "clumpId_0",
                         UsdGenAuthoredPlaneType::Int32,
                         UsdGenAuthoredPlaneDomain::Primitive, 1);
        CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(legacyIds)));
        for (char const *type : {"UsdGenCurl", "UsdGenBend", "UsdGenWind"}) {
            auto unsupported = weighted;
            unsupported.nodes[1].type = TfToken(type);
            UsdGenDiagnostics unsupportedDiagnostics;
            CHECK(!CompileVulkanSourceWidthPlan(usdGenUsd::Lower(unsupported), &unsupportedDiagnostics) &&
                  unsupportedDiagnostics.HasErrors());
        }
    }
    {
        auto c3 = MakeRootedDesc();
        UsdGenNodeDesc grow;
        grow.path = SdfPath("/Groom/Ops/disabledGrow"); grow.type = TfToken("UsdGenGrow");
        grow.inputs = {c3.nodes.front().path}; grow.enabled = false;
        grow.params = {{TfToken("segments"),VtValue(5),false},
                       {TfToken("length"),VtValue(2.f),false}};
        c3.nodes = {c3.nodes.front(),grow}; c3.terminal = grow.path;
        auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(c3)); CHECK(handle);
        auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
        CHECK(plan->Steps().size() == 1 && !plan->Steps()[0].emptyGenerator &&
              !plan->Steps()[0].disabled);
        CHECK(handle->Metadata()->Tasks()[1].topologyBarrier);
        auto scatterDesc = c3;
        scatterDesc.curveSets.clear();
        auto &scatter = scatterDesc.nodes.front();
        scatter.type = TfToken("UsdGenScatter"); scatter.curves.clear();
        scatter.params = {{TfToken("density"),VtValue(0.f),false}};
        handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(scatterDesc)); CHECK(handle);
        plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
        CHECK(plan->Steps().size() == 1 && plan->Steps()[0].emptyGenerator &&
              !plan->Steps()[0].disabled);
    }
    {
        auto inferred = MakeDesc();
        inferred.terminal = SdfPath();
        std::reverse(inferred.nodes.begin(), inferred.nodes.end());
        auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(inferred));
        CHECK(handle);
        auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
        CHECK(plan->Descriptor()->terminal == usdGenUsd::KeyOf(SdfPath("/Groom/Ops/width")));
        CHECK(plan->WidthNodeKey() == plan->Descriptor()->terminal);
        auto sourceOnly = MakeDesc();
        sourceOnly.nodes.resize(1);
        sourceOnly.terminal = SdfPath();
        CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(sourceOnly)));
        auto ambiguous = MakeDesc();
        auto sibling = ambiguous.nodes.back();
        sibling.path = SdfPath("/Groom/Ops/siblingWidth");
        ambiguous.nodes.push_back(sibling);
        ambiguous.terminal = SdfPath();
        CHECK(Rejects(ambiguous));
        ambiguous.terminal = SdfPath("/Groom/Ops/missing");
        CHECK(Rejects(ambiguous));
        auto cycle = MakeDesc();
        cycle.nodes.front().inputs = {cycle.nodes.back().path};
        cycle.terminal = SdfPath();
        CHECK(Rejects(cycle));
    }
    {
        for (bool authoredFrames : {false, true}) {
            auto singular = authoredFrames ? MakeRootedDesc() : MakeDesc();
            singular.curveSets.front().worldMatrix[0][0] = 0.0;
            auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(singular));
            CHECK(handle);
            auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
            UsdGenParamView params{plan->Descriptor().get(), &plan->Descriptor()->nodes.front()};
            UsdGenCaptureContext capture;
            capture.desc = plan->Descriptor().get(); capture.params = &params;
            UsdGenCurveBuffer decoded;
            CHECK(UsdGenCurveLoader::Load(capture, plan->Source(), &decoded, nullptr));
            CHECK(decoded.totalCurves == 1 && decoded.totalCvs == 2);
            if (authoredFrames) {
                CHECK(decoded.rootT.size() == 1);
                singular.curveSets.front().rootFrame.clear();
                CHECK(Rejects(singular));
            } else {
                CHECK(decoded.rootT.empty() && decoded.rootPrim.empty());
            }
        }
    }
    {
        // CUDA accepts description publication transforms independently of
        // C3 point space, CatmullRom C3s, promoted nonperiodic wrapping and
        // constant widths. Exercise the retained descriptor and the exact
        // shared decoder used by native Vulkan source capture without a GPU.
        auto desc = MakeDesc();
        desc.xformMatrix.SetTranslate(GfVec3d(4, 5, 6));
        desc.curveBasis = TfToken("catmullRom");
        auto &curves = desc.curveSets.front();
        curves.basis = TfToken("catmullRom");
        curves.wrap = TfToken("nonperiodic");
        curves.widthsInterpolation = TfToken("constant");
        curves.widths = {.375f};
        desc.nodes[0].params.push_back({TfToken("interpolationMode"),
                                      VtValue(TfToken("none")), false});
        desc.nodes[0].params.push_back({TfToken("densityMultiplier"), VtValue(2.f), false});
        UsdGenRampDesc identity;
        identity.knots = {{0.f, 1.f}, {1.f, 1.f}};
        desc.nodes[0].ramps.push_back(identity);
        auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(desc));
        CHECK(handle);
        auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
        CHECK(plan->Descriptor()->xformMatrix == desc.xformMatrix);
        CHECK(usdGenUsd::Presentation(*plan->Descriptor()) &&
              usdGenUsd::Presentation(*plan->Descriptor())->basis == desc.curveBasis);
        CHECK(plan->Source().basis == curves.basis && plan->Source().wrap == curves.wrap);
        UsdGenParamView params{plan->Descriptor().get(), &plan->Descriptor()->nodes.front()};
        UsdGenCaptureContext capture;
        capture.desc = plan->Descriptor().get();
        capture.params = &params;
        UsdGenCurveBuffer decoded;
        CHECK(UsdGenCurveLoader::Load(capture, plan->Source(), &decoded, nullptr));
        CHECK(decoded.totalCurves == 1 && decoded.totalCvs == 2);
        CHECK(decoded.width == VtFloatArray({.375f, .375f}));
        CHECK(decoded.px == VtFloatArray({0.f, 0.f}));
        CHECK(decoded.py == VtFloatArray({0.f, 1.f}));
        CHECK(decoded.pz == VtFloatArray({0.f, 0.f}));
        auto malformed = desc;
        malformed.curveSets.front().widths.push_back(.5f);
        CHECK(Rejects(malformed));
        malformed = desc;
        malformed.nodes[0].ramps.front().knots.back()[1] = .5f;
        CHECK(Rejects(malformed));
        malformed = desc;
        malformed.curveSets.front().basis = TfToken("bezier");
        CHECK(Rejects(malformed));
        malformed = desc;
        malformed.curveSets.front().wrap = TfToken("periodic");
        CHECK(Rejects(malformed));
    }
    {
        // A source plus 80 valid Width stages exceeds both former admission
        // caps. Every task still has ordered uint32 dependency/value IDs.
        auto chain = MakeDesc();
        auto width = chain.nodes.back();
        chain.nodes.resize(1);
        for (unsigned i = 0; i != 80; ++i) {
            auto stage = width;
            stage.path = SdfPath("/Groom/Ops/longWidth" + std::to_string(i));
            stage.inputs = {chain.nodes.back().path};
            chain.nodes.push_back(stage);
        }
        chain.terminal = chain.nodes.back().path;
        auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(chain));
        CHECK(handle && handle->Metadata()->Tasks().size() == 82);
        auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
        CHECK(plan->Steps().size() == 80);
        for (size_t i = 0; i < plan->Steps().size(); ++i)
            CHECK(plan->Steps()[i].input == i);
    }
    {
        auto sourceOnly = MakeDesc();
        sourceOnly.nodes.resize(1);
        sourceOnly.terminal = sourceOnly.nodes.front().path;
        auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(sourceOnly));
        CHECK(handle);
        auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
        CHECK(plan->Steps().size() == 1 && plan->Steps()[0].disabled);
        CHECK(handle->Metadata()->Operators().size() == 1);
        sourceOnly.nodes[0].params.push_back({TfToken("resampleTo"), VtValue(5), false});
        handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(sourceOnly));
        CHECK(handle);
        plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
        CHECK(plan->Steps().size() == 1 &&
              plan->Steps()[0].kind == VulkanSourceWidthStage::Kind::Resample);
        auto const& publication = handle->Metadata()->Tasks().back();
        CHECK(publication.resources[0].resource == UsdGenExecutionDataKind::ElementGeometry);
        CHECK(publication.resources[1].resource == UsdGenExecutionDataKind::Widths);
        CHECK(handle->Metadata()->Values()[publication.resources[0].inputValue].resource ==
              UsdGenExecutionDataKind::ElementGeometry);
        CHECK(handle->Metadata()->Values()[publication.resources[1].inputValue].resource ==
              UsdGenExecutionDataKind::Widths);
    }
    // A muted Width compiles to a disabled passthrough stage instead of
    // failing the enabled gate: toggling must alias, never reject.
    {
        auto muted = MakeDesc();
        muted.nodes[1].enabled = false;
        muted.nodes[1].params[0].value = VtValue(9.f);
        UsdGenDiagnostics mutedDiagnostics;
        auto mutedHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(muted), &mutedDiagnostics);
        CHECK(mutedHandle && !mutedDiagnostics.HasErrors());
        auto mutedPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
            mutedHandle->Payload());
        CHECK(mutedPlan && mutedPlan->Steps().size() == 1 &&
              mutedPlan->Steps()[0].disabled);
    }
    {
        // A muted Length compiles to a disabled stage with the same rule;
        // the shared executor skip hook aliases it without dispatch.
        auto mutedLength = MakeDesc();
        UsdGenNodeDesc length;
        length.path = SdfPath("/Groom/Ops/length");
        length.type = TfToken("UsdGenLength");
        length.inputs = {mutedLength.nodes[0].path};
        length.params = {{TfToken("length:value"), VtValue(.5f), false},
                         {TfToken("length:mode"), VtValue(TfToken("scale")), false}};
        length.enabled = false;
        mutedLength.nodes[1].inputs = {length.path};
        mutedLength.nodes.insert(mutedLength.nodes.begin() + 1, length);
        mutedLength.terminal = mutedLength.nodes.back().path;
        UsdGenDiagnostics lengthDiagnostics;
        auto lengthHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(mutedLength), &lengthDiagnostics);
        CHECK(lengthHandle && !lengthDiagnostics.HasErrors());
        auto lengthPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
            lengthHandle->Payload());
        CHECK(lengthPlan && lengthPlan->Steps().size() == 2 &&
              lengthPlan->Steps()[0].disabled &&
              lengthPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthScale &&
              !lengthPlan->Steps()[1].disabled);
        mutedLength.nodes[1].params.push_back({TfToken("cullThreshold"), VtValue(.75f), false});
        lengthHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(mutedLength), &lengthDiagnostics);
        CHECK(lengthHandle && !lengthDiagnostics.HasErrors());
        lengthPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(lengthHandle->Payload());
        CHECK(lengthPlan->Steps().size() == 3 && lengthPlan->Steps()[0].disabled &&
              lengthPlan->Steps()[1].disabled && !lengthPlan->Steps()[2].disabled);
    }
    {
        auto deformDesc = MakeRootedDesc();
        auto& surface = deformDesc.surfaces.front();
        surface.restPoints.push_back({0, 0, 1});
        surface.points = surface.restPoints;
        for (auto& point : surface.points) point[2] += 2;
        auto& source = deformDesc.curveSets.front();
        source.curveVertexCounts = {2, 2};
        source.points = {{0, 0, 0}, {0, 1, 0}, {.5f, 0, 0}, {.5f, 1, 0}};
        source.rest = source.points;
        source.widths = {.02f, .02f, .03f, .03f};
        source.curveId = {99, 3};
        source.skinPrim = {0, 0};
        source.skinPrimUv = {{0, 0}, {.5f, 0}};
        source.rootFrame = usdGen::UsdGenMatrix4dArray(2, usdGen::UsdGenMatrix4d(1.0));
        UsdGenNodeDesc deform;
        deform.path = SdfPath("/Groom/Ops/deform");
        deform.type = TfToken("UsdGenDeform");
        deform.inputs = {deformDesc.nodes.front().path};
        deform.surfaces = {surface.path};
        deform.params = {{TfToken("rbfSamples"), VtValue(4), false}};
        deformDesc.nodes = {deformDesc.nodes.front(), deform};
        deformDesc.terminal = deform.path;
        UsdGenDiagnostics messages;
        auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(deformDesc), &messages);
        CHECK(handle && !messages.HasErrors());
        auto movedDescription = deformDesc;
        movedDescription.xformMatrix.SetTranslate(GfVec3d(1, 2, 3));
        CHECK(Rejects(movedDescription));
        auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
        CHECK(plan->Steps()[0].deform.rootStableIds == std::vector<uint64_t>({99, 3}));
        CHECK(plan->Steps()[0].deform.rootTargets.size() == 6 &&
              plan->Steps()[0].deform.rootTargets[0] == 0 &&
              plan->Steps()[0].deform.rootTargets[3] == .5f);
        auto const& publication = handle->Metadata()->Tasks().back();
        CHECK(handle->Metadata()->Values()[publication.resources[0].inputValue].resource ==
              UsdGenExecutionDataKind::ElementGeometry);
        CHECK(handle->Metadata()->Values()[publication.resources[1].inputValue].resource ==
              UsdGenExecutionDataKind::Widths);
        UsdGenNodeDesc cull;
        cull.path = SdfPath("/Groom/Ops/cull"); cull.type = TfToken("UsdGenLength");
        cull.inputs = {deformDesc.nodes.front().path};
        cull.params = {{TfToken("length:mode"), VtValue(TfToken("cull")), false},
                       {TfToken("cullThreshold"), VtValue(.5f), false}};
        deformDesc.nodes.back().inputs = {cull.path};
        deformDesc.nodes.insert(deformDesc.nodes.begin() + 1, cull);
        CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(deformDesc)));
        auto width = MakeDesc().nodes.back(); width.inputs = {deform.path};
        deformDesc.nodes.push_back(width);
        auto repeated = deform; repeated.path = SdfPath("/Groom/Ops/deformAgain");
        repeated.inputs = {width.path};
        deformDesc.nodes.push_back(repeated); deformDesc.terminal = repeated.path;
        CHECK(Rejects(deformDesc));
        // Deformation provenance also survives a topology barrier.
        deformDesc.nodes.pop_back(); deformDesc.nodes.pop_back();
        auto after = cull; after.path = SdfPath("/Groom/Ops/cullAfter");
        after.inputs = {deform.path}; deformDesc.nodes.push_back(after);
        repeated.inputs = {after.path}; deformDesc.nodes.push_back(repeated);
        CHECK(Rejects(deformDesc));
    }
    auto desc = MakeDesc();
    UsdGenDiagnostics diagnostics;
    auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(desc), &diagnostics);
    CHECK(handle && !diagnostics.HasErrors());
    CHECK(handle->Backend() == usdGenTestUtils::Backend("vulkan"));
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    CHECK(GetUsdGenExecutionBackendContract(usdGenTestUtils::Backend("vulkan")).Available());
#else
    CHECK(!GetUsdGenExecutionBackendContract(usdGenTestUtils::Backend("vulkan")).Available());
#endif

    auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
    CHECK(plan && plan->Descriptor() && plan->Descriptor().get() != static_cast<void const *>(&desc));
    CHECK(plan->SourceNodeKey() == usdGenUsd::KeyOf(desc.nodes[0].path) && plan->WidthNodeKey() == usdGenUsd::KeyOf(desc.nodes[1].path));
    CHECK(plan->CurveGeneration() == 19 && plan->TopologyVersion() == 19 && plan->ValueVersion() == 19);
    CHECK(plan->SourceControls().useRest && plan->SourceControls().hasAuthoredRest &&
          !plan->SourceControls().restFromCurrentPoints &&
          plan->SourceControls().idSource == TfToken("primvar") &&
          plan->SourceControls().expectEpoch == "epoch-19" &&
          plan->SourceControls().staleAction == TfToken("block") &&
          plan->SourceControls().rebind == TfToken("never"));
    CHECK(plan->Width().width == .125f && !plan->Width().replace);

    // The descriptor and selected C3 snapshot are owned by the plan, not its caller.
    desc.curveSets[0].points[0] = usdGen::UsdGenVec3f(9, 9, 9);
    desc.nodes[1].params[0].value = VtValue(.5f);
    CHECK(plan->Source().points[0] == usdGen::UsdGenVec3f(0, 0, 0) &&
          plan->Descriptor()->nodes[1].params[0].value == VtValue(.125f) &&
          plan->Width().width == .125f);

    auto metadata = handle->Metadata();
    CHECK(metadata && metadata->Backend() == "vulkan" && metadata->CapabilityVersion() == 1 &&
          metadata->Shape() == UsdGenExecutionPlanShape::LinearAuthoredChain &&
          metadata->TerminalTask() == 2 && metadata->Operators().size() == 2 &&
          metadata->Tasks().size() == 3 && metadata->Values().size() == 7);
    CHECK(metadata->Operators()[0].key == usdGenUsd::KeyOf(SdfPath("/Groom/Ops/source")) &&
          metadata->Operators()[0].authoredOrderKey == 0 &&
          metadata->Operators()[0].type == TfToken("UsdGenCurveSource") &&
          metadata->Operators()[1].key == usdGenUsd::KeyOf(SdfPath("/Groom/Ops/width")) &&
          metadata->Operators()[1].authoredOrderKey == 1 &&
          metadata->Operators()[1].type == TfToken("UsdGenWidth"));
    auto const& source = metadata->Tasks()[0]; auto const& width = metadata->Tasks()[1];
    auto const& publication = metadata->Tasks()[2];
    CHECK(source.id == 0 && source.semanticNode == 0 && source.kind == UsdGenExecutionTaskKind::Source &&
          source.key == usdGenUsd::KeyOf(SdfPath("/Groom/Ops/source")) && source.resources.size() == 5 &&
          source.resources[0].resource == UsdGenExecutionDataKind::ElementGeometry &&
          source.resources[0].access == UsdGenExecutionResourceAccess::ReadWrite &&
          source.resources[0].inputValue == 0 && source.resources[0].outputValue == 1 &&
          source.resources[1].resource == UsdGenExecutionDataKind::Widths &&
          source.resources[1].outputValue == 2);
    CHECK(width.id == 1 && width.semanticNode == 1 && width.kind == UsdGenExecutionTaskKind::Operator &&
          width.resources.size() == 5 && width.resources[0].inputValue == 1 &&
          width.resources[1].inputValue == 2 && width.resources[1].outputValue == 3 &&
          width.dependencies == std::vector<uint32_t>{0} &&
          HasDependency(width, 0, UsdGenExecutionDependencySemanticData));
    CHECK(publication.id == 2 && publication.kind == UsdGenExecutionTaskKind::Publication &&
          publication.resources.size() == 5 && publication.resources[0].inputValue == 1 &&
          publication.resources[1].inputValue == 3 &&
          publication.dependencies == std::vector<uint32_t>({0, 1}) &&
          HasDependency(publication, 0, UsdGenExecutionDependencyLifetimePublicationJoin) &&
          HasDependency(publication, 1, UsdGenExecutionDependencyLifetimePublicationJoin));
    CHECK(metadata->Values()[0].id == 0 && metadata->Values()[0].resource == UsdGenExecutionDataKind::ElementGeometry &&
          metadata->Values()[0].producerTask == UINT32_MAX && metadata->Values()[0].storage == UsdGenExecutionValueStorage::ExternalImmutable &&
          metadata->Values()[1].id == 1 && metadata->Values()[1].resource == UsdGenExecutionDataKind::ElementGeometry &&
          metadata->Values()[1].producerTask == 0 && metadata->Values()[1].version == 1 && metadata->Values()[1].storage == UsdGenExecutionValueStorage::JobOwnedImmutable &&
          metadata->Values()[2].id == 2 && metadata->Values()[2].resource == UsdGenExecutionDataKind::Widths &&
          metadata->Values()[2].producerTask == 0 && metadata->Values()[2].version == 1 && metadata->Values()[2].storage == UsdGenExecutionValueStorage::JobOwnedImmutable &&
          metadata->Values()[3].id == 3 && metadata->Values()[3].resource == UsdGenExecutionDataKind::Widths &&
          metadata->Values()[3].producerTask == 1 && metadata->Values()[3].version == 2 && metadata->Values()[3].storage == UsdGenExecutionValueStorage::PublishedImmutable);
    for (uint32_t valueId : {4u, 5u, 6u}) {
        auto const kind = valueId == 4 ? UsdGenExecutionDataKind::ElementTopology
                         : valueId == 5 ? UsdGenExecutionDataKind::StableIds
                                        : UsdGenExecutionDataKind::NamedChannels;
        auto const resourceIndex = valueId - 2;
        CHECK(metadata->Values()[valueId].resource == kind &&
              metadata->Values()[valueId].producerTask == 0 &&
              metadata->Values()[valueId].storage == UsdGenExecutionValueStorage::JobOwnedImmutable);
        CHECK(source.resources[resourceIndex].resource == kind &&
              source.resources[resourceIndex].outputValue == valueId);
        for (auto const* task : {&width, &publication})
            CHECK(task->resources[resourceIndex].resource == kind &&
                  task->resources[resourceIndex].access == UsdGenExecutionResourceAccess::Read &&
                  task->resources[resourceIndex].producerTask == 0 &&
                  task->resources[resourceIndex].inputValue == valueId);
    }
    CHECK(metadata->Values()[6].id == 6 &&
          metadata->Values()[6].resource == UsdGenExecutionDataKind::NamedChannels &&
          metadata->Values()[6].producerTask == 0 &&
          metadata->Values()[6].storage == UsdGenExecutionValueStorage::JobOwnedImmutable);
    auto lengthDesc = MakeDesc();
    AddAuthoredPlane(&lengthDesc.curveSets[0], "pointPlane",
                     UsdGenAuthoredPlaneType::Float32,
                     UsdGenAuthoredPlaneDomain::Point, 2);
    UsdGenNodeDesc lengthNode;
    lengthNode.path = SdfPath("/Groom/Ops/length"); lengthNode.type = TfToken("UsdGenLength");
    lengthNode.inputs = {lengthDesc.nodes[0].path};
    lengthNode.params = {{TfToken("length:value"), VtValue(1.25), false}};
    lengthDesc.nodes[1].inputs = {lengthNode.path};
    lengthDesc.nodes.insert(lengthDesc.nodes.begin() + 1, lengthNode);
    auto lengthDiagnostics = UsdGenDiagnostics{};
    auto lengthHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(lengthDesc), &lengthDiagnostics);
    if (!lengthHandle) for (auto const& error : lengthDiagnostics.errors)
        std::fprintf(stderr, "lengthHandle: %s\n", error.c_str());
    CHECK(lengthHandle);
    auto randomLengthDesc = lengthDesc;
    randomLengthDesc.nodes[1].seed = -17;
    randomLengthDesc.nodes[1].params.push_back(
        {TfToken("length:random"), VtValue(usdGen::UsdGenVec2f(1.5f, .5f)), false});
    auto randomLengthHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(randomLengthDesc));
    CHECK(randomLengthHandle);
    auto randomLengthPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(randomLengthHandle->Payload());
    CHECK(randomLengthPlan && randomLengthPlan->Steps()[0].randomLo == 1.5f &&
          randomLengthPlan->Steps()[0].randomHi == .5f && randomLengthPlan->Steps()[0].randomSeed == -17 &&
          randomLengthPlan->Steps().size() == 2);
    for (int seed : {std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}) {
        auto seeded = randomLengthDesc;
        seeded.nodes[1].seed = seed;
        auto seededHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(seeded));
        CHECK(seededHandle);
        auto seededPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(seededHandle->Payload());
        CHECK(seededPlan && seededPlan->Steps()[0].randomSeed == seed);
    }
    // Culling is the topology-changing Length form.  The control envelope is
    // literal and finite; length:value is accepted for schema compatibility
    // but does not affect cull admission.
    auto cullDesc = MakeRootedDesc();
    UsdGenNodeDesc cull;
    cull.path = SdfPath("/Groom/Ops/cull"); cull.type = TfToken("UsdGenLength");
    cull.inputs = {cullDesc.nodes[0].path};
    cull.params = {{TfToken("length:mode"), VtValue(TfToken("cull")), false},
                   {TfToken("length:value"), VtValue(1.25f), false},
                   {TfToken("length:random"), VtValue(usdGen::UsdGenVec2f(1,1)), false},
                   {TfToken("minRemainingLength"), VtValue(0.0f), false},
                   {TfToken("cullThreshold"), VtValue(.25f), false}};
    auto cullWidth = cullDesc.nodes[1]; cullWidth.path = SdfPath("/Groom/Ops/cullWidth");
    cullWidth.inputs = {cull.path};
    cullDesc.nodes = {cullDesc.nodes[0], cull, cullWidth}; cullDesc.terminal = cullWidth.path;
    diagnostics = {};
    auto cullHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(cullDesc), &diagnostics);
    CHECK(cullHandle && !diagnostics.HasErrors());
    auto cullMetadata = cullHandle->Metadata();
    CHECK(cullMetadata && cullMetadata->Tasks().size() == 4 && cullMetadata->Values().size() >= 9);
    auto findTask = [](std::shared_ptr<const UsdGenExecutionPlanMetadata> const& metadata,
                       char const* path) -> UsdGenExecutionTaskMetadata const* {
        auto found = std::find_if(metadata->Tasks().begin(), metadata->Tasks().end(),
            [&](auto const& task) { return task.key == usdGenUsd::KeyOf(SdfPath(path)); });
        return found == metadata->Tasks().end() ? nullptr : &*found;
    };
    auto const* cullTask = findTask(cullMetadata, "/Groom/Ops/cull");
    auto const* cullWidthTask = findTask(cullMetadata, "/Groom/Ops/cullWidth");
    CHECK(cullTask && cullTask->topologyBarrier && cullWidthTask &&
          cullTask->resources.size() == 6);
    auto hasUse = [](UsdGenExecutionTaskMetadata const& task, UsdGenExecutionDataKind kind,
                     UsdGenExecutionResourceAccess access) {
        return std::any_of(task.resources.begin(), task.resources.end(),
            [&](auto const& use) { return use.resource == kind && use.access == access; });
    };
    for (auto kind : {UsdGenExecutionDataKind::ElementGeometry,
                      UsdGenExecutionDataKind::Widths,
                      UsdGenExecutionDataKind::ElementTopology,
                      UsdGenExecutionDataKind::StableIds,
                      UsdGenExecutionDataKind::NamedChannels,
                      UsdGenExecutionDataKind::RootBindings})
        CHECK(hasUse(*cullTask, kind, UsdGenExecutionResourceAccess::ReadWrite));
    CHECK(cullWidthTask->resources[0].resource == UsdGenExecutionDataKind::ElementGeometry &&
          cullWidthTask->resources[0].inputValue != UINT32_MAX);
    auto findUse = [](UsdGenExecutionTaskMetadata const& task,
                      UsdGenExecutionDataKind kind,
                      UsdGenExecutionResourceAccess access,
                      uint32_t inputValue = UINT32_MAX) -> UsdGenExecutionResourceUse const* {
        auto found = std::find_if(task.resources.begin(), task.resources.end(),
            [&](auto const& use) { return use.resource == kind && use.access == access &&
                (inputValue == UINT32_MAX || use.inputValue == inputValue); });
        return found == task.resources.end() ? nullptr : &*found;
    };
    auto const* sourceTopology = findUse(cullMetadata->Tasks()[0],
        UsdGenExecutionDataKind::ElementTopology, UsdGenExecutionResourceAccess::Write);
    auto const* cullTopology = findUse(*cullTask,
        UsdGenExecutionDataKind::ElementTopology, UsdGenExecutionResourceAccess::ReadWrite);
    auto const* widthTopology = findUse(*cullWidthTask,
        UsdGenExecutionDataKind::ElementTopology, UsdGenExecutionResourceAccess::Read);
    CHECK(sourceTopology && sourceTopology->outputValue == 4 && sourceTopology->producerTask == UINT32_MAX &&
          cullTopology && cullTopology->producerTask == 0 && cullTopology->inputValue == 4 &&
          cullTopology->outputValue != UINT32_MAX && widthTopology &&
          widthTopology->producerTask == cullTask->id &&
          widthTopology->inputValue == cullTopology->outputValue);
    CHECK(cullMetadata->Values()[cullTopology->outputValue].resource == UsdGenExecutionDataKind::ElementTopology &&
          cullMetadata->Values()[cullTopology->outputValue].producerTask == cullTask->id);
    auto cullPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(cullHandle->Payload());
    CHECK(cullPlan && cullPlan->Steps().size() == 2 &&
          cullPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthCull &&
          cullPlan->Steps()[0].cullThreshold == .25f);
    auto cutExtendCull = cullDesc;
    cutExtendCull.nodes[1].params.push_back({TfToken("length:method"), VtValue(TfToken("cutExtend")), false});
    cutExtendCull.nodes[1].params.push_back({TfToken("rebuild"), VtValue(TfToken("keepParam")), false});
    diagnostics = {};
    auto cutExtendCullHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(cutExtendCull), &diagnostics);
    CHECK(cutExtendCullHandle && !diagnostics.HasErrors());
    auto cutExtendCullPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(cutExtendCullHandle->Payload());
    CHECK(cutExtendCullPlan && cutExtendCullPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthCull &&
          cutExtendCullPlan->Steps()[0].lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend);
    auto scaleThreshold = randomLengthDesc;
    scaleThreshold.nodes[1].params.push_back({TfToken("length:method"), VtValue(TfToken("cutExtend")), false});
    scaleThreshold.nodes[1].params.push_back({TfToken("rebuild"), VtValue(TfToken("reparam")), false});
    scaleThreshold.nodes[1].params.push_back({TfToken("cullThreshold"), VtValue(.25f), false});
    diagnostics = {};
    auto scaleThresholdHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(scaleThreshold), &diagnostics);
    CHECK(scaleThresholdHandle && !diagnostics.HasErrors());
    auto scaleThresholdPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(scaleThresholdHandle->Payload());
    CHECK(scaleThresholdPlan && scaleThresholdPlan->Steps().size() == 3 &&
          scaleThresholdPlan->IntermediateCount() == 2 &&
          scaleThresholdPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthScale &&
          scaleThresholdPlan->Steps()[1].kind == VulkanSourceWidthStage::Kind::LengthCull &&
          scaleThresholdPlan->Steps()[0].lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend &&
          scaleThresholdPlan->Steps()[1].lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend &&
          scaleThresholdPlan->Steps()[0].lengthRebuild == VulkanSourceWidthStage::LengthRebuild::Reparam &&
          scaleThresholdPlan->Steps()[1].lengthRebuild == VulkanSourceWidthStage::LengthRebuild::Reparam &&
          scaleThresholdPlan->Steps()[1].cullThreshold == .25f);
    CHECK(scaleThresholdPlan->Steps()[0].randomLo == 1.5f &&
          scaleThresholdPlan->Steps()[0].randomHi == .5f &&
          scaleThresholdPlan->Steps()[0].randomSeed == -17 &&
          scaleThresholdPlan->Steps()[1].randomLo == 1.5f &&
          scaleThresholdPlan->Steps()[1].randomHi == .5f &&
          scaleThresholdPlan->Steps()[1].randomSeed == -17);
    auto scaleThresholdMetadata = scaleThresholdHandle->Metadata();
    CHECK(scaleThresholdMetadata && scaleThresholdMetadata->Tasks().size() == 5);
    CHECK(scaleThresholdPlan->LengthNodeKey() == usdGenUsd::KeyOf(scaleThreshold.nodes[1].path));
    CHECK(scaleThresholdMetadata->FindSemanticTask(1) &&
          scaleThresholdMetadata->FindSemanticTask(1)->key == usdGenUsd::KeyOf(scaleThreshold.nodes[1].path) &&
          scaleThresholdMetadata->FindSemanticTask(1)->topologyBarrier);
    auto const* authoredLengthTask = findTask(scaleThresholdMetadata, "/Groom/Ops/length");
    auto finalCullTask = std::find_if(scaleThresholdMetadata->Tasks().begin(),
        scaleThresholdMetadata->Tasks().end(), [](auto const& task) {
            return task.key == usdGenUsd::KeyOf(SdfPath("/Groom/Ops/length")) &&
                   task.kind == UsdGenExecutionTaskKind::Operator;
        });
    auto transformTask = std::find_if(scaleThresholdMetadata->Tasks().begin(),
        scaleThresholdMetadata->Tasks().end(), [&](auto const& task) {
            return task.semanticNode == UINT32_MAX &&
                   task.type == TfToken("VulkanLengthTransform") &&
                   scaleThresholdPlan->Descriptor()->Label(task.key).find("__vulkanLengthTransform") !=
                       std::string::npos;
        });
    CHECK(authoredLengthTask && transformTask != scaleThresholdMetadata->Tasks().end() &&
          finalCullTask != scaleThresholdMetadata->Tasks().end() &&
          transformTask->key != authoredLengthTask->key &&
          finalCullTask->semanticNode == 1 &&
          finalCullTask->key == authoredLengthTask->key);
    auto const* finalCullTopology = findUse(*finalCullTask,
        UsdGenExecutionDataKind::ElementTopology, UsdGenExecutionResourceAccess::ReadWrite);
    CHECK(finalCullTopology && finalCullTopology->inputValue != UINT32_MAX &&
          finalCullTopology->outputValue != UINT32_MAX &&
          scaleThresholdMetadata->Values()[finalCullTopology->outputValue].resource ==
              UsdGenExecutionDataKind::ElementTopology &&
          scaleThresholdMetadata->Values()[finalCullTopology->outputValue].producerTask ==
              finalCullTask->id);
    auto const* finalCullWidth = findTask(scaleThresholdMetadata, "/Groom/Ops/width");
    auto const* finalCullGeometry = finalCullWidth ? findUse(*finalCullWidth,
        UsdGenExecutionDataKind::ElementGeometry, UsdGenExecutionResourceAccess::Read) : nullptr;
    CHECK(finalCullWidth && finalCullGeometry &&
          finalCullGeometry->producerTask == finalCullTask->id);
    auto finalCullIds = findUse(*finalCullTask, UsdGenExecutionDataKind::StableIds,
                                UsdGenExecutionResourceAccess::ReadWrite);
    CHECK(finalCullIds && finalCullIds->inputValue == 5 && finalCullIds->producerTask == 0);
    auto collisionDesc = scaleThreshold;
    UsdGenNodeDesc collision;
    collision.path = SdfPath("/Groom/Ops/length/__vulkanLengthTransform0");
    collision.type = TfToken("UsdGenWidth");
    collision.inputs = {collisionDesc.nodes[1].path};
    collisionDesc.nodes.push_back(collision);
    collisionDesc.nodes[2].inputs = {collision.path};
    diagnostics = {};
    auto collisionHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(collisionDesc), &diagnostics);
    CHECK(collisionHandle && !diagnostics.HasErrors());
    auto collisionMetadata = collisionHandle->Metadata();
    CHECK(collisionMetadata);
    auto collisionPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(collisionHandle->Payload());
    auto collisionFinalCull = std::find_if(collisionMetadata->Tasks().begin(),
        collisionMetadata->Tasks().end(), [](auto const& task) {
            return task.key == usdGenUsd::KeyOf(SdfPath("/Groom/Ops/length")) &&
                   task.semanticNode == 1;
        });
    CHECK(collisionFinalCull != collisionMetadata->Tasks().end() &&
          std::count_if(collisionMetadata->Tasks().begin(), collisionMetadata->Tasks().end(),
              [&](auto const& task) {
                  return task.semanticNode == UINT32_MAX &&
                         collisionPlan->Descriptor()->Label(task.key).find("__vulkanLengthTransform") !=
                             std::string::npos;
              }) == 1);
    std::set<UsdGenNodeKey> taskPaths;
    for (auto const& task : collisionMetadata->Tasks()) CHECK(taskPaths.insert(task.key).second);
    // Expanded native stages retain dependency ordering across the former
    // 64-stage limit, independently of the authored graph cardinality.
    auto boundary = MakeDesc();
    auto boundaryWidth = boundary.nodes.back();
    boundary.nodes.resize(1);
    auto previous = boundary.nodes.front().path;
    for (unsigned i = 0; i != 32; ++i) {
        auto length = scaleThreshold.nodes[1];
        length.path = SdfPath("/Groom/Ops/boundary" + std::to_string(i));
        length.inputs = {previous};
        if (i == 31) length.params.back().value = VtValue(0.f);
        previous = length.path; boundary.nodes.push_back(length);
    }
    boundaryWidth.inputs = {previous}; boundary.nodes.push_back(boundaryWidth);
    boundary.terminal = boundaryWidth.path;
    auto boundaryHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(boundary)); CHECK(boundaryHandle);
    auto boundaryPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(boundaryHandle->Payload());
    CHECK(boundaryPlan->Steps().size() == 64 && boundaryPlan->IntermediateCount() == 63);
    boundary.nodes[32].params.back().value = VtValue(.25f);
    boundaryHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(boundary)); CHECK(boundaryHandle);
    boundaryPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(boundaryHandle->Payload());
    CHECK(boundaryPlan->Steps().size() == 65 && boundaryPlan->IntermediateCount() == 64);
    CHECK(boundaryHandle->Metadata()->Tasks().back().id == 66);

    auto repeatedCull = cullDesc;
    auto cull2 = cull;
    cull2.path = SdfPath("/Groom/Ops/cull2"); cull2.inputs = {cull.path};
    repeatedCull.nodes.insert(repeatedCull.nodes.end() - 1, cull2);
    repeatedCull.nodes.back().inputs = {cull2.path}; repeatedCull.terminal = repeatedCull.nodes.back().path;
    diagnostics = {};
    auto repeatedHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(repeatedCull), &diagnostics);
    CHECK(repeatedHandle && !diagnostics.HasErrors());
    auto repeatedPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(repeatedHandle->Payload());
    CHECK(repeatedPlan && repeatedPlan->Steps().size() == 3 &&
          repeatedPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthCull &&
          repeatedPlan->Steps()[1].kind == VulkanSourceWidthStage::Kind::LengthCull);
    auto repeatedMetadata = repeatedHandle->Metadata();
    auto const* cull2Task = findTask(repeatedMetadata, "/Groom/Ops/cull2");
    auto const* repeatedWidthTask = findTask(repeatedMetadata, "/Groom/Ops/cullWidth");
    auto const* firstCullTask = findTask(repeatedMetadata, "/Groom/Ops/cull");
    CHECK(repeatedMetadata && cull2Task && repeatedWidthTask && firstCullTask);
    auto const* cull2Topology = findUse(*cull2Task,
        UsdGenExecutionDataKind::ElementTopology, UsdGenExecutionResourceAccess::ReadWrite);
    auto const* repeatedWidthTopology = findUse(*repeatedWidthTask,
        UsdGenExecutionDataKind::ElementTopology, UsdGenExecutionResourceAccess::Read);
    CHECK(cull2Topology && cull2Topology->producerTask == firstCullTask->id &&
          cull2Topology->inputValue != 4 && cull2Topology->outputValue != UINT32_MAX &&
          repeatedWidthTopology && repeatedWidthTopology->producerTask == cull2Task->id &&
          repeatedWidthTopology->inputValue == cull2Topology->outputValue &&
          repeatedMetadata->Values()[cull2Topology->outputValue].resource == UsdGenExecutionDataKind::ElementTopology &&
          repeatedMetadata->Values()[cull2Topology->outputValue].producerTask == cull2Task->id);
    auto const* cull2Named = findUse(*cull2Task,
        UsdGenExecutionDataKind::NamedChannels, UsdGenExecutionResourceAccess::ReadWrite);
    CHECK(cull2Named && cull2Named->inputValue != 6 &&
          cull2Named->producerTask == firstCullTask->id);

    auto branchEqual = cullDesc;
    auto rightWidth = cullWidth; rightWidth.path = SdfPath("/Groom/Ops/rightWidth");
    rightWidth.inputs = {cull.path};
    UsdGenNodeDesc branchBlend; branchBlend.path = SdfPath("/Groom/Ops/cullBlend");
    branchBlend.type = TfToken("UsdGenWidthBlend");
    branchBlend.inputs = {branchEqual.nodes[2].path, rightWidth.path};
    branchBlend.params = {{TfToken("widthBlend:weight"), VtValue(.5f), false}};
    branchEqual.nodes = {branchEqual.nodes[0], cull, branchEqual.nodes[2], rightWidth, branchBlend};
    branchEqual.terminal = branchBlend.path;
    diagnostics = {};
    auto branchHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(branchEqual), &diagnostics);
    CHECK(branchHandle && !diagnostics.HasErrors());
    auto branchMetadata = branchHandle->Metadata();
    CHECK(branchMetadata);
    auto branchPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(branchHandle->Payload());
    CHECK(branchPlan && branchPlan->Steps().back().kind == VulkanSourceWidthStage::Kind::WidthBlend &&
          !branchPlan->Steps().back().requiresNonWidthProof);
    auto unequal = branchEqual;
    auto rightCull = cull; rightCull.path = SdfPath("/Groom/Ops/rightCull");
    rightCull.inputs = {unequal.nodes[0].path};
    unequal.nodes.insert(unequal.nodes.end() - 1, rightCull);
    unequal.nodes[3].inputs = {rightCull.path};
    unequal.terminal = unequal.nodes.back().path;
    diagnostics = {};
    auto unequalHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(unequal), &diagnostics);
    CHECK(unequalHandle && !diagnostics.HasErrors());
    auto unequalPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(unequalHandle->Payload());
    CHECK(unequalPlan && unequalPlan->Steps().back().requiresNonWidthProof);
    auto unequalMetadata = unequalHandle->Metadata();
    auto const* unequalRightCull = findTask(unequalMetadata, "/Groom/Ops/rightCull");
    auto const* unequalRightWidth = findTask(unequalMetadata, "/Groom/Ops/rightWidth");
    auto const* unequalBlend = findTask(unequalMetadata, "/Groom/Ops/cullBlend");
    CHECK(unequalMetadata && unequalRightCull && unequalRightWidth && unequalBlend);
    // An independent equal-parameter cull has its own non-width lineage. The
    // comparison packet must bind every right-branch value to that producer.
    for (auto kind : {UsdGenExecutionDataKind::ElementGeometry,
                      UsdGenExecutionDataKind::ElementTopology,
                      UsdGenExecutionDataKind::StableIds,
                      UsdGenExecutionDataKind::NamedChannels,
                      UsdGenExecutionDataKind::RootBindings}) {
        auto found = std::find_if(unequalBlend->resources.begin(), unequalBlend->resources.end(),
            [&](auto const& use) { return use.resource == kind &&
                use.access == UsdGenExecutionResourceAccess::Read &&
                use.producerTask == unequalRightCull->id; });
        auto const* use = found == unequalBlend->resources.end() ? nullptr : &*found;
        CHECK(use && use->producerTask == unequalRightCull->id && use->inputValue != UINT32_MAX);
    }
    // Width is excluded from the non-width comparison and comes from the
    // subsequent right Width operator, not its Cull predecessor.
    auto rightWidthsIt = std::find_if(unequalBlend->resources.begin(), unequalBlend->resources.end(),
        [&](auto const& use) { return use.resource == UsdGenExecutionDataKind::Widths &&
            use.access == UsdGenExecutionResourceAccess::Read &&
            use.producerTask == unequalRightWidth->id; });
    CHECK(rightWidthsIt != unequalBlend->resources.end() && rightWidthsIt->inputValue != UINT32_MAX);
    auto lengthPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(lengthHandle->Payload());
    CHECK(lengthPlan->HasLength() && lengthPlan->LengthFactor() == 1.25f && lengthPlan->LengthNodeKey() == usdGenUsd::KeyOf(lengthNode.path) &&
          lengthPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthScale &&
          lengthPlan->Steps()[0].lengthMode == VulkanSourceWidthStage::LengthMode::Scale);
    auto cutExtendScale = lengthDesc;
    cutExtendScale.nodes[1].params.push_back({TfToken("length:method"), VtValue(TfToken("cutExtend")), false});
    cutExtendScale.nodes[1].params.push_back({TfToken("rebuild"), VtValue(TfToken("keepParam")), false});
    diagnostics = {};
    auto cutExtendScaleHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(cutExtendScale), &diagnostics);
    CHECK(cutExtendScaleHandle && !diagnostics.HasErrors());
    auto cutExtendScalePlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(cutExtendScaleHandle->Payload());
    CHECK(cutExtendScalePlan && cutExtendScalePlan->Steps()[0].lengthMethod ==
          VulkanSourceWidthStage::LengthMethod::CutExtend);
    auto lengthMetadata = lengthHandle->Metadata();
    auto animatedLength = lengthDesc;
    animatedLength.nodes[1].params[0].animated = true;
    CHECK(Rejects(animatedLength));
    CHECK(lengthMetadata->Tasks().size() == 4 && lengthMetadata->Values().size() == 8 && lengthMetadata->TerminalTask() == 3);
    CHECK(lengthMetadata->Values()[7].producerTask == 1 && lengthMetadata->Values()[3].producerTask == 2);
    auto const* scaleTopology = findUse(lengthMetadata->Tasks()[1],
        UsdGenExecutionDataKind::ElementTopology, UsdGenExecutionResourceAccess::Read, 4);
    auto const* scaleWidthTopology = findUse(lengthMetadata->Tasks()[2],
        UsdGenExecutionDataKind::ElementTopology, UsdGenExecutionResourceAccess::Read, 4);
    CHECK(lengthMetadata->Values()[4].resource == UsdGenExecutionDataKind::ElementTopology &&
          lengthMetadata->Values()[4].producerTask == 0 && lengthMetadata->Values()[4].version == 1 &&
          scaleTopology && scaleTopology->producerTask == 0 &&
          scaleWidthTopology && scaleWidthTopology->producerTask == 0);
    CHECK(lengthMetadata->Tasks()[1].resources[0].inputValue == 1 && lengthMetadata->Tasks()[1].resources[0].outputValue == 7);
    CHECK(lengthMetadata->Tasks()[2].resources[0].inputValue == 7 && lengthMetadata->Tasks()[3].resources[0].inputValue == 7);
    // Literal Length set is admitted as fixed-topology point COW.  The source
    // topology and named-channel bundle remain inherited from Source, while
    // the geometry value is replaced and consumed by the terminal Width.
    auto setDesc = lengthDesc;
    setDesc.nodes[1].params.push_back({TfToken("length:mode"), VtValue(TfToken("set")), false});
    diagnostics = {};
    auto setHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(setDesc), &diagnostics);
    CHECK(setHandle && !diagnostics.HasErrors());
    auto setPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(setHandle->Payload());
    CHECK(setPlan && setPlan->Steps().size() == 2 &&
          setPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthScale &&
          setPlan->Steps()[0].lengthMode == VulkanSourceWidthStage::LengthMode::Set &&
          setPlan->Steps()[0].factor == 1.25f);
    auto cutExtendSet = setDesc;
    cutExtendSet.nodes[1].params.push_back({TfToken("length:method"), VtValue(TfToken("cutExtend")), false});
    cutExtendSet.nodes[1].params.push_back({TfToken("rebuild"), VtValue(TfToken("keepParam")), false});
    diagnostics = {};
    auto cutExtendSetHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(cutExtendSet), &diagnostics);
    CHECK(cutExtendSetHandle && !diagnostics.HasErrors());
    auto cutExtendSetPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(cutExtendSetHandle->Payload());
    CHECK(cutExtendSetPlan && cutExtendSetPlan->Steps()[0].lengthMode == VulkanSourceWidthStage::LengthMode::Set &&
          cutExtendSetPlan->Steps()[0].lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend);
    auto setMetadata = setHandle->Metadata();
    CHECK(setMetadata && setMetadata->Tasks().size() == 4 && setMetadata->Values().size() == 8 &&
          setMetadata->TerminalTask() == 3);
    auto const* setTask = findTask(setMetadata, "/Groom/Ops/length");
    auto const* setWidthTask = findTask(setMetadata, "/Groom/Ops/width");
    CHECK(setTask && setWidthTask && !setTask->topologyBarrier);
    auto const* setGeometry = findUse(*setTask, UsdGenExecutionDataKind::ElementGeometry,
                                      UsdGenExecutionResourceAccess::ReadWrite);
    auto const* setTopology = findUse(*setTask, UsdGenExecutionDataKind::ElementTopology,
                                      UsdGenExecutionResourceAccess::Read, 4);
    auto const* setNamed = findUse(*setTask, UsdGenExecutionDataKind::NamedChannels,
                                   UsdGenExecutionResourceAccess::Read, 6);
    auto const* setWidthGeometry = findUse(*setWidthTask, UsdGenExecutionDataKind::ElementGeometry,
                                           UsdGenExecutionResourceAccess::Read);
    CHECK(setGeometry && setGeometry->inputValue == 1 && setGeometry->outputValue == 7 &&
          setGeometry->producerTask == 0 && setTopology && setTopology->producerTask == 0 &&
          setNamed && setNamed->producerTask == 0 && setWidthGeometry &&
          setWidthGeometry->inputValue == 7 && setWidthGeometry->producerTask == setTask->id &&
          setMetadata->Values()[4].producerTask == 0 &&
          setMetadata->Values()[6].producerTask == 0 &&
          setMetadata->Values()[7].resource == UsdGenExecutionDataKind::ElementGeometry &&
          setMetadata->Values()[7].producerTask == setTask->id);
    auto setPublication = setMetadata->Tasks().back();
    auto const* publicationTopology = findUse(setPublication,
        UsdGenExecutionDataKind::ElementTopology, UsdGenExecutionResourceAccess::Read, 4);
    auto const* publicationNamed = findUse(setPublication,
        UsdGenExecutionDataKind::NamedChannels, UsdGenExecutionResourceAccess::Read, 6);
    CHECK(publicationTopology && publicationTopology->producerTask == 0 &&
          publicationNamed && publicationNamed->producerTask == 0);

    // A Set before Cull leaves Cull consuming Source's topology; a Set after
    // Cull consumes Cull's COW topology and publishes a fresh geometry value.
    auto setThenCull = cullDesc;
    auto setBeforeCull = cull;
    setBeforeCull.path = SdfPath("/Groom/Ops/set");
    setBeforeCull.inputs = {setThenCull.nodes[0].path};
    setBeforeCull.params = {{TfToken("length:mode"), VtValue(TfToken("set")), false},
                            {TfToken("length:value"), VtValue(1.25f), false}};
    setThenCull.nodes[1].inputs = {setBeforeCull.path};
    setThenCull.nodes.insert(setThenCull.nodes.begin() + 1, setBeforeCull);
    diagnostics = {};
    auto setThenCullHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(setThenCull), &diagnostics);
    CHECK(setThenCullHandle && !diagnostics.HasErrors());
    auto setThenCullMetadata = setThenCullHandle->Metadata();
    auto const* beforeSetTask = findTask(setThenCullMetadata, "/Groom/Ops/set");
    auto const* afterCullTask = findTask(setThenCullMetadata, "/Groom/Ops/cull");
    CHECK(beforeSetTask && afterCullTask);
    auto const* beforeSetTopology = findUse(*beforeSetTask, UsdGenExecutionDataKind::ElementTopology,
                                            UsdGenExecutionResourceAccess::Read, 4);
    auto const* afterCullTopology = findUse(*afterCullTask, UsdGenExecutionDataKind::ElementTopology,
                                            UsdGenExecutionResourceAccess::ReadWrite);
    CHECK(beforeSetTopology && beforeSetTopology->producerTask == 0 &&
          afterCullTopology && afterCullTopology->producerTask == 0 &&
          afterCullTopology->inputValue == 4);

    auto cullThenSet = cullDesc;
    auto setAfterCull = cull;
    setAfterCull.path = SdfPath("/Groom/Ops/set");
    setAfterCull.inputs = {cull.path};
    setAfterCull.params = {{TfToken("length:mode"), VtValue(TfToken("set")), false},
                           {TfToken("length:value"), VtValue(1.25f), false}};
    cullThenSet.nodes.back().inputs = {setAfterCull.path};
    cullThenSet.nodes.insert(cullThenSet.nodes.end() - 1, setAfterCull);
    diagnostics = {};
    auto cullThenSetHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(cullThenSet), &diagnostics);
    CHECK(cullThenSetHandle && !diagnostics.HasErrors());
    auto cullThenSetMetadata = cullThenSetHandle->Metadata();
    auto const* priorCullTask = findTask(cullThenSetMetadata, "/Groom/Ops/cull");
    auto const* afterSetTask = findTask(cullThenSetMetadata, "/Groom/Ops/set");
    CHECK(priorCullTask && afterSetTask);
    auto const* priorCullTopology = findUse(*priorCullTask, UsdGenExecutionDataKind::ElementTopology,
                                            UsdGenExecutionResourceAccess::ReadWrite);
    auto const* afterSetTopology = findUse(*afterSetTask, UsdGenExecutionDataKind::ElementTopology,
                                           UsdGenExecutionResourceAccess::Read, priorCullTopology->outputValue);
    auto const* afterSetGeometry = findUse(*afterSetTask, UsdGenExecutionDataKind::ElementGeometry,
                                           UsdGenExecutionResourceAccess::ReadWrite);
    CHECK(priorCullTopology && afterSetTopology && afterSetTopology->producerTask == priorCullTask->id &&
          afterSetGeometry && afterSetGeometry->producerTask == priorCullTask->id &&
          afterSetGeometry->outputValue != UINT32_MAX);
    {
    auto reparamCullThenSet = cullThenSet;
    for (auto& node : reparamCullThenSet.nodes) if (node.path == setAfterCull.path) {
        node.params.push_back({TfToken("length:method"), VtValue(TfToken("cutExtend")), false});
        node.params.push_back({TfToken("rebuild"), VtValue(TfToken("reparam")), false});
    }
    diagnostics = {};
    auto reparamCullThenSetHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(reparamCullThenSet), &diagnostics);
    CHECK(reparamCullThenSetHandle && !diagnostics.HasErrors());
    auto reparamCullThenSetMetadata = reparamCullThenSetHandle->Metadata();
    auto const* reparamCull = findTask(reparamCullThenSetMetadata, "/Groom/Ops/cull");
    auto const* reparamSet = findTask(reparamCullThenSetMetadata, "/Groom/Ops/set");
    auto const* reparamWidth = findTask(reparamCullThenSetMetadata, "/Groom/Ops/cullWidth");
    CHECK(reparamCull && reparamSet && reparamWidth);
    auto const* reparamNamed = findUse(*reparamSet, UsdGenExecutionDataKind::NamedChannels,
                                       UsdGenExecutionResourceAccess::Read, UINT32_MAX);
    auto const* cullNamedOutput = findUse(*reparamCull,
        UsdGenExecutionDataKind::NamedChannels, UsdGenExecutionResourceAccess::ReadWrite);
    auto const* widthNamed = findUse(*reparamWidth,
        UsdGenExecutionDataKind::NamedChannels, UsdGenExecutionResourceAccess::Read);
    CHECK(reparamSet->kind == UsdGenExecutionTaskKind::Operator &&
          reparamSet->semanticNode != UINT32_MAX && reparamNamed && cullNamedOutput &&
          reparamNamed->producerTask == reparamCull->id &&
          reparamNamed->inputValue == cullNamedOutput->outputValue &&
          reparamNamed->inputValue != 6 && widthNamed &&
          widthNamed->inputValue == reparamNamed->inputValue &&
          widthNamed->producerTask == reparamCull->id);
    }

    // Non-neutral Set controls remain rejected.
    auto badSetRandom = setDesc;
    badSetRandom.nodes[1].params.push_back({TfToken("length:random"), VtValue(usdGen::UsdGenVec2f(1.5f, .5f)), false});
    diagnostics = {};
    auto randomSetHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(badSetRandom), &diagnostics);
    CHECK(randomSetHandle && !diagnostics.HasErrors());
    auto randomSetPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(randomSetHandle->Payload());
    CHECK(randomSetPlan && randomSetPlan->Steps()[0].randomLo == 1.5f &&
          randomSetPlan->Steps()[0].randomHi == .5f && randomSetPlan->Steps()[0].randomSeed == 0);
    for (VtValue invalidRange : {VtValue(usdGen::UsdGenVec2f(-1.f, 1.f)),
                                 VtValue(usdGen::UsdGenVec2f(std::numeric_limits<float>::quiet_NaN(), 1.f)),
                                 VtValue(usdGen::UsdGenVec2f(1.f, std::numeric_limits<float>::infinity()))}) {
        auto invalidRandom = setDesc;
        invalidRandom.nodes[1].params.push_back({TfToken("length:random"), invalidRange, false});
        CHECK(Rejects(invalidRandom));
    }
    auto randomWrongType = setDesc;
    randomWrongType.nodes[1].params.push_back({TfToken("length:random"), VtValue(1.0f), false});
    CHECK(Rejects(randomWrongType));
    auto randomDuplicate = setDesc;
    randomDuplicate.nodes[1].params.push_back({TfToken("length:random"), VtValue(usdGen::UsdGenVec2f(1, 1)), false});
    randomDuplicate.nodes[1].params.push_back({TfToken("length:random"), VtValue(usdGen::UsdGenVec2f(2, 2)), false});
    CHECK(Rejects(randomDuplicate));
    auto animatedRandom = setDesc;
    animatedRandom.nodes[1].params.push_back(
        {TfToken("length:random"), VtValue(usdGen::UsdGenVec2f(1.5f, .5f)), true});
    CHECK(Rejects(animatedRandom));
    auto inventedSeed = setDesc;
    inventedSeed.nodes[1].params.push_back({TfToken("length:seed"), VtValue(7), false});
    CHECK(Rejects(inventedSeed));
    auto setThreshold = setDesc;
    setThreshold.nodes[1].params.push_back({TfToken("cullThreshold"), VtValue(.25f), false});
    diagnostics = {};
    auto setThresholdHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(setThreshold), &diagnostics);
    CHECK(setThresholdHandle && !diagnostics.HasErrors());
    auto setThresholdPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(setThresholdHandle->Payload());
    CHECK(setThresholdPlan && setThresholdPlan->Steps().size() == 3 &&
          setThresholdPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthScale &&
          setThresholdPlan->Steps()[1].kind == VulkanSourceWidthStage::Kind::LengthCull &&
          setThresholdPlan->Steps()[1].lengthMode == VulkanSourceWidthStage::LengthMode::Set &&
          setThresholdPlan->Steps()[1].cullThreshold == .25f);
    auto zeroDoubleThreshold = lengthDesc;
    zeroDoubleThreshold.nodes[1].params.push_back({TfToken("cullThreshold"), VtValue(0.0), false});
    diagnostics = {};
    auto zeroDoubleHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(zeroDoubleThreshold), &diagnostics);
    CHECK(zeroDoubleHandle && !diagnostics.HasErrors());
    auto zeroDoublePlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(zeroDoubleHandle->Payload());
    CHECK(zeroDoublePlan && zeroDoublePlan->Steps().size() == 2);
    for (double threshold : {-1.0, std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
        auto rejected = lengthDesc;
        rejected.nodes[1].params.push_back({TfToken("cullThreshold"), VtValue(threshold), false});
        CHECK(Rejects(rejected));
    }
    auto badSetRemaining = setDesc;
    badSetRemaining.nodes[1].params.push_back({TfToken("minRemainingLength"), VtValue(.25f), false});
    diagnostics = {};
    auto minimumHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(badSetRemaining), &diagnostics);
    CHECK(minimumHandle && !diagnostics.HasErrors());
    auto minimumPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(minimumHandle->Payload());
    CHECK(minimumPlan && minimumPlan->Steps().size() == 2 &&
          minimumPlan->Steps()[0].minRemainingLength == .25f &&
          minimumPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthScale);
    auto minimumMetadata = minimumHandle->Metadata();
    CHECK(minimumMetadata);
    auto const* minimumTask = minimumMetadata->FindSemanticTask(1);
    CHECK(minimumTask && !minimumTask->topologyBarrier);
    auto minimumTopology = findUse(*minimumTask, UsdGenExecutionDataKind::ElementTopology,
                                  UsdGenExecutionResourceAccess::Read);
    CHECK(minimumTopology && minimumTopology->inputValue == 4 &&
          minimumTopology->producerTask == 0);
    auto minimumDoubleZero = setDesc;
    minimumDoubleZero.nodes[1].params.push_back({TfToken("minRemainingLength"), VtValue(0.0), false});
    auto minimumDoubleZeroHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(minimumDoubleZero), &diagnostics);
    CHECK(minimumDoubleZeroHandle);
    auto minimumDoubleZeroPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
        minimumDoubleZeroHandle->Payload());
    CHECK(minimumDoubleZeroPlan && minimumDoubleZeroPlan->Steps().size() == 2 &&
          minimumDoubleZeroPlan->Steps()[0].minRemainingLength == 0);
    auto minimumThreshold = badSetRemaining;
    minimumThreshold.nodes[1].params.push_back({TfToken("cullThreshold"), VtValue(.25f), false});
    diagnostics = {};
    auto minimumThresholdHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(minimumThreshold), &diagnostics);
    CHECK(minimumThresholdHandle && !diagnostics.HasErrors());
    auto minimumThresholdPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(minimumThresholdHandle->Payload());
    CHECK(minimumThresholdPlan && minimumThresholdPlan->Steps().size() == 3 &&
          minimumThresholdPlan->Steps()[0].minRemainingLength == .25f &&
          minimumThresholdPlan->Steps()[1].minRemainingLength == .25f &&
          minimumThresholdPlan->Steps()[1].kind == VulkanSourceWidthStage::Kind::LengthCull);
    auto cullMinimum = cullDesc;
    for (auto& parameter : cullMinimum.nodes[1].params)
        if (parameter.name == TfToken("minRemainingLength")) parameter.value = VtValue(.25f);
    diagnostics = {};
    auto cullMinimumHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(cullMinimum), &diagnostics);
    CHECK(cullMinimumHandle && !diagnostics.HasErrors());
    auto cullMinimumPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(cullMinimumHandle->Payload());
    CHECK(cullMinimumPlan && cullMinimumPlan->Steps().size() == 2 &&
          cullMinimumPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthCull &&
          cullMinimumPlan->Steps()[0].minRemainingLength == .25f);
    auto randomCull = cullDesc;
    for (auto& parameter : randomCull.nodes[1].params)
        if (parameter.name == TfToken("length:random"))
            parameter.value = VtValue(usdGen::UsdGenVec2f(1.5f, .5f));
    auto randomCullHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(randomCull), &diagnostics);
    CHECK(randomCullHandle);
    auto randomCullPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(randomCullHandle->Payload());
    CHECK(randomCullPlan && randomCullPlan->Steps().size() == 2 &&
          randomCullPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthCull);
    auto cullMinimumMetadata = cullMinimumHandle->Metadata();
    CHECK(cullMinimumMetadata);
    auto const* cullMinimumTask = findTask(cullMinimumMetadata, "/Groom/Ops/cull");
    CHECK(cullMinimumTask && cullMinimumTask->topologyBarrier);
    auto cullMinimumTopology = findUse(*cullMinimumTask,
        UsdGenExecutionDataKind::ElementTopology, UsdGenExecutionResourceAccess::ReadWrite);
    CHECK(cullMinimumTopology && cullMinimumTopology->inputValue == 4 &&
          cullMinimumTopology->producerTask == 0);
    for (VtValue invalid : {VtValue(-.25f), VtValue(std::numeric_limits<float>::quiet_NaN()),
                            VtValue(std::numeric_limits<float>::infinity()),
                            VtValue(std::numeric_limits<double>::infinity()),
                            VtValue(std::numeric_limits<double>::max()), VtValue(TfToken("bad"))}) {
        auto invalidMinimum = setDesc;
        invalidMinimum.nodes[1].params.push_back({TfToken("minRemainingLength"), invalid, false});
        CHECK(Rejects(invalidMinimum));
        auto invalidCullMinimum = cullDesc;
        for (auto& parameter : invalidCullMinimum.nodes[1].params)
            if (parameter.name == TfToken("minRemainingLength")) parameter.value = invalid;
        CHECK(Rejects(invalidCullMinimum));
    }
    auto duplicateMinimum = setDesc;
    duplicateMinimum.nodes[1].params.push_back({TfToken("minRemainingLength"), VtValue(.25f), false});
    duplicateMinimum.nodes[1].params.push_back({TfToken("minRemainingLength"), VtValue(.5f), false});
    CHECK(Rejects(duplicateMinimum));
    auto duplicateCullMinimum = cullDesc;
    duplicateCullMinimum.nodes[1].params.push_back(
        {TfToken("minRemainingLength"), VtValue(.25f), false});
    CHECK(Rejects(duplicateCullMinimum));
    auto badSetMap = setDesc;
    badSetMap.nodes[1].maps.push_back(SdfPath("/Groom/Map"));
    CHECK(Rejects(badSetMap));
    auto badCutReparam = cutExtendScale;
    badCutReparam.nodes[1].params.back().value = VtValue(TfToken("reparam"));
    diagnostics = {};
    auto reparamScaleHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(badCutReparam), &diagnostics);
    CHECK(reparamScaleHandle && !diagnostics.HasErrors());
    auto reparamScalePlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(reparamScaleHandle->Payload());
    CHECK(reparamScalePlan && reparamScalePlan->Steps()[0].lengthRebuild ==
          VulkanSourceWidthStage::LengthRebuild::Reparam);
    auto reparamSet = cutExtendSet;
    reparamSet.nodes[1].params.back().value = VtValue(TfToken("reparam"));
    diagnostics = {};
    auto reparamSetHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(reparamSet), &diagnostics);
    CHECK(reparamSetHandle && !diagnostics.HasErrors());
    auto reparamSetPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(reparamSetHandle->Payload());
    CHECK(reparamSetPlan && reparamSetPlan->Steps()[0].lengthMode == VulkanSourceWidthStage::LengthMode::Set &&
          reparamSetPlan->Steps()[0].lengthRebuild == VulkanSourceWidthStage::LengthRebuild::Reparam);
    auto reparamCull = cutExtendCull;
    reparamCull.nodes[1].params.back().value = VtValue(TfToken("reparam"));
    diagnostics = {};
    auto reparamCullHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(reparamCull), &diagnostics);
    CHECK(reparamCullHandle && !diagnostics.HasErrors());
    auto reparamCullPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(reparamCullHandle->Payload());
    CHECK(reparamCullPlan && reparamCullPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthCull &&
          reparamCullPlan->Steps()[0].lengthRebuild == VulkanSourceWidthStage::LengthRebuild::Reparam);
    auto badRebuild = cutExtendScale;
    badRebuild.nodes[1].params.back().value = VtValue(TfToken("unsupported"));
    CHECK(Rejects(badRebuild));
    auto badCutExpression = cutExtendScale;
    badCutExpression.nodes[1].expressionBindings.push_back({});
    CHECK(Rejects(badCutExpression));
    // Isolated EnvelopeV1 compiler coverage. Keep this fixture independent of
    // the established Length/random/minimum descriptors above.
    {
        auto envelopeDesc = MakeDesc();
        UsdGenNodeDesc envelope;
        envelope.path = SdfPath("/Groom/Ops/envelope"); envelope.type = TfToken("UsdGenLength");
        envelope.inputs = {envelopeDesc.nodes[0].path};
        envelope.params = {{TfToken("length:value"), VtValue(2.f), false},
                           {TfToken("mask"), VtValue(.5f), false}};
        envelopeDesc.nodes[1].inputs = {envelope.path};
        envelopeDesc.nodes.insert(envelopeDesc.nodes.begin() + 1, envelope);
        auto envelopeHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(envelopeDesc));
        CHECK(envelopeHandle);
        auto envelopePlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(envelopeHandle->Payload());
        CHECK(envelopePlan && envelopePlan->Steps().size() == 2 &&
              envelopePlan->Steps()[0].lengthBlend == 1.f &&
              envelopePlan->Steps()[0].lengthMaskAmount == .5f &&
              !envelopePlan->Steps()[0].lengthCullOnly);
        auto envelopeThreshold = envelopeDesc;
        envelopeThreshold.nodes[1].params.push_back({TfToken("cullThreshold"), VtValue(3.f), false});
        auto envelopeThresholdHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(envelopeThreshold));
        CHECK(envelopeThresholdHandle);
        auto envelopeThresholdPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(envelopeThresholdHandle->Payload());
        CHECK(envelopeThresholdPlan && envelopeThresholdPlan->Steps().size() == 3 &&
              envelopeThresholdPlan->Steps()[1].cullThreshold == 3.f);
        auto zeroEnvelopeThreshold = envelopeDesc;

        zeroEnvelopeThreshold.nodes[1].params[1].value = VtValue(0.f);
        zeroEnvelopeThreshold.nodes[1].params.push_back({TfToken("cullThreshold"), VtValue(3.f), false});
        auto zeroEnvelopeThresholdHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(zeroEnvelopeThreshold));
        CHECK(zeroEnvelopeThresholdHandle);
        auto zeroEnvelopeThresholdPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
            zeroEnvelopeThresholdHandle->Payload());
        CHECK(zeroEnvelopeThresholdPlan && zeroEnvelopeThresholdPlan->Steps().size() == 3 &&
              zeroEnvelopeThresholdPlan->Steps()[1].cullThreshold == 3.f);
        auto envelopeCull = envelopeDesc;
        envelopeCull.nodes[1].params = {{TfToken("length:mode"), VtValue(TfToken("cull")), false},
                                        {TfToken("mask"), VtValue(.5f), false}};
        auto envelopeCullHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(envelopeCull));
        CHECK(envelopeCullHandle);
        auto envelopeCullMetadata = envelopeCullHandle->Metadata();
        auto envelopeCullPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(envelopeCullHandle->Payload());
        CHECK(envelopeCullPlan && envelopeCullPlan->Steps().size() == 3 &&
              envelopeCullPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthScale &&
              envelopeCullPlan->Steps()[0].lengthCullOnly &&
              envelopeCullPlan->Steps()[1].kind == VulkanSourceWidthStage::Kind::LengthCull &&
              !envelopeCullPlan->Steps()[1].lengthCullOnly);
        CHECK(envelopeCullMetadata && envelopeCullMetadata->Tasks().size() == 5 &&
              envelopeCullMetadata->Tasks()[1].semanticNode == UINT32_MAX &&
              !envelopeCullMetadata->Tasks()[1].topologyBarrier &&
              envelopeCullMetadata->Tasks()[2].semanticNode == 1 &&
              envelopeCullMetadata->Tasks()[2].topologyBarrier);
        auto neutralCull = envelopeCull;
        neutralCull.nodes[1].params[1] = {TfToken("mask"), VtValue(1.f), false};
        auto neutralCullHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(neutralCull));
        CHECK(neutralCullHandle);
        auto neutralCullPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(neutralCullHandle->Payload());
        CHECK(neutralCullPlan && neutralCullPlan->Steps().size() == 2 &&
              neutralCullPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthCull);
        auto rejectEnvelope = [&](UsdGenNodeDesc node) {
            auto rejected = envelopeDesc; rejected.nodes[1] = std::move(node);
            return Rejects(rejected);
        };
        // The removed node-level mute leaves `mask` as the single envelope
        // control: out-of-range values must reject regardless of `length:value`.
        for (VtValue value : {VtValue(-1.f), VtValue(std::numeric_limits<float>::quiet_NaN()),
                              VtValue(std::numeric_limits<float>::infinity()), VtValue(2.f)}) {
            auto badMask = envelope; badMask.params[1].value = value; CHECK(rejectEnvelope(badMask));
        }
        auto wrongMask = envelope; wrongMask.params[1].value = VtValue(TfToken("bad")); CHECK(rejectEnvelope(wrongMask));
        auto animatedMask = envelope; animatedMask.params[1].animated = true; CHECK(rejectEnvelope(animatedMask));
        auto literalBlend = envelope; literalBlend.params.push_back({TfToken("blend"), VtValue(.5f), false}); CHECK(rejectEnvelope(literalBlend));
    }
    auto reversed = MakeDesc();
    std::swap(reversed.nodes[0], reversed.nodes[1]);
    CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(reversed)));
    auto const& memory = metadata->MemoryEstimate();
    // These are checked descriptor byte components, not a claim about native
    // Vulkan allocation/admission before an executor has been registered.
    CHECK(!memory.memoryAvailable && !memory.conservativeUpperBound && !memory.runtimeRefinementAvailable &&
          memory.concurrentPeakBytes > 0 && memory.immutableSharedInputBytes > 0 &&
          !source.estimate.memoryAvailable && !source.estimate.memoryConservativeUpperBound &&
          source.estimate.retainedOutputBytes > 0 && width.estimate.retainedOutputBytes > 0);

    // Width uses the same authored LUT profile and interpolation as CUDA.
    auto profileDesc = MakeDesc();
    profileDesc.nodes[1].params.push_back({TfToken("width:knots"), VtValue(usdGen::UsdGenVec2fArray{}), false});
    profileDesc.nodes[1].params.push_back({TfToken("width:interpolation"), VtValue(TfToken("catmullRom")), false});
    auto profileHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(profileDesc));
    CHECK(profileHandle);
    CHECK(std::static_pointer_cast<const VulkanSourceWidthPlan>(profileHandle->Payload())->Steps().size() == 1);
    for (auto token : {TfToken("taper"), TfToken("taperStart"), TfToken("rootScale"), TfToken("tipScale"), TfToken("label")}) {
        auto bad = MakeDesc();
        bad.nodes[1].params.push_back({token, VtValue(.5f), false});
        CHECK(!CompileVulkanSourceWidthPlan(usdGenUsd::Lower(bad)));
    }
    {
        auto bad = MakeDesc();
        bad.nodes[1].params.push_back({TfToken("width:interpolation"), VtValue(TfToken("linear")), false});
        CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(bad)));
        auto knots = MakeDesc();
        knots.nodes[1].params.push_back({TfToken("width:knots"), VtValue(usdGen::UsdGenVec2fArray{{0.f, 1.f}}), false});
        auto knotHandle=CompileVulkanSourceWidthPlan(usdGenUsd::Lower(knots));
        CHECK(knotHandle);
        auto knotPlan=std::static_pointer_cast<const VulkanSourceWidthPlan>(knotHandle->Payload());
        CHECK(knotPlan->Steps()[0].width.extended && knotPlan->Steps()[0].width.profile[0]==1.0f);
        knots.nodes[1].params.back().value=VtValue(usdGen::UsdGenVec2fArray{{-.1f,1.f}});
        CHECK(!CompileVulkanSourceWidthPlan(usdGenUsd::Lower(knots)));
    }

    auto fanout = MakeFanoutDesc();
    auto fanoutHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(fanout));
    CHECK(fanoutHandle);
    auto fanoutPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(fanoutHandle->Payload());
    CHECK(fanoutPlan && fanoutPlan->Steps().size() == 3 && fanoutPlan->IntermediateCount() == 2);
    CHECK(fanoutPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::Width &&
          fanoutPlan->Steps()[0].input == 0 && fanoutPlan->Steps()[0].key == usdGenUsd::KeyOf(fanout.nodes[2].path));
    CHECK(fanoutPlan->Steps()[1].kind == VulkanSourceWidthStage::Kind::Width &&
          fanoutPlan->Steps()[1].input == 0 && fanoutPlan->Steps()[1].key == usdGenUsd::KeyOf(fanout.nodes[3].path));
    CHECK(fanoutPlan->Steps()[2].kind == VulkanSourceWidthStage::Kind::WidthBlend &&
          fanoutPlan->Steps()[2].input == 2 && fanoutPlan->Steps()[2].rightInput == 1 &&
          fanoutPlan->Steps()[2].blend == .25f);
    auto wrongOrigins = fanout;
    UsdGenNodeDesc changedPoints;
    changedPoints.path = SdfPath("/Groom/Ops/changedPoints"); changedPoints.type = TfToken("UsdGenLength");
    changedPoints.inputs = {wrongOrigins.nodes[0].path};
    changedPoints.params = {{TfToken("length:value"), VtValue(1.0), false}};
    wrongOrigins.nodes[2].inputs = {changedPoints.path};
    wrongOrigins.nodes.push_back(changedPoints);
    auto proofHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(wrongOrigins));
    CHECK(proofHandle);
    auto proofPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(proofHandle->Payload());
    CHECK(proofPlan && proofPlan->Steps().back().requiresNonWidthProof);
    CHECK(!fanoutPlan->Steps().back().requiresNonWidthProof);

    // Vulkan plans preserve the descriptor's complete authored named-channel
    // schema. Cover every admitted scalar type/domain and the complete arity
    // range; source capture performs the corresponding payload validation.
    for (auto type : {UsdGenAuthoredPlaneType::Float32, UsdGenAuthoredPlaneType::Int32}) {
        for (auto domain : {UsdGenAuthoredPlaneDomain::Point,
                            UsdGenAuthoredPlaneDomain::Primitive,
                            UsdGenAuthoredPlaneDomain::Groom}) {
            for (uint8_t arity = 1; arity != 17; ++arity) {
                auto named = MakeDesc();
                std::string name = "authored_" + std::to_string(unsigned(type)) + "_" +
                    std::to_string(unsigned(domain)) + "_" + std::to_string(arity);
                AddAuthoredPlane(&named.curveSets[0], name.c_str(), type, domain, arity);
                CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(named)));
            }
        }
    }
    {
        auto bad = MakeDesc();
        AddAuthoredPlane(&bad.curveSets[0], "badArity", UsdGenAuthoredPlaneType::Float32,
                         UsdGenAuthoredPlaneDomain::Point, 1);
        bad.curveSets[0].authoredPlanes.back().arity = 0;
        CHECK(Rejects(bad));
    }
    {
        auto bad = MakeDesc();
        AddAuthoredPlane(&bad.curveSets[0], "badCount", UsdGenAuthoredPlaneType::Int32,
                         UsdGenAuthoredPlaneDomain::Primitive, 1);
        bad.curveSets[0].authoredPlanes.back().intValues.push_back(9);
        CHECK(Rejects(bad));
    }
    {
        auto bad = MakeDesc();
        AddAuthoredPlane(&bad.curveSets[0], "duplicate", UsdGenAuthoredPlaneType::Float32,
                         UsdGenAuthoredPlaneDomain::Point, 1);
        AddAuthoredPlane(&bad.curveSets[0], "duplicate", UsdGenAuthoredPlaneType::Int32,
                         UsdGenAuthoredPlaneDomain::Primitive, 1);
        CHECK(Rejects(bad));
    }
    {
        auto bad = MakeDesc();
        AddAuthoredPlane(&bad.curveSets[0], "points", UsdGenAuthoredPlaneType::Float32,
                         UsdGenAuthoredPlaneDomain::Point, 1);
        CHECK(Rejects(bad));
    }
    for (char const* reserved : {"width", "sourceRootT"}) {
        auto bad = MakeDesc();
        AddAuthoredPlane(&bad.curveSets[0], reserved, UsdGenAuthoredPlaneType::Float32,
                         UsdGenAuthoredPlaneDomain::Point, 1);
        CHECK(Rejects(bad));
    }
    for (int variant = 0; variant != 7; ++variant) {
        auto bad = MakeDesc();
        AddAuthoredPlane(&bad.curveSets[0], "badSchema", UsdGenAuthoredPlaneType::Float32,
                         UsdGenAuthoredPlaneDomain::Point, 1);
        auto& plane = bad.curveSets[0].authoredPlanes.back();
        if (variant == 0) plane.name = TfToken();
        if (variant == 1) plane.arity = 17;
        if (variant == 2) plane.type = static_cast<UsdGenAuthoredPlaneType>(255);
        if (variant == 3) plane.domain = static_cast<UsdGenAuthoredPlaneDomain>(255);
        if (variant == 4) plane.floatValues[0] = std::numeric_limits<float>::quiet_NaN();
        if (variant == 5) plane.floatValues[0] = std::numeric_limits<float>::infinity();
        if (variant == 6) plane.intValues = {1};
        CHECK(Rejects(bad));
    }

    // Named channels are one immutable source-produced bundle. They increase
    // the source estimate and remain an explicit read at every Width/Length/
    // Blend/publication task rather than becoming an untracked side payload.
    auto namedFanout = MakeFanoutDesc();
    AddAuthoredPlane(&namedFanout.curveSets[0], "vertex16", UsdGenAuthoredPlaneType::Float32,
                     UsdGenAuthoredPlaneDomain::Point, 16);
    AddAuthoredPlane(&namedFanout.curveSets[0], "uniformInt", UsdGenAuthoredPlaneType::Int32,
                     UsdGenAuthoredPlaneDomain::Primitive, 1);
    AddAuthoredPlane(&namedFanout.curveSets[0], "constant4", UsdGenAuthoredPlaneType::Float32,
                     UsdGenAuthoredPlaneDomain::Groom, 4);
    auto namedHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(namedFanout)); CHECK(namedHandle);
    auto namedMetadata = namedHandle->Metadata(); CHECK(namedMetadata);
    CHECK(namedMetadata->Tasks().size() == 5 &&
          namedMetadata->Values().size() == fanoutHandle->Metadata()->Values().size() &&
          namedMetadata->Tasks()[0].estimate.retainedOutputBytes > source.estimate.retainedOutputBytes);
    auto const namedValue = std::find_if(namedMetadata->Values().begin(), namedMetadata->Values().end(),
        [](auto const& value) { return value.resource == UsdGenExecutionDataKind::NamedChannels; });
    CHECK(namedValue != namedMetadata->Values().end() && namedValue->id == 6 &&
          namedValue->producerTask == 0);
    for (auto const& task : namedMetadata->Tasks()) {
        auto const channel = std::find_if(task.resources.begin(), task.resources.end(),
            [](auto const& resource) { return resource.resource == UsdGenExecutionDataKind::NamedChannels; });
        CHECK(channel != task.resources.end());
        if (task.kind == UsdGenExecutionTaskKind::Source) {
            CHECK(channel->producerTask == UINT32_MAX && channel->inputValue == UINT32_MAX &&
                  channel->access == UsdGenExecutionResourceAccess::Write && channel->outputValue == 6);
        } else {
            CHECK(channel->producerTask == 0 && channel->access == UsdGenExecutionResourceAccess::Read && channel->inputValue == 6);
        }
    }

    for (int variant = 0; variant != 10; ++variant) {
        auto bad = MakeDesc();
        if (variant == 0) bad.nodes[1].params.push_back({TfToken("taper"), VtValue(-1.0f), false});
        if (variant == 1) bad.nodes[1].params.push_back({TfToken("taperStart"), VtValue(1.1f), false});
        if (variant == 2) bad.nodes[1].params.push_back({TfToken("tipScale"), VtValue(-1.0f), false});
        if (variant == 3) bad.nodes[1].maps.push_back(SdfPath("/map"));
        if (variant == 4) bad.expressions.push_back({SdfPath("/expression"), "1", {}});
        if (variant == 5) bad.nodes[1].inputs = {SdfPath("/wrong")};
        if (variant == 6) bad.nodes[0].curves.clear();
        if (variant == 7) bad.nodes[1].path = bad.nodes[0].path;
        if (variant == 8) bad.xformMatrix[3][0] = std::numeric_limits<double>::quiet_NaN();
        if (variant == 9) bad.nodes[1].inputs.push_back(SdfPath("/extra"));
        if (variant == 3 || variant==4) {
            // Phase 2 maps: width legacy maps are carried (never sampled),
            // exactly like CUDA ValidateWidth. Rejecting would diverge.
            UsdGenDiagnostics mapDiagnostics;
            auto mapHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(bad), &mapDiagnostics);
            CHECK(mapHandle && !mapDiagnostics.HasErrors());
        } else {
            CHECK(Rejects(bad));
        }
    }
    for (int variant = 0; variant != 4; ++variant) {
        auto bad = MakeDesc();
        if (variant == 0) bad.curveSets[0].curveVertexCounts = {1};
        if (variant == 1) bad.curveSets[0].points[0][0] = std::numeric_limits<float>::quiet_NaN();
        if (variant == 2) bad.curveSets[0].curveId = {41, 41};
        if (variant == 3) bad.curveSets[0].points.pop_back();
        CHECK(Rejects(bad));
    }
    for (int variant = 0; variant != 5; ++variant) {
        auto bad = MakeDesc();
        if (variant == 0) bad.curveSets[0].rest.clear();
        if (variant == 1) bad.curveSets[0].restFromCurrentPoints = true;
        if (variant == 2) bad.nodes[0].params[4].value = VtValue(TfToken("onError"));
        if (variant == 3) bad.nodes[0].params[4].value = VtValue(TfToken("always"));
        if (variant == 4) bad.nodes[0].params[2].value = VtValue(std::string("other-epoch"));
        CHECK(Rejects(bad));
    }
    auto empty = MakeDesc();
    empty.curveSets[0].curveVertexCounts.clear(); empty.curveSets[0].points.clear();
    empty.curveSets[0].rest.clear(); empty.curveSets[0].widths.clear(); empty.curveSets[0].curveId.clear();
    CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(empty)));

    // A resolved single surface plus authored root channels is a structural
    // admission fact.  Compilation does not inspect face/UV geometry.
    auto rooted = MakeRootedDesc();
    auto rootedHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(rooted));
    CHECK(rootedHandle);
    auto rootedPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(rootedHandle->Payload());
    CHECK(rootedPlan && rootedPlan->HasRootBindings());
    auto rootedMetadata = rootedHandle->Metadata();
    CHECK(rootedMetadata && rootedMetadata->Values().size() == 8);
    auto const& rootedSource = rootedMetadata->Tasks()[0];
    CHECK(rootedSource.resources.size() == 6 &&
          rootedSource.resources.back().resource == UsdGenExecutionDataKind::RootBindings &&
          rootedSource.resources.back().access == UsdGenExecutionResourceAccess::Write &&
          rootedSource.resources.back().outputValue == 7);
    CHECK(rootedMetadata->Values()[7].id == 7 &&
          rootedMetadata->Values()[7].resource == UsdGenExecutionDataKind::RootBindings &&
          rootedMetadata->Values()[7].producerTask == 0 &&
          rootedMetadata->Values()[7].storage == UsdGenExecutionValueStorage::JobOwnedImmutable);
    CHECK(rootedMetadata->Tasks().size() == 3);
    for (auto const& task : rootedMetadata->Tasks()) {
        auto root = std::find_if(task.resources.begin(), task.resources.end(),
            [](auto const& resource) { return resource.resource == UsdGenExecutionDataKind::RootBindings; });
        CHECK(root != task.resources.end());
        if (task.kind == UsdGenExecutionTaskKind::Source)
            CHECK(root->access == UsdGenExecutionResourceAccess::Write && root->outputValue == 7);
        else
            CHECK(root->access == UsdGenExecutionResourceAccess::Read && root->inputValue == 7);
    }
    CHECK(rootedMetadata->Tasks()[0].estimate.retainedOutputBytes -
              metadata->Tasks()[0].estimate.retainedOutputBytes == 48);
    CHECK(VtIntArray(UsdGenCurveSetInputView(UsdGenCurveSetInputs(*rootedPlan->Descriptor())[0]).skinPrim.begin(),
                     UsdGenCurveSetInputView(UsdGenCurveSetInputs(*rootedPlan->Descriptor())[0]).skinPrim.end()) ==
          VtIntArray({0}));
    rooted.curveSets[0].skinPrim[0] = 99;
    rooted.curveSets[0].rootFrame[0][0][0] = 2.0;
    CHECK(VtIntArray(rootedPlan->Source().skinPrim.begin(), rootedPlan->Source().skinPrim.end()) ==
              VtIntArray({0}) &&
          rootedPlan->Source().rootFrame[0] == usdGen::UsdGenMatrix4d(1.0));

    // Source and bound surface matrices describe relative coordinates, not
    // a request to bake world-space positions into the captured C3 planes.
    for (int variant = 0; variant != 5; ++variant) {
        auto affine = MakeRootedDesc();
        usdGen::UsdGenMatrix4d matrix(1.0);
        if (variant == 0) { matrix[3][0] = 3; matrix[3][1] = -4; }
        if (variant == 1) { matrix[0][0] = 0; matrix[0][1] = 1;
                            matrix[1][0] = -1; matrix[1][1] = 0; }
        if (variant == 2) { matrix[0][0] = 2; matrix[1][1] = 3; matrix[2][2] = .5; }
        if (variant == 3) { matrix[0][1] = .25; matrix[2][0] = -.5; }
        if (variant == 4) matrix[0][0] = -1;
        affine.curveSets[0].worldMatrix = matrix;
        affine.surfaces[0].worldMatrix = matrix;
        auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(affine)); CHECK(handle);
        auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
        CHECK(plan->Source().worldMatrix == matrix &&
              UsdGenSurfaceInputView(UsdGenSurfaceInputs(*plan->Descriptor())[0]).worldMatrix == matrix);
        CHECK(handle->Metadata()->Tasks()[0].estimate.retainedOutputBytes ==
              rootedMetadata->Tasks()[0].estimate.retainedOutputBytes);
        affine.curveSets[0].worldMatrix[3][0] += 10;
        affine.surfaces[0].worldMatrix[0][0] = 0;
        CHECK(plan->Source().worldMatrix == matrix &&
              UsdGenSurfaceInputView(UsdGenSurfaceInputs(*plan->Descriptor())[0]).worldMatrix == matrix);
    }
    for (int target = 0; target != 2; ++target) {
        for (int variant = 0; variant != 5; ++variant) {
            auto bad = MakeRootedDesc();
            // Without authored frames, source inversion is required to
            // construct surface frames; an authored-frame source never
            // needs it and legitimately accepts a singular affine matrix.
            if (!target) bad.curveSets[0].rootFrame.clear();
            auto& matrix = target ? bad.surfaces[0].worldMatrix : bad.curveSets[0].worldMatrix;
            if (variant == 0) matrix[0][0] = 0;
            if (variant == 1) matrix[0][3] = .1;
            if (variant == 2) matrix[1][1] = std::numeric_limits<double>::infinity();
            if (variant == 3) matrix[2][2] = std::numeric_limits<double>::quiet_NaN();
            if (variant == 4) matrix[0][0] = std::numeric_limits<double>::epsilon() / 2;
            CHECK(Rejects(bad));
        }
    }
    {
        auto unrelated = MakeRootedDesc();
        auto surface = unrelated.surfaces[0];
        surface.path = SdfPath("/Groom/UnusedSurface");
        surface.worldMatrix = usdGen::UsdGenMatrix4d(0.0);
        unrelated.surfaces.push_back(surface);
        CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(unrelated)));
    }

    // Structural relationship/frame/cardinality errors fail closed.  A
    // geometric face/UV error remains an execution-time drop/rebind concern.
    for (int variant = 0; variant != 6; ++variant) {
        auto bad = MakeRootedDesc();
        if (variant == 0) bad.nodes[0].surfaces.push_back(bad.surfaces[0].path);
        if (variant == 1) bad.nodes[0].surfaces[0] = SdfPath("/Groom/Missing");
        if (variant == 2) bad.surfaces.push_back(bad.surfaces[0]);
        if (variant == 3) bad.curveSets[0].rootFrame[0][0][3] = 1.0;
        if (variant == 4) bad.curveSets[0].rootFrame[0][0][0] = 2.0;
        if (variant == 5) bad.curveSets[0].rootFrame.push_back(usdGen::UsdGenMatrix4d(1.0));
        CHECK(Rejects(bad));
    }
    {
        // Row-vector affine frames may carry a translated origin. Only
        // perspective components, not this translation, invalidate them.
        auto translated = MakeRootedDesc();
        translated.curveSets[0].rootFrame[0][3][0] = 1.0;
        CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(translated)));
        auto bad = MakeRootedDesc();
        bad.curveSets[0].skinPrimUv.clear();
        CHECK(Rejects(bad));
        bad.nodes[0].params.back().value = VtValue(TfToken("onError"));
        CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(bad)));
        bad.nodes[0].params.back().value = VtValue(TfToken("always"));
        CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(bad)));
    }
    for (TfToken const& policy : {TfToken("onError"), TfToken("always")}) {
        auto unbound = MakeDesc();
        unbound.nodes[0].params.back().value = VtValue(policy);
        CHECK(Rejects(unbound));
    }
    {
        auto geometric = MakeRootedDesc();
        geometric.surfaces[0].faceVertexIndices = {0, 0, 0};
        geometric.curveSets[0].skinPrimUv[0] = usdGen::UsdGenVec2f(9, 9);
        CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(geometric)));
    }
    CHECK(!std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload())->HasRootBindings());

    // Session injection builds the same routing graph without evaluating
    // geometry. Without the runtime the process-wide Vulkan route stays
    // unavailable; with it, direct compilation validates like CUDA.
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    std::shared_ptr<const UsdGenExecutionPlanHandle> injected;
    auto input = MakeDesc();
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    CHECK(usdGenUsd::Compile(compiler, input, &graph).ok);
#else
    CHECK(!usdGenUsd::Compile(compiler, input, &graph).ok);
#endif
    unsigned calls = 0;
    auto compile = [&](UsdGenGraphSpec const& exact, UsdGenDiagnostics* messages) {
        ++calls;
        return CompileVulkanSourceWidthPlan(exact, messages);
    };
    CHECK(usdGenUsd::CompileInjectedDevice(compiler, input, &graph, compile, &injected).ok);
    CHECK(calls == 1 && injected && graph.NodeCount() == 2);
    CHECK(graph.Spec().executionBackend == usdGenTestUtils::Backend("vulkan"));
    CHECK(usdGenUsd::NodeIdForPath(graph, input.terminal) == 1);
    CHECK(NodeBuffer(graph, 0).totalCvs == 0 && NodeBuffer(graph, 1).totalCvs == 0);
    auto retained = injected;
    auto rejected = input;
    rejected.nodes[1].params.push_back({TfToken("taper"), VtValue(1.1f), false});
    CHECK(!usdGenUsd::CompileInjectedDevice(compiler, rejected, &graph, compile, &injected).ok);
    CHECK(injected == retained && graph.NodeCount() == 2 && graph.Spec().nodes[1].params.size() == 2);
    // Even a faulty provider cannot bypass common operator/graph checks.
    rejected = input;
    rejected.nodes[1].type = TfToken("MissingInjectedKernel");
    auto staleProvider = [retained](auto const&, auto*) { return retained; };
    CHECK(!usdGenUsd::CompileInjectedDevice(compiler, rejected, &graph, staleProvider, &injected).ok);
    CHECK(injected == retained && graph.NodeCount() == 2);
    rejected = input;
    rejected.executionBackend = kUsdGenBackendCpu;
    CHECK(!usdGenUsd::CompileInjectedDevice(compiler, rejected, &graph, compile, &injected).ok);
    CHECK(!usdGenUsd::CompileInjectedDevice(compiler, input, &graph, {}, &injected).ok);
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    CHECK(usdGenUsd::Compile(compiler, input, &graph).ok);
#else
    CHECK(!usdGenUsd::Compile(compiler, input, &graph).ok);
#endif
    CHECK(injected == retained && graph.NodeCount() == 2);
    return 0;
}
