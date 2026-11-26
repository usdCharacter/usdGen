// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "growVkPlan.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGen/op.h"
#include "usdGenCurves/types/curveCapture.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/vulkan/vkStages.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
namespace usdGen::vulkan {
namespace {
bool Fail(UsdGenDiagnostics* diag, std::string const& text) {
    if (diag) diag->Error("Vulkan Grow: " + text);
    return false;
}
bool Scalar(VtValue const& v) { return v.IsHolding<float>() || v.IsHolding<double>(); }
GfMatrix4d Frame(GfVec3f t, GfVec3f b, GfVec3f n) {
    GfMatrix4d frame(1.0);
    frame.SetRow(0, GfVec4d(t[0],t[1],t[2],0));
    frame.SetRow(1, GfVec4d(b[0],b[1],b[2],0));
    frame.SetRow(2, GfVec4d(n[0],n[1],n[2],0));
    return frame;
}
bool SurfaceValid(UsdGenSurfaceInputView const& surface) {
    if (!surface.uv.empty() && surface.uv.size() != surface.restPoints.size()) return false;
    size_t corners = 0;
    for (int count : surface.faceVertexCounts) {
        if (count < 3 || size_t(count) > SIZE_MAX - corners) return false;
        corners += size_t(count);
    }
    if (corners != surface.faceVertexIndices.size()) return false;
    for (int index : surface.faceVertexIndices)
        if (index < 0 || size_t(index) >= surface.restPoints.size()) return false;
    for (auto const& p : surface.restPoints) for (int k=0;k<3;++k) if (!std::isfinite(p[k])) return false;
    for (auto const& uv : surface.uv) for (int k=0;k<2;++k) if (!std::isfinite(uv[k])) return false;
    std::set<int> faces;
    for (int face : surface.subsetFaces)
        if (face < 0 || size_t(face) >= surface.faceVertexCounts.size() || !faces.insert(face).second) return false;
    return !surface.restFromCurrentPoints;
}
} // namespace
bool ValidateVulkanGrow(UsdGenGraphSpec const& desc, UsdGenOpNode const& node,
    CurveGrowPipeline::Controls* output, UsdGenDiagnostics* diag) {
    if (!output || !node.mode.IsEmpty() || node.inputs.size() != 1 || !node.references.empty() ||
        !node.curves.empty() || !node.expressionBindings.empty() || !node.ramps.empty())
        return Fail(diag, "unsupported inputs, expressions, ramps or mode");
    static constexpr char const* kAllowed[] = {"segments", "length", "lengthRandom", "lift",
        "azimuth", "azimuthRandom", "direction", "directionVector"};
    auto const allowed = [](TfToken const& name) {
        for (char const* a : kAllowed) if (name.GetString() == a) return true;
        return false;
    };
    using namespace vulkanP;
    UsdGenVulkanKernel const grow = UsdGenVulkanKernelFor(UsdGenVulkanStage::Grow);
    std::set<TfToken> seen;
    for (auto const& param : node.params) {
        auto const& name = param.name; auto const& v = param.value;
        bool valid = !param.animated && seen.insert(name).second && allowed(name);
        if (name == grow.Param(kGrowSegments)) valid &= v.IsHolding<int>();
        else if (name == grow.Param(kGrowLengthRandom)) valid &= v.IsHolding<GfVec2f>();
        else if (name == grow.Param(kGrowDirectionVector)) valid &= v.IsHolding<GfVec3f>();
        else if (name == grow.Param(kGrowDirection)) valid &= v.IsHolding<TfToken>() || v.IsHolding<std::string>();
        else valid &= Scalar(v);
        if (!valid) return Fail(diag, "unsupported, duplicate or malformed control " + name.GetString());
    }
    UsdGenVulkanTokens const& vk = UsdGenVkTokens();
    UsdGenStackParams const resolved(desc, node, *grow.record);
    UsdGenParamView const& p = resolved.View(); CurveGrowPipeline::Controls c;
    int const cv = p.GetInt(grow.Slot(kGrowSegments),8);
    c.length = p.GetDouble(grow.Slot(kGrowLength),1.0); c.seed = node.seed;
    VtValue const randomFallback(GfVec2f(1,1)), vectorFallback(GfVec3f(0,1,0));
    auto const random = p.GetValue(grow.Slot(kGrowLengthRandom),randomFallback).UncheckedGet<GfVec2f>();
    c.randomLo = std::min(random[0],random[1]); c.randomHi = std::max(random[0],random[1]);
    double const lift = p.GetDouble(grow.Slot(kGrowLift),0);
    double const azimuth = p.GetDouble(grow.Slot(kGrowAzimuth),0);
    double const azimuthRandom = p.GetDouble(grow.Slot(kGrowAzimuthRandom),0);
    TfToken const direction = p.GetToken(grow.Slot(kGrowDirection),vk.surfaceNormal);
    auto const vector = p.GetValue(grow.Slot(kGrowDirectionVector),vectorFallback).UncheckedGet<GfVec3f>();
    if (cv < 2 || cv > 64 || !std::isfinite(c.length) || c.length < 0 || !std::isfinite(float(c.length)) ||
        !std::isfinite(c.randomLo) || !std::isfinite(c.randomHi) || c.randomLo < 0 || c.randomHi < 0 ||
        !std::isfinite(lift) || lift < -90 || lift > 90 || !std::isfinite(azimuth) || azimuth < -360 || azimuth > 360 ||
        !std::isfinite(azimuthRandom) || azimuthRandom < 0 || azimuthRandom > 1 ||
        (direction != vk.surfaceNormal && direction != vk.vector) ||
        !std::isfinite(vector[0]) || !std::isfinite(vector[1]) || !std::isfinite(vector[2]))
        return Fail(diag, "invalid literal controls");
    c.cvCount = uint32_t(cv); c.lift = float(lift); c.azimuth = float(azimuth); c.azimuthRandom = float(azimuthRandom);
    c.fallbackWidth = desc.defaultWidth;
    c.direction = direction == vk.vector ? CurveGrowPipeline::Direction::Literal : CurveGrowPipeline::Direction::RootNormal;
    for(int k=0;k<3;++k) c.literalDirection[k] = vector[k];
    *output = c; return true;
}
bool NormalizeVulkanGeneratorGraph(UsdGenGraphSpec const& input, UsdGenGraphSpec* output, UsdGenDiagnostics* diag) {
    if (!output) return false;
    UsdGenGraphSpec desc = input;
    UsdGenEngineTokens const& et = UsdGenTokens();
    UsdGenVulkanTokens const& vk = UsdGenVkTokens();
    // Lowered sources become the op whose Vulkan stage is CurveSource.
    UsdGenVulkanKernel const source = UsdGenVulkanKernelFor(UsdGenVulkanStage::CurveSource);
    auto const lowerToSource = [&](UsdGenOpNode& node, bool useRest) {
        node.type = source.record->Name(); node.opId = source.record->Id();
        node.params = {{source.Param(vulkanP::kCsUseRest),VtValue(useRest),false},
                       {source.Param(vulkanP::kCsRebind),VtValue(vk.never),false}};
    };
    for (auto& node : desc.nodes) {
        UsdGenVulkanKernel const kernel = UsdGenVulkanKernelOf(node.type);
        if (kernel.Is(UsdGenVulkanStage::ReferenceSource)) {
            if (node.references.size() != 1 || !node.curves.empty() || !node.inputs.empty() ||
                !node.maps.empty() || !node.mapBindings.empty() || !node.surfaces.empty() ||
                !node.params.empty() || !node.ramps.empty() || !node.expressionBindings.empty() || !node.mode.IsEmpty())
                return Fail(diag, "ReferenceSource requires one authored reference");
            UsdGenInputRecord* record = nullptr;
            for (UsdGenInputRecord& candidate : desc.MutableInputs(UsdGenCurveSetInputKind()))
                if (candidate.key == node.references.front()) { record = &candidate; break; }
            if (!record || UsdGenCurveSetInputView::Of(*record).role != UsdGenLane::Reference)
                return Fail(diag, "unresolved ReferenceSource");
            // The record is immutable: lower a copy and repack it.
            UsdGenCurveSetData curves;
            UsdGenCopyCurveSetData(UsdGenCurveSetInputView::Of(*record), &curves);
            curves.role = UsdGenLane::Chunked; curves.curveRole = et.hair;
            // CUDA ReferenceSource imports the snapshot as current geometry.
            node.curves = node.references; node.references.clear();
            lowerToSource(node, false);
            if (curves.rootFrame.empty() && !curves.skinPrim.empty())
                curves.rootFrame = VtMatrix4dArray(curves.curveVertexCounts.size(), Frame({1,0,0},{0,0,-1},{0,1,0}));
            *record = UsdGenCurveSetRecord(curves, record->key, record->label);
        } else if (kernel.Is(UsdGenVulkanStage::Scatter)) {
            if (!node.inputs.empty() || !node.references.empty() || !node.curves.empty() ||
                !node.maps.empty() || !node.mapBindings.empty() || !node.expressionBindings.empty() || !node.ramps.empty() ||
                node.surfaces.size() != 1 || !node.mode.IsEmpty())
                return Fail(diag,"Scatter requires one rest surface and literal density");
            std::set<TfToken> seen;
            for (auto const& p : node.params) {
                bool valid = !p.animated && seen.insert(p.name).second;
                if (p.name == kernel.Param(vulkanP::kScDensity)) valid &= Scalar(p.value);
                else if (p.name == kernel.Param(vulkanP::kScFlip)) valid &= p.value.IsHolding<bool>() && !p.value.UncheckedGet<bool>();
                else if (p.name == kernel.Param(vulkanP::kScSubdivisionLevel)) valid &= p.value.IsHolding<int>() && p.value.UncheckedGet<int>() == 0;
                else valid = false;
                if (!valid) return Fail(diag,"unsupported Scatter control " + p.name.GetString());
            }
            UsdGenSpan<const UsdGenInputRecord> const surfaces = UsdGenSurfaceInputs(desc);
            auto surface = std::find_if(surfaces.begin(),surfaces.end(),
                [&](UsdGenInputRecord const& value){return value.key == node.surfaces.front();});
            if (surface == surfaces.end() || !SurfaceValid(UsdGenSurfaceInputView::Of(*surface)))
                return Fail(diag,"invalid rest surface");
            auto op = UsdGenOpRegistry::Get().Create(kernel.record->Id());
            if (!op) return Fail(diag,"Scatter bind failed");
            // Slot-resolved view: the kernel reads its parameters by slot.
            UsdGenStackParams const resolved(desc, node, *kernel.record);
            UsdGenParamView const &params = resolved.View();
            if (!op->Bind(params,diag)) return Fail(diag,"Scatter bind failed");
            auto capture = op->CreateCapture();
            UsdGenCaptureContext ctx; ctx.desc = &desc; ctx.params = &params; ctx.diag = diag;
            ctx.surface = static_cast<UsdGenSurfaceId>(surface - surfaces.begin()); ctx.seed = uint32_t(node.seed);
            // A source: no input.
            if (!capture || (node.enabled && !op->Capture(ctx,UsdGenOpInputs{},capture.get(),diag))) return Fail(diag,"Scatter capture failed");
            auto const& roots = static_cast<UsdGenCurveCapture const &>(*capture).Buffer();
            size_t const n = node.enabled ? roots.totalCurves : 0;
            if (n > UINT32_MAX / 2 || (n && (roots.totalCvs != n || roots.curveId.size() != n ||
                roots.rootPrim.size()!=n || roots.rootUV.size()!=n || roots.rootT.size()!=n || roots.rootB.size()!=n || roots.rootN.size()!=n)))
                return Fail(diag,"invalid captured roots");
            UsdGenNodeKey const curvesKey = desc.DeriveKey(node.key, vk.capturedRoots);
            UsdGenCurveSetData curves;
            curves.role = UsdGenLane::Chunked; curves.curveRole = et.hair;
            curves.curveVertexCounts = VtIntArray(n,2); curves.points.resize(2*n); curves.rest.resize(2*n);
            curves.widths = VtFloatArray(2*n,desc.defaultWidth); curves.curveId = n ? roots.curveId : VtArray<uint64_t>();
            curves.skinPrim = n ? roots.rootPrim : VtIntArray(); curves.skinPrimUv = n ? roots.rootUV : VtVec2fArray(); curves.rootFrame.resize(n);
            for (size_t i=0;i<n;++i) {
                auto const root = GfVec3f(roots.px[i],roots.py[i],roots.pz[i]);
                curves.points[2*i] = curves.points[2*i+1] = curves.rest[2*i] = curves.rest[2*i+1] = root;
                curves.rootFrame[i] = Frame(roots.rootT[i],roots.rootB[i],roots.rootN[i]);
            }
            node.enabled = true; node.curves = {curvesKey}; node.surfaces.clear();
            lowerToSource(node, true);
            desc.inputs.push_back(UsdGenCurveSetRecord(curves, curvesKey, TfToken()));
            desc.SortInputs();
        }
    }
    *output = std::move(desc); return true;
}
bool IsVulkanCapturedScatter(UsdGenGraphSpec const& desc, UsdGenCurveSetInputView const& source) noexcept {
    return desc.DerivedName(source.key) == UsdGenVkTokens().capturedRoots;
}
bool CaptureVulkanScatterSource(UsdGenGraphSpec const& desc, UsdGenCurveSetInputView const& source,
    uint64_t topology, uint64_t value, UsdGenCurveBuffer* output) {
    if (!output || !IsVulkanCapturedScatter(desc, source)) return false;
    UsdGenCurveBuffer b; b.totalCurves = uint32_t(source.curveVertexCounts.size());
    b.totalCvs = uint32_t(source.points.size()); b.topologyVersion = topology; b.valueVersion = value;
    b.px.resize(b.totalCvs); b.py.resize(b.totalCvs); b.pz.resize(b.totalCvs); b.hairT.resize(b.totalCvs);
    for (uint32_t i=0;i<b.totalCvs;++i) {
        b.px[i]=source.points[i][0]; b.py[i]=source.points[i][1]; b.pz[i]=source.points[i][2]; b.hairT[i]=float(i%2);
    }
    b.rest.assign(source.rest.begin(), source.rest.end());
    b.width.assign(source.widths.begin(), source.widths.end());
    b.curveId.assign(source.curveId.begin(), source.curveId.end());
    b.rootPrim.assign(source.skinPrim.begin(), source.skinPrim.end());
    b.rootUV.assign(source.skinPrimUv.begin(), source.skinPrimUv.end());
    b.rootT.resize(b.totalCurves); b.rootB.resize(b.totalCurves); b.rootN.resize(b.totalCurves);
    for(uint32_t i=0;i<b.totalCurves;++i) {
        b.rootT[i]=GfVec3f(source.rootFrame[i].GetRow3(0));
        b.rootB[i]=GfVec3f(source.rootFrame[i].GetRow3(1));
        b.rootN[i]=GfVec3f(source.rootFrame[i].GetRow3(2));
    }
    for(uint32_t i=0;i<b.totalCurves;i+=512) {
        UsdGenChunkDesc chunk; chunk.firstCurve=i; chunk.curveCount=chunk.liveCount=std::min(512u,b.totalCurves-i);
        chunk.firstCv=2*i; chunk.cvCount=2; b.chunks.push_back(chunk);
    }
    *output=std::move(b); return true;
}
void CompleteVulkanGrowSource(UsdGenCurveBuffer* b) {
    if (!b) return;
    size_t const n=b->totalCurves;
    if (b->rootPrim.empty()) b->rootPrim=VtIntArray(n,-1);
    if (b->rootUV.empty()) b->rootUV=VtVec2fArray(n,GfVec2f(0));
    if (b->rootT.empty()) b->rootT=VtVec3fArray(n,GfVec3f(1,0,0));
    if (b->rootB.empty()) b->rootB=VtVec3fArray(n,GfVec3f(0,0,-1));
    if (b->rootN.empty()) b->rootN=VtVec3fArray(n,GfVec3f(0,1,0));
}
} // namespace usdGen::vulkan
