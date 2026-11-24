// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "growVkPlan.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
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
bool SurfaceValid(UsdGenSurfaceDesc const& surface) {
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
bool ValidateVulkanGrow(UsdGenGraphDesc const& desc, UsdGenNodeDesc const& node,
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
    std::set<TfToken> seen;
    for (auto const& param : node.params) {
        auto const& name = param.name; auto const& v = param.value;
        bool valid = !param.animated && seen.insert(name).second && allowed(name);
        if (name == TfToken("segments")) valid &= v.IsHolding<int>();
        else if (name == TfToken("lengthRandom")) valid &= v.IsHolding<GfVec2f>();
        else if (name == TfToken("directionVector")) valid &= v.IsHolding<GfVec3f>();
        else if (name == TfToken("direction")) valid &= v.IsHolding<TfToken>() || v.IsHolding<std::string>();
        else valid &= Scalar(v);
        if (!valid) return Fail(diag, "unsupported, duplicate or malformed control " + name.GetString());
    }
    UsdGenParamView p{&desc, &node}; CurveGrowPipeline::Controls c;
    int const cv = p.GetInt(TfToken("segments"),8);
    c.length = p.GetDouble(TfToken("length"),1.0); c.seed = node.seed;
    auto const random = p.GetVtValue(TfToken("lengthRandom"),VtValue(GfVec2f(1,1))).UncheckedGet<GfVec2f>();
    c.randomLo = std::min(random[0],random[1]); c.randomHi = std::max(random[0],random[1]);
    double const lift = p.GetDouble(TfToken("lift"),0);
    double const azimuth = p.GetDouble(TfToken("azimuth"),0);
    double const azimuthRandom = p.GetDouble(TfToken("azimuthRandom"),0);
    auto const direction = p.GetToken(TfToken("direction"),TfToken("surfaceNormal"));
    auto const vector = p.GetVtValue(TfToken("directionVector"),VtValue(GfVec3f(0,1,0))).UncheckedGet<GfVec3f>();
    if (cv < 2 || cv > 64 || !std::isfinite(c.length) || c.length < 0 || !std::isfinite(float(c.length)) ||
        !std::isfinite(c.randomLo) || !std::isfinite(c.randomHi) || c.randomLo < 0 || c.randomHi < 0 ||
        !std::isfinite(lift) || lift < -90 || lift > 90 || !std::isfinite(azimuth) || azimuth < -360 || azimuth > 360 ||
        !std::isfinite(azimuthRandom) || azimuthRandom < 0 || azimuthRandom > 1 ||
        (direction != TfToken("surfaceNormal") && direction != TfToken("vector")) ||
        !std::isfinite(vector[0]) || !std::isfinite(vector[1]) || !std::isfinite(vector[2]))
        return Fail(diag, "invalid literal controls");
    c.cvCount = uint32_t(cv); c.lift = float(lift); c.azimuth = float(azimuth); c.azimuthRandom = float(azimuthRandom);
    c.fallbackWidth = desc.defaultWidth;
    c.direction = direction == TfToken("vector") ? CurveGrowPipeline::Direction::Literal : CurveGrowPipeline::Direction::RootNormal;
    for(int k=0;k<3;++k) c.literalDirection[k] = vector[k];
    *output = c; return true;
}
bool NormalizeVulkanGeneratorGraph(UsdGenGraphDesc const& input, UsdGenGraphDesc* output, UsdGenDiagnostics* diag) {
    if (!output) return false;
    UsdGenGraphDesc desc = input;
    usdGenRegisterM1Operators();
    for (auto& node : desc.nodes) {
        if (node.type == TfToken("UsdGenReferenceSource")) {
            if (node.references.size() != 1 || !node.curves.empty() || !node.inputs.empty() ||
                !node.maps.empty() || !node.mapBindings.empty() || !node.surfaces.empty() ||
                !node.params.empty() || !node.ramps.empty() || !node.expressionBindings.empty() || !node.mode.IsEmpty())
                return Fail(diag, "ReferenceSource requires one authored reference");
            auto curves = std::find_if(desc.curveSets.begin(),desc.curveSets.end(),
                [&](auto const& value) { return value.path == node.references.front(); });
            if (curves == desc.curveSets.end() || curves->role != UsdGenRole::Reference)
                return Fail(diag, "unresolved ReferenceSource");
            curves->role = UsdGenRole::Curves; curves->curveRole = TfToken("hair");
            // CUDA ReferenceSource imports the snapshot as current geometry.
            node.type = TfToken("UsdGenCurveSource"); node.curves = node.references; node.references.clear();
            node.params = {{TfToken("useRest"),VtValue(false),false},{TfToken("rebind"),VtValue(TfToken("never")),false}};
            if (curves->rootFrame.empty() && !curves->skinPrim.empty())
                curves->rootFrame = VtMatrix4dArray(curves->curveVertexCounts.size(), Frame({1,0,0},{0,0,-1},{0,1,0}));
        } else if (node.type == TfToken("UsdGenScatter")) {
            if (!node.inputs.empty() || !node.references.empty() || !node.curves.empty() ||
                !node.maps.empty() || !node.mapBindings.empty() || !node.expressionBindings.empty() || !node.ramps.empty() ||
                node.surfaces.size() != 1 || !node.mode.IsEmpty())
                return Fail(diag,"Scatter requires one rest surface and literal density");
            std::set<TfToken> seen;
            for (auto const& p : node.params) {
                bool valid = !p.animated && seen.insert(p.name).second;
                if (p.name == TfToken("density")) valid &= Scalar(p.value);
                else if (p.name == TfToken("flip")) valid &= p.value.IsHolding<bool>() && !p.value.UncheckedGet<bool>();
                else if (p.name == TfToken("subdivisionLevel")) valid &= p.value.IsHolding<int>() && p.value.UncheckedGet<int>() == 0;
                else valid = false;
                if (!valid) return Fail(diag,"unsupported Scatter control " + p.name.GetString());
            }
            auto surface = std::find_if(desc.surfaces.begin(),desc.surfaces.end(),
                [&](auto const& value){return value.path == node.surfaces.front();});
            if (surface == desc.surfaces.end() || !SurfaceValid(*surface)) return Fail(diag,"invalid rest surface");
            auto op = UsdGenOpRegistry::Get().Create(TfToken("UsdGenScatter"));
            UsdGenParamView params{&desc,&node};
            if (!op || !op->Bind(params,diag)) return Fail(diag,"Scatter bind failed");
            auto capture = op->CreateCapture();
            UsdGenCaptureContext ctx; ctx.desc = &desc; ctx.params = &params; ctx.diag = diag;
            ctx.surface = static_cast<UsdGenSurfaceId>(surface - desc.surfaces.begin()); ctx.seed = uint32_t(node.seed);
            UsdGenCurveBuffer empty;
            if (!capture || (node.enabled && !op->Capture(ctx,empty,capture.get(),diag))) return Fail(diag,"Scatter capture failed");
            auto const& roots = capture->Buffer();
            size_t const n = node.enabled ? roots.totalCurves : 0;
            if (n > UINT32_MAX / 2 || (n && (roots.totalCvs != n || roots.curveId.size() != n ||
                roots.rootPrim.size()!=n || roots.rootUV.size()!=n || roots.rootT.size()!=n || roots.rootB.size()!=n || roots.rootN.size()!=n)))
                return Fail(diag,"invalid captured roots");
            UsdGenCurveSetDesc curves; curves.path = node.path.AppendChild(TfToken("__vulkanCapturedRoots"));
            curves.role = UsdGenRole::Curves; curves.curveRole = TfToken("hair");
            curves.curveVertexCounts = VtIntArray(n,2); curves.points.resize(2*n); curves.rest.resize(2*n);
            curves.widths = VtFloatArray(2*n,desc.defaultWidth); curves.curveId = n ? roots.curveId : VtArray<uint64_t>();
            curves.skinPrim = n ? roots.rootPrim : VtIntArray(); curves.skinPrimUv = n ? roots.rootUV : VtVec2fArray(); curves.rootFrame.resize(n);
            for (size_t i=0;i<n;++i) {
                auto const root = GfVec3f(roots.px[i],roots.py[i],roots.pz[i]);
                curves.points[2*i] = curves.points[2*i+1] = curves.rest[2*i] = curves.rest[2*i+1] = root;
                curves.rootFrame[i] = Frame(roots.rootT[i],roots.rootB[i],roots.rootN[i]);
            }
            node.type = TfToken("UsdGenCurveSource"); node.enabled = true; node.curves = {curves.path}; node.surfaces.clear();
            node.params = {{TfToken("useRest"),VtValue(true),false},{TfToken("rebind"),VtValue(TfToken("never")),false}};
            desc.curveSets.push_back(std::move(curves));
        }
    }
    *output = std::move(desc); return true;
}
bool IsVulkanCapturedScatter(UsdGenCurveSetDesc const& source) noexcept {
    return source.path.GetNameToken() == TfToken("__vulkanCapturedRoots");
}
bool CaptureVulkanScatterSource(UsdGenCurveSetDesc const& source, uint64_t topology,
    uint64_t value, UsdGenCurveBuffer* output) {
    if (!output || !IsVulkanCapturedScatter(source)) return false;
    UsdGenCurveBuffer b; b.totalCurves = uint32_t(source.curveVertexCounts.size());
    b.totalCvs = uint32_t(source.points.size()); b.topologyVersion = topology; b.valueVersion = value;
    b.px.resize(b.totalCvs); b.py.resize(b.totalCvs); b.pz.resize(b.totalCvs); b.hairT.resize(b.totalCvs);
    for (uint32_t i=0;i<b.totalCvs;++i) {
        b.px[i]=source.points[i][0]; b.py[i]=source.points[i][1]; b.pz[i]=source.points[i][2]; b.hairT[i]=float(i%2);
    }
    b.rest=source.rest; b.width=source.widths; b.curveId=source.curveId;
    b.rootPrim=source.skinPrim; b.rootUV=source.skinPrimUv;
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
