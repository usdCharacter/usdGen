// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenHydraCurves/curveColumns.h"
#include "usdGenUsd/convert/math.h"
#include "usdGen/api/inputDesc.h"
#include "usdGen/api/engineTokens.h"
#include "usdGen/api/ext/tokenScope.h"
#include "usdGen/api/ext/typeRegistry.h"

#include <cstddef>
#include <cstdint>

namespace usdGen {

namespace {

// Written by registration (before the registries freeze); read after it.
struct Names {
    UsdGenToken statics[UsdGenCurveSlots::Count];
    UsdGenToken sourceColor;
    UsdGenToken points, faceVertexCounts, faceVertexIndices;
    UsdGenTypeId curveSet = UsdGenTypeId::Invalid;
    UsdGenTypeId surfaceKind = UsdGenTypeId::Invalid;
    UsdGenInputDesc const *surfaceDesc = nullptr;
    size_t worldMatrix = SIZE_MAX;   // index in a surface record's values
};
Names gNames;

// Values per element of each static channel (packed arity).
constexpr uint32_t kPerItem[UsdGenCurveSlots::Count] = {1, 1, 1, 3, 1, 1, 1, 1, 2, 3, 3, 3};

template <class T>
UsdGenSpan<const T> Col(UsdGenElementSetView const &s, UsdGenSlot slot, uint32_t perItem) noexcept
{
    if (!s.Has(slot)) return {};
    return {s.Column<T>(slot), size_t(s.Values(slot) / perItem)};
}

UsdGenCurvePlane PlaneOf(UsdGenElementSetView const &s, UsdGenSlot slot) noexcept
{
    UsdGenEngineTokens const &tk = UsdGenTokens();
    UsdGenChannelDesc const &d = (*s.channels)[slot];
    UsdGenCurvePlane p;
    p.name = d.name;
    p.interpolation = d.level == 0 ? tk.constant : d.level == 1 ? tk.uniform : tk.vertex;
    p.arity = d.arity;
    if (d.scalar == UsdGenScalar::F32) {
        p.type = tk.float_;
        p.f = Col<float>(s, slot, 1);
    } else if (d.scalar == UsdGenScalar::I32) {
        p.type = tk.int_;
        p.i = Col<int>(s, slot, 1);
    }
    return p;
}

bool Resolve(UsdGenChannelTable const &table, UsdGenCurveSlots *c) noexcept
{
    uint64_t const digest = table.Digest();
    if (c->ok && c->digest == digest) return true;
    c->ok = false;
    c->digest = digest;
    for (uint8_t k = 0; k < UsdGenCurveSlots::Count; ++k) {
        if (gNames.statics[k].IsEmpty()) return false;
        UsdGenSlot const s = table.Find(gNames.statics[k]);
        if (s == kUsdGenNoSlot || s >= table.StaticCount()) return false;
        c->slot[k] = s;
    }
    c->ok = true;
    return true;
}

}  // namespace

UsdGenCurveColumns UsdGenCurveColumns::Of(UsdGenElementSetView const &s, UsdGenCurveSlots *slots) noexcept
{
    UsdGenCurveColumns v;
    if (!s.channels || !s.columns || !s.values) return v;
    UsdGenCurveSlots local;
    UsdGenCurveSlots &c = slots ? *slots : local;
    if (!Resolve(*s.channels, &c)) return v;
    using S = UsdGenCurveSlots;
    v._set = s;
    v.px = Col<float>(s, c.slot[S::Px], kPerItem[S::Px]);
    v.py = Col<float>(s, c.slot[S::Py], kPerItem[S::Py]);
    v.pz = Col<float>(s, c.slot[S::Pz], kPerItem[S::Pz]);
    v.rest = Col<GfVec3f>(s, c.slot[S::Rest], kPerItem[S::Rest]);
    v.width = Col<float>(s, c.slot[S::Width], kPerItem[S::Width]);
    v.hairT = Col<float>(s, c.slot[S::HairT], kPerItem[S::HairT]);
    v.curveId = Col<uint64_t>(s, c.slot[S::CurveId], kPerItem[S::CurveId]);
    v.rootPrim = Col<int>(s, c.slot[S::RootPrim], kPerItem[S::RootPrim]);
    v.rootUV = Col<GfVec2f>(s, c.slot[S::RootUV], kPerItem[S::RootUV]);
    v.rootT = Col<GfVec3f>(s, c.slot[S::RootT], kPerItem[S::RootT]);
    v.rootN = Col<GfVec3f>(s, c.slot[S::RootN], kPerItem[S::RootN]);
    v.rootB = Col<GfVec3f>(s, c.slot[S::RootB], kPerItem[S::RootB]);
    if (s.offsets[2])
        v.cvOffsets = {reinterpret_cast<int const *>(s.offsets[2]), size_t(s.count[1]) + 1};
    v.totalCurves = s.count[1];
    v.totalCvs = s.count[2];
    v.topologyVersion = s.topologyVersion;
    v.valueVersion = s.valueVersion;
    return v;
}

uint32_t UsdGenCurveColumns::ExtraCurveCount() const noexcept
{
    return _set.channels ? uint32_t(_set.channels->DynamicRun(1).size()) : 0u;
}

UsdGenCurvePlane UsdGenCurveColumns::ExtraCurve(uint32_t k) const noexcept
{
    return PlaneOf(_set, _set.channels->DynamicRun(1)[k]);
}

uint32_t UsdGenCurveColumns::ExtraCvCount() const noexcept
{
    return _set.channels ? uint32_t(_set.channels->DynamicRun(2).size()) : 0u;
}

UsdGenCurvePlane UsdGenCurveColumns::ExtraCv(uint32_t k) const noexcept
{
    return PlaneOf(_set, _set.channels->DynamicRun(2)[k]);
}

UsdGenTypeId UsdGenHydraCurveSetType() noexcept
{
    return gNames.curveSet;
}

UsdGenToken const &UsdGenHydraSourceColorPlane() noexcept
{
    return gNames.sourceColor;
}

UsdGenSpan<const UsdGenInputRecord> UsdGenHydraSurfaceInputs(UsdGenGraphSpec const &spec) noexcept
{
    if (gNames.surfaceKind == UsdGenTypeId::Invalid) return {};
    return spec.Inputs(gNames.surfaceKind);
}

UsdGenInputRecord const *UsdGenHydraFindSurface(UsdGenGraphSpec const &spec, UsdGenNodeKey key) noexcept
{
    if (gNames.surfaceKind == UsdGenTypeId::Invalid) return nullptr;
    return spec.FindInput(gNames.surfaceKind, key);
}

uint32_t UsdGenHydraSurfaceId(UsdGenInputRecord const &record) noexcept
{
    UsdGenInputDesc const *const desc = gNames.surfaceDesc;
    return desc && desc->surfaceId ? desc->surfaceId(record) : 0u;
}

UsdGenHydraSurfaceMesh UsdGenHydraSurfaceMeshOf(UsdGenInputRecord const &record) noexcept
{
    UsdGenHydraSurfaceMesh mesh;
    if (record.data) {
        UsdGenChannelTable const &table = record.data->Channels();
        auto column = [&](UsdGenToken const &name) {
            return name.IsEmpty() ? kUsdGenNoSlot : table.Find(name);
        };
        UsdGenSlot const points = column(gNames.points);
        UsdGenSlot const counts = column(gNames.faceVertexCounts);
        UsdGenSlot const indices = column(gNames.faceVertexIndices);
        if (points != kUsdGenNoSlot) mesh.points = record.Column<GfVec3f>(points);
        if (counts != kUsdGenNoSlot) mesh.faceVertexCounts = record.Column<int>(counts);
        if (indices != kUsdGenNoSlot) mesh.faceVertexIndices = record.Column<int>(indices);
    }
    if (gNames.worldMatrix != SIZE_MAX)
        mesh.worldMatrix = usdGenUsd::ToHost(
            record.Get<usdGen::UsdGenMatrix4d>(gNames.worldMatrix, usdGen::UsdGenMatrix4d(1.0)));
    return mesh;
}

bool UsdGenHydraCurvesResolveNames(char const **reason)
{
    UsdGenCompileTokenScope scope;
    static constexpr char const *kStatics[UsdGenCurveSlots::Count] = {
        "__px", "__py", "__pz", "__rest", "__width", "__hairT",
        "__curveId", "__rootPrim", "__rootUV", "__rootT", "__rootN", "__rootB"};
    Names &n = gNames;
    for (uint8_t k = 0; k < UsdGenCurveSlots::Count; ++k)
        n.statics[k] = UsdGenInternToken(kStatics[k]);
    n.sourceColor = UsdGenInternToken("usdGenSourceColor");
    n.points = UsdGenInternToken("points");
    n.faceVertexCounts = UsdGenInternToken("faceVertexCounts");
    n.faceVertexIndices = UsdGenInternToken("faceVertexIndices");
    n.curveSet = UsdGenTypeRegistry::Get().Find(UsdGenInternToken("UsdGenCurveSet"));
    n.surfaceKind = UsdGenInputKindWithRole(UsdGenInputRole::Surface);
    n.surfaceDesc = UsdGenFindInputDesc(n.surfaceKind);
    n.worldMatrix = SIZE_MAX;
    if (n.surfaceDesc) {
        UsdGenToken const worldMatrix = UsdGenInternToken("worldMatrix");
        for (size_t i = 0; i < n.surfaceDesc->attributes.size(); ++i)
            if (n.surfaceDesc->attributes[i] == worldMatrix) n.worldMatrix = i;
    }
    if (n.curveSet == UsdGenTypeId::Invalid) {
        if (reason) *reason = "the type UsdGenCurveSet is not registered (usdGenCurves did not load)";
        return false;
    }
    return true;
}

}  // namespace usdGen
