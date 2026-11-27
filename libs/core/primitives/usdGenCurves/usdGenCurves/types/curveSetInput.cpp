// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGenCurves/types/curveTokens.h"
#include "usdGenCurves/types/curveBuffer.h"
#include "usdGenCurves/types/curveSet.h"

#include "usdGen/compile/graph.h"   // kUsdGenMaxExtraPlaneSlots
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/token.h"
#include "usdGen/api/cowArray.h"
#include "usdGenMath/math.h"
#include "usdGen/api/value.h"
#include "usdGen/api/token.h"
#include "usdGen/api/span.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace usdGen {

namespace {

/// One per process, published by the registration (never freed: the type
/// registry points at it).
struct CurveInputState {
    UsdGenChannelDesc channels[kUsdGenCurveInputSlots];
    UsdGenInputSlot slots[kUsdGenCurveInputSlots];
    UsdGenToken attributes[kUsdGenCurveValues];
    UsdGenElementLayout layout;
    UsdGenInputDesc input;
    UsdGenTypeId kind = UsdGenTypeId::Invalid;
};

std::atomic<CurveInputState *> gCurveInput{nullptr};

CurveInputState const *State() noexcept
{
    if (CurveInputState const *s = gCurveInput.load(std::memory_order_acquire)) return s;
    (void)UsdGenOpRegistry::Get();   // its constructor registers the built-ins
    return gCurveInput.load(std::memory_order_acquire);
}

template <class T>
UsdGenSpan<const T> Arr(UsdGenCowArray<T> const &a) noexcept
{
    return {a.cdata(), a.size()};
}

constexpr size_t kPlaneInfo = 5;   // type, domain, arity, floats, ints
UsdGenMatrix4d const kIdentity(1.0);

bool MaterializeReference(UsdGenInputRecord const &record,
                          UsdGenRef<const UsdGenElementSet> *out, std::string *error);

/// Authored-plane checks (moved from the compiler).
bool ValidateRecord(UsdGenGraphSpec const &desc, UsdGenInputRecord const &record,
                    std::vector<std::string> *errors)
{
    constexpr char const *kReservedNames[] = {
        "points", "rest", "widths", "hairT", "curveOffsets", "curveId",
        "skinprim", "skinprimuv", "displayColor"};
    auto reserved = [&](UsdGenToken const &name) {
        for (char const *r : kReservedNames)
            if (std::strcmp(name.GetText(), r) == 0) return true;
        return false;
    };
    bool ok = true;
    UsdGenCurveSetInputView const curveSet = UsdGenCurveSetInputView::Of(record);
    size_t pointCount = 0;
    for (int count : curveSet.curveVertexCounts) {
        if (count < 0 || pointCount > std::numeric_limits<size_t>::max() -
                              static_cast<size_t>(count)) {
            errors->push_back("UsdGenCompiler: curve set '" +
                desc.Label(curveSet.key) + "' has invalid topology for authored planes");
            ok = false;
            pointCount = 0;
            break;
        }
        pointCount += static_cast<size_t>(count);
    }
    std::set<UsdGenToken> names;
    for (UsdGenAuthoredPlaneView const &plane : curveSet.authoredPlanes) {
        bool const validName = !plane.name.IsEmpty() && names.insert(plane.name).second;
        if (!validName) {
            errors->push_back("UsdGenCompiler: curve set '" +
                desc.Label(curveSet.key) + "' has an empty or duplicate authored plane '" +
                plane.name.GetString() + "'");
            ok = false;
            continue;
        }
        if (reserved(plane.name)) {
            errors->push_back("UsdGenCompiler: authored plane '" +
                plane.name.GetString() + "' on curve set '" +
                desc.Label(curveSet.key) +
                "' collides with a reserved C3 channel");
            ok = false;
            continue;
        }
        if (plane.arity < 1 || plane.arity > kUsdGenMaxExtraPlaneSlots) {
            errors->push_back("UsdGenCompiler: authored plane '" +
                plane.name.GetString() + "' on curve set '" + desc.Label(curveSet.key) +
                "' has arity outside [1,16]");
            ok = false;
            continue;
        }
        size_t elements = 0;
        switch (plane.domain) {
        case UsdGenAuthoredPlaneDomain::Point: elements = pointCount; break;
        case UsdGenAuthoredPlaneDomain::Primitive:
            elements = curveSet.curveVertexCounts.size(); break;
        case UsdGenAuthoredPlaneDomain::Groom: elements = 1; break;
        default:
            errors->push_back("UsdGenCompiler: authored plane '" +
                plane.name.GetString() + "' has an invalid domain");
            ok = false;
            continue;
        }
        if (elements > std::numeric_limits<size_t>::max() / plane.arity) {
            errors->push_back("UsdGenCompiler: authored plane '" +
                plane.name.GetString() + "' value count overflows this platform");
            ok = false;
            continue;
        }
        size_t const expected = elements * plane.arity;
        bool payload = false;
        switch (plane.type) {
        case UsdGenAuthoredPlaneType::Float32:
            payload = plane.floatValues.size() == expected && plane.intValues.empty();
            break;
        case UsdGenAuthoredPlaneType::Int32:
            payload = plane.intValues.size() == expected && plane.floatValues.empty();
            break;
        default: break;
        }
        if (!payload) {
            errors->push_back("UsdGenCompiler: authored plane '" +
                plane.name.GetString() + "' has invalid type or cardinality");
            ok = false;
        }
    }
    return ok;
}

void MixMatrix(UsdGenInputDigest &d, int lane, UsdGenMatrix4d const &m)
{
    for (int row = 0; row != 4; ++row) for (int col = 0; col != 4; ++col) {
        uint64_t bits = 0;
        double const component = m[row][col];
        std::memcpy(&bits, &component, sizeof(bits));
        d.Mix(lane, bits);
    }
}

/// Execution cache key terms (moved from the session cooker).
void DigestRecord(UsdGenInputRecord const &record, UsdGenInputDigest &d, bool layout)
{
    UsdGenCurveSetInputView const curve = UsdGenCurveSetInputView::Of(record);
    if (layout) {
        d.Mix(0, curve.key.value);
        d.Mix(0, curve.curveVertexCounts.size());
        d.Mix(0, curve.points.size());
        return;
    }
    d.Mix(0, curve.key.value);
    d.Mix(1, static_cast<uint64_t>(curve.role));
    d.MixText(0, curve.curveRole.GetString());
    d.MixText(1, curve.type.GetString());
    d.MixText(0, curve.basis.GetString());
    d.MixText(1, curve.wrap.GetString());
    d.MixText(0, curve.widthsInterpolation.GetString());
    d.Mix(1, static_cast<uint64_t>(curve.restFromCurrentPoints));
    d.MixSpan(0, curve.curveVertexCounts);
    d.MixSpan(1, curve.points);
    d.MixSpan(0, curve.rest);
    MixMatrix(d, 1, curve.worldMatrix);
    d.MixSpan(0, curve.widths);
    d.MixSpan(1, curve.skinPrim);
    d.MixSpan(0, curve.curveId);
    d.MixSpan(1, curve.skinPrimUv);
    d.MixSpan(0, curve.rootFrame);
    d.MixText(0, curve.frozenEpoch);
    d.Mix(1, curve.curveGeneration);
    d.Mix(1, static_cast<uint64_t>(curve.authoredPlanes.size()));
    for (UsdGenAuthoredPlaneView const &plane : curve.authoredPlanes) {
        d.MixText(0, plane.name.GetString());
        d.Mix(1, static_cast<uint64_t>(plane.type));
        d.Mix(0, static_cast<uint64_t>(plane.domain));
        d.Mix(1, plane.arity);
        d.MixSpan(0, plane.floatValues);
        d.MixSpan(1, plane.intValues);
    }
}

bool RecordLane(UsdGenInputRecord const &record, UsdGenLane *out)
{
    *out = UsdGenCurveSetInputView::Of(record).role;
    return true;
}

/// The authored plane names, as many as UsdGenAuthoredPlanes::Of reads.
UsdGenSpan<const UsdGenToken> AuthoredChannels(UsdGenInputRecord const &record)
{
    UsdGenValue const *names = record.Value(kUsdGenCurveValuePlaneNames);
    if (!names || !names->IsHolding<UsdGenTokenArray>()) return {};
    UsdGenTokenArray const &array = names->UncheckedGet<UsdGenTokenArray>();
    size_t const count =
        std::min(array.size(), record.Column<int>(kUsdGenCurveInputPlaneInfo).size() / kPlaneInfo);
    return {array.cdata(), count};
}

}  // namespace

UsdGenTypeId UsdGenRegisterCurveSetInputKind(UsdGenRegistrar &r)
{
    if (CurveInputState const *s = gCurveInput.load(std::memory_order_acquire)) return s->kind;
    CurveInputState *fresh = new CurveInputState;   // startup; never freed
    {
        UsdGenCompileTokenScope scope;
        using S = UsdGenScalar;
        struct Row { UsdGenCurveInputSlot slot; char const *channel, *attribute; S scalar; uint8_t arity; uint8_t flags; };
        static constexpr Row kRows[] = {
            {kUsdGenCurveInputPoints, "points", "points", S::F32, 3, kUsdGenInputRequired},
            {kUsdGenCurveInputCounts, "curveVertexCounts", "curveVertexCounts", S::I32, 1,
             kUsdGenInputRequired | kUsdGenInputCounts},
            {kUsdGenCurveInputRest, "rest", "rest", S::F32, 3, kUsdGenInputPrimvar},
            {kUsdGenCurveInputWidths, "widths", "widths", S::F32, 1, 0},
            {kUsdGenCurveInputSkinPrim, "skinPrim", "skinprim", S::I32, 1, kUsdGenInputPrimvar},
            {kUsdGenCurveInputCurveId, "curveId", "usdGen:curveId", S::U64, 1, kUsdGenInputPrimvar},
            {kUsdGenCurveInputSkinPrimUv, "skinPrimUv", "skinprimuv", S::F32, 2, kUsdGenInputPrimvar},
            {kUsdGenCurveInputRootFrame, "rootFrame", "usdGen:rootFrame", S::F64, 16, kUsdGenInputPrimvar},
            {kUsdGenCurveInputPlaneInfo, "authoredPlanes:info", "usdGen:authoredPlanes:info", S::I32, 5, 0},
            {kUsdGenCurveInputPlaneFloats, "authoredPlanes:floats", "usdGen:authoredPlanes:floats", S::F32, 1, 0},
            {kUsdGenCurveInputPlaneInts, "authoredPlanes:ints", "usdGen:authoredPlanes:ints", S::I32, 1, 0},
            {kUsdGenCurveInputCageOwnerIds, "cage:ownerIds", "usdGen:surfaceCage:ownerIds", S::I32, 1, 0},
            {kUsdGenCurveInputCageOwnerDensities, "cage:ownerDensities", "usdGen:surfaceCage:ownerDensities", S::F32, 1, 0},
            {kUsdGenCurveInputCageOwnerSeeds, "cage:ownerSeeds", "usdGen:surfaceCage:ownerSeeds", S::I32, 1, 0},
            {kUsdGenCurveInputCageOwnerCvCounts, "cage:ownerCvCounts", "usdGen:surfaceCage:ownerCvCounts", S::I32, 1, 0},
            {kUsdGenCurveInputCageOwnerEdgeBias, "cage:ownerEdgeBias", "usdGen:surfaceCage:ownerEdgeBias", S::F32, 1, 0},
            {kUsdGenCurveInputCageOwnerLengthProfileOffsets, "cage:ownerLengthProfileOffsets",
             "usdGen:surfaceCage:ownerLengthProfileOffsets", S::I32, 1, 0},
            {kUsdGenCurveInputCageOwnerLengthProfile, "cage:ownerLengthProfile",
             "usdGen:surfaceCage:ownerLengthProfile", S::F32, 2, 0},
            {kUsdGenCurveInputCageNormalizedT, "cage:normalizedT", "usdGen:surfaceCage:normalizedT", S::F32, 1, 0},
            {kUsdGenCurveInputCageTriangles, "cage:triangles", "usdGen:surfaceCage:triangles", S::I32, 3, 0},
            {kUsdGenCurveInputCageTriangleOwnerIndices, "cage:triangleOwnerIndices",
             "usdGen:surfaceCage:triangleOwnerIndices", S::I32, 1, 0},
            {kUsdGenCurveInputCageTriangleRootCharts, "cage:triangleRootCharts",
             "usdGen:surfaceCage:triangleRootCharts", S::F32, 2, 0},
            {kUsdGenCurveInputCageOwnerChartCentroids, "cage:ownerChartCentroids",
             "usdGen:surfaceCage:ownerChartCentroids", S::F32, 2, 0},
            {kUsdGenCurveInputCageOwnerChartMeanRadii, "cage:ownerChartMeanRadii",
             "usdGen:surfaceCage:ownerChartMeanRadii", S::F32, 1, 0},
        };
        static_assert(sizeof(kRows) / sizeof(kRows[0]) == kUsdGenCurveInputSlots, "one row per slot");
        for (Row const &row : kRows) {
            UsdGenChannelDesc &d = fresh->channels[row.slot];
            d.name = UsdGenInternToken(row.channel);
            d.scalar = row.scalar;
            d.arity = row.arity;
            d.level = 0;
            d.flags = row.arity > 1 ? uint8_t(kUsdGenChannelPacked) : uint8_t(0);
            UsdGenInputSlot &in = fresh->slots[row.slot];
            in.attribute = UsdGenInternToken(row.attribute);
            in.channel = row.slot;
            in.level = (row.flags & kUsdGenInputCounts) ? 2 : 0;
            in.flags = row.flags;
        }
        static constexpr char const *kValues[kUsdGenCurveValues] = {
            "usdGen:role", "primvars:usdGen:role", "type", "basis", "wrap",
            "widthsInterpolation", "usdGen:restFromCurrentPoints", "worldMatrix",
            "primvars:usdGen:frozenEpoch", "usdGen:authoredPlanes:names", "usdGen:surfaceCage"};
        for (size_t i = 0; i < kUsdGenCurveValues; ++i)
            fresh->attributes[i] = UsdGenInternToken(kValues[i]);
        fresh->input.schemaType = UsdGenInternToken("BasisCurves");
        fresh->input.dynamicPrimvarPrefix = UsdGenInternToken("primvars:");
    }
    fresh->layout.levels = 1;
    fresh->layout.channels = UsdGenSpan<const UsdGenChannelDesc>(fresh->channels, kUsdGenCurveInputSlots);
    fresh->input.slots = UsdGenSpan<const UsdGenInputSlot>(fresh->slots, kUsdGenCurveInputSlots);
    fresh->input.attributes = UsdGenSpan<const UsdGenToken>(fresh->attributes, kUsdGenCurveValues);
    fresh->input.materialize = &MaterializeReference;
    fresh->input.displayName = "curve set";
    fresh->input.validate = &ValidateRecord;
    fresh->input.digest = &DigestRecord;
    fresh->input.lane = &RecordLane;
    fresh->input.authoredChannels = &AuthoredChannels;
    UsdGenTypeDef def;
    def.name = "UsdGenCurveSetInput";
    def.layout = &fresh->layout;
    def.input = &fresh->input;
    UsdGenTypeId const id = r.RegisterType(def);
    if (id == UsdGenTypeId::Invalid) {
        delete fresh;
        CurveInputState const *s = gCurveInput.load(std::memory_order_acquire);
        return s ? s->kind : UsdGenTypeId::Invalid;
    }
    fresh->kind = id;
    fresh->layout.primitive = id;
    fresh->input.kind = id;
    CurveInputState *expected = nullptr;
    gCurveInput.compare_exchange_strong(expected, fresh, std::memory_order_acq_rel,
                                        std::memory_order_acquire);
    return id;
}

UsdGenTypeId UsdGenCurveSetInputKind() noexcept
{
    CurveInputState const *s = State();
    return s ? s->kind : UsdGenTypeId::Invalid;
}

UsdGenSpan<const UsdGenInputRecord> UsdGenCurveSetInputs(UsdGenGraphSpec const &spec) noexcept
{
    return spec.Inputs(UsdGenCurveSetInputKind());
}

UsdGenInputRecord const *UsdGenFindCurveSetInput(UsdGenGraphSpec const &spec, UsdGenNodeKey key) noexcept
{
    return spec.FindInput(UsdGenCurveSetInputKind(), key);
}

// ---- authored planes ----------------------------------------------------------

UsdGenAuthoredPlanes UsdGenAuthoredPlanes::Of(UsdGenInputRecord const &record) noexcept
{
    UsdGenAuthoredPlanes p;
    UsdGenValue const *names = record.Value(kUsdGenCurveValuePlaneNames);
    if (!names || !names->IsHolding<UsdGenTokenArray>()) return p;
    p._names = names->UncheckedGet<UsdGenTokenArray>();
    p._info = record.Column<int>(kUsdGenCurveInputPlaneInfo);
    p._count = std::min(p._names.size(), p._info.size() / kPlaneInfo);
    p._floats = record.Column<float>(kUsdGenCurveInputPlaneFloats);
    p._ints = record.Column<int>(kUsdGenCurveInputPlaneInts);
    return p;
}

UsdGenAuthoredPlanes UsdGenAuthoredPlanes::Of(std::vector<UsdGenAuthoredPlaneDesc> const &planes) noexcept
{
    UsdGenAuthoredPlanes p;
    p._count = planes.size();
    p._descs = planes.data();
    return p;
}

UsdGenAuthoredPlaneView UsdGenAuthoredPlanes::_At(size_t i, size_t floats, size_t ints) const noexcept
{
    UsdGenAuthoredPlaneView v;
    if (i >= _count) return v;
    if (_descs) {
        UsdGenAuthoredPlaneDesc const &d = _descs[i];
        v.name = d.name;
        v.type = d.type;
        v.domain = d.domain;
        v.arity = d.arity;
        v.floatValues = Arr(d.floatValues);
        v.intValues = Arr(d.intValues);
        return v;
    }
    int const *info = _info.data() + i * kPlaneInfo;
    v.name = _names[i];
    v.type = UsdGenAuthoredPlaneType(uint8_t(info[0]));
    v.domain = UsdGenAuthoredPlaneDomain(uint8_t(info[1]));
    v.arity = uint8_t(info[2]);
    size_t const nf = size_t(info[3]), ni = size_t(info[4]);
    if (floats + nf <= _floats.size()) v.floatValues = {_floats.data() + floats, nf};
    if (ints + ni <= _ints.size()) v.intValues = {_ints.data() + ints, ni};
    return v;
}

void UsdGenAuthoredPlanes::_Advance(size_t i, size_t *floats, size_t *ints) const noexcept
{
    if (_descs || i >= _count) return;
    *floats += size_t(_info[i * kPlaneInfo + 3]);
    *ints += size_t(_info[i * kPlaneInfo + 4]);
}

UsdGenAuthoredPlaneView UsdGenAuthoredPlanes::operator[](size_t i) const noexcept
{
    size_t floats = 0, ints = 0;
    for (size_t k = 0; k < i && k < _count; ++k) _Advance(k, &floats, &ints);
    return _At(i, floats, ints);
}

// ---- view -----------------------------------------------------------------------

UsdGenCurveSetInputView UsdGenCurveSetInputView::Of(UsdGenInputRecord const &record) noexcept
{
    UsdGenCurveSetInputView v;
    v.key = record.key;
    v.role = UsdGenLane(uint8_t(record.Get<int>(kUsdGenCurveValueRole, 0)));
    v.curveRole = record.Get<UsdGenToken>(kUsdGenCurveValueCurveRole, UsdGenToken());
    v.curveVertexCounts = record.Column<int>(kUsdGenCurveInputCounts);
    v.points = record.Column<UsdGenVec3f>(kUsdGenCurveInputPoints);
    v.rest = record.Column<UsdGenVec3f>(kUsdGenCurveInputRest);
    v.restFromCurrentPoints = record.Get<bool>(kUsdGenCurveValueRestFromCurrentPoints, false);
    v.worldMatrix = record.Get<UsdGenMatrix4d>(kUsdGenCurveValueWorldMatrix, kIdentity);
    v.widths = record.Column<float>(kUsdGenCurveInputWidths);
    v.type = record.Get<UsdGenToken>(kUsdGenCurveValueType, UsdGenCurveTokens().cubic);
    v.basis = record.Get<UsdGenToken>(kUsdGenCurveValueBasis, UsdGenCurveTokens().bspline);
    v.wrap = record.Get<UsdGenToken>(kUsdGenCurveValueWrap, UsdGenCurveTokens().pinned);
    v.widthsInterpolation = record.Get<UsdGenToken>(kUsdGenCurveValueWidthsInterpolation, UsdGenCurveTokens().vertexWidths);
    v.skinPrim = record.Column<int>(kUsdGenCurveInputSkinPrim);
    v.curveId = record.Column<uint64_t>(kUsdGenCurveInputCurveId);
    v.skinPrimUv = record.Column<UsdGenVec2f>(kUsdGenCurveInputSkinPrimUv);
    v.rootFrame = record.Column<UsdGenMatrix4d>(kUsdGenCurveInputRootFrame);
    if (UsdGenValue const *epoch = record.Value(kUsdGenCurveValueFrozenEpoch);
        epoch && epoch->IsHolding<std::string>()) {
        v.frozenEpochHolder = *epoch;   // shares the held string
        v.frozenEpoch = v.frozenEpochHolder.UncheckedGet<std::string>();
    }
    v.authoredPlanes = UsdGenAuthoredPlanes::Of(record);
    if (record.Get<bool>(kUsdGenCurveValueSurfaceCage, false)) {
        UsdGenSurfaceCageView c;
        c.ownerIds = record.Column<int>(kUsdGenCurveInputCageOwnerIds);
        c.ownerDensities = record.Column<float>(kUsdGenCurveInputCageOwnerDensities);
        c.ownerSeeds = record.Column<int>(kUsdGenCurveInputCageOwnerSeeds);
        c.ownerCvCounts = record.Column<int>(kUsdGenCurveInputCageOwnerCvCounts);
        c.ownerEdgeBias = record.Column<float>(kUsdGenCurveInputCageOwnerEdgeBias);
        c.ownerLengthProfileOffsets = record.Column<int>(kUsdGenCurveInputCageOwnerLengthProfileOffsets);
        c.ownerLengthProfile = record.Column<UsdGenVec2f>(kUsdGenCurveInputCageOwnerLengthProfile);
        c.normalizedT = record.Column<float>(kUsdGenCurveInputCageNormalizedT);
        c.triangles = record.Column<UsdGenVec3i>(kUsdGenCurveInputCageTriangles);
        c.triangleOwnerIndices = record.Column<int>(kUsdGenCurveInputCageTriangleOwnerIndices);
        c.triangleRootCharts = record.Column<UsdGenVec2f>(kUsdGenCurveInputCageTriangleRootCharts);
        c.ownerChartCentroids = record.Column<UsdGenVec2f>(kUsdGenCurveInputCageOwnerChartCentroids);
        c.ownerChartMeanRadii = record.Column<float>(kUsdGenCurveInputCageOwnerChartMeanRadii);
        v.surfaceCage = UsdGenSurfaceCageRef(c);
    }
    v.curveGeneration = record.generation;
    v.data = record.data;
    return v;
}

UsdGenCurveSetInputView UsdGenCurveSetInputView::Of(UsdGenCurveSetData const &d,
                                                    UsdGenNodeKey key) noexcept
{
    UsdGenCurveSetInputView v;
    v.key = key;
    v.role = d.role;
    v.curveRole = d.curveRole;
    v.curveVertexCounts = Arr(d.curveVertexCounts);
    v.points = Arr(d.points);
    v.rest = Arr(d.rest);
    v.restFromCurrentPoints = d.restFromCurrentPoints;
    v.worldMatrix = d.worldMatrix;
    v.widths = Arr(d.widths);
    v.type = d.type;
    v.basis = d.basis;
    v.wrap = d.wrap;
    v.widthsInterpolation = d.widthsInterpolation;
    v.skinPrim = Arr(d.skinPrim);
    v.curveId = Arr(d.curveId);
    v.skinPrimUv = Arr(d.skinPrimUv);
    v.rootFrame = Arr(d.rootFrame);
    v.frozenEpoch = d.frozenEpoch;
    v.authoredPlanes = UsdGenAuthoredPlanes::Of(d.authoredPlanes);
    if (d.surfaceCage) {
        UsdGenSurfaceCagePayload const &p = *d.surfaceCage;
        UsdGenSurfaceCageView c;
        c.ownerIds = Arr(p.ownerIds);
        c.ownerDensities = Arr(p.ownerDensities);
        c.ownerSeeds = Arr(p.ownerSeeds);
        c.ownerCvCounts = Arr(p.ownerCvCounts);
        c.ownerEdgeBias = Arr(p.ownerEdgeBias);
        c.ownerLengthProfileOffsets = Arr(p.ownerLengthProfileOffsets);
        c.ownerLengthProfile = Arr(p.ownerLengthProfile);
        c.normalizedT = Arr(p.normalizedT);
        c.triangles = Arr(p.triangles);
        c.triangleOwnerIndices = Arr(p.triangleOwnerIndices);
        c.triangleRootCharts = Arr(p.triangleRootCharts);
        c.ownerChartCentroids = Arr(p.ownerChartCentroids);
        c.ownerChartMeanRadii = Arr(p.ownerChartMeanRadii);
        v.surfaceCage = UsdGenSurfaceCageRef(c);
    }
    v.curveGeneration = d.curveGeneration;
    return v;
}

// ---- packer ---------------------------------------------------------------------

UsdGenInputRecord UsdGenCurveSetRecord(UsdGenCurveSetData const &d, UsdGenNodeKey key,
                                       UsdGenToken const &label, UsdGenInputColumnCache *cache)
{
    UsdGenInputRecordWriter w(UsdGenCurveSetInputKind());
    w.UseCache(cache, key);
    w.Array(kUsdGenCurveInputPoints, d.points);
    w.Array(kUsdGenCurveInputCounts, d.curveVertexCounts);
    w.Array(kUsdGenCurveInputRest, d.rest);
    w.Array(kUsdGenCurveInputWidths, d.widths);
    w.Array(kUsdGenCurveInputSkinPrim, d.skinPrim);
    w.Array(kUsdGenCurveInputCurveId, d.curveId);
    w.Array(kUsdGenCurveInputSkinPrimUv, d.skinPrimUv);
    w.Array(kUsdGenCurveInputRootFrame, d.rootFrame);
    std::vector<int> info;
    std::vector<float> floats;
    std::vector<int> ints;
    UsdGenTokenArray names;
    info.reserve(d.authoredPlanes.size() * kPlaneInfo);
    names.reserve(d.authoredPlanes.size());
    for (UsdGenAuthoredPlaneDesc const &plane : d.authoredPlanes) {
        names.push_back(plane.name);
        info.push_back(int(plane.type));
        info.push_back(int(plane.domain));
        info.push_back(int(plane.arity));
        info.push_back(int(plane.floatValues.size()));
        info.push_back(int(plane.intValues.size()));
        floats.insert(floats.end(), plane.floatValues.begin(), plane.floatValues.end());
        ints.insert(ints.end(), plane.intValues.begin(), plane.intValues.end());
    }
    w.Column(kUsdGenCurveInputPlaneInfo, info.data(), info.size());
    w.Column(kUsdGenCurveInputPlaneFloats, floats.data(), floats.size());
    w.Column(kUsdGenCurveInputPlaneInts, ints.data(), ints.size());
    if (UsdGenSurfaceCagePayload const *c = d.surfaceCage.get()) {
        w.Array(kUsdGenCurveInputCageOwnerIds, c->ownerIds);
        w.Array(kUsdGenCurveInputCageOwnerDensities, c->ownerDensities);
        w.Array(kUsdGenCurveInputCageOwnerSeeds, c->ownerSeeds);
        w.Array(kUsdGenCurveInputCageOwnerCvCounts, c->ownerCvCounts);
        w.Array(kUsdGenCurveInputCageOwnerEdgeBias, c->ownerEdgeBias);
        w.Array(kUsdGenCurveInputCageOwnerLengthProfileOffsets, c->ownerLengthProfileOffsets);
        w.Array(kUsdGenCurveInputCageOwnerLengthProfile, c->ownerLengthProfile);
        w.Array(kUsdGenCurveInputCageNormalizedT, c->normalizedT);
        w.Array(kUsdGenCurveInputCageTriangles, c->triangles);
        w.Array(kUsdGenCurveInputCageTriangleOwnerIndices, c->triangleOwnerIndices);
        w.Array(kUsdGenCurveInputCageTriangleRootCharts, c->triangleRootCharts);
        w.Array(kUsdGenCurveInputCageOwnerChartCentroids, c->ownerChartCentroids);
        w.Array(kUsdGenCurveInputCageOwnerChartMeanRadii, c->ownerChartMeanRadii);
    }
    w.Value(kUsdGenCurveValueRole, UsdGenValue(int(d.role)));
    w.Value(kUsdGenCurveValueCurveRole, UsdGenValue(d.curveRole));
    w.Value(kUsdGenCurveValueType, UsdGenValue(d.type));
    w.Value(kUsdGenCurveValueBasis, UsdGenValue(d.basis));
    w.Value(kUsdGenCurveValueWrap, UsdGenValue(d.wrap));
    w.Value(kUsdGenCurveValueWidthsInterpolation, UsdGenValue(d.widthsInterpolation));
    w.Value(kUsdGenCurveValueRestFromCurrentPoints, UsdGenValue(d.restFromCurrentPoints));
    w.Value(kUsdGenCurveValueWorldMatrix, UsdGenValue(d.worldMatrix));
    w.Value(kUsdGenCurveValueFrozenEpoch, UsdGenValue(d.frozenEpoch));
    w.Value(kUsdGenCurveValuePlaneNames, UsdGenValue(std::move(names)));
    w.Value(kUsdGenCurveValueSurfaceCage, UsdGenValue(bool(d.surfaceCage)));
    return w.Finish(key, UsdGenNodeKey{}, label, d.curveGeneration);
}

template <class T>
static void CopyTo(UsdGenSpan<const T> from, UsdGenCowArray<T> *to)
{
    to->assign(from.begin(), from.end());
}

void UsdGenCopyCurveSetData(UsdGenCurveSetInputView const &v, UsdGenCurveSetData *out)
{
    out->role = v.role;
    out->curveRole = v.curveRole;
    CopyTo(v.curveVertexCounts, &out->curveVertexCounts);
    CopyTo(v.points, &out->points);
    CopyTo(v.rest, &out->rest);
    out->restFromCurrentPoints = v.restFromCurrentPoints;
    out->worldMatrix = v.worldMatrix;
    CopyTo(v.widths, &out->widths);
    out->type = v.type;
    out->basis = v.basis;
    out->wrap = v.wrap;
    out->widthsInterpolation = v.widthsInterpolation;
    CopyTo(v.skinPrim, &out->skinPrim);
    CopyTo(v.curveId, &out->curveId);
    CopyTo(v.skinPrimUv, &out->skinPrimUv);
    CopyTo(v.rootFrame, &out->rootFrame);
    out->frozenEpoch.assign(v.frozenEpoch.data(), v.frozenEpoch.size());
    out->authoredPlanes.clear();
    out->authoredPlanes.reserve(v.authoredPlanes.size());
    for (UsdGenAuthoredPlaneView const &p : v.authoredPlanes) {
        UsdGenAuthoredPlaneDesc d;
        d.name = p.name;
        d.type = p.type;
        d.domain = p.domain;
        d.arity = p.arity;
        CopyTo(p.floatValues, &d.floatValues);
        CopyTo(p.intValues, &d.intValues);
        out->authoredPlanes.push_back(std::move(d));
    }
    out->surfaceCage.reset();
    if (v.surfaceCage) {
        auto cage = std::make_shared<UsdGenSurfaceCagePayload>();
        UsdGenSurfaceCageView const &c = *v.surfaceCage;
        CopyTo(c.ownerIds, &cage->ownerIds);
        CopyTo(c.ownerDensities, &cage->ownerDensities);
        CopyTo(c.ownerSeeds, &cage->ownerSeeds);
        CopyTo(c.ownerCvCounts, &cage->ownerCvCounts);
        CopyTo(c.ownerEdgeBias, &cage->ownerEdgeBias);
        CopyTo(c.ownerLengthProfileOffsets, &cage->ownerLengthProfileOffsets);
        CopyTo(c.ownerLengthProfile, &cage->ownerLengthProfile);
        CopyTo(c.normalizedT, &cage->normalizedT);
        CopyTo(c.triangles, &cage->triangles);
        CopyTo(c.triangleOwnerIndices, &cage->triangleOwnerIndices);
        CopyTo(c.triangleRootCharts, &cage->triangleRootCharts);
        CopyTo(c.ownerChartCentroids, &cage->ownerChartCentroids);
        CopyTo(c.ownerChartMeanRadii, &cage->ownerChartMeanRadii);
        out->surfaceCage = std::move(cage);
    }
    out->curveGeneration = v.curveGeneration;
}

// ---- materialize: the reference set (was compiler.cpp BuildReferenceValue) -------

namespace {

bool MaterializeReference(UsdGenInputRecord const &record,
                          UsdGenRef<const UsdGenElementSet> *out, std::string *error)
{
    UsdGenCurveSetInputView const curves = UsdGenCurveSetInputView::Of(record);
    if (curves.role != UsdGenLane::Reference) return true;   // a CurveSource's rails: none
    if (!curves.key.IsValid()) {
        *error = "reference curve set has an empty path";
        return false;
    }
    std::string const &label = record.label.GetString();
    uint64_t totalCvs = 0;
    for (int count : curves.curveVertexCounts) {
        if (count < 0) {
            *error = "reference curve set '" + label + "' has a negative curve vertex count";
            return false;
        }
        totalCvs += static_cast<uint32_t>(count);
    }
    if (totalCvs > UINT32_MAX || curves.points.size() != totalCvs ||
        (!curves.rest.empty() && curves.rest.size() != totalCvs) ||
        (!curves.widths.empty() && curves.widths.size() != totalCvs) ||
        (!curves.curveId.empty() &&
         curves.curveId.size() != curves.curveVertexCounts.size()) ||
        (!curves.skinPrim.empty() &&
         curves.skinPrim.size() != curves.curveVertexCounts.size()) ||
        (!curves.skinPrimUv.empty() &&
         curves.skinPrimUv.size() != curves.curveVertexCounts.size()) ||
        (!curves.rootFrame.empty() &&
         curves.rootFrame.size() != curves.curveVertexCounts.size())) {
        *error = "reference curve set '" + label +
            "' has inconsistent topology/value array sizes";
        return false;
    }

    // A reference value owns the optional C3 rest plane as immutable COW
    // payload.  Validate it before constructing the shared value: an absent
    // rest plane follows the CUDA/reference contract and falls back to the
    // authored points, while a present plane must be complete and finite.
    for (UsdGenVec3f const &rest : curves.rest) {
        if (!std::isfinite(rest[0]) || !std::isfinite(rest[1]) ||
            !std::isfinite(rest[2])) {
            *error = "reference curve set '" + label + "' has non-finite rest points";
            return false;
        }
    }

    UsdGenCurveBuffer buffer;   // builder, converted once below
    buffer.totalCurves = static_cast<uint32_t>(curves.curveVertexCounts.size());
    buffer.totalCvs = static_cast<uint32_t>(totalCvs);
    buffer.topologyVersion = curves.curveGeneration;
    buffer.valueVersion = curves.curveGeneration;
    bool uniform = true;
    int const firstCount = curves.curveVertexCounts.empty()
        ? 0 : curves.curveVertexCounts.front();
    for (int count : curves.curveVertexCounts) uniform = uniform && count == firstCount;
    if (!uniform) {
        buffer.cvOffsets.resize(buffer.totalCurves + 1, 0);
        for (uint32_t i = 0; i < buffer.totalCurves; ++i)
            buffer.cvOffsets[i + 1] = buffer.cvOffsets[i] + curves.curveVertexCounts[i];
    }
    CopyTo(curves.widths, &buffer.width);
    CopyTo(curves.rest.empty() ? curves.points : curves.rest, &buffer.rest);
    CopyTo(curves.curveId, &buffer.curveId);
    if (buffer.curveId.empty()) {
        buffer.curveId.resize(buffer.totalCurves);
        for (uint32_t i = 0; i < buffer.totalCurves; ++i) buffer.curveId[i] = i;
    }
    CopyTo(curves.skinPrim, &buffer.rootPrim);
    CopyTo(curves.skinPrimUv, &buffer.rootUV);
    if (!curves.rootFrame.empty()) {
        // UsdGenMatrix4d is row-major/row-vector.  Root-frame consumers use the
        // same tangent/binormal/normal row convention as the CPU kernels.
        buffer.rootT.resize(buffer.totalCurves);
        buffer.rootB.resize(buffer.totalCurves);
        buffer.rootN.resize(buffer.totalCurves);
        for (uint32_t i = 0; i < buffer.totalCurves; ++i) {
            UsdGenMatrix4d const &frame = curves.rootFrame[i];
            UsdGenVec3d const tangent = frame.GetRow3(0);
            UsdGenVec3d const binormal = frame.GetRow3(1);
            UsdGenVec3d const normal = frame.GetRow3(2);
            buffer.rootT[i] = UsdGenVec3f(tangent);
            buffer.rootB[i] = UsdGenVec3f(binormal);
            buffer.rootN[i] = UsdGenVec3f(normal);
        }
    }
    buffer.px.resize(buffer.totalCvs);
    buffer.py.resize(buffer.totalCvs);
    buffer.pz.resize(buffer.totalCvs);
    buffer.hairT.resize(buffer.totalCvs);
    uint32_t cv = 0;
    for (int count : curves.curveVertexCounts) {
        for (int i = 0; i < count; ++i, ++cv) {
            UsdGenVec3f const &point = curves.points[cv];
            buffer.px[cv] = point[0];
            buffer.py[cv] = point[1];
            buffer.pz[cv] = point[2];
            buffer.hairT[cv] = count > 1 ? float(i) / float(count - 1) : 0.0f;
        }
    }
    UsdGenEngineTokens const &tokens = UsdGenTokens();
    for (UsdGenAuthoredPlaneView const &authored : curves.authoredPlanes) {
        UsdGenPlane plane;
        plane.name = authored.name;
        plane.arity = authored.arity;
        switch (authored.domain) {
        case UsdGenAuthoredPlaneDomain::Point:
            plane.interpolation = tokens.vertex;
            break;
        case UsdGenAuthoredPlaneDomain::Primitive:
            plane.interpolation = tokens.uniform;
            break;
        case UsdGenAuthoredPlaneDomain::Groom:
            plane.interpolation = tokens.constant;
            break;
        default:
            *error = "reference curve set '" + label + "' has an invalid authored plane domain";
            return false;
        }
        if (authored.type == UsdGenAuthoredPlaneType::Float32) {
            plane.type = tokens.float_;
            CopyTo(authored.floatValues, &plane.f);
        } else if (authored.type == UsdGenAuthoredPlaneType::Int32) {
            plane.type = tokens.int_;
            CopyTo(authored.intValues, &plane.i);
        } else {
            *error = "reference curve set '" + label + "' has an invalid authored plane type";
            return false;
        }
        if (plane.interpolation == tokens.vertex)
            buffer.extraCv.push_back(std::move(plane));
        else
            buffer.extraCurve.push_back(std::move(plane));
    }
    auto byName = [](UsdGenPlane const &a, UsdGenPlane const &b) {
        return a.name < b.name;
    };
    std::sort(buffer.extraCv.begin(), buffer.extraCv.end(), byName);
    std::sort(buffer.extraCurve.begin(), buffer.extraCurve.end(), byName);
    // One arena at compile; exported so no cook ever recycles it in place.
    UsdGenRef<const UsdGenElementSet> set = UsdGenCurveSetFromBuffer(
        buffer, buffer.topologyVersion, buffer.valueVersion, true);
    if (!set) {
        *error = "reference curve set '" + label + "' is not a consistent curve set";
        return false;
    }
    set->Export();
    *out = std::move(set);
    return true;
}

}  // namespace

}  // namespace usdGen
