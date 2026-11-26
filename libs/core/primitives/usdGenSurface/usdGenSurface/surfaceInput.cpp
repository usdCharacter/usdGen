// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenSurface/surfaceInput.h"

#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/token.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <utility>

namespace usdGen {

namespace {

struct SurfaceInputState {
    UsdGenChannelDesc channels[kUsdGenSurfaceInputSlots];
    UsdGenInputSlot slots[kUsdGenSurfaceInputSlots];
    UsdGenToken attributes[kUsdGenSurfaceValues];
    UsdGenElementLayout layout;
    UsdGenInputDesc input;
    UsdGenTypeId kind = UsdGenTypeId::Invalid;
};

std::atomic<SurfaceInputState *> gSurfaceInput{nullptr};
GfMatrix4d const kIdentity(1.0);

SurfaceInputState const *State() noexcept
{
    if (SurfaceInputState const *s = gSurfaceInput.load(std::memory_order_acquire)) return s;
    (void)UsdGenOpRegistry::Get();   // its constructor registers the built-ins
    return gSurfaceInput.load(std::memory_order_acquire);
}

template <class T>
UsdGenSpan<const T> Arr(VtArray<T> const &a) noexcept
{
    return {a.cdata(), a.size()};
}

template <class T>
void CopyTo(UsdGenSpan<const T> from, VtArray<T> *to)
{
    to->assign(from.begin(), from.end());
}

// Moved from graph.cpp (RestSpec).
// Word-at-a-time FNV-1a over a surface's Default-space content: topology,
// rest points and normals, uv, subset and world matrix. The posed points,
// shutter samples and velocities are deliberately not terms.
class RestContentHash
{
public:
    void Bytes(void const *data, size_t size)
    {
        auto const *bytes = static_cast<unsigned char const *>(data);
        size_t i = 0;
        for (; i + 8 <= size; i += 8) {
            uint64_t word;
            std::memcpy(&word, bytes + i, 8);
            Mix(word);
        }
        uint64_t tail = 0;
        if (i < size) std::memcpy(&tail, bytes + i, size - i);
        Mix(tail);
        Mix(uint64_t(size));
    }
    template <class T> void Array(UsdGenSpan<const T> a)
    {
        Bytes(a.empty() ? nullptr : a.cdata(), a.size() * sizeof(T));
    }
    void Mix(uint64_t word) { _h ^= word; _h *= 0x100000001b3ULL; }
    uint64_t Value() const { return _h ? _h : 1; }

private:
    uint64_t _h = 1469598103934665603ULL;
};

}  // namespace

UsdGenTypeId UsdGenRegisterSurfaceInputKind(UsdGenRegistrar &r)
{
    if (SurfaceInputState const *s = gSurfaceInput.load(std::memory_order_acquire)) return s->kind;
    SurfaceInputState *fresh = new SurfaceInputState;   // startup; never freed
    {
        UsdGenCompileTokenScope scope;
        using S = UsdGenScalar;
        struct Row { UsdGenSurfaceInputSlot slot; char const *channel, *attribute; S scalar; uint8_t arity; uint8_t flags; };
        static constexpr Row kRows[] = {
            {kUsdGenSurfaceInputFaceVertexCounts, "faceVertexCounts", "faceVertexCounts", S::I32, 1,
             kUsdGenInputRequired},
            {kUsdGenSurfaceInputFaceVertexIndices, "faceVertexIndices", "faceVertexIndices", S::I32, 1,
             kUsdGenInputRequired},
            {kUsdGenSurfaceInputHoleIndices, "holeIndices", "holeIndices", S::I32, 1, 0},
            {kUsdGenSurfaceInputCreaseIndices, "creaseIndices", "creaseIndices", S::I32, 1, 0},
            {kUsdGenSurfaceInputCreaseLengths, "creaseLengths", "creaseLengths", S::I32, 1, 0},
            {kUsdGenSurfaceInputCornerIndices, "cornerIndices", "cornerIndices", S::I32, 1, 0},
            {kUsdGenSurfaceInputCreaseSharpnesses, "creaseSharpnesses", "creaseSharpnesses", S::F32, 1, 0},
            {kUsdGenSurfaceInputCornerSharpnesses, "cornerSharpnesses", "cornerSharpnesses", S::F32, 1, 0},
            {kUsdGenSurfaceInputRestPoints, "restPoints", "usdGen:rest:points", S::F32, 3, 0},
            {kUsdGenSurfaceInputRestNormals, "restNormals", "usdGen:rest:normals", S::F32, 3, 0},
            {kUsdGenSurfaceInputPoints, "points", "points", S::F32, 3, kUsdGenInputRequired},
            {kUsdGenSurfaceInputSamplePoints, "samples:points", "usdGen:samples:points", S::F32, 3, 0},
            {kUsdGenSurfaceInputSampleTimes, "samples:times", "usdGen:samples:times", S::F64, 1, 0},
            {kUsdGenSurfaceInputSampleCounts, "samples:counts", "usdGen:samples:counts", S::I32, 1, 0},
            {kUsdGenSurfaceInputVelocities, "velocities", "velocities", S::F32, 3, 0},
            {kUsdGenSurfaceInputUv, "uv", "st", S::F32, 2, kUsdGenInputPrimvar},
            {kUsdGenSurfaceInputSubsetFaces, "subsetFaces", "indices", S::I32, 1, 0},
            {kUsdGenSurfaceInputDensityMultiplier, "densityMultiplier", "usdGen:paint:density", S::F32, 1,
             kUsdGenInputPrimvar},
        };
        static_assert(sizeof(kRows) / sizeof(kRows[0]) == kUsdGenSurfaceInputSlots, "one row per slot");
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
            in.flags = row.flags;
        }
        static constexpr char const *kValues[kUsdGenSurfaceValues] = {
            "usdGen:surfaceId", "subdivisionScheme", "orientation", "interpolateBoundary",
            "faceVaryingLinearInterpolation", "triangleSubdivisionRule", "creaseMethod",
            "usdGen:rest:normals:interpolation", "usdGen:restFromCurrentPoints", "usdGen:isSubset",
            "worldMatrix"};
        for (size_t i = 0; i < kUsdGenSurfaceValues; ++i)
            fresh->attributes[i] = UsdGenInternToken(kValues[i]);
        fresh->input.schemaType = UsdGenInternToken("Mesh");
    }
    fresh->layout.levels = 1;
    fresh->layout.channels = UsdGenSpan<const UsdGenChannelDesc>(fresh->channels, kUsdGenSurfaceInputSlots);
    fresh->input.slots = UsdGenSpan<const UsdGenInputSlot>(fresh->slots, kUsdGenSurfaceInputSlots);
    fresh->input.attributes = UsdGenSpan<const UsdGenToken>(fresh->attributes, kUsdGenSurfaceValues);
    UsdGenTypeDef def;
    def.name = "UsdGenSurfaceInput";
    def.layout = &fresh->layout;
    def.input = &fresh->input;
    UsdGenTypeId const id = r.RegisterType(def);
    if (id == UsdGenTypeId::Invalid) {
        delete fresh;
        SurfaceInputState const *s = gSurfaceInput.load(std::memory_order_acquire);
        return s ? s->kind : UsdGenTypeId::Invalid;
    }
    fresh->kind = id;
    fresh->layout.primitive = id;
    fresh->input.kind = id;
    SurfaceInputState *expected = nullptr;
    gSurfaceInput.compare_exchange_strong(expected, fresh, std::memory_order_acq_rel,
                                          std::memory_order_acquire);
    return id;
}

UsdGenTypeId UsdGenSurfaceInputKind() noexcept
{
    SurfaceInputState const *s = State();
    return s ? s->kind : UsdGenTypeId::Invalid;
}

UsdGenSpan<const UsdGenInputRecord> UsdGenSurfaceInputs(UsdGenGraphSpec const &spec) noexcept
{
    return spec.Inputs(UsdGenSurfaceInputKind());
}

UsdGenSurfaceInputView UsdGenSurfaceAt(UsdGenGraphSpec const &spec, size_t id) noexcept
{
    UsdGenSpan<const UsdGenInputRecord> const surfaces = UsdGenSurfaceInputs(spec);
    return id < surfaces.size() ? UsdGenSurfaceInputView::Of(surfaces[id]) : UsdGenSurfaceInputView{};
}

UsdGenInputRecord const *UsdGenFindSurfaceInput(UsdGenGraphSpec const &spec, UsdGenNodeKey key) noexcept
{
    return spec.FindInput(UsdGenSurfaceInputKind(), key);
}

void UsdGenSetSurfaceId(UsdGenInputRecord &record, UsdGenSurfaceId id) noexcept
{
    if (kUsdGenSurfaceValueId < record.values.size())
        record.values[kUsdGenSurfaceValueId].value = VtValue(int(id));
}

// ---- samples ----------------------------------------------------------------------

UsdGenSurfaceSamples UsdGenSurfaceSamples::Of(UsdGenInputRecord const &record) noexcept
{
    UsdGenSurfaceSamples s;
    s._times = record.Column<double>(kUsdGenSurfaceInputSampleTimes);
    s._counts = record.Column<int>(kUsdGenSurfaceInputSampleCounts);
    s._points = record.Column<GfVec3f>(kUsdGenSurfaceInputSamplePoints);
    s._count = std::min(s._times.size(), s._counts.size());
    return s;
}

UsdGenSurfaceSamples UsdGenSurfaceSamples::Of(std::vector<UsdGenSurfaceSample> const &samples) noexcept
{
    UsdGenSurfaceSamples s;
    s._count = samples.size();
    s._samples = samples.data();
    return s;
}

UsdGenSurfaceSampleView UsdGenSurfaceSamples::operator[](size_t i) const noexcept
{
    UsdGenSurfaceSampleView v;
    if (i >= _count) return v;
    if (_samples) {
        v.time = _samples[i].time;
        v.points = Arr(_samples[i].points);
        return v;
    }
    size_t first = 0;
    for (size_t k = 0; k < i; ++k) first += size_t(_counts[k]);
    size_t const n = size_t(_counts[i]);
    v.time = _times[i];
    if (first + n <= _points.size()) v.points = {_points.data() + first, n};
    return v;
}

// ---- view -------------------------------------------------------------------------

UsdGenSurfaceInputView UsdGenSurfaceInputView::Of(UsdGenInputRecord const &record) noexcept
{
    UsdGenEngineTokens const &tk = UsdGenTokens();
    UsdGenSurfaceInputView v;
    v.key = record.key;
    v.owner = record.owner;
    v.id = UsdGenSurfaceId(record.Get<int>(kUsdGenSurfaceValueId, 0));
    v.faceVertexCounts = record.Column<int>(kUsdGenSurfaceInputFaceVertexCounts);
    v.faceVertexIndices = record.Column<int>(kUsdGenSurfaceInputFaceVertexIndices);
    v.subdivisionScheme = record.Get<TfToken>(kUsdGenSurfaceValueSubdivisionScheme, tk.none);
    v.orientation = record.Get<TfToken>(kUsdGenSurfaceValueOrientation, tk.rightHanded);
    v.interpolateBoundary = record.Get<TfToken>(kUsdGenSurfaceValueInterpolateBoundary, tk.edgeAndCorner);
    v.faceVaryingLinearInterpolation =
        record.Get<TfToken>(kUsdGenSurfaceValueFaceVaryingLinearInterpolation, tk.cornersPlus1);
    v.triangleSubdivisionRule =
        record.Get<TfToken>(kUsdGenSurfaceValueTriangleSubdivisionRule, tk.catmullClark);
    v.creaseMethod = record.Get<TfToken>(kUsdGenSurfaceValueCreaseMethod, tk.uniform);
    v.holeIndices = record.Column<int>(kUsdGenSurfaceInputHoleIndices);
    v.creaseIndices = record.Column<int>(kUsdGenSurfaceInputCreaseIndices);
    v.creaseLengths = record.Column<int>(kUsdGenSurfaceInputCreaseLengths);
    v.cornerIndices = record.Column<int>(kUsdGenSurfaceInputCornerIndices);
    v.creaseSharpnesses = record.Column<float>(kUsdGenSurfaceInputCreaseSharpnesses);
    v.cornerSharpnesses = record.Column<float>(kUsdGenSurfaceInputCornerSharpnesses);
    v.restPoints = record.Column<GfVec3f>(kUsdGenSurfaceInputRestPoints);
    v.restNormals = record.Column<GfVec3f>(kUsdGenSurfaceInputRestNormals);
    v.restNormalDomain = UsdGenSurfaceNormalDomain(uint8_t(record.Get<int>(kUsdGenSurfaceValueRestNormalDomain, 0)));
    v.restFromCurrentPoints = record.Get<bool>(kUsdGenSurfaceValueRestFromCurrentPoints, false);
    v.points = record.Column<GfVec3f>(kUsdGenSurfaceInputPoints);
    v.samples = UsdGenSurfaceSamples::Of(record);
    v.velocities = record.Column<GfVec3f>(kUsdGenSurfaceInputVelocities);
    v.uv = record.Column<GfVec2f>(kUsdGenSurfaceInputUv);
    v.subsetFaces = record.Column<int>(kUsdGenSurfaceInputSubsetFaces);
    v.isSubset = record.Get<bool>(kUsdGenSurfaceValueIsSubset, false);
    v.densityMultiplier = record.Column<float>(kUsdGenSurfaceInputDensityMultiplier);
    v.worldMatrix = record.Get<GfMatrix4d>(kUsdGenSurfaceValueWorldMatrix, kIdentity);
    v.surfaceGeneration = record.generation;
    v.data = record.data;
    return v;
}

UsdGenSurfaceInputView UsdGenSurfaceInputView::Of(UsdGenSurfaceData const &d, UsdGenNodeKey key,
                                                  UsdGenNodeKey owner) noexcept
{
    UsdGenSurfaceInputView v;
    v.key = key;
    v.owner = owner;
    v.id = d.id;
    v.faceVertexCounts = Arr(d.faceVertexCounts);
    v.faceVertexIndices = Arr(d.faceVertexIndices);
    v.subdivisionScheme = d.subdivisionScheme;
    v.orientation = d.orientation;
    v.interpolateBoundary = d.interpolateBoundary;
    v.faceVaryingLinearInterpolation = d.faceVaryingLinearInterpolation;
    v.triangleSubdivisionRule = d.triangleSubdivisionRule;
    v.creaseMethod = d.creaseMethod;
    v.holeIndices = Arr(d.holeIndices);
    v.creaseIndices = Arr(d.creaseIndices);
    v.creaseLengths = Arr(d.creaseLengths);
    v.cornerIndices = Arr(d.cornerIndices);
    v.creaseSharpnesses = Arr(d.creaseSharpnesses);
    v.cornerSharpnesses = Arr(d.cornerSharpnesses);
    v.restPoints = Arr(d.restPoints);
    v.restNormals = Arr(d.restNormals);
    v.restNormalDomain = d.restNormalDomain;
    v.restFromCurrentPoints = d.restFromCurrentPoints;
    v.points = Arr(d.points);
    v.samples = UsdGenSurfaceSamples::Of(d.samples);
    v.velocities = Arr(d.velocities);
    v.uv = Arr(d.uv);
    v.subsetFaces = Arr(d.subsetFaces);
    v.isSubset = d.isSubset;
    v.densityMultiplier = Arr(d.densityMultiplier);
    v.worldMatrix = d.worldMatrix;
    v.surfaceGeneration = d.surfaceGeneration;
    return v;
}

// ---- packer -----------------------------------------------------------------------

UsdGenInputRecord UsdGenSurfaceRecord(UsdGenSurfaceData const &d, UsdGenNodeKey key,
                                      UsdGenNodeKey owner, TfToken const &label,
                                      UsdGenInputColumnCache *cache)
{
    UsdGenInputRecordWriter w(UsdGenSurfaceInputKind());
    w.UseCache(cache, key);
    w.Array(kUsdGenSurfaceInputFaceVertexCounts, d.faceVertexCounts);
    w.Array(kUsdGenSurfaceInputFaceVertexIndices, d.faceVertexIndices);
    w.Array(kUsdGenSurfaceInputHoleIndices, d.holeIndices);
    w.Array(kUsdGenSurfaceInputCreaseIndices, d.creaseIndices);
    w.Array(kUsdGenSurfaceInputCreaseLengths, d.creaseLengths);
    w.Array(kUsdGenSurfaceInputCornerIndices, d.cornerIndices);
    w.Array(kUsdGenSurfaceInputCreaseSharpnesses, d.creaseSharpnesses);
    w.Array(kUsdGenSurfaceInputCornerSharpnesses, d.cornerSharpnesses);
    w.Array(kUsdGenSurfaceInputRestPoints, d.restPoints);
    w.Array(kUsdGenSurfaceInputRestNormals, d.restNormals);
    w.Array(kUsdGenSurfaceInputPoints, d.points);
    std::vector<GfVec3f> samplePoints;
    std::vector<double> sampleTimes;
    std::vector<int> sampleCounts;
    sampleTimes.reserve(d.samples.size());
    sampleCounts.reserve(d.samples.size());
    for (UsdGenSurfaceSample const &sample : d.samples) {
        sampleTimes.push_back(sample.time);
        sampleCounts.push_back(int(sample.points.size()));
        samplePoints.insert(samplePoints.end(), sample.points.begin(), sample.points.end());
    }
    w.Column(kUsdGenSurfaceInputSamplePoints, samplePoints.data(), samplePoints.size());
    w.Column(kUsdGenSurfaceInputSampleTimes, sampleTimes.data(), sampleTimes.size());
    w.Column(kUsdGenSurfaceInputSampleCounts, sampleCounts.data(), sampleCounts.size());
    w.Array(kUsdGenSurfaceInputVelocities, d.velocities);
    w.Array(kUsdGenSurfaceInputUv, d.uv);
    w.Array(kUsdGenSurfaceInputSubsetFaces, d.subsetFaces);
    w.Array(kUsdGenSurfaceInputDensityMultiplier, d.densityMultiplier);
    w.Value(kUsdGenSurfaceValueId, VtValue(int(d.id)));
    w.Value(kUsdGenSurfaceValueSubdivisionScheme, VtValue(d.subdivisionScheme));
    w.Value(kUsdGenSurfaceValueOrientation, VtValue(d.orientation));
    w.Value(kUsdGenSurfaceValueInterpolateBoundary, VtValue(d.interpolateBoundary));
    w.Value(kUsdGenSurfaceValueFaceVaryingLinearInterpolation, VtValue(d.faceVaryingLinearInterpolation));
    w.Value(kUsdGenSurfaceValueTriangleSubdivisionRule, VtValue(d.triangleSubdivisionRule));
    w.Value(kUsdGenSurfaceValueCreaseMethod, VtValue(d.creaseMethod));
    w.Value(kUsdGenSurfaceValueRestNormalDomain, VtValue(int(d.restNormalDomain)));
    w.Value(kUsdGenSurfaceValueRestFromCurrentPoints, VtValue(d.restFromCurrentPoints));
    w.Value(kUsdGenSurfaceValueIsSubset, VtValue(d.isSubset));
    w.Value(kUsdGenSurfaceValueWorldMatrix, VtValue(d.worldMatrix));
    return w.Finish(key, owner, label, d.surfaceGeneration);
}

bool UsdGenRestSurfaceRecord(UsdGenInputRecord const &posed, UsdGenInputRecord *out)
{
    UsdGenSurfaceInputView const s = UsdGenSurfaceInputView::Of(posed);
    if (!posed.data || s.restFromCurrentPoints || s.restPoints.empty() ||
        s.restPoints.size() != s.points.size())
        return false;   // no authored rest: Default space is the current geometry
    UsdGenElementSet const &from = *posed.data;
    UsdGenInputRecordWriter w(posed.kind);
    for (UsdGenSlot slot = 0; slot < kUsdGenSurfaceInputSlots; ++slot) {
        switch (slot) {
        case kUsdGenSurfaceInputPoints:
            w.Share(slot, from, kUsdGenSurfaceInputRestPoints);
            break;
        case kUsdGenSurfaceInputVelocities:
            break;
        case kUsdGenSurfaceInputSamplePoints:
        case kUsdGenSurfaceInputSampleTimes:
        case kUsdGenSurfaceInputSampleCounts:
            break;   // below
        default:
            w.Share(slot, from, slot);
        }
    }
    double const sampleTime = s.samples.empty() ? 0.0 : s.samples[0].time;
    int const sampleCount = int(s.restPoints.size());
    if (!s.samples.empty()) {
        w.Share(kUsdGenSurfaceInputSamplePoints, from, kUsdGenSurfaceInputRestPoints);
        w.Column(kUsdGenSurfaceInputSampleTimes, &sampleTime, 1);
        w.Column(kUsdGenSurfaceInputSampleCounts, &sampleCount, 1);
    }
    size_t index = 0;
    for (UsdGenParamValue const &value : posed.values) w.Value(index++, value.value);

    RestContentHash h;
    h.Mix(0x5265737453726675ULL);   // distinct from any posed generation domain
    h.Array(s.faceVertexCounts);
    h.Array(s.faceVertexIndices);
    h.Array(s.restPoints);
    h.Array(s.restNormals);
    h.Mix(uint64_t(s.restNormalDomain));
    h.Array(s.uv);
    h.Array(s.subsetFaces);
    h.Mix(s.isSubset ? 1u : 0u);
    h.Bytes(s.worldMatrix.GetArray(), 16 * sizeof(double));
    *out = w.Finish(posed.key, posed.owner, posed.label, h.Value());
    return true;
}

void UsdGenCopySurfaceData(UsdGenSurfaceInputView const &v, UsdGenSurfaceData *out)
{
    out->id = v.id;
    CopyTo(v.faceVertexCounts, &out->faceVertexCounts);
    CopyTo(v.faceVertexIndices, &out->faceVertexIndices);
    out->subdivisionScheme = v.subdivisionScheme;
    out->orientation = v.orientation;
    out->interpolateBoundary = v.interpolateBoundary;
    out->faceVaryingLinearInterpolation = v.faceVaryingLinearInterpolation;
    out->triangleSubdivisionRule = v.triangleSubdivisionRule;
    out->creaseMethod = v.creaseMethod;
    CopyTo(v.holeIndices, &out->holeIndices);
    CopyTo(v.creaseIndices, &out->creaseIndices);
    CopyTo(v.creaseLengths, &out->creaseLengths);
    CopyTo(v.cornerIndices, &out->cornerIndices);
    CopyTo(v.creaseSharpnesses, &out->creaseSharpnesses);
    CopyTo(v.cornerSharpnesses, &out->cornerSharpnesses);
    CopyTo(v.restPoints, &out->restPoints);
    CopyTo(v.restNormals, &out->restNormals);
    out->restNormalDomain = v.restNormalDomain;
    out->restFromCurrentPoints = v.restFromCurrentPoints;
    CopyTo(v.points, &out->points);
    out->samples.clear();
    for (UsdGenSurfaceSampleView const &sample : v.samples) {
        UsdGenSurfaceSample copy;
        copy.time = sample.time;
        CopyTo(sample.points, &copy.points);
        out->samples.push_back(std::move(copy));
    }
    CopyTo(v.velocities, &out->velocities);
    CopyTo(v.uv, &out->uv);
    CopyTo(v.subsetFaces, &out->subsetFaces);
    out->isSubset = v.isSubset;
    CopyTo(v.densityMultiplier, &out->densityMultiplier);
    out->worldMatrix = v.worldMatrix;
    out->surfaceGeneration = v.surfaceGeneration;
}

}  // namespace usdGen
