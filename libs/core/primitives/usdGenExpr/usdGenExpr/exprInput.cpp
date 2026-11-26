// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenExpr/exprInput.h"

#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/token.h"

#include <atomic>
#include <cstring>
#include <utility>

namespace usdGen {

bool operator==(UsdGenExpressionOutputDesc const &a, UsdGenExpressionOutputDesc const &b) noexcept
{
    return a.name == b.name && a.nativeType == b.nativeType &&
           a.shape.scalar == b.shape.scalar && a.shape.elementCount == b.shape.elementCount &&
           a.shape.components == b.shape.components && a.shape.rows == b.shape.rows &&
           a.shape.columns == b.shape.columns && a.shape.isArray == b.shape.isArray;
}

bool operator==(UsdGenExpressionSlot const &a, UsdGenExpressionSlot const &b) noexcept
{
    return a.name == b.name && a.targets == b.targets && a.geometries == b.geometries &&
           a.maps == b.maps;
}

namespace {

using Outputs = std::vector<UsdGenExpressionOutputDesc>;
using Slots = std::vector<UsdGenExpressionSlot>;

struct ExprInputState {
    UsdGenToken expressionAttributes[kUsdGenExpressionValues];
    UsdGenElementLayout expressionLayout;
    UsdGenInputDesc expression;
    UsdGenChannelDesc geometryChannels[kUsdGenGeometryInputSlots];
    UsdGenInputSlot geometrySlots[kUsdGenGeometryInputSlots];
    UsdGenToken geometryAttributes[kUsdGenGeometryValues];
    UsdGenElementLayout geometryLayout;
    UsdGenInputDesc geometry;
    UsdGenTypeId expressionKind = UsdGenTypeId::Invalid, geometryKind = UsdGenTypeId::Invalid;
};

std::atomic<ExprInputState *> gExprInput{nullptr};
GfMatrix4d const kIdentity(1.0);

ExprInputState const *State() noexcept
{
    if (ExprInputState const *s = gExprInput.load(std::memory_order_acquire)) return s;
    (void)UsdGenOpRegistry::Get();   // its constructor registers the built-ins
    return gExprInput.load(std::memory_order_acquire);
}

template <class T>
UsdGenSpan<const T> Arr(VtArray<T> const &a) noexcept
{
    return {a.cdata(), a.size()};
}

void MixShape(UsdGenInputDigest &d, int lane, expr::ValueShape const &shape)
{
    d.Mix(lane, static_cast<uint64_t>(shape.scalar));
    d.Mix(lane, shape.elementCount);
    d.Mix(lane, shape.components);
    d.Mix(lane, shape.rows);
    d.Mix(lane, shape.columns);
    d.Mix(lane, static_cast<uint64_t>(shape.isArray));
}

/// Execution cache key terms (moved from the session cooker); no layout terms.
void DigestExpression(UsdGenInputRecord const &record, UsdGenInputDigest &d, bool layout)
{
    if (layout) return;
    UsdGenExpressionInputView const expression = UsdGenExpressionInputView::Of(record);
    d.Mix(0, expression.key.value);
    d.MixText(1, expression.source);
    d.Mix(0, static_cast<uint64_t>(expression.outputs.size()));
    for (UsdGenExpressionOutputDesc const &output : expression.outputs) {
        d.MixText(1, output.name.GetString());
        d.MixText(0, output.nativeType.GetString());
        MixShape(d, 1, output.shape);
    }
    d.Mix(0, static_cast<uint64_t>(expression.inputs.size()));
    for (UsdGenExpressionSlot const &input : expression.inputs) {
        d.MixText(1, input.name.GetString());
        for (UsdGenNodeKey const key : input.targets) d.Mix(0, key.value);
        for (UsdGenNodeKey const key : input.geometries) d.Mix(1, key.value);
        for (UsdGenNodeKey const key : input.maps) d.Mix(0, key.value);
    }
}

/// Geometry an expression samples is an execution input like any other:
/// its content moves the key.
void DigestGeometry(UsdGenInputRecord const &record, UsdGenInputDigest &d, bool layout)
{
    if (layout) return;
    UsdGenGeometryInputView const geometry = UsdGenGeometryInputView::Of(record);
    d.Mix(0, geometry.key.value);
    d.Mix(1, static_cast<uint64_t>(geometry.kind));
    d.MixSpan(0, geometry.counts);
    d.MixSpan(1, geometry.indices);
    d.MixSpan(0, geometry.points);
    d.MixSpan(1, geometry.rest);
    d.MixSpan(0, geometry.normals);
    d.MixSpan(1, geometry.ids);
    d.MixSpan(0, geometry.subsetFaces);
    d.Mix(1, geometry.isSubset ? 1u : 0u);
    for (int row = 0; row != 4; ++row) for (int col = 0; col != 4; ++col) {
        uint64_t bits = 0;
        double const component = geometry.worldMatrix[row][col];
        std::memcpy(&bits, &component, sizeof(bits));
        d.Mix(0, bits);
    }
    d.Mix(1, geometry.generation);
}

}  // namespace

void UsdGenRegisterExprInputKinds(UsdGenRegistrar &r)
{
    if (gExprInput.load(std::memory_order_acquire)) return;
    ExprInputState *fresh = new ExprInputState;   // startup; never freed
    {
        UsdGenCompileTokenScope scope;
        static constexpr char const *kExpression[kUsdGenExpressionValues] = {
            "usdGen:source", "usdGen:outputs", "usdGen:inputs"};
        for (size_t i = 0; i < kUsdGenExpressionValues; ++i)
            fresh->expressionAttributes[i] = UsdGenInternToken(kExpression[i]);
        using S = UsdGenScalar;
        struct Row { UsdGenGeometryInputSlot slot; char const *channel, *attribute; S scalar; uint8_t arity; };
        static constexpr Row kRows[] = {
            {kUsdGenGeometryInputCounts, "counts", "faceVertexCounts", S::I32, 1},
            {kUsdGenGeometryInputIndices, "indices", "faceVertexIndices", S::I32, 1},
            {kUsdGenGeometryInputPoints, "points", "points", S::F32, 3},
            {kUsdGenGeometryInputRest, "rest", "rest", S::F32, 3},
            {kUsdGenGeometryInputNormals, "normals", "normals", S::F32, 3},
            {kUsdGenGeometryInputIds, "ids", "ids", S::U64, 1},
            {kUsdGenGeometryInputSubsetFaces, "subsetFaces", "indices", S::I32, 1},
        };
        static_assert(sizeof(kRows) / sizeof(kRows[0]) == kUsdGenGeometryInputSlots, "one row per slot");
        for (Row const &row : kRows) {
            UsdGenChannelDesc &d = fresh->geometryChannels[row.slot];
            d.name = UsdGenInternToken(row.channel);
            d.scalar = row.scalar;
            d.arity = row.arity;
            d.flags = row.arity > 1 ? uint8_t(kUsdGenChannelPacked) : uint8_t(0);
            UsdGenInputSlot &in = fresh->geometrySlots[row.slot];
            in.attribute = UsdGenInternToken(row.attribute);
            in.channel = row.slot;
        }
        static constexpr char const *kGeometry[kUsdGenGeometryValues] = {
            "typeName", "usdGen:isSubset", "worldMatrix"};
        for (size_t i = 0; i < kUsdGenGeometryValues; ++i)
            fresh->geometryAttributes[i] = UsdGenInternToken(kGeometry[i]);
    }
    fresh->expressionLayout.levels = 1;
    fresh->expression.attributes =
        UsdGenSpan<const UsdGenToken>(fresh->expressionAttributes, kUsdGenExpressionValues);
    fresh->geometryLayout.levels = 1;
    fresh->geometryLayout.channels =
        UsdGenSpan<const UsdGenChannelDesc>(fresh->geometryChannels, kUsdGenGeometryInputSlots);
    fresh->geometry.slots = UsdGenSpan<const UsdGenInputSlot>(fresh->geometrySlots, kUsdGenGeometryInputSlots);
    fresh->geometry.attributes = UsdGenSpan<const UsdGenToken>(fresh->geometryAttributes, kUsdGenGeometryValues);
    fresh->expression.digest = &DigestExpression;
    fresh->geometry.digest = &DigestGeometry;

    UsdGenTypeDef def;
    def.name = "UsdGenExpressionInput";
    def.layout = &fresh->expressionLayout;
    def.input = &fresh->expression;
    UsdGenTypeId const expression = r.RegisterType(def);
    def.name = "UsdGenGeometryInput";
    def.layout = &fresh->geometryLayout;
    def.input = &fresh->geometry;
    UsdGenTypeId const geometry = r.RegisterType(def);
    if (expression == UsdGenTypeId::Invalid || geometry == UsdGenTypeId::Invalid) {
        delete fresh;
        return;
    }
    fresh->expressionKind = expression;
    fresh->expressionLayout.primitive = expression;
    fresh->expression.kind = expression;
    fresh->geometryKind = geometry;
    fresh->geometryLayout.primitive = geometry;
    fresh->geometry.kind = geometry;
    ExprInputState *expected = nullptr;
    gExprInput.compare_exchange_strong(expected, fresh, std::memory_order_acq_rel,
                                       std::memory_order_acquire);
}

UsdGenTypeId UsdGenExpressionInputKind() noexcept
{
    ExprInputState const *s = State();
    return s ? s->expressionKind : UsdGenTypeId::Invalid;
}

UsdGenTypeId UsdGenGeometryInputKind() noexcept
{
    ExprInputState const *s = State();
    return s ? s->geometryKind : UsdGenTypeId::Invalid;
}

UsdGenSpan<const UsdGenInputRecord> UsdGenExpressionInputs(UsdGenGraphSpec const &spec) noexcept
{
    return spec.Inputs(UsdGenExpressionInputKind());
}

UsdGenSpan<const UsdGenInputRecord> UsdGenGeometryInputs(UsdGenGraphSpec const &spec) noexcept
{
    return spec.Inputs(UsdGenGeometryInputKind());
}

UsdGenInputRecord const *UsdGenFindExpressionInput(UsdGenGraphSpec const &spec, UsdGenNodeKey key) noexcept
{
    return spec.FindInput(UsdGenExpressionInputKind(), key);
}

UsdGenInputRecord const *UsdGenFindGeometryInput(UsdGenGraphSpec const &spec, UsdGenNodeKey key) noexcept
{
    return spec.FindInput(UsdGenGeometryInputKind(), key);
}

UsdGenExpressionInputView UsdGenExpressionInputView::Of(UsdGenInputRecord const &record) noexcept
{
    UsdGenExpressionInputView v;
    v.key = record.key;
    if (VtValue const *source = record.Value(kUsdGenExpressionValueSource);
        source && source->IsHolding<std::string>()) {
        v.sourceHolder = *source;
        v.source = v.sourceHolder.UncheckedGet<std::string>();
    }
    if (VtValue const *outputs = record.Value(kUsdGenExpressionValueOutputs);
        outputs && outputs->IsHolding<Outputs>()) {
        v.outputsHolder = *outputs;
        Outputs const &o = v.outputsHolder.UncheckedGet<Outputs>();
        v.outputs = {o.data(), o.size()};
    }
    if (VtValue const *inputs = record.Value(kUsdGenExpressionValueInputs);
        inputs && inputs->IsHolding<Slots>()) {
        v.inputsHolder = *inputs;
        Slots const &s = v.inputsHolder.UncheckedGet<Slots>();
        v.inputs = {s.data(), s.size()};
    }
    return v;
}

UsdGenExpressionInputView UsdGenExpressionInputView::Of(UsdGenExpressionData const &d,
                                                        UsdGenNodeKey key) noexcept
{
    UsdGenExpressionInputView v;
    v.key = key;
    v.source = d.source;
    v.outputs = {d.outputs.data(), d.outputs.size()};
    v.inputs = {d.inputs.data(), d.inputs.size()};
    return v;
}

UsdGenGeometryInputView UsdGenGeometryInputView::Of(UsdGenInputRecord const &record) noexcept
{
    UsdGenGeometryInputView v;
    v.key = record.key;
    v.kind = UsdGenGeometryKind(uint8_t(record.Get<int>(kUsdGenGeometryValueKind,
                                                        int(UsdGenGeometryKind::Points))));
    v.counts = record.Column<int>(kUsdGenGeometryInputCounts);
    v.indices = record.Column<int>(kUsdGenGeometryInputIndices);
    v.points = record.Column<GfVec3f>(kUsdGenGeometryInputPoints);
    v.rest = record.Column<GfVec3f>(kUsdGenGeometryInputRest);
    v.normals = record.Column<GfVec3f>(kUsdGenGeometryInputNormals);
    v.ids = record.Column<uint64_t>(kUsdGenGeometryInputIds);
    v.subsetFaces = record.Column<int>(kUsdGenGeometryInputSubsetFaces);
    v.isSubset = record.Get<bool>(kUsdGenGeometryValueIsSubset, false);
    v.worldMatrix = record.Get<GfMatrix4d>(kUsdGenGeometryValueWorldMatrix, kIdentity);
    v.generation = record.generation;
    v.data = record.data;
    return v;
}

UsdGenGeometryInputView UsdGenGeometryInputView::Of(UsdGenGeometryData const &d,
                                                    UsdGenNodeKey key) noexcept
{
    UsdGenGeometryInputView v;
    v.key = key;
    v.kind = d.kind;
    v.counts = Arr(d.counts);
    v.indices = Arr(d.indices);
    v.points = Arr(d.points);
    v.rest = Arr(d.rest);
    v.normals = Arr(d.normals);
    v.ids = Arr(d.ids);
    v.subsetFaces = Arr(d.subsetFaces);
    v.isSubset = d.isSubset;
    v.worldMatrix = d.worldMatrix;
    v.generation = d.generation;
    return v;
}

UsdGenInputRecord UsdGenExpressionRecord(UsdGenExpressionData const &d, UsdGenNodeKey key,
                                         TfToken const &label)
{
    UsdGenInputRecordWriter w(UsdGenExpressionInputKind());
    w.Value(kUsdGenExpressionValueSource, VtValue(d.source));
    w.Value(kUsdGenExpressionValueOutputs, VtValue(d.outputs));
    w.Value(kUsdGenExpressionValueInputs, VtValue(d.inputs));
    return w.Finish(key, UsdGenNodeKey{}, label, 0);
}

UsdGenInputRecord UsdGenGeometryRecord(UsdGenGeometryData const &d, UsdGenNodeKey key,
                                       TfToken const &label, UsdGenInputColumnCache *cache)
{
    UsdGenInputRecordWriter w(UsdGenGeometryInputKind());
    w.UseCache(cache, key);
    w.Array(kUsdGenGeometryInputCounts, d.counts);
    w.Array(kUsdGenGeometryInputIndices, d.indices);
    w.Array(kUsdGenGeometryInputPoints, d.points);
    w.Array(kUsdGenGeometryInputRest, d.rest);
    w.Array(kUsdGenGeometryInputNormals, d.normals);
    w.Array(kUsdGenGeometryInputIds, d.ids);
    w.Array(kUsdGenGeometryInputSubsetFaces, d.subsetFaces);
    w.Value(kUsdGenGeometryValueKind, VtValue(int(d.kind)));
    w.Value(kUsdGenGeometryValueIsSubset, VtValue(d.isSubset));
    w.Value(kUsdGenGeometryValueWorldMatrix, VtValue(d.worldMatrix));
    return w.Finish(key, UsdGenNodeKey{}, label, d.generation);
}

}  // namespace usdGen
