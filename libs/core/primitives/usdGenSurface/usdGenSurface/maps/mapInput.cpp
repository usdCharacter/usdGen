// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenSurface/maps/mapInput.h"
#include "usdGenSurface/imagePayload.h"

#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/token.h"

#include <atomic>
#include <utility>

namespace usdGen {

namespace {

struct MapInputState {
    UsdGenChannelDesc channels[kUsdGenMapInputSlots];
    UsdGenInputSlot slots[kUsdGenMapInputSlots];
    UsdGenToken attributes[kUsdGenMapValues];
    UsdGenElementLayout layout;
    UsdGenInputDesc input;
    UsdGenTypeId kind = UsdGenTypeId::Invalid;
};

std::atomic<MapInputState *> gMapInput{nullptr};

MapInputState const *State() noexcept
{
    if (MapInputState const *s = gMapInput.load(std::memory_order_acquire)) return s;
    (void)UsdGenOpRegistry::Get();   // its constructor registers the built-ins
    return gMapInput.load(std::memory_order_acquire);
}

using ImagePayloadPtr = std::shared_ptr<const UsdGenImagePayload>;

/// Intake: the map type's handle, once per compile.
void ResolveTypeId(UsdGenInputRecord &record)
{
    if (record.values.size() <= kUsdGenMapValueTypeId) return;
    TfToken const &type = record.Get<TfToken>(kUsdGenMapValueType, TfToken());
    record.values[kUsdGenMapValueTypeId].value =
        VtValue(int(UsdGenMutableTypeRegistry().Find(type)));
}

}  // namespace

UsdGenTypeId UsdGenRegisterMapInputKind(UsdGenRegistrar &r)
{
    if (MapInputState const *s = gMapInput.load(std::memory_order_acquire)) return s->kind;
    MapInputState *fresh = new MapInputState;   // startup; never freed
    {
        UsdGenCompileTokenScope scope;
        UsdGenChannelDesc &d = fresh->channels[kUsdGenMapInputPaintValues];
        d.name = UsdGenInternToken("paintValues");
        d.scalar = UsdGenScalar::F32;
        UsdGenInputSlot &in = fresh->slots[kUsdGenMapInputPaintValues];
        in.attribute = UsdGenInternToken("usdGen:map:paintValues");
        in.channel = kUsdGenMapInputPaintValues;
        static constexpr char const *kValues[kUsdGenMapValues] = {
            "typeName", "usdGen:map:resolvedAssetPath", "usdGen:map:primvar",
            "usdGen:map:primvarInterpolation", "usdGen:map:imagePayload", "usdGen:map:typeId"};
        for (size_t i = 0; i < kUsdGenMapValues; ++i)
            fresh->attributes[i] = UsdGenInternToken(kValues[i]);
        fresh->input.dynamicPrimvarPrefix = UsdGenInternToken("usdGen:");
    }
    fresh->layout.levels = 1;
    fresh->layout.channels = UsdGenSpan<const UsdGenChannelDesc>(fresh->channels, kUsdGenMapInputSlots);
    fresh->input.slots = UsdGenSpan<const UsdGenInputSlot>(fresh->slots, kUsdGenMapInputSlots);
    fresh->input.attributes = UsdGenSpan<const UsdGenToken>(fresh->attributes, kUsdGenMapValues);
    fresh->input.intake = &ResolveTypeId;
    UsdGenTypeDef def;
    def.name = "UsdGenMapInput";
    def.layout = &fresh->layout;
    def.input = &fresh->input;
    UsdGenTypeId const id = r.RegisterType(def);
    if (id == UsdGenTypeId::Invalid) {
        delete fresh;
        MapInputState const *s = gMapInput.load(std::memory_order_acquire);
        return s ? s->kind : UsdGenTypeId::Invalid;
    }
    fresh->kind = id;
    fresh->layout.primitive = id;
    fresh->input.kind = id;
    MapInputState *expected = nullptr;
    gMapInput.compare_exchange_strong(expected, fresh, std::memory_order_acq_rel,
                                      std::memory_order_acquire);
    return id;
}

UsdGenTypeId UsdGenMapInputKind() noexcept
{
    MapInputState const *s = State();
    return s ? s->kind : UsdGenTypeId::Invalid;
}

UsdGenSpan<const UsdGenInputRecord> UsdGenMapInputs(UsdGenGraphSpec const &spec) noexcept
{
    return spec.Inputs(UsdGenMapInputKind());
}

UsdGenInputRecord const *UsdGenFindMapInput(UsdGenGraphSpec const &spec, UsdGenNodeKey key) noexcept
{
    return spec.FindInput(UsdGenMapInputKind(), key);
}

UsdGenMapInputView UsdGenMapInputView::Of(UsdGenInputRecord const &record) noexcept
{
    UsdGenMapInputView v;
    v.key = record.key;
    v.paintSurface = record.owner;
    v.type = record.Get<TfToken>(kUsdGenMapValueType, TfToken());
    if (VtValue const *path = record.Value(kUsdGenMapValueResolvedAssetPath);
        path && path->IsHolding<std::string>()) {
        v.resolvedAssetPathHolder = *path;
        v.resolvedAssetPath = v.resolvedAssetPathHolder.UncheckedGet<std::string>();
    }
    v.textureGeneration = record.generation;
    v.paintPrimvar = record.Get<TfToken>(kUsdGenMapValuePaintPrimvar, TfToken());
    v.paintInterpolation = record.Get<TfToken>(kUsdGenMapValuePaintInterpolation, TfToken());
    v.paintValues = record.Column<float>(kUsdGenMapInputPaintValues);
    if (record.values.size() > kUsdGenMapValues)
        v.params = {record.values.data() + kUsdGenMapValues, record.values.size() - kUsdGenMapValues};
    if (VtValue const *image = record.Value(kUsdGenMapValueImagePayload);
        image && image->IsHolding<ImagePayloadPtr>())
        v.imagePayload = image->UncheckedGet<ImagePayloadPtr>();
    v.typeId = UsdGenTypeId(uint16_t(record.Get<int>(kUsdGenMapValueTypeId,
                                                     int(UsdGenTypeId::Invalid))));
    v.data = record.data;
    return v;
}

UsdGenMapInputView UsdGenMapInputView::Of(UsdGenMapData const &d, UsdGenNodeKey key,
                                          UsdGenNodeKey paintSurface) noexcept
{
    UsdGenMapInputView v;
    v.key = key;
    v.paintSurface = paintSurface;
    v.type = d.type;
    v.resolvedAssetPath = d.resolvedAssetPath;
    v.textureGeneration = d.textureGeneration;
    v.paintPrimvar = d.paintPrimvar;
    v.paintInterpolation = d.paintInterpolation;
    v.paintValues = {d.paintValues.cdata(), d.paintValues.size()};
    v.params = {d.params.data(), d.params.size()};
    v.imagePayload = d.imagePayload;
    v.typeId = d.typeId;
    return v;
}

UsdGenInputRecord UsdGenMapRecord(UsdGenMapData const &d, UsdGenNodeKey key,
                                  UsdGenNodeKey paintSurface, TfToken const &label,
                                  UsdGenInputColumnCache *cache)
{
    UsdGenInputRecordWriter w(UsdGenMapInputKind());
    w.UseCache(cache, key);
    w.Array(kUsdGenMapInputPaintValues, d.paintValues);
    w.Value(kUsdGenMapValueType, VtValue(d.type));
    w.Value(kUsdGenMapValueResolvedAssetPath, VtValue(d.resolvedAssetPath));
    w.Value(kUsdGenMapValuePaintPrimvar, VtValue(d.paintPrimvar));
    w.Value(kUsdGenMapValuePaintInterpolation, VtValue(d.paintInterpolation));
    w.Value(kUsdGenMapValueImagePayload, VtValue(d.imagePayload));
    w.Value(kUsdGenMapValueTypeId, VtValue(int(d.typeId)));
    for (UsdGenParamValue const &param : d.params) w.Extra(param);
    return w.Finish(key, paintSurface, label, d.textureGeneration);
}

void UsdGenCopyMapData(UsdGenMapInputView const &v, UsdGenMapData *out)
{
    out->type = v.type;
    out->resolvedAssetPath.assign(v.resolvedAssetPath.data(), v.resolvedAssetPath.size());
    out->textureGeneration = v.textureGeneration;
    out->paintPrimvar = v.paintPrimvar;
    out->paintInterpolation = v.paintInterpolation;
    out->paintValues.assign(v.paintValues.begin(), v.paintValues.end());
    out->params.assign(v.params.begin(), v.params.end());
    out->imagePayload = v.imagePayload;
    out->typeId = v.typeId;
}

}  // namespace usdGen
