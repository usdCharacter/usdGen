// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenSurface/maps/mapInput.h"
#include "usdGen/api/value.h"
#include "usdGen/api/token.h"
#include "usdGen/api/span.h"
#include "usdGenSurface/imagePayload.h"

#include "usdGen/api/ext/opRegistry.h"
#include "usdGen/api/ext/registrar.h"
#include "usdGen/api/ext/tokenScope.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

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

/// The record's image payload value: a Boxed payload of usdGenSurface
/// (pointer identity, as the shared_ptr it holds).
struct ImagePayloadBox final : UsdGenBoxed, UsdGenPooled {
    static constexpr uint64_t kBoxTag = UsdGenBoxTag("usdGenSurface.imagePayload");
    explicit ImagePayloadBox(ImagePayloadPtr p) : payload(std::move(p)) {}
    uint64_t Hash() const noexcept override { return uint64_t(reinterpret_cast<uintptr_t>(payload.get())); }
    bool Equal(UsdGenBoxed const &o) const noexcept override
    {
        return payload == static_cast<ImagePayloadBox const &>(o).payload;
    }
    ImagePayloadPtr payload;
};

/// Intake: the map type's handle, once per compile.
void ResolveTypeId(UsdGenInputRecord &record)
{
    if (record.values.size() <= kUsdGenMapValueTypeId) return;
    UsdGenToken const &type = record.Get<UsdGenToken>(kUsdGenMapValueType, UsdGenToken());
    record.values[kUsdGenMapValueTypeId].value =
        UsdGenValue(int(UsdGenMutableTypeRegistry().Find(type)));
}

// FNV-1a 64-bit (the compiler's map value identity, moved here).
uint64_t Fnv1a(uint64_t h, void const *data, size_t n)
{
    auto const *p = static_cast<unsigned char const *>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

uint64_t Fnv1aI64(uint64_t h, int64_t v)
{
    uint64_t u = static_cast<uint64_t>(v);
    return Fnv1a(h, &u, sizeof(u));
}

uint64_t Fnv1aCstr(uint64_t h, char const *s)
{
    for (auto const *p = reinterpret_cast<unsigned char const *>(s); *p; ++p) {
        h ^= *p; h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t Fnv1aText(uint64_t h, std::string_view s)
{
    for (char c : s) {   // as Fnv1aCstr: stops at a NUL
        if (!c) break;
        h ^= static_cast<unsigned char>(c); h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t Fnv1aToken(uint64_t h, UsdGenToken const &t)
{
    return Fnv1aI64(Fnv1aCstr(h, t.GetText()), static_cast<int64_t>(t.Hash()));
}

uint64_t MapValueIdentity(UsdGenMapInputView const &map)
{
    uint64_t h = Fnv1aI64(1469598103934665603ULL, static_cast<int64_t>(map.key.value));
    h = Fnv1aToken(h, map.type);
    h = Fnv1aText(h, map.resolvedAssetPath);
    h = Fnv1aI64(h, static_cast<int64_t>(map.textureGeneration));
    // A UsdGenPaintMap's payload is the surface primvar snapshot, not a
    // file: a repaint changes only paintValues, so they are the identity a
    // direct map-slot consumer re-captures on (DigestRecord folds the same
    // fields into the execution-cache key).
    if (!map.paintValues.empty() || map.paintSurface.IsValid()) {
        h = Fnv1aI64(h, static_cast<int64_t>(map.paintSurface.value));
        h = Fnv1aToken(h, map.paintPrimvar);
        h = Fnv1aToken(h, map.paintInterpolation);
        h = Fnv1aI64(h, static_cast<int64_t>(map.paintValues.size()));
        for (float v : map.paintValues) {
            uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            h = Fnv1aI64(h, static_cast<int64_t>(bits));
        }
    }
    std::vector<UsdGenParamValue const *> params;
    params.reserve(map.params.size());
    for (UsdGenParamValue const &param : map.params) params.push_back(&param);
    std::sort(params.begin(), params.end(), [](auto const *a, auto const *b) {
        if (a->name != b->name) return a->name < b->name;
        if (a->value.GetHash() != b->value.GetHash())
            return a->value.GetHash() < b->value.GetHash();
        return a->animated < b->animated;
    });
    for (UsdGenParamValue const *param : params) {
        h = Fnv1aToken(h, param->name);
        h = Fnv1aI64(h, static_cast<int64_t>(param->value.GetHash()));
        h = Fnv1aI64(h, param->animated ? 1 : 0);
    }
    return h;
}

void ResolveMap(UsdGenInputRecord const &record, UsdGenResolvedMapValue *value)
{
    UsdGenMapInputView const map = UsdGenMapInputView::Of(record);
    value->type = map.type;
    value->resolvedAssetPath = UsdGenText(std::string_view(map.resolvedAssetPath.data(), map.resolvedAssetPath.size()));
    value->textureGeneration = map.textureGeneration;
    value->identity = MapValueIdentity(map);
}

bool PaintSource(UsdGenInputRecord const &record, UsdGenNodeKey *surface, UsdGenToken *primvar)
{
    UsdGenMapInputView const map = UsdGenMapInputView::Of(record);
    if (!UsdGenMapHas(map, kUsdGenTypeMapPaintSnapshot)) return false;
    *surface = map.paintSurface;
    *primvar = map.paintPrimvar;
    return true;
}

/// Execution cache key terms (moved from the session cooker); no layout terms.
void DigestRecord(UsdGenInputRecord const &record, UsdGenInputDigest &d, bool layout)
{
    if (layout) return;
    UsdGenMapInputView const map = UsdGenMapInputView::Of(record);
    d.Mix(0, map.key.value);
    d.MixText(1, map.type.GetString());
    d.MixText(0, map.resolvedAssetPath);
    d.Mix(1, map.textureGeneration);
    // A paint map's snapshot is capture content, not identity: the
    // path, type and generation never move under a repaint, so the
    // values themselves join the key or a length stroke would hit the
    // pre-stroke cache entry and the groom would freeze (the density
    // multiplier's row is the same contract).
    if (UsdGenMapHas(map, kUsdGenTypeMapPaintSnapshot))
        d.MixSpan(0, map.paintValues);
    d.Mix(0, static_cast<uint64_t>(map.params.size()));
    for (UsdGenParamValue const &param : map.params) {
        d.MixText(1, param.name.GetString());
        d.Mix(0, param.value.GetHash());
        d.Mix(1, static_cast<uint64_t>(param.animated));
    }
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
    fresh->input.role = UsdGenInputRole::Map;
    fresh->input.digest = &DigestRecord;
    fresh->input.resolveMap = &ResolveMap;
    fresh->input.paintSource = &PaintSource;
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
    v.type = record.Get<UsdGenToken>(kUsdGenMapValueType, UsdGenToken());
    if (UsdGenValue const *path = record.Value(kUsdGenMapValueResolvedAssetPath);
        path && path->IsHolding<std::string>()) {
        v.resolvedAssetPathHolder = *path;
        v.resolvedAssetPath = v.resolvedAssetPathHolder.UncheckedGet<std::string>();
    }
    v.textureGeneration = record.generation;
    v.paintPrimvar = record.Get<UsdGenToken>(kUsdGenMapValuePaintPrimvar, UsdGenToken());
    v.paintInterpolation = record.Get<UsdGenToken>(kUsdGenMapValuePaintInterpolation, UsdGenToken());
    v.paintValues = record.Column<float>(kUsdGenMapInputPaintValues);
    if (record.values.size() > kUsdGenMapValues)
        v.params = {record.values.data() + kUsdGenMapValues, record.values.size() - kUsdGenMapValues};
    if (UsdGenValue const *image = record.Value(kUsdGenMapValueImagePayload);
        image && image->GetBoxed<ImagePayloadBox>())
        v.imagePayload = image->GetBoxed<ImagePayloadBox>()->payload;
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
                                  UsdGenNodeKey paintSurface, UsdGenText const &label,
                                  UsdGenInputColumnCache *cache)
{
    UsdGenInputRecordWriter w(UsdGenMapInputKind());
    w.UseCache(cache, key);
    w.Array(kUsdGenMapInputPaintValues, d.paintValues);
    w.Value(kUsdGenMapValueType, UsdGenValue(d.type));
    w.Value(kUsdGenMapValueResolvedAssetPath, UsdGenValue(d.resolvedAssetPath));
    w.Value(kUsdGenMapValuePaintPrimvar, UsdGenValue(d.paintPrimvar));
    w.Value(kUsdGenMapValuePaintInterpolation, UsdGenValue(d.paintInterpolation));
    w.Value(kUsdGenMapValueImagePayload,
            UsdGenValue::Box(ImagePayloadBox::kBoxTag, UsdGenRef<const UsdGenBoxed>(new ImagePayloadBox(d.imagePayload))));
    w.Value(kUsdGenMapValueTypeId, UsdGenValue(int(d.typeId)));
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
