// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenSurface/types/rootSet.h"
#include "usdGenMath/math.h"
#include "usdGen/api/span.h"

#include "usdGen/api/writeSet.h"
#include "usdGen/api/ext/opRegistry.h"
#include "usdGen/api/ext/registrar.h"
#include "usdGen/api/ext/tokenScope.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <vector>

namespace usdGen {

namespace {

/// One per process, published once by the registration that registered the type.
struct RootSetState {
    UsdGenChannelDesc statics[kUsdGenRootStaticSlots];
    UsdGenElementLayout layout;
    UsdGenTypeId type = UsdGenTypeId::Invalid;
};

std::atomic<RootSetState *> gRootSet{nullptr};
UsdGenElementLayout const kNoLayout{};

RootSetState const *State() noexcept
{
    if (RootSetState const *s = gRootSet.load(std::memory_order_acquire)) return s;
    (void)UsdGenOpRegistry::Get();   // its constructor registers the built-ins
    return gRootSet.load(std::memory_order_acquire);
}

UsdGenChannelDesc Static(char const *name, UsdGenScalar scalar, uint8_t arity,
                         UsdGenChannelRole role, uint8_t component = 0, uint8_t flags = 0)
{
    UsdGenChannelDesc d;
    d.name = UsdGenInternToken(name);
    d.scalar = scalar;
    d.arity = arity;
    d.level = 1;
    d.role = role;
    d.component = component;
    d.flags = flags;
    return d;
}

// Same group arithmetic as the curve tiles: groupTarget groups, clamped to [32, 256].
constexpr uint32_t kGroupMin = 32, kGroupMax = 256;

struct GroupLayout {
    uint32_t chunks = 0, perGroup = 0, groups = 0;
    uint32_t Boundary(uint32_t g) const noexcept
    {
        int64_t const packed = int64_t(g) * int64_t(perGroup);
        int64_t const tail = int64_t(chunks) - int64_t(groups - g);
        return uint32_t(std::min(packed, tail));
    }
    uint16_t GroupOf(uint32_t chunk) const noexcept
    {
        uint32_t lo = 0, hi = groups;
        while (hi - lo > 1) {
            uint32_t const mid = lo + (hi - lo) / 2;
            if (Boundary(mid) <= chunk) lo = mid;
            else hi = mid;
        }
        return uint16_t(lo);
    }
};

GroupLayout Groups(uint32_t chunks, uint32_t groupTarget) noexcept
{
    GroupLayout l;
    l.chunks = chunks;
    if (!chunks) return l;
    uint32_t const target = std::clamp(groupTarget, kGroupMin, kGroupMax);
    l.perGroup = std::max(1u, (chunks + target - 1) / target);
    uint32_t n = (chunks + l.perGroup - 1) / l.perGroup;
    n = std::max(kGroupMin, std::min(kGroupMax, n));
    l.groups = std::min(chunks, n);
    return l;
}

/// Root slots of 'table': the RootSet's own, else the static slots with the
/// root names and the same scalar and arity. False without px/py/pz.
bool ResolveSlots(UsdGenChannelTable const &table, UsdGenSlot out[kUsdGenRootStaticSlots]) noexcept
{
    RootSetState const *s = State();
    if (!s) return false;
    if (&table.Layout() == &s->layout) {
        for (uint32_t r = 0; r < kUsdGenRootStaticSlots; ++r) out[r] = UsdGenSlot(r);
        return true;
    }
    for (uint32_t r = 0; r < kUsdGenRootStaticSlots; ++r) out[r] = kUsdGenNoSlot;
    for (UsdGenSlot i = 0; i < table.StaticCount(); ++i) {
        UsdGenChannelDesc const &d = table[i];
        for (uint32_t r = 0; r < kUsdGenRootStaticSlots; ++r) {
            UsdGenChannelDesc const &want = s->statics[r];
            if (d.name == want.name && d.scalar == want.scalar && d.arity == want.arity) {
                out[r] = i;
                break;
            }
        }
    }
    return out[kUsdGenRootPx] != kUsdGenNoSlot && out[kUsdGenRootPy] != kUsdGenNoSlot &&
           out[kUsdGenRootPz] != kUsdGenNoSlot;
}

UsdGenTypeDef RootSetDef() noexcept
{
    UsdGenTypeDef def;
    def.name = "UsdGenRootSet";
    return def;
}

template <class T>
UsdGenSpan<const T> Col(UsdGenElementSetView const &s, UsdGenSlot slot, uint32_t perItem = 1) noexcept
{
    if (slot == kUsdGenNoSlot || !s.Has(slot)) return {};
    return {s.Column<T>(slot), size_t(s.Values(slot) / perItem)};
}

}  // namespace

UsdGenTypeId UsdGenRegisterRootSetType(UsdGenRegistrar &r)
{
    if (RootSetState const *s = gRootSet.load(std::memory_order_acquire)) return s->type;
    RootSetState *fresh = new RootSetState;   // startup; never freed (the registry points at it)
    {
        UsdGenCompileTokenScope scope;
        using S = UsdGenScalar;
        using R = UsdGenChannelRole;
        UsdGenChannelDesc *d = fresh->statics;
        // Names match the curve layout's per-element channels.
        d[kUsdGenRootPx] = Static("__px", S::F32, 1, R::Position, 0);
        d[kUsdGenRootPy] = Static("__py", S::F32, 1, R::Position, 1);
        d[kUsdGenRootPz] = Static("__pz", S::F32, 1, R::Position, 2);
        d[kUsdGenRootId] = Static("__curveId", S::U64, 1, R::Id);
        d[kUsdGenRootPrim] = Static("__rootPrim", S::I32, 1, R::Other);
        d[kUsdGenRootUV] = Static("__rootUV", S::F32, 2, R::Uv, 0, kUsdGenChannelPacked);
        d[kUsdGenRootT] = Static("__rootT", S::F32, 3, R::Frame, 0, kUsdGenChannelPacked);
        d[kUsdGenRootN] = Static("__rootN", S::F32, 3, R::Frame, 1, kUsdGenChannelPacked);
        d[kUsdGenRootB] = Static("__rootB", S::F32, 3, R::Frame, 2, kUsdGenChannelPacked);
    }
    fresh->layout.levels = 1;
    fresh->layout.channels = UsdGenSpan<const UsdGenChannelDesc>(fresh->statics, kUsdGenRootStaticSlots);
    fresh->layout.partition = &UsdGenRootSetPartition;
    fresh->layout.groups = &UsdGenRootSetGroups;
    UsdGenTypeDef def = RootSetDef();
    // Explicit conversion for graphs that read roots as one-CV curves
    // (Scatter -> Noise -> Width -> GuideInterpolate in the examples).
    def.traits = kUsdGenTypeConvertsToDefault;
    def.layout = &fresh->layout;
    UsdGenTypeId const id = r.RegisterType(def);
    if (id == UsdGenTypeId::Invalid) {
        delete fresh;
        RootSetState const *s = gRootSet.load(std::memory_order_acquire);
        return s ? s->type : UsdGenTypeId::Invalid;
    }
    fresh->type = id;
    fresh->layout.primitive = id;
    RootSetState *expected = nullptr;
    gRootSet.compare_exchange_strong(expected, fresh, std::memory_order_acq_rel,
                                     std::memory_order_acquire);
    return id;
}

char const *UsdGenRootSetTypeName() noexcept
{
    return RootSetDef().name;
}

UsdGenTypeId UsdGenRootSetType() noexcept
{
    RootSetState const *s = State();
    return s ? s->type : UsdGenTypeId::Invalid;
}

UsdGenElementLayout const &UsdGenRootSetLayout() noexcept
{
    RootSetState const *s = State();
    return s ? s->layout : kNoLayout;
}

uint32_t UsdGenRootSetPartition(uint32_t elements, UsdGenElementSetView const &,
                                UsdGenPartitionParams const &params, UsdGenSpan<UsdGenChunkRange> out)
{
    uint32_t const size = std::max(1u, params.chunkSize);
    uint32_t const chunks = elements ? uint32_t((uint64_t(elements) + size - 1) / size) : 0u;
    GroupLayout const groups = Groups(chunks, params.groupTarget);
    size_t const n = std::min<size_t>(chunks, out.size());
    for (size_t c = 0; c < n; ++c) {
        UsdGenChunkRange r;
        r.first[0] = uint32_t(c) * size;
        r.count[0] = std::min(size, elements - r.first[0]);
        r.live = r.count[0];
        r.group = groups.GroupOf(uint32_t(c));
        out[c] = r;
    }
    return chunks;
}

uint32_t UsdGenRootSetGroups(uint32_t chunks, UsdGenPartitionParams const &params,
                             UsdGenSpan<UsdGenChunkGroup> out)
{
    GroupLayout const groups = Groups(chunks, params.groupTarget);
    size_t const n = std::min<size_t>(groups.groups, out.size());
    for (size_t g = 0; g < n; ++g) {
        UsdGenChunkGroup group;
        group.id = uint16_t(g);
        group.firstChunk = groups.Boundary(uint32_t(g));
        group.chunkCount = groups.Boundary(uint32_t(g) + 1) - group.firstChunk;
        out[g] = group;
    }
    return groups.groups;
}

// ---- views -----------------------------------------------------------------------

UsdGenRootSetView UsdGenRootSetView::Of(UsdGenElementSetView const &s) noexcept
{
    UsdGenRootSetView v;
    UsdGenSlot slots[kUsdGenRootStaticSlots];
    if (!s.channels || !s.columns || !s.values || !ResolveSlots(*s.channels, slots)) return v;
    uint32_t const roots = s.count[1];
    // One position per element: values == roots, uniform at a deeper level.
    for (uint32_t r = kUsdGenRootPx; r <= kUsdGenRootPz; ++r) {
        UsdGenSlot const slot = slots[r];
        uint8_t const level = (*s.channels)[slot].level;
        if (level < 1 || level > kUsdGenMaxLevels) return v;
        if (level >= 2 && s.offsets[level]) return v;
        if (roots && (!s.Has(slot) || s.Values(slot) != roots)) return v;
    }
    v._set = s;
    v.px = Col<float>(s, slots[kUsdGenRootPx]);
    v.py = Col<float>(s, slots[kUsdGenRootPy]);
    v.pz = Col<float>(s, slots[kUsdGenRootPz]);
    v.curveId = Col<uint64_t>(s, slots[kUsdGenRootId]);
    v.rootPrim = Col<int>(s, slots[kUsdGenRootPrim]);
    v.rootUV = Col<UsdGenVec2f>(s, slots[kUsdGenRootUV], 2);
    v.rootT = Col<UsdGenVec3f>(s, slots[kUsdGenRootT], 3);
    v.rootN = Col<UsdGenVec3f>(s, slots[kUsdGenRootN], 3);
    v.rootB = Col<UsdGenVec3f>(s, slots[kUsdGenRootB], 3);
    v.totalCurves = roots;
    v.topologyVersion = s.topologyVersion;
    v.valueVersion = s.valueVersion;
    v.valid = true;
    return v;
}

uint32_t UsdGenRootSetView::PlaneCount() const noexcept
{
    return _set.channels ? uint32_t(_set.channels->DynamicRun(1).size()) : 0u;
}

UsdGenRootPlaneView UsdGenRootSetView::Plane(uint32_t k) const noexcept
{
    UsdGenRootPlaneView p;
    if (!_set.channels) return p;
    UsdGenSpan<const UsdGenSlot> const run = _set.channels->DynamicRun(1);
    if (k >= run.size()) return p;
    UsdGenSlot const slot = run[k];
    UsdGenChannelDesc const &d = (*_set.channels)[slot];
    p.name = d.name;
    p.scalar = d.scalar;
    p.arity = d.arity;
    p.level = d.level;
    if (d.scalar == UsdGenScalar::F32) p.f = Col<float>(_set, slot);
    else if (d.scalar == UsdGenScalar::I32) p.i = Col<int>(_set, slot);
    return p;
}

UsdGenRef<const UsdGenElementSet>
UsdGenRootSetFromView(UsdGenRootSetView const &v, uint64_t topologyVersion, uint64_t valueVersion)
{
    UsdGenElementLayout const &layout = UsdGenRootSetLayout();
    if (!v.valid || !layout.channels.size()) return {};
    uint32_t const n = v.totalCurves;
    std::vector<UsdGenChannelDesc> dynamic;
    for (uint32_t k = 0; k < v.PlaneCount(); ++k) {
        UsdGenRootPlaneView const p = v.Plane(k);
        UsdGenChannelDesc d;
        d.name = p.name;
        d.scalar = p.scalar;
        d.arity = p.arity;
        d.level = p.level;
        dynamic.push_back(d);
    }
    UsdGenRef<const UsdGenChannelTable> table = UsdGenChannelTable::Make(
        layout, UsdGenSpan<const UsdGenChannelDesc>(dynamic.data(), dynamic.size()));
    uint32_t const counts[1] = {n};
    UsdGenWriteSet w = UsdGenWriteSet::Fresh(table, UsdGenSpan<const uint32_t>(counts, 1));
    struct Column { UsdGenSlot slot; void const *data; size_t bytes; };
    std::vector<Column> columns;
    auto add = [&](UsdGenSlot slot, void const *data, size_t bytes) {
        if (slot != kUsdGenNoSlot && data && bytes) columns.push_back({slot, data, bytes});
    };
    // Explicit element sizes: the Gf types stay incomplete here.
    add(kUsdGenRootPx, v.px.data(), v.px.size() * 4);
    add(kUsdGenRootPy, v.py.data(), v.py.size() * 4);
    add(kUsdGenRootPz, v.pz.data(), v.pz.size() * 4);
    add(kUsdGenRootId, v.curveId.data(), v.curveId.size() * 8);
    add(kUsdGenRootPrim, v.rootPrim.data(), v.rootPrim.size() * 4);
    add(kUsdGenRootUV, v.rootUV.data(), v.rootUV.size() * 8);
    add(kUsdGenRootT, v.rootT.data(), v.rootT.size() * 12);
    add(kUsdGenRootN, v.rootN.data(), v.rootN.size() * 12);
    add(kUsdGenRootB, v.rootB.data(), v.rootB.size() * 12);
    for (uint32_t k = 0; k < v.PlaneCount(); ++k) {
        UsdGenRootPlaneView const p = v.Plane(k);
        UsdGenSlot const slot = table->Find(p.name);
        if (p.scalar == UsdGenScalar::F32) add(slot, p.f.data(), p.f.size() * 4);
        else if (p.scalar == UsdGenScalar::I32) add(slot, p.i.data(), p.i.size() * 4);
    }
    std::vector<UsdGenSlot> written;
    for (Column const &c : columns) {
        if (w.ValueCount(c.slot) * UsdGenScalarBytes((*table)[c.slot].scalar) != c.bytes) return {};
        written.push_back(c.slot);
    }
    w.Plan(UsdGenSpan<const UsdGenSlot>(written.data(), written.size()));
    for (Column const &c : columns) {
        UsdGenSpan<std::byte> const dst = w.Write<std::byte>(c.slot);
        if (dst.size() == c.bytes) std::memcpy(dst.data(), c.data, c.bytes);
    }
    w.SetVersionsExact(topologyVersion, valueVersion);
    return std::move(w).Seal();
}

}  // namespace usdGen
