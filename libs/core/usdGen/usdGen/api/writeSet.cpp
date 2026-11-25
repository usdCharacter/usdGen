// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/writeSet.h"

#include <atomic>
#include <cstring>

namespace usdGen {

namespace {

std::atomic<uint64_t> gVersion{0};
uint64_t NextVersion() noexcept { return gVersion.fetch_add(1, std::memory_order_relaxed) + 1; }

constexpr uint64_t Align(uint64_t n) noexcept
{
    return (n + UsdGenArena::kColumnAlign - 1) & ~uint64_t(UsdGenArena::kColumnAlign - 1);
}

enum : uint8_t { kAbsent, kShared, kWritten, kCopied };

}  // namespace

struct UsdGenWriteSet::Impl : UsdGenPooled {
    struct Slot {
        void const *column = nullptr;            // kShared / kCopied: the source column
        UsdGenArena const *arena = nullptr;      // kShared: the source arena
        uint64_t version = 0;                    // kShared / kCopied: the source version
        uint64_t at = 0;                         // kWritten / kCopied: byte position in the delta
        uint32_t values = 0;
        uint8_t kind = kAbsent;
    };
    UsdGenRef<const UsdGenChannelTable> channels;
    UsdGenRef<const UsdGenElementSet> base;
    UsdGenRef<UsdGenArena> delta;                // the recycled candidate before Plan
    UsdGenPoolVector<Slot> slots;
    UsdGenPoolVector<UsdGenRef<const UsdGenArena>> held;   // Share() sources
    uint32_t count[kUsdGenMaxLevels + 1] = {1, 0, 0, 0};
    uint32_t const *offsets[kUsdGenMaxLevels + 1] = {};    // shared (Derive)
    UsdGenArena const *offsetArena[kUsdGenMaxLevels + 1] = {};
    uint64_t offsetAt[kUsdGenMaxLevels + 1] = {};
    uint8_t offsetKind[kUsdGenMaxLevels + 1] = {};         // kShared, kWritten (Fresh, planned), kCopied
    bool offsetWritten[kUsdGenMaxLevels + 1] = {};
    uint64_t topologyOverride = 0, valueOverride = 0;
    bool fresh = false, planned = false;

    uint32_t ValuesFor(UsdGenChannelDesc const &d) const noexcept
    {
        return d.level <= kUsdGenMaxLevels ? count[d.level] * uint32_t(d.arity) : 0u;
    }
    std::byte *DeltaAt(uint64_t at) const noexcept { return delta->Payload() + at; }
};

namespace {

UsdGenArena const *ArenaHolding(UsdGenElementSet const &set, void const *p) noexcept
{
    auto const *b = static_cast<std::byte const *>(p);
    for (uint8_t i = 0; i < set.ArenaCount(); ++i) {
        UsdGenArena const &a = set.ArenaAt(i);
        if (b >= a.Payload() && b < a.Payload() + a.Capacity()) return &a;
    }
    return nullptr;
}

}  // namespace

UsdGenWriteSet UsdGenWriteSet::Derive(UsdGenRef<const UsdGenElementSet> base,
                                      UsdGenRef<const UsdGenChannelTable> channels)
{
    if (!base) {
        if (!channels) return UsdGenWriteSet(nullptr);
        return Fresh(std::move(channels), {});
    }
    UsdGenElementSet const &b = *base;
    if (!channels) channels = b._channels;
    Impl *impl = new Impl;
    UsdGenChannelTable const &table = *channels;
    UsdGenChannelTable const &from = *b._channels;
    for (uint8_t l = 0; l <= kUsdGenMaxLevels; ++l) {
        impl->count[l] = b._count[l];
        if (b._offsets[l]) {
            impl->offsets[l] = b._offsets[l];
            impl->offsetArena[l] = ArenaHolding(b, b._offsets[l]);
            impl->offsetKind[l] = kShared;
        }
    }
    impl->slots.resize(table.Size());
    bool const same = &table == &from;
    bool const child = !same && table._parent.Get() == &from;
    for (UsdGenSlot s = 0; s < table.Size(); ++s) {
        Impl::Slot &slot = impl->slots[s];
        UsdGenChannelDesc const &d = table[s];
        slot.values = impl->ValuesFor(d);
        UsdGenSlot const bs = same ? s : child ? table._fromParent[s] : from.Find(d.name);
        if (bs == kUsdGenNoSlot || !b.Has(bs) || !UsdGenSameLayout(d, from[bs])) continue;
        slot.kind = kShared;
        slot.column = b._columns[bs];
        slot.arena = b._arenas[b._columnArena[bs]].Get();
        slot.version = b._versions[bs];
        slot.values = b._values[bs];
    }
    impl->channels = std::move(channels);
    impl->base = std::move(base);
    return UsdGenWriteSet(impl);
}

UsdGenWriteSet UsdGenWriteSet::Fresh(UsdGenRef<const UsdGenChannelTable> channels,
                                     UsdGenSpan<const uint32_t> counts)
{
    if (!channels) return UsdGenWriteSet(nullptr);
    Impl *impl = new Impl;
    impl->fresh = true;
    for (size_t l = 0; l < counts.size() && l < kUsdGenMaxLevels; ++l) impl->count[l + 1] = counts[l];
    impl->slots.resize(channels->Size());
    for (UsdGenSlot s = 0; s < channels->Size(); ++s)
        impl->slots[s].values = impl->ValuesFor((*channels)[s]);
    impl->channels = std::move(channels);
    return UsdGenWriteSet(impl);
}

void UsdGenWriteSet::Recycle(UsdGenRef<const UsdGenElementSet> previous) noexcept
{
    if (!_impl || _impl->planned || !previous || previous->RefCount() != 1) return;
    UsdGenElementSet &p = const_cast<UsdGenElementSet &>(*previous);   // the last reference: dies below
    if (p._delta >= p._arenaCount) return;
    UsdGenArena const &arena = *p._arenas[p._delta];
    if (arena.IsShared() || arena.RefCount() != 1) return;
    _impl->delta = UsdGenRef<UsdGenArena>::Adopt(const_cast<UsdGenArena *>(p._arenas[p._delta].Detach()));
}

void UsdGenWriteSet::Plan(UsdGenSpan<const UsdGenSlot> written)
{
    if (!_impl || _impl->planned) return;
    Impl &im = *_impl;
    im.planned = true;
    UsdGenChannelTable const &table = *im.channels;
    for (UsdGenSlot s : written) {
        if (s >= im.slots.size()) continue;
        Impl::Slot &slot = im.slots[s];
        slot.kind = kWritten;
        slot.values = im.ValuesFor(table[s]);
        slot.column = nullptr;
        slot.arena = nullptr;
    }
    uint8_t const levels = table.Layout().levels > kUsdGenMaxLevels ? kUsdGenMaxLevels : table.Layout().levels;
    if (im.fresh)
        for (uint8_t l = 2; l <= levels; ++l) im.offsetKind[l] = kWritten;

    // Arenas the result would hold; past the limit every shared column moves into the delta.
    UsdGenArena const *seen[kUsdGenMaxArenasPerSet + 1] = {};
    uint32_t distinct = 0;
    bool needDelta = false;
    auto note = [&](UsdGenArena const *a) {
        for (uint32_t i = 0; i < distinct; ++i)
            if (seen[i] == a) return;
        if (distinct <= kUsdGenMaxArenasPerSet) seen[distinct] = a;
        ++distinct;
    };
    for (uint8_t l = 1; l <= kUsdGenMaxLevels; ++l) {
        if (im.offsetKind[l] == kShared) note(im.offsetArena[l]);
        else if (im.offsetKind[l] == kWritten) needDelta = true;
    }
    for (Impl::Slot const &slot : im.slots) {
        if (slot.kind == kShared) note(slot.arena);
        else if (slot.kind == kWritten) needDelta = true;
    }
    if (distinct + (needDelta ? 1u : 0u) > kUsdGenMaxArenasPerSet) {
        needDelta = true;
        for (uint8_t l = 1; l <= kUsdGenMaxLevels; ++l)
            if (im.offsetKind[l] == kShared) im.offsetKind[l] = kCopied;
        for (Impl::Slot &slot : im.slots)
            if (slot.kind == kShared) slot.kind = kCopied;
    }

    uint64_t cursor = 0;
    for (uint8_t l = 1; l <= kUsdGenMaxLevels; ++l) {
        if (im.offsetKind[l] != kWritten && im.offsetKind[l] != kCopied) continue;
        im.offsetAt[l] = cursor;
        cursor += Align((uint64_t(im.count[l - 1]) + 1) * sizeof(uint32_t));
    }
    for (UsdGenSlot s = 0; s < im.slots.size(); ++s) {
        Impl::Slot &slot = im.slots[s];
        if (slot.kind != kWritten && slot.kind != kCopied) continue;
        slot.at = cursor;
        cursor += Align(uint64_t(slot.values) * UsdGenScalarBytes(table[s].scalar));
    }
    if (!needDelta) {
        im.delta.Reset();
        return;
    }
    if (!im.delta || im.delta->Capacity() < cursor) {
        im.delta.Reset();   // back to the pool before the next block is taken
        im.delta = UsdGenArena::Acquire(size_t(cursor));
    }
    for (uint8_t l = 1; l <= kUsdGenMaxLevels; ++l)
        if (im.offsetKind[l] == kCopied)
            std::memcpy(im.DeltaAt(im.offsetAt[l]), im.offsets[l],
                        (size_t(im.count[l - 1]) + 1) * sizeof(uint32_t));
    for (UsdGenSlot s = 0; s < im.slots.size(); ++s) {
        Impl::Slot const &slot = im.slots[s];
        if (slot.kind == kCopied)
            std::memcpy(im.DeltaAt(slot.at), slot.column,
                        size_t(slot.values) * UsdGenScalarBytes(table[s].scalar));
    }
}

UsdGenSpan<uint32_t> UsdGenWriteSet::WriteOffsets(uint8_t level)
{
    if (!_impl || !_impl->planned || !_impl->fresh || level < 2 || level > kUsdGenMaxLevels ||
        _impl->offsetKind[level] != kWritten)
        return {};
    _impl->offsetWritten[level] = true;
    return {reinterpret_cast<uint32_t *>(_impl->DeltaAt(_impl->offsetAt[level])),
            size_t(_impl->count[level - 1]) + 1};
}

void *UsdGenWriteSet::WriteRaw(UsdGenSlot slot, uint8_t sample)
{
    if (!_impl || !_impl->planned || sample != 0 || slot >= _impl->slots.size() ||
        _impl->slots[slot].kind != kWritten)
        return nullptr;
    return _impl->DeltaAt(_impl->slots[slot].at);
}

size_t UsdGenWriteSet::ValueCount(UsdGenSlot slot) const noexcept
{
    return _impl && slot < _impl->slots.size() ? _impl->slots[slot].values : 0u;
}

size_t UsdGenWriteSet::ValueBytes(UsdGenSlot slot) const noexcept
{
    if (!_impl || slot >= _impl->slots.size()) return 0;
    return size_t(_impl->slots[slot].values) * UsdGenScalarBytes((*_impl->channels)[slot].scalar);
}

void UsdGenWriteSet::Share(UsdGenSlot slot, UsdGenElementSet const &from, UsdGenSlot fromSlot)
{
    if (!_impl || _impl->planned || slot >= _impl->slots.size()) return;
    Impl::Slot &s = _impl->slots[slot];
    UsdGenChannelDesc const &d = (*_impl->channels)[slot];
    if (!from.Has(fromSlot) || !UsdGenSameLayout(d, from.Channels()[fromSlot]) ||
        from._values[fromSlot] != _impl->ValuesFor(d)) {
        s.kind = kAbsent;
        return;
    }
    UsdGenArena const *arena = from._arenas[from._columnArena[fromSlot]].Get();
    _impl->held.push_back(UsdGenRef<const UsdGenArena>(arena));
    s.kind = kShared;
    s.column = from._columns[fromSlot];
    s.arena = arena;
    s.version = from._versions[fromSlot];
    s.values = from._values[fromSlot];
}

void UsdGenWriteSet::Drop(UsdGenSlot slot)
{
    if (_impl && slot < _impl->slots.size()) _impl->slots[slot].kind = kAbsent;
}

void UsdGenWriteSet::SetVersions(uint64_t topologyVersion, uint64_t valueVersion) noexcept
{
    if (!_impl) return;
    _impl->topologyOverride = topologyVersion;
    _impl->valueOverride = valueVersion;
}

UsdGenRef<const UsdGenElementSet> UsdGenWriteSet::Seal() &&
{
    if (!_impl) return {};
    if (!_impl->planned) Plan({});
    Impl &im = *_impl;
    UsdGenChannelTable const &table = *im.channels;
    UsdGenRef<UsdGenElementSet> set(new UsdGenElementSet);
    UsdGenElementSet &out = *set;
    out._channels = im.channels;
    for (uint8_t l = 0; l <= kUsdGenMaxLevels; ++l) out._count[l] = im.count[l];
    size_t const n = table.Size();
    out._columns.assign(n, nullptr);
    out._values.resize(n);
    out._columnArena.assign(n, uint8_t(0xFF));
    out._versions.assign(n, 0);

    // Shared arenas first (offsets, then columns, in slot order), then the delta.
    auto arenaIndex = [&out](UsdGenArena const *a) -> uint8_t {
        for (uint8_t i = 0; i < out._arenaCount; ++i)
            if (out._arenas[i].Get() == a) return i;
        if (out._arenaCount >= kUsdGenMaxArenasPerSet) return 0xFF;
        out._arenas[out._arenaCount] = UsdGenRef<const UsdGenArena>(a);
        return out._arenaCount++;
    };
    for (uint8_t l = 1; l <= kUsdGenMaxLevels; ++l)
        if (im.offsetKind[l] == kShared && arenaIndex(im.offsetArena[l]) != 0xFF)
            out._offsets[l] = im.offsets[l];
    bool wrote = false, usesDelta = false;
    for (UsdGenSlot s = 0; s < n; ++s) {
        Impl::Slot const &slot = im.slots[s];
        out._values[s] = slot.values;
        if (slot.kind == kShared) {
            uint8_t const i = arenaIndex(slot.arena);
            if (i == 0xFF) continue;   // Share after Plan past the arena limit
            out._columns[s] = slot.column;
            out._columnArena[s] = i;
            out._versions[s] = slot.version;
        } else if (slot.kind == kWritten || slot.kind == kCopied) {
            usesDelta = true;
            wrote = wrote || slot.kind == kWritten;
        }
    }
    for (uint8_t l = 1; l <= kUsdGenMaxLevels; ++l)
        usesDelta = usesDelta || im.offsetKind[l] == kCopied ||
                    (im.offsetKind[l] == kWritten && im.offsetWritten[l]);

    uint64_t const topology = im.topologyOverride ? im.topologyOverride
        : im.fresh || !im.base ? NextVersion() : im.base->_topologyVersion;
    uint64_t const value = im.valueOverride ? im.valueOverride
        : im.fresh || !im.base || wrote ? NextVersion() : im.base->_valueVersion;
    out._topologyVersion = topology;
    out._valueVersion = value;

    if (usesDelta && im.delta && out._arenaCount < kUsdGenMaxArenasPerSet) {
        uint8_t const d = out._arenaCount++;
        out._delta = d;
        for (uint8_t l = 1; l <= kUsdGenMaxLevels; ++l)
            if (im.offsetKind[l] == kCopied || (im.offsetKind[l] == kWritten && im.offsetWritten[l]))
                out._offsets[l] = reinterpret_cast<uint32_t const *>(im.DeltaAt(im.offsetAt[l]));
        for (UsdGenSlot s = 0; s < n; ++s) {
            Impl::Slot const &slot = im.slots[s];
            if (slot.kind != kWritten && slot.kind != kCopied) continue;
            out._columns[s] = im.DeltaAt(slot.at);
            out._columnArena[s] = d;
            out._versions[s] = slot.kind == kWritten ? value : slot.version;
        }
        out._arenas[d] = UsdGenRef<const UsdGenArena>(std::move(im.delta));
    }
    delete _impl;
    _impl = nullptr;
    return set;
}

UsdGenWriteSet::~UsdGenWriteSet()
{
    delete _impl;
}

}  // namespace usdGen
