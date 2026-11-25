// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/elementSet.h"
#include "usdGen/api/writeSet.h"

#include <algorithm>

namespace usdGen {

namespace {

uint8_t RunOf(UsdGenChannelDesc const &d) noexcept
{
    return d.level < 1 ? 1 : d.level > kUsdGenMaxLevels ? kUsdGenMaxLevels : d.level;
}

uint64_t Fnv(uint64_t h, uint64_t v) noexcept
{
    for (int i = 0; i < 8; ++i, v >>= 8) {
        h ^= v & 0xFFu;
        h *= 0x100000001b3ull;
    }
    return h;
}

/// True when added[i] is the last entry of its name (the one that wins).
bool IsLastOfName(UsdGenSpan<const UsdGenChannelDesc> added, size_t i) noexcept
{
    for (size_t j = i + 1; j < added.size(); ++j)
        if (added[j].name == added[i].name) return false;
    return true;
}

}  // namespace

UsdGenRef<const UsdGenChannelTable>
UsdGenChannelTable::_Build(UsdGenElementLayout const &layout, UsdGenSpan<const UsdGenChannelDesc> base,
                           uint16_t staticCount, UsdGenSpan<const UsdGenChannelDesc> added,
                           UsdGenChannelTable const *parent)
{
    UsdGenRef<UsdGenChannelTable> t(new UsdGenChannelTable);
    t->_layout = &layout;
    t->_staticCount = staticCount;
    UsdGenPoolVector<UsdGenChannelDesc> dyn(base.begin() + staticCount, base.end());
    t->_descs.assign(base.begin(), base.begin() + staticCount);
    for (UsdGenChannelDesc const &d : added) {
        auto same = [&d](UsdGenChannelDesc const &e) { return e.name == d.name; };
        auto s = std::find_if(t->_descs.begin(), t->_descs.end(), same);
        if (s != t->_descs.end()) { *s = d; continue; }
        auto e = std::find_if(dyn.begin(), dyn.end(), same);
        if (e != dyn.end()) *e = d;
        else dyn.push_back(d);
    }
    std::sort(dyn.begin(), dyn.end(), [](UsdGenChannelDesc const &a, UsdGenChannelDesc const &b) {
        uint8_t const ra = RunOf(a), rb = RunOf(b);
        return ra != rb ? ra < rb : a.name < b.name;
    });
    t->_descs.insert(t->_descs.end(), dyn.begin(), dyn.end());
    t->_dynamic.resize(dyn.size());
    size_t i = 0;
    for (uint8_t r = 0; r <= kUsdGenMaxLevels + 1; ++r) {
        while (i < dyn.size() && RunOf(dyn[i]) < r) ++i;
        t->_runBegin[r] = uint16_t(i);
    }
    for (size_t k = 0; k < dyn.size(); ++k) t->_dynamic[k] = UsdGenSlot(staticCount + k);

    uint64_t h = Fnv(0xcbf29ce484222325ull, uint64_t(reinterpret_cast<uintptr_t>(&layout)));
    h = Fnv(h, staticCount);
    for (UsdGenChannelDesc const &d : t->_descs) {
        h = Fnv(h, d.name.Hash());
        h = Fnv(h, uint64_t(d.scalar) | uint64_t(d.arity) << 8 | uint64_t(d.level) << 16 |
                       uint64_t(d.role) << 24 | uint64_t(d.component) << 32 | uint64_t(d.flags) << 40);
    }
    t->_digest = h;

    if (parent) {
        t->_parent = UsdGenRef<const UsdGenChannelTable>(parent);
        t->_fromParent.resize(t->_descs.size());
        for (size_t s = 0; s < t->_descs.size(); ++s)
            t->_fromParent[s] = s < staticCount ? UsdGenSlot(s) : parent->Find(t->_descs[s].name);
    }
    return t;
}

UsdGenRef<const UsdGenChannelTable>
UsdGenChannelTable::Make(UsdGenElementLayout const &layout, UsdGenSpan<const UsdGenChannelDesc> dynamic)
{
    return _Build(layout, layout.channels, uint16_t(layout.channels.size()), dynamic, nullptr);
}

UsdGenRef<const UsdGenChannelTable>
UsdGenChannelTable::With(UsdGenSpan<const UsdGenChannelDesc> added) const
{
    bool changes = false;
    for (size_t i = 0; i < added.size() && !changes; ++i) {
        if (!IsLastOfName(added, i)) continue;
        UsdGenSlot const s = Find(added[i].name);
        changes = s == kUsdGenNoSlot || _descs[s] != added[i];
    }
    if (!changes) return UsdGenRef<const UsdGenChannelTable>(this);
    return _Build(*_layout, UsdGenSpan<const UsdGenChannelDesc>(_descs.data(), _descs.size()),
                  _staticCount, added, this);
}

UsdGenSlot UsdGenChannelTable::Find(UsdGenToken const &name) const noexcept
{
    for (uint16_t s = 0; s < _staticCount; ++s)
        if (_descs[s].name == name) return s;
    for (uint8_t r = 1; r <= kUsdGenMaxLevels; ++r) {
        UsdGenSlot const *lo = _dynamic.data() + _runBegin[r];
        UsdGenSlot const *hi = _dynamic.data() + _runBegin[r + 1];
        UsdGenSlot const *it = std::lower_bound(lo, hi, name, [this](UsdGenSlot s, UsdGenToken const &n) {
            return _descs[s].name < n;
        });
        if (it != hi && _descs[*it].name == name) return *it;
    }
    return kUsdGenNoSlot;
}

UsdGenElementSetView UsdGenElementSet::View() const noexcept
{
    UsdGenElementSetView v;
    v.channels = _channels.Get();
    for (uint8_t l = 0; l <= kUsdGenMaxLevels; ++l) {
        v.count[l] = _count[l];
        v.offsets[l] = _offsets[l];
    }
    v.columns = _columns.data();
    v.values = _values.data();
    v.topologyVersion = _topologyVersion;
    v.valueVersion = _valueVersion;
    return v;
}

uint64_t UsdGenElementSet::ColumnOffset(UsdGenSlot s) const noexcept
{
    if (!Has(s)) return 0;
    return uint64_t(static_cast<std::byte const *>(_columns[s]) - _arenas[_columnArena[s]]->Payload());
}

void UsdGenElementSet::Export() const noexcept
{
    for (uint8_t i = 0; i < _arenaCount; ++i) _arenas[i]->MarkShared();
}

UsdGenRef<const UsdGenElementSet> UsdGenElementSet::Empty(UsdGenElementLayout const &layout)
{
    uint32_t const zeros[kUsdGenMaxLevels] = {};
    uint8_t const levels = layout.levels > kUsdGenMaxLevels ? kUsdGenMaxLevels : layout.levels;
    UsdGenWriteSet w = UsdGenWriteSet::Fresh(UsdGenChannelTable::Make(layout, {}),
                                             UsdGenSpan<const uint32_t>(zeros, levels));
    w.Plan({});
    return std::move(w).Seal();
}

}  // namespace usdGen
